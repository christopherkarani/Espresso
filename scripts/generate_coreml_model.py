#!/usr/bin/env python3
"""Generate a Core ML model matching the Espresso transformer architecture."""

import argparse
import os
import pathlib

import coremltools as ct
import numpy as np
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types

DIM = 768
HIDDEN = 2048
SEQ_LEN = 256
HEADS = 12
HEAD_DIM = DIM // HEADS
WEIGHT_SCALE = 0.02
RNG = np.random.default_rng(1234)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a Core ML transformer model for Espresso benchmarks."
    )
    parser.add_argument(
        "--layers",
        type=int,
        default=1,
        help="Number of transformer layers to stack",
    )
    parser.add_argument(
        "--weight-mode",
        choices=["random", "zero"],
        default="random",
        help="Use random fp16 weights or an exact zero-weight trunk",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output .mlpackage path (defaults to benchmarks/models/transformer_<layers>layer.mlpackage)",
    )
    return parser.parse_args()


def random_weight(shape):
    return RNG.standard_normal(shape).astype(np.float16) * WEIGHT_SCALE


def make_weight(shape, mode: str):
    if mode == "zero":
        return np.zeros(shape, dtype=np.float16)
    return random_weight(shape)


def rms_norm(x, prefix: str):
    weight = mb.const(
        val=np.ones((1, DIM, 1, 1), dtype=np.float16),
        name=f"{prefix}_weight",
    )
    x_sq = mb.mul(x=x, y=x, name=f"{prefix}_sq")
    x_mean = mb.reduce_mean(x=x_sq, axes=[1], keep_dims=True, name=f"{prefix}_mean")
    eps = mb.const(val=np.float16(1e-5), name=f"{prefix}_eps")
    x_mean_eps = mb.add(x=x_mean, y=eps, name=f"{prefix}_mean_eps")
    x_rsqrt = mb.rsqrt(x=x_mean_eps, name=f"{prefix}_rsqrt")
    x_norm_pre = mb.mul(x=x, y=x_rsqrt, name=f"{prefix}_norm_pre")
    return mb.mul(x=x_norm_pre, y=weight, name=f"{prefix}_norm")


def transformer_block(x, layer_index: int, weight_mode: str):
    prefix = f"l{layer_index}"

    x_norm = rms_norm(x, f"{prefix}_rms_att")

    wq = mb.const(val=make_weight((DIM, DIM, 1, 1), weight_mode), name=f"{prefix}_wq")
    wk = mb.const(val=make_weight((DIM, DIM, 1, 1), weight_mode), name=f"{prefix}_wk")
    wv = mb.const(val=make_weight((DIM, DIM, 1, 1), weight_mode), name=f"{prefix}_wv")

    q = mb.conv(x=x_norm, weight=wq, name=f"{prefix}_q_proj")
    k = mb.conv(x=x_norm, weight=wk, name=f"{prefix}_k_proj")
    v = mb.conv(x=x_norm, weight=wv, name=f"{prefix}_v_proj")

    q_r = mb.reshape(x=q, shape=[1, HEADS, HEAD_DIM, SEQ_LEN], name=f"{prefix}_q_reshape")
    k_r = mb.reshape(x=k, shape=[1, HEADS, HEAD_DIM, SEQ_LEN], name=f"{prefix}_k_reshape")
    v_r = mb.reshape(x=v, shape=[1, HEADS, HEAD_DIM, SEQ_LEN], name=f"{prefix}_v_reshape")

    q_t = mb.transpose(x=q_r, perm=[0, 1, 3, 2], name=f"{prefix}_q_transpose")
    v_t = mb.transpose(x=v_r, perm=[0, 1, 3, 2], name=f"{prefix}_v_transpose")

    scale = mb.const(val=np.float16(1.0 / np.sqrt(HEAD_DIM)), name=f"{prefix}_scale")
    scores = mb.matmul(x=q_t, y=k_r, name=f"{prefix}_attn_scores_raw")
    scores = mb.mul(x=scores, y=scale, name=f"{prefix}_attn_scores")

    mask_np = np.triu(np.full((SEQ_LEN, SEQ_LEN), -1e4, dtype=np.float16), k=1)
    mask = mb.const(val=mask_np.reshape(1, 1, SEQ_LEN, SEQ_LEN), name=f"{prefix}_causal_mask")
    scores = mb.add(x=scores, y=mask, name=f"{prefix}_attn_scores_masked")

    attn_weights = mb.softmax(x=scores, axis=-1, name=f"{prefix}_attn_weights")
    attn_out = mb.matmul(x=attn_weights, y=v_t, name=f"{prefix}_attn_out_heads")
    attn_out = mb.transpose(x=attn_out, perm=[0, 1, 3, 2], name=f"{prefix}_attn_out_transpose")
    attn_out = mb.reshape(x=attn_out, shape=[1, DIM, 1, SEQ_LEN], name=f"{prefix}_attn_out_concat")

    wo = mb.const(val=make_weight((DIM, DIM, 1, 1), weight_mode), name=f"{prefix}_wo")
    o_out = mb.conv(x=attn_out, weight=wo, name=f"{prefix}_o_proj")
    x2 = mb.add(x=x, y=o_out, name=f"{prefix}_residual_attn")

    x2_norm = rms_norm(x2, f"{prefix}_rms_ffn")

    w1 = mb.const(val=make_weight((HIDDEN, DIM, 1, 1), weight_mode), name=f"{prefix}_w1")
    w3 = mb.const(val=make_weight((HIDDEN, DIM, 1, 1), weight_mode), name=f"{prefix}_w3")
    w2 = mb.const(val=make_weight((DIM, HIDDEN, 1, 1), weight_mode), name=f"{prefix}_w2")

    h1 = mb.conv(x=x2_norm, weight=w1, name=f"{prefix}_ffn_w1")
    h3 = mb.conv(x=x2_norm, weight=w3, name=f"{prefix}_ffn_w3")
    silu = mb.sigmoid(x=h1, name=f"{prefix}_sigmoid_h1")
    silu = mb.mul(x=h1, y=silu, name=f"{prefix}_silu_h1")
    gate = mb.mul(x=silu, y=h3, name=f"{prefix}_gate_out")
    ffn_out = mb.conv(x=gate, weight=w2, name=f"{prefix}_ffn_w2")

    return mb.add(x=x2, y=ffn_out, name=f"{prefix}_residual_ffn")


def build_transformer_stack(layer_count: int, weight_mode: str):
    @mb.program(
        input_specs=[mb.TensorSpec(shape=(1, DIM, 1, SEQ_LEN), dtype=types.fp16)]
    )
    def transformer(x):
        current = x
        for layer_index in range(layer_count):
            current = transformer_block(current, layer_index, weight_mode)
        return current

    return transformer


def default_output_path(layer_count: int, script_dir: str) -> str:
    output_dir = os.path.join(script_dir, "..", "benchmarks", "models")
    os.makedirs(output_dir, exist_ok=True)
    if layer_count == 1:
        filename = "transformer_layer.mlpackage"
    else:
        filename = f"transformer_{layer_count}layer.mlpackage"
    return os.path.join(output_dir, filename)


def main() -> None:
    args = parse_args()
    if args.layers <= 0:
        raise SystemExit("--layers must be > 0")

    print("Generating Core ML transformer model...")
    print(
        f"  layers={args.layers}, dim={DIM}, hidden={HIDDEN}, "
        f"seq_len={SEQ_LEN}, heads={HEADS}, weight_mode={args.weight_mode}"
    )

    prog = build_transformer_stack(args.layers, args.weight_mode)

    model = ct.convert(
        prog,
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.macOS15,
    )

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = args.output or default_output_path(args.layers, script_dir)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    model.save(output_path)

    total_size = sum(
        path.stat().st_size for path in pathlib.Path(output_path).rglob("*") if path.is_file()
    )
    print(f"  Saved to: {output_path}")
    print(f"  Model size: {total_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
