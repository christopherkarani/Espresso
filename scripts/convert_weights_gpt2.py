#!/usr/bin/env python3
"""Convert HuggingFace GPT-2 weights to Espresso BLOBFILE layout.

Requires:
    pip install torch transformers

Usage:
    ./scripts/convert_weights_gpt2.py --model gpt2 --output /tmp/gpt2_124m
"""

from __future__ import annotations

import argparse
import json
import os
import struct
from pathlib import Path

import torch
from transformers import GPT2LMHeadModel


def make_blob_header(data_size: int) -> bytes:
    header = bytearray(128)
    header[0] = 0x01
    header[4] = 0x02
    header[64:68] = bytes([0xEF, 0xBE, 0xAD, 0xDE])
    header[68] = 0x01
    struct.pack_into("<I", header, 72, data_size)
    struct.pack_into("<I", header, 80, 128)
    return bytes(header)


def write_blob(tensor: torch.Tensor, path: Path, transpose: bool = False) -> None:
    array = tensor.detach().cpu().float()
    if transpose:
        array = array.transpose(0, 1).contiguous()
    payload = array.to(torch.float16).numpy().tobytes()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(make_blob_header(len(payload)))
        handle.write(payload)


def write_causal_masks(output_dir: Path, max_seq: int) -> None:
    mask_dir = output_dir / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    size = 1
    while size <= max_seq:
        mask = torch.zeros((size, size), dtype=torch.float16)
        mask = torch.triu(mask.fill_(-1e4), diagonal=1) + torch.tril(torch.zeros_like(mask))
        payload = mask.numpy().tobytes()
        with (mask_dir / f"causal_{size}.bin").open("wb") as handle:
            handle.write(make_blob_header(len(payload)))
            handle.write(payload)
        size *= 2


def save_gpt2_artifacts(
    model: GPT2LMHeadModel,
    output_dir: Path,
    metadata_name: str = "gpt2_124m",
) -> None:
    state = model.state_dict()
    config = model.config

    metadata = {
        "name": metadata_name,
        "nLayer": config.n_layer,
        "nHead": config.n_head,
        "nKVHead": config.n_head,
        "dModel": config.n_embd,
        "headDim": config.n_embd // config.n_head,
        "hiddenDim": config.n_inner or (4 * config.n_embd),
        "vocab": config.vocab_size,
        "maxSeq": config.n_positions,
        "normEps": config.layer_norm_epsilon,
        "architecture": "gpt2",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    write_blob(state["transformer.wte.weight"], output_dir / "embeddings" / "token.bin")
    write_blob(state["transformer.wpe.weight"], output_dir / "embeddings" / "position.bin")
    write_blob(state["transformer.ln_f.weight"], output_dir / "final_norm_gamma.bin")
    write_blob(state["transformer.ln_f.bias"], output_dir / "final_norm_beta.bin")
    write_blob(state["lm_head.weight"], output_dir / "lm_head.bin")

    for layer in range(config.n_layer):
        layer_dir = output_dir / "layers" / str(layer)
        prefix = f"transformer.h.{layer}"

        write_blob(state[f"{prefix}.ln_1.weight"], layer_dir / "ln_1_gamma.bin")
        write_blob(state[f"{prefix}.ln_1.bias"], layer_dir / "ln_1_beta.bin")

        qkv_weight = state[f"{prefix}.attn.c_attn.weight"]
        qkv_bias = state[f"{prefix}.attn.c_attn.bias"]
        wq, wk, wv = qkv_weight.split(config.n_embd, dim=1)
        bq, bk, bv = qkv_bias.split(config.n_embd, dim=0)

        write_blob(wq, layer_dir / "wq.bin", transpose=True)
        write_blob(wk, layer_dir / "wk.bin", transpose=True)
        write_blob(wv, layer_dir / "wv.bin", transpose=True)
        write_blob(bq, layer_dir / "bq.bin")
        write_blob(bk, layer_dir / "bk.bin")
        write_blob(bv, layer_dir / "bv.bin")

        write_blob(state[f"{prefix}.attn.c_proj.weight"], layer_dir / "wo.bin", transpose=True)
        write_blob(state[f"{prefix}.attn.c_proj.bias"], layer_dir / "bo.bin")

        write_blob(state[f"{prefix}.ln_2.weight"], layer_dir / "ln_2_gamma.bin")
        write_blob(state[f"{prefix}.ln_2.bias"], layer_dir / "ln_2_beta.bin")
        write_blob(state[f"{prefix}.mlp.c_fc.weight"], layer_dir / "w1.bin", transpose=True)
        write_blob(state[f"{prefix}.mlp.c_fc.bias"], layer_dir / "b1.bin")
        write_blob(state[f"{prefix}.mlp.c_proj.weight"], layer_dir / "w2.bin", transpose=True)
        write_blob(state[f"{prefix}.mlp.c_proj.bias"], layer_dir / "b2.bin")

    write_causal_masks(output_dir, config.n_positions)


def convert_pretrained_gpt2(
    model_name: str,
    output_dir: Path,
    cache_dir: str | None = None,
    metadata_name: str = "gpt2_124m",
) -> None:
    model = GPT2LMHeadModel.from_pretrained(model_name, cache_dir=cache_dir)
    save_gpt2_artifacts(model, output_dir, metadata_name=metadata_name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gpt2", help="HuggingFace GPT-2 model name or local path")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--cache-dir", default=None, help="Optional Hugging Face cache directory")
    parser.add_argument("--metadata-name", default="gpt2_124m", help="Name written into metadata.json")
    args = parser.parse_args()

    output_dir = Path(args.output)
    convert_pretrained_gpt2(
        args.model,
        output_dir,
        cache_dir=args.cache_dir,
        metadata_name=args.metadata_name,
    )


if __name__ == "__main__":
    main()
