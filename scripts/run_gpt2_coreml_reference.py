#!/usr/bin/env python3
"""Run GPT-2 Core ML compare baselines for espresso-generate.

Non-streaming mode prints one JSON object matching CoreMLComparisonResult.
With `--emit-events`, prints NDJSON events: compile / token / completed.

Requires: numpy, coremltools (macOS). Final LayerNorm + LM head are applied from
Espresso weight blobs so the Core ML package only needs to emit trunk hidden states.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np


BLOBFILE_HEADER_BYTES = 128


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coreml-model", required=True, help="Path to gpt2_seqN.mlpackage")
    parser.add_argument("--weights", required=True, help="Espresso GPT-2 weights directory")
    parser.add_argument(
        "--prompt-tokens",
        required=True,
        help="Comma-separated prompt token ids",
    )
    parser.add_argument("--seq-len", required=True, type=int)
    parser.add_argument("--max-tokens", required=True, type=int)
    parser.add_argument("--temperature", required=True, type=float)
    parser.add_argument("--warmup", required=True, type=int)
    parser.add_argument("--iterations", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--compute-units", required=True)
    parser.add_argument(
        "--emit-events",
        action="store_true",
        help="Emit NDJSON stream events instead of a single JSON object",
    )
    args = parser.parse_args(argv)
    args.prompt_tokens = parse_prompt_tokens(args.prompt_tokens)
    return args


def parse_prompt_tokens(raw: str) -> list[int]:
    tokens: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        tokens.append(int(part))
    if not tokens:
        raise ValueError("prompt-tokens must contain at least one token id")
    return tokens


def build_comparison_result(
    *,
    generated_tokens: list[int],
    compile_time_ms: float,
    first_token_latency_ms: float,
    tokens_per_second: float,
    median_token_ms: float,
    p95_token_ms: float,
    token_latencies_ms: list[float],
    total_time_ms: float,
    compute_units: str,
    seq_len: int,
) -> dict[str, Any]:
    return {
        "generated_tokens": generated_tokens,
        "compile_time_ms": compile_time_ms,
        "first_token_latency_ms": first_token_latency_ms,
        "tokens_per_second": tokens_per_second,
        "median_token_ms": median_token_ms,
        "p95_token_ms": p95_token_ms,
        "token_latencies_ms": token_latencies_ms,
        "total_time_ms": total_time_ms,
        "compute_units": compute_units,
        "seq_len": seq_len,
    }


def _load_blobfile(path: Path) -> np.ndarray:
    import numpy as np

    data = path.read_bytes()
    if len(data) < BLOBFILE_HEADER_BYTES:
        raise ValueError(f"BLOBFILE too small: {path}")
    payload_size = struct.unpack_from("<I", data, 72)[0]
    payload = data[BLOBFILE_HEADER_BYTES : BLOBFILE_HEADER_BYTES + payload_size]
    if len(payload) != payload_size:
        raise ValueError(f"BLOBFILE truncated: {path}")
    return np.frombuffer(payload, dtype=np.float16).astype(np.float32)


def load_top_level_weights(weights_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gamma_candidates = ["final_norm_gamma.bin", "ln_f_gamma.bin", "final_norm.bin"]
    beta_candidates = ["final_norm_beta.bin", "ln_f_beta.bin"]
    head_candidates = ["lm_head.bin", "classifier.bin"]

    def first_existing(names: list[str]) -> Path:
        for name in names:
            candidate = weights_dir / name
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(f"Missing one of {names} under {weights_dir}")

    gamma = _load_blobfile(first_existing(gamma_candidates))
    beta = _load_blobfile(first_existing(beta_candidates))
    lm_head = _load_blobfile(first_existing(head_candidates))
    if lm_head.size % gamma.size != 0:
        raise ValueError("lm_head size is not divisible by hidden size")
    return gamma, beta, lm_head.reshape(-1, gamma.size)


def layer_norm(hidden: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    mean = hidden.mean()
    var = ((hidden - mean) ** 2).mean()
    normalized = (hidden - mean) / math.sqrt(float(var) + eps)
    return normalized * gamma + beta


def sample_token(logits: np.ndarray, temperature: float, rng: np.random.Generator) -> int:
    import numpy as np

    if temperature <= 0:
        return int(np.argmax(logits))
    scaled = logits / temperature
    scaled = scaled - scaled.max()
    probs = np.exp(scaled)
    probs = probs / probs.sum()
    return int(rng.choice(len(probs), p=probs))


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    index = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return ordered[index]


def map_compute_units(name: str):
    import coremltools as ct

    mapping = {
        "all": ct.ComputeUnit.ALL,
        "cpu_only": ct.ComputeUnit.CPU_ONLY,
        "cpu_and_gpu": ct.ComputeUnit.CPU_AND_GPU,
        "cpu_and_ne": ct.ComputeUnit.CPU_AND_NE,
        "cpu_and_neural_engine": ct.ComputeUnit.CPU_AND_NE,
    }
    key = name.lower()
    if key not in mapping:
        raise ValueError(f"Unsupported compute units: {name}")
    return mapping[key]


def run_reference(args: argparse.Namespace) -> dict[str, Any]:
    import coremltools as ct
    import numpy as np

    prompt_tokens = list(args.prompt_tokens)
    if args.max_tokens <= 0:
        raise ValueError("--max-tokens must be > 0")
    if len(prompt_tokens) + args.max_tokens > args.seq_len:
        raise ValueError("prompt + max-tokens exceeds --seq-len")

    weights_dir = Path(args.weights).expanduser().resolve()
    gamma, beta, lm_head = load_top_level_weights(weights_dir)

    compile_start = time.perf_counter()
    model = ct.models.MLModel(str(Path(args.coreml_model).expanduser()), compute_units=map_compute_units(args.compute_units))
    compile_time_ms = (time.perf_counter() - compile_start) * 1000.0

    input_name = model.input_description._fd_spec[0].name  # type: ignore[attr-defined]
    output_name = model.output_description._fd_spec[0].name  # type: ignore[attr-defined]
    model_seq = int(model.get_spec().description.input[0].type.multiArrayType.shape[1])
    if args.seq_len > model_seq:
        raise ValueError(f"Requested seq-len {args.seq_len} exceeds model capacity {model_seq}")

    def predict_hidden(token_ids: list[int]) -> np.ndarray:
        padded = np.zeros((1, model_seq), dtype=np.int32)
        padded[0, : len(token_ids)] = np.asarray(token_ids, dtype=np.int32)
        outputs = model.predict({input_name: padded})
        hidden = np.asarray(outputs[output_name], dtype=np.float32)
        # Accept [1, S, H] or [1, H, 1, S]-style layouts by normalizing to [S, H].
        if hidden.ndim == 4 and hidden.shape[1] == gamma.size:
            # [1, H, 1, S] → [S, H]
            hidden = np.transpose(hidden[0, :, 0, :], (1, 0))
        elif hidden.ndim == 3:
            hidden = hidden[0]
        else:
            raise ValueError(f"Unexpected Core ML hidden shape: {hidden.shape}")
        index = len(token_ids) - 1
        return hidden[index]

    def run_once(emit) -> tuple[list[int], list[float], float, float]:
        rng = np.random.default_rng(args.seed)
        generated: list[int] = []
        latencies: list[float] = []
        context = list(prompt_tokens)
        total_start = time.perf_counter()

        step_start = time.perf_counter()
        hidden = predict_hidden(context)
        normed = layer_norm(hidden, gamma, beta)
        logits = lm_head @ normed
        token = sample_token(logits, args.temperature, rng)
        first_ms = (time.perf_counter() - step_start) * 1000.0
        generated.append(token)
        latencies.append(first_ms)
        context.append(token)
        if emit:
            emit(
                {
                    "type": "token",
                    "token": token,
                    "token_index": 0,
                    "elapsed_ms": first_ms,
                    "token_latency_ms": first_ms,
                    "tokens_per_second": 1000.0 / first_ms if first_ms > 0 else 0.0,
                }
            )

        for index in range(1, args.max_tokens):
            step_start = time.perf_counter()
            hidden = predict_hidden(context)
            normed = layer_norm(hidden, gamma, beta)
            logits = lm_head @ normed
            token = sample_token(logits, args.temperature, rng)
            step_ms = (time.perf_counter() - step_start) * 1000.0
            generated.append(token)
            latencies.append(step_ms)
            context.append(token)
            elapsed = (time.perf_counter() - total_start) * 1000.0
            if emit:
                emit(
                    {
                        "type": "token",
                        "token": token,
                        "token_index": index,
                        "elapsed_ms": elapsed,
                        "token_latency_ms": step_ms,
                        "tokens_per_second": (index + 1) * 1000.0 / elapsed if elapsed > 0 else 0.0,
                    }
                )

        total_ms = (time.perf_counter() - total_start) * 1000.0
        return generated, latencies, first_ms, total_ms

    def emit_line(payload: dict[str, Any]) -> None:
        print(json.dumps(payload, separators=(",", ":")), flush=True)

    if args.emit_events:
        emit_line(
            {
                "type": "compile",
                "compile_time_ms": compile_time_ms,
                "compute_units": args.compute_units,
                "seq_len": model_seq,
            }
        )

    aggregated: list[float] = []
    last: tuple[list[int], list[float], float, float] | None = None
    for iteration in range(args.warmup + args.iterations):
        emit = emit_line if args.emit_events and iteration == (args.warmup + args.iterations - 1) else None
        measured = run_once(emit)
        if iteration >= args.warmup:
            last = measured
            aggregated.extend(measured[1])

    if last is None:
        raise RuntimeError("No measured iteration completed")

    generated_tokens, token_latencies_ms, first_token_latency_ms, total_time_ms = last
    tokens_per_second = (
        len(generated_tokens) * 1000.0 / total_time_ms if total_time_ms > 0 else 0.0
    )
    result = build_comparison_result(
        generated_tokens=generated_tokens,
        compile_time_ms=compile_time_ms,
        first_token_latency_ms=first_token_latency_ms,
        tokens_per_second=tokens_per_second,
        median_token_ms=percentile(aggregated, 0.5),
        p95_token_ms=percentile(aggregated, 0.95),
        token_latencies_ms=token_latencies_ms,
        total_time_ms=total_time_ms,
        compute_units=args.compute_units,
        seq_len=model_seq,
    )

    if args.emit_events:
        completed = {"type": "completed", **result}
        emit_line(completed)
    else:
        print(json.dumps(result), flush=True)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        run_reference(args)
    except Exception as exc:  # pragma: no cover
        print(f"run_gpt2_coreml_reference failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
