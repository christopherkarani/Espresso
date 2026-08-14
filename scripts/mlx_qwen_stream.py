#!/usr/bin/env python3
"""Stream Qwen2.5-1.5B-Instruct from MLX as JSON lines.

Fairness contract:
  * Default load is native fp16/bf16 of Qwen/Qwen2.5-1.5B-Instruct.
  * Quantized checkpoints are rejected unless --allow-quant is set.
  * tok/s is completion-only; load/compile ms is a separate event.
  * The prompt is used as-is (Espresso already applied the chat template).

Protocol:
  stdout JSON lines:
    {"type":"hello","precision":"float16","quantized":false,"repo":"..."}
    {"type":"compile","compile_time_ms":...}
    then, per stdin request {"prompt":"...","max_tokens":N}:
      {"type":"token",...}
      {"type":"completed",...}
      {"type":"ready"}
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any


def emit(payload: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def emit_error(message: str, code: int = 2) -> int:
    emit({"type": "error", "message": message})
    return code


def inspect_precision(model: Any, config: dict[str, Any]) -> tuple[str, bool]:
    quantized = bool(config.get("quantization") or config.get("quantization_config"))
    precision = "unknown"
    try:
        import mlx.core as mx
        from mlx.utils import tree_flatten

        for _, value in tree_flatten(model.parameters()):
            dtype = str(getattr(value, "dtype", ""))
            if "float16" in dtype:
                precision = "float16"
                break
            if "bfloat16" in dtype:
                precision = "bfloat16"
                break
            if "float32" in dtype:
                precision = "float32"
    except Exception:
        pass
    if quantized and precision in {"unknown", "float16", "bfloat16", "float32"}:
        bits = None
        quant = config.get("quantization") or config.get("quantization_config") or {}
        if isinstance(quant, dict):
            bits = quant.get("bits")
        precision = f"{bits}-bit" if bits else "quantized"
    return precision, quantized


def encode_prompt(tokenizer: Any, prompt: str) -> Any:
    if "<|im_start|>" in prompt:
        return tokenizer.encode(prompt, add_special_tokens=False)
    return prompt


def generate_one(model: Any, tokenizer: Any, prompt: str, max_tokens: int, compile_ms: float) -> None:
    from mlx_lm import stream_generate
    import mlx.core as mx

    encoded = encode_prompt(tokenizer, prompt)
    started = time.perf_counter()
    text_parts: list[str] = []
    token_count = 0
    ttft_ms = 0.0
    last_tps = 0.0

    for response in stream_generate(
        model,
        tokenizer,
        encoded,
        max_tokens=max_tokens,
        sampler=lambda logits: mx.argmax(logits, axis=-1),
    ):
        token_count = int(response.generation_tokens)
        piece = response.text or ""
        text_parts.append(piece)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if token_count == 1:
            ttft_ms = elapsed_ms
        last_tps = completion_tokens_per_second(token_count, elapsed_ms)
        emit(
            {
                "type": "token",
                "text": piece,
                "token_index": token_count,
                "elapsed_ms": elapsed_ms,
                "token_latency_ms": ttft_ms if token_count == 1 else 0.0,
                "tokens_per_second": last_tps,
            }
        )

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    emit(
        {
            "type": "completed",
            "text": "".join(text_parts),
            "compile_time_ms": compile_ms,
            "first_token_latency_ms": ttft_ms,
            "tokens_per_second": completion_tokens_per_second(token_count, elapsed_ms),
            "generation_tokens": token_count,
        }
    )
    emit({"type": "ready"})


def completion_tokens_per_second(generated_token_count: int, completion_ms: float) -> float:
    if generated_token_count <= 0 or completion_ms <= 0:
        return 0.0
    return generated_token_count / (completion_ms / 1000.0)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--model-path", required=True, help="Local snapshot or HF repo id")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument(
        "--allow-quant",
        action="store_true",
        help="Permit a quantized MLX checkpoint. Both TUI footers must label it.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        from mlx_lm import load
    except ImportError:
        return emit_error(
            "MLX is not installed. Install the native-precision (fp16/bf16) runtime, then retry:\n"
            "  python3 -m pip install mlx-lm\n"
            "Do not install a 4-bit quantized build to make this pane work.",
            code=2,
        )

    load_started = time.perf_counter()
    try:
        model, tokenizer, config = load(args.model_path, return_config=True)
    except Exception as exc:  # noqa: BLE001 — surface the load failure to Swift
        return emit_error(f"Failed to load MLX model at {args.model_path}: {exc}", code=2)

    compile_ms = (time.perf_counter() - load_started) * 1000.0
    precision, quantized = inspect_precision(model, config)
    if quantized and not args.allow_quant:
        return emit_error(
            "MLX loaded a quantized checkpoint without --mlx-quant. "
            "Unlabeled 4-bit is rejected. Re-run with --mlx-quant 4bit so both footers label it, "
            "or load native fp16/bf16.",
            code=3,
        )

    emit(
        {
            "type": "hello",
            "precision": precision,
            "quantized": quantized,
            "repo": args.repo,
        }
    )
    emit({"type": "compile", "compile_time_ms": compile_ms})
    emit({"type": "ready"})

    remaining_compile_ms = compile_ms
    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError as exc:
            return emit_error(f"Invalid MLX request: {exc}")
        prompt = str(request.get("prompt") or "")
        max_tokens = int(request.get("max_tokens") or args.max_tokens)
        generate_one(model, tokenizer, prompt, max_tokens, remaining_compile_ms)
        remaining_compile_ms = 0.0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
