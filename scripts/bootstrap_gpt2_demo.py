#!/usr/bin/env python3
"""Bootstrap default GPT-2 demo weights + tokenizer for espresso-generate.

Invoked by EspressoGenerate as:

    bootstrap_gpt2_demo.py \\
        --weights-out <dir> \\
        --tokenizer-out <dir> \\
        --cache-dir <dir>

Requires: numpy, torch, transformers
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any, Callable


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights-out", required=True, help="Espresso BLOBFILE weights directory")
    parser.add_argument("--tokenizer-out", required=True, help="Directory for vocab.json + merges.txt")
    parser.add_argument("--cache-dir", required=True, help="Hugging Face cache directory")
    parser.add_argument("--model", default="gpt2", help="Hugging Face GPT-2 model id or local path")
    parser.add_argument(
        "--metadata-name",
        default="gpt2_124m",
        help="Name written into metadata.json (default: gpt2_124m)",
    )
    return parser.parse_args(argv)


def _load_convert_pretrained() -> Callable[..., None]:
    path = SCRIPT_DIR / "convert_weights_gpt2.py"
    spec = importlib.util.spec_from_file_location("convert_weights_gpt2", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.convert_pretrained_gpt2


def bootstrap_gpt2_demo(
    *,
    model_name: str,
    weights_out: Path,
    tokenizer_out: Path,
    cache_dir: Path,
    metadata_name: str = "gpt2_124m",
    convert_pretrained_gpt2: Callable[..., None] | None = None,
    GPT2Tokenizer: Any = None,
) -> None:
    if GPT2Tokenizer is None:
        try:
            from transformers import GPT2Tokenizer as HFGPT2Tokenizer
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "transformers is required to bootstrap the GPT-2 demo. "
                "Install torch + transformers, or set ESPRESSO_TOOLS_PYTHON to a Python that has them."
            ) from exc
        GPT2Tokenizer = HFGPT2Tokenizer

    if convert_pretrained_gpt2 is None:
        convert_pretrained_gpt2 = _load_convert_pretrained()

    weights_out = Path(weights_out).expanduser().resolve()
    tokenizer_out = Path(tokenizer_out).expanduser().resolve()
    cache_dir = Path(cache_dir).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"Converting {model_name} weights → {weights_out}", flush=True)
    convert_pretrained_gpt2(
        model_name,
        weights_out,
        cache_dir=str(cache_dir),
        metadata_name=metadata_name,
    )

    print(f"Saving GPT-2 tokenizer → {tokenizer_out}", flush=True)
    tokenizer_out.mkdir(parents=True, exist_ok=True)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name, cache_dir=str(cache_dir))
    tokenizer.save_pretrained(str(tokenizer_out))

    vocab = tokenizer_out / "vocab.json"
    merges = tokenizer_out / "merges.txt"
    if not vocab.is_file() or not merges.is_file():
        raise RuntimeError(
            f"Tokenizer save did not produce vocab.json + merges.txt under {tokenizer_out}"
        )

    print("GPT-2 demo artifacts ready.", flush=True)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    bootstrap_gpt2_demo(
        model_name=args.model,
        weights_out=Path(args.weights_out),
        tokenizer_out=Path(args.tokenizer_out),
        cache_dir=Path(args.cache_dir),
        metadata_name=args.metadata_name,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
