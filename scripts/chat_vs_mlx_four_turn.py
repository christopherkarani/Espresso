#!/usr/bin/env python3
"""Drive `./espresso chat --vs mlx --plain --greedy` through four scripted turns.

This is the later GIF harness. It does not record a GIF.

Usage:
  python3 scripts/chat_vs_mlx_four_turn.py
  python3 scripts/chat_vs_mlx_four_turn.py --model ~/Library/Caches/Espresso/qwen25-15b/Qwen2.5-1.5B-Instruct.esp
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

DEFAULT_MODEL = Path.home() / "Library/Caches/Espresso/qwen25-15b/Qwen2.5-1.5B-Instruct.esp"

TURNS = [
    "what is a good way to learn Swift concurrency?",
    "give me a one-sentence actor example",
    "what is structured concurrency?",
    "summarize that in two sentences",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--espresso",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "espresso",
    )
    parser.add_argument("-n", "--max-tokens", type=int, default=64)
    parser.add_argument("--mlx-quant", default=None, help="Optional labeled MLX quant, e.g. 4bit")
    parser.add_argument("--timeout", type=int, default=3600)
    return parser.parse_args()


def build_command(args: argparse.Namespace) -> list[str]:
    command = [
        str(args.espresso),
        "chat",
        "--vs",
        "mlx",
        "--plain",
        "--greedy",
        "--power",
        "--model",
        str(args.model),
        "-n",
        str(args.max_tokens),
    ]
    if args.mlx_quant:
        command.extend(["--mlx-quant", args.mlx_quant])
    return command


def main() -> int:
    args = parse_args()
    if not args.model.exists():
        print(f"model bundle not found: {args.model}", file=sys.stderr)
        return 2
    command = build_command(args)
    stdin = "\n".join(TURNS) + "\n/exit\n"
    print(" ".join(command), file=sys.stderr)
    completed = subprocess.run(
        command,
        input=stdin,
        text=True,
        timeout=args.timeout,
        check=False,
    )
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
