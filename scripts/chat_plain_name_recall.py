#!/usr/bin/env python3
"""Drive `./espresso chat --plain --greedy` through a 10-turn name-recall smoke.

Turn 1 introduces the name Ada. A later turn asks "what is my name?".
Under --greedy the reply must mention Ada; otherwise history was dropped.

Chat always forces ESPRESSO_REALMODEL_DISABLE_HYBRID_FALLBACK=1. The script
fails if a footer is missing path=hybrid.

Usage:
  python3 scripts/chat_plain_name_recall.py
  python3 scripts/chat_plain_name_recall.py --model ~/Library/Caches/Espresso/qwen25-15b/Qwen2.5-1.5B-Instruct.esp
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

DEFAULT_MODEL = Path.home() / "Library/Caches/Espresso/qwen25-15b/Qwen2.5-1.5B-Instruct.esp"

TURNS = [
    "my name is Ada",
    "nice to meet you",
    "I enjoy writing Swift",
    "what language do I enjoy?",
    "I live in Nairobi",
    "what is my name?",
    "where do I live?",
    "say hello in one sentence",
    "remind me who I am",
    "goodbye",
]

RECALL_TURN_INDEX = 5  # 0-based: "what is my name?"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help="Packed .esp bundle (default: %(default)s)",
    )
    parser.add_argument(
        "--espresso",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "espresso",
        help="Path to the ./espresso launcher",
    )
    parser.add_argument("-n", "--max-tokens", type=int, default=48)
    parser.add_argument(
        "--transcript",
        type=Path,
        default=Path("docs/qwen15b-chat-name-recall.txt"),
        help="Where to write the captured --plain transcript",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=3600,
        help="Seconds to wait for the whole 10-turn run",
    )
    return parser.parse_args()


def extract_assistant_replies(stdout: str) -> list[str]:
    replies: list[str] = []
    current: list[str] = []
    in_reply = False
    for raw in stdout.splitlines():
        line = raw.rstrip("\n")
        marker = "qwen> "
        if marker in line:
            if in_reply and current:
                replies.append("\n".join(current).strip())
            current = [line[line.index(marker) + len(marker) :]]
            in_reply = True
            continue
        if in_reply:
            if line.startswith("you> ") or line.startswith("tok/s "):
                replies.append("\n".join(current).strip())
                current = []
                in_reply = False
            else:
                current.append(line)
    if in_reply and current:
        replies.append("\n".join(current).strip())
    return replies


def main() -> int:
    args = parse_args()
    model = args.model.expanduser()
    if not model.exists():
        print(f"error: model bundle not found: {model}", file=sys.stderr)
        return 2

    stdin = "\n".join(TURNS) + "\n/exit\n"
    command = [
        str(args.espresso),
        "chat",
        "--model",
        str(model),
        "--plain",
        "--greedy",
        "-n",
        str(args.max_tokens),
    ]
    env = os.environ.copy()
    env["ESPRESSO_REALMODEL_DISABLE_HYBRID_FALLBACK"] = "1"
    print("running:", " ".join(command), file=sys.stderr)
    completed = subprocess.run(
        command,
        input=stdin,
        text=True,
        capture_output=True,
        timeout=args.timeout,
        env=env,
    )
    transcript = (
        f"$ {' '.join(command)}\n"
        f"# stdin turns:\n"
        + "\n".join(f"# {index + 1}. {turn}" for index, turn in enumerate(TURNS))
        + "\n\n--- stdout ---\n"
        + completed.stdout
        + "\n--- stderr ---\n"
        + completed.stderr
        + f"\n--- exit {completed.returncode} ---\n"
    )
    args.transcript.parent.mkdir(parents=True, exist_ok=True)
    args.transcript.write_text(transcript)
    print(f"wrote {args.transcript}", file=sys.stderr)

    if completed.returncode != 0:
        print(completed.stderr, file=sys.stderr)
        print(f"error: chat exited {completed.returncode}", file=sys.stderr)
        return completed.returncode or 1

    if "path=hybrid" not in completed.stdout:
        print("error: footer missing path=hybrid (silent CPU fallback?)", file=sys.stderr)
        return 1

    replies = extract_assistant_replies(completed.stdout)
    if len(replies) < RECALL_TURN_INDEX + 1:
        print(
            f"error: expected at least {RECALL_TURN_INDEX + 1} assistant replies, got {len(replies)}",
            file=sys.stderr,
        )
        return 1

    recall = replies[RECALL_TURN_INDEX]
    if "ada" not in recall.lower():
        print(
            "error: name-recall failed; later turn did not mention Ada.\n"
            f"reply: {recall!r}\n"
            "This is a history/template bug, not sampling noise (--greedy).",
            file=sys.stderr,
        )
        return 1

    print(f"ok: {len(replies)} replies, recall={recall!r}, path=hybrid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
