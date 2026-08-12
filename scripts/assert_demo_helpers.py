#!/usr/bin/env python3
"""Fail if espresso-generate's required GPT-2 demo helper scripts are missing.

Keeps the primary `./espresso` / doctor bootstrap path clone-ready.
"""

from __future__ import annotations

import sys
from pathlib import Path


REQUIRED = (
    "bootstrap_gpt2_demo.py",
    "export_gpt2_coreml.py",
    "run_gpt2_coreml_reference.py",
    "convert_weights_gpt2.py",
)


def main() -> int:
    scripts_dir = Path(__file__).resolve().parent
    missing = [name for name in REQUIRED if not (scripts_dir / name).is_file()]
    if missing:
        print(
            "Missing GPT-2 demo helper scripts required by espresso-generate:\n  - "
            + "\n  - ".join(missing),
            file=sys.stderr,
        )
        return 1
    print("ok: demo helpers present:")
    for name in REQUIRED:
        print(f"  - {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
