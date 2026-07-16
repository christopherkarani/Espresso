#!/usr/bin/env python3
"""Fail if README public claim numbers drift from benchmarks/results/latest.json.

Product rule: only latest.json-backed figures may appear as headline README claims.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    latest_path = root / "benchmarks" / "results" / "latest.json"
    readme_path = root / "README.md"

    latest = json.loads(latest_path.read_text())
    readme = readme_path.read_text()

    results = {row["name"]: row for row in latest["results"]}
    required = [
        ("Espresso ANE (recurrent fused, 6-layer)", "tokens_per_sec", r"\*\*519\*\*"),
        ("Espresso ANE (recurrent fused, 6-layer)", "ms_per_token", r"\*\*1\.93\*\*"),
        ("Espresso ANE (direct transformer, 6-layer)", "tokens_per_sec", r"\b153\b"),
        ("CoreML (cpuAndNeuralEngine baseline)", "tokens_per_sec", r"\b152\b"),
    ]

    errors: list[str] = []
    for name, field, pattern in required:
        if name not in results:
            errors.append(f"missing result row in latest.json: {name}")
            continue
        value = results[name][field]
        if field == "tokens_per_sec":
            expected = int(round(float(value)))
        else:
            # README uses 2-decimal ms/token for the fused row
            expected = float(value)
            if not re.search(pattern, readme):
                errors.append(
                    f"README missing claim for {name}.{field} "
                    f"(latest={value!r}, pattern={pattern!r})"
                )
            continue
        if not re.search(pattern, readme):
            errors.append(
                f"README missing claim for {name}.{field} "
                f"(latest={expected}, pattern={pattern!r})"
            )

    speedup = latest.get("speedup_vs_coreml")
    if speedup is None:
        errors.append("latest.json missing speedup_vs_coreml")
    else:
        # Accept the canonical README form 3.41× (matches latest.json)
        expected = f"{float(speedup):.2f}"
        if f"{expected}×" not in readme and f"{expected}x" not in readme.lower():
            errors.append(
                f"README missing speedup_vs_coreml claim (latest={speedup})"
            )

    # Inflated peaks that must not re-enter the README as product claims
    banned = [
        (r"\b926\b", "926 (trunk-only peak, not a latest.json claim)"),
        (r"4\.76\s*[x×]", "4.76× (not the end-to-end latest.json speedup)"),
        (r"\b196\b.*tok", "196 tok/s CoreML (stale; latest.json uses 152)"),
    ]
    for pattern, label in banned:
        if re.search(pattern, readme, flags=re.IGNORECASE):
            errors.append(f"README contains non-claim number: {label}")

    if errors:
        print("README claim check failed:", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    print(
        f"ok: README claims match {latest_path.relative_to(root)} "
        f"(speedup_vs_coreml={speedup}, results={len(results)})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
