#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$ROOT/Tests/MILGeneratorTests/Fixtures"

mkdir -p "$OUT_DIR"

cd "$ROOT/training"

xcrun clang -O2 -fobjc-arc -framework Foundation \
    golden_mil_gen.m -o /tmp/golden_mil_gen

/tmp/golden_mil_gen "$OUT_DIR"

echo "Wrote fixtures to: $OUT_DIR"

