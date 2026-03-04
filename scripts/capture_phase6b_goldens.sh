#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -f "$ROOT/training/capture_phase6b_goldens.m" ]]; then
  SRC="$ROOT/training/capture_phase6b_goldens.m"
elif [[ -f "$ROOT/archive/training/capture_phase6b_goldens.m" ]]; then
  SRC="$ROOT/archive/training/capture_phase6b_goldens.m"
else
  echo "[phase6b] ERROR: capture_phase6b_goldens.m not found in training/ or archive/training/" >&2
  exit 1
fi

BIN_DIR="$ROOT/.build/phase7-tools"
BIN="$BIN_DIR/capture_phase6b_goldens"
mkdir -p "$BIN_DIR"

echo "[phase6b] building ObjC fixture capture tool"
xcrun clang -O2 -Wno-deprecated-declarations -fobjc-arc \
  -o "$BIN" "$SRC" \
  -framework Foundation -framework CoreML -framework IOSurface -ldl

echo "[phase6b] generating fixture binaries"
"$BIN"

echo "[phase6b] done"
