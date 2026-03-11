#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export INPUT_MODE="${INPUT_MODE:-echo}"
export REPEATS="${REPEATS:-7}"
export WARMUP="${WARMUP:-3}"
export ITERATIONS="${ITERATIONS:-20}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8}"
export MAX_SEQUENCE_TOKENS="${MAX_SEQUENCE_TOKENS:-32}"
export LAYER_COUNT="${LAYER_COUNT:-6}"
export CONTROL_BACKEND="${CONTROL_BACKEND:-fused-triplet}"
export TWO_STEP_BACKEND="${TWO_STEP_BACKEND:-fused-triplet}"
export OUTPUT_HEAD_BACKEND="${OUTPUT_HEAD_BACKEND:-ane-rmsnorm-classifier}"
export RESULTS_DIR="${RESULTS_DIR:-$ROOT/results/publishable-3x-echo-$(date +%Y%m%d-%H%M%S)}"

exec "$ROOT/scripts/reproduce_exact_4x.sh"
