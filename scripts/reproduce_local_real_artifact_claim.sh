#!/usr/bin/env bash
set -euo pipefail

# Public reproduction entry point for the non-echo exact decode release claim.
# This script:
# 1. builds a local-text token dataset from the repo,
# 2. exports the matching recurrent artifact + future sidecar,
# 3. writes an offline exact-acceptance gate,
# 4. generates a matching zero-weight CoreML trunk, and
# 5. runs the matched ANE/CoreML public harness.
# The default contract reproduces the release documented in
# docs/releases/2026-03-11-non-echo-exact-decode.md.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="${RESULTS_DIR:-$ROOT/results/real-artifact-$(date +%Y%m%d-%H%M%S)}"
DATASET_PATH="$RESULTS_DIR/local-text.uint16.bin"
ARTIFACT_PREFIX="$RESULTS_DIR/local-bigram"
OFFLINE_GATE_JSON="$RESULTS_DIR/offline-gate.json"
PUBLIC_RESULTS_DIR="$RESULTS_DIR/public-harness"
COREML_MODEL="$RESULTS_DIR/transformer_6layer_zero.mlpackage"
COREMLTOOLS_PYTHON="${COREMLTOOLS_PYTHON:-/tmp/coremltools312-install-venv/bin/python}"

LAYER_COUNT="${LAYER_COUNT:-6}"
REPEATS="${REPEATS:-5}"
WARMUP="${WARMUP:-3}"
ITERATIONS="${ITERATIONS:-20}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8}"
MAX_SEQUENCE_TOKENS="${MAX_SEQUENCE_TOKENS:-32}"
MAX_CORPUS_BYTES="${MAX_CORPUS_BYTES:-262144}"
CONTROL_BACKEND="${CONTROL_BACKEND:-identity-zero-trunk}"
TWO_STEP_BACKEND="${TWO_STEP_BACKEND:-identity-zero-trunk}"

mkdir -p "$RESULTS_DIR"

echo "Building local text dataset into $DATASET_PATH"
swift run espresso-train \
  --build-local-text-dataset "$DATASET_PATH" \
  --text-root "$ROOT/Sources" \
  --text-root "$ROOT/docs" \
  --text-root "$ROOT/scripts" \
  --text-root "$ROOT/tasks" \
  --max-corpus-bytes "$MAX_CORPUS_BYTES"

echo "Exporting local bigram artifacts into $ARTIFACT_PREFIX.*"
swift run espresso-train \
  --data "$DATASET_PATH" \
  --export-local-bigram-prefix "$ARTIFACT_PREFIX" \
  --artifact-layer-count "$LAYER_COUNT" \
  --offline-acceptance-json "$OFFLINE_GATE_JSON" \
  --gate-max-new-tokens "$MAX_NEW_TOKENS"

PROMPT_TOKEN="$(jq -r '.promptToken' "$ARTIFACT_PREFIX.manifest.json")"

if [[ ! -x "$COREMLTOOLS_PYTHON" ]]; then
  PY312="${PY312:-/opt/homebrew/opt/python@3.12/bin/python3.12}"
  if [[ ! -x "$PY312" ]]; then
    echo "Expected Python 3.12 at $PY312" >&2
    exit 1
  fi
  echo "Bootstrapping coremltools venv at /tmp/coremltools312-install-venv"
  "$PY312" -m venv /tmp/coremltools312-install-venv
  /tmp/coremltools312-install-venv/bin/pip install coremltools
  COREMLTOOLS_PYTHON="/tmp/coremltools312-install-venv/bin/python"
fi

echo "Generating matching zero-weight CoreML trunk into $COREML_MODEL"
"$COREMLTOOLS_PYTHON" "$ROOT/scripts/generate_coreml_model.py" \
  --layers "$LAYER_COUNT" \
  --weight-mode zero \
  --output "$COREML_MODEL"

echo "Running public recurrent-checkpoint harness"
RESULTS_DIR="$PUBLIC_RESULTS_DIR" \
INPUT_MODE="recurrent-checkpoint" \
CONTROL_BACKEND="$CONTROL_BACKEND" \
TWO_STEP_BACKEND="$TWO_STEP_BACKEND" \
RECURRENT_CHECKPOINT="$ARTIFACT_PREFIX.recurrent.bin" \
FUTURE_SIDECAR="$ARTIFACT_PREFIX.future-sidecar.bin" \
GENERATION_MODEL="$ARTIFACT_PREFIX.generation.bin" \
COREML_MODEL="$COREML_MODEL" \
PROMPT_TOKEN="$PROMPT_TOKEN" \
REPEATS="$REPEATS" \
WARMUP="$WARMUP" \
ITERATIONS="$ITERATIONS" \
MAX_NEW_TOKENS="$MAX_NEW_TOKENS" \
MAX_SEQUENCE_TOKENS="$MAX_SEQUENCE_TOKENS" \
LAYER_COUNT="$LAYER_COUNT" \
"$ROOT/scripts/reproduce_exact_4x.sh"

{
  echo "results_dir=$RESULTS_DIR"
  echo "dataset=$DATASET_PATH"
  echo "artifact_prefix=$ARTIFACT_PREFIX"
  echo "offline_gate_json=$OFFLINE_GATE_JSON"
  echo "coreml_model=$COREML_MODEL"
  echo "prompt_token=$PROMPT_TOKEN"
  echo "offline_committed_exact_tokens_per_pass=$(jq -r '.committed_exact_tokens_per_pass' "$OFFLINE_GATE_JSON")"
  echo "offline_accepted_future_tokens_per_pass=$(jq -r '.accepted_future_tokens_per_pass' "$OFFLINE_GATE_JSON")"
  echo "offline_parity_status=$(jq -r '.parity_status' "$OFFLINE_GATE_JSON")"
  echo "control_backend=$CONTROL_BACKEND"
  echo "two_step_backend=$TWO_STEP_BACKEND"
  echo "public_summary=$PUBLIC_RESULTS_DIR/summary.txt"
} | tee "$RESULTS_DIR/claim-summary.txt"
