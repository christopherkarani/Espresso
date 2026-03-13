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
OUTPUT_HEAD_BACKEND="${OUTPUT_HEAD_BACKEND:-ane-rmsnorm-classifier}"

mkdir -p "$RESULTS_DIR"
claim_start_epoch=$(date +%s)

# Signal trap: mark results as interrupted for regression diagnosis
cleanup_on_interrupt() {
  echo ""
  echo "INTERRUPTED at $(date -Iseconds) (signal received)" | tee "$RESULTS_DIR/INTERRUPTED"
  exit 130
}
trap cleanup_on_interrupt INT TERM

echo "=== Espresso Claim Reproduction ==="
echo "timestamp=$(date -Iseconds)"
echo "results_dir=$RESULTS_DIR"
echo ""

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
OUTPUT_HEAD_BACKEND="$OUTPUT_HEAD_BACKEND" \
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
  echo "output_head_backend=$OUTPUT_HEAD_BACKEND"
  echo "public_summary=$PUBLIC_RESULTS_DIR/summary.txt"
  echo "public_summary_json=$PUBLIC_RESULTS_DIR/summary.json"
  # Propagate key metrics from inner harness summary if available
  if [[ -f "$PUBLIC_RESULTS_DIR/summary.json" ]]; then
    echo "harness_probe_version=$(jq -r '.probe_version // "n/a"' "$PUBLIC_RESULTS_DIR/summary.json")"
    echo "harness_two_step_median_ms=$(jq -r '.two_step.median_ms_per_token' "$PUBLIC_RESULTS_DIR/summary.json")"
    echo "harness_two_step_p95_ms=$(jq -r '.two_step.p95_ms_per_token // "n/a"' "$PUBLIC_RESULTS_DIR/summary.json")"
    echo "harness_two_step_p99_ms=$(jq -r '.two_step.p99_ms_per_token // "n/a"' "$PUBLIC_RESULTS_DIR/summary.json")"
    echo "harness_coreml_median_ms=$(jq -r '.coreml.median_ms_per_token' "$PUBLIC_RESULTS_DIR/summary.json")"
    echo "harness_speedup_median=$(jq -r '.two_step_speedup_vs_coreml' "$PUBLIC_RESULTS_DIR/summary.json")"
    echo "harness_speedup_min=$(jq -r '.two_step_speedup_min' "$PUBLIC_RESULTS_DIR/summary.json")"
    echo "harness_speedup_max=$(jq -r '.two_step_speedup_max' "$PUBLIC_RESULTS_DIR/summary.json")"
    echo "harness_all_parity=$(jq -r '.all_parity_match' "$PUBLIC_RESULTS_DIR/summary.json")"
    echo "harness_two_step_cv=$(jq -r '.two_step.cv' "$PUBLIC_RESULTS_DIR/summary.json")"
    echo "harness_control_cv=$(jq -r '.control.cv' "$PUBLIC_RESULTS_DIR/summary.json")"
    echo "harness_coreml_cv=$(jq -r '.coreml.cv' "$PUBLIC_RESULTS_DIR/summary.json")"
    echo "harness_speedup_cv=$(jq -r '.two_step_speedup_cv // "n/a"' "$PUBLIC_RESULTS_DIR/summary.json")"
    echo "harness_two_step_mean_ms=$(jq -r '.two_step.mean_ms_per_token // "n/a"' "$PUBLIC_RESULTS_DIR/summary.json")"
    echo "harness_two_step_stddev_ms=$(jq -r '.two_step.stddev_ms_per_token // "n/a"' "$PUBLIC_RESULTS_DIR/summary.json")"
    echo "harness_two_step_iqr_ms=$(jq -r '.two_step.iqr_ms // "n/a"' "$PUBLIC_RESULTS_DIR/summary.json")"
    echo "harness_control_mean_ms=$(jq -r '.control.mean_ms_per_token // "n/a"' "$PUBLIC_RESULTS_DIR/summary.json")"
    echo "harness_control_stddev_ms=$(jq -r '.control.stddev_ms_per_token // "n/a"' "$PUBLIC_RESULTS_DIR/summary.json")"
    echo "harness_control_iqr_ms=$(jq -r '.control.iqr_ms // "n/a"' "$PUBLIC_RESULTS_DIR/summary.json")"
    echo "harness_coreml_mean_ms=$(jq -r '.coreml.mean_ms_per_token // "n/a"' "$PUBLIC_RESULTS_DIR/summary.json")"
    echo "harness_coreml_stddev_ms=$(jq -r '.coreml.stddev_ms_per_token // "n/a"' "$PUBLIC_RESULTS_DIR/summary.json")"
    echo "harness_coreml_iqr_ms=$(jq -r '.coreml.iqr_ms // "n/a"' "$PUBLIC_RESULTS_DIR/summary.json")"
    echo "harness_speedup_mean=$(jq -r '.two_step_speedup_mean // "n/a"' "$PUBLIC_RESULTS_DIR/summary.json")"
    echo "harness_speedup_stddev=$(jq -r '.two_step_speedup_stddev // "n/a"' "$PUBLIC_RESULTS_DIR/summary.json")"
    echo "harness_total_elapsed_s=$(jq -r '.total_elapsed_s // "n/a"' "$PUBLIC_RESULTS_DIR/summary.json")"
    echo "harness_requested_repeats=$(jq -r '.requested_repeats // "n/a"' "$PUBLIC_RESULTS_DIR/summary.json")"
    echo "harness_valid_runs=$(jq -r '.valid_runs' "$PUBLIC_RESULTS_DIR/summary.json")"
    echo "harness_failed_runs=$(jq -r '.failed_runs // 0' "$PUBLIC_RESULTS_DIR/summary.json")"
    echo "harness_probe_sha256=$(jq -r '.artifact_hashes.probe_sha256 // "n/a"' "$PUBLIC_RESULTS_DIR/summary.json")"
    echo "harness_coreml_model_sha256=$(jq -r '.artifact_hashes.coreml_model_sha256 // "n/a"' "$PUBLIC_RESULTS_DIR/summary.json")"
    echo "harness_recurrent_sha256=$(jq -r '.artifact_hashes.recurrent_checkpoint_sha256 // "n/a"' "$PUBLIC_RESULTS_DIR/summary.json")"
    echo "harness_sidecar_sha256=$(jq -r '.artifact_hashes.future_sidecar_sha256 // "n/a"' "$PUBLIC_RESULTS_DIR/summary.json")"
    echo "harness_generation_sha256=$(jq -r '.artifact_hashes.generation_model_sha256 // "n/a"' "$PUBLIC_RESULTS_DIR/summary.json")"
  fi
  # Propagate gate status from inner harness (prefer JSON source)
  if [[ -f "$PUBLIC_RESULTS_DIR/summary.json" ]]; then
    echo "harness_gate_status=$(jq -r '.reproducibility.gate_status // "unknown"' "$PUBLIC_RESULTS_DIR/summary.json")"
    echo "harness_outlier_count=$(jq -r '.reproducibility.outlier_count // 0' "$PUBLIC_RESULTS_DIR/summary.json")"
    echo "harness_outlier_two_step=$(jq -r '.reproducibility.outlier_detail.two_step.count // 0' "$PUBLIC_RESULTS_DIR/summary.json")"
    echo "harness_outlier_control=$(jq -r '.reproducibility.outlier_detail.control.count // 0' "$PUBLIC_RESULTS_DIR/summary.json")"
    echo "harness_outlier_coreml=$(jq -r '.reproducibility.outlier_detail.coreml.count // 0' "$PUBLIC_RESULTS_DIR/summary.json")"
    echo "harness_outlier_speedup=$(jq -r '.reproducibility.outlier_detail.speedup.count // 0' "$PUBLIC_RESULTS_DIR/summary.json")"
  elif [[ -f "$PUBLIC_RESULTS_DIR/summary.txt" ]]; then
    gate_line="$(grep '^gate_status=' "$PUBLIC_RESULTS_DIR/summary.txt" || true)"
    if [[ -n "$gate_line" ]]; then
      echo "harness_$gate_line"
    fi
  fi
  claim_elapsed_s=$(( $(date +%s) - claim_start_epoch ))
  echo "claim_total_elapsed_s=$claim_elapsed_s"
} | tee "$RESULTS_DIR/claim-summary.txt"
