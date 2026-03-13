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

# Bump this when the claim-summary output contract changes.
CLAIM_VERSION=3

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
DRY_RUN="${DRY_RUN:-0}"

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
echo "claim_version=$CLAIM_VERSION"
echo "timestamp=$(date -Iseconds)"
echo "git_commit=$(git -C "$ROOT" rev-parse HEAD)"
echo "git_branch=$(git -C "$ROOT" rev-parse --abbrev-ref HEAD)"
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
harness_exit=0
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
DRY_RUN="$DRY_RUN" \
"$ROOT/scripts/reproduce_exact_4x.sh" || harness_exit=$?
# Exit code 2 = gate fail (parity), 1 = runtime error, 0 = pass/warn
if [[ $harness_exit -eq 1 ]]; then
  echo "FATAL: Inner harness failed with runtime error (exit 1)" >&2
  exit 1
fi

{
  echo "claim_version=$CLAIM_VERSION"
  echo "timestamp=$(date -Iseconds)"
  echo "git_commit=$(git -C "$ROOT" rev-parse HEAD)"
  echo "git_branch=$(git -C "$ROOT" rev-parse --abbrev-ref HEAD)"
  echo "results_dir=$RESULTS_DIR"
  echo "dataset=$DATASET_PATH"
  echo "dataset_sha256=$(shasum -a 256 "$DATASET_PATH" | awk '{print $1}')"
  echo "dataset_bytes=$(wc -c < "$DATASET_PATH" | tr -d ' ')"
  echo "max_corpus_bytes=$MAX_CORPUS_BYTES"
  echo "artifact_prefix=$ARTIFACT_PREFIX"
  echo "artifact_manifest_sha256=$(shasum -a 256 "$ARTIFACT_PREFIX.manifest.json" | awk '{print $1}')"
  echo "offline_gate_json=$OFFLINE_GATE_JSON"
  echo "offline_gate_sha256=$(shasum -a 256 "$OFFLINE_GATE_JSON" | awk '{print $1}')"
  echo "coreml_model=$COREML_MODEL"
  if [[ -d "$COREML_MODEL" ]]; then
    echo "coreml_model_sha256=$(find "$COREML_MODEL" -type f | sort | xargs shasum -a 256 | shasum -a 256 | awk '{print $1}')"
  elif [[ -f "$COREML_MODEL" ]]; then
    echo "coreml_model_sha256=$(shasum -a 256 "$COREML_MODEL" | awk '{print $1}')"
  fi
  echo "prompt_token=$PROMPT_TOKEN"
  echo "recurrent_checkpoint_sha256=$(shasum -a 256 "$ARTIFACT_PREFIX.recurrent.bin" | awk '{print $1}')"
  echo "future_sidecar_sha256=$(shasum -a 256 "$ARTIFACT_PREFIX.future-sidecar.bin" | awk '{print $1}')"
  echo "generation_model_sha256=$(shasum -a 256 "$ARTIFACT_PREFIX.generation.bin" | awk '{print $1}')"
  echo "offline_committed_exact_tokens_per_pass=$(jq -r '.committed_exact_tokens_per_pass' "$OFFLINE_GATE_JSON")"
  echo "offline_accepted_future_tokens_per_pass=$(jq -r '.accepted_future_tokens_per_pass' "$OFFLINE_GATE_JSON")"
  echo "offline_parity_status=$(jq -r '.parity_status' "$OFFLINE_GATE_JSON")"
  echo "control_backend=$CONTROL_BACKEND"
  echo "two_step_backend=$TWO_STEP_BACKEND"
  echo "output_head_backend=$OUTPUT_HEAD_BACKEND"
  echo "public_summary=$PUBLIC_RESULTS_DIR/summary.txt"
  echo "public_summary_json=$PUBLIC_RESULTS_DIR/summary.json"
  # Propagate key metrics from inner harness summary if available
  # Extract all harness metrics from summary.json in a single jq call
  if [[ -f "$PUBLIC_RESULTS_DIR/summary.json" ]]; then
    jq -r '
      def na: . // "n/a";
      "harness_version=\(.harness_version | na)",
      "harness_probe_version=\(.probe_version | na)",
      "harness_two_step_median_ms=\(.two_step.median_ms_per_token)",
      "harness_two_step_p95_ms=\(.two_step.p95_ms_per_token | na)",
      "harness_two_step_p99_ms=\(.two_step.p99_ms_per_token | na)",
      "harness_coreml_median_ms=\(.coreml.median_ms_per_token)",
      "harness_coreml_p95_ms=\(.coreml.p95_ms_per_token | na)",
      "harness_coreml_p99_ms=\(.coreml.p99_ms_per_token | na)",
      "harness_speedup_median=\(.two_step_speedup_vs_coreml)",
      "harness_speedup_min=\(.two_step_speedup_min)",
      "harness_speedup_max=\(.two_step_speedup_max)",
      "harness_all_parity=\(.all_parity_match)",
      "harness_parity_total=\(.parity_total | na)",
      "harness_parity_match_counts=\(.per_run_parity_match_count // [] | tojson)",
      "harness_two_step_cv=\(.two_step.cv)",
      "harness_control_median_ms=\(.control.median_ms_per_token)",
      "harness_control_p95_ms=\(.control.p95_ms_per_token | na)",
      "harness_control_p99_ms=\(.control.p99_ms_per_token | na)",
      "harness_control_cv=\(.control.cv)",
      "harness_coreml_cv=\(.coreml.cv)",
      "harness_speedup_cv=\(.two_step_speedup_cv | na)",
      "harness_two_step_mean_ms=\(.two_step.mean_ms_per_token | na)",
      "harness_two_step_stddev_ms=\(.two_step.stddev_ms_per_token | na)",
      "harness_two_step_iqr_ms=\(.two_step.iqr_ms | na)",
      "harness_control_mean_ms=\(.control.mean_ms_per_token | na)",
      "harness_control_stddev_ms=\(.control.stddev_ms_per_token | na)",
      "harness_control_iqr_ms=\(.control.iqr_ms | na)",
      "harness_coreml_mean_ms=\(.coreml.mean_ms_per_token | na)",
      "harness_coreml_stddev_ms=\(.coreml.stddev_ms_per_token | na)",
      "harness_coreml_iqr_ms=\(.coreml.iqr_ms | na)",
      "harness_speedup_mean=\(.two_step_speedup_mean | na)",
      "harness_speedup_stddev=\(.two_step_speedup_stddev | na)",
      "harness_control_speedup=\(.control_speedup_vs_coreml | na)",
      "harness_control_speedup_min=\(.control_speedup_min | na)",
      "harness_control_speedup_max=\(.control_speedup_max | na)",
      "harness_control_speedup_cv=\(.control_speedup_cv | na)",
      "harness_total_elapsed_s=\(.total_elapsed_s | na)",
      "harness_requested_repeats=\(.requested_repeats | na)",
      "harness_valid_runs=\(.valid_runs)",
      "harness_failed_runs=\(.failed_runs // 0)",
      "harness_probe_sha256=\(.artifact_hashes.probe_sha256 | na)",
      "harness_coreml_model_sha256=\(.artifact_hashes.coreml_model_sha256 | na)",
      "harness_recurrent_sha256=\(.artifact_hashes.recurrent_checkpoint_sha256 | na)",
      "harness_sidecar_sha256=\(.artifact_hashes.future_sidecar_sha256 | na)",
      "harness_generation_sha256=\(.artifact_hashes.generation_model_sha256 | na)",
      "harness_chip=\(.host.chip | na)",
      "harness_thermal_pressure=\(.host.thermal_pressure | na)",
      "harness_power_source=\(.host.power_source | na)",
      "harness_total_stderr_lines=\(.per_run_stderr_lines // [] | add // 0)",
      "harness_contract_hash=\(.benchmark_contract.contract_hash | na)",
      "harness_probe_wall_range_s=\(.probe_wall_range_s | na)",
      "harness_outer_elapsed_range_s=\(.outer_elapsed_range_s | na)",
      "harness_gate_status=\(.reproducibility.gate_status // "unknown")",
      "harness_outlier_count=\(.reproducibility.outlier_count // 0)",
      "harness_outlier_two_step=\(.reproducibility.outlier_detail.two_step.count // 0)",
      "harness_outlier_control=\(.reproducibility.outlier_detail.control.count // 0)",
      "harness_outlier_coreml=\(.reproducibility.outlier_detail.coreml.count // 0)",
      "harness_outlier_speedup=\(.reproducibility.outlier_detail.speedup.count // 0)",
      "harness_gate_warnings=\(.reproducibility.warnings // [] | tojson)"
    ' "$PUBLIC_RESULTS_DIR/summary.json"
  elif [[ -f "$PUBLIC_RESULTS_DIR/summary.txt" ]]; then
    gate_line="$(grep '^gate_status=' "$PUBLIC_RESULTS_DIR/summary.txt" || true)"
    if [[ -n "$gate_line" ]]; then
      echo "harness_$gate_line"
    fi
  fi
  claim_elapsed_s=$(( $(date +%s) - claim_start_epoch ))
  echo "claim_total_elapsed_s=$claim_elapsed_s"
  echo "harness_exit_code=$harness_exit"
} | tee "$RESULTS_DIR/claim-summary.txt"

# Propagate inner harness gate exit code (2 = gate fail)
exit "$harness_exit"
