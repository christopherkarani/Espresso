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
CLAIM_VERSION=7

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

# Early validation of claim-level parameters (mirrors harness validation)
# Fail fast before expensive pipeline steps (dataset build, artifact export, CoreML gen)
if [[ "$REPEATS" -lt 3 || $((REPEATS % 2)) -ne 1 ]]; then
  echo "REPEATS must be an odd integer >= 3 (got: $REPEATS)" >&2
  exit 1
fi
if [[ "$LAYER_COUNT" -lt 1 ]]; then
  echo "LAYER_COUNT must be >= 1 (got: $LAYER_COUNT)" >&2
  exit 1
fi
if [[ "$MAX_NEW_TOKENS" -lt 1 ]]; then
  echo "MAX_NEW_TOKENS must be >= 1 (got: $MAX_NEW_TOKENS)" >&2
  exit 1
fi
if [[ "$MAX_SEQUENCE_TOKENS" -lt $((MAX_NEW_TOKENS + 1)) ]]; then
  echo "MAX_SEQUENCE_TOKENS ($MAX_SEQUENCE_TOKENS) must be >= MAX_NEW_TOKENS + 1 ($((MAX_NEW_TOKENS + 1)))" >&2
  exit 1
fi
if [[ "$WARMUP" -lt 0 ]]; then
  echo "WARMUP must be >= 0 (got: $WARMUP)" >&2
  exit 1
fi
if [[ "$ITERATIONS" -lt 1 ]]; then
  echo "ITERATIONS must be >= 1 (got: $ITERATIONS)" >&2
  exit 1
fi
if [[ "$MAX_CORPUS_BYTES" -lt 1 ]]; then
  echo "MAX_CORPUS_BYTES must be >= 1 (got: $MAX_CORPUS_BYTES)" >&2
  exit 1
fi

# Pre-check: ensure coremltools Python or a fallback Python 3.12 is available
# This catches missing Python before spending minutes on dataset build + artifact export
if [[ ! -x "$COREMLTOOLS_PYTHON" ]]; then
  PY312_PRECHECK="${PY312:-/opt/homebrew/opt/python@3.12/bin/python3.12}"
  if [[ ! -x "$PY312_PRECHECK" ]]; then
    echo "FATAL: Neither COREMLTOOLS_PYTHON ($COREMLTOOLS_PYTHON) nor Python 3.12 ($PY312_PRECHECK) found — cannot generate CoreML model" >&2
    exit 1
  fi
fi

# Pre-check: jq is required for JSON validation and summary extraction
if ! command -v jq &>/dev/null; then
  echo "FATAL: jq is required but not found on PATH" >&2
  exit 1
fi

mkdir -p "$RESULTS_DIR"
# Check for stale claim data in the results directory
if [[ -f "$RESULTS_DIR/claim-summary.txt" ]]; then
  echo "FATAL: results directory $RESULTS_DIR already contains claim-summary.txt — stale data would contaminate results. Use a clean directory." >&2
  exit 1
fi
claim_start_epoch=$(date +%s)

# Signal trap: mark results as interrupted for regression diagnosis
cleanup_on_interrupt() {
  echo ""
  echo "INTERRUPTED at $(date -Iseconds) (signal received)" | tee "$RESULTS_DIR/INTERRUPTED"
  exit 130
}
trap cleanup_on_interrupt INT TERM

git_commit_start="$(git -C "$ROOT" rev-parse HEAD)"

git_dirty="$(git -C "$ROOT" status --porcelain 2>/dev/null | head -1)"

echo "=== Espresso Claim Reproduction ==="
echo "claim_version=$CLAIM_VERSION"
echo "timestamp=$(date -Iseconds)"
echo "git_commit=$git_commit_start"
echo "git_branch=$(git -C "$ROOT" rev-parse --abbrev-ref HEAD)"
echo "git_dirty=$([ -n "$git_dirty" ] && echo "true" || echo "false")"
echo "results_dir=$RESULTS_DIR"
echo "disk_free_mb_start=$(df -m "$RESULTS_DIR" 2>/dev/null | awk 'NR==2{print $4}' || echo unknown)"
echo "swift_version=$(swift --version 2>/dev/null | head -1 || echo unknown)"
echo "hostname=$(hostname 2>/dev/null || echo unknown)"
echo "macos_version=$(sw_vers -productVersion 2>/dev/null || echo unknown)"
echo "claim_pid=$$"
echo "memory_free_pct_start=$(sysctl -n kern.memorystatus_level 2>/dev/null || echo unknown)"
echo "thermal_pressure_start=$(pmset -g therm 2>/dev/null | grep -i 'cpu.*speed' | head -1 || echo unknown)"
echo "load_average_start=$(sysctl -n vm.loadavg 2>/dev/null || echo unknown)"
power_source_start="$(pmset -g batt 2>/dev/null | head -1 | sed "s/.*'\(.*\)'.*/\1/" || echo unknown)"
echo "power_source_start=$power_source_start"
echo ""

if [[ "$DRY_RUN" == "1" ]]; then
  echo "=== DRY_RUN: claim-level contract ==="
  echo "claim_version=$CLAIM_VERSION"
  echo "git_commit=$git_commit_start"
  echo "git_branch=$(git -C "$ROOT" rev-parse --abbrev-ref HEAD)"
  echo "git_dirty=$([ -n "$git_dirty" ] && echo "true" || echo "false")"
  echo "results_dir=$RESULTS_DIR"
  echo "layer_count=$LAYER_COUNT"
  echo "repeats=$REPEATS"
  echo "warmup=$WARMUP"
  echo "iterations=$ITERATIONS"
  echo "max_new_tokens=$MAX_NEW_TOKENS"
  echo "max_sequence_tokens=$MAX_SEQUENCE_TOKENS"
  echo "max_corpus_bytes=$MAX_CORPUS_BYTES"
  echo "control_backend=$CONTROL_BACKEND"
  echo "two_step_backend=$TWO_STEP_BACKEND"
  echo "output_head_backend=$OUTPUT_HEAD_BACKEND"
  echo "coremltools_python=$COREMLTOOLS_PYTHON"
  echo "chip=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo unknown)"
  echo "hw_model=$(sysctl -n hw.model 2>/dev/null || echo unknown)"
  echo "ncpu=$(sysctl -n hw.ncpu 2>/dev/null || echo unknown)"
  echo "physical_memory_gb=$(sysctl -n hw.memsize 2>/dev/null | awk '{printf "%.1f", $1/1073741824}' || echo unknown)"
  echo "power_source=$(pmset -g batt 2>/dev/null | head -1 | sed "s/.*'\(.*\)'.*/\1/" || echo unknown)"
  echo "thermal_pressure=$(pmset -g therm 2>/dev/null | grep -i 'cpu.*speed' | head -1 || echo unknown)"
  echo "load_average=$(sysctl -n vm.loadavg 2>/dev/null || echo unknown)"
  echo "macos_version=$(sw_vers -productVersion 2>/dev/null || echo unknown)"
  echo "macos_build=$(sw_vers -buildVersion 2>/dev/null || echo unknown)"
  echo "swift_version=$(swift --version 2>/dev/null | head -1 || echo unknown)"
  echo "hostname=$(hostname 2>/dev/null || echo unknown)"
  echo "claim_pid=$$"
  echo "memory_free_pct=$(sysctl -n kern.memorystatus_level 2>/dev/null || echo unknown)"
  echo "disk_free_mb=$(df -m "$RESULTS_DIR" 2>/dev/null | awk 'NR==2{print $4}' || echo unknown)"
  echo "=== DRY_RUN: skipping pipeline (dataset build, artifact export, CoreML gen, harness) ==="
  exit 0
fi

dataset_build_start=$(date +%s)
echo "Building local text dataset into $DATASET_PATH"
swift run espresso-train \
  --build-local-text-dataset "$DATASET_PATH" \
  --text-root "$ROOT/Sources" \
  --text-root "$ROOT/docs" \
  --text-root "$ROOT/scripts" \
  --text-root "$ROOT/tasks" \
  --max-corpus-bytes "$MAX_CORPUS_BYTES"
dataset_build_elapsed_s=$(( $(date +%s) - dataset_build_start ))

if [[ ! -s "$DATASET_PATH" ]]; then
  echo "FATAL: dataset build succeeded but $DATASET_PATH is missing or empty" >&2
  exit 1
fi

artifact_export_start=$(date +%s)
echo "Exporting local bigram artifacts into $ARTIFACT_PREFIX.*"
swift run espresso-train \
  --data "$DATASET_PATH" \
  --export-local-bigram-prefix "$ARTIFACT_PREFIX" \
  --artifact-layer-count "$LAYER_COUNT" \
  --offline-acceptance-json "$OFFLINE_GATE_JSON" \
  --gate-max-new-tokens "$MAX_NEW_TOKENS"
artifact_export_elapsed_s=$(( $(date +%s) - artifact_export_start ))

for expected_artifact in "$ARTIFACT_PREFIX.manifest.json" "$ARTIFACT_PREFIX.recurrent.bin" "$ARTIFACT_PREFIX.generation.bin" "$ARTIFACT_PREFIX.future-sidecar.bin" "$OFFLINE_GATE_JSON"; do
  if [[ ! -s "$expected_artifact" ]]; then
    echo "FATAL: artifact export succeeded but $expected_artifact is missing or empty" >&2
    exit 1
  fi
done

# Validate that JSON artifacts parse correctly
if ! jq -e '.promptToken' "$ARTIFACT_PREFIX.manifest.json" >/dev/null 2>&1; then
  echo "FATAL: manifest.json missing or has no promptToken field" >&2
  exit 1
fi
if ! jq -e '.parity_status' "$OFFLINE_GATE_JSON" >/dev/null 2>&1; then
  echo "FATAL: offline gate JSON missing or has no parity_status field" >&2
  exit 1
fi
if ! jq -e '.committed_exact_tokens_per_pass and .accepted_future_tokens_per_pass' "$OFFLINE_GATE_JSON" >/dev/null 2>&1; then
  echo "FATAL: offline gate JSON missing token accounting fields" >&2
  exit 1
fi
offline_parity_val="$(jq -r '.parity_status' "$OFFLINE_GATE_JSON")"
if [[ "$offline_parity_val" != "match" && "$offline_parity_val" != "mismatch" ]]; then
  echo "FATAL: offline gate parity_status has unexpected value: '$offline_parity_val' (expected match|mismatch)" >&2
  exit 1
fi

PROMPT_TOKEN="$(jq -r '.promptToken' "$ARTIFACT_PREFIX.manifest.json")"
if ! [[ "$PROMPT_TOKEN" =~ ^[0-9]+$ ]]; then
  echo "FATAL: promptToken from manifest is not a valid non-negative integer: '$PROMPT_TOKEN'" >&2
  exit 1
fi

# Cross-validate promptToken is within vocab range
manifest_vocab_size="$(jq -r '.vocabSize // empty' "$ARTIFACT_PREFIX.manifest.json" 2>/dev/null || true)"
if [[ -n "$manifest_vocab_size" && "$manifest_vocab_size" =~ ^[0-9]+$ ]]; then
  if [[ "$PROMPT_TOKEN" -ge "$manifest_vocab_size" ]]; then
    echo "FATAL: promptToken=$PROMPT_TOKEN is >= vocabSize=$manifest_vocab_size" >&2
    exit 1
  fi
fi

# Cross-validate manifest layerCount matches the claim contract
manifest_layer_count="$(jq -r '.layerCount // empty' "$ARTIFACT_PREFIX.manifest.json" 2>/dev/null || true)"
if [[ -n "$manifest_layer_count" && "$manifest_layer_count" != "$LAYER_COUNT" ]]; then
  echo "FATAL: manifest layerCount=$manifest_layer_count does not match claim LAYER_COUNT=$LAYER_COUNT" >&2
  exit 1
fi

# Cross-validate manifest tokenCount against dataset
dataset_byte_count="$(wc -c < "$DATASET_PATH" | tr -d ' ')"
expected_token_count=$((dataset_byte_count / 2))
manifest_token_count="$(jq -r '.tokenCount // empty' "$ARTIFACT_PREFIX.manifest.json" 2>/dev/null || true)"
if [[ -n "$manifest_token_count" && "$manifest_token_count" != "$expected_token_count" ]]; then
  echo "WARNING: manifest tokenCount=$manifest_token_count but dataset has $expected_token_count tokens ($dataset_byte_count bytes / 2)"
fi

# Compute claim-level contract hash (same algorithm as harness) for cross-validation
CLAIM_CONTRACT_HASH="$(printf '%s\n' \
  "input_mode=recurrent-checkpoint" \
  "control_backend=$CONTROL_BACKEND" \
  "two_step_backend=$TWO_STEP_BACKEND" \
  "output_head_backend=$OUTPUT_HEAD_BACKEND" \
  "warmup=$WARMUP" \
  "iterations=$ITERATIONS" \
  "max_new_tokens=$MAX_NEW_TOKENS" \
  "max_sequence_tokens=$MAX_SEQUENCE_TOKENS" \
  "layer_count=$LAYER_COUNT" \
  "prompt_token=$PROMPT_TOKEN" \
  | shasum -a 256 | awk '{print $1}')"

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

coreml_gen_start=$(date +%s)
echo "Generating matching zero-weight CoreML trunk into $COREML_MODEL"
"$COREMLTOOLS_PYTHON" "$ROOT/scripts/generate_coreml_model.py" \
  --layers "$LAYER_COUNT" \
  --weight-mode zero \
  --output "$COREML_MODEL"
coreml_gen_elapsed_s=$(( $(date +%s) - coreml_gen_start ))

if [[ ! -e "$COREML_MODEL" ]]; then
  echo "FATAL: CoreML model generation succeeded but $COREML_MODEL is missing" >&2
  exit 1
fi
if [[ -d "$COREML_MODEL" ]]; then
  coreml_files="$(find "$COREML_MODEL" -type f | wc -l | tr -d ' ')"
  if [[ "$coreml_files" -lt 1 ]]; then
    echo "FATAL: CoreML model directory $COREML_MODEL contains no files" >&2
    exit 1
  fi
fi

echo "Running public recurrent-checkpoint harness"
harness_start_epoch=$(date +%s)
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
harness_elapsed_s=$(( $(date +%s) - harness_start_epoch ))
# Exit code 2 = gate fail (parity), 1 = runtime error, 0 = pass/warn
if [[ $harness_exit -eq 1 ]]; then
  echo "FATAL: Inner harness failed with runtime error (exit 1)" >&2
  exit 1
fi

# Validate harness produced a valid summary.json
EXPECTED_HARNESS_VERSION=8
if [[ -f "$PUBLIC_RESULTS_DIR/summary.json" ]]; then
  actual_hv="$(jq -r '.harness_version // empty' "$PUBLIC_RESULTS_DIR/summary.json" 2>/dev/null || true)"
  if [[ -z "$actual_hv" ]]; then
    echo "WARNING: harness summary.json missing harness_version field" >&2
  elif [[ "$actual_hv" != "$EXPECTED_HARNESS_VERSION" ]]; then
    echo "WARNING: harness version mismatch: expected=$EXPECTED_HARNESS_VERSION actual=$actual_hv" >&2
  fi
  # Also validate probe version
  actual_pv="$(jq -r '.probe_version // empty' "$PUBLIC_RESULTS_DIR/summary.json" 2>/dev/null || true)"
  if [[ -n "$actual_pv" && "$actual_pv" != "7" ]]; then
    echo "WARNING: probe version mismatch: expected=7 actual=$actual_pv" >&2
  fi
fi

{
  echo "claim_version=$CLAIM_VERSION"
  echo "claim_contract_hash=$CLAIM_CONTRACT_HASH"
  echo "timestamp=$(date -Iseconds)"
  echo "git_commit=$(git -C "$ROOT" rev-parse HEAD)"
  echo "git_branch=$(git -C "$ROOT" rev-parse --abbrev-ref HEAD)"
  echo "git_dirty=$([ -n "$git_dirty" ] && echo "true" || echo "false")"
  echo "results_dir=$RESULTS_DIR"
  echo "dataset=$DATASET_PATH"
  echo "dataset_build_elapsed_s=$dataset_build_elapsed_s"
  echo "artifact_export_elapsed_s=$artifact_export_elapsed_s"
  echo "coreml_gen_elapsed_s=$coreml_gen_elapsed_s"
  echo "harness_elapsed_s=$harness_elapsed_s"
  echo "dataset_sha256=$(shasum -a 256 "$DATASET_PATH" | awk '{print $1}')"
  dataset_bytes="$(wc -c < "$DATASET_PATH" | tr -d ' ')"
  echo "dataset_bytes=$dataset_bytes"
  if (( dataset_bytes % 2 != 0 )); then
    echo "WARNING: dataset file has odd byte count ($dataset_bytes) — expected uint16 encoding"
  fi
  if (( dataset_bytes > MAX_CORPUS_BYTES * 2 )); then
    echo "WARNING: dataset_bytes ($dataset_bytes) exceeds 2x MAX_CORPUS_BYTES ($MAX_CORPUS_BYTES) — tokenization may have expanded beyond expected bounds"
  fi
  echo "dataset_tokens=$((dataset_bytes / 2))"
  echo "max_corpus_bytes=$MAX_CORPUS_BYTES"
  echo "artifact_prefix=$ARTIFACT_PREFIX"
  echo "artifact_manifest_sha256=$(shasum -a 256 "$ARTIFACT_PREFIX.manifest.json" | awk '{print $1}')"
  echo "offline_gate_json=$OFFLINE_GATE_JSON"
  echo "offline_gate_sha256=$(shasum -a 256 "$OFFLINE_GATE_JSON" | awk '{print $1}')"
  echo "coreml_model=$COREML_MODEL"
  if [[ -d "$COREML_MODEL" ]]; then
    echo "coreml_model_file_count=$(find "$COREML_MODEL" -type f | wc -l | tr -d ' ')"
    echo "coreml_model_total_bytes=$(find "$COREML_MODEL" -type f -exec stat -f%z {} + 2>/dev/null | awk '{s+=$1}END{print s+0}' || echo unknown)"
  fi
  echo "coremltools_python_version=$("$COREMLTOOLS_PYTHON" --version 2>/dev/null || echo unknown)"
  echo "coremltools_version=$("$COREMLTOOLS_PYTHON" -c 'import coremltools; print(coremltools.__version__)' 2>/dev/null || echo unknown)"
  if [[ -d "$COREML_MODEL" ]]; then
    echo "coreml_model_sha256=$(find "$COREML_MODEL" -type f | sort | xargs shasum -a 256 | shasum -a 256 | awk '{print $1}')"
  elif [[ -f "$COREML_MODEL" ]]; then
    echo "coreml_model_sha256=$(shasum -a 256 "$COREML_MODEL" | awk '{print $1}')"
  fi
  echo "prompt_token=$PROMPT_TOKEN"
  echo "recurrent_checkpoint_sha256=$(shasum -a 256 "$ARTIFACT_PREFIX.recurrent.bin" | awk '{print $1}')"
  echo "recurrent_checkpoint_bytes=$(wc -c < "$ARTIFACT_PREFIX.recurrent.bin" | tr -d ' ')"
  echo "future_sidecar_sha256=$(shasum -a 256 "$ARTIFACT_PREFIX.future-sidecar.bin" | awk '{print $1}')"
  echo "future_sidecar_bytes=$(wc -c < "$ARTIFACT_PREFIX.future-sidecar.bin" | tr -d ' ')"
  echo "generation_model_sha256=$(shasum -a 256 "$ARTIFACT_PREFIX.generation.bin" | awk '{print $1}')"
  echo "generation_model_bytes=$(wc -c < "$ARTIFACT_PREFIX.generation.bin" | tr -d ' ')"
  echo "manifest_bytes=$(wc -c < "$ARTIFACT_PREFIX.manifest.json" | tr -d ' ')"
  echo "offline_gate_bytes=$(wc -c < "$OFFLINE_GATE_JSON" | tr -d ' ')"
  echo "offline_committed_exact_tokens_per_pass=$(jq -r '.committed_exact_tokens_per_pass' "$OFFLINE_GATE_JSON")"
  echo "offline_accepted_future_tokens_per_pass=$(jq -r '.accepted_future_tokens_per_pass' "$OFFLINE_GATE_JSON")"
  echo "offline_parity_status=$(jq -r '.parity_status' "$OFFLINE_GATE_JSON")"
  echo "control_backend=$CONTROL_BACKEND"
  echo "two_step_backend=$TWO_STEP_BACKEND"
  echo "output_head_backend=$OUTPUT_HEAD_BACKEND"
  echo "public_summary=$PUBLIC_RESULTS_DIR/summary.txt"
  echo "public_summary_json=$PUBLIC_RESULTS_DIR/summary.json"
  if [[ -f "$PUBLIC_RESULTS_DIR/summary.json" ]]; then
    echo "summary_json_sha256=$(shasum -a 256 "$PUBLIC_RESULTS_DIR/summary.json" | awk '{print $1}')"
  fi
  # Propagate key metrics from inner harness summary if available
  # Extract all harness metrics from summary.json in a single jq call
  if [[ -f "$PUBLIC_RESULTS_DIR/summary.json" ]]; then
    if ! jq -r '
      def na: . // "n/a";
      "harness_version=\(.harness_version | na)",
      "harness_git_commit=\(.git_commit | na)",
      "harness_git_branch=\(.git_branch | na)",
      "harness_timestamp=\(.timestamp | na)",
      "harness_probe_version=\(.probe_version | na)",
      "harness_two_step_median_ms=\(.two_step.median_ms_per_token | na)",
      "harness_two_step_min_ms=\(.two_step.min_ms_per_token | na)",
      "harness_two_step_max_ms=\(.two_step.max_ms_per_token | na)",
      "harness_two_step_p95_ms=\(.two_step.p95_ms_per_token | na)",
      "harness_two_step_p99_ms=\(.two_step.p99_ms_per_token | na)",
      "harness_coreml_median_ms=\(.coreml.median_ms_per_token | na)",
      "harness_coreml_min_ms=\(.coreml.min_ms_per_token | na)",
      "harness_coreml_max_ms=\(.coreml.max_ms_per_token | na)",
      "harness_coreml_p95_ms=\(.coreml.p95_ms_per_token | na)",
      "harness_coreml_p99_ms=\(.coreml.p99_ms_per_token | na)",
      "harness_speedup_median=\(.two_step_speedup_vs_coreml | na)",
      "harness_speedup_min=\(.two_step_speedup_min | na)",
      "harness_speedup_max=\(.two_step_speedup_max | na)",
      "harness_all_parity=\(.all_parity_match | na)",
      "harness_parity_total=\(.parity_total | na)",
      "harness_per_run_parity=\(.per_run_parity // [] | tojson)",
      "harness_parity_match_counts=\(.per_run_parity_match_count // [] | tojson)",
      "harness_two_step_cv=\(.two_step.cv | na)",
      "harness_control_median_ms=\(.control.median_ms_per_token | na)",
      "harness_control_min_ms=\(.control.min_ms_per_token | na)",
      "harness_control_max_ms=\(.control.max_ms_per_token | na)",
      "harness_control_p95_ms=\(.control.p95_ms_per_token | na)",
      "harness_control_p99_ms=\(.control.p99_ms_per_token | na)",
      "harness_control_cv=\(.control.cv | na)",
      "harness_coreml_cv=\(.coreml.cv | na)",
      "harness_speedup_cv=\(.two_step_speedup_cv | na)",
      "harness_two_step_mean_ms=\(.two_step.mean_ms_per_token | na)",
      "harness_two_step_stddev_ms=\(.two_step.stddev_ms_per_token | na)",
      "harness_two_step_iqr_ms=\(.two_step.iqr_ms | na)",
      "harness_two_step_proposer_ms=\(.two_step.breakdown.proposer_ms_per_pass | na)",
      "harness_two_step_verifier_trunk_ms=\(.two_step.breakdown.verifier_trunk_ms_per_pass | na)",
      "harness_two_step_verifier_logits_ms=\(.two_step.breakdown.verifier_logits_ms_per_pass | na)",
      "harness_two_step_state_advance_ms=\(.two_step.breakdown.state_advance_ms_per_pass | na)",
      "harness_control_mean_ms=\(.control.mean_ms_per_token | na)",
      "harness_control_stddev_ms=\(.control.stddev_ms_per_token | na)",
      "harness_control_iqr_ms=\(.control.iqr_ms | na)",
      "harness_control_trunk_ms=\(.control.breakdown.trunk_ms_per_token | na)",
      "harness_control_logits_ms=\(.control.breakdown.logits_ms_per_token | na)",
      "harness_coreml_mean_ms=\(.coreml.mean_ms_per_token | na)",
      "harness_coreml_stddev_ms=\(.coreml.stddev_ms_per_token | na)",
      "harness_coreml_iqr_ms=\(.coreml.iqr_ms | na)",
      "harness_coreml_trunk_ms=\(.coreml.breakdown.trunk_ms_per_token | na)",
      "harness_coreml_logits_ms=\(.coreml.breakdown.logits_ms_per_token | na)",
      "harness_speedup_mean=\(.two_step_speedup_mean | na)",
      "harness_speedup_stddev=\(.two_step_speedup_stddev | na)",
      "harness_speedup_iqr=\(.two_step_speedup_iqr | na)",
      "harness_control_speedup=\(.control_speedup_vs_coreml | na)",
      "harness_control_speedup_min=\(.control_speedup_min | na)",
      "harness_control_speedup_max=\(.control_speedup_max | na)",
      "harness_control_speedup_cv=\(.control_speedup_cv | na)",
      "harness_control_speedup_iqr=\(.control_speedup_iqr | na)",
      "harness_per_run_speedups=\(.per_run_speedups // [] | tojson)",
      "harness_per_run_control_speedups=\(.per_run_control_speedups // [] | tojson)",
      "harness_per_run_two_step_medians_ms=\(.two_step.per_run_medians_ms // [] | tojson)",
      "harness_per_run_two_step_p95s_ms=\(.two_step.per_run_p95s_ms // [] | tojson)",
      "harness_per_run_control_medians_ms=\(.control.per_run_medians_ms // [] | tojson)",
      "harness_per_run_control_p95s_ms=\(.control.per_run_p95s_ms // [] | tojson)",
      "harness_per_run_coreml_medians_ms=\(.coreml.per_run_medians_ms // [] | tojson)",
      "harness_per_run_coreml_p95s_ms=\(.coreml.per_run_p95s_ms // [] | tojson)",
      "harness_total_elapsed_s=\(.total_elapsed_s | na)",
      "harness_probe_build_elapsed_s=\(.probe_build_elapsed_s | na)",
      "harness_requested_repeats=\(.requested_repeats | na)",
      "harness_valid_runs=\(.valid_runs | na)",
      "harness_failed_runs=\(.failed_runs // 0)",
      "harness_valid_run_files=\(.valid_run_files // [] | tojson)",
      "harness_metadata_sha256=\(.artifact_hashes.metadata_sha256 | na)",
      "harness_probe_sha256=\(.artifact_hashes.probe_sha256 | na)",
      "harness_coreml_model_sha256=\(.artifact_hashes.coreml_model_sha256 | na)",
      "harness_recurrent_sha256=\(.artifact_hashes.recurrent_checkpoint_sha256 | na)",
      "harness_sidecar_sha256=\(.artifact_hashes.future_sidecar_sha256 | na)",
      "harness_generation_sha256=\(.artifact_hashes.generation_model_sha256 | na)",
      "harness_hostname=\(.per_run_hostnames // [] | map(select(. != null)) | unique | join(",") | if . == "" then "n/a" else . end)",
      "harness_hw_model=\(.host.hw_model | na)",
      "harness_chip=\(.host.chip | na)",
      "harness_macos_version=\(.host.macos_version | na)",
      "harness_macos_build=\(.host.macos_build | na)",
      "harness_thermal_pressure=\(.host.thermal_pressure | na)",
      "harness_thermal_pressure_end=\(.host.thermal_pressure_end | na)",
      "harness_load_average=\(.host.load_average | na)",
      "harness_load_average_end=\(.host.load_average_end | na)",
      "harness_power_source=\(.host.power_source | na)",
      "harness_power_source_end=\(.host.power_source_end | na)",
      "harness_swift_version=\(.toolchain.swift_version | na)",
      "harness_jq_version=\(.toolchain.jq_version | na)",
      "harness_ncpu=\(.host.ncpu | na)",
      "harness_physical_memory_gb=\(.host.physical_memory_gb | na)",
      "harness_memory_free_pct=\(.host.memory_free_pct | na)",
      "harness_memory_free_pct_end=\(.host.memory_free_pct_end | na)",
      "harness_disk_free_mb_start=\(.host.disk_free_mb_start | na)",
      "harness_disk_free_mb_end=\(.host.disk_free_mb_end | na)",
      "harness_control_init_wall_ms=\(.init_times.control_init_wall_ms | na)",
      "harness_two_step_init_wall_ms=\(.init_times.two_step_init_wall_ms | na)",
      "harness_coreml_compile_ms=\(.init_times.coreml_compile_ms | na)",
      "harness_control_reported_compile_ms=\(.init_times.control_reported_compile_ms | na)",
      "harness_two_step_reported_compile_ms=\(.init_times.two_step_reported_compile_ms | na)",
      "harness_control_init_wall_min_ms=\(.init_times.control_init_wall_min_ms | na)",
      "harness_control_init_wall_max_ms=\(.init_times.control_init_wall_max_ms | na)",
      "harness_two_step_init_wall_min_ms=\(.init_times.two_step_init_wall_min_ms | na)",
      "harness_two_step_init_wall_max_ms=\(.init_times.two_step_init_wall_max_ms | na)",
      "harness_coreml_compile_min_ms=\(.init_times.coreml_compile_min_ms | na)",
      "harness_coreml_compile_max_ms=\(.init_times.coreml_compile_max_ms | na)",
      "harness_per_run_control_init_ms=\(.init_times.per_run_control_init_wall_ms // [] | tojson)",
      "harness_per_run_two_step_init_ms=\(.init_times.per_run_two_step_init_wall_ms // [] | tojson)",
      "harness_per_run_coreml_compile_ms=\(.init_times.per_run_coreml_compile_ms // [] | tojson)",
      "harness_committed_tokens_per_pass=\(.token_accounting.committed_exact_tokens_per_pass | na)",
      "harness_accepted_future_tokens_per_pass=\(.token_accounting.accepted_future_tokens_per_pass | na)",
      "harness_committed_per_run=\(.token_accounting.committed_per_run // [] | tojson)",
      "harness_accepted_per_run=\(.token_accounting.accepted_per_run // [] | tojson)",
      "harness_per_run_input_modes=\(.per_run_input_modes // [] | tojson)",
      "harness_per_run_layer_counts=\(.per_run_layer_counts // [] | tojson)",
      "harness_per_run_build_configurations=\(.per_run_build_configurations // [] | tojson)",
      "harness_per_run_os_versions=\(.per_run_os_versions // [] | tojson)",
      "harness_per_run_process_ids=\(.per_run_process_ids // [] | tojson)",
      "harness_per_run_iteration_counts=\(.per_run_iteration_counts // [] | tojson)",
      "harness_per_run_control_generated=\(.per_run_control_generated_count // [] | tojson)",
      "harness_per_run_two_step_generated=\(.per_run_two_step_generated_count // [] | tojson)",
      "harness_per_run_stderr_lines=\(.per_run_stderr_lines // [] | tojson)",
      "harness_total_stderr_lines=\(.per_run_stderr_lines // [] | add // 0)",
      "harness_per_run_two_step_proposer_ms=\(.two_step.breakdown.per_run_proposer_ms // [] | tojson)",
      "harness_per_run_two_step_verifier_trunk_ms=\(.two_step.breakdown.per_run_verifier_trunk_ms // [] | tojson)",
      "harness_per_run_two_step_verifier_logits_ms=\(.two_step.breakdown.per_run_verifier_logits_ms // [] | tojson)",
      "harness_per_run_control_trunk_ms=\(.control.breakdown.per_run_trunk_ms // [] | tojson)",
      "harness_per_run_control_logits_ms=\(.control.breakdown.per_run_logits_ms // [] | tojson)",
      "harness_per_run_two_step_state_advance_ms=\(.two_step.breakdown.per_run_state_advance_ms // [] | tojson)",
      "harness_per_run_coreml_trunk_ms=\(.coreml.breakdown.per_run_trunk_ms // [] | tojson)",
      "harness_per_run_coreml_logits_ms=\(.coreml.breakdown.per_run_logits_ms // [] | tojson)",
      "harness_per_run_control_backends=\(.per_run_control_backends // [] | tojson)",
      "harness_per_run_two_step_backends=\(.per_run_two_step_backends // [] | tojson)",
      "harness_per_run_output_head_backends=\(.per_run_output_head_backends // [] | tojson)",
      "harness_per_run_prompt_tokens=\(.per_run_prompt_tokens // [] | tojson)",
      "harness_per_run_max_new_tokens=\(.per_run_max_new_tokens // [] | tojson)",
      "harness_per_run_max_sequence_tokens=\(.per_run_max_sequence_tokens // [] | tojson)",
      "harness_per_run_warmup=\(.per_run_warmup // [] | tojson)",
      "harness_per_run_iterations=\(.per_run_iterations // [] | tojson)",
      "harness_per_run_trunk_lane_spatials=\(.per_run_trunk_lane_spatials // [] | tojson)",
      "harness_per_run_output_head_lane_spatials=\(.per_run_output_head_lane_spatials // [] | tojson)",
      "harness_per_run_two_step_p99s_ms=\(.two_step.per_run_p99s_ms // [] | tojson)",
      "harness_per_run_control_p99s_ms=\(.control.per_run_p99s_ms // [] | tojson)",
      "harness_per_run_coreml_p99s_ms=\(.coreml.per_run_p99s_ms // [] | tojson)",
      "harness_per_run_two_step_iter_min_ms=\(.two_step.per_run_iteration_min_ms // [] | tojson)",
      "harness_per_run_two_step_iter_max_ms=\(.two_step.per_run_iteration_max_ms // [] | tojson)",
      "harness_per_run_control_iter_min_ms=\(.control.per_run_iteration_min_ms // [] | tojson)",
      "harness_per_run_control_iter_max_ms=\(.control.per_run_iteration_max_ms // [] | tojson)",
      "harness_per_run_coreml_iter_min_ms=\(.coreml.per_run_iteration_min_ms // [] | tojson)",
      "harness_per_run_coreml_iter_max_ms=\(.coreml.per_run_iteration_max_ms // [] | tojson)",
      "harness_contract_hash=\(.benchmark_contract.contract_hash | na)",
      "harness_contract_input_mode=\(.benchmark_contract.input_mode | na)",
      "harness_contract_warmup=\(.benchmark_contract.warmup | na)",
      "harness_contract_iterations=\(.benchmark_contract.iterations | na)",
      "harness_contract_layer_count=\(.benchmark_contract.layer_count | na)",
      "harness_contract_control_backend=\(.benchmark_contract.control_backend | na)",
      "harness_contract_two_step_backend=\(.benchmark_contract.two_step_backend | na)",
      "harness_contract_output_head_backend=\(.benchmark_contract.output_head_backend | na)",
      "harness_contract_max_new_tokens=\(.benchmark_contract.max_new_tokens | na)",
      "harness_contract_max_sequence_tokens=\(.benchmark_contract.max_sequence_tokens | na)",
      "harness_contract_prompt_token=\(.benchmark_contract.prompt_token | na)",
      "harness_contract_cv_threshold=\(.benchmark_contract.cv_threshold | na)",
      "harness_contract_duration_budget_s=\(.benchmark_contract.duration_budget_s | na)",
      "harness_per_run_timestamps=\(.per_run_timestamps // [] | tojson)",
      "harness_per_run_probe_versions=\(.per_run_probe_versions // [] | tojson)",
      "harness_first_run_timestamp=\(.first_run_timestamp | na)",
      "harness_last_run_timestamp=\(.last_run_timestamp | na)",
      "harness_per_run_wall_elapsed_s=\(.per_run_wall_elapsed_s // [] | tojson)",
      "harness_per_run_outer_elapsed_s=\(.per_run_outer_elapsed_s // [] | tojson)",
      "harness_sum_probe_wall_elapsed_s=\(.sum_probe_wall_elapsed_s | na)",
      "harness_sum_outer_elapsed_s=\(.sum_outer_elapsed_s | na)",
      "harness_probe_wall_range_s=\(.probe_wall_range_s | na)",
      "harness_outer_elapsed_range_s=\(.outer_elapsed_range_s | na)",
      "harness_gate_status=\(.reproducibility.gate_status // "unknown")",
      "harness_outlier_count=\(.reproducibility.outlier_count // 0)",
      "harness_outlier_two_step=\(.reproducibility.outlier_detail.two_step.count // 0)",
      "harness_outlier_control=\(.reproducibility.outlier_detail.control.count // 0)",
      "harness_outlier_coreml=\(.reproducibility.outlier_detail.coreml.count // 0)",
      "harness_outlier_speedup=\(.reproducibility.outlier_detail.speedup.count // 0)",
      "harness_gate_warnings=\(.reproducibility.warnings // [] | tojson)"
    ' "$PUBLIC_RESULTS_DIR/summary.json"; then
      echo "WARNING: jq extraction from summary.json failed — harness metrics may be incomplete"
    fi
  elif [[ -f "$PUBLIC_RESULTS_DIR/summary.txt" ]]; then
    gate_line="$(grep '^gate_status=' "$PUBLIC_RESULTS_DIR/summary.txt" || true)"
    if [[ -n "$gate_line" ]]; then
      echo "harness_$gate_line"
    fi
  fi
  # Cross-validate offline gate parity with harness parity
  if [[ -f "$PUBLIC_RESULTS_DIR/summary.json" ]]; then
    harness_parity="$(jq -r '.all_parity_match' "$PUBLIC_RESULTS_DIR/summary.json" 2>/dev/null || echo "unknown")"
    offline_parity="$(jq -r '.parity_status' "$OFFLINE_GATE_JSON" 2>/dev/null || echo "unknown")"
    if [[ "$offline_parity" == "match" && "$harness_parity" != "true" ]]; then
      echo "WARNING: offline gate shows parity=match but harness shows all_parity_match=$harness_parity"
    fi
    # Cross-validate token accounting
    offline_committed="$(jq -r '.committed_exact_tokens_per_pass' "$OFFLINE_GATE_JSON" 2>/dev/null || echo "")"
    harness_committed="$(jq -r '.token_accounting.committed_exact_tokens_per_pass' "$PUBLIC_RESULTS_DIR/summary.json" 2>/dev/null || echo "")"
    if [[ -n "$offline_committed" && -n "$harness_committed" && "$offline_committed" != "$harness_committed" ]]; then
      echo "WARNING: token accounting mismatch: offline committed=$offline_committed harness committed=$harness_committed"
    fi
    offline_accepted="$(jq -r '.accepted_future_tokens_per_pass' "$OFFLINE_GATE_JSON" 2>/dev/null || echo "")"
    harness_accepted="$(jq -r '.token_accounting.accepted_future_tokens_per_pass' "$PUBLIC_RESULTS_DIR/summary.json" 2>/dev/null || echo "")"
    if [[ -n "$offline_accepted" && -n "$harness_accepted" && "$offline_accepted" != "$harness_accepted" ]]; then
      echo "WARNING: token accounting mismatch: offline accepted_future=$offline_accepted harness accepted_future=$harness_accepted"
    fi
    # Cross-validate claim-level contract params match harness contract
    harness_contract_layer="$(jq -r '.benchmark_contract.layer_count // empty' "$PUBLIC_RESULTS_DIR/summary.json" 2>/dev/null || true)"
    if [[ -n "$harness_contract_layer" && "$harness_contract_layer" != "$LAYER_COUNT" ]]; then
      echo "WARNING: claim LAYER_COUNT=$LAYER_COUNT but harness contract layer_count=$harness_contract_layer"
    fi
    harness_contract_max_new="$(jq -r '.benchmark_contract.max_new_tokens // empty' "$PUBLIC_RESULTS_DIR/summary.json" 2>/dev/null || true)"
    if [[ -n "$harness_contract_max_new" && "$harness_contract_max_new" != "$MAX_NEW_TOKENS" ]]; then
      echo "WARNING: claim MAX_NEW_TOKENS=$MAX_NEW_TOKENS but harness contract max_new_tokens=$harness_contract_max_new"
    fi
    harness_contract_max_seq="$(jq -r '.benchmark_contract.max_sequence_tokens // empty' "$PUBLIC_RESULTS_DIR/summary.json" 2>/dev/null || true)"
    if [[ -n "$harness_contract_max_seq" && "$harness_contract_max_seq" != "$MAX_SEQUENCE_TOKENS" ]]; then
      echo "WARNING: claim MAX_SEQUENCE_TOKENS=$MAX_SEQUENCE_TOKENS but harness contract max_sequence_tokens=$harness_contract_max_seq"
    fi
    harness_contract_warmup="$(jq -r '.benchmark_contract.warmup // empty' "$PUBLIC_RESULTS_DIR/summary.json" 2>/dev/null || true)"
    if [[ -n "$harness_contract_warmup" && "$harness_contract_warmup" != "$WARMUP" ]]; then
      echo "WARNING: claim WARMUP=$WARMUP but harness contract warmup=$harness_contract_warmup"
    fi
    harness_contract_iters="$(jq -r '.benchmark_contract.iterations // empty' "$PUBLIC_RESULTS_DIR/summary.json" 2>/dev/null || true)"
    if [[ -n "$harness_contract_iters" && "$harness_contract_iters" != "$ITERATIONS" ]]; then
      echo "WARNING: claim ITERATIONS=$ITERATIONS but harness contract iterations=$harness_contract_iters"
    fi
    harness_contract_ctrl="$(jq -r '.benchmark_contract.control_backend // empty' "$PUBLIC_RESULTS_DIR/summary.json" 2>/dev/null || true)"
    if [[ -n "$harness_contract_ctrl" && "$harness_contract_ctrl" != "$CONTROL_BACKEND" ]]; then
      echo "WARNING: claim CONTROL_BACKEND=$CONTROL_BACKEND but harness contract control_backend=$harness_contract_ctrl"
    fi
    harness_contract_two="$(jq -r '.benchmark_contract.two_step_backend // empty' "$PUBLIC_RESULTS_DIR/summary.json" 2>/dev/null || true)"
    if [[ -n "$harness_contract_two" && "$harness_contract_two" != "$TWO_STEP_BACKEND" ]]; then
      echo "WARNING: claim TWO_STEP_BACKEND=$TWO_STEP_BACKEND but harness contract two_step_backend=$harness_contract_two"
    fi
    harness_contract_head="$(jq -r '.benchmark_contract.output_head_backend // empty' "$PUBLIC_RESULTS_DIR/summary.json" 2>/dev/null || true)"
    if [[ -n "$harness_contract_head" && "$harness_contract_head" != "$OUTPUT_HEAD_BACKEND" ]]; then
      echo "WARNING: claim OUTPUT_HEAD_BACKEND=$OUTPUT_HEAD_BACKEND but harness contract output_head_backend=$harness_contract_head"
    fi
    harness_contract_prompt="$(jq -r '.benchmark_contract.prompt_token // empty' "$PUBLIC_RESULTS_DIR/summary.json" 2>/dev/null || true)"
    if [[ -n "$harness_contract_prompt" && "$harness_contract_prompt" != "$PROMPT_TOKEN" ]]; then
      echo "WARNING: claim PROMPT_TOKEN=$PROMPT_TOKEN but harness contract prompt_token=$harness_contract_prompt"
    fi
    harness_contract_input="$(jq -r '.benchmark_contract.input_mode // empty' "$PUBLIC_RESULTS_DIR/summary.json" 2>/dev/null || true)"
    if [[ -n "$harness_contract_input" && "$harness_contract_input" != "recurrent-checkpoint" ]]; then
      echo "WARNING: expected input_mode=recurrent-checkpoint but harness used input_mode=$harness_contract_input"
    fi
    # Cross-validate contract hash: claim and harness must agree on the full parameter set
    harness_hash="$(jq -r '.benchmark_contract.contract_hash // empty' "$PUBLIC_RESULTS_DIR/summary.json" 2>/dev/null || true)"
    if [[ -n "$harness_hash" && "$harness_hash" != "$CLAIM_CONTRACT_HASH" ]]; then
      echo "WARNING: contract hash mismatch: claim=$CLAIM_CONTRACT_HASH harness=$harness_hash"
    fi
    # Cross-validate valid run count against requested repeats
    harness_valid="$(jq -r '.valid_runs // empty' "$PUBLIC_RESULTS_DIR/summary.json" 2>/dev/null || true)"
    if [[ -n "$harness_valid" && "$harness_valid" != "$REPEATS" ]]; then
      echo "WARNING: harness completed $harness_valid valid runs out of $REPEATS requested"
    fi
    # Cross-validate git commit: claim and harness should be on the same commit
    harness_git="$(jq -r '.git_commit // empty' "$PUBLIC_RESULTS_DIR/summary.json" 2>/dev/null || true)"
    if [[ -n "$harness_git" && "$harness_git" != "$git_commit_start" ]]; then
      echo "WARNING: claim git_commit=$git_commit_start but harness git_commit=$harness_git"
    fi
    # Cross-validate hostname: claim pipeline must run on same host as harness
    claim_hostname="$(hostname 2>/dev/null || echo "")"
    harness_hostnames="$(jq -r '.per_run_hostnames // [] | map(select(. != null)) | unique | join(",")' "$PUBLIC_RESULTS_DIR/summary.json" 2>/dev/null || true)"
    if [[ -n "$claim_hostname" && -n "$harness_hostnames" && "$harness_hostnames" != *"$claim_hostname"* ]]; then
      echo "WARNING: claim hostname=$claim_hostname not found in harness hostnames=$harness_hostnames"
    fi
    # Cross-validate artifact hashes: claim exports vs harness inputs
    claim_recurrent_sha="$(shasum -a 256 "$ARTIFACT_PREFIX.recurrent.bin" 2>/dev/null | awk '{print $1}')"
    harness_recurrent_sha="$(jq -r '.artifact_hashes.recurrent_checkpoint_sha256 // empty' "$PUBLIC_RESULTS_DIR/summary.json" 2>/dev/null || true)"
    if [[ -n "$claim_recurrent_sha" && -n "$harness_recurrent_sha" && "$claim_recurrent_sha" != "$harness_recurrent_sha" ]]; then
      echo "WARNING: recurrent checkpoint SHA mismatch: claim=$claim_recurrent_sha harness=$harness_recurrent_sha"
    fi
    claim_sidecar_sha="$(shasum -a 256 "$ARTIFACT_PREFIX.future-sidecar.bin" 2>/dev/null | awk '{print $1}')"
    harness_sidecar_sha="$(jq -r '.artifact_hashes.future_sidecar_sha256 // empty' "$PUBLIC_RESULTS_DIR/summary.json" 2>/dev/null || true)"
    if [[ -n "$claim_sidecar_sha" && -n "$harness_sidecar_sha" && "$claim_sidecar_sha" != "$harness_sidecar_sha" ]]; then
      echo "WARNING: future sidecar SHA mismatch: claim=$claim_sidecar_sha harness=$harness_sidecar_sha"
    fi
    claim_gen_sha="$(shasum -a 256 "$ARTIFACT_PREFIX.generation.bin" 2>/dev/null | awk '{print $1}')"
    harness_gen_sha="$(jq -r '.artifact_hashes.generation_model_sha256 // empty' "$PUBLIC_RESULTS_DIR/summary.json" 2>/dev/null || true)"
    if [[ -n "$claim_gen_sha" && -n "$harness_gen_sha" && "$claim_gen_sha" != "$harness_gen_sha" ]]; then
      echo "WARNING: generation model SHA mismatch: claim=$claim_gen_sha harness=$harness_gen_sha"
    fi
  fi
  # Validate that all critical artifact files still exist and are non-empty
  missing_artifacts=""
  for artifact_file in \
    "$DATASET_PATH" \
    "$ARTIFACT_PREFIX.recurrent.bin" \
    "$ARTIFACT_PREFIX.future-sidecar.bin" \
    "$ARTIFACT_PREFIX.generation.bin" \
    "$ARTIFACT_PREFIX.manifest.json" \
    "$OFFLINE_GATE_JSON"; do
    if [[ ! -s "$artifact_file" ]]; then
      missing_artifacts="${missing_artifacts}$(basename "$artifact_file") "
    fi
  done
  if [[ -n "$missing_artifacts" ]]; then
    echo "WARNING: artifact file(s) missing or empty at claim time: $missing_artifacts"
  fi
  echo "disk_free_mb_end=$(df -m "$RESULTS_DIR" 2>/dev/null | awk 'NR==2{print $4}' || echo unknown)"
  echo "memory_free_pct_end=$(sysctl -n kern.memorystatus_level 2>/dev/null || echo unknown)"
  echo "thermal_pressure_end=$(pmset -g therm 2>/dev/null | grep -i 'cpu.*speed' | head -1 || echo unknown)"
  echo "load_average_end=$(sysctl -n vm.loadavg 2>/dev/null || echo unknown)"
  power_source_end="$(pmset -g batt 2>/dev/null | head -1 | sed "s/.*'\(.*\)'.*/\1/" || echo unknown)"
  echo "power_source_end=$power_source_end"
  # Claim-level environment drift warnings
  mem_end="$(sysctl -n kern.memorystatus_level 2>/dev/null || echo 100)"
  if [[ "$mem_end" -lt 20 ]] 2>/dev/null; then
    echo "CLAIM_WARNING: LOW_MEMORY — memory_free_pct=$mem_end at claim end"
  fi
  disk_end="$(df -m "$RESULTS_DIR" 2>/dev/null | awk 'NR==2{print $4}' || echo 0)"
  if [[ "$disk_end" -lt 512 ]] 2>/dev/null; then
    echo "CLAIM_WARNING: LOW_DISK_SPACE — only ${disk_end}MB free at claim end"
  fi
  if [[ "$power_source_start" != "$power_source_end" && "$power_source_start" != "unknown" && "$power_source_end" != "unknown" ]]; then
    echo "CLAIM_WARNING: POWER_DRIFT — power source changed during claim (start='$power_source_start', end='$power_source_end')"
  fi
  claim_elapsed_s=$(( $(date +%s) - claim_start_epoch ))
  echo "claim_total_elapsed_s=$claim_elapsed_s"
  pipeline_sum_s=$((dataset_build_elapsed_s + artifact_export_elapsed_s + coreml_gen_elapsed_s + harness_elapsed_s))
  echo "pipeline_sum_s=$pipeline_sum_s"
  echo "pipeline_overhead_s=$((claim_elapsed_s - pipeline_sum_s))"
  git_commit_end="$(git -C "$ROOT" rev-parse HEAD)"
  echo "git_commit_end=$git_commit_end"
  if [[ "$git_commit_end" != "$git_commit_start" ]]; then
    echo "WARNING: git HEAD changed during claim run (start=$git_commit_start end=$git_commit_end)"
  fi
  echo "harness_exit_code=$harness_exit"
} | tee "$RESULTS_DIR/claim-summary.txt"

# Append warning count to claim-summary
claim_warnings="$(grep -c '^WARNING:\|^CLAIM_WARNING:' "$RESULTS_DIR/claim-summary.txt" 2>/dev/null || echo 0)"
echo "claim_warning_count=$claim_warnings" | tee -a "$RESULTS_DIR/claim-summary.txt"

# Propagate inner harness gate exit code (2 = gate fail)
exit "$harness_exit"
