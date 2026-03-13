#!/usr/bin/env bash
set -euo pipefail

# Bump this when the harness summary.json contract changes (new fields, renamed keys, etc.).
HARNESS_VERSION=7

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRATCH_PATH="${SCRATCH_PATH:-/tmp/espresso-ane-multitoken-release}"
RESULTS_DIR="${RESULTS_DIR:-$ROOT/results/exact-4x-$(date +%Y%m%d-%H%M%S)}"
PROBE="$SCRATCH_PATH/release/espresso-multitoken-probe"

REPEATS="${REPEATS:-5}"
WARMUP="${WARMUP:-3}"
ITERATIONS="${ITERATIONS:-20}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8}"
MAX_SEQUENCE_TOKENS="${MAX_SEQUENCE_TOKENS:-32}"
LAYER_COUNT="${LAYER_COUNT:-6}"
CONTROL_BACKEND="${CONTROL_BACKEND:-fused-triplet}"
TWO_STEP_BACKEND="${TWO_STEP_BACKEND:-fused-triplet}"
OUTPUT_HEAD_BACKEND="${OUTPUT_HEAD_BACKEND:-ane-rmsnorm-classifier}"
INPUT_MODE="${INPUT_MODE:-echo}"
COREML_MODEL="${COREML_MODEL:-$ROOT/benchmarks/models/transformer_6layer.mlpackage}"
RECURRENT_CHECKPOINT="${RECURRENT_CHECKPOINT:-}"
FUTURE_SIDECAR="${FUTURE_SIDECAR:-}"
GENERATION_MODEL="${GENERATION_MODEL:-}"
PROMPT_TOKEN="${PROMPT_TOKEN:-0}"
DRY_RUN="${DRY_RUN:-0}"
CV_THRESHOLD="${CV_THRESHOLD:-0.10}"
DURATION_BUDGET_S="${DURATION_BUDGET_S:-600}"

if [[ "$REPEATS" -lt 3 || $((REPEATS % 2)) -ne 1 ]]; then
  echo "REPEATS must be an odd integer >= 3" >&2
  exit 1
fi

if [[ "$ITERATIONS" -lt 1 ]]; then
  echo "ITERATIONS must be >= 1" >&2
  exit 1
fi

if [[ "$WARMUP" -lt 0 ]]; then
  echo "WARMUP must be >= 0" >&2
  exit 1
fi

if [[ "$LAYER_COUNT" -lt 1 ]]; then
  echo "LAYER_COUNT must be >= 1" >&2
  exit 1
fi

if [[ "$MAX_NEW_TOKENS" -lt 1 ]]; then
  echo "MAX_NEW_TOKENS must be >= 1" >&2
  exit 1
fi

if [[ "$MAX_SEQUENCE_TOKENS" -lt 1 ]]; then
  echo "MAX_SEQUENCE_TOKENS must be >= 1" >&2
  exit 1
fi

if [[ "$MAX_SEQUENCE_TOKENS" -lt $((MAX_NEW_TOKENS + 1)) ]]; then
  echo "MAX_SEQUENCE_TOKENS ($MAX_SEQUENCE_TOKENS) must be >= MAX_NEW_TOKENS + 1 ($((MAX_NEW_TOKENS + 1)))" >&2
  exit 1
fi

if ! [[ "$PROMPT_TOKEN" =~ ^[0-9]+$ ]]; then
  echo "PROMPT_TOKEN must be a non-negative integer (got: $PROMPT_TOKEN)" >&2
  exit 1
fi

case "$CONTROL_BACKEND" in
  single|fused-pair|fused-triplet|identity-zero-trunk) ;;
  *) echo "Unsupported CONTROL_BACKEND=$CONTROL_BACKEND (expected single|fused-pair|fused-triplet|identity-zero-trunk)" >&2; exit 1 ;;
esac

case "$TWO_STEP_BACKEND" in
  single|fused-pair|fused-triplet|identity-zero-trunk) ;;
  *) echo "Unsupported TWO_STEP_BACKEND=$TWO_STEP_BACKEND (expected single|fused-pair|fused-triplet|identity-zero-trunk)" >&2; exit 1 ;;
esac

case "$OUTPUT_HEAD_BACKEND" in
  cpu|ane-classifier|ane-rmsnorm-classifier) ;;
  *) echo "Unsupported OUTPUT_HEAD_BACKEND=$OUTPUT_HEAD_BACKEND (expected cpu|ane-classifier|ane-rmsnorm-classifier)" >&2; exit 1 ;;
esac

if [[ "$CONTROL_BACKEND" == "fused-pair" ]] && (( LAYER_COUNT % 2 != 0 )); then
  echo "fused-pair CONTROL_BACKEND requires even LAYER_COUNT (got $LAYER_COUNT)" >&2
  exit 1
fi
if [[ "$CONTROL_BACKEND" == "fused-triplet" ]] && (( LAYER_COUNT % 3 != 0 )); then
  echo "fused-triplet CONTROL_BACKEND requires LAYER_COUNT divisible by 3 (got $LAYER_COUNT)" >&2
  exit 1
fi
if [[ "$TWO_STEP_BACKEND" == "fused-pair" ]] && (( LAYER_COUNT % 2 != 0 )); then
  echo "fused-pair TWO_STEP_BACKEND requires even LAYER_COUNT (got $LAYER_COUNT)" >&2
  exit 1
fi
if [[ "$TWO_STEP_BACKEND" == "fused-triplet" ]] && (( LAYER_COUNT % 3 != 0 )); then
  echo "fused-triplet TWO_STEP_BACKEND requires LAYER_COUNT divisible by 3 (got $LAYER_COUNT)" >&2
  exit 1
fi

if [[ ! -e "$COREML_MODEL" ]]; then
  echo "CoreML model not found at $COREML_MODEL" >&2
  exit 1
fi

case "$INPUT_MODE" in
  echo)
    ;;
  recurrent-checkpoint)
    if [[ -z "$RECURRENT_CHECKPOINT" ]]; then
      echo "RECURRENT_CHECKPOINT is required when INPUT_MODE=recurrent-checkpoint" >&2
      exit 1
    fi
    if [[ ! -f "$RECURRENT_CHECKPOINT" ]]; then
      echo "Recurrent checkpoint not found at $RECURRENT_CHECKPOINT" >&2
      exit 1
    fi
    if [[ -z "$GENERATION_MODEL" ]]; then
      echo "GENERATION_MODEL is required for CoreML comparison when INPUT_MODE=recurrent-checkpoint" >&2
      exit 1
    fi
    if [[ ! -e "$GENERATION_MODEL" ]]; then
      echo "Generation model not found at $GENERATION_MODEL" >&2
      exit 1
    fi
    if [[ -z "$FUTURE_SIDECAR" ]]; then
      echo "FUTURE_SIDECAR is required when INPUT_MODE=recurrent-checkpoint" >&2
      exit 1
    fi
    if [[ ! -f "$FUTURE_SIDECAR" ]]; then
      echo "Future sidecar not found at $FUTURE_SIDECAR" >&2
      exit 1
    fi
    ;;
  *)
    echo "Unsupported INPUT_MODE=$INPUT_MODE (expected echo|recurrent-checkpoint)" >&2
    exit 1
    ;;
esac

mkdir -p "$RESULTS_DIR"

# Warn if results directory already has run data from a previous invocation
if ls "$RESULTS_DIR"/run-*.json >/dev/null 2>&1; then
  echo "WARNING: results directory $RESULTS_DIR already contains run-*.json files — stale data may contaminate aggregation" >&2
fi

# Capture git HEAD at harness start for drift detection
GIT_COMMIT_START="$(git -C "$ROOT" rev-parse HEAD)"

# Signal trap: mark results as interrupted for regression diagnosis
cleanup_on_interrupt() {
  echo ""
  echo "INTERRUPTED at $(date -Iseconds) (signal received)" | tee "$RESULTS_DIR/INTERRUPTED"
  exit 130
}
trap cleanup_on_interrupt INT TERM

{
  echo "harness_version=$HARNESS_VERSION"
  echo "timestamp=$(date -Iseconds)"
  echo "git_commit=$GIT_COMMIT_START"
  echo "git_branch=$(git -C "$ROOT" rev-parse --abbrev-ref HEAD)"
  echo "git_dirty=$(git -C "$ROOT" status --porcelain 2>/dev/null | head -1 | grep -c . || echo 0)"
  echo "swift_version=$(swift --version | tr '\n' ' ')"
  echo "uname=$(uname -a)"
  echo "input_mode=$INPUT_MODE"
  echo "coreml_model=$COREML_MODEL"
  echo "recurrent_checkpoint=${RECURRENT_CHECKPOINT:-<none>}"
  echo "future_sidecar=${FUTURE_SIDECAR:-<none>}"
  echo "generation_model=${GENERATION_MODEL:-<none>}"
  echo "prompt_token=$PROMPT_TOKEN"
  echo "repeats=$REPEATS warmup=$WARMUP iterations=$ITERATIONS max_new_tokens=$MAX_NEW_TOKENS max_sequence_tokens=$MAX_SEQUENCE_TOKENS layer_count=$LAYER_COUNT"
  if [[ -n "$RECURRENT_CHECKPOINT" ]]; then
    echo "recurrent_checkpoint_sha256=$(shasum -a 256 "$RECURRENT_CHECKPOINT" | awk '{print $1}')"
  fi
  if [[ -n "$FUTURE_SIDECAR" ]]; then
    echo "future_sidecar_sha256=$(shasum -a 256 "$FUTURE_SIDECAR" | awk '{print $1}')"
  fi
  if [[ -n "$GENERATION_MODEL" ]]; then
    echo "generation_model_sha256=$(shasum -a 256 "$GENERATION_MODEL" | awk '{print $1}')"
  fi
  # CoreML model hash (directory — hash all file contents deterministically)
  if [[ -d "$COREML_MODEL" ]]; then
    echo "coreml_model_sha256=$(find "$COREML_MODEL" -type f | sort | xargs shasum -a 256 | shasum -a 256 | awk '{print $1}')"
  elif [[ -f "$COREML_MODEL" ]]; then
    echo "coreml_model_sha256=$(shasum -a 256 "$COREML_MODEL" | awk '{print $1}')"
  fi
  # System environment snapshot for regression diagnosis
  echo "chip=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo unknown)"
  echo "hw_model=$(sysctl -n hw.model 2>/dev/null || echo unknown)"
  echo "physical_memory_gb=$(( $(sysctl -n hw.memsize 2>/dev/null || echo 0) / 1073741824 ))"
  echo "ncpu=$(sysctl -n hw.ncpu 2>/dev/null || echo unknown)"
  echo "thermal_pressure=$(pmset -g therm 2>/dev/null | grep -i 'cpu.*speed' | head -1 || echo unknown)"
  echo "load_average=$(sysctl -n vm.loadavg 2>/dev/null || echo unknown)"
  echo "macos_version=$(sw_vers -productVersion 2>/dev/null || echo unknown)"
  echo "macos_build=$(sw_vers -buildVersion 2>/dev/null || echo unknown)"
  echo "power_source=$(pmset -g batt 2>/dev/null | head -1 | sed "s/.*'\(.*\)'.*/\1/" || echo unknown)"
  echo "memory_free_pct=$(sysctl -n kern.memorystatus_level 2>/dev/null || echo unknown)"
} > "$RESULTS_DIR/metadata.txt"

build_start_epoch=$(date +%s)
echo "Building release probe into $SCRATCH_PATH"
swift build -c release --product espresso-multitoken-probe --scratch-path "$SCRATCH_PATH"
PROBE_BUILD_ELAPSED_S=$(( $(date +%s) - build_start_epoch ))
echo "Probe build completed in ${PROBE_BUILD_ELAPSED_S}s"

if [[ ! -x "$PROBE" ]]; then
  echo "FATAL: Probe binary not found or not executable at $PROBE" >&2
  exit 1
fi

# Record probe binary hash for reproducibility
PROBE_SHA256="$(shasum -a 256 "$PROBE" | awk '{print $1}')"
echo "probe_sha256=$PROBE_SHA256" >> "$RESULTS_DIR/metadata.txt"

METADATA_SHA256="$(shasum -a 256 "$RESULTS_DIR/metadata.txt" | awk '{print $1}')"

# Precompute artifact hashes for summary.json
if [[ -d "$COREML_MODEL" ]]; then
  COREML_MODEL_SHA256="$(find "$COREML_MODEL" -type f | sort | xargs shasum -a 256 | shasum -a 256 | awk '{print $1}')"
elif [[ -f "$COREML_MODEL" ]]; then
  COREML_MODEL_SHA256="$(shasum -a 256 "$COREML_MODEL" | awk '{print $1}')"
else
  COREML_MODEL_SHA256=""
fi
RECURRENT_SHA256=""
if [[ -n "${RECURRENT_CHECKPOINT:-}" && -f "$RECURRENT_CHECKPOINT" ]]; then
  RECURRENT_SHA256="$(shasum -a 256 "$RECURRENT_CHECKPOINT" | awk '{print $1}')"
fi
FUTURE_SIDECAR_SHA256=""
if [[ -n "${FUTURE_SIDECAR:-}" && -f "$FUTURE_SIDECAR" ]]; then
  FUTURE_SIDECAR_SHA256="$(shasum -a 256 "$FUTURE_SIDECAR" | awk '{print $1}')"
fi
GENERATION_MODEL_SHA256=""
if [[ -n "${GENERATION_MODEL:-}" && -e "$GENERATION_MODEL" ]]; then
  GENERATION_MODEL_SHA256="$(shasum -a 256 "$GENERATION_MODEL" | awk '{print $1}')"
fi

# Verify jq is available (required for aggregation)
if ! command -v jq &>/dev/null; then
  echo "FATAL: jq is required but not found on PATH" >&2
  exit 1
fi

# Canonical contract hash for fast cross-run equality comparison
CONTRACT_HASH="$(printf '%s\n' \
  "input_mode=$INPUT_MODE" \
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

if [[ "$DRY_RUN" == "1" ]]; then
  echo "=== Dry Run: All prerequisites validated ==="
  echo "harness_version=$HARNESS_VERSION"
  echo "git_commit=$GIT_COMMIT_START"
  echo "git_branch=$(git -C "$ROOT" rev-parse --abbrev-ref HEAD)"
  echo "contract_hash=$CONTRACT_HASH"
  echo "probe=$PROBE"
  echo "probe_sha256=$PROBE_SHA256"
  echo "probe_build_elapsed_s=$PROBE_BUILD_ELAPSED_S"
  echo "coreml_model=$COREML_MODEL"
  echo "coreml_model_sha256=${COREML_MODEL_SHA256:-n/a}"
  echo "jq=$(command -v jq)"
  echo "jq_version=$(jq --version 2>/dev/null || echo unknown)"
  echo "input_mode=$INPUT_MODE"
  echo "control_backend=$CONTROL_BACKEND"
  echo "two_step_backend=$TWO_STEP_BACKEND"
  echo "output_head_backend=$OUTPUT_HEAD_BACKEND"
  echo "repeats=$REPEATS warmup=$WARMUP iterations=$ITERATIONS"
  echo "layer_count=$LAYER_COUNT max_new_tokens=$MAX_NEW_TOKENS max_sequence_tokens=$MAX_SEQUENCE_TOKENS"
  echo "prompt_token=$PROMPT_TOKEN"
  echo "recurrent_checkpoint=${RECURRENT_CHECKPOINT:-<none>}"
  echo "future_sidecar=${FUTURE_SIDECAR:-<none>}"
  echo "generation_model=${GENERATION_MODEL:-<none>}"
  echo "results_dir=$RESULTS_DIR"
  echo "cv_threshold=$CV_THRESHOLD"
  echo "duration_budget_s=$DURATION_BUDGET_S"
  echo "chip=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo unknown)"
  echo "power_source=$(pmset -g batt 2>/dev/null | head -1 | sed "s/.*'\(.*\)'.*/\1/" || echo unknown)"
  echo "thermal_pressure=$(pmset -g therm 2>/dev/null | grep -i 'cpu.*speed' | head -1 || echo unknown)"
  exit 0
fi

COMMON_ARGS=(
  --mode compare
  --input "$INPUT_MODE"
  --compare-coreml
  --coreml-model "$COREML_MODEL"
  --prompt-token "$PROMPT_TOKEN"
  --warmup "$WARMUP"
  --iterations "$ITERATIONS"
  --max-new-tokens "$MAX_NEW_TOKENS"
  --max-sequence-tokens "$MAX_SEQUENCE_TOKENS"
  --layer-count "$LAYER_COUNT"
  --control-backend "$CONTROL_BACKEND"
  --two-step-backend "$TWO_STEP_BACKEND"
  --output-head-backend "$OUTPUT_HEAD_BACKEND"
)

if [[ "$INPUT_MODE" == "recurrent-checkpoint" ]]; then
  COMMON_ARGS+=(--recurrent-checkpoint "$RECURRENT_CHECKPOINT")
  COMMON_ARGS+=(--future-sidecar "$FUTURE_SIDECAR")
fi

if [[ -n "$GENERATION_MODEL" ]]; then
  COMMON_ARGS+=(--generation-model "$GENERATION_MODEL")
fi

# Capture pre-benchmark thermal, load, and disk for drift comparison
THERMAL_START="$(pmset -g therm 2>/dev/null | grep -i 'cpu.*speed' | head -1 || echo unknown)"
LOAD_START="$(sysctl -n vm.loadavg 2>/dev/null || echo unknown)"
DISK_FREE_MB_START="$(df -m "$RESULTS_DIR" 2>/dev/null | awk 'NR==2{print $4}' || echo 0)"

failed_runs=0
benchmark_start_epoch=$(date +%s)
for run in $(seq 1 "$REPEATS"); do
  echo "Run $run/$REPEATS"
  run_start=$(date +%s)
  run_exit=0
  "$PROBE" "${COMMON_ARGS[@]}" \
    > "$RESULTS_DIR/run-$run.json" \
    2> "$RESULTS_DIR/run-$run.stderr.log" || run_exit=$?
  run_elapsed=$(( $(date +%s) - run_start ))
  stderr_lines=$(wc -l < "$RESULTS_DIR/run-$run.stderr.log" | tr -d ' ')
  echo "  elapsed: ${run_elapsed}s (exit=$run_exit, stderr=$stderr_lines lines)"
  echo "$run_elapsed" > "$RESULTS_DIR/run-$run.elapsed_s"
  if [[ $run_exit -ne 0 ]]; then
    echo "WARNING: Run $run exited with code $run_exit" >&2
    echo "  stderr tail: $(tail -3 "$RESULTS_DIR/run-$run.stderr.log")" >&2
    failed_runs=$((failed_runs + 1))
  elif ! jq -e '.two_step.median_ms_per_token and .control.median_ms_per_token and .coreml.median_ms_per_token' "$RESULTS_DIR/run-$run.json" >/dev/null 2>&1; then
    echo "WARNING: Run $run produced invalid JSON (missing required median fields)" >&2
    failed_runs=$((failed_runs + 1))
  fi
done
total_benchmark_elapsed=$(( $(date +%s) - benchmark_start_epoch ))

if [[ $failed_runs -eq $REPEATS ]]; then
  echo "FATAL: All $REPEATS runs failed. See stderr logs in $RESULTS_DIR" >&2
  exit 1
fi
if [[ $failed_runs -gt 0 ]]; then
  echo "WARNING: $failed_runs/$REPEATS runs failed. Summary will use remaining valid runs." >&2
fi

# Collect valid run JSONs (skip empty or malformed files from failed runs)
valid_runs=()
valid_outer_elapsed=()
valid_stderr_lines=()
for f in "$RESULTS_DIR"/run-*.json; do
  if jq -e '.two_step.median_ms_per_token and .control.median_ms_per_token and .coreml.median_ms_per_token' "$f" >/dev/null 2>&1; then
    valid_runs+=("$f")
    # Collect matching outer elapsed_s file
    elapsed_file="${f%.json}.elapsed_s"
    if [[ -f "$elapsed_file" ]]; then
      valid_outer_elapsed+=("$(cat "$elapsed_file")")
    else
      valid_outer_elapsed+=("null")
    fi
    # Collect stderr line count for the corresponding run
    stderr_file="${f%.json}.stderr.log"
    if [[ -f "$stderr_file" ]]; then
      valid_stderr_lines+=("$(wc -l < "$stderr_file" | tr -d ' ')")
    else
      valid_stderr_lines+=("null")
    fi
  fi
done

if [[ ${#valid_runs[@]} -eq 0 ]]; then
  echo "FATAL: No valid run JSONs found in $RESULTS_DIR" >&2
  exit 1
fi
# Minimum 3 valid runs needed for meaningful statistics (median, IQR, Tukey fences)
MIN_VALID_RUNS=3
echo "Summarizing ${#valid_runs[@]} valid run(s) out of $REPEATS"

two_step_median_ms="$(jq -s 'map(.two_step.median_ms_per_token) | sort | .[((length - 1) / 2 | floor)]' "${valid_runs[@]}")"
control_median_ms="$(jq -s 'map(.control.median_ms_per_token) | sort | .[((length - 1) / 2 | floor)]' "${valid_runs[@]}")"
coreml_median_ms="$(jq -s 'map(.coreml.median_ms_per_token) | sort | .[((length - 1) / 2 | floor)]' "${valid_runs[@]}")"
speedup_median="$(jq -s 'map(.two_step_speedup_vs_coreml) | sort | .[((length - 1) / 2 | floor)]' "${valid_runs[@]}")"
committed_tokens_per_pass="$(jq -s 'map(.two_step.median_committed_exact_tokens_per_pass) | sort | .[((length - 1) / 2 | floor)]' "${valid_runs[@]}")"
accepted_future_tokens_per_pass="$(jq -s 'map(.two_step.median_accepted_future_tokens_per_pass) | sort | .[((length - 1) / 2 | floor)]' "${valid_runs[@]}")"
all_parity_match="$(jq -s 'all(.[]; .parity_status == "match")' "${valid_runs[@]}")"

# p95/p99 tail latency (median-of-per-run-percentiles across repeats)
two_step_p95_ms="$(jq -s 'map(.two_step.p95_ms_per_token // empty) | if length == 0 then "n/a" else sort | .[((length - 1) / 2 | floor)] end' "${valid_runs[@]}")"
two_step_p99_ms="$(jq -s 'map(.two_step.p99_ms_per_token // empty) | if length == 0 then "n/a" else sort | .[((length - 1) / 2 | floor)] end' "${valid_runs[@]}")"
control_p95_ms="$(jq -s 'map(.control.p95_ms_per_token // empty) | if length == 0 then "n/a" else sort | .[((length - 1) / 2 | floor)] end' "${valid_runs[@]}")"
control_p99_ms="$(jq -s 'map(.control.p99_ms_per_token // empty) | if length == 0 then "n/a" else sort | .[((length - 1) / 2 | floor)] end' "${valid_runs[@]}")"
coreml_p95_ms="$(jq -s 'map(.coreml.p95_ms_per_token // empty) | if length == 0 then "n/a" else sort | .[((length - 1) / 2 | floor)] end' "${valid_runs[@]}")"
coreml_p99_ms="$(jq -s 'map(.coreml.p99_ms_per_token // empty) | if length == 0 then "n/a" else sort | .[((length - 1) / 2 | floor)] end' "${valid_runs[@]}")"

# Cross-run spread: min, max, coefficient of variation of per-run medians
two_step_min_ms="$(jq -s 'map(.two_step.median_ms_per_token) | min' "${valid_runs[@]}")"
two_step_max_ms="$(jq -s 'map(.two_step.median_ms_per_token) | max' "${valid_runs[@]}")"
two_step_cv="$(jq -s 'map(.two_step.median_ms_per_token) | (length) as $n | (add / $n) as $mean | if $mean == 0 then 0 else (map(. - $mean | . * .) | add / $n | sqrt) / $mean end' "${valid_runs[@]}")"
control_min_ms="$(jq -s 'map(.control.median_ms_per_token) | min' "${valid_runs[@]}")"
control_max_ms="$(jq -s 'map(.control.median_ms_per_token) | max' "${valid_runs[@]}")"
control_cv="$(jq -s 'map(.control.median_ms_per_token) | (length) as $n | (add / $n) as $mean | if $mean == 0 then 0 else (map(. - $mean | . * .) | add / $n | sqrt) / $mean end' "${valid_runs[@]}")"
coreml_min_ms="$(jq -s 'map(.coreml.median_ms_per_token) | min' "${valid_runs[@]}")"
coreml_max_ms="$(jq -s 'map(.coreml.median_ms_per_token) | max' "${valid_runs[@]}")"
coreml_cv="$(jq -s 'map(.coreml.median_ms_per_token) | (length) as $n | (add / $n) as $mean | if $mean == 0 then 0 else (map(. - $mean | . * .) | add / $n | sqrt) / $mean end' "${valid_runs[@]}")"
speedup_cv="$(jq -s 'map(.two_step_speedup_vs_coreml) | (length) as $n | (add / $n) as $mean | if $mean == 0 then 0 else (map(. - $mean | . * .) | add / $n | sqrt) / $mean end' "${valid_runs[@]}")"
speedup_min="$(jq -s 'map(.two_step_speedup_vs_coreml) | min' "${valid_runs[@]}")"
speedup_max="$(jq -s 'map(.two_step_speedup_vs_coreml) | max' "${valid_runs[@]}")"
control_speedup="$(jq -s 'map(.control_speedup_vs_coreml // null) | if all(. != null) then sort | .[((length - 1) / 2 | floor)] else "n/a" end' "${valid_runs[@]}")"
control_speedup_min="$(jq -s 'map(.control_speedup_vs_coreml // null) | map(select(. != null)) | if length > 0 then min else "n/a" end' "${valid_runs[@]}")"

# Mean, stddev, IQR for summary.txt (already in summary.json via inline jq)
two_step_mean_ms="$(jq -s 'map(.two_step.median_ms_per_token) | add / length' "${valid_runs[@]}")"
two_step_stddev_ms="$(jq -s 'map(.two_step.median_ms_per_token) | (length) as $n | (add / $n) as $mean | (map(. - $mean | . * .) | add / $n | sqrt)' "${valid_runs[@]}")"
two_step_iqr_ms="$(jq -s 'map(.two_step.median_ms_per_token) | sort | if length < 4 then (last - first) else (.[((length * 3 / 4) | floor)] - .[((length / 4) | floor)]) end' "${valid_runs[@]}")"
control_mean_ms="$(jq -s 'map(.control.median_ms_per_token) | add / length' "${valid_runs[@]}")"
control_stddev_ms="$(jq -s 'map(.control.median_ms_per_token) | (length) as $n | (add / $n) as $mean | (map(. - $mean | . * .) | add / $n | sqrt)' "${valid_runs[@]}")"
control_iqr_ms="$(jq -s 'map(.control.median_ms_per_token) | sort | if length < 4 then (last - first) else (.[((length * 3 / 4) | floor)] - .[((length / 4) | floor)]) end' "${valid_runs[@]}")"
coreml_mean_ms="$(jq -s 'map(.coreml.median_ms_per_token) | add / length' "${valid_runs[@]}")"
coreml_stddev_ms="$(jq -s 'map(.coreml.median_ms_per_token) | (length) as $n | (add / $n) as $mean | (map(. - $mean | . * .) | add / $n | sqrt)' "${valid_runs[@]}")"
coreml_iqr_ms="$(jq -s 'map(.coreml.median_ms_per_token) | sort | if length < 4 then (last - first) else (.[((length * 3 / 4) | floor)] - .[((length / 4) | floor)]) end' "${valid_runs[@]}")"
speedup_mean="$(jq -s 'map(.two_step_speedup_vs_coreml) | add / length' "${valid_runs[@]}")"
speedup_stddev="$(jq -s 'map(.two_step_speedup_vs_coreml) | (length) as $n | (add / $n) as $mean | (map(. - $mean | . * .) | add / $n | sqrt)' "${valid_runs[@]}")"
speedup_iqr="$(jq -s 'map(.two_step_speedup_vs_coreml) | sort | if length < 4 then (last - first) else (.[((length * 3 / 4) | floor)] - .[((length / 4) | floor)]) end' "${valid_runs[@]}")"

{
  echo "harness_version=$HARNESS_VERSION"
  echo "contract_hash=$CONTRACT_HASH"
  echo "results_dir=$RESULTS_DIR"
  echo "requested_repeats=$REPEATS"
  echo "valid_runs=${#valid_runs[@]}"
  echo "failed_runs=$failed_runs"
  echo "two_step_median_ms_per_token=$two_step_median_ms"
  echo "two_step_p95_ms_per_token=$two_step_p95_ms"
  echo "two_step_p99_ms_per_token=$two_step_p99_ms"
  echo "two_step_min_ms_per_token=$two_step_min_ms"
  echo "two_step_max_ms_per_token=$two_step_max_ms"
  echo "two_step_mean_ms_per_token=$two_step_mean_ms"
  echo "two_step_stddev_ms_per_token=$two_step_stddev_ms"
  echo "two_step_iqr_ms=$two_step_iqr_ms"
  echo "two_step_cv=$two_step_cv"
  echo "control_median_ms_per_token=$control_median_ms"
  echo "control_p95_ms_per_token=$control_p95_ms"
  echo "control_p99_ms_per_token=$control_p99_ms"
  echo "control_min_ms_per_token=$control_min_ms"
  echo "control_max_ms_per_token=$control_max_ms"
  echo "control_mean_ms_per_token=$control_mean_ms"
  echo "control_stddev_ms_per_token=$control_stddev_ms"
  echo "control_iqr_ms=$control_iqr_ms"
  echo "control_cv=$control_cv"
  echo "coreml_median_ms_per_token=$coreml_median_ms"
  echo "coreml_p95_ms_per_token=$coreml_p95_ms"
  echo "coreml_p99_ms_per_token=$coreml_p99_ms"
  echo "coreml_min_ms_per_token=$coreml_min_ms"
  echo "coreml_max_ms_per_token=$coreml_max_ms"
  echo "coreml_mean_ms_per_token=$coreml_mean_ms"
  echo "coreml_stddev_ms_per_token=$coreml_stddev_ms"
  echo "coreml_iqr_ms=$coreml_iqr_ms"
  echo "coreml_cv=$coreml_cv"
  echo "two_step_speedup_vs_coreml=$speedup_median"
  echo "two_step_speedup_min=$speedup_min"
  echo "two_step_speedup_max=$speedup_max"
  echo "two_step_speedup_mean=$speedup_mean"
  echo "two_step_speedup_stddev=$speedup_stddev"
  echo "two_step_speedup_cv=$speedup_cv"
  echo "two_step_speedup_iqr=$speedup_iqr"
  echo "control_speedup_vs_coreml=$control_speedup"
  echo "total_elapsed_s=$total_benchmark_elapsed"
  echo "probe_build_elapsed_s=$PROBE_BUILD_ELAPSED_S"
  echo "committed_exact_tokens_per_pass=$committed_tokens_per_pass"
  echo "accepted_future_tokens_per_pass=$accepted_future_tokens_per_pass"
  echo "all_parity_match=$all_parity_match"
  parity_counts="$(jq -s '[.[] | .parity_match_count // null] | map(select(. != null))' "${valid_runs[@]}" 2>/dev/null || echo "[]")"
  echo "parity_match_counts=$parity_counts"
} | tee "$RESULTS_DIR/summary.txt"

# Machine-readable aggregate JSON combining all per-run data with summary stats
jq -s \
  --argjson harness_version "$HARNESS_VERSION" \
  --arg dir "$RESULTS_DIR" \
  --arg ts "$(date -Iseconds)" \
  --arg commit "$GIT_COMMIT_START" \
  --arg branch "$(git -C "$ROOT" rev-parse --abbrev-ref HEAD)" \
  --arg input_mode "$INPUT_MODE" \
  --argjson warmup "$WARMUP" \
  --argjson iterations "$ITERATIONS" \
  --argjson max_new_tokens "$MAX_NEW_TOKENS" \
  --argjson max_seq "$MAX_SEQUENCE_TOKENS" \
  --argjson layers "$LAYER_COUNT" \
  --arg control_backend "$CONTROL_BACKEND" \
  --arg two_step_backend "$TWO_STEP_BACKEND" \
  --arg output_head_backend "$OUTPUT_HEAD_BACKEND" \
  --arg cv_thresh "$CV_THRESHOLD" \
  --arg duration_budget "$DURATION_BUDGET_S" \
  --argjson requested_repeats "$REPEATS" \
  --argjson failed "$failed_runs" \
  --argjson total_elapsed_s "$total_benchmark_elapsed" \
  --argjson probe_build_elapsed_s "$PROBE_BUILD_ELAPSED_S" \
  --arg hw_model "$(sysctl -n hw.model 2>/dev/null || echo unknown)" \
  --arg load_avg "$LOAD_START" \
  --arg macos_version "$(sw_vers -productVersion 2>/dev/null || echo unknown)" \
  --arg macos_build "$(sw_vers -buildVersion 2>/dev/null || echo unknown)" \
  --arg power_source "$(pmset -g batt 2>/dev/null | head -1 | sed "s/.*'\(.*\)'.*/\1/" || echo unknown)" \
  --arg chip "$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo unknown)" \
  --argjson ncpu "$(sysctl -n hw.ncpu 2>/dev/null || echo null)" \
  --argjson physical_memory_gb "$(( $(sysctl -n hw.memsize 2>/dev/null || echo 0) / 1073741824 ))" \
  --arg thermal_pressure "$THERMAL_START" \
  --arg thermal_pressure_end "$(pmset -g therm 2>/dev/null | grep -i 'cpu.*speed' | head -1 || echo unknown)" \
  --arg load_avg_end "$(sysctl -n vm.loadavg 2>/dev/null || echo unknown)" \
  --argjson memory_free_pct "$(sysctl -n kern.memorystatus_level 2>/dev/null || echo null)" \
  --arg swift_version "$(swift --version 2>/dev/null | head -1 || echo unknown)" \
  --arg jq_version "$(jq --version 2>/dev/null || echo unknown)" \
  --argjson outer_elapsed "$(printf '%s\n' "${valid_outer_elapsed[@]}" | jq -s '.')" \
  --argjson prompt_token "${PROMPT_TOKEN:-null}" \
  --argjson stderr_lines "$(printf '%s\n' "${valid_stderr_lines[@]}" | jq -s '.')" \
  --arg contract_hash "$CONTRACT_HASH" \
  --arg metadata_sha256 "$METADATA_SHA256" \
  --arg probe_sha256 "$PROBE_SHA256" \
  --arg coreml_sha256 "${COREML_MODEL_SHA256:-}" \
  --arg recurrent_sha256 "${RECURRENT_SHA256:-}" \
  --arg sidecar_sha256 "${FUTURE_SIDECAR_SHA256:-}" \
  --arg generation_sha256 "${GENERATION_MODEL_SHA256:-}" \
  --argjson run_files "$(printf '%s\n' "${valid_runs[@]}" | while read -r f; do basename "$f"; done | jq -nR '[inputs | select(length > 0)]')" \
'{
  harness_version: $harness_version,
  probe_version: (map(.probe_version // null) | .[0]),
  per_run_probe_versions: (map(.probe_version // null)),
  per_run_build_configurations: (map(.build_configuration // null)),
  per_run_input_modes: (map(.input_mode // null)),
  per_run_control_backends: (map(.control_backend // null)),
  per_run_two_step_backends: (map(.two_step_backend // null)),
  per_run_output_head_backends: (map(.output_head_backend // null)),
  per_run_layer_counts: (map(.layer_count // null)),
  per_run_prompt_tokens: (map(.prompt_token // null)),
  per_run_max_new_tokens: (map(.max_new_tokens // null)),
  per_run_max_sequence_tokens: (map(.max_sequence_tokens // null)),
  per_run_hostnames: (map(.hostname // null)),
  per_run_os_versions: (map(.os_version // null)),
  per_run_process_ids: (map(.process_id // null)),
  per_run_warmup: (map(.warmup // null)),
  per_run_iterations: (map(.iterations // null)),
  results_dir: $dir,
  timestamp: $ts,
  git_commit: $commit,
  git_branch: $branch,
  host: {hw_model: $hw_model, chip: $chip, ncpu: $ncpu, physical_memory_gb: $physical_memory_gb, thermal_pressure: $thermal_pressure, thermal_pressure_end: $thermal_pressure_end, load_average: $load_avg, load_average_end: $load_avg_end, memory_free_pct: $memory_free_pct, macos_version: $macos_version, macos_build: $macos_build, power_source: $power_source},
  toolchain: {swift_version: $swift_version, jq_version: $jq_version},
  benchmark_contract: {
    contract_hash: $contract_hash,
    input_mode: $input_mode,
    control_backend: $control_backend,
    two_step_backend: $two_step_backend,
    output_head_backend: $output_head_backend,
    warmup: $warmup,
    iterations: $iterations,
    max_new_tokens: $max_new_tokens,
    max_sequence_tokens: $max_seq,
    layer_count: $layers,
    prompt_token: $prompt_token,
    cv_threshold: ($cv_thresh | tonumber),
    duration_budget_s: ($duration_budget | tonumber)
  },
  artifact_hashes: {
    metadata_sha256: $metadata_sha256,
    probe_sha256: $probe_sha256,
    coreml_model_sha256: (if $coreml_sha256 == "" then null else $coreml_sha256 end),
    recurrent_checkpoint_sha256: (if $recurrent_sha256 == "" then null else $recurrent_sha256 end),
    future_sidecar_sha256: (if $sidecar_sha256 == "" then null else $sidecar_sha256 end),
    generation_model_sha256: (if $generation_sha256 == "" then null else $generation_sha256 end)
  },
  probe_build_elapsed_s: $probe_build_elapsed_s,
  requested_repeats: $requested_repeats,
  valid_runs: (length),
  failed_runs: $failed,
  total_elapsed_s: $total_elapsed_s,
  init_times: {
    control_init_wall_ms: (map(.control.init_wall_ms // null) | if all(. != null) then sort | .[((length - 1) / 2 | floor)] else null end),
    control_reported_compile_ms: (map(.control.reported_compile_ms // null) | if all(. != null) then sort | .[((length - 1) / 2 | floor)] else null end),
    two_step_init_wall_ms: (map(.two_step.init_wall_ms // null) | if all(. != null) then sort | .[((length - 1) / 2 | floor)] else null end),
    two_step_reported_compile_ms: (map(.two_step.reported_compile_ms // null) | if all(. != null) then sort | .[((length - 1) / 2 | floor)] else null end),
    coreml_compile_ms: (map(.coreml.reported_compile_ms // null) | if all(. != null) then sort | .[((length - 1) / 2 | floor)] else null end),
    control_init_wall_min_ms: (map(.control.init_wall_ms // null) | map(select(. != null)) | if length > 0 then min else null end),
    control_init_wall_max_ms: (map(.control.init_wall_ms // null) | map(select(. != null)) | if length > 0 then max else null end),
    two_step_init_wall_min_ms: (map(.two_step.init_wall_ms // null) | map(select(. != null)) | if length > 0 then min else null end),
    two_step_init_wall_max_ms: (map(.two_step.init_wall_ms // null) | map(select(. != null)) | if length > 0 then max else null end),
    coreml_compile_min_ms: (map(.coreml.reported_compile_ms // null) | map(select(. != null)) | if length > 0 then min else null end),
    coreml_compile_max_ms: (map(.coreml.reported_compile_ms // null) | map(select(. != null)) | if length > 0 then max else null end),
    per_run_control_init_wall_ms: (map(.control.init_wall_ms // null)),
    per_run_two_step_init_wall_ms: (map(.two_step.init_wall_ms // null)),
    per_run_coreml_compile_ms: (map(.coreml.reported_compile_ms // null))
  },
  two_step: {
    median_ms_per_token: (map(.two_step.median_ms_per_token) | sort | .[((length - 1) / 2 | floor)]),
    p95_ms_per_token: (map(.two_step.p95_ms_per_token // empty) | if length == 0 then null else sort | .[((length - 1) / 2 | floor)] end),
    p99_ms_per_token: (map(.two_step.p99_ms_per_token // empty) | if length == 0 then null else sort | .[((length - 1) / 2 | floor)] end),
    min_ms_per_token: (map(.two_step.median_ms_per_token) | min),
    max_ms_per_token: (map(.two_step.median_ms_per_token) | max),
    mean_ms_per_token: (map(.two_step.median_ms_per_token) | add / length),
    stddev_ms_per_token: (map(.two_step.median_ms_per_token) | (length) as $n | (add / $n) as $mean | (map(. - $mean | . * .) | add / $n | sqrt)),
    cv: (map(.two_step.median_ms_per_token) | (length) as $n | (add / $n) as $mean | if $mean == 0 then 0 else (map(. - $mean | . * .) | add / $n | sqrt) / $mean end),
    per_run_medians_ms: (map(.two_step.median_ms_per_token)),
    per_run_p95s_ms: (map(.two_step.p95_ms_per_token // null)),
    per_run_p99s_ms: (map(.two_step.p99_ms_per_token // null)),
    per_run_iteration_min_ms: (map(.two_step.raw_token_latencies_ms // null | if . != null then min else null end)),
    per_run_iteration_max_ms: (map(.two_step.raw_token_latencies_ms // null | if . != null then max else null end)),
    iqr_ms: (map(.two_step.median_ms_per_token) | sort | if length < 4 then (last - first) else (.[((length * 3 / 4) | floor)] - .[((length / 4) | floor)]) end),
    breakdown: {
      proposer_ms_per_pass: (map(.two_step.median_proposer_ms_per_pass // null) | if all(. != null) then sort | .[((length - 1) / 2 | floor)] else null end),
      verifier_trunk_ms_per_pass: (map(.two_step.median_verifier_trunk_ms_per_pass // null) | if all(. != null) then sort | .[((length - 1) / 2 | floor)] else null end),
      verifier_logits_ms_per_pass: (map(.two_step.median_verifier_logits_ms_per_pass // null) | if all(. != null) then sort | .[((length - 1) / 2 | floor)] else null end),
      state_advance_ms_per_pass: (map(.two_step.median_state_advance_ms_per_pass // null) | if all(. != null) then sort | .[((length - 1) / 2 | floor)] else null end),
      per_run_proposer_ms: (map(.two_step.median_proposer_ms_per_pass // null)),
      per_run_verifier_trunk_ms: (map(.two_step.median_verifier_trunk_ms_per_pass // null)),
      per_run_verifier_logits_ms: (map(.two_step.median_verifier_logits_ms_per_pass // null)),
      per_run_state_advance_ms: (map(.two_step.median_state_advance_ms_per_pass // null))
    }
  },
  control: {
    median_ms_per_token: (map(.control.median_ms_per_token) | sort | .[((length - 1) / 2 | floor)]),
    p95_ms_per_token: (map(.control.p95_ms_per_token // empty) | if length == 0 then null else sort | .[((length - 1) / 2 | floor)] end),
    p99_ms_per_token: (map(.control.p99_ms_per_token // empty) | if length == 0 then null else sort | .[((length - 1) / 2 | floor)] end),
    min_ms_per_token: (map(.control.median_ms_per_token) | min),
    max_ms_per_token: (map(.control.median_ms_per_token) | max),
    mean_ms_per_token: (map(.control.median_ms_per_token) | add / length),
    stddev_ms_per_token: (map(.control.median_ms_per_token) | (length) as $n | (add / $n) as $mean | (map(. - $mean | . * .) | add / $n | sqrt)),
    cv: (map(.control.median_ms_per_token) | (length) as $n | (add / $n) as $mean | if $mean == 0 then 0 else (map(. - $mean | . * .) | add / $n | sqrt) / $mean end),
    per_run_medians_ms: (map(.control.median_ms_per_token)),
    per_run_p95s_ms: (map(.control.p95_ms_per_token // null)),
    per_run_p99s_ms: (map(.control.p99_ms_per_token // null)),
    per_run_iteration_min_ms: (map(.control.raw_token_latencies_ms // null | if . != null then min else null end)),
    per_run_iteration_max_ms: (map(.control.raw_token_latencies_ms // null | if . != null then max else null end)),
    iqr_ms: (map(.control.median_ms_per_token) | sort | if length < 4 then (last - first) else (.[((length * 3 / 4) | floor)] - .[((length / 4) | floor)]) end),
    breakdown: {
      trunk_ms_per_token: (map(.control.median_trunk_ms_per_token // null) | if all(. != null) then sort | .[((length - 1) / 2 | floor)] else null end),
      logits_ms_per_token: (map(.control.median_logits_ms_per_token // null) | if all(. != null) then sort | .[((length - 1) / 2 | floor)] else null end),
      per_run_trunk_ms: (map(.control.median_trunk_ms_per_token // null)),
      per_run_logits_ms: (map(.control.median_logits_ms_per_token // null))
    }
  },
  coreml: {
    median_ms_per_token: (map(.coreml.median_ms_per_token) | sort | .[((length - 1) / 2 | floor)]),
    p95_ms_per_token: (map(.coreml.p95_ms_per_token // empty) | if length == 0 then null else sort | .[((length - 1) / 2 | floor)] end),
    p99_ms_per_token: (map(.coreml.p99_ms_per_token // empty) | if length == 0 then null else sort | .[((length - 1) / 2 | floor)] end),
    min_ms_per_token: (map(.coreml.median_ms_per_token) | min),
    max_ms_per_token: (map(.coreml.median_ms_per_token) | max),
    mean_ms_per_token: (map(.coreml.median_ms_per_token) | add / length),
    stddev_ms_per_token: (map(.coreml.median_ms_per_token) | (length) as $n | (add / $n) as $mean | (map(. - $mean | . * .) | add / $n | sqrt)),
    cv: (map(.coreml.median_ms_per_token) | (length) as $n | (add / $n) as $mean | if $mean == 0 then 0 else (map(. - $mean | . * .) | add / $n | sqrt) / $mean end),
    per_run_medians_ms: (map(.coreml.median_ms_per_token)),
    per_run_p95s_ms: (map(.coreml.p95_ms_per_token // null)),
    per_run_p99s_ms: (map(.coreml.p99_ms_per_token // null)),
    per_run_iteration_min_ms: (map(.coreml.raw_token_latencies_ms // null | if . != null then min else null end)),
    per_run_iteration_max_ms: (map(.coreml.raw_token_latencies_ms // null | if . != null then max else null end)),
    iqr_ms: (map(.coreml.median_ms_per_token) | sort | if length < 4 then (last - first) else (.[((length * 3 / 4) | floor)] - .[((length / 4) | floor)]) end),
    breakdown: {
      trunk_ms_per_token: (map(.coreml.median_trunk_ms_per_token // null) | if all(. != null) then sort | .[((length - 1) / 2 | floor)] else null end),
      logits_ms_per_token: (map(.coreml.median_logits_ms_per_token // null) | if all(. != null) then sort | .[((length - 1) / 2 | floor)] else null end),
      per_run_trunk_ms: (map(.coreml.median_trunk_ms_per_token // null)),
      per_run_logits_ms: (map(.coreml.median_logits_ms_per_token // null))
    }
  },
  two_step_speedup_vs_coreml: (map(.two_step_speedup_vs_coreml) | sort | .[((length - 1) / 2 | floor)]),
  two_step_speedup_min: (map(.two_step_speedup_vs_coreml) | min),
  two_step_speedup_max: (map(.two_step_speedup_vs_coreml) | max),
  two_step_speedup_mean: (map(.two_step_speedup_vs_coreml) | add / length),
  two_step_speedup_stddev: (map(.two_step_speedup_vs_coreml) | (length) as $n | (add / $n) as $mean | (map(. - $mean | . * .) | add / $n | sqrt)),
  two_step_speedup_cv: (map(.two_step_speedup_vs_coreml) | (length) as $n | (add / $n) as $mean | if $mean == 0 then 0 else (map(. - $mean | . * .) | add / $n | sqrt) / $mean end),
  two_step_speedup_iqr: (map(.two_step_speedup_vs_coreml) | sort | if length < 4 then (last - first) else (.[((length * 3 / 4) | floor)] - .[((length / 4) | floor)]) end),
  per_run_speedups: (map(.two_step_speedup_vs_coreml)),
  control_speedup_vs_coreml: (map(.control_speedup_vs_coreml // null) | if all(. != null) then sort | .[((length - 1) / 2 | floor)] else null end),
  control_speedup_min: (map(.control_speedup_vs_coreml // null) | map(select(. != null)) | if length > 0 then min else null end),
  control_speedup_max: (map(.control_speedup_vs_coreml // null) | map(select(. != null)) | if length > 0 then max else null end),
  control_speedup_cv: (map(.control_speedup_vs_coreml // null) | map(select(. != null)) | if length >= 2 then (length) as $n | (add / $n) as $mean | if $mean == 0 then 0 else (map(. - $mean | . * .) | add / $n | sqrt) / $mean end else null end),
  control_speedup_iqr: (map(.control_speedup_vs_coreml // null) | map(select(. != null)) | sort | if length < 4 then (if length > 0 then (last - first) else null end) else (.[((length * 3 / 4) | floor)] - .[((length / 4) | floor)]) end),
  per_run_control_speedups: (map(.control_speedup_vs_coreml // null)),
  token_accounting: {
    committed_exact_tokens_per_pass: (map(.two_step.median_committed_exact_tokens_per_pass) | sort | .[((length - 1) / 2 | floor)]),
    accepted_future_tokens_per_pass: (map(.two_step.median_accepted_future_tokens_per_pass) | sort | .[((length - 1) / 2 | floor)]),
    committed_per_run: (map(.two_step.median_committed_exact_tokens_per_pass)),
    accepted_per_run: (map(.two_step.median_accepted_future_tokens_per_pass))
  },
  committed_exact_tokens_per_pass: (map(.two_step.median_committed_exact_tokens_per_pass) | sort | .[((length - 1) / 2 | floor)]),
  accepted_future_tokens_per_pass: (map(.two_step.median_accepted_future_tokens_per_pass) | sort | .[((length - 1) / 2 | floor)]),
  all_parity_match: (all(.[]; .parity_status == "match")),
  per_run_parity: (map(.parity_status)),
  per_run_parity_match_count: (map(.parity_match_count // null)),
  parity_total: (map(.parity_total // null) | .[0]),
  per_run_control_generated_count: (map(.control.generated_tokens // null | if . != null then length else null end)),
  per_run_two_step_generated_count: (map(.two_step.generated_tokens // null | if . != null then length else null end)),
  per_run_timestamps: (map(.probe_timestamp // null)),
  per_run_iteration_counts: (map(.measured_iteration_count // null)),
  first_run_timestamp: (map(.probe_timestamp // null) | map(select(. != null)) | sort | first // null),
  last_run_timestamp: (map(.probe_timestamp // null) | map(select(. != null)) | sort | last // null),
  valid_run_files: $run_files,
  per_run_wall_elapsed_s: (map(.probe_wall_elapsed_s // null)),
  per_run_outer_elapsed_s: $outer_elapsed,
  per_run_stderr_lines: $stderr_lines,
  total_stderr_lines: ($stderr_lines | map(. // 0) | add),
  sum_probe_wall_elapsed_s: (map(.probe_wall_elapsed_s // 0) | add),
  sum_outer_elapsed_s: ($outer_elapsed | map(. // 0) | add),
  probe_wall_range_s: (map(.probe_wall_elapsed_s // null) | map(select(. != null)) | if length >= 2 then (max - min) else null end),
  outer_elapsed_range_s: ($outer_elapsed | map(select(. != null)) | if length >= 2 then (max - min) else null end)
}' "${valid_runs[@]}" > "$RESULTS_DIR/summary.json"

# Validate summary.json was written correctly
if [[ ! -s "$RESULTS_DIR/summary.json" ]] || ! jq -e '.valid_runs' "$RESULTS_DIR/summary.json" >/dev/null 2>&1; then
  echo "FATAL: summary.json is empty or malformed after aggregation" >&2
  exit 1
fi

# Reproducibility gate: warn on high cross-run variance or parity failure
gate_status="pass"
gate_warnings=""

# Environment quality warnings (non-failing)
power_source="$(pmset -g batt 2>/dev/null | head -1 | sed "s/.*'\(.*\)'.*/\1/" || echo unknown)"
if [[ "$power_source" != "AC Power" && "$power_source" != "unknown" ]]; then
  gate_warnings="${gate_warnings}BATTERY_POWER: running on '${power_source}' — frequency scaling may reduce reproducibility\n"
fi
if [[ "$total_benchmark_elapsed" -gt "$DURATION_BUDGET_S" ]]; then
  gate_warnings="${gate_warnings}LONG_DURATION: total ${total_benchmark_elapsed}s exceeds budget ${DURATION_BUDGET_S}s — thermal throttling may affect results\n"
fi
# Thermal drift: compare pre- and post-benchmark thermal pressure
THERMAL_END="$(pmset -g therm 2>/dev/null | grep -i 'cpu.*speed' | head -1 || echo unknown)"
if [[ "$THERMAL_START" != "$THERMAL_END" && "$THERMAL_START" != "unknown" && "$THERMAL_END" != "unknown" ]]; then
  gate_warnings="${gate_warnings}THERMAL_DRIFT: thermal pressure changed during benchmark (start='${THERMAL_START}', end='${THERMAL_END}')\n"
fi

# Load average drift: warn if 1-min load increased significantly
LOAD_END="$(sysctl -n vm.loadavg 2>/dev/null || echo unknown)"
load_start_1m="$(echo "$LOAD_START" | awk '{gsub(/[{}]/,""); print $1}' 2>/dev/null || echo "")"
load_end_1m="$(echo "$LOAD_END" | awk '{gsub(/[{}]/,""); print $1}' 2>/dev/null || echo "")"
if [[ -n "$load_start_1m" && -n "$load_end_1m" && "$load_start_1m" != "unknown" && "$load_end_1m" != "unknown" ]]; then
  if jq -e --arg s "$load_start_1m" --arg e "$load_end_1m" \
    '($e | tonumber) > ([$s | tonumber, 1.0] | max) * 2' <<< 'null' >/dev/null 2>&1; then
    gate_warnings="${gate_warnings}LOAD_DRIFT: 1-min load average increased significantly (start=${load_start_1m} end=${load_end_1m})\n"
  fi
fi

# Disk space check: warn if free space dropped below 512MB
DISK_FREE_MB_END="$(df -m "$RESULTS_DIR" 2>/dev/null | awk 'NR==2{print $4}' || echo 0)"
if [[ "$DISK_FREE_MB_END" -lt 512 ]] 2>/dev/null; then
  gate_warnings="${gate_warnings}LOW_DISK_SPACE: only ${DISK_FREE_MB_END}MB free (started with ${DISK_FREE_MB_START}MB) — results may be unreliable\n"
fi

# Memory pressure check: warn if system memory is critically low
MEMORY_FREE_PCT_END="$(sysctl -n kern.memorystatus_level 2>/dev/null || echo 100)"
if [[ "$MEMORY_FREE_PCT_END" -lt 20 ]] 2>/dev/null; then
  gate_warnings="${gate_warnings}LOW_MEMORY: system memory free at ${MEMORY_FREE_PCT_END}% — swapping may affect benchmark timing\n"
fi

if [[ ${#valid_runs[@]} -lt $MIN_VALID_RUNS ]]; then
  gate_status="warn"
  gate_warnings="${gate_warnings}TOO_FEW_VALID_RUNS: only ${#valid_runs[@]} valid runs (minimum $MIN_VALID_RUNS for reliable statistics)\n"
fi

if [[ $failed_runs -gt 0 ]]; then
  gate_status="warn"
  gate_warnings="${gate_warnings}FAILED_RUNS: ${failed_runs}/${REPEATS} runs failed\n"
fi

# Build configuration check: warn if any run was built in debug mode
debug_runs="$(jq -s 'map(.build_configuration // null) | map(select(. == "debug")) | if length > 0 then length else empty end' "${valid_runs[@]}" 2>/dev/null || echo "")"
if [[ -n "$debug_runs" ]]; then
  gate_status="warn"
  gate_warnings="${gate_warnings}DEBUG_BUILD: ${debug_runs} run(s) used debug build configuration — results are not representative\n"
fi

# Build configuration consistency check
build_config_mismatch="$(jq -s 'map(.build_configuration // null) | map(select(. != null)) | unique | if length > 1 then . else empty end' "${valid_runs[@]}" 2>/dev/null || echo "")"
if [[ -n "$build_config_mismatch" ]]; then
  gate_status="warn"
  gate_warnings="${gate_warnings}BUILD_CONFIG_MISMATCH: runs used different build configurations: ${build_config_mismatch}\n"
fi

# Probe version consistency check
probe_version_mismatch="$(jq -s 'map(.probe_version // null) | unique | if length > 1 then . else empty end' "${valid_runs[@]}" 2>/dev/null || echo "")"
if [[ -n "$probe_version_mismatch" ]]; then
  gate_status="warn"
  gate_warnings="${gate_warnings}PROBE_VERSION_MISMATCH: runs used different probe versions: ${probe_version_mismatch}\n"
fi

# Hostname consistency check
hostname_mismatch="$(jq -s 'map(.hostname // null) | map(select(. != null)) | unique | if length > 1 then . else empty end' "${valid_runs[@]}" 2>/dev/null || echo "")"
if [[ -n "$hostname_mismatch" ]]; then
  gate_status="fail"
  gate_warnings="${gate_warnings}HOSTNAME_MISMATCH: runs executed on different hosts: ${hostname_mismatch}\n"
fi

# OS version consistency check
os_version_mismatch="$(jq -s 'map(.os_version // null) | map(select(. != null)) | unique | if length > 1 then . else empty end' "${valid_runs[@]}" 2>/dev/null || echo "")"
if [[ -n "$os_version_mismatch" ]]; then
  gate_status="warn"
  gate_warnings="${gate_warnings}OS_VERSION_MISMATCH: runs report different OS versions: ${os_version_mismatch}\n"
fi

# Process ID uniqueness check (detect shared processes)
pid_dupes="$(jq -s 'map(.process_id // null) | map(select(. != null)) | group_by(.) | map(select(length > 1) | .[0]) | if length > 0 then . else empty end' "${valid_runs[@]}" 2>/dev/null || echo "")"
if [[ -n "$pid_dupes" ]]; then
  gate_status="fail"
  gate_warnings="${gate_warnings}PID_COLLISION: multiple runs share process IDs: ${pid_dupes}\n"
fi

# Contract field consistency checks — gate fail on any mismatch
check_contract_field() {
  local field="$1" expected="$2" jq_type="${3:-string}"
  local mismatch=""
  if [[ "$jq_type" == "number" ]]; then
    mismatch="$(jq -s --argjson expected "$expected" --arg field "$field" \
      'map(.[$field] // null) | map(select(. != null and . != $expected)) | if length > 0 then . else empty end' \
      "${valid_runs[@]}" 2>/dev/null || echo "")"
  else
    mismatch="$(jq -s --arg expected "$expected" --arg field "$field" \
      'map(.[$field] // null) | map(select(. != null and . != $expected)) | if length > 0 then . else empty end' \
      "${valid_runs[@]}" 2>/dev/null || echo "")"
  fi
  if [[ -n "$mismatch" ]]; then
    gate_status="fail"
    gate_warnings="${gate_warnings}CONTRACT_MISMATCH: ${field} inconsistent with contract (${expected}): ${mismatch}\n"
  fi
}

check_contract_field input_mode "$INPUT_MODE"
check_contract_field control_backend "$CONTROL_BACKEND"
check_contract_field two_step_backend "$TWO_STEP_BACKEND"
check_contract_field output_head_backend "$OUTPUT_HEAD_BACKEND"
check_contract_field layer_count "$LAYER_COUNT" number
check_contract_field max_new_tokens "$MAX_NEW_TOKENS" number
check_contract_field max_sequence_tokens "$MAX_SEQUENCE_TOKENS" number
check_contract_field warmup "$WARMUP" number
check_contract_field iterations "$ITERATIONS" number
check_contract_field prompt_token "$PROMPT_TOKEN" number

# Generated token count check (detect premature termination)
short_gen="$(jq -s --argjson expected "$MAX_NEW_TOKENS" \
  '[.[] | {control: (.control.generated_tokens // null | if . != null then length else null end), two_step: (.two_step.generated_tokens // null | if . != null then length else null end)} | select(.control != $expected or .two_step != $expected)] | if length > 0 then . else empty end' \
  "${valid_runs[@]}" 2>/dev/null || echo "")"
if [[ -n "$short_gen" ]]; then
  gate_status="warn"
  gate_warnings="${gate_warnings}SHORT_GENERATION: some runs generated fewer than max_new_tokens ($MAX_NEW_TOKENS) tokens\n"
fi

# Iteration count consistency (detect early termination)
iter_mismatch="$(jq -s --argjson expected "$ITERATIONS" \
  'map(.measured_iteration_count // null) | map(select(. != null and . != $expected)) | if length > 0 then . else empty end' \
  "${valid_runs[@]}" 2>/dev/null || echo "")"
if [[ -n "$iter_mismatch" ]]; then
  gate_status="warn"
  gate_warnings="${gate_warnings}ITERATION_COUNT_MISMATCH: some runs measured ${iter_mismatch} iterations (expected ${ITERATIONS})\n"
fi

# Timestamp monotonicity check (detect clock skew or out-of-order execution)
ts_nonmono="$(jq -s '
  [.[] | .probe_timestamp // null] | map(select(. != null)) |
  if length < 2 then empty
  else . as $ts |
    [range(1; $ts | length) | select($ts[.] < $ts[. - 1])] |
    if length > 0 then "non-monotonic at indices: \(.)" else empty end
  end
' "${valid_runs[@]}" 2>/dev/null || echo "")"
if [[ -n "$ts_nonmono" ]]; then
  gate_warnings="${gate_warnings}TIMESTAMP_ORDER: probe timestamps are not strictly increasing ($ts_nonmono)\n"
fi

# Per-run wall-time variance: warn if any run took >2x the median
wall_outlier="$(jq -s 'map(.probe_wall_elapsed_s // null) | map(select(. != null)) |
  if length < 3 then empty
  else sort | .[((length - 1) / 2 | floor)] as $med |
    map(select(. > $med * 2)) |
    if length > 0 then "runs with >2x median wall time: \(.)" else empty end
  end' "${valid_runs[@]}" 2>/dev/null || echo "")"
if [[ -n "$wall_outlier" ]]; then
  gate_warnings="${gate_warnings}WALL_TIME_OUTLIER: ${wall_outlier}\n"
fi

if [[ "$all_parity_match" != "true" ]]; then
  gate_status="fail"
  parity_detail="$(jq -s '[.[] | {run: input_line_number, status: .parity_status, match_count: (.parity_match_count // "n/a"), total: (.parity_total // "n/a")} | select(.status != "match")] | map("\(.run): \(.match_count)/\(.total)") | join(", ")' "${valid_runs[@]}" 2>/dev/null || echo "detail unavailable")"
  gate_warnings="${gate_warnings}PARITY_MISMATCH: not all runs produced matching tokens (${parity_detail})\n"
fi

for path_label in two_step control coreml speedup; do
  cv_var="${path_label}_cv"
  cv_val="${!cv_var}"
  if [[ -n "$cv_val" ]] && jq -e --arg cv "$cv_val" --arg thresh "$CV_THRESHOLD" \
    '($cv | tonumber) > ($thresh | tonumber)' <<< 'null' >/dev/null 2>&1; then
    gate_status="warn"
    gate_warnings="${gate_warnings}HIGH_CV: ${path_label} CV=${cv_val} exceeds threshold ${CV_THRESHOLD}\n"
  fi
done

# Cold compile detection: warn if first run init time is >3x the median
cold_compile="$(jq -s '
  def cold_check(label):
    map(select(. != null)) |
    if length >= 3 then
      .[0] as $first | (sort | .[((length - 1) / 2 | floor)]) as $med |
      if $med > 0 and ($first / $med) > 3 then
        "\(label) first=\($first)ms median=\($med)ms (\($first / $med | . * 100 | floor / 100)x)"
      else empty end
    else empty end;
  ([.[] | .control.init_wall_ms // null] | cold_check("control")),
  ([.[] | .two_step.init_wall_ms // null] | cold_check("two_step")),
  ([.[] | .coreml.reported_compile_ms // null] | cold_check("coreml"))
' "${valid_runs[@]}" 2>/dev/null || echo "")"
if [[ -n "$cold_compile" ]]; then
  gate_warnings="${gate_warnings}COLD_COMPILE: ${cold_compile}\n"
fi

# Git HEAD drift detection: warn if HEAD changed during benchmark
GIT_COMMIT_END="$(git -C "$ROOT" rev-parse HEAD)"
if [[ "$GIT_COMMIT_END" != "$GIT_COMMIT_START" ]]; then
  gate_status="warn"
  gate_warnings="${gate_warnings}GIT_DRIFT: HEAD changed during benchmark (start=${GIT_COMMIT_START} end=${GIT_COMMIT_END})\n"
fi

# Speedup floor: warn if any run shows two_step slower than coreml
if jq -e --arg min "$speedup_min" '($min | tonumber) < 1.0' <<< 'null' >/dev/null 2>&1; then
  gate_status="warn"
  gate_warnings="${gate_warnings}SPEEDUP_BELOW_1X: minimum two_step speedup=${speedup_min} (slower than coreml in at least one run)\n"
fi
if [[ "$control_speedup_min" != "n/a" ]] && jq -e --arg min "$control_speedup_min" '($min | tonumber) < 1.0' <<< 'null' >/dev/null 2>&1; then
  gate_status="warn"
  gate_warnings="${gate_warnings}CONTROL_SPEEDUP_BELOW_1X: minimum control speedup=${control_speedup_min} (slower than coreml in at least one run)\n"
fi

echo ""
echo "=== Reproducibility Gate ==="
echo "status=$gate_status"
if [[ -n "$gate_warnings" ]]; then
  printf "%b" "$gate_warnings"
fi
echo "cv_threshold=$CV_THRESHOLD"
echo "==="

{
  echo "gate_status=$gate_status"
  echo "cv_threshold=$CV_THRESHOLD"
  if [[ -n "$gate_warnings" ]]; then
    printf "%b" "$gate_warnings" | sed 's/^/gate_warning=/'
  fi
} >> "$RESULTS_DIR/summary.txt"

# Outlier detection: flag runs outside 1.5x IQR (Tukey fences) across all paths
outlier_count=0
outlier_json="{}"

tukey_detect() {
  local label="$1" jq_path="$2"
  # Inject source filename into each JSON before slurping (input_filename is broken in -s mode)
  local tagged_input
  tagged_input="$(for f in "${valid_runs[@]}"; do jq --arg src "$(basename "$f")" '. + {_src: $src}' "$f"; done)"
  echo "$tagged_input" | jq -s --arg label "$label" "
    [.[] | {file: ._src, val: ${jq_path}}] |
    [.[] | select(.val != null)] |
    sort_by(.val) |
    if length < 4 then {label: \$label, count: 0, outliers: [], fences: null}
    else
      (length) as \$n |
      (.[\$n / 4 | floor].val) as \$q1 |
      (.[\$n * 3 / 4 | floor].val) as \$q3 |
      (\$q3 - \$q1) as \$iqr |
      (\$q1 - 1.5 * \$iqr) as \$lo |
      (\$q3 + 1.5 * \$iqr) as \$hi |
      {label: \$label, count: ([.[] | select(.val < \$lo or .val > \$hi)] | length),
       outliers: [.[] | select(.val < \$lo or .val > \$hi)],
       fences: {q1: \$q1, q3: \$q3, iqr: \$iqr, lo: \$lo, hi: \$hi}}
    end
  "
}

if [[ ${#valid_runs[@]} -ge 4 ]]; then
  two_step_outliers="$(tukey_detect two_step '.two_step.median_ms_per_token')"
  control_outliers="$(tukey_detect control '.control.median_ms_per_token')"
  coreml_outliers="$(tukey_detect coreml '.coreml.median_ms_per_token')"
  speedup_outliers="$(tukey_detect speedup '.two_step_speedup_vs_coreml')"

  outlier_count=0
  for oj in "$two_step_outliers" "$control_outliers" "$coreml_outliers" "$speedup_outliers"; do
    c="$(echo "$oj" | jq '.count')"
    outlier_count=$((outlier_count + c))
  done

  outlier_json="$(jq -n \
    --argjson ts "$two_step_outliers" \
    --argjson ct "$control_outliers" \
    --argjson cm "$coreml_outliers" \
    --argjson sp "$speedup_outliers" \
    '{two_step: $ts, control: $ct, coreml: $cm, speedup: $sp}')"

  if [[ $outlier_count -gt 0 ]]; then
    if [[ "$gate_status" == "pass" ]]; then
      gate_status="warn"
    fi
    gate_warnings="${gate_warnings}OUTLIERS_DETECTED: ${outlier_count} outlier(s) across paths (Tukey 1.5x IQR)\n"
    echo ""
    echo "=== Outlier Detection ==="
    echo "total_outlier_count=$outlier_count"
    for path_name in two_step control coreml speedup; do
      path_count="$(echo "$outlier_json" | jq --arg p "$path_name" '.[$p].count')"
      if [[ "$path_count" -gt 0 ]]; then
        echo "${path_name}_outliers=$path_count"
        echo "$outlier_json" | jq -r --arg p "$path_name" '.[$p].outliers[] | "  \(.file): \(.val)"'
      fi
    done
    echo "==="
  fi
fi

# Merge gate status and outlier info into summary.json
gate_warnings_json="[]"
if [[ -n "$gate_warnings" ]]; then
  gate_warnings_json="$(printf '%b' "$gate_warnings" | sed '/^$/d' | jq -nR '[inputs | select(length > 0)]')"
fi
gate_json="$(jq -n \
  --arg status "$gate_status" \
  --arg cv_thresh "$CV_THRESHOLD" \
  --argjson outlier_count "$outlier_count" \
  --argjson warnings "$gate_warnings_json" \
  '{gate_status: $status, cv_threshold: ($cv_thresh | tonumber), outlier_count: $outlier_count, warnings: $warnings}')"
if [[ ${#valid_runs[@]} -ge 4 ]]; then
  gate_json="$(echo "$gate_json" | jq --argjson od "$outlier_json" '. + {outlier_detail: $od}')"
fi
jq --argjson gate "$gate_json" '. + {reproducibility: $gate}' "$RESULTS_DIR/summary.json" > "$RESULTS_DIR/summary.json.tmp" \
  && mv "$RESULTS_DIR/summary.json.tmp" "$RESULTS_DIR/summary.json"

# Final validation of summary.json after gate merge
if ! jq -e '.reproducibility.gate_status' "$RESULTS_DIR/summary.json" >/dev/null 2>&1; then
  echo "FATAL: summary.json corrupted during gate merge" >&2
  exit 1
fi

echo "Wrote raw JSON, stderr logs, and summary.json to $RESULTS_DIR"

# Exit code reflects gate status for CI integration:
#   0 = pass (or warn), 2 = fail (parity mismatch), 1 = reserved for runtime errors
if [[ "$gate_status" == "fail" ]]; then
  exit 2
fi
