#!/usr/bin/env bash
set -euo pipefail

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

if [[ "$REPEATS" -lt 3 || $((REPEATS % 2)) -ne 1 ]]; then
  echo "REPEATS must be an odd integer >= 3" >&2
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

{
  echo "timestamp=$(date -Iseconds)"
  echo "git_commit=$(git -C "$ROOT" rev-parse HEAD)"
  echo "git_branch=$(git -C "$ROOT" rev-parse --abbrev-ref HEAD)"
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
  # System environment snapshot for regression diagnosis
  echo "chip=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo unknown)"
  echo "hw_model=$(sysctl -n hw.model 2>/dev/null || echo unknown)"
  echo "physical_memory_gb=$(( $(sysctl -n hw.memsize 2>/dev/null || echo 0) / 1073741824 ))"
  echo "ncpu=$(sysctl -n hw.ncpu 2>/dev/null || echo unknown)"
  echo "thermal_pressure=$(pmset -g therm 2>/dev/null | grep -i 'cpu.*speed' | head -1 || echo unknown)"
  echo "load_average=$(sysctl -n vm.loadavg 2>/dev/null || echo unknown)"
} > "$RESULTS_DIR/metadata.txt"

echo "Building release probe into $SCRATCH_PATH"
swift build -c release --product espresso-multitoken-probe --scratch-path "$SCRATCH_PATH"

# Verify jq is available (required for aggregation)
if ! command -v jq &>/dev/null; then
  echo "FATAL: jq is required but not found on PATH" >&2
  exit 1
fi

if [[ "$DRY_RUN" == "1" ]]; then
  echo "=== Dry Run: All prerequisites validated ==="
  echo "probe=$PROBE"
  echo "coreml_model=$COREML_MODEL"
  echo "jq=$(command -v jq)"
  echo "input_mode=$INPUT_MODE"
  echo "repeats=$REPEATS warmup=$WARMUP iterations=$ITERATIONS"
  echo "layer_count=$LAYER_COUNT max_new_tokens=$MAX_NEW_TOKENS"
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
  echo "  elapsed: ${run_elapsed}s (exit=$run_exit)"
  echo "$run_elapsed" > "$RESULTS_DIR/run-$run.elapsed_s"
  if [[ $run_exit -ne 0 ]]; then
    echo "WARNING: Run $run exited with code $run_exit" >&2
    echo "  stderr tail: $(tail -3 "$RESULTS_DIR/run-$run.stderr.log")" >&2
    failed_runs=$((failed_runs + 1))
  elif ! jq -e '.two_step.median_ms_per_token' "$RESULTS_DIR/run-$run.json" >/dev/null 2>&1; then
    echo "WARNING: Run $run produced invalid JSON (missing two_step.median_ms_per_token)" >&2
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
for f in "$RESULTS_DIR"/run-*.json; do
  if jq -e '.two_step.median_ms_per_token' "$f" >/dev/null 2>&1; then
    valid_runs+=("$f")
  fi
done

if [[ ${#valid_runs[@]} -eq 0 ]]; then
  echo "FATAL: No valid run JSONs found in $RESULTS_DIR" >&2
  exit 1
fi
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

{
  echo "results_dir=$RESULTS_DIR"
  echo "requested_repeats=$REPEATS"
  echo "valid_runs=${#valid_runs[@]}"
  echo "failed_runs=$failed_runs"
  echo "two_step_median_ms_per_token=$two_step_median_ms"
  echo "two_step_p95_ms_per_token=$two_step_p95_ms"
  echo "two_step_p99_ms_per_token=$two_step_p99_ms"
  echo "two_step_min_ms_per_token=$two_step_min_ms"
  echo "two_step_max_ms_per_token=$two_step_max_ms"
  echo "two_step_cv=$two_step_cv"
  echo "control_median_ms_per_token=$control_median_ms"
  echo "control_p95_ms_per_token=$control_p95_ms"
  echo "control_p99_ms_per_token=$control_p99_ms"
  echo "control_min_ms_per_token=$control_min_ms"
  echo "control_max_ms_per_token=$control_max_ms"
  echo "control_cv=$control_cv"
  echo "coreml_median_ms_per_token=$coreml_median_ms"
  echo "coreml_p95_ms_per_token=$coreml_p95_ms"
  echo "coreml_p99_ms_per_token=$coreml_p99_ms"
  echo "coreml_min_ms_per_token=$coreml_min_ms"
  echo "coreml_max_ms_per_token=$coreml_max_ms"
  echo "coreml_cv=$coreml_cv"
  echo "two_step_speedup_vs_coreml=$speedup_median"
  echo "two_step_speedup_min=$(jq -s 'map(.two_step_speedup_vs_coreml) | min' "${valid_runs[@]}")"
  echo "two_step_speedup_max=$(jq -s 'map(.two_step_speedup_vs_coreml) | max' "${valid_runs[@]}")"
  echo "two_step_speedup_cv=$speedup_cv"
  echo "total_elapsed_s=$total_benchmark_elapsed"
  echo "committed_exact_tokens_per_pass=$committed_tokens_per_pass"
  echo "accepted_future_tokens_per_pass=$accepted_future_tokens_per_pass"
  echo "all_parity_match=$all_parity_match"
} | tee "$RESULTS_DIR/summary.txt"

# Machine-readable aggregate JSON combining all per-run data with summary stats
jq -s \
  --arg dir "$RESULTS_DIR" \
  --arg ts "$(date -Iseconds)" \
  --arg commit "$(git -C "$ROOT" rev-parse HEAD)" \
  --arg branch "$(git -C "$ROOT" rev-parse --abbrev-ref HEAD)" \
  --arg input_mode "$INPUT_MODE" \
  --argjson warmup "$WARMUP" \
  --argjson iterations "$ITERATIONS" \
  --argjson max_new_tokens "$MAX_NEW_TOKENS" \
  --argjson max_seq "$MAX_SEQUENCE_TOKENS" \
  --argjson layers "$LAYER_COUNT" \
  --arg control_backend "$CONTROL_BACKEND" \
  --arg two_step_backend "$TWO_STEP_BACKEND" \
  --argjson requested_repeats "$REPEATS" \
  --argjson failed "$failed_runs" \
  --argjson total_elapsed_s "$total_benchmark_elapsed" \
  --arg hw_model "$(sysctl -n hw.model 2>/dev/null || echo unknown)" \
  --arg load_avg "$(sysctl -n vm.loadavg 2>/dev/null || echo unknown)" \
'{
  results_dir: $dir,
  timestamp: $ts,
  git_commit: $commit,
  git_branch: $branch,
  host: {hw_model: $hw_model, load_average: $load_avg},
  benchmark_contract: {
    input_mode: $input_mode,
    control_backend: $control_backend,
    two_step_backend: $two_step_backend,
    warmup: $warmup,
    iterations: $iterations,
    max_new_tokens: $max_new_tokens,
    max_sequence_tokens: $max_seq,
    layer_count: $layers
  },
  requested_repeats: $requested_repeats,
  valid_runs: (length),
  failed_runs: $failed,
  total_elapsed_s: $total_elapsed_s,
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
    iqr_ms: (map(.two_step.median_ms_per_token) | sort | if length < 4 then (last - first) else (.[((length * 3 / 4) | floor)] - .[((length / 4) | floor)]) end)
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
    iqr_ms: (map(.control.median_ms_per_token) | sort | if length < 4 then (last - first) else (.[((length * 3 / 4) | floor)] - .[((length / 4) | floor)]) end)
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
    iqr_ms: (map(.coreml.median_ms_per_token) | sort | if length < 4 then (last - first) else (.[((length * 3 / 4) | floor)] - .[((length / 4) | floor)]) end)
  },
  two_step_speedup_vs_coreml: (map(.two_step_speedup_vs_coreml) | sort | .[((length - 1) / 2 | floor)]),
  two_step_speedup_min: (map(.two_step_speedup_vs_coreml) | min),
  two_step_speedup_max: (map(.two_step_speedup_vs_coreml) | max),
  two_step_speedup_mean: (map(.two_step_speedup_vs_coreml) | add / length),
  two_step_speedup_stddev: (map(.two_step_speedup_vs_coreml) | (length) as $n | (add / $n) as $mean | (map(. - $mean | . * .) | add / $n | sqrt)),
  two_step_speedup_cv: (map(.two_step_speedup_vs_coreml) | (length) as $n | (add / $n) as $mean | if $mean == 0 then 0 else (map(. - $mean | . * .) | add / $n | sqrt) / $mean end),
  control_speedup_vs_coreml: (map(.control_speedup_vs_coreml // null) | if all(. != null) then sort | .[((length - 1) / 2 | floor)] else null end),
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
  per_run_timestamps: (map(.probe_timestamp // null))
}' "${valid_runs[@]}" > "$RESULTS_DIR/summary.json"

# Reproducibility gate: warn on high cross-run variance or parity failure
CV_THRESHOLD="${CV_THRESHOLD:-0.10}"
gate_status="pass"
gate_warnings=""

if [[ "$all_parity_match" != "true" ]]; then
  gate_status="fail"
  gate_warnings="${gate_warnings}PARITY_MISMATCH: not all runs produced matching tokens\n"
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

# Outlier detection: flag runs outside 1.5x IQR (Tukey fences)
outlier_count=0
outlier_runs=""
if [[ ${#valid_runs[@]} -ge 4 ]]; then
  outlier_info="$(jq -s '
    [.[] | {file: input_filename, val: .two_step.median_ms_per_token}] |
    sort_by(.val) |
    (length) as $n |
    (.[($n / 4 | floor)].val) as $q1 |
    (.[($n * 3 / 4 | floor)].val) as $q3 |
    ($q3 - $q1) as $iqr |
    ($q1 - 1.5 * $iqr) as $lo |
    ($q3 + 1.5 * $iqr) as $hi |
    {q1: $q1, q3: $q3, iqr: $iqr, lo_fence: $lo, hi_fence: $hi,
     outliers: [.[] | select(.val < $lo or .val > $hi)]}
  ' "${valid_runs[@]}")"
  outlier_count="$(echo "$outlier_info" | jq '.outliers | length')"
  if [[ $outlier_count -gt 0 ]]; then
    outlier_runs="$(echo "$outlier_info" | jq -r '.outliers[] | "\(.file): \(.val) ms/token"')"
    echo ""
    echo "=== Outlier Detection ==="
    echo "outlier_count=$outlier_count"
    echo "$outlier_runs"
    echo "fences: $(echo "$outlier_info" | jq -r '"[\(.lo_fence), \(.hi_fence)]"')"
    echo "==="
  fi
fi

# Merge gate status and outlier info into summary.json
gate_json="$(jq -n \
  --arg status "$gate_status" \
  --arg cv_thresh "$CV_THRESHOLD" \
  --argjson outlier_count "$outlier_count" \
  '{gate_status: $status, cv_threshold: ($cv_thresh | tonumber), outlier_count: $outlier_count}')"
if [[ ${#valid_runs[@]} -ge 4 && -n "$outlier_info" ]]; then
  gate_json="$(echo "$gate_json" | jq --argjson oi "$outlier_info" '. + {outlier_fences: {lo: $oi.lo_fence, hi: $oi.hi_fence, q1: $oi.q1, q3: $oi.q3, iqr: $oi.iqr}}')"
fi
jq --argjson gate "$gate_json" '. + {reproducibility: $gate}' "$RESULTS_DIR/summary.json" > "$RESULTS_DIR/summary.json.tmp" \
  && mv "$RESULTS_DIR/summary.json.tmp" "$RESULTS_DIR/summary.json"

echo "Wrote raw JSON, stderr logs, and summary.json to $RESULTS_DIR"
