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
} > "$RESULTS_DIR/metadata.txt"

echo "Building release probe into $SCRATCH_PATH"
swift build -c release --product espresso-multitoken-probe --scratch-path "$SCRATCH_PATH"

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

for run in $(seq 1 "$REPEATS"); do
  echo "Run $run/$REPEATS"
  "$PROBE" "${COMMON_ARGS[@]}" \
    > "$RESULTS_DIR/run-$run.json" \
    2> "$RESULTS_DIR/run-$run.stderr.log"
done

two_step_median_ms="$(jq -s 'map(.two_step.median_ms_per_token) | sort | .[((length - 1) / 2 | floor)]' "$RESULTS_DIR"/run-*.json)"
control_median_ms="$(jq -s 'map(.control.median_ms_per_token) | sort | .[((length - 1) / 2 | floor)]' "$RESULTS_DIR"/run-*.json)"
coreml_median_ms="$(jq -s 'map(.coreml.median_ms_per_token) | sort | .[((length - 1) / 2 | floor)]' "$RESULTS_DIR"/run-*.json)"
speedup_median="$(jq -s 'map(.two_step_speedup_vs_coreml) | sort | .[((length - 1) / 2 | floor)]' "$RESULTS_DIR"/run-*.json)"
committed_tokens_per_pass="$(jq -s 'map(.two_step.median_committed_exact_tokens_per_pass) | sort | .[((length - 1) / 2 | floor)]' "$RESULTS_DIR"/run-*.json)"
accepted_future_tokens_per_pass="$(jq -s 'map(.two_step.median_accepted_future_tokens_per_pass) | sort | .[((length - 1) / 2 | floor)]' "$RESULTS_DIR"/run-*.json)"
all_parity_match="$(jq -s 'all(.[]; .parity_status == "match")' "$RESULTS_DIR"/run-*.json)"

{
  echo "results_dir=$RESULTS_DIR"
  echo "two_step_median_ms_per_token=$two_step_median_ms"
  echo "control_median_ms_per_token=$control_median_ms"
  echo "coreml_median_ms_per_token=$coreml_median_ms"
  echo "two_step_speedup_vs_coreml=$speedup_median"
  echo "committed_exact_tokens_per_pass=$committed_tokens_per_pass"
  echo "accepted_future_tokens_per_pass=$accepted_future_tokens_per_pass"
  echo "all_parity_match=$all_parity_match"
} | tee "$RESULTS_DIR/summary.txt"

echo "Wrote raw JSON and stderr logs to $RESULTS_DIR"
