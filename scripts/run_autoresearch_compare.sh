#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./scripts/run_autoresearch_compare.sh [options] [prompt...]

Options:
  --prompt TEXT         Prompt text. Trailing args are joined when omitted.
  --max-tokens N        Max generated tokens (default: 128)
  --warmup N            Warmup iterations (default: 1)
  --iterations N        Measured iterations (default: 3)
  --output-dir DIR      Output directory. Defaults to results/autoresearch/compare-<timestamp>
  --results-tsv PATH    Append a summary row to PATH. Defaults to ./autoresearch-results.tsv when present.
  --model NAME          Explicit model registry key or alias
  --weights DIR         Explicit weights directory
  --tokenizer DIR       Explicit tokenizer directory
  --coreml-model PATH   Explicit Core ML model path
  --coreml-seq-len N    Fixed Core ML sequence length
  --compute-units NAME  Core ML compute units
  --prompt-id ID        Tag this run with a prompt identifier (flows into TSV)
  --power               Require power telemetry
  --no-power            Disable power telemetry (default)
  -h, --help            Show help
EOF
}

PROMPT="${ESPRESSO_AUTORESEARCH_PROMPT:-}"
MODEL_NAME="${ESPRESSO_AUTORESEARCH_MODEL:-}"
MAX_TOKENS="${ESPRESSO_AUTORESEARCH_MAX_TOKENS:-128}"
WARMUP="${ESPRESSO_AUTORESEARCH_WARMUP:-1}"
ITERATIONS="${ESPRESSO_AUTORESEARCH_ITERATIONS:-3}"
OUTPUT_DIR=""
RESULTS_TSV=""
WEIGHTS_DIR="${ESPRESSO_AUTORESEARCH_WEIGHTS_DIR:-}"
TOKENIZER_DIR="${ESPRESSO_AUTORESEARCH_TOKENIZER_DIR:-}"
COREML_MODEL="${ESPRESSO_AUTORESEARCH_COREML_MODEL:-}"
COREML_SEQ_LEN="${ESPRESSO_AUTORESEARCH_COREML_SEQ_LEN:-}"
COMPUTE_UNITS="${ESPRESSO_AUTORESEARCH_COREML_COMPUTE_UNITS:-}"
PROMPT_ID="${ESPRESSO_AUTORESEARCH_PROMPT_ID:-default}"
POWER_FLAG="--no-power"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --prompt)
      PROMPT="$2"
      shift 2
      ;;
    --max-tokens)
      MAX_TOKENS="$2"
      shift 2
      ;;
    --warmup)
      WARMUP="$2"
      shift 2
      ;;
    --iterations)
      ITERATIONS="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --results-tsv)
      RESULTS_TSV="$2"
      shift 2
      ;;
    --model)
      MODEL_NAME="$2"
      shift 2
      ;;
    --weights)
      WEIGHTS_DIR="$2"
      shift 2
      ;;
    --tokenizer)
      TOKENIZER_DIR="$2"
      shift 2
      ;;
    --coreml-model)
      COREML_MODEL="$2"
      shift 2
      ;;
    --coreml-seq-len)
      COREML_SEQ_LEN="$2"
      shift 2
      ;;
    --compute-units)
      COMPUTE_UNITS="$2"
      shift 2
      ;;
    --prompt-id)
      PROMPT_ID="$2"
      shift 2
      ;;
    --power)
      POWER_FLAG="--power"
      shift
      ;;
    --no-power)
      POWER_FLAG="--no-power"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      break
      ;;
    *)
      break
      ;;
  esac
done

if [[ $# -gt 0 ]]; then
  if [[ -n "$PROMPT" ]]; then
    echo "Prompt already set via --prompt or environment." >&2
    exit 1
  fi
  PROMPT="$*"
fi

if [[ -z "$PROMPT" ]]; then
  PROMPT="Hello"
fi

if [[ -z "$OUTPUT_DIR" ]]; then
  mkdir -p results/autoresearch
  OUTPUT_DIR="results/autoresearch/compare-$(date +%Y%m%d-%H%M%S)"
fi

if [[ -z "$RESULTS_TSV" && -f autoresearch-results.tsv ]]; then
  RESULTS_TSV="autoresearch-results.tsv"
fi

COMMAND=(
  ./espresso
  bench
  "$POWER_FLAG"
  --output-dir "$OUTPUT_DIR"
  --compare-warmup "$WARMUP"
  --compare-iterations "$ITERATIONS"
  --max-tokens "$MAX_TOKENS"
)

if [[ -n "$WEIGHTS_DIR" ]]; then
  COMMAND+=(--weights "$WEIGHTS_DIR")
fi
if [[ -n "$TOKENIZER_DIR" ]]; then
  COMMAND+=(--tokenizer "$TOKENIZER_DIR")
fi
if [[ -n "$MODEL_NAME" ]]; then
  COMMAND+=(--model "$MODEL_NAME")
fi
if [[ -n "$COREML_MODEL" ]]; then
  COMMAND+=(--coreml-model "$COREML_MODEL")
fi
if [[ -n "$COREML_SEQ_LEN" ]]; then
  COMMAND+=(--coreml-seq-len "$COREML_SEQ_LEN")
fi
if [[ -n "$COMPUTE_UNITS" ]]; then
  COMMAND+=(--coreml-compute-units "$COMPUTE_UNITS")
fi
COMMAND+=("$PROMPT")

echo "Running benchmark:" >&2
printf '  %q' "${COMMAND[@]}" >&2
printf '\n' >&2
"${COMMAND[@]}"

COMPARE_JSON="$OUTPUT_DIR/compare.json"
if [[ ! -f "$COMPARE_JSON" ]]; then
  echo "Missing compare report: $COMPARE_JSON" >&2
  exit 1
fi

if command -v jq >/dev/null 2>&1; then
  jq -r '
    "report_dir='"$OUTPUT_DIR"'",
    "espresso_tok_s=\(.espresso.tokens_per_second)",
    "coreml_tok_s=\(.coreml.tokens_per_second)",
    "speedup_vs_coreml=\((.espresso.tokens_per_second / (.coreml.tokens_per_second | if . == 0 then 1e-9 else . end)))",
    "token_match=\(.token_match)",
    "text_match=\(.text_match)",
    "espresso_first_token_ms=\(.espresso.first_token_latency_ms)",
    "coreml_first_token_ms=\(.coreml.first_token_latency_ms)",
    "espresso_median_token_ms=\(.espresso.median_token_ms)",
    "coreml_median_token_ms=\(.coreml.median_token_ms)",
    "espresso_p95_token_ms=\(.espresso.p95_token_ms)",
    "coreml_p95_token_ms=\(.coreml.p95_token_ms)",
    "espresso_compile_ms=\(.espresso.compile_time_ms)",
    "coreml_compile_ms=\(.coreml.compile_time_ms)"
  ' "$COMPARE_JSON"

  if [[ -n "$RESULTS_TSV" ]]; then
    if [[ ! -f "$RESULTS_TSV" ]]; then
      printf 'timestamp\tcommit\tstatus\tprimary_metric\tespresso_tokens_per_second\tcoreml_tokens_per_second\tspeedup_vs_coreml\ttoken_match\ttext_match\tespresso_first_token_ms\tcoreml_first_token_ms\tespresso_median_token_ms\tcoreml_median_token_ms\tespresso_p95_token_ms\tcoreml_p95_token_ms\tespresso_compile_ms\tcoreml_compile_ms\toutput_dir\tprompt_id\tchange_summary\n' >"$RESULTS_TSV"
    fi
    jq -r --arg timestamp "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
          --arg commit "$(git rev-parse --short HEAD)" \
          --arg output_dir "$OUTPUT_DIR" \
          --arg prompt_id "$PROMPT_ID" '
      [
        $timestamp,
        $commit,
        "measured",
        "espresso_tokens_per_second",
        .espresso.tokens_per_second,
        .coreml.tokens_per_second,
        (.espresso.tokens_per_second / (.coreml.tokens_per_second | if . == 0 then 1e-9 else . end)),
        .token_match,
        .text_match,
        .espresso.first_token_latency_ms,
        .coreml.first_token_latency_ms,
        .espresso.median_token_ms,
        .coreml.median_token_ms,
        .espresso.p95_token_ms,
        .coreml.p95_token_ms,
        .espresso.compile_time_ms,
        .coreml.compile_time_ms,
        $output_dir,
        $prompt_id,
        ""
      ] | @tsv
    ' "$COMPARE_JSON" >>"$RESULTS_TSV"
    echo "results_tsv=$RESULTS_TSV"
  fi
else
  echo "jq not found; skipping metric summary and TSV append." >&2
fi
