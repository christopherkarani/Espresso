#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./scripts/run_stories_generate_benchmark.sh [options] [prompt...]

Runs the exact Stories110M warm Espresso benchmark path:
  ./espresso generate --benchmark-generate ...

Options:
  --prompt TEXT         Prompt text. Trailing args are joined when omitted.
  --weights DIR         Stories weights directory.
  --tokenizer DIR       Stories tokenizer directory.
  --max-tokens N        Max generated tokens (default: 32)
  --warmup N            Warmup iterations (default: 1)
  --iterations N        Measured iterations (default: 3)
  --temperature VALUE   Sampling temperature (default: 0)
  --json                Emit JSON instead of text output.
  -h, --help            Show help.
EOF
}

DEFAULT_STORIES_DIR="$HOME/Library/Application Support/Espresso/demo/stories110m"
PROMPT="${ESPRESSO_AUTORESEARCH_PROMPT:-}"
WEIGHTS_DIR="${ESPRESSO_AUTORESEARCH_WEIGHTS_DIR:-$DEFAULT_STORIES_DIR}"
TOKENIZER_DIR="${ESPRESSO_AUTORESEARCH_TOKENIZER_DIR:-$DEFAULT_STORIES_DIR}"
MAX_TOKENS="${ESPRESSO_AUTORESEARCH_MAX_TOKENS:-32}"
WARMUP="${ESPRESSO_AUTORESEARCH_WARMUP:-1}"
ITERATIONS="${ESPRESSO_AUTORESEARCH_ITERATIONS:-3}"
TEMPERATURE="${ESPRESSO_TEMPERATURE:-0}"
JSON_FLAG=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --prompt)
      PROMPT="$2"
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
    --temperature)
      TEMPERATURE="$2"
      shift 2
      ;;
    --json)
      JSON_FLAG="--json"
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

exec ./espresso generate \
  --benchmark-generate \
  --model stories110m \
  --weights "$WEIGHTS_DIR" \
  --tokenizer "$TOKENIZER_DIR" \
  --compare-warmup "$WARMUP" \
  --compare-iterations "$ITERATIONS" \
  --max-tokens "$MAX_TOKENS" \
  --temperature "$TEMPERATURE" \
  ${JSON_FLAG:+$JSON_FLAG} \
  "$PROMPT"
