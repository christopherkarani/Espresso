#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./scripts/setup_autoresearch_lane.sh <lane> [options]

Lanes:
  benchmark-referee
  incremental-decode
  head-path
  bucket-io

Legacy aliases:
  output-head -> head-path
  exact-work -> incremental-decode
  ttft -> bucket-io

Options:
  --worktree PATH         Initialize an existing worktree at PATH.
  --create-worktree PATH  Create a new worktree at PATH, then initialize it.
  --branch NAME           Branch name to use with --create-worktree.
  --local-dir NAME        Local autoresearch directory name (default: .autoresearch).
  --results-file NAME     Results TSV filename (default: autoresearch-results.tsv).
  -h, --help              Show help.
EOF
}

escape_sed() {
  printf '%s' "$1" | sed -e 's/[\/&]/\\&/g'
}

require_lane() {
  case "$1" in
    benchmark-referee|incremental-decode|head-path|bucket-io|output-head|exact-work|ttft) ;;
    *)
      echo "Unknown lane: $1" >&2
      usage
      exit 1
      ;;
  esac
}

canonical_lane() {
  case "$1" in
    output-head) echo "head-path" ;;
    exact-work) echo "incremental-decode" ;;
    ttft) echo "bucket-io" ;;
    *) echo "$1" ;;
  esac
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

if [[ "$1" == "-h" || "$1" == "--help" ]]; then
  usage
  exit 0
fi

LANE="$1"
shift
require_lane "$LANE"
CANONICAL_LANE="$(canonical_lane "$LANE")"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TARGET_ROOT="$REPO_ROOT"
CREATE_WORKTREE=""
BRANCH_NAME=""
LOCAL_DIR=".autoresearch"
RESULTS_FILE="autoresearch-results.tsv"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --worktree)
      TARGET_ROOT="$2"
      shift 2
      ;;
    --create-worktree)
      CREATE_WORKTREE="$2"
      TARGET_ROOT="$2"
      shift 2
      ;;
    --branch)
      BRANCH_NAME="$2"
      shift 2
      ;;
    --local-dir)
      LOCAL_DIR="$2"
      shift 2
      ;;
    --results-file)
      RESULTS_FILE="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -n "$CREATE_WORKTREE" ]]; then
  if [[ -z "$BRANCH_NAME" ]]; then
    BRANCH_NAME="feat/autoresearch-${CANONICAL_LANE}-$(date +%Y%m%d-%H%M%S)"
  fi
  git -C "$REPO_ROOT" worktree add "$CREATE_WORKTREE" -b "$BRANCH_NAME"
fi

TARGET_ROOT="$(cd "$TARGET_ROOT" && pwd)"
git -C "$TARGET_ROOT" rev-parse --show-toplevel >/dev/null

TEMPLATE_PATH="$REPO_ROOT/docs/autoresearch/lanes/${CANONICAL_LANE}.md"
PROMPT_PATH="$REPO_ROOT/docs/autoresearch/prompts/${CANONICAL_LANE}.txt"

LOCAL_PATH="$TARGET_ROOT/$LOCAL_DIR"
PROGRAM_PATH="$LOCAL_PATH/program.md"
README_PATH="$LOCAL_PATH/README.md"
PROMPT_OUT="$LOCAL_PATH/PROMPT.txt"
ENV_OUT="$LOCAL_PATH/env.sh"
BENCH_OUT="$LOCAL_PATH/bench.sh"
RESULTS_PATH="$TARGET_ROOT/$RESULTS_FILE"

mkdir -p "$LOCAL_PATH"

ROOT_ESCAPED="$(escape_sed "$TARGET_ROOT")"
sed "s|{{WORKTREE_ROOT}}|$ROOT_ESCAPED|g" "$TEMPLATE_PATH" >"$PROGRAM_PATH"
cp "$PROMPT_PATH" "$PROMPT_OUT"

if [[ ! -f "$ENV_OUT" ]]; then
  cat >"$ENV_OUT" <<'EOF'
#!/usr/bin/env bash

# Pin the official autoresearch throughput contract here.
export ESPRESSO_AUTORESEARCH_PROMPT="${ESPRESSO_AUTORESEARCH_PROMPT:-}"
export ESPRESSO_AUTORESEARCH_MODEL="${ESPRESSO_AUTORESEARCH_MODEL:-gpt2_124m}"
export ESPRESSO_AUTORESEARCH_MAX_TOKENS="${ESPRESSO_AUTORESEARCH_MAX_TOKENS:-128}"
export ESPRESSO_AUTORESEARCH_WARMUP="${ESPRESSO_AUTORESEARCH_WARMUP:-3}"
export ESPRESSO_AUTORESEARCH_ITERATIONS="${ESPRESSO_AUTORESEARCH_ITERATIONS:-5}"
export ESPRESSO_AUTORESEARCH_SUITE_RUNS="${ESPRESSO_AUTORESEARCH_SUITE_RUNS:-3}"
export ESPRESSO_AUTORESEARCH_COLD_RUN="${ESPRESSO_AUTORESEARCH_COLD_RUN:-1}"
export ESPRESSO_AUTORESEARCH_PROMPTS="${ESPRESSO_AUTORESEARCH_PROMPTS:-scripts/benchmark-prompts.txt}"
export ESPRESSO_AUTORESEARCH_MERGE_THRESHOLD="${ESPRESSO_AUTORESEARCH_MERGE_THRESHOLD:-2}"
export ESPRESSO_AUTORESEARCH_WEIGHTS_DIR="${ESPRESSO_AUTORESEARCH_WEIGHTS_DIR:-$HOME/Library/Application Support/Espresso/demo/gpt2_124m}"
export ESPRESSO_AUTORESEARCH_TOKENIZER_DIR="${ESPRESSO_AUTORESEARCH_TOKENIZER_DIR:-$HOME/Library/Application Support/Espresso/demo/gpt2_tokenizer}"
export ESPRESSO_AUTORESEARCH_COREML_MODEL="${ESPRESSO_AUTORESEARCH_COREML_MODEL:-}"
export ESPRESSO_AUTORESEARCH_COREML_SEQ_LEN="${ESPRESSO_AUTORESEARCH_COREML_SEQ_LEN:-}"
export ESPRESSO_AUTORESEARCH_COREML_COMPUTE_UNITS="${ESPRESSO_AUTORESEARCH_COREML_COMPUTE_UNITS:-cpu_and_neural_engine}"
EOF
  chmod +x "$ENV_OUT"
fi

cat >"$BENCH_OUT" <<EOF
#!/usr/bin/env bash
set -euo pipefail

ROOT="\$(cd "\$(dirname "\${BASH_SOURCE[0]}")/.." && pwd)"
cd "\$ROOT"

source "\$ROOT/$LOCAL_DIR/env.sh"

cmd=(
  ./scripts/run_autoresearch_suite.sh
  --runs "\${ESPRESSO_AUTORESEARCH_SUITE_RUNS:-3}"
  --warmup "\$ESPRESSO_AUTORESEARCH_WARMUP"
  --iterations "\$ESPRESSO_AUTORESEARCH_ITERATIONS"
  --max-tokens "\$ESPRESSO_AUTORESEARCH_MAX_TOKENS"
  --prompts "\${ESPRESSO_AUTORESEARCH_PROMPTS:-scripts/benchmark-prompts.txt}"
)

if [[ "\${ESPRESSO_AUTORESEARCH_COLD_RUN:-1}" -eq 0 ]]; then
  cmd+=(--no-cold)
fi

if [[ -n "\${ESPRESSO_AUTORESEARCH_MODEL:-}" ]]; then
  cmd+=(--model "\$ESPRESSO_AUTORESEARCH_MODEL")
fi
if [[ -n "\${ESPRESSO_AUTORESEARCH_WEIGHTS_DIR:-}" ]]; then
  cmd+=(--weights "\$ESPRESSO_AUTORESEARCH_WEIGHTS_DIR")
fi
if [[ -n "\${ESPRESSO_AUTORESEARCH_TOKENIZER_DIR:-}" ]]; then
  cmd+=(--tokenizer "\$ESPRESSO_AUTORESEARCH_TOKENIZER_DIR")
fi
if [[ -n "\${ESPRESSO_AUTORESEARCH_COREML_MODEL:-}" ]]; then
  cmd+=(--coreml-model "\$ESPRESSO_AUTORESEARCH_COREML_MODEL")
fi
if [[ -n "\${ESPRESSO_AUTORESEARCH_COREML_SEQ_LEN:-}" ]]; then
  cmd+=(--coreml-seq-len "\$ESPRESSO_AUTORESEARCH_COREML_SEQ_LEN")
fi
if [[ -n "\${ESPRESSO_AUTORESEARCH_COREML_COMPUTE_UNITS:-}" ]]; then
  cmd+=(--compute-units "\$ESPRESSO_AUTORESEARCH_COREML_COMPUTE_UNITS")
fi

exec "\${cmd[@]}"
EOF
chmod +x "$BENCH_OUT"

cat >"$README_PATH" <<EOF
# Autoresearch

Local autoresearch sandbox for lane \`$LANE\`.

- Program: \`$LOCAL_DIR/program.md\`
- Prompt: \`$LOCAL_DIR/PROMPT.txt\`
- Env: \`$LOCAL_DIR/env.sh\`
- Benchmark wrapper: \`$LOCAL_DIR/bench.sh\`
- Results log: \`$RESULTS_FILE\`

Tracked references:

- \`docs/autoresearch/README.md\`
- \`docs/autoresearch/lanes/${CANONICAL_LANE}.md\`

Local workflow:

1. Fill or override \`$LOCAL_DIR/env.sh\`
2. Read \`$LOCAL_DIR/program.md\`
3. Paste \`$LOCAL_DIR/PROMPT.txt\` into your agent
4. Run \`$LOCAL_DIR/bench.sh\` if this worktree is the benchmark referee lane
EOF

if [[ ! -f "$RESULTS_PATH" ]]; then
  cat >"$RESULTS_PATH" <<'EOF'
timestamp	commit	status	primary_metric	espresso_tokens_per_second	coreml_tokens_per_second	speedup_vs_coreml	token_match	text_match	espresso_first_token_ms	coreml_first_token_ms	espresso_median_token_ms	coreml_median_token_ms	espresso_p95_token_ms	coreml_p95_token_ms	espresso_compile_ms	coreml_compile_ms	output_dir	change_summary
EOF
fi

EXCLUDE_FILE="$(git -C "$TARGET_ROOT" rev-parse --git-path info/exclude)"
mkdir -p "$(dirname "$EXCLUDE_FILE")"
touch "$EXCLUDE_FILE"

for pattern in "$LOCAL_DIR/" "$RESULTS_FILE"; do
  if ! grep -qxF "$pattern" "$EXCLUDE_FILE"; then
    echo "$pattern" >>"$EXCLUDE_FILE"
  fi
done

cat <<EOF
Initialized autoresearch lane '$LANE' in:
  $TARGET_ROOT

Created:
  $PROGRAM_PATH
  $README_PATH
  $PROMPT_OUT
  $ENV_OUT
  $BENCH_OUT
  $RESULTS_PATH

Next:
  cd $TARGET_ROOT
  cat $PROMPT_OUT
  cat $ENV_OUT
  $BENCH_OUT
EOF
