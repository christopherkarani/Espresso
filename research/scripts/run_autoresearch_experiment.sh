#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./scripts/run_autoresearch_experiment.sh --ref <git-ref> [options] [-- <suite args...>]

Creates a disposable detached worktree for a candidate ref, runs the referee
benchmark there, and automatically removes the worktree when the candidate
misses the keep gates. This mirrors an experiment-then-revert loop without
destructive resets in the main repo.

Options:
  --ref REF             Candidate git ref to benchmark. Required.
  --baseline FILE       Baseline suite-summary.json for keep/revert judgment.
  --worktree PATH       Worktree path. Defaults to .worktrees/autoresearch-<ref>-<timestamp>.
  --env-file PATH       Optional shell file to source before running the benchmark.
  --bench-script PATH   Benchmark script relative to the worktree root.
                        Default: ./scripts/run_autoresearch_suite.sh
  --threshold N         Required improvement percent for --baseline judgment. Default: 2
  --keep-on-fail        Preserve the worktree even when the candidate fails.
  -h, --help            Show help.
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

CANDIDATE_REF=""
BASELINE=""
WORKTREE=""
ENV_FILE=""
BENCH_SCRIPT="./scripts/run_autoresearch_suite.sh"
THRESHOLD=2
KEEP_ON_FAIL=0

POSITIONAL=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --ref)
      CANDIDATE_REF="$2"
      shift 2
      ;;
    --baseline)
      BASELINE="$2"
      shift 2
      ;;
    --worktree)
      WORKTREE="$2"
      shift 2
      ;;
    --env-file)
      ENV_FILE="$2"
      shift 2
      ;;
    --bench-script)
      BENCH_SCRIPT="$2"
      shift 2
      ;;
    --threshold)
      THRESHOLD="$2"
      shift 2
      ;;
    --keep-on-fail)
      KEEP_ON_FAIL=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      POSITIONAL+=("$@")
      break
      ;;
    *)
      POSITIONAL+=("$1")
      shift
      ;;
  esac
done

if [[ -z "$CANDIDATE_REF" ]]; then
  echo "Missing required --ref." >&2
  usage
  exit 1
fi

if [[ -n "$BASELINE" && ! -f "$BASELINE" ]]; then
  echo "Baseline file not found: $BASELINE" >&2
  exit 1
fi

if [[ -n "$ENV_FILE" ]]; then
  ENV_FILE="$(cd "$(dirname "$ENV_FILE")" && pwd)/$(basename "$ENV_FILE")"
  if [[ ! -f "$ENV_FILE" ]]; then
    echo "Env file not found: $ENV_FILE" >&2
    exit 1
  fi
fi

if [[ -z "$WORKTREE" ]]; then
  REF_SLUG="$(printf '%s' "$CANDIDATE_REF" | tr '/:@ ' '____')"
  WORKTREE="$REPO_ROOT/.worktrees/autoresearch-${REF_SLUG}-$(date +%Y%m%d-%H%M%S)"
fi

OUTPUT_DIR="$WORKTREE/results/autoresearch/experiment-$(date +%Y%m%d-%H%M%S)"
JUDGE_JSON="$OUTPUT_DIR/judge.json"
KEEP_WORKTREE=0

cleanup() {
  if [[ "$KEEP_WORKTREE" -eq 1 ]]; then
    return
  fi
  if git -C "$REPO_ROOT" worktree list --porcelain | grep -Fq "worktree $WORKTREE"; then
    git -C "$REPO_ROOT" worktree remove --force "$WORKTREE" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

mkdir -p "$(dirname "$WORKTREE")"
git -C "$REPO_ROOT" worktree add --detach "$WORKTREE" "$CANDIDATE_REF" >/dev/null

pushd "$WORKTREE" >/dev/null
if [[ -n "$ENV_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$ENV_FILE"
fi

BENCH_COMMAND=(
  "$BENCH_SCRIPT"
  --output-dir "$OUTPUT_DIR"
)
if [[ ${#POSITIONAL[@]} -gt 0 ]]; then
  BENCH_COMMAND+=("${POSITIONAL[@]}")
fi

echo "Running candidate benchmark in disposable worktree:" >&2
printf '  %q' "${BENCH_COMMAND[@]}" >&2
printf '\n' >&2
"${BENCH_COMMAND[@]}"
popd >/dev/null

CANDIDATE_SUMMARY="$OUTPUT_DIR/suite-summary.json"
if [[ ! -f "$CANDIDATE_SUMMARY" ]]; then
  echo "Candidate suite summary not found: $CANDIDATE_SUMMARY" >&2
  exit 1
fi

if [[ -z "$BASELINE" ]]; then
  KEEP_WORKTREE=1
  echo "candidate_summary=$CANDIDATE_SUMMARY"
  echo "verdict=no-baseline"
  echo "worktree_kept=$WORKTREE"
  exit 0
fi

"$WORKTREE/scripts/judge_suite_results.sh" \
  --baseline "$BASELINE" \
  --threshold "$THRESHOLD" \
  --json \
  "$CANDIDATE_SUMMARY" | tee "$JUDGE_JSON"

MERGE_RECOMMENDED="$(jq -r '.merge_recommended' "$JUDGE_JSON")"
if [[ "$MERGE_RECOMMENDED" == "true" ]]; then
  KEEP_WORKTREE=1
  echo "verdict=keep"
  echo "worktree_kept=$WORKTREE"
  exit 0
fi

if [[ "$KEEP_ON_FAIL" -eq 1 ]]; then
  KEEP_WORKTREE=1
  echo "verdict=revert-requested-but-preserved"
  echo "worktree_kept=$WORKTREE"
  exit 1
fi

echo "verdict=revert"
echo "worktree_removed=$WORKTREE"
exit 1
