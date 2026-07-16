#!/usr/bin/env bash
set -euo pipefail

# Judge script: evaluate suite results and optionally compare baseline vs candidate.
# Two modes:
#   1. Single summary:  ./scripts/judge_suite_results.sh <suite-summary.json>
#   2. Comparison:       ./scripts/judge_suite_results.sh --baseline <baseline.json> <candidate.json>

usage() {
  cat <<'EOF'
Usage:
  ./scripts/judge_suite_results.sh [options] <suite-summary.json>

Modes:
  Single summary (default):
    Reads suite-summary.json, prints metrics table, exits 0 if all gates pass, 1 otherwise.

  Comparison:
    --baseline <file>     Baseline suite-summary.json to compare against.
                          Computes per-prompt delta%, applies threshold, outputs verdict JSON.

Options:
  --threshold N           Minimum improvement % for merge recommendation (default: 2)
  --json                  Output machine-readable JSON verdict to stdout (comparison mode)
  -h, --help              Show help
EOF
}

# --- Dependency check ---
if ! command -v jq >/dev/null 2>&1; then
  echo "FATAL: jq is required. Install it and retry." >&2
  exit 1
fi

BASELINE=""
THRESHOLD=2
JSON_OUTPUT=0

POSITIONAL=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --baseline)    BASELINE="$2"; shift 2 ;;
    --threshold)   THRESHOLD="$2"; shift 2 ;;
    --json)        JSON_OUTPUT=1; shift ;;
    -h|--help)     usage; exit 0 ;;
    -*)            echo "Unknown option: $1" >&2; usage; exit 1 ;;
    *)             POSITIONAL+=("$1"); shift ;;
  esac
done

if [[ ${#POSITIONAL[@]} -ne 1 ]]; then
  echo "ERROR: Exactly one suite-summary.json file required." >&2
  usage
  exit 1
fi

CANDIDATE="${POSITIONAL[0]}"

if [[ ! -f "$CANDIDATE" ]]; then
  echo "FATAL: File not found: $CANDIDATE" >&2
  exit 1
fi

# ============================================================
# Mode 1: Single summary
# ============================================================
if [[ -z "$BASELINE" ]]; then
  echo "=== Suite Results ===" >&2
  jq -r '
    "Commit: \(.commit)",
    "Timestamp: \(.timestamp)",
    "Config: \(.config.runs) runs, warmup=\(.config.warmup), iterations=\(.config.iterations), max_tokens=\(.config.max_tokens)",
    "",
    "Per-prompt results:",
    (.per_prompt[] |
      "  \(.prompt_id):",
      "    Espresso: \(.espresso_tok_s.median) tok/s (min=\(.espresso_tok_s.min), max=\(.espresso_tok_s.max))",
      "    CoreML:   \(.coreml_tok_s.median) tok/s (min=\(.coreml_tok_s.min), max=\(.coreml_tok_s.max))",
      "    Speedup:  \(.speedup.median)x (min=\(.speedup.min)x, max=\(.speedup.max)x)",
      "    Token match: \(if .all_token_match then "PASS" else "FAIL" end)",
      "    Text match:  \(if .all_text_match then "PASS" else "FAIL" end)"
    ),
    "",
    "Aggregate:",
    "  Espresso median: \(.aggregate.espresso_tok_s_median) tok/s",
    "  CoreML median:   \(.aggregate.coreml_tok_s_median) tok/s",
    "  Speedup median:  \(.aggregate.speedup_median)x",
    "",
    "Verdict: \(if .verdict.all_correctness_gates_pass then "ALL GATES PASS" else "GATES FAILED" end)"
  ' "$CANDIDATE" >&2

  PASS="$(jq -r '.verdict.all_correctness_gates_pass' "$CANDIDATE")"
  if [[ "$PASS" == "true" ]]; then
    exit 0
  else
    exit 1
  fi
fi

# ============================================================
# Mode 2: Comparison (baseline vs candidate)
# ============================================================
if [[ ! -f "$BASELINE" ]]; then
  echo "FATAL: Baseline file not found: $BASELINE" >&2
  exit 1
fi

echo "=== Comparison: baseline vs candidate ===" >&2
echo "Baseline:  $BASELINE" >&2
echo "Candidate: $CANDIDATE" >&2
echo "Threshold: ${THRESHOLD}% improvement required on every prompt" >&2
echo "" >&2

# Build verdict JSON
VERDICT="$(jq -n \
  --slurpfile baseline "$BASELINE" \
  --slurpfile candidate "$CANDIDATE" \
  --argjson threshold "$THRESHOLD" '

  # Contract checks: same prompt set and same expected run count.
  ($baseline[0].config.runs // 0) as $expected_runs |
  ($candidate[0].config.runs // 0) as $candidate_runs |
  # Index baseline per_prompt by prompt_id
  ($baseline[0].per_prompt | map({(.prompt_id): .}) | add // {}) as $base_map |
  ($candidate[0].per_prompt | map({(.prompt_id): .}) | add // {}) as $cand_map |
  ($base_map | keys | sort) as $base_prompt_ids |
  ($cand_map | keys | sort) as $candidate_prompt_ids |
  (($base_prompt_ids | length) == ($candidate_prompt_ids | length) and
   ($base_prompt_ids == $candidate_prompt_ids)) as $prompt_sets_equal |

  $base_prompt_ids as $prompt_ids |

  # Per-prompt deltas
  [
    $prompt_ids[] | . as $pid |
    ($base_map[$pid] // null) as $b |
    ($cand_map[$pid] // null) as $c |
    if $b == null or $c == null then
      { prompt_id: $pid, error: "missing in baseline or candidate", meets_bar: false }
    else
      ($c.espresso_tok_s.median) as $c_val |
      ($b.espresso_tok_s.median) as $b_val |
      (if $b_val == 0 then null else (($c_val - $b_val) / $b_val * 100) end) as $delta_pct |
      {
        prompt_id: $pid,
        baseline_espresso_tok_s: $b_val,
        candidate_espresso_tok_s: $c_val,
        delta_pct: ($delta_pct // null),
        runs_expected: $expected_runs,
        candidate_runs: ($c.n_runs // 0),
        runs_match: ((($c.n_runs // 0) == $expected_runs) and ($candidate_runs == $expected_runs)),
        meets_bar: (if $delta_pct == null then false else $delta_pct >= $threshold end),
        baseline_token_match: $b.all_token_match,
        candidate_token_match: $c.all_token_match,
        baseline_text_match: $b.all_text_match,
        candidate_text_match: $c.all_text_match
      }
    end
  ] as $per_prompt_deltas |

  # Overall verdict
  {
    baseline_commit: $baseline[0].commit,
    candidate_commit: $candidate[0].commit,
    threshold_pct: $threshold,
    expected_runs: $expected_runs,
    prompt_sets_equal: $prompt_sets_equal,
    per_prompt: $per_prompt_deltas,
    all_prompts_meet_bar: ([$per_prompt_deltas[].meets_bar] | all),
    all_runs_match: ([$per_prompt_deltas[].runs_match] | all),
    all_correctness_gates_pass: (
      [$per_prompt_deltas[].candidate_token_match // false] | all
    ) and (
      [$per_prompt_deltas[].candidate_text_match // false] | all
    ),
    merge_recommended: (
      $prompt_sets_equal and
      ([$per_prompt_deltas[].meets_bar] | all) and
      ([$per_prompt_deltas[].candidate_token_match // false] | all) and
      ([$per_prompt_deltas[].candidate_text_match // false] | all) and
      $prompt_sets_equal and
      ([$per_prompt_deltas[].runs_match] | all)
    )
  }
')"

# Print human-readable summary to stderr
echo "$VERDICT" | jq -r '
  def fmt_pct:
    if . == null then "N/A"
    else (((. * 100) | round) / 100 | tostring) + "%"
    end;
  "Per-prompt comparison:",
  (.per_prompt[] |
    "  \(.prompt_id):",
    "    Baseline:  \(.baseline_espresso_tok_s // "N/A") tok/s",
    "    Candidate: \(.candidate_espresso_tok_s // "N/A") tok/s",
    "    Delta:     \(.delta_pct | fmt_pct)",
    "    Runs:      candidate \(.candidate_runs // "N/A") expected \(.runs_expected // "N/A")",
    "    Meets bar: \(if .meets_bar then "YES" else "NO" end)",
    "    Correctness: token=\(.candidate_token_match // "N/A"), text=\(.candidate_text_match // "N/A")"
  ),
  "",
  "Overall:",
  "  All prompts meet bar:        \(if .all_prompts_meet_bar then "YES" else "NO" end)",
  "  All correctness gates pass:  \(if .all_correctness_gates_pass then "YES" else "NO" end)",
  "  Prompt set match:            \(if .prompt_sets_equal then "YES" else "NO" end)",
  "  Runs match:                 \(if .all_runs_match then "YES" else "NO" end)",
  "  Merge recommended:           \(if .merge_recommended then "YES" else "NO" end)"
' >&2

# Output JSON verdict to stdout if requested or always in comparison mode
if [[ "$JSON_OUTPUT" -eq 1 ]]; then
  echo "$VERDICT"
else
  # Always output JSON verdict in comparison mode
  echo "$VERDICT"
fi

# Exit code based on merge recommendation
MERGE="$(echo "$VERDICT" | jq -r '.merge_recommended')"
if [[ "$MERGE" == "true" ]]; then
  exit 0
else
  exit 1
fi
