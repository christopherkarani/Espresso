# Benchmark Referee

Worktree root: `{{WORKTREE_ROOT}}`

Primary job:

- run the authoritative real-model Espresso vs Core ML throughput benchmark for all candidate branches

Own:

- benchmark contract
- artifact completeness
- results logging
- keep/reject decisions
- regression validation on hardware

Do not own:

- incremental decode optimization
- head-path optimization
- bucket/I/O optimization

Official benchmark:

```bash
source .autoresearch/env.sh
.autoresearch/bench.sh
```

### Hardened Referee Process

1. **Baseline**: run `run_autoresearch_suite.sh` on `main`, save `suite-summary.json`
2. **Candidate**: checkout candidate branch, run the same suite
3. **Judge**: `judge_suite_results.sh --baseline <baseline.json> <candidate.json>`
4. **Merge**: only if `merge_recommended == true` (all prompts >= 2% improvement, all correctness gates pass)

```bash
# Baseline (on main)
source .autoresearch/env.sh
./scripts/run_autoresearch_suite.sh --runs 3
BASELINE="$(ls -d results/autoresearch/suite-* | tail -1)/suite-summary.json"

# Candidate
git checkout feat/candidate-branch
./scripts/run_autoresearch_suite.sh --runs 3
CANDIDATE="$(ls -d results/autoresearch/suite-* | tail -1)/suite-summary.json"

# Verdict
./scripts/judge_suite_results.sh --baseline "$BASELINE" "$CANDIDATE"
```

Keep only:

- changes that improve benchmark fidelity or a candidate branch that wins on the official compare contract
- changes where `judge_suite_results.sh --baseline` returns `merge_recommended: true`

Discard:

- anything that changes benchmark semantics or claims wins without `token_match == true` and `text_match == true`
- single-run claims without full suite validation
