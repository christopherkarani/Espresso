# autoresearch-espresso

Autonomous experimentation on ANE serving optimizations for the retained **stories110m** release lane.

The harness is inference-first. It optimizes the hardened suite median `tok/s` while enforcing the retained bundle, prompt suite, and baseline quality/performance gates.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (for example `apr8`). The branch `autoresearch-espresso/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autoresearch-espresso/<tag>` from current main.
3. **Read the in-scope files**: focus on these areas:
   - `Sources/ANEBuilder/` — kernel graph builders
   - `Sources/ANEPasses/` — optimization passes
   - `Sources/ANERuntime/` — compilation, eval, decode kernels
   - `Sources/Espresso/DecodeForwardPass.swift` — decode path policy and hot loop behavior
   - `Sources/MILGenerator/` — MIL emission patterns
   - `Sources/EspressoGenerate/` — hardened suite benchmark CLI
   - `scripts/run_autoresearch_suite.sh` — suite wrapper
   - `autoresearch-espresso/experiment_runner.py` — fixed evaluation harness
4. **Verify the release build**: `swift build --product espresso-generate -c release`
5. **Verify the retained benchmark inputs exist**:
   - `.build/release-bundles/stories110m-smoke.esp`
   - `scripts/stories_release_benchmark_prompts.txt`
   - the retained baseline `suite-summary.json`
   - the local Core ML package for the qualified compare lane
6. **Establish baseline**: run `python autoresearch-espresso/experiment_runner.py benchmark`
7. **Confirm and go**: once the harness works, start the loop.

## Experimentation

Each experiment modifies Swift source, rebuilds the release binary, and runs the hardened suite contract. Launch it with:

```bash
python autoresearch-espresso/experiment_runner.py full --description "<change>"
```

The harness uses:
- the retained Stories `.esp` bundle
- the retained Stories prompt suite
- fixed `max_tokens`, `runs`, `warmup`, and `iterations`
- baseline regression thresholds for `tok/s`, TTFT, median token latency, and p95 token latency

**What you CAN do:**
- Modify Swift source in `Sources/ANEBuilder/`, `Sources/ANEPasses/`, `Sources/ANERuntime/`, `Sources/Espresso/`, and `Sources/MILGenerator/`
- Change decode strategy defaults, fusion behavior, cache layout, compilation policy, and serving-path runtime structure

**What you CANNOT do during the loop:**
- Modify `autoresearch-espresso/experiment_runner.py`
- Change the retained benchmark bundle, prompt suite, baseline summary, or regression thresholds
- Modify `Package.swift` to add dependencies
- Change model dimensions as part of the serving benchmark contract

## Objective

The goal is:

1. maximize suite median `tok/s`
2. preserve exact token/text parity across the suite
3. stay within the retained latency regression gates

This is not a generic training-research loop. It is a serving optimization loop.

## Quality And Performance Gates

An experiment is a **discard** if any of these happen:
- token parity fails on any prompt
- text parity fails on any prompt
- suite correctness gates fail
- suite performance gates fail

An experiment is a **keep** only if:
- correctness gates pass
- performance gates pass
- suite median `tok/s` beats the best retained `keep` run

## Output format

The harness prints the retained serving metrics, for example:

```text
[suite] tok/s=77.69 ttft_ms=1.72 median_token_ms=13.86 p95_token_ms=15.50
[quality] token_match=PASS text_match=PASS correctness_gates=PASS performance_gates=PASS
[baseline] merge_recommended=YES
[artifacts] suite_dir=/abs/path/to/results/autoresearch/suite-...
[results] commit=abc1234 previous_best_tok_s=76.10 suggested_status=keep
```

## Results logging

The harness automatically appends to:

```text
autoresearch-espresso/suite-results.tsv
```

Schema:

```text
timestamp	commit	status	espresso_tokens_per_second	espresso_ttft_ms	espresso_median_token_ms	espresso_p95_token_ms	all_token_match	all_text_match	correctness_gates_pass	performance_gates_pass	merge_recommended	output_dir	baseline_summary	change_summary
```

This file is the source of truth for keep/discard decisions in the loop.

## The loop

LOOP FOREVER:

1. Check current branch and commit
2. Make one concrete serving-path change
3. `git commit -m "experiment: <description>"`
4. Run:
   `python autoresearch-espresso/experiment_runner.py full --description "<description>" > run.log 2>&1`
5. Read:
   `grep "\[suite\]\|\[quality\]\|\[results\]" run.log`
6. If the run crashed, inspect:
   `tail -n 50 run.log`
7. If the harness suggests `keep`, continue from that commit
8. If the harness suggests `discard` or `crash`, reset back and try another idea

## Timeout policy

- A full release build plus suite run should stay within roughly 15 minutes
- If it exceeds the harness timeout, treat it as a crash

## Good experiment targets

High leverage:
- decode kernel set selection
- lane spatial defaults
- fused exact-head behavior
- KV/cache layout and reuse
- Metal vs ANE attention split
- pass pipeline improvements that reduce emitted graph overhead
- compile/cache reuse on the serving lane

Lower leverage:
- generic refactors without a serving-path effect
- documentation-only changes
- training-only improvements

## Principles

- prefer changes that improve the retained shipping lane, not just a synthetic microbench
- prefer same-binary comparisons
- prefer smaller, attributable changes over broad rewrites
- keep the harness contract fixed while experimenting
