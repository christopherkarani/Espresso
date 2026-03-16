# Autoresearch

Tracked setup for real-model throughput work on Espresso vs Core ML.

This is intentionally anchored to the shipped GPT-2 compare path on `main`, not the older local-artifact decode benchmark still described in the top-level README.

## Source Of Truth

Use `./scripts/run_autoresearch_compare.sh` as the official benchmark entrypoint.

It wraps `./espresso bench` and exports:

- `compare.json`
- `summary.csv`
- `summary.md`
- `espresso_token_latencies.csv`
- `coreml_token_latencies.csv`

Primary scoreboard:

- `speedup_vs_coreml = espresso.tokens_per_second / coreml.tokens_per_second`

Primary Espresso optimization metric:

- `espresso.tokens_per_second`

Hard gates:

- `token_match == true`
- `text_match == true`

Guardrails:

- `espresso.first_token_latency_ms`
- `espresso.compile_time_ms`
- `espresso.median_token_ms`
- `espresso.p95_token_ms`
- no benchmark contract drift

## Current Throughput Focus

The current real-model GPT-2 path is dominated by three concrete costs in `Sources/RealModelInference/RealModelInferenceEngine.swift`:

- full prompt embedding rebuild plus full input-surface rewrite every token
- full-sequence attention and FFN over the active bucket every token
- full hidden-state readback plus CPU logits every token

Those are the surfaces the implementation lanes target.

The compare bootstrap path still depends on GPT-2 helper scripts referenced by `GPT2DemoSupport.swift`. If your local checkout does not have those helpers available, pin explicit `weights`, `tokenizer`, and `coreml-model` paths in `.autoresearch/env.sh` so the benchmark contract remains runnable.

## Operating Model

- Run exactly one hardware benchmark referee at a time.
- Run implementation agents in separate worktrees.
- Implementation agents do not claim official throughput wins.
- Only referee measurements count.

Recommended lanes:

- `benchmark-referee`
- `incremental-decode`
- `head-path`
- `bucket-io`

## Why This Model

Parallel ANE benchmarks collide on the same hardware and make tokens/sec numbers noisy. The clean pattern is:

1. implementation agents make candidate changes in isolation
2. the benchmark referee validates those changes serially on hardware
3. only referee-validated wins get merged

## Scaffold A Lane

From the repo root or any Espresso worktree:

```bash
./scripts/setup_autoresearch_lane.sh benchmark-referee
./scripts/setup_autoresearch_lane.sh incremental-decode
./scripts/setup_autoresearch_lane.sh head-path
./scripts/setup_autoresearch_lane.sh bucket-io
```

To create and initialize a new worktree in one step:

```bash
./scripts/setup_autoresearch_lane.sh incremental-decode \
  --create-worktree .worktrees/autoresearch-incremental-decode-$(date +%Y%m%d-%H%M%S) \
  --branch feat/autoresearch-incremental-decode
```

The script creates a local, ignored `.autoresearch/` directory in the target worktree with:

- `README.md`
- `program.md`
- `PROMPT.txt`
- `env.sh`
- `bench.sh`

It also creates an ignored `autoresearch-results.tsv` log keyed to the real-model throughput metrics.

## Hardened Contract

The benchmark harness is hardened for trustworthy multi-prompt, multi-run measurement:

- **Prompt suite**: `scripts/benchmark-prompts.txt` defines 3 fixed prompts (short/medium/long) exercising different CoreML sequence-length buckets
- **Suite runner**: `scripts/run_autoresearch_suite.sh` runs 1 cold + K warm runs across all prompts, aggregates median/min/max per prompt
- **Judge script**: `scripts/judge_suite_results.sh` evaluates single summaries or compares baseline vs candidate with threshold gating

### Merge Bar

A candidate branch is merge-eligible only when:

1. All runs pass `token_match == true` and `text_match == true`
2. Candidate shows >= 2% improvement on `espresso_tok_s` median on **every** prompt
3. The bar holds across all K suite runs (default 3/3)
4. Only `judge_suite_results.sh --baseline` verdicts count

### Suite Workflow

```bash
# 1. Run baseline suite on main
./scripts/run_autoresearch_suite.sh --runs 3
cp results/autoresearch/suite-*/suite-summary.json /tmp/baseline.json

# 2. Switch to candidate branch, run suite
git checkout feat/my-optimization
./scripts/run_autoresearch_suite.sh --runs 3

# 3. Judge
./scripts/judge_suite_results.sh --baseline /tmp/baseline.json \
  results/autoresearch/suite-*/suite-summary.json
```

### Known Limitations

- **"Cold" is process-cold, not machine-cold.** ANE kernel caches in `~/Library/Caches/` persist across runs. The cold run warms the OS cache, not the ANE cache.
- **jq required.** Aggregation needs jq. Scripts fail hard with a clear message if absent.
- **Suite runtime.** 1 cold + 9 warm invocations (~5 min at ~30s each). Acceptable for referee-grade measurement.
- **No prompt-length normalization.** Each prompt gets its own CoreML sequence length bucket via existing `nextPowerOfTwo` logic. This is intentional — it tests real behavior.

## Lane Templates

- [benchmark-referee](lanes/benchmark-referee.md)
- [incremental-decode](lanes/incremental-decode.md)
- [head-path](lanes/head-path.md)
- [bucket-io](lanes/bucket-io.md)
