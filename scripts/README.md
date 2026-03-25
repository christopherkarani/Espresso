# Scripts

This folder contains stable scripts that developers can run to reproduce models and benchmark claims.

Tracked scripts are intentionally limited to:

- `ensure_coreml_model.sh`
- `generate_coreml_model.py`
- `reproduce_local_real_artifact_claim.sh`
- `run_power_benchmark.sh`
- `run_autoresearch_compare.sh`
- `run_autoresearch_suite.sh`
- `judge_suite_results.sh`
- `run_anemll_coreml_benchmark.py`
- `stories_model_identity.py`
- `benchmark-prompts.txt`
- `setup_autoresearch_lane.sh`

For one-off or personal helper scripts, use `scripts/internal/` locally. That area is ignored by git and does not appear on GitHub.

## Autoresearch

Use `setup_autoresearch_lane.sh` to initialize a worktree-local `.autoresearch/` sandbox and ignored `autoresearch-results.tsv` for a lane such as:

- `benchmark-referee`
- `incremental-decode`
- `head-path`
- `bucket-io`

Use `run_autoresearch_compare.sh` as the source-of-truth real-model throughput benchmark wrapper for autoresearch lanes. It drives `./espresso bench`, exports the compare report, and appends a results row when `--results-tsv` is passed or when `autoresearch-results.tsv` is present in the current directory. Supports `--prompt-id` to tag runs by prompt.

Use `run_autoresearch_suite.sh` to run the full hardened benchmark suite: multiple prompts from `benchmark-prompts.txt` across multiple runs, with cold-run gating and per-prompt median/min/max aggregation into `suite-summary.json`. When `--results-tsv` is provided, it forwards the raw compare rows for every prompt/run pair into that TSV.

Use `judge_suite_results.sh` to evaluate a single suite summary or compare baseline vs candidate with a configurable improvement threshold (default 2%). Outputs a machine-readable verdict with `merge_recommended`.

Use `run_anemll_coreml_benchmark.py` to measure fixed-token generation against the public exact-base `anemll` Core ML Llama 3.2 1B packages. It is intended for publishable Espresso-vs-Core ML comparisons when the Core ML baseline is split across embeddings, FFN/prefill, and lm-head compiled models instead of a single `.mlpackage`.

`benchmark-prompts.txt` defines the fixed prompt suite (short/medium/long) used by the suite runner.

The lane scaffold also creates:

- `.autoresearch/env.sh` to pin the compare contract for that worktree
- `.autoresearch/bench.sh` to run the official throughput wrapper with the lane-local contract
