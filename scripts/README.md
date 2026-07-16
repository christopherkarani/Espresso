# Scripts

Stable, product-facing helper scripts. Research-only tooling lives under
[`../research/scripts/`](../research/scripts/).

## Tracked product scripts

| Script | Purpose |
|--------|---------|
| `ensure_coreml_model.sh` | Ensure a CoreML baseline model is available for compare benches |
| `generate_coreml_model.py` | Build CoreML packages used in comparisons |
| `reproduce_local_real_artifact_claim.sh` | Reproduce the public real-artifact claim path |
| `run_power_benchmark.sh` | Power / energy-oriented local runs |
| `generate-benchmark-dashboard.sh` | Regenerate `docs/benchmarks.md` from `latest.json` |
| `assert_readme_claims.py` | CI guard: README numbers must match `latest.json` |
| `convert_weights_gpt2.py` / `convert_weights_llama.py` | Weight conversion helpers |
| `espresso_llama_weights.py` / `stories_model_identity.py` | Stories/Llama weight identity helpers |
| `export_llama_coreml.py` | Export Llama-family CoreML baselines |
| `benchmark-prompts.txt` / `stories_prompt_suite.txt` | Fixed prompt suites |

Python unit tests for product helpers: `scripts/tests/`.

## Research scripts (not product)

Autoresearch suite runners, Stories distillation, factored-head packaging, and
rejected draft packaging live in `research/scripts/`. Do not link them from the
README product journey.
