# Benchmarks

- `benchmarks/results/latest.json` — **checked-in** public claims source of truth
- `benchmarks/results/*` (other files) — local run dumps, gitignored
- `benchmarks/models/` — placeholders / curated benchmark inputs

Public README and site numbers must match `latest.json`. Refresh that file only
when intentionally publishing a new claim, and update the README table in the
same change. CI runs `scripts/assert_readme_claims.py` to enforce this.

Reproduce:

```bash
RESULTS_DIR=results/$(date +%Y%m%d-%H%M%S) \
REPEATS=5 WARMUP=3 ITERATIONS=20 \
./scripts/reproduce_local_real_artifact_claim.sh
```
