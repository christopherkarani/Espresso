# 2026-03-11 Release — Exact Non-Echo ANE Decode

Release tag: `public-non-echo-exact-decode-20260311`  
Reference throughput commit: `ebd3c38`

## Quote-Safe Public Claim

On a reproducible non-echo local-artifact benchmark, Espresso's exact recurrent-native two-token ANE decoder achieved a matched same-session median of `1.0806302083333332 ms/token` versus `5.085307291666668 ms/token` for CoreML `.cpuAndNeuralEngine`, a `4.7583224488025415x` speedup, while preserving exact parity and committing `2` exact tokens per pass.

## What Was Shipped

- The exact non-echo `identity-zero-trunk` two-token path with the future proposer moved from CPU onto ANE.
- A one-command reproduction surface: [scripts/reproduce_local_real_artifact_claim.sh](../../scripts/reproduce_local_real_artifact_claim.sh)
- A checked-in benchmark artifact:
  - [artifacts/benchmarks/exact-decode-non-echo/latest.json](../../artifacts/benchmarks/exact-decode-non-echo/latest.json)
  - [artifacts/benchmarks/exact-decode-non-echo/latest.csv](../../artifacts/benchmarks/exact-decode-non-echo/latest.csv)
  - [artifacts/benchmarks/exact-decode-non-echo/latest.md](../../artifacts/benchmarks/exact-decode-non-echo/latest.md)
- Updated top-level README language so the public claim is visible without reading the full lab notebook.

## Measured Result

- Exact two-step ANE decode: `1.0806302083333332 ms/token`
- Matched one-token ANE control: `1.0957500000000002 ms/token`
- Matched CoreML `.cpuAndNeuralEngine`: `5.085307291666668 ms/token`
- Exact two-step speedup vs CoreML: `4.7583224488025415x`
- Exact one-token ANE control speedup vs CoreML: `4.640428016426192x`
- Exactness: parity `match`, `committed exact tokens/pass = 2`, `accepted future tokens/pass = 1`

Representative same-session sample:

- Exact two-step: `1.0607916666666668 ms/token`
- Matched one-token ANE control: `1.0760208333333334 ms/token`
- Proposer: `0.9317604166666666 ms/pass`
- Verifier logits: `0.9893697916666667 ms/pass`
- Verifier trunk: `0.000010416666666666666 ms/pass`

## Reproduce

```bash
RESULTS_DIR=results/non-echo-public-$(date +%Y%m%d-%H%M%S) \
REPEATS=5 WARMUP=3 ITERATIONS=20 \
./scripts/reproduce_local_real_artifact_claim.sh
```

## Important Caveats

- This is a non-echo local artifact family, not a pretrained production checkpoint.
- The claim depends on the explicit `identity-zero-trunk` backend.
- The generic recurrent ANE kernel remains a negative result off-echo.
- Only exact committed tokens are counted.

## What Is Actually Impressive Here

- The path commits `2` exact tokens per pass with parity preserved.
- The gain is not just over CoreML; on this artifact family it also beats the matched one-token ANE identity control.
- The win comes from a real recurrent-native architecture change plus ANE proposer placement, not approximate speculative decoding.
