# Exact Non-Echo Decode Benchmark

Reference commit: `ebd3c38`  
Intended release tag: `public-non-echo-exact-decode-20260311`

## Claim

On a matched same-session non-echo local-artifact benchmark, Espresso's exact recurrent-native two-token ANE decode path measured:

- Exact two-step ANE decode: `1.0806302083333332 ms/token`
- Matched one-token ANE control: `1.0957500000000002 ms/token`
- Matched CoreML `.cpuAndNeuralEngine`: `5.085307291666668 ms/token`
- Exact two-step speedup vs CoreML: `4.7583224488025415x`
- Exactness: parity `match`, `committed exact tokens/pass = 2`, `accepted future tokens/pass = 1`

## Benchmark Contract

- Input mode: `recurrent-checkpoint`
- Artifact family: local bigram recurrent artifact exported from repo text
- Control backend: `identity-zero-trunk`
- Two-step backend: `identity-zero-trunk`
- Output-head backend: `ane-rmsnorm-classifier`
- Prompt token: `35`
- Repeats: `5`
- Warmup: `3`
- Timed iterations: `20`
- Max new tokens: `8`
- Max sequence tokens: `32`
- Layer count: `6`

## Reproduce

```bash
RESULTS_DIR=results/non-echo-public-$(date +%Y%m%d-%H%M%S) \
REPEATS=5 WARMUP=3 ITERATIONS=20 \
./scripts/reproduce_local_real_artifact_claim.sh
```

## Caveats

- This is a non-echo local artifact family, not a pretrained production checkpoint.
- The claim depends on the explicit `identity-zero-trunk` backend.
- The generic recurrent ANE kernel remains a negative result off-echo.
- Only exact committed tokens are counted.
