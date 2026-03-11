# Espresso — Reverse-Engineered Apple Neural Engine Research

Training neural networks and exact decode paths directly on Apple's Neural Engine (ANE) via reverse-engineered private APIs. No CoreML training APIs, no Metal, no GPU for the core ANE path.

## Latest Public Decode Release

The strongest public decode result in this branch is an exact non-echo recurrent two-token ANE decode path on a local real-artifact family:

- Exact two-step ANE decode: `1.0806302083333332 ms/token`
- Matched one-token ANE control: `1.0957500000000002 ms/token`
- Matched CoreML `.cpuAndNeuralEngine`: `5.085307291666668 ms/token`
- Exact two-step speedup vs CoreML: `4.7583224488025415x`
- Exactness: parity `match`, `committed exact tokens/pass = 2`, `accepted future tokens/pass = 1`

Public evidence:
- Human summary: [artifacts/benchmarks/exact-decode-non-echo/latest.md](artifacts/benchmarks/exact-decode-non-echo/latest.md)
- Machine-readable artifact: [artifacts/benchmarks/exact-decode-non-echo/latest.json](artifacts/benchmarks/exact-decode-non-echo/latest.json)
- Release note: [docs/releases/2026-03-11-non-echo-exact-decode.md](docs/releases/2026-03-11-non-echo-exact-decode.md)
- Reproduction command: [scripts/reproduce_local_real_artifact_claim.sh](scripts/reproduce_local_real_artifact_claim.sh)
- Full lab notebook: [docs/fused-decode-and-next-steps.md](docs/fused-decode-and-next-steps.md)

Reproduce the public claim:

```bash
RESULTS_DIR=results/non-echo-public-$(date +%Y%m%d-%H%M%S) \
REPEATS=5 WARMUP=3 ITERATIONS=20 \
./scripts/reproduce_local_real_artifact_claim.sh
```

What this claim is:
- A matched same-session ANE/CoreML decode benchmark
- Exact recurrent-native two-token generation with no approximate commits
- A non-echo local artifact family built and exported by this repo

What this claim is not:
- Not a pretrained production checkpoint result
- Not a repaired generic recurrent ANE kernel; it depends on the explicit `identity-zero-trunk` backend
- Not a blanket claim about all Apple-model inference or all CoreML workloads

## What This Is

A from-scratch implementation of transformer training (forward + backward pass) and exact recurrent decode experiments running on the ANE in Apple Silicon. The ANE is a 15.8 TFLOPS (M4) inference accelerator that Apple does not expose directly for training or for arbitrary custom decode graphs. This project reverse-engineers the `_ANEClient` / `_ANECompiler` private APIs and the MIL (Model Intermediate Language) format to run custom compute graphs directly on ANE hardware.

**Historical ObjC prototype result (M4, single transformer layer, dim=768, seq=512):**
- 9.3 ms/step, 11.2% ANE utilization (1.78 TFLOPS sustained)

**Current Swift Phase 9 benchmark result (M3 Max, 12-layer Stories110M config):**
- Swift train: 131.2 ms/step
- ObjC train: 145.5 ms/step
- Swift/ObjC ratio: 0.901718
- Overall benchmark grade: S+ (100.00)

## Architecture

The training loop uses 6 ANE kernels per step:

| Kernel | Function | Weights |
|--------|----------|---------|
| `kFwdAttn` | RMSNorm + QKV projection + SDPA + output projection | Wq, Wk, Wv, Wo, rms1, mask |
| `kFwdFFN` | RMSNorm + SwiGLU FFN (W1, W3, SiLU, W2) | W1, W2, W3, rms2 |
| `kFFNBwd` | FFN backward (W2^T + SiLU_bwd + W1^T + W3^T) | W2^T, W1^T, W3^T |
| `kSdpaBwd1` | Wo^T + SDPA backward part 1 (dV, probs, dp) | Wo^T, mask |
| `kSdpaBwd2` | SDPA backward part 2 (softmax grad, dQ, dK) | — |
| `kQKVb` | QKV backward (Wq^T + Wk^T + Wv^T → dx) | Wq^T, Wk^T, Wv^T |

CPU handles: RMSNorm backward, residual connections, loss computation, dW gradient accumulation (cblas_sgemm), Adam optimizer updates.

Key optimizations:
- **Channel-first CPU layout** — matches ANE IOSurface `[1,C,1,S]` format, eliminates all transpose overhead
- **vDSP vectorized RMSNorm** — 10x faster than naive (6.7ms → 0.7ms)
- **GCD async cblas overlap** — dW gradient sgemms run in parallel with ANE evals on a serial dispatch queue
- **Deferred cblas wait** — wait pushed into next step's forward pass for maximum overlap
- **ANE RMSNorm fusion** — RMSNorm folded into forward kernels as MIL ops (reduce_sum + pow + mul)
- **Wo^T fusion** — output projection backward merged into SDPA backward kernel
- **Forward taps** — Q, K, V, attention scores, hidden states exposed via concat outputs, avoiding CPU recompute
- **exec() restart** — bypasses ~119 ANE compile limit per process

## File Structure

```
├── api_exploration.m       # Initial ANE API discovery
├── inmem_basic.m           # In-memory MIL compilation proof-of-concept
├── inmem_bench.m           # ANE dispatch latency benchmarks
├── inmem_peak.m            # Peak TFLOPS measurement (2048x2048 matmul)
├── sram_bench.m            # ANE SRAM bandwidth probing
├── sram_probe.m            # SRAM size/layout exploration
└── training/
    ├── ane_runtime.h       # ANE private API wrapper (compile, eval, IOSurface)
    ├── ane_mil_gen.h       # MIL program generation helpers
    ├── model.h             # Model weight initialization and blob builders
    ├── forward.h           # Forward pass MIL generators
    ├── backward.h          # Backward pass MIL generators
    ├── train.m             # Minimal training loop (early prototype)
    ├── tiny_train.m        # 2-layer tiny model training
    ├── train_large.m       # Main: single-layer dim=768 training (optimized)
    ├── test_*.m            # Unit tests for individual kernels
    └── Makefile
```

## Building

Requires macOS 15+ on Apple Silicon (tested on M4).

```bash
# Build the main training program
xcrun clang -O2 -framework Foundation -framework IOSurface \
  -framework CoreML -framework Accelerate -ldl -lobjc \
  -o train_large training/train_large.m

# Run
./train_large
```

No external dependencies. Uses only system frameworks + private ANE APIs resolved at runtime via `objc_msgSend`.

## Swift Validation Lanes (Phase 8)

The Swift rewrite keeps strict env-gated lanes so default `swift test` remains deterministic.

```bash
# G0 baseline
swift build
swift test

# G1 hardware correctness lane
ANE_HARDWARE_TESTS=1 swift test --filter "ANERuntimeTests|EspressoTests|CrossValidationTests"

# G2 ObjC cross-validation lane
OBJC_CROSS_VALIDATION=1 ANE_HARDWARE_TESTS=1 PHASE8_BENCHMARKS=1 swift test --filter CrossValidationTests

# Phase 8 benchmark artifacts
./scripts/phase8_benchmark.py
```

Environment gates:
- `ANE_HARDWARE_TESTS=1`: enable ANE runtime/hardware tests.
- `OBJC_CROSS_VALIDATION=1`: enable ObjC fixture parity tests.
- `PHASE8_BENCHMARKS=1`: emit structured CrossValidation benchmark metrics.
- `ESPRESSO_INTEGRATION_TESTS=1`: enable integration tests requiring dataset/model assets.
- `ESPRESSO_PERF_TESTS=1`: enable perf lane tests (M4-targeted benchmark test is expected to skip on non-M4 hosts).

## Performance Baseline & Progress

### Baseline Snapshot (Copied Reference)

Source: `artifacts/benchmarks/phase8/latest.json` baseline snapshot provided at task start.

| field | value |
| --- | --- |
| timestamp | `2026-03-04T21:07:16.203258+00:00` |
| hardware | `Mac15,11 (Apple M3 Max)` |
| OS | `macOS 26.0 (25A354)` |
| toolchain | `Swift 6.2.4` |
| swift train | `267.8 ms/step` |
| objc train | `163.6 ms/step` |
| ratio (swift/objc) | `1.636919` |
| performance score | `36.31 (F)` |
| overall grade | `A (87.26)` |
| gate status | `G0..G4 pass`, `G5 fail (-12.74%)` |

### Baseline Numerical Parity

| area | max abs diff | mean abs diff | swift_eval_ms | status |
| --- | --- | --- | --- | --- |
| `full_fused_forward` | `0.0` | `0.0` | `0.5064` | pass |
| `fused_backward` | `0.0` | `0.0` | `0.5492` | pass |

### Optimization Iterations

| change id | hypothesis | swift ms/step | objc ms/step | ratio | delta vs baseline | pass/fail notes |
| --- | --- | --- | --- | --- | --- | --- |
| `B0` | Baseline reference snapshot | `267.8` | `163.6` | `1.636919` | `0.0 ms (0.00%)` | parity pass, `G5` fail in baseline snapshot |
| `O1` | Add step-level telemetry + surface handle cache + batched surface reads + CPU workspace reuse | `271.7` | `155.5` | `1.747267` | `+3.9 ms (+1.46%)` | parity pass, temporary perf regression vs baseline |
| `O2` | Build-mode fairness (`swift -c release`) + parser fix for ObjC timing breakdown attribution | `149.8` | `143.3` | `1.045359` | `-118.0 ms (-44.06%)` | parity pass, stability pass, `G0..G5` pass |
| `O3` | Move hot `SurfaceIO` read/write/batched-read paths to C interop (lower Swift hot-loop overhead) | `131.2` | `145.5` | `0.901718` | `-136.6 ms (-51.01%)` | parity pass, stability pass, `G0..G5` pass; Swift faster than ObjC |

### Current Best Result

- Timestamp: `2026-03-04T22:26:15.868381+00:00`
- Swift train: `131.2 ms/step`
- ObjC train: `145.5 ms/step`
- Ratio: `0.901718` (improved by `-0.735201`, `-44.91%` vs baseline ratio `1.636919`)
- Performance score: `100.00 (S+)`
- Overall score: `100.00 (S+)`

### Remaining Bottlenecks (Current Best Step Breakdown)

| component | Swift ms/step | ObjC ms/step | delta |
| --- | --- | --- | --- |
| `t_ane` | `46.375` | `17.331` | `+29.044` |
| `t_io` | `32.387` | `15.224` | `+17.163` |
| `t_cls` | `10.207` | `10.620` | `-0.413` |
| `t_elem` | `20.861` | `22.473` | `-1.612` |
| `t_rms` | `0.097` | `0.112` | `-0.015` |
| `t_cblas_wait` | `0.001` | `0.002` | `-0.001` |

Primary remaining bottlenecks are ANE execution time and ANE I/O conversion/copy overhead.

### Gate Status Summary (Current Best Run)

| gate | status | evidence |
| --- | --- | --- |
| `G0` | pass | `swift build rc=0, swift test rc=0` |
| `G1` | pass | hardware correctness lane rc=0 |
| `G2` | pass | ObjC cross-validation lane rc=0, metrics emitted=2 |
| `G3` | pass | `latest.json/csv/md` complete with metadata |
| `G4` | pass | grade/parity completeness checks present |
| `G5` | pass | regression threshold check: `Within threshold` |

## How It Works

1. **MIL generation** — Objective-C code constructs MIL program text at runtime, specifying convolutions (for linear layers), matmul (for attention), softmax, element-wise ops
2. **In-memory compilation** — `_ANEInMemoryModelDescriptor` compiles MIL text + weight blobs directly to ANE programs, no disk mlmodelc needed
3. **IOSurface I/O** — Input/output tensors passed via IOSurface shared memory in `[1, channels, 1, spatial]` format (fp16)
4. **Weight embedding** — Weights baked into ANE programs as BLOBFILE constants; recompiled each batch when weights change
5. **Gradient flow** — Forward taps expose intermediates needed for backward; backward kernels compute dx (input gradients) on ANE; dW (weight gradients) computed on CPU via cblas

## Limitations

- **SDPA causal masking** — ANE hardware ignores `attn_mask` in SDPA ops; causal attention is decomposed into separate Q@K^T (ANE) → mask+softmax (ANE via add+softmax) → scores@V (ANE)
- **~119 compile limit** — ANE compiler leaks resources; worked around via `exec()` restart with checkpoint
- **Single layer** — Currently trains one transformer layer; multi-layer would need pipeline scheduling
- **Synthetic data** — Currently uses random data for benchmarking; real tokenized data support is WIP

## Performance History

| Optimization | ms/step | ANE util |
|---|---|---|
| Baseline (vDSP transpose) | 33.5 | 3.1% |
| Channel-first layout | 20.3 | 5.2% |
| vDSP vectorized RMSNorm | 14.2 | 7.4% |
| GCD async cblas overlap | 11.4 | 9.2% |
| ANE RMSNorm fusion | 11.4 | 9.2% |
| Wo^T fusion (7→6 kernels) | 11.4 | 9.2% |
| Deferred cblas wait | **9.3** | **11.2%** |

## Disclaimer

This project is independent research into Apple Neural Engine architecture. It uses undocumented APIs discovered through runtime introspection for research and educational purposes under fair use and interoperability provisions (see *Sega v. Accolade*, 1992; DMCA §1201(f)). No Apple proprietary code or binaries are included in this repository. This project is not affiliated with or endorsed by Apple Inc. Use at your own risk.

## License

MIT — see [LICENSE](LICENSE)
# swift-ane
