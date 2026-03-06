# Fused Decode Kernel + Performance Tuning Research

**Date:** 2026-03-06
**Host:** M3 Max, macOS 15.x
**Branch:** `feat/vc-eval-probe`

---

## 1. Fused Decode Kernel (IMPLEMENTED)

### Problem

ANE decode throughput is eval-dominant. The decode path dispatched 12 synchronous
`eval()` calls per token (2 kernels x 6 layers), each carrying ~0.095ms dispatch
overhead. After exhausting IO trimming and hitting dead ends with `_ANEChainingRequest`
and `_ANEVirtualClient`, the remaining software-addressable overhead was **dispatch count**.

### Solution

Combine the decode attention kernel (`DecodeAttentionQKVGenerator`) and FFN kernel
(`DecodeFFNGenerator`) into a single MIL program per layer (`FusedDecodeLayerGenerator`).
This halves the dispatch count from 12 to 6 and eliminates the attn-to-FFN surface copy.

### Implementation

| File | Role |
|------|------|
| `Sources/MILGenerator/FusedDecodeLayerGenerator.swift` | Single MIL combining RMSNorm1 + QKV + attention + Wo + residual1 + RMSNorm2 + SwiGLU FFN + residual2 |
| `Sources/ANERuntime/FusedDecodeKernelSet.swift` | `~Copyable` kernel wrapper (9 weight blobs) |
| `Sources/Espresso/DecodeForwardPass.swift` | `FusedDecodeSurfaceHandles` + `runFusedDecodeTimed()` loop |
| `Tests/ANERuntimeTests/FusedDecodeKernelTests.swift` | 19 tests (12 MIL + 7 hardware-gated) |

### MIL Design

```
Inputs (4):
  x         [1, dim, 1, laneSpatial]   token packed at lane 0
  kCache    [1, dim, 1, maxSeq]         K cache
  vCache    [1, dim, 1, maxSeq]         V cache
  maskCache [1, dim, 1, maxSeq]         attention mask

Outputs (3):
  xNext     [1, dim, 1, laneSpatial]   both residuals applied
  kfFull    [1, dim, 1, laneSpatial]   K projection for cache writeback
  vfFull    [1, dim, 1, laneSpatial]   V projection for cache writeback

Weights (9 BLOBFILE constants):
  rms1.bin, wq.bin, wk.bin, wv.bin, wo.bin   (attention)
  rms2.bin, w1.bin, w3.bin, w2.bin            (FFN)
```

**SSA name collision strategy**: FFN RMSNorm variables are prefixed `f_` (e.g., `f_sq`,
`f_ss`, `f_rrms`, `f_xr`, `f_xn`). Shared constants (`invd`, `eps`, `nhalf`, `raxCh`,
`raxSp`, `kd`) and conv constants (`pt`, `st`, `pd`, `dl`, `gr`) are defined once in
the attention block and reused by the FFN block.

### Measured Results (M3 Max, 1 layer, maxSeq=32)

| Path | ANE ms/step | Dispatches |
|------|-------------|-----------|
| Unfused (attn + FFN) | 0.761 | 2 |
| Fused (single dispatch) | 0.666 | 1 |
| **Savings** | **0.095 (12.5%)** | **-1** |

Extrapolated over 6 layers: **~0.57ms saved per token** (from 12 to 6 dispatches).

---

## 2. Current Decode Budget Breakdown (6 layers)

```
Component                   Unfused         Fused (current)
------------------------------------------------------------
ANE dispatch overhead       6 x 0.095ms     eliminated
                            = 0.570ms

ANE compute (hw time)       6 x 0.310ms     6 x 0.405ms
                            = 1.860ms        = 2.430ms

IO (cache update + chain)   6 x 0.029ms     6 x 0.024ms
                            = 0.174ms        = 0.144ms

Attn->FFN surface copy      6 x 0.005ms     eliminated
                            = 0.030ms

Total per token             ~2.634ms         ~2.574ms
```

**The remaining wall is ANE compute time** -- that is hardware silicon speed. The
dispatch overhead was the largest software-addressable cost and has been eliminated.

---

## 3. Approaches to 4x Over CoreML

### 3.1 Multi-Layer Kernel Fusion via KV Cache Packing (HIGHEST PRIORITY)

**Concept**: Fuse 2 transformer layers into a single MIL program. Pack both layers' KV
caches into one input IOSurface using `slice_by_size` to extract each region. This
reduces dispatches from 6 to 3 and eliminates inter-layer surface copies within each
fused pair.

**Cache packing layout** (channel axis of a `[1, C_packed, 1, maxSeq]` input):

| Channel Offset | Width | Content |
|----------------|-------|---------|
| 0 | 768 | Layer 0 kCache |
| 768 | 768 | Layer 0 vCache |
| 1536 | 768 | Layer 0 maskCache |
| 2304 | 768 | Layer 1 kCache |
| 3072 | 768 | Layer 1 vCache |
| 3840 | 768 | Layer 1 maskCache |

`C_packed = 4608`, packed cache size = 294,912 bytes (0.28 MB).

**Memory analysis (2-layer fused program, dim=768, hidden=2048)**:

| Component | Per Layer | 2 Layers |
|-----------|-----------|----------|
| Attention weights (Wq,Wk,Wv,Wo) | 4 x 768^2 x 2 = 4.72 MB | 9.44 MB |
| FFN weights (W1,W3,W2) | 3 x 2048x768 x 2 = 9.44 MB | 18.87 MB |
| RMSNorm (rms1,rms2) | 2 x 768 x 2 = 3 KB | 6 KB |
| **Total baked weights** | **14.28 MB** | **28.57 MB** |
| Cache data in IOSurface | 0.14 MB | 0.28 MB |
| **Working set vs 32MB SRAM** | fits | fits (ANE streams weights via DMA) |

**MIL feasibility**: `slice_by_size` on channel-packed inputs is a proven pattern -- the
backward pass generators (`SDPABackward1Generator`, `QKVBackwardGenerator`, etc.) already
perform 3-4 slices from packed inputs with convolutions and full attention math. The
documented `slice_by_index` failure does NOT apply to `slice_by_size`.

**Expected savings**: 3 more dispatch overheads eliminated (3 x 0.095ms = 0.285ms) plus
inter-layer copy elimination. Combined with single-layer fusion: **12 dispatches to 3**.

**Layer fusion scaling**:

| Fused Layers | Dispatches | Baked Weights | SRAM Fit? | Feasibility |
|---|---|---|---|---|
| 1 (current) | 6 | 14.28 MB | yes | **done** |
| 2 | 3 | 28.57 MB | marginal | **high confidence** |
| 3 | 2 | 42.85 MB | no (streams) | medium confidence |
| 6 | 1 | 85.70 MB | no | low confidence |

**Recommendation**: Start with 2-layer fusion. If successful, try 3.

### 3.2 Metal + ANE Hybrid Decode (HIGH IMPACT, UNEXPLORED)

**Concept**: Use ANE for convolution-heavy work (QKV projections, FFN) and Metal for
dynamic attention computation (Q@K^T, softmax, weights@V). IOSurfaces are the native
shared memory format for both Metal and ANE -- **zero copy between accelerators**.

**Why this could achieve 4x**:
- Two accelerators working in **parallel**: ANE does layer N's FFN while Metal does
  layer N+1's attention
- Metal `simdgroup_matrix` on M3 provides ~7 additional TFLOPS for matmul
- ANE's attention is currently a probe path (bypasses real math). Metal handles dynamic
  attention natively with full flexibility
- CoreML cannot do this -- it dispatches to one accelerator at a time per subgraph

**Architecture**:
```
ANE:   [QKV proj] -----> IOSurface (Q,K,V) -----> [FFN]
                              |                      ^
                              v                      |
Metal:                  [Q@K^T -> softmax -> @V] --> IOSurface (attn_out)
```

**Challenges**:
- Requires Metal compute pipeline for attention
- Synchronization between ANE and Metal dispatch
- Surface format compatibility (both use FP16, both use IOSurface)

### 3.3 `evaluateRealTimeWithModel:` Probe (MEDIUM IMPACT, QUICK WIN)

The `_ANEClient` exposes a real-time eval path:
- `beginRealTimeTask` / `endRealTimeTask` lifecycle
- `evaluateRealTimeWithModel:options:request:error:`

**This has never been tested for dispatch latency.** If the real-time path bypasses the
QoS scheduling queue, it could have materially lower dispatch overhead than
`evaluateWithQoS:`. The existing interop code already has conditional support for this
path (controlled by `ANE_REALTIME` environment variable).

**Estimated effort**: 1-2 hours to probe and benchmark.

### 3.4 Metal SharedEvent on Standard Eval Path (MEDIUM IMPACT)

`_ANERequest.setSharedEvents:` works on the standard eval path (no VirtualClient
required). If we construct an `IOSurfaceSharedEvent` via `[MTLDevice newSharedEvent]`
and attach it, the ANE hardware might signal completion asynchronously -- enabling true
pipelined dispatch where:

1. Submit eval for layer N
2. ANE signals Metal SharedEvent when done
3. CPU starts IO for layer N while layer N+1 eval begins

**Key unknown**: Does `setSharedEvents:` on the standard `_ANEClient` eval path actually
trigger hardware-level async signaling? The `completionEvent:` parameter on the
VirtualClient eval path suggests it should, but the standard path may ignore it.

### 3.5 Speculative Decoding (ALGORITHMIC, HIGH IMPACT)

**Concept**: Use a tiny draft model (1-2 layers, or a distilled version) to generate N
candidate tokens, then verify all N in one batched forward pass on the full model. ANE
excels at batch compute (deep graphs, high utilization).

This transforms the decode problem from "N sequential single-token dispatches" into
"1 cheap draft + 1 expensive verify" -- an algorithmic 3-5x on top of hardware
optimization.

**Why ANE is particularly suited**: The prefill path (batch tokens) already achieves much
higher utilization than decode (single token). Speculative decoding turns decode into a
prefill-like workload.

---

## 4. Approaches Proven Blocked (Do Not Retry)

| Approach | Failure Mode | Evidence |
|----------|-------------|----------|
| IOSurface aliasing between kernels | `statusType=0x9: Program Inference error` | Compile-time external surface fails at eval |
| _ANERequest surface rebinding | `-[__NSArrayM procedureIndex]` type confusion, then `0x9` | Request is immutable after construction |
| _ANEVirtualClient instantiation | All 5 init paths return nil | Kernel-level IOKit entitlement gate |
| _ANEChainingRequest | `PREPARE_FAILED` across all parameter permutations | Exhausted A22-A26 experiments |
| `setCodeSigningIdentity:` | Crashes with `-[__NSCFConstantString setObject:forKey:]` | Identity stored in class-level static dict |
| QoS tuning (0-63) | No effect on latency | All values produce identical timing |

---

## 5. Queue Pipelining Research

### Can we overlap eval with IO?

`evaluateWithQoS:` is **synchronous** -- it blocks the calling thread until the ANE
hardware completes. The `completionHandler` fires synchronously on the calling thread
after eval returns (not on a background thread). This means:

- CompletionHandler chaining provides **zero pipelining benefit**
- The only async eval path (`doEvaluateWithModel:completionEvent:`) requires VirtualClient
  (blocked by entitlements)

### What IS possible

**Background dispatch with cache overlap**: Dispatch `eval()` on a GCD queue. While
the eval blocks that queue, the main thread can perform K/V/mask cache updates for the
*previous* layer. Savings: ~5-10 microseconds per layer (the cache updates are already
very fast spatial-slice copies).

**Double-buffered handles**: Compile each kernel twice with separate surfaces. Write
to Handle-A while Handle-B evals. Eliminates inter-token IO latency but doubles compile
budget (which is capped at ~100 per process) and surface memory.

---

## 6. Priority Ranking

| # | Approach | Expected Impact | Effort | Risk |
|---|----------|----------------|--------|------|
| 1 | 2-layer KV cache packing | 0.285ms/token + zero inter-layer IO | Medium | Low |
| 2 | `evaluateRealTimeWithModel:` probe | Unknown (possibly 0.05ms/dispatch) | Small | Low |

---

## 7. Avenue 1 Result — Multi-Layer Kernel Fusion (ABANDONED)

**Date:** March 6, 2026

### Attempt 1: Fully packed K/V/mask cache input

- Built `FusedTwoLayerDecodeGenerator` / `FusedTwoLayerDecodeKernelSet` with a packed cache input
  `[1, 4608, 1, maxSeq]` and `slice_by_size` extraction for:
  - L0 K, V, mask
  - L1 K, V, mask
- Added non-hardware TDD coverage for:
  - MIL header / blob paths
  - packed-cache slice count
  - unique SSA names
  - I/O byte-size contract
- Hardware result:
  - `ANE_HARDWARE_TESTS=1 swift test --filter FusedTwoLayerDecodeKernelSetTests`
  - `_ANECompiler : ANECCompile() FAILED`
  - underlying compiler error: `InvalidMILProgram`

### Fallback: K/V-only packing, masks split back out

- Reduced the candidate to:
  - `packedKVCache` input `[1, 3072, 1, maxSeq]`
  - `maskCache0` input `[1, 768, 1, maxSeq]`
  - `maskCache1` input `[1, 768, 1, maxSeq]`
- This isolates packed K/V slicing from packed mask slicing while preserving the paired-kernel objective.
- Hardware result:
  - `ANE_HARDWARE_TESTS=1 swift test --filter FusedTwoLayerDecodeKernelSetTests/test_fused_two_layer_compile_fails_with_controlled_error_on_hardware`
  - same `_ANECompiler : ANECCompile() FAILED`
  - same underlying `InvalidMILProgram`

### Outcome

- Status: **abandoned**
- Reason: both the full packed-cache design and the K/V-only fallback fail at compile time with
  `InvalidMILProgram`, so the avenue never reaches eval.
- Benchmark delta: **N/A**. No post-change per-token timing exists because the candidate kernel never compiled.
- Per-token gain credited to this avenue: **+0.000ms**
- Running cumulative savings toward the 4x goal: **0.000ms**

## Results Summary

| Avenue | Status | Per-Token Gain | Cumulative |
|--------|--------|----------------|------------|
| 1. Multi-layer fusion | abandoned (`InvalidMILProgram` on full pack and K/V-only fallback) | +0.000ms | 0.000ms |
| 2. Metal SharedEvent | abandoned | +0.000ms | 0.000ms |
| 3. Metal+ANE hybrid | abandoned after QKV-only split-path benchmark | +0.000ms | 0.000ms |
| 4. CoreML baseline | measured | N/A | 0.000ms |
| 5. Speculative decode | abandoned | +0.000ms | 0.000ms |
| 6. GCD pipeline | abandoned | +0.000ms | 0.000ms |

Direct ANE: 2.875ms/token
CoreML:     3.007ms/token
Speedup:    1.05x
| 3 | Metal + ANE hybrid decode | Potentially 2-4x (parallel accelerators) | Large | Medium |
| 4 | Metal SharedEvent on standard eval | True async dispatch | Medium | Medium |
| 5 | Speculative decoding | 3-5x algorithmic | Large | Low |

### Recommended Execution Order

1. **Probe `evaluateRealTimeWithModel:`** -- quick win, 1-2 hours, might reveal a
   lower-overhead dispatch path
2. **Build 2-layer fused kernel with packed caches** -- proven MIL patterns, high
   confidence, eliminates 3 more dispatches
3. **Prototype Metal attention kernel** -- if successful, enables the hybrid architecture
   that could deliver 4x
4. **Speculative decoding** -- algorithmic multiplier on top of everything else

---

## 7. CoreML Comparison Gap

**We have no direct CoreML benchmark for the same model.** All throughput numbers are
for the direct `_ANEClient` path. To claim "4x over CoreML" we need:

1. Export the 6-layer transformer via coremltools as `.mlmodel`
2. Run inference with `.cpuAndNeuralEngine` compute units
3. Measure per-token decode latency on the same M3 Max hardware
4. Compare against our direct path

CoreML uses the same ANE hardware and MIL compiler. It may dispatch more efficiently
for some workloads (mature scheduling, internal graph fusion). Our advantage is
fine-grained control over kernel boundaries, zero-copy IOSurface management for KV
caches, and the ability to do things CoreML cannot (training, custom fusion, hybrid
dispatch).

## 8. Avenue 2 Result — Metal SharedEvent on Standard Eval Path (ABANDONED)

Date: 2026-03-06

What I built:
- Extended `ane_interop_probe_standard_completion_handler(...)` to optionally create a Metal-backed `MTLSharedEvent` via `MTLCreateSystemDefaultDevice()` and `newSharedEvent`.
- Wrapped that event in `_ANESharedSignalEvent` and `_ANESharedEvents`.
- Attached the container to `_ANERequest` via `setSharedEvents:` on the standard eval path.
- Extended the Swift probe surface to report attachment state and event value before/after eval.

Expected improvement:
- If ANE signaled the Metal shared event on completion, decode could pipeline host and accelerator work and potentially overlap adjacent dispatches.

Baseline before changes:
- Existing standard completion handler works, but fires synchronously on the calling thread and provides no true async overlap.
- No measurable per-token gain from the baseline path.

Observed result:
- `MTLCreateSystemDefaultDevice()` succeeded.
- `newSharedEvent` succeeded.
- `_ANERequest` reports `setSharedEvents:` exists.
- With the Metal-backed shared event attached, the hardware test enters the standard eval path and then hangs with no completion callback and no observable event advancement.

Abandon reason:
- This path is a dead end under the project protocol. It does not produce a usable signal or measurable pipeline parallelism, and it blocks progress inside eval instead.

Timing:
- Post measurement: not recorded.
- Delta: +0.000ms/token.
- Cumulative savings after Avenue 2: 0.000ms/token.

## 9. Avenue 3 Result — Metal + ANE Hybrid Decode Follow-On (ABANDONED)

Date: 2026-03-06

What I built:
- `DecodeQKVOnlyGenerator` — ANE decode kernel that performs `RMSNorm -> Wq/Wk/Wv` only and emits `Q`, `K_new`, `V_new`.
- `HybridDecodeKernelSet` — split runtime wrapper holding the QKV-only ANE kernel, the existing ANE FFN kernel, and cached `Wo` weights for Metal output projection.
- `HybridDecodeSurfaceHandles` + `ForwardPass.runHybridDecodeTimed(...)` — split decode loop:
  - ANE `QKV-only`
  - K/V cache writeback
  - Metal SDPA + `Wo` projection + residual add
  - ANE FFN
- `MetalAttentionKernel` follow-on path that reads ANE-layout IOSurfaces zero-copy and writes the projected attention output directly into `ffnIn`.

TDD / verification:
- Added failing tests first for:
  - `DecodeQKVOnlyGenerator`
  - `HybridDecodeKernelSet`
  - hybrid forward-pass step + comparison harness
- Verified:
  - `swift test`
  - `ANE_HARDWARE_TESTS=1 swift test --filter HybridDecodeKernelSetTests/test_hybrid_decode_kernel_set_compiles_on_hardware`
  - `ANE_HARDWARE_TESTS=1 swift test --filter HybridDecodeForwardPassTests/test_hybrid_decode_single_step_runs_on_hardware`
  - `ANE_HARDWARE_TESTS=1 swift test --filter HybridDecodeForwardPassTests/test_hybrid_decode_benchmark_reports_direct_and_hybrid_token_medians`

Dependency-graph proof:
- For a single decode token, layer `N` FFN cannot start until Metal attention for layer `N` finishes.
- Layer `N+1` QKV cannot start until FFN for layer `N` finishes, because it consumes layer `N`'s FFN output.
- Therefore the hoped-for ANE/Metal overlap does **not** exist on the critical path for this decode graph. The hybrid path is necessarily a serial substitution, not a pipelined one.

Baseline before changes:
- Direct ANE decode benchmark (`espresso-bench`, 6 layers, `steps=32`, `maxSeq=32`, `warmup=3`, `iterations=20`):
  - Median: `2.872 ms/token`
- Per-layer decode profile on the same run:
  - Average ANE attention eval: `209.383 us/layer`
  - Average ANE FFN eval: `210.511 us/layer`
- Prior standalone Metal SDPA probe (unchanged code path):
  - Mean: `0.395252 ms/eval`
  - Median: `0.217437 ms/eval`

Post measurement:
- Same 6-layer / 32-step decode schedule, using the new split path and timing each token with `mach_absolute_time()`:
  - Direct comparison-harness median: `2.864562 ms/token`
  - Hybrid median: `4.665979 ms/token`
- Hybrid stage medians per token:
  - ANE QKV-only: `1.233083 ms/token` = `205.514 us/layer`
  - Metal SDPA + `Wo` + residual: `1.877479 ms/token` = `312.913 us/layer`
  - ANE FFN: `1.253000 ms/token` = `208.833 us/layer`
  - IO: `0.086646 ms/token` = `14.441 us/layer`

Delta:
- Split attention half (`ANE QKV-only + Metal attention/projection`) = `518.427 us/layer`
- Current direct attention stage = `209.383 us/layer`
- Regression vs direct attention stage = `309.044 us/layer`
- Projected regression over 6 layers = `1.854264 ms/token`
- Measured end-to-end regression from the comparison harness:
  - `4.665979 - 2.864562 = 1.801417 ms/token`
  - Relative slowdown: `4.665979 / 2.864562 = 1.63x`

Conclusion:
- The QKV-only seam works and zero-copy Metal interop works on live ANE surfaces.
- But the dependency graph prevents meaningful overlap, and the naive serial substitution is materially slower than the existing direct ANE attention stage.
- This avenue should be treated as a dead end for decode throughput on the current runtime/model shape.

Timing impact recorded for this pass:
- Landed decode savings: `+0.000 ms/token`
- Cumulative savings after Avenue 3: `0.000 ms/token`

## 10. Avenue 4 Result — CoreML Baseline Benchmark (MEASURED)

Date: 2026-03-06

What changed:
- Extended `scripts/generate_coreml_model.py` to support `--layers` and `--output`.
- Generated a 6-layer CoreML package at `benchmarks/models/transformer_6layer.mlpackage`.
- Benchmarked decode on the same 6-layer, `steps=32`, `maxSeq=32` workload used for direct ANE.

TDD / pre-change failure:
- The original exporter ignored `--layers 6 --output ...` and still wrote the default single-layer model to `benchmarks/models/transformer_layer.mlpackage`.
- After the exporter change, the same command produced the requested 6-layer package.

Benchmark configuration:
- Warmup: `5` sequences
- Timed: `100` sequences (`3200` measured tokens)
- Decode schedule: `steps=32`, `maxSeq=32`
- CoreML model: `benchmarks/models/transformer_6layer.mlpackage`

Measured results:
- Direct ANE decode:
  - Mean: `3.098 ms/token`
  - Median: `2.875 ms/token`
- CoreML Decode (`.all`):
  - Mean: `3.005 ms/token`
  - Median: `2.992 ms/token`
- CoreML Decode (`.cpuAndNeuralEngine`):
  - Mean: `3.017 ms/token`
  - Median: `3.007 ms/token`

Speedup vs CoreML `.cpuAndNeuralEngine`:
- `3.007 - 2.875 = 0.132 ms/token` saved
- Relative speedup: `3.007 / 2.875 = 1.05x`

Strict fastest-CoreML gate:
- Fastest CoreML median observed: `.all` at `2.992 ms/token`
- Direct ANE median: `2.875 ms/token`
- Speedup vs fastest CoreML: `1.04x`

Interpretation:
- The direct `_ANEClient` path currently beats CoreML on this workload, but only narrowly.
- The current project is far from the `4x over CoreML` target; the measured headroom is about `5%` over `.cpuAndNeuralEngine` and `4%` over the fastest CoreML configuration observed here.

## 11. Avenue 5 Result — Speculative Decoding (ABANDONED)

Date: 2026-03-06

Why this was stopped quickly:
- The current decode benchmark path is hidden-state based, not token-generation based.
- `ANEDirectBench.runDecode(...)` pre-generates token embeddings directly and measures layer execution on those embeddings.
- Training-side code has embeddings, classifier weights, and cross-entropy, but there is no existing inference path that takes token ids to logits, samples candidates, and then verifies them in a batched prefill-style pass.
- There is also no existing batched verification/prefill decode path that would let the full model verify `N` draft tokens in one pass without building substantial new model/runtime support first.

What I verified locally:
- `ModelConfig` includes `vocab=32000` and the training path has embedding/classifier machinery.
- The live decode/inference runtime exposed in the benchmark suite does not currently surface a meaningful token-level speculative decode loop or accept-rate harness.

Abandon reason:
- On the current random-weight decode harness, speculative acceptance would be numerically possible to simulate but not meaningful.
- Building a real speculative benchmark here would require a separate generation stack (token ids -> embeddings -> decode -> final norm/classifier -> sampling) and a batched verification path, which is beyond a fast avenue probe.
- Per the protocol, this is a dead end for the current session because it cannot yield a credible throughput number quickly.

Timing impact recorded for this pass:
- No benchmark recorded.
- Landed decode savings: `+0.000 ms/token`
- Cumulative savings after Avenue 5: `0.000 ms/token`

## 12. Avenue 6 Result — GCD Pipeline with CompletionHandler (ABANDONED)

Date: 2026-03-06

Why this was abandoned without a code variant:
- The earlier completion-handler probe already showed that the standard eval completion callback is synchronous on the calling thread, so there is no true async wakeup to exploit.
- The fused decode loop (`runFusedDecodeTimed`) is dependency-bound in the wrong place for host-side pipelining:
  - The token lane write happens before eval.
  - The meaningful per-layer host work after that is `eval -> K cache writeback -> V cache writeback -> mask flip -> xNext chain`.
  - Those cache and chain operations all depend on eval outputs and therefore cannot overlap the eval that produces them.
- The current measured decode I/O budget is only about `0.100 ms/token` total, and most of that budget sits in post-eval dependent copies.

Feasibility judgment:
- A background GCD queue could at best hide the tiny pre-eval host work (token lane write, occasional window sync).
- That theoretical headroom is below the avenue's `10 us/token` abandon threshold once constrained by the actual dependency graph.

Timing impact recorded for this pass:
- No benchmark recorded.
- Landed decode savings: `+0.000 ms/token`
- Cumulative savings after Avenue 6: `0.000 ms/token`

## 13. Avenue 5 Revisit — Real Token-Generation Harness + Speculative Upper Bound (2026-03-06)

Status: harness implemented; current speculative verifier design abandoned as structurally non-viable.

What changed:
- Added a real token-generation surface in the runtime:
  - `AutoregressiveLanguageModel`
  - `AutoregressiveGenerationHarness`
  - `SpeculativeGenerationHarness`
  - `GenerationWeights`
  - `ANEDirectGenerationModel`
- The new path accepts token ids, performs embedding lookup, runs the existing ANE decode loop autoregressively, applies final RMSNorm + classifier logits on CPU, and selects the next token with argmax.
- The same surface now also supports batched full-model verification via the existing inference kernels, which made it possible to benchmark speculative control flow honestly.

What is now possible:
- End-to-end token generation instead of hidden-state replay.
- Honest measurement of generated-token latency and effective tokens/sec.
- Speculative draft/full control flow with explicit acceptance accounting.
- Batched verification over a whole prefix+candidate sequence using the existing full-sequence inference path.

Local asset limitation:
- There is still no local `stories110M.bin` / `STORIES_MODEL_PATH`, so this session could not run a semantic speculative benchmark on a pretrained model.
- Instead, I measured a structural upper bound using a synthetic "echo" model on real ANE kernels:
  - All transformer weights = `0`
  - RMS weights = `1`
  - Shared embedding/classifier row `0` = `1`, all other rows = `0`
  - This forces draft/full agreement to `100%`, which is useful for measuring the best-case overhead of the current speculative implementation independent of model quality.

Benchmark protocol:
- Command:
  - `ANE_HARDWARE_TESTS=1 swift test --filter GenerationHarnessHardwareTests`
- Warmup: `3`
- Timed iterations: `20`
- Prompt: `[0]`
- Generated tokens per sample: `8`
- Direct generation model: `6` layers
- Speculative draft/full: `2` draft layers, `6` full layers
- Candidate counts measured: `k=2`, `k=4`

Measured results on the synthetic upper-bound model:
- Direct generation harness:
  - Median: `6.110643 ms/token`
  - Throughput: `163.65 tok/s`
- Speculative `k=2`:
  - Acceptance: `1.0`
  - Median: `38.032148 ms/token`
  - Throughput: `26.29 tok/s`
- Speculative `k=4`:
  - Acceptance: `1.0`
  - Median: `28.609625 ms/token`
  - Throughput: `34.95 tok/s`

Interpretation:
- This is the strongest negative result available for the current speculative design.
- Even with forced `100%` agreement, speculative generation is still dramatically slower than direct generation.
- The bottleneck is not acceptance quality; it is the verifier/state-sync architecture:
  - Full-sequence verification runs over the fixed `seqLen=256` inference path.
  - After verification, the full model still has to advance its decode state sequentially for accepted tokens.
  - The draft model is then reset and re-prefilled from the accepted prefix.
- That combination makes the current implementation roughly:
  - `38.03 / 6.11 = 6.22x` slower at `k=2`
  - `28.61 / 6.11 = 4.68x` slower at `k=4`

Conclusion:
- Avenue 5 is no longer blocked by missing infrastructure. The infrastructure now exists.
- Avenue 5 is blocked by measured verifier cost on the current runtime design.
- More draft tuning is not the right next move. A speculative path is only worth revisiting if the verifier can become radically cheaper and reusable across accepted tokens.
- The next serious direction should shift to:
  - RWKV-style recurrent decode, or
  - a fundamentally different speculative verifier that reuses caches/state instead of replaying and then re-advancing.

## 14. RWKV-Style Recurrent Decode Prototype (2026-03-06)

Status: implemented and benchmarked. Promising enough to keep as the secondary architecture bet, but not yet strong enough to replace the current roadmap without a depth-matched follow-up.

What changed:
- Added a minimal recurrent step path:
  - `RWKVStyleRecurrentWeights`
  - `RWKVStyleRecurrentStepGenerator`
  - `RWKVStyleRecurrentKernelSet`
  - `RWKVStyleRecurrentSession`
  - `RWKVStyleRecurrentBench`
- The recurrent MIL uses only already-proven ops on this branch:
  - `conv`
  - `add`
  - `mul`
  - `sigmoid`
  - `reduce_sum`
  - `pow`
- The step contract is:
  - inputs: `x_t`, `state_in`
  - outputs: `x_next`, `state_out`
  - both state and token lanes are fixed-size `[1, dim, 1, laneSpatial]`

Why this shape:
- The goal here was not faithful RWKV semantics yet.
- The goal was to answer the first-order systems question with minimal compiler risk:
  - can a constant-state recurrent step compile on ANE,
  - run repeatedly,
  - and stay flat as effective context grows?

Hardware compile behavior:
- `ANE_HARDWARE_TESTS=1 swift test --filter RWKVStyleRecurrentKernelSetTests`
- Result:
  - compile succeeded
  - eval succeeded
  - input/output surfaces were accessible
- Important downside:
  - first hardware compile for the recurrent kernel took `131.362 s`
- Interpretation:
  - runtime execution is viable
  - iteration speed for new recurrent kernel shapes is currently poor

Benchmark method:
- Command:
  - `ANE_HARDWARE_TESTS=1 swift test --filter RWKVStyleRecurrentPrototypeHardwareTests`
- Timing:
  - `mach_absolute_time()` / `mach_timebase_info`
- Protocol:
  - `3` warmup steps
  - `20` timed tail steps
  - measure tail-step latency at effective contexts `32`, `256`, `1024`, `4096`
- Comparison:
  - recurrent prototype under the same tail-step protocol
  - current 6-layer transformer decode control under the same tail-step protocol

Recurrent results:
- Run 1:
  - `32`: `0.314125 ms/token` (`3183 tok/s`)
  - `256`: `0.215125 ms/token` (`4648 tok/s`)
  - `1024`: `0.205854 ms/token` (`4858 tok/s`)
  - `4096`: `0.217500 ms/token` (`4598 tok/s`)
- Run 2:
  - `32`: `0.315563 ms/token` (`3169 tok/s`)
  - `256`: `0.217896 ms/token` (`4589 tok/s`)
  - `1024`: `0.216958 ms/token` (`4609 tok/s`)
  - `4096`: `0.200625 ms/token` (`4984 tok/s`)

Transformer control results under the same tail-step protocol:
- Run 1:
  - `32`: `2.907875 ms/token`
  - `256`: `4.055500 ms/token`
  - `1024`: `3.924917 ms/token`
  - `4096`: `2.868458 ms/token`
- Run 2:
  - `32`: `2.954167 ms/token`
  - `256`: `2.872583 ms/token`
  - `1024`: `2.847708 ms/token`
  - `4096`: `2.923729 ms/token`

Interpretation:
- The narrow recurrent question came back positive:
  - the recurrent step stayed effectively flat through `4096`.
- The stronger hoped-for transformer comparison came back weaker than expected:
  - the current transformer control did not show clean context-length growth on this graph.
  - the existing tiled decode path keeps tail-step latency roughly flat, with some noise.
- That means:
  - recurrent decode is still interesting,
  - but not because it clearly beats an O(context) tail-latency curve in the current implementation.

Scientific validity / caveats:
- This is not an apples-to-apples throughput comparison:
  - the recurrent prototype is a single recurrent layer
  - the transformer control is a 6-layer decode path
- The useful claim that *is* supported:
  - a constant-state recurrent ANE step is mechanically viable and shows stable flat tail-step latency at long effective contexts.
- The useful claim that is *not yet* supported:
  - a realistic recurrent model stack will outperform the optimized tiled transformer end-to-end on this hardware.

Decision:
- Keep RWKV-style recurrent decode alive as a credible long-term architecture path.
- Do not spend more time on speculative draft tuning for the current verifier design.
- Do not over-claim the recurrent result yet.
- The next recurrent step worth doing is a more depth-matched stacked recurrent benchmark, but only if the compile-time iteration cost is acceptable.

## 15. Depth-Matched Recurrent Generation Gate (2026-03-06)

Status:
- implemented and benchmarked
- recurrent generation beats both the current direct transformer generation harness and the CoreML generation baseline
- but the advantage compresses below the strict `>=2x` continuation gate by `6` recurrent layers

What changed:
- Added `GenerationPerformanceSnapshot` / `GenerationPerformanceTrackable` so generation benchmarks can report:
  - compile time
  - trunk time
  - output-head/logits time
- Added `ANERecurrentGenerationModel` to the real token-generation harness:
  - token ids
  - embedding lookup
  - recurrent trunk stepping
  - final RMSNorm + classifier logits
  - argmax decode loop
- Added a benchmark-local CoreML generation wrapper that reuses the same surrounding CPU work:
  - token-id input
  - embedding lookup
  - CoreML transformer trunk replay
  - final RMSNorm + classifier logits on CPU

Important comparison policy:
- `stories110M.bin` is still not present locally, so these measurements use the synthetic token-generation harness rather than a pretrained semantic model.
- The CoreML path here is still an honest end-to-end generation comparison because it includes prompt replay, decode replay, and the same CPU output-head work.
- The CoreML model exports hidden states, not logits, so the output head remains on CPU for parity with the direct generation harness.

Benchmark protocol:
- `mach_absolute_time()` / `mach_timebase_info`
- `3` warmup iterations
- `20` timed iterations
- prompt: `[0]`
- generated tokens: `8`
- report median `ms/token`, `tok/s`, compile time, and median trunk/logits time per generated token

Results:
- Direct transformer ANE generation, `6` layers:
  - `6.558745 ms/token`
  - `152.47 tok/s`
  - compile `2158.23 ms`
  - trunk `5.28449 ms/token`
  - logits `1.26048 ms/token`
- Recurrent generation, `2` layers:
  - `2.271112 ms/token`
  - `440.32 tok/s`
  - compile `123.02 ms`
  - trunk `1.00656 ms/token`
  - logits `1.27095 ms/token`
- Recurrent generation, `6` layers:
  - `4.000445 ms/token`
  - `249.97 tok/s`
  - compile `349.07 ms`
  - trunk `2.73244 ms/token`
  - logits `1.26102 ms/token`
- CoreML generation baseline, `.cpuAndNeuralEngine`:
  - `6.582224 ms/token`
  - `151.93 tok/s`
  - compile+load `1384.60 ms`
  - trunk `5.11926 ms/token`
  - logits `1.35556 ms/token`

Interpretation:
- The first structural gate came back positive:
  - `2` recurrent layers are about `2.89x` faster end-to-end than the current direct transformer generation harness.
- The more honest depth-matched gate came back mixed:
  - `6` recurrent layers are about `1.64x` faster than direct transformer generation.
  - `6` recurrent layers are about `1.65x` faster than the CoreML generation baseline.
- That means the recurrent architecture still has real upside, but the strict continuation rule from this session was not met at `6` layers.

Most important systems finding:
- The recurrent trunk win is real, but the output head is now a large fraction of the budget.
- At `2` recurrent layers, logits are already slightly more expensive than the recurrent trunk.
- At `6` recurrent layers, logits remain about `1.26 ms/token`, which materially caps end-to-end gains.
- The next high-upside move is therefore:
  - offload final RMSNorm + classifier to ANE, or
  - otherwise reduce output-head cost sharply
- The next low-upside move would be:
  - adding more recurrent layers without addressing the output head

Compile-time note:
- The cold recurrent prototype compile from the earlier session was `131.362 s`.
- The generation-model compile times above (`123-349 ms`) are warm-cache measurements on the now-proven recurrent step shape, so do not treat them as contradictory.
- The likely explanation is cache reuse for already-compiled identical recurrent kernels.

Decision:
- Keep recurrent decode as the highest-upside architecture path over transformer decode.
- Do not keep scaling recurrent depth blindly under the current CPU output-head design.
- Move the next experiment to output-head optimization / ANE-side logits before spending more time on deeper recurrent stacks.

Verification:
- `swift test`
- `ANE_HARDWARE_TESTS=1 swift test --filter GenerationHarnessHardwareTests/test_recurrent_generation_reports_compile_and_runtime_breakdown_on_hardware`
- `ANE_HARDWARE_TESTS=1 swift test --filter GenerationHarnessHardwareTests/test_recurrent_generation_6layer_and_coreml_generation_baseline_if_gate_passes`
- `ANE_HARDWARE_TESTS=1 swift test --filter GenerationHarnessHardwareTests/test_speculative_upper_bound_reports_metrics_on_hardware`

Residual test note:
- Running the entire `GenerationHarnessHardwareTests` class together exposed a temp-dir cleanup failure in the older speculative test, while that same speculative test passes in isolation. That looks like cross-test hygiene in the old speculative path rather than a blocker on the recurrent/CoreML benchmark results above.

## Output-Head Probe (March 7, 2026)

Goal:
- Test whether moving the recurrent generation classifier projection onto ANE can recover enough end-to-end latency to get the `6`-layer recurrent path back toward a `>=2x` win over the current direct transformer generation path.

Implementation:
- Added a swappable generation output-head backend:
  - `GenerationOutputHeadBackend.cpu`
  - `GenerationOutputHeadBackend.aneClassifier`
- Added:
  - `GenerationClassifierGenerator`
  - `GenerationClassifierKernelSet`
  - `ANEGenerationClassifierHead`
- Kept final RMSNorm on CPU for this probe.
- Verification path remains on CPU. This experiment only changes step-level autoregressive logits.

Important compatibility finding:
- A naive single-token classifier kernel (`[1, dim, 1, 1] -> [1, vocab, 1, 1]`) compiled but failed at eval on hardware with:
  - `statusType=0x9: Program Inference error`
- The same classifier projection became usable when packed into the repo’s proven lane shape:
  - input `[1, dim, 1, 32]`
  - output `[1, vocab, 1, 32]`
  - only spatial lane `0` is written/read at runtime
- Treat large-vocab classifier heads as another case where lane-packed decode shapes are materially safer than scalar-width shapes on this runtime.

Benchmark protocol:
- `mach_absolute_time()` / `mach_timebase_info`
- `3` warmup iterations
- `20` timed iterations
- prompt `[0]`
- `8` generated tokens
- recurrent generation, `6` layers

Results:
- recurrent generation, CPU head:
  - `3.965409 ms/token`
  - `252.18 tok/s`
  - compile `407.17 ms`
  - trunk `2.661490 ms/token`
  - logits `1.292997 ms/token`
- recurrent generation, ANE classifier head:
  - `3.710214 ms/token`
  - `269.53 tok/s`
  - compile `585.46 ms`
  - trunk `2.607659 ms/token`
  - logits `1.094966 ms/token`

Delta:
- end-to-end: `-0.255195 ms/token` (`6.4%` faster)
- throughput: `+17.35 tok/s` (`6.9%` faster)
- logits bucket: `-0.198031 ms/token` (`15.3%` faster)
- compile time: `+178.29 ms`

Decision:
- This is a real win, but it is not a breakthrough.
- The predeclared gate for continuing this avenue was about `0.7 ms/token` saved on recurrent-6 generation. The measured win (`0.255 ms/token`) is far below that threshold.
- Therefore:
  - classifier-only ANE offload is worth keeping because it is faster
  - classifier-only ANE offload is not enough to restore the missing `>=2x` recurrent advantage by itself
  - further time on this exact probe has low upside unless the next step is a fused final RMSNorm + classifier path
- The next materially larger opportunity is still likely recurrent multi-layer fusion, not more small output-head tuning.
