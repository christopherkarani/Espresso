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

## 2026-03-10 — Rejected contiguous-shard CPU staged exact head

### Attempt

Added an exact branch-and-bound output-head seam for direct-select generation:

- stage 1: compute admissible per-shard upper bounds from shard center/radius summaries on the normalized token vector
- stage 2: run exact CPU BLAS scoring only on shards whose bound can still beat the current best exact score
- tie policy preserved the current first-max argmax behavior

This was intentionally CPU-routed for the first probe because the branch already had strong evidence that extra ANE head staging/regrouping regresses and full fused recurrent+head attachment is compiler-blocked.

### Hardware measurement

Matched recurrent fused-triplet direct-select comparison (`warmup=3`, `iterations=20`, `maxNewTokens=8`, echo weights):

| Path | Median ms/token | tok/s | Compile/init ms | Trunk ms/token | Head/logits ms/token |
|---|---:|---:|---:|---:|---:|
| Control: fused-triplet + `.aneRMSNormClassifier` | `2.3946458333333336` | `417.60` | `542.6085` | `1.259015625` | `1.0993333333333333` |
| Contiguous-shard staged exact CPU head | `28.1385625` | `35.54` | `5176.693166666667` | `1.7956223958333335` | `26.325609375` |

Exactness/parity check:

- generated tokens matched the control on the hardware echo test
- the path is exact, but not competitive

### Interpretation

This avenue is rejected in its contiguous-shard form.

- The shard bounds were too loose to prune enough work.
- The runtime effectively degenerated into repeated CPU block GEMMs, so head time exploded instead of shrinking.
- Init/compile time also regressed badly because the staged head now has to build the shard summaries up front.

### Decision

- keep the exact branch-and-bound seam and tests as reference infrastructure
- reject contiguous-shard CPU staged exact head as a throughput path
- if exact head work continues, it must use materially better block geometry (for example clustered blocks with admissible bounds), not contiguous shards
- otherwise move to the next architecture class rather than spending more time tuning this path

## 2026-03-10 — Rejected clustered exact CPU staged head

### Attempt

Replaced contiguous token blocks with deterministic non-contiguous clustered blocks:

- projected clustering over classifier rows
- exact full-dimension center/radius summaries per cluster
- exact block scoring only on clusters whose upper bound could still beat the current best score

This preserved exact argmax semantics while materially improving the geometry over contiguous shards.

### Hardware measurement

Matched recurrent fused-triplet direct-select comparison (`warmup=3`, `iterations=20`, `maxNewTokens=8`, echo weights):

| Path | Median ms/token | tok/s | Compile/init ms | Trunk ms/token | Head/logits ms/token |
|---|---:|---:|---:|---:|---:|
| Control: fused-triplet + `.aneRMSNormClassifier` | `2.7799114583333333` | `359.72` | `32466.812541666666` | `1.46640625` | `1.3099739583333334` |
| Clustered exact CPU staged head | `7.072252604166667` | `141.40` | `10454.488791666667` | `1.738265625` | `5.315468750000001` |

### Interpretation

- Better geometry helped a lot relative to contiguous shards (`7.07` vs `28.14 ms/token`).
- It still lost badly to the current exact control.
- The head/logits bucket remained the dominant regression, so the family is still not competitive for single-stream decode.

### Decision

- reject clustered exact CPU staged head as the next performance path
- keep the clustered implementation only as evidence that geometry mattered but did not change the ranking
- move to a recurrent-native multi-token architecture instead of spending more time on this CPU staged exact-head family
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

## Fused Output-Head Follow-up (March 7, 2026)

Goal:
- Test the only still-credible head-side follow-up after classifier-only ANE offload: fuse final RMSNorm and classifier into one lane-packed ANE head.

Implementation:
- Added:
  - `GenerationRMSNormClassifierGenerator`
  - `GenerationRMSNormClassifierKernelSet`
  - `GenerationOutputHeadBackend.aneRMSNormClassifier`
- Kept the same proven lane-packed shape:
  - input `[1, dim, 1, 32]`
  - output `[1, vocab, 1, 32]`
  - write/read only spatial lane `0`
- Verification stays on CPU. This is still a step-level autoregressive head path only.

Benchmark protocol:
- `mach_absolute_time()` / `mach_timebase_info`
- `3` warmup iterations
- `20` timed iterations
- prompt `[0]`
- `8` generated tokens
- recurrent generation, `6` layers

Results:
- recurrent generation, CPU head:
  - `4.128354 ms/token`
  - `242.23 tok/s`
  - compile `445.79 ms`
  - trunk `2.721484 ms/token`
  - head `1.375232 ms/token`
- recurrent generation, ANE classifier head:
  - `3.761755 ms/token`
  - `265.83 tok/s`
  - compile `634.18 ms`
  - trunk `2.655479 ms/token`
  - head `1.098206 ms/token`
- recurrent generation, fused ANE RMSNorm+classifier head:
  - `3.564776 ms/token`
  - `280.54 tok/s`
  - compile `612.15 ms`
  - trunk `2.615005 ms/token`
  - head `0.936549 ms/token`

Delta:
- fused vs classifier-only ANE head:
  - `-0.196979 ms/token` (`5.24%` faster)
  - `+14.71 tok/s` (`5.53%` faster)
  - head bucket improved by `0.161656 ms/token`
- fused vs CPU head:
  - `-0.563578 ms/token` (`13.65%` faster)
- remaining gap to `>=2x` over direct transformer generation (`6.558745 ms/token`):
  - about `0.285404 ms/token`

Interpretation:
- The user instinct was directionally correct: there is still real signal in the head.
- Fusing RMSNorm into the ANE head produced another measurable win beyond classifier-only offload.
- But the follow-up still missed the predeclared continuation gate of `>=0.25 ms/token` additional end-to-end savings over the current ANE classifier head.

Most likely next limiter:
- The fused head still returns full-vocab logits to the host every step.
- That means the next head-side win probably needs a different mechanism:
  - on-device token selection / argmax
  - or another reduction that materially shrinks logits readback

Decision:
- Keep the fused head because it is faster.
- Do not spend unbounded time tuning this same full-logits-return path further.
- If head-side work continues, make it a bounded probe around avoiding full-vocab logits readback.
- Otherwise, return to the higher-upside path: recurrent multi-layer fusion.

## Reduced Readback / Direct Surface Argmax (March 7, 2026)

Goal:
- Test the next honest head-side follow-up after the fused ANE `RMSNorm + classifier` head:
  avoid full-vocab logits materialization for `argmax` generation.

Hypothesis:
- The fused head made the classifier path fast enough that reading a full `vocab`-length logits vector back to CPU every step is now a credible limiter.
- A bounded host-side direct argmax scan on the ANE output IOSurface can answer that question without betting on unsupported MIL `argmax` / `topk` ops.

Implementation:
- Added a small C interop helper:
  - `ane_interop_io_argmax_fp16_spatial_slice`
- Exposed it in Swift:
  - `SurfaceIO.argmaxFP16SpatialSlice(...)`
- Added direct-selection helpers to:
  - `ANEGenerationClassifierHead`
  - `ANEGenerationRMSNormClassifierHead`
- Added an explicit generation harness for models that can return the selected token directly:
  - `DirectTokenSelectionGenerationHarness`
- Kept the existing `AutoregressiveGenerationHarness` unchanged as the materialized-logits control.

Correctness checks:
- `SurfaceIO` direct argmax matches materialized-lane argmax in unit tests.
- The direct-selection harness preserves token outputs in unit tests.
- On hardware, the direct-selection and materialized fused-head paths produced the same echo tokens before benchmarking.

Benchmark setup:
- recurrent generation
- `6` layers
- fused ANE `RMSNorm + classifier` head
- prompt `[0]`
- `8` generated tokens
- `3` warmup
- `20` timed iterations
- median reported

Results:
- Materialized logits control:
  - `3.568122 ms/token`
  - `280.26 tok/s`
  - compile `663.66 ms`
  - trunk `2.614922 ms/token`
  - head `0.946177 ms/token`
- Direct surface argmax:
  - `2.467362 ms/token`
  - `405.29 tok/s`
  - compile `659.94 ms`
  - trunk `1.700138 ms/token`
  - head `0.768669 ms/token`

Replications:
- `3.620622 -> 2.454737 ms/token`
- `3.637203 -> 2.451047 ms/token`

Delta from the parity+benchmark run:
- `1.100760 ms/token` faster end-to-end (`30.85%`)
- `125.03 tok/s` faster (`44.61%`)
- head bucket down by `0.177508 ms/token`
- observed trunk bucket down by `0.914784 ms/token`

Interpretation:
- The reduced-readback path clearly removed real head-side overhead.
- The total end-to-end win is much larger than the head-bucket reduction alone.
- Inference from the data:
  the shorter head-side path also appears to reduce host-side delay enough to improve the observed recurrent trunk cadence in this autoregressive harness.

Decision:
- This probe decisively passed the continuation gate.
- Avoiding full-vocab logits materialization is now a proven performance avenue, not just a hypothesis.
- Do not switch away from head-side work yet.
- The next honest head-side probes are:
  - reuse one read lock and stream chunked reads if another small gain remains
  - or move token selection further toward minimal-return / on-device selection
- Recurrent multi-layer fusion still matters, but it is no longer the automatic next step after the fused head.

## Recurrent Multi-Layer Fusion Follow-up (March 7, 2026)

Goal:
- Push the recurrent direct-select path from the reduced-readback breakthrough (`~2.45-2.47 ms/token`) to a real `3x`-over-CoreML result by removing more recurrent eval boundaries.

### Fused Two-Layer Recurrent Step

Implementation:
- Added:
  - `RWKVStyleFusedTwoLayerStepGenerator`
  - `RWKVStyleFusedTwoLayerKernelSet`
  - `RWKVStyleFusedTwoLayerSession`
- Extended `ANERecurrentGenerationModel` with `RecurrentGenerationTrunkBackend` and a `.fusedTwoLayerPairs` mode.
- Kept the output-head breakthrough intact:
  - fused ANE `RMSNorm + classifier`
  - direct surface argmax / direct token selection

Correctness:
- New generator and kernel compile-spec tests pass.
- Hardware parity test confirms fused-pair direct-select emits the same echo tokens as the single-layer-stack recurrent control.

Hardware compare:
- Fused-pair runs were noisy in absolute terms but directionally consistent.
- Observed direct-select compare runs:
  - single-layer recurrent control `6.0231 ms/token` vs fused-pair `3.7812 ms/token`
  - single-layer recurrent control `3.4296 ms/token` vs fused-pair `2.3074 ms/token`
  - single-layer recurrent control `4.5684 ms/token` vs fused-pair `2.5795 ms/token`

Interpretation:
- The absolute harness is still noisy enough that single-run claims would be dishonest.
- The direction is clear: fused recurrent pairs materially reduce both trunk time and end-to-end token latency.
- This passed the material-gain gate and justified one more bounded push.

### Smaller Trunk Lane Widths

Hypothesis:
- The recurrent trunk still runs lane-packed surfaces with `laneSpatial=32` while only lane `0` carries live token data, so reducing trunk lane width might recover the remaining `~0.11-0.39 ms/token`.

Result:
- The lane-32 fused-pair baseline still ran in the sweep harness:
  - `2.530901 ms/token`
  - `395.14 tok/s`
- The first smaller trunk lane width immediately hit:
  - `statusType=0x9`

Decision:
- Treat smaller recurrent trunk lane widths as structurally blocked for this MIL shape/runtime.
- Do not spend more time on trunk lane packing reduction unless the graph changes materially.

### Fused Three-Layer Recurrent Step

Implementation:
- Added:
  - `RWKVStyleFusedThreeLayerStepGenerator`
  - `RWKVStyleFusedThreeLayerKernelSet`
  - `RWKVStyleFusedThreeLayerSession`
- Extended `ANERecurrentGenerationModel` with `.fusedThreeLayerTriplets`.
- For a `6`-layer recurrent trunk, this reduces the direct-select recurrent hot path to `2` fused evals instead of `3`.

Correctness:
- New generator and kernel compile-spec tests pass.
- Non-hardware validation rejects non-multiples of `3`.
- Hardware parity test confirms fused-triplet direct-select matches fused-pair direct-select token outputs.

Hardware compare:
- Fused pair direct-select:
  - `2.640195 ms/token`
  - `378.76 tok/s`
  - compile `523.39 ms`
  - trunk `1.447555 ms/token`
  - logits `1.140951 ms/token`
- Fused triplet direct-select:
  - `2.211750 ms/token`
  - `452.13 tok/s`
  - compile `560.54 ms`
  - trunk `1.139904 ms/token`
  - logits `1.090958 ms/token`

Replication:
- second run:
  - fused pair `2.474672 ms/token`
  - fused triplet `2.211995 ms/token`
- a noisier third run still preserved direction:
  - fused pair `3.316388 ms/token`
  - fused triplet `2.468479 ms/token`

Interpretation:
- Fused triplets are the best recurrent result on this branch so far.
- The strongest repeated fused-triplet runs are clustered around:
  - `2.212 ms/token`
  - `~452 tok/s`
- Against the standing CoreML generation baseline (`6.582224 ms/token`, `151.93 tok/s`), that is about:
  - `2.98x` faster
- This is an honest near-`3x` result, but not a stable `>=3x` claim yet.

### Output-Head Lane Width Follow-up on Fused Triplets

Hypothesis:
- After triplet fusion, the remaining gap to `3x` is tiny enough that a smaller fused ANE output-head lane width might close it.

Result:
- Baseline fused-triplet direct-select with output-head lane `32` in the sweep harness:
  - `2.204732 ms/token`
  - `453.57 tok/s`
  - compile `614.12 ms`
  - trunk `1.159018 ms/token`
  - logits `1.051698 ms/token`
- Smaller fused output-head lane widths were all unsupported on hardware:
  - lane `16`: ANE fused output-head eval failed
  - lane `8`: ANE fused output-head eval failed
  - lane `1`: ANE fused output-head eval failed

Decision:
- Keep output-head lane width at `32`.
- Treat smaller fused output-head lanes as blocked for this graph/runtime.

### Current Honest Status

- Best branch result:
  - fused-triplet recurrent direct-select around `2.205-2.212 ms/token`
  - `~452-454 tok/s`
- Relative to CoreML generation baseline:
  - roughly `2.98x-2.99x`
- Relative to the older direct transformer generation harness (`6.558745 ms/token`):
  - roughly `2.97x`

Conclusion:
- Multi-layer recurrent fusion is the first path on this branch that gets genuinely close to a `3x` ANE-over-CoreML generation result.
- The remaining gap is now small enough that further work would be micro-optimization territory, not another large architectural unlock.
- The bounded micro-paths tested after triplet fusion were both blocked:
  - smaller recurrent trunk lane widths -> `statusType=0x9`
  - smaller fused output-head lane widths -> ANE eval failure / `statusType=0x9`
- So the honest call is:
  - this branch reached an optimized near-`3x` state
  - but did not produce a stable, repeatable `>=3x` claim on the current benchmark harness

## 13. Concurrent Multistream Generation Benchmark (2026-03-08)

What was built:
- A true concurrent multistream generation benchmark in `GenerationHarnessHardwareTests`.
- ANE path: recurrent fused-triplet direct-select with fused ANE RMSNorm + classifier head.
- CoreML path: separate `CoreMLGenerationBenchmarkModel` instance per stream.
- Each stream owns isolated state, model/runtime setup, and a dedicated host queue.
- Each timed round launches one full generation request per stream concurrently and measures wall-clock round time.
- Reporting normalizes by `streams * maxNewTokens` so throughput stays honest.

Benchmark contract:
- Prompt: `[0]`
- Decode length: `8` new tokens
- Warmup: `3` rounds
- Timed: `20` rounds
- Stream counts: `1`, `2`, `3`, `4`
- Timing: `mach_absolute_time()` via the existing generation clock path

Single-stream control rerun before the new benchmark:
- Best ANE direct-select control: `2.211003 ms/token`, `452.28 tok/s`
- Standing CoreML generation control: `7.044784 ms/token`, `141.95 tok/s`

Repeated matched concurrent runs:

| Streams | ANE ms/token (run 1) | ANE tok/s (run 1) | ANE ms/token (run 2) | ANE tok/s (run 2) | CoreML ms/token (run 1) | CoreML tok/s (run 1) | CoreML ms/token (run 2) | CoreML tok/s (run 2) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2.294083 | 435.90 | 2.195065 | 455.57 | 15.377154 | 65.03 | 15.450174 | 64.72 |
| 2 | 1.183065 | 845.26 | 1.196073 | 836.07 | 8.187806 | 122.13 | 8.299697 | 120.49 |
| 3 | 0.885752 | 1128.98 | 0.874269 | 1143.81 | 5.583365 | 179.10 | 5.537411 | 180.59 |
| 4 | 0.808975 | 1236.13 | 0.814136 | 1228.30 | 4.637305 | 215.64 | 4.643027 | 215.38 |

Scaling analysis against each path's own matched 1-stream concurrent baseline:
- ANE aggregate scaling (run 2):
  - `2 streams`: `1.84x`
  - `3 streams`: `2.51x`
  - `4 streams`: `2.70x`
- CoreML aggregate scaling (run 2):
  - `2 streams`: `1.86x`
  - `3 streams`: `2.79x`
  - `4 streams`: `3.33x`
- Concurrency-scaling ratio gate (`ANE scaling / CoreML scaling > 1.34`) failed:
  - `2 streams`: about `0.99`
  - `3 streams`: about `0.90`
  - `4 streams`: about `0.81`

Interpretation:
- Absolute ANE serving throughput increased substantially with stream count, reaching about `1.23k tok/s` at `4` streams.
- The matched concurrent benchmark also shows ANE above `4x` CoreML in absolute throughput at every stream count.
- However, this is **not** evidence that true multistream concurrency is the unlocking lever toward `4x`.
- The key reason is methodological: the matched concurrent CoreML setup regressed sharply already at `1` stream (`~15.4 ms/token`) relative to the standing single-stream CoreML control (`~7.04 ms/token`), while ANE's `1`-stream concurrent measurement stayed close to its prior control.
- Inference: the queue-isolated concurrent serving harness changes CoreML's single-stream behavior materially, so the >`4x` absolute ratio here comes mostly from the changed serving contract rather than from ANE gaining a superior concurrency scaling factor.
- The primary hypothesis therefore fails: ANE does **not** scale materially better than CoreML under this benchmark. CoreML scales similarly or better from its own matched concurrent baseline.

Decision:
- Status: implemented, measured, and stopped.
- Keep the benchmark as a serving-path measurement tool.
- Do **not** use this avenue as the main causal explanation for a credible `4x over CoreML` claim.
- Next best move returns to the fallback plan: reduce remaining direct-select host overhead on the ANE path, and treat the queue-induced CoreML regression as a separate measurement artifact to characterize if a serving claim is needed later.
## 2026-03-08 - Direct-select argmax micro-optimization

- Change: optimized `ane_interop_io_argmax_fp16_spatial_slice` in `Sources/ANEInterop/surface_io.c`
  - switched from scalar `float`-converting scan to pointer-stride arithmetic
  - unrolled the channel walk `4x`
  - kept comparisons in `_Float16` and converted to `float` only for the final reported value
- Guard tests:
  - `swift test --filter ANETypesTests/test_surface_argmax_fp16_spatial_slice_matches_materialized_argmax`
  - `swift test --filter ANETypesTests/test_surface_argmax_fp16_spatial_slice_respects_channel_offset_and_tail`
- Baseline before change:
  - fused-triplet direct-select `2.2018489583333336 ms/token`, `454.16396411257176 tok/s`
- Post-change repeated runs:
  - `2.2109479166666666 ms/token`, `452.29748776856803 tok/s`
  - `2.154953125 ms/token`, `464.0636883108018 tok/s`
  - `2.159026041666667 ms/token`, `463.1718306072166 tok/s`
- Post-change median across the three repeated runs:
  - `2.159026041666667 ms/token`, `463.1718306072166 tok/s`
- Delta versus the fresh baseline:
  - `0.04282291666666665 ms/token` faster
  - about `1.94%` lower latency
  - about `9.01 tok/s` higher throughput
- Interpretation:
  - measured win is small but positive and above the immediate regression/no-regression threshold for this avenue
  - most of the remaining single-stream budget is still in trunk + output-head boundaries, so this is a saved micro-gain rather than a path-changing breakthrough
## 2026-03-08 - Direct-select argmax ILP widening

- Change: widened `ane_interop_io_argmax_fp16_spatial_slice` to keep `8` independent maxima in flight before a stable first-max reduction
  - goal: reduce loop-carried dependency in the `32_000`-channel, stride-`32` scan used by the current best direct-select head
  - preserved first-max semantics explicitly during the final reduction
- Guard tests:
  - `swift test --filter ANETypesTests`
  - added `ANETypesTests/test_surface_argmax_fp16_spatial_slice_prefers_first_max_across_unrolled_groups`
- Control before this change:
  - median from the previous saved argmax optimization: `2.159026041666667 ms/token`
- Post-change repeated runs:
  - `2.129125 ms/token`, `469.67654812253164 tok/s`
  - `2.1766796875 ms/token`, `459.42070485198116 tok/s`
  - `2.1179270833333335 ms/token`, `472.15979807418256 tok/s`
- Post-change median across the three repeated runs:
  - `2.129125 ms/token`, `469.67654812253164 tok/s`
- Delta versus the prior control:
  - `0.02990104166666683 ms/token` faster
  - about `1.38%` lower latency
- Cumulative direct-select argmax savings from the fresh `2026-03-08` pre-optimization baseline:
  - baseline `2.2018489583333336 ms/token`
  - current `2.129125 ms/token`
  - cumulative saved `0.07272395833333361 ms/token`
  - cumulative reduction about `3.30%`
- Interpretation:
  - another real but still small micro-gain
  - this confirms the output-surface argmax path is movable, but not enough by itself to close the remaining gap to `6x`
## 2026-03-08 - Rejected lane-0-only recurrent state/input copies

- Hypothesis:
  - because the current fused-triplet direct-select path only consumes lane `0`, per-token full-width `xIn` clear and `stateOut -> stateIn` copies in `RWKVStyleFusedThreeLayerSession.step` might be reducible to lane-`0` slice operations only
- Guard:
  - added hardware parity test `test_recurrent_generation_fused_triplet_direct_select_vs_autoregressive_materialized_on_hardware`
  - parity stayed green, so semantics on the echo path did not obviously break
- Rejected implementation:
  - removed per-token full `xIn` clear
  - narrowed the three recurrent state carry copies to `SurfaceIO.copyFP16SpatialSlice(... spatialIndex: 0 ...)`
- Measured result:
  - control before change: `2.129125 ms/token`
  - post-change runs:
    - `2.243302083333333 ms/token`
    - `2.1020781250000002 ms/token`
    - `2.1712968750000003 ms/token`
  - post-change median: `2.1712968750000003 ms/token`
  - delta: about `0.04217187500000029 ms/token` slower (`~1.98%` regression)
- Interpretation:
  - semantic parity on the echo path was not enough; the narrowed copy path regressed throughput and was removed
  - inference: either the full-width copies are helping memory locality/cache behavior, or the slice-copy path itself is more expensive than expected
## 2026-03-08 - Blocked full fused triplet + head session

- Goal:
  - fuse the final recurrent triplet and the ANE RMSNorm+classifier head into one kernel that returns:
    - `xNext`
    - `stateOut0`
    - `stateOut1`
    - `stateOut2`
    - `logits`
- Why:
  - this was the first boundary-removal path with enough upside to matter for `4x` and `6x`
  - it could have removed the trunk→head dispatch boundary and one host-side handoff
- What was built:
  - committed scaffolding:
    - `RWKVStyleFusedThreeLayerRMSNormClassifierGenerator`
    - `RWKVStyleFusedThreeLayerRMSNormClassifierKernelSet`
    - generator/kernelset contract tests
  - attempted runtime/session wiring:
    - direct hardware smoke test for a new fused session
    - temporary fused session/runtime implementation
- What failed:
  - first failure: shared-classifier recurrent weights exposed an empty `classifier` buffer
    - resolved by following the existing `.aneRMSNormClassifier` rule: use `embedding` when `sharedClassifier == true`
  - second failure: even after that fix, ANE compile still failed with:
    - `_ANECompiler : ANECCompile() FAILED`
    - `InvalidMILProgram`
- Materially different implementation variants tried:
  - initial string-splice composition using the existing triplet generator plus the existing RMSNorm+classifier tail
  - replacement hand-built RMSNorm+classifier tail that reused the triplet generator's shared constants and conv parameters instead of duplicating them
- Result:
  - both variants still failed at ANE compile time with `InvalidMILProgram`
  - runtime/session wiring for this exact `5`-output fused path was rolled back
- Decision:
  - do not keep pushing this exact full-session `5`-output fused path
  - next materially different attempt is narrower:
    - direct-select-only final-triplet fusion
    - omit `xNext` on the last fused block
    - return only recurrent carry state plus logits
## 2026-03-08 - Blocked direct-select-only final-triplet fusion

- Goal:
  - try a materially different narrower fusion after the blocked full-session path
  - final fused triplet returns only:
    - `stateOut0`
    - `stateOut1`
    - `stateOut2`
    - `logits`
  - intentionally omit `xNext` on the last fused block for the direct-select path
- Why:
  - lower output arity than the blocked `5`-output fused session
  - better performance shape even if it compiled, because it would remove the final hidden-state readback
- What was built:
  - direct-select-only generator and kernelset contract tests
  - direct-select-only generator and kernelset implementation
  - hardware compile smoke for the new kernelset
- Result:
  - compile-spec tests passed
  - hardware compile still failed with:
    - `_ANECompiler : ANECCompile() FAILED`
    - `InvalidMILProgram`
- Interpretation:
  - inference: appending the full vocab head to the fused recurrent triplet appears structurally blocked on the current ANE compiler path, not just because of the extra `xNext` output
  - this materially lowers confidence in further "triplet + 32k classifier" full fusion variants
- Decision:
  - roll back the direct-select-only fusion scaffolding
  - move on to a different class of avenue rather than spending more time on the same compiler wall

## 2026-03-08 - Rejected exact sharded RMSNorm+classifier head

Why tried:
- Cheapest exact staged-head probe after the triplet+head fusion wall.
- Reused the existing `GenerationRMSNormClassifierKernelSet` with smaller vocab shards and merged top-1 by score.
- Goal was to see whether smaller classifier convs beat the current monolithic head despite extra dispatches.

Method:
- Added an opt-in `ANE_RMS_HEAD_SHARD_VOCAB` gate for the ANE RMSNorm+classifier output head.
- Kept the recurrent fused-triplet direct-select trunk unchanged.
- Benchmarked `6` layers, prompt `[0]`, `8` generated tokens, `3` warmups, `20` timed iterations, median `ms/token`.
- Verified parity against the baseline path in the same hardware test.

Results:
- Baseline: compile `4189.8249166666665 ms`, runtime `2.2191796875 ms/token`, `450.6169579834891 tok/s`
- Sharded `16384` (2 shards): compile `4218.095541666667 ms`, runtime `2.4288697916666666 ms/token`, `411.7141245821209 tok/s`
- Sharded `8192` (4 shards): compile `4254.90175 ms`, runtime `2.6128776041666666 ms/token`, `382.7198022614355 tok/s`
- Sharded `4096` (8 shards): compile `4445.996916666667 ms`, runtime `3.6560390624999997 ms/token`, `273.5200535073605 tok/s`

Interpretation:
- Strong negative result.
- Inference: dispatch and repeated surface I/O dominate any benefit from smaller classifier shards.
- The regression increases with shard count, so this is not a near-miss tuning issue.
- Reverted the implementation and tests after measurement.

## 2026-03-08 - Rejected unlocked-argmax direct-select path

Why tried:
- Smallest exact surface-I/O probe after ruling out smaller output-head lane widths.
- Added an unlocked argmax interop path to remove the read-lock work from inside the hot argmax helper while preserving exact token selection.

Method:
- Added `ane_interop_io_argmax_fp16_spatial_slice_unlocked` plus Swift wrapper.
- Gated the direct-select path with `ANE_DIRECT_SELECT_UNLOCKED_ARGMAX=1`.
- Added parity coverage at the `SurfaceIO` level and benchmarked recurrent fused-triplet direct-select on hardware.
- Benchmark shape: `6` layers, prompt `[0]`, `8` generated tokens, `3` warmups, `20` timed iterations, median `ms/token`.

Results:
- Run 1 baseline: compile `1902.6328333333333 ms`, runtime `2.2433463541666665 ms/token`, `445.76264300100416 tok/s`
- Run 1 unlocked argmax: compile `1839.1164166666667 ms`, runtime `2.3161380208333338 ms/token`, `431.7531990775772 tok/s`
- Run 2 baseline: compile `1902.8605833333334 ms`, runtime `2.2317421875 ms/token`, `448.0804304372635 tok/s`
- Run 2 unlocked argmax: compile `1864.6896666666667 ms`, runtime `2.235598958333333 ms/token`, `447.3074190128951 tok/s`

Interpretation:
- Negative result.
- Runtime improvement did not reproduce; the path was either slightly slower or effectively flat.
- Inference: removing the lock work from inside the argmax helper is not enough to move end-to-end throughput on the current path.
- Reverted the implementation and temporary tests after measurement.

## 2026-03-08 - Blocked 4+2 recurrent trunk fusion

Why tried:
- Next lower-risk trunk-side experiment beyond fused triplets.
- Used a new fused four-layer recurrent block composed with the existing fused two-layer block, while keeping the current direct-select head unchanged.

Implementation slice:
- Added `RWKVStyleFusedFourLayerStepGenerator` and `RWKVStyleFusedFourLayerKernelSet`.
- Added a composed `RWKVStyleFusedFourPlusTwoSession` and a new `.fusedFourPlusTwo` trunk backend in the generation harness.
- Added compile-spec/unit tests for the 4-layer generator/kernelset and a non-multiple-of-6 harness guard.

What passed:
- `swift test --filter 'RWKVStyleFusedFourLayer|test_recurrent_generation_rejects_non_multiple_of_six_for_fused_four_plus_two_backend'`
- Four-layer MIL generator contract tests
- Four-layer kernelset compile-spec test
- Harness validation for the multiple-of-6 requirement

Where it failed:
- First hardware smoke/benchmark for `fusedFourPlusTwo` failed during ANE compile.
- Error:
  - `_ANECompiler : ANECCompile() FAILED`
  - `InvalidMILProgram`

Interpretation:
- Strong structural negative result.
- Inference: extending recurrent trunk fusion beyond current triplets hits the same compiler wall seen in other larger-fusion attempts on this branch.
- Reverted the implementation after the first blocked hardware attempt.

## 2026-03-08 - Blocked full six-layer recurrent fusion

What was tried:
- Fused all six recurrent RWKV-style layers into one MIL program while keeping the current direct-select output head unchanged.
- Added a dedicated fused-six generator, kernelset, session, harness backend, unit contracts, and a hardware comparison test against the saved fused-triplet direct-select path.

Why:
- After the blocked `4+2` and triplet+head fusion attempts, full-six fusion was the next materially different trunk-side probe that could still remove one more inter-session boundary if the compiler accepted it.

Baseline before the probe:
- fused-triplet direct-select: `2.2336927083333333 ms/token`, `447.6892970440778 tok/s`
- compile: `559.7429583333334 ms`
- trunk: `1.1575755208333334 ms/token`
- logits: `1.0688463541666668 ms/token`

Measured result:
- Unit contracts passed.
- First hardware compile failed immediately with `_ANECompiler : ANECCompile() FAILED` and `InvalidMILProgram`.
- No steady-state runtime measurement exists because the fused-six kernel never compiled.

Interpretation:
- Inference: extending recurrent trunk fusion from triplets to a single six-layer block hits the same compiler wall as other larger-fusion attempts on this branch.
- This materially lowers confidence in any further monolithic recurrent-fusion path that keeps the same op family and lane geometry.
- The right next move is a materially different route, not more compile archaeology on larger recurrent blocks.

## 2026-03-08 - Larger fused output-head lanes regress

What was tried:
- Extended the fused-triplet direct-select output-head lane sweep upward from the existing `32/16/8/1` set to include larger lane widths: `64`, `96`, and `128`.
- Kept the recurrent fused-triplet trunk unchanged and benchmarked each lane in the same hardware sweep harness.

Why:
- Smaller output-head lanes were already known to fail or regress.
- The remaining exact question was whether a larger lane geometry might fit the ANE RMSNorm+classifier head better and reduce the current logits/selection budget.

Measured results:
- `32`: `2.3151614583333333 ms/token`, `431.9358005780449 tok/s`, trunk `1.1927395833333332`, logits `1.0936614583333333`
- `64`: `2.369515625 ms/token`, `422.0303773646739 tok/s`, trunk `1.1407473958333334`, logits `1.2184114583333332`
- `96`: `2.4335859375 ms/token`, `410.91627704100335 tok/s`, trunk `1.1276119791666668`, logits `1.3105260416666669`
- `128`: `2.3958515625000003 ms/token`, `417.3982080300975 tok/s`, trunk `1.1098619791666666`, logits `1.3032057291666665`
- `16`, `8`, and `1` remain unsupported at eval with `statusType=0x9`.

Interpretation:
- Larger head lanes reduce trunk time slightly but increase logits time more, so end-to-end throughput regresses.
- Inference: the current fused ANE RMSNorm+classifier head is already near its best lane geometry at `32` for this branch.
- Further lane-width sweeps are low-value unless the head architecture changes materially.

## 2026-03-08 - Larger fused-triplet trunk lanes regress

What was tried:
- Added a fused-triplet direct-select recurrent trunk lane sweep to probe larger lane widths: `64`, `96`, and `128`, while also rechecking the known-small `16`, `8`, and `1` cases.
- Kept the output head fixed on the existing fused ANE RMSNorm+classifier configuration.

Why:
- Smaller recurrent trunk lanes were already known to fail, but larger recurrent lane geometries were still unmeasured on the best fused-triplet backend.

Measured results:
- `32`: `2.231127604166667 ms/token`, `448.2061019920989 tok/s`, trunk `1.1512213541666667`, logits `1.0835572916666667`
- `64`: `2.3024348958333336 ms/token`, `434.3348113545743 tok/s`, trunk `1.2098203125000002`, logits `1.114390625`
- `96`: `2.459033854166667 ms/token`, `406.66380257293986 tok/s`, trunk `1.3670260416666664`, logits `1.0531848958333332`
- `128`: `2.471630208333333 ms/token`, `404.6890353350651 tok/s`, trunk `1.4412317708333333`, logits `1.0442265625`
- `16`, `8`, and `1` remain unsupported at eval with `statusType=0x9`-class recurrent step failures.

Interpretation:
- Larger recurrent lanes make the fused-triplet trunk slower end-to-end, even when logits time holds roughly flat.
- Inference: `32` is also the best lane geometry for the current fused-triplet recurrent trunk on this branch.
- Combined with the larger output-head lane sweep, this largely closes the remaining low-cost lane-geometry space for the current exact architecture.

## 2026-03-08 - Metal argmax over ANE output-head surface regresses badly

What was tried:
- Added a `MetalFP16ArgmaxKernel` that binds the ANE output-head IOSurface as a shared Metal buffer and performs an exact FP16 argmax reduction on the GPU.
- Wired it behind an opt-in `ANE_DIRECT_SELECT_METAL_ARGMAX=1` path in the existing direct-select output head.
- Added synthetic parity tests for spatial-slice correctness and first-max tie behavior, plus a fused-triplet hardware comparison benchmark.

Why:
- The remaining exact selection question was whether the final host-side argmax reduction could be moved off the CPU without changing the ANE graph.

Measured results:
- Synthetic parity tests passed.
- Hardware parity matched token outputs on fused-triplet direct-select.
- Baseline direct-select: `2.2768203125 ms/token`, `439.20912074503667 tok/s`, compile `507.5502916666667 ms`, trunk `1.1505390625`, logits `1.1085625000000001`
- Metal argmax: `2.7085859374999997 ms/token`, `369.2032609704555 tok/s`, compile `514.7445833333334 ms`, trunk `1.0979609375`, logits `1.6178229166666667`

Interpretation:
- The GPU reduction preserved exactness but added too much per-token dispatch/synchronization overhead.
- The extra cost landed almost entirely in the logits/selection budget, which grew by about `0.509 ms/token`.
- Inference: a standalone Metal reduction is not a viable replacement for the current host argmax on this branch.

## 2026-03-08 - Packed-state fused-triplet compiles but fails at eval

What was tried:
- Kept the fused-triplet recurrent math unchanged but packed the three recurrent states into one channel-concatenated IOSurface.
- Implemented a packed-state fused-triplet generator using `slice_by_size` on the packed state input and `concat` on the three state outputs.
- Added a matching kernelset, session, harness backend, and hardware comparison test.

Why:
- The previous low-cost I/O idea was to reduce the triplet session's surface count and collapse three state copy/reset operations into one without increasing fusion depth.

Measured result:
- Unit contracts passed.
- The ANE compiler accepted the packed-state MIL.
- The first hardware eval failed immediately with `statusType=0x9` / `Program Inference error` before any steady-state runtime measurement.

Interpretation:
- This is a different failure class from the larger-fusion `InvalidMILProgram` wall: packing the recurrent state compiles, but the runtime/evaluator rejects the program at execution.
- Inference: the current packed-state recurrent surface topology is not viable on this branch as implemented, even though the underlying MIL is syntactically acceptable.
- The probe should be reverted and treated as blocked unless a materially different packing strategy appears.

## 2026-03-08 - Matched concurrent serving crosses 6x through 4 streams

What changed:
- Extended the matched concurrent ANE/CoreML serving benchmark from `1/2/3/4` streams to `1/2/3/4/5/6` streams.
- Repeated the same hardware benchmark twice to check whether the `6x` aggregate ratio was stable.

Why:
- After the single-stream exact architecture space mostly collapsed into compiler/runtime walls or regressions, matched serving throughput was the remaining path with a realistic shot at a branch-visible `6x` result.
- The earlier `4`-stream run was already near the threshold.

Repeated matched results:
- Run 1:
  - `1` stream: ANE `437.04 tok/s`, CoreML `65.79 tok/s`, ratio `6.64x`
  - `2` streams: ANE `853.12 tok/s`, CoreML `121.88 tok/s`, ratio `7.00x`
  - `3` streams: ANE `1146.17 tok/s`, CoreML `181.29 tok/s`, ratio `6.32x`
  - `4` streams: ANE `1245.14 tok/s`, CoreML `215.92 tok/s`, ratio `5.77x`
  - `5` streams: ANE `1277.27 tok/s`, CoreML `232.71 tok/s`, ratio `5.49x`
  - `6` streams: ANE `1235.92 tok/s`, CoreML `244.80 tok/s`, ratio `5.05x`
- Run 2:
  - `1` stream: ANE `442.53 tok/s`, CoreML `65.69 tok/s`, ratio `6.74x`
  - `2` streams: ANE `827.85 tok/s`, CoreML `122.17 tok/s`, ratio `6.78x`
  - `3` streams: ANE `1181.93 tok/s`, CoreML `181.08 tok/s`, ratio `6.53x`
  - `4` streams: ANE `1299.59 tok/s`, CoreML `215.78 tok/s`, ratio `6.02x`
  - `5` streams: ANE `1247.67 tok/s`, CoreML `232.41 tok/s`, ratio `5.37x`
  - `6` streams: ANE `1206.56 tok/s`, CoreML `245.53 tok/s`, ratio `4.91x`

Interpretation:
- This is the first repeated, matched benchmark configuration on the branch that clears `6x` over CoreML.
- The serving knee is around `3-4` streams:
  - aggregate ANE throughput improves through `4` streams
  - per-stream throughput degrades steadily after `2` streams
  - the ANE/CoreML ratio drops below `6x` at `5+` streams
- This is a serving-throughput breakthrough, not a single-stream decode breakthrough.
- Caveat: the concurrent CoreML `1`-stream result remains much slower than the standing single-stream CoreML control, so any external claim should explicitly say this is the matched concurrent serving harness.
