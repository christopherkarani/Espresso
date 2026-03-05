# Research Task 5: IOSurface I/O Elimination Strategies

**Date**: 2026-03-05
**Branch**: checkpoint/phase8-ane-inference-20260305
**Scope**: Areas 2a (Zero-Copy Surface Chaining), 2b (FP16-Native Path), 2c (Double-Buffered I/O)

---

## 1. Current I/O Pattern -- Exactly What Happens Per Layer

### Inference Forward Pass (ForwardPass.runInferenceTimed)

Each of the 6 layers executes two ANE kernels (attention + FFN) with fused residual additions. The per-layer I/O flow is:

```
Step 1: writeFP16(attnIn, xCur)        -- FP32->FP16 convert + IOSurfaceLock/Write/Unlock
Step 2: eval(fwdAttn)                   -- ANE executes SDPA + residual
Step 3: [handoff: attnOut -> ffnIn]     -- mode-dependent (see below)
Step 4: eval(fwdFFN)                    -- ANE executes FFN + residual
Step 5: readFP16(ffnOut, xCur)          -- IOSurfaceLock/Read/Unlock + FP16->FP32 convert
```

### Handoff Modes (Step 3)

**`.cpuRoundTrip` (default)**:
```
3a: readFP16(attnOut -> xCur)           -- Lock, FP16->FP32 NEON convert, Unlock
3b: writeFP16(xCur -> ffnIn)            -- Lock, FP32->FP16 NEON convert, Unlock
```
This requires 2 lock/unlock cycles + 2 NEON conversion passes (FP16->FP32 then FP32->FP16).

**`.fp16SurfaceCopy`**:
```
3: copyFP16(attnOut -> ffnIn)           -- Lock both, memcpy FP16 data, Unlock both
```
This requires 2 lock/unlock cycles but only 1 memcpy (no FP16<->FP32 conversion).

### Data Sizes (ModelConfig: dim=768, seqLen=256)

| Surface | Size (bytes) | Size (FP16 elements) |
|---|---|---|
| attnIn | 768 * 256 * 2 = 393,216 | 196,608 |
| attnOut | 768 * 256 * 2 = 393,216 | 196,608 |
| ffnIn | 768 * 256 * 2 = 393,216 | 196,608 |
| ffnOut | 768 * 256 * 2 = 393,216 | 196,608 |

Total per layer: 4 surfaces, 1.5 MiB.
Total 6 layers: 24 surfaces, 9.4 MiB.

### Surface Lifecycle

Each kernel (`ANEKernel`) owns its IOSurfaces, allocated at compile time in `ane_interop_compile()`:
- Input surfaces: allocated via `ane_interop_create_surface(inputBytes)`
- Output surfaces: allocated via `ane_interop_create_surface(outputBytes)`
- Wrapped as `_ANEIOSurfaceObject` and passed to `_ANERequest` at compile time
- The `_ANERequest` is created once and reused for all `eval()` calls
- **Critical**: The request binds specific IOSurface objects permanently. The ANE reads from/writes to these exact surfaces on every eval.

### Measured Overhead

From benchmark data (6 layers, median):
- Total Surface I/O: 0.342 ms (6.0% of total)
- Per-layer Surface I/O: ~0.057 ms
- Per single read or write: ~0.014 ms (including lock + convert + unlock)
- ANE kernel time: 5.343 ms (94.0%)

---

## 2. Zero-Copy Surface Chaining Analysis (Area 2a)

### Can attnOut become ffnIn without any copy?

**No, not with the current ANE request binding model.** Here is why:

1. **IOSurfaces are bound at request creation time.** In `ane_interop.m:305-336`, the `_ANERequest` is constructed with specific `_ANEIOSurfaceObject` wrappers for inputs and outputs. These are fixed for the lifetime of the request.

2. **Each kernel has its own independent request.** `fwdAttn` has `request_A` with `{attnIn, attnOut}`, and `fwdFFN` has `request_B` with `{ffnIn, ffnOut}`. There is no mechanism to tell `request_B` to read from `attnOut` instead of `ffnIn`.

3. **No surface rebinding API exists.** There is no discovered method like `setInput:atIndex:` on `_ANERequest` or `_ANEInMemoryModel` that would allow changing the IOSurface after request creation.

### Could we compile fwdFFN with attnOut as its input surface?

**Theoretically yes, but impractical.** You would need to:
- Get `fwdAttn`'s output IOSurfaceRef after compiling attention
- Pass it as the input surface when constructing `fwdFFN`'s `_ANERequest`

This would require modifying `ane_interop_compile()` to accept pre-existing IOSurfaces instead of always creating new ones. The API modifications needed:

```c
// Hypothetical: compile with external surfaces
ANEHandle *ane_interop_compile_with_surfaces(
    const uint8_t *milText, size_t milLen,
    /* weights... */
    int nInputs, IOSurfaceRef *externalInputs,   // use these instead of creating new
    int nOutputs, IOSurfaceRef *externalOutputs
);
```

**Feasibility assessment: MEDIUM.** The `_ANEIOSurfaceObject` wrapper accepts any IOSurfaceRef via `objectWithIOSurface:`. So we could pass `attnOut` as the input surface for FFN's request. This would eliminate the inter-kernel copy entirely.

**However**, there are critical constraints:
- The surfaces must have matching allocation sizes (attnOut must be >= ffnIn bytes). In inference mode, both are `dim * seqLen * 2 = 393,216 bytes` -- they match.
- We must compile attention first, then FFN (serial compilation dependency).
- The output surface of attention must not be reused by any other kernel simultaneously.

### How would kernel fusion change this?

If SDPA+FFN are fused into one MIL program, intermediate tensors (the x2 result) never leave the ANE at all. They stay in the ANE's internal SRAM/registers. This makes surface chaining **moot** for intra-layer data flow. The only surfaces needed would be the fused kernel's input and output.

---

## 3. Lock Overhead Analysis (Area 2b)

### What IOSurfaceLock/Unlock actually does

IOSurfaceLock is a kernel-level operation that:
1. Ensures cache coherency between CPU and device (ANE) access
2. Provides mutual exclusion for concurrent surface access
3. May trigger a CPU cache flush/invalidate

### Measured lock overhead

From the per-operation timing:
- A full write cycle (lock + NEON FP32->FP16 convert 196K elements + unlock): ~0.014 ms
- A full read cycle (lock + NEON FP16->FP32 convert 196K elements + unlock): ~0.014 ms
- The NEON conversion at 196K elements (8 elements per NEON iteration): ~24K iterations

Estimated lock/unlock overhead per call: **< 0.002 ms** (sub-2-microsecond). The conversion dominates, not the lock.

### Can we skip locks between sequential evals?

**Unsafe in the general case, but potentially safe for our specific pattern.**

The ANE `eval()` is synchronous -- it blocks until completion. After `eval()` returns:
- The ANE has finished writing to the output surface
- No other process/thread is accessing the surface
- The CPU cache may be stale for the output surface data

**The real issue is cache coherency, not mutual exclusion.** After ANE writes to an output surface, the CPU needs to see the updated data. `IOSurfaceLock` likely triggers a cache invalidate that ensures this.

**Skipping locks is risky** because:
1. The ANE may use DMA that doesn't go through CPU cache hierarchy
2. Without the lock, the CPU might read stale cached data
3. There is no public documentation on ANE cache coherency guarantees

### Unlocked I/O already exists

The codebase already has `ane_interop_io_write_fp16_unlocked()` and `ane_interop_io_read_fp16_unlocked()` plus Swift wrappers `SurfaceIO.writeFP16Unlocked()` / `SurfaceIO.readFP16Unlocked()`. These skip the lock/unlock calls but still do the FP16<->FP32 conversion.

### Amortized locking

The existing `ane_interop_io_read_fp16_batched()` already amortizes locking -- it locks once, reads multiple regions, then unlocks. This is optimal for the training path where multiple channel slices are read from one surface.

For inference, each surface has only one region (dim channels at offset 0), so batching provides no benefit.

### Recommendation

Lock overhead is negligible (<4 microseconds per cycle). **Do not optimize locks.** The conversion overhead (NEON FP16<->FP32) is the real cost, and the only way to eliminate it is to eliminate the conversion entirely (by staying in FP16 or by not doing the transfer at all).

---

## 4. Double-Buffering Analysis (Area 2c)

### Concept

Use two sets of surfaces and alternate between them:
- While ANE writes results to Set A, CPU reads previous results from Set B
- While CPU writes next input to Set A, ANE processes data from Set B

### Applicability to Espresso

**Double-buffering provides no benefit here** because:

1. **ANE eval() is synchronous.** We cannot overlap ANE computation with CPU I/O. The `evaluateWithQoS:options:request:error:` call blocks until the ANE finishes. There is no async eval API (confirmed in Task 4 research).

2. **The ANE is a single execution unit.** We cannot run two kernels simultaneously. One kernel must finish before the next starts.

3. **CPU I/O is fast relative to ANE compute.** Surface I/O is 6% of total time. Even if we could perfectly overlap it, the maximum speedup is 6%.

4. **No `_ANEChainingRequest` pipelining is available.** Without being able to queue multiple eval requests, there is nothing to overlap with.

### Could double-buffering help with GCD-based overlap?

In theory, while layer N's FFN eval runs on ANE, a GCD queue could prepare layer (N+1)'s attention input surface. This requires:
- Pre-allocated surfaces for layer N+1 (already done via `InferenceSurfaceHandles`)
- Writing xCur to layer (N+1)'s attnIn while FFN eval is in progress

**Problem**: We don't have xCur for layer N+1 until FFN eval completes (xCur = FFN output). So we cannot prepare the next layer's input while the current layer is executing.

**Exception**: If we used surface chaining (Section 2), where FFN output flows directly to the next layer's attention input, then we could potentially queue the next eval immediately. But this requires the surface binding approach described above.

### Surface count reduction

Current: 4 surfaces per layer * 6 layers = 24 surfaces.

With surface chaining (attnOut == ffnIn): 3 surfaces per layer = 18 surfaces. Saves 6 surfaces (2.4 MiB).

With cross-layer chaining (ffnOut_L == attnIn_L+1): only 2 unique surfaces per layer boundary. With careful aliasing, the entire 6-layer pass could use as few as 3 surfaces (ping-pong input/output + one intermediate).

---

## 5. Interaction with Kernel Fusion

### Intra-Layer Fusion (SDPA + FFN -> Single Kernel)

If attention and FFN are fused into one MIL program per layer:

**I/O eliminated:**
- The inter-kernel handoff (Step 3) disappears entirely. No attnOut->ffnIn copy.
- Per-layer I/O reduces to: 1 write (xCur -> fusedIn) + 1 read (fusedOut -> xCur)
- From 3 I/O operations per layer to 2 (or 1 with surface chaining)

**Surface count:**
- Per layer: 2 surfaces (fusedIn, fusedOut) instead of 4
- Total: 12 surfaces instead of 24

**Estimated I/O savings:**
- Current I/O per layer: ~0.057 ms (3 operations: write + handoff + read)
- With intra-layer fusion: ~0.028 ms (2 operations: write + read)
- With fusion + cross-layer chaining: ~0.014 ms (1 operation: read only, if fusedOut == next attnIn)
- Savings: 0.029 ms per layer = 0.174 ms for 6 layers (51% I/O reduction)

### Inter-Layer Fusion (Multiple Layers -> Single Kernel)

If multiple layers are fused (e.g., layers 0-1 as one kernel):

**Additional I/O eliminated:**
- The cross-layer transfer (ffnOut_L -> attnIn_L+1) disappears
- Only the first layer's input and last layer's output touch IOSurfaces

**With full 6-layer fusion (one giant kernel):**
- Total I/O: 1 write + 1 read = 2 operations total
- ~0.028 ms total I/O for the entire model
- Savings: 0.314 ms (92% I/O reduction)
- But: likely impractical due to ANE program size limits, compilation time, and inability to interleave CPU operations

### Full-Model Fusion: Surface Count

| Fusion Strategy | Surfaces Needed | I/O Operations | Est. I/O Time |
|---|---|---|---|
| No fusion (current) | 24 (4/layer) | 18 (3/layer) | 0.342 ms |
| Intra-layer fusion (SDPA+FFN) | 12 (2/layer) | 12 (2/layer) | 0.228 ms |
| Intra-layer + cross-layer chaining | 3 (ping-pong) | 7 (write + 6 reads) | 0.133 ms |
| Full 6-layer fusion | 2 (in + out) | 2 (write + read) | 0.028 ms |

---

## 6. Minimum Surface Count Analysis

### Current Architecture (No Fusion)

**Per layer**: 4 surfaces (attnIn, attnOut, ffnIn, ffnOut)
**Total**: 24 surfaces for 6 layers
**Memory**: 24 * 393,216 = 9.4 MiB

### With Surface Aliasing (No Fusion)

Since layers execute sequentially and never overlap:
- Layer L's surfaces are free after Layer L completes
- We could reuse Layer 0's surfaces for Layer 1, etc.

**Minimum with aliasing**: 4 surfaces total (reused across all layers)
**Requires**: Modifying `_ANERequest` to use the same surfaces, OR using the surface-chaining compilation approach from Section 2.

**Problem**: `_ANERequest` is bound at compile time. Each kernel's request references specific surfaces. Reusing surfaces across kernels requires either:
1. Compiling all kernels with shared surfaces (serial compilation dependency)
2. Rebuilding `_ANERequest` objects per-inference (unknown if supported)

### With Intra-Layer Fusion

**Per layer**: 2 surfaces (fusedIn, fusedOut)
**Minimum with aliasing**: 2 surfaces (ping-pong)
**Memory**: 2 * 393,216 = 768 KiB

### With Full Model Fusion

**Total**: 2 surfaces (modelIn, modelOut)
**Memory**: 2 * 393,216 = 768 KiB

---

## 7. Concrete Recommendations (Prioritized)

### Tier 0: Kernel Fusion Makes Most I/O Optimization Moot

**The single highest-impact optimization is intra-layer kernel fusion (SDPA+FFN).** This eliminates:
- The inter-kernel handoff entirely (0.014-0.028 ms per layer)
- Half the surface allocations
- All FP16<->FP32 conversion overhead for intermediate results

Surface I/O is 6% of total inference time. Even eliminating ALL surface I/O (theoretical maximum) saves only 0.342 ms out of 4.595 ms (7.4% speedup). **The 10x goal will not be achieved through I/O optimization alone.** Kernel fusion addresses the 94% ANE time, which is where the real gains are.

### Tier 1: High Impact, Moderate Effort

| # | Recommendation | Est. Impact | Effort | Risk |
|---|---|---|---|---|
| 1 | **Intra-layer kernel fusion** (Task 6 scope) | Eliminates 33% of I/O + potential ANE speedup from reduced kernel dispatch overhead | 2-3 days | MEDIUM |
| 2 | **Surface binding at compile time** -- modify `ane_interop_compile` to accept external IOSurfaces, bind attnOut as ffnIn | Eliminates inter-kernel copy (0.17 ms for 6 layers) | 1 day | LOW |
| 3 | **Cross-layer surface chaining** -- bind layer N ffnOut as layer N+1 attnIn | Eliminates inter-layer write (0.08 ms for 6 layers) | 0.5 days (after #2) | LOW |

### Tier 2: Moderate Impact, Low Effort

| # | Recommendation | Est. Impact | Effort | Risk |
|---|---|---|---|---|
| 4 | **Enable `.fp16SurfaceCopy` as default** for inference | Saves conversion overhead (~0.05 ms for 6 layers vs cpuRoundTrip) | Trivial | NONE |
| 5 | **FP16-native hidden state** -- keep xCur in FP16 throughout, only convert at model boundaries | Eliminates 10 of 12 FP16<->FP32 conversions | 0.5 days | LOW |
| 6 | **Surface aliasing** -- compile all 12 inference kernels with only 2 ping-pong surfaces | Saves 10 surface allocations (3.8 MiB memory, negligible latency) | 1 day | MEDIUM |

### Tier 3: Low Impact, Speculative

| # | Recommendation | Est. Impact | Effort | Risk |
|---|---|---|---|---|
| 7 | **Lock elimination** for sequential eval pattern | < 0.024 ms total savings | Trivial | MEDIUM (cache coherency risk) |
| 8 | **Double buffering** | No benefit without async eval | N/A | N/A |
| 9 | **`_ANEChainingRequest` exploration** | Unknown; could enable true zero-copy chaining | 1-2 days | HIGH |

### Summary: Maximum I/O Optimization Impact

| Strategy | I/O Time | Savings vs Current | % of Total |
|---|---|---|---|
| Current (cpuRoundTrip) | 0.342 ms | baseline | 7.4% |
| fp16SurfaceCopy (already implemented) | ~0.290 ms | 0.052 ms | 6.3% |
| Surface binding (attnOut=ffnIn) | ~0.170 ms | 0.172 ms | 3.7% |
| + Cross-layer chaining | ~0.100 ms | 0.242 ms | 2.2% |
| + FP16-native hidden state | ~0.050 ms | 0.292 ms | 1.1% |
| Intra-layer fusion (1 kernel/layer) | ~0.170 ms | 0.172 ms | 3.7% |
| Fusion + chaining | ~0.028 ms | 0.314 ms | 0.6% |

### Key Insight

**Surface I/O optimization has diminishing returns.** The maximum possible gain from eliminating ALL I/O is 0.342 ms (7.4%). The practical gain from implementable optimizations is 0.17-0.29 ms (3.7-6.3%).

**The path to 10x is through ANE kernel time reduction** (kernel fusion reducing dispatch overhead, MIL optimization reducing per-kernel execution time, tile alignment). Surface I/O is a secondary concern that will naturally improve as kernel fusion reduces the number of CPU<->ANE transfers.

---

## 8. Implementation Roadmap

### Phase A: Quick Wins (Day 1)

1. Switch inference default to `.fp16SurfaceCopy`
2. Benchmark to confirm improvement

### Phase B: Surface Binding (Day 2-3, if not pursuing fusion)

1. Add `ane_interop_compile_with_surfaces()` variant to `ane_interop.m`
2. Modify `InferenceKernelSet` to compile FFN with attention's output surface as input
3. Modify `ForwardPass.runInferenceTimed` to skip handoff step entirely
4. Benchmark

### Phase C: Kernel Fusion (Day 2-5, preferred path)

1. Implement fused SDPA+FFN MIL generator (Task 6)
2. This automatically eliminates inter-kernel I/O
3. Implement cross-layer surface chaining if beneficial
4. Benchmark

### Phase D: FP16-Native Path (Day 4-6, after fusion)

1. Store xCur as FP16 buffer between layers
2. Only convert FP32<->FP16 at model input/output boundaries
3. Requires FP16 TensorBuffer variant or raw buffer management
4. Benchmark

---

## Sources

- Codebase analysis: `Sources/ANEInterop/ane_interop.m` (surface lifecycle, request binding)
- Codebase analysis: `Sources/ANEInterop/surface_io.c` (lock patterns, copy operations)
- Codebase analysis: `Sources/Espresso/ForwardPass.swift` (inference I/O flow)
- Codebase analysis: `Sources/ANERuntime/ANEKernel.swift` (surface ownership)
- Research Task 4 findings: No async eval, no surface rebinding API, `_ANEChainingRequest` unexplored
- IOSurface framework documentation: Lock/Unlock semantics, cache coherency
- Benchmark data: 6-layer inference median 4.595 ms, I/O 0.342 ms (6.0%)
