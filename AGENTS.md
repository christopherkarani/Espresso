# ANE 4x Decode Throughput — Sequential Avenue Exploration

You are continuing a multi-session research and implementation effort to achieve **4x decode inference throughput over CoreML** on the Apple Neural Engine, using reverse-engineered private APIs in Swift 6.2.

## Your Mission

Systematically tackle each optimization avenue listed below **in order**. For each avenue:

1. **Read the prior findings** before writing any code — consult `docs/fused-decode-and-next-steps.md`, `docs/vc-probe-results.md`, and the project MEMORY.md
2. **Build a benchmark** that isolates the avenue's impact (TDD: test first, implementation second)
3. **Measure baseline** before your changes, measure after, report the delta
4. **Abandon immediately** if the avenue hits an entitlement wall, silent nil return, `statusType=0x9`, or any crash that can't be recovered — do NOT spend more than ~30 minutes diagnosing a dead end. Document what failed and move on.
5. **Commit** each avenue's work atomically (one commit per avenue, or per meaningful sub-step)
6. **Update docs** — append your findings to `docs/fused-decode-and-next-steps.md` with measured numbers
7. **Update MEMORY.md** if you discover new ANE facts, gotchas, or confirmed patterns

## Current State (Where You're Picking Up)

**Branch:** `feat/vc-eval-probe`
**Working directory:** `/Users/chriskarani/CodingProjects/Espresso/.worktrees/vc-probe`
**Build:** `swift build` passes cleanly
**Tests:** All non-hardware tests pass. Hardware tests require `ANE_HARDWARE_TESTS=1`.

### What's Already Done

| Avenue | Status | Result |
|--------|--------|--------|
| Single-layer fused decode kernel (attn+FFN) | **DONE** | 0.095ms/step saved per layer (12.5%), 12→6 dispatches |
| VirtualClient eval path | **BLOCKED** | Kernel IOKit entitlement gate, all 5 init paths return nil |
| `_ANEChainingRequest` | **BLOCKED** | `PREPARE_FAILED` across all parameter permutations |
| IOSurface aliasing | **BLOCKED** | `statusType=0x9` at eval time |
| `evaluateRealTimeWithModel:` | **BLOCKED** | All 5 selectors exist. `loadRealTimeModel:` returns NO with no error. Silent entitlement gate. |
| CompletionHandler on standard eval | **WORKS** | Fires synchronously on calling thread after eval. Not true async. |
| QoS tuning (0-63) | **BLOCKED** | No effect on latency |

### Current Decode Budget (6-layer transformer, M3 Max)

```
Component                     Fused (current)
----------------------------------------------
ANE compute (hw time)         6 x 0.405ms = 2.430ms
IO (cache update + chain)     6 x 0.024ms = 0.144ms
Total per token               ~2.574ms
```

The remaining wall is **ANE compute time** (hardware silicon speed). Software-addressable dispatch overhead has been eliminated via single-layer fusion.

## Avenues to Explore (In Order)

### Avenue 1: Multi-Layer Kernel Fusion (2-Layer Packed KV Caches)

**Priority: HIGHEST. Confidence: HIGH.**

Fuse 2 transformer layers into a single MIL program. Pack both layers' KV caches into one input IOSurface using `slice_by_size` on the channel axis.

**What to build:**
- `FusedTwoLayerDecodeGenerator` in `Sources/MILGenerator/` — extends `FusedDecodeLayerGenerator` pattern to emit two layers sequentially in one MIL program. Use `slice_by_size` to extract per-layer K/V/mask from a channel-packed input.
- Cache packing layout: `[1, 4608, 1, maxSeq]` where channels 0-767=L0 kCache, 768-1535=L0 vCache, 1536-2303=L0 maskCache, 2304-3071=L1 kCache, 3072-3839=L1 vCache, 3840-4607=L1 maskCache
- 18 weight blobs (9 per layer), 2 inputs (x + packed_cache), 3 outputs (xNext + kfFull_L1 + vfFull_L1) — or more outputs if both layers' K/V need writeback
- `FusedTwoLayerDecodeKernelSet` in `Sources/ANERuntime/`
- Tests proving: MIL generates correctly, compiles on hardware, eval succeeds, timing comparison vs 2x single-layer dispatch

**Known patterns to follow:**
- `SDPABackward1Generator`, `QKVBackwardGenerator` already do 3-4 `slice_by_size` ops from packed inputs with convolutions. This is proven to compile and eval.
- `slice_by_index` on function inputs fails for complex programs (MEMORY.md gotcha). Use `slice_by_size` instead — it works.
- SSA name collision: prefix Layer 1 variables with `l1_` (Layer 0 keeps existing names from `FusedDecodeLayerGenerator`)

**Expected gain:** ~0.285ms/token (3 fewer dispatches at 0.095ms each) + inter-layer copy elimination.

**Abandon criteria:** If `slice_by_size` on the packed cache input produces `InvalidMILProgram` or `statusType=0x9`, try reducing to just K+V packing (no mask). If that also fails, abandon and document.

**If successful:** Try 3-layer fusion (42.85 MB baked weights, exceeds 32 MB SRAM but ANE streams via DMA). Document whether weight streaming degrades throughput.

### Avenue 2: Metal SharedEvent on Standard Eval Path

**Priority: MEDIUM. Confidence: MEDIUM.**

Construct an `IOSurfaceSharedEvent` via Metal (`[MTLDevice newSharedEvent]`), wrap it in `_ANESharedWaitEvent` / `_ANESharedEvents`, and attach to `_ANERequest` via `setSharedEvents:`. Test whether the ANE hardware signals the event after eval completes.

**What to build:**
- New probe function `ane_interop_probe_metal_shared_event()` in `ane_interop.m`
- Import Metal framework via `dlopen` (same pattern as ANE framework loading)
- Create `MTLSharedEvent` → cast to `IOSurfaceSharedEvent` → build `_ANESharedWaitEvent` via factory → build `_ANESharedEvents` container → attach to request
- Eval and check if shared event value increments
- If it works: measure whether we can dispatch eval N+1 while event from eval N hasn't fired yet (true pipeline parallelism)

**Key reference:** `docs/vc-probe-results.md` Section 5 documents that `IOSurfaceSharedEvent +new` returns nil, but Metal factory should work. Section 8 confirms `setSharedEvents:` exists on `_ANERequest`.

**Abandon criteria:** If `setSharedEvents:` crashes, or eval with shared events attached returns `0x9`, or event value never changes after eval, abandon and document.

### Avenue 3: Metal + ANE Hybrid Decode

**Priority: HIGH IMPACT. Confidence: MEDIUM. Effort: LARGE.**

Split the decode pipeline: ANE handles convolution-heavy work (QKV projections, FFN), Metal handles attention math (Q@K^T, softmax, @V). IOSurfaces enable zero-copy data sharing.

**What to build:**
- Metal compute shader for scaled dot-product attention: Q@K^T / sqrt(d) + mask → softmax → @V
- Metal kernel reads Q, K, V from IOSurface (FP16, `[1, dim, 1, spatial]` layout)
- Metal kernel writes attention output to IOSurface
- ANE QKV projection kernel (reuse existing `DecodeAttentionQKVGenerator` prefix, outputting Q/K/V)
- ANE FFN kernel (reuse existing `DecodeFFNGenerator`)
- Pipeline: ANE eval (QKV) → Metal attention → ANE eval (FFN), with potential overlap

**Architecture:**
```
ANE:   [QKV proj] → IOSurface(Q,K,V) → [FFN]
                         |                  ^
                         v                  |
Metal:             [Q@K^T → softmax → @V] → IOSurface(attn_out)
```

**Key advantage:** Two accelerators can work in parallel. While ANE runs layer N's FFN, Metal can run layer N+1's attention. CoreML cannot do this.

**Implementation approach:**
1. First: standalone Metal attention kernel that reads/writes IOSurface (prove correctness)
2. Second: wire into decode loop replacing the ANE attention probe path
3. Third: measure latency — if Metal attention is faster than ANE attention, we gain even without parallelism
4. Fourth: attempt overlap (dispatch Metal async while ANE evals)

**Abandon criteria:** If IOSurface FP16 data is not directly accessible from Metal compute shaders without copies, or if Metal attention latency exceeds ANE attention latency by >2x, abandon the parallel approach (still useful for correctness if Metal attention works).

### Avenue 4: CoreML Baseline Benchmark

**Priority: CRITICAL FOR CLAIMS. Effort: MEDIUM.**

We have **zero CoreML comparison data**. All throughput numbers are for the direct `_ANEClient` path. To validate "4x over CoreML" we need a head-to-head benchmark.

**What to build:**
- Export the 6-layer transformer as a CoreML `.mlpackage` via coremltools (Python script)
- Load in Swift via `MLModel` with `.cpuAndNeuralEngine` compute units
- Measure per-token decode latency on the same M3 Max hardware
- Compare: CoreML decode latency vs our direct path decode latency
- Run 100+ iterations, report median and p99

**Key insight from docs:** CoreML uses the same ANE hardware and MIL compiler internally. Our advantage is fine-grained kernel boundaries, zero-copy IOSurface KV cache management, and hybrid dispatch capability. CoreML may have better internal graph fusion for some workloads.

**Abandon criteria:** None — this is measurement, not implementation. Even if CoreML is faster, we need to know by how much.

### Avenue 5: Speculative Decoding

**Priority: HIGH IMPACT. Effort: LARGE. Risk: LOW.**

Use a tiny draft model (1-2 layers) to generate N candidate tokens, then verify all N in one batched forward pass on the full model.

**What to build:**
- Draft model: 1 or 2 layer transformer with same dim/vocab (compile separate kernels)
- Draft decode: run N steps (e.g., N=4) generating candidate tokens
- Verification: batch the N candidates into a single prefill-like forward pass on the full 6-layer model
- Accept/reject logic: compare draft logits vs full model logits, accept matching prefix
- Measure: effective tokens/second including draft overhead + verification

**Why ANE is suited:** The prefill path (batch tokens) achieves much higher ANE utilization than single-token decode. Speculative decoding converts decode into prefill-like workload.

**Abandon criteria:** If the draft model accuracy is too low (accept rate < 50%), the overhead doesn't amortize. Try N=2, N=4, N=8 and measure accept rates.

### Avenue 6: GCD Pipeline with CompletionHandler

**Priority: LOW. Expected gain: ~5-30µs per layer.**

Dispatch eval on a background GCD queue. While eval blocks that queue, the main thread prepares K/V/mask cache updates for the next layer. The CompletionHandler signals when eval completes.

**What to build:**
- `runFusedDecodePipelined()` variant in `DecodeForwardPass.swift`
- Background dispatch queue for eval
- Main thread does cache IO while eval runs
- CompletionHandler or `dispatch_semaphore` for synchronization
- Benchmark: pipelined vs sequential decode loop

**Abandon criteria:** If measured savings < 10µs per token (below noise floor), abandon.

## Benchmarking Protocol

For every avenue, follow this protocol:

1. **Baseline measurement:** Run `ANE_HARDWARE_TESTS=1 swift test --filter <relevant_test>` and record timing BEFORE your changes
2. **Implementation:** Build the avenue
3. **Post measurement:** Same test, record timing AFTER
4. **Delta:** Report `baseline_ms - post_ms = savings_ms (X%)`
5. **Extrapolation:** Project savings over full 6-layer decode
6. **Cumulative:** Track running total of all savings across avenues

**Timing methodology:**
- Use `mach_absolute_time()` with `mach_timebase_info` for wall-clock
- Run 20+ iterations minimum, report median (not mean — avoids outlier skew)
- Include 3-5 warmup iterations before timed runs
- Report both per-eval and per-token (6 layers) numbers

## Files to Read First

Before starting ANY implementation, read these files to understand the existing architecture:

| File | Why |
|------|-----|
| `docs/fused-decode-and-next-steps.md` | Complete roadmap and budget breakdown |
| `docs/vc-probe-results.md` | All VirtualClient/SharedEvents findings |
| `Sources/MILGenerator/FusedDecodeLayerGenerator.swift` | Pattern for fused MIL generation |
| `Sources/ANERuntime/FusedDecodeKernelSet.swift` | Pattern for ~Copyable kernel wrappers |
| `Sources/Espresso/DecodeForwardPass.swift` | Decode loop + surface handles |
| `Tests/ANERuntimeTests/FusedDecodeKernelTests.swift` | Test patterns for decode kernels |
| `Sources/MILGenerator/MILBuilder.swift` | MIL text generation utilities |
| `Sources/ANEInterop/ane_interop.m` | C interop layer (real-time probe, chaining probe patterns) |
| `Sources/ANEInterop/include/ane_interop.h` | All C API types and functions |

## Proven Dead Ends (Do NOT Retry)

| Approach | Failure Mode |
|----------|-------------|
| `_ANEVirtualClient` instantiation (all 5 paths) | Kernel IOKit entitlement gate → nil |
| `_ANEChainingRequest` | `PREPARE_FAILED` across all permutations |
| IOSurface aliasing between kernels | `statusType=0x9` at eval |
| `_ANERequest` surface rebinding | Type confusion crash → `0x9` |
| `evaluateRealTimeWithModel:` | `loadRealTimeModel:` returns NO silently |
| `setCodeSigningIdentity:` | Crashes (internal dictionary mutation) |
| QoS tuning (0-63) | No effect on latency |
| `IOSurfaceSharedEvent +new` | Returns nil (must use Metal factory) |

## Code Conventions

- **Swift 6.2 strict concurrency** with `.swiftLanguageMode(.v6)`
- **`~Copyable` value types** for kernel wrappers and tensor buffers
- **Typed throws** — `throws(ANEError)` not generic `Error`
- **Channel-first layout** — `[1, C, 1, S]` for all tensors
- **TDD approach** — write tests first, then implementation
- **Commit frequently** with descriptive messages
- **No external dependencies** — only Apple system frameworks
- MIL program version: `program(1.3)`, function dialect: `func main<ios18>(...)`
- Weight blobs via `WeightBlob.build(from:rows:cols:)`
- Hardware-gated tests check `ANE_HARDWARE_TESTS=1` environment variable

## Success Criteria

The session is successful if:
1. Each avenue is explored, benchmarked, and documented (even if abandoned)
2. Cumulative measured gains are quantified
3. A CoreML baseline exists for comparison
4. `docs/fused-decode-and-next-steps.md` is updated with all findings
5. All tests pass (`swift test` for non-hardware, `ANE_HARDWARE_TESTS=1 swift test` for hardware)
6. A clear summary shows: "We are Xms/token on direct ANE vs Yms/token on CoreML = Z% faster"
