# ANE Training Codebase: Swift 6.2 Rewrite Plan

## Context

The ANE training codebase (~6,100 lines Obj-C/C) trains a 12-layer Llama2 transformer on Apple Neural Engine using reverse-engineered private APIs. It achieves 9.3 ms/step on M4 at 11.2% ANE utilization. The goal is a full Swift 6.2 rewrite that preserves numerical equivalence and performance, using strict concurrency, `~Copyable` types, typed throws, and idiomatic Swift — while keeping a thin C/ObjC shim for private API interop and NEON intrinsics.

---

## Executive Summary

Incremental 6-phase bottom-up rewrite. A thin C/ObjC bridging target (`ANEInterop`) encapsulates all private API access (~200 lines) permanently. Everything else becomes pure Swift across 5 targets: `ANETypes` (config + IOSurface I/O), `MILGenerator` (MIL text generation), `CPUOps` (vDSP/Accelerate operations), `ANERuntime` (kernel lifecycle), and `Espresso` + `EspressoTrain` (training loop executable). Each phase is TDD: write Swift tests first against the existing ObjC as reference, then implement until green. Numerical equivalence verified at every phase boundary (1e-4 fp32, 1e-2 fp16). Performance gate: no regression below 9.3 ms/step.

---

## Architecture Decision Records

### ADR-1: Package Structure — Multi-target SwiftPM

**Recommendation**: SwiftPM with 7 targets (1 C, 6 Swift).

**Rationale**: The dependency graph is already visible in the header includes. Separate targets give: (a) testability — `MILGenerator` and `CPUOps` are pure functions testable without ANE hardware, (b) incremental builds — MIL generators change rarely, training loop changes often, (c) C interop isolation — only `ANEInterop` imports `arm_neon.h` and `objc/message.h`.

```
ANEInterop (C/ObjC)
    ↓
ANETypes (Swift) ─────────────┐
    ↓            ↓             │
MILGenerator   CPUOps         │
    ↓            ↓             │
ANERuntime (depends on all 3) │
    ↓                          │
Espresso ──────────────────────┘
    ↓
EspressoTrain (executable)
```

### ADR-2: Private API Interop — Thin ObjC Bridging Module

**Recommendation**: C/ObjC target with clean C function API.

**Rationale**: The codebase uses 14 distinct `objc_msgSend` cast signatures (mix of `id`, `BOOL`, `NSUInteger`, `void` return types with varying parameter counts). Swift cannot express these safely — `unsafeBitCast` to `@convention(c)` for each variant is fragile and untestable. An ObjC shim wraps all `dlopen`, `NSClassFromString`, `objc_msgSend` casts, `CFBridgingRetain/Release`, and `@autoreleasepool` management behind a clean C header. Swift calls `ane_interop_compile()`, `ane_interop_eval()`, etc. When Apple changes private API signatures, you fix ONE `.m` file. Audit all 14 signatures against the C API surface to ensure full coverage.

### ADR-3: NEON SIMD — Accelerate vImageConvert + C Shim Fallback

**Recommendation**: Primary: Accelerate `vImageConvert_PlanarFtoPlanar16F` / `vImageConvert_Planar16FtoPlanarF`. Fallback: Keep existing NEON functions in C shim.

**Rationale**: Swift doesn't expose `arm_neon.h`. Swift SIMD types lack `Float16` vectors (`SIMD8<Float16>` doesn't exist). Accelerate provides Apple-tuned conversion using the same underlying NEON hardware, handling alignment/tail elements automatically. Keep C NEON shim for A/B benchmarking during the rewrite — these are only 16 lines in the hot I/O path (~288 calls/step).

### ADR-4: MIL Generation — Swift String Interpolation (Direct Port)

**Recommendation**: Swift string interpolation with `String(format:)` for precise float formatting.

**Rationale**: A result builder DSL would be elegant but the 6 generators total only ~286 lines of fairly regular string patterns. The risk of introducing a DSL is high — character-identical MIL output is required (even whitespace differences can cause ANE compile failures). Direct port with string interpolation minimizes divergence risk. Can upgrade to DSL in a future iteration once the Swift codebase is stable.

**CRITICAL — Locale-safe float formatting**: `String(format:)` in Swift uses the current locale by default. On European locales (e.g., `de_DE`), `String(format: "%.6f", 3.14)` produces `"3,140000"` (comma), which will cause silent ANE compile failures. **Always use the POSIX locale**:
```swift
String(format: "%.6f", locale: Locale(identifier: "en_US_POSIX"), value)
```
This applies to ALL float formatting in MIL generators, including scale factors, epsilon, and any numeric constant embedded in MIL text.

### ADR-5: Memory Management — `~Copyable` Structs with `UnsafeMutableBufferPointer`

**Recommendation**: Noncopyable structs wrapping `UnsafeMutableBufferPointer<Float>` with `deinit`.

**Rationale**: The steady-state working set is **~2.1 GiB** for Stories110M once kernels are compiled (DIM=768, HIDDEN=2048, SEQ=256, NLAYERS=12). Approx breakdown (MiB):
- **Layer weights**: 324.1 MiB
- **Layer gradients (accumulators)**: 324.1 MiB
- **Layer Adam (m+v)**: 648.1 MiB
- **Embedding weights**: 93.75 MiB
- **Embedding gradient**: 93.75 MiB
- **Embedding Adam (m+v)**: 187.5 MiB
- **Saved activations (for backward)**: ~135–162 MiB (Swift plan keeps Q/K/V on IOSurface; ObjC allocates CPU Q/K/V buffers but never reads them)
- **fp16 IOSurfaces for all per-layer kernels**: ~268.5 MiB
- **Logits + dlogits**: 62.5 MiB
- **Scratch buffers (dy/dx/dq/… excluding logits)**: ~12.3 MiB

At this scale, **accidental copies are catastrophic** (e.g., one `VOCAB*DIM` Float32 buffer is 93.75 MiB). `~Copyable` makes `let w2 = w1` a compile error, forcing explicit `borrowing`/`consuming`/`inout`. `deinit` provides RAII cleanup. `withUnsafePointer` gives zero-cost access for vDSP/cblas calls. Binary checkpoint save/load maps directly to `fwrite`/`fread` of the underlying buffer.

```swift
public struct TensorBuffer: ~Copyable {
    private let storage: UnsafeMutableBufferPointer<Float>
    public let count: Int
    public init(count: Int, zero: Bool = false) { ... }
    deinit { storage.baseAddress?.deallocate() }
    public borrowing func withPointer<T>(_ body: (UnsafePointer<Float>) throws -> T) rethrows -> T
    public mutating func withMutablePointer<T>(_ body: (UnsafeMutablePointer<Float>) throws -> T) rethrows -> T
}
```

### ADR-6: Concurrency — Keep GCD (Wrapped in Swift)

**Recommendation**: Direct GCD translation with thin Swift wrapper. No actors, no structured concurrency.

**Rationale**: The 9.3ms timing budget is binding. The concurrency pattern is simple: 1 serial dispatch queue, 1 dispatch group, 4 fire-and-forget `cblas_sgemm` blocks per layer, 3 barrier waits. Actor hops cost 1-5us each; with 48 hops/step that's 48-240us wasted (0.5-2.6% of budget) for zero benefit. `cblas_sgemm` is thread-safe and doesn't need actor isolation. The `UnsafeMutablePointer<Float>` captures in blocks need `@unchecked Sendable` wrappers regardless of approach. GCD is the right tool for this specific pattern.

```swift
public final class GradientAccumulator: @unchecked Sendable {
    private let queue = DispatchQueue(label: "ane.dw.cblas", qos: .userInitiated)
    private let group = DispatchGroup()
    public func accumulateAsync(_ work: @escaping @Sendable () -> Void) { ... }
    public func barrier() { group.wait() }
}
```

### ADR-7: Build System — SwiftPM with Mixed-Language Targets

**Recommendation**: SwiftPM.

**Rationale**: SwiftPM handles mixed ObjC/Swift targets natively (auto-generates module maps). No explicit ANE framework linking needed (loaded via `dlopen`). Per-target language mode (`.swiftLanguageMode(.v6)`) enables strict concurrency checking. Incremental builds track target-level dependencies. No `.xcodeproj` merge conflicts for a research tool.

---

## Phase Plan

### Phase 1: ANEInterop — C/ObjC Bridging Target (M)

**Scope**: Create new C/ObjC code extracted from `stories_io.h:87-134` and `ane_runtime.h:24-160`. This is the permanent interop layer.

**SwiftPM target**: `ANEInterop` (C language target)

**C API surface** (`ane_interop.h`):
```c
void ane_interop_init(void);
IOSurfaceRef ane_interop_create_surface(size_t bytes);
ANEHandle *ane_interop_compile(const uint8_t *milText, size_t milLen,
                                const char **weightPaths, const uint8_t **weightDatas,
                                const size_t *weightLens, int weightCount,
                                int nInputs, const size_t *inputSizes,
                                int nOutputs, const size_t *outputSizes);
bool ane_interop_eval(ANEHandle *handle);
IOSurfaceRef ane_interop_get_input(ANEHandle *handle, int index);
IOSurfaceRef ane_interop_get_output(ANEHandle *handle, int index);
IOSurfaceRef ane_interop_copy_input(ANEHandle *handle, int index);
IOSurfaceRef ane_interop_copy_output(ANEHandle *handle, int index);
void ane_interop_free(ANEHandle *handle);
int ane_interop_compile_count(void);
void ane_interop_set_compile_count(int value);
// NEON fp16<->fp32
void ane_interop_cvt_f32_to_f16(void *dst, const float *src, int count);
void ane_interop_cvt_f16_to_f32(float *dst, const void *src, int count);
// IOSurface fp16 helpers (backward pass hot path — 72 io_copy + 24 io_write_fp16_at calls/step)
bool ane_interop_io_copy(IOSurfaceRef dst, int dst_ch_off,
                         IOSurfaceRef src, int src_ch_off,
                         int channels, int spatial);
bool ane_interop_io_write_fp16_at(IOSurfaceRef surface, int ch_off,
                                  const float *data, int channels, int spatial);
```

**C interop boundary**: ALL `objc_msgSend` casts, `dlopen`, `NSClassFromString`, `CFBridgingRetain/Release`, `arm_neon.h` intrinsics, `@autoreleasepool`, and IOSurface fp16 I/O helpers (`io_copy`, `io_write_fp16_at`) stay in C/ObjC permanently.

**Tests (TDD, write first)**:
- `test_init_idempotent` — call twice, no crash
- `test_create_surface_valid` — verify `IOSurfaceGetAllocSize == requested`
- `test_compile_invalid_mil_returns_nil` — garbage input, expect NULL
- `test_compile_identity_kernel` — minimal MIL (cast fp32->fp16->fp32), eval, verify roundtrip
- `test_compile_count_increments` — reset, compile, verify count == 1
- `test_neon_f32_f16_roundtrip` — 100K random values, max error < 1e-2
- `test_io_copy_between_surfaces` — write data to surface A, `io_copy` to surface B at offset, verify fp16 data matches
- `test_io_write_fp16_at_offset` — write fp32 data at channel offset, read back, verify within 1e-2

**Verification**: All tests pass. `nm` confirms all symbols exported. Identity kernel produces numerically correct output.

**Risks**: Private API signature changes across macOS. Mitigation: pin to macOS 15+, document selectors.

**Done when**: All 8 tests green, identity kernel correct within 1e-2, zero ObjC imports visible to Swift consumers.

---

### Phase 2: ANETypes — Swift Types + IOSurface I/O (L)

**Scope**: Rewrite `stories_config.h` (types, constants) and IOSurface I/O helpers from `stories_io.h` into Swift.

**SwiftPM target**: `ANETypes` (depends on `ANEInterop`)

**Swift design**:
- `ModelConfig` — enum namespace for all constants (DIM=768, HIDDEN=2048, etc.)
- `LayerWeights: ~Copyable` — 9 `UnsafeMutableBufferPointer<Float>` fields with `deinit`
- `AdamState: ~Copyable` — m, v buffers with count
- `LayerAdam: ~Copyable` — 9 `AdamState` fields mirroring C `LayerAdam` (`Wq/Wk/Wv/Wo/W1/W2/W3/rmsAtt/rmsFfn`)
- `LayerActivations: ~Copyable` — 13 fp32 buffers matching current implementation: `layerIn`, `xnorm`, `Q`, `K`, `V`, `attnOut`, `oOut`, `x2`, `x2norm`, `h1`, `h3`, `siluOut`, `ffnOut`
- `LayerGradients: ~Copyable` — 9 gradient accumulator buffers with `zero()` method
- `CheckpointHeader` — `@frozen` struct with mandatory `validateLayout()` assertions (size, alignment, and field offsets for all Double fields) to catch Swift/C layout divergence. Preferred: define `CkptHdr` in `ANEInterop` C header and import directly
- `SurfaceIO` — enum with `writeFP16(to:data:channels:spatial:)`, `readFP16(from:into:channelOffset:channels:spatial:)`, `writeFP16At(to:channelOffset:data:channels:spatial:)`, and `copyFP16(dst:dstChannelOffset:src:srcChannelOffset:channels:spatial:)` — the last two call C shim functions for the backward pass hot path (72 `io_copy` + 24 `io_write_fp16_at` calls per step)
- `WeightBlob` — enum with `build(from:rows:cols:)` and `buildTransposed(from:rows:cols:)` producing `Data` with 128-byte header + fp16 payload

**C interop boundary**: `ane_interop_cvt_f32_to_f16/f16_to_f32` for NEON conversion, `ane_interop_io_copy` and `ane_interop_io_write_fp16_at` for backward pass IOSurface data movement. `IOSurfaceLock/Unlock` called directly from Swift (public C API).

**Tests (TDD)**:
- `test_layer_weights_alloc_dealloc_no_leak` — alloc/dealloc 100 iterations
- `test_adam_state_initialized_to_zero` — verify all m/v elements == 0
- `test_build_blob_header_magic` — bytes [0]=0x01, [4]=0x02, [64..67]=0xDEADBEEF
- `test_build_blob_fp16_accuracy` — known values, verify within 1e-2
- `test_build_blob_transposed_layout` — 3x4 matrix, verify column-major fp16
- `test_surface_write_read_roundtrip` — 100K random values, max error < 1e-2
- `test_surface_read_with_channel_offset` — read partial channel range
- `test_surface_write_fp16_at_offset` — write fp32 data at non-zero channel offset, read back entire surface, verify only target region changed
- `test_surface_copy_fp16_between_surfaces` — write to surface A, `copyFP16` to surface B at offset, verify fp16 data integrity
- `test_checkpoint_header_layout` — `MemoryLayout<CheckpointHeader>.size == 96`, plus field offset assertions (`cumCompile` at 48, `cumTrain` at 56, `cumWall` at 64, `cumSteps` at 72, `adamT` at 80, `pad2` at 92) to catch Swift/C layout divergence

**Verification**: Blob header/layout/transpose behavior and fp16 payload checks match fixtures and C layout expectations. fp16 roundtrip max error < 1e-2. Checkpoint header size AND field offsets match C struct (validated via `MemoryLayout.offset(of:)`).

**Risks**: `~Copyable` ownership semantics with `borrowing`/`consuming`/`inout`. Mitigation: extensive ownership tests.

**Done when**: All 10 tests green, blob byte-identical to ObjC, no memory leaks on 100-iteration loop.

---

### Phase 3: MILGenerator — MIL Text Generation (L) [parallel with Phase 4]

**Scope**: Rewrite `stories_mil.h` (6 generators) and `ane_mil_gen.h` (4 generic generators) into Swift.

**SwiftPM target**: `MILGenerator` (depends on `ANETypes`)

**Swift design**:
- `MILProgramGenerator` protocol — `milText: String`, `inputBytes: Int`, `outputByteSizes: [Int]` (single-output generators expose one-element arrays)
- 6 conforming structs: `SDPAForwardGenerator`, `FFNForwardGenerator`, `FFNBackwardGenerator`, `SDPABackward1Generator`, `SDPABackward2Generator`, `QKVBackwardGenerator`
- `GenericMIL` — enum with `conv()`, `matmul()`, `fusedQKV()`, `fusedFFNUp()` static methods
- `CausalMask` — enum with cached `blob(seqLen:) -> Data` for causal attention mask (match ObjC `g_mask_blob` behavior). Mask values must match ObjC exactly: fp16(0) on/under diagonal, fp16(-65504) above diagonal (min finite fp16, not `-inf`).
- **CRITICAL**: Use `String(format: "%.6f", locale: Locale(identifier: "en_US_POSIX"), value)` wherever ObjC uses `%f` — plain `String(format:)` is locale-dependent and will produce commas on European locales, causing silent ANE compile failures

**C interop boundary**: None. Pure Swift string operations. Causal mask blob calls `ane_interop_cvt_f32_to_f16` via `ANETypes`.

**Tests (TDD)**:
- `test_sdpa_fwd_text_matches_objc` — character-by-character comparison against ObjC output
- `test_ffn_fwd_text_matches_objc` — same
- `test_ffn_bwd_text_matches_objc` — same
- `test_sdpa_bwd1_text_matches_objc` — same
- `test_sdpa_bwd2_text_matches_objc` — same
- `test_qkvb_text_matches_objc` — same
- `test_causal_mask_diagonal_zero_upper_neg65504` — verify fp16 mask uses 0 / -65504 exactly (not `-inf`)
- `test_causal_mask_blob_cached_identity`
- `test_fused_qkv_blob_offsets_correct`

**Verification**: Character-identical MIL text comparison against checked-in ObjC-generated fixtures. This is critical — even whitespace differences cause ANE compile failures.

**Risks**: Swift `String(format:)` is locale-dependent (commas on European locales). Mitigation: Always use `Locale(identifier: "en_US_POSIX")` for all float formatting. Add a test that runs with `Locale(identifier: "de_DE")` to catch any missed format calls.

**Done when**: All 6 generators produce character-identical text, all fixture/contract tests green. ANE compile integration is validated in Phase 5 runtime tests.

---

### Phase 4: CPUOps — Accelerate/vDSP Operations (M) [parallel with Phase 3]

**Scope**: Rewrite `stories_cpu_ops.h` (RMSNorm, cross-entropy, Adam, embedding) plus CPU ops from `forward.h` (RoPE, SiLU) and `backward.h` (RMSNorm backward, attention backward, RoPE backward).

**SwiftPM target**: `CPUOps` (depends on `ANETypes`)

**Swift design**:
- `RMSNorm` — enum with `forward(output:input:weights:dim:seqLen:)` and `backward(dx:dw:dy:x:weights:dim:seqLen:)` using vDSP
- `CrossEntropy` — enum with `lossAndGradient(dlogits:logits:targets:vocabSize:seqLen:) -> Float` using vDSP_mtrans, vvexpf
- `AdamOptimizer` — enum with `update(weights:gradients:state:timestep:lr:beta1:beta2:eps:)`
- `Embedding` — enum with `lookup()` and `backward()` for channel-first layout
- `RoPE` — enum with `apply()` and `backward()` for rotary position embedding
- `SiLU` — enum with `forward()` and `backward()`

**C interop boundary**: None. All vDSP/cblas/Accelerate calls from Swift via `import Accelerate`.

**Tests (TDD)**:
- `test_rmsnorm_forward_known_values` — dim=4, seq=2, verify within 1e-5
- `test_rmsnorm_backward_numerical_gradient_check` — finite difference vs analytic, relative error < 1e-3
- `test_cross_entropy_uniform_logits` — loss = log(V)
- `test_cross_entropy_gradient_sums_to_zero`
- `test_adam_single_step_known_values`
- `test_adam_bias_correction`
- `test_embedding_lookup_correct_rows`
- `test_embedding_backward_accumulates`
- `test_rope_forward_backward_consistency`
- `test_silu_forward_backward_consistency`

**Verification**: Numerical gradient checking for all backward ops (relative error < 1e-3). Cross-entropy matches numpy reference within 1e-5.

**Risks**: vDSP parameter ordering. Mitigation: dedicated matrix multiply tests.

**Done when**: All 10 tests green, gradient checks pass on 100 random inputs, no Accelerate deprecation warnings.

---

### Phase 5: ANERuntime — Kernel Lifecycle in Swift (XL)

**Scope**: Build Swift kernel lifecycle layer replacing `compile_kern_mil_w`/`ane_eval`/`free_kern` from `stories_io.h`, kernel orchestration from `train_large.m:59-107`, and model loading from `model.h`.

**SwiftPM target**: `ANERuntime` (depends on `ANEInterop`, `ANETypes`, `MILGenerator`)

**Swift design**:
- `ANEKernel: ~Copyable` — owns `OpaquePointer` (ANEHandle*), typed throws `ANEError`, `deinit` calls `ane_interop_free`
- `ANEError` — `enum: Error, Sendable` with `.compilationFailed`, `.evaluationFailed`, `.compileBudgetExhausted`, `.surfaceAllocationFailed`
- `LayerKernelSet: ~Copyable` — owns 5 weight-bearing `ANEKernel` instances (fwdAttn, fwdFFN, ffnBwd, sdpaBwd1, qkvBwd), compiles from `borrowing LayerWeights`. Recompiled each batch when weights change.
- `StaticKernel: ~Copyable` — owns 1 weight-free `ANEKernel` (sdpaBwd2). Compiled once at startup, NOT recompiled with weight changes. One per layer. Different lifecycle from `LayerKernelSet`: freed at exec-restart, re-compiled after restart (see `train_large.m:303-307,323,350-356`).
- `ModelWeightLoader` — loads pretrained weights from llama2.c `.bin` format (Appendix A.5, `train_large.m:13-55` parity)
- `CompileBudget` — thin wrapper over `ane_interop_compile_count()` (C-level, no swift-atomics dependency)

```swift
public struct ANEKernel: ~Copyable {
    private let handle: OpaquePointer
    public init(milText: String, weights: [(path: String, data: Data)],
                inputBytes: Int, outputBytes: Int) throws(ANEError)
    public func eval() throws(ANEError)
    public var inputSurface: IOSurfaceRef { get }
    public var outputSurface: IOSurfaceRef { get }
    deinit { ane_interop_free(handle) }
}
```

**C interop boundary**: All calls through `ANEInterop` C API (Phase 1). No new C code.

**Tests (TDD)**:
- `test_compile_identity_kernel_succeeds`
- `test_compile_invalid_mil_throws`
- `test_eval_identity_roundtrip` — [1,2,3,4] -> eval -> verify within 1e-2
- `test_kernel_deinit_calls_free`
- `test_compile_layer_kernels_with_random_weights` — all 5 compile
- `test_fwd_attn_output_has_6xdim_channels`
- `test_fwd_attn_numerical_equivalence_with_objc` — same input+weights, max |diff| < 1e-2
- `test_fwd_ffn_numerical_equivalence_with_objc`
- `test_ffn_bwd_numerical_equivalence_with_objc`
- `test_model_weights_bin_layout_small` — tiny synthetic file, validate segment order and sizes
- `test_model_weights_vocab_sign_shared_vs_unshared` — vocab_size sign controls wcls presence
- `test_load_stories110m_weights` — integration (local asset), validate header matches ModelConfig
- `test_sdpa_bwd2_compile_once_reuse` — compile weight-free sdpaBwd2 once, eval with multiple different inputs, verify output changes correctly
- `test_sdpa_bwd2_lifecycle_independent` — verify sdpaBwd2 survives LayerKernelSet recompilation (different lifecycle)
- `test_backward_iosurface_copy_chain` — verify full forward→backward IOSurface data flow: fwdAttn output channel offsets feed correctly into sdpaBwd1/sdpaBwd2/qkvBwd inputs via `io_copy`

**Verification**: Compile both ObjC and Swift kernels with identical weights, feed identical input, compare element-by-element. Tolerance: max |swift - objc| < 1e-2. Keep ObjC code alive temporarily as reference.

**Risks**:
1. Weight blob format mismatch (even 1 wrong header byte = silent failure). Mitigation: byte-identical blob tests from Phase 2.
2. `ane_interop_eval` returns `false` on failure — Swift must propagate this as `throws(ANEError)`, not silently continue. The ObjC code ignores eval failures.
3. IOSurface allocation can fail under memory pressure — `ane_interop_create_surface` returns NULL. Wrap in `guard let` with `ANEError.surfaceAllocationFailed`.

**Done when**: All 13 tests green, forward attn/FFN match ObjC within 1e-2 on 10 random inputs, layer kernel set compiles in < 2000ms.

---

### Phase 6: Espresso — Forward, Backward, Loop, Checkpoint (XL)

**Scope**: Rewrite `train_large.m` (688 lines) into the final training pipeline. The production forward pass (lines 384-420) and backward pass (lines 461-575) are inline in `train_large.m` using fused ANE kernels — NOT from `forward.h`/`backward.h` (which implement a separate per-weight conv architecture used by test executables). CPU backward ops (rmsnorm_bwd, attention_bwd, etc.) were already ported in Phase 4.

**SwiftPM targets**: `Espresso` (depends on `ANERuntime`, `CPUOps`, `ANETypes`) + `EspressoTrain` (executable)

**Swift design**:
- `ForwardPass` — static `run()` iterating 12 layers: `SurfaceIO.writeFP16` -> `kernel.eval()` -> `SurfaceIO.readFP16` -> `vDSP_vadd` (residual)
- `BackwardPass` — static `run()` in reverse order: ANE backward kernels + async cblas dW via `GradientAccumulator`
- `Checkpoint` — `save()`/`load()` matching existing binary format exactly for cross-compatibility
- `TokenDataset` — mmaps `DATA_PATH` as `UInt16` tokens, validates `nTokens >= SEQ + 1`, munmaps on shutdown
- `Sampler` — uses `srand48(42 + startStep)` + `drand48()` to sample `pos ∈ [0, nTokens-SEQ-1)` exactly matching `train_large.m`
- `ExecRestart` — calls `execl(argv[0], "--resume")` via `import Darwin` when compile budget exhausted
- `GradientAccumulator` — `@unchecked Sendable` class wrapping `DispatchQueue` + `DispatchGroup`
- `EspressoTrain` — `@main` entry point with argument parsing, training loop, telemetry

**C interop boundary**: `execl()`/`srand48()`/`drand48()` via `import Darwin`, `mmap()/munmap()` via `import Darwin`, `cblas_sgemm` via `import Accelerate`, `DispatchQueue`/`DispatchGroup` from Swift Dispatch.

**Tests (TDD)**:
- `test_forward_single_layer_output_nonzero_finite`
- `test_forward_12_layers_no_nan`
- `test_backward_produces_nonzero_gradients`
- `test_backward_residual_gradient_flow`
- `test_checkpoint_save_load_roundtrip` — byte-identical weights
- `test_checkpoint_segment_order_small` — validates binary segment order/offsets without requiring a full-size checkpoint fixture
- `test_checkpoint_binary_compatible_with_objc` — load ObjC checkpoint in Swift
- `test_single_step_loss_matches_objc` — |swift - objc| < 0.01
- `test_10_steps_loss_decreases` — overfit on tiny pattern
- `test_gradient_accumulation_averages` — ACCUM_STEPS=2, verify 1/2 scaling
- `test_1_step_gradients_match_objc` — per-layer gradient norm relative error < 5%
- `test_100_steps_benchmark` — verify <= 9.3 ms/step on M4
- `test_gradient_accumulation_scaling` — verify `1.0/steps_batch` scaling applied before Adam (maps to train_large.m:601)
- `test_exec_restart_checkpoint_roundtrip` — save checkpoint at step N, load with `--resume`, verify training state (weights, adam_t, cum_* counters) restored exactly

**Verification**:
1. Start both ObjC and Swift from same weights + same `DATA_PATH` + same `srand48(42 + startStep)` seed and `drand48()` sampling
2. Run 10 steps, compare per-step loss (|diff| < 0.01), gradient norms (relative error < 5%), weight deltas (cosine similarity > 0.99)
3. Performance: 100-step benchmark, target <= 9.3 ms/step on M4

**Risks**:
1. `exec()` from Swift — ARC/deinit won't run (process image replaced). Mitigation: save checkpoint BEFORE exec.
2. GCD async dW overlap timing — must match ObjC pattern exactly. Mitigation: same DispatchGroup/wait pattern.
3. Performance regression from Swift overhead. Mitigation: `-O` optimization, profile with Instruments.
4. `~Copyable` + closures — capturing `borrowing` params in GCD blocks needs explicit buffer copies. Mitigation: same manual memcpy pattern as ObjC.
5. Checkpoint save failure (disk full, permissions) loses training progress. Mitigation: verify `fwrite` return value, report error before `execl`.
6. Kernel compile failure should force an `exec()` restart (ObjC sets `g_compile_count = MAX_COMPILES` at `train_large.m:348`). Mitigation: on any compile failure, call `ane_interop_set_compile_count(MAX_COMPILES)` and `continue` so the next loop iteration hits the compile budget check + checkpoint + exec-restart path.

**exec-restart lifecycle (must execute in this exact order)**:
1. `accumulator.barrier()` — wait for all async cblas dW blocks to complete
2. Free all `LayerKernelSet` instances (weight-bearing kernels)
3. Free all `StaticKernel` instances (sdpaBwd2)
4. Save checkpoint (verify write success before proceeding)
5. `fflush(stdout); fflush(stderr)` — flush all buffered output
6. `execl(argv[0], argv[0], "--resume", nil)` — replaces process image
NOTE: After `execl`, NO Swift deinit runs, NO ARC cleanup happens. The OS reclaims all memory. GCD blocks in flight are destroyed by the kernel.

**Done when**: All 14 tests green, loss matches ObjC within 0.01, gradients within 5%, benchmark <= 9.3 ms/step, exec restart works, checkpoint cross-compatible.

---

## Dependency Graph

```
Phase 1 (ANEInterop)
    ↓
Phase 2 (ANETypes)
    ↓          ↓
Phase 3      Phase 4        ← CAN RUN IN PARALLEL
(MILGen)     (CPUOps)
    ↓          ↓
Phase 5 (ANERuntime)
    ↓
Phase 6 (Espresso + EspressoTrain)
```

**Critical path**: 1 → 2 → 5 → 6 (Phases 3 & 4 parallelizable)

---

## SwiftPM Package Structure

```
Package.swift (swift-tools-version: 6.0)
Sources/
    ANEInterop/           (C target — PERMANENT)
        include/ane_interop.h
        ane_interop.m     (ObjC: dlopen, objc_msgSend, CFBridging)
        neon_convert.c    (NEON fp16<->fp32 shims)
        surface_io.c      (IOSurface fp16 io_copy + io_write_fp16_at hot path)
    ANETypes/             (Swift, depends on ANEInterop)
        ModelConfig.swift, TensorBuffer.swift, LayerStorage.swift,
        LayerWeights.swift, AdamState.swift, LayerAdam.swift,
        LayerActivations.swift, LayerGradients.swift, CheckpointHeader.swift,
        SurfaceIO.swift, WeightBlob.swift
    MILGenerator/         (Swift, depends on ANETypes)
        SDPAForwardGenerator.swift, FFNForwardGenerator.swift,
        FFNBackwardGenerator.swift, SDPABackward1Generator.swift,
        SDPABackward2Generator.swift, QKVBackwardGenerator.swift,
        GenericMIL.swift, CausalMask.swift
    CPUOps/               (Swift, depends on ANETypes)
        RMSNorm.swift, CrossEntropy.swift, AdamOptimizer.swift,
        Embedding.swift, RoPE.swift, SiLU.swift
    ANERuntime/           (Swift, depends on ANEInterop + ANETypes + MILGenerator)
        ANEKernel.swift, ANEError.swift, LayerKernelSet.swift,
        StaticKernel.swift, ModelWeightLoader.swift, CompileBudget.swift
    Espresso/             (Swift, depends on ANERuntime + CPUOps + ANETypes)
        ForwardPass.swift, BackwardPass.swift, Checkpoint.swift,
        TokenDataset.swift, Sampler.swift,
        ExecRestart.swift, GradientAccumulator.swift
    EspressoTrain/        (Swift executable, depends on Espresso)
        main.swift
Tests/
    ANEInteropTests/, ANETypesTests/, MILGeneratorTests/,
    CPUOpsTests/, ANERuntimeTests/, EspressoTests/
```

---

## Verification Strategy

| Phase | Method | Tolerance |
|-------|--------|-----------|
| 1 | Identity kernel eval roundtrip | fp16: 1e-2 (fp16 has 10 mantissa bits; 1e-4 only holds near 1.0) |
| 2 | Blob byte-identical to ObjC; fp16 roundtrip | fp16: 1e-2 |
| 3 | MIL text character-identical to ObjC generators | Exact match |
| 4 | Numerical gradient check (finite difference) | Relative: 1e-3 |
| 5 | Same-input kernel output: Swift vs ObjC | fp16: 1e-2 |
| 6 | End-to-end loss, gradients, benchmark | Loss: 0.01, Grad: 5%, Perf: 9.3ms |

**ObjC reference kept alive** through Phase 5 for numerical comparison. Can be removed after Phase 6 verification.

---

## Key Files

### Production Code (rewritten)

| File | Lines | Rewritten In |
|------|-------|-------------|
| `training/stories_config.h` | 190 | Phase 2 (types, constants, alloc helpers) |
| `training/stories_io.h` | 135 | Phase 1 (C shim: kernel lifecycle, NEON, io_copy, io_write_fp16_at) + Phase 2 (Swift: SurfaceIO, WeightBlob) |
| `training/ane_runtime.h` | 161 | Phase 1 (C shim reference for multi-I/O ANEKernel pattern) |
| `training/stories_mil.h` | 287 | Phase 3 (6 MIL generators) |
| `training/ane_mil_gen.h` | 209 | Phase 3 (generic MIL + blob builders) |
| `training/stories_cpu_ops.h` | 130 | Phase 4 (rmsnorm, cross-entropy, adam, embed) |
| `training/train_large.m:59-107` | ~50 | Phase 5 (compile_layer_kernels, compile_sdpa_bwd2, free_layer_kernels) |
| `training/train_large.m:384-420` | ~36 | Phase 6 (forward pass: 12-layer fused-kernel ANE loop) |
| `training/train_large.m:461-575` | ~114 | Phase 6 (backward pass: reverse-order ANE kernels + async cblas dW) |
| `training/train_large.m` (rest) | ~488 | Phase 6 (weight loading, checkpoint, exec-restart, training loop, telemetry) |

### Reference Only (NOT used by train_large.m — CPU ops ported in Phase 4)

| File | Lines | Use During Rewrite |
|------|-------|--------------------|
| `training/model.h` | 257 | Reference: per-weight conv kernel architecture (NOT the fused-kernel architecture used by train_large.m). CPU model loading patterns inform Phase 5 `ModelWeightLoader`. |
| `training/forward.h` | 180 | Reference: CPU implementations of RoPE, SiLU, attention, rmsnorm — port these to Phase 4 `CPUOps`. NOT the production forward pass (which is inline in train_large.m using fused ANE kernels). |
| `training/backward.h` | 309 | Reference: CPU implementations of rmsnorm_bwd, attention_bwd, rope_bwd, silu_bwd, gradient clipping — port to Phase 4 `CPUOps`. NOT the production backward pass (which is inline in train_large.m using ANE kernels + async cblas). |

**NOTE**: `train_large.m` includes `stories_io.h`, `stories_mil.h`, `stories_cpu_ops.h`. It does NOT include `model.h`, `forward.h`, or `backward.h`. These files implement a separate architecture (per-weight conv kernels) used by test executables.

---

## Resolved Decisions

- **Atomic compile counter**: Use C-level `ane_interop_compile_count()` from ANEInterop. No swift-atomics dependency.
- **macOS version floor**: macOS 15+ (Sequoia) only.
- **MIL test data**: Generate golden files from current ObjC code, commit as test fixtures. Swift tests compare against static files.
- **Model weights**: `stories110M.bin` available locally for integration tests in Phases 5-6.
- **Float formatting locale**: All `String(format:)` calls in MIL generators MUST use `Locale(identifier: "en_US_POSIX")` to prevent comma decimal separators on non-US locales.
- **CI without ANE hardware**: In XCTest, have ANE-dependent tests (Phases 1, 5, 6 kernel tests) `throw XCTSkip(...)` when ANE is unavailable so CI can still run the ~60% of tests that don't require hardware (Phases 2-4 pure logic, Phase 3 MIL text generation, checkpoint layout tests). Detect ANE availability via `dlopen`/class resolution in `ANEInterop`.

---

## Rollback Strategy

Each phase maintains backward compatibility with the existing ObjC codebase:

- **ObjC reference preservation**: The existing `Makefile` and all `.m`/`.h` files remain untouched throughout the rewrite. `make train_large` continues to work as the reference build. The ObjC code is only removed after Phase 6 verification passes.
- **Phase-level rollback**: If a phase fails numerical equivalence, delete the failing Swift target directory and its test target. No other targets are affected due to the bottom-up dependency structure.
- **Verification harness**: Phases 3-6 use a dual-execution test pattern: call the ObjC function, call the Swift function, compare outputs. This requires keeping the ObjC code compilable via the existing Makefile throughout.
- **Build system transition**: The existing `Makefile` (27 lines) remains the primary build for ObjC throughout. SwiftPM is introduced in Phase 1 and grows incrementally. Both build systems coexist until Phase 6 verification completes.

---

## Existing Test Assets

The codebase includes 10 test executables (~2,300 lines total) and reference implementations that should be leveraged:

| File | Lines | Use During Rewrite |
|------|-------|--------------------|
| `test_weight_reload.m` | 253 | Golden reference for Phase 5 (ModelWeightLoader) |
| `test_ane_advanced.m` | 245 | Validation for Phase 5 (ANEKernel lifecycle) |
| `test_ane_sdpa5.m` | 297 | Golden reference for Phase 5 (SDPA kernel equivalence) |
| `test_ane_causal_attn.m` | 295 | Validation for Phase 3 (CausalMask) + Phase 5 |
| `test_full_fused.m` | 379 | End-to-end kernel fusion validation (Phase 5-6) |
| `test_fused_qkv.m` | 265 | Golden reference for Phase 3 (fused QKV blob offsets) |
| `test_fused_bwd.m` | 184 | Backward kernel validation (Phase 5-6) |
| `test_perf_stats.m` | 233 | Benchmark reference for Phase 6 (9.3ms target) |
| `test_conv_attn3.m` | 276 | Conv-based attention validation (Phase 5) |
| `test_qos_sweep.m` | 157 | QoS parameter reference (Phase 5) |
| `tiny_train.m` | 593 | Self-contained training reference — use as simpler golden reference for Phase 6 end-to-end verification alongside `train_large.m` |

**Strategy**: Run existing ObjC test executables via `make` to generate expected outputs, then compare against Swift implementations. Do NOT port these to XCTest — they serve as external oracles.

---

## Appendix A: Critical Swift 6.2 Patterns

### A.1: ~Copyable + GCD Block Captures (Backward Pass)

`Array<~Copyable>` is NOT supported in Swift 6.2 (SE-0437 deferred stdlib collection adoption). `InlineArray<N, Element>` (SE-0453) works for fixed-size but has no `Sequence`/`Collection` conformance. Use `UnsafeMutableBufferPointer` as container:

```swift
/// Fixed-size container for ~Copyable layer data (replaces C array of structs).
/// Uses coroutine accessors (`_read`/`_modify`) supported in Swift 6.2.4 for
/// safe borrow/mutate access to noncopyable elements without copying.
struct LayerStorage<Element: ~Copyable>: ~Copyable {
    private let storage: UnsafeMutableBufferPointer<Element>
    let count: Int
    init(count: Int, initializer: (Int) -> Element) { ... }
    subscript(index: Int) -> Element {
        _read { yield storage[index] }
        _modify { yield &storage[index] }
    }
    deinit { storage.baseAddress!.deinitialize(count: count); storage.deallocate() }
}
```

For GCD block captures, use `~Copyable` + `@unchecked Sendable` wrapper with `deinit` for automatic cleanup (prevents leaks if GCD block is cancelled or process calls `exec()`):

```swift
/// Owns an exclusive heap copy of a float buffer. ~Copyable prevents aliasing,
/// @unchecked Sendable because we guarantee no aliasing after construction
/// (same safety as C malloc+memcpy). deinit provides automatic cleanup.
struct SendableBuffer: ~Copyable, @unchecked Sendable {
    let pointer: UnsafeMutablePointer<Float>
    let count: Int
    init(copying source: UnsafePointer<Float>, count: Int) {
        self.count = count
        self.pointer = .allocate(capacity: count)
        self.pointer.initialize(from: source, count: count)
    }
    deinit { pointer.deinitialize(count: count); pointer.deallocate() }
}

/// Sendable pointer wrapper for gradient accumulators (safe: serial queue + batch-lifetime).
struct SendablePointer: @unchecked Sendable {
    let pointer: UnsafeMutablePointer<Float>
    init(_ p: UnsafeMutablePointer<Float>) { self.pointer = p }
}

/// Sendable const pointer wrapper for direct pointer captures (e.g. dembed block).
struct SendableConstPointer: @unchecked Sendable {
    let pointer: UnsafePointer<Float>
    init(_ p: UnsafePointer<Float>) { self.pointer = p }
}
```

Backward pass dispatch pattern (maps to train_large.m:483-491):

```swift
// Snapshot buffers (heap copy, like ObjC malloc+memcpy)
// SendableBuffer is ~Copyable — consumed by the closure, deinit runs automatically after use
var captDffn = SendableBuffer(copying: dffn, count: seq * dim)
var captSilu = SendableBuffer(copying: siluOut, count: seq * hidden)
let grW2 = SendablePointer(grads[L].W2.baseAddress!)

accumulator.enqueue { [captDffn = consume captDffn, captSilu = consume captSilu] in
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                Int32(dim), Int32(hidden), Int32(seq), 1.0,
                captDffn.pointer, Int32(seq),
                captSilu.pointer, Int32(seq),
                1.0, grW2.pointer, Int32(hidden))
    // captDffn and captSilu deinit runs here automatically
}
```

### A.2: exec() Restart

```swift
import Darwin

enum ExecRestart {
    static func restart(step: Int, compileCount: Int, loss: Float) -> Never {
        fflush(stdout); fflush(stderr)
        print("[exec() restart step \(step), \(compileCount) compiles, loss=\(String(format: "%.4f", locale: Locale(identifier: "en_US_POSIX"), loss))]")
        fflush(stdout)
        // execl replaces process image — NO deinit, NO ARC cleanup runs.
        // OS reclaims all memory. Only fflush matters.
        let path = CommandLine.arguments[0]
        path.withCString { p in
            "--resume".withCString { r in
                execl(p, p, r, nil as UnsafePointer<CChar>?)
            }
        }
        perror("execl"); exit(1)
    }
}
```

**Key**: `execl()` does NOT run Swift deinit. Save checkpoint + `fflush` + free ANE kernels BEFORE calling. GCD blocks in flight are destroyed by the kernel.

### A.3: CheckpointHeader Binary Layout

**WARNING**: `@frozen` guarantees ABI stability between Swift modules but does NOT guarantee C-identical memory layout. Swift's layout algorithm may pack fields into tail padding of nested structs differently than C. For this specific struct (all flat scalars, no nesting), the layouts happen to match — but this MUST be verified at compile time.

**Preferred approach**: Define `CkptHdr` in the `ANEInterop` C target header and import it into Swift. This inherits C layout automatically. If a native Swift struct is preferred, use the `@frozen` version below with mandatory layout assertions:

```swift
@frozen
public struct CheckpointHeader {
    public var magic: Int32          // offset 0
    public var version: Int32        // offset 4
    public var step: Int32           // offset 8
    public var totalSteps: Int32     // offset 12
    public var nLayers: Int32        // offset 16
    public var vocabSize: Int32      // offset 20
    public var dim: Int32            // offset 24
    public var hiddenDim: Int32      // offset 28
    public var nHeads: Int32         // offset 32
    public var seqLen: Int32         // offset 36
    public var lr: Float             // offset 40
    public var loss: Float           // offset 44
    public var cumCompile: Double    // offset 48
    public var cumTrain: Double      // offset 56
    public var cumWall: Double       // offset 64
    public var cumSteps: Int32       // offset 72
    public var cumBatches: Int32     // offset 76
    public var adamT: Int32          // offset 80
    public var pad0: Int32           // offset 84
    public var pad1: Int32           // offset 88
    public var pad2: Int32           // offset 92
    // Total: 96 bytes, alignment: 8

    /// Call once at startup to verify layout matches C struct.
    /// Catches any Swift compiler layout divergence before silent data corruption.
    public static func validateLayout() {
        assert(MemoryLayout<CheckpointHeader>.size == 96, "Size mismatch with C CkptHdr")
        assert(MemoryLayout<CheckpointHeader>.alignment == 8, "Alignment mismatch")
        assert(MemoryLayout<CheckpointHeader>.offset(of: \.cumCompile)! == 48, "Double field offset mismatch")
        assert(MemoryLayout<CheckpointHeader>.offset(of: \.cumTrain)! == 56, "Double field offset mismatch")
        assert(MemoryLayout<CheckpointHeader>.offset(of: \.cumWall)! == 64, "Double field offset mismatch")
        assert(MemoryLayout<CheckpointHeader>.offset(of: \.cumSteps)! == 72, "Post-Double field offset mismatch")
        assert(MemoryLayout<CheckpointHeader>.offset(of: \.adamT)! == 80, "Post-Double field offset mismatch")
        assert(MemoryLayout<CheckpointHeader>.offset(of: \.pad2)! == 92, "Tail padding mismatch")
    }
}
```

Use individual `pad0/pad1/pad2` fields (not tuple) for `MemoryLayout.offset(of:)` verification. Read/write via `withUnsafeBytes(of: &self) { fwrite($0.baseAddress!, 1, $0.count, file) }`. Call `CheckpointHeader.validateLayout()` at process start in `EspressoTrain/main.swift`.

### A.4: Checkpoint Binary Format (EXACT)

Maps to `training/train_large.m:110-179`. All values are native-endian (little-endian on Apple Silicon). All payload segments are raw contiguous `float` (Float32) arrays.

**File segment order**:
1. Header: `CkptHdr` / `CheckpointHeader` (96 bytes)
2. Per-layer payload, for `L = 0 ..< NLAYERS`, in this exact order:
   - Weights: `Wq`, `Wk`, `Wv`, `Wo` (each `WQ_SZ` floats), then `W1` (`W1_SZ`), `W2` (`W2_SZ`), `W3` (`W3_SZ`), then `rms_att` (`DIM`), `rms_ffn` (`DIM`)
   - Adam (m then v for each parameter, exact order):
     - `Wq.m`, `Wq.v`, `Wk.m`, `Wk.v`, `Wv.m`, `Wv.v`, `Wo.m`, `Wo.v`
     - `W1.m`, `W1.v`, `W2.m`, `W2.v`, `W3.m`, `W3.v`
     - `rms_att.m`, `rms_att.v`, `rms_ffn.m`, `rms_ffn.v`
3. Global payload after all layers:
   - `rms_final` (`DIM`)
   - `arms_final.m` (`DIM`), then `arms_final.v` (`DIM`)
   - `embed` (`VOCAB * DIM`)
   - `aembed.m` (`VOCAB * DIM`), then `aembed.v` (`VOCAB * DIM`)

**Guardrails**:
- Validate header dimensions (`n_layers/dim/hidden_dim/vocab_size/seq_len`) before reading bulk payload.
- Use `Int32` for all C `int` fields (never Swift `Int`) and `Float` for all C `float` payloads to avoid silent layout/size drift.

---

### A.5: Pretrained Weights Binary Format (llama2.c, `stories110M.bin`)

Maps to `training/train_large.m:13-55` and `training/stories_config.h:108-110`.

**Header**: `Llama2Config` (7 × 32-bit signed integers, native-endian).
Field order:
1. `dim`
2. `hidden_dim`
3. `n_layers`
4. `n_heads`
5. `n_kv_heads`
6. `vocab_size` (**sign encodes sharing**; `V = abs(vocab_size)`)
7. `seq_len`

**Payload**: contiguous Float32 arrays (native-endian), in this exact order:
1. `embed` — `V * dim` floats (`[V, dim]` row-major)
2. For `L = 0 ..< n_layers`:
   - `rms_att[L]` — `dim` floats
3. For `L = 0 ..< n_layers`:
   - `Wq[L]` — `dim * dim` floats
4. For `L = 0 ..< n_layers`:
   - `Wk[L]` — `dim * dim` floats
5. For `L = 0 ..< n_layers`:
   - `Wv[L]` — `dim * dim` floats
6. For `L = 0 ..< n_layers`:
   - `Wo[L]` — `dim * dim` floats
7. For `L = 0 ..< n_layers`:
   - `rms_ffn[L]` — `dim` floats
8. For `L = 0 ..< n_layers`:
   - `W1[L]` — `hidden_dim * dim` floats
9. For `L = 0 ..< n_layers`:
   - `W2[L]` — `dim * hidden_dim` floats
10. For `L = 0 ..< n_layers`:
   - `W3[L]` — `hidden_dim * dim` floats
11. `rms_final` — `dim` floats
12. Optional `wcls` — `V * dim` floats only when `vocab_size < 0` (unshared classifier). When `vocab_size > 0`, classifier weights are shared with `embed`.

**Loader rules** (`ModelWeightLoader`):
- Validate `dim/hidden_dim/n_layers` match `ModelConfig` for Stories110M before reading bulk payload.
- Support both shared and unshared classifier weights. If unshared weights exist but the training pipeline assumes sharing, fail fast with a clear error (or plumb `wcls` through explicitly as a separate buffer).
- For tests: generate a tiny synthetic file with small config values to validate segment order and the `vocab_size` sign rule without depending on the real 100MB+ asset.

---

## Appendix B: Training Loop Control Flow (Exact Sequence)

### B.1: Forward Pass Per Layer (maps to train_large.m:385-420)

```
For L = 0..11:
  1. memcpy(acts[L].layerIn ← x_cur)              // save for rmsnorm bwd
  2. accumulator.barrier()                          // wait for prior layer dW
  3. io_write_fp16(fwdAttn.ioIn ← x_cur, DIM, SEQ)
  4. ane_eval(fwdAttn)
  5. io_read_fp16(fwdAttn.ioOut → o_out,    offset=0,     DIM, SEQ)
  6. io_read_fp16(fwdAttn.ioOut → attn_out, offset=4*DIM, DIM, SEQ)
  7. io_read_fp16(fwdAttn.ioOut → xnorm,    offset=5*DIM, DIM, SEQ)
  8. vDSP_vadd(x_cur + o_out → x2)                 // residual
  9. io_write_fp16(fwdFFN.ioIn ← x2, DIM, SEQ)
  10. ane_eval(fwdFFN)
  11. io_read_fp16(fwdFFN.ioOut → ffn_out,  offset=0,           DIM, SEQ)
  12. io_read_fp16(fwdFFN.ioOut → h1,       offset=DIM,         HIDDEN, SEQ)
  13. io_read_fp16(fwdFFN.ioOut → h3,       offset=DIM+HIDDEN,  HIDDEN, SEQ)
  14. io_read_fp16(fwdFFN.ioOut → silu_out, offset=DIM+2*HIDDEN,HIDDEN, SEQ)
  15. io_read_fp16(fwdFFN.ioOut → x2norm,   offset=DIM+3*HIDDEN,DIM, SEQ)
  16. vDSP_vadd(x2 + ffn_out → x_cur)              // residual, next layer input
```

### B.2: Backward Pass Per Layer (maps to train_large.m:461-575)

```
For L = 11 down to 0:
  // Offsets below are channel offsets (not bytes). spatial = SEQ. SCORE_CH = HEADS * SEQ.

  1. FFN backward (ANE):
     a. io_write_fp16_at(kern[L].ffnBwd->ioIn, dst=0,        src=dffn,     channels=DIM,        spatial=SEQ)
     b. io_copy(kern[L].ffnBwd->ioIn,         dst=DIM,      kern[L].fwdFFN->ioOut, src=DIM, channels=2*HIDDEN, spatial=SEQ)  // h1|h3
     c. ane_eval(kern[L].ffnBwd)
     d. io_read_fp16(kern[L].ffnBwd->ioOut,   src=0,        → dx_ffn,     channels=DIM,        spatial=SEQ)
     e. io_read_fp16(kern[L].ffnBwd->ioOut,   src=DIM,      → dh1,        channels=HIDDEN,     spatial=SEQ)
     f. io_read_fp16(kern[L].ffnBwd->ioOut,   src=DIM+HIDDEN→ dh3,        channels=HIDDEN,     spatial=SEQ)
     g. ASYNC dW: captDffn=copy(dffn), captSilu=copy(silu), captDh1, captDh3, captX2n
        → cblas: dW2 += dffn @ silu^T, dW1 += dh1 @ x2norm^T, dW3 += dh3 @ x2norm^T

  2. RMSNorm2 backward (CPU):
     a. rmsnorm_bwd(dx2, grads[L].rmsFfn, dx_ffn, x2, weights[L].rmsFfn)
     b. dx2 += dy  // residual gradient

  3. dWo async (CPU):
     a. captDo=copy(dx2), captAttn=copy(attn_out)
        → cblas: dWo += dx2 @ attn_out^T

  4. SDPA backward (ANE, two kernels):
     a. io_copy(kern[L].sdpaBwd1->ioIn, dst=0,      kern[L].fwdAttn->ioOut, src=DIM, channels=3*DIM,      spatial=SEQ)  // Q|K|V
     b. io_write_fp16_at(kern[L].sdpaBwd1->ioIn, dst=3*DIM, src=dx2,       channels=DIM,        spatial=SEQ)
     c. ane_eval(kern[L].sdpaBwd1)  → dv, dscores
     d. io_copy(sdpaBwd2[L]->ioIn,  dst=0,      kern[L].sdpaBwd1->ioOut, src=DIM, channels=2*SCORE_CH, spatial=SEQ)  // dscores
     e. io_copy(sdpaBwd2[L]->ioIn,  dst=2*SCORE_CH, kern[L].fwdAttn->ioOut, src=DIM, channels=2*DIM, spatial=SEQ)      // Q|K
     f. ane_eval(sdpaBwd2[L])  → dq, dk
     g. io_read_fp16(sdpaBwd2[L]->ioOut, src=0,   → dq, channels=DIM, spatial=SEQ)
     h. io_read_fp16(sdpaBwd2[L]->ioOut, src=DIM, → dk, channels=DIM, spatial=SEQ)
     i. io_read_fp16(kern[L].sdpaBwd1->ioOut, src=0, → dv, channels=DIM, spatial=SEQ)

  5. dWq/dWk/dWv async (CPU):
     a. captDq, captDk, captDv, captXnorm
        → cblas: dWq += dq @ xnorm^T, dWk += dk @ xnorm^T, dWv += dv @ xnorm^T

  6. QKV backward (ANE):
     a. io_copy(kern[L].qkvBwd->ioIn, dst=0,      sdpaBwd2[L]->ioOut,        src=0, channels=2*DIM, spatial=SEQ)  // dq|dk
     b. io_copy(kern[L].qkvBwd->ioIn, dst=2*DIM,  kern[L].sdpaBwd1->ioOut,   src=0, channels=DIM,   spatial=SEQ)  // dv
     c. ane_eval(kern[L].qkvBwd)  → dx_attn
     d. io_read_fp16(kern[L].qkvBwd->ioOut, src=0 → dx_attn, channels=DIM, spatial=SEQ)

  7. RMSNorm1 backward (CPU):
     a. rmsnorm_bwd(dx_rms1, grads[L].rmsAtt, dx_attn, layer_in, weights[L].rmsAtt)
     b. dy = dx_rms1 + dx2  // propagate to previous layer
```

**Exact IOSurface I/O Table (Per Layer, Backward Pass)**:

| # | API | dst surface/buffer | dst ch off | src surface/buffer | src ch off | channels | spatial | B.4 layout mapping |
|---|---|---|---|---|---|---|---|---|
| 1 | `io_write_fp16_at` | `kern[L].ffnBwd->ioIn` | 0 | `CPU:dffn` | 0 | `DIM` | `SEQ` | ffnBwd input `[dffn\|h1\|h3]` |
| 2 | `io_copy` | `kern[L].ffnBwd->ioIn` | `DIM` | `kern[L].fwdFFN->ioOut` | `DIM` | `2*HIDDEN` | `SEQ` | fwdFFN output `[ffn\|h1\|h3\|silu\|x2norm]` → copy `h1\|h3` |
| 3 | `io_read_fp16` | `CPU:dx_ffn` | 0 | `kern[L].ffnBwd->ioOut` | 0 | `DIM` | `SEQ` | ffnBwd output `[dx\|dh1\|dh3]` |
| 4 | `io_read_fp16` | `CPU:dh1` | 0 | `kern[L].ffnBwd->ioOut` | `DIM` | `HIDDEN` | `SEQ` | ffnBwd output `dh1` |
| 5 | `io_read_fp16` | `CPU:dh3` | 0 | `kern[L].ffnBwd->ioOut` | `DIM+HIDDEN` | `HIDDEN` | `SEQ` | ffnBwd output `dh3` |
| 6 | `io_copy` | `kern[L].sdpaBwd1->ioIn` | 0 | `kern[L].fwdAttn->ioOut` | `DIM` | `3*DIM` | `SEQ` | fwdAttn output `[o\|Q\|K\|V\|attn\|xnorm]` → copy `Q\|K\|V` |
| 7 | `io_write_fp16_at` | `kern[L].sdpaBwd1->ioIn` | `3*DIM` | `CPU:dx2` | 0 | `DIM` | `SEQ` | sdpaBwd1 input `[Q\|K\|V\|dx2]` |
| 8 | `io_copy` | `sdpaBwd2[L]->ioIn` | 0 | `kern[L].sdpaBwd1->ioOut` | `DIM` | `2*SCORE_CH` | `SEQ` | sdpaBwd1 output `[dv\|dscores]` → copy `dscores` |
| 9 | `io_copy` | `sdpaBwd2[L]->ioIn` | `2*SCORE_CH` | `kern[L].fwdAttn->ioOut` | `DIM` | `2*DIM` | `SEQ` | sdpaBwd2 input `[dscores\|Q\|K]` |
| 10 | `io_read_fp16` | `CPU:dq` | 0 | `sdpaBwd2[L]->ioOut` | 0 | `DIM` | `SEQ` | sdpaBwd2 output `[dq\|dk]` |
| 11 | `io_read_fp16` | `CPU:dk` | 0 | `sdpaBwd2[L]->ioOut` | `DIM` | `DIM` | `SEQ` | sdpaBwd2 output `dk` |
| 12 | `io_read_fp16` | `CPU:dv` | 0 | `kern[L].sdpaBwd1->ioOut` | 0 | `DIM` | `SEQ` | sdpaBwd1 output `dv` |
| 13 | `io_copy` | `kern[L].qkvBwd->ioIn` | 0 | `sdpaBwd2[L]->ioOut` | 0 | `2*DIM` | `SEQ` | qkvBwd input `[dq\|dk\|dv]` (copy dq\|dk) |
| 14 | `io_copy` | `kern[L].qkvBwd->ioIn` | `2*DIM` | `kern[L].sdpaBwd1->ioOut` | 0 | `DIM` | `SEQ` | qkvBwd input `dv` |
| 15 | `io_read_fp16` | `CPU:dx_attn` | 0 | `kern[L].qkvBwd->ioOut` | 0 | `DIM` | `SEQ` | qkvBwd output `[dx_attn]` |

### B.3: Async dW Blocks Summary

| Block | Per-Layer | Captured Buffers | cblas Operations | Gradient Targets |
|-------|-----------|------------------|------------------|------------------|
| dembed | Once/step | `dlogits`, `x_final` (const) + `gembed` (mut) captured as `SendableConstPointer`/`SendablePointer` (no heap copy) | dlogits @ x_final^T | gembed[VOCAB,DIM] |
| FFN dW | 12x/step | dffn, silu, dh1, dh3, x2norm (5 bufs) | dW2, dW1, dW3 (3 sgemm) | gr.W2, gr.W1, gr.W3 |
| dWo | 12x/step | dx2, attn_out (2 bufs) | dWo (1 sgemm) | gr.Wo |
| QKV dW | 12x/step | dq, dk, dv, xnorm (4 bufs) | dWq, dWk, dWv (3 sgemm) | gr.Wq, gr.Wk, gr.Wv |

**Barrier points**: (1) Before each fwdAttn write (line 392), (2) Before embed_backward (line 578), (3) Before Adam update (line 601).

### B.4: IOSurface Channel Layout Summary

| Kernel | Input Layout | Output Layout |
|--------|-------------|---------------|
| fwdAttn | [DIM, SEQ] | [o\|Q\|K\|V\|attn\|xnorm] = [6*DIM, SEQ] |
| fwdFFN | [DIM, SEQ] | [ffn\|h1\|h3\|silu\|x2norm] = [2*DIM+3*HIDDEN, SEQ] |
| ffnBwd | [dffn\|h1\|h3] = [DIM+2*HIDDEN, SEQ] | [dx\|dh1\|dh3] = [DIM+2*HIDDEN, SEQ] |
| sdpaBwd1 | [Q\|K\|V\|dx2] = [4*DIM, SEQ] | [dv\|dscores] = [DIM+2*SCORE_CH, SEQ] |
| sdpaBwd2 | [dscores\|Q\|K] = [2*SCORE_CH+2*DIM, SEQ] | [dq\|dk] = [2*DIM, SEQ] |
| qkvBwd | [dq\|dk\|dv] = [3*DIM, SEQ] | [dx_attn] = [DIM, SEQ] |

Where SCORE_CH = HEADS * SEQ = 12 * 256 = 3072.

---

## Appendix C: Package.swift (Complete)

```swift
// swift-tools-version: 6.0
// NOTE: swift-tools-version 6.0 is intentional — it provides all needed Swift 6.x language
// features. 6.1 adds Package Traits, 6.2 adds diagnostic group control, neither is needed here.
// WARNING: .unsafeFlags usage below prevents this package from being consumed as a dependency
// by other SwiftPM packages. Acceptable for a standalone training tool.
import PackageDescription

let package = Package(
    name: "Espresso",
    platforms: [.macOS(.v15)],
    products: [
        .executable(name: "espresso-train", targets: ["EspressoTrain"]),
        .library(name: "Espresso", targets: ["Espresso"]),
    ],
    targets: [
        .target(
            name: "ANEInterop",
            path: "Sources/ANEInterop",
            publicHeadersPath: "include",
            cSettings: [
                .unsafeFlags(["-fobjc-arc"]),
                .headerSearchPath("include"),
            ],
            linkerSettings: [
                .linkedFramework("Foundation"),
                .linkedFramework("CoreML"),
                .linkedFramework("IOSurface"),
                .unsafeFlags(["-ldl"]),
            ]
        ),
        .target(
            name: "ANETypes",
            dependencies: ["ANEInterop"],
            path: "Sources/ANETypes",
            swiftSettings: [.swiftLanguageMode(.v6)],
            linkerSettings: [.linkedFramework("IOSurface")]
        ),
        .target(
            name: "MILGenerator",
            dependencies: ["ANETypes"],
            path: "Sources/MILGenerator",
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),
        .target(
            name: "CPUOps",
            dependencies: ["ANETypes"],
            path: "Sources/CPUOps",
            swiftSettings: [.swiftLanguageMode(.v6)],
            linkerSettings: [.linkedFramework("Accelerate")]
        ),
        .target(
            name: "ANERuntime",
            dependencies: ["ANEInterop", "ANETypes", "MILGenerator"],
            path: "Sources/ANERuntime",
            swiftSettings: [.swiftLanguageMode(.v6)],
            linkerSettings: [.linkedFramework("IOSurface")]
        ),
        .target(
            name: "Espresso",
            dependencies: ["ANERuntime", "CPUOps", "ANETypes"],
            path: "Sources/Espresso",
            swiftSettings: [.swiftLanguageMode(.v6)],
            linkerSettings: [.linkedFramework("Accelerate")]
        ),
        .executableTarget(
            name: "EspressoTrain",
            dependencies: ["Espresso"],
            path: "Sources/EspressoTrain",
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),
        .testTarget(name: "ANEInteropTests", dependencies: ["ANEInterop"], swiftSettings: [.swiftLanguageMode(.v6)]),
        .testTarget(name: "ANETypesTests", dependencies: ["ANETypes"], swiftSettings: [.swiftLanguageMode(.v6)]),
        .testTarget(
            name: "MILGeneratorTests",
            dependencies: ["MILGenerator"],
            path: "Tests/MILGeneratorTests",
            resources: [.process("Fixtures")],
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),
        .testTarget(name: "CPUOpsTests", dependencies: ["CPUOps"], swiftSettings: [.swiftLanguageMode(.v6)]),
        .testTarget(name: "ANERuntimeTests", dependencies: ["ANERuntime"], swiftSettings: [.swiftLanguageMode(.v6)]),
        .testTarget(name: "EspressoTests", dependencies: ["Espresso"], swiftSettings: [.swiftLanguageMode(.v6)]),
    ]
)
```

---

## Appendix D: Golden File Test Infrastructure

### D.1: Generating Golden Files

Script to compile and run ObjC generators, capturing output:

```bash
#!/bin/bash
# scripts/generate_golden_mil.sh
# Compiles a small ObjC program that calls each MIL generator and writes to files.
cd training
xcrun clang -O2 -fobjc-arc -framework Foundation \
    -DGENERATE_GOLDEN=1 golden_mil_gen.m -o /tmp/golden_mil_gen
/tmp/golden_mil_gen Tests/MILGeneratorTests/Fixtures/
```

The `golden_mil_gen.m` helper calls each gen_*() function and writes the NSString to a file.

### D.2: Swift Test Pattern

```swift
func test_sdpa_fwd_text_matches_golden() throws {
    let golden = try String(contentsOfFile: fixturesPath + "/sdpa_fwd_taps.mil")
    let generator = SDPAForwardGenerator()
    XCTAssertEqual(generator.milText, golden,
        "MIL text differs from golden file. Regenerate with scripts/generate_golden_mil.sh")
}
```

### D.3: Float Formatting

ObjC `%f` defaults to 6 decimal places. Swift `\(Float)` may produce different precision.

**CRITICAL**: `String(format:)` is locale-dependent — on European locales it produces commas instead of periods, silently breaking MIL text. Rule: **always use `String(format: "%.6f", locale: Locale(identifier: "en_US_POSIX"), value)`** in MIL generators for any float constant (scale factors, epsilon, etc.).

Add a regression test to verify locale invariance. **NOTE**: `Locale.current` is read-only in Swift — you cannot override it at runtime. Two approaches:

**Approach A (preferred)**: Use `setenv` before the format call to override the C locale used by `String(format:)`:
```swift
func test_locale_does_not_affect_mil_formatting() throws {
    // Force process-level locale to German (comma decimal separator)
    let old = String(cString: setlocale(LC_ALL, nil))
    defer { _ = setlocale(LC_ALL, old) }
    setenv("LC_ALL", "de_DE.UTF-8", 1)
    setenv("LANG", "de_DE.UTF-8", 1)
    guard setlocale(LC_ALL, "") != nil else { throw XCTSkip("de_DE.UTF-8 locale not available") }
    let generator = SDPAForwardGenerator()
    let golden = try String(contentsOfFile: fixturesPath + "/sdpa_fwd_taps.mil")
    XCTAssertEqual(generator.milText, golden,
        "MIL text is locale-dependent! Use Locale(identifier: \"en_US_POSIX\") in all String(format:) calls")
}
```

**Approach B (static analysis)**: Grep all `String(format:` calls in `Sources/MILGenerator/` and verify every one passes an explicit `locale:` parameter. Add as a build-time script phase or pre-commit hook.
