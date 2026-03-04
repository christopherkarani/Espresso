# ANE Swift 6.2 Rewrite — Plan Review

## Executive Summary

The plan is **architecturally sound** and demonstrates exceptional depth in understanding both the ObjC codebase and Swift 6.2's constraints. The 6-phase bottom-up structure, ADRs, and TDD verification strategy are well-designed. However, the review uncovered **3 critical issues**, **5 high-severity issues**, and **4 medium-severity issues** that must be addressed before implementation begins:

1. **CRITICAL**: The Key Files table creates a false dependency on `model.h`/`forward.h`/`backward.h`, which are NOT used by `train_large.m`. The production training loop uses inline fused-kernel patterns with `stories_*.h` headers — the plan's Phase descriptions are correct, but the Key Files table is misleading.
2. **CRITICAL**: Two IOSurface functions used extensively in the backward pass (`io_copy`, `io_write_fp16_at`) are absent from the proposed C API surface.
3. **CRITICAL**: SE-0474 (yielding accessors) IS stable in Swift 6.2 — the plan's reliance on `_read`/`_modify` compiler internals is unnecessary and fragile.

The plan gets the hard things right (GCD over actors, `~Copyable` ownership, locale-safe formatting, checkpoint layout validation). The issues below are fixable without architectural changes.

---

## Step 1: Codebase Validation

### Line Counts

| File | Plan Claims | Actual | Delta | Verdict |
|------|------------|--------|-------|---------|
| `stories_config.h` | 190 | 190 | 0 | MATCH |
| `stories_io.h` | 135 | 135 | 0 | MATCH |
| `ane_runtime.h` | 161 | 161 | 0 | MATCH |
| `stories_mil.h` | 286 | 287 | +1 | MATCH (within tolerance) |
| `ane_mil_gen.h` | 209 | 209 | 0 | MATCH |
| `stories_cpu_ops.h` | 130 | 130 | 0 | MATCH |
| `model.h` | 256 | 257 | +1 | MATCH (within tolerance) |
| `forward.h` | 179 | 180 | +1 | MATCH (within tolerance) |
| `backward.h` | 308 | 309 | +1 | MATCH (within tolerance) |
| `train_large.m` | 687 | 688 | +1 | MATCH (within tolerance) |

All line counts match within +/- 1 line tolerance. No phantom files.

### Include/Dependency Validation

**CRITICAL FINDING**: `train_large.m` includes:
```c
#include "stories_io.h"      // ← includes stories_config.h transitively
#include "stories_mil.h"
#include "stories_cpu_ops.h"
```

It does **NOT** include `model.h`, `forward.h`, or `backward.h`. These three files form a **separate, alternative architecture** that uses per-weight conv kernels (individual Q/K/V/O projections) rather than the fused kernel approach used in production. They are consumed by the test executables (`test_ane_advanced.m`, `test_ane_sdpa5.m`, etc.) but NOT by the production training loop.

### Key Files Table Assessment

| File | Plan's "Rewritten In" | Actual Role | Verdict |
|------|----------------------|-------------|---------|
| `stories_config.h` | Phase 2 | Correct — types, constants, alloc helpers | OK |
| `stories_io.h` | Phase 1 (C) + Phase 2 (Swift) | Correct — IOSurface + NEON + kernel lifecycle | OK |
| `stories_mil.h` | Phase 3 | Correct — 6 MIL generators | OK |
| `ane_mil_gen.h` | Phase 3 | Correct — generic MIL + blob builders | OK |
| `stories_cpu_ops.h` | Phase 4 | Correct — rmsnorm, cross-entropy, adam, embed | OK |
| `model.h` | Phase 5 | **MISLEADING** — NOT used by train_large.m | FIX |
| `forward.h` | Phase 6 | **MISLEADING** — NOT used by train_large.m | FIX |
| `backward.h` | Phase 6 | **MISLEADING** — NOT used by train_large.m | FIX |
| `train_large.m` | Phase 6 | Correct — production training loop | OK |

The plan's **Phase descriptions** correctly describe the `train_large.m` architecture (fused kernels, IOSurface `io_copy`, GCD dispatch). The confusion is only in the Key Files table. `model.h`/`forward.h`/`backward.h` contain useful reference implementations (CPU attention forward/backward, RoPE, SiLU, gradient clipping) that Phase 4 should port, but they are NOT the code being replaced.

### Symbol Coverage

| Symbol Category | Count in ObjC | Covered by Plan | Missing |
|-----------------|--------------|-----------------|---------|
| `objc_msgSend` cast signatures | 14 | 14 (ADR-2) | 0 |
| `dlopen` calls | 1 | 1 (ane_interop_init) | 0 |
| `NSClassFromString` | 4 classes (g_D, g_I, g_AR, g_AIO) | 4 (ane_interop_init) | 0 |
| `CFBridgingRetain` | 4 (model, request, tmpDir, wI) | All (ane_interop_compile) | 0 |
| `@autoreleasepool` | 1 (compile_kern_mil_w) | 1 (ane_interop.m) | 0 |
| `__sync_fetch_and_add` | 1 (g_compile_count) | 1 (ane_interop_compile_count) | 0 |
| `io_copy` | 6 uses in backward pass | **0** | **1 function** |
| `io_write_fp16_at` | 2 uses in backward pass | **0** | **1 function** |

---

## Step 2: Swift 6.2 Assumptions Audit

| # | Claim | Verdict | Evidence | Impact | Fix |
|---|-------|---------|----------|--------|-----|
| 1 | `~Copyable` structs with `deinit` (ADR-5) | **CORRECT** | SE-0390 stable since Swift 5.9; deinit on noncopyable types works | None | — |
| 2 | `Array<~Copyable>` not supported (Appendix A.1) | **CORRECT** | SE-0437 deferred stdlib collection adoption of noncopyable types | None — plan correctly uses `UnsafeMutableBufferPointer` workaround | — |
| 3 | `InlineArray` has no `Sequence`/`Collection` (Appendix A.1) | **CORRECT** | SE-0453 `InlineArray` lacks protocol conformances | None | — |
| 4 | `_read`/`_modify` for `LayerStorage` subscript (Appendix A.1) | **WRONG** | SE-0474 (`yielding borrow`/`yielding mutate`) IS stable in Swift 6.2 | **HIGH** — `_read`/`_modify` are unsupported compiler internals that could break without notice | Use `yielding borrow { ... }` and `yielding mutate { ... }` instead |
| 5 | `@frozen` does NOT guarantee C layout (Appendix A.3) | **CORRECT** | `@frozen` is ABI-stable between Swift modules; NOT layout-compatible with C | None — plan correctly adds `validateLayout()` assertions | — |
| 6 | `swift-tools-version: 6.0` with `.swiftLanguageMode(.v6)` (ADR-7) | **CORRECT** | SwiftPM 6.0 supports per-target language modes | None | — |
| 7 | `~Copyable` + `@unchecked Sendable` compose (Appendix A.1) | **CORRECT** | Both are independent conformance/suppression markers | None | — |
| 8 | Typed throws `init() throws(ANEError)` on `~Copyable` | **RISK** | Known compiler bug (#68122) with interleaved throw/property-init in `~Copyable` initializers | **MEDIUM** — may hit compiler crash | Order all property assignments before any `throw`; or use two-phase init pattern |
| 9 | `String(format:locale:)` locale safety (ADR-4) | **CORRECT** | `String(format:)` without locale IS locale-dependent | None — plan correctly uses POSIX locale | — |
| 10 | GCD blocks with `~Copyable` captures (Appendix A.1) | **CORRECT** | `[captX = consume captX]` pattern works for `~Copyable` values in escaping closures | None | — |

### Additional SE Proposal Notes

| SE Proposal | Plan References | Status in Swift 6.2 | Impact |
|-------------|----------------|---------------------|--------|
| SE-0474 | Mentioned as future replacement for `_read`/`_modify` | **STABLE** — available now | **Use it.** Replace all `_read`/`_modify` with `yielding borrow`/`yielding mutate` |
| SE-0507 | Mentioned as tracked, not yet stable | **Still under review** — not available | Plan correctly notes this. SE-0474 covers the needed functionality |
| SE-0390 | Implicit (noncopyable types) | Stable since 5.9 | No issue |
| SE-0453 | InlineArray | Stable, but lacking Collection conformance | Plan correctly works around this |

---

## Step 3: Numerical Correctness

### Phase 1 Tolerance

| Metric | Plan Specifies | Correct Value | Issue |
|--------|---------------|---------------|-------|
| Identity kernel fp16 roundtrip | 1e-4 | **1e-2** | fp16 has only 10 mantissa bits; values outside [0.5, 1.0] can have errors > 1e-4. Phase 2 and 5 correctly use 1e-2. Phase 1 should match. |

The identity kernel performs fp32→fp16→fp32 roundtrip. For a value like `3.14159`, fp16 representation introduces error of approximately `3.14159 - 3.140625 = 0.000965`, which passes 1e-2 but may fail 1e-4. Larger values (e.g., embedding weights in range [-1, 1] with fine detail) can also exceed 1e-4.

### Phase 3 Exact Match
Correct. MIL text is compiled by ANE as a string — any character difference causes compile failure. Exact character match is the right tolerance.

### Phase 4 Gradient Check (1e-3)
Appropriate. Finite difference gradient checking with `h=1e-4` on single-precision floats typically achieves relative error of 1e-3 to 1e-4. The plan's tolerance is conservative and correct.

### Phase 6 End-to-End Tolerances
| Metric | Tolerance | Assessment |
|--------|-----------|------------|
| Loss match vs ObjC | 0.01 | Reasonable — fp16 quantization across 12 layers compounds |
| Gradient norm match | 5% relative | Reasonable — backward pass has more fp16 roundtrips |
| Performance | ≤ 9.3 ms/step | Correct — this is the current ObjC benchmark on M4 |

### Missing: Gradient Clipping Verification
`backward.h:121-150` implements `model_clip_gradients()` with L2 norm clipping. The plan does not mention gradient clipping in any phase or test. However, inspecting `train_large.m`, gradient clipping is NOT called in the production loop — it exists only in the `backward.h` reference implementation. **No fix needed** unless gradient clipping is desired for the Swift version.

---

## Step 4: C Interop Coverage

### Proposed API vs Required Functions

| ObjC Function | In Proposed C API | Used By | Verdict |
|---------------|-------------------|---------|---------|
| `ane_init()` / dlopen+NSClassFromString | `ane_interop_init()` | stories_config.h:117 | COVERED |
| `make_surface()` | `ane_interop_create_surface()` | stories_io.h:6 | COVERED |
| `compile_kern_mil_w()` | `ane_interop_compile()` | stories_io.h:87 | COVERED |
| `ane_eval()` | `ane_interop_eval()` | stories_io.h:131 | COVERED |
| `free_kern()` | `ane_interop_free()` | stories_io.h:122 | COVERED |
| `cvt_f32_f16()` | `ane_interop_cvt_f32_to_f16()` | stories_io.h:50 | COVERED |
| `cvt_f16_f32()` | `ane_interop_cvt_f16_to_f32()` | stories_io.h:41 | COVERED |
| `io_write_fp16()` | Reconstructible in Swift | stories_io.h:61 | OK — uses IOSurfaceLock + cvt |
| `io_read_fp16()` | Reconstructible in Swift | stories_io.h:66 | OK — uses IOSurfaceLock + cvt |
| **`io_copy()`** | **NOT IN API** | stories_io.h:71, train_large.m:471,510,513,514,537,538 | **CRITICAL** |
| **`io_write_fp16_at()`** | **NOT IN API** | stories_io.h:80, train_large.m:470,511 | **CRITICAL** |
| `g_compile_count` | `ane_interop_compile_count()` | stories_config.h:115 | COVERED |
| IOSurface I/O index access | `ane_interop_get_input/output()` | implicit | COVERED |

### io_copy Analysis

`io_copy` performs a direct `memcpy` between two IOSurface addresses at specified channel offsets — staying entirely in fp16 space. It is called **6 times per layer per step** in the backward pass:

1. `io_copy(ffnBwd->ioIn, DIM, fwdFFN->ioOut, DIM, 2*HIDDEN, SEQ)` — copy h1,h3 activations
2. `io_copy(sdpaBwd1->ioIn, 0, fwdAttn->ioOut, DIM, 3*DIM, SEQ)` — copy Q,K,V
3. `io_copy(sdpaBwd2->ioIn, 0, sdpaBwd1->ioOut, DIM, 2*SCORE_CH, SEQ)` — copy dscores
4. `io_copy(sdpaBwd2->ioIn, 2*SCORE_CH, fwdAttn->ioOut, DIM, 2*DIM, SEQ)` — copy Q,K
5. `io_copy(qkvBwd->ioIn, 0, sdpaBwd2->ioOut, 0, 2*DIM, SEQ)` — copy dq,dk
6. `io_copy(qkvBwd->ioIn, 2*DIM, sdpaBwd1->ioOut, 0, DIM, SEQ)` — copy dv

That's **72 io_copy calls per step** (6 per layer × 12 layers). This is a **hot path** function.

### io_write_fp16_at Analysis

`io_write_fp16_at` writes fp32 data to an IOSurface at a specific channel offset (not from offset 0). Used **2 times per layer per step** in the backward pass:

1. `io_write_fp16_at(ffnBwd->ioIn, 0, dffn, DIM, SEQ)` — write dffn at offset 0
2. `io_write_fp16_at(sdpaBwd1->ioIn, 3*DIM, dx2, DIM, SEQ)` — write dx2 at offset 3*DIM

That's **24 calls per step**. This is an offset variant of `io_write_fp16` and can be implemented in Swift using `IOSurfaceLock` + `cvt_f32_to_f16` at a pointer offset.

### NSDictionary Weight Reconstruction

The proposed API uses flat arrays (`weightPaths`, `weightDatas`, `weightLens`, `weightCount`) instead of `NSDictionary`. This is correct — the C shim can reconstruct the NSDictionary internally from the flat arrays. The plan handles this correctly.

---

## Step 5: Performance Risk Assessment

| Risk | Mechanism | Impact Estimate | Mitigation |
|------|-----------|-----------------|------------|
| `~Copyable` ownership overhead | Compile-time only; zero runtime cost at `-O` | **None** | N/A |
| Swift function call overhead | Inlines at `-O`; same codegen as C for leaf functions | **None** | Verify with `-emit-sil` for critical paths |
| GCD dispatch pattern | Identical to ObjC (serial queue + group) | **None** | Plan correctly chose GCD over actors |
| `io_copy` in Swift vs C | Swift IOSurface calls cross module boundary; not auto-inlineable | **LOW** (~0.1ms/step) | Implement `io_copy` in C shim alongside existing NEON functions, or in Swift with `@inlinable` |
| IOSurfaceLock/Unlock frequency | Same call count as ObjC | **None** | N/A |
| cblas_sgemm from Swift | Identical ABI; `import Accelerate` has zero overhead | **None** | N/A |
| vDSP calls from Swift | Same as cblas — zero overhead | **None** | N/A |
| String(format:locale:) in MIL gen | Only at compile time, not in training loop | **None** | N/A |
| exec() restart | Identical pattern — `execl` is POSIX, works from Swift | **None** | Plan correctly uses `Darwin.execl` |
| Potential ARC retain/release of IOSurfaceRef | Swift may insert ARC ops when passing IOSurfaceRef | **LOW** (~0.05ms/step) | Use `Unmanaged<IOSurfaceRef>` if profiling shows overhead |

**Overall assessment**: The performance risk is **very low**. The main concerns are `io_copy` (if implemented in Swift rather than C) and potential ARC overhead on IOSurfaceRef. Neither is likely to breach the 9.3ms/step budget, but both should be profiled during Phase 5.

---

## Step 6: Test Plan Gaps

### Missing Tests

| # | Gap | Severity | Phase | Proposed Test |
|---|-----|----------|-------|---------------|
| 1 | `io_copy` not tested (because it's not in the API) | **HIGH** | 1 or 2 | `test_io_copy_fp16_roundtrip` — copy between surfaces at offset, verify data integrity |
| 2 | `io_write_fp16_at` not tested | **HIGH** | 2 | `test_io_write_fp16_at_offset` — write at channel offset, verify fp16 data at correct position |
| 3 | Locale test may not work as written | **MEDIUM** | 3 | Plan's `test_mil_generation_locale_invariant` sets `Locale.current` but `Locale.current` is read-only. Must override via `ProcessInfo` environment or validate at `String(format:locale:)` call sites |
| 4 | No integration test for backward pass IOSurface data flow | **MEDIUM** | 5 | `test_backward_iosurface_copy_chain` — verify data flows correctly through io_copy chain |
| 5 | No test for `exec()` restart checkpoint roundtrip | **MEDIUM** | 6 | `test_exec_restart_preserves_checkpoint` — save checkpoint, simulate restart, verify training resumes from correct state |
| 6 | sdpaBwd2 kernel lifecycle not tested | **HIGH** | 5 | `test_sdpa_bwd2_compile_once_reuse` — compile once, eval with multiple inputs, verify output |
| 7 | Gradient accumulation with scaling factor | **MEDIUM** | 6 | Plan has `test_gradient_accumulation_averages` but should verify the `1.0/steps_batch` scaling from train_large.m:601 |

### Existing Test Assets Usage

The plan correctly identifies 10 existing test executables as external oracles. The strategy of running ObjC tests to generate expected outputs is sound. However, the plan should explicitly list which existing tests validate which Swift phases:

| ObjC Test | Swift Phase Validated |
|-----------|---------------------|
| `test_weight_reload.m` | Phase 5 (ModelWeightLoader) |
| `test_ane_advanced.m` | Phase 5 (ANEKernel lifecycle) |
| `test_ane_sdpa5.m` | Phase 5 (SDPA equivalence) |
| `test_full_fused.m` | Phase 5-6 (end-to-end fused kernel) |
| `test_fused_qkv.m` | Phase 3 (QKV blob offsets) |
| `test_fused_bwd.m` | Phase 5-6 (backward kernels) |
| `test_perf_stats.m` | Phase 6 (benchmark 9.3ms target) |
| `tiny_train.m` | Phase 6 (simpler golden reference) |

---

## Step 7: Unaddressed Risks

### 7.1 sdpaBwd2 Kernel Lifecycle (HIGH)

The plan's `LayerKernelSet` (Phase 5) owns 5 kernels: `fwdAttn`, `fwdFFN`, `ffnBwd`, `sdpaBwd1`, `qkvBwd`. But `train_large.m` also uses `sdpaBwd2` — a **weight-free** kernel compiled once per layer (line 303-307), stored in a separate `Kern *sdpaBwd2[NLAYERS]` array, and freed separately at exec-restart time (line 323).

The plan's Phase 5 description mentions `LayerKernelSet` owns "5 ANEKernel instances" but sdpaBwd2 is not accounted for. It has a different lifecycle:
- Compiled once at startup (no weights, so no recompilation needed)
- Shared across all accumulation steps
- Freed at exec-restart
- Re-compiled after exec-restart

**Fix**: Add a separate `StaticKernelSet` or `WeightFreeKernel` type in Phase 5, or add sdpaBwd2 to `LayerKernelSet` with a flag indicating it doesn't participate in weight-recompilation.

### 7.2 Architecture Confusion (HIGH)

The Key Files table lists `model.h`, `forward.h`, `backward.h` as being "rewritten in" Phases 5-6. But these files implement a **different architecture** (per-weight conv kernels with `ane_conv_eval`) than what `train_large.m` uses (fused kernels with direct IOSurface I/O).

The plan's Phase descriptions correctly describe the train_large.m architecture. But a developer following the Key Files table would attempt to port `model.h`'s `Model` struct (with per-layer `kern_q/kern_k/kern_v/kern_o` arrays) instead of `train_large.m`'s inline `LayerKernels` struct (with fused `fwdAttn/fwdFFN/ffnBwd/sdpaBwd1/qkvBwd`).

**Fix**: Update Key Files table. Replace `model.h → Phase 5` and `forward.h/backward.h → Phase 6` with:
- `model.h` → "Reference only — CPU ops ported in Phase 4, NOT the kernel architecture"
- `forward.h` → "Reference only — CPU ops (RoPE, SiLU, attention) ported in Phase 4"
- `backward.h` → "Reference only — CPU backward ops (rmsnorm_bwd, attention_bwd, rope_bwd, silu_bwd, gradient clipping) ported in Phase 4"
- Add: `train_large.m:300-350 (compile/free)` → Phase 5, `train_large.m:384-575 (forward/backward)` → Phase 6

### 7.3 Error Recovery Mid-Training (MEDIUM)

`train_large.m` has minimal error recovery: if a kernel compile fails, it sets `g_compile_count = MAX_COMPILES` to force an exec-restart (line 348). The plan doesn't address what happens if:
- An `ane_eval` call fails mid-forward-pass
- IOSurface allocation fails (system memory pressure)
- Checkpoint write fails (disk full)

In the ObjC code these silently corrupt state or crash. The Swift rewrite should define typed error handling for at least checkpoint save failures (which lose training progress).

### 7.4 CI Without ANE Hardware (MEDIUM)

The plan doesn't address CI testing. Phases 1, 5, and 6 require ANE hardware (`ane_interop_compile`, `ane_eval`). On CI machines without ANE:
- Phase 3 (MIL text generation) — testable without hardware (pure string operations)
- Phase 4 (CPU ops) — testable without hardware (pure Accelerate)
- Phase 2 (IOSurface I/O) — IOSurface is available on macOS even without ANE
- Phase 1 (ANE interop) — requires ANE; `dlopen` will fail

**Recommendation**: Tag ANE-dependent tests with a `#if` guard or test trait so CI can run the ~60% of tests that don't require hardware.

### 7.5 Checkpoint Migration (LOW)

The plan ensures binary compatibility with ObjC `CkptHdr` format via `validateLayout()`. But there's no mention of checkpoint version migration. Current checkpoint format is version 2 (stories_config.h:97). If the Swift rewrite needs to change the format (e.g., adding new fields), there should be a version check at load time.

### 7.6 Backward Pass Data Layout Assumptions (MEDIUM)

The backward pass relies heavily on specific channel offsets in IOSurface outputs. For example, `fwdAttn` output is `[o|Q|K|V|attn|xnorm] = [6*DIM, SEQ]` — the backward pass reads Q,K,V at offset `DIM` (line 510). If the MIL generator changes output channel ordering, the backward pass silently reads wrong data. The plan should add an integration test that verifies the full forward→backward IOSurface data flow through all 6 kernel types.

---

## Consolidated Fix List

| # | Severity | Phase | Fix Description |
|---|----------|-------|-----------------|
| 1 | **CRITICAL** | 1 | Add `io_copy(dst, dst_ch_off, src, src_ch_off, channels, spatial)` to `ane_interop.h` C API. This is called 72 times/step in the backward pass and operates directly on IOSurface fp16 data. |
| 2 | **CRITICAL** | 2 | Add `io_write_fp16_at(surface, ch_offset, data, channels, spatial)` to Swift `SurfaceIO` (can be implemented in Swift using IOSurfaceLock + cvt_f32_to_f16 at pointer offset). Add test. |
| 3 | **CRITICAL** | All | Replace all `_read`/`_modify` usage in `LayerStorage` with SE-0474 `yielding borrow`/`yielding mutate`. Remove the SE-0507 tracking comment. Update Appendix A.1. |
| 4 | **HIGH** | 5 | Add `sdpaBwd2` kernel lifecycle management. Either expand `LayerKernelSet` to 6 kernels with a `static` flag, or create a separate `StaticKernel` type compiled once and shared. |
| 5 | **HIGH** | Plan | Fix Key Files table: `model.h`/`forward.h`/`backward.h` should be marked "Reference only" with CPU ops ported in Phase 4. Add line ranges from `train_large.m` as the actual targets for Phases 5-6. |
| 6 | **HIGH** | 1 | Fix Phase 1 tolerance: change "identity kernel correct within 1e-4" to "within 1e-2" to match fp16 precision. 1e-4 is achievable only for values near 1.0; general floats need 1e-2. |
| 7 | **HIGH** | 2 | Add tests for `io_copy` and `io_write_fp16_at` once they exist in the API. |
| 8 | **MEDIUM** | 5 | Add integration test for backward pass IOSurface data flow through the full forward→backward kernel chain. Verify channel offsets match between MIL generators and IO read/write calls. |
| 9 | **MEDIUM** | 3 | Fix locale test: `Locale.current` is read-only in Swift. Either use `setenv("LC_ALL", "de_DE", 1)` before process launch, or validate locale-safety by checking that all format calls explicitly pass the `locale:` parameter (linter/grep check). |
| 10 | **MEDIUM** | 5 | Add typed error handling for ANE eval failures and IOSurface allocation failures in `ANEKernel` and `SurfaceIO`. At minimum, `ane_interop_eval` should return `bool` (which it does) and Swift should handle `false`. |
| 11 | **MEDIUM** | 6 | Document exec-restart lifecycle: which resources must be freed before `execl`, which are cleaned up by the OS, and verify checkpoint save completes before exec. |
| 12 | **LOW** | All | Add CI test tagging: mark ANE-dependent tests so CI without hardware can still run ~60% of the test suite (Phases 2-4 pure logic tests). |
