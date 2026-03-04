# Phase 6b: Kernel Numerical Equivalence & Hardware Verification

<role>
You are a senior Swift 6.2 systems engineer completing the final verification sub-phase of a bottom-up Swift rewrite of an ANE training codebase. Phases 1–6 are fully implemented (143 tests, 0 failures). Phase 6b fills the remaining test gaps and closes the hardware verification gates deferred from Phases 5 and 6.

You have deep expertise in:
- Swift 6.2 `~Copyable` ownership and strict concurrency
- IOSurface fp16 channel-offset addressing and copy chains
- ANE kernel compilation and eval via private API (ANEInterop shim)
- XCTest with tiered environment-variable gating
- ObjC clang build toolchain for cross-validation
</role>

---

<context>

## Implementation Roadmap

**You MUST follow `tasks/phase6b-todolist.md` step-by-step.** Mark items `[x]` as you complete them. Run verification gates at the end of each group.

Groups:
- **Group 0**: Pre-flight (baseline + hardware baseline)
- **Group 1**: 4 kernel numerical equivalence tests (new tests in ANERuntimeTests)
- **Group 2**: Backward IOSurface copy chain test (new test in EspressoTests)
- **Group 3**: Hardware verification gates (run integration + perf tests, document results)
- **Group 4**: Checklist closeout and lessons

---

## Why Phase 6b Exists

Phase 5b implemented `LayerKernelSet`, `StaticKernel`, and `ModelWeightLoader` and deferred 11 tests. Most were later covered by Phase 5b/6 tests under different names. Four remain genuinely uncovered:

| Deferred Test | Status | Reason |
|---|---|---|
| `test_compile_layer_kernels_with_random_weights` | Covered by `test_layer_kernel_set_compiles_all_five_and_surface_sizes` | Same coverage |
| `test_fwd_attn_output_has_6xdim_channels` | **MISSING** | No test reads actual output regions |
| `test_fwd_attn_numerical_equivalence_with_objc` | **MISSING** | No numerical comparison against ObjC |
| `test_fwd_ffn_numerical_equivalence_with_objc` | **MISSING** | No numerical comparison against ObjC |
| `test_ffn_bwd_numerical_equivalence_with_objc` | **MISSING** | No numerical comparison against ObjC |
| `test_model_weights_bin_layout_small` | Covered by `test_model_weight_loader_payload_layout_matches_llama2c_order_and_sizes` | Same coverage |
| `test_model_weights_vocab_sign_shared_vs_unshared` | Covered by `test_model_weight_loader_vocab_mismatch_fails_fast` | Same coverage |
| `test_load_stories110m_weights` | Covered by `test_load_stories110m_weights_integration` | Same coverage |
| `test_sdpa_bwd2_compile_once_reuse` | Covered by `test_static_kernel_eval_produces_output` | Same coverage |
| `test_sdpa_bwd2_lifecycle_independent` | Covered by `test_static_kernel_survives_layer_kernel_set_dealloc` | Same coverage |
| `test_backward_iosurface_copy_chain` | **MISSING** | No test verifies backward copy wiring |

Phase 6 hardware verification gates are also unchecked:
- Forward attn/FFN match ObjC within 1e-2 — unchecked
- Loss matches ObjC within 0.01, gradients within 5%, benchmark <= 9.3 ms/step — unchecked

Phase 6b closes exactly these gaps.

---

## Existing Test Infrastructure (Do Not Duplicate)

The following helpers already exist in `ANERuntimeTests.swift` — use them directly:

```swift
// Hardware gate
func requireANEHardwareTestsEnabled() throws {
    guard ProcessInfo.processInfo.environment["ANE_HARDWARE_TESTS"] == "1" else {
        throw XCTSkip("Requires ANE hardware (ANE_HARDWARE_TESTS=1)")
    }
}

// Weight filler
func fillLayerWeights(_ weights: LayerWeights, value: Float) { ... }

// Surface byte sizes (precomputed constants)
let dim = ModelConfig.dim          // 768
let hidden = ModelConfig.hidden    // 2048
let seqLen = ModelConfig.seqLen    // 256
let heads = ModelConfig.heads      // 12
let scoreCh = ModelConfig.scoreCh  // 3072 = heads * seqLen

let fwdAttnInputBytes: Int   // dim * seqLen * 2
let fwdAttnOutputBytes: Int  // 6 * dim * seqLen * 2
let fwdFFNInputBytes: Int    // dim * seqLen * 2
let fwdFFNOutputBytes: Int   // (dim + 3*hidden + dim) * seqLen * 2
let ffnBwdInputBytes: Int
let ffnBwdOutputBytes: Int
let sdpaBwd1InputBytes: Int  // (3*dim + dim) * seqLen * 2 = 4*dim*seqLen*2
let sdpaBwd1OutputBytes: Int // (dim + 2*scoreCh) * seqLen * 2
let sdpaBwd2InputBytes: Int  // (2*scoreCh + 2*dim) * seqLen * 2
let sdpaBwd2OutputBytes: Int // 2*dim * seqLen * 2
let qkvBwdInputBytes: Int    // 3*dim * seqLen * 2
let qkvBwdOutputBytes: Int   // dim * seqLen * 2
```

Existing tests use `value: 0.01` (constant fill) for kernel compilation — use the same for determinism.

---

## Cross-Validation Gating Pattern

Add `OBJC_CROSS_VALIDATION=1` as a new gate tier for tests that compare against ObjC golden values:

```swift
func requireObjCCrossValidation() throws {
    guard ProcessInfo.processInfo.environment["OBJC_CROSS_VALIDATION"] == "1" else {
        throw XCTSkip("ObjC cross-validation test (set OBJC_CROSS_VALIDATION=1)")
    }
}
```

Run order in CI: `swift test` → `ANE_HARDWARE_TESTS=1 swift test` → `ANE_HARDWARE_TESTS=1 OBJC_CROSS_VALIDATION=1 swift test`

</context>

---

<objc_reference>

## ObjC Test Executables Used in Phase 6b

Only 2 executables are needed for Phase 6b (full 10-executable cross-validation is Phase 7):

### test_full_fused.m (379 lines)
**Build**: `xcrun clang -O2 -Wno-deprecated-declarations -fobjc-arc -o training/test_full_fused training/test_full_fused.m -framework Foundation -framework CoreML -framework IOSurface -ldl`

**What it does**: Compiles the full fused SDPA attention kernel (QKV convs → reshape → matmul(Q,K^T) → scale+mask → softmax → matmul(scores,V) → Wo conv), writes a random input, evals, and prints output values.

**Key constants**: DIM=768, HEADS=12, HD=64, HIDDEN=2048, SEQ=64 (NOT 256)

**Cross-validation strategy**: Use `srand(42)` in ObjC to seed weights and input. Match the same seed/pattern in Swift. Compare fwdAttn output region 0 (oOut, DIM channels) within 1e-2.

**Caveat**: test_full_fused.m uses SEQ=64, not ModelConfig.seqLen=256. Create a temporary `ModelConfig`-like context for this test with `seqLen=64` OR allocate surfaces with the right byte count manually (seqLen=64 × dim × 2 bytes).

### test_fused_bwd.m (184 lines)
**Build**: `xcrun clang -O2 -Wno-deprecated-declarations -fobjc-arc -o training/test_fused_bwd training/test_fused_bwd.m -framework Foundation -framework CoreML -framework IOSurface -ldl`

**What it does**: Tests fused backward dx kernels (QKV backward: 3 transposed convs on slices; W1b+W3b backward). Verifies the backward can compute dx from dy through the fused kernel.

**Cross-validation strategy**: Same seed approach. Compare dx output (qkvBwd output) within 1e-2.

### tiny_train.m (593 lines)
**What it is**: A 2-layer linear model (`y = W2 @ relu(W1 @ x)`, MSE loss, SGD). **NOT** the Llama2 architecture. Do NOT compare loss values with `espresso-train`.

**Purpose in Phase 6b**: Verify the exec-restart + compile-budget-exhaustion contract is behaviorally consistent. Inspect source to confirm Swift's `ExecRestart.restartArgv` matches ObjC's `execl(argv[0], argv[0], "--resume", NULL)` contract:
- ObjC restart args: `[argv[0], "--resume"]` (no other args preserved — ObjC uses hardcoded #defines for all paths)
- Swift restart args: `[execPath] + filteredArgs + ["--resume"]` (preserves user-supplied args)
- The difference is intentional: ObjC uses hardcoded `CKPT_PATH`/`DATA_PATH`/`MODEL_PATH` so args don't need preserving. Swift uses CLI args. Both produce a `--resume` restart. This is correct behavior for each implementation.

</objc_reference>

---

<iosurface_layout>

## IOSurface Channel Layout Reference

**Channel offsets are in units of channels (not bytes). Bytes = channels × seqLen × 2 (fp16).**

### fwdAttn Output — `[6*DIM, SEQ]`
| Offset | Channels | Name | Used In |
|---|---|---|---|
| 0 | DIM | oOut | Residual add: `x2 = xCur + oOut` |
| DIM | DIM | Q | sdpaBwd1 input, sdpaBwd2 input |
| 2*DIM | DIM | K | sdpaBwd1 input, sdpaBwd2 input |
| 3*DIM | DIM | V | sdpaBwd1 input |
| 4*DIM | DIM | attnOut | Async dWo: `dWo += dx2 @ attnOut^T` |
| 5*DIM | DIM | xnorm | Async dWq/dWk/dWv and RMSNorm1 backward |

### sdpaBwd1 Input — `[4*DIM, SEQ]`
| Offset | Channels | Source | Operation |
|---|---|---|---|
| 0 | 3*DIM | fwdAttn output @ DIM | `io_copy(fwdAttn.out, srcOff=DIM, sdpaBwd1.in, dstOff=0, ch=3*DIM)` |
| 3*DIM | DIM | dx2 (CPU buffer) | `io_write_fp16(sdpaBwd1.in, off=3*DIM, dx2, ch=DIM)` |

### sdpaBwd1 Output — `[DIM + 2*SCORE_CH, SEQ]`
| Offset | Channels | Name | Used In |
|---|---|---|---|
| 0 | DIM | dv | qkvBwd input @ 2*DIM (NOT from sdpaBwd2) |
| DIM | 2*SCORE_CH | dscores | sdpaBwd2 input @ 0 |

### sdpaBwd2 Input — `[2*SCORE_CH + 2*DIM, SEQ]`
| Offset | Channels | Source |
|---|---|---|
| 0 | 2*SCORE_CH | sdpaBwd1 output @ DIM |
| 2*SCORE_CH | 2*DIM | fwdAttn output @ DIM (Q|K only) |

### sdpaBwd2 Output — `[2*DIM, SEQ]`
| Offset | Channels | Name | Used In |
|---|---|---|---|
| 0 | DIM | dq | qkvBwd input @ 0 |
| DIM | DIM | dk | qkvBwd input @ DIM |

### qkvBwd Input — `[3*DIM, SEQ]`
| Offset | Channels | Source |
|---|---|---|
| 0 | DIM | dq from sdpaBwd2 output @ 0 |
| DIM | DIM | dk from sdpaBwd2 output @ DIM |
| 2*DIM | DIM | dv from sdpaBwd1 output @ 0 |

**Critical invariant**: `dv` comes from `sdpaBwd1`, NOT `sdpaBwd2`. `dq`/`dk` come from `sdpaBwd2`.

</iosurface_layout>

---

<instructions>

## Implementation Rules

1. **Append to existing test files** — do NOT create new test targets. Phase 6b adds tests to `ANERuntimeTests.swift` and `EspressoTests.swift`.

2. **Use existing test helpers** — `requireANEHardwareTestsEnabled()`, `fillLayerWeights`, byte-size constants. Do not redefine them.

3. **Golden values are hardcoded constants** — for cross-validation tests, run the ObjC executable once, capture specific output values, hardcode them in the Swift test as `let expected: [Float] = [...]`. Tests must be deterministic without running ObjC at test time.

4. **SEQ mismatch** — `test_full_fused.m` uses SEQ=64 but `ModelConfig.seqLen=256`. When comparing against it, allocate surfaces manually with `seqLen=64` bytes, or create a local `let testSeqLen = 64` constant for that test. Do not change `ModelConfig.seqLen`.

5. **Tolerance is per-element max** — use `zip(swift, objc).max(by: { abs($0.0 - $0.1) < abs($1.0 - $1.1) })` to find and report the worst-case element difference.

6. **Document verification results** — after running integration and perf tests in Group 3, write actual values (ms/step, loss diff, gradient norm ratio) into `tasks/todo.md`.

7. **Gate tiers must compose** — `OBJC_CROSS_VALIDATION=1` tests ALSO require `ANE_HARDWARE_TESTS=1`. The test must check both gates.

8. **tiny_train.m is a different model** — Do NOT run it and compare loss with espresso-train. Use it only to verify the exec-restart behavioral contract by source inspection and unit testing `ExecRestart.restartArgv`.

9. **Never loosen performance target** — if `test_100_steps_benchmark` fails the 9.3ms gate, document the actual result and investigate. Do not change the assertion.

10. **TDD**: Write each test, verify it skips or fails, then make it pass.

</instructions>

---

<tests>

## Test Specifications (6 new tests total)

### In ANERuntimeTests.swift (4 tests)

#### `test_fwd_attn_output_has_6xdim_channels`
**Gate**: `ANE_HARDWARE_TESTS=1`
**Weights**: `fillLayerWeights(_, value: 0.01)`
**Input**: ramp `Float(i % 64 + 1) * 0.01` for `i in 0..<(dim*seqLen)`
**Steps**:
1. Compile LayerKernelSet
2. Write input to `fwdAttn.inputSurface(at:0)` — `dim` channels, `seqLen` spatial
3. `try kernels.fwdAttn.eval()`
4. Read 6 regions from `fwdAttn.outputSurface(at:0)` at offsets [0, DIM, 2*DIM, 3*DIM, 4*DIM, 5*DIM], each DIM channels
5. For each region: assert `allSatisfy(\.isFinite)` AND `contains(where: { $0 != 0 })`

#### `test_fwd_attn_numerical_equivalence_with_objc`
**Gate**: `ANE_HARDWARE_TESTS=1` + `OBJC_CROSS_VALIDATION=1`
**Steps**:
1. Run `./training/test_full_fused` (build first if needed), capture oOut values for a known seed
2. Hardcode those values as `let expectedOOut: [Float]` in the test
3. Compile LayerKernelSet with the same weight/input pattern (SEQ=64 for this test)
4. Eval fwdAttn, read oOut (offset 0, DIM channels, SEQ=64 spatial)
5. Assert `zip(actual, expectedOOut).map { abs($0 - $1) }.max()! < 1e-2`

#### `test_fwd_ffn_numerical_equivalence_with_objc`
**Gate**: same as above
**Steps**: Same pattern for fwdFFN. Feed oOut as input. Compare ffnOut (offset 0) against golden.

#### `test_ffn_bwd_numerical_equivalence_with_objc`
**Gate**: same as above
**Reference**: `./training/test_fused_bwd` output
**Steps**: Same pattern for ffnBwd. Compare dx output (qkvBwd output at offset 0) against golden.

### In EspressoTests.swift (2 tests)

#### `test_backward_iosurface_copy_chain`
**Gate**: `ANE_HARDWARE_TESTS=1`
**Steps**: (detailed in todolist Group 2A)
Compile 1 LayerKernelSet + 1 StaticKernel. Run 1-layer forward. Then verify 9 copy/eval steps in the backward chain by reading each destination surface after each operation. Assert the dv/dq/dk invariant explicitly.

#### `test_exec_restart_matches_objc_contract`
**Gate**: None (pure logic — no hardware needed)
**Steps**:
1. Call `ExecRestart.restartArgv(currentArguments: ["./espresso-train", "--steps", "100", "--lr", "0.0003"], resolvedExecPath: "/abs/path/espresso-train")`
2. Assert result == `["/abs/path/espresso-train", "--steps", "100", "--lr", "0.0003", "--resume"]`
3. Test idempotent: call with `--resume` already present → still exactly one `--resume` at end
4. Assert that tiny_train.m's ObjC restart `execl(argv[0], argv[0], "--resume", NULL)` is equivalent in behavior to Swift's `restartArgv` for the case where there are no path args (both produce `[execPath, "--resume"]`)

</tests>

---

<verification>

## Verification Checklist

| Check | Command | Expected |
|---|---|---|
| All non-hardware tests pass | `swift test` | 143+ executed, 0 failures |
| Hardware kernel tests pass | `ANE_HARDWARE_TESTS=1 swift test --filter ANERuntimeTests` | New tests green |
| Backward chain test passes | `ANE_HARDWARE_TESTS=1 swift test --filter test_backward_iosurface_copy_chain` | Pass |
| Cross-validation passes | `ANE_HARDWARE_TESTS=1 OBJC_CROSS_VALIDATION=1 swift test --filter ANERuntimeTests` | |
| Integration tests pass | `ANE_HARDWARE_TESTS=1 ESPRESSO_INTEGRATION_TESTS=1 swift test` | All 4 integration tests green |
| Compile time gate | `test_layer_kernel_set_compile_time_under_2000ms` | < 2000ms |
| Performance gate | `ANE_HARDWARE_TESTS=1 ESPRESSO_PERF_TESTS=1 swift test` | <= 9.3 ms/step |
| Todo updated | Check Phase 5 + Phase 6 unchecked items | All resolved |

</verification>
