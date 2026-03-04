# Phase 6 Implementation Todolist

> **Usage**: This is your implementation roadmap. Work through items IN ORDER. Mark each `[x]` as you complete it.
> Do NOT skip ahead — earlier items create the foundations that later items depend on.
> After each GROUP, run the verification gate before proceeding.

---

## Group 0: Pre-Flight Checks
- [x] Read `tasks/phase6-prompt.md` fully — understand every constraint before writing code
- [x] Run `swift build` to confirm baseline compiles clean
- [x] Run `swift test` to confirm baseline passes (expect: 111 executed, 20 skipped, 0 failures)
- [x] Read `Sources/Espresso/Espresso.swift` — note it's a placeholder `public enum Espresso {}`; you will replace it
- [x] Read `Sources/EspressoTrain/main.swift` — note it's a placeholder stub; you will replace it
- [x] Read `Tests/EspressoTests/PlaceholderTests.swift` — note it's empty; you will replace it

**Gate 0**: Baseline green. You understand the full prompt. Proceed.

---

## Group 1: Leaf Types (No Dependencies on Other New Files)

### 1Z: LayerStorage Throwing Initializer (Phase 6 Support)
- [x] Add a throwing initializer to `Sources/ANETypes/LayerStorage.swift` so we can store throw-initialized `StaticKernel`/`LayerKernelSet` without unsafe pointer plumbing
- [x] Add an ANETypes test verifying partial initialization is cleaned up on throw
- [x] Run `swift test --filter ANETypesTests` — new LayerStorage test passes

### 1A: GradientAccumulator
- [x] Write `test_gradient_accumulator_enqueue_and_barrier` in `Tests/EspressoTests/EspressoTests.swift`
  - Enqueue 3 blocks that increment a counter, call barrier, assert counter == 3
- [x] Write `test_gradient_accumulator_wait_all` — enqueue 5 blocks, waitAll, verify all completed
- [x] Implement `Sources/Espresso/GradientAccumulator.swift`
  - `final class GradientAccumulator: @unchecked Sendable`
  - Wraps `DispatchQueue(label: "dw_cblas", attributes: [])` (serial) + `DispatchGroup`
  - Methods: `enqueue(_ block: @escaping @Sendable () -> Void)`, `barrier()`, `waitAll()`
  - `barrier()` calls `group.wait(timeout: .distantFuture)` — maps to `dispatch_group_wait(dw_grp, DISPATCH_TIME_FOREVER)` at train_large.m:392,578,601
  - `enqueue` calls `group.enter()`, `queue.async { block(); group.leave() }` — maps to `dispatch_group_async(dw_grp, dw_q, ^{ ... })` at train_large.m:448,483,503,526
- [x] Run `swift test --filter EspressoTests` — GradientAccumulator tests pass

### 1B: TokenDataset
- [x] Write `test_token_dataset_small_file` — create temp file with known UInt16 values, open, verify access
- [x] Write `test_token_dataset_validates_minimum_size` — file with < seqLen+1 tokens should fail
- [x] Implement `Sources/Espresso/TokenDataset.swift`
  - `struct TokenDataset: ~Copyable`
  - `init(path: String) throws` — `open()` + `fstat()` + `mmap(PROT_READ, MAP_PRIVATE)` via `import Darwin`
  - Properties: `nTokens: Int`, `subscript(offset: Int) -> UnsafePointer<UInt16>` for slice access
  - `deinit` — `munmap()` + `close(fd)`
  - Maps to train_large.m:274-281
- [x] Run `swift test --filter EspressoTests` — TokenDataset tests pass

### 1C: Sampler
- [x] Write `test_sampler_deterministic_sequence` — seed with 42, call 10 times, verify exact reproducible sequence
- [x] Write `test_sampler_range_valid` — all positions in `[0, nTokens - seqLen - 1)`
- [x] Implement `Sources/Espresso/Sampler.swift`
  - `enum Sampler`
  - `static func seed(startStep: Int)` — calls `srand48(42 + startStep)` (maps to train_large.m:317)
  - `static func samplePosition(maxPos: Int) -> Int` — `Int(drand48() * Double(maxPos))` (maps to train_large.m:374-376)
- [x] Run `swift test --filter EspressoTests` — Sampler tests pass

### 1D: ExecRestart
- [x] Write `test_exec_restart_formats_message` — verify log message format (no actual exec)
- [x] Implement `Sources/Espresso/ExecRestart.swift`
  - `enum ExecRestart`
  - `static func restart(step: Int, compileCount: Int, loss: Float) -> Never`
  - `fflush(stdout); fflush(stderr)` before print
  - Use `CommandLine.arguments[0]` for binary path
  - `execl(path, path, "--resume", nil)` via `withCString`
  - Maps to train_large.m:322-333
- [x] Run `swift test --filter EspressoTests` — ExecRestart test passes

### 1E: Checkpoint
- [x] Write `test_checkpoint_save_load_roundtrip` — create tiny synthetic weights, save, load back, verify byte-identical
- [x] Write `test_checkpoint_segment_order_small` — tiny config (dim=4, hidden=8, nLayers=1, vocab=10), verify binary offsets
- [x] Write `test_checkpoint_header_validation` — corrupt magic or version, verify load fails
- [x] Implement `Sources/Espresso/Checkpoint.swift`
  - `enum Checkpoint`
  - `struct CheckpointMeta` — step/totalSteps/lr/loss/cum*/adamT
  - Production API avoids huge extra allocations:
    - `static func save(path:meta:layers:layerAdam:rmsFinal:adamRmsFinal:embed:adamEmbed:) throws`
    - `static func load(path:intoLayers:intoLayerAdam:intoRmsFinal:intoAdamRmsFinal:intoEmbed:intoAdamEmbed:) throws -> CheckpointMeta`
  - Segment order MUST match train_large.m:110-181 EXACTLY:
    - Header (96 bytes)
    - Per-layer L=0..<nLayers: Wq,Wk,Wv,Wo,W1,W2,W3,rmsAtt,rmsFfn, then Adam m/v pairs
    - Global: rmsFinal, arms.m, arms.v, embed, aembed.m, aembed.v
  - **CRITICAL**: Checkpoint format writes per-layer (all of layer L, then L+1). Pretrained format writes per-type (all Wq, then all Wk). These are DIFFERENT.
- [x] Run `swift test --filter EspressoTests` — Checkpoint tests pass

**Gate 1**: All leaf types compile, tests pass. Run `swift build && swift test --filter EspressoTests`.

---

## Group 2: Forward Pass

### 2A: ForwardPass Implementation
- [x] Write `test_forward_single_layer_output_nonzero_finite` — compile 1 layer's kernels with random weights, run forward, verify output has no NaN/Inf and is nonzero
- [x] Write `test_forward_12_layers_no_nan` — full 12-layer forward, verify all activations finite
- [x] Implement `Sources/Espresso/ForwardPass.swift`
  - `enum ForwardPass`
  - `static func run(xCur:acts:kernels:accumulator:dim:seqLen:) throws(ANEError)` — iterates L=0..<nLayers
  - Per-layer sequence (maps to train_large.m:384-420):
    1. `memcpy(acts[L].layerIn ← xCur)` — save for RMSNorm backward
    2. `accumulator.barrier()` — wait for prior layer's async dW (maps to line 392)
    3. `SurfaceIO.writeFP16(to: kernels[L].fwdAttn.inputSurface(at:0), data: xCur, channels: dim, spatial: seqLen)`
    4. `try kernels[L].fwdAttn.eval()`
    5. Read fwdAttn output → oOut (offset=0, dim), attnOut (offset=4*dim, dim), xnorm (offset=5*dim, dim)
    6. `vDSP_vadd(xCur, 1, oOut, 1, x2, 1, count)` — residual
    7. `SurfaceIO.writeFP16(to: kernels[L].fwdFFN.inputSurface(at:0), data: x2, channels: dim, spatial: seqLen)`
    8. `try kernels[L].fwdFFN.eval()`
    9. Read fwdFFN output → ffnOut (0, dim), h1 (dim, hidden), h3 (dim+hidden, hidden), siluOut (dim+2*hidden, hidden), x2norm (dim+3*hidden, dim)
    10. `vDSP_vadd(x2, 1, ffnOut, 1, xCur, 1, count)` — residual, becomes next layer input
  - **CRITICAL channel offsets**: fwdAttn output is [o|Q|K|V|attn|xnorm] = [6*DIM, SEQ]. Note: Q(offset=DIM), K(offset=2*DIM), V(offset=3*DIM) are NOT read in forward but ARE needed in backward (preserved on IOSurface).
- [x] Run `swift test --filter EspressoTests` — Forward tests pass

**Gate 2**: Forward pass compiles and produces finite non-zero output. Run `swift test --filter EspressoTests`.

---

## Group 3: Backward Pass

### 3A: SendableBuffer helpers (if not already defined)
- [x] Define `SendableBuffer: ~Copyable, @unchecked Sendable` in `Sources/Espresso/GradientAccumulator.swift` (or its own file)
  - `init(copying source: UnsafePointer<Float>, count: Int)` — heap alloc + memcpy
  - `deinit` — deallocate
  - Maps to train_large.m:478-482 (malloc+memcpy pattern)
- [x] Define `SendablePointer: @unchecked Sendable` — wraps `UnsafeMutablePointer<Float>` for gradient accumulator captures
- [x] Define `SendableConstPointer: @unchecked Sendable` — wraps `UnsafePointer<Float>` for read-only captures (dembed block)

### 3B: BackwardPass Implementation
- [x] Write `test_backward_produces_nonzero_gradients` — run forward+backward, verify gradient buffers are nonzero
- [x] Write `test_backward_residual_gradient_flow` — verify dy propagates through both skip connections: dy_prev = dx_rms1 + dx2
- [x] Implement `Sources/Espresso/BackwardPass.swift`
  - `enum BackwardPass`
  - `static func run(dy:acts:kernels:staticKernels:grads:weights:accumulator:dim:hidden:seqLen:heads:) throws(ANEError)`
  - Iterates L = (nLayers-1) down to 0 (maps to train_large.m:461-575)
  - Per-layer backward sequence (15 IOSurface I/O operations):
    1. **FFN backward (ANE)**: Write dffn to ffnBwd input at offset 0; copy h1|h3 from fwdFFN output to ffnBwd input at DIM; eval; read dx_ffn, dh1, dh3
    2. **Async dW FFN**: Capture dffn, siluOut, dh1, dh3, x2norm as SendableBuffers → enqueue cblas: dW2 += dffn @ silu^T, dW1 += dh1 @ x2norm^T, dW3 += dh3 @ x2norm^T
    3. **RMSNorm2 backward (CPU)**: `RMSNorm.backward(dx: dx2, dw: grads[L].rmsFfn, dy: dxFfn, x: acts[L].x2, weights: weights[L].rmsFfn)` then `dx2 += dy` (residual)
    4. **Async dWo**: Capture dx2, attnOut → enqueue cblas: dWo += dx2 @ attnOut^T
    5. **SDPA backward (ANE, 2 kernels)**:
       - sdpaBwd1: copy Q|K|V from fwdAttn output(offset=DIM, 3*DIM) to sdpaBwd1 input; write dx2 at offset 3*DIM; eval
       - sdpaBwd2: copy dscores from sdpaBwd1 output(offset=DIM, 2*SCORE_CH) to sdpaBwd2 input; copy Q|K from fwdAttn output(offset=DIM, 2*DIM) to sdpaBwd2 at 2*SCORE_CH; eval
       - Read: dq from sdpaBwd2 out(0), dk from sdpaBwd2 out(DIM), dv from sdpaBwd1 out(0)
    6. **Async dWq/dWk/dWv**: Capture dq, dk, dv, xnorm → enqueue cblas: dWq,dWk,dWv += d* @ xnorm^T
    7. **QKV backward (ANE)**: copy dq|dk from sdpaBwd2 out(0, 2*DIM) to qkvBwd input; copy dv from sdpaBwd1 out(0, DIM) to qkvBwd at 2*DIM; eval; read dx_attn
    8. **RMSNorm1 backward (CPU)**: `RMSNorm.backward(dx: dxRms1, dw: grads[L].rmsAtt, dy: dxAttn, x: acts[L].layerIn, weights: weights[L].rmsAtt)`
    9. **Propagate gradient**: `dy[i] = dxRms1[i] + dx2[i]` for all i — maps to train_large.m:573

  - **CRITICAL**: After the layer loop, call `accumulator.barrier()` before `Embedding.backward()` (maps to line 578)
  - **CRITICAL**: The 4th IOSurface copy for sdpaBwd1 (step 5a) copies from fwdAttn.ioOut at src offset DIM, NOT offset 0. The fwdAttn output layout is [o|Q|K|V|attn|xnorm], so Q starts at offset DIM.
- [x] Run `swift test --filter EspressoTests` — Backward tests pass

**Gate 3**: Backward pass produces nonzero gradients with correct residual flow. Run `swift test --filter EspressoTests`.

---

## Group 4: Training Loop (main.swift)

### 4A: Gradient Accumulation + Adam Tests
- [x] Write `test_gradient_accumulation_averages` — run 2 accumulation steps, verify gradients are scaled by 1/2 before Adam
- [x] Write `test_gradient_accumulation_scaling` — verify `1.0/steps_batch` scaling applied to ALL 9 per-layer gradient buffers + rmsFinal + embed gradients before Adam update

### 4B: Integration Tests
- [x] Write `test_single_step_loss_matches_objc` — load same pretrained weights, same input tokens, verify |swift_loss - objc_loss| < 0.01 (gated by ANE hardware + data file)
- [x] Write `test_10_steps_loss_decreases` — verify loss[step 10] < loss[step 0] (overfit scenario)
- [x] Write `test_1_step_gradients_match_objc` — per-layer gradient L2 norm relative error < 5%
- [x] Write `test_checkpoint_binary_compatible_with_objc` — load ObjC-generated checkpoint in Swift, verify weights match
- [x] Write `test_exec_restart_checkpoint_roundtrip` — save at step N, simulate --resume load, verify state identical

### 4C: main.swift Implementation
- [x] Remove placeholder stub from `Sources/EspressoTrain/main.swift`
- [x] Remove placeholder from `Sources/Espresso/Espresso.swift` (or keep as namespace if needed)
- [x] Remove placeholder from `Tests/EspressoTests/PlaceholderTests.swift`
- [x] Implement `Sources/EspressoTrain/main.swift` entry point:
  - [x] Call `CheckpointHeader.validateLayout()` at startup
  - [x] Parse args: `--resume`, `--steps N`, `--lr F` (maps to train_large.m:196-201)
  - [x] Allocate per-layer state: `LayerStorage<LayerWeights>`, `LayerStorage<LayerAdam>`, `LayerStorage<LayerActivations>`, `LayerStorage<LayerGradients>` (maps to lines 203-215)
  - [x] Allocate globals: rmsFinal, embed, grms_final, gembed, adamRmsFinal, adamEmbed (maps to lines 217-223)
  - [x] Resume or load pretrained weights (maps to lines 230-254)
  - [x] Open TokenDataset (maps to lines 274-281)
  - [x] Allocate scratch buffers: dy, dffn, dh1, dh3, dx_ffn, dx2, dq, dk, dv, dx_attn, x_cur, x_final, logits, dlogits (maps to lines 284-300). NOTE: The ObjC `do_out_buf` is unnecessary in Swift — `SendableBuffer` copies directly from dx2.
  - [x] Compile static sdpaBwd2 kernels: `LayerStorage<StaticKernel>` (maps to lines 302-307)
  - [x] Create GradientAccumulator (maps to lines 309-310)
  - [x] Seed sampler: `Sampler.seed(startStep: startStep)` (maps to line 317)
  - [x] **Outer training loop** `while step < totalSteps` (maps to line 320):
    - [x] Check compile budget → exec restart if exhausted (maps to lines 322-333)
    - [x] Compile all LayerKernelSets (maps to lines 336-356)
    - [x] Zero gradient accumulators (maps to lines 362-365)
    - [x] **Inner accumulation loop** `for a in 0..<ACCUM_STEPS where step < totalSteps` (maps to line 371):
      - [x] Sample position, get input/target tokens (maps to lines 374-377)
      - [x] Embedding lookup → x_cur (maps to lines 380-382)
      - [x] ForwardPass.run() — 12 layers (maps to lines 384-420)
      - [x] Final RMSNorm: `RMSNorm.forward(xFinal, xCur, rmsFinal, dim, seqLen)` (maps to lines 423-425)
      - [x] Classifier: `cblas_sgemm` embed @ xFinal → logits (maps to lines 428-431)
      - [x] Cross-entropy loss (maps to lines 434-436)
      - [x] Classifier backward: `cblas_sgemm` embedᵀ @ dlogits → dy; async dembed += dlogits @ xFinalᵀ (maps to lines 443-452)
      - [x] Final RMSNorm backward (maps to lines 455-458)
      - [x] BackwardPass.run() — 12 layers reverse (maps to lines 461-575)
      - [x] Embedding backward: `accumulator.barrier()` then `Embedding.backward()` (maps to lines 578-579)
      - [x] Print step telemetry (maps to lines 582-593)
    - [x] Wait all async dW: `accumulator.waitAll()` (maps to line 601)
    - [x] Scale gradients by `1.0/steps_batch` (maps to lines 604-628):
      - Per-layer: all 9 gradient buffers *= gsc
      - Global: grms_final *= gsc, gembed *= gsc
    - [x] Adam update: increment adam_t, update all per-layer weights + rmsFinal + embed (maps to lines 605-628)
    - [x] Print batch telemetry (maps to lines 630-647)
  - [x] Print efficiency report (maps to lines 650-667)
  - [x] Save final checkpoint
- [x] Run `swift build` — verify EspressoTrain compiles
- [x] Run `swift test --filter EspressoTests` — all Phase 6 tests pass

### 4D: Performance Test
- [x] Write `test_100_steps_benchmark` — 100 steps, verify <= 9.3 ms/step on M4 (gated by ANE hardware)

**Gate 4**: Full training loop compiles and runs. Integration tests pass.

---

## Group 5: Final Verification

- [x] Run `swift build` — clean compile, zero warnings
- [x] Run `swift test` — ALL tests pass (expect: 111 prior + Phase 6 new tests, with ANE-dependent tests skipped on non-ANE hosts)
- [x] Verify no regressions: prior 111 tests still pass
- [x] Review all new files against C source line references — spot-check 5 critical mappings:
  1. Checkpoint segment order matches train_large.m:125-144
  2. Forward fwdAttn output channel offsets match train_large.m:398-400
  3. Backward sdpaBwd1/sdpaBwd2 io_copy chain matches train_large.m:510-515
  4. Gradient scaling `1.0/steps_batch` applied to ALL buffers including rmsFinal and embed (train_large.m:604-628)
  5. dy propagation: `dy = dx_rms1 + dx2` at each layer (train_large.m:573)
- [x] Update `tasks/todo.md` — mark Phase 6 items complete
- [x] Update `tasks/lessons.md` — record any new patterns learned

**Gate 5**: All tests green. Code review complete. Phase 6 done.
