# ANE Training Codebase: Swift 6.2 Rewrite

## Status: Phase 1–7 Implemented (Swift rewrite complete; ObjC archived as reference)

## ANE 10x Tuning Program (2026-03-05)
Goal: iterate aggressively until ANE direct is 10x faster than Core ML across compute-only + end-to-end benchmarks, or prove a hard floor via ANE `hwExecutionTime`.

- [ ] Group 0: Lock baselines (Core ML `.all`/`.cpuAndNeuralEngine`/`.cpuAndGPU` + ANE inference) for both benchmark modes
- [ ] Group 1: Add microsecond-level per-kernel profiler (attn/ffn: write/eval/read + gaps) with CSV output
- [ ] Group 2: Add `_ANEPerformanceStats` plumbing to split eval host time vs `hwExecutionTime`
- [ ] Group 3: Remove measurement noise (pre-resolve IOSurface handles; explicitly bucket surface lookup time)
- [ ] Group 4: Eliminate intermediate FP32 round-trips (surface-to-surface FP16 handoff attnOut -> ffnIn)
- [ ] Group 5: FP16-resident hidden-state mode (only boundary CPU I/O)
- [ ] Group 6: True zero-copy chaining (surface rebinding or `_ANEChainingRequest` loopback path)
- [ ] Group 7: Real-time eval path experiments (`evaluateRealTimeWithModel`, queueDepth, options sweeps)
- [ ] Group 8: Autotune harness to sweep combinations and keep best-known results
- [ ] Group 9: Publish evidence: updated perf report + “why 10x is/isn’t possible” analysis with `hwExecutionTime` floor

## ANE 10x Decode + KV Cache (2026-03-05)
Goal: land autoregressive decode with persistent FP16 KV-cache surfaces and prove >=10x tokens/sec vs fastest Core ML naive decode baseline.

- [x] Group 0: Add decode tests first (SurfaceIO slice edge-cases, decode state sequencing, hardware-gated decode correctness)
- [x] Group 1: Add decode MIL generators (`decode_attn_qkv`, `decode_ffn`) with byte-contract and structure tests
- [x] Group 2: Add runtime decode path (`DecodeKernelSet`, `DecodeSurfaceHandles`, `ForwardPass.runDecodeTimed`) with KV-cache + mask updates
- [x] Group 3: Add benchmark decode mode (`--decode`, `--decode-steps`, `--decode-max-seq`) and Core ML naive decode baseline
- [x] Group 4: Emit decode artifacts (`summary.txt`, `summary.json`, per-token latency CSVs, per-layer decode kernel profile CSV)
- [x] Group 5: Extend prefill profiling with host-vs-hw overhead splits and summary table
- [x] Group 6: Verify and capture reproducible evidence (build/tests + benchmark reruns within ±2%)

### Decode Phase Review (2026-03-05)
- Decode kernels compile (`decode_attn_qkv` + `decode_ffn`) and runtime wiring/CSV/reporting are integrated.
- Prefill profiling now reports host vs `hwExecutionTime`, host overhead, and lock/body/unlock IO splits per kernel/layer in `summary.txt`, CSV, and `summary.json`.
- Blocker: decode attention kernel still fails at first ANE eval on this host with:
  - `statusType=0x9: Program Inference error`
  - Repro: `ESPRESSO_BENCH_SEED=1 .build/release/espresso-bench --decode --ane-only --warmup 0 --iterations 1 --decode-steps 1 --decode-max-seq 16 --output /tmp/decode_restored_smoke`
- Additional hard-proof diagnostics completed:
  - Decode FFN eval fails for lane spatial `< 32` and succeeds for `>= 32` on this host (same `statusType=0x9` failure mode below threshold).
  - Decode lane-pack runtime path now supports configurable lane width (`ESPRESSO_DECODE_LANE_SPATIAL`, clamped to default `32`) and zero-copy lane packing/unpacking via `copyFP16SpatialSlice`.
  - `_ANERequest` constructor path now includes `weightsBuffer` selector fallback in `ane_interop.m` (no change to decode failure signature).
  - Even with a probe decode-attn MIL that bypasses softmax/matmul attention math, decode-attn still fails at first eval (`statusType=0x9`), indicating the blocker is below high-level attention logic and likely in a lower-level ANE decode-kernel compatibility constraint.
- 2026-03-05 isolation update:
  - Added a hardware decode probe matrix (`ANERuntimeTests.test_decode_probe_passthrough_4in_3out_eval`) and verified this host only evaluates decode probes when non-token inputs stay in a narrow shape family (`x`, `x+k`, `x+k+v`, and `x+k+v+denseMask` at `maxSeq=32`).
  - Hard constraints observed on this host: decode-attn kernels fail with `statusType=0x9` when using channel-1 mask inputs, concatenated `2*dim` cache inputs, or `maxSeq > 32` for auxiliary decode inputs.
  - Implemented decode-attn IO contract shift to dense mask cache (`maskCache: [1, dim, 1, maxSeq]`) and kept KV as separate inputs; decode runtime now executes reliably for `decode-max-seq=32`.
  - Added explicit decode guardrails: current ANE decode path requires `decode-max-seq == decode lane spatial == 32` (fail-fast error otherwise).
  - Added hardware-gated decode state correctness test (`InferenceOptimizationTests.test_decode_kv_cache_updates_and_mask_progresses_on_hardware`) validating KV slice writes and mask progression.
- 2026-03-05 benchmark snapshot (current path):
  - `ESPRESSO_BENCH_SEED=1 .build/release/espresso-bench --decode --profile-kernels --warmup 50 --iterations 500 --decode-steps 32 --decode-max-seq 32 --output /tmp/decode_profile_ane10x_20260305`
  - ANE decode: mean `0.651 ms/token`, median `0.629 ms/token`, `1535.8 tok/s`
  - Core ML naive decode fastest median (`.cpuAndNeuralEngine`): `1.711 ms/token`
  - Speedup vs fastest Core ML naive decode: `2.72x` (strict gate at `maxSeq=32`)
- 2026-03-05 prefill profiling snapshot:
  - `ESPRESSO_BENCH_SEED=1 .build/release/espresso-bench --perf-stats --inference-only --profile-kernels --warmup 50 --iterations 1000 --output /tmp/prefill_profile_ane10x_20260305`
  - Summary table now reports host eval + host overhead + IO lock/body/unlock + handoff split per layer.
  - `hwExecutionTime` remains unavailable on this host build (`_ANEPerformanceStats` request-buffer factory returns nil), and summary explicitly marks HW columns as unavailable.
- 2026-03-05 post-commit artifact verification (`git_sha=bdb0bd76209701cf910972693e5df0ee58f7ce85`):
  - Decode profile: `/tmp/decode_profile_bdb0bd7` (`--decode --profile-kernels --warmup 20 --iterations 200 --decode-steps 32 --decode-max-seq 32`)
    - ANE median `0.646 ms/token`, fastest Core ML median `1.722 ms/token`, strict speedup `2.67x`.
  - Prefill profile: `/tmp/prefill_profile_bdb0bd7` (`--perf-stats --inference-only --profile-kernels --warmup 20 --iterations 200`)
    - ANE median `1.872 ms`, Core ML `.all` median `1.541 ms`.
- 2026-03-05 reproducibility check:
  - Run1: `/tmp/decode_repro_ane10x_run1` (`ANE_COMPILE_CACHE_POLICY=auto`) median `0.648479 ms/token`
  - Run2: `/tmp/decode_repro_ane10x_run2` (`ANE_COMPILE_CACHE_POLICY=preferCached`) median `0.648500 ms/token`
  - Delta: `0.0032%` (within ±2% target)
  - Compile stability note: compile time swung from `49.49 s` (auto) to `52.7 ms` (preferCached); prefer cached policy for iteration/repro.
- Next step: isolate the smallest eval-passing vs eval-failing multi-input MIL shape family (especially mixed input tensors and lane-packed outputs), then re-expand toward full decode attention.

### Decode Artifact Index (2026-03-05)
- [x] `/tmp/decode_profile_ane10x_20260305/summary.txt`
- [x] `/tmp/decode_profile_ane10x_20260305/summary.json`
- [x] `/tmp/decode_profile_ane10x_20260305/ane_decode_token_latencies.csv`
- [x] `/tmp/decode_profile_ane10x_20260305/ane_decode_kernel_profile.csv`
- [x] `/tmp/decode_profile_ane10x_20260305/coreml_decode_all_token_latencies.csv`
- [x] `/tmp/decode_profile_ane10x_20260305/coreml_decode_cpuandneuralengine_token_latencies.csv`

## Benchmark Suite v1 (2026-03-05)
- [x] Task 1: Add EspressoBench target scaffold to Package.swift
- [x] Task 2: BenchmarkRunner — measurement harness (stats, signposts, progress)
- [x] Task 3: FLOPCalculator — TFLOPS and utilization metrics
- [x] Task 4: ResultsFormatter — report + CSV output
- [x] Task 5: ThermalMonitor — thermal state tracking
- [x] Task 6: ANEDirectBench — core ANE benchmark (inlined loop, ~Copyable safe)
- [x] Task 7: CoreMLBench — Core ML baseline with 3 compute unit configs
- [x] Task 8: Python coremltools model generator script
- [x] Task 9: Main entry point — CLI orchestration
- [x] Task 10: Integration test — build + smoke test
- [x] Task 11: Power benchmark script (powermetrics wrapper)
- [x] Task 12: Gitignore and cleanup

### Benchmark v1 Smoke Test Results (2026-03-05)
- `swift build -c release --product espresso-bench` => clean, 0 warnings
- `.build/release/espresso-bench --ane-only --warmup 3 --iterations 5` => success
- Chip detected: Apple M3 Max
- Mean: 1.207 ms, Median: 1.208 ms (5 iterations)
- Sustained TFLOPS: 3.17, ANE Utilization: 17.6%
- Time breakdown: ANE 81.9%, I/O 16.3%, CPU 1.8%
- CSV + summary.txt output verified in benchmarks/results/

## Forward Pass Inference Optimizations (2026-03-05)
- [x] Opt 1: Inference MIL generators with fused residuals (SDPAForwardInferenceGenerator, FFNForwardInferenceGenerator)
- [x] Opt 2: InferenceKernelSet — lightweight 2-kernel compilation (~Copyable struct)
- [x] Opt 3: ForwardPass.runInference() — streamlined inference path (no LayerActivations, no CPU residuals)
- [x] Opt 4: Unlocked I/O variants — lock/unlock primitives + unlocked read/write in C + Swift wrappers
- [x] Opt 5: Benchmark integration — --inference flag, ANEDirectBench.runInference(), results formatting

### Verification
- [x] `swift build` => 0 errors, 0 warnings
- [x] `swift test` => 168 tests, 0 failures (no regressions to training path)

### Files Created
- `Sources/MILGenerator/SDPAForwardInferenceGenerator.swift`
- `Sources/MILGenerator/FFNForwardInferenceGenerator.swift`
- `Sources/ANERuntime/InferenceKernelSet.swift`

### Files Modified
- `Sources/Espresso/ForwardPass.swift` — added InferenceSurfaceHandles, runInference(), runInferenceTimed()
- `Sources/ANEInterop/surface_io.c` — added lock/unlock + unlocked read/write
- `Sources/ANEInterop/include/ane_interop.h` — declared new C functions
- `Sources/ANETypes/SurfaceIO.swift` — added Swift wrappers for lock/unlock/unlocked I/O
- `Sources/EspressoBench/ANEDirectBench.swift` — added runInference() benchmark
- `Sources/EspressoBench/main.swift` — added --inference flag
- `Sources/EspressoBench/ResultsFormatter.swift` — inference results + comparison formatting

## Phase 10 Performance Push (2026-03-05)
- [ ] Group 0: Add TDD coverage for C-side batched `write-at` and batched `copy` SurfaceIO APIs
- [ ] Group 1: Implement C interop batched `write-at` + batched `copy` and Swift wrappers
- [ ] Group 2: Repack SDPA backward flow to reduce per-layer ANE dispatches
- [ ] Group 3: Add benchmark warmup + median aggregation controls (strict gates unchanged)
- [ ] Group 4: Run targeted + full verification and capture perf deltas in artifacts/README

## Phase 9 Performance Investigation & Optimization (2026-03-05)
- [x] Group 0: Baseline hardening + evidence capture
- [x] Group 1: Add ObjC-style Swift step telemetry (`t_ane`, `t_io`, `t_cls`, `t_elem`, `t_rms`, `t_cblas_wait`, `ms_per_step`)
- [x] Group 2: Extend Phase 8 benchmark parser/artifacts to capture and report timing breakdown
- [x] Group 3: Optimization batch O1 (surface handle caching in hot loops)
- [x] Group 4: Optimization batch O2 (batched SurfaceIO read paths with fewer lock/unlock cycles)
- [x] Group 5: Optimization batch O3 (CPU workspace reuse for CrossEntropy and RMSNorm)
- [x] Group 6: Optimization batch O4 (synchronization/barrier tuning + attribution)
- [x] Group 7: Full gate rerun + benchmark artifacts refresh
- [x] Group 8: README performance baseline/progress update + final report notes

### Phase 9 Verification Commands
- [x] `swift build`
- [x] `swift test`
- [x] `ANE_HARDWARE_TESTS=1 swift test --filter "ANERuntimeTests|EspressoTests|CrossValidationTests"`
- [x] `OBJC_CROSS_VALIDATION=1 ANE_HARDWARE_TESTS=1 PHASE8_BENCHMARKS=1 swift test --filter CrossValidationTests`
- [x] `./scripts/phase8_benchmark.py --regression-threshold-pct 5 --max-retries 1`

### Phase 9 Review
- [x] Baseline reproduced and documented with timestamp/hardware/toolchain
- [x] Per-batch hypothesis/result table captured (delta vs baseline, pass/fail notes)
- [x] Numerical parity + stability confirmed non-regressed
- [x] Remaining bottlenecks and next ROI actions recorded

### Phase 9 Outcome Summary (2026-03-05)
- Baseline reference copied into README:
  - timestamp `2026-03-04T21:07:16.203258+00:00`
  - hardware `Mac15,11 (Apple M3 Max)`
  - OS `macOS 26.0 (25A354)`
  - toolchain `Swift 6.2.4`
  - performance `Swift 267.8 ms/step`, `ObjC 163.6 ms/step`, ratio `1.636919`
- Iteration outcomes captured in README "Performance Baseline & Progress":
  - `O1` (telemetry + surface/cache/workspace changes): `271.7 ms/step`, ratio `1.747267` (regressed vs baseline)
  - `O2` (build fairness release mode + ObjC breakdown parsing fix): `149.8 ms/step`, ObjC `143.3 ms/step`, ratio `1.045359`
  - `O3` (C fast-path SurfaceIO read/write/batched read): `131.2 ms/step`, ObjC `145.5 ms/step`, ratio `0.901718` (Swift faster than ObjC)
- Current best benchmark artifact set:
  - `artifacts/benchmarks/phase8/latest.json`
  - `artifacts/benchmarks/phase8/latest.csv`
  - `artifacts/benchmarks/phase8/latest.md`
  - metadata timestamp `2026-03-04T22:26:15.868381+00:00`
- Gate status (current best run): `G0..G5 pass`, overall `S+ (100.00)`.
- Remaining bottlenecks from breakdown: `t_ane` and `t_io` dominate Swift-vs-ObjC delta.

## Phase 7 Execution Checklist (2026-03-04)
- [x] Group 0 pre-flight complete
- [x] Group 1 fixture binaries generated from ObjC oracle and committed
- [x] Group 2 cross-validation harness added (`scripts/cross_validate.sh`, `CrossValidationTests`)
- [x] Group 3 hardware/integration/perf-gated sweeps run and documented
- [x] Group 4 ObjC source archived to `archive/training`, active `training/` reduced to `golden_outputs/`
- [x] Group 5 documentation + final verification completed

### Phase 7 Verification Results (2026-03-04)
- `swift build` => pass
- `swift test` => pass (`156` executed, `41` skipped, `0` failures)
- `ANE_HARDWARE_TESTS=1 OBJC_CROSS_VALIDATION=1 swift test --filter "ANERuntimeTests|EspressoTests|CrossValidationTests"` => pass (`66` executed, `8` skipped, `0` failures)
- `ANE_HARDWARE_TESTS=1 ESPRESSO_INTEGRATION_TESTS=1 ESPRESSO_CKPT_COMPAT_TESTS=1 ESPRESSO_GRADIENT_PARITY_TESTS=1 swift test --filter EspressoTests` => pass (`27` executed, `2` skipped, `0` failures)
- `ANE_HARDWARE_TESTS=1 ESPRESSO_PERF_TESTS=1 swift test --filter test_100_steps_benchmark` => skipped (expected on M3 Max; M4-only target)
- `OBJC_CROSS_VALIDATION=1 ANE_HARDWARE_TESTS=1 swift test --filter CrossValidationTests` => pass (`6` executed, `0` failures)
- `swift build -c release` => pass
- `grep -r "func test_" Tests/ | wc -l` => `156`
- `find Sources -name "*.swift" | xargs wc -l | tail -n 1` => `4759 total`

### Phase 7 Artifacts
- Group 1 fixture binaries:
  - `Tests/ANERuntimeTests/Fixtures/fwd_attn_oOut_seq256_f32le.bin` (`786432` bytes)
  - `Tests/ANERuntimeTests/Fixtures/fwd_ffn_y_seq256_f32le.bin` (`786432` bytes)
  - `Tests/ANERuntimeTests/Fixtures/ffn_bwd_dx_seq256_f32le.bin` (`786432` bytes)
- Group 2 golden outputs:
  - `training/golden_outputs/test_*.txt` (10 stdout fixtures)
  - `training/golden_outputs/*.bin` + `*.mil` binary oracle inputs/outputs for CV tests

### Phase 7 Host Notes (M3 Max)
- `test_10_steps_loss_decreases` skips without `STORIES_MODEL_PATH`/`stories110M.bin`.
- `test_100_steps_benchmark` is correctly skipped on non-M4 hardware.
- `ANEInteropTests.test_compile_identity_kernel` remains flaky in unfiltered full-hardware runs; use filtered hardware sweeps as documented.
- Several ObjC probe executables are unstable on this host (`test_weight_reload`, `test_perf_stats`, `test_ane_advanced`, `test_fused_bwd`, `test_ane_sdpa5`); `scripts/cross_validate.sh` now fails fast (strict mode) instead of reporting success with stale/mixed artifacts.
- Failed cross-validation capture logs are emitted to `.build/phase7-cross-validate/failed/` for diagnosis.

## Phase 8 Implementation Checklist (2026-03-04)
- [x] Group 0: Read source docs/context (`tasks/phase8-plan.md` if present, `tasks/phase7-todolist.md`, `tasks/todo.md`, `tasks/lessons.md`)
- [x] Group 1: ANEInterop baseline stabilization for host-unstable identity probe
- [x] Group 2: Cross-validation strictness split (required vs optional probes) + benchmark metric emission in `CrossValidationTests`
- [x] Group 3: Benchmark artifacts + grading pipeline (`latest.json`, `latest.csv`, `latest.md`)
- [x] Group 4: CI/doc updates for Phase 8 lanes and env gates
- [x] Group 5: Final verification sweep + review evidence + lessons update

### Phase 8 Gate Checklist
- [x] G0 Build/Test baseline
- [x] G1 Hardware correctness lane
- [x] G2 ObjC cross-validation lane
- [x] G3 Benchmark artifact completeness
- [x] G4 Grading gate
- [x] G5 Regression gate

### Phase 8 Review Evidence
- G0: `swift build` + `swift test` => pass (`artifacts/benchmarks/phase8/logs/g0_swift_build-attempt1.log`, `g0_swift_test-attempt1.log`), `159` tests executed, `41` skipped, `0` failures.
- G1: `ANE_HARDWARE_TESTS=1 swift test --filter "ANERuntimeTests|EspressoTests|CrossValidationTests"` => pass (`66` executed, `17` skipped, `0` failures). Host-expected skips captured (baseline unavailable for one eval test, integration env off, ObjC CV env off).
- G2: `OBJC_CROSS_VALIDATION=1 ANE_HARDWARE_TESTS=1 PHASE8_BENCHMARKS=1 swift test --filter CrossValidationTests` => pass (`6` executed, `0` failures). Emitted metric JSON rows for `full_fused_forward` + `fused_backward`.
- G3: `artifacts/benchmarks/phase8/latest.json`, `latest.csv`, `latest.md` exist and include current timestamp (`2026-03-04T20:58:01.288988+00:00`) + git SHA (`f2d1da0e30c3cc964297c9b3112f55a1191730da`).
- G4: Overall grade present (`S+ + 100.00`), parity section includes tolerance checks (`max_abs_diff=0`, `mean_abs_diff=0`, pass), and missing performance metrics are explicitly marked `N/A` with evidence (`archive/training/train_large.m` missing).
- G5: Regression comparison executed against previous snapshot in `latest.json`; status `PASS` (`Within threshold`, abs change `0.0`, pct change `0.0%`, threshold `5%`).

## Phase 6b Execution Checklist (2026-03-04)
- [x] Group 0 pre-flight baseline green (`swift build`, `swift test`, `ANE_HARDWARE_TESTS=1 swift test --filter ANERuntimeTests`)
- [x] Group 1 ANERuntime additions complete (`requireObjCCrossValidation`, fwdAttn 6xDIM test, 3 numerical equivalence tests)
- [x] Group 2 Espresso backward IOSurface chain test complete
- [x] Group 3 hardware gates complete (compile-time, integration, perf, exec-restart ObjC contract)
- [x] Group 4 closeout complete (`tasks/phase6b-todolist.md`, `tasks/todo.md`, `tasks/lessons.md` updated)

### Phase 6b Verification Results (2026-03-04)
- `swift build` => pass
- `swift test` => pass (`150` executed, `35` skipped, `0` failures)
- `ANE_HARDWARE_TESTS=1 swift test --filter ANERuntimeTests` => pass (`33` executed, `5` skipped, `0` failures)
- `ANE_HARDWARE_TESTS=1 OBJC_CROSS_VALIDATION=1 swift test --filter ANERuntimeTests` => pass (`33` executed, `6` skipped, `0` failures)
- Cross-validation fixture status: strict full-vector tests are fixture-backed and passing with committed fixtures:
- `Tests/ANERuntimeTests/Fixtures/fwd_attn_oOut_seq256_f32le.bin`
- `Tests/ANERuntimeTests/Fixtures/fwd_ffn_y_seq256_f32le.bin`
- `Tests/ANERuntimeTests/Fixtures/ffn_bwd_dx_seq256_f32le.bin`
- `ANE_HARDWARE_TESTS=1 swift test --filter test_backward_iosurface_copy_chain` => pass
- `ANE_HARDWARE_TESTS=1 ESPRESSO_INTEGRATION_TESTS=1 ESPRESSO_CKPT_COMPAT_TESTS=1 ESPRESSO_GRADIENT_PARITY_TESTS=1 swift test --filter EspressoTests` => pass (`27` executed, `2` skipped, `0` failures)
- Integration measured values:
- loss(step0): ObjC=`10.672213`, Swift=`10.672213`, abs diff=`0.000000`
- gradient m-norm relative error: max=`0.000000`, mean=`0.000000` (12/12 layers)
- `ANE_HARDWARE_TESTS=1 ESPRESSO_PERF_TESTS=1 swift test --filter test_100_steps_benchmark` => skipped on this host (`Perf target is M4; got Apple M3 Max`)
- `xcrun clang ... -o archive/training/tiny_train archive/training/tiny_train.m ...` => build pass
- `./archive/training/tiny_train --steps 5` => runtime failure on this host (`Initial compile failed!`)

## Post-Phase 6 Review Fixes (2026-03-04)
- [x] TDD: update `CPUOpsTests.test_rmsnorm_backward_numerical_gradient_check` finite-diff to use non-unit weights (should fail with current bug)
- [x] Fix `RMSNorm.backward` dx derivative for non-unit weights (`dx = rrms * (dy*w - x*dot)`), keep `dw` accumulation semantics
- [x] Verify: `swift test --filter CPUOpsTests.test_rmsnorm_backward_numerical_gradient_check`
- [x] TokenDataset: set FD_CLOEXEC on dataset fd (prevent leaks across `exec*` restarts)
- [x] TDD: add EspressoTests coverage that TokenDataset fd has `FD_CLOEXEC` set
- [x] ExecRestart: preserve CLI args across restart and resolve executable path robustly; ensure `--resume` included exactly once
- [x] TDD: add EspressoTests coverage for ExecRestart argv building and exec-path resolution (pure functions; no real exec)
- [x] ModelWeightLoader: validate vocab magnitude (and seqLen/heads as needed) in header config check to fail fast
- [x] TDD: add ANERuntimeTests coverage for vocab mismatch (expect `.configMismatch`)
- [x] Gate: `swift build && swift test`
- [x] Document results and lessons in `tasks/todo.md` + `tasks/lessons.md`

Review notes:
- Fixed `RMSNorm.backward` dx derivative for non-unit weights; expanded finite-diff coverage to prevent regression.
- `TokenDataset` now sets `FD_CLOEXEC` (`O_CLOEXEC` + `fcntl`) to avoid fd leaks across exec-restarts; added unit test.
- `ExecRestart` now resolves executable path via dyld + uses `execvp`, preserves CLI args, and normalizes `--resume` to exactly once; added unit tests (pure, no exec).
- `ModelWeightLoader` now validates header vocab magnitude + seqLen + heads/kvHeads before reading payload; added vocab-mismatch test.

Verification (2026-03-04):
- `swift build` => pass
- `swift test` => 143 executed, 29 skipped, 0 failures

## Pointer Marshaling Design Review (2026-03-03)
- [x] Inspect `ane_interop_compile` C ABI contract (`milText`, `weightPaths`, `weightDatas`, `weightLens`)
- [x] Audit current Swift pointer marshalling style in `ANEInteropTests`
- [x] Design Swift 6.2-safe marshaling helper for `weights: [(path: String, data: Data)]`
- [x] Verify design against repo style (preconditions/guards/lifetime scope patterns)
- [x] Document pitfalls to avoid + recommended tests
- [x] Add review notes/results in this section

Review notes:
- Keep all FFI pointers in one lexical scope using nested `withUnsafe*` closures (same pattern already used in `ANEInteropTests`).
- Do not escape `Data.withUnsafeBytes` or `String.withCString` pointers into storage that outlives their closure.
- Use dynamic recursion to build all weight data pointers simultaneously without copying large `Data` payloads.
- Keep C `size_t` arrays as Swift `[Int]` on this target; avoid `[Int32]` truncation.
- Preserve C contract for empty payloads: `len == 0` may pair with `data == nil`.

## Phase 1–3 Audit (2026-03-03)
- [x] Run `swift test` (capture output snippet)
- [x] Run `ANE_HARDWARE_TESTS=1 swift test --filter ANEInteropTests` (if on ANE-capable machine)
- [x] Investigate `ANEInteropTests.test_compile_identity_kernel` failure (memory safety + IOSurface semantics + numerics)
- [x] Phase 1 line-by-line read + cross-ref training headers
- [x] Phase 2 line-by-line read + cross-ref training headers
- [x] Phase 3 line-by-line read + cross-ref training headers
- [x] API surface table (Phase 1–3) + coverage gaps
- [x] Implement fixes + add/adjust tests (target: 10+ findings)
- [x] Docs consistency: `tasks/rewrite-plan.md` vs implementation
- [x] Add audit review summary (findings + fix list)

### Test Improvements (2026-03-03)
- [x] Add ANEInterop index bounds tests for get_input/get_output
- [x] Add ANEInterop compile-count set/get non-hardware coverage
- [x] Add ANETypes tests for zero-shape blobs and LayerGradients reset
- [x] Add ANETypes LayerStorage read/modify coverage
- [x] Add MILGenerator tests for fused payload ordering and throw paths
- [x] Add MILGenerator non-default causal-mask and fused-header coverage
- [x] Expand locale and causal-mask header assertions in MIL tests
- [x] Run targeted test targets and full `swift test`

### Phase 2 (ANETypes) Focus Audit — Ownership, Lifetimes, Layout (2026-03-03)
- [x] Ownership: audit `~Copyable` move/use patterns (no implicit copies, no escapes)
- [x] Lifetimes: audit Unsafe* pointers and `Data.withUnsafeBytes` usage (no escaping pointers)
- [x] IOSurface I/O: audit lock/unlock + bounds + offset math (`SurfaceIO` + C shim)
- [x] Layout parity: verify `CheckpointHeader` matches training C struct (size/align/offsets/endianness)
- [x] Perf: identify hidden allocations/copies (Data<->Array, slicing, intermediate buffers)
- [x] Add/adjust tests to catch any issues found
- [x] Write review findings here with file:line + fix + test

### Phase 1–3 Audit Review (2026-03-03)
- Findings triaged: 15 total (2 critical, 8 high, 4 medium, 1 low)
- Implemented fix areas:
  - Phase 1: interop IO safety contracts (`bool` return), bounds/overflow guards, lock result checks, self-copy memmove semantics, compile pointer preflight, retained surface copy accessors
  - Phase 2: `TensorBuffer` typed binding, `LayerActivations` Q/K/V parity and order, SurfaceIO bounds/no-op checks, `ModelConfig` parity constants, checkpoint default magic/version, fp16 blob passthrough builder
  - Phase 3: full GenericMIL fixture parity tests, full fused blob parity fixtures vs ObjC reference bytes, byte-contract tests for all generators, locale-explicit format static check
- Test status:
  - `swift test` => pass (60 executed, 3 skipped, 0 failures)
  - `ANE_HARDWARE_TESTS=1 swift test --filter ANEInteropTests` => fail (14 executed, 2 failures) at identity eval path
- Root-cause bucket status (hardware run):
  - Primary observed bucket: runtime request/eval path mismatch (`ANEProgramProcessRequestDirect` inference failure)
  - Additional verification: source-of-truth `training/ane_runtime.h` harness reproduces same identity eval failure on this host, so failure is not yet isolated to Swift interop implementation.

---

## Pre-Phase: Project Setup
- [x] Create `Package.swift` (swift-tools-version: 6.0) with all 7 targets
- [x] Verify existing `Makefile` still builds ObjC (`make -C training train_large`) — keep as reference throughout
- [x] Pin Swift toolchain version in `.swift-version` (Swift 6.2.4) for reproducible strict-concurrency diagnostics
- [x] Audit all 14 `objc_msgSend` signatures in `ane_runtime.h` + `stories_io.h` against proposed C API

---

## Phase 1: ANEInterop — C/ObjC Bridging Target (M)
- [x] Create `Sources/ANEInterop/include/ane_interop.h` — C API header (cover all 14 objc_msgSend signatures + io_copy + io_write_fp16_at + compile count get/set)
- [x] Create `Sources/ANEInterop/ane_interop.m` — ObjC impl (dlopen, objc_msgSend, CFBridging)
- [x] Create `Sources/ANEInterop/neon_convert.c` — NEON fp16<->fp32 shims
- [x] Create `Sources/ANEInterop/surface_io.c` — `ane_interop_io_copy` + `ane_interop_io_write_fp16_at` (backward pass hot path: 72 + 24 calls/step)
- [x] Write tests FIRST (TDD):
  - [x] `test_init_idempotent` — call twice, no crash
  - [x] `test_create_surface_valid` — verify IOSurfaceGetAllocSize == requested
  - [x] `test_compile_invalid_mil_returns_nil` — garbage input, expect NULL (gated by `ANE_HARDWARE_TESTS=1`)
  - [x] `test_compile_identity_kernel` — minimal MIL cast fp32->fp16->fp32, verify roundtrip (gated by `ANE_HARDWARE_TESTS=1`)
  - [x] `test_compile_count_increments` — reset, compile, verify count == 1 (gated by `ANE_HARDWARE_TESTS=1`)
  - [x] `test_neon_f32_f16_roundtrip` — 100K random values, max error < 1e-2
  - [x] `test_io_copy_between_surfaces` — write data to surface A, io_copy to surface B at offset, verify fp16 data
  - [x] `test_io_write_fp16_at_offset` — write fp32 data at channel offset, read back, verify within 1e-2
- [x] Verify: Pure tests green; ANE compile/eval tests gated behind `ANE_HARDWARE_TESTS=1`
- [x] Follow-up hardening:
  - [x] Fail fast + cleanup if request creation fails (avoid “compile succeeds, eval always fails”)
  - [x] Handle NULL IOSurface allocations during compile (cleanup partial handles safely)

## Phase 2: ANETypes — Swift Types + IOSurface I/O (L)
- [x] Create `Sources/ANETypes/ModelConfig.swift` — enum namespace for all constants
- [x] Create `Sources/ANETypes/TensorBuffer.swift` — `~Copyable` buffer wrapper
- [x] Create `Sources/ANETypes/LayerWeights.swift` — 9 buffer fields with deinit
- [x] Create `Sources/ANETypes/AdamState.swift` — m, v buffers
- [x] Create `Sources/ANETypes/LayerAdam.swift` — 9 AdamState fields (Wq/Wk/Wv/Wo/W1/W2/W3/rmsAtt/rmsFfn)
- [x] Create `Sources/ANETypes/LayerActivations.swift` — 13 buffer fields (`layerIn`, `xnorm`, `Q/K/V`, `attnOut`, `oOut`, `x2`, `x2norm`, `h1`, `h3`, `siluOut`, `ffnOut`)
- [x] Create `Sources/ANETypes/LayerGradients.swift` — 9 gradient accumulators + zero()
- [x] Create `Sources/ANETypes/CheckpointHeader.swift` — @frozen struct with `validateLayout()` assertions (size=96, alignment=8, offsets incl. cum* Doubles @48/56/64, cumSteps@72, adamT@80, pad2@92)
- [x] Create `Sources/ANETypes/SurfaceIO.swift` — writeFP16/readFP16 via C shim
- [x] Create `Sources/ANETypes/WeightBlob.swift` — build/buildTransposed with 128-byte header
- [x] Create `Sources/ANETypes/LayerStorage.swift` — fixed-size ~Copyable container using coroutine accessors (`_read`/`_modify`)
- [x] Write tests FIRST (TDD):
  - [x] `test_layer_weights_alloc_dealloc_no_leak` — 100 iterations
  - [x] `test_adam_state_initialized_to_zero`
  - [x] `test_build_blob_header_magic` — bytes [0]=0x01, [4]=0x02, [64..67]=0xDEADBEEF
  - [x] `test_build_blob_fp16_accuracy` — known values, within 1e-2
  - [x] `test_build_blob_transposed_layout` — 3x4 matrix, column-major fp16
  - [x] `test_surface_write_read_roundtrip` — 100K random values, max error < 1e-2
  - [x] `test_surface_read_with_channel_offset`
  - [x] `test_surface_write_fp16_at_offset` — write fp32 data at non-zero channel offset, verify only target region changed
  - [x] `test_surface_copy_fp16_between_surfaces` — write to surface A, copyFP16 to surface B at offset, verify fp16 data integrity
  - [x] `test_checkpoint_header_layout` — size==96 AND field offset assertions (cumCompile@48, cumTrain@56, cumWall@64, cumSteps@72, adamT@80, pad2@92)
- [x] Verify: Blob header/layout tests green; fp16 roundtrip < 1e-2; checkpoint offsets match C
- [x] Follow-up hardening:
  - [x] `TensorBuffer` allocates with 64B alignment (cblas/vDSP-friendly)
  - [x] Confirm SE-0474 yielding accessors are not accepted by Apple Swift 6.2.4; keep `_read/_modify` (isolated to `LayerStorage`)
  - [x] `CheckpointHeader.validateLayout()` asserts offsets for *all* fields (not just cum* / tail)

## Phase 3: MILGenerator — MIL Text Generation (L) [parallel with Phase 4]
- [x] Create `Sources/MILGenerator/GenericMIL.swift` — conv, matmul, fusedQKV, fusedFFNUp
- [x] Create `Sources/MILGenerator/CausalMask.swift` — cached blob(seqLen:) -> Data (match ObjC g_mask_blob; fp16 0 / -65504)
- [x] Create `Sources/MILGenerator/SDPAForwardGenerator.swift`
- [x] Create `Sources/MILGenerator/FFNForwardGenerator.swift`
- [x] Create `Sources/MILGenerator/FFNBackwardGenerator.swift`
- [x] Create `Sources/MILGenerator/SDPABackward1Generator.swift`
- [x] Create `Sources/MILGenerator/SDPABackward2Generator.swift`
- [x] Create `Sources/MILGenerator/QKVBackwardGenerator.swift`
- [x] **CRITICAL**: All float formatting uses `Locale(identifier: "en_US_POSIX")`
- [x] Generate golden MIL files from ObjC (scripts/generate_golden_mil.sh)
- [x] Write tests FIRST (TDD):
  - [x] `test_sdpa_fwd_text_matches_objc` — character-by-character
  - [x] `test_ffn_fwd_text_matches_objc`
  - [x] `test_ffn_bwd_text_matches_objc`
  - [x] `test_sdpa_bwd1_text_matches_objc`
  - [x] `test_sdpa_bwd2_text_matches_objc`
  - [x] `test_qkvb_text_matches_objc`
  - [x] `test_causal_mask_diagonal_zero_upper_neg65504` — verify fp16 mask uses 0 / -65504 exactly
  - [x] `test_causal_mask_blob_cached_identity`
  - [x] `test_fused_qkv_blob_offsets_correct`
  - [x] `test_fused_ffnup_blob_offsets_correct`
  - [x] `test_locale_does_not_affect_mil_formatting` — run with de_DE locale, verify identical output
- [x] Verify: Character-identical MIL text + locale-invariant output (ANE compile covered in Phase 5)

## Phase 4: CPUOps — Accelerate/vDSP Operations (M) [parallel with Phase 3]
- [x] Create `Sources/CPUOps/RMSNorm.swift` — forward + backward via vDSP
- [x] Create `Sources/CPUOps/CrossEntropy.swift` — lossAndGradient via vDSP
- [x] Create `Sources/CPUOps/AdamOptimizer.swift` — update with bias correction
- [x] Create `Sources/CPUOps/Embedding.swift` — lookup + backward, channel-first
- [x] Create `Sources/CPUOps/RoPE.swift` — apply + backward
- [x] Create `Sources/CPUOps/SiLU.swift` — forward + backward
- [x] Write tests FIRST (TDD):
  - [x] `test_rmsnorm_forward_known_values` — dim=4, seq=2, within 1e-5
  - [x] `test_rmsnorm_backward_numerical_gradient_check` — rel error < 1e-3
  - [x] `test_cross_entropy_uniform_logits` — loss = log(V)
  - [x] `test_cross_entropy_matches_reference_and_scaling` — full loss/grad parity vs scalar reference
  - [x] `test_cross_entropy_gradient_sums_to_zero`
  - [x] `test_adam_single_step_known_values`
  - [x] `test_adam_bias_correction`
  - [x] `test_adam_multi_element_reference_parity` — vector-indexing parity vs scalar reference loop
  - [x] `test_embedding_lookup_correct_rows`
  - [x] `test_embedding_backward_accumulates` — non-uniform gradient fixture to validate channel-first indexing
  - [x] `test_rope_forward_backward_matches_reference` — direct forward/backward parity on multiple shapes
  - [x] `test_rope_forward_backward_consistency`
  - [x] `test_silu_forward_matches_closed_form`
  - [x] `test_silu_backward_matches_numerical_derivative_of_forward`
- [x] Verify: Gradient checks pass on seeded random configs, no Accelerate warnings

### Phase 4 Review (2026-03-03)
- Added `CPUOps` implementation files: `RMSNorm.swift`, `CrossEntropy.swift`, `AdamOptimizer.swift`, `Embedding.swift`, `RoPE.swift`, `SiLU.swift`.
- Replaced placeholder tests with full `CPUOpsTests.swift` and deterministic `SplitMix64` RNG.
- Removed placeholders: `Sources/CPUOps/CPUOps.swift`, `Tests/CPUOpsTests/PlaceholderTests.swift`.
- Post-review correction: restored `RMSNorm.backward` to exact C-parity semantics and updated test coverage to assert non-unit-weight C parity explicitly.
- Hardening follow-up:
  - Added explicit bounds contracts for CE targets and embedding token IDs.
  - Extended tests to cover CE gradient scaling, Adam vector parity, embedding index mapping, RoPE direct parity, and explicit SiLU forward validation.
- Verification:
  - `swift test --filter CPUOpsTests` => pass (14 tests, 0 failures)
  - `swift build 2>&1 | grep -i \"error\\|warning\"` => no output
  - `swift test` => pass (88 executed, 4 skipped, 0 failures)

## Phase 5: ANERuntime — Kernel Lifecycle in Swift (XL)
- [x] Create `Sources/ANERuntime/ANEError.swift` — typed throws enum (.compilationFailed, .evaluationFailed, .compileBudgetExhausted, .surfaceAllocationFailed)
- [x] Create `Sources/ANERuntime/ANEKernel.swift` — ~Copyable, owns OpaquePointer, eval() throws on failure
- [x] Create `Sources/ANERuntime/LayerKernelSet.swift` — owns 5 weight-bearing ANEKernel instances (recompiled each batch)
- [x] Create `Sources/ANERuntime/StaticKernel.swift` — owns weight-free sdpaBwd2 kernel (compiled once, different lifecycle from LayerKernelSet)
- [x] Create `Sources/ANERuntime/ModelWeightLoader.swift` — load pretrained llama2.c `.bin` (Appendix A.5, train_large.m parity)
- [x] Create `Sources/ANERuntime/CompileBudget.swift` — wrapper over C compile count
- [x] Cross-validate against existing ObjC test executables (test_full_fused/test_fused_bwd built and invoked; compile failure behavior captured on this host)
- [ ] Write tests FIRST (TDD):
  - [x] `test_compile_identity_kernel_succeeds`
  - [x] `test_compile_invalid_mil_throws`
  - [x] `test_eval_identity_roundtrip` — [1,2,3,4] verify within 1e-2 (host-gated skip when ANE eval unavailable)
  - [x] `test_kernel_deinit_calls_free`
  - [x] `test_eval_returns_throws_on_failure`
  - [x] `test_compile_budget_tracks_count`
  - [x] `test_compile_budget_exhausted_blocks_compile`
  - [x] `test_compile_layer_kernels_with_random_weights` (covered by `test_layer_kernel_set_compiles_all_five_and_surface_sizes`)
  - [x] `test_fwd_attn_output_has_6xdim_channels`
  - [x] `test_fwd_attn_numerical_equivalence_with_objc` — max |diff| < 1e-2
  - [x] `test_fwd_ffn_numerical_equivalence_with_objc`
  - [x] `test_ffn_bwd_numerical_equivalence_with_objc`
  - [x] `test_model_weights_bin_layout_small` (covered by `test_model_weight_loader_payload_layout_matches_llama2c_order_and_sizes`)
  - [x] `test_model_weights_vocab_sign_shared_vs_unshared` (covered by `test_model_weight_loader_vocab_mismatch_fails_fast`)
  - [x] `test_load_stories110m_weights` (covered by `test_load_stories110m_weights_integration`; skips without local model file)
  - [x] `test_sdpa_bwd2_compile_once_reuse` (covered by `test_static_kernel_eval_produces_output`)
  - [x] `test_sdpa_bwd2_lifecycle_independent` (covered by `test_static_kernel_survives_layer_kernel_set_dealloc`)
  - [x] `test_backward_iosurface_copy_chain` — full forward→backward IOSurface data flow through io_copy chain
- [ ] Verify: Forward attn/FFN match ObjC within 1e-2, layer kernels compile < 2000ms (requires committed seq64 ObjC fixture files)

### Phase 5b Execution Checklist (2026-03-03)
- [x] Re-verify baseline before edits: `swift test --filter ANERuntimeTests` (2 pass, 8 skipped)
- [x] Append 12 Phase 5b tests in `Tests/ANERuntimeTests/ANERuntimeTests.swift` (TDD first)
- [x] Run `swift test --filter ANERuntimeTests` and capture expected pre-implementation failures
- [x] Implement `Sources/ANERuntime/StaticKernel.swift`
- [x] Implement `Sources/ANERuntime/LayerKernelSet.swift`
- [x] Implement `Sources/ANERuntime/ModelWeightLoader.swift`
- [x] Re-run `swift test --filter ANERuntimeTests` to green
- [x] Run full regression `swift test`
- [x] Write Phase 5b review notes and verification results in this file

### Phase 5b Review (2026-03-03)
- Added runtime files:
  - `Sources/ANERuntime/LayerKernelSet.swift`
  - `Sources/ANERuntime/StaticKernel.swift`
  - `Sources/ANERuntime/ModelWeightLoader.swift`
- Updated tests:
  - `Tests/ANERuntimeTests/ANERuntimeTests.swift` now uses `@testable import ANERuntime`.
  - Appended 12 Phase 5b tests (kernel lifecycle + model loader parsing/error/integration behavior).
- Implementation details:
  - `LayerKernelSet` compiles all 5 weight-bearing kernels with exact weight-path naming and normal/transposed blob mapping parity to `train_large.m`.
  - `StaticKernel` compiles weight-free `SDPABackward2Generator` once with empty weight array.
  - `ModelWeightLoader` performs llama2.c header parse + config validation + strict `fread` truncation checks + per-type-all-layers payload read order.
  - Added internal `ModelWeightLoader.parseHeader(from:)` for direct header parsing tests.
- Verification:
  - `swift build` => pass
  - `swift test --filter ANERuntimeTests` => pass (22 executed, 16 skipped, 0 failures)
  - `swift test` => pass (111 executed, 20 skipped, 0 failures)

### Phase 5a Review (2026-03-03)
- Added runtime files: `Sources/ANERuntime/ANEError.swift`, `Sources/ANERuntime/ANEKernel.swift`, `Sources/ANERuntime/CompileBudget.swift`.
- Replaced placeholders: removed `Sources/ANERuntime/ANERuntime.swift` and `Tests/ANERuntimeTests/PlaceholderTests.swift`; added `Tests/ANERuntimeTests/ANERuntimeTests.swift` with 10 tests.
- Updated `Package.swift` `ANERuntimeTests` target dependencies to import `ANEInterop`, `ANETypes`, and `MILGenerator` directly.
- Hardening follow-up:
  - Added compile-failure reason plumbing in C interop (`ane_interop_last_compile_error`) and duplicate weight-path rejection.
  - Added explicit ownership/diagnostic APIs in C interop (`ane_interop_set_force_eval_failure`, `ane_interop_live_handle_count`) and retained annotation for `ane_interop_create_surface`.
  - Updated `ANEKernel` to use serialized compile-budget gate (no check/compile race in-process), map compile failures to typed `ANEError`, and return retained surfaces with typed surface index/access errors.
  - Updated `CompileBudget.setCount` to typed throws instead of precondition trap.
  - Expanded tests: forced eval-failure typed throw path, typed invalid-surface-index errors, duplicate weight-path contract, handle deinit/live-count verification, non-hardware compile-budget exhaustion coverage, and tolerance tightened to `1e-2`.
- Verification:
  - `swift build` => pass
  - `swift test --filter ANERuntimeTests` => pass (10 executed, 8 skipped, 0 failures)
  - `ANE_HARDWARE_TESTS=1 swift test --filter ANERuntimeTests` => pass (10 executed, 0 skipped, 0 failures)
  - `swift test` => pass (99 executed, 12 skipped, 0 failures)
- Host note:
  - `ANE_HARDWARE_TESTS=1 swift test --filter ANEInteropTests.test_compile_identity_kernel` still fails with `ANEProgramProcessRequestDirect ... Program Inference error` on this host (external ANE eval instability).

## Phase 6: Espresso — Forward, Backward, Loop, Checkpoint (XL)
- [x] Create `Sources/Espresso/GradientAccumulator.swift` — @unchecked Sendable
- [x] Create `Sources/Espresso/ForwardPass.swift` — 12-layer forward
- [x] Create `Sources/Espresso/BackwardPass.swift` — reverse order + async cblas (maps to train_large.m:461-575)
- [x] Create `Sources/Espresso/Checkpoint.swift` — save/load binary format
- [x] Create `Sources/Espresso/TokenDataset.swift` — mmap DATA_PATH as UInt16 tokens (train_large.m parity)
- [x] Create `Sources/Espresso/Sampler.swift` — srand48(42+startStep) + drand48() sampling (train_large.m parity)
- [x] Create `Sources/Espresso/ExecRestart.swift` — exec restart (implemented via execv; behavior matches execl contract)
- [x] Create `Sources/EspressoTrain/main.swift` — @main entry point, call `CheckpointHeader.validateLayout()` at startup
- [ ] Cross-validate against tiny_train.m (593 lines) as simpler golden reference alongside train_large.m
- [x] Write tests FIRST (TDD):
  - [x] `test_forward_single_layer_output_nonzero_finite` (ANE-gated)
  - [x] `test_forward_12_layers_no_nan` (ANE-gated)
  - [x] `test_backward_produces_nonzero_gradients` (ANE-gated)
  - [x] `test_backward_residual_gradient_flow` (ANE-gated)
  - [x] `test_checkpoint_save_load_roundtrip` — byte-identical weights
  - [x] `test_checkpoint_segment_order_small` — validates binary segment order/offsets without requiring a full-size checkpoint fixture
  - [x] `test_checkpoint_binary_compatible_with_objc` (ANE-gated + env gated)
  - [x] `test_single_step_loss_matches_objc` — |diff| < 0.01 (ANE-gated + env gated)
  - [x] `test_10_steps_loss_decreases` — overfit on tiny pattern (ANE-gated + env gated)
  - [x] `test_gradient_accumulation_averages` — ACCUM_STEPS=2
  - [x] `test_1_step_gradients_match_objc` — per-layer Adam-m norm rel error < 5% (ANE-gated + env gated)
- [x] `test_100_steps_benchmark` — verify <= 9.3 ms/step on M4 (ANE-gated + env gated)
  - [x] `test_gradient_accumulation_scaling` — verify 1.0/steps_batch scaling applied before Adam
  - [x] `test_exec_restart_checkpoint_roundtrip` — save at step N, restart with --resume, state restored
- [ ] Verify: Loss matches ObjC within 0.01, gradients within 5%, benchmark <= 9.3 ms/step (M4 benchmark gate skipped on Apple M3 Max host)

### Phase 6 Review (2026-03-04)
- Implemented Phase 6 Swift rewrite of `training/train_large.m` across `Espresso` + `EspressoTrain` (checkpoint, forward/backward, training loop, exec-restart).
- Added integration tests that cross-run ObjC `training/train_large` and Swift `espresso-train` (gated by `ANE_HARDWARE_TESTS=1` and `ESPRESSO_INTEGRATION_TESTS=1`).
- Added perf benchmark test (gated by `ANE_HARDWARE_TESTS=1` and `ESPRESSO_PERF_TESTS=1`, plus M4 check).
- Spot-check mappings (C parity):
  - Checkpoint segment order: per-layer weights then per-layer Adam m/v, then globals.
  - Forward fwdAttn output offsets: oOut@0, attnOut@4*dim, xnorm@5*dim.
  - Backward sdpaBwd1/sdpaBwd2 chain matches 15-op IOSurface table (dscores copy from sdpaBwd1 out @dim; Q|K copy from fwdAttn out @dim).
  - Gradient scaling uses `1.0/stepsBatch` for all 9 per-layer grads + `grmsFinal` + `gembed`, before Adam.
  - Residual gradient propagation uses `dy = dxRms1 + dx2` per layer.
- Verification:
  - `swift build` => pass (no warnings)
  - `swift test` => pass (139 executed, 29 skipped, 0 failures)

---

## Rollback Strategy
- [x] Keep existing `Makefile` and all `.m`/`.h` files untouched throughout (reference build)
- [x] Both build systems (Makefile + SwiftPM) coexist until Phase 6 verification passes
- [ ] If a phase fails: delete failing Swift target directory + test target, no other targets affected
- [ ] ObjC code removed only after Phase 6 end-to-end verification passes

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

Critical path: 1 → 2 → 5 → 6 (Phases 3 & 4 parallelizable)

---

## Verification Tolerances

| Phase | Method | Tolerance |
|-------|--------|-----------|
| 1 | Identity kernel eval roundtrip | fp16: 1e-2 |
| 2 | Blob byte-identical to ObjC; fp16 roundtrip; header offsets | fp16: 1e-2 |
| 3 | MIL text character-identical to ObjC (locale-invariant) | Exact match |
| 4 | Numerical gradient check (finite difference) | Relative: 1e-3 |
| 5 | Same-input kernel output: Swift vs ObjC | fp16: 1e-2 |
| 6 | End-to-end loss, gradients, benchmark | Loss: 0.01, Grad: 5%, Perf: 9.3ms |

---

## Phase 9: ANE Inference 10x vs Core ML (2026-03-05)

Goal: maximize advantage vs Core ML by eliminating structural overhead (dispatch, I/O, coherency) and matching or exceeding Core ML's compiler/runtime optimizations. Track each optimization with: correctness test, benchmark deltas, raw CSV outputs.

### A) Measurement Infrastructure (Must Be Trusted)
- [ ] Land inference-only benchmark flow (`--inference-only`) to avoid training compile budget issues during tuning.
- [ ] Land per-kernel stage profiling (`--profile-kernels`) and CSV export (`ane_inference_kernel_profile.csv`).
- [ ] Benchmark: baseline inference-only profile (ANE-only):
  - [ ] `.build/release/espresso-bench --inference-only --ane-only --profile-kernels --warmup 50 --iterations 500 --output /tmp/ane10x_profile_baseline`
- [ ] Benchmark: fp16 handoff inference-only profile (ANE-only):
  - [ ] `.build/release/espresso-bench --inference-only --ane-only --profile-kernels --inference-fp16-handoff --warmup 50 --iterations 500 --output /tmp/ane10x_profile_fp16`
- [ ] Write a short readout in this file: which segment dominates (attn eval vs ffn eval vs read/write vs gaps).

### B) True Hardware Execution Time (Split Wall-Time vs HW)
- [ ] Integrate `_ANEPerformanceStats` on the private API path (must use factory methods; `alloc/init` may return `nil`).
- [ ] Export per-eval `hwExecutionTimeUS` (and any other useful fields) to CSV alongside wall-time segments.
- [ ] Verify: wall-time ~= hw-time + overhead; quantify overhead and whether it changes with each optimization.

### C) IOSurface Copy / Coherency Experiments (Fix fp16 handoff Regression)
- [ ] Implement multiple FP16 handoff strategies in `SurfaceIO`:
  - [ ] Strategy 1: lock src, memcpy -> temp, unlock; lock dst, memcpy temp -> dst, unlock.
  - [ ] Strategy 2: reverse lock order (dst then src) and benchmark.
  - [ ] Strategy 3: avoid holding 2 surface locks simultaneously (if current impl does).
- [ ] Add ANE hardware correctness test: fp16 handoff strategies match CPU round-trip (tight tolerances).
- [ ] Benchmark each strategy with `--profile-kernels` and compare `*_eval_us` vs baseline.

### D) Kernel Dispatch Overhead: Zero-Gap Chaining / Realtime Eval
- [ ] Prototype chaining (attn -> ffn) using `_ANEChainingRequest` or equivalent if exposed.
- [ ] Prototype realtime evaluation path (if available) to reduce scheduler overhead.
- [ ] Benchmark: measure `gap_attn_to_ffn_us` and driver overhead deltas.

### E) Kernel Fusion (Reduce Dispatch Count)
- [ ] Attempt a single fused MIL program for one transformer layer (attn + ffn).
- [ ] If compile rejects full fusion, bisect: fuse (attn block) and/or (ffn block) further and measure dispatch counts.

### F) Automated Tuning Harness
- [ ] Add a small runner that sweeps knobs (handoff strategy, queue depth, chaining on/off, realtime on/off) and writes a `best_of.json` + CSVs.
- [ ] Run nightly-style sweep locally for 30-60 mins to find best stable config.

### G) Theoretical Bound / Reality Check
- [ ] Write `docs/ane_roofline.md`: MACs + bytes moved for this layer; estimate lower bound latency.
- [ ] Decide: is “10x faster than Core ML” physically possible for the identical workload? If not, document the max-possible gap and why.

### Phase 9 Review (Fill In As We Iterate)
- [ ] Best observed inference median/mean vs Core ML (include absolute ms, speedup, utilization).
- [ ] Bottleneck summary (top 3): what is structural vs fixable.
