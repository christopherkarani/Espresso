# Lessons Learned

## 2026-03-04 — Phase 7 Golden Fixtures, Cross-Validation, Archival
- Cross-validation fixtures must be generated from ObjC kernels, not from Swift outputs; Swift-vs-Swift fixture generation is circular and invalid as an oracle.
- When archival changes source locations (`training` -> `archive/training`), all build/capture scripts must resolve both active and archived paths to avoid silently breaking maintenance workflows.
- Golden capture scripts should write atomically via temp files and only replace existing fixtures with non-empty output; probe crashes can otherwise clobber known-good goldens.
- Golden capture scripts should be strict-by-default: any probe/generator failure must return non-zero so CI/users never get a false "goldens refreshed" signal.
- ANE compile-time assertions are sensitive to host load; never run multiple hardware test suites in parallel when compile-latency thresholds are part of the test contract.
- Keep `training/golden_outputs` stable after archival; tests should depend on committed golden artifacts only, not on re-running unstable ObjC probes during test execution.

## 2026-03-03 — CPUOps Phase 4 hardening
- Add explicit bounds contracts on pointer-indexed APIs (`targets`, `tokens`) before merge. Relying on implicit array/pointer traps is not an acceptable API contract.
- For numerical ops, add at least one direct parity test against an independent scalar reference, not only algebraic invariants (for example, RoPE invertibility alone is insufficient).
- Ensure tests named "forward/backward consistency" exercise both paths directly; derivative-only checks can leave forward path regressions undetected.
- Use non-uniform seeded fixtures for accumulation/indexing tests so layout/index bugs cannot hide behind symmetric data.

## 2026-03-03 — Phase 5a runtime review follow-up
- For C interop APIs with ambiguous `NULL` returns, always expose a machine-readable error code so Swift can preserve typed error semantics instead of collapsing to a generic failure.
- Avoid test skip logic that can mask behavior regressions; if host capability is unstable, assert that runtime behavior matches a same-run baseline rather than unconditional `XCTSkip`.
- Public runtime APIs should avoid `precondition` for recoverable misuse (for example, index validation); return typed errors and cover those paths with explicit tests.

## 2026-03-04 — Phase 6 Parity + Integration/Perf Tests
- When matching C `drand48()`-based init, reproduce both RNG draw ordering (interleaved loops) and C promotion semantics (compute in Double, cast once) for bit-identical parity.
- For long-running ANE + external-process tests, gate with `ANE_HARDWARE_TESTS=1` and an explicit opt-in env var (for example, `ESPRESSO_INTEGRATION_TESTS=1`) to keep default `swift test` fast and reliable.
- Treat warnings as failures in test helpers too; avoid deprecated APIs (for example, `String(cString:)`).

## 2026-03-04 — Review Fixes (RMSNorm, Exec Restart, Weight Header)
- For normalization layers with learnable weights, finite-difference checks must use non-unit weights; unit-weight fixtures can mask whole classes of gradient bugs.
- Don’t “parity test” against a reference implementation unless the reference is independently validated; otherwise you can lock in a wrong derivative and still get green tests.
- Any codebase that uses `exec*` restarts must set `FD_CLOEXEC` (or `O_CLOEXEC`) on long-lived fds (datasets, logs, etc.), because Swift deinits won’t run across exec.
- Exec-restart should preserve full CLI state and must not assume `argv[0]` contains a `/`; prefer dyld-based executable resolution and/or `execvp` for PATH semantics.
- Binary weight loaders must validate header fields that size allocations (at minimum vocab magnitude, seqLen, heads) before reading payload to avoid truncated/OOB failures later.

## 2026-03-04 — Phase 6b Kernel/Hardware Verification
- Keep cross-validation tests behind a composable gate tier (`ANE_HARDWARE_TESTS=1` + `OBJC_CROSS_VALIDATION=1`) so default `swift test` remains deterministic and fast.
- Backward IOSurface chain invariants must be asserted explicitly: `dv` is sourced from `sdpaBwd1` output offset `0`, while `dq|dk` are sourced from `sdpaBwd2` output offset `0`.
- `tiny_train.m` is useful only for exec-restart/checkpoint lifecycle contract checks; it is not numerically comparable to `espresso-train` loss because it is a different model.
- Host-dependent gates (M4 perf target, local model assets, ANE interop instability) should be documented as explicit blocked conditions in checklist/todo artifacts, not silently treated as pass.
