# Lessons Learned

## 2026-03-04 — Phase 8 Benchmarking + Gates
- Keep cross-validation strictness split explicit: required probes must fail the run, optional probes may warn but must never mask required artifact failures.
- Emit benchmark metrics from tests in machine-readable JSON (`PHASE8_BENCHMARKS=1`) so grading/parity logic does not depend on brittle text parsing.
- Benchmark reports must always include timestamp + git SHA (or `no git repo`) in every artifact format (`json`, `csv`, `md`) to preserve auditability.
- If performance cannot be measured (host/repo limitation), mark perf as `N/A`, redistribute score weights proportionally, and attach exact blocker evidence; never silently treat missing perf data as pass.
- For Python orchestration that shells out to Swift tools, either route execution through an allowed context (elevated run) or force local cache/module paths to avoid environment-dependent failures.

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

## 2026-03-06 — Experiment Tracking Discipline
- Use Wax MCP continuously during long ANE tuning loops: record the baseline marker, active clean worktree/branch, each experiment hypothesis, artifact directories, verdicts, and rollback notes as the work progresses.
- Treat Wax memory as part of the performance workflow, not an optional afterthought; if the user corrects memory/experiment tracking behavior, update both Wax and `tasks/lessons.md` immediately.
- When a build, test, or benchmark run hits a blocker, record the failure mode, current hypothesis, and next action in Wax session memory before continuing. The issue trail matters as much as the winning runs.

## 2026-03-10 — Wax Checkpoint Discipline
- At every major phase boundary, do all four Wax actions together: remember the key finding, write a handoff checkpoint, flush immediately, and keep a final promotion step reserved for durable confirmed results.
- Do not wait until the end of the session to create handoffs; phase checkpoints are part of the execution loop for long-running ANE experiments.

## 2026-03-06 — Performance Reporting Discipline
- Every tuning cycle must document three things explicitly: what was tried, what worked, and what did not work.
- Always explain why a change is believed to have helped, regressed, or failed to confirm; artifact-backed attribution is part of the deliverable, not optional commentary.
- Do not discard confirmed sub-1% performance improvements as “too small” without documenting them. If a small gain survives repeated confirmation and tails stay flat, commit it, document it, and then continue iterating from the improved baseline.

## 2026-03-06 — Breakthrough-Only Tuning
- When the user allows more tuning time, treat that as conditional, not open-ended: spend extra cycles only when the current evidence suggests a real architectural or scaling breakthrough rather than local noise.
- Before extending an experiment, state the breakthrough hypothesis, the concrete metric that would validate it, and the stop condition that ends the tuning loop.
- Use a scientific-critique frame for performance claims: reject time sinks driven by anecdote, uncontrolled comparisons, or metrics that cannot distinguish a real scaling advantage from benchmark variance.

## 2026-03-10 — Commit Cadence
- When the user asks for frequent commits, checkpoint smaller validated milestones instead of batching multiple tuning steps into one large uncommitted change.
- For ANE performance work, prefer commits at green boundaries: tests passing, baseline captured, or one measured avenue wired end-to-end.

## 2026-03-10 — Wax Note Format
- After every major phase, write the session note with the exact branch name and avenue name using the format: `branch`, `avenue`, `outcome`, `measurements`, `next`.
- Promote any confirmed result immediately into durable Wax memory with the same five-field format; do not leave confirmed findings in session-only notes.
- Before any pause or handoff point, write a Wax handoff for project `Espresso`, include pending tasks, then flush right away.
- Make the branch marker recoverable from Wax by always starting the note with the exact `branch:` line and flushing immediately after the write.

## 2026-03-10 — Verification Plumbing
- Keep SwiftPM test-target dependencies aligned with actual test imports; stale dependency lists can create false `no such module` failures and block focused verification of unrelated ANE work.
- For hardware-gated generation tests, run the committed XCTest seam with `ANE_HARDWARE_TESTS=1` before trusting any disposable probe; if the run stalls, sample the test process and record the exact compile stack instead of inferring the blocker.
- When hardware truth is blocked before first output, split the seam into compile/init-only control and compile/init-only candidate tests before touching runtime metrics; this cleanly separates ANE compile stalls from decode/runtime regressions.

## 2026-03-10 — Student Pivot Packaging
- When a new decode contract needs trainable artifacts but runtime truth is blocked, add a separate sidecar format instead of extending the base checkpoint/resume blob; future-head artifacts should version independently from optimizer-resume state.
- Seed new auxiliary heads from the teacher weights that already define the contract entrypoint (`rmsFinal` and classifier/embedding) so export produces a recoverable training start, not an empty placeholder.

## 2026-03-10 — Throughput Search Persistence
- When the user explicitly asks for an open-ended throughput hunt, do not treat a bounded checkpoint or one failed gate as a stopping point; convert immediately into an iterative search loop with longer-budget probes, new avenues, and exhaustive attempt logging.
- A failed fast-path benchmark timeout is evidence about that timeout budget, not necessarily about the architecture ceiling; before pivoting permanently, try at least one lower-overhead or longer-budget measurement route and document the result.

## 2026-03-10 — Matched-Control Scaling Ladder
- A shallow exact throughput win is not enough; map the same architecture against the 1-layer, 2-layer, 3-layer, and strong 6-layer controls before calling it a scalable breakthrough.
- When the sign of a hardware result is close or flips across runs, repeat the full benchmark at the same iteration count and report the repeated medians instead of the best-looking run.
- If `xctest` blocks hardware truth, move immediately to a lower-overhead fresh-process probe so compile/init latency and runtime throughput can be separated honestly.

## 2026-03-10 — Mirror The Winning Control Path
- State reuse alone was not enough; the exact two-step path only started winning beyond 1 layer after its verifier trunk inherited the same pair fusion structure that already made the control fast.
- When the control path has a known fusion win, test the exact multi-token path with the same grouping strategy before assuming the architecture itself has plateaued.

## 2026-03-10 — Batch Prepared Activations Before Reopening Training
- Once exact two-step state reuse is working, check whether the verifier head is still paying one ANE eval per prepared activation; batching those prepared activations through the existing spatial lanes can be a breakthrough-sized win without changing the trunk contract.
- Measure the pair-eval head path with the same exact release probe before pivoting to student training; the head boundary can still hide more upside than a new proposer.
- Treat noisy first hardware reruns as potential session outliers, not instant regressions; rerun the identical probe enough times to report repeated medians before killing a route that changed the dominant cost center.

## 2026-03-10 — Public Claim Repro Gate
- Do not compare a new ANE result to saved CoreML numbers when deciding what is publicly claimable; rerun CoreML in the same session and under the same fresh-process harness first.
- For probe-style benchmarks, synthetic shortcuts must be explicit in the CLI and in the claim text; never leave an `echo` path as an implicit default when the output may be quoted publicly.
- A one-command reproduction script should treat `.mlpackage` as a directory artifact, capture raw JSON for every run, and summarize medians from those raw artifacts rather than from copied notes.

## 2026-03-11 — Real-Artifact Benchmark Plumbing
- For benchmark-only generation artifacts, do not reuse the full pretrained-model loader when the artifact can legitimately vary layer count; add a dedicated head-weight loader that validates shape-critical fields without hardcoding `ModelConfig.nLayers`.
- When moving off synthetic prompts, the prompt token must come from the saved artifact manifest or corpus, not a hardcoded `0`; otherwise a “real-data” rerun silently falls back to an unrepresentative start token.
- CoreML export tooling for reproducible local benchmarks should pin to a supported Python version (`3.12` here); the Python `3.14` `coremltools` install lacked the native `BlobWriter`/`libcoremlpython` pieces needed to save `.mlpackage` models.

## 2026-03-11 — Synthetic Echo Can Hide Zero-Output ANE Failures
- Never trust a new ANE decode substrate just because the synthetic echo harness stays on token `0`; an all-zero output surface also argmaxes to `0` and can look superficially correct.
- Before treating any off-echo artifact benchmark as valid, add at least one hardware correctness seam that expects a nonzero token and verify the ANE path matches a CPU teacher on that exact prompt.
- When a real-artifact benchmark collapses to zeros, revert the debug code after capturing the negative result in docs/results/memory; keep the branch clean and do not let failing probes masquerade as implementation progress.

## 2026-03-11 — Zero-Trunk Artifacts Need An Explicit Exact Backend
- When the saved recurrent artifact is mathematically zero-trunk, do not keep forcing it through a broken generic recurrent kernel just because the interface exists; first prove the kernel is wrong with a raw-surface hardware probe, then add the smallest exact backend that matches the artifact contract.
- Keep that specialization explicit in the benchmark contract and CLI (`identity-zero-trunk` here); hidden auto-detection would make the public claim harder to audit.
- A stronger public story can come from a non-echo artifact family even if the multi-token path is not the fastest ANE variant, as long as the same matched harness shows exact parity and `>= 3x` over CoreML.

## 2026-03-11 — Non-Echo Two-Step Needed An ANE Proposer, Not More CPU Polishing
- Once exact non-echo acceptance is already `2` committed tokens/pass with parity `match`, stop treating proposer quality as the blocker and measure proposer placement directly; the remaining gap can be pure systems cost.
- On the `identity-zero-trunk` artifact family, moving the future proposer from CPU onto an ANE fused head was enough to turn the exact two-step path from slower-than-control into faster-than-control in the matched release harness.
- Keep proposer-only lane sweeps bounded and reversible; smaller proposer lanes (`1` and `8` here) hit ANE `statusType=0x9`, so failed geometry probes should be reverted immediately and kept only as docs/results evidence.

## 2026-03-11 — A Frozen Benchmark Is Not A Public Release
- A recoverable commit and a local result bundle are not enough for a public-quality release; the claim has to be visible at the top of the README, backed by a checked-in benchmark artifact, and tied to a release note or tag.
- Keep the public caveats adjacent to the benchmark number, not buried in a long running notebook, so external readers cannot misread the scope.
- Public benchmark artifacts should be lightweight and checked in (`json/csv/md`) even when the raw run directories stay untracked.
