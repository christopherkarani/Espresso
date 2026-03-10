# TODO

- [ ] Re-establish the exact single-stream control on this worktree: fused-triplet recurrent direct-select with fused ANE RMSNorm+classifier head, lane `32`, and matched hardware benchmark settings (`warmup=3`, `iterations=20`, `maxNewTokens=8`).
- [x] Add failing unit tests for the recurrent-native `k=2` exact pass contract: prefix-only acceptance, exact committed-token accounting, verifier correction without replay/prefill reset, reusable state-snapshot reporting, and state-discard on rejection.
- [x] Add a failing hardware benchmark seam that reports per-pass medians for proposer, verifier trunk, verifier logits, state advance, `accepted_exact_tokens/pass`, `committed_exact_tokens/pass`, and net effective `ms/token`.
- [x] Implement the Stage 1 upper-bound exact `k=2` architecture using the proven recurrent control plus a materially different verifier/state-reuse contract; stop immediately if the same-run control still wins or committed exact tokens per pass stay near `1`.
- [x] Replace the replay-heavy state-advance path with a materially new two-step branch-state-promotion architecture that prepares `stateMid` and `stateOut` in one recurrent pass and promotes the accepted state without a second recurrent decode in the model path.
- [ ] Implement the Stage 2 real recurrent future-token proposer only if Stage 1 materially beats the same-run exact control; keep exactness prefix-only and do not present upper-bound acceptance as real-model acceptance.
- [x] Append findings to `docs/fused-decode-and-next-steps.md`, capture review notes below, flush/update Wax memory, and revert any dead-end code before finalizing.

# Review

- `swift build --build-tests` succeeded after fixing stale SwiftPM test-target dependencies (`MILGeneratorTests -> ANERuntime`, `ANERuntimeTests -> Espresso`) and unblocked filtered package test execution in this worktree.
- `swift test --skip-build --filter GenerationHarnessTests` passed, including the new exact `k=2` accounting/state-cost tests.
- `swift test --skip-build --filter RWKVStyleTwoStepRecurrentGeneratorTests` passed, covering the new three-input/four-output two-step recurrent MIL contract.
- The earlier Stage 1 upper-bound seam remains rejected: it only reached a second committed token by paying a second `decodeSelectedToken` call, recorded as `stateAdvanceLatencyMs`.
- The current avenue is materially different in code structure: the new two-step recurrent path prepares `stateMid` and `stateOut` in one recurrent pass and promotes one of those prepared states on commit instead of issuing a second recurrent decode in the model path.
- Same-session hardware benchmarking still did not produce throughput numbers. With `ANE_HARDWARE_TESTS=1`, the committed hardware seam stalled before timed iterations while compiling the single-layer recurrent control inside `ANERecurrentGenerationModel.compileSingleLayerSessions` -> `RWKVStyleRecurrentKernelSet.compileStep` -> `ANEKernel.init` -> `_ANEClient compileModel`.
- Stage 2 was not started. The reusable state-promotion path is now implemented, but it remains measurement-blocked until the same-session exact control compiles and runs cleanly enough to produce medians.
