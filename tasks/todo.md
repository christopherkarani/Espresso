# TODO

- [ ] Re-establish the exact single-stream control on this worktree: fused-triplet recurrent direct-select with fused ANE RMSNorm+classifier head, lane `32`, and matched hardware benchmark settings (`warmup=3`, `iterations=20`, `maxNewTokens=8`).
- [x] Add failing unit tests for the recurrent-native `k=2` exact pass contract: prefix-only acceptance, exact committed-token accounting, verifier correction without replay/prefill reset, reusable state-snapshot reporting, and state-discard on rejection.
- [x] Add a failing hardware benchmark seam that reports per-pass medians for proposer, verifier trunk, verifier logits, state advance, `accepted_exact_tokens/pass`, `committed_exact_tokens/pass`, and net effective `ms/token`.
- [x] Implement the Stage 1 upper-bound exact `k=2` architecture using the proven recurrent control plus a materially different verifier/state-reuse contract; stop immediately if the same-run control still wins or committed exact tokens per pass stay near `1`.
- [ ] Implement the Stage 2 real recurrent future-token proposer only if Stage 1 materially beats the same-run exact control; keep exactness prefix-only and do not present upper-bound acceptance as real-model acceptance.
- [x] Append findings to `docs/fused-decode-and-next-steps.md`, capture review notes below, flush/update Wax memory, and revert any dead-end code before finalizing.

# Review

- `swift build --build-tests` succeeded and unblocked filtered package test execution in this worktree.
- `swift test --skip-build --filter GenerationHarnessTests` passed, including the new exact `k=2` accounting/state-cost tests.
- The Stage 1 upper-bound implementation is intentionally honest: a second committed exact token is only obtained by paying a second `decodeSelectedToken` call, recorded as `stateAdvanceLatencyMs`.
- Same-session hardware benchmarking did not produce throughput numbers. The run stalled for more than three minutes while compiling the fused-triplet recurrent control inside `_ANEClient compileModel`, so the exact control could not be re-established on this worktree.
- Stage 2 was not started. Without reusable multi-token state advancement, the current Stage 1 seam does not raise the credible ceiling beyond one expensive recurrent step per committed token.
