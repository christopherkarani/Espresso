# TODO

- [x] Reject contiguous exact staged CPU head with measured hardware regression.
- [x] Reject clustered exact staged CPU head with measured hardware regression.
- [x] Implement and measure the `k=2` recurrent branch/commit verifier substrate.
- [x] Use a `2`-layer recurrent proposer against the `6`-layer fused-triplet verifier path.
- [x] Measure `accepted_exact_tokens/pass`, proposer, verifier trunk/logits, checkpoint-copy, and net `ms/token`.
- [x] Kill the avenue when amortization still regresses materially versus fused-triplet direct-select.
- [ ] Next avenue: only pursue a materially different multi-token architecture with real verifier amortization beyond one exact recurrent step per committed token.
- [ ] If exact-head work is revisited, require a materially different admissible geometry, not more contiguous/clustered CPU staging.
- [x] Append all attempted Avenue 2 findings to `docs/fused-decode-and-next-steps.md`.
- [x] Update Wax session memory with confirmed findings.
- [ ] Update durable Wax memory if the local long-term `waxmcp remember` path is responsive.

# Review

- Avenue 1 is closed: both contiguous and clustered exact CPU staged heads regressed.
- Avenue 2 current `k=2` branch/commit architecture is closed.
- Measurement on echo recurrent weights reached the acceptance ceiling (`accepted_exact_tokens/pass = 2.0`) and still regressed to `4.1569722222222225 ms/token` versus the same-run fused-triplet direct-select control at `2.382458333333333 ms/token`.
- Interpretation: the current branch/commit design remains structurally too sequential; even perfect acceptance does not amortize the proposer + verifier cost enough to beat the exact single-stream control.
