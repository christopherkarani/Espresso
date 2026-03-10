# TODO

- [x] Preserve `3e6cced` as the exact two-step architecture checkpoint, `2e49cab` as the compile/init gate evidence, and `13c688b` as the student-sidecar artifact checkpoint.
- [x] Keep `f710d13`, `6188715`, and `2c864c2` as the recoverable release-probe, fused-pair, and fused-triplet exact multi-token breakthroughs.
- [x] Head-batching hypothesis: add failing tests for pair-slice ANE output-head I/O and exact-two-step batched verifier selection before implementation.
- [x] Implement one-ANE-eval batched RMSNorm+classifier selection for the prepared activation pair on the fused-triplet exact two-step path.
- [x] Rebuild and rerun the standalone release probe against the exact fused-triplet control; report compile/init, exact parity, committed exact tokens/pass, accepted future tokens/pass, proposer, verifier trunk, verifier logits, state advance, and effective `ms/token`.
- [x] Kill the head-batching route immediately if it does not move the exact fused-triplet path clearly toward the `4x` window (`<= 1.761196 ms/token` against the looser CoreML control, `<= 1.645556 ms/token` against the standing baseline).
- [ ] If more upside is needed beyond the new exact `4x` win, measure follow-on work against the new head-batched control instead of reopening the old pre-batching baselines.
- [ ] Keep attempt logging exhaustive in `docs/fused-decode-and-next-steps.md`, update review notes below after every major result, write Wax session + durable notes after each confirmed result, hand off at major checkpoints, and flush immediately.

# Review

- Open-ended throughput search resumed from `feat/ane-multitoken` after the student-sidecar checkpoint.
- The previous hardware gate result remains: both compile/init-only seams failed to reach first output within roughly `45s`, so no honest same-session medians were reported from that bounded pass.
- The student-route artifact seam is now in place: `TwoStepStudentCheckpoint`, focused tests passed, and `espresso-train --export-two-step-student` writes a recoverable sidecar artifact without changing the base checkpoint format.
- The new standalone release probe recovered hardware truth outside `xctest`: exact parity held on every measured comparison, `committed_exact_tokens/pass` stayed at `2.0`, and `accepted_future_tokens/pass` stayed at `1.0` on the echo checkpoint family.
- Compile/init is not a hard deadlock in the fresh probe: one 6-layer compile/init-only run measured control `36625.966 ms` versus two-step `812.478 ms`.
- Repeated 1-layer matched-control runs all favored exact two-step: control `1.452750`, `1.768331`, `1.788609 ms/token`; two-step `1.354299`, `1.419352`, `1.484302 ms/token`.
- The deeper scaling boundary is still negative on the current checkpoint family: 2-layer fused-pair was noisy and centered slightly behind control, while 3-layer fused-triplet lost in both repeats and 6-layer fused-triplet lost in the initial run.
- Fusing the exact two-step trunk into pair sessions materially changed the ceiling: repeated 2-layer fused-pair runs now favored exact two-step (`1.534096`, `1.556641 ms/token`) over the matched fused-pair control (`2.124839`, `1.679589 ms/token`) with exact parity and `2.0` committed exact tokens/pass.
- The pair-fused win extends through 4 layers: control `2.195484`, `2.334737 ms/token` versus fused-pair two-step `2.149909`, `2.234477 ms/token`.
- The 6-layer gap narrowed sharply but did not close yet: fused-triplet control `2.146151`, `2.293576 ms/token` versus 6-layer two-step built from fused pairs `2.317794`, `2.529677 ms/token`.
- Extending the same idea to fused triplets closed that remaining gap: repeated 6-layer exact runs now favored fused-triplet two-step (`2.197565`, `2.176102 ms/token`) over the strong fused-triplet control (`2.616013`, `2.397878 ms/token`) with exact parity and `2.0` committed exact tokens/pass.
- Batched verifier-head eval is the next exact breakthrough: compile/init-only release probe measured control `811.378125 ms` versus two-step `900.605500 ms`, and repeated 6-layer exact runs measured control `2.786466`, `2.339284`, `2.243776 ms/token` versus batched-head two-step `2.409169`, `1.365719`, `1.564339 ms/token`.
- Exact parity stayed `match` on all three batched-head runs, `committed_exact_tokens/pass` stayed `2.0`, `accepted_future_tokens/pass` stayed `1.0`, and the three-run medians (`2.339284` control vs `1.564339` two-step) clear `4x` over the standing `6.582224 ms/token` CoreML baseline.
