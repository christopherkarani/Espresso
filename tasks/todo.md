# TODO

- [ ] Preserve the failed local real-artifact hardware route only as docs/results/memory; keep the implementation clean.
- [ ] Re-run the exact synthetic `echo` harness with repeated fresh-process medians and verify parity plus `committed_exact_tokens/pass > 1`.
- [ ] Package the synthetic `echo` result as the only publishable `>= 3x` claim surface on this branch, with explicit separation from the weaker `recurrent-checkpoint` route.
- [ ] If the refreshed `echo` median drops below `3x`, try exactly one bounded performance hypothesis on the winning exact path and remeasure.
- [ ] Update docs, lessons, Wax notes, handoff, and review with the final claim scope and every rejected avenue.

# Review

- Current strongest honest positive result on this branch is still the exact synthetic `echo` same-session harness at about `3.6986x` over matched CoreML, with parity `match`, `committed_exact_tokens/pass = 2`, and `accepted_future_tokens/pass = 1`.
- The local real-artifact avenue is now a strong negative result, not a publishable claim: the ANE recurrent hardware path outputs all-zero tokens on focused one-hot seams and on the local bigram artifact, while the direct ANE local-artifact spike also collapses to zeros and lands at about `0.98x` vs CoreML.
