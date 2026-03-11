# TODO

- [x] Preserve the current synthetic `echo` `>= 3x` claim as the control and do not weaken its exactness or reporting contract.
- [x] Add a minimal non-echo hardware correctness seam that must emit a nonzero token and must match a CPU teacher before any new throughput rerun is trusted.
- [x] Isolate the root cause of the local non-echo zero-output failure across recurrent trunk, serialized weights, and direct ANE output-head paths.
- [x] Fix the smallest proven bug and rerun the non-echo hardware correctness seam until the ANE path is functionally valid.
- [x] If the non-echo seam turns green, rerun the matched recurrent-checkpoint harness with repeated medians and exact parity requirements.
- [x] Update docs, lessons, Wax notes, handoff, and review with the stronger claim if it exists, or with a sharply measured negative result if it does not.

# Review

- The exact synthetic `echo` control remains intact at `3.5383781787824304x` and `3.7377633193960738x` over matched CoreML with parity `match`, `committed_exact_tokens/pass = 2`, and `accepted_future_tokens/pass = 1`.
- The generic RWKV-style recurrent ANE cell is still invalid on a non-echo one-hot seam: raw `xIn` writes peaked at token `35` with value `1.0`, but both `xOut` and `stateOut` stayed all zero after eval, so that route remains a documented negative result.
- The stronger non-echo claim now exists on an exact local bigram artifact with an explicit `identity-zero-trunk` backend:
  - focused hardware parity tests matched the CPU teacher for both one-token and exact two-token generation on prompt `35`
  - offline gate: parity `match`, `committed exact tokens/pass = 2`, `accepted future tokens/pass = 1`
  - one-command wrapper (`scripts/reproduce_local_real_artifact_claim.sh`) reran the matched recurrent-checkpoint/CoreML harness at:
    - exact two-step `1.2012578125 ms/token`
    - exact one-token ANE control `1.0598854166666667 ms/token`
    - matched zero-weight `6`-layer CoreML trunk `4.7705807291666664 ms/token`
    - exact two-step speedup vs CoreML `3.9541428963040195x`
    - parity `match`
- Important nuance: on this non-echo artifact family the exact two-step path is now publishably `>= 3x` over matched CoreML, but it is still slower than the one-token ANE identity control because proposer cost remains CPU-side.
