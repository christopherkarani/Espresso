# TODO

- [x] Promote `ebd3c38` from an internal milestone to a public-release surface without changing the measured claim.
- [x] Rewrite the top-level README so the non-echo exact decode result is the first public benchmark story, with explicit caveats and one-command repro.
- [x] Add a checked-in benchmark artifact for the non-echo release claim that is stable enough to link publicly.
- [x] Add a release note document tied to the exact claim, exact caveats, repro command, and reference commit.
- [x] Create a local release tag for the public packaging milestone and leave the worktree clean apart from untracked raw result bundles.
- [x] Update lessons, Wax notes, handoff, and review with the release-packaging outcome.

# Review

- Current code/control milestone is `ebd3c38`:
  - exact two-step `1.0806302083333332 ms/token`
  - exact one-token ANE control `1.0957500000000002 ms/token`
  - matched zero-weight `6`-layer CoreML `5.085307291666668 ms/token`
  - exact two-step speedup vs CoreML `4.7583224488025415x`
  - exact one-token ANE control speedup vs CoreML `4.640428016426192x`
  - parity `match`
  - committed exact tokens/pass `2`
  - accepted future tokens/pass `1`
- The code/result is frozen enough for recovery, but the repo is not yet public-release quality:
  - README now leads with the new non-echo decode claim
  - checked-in benchmark artifacts now exist under `artifacts/benchmarks/exact-decode-non-echo/`
  - release notes now exist under `docs/releases/2026-03-11-non-echo-exact-decode.md`
  - the remaining packaging step is to tag and push the milestone, not to invent more prose
- The public claim must stay constrained:
  - non-echo local artifact family
  - exact parity preserved
  - explicit `identity-zero-trunk` backend
  - not a pretrained production checkpoint claim
