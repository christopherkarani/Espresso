# TODO

## 2026-03-13 autoresearch-multistream review fixes

- [x] Replace incorrect grouped-weight prefix slicing with a shared row-aware grouped repacker.
- [x] Remove the stale `RWKVStyleFusedSixLayerKernelSet` compile probe from hardware tests without losing the intended compile-sweep coverage.
- [x] Harden `SurfaceIO`/ANE interop APIs with vocab, shape, and IOSurface allocation validation.
- [x] Harden the Metal expansion/argmax fast paths against invalid IOSurface sizes and `UInt16` token-id overflow.
- [x] Verify with targeted `swift build` and the previously failing filtered hardware test compile path.

- [x] Promote `ebd3c38` from an internal milestone to a public-release surface without changing the measured claim.
- [x] Rewrite the top-level README so the non-echo exact decode result is the first public benchmark story, with explicit caveats and one-command repro.
- [x] Add a checked-in benchmark artifact for the non-echo release claim that is stable enough to link publicly.
- [x] Add a release note document tied to the exact claim, exact caveats, repro command, and reference commit.
- [x] Create a local release tag for the public packaging milestone and leave the worktree clean apart from untracked raw result bundles.
- [x] Update lessons, Wax notes, handoff, and review with the release-packaging outcome.

# Review

## 2026-03-13 autoresearch-multistream review fixes

- Implemented:
  - grouped dense row-major weights now repack per output-row group instead of truncating a global prefix
  - stale six-layer compile probe replaced with two existing triplet compile probes
  - `writeEmbeddingBatchFP16` now carries `vocabSize` explicitly and rejects out-of-range token IDs in C
  - lockless argmax and fused expansion argmax now validate IOSurface bounds and `UInt16` index width
  - `ane_interop_rebind_input` now rebuilds requests transactionally and rejects undersized replacement surfaces
  - Metal fast paths now validate IOSurface allocation sizes before `memcpy` / `bytesNoCopy`
- Verification:
  - `swift build` in `/tmp/espresso-fix-autoresearch`: passed
  - `swift build --target ANERuntimeTests` in `/tmp/espresso-fix-autoresearch`: passed
  - `swift test --filter GenerationHarnessHardwareTests/test_batched_multistream_scaling_reports_aggregate_throughput_on_hardware` now gets past the missing `RWKVStyleFusedSixLayerKernelSet` symbol, but the `EspressoTests` target still ends in a pre-existing compiler `fatalError` after a large volume of Swift 6 Sendable-capture diagnostics from `GenerationHarnessHardwareTests.swift`

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
- README hardening pass:
  - lead now scopes the performance number to the reproducible non-echo local-artifact benchmark
  - repro notes now state that first-run `coremltools` bootstrap may occur
  - public copy now avoids broader "CoreML in general" wording
