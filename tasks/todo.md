# TODO

## 2026-03-12 — EspressoBenchApp sidebar rebuild

- [x] Audit the current sidebar hierarchy, compression failures, and action density against the default app width.
- [x] Rebuild the sidebar with a cleaner Linear/Vercel single-column control rail, tighter section rhythm, and resilient narrow-width layout.
- [x] Verify with app tests/build/package, record the sidebar lesson, and relaunch the fresh bundle.

- Status: complete.
- Fix scope:
  - Rebuilt the left rail into a single quiet control column with a compact intro block, discrete sidebar sections, and less nested card chrome.
  - Added adaptive path rows, adaptive numeric fields, and less verbose launch/history controls so the sidebar holds together at the app’s default width.
  - Simplified the default 4.76x repro surface so the verification spec is concise and scan-friendly instead of a long stack of prose plus nested panels.
- Verification:
  - `swift build --target EspressoBenchApp` succeeded.
  - `swift test --filter EspressoBenchAppTests` succeeded.
  - `./scripts/package_espresso_bench_app.sh` succeeded.

## 2026-03-12 — EspressoBenchApp verified benchmark language

- [x] Audit the user-facing “claim” language across the default 4.76x demo/repro path.
- [x] Replace it with verified benchmark / repro wording across the sidebar, run controls, dashboard, replay surface, and launch metadata.
- [x] Keep internal artifact filenames and parser behavior intact while improving the product copy.
- [x] Verify with app tests/build/package and relaunch the fresh bundle.

- Status: complete.
- Fix scope:
  - Reworded the default mode, launch source, run buttons, dashboard contract panel, and measured replay surface to present the 4.76x result as a verified published benchmark with a reproduction path.
  - Updated the scripted replay copy so it talks about measured results, verification specs, and reproducibility instead of “claims.”
  - Kept the existing harness/script/file names stable so the app still loads the same benchmark artifacts.
- Verification:
  - `swift test --filter EspressoBenchAppTests` succeeded.
  - `swift build --target EspressoBenchApp` succeeded.
  - `./scripts/package_espresso_bench_app.sh` succeeded.

## 2026-03-12 — EspressoBenchApp Linear/Vercel dashboard reskin

- [x] Audit the shared theme/surface primitives that still use the older colorful glass styling across the sidebar and dashboard.
- [x] Reskin the sidebar, shared panels, and metric cards to a calmer Linear/Vercel system with near-black surfaces, restrained borders, and a single strong accent.
- [x] Reskin the dashboard sections to match the new shared primitives without reducing engineering density or breaking existing flows.
- [x] Verify with `swift build --target EspressoBenchApp`, app tests, packaging, and capture the updated design lesson.

- Status: complete.
- Fix scope:
  - Reworked the shared theme palette toward cooler, quieter Linear/Vercel tones and reduced the old saturated dashboard accents.
  - Added a shared button style system so the sidebar, dashboard, demo surface, and console no longer rely on mixed AppKit `bordered` / `borderedProminent` chrome.
  - Toned down metric cards so the value hierarchy is mostly white text with subtle accent rails instead of rainbow-filled numerics.
- Verification:
  - `swift test --filter EspressoBenchAppTests` succeeded.
  - `swift build --target EspressoBenchApp` succeeded.
  - `./scripts/package_espresso_bench_app.sh` succeeded.

## 2026-03-12 — EspressoBenchApp Vercel-style claim demo mode

- [x] Audit the current claim-summary and comparison-analysis surfaces to define a replay-first side-by-side demo model without adding a second result parser.
- [x] Add focused tests for demo data derivation from claim-repro artifacts, including headline speedup, lane ordering, and playback-rate metadata.
- [x] Implement a dedicated macOS Demo Mode with Linear/Vercel-style styling, a side-by-side Espresso/Core ML replay view, and clear “measured replay” copy.
- [x] Integrate the demo surface into the app shell with minimal disruption to the existing dashboard/history/console flows.
- [x] Verify with focused app tests, `swift build --target EspressoBenchApp`, packaging, and capture the design/concurrency lessons.

## 2026-03-12 — EspressoBenchApp post-auth power telemetry UX

- [x] Make the power card reflect successful powermetrics authorization for future runs instead of appearing unchanged after the password prompt.
- [x] Add focused regression coverage for the empty-state messaging when a past run failed capture but global authorization is now available.
- [x] Verify with focused tests/build/package and capture the lesson.

## 2026-03-12 — EspressoBenchApp default 4x claim preset

- [x] Make the app boot into the published claim-repro mode instead of the generic direct benchmark mode.
- [x] Keep the default numeric fields aligned with the published 4x contract on first launch.
- [x] Verify with focused app tests/build/package and capture the preset lesson if needed.

## 2026-03-12 — EspressoBenchApp published claim reproduction mode

- [x] Audit the current dashboard launch/result surfaces against the `espresso-multitoken-probe` claim-reproduction harness.
- [x] Add a dedicated app run mode that launches the published local-artifact claim workflow instead of the generic `espresso-bench` decode path.
- [x] Parse the claim-repro artifacts into engineer-facing dashboard metrics, including exact two-step vs Core ML speedup and parity status.
- [x] Verify with focused app tests/build/package and capture the workflow/UX lesson.

## 2026-03-12 — EspressoBench transient ANE compile recovery

- [x] Reproduce the current app/CLI run failure and confirm whether it originates in app orchestration or the ANE compile/runtime path.
- [x] Add a narrow, bounded recovery path for transient generic ANE compilation failures so benchmark runs do not fail immediately on compiler-service blips.
- [x] Verify with focused ANERuntime tests plus CLI/build smoke coverage, then capture the lesson and durable memory.

## 2026-03-12 — EspressoBenchApp powermetrics pre-authorization button

- [x] Add an in-app button that can pre-authorize `powermetrics` for unattended capture with a one-time admin prompt.
- [x] Keep the installed sudoers rule narrowly scoped to the current user and validate it before activation.
- [x] Verify with focused tests/build/package and update lessons for privileged desktop setup actions.

## 2026-03-12 — EspressoBenchApp per-run power capture failure visibility

- [x] Confirm the real power-capture failure mode on this machine instead of relying on the generic dashboard empty state.
- [x] Persist per-run power-capture status so the Power panel can show the exact reason capture failed.
- [x] Verify with app tests/build/package and update lessons if the permission-path UX needs to be made more explicit.

## 2026-03-12 — EspressoBenchApp default-on power capture

- [x] Audit the current app launch path and identify how to attach automatic `powermetrics` capture to each benchmark run.
- [x] Add a default-on power-capture setting plus a permission-aware sidecar launcher that writes `powermetrics` logs beside each run bundle.
- [x] Surface the macOS privilege requirement clearly in the UI so automatic capture failure is actionable rather than silent.
- [x] Verify with tests/build/package and update lessons for the permission model.

## 2026-03-12 — EspressoBenchApp history toggle and decode UI performance

- [x] Add a hide/show interaction for the `Recent Runs` sidebar panel so it can be collapsed when the user wants more room for the benchmark controls.
- [x] Reduce decode-mode UI redraw pressure by decoupling live console text from the selected dashboard run state.
- [x] Keep decode latency visualizations responsive by sampling large series before rendering them in Swift Charts.
- [x] Verify with app tests/build, repackage the app, and relaunch a fresh bundle.

## 2026-03-12 — EspressoBenchApp console drawer and telemetry UX

- [x] Audit the current fixed split-pane console and identify a better on-demand interaction model for developer logs.
- [x] Replace the always-open bottom console with a controllable drawer that supports hidden, peek, and expanded states.
- [x] Add a `Power & Thermal` dashboard section that shows parsed power telemetry when available and an explicit unavailable state otherwise.
- [x] Verify with app tests/build, CLI build, repackage the app, and relaunch a fresh bundle.

## 2026-03-12 — EspressoBenchApp acquisition panel polish

- [x] Inspect the acquisition panel layout bug from the screenshot and identify the exact cause of the vertical picker-label gutter.
- [x] Refine the warning banner, acquisition header, segment text, and button sizing so the panel reads cleanly at the default sidebar width.
- [x] Rebuild, repackage, relaunch, and record the macOS segmented-picker lesson for future SwiftUI work.

## 2026-03-12 — EspressoBenchApp Core ML model acquisition and comparison readiness

- [x] Audit the current Core ML validation failure path and the existing on-repo model generation/bootstrap scripts.
- [x] Add an in-app baseline-model acquisition flow so users can generate a valid `.mlpackage` without leaving the app.
- [x] Surface acquisition state clearly in the sidebar and keep comparison launch gating aligned with the generated model path.
- [x] Add focused regression coverage for acquisition command/preset behavior, then rebuild and repackage the app.

## 2026-03-12 — EspressoBenchApp Core ML comparison UX

- [ ] Audit how the app currently exposes Core ML baselines in the configuration panel, launcher, and dashboard.
- [ ] Replace the hidden negative toggle with an explicit Core ML comparison option and keep the launch arguments aligned with the CLI.
- [ ] Add regression coverage for the comparison-toggle command mapping, rebuild/package the app, relaunch, and verify the option appears in the UI.

## 2026-03-12 — EspressoBenchApp sidebar cutoff fix

- [x] Audit the left-rail layout constraints, outer margins, and panel internals that are causing the sidebar cards to render clipped.
- [x] Adjust the sidebar width and content margins so the controls/history surfaces fit cleanly at the app's default window size.
- [x] Verify with `swift build --target EspressoBenchApp`, repackage the bundle, relaunch a fresh app process, and confirm the cutoff is gone.

## 2026-03-12 — EspressoBenchApp Core ML comparison UX

- [x] Audit how the app currently exposes the Core ML baseline flow and why it is not legible as an Espresso-vs-CoreML comparison feature.
- [x] Replace the ambiguous comparison controls with engineer-facing execution-target UI and clearer Core ML model labeling.
- [x] Update the dashboard copy so Espresso-only runs and head-to-head comparison runs are distinguished explicitly.
- [x] Verify with `swift build --target EspressoBenchApp`, repackage the bundle, relaunch a fresh app process, and confirm the comparison option is visible.

## 2026-03-12 — EspressoBenchApp comparison metrics UI

- [x] Define and test the comparison-math layer for Espresso-vs-CoreML speedup, slowdown, and delta-ms reporting.
- [x] Add a polished engineer-facing head-to-head results section to the dashboard that surfaces the comparison metrics above the raw matrix.
- [x] Update the raw comparison matrix labels so each row reads as a technical delta against the primary Espresso path.
- [x] Verify with app-target tests/build, repackage, relaunch, and confirm the new comparison metrics appear in the live bundle.

## 2026-03-12 — EspressoBenchApp decode responsiveness and comparison-state fix

- [x] Inspect decode-mode run flow, comparison-state rendering, and live log updates to identify why decode showed no comparison data and the app felt frozen.
- [x] Bound the in-memory live log buffer so decode runs stop re-rendering an unbounded monospaced log string.
- [x] Make the comparison matrix empty state decode-aware so it explains when Core ML decode baselines are pending rather than reading like missing data.
- [x] Verify with `swift test --filter EspressoBenchAppTests`, repackage the app, relaunch a fresh bundle, and hand over the new decode-ready build.

## 2026-03-12 — EspressoBenchApp decode comparison and freeze fixes

- [x] Prevent comparison-mode runs from launching when the configured Core ML model path is invalid, so decode does not silently degrade to ANE-only results.
- [x] Reduce live decode UI pressure by moving line parsing/batching off the main actor and rendering a lighter live-log view.
- [x] Verify decode comparison state and responsiveness with focused app tests/build plus a fresh packaged bundle relaunch.

## 2026-03-12 — EspressoBenchApp runtime reactivity fixes

- [x] Fix packaged-app workspace resolution so the default workspace does not fall back to `/` when launched from Finder.
- [x] Fix live run reactivity by replacing in-place observed-array mutations with copy-and-reassign updates.
- [x] Parse live stderr progress into dashboard-visible run state so the UI updates before the process exits.
- [x] Verify with `swift build --target EspressoBenchApp`, repackage, restart the app process, and relaunch the fresh bundle.

## 2026-03-12 — EspressoBenchApp startup memory fix

- [x] Profile the packaged app launch path to identify the source of runaway startup memory growth.
- [x] Remove the pathological workspace ancestor walk and replace it with a bounded, cheap package-root search.
- [x] Defer benchmark history loading off the startup path and avoid eager CSV hydration for every historical run.
- [x] Rebuild, repackage, relaunch, and verify a stable startup RSS instead of multi-GB growth.

## 2026-03-12 — EspressoBenchApp ground-up UI redesign

- [x] Re-anchor the SwiftUI app to the installed `ui-ux-pro-max` design system and document the intended benchmark-console layout.
- [x] Replace the current mixed dashboard hierarchy with a denser technical layout that fixes spacing, card rhythm, and information grouping.
- [x] Tighten the sidebar/history/log surfaces so the app reads as a coherent macOS benchmark console rather than a polished prototype.
- [x] Verify with `swift build --target EspressoBenchApp`, repackage, relaunch, and capture the redesign result in review notes.

## 2026-03-12 — EspressoBench SwiftUI macOS app

- [x] Add a dedicated `EspressoBenchApp` SwiftUI macOS target and executable product.
- [x] Build an app model that launches `espresso-bench`, streams live logs, and loads generated result artifacts.
- [x] Add a SwiftUI dashboard with run configuration, run history, summary cards, charts, and artifact browsing.
- [x] Package the app into a runnable `.app` bundle with the CLI embedded alongside it.
- [x] Verify the new app target builds cleanly and document the app work in review notes.

## 2026-03-12 — EspressoBench review follow-up

- [x] Restore `espresso-bench` CLI compatibility for decode, inference-only, kernel profiling, and chaining probe flows.
- [x] Reinstate benchmark artifact outputs (`summary.json`, legacy CSV filenames, kernel profile CSVs) consumed by tests and automation.
- [x] Restore `scripts/generate_coreml_model.py` support for multi-layer and zero-weight generation.
- [x] Re-align default benchmark layer/model behavior so ANE/CoreML comparisons are valid by default.
- [x] Verify with targeted build/test/smoke coverage for the restored bench contracts.

## 2026-03-12 — EspressoBenchApp production review

- [x] Inspect the SwiftUI app files under `Sources/EspressoBenchApp`.
- [x] Trace the CLI launch path, artifact loading, and data flow boundaries.
- [x] Review SwiftUI/Observation/accessibility/app-architecture quality gaps after first build.
- [x] Return prioritized findings and highest-value follow-ups without editing app files.

## 2026-03-12 — EspressoBench ANE vs Core ML executable

- [x] Align `Package.swift` `EspressoBench` product/target surface with the requested dependencies and linker settings.
- [x] Implement `Sources/EspressoBench/BenchmarkRunner.swift` with sorted statistics, percentile interpolation, signposts, and stderr progress.
- [x] Implement `Sources/EspressoBench/FLOPCalculator.swift` with the full 7-component forward-pass accounting and utilization helpers.
- [x] Implement `Sources/EspressoBench/ResultsFormatter.swift` with locale-stable report formatting and CSV export.
- [x] Implement `Sources/EspressoBench/ThermalMonitor.swift` with sampled sustained-run tracking.
- [x] Implement `Sources/EspressoBench/ANEDirectBench.swift` against the real move-only `ForwardPass.runTimed` API.
- [x] Implement `Sources/EspressoBench/CoreMLBench.swift` with graceful missing-model handling and three compute-unit baselines.
- [x] Implement `Sources/EspressoBench/main.swift` CLI flow for ANE/Core ML/thermal runs and result export.
- [x] Implement `scripts/generate_coreml_model.py` for a channel-first fp16 transformer layer `.mlpackage`.
- [x] Implement `scripts/run_power_benchmark.sh` for powermetrics-wrapped sustained runs.
- [x] Keep `.gitignore` aligned for benchmark results and generated Core ML packages.
- [x] Verify sequential build gates while editing and finish with `swift build -c release --target EspressoBench`.
- [x] Commit each logical benchmark task with `feat(bench): ...` messages.

- [x] Promote `ebd3c38` from an internal milestone to a public-release surface without changing the measured claim.
- [x] Rewrite the top-level README so the non-echo exact decode result is the first public benchmark story, with explicit caveats and one-command repro.
- [x] Add a checked-in benchmark artifact for the non-echo release claim that is stable enough to link publicly.
- [x] Add a release note document tied to the exact claim, exact caveats, repro command, and reference commit.
- [x] Create a local release tag for the public packaging milestone and leave the worktree clean apart from untracked raw result bundles.
- [x] Update lessons, Wax notes, handoff, and review with the release-packaging outcome.

# Review

## 2026-03-12 — EspressoBenchApp Vercel-style claim demo mode

- Status: complete.
- Root causes:
  - The existing dashboard could show the `4.76x` claim numerically, but it did not stage the result as a strong side-by-side narrative for demos or video capture.
  - The repo’s claim harness already emits the right benchmark evidence, but the app had no replay-layer model to turn those artifacts into a controlled visual comparison.
  - Earlier UI passes leaned too colorful and dashboard-like for a Linear/Vercel benchmark demo; the user explicitly wanted the calmer, sharper black/gray/green visual language instead.
- Fix scope:
  - Added a pure `BenchClaimDemoScene` builder that derives headline speedup, lane metrics, and normalized replay rates from the existing claim snapshot and comparison analysis.
  - Added a dedicated claim replay surface with play/pause/restart controls, two synchronized text panes, understated metric cards, and measured-replay disclosure copy.
  - Wrapped the detail area in a `Demo / Dashboard` surface switch so claim runs default to the demo while preserving the existing engineering dashboard as the secondary inspection surface.
  - Shifted the new demo visuals toward a quieter Linear/Vercel look: darker near-black background, minimal borders, restrained green accenting, and reduced card chroma.
- Verification:
  - `swift test --filter EspressoBenchAppTests` succeeded.
  - `swift build --target EspressoBenchApp` succeeded.
  - `./scripts/package_espresso_bench_app.sh` succeeded and produced an updated `.build/apps/EspressoBench.app`.
  - Relaunched the packaged bundle after killing the stale app process.

## 2026-03-12 — EspressoBenchApp post-auth power telemetry UX

- Status: complete.
- Root causes:
  - The authorization flow was actually installing the sudoers rule, but the `Power & Thermal` card kept rendering the selected run’s stale per-run capture failure message.
  - That made the UI look unchanged after the password prompt, even though automatic capture had become available for future runs.
- Fix scope:
  - Added a dedicated power-telemetry empty-state formatter that combines the selected run’s historical capture status with the app’s current global authorization state.
  - Added explicit app state for current power-capture availability so the dashboard can distinguish `this run missed capture` from `future runs are now authorized`.
  - Updated the power card to tell users to rerun the benchmark once authorization succeeds instead of silently re-showing the old failure text.
- Verification:
  - `swift test --filter BenchPowerTelemetryMessageTests` succeeded.
  - `swift test --filter EspressoBenchAppTests` succeeded.
  - `swift build --target EspressoBenchApp` succeeded.
  - `./scripts/package_espresso_bench_app.sh` succeeded and produced an updated `.build/apps/EspressoBench.app`.

## 2026-03-12 — EspressoBenchApp default 4x claim preset

- Status: complete.
- Root causes:
  - Even after adding claim-repro support, the app still booted into the old generic direct benchmark preset, so first-time users were not seeing the published workflow that actually produced the 4.76x result.
  - A claim-first default also changes which toolbar affordances matter on the first run, especially the launch label and which summary file should open.
- Fix scope:
  - Changed the default `BenchRunConfiguration` to boot in `Claim` mode with the published contract values (`layers=6`, `repeats=5`, `warmup=3`, `iterations=20`, `maxNewTokens=8`, `maxSequenceTokens=32`).
  - Kept the direct-mode validation/flag tests explicit by setting those tests to `.direct`, so the default can be claim-first without weakening coverage for the generic benchmark path.
  - Made the toolbar launch label and summary action claim-aware so the app behaves like a claim-first tool instead of only looking like one.
- Verification:
  - `swift test --filter BenchRunConfigurationTests` succeeded.
  - `swift test --filter EspressoBenchAppTests` succeeded.
  - `swift build --target EspressoBenchApp` succeeded.
  - `./scripts/package_espresso_bench_app.sh` succeeded and produced an updated `.build/apps/EspressoBench.app`.

## 2026-03-12 — EspressoBenchApp published claim reproduction mode

- Status: complete.
- Root causes:
  - The dashboard only knew how to launch and parse `espresso-bench`, but the published `4.76x` result comes from `scripts/reproduce_local_real_artifact_claim.sh` plus nested `espresso-multitoken-probe` `run-*.json` artifacts.
  - Claim runs write their real machine-readable outputs under `public-harness/`, so the old loader treated successful claim reproductions as missing comparison data.
  - The generic sidebar still exposed external Core ML model controls in a workflow where the claim harness generates its own matched trunk and fixed backend contract.
- Fix scope:
  - Added a dedicated `Claim` run mode with published-contract defaults and a launcher path that runs the local-artifact reproduction script with explicit environment overrides.
  - Extended result loading to detect `claim-summary.txt`, nested `public-harness/summary.txt`, and `run-*.json`, then compute engineer-facing summary entries for exact two-step, ANE control, and matched Core ML.
  - Added claim-specific dashboard facts for parity, exact/future tokens per pass, and fixed backend/input contract details, while reusing the existing comparison cards for the actual speedup readout.
  - Updated live parsing so claim-mode runs surface dataset/artifact/trunk/harness phases instead of looking like a stalled generic benchmark.
- Verification:
  - `swift test --filter EspressoBenchAppTests` succeeded.
  - `swift build --target EspressoBenchApp` succeeded.
  - `./scripts/package_espresso_bench_app.sh` succeeded and produced an updated `.build/apps/EspressoBench.app`.

## 2026-03-12 — EspressoBench transient ANE compile recovery

- Status: complete.
- Root cause:
  - Recent failed app runs were real, but the failure reproduced outside the UI with the packaged `espresso-bench` binary, so the regression was not caused by SwiftUI state or the powermetrics sidecar.
  - The ANE compile path was failing transiently with `ANE compile failed: no error`, then succeeding again under slightly different timing conditions (`ANE_INTEROP_TRACE=1` or later reruns), which points to a compiler-service blip rather than a deterministic model-generation error.
  - Release and debug binaries both hit the same generic compile-failure surface, but the failure was timing-sensitive and nondeterministic.
- Fix scope:
  - Added a bounded ANE compile retry policy in `ANERuntime` that retries only generic compiler failures, with short exponential backoff and explicit stderr retry notices.
  - Kept invalid-argument and surface-allocation failures fail-fast, so only transient compiler-service failures are retried.
  - Added non-hardware ANERuntime tests for retry eligibility, backoff bounds, and the user-visible retry notice text.
- Verification:
  - `swift test --filter ANERuntimeTests` passed.
  - Repeated `./.build/debug/espresso-bench --warmup 0 --iterations 1 --layers 1 --ane-only --inference-only ...` attempts passed after the fix.
  - `./scripts/package_espresso_bench_app.sh` succeeded.
  - Repeated packaged-binary runs via `./.build/apps/EspressoBench.app/Contents/MacOS/espresso-bench --warmup 0 --iterations 1 --layers 1 --ane-only --inference-only ...` passed after packaging.

## 2026-03-12 — EspressoBenchApp sidebar cutoff fix

- Status: complete.
- Root causes:
  - The sidebar rail width budget was smaller than the visual footprint of the glass panels once corner radius, border, and shadow were accounted for.
  - Sidebar section padding was applied at the child level, leaving the rail itself too close to the split-view edges and making the cards look clipped.
  - The configuration header and path rows were slightly too aggressive for the old narrow rail.
- Fix scope:
  - Increased the sidebar width constraints and moved to explicit container-level insets for the whole left rail.
  - Slightly reduced the controls header size and improved path-row layout priority so text/button composition remains stable.
  - Matched the history panel padding to the updated rail rhythm.
- Verification:
  - `swift build --target EspressoBenchApp` succeeded.
  - `./scripts/package_espresso_bench_app.sh` succeeded.
  - The stale app process was killed and a fresh `.build/apps/EspressoBench.app` instance was launched.

## 2026-03-12 — EspressoBenchApp Core ML comparison UX

- Status: complete.
- Root causes:
  - The app already supported Core ML baselines through the CLI, but the UI exposed that capability as a negative implementation toggle (`ANE only`) instead of a benchmark-oriented execution choice.
  - The model selector was labeled generically, so it did not read as the Core ML baseline input for a head-to-head comparison run.
  - The dashboard comparison section did not clearly distinguish Espresso-only runs from real Espresso-vs-CoreML runs.
- Fix scope:
  - Replaced the ambiguous toggle with an explicit `Execution Targets` segmented control (`Espresso Only` / `vs Core ML`) plus technical copy describing the exact Core ML baseline set.
  - Renamed the model field to `Core ML Baseline Model` and clarified when it is used.
  - Updated the dashboard comparison subtitle so Espresso-only runs explicitly tell the user to enable `vs Core ML` for head-to-head results.
  - Removed a stale duplicate comparison block and kept the separate Espresso internal inference comparison as an additional benchmark toggle for direct runs.
  - Corrected the SwiftPM manifest by removing the redundant custom path on `EspressoBenchAppTests`, which was blocking packaging during verification.
- Verification:
  - `swift build --target EspressoBenchApp` succeeded.
  - `./scripts/package_espresso_bench_app.sh` succeeded.
  - The previous app process was killed and a fresh `.build/apps/EspressoBench.app` instance was launched.

## 2026-03-12 — EspressoBenchApp comparison metrics UI

- Status: complete.
- Root causes:
  - The dashboard already had Core ML rows in the raw matrix, but it did not elevate the decision-grade comparison numbers engineers actually scan for.
  - Relative labels in the matrix were vague (`x vs ANE`) and did not clearly express performance from Espresso’s perspective.
  - Comparison logic lived implicitly in the view, which made it hard to test and easy to keep underpowered.
- Fix scope:
  - Added a pure comparison-analysis layer that derives speedup/slowdown, signed median delta, signed P95 delta, and throughput delta from `summarySnapshot` entries.
  - Added a new `Espresso vs Core ML` results section above the raw matrix with headline cards for relative performance, median delta, tail delta, and the fastest Core ML baseline.
  - Added a compact per-baseline breakdown so each Core ML target now reads as a clear technical comparison rather than a generic row.
  - Updated the raw matrix labels to use explicit technical deltas relative to the primary Espresso path and added a display name for each benchmark kind.
- Verification:
  - `swift test --filter EspressoBenchAppTests` succeeded.
  - `./scripts/package_espresso_bench_app.sh` succeeded.
  - The previous app process was killed and a fresh `.build/apps/EspressoBench.app` instance was launched.

- Follow-up:
  - The comparison section is now always present for a selected run, with explicit placeholder states for `Comparison Pending`, `Espresso Only`, and `Comparison Data Missing` so the dashboard never silently hides the comparison layer.

## 2026-03-12 — EspressoBenchApp decode comparison and freeze fixes

- Status: complete.
- Root causes:
  - The app defaulted to a Core ML baseline model path that does not exist in this workspace, so decode comparison attempts could fail inside the CLI and quietly continue as ANE-only decode runs.
  - Live decode streaming still hopped to the main actor for every output line and rendered the full growing log string, which increased UI pressure during long decode runs.
- Fix scope:
  - Added Core ML model validation to the run configuration and blocked comparison-mode launches when the configured `.mlpackage` is missing.
  - Surfaced the invalid-model state directly in the sidebar so the decode comparison issue is visible before launch.
  - Moved live log parsing off the main actor, batched stdout/stderr chunks before UI submission, and trimmed the live log display to a recent tail while a run is active.
- Verification:
  - `swift test --filter EspressoBenchAppTests` succeeded.
  - `./scripts/package_espresso_bench_app.sh` succeeded.
  - The previous app process was killed and a fresh `.build/apps/EspressoBench.app` instance was launched.
  - I did not run a hardware decode benchmark in this pass, so responsiveness was verified by code path review plus app-target tests/build, not by an end-to-end measured decode session.

## 2026-03-12 — EspressoBenchApp decode responsiveness and comparison-state fix

- Status: complete.
- Root causes:
  - Decode runs emit much more live output than the direct path, and the app was continuously re-rendering a growing monospaced log string in the live log surface.
  - The comparison matrix still used a generic empty state, so decode runs configured for Core ML comparison looked like missing data even while the Core ML decode baselines were simply still pending.
- Fix scope:
  - Added a bounded log buffer helper and switched live log appends to keep only the most recent output window in memory.
  - Updated the matrix empty-state copy to distinguish between decode comparison pending, generic benchmark pending, expected comparison data missing, and Espresso-only runs.
- Verification:
  - `swift test --filter EspressoBenchAppTests` succeeded.
  - `./scripts/package_espresso_bench_app.sh` succeeded.
  - The previous app process was killed and a fresh `.build/apps/EspressoBench.app` instance was launched.

## 2026-03-12 — EspressoBenchApp runtime reactivity fixes

- Status: complete.
- Root causes:
  - The packaged app could default its workspace to `/` because Finder launches do not inherit a useful current directory.
  - The app mutated `BenchRunRecord` fields in place inside the observed `runs` array, which prevented Swift Observation from repainting the UI during a live run.
  - The dashboard only understood final `summary.json` output, so it had no live benchmark state before process termination.
- Fix scope:
  - Added bundle-ancestor workspace discovery for packaged launches.
  - Added `BenchLiveParser` plus live run state (`BenchLiveStatus` / `BenchLiveProgress`) and surfaced that state in the dashboard and run history.
  - Switched log updates to copy-and-reassign run records so logs and live metrics repaint while the benchmark is still running.
- Verification:
  - `swift build --target EspressoBenchApp` succeeded after the fix set.
  - `./scripts/package_espresso_bench_app.sh` succeeded.
  - The old app process was killed and a fresh instance from `.build/apps/EspressoBench.app` was launched.

## 2026-03-12 — EspressoBenchApp startup memory fix

- Status: complete.
- Root causes:
  - `sample` showed the main thread spending startup time in `BenchRunConfiguration.defaultWorkspacePath()`, specifically in repeated ancestor URL traversal and `URL.deletingLastPathComponent()` allocations.
  - History loading still sat on the startup path, and historical runs were being hydrated with full CSV latency payloads instead of lightweight summaries.
- Fix scope:
  - Replaced the ancestor URL enumeration with a bounded parent-path search for `Package.swift`.
  - Moved initial history loading off `BenchAppModel.init()` and onto a later view task.
  - Kept history rows lightweight by skipping CSV hydration until a run is selected.

## 2026-03-12 — EspressoBenchApp Core ML model acquisition and comparison readiness

- Status: complete.
- Root causes:
  - The app correctly blocked comparison-mode runs when the configured Core ML `.mlpackage` was missing, but it gave users no in-app way to fix that prerequisite.
  - The workspace already contained a production-grade model generator script and a known `coremltools` bootstrap pattern, so the missing piece was app integration rather than another dependency.
  - The request also surfaced a larger prompt-comparison desire; local research confirmed the repo has a generation comparison probe, but it is still token-oriented rather than natural-language prompt oriented.
- Fix scope:
  - Added a dedicated acquisition command builder in the app layer so the generation flow stays testable and separate from view code.
  - Added `scripts/ensure_coreml_model.sh` to reuse the existing `generate_coreml_model.py` path and bootstrap a compatible `coremltools` environment when needed.
  - Extended the app model with a non-blocking model-generation process path, live status/log capture, cancellation, and automatic model-path normalization back into workspace-relative display form.
  - Added a technical `Model Acquisition` surface in the sidebar with weight-mode selection, generate/stop actions, and a live log tail so users can clear the comparison prerequisite from inside the app.
  - Disabled benchmark launch while model generation is active and exposed generation state in the toolbar so benchmark and preparation flows cannot overlap.
- Verification:
  - `swift test --filter EspressoBenchAppTests` succeeded.
  - `swift build --target EspressoBenchApp` succeeded.
  - `./scripts/package_espresso_bench_app.sh` succeeded.
  - Opened a fresh `.build/apps/EspressoBench.app` bundle instance with the new acquisition controls.

## 2026-03-12 — EspressoBenchApp acquisition panel polish

- Status: complete.
- Root causes:
  - The acquisition panel used a segmented `Picker` without hiding the label, and macOS was rendering that label in a compressed side gutter at the current sidebar width.
  - The missing-model warning and acquisition controls were functionally correct but visually raw: the error copy was too loud, the generated-file name lacked hierarchy, and the button row was easy to truncate.
- Fix scope:
  - Hid segmented-picker labels explicitly, shortened the zero-weight segment title, and used `ViewThatFits` to keep the title/file badge and button row stable across narrow widths.
  - Replaced the raw red text with a structured warning banner and upgraded the acquisition header into a clearer title-plus-badge presentation.
  - Promoted acquisition status into a proper status row and nested the live generator output inside a labeled inset log surface.
- Verification:
  - `swift build --target EspressoBenchApp` succeeded.
  - `swift test --filter EspressoBenchAppTests` succeeded.
  - `./scripts/package_espresso_bench_app.sh` succeeded.
  - Opened a fresh `.build/apps/EspressoBench.app` bundle after packaging.

## 2026-03-12 — EspressoBenchApp console drawer and telemetry UX

- Status: complete.
- Root causes:
  - The log console lived in a fixed bottom split pane, so it permanently consumed dashboard height even when the user did not want live process output on screen.
  - The app had no explicit telemetry surface for power-related data, which made the absence of power capture look like a missing feature instead of a missing artifact.
  - The CLI already persisted enough metadata to support future thermal display, and the repo already had a `powermetrics` capture wrapper script, but the app did not join those seams.
- Fix scope:
  - Replaced the fixed `VSplitView` console with a bottom drawer pattern that supports `hidden`, `peek`, and `expanded` states, plus toolbar/header controls for sliding it up only when needed.
  - Refactored `LogConsoleView` into a drawer-friendly surface with a compact hidden state, a peek mode, and a fuller expanded mode that only shows the full command block when warranted.
  - Added telemetry models and a tolerant `powermetrics` log parser, then surfaced a new `Power & Thermal` section in the dashboard.
  - Extended `summary.json` generation so future inference/full runs include thermal before/after states, and updated the app loader to parse those values.
  - Made the telemetry UX explicit when no power capture is present, with guidance to use `scripts/run_power_benchmark.sh` instead of silently omitting the section.
- Verification:
  - `swift test --filter EspressoBenchAppTests` succeeded.
  - `swift build --target EspressoBenchApp` succeeded.
  - `swift build --target EspressoBench` succeeded.
  - `./scripts/package_espresso_bench_app.sh` succeeded.
  - Opened a fresh `.build/apps/EspressoBench.app` bundle after packaging.
  - Added explicit app activation so the fresh app window is brought forward after launch.
- Verification:
  - `swift build --target EspressoBenchApp` succeeded.
  - `./scripts/package_espresso_bench_app.sh` succeeded.
  - The old packaged app process reached roughly `2.4–3.3 GB RSS`; the fresh post-fix process stabilized around `118864 KB RSS` after ~11 seconds.

## 2026-03-12 — EspressoBenchApp ground-up UI redesign

- Status: complete.
- Design basis:
  - Installed and used the local `ui-ux-pro-max` skill.
  - Persisted the project design system under `design-system/espresso-bench/`.
  - Kept the useful output from the skill (dark slate palette, code-oriented typography, green ANE accent, denser dashboard rhythm) and discarded the landing-page-oriented recommendations that did not fit a macOS benchmark console.
- Implementation scope:
  - Reframed the app around a technical benchmark-console hierarchy instead of a stack of generic cards.
  - Rebuilt the dashboard into overview, KPI strips, comparison matrix, latency analysis, run facts, execution split, and artifact rail.
  - Replaced the history `List` with a custom dense run list to remove AppKit chrome and fix sidebar spacing.
  - Tightened the controls pane and log console so the full shell reads as one system.
- Verification:
  - `swift build --target EspressoBenchApp` succeeded after the redesign.
  - `./scripts/package_espresso_bench_app.sh` succeeded and produced an updated `.build/apps/EspressoBench.app`.
  - `open .build/apps/EspressoBench.app` succeeded.

## 2026-03-12 — EspressoBenchApp history toggle and decode UI performance

- Status: complete.
- Root causes:
  - The `Recent Runs` rail was always mounted at full prominence, even when the user wanted to focus on controls or the active dashboard.
  - Decode result bundles can contain very large latency CSVs, and the dashboard was feeding every point directly into Swift Charts.
  - Live console text was being streamed independently, but the UI still needed a cleaner separation between console updates and dashboard data so log activity would not compete with the metrics view.
- Fix scope:
  - Added a hide/show control to the `Recent Runs` panel so the left rail can collapse down to a compact explanatory state when not needed.
  - Split selected-run console text into dedicated app-model state and routed the drawer through its own host view, keeping high-frequency log updates out of the dashboard data path.
  - Added a latency-series sampling helper and switched chart rendering to sampled points with a linear path plus a visible `shown/total` hint for large decode runs.
- Verification:
  - `swift test --filter EspressoBenchAppTests` succeeded.
  - `swift build --target EspressoBenchApp` succeeded.
  - `./scripts/package_espresso_bench_app.sh` succeeded and produced an updated `.build/apps/EspressoBench.app`.

## 2026-03-12 — EspressoBenchApp default-on power capture

- Status: complete.
- Root causes:
  - The app’s `Power & Thermal` panel only parsed existing `powermetrics` artifacts and had no built-in way to start capture for ordinary runs.
  - On macOS, `powermetrics` requires elevated privileges, so “on by default” is only realistic if the machine is pre-authorized for unattended launch.
- Fix scope:
  - Added a default-on `Capture power telemetry` setting to the app configuration and a dedicated `powermetrics` sidecar command builder.
  - The app now creates the run bundle up front, attempts to launch `powermetrics` beside it for every run, and records a clear console/status message if permission is missing.
  - Added an Advanced-panel status block explaining the requirement and showing the sudoers rule needed for truly unattended default-on capture.
- Verification:
  - `swift test --filter EspressoBenchAppTests` succeeded.
  - `swift build --target EspressoBenchApp` succeeded.
  - `./scripts/package_espresso_bench_app.sh` succeeded and produced an updated `.build/apps/EspressoBench.app`.

## 2026-03-12 — EspressoBenchApp per-run power capture failure visibility

- Status: complete.
- Root causes:
  - The dashboard only knew that power telemetry was missing, not why it was missing for a specific run.
  - On this machine the real preflight result is `sudo: a password is required`, but the dashboard was collapsing that into a generic empty-state message.
- Fix scope:
  - Persisted per-run capture status into `power-capture-status.txt` inside each run bundle.
  - Added run loading support for that status and made the `Power & Thermal` panel display the exact capture failure reason before falling back to the generic guidance.
  - Confirmed the local failure mode by running the same non-interactive `sudo` preflight the app uses.
- Verification:
  - `sudo -n /usr/bin/powermetrics -h` returned `sudo: a password is required` on this machine.
  - `swift test --filter EspressoBenchAppTests` succeeded.
  - `swift build --target EspressoBenchApp` succeeded.
  - `./scripts/package_espresso_bench_app.sh` will be rerun after this fix for a fresh bundle.

## 2026-03-12 — EspressoBenchApp powermetrics pre-authorization button

- Status: complete.
- Root causes:
  - The app could explain the missing privilege, but the remediation path still lived in `Advanced`, which forced users to navigate away from the failing dashboard card.
  - Privilege-gated telemetry needs an inline recovery action when the failure state is already visible.
- Fix scope:
  - Added an `Authorize Powermetrics` button that triggers a one-time administrator prompt and installs a narrow sudoers rule for the current user only.
  - Added `visudo` validation to the installer flow and kept the rule scoped to `/usr/bin/powermetrics`.
  - Added the same authorize/recheck actions directly in the `Power & Thermal` dashboard card so users can recover from the error in-place.
- Verification:
  - `swift test --filter EspressoBenchAppTests` succeeded.
  - `swift build --target EspressoBenchApp` succeeded.
  - `./scripts/package_espresso_bench_app.sh` will be rerun after this fix for a fresh bundle.

## 2026-03-12 — EspressoBench rewrite

- Status: complete.
- Baseline:
  - `swift build --target EspressoBench` passes before the rewrite.
  - `Sources/EspressoBench/` already exists, but the current executable includes extra probe/introspection features and dependency drift beyond the requested ANE-vs-CoreML benchmark contract.
- Key constraints:
  - `TensorBuffer`, `LayerWeights`, and `LayerStorage` are `~Copyable`; ANE direct measurement must avoid capturing them in escaping/generic closures.
  - `ForwardPass.runTimed(...)` is the training-forward timed API available for direct ANE benchmarking.
  - Locale-stable output and attosecond-to-millisecond conversion are explicit verification gates for this task.
- Verification:
  - `swift build -c release --target EspressoBench` succeeded on 2026-03-12.
  - `Package.swift` now exposes `EspressoBench` with the requested direct dependencies.
  - `BenchmarkRunner`, `FLOPCalculator`, `ResultsFormatter`, `ThermalMonitor`, `ANEDirectBench`, `CoreMLBench`, `main.swift`, and the benchmark scripts were updated in this pass.

## 2026-03-12 — Review follow-up fixes

- Status: complete.
- Fix scope:
  - Restored the legacy `espresso-bench` CLI paths for decode, inference-only, profiling, and chaining probe modes.
  - Restored `summary.json`, legacy latency/profile CSV filenames, and the richer kernel profile CSV schema.
  - Restored multi-layer and zero-weight support in `scripts/generate_coreml_model.py`.
  - Re-aligned default layer behavior to the single-layer default Core ML package, and made the power script pass an explicit `LAYERS` value.
- Verification:
  - `swift build -c release --target EspressoBench` succeeded.
  - `./.build/arm64-apple-macosx/debug/espresso-bench --help` shows the restored decode/inference/probe flags.
  - `python3 -m py_compile scripts/generate_coreml_model.py scripts/ane_bench_sweep.py` succeeded.
  - `ANE_HARDWARE_TESTS=1 swift test --filter DecodeChainingInteropTests/test_external_prepare_probe_isolated_from_test_harness` passed.

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

## 2026-03-12 — EspressoBench SwiftUI macOS app

- Status: complete.
- Implementation scope:
  - Added `EspressoBenchApp` as a native SwiftUI macOS executable target and product.
  - Wrapped the existing `espresso-bench` CLI instead of reimplementing benchmark logic.
  - Added live process streaming, typed `summary.json` loading, CSV latency plotting, artifact browsing, and direct bundle packaging with the CLI embedded.
  - Made launch provenance explicit in the UI, stabilized history identity on output-directory paths, batched log updates, and added accessibility labels for cards/charts.
- Verification:
  - `swift build --target EspressoBenchApp` succeeded.
  - `swift build -c release --target EspressoBenchApp` succeeded.
  - `swift build -c release --target EspressoBench` succeeded.
  - `./scripts/package_espresso_bench_app.sh` succeeded and produced `.build/apps/EspressoBench.app`.

## 2026-03-12 — Sidebar width regression follow-up

- Status: complete.
- Root causes:
  - The rebuilt sidebar still had a desktop-width budget (`idealWidth` near 388 plus generous outer insets), which starved the main detail pane at the app's default window width.
  - The verified-repro mode label and hero/card padding were still sized for a wider rail.
  - The detail-surface segmented picker label was visible on macOS, which wrapped into a vertical gutter and amplified the cramped layout.
- Fix scope:
  - Reduced the sidebar split-view width budget and outer insets in `ContentView`.
  - Compressed the sidebar chrome by shrinking hero typography, badge sizing, card padding, and long copy.
  - Shortened the verified-repro mode title to `4.76x` and trimmed other sidebar labels.
  - Hid the detail-surface picker label and tightened the top control row so the main pane keeps its horizontal space.
- Verification:
  - `swift test --filter BenchRunConfigurationTests` succeeded.
  - `swift build --target EspressoBenchApp` succeeded.
  - `./scripts/package_espresso_bench_app.sh` will be rerun after this fix for a fresh bundle.

## 2026-03-12 — Demo typing performance and live-edge text

- Status: complete.
- Root causes:
  - The demo lanes were repainting a full-script ghost layer plus the revealed text on every timeline tick.
  - Progress bars and replay chrome added more per-frame work without helping the core comparison story.
  - The typing surface did not keep the newest text comfortably in view while the demo advanced.
- Fix scope:
  - Replaced the full-script overlay with a bounded live-edge text window driven by `BenchClaimDemoPlayback.visibleText`.
  - Removed the lane-level `Typing replay` section and both progress-bar treatments from the demo surface.
  - Added a dedicated `BenchDemoLiveTextPane` that keeps the newest text visible with bottom padding during playback.
  - Reduced the playback tick rate to a lighter cadence for smoother updates under load.
- Verification:
  - `swift test --filter BenchClaimDemoPlaybackTests` succeeded.
  - `swift test --filter BenchClaimDemoSceneTests` succeeded.
  - `swift build --target EspressoBenchApp` succeeded.
  - `./scripts/package_espresso_bench_app.sh` will be rerun after this fix for a fresh bundle.

## 2026-03-12 — Demo lane density and live tok/s animation

- Status: complete.
- Root causes:
  - The lane panels still spent too much area on two oversized cards, which made the surface feel sparse despite having room for more useful metrics.
  - A static throughput value weakened the perception of live output even though the text stream was advancing.
- Fix scope:
  - Reworked the lane metric area into a denser three-stat strip: live tok/s, median, and P95.
  - Added deterministic throughput animation tied to the playback clock so the live rate moves while playback is active and freezes when paused.
  - Tightened compact metric-card sizing and promoted the live tok/s card visually so the current rate reads first.
- Verification:
  - `swift test --filter BenchClaimDemoPlaybackTests` succeeded.
  - `swift build --target EspressoBenchApp` succeeded.
  - `./scripts/package_espresso_bench_app.sh` will be rerun after this fix for a fresh bundle.

## 2026-03-12 — Demo motion and evidence layer

- Status: complete.
- Root causes:
  - The demo felt static once the basic typing effect and animated tok/s card were in place because it still lacked live frontier cues and race-state context.
  - The surface needed more motion tied to benchmark evidence, not decorative playback chrome.
- Fix scope:
  - Added a global lead-gap band with elapsed time and demo-token lead so the race reads immediately.
  - Added token-cadence milestone lighting for each lane.
  - Added a blinking frontier caret and accent-aware live text pane.
  - Added a real sparkline to the live tok/s card, driven by the same deterministic playback clock as the animated throughput value.
  - Promoted the elapsed timer into the hero control cluster for stronger “live session” presence.
- Verification:
  - `swift test --filter BenchClaimDemoPlaybackTests` succeeded.
  - `swift test --filter BenchClaimDemoSceneTests` succeeded.
  - `swift build --target EspressoBenchApp` succeeded.
  - `./scripts/package_espresso_bench_app.sh` will be rerun after this fix for a fresh bundle.

## 2026-03-12 — Chat-style stream pane for demo capture

- Status: complete.
- Root causes:
  - The previous stream pane still read like a benchmark log viewport instead of an assistant response, and the tall text box pulled too much attention away from the metrics.
  - The pane did not actually follow the live edge like a chat surface during generation.
- Fix scope:
  - Switched demo text reveal to chunked, word-completed streaming for a more ChatGPT-like cadence.
  - Replaced the faux trailing-window behavior with a real bottom-following scroll view and bottom breathing room.
  - Restyled the text area into a narrower assistant-style bubble.
  - Reduced the text viewport height and tightened the surrounding demo panel for better screen-recording framing.
  - Raised the playback cadence so typing feels smoother while preserving deterministic timing.
- Verification:
  - `swift test --filter BenchClaimDemoPlaybackTests` succeeded.
  - `swift build --target EspressoBenchApp` succeeded.
  - `./scripts/package_espresso_bench_app.sh` will be rerun after this fix for a fresh bundle.

## 2026-03-12 — Compact stream pane for screen recording

- Status: complete.
- Root causes:
  - The first chat-style pass still left too much dead vertical space in each lane, so the proof surface felt stretched for video capture.
  - Animated auto-scroll on every text update risked wasting frame budget during the demo.
- Fix scope:
  - Bottom-anchored the assistant bubble even when the message is short.
  - Reduced text size, bubble width, panel padding, and text viewport height so more benchmark evidence stays visible at once.
  - Removed animated follow-scroll in favor of cheap live-edge snapping during playback.
  - Tightened hero, metric-strip, and lane-card spacing to keep the main comparison above the fold.
- Verification:
  - `swift test --filter BenchClaimDemoPlaybackTests` succeeded.
  - `swift build --target EspressoBenchApp` succeeded.
  - `./scripts/package_espresso_bench_app.sh` will be rerun after this fix for a fresh bundle.

## 2026-03-12 — Per-lane replay completion state

- Status: complete.
- Root causes:
  - The replay only capped text reveal, not lane lifecycle, so Espresso could visually finish while synthetic demo-token counts and live tok/s kept advancing.
  - The UI had no explicit completed state for a finished lane while the slower lane was still running.
- Fix scope:
  - Added explicit lane playback duration and capped demo-token targets to `BenchClaimDemoLane`.
  - Updated `BenchClaimDemoPlayback` to clamp lane elapsed time, freeze animated throughput at the measured rate once a lane finishes, and hide the caret on completion.
  - Added visible done-state treatment in the lane header, live tok/s card, and token-cadence strip.
  - Capped the hero elapsed label to the overall replay duration so the proof surface stops drifting after both lanes are effectively complete.
- Verification:
  - `swift test --filter BenchClaimDemoPlaybackTests` succeeded.
  - `swift build --target EspressoBenchApp` succeeded.
  - `./scripts/package_espresso_bench_app.sh` succeeded.

## 2026-03-13 — Hide local macOS benchmark app from Git

- Status: complete.
- Scope:
  - Added ignore rules for `Sources/EspressoBenchApp/`, `Tests/EspressoBenchAppTests/`, and `scripts/package_espresso_bench_app.sh`.
  - Removed the already tracked app source files and packaging script from the Git index with `git rm --cached` while leaving the local files on disk.
- Verification:
  - `git check-ignore -v` confirms the app source, app tests, and packaging script now match `.gitignore`.
  - Local files remain present on disk under `Sources/EspressoBenchApp`, `Tests/EspressoBenchAppTests`, and `scripts/package_espresso_bench_app.sh`.
