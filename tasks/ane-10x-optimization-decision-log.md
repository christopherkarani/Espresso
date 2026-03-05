# ANE 10x Optimization Decision Log

Date: 2026-03-06  
Repo/worktree: `/private/tmp/espresso-mainworktree-clean`  
Branch: `feat/ane-10x-max`  
Latest commit at time of this document: `13d21df`

---

## 1) Goal and constraints

Primary goal:
- Achieve large decode speedup vs Core ML baseline by using ANE-direct decode with persistent FP16 KV cache.

Secondary goal:
- Reduce/close prefill gap vs Core ML where possible.

Hard constraints observed on this host:
- Decode path currently constrained to `decode-max-seq == 32` for stable ANE eval on mixed-input decode-attention kernels.
- `_ANEPerformanceStats` request-buffer factory returns nil on this host build, so `hwExecutionTime` is currently unavailable in profiling output.

---

## 2) Measurement protocol used

Common protocol:
- Deterministic seed: `ESPRESSO_BENCH_SEED=1`
- Output artifacts: `summary.txt`, `summary.json`, CSV latency arrays, CSV kernel profiles.
- Decode metric gate: strict speedup vs fastest Core ML naive decode median.

Main benchmark commands used:
- Prefill profile:
  - `ANE_COMPILE_CACHE_POLICY=preferCached ESPRESSO_BENCH_SEED=1 .build/release/espresso-bench --perf-stats --inference-only --profile-kernels --warmup 20 --iterations 200 --output /tmp/prefill_profile_bdb0bd7`
- Decode profile:
  - `ANE_COMPILE_CACHE_POLICY=preferCached ESPRESSO_BENCH_SEED=1 .build/release/espresso-bench --decode --profile-kernels --warmup 20 --iterations 200 --decode-steps 32 --decode-max-seq 32 --output /tmp/decode_profile_bdb0bd7`

---

## 3) What was tried, why, and what happened

| ID | Change tried | Why it was tried | Evidence / artifacts | Outcome | Decision |
|---|---|---|---|---|---|
| A1 | Added FP16 spatial slice copy primitive (`copyFP16SpatialSlice`) | Needed for incremental KV-cache slice writes and decode mask updates without CPU tensor rebuilds | Commit `c510af1`; tests in `Tests/ANETypesTests/ANETypesTests.swift` | Primitive works and is used by decode runtime | **SHIP** |
| A2 | Added decode lane-packing runtime + diagnostics | Isolate why decode kernels fail and make decode contracts explicit | Commit `abff150`; probe logs under `/tmp/decode_probe_seq_*.log` and `/tmp/decode_ffn_spatial_*.log` | Found hard host constraint: decode probe passes at seq=32, fails at >32 with `statusType=0x9`; FFN path fails for spatial `<32` | **SHIP** (diagnostic + guardrails) |
| A3 | Decode-attention input contract shift to dense mask cache + separate K/V caches | Avoid failing channel-1 mask and concatenated cache input patterns | Commit `2d852c5`; decode runtime files + hardware tests | Decode became stable for supported contract (`maxSeq=32`) | **SHIP** |
| A4 | Hardware-gated decode correctness tests (KV updates + mask progression) | Ensure decode state updates are correct before perf tuning | `ANE_HARDWARE_TESTS=1 swift test --filter InferenceOptimizationTests/test_decode_kv_cache_updates_and_mask_progresses_on_hardware` | Passes | **SHIP** |
| A5 | Decode probe matrix test (`x`, `x+k`, `x+k+v`, `x+k+v+denseMask`, etc.) | Hard-proof which shape families eval/fail | `ANE_HARDWARE_TESTS=1 swift test --filter ANERuntimeTests/test_decode_probe_passthrough_4in_3out_eval` | Confirms passing/failing families; failing variants map to same inference error | **SHIP** |
| A6 | Decode CLI mode + Core ML naive decode baseline + CSV artifacts | Needed strict apples-to-apples decode throughput gate and artifact outputs | Commit `2d852c5`, `bdb0bd7`; decode summaries under `/tmp/decode_*` | Strict decode speedup demonstrated (2.67x–2.72x depending run) | **SHIP** |
| A7 | Inference kernel profile expansion (host eval, host overhead, lock/body/unlock IO, handoff split) | Needed decomposition of residual prefill gap and host overhead attribution | Commit `bdb0bd7`; profile summaries under `/tmp/prefill_profile_*` | Full breakdown available in summary/CSV/JSON | **SHIP** |
| A8 | Added run metadata JSON (`git_sha`, env, device/options) | Reproducibility requirement and machine-readable audit trail | `Sources/EspressoBench/RunMetadata.swift`; `summary.json` in output dirs | Metadata now present in decode/prefill outputs | **SHIP** |
| A9 | `_ANERequest` constructor fallback with weightsBuffer selector | Improve compatibility across private runtime selector variants | `Sources/ANEInterop/ane_interop.m`; commit lineage `abff150`, `bdb0bd7` | Kept safe fallback; no decode-failure signature change | **ITERATE** |
| A10 | Perf-stats object fallback experiments in interop | Try to recover `hwExecutionTime` data | Local experiments + tests (`ANEPerfStatsTests`) | Forcing placeholder perf-stats object caused runtime exceptions; reverted. Host still reports factory nil. | **ABANDON** (unsafe path) |
| A11 | Decode eval path sweep (`inmem`, `client`, `clientDirect`, `realtime`) | Test whether failure depended on eval path rather than kernel contract | `/tmp/decode_evalpath_*.log` | All eval paths failed identically for unsupported decode-attn contract (`statusType=0x9`) | **ABANDON** (for this failure mode) |
| A12 | Compile cache policy reproducibility check (`auto` vs `preferCached`) | Improve compile stability and iteration time | `/tmp/decode_repro_ane10x_run1`, `/tmp/decode_repro_ane10x_run2` | Median latency stable (`0.648479` vs `0.648500` ms/token; delta `0.0032%`), compile time collapsed (`49.49s` → `52.7ms`) | **SHIP** (`preferCached`) |
| A13 | Decode tile-sync optimization (boundary-only window sync + incremental lane cache updates + removed redundant lane-zero copies) | Full-window cache sync every token was dominating decode IO at `maxSeq > 32` | Commit `5c016ec`; `/tmp/decode_syncopt_before_max128_20260305`, `/tmp/decode_syncopt_after_max128_20260305`, `/tmp/decode_syncopt_before_max256_20260305`, `/tmp/decode_syncopt_after_max256_20260305` | Large, repeatable IO reduction and throughput gain for tiled decode contexts (`128/256`) while preserving hardware correctness tests | **SHIP** |
| A14 | Decode runtime option sweep on top of A13 (`queue depth`, `eval path`, `power`, `wired`, `pool`, `late latch`, `fences`, `fw signal`) | Try to stack additional constant-factor wins after tile-sync optimization | `/tmp/decode_syncopt_opts_max128_20260305`, `/tmp/decode_syncopt_confirm_evalpath_max128_20260305` | One-shot sweep suggested `ANE_EVAL_PATH=clientDirect`, but 3x confirmation showed baseline parity/no stable gain (median baseline `~0.672–0.676`, candidate `~0.677–0.679`) | **ABANDON** (no reproducible gain) |
| A15 | Prefill option combo re-check (`ANE_QUEUE_DEPTH=32` + `ANE_MEMORY_POOL_ID=1`) under sequential runs | Verify whether previous prefill gains hold under non-contented, reproducible setup | `/tmp/prefill_syncopt_baseline_seq_20260305`, `/tmp/prefill_syncopt_combo_qd32_pool1_seq_20260305` | Combo regressed vs baseline (median `1.927` vs `1.855` ms); no bottleneck improvement | **ABANDON** |
| A16 | Fused decode layer prototype (`decode_attn_probe + decode_ffn` in one kernel behind `ESPRESSO_DECODE_LAYER_MODE=fusedLayer`) | Remove one eval per layer/token and test the biggest remaining decode dispatch lever | Quick split2 baseline `/tmp/decode_fused_cycle1_split2_quick_20260306`; compile stall sample `/tmp/espresso-bench_2026-03-06_014704_vEMy.sample.txt` | Prototype built and unit tests passed, but every fused decode compile stalled inside `_ANEClient compileModel`, including a stripped-down passthrough probe. No artifact directory was produced for the fused attempts. | **ABANDON** (compile instability / unsafe) |
| A17 | Decode runtime option re-sweep at `maxSeq=32` in clean worktree (`preferCached`, `clientDirect`, queue depth, power, wired) | Re-check whether the current cool baseline still has any unclaimed runtime-option win before more invasive decode work | `/tmp/decode_sweep_max32_quick_20260306` | `preferCached` baseline remained best. Confirmation medians: baseline `0.4888/0.4941/0.4940`, `clientDirect` `0.4955/0.4954/0.4926`; median-of-medians delta `+0.27%` against `clientDirect`. `keepModelWired` regressed hard to median `0.587`. | **ABANDON** (runtime tuning exhausted for this shape) |
| A18 | Request-level decode input surface rebinding (`attnOut -> ffnIn`, `ffnOut_L -> attnIn_L+1`) with request rebuild and client refresh attempts | Lowest-risk route to remove explicit copy steps and build groundwork for chaining | Hardware tests only; no benchmark artifact shipped because correctness/runtime gate failed before profiling | Undersized-surface rejection worked, but equal-size input rebinding caused private-runtime selector mismatches (`-[__NSArrayM procedureIndex]`, then `-[_ANERequest executionDelay]`) and still fell through to `statusType=0x9` ANE inference errors. Real rebinding is unsafe on this host/runtime. | **ABANDON** (fast rollback required) |

---

## 4) Key benchmark snapshots (what changed)

### Decode milestones

1) Stable decode contract snapshot (`2d852c5`):
- Artifact: `/tmp/decode_profile_ane10x_20260305`
- ANE median: `0.629 ms/token`
- Fastest Core ML naive median: `1.711 ms/token`
- Strict speedup: `2.72x`

2) Post-commit verification (`bdb0bd7`):
- Artifact: `/tmp/decode_profile_bdb0bd7`
- ANE median: `0.646 ms/token`
- Fastest Core ML naive median: `1.722 ms/token`
- Strict speedup: `2.67x`

Interpretation:
- Decode path is materially faster than Core ML naive baseline on current supported contract.
- 10x target is not achieved under current `maxSeq=32` constraint.

3) Tile-sync optimization A/B (same SHA, forced-old path vs optimized path):
- `maxSeq=128`:
  - Before (forced old sync each token): `/tmp/decode_syncopt_before_max128_20260305`
    - mean `0.692`, median `0.685`, p95 `0.748`, p99 `0.820`, `1445 tok/s`
    - breakdown: ANE `0.409 ms`, IO `0.219 ms`
  - After (boundary-only sync + incremental tile updates): `/tmp/decode_syncopt_after_max128_20260305`
    - mean `0.490`, median `0.483`, p95 `0.545`, p99 `0.693`, `2040 tok/s`
    - breakdown: ANE `0.395 ms`, IO `0.032 ms`
- `maxSeq=256`:
  - Before: `/tmp/decode_syncopt_before_max256_20260305`
    - mean `0.691`, median `0.685`, p95 `0.751`, p99 `0.817`, `1448 tok/s`
    - breakdown: ANE `0.409 ms`, IO `0.217 ms`
  - After: `/tmp/decode_syncopt_after_max256_20260305`
    - mean `0.535`, median `0.492`, p95 `0.700`, p99 `0.836`, `1869 tok/s`
    - breakdown: ANE `0.436 ms`, IO `0.035 ms`

4) Strict fairness snapshot after A13 (`maxSeq=128`):
- Artifact: `/tmp/decode_syncopt_coreml_max128_20260305`
- ANE median: `0.488 ms/token`
- Fastest Core ML naive median: `0.988 ms/token` (`.cpuAndNeuralEngine`)
- Strict speedup: `2.02x`

5) Thermal variance note (`maxSeq=32`, mixed ANE+CoreML run order):
- Mixed fairness runs showed ANE median drift to `~0.67 ms` in hot runs:
  - `/tmp/decode_syncopt_coreml_max32_20260305`
  - `/tmp/decode_syncopt_coreml_max32_r2_20260305`
- Standalone ANE-only confirmation remains fast/stable:
  - `/tmp/decode_syncopt_confirm_max32_20260305` median `0.486 ms`, mean `0.488 ms`, `2047 tok/s`.
Inference: mixed-run thermal state materially affects ANE median on this host; treat strict speedup claims from long mixed runs conservatively and include artifact provenance.

6) Decode runtime option sweep (post-A13):
- Quick matrix: `/tmp/decode_syncopt_opts_max128_20260305`
  - Best one-shot median appeared to be `ANE_EVAL_PATH=clientDirect` (`0.507 ms`) vs baseline (`0.663 ms`) in that run order.
- Confirmation repeats: `/tmp/decode_syncopt_confirm_evalpath_max128_20260305`
  - Baseline medians: `0.672`, `0.672`, `0.676`
  - `clientDirect` medians: `0.677`, `0.679`, `0.679`
- Verdict: runtime option did not hold up under repeats; keep default eval path.

### Prefill snapshots

1) Earlier favorable run:
- Artifact: `/tmp/prefill_profile_phase2_20260305`
- ANE median: `0.773 ms`
- Core ML `.all` median: `1.704 ms`

2) Later heavier run:
- Artifact: `/tmp/prefill_profile_ane10x_20260305`
- ANE median: `1.978 ms`
- Core ML `.all` median: `1.712 ms`

3) Post-commit verification:
- Artifact: `/tmp/prefill_profile_bdb0bd7`
- ANE median: `1.872 ms`
- Core ML `.all` median: `1.541 ms`

4) Sequential re-check (this pass):
- Baseline (ANE only): `/tmp/prefill_syncopt_baseline_seq_20260305`
  - mean `1.827`, median `1.855`, p95 `2.077`, p99 `2.184`
- QueueDepth+Pool combo: `/tmp/prefill_syncopt_combo_qd32_pool1_seq_20260305`
  - mean `1.903`, median `1.927`, p95 `2.109`, p99 `2.211`
- Combo verdict: regression; do not ship.

5) Fairness snapshot:
- `/tmp/prefill_syncopt_coreml_20260305`
  - ANE median `1.853 ms`
  - Core ML medians: `.all 1.684`, `.cpuAndNeuralEngine 1.718`, `.cpuAndGPU 1.135`
  - ANE remains slower than fastest Core ML baseline on this host/profile.

Interpretation:
- Prefill remains high-variance and can be slower than Core ML in longer/hotter runs.
- Decomposition shows host-side eval is dominant in measured profile mode on this host; `hwExecutionTime` unavailable prevents hard on-chip floor attribution.

---

## 5) Decision summary (why each call was made)

Decisions shipped:
- Keep decode architecture with persistent KV + slice updates because it is the only proven path to meaningful decode throughput gains.
- Keep strict decode guardrails (`maxSeq=32`) to avoid invalid ANE eval contracts and false performance claims.
- Keep detailed kernel profile reporting and metadata because reproducibility and root-cause attribution improved materially.
- Keep `ANE_COMPILE_CACHE_POLICY=preferCached` for iteration/repro because compile variance was extreme in auto mode.

Decisions iterated/abandoned:
- Did not keep unsafe perf-stats object fallback because it caused runtime exceptions (`unrecognized selector`) in private runtime.
- Did not keep eval-path-based workaround efforts for decode-attn contract failures because failure signature was invariant across paths.

---

## 6) Remaining optimization avenues (prioritized)

### P0 (highest ROI, still unblocked)

1) Lift `maxSeq=32` decode constraint with tiled/lane-packed cache layout
- Why: most direct path to higher decode speedup and meaningful scaling to real context lengths.
- Expected impact: largest decode gain potential.
- Risk: ANE contract fragility on mixed input shapes; requires careful probe-guided expansion.

2) Reduce decode dispatch overhead (attn+ffn fusion and/or deeper chaining)
- Why: current path still pays two evals per layer/token.
- Expected impact: significant constant-factor win in token latency.
- Risk: compiler splitting/failure and private runtime instability.

### P1 (important, but after P0)

3) Build/benchmark a stronger Core ML decode baseline (stateful KV if feasible)
- Why: stricter competitiveness claim and better understanding of true advantage margin.
- Expected impact: not necessarily speedup gain, but improves credibility and target calibration.
- Risk: coremltools complexity and ANE compiler stability.

4) Decode-specific ANE option sweeps (queue depth, eval path, memory pool, fences, latching)
- Why: could reduce host overhead and improve consistency.
- Expected impact: moderate constant-factor gains.
- Risk: unstable behavior and hard-to-reproduce wins without disciplined harnessing.

### P2 (prefill-focused, expected limited upside)

5) Prefill MIL-level variant experiments (RMSNorm forms, larger fusions, attention lowering alternatives)
- Why: only remaining path if trying to close residual prefill delta.
- Expected impact: low-to-moderate, likely not multi-x.
- Risk: high engineering time for potentially marginal gain.

6) Recover `hwExecutionTime` attribution path safely
- Why: needed for hard bottleneck proof and to distinguish host vs on-chip ceilings.
- Expected impact: diagnostic quality; may redirect optimization effort.
- Risk: private API behavior differs by host OS/runtime.

---

## 7) Reality check on targets

Decode:
- 6x may be possible only after removing current shape constraints and reducing dispatch overhead further.
- Current demonstrated range is ~2.67x–2.72x at supported contract (`maxSeq=32`).

Prefill:
- Based on observed behavior and architecture, large multi-x prefill gains vs Core ML are unlikely on this host/path.
- Practical target is parity to modest win; not 3–6x.

---

## 8) Re-run checklist (exact artifacts expected)

Decode profile run:
- Output dir should include:
  - `summary.txt`
  - `summary.json`
  - `ane_decode_token_latencies.csv`
  - `ane_decode_kernel_profile.csv`
  - `coreml_decode_all_token_latencies.csv`
  - `coreml_decode_cpuandneuralengine_token_latencies.csv`

Prefill profile run:
- Output dir should include:
  - `summary.txt`
  - `summary.json`
  - `ane_inference_latencies.csv`
  - `ane_inference_kernel_profile.csv`
  - Core ML latency CSVs

---

## 9) Tried vs Regressed (2026-03-06 update)

### Shipped improvements (kept)
- A13 decode tile-sync optimization (`5c016ec`) is a real win for tiled decode contexts:
  - `maxSeq=128`: median `0.685 -> 0.483 ms/token`, throughput `1445 -> 2040 tok/s`
  - `maxSeq=256`: median `0.685 -> 0.492 ms/token`, throughput `1448 -> 1869 tok/s`
  - Primary bottleneck shift: IO `~0.219/0.217 ms` down to `~0.032/0.035 ms` per token.

### Regressions or non-reproducible results (not kept)
- Strict decode speedup headline regressed vs earlier best:
  - Earlier best snapshot: `~2.67x–2.72x` (at `maxSeq=32`)
  - Recent strict snapshots: `~2.02x–2.20x`
  - Evidence: `/tmp/decode_syncopt_coreml_max128_20260305`, `/tmp/decode_syncopt_coreml_max32_20260305`, `/tmp/decode_syncopt_coreml_max32_r2_20260305`
  - Note: mixed ANE+CoreML run ordering showed thermal drift; ANE-only confirm still measured `~0.486 ms` median at `maxSeq=32` (`/tmp/decode_syncopt_confirm_max32_20260305`).
- Decode runtime option candidate (`ANE_EVAL_PATH=clientDirect`) regressed on confirmation:
  - One-shot looked faster, but 3x confirmations showed parity/slight regression vs baseline.
  - Evidence: `/tmp/decode_syncopt_opts_max128_20260305`, `/tmp/decode_syncopt_confirm_evalpath_max128_20260305`
  - Decision: `ABANDON`.
- Clean-worktree decode option re-sweep at `maxSeq=32` also failed to produce a better runtime configuration:
  - Evidence: `/tmp/decode_sweep_max32_quick_20260306`
  - Baseline `preferCached` remained best at median `0.486 ms` in the quick pass.
  - Confirmation median-of-medians:
    - baseline `preferCached`: `0.4940 ms`
    - `ANE_EVAL_PATH=clientDirect`: `0.4954 ms`
  - `clientDirect` therefore regressed by `0.27%`, and `ANE_KEEP_MODEL_WIRED=1` materially regressed to median `0.587 ms`.
  - Decision: `ABANDON` further runtime knob chasing for this shape until a structural change lands.
- Prefill high-ROI combo (`ANE_QUEUE_DEPTH=32 + ANE_MEMORY_POOL_ID=1`) regressed in sequential re-check:
  - Baseline median `1.855 ms`
  - Combo median `1.927 ms`
  - Evidence: `/tmp/prefill_syncopt_baseline_seq_20260305`, `/tmp/prefill_syncopt_combo_qd32_pool1_seq_20260305`
  - Decision: `ABANDON`.
- Fused decode layer prototype (`ESPRESSO_DECODE_LAYER_MODE=fusedLayer`) was discarded before fairness runs:
  - Split2 control run: `/tmp/decode_fused_cycle1_split2_quick_20260306`
    - mean `0.665 ms/token`, median `0.666 ms/token`, p95 `0.716 ms`, p99 `0.769 ms`, throughput `1504 tok/s`
    - ANE kernel `0.586 ms`, surface I/O `0.020 ms`
  - Fused attempts:
    - `ESPRESSO_DECODE_LAYER_MODE=fusedLayer` quick run stalled after `Compiling 1 decode ANE kernels (mode=fusedLayer)...`
    - `ESPRESSO_DECODE_LAYER_MODE=fusedLayer ESPRESSO_DECODE_ATTN_PROBE_MODE=passthrough` also stalled in the same phase
    - sample `/tmp/espresso-bench_2026-03-06_014704_vEMy.sample.txt` shows the process blocked under `ane_interop_compile -> -[_ANEInMemoryModel compileWithQoS:options:error:] -> -[_ANEClient compileModel:options:qos:error:]`
  - Why this matters:
    - The prototype never reached eval, so the failure is structural compile instability, not a performance regression that can be tuned around with runtime options.
    - Current quick baseline shows decode token time is still mostly kernel/eval (`0.586 ms`) rather than I/O (`0.020 ms`), so small boundary-copy wins alone will not approach the 6x target.
  - Decision: `ABANDON` this fused graph shape and revert the prototype before moving to the next decode experiment.
- Decode request-level input rebinding (`ESPRESSO_DECODE_BIND_INPUT_CHAIN`) was discarded before benchmarking:
  - Targeted hardware tests:
    - undersized-surface rejection passed
    - valid-size rebinding failed in real ANE eval
  - Failure signatures:
    - `_ANEClient buffersReadyWithModel:inputBuffers:...` with raw arrays throws `-[__NSArrayM procedureIndex]`
    - retrying with rebuilt `_ANERequest` throws `-[_ANERequest executionDelay]`
    - eval still fails with `statusType=0x9: Program Inference error`
  - Interpretation:
    - Rebuilding `_ANERequest` is not sufficient to safely rebind decode input surfaces after model load on this host/runtime.
    - This path is too unstable to benchmark or ship, so it was reverted immediately rather than iterated in the hot path.
  - Decision: `ABANDON`.
- Decode compile-time external surface aliasing was discarded before benchmarking:
  - Scope attempted:
    - added a surface-aware compile path so kernels could be built against externally supplied IOSurfaces
    - attempted intra-layer decode chaining by compiling `decodeFFN` with `decodeAttnQKV` output 0 as its input surface
  - Targeted hardware tests:
    - `ANERuntimeTests/test_kernel_rejects_undersized_external_input_surface` passed
    - `ANERuntimeTests/test_kernel_uses_provided_external_input_surface_and_evals` failed with `statusType=0x9: Program Inference error`
    - `InferenceOptimizationTests/test_decode_compile_time_chained_stack_matches_baseline_on_hardware` drifted to `NaN`
  - Interpretation:
    - the current ANE runtime contract on this host does not safely support externally supplied input IOSurfaces for these compiled kernels
    - because the primitive itself failed on real hardware, the path was reverted immediately instead of benchmarked
  - Decision: `ABANDON`.

### Invalid/discarded evidence
- Parallel prefill benchmark invocation was discarded as non-defensible due host contention; only sequential reruns are used for conclusions in this document.
- Cross-worktree compare to `main` (`f2d1da0`) on this host cannot produce decode/prefill parity numbers because:
  - `main` does not include the `espresso-bench` product (`swift build -c release --product espresso-bench` fails: `no product named 'espresso-bench'`).
  - The only shared perf test (`test_100_steps_benchmark`) is M4-gated and skips on Apple M3 Max in both worktrees.
Inference: strict decode/prefill regression assessment must be done against the nearest benchmark-capable base in this branch lineage, not `main`, on this host.

---

## 10) Tried vs Hypothesis (Max-Performance Plan)

### Proven tried paths (evidence-backed)
- Decode tiled IO reduction (A13): **works** and should be preserved as baseline.
- Decode runtime option stacking after A13 (A14): **no stable gain**; keep default path.
- Decode compile-time external surface aliasing: **unsafe/abandoned**; do not reuse as a dispatch-reduction path.
- Prefill queue-depth/memory-pool combo (A15): **regressed** under sequential rerun; do not reuse.
- Perf-stats fallback hacks (A10): **unsafe/abandoned**; do not pursue private selector spoofing.

### Performance-maximization hypotheses (next experiments)

| H# | Hypothesis | Why it could move needle | Expected impact (decode first) | Validation plan | Ship gate |
|---|---|---|---|---|---|
| H1 | **Dispatch reduction**: collapse per-token eval count (attn+ffn fusion/chaining) | Current decode still pays 2 evals/layer/token, so host dispatch overhead remains structural | High (largest remaining decode constant-factor lever) | Implement behind flag, run A/B at `maxSeq=32/128/256`, verify hardware correctness tests + parity checks | `SHIP` only if median gain holds across repeated runs (not one-shot) |
| H2 | **Decode perf-stats + compile/eval attribution** | We need to prove whether remaining decode headroom is host dispatch or kernel execution before deeper interop work | Medium (measurement quality + experiment targeting) | Add decode-side `lastHWExecutionTimeNS`/compile timing into artifacts; compare split2 baseline vs any future structural change | `SHIP` when artifacts expose actionable attribution without changing decode behavior |
| H3 | **Decode chaining via `_ANEChainingRequest` / `prepareChainingWithModel`** | Request-level rebinding is unsafe, so the next credible dispatch-reduction route is runtime-native chaining | Medium-to-high | Build a minimal hardware proof-of-life on a tiny multi-procedure or chained request path, then only integrate into decode if the primitive works | `SHIP` only after deterministic correctness and repeated median gain |
| H4 | **Stable thermal/fairness protocol**: isolate ANE/CoreML run-order effects | Current strict speedup is sensitive to thermal drift, masking real regression/improvement signal | Medium (measurement quality + defensibility) | Interleaved repeated runs with cooldown windows + median-of-medians reporting | `SHIP` as reporting protocol once reproducible |
| H5 | **Decode maxSeq scaling by contract-safe tiling variants** | If ANE contract permits wider auxiliary inputs safely, fewer sync events may be needed | Medium-to-high if contract-safe | Probe-guided shape expansion tests, then benchmark only passing families | `SHIP` only with stable eval + correctness |
| H6 | **Prefill fusion/dispatch tuning only** (no risky option combos) | Prefill appears host-eval dominated; options alone did not hold | Low-to-medium | Evaluate only high-ROI fusion/chaining variants with strict repeated runs | `ITERATE` unless repeatable gain > noise floor |
| H7 | **Stateful Core ML decode baseline (reported separately)** | Tightens fairness and prevents overstating speedup claims | Reporting impact, not direct speed gain | Add optional baseline mode and keep strict naive baseline unchanged | `SHIP` if stable and reproducible |

### Practical ceiling hypothesis (current evidence)
- With current kernel contract and 2-eval decode dispatch, decode speedups above current `~2x` strict likely require **architectural dispatch reduction (H1)** rather than additional runtime option sweeps.
- Prefill large multi-x gains are unlikely on this host path without a meaningful reduction in host eval overhead; short-term expectation is parity-to-modest gains, not 3–6x.

### Next queued experiment
- Build a minimal `_ANEChainingRequest` proof-of-life in the interop layer and validate it on hardware before touching decode hot-path code again.
