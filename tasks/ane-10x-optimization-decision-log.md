# ANE 10x Optimization Decision Log

Date: 2026-03-05  
Repo/worktree: `/private/tmp/espresso-ane-10x-max`  
Branch: `feat/ane-10x-max`  
Latest commit at time of this document: `f69e711`

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
| A13 | Decode tile-sync optimization (boundary-only window sync + incremental lane cache updates + removed redundant lane-zero copies) | Full-window cache sync every token was dominating decode IO at `maxSeq > 32` | Commit in this pass; `/tmp/decode_syncopt_before_max128_20260305`, `/tmp/decode_syncopt_after_max128_20260305`, `/tmp/decode_syncopt_before_max256_20260305`, `/tmp/decode_syncopt_after_max256_20260305` | Large, repeatable IO reduction and throughput gain for tiled decode contexts (`128/256`) while preserving hardware correctness tests | **SHIP** |

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
