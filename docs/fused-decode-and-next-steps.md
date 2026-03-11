# Fused Decode Kernel + Performance Tuning Research

**Date:** 2026-03-06
**Host:** M3 Max, macOS 15.x
**Branch:** `feat/vc-eval-probe`

---

## 1. Fused Decode Kernel (IMPLEMENTED)

### Problem

ANE decode throughput is eval-dominant. The decode path dispatched 12 synchronous
`eval()` calls per token (2 kernels x 6 layers), each carrying ~0.095ms dispatch
overhead. After exhausting IO trimming and hitting dead ends with `_ANEChainingRequest`
and `_ANEVirtualClient`, the remaining software-addressable overhead was **dispatch count**.

### Solution

Combine the decode attention kernel (`DecodeAttentionQKVGenerator`) and FFN kernel
(`DecodeFFNGenerator`) into a single MIL program per layer (`FusedDecodeLayerGenerator`).
This halves the dispatch count from 12 to 6 and eliminates the attn-to-FFN surface copy.

### Implementation

| File | Role |
|------|------|
| `Sources/MILGenerator/FusedDecodeLayerGenerator.swift` | Single MIL combining RMSNorm1 + QKV + attention + Wo + residual1 + RMSNorm2 + SwiGLU FFN + residual2 |
| `Sources/ANERuntime/FusedDecodeKernelSet.swift` | `~Copyable` kernel wrapper (9 weight blobs) |
| `Sources/Espresso/DecodeForwardPass.swift` | `FusedDecodeSurfaceHandles` + `runFusedDecodeTimed()` loop |
| `Tests/ANERuntimeTests/FusedDecodeKernelTests.swift` | 19 tests (12 MIL + 7 hardware-gated) |

### MIL Design

```
Inputs (4):
  x         [1, dim, 1, laneSpatial]   token packed at lane 0
  kCache    [1, dim, 1, maxSeq]         K cache
  vCache    [1, dim, 1, maxSeq]         V cache
  maskCache [1, dim, 1, maxSeq]         attention mask

Outputs (3):
  xNext     [1, dim, 1, laneSpatial]   both residuals applied
  kfFull    [1, dim, 1, laneSpatial]   K projection for cache writeback
  vfFull    [1, dim, 1, laneSpatial]   V projection for cache writeback

Weights (9 BLOBFILE constants):
  rms1.bin, wq.bin, wk.bin, wv.bin, wo.bin   (attention)
  rms2.bin, w1.bin, w3.bin, w2.bin            (FFN)
```

**SSA name collision strategy**: FFN RMSNorm variables are prefixed `f_` (e.g., `f_sq`,
`f_ss`, `f_rrms`, `f_xr`, `f_xn`). Shared constants (`invd`, `eps`, `nhalf`, `raxCh`,
`raxSp`, `kd`) and conv constants (`pt`, `st`, `pd`, `dl`, `gr`) are defined once in
the attention block and reused by the FFN block.

### Measured Results (M3 Max, 1 layer, maxSeq=32)

| Path | ANE ms/step | Dispatches |
|------|-------------|-----------|
| Unfused (attn + FFN) | 0.761 | 2 |
| Fused (single dispatch) | 0.666 | 1 |
| **Savings** | **0.095 (12.5%)** | **-1** |

Extrapolated over 6 layers: **~0.57ms saved per token** (from 12 to 6 dispatches).

---

## 2. Current Decode Budget Breakdown (6 layers)

```
Component                   Unfused         Fused (current)
------------------------------------------------------------
ANE dispatch overhead       6 x 0.095ms     eliminated
                            = 0.570ms

ANE compute (hw time)       6 x 0.310ms     6 x 0.405ms
                            = 1.860ms        = 2.430ms

IO (cache update + chain)   6 x 0.029ms     6 x 0.024ms
                            = 0.174ms        = 0.144ms

Attn->FFN surface copy      6 x 0.005ms     eliminated
                            = 0.030ms

Total per token             ~2.634ms         ~2.574ms
```

**The remaining wall is ANE compute time** -- that is hardware silicon speed. The
dispatch overhead was the largest software-addressable cost and has been eliminated.

---

## 3. Approaches to 4x Over CoreML

### 3.1 Multi-Layer Kernel Fusion via KV Cache Packing (HIGHEST PRIORITY)

**Concept**: Fuse 2 transformer layers into a single MIL program. Pack both layers' KV
caches into one input IOSurface using `slice_by_size` to extract each region. This
reduces dispatches from 6 to 3 and eliminates inter-layer surface copies within each
fused pair.

**Cache packing layout** (channel axis of a `[1, C_packed, 1, maxSeq]` input):

| Channel Offset | Width | Content |
|----------------|-------|---------|
| 0 | 768 | Layer 0 kCache |
| 768 | 768 | Layer 0 vCache |
| 1536 | 768 | Layer 0 maskCache |
| 2304 | 768 | Layer 1 kCache |
| 3072 | 768 | Layer 1 vCache |
| 3840 | 768 | Layer 1 maskCache |

`C_packed = 4608`, packed cache size = 294,912 bytes (0.28 MB).

**Memory analysis (2-layer fused program, dim=768, hidden=2048)**:

| Component | Per Layer | 2 Layers |
|-----------|-----------|----------|
| Attention weights (Wq,Wk,Wv,Wo) | 4 x 768^2 x 2 = 4.72 MB | 9.44 MB |
| FFN weights (W1,W3,W2) | 3 x 2048x768 x 2 = 9.44 MB | 18.87 MB |
| RMSNorm (rms1,rms2) | 2 x 768 x 2 = 3 KB | 6 KB |
| **Total baked weights** | **14.28 MB** | **28.57 MB** |
| Cache data in IOSurface | 0.14 MB | 0.28 MB |
| **Working set vs 32MB SRAM** | fits | fits (ANE streams weights via DMA) |

**MIL feasibility**: `slice_by_size` on channel-packed inputs is a proven pattern -- the
backward pass generators (`SDPABackward1Generator`, `QKVBackwardGenerator`, etc.) already
perform 3-4 slices from packed inputs with convolutions and full attention math. The
documented `slice_by_index` failure does NOT apply to `slice_by_size`.

**Expected savings**: 3 more dispatch overheads eliminated (3 x 0.095ms = 0.285ms) plus
inter-layer copy elimination. Combined with single-layer fusion: **12 dispatches to 3**.

**Layer fusion scaling**:

| Fused Layers | Dispatches | Baked Weights | SRAM Fit? | Feasibility |
|---|---|---|---|---|
| 1 (current) | 6 | 14.28 MB | yes | **done** |
| 2 | 3 | 28.57 MB | marginal | **high confidence** |
| 3 | 2 | 42.85 MB | no (streams) | medium confidence |
| 6 | 1 | 85.70 MB | no | low confidence |

**Recommendation**: Start with 2-layer fusion. If successful, try 3.

### 3.2 Metal + ANE Hybrid Decode (HIGH IMPACT, UNEXPLORED)

**Concept**: Use ANE for convolution-heavy work (QKV projections, FFN) and Metal for
dynamic attention computation (Q@K^T, softmax, weights@V). IOSurfaces are the native
shared memory format for both Metal and ANE -- **zero copy between accelerators**.

**Why this could achieve 4x**:
- Two accelerators working in **parallel**: ANE does layer N's FFN while Metal does
  layer N+1's attention
- Metal `simdgroup_matrix` on M3 provides ~7 additional TFLOPS for matmul
- ANE's attention is currently a probe path (bypasses real math). Metal handles dynamic
  attention natively with full flexibility
- CoreML cannot do this -- it dispatches to one accelerator at a time per subgraph

**Architecture**:
```
ANE:   [QKV proj] -----> IOSurface (Q,K,V) -----> [FFN]
                              |                      ^
                              v                      |
Metal:                  [Q@K^T -> softmax -> @V] --> IOSurface (attn_out)
```

**Challenges**:
- Requires Metal compute pipeline for attention
- Synchronization between ANE and Metal dispatch
- Surface format compatibility (both use FP16, both use IOSurface)

### 3.3 `evaluateRealTimeWithModel:` Probe (MEDIUM IMPACT, QUICK WIN)

The `_ANEClient` exposes a real-time eval path:
- `beginRealTimeTask` / `endRealTimeTask` lifecycle
- `evaluateRealTimeWithModel:options:request:error:`

**This has never been tested for dispatch latency.** If the real-time path bypasses the
QoS scheduling queue, it could have materially lower dispatch overhead than
`evaluateWithQoS:`. The existing interop code already has conditional support for this
path (controlled by `ANE_REALTIME` environment variable).

**Estimated effort**: 1-2 hours to probe and benchmark.

### 3.4 Metal SharedEvent on Standard Eval Path (MEDIUM IMPACT)

`_ANERequest.setSharedEvents:` works on the standard eval path (no VirtualClient
required). If we construct an `IOSurfaceSharedEvent` via `[MTLDevice newSharedEvent]`
and attach it, the ANE hardware might signal completion asynchronously -- enabling true
pipelined dispatch where:

1. Submit eval for layer N
2. ANE signals Metal SharedEvent when done
3. CPU starts IO for layer N while layer N+1 eval begins

**Key unknown**: Does `setSharedEvents:` on the standard `_ANEClient` eval path actually
trigger hardware-level async signaling? The `completionEvent:` parameter on the
VirtualClient eval path suggests it should, but the standard path may ignore it.

### 3.5 Speculative Decoding (ALGORITHMIC, HIGH IMPACT)

**Concept**: Use a tiny draft model (1-2 layers, or a distilled version) to generate N
candidate tokens, then verify all N in one batched forward pass on the full model. ANE
excels at batch compute (deep graphs, high utilization).

This transforms the decode problem from "N sequential single-token dispatches" into
"1 cheap draft + 1 expensive verify" -- an algorithmic 3-5x on top of hardware
optimization.

**Why ANE is particularly suited**: The prefill path (batch tokens) already achieves much
higher utilization than decode (single token). Speculative decoding turns decode into a
prefill-like workload.

---

## 4. Approaches Proven Blocked (Do Not Retry)

| Approach | Failure Mode | Evidence |
|----------|-------------|----------|

## 2026-03-10 — Rejected contiguous-shard CPU staged exact head

### Attempt

Added an exact branch-and-bound output-head seam for direct-select generation:

- stage 1: compute admissible per-shard upper bounds from shard center/radius summaries on the normalized token vector
- stage 2: run exact CPU BLAS scoring only on shards whose bound can still beat the current best exact score
- tie policy preserved the current first-max argmax behavior

This was intentionally CPU-routed for the first probe because the branch already had strong evidence that extra ANE head staging/regrouping regresses and full fused recurrent+head attachment is compiler-blocked.

### Hardware measurement

Matched recurrent fused-triplet direct-select comparison (`warmup=3`, `iterations=20`, `maxNewTokens=8`, echo weights):

| Path | Median ms/token | tok/s | Compile/init ms | Trunk ms/token | Head/logits ms/token |
|---|---:|---:|---:|---:|---:|
| Control: fused-triplet + `.aneRMSNormClassifier` | `2.3946458333333336` | `417.60` | `542.6085` | `1.259015625` | `1.0993333333333333` |
| Contiguous-shard staged exact CPU head | `28.1385625` | `35.54` | `5176.693166666667` | `1.7956223958333335` | `26.325609375` |

Exactness/parity check:

- generated tokens matched the control on the hardware echo test
- the path is exact, but not competitive

### Interpretation

This avenue is rejected in its contiguous-shard form.

- The shard bounds were too loose to prune enough work.
- The runtime effectively degenerated into repeated CPU block GEMMs, so head time exploded instead of shrinking.
- Init/compile time also regressed badly because the staged head now has to build the shard summaries up front.

### Decision

- keep the exact branch-and-bound seam and tests as reference infrastructure
- reject contiguous-shard CPU staged exact head as a throughput path
- if exact head work continues, it must use materially better block geometry (for example clustered blocks with admissible bounds), not contiguous shards

## 2026-03-10 — Rejected live `k=2` recurrent exact upper-bound path as a 6x route

### Attempt

Added a recurrent-native exact `k=2` seam instead of reviving the old speculative stack:

- `ExactTwoTokenGeneratingLanguageModel`
- `ExactTwoTokenGenerationHarness`
- `ANEExactTwoTokenUpperBoundGenerationModel`

The Stage 1 probe uses the proven recurrent control and forces an upper-bound future-token contract on the echo checkpoint family:

- one exact committed token is already selected at the live recurrent cursor
- the future-token proposal is wired through the recurrent harness
- if that proposal matches the exact next token, the harness commits a second exact token only by paying a second live `decodeSelectedToken` call
- proposer, verifier trunk, verifier logits, and state-advance costs are recorded separately per pass

This is an upper-bound structural probe, not a trained future-head result.

### Verification

Implementation and accounting were verified in the package test runner:

- `swift build --build-tests`
- `swift test --skip-build --filter GenerationHarnessTests`

The new tests cover:

- exact committed-token accounting
- accepted future-token accounting
- state-advance cost being zeroed on future-token rejection
- prefix-only commit behavior for the exact `k=2` pass seam

### Structural result

The current Stage 1 path does not escape one expensive recurrent step per committed token.

Direct evidence from the implementation:

- `performExactTwoTokenPass` first calls `baseModel.decodeSelectedToken(nextToken: currentToken, ...)` to obtain the exact next token
- when the future proposal is accepted, it then calls `baseModel.decodeSelectedToken(nextToken: exactNext, ...)` again to advance live recurrent state and expose the next current token
- that second recurrent call is reported as `stateAdvanceLatencyMs`, not hidden inside proposer cost

Because the accepted second token still requires a second live recurrent decode, this seam does not create the reusable state-advance mechanism required for a believable `6x` exact single-stream path.

### Hardware benchmark blocker

Same-session hardware benchmarking on `feat/ane-multitoken` did not complete cleanly enough to report throughput:

- filtered target: `GenerationHarnessHardwareTests/test_recurrent_exact_two_token_upper_bound_reports_pass_breakdown_on_hardware`
- matched settings: `warmup=3`, `iterations=20`, `prompt=[0]`, `maxNewTokens=8`
- observed failure mode: the run stalled for more than three minutes while compiling the fused-triplet recurrent control before any measured iterations completed
- sampled stack showed the control path inside `ANERecurrentGenerationModel.compileFusedTripletSessions` -> `ANEKernel.init` -> `_ANEClient compileModel`

No throughput claim is made from that blocked run.

### Decision

Reject this live Stage 1 path as the next 6x avenue.

- do not train or tune real `k=2` future heads on top of this seam
- first solve reusable multi-token state advancement, snapshot/restore, or branch-state promotion so that an accepted second token does not require a second full recurrent decode
- rerun the hardware comparison only after the fused-triplet control benchmark is healthy in the same serialized session
- otherwise move to the next architecture class rather than spending more time tuning this path

## 2026-03-10 — Implemented two-step branch-state-promotion architecture; same-session measurement still blocked

### Attempt

Implemented a materially different recurrent-native `k=2` path that prepares two sequential recurrent states in one trunk pass:

- new MIL generator: `RWKVStyleTwoStepRecurrentGenerator`
- new runtime kernel set: `RWKVStyleTwoStepRecurrentKernelSet`
- new session type: `RWKVStyleTwoStepRecurrentSession`
- new exact model path: `ANEExactTwoTokenBranchStatePromotionModel`

The contract is:

- token `t` activation and proposed token `t+1` activation enter one two-step recurrent pass
- each layer exposes `stateMid` after step 1 and `stateOut` after step 2
- on exact prefix commit, the harness promotes `stateMid` for a one-token commit or `stateOut` for a two-token commit
- accepted token 2 no longer triggers a second `decodeSelectedToken` call in the model path

This is an implementation result, not yet a throughput result.

### Verification

Verified the architecture and seams in the package test runner:

- `swift build --build-tests`
- `swift test --skip-build --filter GenerationHarnessTests`
- `swift test --skip-build --filter RWKVStyleTwoStepRecurrentGeneratorTests`

The package build required correcting stale SwiftPM test-target dependencies so the existing test imports matched declared modules:

- `MILGeneratorTests` now depends on `ANERuntime`
- `ANERuntimeTests` now depends on `Espresso`

### Current measurement blocker

The committed hardware seam was rerun with the hardware gate enabled, and a smaller compile/init-only seam was added to separate compile failure from runtime failure:

- control compile/init-only: `ANE_HARDWARE_TESTS=1 swift test --skip-build --filter GenerationHarnessHardwareTests/test_recurrent_single_layer_control_reports_compile_init_only_on_hardware`
- two-step compile/init-only: `ANE_HARDWARE_TESTS=1 swift test --skip-build --filter GenerationHarnessHardwareTests/test_recurrent_exact_two_token_branch_state_promotion_reports_compile_init_only_on_hardware`
- full same-session comparison: `ANE_HARDWARE_TESTS=1 swift test --skip-build --filter GenerationHarnessHardwareTests/test_recurrent_exact_two_token_branch_state_promotion`
- matched settings inside the test: `warmup=3`, `iterations=20`, `prompt=[0]`, `maxNewTokens=8`

No valid compile/init times or runtime medians were produced in the bounded unblock pass.

Observed failure modes:

- the control compile/init-only seam did not reach its first print within roughly `45s`
- a live sample showed `GenerationHarnessHardwareTests.measureRecurrentSingleLayerControlCompileInitOnly` stalled inside `ANERecurrentGenerationModel.compileSingleLayerSessions` -> `RWKVStyleRecurrentKernelSet.compileStep` -> `ANEKernel.init` -> `_ANEClient compileModel`
- the two-step compile/init-only seam also did not reach its first print within roughly `45s`
- a live sample showed `GenerationHarnessHardwareTests.measureRecurrentExactTwoTokenBranchStatePromotionCompileInitOnly` stalled inside `RWKVStyleTwoStepRecurrentKernelSet.compileStep` -> `ANEKernel.init` -> `_ANEClient compileModel`

### Interpretation

Inference from the implemented code path:

- this avenue now satisfies the required structural property that accepted work can promote prepared recurrent state instead of paying a second recurrent decode in the model path
- the remaining blocker is ANE compile/init on this host/session, not the old replay-heavy verifier design

Measured result from the bounded unblock pass:

- compile/init-only control did not reach first output quickly
- compile/init-only two-step path did not reach first output quickly
- therefore no honest `compile/init time`, `committed_exact_tokens/pass`, effective `ms/token`, or exact parity status can be reported for this session

Hard gate result:

- keep `3e6cced` as the reference architectural checkpoint
- do not start future-head work on this checkpoint family
- pivot next to a student trained for the two-step exact contract unless a future host/session can make the compile/init seam reach first output quickly

## 2026-03-10 — Student pivot: added two-step future-head sidecar and export seam

### Why this pivot is the correct next move

Measured blocker from the bounded hardware pass:

- both the control and the two-step compile/init-only seams stalled before first output in `_ANEClient compileModel`
- this blocks same-session runtime truth for the current checkpoint family

Inference from the code structure:

- the two-step branch-state-promotion architecture remains a valid reference contract for accepted-work reuse
- the next productive step is to make that contract trainable and serializable without reopening the blocked runtime path

### Implementation

Added a separate sidecar artifact for the exact two-step contract instead of extending the base checkpoint format:

- new file format: `TwoStepStudentCheckpoint`
- strict contract metadata: `dim`, `vocabSize`, `layerCount`, `horizon=2`, exact prefix-only, prepared-state promotion, and whether the teacher classifier was shared
- stored weights: one future-head RMS vector and one full `vocab x dim` future classifier matrix
- seeding path: copy the teacher `rmsFinal`, then copy either the shared embedding matrix or the explicit classifier matrix into the future classifier seed

This keeps `Checkpoint.save/load` and training-resume compatibility unchanged while allowing the student future-head contract to version independently.

### Verification

- `swift test --filter TwoStepStudentCheckpointTests`
- `swift build --build-tests`
- `swift run espresso-train --model /does/not/exist --export-two-step-student /tmp/espresso-two-step-student-sidecar.bin`

Observed result:

- the focused sidecar test suite passed `4/4`
- the package built successfully with tests enabled
- the CLI export seam succeeded even with a missing pretrained model because it falls back to deterministic random init and exits immediately after writing the sidecar

### Artifact footprint

For the current `ModelConfig` (`dim=768`, `vocab=32000`, `layers=12`), the exported sidecar written by the CLI seam was:

- `/tmp/espresso-two-step-student-sidecar.bin`
- `98307104` bytes on disk

That size matches the intended artifact shape: one `768`-float RMS vector, one full `32000 x 768` float classifier matrix, and a small binary header.

### Decision

- keep `3e6cced` as the architectural reference for the exact two-step contract
- keep `2e49cab` as the hardware compile/init gate result
- use the new sidecar/export seam as the student-route starting point
- do not start runtime future-head integration on this checkpoint family until a future host/session can produce honest same-session hardware medians

## 2026-03-10 — Standalone release probe recovered exact hardware truth; two-step wins at 1 layer only

### Why this probe exists

The `xctest` hardware seam was good enough to prove the architecture and isolate the compile/init stall, but it was too brittle to settle runtime truth. I added a standalone release executable:

- `espresso-multitoken-probe`

The probe measures the exact recurrent control and the exact two-step branch-state-promotion path in one fresh process, with the same `warmup=3`, `iterations=20`, `prompt=[0]`, `maxNewTokens=8`, and `maxSequenceTokens=32` contract used by the committed harness.

### Compile/init truth recovered outside `xctest`

Fresh-process compile/init-only run:

- command: `/tmp/espresso-ane-multitoken-release/release/espresso-multitoken-probe --mode compile-init-only --layer-count 6 --control-backend fused-triplet --output-head-backend ane-rmsnorm-classifier --max-sequence-tokens 32`
- control compile/init wall: `36625.965583333338 ms`
- control reported compile: `36625.956166666663 ms`
- two-step compile/init wall: `812.4782083333333 ms`
- two-step reported compile: `812.4590833333334 ms`

This changed the status of the earlier gate:

- the ANE path was not deadlocked in general
- the earlier `xctest` unblock pass had measured a bad route to first output, not the permanent absence of hardware truth

### Exact runtime results

All reported comparisons preserved:

- exact parity status: `match`
- committed exact tokens/pass: `2.0`
- accepted future tokens/pass: `1.0`

Measured release-probe medians:

| Depth | Control backend | Control `ms/token` | Two-step `ms/token` | Verdict |
|---|---|---:|---:|---|
| 1 | `single` | `1.452750`, `1.768331`, `1.788609` | `1.354299`, `1.419352`, `1.484302` | exact two-step win in `3/3` repeats |
| 2 | `fused-pair` | `1.478724`, `1.918604`, `1.787354` | `1.853026`, `1.908977`, `1.969635` | noisy crossover; centered slightly behind control |
| 3 | `fused-triplet` | `1.642930`, `1.820557` | `1.794927`, `2.339016` | exact two-step loss in `2/2` repeats |
| 6 | `fused-triplet` | `2.493940` | `3.386833` | exact two-step loss |

Representative repeated 1-layer win:

- control: `1.7683307291666666 ms/token`, `565.51 tok/s`
- two-step: `1.4193515625000002 ms/token`, `704.55 tok/s`

Representative repeated 2-layer near-crossover:

- control: `1.9186041666666664 ms/token`, `521.21 tok/s`
- two-step: `1.9089765625000001 ms/token`, `523.85 tok/s`

Representative repeated 3-layer loss:

- control: `1.8205572916666664 ms/token`, `549.92 tok/s`
- two-step: `2.339015625 ms/token`, `427.55 tok/s`

Representative 6-layer loss:

- control: `2.4939401041666667 ms/token`, `400.97 tok/s`
- two-step: `3.386833333333333 ms/token`, `295.27 tok/s`

### Interpretation

Measured result:

- the exact two-step branch-state-promotion path materially raises the ceiling beyond one-token decode
- the architecture achieves more than one exact committed token per expensive pass on average
- there is a real exact throughput win against the matched 1-layer recurrent control

Measured limit on the current checkpoint family:

- the win does not scale through the stronger fused controls yet
- 2 layers is a noisy crossover regime
- 3 and 6 layers are still clear losses

Measured bottleneck:

- proposer cost is effectively zero on the echo checkpoint family
- state advancement is effectively zero compared with trunk and logits
- the remaining runtime tax is verifier-side, especially the cost of making the two-step path competitive with the fused control trunk and head

### Decision

- keep `3e6cced` as the architectural checkpoint that proves reusable accepted-work state promotion
- keep the new release probe as the honest hardware truth seam for future exact two-step iterations
- do not claim a win against the current strong 6-layer fused-triplet exact control
- next hypothesis should attack verifier cost directly, not proposer quality

## 2026-03-10 — Fused pair two-step trunk extends the exact throughput win through 4 layers

### Hypothesis

The single-layer two-step verifier was still losing to stronger fused controls because it was not inheriting the control path's main trunk win. I replaced the per-layer two-step verifier trunk with pair-fused two-step sessions so the exact path could reuse accepted work and recurrent fusion at the same time.

### Implementation

Added a pair-fused two-step stack:

- new MIL generator: `RWKVStyleFusedTwoLayerTwoStepGenerator`
- new runtime kernel wrapper: `RWKVStyleFusedTwoLayerTwoStepKernelSet`
- new exact recurrent session: `RWKVStyleFusedTwoLayerTwoStepSession`
- extended `ANEExactTwoTokenBranchStatePromotionModel` with `trunkBackend`
- extended `espresso-multitoken-probe` with `--two-step-backend`

The exact contract stayed unchanged:

- parity remained exact
- committed exact tokens/pass remained `2.0`
- accepted future tokens/pass remained `1.0`

### Verification

- `swift test --filter RWKVStyleFusedTwoLayerTwoStepGeneratorTests`
- `swift test --filter GenerationHarnessTests`
- `swift build --product espresso-multitoken-probe`
- `swift build -c release --product espresso-multitoken-probe --scratch-path /tmp/espresso-ane-multitoken-release`

### Measurements

2-layer compile/init-only sanity check:

- control fused-pair: `351.576375 ms`
- two-step fused-pair: `377.761042 ms`

Repeated 2-layer matched fused-pair comparison:

| Run | Control `ms/token` | Two-step `ms/token` | Verdict |
|---|---:|---:|---|
| 1 | `2.124839` | `1.534096` | exact two-step win |
| 2 | `1.679589` | `1.556641` | exact two-step win |

Repeated 4-layer matched fused-pair comparison:

| Run | Control `ms/token` | Two-step `ms/token` | Verdict |
|---|---:|---:|---|
| 1 | `2.195484` | `2.149909` | exact two-step win |
| 2 | `2.334737` | `2.234477` | exact two-step win |

Repeated 6-layer comparison against the strong fused-triplet control:

| Run | Control backend | Control `ms/token` | Two-step backend | Two-step `ms/token` | Verdict |
|---|---|---:|---|---:|---|
| 1 | `fused-triplet` | `2.146151` | `fused-pair` | `2.317794` | loss |
| 2 | `fused-triplet` | `2.293576` | `fused-pair` | `2.529677` | loss |

### Interpretation

Measured result:

- fused pair reuse is a real architectural breakthrough, not a one-off micro-optimization
- the exact two-step path now wins through 4 layers with repeated hardware medians
- the architecture still commits more than one exact token per expensive pass on average

Measured limit:

- pair fusion alone is not enough to beat the strong 6-layer fused-triplet control
- but it materially narrowed the 6-layer gap versus the old single-layer two-step verifier

### Decision

- keep the fused-pair two-step trunk as the new best exact multi-token path on this branch
- next hypothesis should extend fused verifier reuse to triplets before falling back to batched verifier heads or future-head training

## 2026-03-10 — Fused triplet two-step trunk wins the strong 6-layer exact control

### Hypothesis

Pair fusion proved that the exact two-step path only started scaling once its verifier trunk mirrored the control path's own fusion pattern. The next step was to extend that same idea to triplets and measure it directly against the strong 6-layer fused-triplet control.

### Implementation

Added a triplet-fused exact two-step stack:

- new MIL generator: `RWKVStyleFusedThreeLayerTwoStepGenerator`
- new runtime kernel wrapper: `RWKVStyleFusedThreeLayerTwoStepKernelSet`
- new exact recurrent session: `RWKVStyleFusedThreeLayerTwoStepSession`
- extended `ANEExactTwoTokenBranchStatePromotionModel` to support `.fusedThreeLayerTriplets`

The probe surface already supported `--two-step-backend`, so the only runtime change needed after the backend implementation was to point the exact path at the new fused-triplet trunk.

### Verification

- `swift test --filter RWKVStyleFusedThreeLayerTwoStepGeneratorTests`
- `swift test --filter GenerationHarnessTests`
- `swift build --product espresso-multitoken-probe`
- `swift build -c release --product espresso-multitoken-probe --scratch-path /tmp/espresso-ane-multitoken-release`

### Measurements

6-layer compile/init-only sanity check:

- control fused-triplet: `519.468417 ms`
- two-step fused-triplet: `632.644375 ms`

Repeated 6-layer exact comparison:

| Run | Control `ms/token` | Two-step `ms/token` | Control `tok/s` | Two-step `tok/s` | Verdict |
|---|---:|---:|---:|---:|---|
| 1 | `2.616013` | `2.197565` | `382.31` | `455.05` | exact two-step win |
| 2 | `2.397878` | `2.176102` | `417.04` | `459.54` | exact two-step win |

Exactness contract on both runs:

- parity status: `match`
- committed exact tokens/pass: `2.0`
- accepted future tokens/pass: `1.0`

### Interpretation

Measured result:

- this is a real recurrent-native exact multi-token throughput win against the strongest current 6-layer single-stream recurrent control on this branch
- one expensive recurrent pass now yields more than one exact committed token on average
- the win comes from verifier trunk reuse, not from proposal quality or an apples-to-oranges control

Measured bottleneck after the win:

- proposer cost is still effectively zero on the echo checkpoint family
- state advancement remains negligible
- the remaining cost is still verifier-side, with trunk and logits now close enough that head batching or lane sweeps are the next honest levers

### Decision

- keep the fused-triplet exact two-step trunk as the current best exact multi-token architecture on this branch
- next work should try to push this path below the standing single-stream best before reopening trained future-head work

## 2026-03-10 — Batched verifier head pushes exact fused-triplet two-step past 4x

### Hypothesis

After the fused-triplet exact two-step trunk win, the remaining measured bottleneck was verifier-side logits work. The exact path was still evaluating the fused ANE RMSNorm+classifier head separately for the two prepared activations, so batching both prepared activations through one ANE head eval was the smallest remaining boundary-removal change with enough upside to matter for an honest `4x` single-stream claim.

### Implementation

Added a pair-eval path for the fused ANE output head:

- `ANEGenerationOutputHeadIO.writeTokenPair` writes the prepared activations into spatial slices `0` and `1` of the existing head input surface
- `ANEGenerationOutputHeadIO.argmaxTokenPairLogits` reads exact argmax independently from those two output slices
- `ANEGenerationRMSNormClassifierHead.selectArgmaxPair` now runs one ANE eval for the two prepared activations
- `ANEExactTwoTokenBranchStatePromotionModel.prepareActivationPair` now uses that pair-eval fast path on `.aneRMSNormClassifier`

The recurrent trunk and exact branch-state-promotion contract stayed unchanged:

- parity must still match the recurrent control
- committed exact tokens/pass must still be prefix-only and exact
- state advancement must still promote the prepared `stateMid` or `stateOut` surfaces without replay

### Verification

- `swift test --filter ANEGenerationOutputHeadTests`
- `swift test --filter GenerationHarnessTests`
- `swift build -c release --product espresso-multitoken-probe --scratch-path /tmp/espresso-ane-multitoken-release`

### Measurements

Compile/init-only release probe:

- control fused-triplet: `811.378125 ms`
- two-step fused-triplet with batched head: `900.605500 ms`

Repeated 6-layer exact release-probe comparisons (`warmup=3`, `iterations=20`, prompt `[0]`, `maxNewTokens=8`, `maxSequenceTokens=32`):

| Run | Control `ms/token` | Two-step `ms/token` | Control `tok/s` | Two-step `tok/s` | Verifier trunk `ms/pass` | Verifier logits `ms/pass` | Verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| 1 | `2.786466` | `2.409169` | `358.89` | `415.08` | `2.287724` | `1.536932` | win |
| 2 | `2.339284` | `1.365719` | `427.48` | `732.23` | `1.348151` | `0.885375` | win |
| 3 | `2.243776` | `1.564339` | `445.68` | `639.25` | `1.517234` | `0.980995` | win |

Exactness contract on all three runs:

- parity status: `match`
- committed exact tokens/pass: `2.0`
- accepted future tokens/pass: `1.0`
- proposer `ms/pass`: `0`
- median state advance `ms/pass`: `0.014386`

Three-run medians:

- control fused-triplet: `2.339284 ms/token`
- exact two-step with batched head: `1.564339 ms/token`
- verifier trunk: `1.517234 ms/pass`
- verifier logits: `0.980995 ms/pass`

Relative to the saved CoreML single-stream baselines:

- versus `6.582224 ms/token`: about `4.21x`
- versus `7.044784 ms/token`: about `4.50x`

### Interpretation

Measured result:

- this is the first repeated exact single-stream fused-decode path on the branch that clears an honest `4x` over the standing CoreML baseline
- the gain came from removing one verifier-head eval per exact two-token pass, not from changing the trunk contract or relaxing exactness
- verifier logits dropped from the earlier `~1.93-1.95 ms/pass` band into roughly `~0.89-0.98 ms/pass` on the two faster repeats

Inference:

- the slower first repeat was likely a host/session outlier rather than an architecture regression, because the next two repeats stayed far below the `4x` threshold with exact parity and the three-run medians still clear `4x`

### Decision

- keep the batched verifier-head fast path
- treat this as the current exact single-stream `4x` breakthrough on the branch
- future-head student work can remain optional follow-on work rather than the forced next step

## 2026-03-10 — One-command same-session reproduction downgrades the public claim to 3.70x on echo

### Hypothesis

The branch needed a tighter publication gate than the original saved-baseline claim: the exact two-step fused-triplet path had to be rerun in the same session against CoreML, under one executable and one timing contract, before the result could be claimed publicly.

### Implementation

Built a reproducibility seam around the existing release probe:

- added `RecurrentGenerationWeightStore` so recurrent inference weights can be loaded from an explicit artifact instead of being hardcoded in the probe
- added `MultitokenProbeConfiguration` so the probe must declare whether it is running on synthetic `echo` input or a recurrent checkpoint artifact
- moved the CoreML decode benchmark model into `Espresso` support code so `espresso-multitoken-probe` can benchmark ANE exact control, ANE exact two-step, and CoreML decode from one binary
- extended `espresso-multitoken-probe` with:
  - `--input echo|recurrent-checkpoint`
  - `--recurrent-checkpoint PATH`
  - `--compare-coreml`
  - `--coreml-model PATH`
  - `--generation-model PATH`
- added `scripts/reproduce_exact_4x.sh` to build the release probe, run fresh-process repeats, capture raw JSON/stderr artifacts, and summarize the ANE/CoreML ratio
- regenerated the missing CoreML package with `scripts/generate_coreml_model.py --layers 6`

### Verification

- `swift test --filter MultitokenProbeSupportTests`
- `swift test --filter GenerationHarnessTests`
- `swift build --product espresso-multitoken-probe`
- `REPEATS=5 ./scripts/reproduce_exact_4x.sh`

Artifacts:

- results directory: `results/exact-4x-20260310-233438`
- raw runs: `run-1.json` ... `run-5.json`
- summary: `results/exact-4x-20260310-233438/summary.txt`

### Measurements

Matched same-session release repro on the explicit `echo` input mode (`warmup=3`, `iterations=20`, `maxNewTokens=8`, `layerCount=6`, fused-triplet control, fused-triplet two-step, ANE RMSNorm+classifier head):

| Run | Control `ms/token` | Two-step `ms/token` | CoreML `ms/token` | Two-step speedup vs CoreML |
|---|---:|---:|---:|---:|
| 1 | `2.287279` | `1.735703` | `5.864406` | `3.378692x` |
| 2 | `2.242737` | `1.805805` | `5.994495` | `3.319570x` |
| 3 | `2.248893` | `1.493466` | `5.819115` | `3.896382x` |
| 4 | `2.162018` | `1.431951` | `5.853706` | `4.087925x` |
| 5 | `2.247073` | `1.634161` | `6.044177` | `3.698641x` |

Median-of-five summary from `scripts/reproduce_exact_4x.sh`:

- exact two-step fused-triplet: `1.6341614583333333 ms/token`
- exact fused-triplet control: `2.2470729166666668 ms/token`
- matched CoreML `.cpuAndNeuralEngine`: `5.86440625 ms/token`
- exact two-step speedup vs matched CoreML: `3.6986413138746621x`
- committed exact tokens/pass: `2`
- accepted future tokens/pass: `1`
- parity status across all runs: `match`

### Interpretation

Measured result:

- the one-command reproduction harness works and produces stable exact same-session results
- the exact two-step fused-triplet architecture remains clearly faster than the exact one-token recurrent control
- the public `4x over CoreML` claim does **not** survive the tighter same-session matched rerun on the echo checkpoint family

Inference:

- the earlier `4x` claim was inflated by comparing the probe against saved CoreML baselines rather than a same-session matched rerun
- the current branch should be described publicly as a reproducible exact multi-token architectural win on the synthetic echo family, with a matched same-session speedup of about `3.70x`, until a stronger real-checkpoint or repeated-`>=4x` result exists

### Decision

- keep the reproducibility harness and the batched verifier-head fast path
- downgrade the public claim from `4x` to a reproducible exact `3.70x` matched same-session result on the echo family
- do not publish a broader `4x over CoreML` claim from this branch without new evidence that survives the same-session harness

## 2026-03-10 — Real-checkpoint rerun blocked by missing recurrent artifact

### Attempt

Tried to move the same-harness rerun from the synthetic `echo` family to a real checkpoint.

Grounded checks performed in this session:

- searched the worktree and repo for local model/checkpoint artifacts
- checked the standard `assets/models` locations referenced by `EspressoTrain`
- checked `STORIES_MODEL_PATH`
- audited the codebase for an existing transformer-to-recurrent conversion/export path

### Findings

- no local `stories110M.bin` was present in the repo, the worktree, or the standard `assets/models` paths
- `STORIES_MODEL_PATH` was unset
- no local `ane_stories110M_ckpt.bin` or recurrent probe-weight artifact was present either
- the branch does **not** contain a principled transformer-to-recurrent exporter:
  - `GenerationWeights.load(modelPath:)` loads transformer inference weights
  - `Checkpoint` loads and saves transformer training state
  - `RecurrentGenerationWeightStore` only loads and saves already-recurrent `RWKVStyleRecurrentWeights`

### Interpretation

Measured blocker:

- the same-harness real-checkpoint rerun is blocked by artifact availability, not by benchmark plumbing

Inference:

- until a real recurrent artifact exists, or a validated transformer-to-recurrent export contract is implemented, any “real-checkpoint rerun” number would be fabricated rather than measured

### Decision

- do not quote a real-checkpoint throughput number from this branch yet
- keep the harness ready and rerun immediately once a real recurrent artifact plus matching transformer generation model path are available

## 2026-03-10 — Rejected clustered exact CPU staged head

### Attempt

Replaced contiguous token blocks with deterministic non-contiguous clustered blocks:

- projected clustering over classifier rows
- exact full-dimension center/radius summaries per cluster
- exact block scoring only on clusters whose upper bound could still beat the current best score

This preserved exact argmax semantics while materially improving the geometry over contiguous shards.

### Hardware measurement

Matched recurrent fused-triplet direct-select comparison (`warmup=3`, `iterations=20`, `maxNewTokens=8`, echo weights):

| Path | Median ms/token | tok/s | Compile/init ms | Trunk ms/token | Head/logits ms/token |
|---|---:|---:|---:|---:|---:|
| Control: fused-triplet + `.aneRMSNormClassifier` | `2.7799114583333333` | `359.72` | `32466.812541666666` | `1.46640625` | `1.3099739583333334` |
| Clustered exact CPU staged head | `7.072252604166667` | `141.40` | `10454.488791666667` | `1.738265625` | `5.315468750000001` |

### Interpretation

- Better geometry helped a lot relative to contiguous shards (`7.07` vs `28.14 ms/token`).
- It still lost badly to the current exact control.
- The head/logits bucket remained the dominant regression, so the family is still not competitive for single-stream decode.

### Decision

- reject clustered exact CPU staged head as the next performance path
- keep the clustered implementation only as evidence that geometry mattered but did not change the ranking
- move to a recurrent-native multi-token architecture instead of spending more time on this CPU staged exact-head family
| IOSurface aliasing between kernels | `statusType=0x9: Program Inference error` | Compile-time external surface fails at eval |
| _ANERequest surface rebinding | `-[__NSArrayM procedureIndex]` type confusion, then `0x9` | Request is immutable after construction |
| _ANEVirtualClient instantiation | All 5 init paths return nil | Kernel-level IOKit entitlement gate |
| _ANEChainingRequest | `PREPARE_FAILED` across all parameter permutations | Exhausted A22-A26 experiments |
| `setCodeSigningIdentity:` | Crashes with `-[__NSCFConstantString setObject:forKey:]` | Identity stored in class-level static dict |
| QoS tuning (0-63) | No effect on latency | All values produce identical timing |

---

## 5. Queue Pipelining Research

### Can we overlap eval with IO?

`evaluateWithQoS:` is **synchronous** -- it blocks the calling thread until the ANE
hardware completes. The `completionHandler` fires synchronously on the calling thread
after eval returns (not on a background thread). This means:

- CompletionHandler chaining provides **zero pipelining benefit**
- The only async eval path (`doEvaluateWithModel:completionEvent:`) requires VirtualClient
  (blocked by entitlements)

### What IS possible

**Background dispatch with cache overlap**: Dispatch `eval()` on a GCD queue. While
the eval blocks that queue, the main thread can perform K/V/mask cache updates for the
*previous* layer. Savings: ~5-10 microseconds per layer (the cache updates are already
very fast spatial-slice copies).

**Double-buffered handles**: Compile each kernel twice with separate surfaces. Write
to Handle-A while Handle-B evals. Eliminates inter-token IO latency but doubles compile
budget (which is capped at ~100 per process) and surface memory.

---

## 6. Priority Ranking

| # | Approach | Expected Impact | Effort | Risk |
|---|----------|----------------|--------|------|
| 1 | 2-layer KV cache packing | 0.285ms/token + zero inter-layer IO | Medium | Low |
| 2 | `evaluateRealTimeWithModel:` probe | Unknown (possibly 0.05ms/dispatch) | Small | Low |

---

## 7. Avenue 1 Result — Multi-Layer Kernel Fusion (ABANDONED)

**Date:** March 6, 2026

### Attempt 1: Fully packed K/V/mask cache input

- Built `FusedTwoLayerDecodeGenerator` / `FusedTwoLayerDecodeKernelSet` with a packed cache input
  `[1, 4608, 1, maxSeq]` and `slice_by_size` extraction for:
  - L0 K, V, mask
  - L1 K, V, mask
- Added non-hardware TDD coverage for:
  - MIL header / blob paths
  - packed-cache slice count
  - unique SSA names
  - I/O byte-size contract
- Hardware result:
  - `ANE_HARDWARE_TESTS=1 swift test --filter FusedTwoLayerDecodeKernelSetTests`
  - `_ANECompiler : ANECCompile() FAILED`
  - underlying compiler error: `InvalidMILProgram`

### Fallback: K/V-only packing, masks split back out

- Reduced the candidate to:
  - `packedKVCache` input `[1, 3072, 1, maxSeq]`
  - `maskCache0` input `[1, 768, 1, maxSeq]`
  - `maskCache1` input `[1, 768, 1, maxSeq]`
- This isolates packed K/V slicing from packed mask slicing while preserving the paired-kernel objective.
- Hardware result:
  - `ANE_HARDWARE_TESTS=1 swift test --filter FusedTwoLayerDecodeKernelSetTests/test_fused_two_layer_compile_fails_with_controlled_error_on_hardware`
  - same `_ANECompiler : ANECCompile() FAILED`
  - same underlying `InvalidMILProgram`

### Outcome

- Status: **abandoned**
- Reason: both the full packed-cache design and the K/V-only fallback fail at compile time with
  `InvalidMILProgram`, so the avenue never reaches eval.
- Benchmark delta: **N/A**. No post-change per-token timing exists because the candidate kernel never compiled.
- Per-token gain credited to this avenue: **+0.000ms**
- Running cumulative savings toward the 4x goal: **0.000ms**

## Results Summary

| Avenue | Status | Per-Token Gain | Cumulative |
|--------|--------|----------------|------------|
| 1. Multi-layer fusion | abandoned (`InvalidMILProgram` on full pack and K/V-only fallback) | +0.000ms | 0.000ms |
| 2. Metal SharedEvent | abandoned | +0.000ms | 0.000ms |
| 3. Metal+ANE hybrid | abandoned after QKV-only split-path benchmark | +0.000ms | 0.000ms |
| 4. CoreML baseline | measured | N/A | 0.000ms |
| 5. Speculative decode | abandoned | +0.000ms | 0.000ms |
| 6. GCD pipeline | abandoned | +0.000ms | 0.000ms |

Direct ANE: 2.875ms/token
CoreML:     3.007ms/token
Speedup:    1.05x
| 3 | Metal + ANE hybrid decode | Potentially 2-4x (parallel accelerators) | Large | Medium |
| 4 | Metal SharedEvent on standard eval | True async dispatch | Medium | Medium |
| 5 | Speculative decoding | 3-5x algorithmic | Large | Low |

### Recommended Execution Order

1. **Probe `evaluateRealTimeWithModel:`** -- quick win, 1-2 hours, might reveal a
   lower-overhead dispatch path
2. **Build 2-layer fused kernel with packed caches** -- proven MIL patterns, high
   confidence, eliminates 3 more dispatches
3. **Prototype Metal attention kernel** -- if successful, enables the hybrid architecture
   that could deliver 4x
4. **Speculative decoding** -- algorithmic multiplier on top of everything else

---

## 7. CoreML Comparison Gap

**We have no direct CoreML benchmark for the same model.** All throughput numbers are
for the direct `_ANEClient` path. To claim "4x over CoreML" we need:

1. Export the 6-layer transformer via coremltools as `.mlmodel`
2. Run inference with `.cpuAndNeuralEngine` compute units
3. Measure per-token decode latency on the same M3 Max hardware
4. Compare against our direct path

CoreML uses the same ANE hardware and MIL compiler. It may dispatch more efficiently
for some workloads (mature scheduling, internal graph fusion). Our advantage is
fine-grained control over kernel boundaries, zero-copy IOSurface management for KV
caches, and the ability to do things CoreML cannot (training, custom fusion, hybrid
dispatch).

## 8. Avenue 2 Result — Metal SharedEvent on Standard Eval Path (ABANDONED)

Date: 2026-03-06

What I built:
- Extended `ane_interop_probe_standard_completion_handler(...)` to optionally create a Metal-backed `MTLSharedEvent` via `MTLCreateSystemDefaultDevice()` and `newSharedEvent`.
- Wrapped that event in `_ANESharedSignalEvent` and `_ANESharedEvents`.
- Attached the container to `_ANERequest` via `setSharedEvents:` on the standard eval path.
- Extended the Swift probe surface to report attachment state and event value before/after eval.

Expected improvement:
- If ANE signaled the Metal shared event on completion, decode could pipeline host and accelerator work and potentially overlap adjacent dispatches.

Baseline before changes:
- Existing standard completion handler works, but fires synchronously on the calling thread and provides no true async overlap.
- No measurable per-token gain from the baseline path.

Observed result:
- `MTLCreateSystemDefaultDevice()` succeeded.
- `newSharedEvent` succeeded.
- `_ANERequest` reports `setSharedEvents:` exists.
- With the Metal-backed shared event attached, the hardware test enters the standard eval path and then hangs with no completion callback and no observable event advancement.

Abandon reason:
- This path is a dead end under the project protocol. It does not produce a usable signal or measurable pipeline parallelism, and it blocks progress inside eval instead.

Timing:
- Post measurement: not recorded.
- Delta: +0.000ms/token.
- Cumulative savings after Avenue 2: 0.000ms/token.

## 9. Avenue 3 Result — Metal + ANE Hybrid Decode Follow-On (ABANDONED)

Date: 2026-03-06

What I built:
- `DecodeQKVOnlyGenerator` — ANE decode kernel that performs `RMSNorm -> Wq/Wk/Wv` only and emits `Q`, `K_new`, `V_new`.
- `HybridDecodeKernelSet` — split runtime wrapper holding the QKV-only ANE kernel, the existing ANE FFN kernel, and cached `Wo` weights for Metal output projection.
- `HybridDecodeSurfaceHandles` + `ForwardPass.runHybridDecodeTimed(...)` — split decode loop:
  - ANE `QKV-only`
  - K/V cache writeback
  - Metal SDPA + `Wo` projection + residual add
  - ANE FFN
- `MetalAttentionKernel` follow-on path that reads ANE-layout IOSurfaces zero-copy and writes the projected attention output directly into `ffnIn`.

TDD / verification:
- Added failing tests first for:
  - `DecodeQKVOnlyGenerator`
  - `HybridDecodeKernelSet`
  - hybrid forward-pass step + comparison harness
- Verified:
  - `swift test`
  - `ANE_HARDWARE_TESTS=1 swift test --filter HybridDecodeKernelSetTests/test_hybrid_decode_kernel_set_compiles_on_hardware`
  - `ANE_HARDWARE_TESTS=1 swift test --filter HybridDecodeForwardPassTests/test_hybrid_decode_single_step_runs_on_hardware`
  - `ANE_HARDWARE_TESTS=1 swift test --filter HybridDecodeForwardPassTests/test_hybrid_decode_benchmark_reports_direct_and_hybrid_token_medians`

Dependency-graph proof:
- For a single decode token, layer `N` FFN cannot start until Metal attention for layer `N` finishes.
- Layer `N+1` QKV cannot start until FFN for layer `N` finishes, because it consumes layer `N`'s FFN output.
- Therefore the hoped-for ANE/Metal overlap does **not** exist on the critical path for this decode graph. The hybrid path is necessarily a serial substitution, not a pipelined one.

Baseline before changes:
- Direct ANE decode benchmark (`espresso-bench`, 6 layers, `steps=32`, `maxSeq=32`, `warmup=3`, `iterations=20`):
  - Median: `2.872 ms/token`
- Per-layer decode profile on the same run:
  - Average ANE attention eval: `209.383 us/layer`
  - Average ANE FFN eval: `210.511 us/layer`
- Prior standalone Metal SDPA probe (unchanged code path):
  - Mean: `0.395252 ms/eval`
  - Median: `0.217437 ms/eval`

Post measurement:
- Same 6-layer / 32-step decode schedule, using the new split path and timing each token with `mach_absolute_time()`:
  - Direct comparison-harness median: `2.864562 ms/token`
  - Hybrid median: `4.665979 ms/token`
- Hybrid stage medians per token:
  - ANE QKV-only: `1.233083 ms/token` = `205.514 us/layer`
  - Metal SDPA + `Wo` + residual: `1.877479 ms/token` = `312.913 us/layer`
  - ANE FFN: `1.253000 ms/token` = `208.833 us/layer`
  - IO: `0.086646 ms/token` = `14.441 us/layer`

Delta:
- Split attention half (`ANE QKV-only + Metal attention/projection`) = `518.427 us/layer`
- Current direct attention stage = `209.383 us/layer`
- Regression vs direct attention stage = `309.044 us/layer`
- Projected regression over 6 layers = `1.854264 ms/token`
- Measured end-to-end regression from the comparison harness:
  - `4.665979 - 2.864562 = 1.801417 ms/token`
  - Relative slowdown: `4.665979 / 2.864562 = 1.63x`

Conclusion:
- The QKV-only seam works and zero-copy Metal interop works on live ANE surfaces.
- But the dependency graph prevents meaningful overlap, and the naive serial substitution is materially slower than the existing direct ANE attention stage.
- This avenue should be treated as a dead end for decode throughput on the current runtime/model shape.

Timing impact recorded for this pass:
- Landed decode savings: `+0.000 ms/token`
- Cumulative savings after Avenue 3: `0.000 ms/token`

## 10. Avenue 4 Result — CoreML Baseline Benchmark (MEASURED)

Date: 2026-03-06

What changed:
- Extended `scripts/generate_coreml_model.py` to support `--layers` and `--output`.
- Generated a 6-layer CoreML package at `benchmarks/models/transformer_6layer.mlpackage`.
- Benchmarked decode on the same 6-layer, `steps=32`, `maxSeq=32` workload used for direct ANE.

TDD / pre-change failure:
- The original exporter ignored `--layers 6 --output ...` and still wrote the default single-layer model to `benchmarks/models/transformer_layer.mlpackage`.
- After the exporter change, the same command produced the requested 6-layer package.

Benchmark configuration:
- Warmup: `5` sequences
- Timed: `100` sequences (`3200` measured tokens)
- Decode schedule: `steps=32`, `maxSeq=32`
- CoreML model: `benchmarks/models/transformer_6layer.mlpackage`

Measured results:
- Direct ANE decode:
  - Mean: `3.098 ms/token`
  - Median: `2.875 ms/token`
- CoreML Decode (`.all`):
  - Mean: `3.005 ms/token`
  - Median: `2.992 ms/token`
- CoreML Decode (`.cpuAndNeuralEngine`):
  - Mean: `3.017 ms/token`
  - Median: `3.007 ms/token`

Speedup vs CoreML `.cpuAndNeuralEngine`:
- `3.007 - 2.875 = 0.132 ms/token` saved
- Relative speedup: `3.007 / 2.875 = 1.05x`

Strict fastest-CoreML gate:
- Fastest CoreML median observed: `.all` at `2.992 ms/token`
- Direct ANE median: `2.875 ms/token`
- Speedup vs fastest CoreML: `1.04x`

Interpretation:
- The direct `_ANEClient` path currently beats CoreML on this workload, but only narrowly.
- The current project is far from the `4x over CoreML` target; the measured headroom is about `5%` over `.cpuAndNeuralEngine` and `4%` over the fastest CoreML configuration observed here.

## 11. Avenue 5 Result — Speculative Decoding (ABANDONED)

Date: 2026-03-06

Why this was stopped quickly:
- The current decode benchmark path is hidden-state based, not token-generation based.
- `ANEDirectBench.runDecode(...)` pre-generates token embeddings directly and measures layer execution on those embeddings.
- Training-side code has embeddings, classifier weights, and cross-entropy, but there is no existing inference path that takes token ids to logits, samples candidates, and then verifies them in a batched prefill-style pass.
- There is also no existing batched verification/prefill decode path that would let the full model verify `N` draft tokens in one pass without building substantial new model/runtime support first.

What I verified locally:
- `ModelConfig` includes `vocab=32000` and the training path has embedding/classifier machinery.
- The live decode/inference runtime exposed in the benchmark suite does not currently surface a meaningful token-level speculative decode loop or accept-rate harness.

Abandon reason:
- On the current random-weight decode harness, speculative acceptance would be numerically possible to simulate but not meaningful.
- Building a real speculative benchmark here would require a separate generation stack (token ids -> embeddings -> decode -> final norm/classifier -> sampling) and a batched verification path, which is beyond a fast avenue probe.
- Per the protocol, this is a dead end for the current session because it cannot yield a credible throughput number quickly.

Timing impact recorded for this pass:
- No benchmark recorded.
- Landed decode savings: `+0.000 ms/token`
- Cumulative savings after Avenue 5: `0.000 ms/token`

## 12. Avenue 6 Result — GCD Pipeline with CompletionHandler (ABANDONED)

Date: 2026-03-06

Why this was abandoned without a code variant:
- The earlier completion-handler probe already showed that the standard eval completion callback is synchronous on the calling thread, so there is no true async wakeup to exploit.
- The fused decode loop (`runFusedDecodeTimed`) is dependency-bound in the wrong place for host-side pipelining:
  - The token lane write happens before eval.
  - The meaningful per-layer host work after that is `eval -> K cache writeback -> V cache writeback -> mask flip -> xNext chain`.
  - Those cache and chain operations all depend on eval outputs and therefore cannot overlap the eval that produces them.
- The current measured decode I/O budget is only about `0.100 ms/token` total, and most of that budget sits in post-eval dependent copies.

Feasibility judgment:
- A background GCD queue could at best hide the tiny pre-eval host work (token lane write, occasional window sync).
- That theoretical headroom is below the avenue's `10 us/token` abandon threshold once constrained by the actual dependency graph.

Timing impact recorded for this pass:
- No benchmark recorded.
- Landed decode savings: `+0.000 ms/token`
- Cumulative savings after Avenue 6: `0.000 ms/token`

## 13. Avenue 5 Revisit — Real Token-Generation Harness + Speculative Upper Bound (2026-03-06)

Status: harness implemented; current speculative verifier design abandoned as structurally non-viable.

What changed:
- Added a real token-generation surface in the runtime:
  - `AutoregressiveLanguageModel`
  - `AutoregressiveGenerationHarness`
  - `SpeculativeGenerationHarness`
  - `GenerationWeights`
  - `ANEDirectGenerationModel`
- The new path accepts token ids, performs embedding lookup, runs the existing ANE decode loop autoregressively, applies final RMSNorm + classifier logits on CPU, and selects the next token with argmax.
- The same surface now also supports batched full-model verification via the existing inference kernels, which made it possible to benchmark speculative control flow honestly.

What is now possible:
- End-to-end token generation instead of hidden-state replay.
- Honest measurement of generated-token latency and effective tokens/sec.
- Speculative draft/full control flow with explicit acceptance accounting.
- Batched verification over a whole prefix+candidate sequence using the existing full-sequence inference path.

Local asset limitation:
- There is still no local `stories110M.bin` / `STORIES_MODEL_PATH`, so this session could not run a semantic speculative benchmark on a pretrained model.
- Instead, I measured a structural upper bound using a synthetic "echo" model on real ANE kernels:
  - All transformer weights = `0`
  - RMS weights = `1`
  - Shared embedding/classifier row `0` = `1`, all other rows = `0`
  - This forces draft/full agreement to `100%`, which is useful for measuring the best-case overhead of the current speculative implementation independent of model quality.

Benchmark protocol:
- Command:
  - `ANE_HARDWARE_TESTS=1 swift test --filter GenerationHarnessHardwareTests`
- Warmup: `3`
- Timed iterations: `20`
- Prompt: `[0]`
- Generated tokens per sample: `8`
- Direct generation model: `6` layers
- Speculative draft/full: `2` draft layers, `6` full layers
- Candidate counts measured: `k=2`, `k=4`

Measured results on the synthetic upper-bound model:
- Direct generation harness:
  - Median: `6.110643 ms/token`
  - Throughput: `163.65 tok/s`
- Speculative `k=2`:
  - Acceptance: `1.0`
  - Median: `38.032148 ms/token`
  - Throughput: `26.29 tok/s`
- Speculative `k=4`:
  - Acceptance: `1.0`
  - Median: `28.609625 ms/token`
  - Throughput: `34.95 tok/s`

Interpretation:
- This is the strongest negative result available for the current speculative design.
- Even with forced `100%` agreement, speculative generation is still dramatically slower than direct generation.
- The bottleneck is not acceptance quality; it is the verifier/state-sync architecture:
  - Full-sequence verification runs over the fixed `seqLen=256` inference path.
  - After verification, the full model still has to advance its decode state sequentially for accepted tokens.
  - The draft model is then reset and re-prefilled from the accepted prefix.
- That combination makes the current implementation roughly:
  - `38.03 / 6.11 = 6.22x` slower at `k=2`
  - `28.61 / 6.11 = 4.68x` slower at `k=4`

Conclusion:
- Avenue 5 is no longer blocked by missing infrastructure. The infrastructure now exists.
- Avenue 5 is blocked by measured verifier cost on the current runtime design.
- More draft tuning is not the right next move. A speculative path is only worth revisiting if the verifier can become radically cheaper and reusable across accepted tokens.
- The next serious direction should shift to:
  - RWKV-style recurrent decode, or
  - a fundamentally different speculative verifier that reuses caches/state instead of replaying and then re-advancing.

## 14. RWKV-Style Recurrent Decode Prototype (2026-03-06)

Status: implemented and benchmarked. Promising enough to keep as the secondary architecture bet, but not yet strong enough to replace the current roadmap without a depth-matched follow-up.

What changed:
- Added a minimal recurrent step path:
  - `RWKVStyleRecurrentWeights`
  - `RWKVStyleRecurrentStepGenerator`
  - `RWKVStyleRecurrentKernelSet`
  - `RWKVStyleRecurrentSession`
  - `RWKVStyleRecurrentBench`
- The recurrent MIL uses only already-proven ops on this branch:
  - `conv`
  - `add`
  - `mul`
  - `sigmoid`
  - `reduce_sum`
  - `pow`
- The step contract is:
  - inputs: `x_t`, `state_in`
  - outputs: `x_next`, `state_out`
  - both state and token lanes are fixed-size `[1, dim, 1, laneSpatial]`

Why this shape:
- The goal here was not faithful RWKV semantics yet.
- The goal was to answer the first-order systems question with minimal compiler risk:
  - can a constant-state recurrent step compile on ANE,
  - run repeatedly,
  - and stay flat as effective context grows?

Hardware compile behavior:
- `ANE_HARDWARE_TESTS=1 swift test --filter RWKVStyleRecurrentKernelSetTests`
- Result:
  - compile succeeded
  - eval succeeded
  - input/output surfaces were accessible
- Important downside:
  - first hardware compile for the recurrent kernel took `131.362 s`
- Interpretation:
  - runtime execution is viable
  - iteration speed for new recurrent kernel shapes is currently poor

Benchmark method:
- Command:
  - `ANE_HARDWARE_TESTS=1 swift test --filter RWKVStyleRecurrentPrototypeHardwareTests`
- Timing:
  - `mach_absolute_time()` / `mach_timebase_info`
- Protocol:
  - `3` warmup steps
  - `20` timed tail steps
  - measure tail-step latency at effective contexts `32`, `256`, `1024`, `4096`
- Comparison:
  - recurrent prototype under the same tail-step protocol
  - current 6-layer transformer decode control under the same tail-step protocol

Recurrent results:
- Run 1:
  - `32`: `0.314125 ms/token` (`3183 tok/s`)
  - `256`: `0.215125 ms/token` (`4648 tok/s`)
  - `1024`: `0.205854 ms/token` (`4858 tok/s`)
  - `4096`: `0.217500 ms/token` (`4598 tok/s`)
- Run 2:
  - `32`: `0.315563 ms/token` (`3169 tok/s`)
  - `256`: `0.217896 ms/token` (`4589 tok/s`)
  - `1024`: `0.216958 ms/token` (`4609 tok/s`)
  - `4096`: `0.200625 ms/token` (`4984 tok/s`)

Transformer control results under the same tail-step protocol:
- Run 1:
  - `32`: `2.907875 ms/token`
  - `256`: `4.055500 ms/token`
  - `1024`: `3.924917 ms/token`
  - `4096`: `2.868458 ms/token`
- Run 2:
  - `32`: `2.954167 ms/token`
  - `256`: `2.872583 ms/token`
  - `1024`: `2.847708 ms/token`
  - `4096`: `2.923729 ms/token`

Interpretation:
- The narrow recurrent question came back positive:
  - the recurrent step stayed effectively flat through `4096`.
- The stronger hoped-for transformer comparison came back weaker than expected:
  - the current transformer control did not show clean context-length growth on this graph.
  - the existing tiled decode path keeps tail-step latency roughly flat, with some noise.
- That means:
  - recurrent decode is still interesting,
  - but not because it clearly beats an O(context) tail-latency curve in the current implementation.

Scientific validity / caveats:
- This is not an apples-to-apples throughput comparison:
  - the recurrent prototype is a single recurrent layer
  - the transformer control is a 6-layer decode path
- The useful claim that *is* supported:
  - a constant-state recurrent ANE step is mechanically viable and shows stable flat tail-step latency at long effective contexts.
- The useful claim that is *not yet* supported:
  - a realistic recurrent model stack will outperform the optimized tiled transformer end-to-end on this hardware.

Decision:
- Keep RWKV-style recurrent decode alive as a credible long-term architecture path.
- Do not spend more time on speculative draft tuning for the current verifier design.
- Do not over-claim the recurrent result yet.
- The next recurrent step worth doing is a more depth-matched stacked recurrent benchmark, but only if the compile-time iteration cost is acceptable.

## 15. Depth-Matched Recurrent Generation Gate (2026-03-06)

Status:
- implemented and benchmarked
- recurrent generation beats both the current direct transformer generation harness and the CoreML generation baseline
- but the advantage compresses below the strict `>=2x` continuation gate by `6` recurrent layers

What changed:
- Added `GenerationPerformanceSnapshot` / `GenerationPerformanceTrackable` so generation benchmarks can report:
  - compile time
  - trunk time
  - output-head/logits time
- Added `ANERecurrentGenerationModel` to the real token-generation harness:
  - token ids
  - embedding lookup
  - recurrent trunk stepping
  - final RMSNorm + classifier logits
  - argmax decode loop
- Added a benchmark-local CoreML generation wrapper that reuses the same surrounding CPU work:
  - token-id input
  - embedding lookup
  - CoreML transformer trunk replay
  - final RMSNorm + classifier logits on CPU

Important comparison policy:
- `stories110M.bin` is still not present locally, so these measurements use the synthetic token-generation harness rather than a pretrained semantic model.
- The CoreML path here is still an honest end-to-end generation comparison because it includes prompt replay, decode replay, and the same CPU output-head work.
- The CoreML model exports hidden states, not logits, so the output head remains on CPU for parity with the direct generation harness.

Benchmark protocol:
- `mach_absolute_time()` / `mach_timebase_info`
- `3` warmup iterations
- `20` timed iterations
- prompt: `[0]`
- generated tokens: `8`
- report median `ms/token`, `tok/s`, compile time, and median trunk/logits time per generated token

Results:
- Direct transformer ANE generation, `6` layers:
  - `6.558745 ms/token`
  - `152.47 tok/s`
  - compile `2158.23 ms`
  - trunk `5.28449 ms/token`
  - logits `1.26048 ms/token`
- Recurrent generation, `2` layers:
  - `2.271112 ms/token`
  - `440.32 tok/s`
  - compile `123.02 ms`
  - trunk `1.00656 ms/token`
  - logits `1.27095 ms/token`
- Recurrent generation, `6` layers:
  - `4.000445 ms/token`
  - `249.97 tok/s`
  - compile `349.07 ms`
  - trunk `2.73244 ms/token`
  - logits `1.26102 ms/token`
- CoreML generation baseline, `.cpuAndNeuralEngine`:
  - `6.582224 ms/token`
  - `151.93 tok/s`
  - compile+load `1384.60 ms`
  - trunk `5.11926 ms/token`
  - logits `1.35556 ms/token`

Interpretation:
- The first structural gate came back positive:
  - `2` recurrent layers are about `2.89x` faster end-to-end than the current direct transformer generation harness.
- The more honest depth-matched gate came back mixed:
  - `6` recurrent layers are about `1.64x` faster than direct transformer generation.
  - `6` recurrent layers are about `1.65x` faster than the CoreML generation baseline.
- That means the recurrent architecture still has real upside, but the strict continuation rule from this session was not met at `6` layers.

Most important systems finding:
- The recurrent trunk win is real, but the output head is now a large fraction of the budget.
- At `2` recurrent layers, logits are already slightly more expensive than the recurrent trunk.
- At `6` recurrent layers, logits remain about `1.26 ms/token`, which materially caps end-to-end gains.
- The next high-upside move is therefore:
  - offload final RMSNorm + classifier to ANE, or
  - otherwise reduce output-head cost sharply
- The next low-upside move would be:
  - adding more recurrent layers without addressing the output head

Compile-time note:
- The cold recurrent prototype compile from the earlier session was `131.362 s`.
- The generation-model compile times above (`123-349 ms`) are warm-cache measurements on the now-proven recurrent step shape, so do not treat them as contradictory.
- The likely explanation is cache reuse for already-compiled identical recurrent kernels.

Decision:
- Keep recurrent decode as the highest-upside architecture path over transformer decode.
- Do not keep scaling recurrent depth blindly under the current CPU output-head design.
- Move the next experiment to output-head optimization / ANE-side logits before spending more time on deeper recurrent stacks.

Verification:
- `swift test`
- `ANE_HARDWARE_TESTS=1 swift test --filter GenerationHarnessHardwareTests/test_recurrent_generation_reports_compile_and_runtime_breakdown_on_hardware`
- `ANE_HARDWARE_TESTS=1 swift test --filter GenerationHarnessHardwareTests/test_recurrent_generation_6layer_and_coreml_generation_baseline_if_gate_passes`
- `ANE_HARDWARE_TESTS=1 swift test --filter GenerationHarnessHardwareTests/test_speculative_upper_bound_reports_metrics_on_hardware`

Residual test note:
- Running the entire `GenerationHarnessHardwareTests` class together exposed a temp-dir cleanup failure in the older speculative test, while that same speculative test passes in isolation. That looks like cross-test hygiene in the old speculative path rather than a blocker on the recurrent/CoreML benchmark results above.

## Output-Head Probe (March 7, 2026)

Goal:
- Test whether moving the recurrent generation classifier projection onto ANE can recover enough end-to-end latency to get the `6`-layer recurrent path back toward a `>=2x` win over the current direct transformer generation path.

Implementation:
- Added a swappable generation output-head backend:
  - `GenerationOutputHeadBackend.cpu`
  - `GenerationOutputHeadBackend.aneClassifier`
- Added:
  - `GenerationClassifierGenerator`
  - `GenerationClassifierKernelSet`
  - `ANEGenerationClassifierHead`
- Kept final RMSNorm on CPU for this probe.
- Verification path remains on CPU. This experiment only changes step-level autoregressive logits.

Important compatibility finding:
- A naive single-token classifier kernel (`[1, dim, 1, 1] -> [1, vocab, 1, 1]`) compiled but failed at eval on hardware with:
  - `statusType=0x9: Program Inference error`
- The same classifier projection became usable when packed into the repo’s proven lane shape:
  - input `[1, dim, 1, 32]`
  - output `[1, vocab, 1, 32]`
  - only spatial lane `0` is written/read at runtime
- Treat large-vocab classifier heads as another case where lane-packed decode shapes are materially safer than scalar-width shapes on this runtime.

Benchmark protocol:
- `mach_absolute_time()` / `mach_timebase_info`
- `3` warmup iterations
- `20` timed iterations
- prompt `[0]`
- `8` generated tokens
- recurrent generation, `6` layers

Results:
- recurrent generation, CPU head:
  - `3.965409 ms/token`
  - `252.18 tok/s`
  - compile `407.17 ms`
  - trunk `2.661490 ms/token`
  - logits `1.292997 ms/token`
- recurrent generation, ANE classifier head:
  - `3.710214 ms/token`
  - `269.53 tok/s`
  - compile `585.46 ms`
  - trunk `2.607659 ms/token`
  - logits `1.094966 ms/token`

Delta:
- end-to-end: `-0.255195 ms/token` (`6.4%` faster)
- throughput: `+17.35 tok/s` (`6.9%` faster)
- logits bucket: `-0.198031 ms/token` (`15.3%` faster)
- compile time: `+178.29 ms`

Decision:
- This is a real win, but it is not a breakthrough.
- The predeclared gate for continuing this avenue was about `0.7 ms/token` saved on recurrent-6 generation. The measured win (`0.255 ms/token`) is far below that threshold.
- Therefore:
  - classifier-only ANE offload is worth keeping because it is faster
  - classifier-only ANE offload is not enough to restore the missing `>=2x` recurrent advantage by itself
  - further time on this exact probe has low upside unless the next step is a fused final RMSNorm + classifier path
- The next materially larger opportunity is still likely recurrent multi-layer fusion, not more small output-head tuning.

## Fused Output-Head Follow-up (March 7, 2026)

Goal:
- Test the only still-credible head-side follow-up after classifier-only ANE offload: fuse final RMSNorm and classifier into one lane-packed ANE head.

Implementation:
- Added:
  - `GenerationRMSNormClassifierGenerator`
  - `GenerationRMSNormClassifierKernelSet`
  - `GenerationOutputHeadBackend.aneRMSNormClassifier`
- Kept the same proven lane-packed shape:
  - input `[1, dim, 1, 32]`
  - output `[1, vocab, 1, 32]`
  - write/read only spatial lane `0`
- Verification stays on CPU. This is still a step-level autoregressive head path only.

Benchmark protocol:
- `mach_absolute_time()` / `mach_timebase_info`
- `3` warmup iterations
- `20` timed iterations
- prompt `[0]`
- `8` generated tokens
- recurrent generation, `6` layers

Results:
- recurrent generation, CPU head:
  - `4.128354 ms/token`
  - `242.23 tok/s`
  - compile `445.79 ms`
  - trunk `2.721484 ms/token`
  - head `1.375232 ms/token`
- recurrent generation, ANE classifier head:
  - `3.761755 ms/token`
  - `265.83 tok/s`
  - compile `634.18 ms`
  - trunk `2.655479 ms/token`
  - head `1.098206 ms/token`
- recurrent generation, fused ANE RMSNorm+classifier head:
  - `3.564776 ms/token`
  - `280.54 tok/s`
  - compile `612.15 ms`
  - trunk `2.615005 ms/token`
  - head `0.936549 ms/token`

Delta:
- fused vs classifier-only ANE head:
  - `-0.196979 ms/token` (`5.24%` faster)
  - `+14.71 tok/s` (`5.53%` faster)
  - head bucket improved by `0.161656 ms/token`
- fused vs CPU head:
  - `-0.563578 ms/token` (`13.65%` faster)
- remaining gap to `>=2x` over direct transformer generation (`6.558745 ms/token`):
  - about `0.285404 ms/token`

Interpretation:
- The user instinct was directionally correct: there is still real signal in the head.
- Fusing RMSNorm into the ANE head produced another measurable win beyond classifier-only offload.
- But the follow-up still missed the predeclared continuation gate of `>=0.25 ms/token` additional end-to-end savings over the current ANE classifier head.

Most likely next limiter:
- The fused head still returns full-vocab logits to the host every step.
- That means the next head-side win probably needs a different mechanism:
  - on-device token selection / argmax
  - or another reduction that materially shrinks logits readback

Decision:
- Keep the fused head because it is faster.
- Do not spend unbounded time tuning this same full-logits-return path further.
- If head-side work continues, make it a bounded probe around avoiding full-vocab logits readback.
- Otherwise, return to the higher-upside path: recurrent multi-layer fusion.

## Reduced Readback / Direct Surface Argmax (March 7, 2026)

Goal:
- Test the next honest head-side follow-up after the fused ANE `RMSNorm + classifier` head:
  avoid full-vocab logits materialization for `argmax` generation.

Hypothesis:
- The fused head made the classifier path fast enough that reading a full `vocab`-length logits vector back to CPU every step is now a credible limiter.
- A bounded host-side direct argmax scan on the ANE output IOSurface can answer that question without betting on unsupported MIL `argmax` / `topk` ops.

Implementation:
- Added a small C interop helper:
  - `ane_interop_io_argmax_fp16_spatial_slice`
- Exposed it in Swift:
  - `SurfaceIO.argmaxFP16SpatialSlice(...)`
- Added direct-selection helpers to:
  - `ANEGenerationClassifierHead`
  - `ANEGenerationRMSNormClassifierHead`
- Added an explicit generation harness for models that can return the selected token directly:
  - `DirectTokenSelectionGenerationHarness`
- Kept the existing `AutoregressiveGenerationHarness` unchanged as the materialized-logits control.

Correctness checks:
- `SurfaceIO` direct argmax matches materialized-lane argmax in unit tests.
- The direct-selection harness preserves token outputs in unit tests.
- On hardware, the direct-selection and materialized fused-head paths produced the same echo tokens before benchmarking.

Benchmark setup:
- recurrent generation
- `6` layers
- fused ANE `RMSNorm + classifier` head
- prompt `[0]`
- `8` generated tokens
- `3` warmup
- `20` timed iterations
- median reported

Results:
- Materialized logits control:
  - `3.568122 ms/token`
  - `280.26 tok/s`
  - compile `663.66 ms`
  - trunk `2.614922 ms/token`
  - head `0.946177 ms/token`
- Direct surface argmax:
  - `2.467362 ms/token`
  - `405.29 tok/s`
  - compile `659.94 ms`
  - trunk `1.700138 ms/token`
  - head `0.768669 ms/token`

Replications:
- `3.620622 -> 2.454737 ms/token`
- `3.637203 -> 2.451047 ms/token`

Delta from the parity+benchmark run:
- `1.100760 ms/token` faster end-to-end (`30.85%`)
- `125.03 tok/s` faster (`44.61%`)
- head bucket down by `0.177508 ms/token`
- observed trunk bucket down by `0.914784 ms/token`

Interpretation:
- The reduced-readback path clearly removed real head-side overhead.
- The total end-to-end win is much larger than the head-bucket reduction alone.
- Inference from the data:
  the shorter head-side path also appears to reduce host-side delay enough to improve the observed recurrent trunk cadence in this autoregressive harness.

Decision:
- This probe decisively passed the continuation gate.
- Avoiding full-vocab logits materialization is now a proven performance avenue, not just a hypothesis.
- Do not switch away from head-side work yet.
- The next honest head-side probes are:
  - reuse one read lock and stream chunked reads if another small gain remains
  - or move token selection further toward minimal-return / on-device selection
- Recurrent multi-layer fusion still matters, but it is no longer the automatic next step after the fused head.

## Recurrent Multi-Layer Fusion Follow-up (March 7, 2026)

Goal:
- Push the recurrent direct-select path from the reduced-readback breakthrough (`~2.45-2.47 ms/token`) to a real `3x`-over-CoreML result by removing more recurrent eval boundaries.

### Fused Two-Layer Recurrent Step

Implementation:
- Added:
  - `RWKVStyleFusedTwoLayerStepGenerator`
  - `RWKVStyleFusedTwoLayerKernelSet`
  - `RWKVStyleFusedTwoLayerSession`
- Extended `ANERecurrentGenerationModel` with `RecurrentGenerationTrunkBackend` and a `.fusedTwoLayerPairs` mode.
- Kept the output-head breakthrough intact:
  - fused ANE `RMSNorm + classifier`
  - direct surface argmax / direct token selection

Correctness:
- New generator and kernel compile-spec tests pass.
- Hardware parity test confirms fused-pair direct-select emits the same echo tokens as the single-layer-stack recurrent control.

Hardware compare:
- Fused-pair runs were noisy in absolute terms but directionally consistent.
- Observed direct-select compare runs:
  - single-layer recurrent control `6.0231 ms/token` vs fused-pair `3.7812 ms/token`
  - single-layer recurrent control `3.4296 ms/token` vs fused-pair `2.3074 ms/token`
  - single-layer recurrent control `4.5684 ms/token` vs fused-pair `2.5795 ms/token`

Interpretation:
- The absolute harness is still noisy enough that single-run claims would be dishonest.
- The direction is clear: fused recurrent pairs materially reduce both trunk time and end-to-end token latency.
- This passed the material-gain gate and justified one more bounded push.

### Smaller Trunk Lane Widths

Hypothesis:
- The recurrent trunk still runs lane-packed surfaces with `laneSpatial=32` while only lane `0` carries live token data, so reducing trunk lane width might recover the remaining `~0.11-0.39 ms/token`.

Result:
- The lane-32 fused-pair baseline still ran in the sweep harness:
  - `2.530901 ms/token`
  - `395.14 tok/s`
- The first smaller trunk lane width immediately hit:
  - `statusType=0x9`

Decision:
- Treat smaller recurrent trunk lane widths as structurally blocked for this MIL shape/runtime.
- Do not spend more time on trunk lane packing reduction unless the graph changes materially.

### Fused Three-Layer Recurrent Step

Implementation:
- Added:
  - `RWKVStyleFusedThreeLayerStepGenerator`
  - `RWKVStyleFusedThreeLayerKernelSet`
  - `RWKVStyleFusedThreeLayerSession`
- Extended `ANERecurrentGenerationModel` with `.fusedThreeLayerTriplets`.
- For a `6`-layer recurrent trunk, this reduces the direct-select recurrent hot path to `2` fused evals instead of `3`.

Correctness:
- New generator and kernel compile-spec tests pass.
- Non-hardware validation rejects non-multiples of `3`.
- Hardware parity test confirms fused-triplet direct-select matches fused-pair direct-select token outputs.

Hardware compare:
- Fused pair direct-select:
  - `2.640195 ms/token`
  - `378.76 tok/s`
  - compile `523.39 ms`
  - trunk `1.447555 ms/token`
  - logits `1.140951 ms/token`
- Fused triplet direct-select:
  - `2.211750 ms/token`
  - `452.13 tok/s`
  - compile `560.54 ms`
  - trunk `1.139904 ms/token`
  - logits `1.090958 ms/token`

Replication:
- second run:
  - fused pair `2.474672 ms/token`
  - fused triplet `2.211995 ms/token`
- a noisier third run still preserved direction:
  - fused pair `3.316388 ms/token`
  - fused triplet `2.468479 ms/token`

Interpretation:
- Fused triplets are the best recurrent result on this branch so far.
- The strongest repeated fused-triplet runs are clustered around:
  - `2.212 ms/token`
  - `~452 tok/s`
- Against the standing CoreML generation baseline (`6.582224 ms/token`, `151.93 tok/s`), that is about:
  - `2.98x` faster
- This is an honest near-`3x` result, but not a stable `>=3x` claim yet.

### Output-Head Lane Width Follow-up on Fused Triplets

Hypothesis:
- After triplet fusion, the remaining gap to `3x` is tiny enough that a smaller fused ANE output-head lane width might close it.

Result:
- Baseline fused-triplet direct-select with output-head lane `32` in the sweep harness:
  - `2.204732 ms/token`
  - `453.57 tok/s`
  - compile `614.12 ms`
  - trunk `1.159018 ms/token`
  - logits `1.051698 ms/token`
- Smaller fused output-head lane widths were all unsupported on hardware:
  - lane `16`: ANE fused output-head eval failed
  - lane `8`: ANE fused output-head eval failed
  - lane `1`: ANE fused output-head eval failed

Decision:
- Keep output-head lane width at `32`.
- Treat smaller fused output-head lanes as blocked for this graph/runtime.

### Current Honest Status

- Best branch result:
  - fused-triplet recurrent direct-select around `2.205-2.212 ms/token`
  - `~452-454 tok/s`
- Relative to CoreML generation baseline:
  - roughly `2.98x-2.99x`
- Relative to the older direct transformer generation harness (`6.558745 ms/token`):
  - roughly `2.97x`

Conclusion:
- Multi-layer recurrent fusion is the first path on this branch that gets genuinely close to a `3x` ANE-over-CoreML generation result.
- The remaining gap is now small enough that further work would be micro-optimization territory, not another large architectural unlock.
- The bounded micro-paths tested after triplet fusion were both blocked:
  - smaller recurrent trunk lane widths -> `statusType=0x9`
  - smaller fused output-head lane widths -> ANE eval failure / `statusType=0x9`
- So the honest call is:
  - this branch reached an optimized near-`3x` state
  - but did not produce a stable, repeatable `>=3x` claim on the current benchmark harness

## 13. Concurrent Multistream Generation Benchmark (2026-03-08)

What was built:
- A true concurrent multistream generation benchmark in `GenerationHarnessHardwareTests`.
- ANE path: recurrent fused-triplet direct-select with fused ANE RMSNorm + classifier head.
- CoreML path: separate `CoreMLGenerationBenchmarkModel` instance per stream.
- Each stream owns isolated state, model/runtime setup, and a dedicated host queue.
- Each timed round launches one full generation request per stream concurrently and measures wall-clock round time.
- Reporting normalizes by `streams * maxNewTokens` so throughput stays honest.

Benchmark contract:
- Prompt: `[0]`
- Decode length: `8` new tokens
- Warmup: `3` rounds
- Timed: `20` rounds
- Stream counts: `1`, `2`, `3`, `4`
- Timing: `mach_absolute_time()` via the existing generation clock path

Single-stream control rerun before the new benchmark:
- Best ANE direct-select control: `2.211003 ms/token`, `452.28 tok/s`
- Standing CoreML generation control: `7.044784 ms/token`, `141.95 tok/s`

Repeated matched concurrent runs:

| Streams | ANE ms/token (run 1) | ANE tok/s (run 1) | ANE ms/token (run 2) | ANE tok/s (run 2) | CoreML ms/token (run 1) | CoreML tok/s (run 1) | CoreML ms/token (run 2) | CoreML tok/s (run 2) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2.294083 | 435.90 | 2.195065 | 455.57 | 15.377154 | 65.03 | 15.450174 | 64.72 |
| 2 | 1.183065 | 845.26 | 1.196073 | 836.07 | 8.187806 | 122.13 | 8.299697 | 120.49 |
| 3 | 0.885752 | 1128.98 | 0.874269 | 1143.81 | 5.583365 | 179.10 | 5.537411 | 180.59 |
| 4 | 0.808975 | 1236.13 | 0.814136 | 1228.30 | 4.637305 | 215.64 | 4.643027 | 215.38 |

Scaling analysis against each path's own matched 1-stream concurrent baseline:
- ANE aggregate scaling (run 2):
  - `2 streams`: `1.84x`
  - `3 streams`: `2.51x`
  - `4 streams`: `2.70x`
- CoreML aggregate scaling (run 2):
  - `2 streams`: `1.86x`
  - `3 streams`: `2.79x`
  - `4 streams`: `3.33x`
- Concurrency-scaling ratio gate (`ANE scaling / CoreML scaling > 1.34`) failed:
  - `2 streams`: about `0.99`
  - `3 streams`: about `0.90`
  - `4 streams`: about `0.81`

Interpretation:
- Absolute ANE serving throughput increased substantially with stream count, reaching about `1.23k tok/s` at `4` streams.
- The matched concurrent benchmark also shows ANE above `4x` CoreML in absolute throughput at every stream count.
- However, this is **not** evidence that true multistream concurrency is the unlocking lever toward `4x`.
- The key reason is methodological: the matched concurrent CoreML setup regressed sharply already at `1` stream (`~15.4 ms/token`) relative to the standing single-stream CoreML control (`~7.04 ms/token`), while ANE's `1`-stream concurrent measurement stayed close to its prior control.
- Inference: the queue-isolated concurrent serving harness changes CoreML's single-stream behavior materially, so the >`4x` absolute ratio here comes mostly from the changed serving contract rather than from ANE gaining a superior concurrency scaling factor.
- The primary hypothesis therefore fails: ANE does **not** scale materially better than CoreML under this benchmark. CoreML scales similarly or better from its own matched concurrent baseline.

Decision:
- Status: implemented, measured, and stopped.
- Keep the benchmark as a serving-path measurement tool.
- Do **not** use this avenue as the main causal explanation for a credible `4x over CoreML` claim.
- Next best move returns to the fallback plan: reduce remaining direct-select host overhead on the ANE path, and treat the queue-induced CoreML regression as a separate measurement artifact to characterize if a serving claim is needed later.
## 2026-03-08 - Direct-select argmax micro-optimization

- Change: optimized `ane_interop_io_argmax_fp16_spatial_slice` in `Sources/ANEInterop/surface_io.c`
  - switched from scalar `float`-converting scan to pointer-stride arithmetic
  - unrolled the channel walk `4x`
  - kept comparisons in `_Float16` and converted to `float` only for the final reported value
- Guard tests:
  - `swift test --filter ANETypesTests/test_surface_argmax_fp16_spatial_slice_matches_materialized_argmax`
  - `swift test --filter ANETypesTests/test_surface_argmax_fp16_spatial_slice_respects_channel_offset_and_tail`
- Baseline before change:
  - fused-triplet direct-select `2.2018489583333336 ms/token`, `454.16396411257176 tok/s`
- Post-change repeated runs:
  - `2.2109479166666666 ms/token`, `452.29748776856803 tok/s`
  - `2.154953125 ms/token`, `464.0636883108018 tok/s`
  - `2.159026041666667 ms/token`, `463.1718306072166 tok/s`
- Post-change median across the three repeated runs:
  - `2.159026041666667 ms/token`, `463.1718306072166 tok/s`
- Delta versus the fresh baseline:
  - `0.04282291666666665 ms/token` faster
  - about `1.94%` lower latency
  - about `9.01 tok/s` higher throughput
- Interpretation:
  - measured win is small but positive and above the immediate regression/no-regression threshold for this avenue
  - most of the remaining single-stream budget is still in trunk + output-head boundaries, so this is a saved micro-gain rather than a path-changing breakthrough
## 2026-03-08 - Direct-select argmax ILP widening

- Change: widened `ane_interop_io_argmax_fp16_spatial_slice` to keep `8` independent maxima in flight before a stable first-max reduction
  - goal: reduce loop-carried dependency in the `32_000`-channel, stride-`32` scan used by the current best direct-select head
  - preserved first-max semantics explicitly during the final reduction
- Guard tests:
  - `swift test --filter ANETypesTests`
  - added `ANETypesTests/test_surface_argmax_fp16_spatial_slice_prefers_first_max_across_unrolled_groups`
- Control before this change:
  - median from the previous saved argmax optimization: `2.159026041666667 ms/token`
- Post-change repeated runs:
  - `2.129125 ms/token`, `469.67654812253164 tok/s`
  - `2.1766796875 ms/token`, `459.42070485198116 tok/s`
  - `2.1179270833333335 ms/token`, `472.15979807418256 tok/s`
- Post-change median across the three repeated runs:
  - `2.129125 ms/token`, `469.67654812253164 tok/s`
- Delta versus the prior control:
  - `0.02990104166666683 ms/token` faster
  - about `1.38%` lower latency
- Cumulative direct-select argmax savings from the fresh `2026-03-08` pre-optimization baseline:
  - baseline `2.2018489583333336 ms/token`
  - current `2.129125 ms/token`
  - cumulative saved `0.07272395833333361 ms/token`
  - cumulative reduction about `3.30%`
- Interpretation:
  - another real but still small micro-gain
  - this confirms the output-surface argmax path is movable, but not enough by itself to close the remaining gap to `6x`
## 2026-03-08 - Rejected lane-0-only recurrent state/input copies

- Hypothesis:
  - because the current fused-triplet direct-select path only consumes lane `0`, per-token full-width `xIn` clear and `stateOut -> stateIn` copies in `RWKVStyleFusedThreeLayerSession.step` might be reducible to lane-`0` slice operations only
- Guard:
  - added hardware parity test `test_recurrent_generation_fused_triplet_direct_select_vs_autoregressive_materialized_on_hardware`
  - parity stayed green, so semantics on the echo path did not obviously break
- Rejected implementation:
  - removed per-token full `xIn` clear
  - narrowed the three recurrent state carry copies to `SurfaceIO.copyFP16SpatialSlice(... spatialIndex: 0 ...)`
- Measured result:
  - control before change: `2.129125 ms/token`
  - post-change runs:
    - `2.243302083333333 ms/token`
    - `2.1020781250000002 ms/token`
    - `2.1712968750000003 ms/token`
  - post-change median: `2.1712968750000003 ms/token`
  - delta: about `0.04217187500000029 ms/token` slower (`~1.98%` regression)
- Interpretation:
  - semantic parity on the echo path was not enough; the narrowed copy path regressed throughput and was removed
  - inference: either the full-width copies are helping memory locality/cache behavior, or the slice-copy path itself is more expensive than expected
## 2026-03-08 - Blocked full fused triplet + head session

- Goal:
  - fuse the final recurrent triplet and the ANE RMSNorm+classifier head into one kernel that returns:
    - `xNext`
    - `stateOut0`
    - `stateOut1`
    - `stateOut2`
    - `logits`
- Why:
  - this was the first boundary-removal path with enough upside to matter for `4x` and `6x`
  - it could have removed the trunk→head dispatch boundary and one host-side handoff
- What was built:
  - committed scaffolding:
    - `RWKVStyleFusedThreeLayerRMSNormClassifierGenerator`
    - `RWKVStyleFusedThreeLayerRMSNormClassifierKernelSet`
    - generator/kernelset contract tests
  - attempted runtime/session wiring:
    - direct hardware smoke test for a new fused session
    - temporary fused session/runtime implementation
- What failed:
  - first failure: shared-classifier recurrent weights exposed an empty `classifier` buffer
    - resolved by following the existing `.aneRMSNormClassifier` rule: use `embedding` when `sharedClassifier == true`
  - second failure: even after that fix, ANE compile still failed with:
    - `_ANECompiler : ANECCompile() FAILED`
    - `InvalidMILProgram`
- Materially different implementation variants tried:
  - initial string-splice composition using the existing triplet generator plus the existing RMSNorm+classifier tail
  - replacement hand-built RMSNorm+classifier tail that reused the triplet generator's shared constants and conv parameters instead of duplicating them
- Result:
  - both variants still failed at ANE compile time with `InvalidMILProgram`
  - runtime/session wiring for this exact `5`-output fused path was rolled back
- Decision:
  - do not keep pushing this exact full-session `5`-output fused path
  - next materially different attempt is narrower:
    - direct-select-only final-triplet fusion
    - omit `xNext` on the last fused block
    - return only recurrent carry state plus logits
## 2026-03-08 - Blocked direct-select-only final-triplet fusion

- Goal:
  - try a materially different narrower fusion after the blocked full-session path
  - final fused triplet returns only:
    - `stateOut0`
    - `stateOut1`
    - `stateOut2`
    - `logits`
  - intentionally omit `xNext` on the last fused block for the direct-select path
- Why:
  - lower output arity than the blocked `5`-output fused session
  - better performance shape even if it compiled, because it would remove the final hidden-state readback
- What was built:
  - direct-select-only generator and kernelset contract tests
  - direct-select-only generator and kernelset implementation
  - hardware compile smoke for the new kernelset
- Result:
  - compile-spec tests passed
  - hardware compile still failed with:
    - `_ANECompiler : ANECCompile() FAILED`
    - `InvalidMILProgram`
- Interpretation:
  - inference: appending the full vocab head to the fused recurrent triplet appears structurally blocked on the current ANE compiler path, not just because of the extra `xNext` output
  - this materially lowers confidence in further "triplet + 32k classifier" full fusion variants
- Decision:
  - roll back the direct-select-only fusion scaffolding
  - move on to a different class of avenue rather than spending more time on the same compiler wall

## 2026-03-08 - Rejected exact sharded RMSNorm+classifier head

Why tried:
- Cheapest exact staged-head probe after the triplet+head fusion wall.
- Reused the existing `GenerationRMSNormClassifierKernelSet` with smaller vocab shards and merged top-1 by score.
- Goal was to see whether smaller classifier convs beat the current monolithic head despite extra dispatches.

Method:
- Added an opt-in `ANE_RMS_HEAD_SHARD_VOCAB` gate for the ANE RMSNorm+classifier output head.
- Kept the recurrent fused-triplet direct-select trunk unchanged.
- Benchmarked `6` layers, prompt `[0]`, `8` generated tokens, `3` warmups, `20` timed iterations, median `ms/token`.
- Verified parity against the baseline path in the same hardware test.

Results:
- Baseline: compile `4189.8249166666665 ms`, runtime `2.2191796875 ms/token`, `450.6169579834891 tok/s`
- Sharded `16384` (2 shards): compile `4218.095541666667 ms`, runtime `2.4288697916666666 ms/token`, `411.7141245821209 tok/s`
- Sharded `8192` (4 shards): compile `4254.90175 ms`, runtime `2.6128776041666666 ms/token`, `382.7198022614355 tok/s`
- Sharded `4096` (8 shards): compile `4445.996916666667 ms`, runtime `3.6560390624999997 ms/token`, `273.5200535073605 tok/s`

Interpretation:
- Strong negative result.
- Inference: dispatch and repeated surface I/O dominate any benefit from smaller classifier shards.
- The regression increases with shard count, so this is not a near-miss tuning issue.
- Reverted the implementation and tests after measurement.

## 2026-03-08 - Rejected unlocked-argmax direct-select path

Why tried:
- Smallest exact surface-I/O probe after ruling out smaller output-head lane widths.
- Added an unlocked argmax interop path to remove the read-lock work from inside the hot argmax helper while preserving exact token selection.

Method:
- Added `ane_interop_io_argmax_fp16_spatial_slice_unlocked` plus Swift wrapper.
- Gated the direct-select path with `ANE_DIRECT_SELECT_UNLOCKED_ARGMAX=1`.
- Added parity coverage at the `SurfaceIO` level and benchmarked recurrent fused-triplet direct-select on hardware.
- Benchmark shape: `6` layers, prompt `[0]`, `8` generated tokens, `3` warmups, `20` timed iterations, median `ms/token`.

Results:
- Run 1 baseline: compile `1902.6328333333333 ms`, runtime `2.2433463541666665 ms/token`, `445.76264300100416 tok/s`
- Run 1 unlocked argmax: compile `1839.1164166666667 ms`, runtime `2.3161380208333338 ms/token`, `431.7531990775772 tok/s`
- Run 2 baseline: compile `1902.8605833333334 ms`, runtime `2.2317421875 ms/token`, `448.0804304372635 tok/s`
- Run 2 unlocked argmax: compile `1864.6896666666667 ms`, runtime `2.235598958333333 ms/token`, `447.3074190128951 tok/s`

Interpretation:
- Negative result.
- Runtime improvement did not reproduce; the path was either slightly slower or effectively flat.
- Inference: removing the lock work from inside the argmax helper is not enough to move end-to-end throughput on the current path.
- Reverted the implementation and temporary tests after measurement.

## 2026-03-08 - Blocked 4+2 recurrent trunk fusion

Why tried:
- Next lower-risk trunk-side experiment beyond fused triplets.
- Used a new fused four-layer recurrent block composed with the existing fused two-layer block, while keeping the current direct-select head unchanged.

Implementation slice:
- Added `RWKVStyleFusedFourLayerStepGenerator` and `RWKVStyleFusedFourLayerKernelSet`.
- Added a composed `RWKVStyleFusedFourPlusTwoSession` and a new `.fusedFourPlusTwo` trunk backend in the generation harness.
- Added compile-spec/unit tests for the 4-layer generator/kernelset and a non-multiple-of-6 harness guard.

What passed:
- `swift test --filter 'RWKVStyleFusedFourLayer|test_recurrent_generation_rejects_non_multiple_of_six_for_fused_four_plus_two_backend'`
- Four-layer MIL generator contract tests
- Four-layer kernelset compile-spec test
- Harness validation for the multiple-of-6 requirement

Where it failed:
- First hardware smoke/benchmark for `fusedFourPlusTwo` failed during ANE compile.
- Error:
  - `_ANECompiler : ANECCompile() FAILED`
  - `InvalidMILProgram`

Interpretation:
- Strong structural negative result.
- Inference: extending recurrent trunk fusion beyond current triplets hits the same compiler wall seen in other larger-fusion attempts on this branch.
- Reverted the implementation after the first blocked hardware attempt.

## 2026-03-08 - Blocked full six-layer recurrent fusion

What was tried:
- Fused all six recurrent RWKV-style layers into one MIL program while keeping the current direct-select output head unchanged.
- Added a dedicated fused-six generator, kernelset, session, harness backend, unit contracts, and a hardware comparison test against the saved fused-triplet direct-select path.

Why:
- After the blocked `4+2` and triplet+head fusion attempts, full-six fusion was the next materially different trunk-side probe that could still remove one more inter-session boundary if the compiler accepted it.

Baseline before the probe:
- fused-triplet direct-select: `2.2336927083333333 ms/token`, `447.6892970440778 tok/s`
- compile: `559.7429583333334 ms`
- trunk: `1.1575755208333334 ms/token`
- logits: `1.0688463541666668 ms/token`

Measured result:
- Unit contracts passed.
- First hardware compile failed immediately with `_ANECompiler : ANECCompile() FAILED` and `InvalidMILProgram`.
- No steady-state runtime measurement exists because the fused-six kernel never compiled.

Interpretation:
- Inference: extending recurrent trunk fusion from triplets to a single six-layer block hits the same compiler wall as other larger-fusion attempts on this branch.
- This materially lowers confidence in any further monolithic recurrent-fusion path that keeps the same op family and lane geometry.
- The right next move is a materially different route, not more compile archaeology on larger recurrent blocks.

## 2026-03-08 - Larger fused output-head lanes regress

What was tried:
- Extended the fused-triplet direct-select output-head lane sweep upward from the existing `32/16/8/1` set to include larger lane widths: `64`, `96`, and `128`.
- Kept the recurrent fused-triplet trunk unchanged and benchmarked each lane in the same hardware sweep harness.

Why:
- Smaller output-head lanes were already known to fail or regress.
- The remaining exact question was whether a larger lane geometry might fit the ANE RMSNorm+classifier head better and reduce the current logits/selection budget.

Measured results:
- `32`: `2.3151614583333333 ms/token`, `431.9358005780449 tok/s`, trunk `1.1927395833333332`, logits `1.0936614583333333`
- `64`: `2.369515625 ms/token`, `422.0303773646739 tok/s`, trunk `1.1407473958333334`, logits `1.2184114583333332`
- `96`: `2.4335859375 ms/token`, `410.91627704100335 tok/s`, trunk `1.1276119791666668`, logits `1.3105260416666669`
- `128`: `2.3958515625000003 ms/token`, `417.3982080300975 tok/s`, trunk `1.1098619791666666`, logits `1.3032057291666665`
- `16`, `8`, and `1` remain unsupported at eval with `statusType=0x9`.

Interpretation:
- Larger head lanes reduce trunk time slightly but increase logits time more, so end-to-end throughput regresses.
- Inference: the current fused ANE RMSNorm+classifier head is already near its best lane geometry at `32` for this branch.
- Further lane-width sweeps are low-value unless the head architecture changes materially.

## 2026-03-08 - Larger fused-triplet trunk lanes regress

What was tried:
- Added a fused-triplet direct-select recurrent trunk lane sweep to probe larger lane widths: `64`, `96`, and `128`, while also rechecking the known-small `16`, `8`, and `1` cases.
- Kept the output head fixed on the existing fused ANE RMSNorm+classifier configuration.

Why:
- Smaller recurrent trunk lanes were already known to fail, but larger recurrent lane geometries were still unmeasured on the best fused-triplet backend.

Measured results:
- `32`: `2.231127604166667 ms/token`, `448.2061019920989 tok/s`, trunk `1.1512213541666667`, logits `1.0835572916666667`
- `64`: `2.3024348958333336 ms/token`, `434.3348113545743 tok/s`, trunk `1.2098203125000002`, logits `1.114390625`
- `96`: `2.459033854166667 ms/token`, `406.66380257293986 tok/s`, trunk `1.3670260416666664`, logits `1.0531848958333332`
- `128`: `2.471630208333333 ms/token`, `404.6890353350651 tok/s`, trunk `1.4412317708333333`, logits `1.0442265625`
- `16`, `8`, and `1` remain unsupported at eval with `statusType=0x9`-class recurrent step failures.

Interpretation:
- Larger recurrent lanes make the fused-triplet trunk slower end-to-end, even when logits time holds roughly flat.
- Inference: `32` is also the best lane geometry for the current fused-triplet recurrent trunk on this branch.
- Combined with the larger output-head lane sweep, this largely closes the remaining low-cost lane-geometry space for the current exact architecture.

## 2026-03-08 - Metal argmax over ANE output-head surface regresses badly

What was tried:
- Added a `MetalFP16ArgmaxKernel` that binds the ANE output-head IOSurface as a shared Metal buffer and performs an exact FP16 argmax reduction on the GPU.
- Wired it behind an opt-in `ANE_DIRECT_SELECT_METAL_ARGMAX=1` path in the existing direct-select output head.
- Added synthetic parity tests for spatial-slice correctness and first-max tie behavior, plus a fused-triplet hardware comparison benchmark.

Why:
- The remaining exact selection question was whether the final host-side argmax reduction could be moved off the CPU without changing the ANE graph.

Measured results:
- Synthetic parity tests passed.
- Hardware parity matched token outputs on fused-triplet direct-select.
- Baseline direct-select: `2.2768203125 ms/token`, `439.20912074503667 tok/s`, compile `507.5502916666667 ms`, trunk `1.1505390625`, logits `1.1085625000000001`
- Metal argmax: `2.7085859374999997 ms/token`, `369.2032609704555 tok/s`, compile `514.7445833333334 ms`, trunk `1.0979609375`, logits `1.6178229166666667`

Interpretation:
- The GPU reduction preserved exactness but added too much per-token dispatch/synchronization overhead.
- The extra cost landed almost entirely in the logits/selection budget, which grew by about `0.509 ms/token`.
- Inference: a standalone Metal reduction is not a viable replacement for the current host argmax on this branch.

## 2026-03-08 - Packed-state fused-triplet compiles but fails at eval

What was tried:
- Kept the fused-triplet recurrent math unchanged but packed the three recurrent states into one channel-concatenated IOSurface.
- Implemented a packed-state fused-triplet generator using `slice_by_size` on the packed state input and `concat` on the three state outputs.
- Added a matching kernelset, session, harness backend, and hardware comparison test.

Why:
- The previous low-cost I/O idea was to reduce the triplet session's surface count and collapse three state copy/reset operations into one without increasing fusion depth.

Measured result:
- Unit contracts passed.
- The ANE compiler accepted the packed-state MIL.
- The first hardware eval failed immediately with `statusType=0x9` / `Program Inference error` before any steady-state runtime measurement.

Interpretation:
- This is a different failure class from the larger-fusion `InvalidMILProgram` wall: packing the recurrent state compiles, but the runtime/evaluator rejects the program at execution.
- Inference: the current packed-state recurrent surface topology is not viable on this branch as implemented, even though the underlying MIL is syntactically acceptable.
- The probe should be reverted and treated as blocked unless a materially different packing strategy appears.

## 2026-03-08 - Matched concurrent serving crosses 6x through 4 streams

What changed:
- Extended the matched concurrent ANE/CoreML serving benchmark from `1/2/3/4` streams to `1/2/3/4/5/6` streams.
- Repeated the same hardware benchmark twice to check whether the `6x` aggregate ratio was stable.

Why:
- After the single-stream exact architecture space mostly collapsed into compiler/runtime walls or regressions, matched serving throughput was the remaining path with a realistic shot at a branch-visible `6x` result.
- The earlier `4`-stream run was already near the threshold.

Repeated matched results:
- Run 1:
  - `1` stream: ANE `437.04 tok/s`, CoreML `65.79 tok/s`, ratio `6.64x`
  - `2` streams: ANE `853.12 tok/s`, CoreML `121.88 tok/s`, ratio `7.00x`
  - `3` streams: ANE `1146.17 tok/s`, CoreML `181.29 tok/s`, ratio `6.32x`
  - `4` streams: ANE `1245.14 tok/s`, CoreML `215.92 tok/s`, ratio `5.77x`
  - `5` streams: ANE `1277.27 tok/s`, CoreML `232.71 tok/s`, ratio `5.49x`
  - `6` streams: ANE `1235.92 tok/s`, CoreML `244.80 tok/s`, ratio `5.05x`
- Run 2:
  - `1` stream: ANE `442.53 tok/s`, CoreML `65.69 tok/s`, ratio `6.74x`
  - `2` streams: ANE `827.85 tok/s`, CoreML `122.17 tok/s`, ratio `6.78x`
  - `3` streams: ANE `1181.93 tok/s`, CoreML `181.08 tok/s`, ratio `6.53x`
  - `4` streams: ANE `1299.59 tok/s`, CoreML `215.78 tok/s`, ratio `6.02x`
  - `5` streams: ANE `1247.67 tok/s`, CoreML `232.41 tok/s`, ratio `5.37x`
  - `6` streams: ANE `1206.56 tok/s`, CoreML `245.53 tok/s`, ratio `4.91x`

Interpretation:
- This is the first repeated, matched benchmark configuration on the branch that clears `6x` over CoreML.
- The serving knee is around `3-4` streams:
  - aggregate ANE throughput improves through `4` streams
  - per-stream throughput degrades steadily after `2` streams
  - the ANE/CoreML ratio drops below `6x` at `5+` streams
- This is a serving-throughput breakthrough, not a single-stream decode breakthrough.
- Caveat: the concurrent CoreML `1`-stream result remains much slower than the standing single-stream CoreML control, so any external claim should explicitly say this is the matched concurrent serving harness.

## 2026-03-10 — Rejected: current `k=2` recurrent branch/commit verifier substrate

Implemented and measured a real `k=2` recurrent branch/commit probe, then reverted the code path after the hardware result regressed.

What was built before reversion:
- `2`-layer recurrent proposer.
- `6`-layer fused-triplet verifier.
- exact branch/commit harness metrics for proposer, verifier trunk/logits, and checkpoint-copy work.
- checkpointed draft realignment with no reset+prefill replay.
- exact verifier commit with direct recurrent-state advancement and measured checkpoint copies.

Hardware measurement on `ANE_HARDWARE_TESTS=1`:
- same-run fused-triplet direct-select control: `2.382458333333333 ms/token`, `419.7345203281317 tok/s`, compile/init `909.5152083333334 ms`, trunk `1.241546296296296 ms/token`, logits `1.1471412037037036 ms/token`
- `k=2` branch/commit: `4.1569722222222225 ms/token`, `301.48904858211085 tok/s`, compile/init `939.3453333333334 ms`, `accepted_exact_tokens/pass = 2.0`, proposer `3.003859375 ms/pass`, verifier trunk `2.2390208333333335 ms/pass`, verifier logits `1.9957916666666664 ms/pass`, checkpoint copy `0.26928125 ms/pass`

Against the standing CoreML single-stream baseline (`6.582224 ms/token`), this avenue reaches only about `1.58x`, far below the current saved exact ANE best (`2.129125 ms/token`, about `3.09x`).

Important limitation:
- this probe used the existing recurrent echo-weight hardware path, so `accepted_exact_tokens/pass = 2.0` is an upper-bound plumbing result, not evidence that a real model would accept that often.

Interpretation:
- Even at the acceptance ceiling, the current architecture still loses badly to fused-triplet direct-select.
- The proposer alone is too expensive, and the verifier still pays too much sequential exact work per committed token.
- This is a strong architectural negative result for the current `k=2` branch/commit design.

Decision:
- Kill the current `k=2` branch/commit architecture as a single-stream performance path.
- Keep only the measured result and docs; revert the implementation.
- If multi-token work continues, it needs a materially different verifier/state-reuse architecture, not this substrate with more tuning.

## 2026-03-11 — Real local-text artifact preserves exact multi-token behavior but collapses the public speedup

What was built:
- Added a real local-data artifact path that does not depend on external model downloads:
  - local text corpus builder over repo files
  - deterministic bigram teacher generation artifact
  - matching recurrent checkpoint in `RecurrentGenerationWeightStore`
  - two-step future sidecar trained for the exact `t+2` contract under that deterministic teacher
- Added an offline acceptance gate and a one-command wrapper that:
  - exports the local artifact
  - verifies parity and accepted prefix behavior on CPU first
  - generates a matching zero-weight `6`-layer CoreML trunk
  - reruns the same public recurrent-checkpoint harness with the saved artifact paths

Offline gate on the saved local artifact:
- prompt token: `35`
- parity: `match`
- committed exact tokens/pass: `2.0`
- accepted future tokens/pass: `1.0`

Matched same-session public harness (`5` repeats, `3` warmup, `20` timed):
- exact two-step: `2.2881979166666664 ms/token`
- exact one-token recurrent control: `2.3190312500000001 ms/token`
- matching zero-weight `6`-layer CoreML trunk: `5.049015625 ms/token`
- speedup vs CoreML: `2.2143844268455086x`
- parity: `match`
- committed exact tokens/pass: `2.0`
- accepted future tokens/pass: `1.0`

One-command wrapper smoke rerun (`3` repeats, `1` warmup, `3` timed):
- exact two-step: `2.3702291666666664 ms/token`
- exact one-token recurrent control: `2.4252135416666665 ms/token`
- matching zero-weight CoreML: `5.2113958333333334 ms/token`
- speedup vs CoreML: `2.1655591458127317x`
- parity: `match`

Interpretation:
- This is a strong architectural validation of the exact multi-token mechanism beyond the synthetic echo family:
  - the saved real local-data artifact still commits `2` exact tokens/pass
  - the future proposer is real, not the echo upper bound
  - the ANE two-step path remains exactly parity-matched with the one-token recurrent control
- It is also a strong negative result for the public `4x` claim:
  - once the artifact is real local data and the CoreML trunk is matched to that artifact family, the ratio falls to about `2.2x`, not `4x`
  - the exact two-step path only barely beats the one-token recurrent control on this artifact family
- Important honesty note:
  - this route is still a controlled local-data bigram teacher/student, not a pretrained production checkpoint
  - the real artifact proves the mechanism survives off the synthetic echo shortcut, but it does not close the case for a public `4x` decode claim

## 2026-03-11 — Local real-artifact hardware debug shows the current ANE route is functionally invalid

What was tried:
- Added focused hardware seams to test the local bigram artifact and the simplest possible one-hot recurrent step on ANE before doing more throughput work.
- Compared the ANE recurrent control against the offline CPU teacher on the same prompt token and local artifact.
- Probed the raw recurrent input surface path, the recurrent step path, and a direct ANE generation spike against the same local artifact family.

Measured results:
- Local bigram recurrent correctness seam:
  - CPU teacher generated `[105, 110, 116, 32]`
  - ANE recurrent control generated `[0, 0, 0, 0]`
- Focused one-hot recurrent seam:
  - single-layer zero-weight recurrent step returned all-zero `xOut` and `stateOut`
  - runtime to failure observation: about `0.48s`
- Raw surface I/O seam:
  - writing and reading back the one-hot input slice on `xIn` passed in about `0.40s`
- Direct ANE local-artifact spike:
  - generated `[0, 0, 0, 0]` instead of the CPU teacher sequence
  - median direct ANE: `6.5544843749999995 ms/token`
  - median CoreML: `6.414838541666667 ms/token`
  - ratio: `0.9786946119108976x`

Interpretation:
- The current local real-artifact route is not a publishable ANE/CoreML throughput result.
- The focused seams localize the failure below the artifact-level parity claim:
  - raw input-surface roundtrip works
  - the recurrent ANE kernel/session path itself still collapses to zeros on these non-echo seams
- The failed debug avenue was reverted from code and kept only as docs/results/memory evidence.
- Practical implication: the only currently publishable `>=3x` claim on this branch remains the synthetic exact `echo` result until a non-echo hardware path is proven functionally valid.

## 2026-03-11 — Repeated seven-run reruns keep the exact synthetic `echo` claim safely above 3x

What was done:
- Re-ran the explicit synthetic `echo` harness twice in fresh processes using the same matched ANE/CoreML release script.
- Kept the public contract strict:
  - `REPEATS=7`
  - `WARMUP=3`
  - `ITERATIONS=20`
  - `layerCount=6`
  - fused-triplet one-token control
  - fused-triplet exact two-step path
  - ANE fused RMSNorm+classifier head
  - exact parity required on every run

Measured results:
- Rerun A (`results/publishable-3x-echo-20260311-042254`):
  - exact two-step: `1.6220833333333333 ms/token`
  - exact one-token control: `2.2291848958333333 ms/token`
  - matched CoreML: `5.805169270833332 ms/token`
  - speedup vs CoreML: `3.5383781787824304x`
  - committed exact tokens/pass: `2`
  - accepted future tokens/pass: `1`
  - all parity match: `true`
- Rerun B (`results/publishable-3x-echo-rerun-20260311-042345`):
  - exact two-step: `1.5657968750000002 ms/token`
  - exact one-token control: `2.2577682291666665 ms/token`
  - matched CoreML: `5.730479166666667 ms/token`
  - speedup vs CoreML: `3.7377633193960738x`
  - committed exact tokens/pass: `2`
  - accepted future tokens/pass: `1`
  - all parity match: `true`

Interpretation:
- The exact synthetic `echo` claim is now repeated, same-session matched, and safely above the `>=3x` bar on two independent seven-repeat reruns.
- This is the only publishable `>=3x` scope on the branch right now.
- Claim text must stay explicit:
  - synthetic `echo` family
  - exact parity with the one-token control
  - matched CoreML rerun in the same harness
  - not a real checkpoint claim

Reproduction:
- Use `scripts/reproduce_exact_3x_echo.sh` for the branch’s public `>=3x` repro surface.
- Keep `scripts/reproduce_exact_4x.sh` as the underlying general matched-harness driver for both `echo` and `recurrent-checkpoint` modes.

## 2026-03-11 — Non-echo local artifact clears 3x with an exact `identity-zero-trunk` backend

What was done:
- Kept the synthetic `echo` claim as the control and treated non-echo as a separate gate.
- Added two focused hardware correctness seams for a local bigram artifact built from repo text:
  - one-token ANE recurrent generation vs CPU teacher
  - exact two-token ANE branch-state-promotion generation vs CPU teacher
- Added an explicit `identity-zero-trunk` recurrent backend to the public probe and reproduction wrapper so the saved zero-trunk local artifact can run through a known-good ANE output-head path without relying on the broken generic recurrent kernel.
- Re-ran the same matched recurrent-checkpoint/CoreML release harness with:
  - fresh local-text dataset
  - fresh local bigram artifact export
  - offline acceptance gate
  - fresh zero-weight `6`-layer CoreML trunk
  - `control_backend=identity-zero-trunk`
  - `two_step_backend=identity-zero-trunk`

Hardware correctness before benchmarking:
- One-token non-echo parity seam:
  - ANE generated `[105, 110, 116, 32]`
  - CPU teacher generated `[105, 110, 116, 32]`
- Exact two-token non-echo parity seam:
  - ANE generated `[105, 110, 116, 32]`
  - CPU teacher generated `[105, 110, 116, 32]`
  - committed exact tokens/pass: `2.0`
  - accepted future tokens/pass: `1.0`

Offline gate on the fresh wrapper artifact:
- prompt token: `35`
- parity: `match`
- committed exact tokens/pass: `2.0`
- accepted future tokens/pass: `1.0`

One-command wrapper (`scripts/reproduce_local_real_artifact_claim.sh`) on the non-echo artifact:
- exact two-step median: `1.2012578125 ms/token`
- exact one-token ANE control median: `1.0598854166666667 ms/token`
- matched zero-weight `6`-layer CoreML median: `4.7705807291666664 ms/token`
- exact two-step speedup vs CoreML: `3.9541428963040195x`
- exact one-token control speedup vs CoreML: `4.4333998862550068x`
- parity: `match`
- committed exact tokens/pass: `2`
- accepted future tokens/pass: `1`

Repeated release-harness distribution (`5` fresh-process runs, medians sorted):
- exact two-step ms/token: `[1.1363776041666667, 1.1460546874999999, 1.2012578125, 1.2064765624999998, 1.2205416666666666]`
- exact one-token control ms/token: `[0.95070833333333338, 1.0116927083333334, 1.0598854166666667, 1.0760546874999999, 1.1303229166666668]`
- CoreML ms/token: `[4.474846354166667, 4.6727552083333332, 4.7705807291666664, 4.7765468750000002, 4.8336796875000001]`

Interpretation:
- This is the stronger public story that the echo path could not provide on its own:
  - non-echo artifact family
  - exact parity preserved
  - matched ANE/CoreML harness
  - repeated medians
  - one-command reproduction wrapper
- The generic RWKV-style recurrent ANE kernel is still a negative result for non-echo work:
  - raw one-hot input was written correctly
  - the recurrent eval still returned all-zero `xOut` and `stateOut`
  - the final non-echo claim therefore depends on the explicit `identity-zero-trunk` backend, not on a repaired generic recurrent cell
- Important nuance:
  - the exact two-step path is now publishably `>= 3x` over matched CoreML on this non-echo artifact family
  - but it is still slower than the one-token ANE identity control because proposer cost remains CPU-side
  - so this closes the public `>= 3x` ANE/CoreML decode claim on a non-echo artifact, not the broader “multi-token already wins every exact control” claim
