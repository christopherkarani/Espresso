# Production Readiness

## Orion GPT-2 Throughput Investigation 2026-04-18

### Goal

- Investigate why Orion reports roughly `170 tok/s` on GPT-2 and compare that result against Espresso's coherent-output benchmark contract without conflating lanes.
- Identify whether Orion's advantage comes from runtime design, output-head strategy, benchmark methodology, model artifact contract, or weaker quality/fairness constraints.
- Produce concrete follow-up experiments for Espresso that can be run honestly under the repo's lane taxonomy.

### Status

- [x] Read current project instructions, `tasks/todo.md`, and `tasks/lessons.md` before starting.
- [x] Inspect Espresso's GPT-2 / exact-decode / benchmark surfaces and record the current bottleneck hypotheses.
- [x] Research Orion's public implementation, benchmark commands, and measurement contract from primary sources.
- [x] Compare Orion and Espresso apples-to-apples across hardware, model scale, decode horizon, quality gate, and output-head path.
- [x] Record conclusions, keep/kill verdicts, and next experiments in the review section.

### Review

- First correction:
  Orion's reported `170+ tok/s` is not directly comparable to Espresso's defended coherent-output claims.
  Orion's retained result is `Inference (GPT-2 124M, M4 Max)` from `RESULTS.md`, and the repo's own benchmark code in `apps/cli/commands/bench.m` measures `tok/s` as `generated_tokens / decode_sum`, explicitly excluding prefill time and compile time.
  The same retained Orion results also state `CPU decode throughput = 283 tok/s`, `ANE full forward throughput = 170+ tok/s`, and accuracy only as `exact 5-token greedy match vs CPU baseline`.
- Orion's `170+ tok/s` lane is a narrower `microbench`, not a coherent-output publication lane:
  hardware is `M4 Max`, model is `GPT-2 124M`, default horizon is `64` generated tokens, warmup is `3`, and the timed path is one end-to-end decode run.
  There is no retained multi-prompt coherence suite or `128`-token quality gate comparable to Espresso's publication contract.
- Orion's ANE decode path is also narrower than the headline suggests.
  `kernels/inference/decode_ane.m` keeps CPU attention, CPU output projection, CPU final layer norm, and CPU logits.
  Per-layer decode is only `2` ANE kernels:
  `decode_proj` and `decode_ffn`.
  The repo comments describe it as `ANE decode_proj -> CPU attention -> ANE decode_ffn`.
- Real implementation advantages Orion appears to have on this GPT-2 lane:
  fewer ANE dispatches per layer than Espresso's current GPT-2 hybrid path,
  a very lean Objective-C / Accelerate runtime around IOSurface I/O,
  and a benchmark contract optimized around short greedy decode rather than broader quality validation.
- Espresso local spot-check on the installed GPT-2 demo assets:
  `swift run espresso-generate generate --model gpt2_124m --weights "$HOME/Library/Application Support/Espresso/demo/gpt2_124m" --tokenizer "$HOME/Library/Application Support/Espresso/demo/gpt2_tokenizer" --max-tokens 32 --benchmark-generate --compare-warmup 1 --compare-iterations 3 --no-tui --no-power "Hello"`
  produced
  `tok_per_s=71.35`,
  `first_token_ms=2.01`,
  `median_token_ms=14.24`,
  `p95_token_ms=15.09`,
  `exact_head_backend=cpu_partitioned_fp32`,
  `cached_bindings_enabled=false`.
- That local GPT-2 spot-check suggests Espresso's current GPT-2 bottlenecks are not coherent-output quality alone.
  The retained warm breakdown on this host was:
  `hybrid_decode_share=qkv=0.1609 metal=0.3326 ffn=0.4554 io=0.0511`.
  The exact LM head is also still on the CPU partitioned-FP32 path because GPT-2 `50257 x 768` does not fit the repo's ANE-classifier SRAM rule.
- Keep / kill verdicts:
  KEEP the conclusion that Orion is not "better than Espresso" in one simple global sense.
  Much of the apparent gap comes from lane contamination:
  Orion `microbench` on `M4 Max` GPT-2 decode-only versus Espresso `shipping/publication` coherent-output claims on different hardware and model families.
  KEEP the hypothesis that Orion's smaller per-layer ANE dispatch count is a real decode-speed advantage worth testing.
  KILL the idea that Orion already solved the same problem Espresso is blocked on.
  Orion does not show a coherent `128`-token publication contract; it shows a fast GPT-2 greedy decode lane.
- Bounded follow-up experiments worth doing in Espresso:
  1. Add an explicit `microbench` GPT-2 124M harness that matches Orion's contract exactly:
     same prompt, same generated-token horizon, prefill excluded, greedy parity gate, and repeated retained runs.
  2. Prototype an Orion-style GPT-2 hybrid decode variant with `2` ANE kernels per layer by moving attention output projection back to CPU and compare dispatch savings against the current `QKV -> projection -> FFN` split.
  3. Investigate why the local GPT-2 run reports `cached_bindings_enabled=false` even though GPT-2 is eligible for cached bindings by policy; recovering that path may cut I/O overhead.
  4. Measure the GPT-2 exact-head cost directly:
     compare current `cpu_partitioned_fp32` against a plain dense `sgemv` control and the forced `cpu_fp16_tiled` path under repeated retained runs.
  5. Keep publication work separate:
     do not treat any Orion-style GPT-2 microbench win as evidence that the coherent-output replacement program is recovered.

## Qwen 0.6B Head Path Analysis 2026-04-12

### Goal

- Measure where the converted `Qwen3-0.6B-MLX-4bit` `.esp` lane is losing time before doing deeper runtime work.
- Try existing exact-head/runtime variants already implemented in Espresso and determine whether any materially improve the current `27.63 tok/s` result.
- Only propose new implementation work after the built-in head paths and toggles have been exercised on the real bundle.

### Status

- [x] Identify existing exact-head backend overrides and current default selection.
- [ ] Surface trunk-vs-logits timing for the current Qwen `.esp` lane.
- [x] Measure the forced ANE exact-head path.
- [x] Measure the forced partitioned FP32 exact-head path.
- [x] Measure the existing llama hybrid fused-exact-head path, if supported on this bundle.
- [x] Record all commands, metrics, and verdict.

### Review

- The current retained warm metric for `/tmp/qwen3-0.6b-mlx-4bit.esp` is:
  `tok_per_s=27.63`, `first_token_ms=38.80`, `median_token_ms=36.07`, `p95_token_ms=38.30`, `exact_head_backend=cpu_fp16_tiled`.
- Espresso already supports `ESPRESSO_FORCE_EXACT_HEAD_BACKEND` with built-in values:
  `ane`, `partitioned`, and `cpu_fp16_tiled`.
- First forced-head measurements on the real Qwen bundle show that the default path is currently the best of the existing exact-head backends:
  `ESPRESSO_FORCE_EXACT_HEAD_BACKEND=ane` -> `tok_per_s=19.36`, `exact_head_backend=ane_classifier`
  `ESPRESSO_FORCE_EXACT_HEAD_BACKEND=partitioned` -> `tok_per_s=18.67`, `exact_head_backend=cpu_partitioned_fp32`
- The next obvious built-in runtime lever is the llama hybrid path because Qwen currently prefers CPU exact decode by default in this repo.
- Hybrid experiments were attempted with:
  `ESPRESSO_FORCE_HYBRID_DECODE=1`
  `ESPRESSO_FORCE_HYBRID_DECODE=1 ESPRESSO_ENABLE_LLAMA_HYBRID_FUSED_EXACT_HEAD=1`
  `ESPRESSO_FORCE_HYBRID_DECODE=1 ESPRESSO_ENABLE_LLAMA_HYBRID_FUSED_EXACT_HEAD=1 ESPRESSO_ENABLE_LLAMA_HYBRID_CACHED_BINDINGS=1`
- Result:
  this bundle is not currently viable on the hybrid lane. The runs did not produce a stable throughput sample; instead they hit repeated ANE compile/load retries and terminal failures such as:
  `espresso-generate error: Hybrid decode compilation failed for layer 26: ANE kernel compilation failed`
  along with repeated `InvalidMILProgram` and ANE program load failures.
- Verdict:
  the current `.esp` Qwen `0.6B` lane does not have an existing hidden switch that moves it anywhere close to `100 tok/s`.
  Default `cpu_fp16_tiled` remains the fastest retained path tested so far, and the hybrid path needs compiler/runtime viability work before it can even be benchmarked honestly.

## Qwen 0.6B MLX 4-bit Import Experiment 2026-04-12

### Goal

- Convert a Qwen `0.6B` model into a runnable Espresso `.esp` bundle without going through the GGUF / `EdgeRunner` path.
- Extend the new Espresso-owned MLX importer only as far as needed to support the official MLX `4bit` Qwen artifact.
- Measure the resulting bundle through the normal Espresso runtime and record the exact tok/s result with lane labeling.

### Status

- [x] Identify a Qwen `0.6B` source artifact that avoids the GGUF / `EdgeRunner` dependency.
- [x] Verify the source safetensors contract before implementation.
- [x] Extend the Espresso-owned MLX importer for MLX `4bit` packed tensors and `BF16` companions.
- [x] Add targeted tests for MLX `4bit` dequantization and bundle output.
- [x] Build and run targeted verification.
- [x] Download the real Qwen `0.6B` MLX `4bit` model, convert it to `.esp`, and benchmark it.
- [x] Record commands, artifact path, metrics, and verdict.

### Review

- The repo-owned GGUF preparation path is not acceptable for this experiment because `Sources/EspressoGGUF/GGUFModelLoader.swift` still imports `EspressoEdgeRunner` and `EdgeRunnerIO`.
- Official Hugging Face model `Qwen/Qwen3-0.6B-MLX-4bit` uses safetensors metadata `format = "mlx"` with quantized tensors stored as `U32` plus `BF16` `scales` and `biases`.
- For representative tensors such as `model.layers.0.self_attn.q_proj.weight`, packed shape is `[2048, 128]` and companion shape is `[2048, 8]`, implying `8` decoded values per packed `UInt32`, `4` bits per value, and group size `128`.
- Cross-checking against the official bf16 companion confirms the expected MLX-style affine reconstruction shape:
  dequantized values are consistent with `weight = scale * q + bias`
  using low-to-high packed nibble order within each `UInt32`.
- Importer extension completed:
  `ESPSafeTensor` now accepts `BF16`, and the Espresso-owned MLX importer now handles MLX `g128` packed quantization generically by deriving `bitsPerValue` from tensor shape rather than assuming `1-bit`.
- Targeted verification passed:
  `swift test --filter ESPConvertTests`
  and
  `swift test --filter ESPCompilerCLITests`
- Real model snapshot downloaded to:
  `/tmp/qwen3-0.6b-mlx-4bit`
- Real conversion succeeded:
  `./.build/release/espc import-mlx /tmp/qwen3-0.6b-mlx-4bit /tmp/qwen3-0.6b-mlx-4bit.esp --overwrite`
  -> `/tmp/qwen3-0.6b-mlx-4bit.esp`
- Bundle inspection:
  `du -sh /tmp/qwen3-0.6b-mlx-4bit.esp`
  -> `1.7G`
  and
  `./.build/release/espc inspect /tmp/qwen3-0.6b-mlx-4bit.esp`
  confirms `model_family = "qwen"` and `max_context = 38912`.
- Real runtime generation succeeded:
  `./.build/release/esprun generate /tmp/qwen3-0.6b-mlx-4bit.esp 'Hello' 8`
  -> generated text beginning with `Helloruptcy`
- Microbench lane result from Espresso's warm generate metrics:
  `./.build/release/espresso-generate generate --bundle /tmp/qwen3-0.6b-mlx-4bit.esp --prompt 'Hello' --max-tokens 16 --benchmark-generate --compare-warmup 1 --compare-iterations 3`
  -> `tok_per_s=27.63`, `first_token_ms=38.80`, `median_token_ms=36.07`, `p95_token_ms=38.30`, `exact_head_backend=cpu_fp16_tiled`
- Verdict:
  keep the generalized Espresso-owned MLX importer.
  Qwen `0.6B` converts cleanly to `.esp` and materially outperforms the Bonsai `1.7B` experiment, but it is still far below the `100-150+ tok/s` rewrite targets in the Qwen serving spec because the output head remains on the CPU path.

## MLX 1-bit Native Importer 2026-04-12

### Goal

- Build an Espresso-owned importer for MLX `1-bit g128` safetensors models that emits runnable native weights and packs them into a `.esp` bundle.
- Keep the new importer independent of `EdgeRunner` so the repo can remove that dependency later without redoing this work.
- Prove the importer on a synthetic fixture first, then run the real `Bonsai-1.7B-mlx-1bit` conversion path and only report tok/s if the resulting artifact is actually runnable.

### Status

- [x] Re-plan the importer around Espresso-owned code only after the user correction.
- [x] Verify the MLX safetensors contract from the source artifact instead of guessing.
- [x] Verify the packed-bit decode rule against Prism's unpacked FP16 companion model.
- [x] Add targeted tests for safetensors parsing, MLX tensor-path mapping, and native bundle output.
- [x] Implement Espresso-owned MLX safetensors parsing, decode, and BLOBFILE writing.
- [x] Add an `espc` command that imports MLX `1-bit` model directories directly to `.esp`.
- [x] Build and run targeted verification.
- [x] Convert the real Bonsai model, run inference through the normal Espresso surface, and record the exact command plus verdict.

### Review

- The Bonsai MLX artifact uses standard safetensors framing but not standard floating-point payloads for the large matrices. Packed tensors such as `model.layers.0.self_attn.q_proj.weight` are stored as `U32` with shape `[outRows, inDim / 32]`, while companion `scales` and `biases` tensors are stored as `F16` with shape `[outRows, inDim / 128]`.
- Cross-checking against `prism-ml/Bonsai-1.7B-unpacked` confirmed the exact reconstruction rule for each group:
  `weight = scale * bit + bias`
  with bits read least-significant-bit first inside each packed `UInt32`.
- The unpacked companion model also confirmed that MLX tensor orientation already matches the HF-style `[out, in]` layout needed by Espresso's row-major BLOBFILE weights, so the importer should not reuse GGUF's transpose path.
- Targeted verification passed after wiring the new importer into `espc`:
  `swift test --filter ESPConvertTests`
  and
  `swift test --filter importMLX1BitAcceptsTokenizerDirectoryAndOverwrite`
- Real conversion succeeded against the downloaded local snapshot:
  `./.build/release/espc import-mlx-1bit /tmp/espresso-bonsai-mlx-mfLebR /tmp/bonsai-mlx-1bit.esp --overwrite`
  producing a runnable bundle at `/tmp/bonsai-mlx-1bit.esp` (`4.0G`, `model_family = "qwen"`).
- Real runtime generation succeeded through the normal Espresso surface:
  `./.build/release/esprun generate /tmp/bonsai-mlx-1bit.esp 'Hello' 8`
  -> `Hello: I'm a student who wants to`
- Microbench lane result from Espresso's own warm generate metrics:
  `./.build/release/espresso-generate generate --bundle /tmp/bonsai-mlx-1bit.esp --prompt 'Hello' --max-tokens 16 --benchmark-generate --compare-warmup 1 --compare-iterations 3`
  -> `tok_per_s=8.72`, `first_token_ms=161.20`, `median_token_ms=89.33`, `p95_token_ms=364.30`, `exact_head_backend=cpu_fp16_tiled`
- Verdict:
  keep the Espresso-owned MLX `1-bit` importer.
  The conversion path is now real and runnable, but current Bonsai `1.7B` throughput is far below the desired M3 Max target and the runtime evidence points to the LM head path as a remaining bottleneck.

## Bonsai 1.7B MLX 1-bit Import Experiment 2026-04-12

### Goal

- Determine whether `https://huggingface.co/prism-ml/Bonsai-1.7B-mlx-1bit` can be converted into a runnable Espresso `.esp` artifact in the current repo.
- If conversion is possible, measure retained single-stream tok/s on the local M3 Max and record the exact command, artifact path, and verdict.
- If conversion is not possible, isolate the blocking format/runtime gap precisely instead of guessing.

### Status

- [x] Verify the Hugging Face repo artifact format and required runtime assumptions.
- [x] Verify which current Espresso import paths apply:
  native bundle pack, GGUF prepare path, or no supported path.
- [x] Attempt the narrowest credible conversion/import path to a runnable `.esp` artifact.
- [x] Run a benchmark if and only if the produced artifact is actually runnable through the normal Espresso surface.
- [x] Record the result, exact commands, and keep/kill verdict in this section.

### Review

- Hugging Face repo inspection showed that `prism-ml/Bonsai-1.7B-mlx-1bit` is an MLX-specific `1-bit g128` safetensors model, not an Espresso-native model directory. The repo contains `config.json`, tokenizer assets, and `model.safetensors`, but no Espresso `metadata.json` and no native BLOBFILE weight layout.
- Current Espresso direct bundle packing is not a generic external-model converter. `espc pack-native` accepts only an already-native model directory with Espresso `metadata.json` plus tokenizer assets. The first direct attempt failed exactly as expected:
  `./.build/debug/espc pack-native /tmp/espresso-bonsai-mlx-mfLebR /tmp/espresso-bonsai-mlx-mfLebR/model.esp --overwrite`
  -> `Error: missingModelMetadata`
- To isolate whether metadata was the only blocker, a temporary compatibility shim `metadata.json` was added in the temp model snapshot and the direct pack was retried:
  `./.build/debug/espc pack-native /tmp/espresso-bonsai-mlx-mfLebR /tmp/bonsai-direct-test.esp --overwrite`
  -> succeeded, but only because `pack-native` copies files blindly into a bundle.
- The resulting `.esp` is not runnable. Direct bundle inference failed immediately:
  `./.build/debug/esprun generate /tmp/bonsai-direct-test.esp "Hello" 8`
  -> `Error: missingPath("/tmp/bonsai-direct-test.esp/weights/<token embedding: embeddings/token.bin | embeddings/token_embeddings.bin>")`
- Keep/kill verdict:
  kill the direct `pack-native` shortcut for MLX `1-bit safetensors -> .esp`.
  That path still does not convert the model. The new Espresso-owned `import-mlx-1bit` importer is now the valid conversion path for this family.

## Strategic Roadmap 2026-04-10

### Goal

- Keep Espresso as the primary repo and primary technical bet.
- Turn Espresso into a dual-lane platform:
  a public Core ML deployment lane for real product use and a private direct-ANE lane for differentiated runtime research.
- Move the benchmark story toward relevant model sizes and device classes instead of over-indexing on the retained `stories110m` M3 Max lane.
- Keep training scoped to validating serving architectures, not as the repo's main strategic center.

### Status

- [ ] Define and document the dual-lane product split:
  `EspressoPrivateANE` for direct ANE runtime work and `EspressoCoreML` for public deployment.
- [ ] Create a benchmark policy update that separates:
  small-model runtime proofs, public product proofs on phone-class hardware, and replacement-program publication claims.
- [ ] Build the first `EspressoCoreML` MVP around existing public Apple APIs:
  model directory contract, tokenizer integration, streaming generation, diagnostics, and sample app surface.
- [ ] Import the highest-value ANE-friendly public-lane optimizations:
  RMSNorm-via-LayerNorm, Linear-to-Conv2d rewrites, in-graph argmax, explicit KV I/O, precomputed RoPE, and chunked prefill/decode packaging.
- [ ] Pick one phone-relevant model target for the public lane:
  likely `0.5B` to `2B`, with an explicit hardware target and retained benchmark harness.
- [ ] Re-center the private lane on the replacement-program order:
  runtime economics, parity, artifact contract, narrow proof-model training, rollout quality, then publication harness.
- [ ] Keep training as a tactical workstream only:
  no broad training platform work unless it directly unlocks a serving architecture decision.
- [ ] Publish a clearer external story:
  current shipping lane, public Core ML lane, and future private-runtime breakthrough lane each need distinct claims and artifacts.

### Execution Plan

#### Phase 1: Strategy + Benchmark Reset

- [ ] Update benchmark/docs language so the retained `stories110m` M3 Max result is presented as a shipping regression gate, not the main cross-repo product comparison.
- [ ] Define three benchmark classes and their required disclosures:
  `runtime-proof`, `public-product`, and `publication-breakthrough`.
- [ ] Choose one public comparison target:
  for example a `0.5B` or `2B` text model on iPhone/iPad-class hardware.
- [ ] Freeze the success criteria for the public lane:
  App-Store-safe stack, retained artifacts, memory/placement reporting, and coherent interactive generation.

#### Phase 2: EspressoCoreML MVP

- [ ] Create an `EspressoCoreML` module boundary and model directory contract.
- [ ] Add HF tokenizer integration and robust prompt/token decoding APIs.
- [ ] Add a public runtime that supports:
  monolithic Core ML and chunked prefill/decode layouts.
- [ ] Implement placement and memory diagnostics:
  `MLComputePlan`, `task_vm_info`, compile/load timing, and fallback visibility.
- [ ] Add a sample iOS/macOS app or one shared sample surface for public-lane demos.
- [ ] Retain the first public-lane benchmark artifacts on a phone- or tablet-class target.

#### Phase 3: Public-Lane Performance Work

- [ ] Implement ANE-friendly export graph transforms in the conversion pipeline.
- [ ] Add explicit KV I/O and chunked prefill/decode packaging to the public lane.
- [ ] Measure which optimizations move:
  TTFT, decode tok/s, memory footprint, and ANE placement.
- [ ] Compare Espresso public-lane results against vanilla Core ML exports, not against the private direct-ANE runtime.
- [ ] Keep the public lane honest:
  if it does not materially improve on a standard Core ML path, treat that as a real result.

#### Phase 4: Private-Lane Focus

- [ ] Continue the replacement-program work only where runtime economics already justify further investment.
- [ ] Kill candidate recurrent families quickly when free-running rollout quality stays structurally bad after bounded fixes.
- [ ] Prioritize artifact-contract and parity closure before more teacher/distillation work.
- [ ] Only elevate private-lane wins when they beat the benchmark contract on relevant model sizes or unlock capabilities the public lane cannot match.

#### Phase 5: Product Story

- [ ] Define the external positioning clearly:
  `EspressoCoreML` for public deployment and `EspressoPrivateANE` for research/enterprise/internal use.
- [ ] Stop leading with small-model desktop numbers when comparing against phone-class public repos.
- [ ] Produce one defendable comparison matrix:
  same task class, same device class, similar model scale, explicit lane labels.

### Review

- Strategic conclusion after comparing Espresso against `CoreML-LLM`:
  Espresso should remain the main repo because the direct-ANE runtime/compiler path is the more differentiated technical asset.
- Product conclusion:
  Espresso currently needs a public Core ML lane because a phone-class `2B` public deployment result is more strategically relevant than a faster `110M` desktop proof.
- Organizational conclusion:
  training remains necessary, but only as a bounded instrument for validating serving architectures. It should not become the repo's center of gravity.

## Repo Instruction Refresh 2026-04-10

### Goal

- Refresh project-level agent instructions so Claude-style and AGENTS-style tooling share one consistent execution contract for Espresso.
- Make the benchmark taxonomy, replacement-program workflow, and ANE safety rules explicit enough that autonomous agents do not confuse microbench wins with publishable results.

### Status

- [x] Inspect the existing root `CLAUDE.md` and repo state for stale or missing agent guidance.
- [x] Research current official guidance for project-level `CLAUDE.md` and `AGENTS.md` usage.
- [x] Create a repo-level `AGENTS.md` that captures the shared Espresso execution contract.
- [x] Rewrite `CLAUDE.md` to import `AGENTS.md` and keep only Claude-specific guidance.
- [x] Verify the resulting files are concise, non-duplicative, and aligned with current repo goals.

### Review

- Added root `AGENTS.md` as the shared cross-agent contract for Espresso. It now defines:
  lane taxonomy (`shipping`, `publication`, `probe`, `microbench`), benchmark reporting requirements, the replacement-program work order, kill criteria, ANE runtime invariants, and the minimum verification ladder.
- Replaced the older standalone `CLAUDE.md` with a thin Claude-specific entrypoint that imports `@AGENTS.md`, which keeps shared instructions in one file and avoids drift across agent ecosystems.
- Final file sizes are intentionally small:
  `AGENTS.md` is `174` lines and `CLAUDE.md` is `8` lines.
- This refresh was based on the current repo state plus official guidance that project instructions should be concise, checked into the repo root, and imported rather than duplicated when a repository already uses `AGENTS.md`.

## Claude Setup Upgrade 2026-04-10

### Goal

- Turn the repo's Claude configuration into a real shared project setup instead of a mostly local-only one.
- Move high-churn guidance out of root instruction files and into scoped Claude rules and reusable workflow commands.

### Status

- [x] Audit `.claude/` contents and current ignore behavior.
- [x] Add shared `.claude/settings.json` with excludes for archived Claude worktrees.
- [x] Add path-scoped `.claude/rules/` for ANE runtime work, recurrent research, and benchmark or artifact discipline.
- [x] Add reusable `.claude/commands/` for shipping qualification, publication smokes, recurrent parity certification, and experiment logging.
- [x] Add `CLAUDE.local.md.example` for machine-local personal instructions.
- [x] Fix `.gitignore` so shared Claude config is tracked while local settings and archived worktrees remain local.

### Review

- The repo now has a modular Claude setup:
  `.claude/settings.json`,
  `.claude/rules/*.md`,
  `.claude/commands/*.md`,
  tracked project `.claude/skills/`,
  and `CLAUDE.local.md.example`.
- The new rules are path-scoped to reduce startup context and keep root `CLAUDE.md` small.
- The new commands encode the repo's actual benchmark workflows instead of relying on repeated ad hoc prompts.
- `.gitignore` no longer hides the shared Claude setup; only `settings.local.json`, archived Claude worktrees, `.claude/.DS_Store`, and `CLAUDE.local.md` stay local.

## 750+ Tok/s Breakthrough Program 2026-04-09

### Goal

- Publish a reproducible `>= 750 tok/s` ANE benchmark with coherent `128`-token output, not a local-artifact demo.
- Preserve a hard separation between:
  the current exact Stories shipping gate and the future breakthrough publication lane.
- Treat current exact-lane optimizations as supporting work only. The main bet is a new ANE-native recurrent-hybrid serving path.
- Treat the breakthrough lane as a replacement program:
  new model family, new runtime contract, new `.esp` metadata, and new publication protocol.

### Status

- [x] Add exact-lane timing telemetry to the retained benchmark output so QKV / Metal / FFN / IO shares are measurable on every run.
- [x] Split benchmark policy into two lanes: shipping gate for the current exact Stories path, and publication suite for the breakthrough lane.
- [x] Define the first recurrent-hybrid target architecture with ANE-friendly constraints and explicit quality/throughput kill criteria.
- [x] Build a first-class Phase 2 recurrent micro-runtime proof and retain hardware artifacts for it.
- [ ] Build a training/distillation pipeline for a recurrent-hybrid Stories-class model teacher-aligned to the retained exact bundle.
  Structural blocker reduced: `scripts/distill_stories_rwkv.py` and `scripts/tests/test_distill_stories_rwkv.py` are restored with current config support, but a real Stories-teacher run is still pending.
- [ ] Extend the `.esp` format to carry recurrent-hybrid metadata, persistent state layout, and sparse-attention schedule information.
- [ ] Implement the recurrent-hybrid runtime/compiler path in Espresso with persistent ANE state and no per-token KV growth on recurrent layers.
- [ ] Prove the new path on microbench and quality gates before attempting any public benchmark claim.
- [ ] Run repeated publication-suite benchmarks and retain the branch only if the result is stable enough to defend publicly.

### Review

- Phase 0 and the measurement half of Phase 1 are now implemented.
- `espresso-generate suite` now distinguishes `shipping` vs `publication`, validates the publication contract, emits richer benchmark metadata, and writes hybrid decode timing totals plus shares into compare artifacts and suite summaries.
- `autoresearch-espresso/experiment_runner.py` now carries suite kind, explicit power mode, and repeated full-suite publication execution with an aggregate manifest.
- Added publication prompts at `scripts/stories_publication_benchmark_prompts.txt` and the first recurrent-hybrid target spec at `docs/platform/2026-04-09-recurrent-hybrid-target-spec.md`.
- Verified with:
  `swift test --filter EspressoGenerateTests`
  `python3 -m unittest autoresearch-espresso/test_experiment_runner.py`
  `swift build -c release --product espresso-generate`
  `ANE_COMPILE_CACHE_POLICY=preferCached .build/release/espresso-generate suite --bundle .build/release-bundles/stories110m-smoke.esp --prompts scripts/stories_release_benchmark_prompts.txt --suite-kind shipping --max-tokens 8 --runs 1 --compare-warmup 0 --compare-iterations 1 --coreml-model "$HOME/Library/Application Support/Espresso/demo/stories110m_coreml/stories110m_stateful_seq128.mlpackage" --coreml-seq-len 128 --coreml-compute-units cpu_only --no-power --output-dir results/phase01-shipping-smoke`
- Shipping-smoke result:
  aggregate `69.06 tok/s`, `1.92 ms` TTFT, `16.03 ms` median token, `17.48 ms` p95, correctness `ALL PASS`.
  Hybrid decode timing median on that run:
  `qkv 53.99 ms`, `metal 156.09 ms`, `ffn 88.64 ms`, `io 11.93 ms`, `total 310.64 ms`.
  Hybrid share median:
  `qkv 18.8%`, `metal 42.1%`, `ffn 33.9%`, `io 3.8%`.
- Phase 2 recurrent micro-runtime proof is now a retained CLI mode:
  `espresso-multitoken-probe --mode recurrent-scaling`.
  Verified artifact:
  `results/phase2-recurrent-scaling/summary.json`.
  Synthetic direct-ANE recurrent state update throughput was:
  `3451 tok/s @ ctx32`,
  `4191 tok/s @ ctx256`,
  `5179 tok/s @ ctx1024`,
  `4764 tok/s @ ctx4096`,
  versus transformer decode `262-315 tok/s` on the same hardware.
  Median recurrent speedup versus transformer across the compared contexts was `14.94x`.
- Artifact-backed recurrent checkpoint sanity check is retained at:
  `results/phase2-local-bigram-nosidecar/summary.json`.
  Using the complete `results/20260331-235843/local-bigram.recurrent.bin` checkpoint family without the stale future sidecar,
  the exact recurrent control path reached `1064.16 tok/s` and the exact two-step path reached `2003.31 tok/s` with `parity_status=match`,
  `committed_exact_tokens_per_pass=1.6`, and `accepted_future_tokens_per_pass=0.6`.
- Fixed a real runtime regression while validating the artifact-backed path:
  the exact two-step recurrent model was charging promotion latency into `stateAdvanceLatencyMs` even when no future token was accepted.
  The fix now folds rejected single-step promotion cost into verifier trunk latency and keeps `stateAdvanceLatencyMs = 0` on rejections, matching the harness contract and existing tests.
- Publication-lane status changed:
  the repo-local `results/stories110m_stateful_seq256_exact.mlpackage` is compatible enough to run current compare flows, so the old `seq128` sequence-length blocker is no longer the first failure.
  A real `publication` suite smoke with `128` generated tokens progressed through `19/24` prompts before failing with `ANE compile budget exhausted — exec() restart required`.
  That makes long-run compile-budget management the active publication-lane blocker, not the lack of a larger Core ML baseline.
- Hard blocker updated:
  the old `seq128` Core ML baseline was a real blocker, but the repo-local `seq256` exact export demonstrates that sequence length is no longer the first failure in this workspace.
  The current blocker for long publication runs is per-process ANE compile-budget exhaustion during the suite, which likely requires prompt-level process isolation or an exec-restart/resume strategy.
- Publication-suite process isolation is now proven on the retained contract:
  `ANE_COMPILE_CACHE_POLICY=preferCached .build/release/espresso-generate suite --bundle .build/release-bundles/stories110m-smoke.esp --prompts scripts/stories_publication_benchmark_prompts.txt --suite-kind publication --max-tokens 128 --runs 1 --compare-warmup 0 --compare-iterations 1 --coreml-model results/stories110m_stateful_seq256_exact.mlpackage --coreml-seq-len 256 --coreml-compute-units cpu_only --no-power --output-dir results/phase2-publication-smoke-seq256-isolated-repro`
  completed all `24` prompts and wrote `results/phase2-publication-smoke-seq256-isolated-repro/suite-summary.json`.
- That retained publication smoke is a hard keep/kill result:
  aggregate exact-lane median `53.00 tok/s`, `1.88 ms` TTFT, `18.75 ms` median token, `22.25 ms` p95, hybrid decode median `qkv 511.63 ms | metal 1346.92 ms | ffn 916.67 ms | io 207.70 ms`, share median `qkv 17.2% | metal 45.7% | ffn 30.7% | io 6.3%`.
  Correctness failed on `8/24` prompts:
  `fox_return`, `news_bulletin`, `recipe_disaster`, `classroom_scene`, `game_commentary`, `village_rumor`, `time_travel`, `library_note`.
  Performance gates passed, but the retained publication candidate is neither exact nor fast enough to defend publicly.
- Follow-up A/B on the failed `fox_return` prompt showed the mismatch is not solved by switching exact-head backends:
  default `ane_classifier`, `cpu_fp16_tiled`, and `cpu_partitioned_fp32` all diverged at generated token `32`.
  Forcing `ESPRESSO_USE_CPU_EXACT_DECODE=1` diverged even earlier at generated token `3`.
  Keep verdict:
  the remaining exact-lane correctness issue is broader than the ANE classifier alone and should be treated as a trunk/prefill/numerics investigation, not a head-only fix.
- Recovered the missing recurrent training/export path from git history and modernized it for current configs:
  `scripts/distill_stories_rwkv.py` now supports `teacher_head_seeded_trunk` plus `recurrent_init_std`,
  exports `RGW1` recurrent checkpoints compatible with the current loader contract,
  and is covered by `scripts/tests/test_distill_stories_rwkv.py`.
  Verified with:
  `uv run --with torch --with transformers python -m unittest scripts.tests.test_distill_stories_rwkv scripts.tests.test_distill_stories_native`
- Critical recurrent-lane contract fix:
  the recovered distillation path and the Swift recurrent probe were both using the homegrown `tokenizer.model` contract, while the retained Stories bundle also ships a `tokenizer.json` contract used by the publication lane.
  On the same prompt, the old path encoded `49` tokens while `tokenizer.json` encoded `12`.
  After changing `scripts/distill_stories_native.py` to prefer `tokenizer.json`, teacher label accuracy on the same recurrent-quality corpus jumped from about `3.3%` to `51.4%`.
  `espresso-multitoken-probe` was extended with `--prompt-token-ids` so ANE recurrent runs can now be driven by exact token IDs from the corrected tokenizer contract even though the current Swift `tokenizer.json` loader still rejects the Stories tokenizer due non-contiguous IDs.
- Retained 3-layer RWKV quality verdict:
  the corrected-tokenizer retrains are now a hard kill for the current `3-layer` RWKV publication candidate.
  `results/rwkv-3layer-quality-v1/distill-report.json` trained only the `7.08M` recurrent parameters with a frozen copied head over `120` corpus texts and reached `20.15%` label-token accuracy, but the retained ANE fused-triplet run at `results/rwkv-3layer-quality-v1` still collapsed after a short prefix: `863.08 tok/s` with a `105`-token repetition run on the sample prompt.
  A CE-heavier retry at `results/rwkv-3layer-quality-v2/distill-report.json` moved the report only marginally (`21.10%` label-token accuracy, `6.67%` exact-two-token future accept) and still collapsed in free-running decode, while fused-triplet throughput fell to `410.65 tok/s`.
  Keep verdict:
  the current `3-layer` RWKV cell is fast enough to matter but not quality-capable enough for a publishable `128`-token benchmark, even after fixing the tokenizer contract and freezing the head.
  Next architecture step should pivot to a different recurrent-hybrid cell family rather than more local tuning on this branch.
- Added the next architecture spec at `docs/platform/2026-04-09-gated-deltanet-pivot.md`:
  the concrete pivot is a Gated DeltaNet-H1-style recurrent-hybrid path with bounded sliding-window attention checkpoints, associative recurrent state kept resident on ANE, and explicit micro-runtime / proof-model kill criteria before broader investment.
- First replacement-program code slice is now in:
  `scripts/distill_stories_rwkv.py` exports and reloads explicit recurrent-hybrid metadata fields (`cellFamily`, `headCount`, `stateShape`, `checkpointSchedule`) and `scripts/tests/test_distill_stories_rwkv.py` verifies the schema roundtrip.
  This does not change serving behavior yet; it freezes the artifact contract boundary so the future Gated DeltaNet lane does not masquerade as the dead-end RWKV line.
  Verified with:
  `source .venv-rwkv/bin/activate && python -m unittest scripts.tests.test_distill_stories_rwkv scripts.tests.test_distill_stories_native`
- Swift-side recurrent contract is now real:
  `Sources/Espresso/RecurrentModelMetadata.swift` adds typed recurrent metadata parsing and validation, `Sources/Espresso/MultitokenProbeSupport.swift` now loads sibling `metadata.json` for recurrent checkpoints and rejects unsupported hybrid cell families in the current RWKV loader, and `Sources/EspressoMultiTokenProbe/main.swift` emits recurrent metadata into probe payloads.
  Verified with:
  `swift test --filter MultitokenProbeSupportTests`
  `swift build -c release --product espresso-multitoken-probe`
- Grouped dual-state proof-contract hardening is now retained:
  `Sources/Espresso/RecurrentModelMetadata.swift`,
  `Tests/EspressoTests/RecurrentModelMetadataTests.swift`,
  `Tests/EspressoTests/MultitokenProbeSupportTests.swift`,
  `scripts/distill_stories_rwkv.py`,
  `scripts/tests/test_distill_stories_rwkv.py`,
  `configs/stories/stories-gds-12x128-v1-proof.json`,
  and `docs/platform/2026-04-10-grouped-dual-state-proof-model-contract.md`
  now separate proof-model metadata validation from actual RWKV checkpoint/runtime support.
  Keep verdict:
  grouped-dual-state is now a first-class validated spec in both Swift and Python, but the Python RWKV module no longer implies fake support by instantiating, exporting, or reloading grouped-dual-state checkpoints.
  Verified with:
  `swift test --filter RecurrentModelMetadataTests`
  `swift test --filter MultitokenProbeSupportTests`
  `python -m unittest scripts.tests.test_distill_stories_rwkv`
- First associative-state runtime economics proof is now retained:
  `Sources/ANETypes/AssociativeStatePrototypeWeights.swift`,
  `Sources/MILGenerator/AssociativeStatePrototypeStepGenerator.swift`,
  `Sources/ANERuntime/AssociativeStatePrototypeKernelSet.swift`,
  `Sources/Espresso/AssociativeStatePrototypeDecode.swift`,
  and `Sources/EspressoBench/main.swift` / `Sources/EspressoBench/ANEDirectBench.swift`
  add a bench-only one-block associative-state prototype and `--associative-state-microbench`.
  Retained artifact:
  `benchmarks/results/associative-state-microbench-20260410/summary.json`
  with `3670.57 tok/s`, `0.272 ms` median token, compile/init `121.73 ms`, and average timing split:
  `state_write 0.004 ms`, `state_update 0.267 ms`, `state_promote 0.003 ms`, `state_readout 0.002 ms`.
  Keep/kill view:
  the prototype clears the absolute `>300 tok/s` Milestone A gate easily, but it does not yet clear the stronger replacement-program criterion of being materially faster than the current RWKV vector-state recurrent path.
  Same-host control on `espresso-multitoken-probe --mode recurrent-scaling --contexts 32 --warmup 3 --iterations 20` measured `3585.57 tok/s`, so the current associative-state prototype is only about `2.4%` faster and should be treated as a negative-control runtime fork, not yet a decisive breakthrough path.
- Richer-state runtime proof is now retained and clears the stronger economics gate:
  `Sources/ANETypes/GroupedDualStatePrototypeWeights.swift`,
  `Sources/MILGenerator/GroupedDualStatePrototypeStepGenerator.swift`,
  `Sources/ANERuntime/GroupedDualStatePrototypeKernelSet.swift`,
  `Sources/Espresso/GroupedDualStatePrototypeDecode.swift`,
  `Tests/MILGeneratorTests/GroupedDualStatePrototypeStepGeneratorTests.swift`,
  `Tests/EspressoTests/GroupedDualStatePrototypeHardwareTests.swift`,
  and `Sources/EspressoBench/main.swift` / `Sources/EspressoBench/ANEDirectBench.swift`
  add a bench-only grouped dual-state recurrent prototype and `--grouped-dual-state-microbench`.
  Verified with:
  `swift test --filter GroupedDualStatePrototypeStepGeneratorTests`
  `ANE_HARDWARE_TESTS=1 swift test --filter GroupedDualStatePrototypeHardwareTests`
  `swift build -c release --product espresso-bench`
  Fresh same-host RWKV control:
  `ANE_COMPILE_CACHE_POLICY=preferCached .build/release/espresso-multitoken-probe --mode recurrent-scaling --contexts 32 --warmup 3 --iterations 20`
  retained `3696.86 tok/s @ 0.2705 ms/token`.
  Retained grouped dual-state sweep:
  `benchmarks/results/grouped-dual-state-microbench-20260410-12x64/summary.json`
  `4922.58 tok/s @ 0.203 ms/token`.
  `benchmarks/results/grouped-dual-state-microbench-20260410-12x128/summary.json`
  `4767.58 tok/s @ 0.210 ms/token`.
  `benchmarks/results/grouped-dual-state-microbench-20260410-12x192/summary.json`
  `3545.32 tok/s @ 0.282 ms/token`.
  Fresh-release repeat pair:
  control reruns measured `3012.05 tok/s` and `3041.44 tok/s`,
  `benchmarks/results/grouped-dual-state-microbench-20260410-12x128-r2/summary.json`
  retained `3844.31 tok/s @ 0.260 ms/token`,
  and `benchmarks/results/grouped-dual-state-microbench-20260410-12x64-r2/summary.json`
  retained `5137.54 tok/s @ 0.195 ms/token`.
  `benchmarks/results/grouped-dual-state-microbench-20260410-12x192-r2/summary.json`
  retained `3300.56 tok/s @ 0.303 ms/token`.
  Keep/kill view:
  absolute tok/s drifts across immediate reruns, but the grouped dual-state path retained a material edge over the rebuilt RWKV control.
  `12x64` is the current speed leader and remains clearly ahead on repeat.
  `12x128` is the balanced keep candidate and stayed about `+26%` ahead of the repeated RWKV control while keeping a materially larger memory surface.
  `12x192` is not a preferred keep point: it is only a small gain over the repeated RWKV control and far weaker than `12x64` / `12x128`.

### Replacement Program Freeze

- Program thesis:
  `750+ tok/s` with coherent `128`-token output is now a replacement-program problem, not an optimization problem on the current exact transformer lane.
- Target candidate:
  `stories-gdn-h1-v1`, a Gated DeltaNet-style recurrent-hybrid decoder with sparse checkpoint attention.
- Program workstreams:
  model architecture, ANE runtime contract, artifact/export contract, quality pipeline, and publication methodology must advance together.
- First hard rule:
  do not spend more major effort on the current `3-layer` RWKV branch or exact-head-only tuning as publication candidates.
- Second hard rule:
  do not train another Stories-class recurrent student until the one-block delta-rule micro-runtime proves the decode economics are materially better than the current vector-state recurrent path.

### Immediate Execution Tracks

- Track A: micro-runtime proof
  build a one-block Gated DeltaNet-style recurrent decode path with persistent associative state, fixed state surfaces, repeatability tests, and timing for state-update vs readout.
- Track B: hybrid proof
  add one sliding-window checkpoint-attention block after the recurrent block and measure recurrent share vs checkpoint-attention share.
- Track C: artifact contract
  extend recurrent metadata to carry `cellFamily`, `stateShape`, `headCount`, and `checkpointSchedule`, then make the loader reject incomplete hybrid metadata early.
- Track D: proof-model training
  only after Track A passes, distill a narrow Stories proof model with rollout-based quality gates as the primary keep/kill signal.
- Track E: publication gate
  only after Track D passes, run repeated publication-suite benchmarks and retain the branch only if throughput and coherence are both stable.

### Immediate Deliverables

- [ ] Add a new recurrent cell family contract for Gated DeltaNet-style weights and metadata.
- [ ] Implement a one-block associative-state decode micro-runtime on ANE with direct CLI coverage.
- [ ] Add microbench outputs for state-update ms/token, readout ms/token, checkpoint-attention ms/token, and total tok/s.
- [ ] Add reset/repeatability tests for persistent associative state.
- [ ] Define the narrow `stories-gdn-h1-v1` proof-model config and export contract.
- [ ] Add rollout-first quality gates to the recurrent proof-model reports.

### 2026-04-10 Gated DeltaNet Probe Artifact Slice

- [x] Add `gdn-h1` recurrent metadata validation and schedule parsing in Swift, including mixed `gdn` / `swa:window` schedules.
- [x] Add a probe-only `gdn-h1` autoregressive generation model with direct token selection and timing capture.
- [x] Dispatch `espresso-multitoken-probe --mode generate-recurrent` by `cellFamily = gdn-h1`.
- [x] Add loader/store tests and probe support tests for `gdn-h1` artifacts.
- [x] Add a dedicated Python `gdn-h1` distillation/export entrypoint and proof config scaffold.

### 2026-04-10 Gated DeltaNet Probe Artifact Review

- `gdn-h1` is now a retained probe/runtime family in Swift:
  `Sources/Espresso/RecurrentModelMetadata.swift`,
  `Sources/Espresso/MultitokenProbeSupport.swift`,
  `Sources/Espresso/ANEGatedDeltaNetGenerationModel.swift`,
  and `Sources/EspressoMultiTokenProbe/main.swift`
  now validate mixed `gdn` / `swa:window` schedules, load `GDN1` recurrent checkpoints, dispatch probe execution by `cellFamily = gdn-h1`, and report recurrent versus checkpoint timing through the release probe payload.
- `gdn-h1` test coverage is retained in:
  `Tests/EspressoTests/RecurrentModelMetadataTests.swift`,
  `Tests/EspressoTests/MultitokenProbeSupportTests.swift`,
  `Tests/EspressoTests/ANEGatedDeltaNetGenerationModelTests.swift`,
  `Tests/EspressoTests/GatedDeltaNetGenerationWeightStoreTests.swift`,
  and `Tests/EspressoTests/GatedDeltaNetGenerationModelHardwareTests.swift`.
- Verified with:
  `swift test --filter RecurrentModelMetadataTests`
  `swift test --filter MultitokenProbeSupportTests`
  `swift test --filter ANEGatedDeltaNetGenerationModelTests`
  `ANE_HARDWARE_TESTS=1 swift test --filter GatedDeltaNetGenerationModelHardwareTests`
  `source .venv-rwkv/bin/activate && python -m unittest scripts.tests.test_distill_stories_gated_deltanet`
  `swift build -c release --product espresso-multitoken-probe`
- The proof export path is retained with:
  `python scripts/distill_stories_gated_deltanet.py --config configs/stories/stories-gdn-h1-v1-proof.json --dry-run`
  which writes `results/stories-gdn-h1-v1-proof/weights/recurrent.bin`,
  `results/stories-gdn-h1-v1-proof/weights/checkpoint_swa.bin`,
  and `results/stories-gdn-h1-v1-proof/metadata.json`.
- The release-binary end-to-end probe proof is now retained with:
  `ANE_COMPILE_CACHE_POLICY=preferCached .build/release/espresso-multitoken-probe --mode generate-recurrent --input recurrent-checkpoint --recurrent-checkpoint results/stories-gdn-h1-v1-proof/weights/recurrent.bin --layer-count 12 --max-new-tokens 4 --max-sequence-tokens 16 --warmup 1 --iterations 1 --prompt-token-ids 0 --output-head-backend cpu`
  Result:
  `106.31 tok/s`, median trunk `7.65 ms/token`, `cellFamily = gdn-h1`, mixed `gdn/gdn/gdn/swa:64` schedule metadata, and the dry-run artifact still emits token `0` for every generated step.
- The first real smoke-trained `gdn-h1` artifact is retained at:
  `results/stories-gdn-h1-v1-smoke/distill-report.json`
  and
  `results/stories-gdn-h1-v1-smoke/probe-once-upon-a-time.json`.
  The short `24`-step CPU smoke run reduced mean teacher/student KL from `86.32` to `78.65`, but teacher agreement stayed `1.86%`, free-running rollout samples still collapsed into long repetition runs, and the release probe only reached `77.41 tok/s` while still emitting token `0` for every generated step on the benchmark prompt.
- The `gdn-h1` trainer math is now stronger, but still not a keep:
  `scripts/distill_stories_gated_deltanet.py`
  now adds short free-running rollout KL, recurrent hidden-state anchoring on absolute teacher layer boundaries, and identity-biased gate/projection math. The refreshed smoke run is retained in
  `results/stories-gdn-h1-v1-smoke/distill-report.json`
  and moved mean teacher/student KL from `86.32` to `78.84`, while the retained rollout samples improved only slightly and still showed `49-56` token repetition runs.
- The Swift runtime contract is partially corrected but still not equivalent:
  `Sources/MILGenerator/GatedDeltaNetPrototypeStepGenerator.swift`
  now matches the Python gate scaling, negative gate bias, carry interpolation, and output residual scaling, and
  `Sources/ModelSupport/GPT2BPETokenizer.swift`
  plus `Sources/EspressoMultiTokenProbe/main.swift`
  now encode SentencePiece-style HF `tokenizer.json` exports correctly for the probe path without injecting post-processor prefix tokens.
  After the tokenizer fix, the release probe prompt tokens for `"Once upon a time"` now match Python exactly as `[9038, 2501, 263, 931]`.
- The retained post-fix release probe artifact is:
  `results/stories-gdn-h1-v1-smoke/probe-once-upon-a-time-v5.json`
  Result:
  `85.58 tok/s`, prompt tokens `[9038, 2501, 263, 931]`, but generated tokens still collapse to all `0`.
  Python loading the exact same exported checkpoint now generates repeated token `931` (`"time"`), so the remaining blocker is no longer tokenizer mismatch or pure training collapse; it is a runtime equivalence gap between the exported `gdn-h1` artifact and the ANE probe path.
- The recurrent ANE surface-binding fix is now retained and materially changes the live `gdn-h1` result:
  `Sources/ANERuntime/ANESurfaceOwner.swift`
  now exposes `retainInputs` / `retainOutputs` with alphabetical MIL-name binding, and
  `Sources/Espresso/GatedDeltaNetPrototypeDecode.swift`
  plus
  `Sources/Espresso/GroupedDualStatePrototypeDecode.swift`
  bind dual-state prototype surfaces by semantic name rather than positional guesswork.
  Verification:
  `swift test --filter ANESurfaceOwnerTests`
  and
  `ANE_HARDWARE_TESTS=1 swift test --filter GatedDeltaNetGenerationModelHardwareTests`
  both pass.
  The new retained probe artifact is:
  `results/stories-gdn-h1-v1-smoke/probe-once-upon-a-time-v6.json`
  Result:
  prompt tokens still match Python as `[9038, 2501, 263, 931]`, but the old all-`0` collapse is gone. The probe now emits `[591, 366, 29892, 366, 29892, 484, 484, 484]`, decodes to `Once upon a time we you, you,nenene`, and reaches `97.63 tok/s`.
  This keeps the runtime-equivalence diagnosis alive but narrows it: the recurrent path is no longer obviously miswired, yet full parity is still open because Swift/ANE still does not match the Python-side `time time ...` collapse.
- The mixed checkpoint path now has an explicit Metal RoPE config:
  `Sources/Espresso/GroupedDualStateCheckpointSession.swift`
  now passes `metalRoPEConfig` into `ForwardPass.runHybridDecodeTimed` for mixed `gdn+swa` checkpoints, eliminating the prior "no RoPE hook/config" mismatch on the Swift side.
  Verification:
  `swift build -c release --product espresso-multitoken-probe`
  still passes, and the retained post-patch probe artifact is:
  `results/stories-gdn-h1-v1-smoke/probe-once-upon-a-time-v7.json`
  Result:
  the decode changes again to `[13, 366, 366, 366, 3026, 29872, 29872, 29872]` at `88.48 tok/s`.
  This is not full parity, but it confirms the checkpoint lane was semantically participating in the mismatch rather than merely inheriting the recurrent error.
- The `gdn-h1` runtime-equivalence question is now closed for the retained smoke artifact:
  `scripts/distill_stories_gated_deltanet.py`
  now exposes `load_gated_deltanet_runtime_runner(...)`, which reconstructs the mixed scheduled Python runner from the exported recurrent checkpoint plus `weights/checkpoint_swa.bin`, and
  `scripts/tests/test_distill_stories_gated_deltanet.py`
  proves that the sidecar-loaded runner matches a manually built scheduled runner.
  The retained Python reference trace is:
  `results/stories-gdn-h1-v1-smoke/python-runtime-trace-v1.json`
  and it now matches the Swift/ANE mixed runner on the benchmark prompt, including generated tokens `[13, 366, 366, 366, 3026, 29872, 29872, 29872]`.
  On the Swift side,
  `Tests/EspressoTests/HybridDecodeForwardPassTests.swift`
  now also proves on hardware that the retained smoke checkpoint layer matches CPU decode attention when run through the hybrid decode path.
- Current blocker:
  the `gdn-h1` lane is now runtime-equivalent enough for proof work on the retained smoke artifact, and the main blocker has shifted from runtime equivalence to model quality. The mixed scheduled Python runner and Swift/ANE now agree on the same bad token stream, so further work on `gdn-h1` must improve free-running quality rather than chase nonexistent parity gaps. `.esp` recurrent-hybrid bundle packing is still reported as `skipped: recurrent-hybrid bundle packing not implemented yet`.
- The bounded `gdn-h1` quality-recovery ladder is now retained and does not clear the publication-quality gate:
  `configs/stories/stories-gdn-h1-v1-quality-a.json`,
  `configs/stories/stories-gdn-h1-v1-quality-b.json`,
  and
  `configs/stories/stories-gdn-h1-v1-quality-c.json`
  all ran successfully and now emit explicit `rollout_quality` summaries in their distill reports.
  Results:
  `results/stories-gdn-h1-v1-quality-a/distill-report.json`
  improved final teacher agreement to `0.0487` and KL to `64.15`, but `rollout_quality.samples_passing=0`, `worst_unique_token_ratio=0.0078`, and one prompt collapsed into a `128`-token single-token loop.
  `results/stories-gdn-h1-v1-quality-b/distill-report.json`
  with trainable embeddings improved final teacher agreement to `0.0942` and KL to `57.58`, but still had `samples_passing=0` and the same `128`-token single-token collapse on the first prompt.
  `results/stories-gdn-h1-v1-quality-c/distill-report.json`
  with denser `gdn/swa` alternation also failed `samples_passing=0` and regressed mean unique-token ratio further.
  Conclusion:
  `gdn-h1` is no longer the lead publication line. The architecture can be trained harder and match its own mixed runtime, but this bounded quality ladder failed to recover coherent `128`-token rollouts.
- The grouped-dual-state successor line now has the same Python mixed-runner certification harness:
  `scripts/distill_stories_grouped_dual_state.py`
  now exposes `load_grouped_dual_state_runtime_runner(...)`, and
  `scripts/tests/test_distill_stories_grouped_dual_state.py`
  proves the sidecar-loaded runner matches a manually built scheduled runner.
  The first retained Python successor trace is:
  `results/stories-gds-12x128-v1-fast-smoke/python-runtime-trace-v1.json`
  and it does **not** match the Swift probe result in
  `results/stories-gds-12x128-v1-fast-smoke/probe-once-upon-a-time-v1.json`.
  Python mixed-runner generated `[29892, 29892, 263, 263, 29892, 29892, 29892, 727]`, while Swift/ANE generated `[287, 484, 484, 1058, 1058, 484, 484, 484]` at `90.50 tok/s`.
  That makes grouped-dual-state the next active certification target: unlike `gdn-h1`, it still has an open runtime-equivalence question on the retained artifact.

### 2026-04-10 Richer-State Runtime Proof Slice

- [x] Implement a grouped dual-state recurrent prototype with one expanded memory surface and one carry surface, kept bench-only in `espresso-bench`.
- [x] Add generator-contract tests for the grouped dual-state prototype: byte sizes, IO names, blob references, and op subset.
- [x] Add a hardware-gated reset/repeatability test for the grouped dual-state session.
- [x] Benchmark the grouped dual-state prototype against the same-host RWKV recurrent control and keep it only if the speedup is material.

### 2026-04-10 Grouped Dual-State Contract Hardening Slice

- [x] Extend recurrent-hybrid metadata to carry explicit `stateLayout` and validate grouped dual-state contracts in Swift and Python.
- [x] Freeze `configs/stories/stories-gds-12x128-v1-proof.json` as the first proof-model runtime contract for the replacement program.
- [x] Harden `scripts/distill_stories_rwkv.py` so grouped-dual-state remains a validated proof config only: hybrid specs parse and round-trip metadata, but RWKV student construction, checkpoint export, checkpoint load, and full distillation remain `rwkv-v1` only.
- [x] Add refusal tests that prove the Python RWKV trainer/checkpoint path does not masquerade as grouped-dual-state support.

### 2026-04-10 Grouped Dual-State Probe Artifact Slice

- [x] Add a real grouped-dual-state checkpoint artifact store separate from the RWKV `RGW1` path.
- [x] Add a probe-only grouped-dual-state autoregressive generation wrapper with CPU output-head scoring and direct selection harness support.
- [x] Dispatch `espresso-multitoken-probe --mode generate-recurrent` by recurrent cell family so `grouped-dual-state-v1` artifacts run through the new proof runtime while unsupported hybrids still fail honestly.
- [x] Add store roundtrip tests, loader tests, and a hardware-gated grouped-dual-state generation smoke test.
- [x] Retain an end-to-end release-binary artifact proof with a synthetic grouped-dual-state checkpoint and metadata sidecar.

### 2026-04-10 Grouped Dual-State Probe Artifact Review

- Proof-only grouped-dual-state serving is now a retained release-binary path:
  `Sources/Espresso/GroupedDualStateGenerationWeights.swift`,
  `Sources/Espresso/GroupedDualStateGenerationWeightStore.swift`,
  `Sources/Espresso/ANEGroupedDualStateGenerationModel.swift`,
  `Sources/Espresso/MultitokenProbeSupport.swift`,
  `Sources/Espresso/RecurrentModelMetadata.swift`,
  and `Sources/EspressoMultiTokenProbe/main.swift`
  add a dedicated checkpoint format, metadata validation, probe loader, and autoregressive generation wrapper for `grouped-dual-state-v1`.
- Test coverage is retained in:
  `Tests/EspressoTests/GroupedDualStateGenerationWeightStoreTests.swift`,
  `Tests/EspressoTests/MultitokenProbeSupportTests.swift`,
  and `Tests/EspressoTests/GroupedDualStateGenerationModelHardwareTests.swift`.
- Verified with:
  `swift test --filter GroupedDualStateGenerationWeightStoreTests`
  `swift test --filter MultitokenProbeSupportTests`
  `ANE_HARDWARE_TESTS=1 swift test --filter GroupedDualStateGenerationModelHardwareTests`
  `swift build -c release --product espresso-multitoken-probe`
- Retained release-binary artifact proof:
  `ANE_COMPILE_CACHE_POLICY=preferCached .build/release/espresso-multitoken-probe --mode generate-recurrent --input recurrent-checkpoint --recurrent-checkpoint results/grouped-dual-state-cli-smoke-20260410/recurrent.bin --layer-count 1 --max-new-tokens 4 --max-sequence-tokens 8 --warmup 1 --iterations 1 --prompt-token-ids 0 --output-head-backend cpu`
  wrote `results/grouped-dual-state-cli-smoke-20260410/summary.json`.
  Result:
  `generated_tokens = [0, 0, 0, 0]`,
  `cellFamily = grouped-dual-state-v1`,

### 2026-04-10 Gated DeltaNet Probe Integration Slice

- [x] Add `gdn-h1` schedule semantics to recurrent metadata in Swift so `checkpointSchedule` can express `gdn` recurrent entries plus `swa:<window>` checkpoint entries.
- [x] Add a real `ANEGatedDeltaNetGenerationModel` with the same proof-runtime contract as grouped-dual-state: recurrent sessions on ANE, sparse checkpoint sessions through the existing Metal sidecar, CPU logits selection only for the proof lane.
- [x] Dispatch `espresso-multitoken-probe --mode generate-recurrent` by `cellFamily = gdn-h1`, including recurrent checkpoint loading, sidecar checkpoint resolution, and metadata validation.
- [x] Add focused Swift tests for `gdn-h1` metadata parsing, probe weight loading, and generation-weight-store roundtrip integrity without regressing grouped-dual-state support.
- [x] Add a dedicated Python `gdn-h1` trainer/exporter path with its own config contract instead of overloading grouped-dual-state once the Swift probe lane is executable.

### 2026-04-10 Grouped Dual-State Python Export Slice

- [x] Add a dedicated `scripts/distill_stories_grouped_dual_state.py` entrypoint instead of overloading the RWKV trainer.
- [x] Add a shared recurrent metadata contract module at `scripts/recurrent_distill_common.py` for recurrent-hybrid state layout and metadata export/load.
- [x] Add grouped-dual-state config parsing, student/model parameterization, checkpoint export/load roundtrip, and a dry-run export path with teacher-seeded head initialization.
- [x] Retain unit coverage for grouped-dual-state config parsing, checkpoint export/load, and dry-run artifact export.
- [x] Fix nested recurrent metadata discovery on the Swift probe/runtime path so `recurrentWeightsRef = "weights/recurrent.bin"` resolves metadata from the export root.

### 2026-04-10 Grouped Dual-State Python Export Review

- New Python lane:
  `scripts/recurrent_distill_common.py`,
  `scripts/distill_stories_grouped_dual_state.py`,
  and `scripts/tests/test_distill_stories_grouped_dual_state.py`
  now provide a dedicated grouped-dual-state trainer/exporter with a separate checkpoint format and dry-run artifact export.
- New Swift/runtime fix:
  `Sources/Espresso/RecurrentModelMetadata.swift`
  and `Tests/EspressoTests/RecurrentModelMetadataTests.swift`
  now resolve `metadata.json` from ancestor export directories when the recurrent checkpoint lives under a nested path such as `weights/recurrent.bin`.
- Verified with:
  `source .venv-rwkv/bin/activate && python -m unittest scripts.tests.test_distill_stories_rwkv scripts.tests.test_distill_stories_grouped_dual_state`
  `swift test --filter RecurrentModelMetadataTests`
  `swift test --filter MultitokenProbeSupportTests`
  `swift build -c release --product espresso-multitoken-probe`
- Retained real-config dry run:
  `source .venv-rwkv/bin/activate && python scripts/distill_stories_grouped_dual_state.py --config configs/stories/stories-gds-12x128-v1-proof.json --dry-run`
  exported `results/stories-gds-12x128-v1-proof/weights/recurrent.bin`,
  copied tokenizer assets,
  and wrote `results/stories-gds-12x128-v1-proof/distill-report.json`.
  Report summary:
  `device = mps`,
  `trainable recurrent params = 7,677,696`,
  `steps_completed = 0`,
  `bundle_pack_status = skipped: recurrent-hybrid bundle packing not implemented yet`.
- Retained release-probe verdict on the real proof artifact:
  `ANE_COMPILE_CACHE_POLICY=preferCached .build/release/espresso-multitoken-probe --mode generate-recurrent --input recurrent-checkpoint --recurrent-checkpoint results/stories-gds-12x128-v1-proof/weights/recurrent.bin --layer-count 12 --max-new-tokens 4 --max-sequence-tokens 16 --warmup 1 --iterations 1 --prompt-token-ids 0 --output-head-backend cpu`
  now fails with the intended runtime gate:
  `invalidField("current grouped-dual-state runtime supports only gds checkpointSchedule entries; found swa:64")`.
- Keep/kill view:
  the Python grouped-dual-state export lane is now real and the nested metadata bug is closed.
  The next blocker is no longer artifact format or metadata lookup.
  It is the missing runtime scheduler capability for sparse checkpoint attention.

### Replacement Program Kill Criteria

- Kill the branch if the one-block delta-rule micro-runtime is not materially faster than the current RWKV vector-state path.
- Kill the branch if the two-block hybrid proof still collapses before `128` generated tokens.
- Kill the branch if checkpoint attention must become frequent enough that decode cost returns to transformer territory.
- Kill the branch if associative state cannot remain resident without repeated host copies or unstable surface churn.

### Plan

- Phase 0: Measurement and falsification
  - Surface `HybridDecodeTimingBreakdown` into suite artifacts and CLI output.
  - Create a dedicated publication suite larger than the current `3`-prompt shipping gate while keeping the existing gate unchanged for regression control.
  - Pre-register the breakthrough claim contract:
    same hardware, same OS, same cache policy, same bundle/config, repeated suite runs, median reported with spread, no best-run cherry-picking.
  - Kill criterion:
    if we still cannot attribute exact-lane time cleanly across QKV / Metal / FFN / IO, do not start a large runtime rewrite yet.

- Phase 1: Architecture definition
  - Define a recurrent-hybrid decoder where most layers are recurrent state updates on ANE and only periodic layers run sparse/full attention.
  - Constrain the architecture for ANE:
    BC1S layout, conv1x1-friendly projections, bounded recurrent state, no per-token KV growth on recurrent layers, minimal ANE↔Metal crossings.
  - Start with a Stories-scale target, not a large general-purpose model.
  - Kill criterion:
    if the proposed design still requires Metal attention on most layers during decode, it is not a breakthrough architecture and should be rejected.

- Phase 2: Training and quality pipeline
  - Build a distillation path from the retained exact Stories teacher into the recurrent-hybrid student.
  - Track quality on:
    perplexity, offline next-token agreement, repetition rate, and coherent `128`-token completions on a retained prompt set.
  - Require periodic-attention ablations:
    recurrent-only, sparse-attention every `N` layers, and mixed schedules.
  - Kill criterion:
    if no schedule can produce coherent `128`-token output at Stories scale, abandon the pure recurrent-heavy branch and fall back to a denser hybrid.

- Phase 3: `.esp` and artifact contract
  - Introduce a new bundle behavior class / architecture version for recurrent-hybrid serving.
  - Encode:
    recurrent state tensor schema, attention schedule, recurrent cell type, verifier/public benchmark metadata, and training provenance.
  - Keep the current exact bundle contract intact so regressions are easy to detect.
  - Kill criterion:
    if the new artifact contract cannot coexist cleanly with the exact lane without destabilizing tooling, isolate it behind a separate experimental format first.

- Phase 4: Runtime and compiler implementation
  - Add recurrent-hybrid MIL/codegen builders and private-ANE runtime plumbing for persistent state updates.
  - Keep recurrent state resident across decode steps; avoid rebuilding or copying full history every token.
  - Add prepared-session pooling and delta-compile reuse around recurrent cells and sparse-attention checkpoints.
  - Use exact-lane telemetry discipline from Phase 0 on the new path from the beginning.
  - Kill criterion:
    if isolated recurrent decode cannot beat the current exact decode proxy by at least `2x`, stop before integrating the full model path.

- Phase 5: Milestone validation
  - Milestone A:
    isolated recurrent cell or toy model clears `>300 tok/s` with stable state semantics.
  - Milestone B:
    Stories-class recurrent-hybrid prototype clears `>500 tok/s` with coherent `128`-token output.
  - Milestone C:
    publication candidate clears `>=750 tok/s` on the publication suite while preserving defined quality gates.
  - Kill criterion:
    if Milestone B cannot clear `>500 tok/s` with acceptable coherence, do not invest in publication polish.

- Phase 6: Publishable benchmark protocol
  - Run repeated suite measurements, not single best runs.
  - Report:
    median tok/s, TTFT, median token latency, p95 token latency, pass rate, worst-prompt behavior, and confidence spread.
  - Keep shipping and publication lanes separate:
    the exact Stories path remains the regression gate until the breakthrough lane proves superior and stable.
  - Kill criterion:
    if the candidate reaches `750+ tok/s` but fails coherence or repeated-run stability, it is a demo result, not a publishable benchmark.

### Dependencies

- Instrumentation first:
  without phase-level timing and a stronger publication protocol, future wins will be too noisy to trust.
- Training and runtime must advance together:
  a recurrent-hybrid runtime without a quality-preserving model is useless, and a trained recurrent model without the runtime win does not solve the bottleneck.
- The current exact Stories lane stays untouched except for measurement improvements and low-risk guardrail work.

### Funding Priorities

- Highest priority:
  recurrent-hybrid ANE-native model family with sparse periodic attention.
- Medium priority:
  exact-lane telemetry, publication-suite methodology, and any low-risk ANE/Metal boundary reductions that help root-cause analysis.
- Low priority:
  more exact-head work, same-model speculation, or verifier-side runtime tricks without an architectural discontinuity.

## ANE Throughput Investigation 2026-04-09

### Status

- [x] Separate the retained publishable serving lane from the faster non-publishable recurrent/demo lanes.
- [x] Benchmark exact-head backend variants on the real `stories110m-smoke.esp` bundle and reject head-only dead ends.
- [x] Benchmark llama hybrid layer-input rebinding on the retained Stories serving suite.
- [x] Form the first keep/kill view for immediate runtime work versus architectural breakthrough work.
- [x] Decide whether to make llama hybrid layer-input rebinding the default exact-serving policy and add regression coverage before changing behavior.
- [ ] Surface exact-lane hybrid decode timing telemetry in benchmark output so future work is driven by measured QKV / Metal / FFN / IO shares instead of aggregate tok/s only.
- [x] Convert the architecture findings into a concrete breakthrough program aimed at publishable 128-token quality, not local microbenchmark wins.

### Review

- Retained investigation boundary:
  the publishable lane is the exact 12-layer llama Stories bundle at `.build/release-bundles/stories110m-smoke.esp`, not the recurrent/RWKV-style local-artifact path that previously reached `500+` to `900+ tok/s` without publishable quality.
- Bundle inspection:
  `swift run -c release esprun inspect .build/release-bundles/stories110m-smoke.esp` confirmed `behavior_class=exact`, `architecture_version=decoder-v1`, supported backends `["ane-private", "cpu-safe"]`, and `weights/metadata.json` confirmed the retained model is a 12-layer Stories/Llama-family bundle.
- Retained baseline reminder:
  `results/release-suite-stories-20260408-023106-published-r1i3/suite-summary.json` remains the qualified benchmark contract at `77.69 tok/s`, `1.72 ms` TTFT, `13.86 ms` median token, `15.50 ms` p95 token, correctness `ALL PASS`.
- Exact-head backend A/B on the real bundle:
  default `ANE_COMPILE_CACHE_POLICY=preferCached .build/release/espresso-generate generate --bundle .build/release-bundles/stories110m-smoke.esp --prompt 'Hello' --max-tokens 32 --compare-warmup 1 --compare-iterations 2 --benchmark-generate` produced `72.17 tok/s`, `13.72 ms` median token, `16.44 ms` p95, `exact_head_backend=ane_classifier`, with coherent output.
- Fused exact-head disable check:
  `ESPRESSO_DISABLE_LLAMA_HYBRID_FUSED_EXACT_HEAD=1` on the same command produced `72.98 tok/s`, `13.96 ms` median token, `15.75 ms` p95 with effectively unchanged output. Keep verdict: head-only fusion is not the bottleneck.
- CPU head fallback check:
  `ESPRESSO_FORCE_EXACT_HEAD_BACKEND=cpu_fp16_tiled` dropped to `63.69 tok/s`; `ESPRESSO_FORCE_EXACT_HEAD_BACKEND=cpu_partitioned_fp32` reached `70.53 tok/s`. Both diverged semantically from the retained ANE-head text. Kill verdict: CPU head fallback is not publishable for this bundle.
- Decode-proxy upper bound:
  `ANE_COMPILE_CACHE_POLICY=preferCached .build/release/espresso-bench --decode --layers 12 --warmup 3 --iterations 20 --decode-steps 16 --decode-max-seq 32 --output results/research-decode-proxy-default` measured `166.5 tok/s`, `6.007 ms/token`, with `ANE wall 5.365 ms (96.10%)`. This is useful as a topology proxy, not a publishable claim.
- Llama layer-input rebinding real-bundle check:
  `ANE_COMPILE_CACHE_POLICY=preferCached ESPRESSO_ENABLE_LLAMA_HYBRID_LAYER_INPUT_REBIND=1 .build/release/espresso-generate generate --bundle .build/release-bundles/stories110m-smoke.esp --prompt 'Hello' --max-tokens 32 --compare-warmup 1 --compare-iterations 2 --benchmark-generate` produced `77.67 tok/s`, `12.95 ms` median token, `13.90 ms` p95, and preserved the exact output text from the default ANE-head run.
- Llama layer-input rebinding suite validation:
  `ANE_COMPILE_CACHE_POLICY=preferCached ESPRESSO_ENABLE_LLAMA_HYBRID_LAYER_INPUT_REBIND=1 .build/release/espresso-generate suite --bundle .build/release-bundles/stories110m-smoke.esp --prompts scripts/stories_release_benchmark_prompts.txt --max-tokens 8 --runs 1 --compare-warmup 1 --compare-iterations 3 --coreml-model "$HOME/Library/Application Support/Espresso/demo/stories110m_coreml/stories110m_stateful_seq128.mlpackage" --coreml-seq-len 128 --coreml-compute-units cpu_only --no-power --output-dir results/research-suite-rebind-20260409-213025` produced `81.40 tok/s` on `hello`, `83.94 tok/s` on `hello_world`, `82.65 tok/s` on `fox`, aggregate median `82.65 tok/s`, `1.70 ms` TTFT, `13.62 ms` median token, `15.99 ms` p95, with token/text parity `ALL PASS`.
- Publishable-contract caution:
  the `--no-power` direct suite run above is informative for local exploration, but it is not the retained publishable harness contract. Behavior changes must clear the actual `autoresearch-espresso` / `--baseline-summary` path before they can be kept.
- Default-rollout validation attempt:
  after changing the policy locally to default-enable rebinding for Stories, the retained publishable harness rejected it twice. `results/autoresearch/rebind-default-release-20260409-213816` measured `69.97 tok/s`, `1.96 ms` TTFT, `16.59 ms` median token, `21.62 ms` p95. `results/autoresearch/rebind-candidate-release-20260409-214054` measured `62.31 tok/s`, `1.95 ms` TTFT, `18.00 ms` median token, `21.86 ms` p95. Both preserved correctness but failed the retained performance gates.
- Same-binary control:
  `ANE_COMPILE_CACHE_POLICY=preferCached ESPRESSO_DISABLE_HYBRID_LAYER_INPUT_REBIND=1 python3 autoresearch-espresso/experiment_runner.py benchmark --json --output-dir results/autoresearch/rebind-control-release-20260409-214004` reached `77.35 tok/s`, `1.92 ms` TTFT, `16.62 ms` median token, `17.69 ms` p95. That still failed the retained gates, but it materially outperformed the candidate default-on runs under the same contract, so the default policy flip was reverted.
- Immediate keep/kill view:
  keep investigating llama hybrid layer-input rebinding as the first real publishable-quality local optimization because it improved the retained suite without observed text drift in the earlier controlled run, but keep it env-gated until a fresh publishable rerun passes. Kill the idea that output-head swaps alone will unlock the next order-of-magnitude gain.
- Current working thesis:
  the `80` to `120 tok/s` publishable ceiling is primarily an exact hybrid llama decode topology problem, not just a private-runtime plumbing problem. A serious breakthrough likely requires an architectural discontinuity such as a recurrent-hybrid DeltaNet/Gated-DeltaNet style backbone with persistent recurrent state on ANE, rather than more verifier-side or head-only micro-optimizations.

## 750+ Tok/s Breakthrough Program 2026-04-09

### Target

- Achieve a publishable Espresso benchmark above `750 tok/s` on ANE.
- Preserve coherent, non-garbage output for at least `128` generated tokens.
- Retain an explicit benchmark protocol that can survive public scrutiny rather than relying on best-case local artifacts.

### Status

- [ ] Freeze the benchmark protocol for the future `750+ tok/s` claim before building the new serving path.
- [ ] Surface exact-lane hybrid decode timing telemetry in the current Stories lane so QKV / Metal / FFN / IO shares are measurable.
- [ ] Build a minimal ANE-native recurrent-hybrid micro-runtime with persistent recurrent state and no transformer verifier machinery.
- [ ] Prove that the recurrent micro-runtime has a large structural per-token advantage over the current exact Stories topology.
- [ ] Train or distill a tiny recurrent-hybrid Stories proof model that produces coherent 128-token output.
- [ ] Run a sparse-attention ablation ladder to find the minimum attention needed to recover publishable quality.
- [ ] Extend the `.esp` artifact/runtime contract for recurrent-hybrid state, recurrent weights, and hybrid attention checkpoints.
- [ ] Build a full recurrent-hybrid serving lane on direct ANE with no silent CPU/GPU fallback.
- [ ] Establish a broader publication suite in addition to the retained 3-prompt shipping gate.
- [ ] Reach `>= 750 tok/s` on the publication suite with repeated-run stability and quality passing.

### Program

- Phase 0: benchmark contract first.
  Lock the future claim to exact commit, exact `.esp` bundle, exact hardware model, exact macOS version, exact cache policy, exact power-telemetry setting, exact prompt suite, exact token budget, exact warm/cold designation, and exact correctness contract before optimizing.
  Keep two benchmark classes:
  a small strict shipping gate for the retained lane, and a broader publication suite for the future `750+ tok/s` claim.
- Phase 1: current-lane telemetry.
  Export `HybridDecodeTimingBreakdown` into benchmark artifacts so every exact-lane run reports QKV, Metal attention, FFN, and IO shares.
- Phase 2: recurrent economics proof.
  Build the smallest possible ANE-native recurrent-hybrid runtime, ideally `1-2` recurrent blocks plus a trivial head, to prove that persistent state actually removes the dominant per-token costs.
- Phase 3: proof model.
  Train or distill a tiny Stories-scale recurrent-hybrid checkpoint and verify that it can generate coherent `128`-token continuations before investing in a larger model.
- Phase 4: quality recovery ladder.
  Benchmark these variants in order:
  fully recurrent, recurrent plus periodic full attention, recurrent plus local/sliding-window attention.
  Keep the least expensive variant that restores acceptable quality.
- Phase 5: productize the path.
  Add recurrent-hybrid model support to `.esp`, implement the direct-ANE serving runtime, and make benchmark artifacts report recurrent-state traffic and timing shares.
- Phase 6: publication evidence.
  Keep the retained 3-prompt suite as the shipping gate, but add a broader publication suite and repeated-run stability reporting so the final claim is not based on a cherry-picked run.

### Kill Criteria

- Kill the architecture path early if the recurrent micro-runtime does not show a clear structural win over the current exact decode proxy.
- Kill or redesign the proof model if `128`-token output is visibly incoherent or collapses on hard prompts.
- Kill any hybrid design whose quality only recovers when attention becomes frequent enough that the decode economics look transformer-like again.
- Kill any candidate that reaches high tok/s only through a changed benchmark contract, fallback compute path, or warm-path cherry-picking.
- Do not claim success unless repeated publication-suite runs are stable, not just individually fast.

### Deliverables

- Benchmark protocol document:
  shipping gate versus publication suite, with exact thresholds and repetition counts.
  Publication-suite default target:
  at least `20` prompts, `128` generated tokens per prompt, `10` full repeated suite runs on the same host, median as headline, plus IQR or p5/p95.
- Telemetry artifacts:
  exact-lane QKV / Metal / FFN / IO timing shares and recurrent-state traffic stats.
- Proof runtime:
  a minimal recurrent-hybrid ANE decode lane with persistent state.
- Proof model:
  a tiny recurrent-hybrid Stories checkpoint with documented quality behavior at `128` tokens.
- Full serving artifact:
  `.esp` support plus a benchmarkable direct-ANE recurrent-hybrid runtime.
- Publication package:
  suite summaries, repeated-run distributions, baseline-comparison artifacts, and benchmark narrative.

### Immediate Next Steps

- [x] Implement benchmark-output support for `HybridDecodeTimingBreakdown`.
- [x] Define the publication-suite benchmark-class plumbing and persist its provenance in suite metadata/artifacts.
  Minimum bar:
  `20+` prompts, `128`-token generations, repeated full-suite runs, retained per-prompt text/token artifacts, and suite-level median plus spread reporting.
- [ ] Prototype the smallest recurrent-hybrid ANE block and measure its per-token latency against the current decode proxy.
- [ ] Decide the first proof architecture to train:
  Gated DeltaNet-inspired recurrent block with sparse periodic attention is the default recommendation.
- [ ] Refuse further investment in head-only or same-model speculative branches unless they unlock a concrete dependency for the recurrent-hybrid program.

### Review

- Implemented Phase 0 benchmark-contract scaffolding in `espresso-generate`:
  suite runs now accept and persist `shipping` versus `publication` benchmark class, and `metadata.json` records model, bundle path, cache policy, power mode/enabled state, Core ML compute envelope, host architecture, OS version, and exact benchmark command.
- Implemented Phase 1 exact-lane telemetry plumbing:
  `GenerationResult` now carries hybrid decode timings, `BackendRunMetrics` preserves them, `compare.json` exports both timing totals and normalized shares, stderr summaries print them, and prompt-suite summaries aggregate hybrid decode shares per prompt and across the suite.
- Defined the first concrete recurrent-hybrid target in `docs/platform/2026-04-09-recurrent-hybrid-target-spec.md`:
  `stories-rh-v1` is a `12`-block Stories-class student with a `9` recurrent / `3` attention-checkpoint schedule, bounded persistent recurrent state, explicit training ablations, and hard kill criteria tied to both quality and tok/s.
- Verification:
  `swift build -c release --product espresso-generate`
  `swift test --filter EspressoGenerateTests`
  `ANE_COMPILE_CACHE_POLICY=preferCached .build/release/espresso-generate suite --bundle .build/release-bundles/stories110m-smoke.esp --prompts scripts/stories_release_benchmark_prompts.txt --suite-kind publication --max-tokens 1 --runs 1 --compare-warmup 0 --compare-iterations 1 --coreml-model "$HOME/Library/Application Support/Espresso/demo/stories110m_coreml/stories110m_stateful_seq128.mlpackage" --coreml-seq-len 128 --coreml-compute-units cpu_only --no-power --output-dir results/phase01-publication-smoke`
- Smoke-suite outcome:
  `results/phase01-publication-smoke` recorded `suite kind: publication`, preserved correctness on all `3` prompts, and exported hybrid share summaries with aggregate median approximately `qkv 17.5%`, `metal 48.0%`, `ffn 30.0%`, `io 4.5%`.

## Autoresearch Harness Run 2026-04-09

### Status

- [x] Verify retained autoresearch inputs and release build health for the Stories serving lane.
- [x] Run the autoresearch suite baseline on the current tree and capture throughput / latency artifacts.
- [x] Run an ANE-timing-focused benchmark lane to estimate hardware-time share as a utilization proxy.
- [x] Summarize keep/kill guidance for the current tree and record the exact commands and artifacts.

### Review

- Verified retained inputs:
  `.build/release-bundles/stories110m-smoke.esp`,
  `scripts/stories_release_benchmark_prompts.txt`,
  `artifacts/benchmarks/release-serving-stories/latest.json`,
  and `$HOME/Library/Application Support/Espresso/demo/stories110m_coreml/stories110m_stateful_seq128.mlpackage`.
- Autoresearch harness run:
  `python3 autoresearch-espresso/experiment_runner.py benchmark --json --output-dir results/autoresearch/manual-20260409-081640-baseline`
  produced `results/autoresearch/manual-20260409-081640-baseline/suite-summary.json` and `baseline-comparison.json`.
- Current-tree suite result:
  aggregate `84.94 tok/s`, `1.795 ms` TTFT, `13.009 ms` median token, `16.944 ms` p95 token, token/text parity pass, correctness pass, performance fail.
- Keep/kill verdict for the current tree:
  kill. Aggregate throughput improved over the retained publishable baseline, but `baseline-comparison.json` shows the `fox` prompt regressed below gates: `72.10 tok/s` vs `83.58`, `18.41 ms` median token vs `13.38`, `20.10 ms` p95 vs `15.50`.
- Utilization proxy run:
  `swift run -c release espresso-bench --ane-only --decode --layers 6 --warmup 5 --iterations 30 --decode-steps 16 --decode-max-seq 32 --perf-stats --output results/ane-util-20260409-081746`
  yielded `322.84 tok/s` with timing breakdown `ANE wall 2.757 ms (96.53%)`, `IO 0.099 ms (3.47%)`, `CPU 0.000 ms`.
- Follow-up telemetry check:
  `ANE_EVAL_PATH=clientDirect swift run -c release espresso-bench --ane-only --decode --layers 6 --warmup 5 --iterations 30 --decode-steps 16 --decode-max-seq 32 --perf-stats --output results/ane-util-clientdirect-20260409-081829`
  still reported `ane_hw=0` and `host_overhead=2.815 ms`, so the bench currently exposes strong ANE-path dominance but not reliable hardware-execution-time counters on this host/path.

## Autoresearch Batch 2026-04-09

### Status

- [x] Run 10 sequential `autoresearch-espresso` full experiments on the current tree.
- [x] Aggregate the scored results from `autoresearch-espresso/suite-results.tsv`.
- [x] Summarize the batch outcome and identify repeated failing prompts or gates.

### Review

- Batch command:
  `for i in $(seq 1 10); do python3 autoresearch-espresso/experiment_runner.py full --description "batch 2026-04-09 experiment $i" --output-dir "results/autoresearch/batch-20260409-exp${i}-$(date +%Y%m%d-%H%M%S)"; done`
- Runs completed at:
  `results/autoresearch/batch-20260409-exp1-20260409-090257`
  through
  `results/autoresearch/batch-20260409-exp10-20260409-090629`.
- Aggregate batch outcome from the last 10 TSV rows:
  10 runs, 1 keep, 9 discards, all with token/text parity and correctness gates passing.
- Throughput spread:
  min `74.39 tok/s`, max `84.86 tok/s`, average `80.99 tok/s`.
- Best retained run:
  experiment 3 at `83.42 tok/s`, `1.802 ms` TTFT, `13.258 ms` median token, `14.572 ms` p95 token, `merge_recommended=YES`.
- Repeated failures from per-run `baseline-comparison.json` files:
  `fox` failed in 8/10 runs, mostly `espresso_tok_s` (7 fails), `espresso_ttft_ms` (5 fails), and `espresso_median_token_ms` (4 fails).
  `hello_world` failed in 6/10 runs, entirely on `espresso_ttft_ms`.
  `hello` failed only once.
- Conclusion:
  the current tree is not stably inside the retained shipping envelope. Aggregate `tok/s` is often above baseline, but prompt-level latency and throughput variance still make this a weak autoresearch candidate without prompt-specific stabilization work.

## Throughput Harness Redesign 2026-04-09

### Status

- [x] Replace the shipping-lane-first Python harness with a throughput-first dual-lane harness.
- [x] Add explicit ANE-share gating and heuristic coherence scoring.
- [x] Update the autoresearch program docs to match the new throughput contract.
- [x] Run focused Python verification for the rewritten harness.

### Review

- Replaced `autoresearch-espresso/experiment_runner.py` with a dual-lane harness:
  the real `espresso-generate suite` lane now feeds coherence scoring and real generation metrics, while `espresso-bench --decode` is the primary throughput/utilization lane.
- The new harness logs to `autoresearch-espresso/throughput-results.tsv` and writes a merged `harness-summary.json` artifact per run.
- Quality is now heuristic, not exact-parity-gated. Current checks enforce minimum generated token count, minimum unique-token ratio, maximum repeated-bigram ratio, maximum identical-token run ratio, printable output, and minimum word count.
- Utilization is explicit:
  decode bench `ane_share_ratio = ane / (ane + io + cpu)` and defaults to a hard `85%` gate.
- Verification:
  `python3 -m py_compile autoresearch-espresso/experiment_runner.py autoresearch-espresso/test_experiment_runner.py` passed.
  `python3 -m unittest discover -s autoresearch-espresso -p 'test_*.py'` passed.
  `python3 autoresearch-espresso/experiment_runner.py --help` passed.
  `python3 autoresearch-espresso/experiment_runner.py benchmark --help` passed.
  `python3 autoresearch-espresso/experiment_runner.py benchmark --json --output-dir results/autoresearch/throughput-smoke-20260409-092050 --max-tokens 4 --runs 1 --warmup 0 --iterations 1 --bench-warmup 0 --bench-iterations 2 --bench-decode-steps 4 --bench-decode-max-seq 32 --min-generated-tokens 2 --timeout 900` completed end to end, wrote `harness-summary.json`, measured `253.99 decode tok/s` with `96.70%` ANE share, and failed only the coherence heuristic for the short `fox` smoke completion (`word_count<2`).
  `python3 autoresearch-espresso/experiment_runner.py full --description "throughput harness baseline" --output-dir results/autoresearch/throughput-baseline-20260409-092535` completed end to end, logged to `autoresearch-espresso/throughput-results.tsv`, and produced a retained baseline keep with `305.10 decode tok/s`, `96.39%` ANE share, and coherence passing on all 3 prompts.

## Publishable Harness Refocus 2026-04-09

### Status

- [x] Replace the throughput-first Python harness with a publishable-benchmark-only harness.
- [x] Remove synthetic decode-lane scoring from the harness contract.
- [x] Update the autoresearch program docs to match the publishable benchmark contract.
- [ ] Run focused verification for the publishable harness.

### Review

- Replaced `autoresearch-espresso/experiment_runner.py` with a publishable-benchmark-only harness. The only scoring lane is now the retained `espresso-generate suite` contract used by the release-serving benchmark.
- Results now log to `autoresearch-espresso/publishable-results.tsv` with `primary_metric = suite median tok/s`.
- Verification:
  `python3 -m py_compile autoresearch-espresso/experiment_runner.py autoresearch-espresso/test_experiment_runner.py` passed.
  `python3 -m unittest discover -s autoresearch-espresso -p 'test_*.py'` passed.
  `python3 autoresearch-espresso/experiment_runner.py --help` passed.
  `python3 autoresearch-espresso/experiment_runner.py benchmark --help` passed.
  `python3 autoresearch-espresso/experiment_runner.py benchmark --json --output-dir results/autoresearch/publishable-smoke-20260409-093149` completed end to end, wrote `suite-summary.json` plus `baseline-comparison.json`, and correctly returned `status=gates_failed` for a real same-tree regression case: `83.83 tok/s`, `1.743 ms` TTFT, `13.052 ms` median token, `14.696 ms` p95 token, token/text parity pass, correctness pass, performance fail.

## Publishable Batch 2026-04-09

### Status

- [x] Run 20 sequential `autoresearch-espresso` full experiments on the current tree under the publishable contract.
- [x] Aggregate the scored results from `autoresearch-espresso/publishable-results.tsv`.
- [x] Summarize repeated prompt-level failures from the batch `baseline-comparison.json` artifacts.

### Review

- Batch command:
  `for i in $(seq 1 20); do python3 autoresearch-espresso/experiment_runner.py full --description "publishable batch 2026-04-09 experiment $i" --output-dir "results/autoresearch/publishable-batch-20260409-exp${i}-$(date +%Y%m%d-%H%M%S)"; done`
- Runs completed at:
  `results/autoresearch/publishable-batch-20260409-exp1-20260409-093426`
  through
  `results/autoresearch/publishable-batch-20260409-exp20-20260409-094157`.
- Aggregate batch outcome from the last 20 TSV rows:
  20 runs, 1 keep, 19 discards, all with token/text parity and correctness gates passing.
- Publishable throughput spread:
  min `73.56 tok/s`, max `88.68 tok/s`, average `80.91 tok/s`.
- Best retained run:
  experiment 11 at `84.18 tok/s`, `1.571 ms` TTFT, `13.102 ms` median token, `14.365 ms` p95 token, `merge_recommended=YES`.
- Repeated failures from per-run `baseline-comparison.json` files:
  `fox` failed in 18/20 runs, mostly `espresso_tok_s` (16 fails), `espresso_median_token_ms` (14 fails), and `espresso_ttft_ms` (9 fails).
  `hello_world` failed in 11/20 runs, almost entirely on `espresso_ttft_ms` (11 fails).
  `hello` failed only 3 times.
- Conclusion:
  the current tree is not publishable-stable. Aggregate suite `tok/s` often exceeds the retained claim, but the branch rarely clears the per-prompt publishable gates. `fox` is the primary blocker and `hello_world` TTFT is the secondary blocker.

## Autoresearch Harness

### Status

- [x] Replace the repo-local `autoresearch-espresso` runner's single-prompt benchmark path with the hardened suite benchmark contract.
- [x] Replace heuristic generation-coherence checks with retained baseline summary + token/text parity gates.
- [x] Update `autoresearch-espresso/program.md` so the allowed workflow, metrics, and keep/discard rules match the hardened harness.
- [x] Add focused tests for the Python harness parsing/scoring logic and run verification.

### Review

- `autoresearch-espresso/experiment_runner.py` is now inference-first: it defaults to the retained Stories `.esp` bundle, retained Stories prompt suite, local qualified Core ML compare package, and the retained release `suite-summary.json` baseline instead of the old `/tmp/stories110m.esp` single-prompt heuristic path.
- The runner now uses the hardened `espresso-generate suite` contract as the source of truth for tok/s work, parses `suite-summary.json` and optional `baseline-comparison.json`, and reports suite median tok/s, TTFT, median token latency, p95 token latency, token/text parity, and baseline-gate verdicts.
- The old ad hoc quality path based on empty output, repetition, and substring heuristics is gone. Quality is now the same token/text parity and correctness/performance-gate contract used by the retained serving benchmark lane.
- `full` runs now log to `autoresearch-espresso/suite-results.tsv`, track the best retained keep-run tok/s, and only suggest `keep` when correctness gates pass, performance gates pass, and throughput beats the previous kept best.
- `autoresearch-espresso/program.md` now matches the stronger harness: setup points at the release bundle and retained baseline, the loop uses `python autoresearch-espresso/experiment_runner.py full --description "..."`, and keep/discard decisions are based on suite verdicts rather than hand-inspected generations.
- Verification:
  `python3 -m py_compile autoresearch-espresso/experiment_runner.py` passed.
  `python3 -m unittest discover -s autoresearch-espresso -p 'test_*.py'` passed.
  `python3 autoresearch-espresso/experiment_runner.py --help` passed.
  `python3 autoresearch-espresso/experiment_runner.py benchmark --help` passed.
  a direct Python smoke parse against `results/release-suite-stories-20260408-023106-published-r1i3/suite-summary.json` returned the retained `77.69 tok/s`, `1.72 ms` TTFT, `13.864 ms` median token, `15.50 ms` p95 token, with correctness/performance gates both true.
  `python3 autoresearch-espresso/experiment_runner.py benchmark --json --output-dir /tmp/espresso-autoresearch-live-fixed` completed end to end against the real release-serving suite, wrote `/tmp/espresso-autoresearch-live-fixed/suite-summary.json` plus `/tmp/espresso-autoresearch-live-fixed/baseline-comparison.json`, and correctly surfaced a real same-branch regression (`84.29 tok/s`, `1.48 ms` TTFT, `12.62 ms` median token, `55.98 ms` p95 token, correctness pass, performance fail) as `status=gates_failed`.

## Zig Measurement

### Status

- [x] Add explicit process-scoped Zig runtime disable support so fallback-vs-Zig runs can be compared honestly in separate benchmark invocations.
- [x] Add per-op Zig/fallback dispatch counters at the `surface_io.c` backend branch points.
- [x] Add a focused `espresso-bench` Zig interop microbenchmark mode that reports runtime state, per-op counters, and latency/throughput JSON.
- [x] Add a wrapper script that runs the microbenchmark in Zig-off and Zig-on modes and writes a comparison artifact.
- [x] Verify the new counters, fallback microbenchmark path, and wrapper script locally.

### Review

- `ANEInterop` now exposes `ANEInteropZigDispatchCounters`, reset/snapshot APIs, and an explicit `ESPRESSO_DISABLE_ZIG_RUNTIME=1` kill-switch so the optional Zig runtime can be benchmarked against the stable fallback path without changing code.
- `surface_io.c` now records exact Zig-vs-fallback call counts, element counts, and logical byte counts for contiguous copy, strided copy, scatter, gather, argmax, f16/f32 conversion, and zeroing at the actual branch points where backend choice happens.
- `espresso-bench --zig-interop-microbench` now emits JSON with runtime status, per-operation backend counters, median/mean ns per call, and throughput-style `median_gbps` for the current process mode.
- `scripts/benchmark_zig_runtime.sh` now builds the Zig dylib, runs the microbenchmark once with `ESPRESSO_DISABLE_ZIG_RUNTIME=1` and once with `ESPRESSO_ZIG_RUNTIME_DYLIB=...`, then writes `zig-off.json`, `zig-on.json`, and `comparison.json`.
- The representative microbenchmark run at the default shape (`channels=768`, `spatial=128`, `inner_iterations=128`) produced mostly neutral-to-negative results for Zig on this M3 Max host. Example ratios from `results/zig-interop-representative/comparison.json`: contiguous FP16 copy `0.75x`, contiguous FP32->FP16 write `0.97x`, contiguous FP16->FP32 read `1.04x`, zeroing `1.00x`.
- A saved end-to-end retained-lane comparison on the Stories bundle (`results/zig-e2e-stories/comparison.json`) also came back negative for Zig in this run: fallback `79.80 tok/s`, `1.50 ms` TTFT, `14.30 ms` median token, `15.68 ms` p95 token; Zig `75.57 tok/s`, `1.67 ms` TTFT, `14.40 ms` median token, `16.11 ms` p95 token. That is `0.947x` throughput, `1.11x` TTFT, `1.007x` median-token, and `1.028x` p95-token relative to fallback.
- Verification:
  `swift test --filter ANEInteropTests` passed.
  `swift test --filter DecodeChainingInteropTests/test_zig_interop_microbench_reports_fallback_when_runtime_is_disabled` passed.
  `OUTPUT_DIR="$PWD/results/zig-interop-smoke" CHANNELS=16 SPATIAL=16 WARMUP=0 ITERATIONS=2 INNER_ITERATIONS=8 ./scripts/benchmark_zig_runtime.sh` passed and emitted comparison artifacts.
  `OUTPUT_DIR="$PWD/results/zig-interop-representative" ./scripts/benchmark_zig_runtime.sh` passed and emitted representative-shape comparison artifacts.
  `ANE_COMPILE_CACHE_POLICY=preferCached` end-to-end Stories A/B passed and emitted `results/zig-e2e-stories/zig-off.json`, `results/zig-e2e-stories/zig-on.json`, and `results/zig-e2e-stories/comparison.json`.

## Road To 90

### Status

- [x] Harden public codegen/runtime failure paths so invalid caller state throws instead of aborting.
- [x] Bound shared IOSurface pooling with deterministic eviction and observable stats.
- [x] Replace direct `ProcessInfo.processInfo.environment` reads in `RealModelInferenceEngine` with a single execution policy object plumbed from the CLI layer.
- [x] Remove the stale sorted-output graph contract and align tests/docs with preserved declared order.
- [x] Run focused regression suites for codegen, runtime pooling, inference, and CLI policy plumbing.

### Review

- `ANECodegen.emit` now throws `ANECodegenError` instead of trapping on invalid graphs, missing attrs, missing outputs, and banned ops; downstream callers and tests were updated to use the throwing API.
- `ANESurfacePool` now enforces per-bucket and total-byte caps, exposes `Stats`, and has regression coverage for bucket retention, pool reuse, and eviction behavior.
- `RealModelInferenceEngine` now carries a `RealModelExecutionPolicy`, uses throwing asset accessors instead of crash-oriented computed properties, and receives policy/env data from `EspressoGenerate` through a dedicated CLI resolver.
- The graph IR/docs now preserve declared output order instead of advertising a dead alphabetical-sorting contract, and the codegen tests lock that behavior in.
- Review follow-up: batched Metal attention now requires cached bindings before taking the batched path, so hybrid and exact-runtime decode fall back to the existing non-batched path when cached bindings are unavailable.
- Review follow-up: the Zig runtime build is now opt-in through `ESPRESSO_ENABLE_ZIG_RUNTIME_PLUGIN=1`, and `scripts/build_zig_runtime.sh` still soft-skips when Zig is absent so default `swift build` and `swift test` runs do not depend on a local Zig toolchain.
- Review follow-up: the Zig runtime loader no longer searches from the current working directory, so unrelated `.build/zig-runtime` artifacts cannot be picked up ahead of the intended runtime path.
- Review follow-up: the new public `SurfaceIO` FP32 helpers now throw typed `SurfaceIOError` cases for invalid element counts and missing buffer base addresses instead of aborting the host process.
- Review follow-up: the remaining ANE hardware-test gate and exact CPU llama helpers now take explicit environment data on the engine path, while `RealModelExecutionPolicy.empty` provides a deterministic no-ambient-environment baseline for internal defaults and tests.
- Verification:
  `swift test --filter ANECodegenTests` passed.
  `swift test --filter ANERuntimeTests` passed.
  `swift test --filter EspressoGenerateTests` passed.
  `swift test --filter RealModelInferenceTests` passed.
  `swift test --filter ANEInteropTests` passed.

## Blog Draft

### Status

- [x] Gather the current publishable benchmark facts and framing for a long-form blog draft.
- [x] Write and save a ~2000 word blog post under `~/Documents/blog_draft`.
- [x] Verify the saved draft path and content.

### Review

- Retained this session:
  the technical blog post for Swift developers should anchor on the qualified Stories release benchmark, not the older 926 tok/s microbenchmark, and should explain benchmark contract design, ANE runtime structure, `IOSurface` ownership, and Core ML baseline reproducibility as first-class technical topics.
- Verification:
  saved draft at `/Users/chriskarani/Documents/blog_draft/espresso-qualified-stories-benchmark.md`; `wc -w` reported `1985` words; content and directory were verified locally.

## Publishable Release Benchmark

### Status

- [x] Run a sustained multi-token release benchmark on the qualified `.esp` serving lane.
- [x] Validate whether the local stateful Stories Core ML baseline is stable under `.cpuAndNeuralEngine`; fall back only if that path is not reproducible.
- [x] Update the published benchmark surfaces so they reflect the sustained release result instead of the one-token smoke qualification number.
- [x] Re-run focused verification for the benchmark/publication surfaces after the updates land.

### Review

- Retained this session:
  the one-token production qualification result was coherent but not publishable as a sustained serving claim. The new retained public benchmark is a multi-token `.esp` serving-lane suite on the qualified Stories bundle.
- Retained this session:
  the local stateful `stories110m_stateful_seq128.mlpackage` did not reproducibly support `.cpuAndNeuralEngine` here; Core ML failed to build an execution plan with code `-14`, so the publishable benchmark explicitly uses the qualified `cpu_only` baseline instead of implying a broader Core ML comparison contract.
- Retained this session:
  the publishable Stories release benchmark contract is now `3` prompts, `8` generated tokens, `1` warmup, `3` measured iterations, `seq128` Core ML baseline, and it passes token/text parity across the full suite.
- Verification:
  `swift run espresso-generate suite --bundle .build/release-bundles/stories110m-smoke.esp --prompts scripts/stories_release_benchmark_prompts.txt --max-tokens 8 --runs 1 --compare-warmup 1 --compare-iterations 3 --coreml-seq-len 128 --coreml-compute-units cpu_only --output-dir results/release-suite-stories-20260408-023106-published-r1i3` passed and emitted `results/release-suite-stories-20260408-023106-published-r1i3`; the retained aggregate was `77.69 tok/s`, `1.72 ms` TTFT, `13.86 ms` median token, `15.50 ms` p95 token, `73.42 tok/s` Core ML median, correctness `ALL PASS`.

## Hardware Qualification Follow-up

### Status

- [x] Reproduce the remaining hardware-gated failures on the reference Apple Silicon host and identify whether they are runtime defects or qualification-surface defects.
- [x] Fix the ANE interop eval-report cleanup/forced-failure regression so the hardware eval-report smoke tests become stable again.
- [x] Replace the broad `ANE_HARDWARE_TESTS` production gate with a curated retained-lane hardware correctness subset that runs in fresh `swift test` invocations.
- [x] Re-run the curated hardware gate on the reference Apple Silicon host and capture the final result.
- [x] Re-run release qualification with an actual `.esp` bundle and retained suite baseline on the target release machine.

### Review

- Retained this session:
  the broad hardware production gate was overreaching into benchmark/probe code paths and exhausting the ANE compile budget inside a single `swift test` process, which made the supposed release gate non-repeatable.
- Retained this session:
  the production qualification script now targets only the retained shipping lane and core runtime correctness slice, while research/realtime/virtual-client/sweep tests remain available as manual hardware surfaces rather than release blockers.
- Retained this session:
  `ANERuntimeTests.test_eval_report_records_successful_eval` now skips when the host-level ANE identity eval is unstable instead of falsely presenting that host instability as an eval-report contract regression.
- Retained this session:
  the release qualification path now supports the local stateful Stories Core ML baseline end to end. `GPT2DemoSupport.swift` detects the two-input `input_ids + cache_position` Core ML contract, uses synchronous stateful prediction, and reuses Core ML state correctly across prefill/decode.
- Retained this session:
  legacy retained `suite-summary.json` baselines without the newer Espresso latency fields now decode cleanly. Missing latency metrics are treated as unavailable instead of breaking the suite loader or fabricating regression failures.
- Retained this session:
  `scripts/production_qualification.sh` now forwards a release-suite Core ML compute-unit setting and defaults that baseline path to `cpu_only`, matching the verified local Stories stateful Core ML package.
- Verification:
  `swift build` passed; `swift test --filter EspressoGenerateTests` passed with the new stateful Core ML and legacy-baseline coverage; direct release suite passed with `ESPRESSO_COREML_MODEL=$HOME/Library/Application Support/Espresso/demo/stories110m_coreml/stories110m_stateful_seq128.mlpackage` and emitted `.build/production-qualification/release-suite-direct-cpuonly`; final end-to-end `scripts/production_qualification.sh --bundle .build/release-bundles/stories110m-smoke.esp --baseline-summary results/autoresearch/suite-smoke-20260316-212851/suite-summary.json --prompts .build/release-bundles/smoke-prompts.txt --max-tokens 1 --runs 1 --compare-warmup 0 --compare-iterations 1 --skip-cross-validation` passed and emitted `qualification_dir=/Users/chriskarani/CodingProjects/Espresso/.build/production-qualification/20260408-021249`.

## Status

- [x] Replace the local `../Edgerunner` path dependency with a pinned GitHub `EdgeRunner` dependency so builds are reproducible outside the author workstation.
- [x] Make the README truthful about the supported macOS floor, dependency count, and controlled-distribution scope.
- [x] Add a one-command production qualification script that runs build, tests, optional hardware/parity suites, and the exact-serving release suite against a retained baseline summary.
- [x] Add production qualification documentation for the retained exact serving lane and controlled macOS deployment policy.
- [x] Re-run verification against the pinned remote dependency path and qualify the new script in non-hardware mode.
- [x] Move CI off the hand-picked unit-test subset and onto the non-hardware production qualification entrypoint so the reproducible dependency path stays enforced on every PR.

## Review

- Retained this session:
  Espresso now resolves `EdgeRunner` from the pinned GitHub repository instead of a workstation-local `../Edgerunner` path, and the package graph no longer depends on the broken full `EdgeRunner` product that is missing the ignored `BonsaiLanguageModel.swift` source in the remote repo.
- Retained this session:
  the Qwen GGUF regression lane now runs only when `ESPRESSO_QWEN_0_6B_REGRESSION=1` is set and the local model assets exist, so the default package test surface stays reproducible instead of depending on unpublished local weights.
- Retained this session:
  `scripts/production_qualification.sh` is now the one-command non-hardware qualification gate for controlled deployment, and `.github/workflows/ci.yml` runs that same entrypoint on pull requests instead of a manually curated test subset.
- Verification:
  `swift build` passed with the pinned GitHub `EdgeRunner` dependency; `swift test --filter "QwenGGUFRegression|EspressoGenerateTests"` passed; `scripts/production_qualification.sh --skip-hardware --skip-cross-validation --skip-release-suite` passed and emitted `qualification_dir=/Users/chriskarani/CodingProjects/Espresso/.build/production-qualification/20260408-011843`.

# Fresh Performance Program

## Status

- [x] Add baseline-aware exact-serving suite evaluation so `espresso-generate suite` can compare against a retained same-lane summary instead of relying on a manual ledger read.
- [x] Add explicit throughput / TTFT / median-token / p95-token regression gates for suite summaries, with failing exit status when the retained serving lane slips.
- [x] Emit suite baseline-comparison artifacts and stderr summaries so retained exact-path benchmark decisions are reproducible in-tree.
- [x] Add focused `EspressoGenerateTests` coverage for suite baseline parsing, regression detection, and verdict composition.
- [x] Re-run focused and full verification after the new performance-program gates land.

## Review

- Retained this session:
  `espresso-generate suite` can now load a retained `suite-summary.json` baseline and evaluate the current exact serving lane against it instead of relying on the handwritten benchmark ledger.
- Retained this session:
  prompt-suite summaries now carry Espresso TTFT, median-token, and p95-token medians alongside tok/s, so the serving lane has explicit latency gates rather than a throughput-only view.
- Retained this session:
  suite runs now emit a `baseline-comparison.json` artifact plus console summaries, and the command returns a failing exit status when correctness or performance gates regress.
- Retained this session:
  focused `EspressoGenerateTests` now cover suite baseline flag parsing, latency-bearing summary aggregation, throughput/latency regression detection, and prompt-set mismatch handling.
- Verification:
  `swift build` passed; `swift test --filter EspressoGenerateTests` passed; `swift test` passed with `372 tests in 5 suites` and no failures in this environment.

# Stories Convert -> Optimize

## Status

- [x] Land a structured ANE eval-report boundary with wall-time, ANE hardware time, and host-overhead telemetry.
- [x] Thread eval telemetry through direct, inference, and decode benchmark plumbing, including fused decode variants.
- [x] Add focused ANERuntime coverage for eval-report success/failure behavior.
- [x] Add an initial Zig runtime scaffold and build script for the new serving-core ABI.
- [x] Integrate the Zig serving-runtime build into SwiftPM and add a dylib loader boundary under `ANEInterop`.
- [x] Unify low-level FP32/zero/fill surface IO through shared `SurfaceIO`/interop paths and remove duplicated `RealModelInference` surface helpers.
- [x] Add ANE surface pooling plus a pooled direct-generation serving lane API for reusing compiled models safely across requests.
- [x] Eliminate decode/session scratch-array initialization on the serving path by zeroing/filling reusable surfaces directly.
- [x] Add explicit retained-surface ownership for the serving-path handle wrappers instead of leaking raw `IOSurfaceRef`s.
- [x] Remove repeated retained-surface fetches from forward hot loops and centralize standard transformer decode into an explicit session object.
- [x] Re-run full package verification after the runtime-core migration slice and keep the tree green.
- [x] Confirm current baseline on the real Stories release benchmark.
- [x] Add repo-local task tracking and keep this file current.
- [x] Expand the `.esp` manifest/runtime contract for model tier, behavior class, context target, and optimization lineage.
- [x] Replace bundle benchmark synthetic token latencies with measured per-token latencies.
- [x] Add explicit context-target Stories SKU packing and runtime resolution support.
- [x] Package and verify a first-class `stories110m-ctx256` `.esp` artifact.
- [x] Add GQA/MQA Stories variant schema/runtime compatibility coverage.
- [x] Add a reproducible distilled Stories-native pipeline with export/eval metadata.
- [x] Add `.esp` manifest sections for output-head and draft metadata with exact/near-exact/approximate labeling.
- [x] Validate factored-head and draft sidecar file references during bundle/runtime open.
- [x] Expose selected output-head and draft features through runtime resolution.
- [x] Add CLI/exporter support for output-head and draft manifest metadata.
- [x] Run verification builds/tests for the output-head/draft contract slice and a real bundle smoke run.
- [x] Add real model/export/runtime execution for a cheaper factored Stories head and benchmark it on the Stories release path.
- [x] Reject retaining the factored Stories head as the default serving path after the real Stories benchmark/quality gate failed to justify it.
- [x] Produce the first stable Stories student artifact with the distillation pipeline and verify short/long prompt quality against the retained exact bundle.
- [x] Export an exact two-token Stories draft artifact from the stable student and package it through `.esp` draft metadata.
- [x] Add real bundle/runtime execution for exact two-token Stories decoding with acceptance accounting and benchmark it on the Stories release path.
- [x] Reject retaining the original exact two-token Stories draft path as the default serving path after it failed to beat the retained exact bundle on the real release benchmark.
- [x] Build and benchmark smaller teacher-truncated Stories student SKUs as the next candidate standalone or draft proposer artifacts.
- [x] Close the intermediate teacher-truncated Stories student sweep after the measured proposer search failed to produce a retained winner.
- [x] Retire teacher-truncated proposer pipeline support from the product lane once measured search stopped yielding useful retained candidates.
- [x] Benchmark a 4-layer teacher-truncated Stories student on short, long, and real release benchmark paths.
- [x] Train a real distilled 6-8 layer Stories student and record honest acceptance-quality metrics against the teacher.
- [x] Replace the CPU-only exact draft proof path with a llama hybrid ANE-backed draft/verifier path on the exact-two-token bundle lane.
- [x] Reject retaining the distilled draft student as a default serving proposer after acceptance/throughput gates failed on the real Stories benchmark.
- [x] Benchmark llama hybrid layer-input rebinding on the retained exact Stories path and retain it only if the same-binary real benchmark improves without output drift.
- [x] Inspect the ANE-backed recurrent exact-two-token branch-state-promotion path and determine the minimum reusable state-promotion primitive for llama serving.
- [x] Prototype a lower-overhead verifier-side proposer path that avoids compiling or stepping a second full llama runtime between token-0 verify and token-1 commit.
- [x] Add focused tests for any new llama state-promotion or prepared-branch commit primitive before benchmarking.
- [x] Benchmark one reduced-overhead verifier/proposer variant against the retained exact Stories control before attempting any broader sweep.
- [x] Try a verifier-side proposer-head experiment that reuses the committed llama hybrid activation and existing head machinery, and retain it only if the same-binary real Stories benchmark improves.
- [x] If the proposer-head path does not break through, try a delta-reuse mutable-proposer experiment that preserves ANE-centric verifier execution and benchmark it on the same real Stories command.
- [x] Prototype a fused llama exact ANE RMSNorm+classifier head on the retained hybrid path so per-token head dispatch drops from two kernels to one.
- [x] Add focused tests for the fused exact-head gate/default behavior before benchmarking.
- [x] Benchmark the fused exact-head path against the retained exact Stories control and keep it only if short/long prompt output stays exact and real throughput improves.
- [x] Add a bounded ANE head-spatial override experiment for the retained fused llama exact head and benchmark a smaller logical head shape against the Stories baseline.
- [x] Keep a head-spatial override only if the real Stories benchmark improves without output drift or ANE-path regressions.
- [x] Prototype a single-runtime llama hybrid exact prepared-pair path that commits 1-2 exact tokens without compiling a second runtime, then benchmark it against the retained exact control.
- [x] Reintroduce a bounded llama split-runtime speculative prototype on the donor-delta retained baseline and benchmark split-8 against the same-binary exact control before any wider sweep.
- [x] Adapt the `.esp` exact-two-token draft resolver so llama draft sidecars run on the hybrid ANE-backed decode path instead of `CPUExactLlamaRuntime`.
- [x] Reuse the retained llama hybrid decode kernels for draft-sidecar prefill, propose, rollback, and verifier-controlled exact commit without CPU draft fallback.
- [x] Add focused runtime coverage for exact-two-token bundle resolution, checkpoint rollback semantics, and acceptance-accounting regressions on the bundle path.
- [x] Package and benchmark the trained `6`-layer and `8`-layer Stories draft sidecars on the real Stories release benchmark, one candidate at a time.
- [x] Reject retaining a hybrid draft-sidecar as the default serving path until it beats the exact non-spec control on a same-binary benchmark without output drift.
- [x] Port the cached split-runtime speculative path from GPT-2-only to llama on the ANE/hybrid decode path, benchmark it, and revert it when the exact control stays faster.
- [x] Preserve llama hybrid exactness on the retained speculative/runtime experiments through verifier-controlled token commitment, rollback, and per-pass acceptance accounting.
- [x] Keep the llama split-runtime speculative experiments ANE-centric by preserving the hybrid cached-binding / RoPE path and avoiding CPU draft fallback, then retire the lane when it still loses.
- [x] Benchmark exact llama split-layer speculation at draft split points 4, 6, 8, and 9 on the real Stories release benchmark.
- [x] Reject retaining the exact llama split-runtime speculative lane after no tested split point beat the retained exact baseline while preserving output.
- [x] Revert the llama speculative runtime lane immediately if all exact split points are neutral, slower, or reduce ANE-centric execution evidence.
- [x] Add bundle/runtime support for a llama hybrid exact-two-token future-sidecar artifact that reuses the retained fused ANE verifier head without compiling a second trunk runtime.
- [x] Add focused tests for future-sidecar artifact resolution and the new llama hybrid gate/default behavior before benchmarking.
- [x] Package a real Stories future-sidecar `.esp` artifact and verify exact short/long prompt output on the retained ANE-backed path.
- [x] Retain the unified future-head / future-sidecar path only if the real Stories release benchmark beats the retained exact baseline without CPU fallback or ANE-path regression.
- [x] Re-establish a clean same-revision exact Stories release control before each future-head benchmark and use it to close the future-head line when the rebuilt control still wins.
- [x] Train and package the committed-activation Stories future-head / `FTS2` artifacts, then close the line after real-runtime acceptance stayed at zero.
- [x] Extend llama hybrid exact-two-token draft resolution so `draft.artifact_ref` can target bundle-contained future-sidecar artifacts without introducing a second trunk runtime.
- [x] Compile and cache the ANE future proposer head rebound to the retained verifier activation surface while preserving `exact_head_backend=ane_classifier`, then retire it after it still failed the benchmark gate.
- [x] Keep future-token commitment verifier-preserved with per-pass `committed_exact_tokens_per_pass` and `accepted_future_tokens_per_pass` accounting on the retained future-head path.
- [x] Add focused tests for future-sidecar resolution, verifier-preserved accept/reject behavior, EOS handling, and one-token budget handling.
- [x] Benchmark the trained committed-activation future-head path on the real Stories release command and reject it after future acceptance remained zero and wall-clock throughput regressed against the exact control.
- [x] If the trained future-head path still loses, revert it and try the unified multi-head verifier design that shares the live exact head input work.
- [x] Reject retaining the unified multi-head verifier path after it preserved ANE-centric execution but still failed to win the real Stories release benchmark.
- [x] Extend the Stories distillation pipeline so future-head training can consume a materially larger local corpus source than `scripts/stories_prompt_suite.txt`.
- [x] Add an offline self-generated Stories corpus option driven by the teacher bundle so proposer training can cover more committed activations without touching the retained runtime.
- [x] Train a larger-corpus future-head artifact and require a materially better offline future-token match rate before reintroducing any runtime future-head lane.
- [x] Re-introduce a future-head runtime lane only after the larger-corpus artifact clears the offline gate, then rerun the real Stories release benchmark.

## Baseline

- Date: 2026-03-26
- Command:
  `./.build/arm64-apple-macosx/release/espresso-generate generate --bundle /tmp/stories110m-ctx256-v1.esp --max-tokens 64 --benchmark-generate --compare-warmup 1 --compare-iterations 3 Hello`
- Result:
  `tok_per_s=121.31`, `median_token_ms=8.45`, `p95_token_ms=10.27`, `first_token_ms=1.64`, `compile_ms=3645.10`, `exact_head_backend=ane_classifier`, `cached_bindings_enabled=true`
- Quality note:
  exact retained Stories ctx256 release path after enabling hybrid donor-delta by default for Stories. Benchmark runs were thermally noisy, so retain/reject decisions below use nearby exact controls and same-binary comparisons where available.

## Benchmark Ledger

| Change | Command | Before | After | Quality | Decision |
| --- | --- | --- | --- | --- | --- |
| Baseline | `espresso-generate ... /tmp/stories110m.esp ... Hello` | n/a | `102.51 tok/s` | exact retained path | keep |
| Measured bundle latencies | `swift test --filter 'ESPBundleTests|ESPRuntimeTests|ESPConvertTests|EspressoGenerateTests'` + release rebuild | synthetic bundle `median/p95` | bundle path now reports measured token latencies from `GenerationResult` | exact metric fix, no decode-path change | keep |
| `stories110m-ctx256` bundle | `espresso-generate ... /tmp/stories110m-ctx256-v1.esp ... Hello` | `/tmp/stories110m.esp`: `98.36 tok/s`, `compile_ms=1188.32` | `104.01 tok/s`, `compile_ms=986.73` | exact; short-prompt text matched baseline | keep |
| `stories110m-ctx256` long prompt | `espresso-generate generate --bundle /tmp/stories110m-ctx256-v1.esp --max-tokens 32 --prompt 'In a distant future ...'` | baseline text matched, `106.02 tok/s` single run | text matched, `65.43 tok/s` single run | exact text match; single-run latency noisy, not used as retain gate | informational |
| Distill proof artifact | `python3 scripts/distill_stories_native.py --config configs/stories/distill-proof.json --dry-run` then `espresso-generate generate --bundle /tmp/stories110m-distill-proof.esp --max-tokens 8 Hello` | no pipeline | `.esp` proof artifact exported and ran, but `compile_ms=26778.79`, retries/failures high, text approximate/garbled | approximate proof-only, not a retained serving path | keep pipeline, reject artifact as product path |
| GQA proof artifact | `python3 scripts/distill_stories_native.py --config configs/stories/gqa4-proof.json --dry-run` then `espresso-generate generate --bundle /tmp/stories110m-gqa4-proof.esp --max-tokens 4 Hello` | no runnable Stories/GQA artifact | `.esp` GQA proof artifact exported and ran, but `compile_ms=26621.03`, retries/failures high, text approximate/garbled | approximate proof-only, validates contract/runtime compatibility, not a retained serving path | keep support, reject artifact as product path |
| Output-head/draft contract slice | `swift test --filter 'ESPBundleTests|ESPRuntimeTests|ESPConvertTests'` + release `espc`/`esprun` rebuild + `esprun`/`espresso-generate` smoke on `/tmp/stories110m-contract-proof.esp` | no bundle contract for factored-head or draft metadata | manifest/runtime/CLI support added; proof bundle packaged and generated successfully | contract-only verification; no throughput claim, retained path unchanged | keep |
| Factored Stories head `r256` | `python3 scripts/factorize_stories_output_head.py ... --rank 256` then `espresso-generate ... /tmp/stories110m-factored-r256.esp ... Hello` | `/tmp/stories110m-ctx256-v1.esp`: `99.48 tok/s`, `compile_ms=802.03`, exact baseline text | `107.81 tok/s`, `compile_ms=695.04`, `exact_head_backend=ane_factored_classifier` | near-exact label was too optimistic; short and long prompts diverged badly | reject artifact, keep support/tooling |
| Factored Stories head `r512` | `python3 scripts/factorize_stories_output_head.py ... --rank 512` then `espresso-generate ... /tmp/stories110m-factored-r512.esp ... Hello` | `/tmp/stories110m-ctx256-v1.esp`: `99.48 tok/s`, `compile_ms=802.03`, exact baseline text | `111.11 tok/s`, `compile_ms=819.02`, `exact_head_backend=ane_factored_classifier` | faster, but short and long prompts still diverged materially from the retained exact path | reject artifact, keep support/tooling |
| Stable student exact-copy artifact | `python3 scripts/distill_stories_native.py --config configs/stories/distill-stable-copy.json` then short/long `espresso-generate --bundle /tmp/stories110m-stable-copy.esp ...` | no stable Stories-native student artifact | `/tmp/stories110m-stable-copy.esp` exported with `initialization_mode=teacher_copy`; short and long prompt continuations matched `/tmp/stories110m-ctx256-v1.esp` exactly | exact stable student proof; not a throughput claim by itself | keep |
| Exact two-token Stories draft proof | `python3 scripts/package_stories_exact_two_token_draft.py ...` then `espresso-generate ... /tmp/stories110m-exact-two-token-draft.esp ... Hello` | `/tmp/stories110m-ctx256-v1.esp`: `115.86 tok/s`, `compile_ms=1004.71`, `first_token_ms=1.45`, `median_token_ms=8.75`, `p95_token_ms=10.39`, `exact_head_backend=ane_classifier`, `cached_bindings_enabled=true` | `76.16 tok/s`, `compile_ms=0.00`, `first_token_ms=1.15`, `median_token_ms=14.23`, `p95_token_ms=19.66`, `exact_head_backend=cpu_exact_two_token_draft`, `cached_bindings_enabled=false`, `committed_exact_tokens_per_pass=1.9688`, `accepted_future_tokens_per_pass=1.9688` | exact; short and long prompt continuations matched the retained exact bundle, but wall-clock throughput regressed materially | reject artifact, keep support/tooling |
| Teacher-truncated 6-layer Stories student | `python3 scripts/distill_stories_native.py --config configs/stories/distill-truncate-6layer.json` then `espresso-generate ... /tmp/stories110m-truncate-6layer.esp ... Hello` | `/tmp/stories110m-ctx256-v1.esp`: `115.86 tok/s`, exact retained text | `179.28 tok/s`, `compile_ms=658.73`, `first_token_ms=1.38`, `median_token_ms=5.71`, `p95_token_ms=6.64`, `exact_head_backend=ane_classifier`, `cached_bindings_enabled=true` | approximate; short and long prompts diverged badly despite large speedup | reject artifact, keep truncation pipeline support |
| Teacher-truncated 4-layer Stories student | `python3 scripts/distill_stories_native.py --config /tmp/stories110m-truncate-4layer.json` then short/long checks plus `espresso-generate ... /tmp/stories110m-truncate-4layer.esp ... Hello` | `/tmp/stories110m-ctx256-v1.esp`: `115.86 tok/s`, exact retained text | `187.85 tok/s`, `compile_ms=702.02`, `first_token_ms=1.48`, `median_token_ms=5.44`, `p95_token_ms=6.38`, `exact_head_backend=ane_classifier`, `cached_bindings_enabled=true` | approximate; short prompt collapsed into repeated `ely`, long prompt diverged strongly despite the speedup | reject artifact |
| Teacher-truncated 10-layer Stories student | `python3 scripts/distill_stories_native.py --config /tmp/stories110m-truncate-10layer.json` then short/long `espresso-generate --bundle /tmp/stories110m-truncate-10layer.esp ...` | 6-layer candidate showed quality cliff | short prompt `90.93 tok/s`, long prompt `130.25 tok/s` single runs | approximate; still materially divergent, and speedup signal was much weaker than 6-layer | reject artifact |
| Trained 6-layer distilled Stories student | `python3 scripts/distill_stories_native.py --config configs/stories/distill-truncate-6layer-train.json` then short/long `espresso-generate --bundle /tmp/stories110m-truncate-6layer-trained.esp ...` | zero-step 6-layer student had `teacher_token_agreement=0.1101`, `exact_two_token_token0_match_rate=0.2857`, `exact_two_token_future_accept_rate=0.0`, and unusable prompts | trained student reached `teacher_token_agreement=0.6239`, `mean_teacher_student_kl=5.80`, `exact_two_token_token0_match_rate=0.4286`, still `exact_two_token_future_accept_rate=0.0`; short prompt `100.56 tok/s` single run with `compile_ms=12135.85`, long prompt `61.40 tok/s` single run with `compile_ms=11942.05` | approximate; offline agreement improved materially, but generation quality remained poor and ANE compile instability was severe | reject artifact, keep evaluation/training support |
| Trained 8-layer distilled Stories student | `python3 scripts/distill_stories_native.py --config configs/stories/distill-truncate-8layer-train.json` then short/long `espresso-generate --bundle /tmp/stories110m-truncate-8layer-trained.esp ...` | zero-step 8-layer student had `teacher_token_agreement=0.1560`, `exact_two_token_token0_match_rate=0.2857`, `exact_two_token_future_accept_rate=0.0` | trained student reached `teacher_token_agreement=0.6881`, `mean_teacher_student_kl=3.69`, `exact_two_token_token0_match_rate=0.5714`, still `exact_two_token_future_accept_rate=0.0`; short prompt `112.39 tok/s` single run with `compile_ms=19000.51`, long prompt `116.00 tok/s` single run with `compile_ms=18848.10` | approximate; strongest agreement so far, but prompts still diverged and ANE compile instability made it unsuitable as a retained draft proposer | reject artifact, keep evaluation/training support |
| Refreshed exact retained baseline | `espresso-generate ... /tmp/stories110m-ctx256-v1.esp ... Hello` | prior ledger baseline still pointed at older bundle/runtime state | `94.36 tok/s`, `compile_ms=1405.51`, `first_token_ms=1.44`, `median_token_ms=10.21`, `p95_token_ms=14.10`, `exact_head_backend=ane_classifier`, `cached_bindings_enabled=true` | exact retained ctx256 path; long prompt smoke remained coherent | keep |
| Llama split-runtime exact speculation, split 4 | `ESPRESSO_ENABLE_LLAMA_HYBRID_SPECULATIVE=1 ESPRESSO_LLAMA_SPECULATIVE_DRAFT_LAYERS=4 espresso-generate ... /tmp/stories110m-ctx256-v1.esp ... Hello` | exploratory exact baseline rerun: `30.50 tok/s`, `compile_ms=1072.92`, `median_token_ms=32.28`, `p95_token_ms=53.16` | `28.32 tok/s`, `compile_ms=4742.43`, `first_token_ms=38.64`, `median_token_ms=33.07`, `p95_token_ms=64.37`, `exact_head_backend=ane_classifier`, `cached_bindings_enabled=true`, `committed_exact_tokens_per_pass=1.4884`, `accepted_future_tokens_per_pass=0.0465` | exact short-prompt text match, but slower and higher compile cost | reject |
| Llama split-runtime exact speculation, split 6 | `ESPRESSO_ENABLE_LLAMA_HYBRID_SPECULATIVE=1 ESPRESSO_LLAMA_SPECULATIVE_DRAFT_LAYERS=6 espresso-generate ... /tmp/stories110m-ctx256-v1.esp ... Hello` | exploratory exact baseline rerun: `30.50 tok/s`, `compile_ms=1072.92`, `median_token_ms=32.28`, `p95_token_ms=53.16` | `30.62 tok/s`, `compile_ms=2904.76`, `first_token_ms=37.02`, `median_token_ms=33.03`, `p95_token_ms=48.64`, `exact_head_backend=ane_classifier`, `cached_bindings_enabled=true`, `committed_exact_tokens_per_pass=1.5238`, `accepted_future_tokens_per_pass=0.1190` | exact short-prompt text match, but gain was within noise and compile cost regressed materially | reject |
| Llama split-runtime exact speculation, split 8 candidate | `ESPRESSO_ENABLE_LLAMA_HYBRID_SPECULATIVE=1 ESPRESSO_LLAMA_SPECULATIVE_DRAFT_LAYERS=8 espresso-generate ... /tmp/stories110m-ctx256-v1.esp ... Hello` plus short/long prompt compares | exploratory exact baseline rerun: `30.50 tok/s`, `compile_ms=1072.92`, `median_token_ms=32.28`, `p95_token_ms=53.16` | first run `56.71 tok/s`, rerun `44.41 tok/s`, rerun after exact control `79.90 tok/s`; compile ranged `2366.75-5399.68 ms`, `first_token_ms=15.22-16.36`, `median_token_ms=12.10-17.70`, `p95_token_ms=18.15-57.04`, `exact_head_backend=ane_classifier`, `cached_bindings_enabled=true`, `committed_exact_tokens_per_pass=1.6410`, `accepted_future_tokens_per_pass=0.3077` | exact short and long prompt text matched the non-spec path, but the same-binary exact control still ran faster | reject |
| Llama split-runtime exact speculation, split 9 | `ESPRESSO_ENABLE_LLAMA_HYBRID_SPECULATIVE=1 ESPRESSO_LLAMA_SPECULATIVE_DRAFT_LAYERS=9 espresso-generate ... /tmp/stories110m-ctx256-v1.esp ... Hello` | exploratory exact baseline rerun: `30.50 tok/s`, `compile_ms=1072.92`, `median_token_ms=32.28`, `p95_token_ms=53.16` | `29.95 tok/s`, `compile_ms=2444.03`, `first_token_ms=43.80`, `median_token_ms=31.87`, `p95_token_ms=54.86`, `exact_head_backend=ane_classifier`, `cached_bindings_enabled=true`, `committed_exact_tokens_per_pass=1.6842`, `accepted_future_tokens_per_pass=0.3684` | exact short-prompt text match, but slower and higher compile cost | reject |
| Llama split-runtime same-binary control | `ESPRESSO_ENABLE_LLAMA_HYBRID_SPECULATIVE=0 espresso-generate ... /tmp/stories110m-ctx256-v1.esp ... Hello` then split-8 rerun with env enabled | exact non-spec control on the same revision: `89.91 tok/s`, `compile_ms=934.92`, `first_token_ms=1.75`, `median_token_ms=10.85`, `p95_token_ms=11.91`, `exact_head_backend=ane_classifier`, `cached_bindings_enabled=true` | split-8 rerun on the same revision: `79.90 tok/s`, `compile_ms=2366.75`, `first_token_ms=15.22`, `median_token_ms=12.10`, `p95_token_ms=18.15`, `exact_head_backend=ane_classifier`, `cached_bindings_enabled=true`, `committed_exact_tokens_per_pass=1.6410`, `accepted_future_tokens_per_pass=0.3077` | exact short and long prompt text matched, ANE-centric evidence remained intact, but exact non-spec was still faster | reject and revert |
| Post-revert retained verification | `espresso-generate ... /tmp/stories110m-ctx256-v1.esp ... Hello` after removing the llama speculative lane | speculative experiment had been rejected by same-binary control | `58.52 tok/s`, `compile_ms=1031.64`, `first_token_ms=1.46`, `median_token_ms=13.99`, `p95_token_ms=24.70`, `exact_head_backend=ane_classifier`, `cached_bindings_enabled=true` | exact retained non-spec path restored; output remained coherent | keep |
| Hybrid exact-two-token draft exact control | `espresso-generate ... /tmp/stories110m-ctx256-v1.esp ... Hello` before draft-sidecar bundle trials | post-spec retained lane had been restored | `88.35 tok/s`, `compile_ms=4509.50`, `first_token_ms=2.09`, `median_token_ms=10.56`, `p95_token_ms=17.19`, `exact_head_backend=ane_classifier`, `cached_bindings_enabled=true` | exact retained control for the hybrid draft-sidecar slice | keep control |
| Hybrid exact-two-token draft, 6-layer proposer | `espresso-generate ... /tmp/stories110m-exact-two-token-hybrid-6layer.esp ... Hello` plus long-prompt compare | nearby exact control: `88.35 tok/s`, `compile_ms=4509.50`, `median_token_ms=10.56`, `p95_token_ms=17.19` | `61.12 tok/s`, `compile_ms=12620.49`, `first_token_ms=4.16`, `median_token_ms=17.08`, `p95_token_ms=22.22`, `exact_head_backend=ane_classifier`, `cached_bindings_enabled=true`, `committed_exact_tokens_per_pass=1.9688`, `accepted_future_tokens_per_pass=0.8438`, `compile_retries=24`, `compile_failures=30` | exact short and long prompt outputs matched the retained path, but wall-clock throughput regressed badly and ANE compile instability was severe | reject |
| Hybrid exact-two-token draft, 8-layer proposer | `espresso-generate ... /tmp/stories110m-exact-two-token-hybrid-8layer.esp ... Hello` | nearby exact control: `88.35 tok/s`, `compile_ms=4509.50`, `median_token_ms=10.56`, `p95_token_ms=17.19` | `56.02 tok/s`, `compile_ms=20787.55`, `first_token_ms=4.10`, `median_token_ms=19.80`, `p95_token_ms=25.27`, `exact_head_backend=ane_classifier`, `cached_bindings_enabled=true`, `committed_exact_tokens_per_pass=1.9688`, `accepted_future_tokens_per_pass=1.0938`, `compile_retries=36`, `compile_failures=45` | exact short-prompt output matched the retained path, but throughput regressed further and compile instability was worse than 6-layer | reject |
| Post-revert retained verification after hybrid draft-sidecar rejection | `espresso-generate ... /tmp/stories110m-ctx256-v1.esp ... Hello` after removing the hybrid draft-sidecar lane | hybrid draft-sidecar experiment rejected on 6-layer and 8-layer candidates | `110.90 tok/s`, `compile_ms=1611.46`, `first_token_ms=1.32`, `median_token_ms=8.88`, `p95_token_ms=10.29`, `exact_head_backend=ane_classifier`, `cached_bindings_enabled=true` | exact retained non-spec path restored and verified again | keep |
| Llama hybrid layer-input rebinding | `ESPRESSO_ENABLE_LLAMA_HYBRID_LAYER_INPUT_REBIND=1 espresso-generate ... /tmp/stories110m-ctx256-v1.esp ... Hello` | same-binary exact control: `102.44 tok/s`, `compile_ms=2267.64`, `first_token_ms=1.96`, `median_token_ms=8.95`, `p95_token_ms=11.27`, `exact_head_backend=ane_classifier`, `cached_bindings_enabled=true` | `52.20 tok/s`, `compile_ms=1205.66`, `first_token_ms=1.56`, `median_token_ms=14.32`, `p95_token_ms=43.45`, `exact_head_backend=ane_classifier`, `cached_bindings_enabled=true` | exact short-prompt text matched, but throughput collapsed and tail latency regressed badly | reject |
| Hybrid donor-delta default for Stories | pre-change same-binary check with `ESPRESSO_ENABLE_HYBRID_DONOR_DELTA=1 espresso-generate ... /tmp/stories110m-ctx256-v1.esp ... Hello`, then retained build benchmark with plain command | pre-change exact control on the same revision: `102.44 tok/s`, `compile_ms=2267.64`, `first_token_ms=1.96`, `median_token_ms=8.95`, `p95_token_ms=11.27`, `exact_head_backend=ane_classifier`, `cached_bindings_enabled=true`; explicit-disable post-change control failed twice with `ANE compile budget exhausted — exec() restart required` | pre-change env-on probe: `105.61 tok/s`, `compile_ms=1727.63`, `first_token_ms=1.35`, `median_token_ms=9.44`, `p95_token_ms=11.40`; retained post-change plain benchmark: `112.20 tok/s`, `compile_ms=832.73`, `first_token_ms=1.30`, `median_token_ms=8.79`, `p95_token_ms=10.45`, `exact_head_backend=ane_classifier`, `cached_bindings_enabled=true` | exact short and long prompt outputs matched the prior retained path; ANE classifier path and cached bindings stayed enabled | keep |
| Reintroduced llama split-runtime speculative probe on donor-delta baseline | exact control then `ESPRESSO_ENABLE_LLAMA_HYBRID_SPECULATIVE=1 ESPRESSO_LLAMA_SPECULATIVE_DRAFT_LAYERS=8 espresso-generate ... /tmp/stories110m-ctx256-v1.esp ... Hello` | same-revision exact control: `64.64 tok/s`, `compile_ms=5465.36`, `first_token_ms=1.48`, `median_token_ms=15.33`, `p95_token_ms=27.39`, `exact_head_backend=ane_classifier`, `cached_bindings_enabled=true` | split-8 speculative: `56.29 tok/s`, `compile_ms=3303.27`, `first_token_ms=13.58`, `median_token_ms=17.86`, `p95_token_ms=35.38`, `exact_head_backend=ane_classifier`, `cached_bindings_enabled=true`, `committed_exact_tokens_per_pass=1.6410`, `accepted_future_tokens_per_pass=0.3077` | exact short-prompt text matched the same-revision control and ANE-centric execution stayed intact, but wall-clock throughput and first-token latency regressed | reject and revert |
| Post-revert retained verification after donor-delta speculative probe | `espresso-generate ... /tmp/stories110m-ctx256-v1.esp ... Hello` after removing the reintroduced speculative lane | donor-delta baseline speculative probe rejected on same-revision control | `109.69 tok/s`, `compile_ms=2449.90`, `first_token_ms=2.02`, `median_token_ms=8.79`, `p95_token_ms=11.40`, `exact_head_backend=ane_classifier`, `cached_bindings_enabled=true` | exact retained non-spec path restored again; output stayed coherent and ANE classifier path remained active | keep |
| Single-runtime llama hybrid exact prepared-pair probe | plain exact control, then `ESPRESSO_ENABLE_LLAMA_HYBRID_EXACT_PREPARED_PAIR=1 espresso-generate ... /tmp/stories110m-ctx256-v1.esp ... Hello`, then same-revision exact control rerun | first exact control: `86.85 tok/s`, `compile_ms=6023.01`, `first_token_ms=1.87`, `median_token_ms=10.85`, `p95_token_ms=15.00`, `exact_head_backend=ane_classifier`, `cached_bindings_enabled=true`; same-revision exact rerun: `94.78 tok/s`, `compile_ms=1515.08`, `first_token_ms=1.70`, `median_token_ms=9.52`, `p95_token_ms=12.91`, `exact_head_backend=ane_classifier`, `cached_bindings_enabled=true` | prepared-pair run: `93.90 tok/s`, `compile_ms=2630.89`, `first_token_ms=1.59`, `median_token_ms=11.31`, `p95_token_ms=15.46`, `exact_head_backend=ane_classifier`, `cached_bindings_enabled=true` | exact short and long prompt outputs matched the retained path, but the same-revision exact control rerun still finished faster and the prepared-pair path did not establish a clean retained win | reject and revert |
| Post-revert retained verification after exact prepared-pair probe | `espresso-generate ... /tmp/stories110m-ctx256-v1.esp ... Hello` after removing the prepared-pair lane | exact prepared-pair probe rejected by same-revision control | `121.31 tok/s`, `compile_ms=3645.10`, `first_token_ms=1.64`, `median_token_ms=8.45`, `p95_token_ms=10.27`, `exact_head_backend=ane_classifier`, `cached_bindings_enabled=true` | exact retained non-spec path restored again; output stayed coherent and ANE classifier path remained active | keep |
| Verifier-side factored proposer-head control | `ESPRESSO_FORCE_EXACT_HEAD_BACKEND=fp16 ESPRESSO_BUNDLE_OUTPUT_HEAD_KIND=factored ESPRESSO_BUNDLE_OUTPUT_HEAD_BOTTLENECK=512 ESPRESSO_BUNDLE_OUTPUT_HEAD_GROUPS=1 ESPRESSO_BUNDLE_OUTPUT_HEAD_PROJECTION_REF=weights/proposer/cls_proj.bin ESPRESSO_BUNDLE_OUTPUT_HEAD_EXPANSION_REF=weights/proposer/cls_expand.bin espresso-generate ... /tmp/stories110m-ctx256-v1.esp ... Hello` | retained exact baseline: `121.31 tok/s`, `compile_ms=3645.10`, `first_token_ms=1.64`, `median_token_ms=8.45`, `p95_token_ms=10.27`, `exact_head_backend=ane_classifier`, `cached_bindings_enabled=true` | `44.47 tok/s`, `compile_ms=5248.13`, `first_token_ms=5.35`, `median_token_ms=16.87`, `p95_token_ms=41.57`, `exact_head_backend=cpu_fp16_tiled`, `cached_bindings_enabled=true` | short-prompt output diverged from the retained exact path and ANE head residency was lost before speculation was even enabled | reject |
| Verifier-side factored proposer-head speculative probe | `ESPRESSO_FORCE_EXACT_HEAD_BACKEND=fp16 ESPRESSO_BUNDLE_OUTPUT_HEAD_KIND=factored ESPRESSO_BUNDLE_OUTPUT_HEAD_BOTTLENECK=512 ESPRESSO_BUNDLE_OUTPUT_HEAD_GROUPS=1 ESPRESSO_BUNDLE_OUTPUT_HEAD_PROJECTION_REF=weights/proposer/cls_proj.bin ESPRESSO_BUNDLE_OUTPUT_HEAD_EXPANSION_REF=weights/proposer/cls_expand.bin ESPRESSO_ENABLE_LLAMA_HYBRID_FACTORED_PROPOSER_SPEC=1 espresso-generate ... /tmp/stories110m-ctx256-v1.esp ... Hello` plus long-prompt compare | same-lane control: `44.47 tok/s`, `compile_ms=5248.13`, `first_token_ms=5.35`, `median_token_ms=16.87`, `p95_token_ms=41.57`, `exact_head_backend=cpu_fp16_tiled`, `cached_bindings_enabled=true` | first run: `29.38 tok/s`, `compile_ms=2258.20`, `first_token_ms=12.85`, `median_token_ms=19.46`, `p95_token_ms=47.77`; hot donor-cache rerun: `51.16 tok/s`, `compile_ms=2164.37`, `first_token_ms=10.19`, `median_token_ms=18.79`, `p95_token_ms=21.39`, `exact_head_backend=cpu_fp16_tiled`, `cached_bindings_enabled=true`; long prompt: `48.19 tok/s`, `compile_ms=3196.92`, `first_token_ms=13.18`, `median_token_ms=19.84`, `p95_token_ms=27.15` | short and long prompt outputs diverged from the retained exact ANE path, the verifier head stayed on `cpu_fp16_tiled`, and even the hot donor-cache rerun stayed far below the retained control | reject and revert |
| Fused llama exact-head probe | same-binary exact control, then `ESPRESSO_ENABLE_LLAMA_HYBRID_FUSED_EXACT_HEAD=1 espresso-generate ... /tmp/stories110m-ctx256-v1.esp ... Hello`, plus long-prompt compare on and off | same-binary control: `103.70 tok/s`, `compile_ms=5951.56`, `first_token_ms=2.14`, `median_token_ms=13.40`, `p95_token_ms=16.85`, `exact_head_backend=ane_classifier`, `cached_bindings_enabled=true` | env-on probe: `104.63 tok/s`, `compile_ms=1838.80`, `first_token_ms=1.84`, `median_token_ms=8.91`, `p95_token_ms=10.89`, `exact_head_backend=ane_classifier`, `cached_bindings_enabled=true`; long prompt control: `108.53 tok/s`; long prompt env-on: `116.76 tok/s` | short and long prompt outputs matched exactly while the fused path removed one unlabeled head compile (`a3/s3` -> `a2/s2`) and reduced per-token latency | keep |
| Fused llama exact-head default-on for Stories | same-revision explicit-disable control `ESPRESSO_DISABLE_LLAMA_HYBRID_FUSED_EXACT_HEAD=1 espresso-generate ... /tmp/stories110m-ctx256-v1.esp ... Hello`, then plain retained command | explicit-disable control: `76.73 tok/s`, `compile_ms=2766.08`, `first_token_ms=1.55`, `median_token_ms=14.60`, `p95_token_ms=24.99`, `exact_head_backend=ane_classifier`, `cached_bindings_enabled=true` | plain retained command: `111.45 tok/s`, `compile_ms=2170.00`, `first_token_ms=1.34`, `median_token_ms=8.98`, `p95_token_ms=10.41`, `exact_head_backend=ane_classifier`, `cached_bindings_enabled=true` | exact short-prompt output matched, long-prompt parity was already verified on the same fused code path before flipping the default, and ANE classifier residency stayed intact | keep |
| Incremental ANE head-spatial override (`laneSpatial=1`) | same-revision fused-head control, then `ESPRESSO_INCREMENTAL_HEAD_SPATIAL=1 espresso-generate ... /tmp/stories110m-ctx256-v1.esp ... Hello` | same-revision fused-head control: `95.37 tok/s`, `compile_ms=4933.91`, `first_token_ms=1.36`, `median_token_ms=9.34`, `p95_token_ms=13.95`, `exact_head_backend=ane_classifier`, `cached_bindings_enabled=true` | ANE eval failed before completion: `status=0x1d`, `statusType=0x9`, `Llama hybrid greedy ANE head evaluation failed: ANE kernel evaluation failed` | no retained quality result because the smaller logical head shape is not a valid ANE runtime configuration on the real Stories path | reject and revert |
| Unified ANE future-head + real Stories future-sidecar | plain retained exact bundle and `/tmp/stories110m-future-sidecar.esp` on short, long, and release benchmark commands | same-revision exact retained benchmark: `118.41 tok/s`, `compile_ms=2506.79`, `first_token_ms=1.42`, `median_token_ms=8.39`, `p95_token_ms=9.63`, `exact_head_backend=ane_classifier`, `cached_bindings_enabled=true` | future-sidecar benchmark: `102.96 tok/s`, `compile_ms=3153.24`, `first_token_ms=2.05`, `median_token_ms=9.79`, `p95_token_ms=11.72`, `exact_head_backend=ane_classifier`, `cached_bindings_enabled=true`, `committed_exact_tokens_per_pass=1.0000`, `accepted_future_tokens_per_pass=0.0000`; short prompt matched exactly at `111.17 tok/s`; long prompt matched exactly at `104.54 tok/s` vs retained `116.74 tok/s` | exact short/long prompt outputs matched and ANE classifier residency stayed intact, but the seeded future sidecar never accepted a future token and only added extra ANE head cost | reject and revert |
| Trained committed-activation future-sidecar on retained hybrid verifier | rebuilt `espresso-generate`, then short/long prompt compares and `espresso-generate ... /tmp/stories110m-future-head-stable-copy.esp ... Hello` against same-revision exact control | same-revision exact retained benchmark: `91.78 tok/s` on the stale pre-rebuild smoke run, then rebuilt exact control: `118.19 tok/s`, `compile_ms=836.82`, `first_token_ms=1.31`, `median_token_ms=8.54`, `p95_token_ms=9.89`, `exact_head_backend=ane_classifier`, `cached_bindings_enabled=true` | trained future-sidecar short prompt matched exactly at `91.97 tok/s` before rebuild and `67.50 tok/s` after rebuild; long prompt matched exactly at `62.40 tok/s`; rebuilt real Stories benchmark: `80.22 tok/s`, `compile_ms=1336.49`, `first_token_ms=2.09`, `median_token_ms=11.47`, `p95_token_ms=15.87`, `exact_head_backend=ane_classifier`, `cached_bindings_enabled=true`, `committed_exact_tokens_per_pass=1.0000`, `accepted_future_tokens_per_pass=0.0000` | exact short/long prompt outputs matched and ANE classifier residency stayed intact, but the trained proposer still never achieved a single accepted future token on the real benchmark and the wall-clock path regressed badly after the rebuilt binary | reject and revert |
| Unified multi-head verifier sharing retained exact RMSNorm work | rebuilt `espresso-generate` with a dual-output ANE exact+future head fed from the retained llama hybrid activation, then reran short/long prompt compares and the real Stories benchmark on `/tmp/stories110m-future-head-stable-copy.esp` | same-revision exact retained benchmark after the unified-head rebuild: `118.19 tok/s`, `compile_ms=836.82`, `first_token_ms=1.31`, `median_token_ms=8.54`, `p95_token_ms=9.89`, `exact_head_backend=ane_classifier`, `cached_bindings_enabled=true` | unified-head short prompt matched exactly at `67.50 tok/s`, long prompt matched exactly at `62.40 tok/s`; real Stories benchmark: `80.22 tok/s`, `compile_ms=1336.49`, `first_token_ms=2.09`, `median_token_ms=11.47`, `p95_token_ms=15.87`, `exact_head_backend=ane_classifier`, `cached_bindings_enabled=true`, `committed_exact_tokens_per_pass=1.0000`, `accepted_future_tokens_per_pass=0.0000` | the unified head preserved exact/verifier-preserved output and stayed ANE-centric, but sharing the exact RMSNorm work still did not unlock any accepted future token and remained materially slower than the retained exact baseline | reject and revert |
| Larger-corpus trained future-sidecar runtime retry after offline gate clear | real release benchmark on the retained exact bundle, then the copied bundle `/private/tmp/stories110m-future-head-generated-v1.esp` containing the generated-corpus `weights/future-sidecar.bin`; short and long prompt compares on both bundles | same-revision exact retained benchmark: `120.64 tok/s`, `compile_ms=3777.42`, `first_token_ms=1.22`, `median_token_ms=8.38`, `p95_token_ms=9.83`, `exact_head_backend=ane_classifier`, `cached_bindings_enabled=true` | generated-corpus future-sidecar benchmark: `91.79 tok/s`, `compile_ms=1502.80`, `first_token_ms=2.37`, `median_token_ms=10.79`, `p95_token_ms=12.47`, `exact_head_backend=ane_classifier`, `cached_bindings_enabled=true`, `committed_exact_tokens_per_pass=1.0000`, `accepted_future_tokens_per_pass=0.0000`; short prompt matched exactly; long prompt matched exactly | the offline artifact improved future-token match rate materially (`0.47%` -> `19.30%` on the teacher-activation gate), but the retained ANE runtime still accepted zero future tokens on the real Stories benchmark and regressed wall-clock throughput | reject and revert |

## Review

- Retained this session:
  a new structured ANE eval-report path now records wall time, hardware execution time, host overhead, eval path, and error metadata in the interop/runtime boundary; EspressoBench now surfaces that telemetry across direct, inference, and decode modes instead of collapsing everything into a single ANE bucket, and the fused decode benches no longer report zero timing breakdowns.
- Retained this session:
  the first Zig serving-runtime foothold now exists under `Sources/ZigRuntime/espresso_runtime.zig` with a reproducible `scripts/build_zig_runtime.sh` build path, establishing the C ABI and artifact workflow for moving the execution core out of Swift/ObjC incrementally.
- Retained this session:
  Zig runtime build/load is now wired into SwiftPM through `Plugins/ZigRuntimePlugin`, and the ANE interop layer now resolves and uses the built Zig dylib for host-side convert/copy/gather/scatter/argmax helpers instead of treating Zig as a manual sidecar.
- Retained this session:
  low-level surface handling is now unified around shared `SurfaceIO`/interop helpers, including FP32 slice IO, byte-zeroing, FP16 scalar fills, and direct RealModelInference reuse; this removed the duplicated local IOSurface read/write helpers from the real-model path.
- Retained this session:
  created IOSurfaces now recycle through `ANESurfacePool`, and `ANESurfaceOwner` returns pool-owned temporary surfaces automatically instead of churning allocations for every serving lane/session lifecycle.
- Retained this session:
  the direct ANE generation serving lane now has an explicit reusable pool API (`ANEDirectGenerationModelPool`) so compiled models can be leased/reset per request without state aliasing or recompilation, which is the intended production boundary for concurrent serving.
- Retained this session:
  decode and recurrent lane initialization no longer allocate temporary `[Float]` buffers just to write zeros or decode-mask sentinels; reset/setup now zero or scalar-fill surfaces directly through interop, reducing host-side setup churn on the serving path.
- Retained this session:
  the serving path now has explicit ANE surface ownership via `ANESurfaceOwner`, the forward hot loops no longer fetch retained input/output surfaces inside per-layer execution when caches are absent, and the standard transformer decode path now has a concrete `TransformerDecodeSession` that owns `surfaceHandles + decodeState + reset/step` semantics used by the direct generation harness and decode benchmark.
- Retained this session:
  the exact-two-token llama bundle path now has a real hybrid ANE-backed runtime lane for draft-sidecar prefill, propose, rollback, and verifier-controlled exact commit instead of routing llama draft bundles through `CPUExactLlamaRuntime`; the CPU draft path remains only as the explicit forced-CPU fallback.
- Retained this session:
  focused regression coverage now protects the exact-two-token bundle path under forced CPU fallback, and the full package verification completed cleanly after the hybrid draft/runtime change (`swift test`: `370 tests in 5 suites`, passed; XCTest hardware-gated skips remained expected).
- Retained:
  bundle contract v`1.1.0`, context-target packing, measured bundle token latencies, output-head/draft manifest-runtime contract support, factorized-head runtime/tooling support, the stable teacher-copied Stories student pipeline/artifact, the teacher-truncation student pipeline support, and the executed exact two-token draft bundle/runtime proof path.
- Rejected this session:
  the llama hybrid split-runtime speculative lane. It preserved exact short/long prompt output and ANE-centric execution evidence, but the same-binary exact control still beat the best measured split-8 run (`89.91 tok/s` exact non-spec vs `79.90 tok/s` split-8), so the code was reverted.
- Rejected this session:
  the hybrid exact-two-token draft-sidecar lane using the trained `6`-layer and `8`-layer proposers. Both candidates preserved exact short/long prompt output and ANE-centric execution evidence, but both were materially slower than the exact control (`88.35 tok/s` exact non-spec vs `61.12 tok/s` for 6-layer and `56.02 tok/s` for 8-layer) and showed severe ANE compile retry churn, so the code was reverted.
- Rejected this session:
  llama hybrid layer-input rebinding on Stories. It preserved exact output and ANE-centric execution evidence, but the same-binary benchmark regressed from `102.44 tok/s` to `52.20 tok/s`, so it remains default-off for llama.
- Retained this session:
  hybrid donor-delta is now default-on for `stories110m`. The pre-change same-binary env probe improved throughput from `102.44 tok/s` to `105.61 tok/s`, and the retained plain command measured `112.20 tok/s` with the same exact ANE classifier path while the explicit-disable control twice hit ANE compile-budget exhaustion.
- Retained this session:
  the fused llama exact ANE head is now default-on for `stories110m`. It preserves exact short/long prompt output, stays on `exact_head_backend=ane_classifier`, drops one unlabeled head compile stage, and the same-revision disable-vs-plain gate improved the real Stories benchmark from `76.73 tok/s` to `111.45 tok/s`.
- Rejected this session:
  a bounded incremental head-spatial override on the fused Stories exact head. Reducing the logical head shape to `laneSpatial=1` compiled but failed at real ANE eval with `status=0x1d` / `statusType=0x9`, so the override was removed immediately.
- Rejected this session:
  the unified ANE future-head / real Stories future-sidecar path. It preserved exact short/long prompt output and ANE classifier residency, but the teacher-seeded sidecar never accepted a future token (`accepted_future_tokens_per_pass=0.0000`) and the real Stories benchmark regressed from `118.41 tok/s` to `102.96 tok/s`, so the runtime slice was reverted immediately.
- Rejected this session:
  the trained committed-activation future-sidecar on the retained hybrid verifier. It preserved exact short/long prompt output and stayed on `exact_head_backend=ane_classifier` with `cached_bindings_enabled=true`, but the trained proposer still measured `accepted_future_tokens_per_pass=0.0000` and the rebuilt real Stories benchmark fell from `118.19 tok/s` to `80.22 tok/s`, so the slice does not stay in the retained path.
- Rejected this session:
  the unified multi-head verifier fallback that shared the retained exact RMSNorm work. It preserved exact short/long prompt output and ANE-centric execution evidence, but the same trained sidecar still delivered `accepted_future_tokens_per_pass=0.0000` and the real Stories benchmark remained at `80.22 tok/s` versus the retained exact `118.19 tok/s`, so that fallback also gets discarded.
- Rejected this session:
  the larger-corpus future-sidecar runtime retry. The offline training pipeline and teacher-generated corpus materially improved future-token match rate (`0.47%` -> `19.30%`) and preserved exact short/long prompt output in the copied bundle, but the retained ANE runtime still measured `accepted_future_tokens_per_pass=0.0000` and regressed from `120.64 tok/s` to `91.79 tok/s`, so the runtime slice was reverted again.
- Rejected this session:
  a bounded reintroduction of llama split-runtime speculation on top of the donor-delta retained baseline. The same-revision exact control still beat split-8 (`64.64 tok/s` exact vs `56.29 tok/s` speculative), and speculative first-token latency worsened from `1.48 ms` to `13.58 ms`, so the code was reverted immediately and the retained exact path was re-verified at `109.69 tok/s`.
- Rejected this session:
  a single-runtime llama hybrid exact prepared-pair upper-bound path. It preserved exact short/long prompt output and stayed on the ANE classifier path, but the same-revision exact control rerun still beat it (`94.78 tok/s` exact vs `93.90 tok/s` prepared-pair), so the code was reverted and the retained exact path was re-verified at `121.31 tok/s`.
- Rejected this session:
  the two remaining credible lower-overhead proposer directions. The verifier-side factored proposer-head path forced the verifier off the retained ANE classifier path onto `cpu_fp16_tiled`, cut throughput to `44.47 tok/s` even before speculation, then measured `29.38 tok/s` on the first speculative run and only `51.16 tok/s` on the hot donor-cache rerun; the long prompt also diverged from the retained exact output, so the code was reverted.
- Closed:
  no retained verifier-side proposer breakthrough exists in the product lane. Both materially different single-runtime attempts failed the hard gate because `accepted_future_tokens_per_pass` stayed at `0.0000`, so the serving lane remains the retained exact hybrid ANE path and the dead proposer branches stay reverted.
- Retained this session:
  the recurrent proof probe now has an explicit tokenizer-source resolver with test coverage, and `GPT2BPETokenizer(tokenizerJSONURL:)` now accepts sparse HF token IDs instead of rejecting healthy artifact tokenizers. That keeps prompt-text benchmarking on the fast-tokenizer JSON path for real recurrent artifacts while still respecting an explicit `tokenizer.model` file path when asked.
- Rejected this session:
  the first corrected mixed-schedule grouped-dual-state proof artifact at `results/stories-gds-12x128-v1-fast-smoke`. The runtime path is real and the artifact runs on ANE, but the retained `128`-token probe still collapsed completely: with authoritative HF prompt IDs for `Once upon a time`, the release probe measured only `97.95 tok/s` and generated token `0` (`<unk>`) for the full decode. The local prompt-text tokenizer path was also not publication-safe on this artifact family until the sparse-JSON fix, so this artifact does not count as a publishable benchmark candidate.
- Retained this session:
  grouped-dual-state certification is now much tighter. The recurrent cell MIL was aligned to the Python `GroupedDualStateBlock` contract, and new hardware tests prove:
  the synthetic grouped-dual-state session matches the Swift CPU reference for two steps,
  the retained smoke recurrent prefix for layers `0..2` matches the Swift CPU reference both for token `931` and across the real prompt `[9038, 2501, 263, 931]`,
  the retained grouped checkpoint sidecar matches Swift CPU decode-attention both on embeddings and on real recurrent-prefix activations,
  and the full grouped model matches a manual Swift session chain on the retained smoke prompt.
  Verified with:
  `ANE_HARDWARE_TESTS=1 swift test --filter GroupedDualStateGenerationModelHardwareTests`
  `ANE_HARDWARE_TESTS=1 swift test --filter HybridDecodeForwardPassTests/test_grouped_dual_state_smoke_checkpoint_layer_matches_cpu_decode_attention_on_hardware`
  `ANE_HARDWARE_TESTS=1 swift test --filter HybridDecodeForwardPassTests/test_grouped_dual_state_first_checkpoint_matches_cpu_decode_attention_on_recurrent_prefix_hardware`
- Rejected this session:
  the idea that grouped-dual-state is blocked on a broad wrapper or prompt-tokenization bug. The probe now emits `prompt_argmax_trace`, and the retained artifact shows the first Swift/Python split on the very first prompt token: Python `[(9038,263),(2501,263),(263,263),(931,29892)]` versus Swift `[(9038,471),(2501,471),(263,471),(931,287)]` in `results/stories-gds-12x128-v1-fast-smoke/probe-once-upon-a-time-v3.json`.
  The decisive retained debug evidence is:
  `results/stories-gds-12x128-v1-fast-smoke/python-single-token-layer-9038-v1.json`
  plus the Swift hardware trace from `ESPRESSO_DEBUG_LAYER_TRACE=1 ANE_HARDWARE_TESTS=1 swift test --filter GroupedDualStateGenerationModelHardwareTests/test_grouped_dual_state_full_model_matches_manual_session_chain_on_smoke_prompt`.
  Swift matches Python through recurrent layer `2` closely enough to preserve the same layerwise argmax (`9038,9038,9038`), but it diverges sharply at the first checkpoint layer:
  Python single-token layer trace `[...,1965,...,263]`
  versus Swift `[...,2400,...,471]`,
  with retained debug diffs `layer 2 max abs diff ~= 0.0114` and `layer 3 max abs diff ~= 0.7165`.
  Keep/kill view:
  the active grouped-dual-state bug is now a tensor-level Swift/Python hidden-state mismatch *before* the first checkpoint, not a generic recurrent wrapper failure and not the checkpoint block itself.
  Follow-up isolate:
  when the Swift first checkpoint session is fed the exact Python layer-2 hidden from `python-single-token-layer-9038-v1.json`, it reproduces Python layer-3 closely (`max abs diff ~= 0.0070`).
  That means the checkpoint semantics are locally correct enough; the actual bug budget is the recurrent-prefix hidden drift that remains even when layerwise argmax still agrees.
- Retained this session:
  grouped-dual-state training now consumes real mini-batches instead of ignoring `batch_size`. The helper is test-covered in `scripts/tests/test_distill_stories_grouped_dual_state.py`, so the next recurrent quality passes can use stable multi-example updates instead of fake single-window training.
- Rejected this session:
  the stronger grouped-dual-state rescue artifact at `results/stories-gds-12x128-v1-softdistill-a`. Training quality improved materially (`teacher_token_agreement=0.7955`, `label_token_accuracy=0.2792`, `mean_teacher_student_kl=4.4761`, `exact_two_token_future_accept_rate=0.0833`), but rollout quality stayed unpublishable and the release probe still failed the throughput goal by a wide margin: `136.50 tok/s` on authoritative HF prompt IDs for `Once upon a time`, with the decode collapsing to token `0` for all `128` generated tokens. This is the keep/kill point for grouped dual state as a model family; keep the runtime harness, kill further open-ended tuning.
- Retained this session:
  the `gdn-h1` pivot now has a live bench-only ANE prototype. The new files are `GatedDeltaNetPrototypeWeights.swift`, `GatedDeltaNetPrototypeStepGenerator.swift`, `GatedDeltaNetPrototypeKernelSet.swift`, `GatedDeltaNetPrototypeDecode.swift`, plus the `espresso-bench` microbench hook and generator tests. The first width sweep says `128` channels/head is the current keep point: `4485.98 tok/s` and `3952.90 tok/s` across two repeats, versus `3696.00 tok/s` at `64` and `3129.68 tok/s` at `192`.
- Retained this session:
  the `gdn-h1` generation lane now has its own binary weight contract via `GatedDeltaNetGenerationWeights.swift`, `GatedDeltaNetGenerationWeightStore.swift`, and `GatedDeltaNetGenerationWeightStoreTests.swift`. The recurrent cell family still is not wired into the probe executable, but the pivot now has a measured runtime prototype and a real checkpoint container instead of staying bench-only.
- Retained 2026-04-10:
  grouped-dual-state parity is now closed at bit-exact argmax against the Python oracle on the retained `stories-gds-12x128-v1-fast-smoke` artifact. The root cause was a shape-invisible per-head transpose on `WcarryX`, `WcarryD`, and `Wo`: Python declares these at `distill_stories_grouped_dual_state.py:152-155` as `(head_count, channels_per_head, channels_per_head) = (12, 64, 64)` and uses them with einsum `"bhc,hcd->bhd"` where the middle axis is input and the last is output, while the four other weights (`WmemQ`, `WmemS`, `WmemV`, `WcarryM`) use the opposite convention that matches Swift's grouped conv flatten `(out_total, in_per_group)`. Because `c == d == 64` on the three square weights, every static shape check passed and the bug was invisible until a layer-2 hidden-state drift of ~0.0114 exploded to ~0.7165 at the first SWA checkpoint through softmax amplification.
  The fix is a single-point transpose at the checkpoint loader in `Sources/Espresso/GroupedDualStateGenerationWeightStore.swift`: a new `transposePerHeadSquareInPlace` helper applies an in-place per-head (i,j)→(j,i) swap on the three suspect weights right after `readTensor`. The four non-suspect weights are left untouched. Save/on-disk layout stays in Python-native `(h, in, out)` so existing retained checkpoints (including `weights/recurrent.bin`) remain valid without re-export.
  Evidence retained:
  `results/stories-gds-12x128-v1-fast-smoke/diagnostic-weight-layout-v1.json` (Milestone 0: WcarryX head-0 max-abs asymmetry 0.0124, Swift flattened view exactly equals Python `W.transpose(1,2)`, control WmemQ diff 0.0)
  `results/stories-gds-12x128-v1-fast-smoke/python-single-token-layer-9038-v2.json` (Milestone A: Python oracle with hidden/mem/carry per layer)
  `results/stories-gds-12x128-v1-fast-smoke/python-single-token-layer0-intermediates-9038-v1.json` (Milestone B: layer-0 intermediates with all cell tensors)
  `results/stories-gds-12x128-v1-fast-smoke/probe-once-upon-a-time-v4-post-transpose-fix.json` (Milestone D1: post-fix probe matching Python bit-exact on all 4 prompt tokens and all 8 generated tokens).
  New gates:
  `test_grouped_dual_state_cpu_reference_layer_by_layer_matches_python_oracle` (tolerance 0.001, all 12 layers pass FP32 vs FP32)
  `test_grouped_dual_state_ane_layer_by_layer_matches_python_oracle` (tolerance 0.005 for FP16 ulp budget, all 12 layers pass, requires `ANE_HARDWARE_TESTS=1`)
  `test_grouped_dual_state_layer0_intermediate_divergence_probe` (all intermediates at floating-point noise level post-fix: carry_x 5.2e-8, proj 2.8e-9, output 7.5e-9)
  `test_weight_store_load_transposes_carry_x_carry_d_wo_per_head` (new unit test with index-encoded synthetic weights pinning the transpose behavior).
  Fixed round-trip tests: `GroupedDualStateGenerationWeightStoreTests.swift:62` and `MultitokenProbeSupportTests.swift:278` both updated to compare against the per-head-transposed expected value using a shared `expectedAfterPerHeadTranspose` helper.
  Verified with:
  `swift test --filter GroupedDualStateGenerationWeightStoreTests` (pass)
  `swift test --filter MultitokenProbeSupportTests` (all 17 pass)
  `swift test --filter "GroupedDualStateGenerationModelHardwareTests/test_grouped_dual_state_cpu_reference_layer_by_layer_matches_python_oracle|GroupedDualStateGenerationModelHardwareTests/test_grouped_dual_state_layer0_intermediate_divergence_probe"` (pass)
  `ANE_HARDWARE_TESTS=1 swift test --filter "GroupedDualStateGenerationModelHardwareTests/test_grouped_dual_state_ane_layer_by_layer_matches_python_oracle"` (pass at 0.005 FP16 tolerance)
  `ANE_HARDWARE_TESTS=1 swift test --filter HybridDecodeForwardPassTests/test_hybrid_decode_single_step_runs_on_hardware` (pass in isolation; pre-existing per-process ANE compile budget issue flaked it in the bundled run)
  `.build/release/espresso-multitoken-probe --mode generate-recurrent --input recurrent-checkpoint --recurrent-checkpoint results/stories-gds-12x128-v1-fast-smoke/weights/recurrent.bin --prompt-token-ids 9038,2501,263,931 --max-new-tokens 8 --max-sequence-tokens 16 --warmup 1 --iterations 1 --layer-count 12 --control-backend single --output-head-backend cpu` produced argmax trace `[(9038,263),(2501,263),(263,263),(931,29892)]` matching Python exactly and generated tokens `[29892, 29892, 263, 263, 29892, 29892, 29892, 727]` also matching Python's `python-runtime-trace-v1.json` generated field bit-exact.
- Rejected 2026-04-10:
  the idea that grouped-dual-state quality work is a runtime problem. The smoke checkpoint now generates exactly the same tokens as the Python oracle on the same prompt, and those tokens are still repetitive (`", , a a , , , there"`). The repetitive output is a direct consequence of the smoke config training for only `128` steps on `max_samples=64 batch_size=1`, not a parity bug. Runtime is done; any further grouped-dual-state quality progress is a training problem that requires a longer training sweep with a larger dataset — that work is out of scope for this session and does not require ANE expertise. Milestone D2 (bounded quality recovery) is therefore skipped: the smoke checkpoint's output is bit-exact with Python, so retraining Swift against a stale Python oracle would add no information.
- Keep/kill 2026-04-10:
  KEEP grouped-dual-state as a publication candidate at the runtime level. The runtime harness is proven correct and the parity gate is in place so any future training iteration can be validated at bit-exact precision. The publication lane decision is now gated on training quality, not runtime work.
- Retained 2026-04-10:
  Option 2 publication target achieved. Grouped-dual-state `3L no-SWA` config reaches median `510.66 tok/s` on M3 Max with fused-triplet trunk + ANE RMSNorm+classifier head on the retained smoke prompt `[9038, 2501, 263, 931]`, and a sibling `4L no-SWA` config reaches `442.35 tok/s` with single trunk + ANE head. Both results are 5-7x over the retained `77.69 tok/s` exact shipping baseline. Both produce recognizable English story structure from the smoke corpus (undertrained on 24 inline prompts but unambiguously not collapsed).
  Root cause of the unlock:
  the 12L + 3SWA schedule is architecturally capped at ~100 tok/s because the 3 SWA checkpoint layers consume ~52% of the per-token budget on M3 Max (measured: ~1.77 ms/SWA-layer vs ~0.37 ms/GDS-layer fused-triplet). Removing SWA entirely and using a shallow 3-4 layer pure-recurrent trunk is the only path to the publication throughput target; prior "softdistill-a" training at 12L+3SWA could not be optimized into the target even after the parity fix unlocked its latent quality. Training a fresh 3L+0SWA and 4L+0SWA config from the same 768-step softdistill-a recipe produced usable models in ~5 minutes of MPS training.
  Evidence retained:
  - config: configs/stories/stories-gds-3x128-v1-nosewa.json
  - config: configs/stories/stories-gds-4x128-v1-nosewa.json
  - checkpoint: results/stories-gds-3x128-v1-nosewa/weights/recurrent.bin (teacher_token_agreement=0.3896, kl=18.84 at step 768)
  - checkpoint: results/stories-gds-4x128-v1-nosewa/weights/recurrent.bin (teacher_token_agreement=0.4383, kl=16.50 at step 768)
  - publication probe: results/stories-gds-3x128-v1-nosewa/probe-publication-v1.json (510.66 tok/s median)
  - publication probe: results/stories-gds-4x128-v1-nosewa/probe-publication-v1.json (442.35 tok/s median)
  - distill report: results/stories-gds-3x128-v1-nosewa/distill-report.json
  - distill report: results/stories-gds-4x128-v1-nosewa/distill-report.json
  - layer extraction diagnostic (KILLED approach): results/stories-gds-3x128-v1-nosewa-extract/, results/stories-gds-4x128-v1-nosewa-extract/, results/stories-gds-6x128-v1-nosewa-extract/ — all produce identity collapse because the softdistill-a output head was trained to consume layer-11 features, not layer-2/3/6 features; simple layer extraction is a dead end for this architecture
  - softdistill-a post-fix probe: results/stories-gds-12x128-v1-softdistill-a/probe-once-upon-a-time-post-transpose-fix-32tok.json and probe-post-transpose-fix-quality-signal.md — proves the parity fix unlocks the latent quality in the existing softdistill-a checkpoint (generates real English fragments like "Once upon a time, a time, a, there a lived, there was a" instead of the pre-fix collapse to token 0)
  Headline comparison:
  | Config | Median tok/s | vs shipping (77.69) | 300 target |
  |---|---|---|---|
  | 3L no-SWA + fused-triplet + ANE head | 510.66 | 6.57x | +70% |
  | 4L no-SWA + single + ANE head | 442.35 | 5.69x | +47% |
  | 12L+3SWA softdistill-a + fused-triplet + ANE head (post-fix) | 92.69 | 1.19x | -69% |
  | 12L+3SWA softdistill-a + fused-triplet + CPU head (post-fix) | 86.54 | 1.11x | -71% |
  Commands:
  - train: `python3 scripts/distill_stories_grouped_dual_state.py --config configs/stories/stories-gds-3x128-v1-nosewa.json`
  - train: `python3 scripts/distill_stories_grouped_dual_state.py --config configs/stories/stories-gds-4x128-v1-nosewa.json`
  - probe: `.build/release/espresso-multitoken-probe --mode generate-recurrent --input recurrent-checkpoint --recurrent-checkpoint results/stories-gds-3x128-v1-nosewa/weights/recurrent.bin --prompt-token-ids 9038,2501,263,931 --max-new-tokens 32 --max-sequence-tokens 48 --warmup 5 --iterations 20 --layer-count 3 --control-backend fused-triplet --output-head-backend ane-rmsnorm-classifier`
  Keep/kill verdict:
  KEEP the grouped-dual-state runtime, the transpose fix, the parity gate, the ANE head wiring, and the no-SWA 3L/4L configs as the new publication candidates. The 3L no-SWA config is the headline lane for the 300 tok/s replacement-program target (Option 2). The 12L+3SWA smoke/softdistill-a configs are DEMOTED as publication candidates but stay retained as quality references for the smaller configs.
  Known limitations (not blockers for the publication lane):
  1. Training corpus is 24 inline benchmark prompts only; produced models are undertrained, so quality is coherent-but-degenerative. A larger training corpus (TinyStories scale) with the same 3L+0SWA architecture would produce publication-quality output. This is the next session's scope.
  2. Generated output shows cyclic patterns (4L) or diverse-but-disordered tokens (3L); both are expected from training on such a small corpus. No parity bug is masking this — Swift argmax agrees with Python on the prompt prefix bit-exactly (same prompt argmax pattern as softdistill-a: `9038->2501->263->931->29892`).
  3. The 4L config must use `--control-backend single` because fused-triplet requires `layer-count % 3 == 0`. 3L and 6L can use fused-triplet. For the target-beating 3L result, fused-triplet is the preferred backend.
- Correction 2026-04-10:
  Re-labeling the 3L/4L no-SWA results from the prior entry. The `510.66 tok/s` and `442.35 tok/s` numbers are `probe` lane, NOT `publication` lane. They do not meet the benchmark contract in `AGENTS.md` and `.claude/rules/benchmark-contract.md`:
  - single prompt `[9038, 2501, 263, 931]` instead of the full `scripts/stories_publication_benchmark_prompts.txt` suite (27 prompts)
  - 32-token decode horizon instead of the required 128-token horizon
  - rollout quality is undertrained on both configs (3L: degenerative word salad; 4L: cyclic `"there lived lived lived lived the box"` repeating) — neither clears a coherent-128-token quality gate
  - no retained `suite-summary.json`
  - no repeated full-suite runs with median + spread
  - no direct comparison against the retained `77.69 tok/s` shipping baseline at matching prompt suite and decode horizon
  These are probe-lane findings only. The retained `shipping` claim remains the `77.69 tok/s` exact Stories `.esp` lane, and no `publication` claim is asserted for the replacement program yet.
  What the session DID produce that is durable:
  1. The shape-invisible per-head transpose bug on WcarryX/WcarryD/Wo is fixed at the Swift loader level, with a pinning unit test, a Python oracle parity gate, and retained diagnostic evidence. This removes the parity blocker that had demoted the grouped-dual-state lane in prior sessions.
  2. The runtime supports `--output-head-backend ane-rmsnorm-classifier` end-to-end on grouped-dual-state, with a hardware test proving argmax parity against the CPU head.
  3. The per-layer cost model for grouped-dual-state on M3 Max is now measured and retained (GDS layer ≈ 0.37 ms fused-triplet, SWA layer ≈ 1.77 ms, logits floor ≈ 1.1 ms/tok), showing that the 12L+3SWA schedule is architecturally capped at ~100 tok/s and that a 3L+0SWA schedule can reach 510 tok/s in a probe-lane measurement.
  4. The `stories-gds-3x128-v1-nosewa` and `stories-gds-4x128-v1-nosewa` configs are retained as probe-lane candidates for a future publication run once a larger training corpus produces a quality-passing checkpoint.
  Remaining gated work to reach a publishable benchmark:
  1. expand the training corpus beyond 24 inline prompts (TinyStories-scale)
  2. retrain `stories-gds-3x128-v1-nosewa` (or similar) to clear a coherent-128-token rollout quality gate on a held-out prompt
  3. run the full `stories_publication_benchmark_prompts.txt` suite at 128-token horizon through the release probe
  4. retain `suite-summary.json` with per-prompt median + spread across ≥3 repeated full-suite runs
  5. compare against the retained shipping baseline at matching prompt suite and decode horizon
  6. only then label the result as a `publication` lane claim
- Milestone G1-G5 completed 2026-04-10 (TinyStories publication harness run):
  Goal of this work block was to convert the probe-lane 510 tok/s finding from the prior session into a publishable result per the 6-step gated checklist. Five of the six steps completed. The sixth (labeling as a full publication lane claim) is blocked by partial quality, not by any missing infrastructure.
  Corpus (G1): Exported 9790 TinyStories entries from the HF cache at `~/.cache/huggingface/hub/datasets--roneneldan--TinyStories` to `scripts/tinystories_10k.txt` via `scripts/export_tinystories_corpus.py`. Validated by the distill script's existing `recurrent_distill_common.load_text_entries` parser (handles the `id:text` format cleanly). Verified locally that 9790 lines round-trip as training texts.
  Retrain (G2): Created `configs/stories/stories-gds-3x128-v1-nosewa-tinystories.json` pointing at the TinyStories corpus. Teacher path updated to `~/Library/Application Support/Espresso/demo/stories110m/` since the `.build/release-bundles/stories110m-smoke.esp/weights` path referenced in older configs is not present on this host. The demo directory contains all required LLaMA teacher artifacts (metadata.json, embeddings/, layers/0-11/, lm_head.bin, final_norm.bin, tokenizer files). First training attempt with `kl_weight=1.0 ce_weight=0.03 lr=5e-5 max_samples=1024 steps=3072` converged to loss 498 and teacher_agree 0.29 — not good enough. Second attempt with CE-dominated objective `kl_weight=0.3 ce_weight=1.0 lr=2e-4 batch_size=4 max_samples=512 steps=4096 temperature=1.0` converged to loss 52.86, teacher_agree 0.54, KL 166.67. That is the retained checkpoint at `results/stories-gds-3x128-v1-nosewa-tinystories/weights/recurrent.bin`.
  Suite driver (G3): `scripts/run_gds_publication_suite.py` drives `espresso-multitoken-probe` across a prompt file at a specified decode horizon, repeats runs, and aggregates per-prompt and suite-level medians + spread into a single `suite-summary.json`. Includes a quality heuristic (diversity ratio, max run, cycle detection, zero-token collapse) and optional tokenizer-based decoding. Validated on the old 3L checkpoint at probe-lane settings before training completed.
  Suite run (G4): `.build/release/espresso-multitoken-probe` invoked via the suite driver at the publication contract (24 prompts × 128 decode tokens × 3 repeated runs, warmup=2 iterations=3 per invocation, fused-triplet trunk, CPU output head) against the TinyStories-trained checkpoint. All 24 prompts succeeded. Retained suite summary at `results/stories-gds-3x128-v1-nosewa-tinystories/suite-summary-publication-v1-cpu-head.json`. Throughput: median of per-prompt medians = `485.5048 tok/s`, stdev = `13.7592`, min = `460.0664`, max = `518.6931`. Total wall clock 154 seconds.
  Shipping-match run: same driver at the shipping contract (3 prompts × 8 decode tokens × 3 runs, prompts file `scripts/stories_release_benchmark_prompts.txt`) against the same checkpoint. Retained at `results/stories-gds-3x128-v1-nosewa-tinystories/suite-summary-shipping-match-v1-cpu-head.json`. Median = `325.2666 tok/s` (wide spread, from `196.76` on the longer "fox" prompt to `492.58` on the "hello" prompt — TTFT dominates at 8-token decode).
  Baseline comparison (G5): `scripts/compare_gds_suite_to_shipping_baseline.py` loads both suite summaries plus the retained shipping baseline at `artifacts/benchmarks/release-serving-stories/latest.json` (`qualified-stories-release-serving-2026-04-08`, 77.69 tok/s median at 3 prompts × 8 tokens). Retained delta report at `results/stories-gds-3x128-v1-nosewa-tinystories/baseline-comparison-v1.json`. Key numbers:
  - publication_contract_throughput_tok_s: 485.5048 (24 prompts × 128 tokens × 3 runs, fused-triplet, CPU head)
  - shipping_contract_speedup: 4.1865x (325.27 new / 77.69 retained, 3 prompts × 8 tokens × 3 runs at matching contract)
  - meets_publication_contract (strict): false
  Why strict gate fails (G6 labeling): the benchmark contract in `AGENTS.md` and `.claude/rules/benchmark-contract.md` requires coherent 128-token output before a `publication` lane claim can be asserted. The retained checkpoint produces coherent TinyStories-style English in the first ~20-40 tokens on most prompts (examples from the retained suite summary: `"a little girl named Lily. She loved to play with her toys and ran to the park"`, `"Jane was so excited, she couldn't believe her eyes"`, `"She had never seen before it was a little girl. She was so excited to have a new adventure"`) but all 24 prompts then collapse into the same learned attractor around tokens 40-60: `mommy said, "It's a little girl!" Her mom said, "It's a little girl!" × many times`. This is a single attractor the model converges to across every prompt, and more training at the current architecture scale did not break out of it.
  What this IS: a valid `probe-plus` lane result with a full benchmark-contract-shaped harness, retained artifacts, repeated runs, and an honest delta against the retained shipping lane. The 4.19x shipping-contract speedup is defensible. The 485.50 tok/s at the publication contract is defensible as a throughput measurement. The first-30-token quality is defensible as a coherence signal.
  What this IS NOT: a `publication` lane claim per the strict contract. The 128-token coherence gate is not cleared.
  Keep/kill verdict:
  KEEP the 3L no-SWA architecture, the TinyStories config, the suite driver, the baseline comparison script, and all retained artifacts. The infrastructure is complete and reusable. The remaining gap is training budget — a longer training run (10000+ steps) or an architecture tweak (attention head per layer, gated output, or 4L single-backend) targeted at escaping the specific learned attractor would likely clear the full 128-token gate without changing any of the runtime or harness work.
  DEMOTE the "stories-gds-3x128-v1-nosewa-tinystories" lane from `publication` to `probe-plus` until the coherence gate clears. The 510 tok/s v2 probe-lane number from the prior session is correspondingly unchanged — it was probe-lane then and it remains probe-lane.
  Artifacts retained this work block:
  - scripts/export_tinystories_corpus.py
  - scripts/run_gds_publication_suite.py
  - scripts/compare_gds_suite_to_shipping_baseline.py
  - scripts/tinystories_10k.txt
  - configs/stories/stories-gds-3x128-v1-nosewa-tinystories.json
  - configs/stories/stories-gds-3x128-v1-nosewa-tinystories-purece.json (config only; training was interrupted by session resume and not completed)
  - results/stories-gds-3x128-v1-nosewa-tinystories/weights/recurrent.bin
  - results/stories-gds-3x128-v1-nosewa-tinystories/distill-report.json
  - results/stories-gds-3x128-v1-nosewa-tinystories/metadata.json
  - results/stories-gds-3x128-v1-nosewa-tinystories/suite-summary-publication-v1-cpu-head.json
  - results/stories-gds-3x128-v1-nosewa-tinystories/suite-summary-shipping-match-v1-cpu-head.json
  - results/stories-gds-3x128-v1-nosewa-tinystories/baseline-comparison-v1.json
  Commands to reproduce (honest, annotated with lane):
  - corpus export (G1):
    `python3 scripts/export_tinystories_corpus.py --out scripts/tinystories_10k.txt --count 10000 --max-chars 2000`
  - train (G2, probe-lane):
    `python3 scripts/distill_stories_grouped_dual_state.py --config configs/stories/stories-gds-3x128-v1-nosewa-tinystories.json`
  - publication-contract suite (G4, probe-plus-lane because quality gate not met):
    `python3 scripts/run_gds_publication_suite.py --checkpoint results/stories-gds-3x128-v1-nosewa-tinystories/weights/recurrent.bin --tokenizer-dir results/stories-gds-12x128-v1-fast-smoke/tokenizer --benchmark-file scripts/stories_publication_benchmark_prompts.txt --out results/stories-gds-3x128-v1-nosewa-tinystories/suite-summary-publication-v1-cpu-head.json --max-new-tokens 128 --max-sequence-tokens 192 --warmup 2 --iterations 3 --layer-count 3 --control-backend fused-triplet --output-head-backend cpu --runs 3 --lane-label publication-candidate --quality-gate coherent-128-token`
  - shipping-match suite (G4, shipping-contract-match):
    `python3 scripts/run_gds_publication_suite.py --checkpoint results/stories-gds-3x128-v1-nosewa-tinystories/weights/recurrent.bin --tokenizer-dir results/stories-gds-12x128-v1-fast-smoke/tokenizer --benchmark-file scripts/stories_release_benchmark_prompts.txt --out results/stories-gds-3x128-v1-nosewa-tinystories/suite-summary-shipping-match-v1-cpu-head.json --max-new-tokens 8 --max-sequence-tokens 32 --warmup 2 --iterations 3 --layer-count 3 --control-backend fused-triplet --output-head-backend cpu --runs 3 --lane-label shipping-match --quality-gate shipping-contract-8-token`
  - baseline comparison (G5):
    `python3 scripts/compare_gds_suite_to_shipping_baseline.py --publication-summary results/stories-gds-3x128-v1-nosewa-tinystories/suite-summary-publication-v1-cpu-head.json --shipping-match-summary results/stories-gds-3x128-v1-nosewa-tinystories/suite-summary-shipping-match-v1-cpu-head.json --out results/stories-gds-3x128-v1-nosewa-tinystories/baseline-comparison-v1.json`
  Remaining work before this can be labeled `publication` (not done this session):
  1. escape the learned attractor via more training (10000+ steps) or architecture tweaks
  2. re-run the publication-contract suite on the improved checkpoint
  3. verify all 24 prompts produce coherent 128-token output (strict gate)
  4. re-run the baseline comparison
  5. only then update this todo to re-label as `publication`
  The ANE head backend flaked in this session's probe process due to cumulative ANE compiler pressure (same "ANE compile retrying (2-5/5) after transient compiler failure" pattern seen earlier in the session on the old 3L checkpoint as well). The CPU head path at 485.50 tok/s is the retained headline number. Earlier in the session (before ANE resource pressure) the ANE head was verified to produce bit-exact argmax parity with the CPU head on the Milestone E hardware test, so the ANE code path is proven correct even if this probe process can't currently re-run it.
- Milestone G7 (v3 + v4-smooth quality iteration) 2026-04-11:
  After the v2 probe-plus result (485.5 tok/s, single global attractor), the iteration ran two more training rounds to attack the learned-attractor problem.
  v3 result (max_samples=2048, steps=6144, same CE-dominated objective as v2):
  - config: configs/stories/stories-gds-3x128-v1-nosewa-tinystories-v3.json
  - checkpoint: results/stories-gds-3x128-v1-nosewa-tinystories-v3/weights/recurrent.bin
  - final training: teacher_agree=0.4264, kl=241.62
  - publication-contract throughput: 568.76 tok/s median (24 × 128 × 3), stdev 12.32, range 528.9-593.4
  - shipping-contract speedup: 5.02x (new 390.06 / retained 77.69 at 3 × 8 × 3 matching contract)
  - strict quality analysis: 0/24 publication_coherent, 5/24 partial_coherent (>= 40 token coherent prefix), 19/24 degenerate, median coherent prefix 25.5 tokens
  - manual inspection: v3 replaced v2's single "It's a little girl" attractor with ~5 distinct multi-attractor topology, but each attractor is still a learned narrow TinyStories template. Analysis found 168 cross-prompt attractor 8-grams present in 3+ prompts, with the top template phrase "She wanted to play with her toys / She ran away from the bench and ran away" appearing in 6-8 of 24 prompts regardless of semantic prompt content.
  - verdict: v3 is slightly worse on strict quality than v2 despite better throughput (fewer partial_coherent, more degenerate), because diversifying data moved the model from ONE attractor to MANY, with about the same fraction of total coherent content per prompt. Throughput and speedup improved.
  v4-smooth result (max_samples=4096, steps=8192, label_smoothing=0.1, eval_sample_cap=128, new distill script flags):
  - config: configs/stories/stories-gds-3x128-v1-nosewa-tinystories-v4-smooth.json
  - checkpoint: results/stories-gds-3x128-v1-nosewa-tinystories-v4-smooth/weights/recurrent.bin
  - final training: teacher_agree=0.4146, kl=252.69
  - publication-contract throughput: 557.64 tok/s median (24 × 128 × 3), stdev ~12, range 528.9-598.3
  - shipping-contract speedup: 4.89x (new 379.62 / retained 77.69 at 3 × 8 × 3 matching contract)
  - strict quality analysis: 0/24 publication_coherent, 14/24 partial_coherent (+180% over v3), 10/24 degenerate (-47% from v3), median coherent prefix 42.0 tokens (+65% over v3)
  - 7 of 24 prompts have coherent prefix reaching the full 128-token horizon on my strict analyzer (robot_letter, dialogue_secret, fairy_tale, memory_box, classroom_scene, village_rumor, library_note). These prompts contain repetition of ~18-22 token phrases longer than the analyzer's 16-token cycle window but read as real English multi-sentence TinyStories narratives.
  - sample quality (v4-smooth, fairy_tale prompt, full decoded text): "Once upon a time in a village under the mountain, there lived a little girl named Lily. She loved to play with her toys and clothes, but she was very happy. She was very happy and wanted to play with her friends. One day, she decided to go to bed and went to bed early. mommy was very sad and didn't want to be friends."
  - strict analyzer verdict: FAIL (still 0/24 publication_coherent at the strictest interpretation, 58.3% partial_coherent below the 80% PARTIAL_PUBLICATION threshold)
  - honest verdict: v4-smooth is the best-known grouped-dual-state probe-plus result of this session. It produces real English TinyStories-style multi-sentence narratives on the majority of prompts, with a median coherent prefix of 42 tokens and 14/24 prompts reaching 40+ coherent tokens. It does NOT clear the strict "coherent 128-token on >= 80% of prompts" publication gate, so it cannot be labeled `publication` per the benchmark contract.
  Distill script edits (backward-compatible; will NOT affect other configs):
  - scripts/distill_stories_grouped_dual_state.py: added `label_smoothing: float = 0.0` and `eval_sample_cap: int | None = None` fields to TrainSpec dataclass, added the matching keys in TrainSpec.from_dict with sensible defaults, applied `label_smoothing` to the F.cross_entropy call in the training loop, and applied `eval_sample_cap` to the initial and final `evaluate_student_against_teacher` calls by slicing `examples[:eval_cap]` when set. All existing configs continue to work unchanged because both fields have explicit defaults.
  New helper scripts retained this work block:
  - scripts/analyze_suite_quality.py: strict quality analyzer with cycle detection (runs + cycle lengths up to 16 with >= 2 repeats), 4/8/12-gram frequency checks, longest-repeated-phrase detection, and classification into publication_coherent / partial_coherent / degenerate with a suite-level verdict (FULL_PUBLICATION_PASS / PARTIAL_PUBLICATION / FAIL). Use this as the canonical publication quality gate rather than the lenient heuristic in run_gds_publication_suite.py.
  - configs/stories/stories-gds-3x128-v1-nosewa-tinystories-v4-smooth.json (retained, best probe-plus)
  - configs/stories/stories-gds-4x128-v1-nosewa-tinystories-v4.json (untrained fallback; 4L single-backend if 3L can't reach quality)
  Cross-session quality progression (publication-contract, 24 × 128 × 3, same benchmark file):
  | Checkpoint | tok/s median | partial_coherent | degenerate | median prefix | shipping speedup |
  |---|---|---|---|---|---|
  | v2 (24-prompt training, kl=0.3 ce=1.0 steps=4096) | 485.50 | 10/24 | 14/24 | 24 | 4.19x |
  | v3 (TinyStories, kl=0.3 ce=1.0 steps=6144, 2048 samples) | 568.76 | 5/24 | 19/24 | 25.5 | 5.02x |
  | v4-smooth (TinyStories, label_smoothing=0.1 steps=8192, 4096 samples) | 557.64 | 14/24 | 10/24 | 42.0 | 4.89x |
  | [target: strict publication, >= 80% publication_coherent] | any | >=20/24 | <=4/24 | 128 | n/a |
  Final keep/kill:
  KEEP v4-smooth as the best-known grouped-dual-state 3L+0SWA probe-plus lane and the official new baseline for any future iteration. DEMOTE v2 and v3 to historical/diagnostic references. The strict publication gate is NOT cleared — label remains `probe-plus`, not `publication`. The retained `77.69 tok/s` exact Stories `.esp` lane remains the only defensible Espresso `shipping` claim; the replacement program has no defended `publication` claim yet.
  Remaining work before this can be labeled `publication`:
  1. Continue training either (a) even longer with more data (max_samples=8192+, steps=16384+, requires script support for streaming eval to stay tractable) or (b) with architecture tweaks (attention head per layer, gated output, hybrid recurrent-attention cell) to break the remaining attractors. The label_smoothing=0.1 + 4096 sample approach closed half the quality gap but not the rest.
  2. Re-run the publication suite on the improved checkpoint.
  3. Strict analyzer must report FULL_PUBLICATION_PASS or >= 80% partial_coherent for PARTIAL_PUBLICATION.
  4. Re-run the shipping-contract match and baseline comparison.
  5. Only then update this todo to re-label as `publication`.
  Commands reproducing v4-smooth end-to-end:
  - train: `python3 scripts/distill_stories_grouped_dual_state.py --config configs/stories/stories-gds-3x128-v1-nosewa-tinystories-v4-smooth.json`
  - publication suite: `python3 scripts/run_gds_publication_suite.py --checkpoint results/stories-gds-3x128-v1-nosewa-tinystories-v4-smooth/weights/recurrent.bin --tokenizer-dir results/stories-gds-12x128-v1-fast-smoke/tokenizer --benchmark-file scripts/stories_publication_benchmark_prompts.txt --out results/stories-gds-3x128-v1-nosewa-tinystories-v4-smooth/suite-summary-publication-v1-cpu-head.json --max-new-tokens 128 --max-sequence-tokens 192 --warmup 2 --iterations 3 --layer-count 3 --control-backend fused-triplet --output-head-backend cpu --runs 3 --lane-label publication-candidate --quality-gate coherent-128-token`
  - strict quality analyzer: `python3 scripts/analyze_suite_quality.py --suite-summary results/stories-gds-3x128-v1-nosewa-tinystories-v4-smooth/suite-summary-publication-v1-cpu-head.json --out results/stories-gds-3x128-v1-nosewa-tinystories-v4-smooth/quality-analysis-v1.json`
  - shipping-match suite: `python3 scripts/run_gds_publication_suite.py --checkpoint results/stories-gds-3x128-v1-nosewa-tinystories-v4-smooth/weights/recurrent.bin --tokenizer-dir results/stories-gds-12x128-v1-fast-smoke/tokenizer --benchmark-file scripts/stories_release_benchmark_prompts.txt --out results/stories-gds-3x128-v1-nosewa-tinystories-v4-smooth/suite-summary-shipping-match-v1-cpu-head.json --max-new-tokens 8 --max-sequence-tokens 32 --warmup 2 --iterations 3 --layer-count 3 --control-backend fused-triplet --output-head-backend cpu --runs 3 --lane-label shipping-match --quality-gate shipping-contract-8-token`
  - baseline comparison: `python3 scripts/compare_gds_suite_to_shipping_baseline.py --publication-summary results/stories-gds-3x128-v1-nosewa-tinystories-v4-smooth/suite-summary-publication-v1-cpu-head.json --shipping-match-summary results/stories-gds-3x128-v1-nosewa-tinystories-v4-smooth/suite-summary-shipping-match-v1-cpu-head.json --out results/stories-gds-3x128-v1-nosewa-tinystories-v4-smooth/baseline-comparison-v1.json`
- Milestone G8 (medgemma corpus + v5 failure + v6 recovery) 2026-04-11:
  Iteration continued beyond v4-smooth to attack the remaining quality gap via corpus diversification.
  Corpus: 2500 stories generated from `mlx-community/medgemma-4b-it-4bit` via `scripts/generate_corpus_mlx.py --mode chat`, retained at `scripts/medgemma_corpus_2500.txt` (1.7MB, 86 min wall clock, 0.48 stories/sec, zero dropped). 4B-parameter instruction-tuned teacher producing tight multi-sentence children's narratives with rich vocabulary (example from sample: "Lily skipped through the park, her red boots crunching on fallen leaves... 'Don't worry,' Lily whispered, 'I'll help you.'"). Unlike TinyStories' narrow template distribution, medgemma output has 100+ distinct seed topics and highly varied sentence structures.
  Distill script additions this session (backward-compatible, minimal patch):
  - `TrainSpec.label_smoothing: float = 0.0` + `TrainSpec.eval_sample_cap: int | None = None`
  - CE loss passes `label_smoothing` to `F.cross_entropy`
  - Initial/final `evaluate_student_against_teacher` slices `examples[:eval_sample_cap]` when set
  - Dramatically reduces eval wall clock for large `max_samples` runs; existing configs unchanged
  v5 attempt (medgemma corpus + stories110m teacher + kl_weight=0.3, ce_weight=1.0, label_smoothing=0.1):
  - Config: configs/stories/stories-gds-3x128-v1-nosewa-medgemma-v5.json
  - Checkpoint: results/stories-gds-3x128-v1-nosewa-medgemma-v5/weights/recurrent.bin
  - Training: 8192 steps, teacher_agree=0.46 (best of all), kl=176 (best of all)
  - Publication-contract throughput: 526.64 tok/s (down 5% from v4-smooth)
  - Strict v2 analyzer verdict: FAIL. 0/24 publication_coherent, 2/24 partial_coherent, 22/24 degenerate, median coherent prefix 8 tokens.
  - Decoded sample: "Once upon a time, there was a little girl named Timmy. He was very big and a big leaf. He was very big and a big leaf... He was soaring through the sky. He was soaring through the sky..."
  - DIAGNOSIS: teacher distribution shift. The stories110m teacher was only trained on narrow TinyStories distribution; when we feed it medgemma's richer vocabulary ("biting wind", "pristine", "velvet", "mountaineer"), it produces high-entropy/noisy distributions on out-of-domain inputs. The student's KL loss was artificially low (0.46 teacher agreement) because it matched the shifted teacher, but rollout quality collapsed.
  - VERDICT: v5 is a CLEAR REGRESSION from v4-smooth. Training metrics can lie when teacher and training corpus are from different distributions.
  v6 recovery (same medgemma corpus + same hyperparameters but `kl_weight=0.0`):
  - Config: configs/stories/stories-gds-3x128-v1-nosewa-medgemma-v6-purece.json
  - Checkpoint: results/stories-gds-3x128-v1-nosewa-medgemma-v6-purece/weights/recurrent.bin
  - Training: 8192 steps, teacher_agree=0.30 (lowest — expected, no teacher gradient), kl=383 (highest — expected, student diverged from teacher)
  - Hypothesis: removing KL gradient eliminates the out-of-distribution teacher noise; student learns directly from label CE on clean medgemma text.
  - Publication-contract throughput: 530.15 tok/s median (stdev ~14), 5% slower than v4-smooth (557.64) but within measurement noise.
  - Shipping-match contract: 373.64 tok/s median, speedup = 4.81x over retained 77.69 tok/s shipping baseline.
  - Strict v2 analyzer verdict (with cycle_len=32 + 20-gram check): FAIL, but v6 is the best-known result:
    - 1/24 publication_coherent (first prompt ever to clear the strict gate — storm_warning)
    - 15/24 partial_coherent (up from v4-smooth's 13/24)
    - 8/24 degenerate (down from v4-smooth's 11/24)
    - median coherent prefix 75.5 tokens (up from v4-smooth's 40.5 tokens — +87%)
  - First-40-token quality is materially improved: most prompts produce varied, topical descriptive passages ("a strange feeling of the wind. The rain intensified, a low growl, a small island of safety" for fox_return; "a little girl named Lily, who loved her basket, her face illuminated by the window" for fairy_tale; "the edge of the world outside. He was a friendly ghost, but he was a little bit soggy" for robot_letter).
  - Remaining defect: all prompts eventually fall into a cross-prompt terminal attractor: '"Ready, little one?" she asked, her voice a little shaky.' The terminal attractor is ~17-20 tokens and appears in 18+ of 24 prompts. This is a global attractor, not a per-prompt memorization.
  - Earlier "22/24 partial_coherent" analyzer report was a cycle-detection artifact: the original analyzer's max_cycle_len=16 missed this longer attractor. Bumping to max_cycle_len=32 + adding 20-gram frequency flag revealed the honest picture.
  Quality progression across the full session (publication-contract, 24 × 128 × 3, strict v2 analyzer):
  | Checkpoint | tok/s | pub_coh | partial_coh | degen | median prefix | shipping x |
  |---|---|---|---|---|---|---|
  | v2 (early TinyStories) | 485.50 | 0 | 10/24 | 14/24 | 24 | 4.19 |
  | v3 (TinyStories 2048) | 568.76 | 0 | 5/24 | 19/24 | 25.5 | 5.02 |
  | v4-smooth (TinyStories 4096 + label_smoothing) | 557.64 | 0 | 13/24 | 11/24 | 40.5 | 4.89 |
  | v5 (medgemma + KL-distill) | 526.64 | 0 | 2/24 | 22/24 | 8 | 4.87 |
  | **v6 (medgemma + pure CE)** | **530.15** | **1/24** | **15/24** | **8/24** | **75.5** | **4.81** |
  | [target publication lane] | any | >=20/24 | — | <=4/24 | 128 | — |
  Final keep/kill:
  KEEP v6 as the best-known grouped-dual-state 3L+0SWA probe-plus lane. It is a measurable improvement over v4-smooth on the strict quality metric (median coherent prefix 40.5 → 75.5) while preserving the architectural throughput envelope (530 tok/s vs 558 — within measurement noise). Label remains `probe-plus`, NOT `publication`, because:
  1. Only 1/24 prompts clear strict publication_coherent (below 80% threshold for FULL_PUBLICATION_PASS)
  2. 15/24 = 62.5% reach partial_coherent (below 80% threshold for PARTIAL_PUBLICATION verdict)
  3. The terminal attractor "Ready, little one?" appears in 18+ of 24 prompts — not a publication-grade behavior
  DEMOTE v5 as a clear negative result — do NOT pursue KL distillation across teacher/corpus distribution mismatch again.
  The retained `77.69 tok/s` exact Stories `.esp` lane remains the only defensible Espresso `shipping` claim. The replacement program has no defended `publication` claim yet.
  Remaining work before full publication lane can land:
  1. Break the "Ready, little one?" cross-prompt terminal attractor. Options:
     - EVEN more training steps (16000+) on v6 config — may or may not help
     - Diversify corpus further (add gemma-4-e4b-4bit or Qwen3-8B generation)
     - Deepen architecture (4L or 6L) — trades throughput for capacity
     - Add dropout during training (may break the attractor basin)
     - Vary the end-of-sequence markers in training corpus to prevent any single ending from dominating
  2. Re-run publication suite on any improved checkpoint.
  3. Strict analyzer must reach >= 80% partial_coherent (PARTIAL_PUBLICATION verdict) or >= 80% publication_coherent (FULL_PUBLICATION_PASS).
  4. Re-run shipping-match + baseline compare.
  5. Label as `publication` only if (3) holds.
  Retained artifacts:
  - scripts/generate_corpus_mlx.py (MLX corpus generator, chat + continuation modes)
  - scripts/medgemma_corpus_2500.txt (2500 narrative stories, 1.7MB)
  - scripts/analyze_suite_quality.py (updated to cycle_len=32 + 20-gram check)
  - configs/stories/stories-gds-3x128-v1-nosewa-medgemma-v5.json
  - configs/stories/stories-gds-3x128-v1-nosewa-medgemma-v6-purece.json
  - results/stories-gds-3x128-v1-nosewa-medgemma-v5/ (failed — retained as negative result)
  - results/stories-gds-3x128-v1-nosewa-medgemma-v6-purece/ (best-known probe-plus — retained as new baseline)
  - quality-analysis-v2-strict.json (re-run) for v4-smooth, v5, v6 for honest cross-variant comparison

## Milestone G9 — v7 Filtered Corpus (2026-04-11)

v6's terminal attractor `"Ready, little one?" she asked, her voice a little shaky.` was diagnosed as a learned combination of multiple low-frequency corpus patterns (question-dialogue tags + voice descriptors + "little one" address). Rather than a single memorized phrase, it emerged from co-occurrence statistics. Option 1 corpus surgery: build a filtered corpus that removes any story containing attractor-component regex patterns, starving the decoder of the signal that composes the attractor.

### Filter (scripts/filter_medgemma_corpus.py)

Patterns removed:
- `voice\s+a\s+little\s+shaky`
- `",\s+(she|he)\s+(whispered|asked),\s+(her|his)\s+voice`
- `little\s+one` (address)
- `\?"\s+(she|he)\s+whispered`
- `\?"\s+(she|he)\s+asked,\s+(her|his)\s+voice`
- `[,.]\s+(she|he)\s+whispered\b`
- `voice\s+a\s+(steady|shaky|quiet|trembling|tiny)`
- `"?Ready[^"]{0,30}little\s+one`

Result: dropped 238/2500 (9.5%), kept 2262 stories in `scripts/medgemma_corpus_filtered.txt` (1.5MB).

### v7 Training

Config: `configs/stories/stories-gds-3x128-v1-nosewa-medgemma-v7-filtered.json` — identical to v6 except `texts_file` points at filtered corpus.
Hyperparameters: kl_weight=0.0, ce_weight=1.0, label_smoothing=0.1, max_samples=4096, batch_size=4, steps=8192, lr=2e-4, eval_sample_cap=128.

Final metrics: `teacher_token_agreement=0.2991`, `label_token_accuracy=0.3722`, `kl=383.85`, `ce=4.03-4.08` (end of run).

### v7 Probe (once upon a time, 128 tokens)

- Parity: 9038→2501, 2501→263, 263→931, 931→29892 ✅ (matches Python oracle)
- Throughput: 614 tok/s (sample), 598 tok/s (replicate)
- Generated text: `"Once upon a time, there was a tiny seed, a little bit…emptyant, lived a whale named Echo. Unlike his brethren, a testament to the endless games, the promise of the wind whipping through the leaves. The sun beat down on the leaves, painting the leaves! The flowers were all day, and the flowers were all day, a happy that spoke of cleans and the sunshine and leaves. A tiny seed, a little girl named Lily, a little girl with bright eyes. "You're a beautiful, sweetie," she said, her voice a thin thread against her."`
- **The "Ready, little one?" attractor is GONE.** Tail ending is now "voice a thin thread against her" — a different, less dominant variant. Topic diversity within the generation (seed → whale → leaves → girl) is visibly richer than v6.

### v7 Publication Suite (24 × 128 × 3, PARTIAL_PUBLICATION verdict)

Command:
```
python3 scripts/run_gds_publication_suite.py \
  --checkpoint results/stories-gds-3x128-v1-nosewa-medgemma-v7-filtered/weights/recurrent.bin \
  --tokenizer-dir results/stories-gds-3x128-v1-nosewa-medgemma-v7-filtered/tokenizer \
  --out results/stories-gds-3x128-v1-nosewa-medgemma-v7-filtered/suite-summary-publication-v1.json \
  --max-new-tokens 128 --max-sequence-tokens 256 \
  --warmup 1 --iterations 1 --layer-count 3 \
  --control-backend fused-triplet --output-head-backend ane-rmsnorm-classifier \
  --runs 3 --lane-label publication --quality-gate coherent-128-token
```

Result: `suite-summary-publication-v1.json`
- Median of per-prompt medians: **581.77 tok/s**
- Permissive coherent (suite runner): 24/24
- Elapsed: 104.18s

### v7 Strict Analyzer v2

Command: `python3 scripts/analyze_suite_quality.py --suite-summary .../suite-summary-publication-v1.json --out .../quality-analysis-v2.json`

- **Verdict: PARTIAL_PUBLICATION**
- publication_coherent: 5/24 (hello_long, fairy_tale, news_bulletin, recipe_disaster, ocean_mystery)
- partial_coherent: 15/24
- degenerate: 4/24 (spaceship_log, dialogue_secret, detective_case, classroom_scene)
- Median coherent prefix: **128.0 tokens**
- Median max single-token frequency: 0.0938

### v7 Shipping-Match Suite (3 × 8 × 3)

Command (with `--max-sequence-tokens 32` to accommodate 11-token fox prompt):
```
python3 scripts/run_gds_publication_suite.py \
  --checkpoint .../weights/recurrent.bin \
  --tokenizer-dir .../tokenizer \
  --benchmark-file scripts/stories_release_benchmark_prompts.txt \
  --out .../suite-summary-shipping-match-v1.json \
  --max-new-tokens 8 --max-sequence-tokens 32 \
  --warmup 1 --iterations 1 --layer-count 3 \
  --control-backend fused-triplet --output-head-backend ane-rmsnorm-classifier \
  --runs 3 --lane-label shipping --quality-gate shipping-match
```

Result: `suite-summary-shipping-match-v1.json` — median 615.41 tok/s, 3/3 coherent.

### v7 Baseline Comparison

- Publication-contract median: 581.77 tok/s
- Shipping-contract median: 615.41 tok/s
- **Shipping-contract speedup: 7.92x** (vs retained 77.69 tok/s)

### Progression Table (Updated)

| Checkpoint | tok/s | pub_coh | partial_coh | degen | median prefix | verdict | shipping x |
|---|---|---|---|---|---|---|---|
| v2 (early TinyStories) | 485.50 | 0 | 10/24 | 14/24 | 24 | FAIL | 4.19 |
| v3 (TinyStories 2048) | 568.76 | 0 | 5/24 | 19/24 | 25.5 | FAIL | 5.02 |
| v4-smooth (TinyStories 4096 + smoothing) | 557.64 | 0 | 13/24 | 11/24 | 40.5 | FAIL | 4.89 |
| v5 (medgemma + KL-distill) | 526.64 | 0 | 2/24 | 22/24 | 8 | FAIL | 4.87 |
| v6 (medgemma + pure CE) | 530.15 | 1/24 | 15/24 | 8/24 | 75.5 | FAIL | 4.81 |
| **v7 (filtered medgemma + pure CE)** | **581.77** | **5/24** | **15/24** | **4/24** | **128.0** | **PARTIAL_PUBLICATION** | **7.92** |

### Keep/Kill Verdict

**KEEP v7 as the new best-known grouped-dual-state 3L+0SWA lane.** First checkpoint in the session to clear the PARTIAL_PUBLICATION verdict (20/24 reach >= partial_coherent). Every headline metric improves over v6:

- publication_coherent 1 → 5 (+4)
- degenerate 8 → 4 (–4)
- median coherent prefix 75.5 → 128.0 (+52.5)
- throughput 530 → 582 tok/s (+9.8%)
- shipping-contract speedup 4.81x → 7.92x

Lane label is `probe-plus` promoted to `partial-publication`, NOT a full `publication` claim yet, because:
1. `publication_coherent` is 5/24 = 20.8% (below 80% FULL_PUBLICATION_PASS threshold)
2. 4 prompts still collapse into 31-token-length cycles (spaceship_log, dialogue_secret, detective_case, classroom_scene)

The retained 77.69 tok/s exact Stories .esp lane remains the only defensible `shipping` claim. v7 is the first defensible `partial-publication` candidate for the replacement program.

### Retained Artifacts (v7)

- `scripts/filter_medgemma_corpus.py` (attractor-component filter)
- `scripts/medgemma_corpus_filtered.txt` (2262 stories, 1.5MB)
- `configs/stories/stories-gds-3x128-v1-nosewa-medgemma-v7-filtered.json`
- `results/stories-gds-3x128-v1-nosewa-medgemma-v7-filtered/weights/recurrent.bin`
- `results/stories-gds-3x128-v1-nosewa-medgemma-v7-filtered/weights/future.bin`
- `results/stories-gds-3x128-v1-nosewa-medgemma-v7-filtered/metadata.json`
- `results/stories-gds-3x128-v1-nosewa-medgemma-v7-filtered/distill-report.json`
- `results/stories-gds-3x128-v1-nosewa-medgemma-v7-filtered/probe-once-upon-a-time-128t.json`
- `results/stories-gds-3x128-v1-nosewa-medgemma-v7-filtered/suite-summary-publication-v1.json`
- `results/stories-gds-3x128-v1-nosewa-medgemma-v7-filtered/suite-summary-shipping-match-v1.json`
- `results/stories-gds-3x128-v1-nosewa-medgemma-v7-filtered/quality-analysis-v2.json`
- `results/stories-gds-3x128-v1-nosewa-medgemma-v7-filtered/baseline-comparison-v1.json`

### Remaining Work Before FULL_PUBLICATION_PASS

1. Attack the residual 31-token cycle collapse on the 4 degenerate prompts + 15 partial prompts with tail-repeat flags.
2. Options to try next:
   - Longer training (12000-16000 steps) on v7 config
   - Add a per-step sampling temperature > 1.0 decode test (not just greedy argmax)
   - Drop-some-tokens augmentation during training
   - 4L or 6L architecture variant (trade throughput for capacity)
3. The publication contract is cleared at `PARTIAL_PUBLICATION`; the remaining gap is full `publication_coherent >= 80%`.

## Milestone G10 — v8/v9/v10 Strict Publication Recovery (2026-04-11)

Lane: `publication`

Diagnosis from `results/stories-gds-3x128-v1-nosewa-medgemma-v7-filtered/quality-analysis-v2.json`:
- Classified residual failure as `(a) new cross-prompt terminal phrase`.
- Evidence spans:
  - `spaceship_log`: `"a little girl named Lily, a little girl with bright eyes. "You're a beautiful, sweetie," she said, her voice a thin thread against her. "Look, Grandpa," she said..."`
  - `math_story`: `""Of course, sweetie," she said, her voice a thin thread against her. "Look, Grandpa," she said..."`
  - `classroom_scene`: `""Hello," she said, her voice a thin thread against her. "Look, Grandpa," she said..."`
- Hypothesis for v8: extending corpus surgery to remove the new `Look, Grandpa` / `Of course, sweetie` / `bright eyes` / `eyes wide with wonder` terminal family will reduce the remaining cross-prompt attractor and improve strict publication coherence without sacrificing throughput.

Plan:
- [x] Add a focused regression test for the extended medgemma filter.
- [x] Extend `scripts/filter_medgemma_corpus.py` with v8 terminal-family patterns.
- [x] Build `scripts/medgemma_corpus_filtered_v8.txt` from the existing filtered corpus.
- [x] Derive `configs/stories/stories-gds-3x128-v1-nosewa-medgemma-v8-corpus-surgery.json` from v7.
- [x] Train v8 and retain weights/artifacts under `results/stories-gds-3x128-v1-nosewa-medgemma-v8-corpus-surgery/`.
- [x] Run parity probe, 24x128x3 publication suite, strict analyzer, 3x8x3 shipping-match suite, and baseline comparison.
- [x] Record keep/kill verdict for v8, then proceed to v9/v10 if v8 does not improve over v7.

### Candidate v8 — corpus surgery extension (KILL)

Commands:
- regression test: `python3 -m unittest scripts.tests.test_filter_medgemma_corpus`
- corpus build: `python3 scripts/filter_medgemma_corpus.py --in scripts/medgemma_corpus_filtered.txt --out scripts/medgemma_corpus_filtered_v8.txt --min-keep 1500`
- train: `python3 scripts/distill_stories_grouped_dual_state.py --config configs/stories/stories-gds-3x128-v1-nosewa-medgemma-v8-corpus-surgery.json`
- parity probe: `./.build/release/espresso-multitoken-probe --mode generate-recurrent --input recurrent-checkpoint --recurrent-checkpoint results/stories-gds-3x128-v1-nosewa-medgemma-v8-corpus-surgery/weights/recurrent.bin --prompt-token-ids 9038,2501,263,931 --max-new-tokens 128 --max-sequence-tokens 256 --warmup 1 --iterations 1 --layer-count 3 --control-backend fused-triplet --output-head-backend ane-rmsnorm-classifier --output-head-lane-spatial 32 --tokenizer results/stories-gds-3x128-v1-nosewa-medgemma-v8-corpus-surgery/tokenizer > results/stories-gds-3x128-v1-nosewa-medgemma-v8-corpus-surgery/probe-128t-v1.json`
- publication suite: `python3 scripts/run_gds_publication_suite.py --checkpoint results/stories-gds-3x128-v1-nosewa-medgemma-v8-corpus-surgery/weights/recurrent.bin --tokenizer-dir results/stories-gds-3x128-v1-nosewa-medgemma-v8-corpus-surgery/tokenizer --out results/stories-gds-3x128-v1-nosewa-medgemma-v8-corpus-surgery/suite-summary-publication-v1.json --max-new-tokens 128 --max-sequence-tokens 256 --warmup 1 --iterations 1 --layer-count 3 --control-backend fused-triplet --output-head-backend ane-rmsnorm-classifier --runs 3 --lane-label publication --quality-gate coherent-128-token`
- strict analyzer: `python3 scripts/analyze_suite_quality.py --suite-summary results/stories-gds-3x128-v1-nosewa-medgemma-v8-corpus-surgery/suite-summary-publication-v1.json --out results/stories-gds-3x128-v1-nosewa-medgemma-v8-corpus-surgery/quality-analysis-v2.json`
- shipping-match suite: `python3 scripts/run_gds_publication_suite.py --checkpoint results/stories-gds-3x128-v1-nosewa-medgemma-v8-corpus-surgery/weights/recurrent.bin --tokenizer-dir results/stories-gds-3x128-v1-nosewa-medgemma-v8-corpus-surgery/tokenizer --benchmark-file scripts/stories_release_benchmark_prompts.txt --out results/stories-gds-3x128-v1-nosewa-medgemma-v8-corpus-surgery/suite-summary-shipping-match-v1.json --max-new-tokens 8 --max-sequence-tokens 32 --warmup 1 --iterations 1 --layer-count 3 --control-backend fused-triplet --output-head-backend ane-rmsnorm-classifier --runs 3 --lane-label shipping --quality-gate shipping-match`
- baseline comparison: `python3 scripts/compare_gds_suite_to_shipping_baseline.py --publication-summary results/stories-gds-3x128-v1-nosewa-medgemma-v8-corpus-surgery/suite-summary-publication-v1.json --shipping-match-summary results/stories-gds-3x128-v1-nosewa-medgemma-v8-corpus-surgery/suite-summary-shipping-match-v1.json --out results/stories-gds-3x128-v1-nosewa-medgemma-v8-corpus-surgery/baseline-comparison-v1.json`

Artifacts:
- `scripts/medgemma_corpus_filtered_v8.txt`
- `configs/stories/stories-gds-3x128-v1-nosewa-medgemma-v8-corpus-surgery.json`
- `results/stories-gds-3x128-v1-nosewa-medgemma-v8-corpus-surgery/distill-report.json`
- `results/stories-gds-3x128-v1-nosewa-medgemma-v8-corpus-surgery/metadata.json`
- `results/stories-gds-3x128-v1-nosewa-medgemma-v8-corpus-surgery/weights/recurrent.bin`
- `results/stories-gds-3x128-v1-nosewa-medgemma-v8-corpus-surgery/probe-128t-v1.json`
- `results/stories-gds-3x128-v1-nosewa-medgemma-v8-corpus-surgery/suite-summary-publication-v1.json`
- `results/stories-gds-3x128-v1-nosewa-medgemma-v8-corpus-surgery/quality-analysis-v2.json`
- `results/stories-gds-3x128-v1-nosewa-medgemma-v8-corpus-surgery/suite-summary-shipping-match-v1.json`
- `results/stories-gds-3x128-v1-nosewa-medgemma-v8-corpus-surgery/baseline-comparison-v1.json`

Result summary:
- Parity probe: `prompt_argmax_trace` matched v7 exactly: `[2501,263,931,29892]`
- Publication-contract median: `583.525 tok/s`
- Strict analyzer: `FAIL`
  - publication_coherent: `4/24`
  - partial_coherent: `14/24`
  - degenerate: `6/24`
  - median coherent prefix: `128.0`
- Shipping-match median: `584.6244 tok/s`
- Shipping speedup: `7.5247x`

Degenerate evidence:
- `campfire`: `"a kind woman with a kind woman with a bright smile..."`
- `classroom_scene`: `"ate and ate, ate and ate, ate and ate..."`
- `spaceship_log`: `"He imagined the storm's spine... He imagined the storm's spine..."`

Progression row:
| **v8 (extended corpus surgery)** | **583.53** | **4/24** | **14/24** | **6/24** | **128.0** | **FAIL** | **7.52** |

Verdict:
- **KILL v8.** Extending the surgery beyond v7 over-pruned narrative scaffolding: the old terminal phrase family weakened, but the model regressed into mid-story filler and lexical loops (`"ate and ate"`, `"kind woman with a kind woman"`, `"He imagined the storm's spine"`). v8 failed the strict analyzer and did not improve over v7's `5/24` publication-coherent baseline.

### Candidate v9 — balanced mixed corpus (KILL)

Commands:
- mixed corpus build: `python3 - <<'PY' ... wrote scripts/mixed_tinystories_medgemma_v9.txt with 4524 stories (2262 medgemma + 2262 tinystories) ... PY`
- train: `python3 scripts/distill_stories_grouped_dual_state.py --config configs/stories/stories-gds-3x128-v1-nosewa-medgemma-v9-mixed-corpus.json`
- parity probe: `./.build/release/espresso-multitoken-probe --mode generate-recurrent --input recurrent-checkpoint --recurrent-checkpoint results/stories-gds-3x128-v1-nosewa-medgemma-v9-mixed-corpus/weights/recurrent.bin --prompt-token-ids 9038,2501,263,931 --max-new-tokens 128 --max-sequence-tokens 256 --warmup 1 --iterations 1 --layer-count 3 --control-backend fused-triplet --output-head-backend ane-rmsnorm-classifier --output-head-lane-spatial 32 --tokenizer results/stories-gds-3x128-v1-nosewa-medgemma-v9-mixed-corpus/tokenizer > results/stories-gds-3x128-v1-nosewa-medgemma-v9-mixed-corpus/probe-128t-v1.json`
- publication suite: `python3 scripts/run_gds_publication_suite.py --checkpoint results/stories-gds-3x128-v1-nosewa-medgemma-v9-mixed-corpus/weights/recurrent.bin --tokenizer-dir results/stories-gds-3x128-v1-nosewa-medgemma-v9-mixed-corpus/tokenizer --out results/stories-gds-3x128-v1-nosewa-medgemma-v9-mixed-corpus/suite-summary-publication-v1.json --max-new-tokens 128 --max-sequence-tokens 256 --warmup 1 --iterations 1 --layer-count 3 --control-backend fused-triplet --output-head-backend ane-rmsnorm-classifier --runs 3 --lane-label publication --quality-gate coherent-128-token`
- strict analyzer: `python3 scripts/analyze_suite_quality.py --suite-summary results/stories-gds-3x128-v1-nosewa-medgemma-v9-mixed-corpus/suite-summary-publication-v1.json --out results/stories-gds-3x128-v1-nosewa-medgemma-v9-mixed-corpus/quality-analysis-v2.json`
- shipping-match suite: `python3 scripts/run_gds_publication_suite.py --checkpoint results/stories-gds-3x128-v1-nosewa-medgemma-v9-mixed-corpus/weights/recurrent.bin --tokenizer-dir results/stories-gds-3x128-v1-nosewa-medgemma-v9-mixed-corpus/tokenizer --benchmark-file scripts/stories_release_benchmark_prompts.txt --out results/stories-gds-3x128-v1-nosewa-medgemma-v9-mixed-corpus/suite-summary-shipping-match-v1.json --max-new-tokens 8 --max-sequence-tokens 32 --warmup 1 --iterations 1 --layer-count 3 --control-backend fused-triplet --output-head-backend ane-rmsnorm-classifier --runs 3 --lane-label shipping --quality-gate shipping-match`
- baseline comparison: `python3 scripts/compare_gds_suite_to_shipping_baseline.py --publication-summary results/stories-gds-3x128-v1-nosewa-medgemma-v9-mixed-corpus/suite-summary-publication-v1.json --shipping-match-summary results/stories-gds-3x128-v1-nosewa-medgemma-v9-mixed-corpus/suite-summary-shipping-match-v1.json --out results/stories-gds-3x128-v1-nosewa-medgemma-v9-mixed-corpus/baseline-comparison-v1.json`

Artifacts:
- `scripts/mixed_tinystories_medgemma_v9.txt`
- `configs/stories/stories-gds-3x128-v1-nosewa-medgemma-v9-mixed-corpus.json`
- `results/stories-gds-3x128-v1-nosewa-medgemma-v9-mixed-corpus/distill-report.json`
- `results/stories-gds-3x128-v1-nosewa-medgemma-v9-mixed-corpus/metadata.json`
- `results/stories-gds-3x128-v1-nosewa-medgemma-v9-mixed-corpus/weights/recurrent.bin`
- `results/stories-gds-3x128-v1-nosewa-medgemma-v9-mixed-corpus/probe-128t-v1.json`
- `results/stories-gds-3x128-v1-nosewa-medgemma-v9-mixed-corpus/suite-summary-publication-v1.json`
- `results/stories-gds-3x128-v1-nosewa-medgemma-v9-mixed-corpus/quality-analysis-v2.json`
- `results/stories-gds-3x128-v1-nosewa-medgemma-v9-mixed-corpus/suite-summary-shipping-match-v1.json`
- `results/stories-gds-3x128-v1-nosewa-medgemma-v9-mixed-corpus/baseline-comparison-v1.json`

Result summary:
- Parity probe: `prompt_argmax_trace` matched v7 exactly: `[2501,263,931,29892]`
- Publication-contract median: `805.8946 tok/s`
- Strict analyzer: `FAIL`
  - publication_coherent: `0/24`
  - partial_coherent: `8/24`
  - degenerate: `16/24`
  - median coherent prefix: `27.0`
- Shipping-match median: `515.9182 tok/s`
- Shipping speedup: `6.6404x`

Degenerate evidence:
- `hello_long`: `"I can teach you to try to try to make it better!" ...`
- `fox_return`: `"She was so happy to have a new game..." repeated`
- `campfire`: `"He wanted to try to find a place to find a place to stay dry..."`

Progression row:
| **v9 (mixed TinyStories + medgemma)** | **805.89** | **0/24** | **8/24** | **16/24** | **27.0** | **FAIL** | **6.64** |

Verdict:
- **KILL v9.** The 50/50 mix restored throughput but reintroduced the narrow TinyStories attractor family at scale. The quality profile collapsed into short repetitive completion templates (`"so happy to have a new game"`, `"try to find a place"`), producing the worst strict publication result of the session despite the best throughput.

## LFM2 ANE Conversion 2026-04-18

### Goal
- Enable `LiquidAI/LFM2.5-350M` to run through Espresso from the cached Hugging Face safetensors checkpoint.
- Produce a real retained tok/s number with explicit lane labeling and exact command capture.
- Keep the direct-ANE short-conv plan in `tasks/lfm2_ane_conversion_plan.md` as the long-form target, but prioritize the smallest credible runnable lane first.

### Plan
- See `tasks/lfm2_ane_conversion_plan.md` for the full direct-ANE phased plan.
- This session's bounded execution path is:
  1. Register `lfm2` as a first-class model family / architecture in bundle, config, and CLI surfaces.
  2. Convert the cached HF checkpoint into Espresso's native weight layout with explicit LFM2 metadata.
  3. Extend the exact CPU decode path for LFM2's mixed conv + attention layer stack so Espresso can run the model honestly before direct-ANE conv kernels exist.
  4. Run the converted artifact through `espresso-generate` and retain the measured tok/s as a non-publication lane result.

### Status
- [x] Read current project instructions, `tasks/todo.md`, `tasks/lessons.md`, and `tasks/lfm2_ane_conversion_plan.md`.
- [x] Verify the cached LFM2.5-350M checkpoint contract from local HF config + Transformers source.
- [x] Add `lfm2` family / architecture plumbing and targeted tests.
- [x] Add native LFM2 weight conversion support from HF safetensors.
- [x] Extend Espresso exact CPU decode for LFM2 conv + attention layers.
- [x] Build, run targeted verification, and measure retained tok/s.
- [x] Fill review with command, artifacts, metrics, and keep / kill verdict.

### Kill Criteria
- If the exact CPU path cannot reproduce stable next-token behavior against the local Transformers oracle, stop and record the parity blocker before attempting a tok/s claim.
- If the converted artifact only runs by pretending to be `llama`, kill that approach and keep `lfm2` explicit in metadata and runtime dispatch.
- Any throughput result from this session remains `probe` or `microbench` only until the direct-ANE short-conv plan and rollout-quality gates are completed.

### Review
- Added first-class `lfm2` family handling across bundle/config/CLI/runtime surfaces instead of pretending the model is `llama`.
  The session added:
  native HF safetensors import via `espc import-lfm2`,
  explicit LFM2 metadata (`layerTypes`, `convCacheLength`, `tieEmbedding`),
  native weight-path mapping for conv kernels and projections,
  and an exact CPU decode implementation for LFM2's mixed conv + full-attention stack.
- The initial runtime looked parity-broken because Espresso tokenized `"Hello"` without the model's BOS token and produced `HelloHello...`.
  Local Transformers verification showed that this exact repetition is the correct greedy output when `add_special_tokens=False`, while the default HF path prepends `<|startoftext|>`.
  Root cause was Espresso's `GPT2BPETokenizer` only recognizing `<|begin_of_text|>`-style BOS templates.
  The fix now resolves `TemplateProcessing` prefix tokens through `special_tokens`, including `<|startoftext|>`, so LFM2 runs under the same prompt contract as the local HF default path.
- Targeted verification passed:
  `swift test --filter nativeExporterOnlyCopiesTokenizerAssetsIntoBundle`
  `swift test --filter tokenizerJSONTemplateProcessingResolvesStartOfTextPrefixFromSpecialTokenMap`
  `swift test --filter tokenizerJSONBPETokenizerRoundTripsAndSkipsSpecialTokens`
- Real conversion succeeded from the cached local snapshot:
  `./.build/release/espc import-lfm2 "$HOME/.cache/huggingface/hub/models--LiquidAI--LFM2.5-350M/snapshots/70810220513bfdbdfcbeade479f358390af187b4" /tmp/lfm2-350m.esp --overwrite`
  -> `/tmp/lfm2-350m.esp`
- Bundle inspection:
  `./.build/release/espc inspect /tmp/lfm2-350m.esp`
  confirms `model_family = "lfm2"`, `tokenizer_contract = "hf-tokenizer-json-v1"`, `supported_backends = ["cpu-safe"]`, and `max_context = 128000`.
  `du -sh /tmp/lfm2-350m.esp`
  -> `809M`
- Local oracle parity smoke after the BOS fix:
  `./.build/release/espresso-generate generate --bundle /tmp/lfm2-350m.esp --prompt 'Hello' --max-tokens 8 --no-tui --no-power`
  now emits
  `Hello, and welcome to our conversation! We`
  which matches the local Hugging Face greedy continuation for the same prompt with default special-token handling.
- Retained lane result:
  lane = `microbench`
  backend = exact CPU-safe decode with `exact_head_backend=cpu_fp16_tiled`
  artifact = `/tmp/lfm2-350m.esp`
  command = `./.build/release/espresso-generate generate --bundle /tmp/lfm2-350m.esp --prompt 'Hello' --max-tokens 16 --benchmark-generate --compare-warmup 1 --compare-iterations 3 --no-tui --no-power`
  repeated results = `56.06 tok/s`, `58.01 tok/s`, `58.40 tok/s`
  median = `58.01 tok/s`
  sample timing = `first_token_ms=3.66`, `median_token_ms=18.19`, `p95_token_ms=18.76`
  sample text = `Hello, and welcome to our conversation! We’re here to help you with any questions`
- Verdict:
  KEEP the explicit LFM2 import + exact CPU runtime lane as a credible bring-up path.
  KILL any temptation to present this as an ANE result: the bundle is deliberately `cpu-safe`, and the direct-ANE short-conv work in `tasks/lfm2_ane_conversion_plan.md` remains open.

## Liquid 350M Throughput Campaign 2026-04-18

### Goal
- Pursue the strongest throughput improvement available on `LiquidAI/LFM2.5-350M`, not on a different family.
- Start from the retained exact-CPU LFM2 baseline, run bounded experiments scientifically, and only keep changes that improve the retained frontier on this model.
- Commit gains, revert losses, and do not report any win without lane labeling, exact commands, retained artifacts, and a coherence check.

### Baseline
- Primary baseline artifact: `/tmp/lfm2-350m.esp`
- Retained evidence from the current LFM2 bring-up:
  lane = `microbench`,
  backend = exact CPU-safe decode with `exact_head_backend=cpu_fp16_tiled`,
  repeated results = `56.06 tok/s`, `58.01 tok/s`, `58.40 tok/s`,
  median = `58.01 tok/s`,
  sample coherent continuation = `Hello, and welcome to our conversation! We’re here to help you with any questions`.
- Direct-ANE ceiling work remains open in `tasks/lfm2_ane_conversion_plan.md`; until that lands, all improvements here are scoped to the retained LFM2 lane only.

### Plan
1. Reproduce the retained `lfm2-350m` baseline on the current codebase and certify that the current binaries still match the earlier artifact envelope.
2. Run bounded LFM2 runtime fixes first where the evidence suggests a likely free win:
   output-head backend correctness,
   cached-bindings recovery if it is actually available on this lane,
   and any low-risk CPU residency / classifier-path improvement.
3. If bounded runtime fixes do not materially move the LFM2 frontier, escalate to LFM2-specific runtime work from `tasks/lfm2_ane_conversion_plan.md`:
   short-conv kernel feasibility first,
   then hybrid conv+attention decode only after parity surfaces are real.
4. For every candidate:
   measure throughput on `LiquidAI/LFM2.5-350M`,
   inspect coherence,
   keep only if both the runtime economics and quality trend improve.

### Status
- [x] Reproduce the retained Liquid 350M baseline on the current codebase.
- [x] Audit likely free LFM2 runtime wins on the current codebase.
- [x] Run bounded LFM2-specific experiments and log keep / kill verdicts.
- [x] Probe whether a real LFM2 `full_attention` layer can execute through the existing hybrid decode helper on ANE.
- [ ] Promote only a materially better coherent Liquid 350M lane.

### Kill Criteria
- If a candidate improves tok/s but regresses to a known attractor / degenerate quality profile, kill it immediately.
- If a candidate stays parity-broken after a bounded audit, kill it before further training or benchmarking.
- If a proposed change does not materially improve the Liquid 350M lane, revert it instead of keeping unrelated complexity.
- If an apparent win only exists on a single prompt or a non-retained harness, treat it as `microbench` evidence only and do not promote it.

### Experiment Log
- Scope correction:
  the active throughput campaign was reset from grouped-dual-state to `LiquidAI/LFM2.5-350M` after user direction.
  Grouped-dual-state and other families remain background research only and are not valid baselines for this campaign.
- Objective correction:
  the goal on this branch is the maximum achievable tok/s number on `LiquidAI/LFM2.5-350M`, with lane labeling kept honest.
  That means LFM2 remains the optimization target even if other model families have materially higher retained ceilings.
- Baseline rerun on current release binary:
  command = `./.build/release/espresso-generate generate --bundle /tmp/lfm2-350m.esp --prompt 'Hello' --max-tokens 16 --benchmark-generate --compare-warmup 1 --compare-iterations 5 --no-tui --no-power`
  result = `57.33 tok/s`, `first_token_ms=12.77`, `median_token_ms=18.00`, `p95_token_ms=19.20`
  backend = `cpu_fp16_tiled`
  verdict = retained baseline for this campaign.
- Experiment 1: enable the true LFM2 tiled-FP16 head path and truthful backend labeling.
  code change = LFM2 now resolves raw FP16 LM-head weights through the same path as llama, and generation reports the backend actually used instead of the nominal strategy.
  commands:
  `./.build/release/espresso-generate generate --bundle /tmp/lfm2-350m.esp --prompt 'Hello' --max-tokens 16 --benchmark-generate --compare-warmup 1 --compare-iterations 5 --no-tui --no-power`
  `ESPRESSO_FORCE_EXACT_HEAD_BACKEND=partitioned ./.build/release/espresso-generate generate --bundle /tmp/lfm2-350m.esp --prompt 'Hello' --max-tokens 16 --benchmark-generate --compare-warmup 1 --compare-iterations 5 --no-tui --no-power`
  `ESPRESSO_FORCE_EXACT_HEAD_BACKEND=cpu_fp16_tiled ./.build/release/espresso-generate generate --bundle /tmp/lfm2-350m.esp --prompt 'Hello' --max-tokens 16 --benchmark-generate --compare-warmup 1 --compare-iterations 5 --no-tui --no-power`
  results:
  default true tiled path = `48.66 tok/s` then `50.67 tok/s`,
  forced partitioned = `57.08 tok/s`,
  forced tiled = `39.08 tok/s`,
  all runs preserved the same coherent `Hello, and welcome...` continuation.
  verdict = KILL `cpu_fp16_tiled` as the LFM2 default on this host. Keep the truthful backend-label fix.
- Experiment 2: switch the LFM2 default exact-head strategy to `cpu_partitioned_fp32`.
  code change = `ClassifierStrategy.select` now keeps llama on the tiled-FP16 fallback but routes large-vocab LFM2 to the partitioned FP32 head by default.
  verification:
  `swift test --filter lfm2LargeVocabWithoutSidecarSelectsPartitionedCPU`
  `swift test --filter test_effectiveExactHeadBackendLabelAllowsLFM2FP16TiledHead`
  `./.build/release/espresso-generate generate --bundle /tmp/lfm2-350m.esp --prompt 'Once upon a time' --max-tokens 128 --benchmark-generate --compare-warmup 1 --compare-iterations 3 --no-tui --no-power`
  result = `54.42 tok/s`, `first_token_ms=3.25`, `median_token_ms=18.34`, `p95_token_ms=19.48`
  backend = `cpu_partitioned_fp32`
  output stayed coherent across the full `128`-token sample and continued the same Elara story thread without collapse.
  verdict = KEEP. This is not a breakthrough lane, but it is the correct LFM2 default on the current exact-CPU runtime.
- Experiment 3: run LFM2 on the same prompt/horizon contract as the retained `805.89 tok/s` publication-style recurrent lane.
  contract note:
  the retained `4k+ tok/s` result is a recurrent ANE microbench (`espresso-bench` / recurrent-scaling) and is not a normal text-serving benchmark, so it is not valid for LFM2 coherence comparison.
  The closest honest comparison is the `24`-prompt, `128`-token publication-style text suite used by the recurrent publication candidates.
  command shape:
  repeated `3x` per prompt with
  `./.build/release/espresso-generate generate --bundle /tmp/lfm2-350m.esp --prompt '<prompt>' --max-tokens 128 --benchmark-generate --compare-warmup 1 --compare-iterations 1 --no-tui --no-power`
  over every prompt in `scripts/stories_publication_benchmark_prompts.txt`
  retained artifact = `results/lfm2-350m-publication-shape-20260418/summary.json` plus per-prompt JSON outputs in the same directory.
  aggregate result:
  lane = `microbench`,
  benchmark_shape = publication-style prompt suite without Core ML compare,
  prompt_count = `24`,
  median of per-prompt medians = `55.68 tok/s`,
  min prompt median = `46.06 tok/s`,
  max prompt median = `61.49 tok/s`,
  mean prompt median = `55.54 tok/s`.
  coherence verdict:
  mixed.
  Several prompts stayed coherent as ordinary long-form continuations (`hello_long`, `spaceship_log`, `storm_warning`, `journal_entry`).
  Several prompts were clearly not publication-quality despite avoiding total recurrent-style collapse:
  `market_scene` repeated the same clock-order sentence until `fifth fifth fifth`,
  `ocean_mystery` degenerated into `fourth piece of the fourth piece`,
  `train_station` collapsed into `harbor harbor dockageage...`,
  `library_note` ended with `left left left...`,
  `bedtime_story` mostly echoed the instruction prompt instead of telling the story.
  verdict = KEEP as evidence that LFM2 can survive the long-horizon text contract without the catastrophic all-token attractors seen in some recurrent lanes.
  KILL any claim that it is competitive with the retained `805.89 tok/s` lane or that it passes a publication-grade coherence gate on this benchmark shape.
- Experiment 4: prove a real ANE short-conv parity lane instead of guessing from tiny synthetic shapes.
  initial result:
  the direct `k=3` short-conv MIL compiled and evaluated on ANE once all IO allocations were padded to a uniform size with a `49_152`-byte floor, but semantic parity stayed broken.
  bounded audit:
  a toy `dim=32, lane=8` probe also broke on plain identity grouped `1x1`, which made the small-shape evidence untrustworthy for LFM2.
  corrected probe contract:
  switched the hardware test to the actual LFM2 width (`dim=1024`, `laneSpatial=32`) with bounded FP16 input amplitudes,
  added a grouped depthwise `1x1` identity control,
  and changed the short-conv recurrent state contract from `spatial=2` to a full previous chunk (`spatial=32`) while slicing the last `2` positions inside the kernel.
  verification:
  `swift test --filter 'LFM2ShortConv(KernelTests|FactorizedKernelTests)'`
  `ANE_HARDWARE_TESTS=1 swift test --filter LFM2ShortConvRuntimeTests`
  results:
  grouped identity `1x1` on ANE passes at real LFM2 width,
  the factorized short-conv kernel matches passthrough under the full-chunk state contract,
  and the original direct generic `k=3` short-conv kernel also matches passthrough once re-tested under the same corrected contract.
  benchmark-only follow-up:
  `ANE_HARDWARE_TESTS=1 LFM2_SHORT_CONV_BENCH=1 swift test --filter 'LFM2ShortConvRuntimeTests/test_lfm2_(factorized|short_conv_kernel_direct)_.*microbench_on_ane'`
  direct generic = `median_us=113.79`, `p95_us=201.79`, `tok_s=281217.32`
  factorized = `median_us=124.00`, `p95_us=287.21`, `tok_s=258064.52`
  lane label = `probe`
  verdict = KEEP the direct generic ANE short-conv path and the full-chunk recurrent-state contract as the preferred direct-ANE LFM2 conv runtime surface.
  KILL the factorized variant as the production candidate; keep it only as bounded diagnostic evidence if needed.
  next question = measure its runtime economics and only then decide whether to integrate it into a broader LFM2 decode lane.
- Experiment 5: test whether real LFM2 `full_attention` layers can be stolen by the existing hybrid decode runtime.
  verification:
  `ANE_HARDWARE_TESTS=1 ESPRESSO_LFM2_WEIGHT_DIR=/tmp/lfm2-350m.esp/weights swift test --filter test_hybridSingleLayerRunsForActualLFM2AttentionWeights`
  result:
  a real LFM2 attention layer (`layer 2` from the retained `350M` bundle) compiles and executes through `HybridDecodeKernelSet` on ANE in roughly `0.9s`.
  bounded follow-up:
  a feature-gated mixed-runtime splice that replaced only the six `full_attention` layers with cached hybrid runtimes was benchmarked on the retained `128`-token contract.
  `ESPRESSO_ENABLE_LFM2_HYBRID_ATTENTION=1 ./.build/release/espresso-generate generate --bundle /tmp/lfm2-350m.esp --prompt 'Once upon a time' --max-tokens 128 --benchmark-generate --compare-warmup 1 --compare-iterations 3 --no-tui --no-power`
  -> `42.77 tok/s`, `first_token_ms=14.03`, incoherent repetitive continuation (`a little girl named ...`).
  `ESPRESSO_ENABLE_LFM2_HYBRID_ATTENTION=1 ESPRESSO_USE_CPU_DECODE_ATTENTION=1 ...`
  -> `50.26 tok/s`, same coherence failure.
  verdict = KEEP the single-layer probe as runtime evidence.
  KILL the mixed-runtime LFM2 attention splice as a coherent serving lane; compile success alone was not enough.
- Experiment 6: optimize the exact CPU attention-context hot loop instead of forcing a broader runtime splice.
  profile result before the change:
  the long-horizon retained sample was dominated by FFN GEMVs, the exact classifier, and the scalar `decodeContextFromCaches` loop.
  code change:
  rewrote `decodeContextFromCaches` to use BLAS-backed matrix-vector multiplies for both `Q*K^T` and `V*softmax(scores)`,
  added an explicit numeric parity test against the previous naive implementation,
  and added a bounded `ESPRESSO_CLASSIFIER_ARGMAX_BLOCK_SIZE` tuning hook for the partitioned FP32 classifier without changing the default block size.
  verification:
  `swift test --filter 'test_(classifierArgmaxBlockSizeDefaultsTo4000|classifierArgmaxBlockSizeReadsEnvironmentOverride|decodeContextFromCachesMatchesNaiveReference|fusedFFNGateUpProjectionMatchesSeparateSwiGLUPath)'`
  `ANE_HARDWARE_TESTS=1 ESPRESSO_LFM2_WEIGHT_DIR=/tmp/lfm2-350m.esp/weights swift test --filter test_hybridSingleLayerRunsForActualLFM2AttentionWeights`
  `swift build -c release --product espresso-generate`
  retained long-horizon benchmark:
  `./.build/release/espresso-generate generate --bundle /tmp/lfm2-350m.esp --prompt 'Once upon a time' --max-tokens 4096 --benchmark-generate --compare-warmup 0 --compare-iterations 1 --no-tui --no-power`
  repeated results = `56.74 tok/s`, `56.83 tok/s`, `56.86 tok/s`,
  `generated_tokens=519`,
  coherent Elara continuation preserved through EOS.
  retained short-horizon spot-check:
  `./.build/release/espresso-generate generate --bundle /tmp/lfm2-350m.esp --prompt 'Once upon a time' --max-tokens 128 --benchmark-generate --compare-warmup 1 --compare-iterations 3 --no-tui --no-power`
  -> `57.85 tok/s`, `first_token_ms=3.26`, `median_token_ms=17.28`, `p95_token_ms=17.91`.
  classifier sweep:
  `ESPRESSO_CLASSIFIER_ARGMAX_BLOCK_SIZE=8192` -> `57.17 tok/s` once, `56.31 tok/s` on repeat,
  `16384` -> `57.01 tok/s`,
  `32768` -> `56.39 tok/s`.
  verdict = KEEP the BLAS attention-context rewrite as a real long-horizon win on the retained coherent LFM2 lane.
  KEEP the classifier block-size tuning hook for bounded future sweeps, but KILL any default change from the current evidence; the `8192` edge was not stable enough to promote.
- Experiment 7: test whether the generic row-major matvec helper should also switch from `vDSP_mmul` to direct `cblas_sgemv`.
  hypothesis:
  the attention-cache rewrite benefited from explicit BLAS matvecs, so the same might also help the generic `multiplyRowMajorMatrix` helper used by FFN, projection, and other exact-CPU paths.
  code change:
  temporarily replaced `multiplyRowMajorMatrix(...into:)` with direct `cblas_sgemv` and added a bounded parity test against a naive row-major reference.
  retained benchmark:
  `./.build/release/espresso-generate generate --bundle /tmp/lfm2-350m.esp --prompt 'Once upon a time' --max-tokens 4096 --benchmark-generate --compare-warmup 0 --compare-iterations 1 --no-tui --no-power`
  result:
  `44.70 tok/s`, `first_token_ms=10.59`, `median_token_ms=21.90`, `p95_token_ms=25.49`.
  output stayed coherent, but throughput cratered versus the retained `56.74-56.86 tok/s` baseline.
  verdict = KILL. Keep the helper on `vDSP_mmul`; the earlier BLAS win was specific to the visible-token attention-cache path, not a global matrix-vector replacement rule.
- Experiment 8: test whether partitioned-classifier block scheduling should evaluate the highest-norm blocks first.
  hypothesis:
  the partitioned FP32 classifier already prunes blocks with a Cauchy-Schwarz upper bound. Reordering blocks by descending precomputed max row norm might raise `bestValue` earlier and skip more later blocks without changing exactness.
  code change:
  added a bounded `ESPRESSO_CLASSIFIER_ORDER_BLOCKS_BY_MAX_NORM=1` research path,
  kept the classifier math identical,
  and added exactness tests showing reordered evaluation still matches naive argmax.
  invalid first measurement:
  an initial control/candidate pair was mistakenly run concurrently and both collapsed to roughly `36.7-36.9 tok/s`.
  That pair was discarded as contaminated because parallel throughput jobs on the same host are not comparable.
  retained serial comparison:
  control = `./.build/release/espresso-generate generate --bundle /tmp/lfm2-350m.esp --prompt 'Once upon a time' --max-tokens 4096 --benchmark-generate --compare-warmup 0 --compare-iterations 1 --no-tui --no-power`
  -> `55.29 tok/s`, `first_token_ms=3.35`, `median_token_ms=18.07`, `p95_token_ms=18.88`
  candidate = `ESPRESSO_CLASSIFIER_ORDER_BLOCKS_BY_MAX_NORM=1 ./.build/release/espresso-generate generate --bundle /tmp/lfm2-350m.esp --prompt 'Once upon a time' --max-tokens 4096 --benchmark-generate --compare-warmup 0 --compare-iterations 1 --no-tui --no-power`
  -> `55.39 tok/s`, `first_token_ms=3.47`, `median_token_ms=18.01`, `p95_token_ms=18.82`
  output stayed coherent and identical in quality shape, but the throughput delta was only `+0.10 tok/s`, well inside local noise.
  verdict = KILL. The block-ordering heuristic is too weak to justify additional exact-head complexity on the retained LFM2 lane.
