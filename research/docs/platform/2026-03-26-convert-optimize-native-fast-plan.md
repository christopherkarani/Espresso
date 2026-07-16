# Espresso Model Strategy: Convert, Optimize, Native-Fast

## Objective

Create a production model strategy for Espresso that does not require every incoming model to be trained from scratch, while still leaving a clear path to very high throughput on selected workloads.

For the current execution window, `Convert` and `Optimize` are in scope. `Native-Fast` is specified now so the interfaces and artifact contracts do not need to change later, but implementation of that lane is deferred.

## Product Lanes

### 1. Convert

Purpose:
- ingest supported external or internally prepared checkpoints
- package them into `.esp`
- run them with the best supported runtime path

Goals:
- maximize model coverage
- keep artifact handling deterministic
- keep conversion failures explicit and machine-readable

Expected performance:
- acceptable throughput
- not peak throughput

Allowed transformations:
- deterministic format conversion
- tokenizer normalization
- manifest/schema validation
- graph/profile normalization
- backend legality checks
- compression packaging that does not require model retraining

Non-goals:
- rewriting architecture semantics
- promising aggressive tok/s targets for arbitrary imports

### 2. Optimize

Purpose:
- start from an existing checkpoint and make it materially faster on Espresso without full pretraining

Goals:
- turn a converted model into a supported performance SKU
- allow architecture-aware surgery plus recovery tuning
- preserve compatibility with the same `.esp` contract

Expected performance:
- significant speedup over `Convert`
- still model-specific, not universal

Allowed transformations:
- quantization-aware fine-tuning
- palettization-aware fine-tuning
- LoRA quality recovery
- head factorization
- output-head specialization
- runtime-specific graph rewrites
- fixed-context bucketing
- structural adapter overlays
- distillation into a structurally similar student
- speculative or draft-verifier runtime support where the model family allows it

Non-goals:
- full from-scratch pretraining as the default path

### 3. Native-Fast

Purpose:
- maintain a curated ANE-first model family for Espresso’s highest-priority workloads

Goals:
- maximum tokens per second
- predictable runtime behavior
- architectural co-design between model and runtime

Expected performance:
- highest throughput tier
- suitable for hard targets such as `350-400 tok/s` on selected workloads

Allowed transformations:
- distillation from stronger teachers
- architecture changes for ANE constraints
- multi-token or draft heads
- factorized heads
- GQA/MQA
- KV-layout changes
- compression-aware training from the outset

Non-goals:
- broad compatibility

## Artifact Contract

Every `.esp` bundle should declare its strategy tier in the manifest:

- `model_tier = compat | optimized | native_fast`
- `optimization_recipe = ...`
- `teacher_model = ...` when distilled
- `draft_model = ...` when speculative
- `performance_target = ...`
- `quality_gate = ...`

This keeps the runtime and release system aware of whether a bundle is merely compatible, performance-tuned, or a curated fast SKU.

## Current Throughput Reality

Current measured Stories warm decode baseline in the bundle-backed runtime is approximately:

- `76 tok/s` with the default path
- `94 tok/s` when forcing the ANE exact head

This means:

- the current path is still far from the `350-400 tok/s` target
- `Optimize` can likely move the current Stories artifact meaningfully, but probably not all the way to `350-400 tok/s`
- `Native-Fast` is the realistic lane for the full target

## Execution Policy

All throughput work in this phase follows the same loop:

1. implement one feature
2. benchmark warm Stories decode
3. keep the change only if throughput improves and correctness holds
4. otherwise revert the feature commit immediately

Rules:

- benchmark against the bundle-backed Stories path, not a one-token smoke run
- use the same prompt and warm benchmark settings for comparisons
- retain only measured wins
- keep each retained optimization in its own commit when practical
- do not stack multiple unmeasured ideas into one patch

## Benchmark Standard

Primary benchmark for this phase:

```bash
./.build/debug/espresso-generate generate \
  --bundle /tmp/stories110m.esp \
  --max-tokens 64 \
  --benchmark-generate \
  --compare-warmup 1 \
  --compare-iterations 3 \
  Hello
```

Required metrics:

- `tok_per_s`
- `median_token_ms`
- `p95_token_ms`
- `first_token_ms`
- `exact_head_backend`
- `cached_bindings_enabled`
- `compile_breakdown`

Correctness gate:

- generated text must be valid and non-crashing
- no new runtime failures
- no benchmark-only mode hacks that diverge from normal generation semantics

## Convert Lane Deliverables

### Phase C1: Format and Import Stability

- keep `.esp` as the only supported runtime artifact
- keep native `.ane` model directories as the only supported pack input
- preserve deterministic packing, signatures, and support-matrix tests

Exit criteria:

- no reintroduction of GGUF or ad hoc runtime-only inputs
- existing pack/inspect/resolve/generate tests remain green

### Phase C2: Coverage Expansion

- broaden supported import families only when they cleanly map to Espresso runtime constraints
- reject unsupported families with explicit diagnostics

Exit criteria:

- every newly supported family has bundle tests, runtime tests, and a declared tier

## Optimize Lane Deliverables

### Phase O1: Head Path Optimization

Target:
- replace default CPU exact head work on Stories with a measured-better ANE head path

Work:
- make the ANE exact/fused head available as the default Stories path when benchmark-positive
- keep CPU fallback intact

Exit criteria:

- Stories benchmark exceeds the current baseline
- correctness holds

### Phase O2: Cached Bindings

Target:
- activate the cached-binding fast path for Stories or fail with precise diagnostics

Work:
- stop silently swallowing cached-binding creation failures
- surface the exact failure layer/surface/shape
- fix the underlying issue so cached bindings stay enabled on Stories

Exit criteria:

- `cached_bindings_enabled=true` on the retained fast path
- warm Stories benchmark improves over the retained prior step

### Phase O3: Fused Decode Runtime

Target:
- replace the current hybrid per-layer decode path with the fused decode path if it wins

Work:
- evaluate wiring `FusedDecodeKernelSet` or `FusedTwoLayerDecodeKernelSet` into the active Stories runtime
- benchmark one-layer fused and two-layer fused variants independently

Exit criteria:

- retained only if faster than the cached-bindings hybrid path

### Phase O4: Additional In-Tree Optimizations

Candidate ideas:

- factorized classifier / expansion-argmax variants
- safe lane-spatial tuning
- donor-delta re-evaluation under the new baseline
- other dormant fast paths already present in-tree

Exit criteria:

- each candidate must beat the retained baseline

## Native-Fast Lane Specification

This lane is not implemented in this phase, but the plan is fixed now.

### Phase N1: Distilled Stories-Fast

Build a curated ANE-first Stories-family student with:

- architecture constrained to Espresso’s fastest decode path
- output head designed for ANE residency or factorization
- optional draft head or multi-token head
- compression-aware training

### Phase N2: Speculative Runtime Integration

Port the existing speculative machinery to the Llama/Stories family:

- draft-verifier runtime split
- checkpoint capture and rollback
- acceptance-rate tracking
- benchmark gates for acceptance and wall-clock gains

### Phase N3: Fast SKU Release

Ship `Stories-Fast` as a `native_fast` bundle with:

- measured throughput target
- declared teacher/distillation lineage
- explicit quality threshold

## Coding Agent Work Queue

Execute in this order:

1. commit the current validated baseline
2. benchmark Stories warm decode and record the baseline
3. make the ANE exact head default for Stories behind a benchmark gate
4. benchmark and keep or revert
5. fix Stories cached-bindings fallback
6. benchmark and keep or revert
7. test fused decode runtime integration
8. benchmark and keep or revert
9. test additional optimize-lane ideas one at a time
10. update docs and review with the final retained set

## Definition of Done for This Phase

This phase is done when:

- the convert/optimize/native-fast strategy is documented
- the current Stories throughput baseline is recorded
- at least the top `Optimize` candidates have been implemented and measured
- every retained optimization has benchmark evidence
- every losing optimization has been reverted
- the repo history reflects only measured improvements
