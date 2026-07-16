# Stories Convert -> Optimize Execution Plan

## Objective

Move Espresso from runtime-only Stories tuning into the model-side `Convert -> Optimize` loop, using `.esp` as the only supported artifact boundary and the real Stories release benchmark as the only throughput scoreboard that counts.

Primary target:
- drive real single-stream Stories throughput materially beyond the current retained path

Stretch target:
- create the model/export/runtime infrastructure required to pursue `300+ tok/s` Stories-class SKUs

Out of scope for this phase:
- App Store/public-lane work
- GGUF support
- synthetic harness numbers presented as product results
- `Native-Fast` implementation beyond interface and pipeline preparation

## Current Baseline

Real Stories throughput must be measured with:

```bash
./.build/arm64-apple-macosx/release/espresso-generate generate \
  --bundle /tmp/stories110m.esp \
  --max-tokens 64 \
  --benchmark-generate \
  --compare-warmup 1 \
  --compare-iterations 3 \
  Hello
```

Current retained reality:
- real Stories release throughput is still about `100 tok/s`
- KV caching already exists in the real Stories path
- cheap runtime-flag tuning did not produce a retained win
- the next credible path is model-side `Optimize` work

Synthetic harness caveat:
- recurrent echo-harness microbenchmarks in `Tests/EspressoTests/GenerationHarnessHardwareTests.swift` are useful for upper-bound intuition only
- they are not valid substitutes for the real Stories release benchmark

## Execution Rules

Every work item in this phase must follow the same discipline:

1. establish or confirm the current Stories release baseline
2. change one thing
3. run the real Stories release benchmark
4. run the relevant quality/correctness checks
5. keep the change only if the measured result is better and quality remains acceptable
6. revert the change immediately if it regresses throughput or quality

Rules:
- do not benchmark in parallel on the same machine
- do not stack multiple unmeasured changes in one patch
- use release builds for throughput decisions
- retain only measured wins
- commit retained wins frequently
- record every retained and rejected experiment in `tasks/todo.md`

## Scoreboard and Quality Gates

### Primary Scoreboard

Use the real Stories release benchmark above.

Required metrics:
- `tok_per_s`
- `median_token_ms`
- `p95_token_ms`
- `first_token_ms`
- `compile_ms`
- `exact_head_backend`
- `cached_bindings_enabled`

### Quality Gates

For any model or decode-path change:
- run at least one fixed short prompt and one longer prompt
- compare token/text behavior against the retained Stories path
- explicitly note whether the change is exact, near-exact, or approximate
- keep approximate behavior behind a feature/model gate unless it is clearly acceptable

### Artifact Gates

Every new Stories SKU or variant must:
- package cleanly into `.esp`
- resolve through `esprun`
- run through `espresso-generate --bundle`
- declare its tier and optimization metadata in the bundle manifest

## Workstreams

## Workstream 1: Fixed Shorter-Context Stories SKUs

### Goal

Create context-specialized Stories bundles such as `stories110m-256`, `stories110m-512`, and `stories110m-1024` to reduce state size, cache movement, and compile-time specialization overhead.

### Deliverables

- bundle metadata support for context-specialized Stories SKUs
- explicit context-target naming and manifest fields
- compiler/export support for producing shorter-context `.esp` variants
- runtime profile resolution that respects SKU context limits
- release benchmark comparisons for each retained SKU

### Code Areas

- `Sources/ESPBundle`
- `Sources/ESPCompiler`
- `Sources/ESPRuntime`
- `Sources/EspressoGenerate`
- Stories packing/export tooling and tests

### Verification

- bundle schema/validation tests
- runtime resolution tests
- generation smoke tests for each SKU
- real Stories release benchmark for retained SKUs

### Exit Criteria

- at least one shorter-context Stories SKU exists as a first-class `.esp` artifact
- runtime selection and manifest handling are production-grade
- retained benchmark evidence shows whether the smaller-context SKU materially helps

## Workstream 2: GQA/MQA Stories Variant

### Goal

Reduce decode-time KV bandwidth by supporting Stories variants with fewer KV heads.

### Deliverables

- config/schema support for `nHead != nKVHead`
- exporter/compiler/runtime handling for GQA/MQA Stories variants
- cache layout and surface-shape verification
- bundle metadata that distinguishes the variant cleanly
- benchmark evidence for any runnable GQA/MQA Stories artifact

### Code Areas

- model config and manifest handling
- bundle/export pipeline
- runtime cache/state setup
- decode-path tests for reduced-KV-head variants

### Verification

- config parsing tests
- cache-shape tests
- runtime generation tests
- real Stories release benchmark if a runnable artifact is produced

### Exit Criteria

- Espresso can package and execute a Stories-class GQA/MQA variant cleanly
- the bundle contract is stable enough for future optimized/distilled variants

## Workstream 3: Distilled Stories-Native Checkpoint

### Goal

Create the first Stories-native student path designed for Espresso constraints rather than preserving the current checkpoint at all costs.

### Deliverables

- a reproducible teacher-student distillation pipeline
- training/eval configs
- export path into `.esp`
- artifact metadata for teacher, recipe, and tier
- local proof that the pipeline is real, even if full training is deferred or partial

### Code Areas

- training/config tooling
- export and bundle packaging
- evaluation and benchmark scripts
- docs for reproducibility and artifact lineage

### Verification

- pipeline execution proof
- checkpoint/export smoke test
- benchmark comparison if a real student artifact is produced
- explicit note when a step is scaffolded vs fully executed

### Exit Criteria

- the distillation pipeline exists as production-quality infrastructure
- any produced student artifact can be bundled and benchmarked through the same release path

## Workstream 4: Cheaper / Factored Output Head

### Goal

Reduce token-selection cost by supporting a cheaper output head for optimized Stories variants.

### Deliverables

- model-side head definition and metadata
- export/runtime support for the factored head
- correctness tests and benchmark wiring
- quality evaluation against the retained Stories path

### Constraints

- do not keep a factored head on the real Stories path unless it wins the real benchmark
- do not confuse microbench head results with end-to-end Stories wins

### Exit Criteria

- factored-head support exists as a clean product surface
- it is either retained for a winning model variant or explicitly documented as a rejected strategy for the current checkpoint

## Workstream 5: Draft or Multi-Token Head

### Goal

Add the first credible model-side path for exceeding the one-token-per-step ceiling.

### Deliverables

- manifest/runtime support for draft or multi-token heads
- verifier/rollback/acceptance accounting
- benchmark output for accepted-token rate and wall-clock throughput
- model/export support for draft-enabled Stories variants

### Constraints

- do not retain speculative complexity if wall-clock Stories throughput does not improve
- exactness vs approximation must be clearly labeled

### Exit Criteria

- a draft or multi-token path exists as a production-grade, benchmarked feature gate
- any retained version demonstrates a real end-to-end Stories win

## Recommended Order

Execute in this order:

1. fixed shorter-context Stories SKUs
2. GQA/MQA Stories variant support
3. distilled Stories-native pipeline
4. cheaper/factored head
5. draft or multi-token head

Rationale:
- shorter-context SKUs are the cheapest structural optimization
- GQA/MQA unlocks better decode economics without immediately requiring the full native-fast lane
- distillation creates the first model family that can actually exploit the later head and draft work
- factored heads and draft heads are most valuable once they are attached to the right model

## Required Documentation and Tracking

Keep these files current:
- `tasks/todo.md`
- this plan document
- any new benchmark/result docs added during the work

Every retained change must record:
- benchmark command
- before throughput
- after throughput
- quality note
- keep/revert decision
- commit SHA if retained

## Definition of Done for This Phase

This phase is complete when:
- the Stories `Convert -> Optimize` plan is implemented, not just described
- at least the first two workstreams are real product code, not placeholders
- the benchmark ledger clearly separates retained wins from rejected experiments
- the next move toward `Native-Fast` is obvious from measured evidence rather than guesswork
