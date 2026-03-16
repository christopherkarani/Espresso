# Head Path

Worktree root: `{{WORKTREE_ROOT}}`

Primary job:

- raise real-model Espresso tokens/sec by reducing final-norm, logits, and token-selection cost per generated token

Own:

- final norm and output-head placement
- hidden-state readback volume
- logits projection
- argmax and sampling path
- ANE, CPU, or Metal head-path choices that preserve exact generation

Good file surfaces:

- `Sources/RealModelInference/RealModelInferenceEngine.swift`
- `Sources/Espresso/ANEGenerationOutputHead.swift`
- `Sources/Espresso/MetalExpansionArgmax.swift`
- `Sources/Espresso/PartitionedArgmax.swift`
- `Sources/Espresso/FP16TiledClassifier.swift`
- `Sources/Espresso/INT4Classifier.swift`

Primary metric:

- `espresso_tokens_per_second` on the real-model compare bench

Hard gates:

- `token_match == true`
- `text_match == true`
- same prompt, weights, tokenizer, and Core ML baseline contract

Do not own:

- benchmark contract changes
- benchmark result grading
- broad decode-state redesign unless required to land a head-path improvement

Default benchmark contract:

```bash
source .autoresearch/env.sh
.autoresearch/bench.sh
```
