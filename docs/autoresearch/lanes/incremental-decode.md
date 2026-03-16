# Incremental Decode

Worktree root: `{{WORKTREE_ROOT}}`

Primary job:

- raise real-model Espresso tokens/sec by eliminating full-sequence decode work that does not need to repeat every token

Own:

- incremental decode state reuse
- KV-cache or equivalent attention reuse
- avoiding full prompt re-embedding every token
- shrinking repeated spatial work in the ANE decode path
- runtime and graph changes needed to preserve exact output while reusing state

Good file surfaces:

- `Sources/RealModelInference/RealModelInferenceEngine.swift`
- `Sources/ANEGraphIR/`
- `Sources/ANEBuilder/`
- `Sources/ANERuntime/`

Primary metric:

- `espresso_tokens_per_second` on the real-model compare bench

Hard gates:

- `token_match == true`
- `text_match == true`
- same prompt, weights, tokenizer, and Core ML baseline contract

Do not own:

- benchmark contract changes
- benchmark result grading
- output-head-only tuning unless required to land incremental decode

Default benchmark contract:

```bash
source .autoresearch/env.sh
.autoresearch/bench.sh
```
