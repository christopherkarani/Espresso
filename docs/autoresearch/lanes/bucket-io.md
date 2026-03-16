# Bucket And I/O

Worktree root: `{{WORKTREE_ROOT}}`

Primary job:

- raise real-model Espresso tokens/sec by reducing per-token surface traffic and wasted bucket work

Own:

- compile-bucket policy
- prompt/prefill reuse
- IOSurface write volume
- IOSurface readback volume
- sequence-length growth strategy
- reuse of compiled state that affects steady-state throughput

Good file surfaces:

- `Sources/RealModelInference/RealModelInferenceEngine.swift`
- `Sources/EspressoGenerate/CLI.swift`
- `Sources/EspressoGenerate/GPT2DemoSupport.swift`

Primary metric:

- `espresso_tokens_per_second` on the real-model compare bench

Hard gates:

- `token_match == true`
- `text_match == true`
- same prompt, weights, tokenizer, and Core ML baseline contract

Do not own:

- benchmark contract changes
- benchmark result grading
- head-only optimization unless required to reduce I/O or bucket overhead

Default benchmark contract:

```bash
source .autoresearch/env.sh
.autoresearch/bench.sh
```
