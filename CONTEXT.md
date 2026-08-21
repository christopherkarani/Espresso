# Espresso

Direct Neural Engine inference for transformers on Apple Silicon. Terms here name serving and decode concepts, not Swift types.

## Language

**Llama serving session**:
One llama-family serving run: compiled decode programs plus the token loop that consumes prompt token ids and emits tokens.
_Avoid_: decode session, serving runtime, generation harness

**Trunk**:
Which decode-step implementation a llama serving session uses for each token: fused hybrid, split hybrid, or exact-CPU.
_Avoid_: path, backend, lane

**Decode step**:
One generated token's worth of hidden-state update and KV-cache write.
_Avoid_: generate, forward pass, eval

**Fused hybrid**:
A decode step that runs one ANE program per transformer layer, attention included.

**Split hybrid**:
A decode step that runs ANE QKV, then host attention, then ANE FFN.

**Exact-CPU**:
A decode step that runs the transformer layer on the CPU. Also the Qwen oracle.
_Avoid_: CPU exact, cpu_fp16_tiled (that is a classifier, not a trunk)
