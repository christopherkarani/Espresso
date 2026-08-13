# Qwen2.5-0.5B-Instruct on the Neural Engine: measured parity

Espresso runs Qwen2.5-0.5B-Instruct through its ANE hybrid decode path and reproduces a
PyTorch fp32 reference on a fixed 12-prompt greedy suite: **10 of 12 sequences match
token-for-token, 341 of 384 tokens agree**, and the two divergences are near-ties that are
explained below rather than waved away.

Everything here is measured on Apple M-series hardware with
`ESPRESSO_REALMODEL_DISABLE_HYBRID_FALLBACK=1`, so nothing silently dropped to the CPU
reference path. Every number has a command that regenerates it.

## Reproducing from a clean clone

```bash
# 1. Download the checkpoint, convert bf16 -> fp16 blobs, pack the .esp bundle
python3 scripts/convert_qwen25_05b_to_esp.py

# 2. Generate on the ANE, refusing any fallback
ESPRESSO_REALMODEL_DISABLE_HYBRID_FALLBACK=1 \
  ./espresso generate --model ~/Library/Caches/Espresso/qwen25-05b/Qwen2.5-0.5B-Instruct.esp \
  -n 24 "The capital of France is"

# 3. Per-layer parity report (writes docs/qwen-parity-layers.md)
python3 scripts/qwen25_pytorch_reference.py layer-parity \
  --backends cpu-fp32 cpu-fp16 ane \
  --report-markdown docs/qwen-parity-layers.md

# 4. End-to-end logit parity (writes docs/qwen-logit-parity.json)
python3 scripts/qwen25_pytorch_reference.py logit-parity \
  --backends cpu-fp32 ane --report-json docs/qwen-logit-parity.json

# 5. Greedy token parity against the committed PyTorch fixture
ANE_HARDWARE_TESTS=1 ESPRESSO_REALMODEL_DISABLE_HYBRID_FALLBACK=1 \
  swift test --filter qwenGreedyParity
```

The converter and the reference script both bootstrap a managed Python venv when `torch`
and `transformers` are missing, so step 1 works on a machine with no Python setup.

## What runs where

The hybrid decode path is a deliberate split, not a degraded mode:

| Stage | Device | Why |
| --- | --- | --- |
| Q/K/V projection (+ bias) | ANE | `hybrid.decodeQKVOnly` kernel, per layer |
| FFN (SwiGLU) and its projection | ANE | `hybrid.decodeFFN`, `hybrid.decodeProjectionFFN` |
| RoPE, attention softmax, KV cache | CPU | Sequential, cache-resident; part of the hybrid design |
| LM head (`151936 x 896`) | CPU (`cpu_fp16_tiled`) | 136.1M fp16 elements (272 MB) against a ~16M-element (32 MB) ANE SRAM budget |

`ESPRESSO_REALMODEL_DISABLE_HYBRID_FALLBACK=1` guards *departures* from that split. With it
set, anything that would route the model to the pure-CPU oracle instead throws
`RealModelInferenceError.hybridFallbackDisabled` naming the stage and the reason, rather
than producing plausible output from the wrong backend.

All 24 layers compile cleanly at Qwen2.5-0.5B's widths: **0 compile retries, 0 compile
failures** across the three kernel classes, in ~20 s. This is a marked change from the
Qwen3-0.6B hybrid attempt recorded in `tasks/lessons.md` (156 retries, 195 failures), and
the difference is width: every Qwen2.5-0.5B dimension is a multiple of 64
(`dModel=896`, `kvDim=128`, `hiddenDim=4864`).

## Conversion fidelity: bf16 to fp16 is lossless here

The checkpoint ships bf16. Converting to fp16 is exact for every normal value, because
bf16's 8-bit mantissa is narrower than fp16's 11-bit mantissa; only the exponent range can
bite. Over all 291 tensors:

- max absolute rounding error: **2.98e-8** (that is 2^-25, the subnormal flush floor)
- fp16 overflows: **0** (largest magnitude weight is 214.0, against fp16's 65504 ceiling)

So the fp16 checkpoint is not a lossy quantization of the reference. Any parity gap
downstream comes from **arithmetic**, not from the weights.

## Per-layer parity

Full tables are in [`qwen-parity-layers.md`](qwen-parity-layers.md). Each layer is fed the
*reference* input hidden states for every position, so errors are measured per layer
instead of compounding down the stack.

| Backend | worst layer | max abs diff | worst relative to layer scale |
| --- | --- | --- | --- |
| `cpu-fp32` (Espresso's CPU oracle) | 2 | 6.10e-4 | 1.24e-5 |
| `cpu-fp16` (fp32 accumulation, fp16 intermediates) | 10 | 7.37e-1 | 1.77e-3 |
| `ane` (hybrid kernels) | 21 | 3.09e+0 | 4.40e-2 |

The fp32 CPU column is the architecture check: at 1e-5 relative, RoPE (theta = 1e6), GQA
(14 Q heads over 2 KV heads), RMSNorm epsilon, SwiGLU, and the Q/K/V biases are all right.

The absolute numbers in the fp16 and ANE columns look alarming until you note the scale
Qwen's residual stream reaches: **~1700** by the middle layers. fp16 spacing at magnitude
1700 is ~1.0, so the residual stream cannot represent anything finer than about 0.5 there
regardless of how the arithmetic is done. That predicts ~1.3 absolute error at layer 3's
scale, which is what both fp16 columns show — and for most middle layers the ANE error
equals the `cpu-fp16` error exactly, i.e. it is fp16 *storage*, not an ANE defect.

Two checks separate rounding from a structural bug at layer 0, where the ANE error is
4.7e-2 and the residual stream is still small:

- The error is zero-mean, uncorrelated with the reference, and flat across positions — the
  signature of rounding, not of a dropped term or a cache/position bug.
- Qwen's `bq` reaches ±79 and `bk` ±130. A dropped Q/K/V bias would show up as O(1) error,
  three orders of magnitude above what is measured, so the bias is being applied on the ANE.

Measured 1.95e-2 relative against a 1.46e-2 prediction from fp16 accumulation over
`dModel=896`.

## End-to-end logit parity

Per-layer parity proves each layer in isolation; the logit measurement proves the error the
whole stack accumulates, which is what actually decides a greedy token. The Swift driver
chains layers (layer N+1 consumes layer N's output) so this is the served computation.
Recorded in [`qwen-logit-parity.json`](qwen-logit-parity.json):

| Backend | worst max abs logit diff | argmax agreement with PyTorch |
| --- | --- | --- |
| `cpu-fp32` | **9.3e-5** | 12/12 |
| `ane` | **0.955** | 12/12 |

The fp32 result is the important one for correctness: through 24 layers, the final norm,
and a 151936-wide tied LM head, Espresso agrees with PyTorch to 1e-4 in logits. The
implementation is right. The ANE column is the price of fp16: **up to ~1 logit of error**.

## Greedy token parity, and the two flips

`Tests/RealModelInferenceTests/QwenParityExactMatchTests.swift` drives generation from the
fixture's prompt token IDs (so this measures the model, not the tokenizer) over 12 prompts
x 32 tokens:

```
exact cases 10/12   matching-prefix tokens 341/384   head=cpu_fp16_tiled
```

Both divergences are near-ties where the reference itself was nearly indifferent, and in
both the runtime landed on **precisely the reference's runner-up token**:

| Case | Diverges at | Reference top-1 / runner-up | Reference top-1/top-2 gap | Runtime chose |
| --- | --- | --- | --- | --- |
| 5 | token 14/32 | 8059 / 7015 | **0.027** | 7015 (the runner-up) |
| 6 | token 7/32 | 2530 / 264 | **0.069** | 264 (the runner-up) |

Gaps of 0.027 and 0.069 sit far inside the ~0.955 logit error fp16 imposes, so these flips
are the expected consequence of the measured precision, not wrong answers. The test encodes
exactly that contract rather than a bare tolerance: every divergence must (a) be the
reference's own runner-up and (b) occur at a top-1/top-2 gap within the measured logit
error. A divergence at a wider gap, or to any other token, fails the test. The fixture
therefore commits the per-step top-2 gap and runner-up token alongside the expected IDs.

Exact match across all 12 sequences is not reachable on this hardware. It would require
fp32 accumulation in the residual stream, and the ANE is an fp16 datapath. The honest
statement is the one above: the implementation agrees with PyTorch to 1e-4 in fp32, and
fp16 execution flips greedy choices only where the model itself was within 0.07 of a tie.

## Reference oracle: two traps worth knowing

1. **`model.generate(do_sample=False)` is not pure greedy.** Qwen2.5's
   `generation_config.json` sets `repetition_penalty: 1.1`, and `generate` applies its
   logits processors regardless of sampling. An early fixture built that way produced 0/10
   agreement and made Espresso look broken. `scripts/qwen25_pytorch_reference.py` uses an
   explicit argmax loop over raw logits instead.
2. **`output_hidden_states` does not give you layer boundaries.** HuggingFace applies the
   final norm before the last entry, which made layer 23 report a 167 max abs diff. The
   script uses forward hooks to capture true pre/post-layer states; layer 23 then reports
   1.4e-4.

## ANE limitations hit, and what was done about each

| Limitation | Resolution |
| --- | --- |
| Q/K/V projection bias (Qwen2 hardcodes it; the llama lane had no path for it) | Plumbed through `LayerWeightPaths`, `DecodeQKVOnlyGenerator` MIL emission, the runtime model tree, and both CPU oracles, behind a `hasQKVBias` flag so bias-free llama artifacts are untouched |
| LM head too large for ANE SRAM (272 MB vs ~32 MB) | Runs on CPU (`cpu_fp16_tiled`) as a deterministic capacity policy, documented above rather than hidden |
| Name-based routing sent anything matching "qwen" to the pure-CPU oracle | Artifacts now declare `preferredDecodePath` in `metadata.json`; the runtime honours it, and the legacy heuristic only applies when an artifact is silent |
| `ESPRESSO_REALMODEL_DISABLE_HYBRID_FALLBACK` was GPT-2-specific, so the llama lane could fall back silently | `resolvedLlamaGenerationPath` throws `hybridFallbackDisabled` naming stage and reason; the GPT-2 compile-failure path was hardened the same way |
| `preferredDecodePath` was dropped when packing a bundle, sending packed Qwen back to CPU | `ESPModelConfigIO` parses and preserves it, pinned by a test |
| Finite per-process ANE compile budget (720 compiles would exhaust it) | The parity test reuses one engine and compiles the maximal bucket once |
| fp16 residual stream at magnitude ~1700 | Irreducible on this hardware; quantified per layer and at the logit level, and the greedy test asserts flips only at near-ties |

## Known gaps

- Greedy only. No sampling parity claim.
- Only Qwen2.5-0.5B-Instruct is measured. 1.5B is untested.
- The prompt suite is 12 fixed prompts at 32 tokens; it is a parity fixture, not a benchmark.
- Attention, RoPE, and the LM head run off the ANE by design (see the table above).
- No throughput claim is made here. Parity was the deliverable.
