# omlx PR #2853 → Espresso: CPU/AMX work sharing

**Date:** 2026-08-20
**Source:** [jundot/omlx#2853](https://github.com/jundot/omlx/pull/2853) (closed, unmerged; head `f640390`)
**Question:** what transfers to Espresso TTFT and tok/s, what is our baseline, what would we actually get.

This is a research note, not a product claim. Numbers below are either cited from the PR, from Espresso's checked-in artifacts, or labeled as estimates.

## What the PR actually is

Prefill-only hybrid for Qwen3.5/3.6/3.8 on M3 Ultra. Decode stays on GPU. Token-generation tok/s is unchanged by construction ([experimental doc](https://github.com/jundot/omlx/blob/f640390fcc378f50786d959dbfb5ecc3bcd784a6/docs/experimental/qwen35_ane_prefill.md)).

Four devices in one MLP gate/up:

1. ANE instance 1 (`kANEFAneInstanceHint = 1`)
2. ANE instance 2 (`kANEFAneInstanceHint = 2`)
3. Quantized GPU suffix (Metal, NAX on M5)
4. Optional FP16 CPU AMX via `BNNSMatMul`

CPU fractions are taken from the GPU suffix, not from the ANE slice. Gate/up, down projection, and residual GDN qkv have independent shares because they sit in different dependency windows.

### Headline numbers (author, M3 Ultra, Qwen3.8-27B AWQ 4.85 bpw, 2048-token PP)

| Config | Prompt tok/s | Versus |
| --- | ---: | --- |
| GPU only | 355.9 (tuner) / ~335–445 depending on run | — |
| Dual ANE + GPU, CPU off | 460.1 | already-hybrid baseline |
| + 13.5% gate/up CPU | 475.3 | +3.3% vs CPU-off |
| + 13.5% gate/up + 20% down CPU | 494.7 | +7.5% vs CPU-off |
| In-app 5-control tuner (45/14/20/45/13) | 517.9 | +45.8% vs that run's GPU-only |

The +45.8% is **versus GPU-only**, not versus the already-hybrid ANE path. The CPU-AMX increment on top of dual-ANE is ~7.5%, and it costs peak RSS: ~31 GiB → ~50 GiB with the 20% down split.

### The secrets that are not the split percentages

**BNNS `n_threads` does not create AMX parallelism.** Isolated FP16 matmul, M3 Ultra (20P+8E, 6 clusters):

| Mode | Workers | Median | Effective cores |
| --- | ---: | ---: | ---: |
| BNNS automatic | auto | 16.171 ms | 3.91 |
| Manual aligned rows | 12 | 15.407 ms | 8.84 |
| Shared-resource | 8 | **14.560 ms** | **6.67** |
| Shared-resource | 16 | 14.423 ms | 13.00 |

Eight workers capture almost all of the 16-worker throughput at half the CPU. Production hybrid: 5.715 ms vs 6.997 ms automatic (**+22.4% isolated**, bit-identical). Mixed P+E shards were a disaster (12 mixed: 35.16 ms vs 15.90 ms P-biased).

Implementation: 64-row-aligned output shards, `BNNSFilterParameters.n_threads = 1`, `dispatch_apply` at `QOS_CLASS_USER_INTERACTIVE`, RAII `__bsdthread_ctl(0x2000, set/clear, worker_index, cluster_concurrency=2)`. Public Mach affinity was rejected by the kernel. Busy-spin core guards were rejected as unsafe. AMX occupancy is ~3 cores-equivalent on M3 Ultra; author reports no extra fan load.

**Eager FP16 rows.** CPU-assigned output channels are dequantized once at load into a separate FP16 clone (`tools/clone_mlx_model_fp16.py`). Source checkpoint is never mutated. Trades RAM and load time for zero per-prompt dequant.

**Procedure banks.** 64 MLP + 48 GDN slices packaged as 112 procedures in **two** resident `_ANEInMemoryModel`s (one per die). Single unpinned procedure of the same slice was 39.5% slower than two pinned evals. ~4 GiB ANE address window per die: dual 53%/50% 27B banks fit Ultra, fail on M3 Max with `0x20004`.

**Host sync is hardware-dependent.**

- M3 Ultra: blocking `waitUntilCompleted` on the pack buffer is **required**. Completion-handler / Metal-callback launch delayed ANE behind the GPU suffix and made the 64-layer body **5.6% slower than GPU-only**.
- M5 Pro: the same two host waits per op (`commit` of MLX's shared stream + `model_->wait`) dominate. Dose-response: 64 MLP layers −17.6% real 16K TTFT vs GPU-only; 16 layers −2.5%. Tuner said +9.4%; interleaved real requests said −27.2% at 16K. The loss scaled with op count, not ANE work.

**Tuner can invert the sign.** `measure_length = sequence_length * 2 + 1` produced `pp=4097` → two full ANE blocks and no GPU tail. Real traffic is `2048 ane + 2047 gpu`. Patching to `* 2` halved the phantom gain (claimed +10.6% → +5.11%). ArraysCache block size 2048 also clamped 4096-wide chunks whenever prefix cache was on.

## Why we must not copy the recipe into Espresso decode

| | omlx #2853 | Espresso today |
| --- | --- | --- |
| Goal metric | prompt-processing tok/s | decode tok/s + first-token wait |
| Shape | fixed 2048-token GEMM | N=1 GEMV decode; prefill is the same N=1 loop |
| Who owns FFN | GPU, with ANE stealing a prefix | ANE already owns QKV + SwiGLU |
| Who owns attention | GPU | **CPU by default for Qwen/Llama** (`prefersCPUDecodeAttention`) |
| Dual ANE | M3 Ultra, instance hints 1 and 2 | single `_ANEClient`, published numbers are M3 Max |
| CPU role | AMX GEMM overlapping ANE+GPU | LM head (`cpu_fp16_tiled` / NEON GEMV) + RoPE + CPU attention |
| Eval | threaded submit + events; still two host waits per op | `ANEKernel.eval()` → blocking `ane_interop_eval` |
| Decode effect of this PR | none | n/a |

Splitting N=1 Qwen FFN rows onto CPU AMX **takes work off the ANE**, which is already the right device for that GEMV. omlx's CPU share only paid because the GPU suffix was the limiter and ANE+GPU were already concurrent. On Espresso Qwen, the limiter at growing context is CPU attention (`docs/qwen15b-chat-name-recall.txt`: 2.1 tok/s @ ctx 51 → 0.3 tok/s @ ctx 367).

The pipelined Metal path in `DecodeForwardPass.runHybridDecodeTimed` is commented as overlapping Metal SDPA[N] with ANE QKV[N+1], but the loop waits for Metal at the **start** of the next iteration before FFN[N] and QKV[N+1]. That is the same class of "sync before the merge" tax the M5 report called out.

## Espresso baseline (do not mix lanes)

### Shipping, M3 Max, `benchmarks/results/latest.json`

| Lane | ms/tok | tok/s |
| --- | ---: | ---: |
| Recurrent fused 6-layer Stories | 1.929 | **519** |
| Direct transformer 6-layer Stories | 6.559 | 153 |
| CoreML `.cpuAndNeuralEngine` | 6.582 | 152 |

TTFT is not in `latest.json`. Recurrent fused budget (blog, same artifact): embedding 0.05 + two 3-layer ANE dispatches 0.89 + ANE classifier 0.94 = 1.93 ms. Head is ~49% of that token.

Research note (not a product claim): real Stories `.esp` serving was still ~100 tok/s on 2026-03-26.

### Real-model serving, Qwen2.5-1.5B-Instruct hybrid

From `docs/qwen15b-chat-name-recall.txt` (10-turn greedy chat, fallback disabled):

| ctx | tok/s | reported TTFT |
| ---: | ---: | ---: |
| 51 | 2.1 | 59 ms |
| 77 | 1.5 | 21 ms |
| 138 | 0.9 | 23 ms |
| 202–349 | 0.4–0.5 | 22–23 ms (one 91 ms) |
| 367 | 0.3 | 108 ms |

**Reported TTFT is not user-visible first-token wait.** `RealModelInferenceEngine` starts `generationStart` **after** the sequential N=1 prefill loop. `firstTokenLatencyMs` is the first LM head (~21 ms for the 151936×1536 fp16 head). Prefill of the whole history is excluded. Chat re-prefills growing history (`README`, `tasks/todo.md`).

User-visible wait for a 50–300 token prompt is therefore "N_prompt sequential hybrid steps + ~21 ms head". At ~2 tok/s-equivalent prefill that is seconds, not 21 ms. This is the same class of harness lie as omlx's `pp=4097`.

Qwen/Llama default: `prefersCPUDecodeAttention == true`. Metal fused SDPA exists but is not the Qwen default. Hybrid generate reports `cachedBindingsEnabled: false` on the Qwen path even when bindings were built.

## What to steal, in order

### 1. Redefine TTFT and stop publishing the LM-head-only number as chat TTFT

Measure wall clock from prompt submit through first emitted token, including re-prefill. Until that exists, any "TTFT win" from an AMX head is a 21 ms microbench.

### 2. Batched prefill kernels (the actual PR shape)

Compile fixed-shape ANE GEMMs at seq ∈ {64, 128, 256} for Qwen QKV/FFN. Prefill the prompt as one or few GEMMs, not 28×2 blocking evals per token.

This is where CPU AMX sharing can pay: leftover output channels of a **wide GEMM** overlap ANE, same as omlx. On M3 Max there is one ANE, so the split is ANE + CPU AMX + Metal attention, not dual-ANE.

Directionally: user-visible TTFT on Qwen 1.5B 256-token prompts should drop from seconds toward a few hundred milliseconds if prefill becomes a handful of GEMMs. Exact multiplier needs a measured prefill profile; do not quote omlx's 45.8%.

### 3. BNNS FP16 AMX row-shard for the Qwen LM head (small, isolated, portable)

Replace `FP16TiledClassifier` / NEON `ane_interop_fp16_gemv_argmax` with the omlx recipe: eager FP16 rows (already the storage format), 64-row shards, `BNNSMatMul` n_threads=1, 8 workers, capability-guarded `__bsdthread_ctl`. Bit-identical vs current first-max rule is required.

Expected: reported TTFT 21 ms → ~8–14 ms on M3 Max class. Decode tok/s at ctx 51 (476 ms/token) moves ~2–4%. At the Stories fused path the head is already on ANE; do not touch that lane.

### 4. Do not sync until the merge — but profile first

On Espresso Qwen, 28 layers × (QKV eval + FFN eval) blocking roundtrips. If host overhead is even 1–2 ms per eval, that is 56–112 ms dead per token **before** attention.

True overlap: submit ANE QKV[L], run CPU/Metal attention[L] without waiting on a previous leftover, wait only before FFN[L]; or on Ultra, pin FFN[L] and QKV[L+1] to two dies. Copying omlx's blocking pack wait "because Ultra needed it" onto M-series Max/M5 would be the M5 footgun.

Instrument `lastHWExecutionTimeNS` vs wall `eval()` per kernel before changing the wait model. Ultra and M5 disagreed in the PR thread.

### 5. Hardware-local tuner, not a 14% constant

Calibrate on one real layer (cheap bank), compile one predicted full model, verify on the **same chunk composition as chat** (including the GPU/CPU tail). CPU AMX will likely be rejected on M5 the way omlx's tuner did. Down-projection sharing stays off until RSS is budgeted.

### 6. Explicitly out of scope for this PR's recipe

- Dual-ANE procedure banks on M3 Max (address window).
- Splitting N=1 decode FFN onto CPU.
- INT8 requant of ANE weights (omlx is approximate; Espresso's Qwen contract is fp16 greedy parity).
- Publishing omlx's 517.9 prompt tok/s next to Espresso's 519 decode tok/s. Different models, different phases, different chips.

## Expected outcome if we proceed (ranges, not promises)

| Work | Touches | vs current Qwen 1.5B hybrid | vs shipping Stories 519 |
| --- | --- | --- | --- |
| AMX LM head only | reported TTFT | 21 ms → 8–14 ms; tok/s +2–8% | none (head already ANE) |
| Batched prefill, ANE only | user-visible TTFT | likely 5–20× on the prefill portion | none |
| Batched prefill + CPU AMX share | user-visible TTFT | extra ~5–15% on top of batched ANE, Ultra/Max class; 0 or negative on M5 | none |
| Dual ANE banks | Ultra prefill only | omlx-like 1.2–1.4× PP; **cannot load on M3 Max** | none |
| True device overlap (no per-op host wait) | decode tok/s | 1.3–2× **if** wall-eval >> hw-eval; else ~0 | modest (already 2 fused dispatches) |
| Default Metal SDPA for Qwen (not this PR, but the actual tok/s limiter) | decode tok/s | likely 3–10× as ctx grows; 2.1→0.3 is O(n) CPU attention | n/a |

**Honest ceiling:** this PR is a prefill-GEMM and AMX-scheduling paper. It does not contain a decode tok/s win. Espresso's Qwen tok/s problem is CPU attention + blocking N=1 eval. Espresso's Qwen TTFT problem is sequential prefill plus a mis-labeled 21 ms head. Steal the AMX shard recipe and the "tune on real traffic" discipline; do not steal the 14/20/13 split table.

## Proof plan if we implement

1. Add a generate-path counter: `prefill_ms`, `ttft_including_prefill_ms`, `decode_tok_s`, per-kernel `eval_wall_us` vs `hw_ns`.
2. Microbench BNNS FP16 sharded GEMV vs current tiled/NEON head on the 1.5B classifier, bit-identical argmax.
3. One fixed-shape seq=64 Qwen FFN prefill kernel vs the current 64-step N=1 loop on the same prompt IDs.
4. Interleave ON/OFF on a heat-soaked machine. Never quote a tuner percentage without the absolute tok/s and the chunk trace.
5. Keep Qwen greedy parity (`docs/qwen15b-parity.md`) as the quality gate. INT8 ANE prefixes are incompatible with that gate.
