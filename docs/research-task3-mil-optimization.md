# Research Task 3: MIL Operation Optimization and Tile Alignment

**Date:** 2026-03-05
**Branch:** checkpoint/phase8-ane-inference-20260305
**Status:** Complete

---

## 1. Current MIL Operation Inventory

### SDPA Forward Inference (`SDPAForwardInferenceGenerator`)

| # | Operation | MIL Op | Shape | Purpose |
|---|-----------|--------|-------|---------|
| 1 | Square input | `mul(x,x)` | [1,768,1,256] | RMSNorm: x^2 |
| 2 | Sum channels | `reduce_sum` | [1,768,1,256] -> [1,1,1,256] | RMSNorm: sum over dim axis |
| 3 | Scale by 1/dim | `mul` | [1,1,1,256] | RMSNorm: mean of squares |
| 4 | Add epsilon | `add` | [1,1,1,256] | RMSNorm: numerical stability |
| 5 | Power(-0.5) | `pow` | [1,1,1,256] | RMSNorm: 1/sqrt(ms) |
| 6 | Multiply rrms | `mul` | [1,768,1,256] | RMSNorm: normalize |
| 7 | Multiply weights | `mul` | [1,768,1,256] | RMSNorm: scale by learned weights |
| 8 | Conv Wq | `conv` | [768,768,1,1] x [1,768,1,256] -> [1,768,1,256] | Q projection |
| 9 | Conv Wk | `conv` | same | K projection |
| 10 | Conv Wv | `conv` | same | V projection |
| 11 | Reshape Q | `reshape` | [1,768,1,256] -> [1,12,64,256] | Split heads |
| 12 | Transpose Q | `transpose` | [1,12,64,256] -> [1,12,256,64] | Heads x Seq x HeadDim |
| 13 | Reshape K | `reshape` | same as Q | Split heads |
| 14 | Transpose K | `transpose` | same as Q | Heads x Seq x HeadDim |
| 15 | Reshape V | `reshape` | same as Q | Split heads |
| 16 | Transpose V | `transpose` | same as Q | Heads x Seq x HeadDim |
| 17 | Q*K^T matmul | `matmul(ty=true)` | [1,12,256,64] x [1,12,256,64]^T -> [1,12,256,256] | Attention scores |
| 18 | Scale scores | `mul` | [1,12,256,256] | Scale by 1/sqrt(headDim) |
| 19 | Add mask | `add` | [1,12,256,256] | Causal mask |
| 20 | Softmax | `softmax` | [1,12,256,256] | Attention weights |
| 21 | Attn*V matmul | `matmul` | [1,12,256,256] x [1,12,256,64] -> [1,12,256,64] | Weighted values |
| 22 | Transpose out | `transpose` | [1,12,256,64] -> [1,12,64,256] | Restore layout |
| 23 | Reshape out | `reshape` | [1,12,64,256] -> [1,768,1,256] | Merge heads |
| 24 | Conv Wo | `conv` | [768,768,1,1] x [1,768,1,256] -> [1,768,1,256] | Output projection |
| 25 | Residual add | `add` | [1,768,1,256] | x + attn(x) |

**Total: 25 operations** (plus ~10 const definitions)

### FFN Forward Inference (`FFNForwardInferenceGenerator`)

| # | Operation | MIL Op | Shape | Purpose |
|---|-----------|--------|-------|---------|
| 1-7 | RMSNorm | same as SDPA | [1,768,1,256] | Identical RMSNorm block |
| 8 | Conv W1 | `conv` | [2048,768,1,1] x [1,768,1,256] -> [1,2048,1,256] | Up projection |
| 9 | Conv W3 | `conv` | same | Gate projection |
| 10 | Sigmoid | `sigmoid` | [1,2048,1,256] | SiLU part 1 |
| 11 | Mul (SiLU) | `mul` | [1,2048,1,256] | SiLU part 2: x*sigmoid(x) |
| 12 | Mul (gate) | `mul` | [1,2048,1,256] | SiLU(h1) * h3 |
| 13 | Conv W2 | `conv` | [768,2048,1,1] x [1,2048,1,256] -> [1,768,1,256] | Down projection |
| 14 | Residual add | `add` | [1,768,1,256] | x + ffn(x) |

**Total: 14 operations** (plus ~10 const definitions)

### Combined per-layer: 39 operations

---

## 2. Optimization Opportunities

### 2a. Use `scaled_dot_product_attention` MIL Op (iOS 18+)

**Current:** 15 operations for attention (ops 11-25 in SDPA: 3 reshapes, 3 transposes, 2 matmuls, 1 scale, 1 mask add, 1 softmax, 1 transpose, 1 reshape, 1 conv)

**Proposed:** Replace ops 11-23 with a single `scaled_dot_product_attention` op, keeping only the Wo conv and residual add.

The `scaled_dot_product_attention` is available in MIL as an ios18 op. It takes Q, K, V tensors and an optional mask, and fuses the entire matmul-scale-mask-softmax-matmul pipeline into a single operation. The CoreML compiler can map this to a fused GPU kernel; however, **ANE acceleration for this op is uncertain**. The research indicates the fused SDPA is "accelerated on the GPU" rather than ANE. Since we are targeting ANE specifically via `_ANEClient`, this op may not help us -- it could even cause fallback to GPU.

**Risk:** HIGH -- may not run on ANE at all.
**Impact:** If it works on ANE: reduces 13 ops to 1, potentially significant. If not: unusable.
**Difficulty:** MEDIUM -- requires testing whether the compiled E5 binary still targets ANE.

### 2b. Split-Head Attention (Apple's Recommended Pattern)

**Current:** Multi-head attention uses reshape to [1,12,64,256] then transpose and matmul across all 12 heads simultaneously.

**Proposed:** Split into 12 independent single-head attention functions, each operating on [1,1,64,256] slices. This is Apple's recommended pattern from their [Deploying Transformers on ANE](https://machinelearning.apple.com/research/neural-engine-transformers) paper.

**Rationale:**
- Smaller per-head tensors (64x256 = 16K elements at fp16 = 32KB) fit easily in ANE's ~32MB on-chip SRAM
- The full attention score matrix per head is 256x256 = 65K elements = 128KB -- comfortably in L2
- Multi-core utilization: 16 ANE cores can process 12 heads in parallel with better scheduling
- Eliminates the reshape [1,768,1,256] -> [1,12,64,256] which may cause memory copies on ANE's unpacked buffer axes

**Implementation:** After Q/K/V convs, use `split` to divide [1,768,1,256] into 12 x [1,64,1,256] tensors, run attention on each independently (using matmul with channel dimensions directly), then `concat` the results back to [1,768,1,256] for Wo.

**Risk:** LOW -- this is Apple's own recommended approach.
**Impact:** HIGH -- potentially 2-5x improvement for attention block based on Apple's DistilBERT results (up to 10x total).
**Difficulty:** HIGH -- significantly more MIL code generation, need to generate 12 parallel attention paths.

### 2c. Eliminate Reshape/Transpose via Einsum-Style Layout

**Current pattern:** reshape [1,C,1,S] -> [1,H,D,S] -> transpose [1,H,S,D] -> matmul

**Apple's recommended pattern:** Use einsum `bchq,bkhc->bkhq` which maps directly to hardware without intermediate reshape/transpose. Specifically:
- Keep Q in [1,C,1,S] format (channel-first, spatial = sequence)
- Key uses a single transpose before Q*K matmul
- No reshapes at all in the attention computation

This eliminates 6 reshape + 4 transpose operations (10 ops -> ~2 ops).

**Risk:** MEDIUM -- requires careful reformulation of matmul dimensions.
**Impact:** MEDIUM-HIGH -- reshape/transpose on ANE involve data movement (not free metadata ops) because ANE buffers have unpacked axes with 64-byte alignment on the last axis. Reorganizing data across these boundaries requires actual memory copies.
**Difficulty:** MEDIUM -- need to reformulate the matmul to work in [1,C,1,S] space directly.

### 2d. Fuse SiLU as a Single MIL Op

**Current:** `sigmoid(x)` + `mul(x, sigmoid(x))` = 2 operations

**Proposed:** MIL has a built-in `silu` operation. Replace 2 ops with 1:
```
silu_out = silu(x=h1)
```

**Risk:** VERY LOW -- `silu` is a standard MIL op available since iOS 15.
**Impact:** LOW -- saves 1 elementwise op, marginal latency improvement.
**Difficulty:** VERY LOW -- trivial code change.

### 2e. Express RMSNorm More Efficiently

**Current:** 7 operations: mul, reduce_sum, mul(1/dim), add(eps), pow(-0.5), mul(normalize), mul(weights)

**Potential optimizations:**
1. Replace `pow(x, -0.5)` with `rsqrt(x)` -- MIL has `rsqrt` op which computes 1/sqrt(x) directly and is likely hardware-accelerated as a single operation rather than a generic power function.
2. The `reduce_sum` + `mul(1/dim)` could potentially be replaced with `reduce_mean` -- MIL has `reduce_mean` which would fuse these into 1 op.

**Combined:** 7 ops -> 5 ops (replace reduce_sum+mul(1/dim) with reduce_mean, replace pow(-0.5) with rsqrt)

**Risk:** LOW -- both `rsqrt` and `reduce_mean` are standard MIL ops.
**Impact:** LOW-MEDIUM -- saves 2 ops per RMSNorm x 2 RMSNorm per layer = 4 fewer ops per layer.
**Difficulty:** VERY LOW -- direct substitution.

### 2f. Use `linear` Instead of `conv` for Projections

**Current:** All projections (Wq, Wk, Wv, Wo, W1, W2, W3) use `conv` with 1x1 kernels, stride 1, no padding.

**Observation:** The ANE benchmark data from maderix shows that "expressing matmul as 1x1 convolution provides substantially better throughput than native matmul operations" because "the ANE is fundamentally a convolution engine." This means our current approach of using `conv` is already optimal.

**Conclusion:** Keep `conv` for all linear projections. This is the right choice for ANE.

**Risk:** N/A
**Impact:** N/A -- already optimal.

---

## 3. Tile Alignment Analysis

### 3a. Current Dimensions

| Parameter | Value | Divisible by 8? | Divisible by 16? | Divisible by 32? | Divisible by 64? |
|-----------|-------|:---:|:---:|:---:|:---:|
| dim | 768 | YES | YES | YES | YES |
| hidden | 2048 | YES | YES | YES | YES |
| heads | 12 | NO (rem 4) | NO | NO | NO |
| headDim | 64 | YES | YES | YES | YES |
| seqLen | 256 | YES | YES | YES | YES |

### 3b. ANE Alignment Requirements

From research, the key alignment facts are:

1. **Last axis must be contiguous and aligned to 64 bytes.** For fp16 (2 bytes), this means the last axis must be a multiple of 32 elements. For fp32 (4 bytes), a multiple of 16 elements.

2. **If the last axis is a singleton (size 1), it gets padded to 64 bytes** -- causing 32x memory bloat in fp16. This is why `[1, C, 1, S]` layout is used rather than `[1, C, S, 1]`.

3. **ANE hardware processes in tiles of 8** according to the KV-cache research ("64 is also a multiple of 8, which aligns with the ANE hardware").

4. **Channel axis should be a multiple of 8** for efficient processing (inferred from the 8-wide tile observation and the fact that ANE has 16 cores).

### 3c. Alignment Assessment of Current Layout

**Input/Output tensors: `[1, 768, 1, 256]`**
- Last axis (S=256): 256 * 2 bytes = 512 bytes. 512 / 64 = 8. PERFECTLY ALIGNED.
- Channel axis (C=768): 768 / 8 = 96. PERFECTLY ALIGNED.

**After reshape for attention: `[1, 12, 64, 256]`**
- Last axis (S=256): ALIGNED (same as above).
- But this 4D shape with dim-1=12 and dim-2=64 may have suboptimal packing. The ANE compiler decides packing for middle axes.

**After transpose: `[1, 12, 256, 64]`**
- Last axis (D=64): 64 * 2 = 128 bytes. 128 / 64 = 2. ALIGNED.
- This is fine for alignment.

**Attention scores: `[1, 12, 256, 256]`**
- Last axis (S=256): ALIGNED.
- Total size per head: 256 * 256 * 2 = 128KB. Fits comfortably in ANE SRAM.

**FFN hidden: `[1, 2048, 1, 256]`**
- Last axis (S=256): ALIGNED.
- Channel axis (C=2048): 2048 / 8 = 256. PERFECTLY ALIGNED.

### 3d. Alignment Verdict

**All current dimensions are well-aligned for ANE processing.** The model config (dim=768, hidden=2048, headDim=64, seqLen=256) is naturally ANE-friendly. No padding is needed.

The `[1, C, 1, S]` layout is correct -- it keeps the sequence length on the last axis (avoiding the singleton-axis padding penalty) and channels on the second axis (matching ANE's convolution-native format).

---

## 4. Reshape/Transpose Cost Analysis

### Are reshape/transpose free on ANE?

**No. Reshape and transpose are NOT free on ANE.** Based on multiple sources:

1. **Apple's own guidance:** "the implementation minimizes memory operations by avoiding all reshapes in the attention mechanism" and using "only one transpose on the key tensor." If these were free, Apple would not recommend avoiding them.

2. **64-byte alignment constraint:** ANE buffers have the last axis "contiguous and aligned to 64 bytes" and unpacked. When reshape changes which logical dimension maps to the last physical axis, data must be physically rearranged to satisfy alignment constraints.

3. **Practical evidence:** The KV-cache research identifies transpose as "costly" and recommends minimizing them.

4. **Precision impact:** The more-ane-transformers NOTES.md reports "Reshaping and transposing operations cause noticeable precision loss" -- indicating these involve actual data movement and fp16 reprocessing.

### Cost Estimate

Each reshape/transpose likely costs:
- Small tensor (e.g., [1,64,1,256]): ~0.01-0.05ms (dominated by dispatch overhead)
- Medium tensor (e.g., [1,768,1,256]): ~0.05-0.1ms
- Large tensor (e.g., [1,12,256,256]): ~0.1-0.2ms

**Current SDPA uses 6 reshapes + 4 transposes = 10 data movement operations.** At an estimated average of 0.03-0.05ms each, this could account for 0.3-0.5ms of the ~0.791ms per-layer latency -- a significant fraction.

---

## 5. Concrete Recommendations (Prioritized)

### Priority 1: Split-Head Attention [HIGH IMPACT, HIGH EFFORT]

**Expected speedup:** 2-5x for the attention block (based on Apple's published results).

Split multi-head attention into 12 independent single-head paths. This:
- Eliminates 3 reshape + 3 transpose ops before attention (replaced by a single `split` along channel axis)
- Eliminates 1 reshape + 1 transpose after attention (replaced by a single `concat`)
- Enables better L2 cache utilization (per-head working set: ~32KB vs ~768KB combined)
- Enables better multi-core utilization (12 heads across 16 cores)

### Priority 2: RMSNorm Optimization [LOW-MEDIUM IMPACT, VERY LOW EFFORT]

Replace `reduce_sum + mul(1/dim)` with `reduce_mean`, and `pow(-0.5)` with `rsqrt`.

Saves 4 ops per layer (2 per RMSNorm x 2 RMSNorm). Zero risk.

### Priority 3: Use `silu` Op [LOW IMPACT, VERY LOW EFFORT]

Replace `sigmoid(x)` + `mul(x, sig)` with `silu(x)`. Saves 1 op per layer.

### Priority 4: Reshape-Free Attention Layout [MEDIUM-HIGH IMPACT, MEDIUM EFFORT]

If split-head attention (Priority 1) is not pursued, reformulate the attention computation to minimize reshape/transpose operations using Apple's einsum-style approach:
- Keep data in [1,C,1,S] format
- Use a single transpose on K only
- Express matmul to work with channel-first format directly

### Priority 5: Explore `scaled_dot_product_attention` Op [UNCERTAIN IMPACT, MEDIUM EFFORT]

Test whether the ios18 `scaled_dot_product_attention` MIL op compiles and runs on ANE (not GPU). If it does, this is the simplest path to fusing 13 attention ops into 1. Requires empirical testing.

---

## 6. Implementation Sketches

### Sketch A: RMSNorm Optimization (Priority 2)

**Current:**
```
sq = mul(x=x, y=x)
rax = const([1])
kd = const(true)
ss = reduce_sum(x=sq, axes=rax, keep_dims=kd)    // sum over channels
invd = const(1/768)
ss2 = mul(x=ss, y=invd)                            // mean of squares
eps = const(0.00001)
ss3 = add(x=ss2, y=eps)
nhalf = const(-0.5)
rrms = pow(x=ss3, y=nhalf)                         // 1/sqrt(mean_sq + eps)
xr = mul(x=x, y=rrms)
rw = const(weights)
xn = mul(x=xr, y=rw)
```

**Optimized (saves 2 ops):**
```
sq = mul(x=x, y=x)
rax = const([1])
kd = const(true)
ms = reduce_mean(x=sq, axes=rax, keep_dims=kd)    // FUSED: mean of squares
eps = const(0.00001)
ms2 = add(x=ms, y=eps)
rrms = rsqrt(x=ms2)                                // FUSED: 1/sqrt directly
xr = mul(x=x, y=rrms)
rw = const(weights)
xn = mul(x=xr, y=rw)
```

### Sketch B: SiLU Fusion (Priority 3)

**Current:**
```
sig = sigmoid(x=h1)
silu = mul(x=h1, y=sig)
gate = mul(x=silu, y=h3)
```

**Optimized (saves 1 op):**
```
silu_out = silu(x=h1)       // Built-in MIL op: x * sigmoid(x)
gate = mul(x=silu_out, y=h3)
```

### Sketch C: Split-Head Attention (Priority 1)

**Current (simplified):**
```
// After QKV convs, we have qf, kf, vf each [1, 768, 1, 256]
q4 = reshape([1,12,64,256], qf)    // split heads
q = transpose([0,1,3,2], q4)        // [1,12,256,64]
k4 = reshape([1,12,64,256], kf)
k = transpose([0,1,3,2], k4)
v4 = reshape([1,12,64,256], vf)
v = transpose([0,1,3,2], v4)
sc1 = matmul(q, k^T)                // [1,12,256,256]
sc2 = mul(sc1, scale)
ms = add(sc2, mask)
aw = softmax(ms)
a4 = matmul(aw, v)                  // [1,12,256,64]
at = transpose(a4)                   // [1,12,64,256]
af = reshape([1,768,1,256], at)      // merge heads
```

**Proposed split-head approach:**
```
// After QKV convs, split channel dimension into 12 heads
// Using split op to divide [1,768,1,256] into 12 x [1,64,1,256]
tensor<int32, [12]> splits = const([64,64,64,64,64,64,64,64,64,64,64,64]);
int32 split_dim = const(1);  // split along channel axis

// Split Q, K, V
q_parts = split(axis=split_dim, split_sizes=splits, x=qf)  // 12 x [1,64,1,256]
k_parts = split(axis=split_dim, split_sizes=splits, x=kf)
v_parts = split(axis=split_dim, split_sizes=splits, x=vf)

// For each head i (0..11):
//   q_i: [1,64,1,256], treat as [1,64,256] via reshape or view
//   k_i: [1,64,1,256]
//   Scores: matmul(q_i^T, k_i) needs [1,256,64] x [1,64,256] -> [1,256,256]
//   Only ONE transpose needed: on Q (or K), not both
//
//   q_i_t = transpose([0,2,1,3], q_i)  // [1,256,64,1] -- but need to handle 4D carefully
//
//   Alternative: reshape to 3D for cleaner matmul
//   q_3d = reshape([1,64,256], q_i)
//   k_3d = reshape([1,64,256], k_i)
//   scores = matmul(tx=true, ty=false, q_3d, k_3d)  // [1,256,256]
//   -- apply scale, mask, softmax --
//   v_3d = reshape([1,64,256], v_i)
//   attn = matmul(tx=false, ty=true, scores, v_3d)   // [1,256,64]
//   attn_ch = transpose/reshape back to [1,64,1,256]

// Concat all 12 heads back to [1,768,1,256]
af = concat(axis=1, values=[a0, a1, ..., a11])
oo = conv(Wo, af)  // output projection
```

**Key advantages of this approach:**
- Each head processes ~32KB working set (vs 384KB combined) -- fits in L2
- Only 1 transpose per head needed (vs 2 in current code)
- 12 heads can be scheduled across 16 ANE cores
- `split` and `concat` along the channel axis in [1,C,1,S] format are likely efficient (channel axis is the natural packing axis)

**Caveats:**
- Generates significantly more MIL text (~12x the attention ops)
- Need to verify that `split` along dim=1 in [1,C,1,S] layout is ANE-efficient
- The 12 causal masks become 12 identical [1,1,256,256] references (or one shared const)

---

## 7. Working Set / SRAM Analysis

The ANE has approximately 32MB of on-chip SRAM. Key working set sizes:

| Tensor | Shape | Size (fp16) | Fits in 32MB SRAM? |
|--------|-------|-------------|:---:|
| Input x | [1,768,1,256] | 384 KB | YES |
| Wq/Wk/Wv/Wo each | [768,768,1,1] | 1.125 MB | YES |
| All 4 attention weights | - | 4.5 MB | YES |
| Q*K^T scores (all heads) | [1,12,256,256] | 1.5 MB | YES |
| Q*K^T scores (per head) | [1,256,256] | 128 KB | YES |
| FFN W1/W3 each | [2048,768,1,1] | 3 MB | YES |
| FFN W2 | [768,2048,1,1] | 3 MB | YES |
| FFN hidden activations | [1,2048,1,256] | 1 MB | YES |
| **Total per layer** | | **~15 MB** | YES |

The entire layer working set (~15MB) fits in the 32MB SRAM, so we should not be hitting DRAM for any intermediate tensors. This is good -- it means our bottleneck is compute/dispatch overhead, not memory bandwidth.

However, with split-head attention, per-head working sets drop to ~1.2MB total (weights + activations), maximizing L2 cache residency and enabling much better pipelining across the 16 cores.

---

## 8. Key Findings Summary

1. **Dimensions are well-aligned.** dim=768, hidden=2048, headDim=64, seqLen=256 are all multiples of 64 (the 64-byte alignment threshold for fp16). No padding needed.

2. **`[1,C,1,S]` layout is correct.** This matches ANE's native IOSurface format and avoids the singleton-axis 32x padding penalty.

3. **Reshape/transpose are NOT free.** They involve actual data movement on ANE due to the 64-byte last-axis alignment requirement. The current SDPA uses 10 reshape/transpose ops, which may account for 30-60% of per-layer latency.

4. **Conv for linear projections is optimal.** ANE is fundamentally a convolution engine. 1x1 conv outperforms native matmul on ANE.

5. **Split-head attention is the biggest win.** Apple's own research shows up to 10x improvement. It reduces data movement ops and improves L2 cache utilization.

6. **Quick wins exist:** `reduce_mean` for RMSNorm, `rsqrt` instead of `pow(-0.5)`, `silu` op -- collectively save 5 ops per layer with near-zero risk.

7. **Graph depth matters.** Deep computation graphs (16-64 chained ops) achieve 94% hardware utilization vs ~30% for single ops. Our current 25+14=39 ops per layer should be deep enough, but fusing to fewer ops could hurt if it reduces graph depth below the optimal range.

---

## Sources

- [Deploying Transformers on the Apple Neural Engine - Apple ML Research](https://machinelearning.apple.com/research/neural-engine-transformers)
- [Inside the M4 Apple Neural Engine, Part 1 - Maderix](https://maderix.substack.com/p/inside-the-m4-apple-neural-engine)
- [Inside the M4 Apple Neural Engine, Part 2: ANE Benchmarks - Maderix](https://maderix.substack.com/p/inside-the-m4-apple-neural-engine-615)
- [Everything we know about the Apple Neural Engine - Hollance](https://github.com/hollance/neural-engine)
- [In Pursuit of Fast KV-Cached Attention for ANE - Stephen Panaro](https://stephenpanaro.com/blog/kv-cache-for-neural-engine)
- [More ANE Transformers Experiments Notes - smpanaro](https://github.com/smpanaro/more-ane-transformers/blob/main/src/experiments/NOTES.md)
- [Apple ml-ane-transformers Reference Implementation](https://github.com/apple/ml-ane-transformers)
- [MIL Ops - coremltools API Reference 8.1](https://apple.github.io/coremltools/source/coremltools.converters.mil.mil.ops.defs.html)
- [MIL Graph Passes - coremltools API Reference 8.1](https://apple.github.io/coremltools/source/coremltools.converters.mil.mil.passes.defs.html)
