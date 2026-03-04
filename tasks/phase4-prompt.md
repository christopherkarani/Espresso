# Phase 4: CPUOps — Accelerate/vDSP Operations

<role>
You are a senior Swift 6.2 engineer specializing in numerical computing with Apple's Accelerate framework (vDSP, vecLib, cblas). You write mathematically correct, performance-equivalent ports of C numerical code into idiomatic Swift. You follow strict TDD — tests written and committed first, implementation second. You understand fp32 precision limits and design tests with appropriate tolerances. You treat vDSP parameter ordering as a safety-critical concern.
</role>

---

<context>

## Project Overview

This is Phase 4 of a 6-phase bottom-up Swift 6.2 rewrite of an Apple Neural Engine training codebase (~6,100 lines Obj-C/C). The codebase trains a 12-layer Llama2 transformer on Apple Neural Engine at 9.3 ms/step on M4.

**Completed phases** (all tests green, 74 tests, 0 failures):
- Phase 1: `ANEInterop` — C/ObjC bridging (17 tests)
- Phase 2: `ANETypes` — Swift types + IOSurface I/O (28 tests)
- Phase 3: `MILGenerator` — MIL text generation (29 tests)

**This phase**: `CPUOps` — pure Swift target, **zero C interop**. All operations use `import Accelerate` directly from Swift. Depends only on `ANETypes` for `ModelConfig` constants and `TensorBuffer`.

**Phases 3 and 4 are parallel** — no dependency between them. Phase 5 (`ANERuntime`) depends on both.

## What You Are Building

**SwiftPM target**: `CPUOps` (depends on `ANETypes`)
**Location**: `Sources/CPUOps/`
**Test target**: `Tests/CPUOpsTests/`
**Current state**: `Sources/CPUOps/CPUOps.swift` contains only `public enum CPUOps {}` — a placeholder.

Six Swift files, each an enum namespace with static methods:

| File | Operations | C Source | Data Layout |
|------|-----------|---------|-------------|
| `RMSNorm.swift` | `forward()` + `backward()` | `stories_cpu_ops.h:7-54` | Channel-first `[DIM, SEQ]` |
| `CrossEntropy.swift` | `lossAndGradient()` | `stories_cpu_ops.h:72-108` | Column-major `[VOCAB, SEQ]` |
| `AdamOptimizer.swift` | `update()` | `stories_cpu_ops.h:56-64` | Flat array |
| `Embedding.swift` | `lookup()` + `backward()` | `stories_cpu_ops.h:112-129` | Channel-first `[DIM, SEQ]` |
| `RoPE.swift` | `apply()` + `backward()` | `forward.h:50-65` + `backward.h:104-119` | Row-major `[SEQ, nHeads*headDim]` |
| `SiLU.swift` | `forward()` + `backward()` | `forward.h:96` + `backward.h:44-47` | Scalar (element-wise) |

## Two Data Layouts — Know Which Is Which

**Channel-first `[DIM, SEQ]`** — used by production code in `train_large.m`:
```
data[dim_index * SEQ + seq_index]
```
vDSP vectorizes along the SEQ dimension (stride 1, contiguous). RMSNorm, CrossEntropy, and Embedding use this layout.

**Row-major `[SEQ, DIM]`** — used by reference code in `forward.h`/`backward.h`:
```
data[seq_index * DIM + dim_index]
```
RoPE uses this layout because the ANE fused kernels output Q/K in this format.

Mixing these layouts produces **silent data corruption** — no error, just wrong values. Every function signature documents which layout it expects.

## Model Constants

From `ANETypes.ModelConfig` (already implemented in Phase 2):
```swift
public enum ModelConfig {
    public static let dim = 768        // model dimension
    public static let hidden = 2048    // FFN hidden dimension
    public static let heads = 12       // attention heads
    public static let headDim = 64     // dim / heads
    public static let seqLen = 256     // sequence length
    public static let nLayers = 12     // transformer layers
    public static let vocab = 32_000   // vocabulary size
}
```

Functions accept dimensions as parameters — they do not hardcode `ModelConfig` values.

## TensorBuffer API (from Phase 2, for reference)

```swift
public struct TensorBuffer: ~Copyable {
    public let count: Int
    public init(count: Int, zeroed: Bool)
    public func withUnsafeMutablePointer<R>(_ body: (UnsafeMutablePointer<Float>) throws -> R) rethrows -> R
    public func withUnsafePointer<R>(_ body: (UnsafePointer<Float>) throws -> R) rethrows -> R
    public func withUnsafeMutableBufferPointer<R>(_ body: (UnsafeMutableBufferPointer<Float>) throws -> R) rethrows -> R
    public func withUnsafeBufferPointer<R>(_ body: (UnsafeBufferPointer<Float>) throws -> R) rethrows -> R
    public func zero()
    deinit // deallocates storage
}
```

Tests may use `TensorBuffer` for convenience, or raw `UnsafeMutablePointer<Float>.allocate(capacity:)` — your choice.

## Production Call Sites (from train_large.m)

These CPU ops are called in the production training loop:
```
Line 381: embed_lookup(x_cur, embed, input_tokens, DIM, SEQ)
Line 424: rmsnorm(x_final, x_cur, rms_final, DIM, SEQ)
Line 434: cross_entropy_loss(dlogits, logits, target_tokens, VOCAB, SEQ)
Line 456: rmsnorm_bwd(dx_rms_final, grms_final, dy, x_cur, rms_final, DIM, SEQ)
Line 495: rmsnorm_bwd(dx2, gr->rms_ffn, dx_ffn, ac->x2, lw[L].rms_ffn, DIM, SEQ)
Line 544: rmsnorm_bwd(dx_rms1, gr->rms_att, dx_attn, ac->layer_in, lw[L].rms_att, DIM, SEQ)
Line 579: embed_backward(gembed, dy, input_tokens, DIM, SEQ)
Lines 614-628: adam_update(...) called 9 times per layer + 3 times for globals
```

RoPE and SiLU are NOT called on CPU in production — they are computed inside fused ANE kernels. They are ported as reference implementations for testing and CPU fallback in Phase 5-6.

</context>

---

<source_code>

## File 1: stories_cpu_ops.h — Production CPU Operations (EXACT C to match)

```c
// stories_cpu_ops.h — CPU operations: RMSNorm, cross-entropy, Adam, softmax
#pragma once
#include "stories_config.h"

static float *g_rms_tmp = NULL;

// RMSNorm forward — channel-first layout [DIM, SEQ]
// x[i*S + t] = dimension i, position t
// w[i] = per-dimension weight
// out[i*S + t] = normalized output
static void rmsnorm(float *out, const float *x, const float *w, int d, int S) {
    if (!g_rms_tmp) g_rms_tmp = (float*)malloc(S*4);
    float *ss = (float*)calloc(S, sizeof(float));
    for (int i=0; i<d; i++) {
        vDSP_vmul(x+i*S, 1, x+i*S, 1, g_rms_tmp, 1, (vDSP_Length)S);
        vDSP_vadd(g_rms_tmp, 1, ss, 1, ss, 1, (vDSP_Length)S);
    }
    float invd = 1.0f/d, eps=1e-5f;
    vDSP_vsmsa(ss, 1, &invd, &eps, ss, 1, (vDSP_Length)S);
    int n = S; vvrsqrtf(ss, ss, &n);
    for (int i=0; i<d; i++) {
        vDSP_vmul(x+i*S, 1, ss, 1, out+i*S, 1, (vDSP_Length)S);
        vDSP_vsmul(out+i*S, 1, &w[i], out+i*S, 1, (vDSP_Length)S);
    }
    free(ss);
}

// RMSNorm backward — channel-first layout [DIM, SEQ]
// dx[i*S+t], dw[i], dy[i*S+t], x[i*S+t], w[i]
// NOTE: dw is ACCUMULATED (dw[i] += s), not overwritten
static void rmsnorm_bwd(float *dx, float *dw, const float *dy,
                        const float *x, const float *w, int d, int S) {
    if (!g_rms_tmp) g_rms_tmp = (float*)malloc(S*4);
    float *ss = (float*)calloc(S, sizeof(float));
    for (int i=0; i<d; i++) {
        vDSP_vmul(x+i*S, 1, x+i*S, 1, g_rms_tmp, 1, (vDSP_Length)S);
        vDSP_vadd(g_rms_tmp, 1, ss, 1, ss, 1, (vDSP_Length)S);
    }
    float invd = 1.0f/d, eps=1e-5f;
    vDSP_vsmsa(ss, 1, &invd, &eps, ss, 1, (vDSP_Length)S);
    float *rrms = (float*)malloc(S*4);
    int n = S; vvrsqrtf(rrms, ss, &n);
    float *dot = (float*)calloc(S, sizeof(float));
    for (int i=0; i<d; i++) {
        vDSP_vmul(dy+i*S, 1, x+i*S, 1, g_rms_tmp, 1, (vDSP_Length)S);
        vDSP_vsma(g_rms_tmp, 1, &w[i], dot, 1, dot, 1, (vDSP_Length)S);
    }
    vDSP_vmul(rrms, 1, rrms, 1, ss, 1, (vDSP_Length)S);       // ss = rrms^2
    vDSP_vsmul(ss, 1, &invd, ss, 1, (vDSP_Length)S);           // ss = rrms^2 / d
    vDSP_vmul(dot, 1, ss, 1, dot, 1, (vDSP_Length)S);          // dot = dot * rrms^2 / d
    for (int i=0; i<d; i++) {
        vDSP_vmul(x+i*S, 1, dot, 1, g_rms_tmp, 1, (vDSP_Length)S);  // tmp = x * dot
        vDSP_vsub(g_rms_tmp, 1, dy+i*S, 1, g_rms_tmp, 1, (vDSP_Length)S); // tmp = dy - tmp
        vDSP_vmul(g_rms_tmp, 1, rrms, 1, g_rms_tmp, 1, (vDSP_Length)S);   // tmp = (dy-x*dot)*rrms
        vDSP_vsmul(g_rms_tmp, 1, &w[i], dx+i*S, 1, (vDSP_Length)S);       // dx = tmp * w
        vDSP_vmul(dy+i*S, 1, x+i*S, 1, g_rms_tmp, 1, (vDSP_Length)S);     // tmp = dy * x
        vDSP_vmul(g_rms_tmp, 1, rrms, 1, g_rms_tmp, 1, (vDSP_Length)S);   // tmp = dy * x * rrms
        float s; vDSP_sve(g_rms_tmp, 1, &s, (vDSP_Length)S);               // s = sum(dy*x*rrms)
        dw[i] += s;                                                         // ACCUMULATE
    }
    free(ss); free(rrms); free(dot);
}

// Adam optimizer — per-parameter update with bias correction
// timestep t starts at 1 (NOT 0)
static void adam_update(float *w, const float *g, AdamState *s,
                        int t, float lr, float b1, float b2, float eps) {
    float bc1 = 1.0f - powf(b1, t), bc2 = 1.0f - powf(b2, t);
    for (size_t i=0; i<s->n; i++) {
        s->m[i] = b1*s->m[i] + (1-b1)*g[i];
        s->v[i] = b2*s->v[i] + (1-b2)*g[i]*g[i];
        float mh = s->m[i]/bc1, vh = s->v[i]/bc2;
        w[i] -= lr * mh / (sqrtf(vh) + eps);
    }
}

// Cross-entropy loss + gradient for logits (column-major: [VOCAB, SEQ])
// logits[v*SEQ+t] = logit for vocab v, position t
// targets[t] = target token id for position t
// Returns mean CE loss, writes dlogits = (softmax(logits) - one_hot(targets)) / S
static float cross_entropy_loss(float *dlogits, const float *logits,
                                const uint16_t *targets, int V, int S) {
    // Transpose [V,S] → [S,V] for per-position softmax (contiguous rows)
    float *buf = (float*)malloc(S * V * 4);
    vDSP_mtrans(logits, 1, buf, 1, (vDSP_Length)S, (vDSP_Length)V);
    float total_loss = 0;
    float invS = 1.0f / S;
    for (int t = 0; t < S; t++) {
        float *row = buf + t * V;
        float maxv;
        vDSP_maxv(row, 1, &maxv, (vDSP_Length)V);
        float neg_max = -maxv;
        vDSP_vsadd(row, 1, &neg_max, row, 1, (vDSP_Length)V);
        int n = V;
        vvexpf(row, row, &n);
        float sum;
        vDSP_sve(row, 1, &sum, (vDSP_Length)V);
        float inv_sum = 1.0f / sum;
        vDSP_vsmul(row, 1, &inv_sum, row, 1, (vDSP_Length)V);
        int tgt = targets[t];
        total_loss -= logf(row[tgt] + 1e-10f);
        row[tgt] -= 1.0f;
        vDSP_vsmul(row, 1, &invS, row, 1, (vDSP_Length)V);
    }
    // Transpose back [S,V] → [V,S]
    vDSP_mtrans(buf, 1, dlogits, 1, (vDSP_Length)V, (vDSP_Length)S);
    free(buf);
    return total_loss / S;
}

// Embedding lookup — channel-first: x[d*seq + t] = embed[tok*dim + d]
// embed: [VOCAB, DIM] row-major. output: [DIM, SEQ] channel-first.
static void embed_lookup(float *x, const float *embed,
                         const uint16_t *tokens, int dim, int seq) {
    for (int t = 0; t < seq; t++) {
        int tok = tokens[t];
        for (int d = 0; d < dim; d++)
            x[d*seq + t] = embed[tok*dim + d];
    }
}

// Embedding backward — channel-first: d_embed[tok*dim + d] += dx[d*seq + t]
// NOTE: ACCUMULATES into d_embed, does not zero first
static void embed_backward(float *d_embed, const float *dx,
                           const uint16_t *tokens, int dim, int seq) {
    for (int t = 0; t < seq; t++) {
        int tok = tokens[t];
        for (int d = 0; d < dim; d++)
            d_embed[tok*dim + d] += dx[d*seq + t];
    }
}
```

## File 2: forward.h — Reference Operations (Row-Major)

```c
// RoPE forward — row-major: data[t * n_heads * head_dim + h * head_dim + i]
// Mutates q and k IN PLACE
static void cpu_rope(float *q, float *k, int S, int n_heads, int head_dim) {
    for (int t = 0; t < S; t++)
        for (int h = 0; h < n_heads; h++)
            for (int i = 0; i < head_dim; i += 2) {
                float freq = 1.0f / powf(10000.0f, (float)i / head_dim);
                float val = t * freq;
                float cos_v = cosf(val), sin_v = sinf(val);
                int off = t * n_heads * head_dim + h * head_dim + i;
                float q0 = q[off], q1 = q[off+1];
                q[off]   = q0 * cos_v - q1 * sin_v;
                q[off+1] = q0 * sin_v + q1 * cos_v;
                float k0 = k[off], k1 = k[off+1];
                k[off]   = k0 * cos_v - k1 * sin_v;
                k[off+1] = k0 * sin_v + k1 * cos_v;
            }
}

// SiLU scalar
static inline float silu_f(float x) { return x / (1.0f + expf(-x)); }
```

## File 3: backward.h — Reference Backward Operations (Row-Major)

```c
// SiLU backward scalar
static inline float silu_backward(float x) {
    float s = 1.0f / (1.0f + expf(-x));
    return s * (1.0f + x * (1.0f - s));
}

// RoPE backward — TRANSPOSED rotation matrix: (cos, +sin; -sin, cos)
// Forward uses:  q_new = q0*cos - q1*sin,  q_new+1 = q0*sin + q1*cos
// Backward uses: dq_new = dq0*cos + dq1*sin, dq_new+1 = -dq0*sin + dq1*cos
static void cpu_rope_backward(float *dq, float *dk, int S, int n_heads, int head_dim) {
    for (int t = 0; t < S; t++)
        for (int h = 0; h < n_heads; h++)
            for (int i = 0; i < head_dim; i += 2) {
                float freq = 1.0f / powf(10000.0f, (float)i / head_dim);
                float val = t * freq;
                float cos_v = cosf(val), sin_v = sinf(val);
                int off = t * n_heads * head_dim + h * head_dim + i;
                float dq0 = dq[off], dq1 = dq[off+1];
                dq[off]   = dq0 * cos_v + dq1 * sin_v;    // +sin (transpose)
                dq[off+1] = -dq0 * sin_v + dq1 * cos_v;   // -sin (transpose)
                float dk0 = dk[off], dk1 = dk[off+1];
                dk[off]   = dk0 * cos_v + dk1 * sin_v;
                dk[off+1] = -dk0 * sin_v + dk1 * cos_v;
            }
}
```

</source_code>

---

<instructions>

Follow this EXACT sequence. Complete each step fully before moving to the next. Keep working until all 10 tests pass — do not stop at the first failure.

## Step 1: Delete the Placeholder

Remove `Sources/CPUOps/CPUOps.swift` — it contains only `public enum CPUOps {}` and is not needed.

## Step 2: Write All Tests First (TDD)

Create `Tests/CPUOpsTests/CPUOpsTests.swift`. Write all 10 tests BEFORE any implementation code. Use a seeded RNG for deterministic random tests:

```swift
// Deterministic RNG for reproducible tests
struct SplitMix64: RandomNumberGenerator {
    private var state: UInt64
    init(seed: UInt64) { state = seed }
    mutating func next() -> UInt64 {
        state &+= 0x9e3779b97f4a7c15
        var z = state
        z = (z ^ (z >> 30)) &* 0xbf58476d1ce4e5b9
        z = (z ^ (z >> 27)) &* 0x94d049bb133111eb
        return z ^ (z >> 31)
    }
}
```

### Test 1: `test_rmsnorm_forward_known_values`
- dim=4, seq=2. Channel-first layout.
- Input x: `[1,2, 3,4, 5,6, 7,8]` meaning x[0*2+0]=1, x[0*2+1]=2, x[1*2+0]=3, x[1*2+1]=4, ...
- Weights w: `[1.0, 1.0, 1.0, 1.0]`
- Hand-compute per position: `ss[t] = sum(x[i*2+t]^2 for i in 0..<4)`, then `ss[t] = ss[t]/4 + 1e-5`, then `rrms[t] = 1/sqrt(ss[t])`, then `out[i*2+t] = x[i*2+t] * rrms[t] * w[i]`
- Position t=0: ss=1+9+25+49=84, ss/4+1e-5=21.00001, rrms=1/sqrt(21.00001)≈0.218218
- Position t=1: ss=4+16+36+64=120, ss/4+1e-5=30.00001, rrms=1/sqrt(30.00001)≈0.182574
- Tolerance: 1e-5

### Test 2: `test_rmsnorm_backward_numerical_gradient_check`
- dim=4, seq=2. Finite difference vs analytic backward.
- Perturb each input element by h=1e-4: `grad_approx = (f(x+h) - f(x-h)) / (2*h)`
- Use a seeded RNG (seed=42) for random inputs, weights, and upstream gradient dy.
- Run the check on 20 random configurations. Relative error < 1e-3 for dx.
- Also verify dw accumulates (pre-set dw to non-zero, confirm it adds, not overwrites).

### Test 3: `test_cross_entropy_uniform_logits`
- V=100, S=4, all logits = 0.0, targets = [0, 1, 2, 3]
- With uniform softmax, P(each) = 1/100, loss per position = -log(1/100) = log(100)
- Expected mean loss: log(100) ≈ 4.60517
- Tolerance: 1e-4

### Test 4: `test_cross_entropy_gradient_sums_to_zero`
- V=50, S=8, random logits (seeded), random targets in [0, V)
- For each position t, sum dlogits[v*S + t] across all v.
- The gradient is (softmax - one_hot) / S. sum(softmax) = 1, sum(one_hot) = 1, so sum = 0.
- Verify |sum| < 1e-5 for each position.

### Test 5: `test_adam_single_step_known_values`
- count=1, t=1, w=[1.0], g=[0.1], m=[0], v=[0], lr=0.001, b1=0.9, b2=0.999, eps=1e-8
- After step: m=0.01, v=0.00001, bc1=0.1, bc2=0.001, m_hat=0.1, v_hat=0.01
- w_new = 1.0 - 0.001 * 0.1 / (sqrt(0.01) + 1e-8) = 1.0 - 0.001 * 0.1 / 0.1 = 1.0 - 0.001 = 0.999
- Tolerance: 1e-6

### Test 6: `test_adam_bias_correction`
- count=1, constant g=1.0, run 200 steps.
- After many steps: m → 1.0 (since g is constant), v → 1.0, bias corrections → 1.0.
- Verify at step 200: |m_hat - 1.0| < 0.01 and |v_hat - 1.0| < 0.01.

### Test 7: `test_embedding_lookup_correct_rows`
- dim=4, seq=3, vocab=5, tokens=[3, 1, 0]
- Create embedding table with known values: embed[tok*dim + d] = Float(tok * 100 + d)
- Verify channel-first output: x[d*seq + t] == embed[tokens[t]*dim + d] for all d, t.

### Test 8: `test_embedding_backward_accumulates`
- dim=4, seq=3, vocab=5, tokens=[2, 2, 1]
- dx = all 1.0 (channel-first)
- Pre-set d_embed to all 0.5 (tests accumulation, not overwrite)
- After backward: d_embed[2*dim + d] should be 0.5 + 2.0 = 2.5 (two positions for token 2)
- d_embed[1*dim + d] should be 0.5 + 1.0 = 1.5 (one position for token 1)
- d_embed[0*dim + d] should remain 0.5 (unused token)

### Test 9: `test_rope_forward_backward_consistency`
- seq=4, nHeads=2, headDim=8 (small enough to verify, pairs at i=0,2,4,6)
- Random q, k (seeded RNG, seed=42). Save originals.
- Apply forward (mutates q, k). Save post-forward.
- Apply backward to post-forward copies.
- Verify backward result matches original within 1e-5 (rotation is orthogonal: R^T @ R = I).

### Test 10: `test_silu_forward_backward_consistency`
- 100 random values in [-5, 5] (seeded RNG, seed=42)
- For each x: numerical derivative = (silu(x+h) - silu(x-h)) / (2*h) with h=1e-4
- Compare against SiLU.backward(x). Relative error < 1e-3 where |analytic| > 1e-6.

## Step 3: Implement Each Operation

Implement six files in `Sources/CPUOps/`. Match the C code's vDSP call sequence exactly — operation order affects numerical results.

### 3a: RMSNorm.swift

```swift
import Accelerate

public enum RMSNorm {
    /// Channel-first forward: x[dim, seq], w[dim] → out[dim, seq]
    public static func forward(
        output: UnsafeMutablePointer<Float>,
        input: UnsafePointer<Float>,
        weights: UnsafePointer<Float>,
        dim: Int,
        seqLen: Int
    )

    /// Channel-first backward: computes dx, ACCUMULATES into dw (dw[i] += ...)
    public static func backward(
        dx: UnsafeMutablePointer<Float>,
        dw: UnsafeMutablePointer<Float>,
        dy: UnsafePointer<Float>,
        x: UnsafePointer<Float>,
        weights: UnsafePointer<Float>,
        dim: Int,
        seqLen: Int
    )
}
```

Implementation rules:
- Use the SAME vDSP call sequence as the C code. Do not "simplify" or reorder — numerical equivalence depends on exact operation ordering.
- Allocate local scratch buffers per call (replacing C's global `g_rms_tmp`). No mutable statics.
- `vvrsqrtf` signature in Swift: `vvrsqrtf(result, input, &count)` where count is `inout Int32`.
- `vDSP_vsub` has REVERSED semantics: `vDSP_vsub(B, 1, A, 1, C, 1, n)` computes `C = A - B`. The C code at line 45 does `vDSP_vsub(g_rms_tmp, dy+i*S, g_rms_tmp, S)` which means `tmp = dy - tmp`. Match this precisely.
- Backward ACCUMULATES dw: `dw[i] += s`. Document this in a comment.

### 3b: CrossEntropy.swift

```swift
public enum CrossEntropy {
    /// Column-major logits [vocab, seq]. Returns mean CE loss. Writes gradient into dlogits.
    public static func lossAndGradient(
        dlogits: UnsafeMutablePointer<Float>,
        logits: UnsafePointer<Float>,
        targets: UnsafePointer<UInt16>,
        vocabSize: Int,
        seqLen: Int
    ) -> Float
}
```

Implementation rules:
- Transpose [V,S] → [S,V] with `vDSP_mtrans` for per-position softmax (contiguous rows).
- Numerically stable softmax: subtract max, exp, normalize.
- Loss uses `logf(prob + 1e-10)` for stability — match the C `1e-10f` exactly.
- Gradient: `(softmax - one_hot) / S` per position, then transpose back.
- `vvexpf` signature: `vvexpf(result, input, &count)` where count is `inout Int32`.
- Targets are `UInt16` — match C's `uint16_t`.

### 3c: AdamOptimizer.swift

```swift
public enum AdamOptimizer {
    /// In-place Adam update with bias correction. Mutates w, m, v.
    public static func update(
        weights: UnsafeMutablePointer<Float>,
        gradients: UnsafePointer<Float>,
        m: UnsafeMutablePointer<Float>,
        v: UnsafeMutablePointer<Float>,
        count: Int,
        timestep: Int,       // starts at 1, NOT 0
        lr: Float,
        beta1: Float,
        beta2: Float,
        eps: Float
    )
}
```

Implementation rules:
- `timestep` starts at 1. At t=0, `powf(0.9, 0)=1` so `bc1=0`, causing division by zero.
- Start with a scalar loop matching the C code exactly. Vectorize with vDSP only if tests pass first.
- Use `powf` (not `pow`) to match C's `powf` for single-precision.

### 3d: Embedding.swift

```swift
public enum Embedding {
    /// Channel-first lookup: output[d*seq + t] = embedding[tok*dim + d]
    public static func lookup(
        output: UnsafeMutablePointer<Float>,
        embedding: UnsafePointer<Float>,
        tokens: UnsafePointer<UInt16>,
        dim: Int,
        seqLen: Int
    )

    /// Channel-first backward: dEmbedding[tok*dim + d] += dx[d*seq + t]
    /// ACCUMULATES — does not zero dEmbedding first.
    public static func backward(
        dEmbedding: UnsafeMutablePointer<Float>,
        dx: UnsafePointer<Float>,
        tokens: UnsafePointer<UInt16>,
        dim: Int,
        seqLen: Int
    )
}
```

Implementation rules:
- Channel-first: `output[d * seqLen + t]`, NOT `output[t * dim + d]`.
- Tokens are `UInt16`.
- Backward ACCUMULATES — caller zeros dEmbedding if needed.

### 3e: RoPE.swift

```swift
public enum RoPE {
    /// Row-major layout: data[t * nHeads * headDim + h * headDim + i]
    /// Mutates q and k IN PLACE.
    public static func apply(
        q: UnsafeMutablePointer<Float>,
        k: UnsafeMutablePointer<Float>,
        seqLen: Int,
        nHeads: Int,
        headDim: Int
    )

    /// Transposed rotation backward. Mutates dq and dk IN PLACE.
    public static func backward(
        dq: UnsafeMutablePointer<Float>,
        dk: UnsafeMutablePointer<Float>,
        seqLen: Int,
        nHeads: Int,
        headDim: Int
    )
}
```

Implementation rules:
- Forward rotation: `(cos, -sin; sin, cos)` — standard complex multiplication.
- Backward rotation: `(cos, +sin; -sin, cos)` — TRANSPOSED rotation matrix (sign flip on sin terms).
- `freq = 1.0 / powf(10000.0, Float(i) / Float(headDim))` — use Float division.
- Process pairs `(i, i+1)` at stride 2. headDim must be even.

### 3f: SiLU.swift

```swift
public enum SiLU {
    /// silu(x) = x / (1 + exp(-x)) = x * sigmoid(x)
    @inlinable
    public static func forward(_ x: Float) -> Float

    /// silu'(x) = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
    @inlinable
    public static func backward(_ x: Float) -> Float
}
```

Implementation rules:
- These are scalar operations. Mark `@inlinable` for the hot loop in Phase 6 (applied across SEQ*HIDDEN = 524,288 elements per layer).
- Use `expf(-x)` (single-precision), matching C's `expf`.

## Step 4: Build and Test

```bash
swift build 2>&1 | grep -i "error\|warning"   # Zero errors, zero Accelerate deprecation warnings
swift test --filter CPUOpsTests                 # All 10 tests pass
```

Fix any failures before proceeding. If a test fails, read the error carefully, check the C source, and fix the implementation — do not weaken the test.

## Step 5: Run Full Test Suite

```bash
swift test    # All 84+ tests pass (Phases 1-4)
```

Verify no regressions in Phase 1-3 tests.

</instructions>

---

<verification_criteria>

| Test | Tolerance | What It Verifies |
|------|-----------|-----------------|
| rmsnorm forward known values | 1e-5 | Exact vDSP call sequence matches C |
| rmsnorm backward gradient check | 1e-3 relative | Gradient correctness via finite difference |
| cross-entropy uniform logits | 1e-4 | Softmax + log + mean loss computation |
| cross-entropy gradient sum | 1e-5 | Gradient normalization: sum(softmax - one_hot) = 0 |
| adam single step | 1e-6 | Bias correction formula correct |
| adam bias correction | convergence | Long-run m_hat, v_hat → true values |
| embedding lookup | exact (0.0) | Channel-first indexing correct |
| embedding backward accumulates | exact (0.0) | Accumulation semantics, not overwrite |
| rope forward/backward | 1e-5 | Orthogonal rotation: R^T @ R = I |
| silu forward/backward | 1e-3 relative | Derivative matches finite difference |

**Phase 4 is DONE when**:
1. All 10 tests green
2. Gradient checks pass on all random configurations
3. Zero Accelerate deprecation warnings in build output
4. Full test suite (Phases 1-4) passes with 0 failures
5. No mutable global state in any implementation

</verification_criteria>

---

<critical_warnings>

1. **vDSP_vsub argument order is REVERSED**: `vDSP_vsub(B, 1, A, 1, C, 1, n)` computes `C = A - B`, NOT `C = B - A`. The C code does `vDSP_vsub(g_rms_tmp, 1, dy+i*S, 1, g_rms_tmp, 1, S)` which computes `tmp = dy - tmp`. Getting this wrong silently produces garbage gradients that look plausible but are numerically wrong.

2. **rmsnorm_bwd and embed_backward ACCUMULATE**: `dw[i] += s` and `d_embed[tok*dim + d] += dx[...]`. They ADD to existing values, they do not overwrite. The caller is responsible for zeroing if fresh gradients are needed. Test this explicitly.

3. **vvrsqrtf count is Int32**: `var n = Int32(seqLen); vvrsqrtf(result, input, &n)`. Using `Int` causes a type error. Same for `vvexpf`.

4. **Channel-first vs row-major**: RMSNorm, CrossEntropy, Embedding use `[DIM, SEQ]` where `data[dim * SEQ + seq]`. RoPE uses `[SEQ, nHeads*headDim]` where `data[seq * stride + offset]`. Mixing these produces silent corruption.

5. **RoPE backward sign flip**: Forward: `q_new = q0*cos - q1*sin`. Backward: `dq_new = dq0*cos + dq1*sin` (PLUS, not minus). This is the transposed rotation matrix. Getting the sign wrong silently corrupts gradients.

6. **Adam timestep starts at 1**: At t=0, bc1 = 1 - pow(0.9, 0) = 0, causing NaN from division by zero. Callers must pass t >= 1.

7. **cross_entropy targets are UInt16**: Matching C's `uint16_t`. Using `Int` silently reinterprets memory.

8. **Do not reorder vDSP operations**: The C code's operation sequence determines the specific floating-point rounding behavior. Reordering "equivalent" operations may change results by up to 1e-4, failing the numerical equivalence tests.

</critical_warnings>

---

<agentic_guidance>

Keep working until all 10 tests pass and the full suite (Phases 1-4) shows 0 failures. If a test fails:
1. Read the failure message carefully — it usually points to the exact issue.
2. Compare your Swift vDSP call sequence against the C source line by line.
3. Check argument ordering, especially for `vDSP_vsub`.
4. Verify you're using the correct data layout (channel-first vs row-major).
5. Fix the implementation, not the test — the tests encode the C code's exact behavior.

Plan your approach before each operation: read the C source, trace the data flow, write the Swift equivalent, then run the specific test. After each file passes, run the full suite to catch regressions.

If you are unsure about a vDSP function signature, check the Accelerate framework documentation. Use `import Accelerate` which provides vDSP, vecLib (vvrsqrtf, vvexpf), and cblas.

</agentic_guidance>
