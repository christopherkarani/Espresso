# 10x Inference Optimization Research

## Baseline Benchmark (Phase 0)

**Hardware:** Apple M3 Max, ANE Peak 18.0 TFLOPS
**Config:** dim=768, hidden=2048, seq=256, heads=12, layers=6
**FLOPs per forward pass:** 22.99 GFLOPs

### Training Forward Pass (30 kernels)
| Metric | Value |
|--------|-------|
| Mean | 8.318 ms |
| Median | 8.276 ms |
| P95 | 8.514 ms |
| Min | 6.556 ms |
| ANE kernel | 6.828 ms (83.0%) |
| Surface I/O | 1.190 ms (14.5%) |
| CPU element | 0.211 ms (2.6%) |
| ANE Utilization | 15.4% |
| Throughput | 121 fwd/sec |

### Inference Forward Pass (12 kernels, fused residuals)
| Metric | Value |
|--------|-------|
| Mean | 5.687 ms |
| Median | 4.595 ms |
| P95 | 10.692 ms |
| Min | 4.243 ms |
| ANE kernel | 5.343 ms (94.0%) |
| Surface I/O | 0.342 ms (6.0%) |
| CPU element | 0.000 ms (0.0%) |
| ANE Utilization | 27.8% |
| Throughput | 218 fwd/sec |

### Per-Layer Breakdown (Inference)
- **Per-layer total:** ~0.766 ms (median 4.595ms / 6 layers)
- **Per-layer ANE kernel:** ~0.891 ms (5.343ms / 6)
- **Per-layer Surface I/O:** ~0.057 ms (0.342ms / 6)

### Target
- Current per-layer: ~0.766 ms
- Target per-layer: ~0.077 ms (10x improvement)
- Hardware floor estimate: 22.99 GFLOPs / 18 TFLOPS = 1.28 ms total = 0.213 ms/layer at 100% utilization

**Note:** At 27.8% ANE utilization, there's 3.6x headroom from utilization alone. Combined with kernel fusion and I/O elimination, 10x is ambitious but directionally plausible.

---

## Research Task #2: Kernel Fusion Feasibility (COMPLETED)

**Finding:** Single-layer SDPA+FFN fusion is feasible. Combined MIL program has ~55 ops, 10 weight blobs, ~13.63MB weights. This eliminates inter-kernel eval() overhead and one full surface I/O round-trip per layer.

**Expected impact:** Eliminates 6 eval() calls (one per layer) and 6 surface read/write cycles. Could save ~0.057ms I/O + significant eval dispatch overhead per layer.

**Recommendation:** Start with single-layer fusion before attempting whole-model fusion.

---

## Research Task #3: MIL Operation Optimization & Tile Alignment (COMPLETED)

### Key Findings

1. **Split-Head Attention (HIGHEST PRIORITY):** Apple's recommended pattern — split 12-head attention into 12 independent single-head paths. Eliminates most reshape/transpose ops, improves L2 cache utilization (32KB per head vs 384KB combined), enables multi-core scheduling across ANE's 16 cores. Apple reports up to 10x improvement with this on DistilBERT.

2. **RMSNorm simplification (quick win):** Replace `reduce_sum + mul(1/dim)` with `reduce_mean`, and `pow(-0.5)` with `rsqrt`. Saves 4 ops per layer.

3. **SiLU fusion (quick win):** Replace `sigmoid + mul` with built-in `silu` MIL op. Saves 1 op per layer.

4. **Conv is already optimal** for linear projections — ANE is fundamentally a convolution engine.

5. **Reshape/transpose are NOT free** — involve actual data movement due to 64-byte alignment constraints. Current SDPA uses 10 reshape/transpose ops, potentially 30-60% of per-layer latency.

6. **All dimensions well-aligned:** dim=768, hidden=2048, headDim=64, seqLen=256 are all multiples of 64 (fp16 64-byte alignment). No padding needed.

7. **`[1,C,1,S]` layout is correct** — matches ANE native IOSurface format.

8. **Working set fits in SRAM** (~15MB total, ANE has 32MB SRAM). Bottleneck is compute/dispatch overhead.

### Priority Implementation Order
1. Split-head attention (highest impact, medium difficulty)
2. Kernel fusion: SDPA+FFN per layer (high impact, medium difficulty)
3. RMSNorm simplification (low impact, trivial difficulty)
4. SiLU fusion (low impact, trivial difficulty)

---

## Research Task #4: ANE Private API Capabilities (COMPLETED)

### Key Findings

1. **No async eval API exists.** `_ANEInMemoryModel.evaluateWithQoS:` is synchronous and blocking.
2. **`_ANEChainingRequest`** is an undiscovered class that may enable chaining compiled models in a single dispatch — potential 15-25% improvement. HIGH RISK / HIGH REWARD.
3. **ANE compiler auto-caches** E5 binaries on disk. First compile ~20-40ms, cache hits near-instant.
4. **`CompileGate.lock` is necessary** for training but the ~119 compile limit is irrelevant for inference (only 8-12 compiles needed).
5. **Weight swapping is impossible** — weights are baked into E5 binaries at compile time.
6. **GCD surface prep pipelining** between layers is feasible (5-10% potential).
7. **Overall: ANE API is not the bottleneck.** Maximum realistic API-level gain is 20-30%. The 10x path is through MIL-level optimization.

---

## Research Task #5: Surface I/O Elimination (COMPLETED)

### Key Findings

1. **Zero-copy chaining not possible** with current ANE request binding model. IOSurfaces are permanently bound to `_ANERequest` at compile time.
2. **Lock overhead is negligible** — under 2 microseconds per lock/unlock cycle. NEON conversion dominates.
3. **Double buffering has no benefit** — eval() is synchronous, can't overlap I/O with computation.
4. **Kernel fusion makes most I/O optimization moot** — fusing SDPA+FFN eliminates inter-kernel handoff. Full 6-layer fusion reduces I/O to 2 operations total (0.028 ms).
5. **Surface I/O is only 6% of inference time** — the 10x goal requires ANE kernel time reduction, not I/O optimization.
6. **Surface binding** (compile FFN with attention's output surface as input) is the only I/O optimization worth pursuing if fusion doesn't happen.

---

## Research Synthesis: Implementation Priority

All research converges on the same conclusion. Ranked by expected impact:

| Priority | Optimization | Expected Impact | Difficulty |
|----------|-------------|----------------|------------|
| **P0** | Split-head attention (12 independent paths) | **Up to 10x** (Apple's claim) | Medium |
| **P1** | Kernel fusion: SDPA+FFN per layer | **2-3x** (halves eval calls + eliminates I/O) | Medium |
| **P2** | RMSNorm: reduce_mean + rsqrt | 5-10% | Trivial |
| **P3** | SiLU fusion | 2-5% | Trivial |
| **P4** | Explore _ANEChainingRequest | 15-25% (if it works) | High risk |
| **P5** | GCD surface prep pipelining | 5-10% | Low |

**Implementation plan:** P0 (split-head attention) first, then P1 (kernel fusion) on top of it. Apply P2/P3 quick wins during fusion implementation. Measure after each step.
