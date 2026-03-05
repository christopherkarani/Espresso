# ANE Inference Optimization — Phase 2 Investigation Prompt

## Context

You are investigating the remaining ~3x performance headroom in Apple Neural Engine (ANE) inference for a 6-layer transformer. Here is the current state:

### Hardware
- **Apple M3 Max**, ANE peak 18.0 TFLOPS (fp16)
- ANE: 16 cores, 32MB SRAM
- IOSurface-based CPU↔ANE data transfer

### Model Config
- `dim=768, hidden=2048, heads=12, headDim=64, seqLen=256, layers=6`
- FLOPs per forward pass: 22.99 GFLOPs
- Channel-first layout: `[1, C, 1, S]` (ANE IOSurface native format)
- All dimensions are 64-byte aligned (multiples of 64 for fp16)

### Current Performance (Fused Inference — best so far)
| Metric | Value |
|--------|-------|
| Kernels | 6 (1 per layer, SDPA+FFN fused) |
| Median latency | 3.871 ms (full 6-layer pass) |
| Per-layer | 0.645 ms |
| ANE utilization | ~33% |
| Speedup vs training | 2.97x |

### Hardware Floor
- Theoretical minimum: 22.99 GFLOPs / 18 TFLOPS = **1.28 ms total = 0.213 ms/layer** at 100% utilization
- Current per-layer (0.645 ms) is **3.0x above** the hardware floor
- **Your goal: close this 3x gap as much as possible**

### What's Already Been Done (Do NOT Re-investigate)
1. Split-head attention (12 independent single-head paths via `slice_by_index`)
2. Kernel fusion (SDPA + FFN in single MIL program per layer)
3. RMSNorm simplification (`reduce_sum + mul(invd)` instead of `reduce_mean`)
4. SiLU fusion (built-in `silu` MIL op)
5. Surface I/O optimization (pre-resolved handles, single read/write per layer)
6. Confirmed: ANE API is NOT the bottleneck (eval is synchronous, no async API exists)
7. Confirmed: weight swapping impossible (baked into E5 binaries at compile time)
8. Confirmed: double buffering has no benefit (eval is synchronous)

### Architecture Overview
```
ANEInterop (ObjC/C — private API bridge)
    ├── ANETypes (value types, tensor descriptors)
    │       ├── MILGenerator (MIL program text generation)
    │       │       └── ANERuntime (compile/eval, IOSurface I/O)
    │       │               └── Espresso (transformer layers, forward pass)
```

Key files:
- `Sources/MILGenerator/FusedLayerInferenceGenerator.swift` — current fused MIL generator
- `Sources/ANERuntime/FusedInferenceKernel.swift` — ~Copyable kernel wrapper
- `Sources/Espresso/ForwardPass.swift` — forward pass orchestration
- `Sources/EspressoBench/ANEDirectBench.swift` — benchmark harness

### MIL Dialect Constraints
- Version: `program(1.3)`, function: `func main<ios18>(...)`
- `reduce_mean` and `rsqrt` are **NOT valid** in raw MIL text — use `reduce_sum + mul(invd)` and `pow(x, -0.5)`
- `slice_by_index` requires explicit `begin_mask`, `end_mask`, `squeeze_mask` tensor params
- Conv2D is optimal for linear projections (ANE is fundamentally a convolution engine)
- Reshape/transpose ops are **NOT free** — they involve actual data movement due to 64-byte alignment

---

## Investigation Mandate

Run **three diagnostic experiments** to isolate where the remaining 3x overhead lives, then propose targeted optimizations based on findings.

### Experiment 1: "Convs-Only" Baseline Kernel

**Purpose:** Measure the pure compute cost of the 7 large convolutions per layer, stripped of ALL attention logic (reshapes, transposes, slicing, matmuls, masks, softmax). This establishes the theoretical minimum compute time for the weight-multiply portion.

**What to build:**
1. Create a new MIL generator (`DiagnosticConvsOnlyGenerator`) that produces a single MIL program containing ONLY:
   - Input: `[1, 768, 1, 256]` fp16
   - Conv Wq: `[768, 768, 1, 1]` → output `[1, 768, 1, 256]`
   - Conv Wk: `[768, 768, 1, 1]` → output `[1, 768, 1, 256]`
   - Conv Wv: `[768, 768, 1, 1]` → output `[1, 768, 1, 256]`
   - Conv Wo: `[768, 768, 1, 1]` → output `[1, 768, 1, 256]`
   - Conv W1: `[2048, 768, 1, 1]` → output `[1, 2048, 1, 256]`
   - Conv W3: `[2048, 768, 1, 1]` → output `[1, 2048, 1, 256]`
   - Conv W2: `[768, 2048, 1, 1]` → output `[1, 768, 1, 256]`
   - Chain them sequentially (output of each feeds into next where dimensions match, or use the input for all and add results)
   - No RMSNorm, no attention score computation, no softmax, no masks, no residuals
2. Compile and benchmark with 500 iterations, 50 warmup
3. **Key metric:** What fraction of the 0.645 ms/layer is pure convolution compute?

**Expected insight:** If convs alone take 0.5+ ms, the bottleneck is compute-bound and we need to reduce FLOPs (e.g., weight quantization, pruning). If convs take < 0.3 ms, the bottleneck is the attention logic overhead (reshapes, transposes, softmax, masks).

### Experiment 2: "Batched-Head" vs "Split-Head" Comparison

**Purpose:** Our split-head approach uses 12 independent `slice_by_index` → single-head matmul paths. The original approach used reshape → transpose → batched matmul. In the *fused* context (single kernel), the tradeoff may be different than in separate kernels. Test both.

**What to build:**
1. Create a variant MIL generator (`DiagnosticBatchedHeadGenerator`) that uses the original reshape→transpose→batched matmul pattern for attention BUT within the fused SDPA+FFN single-kernel structure
2. The attention portion should:
   - Reshape Q,K,V from `[1, 768, 1, 256]` to `[1, 64, 12, 256]` (headDim, nHeads, seqLen)
   - Transpose to `[1, 12, 64, 256]` or whatever layout the batched matmul needs
   - Single batched matmul for Q·K^T across all heads
   - Single batched softmax + mask
   - Single batched matmul for scores·V
   - Reshape back to `[1, 768, 1, 256]`
3. Keep everything else identical to current fused generator (RMSNorm, FFN, residuals)
4. Benchmark both side-by-side: 500 iterations, 50 warmup
5. **Key metric:** Which is faster in the fused context? By how much?

**Expected insight:** Split-head eliminates reshapes but uses 12x more ops. In a single kernel, the ANE compiler may be able to optimize the batched approach better than we can manually split heads. Apple's 10x claim for split-head was for DistilBERT with different dimensions — may not hold for our config.

### Experiment 3: "Whole-Model" Single Kernel

**Purpose:** Test whether fusing ALL 6 layers into a single MIL program eliminates the remaining 5 inter-layer eval() dispatches and surface I/O round-trips.

**What to build:**
1. Create a generator (`DiagnosticWholeModelGenerator`) that produces ONE MIL program containing all 6 transformer layers chained
   - Input: `[1, 768, 1, 256]`
   - Layer 1: RMSNorm1 → SDPA → Wo → residual → RMSNorm2 → FFN → residual
   - Layer 2: same, fed from Layer 1 output
   - ... through Layer 6
   - Output: `[1, 768, 1, 256]`
   - Total weight blobs: 60 (10 per layer × 6 layers)
2. This will be a LARGE MIL program (~1000+ ops). Monitor:
   - Compilation time (may be very long)
   - Whether ANE compiler accepts it at all (there may be size limits)
   - Whether it actually runs faster or slower
3. Benchmark: 500 iterations, 50 warmup
4. **Key metrics:** Total 6-layer latency vs 6 × single-layer latency. Compilation time.

**Expected insight:** If whole-model is faster, the remaining overhead is dispatch/I/O between layers. If it's the same or slower, the ANE compiler may not optimize large graphs well, confirming that per-layer is the right granularity.

---

## Analysis Framework

After running all 3 experiments, produce a breakdown:

```
Current per-layer budget: 0.645 ms
├── Pure convolution compute: ??? ms (from Experiment 1)
├── Attention logic overhead: ??? ms (current - convs)
│   ├── Reshape/transpose ops: ??? ms
│   ├── Score matmuls (Q·K^T, scores·V): ??? ms
│   ├── Softmax + masking: ??? ms
│   └── slice_by_index (12 heads): ??? ms
├── RMSNorm overhead: ??? ms
├── Residual connections: ??? ms
└── Dispatch overhead per layer: ??? ms (from Experiment 3)

Hardware floor: 0.213 ms/layer
Gap: ??? ms (current - floor)
Addressable gap: ??? ms (what we can realistically optimize)
```

### Decision Tree Based on Results

**If convs dominate (> 60% of per-layer time):**
- Investigate INT8 quantization of weight convolutions (ANE supports INT8 natively)
- Investigate whether smaller `hidden` dim (1536 instead of 2048) can be compensated with more layers
- Check if ANE's conv implementation has a sweet spot for certain kernel sizes

**If attention logic dominates (> 40% of per-layer time):**
- Investigate whether the batched-head approach wins (Experiment 2 results)
- Look for ways to reduce reshape/transpose ops further
- Consider whether attention score computation can be approximated (linear attention variants)
- Test if pre-computing Q·K^T mask as a constant tensor eliminates runtime masking cost

**If dispatch overhead is significant (whole-model notably faster):**
- Implement 2-layer or 3-layer fusion as a middle ground
- Investigate `_ANEChainingRequest` for chaining compiled models without full fusion
- Look at whether GCD can overlap surface prep for layer N+1 with eval of layer N

**If all experiments show ~0.2 ms/layer:**
- We're at the hardware floor. Document the achievement and move on.
- The only further path would be reducing FLOPs (model architecture changes, not runtime optimization).

---

## Implementation Guidelines

### MIL Generator Pattern
Follow the existing `FusedLayerInferenceGenerator` structure:
```swift
struct DiagnosticConvsOnlyGenerator: MILProgramGenerator {
    let weights: [WeightBlob]
    var milText: String { /* generate MIL program text */ }
}
```

### Benchmarking Pattern
Use the existing `ANEDirectBench` infrastructure. Add new static methods:
```swift
static func runDiagnosticConvsOnly(warmup: Int, iterations: Int) throws -> Result
static func runDiagnosticBatchedHead(warmup: Int, iterations: Int, nLayers: Int) throws -> Result
static func runDiagnosticWholeModel(warmup: Int, iterations: Int) throws -> Result
```

### Key Constraints
- All MIL programs must use `program(1.3)` and `func main<ios18>(...)`
- Weight blobs use `@parameter` attributes in MIL function signatures
- Use `conv(x, weight, ...)` for all linear projections
- Remember: `reduce_mean` → `reduce_sum + mul(invd)`, `rsqrt` → `pow(x, -0.5)`
- `slice_by_index` needs `begin_mask`, `end_mask`, `squeeze_mask` params
- IOSurface format is `[1, C, 1, S]` with fp16
- Use `~Copyable` patterns for kernel wrappers (see `FusedInferenceKernel`)
- Benchmark with `ContinuousClock` and `StepTimingBreakdown` for consistent metrics

### Output
1. Three benchmark results with full statistics (mean, median, P95, min)
2. The per-layer budget breakdown shown above
3. A concrete recommendation for the highest-impact next optimization
4. Updated `docs/10x-optimization-research.md` with findings

---

## Working Directory

All work must happen in the git worktree at:
```
/Users/chriskarani/CodingProjects/Espresso/.claude/worktrees/10x-inference-optimization
```

Branch: `checkpoint/phase8-ane-inference-20260305`

Do NOT cd to the parent repository. Run all commands from the worktree root.
