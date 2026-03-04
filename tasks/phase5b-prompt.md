# Phase 5b: ANERuntime — LayerKernelSet, StaticKernel, ModelWeightLoader

<role>
You are a senior Swift 6.2 systems engineer specializing in Apple Neural Engine (ANE) kernel compilation and low-level memory management. You are completing Phase 5 of a bottom-up Swift rewrite of an ANE training codebase (~6,100 lines Obj-C/C). Phase 5a (ANEKernel, ANEError, CompileBudget) is already implemented and passing all tests. Your task is Phase 5b: implement the three remaining files in the `ANERuntime` target.
</role>

---

<context>

## Project State

**Completed phases:**
- Phase 1: ANEInterop (C/ObjC shim for private ANE APIs)
- Phase 2: ANETypes (ModelConfig, TensorBuffer, LayerStorage, LayerWeights, WeightBlob, SurfaceIO, etc.)
- Phase 3: MILGenerator (6 MIL program generators + GenericMIL + CausalMask)
- Phase 4: CPUOps (RMSNorm, CrossEntropy, AdamOptimizer, Embedding, RoPE, SiLU)
- Phase 5a: ANERuntime core (ANEKernel, ANEError, CompileBudget)

**Test status:** 99 tests executed, 12 skipped (ANE hardware), 0 failures.

**Target:** `ANERuntime` (depends on `ANEInterop`, `ANETypes`, `MILGenerator`)

---

## Files to Implement

| File | Description | Maps to C source |
|------|-------------|------------------|
| `Sources/ANERuntime/LayerKernelSet.swift` | Owns 5 weight-bearing ANEKernel instances per layer; compiles from LayerWeights | `train_large.m:59-94` (`compile_layer_kernels`) |
| `Sources/ANERuntime/StaticKernel.swift` | Owns 1 weight-free sdpaBwd2 ANEKernel; compiled once, different lifecycle from LayerKernelSet | `train_large.m:97-100` (`compile_sdpa_bwd2`) |
| `Sources/ANERuntime/ModelWeightLoader.swift` | Loads pretrained weights from llama2.c `.bin` format into LayerWeights + global buffers | `train_large.m:13-55` (`load_pretrained`) |

---

## C Source Being Ported (Verbatim)

### train_large.m:59-107 — Kernel Compilation & Lifecycle

```c
// ===== Compile one layer's kernels =====
static bool compile_layer_kernels(LayerKernels *lk, LayerWeights *w) {
    lk->fwdAttn = compile_kern_mil_w(gen_sdpa_fwd_taps(), (@{
        @"@model_path/weights/rms1.bin": @{@"offset":@0, @"data":build_blob(w->rms_att,1,DIM)},
        @"@model_path/weights/wq.bin": @{@"offset":@0, @"data":build_blob(w->Wq,DIM,DIM)},
        @"@model_path/weights/wk.bin": @{@"offset":@0, @"data":build_blob(w->Wk,DIM,DIM)},
        @"@model_path/weights/wv.bin": @{@"offset":@0, @"data":build_blob(w->Wv,DIM,DIM)},
        @"@model_path/weights/wo.bin": @{@"offset":@0, @"data":build_blob(w->Wo,DIM,DIM)},
        @"@model_path/weights/mask.bin": @{@"offset":@0, @"data":get_mask_blob()},
    }), DIM*SEQ*2, 6*DIM*SEQ*2);

    lk->fwdFFN = compile_kern_mil_w(gen_ffn_fwd_taps(), (@{
        @"@model_path/weights/rms2.bin": @{@"offset":@0, @"data":build_blob(w->rms_ffn,1,DIM)},
        @"@model_path/weights/w1.bin": @{@"offset":@0, @"data":build_blob(w->W1,HIDDEN,DIM)},
        @"@model_path/weights/w3.bin": @{@"offset":@0, @"data":build_blob(w->W3,HIDDEN,DIM)},
        @"@model_path/weights/w2.bin": @{@"offset":@0, @"data":build_blob(w->W2,DIM,HIDDEN)},
    }), DIM*SEQ*2, (2*DIM+3*HIDDEN)*SEQ*2);

    lk->ffnBwd = compile_kern_mil_w(gen_ffn_bwd(), (@{
        @"@model_path/weights/w2t.bin": @{@"offset":@0, @"data":build_blob_t(w->W2,DIM,HIDDEN)},
        @"@model_path/weights/w1t.bin": @{@"offset":@0, @"data":build_blob_t(w->W1,HIDDEN,DIM)},
        @"@model_path/weights/w3t.bin": @{@"offset":@0, @"data":build_blob_t(w->W3,HIDDEN,DIM)},
    }), (DIM+2*HIDDEN)*SEQ*2, (DIM+2*HIDDEN)*SEQ*2);

    lk->sdpaBwd1 = compile_kern_mil_w(gen_sdpa_bwd1(), (@{
        @"@model_path/weights/mask.bin": @{@"offset":@0, @"data":get_mask_blob()},
        @"@model_path/weights/wot.bin": @{@"offset":@0, @"data":build_blob_t(w->Wo,DIM,DIM)},
    }), 4*DIM*SEQ*2, (DIM+2*SCORE_CH)*SEQ*2);

    lk->qkvBwd = compile_kern_mil_w(gen_qkvb(), (@{
        @"@model_path/weights/wqt.bin": @{@"offset":@0, @"data":build_blob_t(w->Wq,DIM,DIM)},
        @"@model_path/weights/wkt.bin": @{@"offset":@0, @"data":build_blob_t(w->Wk,DIM,DIM)},
        @"@model_path/weights/wvt.bin": @{@"offset":@0, @"data":build_blob_t(w->Wv,DIM,DIM)},
    }), 3*DIM*SEQ*2, DIM*SEQ*2);

    return lk->fwdAttn && lk->fwdFFN && lk->ffnBwd && lk->sdpaBwd1 && lk->qkvBwd;
}

// Compile weight-free sdpaBwd2 (only needs once, no weights)
static Kern *compile_sdpa_bwd2(void) {
    return compile_kern_mil_w(gen_sdpa_bwd2(), @{},
        (2*SCORE_CH+2*DIM)*SEQ*2, 2*DIM*SEQ*2);
}

static void free_layer_kernels(LayerKernels *lk) {
    free_kern(lk->fwdAttn); free_kern(lk->fwdFFN); free_kern(lk->ffnBwd);
    free_kern(lk->sdpaBwd1); free_kern(lk->qkvBwd);
    // sdpaBwd2 is shared, freed separately
    lk->fwdAttn = lk->fwdFFN = lk->ffnBwd = lk->sdpaBwd1 = lk->qkvBwd = NULL;
}
```

### train_large.m:13-55 — Weight Loading (llama2.c format)

```c
static bool load_pretrained(LayerWeights *lw, float *rms_final, float *embed, const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { printf("Cannot open %s\n", path); return false; }
    Llama2Config cfg;
    fread(&cfg, sizeof(cfg), 1, f);
    printf("  Model config: dim=%d hidden=%d layers=%d heads=%d vocab=%d seq=%d\n",
           cfg.dim, cfg.hidden_dim, cfg.n_layers, cfg.n_heads, abs(cfg.vocab_size), cfg.seq_len);
    if (cfg.dim != DIM || cfg.hidden_dim != HIDDEN || cfg.n_layers != NLAYERS) {
        printf("  ERROR: Config mismatch! Expected dim=%d hidden=%d layers=%d\n", DIM, HIDDEN, NLAYERS);
        fclose(f); return false;
    }
    int V = abs(cfg.vocab_size);
    bool shared = cfg.vocab_size > 0;

    // Read in llama2.c order: embed, rms_att[all], wq[all], wk[all], wv[all], wo[all],
    //                         rms_ffn[all], w1[all], w2[all], w3[all], rms_final, [wcls]
    fread(embed, 4, V * DIM, f);

    // rms_att weights for all layers (contiguous)
    for (int L = 0; L < NLAYERS; L++) fread(lw[L].rms_att, 4, DIM, f);
    // wq for all layers
    for (int L = 0; L < NLAYERS; L++) fread(lw[L].Wq, 4, WQ_SZ, f);
    // wk for all layers
    for (int L = 0; L < NLAYERS; L++) fread(lw[L].Wk, 4, WQ_SZ, f);
    // wv for all layers
    for (int L = 0; L < NLAYERS; L++) fread(lw[L].Wv, 4, WQ_SZ, f);
    // wo for all layers
    for (int L = 0; L < NLAYERS; L++) fread(lw[L].Wo, 4, WO_SZ, f);
    // rms_ffn weights for all layers
    for (int L = 0; L < NLAYERS; L++) fread(lw[L].rms_ffn, 4, DIM, f);
    // w1 for all layers
    for (int L = 0; L < NLAYERS; L++) fread(lw[L].W1, 4, W1_SZ, f);
    // w2 for all layers
    for (int L = 0; L < NLAYERS; L++) fread(lw[L].W2, 4, W2_SZ, f);
    // w3 for all layers
    for (int L = 0; L < NLAYERS; L++) fread(lw[L].W3, 4, W3_SZ, f);
    // rms_final
    fread(rms_final, 4, DIM, f);
    // wcls = embed if shared (we just use embed pointer)

    fclose(f);
    printf("  Loaded pretrained weights (%s)\n", shared ? "shared embed/cls" : "separate cls");
    return true;
}
```

---

## Available Swift APIs (Already Implemented)

### ANEKernel (Phase 5a — Sources/ANERuntime/ANEKernel.swift)

```swift
public struct ANEKernel: ~Copyable {
    /// Compile a MIL program with optional weight blobs into an ANE kernel.
    public init(
        milText: String,
        weights: [(path: String, data: Data)],
        inputSizes: [Int],
        outputSizes: [Int],
        checkBudget: Bool = true
    ) throws(ANEError)

    /// Convenience: single-input, single-output kernel.
    public init(
        milText: String,
        weights: [(path: String, data: Data)],
        inputBytes: Int,
        outputBytes: Int,
        checkBudget: Bool = true
    ) throws(ANEError)

    public func eval() throws(ANEError)
    public func inputSurface(at index: Int) throws(ANEError) -> IOSurfaceRef
    public func outputSurface(at index: Int) throws(ANEError) -> IOSurfaceRef
    deinit  // calls ane_interop_free(handle)
}
```

### ANEError (Phase 5a — Sources/ANERuntime/ANEError.swift)

```swift
public enum ANEError: Error, Sendable, CustomStringConvertible, LocalizedError {
    case invalidArguments(String)
    case compilationFailed
    case evaluationFailed
    case compileBudgetExhausted
    case surfaceAllocationFailed
    case invalidSurfaceIndex(Int)
    case inputSurfaceUnavailable(Int)
    case outputSurfaceUnavailable(Int)
}
```

### CompileBudget (Phase 5a — Sources/ANERuntime/CompileBudget.swift)

```swift
public enum CompileBudget {
    public static let maxCompiles: Int = ModelConfig.maxCompiles  // 100
    public static var currentCount: Int { Int(ane_interop_compile_count()) }
    public static var isExhausted: Bool { currentCount >= maxCompiles }
    public static func setCount(_ value: Int) throws(ANEError)
    public static var remaining: Int { max(0, maxCompiles - currentCount) }
}
```

### ModelConfig (Phase 2 — Sources/ANETypes/ModelConfig.swift)

```swift
public enum ModelConfig {
    public static let dim = 768
    public static let hidden = 2048
    public static let heads = 12
    public static let seqLen = 256
    public static let nLayers = 12
    public static let vocab = 32_000
    public static let accumSteps = 10
    public static let maxCompiles = 100
    public static let kernelsPerLayer = 5
    public static let totalWeightKernels = kernelsPerLayer * nLayers  // 60
    public static let headDim = dim / heads   // 64
    public static let scoreCh = heads * seqLen  // 3072
    public static let wqSize = dim * dim        // 589,824
    public static let woSize = dim * dim        // 589,824
    public static let w1Size = hidden * dim     // 1,572,864
    public static let w2Size = dim * hidden     // 1,572,864
    public static let w3Size = hidden * dim     // 1,572,864
}
```

### TensorBuffer (Phase 2 — Sources/ANETypes/TensorBuffer.swift)

```swift
public struct TensorBuffer: ~Copyable {
    public static let allocationAlignment: Int = 64
    public let count: Int
    public init(count: Int, zeroed: Bool)
    deinit  // deallocates
    public func withUnsafeMutablePointer<R>(_ body: (UnsafeMutablePointer<Float>) throws -> R) rethrows -> R
    public func withUnsafePointer<R>(_ body: (UnsafePointer<Float>) throws -> R) rethrows -> R
    public func withUnsafeMutableBufferPointer<R>(_ body: (UnsafeMutableBufferPointer<Float>) throws -> R) rethrows -> R
    public func withUnsafeBufferPointer<R>(_ body: (UnsafeBufferPointer<Float>) throws -> R) rethrows -> R
    public func zero()
}
```

### LayerWeights (Phase 2 — Sources/ANETypes/LayerWeights.swift)

```swift
public struct LayerWeights: ~Copyable {
    public let Wq: TensorBuffer   // count: wqSize (589,824)
    public let Wk: TensorBuffer   // count: wqSize (589,824)
    public let Wv: TensorBuffer   // count: wqSize (589,824)
    public let Wo: TensorBuffer   // count: woSize (589,824)
    public let W1: TensorBuffer   // count: w1Size (1,572,864)
    public let W2: TensorBuffer   // count: w2Size (1,572,864)
    public let W3: TensorBuffer   // count: w3Size (1,572,864)
    public let rmsAtt: TensorBuffer  // count: dim (768)
    public let rmsFfn: TensorBuffer  // count: dim (768)
    public init()  // allocates all buffers (uninitialized)
}
```

### WeightBlob (Phase 2 — Sources/ANETypes/WeightBlob.swift)

```swift
public enum WeightBlob {
    /// Build blob with 128-byte header + FP16 payload. Normal (row-major) layout.
    public static func build(from weights: UnsafeBufferPointer<Float>, rows: Int, cols: Int) -> Data
    public static func build(from weights: [Float], rows: Int, cols: Int) -> Data

    /// Build blob with transposed layout (C build_blob_t equivalent).
    public static func buildTransposed(from weights: UnsafeBufferPointer<Float>, rows: Int, cols: Int) -> Data
    public static func buildTransposed(from weights: [Float], rows: Int, cols: Int) -> Data

    /// Build blob from pre-converted FP16 data.
    public static func buildFP16(from weights: UnsafeBufferPointer<UInt16>) -> Data
    public static func buildFP16(from weights: [UInt16]) -> Data
}
```

### MIL Program Generators (Phase 3 — Sources/MILGenerator/)

All conform to `MILProgramGenerator` protocol:

```swift
public protocol MILProgramGenerator: Sendable {
    var milText: String { get }
    var inputBytes: Int { get }
    var outputByteSizes: [Int] { get }
}

// Extension provides:
extension MILProgramGenerator {
    public var outputBytes: Int  // outputByteSizes[0] (precondition: single output)
}
```

**The 6 generators:**

| Generator | MIL Function | Input Bytes | Output Bytes | Weights |
|-----------|-------------|-------------|-------------|---------|
| `SDPAForwardGenerator()` | `gen_sdpa_fwd_taps` | `DIM*SEQ*2` | `6*DIM*SEQ*2` | rms1, wq, wk, wv, wo, mask |
| `FFNForwardGenerator()` | `gen_ffn_fwd_taps` | `DIM*SEQ*2` | `(2*DIM+3*HIDDEN)*SEQ*2` | rms2, w1, w3, w2 |
| `FFNBackwardGenerator()` | `gen_ffn_bwd` | `(DIM+2*HIDDEN)*SEQ*2` | `(DIM+2*HIDDEN)*SEQ*2` | w2t, w1t, w3t |
| `SDPABackward1Generator()` | `gen_sdpa_bwd1` | `4*DIM*SEQ*2` | `(DIM+2*SCORE_CH)*SEQ*2` | mask, wot |
| `SDPABackward2Generator()` | `gen_sdpa_bwd2` | `(2*SCORE_CH+2*DIM)*SEQ*2` | `2*DIM*SEQ*2` | **NONE** |
| `QKVBackwardGenerator()` | `gen_qkvb` | `3*DIM*SEQ*2` | `DIM*SEQ*2` | wqt, wkt, wvt |

### CausalMask (Phase 3 — Sources/MILGenerator/CausalMask.swift)

```swift
public enum CausalMask {
    /// Returns cached weight blob for causal attention mask. Thread-safe.
    public static func blob(seqLen: Int) -> Data
}
```

### LayerStorage (Phase 2 — Sources/ANETypes/LayerStorage.swift)

```swift
/// Fixed-size container for ~Copyable elements with coroutine accessors.
public struct LayerStorage<Element: ~Copyable>: ~Copyable {
    public let count: Int
    public init(count: Int, initializer: (Int) -> Element)
    public subscript(index: Int) -> Element {
        _read { ... }
        _modify { ... }
    }
    deinit  // deinitializes + deallocates
}
```

---

## Exact Kernel I/O Byte Sizes (Pre-Computed)

All computed from ModelConfig constants. `MemoryLayout<UInt16>.stride == 2` (fp16).

```
DIM = 768, HIDDEN = 2048, SEQ = 256, HEADS = 12
SCORE_CH = HEADS * SEQ = 3072

fwdAttn:
  inputBytes  = DIM * SEQ * 2       = 768 * 256 * 2       = 393,216
  outputBytes = 6 * DIM * SEQ * 2   = 6 * 768 * 256 * 2   = 2,359,296

fwdFFN:
  inputBytes  = DIM * SEQ * 2                              = 393,216
  outputBytes = (2*DIM + 3*HIDDEN) * SEQ * 2
              = (1536 + 6144) * 256 * 2                    = 3,932,160

ffnBwd:
  inputBytes  = (DIM + 2*HIDDEN) * SEQ * 2
              = (768 + 4096) * 256 * 2                     = 2,490,368
  outputBytes = (DIM + 2*HIDDEN) * SEQ * 2                 = 2,490,368

sdpaBwd1:
  inputBytes  = 4 * DIM * SEQ * 2   = 4 * 768 * 256 * 2   = 1,572,864
  outputBytes = (DIM + 2*SCORE_CH) * SEQ * 2
              = (768 + 6144) * 256 * 2                     = 3,538,944

sdpaBwd2 (STATIC — no weights):
  inputBytes  = (2*SCORE_CH + 2*DIM) * SEQ * 2
              = (6144 + 1536) * 256 * 2                    = 3,932,160
  outputBytes = 2 * DIM * SEQ * 2                          = 786,432

qkvBwd:
  inputBytes  = 3 * DIM * SEQ * 2                          = 1,179,648
  outputBytes = DIM * SEQ * 2                              = 393,216
```

---

## Weight Blob Construction Rules

Each kernel needs specific weight blobs. The C code uses `build_blob(ptr, rows, cols)` for normal layout and `build_blob_t(ptr, rows, cols)` for transposed layout.

**Swift equivalents:**
- `build_blob(w, R, C)` → `w.withUnsafeBufferPointer { WeightBlob.build(from: $0, rows: R, cols: C) }`
- `build_blob_t(w, R, C)` → `w.withUnsafeBufferPointer { WeightBlob.buildTransposed(from: $0, rows: R, cols: C) }`
- `get_mask_blob()` → `CausalMask.blob(seqLen: ModelConfig.seqLen)`

### Per-Kernel Weight Specifications

**fwdAttn** (6 weight blobs):
| Path | TensorBuffer | Blob Method | Rows | Cols |
|------|-------------|-------------|------|------|
| `@model_path/weights/rms1.bin` | `rmsAtt` | `build(rows: 1, cols: DIM)` | 1 | 768 |
| `@model_path/weights/wq.bin` | `Wq` | `build(rows: DIM, cols: DIM)` | 768 | 768 |
| `@model_path/weights/wk.bin` | `Wk` | `build(rows: DIM, cols: DIM)` | 768 | 768 |
| `@model_path/weights/wv.bin` | `Wv` | `build(rows: DIM, cols: DIM)` | 768 | 768 |
| `@model_path/weights/wo.bin` | `Wo` | `build(rows: DIM, cols: DIM)` | 768 | 768 |
| `@model_path/weights/mask.bin` | — | `CausalMask.blob(seqLen:)` | — | — |

**fwdFFN** (4 weight blobs):
| Path | TensorBuffer | Blob Method | Rows | Cols |
|------|-------------|-------------|------|------|
| `@model_path/weights/rms2.bin` | `rmsFfn` | `build(rows: 1, cols: DIM)` | 1 | 768 |
| `@model_path/weights/w1.bin` | `W1` | `build(rows: HIDDEN, cols: DIM)` | 2048 | 768 |
| `@model_path/weights/w3.bin` | `W3` | `build(rows: HIDDEN, cols: DIM)` | 2048 | 768 |
| `@model_path/weights/w2.bin` | `W2` | `build(rows: DIM, cols: HIDDEN)` | 768 | 2048 |

**ffnBwd** (3 weight blobs — ALL TRANSPOSED):
| Path | TensorBuffer | Blob Method | Rows | Cols |
|------|-------------|-------------|------|------|
| `@model_path/weights/w2t.bin` | `W2` | `buildTransposed(rows: DIM, cols: HIDDEN)` | 768 | 2048 |
| `@model_path/weights/w1t.bin` | `W1` | `buildTransposed(rows: HIDDEN, cols: DIM)` | 2048 | 768 |
| `@model_path/weights/w3t.bin` | `W3` | `buildTransposed(rows: HIDDEN, cols: DIM)` | 2048 | 768 |

**sdpaBwd1** (2 weight blobs):
| Path | TensorBuffer | Blob Method | Rows | Cols |
|------|-------------|-------------|------|------|
| `@model_path/weights/mask.bin` | — | `CausalMask.blob(seqLen:)` | — | — |
| `@model_path/weights/wot.bin` | `Wo` | `buildTransposed(rows: DIM, cols: DIM)` | 768 | 768 |

**qkvBwd** (3 weight blobs — ALL TRANSPOSED):
| Path | TensorBuffer | Blob Method | Rows | Cols |
|------|-------------|-------------|------|------|
| `@model_path/weights/wqt.bin` | `Wq` | `buildTransposed(rows: DIM, cols: DIM)` | 768 | 768 |
| `@model_path/weights/wkt.bin` | `Wk` | `buildTransposed(rows: DIM, cols: DIM)` | 768 | 768 |
| `@model_path/weights/wvt.bin` | `Wv` | `buildTransposed(rows: DIM, cols: DIM)` | 768 | 768 |

**sdpaBwd2** — NO WEIGHTS (empty array).

### Worked Example: Building fwdAttn Kernel from `borrowing LayerWeights`

This is the pattern repeated 5 times in `LayerKernelSet.init`. Study this example, then apply it to the other 4 kernels by consulting the weight specification tables.

```swift
// 1. Instantiate the generator to get MIL text and I/O sizes.
let gen = SDPAForwardGenerator()

// 2. Build weight blobs from the borrowed LayerWeights.
//    TensorBuffer.withUnsafeBufferPointer yields UnsafeBufferPointer<Float>,
//    which is exactly what WeightBlob.build(from:rows:cols:) accepts.
let rms1Blob = weights.rmsAtt.withUnsafeBufferPointer {
    WeightBlob.build(from: $0, rows: 1, cols: ModelConfig.dim)
}
let wqBlob = weights.Wq.withUnsafeBufferPointer {
    WeightBlob.build(from: $0, rows: ModelConfig.dim, cols: ModelConfig.dim)
}
// ... wk, wv, wo blobs follow the same pattern ...
let maskBlob = CausalMask.blob(seqLen: ModelConfig.seqLen)

// 3. Assemble the weights array with exact path strings.
//    Order within the array does not matter — ANE matches by path string.
let weightsArray: [(path: String, data: Data)] = [
    ("@model_path/weights/rms1.bin", rms1Blob),
    ("@model_path/weights/wq.bin",   wqBlob),
    ("@model_path/weights/wk.bin",   wkBlob),
    ("@model_path/weights/wv.bin",   wvBlob),
    ("@model_path/weights/wo.bin",   woBlob),
    ("@model_path/weights/mask.bin", maskBlob),
]

// 4. Compile the kernel. Use generator properties for I/O sizes.
self.fwdAttn = try ANEKernel(
    milText: gen.milText,
    weights: weightsArray,
    inputBytes: gen.inputBytes,        // DIM*SEQ*2 = 393,216
    outputBytes: gen.outputBytes       // 6*DIM*SEQ*2 = 2,359,296
)
```

**For backward kernels**, replace `WeightBlob.build` with `WeightBlob.buildTransposed`:
```swift
// ffnBwd example — note buildTransposed and the "t" suffix in path
let w2tBlob = weights.W2.withUnsafeBufferPointer {
    WeightBlob.buildTransposed(from: $0, rows: ModelConfig.dim, cols: ModelConfig.hidden)
}
// Path: "@model_path/weights/w2t.bin"
```

### Single-Surface Multi-Output Semantics

Each generator reports `outputByteSizes` as a **single-element array** (e.g., `SDPAForwardGenerator.outputByteSizes == [2_359_296]`). This is ONE IOSurface with concatenated channels, not 6 separate surfaces. The "6" in `6*DIM*SEQ*2` means the output surface has `6*DIM` channels at `SEQ` spatial. Individual taps are read at channel offsets (e.g., `channelOffset=0` for first DIM channels, `channelOffset=4*DIM` for fifth). This detail matters in Phase 6 (ForwardPass/BackwardPass), not here.

---

## Pretrained Weights Binary Format (llama2.c — `stories110M.bin`)

### Header: `Llama2Config` (7 × Int32 = 28 bytes, native-endian/little-endian on Apple Silicon)

Field order:
1. `dim` (Int32)
2. `hidden_dim` (Int32)
3. `n_layers` (Int32)
4. `n_heads` (Int32)
5. `n_kv_heads` (Int32)
6. `vocab_size` (Int32) — **sign encodes weight sharing**: positive = shared (embed == wcls), negative = separate wcls
7. `seq_len` (Int32)

### Payload: contiguous Float32 arrays, in this EXACT order:
1. `embed` — `abs(vocab_size) * dim` floats
2. For `L = 0 ..< n_layers`: `rms_att[L]` — `dim` floats each
3. For `L = 0 ..< n_layers`: `Wq[L]` — `dim * dim` floats each
4. For `L = 0 ..< n_layers`: `Wk[L]` — `dim * dim` floats each
5. For `L = 0 ..< n_layers`: `Wv[L]` — `dim * dim` floats each
6. For `L = 0 ..< n_layers`: `Wo[L]` — `dim * dim` floats each
7. For `L = 0 ..< n_layers`: `rms_ffn[L]` — `dim` floats each
8. For `L = 0 ..< n_layers`: `W1[L]` — `hidden_dim * dim` floats each
9. For `L = 0 ..< n_layers`: `W2[L]` — `dim * hidden_dim` floats each
10. For `L = 0 ..< n_layers`: `W3[L]` — `hidden_dim * dim` floats each
11. `rms_final` — `dim` floats
12. *(Optional)* `wcls` — `abs(vocab_size) * dim` floats, ONLY when `vocab_size < 0`

**CRITICAL**: The file reads each weight TYPE for ALL layers before moving to the next type. This is NOT per-layer grouping — it is per-parameter-type grouping across all layers.

</context>

---

<task>

## Implementation Instructions

Follow TDD: write tests in `Tests/ANERuntimeTests/ANERuntimeTests.swift` (append to existing file), then implement the three source files. Use `swift test --filter ANERuntimeTests` to verify.

### 1. LayerKernelSet.swift

```swift
import Foundation
import IOSurface
import ANETypes
import MILGenerator

/// Owns the 5 weight-bearing ANE kernels for a single transformer layer.
/// Recompiled each batch when weights change. ~Copyable: deinit frees all 5 kernels.
/// Different lifecycle from StaticKernel (sdpaBwd2).
public struct LayerKernelSet: ~Copyable {
    public let fwdAttn: ANEKernel
    public let fwdFFN: ANEKernel
    public let ffnBwd: ANEKernel
    public let sdpaBwd1: ANEKernel
    public let qkvBwd: ANEKernel

    /// Compile all 5 weight-bearing kernels for one layer from its weights.
    /// Uses `borrowing` to avoid copying the ~324 MiB LayerWeights.
    public init(weights: borrowing LayerWeights) throws(ANEError)
}
```

**Implementation rules:**

1. Each kernel is constructed via `ANEKernel(milText:weights:inputBytes:outputBytes:)`.
2. MIL text comes from the generator's `.milText` property. Instantiate the generator, read `.milText`.
3. Weight blobs are built by calling `TensorBuffer.withUnsafeBufferPointer` on the relevant `LayerWeights` field, then passing to `WeightBlob.build(from:rows:cols:)` or `WeightBlob.buildTransposed(from:rows:cols:)`.
4. The causal mask blob is `CausalMask.blob(seqLen: ModelConfig.seqLen)`.
5. Weight paths use the `@model_path/weights/<name>.bin` convention (exact paths in the weight specification tables above).
6. I/O byte sizes use the pre-computed values from the kernel I/O table above (use the generator's `.inputBytes` and `.outputBytes` / `.outputByteSizes[0]` properties instead of hardcoding).
7. All 5 kernel constructions throw `ANEError`. If any fails, previously constructed kernels are freed by Swift's `~Copyable` cleanup (partial init failure).
8. `deinit` is automatic via `~Copyable` — each `ANEKernel` field calls `ane_interop_free` in its own deinit.

### 2. StaticKernel.swift

```swift
import Foundation
import IOSurface
import MILGenerator

/// Wraps a weight-free sdpaBwd2 ANE kernel.
/// Compiled once at startup (or after exec-restart). NOT recompiled when weights change.
/// Different lifecycle from LayerKernelSet: survives LayerKernelSet recompilation.
/// One instance per layer.
public struct StaticKernel: ~Copyable {
    public let kernel: ANEKernel

    /// Compile a weight-free sdpaBwd2 kernel.
    public init() throws(ANEError)
}
```

**Implementation rules:**

1. Uses `SDPABackward2Generator()` for MIL text and I/O sizes.
2. Empty weights array: `weights: []`.
3. One init call: `ANEKernel(milText:weights:inputBytes:outputBytes:)`.

### 3. ModelWeightLoader.swift

```swift
import Foundation
import Darwin
import ANETypes

/// Errors specific to model weight loading.
public enum ModelLoadError: Error, Sendable, Equatable {
    /// File could not be opened.
    case fileNotFound(String)
    /// Header dimensions do not match ModelConfig.
    case configMismatch(expected: String, got: String)
    /// File is truncated — fewer bytes than expected.
    case truncatedFile(expectedBytes: Int, actualBytes: Int)
}

/// Result of loading pretrained weights.
public struct PretrainedWeights: ~Copyable {
    public let layers: LayerStorage<LayerWeights>
    public let rmsFinal: TensorBuffer
    public let embed: TensorBuffer
    /// True when vocab_size > 0 (classifier shares embed weights).
    public let sharedClassifier: Bool
}

/// Loads pretrained weights from the llama2.c binary format.
public enum ModelWeightLoader {
    /// Load weights from a `.bin` file at the given path.
    /// Validates header dimensions against ModelConfig before reading payload.
    public static func load(from path: String) throws(ModelLoadError) -> PretrainedWeights
}
```

**Implementation rules:**

1. Open file with `fopen(path, "rb")`. If nil, throw `.fileNotFound(path)`.
2. Read 7 × Int32 header. Use `withUnsafeMutablePointer(to: &field) { fread($0, 4, 1, file) }` or read all 7 at once into a buffer.
3. Validate: `dim == ModelConfig.dim`, `hidden_dim == ModelConfig.hidden`, `n_layers == ModelConfig.nLayers`. On mismatch, throw `.configMismatch(...)`.
4. Compute `V = abs(vocab_size)`, `shared = vocab_size > 0`.
5. Allocate:
   - `embed = TensorBuffer(count: V * dim, zeroed: false)`
   - `rmsFinal = TensorBuffer(count: dim, zeroed: false)`
   - `layers = LayerStorage<LayerWeights>(count: nLayers) { _ in LayerWeights() }`
6. Read payload in exact llama2.c order (per-parameter-type, all layers). Use a helper to reduce repetition:

   ```swift
   // Helper: read `count` Float32s into a TensorBuffer via fread.
   func readInto(_ buffer: borrowing TensorBuffer, from file: UnsafeMutablePointer<FILE>) {
       buffer.withUnsafeMutablePointer { ptr in
           fread(ptr, MemoryLayout<Float>.stride, buffer.count, file)
       }
   }

   // Payload read order (matches C exactly):
   readInto(embed, from: file)

   for L in 0..<nLayers { readInto(layers[L].rmsAtt, from: file) }
   for L in 0..<nLayers { readInto(layers[L].Wq, from: file) }
   for L in 0..<nLayers { readInto(layers[L].Wk, from: file) }
   for L in 0..<nLayers { readInto(layers[L].Wv, from: file) }
   for L in 0..<nLayers { readInto(layers[L].Wo, from: file) }
   for L in 0..<nLayers { readInto(layers[L].rmsFfn, from: file) }
   for L in 0..<nLayers { readInto(layers[L].W1, from: file) }
   for L in 0..<nLayers { readInto(layers[L].W2, from: file) }
   for L in 0..<nLayers { readInto(layers[L].W3, from: file) }

   readInto(rmsFinal, from: file)
   ```

   Each `for` loop reads ONE parameter type across ALL 12 layers before moving to the next type. This is the llama2.c layout — getting this order wrong silently corrupts every weight matrix.

7. Close file. Return `PretrainedWeights(layers:rmsFinal:embed:sharedClassifier:)`.
8. Use `defer { fclose(file) }` immediately after successful open.
9. Prefer `fread` via `import Darwin` for binary I/O performance (matching the C code exactly).
10. The C code does not check `fread` return values. For Swift, optionally validate that `fread` returns the expected count and throw `.truncatedFile` if not. This is a strict improvement over the C code.

---

## TDD Test Specifications

Append these tests to the existing `Tests/ANERuntimeTests/ANERuntimeTests.swift` file. Tests requiring ANE hardware use `try requireANEHardwareTestsEnabled()`. Tests that can run without hardware (ModelWeightLoader, CompileBudget logic) skip that guard.

### Test 1: `test_layer_kernel_set_compiles_all_five_and_surface_sizes`
- **Requires:** ANE hardware
- **Steps:** Create a `LayerWeights()`, fill all 9 buffers with small known values (e.g., `0.01` via `vDSP_vfill`), call `LayerKernelSet(weights: lw)`.
- **Verify:** Init does not throw. For each of the 5 kernels, access `inputSurface(at: 0)` and `outputSurface(at: 0)`, verify alloc sizes via `IOSurfaceGetAllocSize`:

  | Kernel | Input >= | Output >= |
  |--------|----------|-----------|
  | `fwdAttn` | 393,216 | 2,359,296 |
  | `fwdFFN` | 393,216 | 3,932,160 |
  | `ffnBwd` | 2,490,368 | 2,490,368 |
  | `sdpaBwd1` | 1,572,864 | 3,538,944 |
  | `qkvBwd` | 1,179,648 | 393,216 |

### Test 2: `test_layer_kernel_set_partial_compile_failure_cleanup`
- **Requires:** ANE hardware
- **Steps:**
  1. Save current compile count.
  2. Set compile budget to `currentCount + 2` (room for only 2 kernels).
  3. Attempt `LayerKernelSet(weights: lw)`.
  4. Restore original compile count in `defer`.
- **Verify:** Init throws `ANEError.compileBudgetExhausted`. The first 2 successfully compiled kernels are freed (no resource leak). Verify via `ane_interop_live_handle_count()` returning to baseline.

### Test 3: `test_layer_kernel_set_deinit_frees_all_handles`
- **Requires:** ANE hardware
- **Steps:**
  1. Record baseline `ane_interop_live_handle_count()`.
  2. In a nested scope, compile `LayerKernelSet(weights: lw)`, verify handle count increased by 5.
  3. Let it go out of scope (deinit fires).
- **Verify:** Handle count returns to baseline after deinit.

### Test 4: `test_static_kernel_compiles_without_weights`
- **Requires:** ANE hardware
- **Steps:** Call `StaticKernel()`.
- **Verify:** Init does not throw. Verify `kernel.inputSurface(at: 0)` and `kernel.outputSurface(at: 0)` sizes match sdpaBwd2 I/O: input >= 3,932,160 bytes, output >= 786,432 bytes.

### Test 5: `test_static_kernel_survives_layer_kernel_set_dealloc`
- **Requires:** ANE hardware
- **Steps:**
  1. Create `StaticKernel()`.
  2. In a nested scope, create `LayerKernelSet(weights:)`, let it go out of scope (triggering deinit of all 5 kernels).
  3. After LayerKernelSet is freed, verify `staticKernel.kernel.eval()` still works (or at minimum, the kernel's surfaces are still accessible).
- **Verify:** StaticKernel remains valid after LayerKernelSet deallocation — proving independent lifecycles.

### Test 6: `test_model_weight_loader_config_mismatch`
- **Does NOT require ANE hardware.**
- **Steps:** Create a synthetic binary file in `/tmp` with a mismatched config header: `dim=4, hidden_dim=8, n_layers=1, n_heads=1, n_kv_heads=1, vocab_size=3, seq_len=2` (7 × Int32 = 28 bytes). No payload needed — validation fails on header.
- **Verify:** `ModelWeightLoader.load(from: path)` throws `ModelLoadError.configMismatch`. The error's `expected` string contains "768" (ModelConfig.dim) and `got` string contains "4" (the file's dim).
- **Cleanup:** Remove temp file in `defer`.

### Test 7: `test_model_weight_loader_file_not_found`
- **Does NOT require ANE hardware.**
- **Steps:** Call `ModelWeightLoader.load(from: "/nonexistent/path/model.bin")`.
- **Verify:** Throws `ModelLoadError.fileNotFound`.

### Test 8: `test_model_weight_loader_header_parsing`
- **Does NOT require ANE hardware.**
- **Design decision:** Extract a `static func parseHeader(from file: UnsafeMutablePointer<FILE>) throws(ModelLoadError) -> (dim: Int32, hiddenDim: Int32, nLayers: Int32, nHeads: Int32, nKvHeads: Int32, vocabSize: Int32, seqLen: Int32)` internal helper (or a `Llama2Header` struct). Expose it as `internal` so tests can call it directly.
- **Steps:**
  1. Write a 28-byte file with header: `dim=768, hidden_dim=2048, n_layers=12, n_heads=12, n_kv_heads=12, vocab_size=+32000, seq_len=256`.
  2. Parse it with the header helper.
- **Verify:** All 7 fields read correctly. `vocabSize > 0` (shared classifier).
  3. Write a second file with `vocab_size=-32000`.
  4. Parse it.
- **Verify:** `vocabSize < 0` (unshared classifier). `abs(vocabSize) == 32000`.

### Test 9: `test_model_weight_loader_truncated_file`
- **Does NOT require ANE hardware.**
- **Steps:** Write a file with a valid ModelConfig-matching header (28 bytes) but only 100 bytes of payload (far fewer than expected).
- **Verify:** If fread checking is implemented, throws `ModelLoadError.truncatedFile`. If fread checking is not implemented, this test documents the behavior (partial read with zeros in unread portions). At minimum, the function does not crash.

### Test 10: `test_load_stories110m_weights_integration`
- **Does NOT require ANE hardware, but requires `stories110M.bin` asset.**
- **Steps:** Check if `../../assets/models/stories110M.bin` (or an env-var override `STORIES_MODEL_PATH`) exists. If not, `XCTSkip`.
  - Call `ModelWeightLoader.load(from: path)`.
- **Verify:**
  - `result.layers.count == 12`
  - `result.rmsFinal.count == 768`
  - `result.embed.count == 32_000 * 768` (= 24,576,000)
  - `result.sharedClassifier == true` (stories110M uses shared embed/cls)
  - Spot-check: `result.layers[0].Wq` has at least one non-zero element.
  - Spot-check: `result.rmsFinal` has at least one non-zero element.

### Test 11: `test_layer_kernel_set_recompile_with_different_weights`
- **Requires:** ANE hardware
- **Steps:**
  1. Create `LayerWeights` filled with value `A` (e.g., 0.01).
  2. Compile `LayerKernelSet(weights: lwA)`.
  3. Create `LayerWeights` filled with value `B` (e.g., 0.02, different).
  4. Compile second `LayerKernelSet(weights: lwB)`.
- **Verify:** Both compile successfully without throwing. This proves kernels can be compiled multiple times with different weights (matching the ObjC recompile-per-batch pattern).

### Test 12: `test_static_kernel_eval_produces_output`
- **Requires:** ANE hardware
- **Steps:**
  1. Compile `StaticKernel()`.
  2. Write known fp16 input data to `kernel.inputSurface(at: 0)` using `SurfaceIO.writeFP16`.
  3. Call `kernel.kernel.eval()`.
  4. Read output from `kernel.kernel.outputSurface(at: 0)` using `SurfaceIO.readFP16`.
- **Verify:** Output is finite (no NaN/Inf) and not all-zero (the kernel actually computed something). Exact numerical values are not tested here — that's Phase 6 verification territory.

</task>

---

<constraints>

## Critical Rules

1. **Swift 6.2 strict concurrency**: All code compiles with `.swiftLanguageMode(.v6)`. No `@Sendable` closure issues.
2. **~Copyable everywhere**: `LayerKernelSet`, `StaticKernel`, `PretrainedWeights` are all `~Copyable`. Use `borrowing` for read-only access to `LayerWeights` in `LayerKernelSet.init`.
3. **Typed throws**: All throwing functions use typed throws (`throws(ANEError)` or `throws(ModelLoadError)`).
4. **Weight blob lifetime**: Build all weight blobs before passing to `ANEKernel.init`. The `Data` objects must stay alive through the init call. Local `let` bindings are sufficient — they live until end of scope.
5. **Weight path strings**: Must match exactly: `@model_path/weights/<name>.bin`. Any typo = silent ANE compile failure.
6. **Transposed weights for backward kernels**: `ffnBwd` uses `buildTransposed` for W2, W1, W3. `sdpaBwd1` uses `buildTransposed` for Wo. `qkvBwd` uses `buildTransposed` for Wq, Wk, Wv. Forward kernels use normal `build`.
7. **Import chain**: `ANERuntime` depends on `ANEInterop`, `ANETypes`, `MILGenerator`. Import only what each file needs.
8. **fread for binary I/O**: `ModelWeightLoader` uses `import Darwin` and `fopen`/`fread`/`fclose` for binary I/O. Match the C code exactly.
9. **Per-type-all-layers read order**: The llama2.c format reads each parameter type across ALL layers before moving to the next type. The read loop structure is `for each type { for L in 0..<nLayers { read into layers[L].field } }`, NOT `for L { for each type { read } }`.
10. **Existing test file**: Append new tests to the EXISTING `Tests/ANERuntimeTests/ANERuntimeTests.swift` file. Use the existing `requireANEHardwareTestsEnabled()` and `makeIdentityKernel()` helpers. Use the existing imports. The file currently has `import ANERuntime` — change it to `@testable import ANERuntime` to access `internal` helpers like `parseHeader` (Test 8).
11. **No new Package.swift changes**: The `ANERuntime` target already includes all needed dependencies (`ANEInterop`, `ANETypes`, `MILGenerator`).
12. **Compiler fallback for `borrowing` in throwing init**: Swift 6.2 has known bugs with `borrowing` parameters in throwing initializers crossing module boundaries ([swift#86609](https://github.com/swiftlang/swift/issues/86609)). If you get `"copy of noncopyable typed value"` or a compiler crash, fall back to accepting an `UnsafePointer`-based interface or a helper function that builds the weight blobs outside the init. The `borrowing` approach is correct and preferred — only use a workaround if the compiler rejects it.
13. **`readInto` helper for ModelWeightLoader**: The `readInto(_ buffer: borrowing TensorBuffer, from: ...)` pattern requires `borrowing` access to a `TensorBuffer` inside a `LayerStorage` subscript. Since `LayerStorage` uses `_modify` coroutine accessors, you get a mutable reference. Call `withUnsafeMutablePointer` (not `withUnsafePointer`) for `fread`. If `borrowing TensorBuffer` causes issues with `_modify`, use a local `func` that takes `UnsafeMutablePointer<Float>` and count instead.

## Verification Criteria

| Check | Threshold |
|-------|-----------|
| `swift build` compiles cleanly | Zero errors, zero warnings |
| `swift test --filter ANERuntimeTests` | All non-hardware tests pass; hardware tests pass with `ANE_HARDWARE_TESTS=1` or skip cleanly |
| LayerKernelSet surface sizes match | Exact byte size match against kernel I/O table |
| StaticKernel independent lifecycle | Survives LayerKernelSet deallocation |
| ModelWeightLoader error paths | fileNotFound and configMismatch throw correctly |
| ModelWeightLoader integration | stories110M loads with correct dimensions and non-zero weights |
| New test count | 12 new tests appended to ANERuntimeTests |
| No regressions | All 99 existing tests still pass |

</constraints>

---

<output_format>

## Deliverables (3 files to create + 1 file to modify)

1. **`Sources/ANERuntime/LayerKernelSet.swift`** — Complete implementation
2. **`Sources/ANERuntime/StaticKernel.swift`** — Complete implementation
3. **`Sources/ANERuntime/ModelWeightLoader.swift`** — Complete implementation with `ModelLoadError`, `PretrainedWeights`, and `ModelWeightLoader`
4. **`Tests/ANERuntimeTests/ANERuntimeTests.swift`** — Append test methods to existing `ANERuntimeTests` class

## Workflow

1. Read the existing test file to understand the helper functions and patterns.
2. Write ALL test methods first (TDD). Run `swift test --filter ANERuntimeTests` — tests should fail (no implementation yet).
3. Implement `StaticKernel.swift` first (simplest — just sdpaBwd2).
4. Implement `LayerKernelSet.swift` (depends on understanding all 5 kernel specs).
5. Implement `ModelWeightLoader.swift` (independent of kernel compilation).
6. Run `swift test --filter ANERuntimeTests` — all tests should pass.
7. Run `swift test` — all 99+ tests should pass with zero regressions.

</output_format>

---

<agentic_guidance>

## Persistence

Keep working until ALL tests pass. If a kernel compile fails, inspect the MIL text and weight blob construction. If `fread` returns wrong values, verify the read order matches the llama2.c format exactly. Only stop when `swift test` shows zero failures and zero regressions.

## Tool Usage

Read source files to verify APIs before calling them. Check `Sources/ANERuntime/ANEKernel.swift` for the exact `init` signature. Check `Sources/MILGenerator/*.swift` for the exact generator API. Verify weight sizes against `ModelConfig`. If unsure about any API, read the source — do not guess.

## Planning

Before writing code:
1. Trace through the C `compile_layer_kernels` function line by line.
2. Map each ObjC dictionary entry `@"path": build_blob(ptr, R, C)` to the Swift equivalent `(path: "path", data: weights.field.withUnsafeBufferPointer { WeightBlob.build(from: $0, rows: R, cols: C) })`.
3. Cross-reference each path, blob method (normal vs transposed), rows, cols, and I/O size against the weight specification tables in this prompt.
4. Verify the I/O sizes match both the pre-computed table AND the generator properties (they must agree).

## Common Pitfalls (Ranked by Frequency)

1. **Wrong blob method**: Backward kernels (ffnBwd, sdpaBwd1, qkvBwd) use `buildTransposed`. Forward kernels (fwdAttn, fwdFFN) use `build`. Getting even one wrong = silent garbage output.
2. **Wrong `rows`/`cols` order**: `build_blob(w->W1, HIDDEN, DIM)` means `rows=HIDDEN, cols=DIM` — the C code puts rows first. Double-check against the weight specification tables.
3. **Forgetting the causal mask**: Both `fwdAttn` AND `sdpaBwd1` need `CausalMask.blob(seqLen: ModelConfig.seqLen)` as `@model_path/weights/mask.bin`. Missing it = ANE compile failure.
4. **llama2.c read order**: Per-type-all-layers, NOT per-layer-all-types. The inner loop is `for L in 0..<nLayers`, the outer structure is the parameter type. This is the #1 source of silent corruption.
5. **`borrowing` compiler issues**: If the compiler rejects `borrowing LayerWeights` in a throwing init, see constraint #12 for the workaround.
6. **Weight path typos**: `@model_path/weights/wqt.bin` (with `t` suffix) for transposed, `@model_path/weights/wq.bin` (no suffix) for normal. A typo means the weight is silently ignored.

</agentic_guidance>
