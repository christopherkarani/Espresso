# Espresso Architecture — Initial State

**Date:** 2026-03-05
**Purpose:** Complete architectural reference for the Espresso ANE training system as of the initial Swift 6.2 rewrite, prior to inference-path optimizations. This document describes what the system is, how every layer works, and why each design decision was made.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Hardware Target: Apple Neural Engine](#2-hardware-target-apple-neural-engine)
3. [Reverse-Engineered Private APIs](#3-reverse-engineered-private-apis)
4. [System Architecture](#4-system-architecture)
5. [Target Dependency Graph](#5-target-dependency-graph)
6. [Layer 1: ANEInterop — C/ObjC Private API Bridge](#6-layer-1-aneinterop)
7. [Layer 2: ANETypes — Value Types and IOSurface I/O](#7-layer-2-anetypes)
8. [Layer 3: MILGenerator — MIL Program Text Generation](#8-layer-3-milgenerator)
9. [Layer 4: CPUOps — Accelerate-Backed CPU Kernels](#9-layer-4-cpuops)
10. [Layer 5: ANERuntime — Kernel Lifecycle Management](#10-layer-5-aneruntime)
11. [Layer 6: Espresso — Transformer Training Orchestration](#11-layer-6-espresso)
12. [The Forward Pass in Detail](#12-the-forward-pass-in-detail)
13. [The Backward Pass in Detail](#13-the-backward-pass-in-detail)
14. [IOSurface Memory Model](#14-iosurface-memory-model)
15. [FP16 Conversion Pipeline](#15-fp16-conversion-pipeline)
16. [Compile Budget and exec() Restart](#16-compile-budget-and-exec-restart)
17. [Performance Baseline](#17-performance-baseline)
18. [Key Design Decisions](#18-key-design-decisions)
19. [Known Limitations](#19-known-limitations)

---

## 1. Project Overview

Espresso is a from-scratch implementation of transformer training (forward + backward pass + optimizer) running directly on the Apple Neural Engine (ANE) in Apple Silicon. Apple does not expose the ANE for training — only inference via Core ML. This project reverse-engineers the `_ANEClient` and `_ANECompiler` private APIs to compile and execute custom compute graphs, including backpropagation kernels, on ANE hardware.

The system was originally implemented in Objective-C (`training/train_large.m`), then rewritten in Swift 6.2 with strict concurrency, move-only types, and typed throws. The Swift rewrite achieves bit-identical numerical parity with the ObjC reference and runs 10% faster on the 12-layer training benchmark.

### What Makes This Unusual

- **No public API exists** for ANE training. Core ML exposes inference only.
- **Weight recompilation** — ANE programs embed weights as constants. Updating weights requires recompiling the entire kernel. This is fundamentally different from GPU training where weights are mutable buffers.
- **FP16 compute, FP32 storage** — ANE operates in FP16. All host-side storage is FP32. Every kernel invocation requires FP32→FP16 conversion on write and FP16→FP32 on read.
- **IOSurface shared memory** — Data passes between CPU and ANE via IOSurface objects, which are kernel-managed shared memory buffers. Access requires explicit lock/unlock syscalls.
- **~119 compile limit** — The ANE compiler leaks resources. After approximately 100 compilations, the process must checkpoint and `exec()` restart to reclaim resources.

---

## 2. Hardware Target: Apple Neural Engine

The ANE is a fixed-function matrix accelerator integrated into Apple Silicon SoCs.

| Property | Value |
|----------|-------|
| Peak throughput (M4) | 15.8 TFLOPS (FP16) |
| Peak throughput (M3 Max) | 18.0 TFLOPS (FP16) |
| Data format | FP16 (IEEE 754 half-precision) |
| Memory interface | IOSurface (shared with CPU via unified memory) |
| Programming model | MIL (Model Intermediate Language) → compiled to ANE micro-ops |
| Supported ops | Conv2D, MatMul, Softmax, ReLU, Sigmoid, Add, Mul, Reduce, Reshape, Transpose, Concat |
| Not supported | Scatter/gather, dynamic shapes, control flow, custom ops |

The ANE is designed for inference workloads: load a compiled model once, run it many times with different inputs. Espresso subverts this by treating each training step as a new "inference" call with updated weights baked into freshly compiled kernels.

---

## 3. Reverse-Engineered Private APIs

Espresso accesses the ANE through four private Objective-C classes resolved at runtime via `dlopen` and `NSClassFromString`:

```
/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine
```

| Class | Role |
|-------|------|
| `_ANEInMemoryModelDescriptor` | Parses MIL program text + weight blobs into a model descriptor |
| `_ANEInMemoryModel` | Wraps a descriptor; provides compile/load/evaluate methods |
| `_ANERequest` | Packages IOSurface inputs/outputs for a single evaluation |
| `_ANEIOSurfaceObject` | Wraps an IOSurfaceRef for the ANE runtime |

### API Call Sequence

```
                ┌──────────────────────────────────────┐
                │ 1. modelWithMILText:weights:options: │
                │    Parse MIL → Descriptor            │
                └──────────────┬───────────────────────┘
                               ↓
                ┌──────────────────────────────────────┐
                │ 2. inMemoryModelWithDescriptor:      │
                │    Wrap → _ANEInMemoryModel          │
                └──────────────┬───────────────────────┘
                               ↓
                ┌──────────────────────────────────────┐
                │ 3. compileWithQoS:21 options: error: │
                │    MIL → ANE bytecode                │
                └──────────────┬───────────────────────┘
                               ↓
                ┌──────────────────────────────────────┐
                │ 4. loadWithQoS:21 options: error:    │
                │    Pin bytecode to ANE hardware      │
                └──────────────┬───────────────────────┘
                               ↓
                ┌──────────────────────────────────────┐
                │ 5. IOSurface allocation              │
                │    Create input/output buffers       │
                └──────────────┬───────────────────────┘
                               ↓
                ┌──────────────────────────────────────┐
                │ 6. requestWithInputs:outputs:...     │
                │    Package surfaces into request     │
                └──────────────┬───────────────────────┘
                               ↓
          ┌─────────────────────────────────────────────────┐
          │ 7. evaluateWithQoS:21 options: request: error:  │
          │    Execute on ANE hardware                      │
          │    (repeatable — same request, different data)   │
          └─────────────────────────────────────────────────┘
```

All `objc_msgSend` calls are performed through the C bridge in `ane_interop.m`. Swift never touches Objective-C directly.

### Weight File Handling

Weights are written to a temporary directory during compilation:

```
/tmp/<hex-session-id>/weights/
├── wq.bin      (768×768 FP16 + 64-byte header)
├── wk.bin
├── wv.bin
├── wo.bin
├── w1.bin      (2048×768 FP16 + 64-byte header)
├── w2.bin      (768×2048 FP16 + 64-byte header)
├── w3.bin
├── rms1.bin    (1×768 FP16 + 64-byte header)
├── rms2.bin
└── mask.bin    (256×256 FP16 causal mask + 64-byte header)
```

The MIL program references these as `BLOBFILE(path=string("@model_path/weights/wq.bin"), offset=uint64(64))`. The ANE compiler resolves `@model_path` to the temp directory at compile time.

---

## 4. System Architecture

### The Six Kernels

Each transformer layer uses 6 ANE kernels per training step:

| Kernel | Direction | Input Shape | Output Shape | Weights |
|--------|-----------|-------------|-------------|---------|
| `fwdAttn` | Forward | `[1,768,1,256]` | `[1,4608,1,256]` (6×dim) | Wq, Wk, Wv, Wo, rms1, mask |
| `fwdFFN` | Forward | `[1,768,1,256]` | `[1,7680,1,256]` (2d+3h) | W1, W2, W3, rms2 |
| `ffnBwd` | Backward | `[1,3584,1,256]` | `[1,6912,1,256]` | W2^T, W1^T, W3^T |
| `sdpaBwd1` | Backward | `[1,3072,1,256]` | `[1,6912,1,256]` | Wo^T, mask |
| `sdpaBwd2` | Backward | `[1,6912,1,256]` | `[1,1536,1,256]` | (none — static) |
| `qkvBwd` | Backward | `[1,2304,1,256]` | `[1,768,1,256]` | Wq^T, Wk^T, Wv^T |

5 of the 6 kernels embed weights and must be recompiled when weights change. `sdpaBwd2` is weight-free (a "static kernel") and compiled once.

### CPU Responsibilities

The CPU handles operations that the ANE cannot perform or that are more efficient on CPU:

| Operation | Framework | Why CPU |
|-----------|-----------|---------|
| RMSNorm backward | Accelerate/vDSP | Requires element-wise division and reduction not easily fused |
| Residual connections | Accelerate/vDSP | Element-wise add between kernel outputs |
| Cross-entropy loss | Accelerate/vDSP | Softmax + log + target indexing |
| dW gradient accumulation | Accelerate/cblas | Large matrix multiply (sgemm) — CPU GEMM is highly optimized |
| Adam optimizer | vDSP | Element-wise with bias correction |
| Embedding lookup/backward | Manual | Scatter/gather not supported on ANE |

---

## 5. Target Dependency Graph

```
ANEInterop (ObjC/C — private API bridge)
    ├── ANETypes (Swift value types, ~Copyable buffers, IOSurface I/O)
    │       ├── MILGenerator (MIL program text generation for all 6 kernels)
    │       │       └── ANERuntime (compile/eval lifecycle, kernel sets, weight loading)
    │       │               └── Espresso (transformer layers, forward/backward, training loop)
    │       │                       ├── EspressoTrain (CLI executable)
    │       │                       └── EspressoBench (benchmark executable)
    │       └── CPUOps (Accelerate-backed CPU kernels: RMSNorm, CrossEntropy, Adam, etc.)
    │               └── Espresso
```

| Target | Language | LOC | Role |
|--------|----------|-----|------|
| `ANEInterop` | ObjC/C | ~600 | `dlopen` bridge to private ANE APIs, IOSurface I/O, NEON FP16 conversion |
| `ANETypes` | Swift 6.2 | ~900 | `~Copyable` value types: `TensorBuffer`, `LayerWeights`, `LayerActivations`, `SurfaceIO` |
| `MILGenerator` | Swift 6.2 | ~800 | Generates MIL program text for all 6 kernel types + causal mask + weight blobs |
| `CPUOps` | Swift 6.2 | ~400 | RMSNorm, CrossEntropy, Adam, Embedding, RoPE, SiLU via Accelerate |
| `ANERuntime` | Swift 6.2 | ~500 | `ANEKernel` (~Copyable), `LayerKernelSet`, `StaticKernel`, `CompileBudget`, model loader |
| `Espresso` | Swift 6.2 | ~700 | `ForwardPass`, `BackwardPass`, `GradientAccumulator`, `Checkpoint`, training loop |
| `EspressoTrain` | Swift 6.2 | ~200 | CLI entry point for training runs |
| `EspressoBench` | Swift 6.2 | ~500 | Benchmark harness comparing ANE Direct vs Core ML |

All Swift targets use `.swiftLanguageMode(.v6)` for strict concurrency checking. No external dependencies — only Apple system frameworks (Foundation, CoreML, IOSurface, Accelerate).

---

## 6. Layer 1: ANEInterop

**Path:** `Sources/ANEInterop/`

### Files

| File | Purpose |
|------|---------|
| `include/ane_interop.h` | C API header — all public declarations |
| `ane_interop.m` | ObjC implementation — dlopen, objc_msgSend, compile/eval |
| `surface_io.c` | IOSurface read/write/copy with lock management |
| `neon_convert.c` | ARM NEON vectorized FP16↔FP32 conversion |

### ANEHandle Lifecycle

The C bridge represents a compiled kernel as an opaque `ANEHandle*`:

```c
struct ANEHandle {
    void *model;           // CFBridgingRetain'd _ANEInMemoryModel
    IOSurfaceRef *ioInputs;    // Array of input IOSurfaces
    IOSurfaceRef *ioOutputs;   // Array of output IOSurfaces
    void *request;         // CFBridgingRetain'd _ANERequest
    void *tmpDir;          // CFBridgingRetain'd NSString (temp weight directory)
    int nInputs, nOutputs;
    size_t *inputBytes;
    size_t *outputBytes;
    bool liveHandleCounted;
};
```

`ane_interop_compile()` allocates this struct and populates all fields. `ane_interop_free()` releases all CF objects, deallocates surfaces, and removes the temp directory.

### Surface I/O Functions

All surface operations follow a strict lock/unlock protocol:

```c
// Write: acquire exclusive lock, convert FP32→FP16, write, unlock
IOSurfaceLock(surface, 0, NULL);                    // Write lock
ane_interop_cvt_f32_to_f16(base, data, count);      // NEON vectorized
IOSurfaceUnlock(surface, 0, NULL);

// Read: acquire read-only lock, convert FP16→FP32, unlock
IOSurfaceLock(surface, kIOSurfaceLockReadOnly, NULL);
ane_interop_cvt_f16_to_f32(data, base + offset, count);
IOSurfaceUnlock(surface, kIOSurfaceLockReadOnly, NULL);
```

Batched variants (`_batched`) amortize the lock overhead by performing multiple region reads/writes under a single lock pair.

### NEON FP16 Conversion

The conversion routines process 8 elements at a time using ARM NEON vector instructions:

```c
// FP16 → FP32 (8 elements per iteration)
float16x8_t h = vld1q_f16(src + i);                    // Load 8 FP16
vst1q_f32(dst + i,     vcvt_f32_f16(vget_low_f16(h)));  // Convert low 4
vst1q_f32(dst + i + 4, vcvt_f32_f16(vget_high_f16(h))); // Convert high 4

// FP32 → FP16 (8 elements per iteration)
float16x8_t h = vcombine_f16(
    vcvt_f16_f32(vld1q_f32(src + i)),      // Convert 4 floats
    vcvt_f16_f32(vld1q_f32(src + i + 4))   // Convert next 4
);
vst1q_f16(dst + i, h);                     // Store 8 FP16
```

Scalar fallback handles the tail elements when count is not a multiple of 8.

### Safety Features

- **Overflow checking** on all size arithmetic via `__builtin_mul_overflow` / `__builtin_add_overflow`
- **Bounds validation** before every memory access against `IOSurfaceGetAllocSize()`
- **Null pointer checks** on all inputs
- **Goto-cleanup pattern** ensures `IOSurfaceUnlock` is called even on error paths
- **Weight path sanitization** rejects `..` and `/` components to prevent directory traversal
- **Atomic globals** for compile count, live handle count, and error tracking

---

## 7. Layer 2: ANETypes

**Path:** `Sources/ANETypes/`

### TensorBuffer (~Copyable)

The fundamental storage type. All weight, activation, and gradient buffers are `TensorBuffer` instances.

```swift
public struct TensorBuffer: ~Copyable {
    public let count: Int
    private let rawStorage: UnsafeMutableRawPointer  // 64-byte aligned
    private let baseAddress: UnsafeMutablePointer<Float>

    public init(count: Int, zeroed: Bool)
    deinit { rawStorage.deallocate() }
}
```

- **64-byte alignment** — compatible with AVX-512 and vDSP vector operations
- **~Copyable** — prevents accidental duplication of large buffers (a single `LayerWeights` is ~31 MB)
- **Closure-based access** — `withUnsafePointer`, `withUnsafeMutableBufferPointer` etc. ensure pointer validity is scoped

### ModelConfig

All model dimensions are compile-time constants:

```swift
public enum ModelConfig {
    public static let dim = 768
    public static let hidden = 2048
    public static let heads = 12
    public static let headDim = 64      // dim / heads
    public static let seqLen = 256
    public static let nLayers = 12
    public static let vocab = 32_000
    public static let maxCompiles = 100
    public static let kernelsPerLayer = 5
}
```

### LayerWeights

9 weight tensors per transformer layer:

```swift
public struct LayerWeights: ~Copyable {
    public let Wq: TensorBuffer      // [768, 768]   = 589,824 floats
    public let Wk: TensorBuffer      // [768, 768]
    public let Wv: TensorBuffer      // [768, 768]
    public let Wo: TensorBuffer      // [768, 768]
    public let W1: TensorBuffer      // [2048, 768]   = 1,572,864 floats
    public let W2: TensorBuffer      // [768, 2048]
    public let W3: TensorBuffer      // [2048, 768]
    public let rmsAtt: TensorBuffer  // [1, 768]
    public let rmsFfn: TensorBuffer  // [1, 768]
}
// Total per layer: ~7.7M floats = ~31 MB (FP32)
```

### LayerActivations

13 activation buffers per layer, stored during the forward pass for use in the backward pass:

```swift
public struct LayerActivations: ~Copyable {
    public let layerIn: TensorBuffer   // [768, 256]  — input to this layer (for RMSNorm backward)
    public let xnorm: TensorBuffer     // [768, 256]  — after RMSNorm (for QKV backward)
    public let Q: TensorBuffer         // [768, 256]  — query projection
    public let K: TensorBuffer         // [768, 256]  — key projection
    public let V: TensorBuffer         // [768, 256]  — value projection
    public let attnOut: TensorBuffer   // [768, 256]  — after Wo projection
    public let oOut: TensorBuffer      // [768, 256]  — attention output (before residual)
    public let x2: TensorBuffer        // [768, 256]  — after attention residual
    public let x2norm: TensorBuffer    // [768, 256]  — after second RMSNorm
    public let h1: TensorBuffer        // [2048, 256] — W1(x) projection
    public let h3: TensorBuffer        // [2048, 256] — W3(x) projection
    public let siluOut: TensorBuffer   // [2048, 256] — SiLU(h1) * h3
    public let ffnOut: TensorBuffer    // [768, 256]  — W2(gate) output (before residual)
}
// Total per layer: 10×196,608 + 3×524,288 = ~3.5M floats = ~14 MB (FP32)
```

### LayerStorage

A fixed-size, `~Copyable` container using coroutine accessors:

```swift
public struct LayerStorage<Element: ~Copyable>: ~Copyable {
    private var storage: [Element]

    public subscript(index: Int) -> Element {
        _read { yield storage[index] }
        _modify { yield &storage[index] }
    }
}
```

The `_read` / `_modify` coroutine accessors allow borrowing elements without copying them out of the array. This is critical for `LayerStorage<LayerKernelSet>` where each element is a ~Copyable struct owning ANE kernel handles.

### SurfaceIO

Swift wrappers for the C surface I/O functions, with precondition-based safety:

```swift
public enum SurfaceIO {
    // Single-region operations
    public static func writeFP16(to:data:channels:spatial:)
    public static func readFP16(from:into:channelOffset:channels:spatial:)

    // Batched operations (single lock for N regions)
    public static func readFP16Batched(from:spatial:regions:)
    public static func writeFP16AtBatched(to:spatial:regions:)

    // Surface-to-surface copies
    public static func copyFP16(dst:dstChannelOffset:src:srcChannelOffset:channels:spatial:)
    public static func copyFP16Batched(dst:src:spatial:regions:)
    public static func copyFP16FromMultipleSources(dst:spatial:regions:)
}
```

### WeightBlob

Converts FP32 weight tensors into the ANE blob format (64-byte header + FP16 payload):

```swift
public enum WeightBlob {
    // Header: [0]=0x01, [4]=0x02, [64..67]=0xDEADBEEF (magic)
    public static func build(from:rows:cols:) -> Data        // Row-major FP16
    public static func buildTransposed(from:rows:cols:) -> Data  // Column-major FP16
}
```

Transposed blobs are used for backward kernels where the weight matrices are applied in the opposite direction (e.g., Wo^T for the SDPA backward pass).

---

## 8. Layer 3: MILGenerator

**Path:** `Sources/MILGenerator/`

### MIL Program Structure

Every kernel is a self-contained MIL program:

```mil
program(1.3)
[buildInfo = dict<string, string>({
    {"coremlc-component-MIL", "3510.2.1"},
    {"coremlc-version", "3505.4.1"},
    {"coremltools-component-milinternal", ""},
    {"coremltools-version", "9.0"}
})]
{
    func main<ios18>(tensor<fp16, [1, 768, 1, 256]> x) {
        // ... computation graph ...
    } -> (out);
}
```

The `buildInfo` metadata mimics coremltools output to satisfy the ANE compiler's version checks.

### Generator Protocol

```swift
public protocol MILProgramGenerator: Sendable {
    var milText: String { get }
    var inputBytes: Int { get }
    var outputByteSizes: [Int] { get }
}
```

### Forward Kernel Output Layout

**SDPAForwardGenerator** outputs 6 concatenated tensors:

```
Output: [1, 6*dim, 1, seqLen] = [1, 4608, 1, 256]

Channel layout:
  [0, dim)         → oo     (output projection result)
  [dim, 2*dim)     → qf     (query projection — for backward)
  [2*dim, 3*dim)   → kf     (key projection — for backward)
  [3*dim, 4*dim)   → vf     (value projection — for backward)
  [4*dim, 5*dim)   → af     (attention output pre-Wo — for backward)
  [5*dim, 6*dim)   → xn     (RMSNorm output — for backward)
```

**FFNForwardGenerator** outputs 5 concatenated tensors:

```
Output: [1, 2*dim + 3*hidden, 1, seqLen] = [1, 7680, 1, 256]

Channel layout:
  [0, dim)                       → y      (FFN output)
  [dim, dim+hidden)              → h1     (W1 projection — for backward)
  [dim+hidden, dim+2*hidden)     → h3     (W3 projection — for backward)
  [dim+2*hidden, dim+3*hidden)   → gate   (SiLU gate — for backward)
  [dim+3*hidden, 2*dim+3*hidden) → xn     (RMSNorm output — for backward)
```

### Why Concatenated Outputs?

The ANE can only write to pre-allocated IOSurface regions. By concatenating all needed activations into a single output tensor, one kernel execution produces everything the backward pass needs. The CPU then reads specific channel slices using `readFP16Batched` with channel offsets.

This is a deliberate trade-off: the ANE writes more data than inference needs (6× for attention, 10× for FFN), but it avoids recomputation during the backward pass.

### MIL Operations Used

| MIL Op | Espresso Usage |
|--------|---------------|
| `conv` (1×1 pointwise) | All linear projections (Wq, Wk, Wv, Wo, W1, W2, W3) |
| `matmul` | Q@K^T (attention scores), scores@V (attention output) |
| `softmax` | Attention weight normalization |
| `add` | Causal mask application, residual connections |
| `mul` | RMSNorm scaling, attention scaling, SiLU gating |
| `sigmoid` | SiLU activation (σ(x) component) |
| `pow` | RMSNorm inverse root (x^(-0.5)) |
| `reduce_sum` | RMSNorm variance computation |
| `reshape` | Multi-head attention head splitting/merging |
| `transpose` | Multi-head dimension reordering |
| `concat` | Combining multiple outputs into single surface |

Linear projections use `conv` (1×1 pointwise convolution) rather than `matmul` because ANE's conv hardware is more efficient for the `[1, C, 1, S]` tensor layout.

### Locale Safety

All numeric formatting uses `Locale(identifier: "en_US_POSIX")` to prevent locale-dependent decimal separator issues (e.g., German locale using `,` instead of `.`).

---

## 9. Layer 4: CPUOps

**Path:** `Sources/CPUOps/`

| Module | Operations | Framework |
|--------|-----------|-----------|
| `RMSNorm.swift` | Forward + backward | vDSP |
| `CrossEntropy.swift` | Loss + gradient | vDSP |
| `AdamOptimizer.swift` | Weight update with bias correction | vDSP |
| `Embedding.swift` | Lookup + backward scatter | Manual |
| `RoPE.swift` | Rotary position encoding fwd/bwd | Manual |
| `SiLU.swift` | SiLU activation fwd/bwd | vDSP |

### RMSNorm

```
Forward:  xnorm = x * rsqrt(mean(x²) + ε) * weight
Backward: dx = rrms * (dy*w - x * dot(dy*w, x) / (sum(x²)/d + ε))
          dw += sum(dy * x * rrms, axis=spatial)
```

The backward pass was corrected post-review to handle non-unit weights correctly. The original implementation had a bug where `dx` was computed without accounting for the weight scaling.

### Cross-Entropy

Uses the numerically stable log-sum-exp formulation:

```
loss = -logits[target] + log(sum(exp(logits - max(logits))))
grad[i] = softmax(logits)[i] - (i == target ? 1 : 0)
```

---

## 10. Layer 5: ANERuntime

**Path:** `Sources/ANERuntime/`

### ANEKernel

The fundamental unit of ANE execution:

```swift
public struct ANEKernel: ~Copyable {
    private let handle: OpaquePointer  // ANEHandle* from C

    public init(milText:weights:inputBytes:outputBytes:) throws(ANEError)
    deinit { ane_interop_free(handle) }

    public func eval() throws(ANEError)
    public func inputSurface(at:) throws(ANEError) -> IOSurfaceRef   // Retained
    public func outputSurface(at:) throws(ANEError) -> IOSurfaceRef  // Retained
}
```

Compilation is serialized via `CompileGate.lock` (an `NSLock`) to prevent concurrent compilations that could exhaust ANE resources.

### LayerKernelSet

Owns the 5 weight-bearing kernels for one transformer layer:

```swift
public struct LayerKernelSet: ~Copyable {
    public let fwdAttn: ANEKernel      // SDPAForwardGenerator
    public let fwdFFN: ANEKernel       // FFNForwardGenerator
    public let ffnBwd: ANEKernel       // FFNBackwardGenerator
    public let sdpaBwd1: ANEKernel     // SDPABackward1Generator
    public let qkvBwd: ANEKernel       // QKVBackwardGenerator
}
```

Recompiled every batch when weights change. Uses `borrowing LayerWeights` to avoid copying ~31 MB per layer.

### StaticKernel

The weight-free `sdpaBwd2` kernel, compiled once and reused:

```swift
public struct StaticKernel: ~Copyable {
    public let kernel: ANEKernel  // SDPABackward2Generator (no weights)
}
```

### CompileBudget

```swift
public enum CompileBudget {
    public static let maxCompiles = 100  // ANE resource limit
    public static var remaining: Int { max(0, maxCompiles - currentCount) }
    public static var isExhausted: Bool { currentCount >= maxCompiles }
}
```

---

## 11. Layer 6: Espresso

**Path:** `Sources/Espresso/`

### GradientAccumulator

Enables CPU gradient computation to overlap with ANE kernel execution:

```swift
public final class GradientAccumulator: @unchecked Sendable {
    private let queue: DispatchQueue    // Serial queue for dW accumulation
    private let group: DispatchGroup    // Tracks pending work

    public func enqueue(_ block: @escaping @Sendable () -> Void)
    public func barrier()  // Block until all enqueued work completes
}
```

The forward pass enqueues cblas_sgemm calls for gradient accumulation. The backward pass calls `barrier()` before using the accumulated gradients.

### StepTimingBreakdown

Fine-grained timing for performance analysis:

```swift
public struct StepTimingBreakdown {
    public var tAne: Double        // ANE kernel execution
    public var tIO: Double         // IOSurface read/write (FP16 conversion)
    public var tCls: Double        // Cross-entropy / classifier
    public var tElem: Double       // CPU element-wise ops (residuals, etc.)
    public var tRms: Double        // RMSNorm backward
    public var tCblasWait: Double  // Time waiting for async cblas to finish
}
```

Uses `mach_absolute_time()` for nanosecond-precision timing.

### Checkpoint

Binary checkpoint format for training state persistence across exec() restarts:

```
[CheckpointHeader: 96 bytes]
[LayerWeights × nLayers]      // Per-type, per-layer: Wq, Wk, ... for all layers
[Adam state × nLayers]        // Per-type, per-layer: m and v buffers
[Global state]                // Cumulative timings, step count, Adam timestep
```

---

## 12. The Forward Pass in Detail

**File:** `Sources/Espresso/ForwardPass.swift`

The training forward pass processes one layer at a time, alternating between ANE kernel execution and CPU operations.

### Per-Layer Flow

```
Input: xCur [768, 256] (FP32, in TensorBuffer)

Step 1: Save layer input
    memcpy(acts[L].layerIn, xCur)
    // Needed for RMSNorm backward: dx requires original input

Step 2: Attention forward (ANE)
    2a. Write xCur to fwdAttn input surface
        SurfaceIO.writeFP16(to: attnIn, data: xCur, channels: 768, spatial: 256)
        // FP32 → FP16 conversion, IOSurface lock/unlock

    2b. Execute attention kernel
        kernels[L].fwdAttn.eval()
        // ANE: RMSNorm → Q,K,V projections → reshape → transpose →
        //       Q@K^T → scale → mask → softmax → scores@V →
        //       transpose → reshape → Wo projection → concat

    2c. Read attention outputs
        SurfaceIO.readFP16Batched(from: attnOut, spatial: 256, regions: [
            (acts[L].oOut,    channelOffset: 0,      channels: 768),  // oo
            (acts[L].attnOut, channelOffset: 4*768,  channels: 768),  // af
            (acts[L].xnorm,   channelOffset: 5*768,  channels: 768),  // xn
        ])
        // FP16 → FP32 conversion, single lock for all 3 regions
        // Note: Q/K/V at offsets 1-3×dim are NOT read back

Step 3: Attention residual (CPU)
    vDSP_vadd(xCur, oOut, x2)
    // x2 = xCur + oOut (element-wise addition via Accelerate)

Step 4: FFN forward (ANE)
    4a. Write x2 to fwdFFN input surface
        SurfaceIO.writeFP16(to: ffnIn, data: x2, channels: 768, spatial: 256)

    4b. Execute FFN kernel
        kernels[L].fwdFFN.eval()
        // ANE: RMSNorm → W1,W3 projections → sigmoid → SiLU gate →
        //       W2 down-projection → concat

    4c. Read FFN outputs
        SurfaceIO.readFP16Batched(from: ffnOut, spatial: 256, regions: [
            (acts[L].ffnOut,  channelOffset: 0,                channels: 768),
            (acts[L].h1,      channelOffset: 768,              channels: 2048),
            (acts[L].h3,      channelOffset: 768+2048,         channels: 2048),
            (acts[L].siluOut, channelOffset: 768+2*2048,       channels: 2048),
            (acts[L].x2norm,  channelOffset: 768+3*2048,       channels: 768),
        ])
        // All 5 regions read under single lock

Step 5: FFN residual (CPU)
    vDSP_vadd(x2, ffnOut, xCur)
    // xCur = x2 + ffnOut

Output: xCur [768, 256] (updated in-place, ready for next layer)
```

### Data Volume Per Layer

| Operation | Direction | Data Size | Lock Pairs |
|-----------|-----------|-----------|------------|
| Write attn input | CPU → ANE | 768×256×2 = 384 KB (FP16) | 1 |
| Read attn output | ANE → CPU | 3×768×256×4 = 2.25 MB (FP32, 3 regions) | 1 |
| Write FFN input | CPU → ANE | 768×256×2 = 384 KB (FP16) | 1 |
| Read FFN output | ANE → CPU | (768+3×2048+768)×256×4 = 7.5 MB (FP32, 5 regions) | 1 |
| **Total I/O per layer** | | **~10.5 MB transferred** | **4 lock pairs** |

The total ANE output surface written is much larger than what the CPU reads back:
- Attention output surface: 4608×256×2 = 2.25 MB (FP16), but CPU reads only 3×768 = 2304 channels
- FFN output surface: 7680×256×2 = 3.75 MB (FP16), CPU reads all 7680 channels

### Surface Handle Caching

`LayerSurfaceHandles` caches the 12 IOSurfaceRef handles (2 per kernel × 5 kernels + 2 for static kernel) to avoid repeated `inputSurface(at:)` / `outputSurface(at:)` calls:

```swift
public struct LayerSurfaceHandles {
    public let fwdAttnIn, fwdAttnOut: IOSurfaceRef
    public let fwdFFNIn, fwdFFNOut: IOSurfaceRef
    public let ffnBwdIn, ffnBwdOut: IOSurfaceRef
    public let sdpaBwd1In, sdpaBwd1Out: IOSurfaceRef
    public let qkvBwdIn, qkvBwdOut: IOSurfaceRef
    public let sdpaBwd2In, sdpaBwd2Out: IOSurfaceRef
}
```

---

## 13. The Backward Pass in Detail

**File:** `Sources/Espresso/BackwardPass.swift`

The backward pass processes layers in reverse order, computing input gradients (`dx`) on the ANE and weight gradients (`dW`) on the CPU.

### Per-Layer Flow (Reverse)

```
1. FFN backward (ANE: ffnBwd kernel)
   Input: [dy, h1, h3, gate]  →  Output: [dx_ffn, dW1, dW3, dW2_intermediate]

2. FFN RMSNorm backward (CPU)
   Input: dx_ffn, x2norm, x2  →  Output: dx_rms2

3. Residual gradient: dy_attn = dx_rms2 + dx_ffn

4. SDPA backward part 1 (ANE: sdpaBwd1 kernel)
   Input: [Q, K, V, dy_attn]  →  Output: [dV, probs, dprobs]

5. SDPA backward part 2 (ANE: sdpaBwd2 kernel — static, no weights)
   Input: [dV, probs, dprobs]  →  Output: [dQ, dK]

6. QKV backward (ANE: qkvBwd kernel)
   Input: [dQ, dK, dV]  →  Output: [dx_qkv]

7. Attention RMSNorm backward (CPU)
   Input: dx_qkv, xnorm, layerIn  →  Output: dx_rms1

8. Residual gradient: dy_prev = dx_rms1 + dy_attn

9. dW accumulation (CPU, async on GradientAccumulator)
   cblas_sgemm for each of Wq, Wk, Wv, Wo, W1, W2, W3, rms1, rms2
```

The backward pass uses 15 IOSurface copy operations per layer to shuttle intermediate gradients between kernels via surface-to-surface copies (avoiding CPU round-trips where possible).

---

## 14. IOSurface Memory Model

IOSurface is a macOS/iOS kernel primitive for sharing memory between processes and hardware accelerators (GPU, ANE, display pipeline). In Espresso:

```
                    ┌──────────────────────────┐
                    │      Unified Memory      │
                    │   (Apple Silicon LPDDR)   │
                    └──────────┬───────────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
         ┌────┴────┐    ┌─────┴─────┐    ┌─────┴─────┐
         │   CPU   │    │    ANE    │    │    GPU    │
         │ (FP32)  │    │  (FP16)  │    │ (unused)  │
         └────┬────┘    └─────┬─────┘    └───────────┘
              │               │
              │  IOSurface    │
              │  Lock/Unlock  │
              │               │
         ┌────┴───────────────┴────┐
         │     IOSurface Buffer    │
         │  Layout: [1, C, 1, S]   │
         │  Format: FP16           │
         │  Size: C × S × 2 bytes  │
         └─────────────────────────┘
```

### Channel-First Layout

ANE natively operates on tensors in `[Batch, Channels, Height, Width]` format. Espresso uses `[1, C, 1, S]` where:
- Batch = 1 (always)
- Channels = embedding dimension or hidden dimension
- Height = 1 (unused)
- Width = sequence length

This means the surface is a flat array of `C × S` FP16 values, with channels as the outer dimension:

```
Memory layout: ch0_s0, ch0_s1, ..., ch0_s255, ch1_s0, ch1_s1, ..., ch767_s255
```

Channel offset arithmetic: to read channels `[ch_off, ch_off + count)`, access elements at byte offset `ch_off × spatial × sizeof(FP16)`.

### Lock Semantics

- **Write lock** (`IOSurfaceLock(surface, 0, NULL)`) — exclusive access, CPU can read and write
- **Read lock** (`IOSurfaceLock(surface, kIOSurfaceLockReadOnly, NULL)`) — shared access, CPU can only read
- **ANE access** — the ANE accesses surfaces without explicit locking during `eval()`. The CPU must not hold any locks during ANE execution.

---

## 15. FP16 Conversion Pipeline

Every piece of data that crosses the CPU↔ANE boundary undergoes precision conversion:

```
CPU (FP32)                    ANE (FP16)
─────────                     ─────────
  weights ──→ WeightBlob.build() ──→ [64-byte header + FP16 payload]
                                      (baked into compiled kernel)

  input   ──→ writeFP16() ─────────→ IOSurface (FP16)
              (NEON vcvt_f16_f32)      ↓
                                    eval() (ANE executes in FP16)
                                       ↓
  output  ←── readFP16() ←──────── IOSurface (FP16)
              (NEON vcvt_f32_f16)
```

### Precision Impact

FP16 has:
- 1 sign bit, 5 exponent bits, 10 mantissa bits
- Range: ±65504
- Precision: ~3.3 decimal digits (vs FP32's ~7.2 digits)

The training forward pass accumulates FP16 rounding errors through:
1. Input conversion (FP32 → FP16)
2. All intermediate computations on ANE (FP16)
3. Output conversion (FP16 → FP32)

Despite this, the system achieves bit-identical parity with the ObjC reference because the ObjC reference uses the same FP16 ANE pipeline.

---

## 16. Compile Budget and exec() Restart

The ANE compiler leaks internal resources (likely ANE program slots or compilation context). After approximately 100 compilations, further attempts will either fail or crash the process.

### Training Loop Compile Pattern

```
Batch 0:  Compile 5 kernels × 12 layers = 60 compiles (+ 12 static = 72 total)
Step 1:   Recompile 5 × 12 = 60 (weights changed via Adam)
          Total: 132 → EXCEEDS BUDGET

Solution: Checkpoint → exec() restart → Resume from checkpoint
```

In practice, the training loop:
1. Runs forward + backward + optimizer for some steps
2. Monitors `CompileBudget.remaining`
3. When budget is low, saves checkpoint via `Checkpoint.save()`
4. Calls `ExecRestart.restart()` which does `execv()` with `--resume` flag
5. The new process loads the checkpoint and continues training

### ExecRestart

```swift
public enum ExecRestart {
    public static func restart(checkpointPath: String, extraArgs: [String] = []) -> Never {
        // Resolve executable path via dyld
        // Preserve original CLI args
        // Add --resume <path> exactly once
        // execv() — replaces current process image
    }
}
```

The `FD_CLOEXEC` flag is set on the dataset file descriptor to prevent fd leaks across the exec boundary.

---

## 17. Performance Baseline

### Inference Benchmark (M3 Max, 1-layer, dim=768, seq=256)

```
=== ANE Direct (Training Forward) ===
Median:  1.106 ms
Mean:    1.124 ms

Time Breakdown:
  ANE kernel:    0.852 ms (76.8%)
  Surface I/O:   0.221 ms (20.0%)
  CPU element:   0.036 ms (3.2%)

Compilation: 679 ms (5 kernels)
Throughput: 904 forward passes/sec
ANE Utilization: 19.2% of 18.0 TFLOPS peak
```

### Training Step Benchmark (M3 Max, 12-layer Stories110M)

```
Swift:  131.2 ms/step
ObjC:   145.5 ms/step
Ratio:  0.90 (Swift 10% faster)
Grade:  S+ (100.00)

Step Breakdown (Swift):
  ANE kernel:     46.4 ms (35.4%)
  Surface I/O:    32.4 ms (24.7%)
  Classwise:      10.2 ms (7.8%)
  Element-wise:   20.9 ms (15.9%)
  RMSNorm:         0.1 ms (0.1%)
  CBLAS wait:      0.0 ms (0.0%)
```

### Core ML Comparison (M3 Max, 1-layer inference)

```
Core ML (.cpuAndNeuralEngine): 1.318 ms median
ANE Direct (training forward): 1.106 ms median
ANE Direct is 1.19x faster than Core ML for the same computation
```

### Historical ObjC Performance (M4, 1-layer, dim=768, seq=512)

```
9.3 ms/step (full training step including backward + optimizer)
11.2% ANE utilization (1.78 TFLOPS sustained)
```

---

## 18. Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **~Copyable everywhere** | Prevents accidental copies of multi-MB buffers. A single `LayerWeights` is 31 MB — one accidental copy could double memory usage. |
| **Typed throws (`throws(ANEError)`)** | Eliminates `as?` downcasting at call sites. Callers know exactly what errors are possible. |
| **Borrowing parameters** | `borrowing LayerWeights` passes a reference without copying. Critical for the compilation path that builds weight blobs. |
| **Channel-first layout** | Matches ANE's native `[1,C,1,S]` format. Eliminates transpose overhead that dominated early prototypes (33.5 → 20.3 ms/step). |
| **Forward taps via concat** | ANE kernels output all activations needed by the backward pass in a single eval. Avoids CPU recomputation of Q, K, V, attention scores, etc. |
| **CPU residual connections** | Residual adds (`x + f(x)`) require both the input `x` and the output `f(x)`. In the training path, both are needed separately for backpropagation, so the add must happen on CPU where both are available as FP32. |
| **Async gradient accumulation** | cblas_sgemm for dW runs on a serial DispatchQueue, overlapping with the next layer's ANE forward pass. This hides ~10 ms of CPU GEMM behind ANE execution. |
| **exec() restart** | The only way to reclaim the ANE compiler's leaked resources. Checkpoint-and-restart is the industry standard approach for long-running ANE workloads. |
| **Locale-explicit formatting** | MIL text generation must produce identical output regardless of system locale. `Locale(identifier: "en_US_POSIX")` prevents `0.5` from becoming `0,5` in German locales. |
| **No external dependencies** | Only Apple system frameworks. Reduces build complexity, eliminates version conflicts, and ensures the project builds on any macOS 15+ machine with Swift 6.0+. |

---

## 19. Known Limitations

| Limitation | Impact | Workaround |
|-----------|--------|------------|
| ~119 compile limit per process | Training must restart periodically | exec() restart with checkpoint |
| ANE ignores `attn_mask` in SDPA ops | Cannot use fused SDPA on ANE | Decompose into matmul → mask → softmax → matmul |
| Weight recompilation per batch | ~680 ms overhead per recompile (5 kernels) | Amortized over accumulation steps (10 steps default) |
| FP16 precision on ANE | ~3.3 digits precision | Acceptable for training; validated against ObjC reference |
| No dynamic shapes | Sequence length is baked into kernels | Recompile for different sequence lengths |
| IOSurface lock overhead | ~8 syscalls per layer per forward pass | Batched reads amortize to 4 lock pairs |
| ANE scheduling opaque | No visibility into ANE queue depth or scheduling | Trust QoS=21 (user-initiated) priority |
| Host-dependent eval stability | Identity kernel eval fails on some M3 Max units | Hardware-gated tests skip gracefully |
| Single-process ANE access | Only one process can use ANE at a time | Training must be the sole ANE consumer |
