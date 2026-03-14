<p align="center">
  <img src="banner.svg" alt="Espresso" width="800">
</p>

<p align="center">
  <strong>Train and run transformers directly on Apple's Neural Engine — 4.76x faster than CoreML.</strong>
</p>

<p align="center">
  <a href="https://github.com/christopherkarani/Espresso/actions/workflows/phase8-matrix.yml"><img src="https://github.com/christopherkarani/Espresso/actions/workflows/phase8-matrix.yml/badge.svg" alt="Build"></a>
  <a href="https://swift.org"><img src="https://img.shields.io/badge/Swift-6.2-orange.svg" alt="Swift 6.2"></a>
  <a href="https://github.com/christopherkarani/Espresso/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT"></a>
  <img src="https://img.shields.io/badge/Platform-macOS_15+-lightgrey.svg" alt="macOS 15+">
  <img src="https://img.shields.io/badge/Dependencies-0-brightgreen.svg" alt="Zero Dependencies">
</p>

---

Espresso bypasses CoreML and compiles MIL programs straight to ANE silicon through reverse-engineered private APIs (`_ANEClient`, `_ANEInMemoryModel`). It manages IOSurface buffers directly, fuses multiple transformer layers into single ANE dispatches, and commits two verified tokens per decode step — no per-token recompilation, no framework overhead.

- **4.76x faster decode** — 1.08 ms/token vs CoreML's 5.09 ms/token on the same 6-layer model
- **Exact two-token generation** — each step commits two parity-verified tokens, not speculation
- **Fused multi-layer kernels** — 3-layer triplet fusion cuts dispatch overhead to 2 ANE evals for 6 layers
- **Zero-copy I/O** — IOSurface buffers with NEON-vectorized reads, vDSP argmax, no marshaling
- **Full backpropagation on ANE** — forward and backward passes with gradient accumulation and Adam
- **Pure Swift 6.2** — `~Copyable` move-only tensors, strict concurrency, typed throws, zero external dependencies

<p align="center">
  <img src="demo.gif" alt="Espresso generating tokens on ANE" width="700">
</p>

## Benchmark

| Path | ms/token | tok/s |
|------|----------|-------|
| Espresso exact two-step ANE decode | **1.08** | **926** |
| CoreML `.cpuAndNeuralEngine` | 5.09 | 196 |
| **Speedup** | | **4.76x** |

6-layer transformer, dim=768, 12 heads, 32k vocab, seqLen=256. M3 Max, macOS 15.

Reproduce it yourself:

```bash
RESULTS_DIR=results/$(date +%Y%m%d-%H%M%S) \
REPEATS=5 WARMUP=3 ITERATIONS=20 \
./scripts/reproduce_local_real_artifact_claim.sh
```

## Requirements

| | Minimum |
|---|---|
| Hardware | Apple Silicon (M1+) with Neural Engine |
| macOS | 15.0+ |
| Swift | 6.0+ |
| Dependencies | None (only Apple system frameworks) |

## Quick Start

```bash
# Build everything
swift build

# Run unit tests (no ANE hardware needed)
swift test

# Run the benchmark
swift run espresso-bench --ane-only --inference --layers 6

# Run the exact decode benchmark
swift run espresso-bench --decode --decode-steps 32 --layers 6
```

Hardware-gated tests (requires ANE device):

```bash
ANE_HARDWARE_TESTS=1 swift test --filter "ANERuntimeTests|EspressoTests|CrossValidationTests"
```

### As a library

Add Espresso to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/christopherkarani/Espresso.git", branch: "main")
]
```

Then import and use:

```swift
import Espresso
import ANERuntime
import ANETypes

// Compile a kernel directly to ANE
let kernel = try ANEKernel(
    milText: generator.milText,
    weights: weightBlobs,
    inputSizes: [inputSize],
    outputSizes: [outputSize]
)

// Evaluate on the Neural Engine
try kernel.eval()

// Read results from IOSurface — zero copy
let output = kernel.outputSurface(at: 0)
```

## How It Works

```
                    ┌─────────────────────┐
                    │   MIL Program Text   │  Generated per-kernel
                    └──────────┬──────────┘
                               ▼
                    ┌─────────────────────┐
                    │  _ANEClient compile  │  Private API (dlopen)
                    └──────────┬──────────┘
                               ▼
                    ┌─────────────────────┐
                    │   ANE E5 Binary      │  Cached by system
                    └──────────┬──────────┘
                               ▼
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
     ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
     │  IOSurface   │ │  IOSurface   │ │  IOSurface   │
     │   (input)    │ │  (weights)   │ │  (output)    │
     └──────┬───────┘ └──────────────┘ └──────┬───────┘
            │          ANE Hardware            │
            └──────────────eval───────────────┘
```

The decode loop compiles once and reuses the program across all steps. KV cache state lives in IOSurface buffers — not marshaled through CoreML's prediction API. Each recurrent step takes the previous hidden state, produces the next, and commits two exact tokens with verified parity. Fused triplet kernels process 3 layers per ANE dispatch, reducing the 6-layer model to just 2 eval calls per step.

## Architecture

```
ANEInterop (ObjC/C — private API bridge)
    └── ANETypes (~Copyable value types, IOSurface I/O)
            ├── MILGenerator (28+ kernel variants)
            │       └── ANERuntime (compile, eval, surface management)
            │               └── Espresso (training, generation, decode)
            │                       ├── EspressoTrain (CLI)
            │                       └── EspressoBench (CLI)
            └── CPUOps (Accelerate/vDSP kernels)
                    └── Espresso
```

| Module | Language | What it does |
|---|---|---|
| **ANEInterop** | ObjC/C | `dlopen` bridge to `_ANEClient`, `_ANEInMemoryModel`; NEON vectorized I/O |
| **ANETypes** | Swift | `~Copyable` tensors, `SurfaceIO`, `ModelConfig`, weight serialization |
| **MILGenerator** | Swift | Generates MIL text for forward, backward, decode, and fused kernels |
| **CPUOps** | Swift | RMSNorm, RoPE, embedding, softmax, Adam via Accelerate/vDSP |
| **ANERuntime** | Swift | Compiles MIL to ANE programs, manages IOSurface buffers, compile budget |
| **Espresso** | Swift | Transformer layers, generation harnesses, exact two-token decode, training |

**Conventions:** Swift 6.2 strict concurrency. `~Copyable` move-only types prevent accidental copies of large buffers. Typed throws. Channel-first `[1, C, 1, S]` layout matching ANE's native IOSurface format. All weights baked into E5 binaries at compile time.

## Testing

```bash
# Unit tests (no hardware needed)
swift test

# ANE hardware tests
ANE_HARDWARE_TESTS=1 swift test --filter "ANERuntimeTests|EspressoTests"

# Full cross-validation (Swift ↔ ObjC numerical parity)
OBJC_CROSS_VALIDATION=1 ANE_HARDWARE_TESTS=1 swift test --filter CrossValidationTests
```

Seven test suites cover MIL generation, tensor operations, CPU kernels, ANE compilation, hardware evaluation, cross-validation, and end-to-end generation correctness.

## Disclaimer

> **App Store**: Apps using private ANE APIs (`_ANEClient`, `_ANEInMemoryModel`) **will be rejected** during App Store review.
>
> **Outside the App Store**: Perfectly fine for internal tools, research, sideloaded apps, enterprise distribution, and open-source projects.

This project uses undocumented private Apple APIs discovered through runtime introspection for research and educational purposes. Results are hardware- and OS-sensitive. The benchmark runs on a local artifact family built by this repo, not a pretrained production model. This project is not affiliated with or endorsed by Apple Inc.

## Further Reading

- [Reproduce the benchmark](scripts/reproduce_local_real_artifact_claim.sh) — end-to-end exact-claim workflow
- [Benchmark artifacts](artifacts/benchmarks/exact-decode-non-echo/) — machine-readable evidence

## Contributing

Contributions welcome. See [GitHub Issues](https://github.com/christopherkarani/Espresso/issues) for bugs and feature requests.

## License

MIT — see [LICENSE](LICENSE).
