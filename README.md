<p align="center">
  <img src=".github/assets/banner.svg" alt="Espresso" width="800">
</p>

<p align="center">
  <strong>Direct Neural Engine inference for transformers on Apple Silicon — 4.76x faster than CoreML.</strong>
</p>

<p align="center">
  <a href="https://github.com/christopherkarani/Espresso/actions/workflows/ci.yml"><img src="https://github.com/christopherkarani/Espresso/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://github.com/christopherkarani/Espresso/actions/workflows/phase8-matrix.yml"><img src="https://github.com/christopherkarani/Espresso/actions/workflows/phase8-matrix.yml/badge.svg" alt="ANE Matrix"></a>
  <a href="https://swift.org"><img src="https://img.shields.io/badge/Swift-6.2-orange.svg" alt="Swift 6.2"></a>
  <a href="https://github.com/christopherkarani/Espresso/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT"></a>
  <img src="https://img.shields.io/badge/macOS-15+-lightgrey.svg" alt="macOS 15+">
  <img src="https://img.shields.io/badge/Dependencies-0-brightgreen.svg" alt="Zero Dependencies">
  <a href="https://github.com/christopherkarani/Espresso/releases"><img src="https://img.shields.io/github/v/release/christopherkarani/Espresso?color=purple" alt="Latest Release"></a>
</p>

---

Espresso compiles MIL programs straight to ANE silicon through reverse-engineered private APIs (`_ANEClient`, `_ANEInMemoryModel`). No CoreML. No per-token recompilation. Just IOSurface buffers, fused multi-layer kernels, and two verified tokens per decode step.

- **4.76x faster decode** — 1.08 ms/token vs CoreML's 5.09 ms/token on the same 6-layer model
- **Fused 3-layer kernels** — 6 transformer layers in 2 ANE dispatches, not 6
- **Zero-copy I/O** — NEON-vectorized reads, vDSP argmax, no marshaling
- **Full training on ANE** — forward + backward passes with gradient accumulation and Adam
- **Pure Swift 6.2** — `~Copyable` move-only tensors, strict concurrency, typed throws, zero dependencies

<p align="center">
  <img src=".github/assets/demo.gif" alt="Espresso generating tokens on ANE" width="700">
</p>

## Quick Start

```bash
git clone https://github.com/christopherkarani/Espresso.git
cd Espresso
./espresso          # builds, downloads demo weights, launches TUI
```

Five lines to first ANE inference in your own project:

```swift
// Package.swift — add the dependency
.package(url: "https://github.com/christopherkarani/Espresso.git", from: "1.0.0")

import ANERuntime

let kernel = try ANEKernel(milText: myMIL, weights: blobs, inputSizes: [input], outputSizes: [output])
try kernel.eval()                          // runs on Neural Engine
let result = kernel.outputSurface(at: 0)  // zero-copy read
```

Other entry points:

```bash
./espresso "Hello"                          # generate text
./espresso doctor                           # check host readiness
./espresso compare --no-power "Hello"       # side-by-side vs CoreML
./espresso install                          # install to ~/.local/bin
swift run espresso-bench --ane-only --inference --layers 6
```

## Benchmark

### Espresso vs CoreML vs llama.cpp

| Backend | ms/token | tok/s | Notes |
|---------|----------|-------|-------|
| **Espresso ANE** (exact two-step) | **1.08** | **926** | Direct ANE, 2 dispatches / 6 layers |
| CoreML `.cpuAndNeuralEngine` | 5.09 | 196 | Apple's standard ANE path |
| llama.cpp Metal | ~12–20 | ~50–85 | GPU path, CPU-bound decode¹ |
| llama.cpp CPU (`ggml`) | ~25–40 | ~25–40 | Pure CPU, no ANE¹ |
| **Espresso speedup vs CoreML** | | **4.76x** | |
| **Espresso speedup vs llama.cpp Metal** | | **~11x** | |

> ¹ llama.cpp has no ANE backend. Metal figures are representative for GPT-2 117M on M3 Max; actual performance varies by quantization and prompt length.
> All Espresso / CoreML numbers: 6-layer local artifact · dim=768 · 12 heads · 32k vocab · seqLen=256 · M3 Max · macOS 15.

<details>
<summary>Reproduce Espresso benchmarks</summary>

```bash
RESULTS_DIR=results/$(date +%Y%m%d-%H%M%S) \
REPEATS=5 WARMUP=3 ITERATIONS=20 \
./scripts/reproduce_local_real_artifact_claim.sh
```

Machine-readable output lands in `artifacts/benchmarks/` and is kept out of git.

</details>

### Platform Compatibility

| SoC | Neural Engine | Tested | Notes |
|-----|---------------|--------|-------|
| M1 / M1 Pro / M1 Max / M1 Ultra | 16-core ANE | ✅ | Full feature set |
| M2 / M2 Pro / M2 Max / M2 Ultra | 16-core ANE | ✅ | Full feature set |
| M3 / M3 Pro / M3 Max | 18-core ANE | ✅ | Reference hardware (M3 Max) |
| M4 / M4 Pro / M4 Max | 38-core ANE | ✅ | Faster compile cache warm-up |
| Intel Mac | — | ❌ | No Neural Engine |
| Apple A-series (iOS) | ✅ | ⚠️ | Requires entitlement; not App Store safe |

macOS 15+ required. iOS / tvOS not supported out of the box (private API entitlements differ per platform).

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
                    │    ANE E5 Binary     │  Cached by system
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

The decode loop compiles once and reuses the program across all steps. KV cache lives in IOSurface buffers — not marshaled through CoreML. Each step produces two exact tokens with verified parity. Fused triplet kernels process 3 layers per dispatch, reducing 6 layers to 2 eval calls.

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

| Module | What it does |
|--------|-------------|
| **ANEInterop** | `dlopen` bridge to `_ANEClient` and `_ANEInMemoryModel`. NEON-vectorized I/O. |
| **ANETypes** | `~Copyable` tensors, `SurfaceIO`, weight serialization, model config. |
| **MILGenerator** | Generates MIL text for forward, backward, decode, and fused kernels. |
| **CPUOps** | RMSNorm, RoPE, embedding, softmax, Adam via Accelerate/vDSP. |
| **ANERuntime** | Compiles MIL to ANE E5 binaries. Manages IOSurface buffers and compile budget. |
| **Espresso** | Transformer layers, generation harnesses, exact two-token decode, training loop. |

## SPM Integration

```swift
// Package.swift
dependencies: [
    .package(url: "https://github.com/christopherkarani/Espresso.git", from: "1.0.0")
],
targets: [
    .target(name: "MyApp", dependencies: [
        .product(name: "ANERuntime", package: "Espresso"),
        .product(name: "ANETypes",   package: "Espresso"),
    ])
]
```

```swift
import ANERuntime
import ANETypes

// 1. Define your kernel shape
let gen = MyMILGenerator(config: .init(dim: 768, heads: 12))

// 2. Compile once to ANE E5 binary
let kernel = try ANEKernel(
    milText: gen.milText,
    weights: gen.weightBlobs,
    inputSizes: [gen.inputSize],
    outputSizes: [gen.outputSize]
)

// 3. Run inference — stays on ANE the whole time
try kernel.eval()

// 4. Read results via zero-copy IOSurface
let output = kernel.outputSurface(at: 0)
```

## Requirements

| | Minimum |
|---|---|
| Hardware | Apple Silicon (M1+) with Neural Engine |
| macOS | 15.0+ |
| Swift | 6.0+ |
| Dependencies | None — only Apple system frameworks |

## Testing

```bash
swift test                                                    # unit tests (no ANE needed)
ANE_HARDWARE_TESTS=1 swift test --filter "ANERuntimeTests|EspressoTests"  # hardware tests
OBJC_CROSS_VALIDATION=1 ANE_HARDWARE_TESTS=1 swift test --filter CrossValidationTests  # parity
```

7 test suites cover MIL generation, tensor ops, CPU kernels, ANE compilation, hardware eval, cross-validation, and end-to-end generation.

## Disclaimer

> **App Store**: Apps using private ANE APIs (`_ANEClient`, `_ANEInMemoryModel`) will be rejected.
>
> **Everywhere else**: Internal tools, research, sideloaded apps, enterprise distribution — all fine.

This project uses undocumented private Apple APIs discovered through runtime introspection. Results are hardware- and OS-dependent. Benchmarks run on a local artifact family built by this repo, not a pretrained production model. Not affiliated with or endorsed by Apple Inc.

## Contributing

Contributions welcome — see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.
File bugs and feature requests via [GitHub Issues](https://github.com/christopherkarani/Espresso/issues).

## License

MIT — see [LICENSE](LICENSE).
