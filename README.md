<p align="center">
  <img src=".github/assets/banner.svg" alt="Espresso" width="800">
</p>

<p align="center">
  <strong>Direct Neural Engine inference for transformers on Apple Silicon — 4.76x faster than CoreML.</strong>
</p>

<p align="center">
  <a href="https://github.com/christopherkarani/Espresso/actions/workflows/phase8-matrix.yml"><img src="https://github.com/christopherkarani/Espresso/actions/workflows/phase8-matrix.yml/badge.svg" alt="Build"></a>
  <a href="https://swift.org"><img src="https://img.shields.io/badge/Swift-6.2-orange.svg" alt="Swift 6.2"></a>
  <a href="https://github.com/christopherkarani/Espresso/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT"></a>
  <img src="https://img.shields.io/badge/macOS-15+-lightgrey.svg" alt="macOS 15+">
  <img src="https://img.shields.io/badge/Dependencies-0-brightgreen.svg" alt="Zero Dependencies">
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

## Try The TUI

From a fresh clone:

```bash
./espresso
```

That one command will:

- build the local Swift targets if needed
- bootstrap the managed GPT-2 demo assets on first run
- launch the side-by-side Espresso vs Core ML TUI

If `./espresso` does not start cleanly, run `./espresso doctor`.

Useful follow-ups:

```bash
./espresso "Hello"
./espresso doctor
./espresso compare --no-power "Hello"
./espresso install
swift run espresso-generate demo
```

`./espresso install` writes a small shim to `~/.local/bin/espresso` that points back to this checkout, so future runs can use `espresso`.

## Benchmark

| Path | ms/token | tok/s |
|------|----------|-------|
| Espresso exact two-step ANE decode | **1.08** | **926** |
| CoreML `.cpuAndNeuralEngine` | 5.09 | 196 |
| **Speedup** | | **4.76x** |

> 6-layer transformer · dim=768 · 12 heads · 32k vocab · seqLen=256 · M3 Max · macOS 15

<details>
<summary>Reproduce it yourself</summary>

```bash
RESULTS_DIR=results/$(date +%Y%m%d-%H%M%S) \
REPEATS=5 WARMUP=3 ITERATIONS=20 \
./scripts/reproduce_local_real_artifact_claim.sh
```

Machine-readable benchmark output is generated locally under `artifacts/benchmarks/` and kept out of git.

</details>

## Quick Start

```bash
./espresso                            # first-run GPT-2 TUI demo
./espresso doctor                     # check host readiness
swift build                                              # build everything
swift test                                               # unit tests (no ANE needed)
swift run espresso-bench --ane-only --inference --layers 6  # benchmark
swift run espresso-bench --decode --decode-steps 32 --layers 6  # decode benchmark
```

Hardware-gated tests (requires ANE):

```bash
ANE_HARDWARE_TESTS=1 swift test --filter "ANERuntimeTests|EspressoTests|CrossValidationTests"
```

### As a dependency

```swift
dependencies: [
    .package(url: "https://github.com/christopherkarani/Espresso.git", branch: "main")
]
```

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

// Read results — zero copy from IOSurface
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

## Requirements

| | Minimum |
|---|---|
| Hardware | Apple Silicon (M1+) with Neural Engine |
| macOS | 15.0+ |
| Swift | 6.0+ |
| Dependencies | None — only Apple system frameworks |

## Testing

```bash
swift test                                                    # unit tests
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

Contributions welcome. File bugs and feature requests via [GitHub Issues](https://github.com/christopherkarani/Espresso/issues).

## License

MIT — see [LICENSE](LICENSE).
