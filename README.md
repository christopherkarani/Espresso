<p align="center">
  <img src=".github/assets/banner.svg" alt="Espresso" width="800">
</p>

<p align="center">
  <strong>Direct Neural Engine inference for transformers on Apple Silicon.</strong>
</p>

<p align="center">
  <a href="https://github.com/christopherkarani/Espresso/actions/workflows/ci.yml"><img src="https://github.com/christopherkarani/Espresso/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://github.com/christopherkarani/Espresso/actions/workflows/phase8-matrix.yml"><img src="https://github.com/christopherkarani/Espresso/actions/workflows/phase8-matrix.yml/badge.svg" alt="ANE Matrix"></a>
  <a href="https://swift.org"><img src="https://img.shields.io/badge/Swift-6.2-orange.svg" alt="Swift 6.2"></a>
  <a href="https://github.com/christopherkarani/Espresso/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT"></a>
  <img src="https://img.shields.io/badge/macOS-15+-lightgrey.svg" alt="macOS 15+">
  <a href="https://github.com/christopherkarani/Espresso/releases"><img src="https://img.shields.io/github/v/release/christopherkarani/Espresso?color=purple" alt="Latest Release"></a>
</p>

---

Espresso compiles MIL programs straight to ANE silicon through reverse-engineered private APIs (`_ANEClient`, `_ANEInMemoryModel`). No CoreML in the hot path. No per-token recompilation. IOSurface buffers, fused multi-layer kernels, and hybrid decode for Llama-family artifacts.

- **Direct ANE path** — private-API compile once, reuse across decode steps
- **Fused multi-layer kernels** — fewer ANE dispatches per token
- **Zero-copy I/O** — NEON-vectorized surface reads, vDSP/Metal where useful
- **Training experiments on ANE** — forward + backward with gradient accumulation and Adam
- **Pure Swift 6.2 core** — `~Copyable` move-only tensors, strict concurrency, typed throws

<p align="center">
  <img src=".github/assets/demo.gif" alt="Espresso generating tokens on ANE" width="700">
</p>

## Quick Start

```bash
git clone https://github.com/christopherkarani/Espresso.git
cd Espresso
./espresso          # builds, downloads demo weights, launches TUI
```

Core ANE kernel usage (library surface):

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
swift run espc pack-native /path/to/model /tmp/model.esp --overwrite
swift run esprun inspect /tmp/model.esp
swift run esprun generate /tmp/model.esp "Hello" 32
```

## ESP Model Platform

Espresso ships a model platform around portable `.esp` bundles and bundle-aware runtime selection.

- `.esp` is the canonical portable model bundle
- `.espc` is the derived compiled-cache layer
- `espc` packs native model directories into `.esp`
- `esprun` inspects, resolves, and runs bundle artifacts
- `espresso-generate --bundle <path>` runs the same bundle boundary used by the runtime

Current docs for this layer:

- [Convert / Optimize / Native-Fast strategy](docs/platform/2026-03-26-convert-optimize-native-fast-plan.md)
- [Stories Convert -> Optimize execution plan](docs/platform/2026-03-26-stories-convert-optimize-execution-plan.md)
- [Stories agent prompt](docs/platform/2026-03-26-stories-convert-optimize-agent-prompt.md)

## Benchmark

Numbers below match the checked-in machine-readable results in
[`benchmarks/results/latest.json`](benchmarks/results/latest.json)
(M3 Max, macOS 15.0, Espresso 1.1.0).

### Espresso vs CoreML (local 6-layer Stories artifact)

| Backend | ms/token | tok/s | Notes |
|---------|----------|-------|-------|
| **Espresso ANE** (recurrent fused, 6-layer) | **1.93** | **519** | Fused 3-layer recurrent decode + ANE classifier |
| Espresso ANE (direct transformer, 6-layer) | 6.56 | 153 | Same model without recurrent fusion |
| CoreML `.cpuAndNeuralEngine` | 6.58 | 152 | Apple's standard ANE path |
| **Espresso speedup vs CoreML** | | **3.41×** | fused recurrent path |

> All Espresso / CoreML numbers: 6-layer local artifact · dim=768 · 12 heads · 32k vocab · seqLen=256 · M3 Max · macOS 15.
> This is a **research / demo artifact family**, not a pretrained production model.

### What these numbers are *not*

- Not a claim about full GPT-2 117M or llama.cpp Metal on the same workload
- Not the only serving path: the retained exact Stories hybrid path (`.esp` bundles)
  typically lands lower in wall-clock tok/s after compile and full decode accounting —
  see the experiment ledger in `tasks/todo.md` for same-binary retain/reject numbers
- Peak research configurations can exceed the table when measuring a narrower path;
  only figures with a checked-in `latest.json` (or a PR artifact from the reproduce script)
  are treated as project claims

<details>
<summary>Reproduce Espresso benchmarks</summary>

```bash
RESULTS_DIR=results/$(date +%Y%m%d-%H%M%S) \
REPEATS=5 WARMUP=3 ITERATIONS=20 \
./scripts/reproduce_local_real_artifact_claim.sh
```

Machine-readable output lands in `artifacts/benchmarks/` and is kept out of git.
Update `benchmarks/results/latest.json` only when you intentionally refresh the public table.

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

The decode loop compiles once and reuses the program across all steps. KV cache lives in IOSurface buffers — not marshaled through CoreML.

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
| **Espresso** | Transformer layers, generation harnesses, decode, training loop. |
| **RealModelInference** | GPT-2 / Llama hybrid serving path used by `espresso-generate`. |
| **ESPBundle / ESPRuntime** | Portable `.esp` bundles and runtime resolution. |

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

// 1. Provide MIL text + weight blobs for your kernel shape
let kernel = try ANEKernel(
    milText: milText,
    weights: weightBlobs,
    inputSizes: [inputByteSize],
    outputSizes: [outputByteSize]
)

// 2. Run inference — stays on ANE the whole time
try kernel.eval()

// 3. Read results via zero-copy IOSurface
let output = try kernel.outputSurface(at: 0)
```

For end-to-end generation, use `RealModelInference` / `esprun` with a prepared model directory or `.esp` bundle rather than hand-writing MIL.

## Dependencies

**Default package graph: zero third-party Swift packages.** Only Apple system frameworks (Foundation, Accelerate, IOSurface, Metal, CoreML).

Optional GGUF / EdgeRunner integration is **not** part of the default graph. It enables only when a local [Edgerunner](https://github.com/christopherkarani) checkout is present:

```bash
# Sibling checkout (default discovery path)
#   ../Edgerunner/Package.swift

# Or explicit path
export ESPRESSO_EDGERUNNER_PATH=/path/to/Edgerunner
swift build

# Force-disable even if Edgerunner is present
export ESPRESSO_DISABLE_GGUF=1
```

## Requirements

| | Minimum |
|---|---|
| Hardware | Apple Silicon (M1+) with Neural Engine |
| macOS | 15.0+ |
| Swift | 6.0+ (6.2 recommended) |
| Dependencies | None required — Apple system frameworks only |

## Testing

```bash
swift test                                                    # unit tests (no ANE needed)
ANE_HARDWARE_TESTS=1 swift test --filter "ANERuntimeTests|EspressoTests"  # hardware tests
OBJC_CROSS_VALIDATION=1 ANE_HARDWARE_TESTS=1 swift test --filter CrossValidationTests  # parity
```

CI runs non-hardware unit tests including ESP bundle/runtime and RealModelInference unit suites. Hardware ANE tests run on self-hosted matrix jobs.

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
