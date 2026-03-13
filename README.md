<p align="center">
  <img src="banner.svg" alt="Espresso Banner" width="800">
</p>

<p align="center">
  <img src="demo.gif" alt="Espresso Demo" width="800">
</p>

# Espresso

Backpropagation and exact token generation on Apple's Neural Engine via reverse-engineered private APIs.

## Benchmark

| Path | ms/token |
|------|----------|
| Espresso exact two-step ANE decode | 1.08 |
| CoreML `.cpuAndNeuralEngine` | 5.09 |
| **Speedup** | **4.76x** |

Reproduce it yourself:

```bash
RESULTS_DIR=results/$(date +%Y%m%d-%H%M%S) \
REPEATS=5 WARMUP=3 ITERATIONS=20 \
./scripts/reproduce_local_real_artifact_claim.sh
```

## What is this

A Swift 6.2 codebase that talks directly to Apple's Neural Engine through `_ANEClient` and `_ANEInMemoryModel`, bypassing CoreML entirely. It compiles MIL programs straight to ANE silicon and manages IOSurface buffers for KV cache state.

The original research direction was transformer backpropagation on ANE. The current headline is exact multitoken decode: a recurrent-native path that commits two verified tokens per pass with no recompilation between steps.

No external dependencies -- only Apple system frameworks (Foundation, IOSurface, Accelerate, and CoreML for the baseline comparison).

## Quick start

Requirements: Apple Silicon, macOS 15+, Swift 6 toolchain.

```bash
swift build
swift test
```

Hardware-gated tests (requires ANE device):

```bash
ANE_HARDWARE_TESTS=1 swift test --filter "ANERuntimeTests|EspressoTests|CrossValidationTests"
```

Full ObjC cross-validation:

```bash
OBJC_CROSS_VALIDATION=1 ANE_HARDWARE_TESTS=1 swift test --filter CrossValidationTests
```

## Architecture

```
ANEInterop (ObjC/C -- private API bridge)
    +-- ANETypes (value types, tensor descriptors)
    |       +-- MILGenerator (MIL program text generation)
    |       |       +-- ANERuntime (compile/eval, IOSurface I/O)
    |       |               +-- Espresso (transformer layers, training loop)
    |       |                       +-- EspressoTrain (CLI executable)
    |       +-- CPUOps (Accelerate-backed CPU kernels)
    |               +-- Espresso
```

| Target | Language | What it does |
|---|---|---|
| ANEInterop | ObjC/C | `dlopen` bridge to `_ANEClient`, `_ANEInMemoryModel`, IOSurface |
| ANETypes | Swift | `~Copyable` value types: `Tensor`, `Shape`, `TensorDescriptor` |
| MILGenerator | Swift | Builds MIL program text for forward and backward kernels |
| CPUOps | Swift | RMSNorm, softmax, loss, Adam via Accelerate/vDSP |
| ANERuntime | Swift | Compiles MIL to ANE programs, manages IOSurface buffers |
| Espresso | Swift | Transformer blocks, attention, FFN, decode paths, training |
| EspressoTrain | Swift | CLI for dataset generation, artifact export, training |

Conventions: Swift 6.2 strict concurrency across all targets. `~Copyable` move-only types for tensors to prevent accidental copies of large buffers. Typed throws. Channel-first `[1, C, 1, S]` tensor layout matching ANE's native IOSurface format.

## How the benchmark works

The decode program compiles once and gets reused across all steps -- no per-token recompilation. KV cache state lives in IOSurface buffers managed by the runtime, not marshaled through CoreML's prediction API. Each recurrent step takes the previous hidden state, produces the next, and commits two exact tokens with verified parity. The CoreML baseline uses the same 6-layer model shape with `.cpuAndNeuralEngine` for a matched comparison.

## Disclaimer

> **App Store**: Apps that use private ANE APIs (`_ANEClient`, `_ANEInMemoryModel`, etc.) **will be rejected** during App Store review. Apple explicitly prohibits the use of private/undocumented APIs in App Store submissions.
>
> **Outside the App Store**: Usage is perfectly fine for apps distributed outside the App Store -- including internal tools, research software, sideloaded apps, enterprise distribution, and open-source projects. There are no restrictions on using private APIs in these contexts.

## Caveats

This project uses undocumented private Apple APIs (`_ANEClient`, `_ANEInMemoryModel`) discovered through runtime introspection for research and educational purposes. Results are hardware- and OS-sensitive. The benchmark runs on a local artifact family built by this repo, not a pretrained production model, and the 4.76x speedup is specific to this exact decode contract. This project is not affiliated with or endorsed by Apple Inc.

## Further reading

- [Local repro harness](scripts/reproduce_local_real_artifact_claim.sh) -- end-to-end exact-claim workflow
- [Exact benchmark script](scripts/reproduce_exact_4x.sh) -- matched ANE vs CoreML measurement entry point
- [Benchmark artifacts](artifacts/benchmarks/exact-decode-non-echo/) -- machine-readable evidence

## License

MIT -- see [LICENSE](LICENSE).
