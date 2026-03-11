# Espresso

**Direct ANE decode research in Swift. Reproducible exact non-echo local-artifact benchmark: 4.76x faster than matched Core ML.**

Espresso is a Swift 6.2 research codebase with an Objective-C/C bridge for running custom training and exact decode experiments directly on Apple's Neural Engine through reverse-engineered private APIs such as `_ANEClient` and `_ANEInMemoryModel`.

## Current Public Release

The strongest checked-in public claim in this branch is a reproducible exact non-echo decode benchmark on a local artifact family:

- Exact two-step ANE decode: `1.0806302083333332 ms/token`
- Matched one-token ANE control: `1.0957500000000002 ms/token`
- Matched CoreML `.cpuAndNeuralEngine`: `5.085307291666668 ms/token`
- Exact two-step speedup vs CoreML: `4.7583224488025415x`
- Exactness: parity `match`, `committed exact tokens/pass = 2`, `accepted future tokens/pass = 1`

Primary evidence:

- Release note: [docs/releases/2026-03-11-non-echo-exact-decode.md](docs/releases/2026-03-11-non-echo-exact-decode.md)
- Human-readable artifact: [artifacts/benchmarks/exact-decode-non-echo/latest.md](artifacts/benchmarks/exact-decode-non-echo/latest.md)
- Machine-readable artifact: [artifacts/benchmarks/exact-decode-non-echo/latest.json](artifacts/benchmarks/exact-decode-non-echo/latest.json)
- Full lab notebook: [docs/fused-decode-and-next-steps.md](docs/fused-decode-and-next-steps.md)

What this claim is:

- A matched same-session ANE/CoreML decode benchmark for this reproducible non-echo local-artifact setup
- An exact recurrent-native two-token decode path with no approximate commits
- A non-echo local artifact family built entirely by this repo

What this claim is not:

- Not a pretrained production checkpoint result
- Not a blanket claim about all CoreML workloads
- Not a claim that the generic recurrent ANE path is fixed for every off-echo artifact family

## Quick Start

Requirements:

- Apple Silicon Mac
- macOS 15+
- Swift 6 toolchain
- Python 3.12 if you want to regenerate CoreML benchmark models locally

Build and baseline test:

```bash
swift build
swift test
```

Hardware-gated validation:

```bash
ANE_HARDWARE_TESTS=1 swift test --filter "ANERuntimeTests|EspressoTests|CrossValidationTests"
OBJC_CROSS_VALIDATION=1 ANE_HARDWARE_TESTS=1 swift test --filter CrossValidationTests
```

One-command public repro:

```bash
RESULTS_DIR=results/non-echo-public-$(date +%Y%m%d-%H%M%S) \
REPEATS=5 WARMUP=3 ITERATIONS=20 \
./scripts/reproduce_local_real_artifact_claim.sh
```

That script will:

1. Build a local text-token dataset from this repository.
2. Export the matching recurrent artifact family.
3. Generate a matching zero-weight 6-layer CoreML trunk.
4. Run the matched ANE/CoreML public harness.

It does not require external model weights, but a first run may bootstrap local Python tooling for `coremltools` if the benchmark export environment is missing.

## Why This Benchmark Wins

In this benchmark, Espresso benefits from a much narrower contract than a general-purpose CoreML entrypoint:

- The exact decode program is compiled once and then reused across repeated decode steps.
- KV/cache state stays in an IOSurface-driven path designed around this artifact family.
- The public claim is measured on an exact two-step decode path, not on a generic checkpoint loader.

That is evidence for this benchmark and this decode contract. It is not evidence that Espresso is faster than CoreML for every model or workload.

## What The Repo Contains

- `Sources/ANEInterop`: Objective-C/C bridge to private ANE APIs and IOSurface helpers
- `Sources/MILGenerator`: Swift MIL text generators for ANE kernels
- `Sources/ANERuntime`: Swift kernel compilation and eval wrappers
- `Sources/Espresso`: training loops, decode paths, recurrent generation harnesses, artifact builders
- `Sources/EspressoTrain`: CLI for dataset generation, artifact export, and training utilities
- `Sources/EspressoMultiTokenProbe`: CLI used by the public decode repro scripts
- `docs/`: release notes, probe writeups, architecture notes, and the running lab notebook
- `artifacts/benchmarks/`: checked-in benchmark evidence for public claims

## Project Scope

Espresso has two main lines of work:

- Transformer training research on ANE using direct private-API compilation and eval
- Exact recurrent-native decode experiments designed to beat matched CoreML baselines

The repository still contains the earlier training work, but the current public launch story is the exact non-echo decode release above.

## Caveats

- This project uses undocumented private Apple APIs and is for research use, not App Store distribution.
- Results are hardware-, OS-, and runtime-sensitive.
- Some experimental paths in the repo use Metal, but the headline public decode claim is an ANE-direct path.
- The public decode claim depends on the explicit `identity-zero-trunk` backend documented in the release note.

## Additional Reading

- [docs/releases/2026-03-11-non-echo-exact-decode.md](docs/releases/2026-03-11-non-echo-exact-decode.md)
- [docs/fused-decode-and-next-steps.md](docs/fused-decode-and-next-steps.md)
- [docs/vc-probe-results.md](docs/vc-probe-results.md)

## Disclaimer

This project is independent research into Apple Neural Engine architecture. It uses undocumented APIs discovered through runtime introspection for research and educational purposes under fair use and interoperability provisions. No Apple proprietary code or binaries are included in this repository. This project is not affiliated with or endorsed by Apple Inc. Use at your own risk.

## License

MIT — see [LICENSE](LICENSE)
