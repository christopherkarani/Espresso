# Espresso — ANE Training in Swift

Backpropagation on Apple Neural Engine via reverse-engineered private APIs. Swift 6.2 rewrite of the original Objective-C implementation.

## Build & Test

```bash
swift build                # Build all targets
swift test                 # Run all tests (~156 tests)
swift test --filter ANETypesTests      # Run a single test target
swift test --filter CrossValidationTests  # Hardware-gated (requires ANE)
```

Requires: macOS 15+, Apple Silicon, Swift 6.0+ toolchain.

**Hardware-gated tests**: `ANERuntimeTests` and `CrossValidationTests` require an actual ANE device. They will fail on CI or Intel Macs. All other test targets run on any macOS machine.

## Architecture — Target Dependency Graph

```
ANEInterop (ObjC/C — private API bridge)
    ├── ANETypes (value types, tensor descriptors)
    │       ├── MILGenerator (MIL program text generation)
    │       │       └── ANERuntime (compile/eval, IOSurface I/O)
    │       │               └── Espresso (transformer layers, training loop)
    │       │                       └── EspressoTrain (CLI executable)
    │       └── CPUOps (Accelerate-backed CPU kernels)
    │               └── Espresso
```

### Targets

| Target | Language | Role |
|---|---|---|
| `ANEInterop` | ObjC/C | `dlopen`-based bridge to `_ANEClient`, `_ANEInMemoryModel`, IOSurface |
| `ANETypes` | Swift | `~Copyable` value types: `Tensor`, `Shape`, `TensorDescriptor` |
| `MILGenerator` | Swift | Builds MIL program text for forward/backward kernels |
| `CPUOps` | Swift | RMSNorm, softmax, loss, Adam — all via Accelerate/vDSP |
| `ANERuntime` | Swift | Compiles MIL → ANE program, manages IOSurface buffers |
| `Espresso` | Swift | Transformer block, attention, FFN, full training orchestration |
| `EspressoTrain` | Swift | CLI entry point for training runs |

## Key Conventions

- **Swift 6.2 strict concurrency** — all targets use `.swiftLanguageMode(.v6)`
- **`~Copyable` value types** — `Tensor`, `Shape` are move-only to prevent accidental copies of large buffers
- **Typed throws** — functions throw specific error types, not generic `Error`
- **Locale-safe formatting** — all numeric formatting uses explicit locale to avoid test flakiness
- **Channel-first layout** — tensors use `[1, C, 1, S]` (ANE IOSurface native format)
- **No external dependencies** — only Apple system frameworks (Foundation, CoreML, IOSurface, Accelerate)

## Test Oracle Data

`training/golden_outputs/` contains binary reference tensors captured from the original ObjC implementation. Cross-validation tests compare Swift outputs against these to verify numerical equivalence.

## Scripts

- `scripts/cross_validate.sh` — Run cross-validation between ObjC reference and Swift
- `scripts/capture_phase6b_goldens.sh` — Capture new golden reference data
- `scripts/generate_golden_mil.sh` — Generate golden MIL programs for comparison
