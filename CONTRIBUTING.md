# Contributing to Espresso

Thank you for your interest in contributing to Espresso. This document covers development setup, coding standards, and the PR process.

## Table of Contents

- [Development Setup](#development-setup)
- [Project Structure](#project-structure)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)
- [Issue Guidelines](#issue-guidelines)

## Development Setup

**Requirements**

- macOS 15.0+
- Xcode 16.2+ (Swift 6.2)
- Apple Silicon Mac (M1 or later) — required for ANE hardware tests

**Clone and build**

```bash
git clone https://github.com/christopherkarani/Espresso.git
cd Espresso
swift build
swift test                    # unit tests, no ANE required
```

**Run the demo**

```bash
./espresso          # builds, downloads GPT-2 weights, launches TUI
./espresso doctor   # check host readiness
```

**Hardware tests** (requires Apple Silicon ANE)

```bash
ANE_HARDWARE_TESTS=1 swift test --filter "ANERuntimeTests|EspressoTests|CrossValidationTests"
```

## Project Structure

```
Sources/
  ANEInterop/          # ObjC/C bridge to _ANEClient private API
  ANETypes/            # ~Copyable tensors, SurfaceIO, weight serialization
  MILGenerator/        # MIL text generation (28+ kernel variants)
  CPUOps/              # CPU fallbacks via Accelerate/vDSP
  ANERuntime/          # Compile, eval, IOSurface management
  Espresso/            # Transformer layers, training, generation
  ANEGraphIR/          # Graph IR with optimization passes
  ANECodegen/          # MIL codegen from Graph IR
  ANEPasses/           # Graph optimization passes
  ANEBuilder/          # End-to-end kernel builder
  ModelSupport/        # GPT-2 and Llama model configs
  DeltaCompilation/    # Delta compilation for LoRA adapters
  LoRAAdapter/         # LoRA adapter support
  RealModelInference/  # GPT-2 real model inference engine
  EspressoGenerate/    # Generation CLI target
Tests/                 # Mirror of Sources structure
scripts/               # Benchmark and reproduction scripts
docs/                  # Architecture docs and research logs
artifacts/             # Generated benchmark artifacts (gitignored)
```

## Coding Standards

**Language**: Swift 6.2 with strict concurrency enabled. All new code must compile under `.swiftLanguageMode(.v6)`.

**Key conventions**:
- Use `~Copyable` for move-only resources (kernels, surfaces, weights)
- Immutable value types by default — avoid mutation
- Typed throws where the error set is bounded
- No external dependencies — Apple system frameworks only
- Files ≤400 lines, functions ≤50 lines
- 80%+ test coverage for new code

**MIL programs**: Kernel generators go in `Sources/MILGenerator/`. Follow the naming pattern `*Generator.swift` and output a `milText: String` property. Test with a corresponding `Tests/MILGeneratorTests/` file.

**Private API surface**: Changes to the `_ANEClient`/`_ANEInMemoryModel` bridge in `ANEInterop` require careful documentation — note the macOS version range tested.

## Testing

Run the full test suite before submitting:

```bash
# Unit tests (no hardware required — runs in CI)
swift test --filter "ANETypesTests|MILGeneratorTests|CPUOpsTests|ANEGraphIRTests|ANECodegenTests|ANEPassesTests|ANEBuilderTests|ModelSupportTests|DeltaCompilationTests|LoRAAdapterTests|MigrationParityTests|EspressoGenerateTests"

# Hardware tests (Apple Silicon required)
ANE_HARDWARE_TESTS=1 swift test --filter "ANERuntimeTests|EspressoTests"

# Cross-validation (ObjC parity)
OBJC_CROSS_VALIDATION=1 ANE_HARDWARE_TESTS=1 swift test --filter CrossValidationTests
```

Write tests first (TDD). Place them in `Tests/<TargetName>Tests/`. The CI pipeline runs all non-hardware tests automatically.

## Submitting Changes

1. **Fork** the repository and create a branch from `main`.
2. **Make your changes** — keep commits focused; one logical change per commit.
3. **Run tests** — all unit tests must pass. Hardware tests are strongly encouraged.
4. **Open a PR** against `main`. Fill in the PR template.
5. **CI must pass** before merge.

Commit message format:

```
<type>: <short summary>

<optional body>
```

Types: `feat`, `fix`, `refactor`, `docs`, `test`, `chore`, `perf`, `ci`

**Benchmark claims**: Any PR that includes a performance claim must include a machine-readable benchmark artifact from `./scripts/reproduce_local_real_artifact_claim.sh`. Self-reported numbers without artifacts will not be accepted.

## Issue Guidelines

- **Bug reports**: Use the bug report template. Include the output of `./espresso doctor`, your hardware, and the exact error.
- **Feature requests**: Use the feature request template. Describe the use case, not just the solution.
- **ANE behavior**: If you encounter `statusType=0x9` or `InvalidMILProgram`, include the MIL snippet and macOS version. These are often hardware/OS-specific.

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
