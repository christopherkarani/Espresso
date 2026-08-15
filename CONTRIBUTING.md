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
- Xcode 26.2+ (Swift 6.2)
- Apple Silicon Mac (M1 or later) — required for ANE hardware tests

**Clone and build**

```bash
git clone https://github.com/christopherkarani/Espresso.git
cd Espresso
swift package resolve   # zero third-party packages
swift build
swift test              # unit tests, no ANE required
```

A clean clone must resolve and build with **no** external package dependencies (Apple frameworks only).

**Run the demo**

```bash
./espresso doctor   # check host readiness (must report scripts OK)
./espresso prepare  # bootstrap GPT-2 demo weights + tokenizer
./espresso          # builds if needed, launches TUI
```

The demo helpers `scripts/bootstrap_gpt2_demo.py`, `export_gpt2_coreml.py`, and
`run_gpt2_coreml_reference.py` must remain tracked — CI runs `scripts/assert_demo_helpers.py`.

**Hardware tests** (requires Apple Silicon ANE)

```bash
ANE_HARDWARE_TESTS=1 swift test --filter "ANERuntimeTests|EspressoTests|CrossValidationTests"
```

## Project Structure

```
Sources/
  ANEInterop/          # ObjC/C bridge to _ANEClient private API
  ANETypes/            # ~Copyable tensors, SurfaceIO, weight serialization
  MILGenerator/        # MIL text generation (kernel variants)
  CPUOps/              # CPU fallbacks via Accelerate/vDSP
  ANERuntime/          # Compile, eval, IOSurface management
  Espresso/            # Transformer layers, training, generation
  ANEGraphIR/          # Graph IR with optimization passes
  ANECodegen/          # MIL codegen from Graph IR
  ANEPasses/           # Graph optimization passes
  ANEBuilder/          # End-to-end kernel builder
  ModelSupport/        # GPT-2 and Llama model configs + ModelFamily
  DeltaCompilation/    # Delta compilation for LoRA adapters
  LoRAAdapter/         # LoRA adapter support
  RealModelInference/  # Real model hybrid inference engine
  EspressoGenerate/    # Generation CLI target
  ESPBundle/           # Portable .esp bundle format
  ESPRuntime/          # Bundle-aware runtime resolution
Tests/                 # Mirror of Sources structure
scripts/               # Benchmark and reproduction scripts
docs/                  # Architecture docs and research logs
artifacts/             # Generated benchmark artifacts (gitignored)
```

## Coding Standards

**Language**: Swift 6.2 with strict concurrency enabled. All new code must compile under `.swiftLanguageMode(.v6)`.

**Key conventions**:
- Use `~Copyable` for move-only resources (kernels, surfaces, weights)
- Prefer immutable value types; document intentional mutation
- Typed throws where the error set is bounded
- Package graph stays free of third-party Swift packages (Apple frameworks only)
- Prefer typed options over new process-environment feature flags
- Model-family special cases go through `ModelFamily` in `ModelSupport` — do not add new string `contains("stories110m")` checks elsewhere
- Prefer smaller, cohesive files and functions for new code. Existing modules include large historical files; do not grow them further without a split plan
- Test new behavior; CI does not currently enforce a coverage percentage

**MIL programs**: Kernel generators go in `Sources/MILGenerator/`. Follow the naming pattern `*Generator.swift` and output a `milText: String` property. Test with a corresponding `Tests/MILGeneratorTests/` file.

**Private API surface**: Changes to the `_ANEClient`/`_ANEInMemoryModel` bridge in `ANEInterop` require careful documentation — note the macOS version range tested.

## Testing

Run tests before submitting:

```bash
# Unit tests (no hardware required — matches CI)
swift test --filter "ANETypesTests|MILGeneratorTests|CPUOpsTests|ANEGraphIRTests|ANECodegenTests|ANEPassesTests|ANEBuilderTests|ModelSupportTests|DeltaCompilationTests|LoRAAdapterTests|MigrationParityTests|EspressoGenerateTests|ESPBundleTests|ESPCompilerTests|ESPRuntimeTests|ESPConvertTests|ESPBenchSupportTests|ESPCompilerCLITests|RealModelInferenceTests"

# Hardware tests (Apple Silicon required)
ANE_HARDWARE_TESTS=1 swift test --filter "ANERuntimeTests|EspressoTests"

# Cross-validation (ObjC parity)
OBJC_CROSS_VALIDATION=1 ANE_HARDWARE_TESTS=1 swift test --filter CrossValidationTests
```

Write tests for new behavior. Place them in `Tests/<TargetName>Tests/`. CI runs the non-hardware filter above on every PR that touches package sources/tests.

## Submitting Changes

1. **Fork** the repository and create a branch from `main`.
2. **Make your changes** — keep commits focused; one logical change per commit.
3. **Run tests** — all unit tests in the CI filter must pass. Hardware tests are strongly encouraged on Apple Silicon.
4. **Open a PR** against `main`. Fill in the PR template.
5. **CI must pass** before merge.

Commit message format:

```
<type>: <short summary>

<optional body>
```

Types: `feat`, `fix`, `refactor`, `docs`, `test`, `chore`, `perf`, `ci`

**Benchmark claims**: Any PR that includes a performance claim must either:
- update `benchmarks/results/latest.json` and the README table together, or
- attach a machine-readable artifact from `./scripts/reproduce_local_real_artifact_claim.sh`

Self-reported numbers without artifacts will not be accepted. Do not put peak research numbers in the README headline without a matching checked-in result file. CI runs `scripts/assert_readme_claims.py` to keep the table honest.

**Product path vs research**: Preferred product journeys are (1) `./espresso` demo, (2) `.esp` + `esprun` / `espresso-generate --bundle`, (3) `ANEKernel` library. Rejected experiment configs and distillation tooling live under `research/` — do not reintroduce them as default entry points or public claim sources.

## Issue Guidelines

- **Bug reports**: Use the bug report template. Include the output of `./espresso doctor`, your hardware, and the exact error.
- **Feature requests**: Use the feature request template. Describe the use case, not just the solution.
- **ANE behavior**: If you encounter `statusType=0x9` or `InvalidMILProgram`, include the MIL snippet and macOS version. These are often hardware/OS-specific.

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
