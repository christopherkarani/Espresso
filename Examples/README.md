# Espresso Examples

Self-contained examples showing how to use Espresso from a standalone Swift package.

## SimpleInference

Generate text with GPT-2 on the Neural Engine in ~20 lines.

```bash
cd SimpleInference
ESPRESSO_WEIGHTS_DIR=~/.espresso/models/gpt2 swift run
ESPRESSO_WEIGHTS_DIR=~/.espresso/models/gpt2 swift run SimpleInference "The sky is"
```

Requires GPT-2 weights. Run `./espresso install` from the Espresso repo to download them.

## BenchmarkSuite

Compare Espresso ANE vs CoreML side-by-side using the repo's built-in bench command.

```bash
cd BenchmarkSuite
swift run
swift run BenchmarkSuite "The Neural Engine is"
```

Requires the `espresso` script to be in your PATH (run `./espresso install`).

## TrainingLoop

Fine-tune a small transformer on a local text corpus using the ANE.

```bash
cd TrainingLoop
CORPUS_DIR=~/my-notes swift run
CORPUS_DIR=~/my-notes TRAIN_STEPS=500 LAYER_COUNT=6 swift run
```

Builds and invokes `espresso-train` from the Espresso package. First run compiles ANE kernels — subsequent runs reuse the system cache.

## Setup

All examples reference Espresso via SPM from GitHub. First run will download and build the package.

For local development, point the dependency at your checkout:

```swift
// In Package.swift
.package(path: "../../")   // relative path to local Espresso clone
```
