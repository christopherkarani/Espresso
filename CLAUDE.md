# CLAUDE.md

Guidance for Claude Code (and other AI assistants) working in this repository.
Read this file before making non-trivial changes — it summarizes the module
layout, build/test workflow, and the project-specific conventions that the
human contributors expect to see respected in PRs.

## What Espresso is

Espresso is a pure-Swift package that compiles MIL (Model Intermediate
Language) programs directly onto the Apple Neural Engine via reverse-engineered
private APIs (`_ANEClient`, `_ANEInMemoryModel`), bypassing CoreML. It runs
transformer inference and training on ANE with IOSurface buffers, fused
multi-layer kernels, and a zero-copy decode loop.

Key properties to keep in mind when editing:

- **Target platform**: macOS 15+ on Apple Silicon (M1+). ANE hardware tests
  can only run on Apple Silicon. CI runs non-hardware unit tests only.
- **Private API surface**: `Sources/ANEInterop` `dlopen`s Apple's private
  `AppleNeuralEngine.framework`. Touch this layer only when you know what
  you're doing and document the macOS versions you verified against.
- **Zero external dependencies** (except the sibling `Edgerunner` package
  declared via a local `.package(path: "../Edgerunner")`). Do not add third
  party packages.
- **Swift 6.2, strict concurrency, language mode v6** on every target.
  `~Copyable` move-only value types are used for kernels, surfaces, and
  weight buffers.
- **Not App Store safe**. Private API usage is deliberate and scoped to
  research, internal tools, sideloaded, and enterprise distribution.

## Repository layout

```
/
├── Package.swift            # SPM manifest, 30+ targets
├── espresso                 # zsh launcher (./espresso demo|doctor|...)
├── README.md                # user-facing docs
├── CONTRIBUTING.md          # human contributor guide (read this too)
├── CLAUDE.md                # this file (tracked — guide for AI assistants)
├── Sources/                 # Swift/ObjC/C source modules
├── Tests/                   # mirror of Sources (one *Tests folder per target)
├── Examples/                # SimpleInference, TrainingLoop, BenchmarkSuite
├── benchmarks/              # benchmark rigs and baseline results
├── configs/                 # model configs
├── docs/                    # public docs (most of docs/ is gitignored)
│   └── platform/            # .esp / .espc platform design notes
├── scripts/                 # reproduction + benchmark scripts (many gitignored)
├── tasks/                   # internal planning (gitignored)
└── .github/workflows/       # CI, benchmark dashboard, Claude integration
```

### Source modules (Sources/)

The dependency graph is strictly layered: lower layers never import higher
layers.

**Low-level ANE bridge**
- `ANEInterop` — ObjC/C layer. `ane_interop.m`, `neon_convert.c`,
  `surface_io.c`. `dlopen` bridge to `_ANEClient` / `_ANEInMemoryModel`,
  NEON-vectorized IO conversions.
- `ANETypes` — `~Copyable` tensor/weight/surface value types
  (`TensorBuffer`, `TensorBufferFP16`, `TensorBufferINT4`, `WeightBlob`,
  `SurfaceIO`, `ModelConfig`, `LayerStorage`, `AdamState`, etc.).
- `CPUOps` — Accelerate/vDSP kernels used as CPU fallbacks and for
  non-ANE ops: `RMSNorm`, `RoPE`, `SiLU`, `Embedding`, `CrossEntropy`,
  `AdamOptimizer`.

**Graph IR and MIL codegen**
- `ANEGraphIR` — typed graph IR for ANE-targeted programs.
- `ANEPasses` — optimization passes over `ANEGraphIR`.
- `ANECodegen` — lowers `ANEGraphIR` → MIL text.
- `ANEBuilder` — high-level kernel builder façade over IR+codegen+passes.
- `MILGenerator` — hand-written MIL text generators for 28+ kernel
  variants (decode, fused two/three-layer, RWKV-style recurrent,
  classifier, FFN/SDPA forward/backward, etc.). Naming pattern:
  `*Generator.swift` exposing a `milText: String`.

**Runtime**
- `ANERuntime` — compiles MIL to ANE E5 binaries (`ANEKernel.swift`),
  manages IOSurface I/O, compile retry/budget (`ANECompileRetryPolicy`,
  `CompileBudget`), and layered kernel-set bundles (`*KernelSet.swift`)
  for decode/generation/fused variants.
- `DeltaCompilation` — delta compilation path for LoRA adapters.
- `LoRAAdapter` — LoRA overlay support on top of `ANEGraphIR` / `ANEBuilder`.

**High-level inference & training**
- `Espresso` — transformer layer glue, forward/backward passes, decode
  loops, generation harness, gradient accumulation, sampler,
  `GenerationHarness`, `RWKVStyle*Decode`, Metal fallbacks, checkpoint.
- `ModelSupport` — `GPT2BPETokenizer`, `SentencePieceTokenizer`,
  `ModelRegistry`, `MultiModelConfig`, `TransformerLayerGraphBuilder`,
  weight loaders.
- `RealModelInference` — orchestrates a real GPT-2/Llama model end-to-end
  (`RealModelInferenceEngine`, `ClassifierStrategy`).

**ESP platform (.esp / .espc bundles)**
- `ESPBundle` — canonical portable `.esp` bundle format.
- `ESPCompiler` — model-config IO and support matrix for `.esp` compiles.
- `ESPConvert` — native model directory → `.esp` exporter.
- `ESPRuntime` — bundle-aware runtime selection
  (`ESPBundledRuntime`, `ESPRuntimeContracts`).
- `ESPBenchSupport` — shared bench plumbing used by executables.

**Executables** (products in `Package.swift`)
- `espresso-generate` (`EspressoGenerate`) — demo CLI, TUI compare, prompt
  suite. `./espresso` wrapper shells out to this.
- `espresso-bench` (`EspressoBench`) — microbenchmarks for ANE/CPU kernels.
- `espresso-train` (`EspressoTrain`) — training loop driver.
- `espresso-multitoken-probe` (`EspressoMultitokenProbe`) — multi-token
  decode verifier.
- `espc` (`ESPCompilerCLI`) — packs native model dirs into `.esp`.
- `esprun` (`ESPRuntimeCLI`) — inspects / resolves / runs `.esp` bundles.
- `EspressoGGUFRunner` + `EspressoGGUF` — GGUF interop (depends on
  `Edgerunner` package).

### Tests (Tests/)

One folder per target, mirroring `Sources/`. Fixture resources live under
each test target's `Fixtures/` and are declared in `Package.swift`.

Three tiers:

1. **Unit tests (no hardware)** — run in CI. Filter set used by CI:
   `ANETypesTests|MILGeneratorTests|CPUOpsTests|ANEGraphIRTests|ANECodegenTests|ANEPassesTests|ANEBuilderTests|ModelSupportTests|DeltaCompilationTests|LoRAAdapterTests|MigrationParityTests|EspressoGenerateTests`
2. **ANE hardware tests** — require Apple Silicon + private API access.
   Gated by `ANE_HARDWARE_TESTS=1`. Covers `ANERuntimeTests`, `EspressoTests`.
3. **Cross-validation parity** — gated by `OBJC_CROSS_VALIDATION=1`
   alongside `ANE_HARDWARE_TESTS=1`. Covers `CrossValidationTests`.

## Build, run, test

Always prefer SPM; do not introduce Xcode projects.

```bash
# Build everything
swift build                          # debug
swift build -c release               # release (matches CI)

# Run unit tests (safe from Linux/CI/non-ANE macs)
swift test --filter "ANETypesTests|MILGeneratorTests|CPUOpsTests|ANEGraphIRTests|ANECodegenTests|ANEPassesTests|ANEBuilderTests|ModelSupportTests|DeltaCompilationTests|LoRAAdapterTests|MigrationParityTests|EspressoGenerateTests"

# Run ANE hardware tests (Apple Silicon only)
ANE_HARDWARE_TESTS=1 swift test --filter "ANERuntimeTests|EspressoTests"

# Run ObjC cross-validation parity (Apple Silicon only)
OBJC_CROSS_VALIDATION=1 ANE_HARDWARE_TESTS=1 swift test --filter CrossValidationTests

# Demo / tooling entry points
./espresso                           # TUI demo (builds + downloads weights)
./espresso "Hello"                   # generate text
./espresso doctor                    # host readiness check
./espresso compare --no-power "Hi"   # side-by-side vs CoreML
./espresso bench --output-dir reports "Hi"
swift run espresso-bench --ane-only --inference --layers 6
swift run espc pack-native /path/to/model /tmp/model.esp --overwrite
swift run esprun inspect /tmp/model.esp
swift run esprun generate /tmp/model.esp "Hello" 32
```

The `./espresso` launcher (zsh) manages a local Clang module cache under
`.build/clang-module-cache` and always invokes Swift with
`--disable-sandbox`. Keep that behavior when editing the launcher.

## Coding conventions (enforced)

Pulled from `CONTRIBUTING.md` and what the codebase already does — match it.

- **Swift language mode v6** on every new target
  (`swiftSettings: [.swiftLanguageMode(.v6)]`).
- **Strict concurrency**: assume Sendable boundaries and typed throws where
  the error set is bounded (e.g. `ANEError`).
- **`~Copyable` value types** for move-only resources (kernels, surfaces,
  weights). Don't silently make something Copyable.
- **Immutable value types by default**. Avoid mutation; avoid classes unless
  you need reference identity.
- **File size**: keep files ≤400 lines, functions ≤50 lines where feasible.
- **Zero dependencies** beyond Apple frameworks and the local `Edgerunner`
  package. Linked frameworks per target are already wired in `Package.swift`
  (`Accelerate`, `CoreML`, `IOSurface`, `Metal`, `Foundation`, `-ldl`).
- **Test coverage**: ≥80% for new code. Prefer TDD — write the failing test
  first in `Tests/<TargetName>Tests/`.
- **MIL generators** go in `Sources/MILGenerator/`, named `*Generator.swift`,
  exposing a `milText: String`, paired with a
  `Tests/MILGeneratorTests/<Name>GeneratorTests.swift`.
- **No speculative abstractions**. Three similar lines is better than a
  premature helper. Don't add config knobs that have no caller.
- **Don't touch `ANEInterop` casually**. If you must, document the macOS
  version range you tested against in the PR and in the file.

## Commits, branches, PRs

- Commit message format: `<type>: <short summary>` with types
  `feat | fix | refactor | docs | test | chore | perf | ci`.
- One logical change per commit. Avoid bundling refactors with fixes.
- Development for this assistant happens on the designated branch
  (see the current session's branch instructions, e.g.
  `claude/add-claude-documentation-z0jHu`). Never push to `main` directly.
- All unit tests must pass before pushing. Hardware tests are strongly
  encouraged where applicable.
- **Never create a PR unless the user explicitly asks for one.**
- **Benchmark claims** in PRs must be backed by a machine-readable artifact
  from `./scripts/reproduce_local_real_artifact_claim.sh`. Self-reported
  numbers without artifacts are rejected.
- CI (`.github/workflows/ci.yml`) runs `swift build -c release` and the
  unit-test filter above on macOS 15 with Xcode 16.2 and 16.3.

## Gitignore awareness (important for agents)

`.gitignore` hides a lot of internal content. If you go looking for files
and they're "missing", check here first:

- `.claude/`, `AGENTS.md` — Claude/agent config is ignored. `CLAUDE.md`
  itself is tracked (this file). `.claude/` local settings and `AGENTS.md`
  stay untracked.
- `tasks/`, `prompts/`, `agents/`, `marketing/`, `archive/`,
  `autoresearch-master/`, `training/`, `results/` — internal planning,
  not committed.
- `docs/` is mostly ignored; only a handful of HTML pages plus
  `docs/benchmarks.md` and `docs/platform/` (planning notes) are tracked.
- `scripts/` is mostly ignored; only the explicitly allowlisted scripts
  (`reproduce_local_real_artifact_claim.sh`, `run_power_benchmark.sh`,
  `run_autoresearch_*.sh`, `judge_suite_results.sh`,
  `generate-benchmark-dashboard.sh`, a few Python helpers, and
  `scripts/tests/`) are tracked.
- `benchmarks/results/` is ignored except for `latest.json`.
- Build outputs: `.build/`, `.swiftpm/`, `Package.resolved`,
  `DerivedData/`, `xcuserdata/`.
- Large weights: `*.gguf`, `*.safetensors`.
- `Sources/EspressoBenchApp/` and `Tests/EspressoBenchAppTests/` are
  deliberately local.

Before adding new files, check whether the target path is ignored — if
the user expects the file to be committed, you may need to either pick a
different location or force-add.

## Architecture invariants to preserve

- **Compile once, decode many**: the decode loop compiles MIL programs
  once and reuses the same program across every token step. Do not
  regress this by adding per-step compilation.
- **KV cache in IOSurface**: the KV cache lives in IOSurface buffers and
  is never marshaled through CoreML. Zero-copy reads on the output path.
- **Fused multi-layer kernels**: 6 transformer layers are intentionally
  dispatched as 2 ANE evals via fused 3-layer kernels. Don't "simplify"
  this back to one layer per dispatch.
- **Exact two-token decode**: each step produces two verified tokens.
  Parity is checked in `CrossValidationTests` / `MigrationParityTests`.
- **Deterministic `.esp` contract**: `ESPBundle` / `ESPCompiler` /
  `ESPRuntime` share a bundle-manifest contract. Schema changes need a
  migration story — see `docs/platform/2026-03-26-*.md`.

## Working notes for AI assistants

- When asked for a "small fix", actually make it small. Don't refactor
  surrounding code, don't add docstrings to code you didn't touch, don't
  add error handling for impossible cases. See the top-level "Doing tasks"
  guidance.
- Prefer the dedicated tools (Read, Edit, Grep, Glob) over shelling out.
- For multi-file exploration use the `Explore` subagent; for directed
  lookups, use `Grep`/`Glob` directly.
- If a task might need ANE hardware to validate, say so explicitly in the
  PR / commit message — CI cannot run those tests.
- If you hit `statusType=0x9` or `InvalidMILProgram` while iterating, the
  MIL program is the suspect, not the runtime. Capture the snippet and
  the macOS version.
- The sibling `Edgerunner` package is expected at `../Edgerunner` relative
  to this repo. If it isn't present, GGUF-related targets will fail to
  resolve — that's a local environment issue, not a bug to fix.
