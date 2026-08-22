# Espresso — Current Plan

Last updated: 2026-08-22 (architecture deepening stack)

## Architecture deepening PRs (2026-08-22, stacked)

- [x] #31 `arch/artifact-loader`: one top-level weight loader (`TopLevelAssetLoader`), misnamed `loadTestingTopLevelAssets` deleted
- [x] #32 `arch/hybrid-step-assembler`: split/fused-hybrid surface bindings moved into ANERuntime; `evalPostAttention()` replaces fusion branching at call sites
- [x] #35 `arch/resolved-engine-policies`: `EnginePolicies` resolved once at `build(environment:)`; serving paths stop reading live process state; runner setenv channel deleted

Follow-ups (not started): kernel-set compile options threaded from policies; DecodeRuntimeOptions statics; CLI env seeding for compile-cache policy.

## What Espresso is today

- Direct-to-ANE inference runtime for Apple Silicon via reverse-engineered private APIs. Zero third-party dependencies.
- Shipping lane: exact `.esp` Stories serving, defended by `benchmarks/results/latest.json` and the reproduce script in `scripts/`.
- Research lanes (recurrent students, speculative drafts, distillation) are quarantined under `research/` and are not product claims.

## Shipped in the productization pass

- [x] Zero-dependency package graph (EdgeRunner/GGUF removed; MLX/LFM2 import stays in `ESPConvert`)
- [x] Clean-clone quick start: `./espresso doctor` → `./espresso prepare` → `./espresso` (bootstrap helpers restored)
- [x] README claims aligned to checked-in `benchmarks/results/latest.json`
- [x] Research configs/scripts quarantined to `research/`
- [x] Dead modules removed (DeltaCompilation, LoRAAdapter, ZigRuntime)
- [x] Runtime hardening: unique compile tmpdirs, bounded exec-restart, throwing SurfaceIO hot paths, MIL float round-trip safety

## Active priorities (in order)

1. **Real-model ANE serving lane** — one model done properly (LFM2-350M or Qwen-0.6B class): sharded device-side top-1 LM head, ANE-resident KV, single fused decode program. Success metric is tok/s **per watt** vs llama.cpp Metal / MLX on the same model, not raw tok/s.
2. **Power story** — power telemetry without sudo, or a documented entitlement path. Perf/watt is the ANE's actual advantage; make it measurable by users.
3. **Publication quality gate** — coherent 128-token output on the publication suite before any new throughput claim.

## Rules that bite (see tasks/lessons.md for full history)

- Every performance result names its lane: `shipping`, `publication`, `probe`, `microbench`.
- Microbench numbers are never publication claims.
- No benchmark work is done without retained artifacts.
- Do not build on removed dependency paths; cherry-pick into Espresso-owned code instead.
- Killed lanes get recorded in `tasks/lessons.md`, then the next successor lane is promoted immediately.
