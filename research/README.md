# Research quarantine

This tree is **not** the product path. It holds rejected or unfinished experiment
configs, distillation/draft tooling, and internal agent plans kept for
historical reference and offline reproduction.

## Product path (use these instead)

| Journey | Entry point |
|---------|-------------|
| Interactive demo | `./espresso` |
| Portable serving | `.esp` bundles via `esprun` / `espresso-generate --bundle` |
| Library integration | `ANERuntime.ANEKernel` (and related ANE* modules) |

Public claims live in [`benchmarks/results/latest.json`](../benchmarks/results/latest.json)
and must match the README benchmark table. Do not promote numbers from this
folder into product docs.

## What is here

- `configs/stories/` — rejected student / draft / future-head / RWKV experiment configs
- `scripts/` — distillation, factored-head packaging, autoresearch suite helpers
- `docs/platform/` — convert/optimize execution plans and agent prompts

## Retain / reject rule

Experiments land here unless a **same-binary** real-artifact benchmark beats the
retained exact Stories path without quality or ANE-path regression. See the
ledger in the local `tasks/todo.md` (gitignored) for historical measurements.
