# Lessons

## 2026-03-14
- When adding a foundation IR type, encode documented invariants at the API boundary and add negative-path tests for malformed states before considering the layer complete.

## 2026-03-16
- When the user asks for autonomy, keep pushing through missing-asset and runtime blockers by acquiring local prerequisites and exercising the real path instead of stopping at the first environmental gap.

## 2026-03-16
- When a blocked runtime path is missing model artifacts, proactively download or convert the required assets and continue verification instead of stopping at the prerequisite check.

## 2026-03-16
- When the user explicitly asks for the most performant and production-ready path, prefer the validated architecture split from a working reference implementation over a quicker fallback that reduces ANE coverage.

## 2026-03-16
- When the user provides a reference implementation, treat it as evidence and a debugging guide rather than a strict implementation spec unless they explicitly require parity.

## 2026-03-16
- For ANE BLOBFILE weights, do not equate the on-disk blob header `dataOffset=128` with the MIL-side `BLOBFILE offset`; the proven compiler contract in Espresso/Orion is MIL `offset=64` over a blob whose payload still begins at byte 128.

## 2026-03-16
- When ANE compilation fails on a graph that is otherwise structurally aligned, inspect codegen-added output alias identities; naming the terminal nodes to the exported output names can remove no-op `identity` ops that still break the compiler on multi-output kernels.

## 2026-03-16
- When `main` still has a working runtime path, restore that baseline first and add optimizations beside it instead of replacing the baseline with an unverified fast path.

## 2026-03-16
- Before calling an autoresearch harness "ready", verify the git allowlist, the generated scaffold outputs, and the results schema against the actual suite workflow instead of assuming docs and scripts stayed aligned.

## 2026-03-18
- When the user names a specific benchmark target model, keep that exact model as the source of truth; do not substitute an easier open-weight fallback unless the user explicitly approves the tradeoff.

## 2026-03-19
- When using agent swarms on hard tasks in this repo, use `gpt-5.4-mini` at `xhigh` for explorer/research agents and reserve `gpt-5.4` at `high` for implementation workers only.

## 2026-03-19
- When freeing disk space during model debugging, verify the canonical source-model and `.artifacts` paths first and only delete disposable temp conversion directories such as `/var/folders/.../T/espresso_gguf_*`.

## 2026-03-20
- When broader verification depends on a missing protected model, start the download in the background immediately and continue local analysis or smaller checks instead of serially blocking on the asset fetch.

## 2026-03-25
- When a throughput idea does not clear the measured keep gate, revert it to an explicit experiment flag and preserve the measurement harness instead of shipping the faster-on-paper default.

## 2026-03-26
- When a baseline shifts materially, rerun previously rejected runtime knobs under the new baseline before treating the old verdict as permanent; donor-delta removal changed the sign of some Stories experiments even though the code in those knobs did not change.

## 2026-03-26
- When a stateful Core ML LLM package is exact on `cpu_only` but `cpu_and_neural_engine` fails with Core ML `-14`, bisect the recurrent layer count before changing math; on this Apple runtime the failure boundary appeared at 3 stateful layers, so further exporter tweaks inside the same single-package stateful format are low-probability.
