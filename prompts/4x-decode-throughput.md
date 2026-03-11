# Prompt: ANE 4x Decode Throughput — Sequential Avenue Exploration

> Paste this prompt to start the continuation session.

---

I'm working on achieving 4x decode inference throughput over CoreML on the Apple Neural Engine. The project is Espresso — a Swift 6.2 codebase that drives ANE directly via reverse-engineered private APIs (`_ANEClient`, `_ANEInMemoryModel`, IOSurface).

## Where we left off

Branch `feat/vc-eval-probe` in the worktree at `/Users/chriskarani/CodingProjects/Espresso/.worktrees/vc-probe`. Build is clean, all tests pass.

We've already:
- Fused the decode attention+FFN into a single MIL dispatch per layer (0.095ms/step saved, 12→6 dispatches)
- Exhaustively probed VirtualClient (blocked by kernel entitlements), ChainingRequest (PREPARE_FAILED), evaluateRealTimeWithModel (loadRealTimeModel silently fails), IOSurface aliasing (0x9 at eval), QoS tuning (no effect)
- Discovered CompletionHandler works on standard eval (fires synchronously, not true async)

The remaining wall is ANE compute time (~2.43ms/token for 6 layers). Software dispatch overhead has been eliminated.

## Your task

Read `AGENTS.md` in the repo root — it contains a detailed execution plan with 6 avenues to explore in order, benchmarking protocol, abandon criteria, and all the context you need.

Execute the avenues **sequentially** (Avenue 1 through 6). For each:
1. Read the prior findings in `docs/` before writing code
2. Build a benchmark that isolates the avenue's impact (TDD)
3. Measure baseline → implement → measure after → report delta
4. Abandon within ~30 minutes if you hit an entitlement wall, silent nil, 0x9 crash, or unrecoverable failure
5. Commit atomically per avenue
6. Update `docs/fused-decode-and-next-steps.md` with measured numbers
7. Update the project memory (`MEMORY.md`) with any new ANE facts

Start with **Avenue 1: Multi-Layer Kernel Fusion (2-layer packed KV caches)** — this is the highest-confidence path using proven `slice_by_size` MIL patterns from the backward pass generators.

Keep working until all 6 avenues are explored or abandoned. At the end, produce a summary table showing cumulative gains and the CoreML comparison number.
