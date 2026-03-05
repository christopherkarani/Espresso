# Research Task 4: ANE Private API Capabilities for Inference Optimization

**Date**: 2026-03-05
**Branch**: checkpoint/phase8-ane-inference-20260305
**Scope**: Areas 1e (Async Eval/Pipelining), 3a (Compilation Caching), 3b (Parallel Compilation), 3c (Weight Swapping)

---

## 1. Current API Usage Inventory

### Classes Used by Espresso

| Class | How Used | File |
|---|---|---|
| `_ANEInMemoryModelDescriptor` | `modelWithMILText:weights:optionsPlist:` -- creates descriptor from MIL text (NSData) + weight dictionary | `ane_interop.m:161-163` |
| `_ANEInMemoryModel` | `inMemoryModelWithDescriptor:` -- creates model from descriptor | `ane_interop.m:169-170` |
| `_ANEInMemoryModel` | `compileWithQoS:options:error:` -- compiles MIL to E5 binary (QoS=21) | `ane_interop.m:224-225` |
| `_ANEInMemoryModel` | `loadWithQoS:options:error:` -- loads compiled program to ANE hardware | `ane_interop.m:231-232` |
| `_ANEInMemoryModel` | `evaluateWithQoS:options:request:error:` -- synchronous eval | `ane_interop.m:361-362` |
| `_ANEInMemoryModel` | `unloadWithQoS:error:` -- unloads from hardware | `ane_interop.m:400-401` |
| `_ANEInMemoryModel` | `hexStringIdentifier` -- gets unique ID for temp directory naming | `ane_interop.m:176` |
| `_ANEIOSurfaceObject` | `objectWithIOSurface:` -- wraps IOSurfaceRef for ANE I/O | `ane_interop.m:308-309` |
| `_ANERequest` | `requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:` | `ane_interop.m:334-336` |

### Current Execution Flow (Per Kernel)

```
compile(MIL+weights) -> load() -> [write IOSurface -> eval() -> read IOSurface]* -> unload() -> free()
```

Key observations:
- **All operations are synchronous** -- `evaluateWithQoS:` blocks until completion.
- **Compilation writes temp files** -- even "in-memory" compilation writes MIL + weight blobs to a temp directory under `NSTemporaryDirectory()/<hexId>/`.
- **Weights are baked at compile time** -- passed via `_ANEInMemoryModelDescriptor`, not updatable post-compilation.
- **One request object per kernel** -- created once at compile time, reused for all evals. The `weightsBuffer` parameter is passed as `nil`.
- **CompileGate.lock (NSLock)** serializes compilations in `ANEKernel.init` to protect the `CompileBudget.isExhausted` check.
- **Compile budget = 100** (`ModelConfig.maxCompiles`), reflecting the known ~119 compile limit per process before resource leaks cause failures.

---

## 2. Async/Pipelining Capabilities (Area 1e)

### What the API Provides

**No async eval method exists.** All discovered selectors on `_ANEInMemoryModel` are synchronous:
- `compileWithQoS:options:error:` -- blocking
- `loadWithQoS:options:error:` -- blocking
- `evaluateWithQoS:options:request:error:` -- blocking
- `unloadWithQoS:error:` -- blocking

No completion-handler or callback-based variants (`evaluateAsyncWithQoS:...completionHandler:`) have been found in any reverse-engineering effort (maderix/ANE, mdaiter/ane, hollance/neural-engine).

### ANE Hardware Queue Depth

Research indicates the ANE supports a **queue depth of 127 in-flight evaluation requests**. However, this is a hardware/driver-level capability that is **not exposed through any known API**. The `evaluateWithQoS:` call blocks until completion, so the queue depth is effectively 1 from the caller's perspective.

### Undiscovered Classes That May Enable Pipelining

| Class | Potential | Status |
|---|---|---|
| `_ANEChainingRequest` | May enable chaining multiple compiled models in a single dispatch -- could eliminate CPU round-trips between kernels | **Unexplored** -- no known usage examples |
| `_ANESharedEvents` | Metal-style fence/signal primitives for GPU-ANE synchronization | **Unexplored** |
| `_ANESharedSignalEvent` | Signal event for cross-accelerator sync | **Unexplored** |
| `_ANESharedWaitEvent` | Wait event for cross-accelerator sync | **Unexplored** |
| `_ANEPerformanceStats` | Hardware performance counters -- could help identify bottlenecks | **Unexplored** |
| `_ANEVirtualClient` | Virtualized ANE access, potentially for multi-process sharing | **Unexplored** |

### Pipelining Assessment

**What IS possible today (without new API discovery):**

1. **GCD-based CPU/ANE overlap**: While one kernel's `eval()` runs on ANE, dispatch CPU work (surface preparation for the next kernel) on a concurrent queue. Espresso already does this partially with `GradientAccumulator` for training.

2. **Surface preparation pipelining**: For inference, while layer N's FFN eval runs, we can prepare layer N+1's attention input surface. This requires pre-allocated surface handles (which `InferenceSurfaceHandles` already provides).

3. **fp16 surface copy (already implemented)**: The `InferenceInterKernelHandoff.fp16SurfaceCopy` mode avoids CPU round-trip between attention and FFN kernels within a layer.

**What is NOT possible today:**

1. **True async eval** -- no way to fire eval() and get notified on completion without blocking a thread.
2. **Kernel chaining** -- no way to chain attention + FFN into a single ANE dispatch without MIL-level fusion.
3. **Pipeline parallelism** -- cannot run two kernels on ANE simultaneously (it's a single execution unit).

### Recommendation for Area 1e

**Priority: MEDIUM** -- The biggest win is already partially captured (fp16SurfaceCopy). Further improvement requires:

1. **Explore `_ANEChainingRequest`** (HIGH RISK, HIGH REWARD): Use runtime introspection (`class_copyMethodList`, `method_getTypeEncoding`) to dump its method signatures. If it allows chaining two `_ANERequest` objects, we could dispatch attention+FFN as one unit, eliminating the CPU round-trip entirely. Estimated impact: **15-25% latency reduction** per layer.

2. **GCD surface prep overlap** (LOW RISK, MODERATE REWARD): While ANE processes layer N's FFN, use a dispatch queue to prepare layer N+1's attention input. Requires careful IOSurface locking. Estimated impact: **5-10% latency reduction** overall.

---

## 3. Compilation Caching (Area 3a)

### Current State

- Compilation takes **~20-40ms** per kernel (first call).
- The ANE compiler **automatically caches E5 binaries** on disk at: `~/Library/Caches/<app>/com.apple.e5rt.e5bundlecache/<build>/<hash>/`
- Cache hits are "effectively free" per external research.
- However, **for inference**, Espresso compiles at startup and the compile cost is amortized. The real issue is the **~119 compile limit per process**.

### Compile Limit Analysis

| Metric | Value |
|---|---|
| Limit per process | ~119 compilations (empirically observed) |
| Espresso budget | 100 (conservative) |
| Kernels per layer (inference) | 2 (attention + FFN) |
| Layers | Configurable; 4 default |
| Total inference compiles | 8 (4 layers x 2 kernels) |
| Budget headroom | 92 remaining after inference setup |

**For pure inference, the compile budget is not a bottleneck.** The limit matters for training (recompile every batch when weights change).

### Serialization of Compiled Programs

**The compiled E5 binary is NOT directly accessible** through any known API. The `_ANEInMemoryModel` does not expose:
- A method to serialize the compiled program to `Data`
- A method to load a pre-compiled E5 binary
- Any handle to the cached E5 file

**However**, the automatic disk cache means:
- First run: compile (20-40ms per kernel)
- Subsequent runs (same MIL + same build): cache hit, near-instant

### Pre-compilation at Build Time

**Not feasible** through current APIs. The E5 binary format is opaque, architecture-specific, and tied to the OS build. The compiler service (`ANECompilerService`) runs as a separate process.

### Workaround for Compile Budget (Training)

The current `exec()` restart approach (in `EspressoTrain/main.swift`) is the only known workaround:
1. Accumulate gradients for N steps (N=10 in Espresso)
2. Save checkpoint
3. `exec()` to restart the process (resets compile count)
4. Resume from checkpoint

### Recommendation for Area 3a

**Priority: LOW for inference, HIGH for training**

For inference optimization:
1. **No action needed** -- 8 compilations fit well within the 100-budget, and disk caching handles repeated runs.
2. **Consider lazy compilation** -- compile kernels on first use rather than all at startup, to improve cold-start latency for single-layer inference.

---

## 4. Parallel Compilation (Area 3b)

### Is CompileGate.lock Necessary?

The `CompileGate.lock` (NSLock) in `ANEKernel.swift:134` serializes two things:
1. Reading `CompileBudget.isExhausted` (the atomic counter check)
2. Calling `compileHandle()` (the actual ANE compilation)

**Analysis:**

The lock protects a **check-then-act** race: without it, two threads could both see `isExhausted == false`, both proceed to compile, and one might hit the limit. This is a valid concern for training but irrelevant for inference (where we compile exactly once per kernel).

**Can compilations run concurrently?**

The ANE compiler (`ANECompilerService`) is an **out-of-process XPC service**. Based on:
- It uses a shared connection (`_ANEClient.sharedConnection`)
- CoreML documentation suggests serialized access to the compiler service
- No evidence that the XPC service handles concurrent requests

**Verdict: Parallel compilation is likely unsafe.** The compiler service may not be thread-safe, and concurrent compilation requests could lead to:
- XPC serialization (no speedup, just overhead)
- Resource contention in `ANECompilerService`
- Accelerated approach to the ~119 compile limit

### Does the Compiler Use Multiple Threads Internally?

The `ANECompilerService` process likely does multi-threaded work internally (LLVM-based compilation), but this is opaque. Dispatching multiple compilations to different queues would not help -- the bottleneck is the XPC service, not the calling thread.

### Recommendation for Area 3b

**Priority: VERY LOW for inference**

1. **Keep the lock for training safety** -- the check-then-act pattern requires it.
2. **For inference, the lock is unnecessary** but harmless -- 8 sequential compilations at ~20ms each = 160ms total startup cost. Not worth optimizing.
3. **If startup latency matters**: Remove the lock for inference-only code paths (where `checkBudget: false` is already available) and compile all kernels in parallel using `DispatchGroup`. Potential savings: from ~160ms sequential to ~40ms parallel (limited by XPC serialization). **Risky and unlikely to help.**

---

## 5. Weight Swapping (Area 3c)

### Can Weights Be Updated Without Recompilation?

**No.** This is a definitive finding confirmed by all research:

1. **Weights are baked into the E5 binary at compile time.** The `_ANEInMemoryModelDescriptor` accepts weights as `NSDictionary` at construction, and they are written to temp files and compiled into the program.

2. **The `weightsBuffer` parameter on `_ANERequest` does NOT override compiled weights.** Espresso correctly passes `nil` for this parameter. External research confirms "no hot-swap path exists."

3. **No `updateWeights:` or `setWeightData:forKey:` method exists** on any discovered class.

4. **IOSurface weight buffers cannot be remapped.** The ANE reads weights from the compiled E5 binary, not from IOSurfaces at eval time.

### Why This Architecture Exists

The ANE is a **graph execution engine** -- it takes a compiled neural network graph and executes the entire thing as one atomic operation. Weights are constant operands in the compiled graph, not runtime parameters. This is fundamentally different from GPU compute, where weights are just buffer arguments.

### Implications for Inference

**Weight swapping is irrelevant for inference** -- weights don't change between eval calls. This is only a concern for training (where weights update every batch).

### Recommendation for Area 3c

**Priority: N/A for inference**

No action needed. For training, the existing `exec()` restart + gradient accumulation approach is the only viable strategy.

---

## 6. Undiscovered APIs

### Discovered But Unused Classes (40+ in AppleNeuralEngine.framework)

| Class | Potential Inference Optimization Value | Risk Level |
|---|---|---|
| `_ANEChainingRequest` | **HIGH** -- could chain attention+FFN per layer | HIGH (completely unexplored) |
| `_ANEPerformanceStats` | **MEDIUM** -- hardware counters for profiling | LOW (read-only, safe to explore) |
| `_ANESharedEvents` | **LOW** for pure ANE, **HIGH** for GPU+ANE hybrid | MEDIUM |
| `_ANEVirtualClient` | **LOW** -- multi-process sharing, not inference speed | LOW |
| `_ANEClient` | **MEDIUM** -- `sharedConnection` may offer different eval path than `_ANEInMemoryModel` | MEDIUM |

### `_ANEClient` vs `_ANEInMemoryModel` Path

Espresso uses the `_ANEInMemoryModel` path exclusively. The `_ANEClient` path (`compileModel:options:qos:error:` / `evaluateWithModel:options:request:qos:error:`) is an alternative that:
- May have different performance characteristics
- May support features not available on the in-memory path
- Used by CoreML internally
- Requires a `_ANEModel` object (typically from a file path)

### Runtime Introspection Strategy

To discover new methods, add a one-time exploration step:
```objc
dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);
unsigned int count;
const char **names = objc_copyClassNamesForImage(
    "/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", &count);
for (unsigned int i = 0; i < count; i++) {
    Class cls = objc_getClass(names[i]);
    // dump class methods + instance methods
}
```

### QoS Parameter (Value 21)

The QoS parameter passed to all operations is hardcoded to `21`. The meaning is unknown but likely maps to an internal quality-of-service tier. Testing other values (e.g., 33, 17, 9) might reveal different scheduling priorities or hardware modes. **Low risk to explore.**

---

## 7. Concrete Recommendations (Prioritized)

### Tier 1: High Impact, Actionable Now

| # | Recommendation | Estimated Impact | Effort | Risk |
|---|---|---|---|---|
| 1 | **Explore `_ANEChainingRequest` via runtime introspection** | 15-25% latency reduction if it enables chaining attention+FFN | 1-2 days | HIGH -- may crash, may not work |
| 2 | **GCD surface prep pipelining** | 5-10% latency reduction | 0.5 days | LOW |
| 3 | **Explore `_ANEPerformanceStats`** for profiling | Indirect -- enables data-driven optimization | 0.5 days | LOW |

### Tier 2: Moderate Impact, Low Effort

| # | Recommendation | Estimated Impact | Effort | Risk |
|---|---|---|---|---|
| 4 | **Test different QoS values** (9, 17, 33) | Unknown, potentially 5-15% | 0.5 days | LOW |
| 5 | **Pre-allocate all IOSurfaces at startup** (already done via `InferenceSurfaceHandles`) | Already captured | Done | N/A |
| 6 | **Verify disk cache is warm** -- confirm E5 bundles are cached across runs | Eliminates 160ms cold start | 0.25 days | NONE |

### Tier 3: Low Priority / Speculative

| # | Recommendation | Estimated Impact | Effort | Risk |
|---|---|---|---|---|
| 7 | **Explore `_ANEClient` direct path** as alternative to `_ANEInMemoryModel` | Unknown | 1 day | MEDIUM |
| 8 | **Remove CompileGate lock for inference-only paths** | Negligible (<1ms) | Trivial | NONE |
| 9 | **Explore `_ANESharedEvents`** for future GPU+ANE hybrid | Future potential | 1-2 days | MEDIUM |

### Summary Impact Assessment

For inference optimization through API-level changes:

- **Maximum realistic gain: 20-30%** latency reduction (if `_ANEChainingRequest` works + surface prep pipelining)
- **Minimum likely gain: 5-10%** (surface prep pipelining alone)
- **The biggest inference wins remain at the MIL/kernel level** (fusion, tile alignment, reduced output channels), not at the API level

### Key Insight for 10x Goal

The ANE API is not the bottleneck. The current synchronous `eval()` pattern adds minimal overhead -- the CPU time between evals (surface I/O, residual additions) is the real target. The research from Tasks 2 and 3 (kernel fusion, MIL optimization) provides higher-leverage improvements than API-level changes.

---

## Sources

- [maderix/ANE -- Training neural networks on Apple Neural Engine via reverse-engineered private APIs](https://github.com/maderix/ANE)
- [Inside the M4 Apple Neural Engine, Part 1: Reverse Engineering](https://maderix.substack.com/p/inside-the-m4-apple-neural-engine)
- [Inside the M4 Apple Neural Engine, Part 2: ANE Benchmarks](https://maderix.substack.com/p/inside-the-m4-apple-neural-engine-615)
- [hollance/neural-engine -- Everything we actually know about the Apple Neural Engine](https://github.com/hollance/neural-engine)
- [mdaiter/ane -- Reverse engineered the Apple Neural Engine](https://github.com/mdaiter/ane)
- [Anemll -- Artificial Neural Engine Machine Learning Library](https://github.com/Anemll/Anemll)
- [Apple Neural Engine Internal -- Black Hat Asia 2021](https://i.blackhat.com/asia-21/Friday-Handouts/as21-Wu-Apple-Neural_Engine.pdf)
- [Apple Developer Forums -- Low level API to take control of Neural Engine](https://developer.apple.com/forums/thread/673627)
