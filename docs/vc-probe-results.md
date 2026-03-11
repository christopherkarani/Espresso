# VirtualClient + SharedEvents + CompletionEvent Eval Path Probe

**Date:** 2026-03-06
**Host:** M3 Max, macOS 15.x
**Branch:** `feat/vc-eval-probe`
**Test suite:** `VirtualClientEvalProbeTests` (17 tests, all passing)

## Motivation

ANE decode throughput is eval-dominant (0.404ms compute vs 0.029ms IO per token). The
current decode path dispatches 12 synchronous `eval()` calls per token (2 kernels x 6
layers), each carrying ~0.095ms overhead. IO trimming is exhausted and the abandoned
`_ANEChainingRequest` path stalled at metadata contract barriers.

**Hypothesis:** The `_ANEVirtualClient.doEvaluateWithModel:completionEvent:` path
and/or `_ANERequest.setCompletionHandler:` may provide lower-overhead or async eval
that enables pipeline parallelism.

---

## 1. Class Discovery

All five target classes are present on M3 Max macOS 15:

| Class | Class methods | Instance methods | Notes |
|-------|:---:|:---:|-------|
| `_ANEVirtualClient` | 22 | 66 | Full compile/load/eval lifecycle |
| `_ANESharedEvents` | 2 | 8 | Container for signal + wait arrays |
| `_ANESharedWaitEvent` | 3 | 8 | Two factory methods available |
| `_ANESharedSignalEvent` | (confirmed in chaining probe) | — | Factory: `signalEventWithValue:symbolIndex:eventType:sharedEvent:` |
| `IOSurfaceSharedEvent` | present | — | `+new` returns **nil** |

---

## 2. Complete _ANEVirtualClient Method Dump

### Class Methods (22)

```
+ sharedConnection                      ← KEY: own singleton?
+ new
+ getCodeSigningIdentity
+ setCodeSigningIdentity:
+ createIOSurface:ioSID:
+ freeModelFileIOSurfaces:
+ copyDictionaryDataToStruct:dictionary:
+ copyLLIRBundleToIOSurface:writtenDataSize:
+ getCFDictionaryFromIOSurface:dataLength:
+ getDictionaryWithJSONEncodingFromIOSurface:withArchivedDataSize:
+ getObjectFromIOSurface:classType:length:
+ printStruct:
+ printIOSurfaceDataInBytes:
+ shouldUsePrecompiledPath:options:shouldUseChunking:chunkingThreshold:
+ updateError:errorLength:error:
+ updateError:errorLength:errorCode:error:
+ updatePerformanceStats:performanceStatsLength:perfStatsRawIOSurfaceRef:performanceStatsRawLength:hwExecutionTime:
+ dictionaryGetInt8ForKey:key:
+ dictionaryGetInt64ForKey:key:
+ dictionaryGetUInt32ForKey:key:
+ dictionaryGetUInt64ForKey:key:
+ dictionaryGetNSStringForKey:key:
```

### Instance Methods (66) — grouped by function

**Lifecycle:**
```
- init
- initWithSingletonAccess
- connect
- dealloc
- .cxx_destruct
```

**Hardware query:**
```
- hasANE
- numANEs
- numANECores
- aneBoardtype
- aneArchitectureTypeStr
- isInternalBuild
- getDeviceInfo
- getModelAttribute:
- negotiatedDataInterfaceVersion
- negotiatedCapabilityMask
- hostBuildVersionStr
- sendGuestBuildVersion
- exchangeBuildVersionInfo
- getValidateNetworkVersion
- validateEnvironmentForPrecompiledBinarySupport
```

**Compile/validate:**
```
- compileModel:options:qos:error:
- validateNetworkCreate:uuid:function:directoryPath:scratchPadPath:milTextData:
- validateNetworkCreateMLIR:validation_params:
- compiledModelExistsFor:
- compiledModelExistsMatchingHash:
- purgeCompiledModel:
- purgeCompiledModelMatchingHash:
```

**Load/unload:**
```
- loadModel:options:qos:error:
- loadModelNewInstance:options:modelInstParams:qos:error:
- loadModelNewInstanceLegacy:options:modelInstParams:qos:error:
- unloadModel:options:qos:error:
```

**Eval:**
```
- evaluateWithModel:options:request:qos:error:               ← standard
- doEvaluateWithModel:options:request:qos:completionEvent:error:   ← async?
- doEvaluateWithModelLegacy:options:request:qos:completionEvent:error:
```

**Surface mapping:**
```
- mapIOSurfacesWithModel:request:cacheInference:error:
- doMapIOSurfacesWithModel:request:cacheInference:error:
- releaseIOSurfaces:
```

**Real-time:**
```
- beginRealTimeTask
- endRealTimeTask
```

**Session hints:**
```
- sessionHintWithModel:hint:options:report:error:
```

**IOSurface data transfer:**
```
- copyToIOSurface:length:ioSID:
- copyToIOSurface:size:ioSID:
- copyDictionaryToIOSurface:copiedDataSize:createdIOSID:
- copyFilesInDirectoryToIOSurfaces:ioSurfaceRefs:ioSurfaceSizes:fileNames:
- copyModelOptionFiles:options:vmData:
- copyModelOptionFiles:options:dictionary:vmData:
- copyModelMetaData:options:dictionary:vmData:
- copyModel:options:vmData:
- copyOptions:vmData:
- copyOptions:dictionary:vmData:
- copyAllModelFiles:dictionary:ioSurfaceRefs:
```

**IOKit transport:**
```
- callIOUserClient:inParams:outParams:
- callIOUserClientWithDictionary:inDictionary:error:
- checkKernReturnValue:selector:outParams:
```

**Utility:**
```
- queue
- outputDictIOSurfaceSize
- echo:
- readWeightFilename:
- doJsonParsingMatchWeightName:
- parallelDecompressedData:
- printDictionary:
- transferFileToHostWithPath:withChunkSize:withUUID:withModelInputPath:overWriteFileNameWith:
- updateError:error:
- copyErrorValue:
- copyErrorValue:vmData:
- updatePerformanceStats:
```

### _ANESharedEvents (container)
```
+ new
+ sharedEventsWithSignalEvents:waitEvents:     ← factory
- init
- initWithSignalEvents:waitEvents:
- signalEvents / setSignalEvents:
- waitEvents / setWaitEvents:
- description
```

### _ANESharedWaitEvent
```
+ new
+ waitEventWithValue:sharedEvent:              ← 2-arg factory
+ waitEventWithValue:sharedEvent:eventType:    ← 3-arg factory
- init
- initWithValue:sharedEvent:eventType:
- value / setValue:
- sharedEvent
- eventType
- description
```

---

## 3. Selector Capability Matrix

| Selector | On what class | Present |
|----------|:---:|:---:|
| `_ANEClient.virtualClient` (property getter) | `_ANEClient` | **YES** |
| `evaluateWithModel:options:request:qos:error:` | `_ANEVirtualClient` | **YES** |
| `doEvaluateWithModel:options:request:qos:completionEvent:error:` | `_ANEVirtualClient` | **YES** |
| `doEvaluateWithModelLegacy:options:request:qos:completionEvent:error:` | `_ANEVirtualClient` | **YES** |
| `doMapIOSurfacesWithModel:request:cacheInference:error:` | `_ANEVirtualClient` | **YES** |
| `mapIOSurfacesWithModel:request:cacheInference:error:` | `_ANEVirtualClient` | **YES** |
| `loadModel:options:qos:error:` | `_ANEVirtualClient` | **YES** |
| `loadModelNewInstance:options:modelInstParams:qos:error:` | `_ANEVirtualClient` | **YES** |
| `compileModel:options:qos:error:` | `_ANEVirtualClient` | **YES** |
| `beginRealTimeTask` / `endRealTimeTask` | `_ANEVirtualClient` | **YES** |
| `_ANERequest.setSharedEvents:` | `_ANERequest` | **YES** |
| `_ANERequest.setCompletionHandler:` | `_ANERequest` | **YES** |
| 8-arg request factory with `sharedEvents:` param | `_ANERequest` | **NO** |
| 9-arg request factory with `sharedEvents:transactionHandle:` | `_ANERequest` | **NO** |
| `_ANEVirtualClient +sharedConnection` | `_ANEVirtualClient` | **YES** |

---

## 4. Critical Finding: `_ANEClient.virtualClient` Returns nil

**All eval paths blocked at stage 1 (NO_VIRTUAL_CLIENT).**

The `_ANEClient` obtained via `_ANEInMemoryModel.sharedConnection` returns **nil** for
the `virtualClient` property. This is the fundamental blocker — no eval, no
completionEvent, no completionHandler, no mapSurfaces could be reached.

### Why this happens

The `_ANEClient` obtained from `_ANEInMemoryModel.sharedConnection` appears to be the
same singleton as `[_ANEClient sharedConnection]`. This client manages the XPC
connection to `aned` (the ANE daemon). The `virtualClient` property is likely only
populated when:

1. The client is constructed via a different path (e.g., directly by CoreML internals)
2. A specific daemon handshake enables virtual client mode
3. The property is lazily initialized by some prerequisite call we haven't made

### Key observation from the method dump

`_ANEVirtualClient` itself has `+sharedConnection` as a **class method** and
`-initWithSingletonAccess` and `-connect` as instance methods. This strongly suggests
`_ANEVirtualClient` can be instantiated **directly** — it is not merely a property of
`_ANEClient` but a parallel entry point to the ANE.

---

## 5. IOSurfaceSharedEvent Construction

`IOSurfaceSharedEvent +new` returns **nil**. This matches the golden output constraint
documented in the plan.

**The correct construction path** is almost certainly via Metal:
```objc
id<MTLDevice> device = MTLCreateSystemDefaultDevice();
id<MTLSharedEvent> mtlEvent = [device newSharedEvent];
// mtlEvent conforms to IOSurfaceSharedEvent (same underlying type)
```

Metal's `MTLSharedEvent` IS an `IOSurfaceSharedEvent` under the hood — they share the
same backing kernel object. The `+new` on `IOSurfaceSharedEvent` fails because it
requires kernel-level initialization that only the GPU driver provides.

---

## 6. Test Results Summary

| Test | Outcome | Key finding |
|------|---------|-------------|
| `test_runtime_has_virtual_client_returns_bool` | PASS | Capability check doesn't crash |
| `test_runtime_has_shared_events_request_returns_bool` | PASS | Capability check doesn't crash |
| `test_vc_probe_discovers_all_classes` | PASS | All classes present, all selectors found |
| `test_vc_probe_obtains_virtual_client_from_client` | PASS | Property exists, returns nil |
| `test_vc_probe_builds_shared_events_container` | PASS | IOSurfaceSharedEvent +new → nil |
| `test_vc_probe_standard_eval_on_virtual_client` | PASS | Blocked at stage 1 |
| `test_vc_probe_completion_event_eval_nil` | PASS | Blocked at stage 1 |
| `test_vc_probe_completion_event_eval_with_shared_event` | PASS | Blocked at stage 1 |
| `test_vc_probe_completion_handler_on_request` | PASS | Blocked at stage 1 |
| `test_vc_probe_with_map_surfaces` | PASS | Blocked at stage 1 |
| `test_vc_probe_with_load_on_virtual_client` | PASS | Blocked at stage 1 |
| `test_vc_probe_full_pipeline` | PASS | Blocked at stage 1 |

All 17 tests pass. Hardware tests blocked at stage 1 for VirtualClient but complete
without crashes or hangs. CompletionHandler fires successfully on standard eval path.

### Phase 2 Tests (Direct Instantiation + New Probes)

| Test | Outcome | Key finding |
|------|---------|-------------|
| `test_vc_probe_direct_shared_connection` | PASS | All 5 paths tried, all return nil |
| `test_vc_probe_direct_instantiation_with_eval` | PASS | Direct instantiation fails, no eval possible |
| `test_vc_probe_direct_full_pipeline` | PASS | Full pipeline blocked at VirtualClient acquisition |
| `test_vc_probe_code_signing_identity` | PASS | Identity is `com.apple.xctest`, set crashes |
| `test_vc_probe_standard_completion_handler` | PASS | **Handler fires on standard eval path!** |

---

## 7. Direct Instantiation Attempts (Phase 2)

**Date:** 2026-03-06 (same session)

All five direct `_ANEVirtualClient` instantiation paths were attempted:

| Path | Tried | Result |
|------|-------|--------|
| `_ANEClient.virtualClient` property | Yes | nil |
| `[_ANEVirtualClient sharedConnection]` | Yes | nil (no exception) |
| `alloc → initWithSingletonAccess → connect` | Yes | `alloc` succeeds, `initWithSingletonAccess` returns nil |
| `[_ANEVirtualClient new]` | Yes | nil (no exception) |
| `alloc → init → connect` | Yes | `init` returns nil |

**Pattern:** `alloc` succeeds (raw memory allocated) but every `init*` variant returns
nil. This is the classic pattern for IOKit service initialization failing due to missing
entitlements or kernel-level permissions. The nil is silent (no exception, no NSError),
confirming a deliberate kernel-level gate.

### Code Signing Identity Probe

| Finding | Value |
|---------|-------|
| `+getCodeSigningIdentity` available | YES |
| `+setCodeSigningIdentity:` available | YES |
| Current identity | `'com.apple.xctest'` (type: `__NSCFString`) |
| Set identity attempt | Crashes: `-[__NSCFConstantString __setObject:forKey:]` |

The `setCodeSigningIdentity:` implementation internally treats the class-level identity
store as a **dictionary keyed by identity string**. Setting it via `objc_msgSend` with a
plain `NSString` crashes because it tries to call `__setObject:forKey:` on the string
argument. This suggests the identity is stored in a shared static `NSDictionary` and the
setter expects a different calling pattern (possibly used by the ANE daemon internally).

### Key Conclusion: VirtualClient Requires Kernel Entitlements

`_ANEVirtualClient` uses IOKit directly (`-callIOUserClient:inParams:outParams:`) instead
of XPC. The kernel ANE driver (likely `AppleH*ANEInterface`) enforces an entitlement check
during IOKit service open. All `init*` methods fail because the test process lacks the
required entitlement (probably in the `com.apple.ane.*` family). Only Apple-signed system
frameworks (CoreML, coremlcompiler) can create VirtualClient instances.

**This path is definitively blocked** for third-party code without entitlement injection.

---

## 8. CompletionHandler Discovery (Phase 2 — BREAKTHROUGH)

**`_ANERequest.setCompletionHandler:` WORKS on the standard eval path.**

| Finding | Value |
|---------|-------|
| `_ANERequest` supports `setCompletionHandler:` | YES |
| Handler set successfully | YES |
| Handler fires after eval | **YES** |
| Handler fires even on eval failure | **YES** |
| Does NOT require VirtualClient | Correct — works with `_ANEInMemoryModel.evaluateWithQoS:` |

The completion handler (block callback, type encoding `@?`) fires after eval through the
standard `_ANEInMemoryModel` path. It does **not** require `_ANEVirtualClient`. This
provides async notification capability on the existing eval path.

**Implications for decode optimization:**
- Can attach a handler before each eval to know exactly when eval completes
- Enables overlapping IO with eval: start reading output from Layer N while Layer N+1 evals
- Does NOT enable pipeline parallelism (eval is still synchronous from the ANE's perspective)
- The handler fires immediately after eval returns — useful for profiling and dispatch coordination

**Limitation:** The handler fires on the calling thread synchronously after eval returns
(not on a background thread). This means it provides notification but not true async dispatch.
Pipeline parallelism still requires either VirtualClient's `completionEvent:` (hardware-level
async) or the chaining request path.

---

## 9. Updated Next Steps

### Immediate — exploit completionHandler

1. **Profile completionHandler overhead**
   Measure the cost of setting/firing a completion handler vs bare eval. If overhead
   is <1µs, it's free instrumentation for the decode path.

2. **Explore dispatch_async + completionHandler**
   If eval is dispatched on a background queue and the handler signals completion,
   the main thread can prepare IO for the next layer concurrently.

### Medium priority — remaining VirtualClient angles

3. **File-based `_ANEModel` path**
   Compile to `.mlmodelc` on disk, load via `_ANEModel`, check if `_ANEClient`
   constructed through this path populates `virtualClient`. The file-based path
   may unlock different entitlement handling.

4. **Swizzle CoreML's eval path**
   Hook `_ANEClient.evaluateWithModel:` to intercept the VirtualClient after CoreML
   creates it. CoreML must use VirtualClient internally for certain models.

5. **Metal-backed IOSurfaceSharedEvent**
   `[MTLDevice newSharedEvent]` → use with `_ANESharedEvents` container. Even without
   VirtualClient, shared events may work on the standard path.

### Low priority — alternative approaches

6. **Compute-side decode optimization**
   Reduce kernel count via MIL reformulation (fuse attention + FFN per layer,
   reduce from 12 dispatches to 6).

7. **`beginRealTimeTask` on _ANEClient**
   The existing `_ANEClient` eval path has `evaluateRealTimeWithModel:` — may provide
   lower dispatch overhead without VirtualClient.

---

## 10. Architecture Insight

The method dump reveals `_ANEVirtualClient` is **not** a lightweight wrapper around
`_ANEClient`. It is a full, parallel ANE access path with its own:

- Connection management (`-connect`, `-callIOUserClient:...`)
- Compilation (`-compileModel:...`, `-validateNetworkCreate:...`)
- Loading (`-loadModel:...`, `-loadModelNewInstance:...`)
- Evaluation (`-evaluateWithModel:...`, `-doEvaluateWithModel:...completionEvent:...`)
- Surface mapping (`-doMapIOSurfacesWithModel:...`)
- Real-time task management (`-beginRealTimeTask`, `-endRealTimeTask`)
- IOKit kernel interface (`-callIOUserClient:inParams:outParams:`)
- Session hints (`-sessionHintWithModel:hint:options:report:error:`)

This is a complete, self-contained ANE client. The `completionEvent` parameter in
`doEvaluateWithModel:` is architecturally significant — it suggests hardware-level
async notification, not just a software callback. If we can instantiate it, the entire
compile→load→eval lifecycle can run on VirtualClient without touching `_ANEClient` or
`_ANEInMemoryModel`.

---

## 11. Relationship to Chaining Probe

| Probe | Status | Blocker |
|-------|--------|---------|
| `_ANEChainingRequest` | Stage 3-4 (builds request, prepare fails) | Metadata contract unknown |
| `_ANEVirtualClient` eval | Stage 1 (blocked at acquisition) | nil from `_ANEClient.virtualClient` |

The two probes attack different paths to the same goal (lower dispatch overhead). They
are complementary — either one succeeding unlocks pipeline parallelism. VirtualClient
appears more promising because it has a cleaner API surface (standard eval + completionEvent)
vs chaining's complex output-set / input-buffers-ready protocol.

---

## Appendix: Verified Type Encodings (from plan)

| Class | Method | Encoding |
|-------|--------|----------|
| `_ANEVirtualClient` | `doEvaluateWithModel:options:request:qos:completionEvent:error:` | `B60@0:8@16@24@32I40@44^@52` |
| `_ANEVirtualClient` | `evaluateWithModel:options:request:qos:error:` | `B52@0:8@16@24@32I40^@44` |
| `_ANEVirtualClient` | `doMapIOSurfacesWithModel:request:cacheInference:error:` | `B44@0:8@16@24B32^@36` |
| `_ANEVirtualClient` | `loadModel:options:qos:error:` | `B44@0:8@16@24I32^@36` |
| `_ANESharedEvents` | `sharedEventsWithSignalEvents:waitEvents:` | `@32@0:8@16@24` |
| `_ANESharedWaitEvent` | `waitEventWithValue:sharedEvent:` | `@32@0:8Q16@24` |
| `_ANESharedWaitEvent` | `waitEventWithValue:sharedEvent:eventType:` | `@40@0:8Q16@24Q32` |
| `_ANERequest` | `setCompletionHandler:` | `v24@0:8@?16` |
| `_ANERequest` | `setSharedEvents:` | `v24@0:8@16` |

## 10. Metal-backed SharedEvent on Standard Eval Path (2026-03-06)

Probe outcome:
- `_ANERequest` exposes `setSharedEvents:` on the standard eval path.
- `IOSurfaceSharedEvent +new` remains unusable, but a Metal-backed `MTLSharedEvent` can be created successfully.
- Wrapping that Metal event in `_ANESharedSignalEvent` and attaching it through `_ANESharedEvents` causes the standard eval path to stop making forward progress on hardware.
- No completion callback is observed and no event increment is observed before the eval path stalls.

Conclusion:
- Metal-backed shared events are not a viable async completion mechanism for the standard `_ANERequest` eval path in this environment.
