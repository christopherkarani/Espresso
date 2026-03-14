import Foundation
import ANEInterop
import IOSurface

public struct ANEChainingProbe: Sendable {
    public let hasChainingRequestClass: Bool
    public let hasPrepareSelector: Bool
    public let hasOutputSetsClass: Bool
    public let hasOutputSetsFactory: Bool
    public let hasOutputSetEnqueueClass: Bool
    public let hasInputBuffersReadyClass: Bool
    public let hasSharedSignalEventClass: Bool
    public let builtOutputSet: Bool
    public let builtOutputSetEnqueue: Bool
    public let builtInputBuffersReady: Bool
    public let builtSharedSignalEvent: Bool
    public let builtRequest: Bool
    public let usedArrayLoopbackSymbolIndices: Bool
    public let usedRealStatsSurface: Bool
    public let requestValidated: Bool
    public let requestValid: Bool
    public let requestValidationFailed: Bool
    public let inputBuffersReadyValidationFailed: Bool
    public let calledEnqueueSets: Bool
    public let enqueueSetsSucceeded: Bool
    public let calledBuffersReady: Bool
    public let buffersReadySucceeded: Bool
    public let prepared: Bool
    public let stage: Int
}

public struct ANEVirtualClientProbe: Sendable {
    public let hasVirtualClientClass: Bool
    public let hasVirtualClientProperty: Bool
    public let hasSharedEventsClass: Bool
    public let hasSharedWaitEventClass: Bool
    public let hasSharedSignalEventClass: Bool
    public let hasIOSurfaceSharedEventClass: Bool
    public let hasDoEvaluateCompletionEvent: Bool
    public let hasStandardEvaluate: Bool
    public let hasMapIOSurfaces: Bool
    public let hasLoadModel: Bool
    public let hasRequestSharedEventsFactory: Bool
    public let hasSetSharedEvents: Bool
    public let hasSetCompletionHandler: Bool
    public let obtainedVirtualClient: Bool
    public let triedPropertyOnClient: Bool
    public let triedDirectSharedConnection: Bool
    public let triedInitWithSingletonAccess: Bool
    public let triedNew: Bool
    public let directConnectSucceeded: Bool
    public let builtIOSurfaceSharedEvent: Bool
    public let builtWaitEvent: Bool
    public let builtSignalEvent: Bool
    public let builtSharedEventsContainer: Bool
    public let builtRequest: Bool
    public let mappedSurfaces: Bool
    public let loadedOnVirtualClient: Bool
    public let standardEvalSucceeded: Bool
    public let completionEventEvalSucceeded: Bool
    public let completionHandlerFired: Bool
    public let stage: Int
}

public struct ANEKernel: ~Copyable {
    private enum CompileGate {
        static let lock = NSLock()
    }

    private let handle: OpaquePointer

    @inline(__always)
    private static func checkedSurfaceIndex(_ index: Int) throws(ANEError) -> Int32 {
        guard index >= 0 && index <= Int(Int32.max) else {
            throw .invalidSurfaceIndex(index)
        }
        return Int32(index)
    }

    @inline(__always)
    private static func mapInteropCompileError() -> ANEError {
        switch ane_interop_last_compile_error() {
        case ANE_INTEROP_COMPILE_ERROR_INVALID_ARGUMENTS:
            return .invalidArguments("ANE interop compile contract rejected arguments")
        case ANE_INTEROP_COMPILE_ERROR_DUPLICATE_WEIGHT_PATH:
            return .invalidArguments("Duplicate weight path detected")
        case ANE_INTEROP_COMPILE_ERROR_SURFACE_ALLOCATION_FAILED:
            return .surfaceAllocationFailed
        default:
            return .compilationFailed
        }
    }

    static func compileWithRetry(
        checkBudget: Bool,
        compileAttempt: () -> OpaquePointer?,
        lastCompileError: () -> Int32 = { ane_interop_last_compile_error() },
        sleepAfterFailedAttempt: (Int) -> Void = ANECompileRetryPolicy.sleepAfterFailedAttempt,
        writeRetryNotice: (String) -> Void = Self.writeCompileRetryNotice
    ) throws(ANEError) -> OpaquePointer? {
        var attemptIndex = 0

        while true {
            let handle: OpaquePointer?
            if checkBudget {
                CompileGate.lock.lock()
                if CompileBudget.isExhausted {
                    CompileGate.lock.unlock()
                    throw .compileBudgetExhausted
                }
                handle = compileAttempt()
                CompileGate.lock.unlock()
            } else {
                handle = compileAttempt()
            }

            if let handle {
                return handle
            }

            let error = lastCompileError()
            guard ANECompileRetryPolicy.shouldRetry(
                lastCompileError: error,
                attemptIndex: attemptIndex
            ) else {
                return nil
            }

            writeRetryNotice(ANECompileRetryPolicy.retryNotice(afterFailedAttempt: attemptIndex))
            sleepAfterFailedAttempt(attemptIndex)
            attemptIndex += 1
        }
    }

    /// Compile a MIL program with optional weight blobs into an ANE kernel.
    ///
    /// - Parameters:
    ///   - milText: MIL program text (will be UTF-8 encoded)
    ///   - weights: Array of (path, data) pairs. Paths follow "@model_path/weights/<name>.bin" convention.
    ///   - inputSizes: Byte sizes for each input IOSurface (typically 1 element for single-input kernels)
    ///   - outputSizes: Byte sizes for each output IOSurface (typically 1 element)
    ///   - checkBudget: If true, checks CompileBudget before compiling. Default true.
    ///
    /// - Throws: `ANEError.compileBudgetExhausted` if budget exceeded (and checkBudget is true).
    ///           `ANEError.compilationFailed` if ANE compilation returns NULL.
    public init(
        milText: String,
        weights: [(path: String, data: Data)],
        inputSizes: [Int],
        outputSizes: [Int],
        checkBudget: Bool = true
    ) throws(ANEError) {
        guard weights.count <= Int(Int32.max),
              inputSizes.count <= Int(Int32.max),
              outputSizes.count <= Int(Int32.max),
              inputSizes.allSatisfy({ $0 >= 0 }),
              outputSizes.allSatisfy({ $0 >= 0 }) else {
            throw .invalidArguments("Counts must fit Int32 and byte sizes must be non-negative")
        }

        guard let milData = milText.data(using: .utf8), !milData.isEmpty else {
            throw .invalidArguments("MIL text must be valid, non-empty UTF-8")
        }

        let weightPaths = weights.map(\.path)
        let weightDatas = weights.map(\.data)
        var pathPointers = [UnsafePointer<CChar>?](repeating: nil, count: weights.count)
        var dataPointers = [UnsafePointer<UInt8>?](repeating: nil, count: weights.count)
        let dataLengths = weightDatas.map(\.count)

        func compileHandle() -> OpaquePointer? {
            ane_interop_init()

            return milData.withUnsafeBytes { milRaw in
                let milBuffer = milRaw.bindMemory(to: UInt8.self)
                guard let milBase = milBuffer.baseAddress else {
                    return nil
                }

                func withWeightPathPointers<R>(_ index: Int, _ body: () -> R) -> R {
                    guard index < weightPaths.count else {
                        return body()
                    }
                    return weightPaths[index].withCString { cPath in
                        pathPointers[index] = cPath
                        return withWeightPathPointers(index + 1, body)
                    }
                }

                func withWeightDataPointers<R>(_ index: Int, _ body: () -> R) -> R {
                    guard index < weightDatas.count else {
                        return body()
                    }
                    if weightDatas[index].isEmpty {
                        dataPointers[index] = nil
                        return withWeightDataPointers(index + 1, body)
                    }
                    return weightDatas[index].withUnsafeBytes { raw in
                        dataPointers[index] = raw.bindMemory(to: UInt8.self).baseAddress
                        return withWeightDataPointers(index + 1, body)
                    }
                }

                return withWeightPathPointers(0) {
                    withWeightDataPointers(0) {
                        pathPointers.withUnsafeMutableBufferPointer { pathBuf in
                            dataPointers.withUnsafeMutableBufferPointer { dataBuf in
                                dataLengths.withUnsafeBufferPointer { lenBuf in
                                    inputSizes.withUnsafeBufferPointer { inputBuf in
                                        outputSizes.withUnsafeBufferPointer { outputBuf in
                                            ane_interop_compile(
                                                milBase,
                                                milBuffer.count,
                                                weights.isEmpty ? nil : pathBuf.baseAddress,
                                                weights.isEmpty ? nil : dataBuf.baseAddress,
                                                weights.isEmpty ? nil : lenBuf.baseAddress,
                                                Int32(weights.count),
                                                Int32(inputSizes.count),
                                                inputSizes.isEmpty ? nil : inputBuf.baseAddress,
                                                Int32(outputSizes.count),
                                                outputSizes.isEmpty ? nil : outputBuf.baseAddress
                                            )
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        let rawHandle = try Self.compileWithRetry(
            checkBudget: checkBudget,
            compileAttempt: compileHandle
        )

        guard let rawHandle else {
            throw Self.mapInteropCompileError()
        }
        self.handle = rawHandle
    }

    private static func writeCompileRetryNotice(_ message: String) {
        guard let data = (message + "\n").data(using: .utf8) else {
            return
        }
        FileHandle.standardError.write(data)
    }

    /// Convenience: single-input, single-output kernel (most common case).
    public init(
        milText: String,
        weights: [(path: String, data: Data)],
        inputBytes: Int,
        outputBytes: Int,
        checkBudget: Bool = true
    ) throws(ANEError) {
        try self.init(
            milText: milText,
            weights: weights,
            inputSizes: [inputBytes],
            outputSizes: [outputBytes],
            checkBudget: checkBudget
        )
    }

    /// Run the compiled kernel on ANE.
    /// Input data must be written to inputSurface before calling.
    /// Output data is available on outputSurface after return.
    public func eval() throws(ANEError) {
        guard ane_interop_eval(handle) else {
            throw .evaluationFailed
        }
    }

    /// Last reported on-chip execution time from `_ANEPerformanceStats` (nanoseconds).
    ///
    /// Returns `0` when perf stats are disabled (default) or unsupported on this host.
    public func lastHWExecutionTimeNS() -> UInt64 {
        ane_interop_last_hw_execution_time_ns(handle)
    }

    public func chainingProbe(
        useRealStatsSurface: Bool = false,
        skipPrepare: Bool = false,
        validateRequest: Bool = true,
        useScalarLoopbackSymbolIndices: Bool = false,
        callEnqueueSets: Bool = false,
        callBuffersReady: Bool = false,
        requestProcedureIndex: UInt32 = 0,
        requestTransactionHandle: UInt64 = 0,
        requestFWEnqueueDelay: UInt64 = 0,
        requestMemoryPoolId: UInt64 = 0,
        enqueueProcedureIndex: UInt32 = 0,
        enqueueSetIndex: UInt32 = 0,
        enqueueSignalValue: UInt64 = 0,
        enqueueSignalNotRequired: Bool = true,
        enqueueOpenLoop: Bool = false,
        readyProcedureIndex: UInt32 = 0,
        readyExecutionDelay: UInt64 = 0,
        useSharedSignalEvent: Bool = false,
        sharedSignalEventValue: UInt64 = 1,
        sharedSignalEventSymbolIndex: UInt32 = 0,
        sharedSignalEventType: Int64 = 0
    ) -> ANEChainingProbe {
        var options = ANEInteropChainingProbeOptions(
            useRealStatsSurface: useRealStatsSurface,
            skipPrepare: skipPrepare,
            validateRequest: validateRequest,
            useScalarLoopbackSymbolIndices: useScalarLoopbackSymbolIndices,
            callEnqueueSets: callEnqueueSets,
            callBuffersReady: callBuffersReady,
            requestProcedureIndex: requestProcedureIndex,
            requestTransactionHandle: requestTransactionHandle,
            requestFWEnqueueDelay: requestFWEnqueueDelay,
            requestMemoryPoolId: requestMemoryPoolId,
            enqueueProcedureIndex: enqueueProcedureIndex,
            enqueueSetIndex: enqueueSetIndex,
            enqueueSignalValue: enqueueSignalValue,
            enqueueSignalNotRequired: enqueueSignalNotRequired,
            enqueueOpenLoop: enqueueOpenLoop,
            readyProcedureIndex: readyProcedureIndex,
            readyExecutionDelay: readyExecutionDelay,
            useSharedSignalEvent: useSharedSignalEvent,
            sharedSignalEventValue: sharedSignalEventValue,
            sharedSignalEventSymbolIndex: sharedSignalEventSymbolIndex,
            sharedSignalEventType: sharedSignalEventType
        )
        var raw = ANEInteropChainingProbeResult()
        ane_interop_probe_chaining_with_options(handle, &options, &raw)
        return ANEChainingProbe(
            hasChainingRequestClass: raw.hasChainingRequestClass,
            hasPrepareSelector: raw.hasPrepareSelector,
            hasOutputSetsClass: raw.hasOutputSetsClass,
            hasOutputSetsFactory: raw.hasOutputSetsFactory,
            hasOutputSetEnqueueClass: raw.hasOutputSetEnqueueClass,
            hasInputBuffersReadyClass: raw.hasInputBuffersReadyClass,
            hasSharedSignalEventClass: raw.hasSharedSignalEventClass,
            builtOutputSet: raw.builtOutputSet,
            builtOutputSetEnqueue: raw.builtOutputSetEnqueue,
            builtInputBuffersReady: raw.builtInputBuffersReady,
            builtSharedSignalEvent: raw.builtSharedSignalEvent,
            builtRequest: raw.builtRequest,
            usedArrayLoopbackSymbolIndices: raw.usedArrayLoopbackSymbolIndices,
            usedRealStatsSurface: raw.usedRealStatsSurface,
            requestValidated: raw.requestValidated,
            requestValid: raw.requestValid,
            requestValidationFailed: raw.requestValidationFailed,
            inputBuffersReadyValidationFailed: raw.inputBuffersReadyValidationFailed,
            calledEnqueueSets: raw.calledEnqueueSets,
            enqueueSetsSucceeded: raw.enqueueSetsSucceeded,
            calledBuffersReady: raw.calledBuffersReady,
            buffersReadySucceeded: raw.buffersReadySucceeded,
            prepared: raw.prepared,
            stage: Int(raw.stage)
        )
    }

    public func virtualClientProbe(
        useCompletionEvent: Bool = false,
        useCompletionHandler: Bool = false,
        useSharedEvents: Bool = false,
        useWaitEvent: Bool = false,
        skipEval: Bool = false,
        mapSurfaces: Bool = false,
        loadOnVirtualClient: Bool = false,
        useDirectInstantiation: Bool = false,
        waitEventValue: UInt64 = 1,
        waitEventType: UInt64 = 0,
        signalSymbolIndex: UInt32 = 0,
        waitSymbolIndex: UInt32 = 0
    ) -> ANEVirtualClientProbe {
        var options = ANEInteropVCProbeOptions(
            useCompletionEvent: useCompletionEvent,
            useCompletionHandler: useCompletionHandler,
            useSharedEvents: useSharedEvents,
            useWaitEvent: useWaitEvent,
            skipEval: skipEval,
            mapSurfaces: mapSurfaces,
            loadOnVirtualClient: loadOnVirtualClient,
            useDirectInstantiation: useDirectInstantiation,
            waitEventValue: waitEventValue,
            waitEventType: waitEventType,
            signalSymbolIndex: signalSymbolIndex,
            waitSymbolIndex: waitSymbolIndex
        )
        var raw = ANEInteropVCProbeResult()
        ane_interop_probe_virtual_client_eval(handle, &options, &raw)
        return ANEVirtualClientProbe(
            hasVirtualClientClass: raw.hasVirtualClientClass,
            hasVirtualClientProperty: raw.hasVirtualClientProperty,
            hasSharedEventsClass: raw.hasSharedEventsClass,
            hasSharedWaitEventClass: raw.hasSharedWaitEventClass,
            hasSharedSignalEventClass: raw.hasSharedSignalEventClass,
            hasIOSurfaceSharedEventClass: raw.hasIOSurfaceSharedEventClass,
            hasDoEvaluateCompletionEvent: raw.hasDoEvaluateCompletionEvent,
            hasStandardEvaluate: raw.hasStandardEvaluate,
            hasMapIOSurfaces: raw.hasMapIOSurfaces,
            hasLoadModel: raw.hasLoadModel,
            hasRequestSharedEventsFactory: raw.hasRequestSharedEventsFactory,
            hasSetSharedEvents: raw.hasSetSharedEvents,
            hasSetCompletionHandler: raw.hasSetCompletionHandler,
            obtainedVirtualClient: raw.obtainedVirtualClient,
            triedPropertyOnClient: raw.triedPropertyOnClient,
            triedDirectSharedConnection: raw.triedDirectSharedConnection,
            triedInitWithSingletonAccess: raw.triedInitWithSingletonAccess,
            triedNew: raw.triedNew,
            directConnectSucceeded: raw.directConnectSucceeded,
            builtIOSurfaceSharedEvent: raw.builtIOSurfaceSharedEvent,
            builtWaitEvent: raw.builtWaitEvent,
            builtSignalEvent: raw.builtSignalEvent,
            builtSharedEventsContainer: raw.builtSharedEventsContainer,
            builtRequest: raw.builtRequest,
            mappedSurfaces: raw.mappedSurfaces,
            loadedOnVirtualClient: raw.loadedOnVirtualClient,
            standardEvalSucceeded: raw.standardEvalSucceeded,
            completionEventEvalSucceeded: raw.completionEventEvalSucceeded,
            completionHandlerFired: raw.completionHandlerFired,
            stage: Int(raw.stage)
        )
    }

    /// Access input IOSurface (retained; safe to hold independently of kernel lifetime).
    public func inputSurface(at index: Int) throws(ANEError) -> IOSurfaceRef {
        let checkedIndex = try Self.checkedSurfaceIndex(index)
        guard let surface = ane_interop_copy_input(handle, checkedIndex) else {
            throw .inputSurfaceUnavailable(index)
        }
        return surface
    }

    /// Access output IOSurface (retained; safe to hold independently of kernel lifetime).
    public func outputSurface(at index: Int) throws(ANEError) -> IOSurfaceRef {
        let checkedIndex = try Self.checkedSurfaceIndex(index)
        guard let surface = ane_interop_copy_output(handle, checkedIndex) else {
            throw .outputSurfaceUnavailable(index)
        }
        return surface
    }

    /// Replace an input surface and rebuild the ANE request.
    /// Used for surface sharing between kernels (zero-copy I/O chaining).
    public func rebindInput(at index: Int, to surface: IOSurfaceRef) throws(ANEError) {
        let checkedIndex = try Self.checkedSurfaceIndex(index)
        guard ane_interop_rebind_input(handle, checkedIndex, surface) else {
            throw .invalidArguments("rebindInput failed at index \(index)")
        }
    }

    public struct CodeSigningProbe: Sendable {
        public let hasGetCodeSigningIdentity: Bool
        public let hasSetCodeSigningIdentity: Bool
        public let gotIdentityString: Bool
        public let setIdentityBeforeInstantiation: Bool
        public let instantiationSucceededAfterSet: Bool
        public let identityString: String
    }

    public static func codeSigningProbe() -> CodeSigningProbe {
        var raw = ANEInteropCodeSigningProbeResult()
        ane_interop_probe_code_signing(&raw)
        let identityStr = withUnsafeBytes(of: raw.identityString) { buf in
            let ptr = buf.baseAddress!.assumingMemoryBound(to: CChar.self)
            return String(cString: ptr)
        }
        return CodeSigningProbe(
            hasGetCodeSigningIdentity: raw.hasGetCodeSigningIdentity,
            hasSetCodeSigningIdentity: raw.hasSetCodeSigningIdentity,
            gotIdentityString: raw.gotIdentityString,
            setIdentityBeforeInstantiation: raw.setIdentityBeforeInstantiation,
            instantiationSucceededAfterSet: raw.instantiationSucceededAfterSet,
            identityString: identityStr
        )
    }

    public struct StandardCompletionProbe: Sendable {
        public let requestHasCompletionHandler: Bool
        public let completionHandlerSet: Bool
        public let requestHasSharedEvents: Bool
        public let metalDeviceCreated: Bool
        public let builtMetalSharedEvent: Bool
        public let builtSignalEvent: Bool
        public let builtSharedEventsContainer: Bool
        public let sharedEventsAttached: Bool
        public let evalSucceeded: Bool
        public let completionHandlerFired: Bool
        public let eventValueAdvanced: Bool
        public let eventValueBefore: UInt64
        public let eventValueAfter: UInt64
        public let evalTimeMS: Double
    }

    public func standardCompletionProbe(useMetalSharedEvent: Bool = false) -> StandardCompletionProbe {
        var raw = ANEInteropStandardCompletionProbeResult()
        ane_interop_probe_standard_completion_handler(handle, useMetalSharedEvent, &raw)
        return StandardCompletionProbe(
            requestHasCompletionHandler: raw.requestHasCompletionHandler,
            completionHandlerSet: raw.completionHandlerSet,
            requestHasSharedEvents: raw.requestHasSharedEvents,
            metalDeviceCreated: raw.metalDeviceCreated,
            builtMetalSharedEvent: raw.builtMetalSharedEvent,
            builtSignalEvent: raw.builtSignalEvent,
            builtSharedEventsContainer: raw.builtSharedEventsContainer,
            sharedEventsAttached: raw.sharedEventsAttached,
            evalSucceeded: raw.evalSucceeded,
            completionHandlerFired: raw.completionHandlerFired,
            eventValueAdvanced: raw.eventValueAdvanced,
            eventValueBefore: raw.eventValueBefore,
            eventValueAfter: raw.eventValueAfter,
            evalTimeMS: raw.evalTimeMS
        )
    }

    // MARK: - Real-time eval probe

    public struct RealTimeEvalProbe: Sendable {
        public let hasBeginRealTimeTask: Bool
        public let hasEndRealTimeTask: Bool
        public let hasLoadRealTimeModel: Bool
        public let hasUnloadRealTimeModel: Bool
        public let hasEvaluateRealTime: Bool
        public let realtimeLoadSucceeded: Bool
        public let realtimeEvalSucceeded: Bool
        public let standardEvalSucceeded: Bool
        public let realtimeEvalsCompleted: Int
        public let standardEvalsCompleted: Int
        public let realtimeTotalMS: Double
        public let standardTotalMS: Double
        public let realtimePerEvalMS: Double
        public let standardPerEvalMS: Double
        public let savedPerEvalMS: Double
        public let savedPercent: Double
    }

    /// Benchmark the real-time eval path vs standard eval on this kernel.
    ///
    /// Runs `nIters` evaluations on both paths and returns per-eval timing.
    /// The real-time path uses `beginRealTimeTask` → `loadRealTimeModel:` →
    /// `evaluateRealTimeWithModel:` → `unloadRealTimeModel:` → `endRealTimeTask`.
    public func realTimeEvalProbe(nIters: Int = 30) -> RealTimeEvalProbe {
        var raw = ANEInteropRealTimeProbeResult()
        ane_interop_probe_realtime_eval(handle, Int32(nIters), &raw)
        return RealTimeEvalProbe(
            hasBeginRealTimeTask: raw.hasBeginRealTimeTask,
            hasEndRealTimeTask: raw.hasEndRealTimeTask,
            hasLoadRealTimeModel: raw.hasLoadRealTimeModel,
            hasUnloadRealTimeModel: raw.hasUnloadRealTimeModel,
            hasEvaluateRealTime: raw.hasEvaluateRealTime,
            realtimeLoadSucceeded: raw.realtimeLoadSucceeded,
            realtimeEvalSucceeded: raw.realtimeEvalSucceeded,
            standardEvalSucceeded: raw.standardEvalSucceeded,
            realtimeEvalsCompleted: Int(raw.realtimeEvalsCompleted),
            standardEvalsCompleted: Int(raw.standardEvalsCompleted),
            realtimeTotalMS: raw.realtimeTotalMS,
            standardTotalMS: raw.standardTotalMS,
            realtimePerEvalMS: raw.realtimePerEvalMS,
            standardPerEvalMS: raw.standardPerEvalMS,
            savedPerEvalMS: raw.savedPerEvalMS,
            savedPercent: raw.savedPercent
        )
    }

    /// Quick check: does this kernel's _ANEClient support the real-time eval path?
    public var hasRealTimeEvalSupport: Bool {
        ane_interop_runtime_has_realtime_eval(handle)
    }

    deinit {
        ane_interop_free(handle)
    }
}
