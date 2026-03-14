import Foundation
import ANETypes
import ANERuntime
import Espresso
import MILGenerator
import Darwin

// MARK: - CLI Argument Parsing

struct BenchmarkOptions {
    var aneOnly: Bool = false
    var inference: Bool = false
    var decode: Bool = false
    var inferenceFP16Handoff: Bool = false
    var inferenceOnly: Bool = false
    var profileKernels: Bool = false
    var sustained: Bool = false
    var warmup: Int = 50
    var iterations: Int = 1000
    var decodeSteps: Int = 32
    var decodeMaxSeq: Int = 32
    var outputDir: String? = nil
    var coreMLModelPath: String = "benchmarks/models/transformer_layer.mlpackage"
    var nLayers: Int = 1
    var dumpANE: Bool = false
    var dumpANEFilter: String? = nil
    var perfStats: Bool = false
    var probeChainingPrepare: Bool = false
    var probeChainingSkipPrepare: Bool = false
    var probeChainingCallBuffersReady: Bool = false
    var probeChainingCallEnqueueSets: Bool = false
    var probeChainingScalarLoopback: Bool = false
    var probeChainingStatsSurfaceMode: String = "output0"
    var probeChainingRequestProcedureIndex: UInt32 = 0
    var probeChainingRequestTransactionHandle: UInt64 = 0
    var probeChainingRequestFWEnqueueDelay: UInt64 = 0
    var probeChainingRequestMemoryPoolId: UInt64 = 0
    var probeChainingEnqueueProcedureIndex: UInt32 = 0
    var probeChainingEnqueueSetIndex: UInt32 = 0
    var probeChainingEnqueueSignalValue: UInt64 = 0
    var probeChainingEnqueueSignalNotRequired: Bool = true
    var probeChainingEnqueueOpenLoop: Bool = false
    var probeChainingReadyProcedureIndex: UInt32 = 0
    var probeChainingReadyExecutionDelay: UInt64 = 0
    var probeChainingUseSharedSignalEvent: Bool = false
    var probeChainingSharedSignalEventValue: UInt64 = 1
    var probeChainingSharedSignalEventSymbolIndex: UInt32 = 0
    var probeChainingSharedSignalEventType: Int64 = 0

    static func parse(_ args: [String]) -> BenchmarkOptions {
        var opts = BenchmarkOptions()
        var i = 1  // skip program name
        while i < args.count {
            switch args[i] {
            case "--ane-only":
                opts.aneOnly = true
            case "--inference":
                opts.inference = true
            case "--decode":
                opts.decode = true
            case "--inference-fp16-handoff":
                opts.inferenceFP16Handoff = true
            case "--inference-only":
                opts.inference = true
                opts.inferenceOnly = true
            case "--profile-kernels":
                opts.profileKernels = true
            case "--sustained":
                opts.sustained = true
            case "--warmup":
                i += 1
                guard i < args.count, let v = Int(args[i]) else {
                    printStderr("--warmup requires an integer argument")
                    exit(1)
                }
                opts.warmup = v
            case "--iterations":
                i += 1
                guard i < args.count, let v = Int(args[i]) else {
                    printStderr("--iterations requires an integer argument")
                    exit(1)
                }
                opts.iterations = v
            case "--decode-steps":
                i += 1
                guard i < args.count, let v = Int(args[i]), v > 0 else {
                    printStderr("--decode-steps requires a positive integer argument")
                    exit(1)
                }
                opts.decodeSteps = v
            case "--decode-max-seq":
                i += 1
                guard i < args.count, let v = Int(args[i]), v > 1 else {
                    printStderr("--decode-max-seq requires an integer >= 2")
                    exit(1)
                }
                opts.decodeMaxSeq = v
            case "--output":
                i += 1
                guard i < args.count else {
                    printStderr("--output requires a path argument")
                    exit(1)
                }
                opts.outputDir = args[i]
            case "--model":
                i += 1
                guard i < args.count else {
                    printStderr("--model requires a path argument")
                    exit(1)
                }
                opts.coreMLModelPath = args[i]
            case "--layers":
                i += 1
                guard i < args.count, let v = Int(args[i]) else {
                    printStderr("--layers requires an integer argument")
                    exit(1)
                }
                opts.nLayers = v
            case "--dump-ane":
                opts.dumpANE = true
            case "--dump-ane-filter":
                i += 1
                guard i < args.count else {
                    printStderr("--dump-ane-filter requires a string argument")
                    exit(1)
                }
                opts.dumpANEFilter = args[i]
            case "--perf-stats":
                opts.perfStats = true
            case "--probe-chaining-prepare":
                opts.probeChainingPrepare = true
            case "--probe-chaining-skip-prepare":
                opts.probeChainingSkipPrepare = true
            case "--probe-chaining-call-buffers-ready":
                opts.probeChainingCallBuffersReady = true
            case "--probe-chaining-call-enqueue-sets":
                opts.probeChainingCallEnqueueSets = true
            case "--probe-chaining-scalar-loopback":
                opts.probeChainingScalarLoopback = true
            case "--probe-chaining-stats-surface":
                i += 1
                guard i < args.count else {
                    printStderr("--probe-chaining-stats-surface requires one of: output0, null, scratch")
                    exit(1)
                }
                let mode = args[i].lowercased()
                guard ["output0", "null", "scratch"].contains(mode) else {
                    printStderr("--probe-chaining-stats-surface requires one of: output0, null, scratch")
                    exit(1)
                }
                opts.probeChainingStatsSurfaceMode = mode
            case "--probe-chaining-request-procedure-index":
                i += 1
                guard i < args.count, let v = UInt32(args[i]) else {
                    printStderr("--probe-chaining-request-procedure-index requires a non-negative integer argument")
                    exit(1)
                }
                opts.probeChainingRequestProcedureIndex = v
            case "--probe-chaining-request-transaction-handle":
                i += 1
                guard i < args.count, let v = UInt64(args[i]) else {
                    printStderr("--probe-chaining-request-transaction-handle requires a non-negative integer argument")
                    exit(1)
                }
                opts.probeChainingRequestTransactionHandle = v
            case "--probe-chaining-request-fw-enqueue-delay":
                i += 1
                guard i < args.count, let v = UInt64(args[i]) else {
                    printStderr("--probe-chaining-request-fw-enqueue-delay requires a non-negative integer argument")
                    exit(1)
                }
                opts.probeChainingRequestFWEnqueueDelay = v
            case "--probe-chaining-request-memory-pool-id":
                i += 1
                guard i < args.count, let v = UInt64(args[i]) else {
                    printStderr("--probe-chaining-request-memory-pool-id requires a non-negative integer argument")
                    exit(1)
                }
                opts.probeChainingRequestMemoryPoolId = v
            case "--probe-chaining-enqueue-procedure-index":
                i += 1
                guard i < args.count, let v = UInt32(args[i]) else {
                    printStderr("--probe-chaining-enqueue-procedure-index requires a non-negative integer argument")
                    exit(1)
                }
                opts.probeChainingEnqueueProcedureIndex = v
            case "--probe-chaining-enqueue-set-index":
                i += 1
                guard i < args.count, let v = UInt32(args[i]) else {
                    printStderr("--probe-chaining-enqueue-set-index requires a non-negative integer argument")
                    exit(1)
                }
                opts.probeChainingEnqueueSetIndex = v
            case "--probe-chaining-enqueue-signal-value":
                i += 1
                guard i < args.count, let v = UInt64(args[i]) else {
                    printStderr("--probe-chaining-enqueue-signal-value requires a non-negative integer argument")
                    exit(1)
                }
                opts.probeChainingEnqueueSignalValue = v
            case "--probe-chaining-enqueue-require-signal":
                opts.probeChainingEnqueueSignalNotRequired = false
            case "--probe-chaining-enqueue-open-loop":
                opts.probeChainingEnqueueOpenLoop = true
            case "--probe-chaining-ready-procedure-index":
                i += 1
                guard i < args.count, let v = UInt32(args[i]) else {
                    printStderr("--probe-chaining-ready-procedure-index requires a non-negative integer argument")
                    exit(1)
                }
                opts.probeChainingReadyProcedureIndex = v
            case "--probe-chaining-ready-execution-delay":
                i += 1
                guard i < args.count, let v = UInt64(args[i]) else {
                    printStderr("--probe-chaining-ready-execution-delay requires a non-negative integer argument")
                    exit(1)
                }
                opts.probeChainingReadyExecutionDelay = v
            case "--probe-chaining-use-shared-signal-event":
                opts.probeChainingUseSharedSignalEvent = true
            case "--probe-chaining-shared-signal-value":
                i += 1
                guard i < args.count, let v = UInt64(args[i]) else {
                    printStderr("--probe-chaining-shared-signal-value requires a non-negative integer argument")
                    exit(1)
                }
                opts.probeChainingSharedSignalEventValue = v
            case "--probe-chaining-shared-signal-symbol-index":
                i += 1
                guard i < args.count, let v = UInt32(args[i]) else {
                    printStderr("--probe-chaining-shared-signal-symbol-index requires a non-negative integer argument")
                    exit(1)
                }
                opts.probeChainingSharedSignalEventSymbolIndex = v
            case "--probe-chaining-shared-signal-event-type":
                i += 1
                guard i < args.count, let v = Int64(args[i]) else {
                    printStderr("--probe-chaining-shared-signal-event-type requires an integer argument")
                    exit(1)
                }
                opts.probeChainingSharedSignalEventType = v
            case "--help", "-h":
                printUsage()
                exit(0)
            default:
                printStderr("Unknown argument: \(args[i])")
                printUsage()
                exit(1)
            }
            i += 1
        }
        return opts
    }

    static func printUsage() {
        print("""
        EspressoBench — ANE Runtime Benchmark Suite

        Usage: espresso-bench [OPTIONS]

        Options:
          --ane-only         Skip Core ML benchmarks
          --inference        Run inference-optimized forward pass (fused residuals)
          --decode           Run autoregressive decode benchmark with KV cache
          --inference-fp16-handoff  Use FP16 surface-to-surface handoff between attn -> ffn
          --inference-only   Run inference benchmark only (skips training benchmark to save compile budget)
          --decode-steps N   Decode tokens per sequence (default: 32)
          --decode-max-seq N Decode KV-cache max sequence length (default: 32)
                            (must be >= decode lane width and currently a multiple of that width)
          --profile-kernels  Record per-kernel stage timing (us) and save CSV
          --sustained        Run 60-second sustained thermal test
          --warmup N         Warmup iterations (default: 50)
          --iterations N     Measured iterations (default: 1000)
          --output DIR       Output directory for results
          --model PATH       Path to Core ML .mlpackage
          --layers N         Number of transformer layers (default: 1)
          --dump-ane         Dump ObjC runtime methods/properties for private ANE classes
          --dump-ane-filter S  Only show methods/properties containing substring S
          --perf-stats       Enable `_ANEPerformanceStats` collection for direct ANE eval (sets ANE_PERF_STATS=1)
          --probe-chaining-prepare  Run isolated chaining prepare probe with primary output stats surface
          --probe-chaining-skip-prepare  Stop the isolated chaining probe before `prepareChainingWithModel`
          --probe-chaining-call-buffers-ready  Call `buffersReadyWithModel` before prepare in the isolated chaining probe
          --probe-chaining-call-enqueue-sets  Call `enqueueSetsWithModel` before prepare in the isolated chaining probe
          --probe-chaining-scalar-loopback  Pass scalar loopback symbol ids instead of arrays
          --probe-chaining-stats-surface MODE  Set chaining stats surface mode: output0, null, scratch (default: output0)
          --probe-chaining-request-procedure-index N  Override chaining request procedureIndex (default: 0)
          --probe-chaining-request-transaction-handle N  Override chaining request transactionHandle (default: 0)
          --probe-chaining-request-fw-enqueue-delay N  Override chaining request fwEnqueueDelay (default: 0)
          --probe-chaining-request-memory-pool-id N  Override chaining request memoryPoolId (default: 0)
          --probe-chaining-enqueue-procedure-index N  Override enqueueSets procedureIndex (default: 0)
          --probe-chaining-enqueue-set-index N  Override enqueueSets setIndex (default: 0)
          --probe-chaining-enqueue-signal-value N  Override enqueueSets signalValue (default: 0)
          --probe-chaining-enqueue-require-signal  Set enqueueSets signalNotRequired=false
          --probe-chaining-enqueue-open-loop  Set enqueueSets isOpenLoop=true
          --probe-chaining-ready-procedure-index N  Override buffersReady procedureIndex (default: 0)
          --probe-chaining-ready-execution-delay N  Override buffersReady executionDelay (default: 0)
          --probe-chaining-use-shared-signal-event  Build a `_ANESharedSignalEvent` and pass it in chaining signalEvents
          --probe-chaining-shared-signal-value N  Override shared signal event value (default: 1)
          --probe-chaining-shared-signal-symbol-index N  Override shared signal event symbolIndex (default: 0)
          --probe-chaining-shared-signal-event-type N  Override shared signal event eventType (default: 0)
          -h, --help         Show this help
        """)
    }
}

// MARK: - Main

let opts = BenchmarkOptions.parse(CommandLine.arguments)

private let chainingProbeChannels = 4
private let chainingProbeSpatial = 8
private let chainingProbeBytes = chainingProbeChannels * chainingProbeSpatial * MemoryLayout<UInt16>.stride

func chainingProbeWeightBlob(channels: Int) -> Data {
    var weights = [Float](repeating: 0, count: channels * channels)
    for i in 0..<channels {
        weights[i * channels + i] = 1
    }
    return WeightBlob.build(from: weights, rows: channels, cols: channels)
}

func makeChainingProbeKernel() throws -> ANEKernel {
    let mil = GenericMIL.conv(inCh: chainingProbeChannels, outCh: chainingProbeChannels, spatial: chainingProbeSpatial)
    return try ANEKernel(
        milText: mil,
        weights: [(path: "@model_path/weights/weight.bin", data: chainingProbeWeightBlob(channels: chainingProbeChannels))],
        inputBytes: chainingProbeBytes,
        outputBytes: chainingProbeBytes
    )
}

func chainingProbeJSON(_ probe: ANEChainingProbe) -> [String: Any] {
    [
        "has_chaining_request_class": probe.hasChainingRequestClass,
        "has_prepare_selector": probe.hasPrepareSelector,
        "has_output_sets_factory": probe.hasOutputSetsFactory,
        "has_output_set_enqueue_class": probe.hasOutputSetEnqueueClass,
        "has_input_buffers_ready_class": probe.hasInputBuffersReadyClass,
        "has_shared_signal_event_class": probe.hasSharedSignalEventClass,
        "built_output_set": probe.builtOutputSet,
        "built_output_set_enqueue": probe.builtOutputSetEnqueue,
        "built_input_buffers_ready": probe.builtInputBuffersReady,
        "built_shared_signal_event": probe.builtSharedSignalEvent,
        "built_request": probe.builtRequest,
        "used_array_loopback_symbol_indices": probe.usedArrayLoopbackSymbolIndices,
        "used_real_stats_surface": probe.usedRealStatsSurface,
        "request_validated": probe.requestValidated,
        "request_valid": probe.requestValid,
        "request_validation_failed": probe.requestValidationFailed,
        "input_buffers_ready_validation_failed": probe.inputBuffersReadyValidationFailed,
        "called_buffers_ready": probe.calledBuffersReady,
        "buffers_ready_succeeded": probe.buffersReadySucceeded,
        "called_enqueue_sets": probe.calledEnqueueSets,
        "enqueue_sets_succeeded": probe.enqueueSetsSucceeded,
        "prepared": probe.prepared,
        "stage": probe.stage,
    ]
}

func printJSONObject(_ object: Any) throws {
    let data = try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys])
    if let text = String(data: data, encoding: .utf8) {
        print(text)
    } else {
        throw NSError(domain: "EspressoBench.JSON", code: 1)
    }
}

func resolveOutputDir(_ opts: BenchmarkOptions) -> String {
    if let dir = opts.outputDir {
        return dir
    }
    let dateFormatter = DateFormatter()
    dateFormatter.dateFormat = "yyyy-MM-dd-HHmmss"
    dateFormatter.locale = Locale(identifier: "en_US_POSIX")
    let timestamp = dateFormatter.string(from: Date())
    return "benchmarks/results/\(timestamp)"
}

struct ChainingPrepareProbeConfig {
    let statsSurfaceMode: String
    let useRealStatsSurface: Bool
    let skipPrepare: Bool
    let callBuffersReady: Bool
    let callEnqueueSets: Bool
    let useScalarLoopbackSymbolIndices: Bool
    let requestProcedureIndex: UInt32
    let requestTransactionHandle: UInt64
    let requestFWEnqueueDelay: UInt64
    let requestMemoryPoolId: UInt64
    let enqueueProcedureIndex: UInt32
    let enqueueSetIndex: UInt32
    let enqueueSignalValue: UInt64
    let enqueueSignalNotRequired: Bool
    let enqueueOpenLoop: Bool
    let readyProcedureIndex: UInt32
    let readyExecutionDelay: UInt64
    let useSharedSignalEvent: Bool
    let sharedSignalEventValue: UInt64
    let sharedSignalEventSymbolIndex: UInt32
    let sharedSignalEventType: Int64

    init(options: BenchmarkOptions) {
        statsSurfaceMode = options.probeChainingStatsSurfaceMode
        useRealStatsSurface = options.probeChainingStatsSurfaceMode == "output0"
        skipPrepare = options.probeChainingSkipPrepare
        callBuffersReady = options.probeChainingCallBuffersReady
        callEnqueueSets = options.probeChainingCallEnqueueSets
        useScalarLoopbackSymbolIndices = options.probeChainingScalarLoopback
        requestProcedureIndex = options.probeChainingRequestProcedureIndex
        requestTransactionHandle = options.probeChainingRequestTransactionHandle
        requestFWEnqueueDelay = options.probeChainingRequestFWEnqueueDelay
        requestMemoryPoolId = options.probeChainingRequestMemoryPoolId
        enqueueProcedureIndex = options.probeChainingEnqueueProcedureIndex
        enqueueSetIndex = options.probeChainingEnqueueSetIndex
        enqueueSignalValue = options.probeChainingEnqueueSignalValue
        enqueueSignalNotRequired = options.probeChainingEnqueueSignalNotRequired
        enqueueOpenLoop = options.probeChainingEnqueueOpenLoop
        readyProcedureIndex = options.probeChainingReadyProcedureIndex
        readyExecutionDelay = options.probeChainingReadyExecutionDelay
        useSharedSignalEvent = options.probeChainingUseSharedSignalEvent
        sharedSignalEventValue = options.probeChainingSharedSignalEventValue
        sharedSignalEventSymbolIndex = options.probeChainingSharedSignalEventSymbolIndex
        sharedSignalEventType = options.probeChainingSharedSignalEventType
    }

    func applyEnvironment() {
        if useRealStatsSurface {
            unsetenv("ANE_INTEROP_CHAINING_PROBE_STATS_SURFACE")
        } else {
            setenv("ANE_INTEROP_CHAINING_PROBE_STATS_SURFACE", statsSurfaceMode, 1)
        }
    }
}

func chainingPrepareProbeSummary(
    options: BenchmarkOptions,
    probeConfig: ChainingPrepareProbeConfig,
    outputDirectory: String,
    resultStatus: String,
    resultMetrics: [String: Any] = [:],
    probe: ANEChainingProbe? = nil,
    error: String? = nil
) -> [String: Any] {
    var summary = RunMetadata.base(mode: "chaining-prepare-probe", options: options, outputDirectory: outputDirectory)
    summary["probe_options"] = [
        "stats_surface_mode": probeConfig.statsSurfaceMode,
        "use_real_stats_surface": probeConfig.useRealStatsSurface,
        "skip_prepare": probeConfig.skipPrepare,
        "call_buffers_ready": probeConfig.callBuffersReady,
        "call_enqueue_sets": probeConfig.callEnqueueSets,
        "scalar_loopback": probeConfig.useScalarLoopbackSymbolIndices,
        "request_procedure_index": probeConfig.requestProcedureIndex,
        "request_transaction_handle": probeConfig.requestTransactionHandle,
        "request_fw_enqueue_delay": probeConfig.requestFWEnqueueDelay,
        "request_memory_pool_id": probeConfig.requestMemoryPoolId,
        "enqueue_procedure_index": probeConfig.enqueueProcedureIndex,
        "enqueue_set_index": probeConfig.enqueueSetIndex,
        "enqueue_signal_value": probeConfig.enqueueSignalValue,
        "enqueue_signal_not_required": probeConfig.enqueueSignalNotRequired,
        "enqueue_open_loop": probeConfig.enqueueOpenLoop,
        "ready_procedure_index": probeConfig.readyProcedureIndex,
        "ready_execution_delay": probeConfig.readyExecutionDelay,
        "use_shared_signal_event": probeConfig.useSharedSignalEvent,
        "shared_signal_event_value": probeConfig.sharedSignalEventValue,
        "shared_signal_event_symbol_index": probeConfig.sharedSignalEventSymbolIndex,
        "shared_signal_event_type": probeConfig.sharedSignalEventType,
        "validate_request": true,
    ]
    var result: [String: Any] = ["status": resultStatus]
    for (key, value) in resultMetrics {
        result[key] = value
    }
    if let error {
        result["error"] = error
    }
    summary["result"] = result
    if let probe {
        summary["probe"] = chainingProbeJSON(probe)
    }
    return summary
}

func writeChainingPrepareProbeSummary(_ summary: [String: Any], outputDir: String) throws {
    try FileManager.default.createDirectory(atPath: outputDir, withIntermediateDirectories: true)
    try RunMetadata.writeJSON(summary, to: "\(outputDir)/summary.json")
}

if opts.dumpANE {
    _ = dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW)
    let classes = [
        "_ANEInMemoryModelDescriptor",
        "_ANEInMemoryModel",
        "_ANERequest",
        "_ANEIOSurfaceObject",
        "_ANEPerformanceStats",
        "_ANEPerformanceCounters",
        "_ANEClient",
        "_ANEVirtualClient",
        "_ANEChainingRequest",
    ]
    for c in classes {
        print(ANEIntrospector.dump(className: c, filter: opts.dumpANEFilter))
    }
    exit(0)
}

if opts.perfStats {
    setenv("ANE_PERF_STATS", "1", 1)
}

if opts.probeChainingPrepare {
    let outputDir = resolveOutputDir(opts)
    let probeConfig = ChainingPrepareProbeConfig(options: opts)
    probeConfig.applyEnvironment()
    let startedSummary = chainingPrepareProbeSummary(
        options: opts,
        probeConfig: probeConfig,
        outputDirectory: outputDir,
        resultStatus: "started"
    )
    try? writeChainingPrepareProbeSummary(startedSummary, outputDir: outputDir)
    printStderr("chaining-prepare-probe: started")
    let compileStart = CFAbsoluteTimeGetCurrent()
    do {
        let kernel = try makeChainingProbeKernel()
        let compileElapsedMS = (CFAbsoluteTimeGetCurrent() - compileStart) * 1000.0
        let kernelCompiledSummary = chainingPrepareProbeSummary(
            options: opts,
            probeConfig: probeConfig,
            outputDirectory: outputDir,
            resultStatus: "kernel_compiled",
            resultMetrics: [
                "compile_elapsed_ms": compileElapsedMS,
            ]
        )
        try? writeChainingPrepareProbeSummary(kernelCompiledSummary, outputDir: outputDir)
        let probeStart = CFAbsoluteTimeGetCurrent()
        let probe = kernel.chainingProbe(
            useRealStatsSurface: probeConfig.useRealStatsSurface,
            skipPrepare: probeConfig.skipPrepare,
            validateRequest: true,
            useScalarLoopbackSymbolIndices: probeConfig.useScalarLoopbackSymbolIndices,
            callEnqueueSets: probeConfig.callEnqueueSets,
            callBuffersReady: probeConfig.callBuffersReady,
            requestProcedureIndex: probeConfig.requestProcedureIndex,
            requestTransactionHandle: probeConfig.requestTransactionHandle,
            requestFWEnqueueDelay: probeConfig.requestFWEnqueueDelay,
            requestMemoryPoolId: probeConfig.requestMemoryPoolId,
            enqueueProcedureIndex: probeConfig.enqueueProcedureIndex,
            enqueueSetIndex: probeConfig.enqueueSetIndex,
            enqueueSignalValue: probeConfig.enqueueSignalValue,
            enqueueSignalNotRequired: probeConfig.enqueueSignalNotRequired,
            enqueueOpenLoop: probeConfig.enqueueOpenLoop,
            readyProcedureIndex: probeConfig.readyProcedureIndex,
            readyExecutionDelay: probeConfig.readyExecutionDelay,
            useSharedSignalEvent: probeConfig.useSharedSignalEvent,
            sharedSignalEventValue: probeConfig.sharedSignalEventValue,
            sharedSignalEventSymbolIndex: probeConfig.sharedSignalEventSymbolIndex,
            sharedSignalEventType: probeConfig.sharedSignalEventType
        )
        let probeElapsedMS = (CFAbsoluteTimeGetCurrent() - probeStart) * 1000.0
        let completedSummary = chainingPrepareProbeSummary(
            options: opts,
            probeConfig: probeConfig,
            outputDirectory: outputDir,
            resultStatus: "completed",
            resultMetrics: [
                "compile_elapsed_ms": compileElapsedMS,
                "probe_elapsed_ms": probeElapsedMS,
            ],
            probe: probe
        )
        try writeChainingPrepareProbeSummary(completedSummary, outputDir: outputDir)
        printStderr("chaining-prepare-probe: completed")
        try printJSONObject(completedSummary)
        exit(0)
    } catch {
        let compileElapsedMS = (CFAbsoluteTimeGetCurrent() - compileStart) * 1000.0
        let failedSummary = chainingPrepareProbeSummary(
            options: opts,
            probeConfig: probeConfig,
            outputDirectory: outputDir,
            resultStatus: "failed",
            resultMetrics: [
                "compile_elapsed_ms": compileElapsedMS,
            ],
            error: String(describing: error)
        )
        try? writeChainingPrepareProbeSummary(failedSummary, outputDir: outputDir)
        printStderr("chaining-prepare-probe: failed: \(error)")
        try? printJSONObject(failedSummary)
        exit(1)
    }
}

let runner = BenchmarkRunner(warmup: opts.warmup, iterations: opts.iterations)

let flopsPerPass = FLOPCalculator.forwardPassFLOPs() * Double(opts.nLayers)
printStderr("Espresso Benchmark Suite")
printStderr("========================")
printStderr(String(format: "Config: dim=%d, hidden=%d, seq=%d, heads=%d, layers=%d",
                   ModelConfig.dim, ModelConfig.hidden, ModelConfig.seqLen, ModelConfig.heads, opts.nLayers))
printStderr(String(format: "FLOPs per forward pass: %.2f GFLOPs", flopsPerPass / 1e9))
printStderr(String(format: "Iterations: %d warmup + %d measured", opts.warmup, opts.iterations))
printStderr("")

let handoff: ForwardPass.InferenceInterKernelHandoff = opts.inferenceFP16Handoff ? .fp16SurfaceCopy : .cpuRoundTrip

func benchmarkStats(_ result: BenchmarkResult) -> [String: Any] {
    [
        "mean_ms": result.mean,
        "median_ms": result.median,
        "p95_ms": result.p95,
        "p99_ms": result.p99,
        "min_ms": result.min,
        "max_ms": result.max,
        "stddev_ms": result.stddev,
        "warmup_count": result.warmupCount,
        "iteration_count": result.iterationCount,
        "tokens_per_second": result.mean > 0 ? 1000.0 / result.mean : 0,
    ]
}

func artifactManifest(_ filenames: [String]) -> [[String: String]] {
    Array(Set(filenames))
        .sorted()
        .map { ["path": $0] }
}

func inferenceProfileAverages(_ profile: InferenceKernelProfile) -> [[String: Any]] {
    profile.layers.indices.map { layerIdx in
        let mean = profile.averageLayerMetrics(layerIndex: layerIdx)
        let layer = profile.layers[layerIdx]
        let hasHWExecutionTime = layer.attnHwNS.contains(where: { $0 > 0 }) || layer.ffnHwNS.contains(where: { $0 > 0 })
        return [
            "layer": layerIdx,
            "samples": mean.sampleCount,
            "hw_execution_time_available": hasHWExecutionTime,
            "attn_eval_host_us": mean.attnEvalUS,
            "attn_hw_us": mean.attnHwUS,
            "attn_host_overhead_us": mean.attnHostOverheadUS,
            "attn_io_lock_us": mean.attnIOLockUS,
            "attn_io_body_us": mean.attnIOBodyUS,
            "attn_io_unlock_us": mean.attnIOUnlockUS,
            "gap_attn_to_ffn_us": mean.gapAttnToFfnUS,
            "handoff_cpu_roundtrip_us": mean.handoffCPUUS,
            "handoff_fp16_copy_us": mean.handoffFP16CopyUS,
            "ffn_eval_host_us": mean.ffnEvalUS,
            "ffn_hw_us": mean.ffnHwUS,
            "ffn_host_overhead_us": mean.ffnHostOverheadUS,
            "ffn_io_lock_us": mean.ffnIOLockUS,
            "ffn_io_body_us": mean.ffnIOBodyUS,
            "ffn_io_unlock_us": mean.ffnIOUnlockUS,
        ] as [String: Any]
    }
}

func decodeProfileAverages(_ profile: DecodeKernelProfile) -> [[String: Any]] {
    func mean(_ values: [Double]) -> Double {
        guard !values.isEmpty else { return 0 }
        return values.reduce(0, +) / Double(values.count)
    }

    func meanInt(_ values: [Int]) -> Double {
        guard !values.isEmpty else { return 0 }
        return Double(values.reduce(0, +)) / Double(values.count)
    }

    var layerDicts: [[String: Any]] = []
    for layerIdx in 0..<profile.layers.count {
        let layer = profile.layers[layerIdx]
        let dict: [String: Any] = [
            "layer": layerIdx,
            "samples": layer.attnEvalUS.count,
            "attn_eval_us_avg": mean(layer.attnEvalUS),
            "ffn_eval_us_avg": mean(layer.ffnEvalUS),
        ]
        layerDicts.append(dict)
    }
    return layerDicts
}

if opts.decode {
    let decodeResult: ANEDirectBench.Result
    do {
        decodeResult = try ANEDirectBench.runDecode(
            warmup: opts.warmup,
            iterations: opts.iterations,
            decodeSteps: opts.decodeSteps,
            decodeMaxSeq: opts.decodeMaxSeq,
            nLayers: opts.nLayers,
            profileKernels: opts.profileKernels
        )
    } catch {
        printStderr("ANE decode benchmark failed: \(error)")
        exit(1)
    }

    var coreMLDecodeResult: CoreMLBench.Result? = nil
    if !opts.aneOnly {
        do {
            coreMLDecodeResult = try CoreMLBench.runNaiveDecode(
                warmup: opts.warmup,
                iterations: opts.iterations,
                decodeSteps: opts.decodeSteps,
                decodeMaxSeq: opts.decodeMaxSeq,
                modelPath: opts.coreMLModelPath
            )
        } catch {
            printStderr("Core ML decode benchmark failed: \(error)")
            printStderr("Continuing with ANE-only decode results...")
        }
    }

    let report = ResultsFormatter.formatDecodeReport(
        decodeResult: decodeResult.benchmarkResult,
        decodeTimingBreakdown: decodeResult.avgTimingBreakdown,
        decodeCompileTimeMs: decodeResult.compileTimeMs,
        decodeTokensPerSecond: decodeResult.tokensPerSecond,
        coreMLDecodeResults: coreMLDecodeResult?.results,
        coreMLLoadTimeMs: coreMLDecodeResult?.modelLoadTimeMs,
        nLayers: opts.nLayers,
        decodeSteps: opts.decodeSteps,
        decodeMaxSeq: opts.decodeMaxSeq
    )
    print(report)

    let outputDir: String
    if let dir = opts.outputDir {
        outputDir = dir
    } else {
        let dateFormatter = DateFormatter()
        dateFormatter.dateFormat = "yyyy-MM-dd-HHmmss"
        dateFormatter.locale = Locale(identifier: "en_US_POSIX")
        let timestamp = dateFormatter.string(from: Date())
        outputDir = "benchmarks/results/\(timestamp)"
    }

    do {
        try FileManager.default.createDirectory(atPath: outputDir, withIntermediateDirectories: true)
        var artifacts: [String] = []
        try ResultsFormatter.writeCSV(
            latencies: decodeResult.benchmarkResult.latencies,
            to: "\(outputDir)/ane_decode_token_latencies.csv"
        )
        artifacts.append("ane_decode_token_latencies.csv")
        if let profile = decodeResult.decodeKernelProfile {
            try ResultsFormatter.writeDecodeKernelProfileCSV(
                profile: profile,
                to: "\(outputDir)/ane_decode_kernel_profile.csv"
            )
            artifacts.append("ane_decode_kernel_profile.csv")
        }
        if let coreML = coreMLDecodeResult {
            for (label, result) in coreML.results {
                let filename = label.lowercased()
                    .replacingOccurrences(of: " ", with: "_")
                    .replacingOccurrences(of: "(", with: "")
                    .replacingOccurrences(of: ")", with: "")
                    .replacingOccurrences(of: ".", with: "")
                try ResultsFormatter.writeCSV(
                    latencies: result.latencies,
                    to: "\(outputDir)/\(filename)_token_latencies.csv"
                )
                artifacts.append("\(filename)_token_latencies.csv")
            }
        }
        try report.write(toFile: "\(outputDir)/summary.txt", atomically: true, encoding: .utf8)
        artifacts.append("summary.txt")
        var summaryJSON = RunMetadata.base(mode: "decode", options: opts, outputDirectory: outputDir)
        summaryJSON["ane_decode"] = [
            "compile_time_ms": decodeResult.compileTimeMs,
            "timing_breakdown_ms": [
                "ane": decodeResult.avgTimingBreakdown.ane,
                "io": decodeResult.avgTimingBreakdown.io,
                "elem": decodeResult.avgTimingBreakdown.elem,
            ],
            "metrics": benchmarkStats(decodeResult.benchmarkResult),
        ]
        summaryJSON["decode_chain_mode"] = ProcessInfo.processInfo.environment["ESPRESSO_DECODE_CHAIN_MODE"] ?? "off"
        if let profile = decodeResult.decodeKernelProfile {
            summaryJSON["ane_decode_profile"] = decodeProfileAverages(profile)
        }
        if let coreML = coreMLDecodeResult {
            let entries = coreML.results.map { (label, result) in
                [
                    "label": label,
                    "metrics": benchmarkStats(result),
                ] as [String: Any]
            }
            summaryJSON["coreml_decode"] = [
                "model_load_time_ms": coreML.modelLoadTimeMs,
                "results": entries,
            ]
            if let fastest = coreML.results.min(by: { $0.result.median < $1.result.median })?.result {
                summaryJSON["speedup_vs_fastest_coreml_decode"] = fastest.median / decodeResult.benchmarkResult.median
            }
        }
        let manifest = artifactManifest(artifacts + ["summary.json"])
        summaryJSON["artifacts"] = manifest
        summaryJSON["artifact_count"] = manifest.count
        try RunMetadata.writeJSON(summaryJSON, to: "\(outputDir)/summary.json")
        printStderr("\nResults saved to: \(outputDir)/")
    } catch {
        printStderr("Failed to save decode results: \(error)")
    }

    exit(0)
}

if opts.inferenceOnly {
    // --- Inference-only flow (skips training compile budget) ---
    let thermalBefore = ThermalMonitor.currentState()

    let inferenceResult: ANEDirectBench.Result
    do {
        inferenceResult = try ANEDirectBench.runInference(
            warmup: opts.warmup,
            iterations: opts.iterations,
            nLayers: opts.nLayers,
            handoff: handoff,
            profileKernels: opts.profileKernels
        )
    } catch {
        printStderr("ANE Inference benchmark failed: \(error)")
        exit(1)
    }

    // Core ML (optional)
    var coreMLResult: CoreMLBench.Result? = nil
    if !opts.aneOnly {
        do {
            coreMLResult = try CoreMLBench.run(runner: runner, modelPath: opts.coreMLModelPath)
        } catch {
            printStderr("Core ML benchmark failed: \(error)")
            printStderr("Continuing with ANE-only results...")
        }
    }

    let thermalAfter = ThermalMonitor.currentState()

    var report = ResultsFormatter.formatInferenceOnlyReport(
        inferenceResult: inferenceResult.benchmarkResult,
        inferenceTimingBreakdown: inferenceResult.avgTimingBreakdown,
        inferenceCompileTimeMs: inferenceResult.compileTimeMs,
        coreMLResults: coreMLResult?.results,
        coreMLLoadTimeMs: coreMLResult?.modelLoadTimeMs,
        flopsPerPass: flopsPerPass,
        nLayers: opts.nLayers,
        thermalBefore: thermalBefore,
        thermalAfter: thermalAfter
    )
    if let profile = inferenceResult.kernelProfile {
        report += ResultsFormatter.formatInferenceKernelProfileSummaryTable(profile: profile, handoff: handoff)
    }
    print(report)

    // Save results
    let outputDir: String
    if let dir = opts.outputDir {
        outputDir = dir
    } else {
        let dateFormatter = DateFormatter()
        dateFormatter.dateFormat = "yyyy-MM-dd-HHmmss"
        dateFormatter.locale = Locale(identifier: "en_US_POSIX")
        let timestamp = dateFormatter.string(from: Date())
        outputDir = "benchmarks/results/\(timestamp)"
    }

    do {
        try FileManager.default.createDirectory(atPath: outputDir, withIntermediateDirectories: true)
        var artifacts: [String] = []

        try ResultsFormatter.writeCSV(
            latencies: inferenceResult.benchmarkResult.latencies,
            to: "\(outputDir)/ane_inference_latencies.csv"
        )
        artifacts.append("ane_inference_latencies.csv")
        if let profile = inferenceResult.kernelProfile {
            try ResultsFormatter.writeInferenceKernelProfileCSV(
                profile: profile,
                to: "\(outputDir)/ane_inference_kernel_profile.csv"
            )
            artifacts.append("ane_inference_kernel_profile.csv")
        }
        if let coreML = coreMLResult {
            for (label, result) in coreML.results {
                let filename = label.lowercased()
                    .replacingOccurrences(of: " ", with: "_")
                    .replacingOccurrences(of: "(", with: "")
                    .replacingOccurrences(of: ")", with: "")
                    .replacingOccurrences(of: ".", with: "")
                try ResultsFormatter.writeCSV(
                    latencies: result.latencies,
                    to: "\(outputDir)/\(filename)_latencies.csv"
                )
                artifacts.append("\(filename)_latencies.csv")
            }
        }

        try report.write(toFile: "\(outputDir)/summary.txt", atomically: true, encoding: .utf8)
        artifacts.append("summary.txt")
        var summaryJSON = RunMetadata.base(mode: "inference-only", options: opts, outputDirectory: outputDir)
        var inferenceEntry: [String: Any] = [
            "compile_time_ms": inferenceResult.compileTimeMs,
            "timing_breakdown_ms": [
                "ane": inferenceResult.avgTimingBreakdown.ane,
                "io": inferenceResult.avgTimingBreakdown.io,
                "elem": inferenceResult.avgTimingBreakdown.elem,
            ],
            "metrics": benchmarkStats(inferenceResult.benchmarkResult),
        ]
        if let profile = inferenceResult.kernelProfile {
            inferenceEntry["kernel_profile_layer_averages_us"] = inferenceProfileAverages(profile)
        }
        summaryJSON["ane_inference"] = inferenceEntry
        if let coreML = coreMLResult {
            summaryJSON["coreml"] = [
                "model_load_time_ms": coreML.modelLoadTimeMs,
                "results": coreML.results.map { entry in
                    [
                        "label": entry.label,
                        "metrics": benchmarkStats(entry.result),
                    ] as [String: Any]
                },
            ]
        }
        summaryJSON["thermal"] = [
            "before": thermalBefore,
            "after": thermalAfter,
        ]
        let manifest = artifactManifest(artifacts + ["summary.json"])
        summaryJSON["artifacts"] = manifest
        summaryJSON["artifact_count"] = manifest.count
        try RunMetadata.writeJSON(summaryJSON, to: "\(outputDir)/summary.json")
        printStderr("\nResults saved to: \(outputDir)/")
    } catch {
        printStderr("Failed to save results: \(error)")
    }

    exit(0)
}

// --- Default flow: training + (optional) inference + (optional) coreml ---

// Benchmark 1: ANE Direct (training forward pass)
let aneResult: ANEDirectBench.Result
do {
    aneResult = try ANEDirectBench.run(warmup: opts.warmup, iterations: opts.iterations, nLayers: opts.nLayers)
} catch {
    printStderr("ANE Direct benchmark failed: \(error)")
    exit(1)
}

// Benchmark 1b: ANE Direct Inference (optional, fused residuals)
var inferenceResult: ANEDirectBench.Result? = nil
if opts.inference {
    do {
        inferenceResult = try ANEDirectBench.runInference(
            warmup: opts.warmup,
            iterations: opts.iterations,
            nLayers: opts.nLayers,
            handoff: handoff,
            profileKernels: opts.profileKernels
        )
    } catch {
        printStderr("ANE Inference benchmark failed: \(error)")
        printStderr("Continuing without inference results...")
    }
}

// Benchmark 2: Core ML (optional)
var coreMLResult: CoreMLBench.Result? = nil
if !opts.aneOnly {
    do {
        coreMLResult = try CoreMLBench.run(runner: runner, modelPath: opts.coreMLModelPath)
    } catch {
        printStderr("Core ML benchmark failed: \(error)")
        printStderr("Continuing with ANE-only results...")
    }
}

// Benchmark 3: Sustained thermal test (optional)
var thermalBefore: String? = nil
var thermalAfter: String? = nil
if opts.sustained {
    printStderr("\n=== Sustained Thermal Test (60 seconds) ===")
    do {
        let thermal = try ANEDirectBench.runSustained(duration: 60.0, nLayers: opts.nLayers)
        thermalBefore = thermal.before
        thermalAfter = thermal.after
        for sample in thermal.samples {
            printStderr(String(format: "    t=%.0fs: %@", sample.time, sample.state))
        }
        printStderr("  Total forward passes: \(thermal.iterations)")
    } catch {
        printStderr("  Thermal test failed: \(error)")
    }
}

// Output report
let inferenceReportData: (result: BenchmarkResult, breakdown: (ane: Double, io: Double, elem: Double), compileMs: Double)?
if let inf = inferenceResult {
    inferenceReportData = (result: inf.benchmarkResult, breakdown: inf.avgTimingBreakdown, compileMs: inf.compileTimeMs)
} else {
    inferenceReportData = nil
}

var report = ResultsFormatter.formatReport(
    aneResult: aneResult.benchmarkResult,
    aneTimingBreakdown: aneResult.avgTimingBreakdown,
    compileTimeMs: aneResult.compileTimeMs,
    inferenceResult: inferenceReportData?.result,
    inferenceTimingBreakdown: inferenceReportData?.breakdown,
    inferenceCompileTimeMs: inferenceReportData?.compileMs,
    coreMLResults: coreMLResult?.results,
    coreMLLoadTimeMs: coreMLResult?.modelLoadTimeMs,
    thermalBefore: thermalBefore,
    thermalAfter: thermalAfter,
    flopsPerPass: flopsPerPass,
    nLayers: opts.nLayers
)
if let profile = inferenceResult?.kernelProfile {
    report += ResultsFormatter.formatInferenceKernelProfileSummaryTable(profile: profile, handoff: handoff)
}
print(report)

// Save results
let outputDir: String
if let dir = opts.outputDir {
    outputDir = dir
} else {
    let dateFormatter = DateFormatter()
    dateFormatter.dateFormat = "yyyy-MM-dd-HHmmss"
    dateFormatter.locale = Locale(identifier: "en_US_POSIX")
    let timestamp = dateFormatter.string(from: Date())
    outputDir = "benchmarks/results/\(timestamp)"
}

do {
    try FileManager.default.createDirectory(atPath: outputDir, withIntermediateDirectories: true)
    var artifacts: [String] = []

    try ResultsFormatter.writeCSV(
        latencies: aneResult.benchmarkResult.latencies,
        to: "\(outputDir)/ane_direct_latencies.csv"
    )
    artifacts.append("ane_direct_latencies.csv")

    if let inf = inferenceResult {
        try ResultsFormatter.writeCSV(
            latencies: inf.benchmarkResult.latencies,
            to: "\(outputDir)/ane_inference_latencies.csv"
        )
        artifacts.append("ane_inference_latencies.csv")
        if let profile = inf.kernelProfile {
            try ResultsFormatter.writeInferenceKernelProfileCSV(
                profile: profile,
                to: "\(outputDir)/ane_inference_kernel_profile.csv"
            )
            artifacts.append("ane_inference_kernel_profile.csv")
        }
    }

    if let coreML = coreMLResult {
        for (label, result) in coreML.results {
            let filename = label.lowercased()
                .replacingOccurrences(of: " ", with: "_")
                .replacingOccurrences(of: "(", with: "")
                .replacingOccurrences(of: ")", with: "")
                .replacingOccurrences(of: ".", with: "")
            try ResultsFormatter.writeCSV(
                latencies: result.latencies,
                to: "\(outputDir)/\(filename)_latencies.csv"
            )
            artifacts.append("\(filename)_latencies.csv")
        }
    }

    try report.write(toFile: "\(outputDir)/summary.txt", atomically: true, encoding: .utf8)
    artifacts.append("summary.txt")
    var summaryJSON = RunMetadata.base(mode: "full", options: opts, outputDirectory: outputDir)
    summaryJSON["ane_direct"] = [
        "compile_time_ms": aneResult.compileTimeMs,
        "timing_breakdown_ms": [
            "ane": aneResult.avgTimingBreakdown.ane,
            "io": aneResult.avgTimingBreakdown.io,
            "elem": aneResult.avgTimingBreakdown.elem,
        ],
        "metrics": benchmarkStats(aneResult.benchmarkResult),
    ]
    if let inf = inferenceResult {
        var inferenceEntry: [String: Any] = [
            "compile_time_ms": inf.compileTimeMs,
            "timing_breakdown_ms": [
                "ane": inf.avgTimingBreakdown.ane,
                "io": inf.avgTimingBreakdown.io,
                "elem": inf.avgTimingBreakdown.elem,
            ],
            "metrics": benchmarkStats(inf.benchmarkResult),
        ]
        if let profile = inf.kernelProfile {
            inferenceEntry["kernel_profile_layer_averages_us"] = inferenceProfileAverages(profile)
        }
        summaryJSON["ane_inference"] = inferenceEntry
    }
    if let coreML = coreMLResult {
        summaryJSON["coreml"] = [
            "model_load_time_ms": coreML.modelLoadTimeMs,
            "results": coreML.results.map { entry in
                [
                    "label": entry.label,
                    "metrics": benchmarkStats(entry.result),
                ] as [String: Any]
            },
        ]
    }
    if let thermalBefore, let thermalAfter {
        summaryJSON["thermal"] = [
            "before": thermalBefore,
            "after": thermalAfter,
        ]
    }
    let manifest = artifactManifest(artifacts + ["summary.json"])
    summaryJSON["artifacts"] = manifest
    summaryJSON["artifact_count"] = manifest.count
    try RunMetadata.writeJSON(summaryJSON, to: "\(outputDir)/summary.json")
    printStderr("\nResults saved to: \(outputDir)/")
} catch {
    printStderr("Failed to save results: \(error)")
}
