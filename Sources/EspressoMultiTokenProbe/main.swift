import Foundation
import Darwin
import CoreML
import ANERuntime
import ANETypes
import Espresso

private enum RunMode: String {
    case compare
    case compileInitOnly = "compile-init-only"
}

private struct Options {
    var mode: RunMode = .compare
    var warmup: Int = 3
    var iterations: Int = 20
    var maxNewTokens: Int = 8
    var maxSequenceTokens: Int = 32
    var layerCount: Int = 1
    var trunkLaneSpatial: Int = 32
    var outputHeadLaneSpatial: Int = 32
    var controlBackend: RecurrentGenerationTrunkBackend = .singleLayer
    var twoStepBackend: RecurrentGenerationTrunkBackend = .singleLayer
    var outputHeadBackend: GenerationOutputHeadBackend = .aneRMSNormClassifier
    var inputModeRaw: String? = nil
    var recurrentCheckpointPath: String? = nil
    var futureSidecarPath: String? = nil
    var compareCoreML: Bool = false
    var coreMLModelPath: String? = nil
    var generationModelPath: String? = nil
    var promptToken: UInt16 = 0

    static func parse(_ argv: [String]) -> Options {
        var options = Options()
        var idx = 1
        while idx < argv.count {
            switch argv[idx] {
            case "--mode":
                idx += 1
                guard idx < argv.count, let mode = RunMode(rawValue: argv[idx]) else {
                    fatal("Expected --mode compare|compile-init-only")
                }
                options.mode = mode
            case "--warmup":
                idx += 1
                options.warmup = parsePositiveInt(argv, idx: idx, flag: "--warmup")
            case "--iterations":
                idx += 1
                options.iterations = parsePositiveInt(argv, idx: idx, flag: "--iterations")
            case "--max-new-tokens":
                idx += 1
                options.maxNewTokens = parsePositiveInt(argv, idx: idx, flag: "--max-new-tokens")
            case "--max-sequence-tokens":
                idx += 1
                options.maxSequenceTokens = parsePositiveInt(argv, idx: idx, flag: "--max-sequence-tokens")
            case "--layer-count":
                idx += 1
                options.layerCount = parsePositiveInt(argv, idx: idx, flag: "--layer-count")
            case "--trunk-lane-spatial":
                idx += 1
                options.trunkLaneSpatial = parsePositiveInt(argv, idx: idx, flag: "--trunk-lane-spatial")
            case "--output-head-lane-spatial":
                idx += 1
                options.outputHeadLaneSpatial = parsePositiveInt(argv, idx: idx, flag: "--output-head-lane-spatial")
            case "--control-backend":
                idx += 1
                guard idx < argv.count else {
                    fatal("Expected --control-backend single|fused-pair|fused-triplet|identity-zero-trunk|identity-zero-trunk-lookup")
                }
                options.controlBackend = parseControlBackend(argv[idx])
            case "--two-step-backend":
                idx += 1
                guard idx < argv.count else {
                    fatal("Expected --two-step-backend single|fused-pair|fused-triplet|identity-zero-trunk|identity-zero-trunk-lookup")
                }
                options.twoStepBackend = parseControlBackend(argv[idx])
            case "--output-head-backend":
                idx += 1
                guard idx < argv.count else {
                    fatal("Expected --output-head-backend cpu|ane-classifier|ane-rmsnorm-classifier")
                }
                options.outputHeadBackend = parseOutputHeadBackend(argv[idx])
            case "--input":
                idx += 1
                guard idx < argv.count else {
                    fatal("Expected --input echo|recurrent-checkpoint")
                }
                options.inputModeRaw = argv[idx]
            case "--recurrent-checkpoint":
                idx += 1
                guard idx < argv.count else {
                    fatal("Expected --recurrent-checkpoint PATH")
                }
                options.recurrentCheckpointPath = argv[idx]
            case "--future-sidecar":
                idx += 1
                guard idx < argv.count else {
                    fatal("Expected --future-sidecar PATH")
                }
                options.futureSidecarPath = argv[idx]
            case "--compare-coreml":
                options.compareCoreML = true
            case "--coreml-model":
                idx += 1
                guard idx < argv.count else {
                    fatal("Expected --coreml-model PATH")
                }
                options.coreMLModelPath = argv[idx]
            case "--generation-model":
                idx += 1
                guard idx < argv.count else {
                    fatal("Expected --generation-model PATH")
                }
                options.generationModelPath = argv[idx]
            case "--prompt-token":
                idx += 1
                guard idx < argv.count, let promptToken = UInt16(argv[idx]) else {
                    fatal("Expected --prompt-token UINT16")
                }
                options.promptToken = promptToken
            case "--help":
                printUsageAndExit()
            default:
                fatal("Unknown argument: \(argv[idx])")
            }
            idx += 1
        }

        guard options.maxSequenceTokens >= options.maxNewTokens + 1 else {
            fatal("--max-sequence-tokens must be >= max-new-tokens + 1")
        }
        if options.controlBackend == .fusedTwoLayerPairs, !options.layerCount.isMultiple(of: 2) {
            fatal("fused-pair control backend requires even --layer-count")
        }
        if options.controlBackend == .fusedThreeLayerTriplets, !options.layerCount.isMultiple(of: 3) {
            fatal("fused-triplet control backend requires --layer-count multiple of 3")
        }
        if options.twoStepBackend == .fusedTwoLayerPairs, !options.layerCount.isMultiple(of: 2) {
            fatal("fused-pair two-step backend requires even --layer-count")
        }
        if options.twoStepBackend == .fusedThreeLayerTriplets, !options.layerCount.isMultiple(of: 3) {
            fatal("fused-triplet two-step backend requires --layer-count multiple of 3")
        }

        return options
    }

    func validatedProbeConfiguration() throws(MultitokenProbeConfigurationError) -> ValidatedMultitokenProbeConfiguration {
        let input: MultitokenProbeInput?
        switch inputModeRaw {
        case nil:
            input = nil
        case "echo":
            input = .echo
        case "recurrent-checkpoint":
            input = .recurrentCheckpoint(path: recurrentCheckpointPath ?? "")
        case let raw?:
            fatal("Unknown --input mode: \(raw)")
        }

        return try MultitokenProbeConfiguration(
            input: input,
            compareCoreML: compareCoreML,
            coreMLModelPath: coreMLModelPath,
            generationModelPath: generationModelPath
        ).validated()
    }
}

private enum ProbeError: Error {
    case invariantViolation(String)
}

@inline(__always)
private func fatal(_ message: String) -> Never {
    fputs("espresso-multitoken-probe error: \(message)\n", stderr)
    exit(1)
}

@inline(__always)
private func printStderr(_ message: String) {
    message.withCString { cstr in
        _ = fputs(cstr, stderr)
        _ = fputs("\n", stderr)
    }
}

@inline(__always)
private func machMilliseconds(_ delta: UInt64) -> Double {
    var info = mach_timebase_info_data_t()
    mach_timebase_info(&info)
    let nanos = (Double(delta) * Double(info.numer)) / Double(info.denom)
    return nanos / 1_000_000.0
}

@inline(__always)
private func parsePositiveInt(_ argv: [String], idx: Int, flag: String) -> Int {
    guard idx < argv.count, let value = Int(argv[idx]), value > 0 else {
        fatal("Expected positive integer for \(flag)")
    }
    return value
}

private func parseControlBackend(_ raw: String) -> RecurrentGenerationTrunkBackend {
    switch raw {
    case "single":
        return .singleLayer
    case "fused-pair":
        return .fusedTwoLayerPairs
    case "fused-triplet":
        return .fusedThreeLayerTriplets
    case "identity-zero-trunk":
        return .identityZeroTrunk
    case "identity-zero-trunk-lookup":
        return .identityZeroTrunkLookup
    default:
        fatal("Unknown control backend: \(raw)")
    }
}

private func parseOutputHeadBackend(_ raw: String) -> GenerationOutputHeadBackend {
    switch raw {
    case "cpu":
        return .cpu
    case "ane-classifier":
        return .aneClassifier
    case "ane-rmsnorm-classifier":
        return .aneRMSNormClassifier
    default:
        fatal("Unknown output-head backend: \(raw)")
    }
}

private func printUsageAndExit() -> Never {
    let usage = """
    Usage: espresso-multitoken-probe [options]
      --mode compare|compile-init-only
      --input echo|recurrent-checkpoint
      --recurrent-checkpoint PATH
      --future-sidecar PATH
      --compare-coreml
      --coreml-model PATH
      --generation-model PATH
      --prompt-token UINT16
      --warmup N
      --iterations N
      --max-new-tokens N
      --max-sequence-tokens N
      --layer-count N
      --control-backend single|fused-pair|fused-triplet|identity-zero-trunk|identity-zero-trunk-lookup
      --two-step-backend single|fused-pair|fused-triplet|identity-zero-trunk|identity-zero-trunk-lookup
      --output-head-backend cpu|ane-classifier|ane-rmsnorm-classifier
      --trunk-lane-spatial N
      --output-head-lane-spatial N
    """
    print(usage)
    exit(0)
}

private func median(_ values: [Double]) -> Double {
    guard !values.isEmpty else { return 0 }
    let sorted = values.sorted()
    let mid = sorted.count / 2
    if sorted.count.isMultiple(of: 2) {
        return (sorted[mid - 1] + sorted[mid]) * 0.5
    }
    return sorted[mid]
}


private func benchmarkDirectSelectionHarness<Model>(
    harness: inout DirectTokenSelectionGenerationHarness<Model>,
    promptTokens: [UInt16],
    maxNewTokens: Int,
    warmup: Int,
    iterations: Int
) throws -> GenerationBenchmarkSample
where Model: DirectTokenSelectingLanguageModel & GenerationPerformanceTrackable, Model: ~Copyable {
    var tokenLatencies: [Double] = []
    var throughput: [Double] = []
    var trunkLatencies: [Double] = []
    var logitsLatencies: [Double] = []
    tokenLatencies.reserveCapacity(iterations)
    throughput.reserveCapacity(iterations)
    trunkLatencies.reserveCapacity(iterations)
    logitsLatencies.reserveCapacity(iterations)

    let compileTimeMs = harness.model.performanceSnapshot.compileTimeMs

    for iter in 0..<(warmup + iterations) {
        let trace = try harness.generate(promptTokens: promptTokens, maxNewTokens: maxNewTokens)
        if iter >= warmup {
            let snapshot = harness.model.performanceSnapshot
            tokenLatencies.append(trace.totalLatencyMs / Double(maxNewTokens))
            throughput.append(trace.tokensPerSecond)
            trunkLatencies.append(snapshot.trunkLatencyMs / Double(maxNewTokens))
            logitsLatencies.append(snapshot.logitsLatencyMs / Double(maxNewTokens))
        }
    }

    return GenerationBenchmarkSample(
        medianTokenMs: median(tokenLatencies),
        medianTokensPerSecond: median(throughput),
        compileTimeMs: compileTimeMs,
        medianTrunkMsPerToken: median(trunkLatencies),
        medianLogitsMsPerToken: median(logitsLatencies)
    )
}

private func benchmarkExactTwoTokenHarness<Model>(
    harness: inout ExactTwoTokenGenerationHarness<Model>,
    promptTokens: [UInt16],
    maxNewTokens: Int,
    warmup: Int,
    iterations: Int
) throws -> ExactTwoTokenBenchmarkSample
where Model: ExactTwoTokenGeneratingLanguageModel & GenerationPerformanceTrackable, Model: ~Copyable {
    var tokenLatencies: [Double] = []
    var throughput: [Double] = []
    var committedExactTokensPerPass: [Double] = []
    var acceptedFutureTokensPerPass: [Double] = []
    var proposerMsPerPass: [Double] = []
    var verifierTrunkMsPerPass: [Double] = []
    var verifierLogitsMsPerPass: [Double] = []
    var stateAdvanceMsPerPass: [Double] = []

    tokenLatencies.reserveCapacity(iterations)
    throughput.reserveCapacity(iterations)
    committedExactTokensPerPass.reserveCapacity(iterations)
    acceptedFutureTokensPerPass.reserveCapacity(iterations)
    proposerMsPerPass.reserveCapacity(iterations)
    verifierTrunkMsPerPass.reserveCapacity(iterations)
    verifierLogitsMsPerPass.reserveCapacity(iterations)
    stateAdvanceMsPerPass.reserveCapacity(iterations)

    let compileTimeMs = harness.model.performanceSnapshot.compileTimeMs

    for iter in 0..<(warmup + iterations) {
        let trace = try harness.generate(promptTokens: promptTokens, maxNewTokens: maxNewTokens)
        if iter >= warmup {
            tokenLatencies.append(trace.totalLatencyMs / Double(maxNewTokens))
            throughput.append(trace.effectiveTokensPerSecond)
            committedExactTokensPerPass.append(trace.committedExactTokensPerPass)
            acceptedFutureTokensPerPass.append(trace.acceptedFutureTokensPerPass)
            proposerMsPerPass.append(trace.proposerLatencyMsPerPass)
            verifierTrunkMsPerPass.append(trace.verifierTrunkLatencyMsPerPass)
            verifierLogitsMsPerPass.append(trace.verifierLogitsLatencyMsPerPass)
            stateAdvanceMsPerPass.append(trace.stateAdvanceLatencyMsPerPass)
        }
    }

    return ExactTwoTokenBenchmarkSample(
        medianTokenMs: median(tokenLatencies),
        medianTokensPerSecond: median(throughput),
        compileTimeMs: compileTimeMs,
        medianCommittedExactTokensPerPass: median(committedExactTokensPerPass),
        medianAcceptedFutureTokensPerPass: median(acceptedFutureTokensPerPass),
        medianProposerMsPerPass: median(proposerMsPerPass),
        medianVerifierTrunkMsPerPass: median(verifierTrunkMsPerPass),
        medianVerifierLogitsMsPerPass: median(verifierLogitsMsPerPass),
        medianStateAdvanceMsPerPass: median(stateAdvanceMsPerPass)
    )
}

private func measureRecurrentControlCompileInitOnly(options: Options) throws -> CompileInitBenchmarkSample {
    let plan = try options.validatedProbeConfiguration()
    let weights = try loadRecurrentGenerationWeights(input: plan.input, layerCount: options.layerCount)
    let start = mach_absolute_time()
    let model = try ANERecurrentGenerationModel(
        weights: weights,
        layerCount: options.layerCount,
        maxSequenceTokens: options.maxSequenceTokens,
        outputHeadBackend: options.outputHeadBackend,
        trunkBackend: options.controlBackend,
        trunkLaneSpatial: options.trunkLaneSpatial,
        outputHeadLaneSpatial: options.outputHeadLaneSpatial
    )
    let wallInitMs = machMilliseconds(mach_absolute_time() - start)
    return CompileInitBenchmarkSample(
        wallInitMs: wallInitMs,
        reportedCompileTimeMs: model.performanceSnapshot.compileTimeMs
    )
}

private func measureTwoStepCompileInitOnly(options: Options) throws -> CompileInitBenchmarkSample {
    let plan = try options.validatedProbeConfiguration()
    let weights = try loadRecurrentGenerationWeights(input: plan.input, layerCount: options.layerCount)
    let start = mach_absolute_time()
    let model: ANEExactTwoTokenBranchStatePromotionModel
    if let futureSidecarPath = options.futureSidecarPath {
        let futureSidecar = try TwoStepStudentCheckpoint.load(path: futureSidecarPath)
        model = try ANEExactTwoTokenBranchStatePromotionModel(
            weights: weights,
            futureSidecar: futureSidecar,
            layerCount: options.layerCount,
            maxSequenceTokens: options.maxSequenceTokens,
            outputHeadBackend: options.outputHeadBackend,
            trunkBackend: options.twoStepBackend,
            trunkLaneSpatial: options.trunkLaneSpatial,
            outputHeadLaneSpatial: options.outputHeadLaneSpatial
        )
    } else {
        model = try ANEExactTwoTokenBranchStatePromotionModel(
            weights: weights,
            layerCount: options.layerCount,
            maxSequenceTokens: options.maxSequenceTokens,
            outputHeadBackend: options.outputHeadBackend,
            trunkBackend: options.twoStepBackend,
            trunkLaneSpatial: options.trunkLaneSpatial,
            outputHeadLaneSpatial: options.outputHeadLaneSpatial
        )
    }
    let wallInitMs = machMilliseconds(mach_absolute_time() - start)
    return CompileInitBenchmarkSample(
        wallInitMs: wallInitMs,
        reportedCompileTimeMs: model.performanceSnapshot.compileTimeMs
    )
}

private func compileOnlyPayload(options: Options) throws -> [String: Any] {
    let plan = try options.validatedProbeConfiguration()
    printStderr("Resetting compile budget")
    try? CompileBudget.setCount(0)

    printStderr("Starting control compile/init")
    let control = try measureRecurrentControlCompileInitOnly(options: options)
    printStderr(String(format: "Control compile/init done in %.3f ms", control.wallInitMs))

    printStderr("Starting two-step compile/init")
    let twoStep = try measureTwoStepCompileInitOnly(options: options)
    printStderr(String(format: "Two-step compile/init done in %.3f ms", twoStep.wallInitMs))

    return [
        "mode": options.mode.rawValue,
        "control_backend": describe(options.controlBackend),
        "two_step_backend": describe(options.twoStepBackend),
        "input_mode": describe(plan.input),
        "layer_count": options.layerCount,
        "output_head_backend": describe(options.outputHeadBackend),
        "max_sequence_tokens": options.maxSequenceTokens,
        "control": [
            "init_wall_ms": control.wallInitMs,
            "reported_compile_ms": control.reportedCompileTimeMs,
        ],
        "two_step": [
            "init_wall_ms": twoStep.wallInitMs,
            "reported_compile_ms": twoStep.reportedCompileTimeMs,
        ],
    ]
}

private func comparePayload(options: Options) throws -> [String: Any] {
    let plan = try options.validatedProbeConfiguration()
    printStderr("Resetting compile budget")
    try? CompileBudget.setCount(0)

    let prompt: [UInt16] = [options.promptToken]
    let weights = try loadRecurrentGenerationWeights(input: plan.input, layerCount: options.layerCount)

    printStderr("Starting control model init")
    let controlInitStart = mach_absolute_time()
    let controlModel = try ANERecurrentGenerationModel(
        weights: weights,
        layerCount: options.layerCount,
        maxSequenceTokens: options.maxSequenceTokens,
        outputHeadBackend: options.outputHeadBackend,
        trunkBackend: options.controlBackend,
        trunkLaneSpatial: options.trunkLaneSpatial,
        outputHeadLaneSpatial: options.outputHeadLaneSpatial
    )
    let controlInitMs = machMilliseconds(mach_absolute_time() - controlInitStart)
    printStderr(String(format: "Control model init done in %.3f ms", controlInitMs))
    var controlHarness = DirectTokenSelectionGenerationHarness(model: controlModel, strategy: .argmax)

    printStderr("Starting two-step model init")
    let twoStepInitStart = mach_absolute_time()
    let twoStepModel: ANEExactTwoTokenBranchStatePromotionModel
    if let futureSidecarPath = options.futureSidecarPath {
        let futureSidecar = try TwoStepStudentCheckpoint.load(path: futureSidecarPath)
        twoStepModel = try ANEExactTwoTokenBranchStatePromotionModel(
            weights: weights,
            futureSidecar: futureSidecar,
            layerCount: options.layerCount,
            maxSequenceTokens: options.maxSequenceTokens,
            outputHeadBackend: options.outputHeadBackend,
            trunkBackend: options.twoStepBackend,
            trunkLaneSpatial: options.trunkLaneSpatial,
            outputHeadLaneSpatial: options.outputHeadLaneSpatial
        )
    } else {
        twoStepModel = try ANEExactTwoTokenBranchStatePromotionModel(
            weights: weights,
            layerCount: options.layerCount,
            maxSequenceTokens: options.maxSequenceTokens,
            outputHeadBackend: options.outputHeadBackend,
            trunkBackend: options.twoStepBackend,
            trunkLaneSpatial: options.trunkLaneSpatial,
            outputHeadLaneSpatial: options.outputHeadLaneSpatial
        )
    }
    let twoStepInitMs = machMilliseconds(mach_absolute_time() - twoStepInitStart)
    printStderr(String(format: "Two-step model init done in %.3f ms", twoStepInitMs))
    var twoStepHarness = ExactTwoTokenGenerationHarness(model: twoStepModel, strategy: .argmax)

    printStderr("Running parity trace")
    let controlParityTrace = try controlHarness.generate(promptTokens: prompt, maxNewTokens: options.maxNewTokens)
    let twoStepParityTrace = try twoStepHarness.generate(promptTokens: prompt, maxNewTokens: options.maxNewTokens)
    let exactParity = controlParityTrace.generatedTokens == twoStepParityTrace.generatedTokens
    printStderr("Parity status: \(exactParity ? "match" : "mismatch")")

    printStderr("Benchmarking control")
    let control = try benchmarkDirectSelectionHarness(
        harness: &controlHarness,
        promptTokens: prompt,
        maxNewTokens: options.maxNewTokens,
        warmup: options.warmup,
        iterations: options.iterations
    )
    printStderr(String(format: "Control median %.6f ms/token", control.medianTokenMs))

    printStderr("Benchmarking two-step")
    let twoStep = try benchmarkExactTwoTokenHarness(
        harness: &twoStepHarness,
        promptTokens: prompt,
        maxNewTokens: options.maxNewTokens,
        warmup: options.warmup,
        iterations: options.iterations
    )
    printStderr(String(format: "Two-step median %.6f ms/token", twoStep.medianTokenMs))

    let coreML: GenerationBenchmarkSample?
    if let request = plan.coreMLRequest {
        printStderr("Benchmarking CoreML")
        coreML = try benchmarkCoreMLGeneration(
            request: request,
            promptTokens: prompt,
            maxNewTokens: options.maxNewTokens,
            warmup: options.warmup,
            iterations: options.iterations,
            maxSequenceTokens: options.maxSequenceTokens
        )
        printStderr(String(format: "CoreML median %.6f ms/token", coreML?.medianTokenMs ?? 0))
    } else {
        coreML = nil
    }

    var payload: [String: Any] = [
        "mode": options.mode.rawValue,
        "control_backend": describe(options.controlBackend),
        "two_step_backend": describe(options.twoStepBackend),
        "input_mode": describe(plan.input),
        "layer_count": options.layerCount,
        "output_head_backend": describe(options.outputHeadBackend),
        "warmup": options.warmup,
        "iterations": options.iterations,
        "max_new_tokens": options.maxNewTokens,
        "max_sequence_tokens": options.maxSequenceTokens,
        "prompt_tokens": prompt.map(Int.init),
        "parity_status": exactParity ? "match" : "mismatch",
        "control": [
            "init_wall_ms": controlInitMs,
            "reported_compile_ms": control.compileTimeMs,
            "median_ms_per_token": control.medianTokenMs,
            "median_tokens_per_second": control.medianTokensPerSecond,
            "median_trunk_ms_per_token": control.medianTrunkMsPerToken,
            "median_logits_ms_per_token": control.medianLogitsMsPerToken,
            "generated_tokens": controlParityTrace.generatedTokens.map(Int.init),
        ],
        "two_step": [
            "init_wall_ms": twoStepInitMs,
            "reported_compile_ms": twoStep.compileTimeMs,
            "median_ms_per_token": twoStep.medianTokenMs,
            "median_tokens_per_second": twoStep.medianTokensPerSecond,
            "median_committed_exact_tokens_per_pass": twoStep.medianCommittedExactTokensPerPass,
            "median_accepted_future_tokens_per_pass": twoStep.medianAcceptedFutureTokensPerPass,
            "median_proposer_ms_per_pass": twoStep.medianProposerMsPerPass,
            "median_verifier_trunk_ms_per_pass": twoStep.medianVerifierTrunkMsPerPass,
            "median_verifier_logits_ms_per_pass": twoStep.medianVerifierLogitsMsPerPass,
            "median_state_advance_ms_per_pass": twoStep.medianStateAdvanceMsPerPass,
            "generated_tokens": twoStepParityTrace.generatedTokens.map(Int.init),
        ],
    ]
    if let request = plan.coreMLRequest, let coreML {
        payload["coreml"] = [
            "model_path": request.modelPath,
            "compute_units": describe(request.computeUnits),
            "head_weights_source": describe(request.headWeightsSource),
            "median_ms_per_token": coreML.medianTokenMs,
            "median_tokens_per_second": coreML.medianTokensPerSecond,
            "reported_compile_ms": coreML.compileTimeMs,
            "median_trunk_ms_per_token": coreML.medianTrunkMsPerToken,
            "median_logits_ms_per_token": coreML.medianLogitsMsPerToken,
        ]
        payload["two_step_speedup_vs_coreml"] = coreML.medianTokenMs / twoStep.medianTokenMs
        payload["control_speedup_vs_coreml"] = coreML.medianTokenMs / control.medianTokenMs
    }
    return payload
}

private func describe(_ backend: RecurrentGenerationTrunkBackend) -> String {
    switch backend {
    case .singleLayer: return "single"
    case .fusedTwoLayerPairs: return "fused-pair"
    case .fusedThreeLayerTriplets: return "fused-triplet"
    case .identityZeroTrunk: return "identity-zero-trunk"
    case .identityZeroTrunkLookup: return "identity-zero-trunk-lookup"
    }
}

private func describe(_ backend: GenerationOutputHeadBackend) -> String {
    switch backend {
    case .cpu: return "cpu"
    case .cpuExactStaged: return "cpu-exact-staged"
    case .cpuExactClustered: return "cpu-exact-clustered"
    case .aneClassifier: return "ane-classifier"
    case .aneRMSNormClassifier: return "ane-rmsnorm-classifier"
    }
}

private func describe(_ input: MultitokenProbeInput) -> String {
    switch input {
    case .echo:
        return "echo"
    case .recurrentCheckpoint:
        return "recurrent-checkpoint"
    }
}

private func describe(_ computeUnits: MLComputeUnits) -> String {
    switch computeUnits {
    case .all:
        return "all"
    case .cpuAndGPU:
        return "cpu-and-gpu"
    case .cpuAndNeuralEngine:
        return "cpu-and-neural-engine"
    case .cpuOnly:
        return "cpu-only"
    @unknown default:
        return "unknown"
    }
}

private func describe(_ source: CoreMLHeadWeightsSource) -> String {
    switch source {
    case .echo:
        return "echo"
    case .generationModel:
        return "generation-model"
    }
}

private func writeJSON(_ payload: [String: Any]) throws {
    let data = try JSONSerialization.data(withJSONObject: payload, options: [.prettyPrinted, .sortedKeys])
    guard let json = String(data: data, encoding: .utf8) else {
        throw ProbeError.invariantViolation("Failed to encode JSON output")
    }
    print(json)
}

private let options = Options.parse(CommandLine.arguments)

do {
    let payload: [String: Any]
    switch options.mode {
    case .compileInitOnly:
        payload = try compileOnlyPayload(options: options)
    case .compare:
        payload = try comparePayload(options: options)
    }
    try writeJSON(payload)
} catch {
    fatal("\(error)")
}
