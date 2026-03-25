import Accelerate
import ANETypes
@preconcurrency import CoreML
import Foundation
import ModelSupport

struct DemoDefaults {
    let repoRoot: URL?
    let workingDirectory: URL
    let stateRoot: URL
    let cacheRoot: URL
    let reportsRoot: URL
    let hfCacheRoot: URL
    let weightsDir: URL
    let tokenizerDir: URL
    let coreMLDir: URL
    let toolsVenvDir: URL
    let scriptsDir: URL?
    let legacyArtifactsRoot: URL?

    private func helperScriptURL(named name: String) -> URL? {
        guard let scriptsDir else { return nil }
        let scriptURL = scriptsDir.appendingPathComponent(name)
        return FileManager().fileExists(atPath: scriptURL.path) ? scriptURL : nil
    }

    var bootstrapScriptURL: URL? {
        helperScriptURL(named: "bootstrap_gpt2_demo.py")
    }

    var exportScriptURL: URL? {
        helperScriptURL(named: "export_gpt2_coreml.py")
    }

    var coreMLReferenceScriptURL: URL? {
        helperScriptURL(named: "run_gpt2_coreml_reference.py")
    }

    var bootstrapScriptAvailable: Bool {
        bootstrapScriptURL != nil
    }

    var exportScriptAvailable: Bool {
        exportScriptURL != nil
    }

    var referenceScriptAvailable: Bool {
        coreMLReferenceScriptURL != nil
    }

    var scriptsAvailable: Bool {
        bootstrapScriptAvailable && exportScriptAvailable
    }

    func coreMLModelURL(sequenceLength: Int, weightsDirURL: URL? = nil) -> URL {
        if let weightsDirURL {
            let parent = weightsDirURL.deletingLastPathComponent()
            let basename = weightsDirURL.lastPathComponent
            let directoryName = basename == "gpt2_124m" ? "gpt2_coreml" : "\(basename)_coreml"
            return parent
                .appendingPathComponent(directoryName, isDirectory: true)
                .appendingPathComponent("gpt2_seq\(sequenceLength).mlpackage")
        }
        return coreMLDir.appendingPathComponent("gpt2_seq\(sequenceLength).mlpackage")
    }
}

struct CoreMLComparisonResult: Decodable, Sendable {
    let generatedTokens: [Int]
    let compileTimeMs: Double
    let firstTokenLatencyMs: Double
    let tokensPerSecond: Double
    let medianTokenMs: Double
    let p95TokenMs: Double
    let tokenLatenciesMs: [Double]
    let totalTimeMs: Double
    let computeUnits: String
    let seqLen: Int
    let isStateful: Bool?
    let loadSucceeded: Bool?
    let computePlanSucceeded: Bool?
    let errorDomain: String?
    let errorCode: Int?
    let errorMessage: String?

    private enum CodingKeys: String, CodingKey {
        case generatedTokens = "generated_tokens"
        case compileTimeMs = "compile_time_ms"
        case firstTokenLatencyMs = "first_token_latency_ms"
        case tokensPerSecond = "tokens_per_second"
        case medianTokenMs = "median_token_ms"
        case p95TokenMs = "p95_token_ms"
        case tokenLatenciesMs = "token_latencies_ms"
        case totalTimeMs = "total_time_ms"
        case computeUnits = "compute_units"
        case seqLen = "seq_len"
        case isStateful = "is_stateful"
        case loadSucceeded = "load_succeeded"
        case computePlanSucceeded = "compute_plan_succeeded"
        case errorDomain = "error_domain"
        case errorCode = "error_code"
        case errorMessage = "error_message"
    }
}

enum CoreMLStreamEvent: Sendable {
    case compile(compileTimeMs: Double, computeUnits: String, seqLen: Int)
    case token(token: TokenID, tokenIndex: Int, elapsedMs: Double, tokenLatencyMs: Double, tokensPerSecond: Double)
    case completed(CoreMLComparisonResult)
}

private struct ProcessOutput {
    let status: Int32
    let stdout: String
    let stderr: String
}

private struct RawCoreMLStreamEvent: Decodable {
    let type: String
    let token: Int?
    let tokenIndex: Int?
    let elapsedMs: Double?
    let tokenLatencyMs: Double?
    let tokensPerSecond: Double?
    let compileTimeMs: Double?
    let computeUnits: String?
    let seqLen: Int?
    let generatedTokens: [Int]?
    let firstTokenLatencyMs: Double?
    let medianTokenMs: Double?
    let p95TokenMs: Double?
    let tokenLatenciesMs: [Double]?
    let totalTimeMs: Double?

    private enum CodingKeys: String, CodingKey {
        case type
        case token
        case tokenIndex = "token_index"
        case elapsedMs = "elapsed_ms"
        case tokenLatencyMs = "token_latency_ms"
        case tokensPerSecond = "tokens_per_second"
        case compileTimeMs = "compile_time_ms"
        case computeUnits = "compute_units"
        case seqLen = "seq_len"
        case generatedTokens = "generated_tokens"
        case firstTokenLatencyMs = "first_token_latency_ms"
        case medianTokenMs = "median_token_ms"
        case p95TokenMs = "p95_token_ms"
        case tokenLatenciesMs = "token_latencies_ms"
        case totalTimeMs = "total_time_ms"
    }
}

private struct NativeCoreMLTopLevelWeights {
    let hiddenSize: Int
    let vocabSize: Int
    let finalNormGamma: [Float]
    let finalNormBeta: [Float]?
    let lmHead: [Float]
}

func normalizeHiddenForCoreMLReference(
    architecture: MultiModelConfig.Architecture,
    hidden: [Float],
    epsilon: Float,
    gamma: [Float],
    beta: [Float]?
) -> [Float] {
    precondition(hidden.count == gamma.count)
    var output = [Float](repeating: 0, count: hidden.count)

    switch architecture {
    case .gpt2:
        let mean = hidden.reduce(0, +) / Float(hidden.count)
        var variance: Float = 0
        for value in hidden {
            let delta = value - mean
            variance += delta * delta
        }
        variance /= Float(hidden.count)
        let inverseStd = 1.0 / sqrtf(variance + epsilon)

        for index in hidden.indices {
            let normalized = (hidden[index] - mean) * inverseStd * gamma[index]
            output[index] = normalized + (beta?[index] ?? 0)
        }
    case .llama:
        var sumSquares: Float = 0
        for value in hidden {
            sumSquares += value * value
        }
        let invRms = 1.0 / sqrtf((sumSquares / Float(hidden.count)) + epsilon)
        for index in hidden.indices {
            output[index] = hidden[index] * invRms * gamma[index]
        }
    }

    return output
}

private struct NativeCoreMLRunResult {
    let generatedTokens: [TokenID]
    let firstTokenLatencyMs: Double
    let tokensPerSecond: Double
    let tokenLatenciesMs: [Double]
    let totalTimeMs: Double
}

private struct NativeSplitMix64: RandomNumberGenerator {
    private var state: UInt64

    init(seed: UInt64) {
        self.state = seed
    }

    mutating func next() -> UInt64 {
        state &+= 0x9E3779B97F4A7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
        return z ^ (z >> 31)
    }
}

struct CoreMLReferenceArraySpec: Equatable, Sendable {
    let name: String
    let shape: [Int]
    let dataType: MLMultiArrayDataType
}

struct StatelessCoreMLReferenceContract: Equatable, Sendable {
    let tokenInput: CoreMLReferenceArraySpec
    let outputFeatureName: String
    let maxSequenceTokens: Int
}

struct StatefulCoreMLReferenceContract: Equatable, Sendable {
    let tokenInput: CoreMLReferenceArraySpec
    let cachePositionInput: CoreMLReferenceArraySpec
    let outputFeatureName: String
    let maxSequenceTokens: Int
}

enum CoreMLReferenceModelContract: Equatable, Sendable {
    case stateless(StatelessCoreMLReferenceContract)
    case stateful(StatefulCoreMLReferenceContract)

    var outputFeatureName: String {
        switch self {
        case .stateless(let contract):
            contract.outputFeatureName
        case .stateful(let contract):
            contract.outputFeatureName
        }
    }

    var maxSequenceTokens: Int {
        switch self {
        case .stateless(let contract):
            contract.maxSequenceTokens
        case .stateful(let contract):
            contract.maxSequenceTokens
        }
    }

    var tokenInput: CoreMLReferenceArraySpec {
        switch self {
        case .stateless(let contract):
            contract.tokenInput
        case .stateful(let contract):
            contract.tokenInput
        }
    }

    var cachePositionInput: CoreMLReferenceArraySpec? {
        switch self {
        case .stateless:
            nil
        case .stateful(let contract):
            contract.cachePositionInput
        }
    }

    var isStateful: Bool {
        if case .stateful = self {
            return true
        }
        return false
    }
}

struct CoreMLReferenceDiagnostics: Sendable, Equatable {
    let isStateful: Bool
    let loadSucceeded: Bool
    let computePlanSucceeded: Bool?
    let errorDomain: String?
    let errorCode: Int?
    let errorMessage: String?

    init(
        isStateful: Bool = false,
        loadSucceeded: Bool = false,
        computePlanSucceeded: Bool? = nil,
        errorDomain: String? = nil,
        errorCode: Int? = nil,
        errorMessage: String? = nil
    ) {
        self.isStateful = isStateful
        self.loadSucceeded = loadSucceeded
        self.computePlanSucceeded = computePlanSucceeded
        self.errorDomain = errorDomain
        self.errorCode = errorCode
        self.errorMessage = errorMessage
    }
}

private struct CoreMLReferenceFailure: Error {
    let message: String
    let diagnostics: CoreMLReferenceDiagnostics
}

func resolveCoreMLReferenceModelContract(
    inputSpecs: [CoreMLReferenceArraySpec],
    outputFeatureNames: [String],
    stateSpecs: [CoreMLReferenceArraySpec],
    requestedSequenceLength: Int
) throws -> CoreMLReferenceModelContract {
    guard requestedSequenceLength > 0 else {
        throw CLIError.runtime("Core ML sequence length must be > 0")
    }
    guard outputFeatureNames.count == 1, let outputFeatureName = outputFeatureNames.first else {
        throw CLIError.runtime("expected exactly one Core ML output, found \(outputFeatureNames.count)")
    }

    if stateSpecs.isEmpty {
        guard inputSpecs.count == 1, let tokenInput = inputSpecs.first else {
            throw CLIError.runtime("expected exactly one Core ML input, found \(inputSpecs.count)")
        }
        guard tokenInput.shape.count == 2 else {
            throw CLIError.runtime("Expected a rank-2 Core ML input, found rank \(tokenInput.shape.count)")
        }
        guard tokenInput.shape[0] == 1 else {
            throw CLIError.runtime("Expected Core ML input shape [1, seq], found \(tokenInput.shape)")
        }
        guard requestedSequenceLength <= tokenInput.shape[1] else {
            throw CLIError.runtime(
                "Requested Core ML sequence length \(requestedSequenceLength) exceeds model capacity \(tokenInput.shape[1])"
            )
        }
        return .stateless(
            StatelessCoreMLReferenceContract(
                tokenInput: tokenInput,
                outputFeatureName: outputFeatureName,
                maxSequenceTokens: tokenInput.shape[1]
            )
        )
    }

    guard inputSpecs.count == 2 else {
        throw CLIError.runtime("expected exactly two Core ML inputs for a stateful model, found \(inputSpecs.count)")
    }

    let tokenCandidates = inputSpecs.filter { $0.dataType == .int32 && $0.shape == [1, 1] }
    let cachePositionCandidates = inputSpecs.filter { $0.dataType == .int32 && $0.shape == [1] }
    guard tokenCandidates.count == 1, let tokenInput = tokenCandidates.first else {
        throw CLIError.runtime("Could not identify stateful token input; expected one int32 [1, 1] input.")
    }
    guard cachePositionCandidates.count == 1, let cachePositionInput = cachePositionCandidates.first else {
        throw CLIError.runtime("Could not identify stateful cache-position input; expected one int32 [1] input.")
    }
    guard tokenInput.name != cachePositionInput.name else {
        throw CLIError.runtime("Stateful Core ML model reuses the same feature for token input and cache position.")
    }

    let stateCapacities = try Set(stateSpecs.map { spec -> Int in
        guard spec.shape.count == 4 else {
            throw CLIError.runtime("State buffer \(spec.name) must be rank-4, found shape \(spec.shape)")
        }
        guard spec.dataType == .float16 else {
            throw CLIError.runtime("State buffer \(spec.name) must use float16, found dtype \(spec.dataType.rawValue)")
        }
        return spec.shape[2]
    })
    guard stateCapacities.count == 1, let maxSequenceTokens = stateCapacities.first else {
        throw CLIError.runtime("Stateful Core ML states disagree on KV cache capacity.")
    }
    guard requestedSequenceLength <= maxSequenceTokens else {
        throw CLIError.runtime(
            "Requested Core ML sequence length \(requestedSequenceLength) exceeds stateful KV capacity \(maxSequenceTokens)"
        )
    }

    return .stateful(
        StatefulCoreMLReferenceContract(
            tokenInput: tokenInput,
            cachePositionInput: cachePositionInput,
            outputFeatureName: outputFeatureName,
            maxSequenceTokens: maxSequenceTokens
        )
    )
}

private final class AsyncResultBox<T>: @unchecked Sendable {
    private let lock = NSLock()
    private var result: Result<T, Error>?

    func store(_ value: Result<T, Error>) {
        lock.lock()
        result = value
        lock.unlock()
    }

    func take() -> Result<T, Error>? {
        lock.lock()
        let snapshot = result
        lock.unlock()
        return snapshot
    }
}

private final class UncheckedSendableBox<T>: @unchecked Sendable {
    let value: T

    init(_ value: T) {
        self.value = value
    }
}

private func runAsyncCoreMLOperation<T>(
    _ operation: @escaping () async throws -> T
) throws -> T {
    let semaphore = DispatchSemaphore(value: 0)
    let box = AsyncResultBox<T>()
    let operationBox = UncheckedSendableBox(operation)
    Task {
        do {
            box.store(.success(try await operationBox.value()))
        } catch {
            box.store(.failure(error))
        }
        semaphore.signal()
    }
    semaphore.wait()
    guard let result = box.take() else {
        throw CLIError.runtime("Async Core ML operation produced no result.")
    }
    return try result.get()
}

private struct NativeCoreMLReferenceRunner {
    let vocabSize: Int
    let maxSequenceTokens: Int
    let compileTimeMs: Double
    let computeUnits: String
    let diagnostics: CoreMLReferenceDiagnostics

    private let hiddenSize: Int
    private let architecture: MultiModelConfig.Architecture
    private let model: MLModel
    private let contract: CoreMLReferenceModelContract
    private let tokenInputArray: MLMultiArray
    private let tokenInputDataType: MLMultiArrayDataType
    private let cachePositionArray: MLMultiArray?
    private let cachePositionDataType: MLMultiArrayDataType?
    private let epsilon: Float
    private let finalNormGamma: [Float]
    private let finalNormBeta: [Float]?
    private let lmHead: [Float]

    private var currentTokens: [TokenID]
    private var modelState: MLState?
    private var stepHidden: [Float]
    private var stepNorm: [Float]
    private var stepLogits: [Float]

    init(
        modelPath: String,
        weightsDir: String,
        architecture: MultiModelConfig.Architecture,
        sequenceLength: Int,
        computeUnits: String,
        epsilon: Float = 1e-5
    ) throws {
        let expandedModelPath = NSString(string: modelPath).expandingTildeInPath
        guard FileManager.default.fileExists(atPath: expandedModelPath) else {
            throw CLIError.usage("Core ML model path does not exist: \(modelPath)")
        }

        let weights = try loadNativeCoreMLTopLevelWeights(
            weightsDir: weightsDir,
            architecture: architecture
        )
        let compileStart = monotonicNow()
        let compiledURL: URL
        do {
            compiledURL = try MLModel.compileModel(at: URL(fileURLWithPath: expandedModelPath))
        } catch {
            throw CLIError.runtime("Core ML compile failed: \(error)")
        }

        let configuration = MLModelConfiguration()
        configuration.computeUnits = try parseCoreMLComputeUnits(computeUnits)
        let computePlanSucceeded = try Self.probeComputePlan(
            modelURL: compiledURL,
            configuration: configuration,
            computeUnits: computeUnits
        )

        let model: MLModel
        do {
            model = try MLModel(contentsOf: compiledURL, configuration: configuration)
        } catch {
            let diagnostics = Self.diagnostics(
                isStateful: false,
                loadSucceeded: false,
                computePlanSucceeded: computePlanSucceeded,
                error: error
            )
            throw CoreMLReferenceFailure(
                message: Self.failureMessage(prefix: "Core ML load failed", error: error),
                diagnostics: diagnostics
            )
        }

        let inputSpecs = try model.modelDescription.inputDescriptionsByName.map { name, description in
            guard let constraint = description.multiArrayConstraint else {
                throw CLIError.runtime("Core ML input '\(name)' is not an MLMultiArray")
            }
            return CoreMLReferenceArraySpec(
                name: name,
                shape: constraint.shape.map(\.intValue),
                dataType: constraint.dataType
            )
        }
        let stateSpecs: [CoreMLReferenceArraySpec]
        if #available(macOS 15.0, *) {
            stateSpecs = try model.modelDescription.stateDescriptionsByName.map { name, description in
                guard let constraint = description.stateConstraint else {
                    throw CLIError.runtime("Core ML state '\(name)' is not an MLState buffer")
                }
                return CoreMLReferenceArraySpec(
                    name: name,
                    shape: constraint.bufferShape,
                    dataType: constraint.dataType
                )
            }
        } else {
            stateSpecs = []
        }
        let contract = try resolveCoreMLReferenceModelContract(
            inputSpecs: inputSpecs.sorted { $0.name < $1.name },
            outputFeatureNames: Array(model.modelDescription.outputDescriptionsByName.keys).sorted(),
            stateSpecs: stateSpecs.sorted { $0.name < $1.name },
            requestedSequenceLength: sequenceLength
        )

        let tokenInputArray: MLMultiArray
        do {
            tokenInputArray = try MLMultiArray(
                shape: contract.tokenInput.shape.map(NSNumber.init(value:)),
                dataType: contract.tokenInput.dataType
            )
        } catch {
            throw CLIError.runtime("failed to allocate Core ML token input array: \(error)")
        }

        let cachePositionArray: MLMultiArray?
        let cachePositionDataType: MLMultiArrayDataType?
        if let cachePositionInput = contract.cachePositionInput {
            do {
                cachePositionArray = try MLMultiArray(
                    shape: cachePositionInput.shape.map(NSNumber.init(value:)),
                    dataType: cachePositionInput.dataType
                )
            } catch {
                throw CLIError.runtime("failed to allocate Core ML cache-position array: \(error)")
            }
            cachePositionDataType = cachePositionInput.dataType
        } else {
            cachePositionArray = nil
            cachePositionDataType = nil
        }

        self.vocabSize = weights.vocabSize
        self.maxSequenceTokens = contract.maxSequenceTokens
        self.compileTimeMs = monotonicMilliseconds(since: compileStart)
        self.computeUnits = computeUnits
        self.diagnostics = CoreMLReferenceDiagnostics(
            isStateful: contract.isStateful,
            loadSucceeded: true,
            computePlanSucceeded: computePlanSucceeded
        )
        self.hiddenSize = weights.hiddenSize
        self.architecture = architecture
        self.model = model
        self.contract = contract
        self.tokenInputArray = tokenInputArray
        self.tokenInputDataType = contract.tokenInput.dataType
        self.cachePositionArray = cachePositionArray
        self.cachePositionDataType = cachePositionDataType
        self.epsilon = epsilon
        self.finalNormGamma = weights.finalNormGamma
        self.finalNormBeta = weights.finalNormBeta
        self.lmHead = weights.lmHead
        self.currentTokens = []
        self.modelState = nil
        self.stepHidden = [Float](repeating: 0, count: weights.hiddenSize)
        self.stepNorm = [Float](repeating: 0, count: weights.hiddenSize)
        self.stepLogits = [Float](repeating: 0, count: weights.vocabSize)

        try zeroArray(tokenInputArray, dataType: tokenInputDataType, label: "token input")
        if let cachePositionArray, let cachePositionDataType {
            try zeroArray(cachePositionArray, dataType: cachePositionDataType, label: "cache position")
        }
    }

    mutating func benchmark(
        promptTokens: [TokenID],
        maxTokens: Int,
        temperature: Float,
        warmup: Int,
        iterations: Int,
        seed: Int,
        onEvent: ((CoreMLStreamEvent) -> Void)? = nil
    ) throws -> CoreMLComparisonResult {
        guard !promptTokens.isEmpty else {
            throw CLIError.runtime("Cannot compare Core ML without prompt tokens.")
        }
        guard maxTokens > 0 else {
            throw CLIError.usage("max-tokens must be > 0 for Core ML compare.")
        }
        guard promptTokens.count + maxTokens <= maxSequenceTokens else {
            throw CLIError.runtime(
                "Prompt length \(promptTokens.count) + max tokens \(maxTokens) exceeds Core ML sequence length \(maxSequenceTokens)"
            )
        }

        onEvent?(.compile(compileTimeMs: compileTimeMs, computeUnits: computeUnits, seqLen: maxSequenceTokens))

        var aggregatedLatencySamples: [Double] = []
        var lastMeasured: NativeCoreMLRunResult?

        for iteration in 0..<(warmup + iterations) {
            let emitEvents = iteration == (warmup + iterations - 1) ? onEvent : nil
            let measured = try runOnce(
                promptTokens: promptTokens,
                maxTokens: maxTokens,
                temperature: temperature,
                seed: seed,
                onEvent: emitEvents
            )
            if iteration >= warmup {
                lastMeasured = measured
                aggregatedLatencySamples.append(contentsOf: measured.tokenLatenciesMs)
            }
        }

        guard let lastMeasured else {
            throw CLIError.runtime("Core ML benchmark did not produce a measured run.")
        }

        let result = CoreMLComparisonResult(
            generatedTokens: lastMeasured.generatedTokens.map(Int.init),
            compileTimeMs: compileTimeMs,
            firstTokenLatencyMs: lastMeasured.firstTokenLatencyMs,
            tokensPerSecond: lastMeasured.tokensPerSecond,
            medianTokenMs: nativePercentile(aggregatedLatencySamples, percentile: 0.5),
            p95TokenMs: nativePercentile(aggregatedLatencySamples, percentile: 0.95),
            tokenLatenciesMs: lastMeasured.tokenLatenciesMs,
            totalTimeMs: lastMeasured.totalTimeMs,
            computeUnits: computeUnits,
            seqLen: maxSequenceTokens,
            isStateful: diagnostics.isStateful,
            loadSucceeded: diagnostics.loadSucceeded,
            computePlanSucceeded: diagnostics.computePlanSucceeded,
            errorDomain: diagnostics.errorDomain,
            errorCode: diagnostics.errorCode,
            errorMessage: diagnostics.errorMessage
        )
        onEvent?(.completed(result))
        return result
    }

    private mutating func runOnce(
        promptTokens: [TokenID],
        maxTokens: Int,
        temperature: Float,
        seed: Int,
        onEvent: ((CoreMLStreamEvent) -> Void)?
    ) throws -> NativeCoreMLRunResult {
        try reset()

        var generatedTokens: [TokenID] = []
        generatedTokens.reserveCapacity(maxTokens)
        var tokenLatenciesMs: [Double] = []
        tokenLatenciesMs.reserveCapacity(maxTokens)
        var rng = NativeSplitMix64(seed: UInt64(bitPattern: Int64(seed)))

        let generationStart = monotonicNow()
        let prefillStart = monotonicNow()
        let firstLogits = try prefill(promptTokens: promptTokens)
        let firstTokenLatencyMs = monotonicMilliseconds(since: prefillStart)
        let firstToken = sampleNativeToken(from: firstLogits, temperature: temperature, using: &rng)
        generatedTokens.append(firstToken)
        tokenLatenciesMs.append(firstTokenLatencyMs)
        onEvent?(
            .token(
                token: firstToken,
                tokenIndex: 1,
                elapsedMs: monotonicMilliseconds(since: generationStart),
                tokenLatencyMs: firstTokenLatencyMs,
                tokensPerSecond: throughput(tokensGenerated: 1, totalTimeMs: monotonicMilliseconds(since: generationStart))
            )
        )

        var currentToken = firstToken
        if maxTokens > 1 {
            for tokenIndex in 2...maxTokens {
                let tokenStart = monotonicNow()
                let logits = try decode(nextToken: currentToken)
                let tokenLatencyMs = monotonicMilliseconds(since: tokenStart)
                currentToken = sampleNativeToken(from: logits, temperature: temperature, using: &rng)
                generatedTokens.append(currentToken)
                tokenLatenciesMs.append(tokenLatencyMs)
                let elapsedMs = monotonicMilliseconds(since: generationStart)
                onEvent?(
                    .token(
                        token: currentToken,
                        tokenIndex: tokenIndex,
                        elapsedMs: elapsedMs,
                        tokenLatencyMs: tokenLatencyMs,
                        tokensPerSecond: throughput(tokensGenerated: tokenIndex, totalTimeMs: elapsedMs)
                    )
                )
            }
        }

        let totalTimeMs = monotonicMilliseconds(since: generationStart)
        return NativeCoreMLRunResult(
            generatedTokens: generatedTokens,
            firstTokenLatencyMs: firstTokenLatencyMs,
            tokensPerSecond: throughput(tokensGenerated: generatedTokens.count, totalTimeMs: totalTimeMs),
            tokenLatenciesMs: tokenLatenciesMs,
            totalTimeMs: totalTimeMs
        )
    }

    private mutating func reset() throws {
        currentTokens.removeAll(keepingCapacity: true)
        modelState = try makeInitialStateIfNeeded()
        stepHidden.withUnsafeMutableBufferPointer { buffer in
            for index in buffer.indices {
                buffer[index] = 0
            }
        }
        stepNorm.withUnsafeMutableBufferPointer { buffer in
            for index in buffer.indices {
                buffer[index] = 0
            }
        }
        stepLogits.withUnsafeMutableBufferPointer { buffer in
            for index in buffer.indices {
                buffer[index] = 0
            }
        }
        try zeroArray(tokenInputArray, dataType: tokenInputDataType, label: "token input")
        if let cachePositionArray, let cachePositionDataType {
            try zeroArray(cachePositionArray, dataType: cachePositionDataType, label: "cache position")
        }
    }

    private mutating func prefill(promptTokens: [TokenID]) throws -> [Float] {
        guard !promptTokens.isEmpty else {
            throw CLIError.runtime("Cannot compare Core ML without prompt tokens.")
        }
        currentTokens.removeAll(keepingCapacity: true)

        switch contract {
        case .stateless:
            currentTokens = promptTokens
            for (position, token) in promptTokens.enumerated() {
                try writeToken(token, at: position, into: tokenInputArray, dataType: tokenInputDataType)
            }
            return try runStatelessPrediction(sequenceLength: promptTokens.count)
        case .stateful:
            var logits: [Float] = []
            logits.reserveCapacity(vocabSize)
            for token in promptTokens {
                logits = try runStatefulPrediction(token: token, cachePosition: currentTokens.count)
                currentTokens.append(token)
            }
            return logits
        }
    }

    private mutating func decode(nextToken: TokenID) throws -> [Float] {
        guard currentTokens.count < maxSequenceTokens else {
            throw CLIError.runtime("Core ML decode overflow at sequence length \(maxSequenceTokens)")
        }

        switch contract {
        case .stateless:
            currentTokens.append(nextToken)
            try writeToken(
                nextToken,
                at: currentTokens.count - 1,
                into: tokenInputArray,
                dataType: tokenInputDataType
            )
            return try runStatelessPrediction(sequenceLength: currentTokens.count)
        case .stateful:
            let logits = try runStatefulPrediction(token: nextToken, cachePosition: currentTokens.count)
            currentTokens.append(nextToken)
            return logits
        }
    }

    private mutating func runStatelessPrediction(sequenceLength: Int) throws -> [Float] {
        let provider: MLDictionaryFeatureProvider
        do {
            provider = try MLDictionaryFeatureProvider(
                dictionary: [contract.tokenInput.name: MLFeatureValue(multiArray: tokenInputArray)]
            )
        } catch {
            throw CLIError.runtime("Core ML feature provider creation failed: \(error)")
        }

        let prediction: MLFeatureProvider
        do {
            prediction = try model.prediction(from: provider)
        } catch {
            throw CLIError.runtime("Core ML prediction failed: \(error)")
        }

        guard let outputArray = prediction.featureValue(for: contract.outputFeatureName)?.multiArrayValue else {
            throw CLIError.runtime("Core ML output '\(contract.outputFeatureName)' missing MLMultiArray value")
        }

        try extractLastHidden(from: outputArray, sequenceLength: sequenceLength)
        projectCurrentLogits()
        return stepLogits
    }

    private mutating func runStatefulPrediction(token: TokenID, cachePosition: Int) throws -> [Float] {
        guard let cachePositionArray, let cachePositionDataType, let cachePositionInput = contract.cachePositionInput else {
            throw CLIError.runtime("Stateful Core ML contract is missing cache-position input metadata.")
        }
        guard let state = modelState else {
            throw CLIError.runtime("Stateful Core ML prediction requested before model state was initialized.")
        }

        try writeToken(token, at: 0, into: tokenInputArray, dataType: tokenInputDataType)
        try writeScalar(Int32(cachePosition), into: cachePositionArray, dataType: cachePositionDataType, label: "cache position")

        let provider: MLDictionaryFeatureProvider
        do {
            provider = try MLDictionaryFeatureProvider(
                dictionary: [
                    contract.tokenInput.name: MLFeatureValue(multiArray: tokenInputArray),
                    cachePositionInput.name: MLFeatureValue(multiArray: cachePositionArray),
                ]
            )
        } catch {
            throw CLIError.runtime("Core ML feature provider creation failed: \(error)")
        }

        let modelBox = UncheckedSendableBox(model)
        let stateBox = UncheckedSendableBox(state)
        let providerBox = UncheckedSendableBox(provider)
        let prediction: MLFeatureProvider
        do {
            prediction = try runAsyncCoreMLOperation {
                if #available(macOS 15.0, *) {
                    return try await modelBox.value.prediction(
                        from: providerBox.value,
                        using: stateBox.value
                    )
                }
                throw CLIError.runtime("Stateful Core ML prediction requires macOS 15.0+")
            }
        } catch {
            throw CLIError.runtime("Core ML stateful prediction failed: \(error)")
        }

        guard let outputArray = prediction.featureValue(for: contract.outputFeatureName)?.multiArrayValue else {
            throw CLIError.runtime("Core ML output '\(contract.outputFeatureName)' missing MLMultiArray value")
        }

        try extractLastHidden(from: outputArray, sequenceLength: 1)
        projectCurrentLogits()
        return stepLogits
    }

    private mutating func extractLastHidden(
        from outputArray: MLMultiArray,
        sequenceLength: Int
    ) throws {
        let outputShape = outputArray.shape.map(\.intValue)
        guard outputShape.count == 3 else {
            throw CLIError.runtime("Expected rank-3 Core ML output, found rank \(outputShape.count)")
        }
        guard outputShape[0] == 1 else {
            throw CLIError.runtime("Expected Core ML output shape [1, seq, hidden], found \(outputShape)")
        }
        guard outputShape[2] == hiddenSize else {
            throw CLIError.runtime(
                "Core ML output hidden size \(outputShape[2]) does not match GPT-2 weights hidden size \(hiddenSize)"
            )
        }
        guard sequenceLength > 0, sequenceLength <= outputShape[1] else {
            throw CLIError.runtime(
                "Requested output slice \(sequenceLength) exceeds Core ML output sequence capacity \(outputShape[1])"
            )
        }

        let sequenceStride = outputArray.strides[1].intValue
        let hiddenStride = outputArray.strides[2].intValue
        let sequenceIndex = sequenceLength - 1

        switch outputArray.dataType {
        case .float16:
            let source = outputArray.dataPointer.bindMemory(to: Float16.self, capacity: outputArray.count)
            for dimIndex in 0..<hiddenSize {
                stepHidden[dimIndex] = Float(source[sequenceIndex * sequenceStride + dimIndex * hiddenStride])
            }
        case .float32:
            let source = outputArray.dataPointer.bindMemory(to: Float.self, capacity: outputArray.count)
            for dimIndex in 0..<hiddenSize {
                stepHidden[dimIndex] = source[sequenceIndex * sequenceStride + dimIndex * hiddenStride]
            }
        default:
            throw CLIError.runtime("Unsupported Core ML output dtype: \(outputArray.dataType.rawValue)")
        }
    }

    private mutating func projectCurrentLogits() {
        let normalized = normalizeHiddenForCoreMLReference(
            architecture: architecture,
            hidden: stepHidden,
            epsilon: epsilon,
            gamma: finalNormGamma,
            beta: finalNormBeta
        )
        stepNorm = normalized

        stepLogits.withUnsafeMutableBufferPointer { logitsBuffer in
            lmHead.withUnsafeBufferPointer { lmHeadBuffer in
                stepNorm.withUnsafeBufferPointer { normBuffer in
                    guard let logitsBase = logitsBuffer.baseAddress,
                          let lmHeadBase = lmHeadBuffer.baseAddress,
                          let normBase = normBuffer.baseAddress else {
                        return
                    }
                    cblas_sgemv(
                        CblasRowMajor,
                        CblasNoTrans,
                        Int32(vocabSize),
                        Int32(hiddenSize),
                        1.0,
                        lmHeadBase,
                        Int32(hiddenSize),
                        normBase,
                        1,
                        0.0,
                        logitsBase,
                        1
                    )
                }
            }
        }
    }

    private func makeInitialStateIfNeeded() throws -> MLState? {
        guard contract.isStateful else {
            return nil
        }
        if #available(macOS 15.0, *) {
            return model.makeState()
        }
        throw CLIError.runtime("Stateful Core ML execution requires macOS 15.0+")
    }

    private func zeroArray(
        _ array: MLMultiArray,
        dataType: MLMultiArrayDataType,
        label: String
    ) throws {
        switch dataType {
        case .int32:
            let pointer = array.dataPointer.bindMemory(to: Int32.self, capacity: array.count)
            for index in 0..<array.count {
                pointer[index] = 0
            }
        default:
            throw CLIError.runtime("Unsupported Core ML \(label) dtype: \(dataType.rawValue)")
        }
    }

    private func writeToken(
        _ token: TokenID,
        at position: Int,
        into array: MLMultiArray,
        dataType: MLMultiArrayDataType
    ) throws {
        guard Int(token) < vocabSize else {
            throw CLIError.runtime("Token \(token) exceeds Core ML vocab size \(vocabSize)")
        }
        guard position >= 0, position < maxSequenceTokens else {
            throw CLIError.runtime("Position \(position) exceeds Core ML sequence length \(maxSequenceTokens)")
        }

        guard array.shape.count >= 2 else {
            throw CLIError.runtime("Core ML token input must be rank-2, found shape \(array.shape)")
        }
        let sequenceStride = array.strides[1].intValue

        switch dataType {
        case .int32:
            let pointer = array.dataPointer.bindMemory(to: Int32.self, capacity: array.count)
            pointer[position * sequenceStride] = Int32(token)
        default:
            throw CLIError.runtime("Unsupported Core ML token input dtype: \(dataType.rawValue)")
        }
    }

    private func writeScalar(
        _ value: Int32,
        into array: MLMultiArray,
        dataType: MLMultiArrayDataType,
        label: String
    ) throws {
        guard array.count == 1 else {
            throw CLIError.runtime("Core ML \(label) array must contain exactly one value, found \(array.count)")
        }

        switch dataType {
        case .int32:
            let pointer = array.dataPointer.bindMemory(to: Int32.self, capacity: array.count)
            pointer[0] = value
        default:
            throw CLIError.runtime("Unsupported Core ML \(label) dtype: \(dataType.rawValue)")
        }
    }

    private static func probeComputePlan(
        modelURL: URL,
        configuration: MLModelConfiguration,
        computeUnits: String
    ) throws -> Bool? {
        guard #available(macOS 14.4, *) else {
            return nil
        }
        let urlBox = UncheckedSendableBox(modelURL)
        let configurationBox = UncheckedSendableBox(configuration)
        do {
            _ = try runAsyncCoreMLOperation {
                try await MLComputePlan.load(
                    contentsOf: urlBox.value,
                    configuration: configurationBox.value
                )
            }
            return true
        } catch {
            if computeUnits == "cpu_and_neural_engine" {
                let diagnostics = diagnostics(
                    isStateful: false,
                    loadSucceeded: false,
                    computePlanSucceeded: false,
                    error: error
                )
                throw CoreMLReferenceFailure(
                    message: failureMessage(prefix: "Core ML compute plan failed", error: error),
                    diagnostics: diagnostics
                )
            }
            return false
        }
    }

    private static func diagnostics(
        isStateful: Bool,
        loadSucceeded: Bool,
        computePlanSucceeded: Bool?,
        error: Error
    ) -> CoreMLReferenceDiagnostics {
        let nsError = error as NSError
        return CoreMLReferenceDiagnostics(
            isStateful: isStateful,
            loadSucceeded: loadSucceeded,
            computePlanSucceeded: computePlanSucceeded,
            errorDomain: nsError.domain,
            errorCode: nsError.code,
            errorMessage: nsError.localizedDescription
        )
    }

    private static func failureMessage(prefix: String, error: Error) -> String {
        let nsError = error as NSError
        return "\(prefix): \(nsError.domain) \(nsError.code) \(nsError.localizedDescription)"
    }
}

struct ReusableGPT2CoreMLBenchmarkRunner {
    private var runner: ReusableCoreMLBenchmarkRunner
    private var shouldReportCompileTime = true

    init(
        modelPath: String,
        weightsDir: String,
        sequenceLength: Int,
        computeUnits: String
    ) throws {
        self.runner = try ReusableCoreMLBenchmarkRunner(
            modelPath: modelPath,
            weightsDir: weightsDir,
            architecture: .gpt2,
            sequenceLength: sequenceLength,
            computeUnits: computeUnits
        )
    }

    mutating func benchmark(
        promptTokens: [TokenID],
        maxTokens: Int,
        temperature: Float,
        warmup: Int,
        iterations: Int,
        seed: Int
    ) throws -> CoreMLComparisonResult {
        let result = try runner.benchmark(
            promptTokens: promptTokens,
            maxTokens: maxTokens,
            temperature: temperature,
            warmup: warmup,
            iterations: iterations,
            seed: seed
        )
        defer { shouldReportCompileTime = false }
        guard shouldReportCompileTime else {
            return CoreMLComparisonResult(
                generatedTokens: result.generatedTokens,
                compileTimeMs: 0,
                firstTokenLatencyMs: result.firstTokenLatencyMs,
                tokensPerSecond: result.tokensPerSecond,
                medianTokenMs: result.medianTokenMs,
                p95TokenMs: result.p95TokenMs,
                tokenLatenciesMs: result.tokenLatenciesMs,
                totalTimeMs: result.totalTimeMs,
                computeUnits: result.computeUnits,
                seqLen: result.seqLen,
                isStateful: result.isStateful,
                loadSucceeded: result.loadSucceeded,
                computePlanSucceeded: result.computePlanSucceeded,
                errorDomain: result.errorDomain,
                errorCode: result.errorCode,
                errorMessage: result.errorMessage
            )
        }
        return result
    }
}

struct ReusableCoreMLBenchmarkRunner {
    private var runner: NativeCoreMLReferenceRunner
    private var shouldReportCompileTime = true

    init(
        modelPath: String,
        weightsDir: String,
        architecture: MultiModelConfig.Architecture,
        sequenceLength: Int,
        computeUnits: String
    ) throws {
        self.runner = try NativeCoreMLReferenceRunner(
            modelPath: modelPath,
            weightsDir: weightsDir,
            architecture: architecture,
            sequenceLength: sequenceLength,
            computeUnits: computeUnits
        )
    }

    mutating func benchmark(
        promptTokens: [TokenID],
        maxTokens: Int,
        temperature: Float,
        warmup: Int,
        iterations: Int,
        seed: Int
    ) throws -> CoreMLComparisonResult {
        let result = try runner.benchmark(
            promptTokens: promptTokens,
            maxTokens: maxTokens,
            temperature: temperature,
            warmup: warmup,
            iterations: iterations,
            seed: seed
        )
        defer { shouldReportCompileTime = false }
        guard shouldReportCompileTime else {
            return CoreMLComparisonResult(
                generatedTokens: result.generatedTokens,
                compileTimeMs: 0,
                firstTokenLatencyMs: result.firstTokenLatencyMs,
                tokensPerSecond: result.tokensPerSecond,
                medianTokenMs: result.medianTokenMs,
                p95TokenMs: result.p95TokenMs,
                tokenLatenciesMs: result.tokenLatenciesMs,
                totalTimeMs: result.totalTimeMs,
                computeUnits: result.computeUnits,
                seqLen: result.seqLen,
                isStateful: result.isStateful,
                loadSucceeded: result.loadSucceeded,
                computePlanSucceeded: result.computePlanSucceeded,
                errorDomain: result.errorDomain,
                errorCode: result.errorCode,
                errorMessage: result.errorMessage
            )
        }
        return result
    }
}

private func loadNativeCoreMLTopLevelWeights(
    weightsDir: String,
    architecture: MultiModelConfig.Architecture
) throws -> NativeCoreMLTopLevelWeights {
    let root = URL(fileURLWithPath: NSString(string: weightsDir).expandingTildeInPath, isDirectory: true).standardizedFileURL
    var isDirectory: ObjCBool = false
    guard FileManager.default.fileExists(atPath: root.path, isDirectory: &isDirectory), isDirectory.boolValue else {
        throw CLIError.usage("Weights directory does not exist: \(weightsDir)")
    }

    let gammaPath = try requiredHelperFile(
        root: root,
        candidates: ["final_norm_gamma.bin", "ln_f_gamma.bin", "rms_final.bin", "final_norm.bin"],
        label: "final norm gamma"
    )
    let finalNormGamma = try loadBlobWeightTable(at: gammaPath.path)
    guard !finalNormGamma.isEmpty else {
        throw CLIError.runtime("Final norm gamma is empty at \(gammaPath.path)")
    }
    let hiddenSize = finalNormGamma.count

    let lmHeadPath = try requiredHelperFile(
        root: root,
        candidates: ["lm_head.bin", "classifier.bin"],
        label: "lm head"
    )
    let lmHead = try loadBlobWeightTable(at: lmHeadPath.path)
    guard lmHead.count.isMultiple(of: hiddenSize) else {
        throw CLIError.runtime("lm head count \(lmHead.count) is not divisible by hidden size \(hiddenSize)")
    }
    let vocabSize = lmHead.count / hiddenSize

    let finalNormBeta: [Float]?
    switch architecture {
    case .gpt2:
        let betaPath = try requiredHelperFile(
            root: root,
            candidates: ["final_norm_beta.bin", "ln_f_beta.bin", "rms_final_beta.bin"],
            label: "final norm beta"
        )
        finalNormBeta = try loadBlobWeightTable(at: betaPath.path, expectedCount: hiddenSize)
    case .llama:
        finalNormBeta = nil
    }

    return NativeCoreMLTopLevelWeights(
        hiddenSize: hiddenSize,
        vocabSize: vocabSize,
        finalNormGamma: finalNormGamma,
        finalNormBeta: finalNormBeta,
        lmHead: lmHead
    )
}

private func requiredHelperFile(root: URL, candidates: [String], label: String) throws -> URL {
    let fileManager = FileManager.default
    for candidate in candidates {
        let url = root.appendingPathComponent(candidate)
        if fileManager.fileExists(atPath: url.path) {
            return url
        }
    }
    throw CLIError.runtime("Missing \(label) in \(root.path)")
}

private func loadBlobWeightTable(at path: String, expectedCount: Int? = nil) throws -> [Float] {
    let values: [Float]
    do {
        values = try BlobWeightLoader.load(from: path)
    } catch {
        throw CLIError.runtime("Failed to load weight blob \(path): \(error)")
    }
    if let expectedCount, values.count != expectedCount {
        throw CLIError.runtime("Invalid weight count for \(path): expected \(expectedCount), got \(values.count)")
    }
    return values
}

private func parseCoreMLComputeUnits(_ rawValue: String) throws -> MLComputeUnits {
    switch rawValue {
    case "all":
        return .all
    case "cpu_and_neural_engine":
        return .cpuAndNeuralEngine
    case "cpu_and_gpu":
        return .cpuAndGPU
    case "cpu_only":
        return .cpuOnly
    default:
        throw CLIError.usage("Expected --coreml-compute-units all|cpu_and_neural_engine|cpu_and_gpu|cpu_only")
    }
}

private func sampleNativeToken<R: RandomNumberGenerator>(
    from logits: [Float],
    temperature: Float,
    using rng: inout R
) -> TokenID {
    if temperature <= 0 {
        return TokenID(logits.enumerated().max(by: { $0.element < $1.element })?.offset ?? 0)
    }

    let maxLogit = logits.max() ?? 0
    var scaled = [Double](repeating: 0, count: logits.count)
    var total = 0.0
    for index in logits.indices {
        let value = exp(Double((logits[index] - maxLogit) / temperature))
        scaled[index] = value
        total += value
    }
    if !total.isFinite || total <= 0 {
        return TokenID(logits.enumerated().max(by: { $0.element < $1.element })?.offset ?? 0)
    }

    var threshold = Double.random(in: 0..<total, using: &rng)
    for index in scaled.indices {
        threshold -= scaled[index]
        if threshold <= 0 {
            return TokenID(index)
        }
    }
    return TokenID(max(0, scaled.count - 1))
}

private func nativePercentile(_ values: [Double], percentile: Double) -> Double {
    guard !values.isEmpty else { return 0 }
    let sorted = values.sorted()
    let clamped = min(max(percentile, 0), 1)
    let index = clamped * Double(sorted.count - 1)
    let lower = Int(index.rounded(.down))
    let upper = min(lower + 1, sorted.count - 1)
    if lower == upper {
        return sorted[lower]
    }
    let fraction = index - Double(lower)
    return sorted[lower] + ((sorted[upper] - sorted[lower]) * fraction)
}

@inline(__always)
private func throughput(tokensGenerated: Int, totalTimeMs: Double) -> Double {
    guard totalTimeMs > 0 else { return 0 }
    return Double(tokensGenerated) * 1000.0 / totalTimeMs
}

@inline(__always)
private func monotonicNow() -> UInt64 {
    DispatchTime.now().uptimeNanoseconds
}

@inline(__always)
private func monotonicMilliseconds(since start: UInt64) -> Double {
    Double(DispatchTime.now().uptimeNanoseconds &- start) / 1_000_000.0
}

func detectDemoDefaults() -> DemoDefaults {
    let fileManager = FileManager()
    let repoRoot = locateRepoRoot(fileManager: fileManager)
    let stateRoot = preferredStateRoot(fileManager: fileManager)
    let cacheRoot = preferredCacheRoot(fileManager: fileManager)
    let scriptsDir = resolvedScriptsDirectory(repoRoot: repoRoot)
    let legacyArtifactsRoot = repoRoot?.appendingPathComponent(".artifacts", isDirectory: true)
    let workingDirectory = repoRoot ?? stateRoot

    return DemoDefaults(
        repoRoot: repoRoot,
        workingDirectory: workingDirectory,
        stateRoot: stateRoot,
        cacheRoot: cacheRoot,
        reportsRoot: stateRoot.appendingPathComponent("reports", isDirectory: true),
        hfCacheRoot: cacheRoot.appendingPathComponent("huggingface", isDirectory: true),
        weightsDir: stateRoot.appendingPathComponent("demo/gpt2_124m", isDirectory: true),
        tokenizerDir: stateRoot.appendingPathComponent("demo/gpt2_tokenizer", isDirectory: true),
        coreMLDir: stateRoot.appendingPathComponent("coreml/gpt2_124m", isDirectory: true),
        toolsVenvDir: stateRoot.appendingPathComponent("tools/python/gpt2-tools-venv", isDirectory: true),
        scriptsDir: scriptsDir,
        legacyArtifactsRoot: legacyArtifactsRoot
    )
}

func shouldUseDefaultGPT2Demo(_ options: Options) -> Bool {
    if options.prepareDemo || options.command == .demo {
        return true
    }
    guard options.weightsDir == nil else {
        return false
    }
    guard let modelName = options.modelName else {
        return true
    }
    return canonicalModelName(for: modelName) == ModelRegistry.gpt2_124m.name
}

func nextPowerOfTwo(_ value: Int) -> Int {
    guard value > 1 else { return 1 }
    var result = 1
    while result < value {
        result <<= 1
    }
    return result
}

func ensureGPT2DemoWeightsAndTokenizer(
    defaults: DemoDefaults,
    allowBootstrap: Bool
) throws {
    try ensureStateDirectories(defaults)
    try migrateLegacyArtifactsIfNeeded(defaults)

    if hasGPT2TokenizerAssets(in: defaults.tokenizerDir),
       FileManager().fileExists(atPath: defaults.weightsDir.appendingPathComponent("metadata.json").path)
    {
        return
    }

    guard allowBootstrap else {
        throw CLIError.usage(
            "Missing default GPT-2 demo artifacts. Re-run without --no-bootstrap, run `espresso-generate doctor`, or pass explicit --weights/--tokenizer paths."
        )
    }

    let bootstrapScript = try requireHelperScript(
        defaults,
        named: "bootstrap_gpt2_demo.py",
        purpose: "Bootstrapping default GPT-2 demo artifacts"
    )
    stderrLine("Preparing default GPT-2 demo artifacts in \(defaults.stateRoot.path)")
    let python = try ensurePythonEnvironment(
        defaults: defaults,
        requiredModules: ["numpy", "torch", "transformers"]
    )
    try runProcessStreaming(
        executable: python,
        arguments: [
            bootstrapScript.path,
            "--weights-out", defaults.weightsDir.path,
            "--tokenizer-out", defaults.tokenizerDir.path,
            "--cache-dir", defaults.hfCacheRoot.path,
        ],
        workingDirectory: defaults.workingDirectory,
        environment: pythonEnvironment(defaults: defaults)
    )
}

func ensureGPT2CoreMLModel(
    defaults: DemoDefaults,
    weightsDir: String,
    sequenceLength: Int,
    explicitModelPath: String?,
    allowBootstrap: Bool
) throws -> String {
    try ensureStateDirectories(defaults)
    try migrateLegacyArtifactsIfNeeded(defaults)

    if let explicitModelPath, !explicitModelPath.isEmpty {
        let expanded = NSString(string: explicitModelPath).expandingTildeInPath
        guard FileManager().fileExists(atPath: expanded) else {
            throw CLIError.usage("Core ML model path does not exist: \(explicitModelPath)")
        }
        return URL(fileURLWithPath: expanded).standardizedFileURL.path
    }

    let weightsURL = URL(fileURLWithPath: weightsDir, isDirectory: true).standardizedFileURL
    let modelURL = defaults.coreMLModelURL(sequenceLength: sequenceLength, weightsDirURL: weightsURL)
    if FileManager().fileExists(atPath: modelURL.path) {
        return modelURL.path
    }

    if let fallbackModel = locateCompatibleCoreMLModel(
        in: modelURL.deletingLastPathComponent(),
        minimumSequenceLength: sequenceLength
    ) {
        return fallbackModel.path
    }

    if let legacyArtifactsRoot = defaults.legacyArtifactsRoot {
        let legacyModel = legacyArtifactsRoot
            .appendingPathComponent("gpt2_coreml", isDirectory: true)
            .appendingPathComponent("gpt2_seq\(sequenceLength).mlpackage")
        if FileManager().fileExists(atPath: legacyModel.path) {
            try copyItemIfMissing(from: legacyModel, to: modelURL)
            return modelURL.path
        }
        if let fallbackModel = locateCompatibleCoreMLModel(
            in: legacyModel.deletingLastPathComponent(),
            minimumSequenceLength: sequenceLength
        ) {
            return fallbackModel.path
        }
    }

    guard allowBootstrap else {
        throw CLIError.usage(
            "Missing GPT-2 Core ML baseline for sequence length \(sequenceLength). Re-run without --no-bootstrap, run `espresso-generate doctor`, or pass --coreml-model."
        )
    }

    let exportScript = try requireHelperScript(
        defaults,
        named: "export_gpt2_coreml.py",
        purpose: "Exporting the GPT-2 Core ML baseline"
    )
    stderrLine("Exporting GPT-2 Core ML baseline for seq_len=\(sequenceLength)")
    let python = try ensurePythonEnvironment(
        defaults: defaults,
        requiredModules: ["numpy", "torch", "coremltools"]
    )
    try runProcessStreaming(
        executable: python,
        arguments: [
            exportScript.path,
            "--weights", weightsURL.path,
            "--output", modelURL.path,
            "--seq-len", String(sequenceLength),
        ],
        workingDirectory: defaults.workingDirectory,
        environment: pythonEnvironment(defaults: defaults)
    )
    return modelURL.path
}

private func locateCompatibleCoreMLModel(in directory: URL, minimumSequenceLength: Int) -> URL? {
    let fileManager = FileManager.default
    guard let children = try? fileManager.contentsOfDirectory(
        at: directory,
        includingPropertiesForKeys: nil,
        options: [.skipsHiddenFiles]
    ) else {
        return nil
    }

    let candidates = children.compactMap { url -> (sequenceLength: Int, url: URL)? in
        guard url.pathExtension == "mlpackage" else { return nil }
        let basename = url.deletingPathExtension().lastPathComponent
        guard basename.hasPrefix("gpt2_seq"),
              let sequenceLength = Int(basename.dropFirst("gpt2_seq".count))
        else {
            return nil
        }
        return (sequenceLength, url)
    }

    return candidates
        .filter { $0.sequenceLength >= minimumSequenceLength }
        .sorted {
            if $0.sequenceLength == $1.sequenceLength {
                return $0.url.path < $1.url.path
            }
            return $0.sequenceLength < $1.sequenceLength
        }
        .first?
        .url
}

func runGPT2CoreMLReference(
    defaults: DemoDefaults,
    coreMLModelPath: String,
    weightsDir: String,
    promptTokens: [TokenID],
    sequenceLength: Int,
    maxTokens: Int,
    temperature: Float,
    warmup: Int,
    iterations: Int,
    computeUnits: String,
    seed: Int,
    allowBootstrap: Bool
) throws -> CoreMLComparisonResult {
    guard !promptTokens.isEmpty else {
        throw CLIError.runtime("Cannot compare Core ML without prompt tokens.")
    }

    if let referenceScript = defaults.coreMLReferenceScriptURL {
        let python = try ensurePythonEnvironment(
            defaults: defaults,
            requiredModules: ["numpy", "coremltools"]
        )
        let promptTokenString = promptTokens.map(String.init).joined(separator: ",")
        let output = try runProcessCaptured(
            executable: python,
            arguments: [
                referenceScript.path,
                "--coreml-model", coreMLModelPath,
                "--weights", weightsDir,
                "--prompt-tokens", promptTokenString,
                "--seq-len", String(sequenceLength),
                "--max-tokens", String(maxTokens),
                "--temperature", String(temperature),
                "--warmup", String(warmup),
                "--iterations", String(iterations),
                "--seed", String(seed),
                "--compute-units", computeUnits,
            ],
            workingDirectory: defaults.workingDirectory,
            environment: pythonEnvironment(defaults: defaults),
            allowBootstrap: allowBootstrap
        )

        guard output.status == 0 else {
            throw CLIError.runtime(
                output.stderr.isEmpty
                    ? "Core ML reference runner failed with status \(output.status)"
                    : output.stderr.trimmingCharacters(in: .whitespacesAndNewlines)
            )
        }

        let decoder = JSONDecoder()
        do {
            return try decoder.decode(CoreMLComparisonResult.self, from: Data(output.stdout.utf8))
        } catch {
            throw CLIError.runtime("Failed to decode Core ML comparison output: \(error)")
        }
    }

    return try runCoreMLReference(
        defaults: defaults,
        architecture: .gpt2,
        coreMLModelPath: coreMLModelPath,
        weightsDir: weightsDir,
        promptTokens: promptTokens,
        sequenceLength: sequenceLength,
        maxTokens: maxTokens,
        temperature: temperature,
        warmup: warmup,
        iterations: iterations,
        computeUnits: computeUnits,
        seed: seed
    )
}

func runGPT2CoreMLReferenceStreaming(
    defaults: DemoDefaults,
    coreMLModelPath: String,
    weightsDir: String,
    promptTokens: [TokenID],
    sequenceLength: Int,
    maxTokens: Int,
    temperature: Float,
    computeUnits: String,
    seed: Int,
    allowBootstrap: Bool,
    onEvent: @escaping (CoreMLStreamEvent) -> Void
) throws -> CoreMLComparisonResult {
    guard !promptTokens.isEmpty else {
        throw CLIError.runtime("Cannot compare Core ML without prompt tokens.")
    }
    if let referenceScript = defaults.coreMLReferenceScriptURL {
        let python = try ensurePythonEnvironment(
            defaults: defaults,
            requiredModules: ["numpy", "coremltools"]
        )
        let promptTokenString = promptTokens.map(String.init).joined(separator: ",")

        var completed: CoreMLComparisonResult?
        let output = try runProcessStreamingLinesCapture(
            executable: python,
            arguments: [
                referenceScript.path,
                "--coreml-model", coreMLModelPath,
                "--weights", weightsDir,
                "--prompt-tokens", promptTokenString,
                "--seq-len", String(sequenceLength),
                "--max-tokens", String(maxTokens),
                "--temperature", String(temperature),
                "--warmup", "0",
                "--iterations", "1",
                "--seed", String(seed),
                "--compute-units", computeUnits,
                "--emit-events",
            ],
            workingDirectory: defaults.workingDirectory,
            environment: pythonEnvironment(defaults: defaults),
            allowBootstrap: allowBootstrap,
            onStdoutLine: { line in
                guard !line.isEmpty else { return }
                do {
                    let raw = try JSONDecoder().decode(RawCoreMLStreamEvent.self, from: Data(line.utf8))
                    switch raw.type {
                    case "compile":
                        if let compileTimeMs = raw.compileTimeMs,
                           let computeUnits = raw.computeUnits,
                           let seqLen = raw.seqLen
                        {
                            onEvent(.compile(compileTimeMs: compileTimeMs, computeUnits: computeUnits, seqLen: seqLen))
                        }
                    case "token":
                        guard let token = raw.token,
                              let tokenIndex = raw.tokenIndex,
                              let elapsedMs = raw.elapsedMs,
                              let tokenLatencyMs = raw.tokenLatencyMs,
                              let tokensPerSecond = raw.tokensPerSecond,
                              token >= 0,
                              token <= Int(TokenID.max)
                        else {
                            return
                        }
                        onEvent(
                            .token(
                                token: TokenID(token),
                                tokenIndex: tokenIndex,
                                elapsedMs: elapsedMs,
                                tokenLatencyMs: tokenLatencyMs,
                                tokensPerSecond: tokensPerSecond
                            )
                        )
                    case "completed":
                        let result = try JSONDecoder().decode(CoreMLComparisonResult.self, from: Data(line.utf8))
                        completed = result
                        onEvent(.completed(result))
                    default:
                        break
                    }
                } catch {
                    stderrLine("espresso-generate warning: failed to parse Core ML stream event: \(error)")
                }
            }
        )

        guard output.status == 0 else {
            throw CLIError.runtime(
                output.stderr.isEmpty
                    ? "Core ML reference runner failed with status \(output.status)"
                    : output.stderr.trimmingCharacters(in: .whitespacesAndNewlines)
            )
        }
        guard let completed else {
            throw CLIError.runtime("Core ML reference runner did not emit a completion event.")
        }
        return completed
    }

    return try runCoreMLReferenceStreaming(
        defaults: defaults,
        architecture: .gpt2,
        coreMLModelPath: coreMLModelPath,
        weightsDir: weightsDir,
        promptTokens: promptTokens,
        sequenceLength: sequenceLength,
        maxTokens: maxTokens,
        temperature: temperature,
        computeUnits: computeUnits,
        seed: seed,
        onEvent: onEvent
    )
}

func runCoreMLReference(
    defaults: DemoDefaults,
    architecture: MultiModelConfig.Architecture,
    coreMLModelPath: String,
    weightsDir: String,
    promptTokens: [TokenID],
    sequenceLength: Int,
    maxTokens: Int,
    temperature: Float,
    warmup: Int,
    iterations: Int,
    computeUnits: String,
    seed: Int
) throws -> CoreMLComparisonResult {
    _ = defaults
    guard !promptTokens.isEmpty else {
        throw CLIError.runtime("Cannot compare Core ML without prompt tokens.")
    }

    var runner = try NativeCoreMLReferenceRunner(
        modelPath: coreMLModelPath,
        weightsDir: weightsDir,
        architecture: architecture,
        sequenceLength: sequenceLength,
        computeUnits: computeUnits
    )
    return try runner.benchmark(
        promptTokens: promptTokens,
        maxTokens: maxTokens,
        temperature: temperature,
        warmup: warmup,
        iterations: iterations,
        seed: seed
    )
}

func runCoreMLReferenceStreaming(
    defaults: DemoDefaults,
    architecture: MultiModelConfig.Architecture,
    coreMLModelPath: String,
    weightsDir: String,
    promptTokens: [TokenID],
    sequenceLength: Int,
    maxTokens: Int,
    temperature: Float,
    computeUnits: String,
    seed: Int,
    onEvent: @escaping (CoreMLStreamEvent) -> Void
) throws -> CoreMLComparisonResult {
    _ = defaults
    guard !promptTokens.isEmpty else {
        throw CLIError.runtime("Cannot compare Core ML without prompt tokens.")
    }

    var runner = try NativeCoreMLReferenceRunner(
        modelPath: coreMLModelPath,
        weightsDir: weightsDir,
        architecture: architecture,
        sequenceLength: sequenceLength,
        computeUnits: computeUnits
    )
    return try runner.benchmark(
        promptTokens: promptTokens,
        maxTokens: maxTokens,
        temperature: temperature,
        warmup: 0,
        iterations: 1,
        seed: seed,
        onEvent: onEvent
    )
}

private func locateRepoRoot(fileManager: FileManager) -> URL? {
    if let override = ProcessInfo.processInfo.environment["ESPRESSO_REPO_ROOT"], !override.isEmpty {
        let candidatePath = NSString(string: override).expandingTildeInPath
        if isRepoRoot(candidatePath, fileManager: fileManager) {
            return URL(fileURLWithPath: candidatePath, isDirectory: true).standardizedFileURL
        }
    }

    let currentDirectoryPath = NSString(string: fileManager.currentDirectoryPath).standardizingPath
    if let resolved = ascendToRepoRoot(startPath: currentDirectoryPath, fileManager: fileManager) {
        return URL(fileURLWithPath: resolved, isDirectory: true).standardizedFileURL
    }

    if let executablePath = Bundle.main.executableURL?.path,
       let resolved = ascendToRepoRoot(
           startPath: (NSString(string: executablePath).deletingLastPathComponent as NSString).standardizingPath,
           fileManager: fileManager
       )
    {
        return URL(fileURLWithPath: resolved, isDirectory: true).standardizedFileURL
    }

    let argv0 = CommandLine.arguments.first ?? ""
    if !argv0.isEmpty {
        let expanded = NSString(string: argv0).expandingTildeInPath
        let absoluteExecutablePath: String
        if expanded.hasPrefix("/") {
            absoluteExecutablePath = expanded
        } else {
            absoluteExecutablePath = (fileManager.currentDirectoryPath as NSString).appendingPathComponent(expanded)
        }
        let executableDirectory = (absoluteExecutablePath as NSString).deletingLastPathComponent
        if let resolved = ascendToRepoRoot(
            startPath: (NSString(string: executableDirectory).standardizingPath),
            fileManager: fileManager
        ) {
            return URL(fileURLWithPath: resolved, isDirectory: true).standardizedFileURL
        }
    }

    return nil
}

private func ascendToRepoRoot(startPath: String, fileManager: FileManager) -> String? {
    var currentPath = NSString(string: startPath).standardizingPath
    while true {
        if isRepoRoot(currentPath, fileManager: fileManager) {
            return currentPath
        }
        let parentPath = (currentPath as NSString).deletingLastPathComponent
        if parentPath.isEmpty || parentPath == currentPath {
            return nil
        }
        currentPath = parentPath
    }
}

private func isRepoRoot(_ candidatePath: String, fileManager: FileManager) -> Bool {
    let scriptsPath = (candidatePath as NSString).appendingPathComponent("scripts")
    let packagePath = (candidatePath as NSString).appendingPathComponent("Package.swift")
    var isDirectory: ObjCBool = false
    return fileManager.fileExists(atPath: packagePath) &&
        fileManager.fileExists(atPath: scriptsPath, isDirectory: &isDirectory) &&
        isDirectory.boolValue
}

private func preferredStateRoot(fileManager: FileManager) -> URL {
    if let override = ProcessInfo.processInfo.environment["ESPRESSO_HOME"], !override.isEmpty {
        return URL(fileURLWithPath: NSString(string: override).expandingTildeInPath, isDirectory: true).standardizedFileURL
    }
    let base = fileManager.urls(for: .applicationSupportDirectory, in: .userDomainMask).first ??
        URL(fileURLWithPath: NSHomeDirectory()).appendingPathComponent("Library/Application Support", isDirectory: true)
    return base.appendingPathComponent("Espresso", isDirectory: true)
}

private func preferredCacheRoot(fileManager: FileManager) -> URL {
    if let override = ProcessInfo.processInfo.environment["ESPRESSO_CACHE_HOME"], !override.isEmpty {
        return URL(fileURLWithPath: NSString(string: override).expandingTildeInPath, isDirectory: true).standardizedFileURL
    }
    let base = fileManager.urls(for: .cachesDirectory, in: .userDomainMask).first ??
        URL(fileURLWithPath: NSHomeDirectory()).appendingPathComponent("Library/Caches", isDirectory: true)
    return base.appendingPathComponent("Espresso", isDirectory: true)
}

private func resolvedScriptsDirectory(repoRoot: URL?) -> URL? {
    if let override = ProcessInfo.processInfo.environment["ESPRESSO_SCRIPTS_DIR"], !override.isEmpty {
        return URL(fileURLWithPath: NSString(string: override).expandingTildeInPath, isDirectory: true).standardizedFileURL
    }
    return repoRoot?.appendingPathComponent("scripts", isDirectory: true)
}

private func requireScriptsDirectory(_ defaults: DemoDefaults) throws -> URL {
    guard let scriptsDir = defaults.scriptsDir else {
        throw CLIError.runtime(
            "Espresso helper scripts are unavailable. Run from a repository checkout or set ESPRESSO_SCRIPTS_DIR to the scripts directory."
        )
    }
    var isDirectory: ObjCBool = false
    guard FileManager().fileExists(atPath: scriptsDir.path, isDirectory: &isDirectory), isDirectory.boolValue else {
        throw CLIError.runtime(
            "Espresso helper scripts directory is unavailable. Run from a repository checkout or set ESPRESSO_SCRIPTS_DIR to the scripts directory."
        )
    }
    return scriptsDir
}

private func requireHelperScript(
    _ defaults: DemoDefaults,
    named name: String,
    purpose: String
) throws -> URL {
    let scriptsDir = try requireScriptsDirectory(defaults)
    let scriptURL = scriptsDir.appendingPathComponent(name)
    guard FileManager().fileExists(atPath: scriptURL.path) else {
        throw CLIError.runtime(
            "\(purpose) requires the missing helper script \(name). Run from a repository checkout that includes helper scripts or set ESPRESSO_SCRIPTS_DIR."
        )
    }
    return scriptURL
}

private func ensureStateDirectories(_ defaults: DemoDefaults) throws {
    let fileManager = FileManager()
    for directory in [
        defaults.stateRoot,
        defaults.cacheRoot,
        defaults.reportsRoot,
        defaults.hfCacheRoot,
        defaults.weightsDir.deletingLastPathComponent(),
        defaults.tokenizerDir.deletingLastPathComponent(),
        defaults.coreMLDir,
        defaults.toolsVenvDir.deletingLastPathComponent(),
    ] {
        try fileManager.createDirectory(at: directory, withIntermediateDirectories: true, attributes: nil)
    }
}

private func migrateLegacyArtifactsIfNeeded(_ defaults: DemoDefaults) throws {
    guard let legacyArtifactsRoot = defaults.legacyArtifactsRoot else {
        return
    }
    let legacyWeights = legacyArtifactsRoot.appendingPathComponent("gpt2_124m", isDirectory: true)
    let legacyTokenizer = legacyArtifactsRoot.appendingPathComponent("gpt2_tokenizer", isDirectory: true)
    let legacyCoreML = legacyArtifactsRoot.appendingPathComponent("gpt2_coreml", isDirectory: true)

    if FileManager().fileExists(atPath: legacyWeights.path),
       !FileManager().fileExists(atPath: defaults.weightsDir.path)
    {
        try copyItemIfMissing(from: legacyWeights, to: defaults.weightsDir)
    }
    if FileManager().fileExists(atPath: legacyTokenizer.path),
       !FileManager().fileExists(atPath: defaults.tokenizerDir.path)
    {
        try copyItemIfMissing(from: legacyTokenizer, to: defaults.tokenizerDir)
    }
    if FileManager().fileExists(atPath: legacyCoreML.path),
       !FileManager().fileExists(atPath: defaults.coreMLDir.path)
    {
        try copyItemIfMissing(from: legacyCoreML, to: defaults.coreMLDir)
    }
}

private func copyItemIfMissing(from source: URL, to destination: URL) throws {
    let fileManager = FileManager()
    if fileManager.fileExists(atPath: destination.path) {
        return
    }
    try fileManager.createDirectory(at: destination.deletingLastPathComponent(), withIntermediateDirectories: true, attributes: nil)
    try fileManager.copyItem(at: source, to: destination)
}

private func ensurePythonEnvironment(
    defaults: DemoDefaults,
    requiredModules: [String]
) throws -> String {
    if let override = ProcessInfo.processInfo.environment["ESPRESSO_TOOLS_PYTHON"], !override.isEmpty {
        if try pythonSupportsModules(override, requiredModules: requiredModules, defaults: defaults) {
            return override
        }
        throw CLIError.runtime("ESPRESSO_TOOLS_PYTHON is set but missing required modules: \(requiredModules.joined(separator: ", "))")
    }

    let bootstrapPython = try preferredBootstrapPython(defaults: defaults)
    let versionTag = try pythonVersionTag(bootstrapPython, defaults: defaults)
    let managedVenvDir = defaults.toolsVenvDir.deletingLastPathComponent()
        .appendingPathComponent("gpt2-tools-\(versionTag)", isDirectory: true)
    let managedPython = managedVenvDir.appendingPathComponent("bin/python3").path
    if try pythonSupportsModules(managedPython, requiredModules: requiredModules, defaults: defaults) {
        return managedPython
    }

    let fileManager = FileManager()
    try fileManager.createDirectory(
        at: managedVenvDir.deletingLastPathComponent(),
        withIntermediateDirectories: true
    )
    if !fileManager.fileExists(atPath: managedPython) {
        stderrLine("Creating managed Python environment at \(managedVenvDir.path)")
        try runProcessStreaming(
            executable: bootstrapPython,
            arguments: ["-m", "venv", managedVenvDir.path],
            workingDirectory: defaults.workingDirectory,
            environment: pythonEnvironment(defaults: defaults)
        )
    }

    try runProcessStreaming(
        executable: managedPython,
        arguments: ["-m", "pip", "install", "--upgrade", "pip"],
        workingDirectory: defaults.workingDirectory,
        environment: pythonEnvironment(defaults: defaults)
    )

    let packages = packagesForModules(requiredModules)
    stderrLine("Installing Python packages: \(packages.joined(separator: ", "))")
    try runProcessStreaming(
        executable: managedPython,
        arguments: ["-m", "pip", "install"] + packages,
        workingDirectory: defaults.workingDirectory,
        environment: pythonEnvironment(defaults: defaults)
    )

    guard try pythonSupportsModules(managedPython, requiredModules: requiredModules, defaults: defaults) else {
        throw CLIError.runtime("Managed Python environment is still missing required modules after installation.")
    }
    return managedPython
}

func preferredBootstrapPython(defaults: DemoDefaults) throws -> String {
    if let override = ProcessInfo.processInfo.environment["ESPRESSO_BOOTSTRAP_PYTHON"], !override.isEmpty {
        return override
    }

    for candidate in ["python3.13", "python3.12", "python3"] {
        let output = try runProcessCaptured(
            executable: candidate,
            arguments: ["--version"],
            workingDirectory: defaults.workingDirectory,
            environment: pythonEnvironment(defaults: defaults),
            allowBootstrap: true
        )
        if output.status == 0 {
            return candidate
        }
    }

    throw CLIError.runtime("Unable to find python3.13, python3.12, or python3 for GPT-2 demo setup.")
}

private func pythonVersionTag(_ executable: String, defaults: DemoDefaults) throws -> String {
    let output = try runProcessCaptured(
        executable: executable,
        arguments: ["-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"],
        workingDirectory: defaults.workingDirectory,
        environment: pythonEnvironment(defaults: defaults),
        allowBootstrap: true
    )
    guard output.status == 0 else {
        throw CLIError.runtime("Unable to determine Python version for \(executable)")
    }
    let trimmed = output.stdout.trimmingCharacters(in: .whitespacesAndNewlines)
    return trimmed.replacingOccurrences(of: ".", with: "_")
}

private func packagesForModules(_ modules: [String]) -> [String] {
    var packages: Set<String> = ["numpy"]
    for module in modules {
        switch module {
        case "numpy":
            packages.insert("numpy")
        case "torch":
            packages.insert("torch")
        case "transformers":
            packages.insert("transformers")
        case "coremltools":
            packages.insert("coremltools")
        default:
            packages.insert(module)
        }
    }
    return packages.sorted()
}

private func pythonSupportsModules(
    _ executable: String,
    requiredModules: [String],
    defaults: DemoDefaults
) throws -> Bool {
    let fileManager = FileManager()
    if executable.contains("/"), !fileManager.fileExists(atPath: executable) {
        return false
    }
    let command = "import " + requiredModules.joined(separator: ", ")
    let output = try runProcessCaptured(
        executable: executable,
        arguments: ["-c", command],
        workingDirectory: defaults.workingDirectory,
        environment: pythonEnvironment(defaults: defaults),
        allowBootstrap: true
    )
    return output.status == 0
}

private func pythonEnvironment(defaults: DemoDefaults) -> [String: String] {
    var environment = ProcessInfo.processInfo.environment
    environment["HF_HOME"] = defaults.hfCacheRoot.path
    environment["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    environment["PYTHONUNBUFFERED"] = "1"
    return environment
}

private func runProcessStreaming(
    executable: String,
    arguments: [String],
    workingDirectory: URL,
    environment: [String: String]
) throws {
    let process = configuredProcess(
        executable: executable,
        arguments: arguments,
        workingDirectory: workingDirectory,
        environment: environment
    )
    process.standardOutput = FileHandle.standardError
    process.standardError = FileHandle.standardError
    try process.run()
    process.waitUntilExit()
    guard process.terminationStatus == 0 else {
        throw CLIError.runtime("Process failed (\(process.terminationStatus)): \(executable) \(arguments.joined(separator: " "))")
    }
}

private func runProcessCaptured(
    executable: String,
    arguments: [String],
    workingDirectory: URL,
    environment: [String: String],
    allowBootstrap: Bool
) throws -> ProcessOutput {
    let process = configuredProcess(
        executable: executable,
        arguments: arguments,
        workingDirectory: workingDirectory,
        environment: environment
    )
    let stdoutPipe = Pipe()
    let stderrPipe = Pipe()
    process.standardOutput = stdoutPipe
    process.standardError = stderrPipe
    do {
        try process.run()
    } catch {
        if allowBootstrap {
            throw error
        }
        throw CLIError.runtime("\(error)")
    }
    process.waitUntilExit()
    let stdout = String(decoding: stdoutPipe.fileHandleForReading.readDataToEndOfFile(), as: UTF8.self)
    let stderr = String(decoding: stderrPipe.fileHandleForReading.readDataToEndOfFile(), as: UTF8.self)
    return ProcessOutput(status: process.terminationStatus, stdout: stdout, stderr: stderr)
}

private func runProcessStreamingLinesCapture(
    executable: String,
    arguments: [String],
    workingDirectory: URL,
    environment: [String: String],
    allowBootstrap: Bool,
    onStdoutLine: @escaping (String) -> Void
) throws -> ProcessOutput {
    let process = configuredProcess(
        executable: executable,
        arguments: arguments,
        workingDirectory: workingDirectory,
        environment: environment
    )
    let stdoutPipe = Pipe()
    let stderrPipe = Pipe()
    process.standardOutput = stdoutPipe
    process.standardError = stderrPipe

    let stdoutAccumulator = LockedStringBuffer()
    let stderrAccumulator = LockedStringBuffer()
    let stdoutParser = LineStreamParser { line in
        stdoutAccumulator.append(line + "\n")
        onStdoutLine(line)
    }
    let stderrParser = LineStreamParser { line in
        stderrAccumulator.append(line + "\n")
        stderrLine(line)
    }

    stdoutPipe.fileHandleForReading.readabilityHandler = { handle in
        let data = handle.availableData
        if data.isEmpty {
            return
        }
        stdoutParser.append(data)
    }
    stderrPipe.fileHandleForReading.readabilityHandler = { handle in
        let data = handle.availableData
        if data.isEmpty {
            return
        }
        stderrParser.append(data)
    }

    do {
        try process.run()
    } catch {
        stdoutPipe.fileHandleForReading.readabilityHandler = nil
        stderrPipe.fileHandleForReading.readabilityHandler = nil
        if allowBootstrap {
            throw error
        }
        throw CLIError.runtime("\(error)")
    }

    process.waitUntilExit()
    stdoutPipe.fileHandleForReading.readabilityHandler = nil
    stderrPipe.fileHandleForReading.readabilityHandler = nil
    stdoutParser.finish()
    stderrParser.finish()

    return ProcessOutput(
        status: process.terminationStatus,
        stdout: stdoutAccumulator.value,
        stderr: stderrAccumulator.value
    )
}

private func configuredProcess(
    executable: String,
    arguments: [String],
    workingDirectory: URL,
    environment: [String: String]
) -> Process {
    let process = Process()
    if executable.contains("/") {
        process.executableURL = URL(fileURLWithPath: executable)
        process.arguments = arguments
    } else {
        process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        process.arguments = [executable] + arguments
    }
    process.currentDirectoryURL = workingDirectory
    process.environment = environment
    return process
}

private final class LockedStringBuffer: @unchecked Sendable {
    private let lock = NSLock()
    private var storage = ""

    func append(_ text: String) {
        lock.lock()
        storage.append(text)
        lock.unlock()
    }

    var value: String {
        lock.lock()
        let snapshot = storage
        lock.unlock()
        return snapshot
    }
}

private final class LineStreamParser: @unchecked Sendable {
    private let lock = NSLock()
    private let onLine: (String) -> Void
    private var buffer = Data()

    init(onLine: @escaping (String) -> Void) {
        self.onLine = onLine
    }

    func append(_ data: Data) {
        lock.lock()
        buffer.append(data)
        emitLockedLines()
        lock.unlock()
    }

    func finish() {
        lock.lock()
        if !buffer.isEmpty {
            let line = String(decoding: buffer, as: UTF8.self).trimmingCharacters(in: .newlines)
            if !line.isEmpty {
                onLine(line)
            }
            buffer.removeAll(keepingCapacity: false)
        }
        lock.unlock()
    }

    private func emitLockedLines() {
        while let newlineIndex = buffer.firstIndex(of: 0x0A) {
            let lineData = buffer[..<newlineIndex]
            buffer.removeSubrange(...newlineIndex)
            let line = String(decoding: lineData, as: UTF8.self).trimmingCharacters(in: .newlines)
            onLine(line)
        }
    }
}
