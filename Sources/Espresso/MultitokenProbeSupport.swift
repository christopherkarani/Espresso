import Accelerate
import CoreML
import Foundation
import CPUOps
import ANETypes

public struct GenerationBenchmarkSample: Sendable, Equatable {
    public let medianTokenMs: Double
    public let medianTokensPerSecond: Double
    public let compileTimeMs: Double
    public let medianTrunkMsPerToken: Double
    public let medianLogitsMsPerToken: Double

    public init(
        medianTokenMs: Double,
        medianTokensPerSecond: Double,
        compileTimeMs: Double,
        medianTrunkMsPerToken: Double,
        medianLogitsMsPerToken: Double
    ) {
        self.medianTokenMs = medianTokenMs
        self.medianTokensPerSecond = medianTokensPerSecond
        self.compileTimeMs = compileTimeMs
        self.medianTrunkMsPerToken = medianTrunkMsPerToken
        self.medianLogitsMsPerToken = medianLogitsMsPerToken
    }
}

public struct ExactTwoTokenBenchmarkSample: Sendable, Equatable {
    public let medianTokenMs: Double
    public let medianTokensPerSecond: Double
    public let compileTimeMs: Double
    public let medianCommittedExactTokensPerPass: Double
    public let medianAcceptedFutureTokensPerPass: Double
    public let medianProposerMsPerPass: Double
    public let medianVerifierTrunkMsPerPass: Double
    public let medianVerifierLogitsMsPerPass: Double
    public let medianStateAdvanceMsPerPass: Double

    public init(
        medianTokenMs: Double,
        medianTokensPerSecond: Double,
        compileTimeMs: Double,
        medianCommittedExactTokensPerPass: Double,
        medianAcceptedFutureTokensPerPass: Double,
        medianProposerMsPerPass: Double,
        medianVerifierTrunkMsPerPass: Double,
        medianVerifierLogitsMsPerPass: Double,
        medianStateAdvanceMsPerPass: Double
    ) {
        self.medianTokenMs = medianTokenMs
        self.medianTokensPerSecond = medianTokensPerSecond
        self.compileTimeMs = compileTimeMs
        self.medianCommittedExactTokensPerPass = medianCommittedExactTokensPerPass
        self.medianAcceptedFutureTokensPerPass = medianAcceptedFutureTokensPerPass
        self.medianProposerMsPerPass = medianProposerMsPerPass
        self.medianVerifierTrunkMsPerPass = medianVerifierTrunkMsPerPass
        self.medianVerifierLogitsMsPerPass = medianVerifierLogitsMsPerPass
        self.medianStateAdvanceMsPerPass = medianStateAdvanceMsPerPass
    }
}

public struct CompileInitBenchmarkSample: Sendable, Equatable {
    public let wallInitMs: Double
    public let reportedCompileTimeMs: Double

    public init(wallInitMs: Double, reportedCompileTimeMs: Double) {
        self.wallInitMs = wallInitMs
        self.reportedCompileTimeMs = reportedCompileTimeMs
    }
}

public enum MultitokenProbeInput: Sendable, Equatable {
    case echo
    case recurrentCheckpoint(path: String)
}

public enum CoreMLHeadWeightsSource: Sendable, Equatable {
    case echo
    case generationModel(path: String)
}

public struct CoreMLBenchmarkRequest: Equatable {
    public let modelPath: String
    public let headWeightsSource: CoreMLHeadWeightsSource
    public let computeUnits: MLComputeUnits

    public init(
        modelPath: String,
        headWeightsSource: CoreMLHeadWeightsSource,
        computeUnits: MLComputeUnits = .cpuAndNeuralEngine
    ) {
        self.modelPath = modelPath
        self.headWeightsSource = headWeightsSource
        self.computeUnits = computeUnits
    }
}

public enum MultitokenProbeConfigurationError: Error, Equatable {
    case missingInput
    case missingRecurrentCheckpointPath
    case missingCoreMLModelPath
    case missingGenerationModelPathForCoreML
}

public struct ValidatedMultitokenProbeConfiguration: Equatable {
    public let input: MultitokenProbeInput
    public let coreMLRequest: CoreMLBenchmarkRequest?

    public init(input: MultitokenProbeInput, coreMLRequest: CoreMLBenchmarkRequest?) {
        self.input = input
        self.coreMLRequest = coreMLRequest
    }
}

public struct MultitokenProbeConfiguration: Equatable {
    public let input: MultitokenProbeInput?
    public let compareCoreML: Bool
    public let coreMLModelPath: String?
    public let generationModelPath: String?

    public init(
        input: MultitokenProbeInput?,
        compareCoreML: Bool,
        coreMLModelPath: String?,
        generationModelPath: String?
    ) {
        self.input = input
        self.compareCoreML = compareCoreML
        self.coreMLModelPath = coreMLModelPath
        self.generationModelPath = generationModelPath
    }

    public func validated() throws(MultitokenProbeConfigurationError) -> ValidatedMultitokenProbeConfiguration {
        guard let input else {
            throw .missingInput
        }
        if case let .recurrentCheckpoint(path) = input, path.isEmpty {
            throw .missingRecurrentCheckpointPath
        }
        guard compareCoreML else {
            return ValidatedMultitokenProbeConfiguration(input: input, coreMLRequest: nil)
        }

        guard let coreMLModelPath, !coreMLModelPath.isEmpty else {
            throw .missingCoreMLModelPath
        }

        let headWeightsSource: CoreMLHeadWeightsSource
        switch input {
        case .echo:
            if let generationModelPath, !generationModelPath.isEmpty {
                headWeightsSource = .generationModel(path: generationModelPath)
            } else {
                headWeightsSource = .echo
            }
        case .recurrentCheckpoint:
            guard let generationModelPath, !generationModelPath.isEmpty else {
                throw .missingGenerationModelPathForCoreML
            }
            headWeightsSource = .generationModel(path: generationModelPath)
        }

        return ValidatedMultitokenProbeConfiguration(
            input: input,
            coreMLRequest: CoreMLBenchmarkRequest(
                modelPath: coreMLModelPath,
                headWeightsSource: headWeightsSource
            )
        )
    }
}

public func makeEchoGenerationWeights(layerCount: Int) -> GenerationWeights {
    let layers = LayerStorage<LayerWeights>(count: layerCount) { _ in
        let weights = LayerWeights()
        fillBuffer(weights.Wq, value: 0)
        fillBuffer(weights.Wk, value: 0)
        fillBuffer(weights.Wv, value: 0)
        fillBuffer(weights.Wo, value: 0)
        fillBuffer(weights.W1, value: 0)
        fillBuffer(weights.W2, value: 0)
        fillBuffer(weights.W3, value: 0)
        fillBuffer(weights.rmsAtt, value: 1)
        fillBuffer(weights.rmsFfn, value: 1)
        return weights
    }

    let rmsFinal = TensorBuffer(count: ModelConfig.dim, zeroed: false)
    fillBuffer(rmsFinal, value: 1)

    let embedding = TensorBuffer(count: ModelConfig.vocab * ModelConfig.dim, zeroed: true)
    embedding.withUnsafeMutablePointer { ptr in
        for dimIdx in 0..<ModelConfig.dim {
            ptr[dimIdx] = 1
        }
    }

    return GenerationWeights(
        layers: layers,
        rmsFinal: rmsFinal,
        embedding: embedding,
        classifier: TensorBuffer(count: 0, zeroed: true),
        sharedClassifier: true
    )
}

public func makeEchoRecurrentGenerationWeights(layerCount: Int) -> RecurrentGenerationWeights {
    let layers = LayerStorage<RWKVStyleRecurrentWeights>(count: layerCount) { _ in
        let weights = RWKVStyleRecurrentWeights()
        fillBuffer(weights.rms, value: 1)
        fillBuffer(weights.Wx, value: 0)
        fillBuffer(weights.Ws, value: 0)
        fillBuffer(weights.Wd, value: 0)
        fillBuffer(weights.Wo, value: 0)
        return weights
    }

    let rmsFinal = TensorBuffer(count: ModelConfig.dim, zeroed: false)
    fillBuffer(rmsFinal, value: 1)

    let embedding = TensorBuffer(count: ModelConfig.vocab * ModelConfig.dim, zeroed: true)
    embedding.withUnsafeMutablePointer { ptr in
        for dimIdx in 0..<ModelConfig.dim {
            ptr[dimIdx] = 1
        }
    }

    return RecurrentGenerationWeights(
        layers: layers,
        rmsFinal: rmsFinal,
        embedding: embedding,
        classifier: TensorBuffer(count: 0, zeroed: true),
        sharedClassifier: true
    )
}

public func loadRecurrentGenerationWeights(
    input: MultitokenProbeInput,
    layerCount: Int
) throws(GenerationError) -> RecurrentGenerationWeights {
    switch input {
    case .echo:
        return makeEchoRecurrentGenerationWeights(layerCount: layerCount)
    case let .recurrentCheckpoint(path):
        do {
            return try RecurrentGenerationWeightStore.load(from: path)
        } catch {
            throw .modelLoadFailed("\(error)")
        }
    }
}

public func loadGenerationWeightsForCoreML(
    source: CoreMLHeadWeightsSource
) throws(GenerationError) -> GenerationWeights {
    switch source {
    case .echo:
        return makeEchoGenerationWeights(layerCount: 1)
    case let .generationModel(path):
        do {
            return try GenerationModelWeightStore.load(path: path)
        } catch {
            throw .modelLoadFailed("\(error)")
        }
    }
}

public func benchmarkCoreMLGeneration(
    request: borrowing CoreMLBenchmarkRequest,
    promptTokens: [UInt16],
    maxNewTokens: Int,
    warmup: Int,
    iterations: Int,
    maxSequenceTokens: Int = ModelConfig.seqLen
) throws(GenerationError) -> GenerationBenchmarkSample {
    let headWeights = try loadGenerationWeightsForCoreML(source: request.headWeightsSource)
    let model = try CoreMLGenerationBenchmarkModel(
        request: request,
        headWeights: headWeights,
        maxSequenceTokens: maxSequenceTokens
    )
    var harness = AutoregressiveGenerationHarness(model: model, strategy: .argmax)
    return try benchmarkAutoregressiveHarness(
        harness: &harness,
        promptTokens: promptTokens,
        maxNewTokens: maxNewTokens,
        warmup: warmup,
        iterations: iterations
    )
}

private func benchmarkAutoregressiveHarness<Model>(
    harness: inout AutoregressiveGenerationHarness<Model>,
    promptTokens: [UInt16],
    maxNewTokens: Int,
    warmup: Int,
    iterations: Int
) throws(GenerationError) -> GenerationBenchmarkSample
where Model: AutoregressiveLanguageModel & GenerationPerformanceTrackable, Model: ~Copyable {
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
        medianTokenMs: GenerationMetrics.median(tokenLatencies),
        medianTokensPerSecond: GenerationMetrics.median(throughput),
        compileTimeMs: compileTimeMs,
        medianTrunkMsPerToken: GenerationMetrics.median(trunkLatencies),
        medianLogitsMsPerToken: GenerationMetrics.median(logitsLatencies)
    )
}

private struct CoreMLGenerationBenchmarkModel: ~Copyable, AutoregressiveLanguageModel, GenerationPerformanceTrackable {
    let vocabSize: Int
    let maxSequenceTokens: Int

    private let model: MLModel
    private let inputFeatureName: String
    private let outputFeatureName: String
    private let inputArray: MLMultiArray
    private let rmsFinal: TensorBuffer
    private let embedding: TensorBuffer
    private let classifier: TensorBuffer
    private let sharedClassifier: Bool
    private let hiddenStates: TensorBuffer
    private let stepHidden: TensorBuffer
    private let stepNorm: TensorBuffer
    private let stepLogits: TensorBuffer
    private let verifyNorm: TensorBuffer
    private let verifyLogits: TensorBuffer
    private let stepRMSWorkspace: RMSNorm.Workspace
    private let verifyRMSWorkspace: RMSNorm.Workspace
    private var currentTokens: [UInt16]
    private(set) var compileTimeMs: Double
    private var trunkLatencyMs: Double
    private var logitsLatencyMs: Double

    var performanceSnapshot: GenerationPerformanceSnapshot {
        GenerationPerformanceSnapshot(
            compileTimeMs: compileTimeMs,
            trunkLatencyMs: trunkLatencyMs,
            logitsLatencyMs: logitsLatencyMs
        )
    }

    init(
        request: borrowing CoreMLBenchmarkRequest,
        headWeights: borrowing GenerationWeights,
        maxSequenceTokens: Int = ModelConfig.seqLen
    ) throws(GenerationError) {
        guard maxSequenceTokens > 0, maxSequenceTokens <= ModelConfig.seqLen else {
            throw .invalidArguments("maxSequenceTokens must be in 1...\(ModelConfig.seqLen)")
        }
        guard FileManager.default.fileExists(atPath: request.modelPath) else {
            throw .modelLoadFailed("CoreML model not found at \(request.modelPath)")
        }

        let compileStart = GenerationClock.now()
        let modelURL = URL(fileURLWithPath: request.modelPath)
        let compiledURL: URL
        do {
            compiledURL = try MLModel.compileModel(at: modelURL)
        } catch {
            throw .modelLoadFailed("CoreML compile failed: \(error)")
        }

        let configuration = MLModelConfiguration()
        configuration.computeUnits = request.computeUnits

        let model: MLModel
        do {
            model = try MLModel(contentsOf: compiledURL, configuration: configuration)
        } catch {
            throw .modelLoadFailed("CoreML load failed: \(error)")
        }

        let inputNames = Array(model.modelDescription.inputDescriptionsByName.keys)
        let outputNames = Array(model.modelDescription.outputDescriptionsByName.keys)
        guard inputNames.count == 1, let inputFeatureName = inputNames.first else {
            throw .modelLoadFailed("expected exactly one CoreML input, found \(inputNames.count)")
        }
        guard outputNames.count == 1, let outputFeatureName = outputNames.first else {
            throw .modelLoadFailed("expected exactly one CoreML output, found \(outputNames.count)")
        }

        let inputArray: MLMultiArray
        do {
            inputArray = try MLMultiArray(
                shape: [1, NSNumber(value: ModelConfig.dim), 1, NSNumber(value: ModelConfig.seqLen)],
                dataType: .float16
            )
        } catch {
            throw .modelLoadFailed("failed to allocate CoreML input array: \(error)")
        }

        self.vocabSize = headWeights.vocabSize
        self.maxSequenceTokens = maxSequenceTokens
        self.model = model
        self.inputFeatureName = inputFeatureName
        self.outputFeatureName = outputFeatureName
        self.inputArray = inputArray
        self.rmsFinal = GenerationWeightCloner.cloneTensor(headWeights.rmsFinal)
        self.embedding = GenerationWeightCloner.cloneTensor(headWeights.embedding)
        self.classifier = headWeights.sharedClassifier
            ? TensorBuffer(count: 0, zeroed: true)
            : GenerationWeightCloner.cloneTensor(headWeights.classifier)
        self.sharedClassifier = headWeights.sharedClassifier
        self.hiddenStates = TensorBuffer(count: ModelConfig.dim * ModelConfig.seqLen, zeroed: true)
        self.stepHidden = TensorBuffer(count: ModelConfig.dim, zeroed: true)
        self.stepNorm = TensorBuffer(count: ModelConfig.dim, zeroed: true)
        self.stepLogits = TensorBuffer(count: headWeights.vocabSize, zeroed: true)
        self.verifyNorm = TensorBuffer(count: ModelConfig.dim * ModelConfig.seqLen, zeroed: true)
        self.verifyLogits = TensorBuffer(count: headWeights.vocabSize * ModelConfig.seqLen, zeroed: true)
        self.stepRMSWorkspace = RMSNorm.Workspace(seqLen: 1)
        self.verifyRMSWorkspace = RMSNorm.Workspace(seqLen: ModelConfig.seqLen)
        self.currentTokens = []
        self.compileTimeMs = GenerationClock.milliseconds(start: compileStart, end: GenerationClock.now())
        self.trunkLatencyMs = 0
        self.logitsLatencyMs = 0

        zeroInputArray()
    }

    mutating func reset() throws(GenerationError) {
        currentTokens.removeAll(keepingCapacity: true)
        hiddenStates.zero()
        stepHidden.zero()
        stepNorm.zero()
        stepLogits.zero()
        verifyNorm.zero()
        verifyLogits.zero()
        trunkLatencyMs = 0
        logitsLatencyMs = 0
        zeroInputArray()
    }

    mutating func prefill(promptTokens: [UInt16]) throws(GenerationError) -> [Float] {
        guard !promptTokens.isEmpty else {
            throw .invalidArguments("promptTokens must not be empty")
        }
        guard promptTokens.count <= maxSequenceTokens else {
            throw .invalidArguments("prompt length \(promptTokens.count) exceeds maxSequenceTokens \(maxSequenceTokens)")
        }

        currentTokens = promptTokens
        for (position, token) in promptTokens.enumerated() {
            try writeToken(token, at: position)
        }
        try runPrediction(sequenceLength: promptTokens.count)
        return try projectCurrentLogits(sequenceIndex: promptTokens.count - 1)
    }

    mutating func decode(nextToken: UInt16) throws(GenerationError) -> [Float] {
        guard currentTokens.count < maxSequenceTokens else {
            throw .invalidArguments("decode overflow at maxSequenceTokens \(maxSequenceTokens)")
        }

        currentTokens.append(nextToken)
        try writeToken(nextToken, at: currentTokens.count - 1)
        try runPrediction(sequenceLength: currentTokens.count)
        return try projectCurrentLogits(sequenceIndex: currentTokens.count - 1)
    }

    mutating func verify(
        sequenceTokens: [UInt16],
        startIndex: Int
    ) throws(GenerationError) -> [[Float]] {
        guard !sequenceTokens.isEmpty else {
            throw .invalidArguments("sequenceTokens must not be empty")
        }
        guard sequenceTokens.count <= maxSequenceTokens else {
            throw .invalidArguments("sequence length \(sequenceTokens.count) exceeds maxSequenceTokens \(maxSequenceTokens)")
        }
        guard startIndex >= 0, startIndex < sequenceTokens.count else {
            throw .invalidArguments("startIndex \(startIndex) must be within sequence length \(sequenceTokens.count)")
        }

        let savedTokens = currentTokens
        defer {
            currentTokens = savedTokens
        }

        zeroInputArray()
        for (position, token) in sequenceTokens.enumerated() {
            try writeToken(token, at: position)
        }
        try runPrediction(sequenceLength: sequenceTokens.count)
        try projectSequenceLogits()

        var outputs: [[Float]] = []
        outputs.reserveCapacity(sequenceTokens.count - startIndex)
        verifyLogits.withUnsafeBufferPointer { logitsPtr in
            for seqPos in startIndex..<sequenceTokens.count {
                var column = [Float](repeating: 0, count: vocabSize)
                for vocabIdx in 0..<vocabSize {
                    column[vocabIdx] = logitsPtr[vocabIdx * ModelConfig.seqLen + seqPos]
                }
                outputs.append(column)
            }
        }

        if savedTokens.isEmpty {
            hiddenStates.zero()
            zeroInputArray()
        } else {
            zeroInputArray()
            for (position, token) in savedTokens.enumerated() {
                try writeToken(token, at: position)
            }
            try runPrediction(sequenceLength: savedTokens.count)
        }

        return outputs
    }

    @inline(__always)
    private func classifierPointer<R>(_ body: (UnsafePointer<Float>) throws -> R) rethrows -> R {
        if sharedClassifier {
            return try embedding.withUnsafePointer(body)
        }
        return try classifier.withUnsafePointer(body)
    }

    private func zeroInputArray() {
        let ptr = inputArray.dataPointer.bindMemory(to: Float16.self, capacity: inputArray.count)
        for idx in 0..<inputArray.count {
            ptr[idx] = 0
        }
    }

    private func writeToken(_ token: UInt16, at position: Int) throws(GenerationError) {
        guard Int(token) < vocabSize else {
            throw .invalidArguments("token \(token) exceeds vocab size \(vocabSize)")
        }
        guard position >= 0, position < maxSequenceTokens else {
            throw .invalidArguments("position \(position) exceeds maxSequenceTokens \(maxSequenceTokens)")
        }

        let ptr = inputArray.dataPointer.bindMemory(to: Float16.self, capacity: inputArray.count)
        let channelStride = inputArray.strides[1].intValue
        let spatialStride = inputArray.strides[3].intValue

        embedding.withUnsafePointer { embeddingPtr in
            let base = Int(token) * ModelConfig.dim
            for dimIdx in 0..<ModelConfig.dim {
                ptr[dimIdx * channelStride + position * spatialStride] = Float16(embeddingPtr[base + dimIdx])
            }
        }
    }

    private mutating func runPrediction(sequenceLength: Int) throws(GenerationError) {
        let trunkStart = GenerationClock.now()
        let provider: MLDictionaryFeatureProvider
        do {
            provider = try MLDictionaryFeatureProvider(
                dictionary: [inputFeatureName: MLFeatureValue(multiArray: inputArray)]
            )
        } catch {
            throw .runtimeFailure("CoreML feature provider creation failed: \(error)")
        }

        let prediction: MLFeatureProvider
        do {
            prediction = try model.prediction(from: provider)
        } catch {
            throw .runtimeFailure("CoreML prediction failed: \(error)")
        }

        guard let outputArray = prediction.featureValue(for: outputFeatureName)?.multiArrayValue else {
            throw .runtimeFailure("CoreML output '\(outputFeatureName)' missing MLMultiArray value")
        }

        do {
            try copyHiddenStates(from: outputArray, sequenceLength: sequenceLength)
        } catch {
            throw .runtimeFailure("CoreML output copy failed: \(error)")
        }

        trunkLatencyMs += GenerationClock.milliseconds(start: trunkStart, end: GenerationClock.now())
    }

    private func copyHiddenStates(
        from outputArray: MLMultiArray,
        sequenceLength: Int
    ) throws(GenerationError) {
        guard sequenceLength > 0, sequenceLength <= maxSequenceTokens else {
            throw .invalidArguments("sequenceLength \(sequenceLength) exceeds maxSequenceTokens \(maxSequenceTokens)")
        }

        let channelStride = outputArray.strides[1].intValue
        let spatialStride = outputArray.strides[3].intValue

        hiddenStates.withUnsafeMutablePointer { dst in
            switch outputArray.dataType {
            case .float16:
                let src = outputArray.dataPointer.bindMemory(to: Float16.self, capacity: outputArray.count)
                for dimIdx in 0..<ModelConfig.dim {
                    let dstBase = dimIdx * ModelConfig.seqLen
                    let srcBase = dimIdx * channelStride
                    for seqIdx in 0..<sequenceLength {
                        dst[dstBase + seqIdx] = Float(src[srcBase + seqIdx * spatialStride])
                    }
                }
            case .float32:
                let src = outputArray.dataPointer.bindMemory(to: Float.self, capacity: outputArray.count)
                for dimIdx in 0..<ModelConfig.dim {
                    let dstBase = dimIdx * ModelConfig.seqLen
                    let srcBase = dimIdx * channelStride
                    for seqIdx in 0..<sequenceLength {
                        dst[dstBase + seqIdx] = src[srcBase + seqIdx * spatialStride]
                    }
                }
            default:
                break
            }
        }

        guard outputArray.dataType == .float16 || outputArray.dataType == .float32 else {
            throw .runtimeFailure("unsupported CoreML output dtype \(outputArray.dataType.rawValue)")
        }
    }

    private mutating func projectCurrentLogits(sequenceIndex: Int) throws(GenerationError) -> [Float] {
        let logitsStart = GenerationClock.now()
        stepHidden.withUnsafeMutablePointer { hiddenPtr in
            hiddenStates.withUnsafePointer { src in
                for dimIdx in 0..<ModelConfig.dim {
                    hiddenPtr[dimIdx] = src[dimIdx * ModelConfig.seqLen + sequenceIndex]
                }
            }
        }

        stepHidden.withUnsafePointer { hiddenPtr in
            stepNorm.withUnsafeMutablePointer { normPtr in
                rmsFinal.withUnsafePointer { rmsPtr in
                    RMSNorm.forward(
                        output: normPtr,
                        input: hiddenPtr,
                        weights: rmsPtr,
                        dim: ModelConfig.dim,
                        seqLen: 1,
                        workspace: stepRMSWorkspace
                    )
                }
            }
        }

        stepLogits.zero()
        stepLogits.withUnsafeMutablePointer { logitsPtr in
            classifierPointer { clsPtr in
                stepNorm.withUnsafePointer { normPtr in
                    BLAS.sgemm(
                        CblasRowMajor,
                        CblasNoTrans,
                        CblasNoTrans,
                        m: Int32(vocabSize),
                        n: 1,
                        k: Int32(ModelConfig.dim),
                        alpha: 1.0,
                        a: clsPtr,
                        lda: Int32(ModelConfig.dim),
                        b: normPtr,
                        ldb: 1,
                        beta: 0.0,
                        c: logitsPtr,
                        ldc: 1
                    )
                }
            }
        }

        logitsLatencyMs += GenerationClock.milliseconds(start: logitsStart, end: GenerationClock.now())
        return stepLogits.withUnsafeBufferPointer { Array($0) }
    }

    private mutating func projectSequenceLogits() throws(GenerationError) {
        let logitsStart = GenerationClock.now()

        hiddenStates.withUnsafePointer { hiddenPtr in
            verifyNorm.withUnsafeMutablePointer { normPtr in
                rmsFinal.withUnsafePointer { rmsPtr in
                    RMSNorm.forward(
                        output: normPtr,
                        input: hiddenPtr,
                        weights: rmsPtr,
                        dim: ModelConfig.dim,
                        seqLen: currentTokens.count,
                        workspace: verifyRMSWorkspace
                    )
                }
            }
        }

        verifyLogits.zero()
        verifyLogits.withUnsafeMutablePointer { logitsPtr in
            classifierPointer { clsPtr in
                verifyNorm.withUnsafePointer { normPtr in
                    BLAS.sgemm(
                        CblasRowMajor,
                        CblasNoTrans,
                        CblasNoTrans,
                        m: Int32(vocabSize),
                        n: Int32(currentTokens.count),
                        k: Int32(ModelConfig.dim),
                        alpha: 1.0,
                        a: clsPtr,
                        lda: Int32(ModelConfig.dim),
                        b: normPtr,
                        ldb: Int32(currentTokens.count),
                        beta: 0.0,
                        c: logitsPtr,
                        ldc: Int32(currentTokens.count)
                    )
                }
            }
        }

        logitsLatencyMs += GenerationClock.milliseconds(start: logitsStart, end: GenerationClock.now())
    }
}

@inline(__always)
private func fillBuffer(_ buffer: borrowing TensorBuffer, value: Float) {
    buffer.withUnsafeMutableBufferPointer { ptr in
        for idx in ptr.indices {
            ptr[idx] = value
        }
    }
}
