import Accelerate
import CPUOps
import Foundation
import ANERuntime
import ANETypes

public enum GenerationError: Error, Sendable, Equatable {
    case invalidArguments(String)
    case modelLoadFailed(String)
    case runtimeFailure(String)
}

public enum TokenSelectionStrategy: Sendable {
    case argmax
}

public protocol AutoregressiveLanguageModel: ~Copyable {
    var vocabSize: Int { get }

    mutating func reset() throws(GenerationError)
    mutating func prefill(promptTokens: [UInt16]) throws(GenerationError) -> [Float]
    mutating func decode(nextToken: UInt16) throws(GenerationError) -> [Float]
    mutating func verify(sequenceTokens: [UInt16], startIndex: Int) throws(GenerationError) -> [[Float]]
}

public struct AutoregressiveGenerationTrace: Sendable {
    public let promptTokens: [UInt16]
    public let generatedTokens: [UInt16]
    public let prefillLatencyMs: Double
    public let decodeLatenciesMs: [Double]

    public var totalLatencyMs: Double {
        prefillLatencyMs + decodeLatenciesMs.reduce(0, +)
    }

    public var medianLatencyMs: Double {
        GenerationMetrics.median(decodeLatenciesMs)
    }

    public var tokensPerSecond: Double {
        guard totalLatencyMs > 0 else { return 0 }
        return Double(generatedTokens.count) * 1000.0 / totalLatencyMs
    }
}

public struct SpeculativeGenerationTrace: Sendable {
    public let promptTokens: [UInt16]
    public let generatedTokens: [UInt16]
    public let candidateCount: Int
    public let draftPrefillLatencyMs: Double
    public let fullPrefillLatencyMs: Double
    public let draftLatenciesMs: [Double]
    public let verificationLatenciesMs: [Double]
    public let fullAdvanceLatenciesMs: [Double]
    public let draftResyncLatenciesMs: [Double]
    public let acceptedPrefixLengths: [Int]
    public let totalDraftCandidates: Int
    public let totalAcceptedCandidates: Int

    public var acceptanceRate: Double {
        guard totalDraftCandidates > 0 else { return 0 }
        return Double(totalAcceptedCandidates) / Double(totalDraftCandidates)
    }

    public var totalLatencyMs: Double {
        draftPrefillLatencyMs
            + fullPrefillLatencyMs
            + draftLatenciesMs.reduce(0, +)
            + verificationLatenciesMs.reduce(0, +)
            + fullAdvanceLatenciesMs.reduce(0, +)
            + draftResyncLatenciesMs.reduce(0, +)
    }

    public var effectiveTokensPerSecond: Double {
        guard totalLatencyMs > 0 else { return 0 }
        return Double(generatedTokens.count) * 1000.0 / totalLatencyMs
    }
}

enum GenerationMetrics {
    @inline(__always)
    static func median(_ values: [Double]) -> Double {
        guard !values.isEmpty else { return 0 }
        let sorted = values.sorted()
        let mid = sorted.count / 2
        if sorted.count.isMultiple(of: 2) {
            return (sorted[mid - 1] + sorted[mid]) * 0.5
        }
        return sorted[mid]
    }
}

enum GenerationWeightCloner {
    @inline(__always)
    static func cloneTensor(_ source: borrowing TensorBuffer) -> TensorBuffer {
        let copy = TensorBuffer(count: source.count, zeroed: false)
        source.withUnsafePointer { src in
            copy.withUnsafeMutablePointer { dst in
                dst.update(from: src, count: source.count)
            }
        }
        return copy
    }

    @inline(__always)
    private static func copyTensor(_ source: borrowing TensorBuffer, into destination: borrowing TensorBuffer) {
        precondition(source.count == destination.count)
        source.withUnsafePointer { src in
            destination.withUnsafeMutablePointer { dst in
                dst.update(from: src, count: source.count)
            }
        }
    }

    static func cloneLayer(_ source: borrowing LayerWeights) -> LayerWeights {
        let layer = LayerWeights()
        copyTensor(source.Wq, into: layer.Wq)
        copyTensor(source.Wk, into: layer.Wk)
        copyTensor(source.Wv, into: layer.Wv)
        copyTensor(source.Wo, into: layer.Wo)
        copyTensor(source.W1, into: layer.W1)
        copyTensor(source.W2, into: layer.W2)
        copyTensor(source.W3, into: layer.W3)
        copyTensor(source.rmsAtt, into: layer.rmsAtt)
        copyTensor(source.rmsFfn, into: layer.rmsFfn)
        return layer
    }

    static func cloneLayers(_ source: borrowing LayerStorage<LayerWeights>) -> LayerStorage<LayerWeights> {
        LayerStorage<LayerWeights>(count: source.count) { idx in
            cloneLayer(source[idx])
        }
    }
}

enum GenerationClock {
    private static let timebase: mach_timebase_info_data_t = {
        var tb = mach_timebase_info_data_t()
        mach_timebase_info(&tb)
        return tb
    }()

    @inline(__always)
    static func now() -> UInt64 {
        mach_absolute_time()
    }

    @inline(__always)
    static func milliseconds(start: UInt64, end: UInt64) -> Double {
        let nanos = (Double(end &- start) * Double(timebase.numer)) / Double(timebase.denom)
        return nanos / 1_000_000.0
    }
}

@inline(__always)
private func selectToken(from logits: [Float], strategy: TokenSelectionStrategy) throws(GenerationError) -> UInt16 {
    guard !logits.isEmpty else {
        throw .invalidArguments("cannot select from empty logits")
    }
    switch strategy {
    case .argmax:
        var bestIndex = 0
        var bestValue = logits[0]
        for idx in 1..<logits.count where logits[idx] > bestValue {
            bestValue = logits[idx]
            bestIndex = idx
        }
        guard let token = UInt16(exactly: bestIndex) else {
            throw .invalidArguments("selected token index \(bestIndex) exceeds UInt16 range")
        }
        return token
    }
}

public struct AutoregressiveGenerationHarness<Model: AutoregressiveLanguageModel>: ~Copyable where Model: ~Copyable {
    public var model: Model
    public let strategy: TokenSelectionStrategy

    public init(model: consuming Model, strategy: TokenSelectionStrategy = .argmax) {
        self.model = model
        self.strategy = strategy
    }

    public mutating func generate(
        promptTokens: [UInt16],
        maxNewTokens: Int
    ) throws(GenerationError) -> AutoregressiveGenerationTrace {
        guard !promptTokens.isEmpty else {
            throw .invalidArguments("promptTokens must not be empty")
        }
        guard maxNewTokens > 0 else {
            throw .invalidArguments("maxNewTokens must be > 0")
        }

        try model.reset()
        let prefillStart = GenerationClock.now()
        var logits = try model.prefill(promptTokens: promptTokens)
        let prefillLatencyMs = GenerationClock.milliseconds(start: prefillStart, end: GenerationClock.now())

        var generatedTokens: [UInt16] = []
        var decodeLatenciesMs: [Double] = []
        generatedTokens.reserveCapacity(maxNewTokens)
        decodeLatenciesMs.reserveCapacity(maxNewTokens)

        for _ in 0..<maxNewTokens {
            let token = try selectToken(from: logits, strategy: strategy)
            generatedTokens.append(token)

            let decodeStart = GenerationClock.now()
            logits = try model.decode(nextToken: token)
            decodeLatenciesMs.append(
                GenerationClock.milliseconds(start: decodeStart, end: GenerationClock.now())
            )
        }

        return AutoregressiveGenerationTrace(
            promptTokens: promptTokens,
            generatedTokens: generatedTokens,
            prefillLatencyMs: prefillLatencyMs,
            decodeLatenciesMs: decodeLatenciesMs
        )
    }
}

public struct SpeculativeGenerationHarness<
    DraftModel: AutoregressiveLanguageModel,
    FullModel: AutoregressiveLanguageModel
>: ~Copyable where DraftModel: ~Copyable, FullModel: ~Copyable {
    public var draftModel: DraftModel
    public var fullModel: FullModel
    public let strategy: TokenSelectionStrategy
    public let candidateCount: Int

    public init(
        draftModel: consuming DraftModel,
        fullModel: consuming FullModel,
        strategy: TokenSelectionStrategy = .argmax,
        candidateCount: Int
    ) {
        self.draftModel = draftModel
        self.fullModel = fullModel
        self.strategy = strategy
        self.candidateCount = candidateCount
    }

    public mutating func generate(
        promptTokens: [UInt16],
        maxNewTokens: Int
    ) throws(GenerationError) -> SpeculativeGenerationTrace {
        guard !promptTokens.isEmpty else {
            throw .invalidArguments("promptTokens must not be empty")
        }
        guard maxNewTokens > 0 else {
            throw .invalidArguments("maxNewTokens must be > 0")
        }
        guard candidateCount > 0 else {
            throw .invalidArguments("candidateCount must be > 0")
        }
        guard draftModel.vocabSize == fullModel.vocabSize else {
            throw .invalidArguments("draft/full vocab sizes must match")
        }

        try draftModel.reset()
        let draftPrefillStart = GenerationClock.now()
        var draftLogits = try draftModel.prefill(promptTokens: promptTokens)
        let draftPrefillLatencyMs = GenerationClock.milliseconds(start: draftPrefillStart, end: GenerationClock.now())

        try fullModel.reset()
        let fullPrefillStart = GenerationClock.now()
        _ = try fullModel.prefill(promptTokens: promptTokens)
        let fullPrefillLatencyMs = GenerationClock.milliseconds(start: fullPrefillStart, end: GenerationClock.now())

        var prefixTokens = promptTokens
        var generatedTokens: [UInt16] = []
        var acceptedPrefixLengths: [Int] = []
        var draftLatenciesMs: [Double] = []
        var verificationLatenciesMs: [Double] = []
        var fullAdvanceLatenciesMs: [Double] = []
        var draftResyncLatenciesMs: [Double] = []
        var totalDraftCandidates = 0
        var totalAcceptedCandidates = 0

        while generatedTokens.count < maxNewTokens {
            let roundCandidates = min(candidateCount, maxNewTokens - generatedTokens.count)

            var candidates: [UInt16] = []
            candidates.reserveCapacity(roundCandidates)
            let draftStart = GenerationClock.now()
            var roundLogits = draftLogits
            for idx in 0..<roundCandidates {
                let token = try selectToken(from: roundLogits, strategy: strategy)
                candidates.append(token)
                if idx + 1 < roundCandidates {
                    roundLogits = try draftModel.decode(nextToken: token)
                }
            }
            draftLatenciesMs.append(
                GenerationClock.milliseconds(start: draftStart, end: GenerationClock.now())
            )

            totalDraftCandidates += candidates.count

            let verifyStart = GenerationClock.now()
            let verificationLogits = try fullModel.verify(
                sequenceTokens: prefixTokens + candidates,
                startIndex: prefixTokens.count - 1
            )
            verificationLatenciesMs.append(
                GenerationClock.milliseconds(start: verifyStart, end: GenerationClock.now())
            )

            var accepted = 0
            while accepted < candidates.count {
                let predicted = try selectToken(from: verificationLogits[accepted], strategy: strategy)
                if predicted != candidates[accepted] {
                    break
                }
                accepted += 1
            }
            acceptedPrefixLengths.append(accepted)
            totalAcceptedCandidates += accepted

            let fullAdvanceStart = GenerationClock.now()
            for idx in 0..<accepted where generatedTokens.count < maxNewTokens {
                let token = candidates[idx]
                prefixTokens.append(token)
                generatedTokens.append(token)
                _ = try fullModel.decode(nextToken: token)
            }

            if generatedTokens.count < maxNewTokens {
                guard accepted < verificationLogits.count else {
                    throw .runtimeFailure("verification logits missing correction step at index \(accepted)")
                }
                let correctionToken = try selectToken(from: verificationLogits[accepted], strategy: strategy)
                prefixTokens.append(correctionToken)
                generatedTokens.append(correctionToken)
                _ = try fullModel.decode(nextToken: correctionToken)
            }

            fullAdvanceLatenciesMs.append(
                GenerationClock.milliseconds(start: fullAdvanceStart, end: GenerationClock.now())
            )

            if generatedTokens.count < maxNewTokens {
                let resyncStart = GenerationClock.now()
                try draftModel.reset()
                draftLogits = try draftModel.prefill(promptTokens: prefixTokens)
                draftResyncLatenciesMs.append(
                    GenerationClock.milliseconds(start: resyncStart, end: GenerationClock.now())
                )
            }
        }

        return SpeculativeGenerationTrace(
            promptTokens: promptTokens,
            generatedTokens: generatedTokens,
            candidateCount: candidateCount,
            draftPrefillLatencyMs: draftPrefillLatencyMs,
            fullPrefillLatencyMs: fullPrefillLatencyMs,
            draftLatenciesMs: draftLatenciesMs,
            verificationLatenciesMs: verificationLatenciesMs,
            fullAdvanceLatenciesMs: fullAdvanceLatenciesMs,
            draftResyncLatenciesMs: draftResyncLatenciesMs,
            acceptedPrefixLengths: acceptedPrefixLengths,
            totalDraftCandidates: totalDraftCandidates,
            totalAcceptedCandidates: totalAcceptedCandidates
        )
    }
}

public struct GenerationWeights: ~Copyable {
    public let layers: LayerStorage<LayerWeights>
    public let rmsFinal: TensorBuffer
    public let embedding: TensorBuffer
    public let classifier: TensorBuffer
    public let sharedClassifier: Bool
    public let vocabSize: Int

    public init(
        layers: consuming LayerStorage<LayerWeights>,
        rmsFinal: consuming TensorBuffer,
        embedding: consuming TensorBuffer,
        classifier: consuming TensorBuffer,
        sharedClassifier: Bool,
        vocabSize: Int = ModelConfig.vocab
    ) {
        self.layers = layers
        self.rmsFinal = rmsFinal
        self.embedding = embedding
        self.classifier = classifier
        self.sharedClassifier = sharedClassifier
        self.vocabSize = vocabSize
    }

    public init(cloning pretrained: borrowing PretrainedWeights) {
        self.layers = GenerationWeightCloner.cloneLayers(pretrained.layers)
        self.rmsFinal = GenerationWeightCloner.cloneTensor(pretrained.rmsFinal)
        self.embedding = GenerationWeightCloner.cloneTensor(pretrained.embed)
        self.classifier = GenerationWeightCloner.cloneTensor(pretrained.classifier)
        self.sharedClassifier = pretrained.sharedClassifier
        self.vocabSize = ModelConfig.vocab
    }

    public static func load(modelPath: String) throws(GenerationError) -> GenerationWeights {
        do {
            let pretrained = try ModelWeightLoader.load(from: modelPath)
            return GenerationWeights(cloning: pretrained)
        } catch {
            throw .modelLoadFailed("\(error)")
        }
    }
}

public struct ANEDirectGenerationModel: ~Copyable, AutoregressiveLanguageModel {
    public let vocabSize: Int
    public let layerCount: Int
    public let decodeMaxSeq: Int

    private let rmsFinal: TensorBuffer
    private let embedding: TensorBuffer
    private let classifier: TensorBuffer
    private let sharedClassifier: Bool
    private let decodeKernels: LayerStorage<DecodeKernelSet>
    private let inferenceKernels: LayerStorage<InferenceKernelSet>
    private let stepNorm: TensorBuffer
    private let stepLogits: TensorBuffer
    private let verifySequence: TensorBuffer
    private let verifyNorm: TensorBuffer
    private let verifyLogits: TensorBuffer
    private let stepRMSWorkspace: RMSNorm.Workspace
    private let verifyRMSWorkspace: RMSNorm.Workspace
    private var decodeHandles: [DecodeSurfaceHandles]
    private var inferenceHandles: [InferenceSurfaceHandles]
    private var decodeState: DecodeState
    private var xCur: TensorBuffer

    public init(
        weights: borrowing GenerationWeights,
        layerCount: Int,
        decodeMaxSeq: Int
    ) throws(GenerationError) {
        guard layerCount > 0 else {
            throw .invalidArguments("layerCount must be > 0")
        }
        guard layerCount <= weights.layers.count else {
            throw .invalidArguments("layerCount \(layerCount) exceeds available layers \(weights.layers.count)")
        }
        guard decodeMaxSeq > 0, decodeMaxSeq <= ModelConfig.seqLen else {
            throw .invalidArguments("decodeMaxSeq must be in 1...\(ModelConfig.seqLen)")
        }
        guard weights.vocabSize > 0 else {
            throw .invalidArguments("vocabSize must be > 0")
        }

        let decodeKernels = try Self.compileDecodeKernels(
            weights: weights,
            layerCount: layerCount,
            decodeMaxSeq: decodeMaxSeq
        )
        let inferenceKernels = try Self.compileInferenceKernels(
            weights: weights,
            layerCount: layerCount
        )
        let decodeHandles = try Self.makeDecodeHandles(
            kernels: decodeKernels,
            layerCount: layerCount,
            decodeMaxSeq: decodeMaxSeq
        )
        let inferenceHandles = try Self.makeInferenceHandles(
            kernels: inferenceKernels,
            layerCount: layerCount
        )
        let decodeState = try Self.makeDecodeState(maxSeq: decodeMaxSeq)

        self.vocabSize = weights.vocabSize
        self.layerCount = layerCount
        self.decodeMaxSeq = decodeMaxSeq
        self.rmsFinal = GenerationWeightCloner.cloneTensor(weights.rmsFinal)
        self.embedding = GenerationWeightCloner.cloneTensor(weights.embedding)
        self.classifier = weights.sharedClassifier
            ? TensorBuffer(count: 0, zeroed: true)
            : GenerationWeightCloner.cloneTensor(weights.classifier)
        self.sharedClassifier = weights.sharedClassifier
        self.decodeKernels = decodeKernels
        self.inferenceKernels = inferenceKernels
        self.stepNorm = TensorBuffer(count: ModelConfig.dim, zeroed: true)
        self.stepLogits = TensorBuffer(count: weights.vocabSize, zeroed: true)
        self.verifySequence = TensorBuffer(count: ModelConfig.dim * ModelConfig.seqLen, zeroed: true)
        self.verifyNorm = TensorBuffer(count: ModelConfig.dim * ModelConfig.seqLen, zeroed: true)
        self.verifyLogits = TensorBuffer(count: weights.vocabSize * ModelConfig.seqLen, zeroed: true)
        self.stepRMSWorkspace = RMSNorm.Workspace(seqLen: 1)
        self.verifyRMSWorkspace = RMSNorm.Workspace(seqLen: ModelConfig.seqLen)
        self.decodeHandles = decodeHandles
        self.inferenceHandles = inferenceHandles
        self.decodeState = decodeState
        self.xCur = TensorBuffer(count: ModelConfig.dim, zeroed: true)
    }

    public static func load(
        modelPath: String,
        layerCount: Int,
        decodeMaxSeq: Int
    ) throws(GenerationError) -> ANEDirectGenerationModel {
        let weights = try GenerationWeights.load(modelPath: modelPath)
        return try ANEDirectGenerationModel(
            weights: weights,
            layerCount: layerCount,
            decodeMaxSeq: decodeMaxSeq
        )
    }

    private static func compileDecodeKernels(
        weights: borrowing GenerationWeights,
        layerCount: Int,
        decodeMaxSeq: Int
    ) throws(GenerationError) -> LayerStorage<DecodeKernelSet> {
        do {
            return try LayerStorage<DecodeKernelSet>(count: layerCount, throwingInitializer: { idx in
                try DecodeKernelSet(weights: weights.layers[idx], maxSeq: decodeMaxSeq)
            })
        } catch {
            throw .runtimeFailure("decode kernel compilation failed: \(error)")
        }
    }

    private static func compileInferenceKernels(
        weights: borrowing GenerationWeights,
        layerCount: Int
    ) throws(GenerationError) -> LayerStorage<InferenceKernelSet> {
        do {
            return try LayerStorage<InferenceKernelSet>(count: layerCount, throwingInitializer: { idx in
                try InferenceKernelSet(weights: weights.layers[idx])
            })
        } catch {
            throw .runtimeFailure("inference kernel compilation failed: \(error)")
        }
    }

    private static func makeDecodeHandles(
        kernels: borrowing LayerStorage<DecodeKernelSet>,
        layerCount: Int,
        decodeMaxSeq: Int
    ) throws(GenerationError) -> [DecodeSurfaceHandles] {
        var decodeHandles: [DecodeSurfaceHandles] = []
        decodeHandles.reserveCapacity(layerCount)
        for idx in 0..<layerCount {
            do {
                decodeHandles.append(try DecodeSurfaceHandles(kernels: kernels[idx], logicalMaxSeq: decodeMaxSeq))
            } catch {
                throw .runtimeFailure("decode surface setup failed: \(error)")
            }
        }
        return decodeHandles
    }

    private static func makeInferenceHandles(
        kernels: borrowing LayerStorage<InferenceKernelSet>,
        layerCount: Int
    ) throws(GenerationError) -> [InferenceSurfaceHandles] {
        var inferenceHandles: [InferenceSurfaceHandles] = []
        inferenceHandles.reserveCapacity(layerCount)
        for idx in 0..<layerCount {
            do {
                inferenceHandles.append(try InferenceSurfaceHandles(kernels: kernels[idx]))
            } catch {
                throw .runtimeFailure("inference surface setup failed: \(error)")
            }
        }
        return inferenceHandles
    }

    private static func makeDecodeState(maxSeq: Int) throws(GenerationError) -> DecodeState {
        do {
            return try DecodeState(maxSeq: maxSeq)
        } catch {
            throw .runtimeFailure("decode state setup failed: \(error)")
        }
    }

    public mutating func reset() throws(GenerationError) {
        ForwardPass.initializeDecodeCachesAndMask(surfaceHandles: decodeHandles)
        decodeState.reset()
        xCur.zero()
    }

    public mutating func prefill(promptTokens: [UInt16]) throws(GenerationError) -> [Float] {
        guard !promptTokens.isEmpty else {
            throw .invalidArguments("promptTokens must not be empty")
        }
        guard promptTokens.count <= decodeMaxSeq else {
            throw .invalidArguments("prompt length \(promptTokens.count) exceeds decodeMaxSeq \(decodeMaxSeq)")
        }

        for token in promptTokens {
            try runDecodeStep(token: token)
        }
        return try projectStepLogits()
    }

    public mutating func decode(nextToken: UInt16) throws(GenerationError) -> [Float] {
        try runDecodeStep(token: nextToken)
        return try projectStepLogits()
    }

    public mutating func verify(
        sequenceTokens: [UInt16],
        startIndex: Int
    ) throws(GenerationError) -> [[Float]] {
        guard !sequenceTokens.isEmpty else {
            throw .invalidArguments("sequenceTokens must not be empty")
        }
        guard sequenceTokens.count <= ModelConfig.seqLen else {
            throw .invalidArguments("sequence length \(sequenceTokens.count) exceeds verify window \(ModelConfig.seqLen)")
        }
        guard startIndex >= 0, startIndex < sequenceTokens.count else {
            throw .invalidArguments("startIndex \(startIndex) must be within sequence length \(sequenceTokens.count)")
        }

        try writeSequenceTokens(sequenceTokens)

        var timings = StepTimingBreakdown()
        do {
            try ForwardPass.runInferenceTimed(
                xCur: verifySequence,
                kernels: inferenceKernels,
                surfaceHandles: inferenceHandles,
                handoff: .fp16SurfaceCopy,
                timings: &timings
            )
        } catch {
            throw .runtimeFailure("verify inference failed: \(error)")
        }

        do {
            try projectSequenceLogits()
        } catch {
            throw .runtimeFailure("verify logits projection failed: \(error)")
        }

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
        return outputs
    }

    @inline(__always)
    private func classifierPointer<R>(_ body: (UnsafePointer<Float>) throws -> R) rethrows -> R {
        if sharedClassifier {
            return try embedding.withUnsafePointer(body)
        }
        return try classifier.withUnsafePointer(body)
    }

    private mutating func runDecodeStep(token: UInt16) throws(GenerationError) {
        guard Int(token) < vocabSize else {
            throw .invalidArguments("token \(token) exceeds vocab size \(vocabSize)")
        }
        guard decodeState.visibleTokenCount < decodeMaxSeq else {
            throw .invalidArguments("decodeState overflow at maxSeq \(decodeMaxSeq)")
        }

        xCur.withUnsafeMutablePointer { dst in
            embedding.withUnsafePointer { embeddingPtr in
                let base = Int(token) * ModelConfig.dim
                for dimIdx in 0..<ModelConfig.dim {
                    dst[dimIdx] = embeddingPtr[base + dimIdx]
                }
            }
        }

        var timings = StepTimingBreakdown()
        do {
            try ForwardPass.runDecodeTimed(
                xCur: xCur,
                kernels: decodeKernels,
                surfaceHandles: decodeHandles,
                decodeState: &decodeState,
                timings: &timings
            )
        } catch {
            throw .runtimeFailure("decode failed: \(error)")
        }
    }

    private func projectStepLogits() throws(GenerationError) -> [Float] {
        xCur.withUnsafePointer { xPtr in
            stepNorm.withUnsafeMutablePointer { normPtr in
                rmsFinal.withUnsafePointer { rmsPtr in
                    RMSNorm.forward(
                        output: normPtr,
                        input: xPtr,
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

        return stepLogits.withUnsafeBufferPointer { Array($0) }
    }

    private mutating func writeSequenceTokens(_ tokens: [UInt16]) throws(GenerationError) {
        verifySequence.zero()
        for token in tokens where Int(token) >= vocabSize {
            throw .invalidArguments("token \(token) exceeds vocab size \(vocabSize)")
        }
        verifySequence.withUnsafeMutablePointer { dst in
            embedding.withUnsafePointer { embeddingPtr in
                for (seqIdx, token) in tokens.enumerated() {
                    let tokenBase = Int(token) * ModelConfig.dim
                    for dimIdx in 0..<ModelConfig.dim {
                        dst[dimIdx * ModelConfig.seqLen + seqIdx] = embeddingPtr[tokenBase + dimIdx]
                    }
                }
            }
        }
    }

    private func projectSequenceLogits() throws(GenerationError) {
        verifyNorm.zero()
        verifyLogits.zero()

        verifySequence.withUnsafePointer { inPtr in
            verifyNorm.withUnsafeMutablePointer { outPtr in
                rmsFinal.withUnsafePointer { rmsPtr in
                    RMSNorm.forward(
                        output: outPtr,
                        input: inPtr,
                        weights: rmsPtr,
                        dim: ModelConfig.dim,
                        seqLen: ModelConfig.seqLen,
                        workspace: verifyRMSWorkspace
                    )
                }
            }
        }

        verifyLogits.withUnsafeMutablePointer { logitsPtr in
            classifierPointer { clsPtr in
                verifyNorm.withUnsafePointer { normPtr in
                    BLAS.sgemm(
                        CblasRowMajor,
                        CblasNoTrans,
                        CblasNoTrans,
                        m: Int32(vocabSize),
                        n: Int32(ModelConfig.seqLen),
                        k: Int32(ModelConfig.dim),
                        alpha: 1.0,
                        a: clsPtr,
                        lda: Int32(ModelConfig.dim),
                        b: normPtr,
                        ldb: Int32(ModelConfig.seqLen),
                        beta: 0.0,
                        c: logitsPtr,
                        ldc: Int32(ModelConfig.seqLen)
                    )
                }
            }
        }
    }
}
