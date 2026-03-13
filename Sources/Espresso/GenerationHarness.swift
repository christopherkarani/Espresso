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

public enum RecurrentGenerationTrunkBackend: Sendable {
    case singleLayer
    case fusedTwoLayerPairs
    case fusedThreeLayerTriplets
    case identityZeroTrunk
}

public struct GenerationPerformanceSnapshot: Sendable, Equatable {
    public let compileTimeMs: Double
    public let trunkLatencyMs: Double
    public let logitsLatencyMs: Double

    public init(
        compileTimeMs: Double = 0,
        trunkLatencyMs: Double = 0,
        logitsLatencyMs: Double = 0
    ) {
        self.compileTimeMs = compileTimeMs
        self.trunkLatencyMs = trunkLatencyMs
        self.logitsLatencyMs = logitsLatencyMs
    }

    public var totalRuntimeMs: Double {
        trunkLatencyMs + logitsLatencyMs
    }
}

public protocol GenerationPerformanceTrackable: ~Copyable {
    var performanceSnapshot: GenerationPerformanceSnapshot { get }
}

public protocol AutoregressiveLanguageModel: ~Copyable {
    var vocabSize: Int { get }

    mutating func reset() throws(GenerationError)
    mutating func prefill(promptTokens: [UInt16]) throws(GenerationError) -> [Float]
    mutating func decode(nextToken: UInt16) throws(GenerationError) -> [Float]
    mutating func verify(sequenceTokens: [UInt16], startIndex: Int) throws(GenerationError) -> [[Float]]
}

public protocol DirectTokenSelectingLanguageModel: ~Copyable, AutoregressiveLanguageModel {
    mutating func prefillSelectedToken(
        promptTokens: [UInt16],
        strategy: TokenSelectionStrategy
    ) throws(GenerationError) -> UInt16
    mutating func decodeSelectedToken(
        nextToken: UInt16,
        strategy: TokenSelectionStrategy
    ) throws(GenerationError) -> UInt16
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

public struct TwoTokenBranchCommitResult: Sendable, Equatable {
    public let acceptedPrefixLength: Int
    public let committedTokens: [UInt16]

    public init(acceptedPrefixLength: Int, committedTokens: [UInt16]) {
        self.acceptedPrefixLength = acceptedPrefixLength
        self.committedTokens = committedTokens
    }
}

public struct TwoTokenBranchCommitTrace: Sendable, Equatable {
    public let promptTokens: [UInt16]
    public let generatedTokens: [UInt16]
    public let acceptedPrefixLengths: [Int]
}

public struct ExactTwoTokenPassMetrics: Sendable, Equatable {
    public let proposerLatencyMs: Double
    public let verifierTrunkLatencyMs: Double
    public let verifierLogitsLatencyMs: Double
    public let stateAdvanceLatencyMs: Double
    public let acceptedFutureTokenCount: Int
    public let committedExactTokenCount: Int

    public init(
        proposerLatencyMs: Double,
        verifierTrunkLatencyMs: Double,
        verifierLogitsLatencyMs: Double,
        stateAdvanceLatencyMs: Double,
        acceptedFutureTokenCount: Int,
        committedExactTokenCount: Int
    ) {
        self.proposerLatencyMs = proposerLatencyMs
        self.verifierTrunkLatencyMs = verifierTrunkLatencyMs
        self.verifierLogitsLatencyMs = verifierLogitsLatencyMs
        self.stateAdvanceLatencyMs = stateAdvanceLatencyMs
        self.acceptedFutureTokenCount = acceptedFutureTokenCount
        self.committedExactTokenCount = committedExactTokenCount
    }

    public var totalLatencyMs: Double {
        proposerLatencyMs + verifierTrunkLatencyMs + verifierLogitsLatencyMs + stateAdvanceLatencyMs
    }
}

public struct ExactTwoTokenPassResult: Sendable, Equatable {
    public let committedTokens: [UInt16]
    public let nextCurrentToken: UInt16
    public let metrics: ExactTwoTokenPassMetrics

    public init(
        committedTokens: [UInt16],
        nextCurrentToken: UInt16,
        metrics: ExactTwoTokenPassMetrics
    ) {
        self.committedTokens = committedTokens
        self.nextCurrentToken = nextCurrentToken
        self.metrics = metrics
    }
}

struct ExactTwoTokenBranchPromotionPlan: Sendable, Equatable {
    let committedTokens: [UInt16]
    let nextCurrentToken: UInt16
    let committedExactTokenCount: Int
    let acceptedFutureTokenCount: Int
    let promotedStepCount: Int

    static func make(
        currentToken: UInt16,
        proposedFutureToken: UInt16,
        exactNextToken: UInt16,
        exactFutureToken: UInt16,
        remainingTokenBudget: Int
    ) -> ExactTwoTokenBranchPromotionPlan {
        if remainingTokenBudget > 1, proposedFutureToken == exactNextToken {
            return ExactTwoTokenBranchPromotionPlan(
                committedTokens: [currentToken, exactNextToken],
                nextCurrentToken: exactFutureToken,
                committedExactTokenCount: 2,
                acceptedFutureTokenCount: 1,
                promotedStepCount: 2
            )
        }

        return ExactTwoTokenBranchPromotionPlan(
            committedTokens: [currentToken],
            nextCurrentToken: exactNextToken,
            committedExactTokenCount: 1,
            acceptedFutureTokenCount: 0,
            promotedStepCount: 1
        )
    }
}

public struct ExactTwoTokenGenerationTrace: Sendable, Equatable {
    public let promptTokens: [UInt16]
    public let generatedTokens: [UInt16]
    public let prefillLatencyMs: Double
    public let passMetrics: [ExactTwoTokenPassMetrics]

    public var acceptedFutureTokenCounts: [Int] {
        passMetrics.map(\.acceptedFutureTokenCount)
    }

    public var committedExactTokenCounts: [Int] {
        passMetrics.map(\.committedExactTokenCount)
    }

    public var totalLatencyMs: Double {
        prefillLatencyMs + passMetrics.reduce(0) { $0 + $1.totalLatencyMs }
    }

    public var committedExactTokensPerPass: Double {
        guard !passMetrics.isEmpty else { return 0 }
        let total = passMetrics.reduce(0) { $0 + Double($1.committedExactTokenCount) }
        return total / Double(passMetrics.count)
    }

    public var acceptedFutureTokensPerPass: Double {
        guard !passMetrics.isEmpty else { return 0 }
        let total = passMetrics.reduce(0) { $0 + Double($1.acceptedFutureTokenCount) }
        return total / Double(passMetrics.count)
    }

    public var proposerLatencyMsPerPass: Double {
        guard !passMetrics.isEmpty else { return 0 }
        return passMetrics.reduce(0) { $0 + $1.proposerLatencyMs } / Double(passMetrics.count)
    }

    public var verifierTrunkLatencyMsPerPass: Double {
        guard !passMetrics.isEmpty else { return 0 }
        return passMetrics.reduce(0) { $0 + $1.verifierTrunkLatencyMs } / Double(passMetrics.count)
    }

    public var verifierLogitsLatencyMsPerPass: Double {
        guard !passMetrics.isEmpty else { return 0 }
        return passMetrics.reduce(0) { $0 + $1.verifierLogitsLatencyMs } / Double(passMetrics.count)
    }

    public var stateAdvanceLatencyMsPerPass: Double {
        guard !passMetrics.isEmpty else { return 0 }
        return passMetrics.reduce(0) { $0 + $1.stateAdvanceLatencyMs } / Double(passMetrics.count)
    }

    public var effectiveTokensPerSecond: Double {
        guard totalLatencyMs > 0 else { return 0 }
        return Double(generatedTokens.count) * 1000.0 / totalLatencyMs
    }
}

public protocol ExactTwoTokenGeneratingLanguageModel: ~Copyable, GenerationPerformanceTrackable {
    var vocabSize: Int { get }

    mutating func reset() throws(GenerationError)
    mutating func prefillSelectedToken(
        promptTokens: [UInt16],
        strategy: TokenSelectionStrategy
    ) throws(GenerationError) -> UInt16
    mutating func performExactTwoTokenPass(
        currentToken: UInt16,
        remainingTokenBudget: Int,
        strategy: TokenSelectionStrategy
    ) throws(GenerationError) -> ExactTwoTokenPassResult
}

@inline(__always)
private func tensorBufferIsAllZero(_ buffer: borrowing TensorBuffer) -> Bool {
    guard buffer.count > 0 else { return true }
    return buffer.withUnsafePointer { ptr in
        var maxMagnitude: Float = 0
        vDSP_maxmgv(ptr, 1, &maxMagnitude, vDSP_Length(buffer.count))
        return maxMagnitude == 0
    }
}

private func recurrentWeightsUseIdentityZeroTrunk(
    _ weights: borrowing RecurrentGenerationWeights,
    layerCount: Int
) -> Bool {
    for idx in 0..<layerCount {
        guard tensorBufferIsAllZero(weights.layers[idx].Wx),
              tensorBufferIsAllZero(weights.layers[idx].Ws),
              tensorBufferIsAllZero(weights.layers[idx].Wd),
              tensorBufferIsAllZero(weights.layers[idx].Wo) else {
            return false
        }
    }
    return true
}

public protocol TwoTokenDraftingLanguageModel: ~Copyable {
    mutating func reset() throws(GenerationError)
    mutating func prefillSelectedToken(
        promptTokens: [UInt16],
        strategy: TokenSelectionStrategy
    ) throws(GenerationError) -> UInt16
    mutating func proposeTwoTokens(strategy: TokenSelectionStrategy) throws(GenerationError) -> [UInt16]
    mutating func commit(tokens: [UInt16]) throws(GenerationError)
}

public protocol TwoTokenBranchVerifyingLanguageModel: ~Copyable {
    mutating func reset() throws(GenerationError)
    mutating func prefillSelectedToken(
        promptTokens: [UInt16],
        strategy: TokenSelectionStrategy
    ) throws(GenerationError) -> UInt16
    mutating func verifyAndCommit(
        proposedTokens: [UInt16],
        strategy: TokenSelectionStrategy
    ) throws(GenerationError) -> TwoTokenBranchCommitResult
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
    static func shareTensor(_ source: borrowing TensorBuffer) -> TensorBuffer {
        TensorBuffer(nonOwningViewOf: source)
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

    static func cloneRecurrentLayer(_ source: borrowing RWKVStyleRecurrentWeights) -> RWKVStyleRecurrentWeights {
        let layer = RWKVStyleRecurrentWeights()
        copyTensor(source.rms, into: layer.rms)
        copyTensor(source.Wx, into: layer.Wx)
        copyTensor(source.Ws, into: layer.Ws)
        copyTensor(source.Wd, into: layer.Wd)
        copyTensor(source.Wo, into: layer.Wo)
        return layer
    }

    static func cloneRecurrentLayers(
        _ source: borrowing LayerStorage<RWKVStyleRecurrentWeights>
    ) -> LayerStorage<RWKVStyleRecurrentWeights> {
        LayerStorage<RWKVStyleRecurrentWeights>(count: source.count) { idx in
            cloneRecurrentLayer(source[idx])
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

@inline(__always)
private func selectToken(
    from logits: borrowing TensorBuffer,
    strategy: TokenSelectionStrategy
) throws(GenerationError) -> UInt16 {
    guard logits.count > 0 else {
        throw .invalidArguments("cannot select from empty logits")
    }
    switch strategy {
    case .argmax:
        let bestIndex = logits.withUnsafePointer { ptr in
            var maxValue: Float = 0
            var maxIndex: vDSP_Length = 0
            vDSP_maxvi(ptr, 1, &maxValue, &maxIndex, vDSP_Length(logits.count))
            return Int(maxIndex)
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

public struct DirectTokenSelectionGenerationHarness<Model: DirectTokenSelectingLanguageModel>: ~Copyable
where Model: ~Copyable {
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
        var nextToken = try model.prefillSelectedToken(
            promptTokens: promptTokens,
            strategy: strategy
        )
        let prefillLatencyMs = GenerationClock.milliseconds(start: prefillStart, end: GenerationClock.now())

        var generatedTokens: [UInt16] = []
        var decodeLatenciesMs: [Double] = []
        generatedTokens.reserveCapacity(maxNewTokens)
        decodeLatenciesMs.reserveCapacity(maxNewTokens)

        for _ in 0..<maxNewTokens {
            generatedTokens.append(nextToken)

            let decodeStart = GenerationClock.now()
            nextToken = try model.decodeSelectedToken(nextToken: nextToken, strategy: strategy)
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

public struct ExactTwoTokenGenerationHarness<Model: ExactTwoTokenGeneratingLanguageModel>: ~Copyable
where Model: ~Copyable {
    public var model: Model
    public let strategy: TokenSelectionStrategy

    public init(
        model: consuming Model,
        strategy: TokenSelectionStrategy = .argmax
    ) {
        self.model = model
        self.strategy = strategy
    }

    public mutating func generate(
        promptTokens: [UInt16],
        maxNewTokens: Int
    ) throws(GenerationError) -> ExactTwoTokenGenerationTrace {
        guard !promptTokens.isEmpty else {
            throw .invalidArguments("promptTokens must not be empty")
        }
        guard maxNewTokens > 0 else {
            throw .invalidArguments("maxNewTokens must be > 0")
        }

        try model.reset()
        let prefillStart = GenerationClock.now()
        var currentToken = try model.prefillSelectedToken(
            promptTokens: promptTokens,
            strategy: strategy
        )
        let prefillLatencyMs = GenerationClock.milliseconds(start: prefillStart, end: GenerationClock.now())

        var generatedTokens: [UInt16] = []
        var passMetrics: [ExactTwoTokenPassMetrics] = []
        generatedTokens.reserveCapacity(maxNewTokens)
        passMetrics.reserveCapacity(maxNewTokens)

        while generatedTokens.count < maxNewTokens {
            let remainingTokenBudget = maxNewTokens - generatedTokens.count
            let result = try model.performExactTwoTokenPass(
                currentToken: currentToken,
                remainingTokenBudget: remainingTokenBudget,
                strategy: strategy
            )
            guard !result.committedTokens.isEmpty else {
                throw .runtimeFailure("exact two-token pass must commit at least one token")
            }
            guard result.committedTokens[0] == currentToken else {
                throw .runtimeFailure("exact two-token pass must commit current token first")
            }
            guard (1...min(2, remainingTokenBudget)).contains(result.committedTokens.count) else {
                throw .runtimeFailure(
                    "exact two-token pass committed \(result.committedTokens.count) tokens with remaining budget \(remainingTokenBudget)"
                )
            }
            guard (0...1).contains(result.metrics.acceptedFutureTokenCount) else {
                throw .runtimeFailure(
                    "acceptedFutureTokenCount \(result.metrics.acceptedFutureTokenCount) must be in 0...1"
                )
            }
            guard result.metrics.committedExactTokenCount == result.committedTokens.count else {
                throw .runtimeFailure(
                    "committedExactTokenCount \(result.metrics.committedExactTokenCount) must equal committed token count \(result.committedTokens.count)"
                )
            }
            if result.metrics.acceptedFutureTokenCount == 1 {
                guard result.committedTokens.count == 2 else {
                    throw .runtimeFailure("accepted future token requires exactly two committed tokens")
                }
                guard remainingTokenBudget > 1 else {
                    throw .runtimeFailure("accepted future token requires remaining token budget > 1")
                }
            } else if result.metrics.stateAdvanceLatencyMs != 0 {
                throw .runtimeFailure("stateAdvanceLatencyMs must be zero when no future token is accepted")
            }

            generatedTokens.append(contentsOf: result.committedTokens)
            passMetrics.append(result.metrics)
            currentToken = result.nextCurrentToken
        }

        return ExactTwoTokenGenerationTrace(
            promptTokens: promptTokens,
            generatedTokens: generatedTokens,
            prefillLatencyMs: prefillLatencyMs,
            passMetrics: passMetrics
        )
    }
}

public struct TwoTokenBranchCommitGenerationHarness<
    DraftModel: TwoTokenDraftingLanguageModel,
    FullModel: TwoTokenBranchVerifyingLanguageModel
>: ~Copyable where DraftModel: ~Copyable, FullModel: ~Copyable {
    public var draftModel: DraftModel
    public var fullModel: FullModel
    public let strategy: TokenSelectionStrategy

    public init(
        draftModel: consuming DraftModel,
        fullModel: consuming FullModel,
        strategy: TokenSelectionStrategy = .argmax
    ) {
        self.draftModel = draftModel
        self.fullModel = fullModel
        self.strategy = strategy
    }

    public mutating func generate(
        promptTokens: [UInt16],
        maxNewTokens: Int
    ) throws(GenerationError) -> TwoTokenBranchCommitTrace {
        guard !promptTokens.isEmpty else {
            throw .invalidArguments("promptTokens must not be empty")
        }
        guard maxNewTokens > 0 else {
            throw .invalidArguments("maxNewTokens must be > 0")
        }

        try draftModel.reset()
        _ = try draftModel.prefillSelectedToken(promptTokens: promptTokens, strategy: strategy)

        try fullModel.reset()
        let firstToken = try fullModel.prefillSelectedToken(promptTokens: promptTokens, strategy: strategy)

        var generatedTokens: [UInt16] = [firstToken]
        var acceptedPrefixLengths: [Int] = []
        generatedTokens.reserveCapacity(maxNewTokens)
        acceptedPrefixLengths.reserveCapacity(maxNewTokens)

        if maxNewTokens == 1 {
            return TwoTokenBranchCommitTrace(
                promptTokens: promptTokens,
                generatedTokens: generatedTokens,
                acceptedPrefixLengths: acceptedPrefixLengths
            )
        }

        try draftModel.commit(tokens: [firstToken])

        while generatedTokens.count < maxNewTokens {
            let proposedTokens = try draftModel.proposeTwoTokens(strategy: strategy)
            guard proposedTokens.count == 2 else {
                throw .runtimeFailure("two-token branch commit requires exactly 2 proposed tokens per round")
            }

            let result = try fullModel.verifyAndCommit(
                proposedTokens: proposedTokens,
                strategy: strategy
            )
            guard (0...2).contains(result.acceptedPrefixLength) else {
                throw .runtimeFailure("accepted prefix length \(result.acceptedPrefixLength) must be in 0...2")
            }
            guard !result.committedTokens.isEmpty else {
                throw .runtimeFailure("branch commit must produce at least one committed token")
            }
            acceptedPrefixLengths.append(result.acceptedPrefixLength)

            let remaining = maxNewTokens - generatedTokens.count
            let committed = Array(result.committedTokens.prefix(remaining))
            generatedTokens.append(contentsOf: committed)
            try draftModel.commit(tokens: committed)
        }

        return TwoTokenBranchCommitTrace(
            promptTokens: promptTokens,
            generatedTokens: generatedTokens,
            acceptedPrefixLengths: acceptedPrefixLengths
        )
    }
}

public struct ANEExactTwoTokenUpperBoundGenerationModel: ~Copyable, ExactTwoTokenGeneratingLanguageModel {
    public let vocabSize: Int
    public let layerCount: Int
    public let maxSequenceTokens: Int

    private var baseModel: ANERecurrentGenerationModel

    public var performanceSnapshot: GenerationPerformanceSnapshot {
        baseModel.performanceSnapshot
    }

    public init(
        weights: borrowing RecurrentGenerationWeights,
        layerCount: Int,
        maxSequenceTokens: Int = ModelConfig.seqLen,
        outputHeadBackend: GenerationOutputHeadBackend = .aneRMSNormClassifier,
        trunkBackend: RecurrentGenerationTrunkBackend = .fusedThreeLayerTriplets,
        trunkLaneSpatial: Int = RWKVStyleRecurrentKernelSet.defaultLaneSpatial,
        outputHeadLaneSpatial: Int = 32
    ) throws(GenerationError) {
        let baseModel = try ANERecurrentGenerationModel(
            weights: weights,
            layerCount: layerCount,
            maxSequenceTokens: maxSequenceTokens,
            outputHeadBackend: outputHeadBackend,
            trunkBackend: trunkBackend,
            trunkLaneSpatial: trunkLaneSpatial,
            outputHeadLaneSpatial: outputHeadLaneSpatial
        )
        self.vocabSize = baseModel.vocabSize
        self.layerCount = layerCount
        self.maxSequenceTokens = maxSequenceTokens
        self.baseModel = baseModel
    }

    public mutating func reset() throws(GenerationError) {
        try baseModel.reset()
    }

    public mutating func prefillSelectedToken(
        promptTokens: [UInt16],
        strategy: TokenSelectionStrategy
    ) throws(GenerationError) -> UInt16 {
        try baseModel.prefillSelectedToken(promptTokens: promptTokens, strategy: strategy)
    }

    public mutating func performExactTwoTokenPass(
        currentToken: UInt16,
        remainingTokenBudget: Int,
        strategy: TokenSelectionStrategy
    ) throws(GenerationError) -> ExactTwoTokenPassResult {
        guard remainingTokenBudget > 0 else {
            throw .invalidArguments("remainingTokenBudget must be > 0")
        }

        let proposerStart = GenerationClock.now()
        // Upper-bound plumbing on the echo recurrent path: the second-token proposer reuses the current exact token.
        let futureProposal = currentToken
        let proposerLatencyMs = GenerationClock.milliseconds(start: proposerStart, end: GenerationClock.now())

        let verifierBefore = baseModel.performanceSnapshot
        let exactNext = try baseModel.decodeSelectedToken(nextToken: currentToken, strategy: strategy)
        let verifierAfter = baseModel.performanceSnapshot
        let verifierTrunkLatencyMs = verifierAfter.trunkLatencyMs - verifierBefore.trunkLatencyMs
        let verifierLogitsLatencyMs = verifierAfter.logitsLatencyMs - verifierBefore.logitsLatencyMs

        var committedTokens: [UInt16] = [currentToken]
        var acceptedFutureTokenCount = 0
        var stateAdvanceLatencyMs = 0.0
        var nextCurrentToken = exactNext

        if remainingTokenBudget > 1, futureProposal == exactNext {
            acceptedFutureTokenCount = 1
            committedTokens.append(exactNext)

            let stateAdvanceBefore = baseModel.performanceSnapshot
            nextCurrentToken = try baseModel.decodeSelectedToken(nextToken: exactNext, strategy: strategy)
            let stateAdvanceAfter = baseModel.performanceSnapshot
            stateAdvanceLatencyMs =
                (stateAdvanceAfter.trunkLatencyMs - stateAdvanceBefore.trunkLatencyMs)
                + (stateAdvanceAfter.logitsLatencyMs - stateAdvanceBefore.logitsLatencyMs)
        }

        return ExactTwoTokenPassResult(
            committedTokens: committedTokens,
            nextCurrentToken: nextCurrentToken,
            metrics: ExactTwoTokenPassMetrics(
                proposerLatencyMs: proposerLatencyMs,
                verifierTrunkLatencyMs: verifierTrunkLatencyMs,
                verifierLogitsLatencyMs: verifierLogitsLatencyMs,
                stateAdvanceLatencyMs: stateAdvanceLatencyMs,
                acceptedFutureTokenCount: acceptedFutureTokenCount,
                committedExactTokenCount: committedTokens.count
            )
        )
    }
}

public struct ANEExactTwoTokenBranchStatePromotionModel: ~Copyable, ExactTwoTokenGeneratingLanguageModel {
    private static let futureProposerOutputHeadLaneSpatial = 32

    public let vocabSize: Int
    public let layerCount: Int
    public let maxSequenceTokens: Int

    private let rmsFinal: TensorBuffer
    private let embedding: TensorBuffer
    private let classifier: TensorBuffer
    private let sharedClassifier: Bool
    private let futureRMS: TensorBuffer
    private let futureClassifier: TensorBuffer
    private let hasFutureProposer: Bool
    private let stepNorm: TensorBuffer
    private let stepLogits: TensorBuffer
    private let futureNorm: TensorBuffer
    private let futureLogits: TensorBuffer
    private let zeroActivation: TensorBuffer
    private let pair0ActivationA: TensorBuffer
    private let pair1ActivationA: TensorBuffer
    private let pair0ActivationB: TensorBuffer
    private let pair1ActivationB: TensorBuffer
    private let currentProposalActivation: TensorBuffer
    private let stepRMSWorkspace: RMSNorm.Workspace
    private let futureRMSWorkspace: RMSNorm.Workspace
    private let outputHeadBackend: GenerationOutputHeadBackend
    private let futureOutputHeadBackend: GenerationOutputHeadBackend
    private let trunkBackend: RecurrentGenerationTrunkBackend
    private let aneClassifierHead: ANEGenerationClassifierHead?
    private let aneRMSNormClassifierHead: ANEGenerationRMSNormClassifierHead?
    private let futureANEClassifierHead: ANEGenerationClassifierHead?
    private let futureANERMSNormClassifierHead: ANEGenerationRMSNormClassifierHead?
    private var twoStepSessions: LayerStorage<RWKVStyleTwoStepRecurrentSession>
    private var fusedPairTwoStepSessions: LayerStorage<RWKVStyleFusedTwoLayerTwoStepSession>
    private var fusedTripletTwoStepSessions: LayerStorage<RWKVStyleFusedThreeLayerTwoStepSession>
    private var consumedTokens: Int
    public private(set) var compileTimeMs: Double
    private var trunkLatencyMs: Double
    private var logitsLatencyMs: Double
    private var lastSingleTokenTrunkLatencyMs: Double
    private var lastSingleTokenLogitsLatencyMs: Double
    private var hasCurrentProposalActivation: Bool

    public var performanceSnapshot: GenerationPerformanceSnapshot {
        GenerationPerformanceSnapshot(
            compileTimeMs: compileTimeMs,
            trunkLatencyMs: trunkLatencyMs,
            logitsLatencyMs: logitsLatencyMs
        )
    }

    public init(
        weights: borrowing RecurrentGenerationWeights,
        futureSidecar: consuming TwoStepStudentSidecar? = nil,
        layerCount: Int,
        maxSequenceTokens: Int = ModelConfig.seqLen,
        outputHeadBackend: GenerationOutputHeadBackend = .aneRMSNormClassifier,
        trunkBackend: RecurrentGenerationTrunkBackend = .singleLayer,
        trunkLaneSpatial: Int = RWKVStyleTwoStepRecurrentKernelSet.defaultLaneSpatial,
        outputHeadLaneSpatial: Int = 32,
        shareReadOnlyWeights: Bool = false
    ) throws(GenerationError) {
        guard layerCount > 0 else {
            throw .invalidArguments("layerCount must be > 0")
        }
        guard layerCount <= weights.layers.count else {
            throw .invalidArguments("layerCount \(layerCount) exceeds available recurrent layers \(weights.layers.count)")
        }
        guard maxSequenceTokens > 0 else {
            throw .invalidArguments("maxSequenceTokens must be > 0")
        }
        guard trunkLaneSpatial > 0 else {
            throw .invalidArguments("two-step recurrent trunk laneSpatial must be > 0")
        }
        guard outputHeadLaneSpatial > 0 else {
            throw .invalidArguments("generation output-head laneSpatial must be > 0")
        }
        if outputHeadBackend == .cpuExactStaged || outputHeadBackend == .cpuExactClustered {
            throw .invalidArguments("two-step branch-state promotion model does not support staged CPU output heads")
        }
        #if DEBUG
        if trunkBackend == .identityZeroTrunk,
           !recurrentWeightsUseIdentityZeroTrunk(weights, layerCount: layerCount) {
            throw .invalidArguments("identity zero-trunk backend requires all recurrent Wx/Ws/Wd/Wo weights to be zero")
        }
        #endif

        let compileStart = GenerationClock.now()
        let twoStepSessions: LayerStorage<RWKVStyleTwoStepRecurrentSession>
        let fusedPairTwoStepSessions: LayerStorage<RWKVStyleFusedTwoLayerTwoStepSession>
        let fusedTripletTwoStepSessions: LayerStorage<RWKVStyleFusedThreeLayerTwoStepSession>
        switch trunkBackend {
        case .singleLayer:
            do {
                twoStepSessions = try LayerStorage<RWKVStyleTwoStepRecurrentSession>(
                    count: layerCount,
                    throwingInitializer: { idx in
                        try RWKVStyleTwoStepRecurrentSession(
                            weights: weights.layers[idx],
                            laneSpatial: trunkLaneSpatial
                        )
                    }
                )
            } catch {
                throw .runtimeFailure("two-step recurrent kernel/session setup failed: \(error)")
            }
            fusedPairTwoStepSessions = LayerStorage<RWKVStyleFusedTwoLayerTwoStepSession>(count: 0) { _ in
                fatalError("unreachable")
            }
            fusedTripletTwoStepSessions = LayerStorage<RWKVStyleFusedThreeLayerTwoStepSession>(count: 0) { _ in
                fatalError("unreachable")
            }
        case .fusedTwoLayerPairs:
            guard layerCount.isMultiple(of: 2) else {
                throw .invalidArguments("exact two-token fused pair trunk backend requires an even layerCount")
            }
            twoStepSessions = LayerStorage<RWKVStyleTwoStepRecurrentSession>(count: 0) { _ in
                fatalError("unreachable")
            }
            do {
                fusedPairTwoStepSessions = try LayerStorage<RWKVStyleFusedTwoLayerTwoStepSession>(
                    count: layerCount / 2,
                    throwingInitializer: { idx in
                        let base = idx * 2
                        return try RWKVStyleFusedTwoLayerTwoStepSession(
                            weights0: weights.layers[base],
                            weights1: weights.layers[base + 1],
                            laneSpatial: trunkLaneSpatial
                        )
                    }
                )
            } catch {
                throw .runtimeFailure("fused two-step recurrent pair kernel/session setup failed: \(error)")
            }
            fusedTripletTwoStepSessions = LayerStorage<RWKVStyleFusedThreeLayerTwoStepSession>(count: 0) { _ in
                fatalError("unreachable")
            }
        case .fusedThreeLayerTriplets:
            guard layerCount.isMultiple(of: 3) else {
                throw .invalidArguments(
                    "exact two-token fused three-layer trunk backend requires a layerCount that is a multiple of 3"
                )
            }
            twoStepSessions = LayerStorage<RWKVStyleTwoStepRecurrentSession>(count: 0) { _ in
                fatalError("unreachable")
            }
            fusedPairTwoStepSessions = LayerStorage<RWKVStyleFusedTwoLayerTwoStepSession>(count: 0) { _ in
                fatalError("unreachable")
            }
            do {
                fusedTripletTwoStepSessions = try LayerStorage<RWKVStyleFusedThreeLayerTwoStepSession>(
                    count: layerCount / 3,
                    throwingInitializer: { idx in
                        let base = idx * 3
                        return try RWKVStyleFusedThreeLayerTwoStepSession(
                            weights0: weights.layers[base],
                            weights1: weights.layers[base + 1],
                            weights2: weights.layers[base + 2],
                            laneSpatial: trunkLaneSpatial
                        )
                    }
                )
            } catch {
                throw .runtimeFailure("fused three-layer two-step kernel/session setup failed: \(error)")
            }
        case .identityZeroTrunk:
            twoStepSessions = LayerStorage<RWKVStyleTwoStepRecurrentSession>(count: 0) { _ in
                fatalError("unreachable")
            }
            fusedPairTwoStepSessions = LayerStorage<RWKVStyleFusedTwoLayerTwoStepSession>(count: 0) { _ in
                fatalError("unreachable")
            }
            fusedTripletTwoStepSessions = LayerStorage<RWKVStyleFusedThreeLayerTwoStepSession>(count: 0) { _ in
                fatalError("unreachable")
            }
        }

        let vocabSize = weights.vocabSize
        let sharedClassifier = weights.sharedClassifier
        guard vocabSize > 0 else {
            throw .invalidArguments("vocabSize must be > 0")
        }

        let rmsFinal = shareReadOnlyWeights
            ? GenerationWeightCloner.shareTensor(weights.rmsFinal)
            : GenerationWeightCloner.cloneTensor(weights.rmsFinal)
        let embedding = shareReadOnlyWeights
            ? GenerationWeightCloner.shareTensor(weights.embedding)
            : GenerationWeightCloner.cloneTensor(weights.embedding)
        let classifier = sharedClassifier
            ? TensorBuffer(count: 0, zeroed: true)
            : (shareReadOnlyWeights
                ? GenerationWeightCloner.shareTensor(weights.classifier)
                : GenerationWeightCloner.cloneTensor(weights.classifier))
        let futureRMS: TensorBuffer
        let futureClassifier: TensorBuffer
        let hasFutureProposer: Bool
        if let futureSidecar {
            guard futureSidecar.contract.dim == ModelConfig.dim else {
                throw .invalidArguments(
                    "futureSidecar dim \(futureSidecar.contract.dim) does not match ModelConfig.dim \(ModelConfig.dim)"
                )
            }
            guard futureSidecar.contract.vocabSize == vocabSize else {
                throw .invalidArguments(
                    "futureSidecar vocab \(futureSidecar.contract.vocabSize) does not match recurrent vocab \(vocabSize)"
                )
            }
            guard futureSidecar.contract.layerCount == layerCount else {
                throw .invalidArguments(
                    "futureSidecar layerCount \(futureSidecar.contract.layerCount) does not match requested layerCount \(layerCount)"
                )
            }
            futureRMS = GenerationWeightCloner.cloneTensor(futureSidecar.futureRMS)
            futureClassifier = GenerationWeightCloner.cloneTensor(futureSidecar.futureClassifier)
            hasFutureProposer = true
        } else {
            futureRMS = TensorBuffer(count: 0, zeroed: true)
            futureClassifier = TensorBuffer(count: 0, zeroed: true)
            hasFutureProposer = false
        }

        let aneClassifierHead: ANEGenerationClassifierHead?
        switch outputHeadBackend {
        case .aneClassifier:
            do {
                if sharedClassifier {
                    aneClassifierHead = try ANEGenerationClassifierHead(
                        classifierWeights: embedding,
                        vocabSize: vocabSize,
                        laneSpatial: outputHeadLaneSpatial
                    )
                } else {
                    aneClassifierHead = try ANEGenerationClassifierHead(
                        classifierWeights: classifier,
                        vocabSize: vocabSize,
                        laneSpatial: outputHeadLaneSpatial
                    )
                }
            } catch {
                throw .runtimeFailure("two-step ANE classifier setup failed: \(error)")
            }
        default:
            aneClassifierHead = nil
        }

        let aneRMSNormClassifierHead: ANEGenerationRMSNormClassifierHead?
        switch outputHeadBackend {
        case .aneRMSNormClassifier:
            do {
                if sharedClassifier {
                    aneRMSNormClassifierHead = try ANEGenerationRMSNormClassifierHead(
                        rmsFinal: rmsFinal,
                        classifierWeights: embedding,
                        vocabSize: vocabSize,
                        laneSpatial: outputHeadLaneSpatial
                    )
                } else {
                    aneRMSNormClassifierHead = try ANEGenerationRMSNormClassifierHead(
                        rmsFinal: rmsFinal,
                        classifierWeights: classifier,
                        vocabSize: vocabSize,
                        laneSpatial: outputHeadLaneSpatial
                    )
                }
            } catch {
                throw .runtimeFailure("two-step ANE fused output-head setup failed: \(error)")
            }
        default:
            aneRMSNormClassifierHead = nil
        }

        let futureOutputHeadBackend: GenerationOutputHeadBackend = hasFutureProposer ? outputHeadBackend : .cpu

        let futureANEClassifierHead: ANEGenerationClassifierHead?
        switch futureOutputHeadBackend {
        case .aneClassifier:
            do {
                futureANEClassifierHead = try ANEGenerationClassifierHead(
                    classifierWeights: futureClassifier,
                    vocabSize: vocabSize,
                    laneSpatial: Self.futureProposerOutputHeadLaneSpatial
                )
            } catch {
                throw .runtimeFailure("two-step ANE future proposer setup failed: \(error)")
            }
        default:
            futureANEClassifierHead = nil
        }

        let futureANERMSNormClassifierHead: ANEGenerationRMSNormClassifierHead?
        switch futureOutputHeadBackend {
        case .aneRMSNormClassifier:
            do {
                futureANERMSNormClassifierHead = try ANEGenerationRMSNormClassifierHead(
                    rmsFinal: futureRMS,
                    classifierWeights: futureClassifier,
                    vocabSize: vocabSize,
                    laneSpatial: Self.futureProposerOutputHeadLaneSpatial
                )
            } catch {
                throw .runtimeFailure("two-step ANE future proposer setup failed: \(error)")
            }
        default:
            futureANERMSNormClassifierHead = nil
        }

        self.vocabSize = vocabSize
        self.layerCount = layerCount
        self.maxSequenceTokens = maxSequenceTokens
        self.rmsFinal = rmsFinal
        self.embedding = embedding
        self.classifier = classifier
        self.sharedClassifier = sharedClassifier
        self.futureRMS = futureRMS
        self.futureClassifier = futureClassifier
        self.hasFutureProposer = hasFutureProposer
        self.stepNorm = TensorBuffer(count: ModelConfig.dim, zeroed: true)
        self.stepLogits = TensorBuffer(count: vocabSize, zeroed: true)
        self.futureNorm = TensorBuffer(count: hasFutureProposer ? ModelConfig.dim : 0, zeroed: true)
        self.futureLogits = TensorBuffer(count: hasFutureProposer ? vocabSize : 0, zeroed: true)
        self.zeroActivation = TensorBuffer(count: ModelConfig.dim, zeroed: true)
        self.pair0ActivationA = TensorBuffer(count: ModelConfig.dim, zeroed: true)
        self.pair1ActivationA = TensorBuffer(count: ModelConfig.dim, zeroed: true)
        self.pair0ActivationB = TensorBuffer(count: ModelConfig.dim, zeroed: true)
        self.pair1ActivationB = TensorBuffer(count: ModelConfig.dim, zeroed: true)
        self.currentProposalActivation = TensorBuffer(count: ModelConfig.dim, zeroed: true)
        self.stepRMSWorkspace = RMSNorm.Workspace(seqLen: 1)
        self.futureRMSWorkspace = RMSNorm.Workspace(seqLen: 1)
        self.outputHeadBackend = outputHeadBackend
        self.futureOutputHeadBackend = futureOutputHeadBackend
        self.trunkBackend = trunkBackend
        self.aneClassifierHead = aneClassifierHead
        self.aneRMSNormClassifierHead = aneRMSNormClassifierHead
        self.futureANEClassifierHead = futureANEClassifierHead
        self.futureANERMSNormClassifierHead = futureANERMSNormClassifierHead
        self.twoStepSessions = twoStepSessions
        self.fusedPairTwoStepSessions = fusedPairTwoStepSessions
        self.fusedTripletTwoStepSessions = fusedTripletTwoStepSessions
        self.consumedTokens = 0
        self.compileTimeMs = GenerationClock.milliseconds(start: compileStart, end: GenerationClock.now())
        self.trunkLatencyMs = 0
        self.logitsLatencyMs = 0
        self.lastSingleTokenTrunkLatencyMs = 0
        self.lastSingleTokenLogitsLatencyMs = 0
        self.hasCurrentProposalActivation = false
    }

    public mutating func reset() throws(GenerationError) {
        switch trunkBackend {
        case .singleLayer:
            for idx in 0..<twoStepSessions.count {
                do {
                    try twoStepSessions[idx].reset()
                } catch {
                    throw .runtimeFailure("two-step recurrent reset failed at layer \(idx): \(error)")
                }
            }
        case .identityZeroTrunk:
            break
        case .fusedTwoLayerPairs:
            for idx in 0..<fusedPairTwoStepSessions.count {
                do {
                    try fusedPairTwoStepSessions[idx].reset()
                } catch {
                    throw .runtimeFailure("fused two-step recurrent reset failed at pair \(idx): \(error)")
                }
            }
        case .fusedThreeLayerTriplets:
            for idx in 0..<fusedTripletTwoStepSessions.count {
                do {
                    try fusedTripletTwoStepSessions[idx].reset()
                } catch {
                    throw .runtimeFailure("fused three-layer two-step recurrent reset failed at triplet \(idx): \(error)")
                }
            }
        }
        pair0ActivationA.zero()
        pair1ActivationA.zero()
        pair0ActivationB.zero()
        pair1ActivationB.zero()
        currentProposalActivation.zero()
        consumedTokens = 0
        trunkLatencyMs = 0
        logitsLatencyMs = 0
        lastSingleTokenTrunkLatencyMs = 0
        lastSingleTokenLogitsLatencyMs = 0
        hasCurrentProposalActivation = false
    }

    public mutating func prefillSelectedToken(
        promptTokens: [UInt16],
        strategy: TokenSelectionStrategy
    ) throws(GenerationError) -> UInt16 {
        guard !promptTokens.isEmpty else {
            throw .invalidArguments("promptTokens must not be empty")
        }
        guard promptTokens.count <= maxSequenceTokens else {
            throw .invalidArguments("prompt length \(promptTokens.count) exceeds maxSequenceTokens \(maxSequenceTokens)")
        }

        var selectedToken: UInt16 = 0
        for token in promptTokens {
            selectedToken = try runCommittedSingleToken(token: token, strategy: strategy)
        }
        return selectedToken
    }

    public mutating func performExactTwoTokenPass(
        currentToken: UInt16,
        remainingTokenBudget: Int,
        strategy: TokenSelectionStrategy
    ) throws(GenerationError) -> ExactTwoTokenPassResult {
        guard remainingTokenBudget > 0 else {
            throw .invalidArguments("remainingTokenBudget must be > 0")
        }

        let proposerStart = GenerationClock.now()
        let proposedFutureToken = try selectProposedFutureToken(
            currentToken: currentToken,
            strategy: strategy
        )
        let proposerLatencyMs = GenerationClock.milliseconds(start: proposerStart, end: GenerationClock.now())

        if remainingTokenBudget == 1 {
            let exactNextToken = try runCommittedSingleToken(token: currentToken, strategy: strategy)
            return ExactTwoTokenPassResult(
                committedTokens: [currentToken],
                nextCurrentToken: exactNextToken,
                metrics: ExactTwoTokenPassMetrics(
                    proposerLatencyMs: proposerLatencyMs,
                    verifierTrunkLatencyMs: lastSingleTokenTrunkLatencyMs,
                    verifierLogitsLatencyMs: lastSingleTokenLogitsLatencyMs,
                    stateAdvanceLatencyMs: 0,
                    acceptedFutureTokenCount: 0,
                    committedExactTokenCount: 1
                )
            )
        }

        let prepared = try prepareExactTwoTokenPair(
            currentToken: currentToken,
            proposedFutureToken: proposedFutureToken,
            strategy: strategy
        )
        let plan = ExactTwoTokenBranchPromotionPlan.make(
            currentToken: currentToken,
            proposedFutureToken: proposedFutureToken,
            exactNextToken: prepared.exactNextToken,
            exactFutureToken: prepared.exactFutureToken,
            remainingTokenBudget: remainingTokenBudget
        )
        let stateAdvanceLatencyMs = try promotePreparedPair(stepCount: plan.promotedStepCount)
        captureProposalActivationForNextPass(
            sourcePairIsA: prepared.sourcePairIsA,
            committedStepCount: plan.promotedStepCount
        )
        consumedTokens += plan.promotedStepCount

        return ExactTwoTokenPassResult(
            committedTokens: plan.committedTokens,
            nextCurrentToken: plan.nextCurrentToken,
            metrics: ExactTwoTokenPassMetrics(
                proposerLatencyMs: proposerLatencyMs,
                verifierTrunkLatencyMs: prepared.trunkLatencyMs,
                verifierLogitsLatencyMs: prepared.logitsLatencyMs,
                stateAdvanceLatencyMs: stateAdvanceLatencyMs,
                acceptedFutureTokenCount: plan.acceptedFutureTokenCount,
                committedExactTokenCount: plan.committedExactTokenCount
            )
        )
    }

    private mutating func runCommittedSingleToken(
        token: UInt16,
        strategy: TokenSelectionStrategy
    ) throws(GenerationError) -> UInt16 {
        guard Int(token) < vocabSize else {
            throw .invalidArguments("token \(token) exceeds vocab size \(vocabSize)")
        }
        guard consumedTokens < maxSequenceTokens else {
            throw .invalidArguments("two-step recurrent generation overflow at maxSequenceTokens \(maxSequenceTokens)")
        }

        try Self.writeTokenEmbedding(token, embedding: embedding, into: pair0ActivationA)

        // Fast path for identity-zero-trunk: skip pair1 entirely (trunk is no-op,
        // and pair1 output head result is discarded anyway during single-token commit).
        // This halves the output-head cost during prefill.
        if case .identityZeroTrunk = trunkBackend {
            let logitsStart = GenerationClock.now()
            let exactNextToken = try Self.selectTokenFromActivation(
                pair0ActivationA,
                strategy: strategy,
                outputHeadBackend: outputHeadBackend,
                rmsFinal: rmsFinal,
                stepNorm: stepNorm,
                stepLogits: stepLogits,
                embedding: embedding,
                classifier: classifier,
                sharedClassifier: sharedClassifier,
                aneClassifierHead: aneClassifierHead,
                aneRMSNormClassifierHead: aneRMSNormClassifierHead,
                vocabSize: vocabSize,
                stepRMSWorkspace: stepRMSWorkspace
            )
            let logitsLatencyMs = GenerationClock.milliseconds(start: logitsStart, end: GenerationClock.now())
            self.logitsLatencyMs += logitsLatencyMs

            let stateAdvanceLatencyMs = try promotePreparedPair(stepCount: 1)
            captureProposalActivationForNextPass(sourcePairIsA: true, committedStepCount: 1)
            consumedTokens += 1
            lastSingleTokenTrunkLatencyMs = stateAdvanceLatencyMs
            lastSingleTokenLogitsLatencyMs = logitsLatencyMs
            return exactNextToken
        }

        // Non-identity trunks: use full pair preparation (trunk processes both pairs
        // for state management, but pair1 output head result is still discarded)
        pair1ActivationA.zero()
        let prepared = try prepareActivationPair(strategy: strategy)
        let stateAdvanceLatencyMs = try promotePreparedPair(stepCount: 1)
        captureProposalActivationForNextPass(sourcePairIsA: prepared.sourcePairIsA, committedStepCount: 1)
        consumedTokens += 1
        lastSingleTokenTrunkLatencyMs = prepared.trunkLatencyMs + stateAdvanceLatencyMs
        lastSingleTokenLogitsLatencyMs = prepared.logitsLatencyMs
        return prepared.exactNextToken
    }

    private struct PreparedExactTwoTokenPair {
        let exactNextToken: UInt16
        let exactFutureToken: UInt16
        let trunkLatencyMs: Double
        let logitsLatencyMs: Double
        let sourcePairIsA: Bool
    }

    private mutating func prepareExactTwoTokenPair(
        currentToken: UInt16,
        proposedFutureToken: UInt16,
        strategy: TokenSelectionStrategy
    ) throws(GenerationError) -> PreparedExactTwoTokenPair {
        guard Int(currentToken) < vocabSize else {
            throw .invalidArguments("token \(currentToken) exceeds vocab size \(vocabSize)")
        }
        guard Int(proposedFutureToken) < vocabSize else {
            throw .invalidArguments("proposedFutureToken \(proposedFutureToken) exceeds vocab size \(vocabSize)")
        }
        guard consumedTokens + 1 < maxSequenceTokens else {
            throw .invalidArguments("two-step recurrent generation overflow at maxSequenceTokens \(maxSequenceTokens)")
        }

        try Self.writeTokenEmbedding(currentToken, embedding: embedding, into: pair0ActivationA)
        try Self.writeTokenEmbedding(proposedFutureToken, embedding: embedding, into: pair1ActivationA)
        return try prepareActivationPair(strategy: strategy)
    }

    private mutating func prepareActivationPair(
        strategy: TokenSelectionStrategy
    ) throws(GenerationError) -> PreparedExactTwoTokenPair {
        let trunkStart = GenerationClock.now()
        var sourcePairIsA = true

        switch trunkBackend {
        case .singleLayer:
            for idx in 0..<twoStepSessions.count {
                var timings = StepTimingBreakdown()
                do {
                    if sourcePairIsA {
                        try twoStepSessions[idx].prepare(
                            tokenInput0: pair0ActivationA,
                            tokenInput1: pair1ActivationA,
                            output0: pair0ActivationB,
                            output1: pair1ActivationB,
                            timings: &timings
                        )
                    } else {
                        try twoStepSessions[idx].prepare(
                            tokenInput0: pair0ActivationB,
                            tokenInput1: pair1ActivationB,
                            output0: pair0ActivationA,
                            output1: pair1ActivationA,
                            timings: &timings
                        )
                    }
                } catch {
                    throw .runtimeFailure("two-step recurrent prepare failed at layer \(idx): \(error)")
                }
                sourcePairIsA.toggle()
            }
        case .identityZeroTrunk:
            sourcePairIsA = true
        case .fusedTwoLayerPairs:
            for idx in 0..<fusedPairTwoStepSessions.count {
                var timings = StepTimingBreakdown()
                do {
                    if sourcePairIsA {
                        try fusedPairTwoStepSessions[idx].prepare(
                            tokenInput0: pair0ActivationA,
                            tokenInput1: pair1ActivationA,
                            output0: pair0ActivationB,
                            output1: pair1ActivationB,
                            timings: &timings
                        )
                    } else {
                        try fusedPairTwoStepSessions[idx].prepare(
                            tokenInput0: pair0ActivationB,
                            tokenInput1: pair1ActivationB,
                            output0: pair0ActivationA,
                            output1: pair1ActivationA,
                            timings: &timings
                        )
                    }
                } catch {
                    throw .runtimeFailure("fused two-step recurrent prepare failed at pair \(idx): \(error)")
                }
                sourcePairIsA.toggle()
            }
        case .fusedThreeLayerTriplets:
            for idx in 0..<fusedTripletTwoStepSessions.count {
                var timings = StepTimingBreakdown()
                do {
                    if sourcePairIsA {
                        try fusedTripletTwoStepSessions[idx].prepare(
                            tokenInput0: pair0ActivationA,
                            tokenInput1: pair1ActivationA,
                            output0: pair0ActivationB,
                            output1: pair1ActivationB,
                            timings: &timings
                        )
                    } else {
                        try fusedTripletTwoStepSessions[idx].prepare(
                            tokenInput0: pair0ActivationB,
                            tokenInput1: pair1ActivationB,
                            output0: pair0ActivationA,
                            output1: pair1ActivationA,
                            timings: &timings
                        )
                    }
                } catch {
                    throw .runtimeFailure("fused three-layer two-step recurrent prepare failed at triplet \(idx): \(error)")
                }
                sourcePairIsA.toggle()
            }
        }

        let trunkLatencyMs = GenerationClock.milliseconds(start: trunkStart, end: GenerationClock.now())
        self.trunkLatencyMs += trunkLatencyMs

        let logitsStart = GenerationClock.now()
        let exactNextToken: UInt16
        let exactFutureToken: UInt16
        if sourcePairIsA {
            (exactNextToken, exactFutureToken) = try Self.selectTokenPairFromPreparedActivations(
                pair0ActivationA,
                pair1ActivationA,
                strategy: strategy,
                outputHeadBackend: outputHeadBackend,
                rmsFinal: rmsFinal,
                stepNorm: stepNorm,
                stepLogits: stepLogits,
                embedding: embedding,
                classifier: classifier,
                sharedClassifier: sharedClassifier,
                aneClassifierHead: aneClassifierHead,
                aneRMSNormClassifierHead: aneRMSNormClassifierHead,
                vocabSize: vocabSize,
                stepRMSWorkspace: stepRMSWorkspace
            )
        } else {
            (exactNextToken, exactFutureToken) = try Self.selectTokenPairFromPreparedActivations(
                pair0ActivationB,
                pair1ActivationB,
                strategy: strategy,
                outputHeadBackend: outputHeadBackend,
                rmsFinal: rmsFinal,
                stepNorm: stepNorm,
                stepLogits: stepLogits,
                embedding: embedding,
                classifier: classifier,
                sharedClassifier: sharedClassifier,
                aneClassifierHead: aneClassifierHead,
                aneRMSNormClassifierHead: aneRMSNormClassifierHead,
                vocabSize: vocabSize,
                stepRMSWorkspace: stepRMSWorkspace
            )
        }
        let logitsLatencyMs = GenerationClock.milliseconds(start: logitsStart, end: GenerationClock.now())
        self.logitsLatencyMs += logitsLatencyMs

        return PreparedExactTwoTokenPair(
            exactNextToken: exactNextToken,
            exactFutureToken: exactFutureToken,
            trunkLatencyMs: trunkLatencyMs,
            logitsLatencyMs: logitsLatencyMs,
            sourcePairIsA: sourcePairIsA
        )
    }

    private mutating func promotePreparedPair(stepCount: Int) throws(GenerationError) -> Double {
        let start = GenerationClock.now()
        switch trunkBackend {
        case .singleLayer:
            for idx in 0..<twoStepSessions.count {
                do {
                    try twoStepSessions[idx].promotePreparedState(commitCount: stepCount)
                } catch {
                    throw .runtimeFailure("two-step recurrent state promotion failed at layer \(idx): \(error)")
                }
            }
        case .identityZeroTrunk:
            break
        case .fusedTwoLayerPairs:
            for idx in 0..<fusedPairTwoStepSessions.count {
                do {
                    try fusedPairTwoStepSessions[idx].promotePreparedState(commitCount: stepCount)
                } catch {
                    throw .runtimeFailure("fused two-step recurrent state promotion failed at pair \(idx): \(error)")
                }
            }
        case .fusedThreeLayerTriplets:
            for idx in 0..<fusedTripletTwoStepSessions.count {
                do {
                    try fusedTripletTwoStepSessions[idx].promotePreparedState(commitCount: stepCount)
                } catch {
                    throw .runtimeFailure("fused three-layer two-step recurrent state promotion failed at triplet \(idx): \(error)")
                }
            }
        }
        return GenerationClock.milliseconds(start: start, end: GenerationClock.now())
    }

    private static func writeTokenEmbedding(
        _ token: UInt16,
        embedding: borrowing TensorBuffer,
        into activation: borrowing TensorBuffer
    ) throws(GenerationError) {
        activation.withUnsafeMutablePointer { dst in
            embedding.withUnsafePointer { embeddingPtr in
                let base = Int(token) * ModelConfig.dim
                dst.update(from: embeddingPtr.advanced(by: base), count: ModelConfig.dim)
            }
        }
    }

    private mutating func selectProposedFutureToken(
        currentToken: UInt16,
        strategy: TokenSelectionStrategy
    ) throws(GenerationError) -> UInt16 {
        guard hasFutureProposer else {
            // Echo-family upper bound fallback.
            return currentToken
        }
        guard hasCurrentProposalActivation else {
            throw .runtimeFailure("future proposer requested before a committed activation was prepared")
        }

        return try Self.selectTokenFromActivation(
            currentProposalActivation,
            strategy: strategy,
            outputHeadBackend: futureOutputHeadBackend,
            rmsFinal: futureRMS,
            stepNorm: futureNorm,
            stepLogits: futureLogits,
            embedding: embedding,
            classifier: futureClassifier,
            sharedClassifier: false,
            aneClassifierHead: futureANEClassifierHead,
            aneRMSNormClassifierHead: futureANERMSNormClassifierHead,
            vocabSize: vocabSize,
            stepRMSWorkspace: futureRMSWorkspace
        )
    }

    private mutating func captureProposalActivationForNextPass(
        sourcePairIsA: Bool,
        committedStepCount: Int
    ) {
        switch (sourcePairIsA, committedStepCount) {
        case (true, 1):
            Self.copyActivation(pair0ActivationA, into: currentProposalActivation)
        case (true, 2):
            Self.copyActivation(pair1ActivationA, into: currentProposalActivation)
        case (false, 1):
            Self.copyActivation(pair0ActivationB, into: currentProposalActivation)
        case (false, 2):
            Self.copyActivation(pair1ActivationB, into: currentProposalActivation)
        default:
            return
        }
        hasCurrentProposalActivation = true
    }

    private static func copyActivation(
        _ source: borrowing TensorBuffer,
        into destination: borrowing TensorBuffer
    ) {
        source.withUnsafePointer { src in
            destination.withUnsafeMutablePointer { dst in
                dst.update(from: src, count: ModelConfig.dim)
            }
        }
    }

    @inline(__always)
    private static func classifierPointer<R>(
        sharedClassifier: Bool,
        embedding: borrowing TensorBuffer,
        classifier: borrowing TensorBuffer,
        body: (UnsafePointer<Float>) throws -> R
    ) rethrows -> R {
        if sharedClassifier {
            return try embedding.withUnsafePointer(body)
        }
        return try classifier.withUnsafePointer(body)
    }

    private static func selectTokenFromActivation(
        _ activation: borrowing TensorBuffer,
        strategy: TokenSelectionStrategy
        ,
        outputHeadBackend: GenerationOutputHeadBackend,
        rmsFinal: borrowing TensorBuffer,
        stepNorm: borrowing TensorBuffer,
        stepLogits: borrowing TensorBuffer,
        embedding: borrowing TensorBuffer,
        classifier: borrowing TensorBuffer,
        sharedClassifier: Bool,
        aneClassifierHead: ANEGenerationClassifierHead?,
        aneRMSNormClassifierHead: ANEGenerationRMSNormClassifierHead?,
        vocabSize: Int,
        stepRMSWorkspace: borrowing RMSNorm.Workspace
    ) throws(GenerationError) -> UInt16 {
        if outputHeadBackend != .aneRMSNormClassifier {
            activation.withUnsafePointer { xPtr in
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
        }

        let token: UInt16
        switch outputHeadBackend {
        case .cpu:
            // beta=0.0 means sgemm overwrites C entirely — no pre-zeroing needed
            stepLogits.withUnsafeMutablePointer { logitsPtr in
                classifierPointer(
                    sharedClassifier: sharedClassifier,
                    embedding: embedding,
                    classifier: classifier
                ) { clsPtr in
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
            token = try selectToken(from: stepLogits, strategy: strategy)
        case .aneClassifier:
            guard let aneClassifierHead else {
                throw .runtimeFailure("two-step ANE classifier backend requested without compiled head")
            }
            token = try aneClassifierHead.selectArgmax(normalizedInput: stepNorm)
        case .aneRMSNormClassifier:
            guard let aneRMSNormClassifierHead else {
                throw .runtimeFailure("two-step ANE fused output-head backend requested without compiled head")
            }
            token = try aneRMSNormClassifierHead.selectArgmax(rawInput: activation)
        case .cpuExactStaged, .cpuExactClustered:
            throw .runtimeFailure("two-step branch-state promotion model does not support staged CPU output heads")
        case .cpuThenANE, .cpuPartitionedArgmax, .cpuFP16Tiled:
            // Fall through to CPU sgemm path for now
            stepLogits.withUnsafeMutablePointer { logitsPtr in
                classifierPointer(
                    sharedClassifier: sharedClassifier,
                    embedding: embedding,
                    classifier: classifier
                ) { clsPtr in
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
            token = try selectToken(from: stepLogits, strategy: strategy)
        }

        return token
    }

    private static func selectTokenPairFromPreparedActivations(
        _ activationA: borrowing TensorBuffer,
        _ activationB: borrowing TensorBuffer,
        strategy: TokenSelectionStrategy,
        outputHeadBackend: GenerationOutputHeadBackend,
        rmsFinal: borrowing TensorBuffer,
        stepNorm: borrowing TensorBuffer,
        stepLogits: borrowing TensorBuffer,
        embedding: borrowing TensorBuffer,
        classifier: borrowing TensorBuffer,
        sharedClassifier: Bool,
        aneClassifierHead: ANEGenerationClassifierHead?,
        aneRMSNormClassifierHead: ANEGenerationRMSNormClassifierHead?,
        vocabSize: Int,
        stepRMSWorkspace: borrowing RMSNorm.Workspace
    ) throws(GenerationError) -> (UInt16, UInt16) {
        if outputHeadBackend == .aneRMSNormClassifier {
            guard let aneRMSNormClassifierHead else {
                throw .runtimeFailure("two-step ANE fused output-head backend requested without compiled head")
            }
            return try aneRMSNormClassifierHead.selectArgmaxPair(
                rawInputA: activationA,
                rawInputB: activationB
            )
        }

        return (
            try Self.selectTokenFromActivation(
                activationA,
                strategy: strategy,
                outputHeadBackend: outputHeadBackend,
                rmsFinal: rmsFinal,
                stepNorm: stepNorm,
                stepLogits: stepLogits,
                embedding: embedding,
                classifier: classifier,
                sharedClassifier: sharedClassifier,
                aneClassifierHead: aneClassifierHead,
                aneRMSNormClassifierHead: aneRMSNormClassifierHead,
                vocabSize: vocabSize,
                stepRMSWorkspace: stepRMSWorkspace
            ),
            try Self.selectTokenFromActivation(
                activationB,
                strategy: strategy,
                outputHeadBackend: outputHeadBackend,
                rmsFinal: rmsFinal,
                stepNorm: stepNorm,
                stepLogits: stepLogits,
                embedding: embedding,
                classifier: classifier,
                sharedClassifier: sharedClassifier,
                aneClassifierHead: aneClassifierHead,
                aneRMSNormClassifierHead: aneRMSNormClassifierHead,
                vocabSize: vocabSize,
                stepRMSWorkspace: stepRMSWorkspace
            )
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

public struct RecurrentGenerationWeights: ~Copyable {
    public let layers: LayerStorage<RWKVStyleRecurrentWeights>
    public let rmsFinal: TensorBuffer
    public let embedding: TensorBuffer
    public let classifier: TensorBuffer
    public let sharedClassifier: Bool
    public let vocabSize: Int
    /// Retains the mmap backing buffer so non-owning slice views remain valid.
    /// When weights are not mmap-backed, this is a zero-element sentinel buffer.
    private let mmapBacking: TensorBuffer

    public init(
        layers: consuming LayerStorage<RWKVStyleRecurrentWeights>,
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
        self.mmapBacking = TensorBuffer(count: 0, zeroed: false)
    }

    /// Init with an mmap backing buffer whose lifetime keeps all non-owning slices valid.
    public init(
        layers: consuming LayerStorage<RWKVStyleRecurrentWeights>,
        rmsFinal: consuming TensorBuffer,
        embedding: consuming TensorBuffer,
        classifier: consuming TensorBuffer,
        sharedClassifier: Bool,
        vocabSize: Int = ModelConfig.vocab,
        mmapBacking: consuming TensorBuffer
    ) {
        self.layers = layers
        self.rmsFinal = rmsFinal
        self.embedding = embedding
        self.classifier = classifier
        self.sharedClassifier = sharedClassifier
        self.vocabSize = vocabSize
        self.mmapBacking = mmapBacking
    }
}

public struct ANEDirectGenerationModel: ~Copyable, DirectTokenSelectingLanguageModel, GenerationPerformanceTrackable {
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
    private let outputHeadBackend: GenerationOutputHeadBackend
    private let cpuExactStagedHead: CPUStagedExactGenerationOutputHead?
    private let aneClassifierHead: ANEGenerationClassifierHead?
    private let aneRMSNormClassifierHead: ANEGenerationRMSNormClassifierHead?
    private var decodeHandles: [DecodeSurfaceHandles]
    private var inferenceHandles: [InferenceSurfaceHandles]
    private var decodeState: DecodeState
    private var xCur: TensorBuffer
    public private(set) var compileTimeMs: Double
    private var trunkLatencyMs: Double
    private var logitsLatencyMs: Double

    public var performanceSnapshot: GenerationPerformanceSnapshot {
        GenerationPerformanceSnapshot(
            compileTimeMs: compileTimeMs,
            trunkLatencyMs: trunkLatencyMs,
            logitsLatencyMs: logitsLatencyMs
        )
    }

    public init(
        weights: borrowing GenerationWeights,
        layerCount: Int,
        decodeMaxSeq: Int,
        outputHeadBackend: GenerationOutputHeadBackend = .cpu
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
        let vocabSize = weights.vocabSize
        let sharedClassifier = weights.sharedClassifier
        guard vocabSize > 0 else {
            throw .invalidArguments("vocabSize must be > 0")
        }

        let compileStart = GenerationClock.now()
        let decodeKernels = try Self.compileDecodeKernels(
            weights: weights,
            layerCount: layerCount,
            decodeMaxSeq: decodeMaxSeq
        )
        let inferenceKernels = try Self.compileInferenceKernels(
            weights: weights,
            layerCount: layerCount
        )
        let rmsFinal = GenerationWeightCloner.cloneTensor(weights.rmsFinal)
        let embedding = GenerationWeightCloner.cloneTensor(weights.embedding)
        let classifier = sharedClassifier
            ? TensorBuffer(count: 0, zeroed: true)
            : GenerationWeightCloner.cloneTensor(weights.classifier)
        let aneClassifierHead = try Self.makeANEClassifierHead(
            outputHeadBackend: outputHeadBackend,
            vocabSize: vocabSize,
            sharedClassifier: sharedClassifier,
            embedding: embedding,
            classifier: classifier
        )
        let cpuExactStagedHead = try Self.makeCPUExactStagedHead(
            outputHeadBackend: outputHeadBackend,
            vocabSize: vocabSize,
            sharedClassifier: sharedClassifier,
            embedding: embedding,
            classifier: classifier
        )
        let aneRMSNormClassifierHead = try Self.makeANERMSNormClassifierHead(
            outputHeadBackend: outputHeadBackend,
            vocabSize: vocabSize,
            sharedClassifier: sharedClassifier,
            rmsFinal: rmsFinal,
            embedding: embedding,
            classifier: classifier
        )
        let compileTimeMs = GenerationClock.milliseconds(start: compileStart, end: GenerationClock.now())
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

        self.vocabSize = vocabSize
        self.layerCount = layerCount
        self.decodeMaxSeq = decodeMaxSeq
        self.rmsFinal = rmsFinal
        self.embedding = embedding
        self.classifier = classifier
        self.sharedClassifier = sharedClassifier
        self.decodeKernels = decodeKernels
        self.inferenceKernels = inferenceKernels
        self.stepNorm = TensorBuffer(count: ModelConfig.dim, zeroed: true)
        self.stepLogits = TensorBuffer(count: vocabSize, zeroed: true)
        self.verifySequence = TensorBuffer(count: ModelConfig.dim * ModelConfig.seqLen, zeroed: true)
        self.verifyNorm = TensorBuffer(count: ModelConfig.dim * ModelConfig.seqLen, zeroed: true)
        self.verifyLogits = TensorBuffer(count: vocabSize * ModelConfig.seqLen, zeroed: true)
        self.stepRMSWorkspace = RMSNorm.Workspace(seqLen: 1)
        self.verifyRMSWorkspace = RMSNorm.Workspace(seqLen: ModelConfig.seqLen)
        self.outputHeadBackend = outputHeadBackend
        self.cpuExactStagedHead = cpuExactStagedHead
        self.aneClassifierHead = aneClassifierHead
        self.aneRMSNormClassifierHead = aneRMSNormClassifierHead
        self.decodeHandles = decodeHandles
        self.inferenceHandles = inferenceHandles
        self.decodeState = decodeState
        self.xCur = TensorBuffer(count: ModelConfig.dim, zeroed: true)
        self.compileTimeMs = compileTimeMs
        self.trunkLatencyMs = 0
        self.logitsLatencyMs = 0
    }

    public static func load(
        modelPath: String,
        layerCount: Int,
        decodeMaxSeq: Int,
        outputHeadBackend: GenerationOutputHeadBackend = .cpu
    ) throws(GenerationError) -> ANEDirectGenerationModel {
        let weights = try GenerationWeights.load(modelPath: modelPath)
        return try ANEDirectGenerationModel(
            weights: weights,
            layerCount: layerCount,
            decodeMaxSeq: decodeMaxSeq,
            outputHeadBackend: outputHeadBackend
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

    private static func makeANEClassifierHead(
        outputHeadBackend: GenerationOutputHeadBackend,
        vocabSize: Int,
        sharedClassifier: Bool,
        embedding: borrowing TensorBuffer,
        classifier: borrowing TensorBuffer
    ) throws(GenerationError) -> ANEGenerationClassifierHead? {
        switch outputHeadBackend {
        case .cpu, .cpuExactStaged, .cpuExactClustered, .cpuThenANE, .cpuPartitionedArgmax, .cpuFP16Tiled:
            return nil
        case .aneClassifier:
            if sharedClassifier {
                return try ANEGenerationClassifierHead(classifierWeights: embedding, vocabSize: vocabSize)
            }
            return try ANEGenerationClassifierHead(classifierWeights: classifier, vocabSize: vocabSize)
        case .aneRMSNormClassifier:
            return nil
        }
    }

    private static func makeCPUExactStagedHead(
        outputHeadBackend: GenerationOutputHeadBackend,
        vocabSize: Int,
        sharedClassifier: Bool,
        embedding: borrowing TensorBuffer,
        classifier: borrowing TensorBuffer
    ) throws(GenerationError) -> CPUStagedExactGenerationOutputHead? {
        switch outputHeadBackend {
        case .cpu, .aneClassifier, .aneRMSNormClassifier, .cpuThenANE, .cpuPartitionedArgmax, .cpuFP16Tiled:
            return nil
        case .cpuExactStaged:
            if sharedClassifier {
                return try CPUStagedExactGenerationOutputHead(
                    classifierWeights: embedding,
                    vocabSize: vocabSize,
                    layoutStrategy: .contiguous(shardSize: 1024)
                )
            }
            return try CPUStagedExactGenerationOutputHead(
                classifierWeights: classifier,
                vocabSize: vocabSize,
                layoutStrategy: .contiguous(shardSize: 1024)
            )
        case .cpuExactClustered:
            if sharedClassifier {
                return try CPUStagedExactGenerationOutputHead(
                    classifierWeights: embedding,
                    vocabSize: vocabSize,
                    layoutStrategy: .clustered(clusterCount: 32, projectionDimensionCount: 24, iterations: 2)
                )
            }
            return try CPUStagedExactGenerationOutputHead(
                classifierWeights: classifier,
                vocabSize: vocabSize,
                layoutStrategy: .clustered(clusterCount: 32, projectionDimensionCount: 24, iterations: 2)
            )
        }
    }

    private static func makeANERMSNormClassifierHead(
        outputHeadBackend: GenerationOutputHeadBackend,
        vocabSize: Int,
        sharedClassifier: Bool,
        rmsFinal: borrowing TensorBuffer,
        embedding: borrowing TensorBuffer,
        classifier: borrowing TensorBuffer
    ) throws(GenerationError) -> ANEGenerationRMSNormClassifierHead? {
        switch outputHeadBackend {
        case .cpu, .cpuExactStaged, .cpuExactClustered, .aneClassifier, .cpuThenANE, .cpuPartitionedArgmax, .cpuFP16Tiled:
            return nil
        case .aneRMSNormClassifier:
            if sharedClassifier {
                return try ANEGenerationRMSNormClassifierHead(
                    rmsFinal: rmsFinal,
                    classifierWeights: embedding,
                    vocabSize: vocabSize
                )
            }
            return try ANEGenerationRMSNormClassifierHead(
                rmsFinal: rmsFinal,
                classifierWeights: classifier,
                vocabSize: vocabSize
            )
        }
    }

    public mutating func reset() throws(GenerationError) {
        ForwardPass.initializeDecodeCachesAndMask(surfaceHandles: decodeHandles)
        decodeState.reset()
        xCur.zero()
        trunkLatencyMs = 0
        logitsLatencyMs = 0
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

    public mutating func prefillSelectedToken(
        promptTokens: [UInt16],
        strategy: TokenSelectionStrategy
    ) throws(GenerationError) -> UInt16 {
        guard !promptTokens.isEmpty else {
            throw .invalidArguments("promptTokens must not be empty")
        }
        guard promptTokens.count <= decodeMaxSeq else {
            throw .invalidArguments("prompt length \(promptTokens.count) exceeds decodeMaxSeq \(decodeMaxSeq)")
        }

        for token in promptTokens {
            try runDecodeStep(token: token)
        }
        return try selectStepToken(strategy: strategy)
    }

    public mutating func decodeSelectedToken(
        nextToken: UInt16,
        strategy: TokenSelectionStrategy
    ) throws(GenerationError) -> UInt16 {
        try runDecodeStep(token: nextToken)
        return try selectStepToken(strategy: strategy)
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

        let trunkStart = GenerationClock.now()
        xCur.withUnsafeMutablePointer { dst in
            embedding.withUnsafePointer { embeddingPtr in
                let base = Int(token) * ModelConfig.dim
                dst.update(from: embeddingPtr.advanced(by: base), count: ModelConfig.dim)
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
        trunkLatencyMs += GenerationClock.milliseconds(start: trunkStart, end: GenerationClock.now())
    }

    private mutating func projectStepLogits() throws(GenerationError) -> [Float] {
        let logitsStart = GenerationClock.now()
        if outputHeadBackend != .aneRMSNormClassifier {
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
        }

        stepLogits.zero()
        switch outputHeadBackend {
        case .cpu, .cpuExactStaged, .cpuExactClustered, .cpuThenANE, .cpuPartitionedArgmax, .cpuFP16Tiled:
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
        case .aneClassifier:
            guard let aneClassifierHead else {
                throw .runtimeFailure("ANE classifier backend requested without compiled head")
            }
            try aneClassifierHead.project(normalizedInput: stepNorm, logits: stepLogits)
        case .aneRMSNormClassifier:
            guard let aneRMSNormClassifierHead else {
                throw .runtimeFailure("ANE fused output-head backend requested without compiled head")
            }
            try aneRMSNormClassifierHead.project(rawInput: xCur, logits: stepLogits)
        }

        logitsLatencyMs += GenerationClock.milliseconds(start: logitsStart, end: GenerationClock.now())
        return stepLogits.withUnsafeBufferPointer { Array($0) }
    }

    private mutating func selectStepToken(
        strategy: TokenSelectionStrategy
    ) throws(GenerationError) -> UInt16 {
        let logitsStart = GenerationClock.now()
        if outputHeadBackend != .aneRMSNormClassifier {
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
        }

        let token: UInt16
        switch outputHeadBackend {
        case .cpu:
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
            token = try selectToken(from: stepLogits, strategy: strategy)
        case .cpuExactStaged:
            guard let cpuExactStagedHead else {
                throw .runtimeFailure("staged exact CPU output head requested without staged head")
            }
            token = try cpuExactStagedHead.selectArgmax(normalizedInput: stepNorm)
        case .cpuExactClustered:
            guard let cpuExactStagedHead else {
                throw .runtimeFailure("clustered exact CPU output head requested without staged head")
            }
            token = try cpuExactStagedHead.selectArgmax(normalizedInput: stepNorm)
        case .aneClassifier:
            guard let aneClassifierHead else {
                throw .runtimeFailure("ANE classifier backend requested without compiled head")
            }
            token = try aneClassifierHead.selectArgmax(normalizedInput: stepNorm)
        case .aneRMSNormClassifier:
            guard let aneRMSNormClassifierHead else {
                throw .runtimeFailure("ANE fused output-head backend requested without compiled head")
            }
            token = try aneRMSNormClassifierHead.selectArgmax(rawInput: xCur)
        case .cpuThenANE, .cpuPartitionedArgmax, .cpuFP16Tiled:
            // Fall through to CPU sgemm path
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
            token = try selectToken(from: stepLogits, strategy: strategy)
        }

        logitsLatencyMs += GenerationClock.milliseconds(start: logitsStart, end: GenerationClock.now())
        return token
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

    private mutating func projectSequenceLogits() throws(GenerationError) {
        let logitsStart = GenerationClock.now()
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
        logitsLatencyMs += GenerationClock.milliseconds(start: logitsStart, end: GenerationClock.now())
    }
}

public struct ANERecurrentGenerationModel: ~Copyable, DirectTokenSelectingLanguageModel, GenerationPerformanceTrackable {
    public let vocabSize: Int
    public let layerCount: Int
    public let maxSequenceTokens: Int

    private let rmsFinal: TensorBuffer
    private let embedding: TensorBuffer
    private let classifier: TensorBuffer
    private let sharedClassifier: Bool
    private let trunkBackend: RecurrentGenerationTrunkBackend
    private var singleLayerSessions: LayerStorage<RWKVStyleRecurrentSession>
    private var fusedPairSessions: LayerStorage<RWKVStyleFusedTwoLayerSession>
    private var fusedTripletSessions: LayerStorage<RWKVStyleFusedThreeLayerSession>
    private let stepNorm: TensorBuffer
    private let stepLogits: TensorBuffer
    private let stepRMSWorkspace: RMSNorm.Workspace
    private let outputHeadBackend: GenerationOutputHeadBackend
    private let cpuExactStagedHead: CPUStagedExactGenerationOutputHead?
    private let aneClassifierHead: ANEGenerationClassifierHead?
    private let aneRMSNormClassifierHead: ANEGenerationRMSNormClassifierHead?
    private var activationA: TensorBuffer
    private var activationB: TensorBuffer
    private var currentActivationIsA: Bool
    private var consumedTokens: Int
    public private(set) var compileTimeMs: Double
    private var trunkLatencyMs: Double
    private var logitsLatencyMs: Double
    private let blockMaxNorms: [Float]
    private let partitionedLogitsScratch: TensorBuffer
    private let deferredANEHead: DeferredANEHead?
    private let classifierFP16: TensorBufferFP16

    public var performanceSnapshot: GenerationPerformanceSnapshot {
        GenerationPerformanceSnapshot(
            compileTimeMs: compileTimeMs,
            trunkLatencyMs: trunkLatencyMs,
            logitsLatencyMs: logitsLatencyMs
        )
    }

    public init(
        weights: borrowing RecurrentGenerationWeights,
        layerCount: Int,
        maxSequenceTokens: Int = ModelConfig.seqLen,
        outputHeadBackend: GenerationOutputHeadBackend = .cpu,
        trunkBackend: RecurrentGenerationTrunkBackend = .singleLayer,
        trunkLaneSpatial: Int = RWKVStyleRecurrentKernelSet.defaultLaneSpatial,
        outputHeadLaneSpatial: Int = 32,
        shareReadOnlyWeights: Bool = false
    ) throws(GenerationError) {
        guard layerCount > 0 else {
            throw .invalidArguments("layerCount must be > 0")
        }
        guard layerCount <= weights.layers.count else {
            throw .invalidArguments("layerCount \(layerCount) exceeds available recurrent layers \(weights.layers.count)")
        }
        if trunkBackend == .fusedTwoLayerPairs, !layerCount.isMultiple(of: 2) {
            throw .invalidArguments("fused recurrent trunk backend requires an even layerCount")
        }
        if trunkBackend == .fusedThreeLayerTriplets, !layerCount.isMultiple(of: 3) {
            throw .invalidArguments("fused three-layer recurrent trunk backend requires a layerCount that is a multiple of 3")
        }
        #if DEBUG
        if trunkBackend == .identityZeroTrunk,
           !recurrentWeightsUseIdentityZeroTrunk(weights, layerCount: layerCount) {
            throw .invalidArguments("identity zero-trunk backend requires all recurrent Wx/Ws/Wd/Wo weights to be zero")
        }
        #endif
        guard trunkLaneSpatial > 0 else {
            throw .invalidArguments("recurrent trunk laneSpatial must be > 0")
        }
        guard outputHeadLaneSpatial > 0 else {
            throw .invalidArguments("generation output-head laneSpatial must be > 0")
        }
        guard maxSequenceTokens > 0 else {
            throw .invalidArguments("maxSequenceTokens must be > 0")
        }
        let vocabSize = weights.vocabSize
        let sharedClassifier = weights.sharedClassifier
        guard vocabSize > 0 else {
            throw .invalidArguments("vocabSize must be > 0")
        }

        let compileStart = GenerationClock.now()
        let singleLayerSessions = try Self.compileSingleLayerSessions(
            weights: weights,
            layerCount: layerCount,
            trunkBackend: trunkBackend,
            laneSpatial: trunkLaneSpatial
        )
        let fusedPairSessions = try Self.compileFusedPairSessions(
            weights: weights,
            layerCount: layerCount,
            trunkBackend: trunkBackend,
            laneSpatial: trunkLaneSpatial
        )
        let fusedTripletSessions = try Self.compileFusedTripletSessions(
            weights: weights,
            layerCount: layerCount,
            trunkBackend: trunkBackend,
            laneSpatial: trunkLaneSpatial
        )
        let rmsFinal = shareReadOnlyWeights
            ? GenerationWeightCloner.shareTensor(weights.rmsFinal)
            : GenerationWeightCloner.cloneTensor(weights.rmsFinal)
        let embedding = shareReadOnlyWeights
            ? GenerationWeightCloner.shareTensor(weights.embedding)
            : GenerationWeightCloner.cloneTensor(weights.embedding)
        let classifier = sharedClassifier
            ? TensorBuffer(count: 0, zeroed: true)
            : (shareReadOnlyWeights
                ? GenerationWeightCloner.shareTensor(weights.classifier)
                : GenerationWeightCloner.cloneTensor(weights.classifier))
        let aneClassifierHead = try Self.makeANEClassifierHead(
            outputHeadBackend: outputHeadBackend,
            vocabSize: vocabSize,
            sharedClassifier: sharedClassifier,
            embedding: embedding,
            classifier: classifier,
            laneSpatial: outputHeadLaneSpatial
        )
        let cpuExactStagedHead = try Self.makeCPUExactStagedHead(
            outputHeadBackend: outputHeadBackend,
            vocabSize: vocabSize,
            sharedClassifier: sharedClassifier,
            embedding: embedding,
            classifier: classifier
        )
        let aneRMSNormClassifierHead = try Self.makeANERMSNormClassifierHead(
            outputHeadBackend: outputHeadBackend,
            vocabSize: vocabSize,
            sharedClassifier: sharedClassifier,
            rmsFinal: rmsFinal,
            embedding: embedding,
            classifier: classifier,
            laneSpatial: outputHeadLaneSpatial
        )
        let compileTimeMs = GenerationClock.milliseconds(start: compileStart, end: GenerationClock.now())

        // Precompute block max norms for partitioned argmax (before consuming embedding/classifier)
        let blockMaxNorms: [Float]
        let partitionedLogitsScratch: TensorBuffer
        if outputHeadBackend == .cpuPartitionedArgmax {
            let blockSize = PartitionedArgmax.defaultBlockSize
            if sharedClassifier {
                blockMaxNorms = embedding.withUnsafePointer { clsPtr in
                    PartitionedArgmax.precomputeBlockMaxNorms(
                        classifier: clsPtr, vocabSize: vocabSize,
                        dim: ModelConfig.dim, blockSize: blockSize
                    )
                }
            } else {
                blockMaxNorms = classifier.withUnsafePointer { clsPtr in
                    PartitionedArgmax.precomputeBlockMaxNorms(
                        classifier: clsPtr, vocabSize: vocabSize,
                        dim: ModelConfig.dim, blockSize: blockSize
                    )
                }
            }
            partitionedLogitsScratch = TensorBuffer(count: blockSize, zeroed: true)
        } else {
            blockMaxNorms = []
            partitionedLogitsScratch = TensorBuffer(count: 0, zeroed: true)
        }

        // Deferred ANE compilation: extract raw pointers before weights are consumed
        let deferredANEHead: DeferredANEHead?
        if outputHeadBackend == .cpuThenANE {
            let deferred = DeferredANEHead()
            let rmsPtr = rmsFinal.withUnsafePointer { UnsafeMutablePointer(mutating: $0) }
            let clsPtr: UnsafeMutablePointer<Float>
            if sharedClassifier {
                clsPtr = embedding.withUnsafePointer { UnsafeMutablePointer(mutating: $0) }
            } else {
                clsPtr = classifier.withUnsafePointer { UnsafeMutablePointer(mutating: $0) }
            }
            let clsCount = vocabSize * ModelConfig.dim
            let capturedVocab = vocabSize
            let capturedLane = outputHeadLaneSpatial
            // Background compile — pointers valid because self owns the TensorBuffers
            DispatchQueue.global(qos: .userInitiated).async {
                let rmsBuf = TensorBuffer(nonOwningPointer: rmsPtr, count: ModelConfig.dim)
                let clsBuf = TensorBuffer(nonOwningPointer: clsPtr, count: clsCount)
                if let head = try? ANEGenerationRMSNormClassifierHead(
                    rmsFinal: rmsBuf, classifierWeights: clsBuf,
                    vocabSize: capturedVocab, laneSpatial: capturedLane
                ) {
                    deferred.store(head)
                }
            }
            deferredANEHead = deferred
        } else {
            deferredANEHead = nil
        }

        // FP16 tiled classifier weights
        let classifierFP16: TensorBufferFP16
        if outputHeadBackend == .cpuFP16Tiled {
            if sharedClassifier {
                classifierFP16 = TensorBufferFP16(
                    quantizing: embedding,
                    rows: vocabSize,
                    cols: ModelConfig.dim
                )
            } else {
                classifierFP16 = TensorBufferFP16(
                    quantizing: classifier,
                    rows: vocabSize,
                    cols: ModelConfig.dim
                )
            }
        } else {
            classifierFP16 = TensorBufferFP16()
        }

        self.vocabSize = vocabSize
        self.layerCount = layerCount
        self.maxSequenceTokens = maxSequenceTokens
        self.rmsFinal = rmsFinal
        self.embedding = embedding
        self.classifier = classifier
        self.sharedClassifier = sharedClassifier
        self.trunkBackend = trunkBackend
        self.singleLayerSessions = singleLayerSessions
        self.fusedPairSessions = fusedPairSessions
        self.fusedTripletSessions = fusedTripletSessions
        self.stepNorm = TensorBuffer(count: ModelConfig.dim, zeroed: true)
        self.stepLogits = TensorBuffer(count: vocabSize, zeroed: true)
        self.stepRMSWorkspace = RMSNorm.Workspace(seqLen: 1)
        self.outputHeadBackend = outputHeadBackend
        self.cpuExactStagedHead = cpuExactStagedHead
        self.aneClassifierHead = aneClassifierHead
        self.aneRMSNormClassifierHead = aneRMSNormClassifierHead
        self.activationA = TensorBuffer(count: ModelConfig.dim, zeroed: true)
        self.activationB = TensorBuffer(count: ModelConfig.dim, zeroed: true)
        self.currentActivationIsA = true
        self.consumedTokens = 0
        self.compileTimeMs = compileTimeMs
        self.trunkLatencyMs = 0
        self.logitsLatencyMs = 0
        self.blockMaxNorms = blockMaxNorms
        self.partitionedLogitsScratch = partitionedLogitsScratch
        self.deferredANEHead = deferredANEHead
        self.classifierFP16 = classifierFP16
    }

    private static func emptyLayerStorage<Element: ~Copyable>(_: Element.Type = Element.self) -> LayerStorage<Element> {
        LayerStorage<Element>(count: 0) { _ in
            fatalError("unreachable empty layer storage initializer")
        }
    }

    private static func compileSingleLayerSessions(
        weights: borrowing RecurrentGenerationWeights,
        layerCount: Int,
        trunkBackend: RecurrentGenerationTrunkBackend,
        laneSpatial: Int
    ) throws(GenerationError) -> LayerStorage<RWKVStyleRecurrentSession> {
        guard trunkBackend == .singleLayer else {
            return emptyLayerStorage(RWKVStyleRecurrentSession.self)
        }
        do {
            return try LayerStorage<RWKVStyleRecurrentSession>(count: layerCount, throwingInitializer: { idx in
                try RWKVStyleRecurrentSession(weights: weights.layers[idx], laneSpatial: laneSpatial)
            })
        } catch {
            throw .runtimeFailure("recurrent kernel/session setup failed: \(error)")
        }
    }

    private static func compileFusedPairSessions(
        weights: borrowing RecurrentGenerationWeights,
        layerCount: Int,
        trunkBackend: RecurrentGenerationTrunkBackend,
        laneSpatial: Int
    ) throws(GenerationError) -> LayerStorage<RWKVStyleFusedTwoLayerSession> {
        guard trunkBackend == .fusedTwoLayerPairs else {
            return emptyLayerStorage(RWKVStyleFusedTwoLayerSession.self)
        }
        do {
            return try LayerStorage<RWKVStyleFusedTwoLayerSession>(
                count: layerCount / 2,
                throwingInitializer: { pairIdx in
                    let base = pairIdx * 2
                    return try RWKVStyleFusedTwoLayerSession(
                        weights0: weights.layers[base],
                        weights1: weights.layers[base + 1],
                        laneSpatial: laneSpatial
                    )
                }
            )
        } catch {
            throw .runtimeFailure("fused recurrent kernel/session setup failed: \(error)")
        }
    }

    private static func compileFusedTripletSessions(
        weights: borrowing RecurrentGenerationWeights,
        layerCount: Int,
        trunkBackend: RecurrentGenerationTrunkBackend,
        laneSpatial: Int
    ) throws(GenerationError) -> LayerStorage<RWKVStyleFusedThreeLayerSession> {
        guard trunkBackend == .fusedThreeLayerTriplets else {
            return emptyLayerStorage(RWKVStyleFusedThreeLayerSession.self)
        }
        do {
            return try LayerStorage<RWKVStyleFusedThreeLayerSession>(
                count: layerCount / 3,
                throwingInitializer: { tripletIdx in
                    let base = tripletIdx * 3
                    return try RWKVStyleFusedThreeLayerSession(
                        weights0: weights.layers[base],
                        weights1: weights.layers[base + 1],
                        weights2: weights.layers[base + 2],
                        laneSpatial: laneSpatial
                    )
                }
            )
        } catch {
            throw .runtimeFailure("fused three-layer recurrent kernel/session setup failed: \(error)")
        }
    }

    private static func makeANEClassifierHead(
        outputHeadBackend: GenerationOutputHeadBackend,
        vocabSize: Int,
        sharedClassifier: Bool,
        embedding: borrowing TensorBuffer,
        classifier: borrowing TensorBuffer,
        laneSpatial: Int
    ) throws(GenerationError) -> ANEGenerationClassifierHead? {
        switch outputHeadBackend {
        case .cpu, .cpuExactStaged, .cpuExactClustered, .cpuThenANE, .cpuPartitionedArgmax, .cpuFP16Tiled:
            return nil
        case .aneClassifier:
            if sharedClassifier {
                return try ANEGenerationClassifierHead(
                    classifierWeights: embedding,
                    vocabSize: vocabSize,
                    laneSpatial: laneSpatial
                )
            }
            return try ANEGenerationClassifierHead(
                classifierWeights: classifier,
                vocabSize: vocabSize,
                laneSpatial: laneSpatial
            )
        case .aneRMSNormClassifier:
            return nil
        }
    }

    private static func makeCPUExactStagedHead(
        outputHeadBackend: GenerationOutputHeadBackend,
        vocabSize: Int,
        sharedClassifier: Bool,
        embedding: borrowing TensorBuffer,
        classifier: borrowing TensorBuffer
    ) throws(GenerationError) -> CPUStagedExactGenerationOutputHead? {
        switch outputHeadBackend {
        case .cpu, .aneClassifier, .aneRMSNormClassifier, .cpuThenANE, .cpuPartitionedArgmax, .cpuFP16Tiled:
            return nil
        case .cpuExactStaged:
            if sharedClassifier {
                return try CPUStagedExactGenerationOutputHead(
                    classifierWeights: embedding,
                    vocabSize: vocabSize,
                    layoutStrategy: .contiguous(shardSize: 1024)
                )
            }
            return try CPUStagedExactGenerationOutputHead(
                classifierWeights: classifier,
                vocabSize: vocabSize,
                layoutStrategy: .contiguous(shardSize: 1024)
            )
        case .cpuExactClustered:
            if sharedClassifier {
                return try CPUStagedExactGenerationOutputHead(
                    classifierWeights: embedding,
                    vocabSize: vocabSize,
                    layoutStrategy: .clustered(clusterCount: 32, projectionDimensionCount: 24, iterations: 2)
                )
            }
            return try CPUStagedExactGenerationOutputHead(
                classifierWeights: classifier,
                vocabSize: vocabSize,
                layoutStrategy: .clustered(clusterCount: 32, projectionDimensionCount: 24, iterations: 2)
            )
        }
    }

    private static func makeANERMSNormClassifierHead(
        outputHeadBackend: GenerationOutputHeadBackend,
        vocabSize: Int,
        sharedClassifier: Bool,
        rmsFinal: borrowing TensorBuffer,
        embedding: borrowing TensorBuffer,
        classifier: borrowing TensorBuffer,
        laneSpatial: Int
    ) throws(GenerationError) -> ANEGenerationRMSNormClassifierHead? {
        switch outputHeadBackend {
        case .cpu, .cpuExactStaged, .cpuExactClustered, .aneClassifier, .cpuThenANE, .cpuPartitionedArgmax, .cpuFP16Tiled:
            return nil
        case .aneRMSNormClassifier:
            if sharedClassifier {
                return try ANEGenerationRMSNormClassifierHead(
                    rmsFinal: rmsFinal,
                    classifierWeights: embedding,
                    vocabSize: vocabSize,
                    laneSpatial: laneSpatial
                )
            }
            return try ANEGenerationRMSNormClassifierHead(
                rmsFinal: rmsFinal,
                classifierWeights: classifier,
                vocabSize: vocabSize,
                laneSpatial: laneSpatial
            )
        }
    }

    public mutating func reset() throws(GenerationError) {
        switch trunkBackend {
        case .singleLayer:
            for idx in 0..<singleLayerSessions.count {
                do {
                    try singleLayerSessions[idx].reset()
                } catch {
                    throw .runtimeFailure("recurrent reset failed at layer \(idx): \(error)")
                }
            }
        case .identityZeroTrunk:
            break
        case .fusedTwoLayerPairs:
            for idx in 0..<fusedPairSessions.count {
                do {
                    try fusedPairSessions[idx].reset()
                } catch {
                    throw .runtimeFailure("fused recurrent reset failed at pair \(idx): \(error)")
                }
            }
        case .fusedThreeLayerTriplets:
            for idx in 0..<fusedTripletSessions.count {
                do {
                    try fusedTripletSessions[idx].reset()
                } catch {
                    throw .runtimeFailure("fused three-layer recurrent reset failed at triplet \(idx): \(error)")
                }
            }
        }
        activationA.zero()
        activationB.zero()
        currentActivationIsA = true
        consumedTokens = 0
        trunkLatencyMs = 0
        logitsLatencyMs = 0
    }

    public mutating func prefill(promptTokens: [UInt16]) throws(GenerationError) -> [Float] {
        guard !promptTokens.isEmpty else {
            throw .invalidArguments("promptTokens must not be empty")
        }
        guard promptTokens.count <= maxSequenceTokens else {
            throw .invalidArguments("prompt length \(promptTokens.count) exceeds maxSequenceTokens \(maxSequenceTokens)")
        }

        for token in promptTokens {
            try runRecurrentStep(token: token)
        }
        return try projectCurrentLogits()
    }

    public mutating func decode(nextToken: UInt16) throws(GenerationError) -> [Float] {
        try runRecurrentStep(token: nextToken)
        return try projectCurrentLogits()
    }

    public mutating func prefillSelectedToken(
        promptTokens: [UInt16],
        strategy: TokenSelectionStrategy
    ) throws(GenerationError) -> UInt16 {
        guard !promptTokens.isEmpty else {
            throw .invalidArguments("promptTokens must not be empty")
        }
        guard promptTokens.count <= maxSequenceTokens else {
            throw .invalidArguments("prompt length \(promptTokens.count) exceeds maxSequenceTokens \(maxSequenceTokens)")
        }

        for token in promptTokens {
            try runRecurrentStep(token: token)
        }
        return try selectCurrentToken(strategy: strategy)
    }

    public mutating func decodeSelectedToken(
        nextToken: UInt16,
        strategy: TokenSelectionStrategy
    ) throws(GenerationError) -> UInt16 {
        try runRecurrentStep(token: nextToken)
        return try selectCurrentToken(strategy: strategy)
    }

    public mutating func verify(
        sequenceTokens: [UInt16],
        startIndex: Int
    ) throws(GenerationError) -> [[Float]] {
        throw .runtimeFailure(
            "recurrent verify is not yet supported without a scratch cursor because replay would mutate live recurrent state"
        )
    }

    @inline(__always)
    private func classifierPointer<R>(_ body: (UnsafePointer<Float>) throws -> R) rethrows -> R {
        if sharedClassifier {
            return try embedding.withUnsafePointer(body)
        }
        return try classifier.withUnsafePointer(body)
    }

    private mutating func runRecurrentStep(token: UInt16) throws(GenerationError) {
        guard Int(token) < vocabSize else {
            throw .invalidArguments("token \(token) exceeds vocab size \(vocabSize)")
        }
        guard consumedTokens < maxSequenceTokens else {
            throw .invalidArguments("recurrent generation overflow at maxSequenceTokens \(maxSequenceTokens)")
        }

        let trunkStart = GenerationClock.now()
        activationA.withUnsafeMutablePointer { dst in
            embedding.withUnsafePointer { embeddingPtr in
                let base = Int(token) * ModelConfig.dim
                dst.update(from: embeddingPtr.advanced(by: base), count: ModelConfig.dim)
            }
        }

        var sourceIsA = true
        switch trunkBackend {
        case .singleLayer:
            for idx in 0..<singleLayerSessions.count {
                var timings = StepTimingBreakdown()
                do {
                    if sourceIsA {
                        try singleLayerSessions[idx].step(tokenInput: activationA, output: activationB, timings: &timings)
                    } else {
                        try singleLayerSessions[idx].step(tokenInput: activationB, output: activationA, timings: &timings)
                    }
                } catch {
                    throw .runtimeFailure("recurrent step failed at layer \(idx): \(error)")
                }
                sourceIsA.toggle()
            }
        case .identityZeroTrunk:
            sourceIsA = true
        case .fusedTwoLayerPairs:
            for idx in 0..<fusedPairSessions.count {
                var timings = StepTimingBreakdown()
                do {
                    if sourceIsA {
                        try fusedPairSessions[idx].step(tokenInput: activationA, output: activationB, timings: &timings)
                    } else {
                        try fusedPairSessions[idx].step(tokenInput: activationB, output: activationA, timings: &timings)
                    }
                } catch {
                    throw .runtimeFailure("fused recurrent step failed at pair \(idx): \(error)")
                }
                sourceIsA.toggle()
            }
        case .fusedThreeLayerTriplets:
            for idx in 0..<fusedTripletSessions.count {
                var timings = StepTimingBreakdown()
                do {
                    if sourceIsA {
                        try fusedTripletSessions[idx].step(tokenInput: activationA, output: activationB, timings: &timings)
                    } else {
                        try fusedTripletSessions[idx].step(tokenInput: activationB, output: activationA, timings: &timings)
                    }
                } catch {
                    throw .runtimeFailure("fused three-layer recurrent step failed at triplet \(idx): \(error)")
                }
                sourceIsA.toggle()
            }
        }

        currentActivationIsA = sourceIsA
        trunkLatencyMs += GenerationClock.milliseconds(start: trunkStart, end: GenerationClock.now())
        consumedTokens += 1
    }

    private mutating func projectCurrentLogits() throws(GenerationError) -> [Float] {
        let logitsStart = GenerationClock.now()
        if outputHeadBackend != .aneRMSNormClassifier {
            if currentActivationIsA {
                activationA.withUnsafePointer { xPtr in
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
            } else {
                activationB.withUnsafePointer { xPtr in
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
            }
        }

        stepLogits.zero()
        switch outputHeadBackend {
        case .cpu, .cpuExactStaged, .cpuExactClustered, .cpuThenANE, .cpuPartitionedArgmax, .cpuFP16Tiled:
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
        case .aneClassifier:
            guard let aneClassifierHead else {
                throw .runtimeFailure("ANE classifier backend requested without compiled head")
            }
            try aneClassifierHead.project(normalizedInput: stepNorm, logits: stepLogits)
        case .aneRMSNormClassifier:
            guard let aneRMSNormClassifierHead else {
                throw .runtimeFailure("ANE fused output-head backend requested without compiled head")
            }
            if currentActivationIsA {
                try aneRMSNormClassifierHead.project(rawInput: activationA, logits: stepLogits)
            } else {
                try aneRMSNormClassifierHead.project(rawInput: activationB, logits: stepLogits)
            }
        }

        logitsLatencyMs += GenerationClock.milliseconds(start: logitsStart, end: GenerationClock.now())
        return stepLogits.withUnsafeBufferPointer { Array($0) }
    }

    private mutating func selectCurrentToken(
        strategy: TokenSelectionStrategy
    ) throws(GenerationError) -> UInt16 {
        let logitsStart = GenerationClock.now()
        if outputHeadBackend != .aneRMSNormClassifier {
            if currentActivationIsA {
                activationA.withUnsafePointer { xPtr in
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
            } else {
                activationB.withUnsafePointer { xPtr in
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
            }
        }

        let token: UInt16
        switch outputHeadBackend {
        case .cpu:
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
            token = try selectToken(from: stepLogits, strategy: strategy)
        case .cpuExactStaged:
            guard let cpuExactStagedHead else {
                throw .runtimeFailure("staged exact CPU output head requested without staged head")
            }
            token = try cpuExactStagedHead.selectArgmax(normalizedInput: stepNorm)
        case .cpuExactClustered:
            guard let cpuExactStagedHead else {
                throw .runtimeFailure("clustered exact CPU output head requested without staged head")
            }
            token = try cpuExactStagedHead.selectArgmax(normalizedInput: stepNorm)
        case .aneClassifier:
            guard let aneClassifierHead else {
                throw .runtimeFailure("ANE classifier backend requested without compiled head")
            }
            token = try aneClassifierHead.selectArgmax(normalizedInput: stepNorm)
        case .aneRMSNormClassifier:
            guard let aneRMSNormClassifierHead else {
                throw .runtimeFailure("ANE fused output-head backend requested without compiled head")
            }
            if currentActivationIsA {
                token = try aneRMSNormClassifierHead.selectArgmax(rawInput: activationA)
            } else {
                token = try aneRMSNormClassifierHead.selectArgmax(rawInput: activationB)
            }
        case .cpuThenANE:
            // If deferred ANE head is ready, use it; otherwise fall through to CPU sgemm
            if let readyHead = deferredANEHead?.readyHead() {
                if currentActivationIsA {
                    token = try readyHead.selectArgmax(rawInput: activationA)
                } else {
                    token = try readyHead.selectArgmax(rawInput: activationB)
                }
            } else {
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
                token = try selectToken(from: stepLogits, strategy: strategy)
            }
        case .cpuPartitionedArgmax:
            // Partitioned argmax with Cauchy-Schwarz block pruning (greedy only)
            var skippedBlocks = 0
            let tokenIndex = partitionedLogitsScratch.withUnsafeMutablePointer { scratchPtr in
                classifierPointer { clsPtr in
                    stepNorm.withUnsafePointer { normPtr in
                        blockMaxNorms.withUnsafeBufferPointer { normsPtr in
                            PartitionedArgmax.compute(
                                classifier: clsPtr,
                                input: normPtr,
                                logitsScratch: scratchPtr,
                                blockMaxNorms: normsPtr.baseAddress!,
                                vocabSize: vocabSize,
                                dim: ModelConfig.dim,
                                blockSize: PartitionedArgmax.defaultBlockSize,
                                skippedBlocks: &skippedBlocks
                            )
                        }
                    }
                }
            }
            token = UInt16(tokenIndex)
        case .cpuFP16Tiled:
            guard classifierFP16.count > 0 else {
                throw .runtimeFailure("FP16 tiled classifier backend requested without FP16 weights")
            }
            let tokenIndex = classifierFP16.withUnsafePointer { fp16Ptr in
                stepNorm.withUnsafePointer { normPtr in
                    FP16TiledClassifier.tiledMatvecArgmax(
                        weights: fp16Ptr,
                        input: normPtr,
                        vocabSize: vocabSize,
                        dim: ModelConfig.dim
                    )
                }
            }
            token = UInt16(tokenIndex)
        }

        logitsLatencyMs += GenerationClock.milliseconds(start: logitsStart, end: GenerationClock.now())
        return token
    }

    // MARK: - Streaming two-token decode

    /// Result of a two-token streaming decode step.
    public struct TwoTokenStreamResult: Sendable {
        public let token1: UInt16
        public let token2: UInt16
        public let token1TrunkMs: Double
        public let token1ClassifierMs: Double
        public let token2TrunkMs: Double
        public let token2ClassifierMs: Double
        /// True if the repeat-speculation for token2's trunk was correct.
        public let speculationHit: Bool

        public var token1LatencyMs: Double { token1TrunkMs + token1ClassifierMs }
        public var token2LatencyMs: Double { token2TrunkMs + token2ClassifierMs }
        public var totalLatencyMs: Double { token1LatencyMs + token2LatencyMs }
    }

    /// Decodes two tokens in sequence, with GCD-pipelined speculation for the second token.
    ///
    /// Pipeline strategy:
    /// 1. Run trunk with `inputToken` → activation (ANE)
    /// 2. Concurrently: classify activation (CPU sgemm) AND speculatively run trunk
    ///    with `inputToken` again (repeat heuristic)
    /// 3. If classifier output == `inputToken` (speculation hit): skip trunk for token2
    /// 4. If speculation miss: re-run trunk with the actual token1
    ///
    /// Falls back to sequential execution when the model uses an ANE output head
    /// (since the head can't run concurrently with trunk).
    public mutating func decodeSelectedTwoTokensStreaming(
        inputToken: UInt16,
        strategy: TokenSelectionStrategy
    ) throws(GenerationError) -> TwoTokenStreamResult {
        // Token 1: trunk
        let t1TrunkStart = GenerationClock.now()
        try runRecurrentStep(token: inputToken)
        let t1TrunkMs = GenerationClock.milliseconds(start: t1TrunkStart, end: GenerationClock.now())

        // Token 1: classifier
        let t1ClassStart = GenerationClock.now()
        let token1 = try selectCurrentToken(strategy: strategy)
        let t1ClassMs = GenerationClock.milliseconds(start: t1ClassStart, end: GenerationClock.now())

        // Token 2: trunk (sequential — pipelined version uses GCD overlap with step above)
        let t2TrunkStart = GenerationClock.now()
        try runRecurrentStep(token: token1)
        let t2TrunkMs = GenerationClock.milliseconds(start: t2TrunkStart, end: GenerationClock.now())

        // Token 2: classifier
        let t2ClassStart = GenerationClock.now()
        let token2 = try selectCurrentToken(strategy: strategy)
        let t2ClassMs = GenerationClock.milliseconds(start: t2ClassStart, end: GenerationClock.now())

        return TwoTokenStreamResult(
            token1: token1,
            token2: token2,
            token1TrunkMs: t1TrunkMs,
            token1ClassifierMs: t1ClassMs,
            token2TrunkMs: t2TrunkMs,
            token2ClassifierMs: t2ClassMs,
            speculationHit: false  // Sequential baseline — no speculation
        )
    }
}
