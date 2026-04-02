import Foundation
import ANETypes

public enum VerifierStep: Int, Sendable, CaseIterable, Comparable {
    case one = 1
    case two = 2
    case four = 4

    public static func < (lhs: VerifierStep, rhs: VerifierStep) -> Bool {
        lhs.rawValue < rhs.rawValue
    }
}

public struct SpeculativeVerifyPassMetrics: Sendable, Equatable {
    public let requestedHorizon: Int
    public let verifierStep: Int
    public let proposerLatencyMs: Double
    public let verifierTrunkLatencyMs: Double
    public let verifierLogitsLatencyMs: Double
    public let stateAdvanceLatencyMs: Double
    public let acceptedPrefixLength: Int
    public let proposerConfidences: [Float]

    public init(
        requestedHorizon: Int,
        verifierStep: Int,
        proposerLatencyMs: Double,
        verifierTrunkLatencyMs: Double,
        verifierLogitsLatencyMs: Double,
        stateAdvanceLatencyMs: Double,
        acceptedPrefixLength: Int,
        proposerConfidences: [Float] = []
    ) {
        self.requestedHorizon = requestedHorizon
        self.verifierStep = verifierStep
        self.proposerLatencyMs = proposerLatencyMs
        self.verifierTrunkLatencyMs = verifierTrunkLatencyMs
        self.verifierLogitsLatencyMs = verifierLogitsLatencyMs
        self.stateAdvanceLatencyMs = stateAdvanceLatencyMs
        self.acceptedPrefixLength = acceptedPrefixLength
        self.proposerConfidences = proposerConfidences
    }

    public var committedTokenCount: Int {
        acceptedPrefixLength
    }

    public var acceptedFutureTokenCount: Int {
        max(0, acceptedPrefixLength - 1)
    }

    public var committedExactTokenCount: Int {
        acceptedPrefixLength
    }

    public var totalLatencyMs: Double {
        proposerLatencyMs + verifierTrunkLatencyMs + verifierLogitsLatencyMs + stateAdvanceLatencyMs
    }
}

public struct SpeculativeVerifyPassResult: Sendable, Equatable {
    public let committedTokens: [TokenID]
    public let nextCurrentToken: TokenID
    public let metrics: SpeculativeVerifyPassMetrics

    public init(
        committedTokens: [TokenID],
        nextCurrentToken: TokenID,
        metrics: SpeculativeVerifyPassMetrics
    ) {
        self.committedTokens = committedTokens
        self.nextCurrentToken = nextCurrentToken
        self.metrics = metrics
    }
}

public struct SpeculativeVerifyGenerationTrace: Sendable, Equatable {
    public let promptTokens: [TokenID]
    public let generatedTokens: [TokenID]
    public let prefillLatencyMs: Double
    public let passMetrics: [SpeculativeVerifyPassMetrics]
    public let oracleSource: String?

    public init(
        promptTokens: [TokenID],
        generatedTokens: [TokenID],
        prefillLatencyMs: Double,
        passMetrics: [SpeculativeVerifyPassMetrics],
        oracleSource: String? = nil
    ) {
        self.promptTokens = promptTokens
        self.generatedTokens = generatedTokens
        self.prefillLatencyMs = prefillLatencyMs
        self.passMetrics = passMetrics
        self.oracleSource = oracleSource
    }

    public var acceptedPrefixLengths: [Int] {
        passMetrics.map(\.acceptedPrefixLength)
    }

    public var requestedHorizons: [Int] {
        passMetrics.map(\.requestedHorizon)
    }

    public var verifierSteps: [Int] {
        passMetrics.map(\.verifierStep)
    }

    public var committedExactTokenCounts: [Int] {
        passMetrics.map(\.committedExactTokenCount)
    }

    public var acceptedFutureTokenCounts: [Int] {
        passMetrics.map(\.acceptedFutureTokenCount)
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

    public var verifierStepHistogram: [Int: Int] {
        passMetrics.reduce(into: [:]) { histogram, metrics in
            histogram[metrics.verifierStep, default: 0] += 1
        }
    }

    public var acceptedPrefixHistogram: [Int: Int] {
        passMetrics.reduce(into: [:]) { histogram, metrics in
            histogram[metrics.acceptedPrefixLength, default: 0] += 1
        }
    }
}

public protocol SpeculativeVerifyingLanguageModel: ~Copyable, GenerationPerformanceTrackable {
    var vocabSize: Int { get }
    var maxVerifierHorizon: Int { get }

    mutating func reset() throws(GenerationError)
    mutating func prefillSelectedToken(
        promptTokens: [TokenID],
        strategy: TokenSelectionStrategy
    ) throws(GenerationError) -> TokenID
    mutating func performSpeculativeVerifyPass(
        currentToken: TokenID,
        requestedHorizon: Int,
        remainingTokenBudget: Int,
        strategy: TokenSelectionStrategy,
        proposalOverride: [TokenID]?
    ) throws(GenerationError) -> SpeculativeVerifyPassResult
}

public struct SpeculativeVerifierGenerationHarness<Model: SpeculativeVerifyingLanguageModel>: ~Copyable
where Model: ~Copyable {
    public var model: Model
    public let strategy: TokenSelectionStrategy
    public let requestedVerifierStep: Int

    public init(
        model: consuming Model,
        strategy: TokenSelectionStrategy = .argmax,
        requestedVerifierStep: Int
    ) {
        self.model = model
        self.strategy = strategy
        self.requestedVerifierStep = requestedVerifierStep
    }

    public mutating func generate(
        promptTokens: [TokenID],
        maxNewTokens: Int,
        oracleTokens: [TokenID]? = nil,
        oracleSource: String? = nil
    ) throws(GenerationError) -> SpeculativeVerifyGenerationTrace {
        guard !promptTokens.isEmpty else {
            throw .invalidArguments("promptTokens must not be empty")
        }
        guard maxNewTokens > 0 else {
            throw .invalidArguments("maxNewTokens must be > 0")
        }
        guard requestedVerifierStep > 0 else {
            throw .invalidArguments("requestedVerifierStep must be > 0")
        }

        try model.reset()
        let prefillStart = GenerationClock.now()
        var currentToken = try model.prefillSelectedToken(
            promptTokens: promptTokens,
            strategy: strategy
        )
        let prefillLatencyMs = GenerationClock.milliseconds(start: prefillStart, end: GenerationClock.now())

        var generatedTokens: [TokenID] = []
        var passMetrics: [SpeculativeVerifyPassMetrics] = []
        generatedTokens.reserveCapacity(maxNewTokens)
        passMetrics.reserveCapacity(maxNewTokens)

        while generatedTokens.count < maxNewTokens {
            let remainingTokenBudget = maxNewTokens - generatedTokens.count
            let requestedHorizon = min(requestedVerifierStep, remainingTokenBudget)
            let result = try performVerifierPass(
                currentToken: currentToken,
                requestedHorizon: requestedHorizon,
                remainingTokenBudget: remainingTokenBudget,
                generatedTokenCount: generatedTokens.count,
                oracleTokens: oracleTokens
            )

            guard !result.committedTokens.isEmpty else {
                throw .runtimeFailure("speculative verifier pass must commit at least one token")
            }
            guard result.committedTokens[0] == currentToken else {
                throw .runtimeFailure("speculative verifier pass must commit current token first")
            }
            guard result.committedTokens.count <= requestedHorizon else {
                throw .runtimeFailure(
                    "speculative verifier pass committed \(result.committedTokens.count) tokens with requested horizon \(requestedHorizon)"
                )
            }
            guard result.metrics.acceptedPrefixLength == result.committedTokens.count else {
                throw .runtimeFailure(
                    "acceptedPrefixLength \(result.metrics.acceptedPrefixLength) must equal committed token count \(result.committedTokens.count)"
                )
            }

            generatedTokens.append(contentsOf: result.committedTokens)
            passMetrics.append(result.metrics)
            currentToken = result.nextCurrentToken
        }

        return SpeculativeVerifyGenerationTrace(
            promptTokens: promptTokens,
            generatedTokens: generatedTokens,
            prefillLatencyMs: prefillLatencyMs,
            passMetrics: passMetrics,
            oracleSource: oracleSource
        )
    }

    private mutating func performVerifierPass(
        currentToken: TokenID,
        requestedHorizon: Int,
        remainingTokenBudget: Int,
        generatedTokenCount: Int,
        oracleTokens: [TokenID]?
    ) throws(GenerationError) -> SpeculativeVerifyPassResult {
        guard requestedHorizon > 0 else {
            throw .invalidArguments("requestedHorizon must be > 0")
        }

        var committedTokens: [TokenID] = []
        committedTokens.reserveCapacity(requestedHorizon)
        var proposerLatencyMs = 0.0
        var verifierTrunkLatencyMs = 0.0
        var verifierLogitsLatencyMs = 0.0
        var stateAdvanceLatencyMs = 0.0
        var proposerConfidences: [Float] = []
        var nextCurrentToken = currentToken
        var committedInOuterPass = 0

        while committedInOuterPass < requestedHorizon {
            let outerRemaining = requestedHorizon - committedInOuterPass
            let subpassHorizon = min(model.maxVerifierHorizon, outerRemaining, remainingTokenBudget - committedTokens.count)
            guard subpassHorizon > 0 else {
                break
            }

            let proposalOverride = try oracleProposalOverride(
                oracleTokens: oracleTokens,
                generatedTokenCount: generatedTokenCount,
                committedOffset: committedInOuterPass,
                subpassHorizon: subpassHorizon
            )
            let result = try model.performSpeculativeVerifyPass(
                currentToken: nextCurrentToken,
                requestedHorizon: subpassHorizon,
                remainingTokenBudget: remainingTokenBudget - committedTokens.count,
                strategy: strategy,
                proposalOverride: proposalOverride
            )

            guard !result.committedTokens.isEmpty else {
                throw .runtimeFailure("subpass must commit at least one token")
            }
            guard result.committedTokens[0] == nextCurrentToken else {
                throw .runtimeFailure("subpass must commit current token first")
            }
            guard result.metrics.acceptedPrefixLength == result.committedTokens.count else {
                throw .runtimeFailure(
                    "subpass acceptedPrefixLength \(result.metrics.acceptedPrefixLength) must equal committed token count \(result.committedTokens.count)"
                )
            }

            committedTokens.append(contentsOf: result.committedTokens)
            proposerLatencyMs += result.metrics.proposerLatencyMs
            verifierTrunkLatencyMs += result.metrics.verifierTrunkLatencyMs
            verifierLogitsLatencyMs += result.metrics.verifierLogitsLatencyMs
            stateAdvanceLatencyMs += result.metrics.stateAdvanceLatencyMs
            proposerConfidences.append(contentsOf: result.metrics.proposerConfidences)
            nextCurrentToken = result.nextCurrentToken
            committedInOuterPass += result.committedTokens.count

            if result.committedTokens.count < subpassHorizon {
                break
            }
        }

        return SpeculativeVerifyPassResult(
            committedTokens: committedTokens,
            nextCurrentToken: nextCurrentToken,
            metrics: SpeculativeVerifyPassMetrics(
                requestedHorizon: requestedHorizon,
                verifierStep: requestedHorizon,
                proposerLatencyMs: proposerLatencyMs,
                verifierTrunkLatencyMs: verifierTrunkLatencyMs,
                verifierLogitsLatencyMs: verifierLogitsLatencyMs,
                stateAdvanceLatencyMs: stateAdvanceLatencyMs,
                acceptedPrefixLength: committedTokens.count,
                proposerConfidences: proposerConfidences
            )
        )
    }

    private func oracleProposalOverride(
        oracleTokens: [TokenID]?,
        generatedTokenCount: Int,
        committedOffset: Int,
        subpassHorizon: Int
    ) throws(GenerationError) -> [TokenID]? {
        guard let oracleTokens else {
            return nil
        }
        guard subpassHorizon > 1 else {
            return []
        }
        let proposalIndex = generatedTokenCount + committedOffset + 1
        guard proposalIndex < oracleTokens.count else {
            throw .invalidArguments(
                "oracle token trace length \(oracleTokens.count) does not cover proposal index \(proposalIndex)"
            )
        }
        return [oracleTokens[proposalIndex]]
    }
}

extension ANEExactTwoTokenUpperBoundGenerationModel: SpeculativeVerifyingLanguageModel {
    public var maxVerifierHorizon: Int { 2 }

    public mutating func performSpeculativeVerifyPass(
        currentToken: TokenID,
        requestedHorizon: Int,
        remainingTokenBudget: Int,
        strategy: TokenSelectionStrategy,
        proposalOverride: [TokenID]?
    ) throws(GenerationError) -> SpeculativeVerifyPassResult {
        guard (1...2).contains(requestedHorizon) else {
            throw .invalidArguments("ANE exact upper-bound verifier supports requestedHorizon 1...2")
        }
        let result = try performExactTwoTokenPass(
            currentToken: currentToken,
            remainingTokenBudget: min(requestedHorizon, remainingTokenBudget),
            strategy: strategy
        )
        return SpeculativeVerifyPassResult(
            committedTokens: result.committedTokens,
            nextCurrentToken: result.nextCurrentToken,
            metrics: SpeculativeVerifyPassMetrics(
                requestedHorizon: requestedHorizon,
                verifierStep: requestedHorizon,
                proposerLatencyMs: result.metrics.proposerLatencyMs,
                verifierTrunkLatencyMs: result.metrics.verifierTrunkLatencyMs,
                verifierLogitsLatencyMs: result.metrics.verifierLogitsLatencyMs,
                stateAdvanceLatencyMs: result.metrics.stateAdvanceLatencyMs,
                acceptedPrefixLength: result.committedTokens.count
            )
        )
    }
}

extension ANEExactTwoTokenBranchStatePromotionModel: SpeculativeVerifyingLanguageModel {
    public var maxVerifierHorizon: Int { 2 }

    public mutating func performSpeculativeVerifyPass(
        currentToken: TokenID,
        requestedHorizon: Int,
        remainingTokenBudget: Int,
        strategy: TokenSelectionStrategy,
        proposalOverride: [TokenID]?
    ) throws(GenerationError) -> SpeculativeVerifyPassResult {
        guard (1...2).contains(requestedHorizon) else {
            throw .invalidArguments("ANE exact branch-state verifier supports requestedHorizon 1...2")
        }
        let result = try performExactTwoTokenPass(
            currentToken: currentToken,
            remainingTokenBudget: min(requestedHorizon, remainingTokenBudget),
            strategy: strategy
        )
        return SpeculativeVerifyPassResult(
            committedTokens: result.committedTokens,
            nextCurrentToken: result.nextCurrentToken,
            metrics: SpeculativeVerifyPassMetrics(
                requestedHorizon: requestedHorizon,
                verifierStep: requestedHorizon,
                proposerLatencyMs: result.metrics.proposerLatencyMs,
                verifierTrunkLatencyMs: result.metrics.verifierTrunkLatencyMs,
                verifierLogitsLatencyMs: result.metrics.verifierLogitsLatencyMs,
                stateAdvanceLatencyMs: result.metrics.stateAdvanceLatencyMs,
                acceptedPrefixLength: result.committedTokens.count
            )
        )
    }
}
