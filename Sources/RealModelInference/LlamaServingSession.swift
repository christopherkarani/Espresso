import Foundation
import ANETypes
import Espresso
import ModelSupport

/// What a decode step hands back to the serving-session loop.
///
/// `.selected` comes from an on-device greedy head that already resolved a token;
/// `.normalizedHidden` carries final-norm hidden state for loop-side sampling.
/// Both were produced by today's loops; the distinction preserves the fast paths.
enum LlamaDecodeProposal {
    case selected(TokenID)
    case normalizedHidden([Float])
}

/// Everything the serving loop collected for one llama serving run.
struct SessionTranscript {
    let promptTokens: [TokenID]
    let generatedTokens: [TokenID]
    let allTokens: [TokenID]
    let tokenLatenciesMs: [Double]
    let firstTokenLatencyMs: Double
    let tokensPerSecond: Double
    let text: String
    let decodeProfileReport: String?
}

/// Shared emission choreography for every decode loop.
///
/// Owns token buffers, latency math, first-token tracking, and `onStep` firing so
/// neither the unified session loop nor the speculative draft loop keeps its own copy.
struct TokenEmitter {
    private(set) var allTokens: [TokenID]
    private(set) var generatedTokens: [TokenID]
    private(set) var tokenLatenciesMs: [Double]
    private(set) var firstTokenLatencyMs: Double = 0

    private let initialPromptTokens: [TokenID]
    private let decodeText: ([TokenID]) -> String
    private let onStep: ((GenerationStep) -> Void)?
    private let generationStart: UInt64
    private var emissionStart: UInt64
    private var firstTokenRecorded = false

    init(
        promptTokens: [TokenID],
        maxTokens: Int,
        generationStart: UInt64,
        decodeText: @escaping ([TokenID]) -> String,
        onStep: ((GenerationStep) -> Void)?
    ) {
        self.initialPromptTokens = promptTokens
        self.allTokens = promptTokens
        self.generatedTokens = []
        self.generatedTokens.reserveCapacity(maxTokens)
        self.tokenLatenciesMs = []
        self.tokenLatenciesMs.reserveCapacity(maxTokens)
        self.decodeText = decodeText
        self.onStep = onStep
        self.generationStart = generationStart
        self.emissionStart = generationStart
    }

    var generatedCount: Int { generatedTokens.count }
    var totalTokenCount: Int { allTokens.count }

    /// Bookkeeping for a resolved token before the EOS decision, mirroring today's
    /// loops: first-token latency is recorded even when the token ends the run.
    mutating func noteResolution(at emissionNow: UInt64) -> Double {
        let tokenLatencyMs = Self.milliseconds(from: emissionNow - emissionStart)
        if !firstTokenRecorded {
            firstTokenLatencyMs = Self.milliseconds(from: emissionNow - generationStart)
            firstTokenRecorded = true
        }
        return tokenLatencyMs
    }

    /// End-of-sequence token: counted in the result but never streamed or decoded,
    /// exactly as today's loops behave.
    mutating func commitEndOfSequence(_ token: TokenID) {
        generatedTokens.append(token)
    }

    /// Full emission: append, measure, fire `onStep`, roll the latency window.
    mutating func commit(_ token: TokenID, tokenLatencyMs: Double, at emissionNow: UInt64) {
        generatedTokens.append(token)
        allTokens.append(token)
        let elapsedMs = Self.milliseconds(from: emissionNow - generationStart)
        tokenLatenciesMs.append(tokenLatencyMs)
        let tokensPerSecond = Double(generatedTokens.count) / max(elapsedMs / 1_000, 1e-9)
        onStep?(
            GenerationStep(
                token: token,
                generatedTokens: generatedTokens,
                text: decodeText(allTokens),
                tokenLatencyMs: tokenLatencyMs,
                elapsedMs: elapsedMs,
                firstTokenLatencyMs: firstTokenLatencyMs,
                tokensPerSecond: tokensPerSecond
            )
        )
        emissionStart = emissionNow
    }

    /// The draft loop's emit shape: measure and commit in one step.
    mutating func emit(_ token: TokenID, at emissionNow: UInt64) {
        let tokenLatencyMs = noteResolution(at: emissionNow)
        commit(token, tokenLatencyMs: tokenLatencyMs, at: emissionNow)
    }

    /// Final throughput over the whole generation window.
    func tokensPerSecond(until generationEnd: UInt64) -> Double {
        let generationTimeMs = Self.milliseconds(from: generationEnd - generationStart)
        return generatedTokens.isEmpty
            ? 0
            : Double(generatedTokens.count) / max(generationTimeMs / 1_000, 1e-9)
    }

    func makeTranscript(decodeProfileReport: String?) -> SessionTranscript {
        SessionTranscript(
            promptTokens: initialPromptTokens,
            generatedTokens: generatedTokens,
            allTokens: allTokens,
            tokenLatenciesMs: tokenLatenciesMs,
            firstTokenLatencyMs: firstTokenLatencyMs,
            tokensPerSecond: tokensPerSecond(until: Self.now()),
            text: decodeText(allTokens),
            decodeProfileReport: decodeProfileReport
        )
    }

    static func now() -> UInt64 { DispatchTime.now().uptimeNanoseconds }

    static func milliseconds(from nanos: UInt64) -> Double {
        Double(nanos) / 1_000_000
    }
}
