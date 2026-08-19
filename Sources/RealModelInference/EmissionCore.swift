import Foundation
import ANETypes
import ModelSupport

/// Where the end-of-sequence token id comes from for one decode loop.
///
/// Injected per model family so no loop hardcodes an EOS source: GPT-2
/// sessions pass the fixed tokenizer constant, llama sessions pass the
/// artifact's optional `config.eosToken`.
enum EOSPolicy: Sendable, Equatable {
    /// One constant token id that always terminates decoding (GPT-2 family).
    case fixed(Int)
    /// The model config's optional `eosToken`; `nil` disables EOS termination.
    case fromConfig(Int?)
}

/// Accumulator owning the four emission concerns shared by every decode loop:
///
/// 1. first-token latency capture,
/// 2. tokens-per-second computation,
/// 3. ``GenerationStep`` construction for `onStep` callbacks,
/// 4. final ``GenerationResult`` assembly.
///
/// Loops keep their own KV-cache handling, speculative-draft logic, and
/// early-return shapes; this type owns only how emitted tokens turn into
/// telemetry and results. Per-token latencies measure emission-to-emission,
/// starting from session start.
struct EmissionCore {
    /// Resolved end-of-sequence id under the injected policy, or `nil` when
    /// this run never terminates early.
    private(set) var eosTokenID: Int?

    /// Session start clock used by every latency and rate computation.
    private let startNanos: UInt64
    /// Clock of the previous emission (initialized to session start).
    private var lastEmissionNanos: UInt64
    private var clock: GenerateClock
    private var firstTokenLatencyMsValue = 0.0
    private var prefillMsValue = 0.0
    private var firstTokenRecorded = false

    /// Prompt plus every fully emitted token; the text carrier.
    private(set) var allTokens: [TokenID]
    /// Emitted output tokens; llama-family terminal tokens land here only.
    private(set) var generatedTokens: [TokenID]
    private(set) var tokenLatenciesMs: [Double]

    private let promptTokens: [TokenID]
    private let onStep: ((GenerationStep) -> Void)?
    private let decodeText: ([Int]) -> String

    init(
        promptTokens: [TokenID],
        capacity: Int,
        eos: EOSPolicy,
        onStep: ((GenerationStep) -> Void)?,
        decodeText: @escaping ([Int]) -> String,
        startNanos: UInt64 = DispatchTime.now().uptimeNanoseconds,
        clock: GenerateClock? = nil
    ) {
        self.eosTokenID = switch eos {
        case .fixed(let tokenID): tokenID
        case .fromConfig(let tokenID): tokenID
        }
        let resolvedClock = clock ?? GenerateClock(submitNS: startNanos)
        self.clock = resolvedClock
        self.startNanos = resolvedClock.submitNS
        self.lastEmissionNanos = resolvedClock.submitNS
        self.allTokens = promptTokens
        self.generatedTokens = []
        self.generatedTokens.reserveCapacity(capacity)
        self.tokenLatenciesMs = []
        self.tokenLatenciesMs.reserveCapacity(capacity)
        self.promptTokens = promptTokens
        self.onStep = onStep
        self.decodeText = decodeText
    }

    // MARK: Loop control inputs

    /// Whether `token` ends decoding under the injected ``EOSPolicy``.
    func terminatesDecoding(_ token: TokenID) -> Bool {
        guard let eosTokenID else { return false }
        return Int(token) == eosTokenID
    }

    /// Fully emitted tokens so far.
    var generatedTokenCount: Int { generatedTokens.count }
    /// Prompt plus fully emitted tokens so far.
    var allTokensCount: Int { allTokens.count }
    /// First-token latency captured so far (`0` before the first emission point).
    var firstTokenLatencyMs: Double { firstTokenLatencyMsValue }

    // MARK: The four concerns

    /// Concern 1 — first-token latency capture.
    ///
    /// Idempotent: loops that must record before their EOS check call this
    /// explicitly, and ``emit(_:at:)`` calls it again as a safety net for
    /// closure-style loops whose only capture point is the emit itself.
    mutating func recordFirstTokenIfFirst(at now: UInt64) {
        guard !firstTokenRecorded else { return }
        clock.markPrefillEnd(at: now)
        let timing = clock.timing(firstTokenNS: now)
        firstTokenLatencyMsValue = timing.firstTokenLatencyMs
        prefillMsValue = timing.prefillMs
        firstTokenRecorded = true
    }

    /// Concerns 1–3 — record one fully emitted token.
    ///
    /// Appends to both token arrays, records the per-token latency against the
    /// previous emission clock, updates the running rate, and fires `onStep`
    /// with the constructed step.
    mutating func emit(_ token: TokenID, at now: UInt64) {
        generatedTokens.append(token)
        allTokens.append(token)
        let tokenLatencyMs = Self.milliseconds(from: now - lastEmissionNanos)
        tokenLatenciesMs.append(tokenLatencyMs)
        recordFirstTokenIfFirst(at: now)
        let elapsedMs = Self.milliseconds(from: now - startNanos)
        let tokensPerSecond = Self.tokensPerSecond(
            count: generatedTokens.count,
            elapsedMs: elapsedMs
        )
        onStep?(
            GenerationStep(
                token: token,
                generatedTokens: generatedTokens,
                text: decodeText(allTokens.map(Int.init)),
                tokenLatencyMs: tokenLatencyMs,
                elapsedMs: elapsedMs,
                firstTokenLatencyMs: firstTokenLatencyMsValue,
                prefillMs: prefillMsValue,
                tokensPerSecond: tokensPerSecond
            )
        )
        lastEmissionNanos = now
    }

    /// Record a llama-family terminal token: it lands in the reported `tokens`
    /// array but not in the decoded text, and fires no step callback.
    mutating func recordTerminalToken(_ token: TokenID) {
        generatedTokens.append(token)
    }

    /// Concern 4 — assemble the final result from accumulated state plus the
    /// trunk-level fields only this loop knows.
    func makeResult(
        compileTimeMs: Double,
        exactHeadBackend: String? = nil,
        cachedBindingsEnabled: Bool = false,
        committedExactTokensPerPass: Double? = nil,
        acceptedFutureTokensPerPass: Double? = nil,
        trunk: Trunk? = nil,
        hopsPerToken: Int? = nil,
        decodeProfileReport: String? = nil,
        tokensPerSecondOverride: Double? = nil,
        textOverride: String? = nil,
        at now: UInt64 = DispatchTime.now().uptimeNanoseconds
    ) -> GenerationResult {
        GenerationResult(
            text: textOverride ?? decodeText(allTokens.map(Int.init)),
            tokens: generatedTokens,
            promptTokens: promptTokens,
            tokenLatenciesMs: tokenLatenciesMs,
            tokensPerSecond: tokensPerSecondOverride ?? Self.tokensPerSecond(
                count: generatedTokens.count,
                elapsedMs: Self.milliseconds(from: now - startNanos)
            ),
            compileTimeMs: compileTimeMs,
            firstTokenLatencyMs: firstTokenLatencyMsValue,
            prefillMs: prefillMsValue,
            exactHeadBackend: exactHeadBackend ?? "unknown",
            cachedBindingsEnabled: cachedBindingsEnabled,
            committedExactTokensPerPass: committedExactTokensPerPass,
            acceptedFutureTokensPerPass: acceptedFutureTokensPerPass,
            trunk: trunk,
            hopsPerToken: hopsPerToken,
            decodeProfileReport: decodeProfileReport
        )
    }

    /// Concern 2 — the one rate formula used in-loop and at result assembly.
    static func tokensPerSecond(count: Int, elapsedMs: Double) -> Double {
        guard count > 0 else { return 0 }
        return Double(count) / max(elapsedMs / 1_000, 1e-9)
    }

    private static func milliseconds(from nanoseconds: UInt64) -> Double {
        Double(nanoseconds) / 1_000_000
    }
}
