import Testing
import ANETypes
@testable import RealModelInference

private func makeEmission(
    promptTokens: [TokenID] = [10, 11],
    capacity: Int = 8,
    eos: EOSPolicy,
    onStep: ((GenerationStep) -> Void)? = nil,
    startNanos: UInt64 = 1_000_000_000
) -> EmissionCore {
    EmissionCore(
        promptTokens: promptTokens,
        capacity: capacity,
        eos: eos,
        onStep: onStep,
        decodeText: { $0.map(String.init).joined(separator: ",") },
        startNanos: startNanos
    )
}

@Suite struct EmissionCoreTests {
    // MARK: EOSPolicy injection

    @Test func fixedPolicyTerminatesOnTheInjectedConstantOnly() {
        let emission = makeEmission(eos: .fixed(50_256))
        #expect(emission.eosTokenID == 50_256)
        #expect(emission.terminatesDecoding(50_256))
        #expect(!emission.terminatesDecoding(50_255))
        #expect(!emission.terminatesDecoding(0))
    }

    @Test func fromConfigPolicyUsesTheArtifactValueAndNilDisablesEOS() {
        let qwenLike = makeEmission(eos: .fromConfig(151_643))
        #expect(qwenLike.terminatesDecoding(151_643))
        #expect(!qwenLike.terminatesDecoding(151_644))

        let noEOS = makeEmission(eos: .fromConfig(nil))
        #expect(noEOS.eosTokenID == nil)
        #expect(!noEOS.terminatesDecoding(0))
    }

    // MARK: Concern 1 — first-token latency capture

    @Test func firstTokenLatencyIsZeroUntilCapturedThenIdempotent() {
        var emission = makeEmission(eos: .fixed(50_256))
        #expect(emission.firstTokenLatencyMs == 0)

        emission.recordFirstTokenIfFirst(at: 1_002_500_000)
        #expect(emission.firstTokenLatencyMs == 2.5)

        emission.recordFirstTokenIfFirst(at: 1_900_000_000)
        #expect(emission.firstTokenLatencyMs == 2.5)
    }

    @Test func emitCapturesFirstTokenLatencyWithoutExplicitRecord() {
        var emission = makeEmission(eos: .fixed(50_256))
        emission.emit(7, at: 1_004_000_000)
        #expect(emission.firstTokenLatencyMs == 4)
    }

    @Test func emitPublishesGenerateClockPrefillAndTTFT() throws {
        var steps: [GenerationStep] = []
        var emission = EmissionCore(
            promptTokens: [10],
            capacity: 4,
            eos: .fixed(50_256),
            onStep: { steps.append($0) },
            decodeText: { $0.map(String.init).joined() },
            clock: GenerateClock(submitNS: 1_000_000_000)
        )
        emission.emit(7, at: 1_080_000_000)
        let step = try #require(steps.first)
        #expect(abs(step.prefillMs - 80) < 1e-9)
        #expect(abs(step.firstTokenLatencyMs - 80) < 1e-9)
        let result = emission.makeResult(compileTimeMs: 0)
        #expect(abs(result.prefillMs - 80) < 1e-9)
        #expect(result.firstTokenLatencyMs == result.ttftIncludingPrefillMs)
    }

    // MARK: Concerns 1–3 — emit bookkeeping and step construction

    @Test func emitAppendsTokensRecordsLatenciesAndFiresSteps() throws {
        var steps: [GenerationStep] = []
        var emission = makeEmission(eos: .fixed(50_256)) { steps.append($0) }

        emission.emit(7, at: 1_001_000_000)
        emission.emit(9, at: 1_003_500_000)

        #expect(emission.generatedTokens == [7, 9])
        #expect(emission.generatedTokenCount == 2)
        #expect(emission.allTokens == [10, 11, 7, 9])
        #expect(emission.allTokensCount == 4)
        #expect(emission.tokenLatenciesMs.count == 2)

        #expect(steps.count == 2)
        let first = try #require(steps.first)
        #expect(first.token == 7)
        #expect(first.generatedTokens == [7])
        #expect(first.text == "10,11,7")
        #expect(first.tokenLatencyMs == 1)
        #expect(first.elapsedMs == 1)
        #expect(first.firstTokenLatencyMs == 1)

        let second = try #require(steps.last)
        #expect(second.token == 9)
        #expect(second.generatedTokens == [7, 9])
        #expect(second.text == "10,11,7,9")
        #expect(second.tokenLatencyMs == 2.5)
        #expect(second.elapsedMs == 3.5)
        #expect(second.firstTokenLatencyMs == 1)
        let secondTPS = EmissionCore.tokensPerSecond(count: 2, elapsedMs: 3.5)
        #expect(abs(second.tokensPerSecond - secondTPS) < 1e-9)
    }

    @Test func terminalTokenLandsInTokensButNotTextAndFiresNoStep() {
        var steps: [GenerationStep] = []
        var emission = makeEmission(promptTokens: [10], eos: .fromConfig(99)) { steps.append($0) }

        emission.recordTerminalToken(99)

        #expect(emission.generatedTokens == [99])
        #expect(emission.allTokens == [10])
        #expect(steps.isEmpty)

        let result = emission.makeResult(
            compileTimeMs: 0,
            trunk: .exactCPU,
            at: 1_010_000_000
        )
        #expect(result.tokens == [99])
        #expect(result.text == "10")
        #expect(result.tokensPerSecond == 100)
    }

    // MARK: Concern 2 — one rate formula

    @Test func tokensPerSecondFormulaIsSharedByLoopAndAssemblyPaths() {
        #expect(EmissionCore.tokensPerSecond(count: 0, elapsedMs: 12.5) == 0)
        let rate = EmissionCore.tokensPerSecond(count: 25, elapsedMs: 1_250)
        #expect(abs(rate - 20) < 1e-9)

        var emission = makeEmission(eos: .fixed(50_256))
        emission.emit(1, at: 1_001_250_000)
        let result = emission.makeResult(compileTimeMs: 0, at: 1_001_250_000)
        #expect(abs(result.tokensPerSecond - EmissionCore.tokensPerSecond(count: 1, elapsedMs: 1.25)) < 1e-9)
    }

    // MARK: Concern 4 — result assembly

    @Test func makeResultDefaultsRepresentTypeLevelAbsence() {
        let emission = makeEmission(eos: .fixed(50_256))
        let result = emission.makeResult(
            compileTimeMs: 3.25,
            at: 1_000_500_000
        )
        #expect(result.text == "10,11")
        #expect(result.tokens.isEmpty)
        #expect(result.promptTokens == [10, 11])
        #expect(result.tokensPerSecond == 0)
        #expect(result.compileTimeMs == 3.25)
        #expect(result.firstTokenLatencyMs == 0)
        #expect(result.trunk == nil)
        #expect(result.decodePath == "unknown")
        #expect(result.exactHeadBackend == "unknown")
        #expect(!result.cachedBindingsEnabled)
        #expect(result.hopsPerToken == nil)
        #expect(result.decodeProfileReport == nil)
    }

    @Test func makeResultPopulatesTrunkFieldsWhenTheLoopSuppliesThem() {
        var emission = makeEmission(eos: .fromConfig(nil))
        emission.emit(5, at: 1_001_000_000)

        let fused = emission.makeResult(
            compileTimeMs: 0,
            trunk: .fusedHybrid,
            hopsPerToken: 6
        )
        #expect(fused.trunk == .fusedHybrid)
        #expect(fused.hopsPerToken == 6)
        #expect(fused.tokens == [5])

        let split = emission.makeResult(compileTimeMs: 0, trunk: .splitHybrid)
        #expect(split.trunk == .splitHybrid)
        #expect(split.exactHeadBackend == "unknown")
    }

    @Test func makeResultOverridesServeDistinctEarlyReturnShapes() {
        var emission = makeEmission(promptTokens: [10], eos: .fromConfig(99))
        emission.recordTerminalToken(99)

        let earlyReturn = emission.makeResult(
            compileTimeMs: 1,
            exactHeadBackend: "cpu_exact_two_token_draft",
            trunk: .exactCPU,
            tokensPerSecondOverride: 0,
            textOverride: "10,99"
        )
        #expect(earlyReturn.tokens == [99])
        #expect(earlyReturn.tokensPerSecond == 0)
        #expect(earlyReturn.firstTokenLatencyMs == 0)
        #expect(earlyReturn.text == "10,99")
        #expect(earlyReturn.trunk == .exactCPU)

        var running = makeEmission(eos: .fromConfig(nil))
        running.emit(5, at: 1_002_000_000)
        let budgetExhausted = running.makeResult(
            compileTimeMs: 1,
            exactHeadBackend: "cpu_exact_two_token_draft",
            trunk: .exactCPU
        )
        #expect(budgetExhausted.committedExactTokensPerPass == nil)
        #expect(budgetExhausted.acceptedFutureTokensPerPass == nil)
        #expect(budgetExhausted.tokens == [5])

        let finalPass = running.makeResult(
            compileTimeMs: 1,
            exactHeadBackend: "cpu_exact_two_token_draft",
            committedExactTokensPerPass: 1.75,
            acceptedFutureTokensPerPass: 0.5,
            trunk: .exactCPU
        )
        #expect(finalPass.committedExactTokensPerPass == 1.75)
        #expect(finalPass.acceptedFutureTokensPerPass == 0.5)
    }
}
