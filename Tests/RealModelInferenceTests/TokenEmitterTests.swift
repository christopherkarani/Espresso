import ANETypes
import Testing
@testable import RealModelInference

struct TokenEmitterTests {
    private func makeEmitter(
        generationStart: UInt64 = 1_000_000,
        maxTokens: Int = 8,
        prompt: [TokenID] = [10, 11],
        onStep: ((GenerationStep) -> Void)? = nil
    ) -> TokenEmitter {
        TokenEmitter(
            promptTokens: prompt,
            maxTokens: maxTokens,
            generationStart: generationStart,
            decodeText: { tokens in tokens.map(String.init).joined(separator: ",") },
            onStep: onStep
        )
    }

    @Test func commitAppendsTokensAndFiresOnStep() {
        var steps: [GenerationStep] = []
        var emitter = makeEmitter(onStep: { steps.append($0) })

        let now1 = TokenEmitter.now()
        let latency1 = emitter.noteResolution(at: now1)
        emitter.commit(20, tokenLatencyMs: latency1, at: now1)
        let now2 = TokenEmitter.now()
        let latency2 = emitter.noteResolution(at: now2)
        emitter.commit(21, tokenLatencyMs: latency2, at: now2)

        #expect(emitter.generatedTokens == [20, 21])
        #expect(emitter.allTokens == [10, 11, 20, 21])
        #expect(emitter.generatedCount == 2)
        #expect(emitter.totalTokenCount == 4)
        #expect(emitter.tokenLatenciesMs.count == 2)
        #expect(steps.count == 2)
        #expect(steps[0].token == 20)
        #expect(steps[0].generatedTokens == [20])
        #expect(steps[0].text == "10,11,20")
        #expect(steps[1].generatedTokens == [20, 21])
        #expect(steps[1].text == "10,11,20,21")
        #expect(steps[0].firstTokenLatencyMs >= 0)
        #expect(steps[1].firstTokenLatencyMs == steps[0].firstTokenLatencyMs)
    }

    @Test func endOfSequenceIsCountedButNeverStreamed() {
        var steps: [GenerationStep] = []
        var emitter = makeEmitter(onStep: { steps.append($0) })

        let now = TokenEmitter.now()
        _ = emitter.noteResolution(at: now)
        emitter.commitEndOfSequence(99)

        #expect(emitter.generatedTokens == [99])
        #expect(emitter.allTokens == [10, 11])
        #expect(steps.isEmpty)
    }

    @Test func firstTokenLatencyRecordedEvenWhenFirstTokenEndsRun() {
        var emitter = makeEmitter()

        let now = TokenEmitter.now()
        _ = emitter.noteResolution(at: now)
        emitter.commitEndOfSequence(42)

        #expect(emitter.firstTokenLatencyMs >= 0)

        var fresh = makeEmitter()
        let laterNow = TokenEmitter.now()
        let latency = fresh.noteResolution(at: laterNow)
        fresh.commit(7, tokenLatencyMs: latency, at: laterNow)
        #expect(fresh.firstTokenLatencyMs >= 0)
        #expect(fresh.firstTokenLatencyMs == fresh.tokenLatenciesMs.first)
    }

    @Test func emitCombinesNoteAndCommit() {
        var steps: [GenerationStep] = []
        var emitter = makeEmitter(onStep: { steps.append($0) })

        let now = TokenEmitter.now()
        emitter.emit(5, at: now)

        #expect(emitter.generatedTokens == [5])
        #expect(emitter.tokenLatenciesMs.count == 1)
        #expect(steps.count == 1)
    }

    @Test func transcriptCarriesBuffersTextAndPromptEcho() {
        var emitter = makeEmitter(maxTokens: 4, prompt: [1, 2, 3])
        let nowA = TokenEmitter.now()
        let latency = emitter.noteResolution(at: nowA)
        emitter.commit(4, tokenLatencyMs: latency, at: nowA)

        let transcript = emitter.makeTranscript(decodeProfileReport: String?.none)
        #expect(transcript.promptTokens == [1, 2, 3])
        #expect(transcript.generatedTokens == [4])
        #expect(transcript.allTokens == [1, 2, 3, 4])
        #expect(transcript.text == "1,2,3,4")
        #expect(transcript.decodeProfileReport == nil)
        #expect(transcript.tokensPerSecond > 0)
        #expect(transcript.tokenLatenciesMs.count == 1)
    }

    @Test func tokensPerSecondEmptyGenerationIsZero() {
        let emitter = makeEmitter()
        #expect(emitter.tokensPerSecond(until: TokenEmitter.now()) == 0)

        let transcript = emitter.makeTranscript(decodeProfileReport: "r")
        #expect(transcript.tokensPerSecond == 0)
        #expect(transcript.decodeProfileReport == "r")
    }
}
