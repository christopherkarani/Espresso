import Foundation
import Testing
@testable import RealModelInference

@Test func test_generateClock_ttft_includes_prefill_and_is_not_head_only() {
    final class Ticks: @unchecked Sendable {
        var idx = 0
        let values: [UInt64] = [0, 80_000_000, 101_000_000]
        func next() -> UInt64 {
            let value = values[min(idx, values.count - 1)]
            idx += 1
            return value
        }
    }
    let ticks = Ticks()
    var clock = GenerateClock(now: { ticks.next() })
    clock.markPrefillEnd()
    let headOnlyMs = 21.0
    let timing = clock.timing(firstTokenNS: 101_000_000)

    #expect(abs(timing.prefillMs - 80) < 1e-9)
    #expect(abs(timing.ttftIncludingPrefillMs - 101) < 1e-9)
    #expect(timing.ttftIncludingPrefillMs >= timing.prefillMs)
    #expect(timing.firstTokenLatencyMs == timing.ttftIncludingPrefillMs)
    #expect(timing.firstTokenLatencyMs != headOnlyMs)
}

@Test func test_generationResult_published_ttft_equals_including_prefill() throws {
    let timing = GenerateTiming(prefillMs: 80, firstTokenLatencyIncludingPrefillMs: 101)
    let result = GenerationResult(
        text: "hi",
        tokens: [1],
        promptTokens: [9, 8, 7],
        tokensPerSecond: 2,
        compileTimeMs: 5,
        firstTokenLatencyMs: timing.firstTokenLatencyMs,
        prefillMs: timing.prefillMs,
        exactHeadBackend: "cpu_fp16_tiled"
    )
    #expect(result.prefillMs == 80)
    #expect(result.ttftIncludingPrefillMs >= result.prefillMs)
    #expect(result.firstTokenLatencyMs == result.ttftIncludingPrefillMs)
    #expect(result.firstTokenLatencyMs == 101)
    let labeled = try result.withDecodePath("hybrid")
    #expect(labeled.prefillMs == result.prefillMs)
    #expect(labeled.firstTokenLatencyMs == result.firstTokenLatencyMs)
}

@Test func test_generationStep_published_ttft_includes_prefill() {
    let step = GenerationStep(
        token: 7,
        generatedTokens: [7],
        text: "hi",
        tokenLatencyMs: 21,
        elapsedMs: 21,
        firstTokenLatencyMs: 101,
        prefillMs: 80,
        tokensPerSecond: 2
    )
    #expect(step.prefillMs == 80)
    #expect(step.firstTokenLatencyMs >= step.prefillMs)
    #expect(step.firstTokenLatencyMs == 101)
}
