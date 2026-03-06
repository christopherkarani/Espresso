import XCTest
@testable import Espresso

private struct FakeVerifyResponse: Equatable {
    let sequenceTokens: [UInt16]
    let startIndex: Int
    let logits: [[Float]]
}

private struct FakeGenerationModel: AutoregressiveLanguageModel {
    let vocabSize: Int
    var prefillLogitsQueue: [[Float]]
    var decodeLogitsQueue: [[Float]]
    var verifyResponses: [FakeVerifyResponse]

    private(set) var resetCount: Int = 0
    private(set) var prefillCalls: [[UInt16]] = []
    private(set) var decodeCalls: [UInt16] = []
    private(set) var verifyCalls: [(sequenceTokens: [UInt16], startIndex: Int)] = []

    mutating func reset() throws(GenerationError) {
        resetCount += 1
    }

    mutating func prefill(promptTokens: [UInt16]) throws(GenerationError) -> [Float] {
        prefillCalls.append(promptTokens)
        guard !prefillLogitsQueue.isEmpty else {
            throw .invalidArguments("missing fake prefill logits")
        }
        return prefillLogitsQueue.removeFirst()
    }

    mutating func decode(nextToken: UInt16) throws(GenerationError) -> [Float] {
        decodeCalls.append(nextToken)
        guard !decodeLogitsQueue.isEmpty else {
            throw .invalidArguments("missing fake decode logits")
        }
        return decodeLogitsQueue.removeFirst()
    }

    mutating func verify(sequenceTokens: [UInt16], startIndex: Int) throws(GenerationError) -> [[Float]] {
        verifyCalls.append((sequenceTokens, startIndex))
        guard !verifyResponses.isEmpty else {
            throw .invalidArguments("missing fake verify response")
        }
        let response = verifyResponses.removeFirst()
        XCTAssertEqual(response.sequenceTokens, sequenceTokens)
        XCTAssertEqual(response.startIndex, startIndex)
        return response.logits
    }
}

final class GenerationHarnessTests: XCTestCase {
    func test_autoregressive_harness_prefills_then_decodes_argmax_tokens() throws {
        let model = FakeGenerationModel(
            vocabSize: 3,
            prefillLogitsQueue: [
                [0.1, 0.9, 0.0],
            ],
            decodeLogitsQueue: [
                [0.1, 0.2, 0.8],
                [0.7, 0.2, 0.1],
                [0.6, 0.3, 0.1],
            ],
            verifyResponses: []
        )

        var harness = AutoregressiveGenerationHarness(
            model: model,
            strategy: .argmax
        )

        let trace = try harness.generate(
            promptTokens: [7, 8],
            maxNewTokens: 3
        )

        XCTAssertEqual(trace.promptTokens, [7, 8])
        XCTAssertEqual(trace.generatedTokens, [1, 2, 0])
        XCTAssertEqual(trace.decodeLatenciesMs.count, 3)
        XCTAssertEqual(harness.model.resetCount, 1)
        XCTAssertEqual(harness.model.prefillCalls, [[7, 8]])
        XCTAssertEqual(harness.model.decodeCalls, [1, 2, 0])
        XCTAssertGreaterThanOrEqual(trace.prefillLatencyMs, 0)
        XCTAssertGreaterThan(trace.tokensPerSecond, 0)
        XCTAssertGreaterThan(trace.totalLatencyMs, 0)
    }

    func test_speculative_harness_tracks_acceptance_and_correction_tokens() throws {
        let draft = FakeGenerationModel(
            vocabSize: 5,
            prefillLogitsQueue: [
                [0.0, 1.0, 0.0, 0.0, 0.0],
            ],
            decodeLogitsQueue: [
                [0.0, 0.0, 1.0, 0.0, 0.0],
            ],
            verifyResponses: []
        )
        let full = FakeGenerationModel(
            vocabSize: 5,
            prefillLogitsQueue: [
                [0.0, 1.0, 0.0, 0.0, 0.0],
            ],
            decodeLogitsQueue: [
                [0.0, 0.0, 0.0, 1.0, 0.0],
                [0.1, 0.2, 0.3, 0.4, 0.5],
            ],
            verifyResponses: [
                FakeVerifyResponse(
                    sequenceTokens: [9, 1, 2],
                    startIndex: 0,
                    logits: [
                        [0.0, 1.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 1.0],
                        [0.1, 0.2, 0.3, 0.4, 0.5],
                    ]
                ),
            ]
        )

        var harness = SpeculativeGenerationHarness(
            draftModel: draft,
            fullModel: full,
            strategy: .argmax,
            candidateCount: 2
        )

        let trace = try harness.generate(
            promptTokens: [9],
            maxNewTokens: 2
        )

        XCTAssertEqual(trace.promptTokens, [9])
        XCTAssertEqual(trace.generatedTokens, [1, 4])
        XCTAssertEqual(trace.acceptedPrefixLengths, [1])
        XCTAssertEqual(trace.totalDraftCandidates, 2)
        XCTAssertEqual(trace.totalAcceptedCandidates, 1)
        XCTAssertEqual(trace.acceptanceRate, 0.5, accuracy: 1e-6)
        XCTAssertEqual(harness.draftModel.prefillCalls, [[9]])
        XCTAssertEqual(harness.draftModel.decodeCalls, [1])
        XCTAssertEqual(harness.fullModel.prefillCalls, [[9]])
        XCTAssertEqual(harness.fullModel.decodeCalls, [1, 4])
        XCTAssertEqual(harness.fullModel.verifyCalls.count, 1)
        XCTAssertGreaterThan(trace.effectiveTokensPerSecond, 0)
        XCTAssertGreaterThan(trace.totalLatencyMs, 0)
    }
}
