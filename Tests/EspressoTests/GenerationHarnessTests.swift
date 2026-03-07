import XCTest
import ANETypes
@testable import Espresso

private func fillGenerationTestBuffer(_ buffer: borrowing TensorBuffer, value: Float) {
    buffer.withUnsafeMutableBufferPointer { ptr in
        for idx in ptr.indices {
            ptr[idx] = value
        }
    }
}

private func makeGenerationTestRecurrentWeights(layerCount: Int) -> RecurrentGenerationWeights {
    let layers = LayerStorage<RWKVStyleRecurrentWeights>(count: layerCount) { _ in
        let weights = RWKVStyleRecurrentWeights()
        fillGenerationTestBuffer(weights.rms, value: 1)
        fillGenerationTestBuffer(weights.Wx, value: 0)
        fillGenerationTestBuffer(weights.Ws, value: 0)
        fillGenerationTestBuffer(weights.Wd, value: 0)
        fillGenerationTestBuffer(weights.Wo, value: 0)
        return weights
    }

    let rmsFinal = TensorBuffer(count: ModelConfig.dim, zeroed: false)
    fillGenerationTestBuffer(rmsFinal, value: 1)

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

private struct FakeFastSelectionModel: DirectTokenSelectingLanguageModel {
    let vocabSize: Int
    let selectedPrefillToken: UInt16
    let selectedDecodeTokens: [UInt16]

    private(set) var resetCount: Int = 0
    private(set) var prefillCalls: [[UInt16]] = []
    private(set) var decodeCalls: [UInt16] = []
    private(set) var prefillSelectedCalls: [[UInt16]] = []
    private(set) var decodeSelectedCalls: [UInt16] = []

    mutating func reset() throws(GenerationError) {
        resetCount += 1
    }

    mutating func prefill(promptTokens: [UInt16]) throws(GenerationError) -> [Float] {
        prefillCalls.append(promptTokens)
        return [0, 1, 0, 0].map(Float.init)
    }

    mutating func decode(nextToken: UInt16) throws(GenerationError) -> [Float] {
        decodeCalls.append(nextToken)
        return [1, 0, 0, 0].map(Float.init)
    }

    mutating func verify(sequenceTokens: [UInt16], startIndex: Int) throws(GenerationError) -> [[Float]] {
        []
    }

    mutating func prefillSelectedToken(
        promptTokens: [UInt16],
        strategy: TokenSelectionStrategy
    ) throws(GenerationError) -> UInt16 {
        prefillSelectedCalls.append(promptTokens)
        return selectedPrefillToken
    }

    mutating func decodeSelectedToken(
        nextToken: UInt16,
        strategy: TokenSelectionStrategy
    ) throws(GenerationError) -> UInt16 {
        decodeSelectedCalls.append(nextToken)
        let idx = decodeSelectedCalls.count - 1
        return selectedDecodeTokens[idx]
    }
}

final class GenerationHarnessTests: XCTestCase {
    func test_generation_performance_snapshot_reports_total_runtime() {
        let snapshot = GenerationPerformanceSnapshot(
            compileTimeMs: 12.5,
            trunkLatencyMs: 3.25,
            logitsLatencyMs: 1.75
        )

        XCTAssertEqual(snapshot.totalRuntimeMs, 5.0, accuracy: 1e-9)
    }

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

    func test_direct_token_selection_harness_uses_model_fast_selection() throws {
        let model = FakeFastSelectionModel(
            vocabSize: 4,
            selectedPrefillToken: 2,
            selectedDecodeTokens: [1, 3, 0]
        )

        var harness = DirectTokenSelectionGenerationHarness(model: model, strategy: .argmax)

        let trace = try harness.generate(promptTokens: [4, 5], maxNewTokens: 3)

        XCTAssertEqual(trace.generatedTokens, [2, 1, 3])
        XCTAssertEqual(harness.model.prefillSelectedCalls, [[4, 5]])
        XCTAssertEqual(harness.model.decodeSelectedCalls, [2, 1, 3])
        XCTAssertTrue(harness.model.prefillCalls.isEmpty)
        XCTAssertTrue(harness.model.decodeCalls.isEmpty)
    }

    func test_autoregressive_harness_materializes_logits_for_fast_selection_model() throws {
        let model = FakeFastSelectionModel(
            vocabSize: 4,
            selectedPrefillToken: 2,
            selectedDecodeTokens: [1, 3, 0]
        )

        var harness = AutoregressiveGenerationHarness(model: model, strategy: .argmax)

        let trace = try harness.generate(promptTokens: [4, 5], maxNewTokens: 2)

        XCTAssertEqual(trace.generatedTokens, [1, 0])
        XCTAssertEqual(harness.model.prefillCalls, [[4, 5]])
        XCTAssertEqual(harness.model.decodeCalls, [1, 0])
        XCTAssertTrue(harness.model.prefillSelectedCalls.isEmpty)
        XCTAssertTrue(harness.model.decodeSelectedCalls.isEmpty)
    }

    func test_recurrent_generation_rejects_odd_layer_count_for_fused_pair_backend() {
        let weights = makeGenerationTestRecurrentWeights(layerCount: 3)

        do {
            _ = try ANERecurrentGenerationModel(
                weights: weights,
                layerCount: 3,
                maxSequenceTokens: 32,
                outputHeadBackend: .cpu,
                trunkBackend: .fusedTwoLayerPairs
            )
            XCTFail("Expected odd fused-pair layer count to throw")
        } catch {
            XCTAssertEqual(error, .invalidArguments("fused recurrent trunk backend requires an even layerCount"))
        }
    }

    func test_recurrent_generation_rejects_non_positive_trunk_lane_spatial() {
        let weights = makeGenerationTestRecurrentWeights(layerCount: 2)

        do {
            _ = try ANERecurrentGenerationModel(
                weights: weights,
                layerCount: 2,
                maxSequenceTokens: 32,
                outputHeadBackend: .cpu,
                trunkBackend: .fusedTwoLayerPairs,
                trunkLaneSpatial: 0
            )
            XCTFail("Expected non-positive trunk laneSpatial to throw")
        } catch {
            XCTAssertEqual(
                error,
                .invalidArguments("recurrent trunk laneSpatial must be > 0")
            )
        }
    }

    func test_recurrent_generation_rejects_non_multiple_of_three_for_fused_triplet_backend() {
        let weights = makeGenerationTestRecurrentWeights(layerCount: 5)

        do {
            _ = try ANERecurrentGenerationModel(
                weights: weights,
                layerCount: 5,
                maxSequenceTokens: 32,
                outputHeadBackend: .cpu,
                trunkBackend: .fusedThreeLayerTriplets
            )
            XCTFail("Expected non-multiple-of-three fused-triplet layer count to throw")
        } catch {
            XCTAssertEqual(
                error,
                .invalidArguments("fused three-layer recurrent trunk backend requires a layerCount that is a multiple of 3")
            )
        }
    }

    func test_recurrent_generation_rejects_non_positive_output_head_lane_spatial() {
        let weights = makeGenerationTestRecurrentWeights(layerCount: 3)

        do {
            _ = try ANERecurrentGenerationModel(
                weights: weights,
                layerCount: 3,
                maxSequenceTokens: 32,
                outputHeadBackend: .aneRMSNormClassifier,
                trunkBackend: .fusedThreeLayerTriplets,
                outputHeadLaneSpatial: 0
            )
            XCTFail("Expected non-positive output-head laneSpatial to throw")
        } catch {
            XCTAssertEqual(
                error,
                .invalidArguments("generation output-head laneSpatial must be > 0")
            )
        }
    }
}
