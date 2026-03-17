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

private func makeGenerationTestNonZeroRecurrentWeights(layerCount: Int) -> RecurrentGenerationWeights {
    let weights = makeGenerationTestRecurrentWeights(layerCount: layerCount)
    weights.layers[0].Wo.withUnsafeMutablePointer { ptr in
        ptr[0] = 0.25
    }
    return weights
}

private func makeGenerationTestRowTensor(_ rows: [[Float]]) -> TensorBuffer {
    let buffer = TensorBuffer(count: rows.count * ModelConfig.dim, zeroed: true)
    buffer.withUnsafeMutablePointer { ptr in
        for rowIndex in 0..<rows.count {
            for dimIndex in 0..<rows[rowIndex].count {
                ptr[rowIndex * ModelConfig.dim + dimIndex] = rows[rowIndex][dimIndex]
            }
        }
    }
    return buffer
}

private struct FakeVerifyResponse: Equatable {
    let sequenceTokens: [TokenID]
    let startIndex: Int
    let logits: [[Float]]
}

private struct FakeGenerationModel: AutoregressiveLanguageModel {
    let vocabSize: Int
    var prefillLogitsQueue: [[Float]]
    var decodeLogitsQueue: [[Float]]
    var verifyResponses: [FakeVerifyResponse]

    private(set) var resetCount: Int = 0
    private(set) var prefillCalls: [[TokenID]] = []
    private(set) var decodeCalls: [TokenID] = []
    private(set) var verifyCalls: [(sequenceTokens: [TokenID], startIndex: Int)] = []

    mutating func reset() throws(GenerationError) {
        resetCount += 1
    }

    mutating func prefill(promptTokens: [TokenID]) throws(GenerationError) -> [Float] {
        prefillCalls.append(promptTokens)
        guard !prefillLogitsQueue.isEmpty else {
            throw .invalidArguments("missing fake prefill logits")
        }
        return prefillLogitsQueue.removeFirst()
    }

    mutating func decode(nextToken: TokenID) throws(GenerationError) -> [Float] {
        decodeCalls.append(nextToken)
        guard !decodeLogitsQueue.isEmpty else {
            throw .invalidArguments("missing fake decode logits")
        }
        return decodeLogitsQueue.removeFirst()
    }

    mutating func verify(sequenceTokens: [TokenID], startIndex: Int) throws(GenerationError) -> [[Float]] {
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
    let selectedPrefillToken: TokenID
    let selectedDecodeTokens: [TokenID]

    private(set) var resetCount: Int = 0
    private(set) var prefillCalls: [[TokenID]] = []
    private(set) var decodeCalls: [TokenID] = []
    private(set) var prefillSelectedCalls: [[TokenID]] = []
    private(set) var decodeSelectedCalls: [TokenID] = []

    mutating func reset() throws(GenerationError) {
        resetCount += 1
    }

    mutating func prefill(promptTokens: [TokenID]) throws(GenerationError) -> [Float] {
        prefillCalls.append(promptTokens)
        return [0, 1, 0, 0].map(Float.init)
    }

    mutating func decode(nextToken: TokenID) throws(GenerationError) -> [Float] {
        decodeCalls.append(nextToken)
        return [1, 0, 0, 0].map(Float.init)
    }

    mutating func verify(sequenceTokens: [TokenID], startIndex: Int) throws(GenerationError) -> [[Float]] {
        []
    }

    mutating func prefillSelectedToken(
        promptTokens: [TokenID],
        strategy: TokenSelectionStrategy
    ) throws(GenerationError) -> TokenID {
        prefillSelectedCalls.append(promptTokens)
        return selectedPrefillToken
    }

    mutating func decodeSelectedToken(
        nextToken: TokenID,
        strategy: TokenSelectionStrategy
    ) throws(GenerationError) -> TokenID {
        decodeSelectedCalls.append(nextToken)
        let idx = decodeSelectedCalls.count - 1
        return selectedDecodeTokens[idx]
    }
}

private struct FakeTwoTokenDraftModel: TwoTokenDraftingLanguageModel {
    var selectedPrefillToken: TokenID
    var proposalQueue: [[TokenID]]

    private(set) var resetCount: Int = 0
    private(set) var prefillSelectedCalls: [[TokenID]] = []
    private(set) var commitCalls: [[TokenID]] = []

    mutating func reset() throws(GenerationError) {
        resetCount += 1
    }

    mutating func prefillSelectedToken(
        promptTokens: [TokenID],
        strategy: TokenSelectionStrategy
    ) throws(GenerationError) -> TokenID {
        prefillSelectedCalls.append(promptTokens)
        return selectedPrefillToken
    }

    mutating func proposeTwoTokens(strategy: TokenSelectionStrategy) throws(GenerationError) -> [TokenID] {
        guard !proposalQueue.isEmpty else {
            throw .invalidArguments("missing fake two-token proposal")
        }
        return proposalQueue.removeFirst()
    }

    mutating func commit(tokens: [TokenID]) throws(GenerationError) {
        commitCalls.append(tokens)
    }
}

private struct FakeTwoTokenVerifierModel: TwoTokenBranchVerifyingLanguageModel {
    var selectedPrefillToken: TokenID
    var verificationQueue: [(proposed: [TokenID], result: TwoTokenBranchCommitResult)]

    private(set) var resetCount: Int = 0
    private(set) var prefillSelectedCalls: [[TokenID]] = []
    private(set) var verifyCalls: [[TokenID]] = []

    mutating func reset() throws(GenerationError) {
        resetCount += 1
    }

    mutating func prefillSelectedToken(
        promptTokens: [TokenID],
        strategy: TokenSelectionStrategy
    ) throws(GenerationError) -> TokenID {
        prefillSelectedCalls.append(promptTokens)
        return selectedPrefillToken
    }

    mutating func verifyAndCommit(
        proposedTokens: [TokenID],
        strategy: TokenSelectionStrategy
    ) throws(GenerationError) -> TwoTokenBranchCommitResult {
        verifyCalls.append(proposedTokens)
        guard !verificationQueue.isEmpty else {
            throw .invalidArguments("missing fake verification result")
        }
        let next = verificationQueue.removeFirst()
        XCTAssertEqual(next.proposed, proposedTokens)
        return next.result
    }
}

private struct FakeExactTwoTokenPassResponse: Equatable {
    let currentToken: TokenID
    let remainingTokenBudget: Int
    let result: ExactTwoTokenPassResult
}

private struct FakeExactTwoTokenModel: ExactTwoTokenGeneratingLanguageModel {
    let vocabSize: Int
    var selectedPrefillToken: TokenID
    var passQueue: [FakeExactTwoTokenPassResponse]

    var performanceSnapshot: GenerationPerformanceSnapshot {
        GenerationPerformanceSnapshot()
    }

    private(set) var resetCount: Int = 0
    private(set) var prefillSelectedCalls: [[TokenID]] = []
    private(set) var passCalls: [(currentToken: TokenID, remainingTokenBudget: Int)] = []

    mutating func reset() throws(GenerationError) {
        resetCount += 1
    }

    mutating func prefillSelectedToken(
        promptTokens: [TokenID],
        strategy: TokenSelectionStrategy
    ) throws(GenerationError) -> TokenID {
        prefillSelectedCalls.append(promptTokens)
        return selectedPrefillToken
    }

    mutating func performExactTwoTokenPass(
        currentToken: TokenID,
        remainingTokenBudget: Int,
        strategy: TokenSelectionStrategy
    ) throws(GenerationError) -> ExactTwoTokenPassResult {
        passCalls.append((currentToken, remainingTokenBudget))
        guard !passQueue.isEmpty else {
            throw .invalidArguments("missing fake exact two-token pass response")
        }
        let next = passQueue.removeFirst()
        XCTAssertEqual(next.currentToken, currentToken)
        XCTAssertEqual(next.remainingTokenBudget, remainingTokenBudget)
        return next.result
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

    func test_two_token_branch_commit_harness_commits_without_draft_reset_or_prefill_replay() throws {
        let draft = FakeTwoTokenDraftModel(
            selectedPrefillToken: 1,
            proposalQueue: [[2, 3]]
        )
        let full = FakeTwoTokenVerifierModel(
            selectedPrefillToken: 1,
            verificationQueue: [
                (
                    proposed: [2, 3],
                    result: TwoTokenBranchCommitResult(
                        acceptedPrefixLength: 1,
                        committedTokens: [2, 4]
                    )
                ),
            ]
        )

        var harness = TwoTokenBranchCommitGenerationHarness(
            draftModel: draft,
            fullModel: full,
            strategy: .argmax
        )

        let trace = try harness.generate(promptTokens: [9], maxNewTokens: 3)

        XCTAssertEqual(trace.generatedTokens, [1, 2, 4])
        XCTAssertEqual(trace.acceptedPrefixLengths, [1])
        XCTAssertEqual(harness.draftModel.resetCount, 1)
        XCTAssertEqual(harness.draftModel.prefillSelectedCalls, [[9]])
        XCTAssertEqual(harness.draftModel.commitCalls, [[1], [2, 4]])
        XCTAssertEqual(harness.fullModel.resetCount, 1)
        XCTAssertEqual(harness.fullModel.prefillSelectedCalls, [[9]])
        XCTAssertEqual(harness.fullModel.verifyCalls, [[2, 3]])
    }

    func test_two_token_branch_commit_harness_records_two_token_acceptance() throws {
        let draft = FakeTwoTokenDraftModel(
            selectedPrefillToken: 7,
            proposalQueue: [[8, 9]]
        )
        let full = FakeTwoTokenVerifierModel(
            selectedPrefillToken: 7,
            verificationQueue: [
                (
                    proposed: [8, 9],
                    result: TwoTokenBranchCommitResult(
                        acceptedPrefixLength: 2,
                        committedTokens: [8, 9]
                    )
                ),
            ]
        )

        var harness = TwoTokenBranchCommitGenerationHarness(
            draftModel: draft,
            fullModel: full,
            strategy: .argmax
        )

        let trace = try harness.generate(promptTokens: [4], maxNewTokens: 3)

        XCTAssertEqual(trace.generatedTokens, [7, 8, 9])
        XCTAssertEqual(trace.acceptedPrefixLengths, [2])
        XCTAssertEqual(harness.draftModel.commitCalls, [[7], [8, 9]])
        XCTAssertEqual(harness.fullModel.verifyCalls, [[8, 9]])
    }

    func test_exact_two_token_harness_tracks_pass_metrics_and_future_acceptance() throws {
        let model = FakeExactTwoTokenModel(
            vocabSize: 32,
            selectedPrefillToken: 5,
            passQueue: [
                FakeExactTwoTokenPassResponse(
                    currentToken: 5,
                    remainingTokenBudget: 3,
                    result: ExactTwoTokenPassResult(
                        committedTokens: [5, 6],
                        nextCurrentToken: 7,
                        metrics: ExactTwoTokenPassMetrics(
                            proposerLatencyMs: 0.05,
                            verifierTrunkLatencyMs: 1.10,
                            verifierLogitsLatencyMs: 0.90,
                            stateAdvanceLatencyMs: 1.15,
                            acceptedFutureTokenCount: 1,
                            committedExactTokenCount: 2
                        )
                    )
                ),
                FakeExactTwoTokenPassResponse(
                    currentToken: 7,
                    remainingTokenBudget: 1,
                    result: ExactTwoTokenPassResult(
                        committedTokens: [7],
                        nextCurrentToken: 8,
                        metrics: ExactTwoTokenPassMetrics(
                            proposerLatencyMs: 0.02,
                            verifierTrunkLatencyMs: 1.05,
                            verifierLogitsLatencyMs: 0.88,
                            stateAdvanceLatencyMs: 0,
                            acceptedFutureTokenCount: 0,
                            committedExactTokenCount: 1
                        )
                    )
                ),
            ]
        )

        var harness = ExactTwoTokenGenerationHarness(model: model, strategy: .argmax)

        let trace = try harness.generate(promptTokens: [9], maxNewTokens: 3)

        XCTAssertEqual(trace.generatedTokens, [5, 6, 7])
        XCTAssertEqual(trace.acceptedFutureTokenCounts, [1, 0])
        XCTAssertEqual(trace.committedExactTokenCounts, [2, 1])
        XCTAssertEqual(harness.model.resetCount, 1)
        XCTAssertEqual(harness.model.prefillSelectedCalls, [[9]])
        XCTAssertEqual(harness.model.passCalls.map { $0.currentToken }, [5, 7])
        XCTAssertEqual(harness.model.passCalls.map { $0.remainingTokenBudget }, [3, 1])
        XCTAssertEqual(trace.committedExactTokensPerPass, 1.5, accuracy: 1e-9)
        XCTAssertEqual(trace.acceptedFutureTokensPerPass, 0.5, accuracy: 1e-9)
    }

    func test_exact_two_token_harness_discards_state_advance_cost_on_future_rejection() throws {
        let model = FakeExactTwoTokenModel(
            vocabSize: 16,
            selectedPrefillToken: 2,
            passQueue: [
                FakeExactTwoTokenPassResponse(
                    currentToken: 2,
                    remainingTokenBudget: 2,
                    result: ExactTwoTokenPassResult(
                        committedTokens: [2],
                        nextCurrentToken: 4,
                        metrics: ExactTwoTokenPassMetrics(
                            proposerLatencyMs: 0.01,
                            verifierTrunkLatencyMs: 1.20,
                            verifierLogitsLatencyMs: 0.70,
                            stateAdvanceLatencyMs: 0,
                            acceptedFutureTokenCount: 0,
                            committedExactTokenCount: 1
                        )
                    )
                ),
                FakeExactTwoTokenPassResponse(
                    currentToken: 4,
                    remainingTokenBudget: 1,
                    result: ExactTwoTokenPassResult(
                        committedTokens: [4],
                        nextCurrentToken: 6,
                        metrics: ExactTwoTokenPassMetrics(
                            proposerLatencyMs: 0.01,
                            verifierTrunkLatencyMs: 1.25,
                            verifierLogitsLatencyMs: 0.75,
                            stateAdvanceLatencyMs: 0,
                            acceptedFutureTokenCount: 0,
                            committedExactTokenCount: 1
                        )
                    )
                ),
            ]
        )

        var harness = ExactTwoTokenGenerationHarness(model: model, strategy: .argmax)

        let trace = try harness.generate(promptTokens: [1], maxNewTokens: 2)

        XCTAssertEqual(trace.generatedTokens, [2, 4])
        XCTAssertEqual(trace.acceptedFutureTokenCounts, [0, 0])
        XCTAssertEqual(trace.committedExactTokenCounts, [1, 1])
        XCTAssertEqual(trace.passMetrics.map { $0.stateAdvanceLatencyMs }, [0, 0])
        XCTAssertEqual(trace.committedExactTokensPerPass, 1.0, accuracy: 1e-9)
        XCTAssertEqual(trace.acceptedFutureTokensPerPass, 0.0, accuracy: 1e-9)
    }

    func test_exact_two_token_branch_promotion_plan_accepts_matching_future_token() {
        let plan = ExactTwoTokenBranchPromotionPlan.make(
            currentToken: 5,
            proposedFutureToken: 6,
            exactNextToken: 6,
            exactFutureToken: 7,
            remainingTokenBudget: 2
        )

        XCTAssertEqual(plan.committedTokens, [5, 6])
        XCTAssertEqual(plan.nextCurrentToken, 7)
        XCTAssertEqual(plan.committedExactTokenCount, 2)
        XCTAssertEqual(plan.acceptedFutureTokenCount, 1)
        XCTAssertEqual(plan.promotedStepCount, 2)
    }

    func test_exact_two_token_branch_promotion_plan_rejects_mismatched_future_token() {
        let plan = ExactTwoTokenBranchPromotionPlan.make(
            currentToken: 5,
            proposedFutureToken: 9,
            exactNextToken: 6,
            exactFutureToken: 7,
            remainingTokenBudget: 2
        )

        XCTAssertEqual(plan.committedTokens, [5])
        XCTAssertEqual(plan.nextCurrentToken, 6)
        XCTAssertEqual(plan.committedExactTokenCount, 1)
        XCTAssertEqual(plan.acceptedFutureTokenCount, 0)
        XCTAssertEqual(plan.promotedStepCount, 1)
    }

    func test_exact_two_token_branch_state_promotion_rejects_odd_layer_count_for_fused_pair_backend() {
        let weights = makeGenerationTestRecurrentWeights(layerCount: 3)

        do {
            _ = try ANEExactTwoTokenBranchStatePromotionModel(
                weights: weights,
                layerCount: 3,
                maxSequenceTokens: 32,
                outputHeadBackend: .cpu,
                trunkBackend: .fusedTwoLayerPairs
            )
            XCTFail("Expected odd fused-pair layer count to throw")
        } catch {
            XCTAssertEqual(
                error,
                .invalidArguments("exact two-token fused pair trunk backend requires an even layerCount")
            )
        }
    }

    func test_exact_two_token_branch_state_promotion_rejects_non_multiple_of_three_for_fused_triplet_backend() {
        let weights = makeGenerationTestRecurrentWeights(layerCount: 5)

        do {
            _ = try ANEExactTwoTokenBranchStatePromotionModel(
                weights: weights,
                layerCount: 5,
                maxSequenceTokens: 32,
                outputHeadBackend: .cpu,
                trunkBackend: .fusedThreeLayerTriplets
            )
            XCTFail("Expected non-multiple-of-three fused-triplet exact two-token backend to throw")
        } catch {
            XCTAssertEqual(
                error,
                .invalidArguments("exact two-token fused three-layer trunk backend requires a layerCount that is a multiple of 3")
            )
        }
    }

    func test_exact_two_token_branch_state_promotion_rejects_non_positive_trunk_lane_spatial() {
        let weights = makeGenerationTestRecurrentWeights(layerCount: 2)

        do {
            _ = try ANEExactTwoTokenBranchStatePromotionModel(
                weights: weights,
                layerCount: 2,
                maxSequenceTokens: 32,
                outputHeadBackend: .cpu,
                trunkBackend: .singleLayer,
                trunkLaneSpatial: 0
            )
            XCTFail("Expected non-positive trunk laneSpatial to throw")
        } catch {
            XCTAssertEqual(
                error,
                .invalidArguments("two-step recurrent trunk laneSpatial must be > 0")
            )
        }
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

    func test_recurrent_generation_identity_zero_trunk_backend_rejects_non_zero_weights() {
        let weights = makeGenerationTestNonZeroRecurrentWeights(layerCount: 1)

        do {
            _ = try ANERecurrentGenerationModel(
                weights: weights,
                layerCount: 1,
                maxSequenceTokens: 32,
                outputHeadBackend: .cpu,
                trunkBackend: .identityZeroTrunk
            )
            XCTFail("Expected non-zero recurrent weights to be rejected by identity zero-trunk backend")
        } catch {
            XCTAssertEqual(
                error,
                .invalidArguments("identity zero-trunk backend requires all recurrent Wx/Ws/Wd/Wo weights to be zero")
            )
        }
    }

    func test_recurrent_generation_identity_zero_trunk_backend_matches_local_bigram_teacher_tokens() throws {
        let recurrentWeights = try LocalBigramArtifactBuilder.buildRecurrentWeights(
            tokens: [35, 105, 110, 116, 32, 105, 110, 116, 32],
            layerCount: 1,
            vocabSize: ModelConfig.vocab
        )
        let model = try ANERecurrentGenerationModel(
            weights: recurrentWeights,
            layerCount: 1,
            maxSequenceTokens: 32,
            outputHeadBackend: .cpu,
            trunkBackend: .identityZeroTrunk
        )
        var harness = DirectTokenSelectionGenerationHarness(model: model, strategy: .argmax)

        let trace = try harness.generate(promptTokens: [35], maxNewTokens: 4)

        XCTAssertEqual(trace.generatedTokens, [105, 110, 116, 32])
    }

    func test_recurrent_generation_cpu_then_ane_backend_preserves_cpu_fallback_contract() throws {
        let recurrentWeights = try LocalBigramArtifactBuilder.buildRecurrentWeights(
            tokens: [35, 105, 110, 116, 32, 105, 110, 116, 32],
            layerCount: 1,
            vocabSize: ModelConfig.vocab
        )
        let model = try ANERecurrentGenerationModel(
            weights: recurrentWeights,
            layerCount: 1,
            maxSequenceTokens: 32,
            outputHeadBackend: .cpuThenANE,
            trunkBackend: .identityZeroTrunk
        )
        var harness = DirectTokenSelectionGenerationHarness(model: model, strategy: .argmax)

        let trace = try harness.generate(promptTokens: [35], maxNewTokens: 4)

        XCTAssertEqual(trace.generatedTokens, [105, 110, 116, 32])
    }

    func test_exact_two_token_identity_zero_trunk_backend_matches_local_bigram_exact_contract() throws {
        let recurrentWeights = try LocalBigramArtifactBuilder.buildRecurrentWeights(
            tokens: [35, 105, 110, 116, 32, 105, 110, 116, 32],
            layerCount: 1,
            vocabSize: ModelConfig.vocab
        )
        let futureSidecar = try LocalBigramArtifactBuilder.buildFutureSidecar(
            tokens: [35, 105, 110, 116, 32, 105, 110, 116, 32],
            layerCount: 1,
            vocabSize: ModelConfig.vocab
        )
        let model = try ANEExactTwoTokenBranchStatePromotionModel(
            weights: recurrentWeights,
            futureSidecar: futureSidecar,
            layerCount: 1,
            maxSequenceTokens: 32,
            outputHeadBackend: .cpu,
            trunkBackend: .identityZeroTrunk
        )
        var harness = ExactTwoTokenGenerationHarness(model: model, strategy: .argmax)

        let trace = try harness.generate(promptTokens: [35], maxNewTokens: 4)

        XCTAssertEqual(trace.generatedTokens, [105, 110, 116, 32])
        XCTAssertEqual(trace.committedExactTokenCounts, [2, 2])
        XCTAssertEqual(trace.acceptedFutureTokenCounts, [1, 1])
        XCTAssertEqual(trace.committedExactTokensPerPass, 2, accuracy: 0.0001)
        XCTAssertEqual(trace.acceptedFutureTokensPerPass, 1, accuracy: 0.0001)
    }

    func test_exact_output_head_shard_summaries_bound_true_logits_per_shard() throws {
        let classifierRows: [[Float]] = [
            [1.0, 0.0],
            [0.6, 0.8],
            [-0.2, 1.1],
            [-0.8, 0.4],
        ]
        let normalizedInput: [Float] = [0.8, 0.6]

        let summaries = try ExactGenerationOutputHeadShardSummary.makeContiguousShards(
            classifierRows: classifierRows,
            shardSize: 2
        )

        XCTAssertEqual(summaries.count, 2)
        XCTAssertEqual(summaries[0].tokenOffset, 0)
        XCTAssertEqual(summaries[1].tokenOffset, 2)

        for summary in summaries {
            let bound = summary.upperBound(forNormalizedInput: normalizedInput)
            let end = summary.tokenOffset + summary.tokenCount
            for tokenIndex in summary.tokenOffset..<end {
                let row = classifierRows[tokenIndex]
                let logit = zip(row, normalizedInput).reduce(Float.zero) { partial, pair in
                    partial + pair.0 * pair.1
                }
                XCTAssertGreaterThanOrEqual(
                    bound + 1e-5,
                    logit,
                    "shard bound must dominate every true logit in shard \(summary.tokenOffset)"
                )
            }
        }
    }

    func test_exact_output_head_bound_search_prunes_only_safe_shards() throws {
        let normalizedInput: [Float] = [1.0, 0.0]
        let summaries = [
            ExactGenerationOutputHeadShardSummary(
                tokenOffset: 0,
                tokenCount: 2,
                center: [0.93, 0.00],
                radius: 0.04
            ),
            ExactGenerationOutputHeadShardSummary(
                tokenOffset: 2,
                tokenCount: 2,
                center: [0.98, 0.00],
                radius: 0.03
            ),
            ExactGenerationOutputHeadShardSummary(
                tokenOffset: 4,
                tokenCount: 2,
                center: [0.20, 0.00],
                radius: 0.10
            ),
        ]

        var evaluatedShardOffsets: [Int] = []
        let result = try ExactGenerationOutputHeadBoundSearch.selectGlobalBest(
            normalizedInput: normalizedInput,
            shardSummaries: summaries
        ) { summary in
            evaluatedShardOffsets.append(summary.tokenOffset)
            switch summary.tokenOffset {
            case 0:
                return (token: 1, score: 0.91)
            case 2:
                return (token: 2, score: 0.96)
            case 4:
                XCTFail("search should prune shard 4 once the exact best score beats its bound")
                return (token: 4, score: 0.20)
            default:
                XCTFail("unexpected shard offset \(summary.tokenOffset)")
                return (token: 0, score: -.infinity)
            }
        }

        XCTAssertEqual(result.token, 2)
        XCTAssertEqual(result.score, 0.96, accuracy: 1e-6)
        XCTAssertEqual(evaluatedShardOffsets, [2, 0])
        XCTAssertEqual(result.evaluatedShardOffsets, [2, 0])
        XCTAssertEqual(result.prunedShardOffsets, [4])
    }

    func test_cpu_exact_clustered_head_matches_dense_argmax_for_non_contiguous_clusters() throws {
        let cluster0 = ExactGenerationOutputHeadCluster(
            summary: ExactGenerationOutputHeadShardSummary(
                tokenOffset: 0,
                tokenCount: 2,
                center: [0.75, 0.25] + Array(repeating: 0, count: ModelConfig.dim - 2),
                radius: 0.35355338
            ),
            tokenIndices: [0, 2],
            weights: makeGenerationTestRowTensor([
                [0.5, 0.5],
                [1.0, 0.0],
            ])
        )
        let cluster1 = ExactGenerationOutputHeadCluster(
            summary: ExactGenerationOutputHeadShardSummary(
                tokenOffset: 1,
                tokenCount: 2,
                center: [-0.25, -0.25] + Array(repeating: 0, count: ModelConfig.dim - 2),
                radius: 0.35355338
            ),
            tokenIndices: [1, 3],
            weights: makeGenerationTestRowTensor([
                [-0.5, 0.0],
                [0.0, -0.5],
            ])
        )
        let head = try CPUStagedExactGenerationOutputHead(
            vocabSize: 4,
            layoutStrategy: .clustered(clusterCount: 2, projectionDimensionCount: 2, iterations: 1),
            clusters: [cluster0, cluster1]
        )
        let normalizedInput = TensorBuffer(count: ModelConfig.dim, zeroed: true)
        normalizedInput.withUnsafeMutablePointer { ptr in
            ptr[0] = 1
        }

        let token = try head.selectArgmax(normalizedInput: normalizedInput)

        XCTAssertEqual(token, 2)
        XCTAssertEqual(head.lastEvaluatedShardCount, 1)
    }
}
