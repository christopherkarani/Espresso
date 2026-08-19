import XCTest
@testable import Espresso

final class PrefillAMXShareTests: XCTestCase {
    func test_tuner_does_not_copy_omlx_split_table_and_targets_batched_prefill() {
        let tuner = PrefillAMXShareTuner(aneChannelAlignment: 32)
        let decodeTrace = ChatChunkMixTrace(promptTokenCounts: [1, 2, 1, 3])
        let decodeDecision = tuner.decide(trace: decodeTrace, outputChannels: 100)
        print(
            "tuner_input decode_chunks=\(decodeTrace.promptTokenCounts) output_channels=100 median=\(decodeTrace.medianPromptTokens) decision_enabled=\(decodeDecision.enabled) leftover_frac=\(decodeDecision.leftoverChannelFraction) leftover_count=\(decodeDecision.leftoverChannelCount) target=\(decodeDecision.target)"
        )
        XCTAssertFalse(decodeDecision.enabled)
        XCTAssertEqual(decodeDecision.target, .batchedPrefillLeftoverChannels)
        XCTAssertNotEqual(decodeDecision.leftoverChannelFraction, 0.14)
        XCTAssertNotEqual(decodeDecision.leftoverChannelFraction, 0.20)
        XCTAssertNotEqual(decodeDecision.leftoverChannelFraction, 0.13)

        let chatTrace = ChatChunkMixTrace(promptTokenCounts: [51, 77, 138, 202, 256])
        let on = tuner.decide(trace: chatTrace, outputChannels: 100)
        print(
            "tuner_input chat_chunks=\(chatTrace.promptTokenCounts) output_channels=100 median=\(chatTrace.medianPromptTokens) decision_enabled=\(on.enabled) leftover_frac=\(on.leftoverChannelFraction) leftover_count=\(on.leftoverChannelCount) target=\(on.target)"
        )
        XCTAssertTrue(on.enabled)
        XCTAssertEqual(on.leftoverChannelCount, 4)
        XCTAssertEqual(on.target, .batchedPrefillLeftoverChannels)
        XCTAssertEqual(on.leftoverChannelFraction, 0.04, accuracy: 1e-12)
        XCTAssertNotEqual(on.leftoverChannelFraction, 0.14)
        XCTAssertNotEqual(on.leftoverChannelFraction, 0.20)
        XCTAssertNotEqual(on.leftoverChannelFraction, 0.13)

        let aligned = tuner.decide(trace: chatTrace, outputChannels: 96)
        print(
            "tuner_input aligned_chunks=\(chatTrace.promptTokenCounts) output_channels=96 decision_enabled=\(aligned.enabled) leftover_frac=\(aligned.leftoverChannelFraction) leftover_count=\(aligned.leftoverChannelCount) target=\(aligned.target)"
        )
        XCTAssertFalse(aligned.enabled)
    }

    func test_leftover_share_applies_to_batched_prefill_gemm_not_n1() {
        let seq = 64
        let dim = 8
        let out = 12
        var input = [Float](repeating: 0, count: seq * dim)
        var weights = [Float](repeating: 0, count: out * dim)
        for i in input.indices { input[i] = Float(i % 5) * 0.2 }
        for i in weights.indices { weights[i] = Float((i * 2) % 9) * 0.1 }

        let baseline = QwenPrefillCompare.compareCPUMatmul(
            sequenceLength: seq,
            dim: dim,
            outputChannels: out,
            input: input,
            weights: weights
        )
        let off = BatchedPrefillGEMM.multiply(
            sequenceLength: seq,
            dim: dim,
            outputChannels: out,
            input: input,
            weights: weights,
            share: .off
        )
        XCTAssertEqual(off.count, baseline.batchedHidden.count)
        for (lhs, rhs) in zip(off, baseline.batchedHidden) {
            XCTAssertEqual(lhs, rhs, accuracy: 1e-4)
        }

        let share = PrefillAMXShareDecision(
            enabled: true,
            leftoverChannelFraction: 4.0 / 12.0,
            leftoverChannelCount: 4,
            target: .batchedPrefillLeftoverChannels
        )
        XCTAssertEqual(share.target, .batchedPrefillLeftoverChannels)
        let split = BatchedPrefillGEMM.multiply(
            sequenceLength: seq,
            dim: dim,
            outputChannels: out,
            input: input,
            weights: weights,
            share: share
        )
        XCTAssertEqual(split.count, baseline.batchedHidden.count)
        for (lhs, rhs) in zip(split, baseline.batchedHidden) {
            XCTAssertEqual(lhs, rhs, accuracy: 1e-3)
        }
    }
}
