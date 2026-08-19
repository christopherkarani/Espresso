import XCTest
import ANETypes
import ANERuntime
@testable import Espresso

final class QwenPrefillCompareTests: XCTestCase {
    func test_seq64_and_seq256_are_the_fixed_prefill_shapes() {
        XCTAssertEqual(QwenPrefillCompare.sequenceLengths, [64, 256])
        XCTAssertEqual(QwenPrefillCompare.sequenceLengths, HybridDecodeKernelSet.prefillSequenceLengths)
    }

    func test_compiled_prefill_qkv_one_eval_matches_n1_hybrid_steps() throws {
        guard ProcessInfo.processInfo.environment["ANE_HARDWARE_TESTS"] == "1" else {
            throw XCTSkip("Set ANE_HARDWARE_TESTS=1 to run seq=64/256 HybridDecodeKernelSet prefill vs N=1 eval")
        }
        let weights = makePrefillCompareWeights()
        for seq in QwenPrefillCompare.sequenceLengths {
            let result: QwenPrefillCompareResult
            do {
                result = try QwenPrefillCompare.compareHybridQKV(sequenceLength: seq, weights: weights)
            } catch {
                throw XCTSkip("ANE prefill vs N=1 eval unavailable for seq=\(seq): \(error)")
            }
            print(
                String(
                    format: "hybrid_prefill_compare seq=%d n1_ms=%.6f batched_ms=%.6f outputs_match=%@ n1_count=%d batched_count=%d tol=%.4f",
                    result.sequenceLength,
                    result.n1WallMs,
                    result.batchedWallMs,
                    result.outputsMatch ? "true" : "false",
                    result.n1Hidden.count,
                    result.batchedHidden.count,
                    result.matchTolerance
                )
            )
            XCTAssertEqual(result.sequenceLength, seq)
            XCTAssertEqual(result.n1Hidden.count, seq * weights.qDim)
            XCTAssertTrue(
                result.outputsMatch,
                "seq=\(seq) batched QKV hidden diverged from N=1 hybrid QKV evals"
            )
            XCTAssertGreaterThan(result.n1WallMs, 0)
            XCTAssertGreaterThan(result.batchedWallMs, 0)
        }
    }

    func test_cpu_n1_loop_matches_batched_gemm_on_same_prompt_ids() {
        for seq in QwenPrefillCompare.sequenceLengths {
            let dim = 8
            let out = 12
            var input = [Float](repeating: 0, count: seq * dim)
            var weights = [Float](repeating: 0, count: out * dim)
            for i in input.indices { input[i] = Float(i % 7) * 0.1 }
            for i in weights.indices { weights[i] = Float((i * 3) % 11) * 0.05 }
            let result = QwenPrefillCompare.compareCPUMatmul(
                sequenceLength: seq,
                dim: dim,
                outputChannels: out,
                input: input,
                weights: weights
            )
            print(
                String(
                    format: "prefill_compare seq=%d n1_ms=%.6f batched_ms=%.6f outputs_match=%@ n1_count=%d batched_count=%d",
                    result.sequenceLength,
                    result.n1WallMs,
                    result.batchedWallMs,
                    result.outputsMatch ? "true" : "false",
                    result.n1Hidden.count,
                    result.batchedHidden.count
                )
            )
            XCTAssertEqual(result.sequenceLength, seq)
            XCTAssertTrue(result.outputsMatch)
            XCTAssertGreaterThanOrEqual(result.n1WallMs, 0)
            XCTAssertGreaterThanOrEqual(result.batchedWallMs, 0)
        }
    }
}

private func makePrefillCompareWeights(value: Float = 0.01) -> LayerWeights {
    let weights = LayerWeights()
    func fill(_ buf: borrowing TensorBuffer, _ value: Float) {
        buf.withUnsafeMutableBufferPointer { ptr in
            for idx in ptr.indices {
                ptr[idx] = value
            }
        }
    }
    fill(weights.Wq, value)
    fill(weights.Wk, value)
    fill(weights.Wv, value)
    fill(weights.Wo, value)
    fill(weights.W1, value)
    fill(weights.W2, value)
    fill(weights.W3, value)
    fill(weights.rmsAtt, 1.0)
    fill(weights.rmsFfn, 1.0)
    return weights
}
