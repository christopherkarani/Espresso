import XCTest
@testable import Espresso
@testable import ANETypes

final class StreamingTwoTokenTests: XCTestCase {

    /// Helper: create zero-weight RecurrentGenerationWeights for identity-zero-trunk testing.
    private func makeZeroWeights(layerCount: Int) -> RecurrentGenerationWeights {
        let layers = LayerStorage<RWKVStyleRecurrentWeights>(count: layerCount) { _ in
            let w = RWKVStyleRecurrentWeights()
            // Zero all weights for identity-zero-trunk validation
            w.rms.zero()
            w.Wx.zero()
            w.Ws.zero()
            w.Wd.zero()
            w.Wo.zero()
            return w
        }

        let rmsFinal = TensorBuffer(count: ModelConfig.dim, zeroed: true)
        // rmsFinal = all ones → RMSNorm becomes identity-like
        rmsFinal.withUnsafeMutablePointer { ptr in
            for i in 0..<ModelConfig.dim { ptr[i] = 1.0 }
        }

        let embeddingCount = ModelConfig.vocab * ModelConfig.dim
        let embedding = TensorBuffer(count: embeddingCount, zeroed: true)
        // Fill embedding with recognizable per-row pattern
        embedding.withUnsafeMutablePointer { ptr in
            for r in 0..<ModelConfig.vocab {
                for c in 0..<ModelConfig.dim {
                    ptr[r * ModelConfig.dim + c] = Float(r) * 0.001 + Float(c) * 0.0001
                }
            }
        }

        let classifier = TensorBuffer(count: 0, zeroed: true)

        return RecurrentGenerationWeights(
            layers: layers,
            rmsFinal: rmsFinal,
            embedding: embedding,
            classifier: classifier,
            sharedClassifier: true,
            vocabSize: ModelConfig.vocab
        )
    }

    /// Streaming two-token decode must produce the same tokens as two individual decode calls.
    func testStreamingTwoTokenMatchesSequential() throws {
        let layerCount = 2
        let weights = makeZeroWeights(layerCount: layerCount)

        // Create two identical models
        var modelA = try ANERecurrentGenerationModel(
            weights: weights,
            layerCount: layerCount,
            outputHeadBackend: .cpu,
            trunkBackend: .identityZeroTrunk
        )
        var modelB = try ANERecurrentGenerationModel(
            weights: weights,
            layerCount: layerCount,
            outputHeadBackend: .cpu,
            trunkBackend: .identityZeroTrunk
        )

        let startToken: UInt16 = 42

        // Sequential: two individual decodes
        let tok1_seq = try modelA.decodeSelectedToken(nextToken: startToken, strategy: .argmax)
        let tok2_seq = try modelA.decodeSelectedToken(nextToken: tok1_seq, strategy: .argmax)

        // Streaming: two-token batch
        let result = try modelB.decodeSelectedTwoTokensStreaming(
            inputToken: startToken,
            strategy: .argmax
        )

        XCTAssertEqual(result.token1, tok1_seq,
                       "Streaming token1 must match sequential token1")
        XCTAssertEqual(result.token2, tok2_seq,
                       "Streaming token2 must match sequential token2")
        XCTAssertGreaterThan(result.token1TrunkMs, 0,
                             "Trunk latency must be recorded")
        XCTAssertGreaterThan(result.token1ClassifierMs, 0,
                             "Classifier latency must be recorded")
    }

    /// Streaming two-token result has reasonable timing fields.
    func testStreamingTwoTokenTimingFields() throws {
        let layerCount = 2
        let weights = makeZeroWeights(layerCount: layerCount)

        var model = try ANERecurrentGenerationModel(
            weights: weights,
            layerCount: layerCount,
            outputHeadBackend: .cpu,
            trunkBackend: .identityZeroTrunk
        )

        let result = try model.decodeSelectedTwoTokensStreaming(
            inputToken: 10,
            strategy: .argmax
        )

        // Total should be sum of parts
        let expectedTotal = result.token1LatencyMs + result.token2LatencyMs
        XCTAssertEqual(result.totalLatencyMs, expectedTotal, accuracy: 1e-6,
                       "Total latency must be sum of token1 and token2 latencies")

        // Sequential baseline: speculation always false
        XCTAssertFalse(result.speculationHit,
                       "Sequential baseline should report no speculation hit")
    }
}
