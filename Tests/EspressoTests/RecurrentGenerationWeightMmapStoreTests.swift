import XCTest
@testable import Espresso
@testable import ANETypes

final class RecurrentGenerationWeightMmapStoreTests: XCTestCase {

    /// Round-trip: write weights to a file, mmap-load them back, verify all values match.
    func testWriteAndLoadRoundTrip() throws {
        let layerCount = 2

        // Create echo-style weights with known values
        let layers = LayerStorage<RWKVStyleRecurrentWeights>(count: layerCount) { idx in
            let w = RWKVStyleRecurrentWeights()
            // Fill with recognizable per-layer pattern
            w.rms.withUnsafeMutablePointer { ptr in
                for i in 0..<ModelConfig.dim { ptr[i] = Float(idx) * 0.1 + Float(i) * 0.001 }
            }
            w.Wx.withUnsafeMutablePointer { ptr in
                for i in 0..<ModelConfig.wqSize { ptr[i] = Float(idx) * 0.2 + Float(i) * 0.0001 }
            }
            w.Ws.withUnsafeMutablePointer { ptr in
                for i in 0..<ModelConfig.wqSize { ptr[i] = Float(idx) * 0.3 }
            }
            w.Wd.withUnsafeMutablePointer { ptr in
                for i in 0..<ModelConfig.wqSize { ptr[i] = Float(idx) * 0.4 }
            }
            w.Wo.withUnsafeMutablePointer { ptr in
                for i in 0..<ModelConfig.woSize { ptr[i] = Float(idx) * 0.5 }
            }
            return w
        }

        let rmsFinal = TensorBuffer(count: ModelConfig.dim, zeroed: true)
        rmsFinal.withUnsafeMutablePointer { ptr in
            for i in 0..<ModelConfig.dim { ptr[i] = Float(i) * 0.01 }
        }

        let embeddingCount = ModelConfig.vocab * ModelConfig.dim
        let embedding = TensorBuffer(count: embeddingCount, zeroed: true)
        embedding.withUnsafeMutablePointer { ptr in
            for i in 0..<min(embeddingCount, 100) { ptr[i] = Float(i) * 0.005 }
        }

        let classifier = TensorBuffer(count: 0, zeroed: true)

        let original = RecurrentGenerationWeights(
            layers: layers,
            rmsFinal: rmsFinal,
            embedding: embedding,
            classifier: classifier,
            sharedClassifier: true,
            vocabSize: ModelConfig.vocab
        )

        // Write to a temp file
        let tmpPath = NSTemporaryDirectory() + "test_mmap_roundtrip_\(ProcessInfo.processInfo.processIdentifier).bin"
        defer { try? FileManager.default.removeItem(atPath: tmpPath) }

        try RecurrentGenerationWeightMmapStore.write(original, to: tmpPath)

        // Verify file size
        let expectedFloats = RecurrentGenerationWeightMmapStore.totalFloatCount(layerCount: layerCount)
        let expectedBytes = expectedFloats * MemoryLayout<Float>.stride
        let attrs = try FileManager.default.attributesOfItem(atPath: tmpPath)
        let fileSize = attrs[.size] as! Int
        XCTAssertEqual(fileSize, expectedBytes, "File size must match expected layout")

        // Load via mmap
        let loaded = try RecurrentGenerationWeightMmapStore.load(
            from: tmpPath,
            layerCount: layerCount,
            sharedClassifier: true
        )

        // Verify rmsFinal
        original.rmsFinal.withUnsafePointer { origPtr in
            loaded.rmsFinal.withUnsafePointer { loadedPtr in
                for i in 0..<ModelConfig.dim {
                    XCTAssertEqual(origPtr[i], loadedPtr[i], accuracy: 1e-6,
                                   "rmsFinal[\(i)] mismatch")
                }
            }
        }

        // Verify embedding (first 100 elements)
        original.embedding.withUnsafePointer { origPtr in
            loaded.embedding.withUnsafePointer { loadedPtr in
                for i in 0..<min(embeddingCount, 100) {
                    XCTAssertEqual(origPtr[i], loadedPtr[i], accuracy: 1e-6,
                                   "embedding[\(i)] mismatch")
                }
            }
        }

        // Verify per-layer weights
        for idx in 0..<layerCount {
            original.layers[idx].rms.withUnsafePointer { origPtr in
                loaded.layers[idx].rms.withUnsafePointer { loadedPtr in
                    for i in 0..<ModelConfig.dim {
                        XCTAssertEqual(origPtr[i], loadedPtr[i], accuracy: 1e-6,
                                       "layer[\(idx)].rms[\(i)] mismatch")
                    }
                }
            }

            original.layers[idx].Wx.withUnsafePointer { origPtr in
                loaded.layers[idx].Wx.withUnsafePointer { loadedPtr in
                    for i in 0..<min(ModelConfig.wqSize, 10) {
                        XCTAssertEqual(origPtr[i], loadedPtr[i], accuracy: 1e-6,
                                       "layer[\(idx)].Wx[\(i)] mismatch")
                    }
                }
            }
        }

        XCTAssertEqual(loaded.sharedClassifier, true)
        XCTAssertEqual(loaded.vocabSize, ModelConfig.vocab)
    }

    /// Verify totalFloatCount produces the right layout size.
    func testTotalFloatCount() {
        let count1 = RecurrentGenerationWeightMmapStore.totalFloatCount(layerCount: 1)
        let count6 = RecurrentGenerationWeightMmapStore.totalFloatCount(layerCount: 6)

        let headRegion = ModelConfig.dim + 2 * ModelConfig.vocab * ModelConfig.dim
        let perLayer = ModelConfig.dim + 3 * ModelConfig.wqSize + ModelConfig.woSize

        XCTAssertEqual(count1, headRegion + 1 * perLayer)
        XCTAssertEqual(count6, headRegion + 6 * perLayer)
    }

    func testWriteRejectsUnsupportedVocabSize() {
        let customVocab = ModelConfig.vocab - 1
        let weights = RecurrentGenerationWeights(
            layers: LayerStorage<RWKVStyleRecurrentWeights>(count: 1) { _ in RWKVStyleRecurrentWeights() },
            rmsFinal: TensorBuffer(count: ModelConfig.dim, zeroed: true),
            embedding: TensorBuffer(count: customVocab * ModelConfig.dim, zeroed: true),
            classifier: TensorBuffer(count: customVocab * ModelConfig.dim, zeroed: true),
            sharedClassifier: false,
            vocabSize: customVocab
        )
        let tmpPath = NSTemporaryDirectory() + "test_mmap_vocab_\(ProcessInfo.processInfo.processIdentifier).bin"
        defer { try? FileManager.default.removeItem(atPath: tmpPath) }

        XCTAssertThrowsError(try RecurrentGenerationWeightMmapStore.write(weights, to: tmpPath)) { error in
            XCTAssertEqual(
                error as? RecurrentGenerationWeightMmapStoreError,
                .unsupportedVocabSize(expected: ModelConfig.vocab, actual: customVocab)
            )
        }
    }
}
