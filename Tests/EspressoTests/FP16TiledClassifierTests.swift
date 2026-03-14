import XCTest
@testable import Espresso
@testable import ANETypes

final class FP16TiledClassifierTests: XCTestCase {

    /// FP16 tiled classifier argmax must match naive FP32 sgemm argmax.
    func testFP16TiledArgmaxMatchesNaive() {
        let vocabSize = 200
        let dim = 32

        // Create classifier weights with known dominant row
        let classifierFP32 = TensorBuffer(count: vocabSize * dim, zeroed: true)
        let input = UnsafeMutablePointer<Float>.allocate(capacity: dim)
        let naiveLogits = UnsafeMutablePointer<Float>.allocate(capacity: vocabSize)
        defer {
            input.deallocate()
            naiveLogits.deallocate()
        }

        // Fill classifier
        classifierFP32.withUnsafeMutablePointer { ptr in
            for r in 0..<vocabSize {
                for c in 0..<dim {
                    ptr[r * dim + c] = Float(r) * 0.01 + Float(c) * 0.001
                }
            }
            // Make row 42 clearly dominant
            for c in 0..<dim {
                ptr[42 * dim + c] = 10.0
            }
        }

        // Input: all ones
        for i in 0..<dim { input[i] = 1.0 }

        // Naive argmax via manual dot products
        classifierFP32.withUnsafePointer { clsPtr in
            for r in 0..<vocabSize {
                var dot: Float = 0
                for c in 0..<dim {
                    dot += clsPtr[r * dim + c] * input[c]
                }
                naiveLogits[r] = dot
            }
        }
        var naiveBest: Float = naiveLogits[0]
        var naiveBestIdx = 0
        for i in 1..<vocabSize {
            if naiveLogits[i] > naiveBest {
                naiveBest = naiveLogits[i]
                naiveBestIdx = i
            }
        }
        XCTAssertEqual(naiveBestIdx, 42, "Row 42 should dominate in naive argmax")

        // Quantize to FP16
        let fp16Weights = TensorBufferFP16(quantizing: classifierFP32, rows: vocabSize, cols: dim)

        // FP16 tiled argmax
        let fp16ArgmaxIdx = fp16Weights.withUnsafePointer { fp16Ptr in
            FP16TiledClassifier.tiledMatvecArgmax(
                weights: fp16Ptr,
                input: UnsafePointer(input),
                vocabSize: vocabSize,
                dim: dim,
                tileRows: 50  // Small tile to exercise tiling
            )
        }

        XCTAssertEqual(fp16ArgmaxIdx, naiveBestIdx,
                       "FP16 tiled argmax must match naive FP32 argmax")
        XCTAssertEqual(fp16ArgmaxIdx, 42, "Row 42 should win")
    }

    /// FP16 tiled classifier correctness with a single tile (tileRows > vocabSize).
    func testSingleTileFP16Argmax() {
        let vocabSize = 10
        let dim = 8

        let classifierFP32 = TensorBuffer(count: vocabSize * dim, zeroed: true)
        let input = UnsafeMutablePointer<Float>.allocate(capacity: dim)
        defer { input.deallocate() }

        // Row 7 dominates
        classifierFP32.withUnsafeMutablePointer { ptr in
            for r in 0..<vocabSize {
                for c in 0..<dim {
                    ptr[r * dim + c] = r == 7 ? 5.0 : 0.1
                }
            }
        }
        for i in 0..<dim { input[i] = 1.0 }

        let fp16Weights = TensorBufferFP16(quantizing: classifierFP32, rows: vocabSize, cols: dim)

        let idx = fp16Weights.withUnsafePointer { fp16Ptr in
            FP16TiledClassifier.tiledMatvecArgmax(
                weights: fp16Ptr,
                input: UnsafePointer(input),
                vocabSize: vocabSize,
                dim: dim,
                tileRows: 1000  // Larger than vocab
            )
        }

        XCTAssertEqual(idx, 7)
    }
}
