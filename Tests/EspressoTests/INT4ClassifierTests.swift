import XCTest
@testable import Espresso
@testable import ANETypes

final class INT4ClassifierTests: XCTestCase {

    /// INT4 quantized classifier argmax must match naive FP32 argmax for a well-separated matrix.
    func testINT4ArgmaxMatchesNaive() {
        let vocabSize = 100
        let dim = 32

        let classifierFP32 = TensorBuffer(count: vocabSize * dim, zeroed: true)
        let input = UnsafeMutablePointer<Float>.allocate(capacity: dim)
        let naiveLogits = UnsafeMutablePointer<Float>.allocate(capacity: vocabSize)
        defer {
            input.deallocate()
            naiveLogits.deallocate()
        }

        // Fill classifier with well-separated rows
        classifierFP32.withUnsafeMutablePointer { ptr in
            for r in 0..<vocabSize {
                for c in 0..<dim {
                    ptr[r * dim + c] = Float(r) * 0.01 + Float(c) * 0.001
                }
            }
            // Make row 73 clearly dominant
            for c in 0..<dim {
                ptr[73 * dim + c] = 10.0
            }
        }

        // Input: all ones
        for i in 0..<dim { input[i] = 1.0 }

        // Naive argmax via dot products
        classifierFP32.withUnsafePointer { clsPtr in
            for r in 0..<vocabSize {
                var dot: Float = 0
                for c in 0..<dim {
                    dot += clsPtr[r * dim + c] * input[c]
                }
                naiveLogits[r] = dot
            }
        }
        var naiveBestIdx = 0
        var naiveBest: Float = naiveLogits[0]
        for i in 1..<vocabSize {
            if naiveLogits[i] > naiveBest {
                naiveBest = naiveLogits[i]
                naiveBestIdx = i
            }
        }
        XCTAssertEqual(naiveBestIdx, 73, "Row 73 should dominate in naive argmax")

        // Quantize to INT4
        let int4Weights = TensorBufferINT4(quantizing: classifierFP32, rows: vocabSize, cols: dim)

        // INT4 argmax
        let iSum = INT4Classifier.inputSum(UnsafePointer(input), dim: dim)
        let int4ArgmaxIdx = INT4Classifier.int4MatvecArgmax(
            weights: int4Weights,
            input: UnsafePointer(input),
            inputSum: iSum,
            vocabSize: vocabSize,
            dim: dim
        )

        XCTAssertEqual(Int(int4ArgmaxIdx), naiveBestIdx,
                       "INT4 argmax must match naive FP32 argmax for well-separated rows")
    }

    /// INT4 full matvec logits must be approximately correct relative to FP32.
    func testINT4MatvecApproximation() {
        let vocabSize = 20
        let dim = 16

        let classifierFP32 = TensorBuffer(count: vocabSize * dim, zeroed: true)
        let input = UnsafeMutablePointer<Float>.allocate(capacity: dim)
        let int4Output = UnsafeMutablePointer<Float>.allocate(capacity: vocabSize)
        defer {
            input.deallocate()
            int4Output.deallocate()
        }

        // Fill with moderate values
        classifierFP32.withUnsafeMutablePointer { ptr in
            for r in 0..<vocabSize {
                for c in 0..<dim {
                    ptr[r * dim + c] = Float(r * dim + c) * 0.01
                }
            }
        }
        for i in 0..<dim { input[i] = 1.0 }

        // Compute reference FP32 logits
        var fp32Logits = [Float](repeating: 0, count: vocabSize)
        classifierFP32.withUnsafePointer { clsPtr in
            for r in 0..<vocabSize {
                var dot: Float = 0
                for c in 0..<dim {
                    dot += clsPtr[r * dim + c] * input[c]
                }
                fp32Logits[r] = dot
            }
        }

        // Quantize and compute INT4 logits
        let int4Weights = TensorBufferINT4(quantizing: classifierFP32, rows: vocabSize, cols: dim)
        let iSum = INT4Classifier.inputSum(UnsafePointer(input), dim: dim)
        INT4Classifier.int4Matvec(
            weights: int4Weights,
            input: UnsafePointer(input),
            inputSum: iSum,
            output: int4Output,
            vocabSize: vocabSize,
            dim: dim
        )

        // Check approximate match (INT4 has up to ~6% quantization error per element)
        for r in 0..<vocabSize {
            let ref = fp32Logits[r]
            let approx = int4Output[r]
            let maxError = abs(ref) * 0.15 + 1.0  // ~15% relative + 1.0 absolute tolerance
            XCTAssertEqual(approx, ref, accuracy: maxError,
                           "INT4 logit[\\(r)] too far from FP32 reference")
        }
    }

    /// INT4 inputSum helper produces correct result.
    func testInputSum() {
        let dim = 16
        let input = UnsafeMutablePointer<Float>.allocate(capacity: dim)
        defer { input.deallocate() }

        for i in 0..<dim { input[i] = Float(i + 1) }

        let sum = INT4Classifier.inputSum(UnsafePointer(input), dim: dim)
        let expected: Float = Float(dim * (dim + 1)) / 2.0  // 1+2+...+16 = 136
        XCTAssertEqual(sum, expected, accuracy: 1e-4)
    }

    /// Odd-dimension handling in INT4 (trailing nibble).
    func testOddDimensionINT4() {
        let vocabSize = 5
        let dim = 7  // Odd

        let classifierFP32 = TensorBuffer(count: vocabSize * dim, zeroed: true)
        let input = UnsafeMutablePointer<Float>.allocate(capacity: dim)
        defer { input.deallocate() }

        // Row 3 dominates
        classifierFP32.withUnsafeMutablePointer { ptr in
            for r in 0..<vocabSize {
                for c in 0..<dim {
                    ptr[r * dim + c] = r == 3 ? 10.0 : 0.5
                }
            }
        }
        for i in 0..<dim { input[i] = 1.0 }

        let int4Weights = TensorBufferINT4(quantizing: classifierFP32, rows: vocabSize, cols: dim)
        let iSum = INT4Classifier.inputSum(UnsafePointer(input), dim: dim)
        let idx = INT4Classifier.int4MatvecArgmax(
            weights: int4Weights,
            input: UnsafePointer(input),
            inputSum: iSum,
            vocabSize: vocabSize,
            dim: dim
        )

        XCTAssertEqual(Int(idx), 3, "Row 3 should win with odd dimension")
    }
}
