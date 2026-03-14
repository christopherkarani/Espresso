import XCTest
@testable import Espresso
@testable import ANETypes

final class PartitionedArgmaxTests: XCTestCase {

    /// Partitioned argmax must match naive sgemm argmax for a random-ish weight matrix.
    func testPartitionedArgmaxMatchesNaive() {
        let vocabSize = 100
        let dim = 32
        let blockSize = 25

        // Create a classifier matrix with a known argmax row
        let classifier = UnsafeMutablePointer<Float>.allocate(capacity: vocabSize * dim)
        let input = UnsafeMutablePointer<Float>.allocate(capacity: dim)
        let logitsScratch = UnsafeMutablePointer<Float>.allocate(capacity: blockSize)
        let naiveLogits = UnsafeMutablePointer<Float>.allocate(capacity: vocabSize)
        defer {
            classifier.deallocate()
            input.deallocate()
            logitsScratch.deallocate()
            naiveLogits.deallocate()
        }

        // Fill classifier with ascending values; row 73 gets a spike
        for r in 0..<vocabSize {
            for c in 0..<dim {
                classifier[r * dim + c] = Float(r) * 0.01 + Float(c) * 0.001
            }
        }
        // Make row 73 clearly dominant
        for c in 0..<dim {
            classifier[73 * dim + c] = 10.0
        }

        // Input: all ones
        for i in 0..<dim { input[i] = 1.0 }

        // Naive argmax via manual dot products
        for r in 0..<vocabSize {
            var dot: Float = 0
            for c in 0..<dim {
                dot += classifier[r * dim + c] * input[c]
            }
            naiveLogits[r] = dot
        }
        var naiveBest: Float = naiveLogits[0]
        var naiveBestIdx = 0
        for i in 1..<vocabSize {
            if naiveLogits[i] > naiveBest {
                naiveBest = naiveLogits[i]
                naiveBestIdx = i
            }
        }

        // Partitioned argmax
        let blockMaxNorms = PartitionedArgmax.precomputeBlockMaxNorms(
            classifier: classifier,
            vocabSize: vocabSize,
            dim: dim,
            blockSize: blockSize
        )

        let stats = PartitionedArgmax.computeWithStats(
            classifier: classifier,
            input: input,
            logitsScratch: logitsScratch,
            blockMaxNorms: blockMaxNorms,
            vocabSize: vocabSize,
            dim: dim,
            blockSize: blockSize
        )

        XCTAssertEqual(stats.tokenIndex, naiveBestIdx, "Partitioned argmax must match naive argmax")
        XCTAssertEqual(stats.tokenIndex, 73, "Row 73 should win")
        XCTAssertEqual(stats.totalBlocks, 4, "100 rows / 25 blockSize = 4 blocks")
        XCTAssertGreaterThan(stats.prunedBlocks, 0, "Should prune at least one block")
    }

    /// Partitioned argmax correctness when blockSize exceeds vocabSize (single block).
    func testSingleBlockPartitionedArgmax() {
        let vocabSize = 10
        let dim = 4
        let blockSize = 100  // Larger than vocabSize

        let classifier = UnsafeMutablePointer<Float>.allocate(capacity: vocabSize * dim)
        let input = UnsafeMutablePointer<Float>.allocate(capacity: dim)
        let logitsScratch = UnsafeMutablePointer<Float>.allocate(capacity: blockSize)
        defer {
            classifier.deallocate()
            input.deallocate()
            logitsScratch.deallocate()
        }

        // Row 5 dominates
        for r in 0..<vocabSize {
            for c in 0..<dim {
                classifier[r * dim + c] = r == 5 ? 5.0 : 0.1
            }
        }
        for i in 0..<dim { input[i] = 1.0 }

        let blockMaxNorms = PartitionedArgmax.precomputeBlockMaxNorms(
            classifier: classifier, vocabSize: vocabSize, dim: dim, blockSize: blockSize
        )

        let stats = PartitionedArgmax.computeWithStats(
            classifier: classifier, input: input, logitsScratch: logitsScratch,
            blockMaxNorms: blockMaxNorms, vocabSize: vocabSize, dim: dim, blockSize: blockSize
        )

        XCTAssertEqual(stats.tokenIndex, 5)
        XCTAssertEqual(stats.totalBlocks, 1)
        XCTAssertEqual(stats.prunedBlocks, 0)
    }
}
