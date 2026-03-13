import Accelerate

/// Vocabulary-partitioned argmax with norm-based upper-bound pruning.
///
/// Partitions the classifier into blocks of `blockSize` rows, computes each block's sgemm,
/// and prunes subsequent blocks whose Cauchy-Schwarz upper bound
/// (precomputed max row norm × input norm) cannot beat the current global best logit.
///
/// The token selected is always identical to the naïve full-sgemm argmax.
public enum PartitionedArgmax {

    public static let defaultBlockSize: Int = 4_000

    /// Precomputes per-block maximum L2 row norms for a classifier weight matrix.
    ///
    /// Called once during model init, not on the hot token-selection path.
    public static func precomputeBlockMaxNorms(
        classifier: UnsafePointer<Float>,
        vocabSize: Int,
        dim: Int,
        blockSize: Int = defaultBlockSize
    ) -> [Float] {
        precondition(vocabSize > 0)
        precondition(dim > 0)
        precondition(blockSize > 0)

        let numBlocks = (vocabSize + blockSize - 1) / blockSize
        var blockMaxNorms = [Float](repeating: 0, count: numBlocks)

        var blockIdx = 0
        var blockStart = 0
        while blockStart < vocabSize {
            let blockEnd = min(blockStart + blockSize, vocabSize)
            var blockMax: Float = 0

            for rowIdx in blockStart..<blockEnd {
                let rowBase = rowIdx * dim
                var sumOfSquares: Float = 0
                vDSP_svesq(classifier.advanced(by: rowBase), 1, &sumOfSquares, vDSP_Length(dim))
                let rowNorm = sqrtf(sumOfSquares)
                if rowNorm > blockMax {
                    blockMax = rowNorm
                }
            }

            blockMaxNorms[blockIdx] = blockMax
            blockIdx += 1
            blockStart = blockEnd
        }

        return blockMaxNorms
    }

    /// Computes argmax of `classifier × input` using block-level norm pruning.
    ///
    /// Correctness guarantee: the returned index is identical to the naïve full sgemm argmax.
    @inline(__always)
    public static func compute(
        classifier: UnsafePointer<Float>,
        input: UnsafePointer<Float>,
        logitsScratch: UnsafeMutablePointer<Float>,
        blockMaxNorms: UnsafePointer<Float>,
        vocabSize: Int,
        dim: Int,
        blockSize: Int,
        skippedBlocks: inout Int
    ) -> Int {
        var inputNormSquared: Float = 0
        vDSP_svesq(input, 1, &inputNormSquared, vDSP_Length(dim))
        let inputNorm = sqrtf(inputNormSquared)

        var bestIndex: Int = 0
        var bestValue: Float = -.infinity
        var blockIdx = 0
        skippedBlocks = 0

        var blockStart = 0
        while blockStart < vocabSize {
            let blockEnd = min(blockStart + blockSize, vocabSize)
            let blockCount = blockEnd - blockStart

            if blockIdx > 0, bestValue > -(Float.infinity) {
                let upperBound = blockMaxNorms[blockIdx] * inputNorm
                if upperBound < bestValue {
                    skippedBlocks += 1
                    blockIdx += 1
                    blockStart = blockEnd
                    continue
                }
            }

            // Compute logits for this block via narrow sgemm
            BLAS.sgemm(
                CblasRowMajor,
                CblasNoTrans,
                CblasNoTrans,
                m: Int32(blockCount),
                n: 1,
                k: Int32(dim),
                alpha: 1.0,
                a: classifier.advanced(by: blockStart * dim),
                lda: Int32(dim),
                b: input,
                ldb: 1,
                beta: 0.0,
                c: logitsScratch,
                ldc: 1
            )

            var blockMaxValue: Float = 0
            var blockMaxIdx: vDSP_Length = 0
            vDSP_maxvi(logitsScratch, 1, &blockMaxValue, &blockMaxIdx, vDSP_Length(blockCount))

            if blockMaxValue > bestValue {
                bestValue = blockMaxValue
                bestIndex = blockStart + Int(blockMaxIdx)
            }

            blockIdx += 1
            blockStart = blockEnd
        }

        return bestIndex
    }

    public struct EvalStats: Sendable {
        public let tokenIndex: Int
        public let evaluatedBlocks: Int
        public let prunedBlocks: Int
        public let totalBlocks: Int

        public var pruneRate: Double {
            guard totalBlocks > 0 else { return 0 }
            return Double(prunedBlocks) / Double(totalBlocks)
        }
    }

    @inline(__always)
    public static func computeWithStats(
        classifier: UnsafePointer<Float>,
        input: UnsafePointer<Float>,
        logitsScratch: UnsafeMutablePointer<Float>,
        blockMaxNorms: [Float],
        vocabSize: Int,
        dim: Int,
        blockSize: Int
    ) -> EvalStats {
        let totalBlocks = (vocabSize + blockSize - 1) / blockSize
        var skippedBlocks = 0
        let tokenIndex = blockMaxNorms.withUnsafeBufferPointer { normsPtr in
            compute(
                classifier: classifier,
                input: input,
                logitsScratch: logitsScratch,
                blockMaxNorms: normsPtr.baseAddress!,
                vocabSize: vocabSize,
                dim: dim,
                blockSize: blockSize,
                skippedBlocks: &skippedBlocks
            )
        }
        return EvalStats(
            tokenIndex: tokenIndex,
            evaluatedBlocks: totalBlocks - skippedBlocks,
            prunedBlocks: skippedBlocks,
            totalBlocks: totalBlocks
        )
    }
}
