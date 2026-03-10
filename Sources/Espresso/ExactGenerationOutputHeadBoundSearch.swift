import Foundation
import Accelerate
import CPUOps
import ANETypes

struct ExactGenerationOutputHeadShardSummary: Sendable, Equatable {
    let tokenOffset: Int
    let tokenCount: Int
    let center: [Float]
    let radius: Float

    init(
        tokenOffset: Int,
        tokenCount: Int,
        center: [Float],
        radius: Float
    ) {
        self.tokenOffset = tokenOffset
        self.tokenCount = tokenCount
        self.center = center
        self.radius = radius
    }

    func upperBound(forNormalizedInput normalizedInput: [Float]) -> Float {
        precondition(normalizedInput.count == center.count)
        let inputNorm = sqrt(normalizedInput.reduce(0 as Float) { partial, value in
            partial + value * value
        })
        let centerDot = zip(center, normalizedInput).reduce(0 as Float) { partial, pair in
            partial + pair.0 * pair.1
        }
        return centerDot + inputNorm * radius
    }

    static func makeContiguousShards(
        classifierRows: [[Float]],
        shardSize: Int
    ) throws(GenerationError) -> [ExactGenerationOutputHeadShardSummary] {
        guard !classifierRows.isEmpty else {
            throw .invalidArguments("classifierRows must not be empty")
        }
        guard shardSize > 0 else {
            throw .invalidArguments("shardSize must be > 0")
        }
        let dim = classifierRows[0].count
        guard dim > 0 else {
            throw .invalidArguments("classifier row dimension must be > 0")
        }
        for (rowIndex, row) in classifierRows.enumerated() where row.count != dim {
            throw .invalidArguments("classifier row \(rowIndex) has dimension \(row.count), expected \(dim)")
        }

        var summaries: [ExactGenerationOutputHeadShardSummary] = []
        summaries.reserveCapacity((classifierRows.count + shardSize - 1) / shardSize)

        var tokenOffset = 0
        while tokenOffset < classifierRows.count {
            let end = min(tokenOffset + shardSize, classifierRows.count)
            let shardRows = Array(classifierRows[tokenOffset..<end])
            summaries.append(
                makeSummary(
                    tokenOffset: tokenOffset,
                    rows: shardRows,
                    dim: dim
                )
            )
            tokenOffset = end
        }

        return summaries
    }

    static func makeContiguousShards(
        classifierWeights: borrowing TensorBuffer,
        vocabSize: Int,
        dim: Int = ModelConfig.dim,
        shardSize: Int
    ) throws(GenerationError) -> [ExactGenerationOutputHeadShardSummary] {
        guard vocabSize > 0 else {
            throw .invalidArguments("vocabSize must be > 0")
        }
        guard dim > 0 else {
            throw .invalidArguments("dim must be > 0")
        }
        guard classifierWeights.count == vocabSize * dim else {
            throw .invalidArguments(
                "classifier weight count \(classifierWeights.count) does not match vocabSize \(vocabSize) * dim \(dim)"
            )
        }

        let rows: [[Float]] = classifierWeights.withUnsafeBufferPointer { buffer in
            var rows = [[Float]]()
            rows.reserveCapacity(vocabSize)
            for tokenIndex in 0..<vocabSize {
                let base = tokenIndex * dim
                rows.append(Array(buffer[base..<(base + dim)]))
            }
            return rows
        }
        return try makeContiguousShards(classifierRows: rows, shardSize: shardSize)
    }

    private static func makeSummary(
        tokenOffset: Int,
        rows: [[Float]],
        dim: Int
    ) -> ExactGenerationOutputHeadShardSummary {
        var center = [Float](repeating: 0, count: dim)
        for row in rows {
            for dimIndex in 0..<dim {
                center[dimIndex] += row[dimIndex]
            }
        }

        let invCount = 1.0 as Float / Float(rows.count)
        for dimIndex in 0..<dim {
            center[dimIndex] *= invCount
        }

        var radius: Float = 0
        for row in rows {
            var distanceSquared: Float = 0
            for dimIndex in 0..<dim {
                let delta = row[dimIndex] - center[dimIndex]
                distanceSquared += delta * delta
            }
            radius = max(radius, sqrt(distanceSquared))
        }

        return ExactGenerationOutputHeadShardSummary(
            tokenOffset: tokenOffset,
            tokenCount: rows.count,
            center: center,
            radius: radius
        )
    }
}

struct ExactGenerationOutputHeadBoundSearchResult: Sendable, Equatable {
    let token: Int
    let score: Float
    let evaluatedShardOffsets: [Int]
    let prunedShardOffsets: [Int]
}

enum ExactGenerationOutputHeadBoundSearch {
    static func selectGlobalBest(
        normalizedInput: [Float],
        shardSummaries: [ExactGenerationOutputHeadShardSummary],
        scoreShard: (ExactGenerationOutputHeadShardSummary) throws(GenerationError) -> (token: Int, score: Float)
    ) throws(GenerationError) -> ExactGenerationOutputHeadBoundSearchResult {
        guard !normalizedInput.isEmpty else {
            throw .invalidArguments("normalizedInput must not be empty")
        }
        guard !shardSummaries.isEmpty else {
            throw .invalidArguments("shardSummaries must not be empty")
        }
        for summary in shardSummaries where summary.center.count != normalizedInput.count {
            throw .invalidArguments(
                "shard summary at offset \(summary.tokenOffset) has dimension \(summary.center.count), expected \(normalizedInput.count)"
            )
        }

        let ordered = shardSummaries.sorted { lhs, rhs in
            let lhsBound = lhs.upperBound(forNormalizedInput: normalizedInput)
            let rhsBound = rhs.upperBound(forNormalizedInput: normalizedInput)
            if lhsBound == rhsBound {
                return lhs.tokenOffset < rhs.tokenOffset
            }
            return lhsBound > rhsBound
        }

        var bestToken: Int?
        var bestScore: Float = -.infinity
        var evaluatedShardOffsets: [Int] = []
        var prunedShardOffsets: [Int] = []

        for (index, summary) in ordered.enumerated() {
            let bound = summary.upperBound(forNormalizedInput: normalizedInput)
            if let bestToken, bestScore > bound {
                prunedShardOffsets.append(contentsOf: ordered[index...].map(\.tokenOffset))
                return ExactGenerationOutputHeadBoundSearchResult(
                    token: bestToken,
                    score: bestScore,
                    evaluatedShardOffsets: evaluatedShardOffsets,
                    prunedShardOffsets: prunedShardOffsets
                )
            }

            let candidate = try scoreShard(summary)
            evaluatedShardOffsets.append(summary.tokenOffset)
            if bestToken == nil
                || candidate.score > bestScore
                || (candidate.score == bestScore && candidate.token < bestToken!)
            {
                bestToken = candidate.token
                bestScore = candidate.score
            }
        }

        guard let bestToken else {
            throw .runtimeFailure("exact output-head bound search produced no candidates")
        }

        return ExactGenerationOutputHeadBoundSearchResult(
            token: bestToken,
            score: bestScore,
            evaluatedShardOffsets: evaluatedShardOffsets,
            prunedShardOffsets: prunedShardOffsets
        )
    }
}

final class CPUStagedExactGenerationOutputHead {
    private let classifierWeights: TensorBuffer
    private let vocabSize: Int
    private let shardSummaries: [ExactGenerationOutputHeadShardSummary]
    private let shardScratch: TensorBuffer

    private(set) var lastEvaluatedShardCount: Int = 0

    init(
        classifierWeights: borrowing TensorBuffer,
        vocabSize: Int,
        shardSize: Int = 1024
    ) throws(GenerationError) {
        guard vocabSize > 0 else {
            throw .invalidArguments("vocabSize must be > 0")
        }
        guard shardSize > 0 else {
            throw .invalidArguments("shardSize must be > 0")
        }
        guard classifierWeights.count == vocabSize * ModelConfig.dim else {
            throw .invalidArguments(
                "classifier weight count \(classifierWeights.count) does not match vocabSize \(vocabSize) * dim \(ModelConfig.dim)"
            )
        }

        self.classifierWeights = GenerationWeightCloner.cloneTensor(classifierWeights)
        self.vocabSize = vocabSize
        self.shardSummaries = try ExactGenerationOutputHeadShardSummary.makeContiguousShards(
            classifierWeights: classifierWeights,
            vocabSize: vocabSize,
            dim: ModelConfig.dim,
            shardSize: shardSize
        )
        self.shardScratch = TensorBuffer(count: min(vocabSize, shardSize), zeroed: true)
    }

    func selectArgmax(
        normalizedInput: borrowing TensorBuffer
    ) throws(GenerationError) -> UInt16 {
        precondition(normalizedInput.count == ModelConfig.dim)

        let normalizedVector = normalizedInput.withUnsafeBufferPointer { Array($0) }
        let result = try ExactGenerationOutputHeadBoundSearch.selectGlobalBest(
            normalizedInput: normalizedVector,
            shardSummaries: shardSummaries
        ) { summary in
            self.scoreShard(summary: summary, normalizedInput: normalizedInput)
        }
        self.lastEvaluatedShardCount = result.evaluatedShardOffsets.count

        guard let token = UInt16(exactly: result.token) else {
            throw .invalidArguments("selected token index \(result.token) exceeds UInt16 range")
        }
        return token
    }

    private func scoreShard(
        summary: ExactGenerationOutputHeadShardSummary,
        normalizedInput: borrowing TensorBuffer
    ) -> (token: Int, score: Float) {
        precondition(normalizedInput.count == ModelConfig.dim)

        classifierWeights.withUnsafePointer { classifierPtr in
            normalizedInput.withUnsafePointer { inputPtr in
                shardScratch.withUnsafeMutablePointer { scratchPtr in
                    BLAS.sgemm(
                        CblasRowMajor,
                        CblasNoTrans,
                        CblasNoTrans,
                        m: Int32(summary.tokenCount),
                        n: 1,
                        k: Int32(ModelConfig.dim),
                        alpha: 1.0,
                        a: classifierPtr.advanced(by: summary.tokenOffset * ModelConfig.dim),
                        lda: Int32(ModelConfig.dim),
                        b: inputPtr,
                        ldb: 1,
                        beta: 0.0,
                        c: scratchPtr,
                        ldc: 1
                    )
                }
            }
        }

        let localBest = shardScratch.withUnsafeBufferPointer { scores in
            var bestIndex = 0
            var bestValue = scores[0]
            if summary.tokenCount > 1 {
                for index in 1..<summary.tokenCount where scores[index] > bestValue {
                    bestValue = scores[index]
                    bestIndex = index
                }
            }
            return (bestIndex, bestValue)
        }

        return (
            token: summary.tokenOffset + localBest.0,
            score: localBest.1
        )
    }
}
