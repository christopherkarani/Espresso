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

enum CPUStagedExactGenerationOutputHeadLayoutStrategy: Sendable, Equatable {
    case contiguous(shardSize: Int)
    case clustered(clusterCount: Int, projectionDimensionCount: Int, iterations: Int)
}

final class ExactGenerationOutputHeadCluster {
    let summary: ExactGenerationOutputHeadShardSummary
    let tokenIndices: [Int]
    let weights: TensorBuffer

    init(
        summary: ExactGenerationOutputHeadShardSummary,
        tokenIndices: [Int],
        weights: consuming TensorBuffer
    ) {
        self.summary = summary
        self.tokenIndices = tokenIndices
        self.weights = weights
    }
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
    private let vocabSize: Int
    private let clusters: [ExactGenerationOutputHeadCluster]
    private let clusterLookup: [Int: Int]
    private let shardSummaries: [ExactGenerationOutputHeadShardSummary]
    private let shardScratch: TensorBuffer
    let layoutStrategy: CPUStagedExactGenerationOutputHeadLayoutStrategy

    private(set) var lastEvaluatedShardCount: Int = 0

    init(
        classifierWeights: borrowing TensorBuffer,
        vocabSize: Int,
        layoutStrategy: CPUStagedExactGenerationOutputHeadLayoutStrategy = .contiguous(shardSize: 1024)
    ) throws(GenerationError) {
        guard vocabSize > 0 else {
            throw .invalidArguments("vocabSize must be > 0")
        }
        guard classifierWeights.count == vocabSize * ModelConfig.dim else {
            throw .invalidArguments(
                "classifier weight count \(classifierWeights.count) does not match vocabSize \(vocabSize) * dim \(ModelConfig.dim)"
            )
        }

        let clusters = try Self.makeClusters(
            classifierWeights: classifierWeights,
            vocabSize: vocabSize,
            layoutStrategy: layoutStrategy
        )
        self.vocabSize = vocabSize
        self.clusters = clusters
        self.clusterLookup = Dictionary(uniqueKeysWithValues: clusters.enumerated().map { ($1.summary.tokenOffset, $0) })
        self.shardSummaries = clusters.map(\.summary)
        self.shardScratch = TensorBuffer(count: clusters.map(\.tokenIndices.count).max() ?? 1, zeroed: true)
        self.layoutStrategy = layoutStrategy
    }

    init(
        vocabSize: Int,
        layoutStrategy: CPUStagedExactGenerationOutputHeadLayoutStrategy,
        clusters: consuming [ExactGenerationOutputHeadCluster]
    ) throws(GenerationError) {
        guard vocabSize > 0 else {
            throw .invalidArguments("vocabSize must be > 0")
        }
        guard !clusters.isEmpty else {
            throw .invalidArguments("clusters must not be empty")
        }

        self.vocabSize = vocabSize
        self.clusters = clusters
        self.clusterLookup = Dictionary(uniqueKeysWithValues: clusters.enumerated().map { ($1.summary.tokenOffset, $0) })
        self.shardSummaries = clusters.map(\.summary)
        self.shardScratch = TensorBuffer(count: clusters.map(\.tokenIndices.count).max() ?? 1, zeroed: true)
        self.layoutStrategy = layoutStrategy
    }

    func selectArgmax(
        normalizedInput: borrowing TensorBuffer
    ) throws(GenerationError) -> TokenID {
        precondition(normalizedInput.count == ModelConfig.dim)

        let normalizedVector = normalizedInput.withUnsafeBufferPointer { Array($0) }
        let result = try ExactGenerationOutputHeadBoundSearch.selectGlobalBest(
            normalizedInput: normalizedVector,
            shardSummaries: shardSummaries
        ) { summary in
            self.scoreCluster(self.cluster(for: summary), normalizedInput: normalizedInput)
        }
        self.lastEvaluatedShardCount = result.evaluatedShardOffsets.count

        guard let token = TokenID(exactly: result.token) else {
            throw .invalidArguments("selected token index \(result.token) exceeds TokenID range")
        }
        return token
    }

    private func cluster(for summary: ExactGenerationOutputHeadShardSummary) -> ExactGenerationOutputHeadCluster {
        guard let clusterIndex = clusterLookup[summary.tokenOffset] else {
            fatalError("missing cluster for summary id \(summary.tokenOffset)")
        }
        return clusters[clusterIndex]
    }

    private func scoreCluster(
        _ cluster: borrowing ExactGenerationOutputHeadCluster,
        normalizedInput: borrowing TensorBuffer
    ) -> (token: Int, score: Float) {
        precondition(normalizedInput.count == ModelConfig.dim)

        cluster.weights.withUnsafePointer { classifierPtr in
            normalizedInput.withUnsafePointer { inputPtr in
                shardScratch.withUnsafeMutablePointer { scratchPtr in
                    BLAS.sgemm(
                        CblasRowMajor,
                        CblasNoTrans,
                        CblasNoTrans,
                        m: Int32(cluster.tokenIndices.count),
                        n: 1,
                        k: Int32(ModelConfig.dim),
                        alpha: 1.0,
                        a: classifierPtr,
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
            if cluster.tokenIndices.count > 1 {
                for index in 1..<cluster.tokenIndices.count where scores[index] > bestValue {
                    bestValue = scores[index]
                    bestIndex = index
                }
            }
            return (bestIndex, bestValue)
        }

        return (
            token: cluster.tokenIndices[localBest.0],
            score: localBest.1
        )
    }

    private static func makeClusters(
        classifierWeights: borrowing TensorBuffer,
        vocabSize: Int,
        layoutStrategy: CPUStagedExactGenerationOutputHeadLayoutStrategy
    ) throws(GenerationError) -> [ExactGenerationOutputHeadCluster] {
        switch layoutStrategy {
        case let .contiguous(shardSize):
            guard shardSize > 0 else {
                throw .invalidArguments("shardSize must be > 0")
            }
            return makeContiguousClusters(
                classifierWeights: classifierWeights,
                vocabSize: vocabSize,
                shardSize: shardSize
            )
        case let .clustered(clusterCount, projectionDimensionCount, iterations):
            guard clusterCount > 0 else {
                throw .invalidArguments("clusterCount must be > 0")
            }
            guard projectionDimensionCount > 0 else {
                throw .invalidArguments("projectionDimensionCount must be > 0")
            }
            guard iterations > 0 else {
                throw .invalidArguments("iterations must be > 0")
            }
            return makeClusteredClusters(
                classifierWeights: classifierWeights,
                vocabSize: vocabSize,
                clusterCount: clusterCount,
                projectionDimensionCount: projectionDimensionCount,
                iterations: iterations
            )
        }
    }

    private static func makeContiguousClusters(
        classifierWeights: borrowing TensorBuffer,
        vocabSize: Int,
        shardSize: Int
    ) -> [ExactGenerationOutputHeadCluster] {
        var clusters: [ExactGenerationOutputHeadCluster] = []
        clusters.reserveCapacity((vocabSize + shardSize - 1) / shardSize)

        var tokenOffset = 0
        var clusterID = 0
        while tokenOffset < vocabSize {
            let end = min(tokenOffset + shardSize, vocabSize)
            clusters.append(
                makeCluster(
                    clusterID: clusterID,
                    tokenIndices: Array(tokenOffset..<end),
                    classifierWeights: classifierWeights
                )
            )
            tokenOffset = end
            clusterID += 1
        }
        return clusters
    }

    private static func makeClusteredClusters(
        classifierWeights: borrowing TensorBuffer,
        vocabSize: Int,
        clusterCount: Int,
        projectionDimensionCount: Int,
        iterations: Int
    ) -> [ExactGenerationOutputHeadCluster] {
        let dim = ModelConfig.dim
        let resolvedClusterCount = min(clusterCount, vocabSize)
        let projectionIndices = makeProjectionIndices(dim: dim, projectionDimensionCount: projectionDimensionCount)
        let projectedRows = makeProjectedRows(
            classifierWeights: classifierWeights,
            vocabSize: vocabSize,
            projectionIndices: projectionIndices
        )
        var centroids = makeInitialCentroids(
            projectedRows: projectedRows,
            vocabSize: vocabSize,
            projectionDimensionCount: projectionIndices.count,
            clusterCount: resolvedClusterCount
        )
        var assignments = [Int](repeating: 0, count: vocabSize)

        for _ in 0..<iterations {
            assignments = assignRowsToCentroids(
                projectedRows: projectedRows,
                vocabSize: vocabSize,
                projectionDimensionCount: projectionIndices.count,
                centroids: centroids
            )
            centroids = recomputeCentroids(
                projectedRows: projectedRows,
                assignments: assignments,
                clusterCount: resolvedClusterCount,
                projectionDimensionCount: projectionIndices.count,
                fallbackCentroids: centroids
            )
        }

        var groupedTokenIndices = Array(repeating: [Int](), count: resolvedClusterCount)
        for tokenIndex in 0..<vocabSize {
            groupedTokenIndices[assignments[tokenIndex]].append(tokenIndex)
        }

        var clusters: [ExactGenerationOutputHeadCluster] = []
        clusters.reserveCapacity(resolvedClusterCount)
        for clusterID in 0..<resolvedClusterCount where !groupedTokenIndices[clusterID].isEmpty {
            clusters.append(
                makeCluster(
                    clusterID: clusterID,
                    tokenIndices: groupedTokenIndices[clusterID],
                    classifierWeights: classifierWeights
                )
            )
        }
        return clusters
    }

    private static func makeProjectionIndices(
        dim: Int,
        projectionDimensionCount: Int
    ) -> [Int] {
        let count = min(dim, projectionDimensionCount)
        if count == dim {
            return Array(0..<dim)
        }
        var indices: [Int] = []
        indices.reserveCapacity(count)
        var seen = Set<Int>()
        for step in 0..<count {
            let index = min(dim - 1, (step * dim) / count)
            if seen.insert(index).inserted {
                indices.append(index)
            }
        }
        if indices.count < count {
            for index in 0..<dim where seen.insert(index).inserted {
                indices.append(index)
                if indices.count == count {
                    break
                }
            }
        }
        return indices
    }

    private static func makeProjectedRows(
        classifierWeights: borrowing TensorBuffer,
        vocabSize: Int,
        projectionIndices: [Int]
    ) -> [Float] {
        let projectionDimensionCount = projectionIndices.count
        return classifierWeights.withUnsafeBufferPointer { weights in
            var projectedRows = [Float](repeating: 0, count: vocabSize * projectionDimensionCount)
            for tokenIndex in 0..<vocabSize {
                let rowBase = tokenIndex * ModelConfig.dim
                let projectedBase = tokenIndex * projectionDimensionCount
                for projectionOffset in 0..<projectionDimensionCount {
                    projectedRows[projectedBase + projectionOffset] =
                        weights[rowBase + projectionIndices[projectionOffset]]
                }
            }
            return projectedRows
        }
    }

    private static func makeInitialCentroids(
        projectedRows: [Float],
        vocabSize: Int,
        projectionDimensionCount: Int,
        clusterCount: Int
    ) -> [Float] {
        var centroids = [Float](repeating: 0, count: clusterCount * projectionDimensionCount)
        for clusterIndex in 0..<clusterCount {
            let tokenIndex = min(vocabSize - 1, (clusterIndex * vocabSize) / clusterCount)
            let rowBase = tokenIndex * projectionDimensionCount
            let centroidBase = clusterIndex * projectionDimensionCount
            for projectionOffset in 0..<projectionDimensionCount {
                centroids[centroidBase + projectionOffset] = projectedRows[rowBase + projectionOffset]
            }
        }
        return centroids
    }

    private static func assignRowsToCentroids(
        projectedRows: [Float],
        vocabSize: Int,
        projectionDimensionCount: Int,
        centroids: [Float]
    ) -> [Int] {
        let clusterCount = centroids.count / projectionDimensionCount
        var assignments = [Int](repeating: 0, count: vocabSize)
        for tokenIndex in 0..<vocabSize {
            let rowBase = tokenIndex * projectionDimensionCount
            var bestCluster = 0
            var bestDistance = Float.infinity
            for clusterIndex in 0..<clusterCount {
                let centroidBase = clusterIndex * projectionDimensionCount
                var distance: Float = 0
                for projectionOffset in 0..<projectionDimensionCount {
                    let delta = projectedRows[rowBase + projectionOffset] - centroids[centroidBase + projectionOffset]
                    distance += delta * delta
                }
                if distance < bestDistance {
                    bestDistance = distance
                    bestCluster = clusterIndex
                }
            }
            assignments[tokenIndex] = bestCluster
        }
        return assignments
    }

    private static func recomputeCentroids(
        projectedRows: [Float],
        assignments: [Int],
        clusterCount: Int,
        projectionDimensionCount: Int,
        fallbackCentroids: [Float]
    ) -> [Float] {
        var centroids = [Float](repeating: 0, count: clusterCount * projectionDimensionCount)
        var counts = [Int](repeating: 0, count: clusterCount)

        for tokenIndex in 0..<assignments.count {
            let clusterIndex = assignments[tokenIndex]
            counts[clusterIndex] += 1
            let rowBase = tokenIndex * projectionDimensionCount
            let centroidBase = clusterIndex * projectionDimensionCount
            for projectionOffset in 0..<projectionDimensionCount {
                centroids[centroidBase + projectionOffset] += projectedRows[rowBase + projectionOffset]
            }
        }

        for clusterIndex in 0..<clusterCount {
            let centroidBase = clusterIndex * projectionDimensionCount
            if counts[clusterIndex] == 0 {
                for projectionOffset in 0..<projectionDimensionCount {
                    centroids[centroidBase + projectionOffset] = fallbackCentroids[centroidBase + projectionOffset]
                }
            } else {
                let invCount = 1.0 as Float / Float(counts[clusterIndex])
                for projectionOffset in 0..<projectionDimensionCount {
                    centroids[centroidBase + projectionOffset] *= invCount
                }
            }
        }
        return centroids
    }

    private static func makeCluster(
        clusterID: Int,
        tokenIndices: [Int],
        classifierWeights: borrowing TensorBuffer
    ) -> ExactGenerationOutputHeadCluster {
        let dim = ModelConfig.dim
        let tokenCount = tokenIndices.count
        let weights = TensorBuffer(count: tokenCount * dim, zeroed: true)
        var center = [Float](repeating: 0, count: dim)

        classifierWeights.withUnsafeBufferPointer { source in
            weights.withUnsafeMutableBufferPointer { destination in
                for localIndex in 0..<tokenCount {
                    let tokenIndex = tokenIndices[localIndex]
                    let sourceBase = tokenIndex * dim
                    let destinationBase = localIndex * dim
                    for dimIndex in 0..<dim {
                        let value = source[sourceBase + dimIndex]
                        destination[destinationBase + dimIndex] = value
                        center[dimIndex] += value
                    }
                }
            }
        }

        let invCount = 1.0 as Float / Float(tokenCount)
        for dimIndex in 0..<dim {
            center[dimIndex] *= invCount
        }

        var radius: Float = 0
        weights.withUnsafeBufferPointer { data in
            for localIndex in 0..<tokenCount {
                let base = localIndex * dim
                var distanceSquared: Float = 0
                for dimIndex in 0..<dim {
                    let delta = data[base + dimIndex] - center[dimIndex]
                    distanceSquared += delta * delta
                }
                radius = max(radius, sqrt(distanceSquared))
            }
        }

        return ExactGenerationOutputHeadCluster(
            summary: ExactGenerationOutputHeadShardSummary(
                tokenOffset: clusterID,
                tokenCount: tokenCount,
                center: center,
                radius: radius
            ),
            tokenIndices: tokenIndices,
            weights: weights
        )
    }
}
