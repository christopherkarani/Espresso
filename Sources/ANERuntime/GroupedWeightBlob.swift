import Foundation
import ANETypes

internal enum GroupedWeightBlob {
    @inline(__always)
    internal static func build(
        from buffer: borrowing TensorBuffer,
        rows: Int,
        colsPerGroup: Int,
        groups: Int
    ) -> Data {
        buffer.withUnsafeBufferPointer { ptr in
            build(from: ptr, rows: rows, colsPerGroup: colsPerGroup, groups: groups)
        }
    }

    @inline(__always)
    internal static func build(
        from weights: UnsafeBufferPointer<Float>,
        rows: Int,
        colsPerGroup: Int,
        groups: Int
    ) -> Data {
        let repacked = repackDenseRowMajorWeights(
            from: weights,
            rows: rows,
            colsPerGroup: colsPerGroup,
            groups: groups
        )
        return WeightBlob.build(from: repacked, rows: rows, cols: colsPerGroup)
    }

    internal static func repackDenseRowMajorWeights(
        from weights: UnsafeBufferPointer<Float>,
        rows: Int,
        colsPerGroup: Int,
        groups: Int
    ) -> [Float] {
        precondition(rows >= 0 && colsPerGroup >= 0)
        precondition(groups > 0)

        let compactCount = rows * colsPerGroup
        if groups == 1 {
            precondition(weights.count == compactCount)
            return Array(weights)
        }

        if weights.count == compactCount {
            return Array(weights)
        }

        let denseCols = colsPerGroup * groups
        precondition(rows.isMultiple(of: groups))
        precondition(weights.count == rows * denseCols)

        let rowsPerGroup = rows / groups
        var repacked = [Float](repeating: 0, count: compactCount)
        for row in 0..<rows {
            let group = row / rowsPerGroup
            let srcStart = row * denseCols + group * colsPerGroup
            let dstStart = row * colsPerGroup
            for col in 0..<colsPerGroup {
                repacked[dstStart + col] = weights[srcStart + col]
            }
        }
        return repacked
    }
}
