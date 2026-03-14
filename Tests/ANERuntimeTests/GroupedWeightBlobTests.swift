import XCTest
@testable import ANERuntime

final class GroupedWeightBlobTests: XCTestCase {
    func test_repackDenseRowMajorWeights_selects_perRowGroupSlice() {
        let dense: [Float] = [
            0, 1, 2, 3,
            4, 5, 6, 7,
            8, 9, 10, 11,
            12, 13, 14, 15,
        ]

        let repacked = dense.withUnsafeBufferPointer {
            GroupedWeightBlob.repackDenseRowMajorWeights(from: $0, rows: 4, colsPerGroup: 2, groups: 2)
        }

        XCTAssertEqual(repacked, [0, 1, 4, 5, 10, 11, 14, 15])
    }
}
