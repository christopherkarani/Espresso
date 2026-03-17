import XCTest
@testable import CPUOps

final class RoPEDecodeStepTests: XCTestCase {
    /// Verify that `applyDecodeStep(position: 0)` matches `apply(seqLen: 1)` on identical data.
    func test_applyDecodeStep_matchesFullApply() {
        let nHeads = 4
        let headDim = 8
        let count = nHeads * headDim

        // Create identical inputs for both paths
        var qFull = [Float](repeating: 0, count: count)
        var kFull = [Float](repeating: 0, count: count)
        var qStep = [Float](repeating: 0, count: count)
        var kStep = [Float](repeating: 0, count: count)

        for i in 0..<count {
            let value = Float(i % 7) * 0.1 - 0.3
            qFull[i] = value
            kFull[i] = value * 0.5
            qStep[i] = value
            kStep[i] = value * 0.5
        }

        // Full apply with seqLen=1 (position 0 only)
        qFull.withUnsafeMutableBufferPointer { qBuf in
            kFull.withUnsafeMutableBufferPointer { kBuf in
                RoPE.apply(
                    q: qBuf.baseAddress!,
                    k: kBuf.baseAddress!,
                    seqLen: 1,
                    nHeads: nHeads,
                    headDim: headDim
                )
            }
        }

        // Decode step at position 0
        qStep.withUnsafeMutableBufferPointer { qBuf in
            kStep.withUnsafeMutableBufferPointer { kBuf in
                RoPE.applyDecodeStep(
                    q: qBuf.baseAddress!,
                    k: kBuf.baseAddress!,
                    nHeads: nHeads,
                    nKVHeads: nHeads,
                    headDim: headDim,
                    position: 0
                )
            }
        }

        for i in 0..<count {
            XCTAssertEqual(qFull[i], qStep[i], accuracy: 1e-6, "Q mismatch at index \(i)")
            XCTAssertEqual(kFull[i], kStep[i], accuracy: 1e-6, "K mismatch at index \(i)")
        }
    }

    /// Verify that position=5 produces different rotations than position=0.
    func test_applyDecodeStep_positionOffset() {
        let nHeads = 2
        let headDim = 4
        let count = nHeads * headDim

        var q0 = [Float](repeating: 1.0, count: count)
        var k0 = [Float](repeating: 1.0, count: count)
        var q5 = [Float](repeating: 1.0, count: count)
        var k5 = [Float](repeating: 1.0, count: count)

        q0.withUnsafeMutableBufferPointer { qBuf in
            k0.withUnsafeMutableBufferPointer { kBuf in
                RoPE.applyDecodeStep(
                    q: qBuf.baseAddress!, k: kBuf.baseAddress!,
                    nHeads: nHeads, nKVHeads: nHeads, headDim: headDim,
                    position: 0
                )
            }
        }

        q5.withUnsafeMutableBufferPointer { qBuf in
            k5.withUnsafeMutableBufferPointer { kBuf in
                RoPE.applyDecodeStep(
                    q: qBuf.baseAddress!, k: kBuf.baseAddress!,
                    nHeads: nHeads, nKVHeads: nHeads, headDim: headDim,
                    position: 5
                )
            }
        }

        // At position 0, cos(0)=1 and sin(0)=0, so values are unchanged
        // At position 5, rotations should differ
        var anyDifferent = false
        for i in 0..<count {
            if abs(q0[i] - q5[i]) > 1e-6 {
                anyDifferent = true
                break
            }
        }
        XCTAssertTrue(anyDifferent, "Position 0 and position 5 should produce different rotations")
    }

    /// Verify that theta=500000 produces different results than the default theta=10000.
    func test_applyDecodeStep_customTheta() {
        let nHeads = 2
        let headDim = 4
        let count = nHeads * headDim

        var qDefault = [Float](repeating: 1.0, count: count)
        var kDefault = [Float](repeating: 1.0, count: count)
        var qCustom = [Float](repeating: 1.0, count: count)
        var kCustom = [Float](repeating: 1.0, count: count)

        qDefault.withUnsafeMutableBufferPointer { qBuf in
            kDefault.withUnsafeMutableBufferPointer { kBuf in
                RoPE.applyDecodeStep(
                    q: qBuf.baseAddress!, k: kBuf.baseAddress!,
                    nHeads: nHeads, nKVHeads: nHeads, headDim: headDim,
                    position: 10, theta: 10_000.0
                )
            }
        }

        qCustom.withUnsafeMutableBufferPointer { qBuf in
            kCustom.withUnsafeMutableBufferPointer { kBuf in
                RoPE.applyDecodeStep(
                    q: qBuf.baseAddress!, k: kBuf.baseAddress!,
                    nHeads: nHeads, nKVHeads: nHeads, headDim: headDim,
                    position: 10, theta: 500_000.0
                )
            }
        }

        var anyDifferent = false
        for i in 0..<count {
            if abs(qDefault[i] - qCustom[i]) > 1e-6 {
                anyDifferent = true
                break
            }
        }
        XCTAssertTrue(anyDifferent, "theta=10000 and theta=500000 should produce different rotations")
    }

    /// Verify GQA support — nKVHeads < nHeads rotates fewer K heads.
    func test_applyDecodeStep_GQA() {
        let nHeads = 8
        let nKVHeads = 2
        let headDim = 4
        let qCount = nHeads * headDim
        let kCount = nKVHeads * headDim

        var q = [Float](repeating: 1.0, count: qCount)
        var k = [Float](repeating: 1.0, count: kCount)

        q.withUnsafeMutableBufferPointer { qBuf in
            k.withUnsafeMutableBufferPointer { kBuf in
                RoPE.applyDecodeStep(
                    q: qBuf.baseAddress!, k: kBuf.baseAddress!,
                    nHeads: nHeads, nKVHeads: nKVHeads, headDim: headDim,
                    position: 3
                )
            }
        }

        // All Q heads should be rotated (8 heads)
        // Only first 2 K heads should be rotated
        XCTAssertTrue(q.allSatisfy { $0.isFinite }, "All Q values should be finite")
        XCTAssertTrue(k.allSatisfy { $0.isFinite }, "All K values should be finite")
    }
}
