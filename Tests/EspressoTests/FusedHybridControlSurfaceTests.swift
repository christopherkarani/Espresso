import Darwin
import IOSurface
import XCTest
@testable import ANEInterop
@testable import ANERuntime
@testable import ANETypes
@testable import Espresso

/// Incremental fused control writes must match the previous full-surface rewrite
/// of mask, position one-hot, and RoPE for sequential token indices.
final class FusedHybridControlSurfaceTests: XCTestCase {
    func test_incremental_writes_match_full_rewrite_for_sequential_tokens() throws {
        let dim = 8
        let headDim = 4
        let maxSeq = 4
        let laneSpatial = 1
        let ropeTheta: Float = 10_000

        for lastToken in 0..<maxSeq {
            let expected = try Self.surfaces(dim: dim, maxSeq: maxSeq, laneSpatial: laneSpatial)
            try Self.writeFullControls(
                expected,
                throughToken: lastToken,
                dim: dim,
                headDim: headDim,
                maxSeq: maxSeq,
                laneSpatial: laneSpatial,
                ropeTheta: ropeTheta
            )

            let actual = try Self.surfaces(dim: dim, maxSeq: maxSeq, laneSpatial: laneSpatial)
            try Self.initializeCaches(actual, dim: dim, maxSeq: maxSeq, laneSpatial: laneSpatial)
            for tokenIndex in 0...lastToken {
                let controls = ForwardPass.FusedControlColumns(
                    dim: dim,
                    headDim: headDim,
                    tokenIndex: tokenIndex,
                    ropeTheta: ropeTheta
                )
                try ForwardPass.writeFusedControlSurfaces(
                    mask: actual.mask,
                    posMask: actual.posMask,
                    ropePack: actual.rope,
                    dim: dim,
                    maxSeq: maxSeq,
                    laneSpatial: laneSpatial,
                    tokenIndex: tokenIndex,
                    controls: controls
                )
            }

            try Self.assertEqual(actual.mask, expected.mask, channels: dim, spatial: maxSeq, label: "mask t=\(lastToken)")
            try Self.assertEqual(actual.posMask, expected.posMask, channels: dim, spatial: maxSeq, label: "pos t=\(lastToken)")
            // channels headDim..<dim stay zero on both paths (init + lane-0 RoPE write).
            try Self.assertEqual(actual.rope, expected.rope, channels: dim, spatial: laneSpatial, label: "rope t=\(lastToken)")
        }
    }

    private struct Surfaces {
        let mask: IOSurfaceRef
        let posMask: IOSurfaceRef
        let rope: IOSurfaceRef
    }

    private static func surfaces(dim: Int, maxSeq: Int, laneSpatial: Int) throws -> Surfaces {
        Surfaces(
            mask: ane_interop_create_surface(dim * maxSeq * 2)!,
            posMask: ane_interop_create_surface(dim * maxSeq * 2)!,
            rope: ane_interop_create_surface(dim * laneSpatial * 2)!
        )
    }

    private static func initializeCaches(_ surfaces: Surfaces, dim: Int, maxSeq: Int, laneSpatial: Int) throws {
        let cacheZeros = [Float](repeating: 0, count: dim * maxSeq)
        let laneZeros = [Float](repeating: 0, count: dim * laneSpatial)
        let masked = [Float](repeating: -1e4, count: dim * maxSeq)
        try cacheZeros.withUnsafeBufferPointer { src in
            try SurfaceIO.writeFP16(to: surfaces.posMask, data: src, channels: dim, spatial: maxSeq)
        }
        try masked.withUnsafeBufferPointer { src in
            try SurfaceIO.writeFP16(to: surfaces.mask, data: src, channels: dim, spatial: maxSeq)
        }
        try laneZeros.withUnsafeBufferPointer { src in
            try SurfaceIO.writeFP16(to: surfaces.rope, data: src, channels: dim, spatial: laneSpatial)
        }
    }

    /// Pre-incremental algorithm: rewrite the full tensors every token.
    private static func writeFullControls(
        _ surfaces: Surfaces,
        throughToken lastToken: Int,
        dim: Int,
        headDim: Int,
        maxSeq: Int,
        laneSpatial: Int,
        ropeTheta: Float
    ) throws {
        let halfDim = headDim / 2
        var mask = [Float](repeating: -1e4, count: dim * maxSeq)
        var pos = [Float](repeating: 0, count: dim * maxSeq)
        for spatial in 0...lastToken {
            for channel in 0..<dim {
                mask[channel * maxSeq + spatial] = 0
            }
        }
        for channel in 0..<dim {
            pos[channel * maxSeq + lastToken] = 1
        }
        var rope = [Float](repeating: 0, count: dim * laneSpatial)
        for idx in 0..<halfDim {
            let angle = Float(lastToken) / powf(ropeTheta, Float(2 * idx) / Float(headDim))
            rope[idx * laneSpatial] = cosf(angle)
            rope[(halfDim + idx) * laneSpatial] = sinf(angle)
        }
        try mask.withUnsafeBufferPointer { src in
            try SurfaceIO.writeFP16(to: surfaces.mask, data: src, channels: dim, spatial: maxSeq)
        }
        try pos.withUnsafeBufferPointer { src in
            try SurfaceIO.writeFP16(to: surfaces.posMask, data: src, channels: dim, spatial: maxSeq)
        }
        try rope.withUnsafeBufferPointer { src in
            try SurfaceIO.writeFP16(to: surfaces.rope, data: src, channels: dim, spatial: laneSpatial)
        }
    }

    private static func assertEqual(
        _ actual: IOSurfaceRef,
        _ expected: IOSurfaceRef,
        channels: Int,
        spatial: Int,
        label: String
    ) throws {
        var actualValues = [Float](repeating: Float.nan, count: channels * spatial)
        var expectedValues = [Float](repeating: Float.nan, count: channels * spatial)
        try actualValues.withUnsafeMutableBufferPointer { dst in
            try SurfaceIO.readFP16(from: actual, into: dst, channelOffset: 0, channels: channels, spatial: spatial)
        }
        try expectedValues.withUnsafeMutableBufferPointer { dst in
            try SurfaceIO.readFP16(from: expected, into: dst, channelOffset: 0, channels: channels, spatial: spatial)
        }
        for idx in actualValues.indices {
            XCTAssertEqual(
                actualValues[idx],
                expectedValues[idx],
                accuracy: 1e-2,
                "\(label) index \(idx)"
            )
        }
    }
}
