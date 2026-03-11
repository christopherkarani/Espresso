import XCTest
import IOSurface
@testable import ANEInterop
@testable import ANETypes
@testable import Espresso

private func makeOutputHeadTestSurface(channels: Int, spatial: Int) -> IOSurfaceRef {
    ane_interop_create_surface(channels * spatial * 2)!
}

final class ANEGenerationOutputHeadTests: XCTestCase {
    func test_output_head_io_write_token_pair_populates_first_two_spatial_slices() throws {
        let spatial = 4
        let surface = makeOutputHeadTestSurface(channels: ModelConfig.dim, spatial: spatial)

        let zeros = Array(repeating: Float(0), count: ModelConfig.dim * spatial)
        zeros.withUnsafeBufferPointer { src in
            SurfaceIO.writeFP16(to: surface, data: src, channels: ModelConfig.dim, spatial: spatial)
        }

        let tokenA = TensorBuffer(count: ModelConfig.dim, zeroed: true)
        let tokenB = TensorBuffer(count: ModelConfig.dim, zeroed: true)
        tokenA.withUnsafeMutableBufferPointer { ptr in
            for idx in ptr.indices {
                ptr[idx] = Float(idx) * 0.25
            }
        }
        tokenB.withUnsafeMutableBufferPointer { ptr in
            for idx in ptr.indices {
                ptr[idx] = Float(idx) * -0.5
            }
        }

        try ANEGenerationOutputHeadIO.writeTokenPair(
            tokenA,
            tokenB,
            to: surface,
            laneSpatial: spatial
        )

        var readA = Array(repeating: Float.nan, count: ModelConfig.dim)
        try readA.withUnsafeMutableBufferPointer { dst in
            try SurfaceIO.readFP16SpatialSlice(
                from: surface,
                channelOffset: 0,
                spatialIndex: 0,
                spatial: spatial,
                into: dst,
                channels: ModelConfig.dim
            )
        }

        var readB = Array(repeating: Float.nan, count: ModelConfig.dim)
        try readB.withUnsafeMutableBufferPointer { dst in
            try SurfaceIO.readFP16SpatialSlice(
                from: surface,
                channelOffset: 0,
                spatialIndex: 1,
                spatial: spatial,
                into: dst,
                channels: ModelConfig.dim
            )
        }

        var readTail = Array(repeating: Float.nan, count: ModelConfig.dim)
        try readTail.withUnsafeMutableBufferPointer { dst in
            try SurfaceIO.readFP16SpatialSlice(
                from: surface,
                channelOffset: 0,
                spatialIndex: 2,
                spatial: spatial,
                into: dst,
                channels: ModelConfig.dim
            )
        }

        tokenA.withUnsafeBufferPointer { expected in
            for idx in expected.indices {
                XCTAssertEqual(readA[idx], expected[idx], accuracy: 1e-2)
            }
        }
        tokenB.withUnsafeBufferPointer { expected in
            for idx in expected.indices {
                XCTAssertEqual(readB[idx], expected[idx], accuracy: 1e-2)
            }
        }
        for value in readTail {
            XCTAssertEqual(value, 0, accuracy: 0)
        }
    }

    func test_output_head_io_argmax_pair_reads_each_spatial_slice_independently() throws {
        let channels = 8
        let spatial = 4
        let surface = makeOutputHeadTestSurface(channels: channels, spatial: spatial)

        let logitsA: [Float] = [-4, 5, -3, 1, 0, 2, -1, 3]
        let logitsB: [Float] = [-2, -1, 7, 4, 6, 1, 0, -3]
        let zeros = Array(repeating: Float(0), count: channels)
        try logitsA.withUnsafeBufferPointer { src in
            try SurfaceIO.writeFP16SpatialSlice(
                to: surface,
                channelOffset: 0,
                spatialIndex: 0,
                spatial: spatial,
                data: src,
                channels: channels
            )
        }
        try logitsB.withUnsafeBufferPointer { src in
            try SurfaceIO.writeFP16SpatialSlice(
                to: surface,
                channelOffset: 0,
                spatialIndex: 1,
                spatial: spatial,
                data: src,
                channels: channels
            )
        }
        try zeros.withUnsafeBufferPointer { src in
            try SurfaceIO.writeFP16SpatialSlice(
                to: surface,
                channelOffset: 0,
                spatialIndex: 2,
                spatial: spatial,
                data: src,
                channels: channels
            )
        }

        let pair = try ANEGenerationOutputHeadIO.argmaxTokenPairLogits(
            from: surface,
            vocabSize: channels,
            laneSpatial: spatial
        )

        XCTAssertEqual(pair.0, 1)
        XCTAssertEqual(pair.1, 2)
    }
}
