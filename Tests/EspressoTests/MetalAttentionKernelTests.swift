import XCTest
import IOSurface
import ANEInterop
import ANERuntime
import ANETypes
@testable import Espresso

final class MetalAttentionKernelTests: XCTestCase {
    func test_kv_head_mapping_defaults_to_grouped_contiguous() {
        XCTAssertEqual(
            MetalAttentionKernel.resolvedKVHeadMappingMode(environment: [:]),
            .groupedContiguous
        )
        XCTAssertEqual(
            MetalAttentionKernel.kvHeadIndex(
                queryHead: 0,
                heads: 16,
                kvHeads: 8,
                mode: .groupedContiguous
            ),
            0
        )
        XCTAssertEqual(
            MetalAttentionKernel.kvHeadIndex(
                queryHead: 7,
                heads: 16,
                kvHeads: 8,
                mode: .groupedContiguous
            ),
            3
        )
        XCTAssertEqual(
            MetalAttentionKernel.kvHeadIndex(
                queryHead: 15,
                heads: 16,
                kvHeads: 8,
                mode: .groupedContiguous
            ),
            7
        )
    }

    func test_kv_head_mapping_supports_modulo_interleaving() {
        XCTAssertEqual(
            MetalAttentionKernel.resolvedKVHeadMappingMode(
                environment: [MetalAttentionKernel.moduloKVHeadMappingEnvKey: "1"]
            ),
            .moduloInterleaved
        )
        XCTAssertEqual(
            MetalAttentionKernel.kvHeadIndex(
                queryHead: 0,
                heads: 16,
                kvHeads: 8,
                mode: .moduloInterleaved
            ),
            0
        )
        XCTAssertEqual(
            MetalAttentionKernel.kvHeadIndex(
                queryHead: 7,
                heads: 16,
                kvHeads: 8,
                mode: .moduloInterleaved
            ),
            7
        )
        XCTAssertEqual(
            MetalAttentionKernel.kvHeadIndex(
                queryHead: 15,
                heads: 16,
                kvHeads: 8,
                mode: .moduloInterleaved
            ),
            7
        )
    }

    func test_metal_attention_matches_reference_on_small_problem() throws {
        let shape = try MetalAttentionShape(heads: 2, headDim: 4, seqLen: 4)
        let q: [Float] = [
            0.25, -0.50, 0.75, 1.00,
            -1.00, 0.50, 0.25, -0.25,
        ]
        let k: [Float] = [
            0.50, 0.25, -0.75, 1.00,
            1.00, -0.50, 0.25, 0.75,
            -0.25, 1.00, 0.50, -0.50,
            0.75, 0.50, -0.25, 0.25,

            -0.50, 0.75, 0.25, -1.00,
            0.25, -0.25, 1.00, 0.50,
            1.00, 0.50, -0.50, 0.25,
            -0.75, 0.25, 0.50, 1.00,
        ]
        let v: [Float] = [
            1.00, 0.00, 0.50, -0.50,
            0.50, 1.00, -0.25, 0.25,
            -0.75, 0.25, 1.25, 0.50,
            0.25, -0.50, 0.75, 1.50,

            -0.50, 1.00, 0.25, 0.75,
            0.75, -0.25, 1.00, -0.50,
            1.25, 0.50, -0.75, 0.25,
            0.00, 0.75, 0.50, 1.25,
        ]
        let mask = [Float](repeating: 0, count: shape.heads * shape.seqLen)

        let kernel = try MetalAttentionKernel()
        let actual = try kernel.run(q: q, k: k, v: v, mask: mask, shape: shape)
        let expected = referenceAttention(q: q, k: k, v: v, mask: mask, shape: shape)

        XCTAssertEqual(actual.count, expected.count)
        for (lhs, rhs) in zip(actual, expected) {
            XCTAssertEqual(lhs, rhs, accuracy: 1e-2)
        }
    }

    func test_metal_attention_benchmark_reports_positive_latency() throws {
        try requireHardwareBenchmarks()
        let shape = try MetalAttentionShape(heads: 12, headDim: 64, seqLen: 32)
        let kernel = try MetalAttentionKernel()
        let result = try kernel.benchmark(shape: shape, warmup: 3, iterations: 20, seed: 0xA11CE)
        print("Metal attention benchmark: mean=\(result.meanMs) ms median=\(result.medianMs) ms zeroCopy=\(result.zeroCopyBindings)")

        XCTAssertTrue(result.zeroCopyBindings)
        XCTAssertGreaterThan(result.medianMs, 0)
        XCTAssertEqual(result.iterations, 20)
    }

    func test_metal_decode_attention_matches_reference_on_small_problem() throws {
        let shape = try MetalDecodeAttentionShape(
            heads: 2,
            headDim: 4,
            visibleTokens: 2,
            cacheStride: 4,
            laneStride: 4
        )
        let dim = shape.heads * shape.headDim
        let qLane: [Float] = [
            0.25, -0.50, 0.75, 1.00,
            -1.00, 0.50, 0.25, -0.25,
        ]
        let residualLane: [Float] = [
            0.10, -0.20, 0.30, -0.40,
            0.05, 0.15, -0.10, 0.20,
        ]
        let kTokens: [Float] = [
            0.50, 0.25, -0.75, 1.00,
            1.00, -0.50, 0.25, 0.75,

            -0.50, 0.75, 0.25, -1.00,
            0.25, -0.25, 1.00, 0.50,
        ]
        let vTokens: [Float] = [
            1.00, 0.00, 0.50, -0.50,
            0.50, 1.00, -0.25, 0.25,

            -0.50, 1.00, 0.25, 0.75,
            0.75, -0.25, 1.00, -0.50,
        ]

        var qSurfaceData = [Float](repeating: 0, count: dim * shape.laneStride)
        var residualSurfaceData = [Float](repeating: 0, count: dim * shape.laneStride)
        var kCacheData = [Float](repeating: 0, count: dim * shape.cacheStride)
        var vCacheData = [Float](repeating: 0, count: dim * shape.cacheStride)
        for channel in 0..<dim {
            qSurfaceData[channel * shape.laneStride] = qLane[channel]
            residualSurfaceData[channel * shape.laneStride] = residualLane[channel]
        }
        for head in 0..<shape.heads {
            for token in 0..<shape.visibleTokens {
                for headOffset in 0..<shape.headDim {
                    let channel = head * shape.headDim + headOffset
                    let sourceIndex = (head * shape.visibleTokens + token) * shape.headDim + headOffset
                    kCacheData[channel * shape.cacheStride + token] = kTokens[sourceIndex]
                    vCacheData[channel * shape.cacheStride + token] = vTokens[sourceIndex]
                }
            }
        }

        let projection = HybridOutputProjectionWeights(
            cacheKey: "test-identity-projection",
            inputDim: dim,
            outputDim: dim,
            rowMajorWeights: identityMatrix(dim: dim),
            rowMajorBias: [Float](repeating: 0, count: dim)
        )
        let kernel = try MetalAttentionKernel()
        let qSurface = makeSurface(bytes: qSurfaceData.count * 2)
        let residualSurface = makeSurface(bytes: residualSurfaceData.count * 2)
        let kCacheSurface = makeSurface(bytes: kCacheData.count * 2)
        let vCacheSurface = makeSurface(bytes: vCacheData.count * 2)
        let outputSurface = makeSurface(bytes: qSurfaceData.count * 2)

        qSurfaceData.withUnsafeBufferPointer { src in
            SurfaceIO.writeFP16(to: qSurface, data: src, channels: dim, spatial: shape.laneStride)
        }
        residualSurfaceData.withUnsafeBufferPointer { src in
            SurfaceIO.writeFP16(to: residualSurface, data: src, channels: dim, spatial: shape.laneStride)
        }
        kCacheData.withUnsafeBufferPointer { src in
            SurfaceIO.writeFP16(to: kCacheSurface, data: src, channels: dim, spatial: shape.cacheStride)
        }
        vCacheData.withUnsafeBufferPointer { src in
            SurfaceIO.writeFP16(to: vCacheSurface, data: src, channels: dim, spatial: shape.cacheStride)
        }

        try kernel.runDecode(
            qSurface: qSurface,
            kCacheSurface: kCacheSurface,
            vCacheSurface: vCacheSurface,
            residualSurface: residualSurface,
            outputSurface: outputSurface,
            shape: shape,
            projection: projection
        )

        var actual = [Float](repeating: 0, count: dim)
        try actual.withUnsafeMutableBufferPointer { dst in
            try SurfaceIO.readFP16SpatialSlice(
                from: outputSurface,
                channelOffset: 0,
                spatialIndex: 0,
                spatial: shape.laneStride,
                into: dst,
                channels: dim
            )
        }

        let expected = referenceDecodeAttention(
            qLane: qLane,
            kCache: kCacheData,
            vCache: vCacheData,
            residualLane: residualLane,
            shape: shape
        )
        XCTAssertEqual(actual.count, expected.count)
        for (lhs, rhs) in zip(actual, expected) {
            XCTAssertEqual(lhs, rhs, accuracy: 1e-2)
        }
    }

    func test_metal_decode_context_into_surface_matches_reference_on_small_problem() throws {
        let shape = try MetalDecodeAttentionShape(
            heads: 2,
            headDim: 4,
            visibleTokens: 2,
            cacheStride: 4,
            laneStride: 4
        )
        let dim = shape.heads * shape.headDim
        let qLane: [Float] = [
            0.25, -0.50, 0.75, 1.00,
            -1.00, 0.50, 0.25, -0.25,
        ]
        let kTokens: [Float] = [
            0.50, 0.25, -0.75, 1.00,
            1.00, -0.50, 0.25, 0.75,

            -0.50, 0.75, 0.25, -1.00,
            0.25, -0.25, 1.00, 0.50,
        ]
        let vTokens: [Float] = [
            1.00, 0.00, 0.50, -0.50,
            0.50, 1.00, -0.25, 0.25,

            -0.50, 1.00, 0.25, 0.75,
            0.75, -0.25, 1.00, -0.50,
        ]

        var qSurfaceData = [Float](repeating: 0, count: dim * shape.laneStride)
        var kCacheData = [Float](repeating: 0, count: dim * shape.cacheStride)
        var vCacheData = [Float](repeating: 0, count: dim * shape.cacheStride)
        for channel in 0..<dim {
            qSurfaceData[channel * shape.laneStride] = qLane[channel]
        }
        for head in 0..<shape.heads {
            for token in 0..<shape.visibleTokens {
                for headOffset in 0..<shape.headDim {
                    let channel = head * shape.headDim + headOffset
                    let sourceIndex = (head * shape.visibleTokens + token) * shape.headDim + headOffset
                    kCacheData[channel * shape.cacheStride + token] = kTokens[sourceIndex]
                    vCacheData[channel * shape.cacheStride + token] = vTokens[sourceIndex]
                }
            }
        }

        let kernel = try MetalAttentionKernel()
        let qSurface = makeSurface(bytes: qSurfaceData.count * 2)
        let kCacheSurface = makeSurface(bytes: kCacheData.count * 2)
        let vCacheSurface = makeSurface(bytes: vCacheData.count * 2)
        let contextSurface = makeSurface(bytes: dim * shape.laneStride * MemoryLayout<Float>.stride)

        qSurfaceData.withUnsafeBufferPointer { src in
            SurfaceIO.writeFP16(to: qSurface, data: src, channels: dim, spatial: shape.laneStride)
        }
        kCacheData.withUnsafeBufferPointer { src in
            SurfaceIO.writeFP16(to: kCacheSurface, data: src, channels: dim, spatial: shape.cacheStride)
        }
        vCacheData.withUnsafeBufferPointer { src in
            SurfaceIO.writeFP16(to: vCacheSurface, data: src, channels: dim, spatial: shape.cacheStride)
        }

        try kernel.runDecodeContextIntoSurface(
            qSurface: qSurface,
            kCacheSurface: kCacheSurface,
            vCacheSurface: vCacheSurface,
            contextSurface: contextSurface,
            shape: shape
        )

        let actual = try readFP32SpatialSlice(
            from: contextSurface,
            spatialIndex: 0,
            spatial: shape.laneStride,
            channels: dim
        )
        let expected = referenceDecodeContext(
            qLane: qLane,
            kCache: kCacheData,
            vCache: vCacheData,
            shape: shape
        )
        XCTAssertEqual(actual.count, expected.count)
        for (lhs, rhs) in zip(actual, expected) {
            XCTAssertEqual(lhs, rhs, accuracy: 1e-4)
        }
    }

    func test_metal_fused_decode_sdpa_into_surface_matches_reference_on_small_problem() throws {
        let shape = try MetalDecodeAttentionShape(
            heads: 2,
            headDim: 4,
            visibleTokens: 2,
            cacheStride: 4,
            laneStride: 4
        )
        let dim = shape.heads * shape.headDim
        let qLane: [Float] = [
            0.25, -0.50, 0.75, 1.00,
            -1.00, 0.50, 0.25, -0.25,
        ]
        let kTokens: [Float] = [
            0.50, 0.25, -0.75, 1.00,
            1.00, -0.50, 0.25, 0.75,

            -0.50, 0.75, 0.25, -1.00,
            0.25, -0.25, 1.00, 0.50,
        ]
        let vTokens: [Float] = [
            1.00, 0.00, 0.50, -0.50,
            0.50, 1.00, -0.25, 0.25,

            -0.50, 1.00, 0.25, 0.75,
            0.75, -0.25, 1.00, -0.50,
        ]

        var qSurfaceData = [Float](repeating: 0, count: dim * shape.laneStride)
        var kCacheData = [Float](repeating: 0, count: dim * shape.cacheStride)
        var vCacheData = [Float](repeating: 0, count: dim * shape.cacheStride)
        for channel in 0..<dim {
            qSurfaceData[channel * shape.laneStride] = qLane[channel]
        }
        for head in 0..<shape.heads {
            for token in 0..<shape.visibleTokens {
                for headOffset in 0..<shape.headDim {
                    let channel = head * shape.headDim + headOffset
                    let sourceIndex = (head * shape.visibleTokens + token) * shape.headDim + headOffset
                    kCacheData[channel * shape.cacheStride + token] = kTokens[sourceIndex]
                    vCacheData[channel * shape.cacheStride + token] = vTokens[sourceIndex]
                }
            }
        }

        let kernel = try MetalAttentionKernel()
        let qSurface = makeSurface(bytes: qSurfaceData.count * 2)
        let kCacheSurface = makeSurface(bytes: kCacheData.count * 2)
        let vCacheSurface = makeSurface(bytes: vCacheData.count * 2)
        let contextSurface = makeSurface(bytes: dim * shape.laneStride * MemoryLayout<Float>.stride)

        qSurfaceData.withUnsafeBufferPointer { src in
            SurfaceIO.writeFP16(to: qSurface, data: src, channels: dim, spatial: shape.laneStride)
        }
        kCacheData.withUnsafeBufferPointer { src in
            SurfaceIO.writeFP16(to: kCacheSurface, data: src, channels: dim, spatial: shape.cacheStride)
        }
        vCacheData.withUnsafeBufferPointer { src in
            SurfaceIO.writeFP16(to: vCacheSurface, data: src, channels: dim, spatial: shape.cacheStride)
        }

        try kernel.runFusedDecodeSDPAIntoSurface(
            qSurface: qSurface,
            kCacheSurface: kCacheSurface,
            vCacheSurface: vCacheSurface,
            contextSurface: contextSurface,
            shape: shape
        )

        let actual = try readFP32SpatialSlice(
            from: contextSurface,
            spatialIndex: 0,
            spatial: shape.laneStride,
            channels: dim
        )
        let expected = referenceDecodeContext(
            qLane: qLane,
            kCache: kCacheData,
            vCache: vCacheData,
            shape: shape
        )
        XCTAssertEqual(actual.count, expected.count)
        for (lhs, rhs) in zip(actual, expected) {
            XCTAssertEqual(lhs, rhs, accuracy: 1e-4)
        }
    }

    func test_metal_fused_decode_sdpa_into_surface_matches_reference_for_gqa() throws {
        let shape = try MetalDecodeAttentionShape(
            heads: 4,
            kvHeads: 2,
            headDim: 2,
            visibleTokens: 3,
            cacheStride: 4,
            laneStride: 4
        )
        let dim = shape.heads * shape.headDim
        let kvDim = shape.kvHeads * shape.headDim
        let qLane: [Float] = [
            0.50, -0.25,
            1.00, 0.75,
            -0.50, 0.20,
            0.10, 0.90,
        ]
        var qSurfaceData = [Float](repeating: 0, count: dim * shape.laneStride)
        var kCacheData = [Float](repeating: 0, count: kvDim * shape.cacheStride)
        var vCacheData = [Float](repeating: 0, count: kvDim * shape.cacheStride)

        for channel in 0..<dim {
            qSurfaceData[channel * shape.laneStride] = qLane[channel]
        }

        let kRows: [[Float]] = [
            [0.20, -0.10, 0.30],
            [0.40, 0.60, -0.50],
            [-0.25, 0.50, 0.75],
            [0.10, -0.20, 0.90],
        ]
        let vRows: [[Float]] = [
            [1.00, 0.50, -0.25],
            [0.25, -0.75, 0.80],
            [-0.40, 0.90, 0.10],
            [0.60, -0.30, 0.45],
        ]
        for channel in 0..<kvDim {
            for token in 0..<shape.visibleTokens {
                kCacheData[channel * shape.cacheStride + token] = kRows[channel][token]
                vCacheData[channel * shape.cacheStride + token] = vRows[channel][token]
            }
        }

        let kernel = try MetalAttentionKernel()
        let qSurface = makeSurface(bytes: qSurfaceData.count * 2)
        let kCacheSurface = makeSurface(bytes: kCacheData.count * 2)
        let vCacheSurface = makeSurface(bytes: vCacheData.count * 2)
        let contextSurface = makeSurface(bytes: dim * shape.laneStride * MemoryLayout<Float>.stride)

        qSurfaceData.withUnsafeBufferPointer { src in
            SurfaceIO.writeFP16(to: qSurface, data: src, channels: dim, spatial: shape.laneStride)
        }
        kCacheData.withUnsafeBufferPointer { src in
            SurfaceIO.writeFP16(to: kCacheSurface, data: src, channels: kvDim, spatial: shape.cacheStride)
        }
        vCacheData.withUnsafeBufferPointer { src in
            SurfaceIO.writeFP16(to: vCacheSurface, data: src, channels: kvDim, spatial: shape.cacheStride)
        }

        try kernel.runFusedDecodeSDPAIntoSurface(
            qSurface: qSurface,
            kCacheSurface: kCacheSurface,
            vCacheSurface: vCacheSurface,
            contextSurface: contextSurface,
            shape: shape
        )

        let actual = try readFP32SpatialSlice(
            from: contextSurface,
            spatialIndex: 0,
            spatial: shape.laneStride,
            channels: dim
        )
        let expected = referenceDecodeContext(
            qLane: qLane,
            kCache: kCacheData,
            vCache: vCacheData,
            shape: shape
        )
        XCTAssertEqual(actual.count, expected.count)
        for (lhs, rhs) in zip(actual, expected) {
            XCTAssertEqual(lhs, rhs, accuracy: 1e-4)
        }
    }

    func test_metal_decode_context_into_surface_matches_reference_for_gqa() throws {
        let shape = try MetalDecodeAttentionShape(
            heads: 4,
            kvHeads: 2,
            headDim: 2,
            visibleTokens: 3,
            cacheStride: 4,
            laneStride: 4
        )
        let dim = shape.heads * shape.headDim
        let kvDim = shape.kvHeads * shape.headDim
        let qLane: [Float] = [
            0.50, -0.25,
            1.00, 0.75,
            -0.50, 0.20,
            0.10, 0.90,
        ]
        var qSurfaceData = [Float](repeating: 0, count: dim * shape.laneStride)
        var kCacheData = [Float](repeating: 0, count: kvDim * shape.cacheStride)
        var vCacheData = [Float](repeating: 0, count: kvDim * shape.cacheStride)

        for channel in 0..<dim {
            qSurfaceData[channel * shape.laneStride] = qLane[channel]
        }

        let kRows: [[Float]] = [
            [0.20, -0.10, 0.30],
            [0.40, 0.60, -0.50],
            [-0.25, 0.50, 0.75],
            [0.10, -0.20, 0.90],
        ]
        let vRows: [[Float]] = [
            [1.00, 0.50, -0.25],
            [0.25, -0.75, 0.80],
            [-0.40, 0.90, 0.10],
            [0.60, -0.30, 0.45],
        ]
        for channel in 0..<kvDim {
            for token in 0..<shape.visibleTokens {
                kCacheData[channel * shape.cacheStride + token] = kRows[channel][token]
                vCacheData[channel * shape.cacheStride + token] = vRows[channel][token]
            }
        }

        let kernel = try MetalAttentionKernel()
        let qSurface = makeSurface(bytes: qSurfaceData.count * 2)
        let kCacheSurface = makeSurface(bytes: kCacheData.count * 2)
        let vCacheSurface = makeSurface(bytes: vCacheData.count * 2)
        let contextSurface = makeSurface(bytes: dim * shape.laneStride * MemoryLayout<Float>.stride)

        qSurfaceData.withUnsafeBufferPointer { src in
            SurfaceIO.writeFP16(to: qSurface, data: src, channels: dim, spatial: shape.laneStride)
        }
        kCacheData.withUnsafeBufferPointer { src in
            SurfaceIO.writeFP16(to: kCacheSurface, data: src, channels: kvDim, spatial: shape.cacheStride)
        }
        vCacheData.withUnsafeBufferPointer { src in
            SurfaceIO.writeFP16(to: vCacheSurface, data: src, channels: kvDim, spatial: shape.cacheStride)
        }

        try kernel.runDecodeContextIntoSurface(
            qSurface: qSurface,
            kCacheSurface: kCacheSurface,
            vCacheSurface: vCacheSurface,
            contextSurface: contextSurface,
            shape: shape
        )

        let actual = try readFP32SpatialSlice(
            from: contextSurface,
            spatialIndex: 0,
            spatial: shape.laneStride,
            channels: dim
        )
        let expected = referenceDecodeContext(
            qLane: qLane,
            kCache: kCacheData,
            vCache: vCacheData,
            shape: shape
        )
        XCTAssertEqual(actual.count, expected.count)
        for (lhs, rhs) in zip(actual, expected) {
            XCTAssertEqual(lhs, rhs, accuracy: 1e-4)
        }
    }

    private func requireHardwareBenchmarks(file: StaticString = #filePath, line: UInt = #line) throws {
        guard ProcessInfo.processInfo.environment["ANE_HARDWARE_TESTS"] == "1" else {
            throw XCTSkip("Set ANE_HARDWARE_TESTS=1 to run Metal hardware benchmarks")
        }
    }

    private func makeSurface(bytes: Int) -> IOSurfaceRef {
        ane_interop_create_surface(bytes)!
    }

    private func readFP32SpatialSlice(
        from surface: IOSurfaceRef,
        spatialIndex: Int,
        spatial: Int,
        channels: Int
    ) throws -> [Float] {
        XCTAssertGreaterThan(spatial, 0)
        XCTAssertGreaterThanOrEqual(spatialIndex, 0)
        XCTAssertLessThan(spatialIndex, spatial)
        let requiredBytes = channels * spatial * MemoryLayout<Float>.stride
        XCTAssertGreaterThanOrEqual(IOSurfaceGetAllocSize(surface), requiredBytes)
        let status = IOSurfaceLock(surface, [], nil)
        XCTAssertEqual(status, kIOReturnSuccess)
        defer { IOSurfaceUnlock(surface, [], nil) }
        let base = IOSurfaceGetBaseAddress(surface).assumingMemoryBound(to: Float.self)
        return (0..<channels).map { channel in
            base[channel * spatial + spatialIndex]
        }
    }

    private func identityMatrix(dim: Int) -> [Float] {
        var values = [Float](repeating: 0, count: dim * dim)
        for index in 0..<dim {
            values[index * dim + index] = 1
        }
        return values
    }

    private func referenceAttention(
        q: [Float],
        k: [Float],
        v: [Float],
        mask: [Float],
        shape: MetalAttentionShape
    ) -> [Float] {
        let scale = 1.0 / sqrt(Float(shape.headDim))
        var output = [Float](repeating: 0, count: shape.heads * shape.headDim)

        for head in 0..<shape.heads {
            var logits = [Float](repeating: 0, count: shape.seqLen)
            for token in 0..<shape.seqLen {
                var dot: Float = 0
                let qBase = head * shape.headDim
                let kBase = (head * shape.seqLen + token) * shape.headDim
                for dim in 0..<shape.headDim {
                    dot += q[qBase + dim] * k[kBase + dim]
                }
                logits[token] = dot * scale + mask[head * shape.seqLen + token]
            }

            let maxLogit = logits.max() ?? 0
            var denom: Float = 0
            var weights = [Float](repeating: 0, count: shape.seqLen)
            for token in 0..<shape.seqLen {
                let value = exp(logits[token] - maxLogit)
                weights[token] = value
                denom += value
            }

            for dim in 0..<shape.headDim {
                var accum: Float = 0
                for token in 0..<shape.seqLen {
                    let normalized = weights[token] / denom
                    let vBase = (head * shape.seqLen + token) * shape.headDim
                    accum += normalized * v[vBase + dim]
                }
                output[head * shape.headDim + dim] = accum
            }
        }

        return output
    }

    private func referenceDecodeAttention(
        qLane: [Float],
        kCache: [Float],
        vCache: [Float],
        residualLane: [Float],
        shape: MetalDecodeAttentionShape
    ) -> [Float] {
        let scale = 1.0 / sqrt(Float(shape.headDim))
        let dim = shape.heads * shape.headDim
        var context = [Float](repeating: 0, count: dim)

        for head in 0..<shape.heads {
            var logits = [Float](repeating: 0, count: shape.visibleTokens)
            let kvHead = head / (shape.heads / shape.kvHeads)
            for token in 0..<shape.visibleTokens {
                var dot: Float = 0
                for headOffset in 0..<shape.headDim {
                    let qChannel = head * shape.headDim + headOffset
                    let kvChannel = kvHead * shape.headDim + headOffset
                    let cacheToken = shape.tokenBase + token
                    dot += qLane[qChannel] * kCache[kvChannel * shape.cacheStride + cacheToken]
                }
                logits[token] = dot * scale
            }

            let maxLogit = logits.max() ?? 0
            var weights = [Float](repeating: 0, count: shape.visibleTokens)
            var denom: Float = 0
            for token in 0..<shape.visibleTokens {
                let value = exp(logits[token] - maxLogit)
                weights[token] = value
                denom += value
            }

            for headOffset in 0..<shape.headDim {
                let channel = head * shape.headDim + headOffset
                let kvChannel = kvHead * shape.headDim + headOffset
                var accum: Float = 0
                for token in 0..<shape.visibleTokens {
                    let cacheToken = shape.tokenBase + token
                    accum += (weights[token] / denom) * vCache[kvChannel * shape.cacheStride + cacheToken]
                }
                context[channel] = accum
            }
        }

        return zip(context, residualLane).map(+)
    }

    private func referenceDecodeContext(
        qLane: [Float],
        kCache: [Float],
        vCache: [Float],
        shape: MetalDecodeAttentionShape
    ) -> [Float] {
        let scale = 1.0 / sqrt(Float(shape.headDim))
        let dim = shape.heads * shape.headDim
        var context = [Float](repeating: 0, count: dim)

        for head in 0..<shape.heads {
            var logits = [Float](repeating: 0, count: shape.visibleTokens)
            let kvHead = head / (shape.heads / shape.kvHeads)
            for token in 0..<shape.visibleTokens {
                var dot: Float = 0
                for headOffset in 0..<shape.headDim {
                    let qChannel = head * shape.headDim + headOffset
                    let kvChannel = kvHead * shape.headDim + headOffset
                    let cacheToken = shape.tokenBase + token
                    dot += qLane[qChannel] * kCache[kvChannel * shape.cacheStride + cacheToken]
                }
                logits[token] = dot * scale
            }

            let maxLogit = logits.max() ?? 0
            var weights = [Float](repeating: 0, count: shape.visibleTokens)
            var denom: Float = 0
            for token in 0..<shape.visibleTokens {
                let value = exp(logits[token] - maxLogit)
                weights[token] = value
                denom += value
            }

            for headOffset in 0..<shape.headDim {
                let channel = head * shape.headDim + headOffset
                let kvChannel = kvHead * shape.headDim + headOffset
                var accum: Float = 0
                for token in 0..<shape.visibleTokens {
                    let cacheToken = shape.tokenBase + token
                    accum += (weights[token] / denom) * vCache[kvChannel * shape.cacheStride + cacheToken]
                }
                context[channel] = accum
            }
        }

        return context
    }

    // MARK: - GQA MetalDecodeAttentionShape validation

    func test_gqa_decode_shape_accepts_valid_head_groups() throws {
        let shape = try MetalDecodeAttentionShape(
            heads: 32, kvHeads: 4, headDim: 64,
            visibleTokens: 10, cacheStride: 256, laneStride: 32
        )
        XCTAssertEqual(shape.heads, 32)
        XCTAssertEqual(shape.kvHeads, 4)
    }

    func test_gqa_decode_shape_rejects_indivisible_kvHeads() {
        XCTAssertThrowsError(
            try MetalDecodeAttentionShape(
                heads: 32, kvHeads: 5, headDim: 64,
                visibleTokens: 10, cacheStride: 256, laneStride: 32
            )
        )
    }

    func test_mha_decode_shape_defaults_kvHeads_to_heads() throws {
        let shape = try MetalDecodeAttentionShape(
            heads: 12, headDim: 64,
            visibleTokens: 10, cacheStride: 256, laneStride: 32
        )
        XCTAssertEqual(shape.kvHeads, 12)
    }

    func test_decode_shape_rejects_token_window_overflow() {
        XCTAssertThrowsError(
            try MetalDecodeAttentionShape(
                heads: 8,
                headDim: 16,
                visibleTokens: 5,
                tokenBase: 4,
                cacheStride: 8,
                laneStride: 32
            )
        )
    }
}
