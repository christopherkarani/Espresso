import Foundation
import IOSurface
import ANEInterop
import ANERuntime
import ANETypes
import CPUOps

extension ForwardPass {
    public static func initializeFusedHybridDecodeCaches(
        surfaceHandles: [FusedHybridDecodeSurfaceHandles]
    ) throws(ANEError) {
        guard let first = surfaceHandles.first else { return }
        let dim = first.dim
        let maxSeq = first.maxSeq
        let laneSpatial = first.laneSpatial
        let cacheZeros = [Float](repeating: 0, count: dim * maxSeq)
        let laneZeros = [Float](repeating: 0, count: dim * laneSpatial)
        let masked = [Float](repeating: -1e4, count: dim * maxSeq)
        for handles in surfaceHandles {
            try mapSurfaceIOToANEError {
                try cacheZeros.withUnsafeBufferPointer { src in
                    try SurfaceIO.writeFP16(to: handles.kCache, data: src, channels: dim, spatial: maxSeq)
                    try SurfaceIO.writeFP16(to: handles.vCache, data: src, channels: dim, spatial: maxSeq)
                    try SurfaceIO.writeFP16(to: handles.posMask, data: src, channels: dim, spatial: maxSeq)
                }
                try masked.withUnsafeBufferPointer { src in
                    try SurfaceIO.writeFP16(to: handles.mask, data: src, channels: dim, spatial: maxSeq)
                }
                try laneZeros.withUnsafeBufferPointer { src in
                    try SurfaceIO.writeFP16(to: handles.xIn, data: src, channels: dim, spatial: laneSpatial)
                    try SurfaceIO.writeFP16(to: handles.ropePack, data: src, channels: dim, spatial: laneSpatial)
                }
            }
        }
    }

    public static func runFusedHybridDecodeTimed(
        xCur: borrowing TensorBuffer,
        kernels: borrowing LayerStorage<FusedHybridDecodeLayerKernelSet>,
        surfaceHandles: [FusedHybridDecodeSurfaceHandles],
        decodeState: inout DecodeState,
        headDim: Int,
        ropeTheta: Float,
        timings: inout HybridDecodeTimingBreakdown
    ) throws(ANEError) {
        precondition(kernels.count > 0)
        precondition(surfaceHandles.count == kernels.count)
        precondition(xCur.count == surfaceHandles[0].dim)

        let tokenIndex = try decodeState.beginTokenStep()
        let first = surfaceHandles[0]
        let dim = first.dim
        let kvDim = first.kvDim
        let maxSeq = first.maxSeq
        let laneSpatial = first.laneSpatial
        precondition(tokenIndex < maxSeq)

        var t0 = RuntimeClock.now()
        for handles in surfaceHandles {
            try writeFusedControlSurfaces(
                handles: handles,
                tokenIndex: tokenIndex,
                headDim: headDim,
                ropeTheta: ropeTheta
            )
        }
        do {
            try mapSurfaceIOToANEError {
                try xCur.withUnsafeBufferPointer { xBuf in
                    try SurfaceIO.writeFP16SpatialSlice(
                        to: first.xIn,
                        channelOffset: 0,
                        spatialIndex: 0,
                        spatial: laneSpatial,
                        data: xBuf,
                        channels: dim
                    )
                }
            }
        } catch {
            throw .invalidArguments("fused hybrid token lane write failed: \(error)")
        }
        timings.tIO += RuntimeClock.ms(RuntimeClock.now() - t0)

        for layerIndex in 0..<kernels.count {
            let handles = surfaceHandles[layerIndex]
            t0 = RuntimeClock.now()
            do {
                try kernels[layerIndex].fusedLayer.eval()
            } catch {
                throw .invalidArguments(
                    "fused hybrid N=1 eval failed at layer \(layerIndex), token \(tokenIndex): \(error)"
                )
            }
            timings.tAneQKV += RuntimeClock.ms(RuntimeClock.now() - t0)

            t0 = RuntimeClock.now()
            do {
                try mapSurfaceIOToANEError {
                    try SurfaceIO.copyFP16SpatialSlice(
                        dst: handles.kCache,
                        dstChannelOffset: 0,
                        dstSpatialIndex: tokenIndex,
                        dstSpatial: maxSeq,
                        src: handles.kNew,
                        srcChannelOffset: 0,
                        srcSpatialIndex: 0,
                        srcSpatial: laneSpatial,
                        channels: kvDim
                    )
                    try SurfaceIO.copyFP16SpatialSlice(
                        dst: handles.vCache,
                        dstChannelOffset: 0,
                        dstSpatialIndex: tokenIndex,
                        dstSpatial: maxSeq,
                        src: handles.vNew,
                        srcChannelOffset: 0,
                        srcSpatialIndex: 0,
                        srcSpatial: laneSpatial,
                        channels: kvDim
                    )
                }
            } catch {
                throw .invalidArguments("fused hybrid KV cache update failed: \(error)")
            }
            if layerIndex + 1 < surfaceHandles.count {
                do {
                    try mapSurfaceIOToANEError {
                        try SurfaceIO.copyFP16SpatialSlice(
                            dst: surfaceHandles[layerIndex + 1].xIn,
                            dstChannelOffset: 0,
                            dstSpatialIndex: 0,
                            dstSpatial: laneSpatial,
                            src: handles.xOut,
                            srcChannelOffset: 0,
                            srcSpatialIndex: 0,
                            srcSpatial: laneSpatial,
                            channels: dim
                        )
                    }
                } catch {
                    throw .invalidArguments("fused hybrid layer chain failed: \(error)")
                }
            }
            timings.tIO += RuntimeClock.ms(RuntimeClock.now() - t0)
        }

        t0 = RuntimeClock.now()
        do {
            try mapSurfaceIOToANEError {
                try xCur.withUnsafeMutableBufferPointer { out in
                    try SurfaceIO.readFP16SpatialSlice(
                        from: surfaceHandles[kernels.count - 1].xOut,
                        channelOffset: 0,
                        spatialIndex: 0,
                        spatial: laneSpatial,
                        into: out,
                        channels: dim
                    )
                }
            }
        } catch {
            throw .invalidArguments("fused hybrid final unpack failed: \(error)")
        }
        timings.tIO += RuntimeClock.ms(RuntimeClock.now() - t0)
        try decodeState.commitTokenStep(expectedIndex: tokenIndex)
    }

    private static func writeFusedControlSurfaces(
        handles: FusedHybridDecodeSurfaceHandles,
        tokenIndex: Int,
        headDim: Int,
        ropeTheta: Float
    ) throws(ANEError) {
        let dim = handles.dim
        let maxSeq = handles.maxSeq
        let laneSpatial = handles.laneSpatial
        let halfDim = headDim / 2

        var mask = [Float](repeating: -1e4, count: dim * maxSeq)
        var pos = [Float](repeating: 0, count: dim * maxSeq)
        for spatial in 0...tokenIndex {
            for channel in 0..<dim {
                mask[channel * maxSeq + spatial] = 0
            }
        }
        for channel in 0..<dim {
            pos[channel * maxSeq + tokenIndex] = 1
        }

        var rope = [Float](repeating: 0, count: dim * laneSpatial)
        for idx in 0..<halfDim {
            let angle = Float(tokenIndex) / powf(ropeTheta, Float(2 * idx) / Float(headDim))
            rope[idx * laneSpatial] = cosf(angle)
            rope[(halfDim + idx) * laneSpatial] = sinf(angle)
        }

        do {
            try mapSurfaceIOToANEError {
                try mask.withUnsafeBufferPointer { src in
                    try SurfaceIO.writeFP16(to: handles.mask, data: src, channels: dim, spatial: maxSeq)
                }
                try pos.withUnsafeBufferPointer { src in
                    try SurfaceIO.writeFP16(to: handles.posMask, data: src, channels: dim, spatial: maxSeq)
                }
                try rope.withUnsafeBufferPointer { src in
                    try SurfaceIO.writeFP16(to: handles.ropePack, data: src, channels: dim, spatial: laneSpatial)
                }
            }
        } catch {
            throw .invalidArguments("fused hybrid control surface write failed: \(error)")
        }
    }
}
