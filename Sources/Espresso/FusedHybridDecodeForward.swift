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
        let controls = FusedControlColumns(dim: dim, headDim: headDim, tokenIndex: tokenIndex, ropeTheta: ropeTheta)
        for handles in surfaceHandles {
            try writeFusedControlSurfaces(handles: handles, tokenIndex: tokenIndex, controls: controls)
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

    /// Per-token control columns shared by every layer of one decode step.
    ///
    /// The mask, position one-hot, and RoPE surfaces are identical across layers, so they are
    /// computed once per step. Only the columns that change are written: the causal mask
    /// keeps `0` for every committed position (set in earlier steps / cache init) and
    /// `-1e4` beyond; the position one-hot moves by clearing the previous column; RoPE
    /// touches channels `0..<headDim` of lane 0 only.
    struct FusedControlColumns {
        let zeros: [Float]
        let ones: [Float]
        let rope: [Float]

        init(dim: Int, headDim: Int, tokenIndex: Int, ropeTheta: Float) {
            let halfDim = headDim / 2
            var rope = [Float](repeating: 0, count: headDim)
            for idx in 0..<halfDim {
                let angle = Float(tokenIndex) / powf(ropeTheta, Float(2 * idx) / Float(headDim))
                rope[idx] = cosf(angle)
                rope[halfDim + idx] = sinf(angle)
            }
            self.zeros = [Float](repeating: 0, count: dim)
            self.ones = [Float](repeating: 1, count: dim)
            self.rope = rope
        }
    }

    private static func writeFusedControlSurfaces(
        handles: FusedHybridDecodeSurfaceHandles,
        tokenIndex: Int,
        controls: FusedControlColumns
    ) throws(ANEError) {
        try writeFusedControlSurfaces(
            mask: handles.mask,
            posMask: handles.posMask,
            ropePack: handles.ropePack,
            dim: handles.dim,
            maxSeq: handles.maxSeq,
            laneSpatial: handles.laneSpatial,
            tokenIndex: tokenIndex,
            controls: controls
        )
    }

    /// Incremental fused-hybrid control writes. Visible for tests that compare
    /// this against a full-surface rewrite of mask / pos / RoPE.
    static func writeFusedControlSurfaces(
        mask: IOSurfaceRef,
        posMask: IOSurfaceRef,
        ropePack: IOSurfaceRef,
        dim: Int,
        maxSeq: Int,
        laneSpatial: Int,
        tokenIndex: Int,
        controls: FusedControlColumns
    ) throws(ANEError) {
        precondition(controls.zeros.count == dim)
        precondition(controls.ones.count == dim)
        // rope has headDim elements; headDim ≤ dim is a model invariant.
        precondition(controls.rope.count <= dim)
        do {
            try mapSurfaceIOToANEError {
                try controls.zeros.withUnsafeBufferPointer { zeroColumn in
                    if tokenIndex > 0 {
                        try SurfaceIO.writeFP16SpatialSlice(
                            to: posMask,
                            channelOffset: 0,
                            spatialIndex: tokenIndex - 1,
                            spatial: maxSeq,
                            data: zeroColumn,
                            channels: dim
                        )
                    }
                    try SurfaceIO.writeFP16SpatialSlice(
                        to: mask,
                        channelOffset: 0,
                        spatialIndex: tokenIndex,
                        spatial: maxSeq,
                        data: zeroColumn,
                        channels: dim
                    )
                }
                try controls.ones.withUnsafeBufferPointer { oneColumn in
                    try SurfaceIO.writeFP16SpatialSlice(
                        to: posMask,
                        channelOffset: 0,
                        spatialIndex: tokenIndex,
                        spatial: maxSeq,
                        data: oneColumn,
                        channels: dim
                    )
                }
                try controls.rope.withUnsafeBufferPointer { ropeColumn in
                    try SurfaceIO.writeFP16SpatialSlice(
                        to: ropePack,
                        channelOffset: 0,
                        spatialIndex: 0,
                        spatial: laneSpatial,
                        data: ropeColumn,
                        channels: controls.rope.count
                    )
                }
            }
        } catch {
            throw .invalidArguments("fused hybrid control surface write failed: \(error)")
        }
    }
}
