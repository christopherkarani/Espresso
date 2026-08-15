import Foundation
import IOSurface
import ANEInterop
import ANERuntime
import ANETypes
import CPUOps

/// Surfaces for one Phase-11 `max_N = 1` fused layer.
///
/// Inputs (alphabetical): kCache, mask, posMask, ropePack, vCache, x
/// Outputs (alphabetical): kNew, vNew, xOut
public struct FusedHybridDecodeSurfaceHandles {
    public let kCache: IOSurfaceRef
    public let mask: IOSurfaceRef
    public let posMask: IOSurfaceRef
    public let ropePack: IOSurfaceRef
    public let vCache: IOSurfaceRef
    public let xIn: IOSurfaceRef
    public let kNew: IOSurfaceRef
    public let vNew: IOSurfaceRef
    public let xOut: IOSurfaceRef
    public let maxSeq: Int
    public let laneSpatial: Int
    public let dim: Int
    public let kvDim: Int

    public init(
        kernels: borrowing FusedHybridDecodeLayerKernelSet,
        sharedMask: IOSurfaceRef? = nil,
        sharedPosMask: IOSurfaceRef? = nil,
        sharedRopePack: IOSurfaceRef? = nil
    ) throws(ANEError) {
        self.kCache = try kernels.fusedLayer.inputSurface(at: 0)
        let ownedMask = try kernels.fusedLayer.inputSurface(at: 1)
        let ownedPos = try kernels.fusedLayer.inputSurface(at: 2)
        let ownedRope = try kernels.fusedLayer.inputSurface(at: 3)
        self.vCache = try kernels.fusedLayer.inputSurface(at: 4)
        self.xIn = try kernels.fusedLayer.inputSurface(at: 5)
        self.kNew = try kernels.fusedLayer.outputSurface(at: 0)
        self.vNew = try kernels.fusedLayer.outputSurface(at: 1)
        self.xOut = try kernels.fusedLayer.outputSurface(at: 2)
        self.maxSeq = kernels.maxSeq
        self.laneSpatial = kernels.laneSpatial
        self.dim = kernels.dim
        self.kvDim = kernels.kvDim

        if let sharedMask {
            try kernels.fusedLayer.rebindInput(at: 1, to: sharedMask)
            self.mask = sharedMask
        } else {
            self.mask = ownedMask
        }
        if let sharedPosMask {
            try kernels.fusedLayer.rebindInput(at: 2, to: sharedPosMask)
            self.posMask = sharedPosMask
        } else {
            self.posMask = ownedPos
        }
        if let sharedRopePack {
            try kernels.fusedLayer.rebindInput(at: 3, to: sharedRopePack)
            self.ropePack = sharedRopePack
        } else {
            self.ropePack = ownedRope
        }
    }
}

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
