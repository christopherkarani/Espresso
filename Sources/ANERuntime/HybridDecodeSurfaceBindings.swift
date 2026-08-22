import Foundation
import IOSurface
import ANEInterop
import ANETypes

/// Bridges untyped `rethrows` boundaries (`withUnsafeBufferPointer` and friends)
/// into `throws(ANEError)` contexts. The stdlib buffer accessors erase typed
/// throws to `any Error`, so SurfaceIO failures must be re-mapped at the boundary.
@inline(__always)
public func mapSurfaceIOToANEError<R>(_ body: () throws -> R) throws(ANEError) -> R {
    do {
        return try body()
    } catch let error as SurfaceIOError {
        throw .surfaceIO(error)
    } catch let error as ANEError {
        throw error
    } catch {
        throw .surfaceIO(.interopCallFailed)
    }
}

/// Decode surfaces for the split hybrid path:
/// ANE QKV-only -> Metal attention/projection -> ANE FFN.
///
/// Supports GQA: K/V caches use `kvDim` (nKVHeads * headDim) which may differ from `dim`.
///
/// The binding knowledge here — which kernel owns which surface index, how the
/// projection/FFN kernels chain onto QKV outputs, fp16 cache sizing — belongs
/// beside `HybridDecodeKernelSet`, not in its consumers.
public struct HybridDecodeSurfaceHandles {
    public let qkvIn: IOSurfaceRef
    public let qOut: IOSurfaceRef
    public let kOut: IOSurfaceRef
    public let vOut: IOSurfaceRef
    public let projectionContextIn: IOSurfaceRef
    public let projectionResidualIn: IOSurfaceRef
    public let projectionOut: IOSurfaceRef
    public let ffnIn: IOSurfaceRef
    public let ffnOut: IOSurfaceRef
    public let kCacheFull: IOSurfaceRef
    public let vCacheFull: IOSurfaceRef
    public let zeroLane: IOSurfaceRef
    public let maxSeq: Int
    public let laneSpatial: Int

    public let dim: Int
    public let qDim: Int
    public let kvDim: Int

    public init(
        kernels: borrowing HybridDecodeKernelSet,
        logicalMaxSeq: Int? = nil,
        dim: Int = ModelConfig.dim,
        qDim: Int? = nil,
        kvDim: Int? = nil
    ) throws(ANEError) {
        let resolvedQDim = qDim ?? dim
        let resolvedKVDim = kvDim ?? dim
        let qkvIn = try kernels.decodeQKVOnly.inputSurface(at: 0)
        let kOut = try kernels.decodeQKVOnly.outputSurface(at: 0)
        let qOut = try kernels.decodeQKVOnly.outputSurface(at: 1)
        let vOut = try kernels.decodeQKVOnly.outputSurface(at: 2)
        let projectionContextIn = try kernels.decodeProjection.inputSurface(at: 0)
        let projectionOut = try kernels.decodeProjection.outputSurface(at: 0)
        let ffnOut = try (kernels.usesFusedPostAttention
            ? kernels.decodeProjection.outputSurface(at: 0)
            : kernels.decodeFFN.outputSurface(at: 0))
        try kernels.decodeProjection.rebindInput(at: 1, to: qkvIn)
        if !kernels.usesFusedPostAttention {
            try kernels.decodeFFN.rebindInput(at: 0, to: projectionOut)
        }

        self.dim = dim
        self.qDim = resolvedQDim
        self.kvDim = resolvedKVDim
        self.qkvIn = qkvIn
        self.kOut = kOut
        self.qOut = qOut
        self.vOut = vOut
        self.projectionContextIn = projectionContextIn
        self.projectionResidualIn = qkvIn
        self.projectionOut = projectionOut
        self.ffnIn = kernels.usesFusedPostAttention ? qkvIn : projectionOut
        self.ffnOut = ffnOut
        self.maxSeq = logicalMaxSeq ?? kernels.maxSeq
        self.laneSpatial = kernels.laneSpatial

        guard let kCacheFull = ane_interop_create_surface(resolvedKVDim * self.maxSeq * 2),
              let vCacheFull = ane_interop_create_surface(resolvedKVDim * self.maxSeq * 2),
              let zeroLane = ane_interop_create_surface(dim * kernels.laneSpatial * 2) else {
            throw .surfaceAllocationFailed
        }
        self.kCacheFull = kCacheFull
        self.vCacheFull = vCacheFull
        self.zeroLane = zeroLane

        let zeroLaneValues = Array(repeating: Float(0), count: dim * kernels.laneSpatial)
        try mapSurfaceIOToANEError { try zeroLaneValues.withUnsafeBufferPointer { src in
            try SurfaceIO.writeFP16(to: zeroLane, data: src, channels: dim, spatial: kernels.laneSpatial)
        } }
    }
}

/// Surfaces for one Phase-11 `max_N = 1` fused hybrid layer.
///
/// Inputs (alphabetical): kCache, mask, posMask, ropePack, vCache, x
/// Outputs (alphabetical): kNew, vNew, xOut
///
/// Binding knowledge travels with the kernel set that defines the layout;
/// shared mask/rope surfaces can be rebound by the caller without knowing
/// positional indices.
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
