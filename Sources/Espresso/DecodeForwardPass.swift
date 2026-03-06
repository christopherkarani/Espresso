import Foundation
import IOSurface
import ANEInterop
import ANERuntime
import ANETypes

/// Decode surfaces for one layer.
public struct DecodeSurfaceHandles {
    public let attnIn: IOSurfaceRef
    public let kCache: IOSurfaceRef
    public let vCache: IOSurfaceRef
    public let maskCache: IOSurfaceRef
    public let kCacheFull: IOSurfaceRef
    public let vCacheFull: IOSurfaceRef
    public let maskCacheFull: IOSurfaceRef
    public let attnX2Out: IOSurfaceRef
    public let attnKOut: IOSurfaceRef
    public let attnVOut: IOSurfaceRef
    public let ffnIn: IOSurfaceRef
    public let ffnOut: IOSurfaceRef
    public let zeroLane: IOSurfaceRef
    public let maskedLane: IOSurfaceRef
    public let tokenScratch: IOSurfaceRef
    public let maxSeq: Int
    public let kernelMaxSeq: Int
    public let laneSpatial: Int

    public init(kernels: borrowing DecodeKernelSet, logicalMaxSeq: Int? = nil) throws(ANEError) {
        self.attnIn = try kernels.decodeAttnQKV.inputSurface(at: 0)
        self.kCache = try kernels.decodeAttnQKV.inputSurface(at: 1)
        self.vCache = try kernels.decodeAttnQKV.inputSurface(at: 2)
        self.maskCache = try kernels.decodeAttnQKV.inputSurface(at: 3)
        self.attnX2Out = try kernels.decodeAttnQKV.outputSurface(at: 0)
        self.attnKOut = try kernels.decodeAttnQKV.outputSurface(at: 1)
        self.attnVOut = try kernels.decodeAttnQKV.outputSurface(at: 2)
        self.ffnIn = try kernels.decodeFFN.inputSurface(at: 0)
        self.ffnOut = try kernels.decodeFFN.outputSurface(at: 0)
        self.maxSeq = logicalMaxSeq ?? kernels.maxSeq
        self.kernelMaxSeq = kernels.kernelMaxSeq
        self.laneSpatial = kernels.laneSpatial

        guard let zeroLane = ane_interop_create_surface(ModelConfig.dim * kernels.laneSpatial * 2),
              let maskedLane = ane_interop_create_surface(ModelConfig.dim * kernels.laneSpatial * 2),
              let tokenScratch = ane_interop_create_surface(ModelConfig.dim * 2) else {
            throw .surfaceAllocationFailed
        }
        self.zeroLane = zeroLane
        self.maskedLane = maskedLane
        self.tokenScratch = tokenScratch

        if self.maxSeq == self.kernelMaxSeq {
            self.kCacheFull = self.kCache
            self.vCacheFull = self.vCache
            self.maskCacheFull = self.maskCache
        } else {
            guard let kCacheFull = ane_interop_create_surface(ModelConfig.dim * self.maxSeq * 2),
                  let vCacheFull = ane_interop_create_surface(ModelConfig.dim * self.maxSeq * 2),
                  let maskCacheFull = ane_interop_create_surface(ModelConfig.dim * self.maxSeq * 2) else {
                throw .surfaceAllocationFailed
            }
            self.kCacheFull = kCacheFull
            self.vCacheFull = vCacheFull
            self.maskCacheFull = maskCacheFull
        }

        let zeroLaneValues = Array(repeating: Float(0), count: ModelConfig.dim * kernels.laneSpatial)
        let maskFill: Float = ProcessInfo.processInfo.environment["ESPRESSO_DECODE_MASK_INIT_ZERO"] == "1" ? 0 : -1e4
        let maskedLaneValues = Array(repeating: maskFill, count: ModelConfig.dim * kernels.laneSpatial)
        zeroLaneValues.withUnsafeBufferPointer { src in
            SurfaceIO.writeFP16(to: zeroLane, data: src, channels: ModelConfig.dim, spatial: kernels.laneSpatial)
        }
        maskedLaneValues.withUnsafeBufferPointer { src in
            SurfaceIO.writeFP16(to: maskedLane, data: src, channels: ModelConfig.dim, spatial: kernels.laneSpatial)
        }
    }
}

public struct DecodeState: Sendable {
    public let maxSeq: Int
    public private(set) var step: Int

    public init(maxSeq: Int, step: Int = 0) throws(ANEError) {
        guard maxSeq > 0 else {
            throw .invalidArguments("decode maxSeq must be > 0")
        }
        guard step >= 0, step <= maxSeq else {
            throw .invalidArguments("decode step must be in 0...\(maxSeq), got \(step)")
        }
        self.maxSeq = maxSeq
        self.step = step
    }

    public var visibleTokenCount: Int { step }

    public mutating func beginTokenStep() throws(ANEError) -> Int {
        guard step < maxSeq else {
            throw .invalidArguments("decode step overflow: step=\(step), maxSeq=\(maxSeq)")
        }
        return step
    }

    public mutating func commitTokenStep(expectedIndex: Int) throws(ANEError) {
        guard expectedIndex == step else {
            throw .invalidArguments("decode state mismatch: expected \(expectedIndex), actual \(step)")
        }
        step += 1
    }

    public mutating func reset() {
        step = 0
    }
}

enum DecodeTiling {
    @inline(__always)
    static func windowBase(for tokenIndex: Int, laneSpatial: Int) -> Int {
        precondition(tokenIndex >= 0)
        precondition(laneSpatial > 0)
        return (tokenIndex / laneSpatial) * laneSpatial
    }

    @inline(__always)
    static func localIndex(for tokenIndex: Int, laneSpatial: Int) -> Int {
        tokenIndex - windowBase(for: tokenIndex, laneSpatial: laneSpatial)
    }

    @inline(__always)
    static func shouldSyncWindow(for tokenIndex: Int, laneSpatial: Int) -> Bool {
        precondition(tokenIndex >= 0)
        precondition(laneSpatial > 0)
        return tokenIndex > 0 && localIndex(for: tokenIndex, laneSpatial: laneSpatial) == 0
    }
}

enum DecodeRuntimeOptions {
    @inline(__always)
    static func forceFullWindowSync(env: [String: String]) -> Bool {
        env["ESPRESSO_DECODE_FORCE_FULL_WINDOW_SYNC"] == "1"
    }

    @inline(__always)
    static var forceFullWindowSync: Bool {
        forceFullWindowSync(env: ProcessInfo.processInfo.environment)
    }
}

enum FusedTwoLayerCacheLayout {
    static let packedKVChannels = 4 * ModelConfig.dim
    static let layer0K = 0
    static let layer0V = ModelConfig.dim
    static let layer1K = 2 * ModelConfig.dim
    static let layer1V = 3 * ModelConfig.dim
}

public struct DecodeKernelProfile: Sendable {
    public struct LayerSamples: Sendable {
        public var attnEvalUS: [Double] = []
        public var attnHwNS: [UInt64] = []
        public var attnHostOverheadUS: [Double] = []
        public var selfMaskUpdateUS: [Double] = []
        public var kCacheUpdateUS: [Double] = []
        public var vCacheUpdateUS: [Double] = []
        public var maskUpdateUS: [Double] = []
        public var x2ToFfnCopyUS: [Double] = []
        public var ffnEvalUS: [Double] = []
        public var ffnHwNS: [UInt64] = []
        public var ffnHostOverheadUS: [Double] = []
        public var ffnToNextAttnCopyUS: [Double] = []
    }

    public private(set) var layers: [LayerSamples]

    public init(layerCount: Int, reservedSamplesPerLayer: Int) {
        precondition(layerCount >= 0)
        self.layers = Array(repeating: LayerSamples(), count: layerCount)
        if reservedSamplesPerLayer > 0 {
            for idx in layers.indices {
                layers[idx].attnEvalUS.reserveCapacity(reservedSamplesPerLayer)
                layers[idx].attnHwNS.reserveCapacity(reservedSamplesPerLayer)
                layers[idx].attnHostOverheadUS.reserveCapacity(reservedSamplesPerLayer)
                layers[idx].selfMaskUpdateUS.reserveCapacity(reservedSamplesPerLayer)
                layers[idx].kCacheUpdateUS.reserveCapacity(reservedSamplesPerLayer)
                layers[idx].vCacheUpdateUS.reserveCapacity(reservedSamplesPerLayer)
                layers[idx].maskUpdateUS.reserveCapacity(reservedSamplesPerLayer)
                layers[idx].x2ToFfnCopyUS.reserveCapacity(reservedSamplesPerLayer)
                layers[idx].ffnEvalUS.reserveCapacity(reservedSamplesPerLayer)
                layers[idx].ffnHwNS.reserveCapacity(reservedSamplesPerLayer)
                layers[idx].ffnHostOverheadUS.reserveCapacity(reservedSamplesPerLayer)
                layers[idx].ffnToNextAttnCopyUS.reserveCapacity(reservedSamplesPerLayer)
            }
        }
    }

    public mutating func record(
        layerIndex: Int,
        attnEvalUS: Double,
        attnHwNS: UInt64,
        attnHostOverheadUS: Double,
        selfMaskUpdateUS: Double,
        kCacheUpdateUS: Double,
        vCacheUpdateUS: Double,
        maskUpdateUS: Double,
        x2ToFfnCopyUS: Double,
        ffnEvalUS: Double,
        ffnHwNS: UInt64,
        ffnHostOverheadUS: Double,
        ffnToNextAttnCopyUS: Double
    ) {
        precondition(layerIndex >= 0 && layerIndex < layers.count)
        layers[layerIndex].attnEvalUS.append(attnEvalUS)
        layers[layerIndex].attnHwNS.append(attnHwNS)
        layers[layerIndex].attnHostOverheadUS.append(attnHostOverheadUS)
        layers[layerIndex].selfMaskUpdateUS.append(selfMaskUpdateUS)
        layers[layerIndex].kCacheUpdateUS.append(kCacheUpdateUS)
        layers[layerIndex].vCacheUpdateUS.append(vCacheUpdateUS)
        layers[layerIndex].maskUpdateUS.append(maskUpdateUS)
        layers[layerIndex].x2ToFfnCopyUS.append(x2ToFfnCopyUS)
        layers[layerIndex].ffnEvalUS.append(ffnEvalUS)
        layers[layerIndex].ffnHwNS.append(ffnHwNS)
        layers[layerIndex].ffnHostOverheadUS.append(ffnHostOverheadUS)
        layers[layerIndex].ffnToNextAttnCopyUS.append(ffnToNextAttnCopyUS)
    }
}

public final class DecodeKernelProfiler: @unchecked Sendable {
    public private(set) var profile: DecodeKernelProfile

    public init(layerCount: Int, reservedSamplesPerLayer: Int) {
        self.profile = DecodeKernelProfile(layerCount: layerCount, reservedSamplesPerLayer: reservedSamplesPerLayer)
    }

    public func record(
        layerIndex: Int,
        attnEvalUS: Double,
        attnHwNS: UInt64,
        attnHostOverheadUS: Double,
        selfMaskUpdateUS: Double,
        kCacheUpdateUS: Double,
        vCacheUpdateUS: Double,
        maskUpdateUS: Double,
        x2ToFfnCopyUS: Double,
        ffnEvalUS: Double,
        ffnHwNS: UInt64,
        ffnHostOverheadUS: Double,
        ffnToNextAttnCopyUS: Double
    ) {
        profile.record(
            layerIndex: layerIndex,
            attnEvalUS: attnEvalUS,
            attnHwNS: attnHwNS,
            attnHostOverheadUS: attnHostOverheadUS,
            selfMaskUpdateUS: selfMaskUpdateUS,
            kCacheUpdateUS: kCacheUpdateUS,
            vCacheUpdateUS: vCacheUpdateUS,
            maskUpdateUS: maskUpdateUS,
            x2ToFfnCopyUS: x2ToFfnCopyUS,
            ffnEvalUS: ffnEvalUS,
            ffnHwNS: ffnHwNS,
            ffnHostOverheadUS: ffnHostOverheadUS,
            ffnToNextAttnCopyUS: ffnToNextAttnCopyUS
        )
    }
}

public struct HybridDecodeTimingBreakdown: Sendable {
    public var tAneQKV: Double
    public var tMetal: Double
    public var tAneFFN: Double
    public var tIO: Double

    public init(
        tAneQKV: Double = 0,
        tMetal: Double = 0,
        tAneFFN: Double = 0,
        tIO: Double = 0
    ) {
        self.tAneQKV = tAneQKV
        self.tMetal = tMetal
        self.tAneFFN = tAneFFN
        self.tIO = tIO
    }
}

/// Decode surfaces for the split hybrid path:
/// ANE QKV-only -> Metal attention/projection -> ANE FFN.
public struct HybridDecodeSurfaceHandles {
    public let qkvIn: IOSurfaceRef
    public let qOut: IOSurfaceRef
    public let kOut: IOSurfaceRef
    public let vOut: IOSurfaceRef
    public let ffnIn: IOSurfaceRef
    public let ffnOut: IOSurfaceRef
    public let kCacheFull: IOSurfaceRef
    public let vCacheFull: IOSurfaceRef
    public let zeroLane: IOSurfaceRef
    public let maxSeq: Int
    public let laneSpatial: Int

    public init(kernels: borrowing HybridDecodeKernelSet, logicalMaxSeq: Int? = nil) throws(ANEError) {
        self.qkvIn = try kernels.decodeQKVOnly.inputSurface(at: 0)
        self.qOut = try kernels.decodeQKVOnly.outputSurface(at: 0)
        self.kOut = try kernels.decodeQKVOnly.outputSurface(at: 1)
        self.vOut = try kernels.decodeQKVOnly.outputSurface(at: 2)
        self.ffnIn = try kernels.decodeFFN.inputSurface(at: 0)
        self.ffnOut = try kernels.decodeFFN.outputSurface(at: 0)
        self.maxSeq = logicalMaxSeq ?? kernels.maxSeq
        self.laneSpatial = kernels.laneSpatial

        guard let kCacheFull = ane_interop_create_surface(ModelConfig.dim * self.maxSeq * 2),
              let vCacheFull = ane_interop_create_surface(ModelConfig.dim * self.maxSeq * 2),
              let zeroLane = ane_interop_create_surface(ModelConfig.dim * kernels.laneSpatial * 2) else {
            throw .surfaceAllocationFailed
        }
        self.kCacheFull = kCacheFull
        self.vCacheFull = vCacheFull
        self.zeroLane = zeroLane

        let zeroLaneValues = Array(repeating: Float(0), count: ModelConfig.dim * kernels.laneSpatial)
        zeroLaneValues.withUnsafeBufferPointer { src in
            SurfaceIO.writeFP16(to: zeroLane, data: src, channels: ModelConfig.dim, spatial: kernels.laneSpatial)
        }
    }
}

/// Decode surfaces for one layer using the fused (attn + FFN) kernel.
///
/// The fused kernel has 4 inputs and 3 outputs:
/// - Inputs:  x (0), kCache (1), vCache (2), maskCache (3)
/// - Outputs: xNext (0), kOut (1), vOut (2)
///
/// No separate FFN surfaces are needed — the attn→FFN copy is eliminated.
public struct FusedDecodeSurfaceHandles {
    public let fusedIn: IOSurfaceRef       // x input [1, dim, 1, laneSpatial]
    public let kCache: IOSurfaceRef        // k cache input [1, dim, 1, kernelMaxSeq]
    public let vCache: IOSurfaceRef        // v cache input [1, dim, 1, kernelMaxSeq]
    public let maskCache: IOSurfaceRef     // mask cache input [1, dim, 1, kernelMaxSeq]
    public let kCacheFull: IOSurfaceRef    // full k cache [1, dim, 1, maxSeq]
    public let vCacheFull: IOSurfaceRef    // full v cache [1, dim, 1, maxSeq]
    public let maskCacheFull: IOSurfaceRef // full mask cache [1, dim, 1, maxSeq]
    public let xNextOut: IOSurfaceRef      // fused output 0 [1, dim, 1, laneSpatial]
    public let kOut: IOSurfaceRef          // fused output 1 [1, dim, 1, laneSpatial]
    public let vOut: IOSurfaceRef          // fused output 2 [1, dim, 1, laneSpatial]
    public let zeroLane: IOSurfaceRef
    public let maskedLane: IOSurfaceRef
    public let tokenScratch: IOSurfaceRef
    public let maxSeq: Int
    public let kernelMaxSeq: Int
    public let laneSpatial: Int

    public init(kernels: borrowing FusedDecodeKernelSet, logicalMaxSeq: Int? = nil) throws(ANEError) {
        self.fusedIn = try kernels.fusedLayer.inputSurface(at: 0)
        self.kCache = try kernels.fusedLayer.inputSurface(at: 1)
        self.vCache = try kernels.fusedLayer.inputSurface(at: 2)
        self.maskCache = try kernels.fusedLayer.inputSurface(at: 3)
        self.xNextOut = try kernels.fusedLayer.outputSurface(at: 0)
        self.kOut = try kernels.fusedLayer.outputSurface(at: 1)
        self.vOut = try kernels.fusedLayer.outputSurface(at: 2)
        self.maxSeq = logicalMaxSeq ?? kernels.maxSeq
        self.kernelMaxSeq = kernels.kernelMaxSeq
        self.laneSpatial = kernels.laneSpatial

        guard let zeroLane = ane_interop_create_surface(ModelConfig.dim * kernels.laneSpatial * 2),
              let maskedLane = ane_interop_create_surface(ModelConfig.dim * kernels.laneSpatial * 2),
              let tokenScratch = ane_interop_create_surface(ModelConfig.dim * 2) else {
            throw .surfaceAllocationFailed
        }
        self.zeroLane = zeroLane
        self.maskedLane = maskedLane
        self.tokenScratch = tokenScratch

        if self.maxSeq == self.kernelMaxSeq {
            self.kCacheFull = self.kCache
            self.vCacheFull = self.vCache
            self.maskCacheFull = self.maskCache
        } else {
            guard let kCacheFull = ane_interop_create_surface(ModelConfig.dim * self.maxSeq * 2),
                  let vCacheFull = ane_interop_create_surface(ModelConfig.dim * self.maxSeq * 2),
                  let maskCacheFull = ane_interop_create_surface(ModelConfig.dim * self.maxSeq * 2) else {
                throw .surfaceAllocationFailed
            }
            self.kCacheFull = kCacheFull
            self.vCacheFull = vCacheFull
            self.maskCacheFull = maskCacheFull
        }

        let zeroLaneValues = Array(repeating: Float(0), count: ModelConfig.dim * kernels.laneSpatial)
        let maskFill: Float = ProcessInfo.processInfo.environment["ESPRESSO_DECODE_MASK_INIT_ZERO"] == "1" ? 0 : -1e4
        let maskedLaneValues = Array(repeating: maskFill, count: ModelConfig.dim * kernels.laneSpatial)
        zeroLaneValues.withUnsafeBufferPointer { src in
            SurfaceIO.writeFP16(to: zeroLane, data: src, channels: ModelConfig.dim, spatial: kernels.laneSpatial)
        }
        maskedLaneValues.withUnsafeBufferPointer { src in
            SurfaceIO.writeFP16(to: maskedLane, data: src, channels: ModelConfig.dim, spatial: kernels.laneSpatial)
        }
    }
}

/// Decode surfaces for one fused two-layer kernel.
///
/// The fused kernel has 2 inputs and 3 outputs:
/// - Inputs:  x (0), packedCache (1)
/// - Outputs: xNext (0), packedK (1), packedV (2)
public struct FusedTwoLayerDecodeSurfaceHandles {
    public let fusedIn: IOSurfaceRef
    public let packedKVCache: IOSurfaceRef
    public let packedKVCacheFull: IOSurfaceRef
    public let maskCache0: IOSurfaceRef
    public let maskCache0Full: IOSurfaceRef
    public let maskCache1: IOSurfaceRef
    public let maskCache1Full: IOSurfaceRef
    public let xNextOut: IOSurfaceRef
    public let kPackedOut: IOSurfaceRef
    public let vPackedOut: IOSurfaceRef
    public let zeroLane: IOSurfaceRef
    public let zeroPackedKVLane: IOSurfaceRef
    public let maskedLane: IOSurfaceRef
    public let maxSeq: Int
    public let kernelMaxSeq: Int
    public let laneSpatial: Int

    public init(kernels: borrowing FusedTwoLayerDecodeKernelSet, logicalMaxSeq: Int? = nil) throws(ANEError) {
        self.fusedIn = try kernels.fusedPair.inputSurface(at: 0)
        self.packedKVCache = try kernels.fusedPair.inputSurface(at: 1)
        self.maskCache0 = try kernels.fusedPair.inputSurface(at: 2)
        self.maskCache1 = try kernels.fusedPair.inputSurface(at: 3)
        self.xNextOut = try kernels.fusedPair.outputSurface(at: 0)
        self.kPackedOut = try kernels.fusedPair.outputSurface(at: 1)
        self.vPackedOut = try kernels.fusedPair.outputSurface(at: 2)
        self.maxSeq = logicalMaxSeq ?? kernels.maxSeq
        self.kernelMaxSeq = kernels.kernelMaxSeq
        self.laneSpatial = kernels.laneSpatial

        let packedKVChannels = FusedTwoLayerCacheLayout.packedKVChannels
        guard let zeroLane = ane_interop_create_surface(ModelConfig.dim * kernels.laneSpatial * 2),
              let zeroPackedKVLane = ane_interop_create_surface(packedKVChannels * kernels.laneSpatial * 2),
              let maskedLane = ane_interop_create_surface(ModelConfig.dim * kernels.laneSpatial * 2) else {
            throw .surfaceAllocationFailed
        }
        self.zeroLane = zeroLane
        self.zeroPackedKVLane = zeroPackedKVLane
        self.maskedLane = maskedLane

        if self.maxSeq == self.kernelMaxSeq {
            self.packedKVCacheFull = self.packedKVCache
            self.maskCache0Full = self.maskCache0
            self.maskCache1Full = self.maskCache1
        } else {
            guard let packedKVCacheFull = ane_interop_create_surface(packedKVChannels * self.maxSeq * 2),
                  let maskCache0Full = ane_interop_create_surface(ModelConfig.dim * self.maxSeq * 2),
                  let maskCache1Full = ane_interop_create_surface(ModelConfig.dim * self.maxSeq * 2) else {
                throw .surfaceAllocationFailed
            }
            self.packedKVCacheFull = packedKVCacheFull
            self.maskCache0Full = maskCache0Full
            self.maskCache1Full = maskCache1Full
        }

        let zeroLaneValues = Array(repeating: Float(0), count: ModelConfig.dim * kernels.laneSpatial)
        let maskFill: Float = ProcessInfo.processInfo.environment["ESPRESSO_DECODE_MASK_INIT_ZERO"] == "1" ? 0 : -1e4
        let zeroPackedKVValues = Array(repeating: Float(0), count: packedKVChannels * kernels.laneSpatial)
        let maskedValues = Array(repeating: maskFill, count: ModelConfig.dim * kernels.laneSpatial)

        zeroLaneValues.withUnsafeBufferPointer { src in
            SurfaceIO.writeFP16(to: zeroLane, data: src, channels: ModelConfig.dim, spatial: kernels.laneSpatial)
        }
        zeroPackedKVValues.withUnsafeBufferPointer { src in
            SurfaceIO.writeFP16(to: zeroPackedKVLane, data: src, channels: packedKVChannels, spatial: kernels.laneSpatial)
        }
        maskedValues.withUnsafeBufferPointer { src in
            SurfaceIO.writeFP16(to: maskedLane, data: src, channels: ModelConfig.dim, spatial: kernels.laneSpatial)
        }
    }
}

public extension ForwardPass {
    private static func synchronizeDecodeWindowCaches(
        handles: DecodeSurfaceHandles,
        windowBase: Int,
        dim: Int
    ) throws(ANEError) {
        let windowSpatial = handles.kernelMaxSeq
        for windowIndex in 0..<windowSpatial {
            let globalIndex = windowBase + windowIndex
            if globalIndex < handles.maxSeq {
                do {
                    try SurfaceIO.copyFP16SpatialSlice(
                        dst: handles.kCache,
                        dstChannelOffset: 0,
                        dstSpatialIndex: windowIndex,
                        dstSpatial: windowSpatial,
                        src: handles.kCacheFull,
                        srcChannelOffset: 0,
                        srcSpatialIndex: globalIndex,
                        srcSpatial: handles.maxSeq,
                        channels: dim
                    )
                    try SurfaceIO.copyFP16SpatialSlice(
                        dst: handles.vCache,
                        dstChannelOffset: 0,
                        dstSpatialIndex: windowIndex,
                        dstSpatial: windowSpatial,
                        src: handles.vCacheFull,
                        srcChannelOffset: 0,
                        srcSpatialIndex: globalIndex,
                        srcSpatial: handles.maxSeq,
                        channels: dim
                    )
                    try SurfaceIO.copyFP16SpatialSlice(
                        dst: handles.maskCache,
                        dstChannelOffset: 0,
                        dstSpatialIndex: windowIndex,
                        dstSpatial: windowSpatial,
                        src: handles.maskCacheFull,
                        srcChannelOffset: 0,
                        srcSpatialIndex: globalIndex,
                        srcSpatial: handles.maxSeq,
                        channels: dim
                    )
                } catch {
                    throw .invalidArguments("decode cache window sync failed at globalIndex=\(globalIndex): \(error)")
                }
            } else {
                do {
                    try SurfaceIO.copyFP16SpatialSlice(
                        dst: handles.kCache,
                        dstChannelOffset: 0,
                        dstSpatialIndex: windowIndex,
                        dstSpatial: windowSpatial,
                        src: handles.zeroLane,
                        srcChannelOffset: 0,
                        srcSpatialIndex: 0,
                        srcSpatial: handles.laneSpatial,
                        channels: dim
                    )
                    try SurfaceIO.copyFP16SpatialSlice(
                        dst: handles.vCache,
                        dstChannelOffset: 0,
                        dstSpatialIndex: windowIndex,
                        dstSpatial: windowSpatial,
                        src: handles.zeroLane,
                        srcChannelOffset: 0,
                        srcSpatialIndex: 0,
                        srcSpatial: handles.laneSpatial,
                        channels: dim
                    )
                    try SurfaceIO.copyFP16SpatialSlice(
                        dst: handles.maskCache,
                        dstChannelOffset: 0,
                        dstSpatialIndex: windowIndex,
                        dstSpatial: windowSpatial,
                        src: handles.maskedLane,
                        srcChannelOffset: 0,
                        srcSpatialIndex: 0,
                        srcSpatial: handles.laneSpatial,
                        channels: dim
                    )
                } catch {
                    throw .invalidArguments("decode padded-window fill failed at windowIndex=\(windowIndex): \(error)")
                }
            }
        }
    }

    static func initializeDecodeCachesAndMask(
        surfaceHandles: [DecodeSurfaceHandles],
        dim: Int = ModelConfig.dim
    ) {
        precondition(dim > 0)
        guard let first = surfaceHandles.first else { return }
        let maxSeq = first.maxSeq
        precondition(maxSeq > 0)

        let zeroCache = Array(repeating: Float(0), count: dim * maxSeq)
        let maskFill: Float = ProcessInfo.processInfo.environment["ESPRESSO_DECODE_MASK_INIT_ZERO"] == "1" ? 0 : -1e4
        let masked = Array(repeating: maskFill, count: dim * maxSeq)

        for handles in surfaceHandles {
            precondition(handles.maxSeq == maxSeq)
            zeroCache.withUnsafeBufferPointer { src in
                SurfaceIO.writeFP16(to: handles.kCacheFull, data: src, channels: dim, spatial: maxSeq)
                SurfaceIO.writeFP16(to: handles.vCacheFull, data: src, channels: dim, spatial: maxSeq)
            }
            masked.withUnsafeBufferPointer { src in
                SurfaceIO.writeFP16(to: handles.maskCacheFull, data: src, channels: dim, spatial: maxSeq)
            }
            if handles.maxSeq != handles.kernelMaxSeq {
                do {
                    try synchronizeDecodeWindowCaches(handles: handles, windowBase: 0, dim: dim)
                } catch {
                    preconditionFailure("decode window init failed: \(error)")
                }
            }
            do {
                try SurfaceIO.copyFP16(
                    dst: handles.attnIn,
                    dstChannelOffset: 0,
                    src: handles.zeroLane,
                    srcChannelOffset: 0,
                    channels: dim,
                    spatial: handles.laneSpatial
                )
                try SurfaceIO.copyFP16(
                    dst: handles.ffnIn,
                    dstChannelOffset: 0,
                    src: handles.zeroLane,
                    srcChannelOffset: 0,
                    channels: dim,
                    spatial: handles.laneSpatial
                )
            } catch {
                preconditionFailure("decode lane zero-init failed: \(error)")
            }
        }
    }

    static func initializeHybridDecodeCaches(
        surfaceHandles: [HybridDecodeSurfaceHandles],
        dim: Int = ModelConfig.dim
    ) {
        precondition(dim > 0)
        guard let first = surfaceHandles.first else { return }
        let maxSeq = first.maxSeq
        precondition(maxSeq > 0)

        let zeroCache = Array(repeating: Float(0), count: dim * maxSeq)
        for handles in surfaceHandles {
            precondition(handles.maxSeq == maxSeq)
            zeroCache.withUnsafeBufferPointer { src in
                SurfaceIO.writeFP16(to: handles.kCacheFull, data: src, channels: dim, spatial: maxSeq)
                SurfaceIO.writeFP16(to: handles.vCacheFull, data: src, channels: dim, spatial: maxSeq)
            }
            do {
                try SurfaceIO.copyFP16(
                    dst: handles.qkvIn,
                    dstChannelOffset: 0,
                    src: handles.zeroLane,
                    srcChannelOffset: 0,
                    channels: dim,
                    spatial: handles.laneSpatial
                )
                try SurfaceIO.copyFP16(
                    dst: handles.ffnIn,
                    dstChannelOffset: 0,
                    src: handles.zeroLane,
                    srcChannelOffset: 0,
                    channels: dim,
                    spatial: handles.laneSpatial
                )
            } catch {
                preconditionFailure("hybrid decode lane zero-init failed: \(error)")
            }
        }
    }

    static func runHybridDecodeTimed(
        xCur: borrowing TensorBuffer,
        kernels: borrowing LayerStorage<HybridDecodeKernelSet>,
        surfaceHandles: [HybridDecodeSurfaceHandles],
        metalAttention: MetalAttentionKernel,
        decodeState: inout DecodeState,
        dim: Int = ModelConfig.dim,
        timings: inout HybridDecodeTimingBreakdown
    ) throws(ANEError) {
        precondition(kernels.count > 0)
        precondition(surfaceHandles.count == kernels.count)
        precondition(dim > 0)
        precondition(xCur.count == dim)

        let maxSeq = decodeState.maxSeq
        for handles in surfaceHandles {
            precondition(handles.maxSeq == maxSeq)
        }

        let tokenIndex = try decodeState.beginTokenStep()
        let visibleTokens = tokenIndex + 1
        let laneSpatial = surfaceHandles[0].laneSpatial
        precondition(laneSpatial > 0)

        var t0 = RuntimeClock.now()
        do {
            try xCur.withUnsafeBufferPointer { xBuf in
                try SurfaceIO.writeFP16SpatialSlice(
                    to: surfaceHandles[0].qkvIn,
                    channelOffset: 0,
                    spatialIndex: 0,
                    spatial: laneSpatial,
                    data: xBuf,
                    channels: dim
                )
            }
        } catch {
            throw .invalidArguments("hybrid decode token lane write failed: \(error)")
        }
        timings.tIO += RuntimeClock.ms(RuntimeClock.now() - t0)

        for layerIndex in 0..<kernels.count {
            let handles = surfaceHandles[layerIndex]

            t0 = RuntimeClock.now()
            do {
                try kernels[layerIndex].decodeQKVOnly.eval()
            } catch {
                throw .invalidArguments("decodeQKVOnly eval failed at layer \(layerIndex), token \(tokenIndex): \(error)")
            }
            timings.tAneQKV += RuntimeClock.ms(RuntimeClock.now() - t0)

            t0 = RuntimeClock.now()
            do {
                try SurfaceIO.copyFP16SpatialSlice(
                    dst: handles.kCacheFull,
                    dstChannelOffset: 0,
                    dstSpatialIndex: tokenIndex,
                    dstSpatial: maxSeq,
                    src: handles.kOut,
                    srcChannelOffset: 0,
                    srcSpatialIndex: 0,
                    srcSpatial: laneSpatial,
                    channels: dim
                )
                try SurfaceIO.copyFP16SpatialSlice(
                    dst: handles.vCacheFull,
                    dstChannelOffset: 0,
                    dstSpatialIndex: tokenIndex,
                    dstSpatial: maxSeq,
                    src: handles.vOut,
                    srcChannelOffset: 0,
                    srcSpatialIndex: 0,
                    srcSpatial: laneSpatial,
                    channels: dim
                )
            } catch {
                throw .invalidArguments("hybrid KV cache update failed: \(error)")
            }
            timings.tIO += RuntimeClock.ms(RuntimeClock.now() - t0)

            let metalShape: MetalDecodeAttentionShape
            do {
                metalShape = try MetalDecodeAttentionShape(
                    heads: ModelConfig.heads,
                    headDim: ModelConfig.headDim,
                    visibleTokens: visibleTokens,
                    cacheStride: maxSeq,
                    laneStride: laneSpatial
                )
            } catch {
                throw .invalidArguments("hybrid metal shape invalid: \(error)")
            }

            t0 = RuntimeClock.now()
            do {
                try metalAttention.runDecode(
                    qSurface: handles.qOut,
                    kCacheSurface: handles.kCacheFull,
                    vCacheSurface: handles.vCacheFull,
                    residualSurface: handles.qkvIn,
                    outputSurface: handles.ffnIn,
                    shape: metalShape,
                    projection: kernels[layerIndex].outputProjection
                )
            } catch {
                throw .invalidArguments("hybrid metal attention failed at layer \(layerIndex), token \(tokenIndex): \(error)")
            }
            timings.tMetal += RuntimeClock.ms(RuntimeClock.now() - t0)

            t0 = RuntimeClock.now()
            do {
                try kernels[layerIndex].decodeFFN.eval()
            } catch {
                throw .invalidArguments("hybrid decodeFFN eval failed at layer \(layerIndex), token \(tokenIndex): \(error)")
            }
            timings.tAneFFN += RuntimeClock.ms(RuntimeClock.now() - t0)

            if layerIndex + 1 < kernels.count {
                t0 = RuntimeClock.now()
                do {
                    try SurfaceIO.copyFP16SpatialSlice(
                        dst: surfaceHandles[layerIndex + 1].qkvIn,
                        dstChannelOffset: 0,
                        dstSpatialIndex: 0,
                        dstSpatial: laneSpatial,
                        src: handles.ffnOut,
                        srcChannelOffset: 0,
                        srcSpatialIndex: 0,
                        srcSpatial: laneSpatial,
                        channels: dim
                    )
                } catch {
                    throw .invalidArguments("hybrid ffn->next-qkv lane pack failed: \(error)")
                }
                timings.tIO += RuntimeClock.ms(RuntimeClock.now() - t0)
            }
        }

        let finalHandles = surfaceHandles[kernels.count - 1]
        t0 = RuntimeClock.now()
        do {
            try xCur.withUnsafeMutableBufferPointer { out in
                try SurfaceIO.readFP16SpatialSlice(
                    from: finalHandles.ffnOut,
                    channelOffset: 0,
                    spatialIndex: 0,
                    spatial: laneSpatial,
                    into: out,
                    channels: dim
                )
            }
        } catch {
            throw .invalidArguments("hybrid final decode lane unpack failed: \(error)")
        }
        timings.tIO += RuntimeClock.ms(RuntimeClock.now() - t0)

        try decodeState.commitTokenStep(expectedIndex: tokenIndex)
    }

    static func runDecodeTimed(
        xCur: borrowing TensorBuffer,
        kernels: borrowing LayerStorage<DecodeKernelSet>,
        surfaceHandles: [DecodeSurfaceHandles],
        decodeState: inout DecodeState,
        dim: Int = ModelConfig.dim,
        timings: inout StepTimingBreakdown,
        profiler: DecodeKernelProfiler? = nil
    ) throws(ANEError) {
        precondition(kernels.count > 0)
        precondition(surfaceHandles.count == kernels.count)
        precondition(dim > 0)
        precondition(xCur.count == dim)
        let maxSeq = decodeState.maxSeq
        for handles in surfaceHandles {
            precondition(handles.maxSeq == maxSeq)
        }

        let tokenIndex = try decodeState.beginTokenStep()
        let laneSpatial = surfaceHandles[0].laneSpatial
        let kernelMaxSeq = surfaceHandles[0].kernelMaxSeq
        let forceFullWindowSync = DecodeRuntimeOptions.forceFullWindowSync
        precondition(laneSpatial > 0)
        for handles in surfaceHandles {
            precondition(handles.laneSpatial == laneSpatial)
            precondition(handles.kernelMaxSeq == kernelMaxSeq)
        }
        let windowBase = DecodeTiling.windowBase(for: tokenIndex, laneSpatial: kernelMaxSeq)
        let windowLocalIndex = DecodeTiling.localIndex(for: tokenIndex, laneSpatial: kernelMaxSeq)

        // CPU touch at decode boundary: write the current token directly into lane 0.
        var t0 = RuntimeClock.now()
        do {
            try xCur.withUnsafeBufferPointer { xBuf in
                try SurfaceIO.writeFP16SpatialSlice(
                    to: surfaceHandles[0].attnIn,
                    channelOffset: 0,
                    spatialIndex: 0,
                    spatial: laneSpatial,
                    data: xBuf,
                    channels: dim
                )
            }
        } catch {
            throw .invalidArguments("decode token lane write failed: \(error)")
        }
        timings.tIO += RuntimeClock.ms(RuntimeClock.now() - t0)

        for L in 0..<kernels.count {
            let handles = surfaceHandles[L]
            let selfMaskUpdateUS: Double = 0

            if handles.maxSeq != handles.kernelMaxSeq {
                if forceFullWindowSync || DecodeTiling.shouldSyncWindow(for: tokenIndex, laneSpatial: kernelMaxSeq) {
                    t0 = RuntimeClock.now()
                    try synchronizeDecodeWindowCaches(handles: handles, windowBase: windowBase, dim: dim)
                    let windowSyncDelta = RuntimeClock.now() - t0
                    timings.tIO += RuntimeClock.ms(windowSyncDelta)
                }
            }

            if ProcessInfo.processInfo.environment["DECODE_EVAL_FFN_ONLY"] == "1" {
                t0 = RuntimeClock.now()
                do {
                    try SurfaceIO.copyFP16(
                        dst: handles.ffnIn,
                        dstChannelOffset: 0,
                        src: handles.zeroLane,
                        srcChannelOffset: 0,
                        channels: dim,
                        spatial: laneSpatial
                    )
                    try SurfaceIO.copyFP16SpatialSlice(
                        dst: handles.ffnIn,
                        dstChannelOffset: 0,
                        dstSpatialIndex: 0,
                        dstSpatial: laneSpatial,
                        src: handles.attnIn,
                        srcChannelOffset: 0,
                        srcSpatialIndex: 0,
                        srcSpatial: laneSpatial,
                        channels: dim
                    )
                    try kernels[L].decodeFFN.eval()
                } catch {
                    throw .invalidArguments("decodeFFN debug eval failed at layer \(L), token \(tokenIndex): \(error)")
                }
                let debugEvalDelta = RuntimeClock.now() - t0
                timings.tAne += RuntimeClock.ms(debugEvalDelta)
                continue
            }

            // Kernel A: decode attention (x2 + k_t + v_t as separate outputs).
            t0 = RuntimeClock.now()
            do {
                try kernels[L].decodeAttnQKV.eval()
            } catch {
                throw .invalidArguments("decodeAttnQKV eval failed at layer \(L), token \(tokenIndex): \(error)")
            }
            let attnEvalDelta = RuntimeClock.now() - t0
            timings.tAne += RuntimeClock.ms(attnEvalDelta)
            let attnEvalUS = RuntimeClock.us(attnEvalDelta)
            let attnHwNS = kernels[L].decodeAttnQKV.lastHWExecutionTimeNS()
            let attnHostOverheadUS = max(0, attnEvalUS - Double(attnHwNS) / 1_000.0)

            // Update K cache at index t from decode-attn K output at spatial lane 0.
            t0 = RuntimeClock.now()
            do {
                try SurfaceIO.copyFP16SpatialSlice(
                    dst: handles.kCacheFull,
                    dstChannelOffset: 0,
                    dstSpatialIndex: tokenIndex,
                    dstSpatial: maxSeq,
                    src: handles.attnKOut,
                    srcChannelOffset: 0,
                    srcSpatialIndex: 0,
                    srcSpatial: laneSpatial,
                    channels: dim
                )
                if handles.maxSeq != handles.kernelMaxSeq && !forceFullWindowSync {
                    try SurfaceIO.copyFP16SpatialSlice(
                        dst: handles.kCache,
                        dstChannelOffset: 0,
                        dstSpatialIndex: windowLocalIndex,
                        dstSpatial: kernelMaxSeq,
                        src: handles.attnKOut,
                        srcChannelOffset: 0,
                        srcSpatialIndex: 0,
                        srcSpatial: laneSpatial,
                        channels: dim
                    )
                }
            } catch {
                throw .invalidArguments("k-cache slice copy failed: \(error)")
            }
            let kUpdateDelta = RuntimeClock.now() - t0
            timings.tIO += RuntimeClock.ms(kUpdateDelta)
            let kUpdateUS = RuntimeClock.us(kUpdateDelta)

            // Update V cache at index t from decode-attn V output at spatial lane 0.
            t0 = RuntimeClock.now()
            do {
                try SurfaceIO.copyFP16SpatialSlice(
                    dst: handles.vCacheFull,
                    dstChannelOffset: 0,
                    dstSpatialIndex: tokenIndex,
                    dstSpatial: maxSeq,
                    src: handles.attnVOut,
                    srcChannelOffset: 0,
                    srcSpatialIndex: 0,
                    srcSpatial: laneSpatial,
                    channels: dim
                )
                if handles.maxSeq != handles.kernelMaxSeq && !forceFullWindowSync {
                    try SurfaceIO.copyFP16SpatialSlice(
                        dst: handles.vCache,
                        dstChannelOffset: 0,
                        dstSpatialIndex: windowLocalIndex,
                        dstSpatial: kernelMaxSeq,
                        src: handles.attnVOut,
                        srcChannelOffset: 0,
                        srcSpatialIndex: 0,
                        srcSpatial: laneSpatial,
                        channels: dim
                    )
                }
            } catch {
                throw .invalidArguments("v-cache slice copy failed: \(error)")
            }
            let vUpdateDelta = RuntimeClock.now() - t0
            timings.tIO += RuntimeClock.ms(vUpdateDelta)
            let vUpdateUS = RuntimeClock.us(vUpdateDelta)

            // Flip expanded mask[:, t] = 0 after eval so token becomes visible to next decode step.
            t0 = RuntimeClock.now()
            do {
                try SurfaceIO.copyFP16SpatialSlice(
                    dst: handles.maskCacheFull,
                    dstChannelOffset: 0,
                    dstSpatialIndex: tokenIndex,
                    dstSpatial: maxSeq,
                    src: handles.zeroLane,
                    srcChannelOffset: 0,
                    srcSpatialIndex: 0,
                    srcSpatial: laneSpatial,
                    channels: dim
                )
                if handles.maxSeq != handles.kernelMaxSeq && !forceFullWindowSync {
                    try SurfaceIO.copyFP16SpatialSlice(
                        dst: handles.maskCache,
                        dstChannelOffset: 0,
                        dstSpatialIndex: windowLocalIndex,
                        dstSpatial: kernelMaxSeq,
                        src: handles.zeroLane,
                        srcChannelOffset: 0,
                        srcSpatialIndex: 0,
                        srcSpatial: laneSpatial,
                        channels: dim
                    )
                }
            } catch {
                throw .invalidArguments("mask flip slice copy failed: \(error)")
            }
            let maskDelta = RuntimeClock.now() - t0
            timings.tIO += RuntimeClock.ms(maskDelta)
            let maskUpdateUS = RuntimeClock.us(maskDelta)

            // Feed x2_t from decode-attn X2 output into FFN input.
            t0 = RuntimeClock.now()
            do {
                try SurfaceIO.copyFP16SpatialSlice(
                    dst: handles.ffnIn,
                    dstChannelOffset: 0,
                    dstSpatialIndex: 0,
                    dstSpatial: laneSpatial,
                    src: handles.attnX2Out,
                    srcChannelOffset: 0,
                    srcSpatialIndex: 0,
                    srcSpatial: laneSpatial,
                    channels: dim
                )
            } catch {
                throw .invalidArguments("attn->ffn lane pack failed: \(error)")
            }
            let x2CopyDelta = RuntimeClock.now() - t0
            timings.tIO += RuntimeClock.ms(x2CopyDelta)
            let x2ToFfnCopyUS = RuntimeClock.us(x2CopyDelta)

            // Kernel B: decode FFN.
            t0 = RuntimeClock.now()
            do {
                try kernels[L].decodeFFN.eval()
            } catch {
                throw .invalidArguments("decodeFFN eval failed at layer \(L), token \(tokenIndex): \(error)")
            }
            let ffnEvalDelta = RuntimeClock.now() - t0
            timings.tAne += RuntimeClock.ms(ffnEvalDelta)
            let ffnEvalUS = RuntimeClock.us(ffnEvalDelta)
            let ffnHwNS = kernels[L].decodeFFN.lastHWExecutionTimeNS()
            let ffnHostOverheadUS = max(0, ffnEvalUS - Double(ffnHwNS) / 1_000.0)

            // Chain FFN output directly to next layer's lane-packed attn input.
            var ffnToNextAttnCopyUS: Double = 0
            if L + 1 < kernels.count {
                t0 = RuntimeClock.now()
                do {
                    try SurfaceIO.copyFP16SpatialSlice(
                        dst: surfaceHandles[L + 1].attnIn,
                        dstChannelOffset: 0,
                        dstSpatialIndex: 0,
                        dstSpatial: laneSpatial,
                        src: handles.ffnOut,
                        srcChannelOffset: 0,
                        srcSpatialIndex: 0,
                        srcSpatial: laneSpatial,
                        channels: dim
                    )
                } catch {
                    throw .invalidArguments("ffn->next-attn lane pack failed: \(error)")
                }
                let nextCopyDelta = RuntimeClock.now() - t0
                timings.tIO += RuntimeClock.ms(nextCopyDelta)
                ffnToNextAttnCopyUS = RuntimeClock.us(nextCopyDelta)
            }

            profiler?.record(
                layerIndex: L,
                attnEvalUS: attnEvalUS,
                attnHwNS: attnHwNS,
                attnHostOverheadUS: attnHostOverheadUS,
                selfMaskUpdateUS: selfMaskUpdateUS,
                kCacheUpdateUS: kUpdateUS,
                vCacheUpdateUS: vUpdateUS,
                maskUpdateUS: maskUpdateUS,
                x2ToFfnCopyUS: x2ToFfnCopyUS,
                ffnEvalUS: ffnEvalUS,
                ffnHwNS: ffnHwNS,
                ffnHostOverheadUS: ffnHostOverheadUS,
                ffnToNextAttnCopyUS: ffnToNextAttnCopyUS
            )
        }

        // CPU touch at decode boundary: read final lane 0 output directly.
        let finalHandles = surfaceHandles[kernels.count - 1]
        let finalOut = finalHandles.ffnOut
        t0 = RuntimeClock.now()
        do {
            try xCur.withUnsafeMutableBufferPointer { out in
                try SurfaceIO.readFP16SpatialSlice(
                    from: finalOut,
                    channelOffset: 0,
                    spatialIndex: 0,
                    spatial: laneSpatial,
                    into: out,
                    channels: dim
                )
            }
        } catch {
            throw .invalidArguments("final decode lane unpack failed: \(error)")
        }
        timings.tIO += RuntimeClock.ms(RuntimeClock.now() - t0)

        try decodeState.commitTokenStep(expectedIndex: tokenIndex)
    }

    // MARK: - Fused decode path

    private static func synchronizeFusedDecodeWindowCaches(
        handles: FusedDecodeSurfaceHandles,
        windowBase: Int,
        dim: Int
    ) throws(ANEError) {
        let windowSpatial = handles.kernelMaxSeq
        for windowIndex in 0..<windowSpatial {
            let globalIndex = windowBase + windowIndex
            if globalIndex < handles.maxSeq {
                do {
                    try SurfaceIO.copyFP16SpatialSlice(
                        dst: handles.kCache, dstChannelOffset: 0,
                        dstSpatialIndex: windowIndex, dstSpatial: windowSpatial,
                        src: handles.kCacheFull, srcChannelOffset: 0,
                        srcSpatialIndex: globalIndex, srcSpatial: handles.maxSeq,
                        channels: dim
                    )
                    try SurfaceIO.copyFP16SpatialSlice(
                        dst: handles.vCache, dstChannelOffset: 0,
                        dstSpatialIndex: windowIndex, dstSpatial: windowSpatial,
                        src: handles.vCacheFull, srcChannelOffset: 0,
                        srcSpatialIndex: globalIndex, srcSpatial: handles.maxSeq,
                        channels: dim
                    )
                    try SurfaceIO.copyFP16SpatialSlice(
                        dst: handles.maskCache, dstChannelOffset: 0,
                        dstSpatialIndex: windowIndex, dstSpatial: windowSpatial,
                        src: handles.maskCacheFull, srcChannelOffset: 0,
                        srcSpatialIndex: globalIndex, srcSpatial: handles.maxSeq,
                        channels: dim
                    )
                } catch {
                    throw .invalidArguments("fused decode cache window sync failed at globalIndex=\(globalIndex): \(error)")
                }
            } else {
                do {
                    try SurfaceIO.copyFP16SpatialSlice(
                        dst: handles.kCache, dstChannelOffset: 0,
                        dstSpatialIndex: windowIndex, dstSpatial: windowSpatial,
                        src: handles.zeroLane, srcChannelOffset: 0,
                        srcSpatialIndex: 0, srcSpatial: handles.laneSpatial,
                        channels: dim
                    )
                    try SurfaceIO.copyFP16SpatialSlice(
                        dst: handles.vCache, dstChannelOffset: 0,
                        dstSpatialIndex: windowIndex, dstSpatial: windowSpatial,
                        src: handles.zeroLane, srcChannelOffset: 0,
                        srcSpatialIndex: 0, srcSpatial: handles.laneSpatial,
                        channels: dim
                    )
                    try SurfaceIO.copyFP16SpatialSlice(
                        dst: handles.maskCache, dstChannelOffset: 0,
                        dstSpatialIndex: windowIndex, dstSpatial: windowSpatial,
                        src: handles.maskedLane, srcChannelOffset: 0,
                        srcSpatialIndex: 0, srcSpatial: handles.laneSpatial,
                        channels: dim
                    )
                } catch {
                    throw .invalidArguments("fused decode padded-window fill failed at windowIndex=\(windowIndex): \(error)")
                }
            }
        }
    }

    static func initializeFusedDecodeCachesAndMask(
        surfaceHandles: [FusedDecodeSurfaceHandles],
        dim: Int = ModelConfig.dim
    ) {
        precondition(dim > 0)
        guard let first = surfaceHandles.first else { return }
        let maxSeq = first.maxSeq
        precondition(maxSeq > 0)

        let zeroCache = Array(repeating: Float(0), count: dim * maxSeq)
        let maskFill: Float = ProcessInfo.processInfo.environment["ESPRESSO_DECODE_MASK_INIT_ZERO"] == "1" ? 0 : -1e4
        let masked = Array(repeating: maskFill, count: dim * maxSeq)

        for handles in surfaceHandles {
            precondition(handles.maxSeq == maxSeq)
            zeroCache.withUnsafeBufferPointer { src in
                SurfaceIO.writeFP16(to: handles.kCacheFull, data: src, channels: dim, spatial: maxSeq)
                SurfaceIO.writeFP16(to: handles.vCacheFull, data: src, channels: dim, spatial: maxSeq)
            }
            masked.withUnsafeBufferPointer { src in
                SurfaceIO.writeFP16(to: handles.maskCacheFull, data: src, channels: dim, spatial: maxSeq)
            }
            if handles.maxSeq != handles.kernelMaxSeq {
                do {
                    try synchronizeFusedDecodeWindowCaches(handles: handles, windowBase: 0, dim: dim)
                } catch {
                    preconditionFailure("fused decode window init failed: \(error)")
                }
            }
            do {
                try SurfaceIO.copyFP16(
                    dst: handles.fusedIn, dstChannelOffset: 0,
                    src: handles.zeroLane, srcChannelOffset: 0,
                    channels: dim, spatial: handles.laneSpatial
                )
            } catch {
                preconditionFailure("fused decode lane zero-init failed: \(error)")
            }
        }
    }

    /// Fused decode loop: one dispatch per layer instead of two.
    ///
    /// Per layer: eval(fused) → K/V/mask cache update → chain xNext to next layer.
    /// Eliminates the attn→FFN surface copy and one dispatch overhead per layer.
    static func runFusedDecodeTimed(
        xCur: borrowing TensorBuffer,
        kernels: borrowing LayerStorage<FusedDecodeKernelSet>,
        surfaceHandles: [FusedDecodeSurfaceHandles],
        decodeState: inout DecodeState,
        dim: Int = ModelConfig.dim,
        timings: inout StepTimingBreakdown,
        profiler: DecodeKernelProfiler? = nil
    ) throws(ANEError) {
        precondition(kernels.count > 0)
        precondition(surfaceHandles.count == kernels.count)
        precondition(dim > 0)
        precondition(xCur.count == dim)
        let maxSeq = decodeState.maxSeq
        for handles in surfaceHandles {
            precondition(handles.maxSeq == maxSeq)
        }

        let tokenIndex = try decodeState.beginTokenStep()
        let laneSpatial = surfaceHandles[0].laneSpatial
        let kernelMaxSeq = surfaceHandles[0].kernelMaxSeq
        let forceFullWindowSync = DecodeRuntimeOptions.forceFullWindowSync
        precondition(laneSpatial > 0)
        for handles in surfaceHandles {
            precondition(handles.laneSpatial == laneSpatial)
            precondition(handles.kernelMaxSeq == kernelMaxSeq)
        }
        let windowBase = DecodeTiling.windowBase(for: tokenIndex, laneSpatial: kernelMaxSeq)
        let windowLocalIndex = DecodeTiling.localIndex(for: tokenIndex, laneSpatial: kernelMaxSeq)

        // CPU touch at decode boundary: write current token into fused input lane 0.
        var t0 = RuntimeClock.now()
        do {
            try xCur.withUnsafeBufferPointer { xBuf in
                try SurfaceIO.writeFP16SpatialSlice(
                    to: surfaceHandles[0].fusedIn,
                    channelOffset: 0,
                    spatialIndex: 0,
                    spatial: laneSpatial,
                    data: xBuf,
                    channels: dim
                )
            }
        } catch {
            throw .invalidArguments("fused decode token lane write failed: \(error)")
        }
        timings.tIO += RuntimeClock.ms(RuntimeClock.now() - t0)

        for L in 0..<kernels.count {
            let handles = surfaceHandles[L]

            if handles.maxSeq != handles.kernelMaxSeq {
                if forceFullWindowSync || DecodeTiling.shouldSyncWindow(for: tokenIndex, laneSpatial: kernelMaxSeq) {
                    t0 = RuntimeClock.now()
                    try synchronizeFusedDecodeWindowCaches(handles: handles, windowBase: windowBase, dim: dim)
                    let windowSyncDelta = RuntimeClock.now() - t0
                    timings.tIO += RuntimeClock.ms(windowSyncDelta)
                }
            }

            // Single fused eval: attention + FFN in one dispatch.
            t0 = RuntimeClock.now()
            do {
                try kernels[L].fusedLayer.eval()
            } catch {
                throw .invalidArguments("fusedDecodeLayer eval failed at layer \(L), token \(tokenIndex): \(error)")
            }
            let fusedEvalDelta = RuntimeClock.now() - t0
            timings.tAne += RuntimeClock.ms(fusedEvalDelta)
            let fusedEvalUS = RuntimeClock.us(fusedEvalDelta)
            let fusedHwNS = kernels[L].fusedLayer.lastHWExecutionTimeNS()
            let fusedHostOverheadUS = max(0, fusedEvalUS - Double(fusedHwNS) / 1_000.0)

            // Update K cache from fused kernel K output.
            t0 = RuntimeClock.now()
            do {
                try SurfaceIO.copyFP16SpatialSlice(
                    dst: handles.kCacheFull, dstChannelOffset: 0,
                    dstSpatialIndex: tokenIndex, dstSpatial: maxSeq,
                    src: handles.kOut, srcChannelOffset: 0,
                    srcSpatialIndex: 0, srcSpatial: laneSpatial,
                    channels: dim
                )
                if handles.maxSeq != handles.kernelMaxSeq && !forceFullWindowSync {
                    try SurfaceIO.copyFP16SpatialSlice(
                        dst: handles.kCache, dstChannelOffset: 0,
                        dstSpatialIndex: windowLocalIndex, dstSpatial: kernelMaxSeq,
                        src: handles.kOut, srcChannelOffset: 0,
                        srcSpatialIndex: 0, srcSpatial: laneSpatial,
                        channels: dim
                    )
                }
            } catch {
                throw .invalidArguments("fused k-cache slice copy failed: \(error)")
            }
            let kUpdateDelta = RuntimeClock.now() - t0
            timings.tIO += RuntimeClock.ms(kUpdateDelta)
            let kUpdateUS = RuntimeClock.us(kUpdateDelta)

            // Update V cache from fused kernel V output.
            t0 = RuntimeClock.now()
            do {
                try SurfaceIO.copyFP16SpatialSlice(
                    dst: handles.vCacheFull, dstChannelOffset: 0,
                    dstSpatialIndex: tokenIndex, dstSpatial: maxSeq,
                    src: handles.vOut, srcChannelOffset: 0,
                    srcSpatialIndex: 0, srcSpatial: laneSpatial,
                    channels: dim
                )
                if handles.maxSeq != handles.kernelMaxSeq && !forceFullWindowSync {
                    try SurfaceIO.copyFP16SpatialSlice(
                        dst: handles.vCache, dstChannelOffset: 0,
                        dstSpatialIndex: windowLocalIndex, dstSpatial: kernelMaxSeq,
                        src: handles.vOut, srcChannelOffset: 0,
                        srcSpatialIndex: 0, srcSpatial: laneSpatial,
                        channels: dim
                    )
                }
            } catch {
                throw .invalidArguments("fused v-cache slice copy failed: \(error)")
            }
            let vUpdateDelta = RuntimeClock.now() - t0
            timings.tIO += RuntimeClock.ms(vUpdateDelta)
            let vUpdateUS = RuntimeClock.us(vUpdateDelta)

            // Flip mask at token index.
            t0 = RuntimeClock.now()
            do {
                try SurfaceIO.copyFP16SpatialSlice(
                    dst: handles.maskCacheFull, dstChannelOffset: 0,
                    dstSpatialIndex: tokenIndex, dstSpatial: maxSeq,
                    src: handles.zeroLane, srcChannelOffset: 0,
                    srcSpatialIndex: 0, srcSpatial: laneSpatial,
                    channels: dim
                )
                if handles.maxSeq != handles.kernelMaxSeq && !forceFullWindowSync {
                    try SurfaceIO.copyFP16SpatialSlice(
                        dst: handles.maskCache, dstChannelOffset: 0,
                        dstSpatialIndex: windowLocalIndex, dstSpatial: kernelMaxSeq,
                        src: handles.zeroLane, srcChannelOffset: 0,
                        srcSpatialIndex: 0, srcSpatial: laneSpatial,
                        channels: dim
                    )
                }
            } catch {
                throw .invalidArguments("fused mask flip slice copy failed: \(error)")
            }
            let maskDelta = RuntimeClock.now() - t0
            timings.tIO += RuntimeClock.ms(maskDelta)
            let maskUpdateUS = RuntimeClock.us(maskDelta)

            // Chain xNext output to next layer's fused input.
            var chainToNextUS: Double = 0
            if L + 1 < kernels.count {
                t0 = RuntimeClock.now()
                do {
                    try SurfaceIO.copyFP16SpatialSlice(
                        dst: surfaceHandles[L + 1].fusedIn, dstChannelOffset: 0,
                        dstSpatialIndex: 0, dstSpatial: laneSpatial,
                        src: handles.xNextOut, srcChannelOffset: 0,
                        srcSpatialIndex: 0, srcSpatial: laneSpatial,
                        channels: dim
                    )
                } catch {
                    throw .invalidArguments("fused->next-layer chain failed: \(error)")
                }
                let nextCopyDelta = RuntimeClock.now() - t0
                timings.tIO += RuntimeClock.ms(nextCopyDelta)
                chainToNextUS = RuntimeClock.us(nextCopyDelta)
            }

            // Record profiling using the existing profiler (fused eval goes into attn slot,
            // FFN fields are zero since there's no separate FFN dispatch).
            profiler?.record(
                layerIndex: L,
                attnEvalUS: fusedEvalUS,
                attnHwNS: fusedHwNS,
                attnHostOverheadUS: fusedHostOverheadUS,
                selfMaskUpdateUS: 0,
                kCacheUpdateUS: kUpdateUS,
                vCacheUpdateUS: vUpdateUS,
                maskUpdateUS: maskUpdateUS,
                x2ToFfnCopyUS: 0,
                ffnEvalUS: 0,
                ffnHwNS: 0,
                ffnHostOverheadUS: 0,
                ffnToNextAttnCopyUS: chainToNextUS
            )
        }

        // CPU touch at decode boundary: read final xNext output.
        let finalHandles = surfaceHandles[kernels.count - 1]
        t0 = RuntimeClock.now()
        do {
            try xCur.withUnsafeMutableBufferPointer { out in
                try SurfaceIO.readFP16SpatialSlice(
                    from: finalHandles.xNextOut,
                    channelOffset: 0,
                    spatialIndex: 0,
                    spatial: laneSpatial,
                    into: out,
                    channels: dim
                )
            }
        } catch {
            throw .invalidArguments("fused final decode lane unpack failed: \(error)")
        }
        timings.tIO += RuntimeClock.ms(RuntimeClock.now() - t0)

        try decodeState.commitTokenStep(expectedIndex: tokenIndex)
    }
}

public extension ForwardPass {
    private static func synchronizeFusedTwoLayerDecodeWindowCaches(
        handles: FusedTwoLayerDecodeSurfaceHandles,
        windowBase: Int,
        dim: Int
    ) throws(ANEError) {
        let packedKVChannels = FusedTwoLayerCacheLayout.packedKVChannels
        let windowSpatial = handles.kernelMaxSeq
        for windowIndex in 0..<windowSpatial {
            let globalIndex = windowBase + windowIndex
            if globalIndex < handles.maxSeq {
                do {
                    try SurfaceIO.copyFP16SpatialSlice(
                        dst: handles.packedKVCache,
                        dstChannelOffset: 0,
                        dstSpatialIndex: windowIndex,
                        dstSpatial: windowSpatial,
                        src: handles.packedKVCacheFull,
                        srcChannelOffset: 0,
                        srcSpatialIndex: globalIndex,
                        srcSpatial: handles.maxSeq,
                        channels: packedKVChannels
                    )
                    try SurfaceIO.copyFP16SpatialSlice(
                        dst: handles.maskCache0,
                        dstChannelOffset: 0,
                        dstSpatialIndex: windowIndex,
                        dstSpatial: windowSpatial,
                        src: handles.maskCache0Full,
                        srcChannelOffset: 0,
                        srcSpatialIndex: globalIndex,
                        srcSpatial: handles.maxSeq,
                        channels: dim
                    )
                    try SurfaceIO.copyFP16SpatialSlice(
                        dst: handles.maskCache1,
                        dstChannelOffset: 0,
                        dstSpatialIndex: windowIndex,
                        dstSpatial: windowSpatial,
                        src: handles.maskCache1Full,
                        srcChannelOffset: 0,
                        srcSpatialIndex: globalIndex,
                        srcSpatial: handles.maxSeq,
                        channels: dim
                    )
                } catch {
                    throw .invalidArguments("fused two-layer cache window sync failed at globalIndex=\(globalIndex): \(error)")
                }
            } else {
                do {
                    try SurfaceIO.copyFP16SpatialSlice(
                        dst: handles.packedKVCache,
                        dstChannelOffset: 0,
                        dstSpatialIndex: windowIndex,
                        dstSpatial: windowSpatial,
                        src: handles.zeroPackedKVLane,
                        srcChannelOffset: 0,
                        srcSpatialIndex: 0,
                        srcSpatial: handles.laneSpatial,
                        channels: packedKVChannels
                    )
                    try SurfaceIO.copyFP16SpatialSlice(
                        dst: handles.maskCache0,
                        dstChannelOffset: 0,
                        dstSpatialIndex: windowIndex,
                        dstSpatial: windowSpatial,
                        src: handles.maskedLane,
                        srcChannelOffset: 0,
                        srcSpatialIndex: 0,
                        srcSpatial: handles.laneSpatial,
                        channels: dim
                    )
                    try SurfaceIO.copyFP16SpatialSlice(
                        dst: handles.maskCache1,
                        dstChannelOffset: 0,
                        dstSpatialIndex: windowIndex,
                        dstSpatial: windowSpatial,
                        src: handles.maskedLane,
                        srcChannelOffset: 0,
                        srcSpatialIndex: 0,
                        srcSpatial: handles.laneSpatial,
                        channels: dim
                    )
                } catch {
                    throw .invalidArguments("fused two-layer padded-window fill failed at windowIndex=\(windowIndex): \(error)")
                }
            }
        }
    }

    static func initializeFusedTwoLayerDecodeCachesAndMask(
        surfaceHandles: [FusedTwoLayerDecodeSurfaceHandles],
        dim: Int = ModelConfig.dim
    ) {
        precondition(dim > 0)
        guard let first = surfaceHandles.first else { return }
        let maxSeq = first.maxSeq
        let maskFill: Float = ProcessInfo.processInfo.environment["ESPRESSO_DECODE_MASK_INIT_ZERO"] == "1" ? 0 : -1e4
        let packedKVChannels = FusedTwoLayerCacheLayout.packedKVChannels
        let packedKVCache = Array(repeating: Float(0), count: packedKVChannels * maxSeq)
        let masked = Array(repeating: maskFill, count: dim * maxSeq)

        for handles in surfaceHandles {
            precondition(handles.maxSeq == maxSeq)
            packedKVCache.withUnsafeBufferPointer { src in
                SurfaceIO.writeFP16(
                    to: handles.packedKVCacheFull,
                    data: src,
                    channels: packedKVChannels,
                    spatial: maxSeq
                )
            }
            masked.withUnsafeBufferPointer { src in
                SurfaceIO.writeFP16(to: handles.maskCache0Full, data: src, channels: dim, spatial: maxSeq)
                SurfaceIO.writeFP16(to: handles.maskCache1Full, data: src, channels: dim, spatial: maxSeq)
            }
            if handles.maxSeq != handles.kernelMaxSeq {
                do {
                    try synchronizeFusedTwoLayerDecodeWindowCaches(
                        handles: handles,
                        windowBase: 0,
                        dim: dim
                    )
                } catch {
                    preconditionFailure("fused two-layer window init failed: \(error)")
                }
            }
            do {
                try SurfaceIO.copyFP16(
                    dst: handles.fusedIn,
                    dstChannelOffset: 0,
                    src: handles.zeroLane,
                    srcChannelOffset: 0,
                    channels: dim,
                    spatial: handles.laneSpatial
                )
            } catch {
                preconditionFailure("fused two-layer lane zero-init failed: \(error)")
            }
        }
    }

    static func runFusedTwoLayerDecodeTimed(
        xCur: borrowing TensorBuffer,
        kernels: borrowing LayerStorage<FusedTwoLayerDecodeKernelSet>,
        surfaceHandles: [FusedTwoLayerDecodeSurfaceHandles],
        decodeState: inout DecodeState,
        dim: Int = ModelConfig.dim,
        timings: inout StepTimingBreakdown,
        profiler: DecodeKernelProfiler? = nil
    ) throws(ANEError) {
        precondition(kernels.count > 0)
        precondition(surfaceHandles.count == kernels.count)
        precondition(dim > 0)
        precondition(xCur.count == dim)
        let maxSeq = decodeState.maxSeq
        for handles in surfaceHandles {
            precondition(handles.maxSeq == maxSeq)
        }

        let tokenIndex = try decodeState.beginTokenStep()
        let laneSpatial = surfaceHandles[0].laneSpatial
        let kernelMaxSeq = surfaceHandles[0].kernelMaxSeq
        let forceFullWindowSync = DecodeRuntimeOptions.forceFullWindowSync
        precondition(laneSpatial > 0)
        for handles in surfaceHandles {
            precondition(handles.laneSpatial == laneSpatial)
            precondition(handles.kernelMaxSeq == kernelMaxSeq)
        }
        let windowBase = DecodeTiling.windowBase(for: tokenIndex, laneSpatial: kernelMaxSeq)
        let windowLocalIndex = DecodeTiling.localIndex(for: tokenIndex, laneSpatial: kernelMaxSeq)

        var t0 = RuntimeClock.now()
        do {
            try xCur.withUnsafeBufferPointer { xBuf in
                try SurfaceIO.writeFP16SpatialSlice(
                    to: surfaceHandles[0].fusedIn,
                    channelOffset: 0,
                    spatialIndex: 0,
                    spatial: laneSpatial,
                    data: xBuf,
                    channels: dim
                )
            }
        } catch {
            throw .invalidArguments("fused two-layer token lane write failed: \(error)")
        }
        timings.tIO += RuntimeClock.ms(RuntimeClock.now() - t0)

        for pairIndex in 0..<kernels.count {
            let handles = surfaceHandles[pairIndex]

            if handles.maxSeq != handles.kernelMaxSeq {
                if forceFullWindowSync || DecodeTiling.shouldSyncWindow(for: tokenIndex, laneSpatial: kernelMaxSeq) {
                    t0 = RuntimeClock.now()
                    try synchronizeFusedTwoLayerDecodeWindowCaches(
                        handles: handles,
                        windowBase: windowBase,
                        dim: dim
                    )
                    let windowSyncDelta = RuntimeClock.now() - t0
                    timings.tIO += RuntimeClock.ms(windowSyncDelta)
                }
            }

            t0 = RuntimeClock.now()
            do {
                try kernels[pairIndex].fusedPair.eval()
            } catch {
                throw .invalidArguments("fusedTwoLayerDecode eval failed at pair \(pairIndex), token \(tokenIndex): \(error)")
            }
            let fusedEvalDelta = RuntimeClock.now() - t0
            timings.tAne += RuntimeClock.ms(fusedEvalDelta)
            let fusedEvalUS = RuntimeClock.us(fusedEvalDelta)
            let fusedHwNS = kernels[pairIndex].fusedPair.lastHWExecutionTimeNS()
            let fusedHostOverheadUS = max(0, fusedEvalUS - Double(fusedHwNS) / 1_000.0)

            let writebackSpecs = [
                (dstOffset: FusedTwoLayerCacheLayout.layer0K, srcSurface: handles.kPackedOut, srcOffset: 0),
                (dstOffset: FusedTwoLayerCacheLayout.layer0V, srcSurface: handles.vPackedOut, srcOffset: 0),
                (dstOffset: FusedTwoLayerCacheLayout.layer1K, srcSurface: handles.kPackedOut, srcOffset: dim),
                (dstOffset: FusedTwoLayerCacheLayout.layer1V, srcSurface: handles.vPackedOut, srcOffset: dim),
            ]
            var packedKVUpdateUS: Double = 0
            for spec in writebackSpecs {
                t0 = RuntimeClock.now()
                do {
                    try SurfaceIO.copyFP16SpatialSlice(
                        dst: handles.packedKVCacheFull,
                        dstChannelOffset: spec.dstOffset,
                        dstSpatialIndex: tokenIndex,
                        dstSpatial: maxSeq,
                        src: spec.srcSurface,
                        srcChannelOffset: spec.srcOffset,
                        srcSpatialIndex: 0,
                        srcSpatial: laneSpatial,
                        channels: dim
                    )
                    if handles.maxSeq != handles.kernelMaxSeq && !forceFullWindowSync {
                        try SurfaceIO.copyFP16SpatialSlice(
                            dst: handles.packedKVCache,
                            dstChannelOffset: spec.dstOffset,
                            dstSpatialIndex: windowLocalIndex,
                            dstSpatial: kernelMaxSeq,
                            src: spec.srcSurface,
                            srcChannelOffset: spec.srcOffset,
                            srcSpatialIndex: 0,
                            srcSpatial: laneSpatial,
                            channels: dim
                        )
                    }
                } catch {
                    throw .invalidArguments("fused two-layer packed K/V writeback failed: \(error)")
                }
                let updateDelta = RuntimeClock.now() - t0
                timings.tIO += RuntimeClock.ms(updateDelta)
                packedKVUpdateUS += RuntimeClock.us(updateDelta)
            }
            var maskUpdateUS: Double = 0
            let maskTargets = [
                (window: handles.maskCache0, full: handles.maskCache0Full),
                (window: handles.maskCache1, full: handles.maskCache1Full),
            ]
            for maskTarget in maskTargets {
                t0 = RuntimeClock.now()
                do {
                    try SurfaceIO.copyFP16SpatialSlice(
                        dst: maskTarget.full,
                        dstChannelOffset: 0,
                        dstSpatialIndex: tokenIndex,
                        dstSpatial: maxSeq,
                        src: handles.zeroLane,
                        srcChannelOffset: 0,
                        srcSpatialIndex: 0,
                        srcSpatial: laneSpatial,
                        channels: dim
                    )
                    if handles.maxSeq != handles.kernelMaxSeq && !forceFullWindowSync {
                        try SurfaceIO.copyFP16SpatialSlice(
                            dst: maskTarget.window,
                            dstChannelOffset: 0,
                            dstSpatialIndex: windowLocalIndex,
                            dstSpatial: kernelMaxSeq,
                            src: handles.zeroLane,
                            srcChannelOffset: 0,
                            srcSpatialIndex: 0,
                            srcSpatial: laneSpatial,
                            channels: dim
                        )
                    }
                } catch {
                    throw .invalidArguments("fused two-layer mask flip failed: \(error)")
                }
                let maskDelta = RuntimeClock.now() - t0
                timings.tIO += RuntimeClock.ms(maskDelta)
                maskUpdateUS += RuntimeClock.us(maskDelta)
            }

            var chainToNextUS: Double = 0
            if pairIndex + 1 < kernels.count {
                t0 = RuntimeClock.now()
                do {
                    try SurfaceIO.copyFP16SpatialSlice(
                        dst: surfaceHandles[pairIndex + 1].fusedIn,
                        dstChannelOffset: 0,
                        dstSpatialIndex: 0,
                        dstSpatial: laneSpatial,
                        src: handles.xNextOut,
                        srcChannelOffset: 0,
                        srcSpatialIndex: 0,
                        srcSpatial: laneSpatial,
                        channels: dim
                    )
                } catch {
                    throw .invalidArguments("fused two-layer -> next pair chain failed: \(error)")
                }
                let nextCopyDelta = RuntimeClock.now() - t0
                timings.tIO += RuntimeClock.ms(nextCopyDelta)
                chainToNextUS = RuntimeClock.us(nextCopyDelta)
            }

            profiler?.record(
                layerIndex: pairIndex,
                attnEvalUS: fusedEvalUS,
                attnHwNS: fusedHwNS,
                attnHostOverheadUS: fusedHostOverheadUS,
                selfMaskUpdateUS: 0,
                kCacheUpdateUS: packedKVUpdateUS,
                vCacheUpdateUS: 0,
                maskUpdateUS: maskUpdateUS,
                x2ToFfnCopyUS: 0,
                ffnEvalUS: 0,
                ffnHwNS: 0,
                ffnHostOverheadUS: 0,
                ffnToNextAttnCopyUS: chainToNextUS
            )
        }

        let finalHandles = surfaceHandles[kernels.count - 1]
        t0 = RuntimeClock.now()
        do {
            try xCur.withUnsafeMutableBufferPointer { out in
                try SurfaceIO.readFP16SpatialSlice(
                    from: finalHandles.xNextOut,
                    channelOffset: 0,
                    spatialIndex: 0,
                    spatial: laneSpatial,
                    into: out,
                    channels: dim
                )
            }
        } catch {
            throw .invalidArguments("fused two-layer final decode lane unpack failed: \(error)")
        }
        timings.tIO += RuntimeClock.ms(RuntimeClock.now() - t0)

        try decodeState.commitTokenStep(expectedIndex: tokenIndex)
    }
}
