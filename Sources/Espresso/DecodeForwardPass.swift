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
    public let attnX2Out: IOSurfaceRef
    public let attnKOut: IOSurfaceRef
    public let attnVOut: IOSurfaceRef
    public let ffnIn: IOSurfaceRef
    public let ffnOut: IOSurfaceRef
    public let zeroLane: IOSurfaceRef
    public let tokenScratch: IOSurfaceRef
    public let maxSeq: Int
    public let laneSpatial: Int

    public init(kernels: borrowing DecodeKernelSet) throws(ANEError) {
        self.attnIn = try kernels.decodeAttnQKV.inputSurface(at: 0)
        self.kCache = try kernels.decodeAttnQKV.inputSurface(at: 1)
        self.vCache = try kernels.decodeAttnQKV.inputSurface(at: 2)
        self.maskCache = try kernels.decodeAttnQKV.inputSurface(at: 3)
        self.attnX2Out = try kernels.decodeAttnQKV.outputSurface(at: 0)
        self.attnKOut = try kernels.decodeAttnQKV.outputSurface(at: 1)
        self.attnVOut = try kernels.decodeAttnQKV.outputSurface(at: 2)
        self.ffnIn = try kernels.decodeFFN.inputSurface(at: 0)
        self.ffnOut = try kernels.decodeFFN.outputSurface(at: 0)
        self.maxSeq = kernels.maxSeq
        self.laneSpatial = kernels.laneSpatial

        guard let zeroLane = ane_interop_create_surface(ModelConfig.dim * kernels.laneSpatial * 2),
              let tokenScratch = ane_interop_create_surface(ModelConfig.dim * 2) else {
            throw .surfaceAllocationFailed
        }
        self.zeroLane = zeroLane
        self.tokenScratch = tokenScratch

        let zeroLaneValues = Array(repeating: Float(0), count: ModelConfig.dim * kernels.laneSpatial)
        zeroLaneValues.withUnsafeBufferPointer { src in
            SurfaceIO.writeFP16(to: zeroLane, data: src, channels: ModelConfig.dim, spatial: kernels.laneSpatial)
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

public extension ForwardPass {
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
                SurfaceIO.writeFP16(to: handles.kCache, data: src, channels: dim, spatial: maxSeq)
                SurfaceIO.writeFP16(to: handles.vCache, data: src, channels: dim, spatial: maxSeq)
            }
            masked.withUnsafeBufferPointer { src in
                SurfaceIO.writeFP16(to: handles.maskCache, data: src, channels: dim, spatial: maxSeq)
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
        precondition(laneSpatial > 0)
        for handles in surfaceHandles {
            precondition(handles.laneSpatial == laneSpatial)
        }

        // CPU touch at decode boundary: write token embedding/state to a compact token scratch surface.
        var t0 = RuntimeClock.now()
        xCur.withUnsafeBufferPointer { xBuf in
            SurfaceIO.writeFP16(to: surfaceHandles[0].tokenScratch, data: xBuf, channels: dim, spatial: 1)
        }
        do {
            try SurfaceIO.copyFP16(
                dst: surfaceHandles[0].attnIn,
                dstChannelOffset: 0,
                src: surfaceHandles[0].zeroLane,
                srcChannelOffset: 0,
                channels: dim,
                spatial: laneSpatial
            )
            try SurfaceIO.copyFP16SpatialSlice(
                dst: surfaceHandles[0].attnIn,
                dstChannelOffset: 0,
                dstSpatialIndex: 0,
                dstSpatial: laneSpatial,
                src: surfaceHandles[0].tokenScratch,
                srcChannelOffset: 0,
                srcSpatialIndex: 0,
                srcSpatial: 1,
                channels: dim
            )
        } catch {
            throw .invalidArguments("decode token lane write failed: \(error)")
        }
        timings.tIO += RuntimeClock.ms(RuntimeClock.now() - t0)

        for L in 0..<kernels.count {
            let handles = surfaceHandles[L]
            let selfMaskUpdateUS: Double = 0

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
                    dst: handles.kCache,
                    dstChannelOffset: 0,
                    dstSpatialIndex: tokenIndex,
                    dstSpatial: maxSeq,
                    src: handles.attnKOut,
                    srcChannelOffset: 0,
                    srcSpatialIndex: 0,
                    srcSpatial: laneSpatial,
                    channels: dim
                )
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
                    dst: handles.vCache,
                    dstChannelOffset: 0,
                    dstSpatialIndex: tokenIndex,
                    dstSpatial: maxSeq,
                    src: handles.attnVOut,
                    srcChannelOffset: 0,
                    srcSpatialIndex: 0,
                    srcSpatial: laneSpatial,
                    channels: dim
                )
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
                    dst: handles.maskCache,
                    dstChannelOffset: 0,
                    dstSpatialIndex: tokenIndex,
                    dstSpatial: maxSeq,
                    src: handles.zeroLane,
                    srcChannelOffset: 0,
                    srcSpatialIndex: 0,
                    srcSpatial: laneSpatial,
                    channels: dim
                )
            } catch {
                throw .invalidArguments("mask flip slice copy failed: \(error)")
            }
            let maskDelta = RuntimeClock.now() - t0
            timings.tIO += RuntimeClock.ms(maskDelta)
            let maskUpdateUS = RuntimeClock.us(maskDelta)

            // Feed x2_t from decode-attn X2 output into FFN input.
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
                    try SurfaceIO.copyFP16(
                        dst: surfaceHandles[L + 1].attnIn,
                        dstChannelOffset: 0,
                        src: surfaceHandles[L + 1].zeroLane,
                        srcChannelOffset: 0,
                        channels: dim,
                        spatial: laneSpatial
                    )
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

        // CPU touch at decode boundary: read final layer output.
        let finalHandles = surfaceHandles[kernels.count - 1]
        let finalOut = finalHandles.ffnOut
        t0 = RuntimeClock.now()
        do {
            try SurfaceIO.copyFP16SpatialSlice(
                dst: finalHandles.tokenScratch,
                dstChannelOffset: 0,
                dstSpatialIndex: 0,
                dstSpatial: 1,
                src: finalOut,
                srcChannelOffset: 0,
                srcSpatialIndex: 0,
                srcSpatial: laneSpatial,
                channels: dim
            )
        } catch {
            throw .invalidArguments("final decode lane unpack failed: \(error)")
        }
        xCur.withUnsafeMutableBufferPointer { out in
            SurfaceIO.readFP16(from: finalHandles.tokenScratch, into: out, channelOffset: 0, channels: dim, spatial: 1)
        }
        timings.tIO += RuntimeClock.ms(RuntimeClock.now() - t0)

        try decodeState.commitTokenStep(expectedIndex: tokenIndex)
    }
}
