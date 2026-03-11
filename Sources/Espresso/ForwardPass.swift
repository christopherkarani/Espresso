import Accelerate
import IOSurface
import ANERuntime
import ANETypes

/// Inference-only surface handles for a single layer (2 kernels, not 5).
public struct InferenceSurfaceHandles {
    public let fwdAttnIn: IOSurfaceRef
    public let fwdAttnOut: IOSurfaceRef
    public let fwdFFNIn: IOSurfaceRef
    public let fwdFFNOut: IOSurfaceRef

    public init(kernels: borrowing InferenceKernelSet) throws(ANEError) {
        self.fwdAttnIn = try kernels.fwdAttn.inputSurface(at: 0)
        self.fwdAttnOut = try kernels.fwdAttn.outputSurface(at: 0)
        self.fwdFFNIn = try kernels.fwdFFN.inputSurface(at: 0)
        self.fwdFFNOut = try kernels.fwdFFN.outputSurface(at: 0)
    }
}

/// Transformer forward pass using ANE kernels plus CPU residual additions.
/// Maps to `train_large.m:384-420`.
public enum ForwardPass {
    @inline(__always)
    private static func requireBase(_ buffer: UnsafeMutableBufferPointer<Float>) -> UnsafeMutablePointer<Float> {
        guard let base = buffer.baseAddress else {
            preconditionFailure("Expected non-empty buffer")
        }
        return base
    }

    /// Runs a forward pass for `kernels.count` layers.
    ///
    /// - Parameters:
    ///   - xCur: In/out hidden state buffer, channel-first `[dim, seqLen]`. Updated in place to the final layer output.
    ///   - acts: Per-layer activation storage. Only a subset is populated; Q/K/V are intentionally not read back to CPU.
    ///   - kernels: Per-layer ANE kernels.
    ///   - accumulator: Async dW accumulator, used here as a barrier before writing fwdAttn inputs.
    public static func run(
        xCur: borrowing TensorBuffer,
        acts: borrowing LayerStorage<LayerActivations>,
        kernels: borrowing LayerStorage<LayerKernelSet>,
        accumulator: GradientAccumulator,
        dim: Int = ModelConfig.dim,
        hidden: Int = ModelConfig.hidden,
        seqLen: Int = ModelConfig.seqLen,
        surfaceHandles: [LayerSurfaceHandles]? = nil
    ) throws(ANEError) {
        var ignoredTimings = StepTimingBreakdown()
        try runTimed(
            xCur: xCur,
            acts: acts,
            kernels: kernels,
            accumulator: accumulator,
            dim: dim,
            hidden: hidden,
            seqLen: seqLen,
            surfaceHandles: surfaceHandles,
            timings: &ignoredTimings
        )
    }

    public static func runTimed(
        xCur: borrowing TensorBuffer,
        acts: borrowing LayerStorage<LayerActivations>,
        kernels: borrowing LayerStorage<LayerKernelSet>,
        accumulator: GradientAccumulator,
        dim: Int = ModelConfig.dim,
        hidden: Int = ModelConfig.hidden,
        seqLen: Int = ModelConfig.seqLen,
        surfaceHandles: [LayerSurfaceHandles]? = nil,
        timings: inout StepTimingBreakdown
    ) throws(ANEError) {
        precondition(dim > 0 && hidden > 0 && seqLen > 0)
        precondition(xCur.count == dim * seqLen)
        precondition(acts.count == kernels.count)
        if let handles = surfaceHandles {
            precondition(handles.count == kernels.count)
        }
        _ = accumulator

        let dimSeq = dim * seqLen
        let dimSeqBytes = dimSeq * MemoryLayout<Float>.stride

        for L in 0..<kernels.count {
            let layerHandles = surfaceHandles?[L]

            // Save input for RMSNorm backward.
            xCur.withUnsafePointer { xPtr in
                acts[L].layerIn.withUnsafeMutablePointer { dst in
                    _ = memcpy(dst, xPtr, dimSeqBytes)
                }
            }

            // Attention forward.
            let attnIn: IOSurfaceRef
            if let handles = layerHandles {
                attnIn = handles.fwdAttnIn
            } else {
                attnIn = try kernels[L].fwdAttn.inputSurface(at: 0)
            }
            var t0 = RuntimeClock.now()
            xCur.withUnsafeBufferPointer { xBuf in
                SurfaceIO.writeFP16(to: attnIn, data: xBuf, channels: dim, spatial: seqLen)
            }
            timings.tIO += RuntimeClock.ms(RuntimeClock.now() - t0)

            t0 = RuntimeClock.now()
            try kernels[L].fwdAttn.eval()
            timings.tAne += RuntimeClock.ms(RuntimeClock.now() - t0)

            let attnOut: IOSurfaceRef
            if let handles = layerHandles {
                attnOut = handles.fwdAttnOut
            } else {
                attnOut = try kernels[L].fwdAttn.outputSurface(at: 0)
            }
            t0 = RuntimeClock.now()
            acts[L].oOut.withUnsafeMutableBufferPointer { oOut in
                acts[L].attnOut.withUnsafeMutableBufferPointer { attnOutBuf in
                    acts[L].xnorm.withUnsafeMutableBufferPointer { xnormBuf in
                        let regions = [
                            SurfaceIO.FP16ReadRegion(destination: requireBase(oOut), channelOffset: 0, channels: dim),
                            SurfaceIO.FP16ReadRegion(destination: requireBase(attnOutBuf), channelOffset: 4 * dim, channels: dim),
                            SurfaceIO.FP16ReadRegion(destination: requireBase(xnormBuf), channelOffset: 5 * dim, channels: dim),
                        ]
                        SurfaceIO.readFP16Batched(from: attnOut, spatial: seqLen, regions: regions)
                    }
                }
            }
            timings.tIO += RuntimeClock.ms(RuntimeClock.now() - t0)
            // NOTE: Q/K/V at offsets 1*dim, 2*dim, 3*dim are intentionally not read back to CPU.

            // Residual: x2 = xCur + oOut.
            t0 = RuntimeClock.now()
            xCur.withUnsafePointer { xPtr in
                acts[L].oOut.withUnsafePointer { oPtr in
                    acts[L].x2.withUnsafeMutablePointer { x2Ptr in
                        vDSP_vadd(xPtr, 1, oPtr, 1, x2Ptr, 1, vDSP_Length(dimSeq))
                    }
                }
            }
            timings.tElem += RuntimeClock.ms(RuntimeClock.now() - t0)

            // FFN forward.
            let ffnIn: IOSurfaceRef
            if let handles = layerHandles {
                ffnIn = handles.fwdFFNIn
            } else {
                ffnIn = try kernels[L].fwdFFN.inputSurface(at: 0)
            }
            t0 = RuntimeClock.now()
            acts[L].x2.withUnsafeBufferPointer { x2Buf in
                SurfaceIO.writeFP16(to: ffnIn, data: x2Buf, channels: dim, spatial: seqLen)
            }
            timings.tIO += RuntimeClock.ms(RuntimeClock.now() - t0)

            t0 = RuntimeClock.now()
            try kernels[L].fwdFFN.eval()
            timings.tAne += RuntimeClock.ms(RuntimeClock.now() - t0)

            let ffnOut: IOSurfaceRef
            if let handles = layerHandles {
                ffnOut = handles.fwdFFNOut
            } else {
                ffnOut = try kernels[L].fwdFFN.outputSurface(at: 0)
            }
            t0 = RuntimeClock.now()
            acts[L].ffnOut.withUnsafeMutableBufferPointer { ffnOutBuf in
                acts[L].h1.withUnsafeMutableBufferPointer { h1Buf in
                    acts[L].h3.withUnsafeMutableBufferPointer { h3Buf in
                        acts[L].siluOut.withUnsafeMutableBufferPointer { siluBuf in
                            acts[L].x2norm.withUnsafeMutableBufferPointer { x2normBuf in
                                let regions = [
                                    SurfaceIO.FP16ReadRegion(destination: requireBase(ffnOutBuf), channelOffset: 0, channels: dim),
                                    SurfaceIO.FP16ReadRegion(destination: requireBase(h1Buf), channelOffset: dim, channels: hidden),
                                    SurfaceIO.FP16ReadRegion(destination: requireBase(h3Buf), channelOffset: dim + hidden, channels: hidden),
                                    SurfaceIO.FP16ReadRegion(destination: requireBase(siluBuf), channelOffset: dim + 2 * hidden, channels: hidden),
                                    SurfaceIO.FP16ReadRegion(destination: requireBase(x2normBuf), channelOffset: dim + 3 * hidden, channels: dim),
                                ]
                                SurfaceIO.readFP16Batched(from: ffnOut, spatial: seqLen, regions: regions)
                            }
                        }
                    }
                }
            }
            timings.tIO += RuntimeClock.ms(RuntimeClock.now() - t0)

            // Residual: xCur = x2 + ffnOut.
            t0 = RuntimeClock.now()
            acts[L].x2.withUnsafePointer { x2Ptr in
                acts[L].ffnOut.withUnsafePointer { ffnPtr in
                    xCur.withUnsafeMutablePointer { xCurPtr in
                        vDSP_vadd(x2Ptr, 1, ffnPtr, 1, xCurPtr, 1, vDSP_Length(dimSeq))
                    }
                }
            }
            timings.tElem += RuntimeClock.ms(RuntimeClock.now() - t0)
        }
    }

    // MARK: - Inference Path (fused residuals, no backward activations)

    public enum InferenceInterKernelHandoff: Sendable {
        /// Baseline: read attn output to CPU (fp32), then write CPU buffer back to FFN input (fp16).
        case cpuRoundTrip
        /// Optimization: copy attn output surface -> FFN input surface directly in fp16.
        /// Avoids intermediate fp16<->fp32 conversions and removes one surface read + one surface write.
        case fp16SurfaceCopy
    }

    /// Inference-only forward pass using `InferenceKernelSet`.
    ///
    /// The inference kernels fuse residual additions inside the MIL program and output
    /// only `dim` channels per kernel (instead of `6*dim` / `2*dim + 3*hidden`).
    /// No `LayerActivations` are needed — backward data is not stored.
    public static func runInference(
        xCur: borrowing TensorBuffer,
        kernels: borrowing LayerStorage<InferenceKernelSet>,
        dim: Int = ModelConfig.dim,
        seqLen: Int = ModelConfig.seqLen,
        surfaceHandles: [InferenceSurfaceHandles]? = nil
    ) throws(ANEError) {
        var ignoredTimings = StepTimingBreakdown()
        try runInferenceTimed(
            xCur: xCur,
            kernels: kernels,
            dim: dim,
            seqLen: seqLen,
            surfaceHandles: surfaceHandles,
            timings: &ignoredTimings
        )
    }

    /// Timed inference-only forward pass. Accumulates timing into `timings`.
    ///
    /// Per-layer flow:
    /// 1. Write xCur → attn input surface
    /// 2. Eval attn kernel (outputs x2 = x + attn(x) via fused residual)
    /// 3. Read x2 from attn output surface → xCur
    /// 4. Write xCur → FFN input surface
    /// 5. Eval FFN kernel (outputs xCur = x2 + ffn(x2) via fused residual)
    /// 6. Read xCur from FFN output surface → xCur
    ///
    /// No CPU residual additions needed. No backward activation reads.
    public static func runInferenceTimed(
        xCur: borrowing TensorBuffer,
        kernels: borrowing LayerStorage<InferenceKernelSet>,
        dim: Int = ModelConfig.dim,
        seqLen: Int = ModelConfig.seqLen,
        surfaceHandles: [InferenceSurfaceHandles]? = nil,
        handoff: InferenceInterKernelHandoff = .cpuRoundTrip,
        profiler: InferenceKernelProfiler? = nil,
        timings: inout StepTimingBreakdown
    ) throws(ANEError) {
        precondition(dim > 0 && seqLen > 0)
        precondition(xCur.count == dim * seqLen)
        if let handles = surfaceHandles {
            precondition(handles.count == kernels.count)
        }

        for L in 0..<kernels.count {
            let layerHandles = surfaceHandles?[L]

            // --- Attention forward (fused residual) ---
            let attnIn: IOSurfaceRef
            if let handles = layerHandles {
                attnIn = handles.fwdAttnIn
            } else {
                attnIn = try kernels[L].fwdAttn.inputSurface(at: 0)
            }

            var attnWriteLockUS: Double = 0
            var attnWriteBodyUS: Double = 0
            var attnWriteUnlockUS: Double = 0

            let attnWriteStart = RuntimeClock.now()
            if profiler == nil {
                xCur.withUnsafeBufferPointer { xBuf in
                    SurfaceIO.writeFP16(to: attnIn, data: xBuf, channels: dim, spatial: seqLen)
                }
            } else {
                let lockStart = RuntimeClock.now()
                precondition(SurfaceIO.lockWrite(attnIn))
                attnWriteLockUS = RuntimeClock.us(RuntimeClock.now() - lockStart)

                let bodyStart = RuntimeClock.now()
                xCur.withUnsafeBufferPointer { xBuf in
                    SurfaceIO.writeFP16Unlocked(to: attnIn, data: xBuf, channels: dim, spatial: seqLen)
                }
                attnWriteBodyUS = RuntimeClock.us(RuntimeClock.now() - bodyStart)

                let unlockStart = RuntimeClock.now()
                precondition(SurfaceIO.unlockWrite(attnIn))
                attnWriteUnlockUS = RuntimeClock.us(RuntimeClock.now() - unlockStart)
            }
            let attnWriteDelta = RuntimeClock.now() - attnWriteStart
            timings.tIO += RuntimeClock.ms(attnWriteDelta)
            let attnWriteUS = RuntimeClock.us(attnWriteDelta)

            var t0 = RuntimeClock.now()
            try kernels[L].fwdAttn.eval()
            let attnEvalDelta = RuntimeClock.now() - t0
            timings.tAne += RuntimeClock.ms(attnEvalDelta)
            let attnEvalUS = RuntimeClock.us(attnEvalDelta)
            let attnHwNS = kernels[L].fwdAttn.lastHWExecutionTimeNS()
            let attnHostOverheadUS = max(0, attnEvalUS - Double(attnHwNS) / 1_000.0)
            let attnEvalEndTick = RuntimeClock.now()

            // Read only dim channels (the fused residual result).
            let attnOut: IOSurfaceRef
            if let handles = layerHandles {
                attnOut = handles.fwdAttnOut
            } else {
                attnOut = try kernels[L].fwdAttn.outputSurface(at: 0)
            }

            // --- FFN forward (fused residual) ---
            let ffnIn: IOSurfaceRef
            if let handles = layerHandles {
                ffnIn = handles.fwdFFNIn
            } else {
                ffnIn = try kernels[L].fwdFFN.inputSurface(at: 0)
            }

            var attnReadUS: Double = 0
            var attnReadLockUS: Double = 0
            var attnReadBodyUS: Double = 0
            var attnReadUnlockUS: Double = 0
            var ffnWriteUS: Double = 0
            var ffnWriteLockUS: Double = 0
            var ffnWriteBodyUS: Double = 0
            var ffnWriteUnlockUS: Double = 0
            var ffnCopyUS: Double = 0

            switch handoff {
            case .cpuRoundTrip:
                let attnReadStart = RuntimeClock.now()
                if profiler == nil {
                    xCur.withUnsafeMutableBufferPointer { xBuf in
                        SurfaceIO.readFP16(
                            from: attnOut,
                            into: xBuf,
                            channelOffset: 0,
                            channels: dim,
                            spatial: seqLen
                        )
                    }
                } else {
                    let lockStart = RuntimeClock.now()
                    precondition(SurfaceIO.lockRead(attnOut))
                    attnReadLockUS = RuntimeClock.us(RuntimeClock.now() - lockStart)

                    let bodyStart = RuntimeClock.now()
                    xCur.withUnsafeMutableBufferPointer { xBuf in
                        SurfaceIO.readFP16Unlocked(
                            from: attnOut,
                            into: xBuf,
                            channelOffset: 0,
                            channels: dim,
                            spatial: seqLen
                        )
                    }
                    attnReadBodyUS = RuntimeClock.us(RuntimeClock.now() - bodyStart)

                    let unlockStart = RuntimeClock.now()
                    precondition(SurfaceIO.unlockRead(attnOut))
                    attnReadUnlockUS = RuntimeClock.us(RuntimeClock.now() - unlockStart)
                }
                let attnReadDelta = RuntimeClock.now() - attnReadStart
                timings.tIO += RuntimeClock.ms(attnReadDelta)
                attnReadUS = RuntimeClock.us(attnReadDelta)

                let ffnWriteStart = RuntimeClock.now()
                if profiler == nil {
                    xCur.withUnsafeBufferPointer { xBuf in
                        SurfaceIO.writeFP16(to: ffnIn, data: xBuf, channels: dim, spatial: seqLen)
                    }
                } else {
                    let lockStart = RuntimeClock.now()
                    precondition(SurfaceIO.lockWrite(ffnIn))
                    ffnWriteLockUS = RuntimeClock.us(RuntimeClock.now() - lockStart)

                    let bodyStart = RuntimeClock.now()
                    xCur.withUnsafeBufferPointer { xBuf in
                        SurfaceIO.writeFP16Unlocked(to: ffnIn, data: xBuf, channels: dim, spatial: seqLen)
                    }
                    ffnWriteBodyUS = RuntimeClock.us(RuntimeClock.now() - bodyStart)

                    let unlockStart = RuntimeClock.now()
                    precondition(SurfaceIO.unlockWrite(ffnIn))
                    ffnWriteUnlockUS = RuntimeClock.us(RuntimeClock.now() - unlockStart)
                }
                let ffnWriteDelta = RuntimeClock.now() - ffnWriteStart
                timings.tIO += RuntimeClock.ms(ffnWriteDelta)
                ffnWriteUS = RuntimeClock.us(ffnWriteDelta)
            case .fp16SurfaceCopy:
                t0 = RuntimeClock.now()
                do {
                    try SurfaceIO.copyFP16(
                        dst: ffnIn,
                        dstChannelOffset: 0,
                        src: attnOut,
                        srcChannelOffset: 0,
                        channels: dim,
                        spatial: seqLen
                    )
                } catch {
                    throw .invalidArguments("SurfaceIO.copyFP16 failed: \(error)")
                }
                let ffnCopyDelta = RuntimeClock.now() - t0
                timings.tIO += RuntimeClock.ms(ffnCopyDelta)
                ffnCopyUS = RuntimeClock.us(ffnCopyDelta)
            }

            let ffnEvalStartTick = RuntimeClock.now()
            let gapUS = RuntimeClock.us(ffnEvalStartTick - attnEvalEndTick)

            t0 = ffnEvalStartTick
            try kernels[L].fwdFFN.eval()
            let ffnEvalDelta = RuntimeClock.now() - t0
            timings.tAne += RuntimeClock.ms(ffnEvalDelta)
            let ffnEvalUS = RuntimeClock.us(ffnEvalDelta)
            let ffnHwNS = kernels[L].fwdFFN.lastHWExecutionTimeNS()
            let ffnHostOverheadUS = max(0, ffnEvalUS - Double(ffnHwNS) / 1_000.0)

            // Read only dim channels (the fused residual result).
            let ffnOut: IOSurfaceRef
            if let handles = layerHandles {
                ffnOut = handles.fwdFFNOut
            } else {
                ffnOut = try kernels[L].fwdFFN.outputSurface(at: 0)
            }

            var ffnReadLockUS: Double = 0
            var ffnReadBodyUS: Double = 0
            var ffnReadUnlockUS: Double = 0

            let ffnReadStart = RuntimeClock.now()
            if profiler == nil {
                xCur.withUnsafeMutableBufferPointer { xBuf in
                    SurfaceIO.readFP16(
                        from: ffnOut,
                        into: xBuf,
                        channelOffset: 0,
                        channels: dim,
                        spatial: seqLen
                    )
                }
            } else {
                let lockStart = RuntimeClock.now()
                precondition(SurfaceIO.lockRead(ffnOut))
                ffnReadLockUS = RuntimeClock.us(RuntimeClock.now() - lockStart)

                let bodyStart = RuntimeClock.now()
                xCur.withUnsafeMutableBufferPointer { xBuf in
                    SurfaceIO.readFP16Unlocked(
                        from: ffnOut,
                        into: xBuf,
                        channelOffset: 0,
                        channels: dim,
                        spatial: seqLen
                    )
                }
                ffnReadBodyUS = RuntimeClock.us(RuntimeClock.now() - bodyStart)

                let unlockStart = RuntimeClock.now()
                precondition(SurfaceIO.unlockRead(ffnOut))
                ffnReadUnlockUS = RuntimeClock.us(RuntimeClock.now() - unlockStart)
            }
            let ffnReadDelta = RuntimeClock.now() - ffnReadStart
            timings.tIO += RuntimeClock.ms(ffnReadDelta)
            let ffnReadUS = RuntimeClock.us(ffnReadDelta)

            profiler?.record(
                layerIndex: L,
                attnWriteUS: attnWriteUS,
                attnWriteLockUS: attnWriteLockUS,
                attnWriteBodyUS: attnWriteBodyUS,
                attnWriteUnlockUS: attnWriteUnlockUS,
                attnEvalUS: attnEvalUS,
                attnHwNS: attnHwNS,
                attnHostOverheadUS: attnHostOverheadUS,
                attnReadUS: attnReadUS,
                attnReadLockUS: attnReadLockUS,
                attnReadBodyUS: attnReadBodyUS,
                attnReadUnlockUS: attnReadUnlockUS,
                ffnWriteUS: ffnWriteUS,
                ffnWriteLockUS: ffnWriteLockUS,
                ffnWriteBodyUS: ffnWriteBodyUS,
                ffnWriteUnlockUS: ffnWriteUnlockUS,
                ffnCopyUS: ffnCopyUS,
                ffnEvalUS: ffnEvalUS,
                ffnHwNS: ffnHwNS,
                ffnHostOverheadUS: ffnHostOverheadUS,
                ffnReadUS: ffnReadUS,
                ffnReadLockUS: ffnReadLockUS,
                ffnReadBodyUS: ffnReadBodyUS,
                ffnReadUnlockUS: ffnReadUnlockUS,
                gapAttnToFfnUS: gapUS
            )
        }
    }
}
