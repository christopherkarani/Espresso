import Accelerate
import IOSurface
import ANERuntime
import ANETypes
import CPUOps
import Darwin

/// Scratch buffers for `BackwardPass`.
///
/// Allocate once and reuse across steps; buffers are overwritten each backward call.
public struct BackwardScratch: ~Copyable {
    public let dffn: TensorBuffer
    public let dxFfn: TensorBuffer
    public let dh1: TensorBuffer
    public let dh3: TensorBuffer

    public let dx2: TensorBuffer
    public let dq: TensorBuffer
    public let dk: TensorBuffer
    public let dv: TensorBuffer
    public let dxAttn: TensorBuffer
    public let dxRms1: TensorBuffer

    public init(dim: Int = ModelConfig.dim, hidden: Int = ModelConfig.hidden, seqLen: Int = ModelConfig.seqLen) {
        precondition(dim > 0 && hidden > 0 && seqLen > 0)
        let dimSeq = dim * seqLen
        let hidSeq = hidden * seqLen

        self.dffn = TensorBuffer(count: dimSeq, zeroed: false)
        self.dxFfn = TensorBuffer(count: dimSeq, zeroed: false)
        self.dh1 = TensorBuffer(count: hidSeq, zeroed: false)
        self.dh3 = TensorBuffer(count: hidSeq, zeroed: false)

        self.dx2 = TensorBuffer(count: dimSeq, zeroed: false)
        self.dq = TensorBuffer(count: dimSeq, zeroed: false)
        self.dk = TensorBuffer(count: dimSeq, zeroed: false)
        self.dv = TensorBuffer(count: dimSeq, zeroed: false)
        self.dxAttn = TensorBuffer(count: dimSeq, zeroed: false)
        self.dxRms1 = TensorBuffer(count: dimSeq, zeroed: false)
    }
}

/// Transformer backward pass using ANE backward kernels + CPU RMSNorm + async cblas dW.
/// Maps to `train_large.m:461-575`.
public enum BackwardPass {
    @inline(__always)
    private static func requireBase(_ buffer: UnsafeMutableBufferPointer<Float>) -> UnsafeMutablePointer<Float> {
        guard let base = buffer.baseAddress else {
            preconditionFailure("Expected non-empty buffer")
        }
        return base
    }

    @inline(__always)
    private static func mapSurfaceError(_ error: SurfaceIOError) -> ANEError {
        switch error {
        case .argumentOutOfRange:
            return .invalidArguments("SurfaceIO argument out of range")
        case .interopCallFailed:
            return .evaluationFailed
        }
    }

    @inline(__always)
    private static func mapSurfaceIO<R>(_ body: () throws -> R) throws(ANEError) -> R {
        do {
            return try body()
        } catch let e as SurfaceIOError {
            throw mapSurfaceError(e)
        } catch {
            throw .evaluationFailed
        }
    }

    public static func run(
        dy: borrowing TensorBuffer,
        acts: borrowing LayerStorage<LayerActivations>,
        kernels: borrowing LayerStorage<LayerKernelSet>,
        staticKernels: borrowing LayerStorage<StaticKernel>,
        grads: borrowing LayerStorage<LayerGradients>,
        weights: borrowing LayerStorage<LayerWeights>,
        scratch: borrowing BackwardScratch,
        accumulator: GradientAccumulator,
        dim: Int = ModelConfig.dim,
        hidden: Int = ModelConfig.hidden,
        seqLen: Int = ModelConfig.seqLen,
        heads: Int = ModelConfig.heads,
        surfaceHandles: [LayerSurfaceHandles]? = nil
    ) throws(ANEError) {
        var ignoredTimings = StepTimingBreakdown()
        try runTimed(
            dy: dy,
            acts: acts,
            kernels: kernels,
            staticKernels: staticKernels,
            grads: grads,
            weights: weights,
            scratch: scratch,
            accumulator: accumulator,
            dim: dim,
            hidden: hidden,
            seqLen: seqLen,
            heads: heads,
            surfaceHandles: surfaceHandles,
            timings: &ignoredTimings
        )
    }

    public static func runTimed(
        dy: borrowing TensorBuffer,
        acts: borrowing LayerStorage<LayerActivations>,
        kernels: borrowing LayerStorage<LayerKernelSet>,
        staticKernels: borrowing LayerStorage<StaticKernel>,
        grads: borrowing LayerStorage<LayerGradients>,
        weights: borrowing LayerStorage<LayerWeights>,
        scratch: borrowing BackwardScratch,
        accumulator: GradientAccumulator,
        dim: Int = ModelConfig.dim,
        hidden: Int = ModelConfig.hidden,
        seqLen: Int = ModelConfig.seqLen,
        heads: Int = ModelConfig.heads,
        surfaceHandles: [LayerSurfaceHandles]? = nil,
        timings: inout StepTimingBreakdown
    ) throws(ANEError) {
        precondition(dim > 0 && hidden > 0 && seqLen > 0 && heads > 0)
        precondition(dy.count == dim * seqLen)
        precondition(acts.count == kernels.count)
        precondition(staticKernels.count == kernels.count)
        precondition(grads.count == kernels.count)
        precondition(weights.count == kernels.count)
        if let handles = surfaceHandles {
            precondition(handles.count == kernels.count)
        }

        let nLayers = kernels.count
        let dimSeq = dim * seqLen
        let dimSeqBytes = dimSeq * MemoryLayout<Float>.stride
        let scoreCh = heads * seqLen

        for L in stride(from: nLayers - 1, through: 0, by: -1) {
            let layerHandles = surfaceHandles?[L]

            // dffn = dy (residual copy).
            dy.withUnsafePointer { dyPtr in
                scratch.dffn.withUnsafeMutablePointer { dst in
                    _ = memcpy(dst, dyPtr, dimSeqBytes)
                }
            }

            // === STEP 1: FFN backward (ANE) ===
            let ffnBwdIn: IOSurfaceRef
            if let handles = layerHandles {
                ffnBwdIn = handles.ffnBwdIn
            } else {
                ffnBwdIn = try kernels[L].ffnBwd.inputSurface(at: 0)
            }
            var ffnWriteError: SurfaceIOError?
            var t0 = RuntimeClock.now()
            scratch.dffn.withUnsafeBufferPointer { dffnBuf in
                do {
                    try SurfaceIO.writeFP16At(
                        to: ffnBwdIn,
                        channelOffset: 0,
                        data: dffnBuf,
                        channels: dim,
                        spatial: seqLen
                    )
                } catch let e as SurfaceIOError {
                    ffnWriteError = e
                } catch {
                    ffnWriteError = .interopCallFailed
                }
            }
            if let e = ffnWriteError {
                throw mapSurfaceError(e)
            }
            timings.tIO += RuntimeClock.ms(RuntimeClock.now() - t0)

            let fwdFFNOut: IOSurfaceRef
            if let handles = layerHandles {
                fwdFFNOut = handles.fwdFFNOut
            } else {
                fwdFFNOut = try kernels[L].fwdFFN.outputSurface(at: 0)
            }
            t0 = RuntimeClock.now()
            try mapSurfaceIO {
                try SurfaceIO.copyFP16(
                    dst: ffnBwdIn,
                    dstChannelOffset: dim,
                    src: fwdFFNOut,
                    srcChannelOffset: dim,
                    channels: 2 * hidden,
                    spatial: seqLen
                )
            }
            timings.tIO += RuntimeClock.ms(RuntimeClock.now() - t0)

            t0 = RuntimeClock.now()
            try kernels[L].ffnBwd.eval()
            timings.tAne += RuntimeClock.ms(RuntimeClock.now() - t0)

            let ffnBwdOut: IOSurfaceRef
            if let handles = layerHandles {
                ffnBwdOut = handles.ffnBwdOut
            } else {
                ffnBwdOut = try kernels[L].ffnBwd.outputSurface(at: 0)
            }
            t0 = RuntimeClock.now()
            scratch.dxFfn.withUnsafeMutableBufferPointer { dxFfnBuf in
                scratch.dh1.withUnsafeMutableBufferPointer { dh1Buf in
                    scratch.dh3.withUnsafeMutableBufferPointer { dh3Buf in
                        let regions = [
                            SurfaceIO.FP16ReadRegion(destination: requireBase(dxFfnBuf), channelOffset: 0, channels: dim),
                            SurfaceIO.FP16ReadRegion(destination: requireBase(dh1Buf), channelOffset: dim, channels: hidden),
                            SurfaceIO.FP16ReadRegion(destination: requireBase(dh3Buf), channelOffset: dim + hidden, channels: hidden),
                        ]
                        SurfaceIO.readFP16Batched(from: ffnBwdOut, spatial: seqLen, regions: regions)
                    }
                }
            }
            timings.tIO += RuntimeClock.ms(RuntimeClock.now() - t0)

            // === STEP 1b: Async dW FFN (CPU) ===
            let grW2 = grads[L].W2.withUnsafeMutablePointer { SendablePointer($0) }
            let grW1 = grads[L].W1.withUnsafeMutablePointer { SendablePointer($0) }
            let grW3 = grads[L].W3.withUnsafeMutablePointer { SendablePointer($0) }

            let captDffn = SendableBuffer(copying: scratch.dffn)
            let captSilu = SendableBuffer(copying: acts[L].siluOut)
            let captDh1 = SendableBuffer(copying: scratch.dh1)
            let captDh3 = SendableBuffer(copying: scratch.dh3)
            let captX2n = SendableBuffer(copying: acts[L].x2norm)

            accumulator.enqueue { [captDffn = consume captDffn, captSilu = consume captSilu, captDh1 = consume captDh1, captDh3 = consume captDh3, captX2n = consume captX2n] in
                BLAS.sgemm(
                    CblasRowMajor, CblasNoTrans, CblasTrans,
                    m: Int32(dim), n: Int32(hidden), k: Int32(seqLen),
                    alpha: 1.0,
                    a: captDffn.pointer, lda: Int32(seqLen),
                    b: captSilu.pointer, ldb: Int32(seqLen),
                    beta: 1.0,
                    c: grW2.pointer, ldc: Int32(hidden)
                )
                BLAS.sgemm(
                    CblasRowMajor, CblasNoTrans, CblasTrans,
                    m: Int32(hidden), n: Int32(dim), k: Int32(seqLen),
                    alpha: 1.0,
                    a: captDh1.pointer, lda: Int32(seqLen),
                    b: captX2n.pointer, ldb: Int32(seqLen),
                    beta: 1.0,
                    c: grW1.pointer, ldc: Int32(dim)
                )
                BLAS.sgemm(
                    CblasRowMajor, CblasNoTrans, CblasTrans,
                    m: Int32(hidden), n: Int32(dim), k: Int32(seqLen),
                    alpha: 1.0,
                    a: captDh3.pointer, lda: Int32(seqLen),
                    b: captX2n.pointer, ldb: Int32(seqLen),
                    beta: 1.0,
                    c: grW3.pointer, ldc: Int32(dim)
                )
            }

            // === STEP 2: RMSNorm2 backward (CPU) ===
            scratch.dx2.withUnsafeMutablePointer { dx2Ptr in
                grads[L].rmsFfn.withUnsafeMutablePointer { dwPtr in
                    scratch.dxFfn.withUnsafePointer { dxFfnPtr in
                        acts[L].x2.withUnsafePointer { x2Ptr in
                            weights[L].rmsFfn.withUnsafePointer { wPtr in
                                RMSNorm.backward(
                                    dx: dx2Ptr,
                                    dw: dwPtr,
                                    dy: dxFfnPtr,
                                    x: x2Ptr,
                                    weights: wPtr,
                                    dim: dim,
                                    seqLen: seqLen
                                )
                            }
                        }
                    }
                }
            }

            // residual: dx2 += dy
            scratch.dx2.withUnsafeMutablePointer { dx2Ptr in
                dy.withUnsafePointer { dyPtr in
                    vDSP_vadd(dx2Ptr, 1, dyPtr, 1, dx2Ptr, 1, vDSP_Length(dimSeq))
                }
            }

            // === STEP 3: Async dWo (CPU) ===
            let grWo = grads[L].Wo.withUnsafeMutablePointer { SendablePointer($0) }
            let captDo = SendableBuffer(copying: scratch.dx2)
            let captAttn = SendableBuffer(copying: acts[L].attnOut)
            accumulator.enqueue { [captDo = consume captDo, captAttn = consume captAttn] in
                BLAS.sgemm(
                    CblasRowMajor, CblasNoTrans, CblasTrans,
                    m: Int32(dim), n: Int32(dim), k: Int32(seqLen),
                    alpha: 1.0,
                    a: captDo.pointer, lda: Int32(seqLen),
                    b: captAttn.pointer, ldb: Int32(seqLen),
                    beta: 1.0,
                    c: grWo.pointer, ldc: Int32(dim)
                )
            }

            // === STEP 4: SDPA backward (ANE) ===
            let sdpa1In: IOSurfaceRef
            if let handles = layerHandles {
                sdpa1In = handles.sdpaBwd1In
            } else {
                sdpa1In = try kernels[L].sdpaBwd1.inputSurface(at: 0)
            }
            let fwdAttnOut: IOSurfaceRef
            if let handles = layerHandles {
                fwdAttnOut = handles.fwdAttnOut
            } else {
                fwdAttnOut = try kernels[L].fwdAttn.outputSurface(at: 0)
            }

            // #6: copy Q|K|V from fwdAttn output (skip o_out at offset 0; Q starts at offset dim)
            t0 = RuntimeClock.now()
            try mapSurfaceIO {
                try SurfaceIO.copyFP16(
                    dst: sdpa1In,
                    dstChannelOffset: 0,
                    src: fwdAttnOut,
                    srcChannelOffset: dim,
                    channels: 3 * dim,
                    spatial: seqLen
                )
            }
            timings.tIO += RuntimeClock.ms(RuntimeClock.now() - t0)

            // #7: write dx2 at offset 3*dim
            var sdpaWriteError: SurfaceIOError?
            t0 = RuntimeClock.now()
            scratch.dx2.withUnsafeBufferPointer { dx2Buf in
                do {
                    try SurfaceIO.writeFP16At(
                        to: sdpa1In,
                        channelOffset: 3 * dim,
                        data: dx2Buf,
                        channels: dim,
                        spatial: seqLen
                    )
                } catch let e as SurfaceIOError {
                    sdpaWriteError = e
                } catch {
                    sdpaWriteError = .interopCallFailed
                }
            }
            if let e = sdpaWriteError {
                throw mapSurfaceError(e)
            }
            timings.tIO += RuntimeClock.ms(RuntimeClock.now() - t0)

            t0 = RuntimeClock.now()
            try kernels[L].sdpaBwd1.eval()
            timings.tAne += RuntimeClock.ms(RuntimeClock.now() - t0)

            // sdpaBwd2 is the weight-free static kernel for this layer.
            let sdpa2In: IOSurfaceRef
            if let handles = layerHandles {
                sdpa2In = handles.sdpaBwd2In
            } else {
                sdpa2In = try staticKernels[L].kernel.inputSurface(at: 0)
            }
            let sdpa2Out: IOSurfaceRef
            if let handles = layerHandles {
                sdpa2Out = handles.sdpaBwd2Out
            } else {
                sdpa2Out = try staticKernels[L].kernel.outputSurface(at: 0)
            }
            let sdpa1Out: IOSurfaceRef
            if let handles = layerHandles {
                sdpa1Out = handles.sdpaBwd1Out
            } else {
                sdpa1Out = try kernels[L].sdpaBwd1.outputSurface(at: 0)
            }

            // #8/#9: stage sdpaBwd2 inputs in one dst lock from two source surfaces.
            t0 = RuntimeClock.now()
            try mapSurfaceIO {
                let regions = [
                    SurfaceIO.FP16SourceCopyRegion(
                        source: sdpa1Out,
                        dstChannelOffset: 0,
                        srcChannelOffset: dim,
                        channels: 2 * scoreCh
                    ),
                    SurfaceIO.FP16SourceCopyRegion(
                        source: fwdAttnOut,
                        dstChannelOffset: 2 * scoreCh,
                        srcChannelOffset: dim,
                        channels: 2 * dim
                    ),
                ]
                try SurfaceIO.copyFP16FromMultipleSources(
                    dst: sdpa2In,
                    spatial: seqLen,
                    regions: regions
                )
            }
            timings.tIO += RuntimeClock.ms(RuntimeClock.now() - t0)

            t0 = RuntimeClock.now()
            try staticKernels[L].kernel.eval()
            timings.tAne += RuntimeClock.ms(RuntimeClock.now() - t0)

            // #10/#11: read dq/dk from sdpaBwd2 output
            t0 = RuntimeClock.now()
            scratch.dq.withUnsafeMutableBufferPointer { dqBuf in
                scratch.dk.withUnsafeMutableBufferPointer { dkBuf in
                    let regions = [
                        SurfaceIO.FP16ReadRegion(destination: requireBase(dqBuf), channelOffset: 0, channels: dim),
                        SurfaceIO.FP16ReadRegion(destination: requireBase(dkBuf), channelOffset: dim, channels: dim),
                    ]
                    SurfaceIO.readFP16Batched(from: sdpa2Out, spatial: seqLen, regions: regions)
                }
            }
            timings.tIO += RuntimeClock.ms(RuntimeClock.now() - t0)

            // #12: read dv from sdpaBwd1 output (dv is NOT produced by sdpaBwd2)
            t0 = RuntimeClock.now()
            scratch.dv.withUnsafeMutableBufferPointer { dst in
                SurfaceIO.readFP16(from: sdpa1Out, into: dst, channelOffset: 0, channels: dim, spatial: seqLen)
            }
            timings.tIO += RuntimeClock.ms(RuntimeClock.now() - t0)

            // === STEP 5: Async dWq/dWk/dWv (CPU) ===
            let grWq = grads[L].Wq.withUnsafeMutablePointer { SendablePointer($0) }
            let grWk = grads[L].Wk.withUnsafeMutablePointer { SendablePointer($0) }
            let grWv = grads[L].Wv.withUnsafeMutablePointer { SendablePointer($0) }

            let captDq = SendableBuffer(copying: scratch.dq)
            let captDk = SendableBuffer(copying: scratch.dk)
            let captDv = SendableBuffer(copying: scratch.dv)
            let captXn = SendableBuffer(copying: acts[L].xnorm)

            accumulator.enqueue { [captDq = consume captDq, captDk = consume captDk, captDv = consume captDv, captXn = consume captXn] in
                BLAS.sgemm(
                    CblasRowMajor, CblasNoTrans, CblasTrans,
                    m: Int32(dim), n: Int32(dim), k: Int32(seqLen),
                    alpha: 1.0,
                    a: captDq.pointer, lda: Int32(seqLen),
                    b: captXn.pointer, ldb: Int32(seqLen),
                    beta: 1.0,
                    c: grWq.pointer, ldc: Int32(dim)
                )
                BLAS.sgemm(
                    CblasRowMajor, CblasNoTrans, CblasTrans,
                    m: Int32(dim), n: Int32(dim), k: Int32(seqLen),
                    alpha: 1.0,
                    a: captDk.pointer, lda: Int32(seqLen),
                    b: captXn.pointer, ldb: Int32(seqLen),
                    beta: 1.0,
                    c: grWk.pointer, ldc: Int32(dim)
                )
                BLAS.sgemm(
                    CblasRowMajor, CblasNoTrans, CblasTrans,
                    m: Int32(dim), n: Int32(dim), k: Int32(seqLen),
                    alpha: 1.0,
                    a: captDv.pointer, lda: Int32(seqLen),
                    b: captXn.pointer, ldb: Int32(seqLen),
                    beta: 1.0,
                    c: grWv.pointer, ldc: Int32(dim)
                )
            }

            // === STEP 6: QKV backward (ANE) ===
            let qkvIn: IOSurfaceRef
            if let handles = layerHandles {
                qkvIn = handles.qkvBwdIn
            } else {
                qkvIn = try kernels[L].qkvBwd.inputSurface(at: 0)
            }
            let qkvOut: IOSurfaceRef
            if let handles = layerHandles {
                qkvOut = handles.qkvBwdOut
            } else {
                qkvOut = try kernels[L].qkvBwd.outputSurface(at: 0)
            }

            // #13/#14: pack qkvBwd input from two source surfaces in one dst lock.
            t0 = RuntimeClock.now()
            try mapSurfaceIO {
                let regions = [
                    SurfaceIO.FP16SourceCopyRegion(
                        source: sdpa2Out,
                        dstChannelOffset: 0,
                        srcChannelOffset: 0,
                        channels: 2 * dim
                    ),
                    SurfaceIO.FP16SourceCopyRegion(
                        source: sdpa1Out,
                        dstChannelOffset: 2 * dim,
                        srcChannelOffset: 0,
                        channels: dim
                    ),
                ]
                try SurfaceIO.copyFP16FromMultipleSources(
                    dst: qkvIn,
                    spatial: seqLen,
                    regions: regions
                )
            }
            timings.tIO += RuntimeClock.ms(RuntimeClock.now() - t0)

            t0 = RuntimeClock.now()
            try kernels[L].qkvBwd.eval()
            timings.tAne += RuntimeClock.ms(RuntimeClock.now() - t0)

            // #15: read dx_attn
            t0 = RuntimeClock.now()
            scratch.dxAttn.withUnsafeMutableBufferPointer { dst in
                SurfaceIO.readFP16(from: qkvOut, into: dst, channelOffset: 0, channels: dim, spatial: seqLen)
            }
            timings.tIO += RuntimeClock.ms(RuntimeClock.now() - t0)

            // === STEP 7: RMSNorm1 backward (CPU) ===
            scratch.dxRms1.withUnsafeMutablePointer { dxRms1Ptr in
                grads[L].rmsAtt.withUnsafeMutablePointer { dwPtr in
                    scratch.dxAttn.withUnsafePointer { dxAttnPtr in
                        acts[L].layerIn.withUnsafePointer { xPtr in
                            weights[L].rmsAtt.withUnsafePointer { wPtr in
                                RMSNorm.backward(
                                    dx: dxRms1Ptr,
                                    dw: dwPtr,
                                    dy: dxAttnPtr,
                                    x: xPtr,
                                    weights: wPtr,
                                    dim: dim,
                                    seqLen: seqLen
                                )
                            }
                        }
                    }
                }
            }

            // === STEP 8: Propagate gradient ===
            dy.withUnsafeMutablePointer { dyPtr in
                scratch.dxRms1.withUnsafePointer { dxRms1Ptr in
                    scratch.dx2.withUnsafePointer { dx2Ptr in
                        vDSP_vadd(dxRms1Ptr, 1, dx2Ptr, 1, dyPtr, 1, vDSP_Length(dimSeq))
                    }
                }
            }
        }

        // Barrier intentionally handled by caller to preserve timing attribution.
    }
}
