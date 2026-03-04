import Accelerate
import ANERuntime
import ANETypes

/// Transformer forward pass using ANE kernels plus CPU residual additions.
/// Maps to `train_large.m:384-420`.
public enum ForwardPass {
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
        seqLen: Int = ModelConfig.seqLen
    ) throws(ANEError) {
        precondition(dim > 0 && hidden > 0 && seqLen > 0)
        precondition(xCur.count == dim * seqLen)
        precondition(acts.count == kernels.count)

        let dimSeq = dim * seqLen
        let dimSeqBytes = dimSeq * MemoryLayout<Float>.stride

        for L in 0..<kernels.count {
            // Save input for RMSNorm backward.
            xCur.withUnsafePointer { xPtr in
                acts[L].layerIn.withUnsafeMutablePointer { dst in
                    _ = memcpy(dst, xPtr, dimSeqBytes)
                }
            }

            // Barrier: ensure no async dW blocks still use surfaces/buffers we're about to overwrite.
            accumulator.barrier()

            // Attention forward.
            let attnIn = try kernels[L].fwdAttn.inputSurface(at: 0)
            xCur.withUnsafeBufferPointer { xBuf in
                SurfaceIO.writeFP16(to: attnIn, data: xBuf, channels: dim, spatial: seqLen)
            }
            try kernels[L].fwdAttn.eval()

            let attnOut = try kernels[L].fwdAttn.outputSurface(at: 0)
            acts[L].oOut.withUnsafeMutableBufferPointer { dst in
                SurfaceIO.readFP16(from: attnOut, into: dst, channelOffset: 0, channels: dim, spatial: seqLen)
            }
            acts[L].attnOut.withUnsafeMutableBufferPointer { dst in
                SurfaceIO.readFP16(from: attnOut, into: dst, channelOffset: 4 * dim, channels: dim, spatial: seqLen)
            }
            acts[L].xnorm.withUnsafeMutableBufferPointer { dst in
                SurfaceIO.readFP16(from: attnOut, into: dst, channelOffset: 5 * dim, channels: dim, spatial: seqLen)
            }
            // NOTE: Q/K/V at offsets 1*dim, 2*dim, 3*dim are intentionally not read back to CPU.

            // Residual: x2 = xCur + oOut.
            xCur.withUnsafePointer { xPtr in
                acts[L].oOut.withUnsafePointer { oPtr in
                    acts[L].x2.withUnsafeMutablePointer { x2Ptr in
                        vDSP_vadd(xPtr, 1, oPtr, 1, x2Ptr, 1, vDSP_Length(dimSeq))
                    }
                }
            }

            // FFN forward.
            let ffnIn = try kernels[L].fwdFFN.inputSurface(at: 0)
            acts[L].x2.withUnsafeBufferPointer { x2Buf in
                SurfaceIO.writeFP16(to: ffnIn, data: x2Buf, channels: dim, spatial: seqLen)
            }
            try kernels[L].fwdFFN.eval()

            let ffnOut = try kernels[L].fwdFFN.outputSurface(at: 0)
            acts[L].ffnOut.withUnsafeMutableBufferPointer { dst in
                SurfaceIO.readFP16(from: ffnOut, into: dst, channelOffset: 0, channels: dim, spatial: seqLen)
            }
            acts[L].h1.withUnsafeMutableBufferPointer { dst in
                SurfaceIO.readFP16(from: ffnOut, into: dst, channelOffset: dim, channels: hidden, spatial: seqLen)
            }
            acts[L].h3.withUnsafeMutableBufferPointer { dst in
                SurfaceIO.readFP16(from: ffnOut, into: dst, channelOffset: dim + hidden, channels: hidden, spatial: seqLen)
            }
            acts[L].siluOut.withUnsafeMutableBufferPointer { dst in
                SurfaceIO.readFP16(from: ffnOut, into: dst, channelOffset: dim + 2 * hidden, channels: hidden, spatial: seqLen)
            }
            acts[L].x2norm.withUnsafeMutableBufferPointer { dst in
                SurfaceIO.readFP16(from: ffnOut, into: dst, channelOffset: dim + 3 * hidden, channels: dim, spatial: seqLen)
            }

            // Residual: xCur = x2 + ffnOut.
            acts[L].x2.withUnsafePointer { x2Ptr in
                acts[L].ffnOut.withUnsafePointer { ffnPtr in
                    xCur.withUnsafeMutablePointer { xCurPtr in
                        vDSP_vadd(x2Ptr, 1, ffnPtr, 1, xCurPtr, 1, vDSP_Length(dimSeq))
                    }
                }
            }
        }
    }
}
