import Foundation
import IOSurface
import ANETypes
import MILGenerator

/// Owns exactly 2 inference-only ANE kernels for a single transformer layer.
///
/// Unlike `LayerKernelSet` which compiles all 5 kernels (2 forward + 3 backward),
/// this only compiles the 2 inference forward kernels using fused-residual MIL generators.
/// This halves compile time and uses inference-optimized kernels that output only `dim` channels
/// instead of `6*dim` (attention) or `2*dim + 3*hidden` (FFN).
///
/// `~Copyable`: deinit frees both kernel handles.
public struct InferenceKernelSet: ~Copyable {
    public let fwdAttn: ANEKernel
    public let fwdFFN: ANEKernel

    @inline(__always)
    private static func buildBlob(from buffer: borrowing TensorBuffer, rows: Int, cols: Int) -> Data {
        buffer.withUnsafeBufferPointer { ptr in
            WeightBlob.build(from: ptr, rows: rows, cols: cols)
        }
    }

    private init(fwdAttn: consuming ANEKernel, fwdFFN: consuming ANEKernel) {
        self.fwdAttn = fwdAttn
        self.fwdFFN = fwdFFN
    }

    /// Compile both inference forward kernels from layer weights.
    /// Uses `borrowing` to avoid copying the ~324 MiB LayerWeights.
    public init(weights: borrowing LayerWeights) throws(ANEError) {
        let compiledAttn = try Self.compileFwdAttn(weights: weights)
        let compiledFFN = try Self.compileFwdFFN(weights: weights)
        self.init(fwdAttn: compiledAttn, fwdFFN: compiledFFN)
    }

    private static func compileFwdAttn(weights: borrowing LayerWeights) throws(ANEError) -> ANEKernel {
        let dim = ModelConfig.dim
        let generator = SDPAForwardInferenceGenerator()

        let rms1Blob = buildBlob(from: weights.rmsAtt, rows: 1, cols: dim)
        let wqBlob = buildBlob(from: weights.Wq, rows: dim, cols: dim)
        let wkBlob = buildBlob(from: weights.Wk, rows: dim, cols: dim)
        let wvBlob = buildBlob(from: weights.Wv, rows: dim, cols: dim)
        let woBlob = buildBlob(from: weights.Wo, rows: dim, cols: dim)
        let maskBlob = CausalMask.blob(seqLen: ModelConfig.seqLen)

        return try ANEKernel(
            milText: generator.milText,
            weights: [
                (path: "@model_path/weights/rms1.bin", data: rms1Blob),
                (path: "@model_path/weights/wq.bin", data: wqBlob),
                (path: "@model_path/weights/wk.bin", data: wkBlob),
                (path: "@model_path/weights/wv.bin", data: wvBlob),
                (path: "@model_path/weights/wo.bin", data: woBlob),
                (path: "@model_path/weights/mask.bin", data: maskBlob),
            ],
            inputBytes: generator.inputBytes,
            outputBytes: generator.outputBytes
        )
    }

    private static func compileFwdFFN(weights: borrowing LayerWeights) throws(ANEError) -> ANEKernel {
        let dim = ModelConfig.dim
        let hidden = ModelConfig.hidden
        let generator = FFNForwardInferenceGenerator()

        let rms2Blob = buildBlob(from: weights.rmsFfn, rows: 1, cols: dim)
        let w1Blob = buildBlob(from: weights.W1, rows: hidden, cols: dim)
        let w3Blob = buildBlob(from: weights.W3, rows: hidden, cols: dim)
        let w2Blob = buildBlob(from: weights.W2, rows: dim, cols: hidden)

        return try ANEKernel(
            milText: generator.milText,
            weights: [
                (path: "@model_path/weights/rms2.bin", data: rms2Blob),
                (path: "@model_path/weights/w1.bin", data: w1Blob),
                (path: "@model_path/weights/w3.bin", data: w3Blob),
                (path: "@model_path/weights/w2.bin", data: w2Blob),
            ],
            inputBytes: generator.inputBytes,
            outputBytes: generator.outputBytes
        )
    }
}
