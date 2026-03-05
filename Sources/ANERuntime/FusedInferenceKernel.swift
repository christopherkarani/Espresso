import Foundation
import IOSurface
import ANETypes
import MILGenerator

/// Owns a single fused ANE kernel for one transformer layer (SDPA + FFN).
///
/// Unlike `InferenceKernelSet` which compiles 2 kernels (attention + FFN),
/// this compiles a single kernel from `FusedLayerInferenceGenerator` that
/// does the entire layer in one ANE dispatch.
///
/// `~Copyable`: deinit frees the kernel handle.
public struct FusedInferenceKernel: ~Copyable {
    public let kernel: ANEKernel

    @inline(__always)
    private static func buildBlob(from buffer: borrowing TensorBuffer, rows: Int, cols: Int) -> Data {
        buffer.withUnsafeBufferPointer { ptr in
            WeightBlob.build(from: ptr, rows: rows, cols: cols)
        }
    }

    /// Compile the fused layer kernel from layer weights.
    public init(weights: borrowing LayerWeights) throws(ANEError) {
        let dim = ModelConfig.dim
        let hidden = ModelConfig.hidden
        let generator = FusedLayerInferenceGenerator()

        // SDPA weights
        let rms1Blob = Self.buildBlob(from: weights.rmsAtt, rows: 1, cols: dim)
        let wqBlob = Self.buildBlob(from: weights.Wq, rows: dim, cols: dim)
        let wkBlob = Self.buildBlob(from: weights.Wk, rows: dim, cols: dim)
        let wvBlob = Self.buildBlob(from: weights.Wv, rows: dim, cols: dim)
        let woBlob = Self.buildBlob(from: weights.Wo, rows: dim, cols: dim)
        let maskBlob = CausalMask.blob(seqLen: ModelConfig.seqLen)

        // FFN weights
        let rms2Blob = Self.buildBlob(from: weights.rmsFfn, rows: 1, cols: dim)
        let w1Blob = Self.buildBlob(from: weights.W1, rows: hidden, cols: dim)
        let w3Blob = Self.buildBlob(from: weights.W3, rows: hidden, cols: dim)
        let w2Blob = Self.buildBlob(from: weights.W2, rows: dim, cols: hidden)

        self.kernel = try ANEKernel(
            milText: generator.milText,
            weights: [
                (path: "@model_path/weights/rms1.bin", data: rms1Blob),
                (path: "@model_path/weights/wq.bin", data: wqBlob),
                (path: "@model_path/weights/wk.bin", data: wkBlob),
                (path: "@model_path/weights/wv.bin", data: wvBlob),
                (path: "@model_path/weights/wo.bin", data: woBlob),
                (path: "@model_path/weights/mask.bin", data: maskBlob),
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
