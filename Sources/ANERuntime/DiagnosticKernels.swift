import Foundation
import IOSurface
import ANETypes
import MILGenerator

// MARK: - Experiment 1: Convolutions-Only Kernel

/// Pure convolution throughput kernel (7 weight blobs, no attention/norms).
///
/// Establishes the compute floor by measuring only the 7 matrix multiplies
/// without any attention logic, RMSNorm, or residual connections.
///
/// `~Copyable`: deinit frees the kernel handle.
public struct ConvsOnlyKernel: ~Copyable {
    public let kernel: ANEKernel

    @inline(__always)
    private static func buildBlob(from buffer: borrowing TensorBuffer, rows: Int, cols: Int) -> Data {
        buffer.withUnsafeBufferPointer { ptr in
            WeightBlob.build(from: ptr, rows: rows, cols: cols)
        }
    }

    /// Compile the convs-only kernel from layer weights.
    public init(weights: borrowing LayerWeights) throws(ANEError) {
        let dim = ModelConfig.dim
        let hidden = ModelConfig.hidden
        let generator = ConvsOnlyGenerator()

        let wqBlob = Self.buildBlob(from: weights.Wq, rows: dim, cols: dim)
        let wkBlob = Self.buildBlob(from: weights.Wk, rows: dim, cols: dim)
        let wvBlob = Self.buildBlob(from: weights.Wv, rows: dim, cols: dim)
        let woBlob = Self.buildBlob(from: weights.Wo, rows: dim, cols: dim)
        let w1Blob = Self.buildBlob(from: weights.W1, rows: hidden, cols: dim)
        let w3Blob = Self.buildBlob(from: weights.W3, rows: hidden, cols: dim)
        let w2Blob = Self.buildBlob(from: weights.W2, rows: dim, cols: hidden)

        self.kernel = try ANEKernel(
            milText: generator.milText,
            weights: [
                (path: "@model_path/weights/wq.bin", data: wqBlob),
                (path: "@model_path/weights/wk.bin", data: wkBlob),
                (path: "@model_path/weights/wv.bin", data: wvBlob),
                (path: "@model_path/weights/wo.bin", data: woBlob),
                (path: "@model_path/weights/w1.bin", data: w1Blob),
                (path: "@model_path/weights/w3.bin", data: w3Blob),
                (path: "@model_path/weights/w2.bin", data: w2Blob),
            ],
            inputBytes: generator.inputBytes,
            outputBytes: generator.outputBytes
        )
    }
}

// MARK: - Experiment 2: Batched-Head Fused Kernel

/// Fused layer kernel using batched reshape→transpose→matmul attention
/// instead of split-head slice_by_index approach.
///
/// Structurally identical to `FusedInferenceKernel` — same 10 weight blobs,
/// same I/O contract — but uses `BatchedHeadFusedGenerator` for the MIL program.
///
/// `~Copyable`: deinit frees the kernel handle.
public struct BatchedHeadFusedKernel: ~Copyable {
    public let kernel: ANEKernel

    @inline(__always)
    private static func buildBlob(from buffer: borrowing TensorBuffer, rows: Int, cols: Int) -> Data {
        buffer.withUnsafeBufferPointer { ptr in
            WeightBlob.build(from: ptr, rows: rows, cols: cols)
        }
    }

    /// Compile the batched-head fused layer kernel from layer weights.
    public init(weights: borrowing LayerWeights) throws(ANEError) {
        let dim = ModelConfig.dim
        let hidden = ModelConfig.hidden
        let generator = BatchedHeadFusedGenerator()

        let rms1Blob = Self.buildBlob(from: weights.rmsAtt, rows: 1, cols: dim)
        let wqBlob = Self.buildBlob(from: weights.Wq, rows: dim, cols: dim)
        let wkBlob = Self.buildBlob(from: weights.Wk, rows: dim, cols: dim)
        let wvBlob = Self.buildBlob(from: weights.Wv, rows: dim, cols: dim)
        let woBlob = Self.buildBlob(from: weights.Wo, rows: dim, cols: dim)
        let maskBlob = CausalMask.blob(seqLen: ModelConfig.seqLen)
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

// MARK: - Experiment 3: Whole-Model Kernel

/// All N transformer layers fused into one MIL program / one ANE kernel.
///
/// Eliminates N-1 inter-layer eval() dispatches and surface I/O round-trips.
/// Takes `LayerStorage<LayerWeights>` and builds all weight blobs (9 per layer + 1 shared mask).
///
/// `~Copyable`: deinit frees the kernel handle.
public struct WholeModelKernel: ~Copyable {
    public let kernel: ANEKernel

    @inline(__always)
    private static func buildBlob(from buffer: borrowing TensorBuffer, rows: Int, cols: Int) -> Data {
        buffer.withUnsafeBufferPointer { ptr in
            WeightBlob.build(from: ptr, rows: rows, cols: cols)
        }
    }

    /// Compile the whole-model kernel from all layer weights.
    ///
    /// - Parameters:
    ///   - layers: All transformer layer weights
    ///   - nLayers: Number of layers (must match `layers.count`)
    public init(layers: borrowing LayerStorage<LayerWeights>, nLayers: Int) throws(ANEError) {
        precondition(nLayers == layers.count)
        let dim = ModelConfig.dim
        let hidden = ModelConfig.hidden
        let generator = WholeModelGenerator(nLayers: nLayers)

        // Build weight blobs: 9 per layer + 1 shared mask
        var weights: [(path: String, data: Data)] = []
        weights.reserveCapacity(9 * nLayers + 1)

        // Shared causal mask (emitted first, referenced by all layers)
        let maskBlob = CausalMask.blob(seqLen: ModelConfig.seqLen)
        weights.append((path: "@model_path/weights/mask.bin", data: maskBlob))

        // Per-layer weight blobs
        for i in 0..<nLayers {
            let rms1Blob = Self.buildBlob(from: layers[i].rmsAtt, rows: 1, cols: dim)
            let wqBlob = Self.buildBlob(from: layers[i].Wq, rows: dim, cols: dim)
            let wkBlob = Self.buildBlob(from: layers[i].Wk, rows: dim, cols: dim)
            let wvBlob = Self.buildBlob(from: layers[i].Wv, rows: dim, cols: dim)
            let woBlob = Self.buildBlob(from: layers[i].Wo, rows: dim, cols: dim)
            let rms2Blob = Self.buildBlob(from: layers[i].rmsFfn, rows: 1, cols: dim)
            let w1Blob = Self.buildBlob(from: layers[i].W1, rows: hidden, cols: dim)
            let w3Blob = Self.buildBlob(from: layers[i].W3, rows: hidden, cols: dim)
            let w2Blob = Self.buildBlob(from: layers[i].W2, rows: dim, cols: hidden)

            weights.append((path: "@model_path/weights/L\(i)_rms1.bin", data: rms1Blob))
            weights.append((path: "@model_path/weights/L\(i)_wq.bin", data: wqBlob))
            weights.append((path: "@model_path/weights/L\(i)_wk.bin", data: wkBlob))
            weights.append((path: "@model_path/weights/L\(i)_wv.bin", data: wvBlob))
            weights.append((path: "@model_path/weights/L\(i)_wo.bin", data: woBlob))
            weights.append((path: "@model_path/weights/L\(i)_rms2.bin", data: rms2Blob))
            weights.append((path: "@model_path/weights/L\(i)_w1.bin", data: w1Blob))
            weights.append((path: "@model_path/weights/L\(i)_w3.bin", data: w3Blob))
            weights.append((path: "@model_path/weights/L\(i)_w2.bin", data: w2Blob))
        }

        self.kernel = try ANEKernel(
            milText: generator.milText,
            weights: weights,
            inputBytes: generator.inputBytes,
            outputBytes: generator.outputBytes
        )
    }
}
