import Foundation
import ANETypes
import MILGenerator

/// Fused 3-layer RWKV trunk + factored RMSNorm + [dim→k→vocab] classifier in one ANE dispatch.
///
/// Inputs:  x [1,dim,1,lane], stateIn0..2 [1,dim,1,lane]
/// Outputs: xNext [1,dim,1,lane], stateOut0..2 [1,dim,1,lane], logits [1,vocab,1,lane]
public struct RWKVStyleFusedThreeLayerFactoredClassifierKernelSet: ~Copyable {
    public let fusedStep: ANEKernel
    public let vocabSize: Int
    public let bottleneck: Int
    public let laneSpatial: Int

    public init(
        weights0: borrowing RWKVStyleRecurrentWeights,
        weights1: borrowing RWKVStyleRecurrentWeights,
        weights2: borrowing RWKVStyleRecurrentWeights,
        rmsFinal: borrowing TensorBuffer,
        classifierProjection: borrowing TensorBuffer,
        classifierExpansion: borrowing TensorBuffer,
        vocabSize: Int,
        bottleneck: Int = 128,
        laneSpatial: Int = 32,
        groups: Int = 1
    ) throws(ANEError) {
        guard laneSpatial > 0 else {
            throw .invalidArguments("laneSpatial must be > 0")
        }
        guard vocabSize > 0 else {
            throw .invalidArguments("vocabSize must be > 0")
        }
        guard bottleneck > 0 else {
            throw .invalidArguments("bottleneck must be > 0")
        }

        let dim = ModelConfig.dim
        let colsPerConv = dim / groups
        // Head convs are always dense (groups=1), so cols = full dim / bottleneck
        let projCols = dim
        let expCols = bottleneck
        let generator = RWKVStyleFusedThreeLayerFactoredClassifierGenerator(
            vocabSize: vocabSize,
            bottleneck: bottleneck,
            laneSpatial: laneSpatial,
            groups: groups
        )

        self.fusedStep = try ANEKernel(
            milText: generator.milText,
            weights: [
                // Layer 0
                (path: "@model_path/weights/rwkv_rms0.bin", data: Self.buildBlob(from: weights0.rms, rows: 1, cols: dim)),
                (path: "@model_path/weights/wx0.bin", data: Self.buildBlob(from: weights0.Wx, rows: dim, cols: colsPerConv)),
                (path: "@model_path/weights/ws0.bin", data: Self.buildBlob(from: weights0.Ws, rows: dim, cols: colsPerConv)),
                (path: "@model_path/weights/wd0.bin", data: Self.buildBlob(from: weights0.Wd, rows: dim, cols: colsPerConv)),
                (path: "@model_path/weights/wo0.bin", data: Self.buildBlob(from: weights0.Wo, rows: dim, cols: colsPerConv)),
                // Layer 1
                (path: "@model_path/weights/rwkv_rms1.bin", data: Self.buildBlob(from: weights1.rms, rows: 1, cols: dim)),
                (path: "@model_path/weights/wx1.bin", data: Self.buildBlob(from: weights1.Wx, rows: dim, cols: colsPerConv)),
                (path: "@model_path/weights/ws1.bin", data: Self.buildBlob(from: weights1.Ws, rows: dim, cols: colsPerConv)),
                (path: "@model_path/weights/wd1.bin", data: Self.buildBlob(from: weights1.Wd, rows: dim, cols: colsPerConv)),
                (path: "@model_path/weights/wo1.bin", data: Self.buildBlob(from: weights1.Wo, rows: dim, cols: colsPerConv)),
                // Layer 2
                (path: "@model_path/weights/rwkv_rms2.bin", data: Self.buildBlob(from: weights2.rms, rows: 1, cols: dim)),
                (path: "@model_path/weights/wx2.bin", data: Self.buildBlob(from: weights2.Wx, rows: dim, cols: colsPerConv)),
                (path: "@model_path/weights/ws2.bin", data: Self.buildBlob(from: weights2.Ws, rows: dim, cols: colsPerConv)),
                (path: "@model_path/weights/wd2.bin", data: Self.buildBlob(from: weights2.Wd, rows: dim, cols: colsPerConv)),
                (path: "@model_path/weights/wo2.bin", data: Self.buildBlob(from: weights2.Wo, rows: dim, cols: colsPerConv)),
                // Head: RMS final + factored classifier
                (path: "@model_path/weights/rms_final.bin", data: Self.buildBlob(from: rmsFinal, rows: 1, cols: dim)),
                (path: "@model_path/weights/cls_proj.bin", data: Self.buildBlob(from: classifierProjection, rows: bottleneck, cols: projCols)),
                (path: "@model_path/weights/cls_expand.bin", data: Self.buildBlob(from: classifierExpansion, rows: vocabSize, cols: expCols)),
            ],
            inputSizes: generator.inputByteSizes,
            outputSizes: generator.outputByteSizes
        )
        self.vocabSize = vocabSize
        self.bottleneck = bottleneck
        self.laneSpatial = laneSpatial
    }

    @inline(__always)
    private static func buildBlob(from buffer: borrowing TensorBuffer, rows: Int, cols: Int) -> Data {
        buffer.withUnsafeBufferPointer { ptr in
            if ptr.count == rows * cols {
                return WeightBlob.build(from: ptr, rows: rows, cols: cols)
            }
            // Slice: take only the first rows*cols elements (for grouped convs)
            let sliced = UnsafeBufferPointer(start: ptr.baseAddress, count: rows * cols)
            return WeightBlob.build(from: sliced, rows: rows, cols: cols)
        }
    }
}
