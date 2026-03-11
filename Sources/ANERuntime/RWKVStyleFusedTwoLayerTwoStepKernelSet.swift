import Foundation
import ANETypes
import MILGenerator

public struct RWKVStyleFusedTwoLayerTwoStepKernelSet: ~Copyable {
    public static let defaultLaneSpatial = 32

    public let step: ANEKernel
    public let laneSpatial: Int

    private init(step: consuming ANEKernel, laneSpatial: Int) {
        self.step = step
        self.laneSpatial = laneSpatial
    }

    public init(
        weights0: borrowing RWKVStyleRecurrentWeights,
        weights1: borrowing RWKVStyleRecurrentWeights,
        laneSpatial: Int = defaultLaneSpatial
    ) throws(ANEError) {
        guard laneSpatial > 0 else {
            throw .invalidArguments("fused two-step recurrent laneSpatial must be > 0")
        }
        let compiled = try Self.compileStep(weights0: weights0, weights1: weights1, laneSpatial: laneSpatial)
        self.init(step: compiled, laneSpatial: laneSpatial)
    }

    private static func compileStep(
        weights0: borrowing RWKVStyleRecurrentWeights,
        weights1: borrowing RWKVStyleRecurrentWeights,
        laneSpatial: Int
    ) throws(ANEError) -> ANEKernel {
        let dim = ModelConfig.dim
        let generator = RWKVStyleFusedTwoLayerTwoStepGenerator(laneSpatial: laneSpatial)
        return try ANEKernel(
            milText: generator.milText,
            weights: [
                (path: "@model_path/weights/rwkv_rms0.bin", data: buildBlob(from: weights0.rms, rows: 1, cols: dim)),
                (path: "@model_path/weights/wx0.bin", data: buildBlob(from: weights0.Wx, rows: dim, cols: dim)),
                (path: "@model_path/weights/ws0.bin", data: buildBlob(from: weights0.Ws, rows: dim, cols: dim)),
                (path: "@model_path/weights/wd0.bin", data: buildBlob(from: weights0.Wd, rows: dim, cols: dim)),
                (path: "@model_path/weights/wo0.bin", data: buildBlob(from: weights0.Wo, rows: dim, cols: dim)),
                (path: "@model_path/weights/rwkv_rms1.bin", data: buildBlob(from: weights1.rms, rows: 1, cols: dim)),
                (path: "@model_path/weights/wx1.bin", data: buildBlob(from: weights1.Wx, rows: dim, cols: dim)),
                (path: "@model_path/weights/ws1.bin", data: buildBlob(from: weights1.Ws, rows: dim, cols: dim)),
                (path: "@model_path/weights/wd1.bin", data: buildBlob(from: weights1.Wd, rows: dim, cols: dim)),
                (path: "@model_path/weights/wo1.bin", data: buildBlob(from: weights1.Wo, rows: dim, cols: dim)),
            ],
            inputSizes: generator.inputByteSizes,
            outputSizes: generator.outputByteSizes
        )
    }

    @inline(__always)
    private static func buildBlob(from buffer: borrowing TensorBuffer, rows: Int, cols: Int) -> Data {
        buffer.withUnsafeBufferPointer { ptr in
            WeightBlob.build(from: ptr, rows: rows, cols: cols)
        }
    }
}
