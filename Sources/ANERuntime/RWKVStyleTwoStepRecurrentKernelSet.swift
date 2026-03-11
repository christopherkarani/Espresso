import Foundation
import ANETypes
import MILGenerator

public struct RWKVStyleTwoStepRecurrentKernelSet: ~Copyable {
    public static let defaultLaneSpatial = 32

    public let step: ANEKernel
    public let laneSpatial: Int

    private init(step: consuming ANEKernel, laneSpatial: Int) {
        self.step = step
        self.laneSpatial = laneSpatial
    }

    public init(
        weights: borrowing RWKVStyleRecurrentWeights,
        laneSpatial: Int = defaultLaneSpatial
    ) throws(ANEError) {
        guard laneSpatial > 0 else {
            throw .invalidArguments("two-step recurrent laneSpatial must be > 0")
        }
        let compiled = try Self.compileStep(weights: weights, laneSpatial: laneSpatial)
        self.init(step: compiled, laneSpatial: laneSpatial)
    }

    private static func compileStep(
        weights: borrowing RWKVStyleRecurrentWeights,
        laneSpatial: Int
    ) throws(ANEError) -> ANEKernel {
        let generator = RWKVStyleTwoStepRecurrentGenerator(laneSpatial: laneSpatial)
        let dim = ModelConfig.dim
        return try ANEKernel(
            milText: generator.milText,
            weights: [
                (path: "@model_path/weights/rwkv_rms.bin", data: buildBlob(from: weights.rms, rows: 1, cols: dim)),
                (path: "@model_path/weights/wx.bin", data: buildBlob(from: weights.Wx, rows: dim, cols: dim)),
                (path: "@model_path/weights/ws.bin", data: buildBlob(from: weights.Ws, rows: dim, cols: dim)),
                (path: "@model_path/weights/wd.bin", data: buildBlob(from: weights.Wd, rows: dim, cols: dim)),
                (path: "@model_path/weights/wo.bin", data: buildBlob(from: weights.Wo, rows: dim, cols: dim)),
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
