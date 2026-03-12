import Foundation
import ANETypes
import MILGenerator

public struct RWKVStyleStateFreeKernelSet: ~Copyable {
    public static let defaultLaneSpatial = 32

    internal enum KernelKind: String, CaseIterable {
        case recurrentStep
    }

    internal struct CompileSpec {
        internal let kind: KernelKind
        internal let milText: String
        internal let weights: [(path: String, data: Data)]
        internal let inputSizes: [Int]
        internal let outputSizes: [Int]
    }

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
            throw .invalidArguments("state-free recurrent laneSpatial must be > 0")
        }
        let compiled = try Self.compileStep(weights: weights, laneSpatial: laneSpatial)
        self.init(step: compiled, laneSpatial: laneSpatial)
    }

    internal static func compileSpecs(
        weights: borrowing RWKVStyleRecurrentWeights,
        laneSpatial: Int
    ) -> [CompileSpec] {
        precondition(laneSpatial > 0)
        return [
            makeStateFreeStepSpec(weights: weights, laneSpatial: laneSpatial),
        ]
    }

    private static func compileStep(
        weights: borrowing RWKVStyleRecurrentWeights,
        laneSpatial: Int
    ) throws(ANEError) -> ANEKernel {
        let spec = makeStateFreeStepSpec(weights: weights, laneSpatial: laneSpatial)
        return try ANEKernel(
            milText: spec.milText,
            weights: spec.weights,
            inputSizes: spec.inputSizes,
            outputSizes: spec.outputSizes
        )
    }

    private static func makeStateFreeStepSpec(
        weights: borrowing RWKVStyleRecurrentWeights,
        laneSpatial: Int
    ) -> CompileSpec {
        let dim = ModelConfig.dim
        let generator = RWKVStyleStateFreeStepGenerator(laneSpatial: laneSpatial)

        return CompileSpec(
            kind: .recurrentStep,
            milText: generator.milText,
            weights: [
                (path: "@model_path/weights/rwkv_rms.bin", data: buildBlob(from: weights.rms, rows: 1, cols: dim)),
                (path: "@model_path/weights/wx.bin", data: buildBlob(from: weights.Wx, rows: dim, cols: dim)),
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
