import Foundation
import ANETypes
import MILGenerator

public struct RWKVStyleFusedThreeLayerKernelSet: ~Copyable {
    public static let defaultLaneSpatial = 32

    internal enum KernelKind: String, CaseIterable {
        case fusedThreeLayerStep
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
    public let includeRMSNorm: Bool

    private init(step: consuming ANEKernel, laneSpatial: Int, includeRMSNorm: Bool) {
        self.step = step
        self.laneSpatial = laneSpatial
        self.includeRMSNorm = includeRMSNorm
    }

    public init(
        weights0: borrowing RWKVStyleRecurrentWeights,
        weights1: borrowing RWKVStyleRecurrentWeights,
        weights2: borrowing RWKVStyleRecurrentWeights,
        laneSpatial: Int = defaultLaneSpatial,
        groups: Int = 1,
        includeRMSNorm: Bool = true
    ) throws(ANEError) {
        guard laneSpatial > 0 else {
            throw .invalidArguments("fused three-layer recurrent laneSpatial must be > 0")
        }
        guard groups > 0 else {
            throw .invalidArguments("fused three-layer recurrent groups must be > 0")
        }
        guard ModelConfig.dim.isMultiple(of: groups) else {
            throw .invalidArguments("fused three-layer recurrent dim \(ModelConfig.dim) must be divisible by groups \(groups)")
        }
        let compiled = try Self.compileStep(
            weights0: weights0,
            weights1: weights1,
            weights2: weights2,
            laneSpatial: laneSpatial,
            groups: groups,
            includeRMSNorm: includeRMSNorm
        )
        self.init(step: compiled, laneSpatial: laneSpatial, includeRMSNorm: includeRMSNorm)
    }

    internal static func compileSpecs(
        weights0: borrowing RWKVStyleRecurrentWeights,
        weights1: borrowing RWKVStyleRecurrentWeights,
        weights2: borrowing RWKVStyleRecurrentWeights,
        laneSpatial: Int,
        groups: Int = 1,
        includeRMSNorm: Bool = true
    ) -> [CompileSpec] {
        precondition(laneSpatial > 0)
        return [
            makeFusedStepSpec(weights0: weights0, weights1: weights1, weights2: weights2, laneSpatial: laneSpatial, groups: groups, includeRMSNorm: includeRMSNorm),
        ]
    }

    private static func compileStep(
        weights0: borrowing RWKVStyleRecurrentWeights,
        weights1: borrowing RWKVStyleRecurrentWeights,
        weights2: borrowing RWKVStyleRecurrentWeights,
        laneSpatial: Int,
        groups: Int = 1,
        includeRMSNorm: Bool = true
    ) throws(ANEError) -> ANEKernel {
        let spec = makeFusedStepSpec(weights0: weights0, weights1: weights1, weights2: weights2, laneSpatial: laneSpatial, groups: groups, includeRMSNorm: includeRMSNorm)
        return try ANEKernel(
            milText: spec.milText,
            weights: spec.weights,
            inputSizes: spec.inputSizes,
            outputSizes: spec.outputSizes
        )
    }

    private static func makeFusedStepSpec(
        weights0: borrowing RWKVStyleRecurrentWeights,
        weights1: borrowing RWKVStyleRecurrentWeights,
        weights2: borrowing RWKVStyleRecurrentWeights,
        laneSpatial: Int,
        groups: Int = 1,
        includeRMSNorm: Bool = true
    ) -> CompileSpec {
        let dim = ModelConfig.dim
        precondition(groups > 0)
        precondition(dim.isMultiple(of: groups))
        let colsPerConv = dim / groups
        let generator = RWKVStyleFusedThreeLayerStepGenerator(laneSpatial: laneSpatial, groups: groups, includeRMSNorm: includeRMSNorm)

        var weights: [(path: String, data: Data)] = []
        weights.reserveCapacity(includeRMSNorm ? 15 : 12)

        func addLayer(_ w: borrowing RWKVStyleRecurrentWeights, index: Int) {
            if includeRMSNorm {
                weights.append((path: "@model_path/weights/rwkv_rms\(index).bin", data: buildBlob(from: w.rms, rows: 1, cols: dim)))
            }
            weights.append((path: "@model_path/weights/wx\(index).bin", data: GroupedWeightBlob.build(from: w.Wx, rows: dim, colsPerGroup: colsPerConv, groups: groups)))
            weights.append((path: "@model_path/weights/ws\(index).bin", data: GroupedWeightBlob.build(from: w.Ws, rows: dim, colsPerGroup: colsPerConv, groups: groups)))
            weights.append((path: "@model_path/weights/wd\(index).bin", data: GroupedWeightBlob.build(from: w.Wd, rows: dim, colsPerGroup: colsPerConv, groups: groups)))
            weights.append((path: "@model_path/weights/wo\(index).bin", data: GroupedWeightBlob.build(from: w.Wo, rows: dim, colsPerGroup: colsPerConv, groups: groups)))
        }

        addLayer(weights0, index: 0)
        addLayer(weights1, index: 1)
        addLayer(weights2, index: 2)

        return CompileSpec(
            kind: .fusedThreeLayerStep,
            milText: generator.milText,
            weights: weights,
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
