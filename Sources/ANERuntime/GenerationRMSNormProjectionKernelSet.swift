import Foundation
import ANETypes
import MILGenerator

/// ANE kernel: RMSNorm + projection conv [dim → bottleneck] only.
/// Output is a small projected surface for CPU-side expansion + argmax.
public struct GenerationRMSNormProjectionKernelSet: ~Copyable {
    public let rmsNormProjection: ANEKernel
    public let bottleneck: Int
    public let laneSpatial: Int
    public let groups: Int

    public init(
        rmsFinal: borrowing TensorBuffer,
        classifierProjection: borrowing TensorBuffer,
        bottleneck: Int = 128,
        laneSpatial: Int = 32,
        groups: Int = 1
    ) throws(ANEError) {
        guard bottleneck > 0 else {
            throw .invalidArguments("bottleneck must be > 0")
        }
        guard laneSpatial > 0 else {
            throw .invalidArguments("laneSpatial must be > 0")
        }
        guard groups > 0 else {
            throw .invalidArguments("groups must be > 0")
        }
        guard ModelConfig.dim.isMultiple(of: groups) else {
            throw .invalidArguments("dim \(ModelConfig.dim) must be divisible by groups \(groups)")
        }
        guard bottleneck.isMultiple(of: groups) else {
            throw .invalidArguments("bottleneck \(bottleneck) must be divisible by groups \(groups)")
        }
        guard rmsFinal.count == ModelConfig.dim else {
            throw .invalidArguments("rmsFinal count \(rmsFinal.count) must equal dim \(ModelConfig.dim)")
        }

        let dim = ModelConfig.dim
        let projColsPerGroup = dim / groups

        let projExpectedCount = bottleneck * projColsPerGroup
        let projDenseCount = bottleneck * dim
        guard classifierProjection.count == projExpectedCount || classifierProjection.count == projDenseCount else {
            throw .invalidArguments(
                "classifierProjection count \(classifierProjection.count) must equal grouped \(projExpectedCount) or dense \(projDenseCount)"
            )
        }

        let generator = GenerationRMSNormProjectionGenerator(
            bottleneck: bottleneck,
            laneSpatial: laneSpatial,
            groups: groups
        )

        let rmsBlob = rmsFinal.withUnsafeBufferPointer { ptr in
            WeightBlob.build(from: ptr, rows: 1, cols: dim)
        }
        let projBlob = GroupedWeightBlob.build(from: classifierProjection, rows: bottleneck, colsPerGroup: projColsPerGroup, groups: groups)

        self.rmsNormProjection = try ANEKernel(
            milText: generator.milText,
            weights: [
                (path: "@model_path/weights/rms_final.bin", data: rmsBlob),
                (path: "@model_path/weights/cls_proj.bin", data: projBlob),
            ],
            inputSizes: generator.inputByteSizes,
            outputSizes: generator.outputByteSizes
        )
        self.bottleneck = bottleneck
        self.laneSpatial = laneSpatial
        self.groups = groups
    }
}
