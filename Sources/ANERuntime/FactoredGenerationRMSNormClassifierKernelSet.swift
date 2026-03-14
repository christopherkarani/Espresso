import Foundation
import ANETypes
import MILGenerator

public struct FactoredGenerationRMSNormClassifierKernelSet: ~Copyable {
    public let rmsNormClassifier: ANEKernel
    public let vocabSize: Int
    public let bottleneck: Int
    public let laneSpatial: Int
    public let groups: Int

    public init(
        rmsFinal: borrowing TensorBuffer,
        classifierProjection: borrowing TensorBuffer,
        classifierExpansion: borrowing TensorBuffer,
        vocabSize: Int,
        bottleneck: Int = 128,
        laneSpatial: Int = 32,
        groups: Int = 1
    ) throws(ANEError) {
        guard vocabSize > 0 else {
            throw .invalidArguments("vocabSize must be > 0")
        }
        guard bottleneck > 0 else {
            throw .invalidArguments("bottleneck must be > 0")
        }
        guard laneSpatial > 0 else {
            throw .invalidArguments("laneSpatial must be > 0")
        }
        guard groups > 0 else {
            throw .invalidArguments("groups must be > 0")
        }
        let dim = ModelConfig.dim
        guard dim.isMultiple(of: groups) else {
            throw .invalidArguments("dim \(dim) must be divisible by groups \(groups)")
        }
        guard bottleneck.isMultiple(of: groups) else {
            throw .invalidArguments("bottleneck \(bottleneck) must be divisible by groups \(groups)")
        }
        guard vocabSize.isMultiple(of: groups) else {
            throw .invalidArguments("vocabSize \(vocabSize) must be divisible by groups \(groups)")
        }
        guard rmsFinal.count == ModelConfig.dim else {
            throw .invalidArguments("rmsFinal count \(rmsFinal.count) must equal dim \(ModelConfig.dim)")
        }

        let projColsPerGroup = dim / groups
        let expColsPerGroup = bottleneck / groups

        let projExpectedCount = bottleneck * projColsPerGroup
        let projDenseCount = bottleneck * dim
        guard classifierProjection.count == projExpectedCount || classifierProjection.count == projDenseCount else {
            throw .invalidArguments(
                "classifierProjection count \(classifierProjection.count) must equal grouped \(projExpectedCount) or dense \(projDenseCount)"
            )
        }
        let expExpectedCount = vocabSize * expColsPerGroup
        let expDenseCount = vocabSize * bottleneck
        guard classifierExpansion.count == expExpectedCount || classifierExpansion.count == expDenseCount else {
            throw .invalidArguments(
                "classifierExpansion count \(classifierExpansion.count) must equal grouped \(expExpectedCount) or dense \(expDenseCount)"
            )
        }

        let generator = FactoredGenerationRMSNormClassifierGenerator(
            vocabSize: vocabSize,
            bottleneck: bottleneck,
            laneSpatial: laneSpatial,
            groups: groups
        )

        let rmsBlob = rmsFinal.withUnsafeBufferPointer { ptr in
            WeightBlob.build(from: ptr, rows: 1, cols: dim)
        }
        let projBlob = GroupedWeightBlob.build(from: classifierProjection, rows: bottleneck, colsPerGroup: projColsPerGroup, groups: groups)
        let expBlob = GroupedWeightBlob.build(from: classifierExpansion, rows: vocabSize, colsPerGroup: expColsPerGroup, groups: groups)

        self.rmsNormClassifier = try ANEKernel(
            milText: generator.milText,
            weights: [
                (path: "@model_path/weights/rms_final.bin", data: rmsBlob),
                (path: "@model_path/weights/cls_proj.bin", data: projBlob),
                (path: "@model_path/weights/cls_expand.bin", data: expBlob),
            ],
            inputSizes: generator.inputByteSizes,
            outputSizes: generator.outputByteSizes
        )
        self.vocabSize = vocabSize
        self.bottleneck = bottleneck
        self.laneSpatial = laneSpatial
        self.groups = groups
    }
}
