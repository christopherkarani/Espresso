import Foundation
import ANETypes
import MILGenerator

public struct FactoredGenerationRMSNormClassifierKernelSet: ~Copyable {
    public let rmsNormClassifier: ANEKernel
    public let vocabSize: Int
    public let bottleneck: Int
    public let laneSpatial: Int

    public init(
        rmsFinal: borrowing TensorBuffer,
        classifierProjection: borrowing TensorBuffer,
        classifierExpansion: borrowing TensorBuffer,
        vocabSize: Int,
        bottleneck: Int = 128,
        laneSpatial: Int = 32
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
        guard rmsFinal.count == ModelConfig.dim else {
            throw .invalidArguments("rmsFinal count \(rmsFinal.count) must equal dim \(ModelConfig.dim)")
        }
        guard classifierProjection.count == bottleneck * ModelConfig.dim else {
            throw .invalidArguments(
                "classifierProjection count \(classifierProjection.count) does not match bottleneck \(bottleneck) * dim \(ModelConfig.dim)"
            )
        }
        guard classifierExpansion.count == vocabSize * bottleneck else {
            throw .invalidArguments(
                "classifierExpansion count \(classifierExpansion.count) does not match vocabSize \(vocabSize) * bottleneck \(bottleneck)"
            )
        }

        let generator = FactoredGenerationRMSNormClassifierGenerator(
            vocabSize: vocabSize,
            bottleneck: bottleneck,
            laneSpatial: laneSpatial
        )

        let rmsBlob = rmsFinal.withUnsafeBufferPointer { ptr in
            WeightBlob.build(from: ptr, rows: 1, cols: ModelConfig.dim)
        }
        let projBlob = classifierProjection.withUnsafeBufferPointer { ptr in
            WeightBlob.build(from: ptr, rows: bottleneck, cols: ModelConfig.dim)
        }
        let expBlob = classifierExpansion.withUnsafeBufferPointer { ptr in
            WeightBlob.build(from: ptr, rows: vocabSize, cols: bottleneck)
        }

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
    }
}
