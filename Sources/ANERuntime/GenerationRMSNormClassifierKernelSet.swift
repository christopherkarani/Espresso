import Foundation
import ANETypes
import MILGenerator

public struct GenerationRMSNormClassifierKernelSet: ~Copyable {
    public enum KernelKind: String, CaseIterable {
        case rmsNormClassifier
    }

    public struct CompileSpec {
        public let kind: KernelKind
        public let milText: String
        public let weights: [(path: String, data: Data)]
        public let inputSizes: [Int]
        public let outputSizes: [Int]
    }

    public let rmsNormClassifier: ANEKernel
    public let vocabSize: Int
    public let laneSpatial: Int

    private init(rmsNormClassifier: consuming ANEKernel, vocabSize: Int, laneSpatial: Int) {
        self.rmsNormClassifier = rmsNormClassifier
        self.vocabSize = vocabSize
        self.laneSpatial = laneSpatial
    }

    public init(
        rmsFinal: borrowing TensorBuffer,
        classifier: borrowing TensorBuffer,
        vocabSize: Int,
        laneSpatial: Int = 32
    ) throws(ANEError) {
        guard vocabSize > 0 else {
            throw .invalidArguments("vocabSize must be > 0")
        }
        guard laneSpatial > 0 else {
            throw .invalidArguments("laneSpatial must be > 0")
        }
        guard rmsFinal.count == ModelConfig.dim else {
            throw .invalidArguments("rmsFinal count \(rmsFinal.count) must equal dim \(ModelConfig.dim)")
        }
        guard classifier.count == vocabSize * ModelConfig.dim else {
            throw .invalidArguments(
                "classifier weight count \(classifier.count) does not match vocabSize \(vocabSize) * dim \(ModelConfig.dim)"
            )
        }
        let compiled = try Self.compileKernel(
            rmsFinal: rmsFinal,
            classifier: classifier,
            vocabSize: vocabSize,
            laneSpatial: laneSpatial
        )
        self.init(rmsNormClassifier: compiled, vocabSize: vocabSize, laneSpatial: laneSpatial)
    }

    public static func compileSpecs(
        rmsFinal: borrowing TensorBuffer,
        classifier: borrowing TensorBuffer,
        vocabSize: Int,
        laneSpatial: Int = 32
    ) -> [CompileSpec] {
        precondition(vocabSize > 0)
        precondition(laneSpatial > 0)
        precondition(rmsFinal.count == ModelConfig.dim)
        precondition(classifier.count == vocabSize * ModelConfig.dim)
        return [
            makeSpec(rmsFinal: rmsFinal, classifier: classifier, vocabSize: vocabSize, laneSpatial: laneSpatial),
        ]
    }

    private static func compileKernel(
        rmsFinal: borrowing TensorBuffer,
        classifier: borrowing TensorBuffer,
        vocabSize: Int,
        laneSpatial: Int
    ) throws(ANEError) -> ANEKernel {
        let spec = makeSpec(rmsFinal: rmsFinal, classifier: classifier, vocabSize: vocabSize, laneSpatial: laneSpatial)
        return try ANEKernel(
            milText: spec.milText,
            weights: spec.weights,
            inputSizes: spec.inputSizes,
            outputSizes: spec.outputSizes
        )
    }

    private static func makeSpec(
        rmsFinal: borrowing TensorBuffer,
        classifier: borrowing TensorBuffer,
        vocabSize: Int,
        laneSpatial: Int
    ) -> CompileSpec {
        let generator = GenerationRMSNormClassifierGenerator(vocabSize: vocabSize, laneSpatial: laneSpatial)
        return CompileSpec(
            kind: .rmsNormClassifier,
            milText: generator.milText,
            weights: [
                (
                    path: "@model_path/weights/rms_final.bin",
                    data: buildBlob(from: rmsFinal, rows: 1, cols: ModelConfig.dim)
                ),
                (
                    path: "@model_path/weights/classifier.bin",
                    data: buildBlob(from: classifier, rows: vocabSize, cols: ModelConfig.dim)
                ),
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
