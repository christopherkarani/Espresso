import Foundation
import ANETypes
import MILGenerator

public struct GenerationClassifierKernelSet: ~Copyable {
    public enum KernelKind: String, CaseIterable {
        case classifier
    }

    public struct CompileSpec {
        public let kind: KernelKind
        public let milText: String
        public let weights: [(path: String, data: Data)]
        public let inputSizes: [Int]
        public let outputSizes: [Int]
    }

    public let classifier: ANEKernel
    public let vocabSize: Int
    public let laneSpatial: Int

    private init(classifier: consuming ANEKernel, vocabSize: Int, laneSpatial: Int) {
        self.classifier = classifier
        self.vocabSize = vocabSize
        self.laneSpatial = laneSpatial
    }

    public init(
        classifier weights: borrowing TensorBuffer,
        vocabSize: Int,
        laneSpatial: Int = 1
    ) throws(ANEError) {
        guard vocabSize > 0 else {
            throw .invalidArguments("vocabSize must be > 0")
        }
        guard laneSpatial > 0 else {
            throw .invalidArguments("laneSpatial must be > 0")
        }
        guard weights.count == vocabSize * ModelConfig.dim else {
            throw .invalidArguments(
                "classifier weight count \(weights.count) does not match vocabSize \(vocabSize) * dim \(ModelConfig.dim)"
            )
        }
        let compiled = try Self.compileClassifier(classifier: weights, vocabSize: vocabSize, laneSpatial: laneSpatial)
        self.init(classifier: compiled, vocabSize: vocabSize, laneSpatial: laneSpatial)
    }

    public static func compileSpecs(
        classifier: borrowing TensorBuffer,
        vocabSize: Int,
        laneSpatial: Int = 1
    ) -> [CompileSpec] {
        precondition(vocabSize > 0)
        precondition(laneSpatial > 0)
        precondition(classifier.count == vocabSize * ModelConfig.dim)
        return [
            makeClassifierSpec(classifier: classifier, vocabSize: vocabSize, laneSpatial: laneSpatial),
        ]
    }

    private static func compileClassifier(
        classifier: borrowing TensorBuffer,
        vocabSize: Int,
        laneSpatial: Int
    ) throws(ANEError) -> ANEKernel {
        let spec = makeClassifierSpec(classifier: classifier, vocabSize: vocabSize, laneSpatial: laneSpatial)
        return try ANEKernel(
            milText: spec.milText,
            weights: spec.weights,
            inputSizes: spec.inputSizes,
            outputSizes: spec.outputSizes
        )
    }

    private static func makeClassifierSpec(
        classifier: borrowing TensorBuffer,
        vocabSize: Int,
        laneSpatial: Int
    ) -> CompileSpec {
        let generator = GenerationClassifierGenerator(vocabSize: vocabSize, laneSpatial: laneSpatial)
        return CompileSpec(
            kind: .classifier,
            milText: generator.milText,
            weights: [
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
