import Foundation
import ANETypes
import MILGenerator

public struct RWKVStyleFusedThreeLayerRMSNormClassifierKernelSet: ~Copyable {
    public let fusedRMSNormClassifier: ANEKernel
    public let vocabSize: Int
    public let laneSpatial: Int

    public struct CompileSpec {
        public let milText: String
        public let weights: [(path: String, data: Data)]
        public let inputSizes: [Int]
        public let outputSizes: [Int]
    }

    private init(
        fusedRMSNormClassifier: consuming ANEKernel,
        vocabSize: Int,
        laneSpatial: Int
    ) {
        self.fusedRMSNormClassifier = fusedRMSNormClassifier
        self.vocabSize = vocabSize
        self.laneSpatial = laneSpatial
    }

    public init(
        weights0: borrowing RWKVStyleRecurrentWeights,
        weights1: borrowing RWKVStyleRecurrentWeights,
        weights2: borrowing RWKVStyleRecurrentWeights,
        rmsFinal: borrowing TensorBuffer,
        classifier: borrowing TensorBuffer,
        vocabSize: Int,
        laneSpatial: Int = 32
    ) throws(ANEError) {
        let spec = Self.compileSpecs(
            weights0: weights0,
            weights1: weights1,
            weights2: weights2,
            rmsFinal: rmsFinal,
            classifier: classifier,
            vocabSize: vocabSize,
            laneSpatial: laneSpatial
        )[0]

        let kernel = try ANEKernel(
            milText: spec.milText,
            weights: spec.weights,
            inputSizes: spec.inputSizes,
            outputSizes: spec.outputSizes
        )
        self.init(
            fusedRMSNormClassifier: kernel,
            vocabSize: vocabSize,
            laneSpatial: laneSpatial
        )
    }

    public static func compileSpecs(
        weights0: borrowing RWKVStyleRecurrentWeights,
        weights1: borrowing RWKVStyleRecurrentWeights,
        weights2: borrowing RWKVStyleRecurrentWeights,
        rmsFinal: borrowing TensorBuffer,
        classifier: borrowing TensorBuffer,
        vocabSize: Int,
        laneSpatial: Int = 32
    ) -> [CompileSpec] {
        let generator = RWKVStyleFusedThreeLayerRMSNormClassifierGenerator(
            vocabSize: vocabSize,
            laneSpatial: laneSpatial
        )
        let trunkSpec = RWKVStyleFusedThreeLayerKernelSet.compileSpecs(
            weights0: weights0,
            weights1: weights1,
            weights2: weights2,
            laneSpatial: laneSpatial
        )[0]
        let headSpec = GenerationRMSNormClassifierKernelSet.compileSpecs(
            rmsFinal: rmsFinal,
            classifier: classifier,
            vocabSize: vocabSize,
            laneSpatial: laneSpatial
        )[0]

        return [
            CompileSpec(
                milText: generator.milText,
                weights: trunkSpec.weights + headSpec.weights,
                inputSizes: generator.inputByteSizes,
                outputSizes: generator.outputByteSizes
            )
        ]
    }
}
