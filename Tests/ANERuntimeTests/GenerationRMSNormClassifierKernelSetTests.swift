import XCTest
import ANETypes
@testable import ANERuntime

private func makeGenerationRMSNormWeights(value: Float = 1.0) -> TensorBuffer {
    let weights = TensorBuffer(count: ModelConfig.dim, zeroed: false)
    weights.withUnsafeMutableBufferPointer { ptr in
        for idx in ptr.indices {
            ptr[idx] = value
        }
    }
    return weights
}

private func makeGenerationRMSNormClassifierWeights(value: Float = 0.01) -> TensorBuffer {
    let weights = TensorBuffer(count: ModelConfig.vocab * ModelConfig.dim, zeroed: false)
    weights.withUnsafeMutableBufferPointer { ptr in
        for idx in ptr.indices {
            ptr[idx] = value
        }
    }
    return weights
}

final class GenerationRMSNormClassifierKernelSetTests: XCTestCase {
    func test_compile_specs_expose_single_fused_rmsnorm_classifier_kernel() {
        let rms = makeGenerationRMSNormWeights()
        let classifier = makeGenerationRMSNormClassifierWeights()
        let specs = GenerationRMSNormClassifierKernelSet.compileSpecs(
            rmsFinal: rms,
            classifier: classifier,
            vocabSize: ModelConfig.vocab,
            laneSpatial: 32
        )

        XCTAssertEqual(specs.count, 1)
        XCTAssertEqual(specs[0].kind, .rmsNormClassifier)
        XCTAssertEqual(specs[0].inputSizes, [ModelConfig.dim * 32 * 2])
        XCTAssertEqual(specs[0].outputSizes, [ModelConfig.vocab * 32 * 2, 1 * 32 * 2])
        XCTAssertEqual(specs[0].weights.count, 2)
        XCTAssertTrue(specs[0].milText.contains("rms_final.bin"))
        XCTAssertTrue(specs[0].milText.contains("classifier.bin"))
    }
}
