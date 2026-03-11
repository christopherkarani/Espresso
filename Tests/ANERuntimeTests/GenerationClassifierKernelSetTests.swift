import XCTest
import ANETypes
@testable import ANERuntime

private func makeGenerationClassifierWeights(value: Float = 0.01) -> TensorBuffer {
    let weights = TensorBuffer(count: ModelConfig.vocab * ModelConfig.dim, zeroed: false)
    weights.withUnsafeMutableBufferPointer { ptr in
        for idx in ptr.indices {
            ptr[idx] = value
        }
    }
    return weights
}

final class GenerationClassifierKernelSetTests: XCTestCase {
    func test_compile_specs_expose_single_classifier_kernel() {
        let classifier = makeGenerationClassifierWeights()
        let specs = GenerationClassifierKernelSet.compileSpecs(
            classifier: classifier,
            vocabSize: ModelConfig.vocab
        )

        XCTAssertEqual(specs.count, 1)
        XCTAssertEqual(specs[0].kind, .classifier)
        XCTAssertEqual(specs[0].inputSizes, [ModelConfig.dim * 2])
        XCTAssertEqual(specs[0].outputSizes, [ModelConfig.vocab * 2])
        XCTAssertEqual(specs[0].weights.count, 1)
        XCTAssertTrue(specs[0].milText.contains("classifier.bin"))
    }
}
