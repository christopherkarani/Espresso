import XCTest
import ANETypes
@testable import MILGenerator

final class GenerationClassifierGeneratorTests: XCTestCase {
    func test_generation_classifier_generator_io_contract_matches_single_token_shapes() {
        let generator = GenerationClassifierGenerator(vocabSize: ModelConfig.vocab)

        XCTAssertEqual(generator.inputBytes, ModelConfig.dim * 2)
        XCTAssertEqual(generator.inputByteSizes, [ModelConfig.dim * 2])
        XCTAssertEqual(generator.outputByteSizes, [ModelConfig.vocab * 2])
    }

    func test_generation_classifier_generator_contains_expected_weight_blob_and_conv() {
        let mil = GenerationClassifierGenerator(vocabSize: ModelConfig.vocab).milText

        XCTAssertTrue(
            mil.contains("func main<ios18>(tensor<fp16, [1, \(ModelConfig.dim), 1, 1]> x)")
        )
        XCTAssertTrue(mil.contains("@model_path/weights/classifier.bin"))
        XCTAssertTrue(mil.contains("tensor<fp16, [\(ModelConfig.vocab), \(ModelConfig.dim), 1, 1]> Wcls"))
        XCTAssertTrue(mil.contains("tensor<fp16, [1, \(ModelConfig.vocab), 1, 1]> logits = conv("))
        XCTAssertTrue(mil.contains("-> (logits);"))
    }

    func test_generation_classifier_generator_stays_inside_proven_conv_subset() {
        let mil = GenerationClassifierGenerator(vocabSize: ModelConfig.vocab).milText

        XCTAssertTrue(mil.contains("conv("))
        XCTAssertFalse(mil.contains("slice_by_index("))
        XCTAssertFalse(mil.contains("softmax("))
        XCTAssertFalse(mil.contains("matmul("))
    }
}
