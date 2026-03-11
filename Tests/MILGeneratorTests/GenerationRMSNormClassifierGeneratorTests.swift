import XCTest
import ANETypes
@testable import MILGenerator

final class GenerationRMSNormClassifierGeneratorTests: XCTestCase {
    func test_generation_rmsnorm_classifier_generator_io_contract_matches_lane_packed_single_token_shapes() {
        let laneSpatial = 32
        let generator = GenerationRMSNormClassifierGenerator(
            vocabSize: ModelConfig.vocab,
            laneSpatial: laneSpatial
        )

        XCTAssertEqual(generator.inputBytes, ModelConfig.dim * laneSpatial * 2)
        XCTAssertEqual(generator.inputByteSizes, [ModelConfig.dim * laneSpatial * 2])
        XCTAssertEqual(generator.outputByteSizes, [ModelConfig.vocab * laneSpatial * 2])
    }

    func test_generation_rmsnorm_classifier_generator_contains_expected_rmsnorm_and_classifier_ops() {
        let laneSpatial = 32
        let mil = GenerationRMSNormClassifierGenerator(
            vocabSize: ModelConfig.vocab,
            laneSpatial: laneSpatial
        ).milText

        XCTAssertTrue(
            mil.contains("func main<ios18>(tensor<fp16, [1, \(ModelConfig.dim), 1, \(laneSpatial)]> x)")
        )
        XCTAssertTrue(mil.contains("@model_path/weights/rms_final.bin"))
        XCTAssertTrue(mil.contains("@model_path/weights/classifier.bin"))
        XCTAssertTrue(mil.contains("reduce_sum"))
        XCTAssertTrue(mil.contains("pow(x="))
        XCTAssertTrue(mil.contains("tensor<fp16, [1,\(ModelConfig.dim),1,\(laneSpatial)]> xn = mul("))
        XCTAssertTrue(mil.contains("tensor<fp16, [1, \(ModelConfig.vocab), 1, \(laneSpatial)]> logits = conv("))
        XCTAssertTrue(mil.contains("-> (logits);"))
    }

    func test_generation_rmsnorm_classifier_generator_stays_inside_proven_op_subset() {
        let mil = GenerationRMSNormClassifierGenerator(
            vocabSize: ModelConfig.vocab,
            laneSpatial: 32
        ).milText

        XCTAssertTrue(mil.contains("reduce_sum"))
        XCTAssertTrue(mil.contains("pow(x="))
        XCTAssertTrue(mil.contains("conv("))
        XCTAssertFalse(mil.contains("slice_by_index("))
        XCTAssertFalse(mil.contains("softmax("))
        XCTAssertFalse(mil.contains("matmul("))
    }
}
