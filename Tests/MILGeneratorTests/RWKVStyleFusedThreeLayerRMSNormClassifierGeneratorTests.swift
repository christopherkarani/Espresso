import XCTest
import ANETypes
@testable import MILGenerator

final class RWKVStyleFusedThreeLayerRMSNormClassifierGeneratorTests: XCTestCase {
    func test_rwkv_style_fused_three_layer_rmsnorm_classifier_generator_io_contract() {
        let lane = 32
        let dim = ModelConfig.dim
        let vocab = ModelConfig.vocab
        let generator = RWKVStyleFusedThreeLayerRMSNormClassifierGenerator(vocabSize: vocab, laneSpatial: lane)
        let stateBytes = dim * lane * 2

        XCTAssertEqual(generator.inputByteSizes, [stateBytes, stateBytes, stateBytes, stateBytes])
        XCTAssertEqual(
            generator.outputByteSizes,
            [stateBytes, stateBytes, stateBytes, stateBytes, vocab * lane * 2, 1 * lane * 2]
        )
    }

    func test_rwkv_style_fused_three_layer_rmsnorm_classifier_generator_signature() {
        let mil = RWKVStyleFusedThreeLayerRMSNormClassifierGenerator(
            vocabSize: ModelConfig.vocab,
            laneSpatial: 32
        ).milText

        XCTAssertEqual(extractMILInputNames(mil), ["x", "stateIn0", "stateIn1", "stateIn2"])
        XCTAssertEqual(extractMILReturnTuple(mil), ["xNext", "stateOut0", "stateOut1", "stateOut2", "logits", "maxVal"])
    }

    func test_rwkv_style_fused_three_layer_rmsnorm_classifier_generator_contains_expected_weight_blobs_and_ops() {
        let mil = RWKVStyleFusedThreeLayerRMSNormClassifierGenerator(
            vocabSize: ModelConfig.vocab,
            laneSpatial: 32
        ).milText

        XCTAssertTrue(mil.contains("@model_path/weights/rwkv_rms0.bin"))
        XCTAssertTrue(mil.contains("@model_path/weights/wo2.bin"))
        XCTAssertTrue(mil.contains("@model_path/weights/rms_final.bin"))
        XCTAssertTrue(mil.contains("@model_path/weights/classifier.bin"))
        XCTAssertTrue(mil.contains("conv("))
    }
}
