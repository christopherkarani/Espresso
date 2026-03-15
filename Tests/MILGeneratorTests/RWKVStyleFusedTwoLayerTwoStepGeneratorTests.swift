import XCTest
import ANETypes
@testable import MILGenerator

final class RWKVStyleFusedTwoLayerTwoStepGeneratorTests: XCTestCase {
    func test_rwkv_style_fused_two_layer_two_step_generator_io_byte_contracts_match_model_shapes() {
        let lane = 32
        let dim = ModelConfig.dim
        let generator = RWKVStyleFusedTwoLayerTwoStepGenerator(laneSpatial: lane)
        let bytes = dim * lane * 2

        XCTAssertEqual(generator.inputByteSizes, [bytes, bytes, bytes, bytes])
        XCTAssertEqual(generator.outputByteSizes, [bytes, bytes, bytes, bytes, bytes, bytes])
    }

    func test_rwkv_style_fused_two_layer_two_step_generator_contains_all_layer_weight_blobs() {
        let mil = RWKVStyleFusedTwoLayerTwoStepGenerator(laneSpatial: 32).milText

        XCTAssertTrue(mil.contains("rwkv_rms0.bin"))
        XCTAssertTrue(mil.contains("wx0.bin"))
        XCTAssertTrue(mil.contains("wo0.bin"))
        XCTAssertTrue(mil.contains("rwkv_rms1.bin"))
        XCTAssertTrue(mil.contains("wx1.bin"))
        XCTAssertTrue(mil.contains("wo1.bin"))
    }

    func test_rwkv_style_fused_two_layer_two_step_generator_has_four_inputs_and_six_outputs() {
        let mil = RWKVStyleFusedTwoLayerTwoStepGenerator(laneSpatial: 32).milText

        XCTAssertEqual(extractMILInputNames(mil), ["x0", "x1", "stateIn0", "stateIn1"])
        XCTAssertEqual(extractMILReturnTuple(mil), ["x0Next", "x1Next", "stateMid0", "stateMid1", "stateOut0", "stateOut1"])
    }

    func test_rwkv_style_fused_two_layer_two_step_generator_stays_inside_proven_op_subset() {
        let mil = RWKVStyleFusedTwoLayerTwoStepGenerator(laneSpatial: 32).milText

        XCTAssertTrue(mil.contains("reduce_sum"))
        XCTAssertTrue(mil.contains("pow(x="))
        XCTAssertTrue(mil.contains("conv("))
        XCTAssertTrue(mil.contains("sigmoid("))
        XCTAssertFalse(mil.contains("softmax("))
        XCTAssertFalse(mil.contains("slice_by_index("))
    }
}
