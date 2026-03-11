import XCTest
import ANETypes
@testable import MILGenerator

final class RWKVStyleFusedThreeLayerTwoStepGeneratorTests: XCTestCase {
    func test_rwkv_style_fused_three_layer_two_step_generator_io_byte_contracts_match_model_shapes() {
        let lane = 32
        let dim = ModelConfig.dim
        let generator = RWKVStyleFusedThreeLayerTwoStepGenerator(laneSpatial: lane)
        let bytes = dim * lane * 2

        XCTAssertEqual(generator.inputByteSizes, [bytes, bytes, bytes, bytes, bytes])
        XCTAssertEqual(generator.outputByteSizes, [bytes, bytes, bytes, bytes, bytes, bytes, bytes, bytes])
    }

    func test_rwkv_style_fused_three_layer_two_step_generator_contains_all_layer_weight_blobs() {
        let mil = RWKVStyleFusedThreeLayerTwoStepGenerator(laneSpatial: 32).milText

        XCTAssertTrue(mil.contains("rwkv_rms0.bin"))
        XCTAssertTrue(mil.contains("wx0.bin"))
        XCTAssertTrue(mil.contains("wo0.bin"))
        XCTAssertTrue(mil.contains("rwkv_rms1.bin"))
        XCTAssertTrue(mil.contains("wx1.bin"))
        XCTAssertTrue(mil.contains("wo1.bin"))
        XCTAssertTrue(mil.contains("rwkv_rms2.bin"))
        XCTAssertTrue(mil.contains("wx2.bin"))
        XCTAssertTrue(mil.contains("wo2.bin"))
    }

    func test_rwkv_style_fused_three_layer_two_step_generator_has_five_inputs_and_eight_outputs() {
        let dim = ModelConfig.dim
        let mil = RWKVStyleFusedThreeLayerTwoStepGenerator(laneSpatial: 32).milText

        XCTAssertTrue(
            mil.contains(
                "func main<ios18>(tensor<fp16, [1, \(dim), 1, 32]> x0, tensor<fp16, [1, \(dim), 1, 32]> x1, tensor<fp16, [1, \(dim), 1, 32]> stateIn0, tensor<fp16, [1, \(dim), 1, 32]> stateIn1, tensor<fp16, [1, \(dim), 1, 32]> stateIn2)"
            )
        )
        XCTAssertTrue(mil.contains("-> (x0Next,x1Next,stateMid0,stateMid1,stateMid2,stateOut0,stateOut1,stateOut2);"))
    }

    func test_rwkv_style_fused_three_layer_two_step_generator_stays_inside_proven_op_subset() {
        let mil = RWKVStyleFusedThreeLayerTwoStepGenerator(laneSpatial: 32).milText

        XCTAssertTrue(mil.contains("reduce_sum"))
        XCTAssertTrue(mil.contains("pow(x="))
        XCTAssertTrue(mil.contains("conv("))
        XCTAssertTrue(mil.contains("sigmoid("))
        XCTAssertFalse(mil.contains("softmax("))
        XCTAssertFalse(mil.contains("slice_by_index("))
    }
}
