import XCTest
import ANETypes
@testable import MILGenerator

final class RWKVStyleFusedTwoLayerStepGeneratorTests: XCTestCase {
    func test_rwkv_style_fused_two_layer_generator_io_byte_contracts_match_model_shapes() {
        let lane = 32
        let dim = ModelConfig.dim
        let gen = RWKVStyleFusedTwoLayerStepGenerator(laneSpatial: lane)
        let bytes = dim * lane * 2

        XCTAssertEqual(gen.inputByteSizes, [bytes, bytes, bytes])
        XCTAssertEqual(gen.outputByteSizes, [bytes, bytes, bytes])
    }

    func test_rwkv_style_fused_two_layer_generator_contains_both_layer_weight_blobs() {
        let mil = RWKVStyleFusedTwoLayerStepGenerator(laneSpatial: 32).milText

        XCTAssertTrue(mil.contains("rwkv_rms0.bin"))
        XCTAssertTrue(mil.contains("wx0.bin"))
        XCTAssertTrue(mil.contains("wo0.bin"))
        XCTAssertTrue(mil.contains("rwkv_rms1.bin"))
        XCTAssertTrue(mil.contains("wx1.bin"))
        XCTAssertTrue(mil.contains("wo1.bin"))
    }

    func test_rwkv_style_fused_two_layer_generator_has_three_inputs_and_three_outputs() {
        let dim = ModelConfig.dim
        let mil = RWKVStyleFusedTwoLayerStepGenerator(laneSpatial: 32).milText

        XCTAssertTrue(
            mil.contains(
                "func main<ios18>(tensor<fp16, [1, \(dim), 1, 32]> x, tensor<fp16, [1, \(dim), 1, 32]> stateIn0, tensor<fp16, [1, \(dim), 1, 32]> stateIn1)"
            )
        )
        XCTAssertTrue(mil.contains("-> (xNext,stateOut0,stateOut1);"))
    }

    func test_rwkv_style_fused_two_layer_generator_stays_inside_proven_op_subset() {
        let mil = RWKVStyleFusedTwoLayerStepGenerator(laneSpatial: 32).milText

        XCTAssertTrue(mil.contains("reduce_sum"))
        XCTAssertTrue(mil.contains("pow(x="))
        XCTAssertTrue(mil.contains("conv("))
        XCTAssertTrue(mil.contains("sigmoid("))
        XCTAssertFalse(mil.contains("softmax("))
        XCTAssertFalse(mil.contains("slice_by_index("))
    }
}
