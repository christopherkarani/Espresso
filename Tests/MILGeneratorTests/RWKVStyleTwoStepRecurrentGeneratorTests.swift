import XCTest
import ANETypes
@testable import MILGenerator

final class RWKVStyleTwoStepRecurrentGeneratorTests: XCTestCase {
    func test_rwkv_style_two_step_recurrent_generator_io_byte_contracts_match_model_shapes() {
        let lane = 32
        let dim = ModelConfig.dim
        let generator = RWKVStyleTwoStepRecurrentGenerator(laneSpatial: lane)
        let bytes = dim * lane * 2

        XCTAssertEqual(generator.inputByteSizes, [bytes, bytes, bytes])
        XCTAssertEqual(generator.outputByteSizes, [bytes, bytes, bytes, bytes])
    }

    func test_rwkv_style_two_step_recurrent_generator_contains_expected_weight_blobs() {
        let mil = RWKVStyleTwoStepRecurrentGenerator(laneSpatial: 32).milText

        XCTAssertTrue(mil.contains("rwkv_rms.bin"))
        XCTAssertTrue(mil.contains("wx.bin"))
        XCTAssertTrue(mil.contains("ws.bin"))
        XCTAssertTrue(mil.contains("wd.bin"))
        XCTAssertTrue(mil.contains("wo.bin"))
    }

    func test_rwkv_style_two_step_recurrent_generator_has_three_inputs_and_four_outputs() {
        let mil = RWKVStyleTwoStepRecurrentGenerator(laneSpatial: 32).milText
        XCTAssertEqual(extractMILInputNames(mil), ["x0", "x1", "stateIn"])
        XCTAssertEqual(extractMILReturnTuple(mil), ["x0Next", "x1Next", "stateMid", "stateOut"])
    }

    func test_rwkv_style_two_step_recurrent_generator_stays_inside_proven_op_subset() {
        let mil = RWKVStyleTwoStepRecurrentGenerator(laneSpatial: 32).milText

        XCTAssertTrue(mil.contains("reduce_sum"))
        XCTAssertTrue(mil.contains("pow(x="))
        XCTAssertTrue(mil.contains("conv("))
        XCTAssertTrue(mil.contains("sigmoid("))
        XCTAssertFalse(mil.contains("softmax("))
        XCTAssertFalse(mil.contains("slice_by_index("))
    }

    func test_rwkv_style_two_step_recurrent_generator_has_unique_ssa_names() {
        let mil = RWKVStyleTwoStepRecurrentGenerator(laneSpatial: 32).milText

        var names: [String] = []
        var scanner = mil[mil.startIndex...]
        let namePrefix = "name=string(\""
        let nameSuffix = "\")"
        while let range = scanner.range(of: namePrefix) {
            let afterPrefix = range.upperBound
            if let endRange = scanner[afterPrefix...].range(of: nameSuffix) {
                names.append(String(scanner[afterPrefix..<endRange.lowerBound]))
                scanner = scanner[endRange.upperBound...]
            } else {
                break
            }
        }

        XCTAssertEqual(names.count, Set(names).count, "Duplicate SSA names found in two-step recurrent MIL")
    }
}
