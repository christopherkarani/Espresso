import XCTest
import ANETypes
@testable import MILGenerator

final class RWKVStyleRecurrentStepGeneratorTests: XCTestCase {
    func test_rwkv_style_recurrent_generator_io_byte_contracts_match_model_shapes() {
        let lane = 32
        let dim = ModelConfig.dim
        let gen = RWKVStyleRecurrentStepGenerator(laneSpatial: lane)
        let bytes = dim * lane * 2

        XCTAssertEqual(gen.inputBytes, bytes)
        XCTAssertEqual(gen.inputByteSizes, [bytes, bytes])
        XCTAssertEqual(gen.outputByteSizes, [bytes, bytes])
    }

    func test_rwkv_style_recurrent_generator_contains_expected_weight_blobs() {
        let mil = RWKVStyleRecurrentStepGenerator(laneSpatial: 32).milText

        XCTAssertTrue(mil.contains("rwkv_rms.bin"))
        XCTAssertTrue(mil.contains("wx.bin"))
        XCTAssertTrue(mil.contains("ws.bin"))
        XCTAssertTrue(mil.contains("wd.bin"))
        XCTAssertTrue(mil.contains("wo.bin"))
    }

    func test_rwkv_style_recurrent_generator_has_two_inputs_and_two_outputs() {
        let mil = RWKVStyleRecurrentStepGenerator(laneSpatial: 32).milText
        let dim = ModelConfig.dim

        XCTAssertTrue(
            mil.contains("func main<ios18>(tensor<fp16, [1, \(dim), 1, 32]> x, tensor<fp16, [1, \(dim), 1, 32]> stateIn)")
        )
        XCTAssertTrue(mil.contains("-> (xNext,stateOut);"))
    }

    func test_rwkv_style_recurrent_generator_stays_inside_proven_op_subset() {
        let mil = RWKVStyleRecurrentStepGenerator(laneSpatial: 32).milText

        XCTAssertTrue(mil.contains("reduce_sum"))
        XCTAssertTrue(mil.contains("pow(x="))
        XCTAssertTrue(mil.contains("conv("))
        XCTAssertTrue(mil.contains("sigmoid("))
        XCTAssertFalse(mil.contains("softmax("))
        XCTAssertFalse(mil.contains("slice_by_index("))
    }

    func test_rwkv_style_recurrent_generator_has_unique_ssa_names() {
        let mil = RWKVStyleRecurrentStepGenerator(laneSpatial: 32).milText

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

        XCTAssertEqual(names.count, Set(names).count, "Duplicate SSA names found in recurrent MIL")
    }
}
