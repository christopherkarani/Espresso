import XCTest
@testable import MILGenerator

final class LFM2ShortConvKernelTests: XCTestCase {
    func test_lfm2_short_conv_generator_io_byte_contracts_match_model_shapes() {
        let dim = 1024
        let laneSpatial = 32
        let kernelWidth = 3
        let generator = LFM2ShortConvStepGenerator(
            dim: dim,
            laneSpatial: laneSpatial,
            kernelWidth: kernelWidth
        )

        let xBytes = dim * laneSpatial * 2
        let stateBytes = dim * laneSpatial * 2

        XCTAssertEqual(generator.inputBytes, xBytes)
        XCTAssertEqual(generator.inputByteSizes, [xBytes, stateBytes])
        XCTAssertEqual(generator.outputByteSizes, [xBytes, stateBytes])
    }

    func test_lfm2_short_conv_generator_contains_expected_weight_blob() {
        let mil = LFM2ShortConvStepGenerator(
            dim: 1024,
            laneSpatial: 32,
            kernelWidth: 3
        ).milText

        XCTAssertTrue(mil.contains("@model_path/weights/lfm2_short_conv.bin"))
    }

    func test_lfm2_short_conv_generator_has_two_inputs_and_two_outputs() {
        let mil = LFM2ShortConvStepGenerator(
            dim: 1024,
            laneSpatial: 32,
            kernelWidth: 3
        ).milText

        XCTAssertEqual(extractMILInputNames(mil), ["x", "convStateIn"])
        XCTAssertEqual(extractMILReturnTuple(mil), ["xNext", "convStateOut"])
    }

    func test_lfm2_short_conv_generator_uses_concat_conv_and_slice() {
        let mil = LFM2ShortConvStepGenerator(
            dim: 1024,
            laneSpatial: 32,
            kernelWidth: 3
        ).milText

        XCTAssertTrue(mil.contains("concat(axis="))
        XCTAssertTrue(mil.contains("conv("))
        XCTAssertTrue(mil.contains("slice_by_size("))
        XCTAssertFalse(mil.contains("slice_by_index("))
        XCTAssertFalse(mil.contains("softmax("))
        XCTAssertFalse(mil.contains("matmul("))
    }

    func test_lfm2_short_conv_generator_emits_kernel_width_three_weight_shape() {
        let mil = LFM2ShortConvStepGenerator(
            dim: 1024,
            laneSpatial: 32,
            kernelWidth: 3
        ).milText

        XCTAssertTrue(mil.contains("tensor<fp16, [1024, 1, 1, 3]> lfm2_short_conv"))
    }

    func test_lfm2_short_conv_generator_has_unique_ssa_names() {
        let mil = LFM2ShortConvStepGenerator(
            dim: 1024,
            laneSpatial: 32,
            kernelWidth: 3
        ).milText

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

        XCTAssertEqual(names.count, Set(names).count, "Duplicate SSA names found in LFM2 short-conv MIL")
    }
}
