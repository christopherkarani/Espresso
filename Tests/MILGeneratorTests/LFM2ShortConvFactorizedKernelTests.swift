import XCTest
@testable import MILGenerator

final class LFM2ShortConvFactorizedKernelTests: XCTestCase {
    func test_lfm2_factorized_short_conv_generator_io_byte_contracts_match_model_shapes() {
        let dim = 1024
        let laneSpatial = 32
        let kernelWidth = 3
        let generator = LFM2ShortConvFactorizedStepGenerator(
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

    func test_lfm2_factorized_short_conv_generator_contains_expected_weight_blobs() {
        let mil = LFM2ShortConvFactorizedStepGenerator(
            dim: 1024,
            laneSpatial: 32,
            kernelWidth: 3
        ).milText

        XCTAssertTrue(mil.contains("@model_path/weights/lfm2_short_conv_tap0.bin"))
        XCTAssertTrue(mil.contains("@model_path/weights/lfm2_short_conv_tap1.bin"))
        XCTAssertTrue(mil.contains("@model_path/weights/lfm2_short_conv_tap2.bin"))
    }

    func test_lfm2_factorized_short_conv_generator_has_two_inputs_and_two_outputs() {
        let mil = LFM2ShortConvFactorizedStepGenerator(
            dim: 1024,
            laneSpatial: 32,
            kernelWidth: 3
        ).milText

        XCTAssertEqual(extractMILInputNames(mil), ["x", "convStateIn"])
        XCTAssertEqual(extractMILReturnTuple(mil), ["xNext", "convStateOut"])
    }

    func test_lfm2_factorized_short_conv_generator_uses_concat_slice_conv_and_add() {
        let mil = LFM2ShortConvFactorizedStepGenerator(
            dim: 1024,
            laneSpatial: 32,
            kernelWidth: 3
        ).milText

        XCTAssertTrue(mil.contains("concat(axis="))
        XCTAssertTrue(mil.contains("slice_by_size("))
        XCTAssertFalse(mil.contains("slice_by_index("))
        XCTAssertTrue(mil.contains("conv("))
        XCTAssertTrue(mil.contains("add(x="))
        XCTAssertFalse(mil.contains("softmax("))
        XCTAssertFalse(mil.contains("matmul("))
    }

    func test_lfm2_factorized_short_conv_generator_uses_1x1_tap_weights() {
        let mil = LFM2ShortConvFactorizedStepGenerator(
            dim: 1024,
            laneSpatial: 32,
            kernelWidth: 3
        ).milText

        XCTAssertTrue(mil.contains("tensor<fp16, [1024, 1, 1, 1]> lfm2_short_conv_tap0"))
        XCTAssertTrue(mil.contains("tensor<fp16, [1024, 1, 1, 1]> lfm2_short_conv_tap1"))
        XCTAssertTrue(mil.contains("tensor<fp16, [1024, 1, 1, 1]> lfm2_short_conv_tap2"))
    }
}
