import XCTest
import ANETypes
@testable import MILGenerator

final class DecodeQKVOnlyGeneratorTests: XCTestCase {
    func test_decode_qkv_only_generator_io_byte_contracts_match_model_shapes() {
        let lane = 32
        let dim = ModelConfig.dim
        let gen = DecodeQKVOnlyGenerator(laneSpatial: lane)

        XCTAssertEqual(gen.inputBytes, dim * lane * 2)
        XCTAssertEqual(gen.inputByteSizes, [dim * lane * 2])
        XCTAssertEqual(
            gen.outputByteSizes,
            [dim * lane * 2, dim * lane * 2, dim * lane * 2]
        )
    }

    func test_decode_qkv_only_generator_contains_only_projection_weights() {
        let mil = DecodeQKVOnlyGenerator(laneSpatial: 32).milText

        XCTAssertTrue(mil.contains("rms1.bin"))
        XCTAssertTrue(mil.contains("wq.bin"))
        XCTAssertTrue(mil.contains("wk.bin"))
        XCTAssertTrue(mil.contains("wv.bin"))

        XCTAssertFalse(mil.contains("wo.bin"))
        XCTAssertFalse(mil.contains("w1.bin"))
        XCTAssertFalse(mil.contains("w2.bin"))
        XCTAssertFalse(mil.contains("w3.bin"))
    }

    func test_decode_qkv_only_generator_emits_q_k_v_without_cache_or_mask_inputs() {
        let mil = DecodeQKVOnlyGenerator(laneSpatial: 32).milText

        XCTAssertTrue(mil.contains("tensor<fp16, [1, \(ModelConfig.dim), 1, 32]> x"))
        XCTAssertFalse(mil.contains("kCache"))
        XCTAssertFalse(mil.contains("vCache"))
        XCTAssertFalse(mil.contains("maskCache"))
        XCTAssertFalse(mil.contains("slice_by_size"))
        XCTAssertTrue(mil.contains("qOut"))
        XCTAssertTrue(mil.contains("kNew"))
        XCTAssertTrue(mil.contains("vNew"))
    }

    func test_decode_qkv_only_generator_gpt2_uses_layernorm_beta_and_qkv_biases() {
        let mil = DecodeQKVOnlyGenerator(
            laneSpatial: 32,
            architecture: .gpt2
        ).milText

        XCTAssertTrue(mil.contains("rms1.bin"))
        XCTAssertTrue(mil.contains("rms1_beta.bin"))
        XCTAssertTrue(mil.contains("bq.bin"))
        XCTAssertTrue(mil.contains("bk.bin"))
        XCTAssertTrue(mil.contains("bv.bin"))
        XCTAssertFalse(mil.contains("wo.bin"))
    }
}
