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

    func test_decode_qkv_only_generator_gqa_output_byte_sizes_reflect_kvDim() {
        let dim = 2048
        let kvDim = 256  // nKVHeads=4, headDim=64
        let lane = 32
        let gen = DecodeQKVOnlyGenerator(dim: dim, kvDim: kvDim, laneSpatial: lane)

        // Outputs are alphabetical: kNew(kvDim), qOut(dim), vNew(kvDim)
        XCTAssertEqual(gen.outputByteSizes, [
            kvDim * lane * 2,
            dim * lane * 2,
            kvDim * lane * 2,
        ])
        // Input stays at dim
        XCTAssertEqual(gen.inputByteSizes, [dim * lane * 2])
    }

    func test_decode_qkv_only_generator_gqa_mil_has_correct_kv_weight_shapes() {
        let dim = 768
        let kvDim = 256
        let lane = 32
        let mil = DecodeQKVOnlyGenerator(dim: dim, kvDim: kvDim, laneSpatial: lane).milText

        // Q weight is [dim, dim] = [768, 768]
        XCTAssertTrue(mil.contains("tensor<fp16, [\(dim), \(dim), 1, 1]>"))
        // K/V weights are [kvDim, dim] = [256, 768]
        XCTAssertTrue(mil.contains("tensor<fp16, [\(kvDim), \(dim), 1, 1]>"))
    }

    func test_decode_qkv_only_generator_mha_kvDim_defaults_to_dim() {
        let dim = 768
        let lane = 32
        let gen = DecodeQKVOnlyGenerator(dim: dim, laneSpatial: lane)

        XCTAssertEqual(gen.kvDim, dim)
        XCTAssertEqual(gen.outputByteSizes, [
            dim * lane * 2,
            dim * lane * 2,
            dim * lane * 2,
        ])
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

    func test_decode_qkv_only_generator_rmsnorm_omits_qkv_bias_by_default() {
        let mil = DecodeQKVOnlyGenerator(laneSpatial: 32, architecture: .rmsNormSwiGLU).milText

        XCTAssertFalse(mil.contains("bq.bin"))
        XCTAssertFalse(mil.contains("bk.bin"))
        XCTAssertFalse(mil.contains("bv.bin"))
        XCTAssertFalse(mil.contains("rms1_beta.bin"))
    }

    /// Qwen2-family layers keep RMSNorm/SwiGLU but bias q/k/v. Dropping those biases is a
    /// silent accuracy bug, so pin the opt-in.
    func test_decode_qkv_only_generator_rmsnorm_emits_qkv_bias_when_requested() {
        let dim = 896
        let kvDim = 128
        let mil = DecodeQKVOnlyGenerator(
            dim: dim,
            qDim: dim,
            kvDim: kvDim,
            laneSpatial: 32,
            architecture: .rmsNormSwiGLU,
            normEps: 1e-6,
            hasQKVBias: true
        ).milText

        XCTAssertTrue(mil.contains("bq.bin"))
        XCTAssertTrue(mil.contains("bk.bin"))
        XCTAssertTrue(mil.contains("bv.bin"))
        // Bias is an add on the conv output, not a LayerNorm beta.
        XCTAssertFalse(mil.contains("rms1_beta.bin"))
        // Bias blobs are broadcast channel vectors added to the conv output.
        XCTAssertTrue(mil.contains("tensor<fp16, [1, \(dim), 1, 1]>"))
        XCTAssertTrue(mil.contains("tensor<fp16, [1, \(kvDim), 1, 1]>"))
        XCTAssertTrue(mil.contains("q_out"))
        XCTAssertTrue(mil.contains("k_out"))
        XCTAssertTrue(mil.contains("v_out"))
    }

    func test_decode_qkv_only_generator_gpt2_always_biases_qkv() {
        let generator = DecodeQKVOnlyGenerator(laneSpatial: 32, architecture: .gpt2, hasQKVBias: false)

        XCTAssertTrue(generator.hasQKVBias)
        XCTAssertTrue(generator.milText.contains("bq.bin"))
    }

    func test_decode_qkv_only_generator_qwen15b_widths_emit_gqa_and_qkv_bias() {
        let dim = 1536
        let kvDim = 256
        let lane = 32
        let gen = DecodeQKVOnlyGenerator(
            dim: dim,
            qDim: dim,
            kvDim: kvDim,
            laneSpatial: lane,
            architecture: .rmsNormSwiGLU,
            normEps: 1e-6,
            hasQKVBias: true
        )
        let mil = gen.milText

        XCTAssertEqual(gen.inputByteSizes, [dim * lane * 2])
        XCTAssertEqual(gen.outputByteSizes, [
            kvDim * lane * 2,
            dim * lane * 2,
            kvDim * lane * 2,
        ])
        XCTAssertTrue(mil.contains("tensor<fp16, [1, \(dim), 1, \(lane)]> x"))
        XCTAssertTrue(mil.contains("tensor<fp16, [\(dim), \(dim), 1, 1]>"))
        XCTAssertTrue(mil.contains("tensor<fp16, [\(kvDim), \(dim), 1, 1]>"))
        XCTAssertTrue(mil.contains("bq.bin"))
        XCTAssertTrue(mil.contains("bk.bin"))
        XCTAssertTrue(mil.contains("bv.bin"))
        XCTAssertFalse(mil.contains("wo.bin"))
        XCTAssertFalse(mil.contains("tanh("))
    }

    func test_decode_qkv_only_generator_uses_custom_norm_epsilon() {
        let mil = DecodeQKVOnlyGenerator(
            dim: 1024,
            qDim: 2048,
            kvDim: 1024,
            laneSpatial: 32,
            architecture: .rmsNormSwiGLU,
            normEps: 1e-6
        ).milText

        XCTAssertTrue(mil.contains("norm_scale"))
        XCTAssertFalse(mil.contains("0.00001"))
    }
}
