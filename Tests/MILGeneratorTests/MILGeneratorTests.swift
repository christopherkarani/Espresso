import XCTest
@testable import MILGenerator
import ANETypes
import ANERuntime
import Darwin

final class MILGeneratorTests: XCTestCase {
    private func fixtureText(_ filename: String) throws -> String {
        let name = (filename as NSString).deletingPathExtension
        let ext = (filename as NSString).pathExtension
        let url =
            Bundle.module.url(forResource: name, withExtension: ext)
            ?? Bundle.module.url(forResource: name, withExtension: ext, subdirectory: "Fixtures")
        guard let url else {
            XCTFail("Fixture not found in test bundle: \(filename)")
            throw XCTSkip("Fixture missing: \(filename)")
        }
        return try String(contentsOf: url, encoding: .utf8)
    }

    private func assertTextMatchesFixture(
        actual: String,
        fixture filename: String,
        file: StaticString = #filePath,
        line: UInt = #line
    ) throws {
        let expected = try fixtureText(filename)
        guard actual != expected else { return }

        let a = Array(actual.utf8)
        let e = Array(expected.utf8)
        let n = min(a.count, e.count)
        var i = 0
        while i < n && a[i] == e[i] { i += 1 }

        let ctxBefore = 60
        let ctxAfter = 60
        let start = max(0, i - ctxBefore)
        let end = min(n, i + ctxAfter)

        let expectedCtx = String(decoding: e[start..<end], as: UTF8.self)
        let actualCtx = String(decoding: a[start..<end], as: UTF8.self)

        XCTFail(
            """
            MIL mismatch at byte \(i) (expectedLen=\(e.count) actualLen=\(a.count))
            expectedCtx: \(String(reflecting: expectedCtx))
            actualCtx:   \(String(reflecting: actualCtx))
            """,
            file: file,
            line: line
        )
    }

    private func fixtureData(_ filename: String) throws -> Data {
        let name = (filename as NSString).deletingPathExtension
        let ext = (filename as NSString).pathExtension
        let url =
            Bundle.module.url(forResource: name, withExtension: ext)
            ?? Bundle.module.url(forResource: name, withExtension: ext, subdirectory: "Fixtures")
        guard let url else {
            XCTFail("Fixture not found in test bundle: \(filename)")
            throw XCTSkip("Fixture missing: \(filename)")
        }
        return try Data(contentsOf: url)
    }

    func test_sdpa_fwd_text_matches_objc() throws {
        let gen = SDPAForwardGenerator()
        try assertTextMatchesFixture(actual: gen.milText, fixture: "sdpa_fwd_taps.mil")
    }

    func test_ffn_fwd_text_matches_objc() throws {
        let gen = FFNForwardGenerator()
        try assertMILSemanticParity(actual: gen.milText, fixture: "ffn_fwd_taps.mil")
    }

    func test_ffn_bwd_text_matches_objc() throws {
        let gen = FFNBackwardGenerator()
        try assertTextMatchesFixture(actual: gen.milText, fixture: "ffn_bwd.mil")
    }

    func test_sdpa_bwd1_text_matches_objc() throws {
        let gen = SDPABackward1Generator()
        try assertTextMatchesFixture(actual: gen.milText, fixture: "sdpa_bwd1.mil")
    }

    func test_sdpa_bwd2_text_matches_objc() throws {
        let gen = SDPABackward2Generator()
        try assertTextMatchesFixture(actual: gen.milText, fixture: "sdpa_bwd2.mil")
    }

    func test_qkvb_text_matches_objc() throws {
        let gen = QKVBackwardGenerator()
        try assertTextMatchesFixture(actual: gen.milText, fixture: "qkvb.mil")
    }

    func test_generic_matmul_text_matches_objc() throws {
        try assertTextMatchesFixture(actual: GenericMIL.matmul(inCh: 4, outCh: 6, spatial: 2), fixture: "matmul.mil")
    }

    func test_generic_conv_text_matches_objc() throws {
        try assertTextMatchesFixture(actual: GenericMIL.conv(inCh: 4, outCh: 6, spatial: 2), fixture: "conv.mil")
    }

    func test_fused_qkv_text_matches_objc() throws {
        try assertTextMatchesFixture(actual: GenericMIL.fusedQKV(dim: 4, spatial: 2), fixture: "fused_qkv.mil")
    }

    func test_fused_ffnup_text_matches_objc() throws {
        try assertTextMatchesFixture(actual: GenericMIL.fusedFFNUp(dim: 4, hiddenDim: 6, spatial: 2), fixture: "fused_ffn.mil")
    }

    func test_causal_mask_diagonal_zero_upper_neg65504() {
        let blob = CausalMask.blob(seqLen: ModelConfig.seqLen)
        XCTAssertEqual(blob.count, 128 + ModelConfig.seqLen * ModelConfig.seqLen * 2)

        let minFP16 = Float16(-65504).bitPattern
        blob.withUnsafeBytes { raw in
            let payload = raw.baseAddress!.advanced(by: 128).assumingMemoryBound(to: UInt16.self)
            for row in 0..<ModelConfig.seqLen {
                for col in 0..<ModelConfig.seqLen {
                    let v = payload[row * ModelConfig.seqLen + col]
                    if col <= row {
                        XCTAssertEqual(v, 0, "mask[\(row),\(col)] should be 0")
                    } else {
                        XCTAssertEqual(v, minFP16, "mask[\(row),\(col)] should be -65504 (fp16)")
                    }
                }
            }
        }
    }

    func test_causal_mask_blob_header_fields() {
        let seqLen = ModelConfig.seqLen
        let blob = CausalMask.blob(seqLen: seqLen)
        XCTAssertEqual(blob.count, 128 + seqLen * seqLen * 2)

        let bytes = [UInt8](blob)
        XCTAssertEqual(bytes[0], 0x01)
        XCTAssertEqual(bytes[4], 0x02)
        XCTAssertEqual(bytes[64], 0xEF)
        XCTAssertEqual(bytes[65], 0xBE)
        XCTAssertEqual(bytes[66], 0xAD)
        XCTAssertEqual(bytes[67], 0xDE)
        XCTAssertEqual(bytes[68], 0x01)

        let ws = UInt32(bytes[72]) | (UInt32(bytes[73]) << 8) | (UInt32(bytes[74]) << 16) | (UInt32(bytes[75]) << 24)
        XCTAssertEqual(ws, UInt32(seqLen * seqLen * 2))
        let off = UInt32(bytes[80]) | (UInt32(bytes[81]) << 8) | (UInt32(bytes[82]) << 16) | (UInt32(bytes[83]) << 24)
        XCTAssertEqual(off, 128)
    }

    func test_causal_mask_blob_cached_identity() {
        let a = CausalMask.blob(seqLen: ModelConfig.seqLen)
        let b = CausalMask.blob(seqLen: ModelConfig.seqLen)

        let ap = a.withUnsafeBytes { $0.baseAddress }
        let bp = b.withUnsafeBytes { $0.baseAddress }
        XCTAssertEqual(ap, bp)
    }

    func test_causal_mask_nondefault_seq_len_cached_identity() {
        let a = CausalMask.blob(seqLen: 13)
        let b = CausalMask.blob(seqLen: 13)
        let ap = a.withUnsafeBytes { $0.baseAddress }
        let bp = b.withUnsafeBytes { $0.baseAddress }
        XCTAssertEqual(ap, bp)
    }

    func test_causal_mask_nondefault_seq_len_layout() {
        let seqLen = 8
        let blob = CausalMask.blob(seqLen: seqLen)
        XCTAssertEqual(blob.count, 128 + seqLen * seqLen * 2)

        let bytes = [UInt8](blob)
        let ws = UInt32(bytes[72]) | (UInt32(bytes[73]) << 8) | (UInt32(bytes[74]) << 16) | (UInt32(bytes[75]) << 24)
        XCTAssertEqual(ws, UInt32(seqLen * seqLen * 2))
        let off = UInt32(bytes[80]) | (UInt32(bytes[81]) << 8) | (UInt32(bytes[82]) << 16) | (UInt32(bytes[83]) << 24)
        XCTAssertEqual(off, 128)

        blob.withUnsafeBytes { raw in
            let payload = raw.baseAddress!.advanced(by: 128).assumingMemoryBound(to: UInt16.self)
            let minFP16 = Float16(-65504).bitPattern
            XCTAssertEqual(payload[0 * seqLen + 0], 0)
            XCTAssertEqual(payload[0 * seqLen + 1], minFP16)
            XCTAssertEqual(payload[3 * seqLen + 2], 0)
            XCTAssertEqual(payload[3 * seqLen + 7], minFP16)
            XCTAssertEqual(payload[7 * seqLen + 7], 0)
        }
    }

    func test_fused_qkv_blob_matches_objc_fixture() throws {
        let dim = 4
        let expected = dim * dim
        let wq = (0..<expected).map { Float($0) + 0.125 }
        let wk = (0..<expected).map { Float($0) + 100.25 }
        let wv = (0..<expected).map { Float($0) - 50.5 }
        let blob = try GenericMIL.buildFusedQKVWeightBlob(wq: wq, wk: wk, wv: wv, dim: dim)
        let ref = try fixtureData("qkv_blob_ref.bin")
        XCTAssertEqual(blob, ref)
    }

    func test_fused_ffnup_blob_matches_objc_fixture() throws {
        let dim = 4
        let hiddenDim = 6
        let expected = hiddenDim * dim
        let w1 = (0..<expected).map { Float($0) + 0.5 }
        let w3 = (0..<expected).map { Float($0) + 400.75 }
        let blob = try GenericMIL.buildFusedFFNUpWeightBlob(w1: w1, w3: w3, hiddenDim: hiddenDim, dim: dim)
        let ref = try fixtureData("ffn_blob_ref.bin")
        XCTAssertEqual(blob, ref)
    }

    func test_fused_qkv_blob_offsets_correct() throws {
        let dim = 4
        let spatial = 2
        let w = Array(repeating: Float(1.0), count: dim * dim)

        let mil = GenericMIL.fusedQKV(dim: dim, spatial: spatial)
        let wsize = dim * dim * 2
        let cs = 64 + wsize
        XCTAssertTrue(mil.contains("offset = uint64(64)"))
        XCTAssertTrue(mil.contains("offset = uint64(\(64 + cs))"))
        XCTAssertTrue(mil.contains("offset = uint64(\(64 + 2 * cs))"))

        let blob = try GenericMIL.buildFusedQKVWeightBlob(wq: w, wk: w, wv: w, dim: dim)
        XCTAssertEqual(blob.count, 64 + 3 * cs)

        blob.withUnsafeBytes { raw in
            func u32(at offset: Int) -> UInt32 {
                raw.load(fromByteOffset: offset, as: UInt32.self).littleEndian
            }

            for chunkIndex in 0..<3 {
                let chunkStart = 64 + chunkIndex * cs
                XCTAssertEqual(raw.load(fromByteOffset: chunkStart + 0, as: UInt8.self), 0xEF)
                XCTAssertEqual(raw.load(fromByteOffset: chunkStart + 1, as: UInt8.self), 0xBE)
                XCTAssertEqual(raw.load(fromByteOffset: chunkStart + 2, as: UInt8.self), 0xAD)
                XCTAssertEqual(raw.load(fromByteOffset: chunkStart + 3, as: UInt8.self), 0xDE)
                XCTAssertEqual(raw.load(fromByteOffset: chunkStart + 4, as: UInt8.self), 0x01)
                // data_size at chunkStart + 8
                XCTAssertEqual(u32(at: chunkStart + 8), UInt32(wsize))
                // data_offset at chunkStart + 16
                XCTAssertEqual(u32(at: chunkStart + 16), UInt32(chunkStart + 64))
            }
        }
    }

    func test_fused_generator_byte_sizes_are_consistent() {
        let dim = 4
        let hidden = 6
        let spatial = 2
        let qkv = GenericMIL.fusedQKV(dim: dim, spatial: spatial)
        let qkvChunkSize = 64 + dim * dim * 2
        XCTAssertTrue(qkv.contains("offset = uint64(64)"))
        XCTAssertTrue(qkv.contains("offset = uint64(\(64 + qkvChunkSize))"))
        XCTAssertTrue(qkv.contains("offset = uint64(\(64 + 2 * qkvChunkSize))"))

        let ffn = GenericMIL.fusedFFNUp(dim: dim, hiddenDim: hidden, spatial: spatial)
        let ffnChunkSize = 64 + hidden * dim * 2
        XCTAssertTrue(ffn.contains("offset = uint64(64)"))
        XCTAssertTrue(ffn.contains("offset = uint64(\(64 + ffnChunkSize))"))
    }

    func test_fused_qkv_blob_chunk_payload_ordering() throws {
        let dim = 4
        let expected = dim * dim
        let wq = (0..<expected).map { Float($0) + 0.125 }
        let wk = (0..<expected).map { Float($0) + 100.25 }
        let wv = (0..<expected).map { Float($0) - 50.5 }

        let blob = try GenericMIL.buildFusedQKVWeightBlob(wq: wq, wk: wk, wv: wv, dim: dim)
        let wsize = expected * 2
        let cs = 64 + wsize

        blob.withUnsafeBytes { raw in
            func decode(_ chunkIndex: Int, _ elementIndex: Int) -> Float {
                let offset = 64 + chunkIndex * cs + 64 + elementIndex * MemoryLayout<UInt16>.stride
                let bits = raw.load(fromByteOffset: offset, as: UInt16.self).littleEndian
                return Float(Float16(bitPattern: bits))
            }

            XCTAssertEqual(decode(0, 0), wq[0], accuracy: 1e-3)
            XCTAssertEqual(decode(0, expected - 1), wq[expected - 1], accuracy: 1e-3)
            XCTAssertEqual(decode(1, 0), wk[0], accuracy: 1e-3)
            XCTAssertEqual(decode(1, expected - 1), wk[expected - 1], accuracy: 1e-3)
            XCTAssertEqual(decode(2, 0), wv[0], accuracy: 1e-3)
            XCTAssertEqual(decode(2, expected - 1), wv[expected - 1], accuracy: 1e-3)
        }
    }

    func test_fused_ffnup_blob_offsets_correct() throws {
        let dim = 4
        let hiddenDim = 6
        let spatial = 2
        let w = Array(repeating: Float(1.0), count: hiddenDim * dim)

        let mil = GenericMIL.fusedFFNUp(dim: dim, hiddenDim: hiddenDim, spatial: spatial)
        let wsize = hiddenDim * dim * 2
        let cs = 64 + wsize
        XCTAssertTrue(mil.contains("offset = uint64(64)"))
        XCTAssertTrue(mil.contains("offset = uint64(\(64 + cs))"))

        let blob = try GenericMIL.buildFusedFFNUpWeightBlob(w1: w, w3: w, hiddenDim: hiddenDim, dim: dim)
        XCTAssertEqual(blob.count, 64 + 2 * cs)

        let bytes = [UInt8](blob)
        XCTAssertEqual(bytes[0], 0x01)
        XCTAssertEqual(bytes[4], 0x02)

        for chunkIndex in 0..<2 {
            let chunkStart = 64 + chunkIndex * cs
            XCTAssertEqual(bytes[chunkStart + 0], 0xEF)
            XCTAssertEqual(bytes[chunkStart + 1], 0xBE)
            XCTAssertEqual(bytes[chunkStart + 2], 0xAD)
            XCTAssertEqual(bytes[chunkStart + 3], 0xDE)
            XCTAssertEqual(bytes[chunkStart + 4], 0x01)
        }

        blob.withUnsafeBytes { raw in
            func u32(at offset: Int) -> UInt32 {
                raw.load(fromByteOffset: offset, as: UInt32.self).littleEndian
            }

            for chunkIndex in 0..<2 {
                let chunkStart = 64 + chunkIndex * cs
                // data_size at chunkStart + 8
                XCTAssertEqual(u32(at: chunkStart + 8), UInt32(wsize))
                // data_offset at chunkStart + 16
                XCTAssertEqual(u32(at: chunkStart + 16), UInt32(chunkStart + 64))
            }
        }
    }

    func test_fused_ffnup_blob_chunk_payload_ordering() throws {
        let dim = 4
        let hiddenDim = 6
        let expected = hiddenDim * dim
        let w1 = (0..<expected).map { Float($0) + 0.5 }
        let w3 = (0..<expected).map { Float($0) + 400.75 }

        let blob = try GenericMIL.buildFusedFFNUpWeightBlob(w1: w1, w3: w3, hiddenDim: hiddenDim, dim: dim)
        let wsize = expected * 2
        let cs = 64 + wsize

        blob.withUnsafeBytes { raw in
            func decode(_ chunkIndex: Int, _ elementIndex: Int) -> Float {
                let offset = 64 + chunkIndex * cs + 64 + elementIndex * MemoryLayout<UInt16>.stride
                let bits = raw.load(fromByteOffset: offset, as: UInt16.self).littleEndian
                return Float(Float16(bitPattern: bits))
            }

            XCTAssertEqual(decode(0, 0), w1[0], accuracy: 1e-3)
            XCTAssertEqual(decode(0, expected - 1), w1[expected - 1], accuracy: 1e-3)
            XCTAssertEqual(decode(1, 0), w3[0], accuracy: 1e-3)
            XCTAssertEqual(decode(1, expected - 1), w3[expected - 1], accuracy: 1e-3)
        }
    }

    func test_fused_weight_blob_throws_on_invalid_counts() {
        XCTAssertThrowsError(try GenericMIL.buildFusedQKVWeightBlob(
            wq: [1, 2, 3],
            wk: [1, 2, 3, 4],
            wv: [1, 2, 3, 4],
            dim: 2
        )) { error in
            XCTAssertEqual(error as? GenericMILError, .invalidWeightCount(expected: 4, got: 3))
        }

        XCTAssertThrowsError(try GenericMIL.buildFusedFFNUpWeightBlob(
            w1: [Float](repeating: 1, count: 8),
            w3: [Float](repeating: 1, count: 7),
            hiddenDim: 2,
            dim: 4
        )) { error in
            XCTAssertEqual(error as? GenericMILError, .invalidWeightCount(expected: 8, got: 7))
        }
    }

    func test_fused_weight_blob_throws_on_overflow_shape() {
        XCTAssertThrowsError(
            try GenericMIL.buildFusedQKVWeightBlob(
                wq: [],
                wk: [],
                wv: [],
                dim: Int.max
            )
        ) { error in
            XCTAssertEqual(error as? GenericMILError, .sizeOverflow)
        }
    }

    func test_generic_fused_contracts_expose_multi_output_sizes() throws {
        let qkv = try GenericMIL.fusedQKVContract(dim: 4, spatial: 2)
        XCTAssertEqual(qkv.inputBytes, 4 * 2 * MemoryLayout<Float>.stride)
        XCTAssertEqual(qkv.outputByteSizes, [32, 32, 32])

        let ffn = try GenericMIL.fusedFFNUpContract(dim: 4, hiddenDim: 6, spatial: 2)
        XCTAssertEqual(ffn.inputBytes, 4 * 2 * MemoryLayout<Float>.stride)
        XCTAssertEqual(ffn.outputByteSizes, [48, 48])
    }

    func test_generator_io_byte_contracts_match_model_shapes() {
        let dim = ModelConfig.dim
        let hidden = ModelConfig.hidden
        let seq = ModelConfig.seqLen
        let scoreCh = ModelConfig.scoreCh

        XCTAssertEqual(SDPAForwardGenerator().inputBytes, dim * seq * 2)
        XCTAssertEqual(SDPAForwardGenerator().outputBytes, 6 * dim * seq * 2)

        XCTAssertEqual(FFNForwardGenerator().inputBytes, dim * seq * 2)
        XCTAssertEqual(FFNForwardGenerator().outputBytes, (2 * dim + 3 * hidden) * seq * 2)

        XCTAssertEqual(FFNBackwardGenerator().inputBytes, (dim + 2 * hidden) * seq * 2)
        XCTAssertEqual(FFNBackwardGenerator().outputBytes, (dim + 2 * hidden) * seq * 2)

        XCTAssertEqual(SDPABackward1Generator().inputBytes, 4 * dim * seq * 2)
        XCTAssertEqual(SDPABackward1Generator().outputBytes, (dim + 2 * scoreCh) * seq * 2)

        XCTAssertEqual(SDPABackward2Generator().inputBytes, (2 * scoreCh + 2 * dim) * seq * 2)
        XCTAssertEqual(SDPABackward2Generator().outputBytes, 2 * dim * seq * 2)

        XCTAssertEqual(QKVBackwardGenerator().inputBytes, 3 * dim * seq * 2)
        XCTAssertEqual(QKVBackwardGenerator().outputBytes, dim * seq * 2)

        let lane = DecodeKernelSet.defaultLaneSpatial
        let decodeAttn = DecodeAttentionQKVGenerator(maxSeq: 64, laneSpatial: lane)
        XCTAssertEqual(decodeAttn.inputBytes, dim * lane * 2)
        XCTAssertEqual(
            decodeAttn.inputByteSizes,
            [dim * lane * 2, dim * 64 * 2, dim * 64 * 2, dim * 64 * 2]
        )
        XCTAssertEqual(decodeAttn.outputByteSizes, [dim * lane * 2, dim * lane * 2, dim * lane * 2])

        XCTAssertEqual(DecodeFFNGenerator(laneSpatial: lane).inputBytes, dim * lane * 2)
        XCTAssertEqual(DecodeFFNGenerator(laneSpatial: lane).outputByteSizes, [dim * lane * 2])
        let decodeProjectionFFN = DecodeProjectionFFNGenerator(laneSpatial: lane)
        XCTAssertEqual(
            decodeProjectionFFN.inputByteSizes,
            [dim * lane * MemoryLayout<Float>.stride, dim * lane * 2]
        )
        XCTAssertEqual(decodeProjectionFFN.outputByteSizes, [dim * lane * 2])
    }

    func test_decode_attention_generator_contains_expected_decode_ops() {
        let gen = DecodeAttentionQKVGenerator(maxSeq: 32, laneSpatial: DecodeKernelSet.defaultLaneSpatial)
        let mil = gen.milText

        XCTAssertTrue(mil.contains("tensor<fp16, [1, \(ModelConfig.dim), 1, \(DecodeKernelSet.defaultLaneSpatial)]> x"))
        XCTAssertTrue(mil.contains("tensor<fp16, [1, \(ModelConfig.dim), 1, 32]> kCache"))
        XCTAssertTrue(mil.contains("tensor<fp16, [1, \(ModelConfig.dim), 1, 32]> vCache"))
        XCTAssertTrue(mil.contains("tensor<fp16, [1, \(ModelConfig.dim), 1, 32]> maskCache"))
        XCTAssertFalse(mil.contains("selfMask"))
        XCTAssertTrue(mil.contains("reduce_sum(x=qfFull,axes=raxSp,keep_dims=kd)"))
        XCTAssertTrue(mil.contains("k_ch"))
        XCTAssertTrue(mil.contains("mask0"))
        XCTAssertTrue(mil.contains("co_probe"))
        XCTAssertTrue(mil.contains("-> (x2,kfFull,vfFull);"))
        XCTAssertFalse(mil.contains("values=(x2,kfFull,vfFull)"))
    }

    func test_decode_ffn_generator_contains_expected_decode_ops() {
        let mil = DecodeFFNGenerator().milText
        XCTAssertEqual(extractMILInputNames(mil), ["x"])
        XCTAssertEqual(extractMILReturnTuple(mil), ["out"])
        XCTAssertTrue(mil.contains("sigmoid("))
        XCTAssertTrue(mil.contains("mul("))
        XCTAssertTrue(mil.contains("add("))
    }

    func test_decode_ffn_generator_gpt2_uses_layernorm_biases_and_gelu() {
        let mil = DecodeFFNGenerator(architecture: .gpt2).milText
        XCTAssertEqual(extractMILInputNames(mil), ["x"])
        XCTAssertEqual(extractMILReturnTuple(mil), ["out"])
        XCTAssertTrue(mil.contains("rms2.bin"))
        XCTAssertTrue(mil.contains("rms2_beta.bin"))
        XCTAssertTrue(mil.contains("b1.bin"))
        XCTAssertTrue(mil.contains("b2.bin"))
        XCTAssertTrue(mil.contains("tanh("))
        XCTAssertFalse(mil.contains("w3.bin"))
    }

    func test_decode_projection_ffn_generator_gpt2_fuses_projection_layernorm_and_ffn() {
        let mil = DecodeProjectionFFNGenerator(architecture: .gpt2).milText
        XCTAssertEqual(extractMILInputNames(mil), ["context", "residual"])
        XCTAssertEqual(extractMILReturnTuple(mil), ["out"])
        XCTAssertTrue(mil.contains("context16"))
        XCTAssertTrue(mil.contains("wo.bin"))
        XCTAssertTrue(mil.contains("bo.bin"))
        XCTAssertTrue(mil.contains("rms2.bin"))
        XCTAssertTrue(mil.contains("rms2_beta.bin"))
        XCTAssertTrue(mil.contains("b1.bin"))
        XCTAssertTrue(mil.contains("b2.bin"))
        XCTAssertTrue(mil.contains("tanh("))
        XCTAssertFalse(mil.contains("w3.bin"))
    }

    func test_mil_builder_append_fp16_uses_fixed_posix_format() {
        var b = MILBuilder()
        b.appendFP16(0.125)
        XCTAssertEqual(b.text, "0.125000")
    }

    func test_string_format_calls_are_locale_explicit() throws {
        let repoRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let milDir = repoRoot.appendingPathComponent("Sources/MILGenerator", isDirectory: true)
        let files = try FileManager.default.contentsOfDirectory(at: milDir, includingPropertiesForKeys: nil)
            .filter { $0.pathExtension == "swift" }

        var missingLocale: [String] = []
        for file in files {
            let text = try String(contentsOf: file, encoding: .utf8)
            for (idx, line) in text.split(separator: "\n", omittingEmptySubsequences: false).enumerated() {
                let l = String(line)
                if l.contains("String(format:") && !l.contains("locale:") {
                    missingLocale.append("\(file.lastPathComponent):\(idx + 1)")
                }
            }
        }

        XCTAssertEqual(missingLocale, [], "Missing locale in String(format:) calls: \(missingLocale.joined(separator: ", "))")
    }

    func test_locale_does_not_affect_mil_formatting() throws {
        let baselineByFixture: [(String, String)] = [
            (SDPAForwardGenerator().milText, "sdpa_fwd_taps.mil"),
            (FFNForwardGenerator().milText, "ffn_fwd_taps.mil"),
            (FFNBackwardGenerator().milText, "ffn_bwd.mil"),
            (SDPABackward1Generator().milText, "sdpa_bwd1.mil"),
            (SDPABackward2Generator().milText, "sdpa_bwd2.mil"),
            (QKVBackwardGenerator().milText, "qkvb.mil"),
        ]
        let oldLocale = String(cString: setlocale(LC_ALL, nil))
        let oldLCAll = getenv("LC_ALL").map { String(cString: $0) }
        let oldLang = getenv("LANG").map { String(cString: $0) }
        defer {
            _ = setlocale(LC_ALL, oldLocale)
            if let oldLCAll {
                setenv("LC_ALL", oldLCAll, 1)
            } else {
                unsetenv("LC_ALL")
            }
            if let oldLang {
                setenv("LANG", oldLang, 1)
            } else {
                unsetenv("LANG")
            }
        }

        setenv("LC_ALL", "de_DE.UTF-8", 1)
        setenv("LANG", "de_DE.UTF-8", 1)
        guard setlocale(LC_ALL, "") != nil else {
            throw XCTSkip("de_DE.UTF-8 locale not available on this system")
        }

        let localizedByFixture: [(String, String)] = [
            (SDPAForwardGenerator().milText, "sdpa_fwd_taps.mil"),
            (FFNForwardGenerator().milText, "ffn_fwd_taps.mil"),
            (FFNBackwardGenerator().milText, "ffn_bwd.mil"),
            (SDPABackward1Generator().milText, "sdpa_bwd1.mil"),
            (SDPABackward2Generator().milText, "sdpa_bwd2.mil"),
            (QKVBackwardGenerator().milText, "qkvb.mil"),
        ]

        for ((baseline, fixture), (localized, localizedFixture)) in zip(baselineByFixture, localizedByFixture) {
            XCTAssertEqual(fixture, localizedFixture)
            XCTAssertEqual(localized, baseline, "Locale changed MIL output for fixture \(fixture)")
        }
    }
}
