import XCTest
import IOSurface
@testable import ANEInterop
@testable import ANETypes

private func makeSurface(bytes: Int) -> IOSurfaceRef {
    ane_interop_create_surface(bytes)!
}

private func readStoriesConfigHeader() throws -> String {
    let repoRoot = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .deletingLastPathComponent()
    let candidates = [
        repoRoot.appendingPathComponent("training/stories_config.h"),
        repoRoot.appendingPathComponent("archive/training/stories_config.h"),
    ]
    for headerURL in candidates where FileManager.default.fileExists(atPath: headerURL.path) {
        return try String(contentsOf: headerURL, encoding: .utf8)
    }
    throw NSError(
        domain: "ANETypesTests",
        code: 1,
        userInfo: [NSLocalizedDescriptionKey: "stories_config.h not found in training/ or archive/training/"]
    )
}

private func parseCDefineInt(_ name: String, in text: String) -> Int? {
    let pattern = #"^\s*#define\s+\#(name)\s+([0-9_]+)\b"#
    guard let regex = try? NSRegularExpression(pattern: pattern, options: [.anchorsMatchLines]) else {
        return nil
    }
    let range = NSRange(text.startIndex..<text.endIndex, in: text)
    guard let match = regex.firstMatch(in: text, options: [], range: range),
          match.numberOfRanges == 2,
          let valueRange = Range(match.range(at: 1), in: text) else {
        return nil
    }
    let token = text[valueRange].replacingOccurrences(of: "_", with: "")
    return Int(token)
}

final class ANETypesTests: XCTestCase {
    func test_tensor_buffer_alignment() {
        let buf = TensorBuffer(count: 1, zeroed: false)
        buf.withUnsafePointer { ptr in
            let addr = UInt(bitPattern: ptr)
            XCTAssertEqual(addr % UInt(TensorBuffer.allocationAlignment), 0)
        }
    }

    func test_tensor_buffer_pointer_roundtrip_stable() {
        let buf = TensorBuffer(count: 8, zeroed: true)
        for i in 0..<1_000 {
            buf.withUnsafeMutablePointer { ptr in
                ptr[3] = Float(i)
            }
            buf.withUnsafePointer { ptr in
                XCTAssertEqual(ptr[3], Float(i), accuracy: 0)
            }
        }
    }

    func test_layer_weights_alloc_dealloc_no_leak() {
        for _ in 0..<100 {
            _ = LayerWeights()
        }
    }

    func test_adam_state_initialized_to_zero() {
        let s = AdamState(count: 1024)
        s.m.withUnsafeBufferPointer { m in
            XCTAssertTrue(m.allSatisfy { $0 == 0 })
        }
        s.v.withUnsafeBufferPointer { v in
            XCTAssertTrue(v.allSatisfy { $0 == 0 })
        }
    }

    func test_layer_storage_throwing_initializer_cleans_up_partial_init() {
        final class CounterBox {
            var created = 0
            var destroyed = 0
        }

        struct Tracked: ~Copyable {
            let box: CounterBox
            init(box: CounterBox) {
                self.box = box
                box.created += 1
            }
            deinit {
                box.destroyed += 1
            }
        }

        enum TestError: Error {
            case boom
        }

        let box = CounterBox()
        do {
            _ = try LayerStorage<Tracked>(count: 5, throwingInitializer: { i in
                if i == 3 { throw TestError.boom }
                return Tracked(box: box)
            })
            XCTFail("Expected initializer to throw")
        } catch TestError.boom {
            XCTAssertEqual(box.created, 3)
            XCTAssertEqual(box.destroyed, 3)
        } catch {
            XCTFail("Unexpected error: \(error)")
        }
    }

    func test_build_blob_header_magic() {
        let w: [Float] = [1, 2, 3, 4]
        let blob = WeightBlob.build(from: w, rows: 1, cols: 4)

        XCTAssertEqual(blob.count, 128 + 8)
        let bytes = [UInt8](blob)

        XCTAssertEqual(bytes[0], 0x01)
        XCTAssertEqual(bytes[4], 0x02)
        XCTAssertEqual(bytes[64], 0xEF)
        XCTAssertEqual(bytes[65], 0xBE)
        XCTAssertEqual(bytes[66], 0xAD)
        XCTAssertEqual(bytes[67], 0xDE)
        XCTAssertEqual(bytes[68], 0x01)

        let ws = UInt32(bytes[72]) | (UInt32(bytes[73]) << 8) | (UInt32(bytes[74]) << 16) | (UInt32(bytes[75]) << 24)
        XCTAssertEqual(ws, 8)
        let off = UInt32(bytes[80]) | (UInt32(bytes[81]) << 8) | (UInt32(bytes[82]) << 16) | (UInt32(bytes[83]) << 24)
        XCTAssertEqual(off, 128)
    }

    func test_build_blob_zero_shape_header_only() {
        let blob = WeightBlob.build(from: [Float](), rows: 0, cols: 0)
        XCTAssertEqual(blob.count, 128)
        let bytes = [UInt8](blob)
        XCTAssertEqual(bytes[0], 0x01)
        XCTAssertEqual(bytes[4], 0x02)
        XCTAssertEqual(bytes[64], 0xEF)
        XCTAssertEqual(bytes[65], 0xBE)
        XCTAssertEqual(bytes[66], 0xAD)
        XCTAssertEqual(bytes[67], 0xDE)
        XCTAssertEqual(bytes[68], 0x01)

        let ws = UInt32(bytes[72]) | (UInt32(bytes[73]) << 8) | (UInt32(bytes[74]) << 16) | (UInt32(bytes[75]) << 24)
        XCTAssertEqual(ws, 0)
        let off = UInt32(bytes[80]) | (UInt32(bytes[81]) << 8) | (UInt32(bytes[82]) << 16) | (UInt32(bytes[83]) << 24)
        XCTAssertEqual(off, 128)
    }

    func test_build_blob_empty_unsafe_buffer_nil_base_does_not_crash() {
        let empty = UnsafeBufferPointer<Float>(start: nil, count: 0)
        let blob = WeightBlob.build(from: empty, rows: 0, cols: 0)
        XCTAssertEqual(blob.count, 128)
    }

    func test_build_blob_fp16_accuracy() {
        let w: [Float] = [0, 1, -1, 3.5, 123.75, -0.25]
        let blob = WeightBlob.build(from: w, rows: 2, cols: 3)

        let payload = blob.dropFirst(128)
        XCTAssertEqual(payload.count, w.count * 2)
        let u16 = payload.withUnsafeBytes { raw in
            Array(raw.bindMemory(to: UInt16.self))
        }

        var decoded = [Float]()
        decoded.reserveCapacity(w.count)
        for bits in u16 {
            decoded.append(Float(Float16(bitPattern: bits)))
        }

        var maxAbsDiff: Float = 0
        for i in 0..<w.count {
            maxAbsDiff = max(maxAbsDiff, abs(decoded[i] - w[i]))
        }
        XCTAssertLessThanOrEqual(maxAbsDiff, 1e-2)
    }

    func test_build_blob_transposed_layout() {
        // 3x4 matrix with unique values.
        let rows = 3
        let cols = 4
        let w = (0..<(rows * cols)).map { Float($0) }
        let blob = WeightBlob.buildTransposed(from: w, rows: rows, cols: cols)

        let payload = blob.dropFirst(128)
        let u16 = payload.withUnsafeBytes { raw in
            Array(raw.bindMemory(to: UInt16.self))
        }

        func f(_ i: Int) -> Float { Float(Float16(bitPattern: u16[i])) }

        // Transposed layout: payload[j*rows + i] = w[i*cols + j]
        for i in 0..<rows {
            for j in 0..<cols {
                XCTAssertEqual(f(j * rows + i), w[i * cols + j], accuracy: 0)
            }
        }
    }

    func test_build_blob_transposed_zero_shape_header_only() {
        let blob = WeightBlob.buildTransposed(from: [Float](), rows: 0, cols: 0)
        XCTAssertEqual(blob.count, 128)
        let bytes = [UInt8](blob)
        XCTAssertEqual(bytes[0], 0x01)
        XCTAssertEqual(bytes[4], 0x02)
        XCTAssertEqual(bytes[64], 0xEF)
        XCTAssertEqual(bytes[65], 0xBE)
        XCTAssertEqual(bytes[66], 0xAD)
        XCTAssertEqual(bytes[67], 0xDE)
        XCTAssertEqual(bytes[68], 0x01)
        let ws = UInt32(bytes[72]) | (UInt32(bytes[73]) << 8) | (UInt32(bytes[74]) << 16) | (UInt32(bytes[75]) << 24)
        XCTAssertEqual(ws, 0)
        let off = UInt32(bytes[80]) | (UInt32(bytes[81]) << 8) | (UInt32(bytes[82]) << 16) | (UInt32(bytes[83]) << 24)
        XCTAssertEqual(off, 128)
    }

    func test_build_blob_fp16_payload_passthrough_exact_bits() {
        let payload: [UInt16] = [0x0000, 0x3C00, 0xC000, 0x7BFF, 0x3555]
        let blob = WeightBlob.buildFP16(from: payload)
        XCTAssertEqual(blob.count, 128 + payload.count * 2)

        blob.withUnsafeBytes { raw in
            let out = raw.baseAddress!.advanced(by: 128).assumingMemoryBound(to: UInt16.self)
            for i in 0..<payload.count {
                XCTAssertEqual(out[i], payload[i], "bit mismatch at index \(i)")
            }
        }
    }

    func test_surface_write_read_roundtrip() {
        let channels = 64
        let spatial = 32
        let count = channels * spatial
        let s = makeSurface(bytes: count * 2)

        let input = (0..<count).map { _ in Float.random(in: -10...10) }
        var output = Array(repeating: Float.nan, count: count)

        input.withUnsafeBufferPointer { inputBuf in
            SurfaceIO.writeFP16(to: s, data: inputBuf, channels: channels, spatial: spatial)
        }
        output.withUnsafeMutableBufferPointer { outBuf in
            SurfaceIO.readFP16(from: s, into: outBuf, channelOffset: 0, channels: channels, spatial: spatial)
        }

        var maxAbsDiff: Float = 0
        for i in 0..<count {
            maxAbsDiff = max(maxAbsDiff, abs(output[i] - input[i]))
        }
        XCTAssertLessThanOrEqual(maxAbsDiff, 1e-2)
    }

    func test_surface_read_with_channel_offset() {
        let totalChannels = 16
        let spatial = 4
        let totalCount = totalChannels * spatial
        let s = makeSurface(bytes: totalCount * 2)

        let input = (0..<totalCount).map { Float($0) * 0.1 - 2.0 }
        input.withUnsafeBufferPointer { inputBuf in
            SurfaceIO.writeFP16(to: s, data: inputBuf, channels: totalChannels, spatial: spatial)
        }

        let chOff = 5
        let channels = 6
        var output = Array(repeating: Float.nan, count: channels * spatial)
        output.withUnsafeMutableBufferPointer { outBuf in
            SurfaceIO.readFP16(from: s, into: outBuf, channelOffset: chOff, channels: channels, spatial: spatial)
        }

        for ch in 0..<channels {
            for sp in 0..<spatial {
                let outIdx = ch * spatial + sp
                let inIdx = (ch + chOff) * spatial + sp
                XCTAssertEqual(output[outIdx], input[inIdx], accuracy: 1e-2)
            }
        }
    }

    func test_surface_write_fp16_at_offset() throws {
        let channels = 8
        let spatial = 4
        let count = channels * spatial
        let s = makeSurface(bytes: count * 2)

        // Start with zeroed surface.
        let zeros = Array(repeating: Float(0), count: count)
        zeros.withUnsafeBufferPointer { z in
            SurfaceIO.writeFP16(to: s, data: z, channels: channels, spatial: spatial)
        }

        let chOff = 3
        let writeChannels = 2
        let data = (0..<(writeChannels * spatial)).map { Float($0) * 0.5 + 2.0 }
        try data.withUnsafeBufferPointer { buf in
            try SurfaceIO.writeFP16At(to: s, channelOffset: chOff, data: buf, channels: writeChannels, spatial: spatial)
        }

        var out = Array(repeating: Float.nan, count: count)
        out.withUnsafeMutableBufferPointer { outBuf in
            SurfaceIO.readFP16(from: s, into: outBuf, channelOffset: 0, channels: channels, spatial: spatial)
        }

        for ch in 0..<channels {
            for sp in 0..<spatial {
                let idx = ch * spatial + sp
                if chOff <= ch && ch < chOff + writeChannels {
                    let srcIdx = (ch - chOff) * spatial + sp
                    XCTAssertEqual(out[idx], data[srcIdx], accuracy: 1e-2)
                } else {
                    XCTAssertEqual(out[idx], 0, accuracy: 0)
                }
            }
        }
    }

    func test_surface_copy_fp16_between_surfaces() throws {
        let channels = 8
        let spatial = 4
        let count = channels * spatial

        let src = makeSurface(bytes: count * 2)
        let dst = makeSurface(bytes: count * 2)

        let input = (0..<count).map { Float($0) * 0.25 - 1.0 }
        input.withUnsafeBufferPointer { inputBuf in
            SurfaceIO.writeFP16(to: src, data: inputBuf, channels: channels, spatial: spatial)
        }

        let zeros = Array(repeating: Float(0), count: count)
        zeros.withUnsafeBufferPointer { z in
            SurfaceIO.writeFP16(to: dst, data: z, channels: channels, spatial: spatial)
        }

        let dstOff = 2
        let copyChannels = 6
        try SurfaceIO.copyFP16(dst: dst, dstChannelOffset: dstOff, src: src, srcChannelOffset: 0, channels: copyChannels, spatial: spatial)

        var out = Array(repeating: Float.nan, count: count)
        out.withUnsafeMutableBufferPointer { outBuf in
            SurfaceIO.readFP16(from: dst, into: outBuf, channelOffset: 0, channels: channels, spatial: spatial)
        }

        for ch in 0..<channels {
            for sp in 0..<spatial {
                let idx = ch * spatial + sp
                if ch < dstOff {
                    XCTAssertEqual(out[idx], 0, accuracy: 0)
                } else {
                    let srcIdx = (ch - dstOff) * spatial + sp
                    XCTAssertEqual(out[idx], input[srcIdx], accuracy: 1e-2)
                }
            }
        }
    }

    func test_surface_write_read_zero_count_noop() throws {
        let s = makeSurface(bytes: 2)
        let emptyIn: [Float] = []
        var emptyOut: [Float] = []

        try emptyIn.withUnsafeBufferPointer { inputBuf in
            SurfaceIO.writeFP16(to: s, data: inputBuf, channels: 0, spatial: 0)
            try SurfaceIO.writeFP16At(to: s, channelOffset: 0, data: inputBuf, channels: 0, spatial: 0)
        }
        emptyOut.withUnsafeMutableBufferPointer { outBuf in
            SurfaceIO.readFP16(from: s, into: outBuf, channelOffset: 0, channels: 0, spatial: 0)
        }
        try SurfaceIO.copyFP16(dst: s, dstChannelOffset: 0, src: s, srcChannelOffset: 0, channels: 0, spatial: 0)
    }

    func test_surfaceio_rejects_out_of_range_int32_arguments() throws {
        let s = makeSurface(bytes: 2)
        let one = [Float](repeating: 1, count: 1)

        try one.withUnsafeBufferPointer { buf in
            XCTAssertThrowsError(
                try SurfaceIO.writeFP16At(
                    to: s,
                    channelOffset: Int(Int32.max) + 1,
                    data: buf,
                    channels: 1,
                    spatial: 1
                )
            )
        }

        XCTAssertThrowsError(
            try SurfaceIO.copyFP16(
                dst: s,
                dstChannelOffset: 0,
                src: s,
                srcChannelOffset: 0,
                channels: Int(Int32.max) + 1,
                spatial: 1
            )
        )
    }

    func test_layer_activations_include_qkv_buffers() {
        let acts = LayerActivations()
        let dimSeq = ModelConfig.dim * ModelConfig.seqLen
        XCTAssertEqual(acts.Q.count, dimSeq)
        XCTAssertEqual(acts.K.count, dimSeq)
        XCTAssertEqual(acts.V.count, dimSeq)
    }

    func test_model_config_derived_totals() {
        XCTAssertEqual(ModelConfig.totalWeightKernels, ModelConfig.nLayers * ModelConfig.kernelsPerLayer)
        XCTAssertEqual(
            ModelConfig.totalParams,
            ModelConfig.nLayers * ModelConfig.layerParams + ModelConfig.dim + ModelConfig.vocab * ModelConfig.dim
        )
    }

    func test_model_config_matches_stories_config_header_constants() throws {
        let cHeader = try readStoriesConfigHeader()
        let expected: [(String, Int)] = [
            ("DIM", ModelConfig.dim),
            ("HIDDEN", ModelConfig.hidden),
            ("HEADS", ModelConfig.heads),
            ("SEQ", ModelConfig.seqLen),
            ("NLAYERS", ModelConfig.nLayers),
            ("VOCAB", ModelConfig.vocab),
            ("ACCUM_STEPS", ModelConfig.accumSteps),
            ("MAX_COMPILES", ModelConfig.maxCompiles),
            ("KERNELS_PER_LAYER", ModelConfig.kernelsPerLayer),
        ]

        for (define, swiftValue) in expected {
            let parsed = parseCDefineInt(define, in: cHeader)
            XCTAssertNotNil(parsed, "Missing #define \(define) in stories_config.h")
            XCTAssertEqual(parsed, swiftValue, "Mismatch for \(define)")
        }
    }

    func test_layer_weights_all_counts_match_model_config() {
        let weights = LayerWeights()
        XCTAssertEqual(weights.Wq.count, ModelConfig.wqSize)
        XCTAssertEqual(weights.Wk.count, ModelConfig.wqSize)
        XCTAssertEqual(weights.Wv.count, ModelConfig.wqSize)
        XCTAssertEqual(weights.Wo.count, ModelConfig.woSize)
        XCTAssertEqual(weights.W1.count, ModelConfig.w1Size)
        XCTAssertEqual(weights.W2.count, ModelConfig.w2Size)
        XCTAssertEqual(weights.W3.count, ModelConfig.w3Size)
        XCTAssertEqual(weights.rmsAtt.count, ModelConfig.dim)
        XCTAssertEqual(weights.rmsFfn.count, ModelConfig.dim)
    }

    func test_layer_adam_all_counts_match_model_config() {
        let adam = LayerAdam()
        XCTAssertEqual(adam.Wq.count, ModelConfig.wqSize)
        XCTAssertEqual(adam.Wk.count, ModelConfig.wqSize)
        XCTAssertEqual(adam.Wv.count, ModelConfig.wqSize)
        XCTAssertEqual(adam.Wo.count, ModelConfig.woSize)
        XCTAssertEqual(adam.W1.count, ModelConfig.w1Size)
        XCTAssertEqual(adam.W2.count, ModelConfig.w2Size)
        XCTAssertEqual(adam.W3.count, ModelConfig.w3Size)
        XCTAssertEqual(adam.rmsAtt.count, ModelConfig.dim)
        XCTAssertEqual(adam.rmsFfn.count, ModelConfig.dim)
    }

    func test_layer_activations_all_counts_match_model_config() {
        let acts = LayerActivations()
        let dimSeq = ModelConfig.dim * ModelConfig.seqLen
        let hidSeq = ModelConfig.hidden * ModelConfig.seqLen
        XCTAssertEqual(acts.layerIn.count, dimSeq)
        XCTAssertEqual(acts.xnorm.count, dimSeq)
        XCTAssertEqual(acts.Q.count, dimSeq)
        XCTAssertEqual(acts.K.count, dimSeq)
        XCTAssertEqual(acts.V.count, dimSeq)
        XCTAssertEqual(acts.attnOut.count, dimSeq)
        XCTAssertEqual(acts.oOut.count, dimSeq)
        XCTAssertEqual(acts.x2.count, dimSeq)
        XCTAssertEqual(acts.x2norm.count, dimSeq)
        XCTAssertEqual(acts.h1.count, hidSeq)
        XCTAssertEqual(acts.h3.count, hidSeq)
        XCTAssertEqual(acts.siluOut.count, hidSeq)
        XCTAssertEqual(acts.ffnOut.count, dimSeq)
    }

    func test_layer_gradients_zero_resets_modified_values() {
        let grads = LayerGradients()

        func writeSentinels(_ buffer: borrowing TensorBuffer) {
            buffer.withUnsafeMutableBufferPointer { ptr in
                guard !ptr.isEmpty else { return }
                ptr[0] = 123.5
                ptr[ptr.count - 1] = -9.25
            }
        }
        func assertSentinelsAreZero(_ buffer: borrowing TensorBuffer, file: StaticString = #filePath, line: UInt = #line) {
            buffer.withUnsafeBufferPointer { ptr in
                guard !ptr.isEmpty else { return }
                XCTAssertEqual(ptr[0], 0, accuracy: 0, file: file, line: line)
                XCTAssertEqual(ptr[ptr.count - 1], 0, accuracy: 0, file: file, line: line)
            }
        }

        writeSentinels(grads.Wq); writeSentinels(grads.Wk); writeSentinels(grads.Wv); writeSentinels(grads.Wo)
        writeSentinels(grads.W1); writeSentinels(grads.W2); writeSentinels(grads.W3)
        writeSentinels(grads.rmsAtt); writeSentinels(grads.rmsFfn)

        grads.zero()

        assertSentinelsAreZero(grads.Wq); assertSentinelsAreZero(grads.Wk); assertSentinelsAreZero(grads.Wv); assertSentinelsAreZero(grads.Wo)
        assertSentinelsAreZero(grads.W1); assertSentinelsAreZero(grads.W2); assertSentinelsAreZero(grads.W3)
        assertSentinelsAreZero(grads.rmsAtt); assertSentinelsAreZero(grads.rmsFfn)
    }

    func test_layer_storage_subscript_read_modify() {
        var storage = LayerStorage<Int>(count: 4) { $0 * 10 }
        XCTAssertEqual(storage.count, 4)
        XCTAssertEqual(storage[0], 0)
        XCTAssertEqual(storage[1], 10)
        XCTAssertEqual(storage[2], 20)
        XCTAssertEqual(storage[3], 30)

        storage[2] = 999
        XCTAssertEqual(storage[2], 999)
    }

    func test_checkpoint_header_layout() {
        CheckpointHeader.validateLayout()
    }

    func test_checkpoint_header_defaults_match_c_header() {
        let hdr = CheckpointHeader()
        XCTAssertEqual(hdr.magic, 0x424C5A54)
        XCTAssertEqual(hdr.version, 2)
    }

    func test_checkpoint_header_all_default_fields() {
        let hdr = CheckpointHeader()
        XCTAssertEqual(hdr.magic, 0x424C5A54)
        XCTAssertEqual(hdr.version, 2)
        XCTAssertEqual(hdr.step, 0)
        XCTAssertEqual(hdr.totalSteps, 0)
        XCTAssertEqual(hdr.nLayers, 0)
        XCTAssertEqual(hdr.vocabSize, 0)
        XCTAssertEqual(hdr.dim, 0)
        XCTAssertEqual(hdr.hiddenDim, 0)
        XCTAssertEqual(hdr.nHeads, 0)
        XCTAssertEqual(hdr.seqLen, 0)
        XCTAssertEqual(hdr.lr, 0)
        XCTAssertEqual(hdr.loss, 0)
        XCTAssertEqual(hdr.cumCompile, 0)
        XCTAssertEqual(hdr.cumTrain, 0)
        XCTAssertEqual(hdr.cumWall, 0)
        XCTAssertEqual(hdr.cumSteps, 0)
        XCTAssertEqual(hdr.cumBatches, 0)
        XCTAssertEqual(hdr.adamT, 0)
        XCTAssertEqual(hdr.pad0, 0)
        XCTAssertEqual(hdr.pad1, 0)
        XCTAssertEqual(hdr.pad2, 0)
    }
}
