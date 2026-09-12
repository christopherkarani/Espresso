import Darwin
import Foundation
import XCTest
@testable import ANETypes
@testable import Espresso

/// Metal GPU LM head (GEMV + two-stage argmax) must agree with the CPU FP16 tiled
/// classifier on every hidden vector. The full-size 1.5B roofline is gated behind
/// `ESPRESSO_LMHEAD_ROOFLINE=1` and the packed `lm_head.bin`.
final class MetalLMHeadArgmaxTests: XCTestCase {

    private static func randomWeights(vocab: Int, dim: Int, seed: UInt64) -> TensorBufferFP16 {
        var rng = SplitMix64(seed: seed)
        let fp32 = TensorBuffer(count: vocab * dim, zeroed: true)
        fp32.withUnsafeMutablePointer { ptr in
            for i in 0..<(vocab * dim) {
                ptr[i] = rng.nextUnitFloat() * 2 - 1
            }
        }
        return TensorBufferFP16(quantizing: fp32, rows: vocab, cols: dim)
    }

    private static func makeHead(_ weights: borrowing TensorBufferFP16, vocab: Int, dim: Int) throws -> MetalLMHeadArgmax {
        try weights.withUnsafePointer { ptr in
            try MetalLMHeadArgmax(
                weightsFP16: UnsafeBufferPointer(start: ptr, count: vocab * dim),
                vocabSize: vocab,
                dim: dim
            )
        }
    }

    private static func cpuArgmax(_ weights: borrowing TensorBufferFP16, hidden: [Float], vocab: Int, dim: Int) -> Int {
        weights.withUnsafePointer { w in
            hidden.withUnsafeBufferPointer { h in
                FP16TiledClassifier.tiledMatvecArgmax(
                    weights: w,
                    input: h.baseAddress!,
                    vocabSize: vocab,
                    dim: dim
                )
            }
        }
    }

    func testMatchesCPUTiledOnRandomWeights() throws {
        // vocab deliberately not a multiple of rowsPerThreadgroup to cover the tail group.
        let vocab = 4_099
        let dim = 512
        let fp16 = Self.randomWeights(vocab: vocab, dim: dim, seed: 0xC0FFEE)
        let head = try Self.makeHead(fp16, vocab: vocab, dim: dim)

        var rng = SplitMix64(seed: 42)
        for trial in 0..<24 {
            let hidden = (0..<dim).map { _ in rng.nextUnitFloat() * 4 - 2 }
            let expected = Self.cpuArgmax(fp16, hidden: hidden, vocab: vocab, dim: dim)
            let actual = try head.argmax(hidden: hidden)
            XCTAssertEqual(actual, expected, "trial \(trial): metal head disagrees with CPU tiled argmax")
        }
    }

    func testDominantRowInLastPartialGroupWins() throws {
        let vocab = 4_099
        let dim = 256
        let fp32 = TensorBuffer(count: vocab * dim, zeroed: true)
        fp32.withUnsafeMutablePointer { ptr in
            for r in 0..<vocab {
                for c in 0..<dim {
                    ptr[r * dim + c] = Float(r % 7) * 0.001
                }
            }
            for c in 0..<dim {
                ptr[(vocab - 1) * dim + c] = 5.0
            }
        }
        let fp16 = TensorBufferFP16(quantizing: fp32, rows: vocab, cols: dim)
        let head = try fp16.withUnsafePointer { ptr in
            try MetalLMHeadArgmax(
                weightsFP16: UnsafeBufferPointer(start: ptr, count: vocab * dim),
                vocabSize: vocab,
                dim: dim
            )
        }
        let hidden = [Float](repeating: 1, count: dim)
        XCTAssertEqual(try head.argmax(hidden: hidden), vocab - 1)
    }

    func testTiesResolveToLowestIndex() throws {
        let vocab = 300
        let dim = 256
        let fp32 = TensorBuffer(count: vocab * dim, zeroed: true)
        fp32.withUnsafeMutablePointer { ptr in
            // Rows 17 and 250 are identical and dominant.
            for c in 0..<dim {
                ptr[17 * dim + c] = 1.0
                ptr[250 * dim + c] = 1.0
            }
        }
        let fp16 = TensorBufferFP16(quantizing: fp32, rows: vocab, cols: dim)
        let head = try fp16.withUnsafePointer { ptr in
            try MetalLMHeadArgmax(
                weightsFP16: UnsafeBufferPointer(start: ptr, count: vocab * dim),
                vocabSize: vocab,
                dim: dim
            )
        }
        let hidden = [Float](repeating: 0.5, count: dim)
        XCTAssertEqual(try head.argmax(hidden: hidden), 17)
    }

    func testRejectsUnsupportedDim() throws {
        let vocab = 16
        let dim = 100
        let fp16 = TensorBufferFP16(quantizing: TensorBuffer(count: vocab * dim, zeroed: true), rows: vocab, cols: dim)
        XCTAssertThrowsError(
            try fp16.withUnsafePointer { ptr in
                try MetalLMHeadArgmax(
                    weightsFP16: UnsafeBufferPointer(start: ptr, count: vocab * dim),
                    vocabSize: vocab,
                    dim: dim
                )
            }
        )
    }

    /// Full Qwen2.5-1.5B head: exactness against the CPU tiled reference and a
    /// latency/bandwidth roofline. Set ESPRESSO_LMHEAD_ROOFLINE=1.
    func test_qwen15b_metal_lm_head_roofline() throws {
        guard ProcessInfo.processInfo.environment["ESPRESSO_LMHEAD_ROOFLINE"] == "1" else {
            throw XCTSkip("Set ESPRESSO_LMHEAD_ROOFLINE=1 to run the 1.5B Metal head microbench")
        }
        let vocab = 151_936
        let dim = 1_536
        let payloadBytes = vocab * dim * MemoryLayout<UInt16>.stride
        let path = Self.lmHeadPath()
        guard FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("missing packed 1.5B lm_head.bin at \(path)")
        }
        let fd = open(path, O_RDONLY)
        guard fd >= 0 else { throw XCTSkip("could not open \(path)") }
        defer { close(fd) }
        var st = stat()
        guard fstat(fd, &st) == 0, Int(st.st_size) >= 128 + payloadBytes else {
            throw XCTSkip("lm_head.bin too small")
        }
        let fileSize = Int(st.st_size)
        guard let mapped = mmap(nil, fileSize, PROT_READ, MAP_PRIVATE, fd, 0), mapped != MAP_FAILED else {
            throw XCTSkip("mmap failed")
        }
        defer { munmap(mapped, fileSize) }
        let weights = UnsafePointer(mapped.advanced(by: 128).assumingMemoryBound(to: UInt16.self))

        let head = try MetalLMHeadArgmax(
            weightsFP16: UnsafeBufferPointer(start: weights, count: vocab * dim),
            vocabSize: vocab,
            dim: dim
        )

        var rng = SplitMix64(seed: 7)
        let hiddenStates: [[Float]] = (0..<16).map { _ in
            (0..<dim).map { _ in rng.nextUnitFloat() * 6 - 3 }
        }
        var mismatches = 0
        for hidden in hiddenStates {
            let expected = hidden.withUnsafeBufferPointer { h in
                FP16TiledClassifier.tiledMatvecArgmax(weights: weights, input: h.baseAddress!, vocabSize: vocab, dim: dim)
            }
            let actual = try head.argmax(hidden: hidden)
            if actual != expected { mismatches += 1 }
        }
        XCTAssertEqual(mismatches, 0, "metal head disagrees with CPU tiled argmax on \(mismatches)/\(hiddenStates.count) hiddens")

        let warmup = 5
        let iters = 50
        for i in 0..<warmup { _ = try head.argmax(hidden: hiddenStates[i % hiddenStates.count]) }
        let start = DispatchTime.now().uptimeNanoseconds
        for i in 0..<iters { _ = try head.argmax(hidden: hiddenStates[i % hiddenStates.count]) }
        let ms = Double(DispatchTime.now().uptimeNanoseconds - start) / 1e6 / Double(iters)
        let gbs = Double(payloadBytes) / (ms / 1_000) / 1e9
        // Dispatch floor: same pipeline over a 256-row slice isolates command-buffer
        // submit/wait overhead from weight streaming.
        let tinyHead = try MetalLMHeadArgmax(
            weightsFP16: UnsafeBufferPointer(start: weights, count: 256 * dim),
            vocabSize: 256,
            dim: dim
        )
        for i in 0..<warmup { _ = try tinyHead.argmax(hidden: hiddenStates[i % hiddenStates.count]) }
        let tinyStart = DispatchTime.now().uptimeNanoseconds
        for i in 0..<iters { _ = try tinyHead.argmax(hidden: hiddenStates[i % hiddenStates.count]) }
        let tinyMs = Double(DispatchTime.now().uptimeNanoseconds - tinyStart) / 1e6 / Double(iters)

        let line = String(
            format: "metal lm_head 151936×1536 fp16: %.3f ms/token  %.1f GB/s  dispatch_floor=%.3f ms  (N=%d)\n",
            ms, gbs, tinyMs, iters
        )
        fputs(line, stderr)
        print(line)
    }

    private static func lmHeadPath() -> String {
        if let override = ProcessInfo.processInfo.environment["ESPRESSO_LM_HEAD_BIN"], !override.isEmpty {
            return override
        }
        let home = FileManager.default.homeDirectoryForCurrentUser.path
        return home + "/Library/Caches/Espresso/qwen25-15b/Qwen2.5-1.5B-Instruct.esp/weights/lm_head.bin"
    }
}

private struct SplitMix64 {
    private var state: UInt64

    init(seed: UInt64) { state = seed }

    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }

    mutating func nextUnitFloat() -> Float {
        Float(next() >> 40) / Float(1 << 24)
    }
}
