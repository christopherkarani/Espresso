import Accelerate
import Darwin
import Foundation
import XCTest
@testable import ANEInterop
@testable import Espresso

/// Phase 10c CPU LM-head roofline. Gated: set ESPRESSO_LMHEAD_ROOFLINE=1.
/// Loads the packed 1.5B `lm_head.bin` (151936 × 1536 fp16) and reports ms / GB/s
/// for (a) allocate-every-call tiled, (b) reused scratch, (c) streaming FP16 GEMV,
/// (d) partitioned Cauchy-Schwarz on a one-time FP32 convert.
final class FP16LMHeadRooflineTests: XCTestCase {

    private static let vocab = 151_936
    private static let dim = 1_536
    private static let payloadBytes = vocab * dim * MemoryLayout<UInt16>.stride
    private static let warmup = 4
    private static let iters = 50

    func test_qwen15b_lm_head_roofline_table() throws {
        guard ProcessInfo.processInfo.environment["ESPRESSO_LMHEAD_ROOFLINE"] == "1" else {
            throw XCTSkip("Set ESPRESSO_LMHEAD_ROOFLINE=1 to run the 1.5B CPU head microbench")
        }

        let blob = try Self.mapLMHead()
        defer { munmap(blob.base, blob.mappedSize) }
        let weights = blob.weights
        let bytes = Double(Self.payloadBytes)

        var hiddenStates = Self.loadDumpedHiddenStates()
        if hiddenStates.count < 8 {
            hiddenStates.append(contentsOf: Self.syntheticHiddenStates(count: max(8, Self.iters) - hiddenStates.count))
        }
        XCTAssertGreaterThanOrEqual(hiddenStates.count, 8)

        let reference = hiddenStates.prefix(8).map { hidden in
            hidden.withUnsafeBufferPointer { h in
                FP16TiledClassifier.tiledMatvecArgmax(
                    weights: weights,
                    input: h.baseAddress!,
                    vocabSize: Self.vocab,
                    dim: Self.dim
                )
            }
        }

        let scratch = FP16TiledClassifier.TileScratch(tileRows: FP16TiledClassifier.tileRows, dim: Self.dim)
        defer { scratch.deallocate() }

        let fp32 = UnsafeMutablePointer<Float>.allocate(capacity: Self.vocab * Self.dim)
        defer { fp32.deallocate() }
        ane_interop_cvt_f16_to_f32(fp32, UnsafeRawPointer(weights), Int32(Self.vocab * Self.dim))
        let blockSize = PartitionedArgmax.defaultBlockSize
        let blockMaxNorms = PartitionedArgmax.precomputeBlockMaxNorms(
            classifier: UnsafePointer(fp32),
            vocabSize: Self.vocab,
            dim: Self.dim,
            blockSize: blockSize
        )
        let logitsScratch = UnsafeMutablePointer<Float>.allocate(capacity: blockSize)
        defer { logitsScratch.deallocate() }
        let fullLogits = UnsafeMutablePointer<Float>.allocate(capacity: Self.vocab)
        defer { fullLogits.deallocate() }
        let maxStripes = 12
        let minStripeRows = (Self.vocab + maxStripes - 1) / maxStripes
        let stripeScratch = UnsafeMutablePointer<UnsafeMutablePointer<Float>>.allocate(capacity: maxStripes)
        for s in 0..<maxStripes {
            stripeScratch[s] = UnsafeMutablePointer<Float>.allocate(capacity: minStripeRows)
        }
        defer {
            for s in 0..<maxStripes { stripeScratch[s].deallocate() }
            stripeScratch.deallocate()
        }
        let stripeResults = UnsafeMutablePointer<(Float, Int)>.allocate(capacity: maxStripes)
        defer { stripeResults.deallocate() }

        func runA(_ hidden: [Float]) -> Int {
            hidden.withUnsafeBufferPointer { h in
                FP16TiledClassifier.tiledMatvecArgmax(
                    weights: weights,
                    input: h.baseAddress!,
                    vocabSize: Self.vocab,
                    dim: Self.dim
                )
            }
        }
        func runB(_ hidden: [Float]) -> Int {
            hidden.withUnsafeBufferPointer { h in
                FP16TiledClassifier.tiledMatvecArgmax(
                    weights: weights,
                    input: h.baseAddress!,
                    vocabSize: Self.vocab,
                    dim: Self.dim,
                    tileRows: FP16TiledClassifier.tileRows,
                    scratch: scratch
                )
            }
        }
        func runC(_ hidden: [Float]) -> Int {
            hidden.withUnsafeBufferPointer { h in
                FP16TiledClassifier.streamingMatvecArgmax(
                    weights: weights,
                    input: h.baseAddress!,
                    vocabSize: Self.vocab,
                    dim: Self.dim
                )
            }
        }
        func runD(_ hidden: [Float]) -> Int {
            var skipped = 0
            return hidden.withUnsafeBufferPointer { h in
                blockMaxNorms.withUnsafeBufferPointer { norms in
                    PartitionedArgmax.compute(
                        classifier: UnsafePointer(fp32),
                        input: h.baseAddress!,
                        logitsScratch: logitsScratch,
                        blockMaxNorms: norms.baseAddress!,
                        vocabSize: Self.vocab,
                        dim: Self.dim,
                        blockSize: blockSize,
                        skippedBlocks: &skipped
                    )
                }
            }
        }
        func runFullSgemm(_ hidden: [Float]) -> Int {
            hidden.withUnsafeBufferPointer { h in
                BLAS.sgemm(
                    CblasRowMajor,
                    CblasNoTrans,
                    CblasNoTrans,
                    m: Int32(Self.vocab),
                    n: 1,
                    k: Int32(Self.dim),
                    alpha: 1.0,
                    a: UnsafePointer(fp32),
                    lda: Int32(Self.dim),
                    b: h.baseAddress!,
                    ldb: 1,
                    beta: 0.0,
                    c: fullLogits,
                    ldc: 1
                )
                var best: Float = 0
                var bestIdx: vDSP_Length = 0
                vDSP_maxvi(fullLogits, 1, &best, &bestIdx, vDSP_Length(Self.vocab))
                return Int(bestIdx)
            }
        }
        func runStriped(_ hidden: [Float], stripes: Int) -> Int {
            hidden.withUnsafeBufferPointer { h in
                let rowsPerStripe = (Self.vocab + stripes - 1) / stripes
                let classifierAddr = UInt(bitPattern: UnsafePointer(fp32))
                let inputAddr = UInt(bitPattern: h.baseAddress!)
                let scratchAddr = UInt(bitPattern: stripeScratch)
                let resultsAddr = UInt(bitPattern: stripeResults)
                DispatchQueue.global(qos: .userInteractive).sync {
                    DispatchQueue.concurrentPerform(iterations: stripes) { s in
                        let start = s * rowsPerStripe
                        guard start < Self.vocab else {
                            UnsafeMutablePointer<(Float, Int)>(bitPattern: resultsAddr)![s] = (-.infinity, 0)
                            return
                        }
                        let count = min(rowsPerStripe, Self.vocab - start)
                        let classifierPtr = UnsafePointer<Float>(bitPattern: classifierAddr)!
                        let inputPtr = UnsafePointer<Float>(bitPattern: inputAddr)!
                        let scratchBase = UnsafeMutablePointer<UnsafeMutablePointer<Float>>(bitPattern: scratchAddr)!
                        let resultsPtr = UnsafeMutablePointer<(Float, Int)>(bitPattern: resultsAddr)!
                        BLAS.sgemm(
                            CblasRowMajor,
                            CblasNoTrans,
                            CblasNoTrans,
                            m: Int32(count),
                            n: 1,
                            k: Int32(Self.dim),
                            alpha: 1.0,
                            a: classifierPtr.advanced(by: start * Self.dim),
                            lda: Int32(Self.dim),
                            b: inputPtr,
                            ldb: 1,
                            beta: 0.0,
                            c: scratchBase[s],
                            ldc: 1
                        )
                        var blockMax: Float = 0
                        var blockMaxIdx: vDSP_Length = 0
                        vDSP_maxvi(scratchBase[s], 1, &blockMax, &blockMaxIdx, vDSP_Length(count))
                        resultsPtr[s] = (blockMax, start + Int(blockMaxIdx))
                    }
                }
                var bestValue = stripeResults[0].0
                var bestIndex = stripeResults[0].1
                for s in 1..<stripes {
                    if stripeResults[s].0 > bestValue {
                        bestValue = stripeResults[s].0
                        bestIndex = stripeResults[s].1
                    }
                }
                return bestIndex
            }
        }
        func runDParallel(_ hidden: [Float]) -> Int {
            hidden.withUnsafeBufferPointer { h in
                blockMaxNorms.withUnsafeBufferPointer { norms in
                    PartitionedArgmax.computeParallel(
                        classifier: UnsafePointer(fp32),
                        input: h.baseAddress!,
                        blockMaxNorms: norms.baseAddress!,
                        vocabSize: Self.vocab,
                        dim: Self.dim,
                        blockSize: blockSize
                    )
                }
            }
        }

        var pruneSkipped = 0
        var pruneTotal = 0
        for hidden in hiddenStates.prefix(8) {
            hidden.withUnsafeBufferPointer { h in
                blockMaxNorms.withUnsafeBufferPointer { norms in
                    var skipped = 0
                    _ = PartitionedArgmax.compute(
                        classifier: UnsafePointer(fp32),
                        input: h.baseAddress!,
                        logitsScratch: logitsScratch,
                        blockMaxNorms: norms.baseAddress!,
                        vocabSize: Self.vocab,
                        dim: Self.dim,
                        blockSize: blockSize,
                        skippedBlocks: &skipped
                    )
                    pruneSkipped += skipped
                    pruneTotal += (Self.vocab + blockSize - 1) / blockSize
                }
            }
        }

        for (i, hidden) in hiddenStates.prefix(8).enumerated() {
            XCTAssertEqual(runB(hidden), reference[i], "reused scratch mismatch at hidden \(i)")
            XCTAssertEqual(runC(hidden), reference[i], "streaming GEMV mismatch at hidden \(i)")
            XCTAssertEqual(runD(hidden), reference[i], "partitioned CS mismatch at hidden \(i)")
            XCTAssertEqual(runDParallel(hidden), reference[i], "parallel CS mismatch at hidden \(i)")
            XCTAssertEqual(runFullSgemm(hidden), reference[i], "full FP32 sgemm mismatch at hidden \(i)")
            XCTAssertEqual(runStriped(hidden, stripes: 8), reference[i], "striped sgemm mismatch at hidden \(i)")
        }

        struct Row {
            let name: String
            let ms: Double
            let bytes: Double
            var gbs: Double { bytes / (ms / 1_000.0) / 1e9 }
        }

        func bench(_ name: String, _ body: ([Float]) -> Int) -> Row {
            for i in 0..<Self.warmup {
                _ = body(hiddenStates[i % hiddenStates.count])
            }
            let start = DispatchTime.now()
            for i in 0..<Self.iters {
                _ = body(hiddenStates[i % hiddenStates.count])
            }
            let elapsedNs = DispatchTime.now().uptimeNanoseconds - start.uptimeNanoseconds
            let ms = Double(elapsedNs) / 1e6 / Double(Self.iters)
            return Row(name: name, ms: ms, bytes: bytes)
        }

        let rows = [
            bench("a tiled allocate-every-call", runA),
            bench("b tiled reused scratch", runB),
            bench("c streaming FP16 GEMV", runC),
            bench("d partitioned Cauchy-Schwarz", runD),
            bench("d2 parallel remaining CS", runDParallel),
            bench("e full FP32 AMX sgemm", runFullSgemm),
            bench("f striped 4-way AMX") { runStriped($0, stripes: 4) },
        ]

        let pruneRate = pruneTotal > 0 ? Double(pruneSkipped) / Double(pruneTotal) : 0
        var table = "lm_head roofline 151936×1536 fp16  bytes=\(Self.payloadBytes)  N=\(Self.iters) warmup=\(Self.warmup)\n"
        table += "hidden_states=\(hiddenStates.count) dumped=\(Self.loadDumpedHiddenStates().count) cs_prune=\(String(format: "%.1f%%", pruneRate * 100))\n"
        for row in rows {
            table += String(format: "%-32@  %7.2f ms  %6.1f GB/s\n", row.name, row.ms, row.gbs)
        }
        table += String(format: "target ≤8.00 ms (~58 GB/s). roofline 467e6/elapsed_s/1e9\n")
        fputs(table, stderr)
        print(table)
    }

    private struct MappedHead {
        let base: UnsafeMutableRawPointer
        let mappedSize: Int
        let weights: UnsafePointer<UInt16>
    }

    private static func mapLMHead() throws -> MappedHead {
        let path = lmHeadPath()
        guard FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("missing packed 1.5B lm_head.bin at \(path)")
        }
        let fd = open(path, O_RDONLY)
        guard fd >= 0 else { throw XCTSkip("could not open \(path)") }
        defer { close(fd) }

        var st = stat()
        guard fstat(fd, &st) == 0 else { throw XCTSkip("fstat failed") }
        let fileSize = Int(st.st_size)
        guard fileSize >= 128 + payloadBytes else {
            throw XCTSkip("lm_head.bin too small: \(fileSize)")
        }

        guard let mapped = mmap(nil, fileSize, PROT_READ, MAP_PRIVATE, fd, 0),
              mapped != MAP_FAILED else {
            throw XCTSkip("mmap failed")
        }
        let header = mapped.assumingMemoryBound(to: UInt8.self)
        let magic = header.advanced(by: 64).withMemoryRebound(to: UInt32.self, capacity: 1) { $0.pointee }
        let dataSize = header.advanced(by: 72).withMemoryRebound(to: UInt32.self, capacity: 1) { $0.pointee }
        let dataOffset = header.advanced(by: 80).withMemoryRebound(to: UInt32.self, capacity: 1) { $0.pointee }
        guard UInt32(littleEndian: magic) == 0xDEAD_BEEF,
              UInt32(littleEndian: dataSize) == UInt32(payloadBytes),
              UInt32(littleEndian: dataOffset) == 128 else {
            munmap(mapped, fileSize)
            throw XCTSkip("unexpected lm_head.bin header magic=\(magic) size=\(dataSize) off=\(dataOffset)")
        }
        let weights = mapped.advanced(by: 128).assumingMemoryBound(to: UInt16.self)
        return MappedHead(base: mapped, mappedSize: fileSize, weights: UnsafePointer(weights))
    }

    private static func lmHeadPath() -> String {
        if let override = ProcessInfo.processInfo.environment["ESPRESSO_LM_HEAD_BIN"], !override.isEmpty {
            return override
        }
        let home = FileManager.default.homeDirectoryForCurrentUser.path
        return home + "/Library/Caches/Espresso/qwen25-15b/Qwen2.5-1.5B-Instruct.esp/weights/lm_head.bin"
    }

    private static func dumpedHiddenPath() -> String {
        ProcessInfo.processInfo.environment["ESPRESSO_DUMP_LM_HEAD_HIDDEN"]
            ?? (NSTemporaryDirectory() + "qwen15b-france-lm-head-hidden.bin")
    }

    private static func loadDumpedHiddenStates() -> [[Float]] {
        let path = dumpedHiddenPath()
        guard let data = try? Data(contentsOf: URL(fileURLWithPath: path)) else { return [] }
        let floats = data.count / MemoryLayout<Float>.stride
        guard floats >= dim, floats.isMultiple(of: dim) else { return [] }
        return data.withUnsafeBytes { raw in
            let src = raw.bindMemory(to: Float.self)
            return stride(from: 0, to: floats, by: dim).map { start in
                Array(src[start..<(start + dim)])
            }
        }
    }

    private static func syntheticHiddenStates(count: Int) -> [[Float]] {
        (0..<count).map { seed in
            var rng = SplitMix64(state: UInt64(seed) &+ 0x9E37_79B9_7F4A_7C15)
            var values = (0..<dim).map { _ in rng.nextFloat() * 0.04 - 0.02 }
            var sumsq: Float = 0
            vDSP_svesq(values, 1, &sumsq, vDSP_Length(dim))
            let inv = 1.0 / (sqrtf(sumsq / Float(dim)) + 1e-6)
            vDSP_vsmul(values, 1, [inv], &values, 1, vDSP_Length(dim))
            return values
        }
    }
}

private struct SplitMix64 {
    var state: UInt64
    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
    mutating func nextFloat() -> Float {
        Float(next() >> 40) / Float(1 << 24)
    }
}
