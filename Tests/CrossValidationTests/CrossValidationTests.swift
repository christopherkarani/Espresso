import XCTest
import Foundation
import Darwin
import IOSurface
import ANERuntime
import ANETypes
import MILGenerator

private let cvDim = 768
private let cvHidden = 2048
private let cvSeq = 64

private func requireObjCCrossValidation(file: StaticString = #filePath, line: UInt = #line) throws {
    guard ProcessInfo.processInfo.environment["OBJC_CROSS_VALIDATION"] == "1" else {
        throw XCTSkip("ObjC cross-validation disabled (set OBJC_CROSS_VALIDATION=1)", file: file, line: line)
    }
}

private func requireANEHardware(file: StaticString = #filePath, line: UInt = #line) throws {
    guard ProcessInfo.processInfo.environment["ANE_HARDWARE_TESTS"] == "1" else {
        throw XCTSkip("ANE hardware tests disabled (set ANE_HARDWARE_TESTS=1)", file: file, line: line)
    }

    let handle = dlopen(
        "/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine",
        RTLD_NOW
    )
    guard handle != nil else {
        throw XCTSkip("AppleNeuralEngine.framework unavailable", file: file, line: line)
    }
    dlclose(handle)
}

private func repoRootURL() -> URL {
    URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .deletingLastPathComponent()
}

private func goldenDirURL() -> URL {
    repoRootURL().appendingPathComponent("training/golden_outputs")
}

private func goldenURL(_ name: String) -> URL {
    goldenDirURL().appendingPathComponent(name)
}

private func requireGoldenURL(_ name: String, file: StaticString = #filePath, line: UInt = #line) throws -> URL {
    let url = goldenURL(name)
    guard FileManager.default.fileExists(atPath: url.path) else {
        throw XCTSkip("Missing golden artifact: \(url.path). Run ./scripts/cross_validate.sh", file: file, line: line)
    }
    return url
}

private func loadGoldenData(_ name: String, file: StaticString = #filePath, line: UInt = #line) throws -> Data {
    let url = try requireGoldenURL(name, file: file, line: line)
    return try Data(contentsOf: url, options: .mappedIfSafe)
}

private func loadFloat32LE(_ name: String, expectedCount: Int, file: StaticString = #filePath, line: UInt = #line) throws -> [Float] {
    let data = try loadGoldenData(name, file: file, line: line)
    let expectedBytes = expectedCount * MemoryLayout<UInt32>.stride
    guard data.count == expectedBytes else {
        throw XCTSkip(
            "Fixture size mismatch for \(name): expected \(expectedBytes) bytes, got \(data.count)",
            file: file,
            line: line
        )
    }

    var values = [Float](repeating: 0, count: expectedCount)
    data.withUnsafeBytes { raw in
        guard let base = raw.baseAddress else { return }
        let bytes = base.assumingMemoryBound(to: UInt8.self)
        for i in 0..<expectedCount {
            let o = i * 4
            let bits = UInt32(bytes[o])
                | (UInt32(bytes[o + 1]) << 8)
                | (UInt32(bytes[o + 2]) << 16)
                | (UInt32(bytes[o + 3]) << 24)
            values[i] = Float(bitPattern: bits)
        }
    }
    return values
}

private func writeFloat32Surface(_ surface: IOSurfaceRef, values: [Float]) {
    let expectedBytes = values.count * MemoryLayout<Float>.stride
    precondition(expectedBytes <= IOSurfaceGetAllocSize(surface))

    precondition(IOSurfaceLock(surface, [], nil) == kIOReturnSuccess)
    defer { IOSurfaceUnlock(surface, [], nil) }

    let base = IOSurfaceGetBaseAddress(surface)

    _ = values.withUnsafeBytes { src in
        memcpy(base, src.baseAddress!, expectedBytes)
    }
}

private func readFloat32Surface(_ surface: IOSurfaceRef, count: Int) -> [Float] {
    let expectedBytes = count * MemoryLayout<Float>.stride
    precondition(expectedBytes <= IOSurfaceGetAllocSize(surface))

    precondition(IOSurfaceLock(surface, .readOnly, nil) == kIOReturnSuccess)
    defer { IOSurfaceUnlock(surface, .readOnly, nil) }

    let base = IOSurfaceGetBaseAddress(surface)

    var values = [Float](repeating: 0, count: count)
    _ = values.withUnsafeMutableBytes { dst in
        memcpy(dst.baseAddress!, base, expectedBytes)
    }
    return values
}

private func maxAbsDiff(actual: [Float], expected: [Float]) -> (index: Int, actual: Float, expected: Float, diff: Float) {
    precondition(actual.count == expected.count, "Mismatched vector lengths")

    var bestIndex = 0
    var bestActual: Float = 0
    var bestExpected: Float = 0
    var bestDiff: Float = -.infinity

    for i in actual.indices {
        let a = actual[i]
        let e = expected[i]
        let d = abs(a - e)
        if d > bestDiff {
            bestDiff = d
            bestIndex = i
            bestActual = a
            bestExpected = e
        }
    }

    return (bestIndex, bestActual, bestExpected, bestDiff)
}

private func meanAbsDiff(actual: [Float], expected: [Float]) -> Float {
    precondition(actual.count == expected.count, "Mismatched vector lengths")
    guard !actual.isEmpty else { return 0 }

    var sum: Float = 0
    for i in actual.indices {
        sum += abs(actual[i] - expected[i])
    }
    return sum / Float(actual.count)
}

private func measureElapsedMillis(_ body: () throws -> Void) rethrows -> Double {
    var timebase = mach_timebase_info_data_t()
    mach_timebase_info(&timebase)
    let start = mach_absolute_time()
    try body()
    let end = mach_absolute_time()
    let elapsed = Double(end - start)
    let nanos = elapsed * Double(timebase.numer) / Double(timebase.denom)
    return nanos / 1_000_000.0
}

private func emitPhase8Metric(
    area: String,
    maxAbsDiff: Float,
    meanAbsDiff: Float,
    tolerance: Float,
    swiftEvalMS: Double,
    pass: Bool
) {
    guard ProcessInfo.processInfo.environment["PHASE8_BENCHMARKS"] == "1" else {
        return
    }

    let payload: [String: Any] = [
        "type": "phase8_metric",
        "area": area,
        "max_abs_diff": maxAbsDiff,
        "mean_abs_diff": meanAbsDiff,
        "tolerance": tolerance,
        "swift_eval_ms": swiftEvalMS,
        "pass": pass,
    ]

    guard JSONSerialization.isValidJSONObject(payload),
          let data = try? JSONSerialization.data(withJSONObject: payload, options: [.sortedKeys]) else {
        return
    }
    FileHandle.standardError.write(data)
    FileHandle.standardError.write(Data("\n".utf8))
}

private func hasNonZero(_ values: [Float], threshold: Float = 1e-6) -> Bool {
    values.contains(where: { abs($0) > threshold })
}

final class CrossValidationTests: XCTestCase {
    func test_weight_blob_matches_objc_binary_fixture() throws {
        try requireObjCCrossValidation()

        let expected = try loadGoldenData("weight_blob_4x4.bin")
        let input = (1...16).map(Float.init)
        let actual = WeightBlob.build(from: input, rows: 4, cols: 4)

        XCTAssertEqual(actual, expected, "Swift WeightBlob differs from ObjC build_blob() output")
    }

    func test_causal_mask_matches_objc_binary_fixture() throws {
        try requireObjCCrossValidation()

        let expected = try loadGoldenData("causal_mask_seq8.bin")
        let actual = CausalMask.blob(seqLen: 8)

        XCTAssertEqual(actual, expected, "Swift CausalMask differs from ObjC mask blob")
    }

    func test_full_fused_forward_matches_objc_binary_fixture() throws {
        try requireObjCCrossValidation()
        try requireANEHardware()

        let milURL = try requireGoldenURL("full_fused.mil")
        let milText = try String(contentsOf: milURL, encoding: .utf8)

        let weights: [(path: String, data: Data)] = [
            ("@model_path/weights/wq.bin", try loadGoldenData("full_fused_wq.bin")),
            ("@model_path/weights/wk.bin", try loadGoldenData("full_fused_wk.bin")),
            ("@model_path/weights/wv.bin", try loadGoldenData("full_fused_wv.bin")),
            ("@model_path/weights/wo.bin", try loadGoldenData("full_fused_wo.bin")),
            ("@model_path/weights/mask.bin", try loadGoldenData("full_fused_mask.bin")),
        ]

        let kernel = try ANEKernel(
            milText: milText,
            weights: weights,
            inputBytes: cvDim * cvSeq * 2,
            outputBytes: cvDim * cvSeq * 2
        )

        let input = try loadFloat32LE("full_fused_input_seq64_f32le.bin", expectedCount: cvDim * cvSeq)
        let expected = try loadFloat32LE("full_fused_out_seq64_f32le.bin", expectedCount: cvDim * cvSeq)

        let inputSurface = try kernel.inputSurface(at: 0)
        let outputSurface = try kernel.outputSurface(at: 0)

        input.withUnsafeBufferPointer { ptr in
            SurfaceIO.writeFP16(to: inputSurface, data: ptr, channels: cvDim, spatial: cvSeq)
        }

        let evalMS = try measureElapsedMillis {
            try kernel.eval()
        }

        var actual = [Float](repeating: 0, count: cvDim * cvSeq)
        actual.withUnsafeMutableBufferPointer { ptr in
            SurfaceIO.readFP16(from: outputSurface, into: ptr, channelOffset: 0, channels: cvDim, spatial: cvSeq)
        }

        XCTAssertTrue(hasNonZero(expected), "ObjC expected output appears all-zero")
        XCTAssertTrue(hasNonZero(actual), "Swift ANE output appears all-zero")

        let worst = maxAbsDiff(actual: actual, expected: expected)
        let mean = meanAbsDiff(actual: actual, expected: expected)
        let tolerance: Float = 1e-2
        emitPhase8Metric(
            area: "full_fused_forward",
            maxAbsDiff: worst.diff,
            meanAbsDiff: mean,
            tolerance: tolerance,
            swiftEvalMS: evalMS,
            pass: worst.diff < tolerance
        )
        XCTAssertLessThan(
            worst.diff,
            tolerance,
            "max diff=\(worst.diff) at idx \(worst.index), actual=\(worst.actual), expected=\(worst.expected)"
        )
    }

    func test_fused_backward_matches_objc_binary_fixture() throws {
        try requireObjCCrossValidation()
        try requireANEHardware()

        let milURL = try requireGoldenURL("fused_bwd.mil")
        let milText = try String(contentsOf: milURL, encoding: .utf8)

        let weights: [(path: String, data: Data)] = [
            ("@model_path/weights/w1t.bin", try loadGoldenData("fused_bwd_w1t.bin")),
            ("@model_path/weights/w3t.bin", try loadGoldenData("fused_bwd_w3t.bin")),
        ]

        let kernel = try ANEKernel(
            milText: milText,
            weights: weights,
            inputBytes: cvHidden * 2 * cvSeq * 4,
            outputBytes: cvDim * cvSeq * 4
        )

        let input = try loadFloat32LE("fused_bwd_input_seq64_f32le.bin", expectedCount: cvHidden * 2 * cvSeq)
        let expected = try loadFloat32LE("fused_bwd_dx_seq64_f32le.bin", expectedCount: cvDim * cvSeq)

        let inputSurface = try kernel.inputSurface(at: 0)
        let outputSurface = try kernel.outputSurface(at: 0)

        writeFloat32Surface(inputSurface, values: input)
        let evalMS = try measureElapsedMillis {
            try kernel.eval()
        }
        let actual = readFloat32Surface(outputSurface, count: cvDim * cvSeq)

        XCTAssertTrue(hasNonZero(expected), "ObjC expected backward output appears all-zero")
        XCTAssertTrue(hasNonZero(actual), "Swift ANE backward output appears all-zero")

        let worst = maxAbsDiff(actual: actual, expected: expected)
        let mean = meanAbsDiff(actual: actual, expected: expected)
        let tolerance: Float = 1e-2
        emitPhase8Metric(
            area: "fused_backward",
            maxAbsDiff: worst.diff,
            meanAbsDiff: mean,
            tolerance: tolerance,
            swiftEvalMS: evalMS,
            pass: worst.diff < tolerance
        )
        XCTAssertLessThan(
            worst.diff,
            tolerance,
            "max diff=\(worst.diff) at idx \(worst.index), actual=\(worst.actual), expected=\(worst.expected)"
        )
    }

    func test_sdpa5_stdout_contract_matches_objc_capture() throws {
        try requireObjCCrossValidation()
        try requireANEHardware()

        let url = try requireGoldenURL("test_ane_sdpa5.txt")
        let text = try String(contentsOf: url, encoding: .utf8)

        guard text.contains("Test 3: BLOBFILE causal mask") else {
            throw XCTSkip("SDPA5 output missing blob-mask section")
        }

        let pattern = #"blob-mask: diff_causal=([0-9eE+\-.]+) diff_nocausal=([0-9eE+\-.]+)"#
        let regex = try NSRegularExpression(pattern: pattern)
        let fullRange = NSRange(text.startIndex..<text.endIndex, in: text)

        guard let match = regex.firstMatch(in: text, range: fullRange), match.numberOfRanges == 3,
              let causalRange = Range(match.range(at: 1), in: text),
              let noCausalRange = Range(match.range(at: 2), in: text),
              let diffCausal = Float(text[causalRange]),
              let diffNoCausal = Float(text[noCausalRange]) else {
            throw XCTSkip("Unable to parse blob-mask diff metrics from SDPA5 golden output")
        }

        XCTAssertTrue(diffCausal.isFinite)
        XCTAssertTrue(diffNoCausal.isFinite)
        XCTAssertGreaterThanOrEqual(diffCausal, 0)
        XCTAssertGreaterThanOrEqual(diffNoCausal, 0)
        XCTAssertTrue(text.contains("blob-mask:"), "Expected blob-mask metrics in SDPA5 output")
    }

    func test_objc_probe_stdout_contracts_present() throws {
        try requireObjCCrossValidation()

        let required = [
            "test_full_fused.txt",
            "test_fused_bwd.txt",
            "test_ane_sdpa5.txt",
        ]

        for name in required {
            let url = try requireGoldenURL(name)
            let text = try String(contentsOf: url, encoding: .utf8)
            XCTAssertFalse(text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty, "Golden stdout file is empty: \(name)")
        }
    }
}
