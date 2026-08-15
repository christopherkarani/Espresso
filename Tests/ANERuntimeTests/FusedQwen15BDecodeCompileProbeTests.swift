import XCTest
import ANETypes
@testable import ANERuntime
@testable import MILGenerator

private func requireANEHardware(file: StaticString = #filePath, line: UInt = #line) throws {
    guard ProcessInfo.processInfo.environment["ANE_HARDWARE_TESTS"] == "1" else {
        throw XCTSkip("Set ANE_HARDWARE_TESTS=1 to run ANE hardware tests", file: file, line: line)
    }
}

private enum Qwen15BFusionProbeSupport {
    static let packedNativeDir: String = {
        let home = FileManager.default.homeDirectoryForCurrentUser.path
        return home + "/Library/Caches/Espresso/qwen25-15b/Qwen2.5-1.5B-Instruct-native"
    }()

    static func verifyPackedShapeOrFail() throws {
        let metadataURL = URL(fileURLWithPath: packedNativeDir)
            .appendingPathComponent("metadata.json")
        let data = try Data(contentsOf: metadataURL)
        let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        XCTAssertEqual(json?["nLayer"] as? Int, 28)
        XCTAssertEqual(json?["dModel"] as? Int, 1536)
        XCTAssertEqual(json?["nHead"] as? Int, 12)
        XCTAssertEqual(json?["nKVHead"] as? Int, 2)
        XCTAssertEqual(json?["headDim"] as? Int, 128)
        XCTAssertEqual(json?["hiddenDim"] as? Int, 8960)
        XCTAssertEqual(json?["vocab"] as? Int, 151_936)
        if let theta = json?["ropeTheta"] as? Double {
            XCTAssertEqual(theta, 1_000_000, accuracy: 1)
        } else if let theta = json?["ropeTheta"] as? Int {
            XCTAssertEqual(theta, 1_000_000)
        } else {
            XCTFail("metadata.json missing ropeTheta")
        }
    }

    static func makeSpec(layerCount: Int) throws -> (FusedHybridDecodeBlockKernelSet.CompileSpec, Bool) {
        try verifyPackedShapeOrFail()
        do {
            let blobs = try FusedHybridDecodeBlockKernelSet.loadPackedQwen15BWeightBlobs(
                layerCount: layerCount,
                nativeDir: packedNativeDir
            )
            return (FusedHybridDecodeBlockKernelSet.makeQwen15BSpec(
                layerCount: layerCount,
                weightBlobs: blobs,
                usedRealWeights: true
            ), true)
        } catch {
            print("  NOTE: packed 1.5B layer weights unavailable (\(error)); using dummy blobs")
            let blobs = FusedHybridDecodeBlockKernelSet.dummyQwen15BWeightBlobs(layerCount: layerCount)
            return (FusedHybridDecodeBlockKernelSet.makeQwen15BSpec(
                layerCount: layerCount,
                weightBlobs: blobs,
                usedRealWeights: false
            ), false)
        }
    }

    static func runCompile(layerCount: Int) throws -> FusedHybridDecodeBlockCompileReport {
        try requireANEHardware()
        let (spec, _) = try makeSpec(layerCount: layerCount)
        let report = FusedHybridDecodeBlockKernelSet.compile(spec)
        print(
            """
              fused_hybrid_block N=\(report.layerCount) \
              pass=\(report.passed) \
              compile_ms=\(String(format: "%.1f", report.compileMs)) \
              programs=\(report.programCount) \
              hops/token=\(report.hopsPerToken) \
              weight_blob_mb=\(String(format: "%.1f", Double(report.weightBlobBytes) / 1_000_000)) \
              real_weights=\(report.usedRealWeights) \
              hex=\(report.hexId) \
              mil=\(report.milDumpPath ?? "-") \
              fail=\(report.failingOp ?? "-") \
              err=\(report.errorDescription ?? "-")
            """
        )
        return report
    }
}

final class FusedHybridDecodeBlockSpecTests: XCTestCase {
    func test_dummy_n1_spec_has_eleven_qwen15b_blobs_and_ios18() {
        let blobs = FusedHybridDecodeBlockKernelSet.dummyQwen15BWeightBlobs(layerCount: 1)
        let spec = FusedHybridDecodeBlockKernelSet.makeQwen15BSpec(
            layerCount: 1,
            weightBlobs: blobs,
            usedRealWeights: false
        )
        XCTAssertEqual(spec.weights.count, 11)
        XCTAssertTrue(spec.milText.contains("func main<ios18>"))
        XCTAssertTrue(spec.milText.contains("l0_bq.bin"))
        XCTAssertGreaterThan(spec.weightBlobBytes, 80_000_000)
        XCTAssertEqual(FusedHybridDecodeBlockGenerator.hopsPerToken(layerCount: 1), 28)
    }
}

/// Stories-width two-layer (`FusedTwoLayerDecodeKernelSet`) is a different graph
/// (768/2048, packed KV, attention probe) and stays expected-fail. This probe is
/// QKV+FFN only at 1536/8960.
final class FusedQwen15BDecodeN1CompileTests: XCTestCase {
    func test_n1_fused_qkv_ffn_compile_at_qwen15b_widths() throws {
        let report = try Qwen15BFusionProbeSupport.runCompile(layerCount: 1)
        if !report.passed {
            XCTFail(
                "N=1 compile failed: \(report.failingOp ?? "?") \(report.errorDescription ?? "") MIL=\(report.milDumpPath ?? "")"
            )
        }
        XCTAssertEqual(report.programCount, 1)
        XCTAssertTrue(report.usedRealWeights, "N=1 must close on real packed 1.5B layer weights")
    }
}

final class FusedQwen15BDecodeN2CompileTests: XCTestCase {
    func test_n2_fused_qkv_ffn_compile_at_qwen15b_widths() throws {
        let report = try Qwen15BFusionProbeSupport.runCompile(layerCount: 2)
        if report.passed {
            print("  NOTE: 1.5B N=2 QKV+FFN compiled. Stories FusedTwoLayerDecodeKernelSet remains expected-fail.")
            XCTAssertTrue(report.usedRealWeights)
        } else {
            XCTAssertNotNil(report.milDumpPath)
            XCTAssertTrue(
                (report.errorDescription ?? "").contains("ANE kernel compilation failed")
                    || (report.failingOp ?? "").contains("InvalidMIL")
                    || (report.failingOp ?? "").contains("ANE kernel compilation failed"),
                "Expected controlled compile error, got \(report.errorDescription ?? "")"
            )
        }
    }
}

final class FusedQwen15BDecodeN4CompileTests: XCTestCase {
    func test_n4_fused_qkv_ffn_compile_at_qwen15b_widths() throws {
        let report = try Qwen15BFusionProbeSupport.runCompile(layerCount: 4)
        if !report.passed {
            XCTAssertNotNil(report.milDumpPath)
        }
    }
}

final class FusedQwen15BDecodeN7CompileTests: XCTestCase {
    func test_n7_fused_qkv_ffn_compile_at_qwen15b_widths() throws {
        let report = try Qwen15BFusionProbeSupport.runCompile(layerCount: 7)
        if !report.passed {
            XCTAssertNotNil(report.milDumpPath)
        }
    }
}
