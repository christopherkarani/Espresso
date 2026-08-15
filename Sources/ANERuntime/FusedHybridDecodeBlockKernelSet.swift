import Darwin
import Foundation
import ANETypes
import MILGenerator

/// Compile-only fused QKV+FFN block. Not wired into generate/chat.
public struct FusedHybridDecodeBlockCompileReport: Sendable {
    public let layerCount: Int
    public let passed: Bool
    public let compileMs: Double
    public let programCount: Int
    public let weightBlobBytes: Int
    public let hexId: String
    public let milDumpPath: String?
    public let failingOp: String?
    public let usedRealWeights: Bool
    public let errorDescription: String?

    public var hopsPerToken: Int {
        FusedHybridDecodeBlockGenerator.hopsPerToken(layerCount: layerCount)
    }
}

public enum FusedHybridDecodeBlockKernelSet {
    public struct CompileSpec: Sendable {
        public let layerCount: Int
        public let milText: String
        public let weights: [(path: String, data: Data)]
        public let inputSizes: [Int]
        public let outputSizes: [Int]
        public let usedRealWeights: Bool

        public var weightBlobBytes: Int {
            weights.reduce(0) { $0 + $1.data.count }
        }
    }

    public static func makeQwen15BSpec(
        layerCount: Int,
        weightBlobs: [(path: String, data: Data)],
        usedRealWeights: Bool
    ) -> CompileSpec {
        let generator = FusedHybridDecodeBlockGenerator.qwen15B(layerCount: layerCount)
        return CompileSpec(
            layerCount: layerCount,
            milText: generator.milText,
            weights: weightBlobs,
            inputSizes: generator.inputByteSizes,
            outputSizes: generator.outputByteSizes,
            usedRealWeights: usedRealWeights
        )
    }

    public static func dummyQwen15BWeightBlobs(layerCount: Int, value: Float = 0.01) -> [(path: String, data: Data)] {
        let dim = FusedHybridDecodeBlockGenerator.Qwen15BShape.dModel
        let qDim = FusedHybridDecodeBlockGenerator.Qwen15BShape.qDim
        let kvDim = FusedHybridDecodeBlockGenerator.Qwen15BShape.kvDim
        let hidden = FusedHybridDecodeBlockGenerator.Qwen15BShape.hiddenDim
        var blobs: [(path: String, data: Data)] = []
        blobs.reserveCapacity(layerCount * 11)
        for layer in 0..<layerCount {
            let prefix = "l\(layer)"
            blobs.append(contentsOf: [
                ("@model_path/weights/\(prefix)_rms1.bin", filledBlob(count: dim, value: 1)),
                ("@model_path/weights/\(prefix)_wq.bin", filledBlob(rows: qDim, cols: dim, value: value)),
                ("@model_path/weights/\(prefix)_wk.bin", filledBlob(rows: kvDim, cols: dim, value: value)),
                ("@model_path/weights/\(prefix)_wv.bin", filledBlob(rows: kvDim, cols: dim, value: value)),
                ("@model_path/weights/\(prefix)_bq.bin", filledBlob(count: qDim, value: value)),
                ("@model_path/weights/\(prefix)_bk.bin", filledBlob(count: kvDim, value: value)),
                ("@model_path/weights/\(prefix)_bv.bin", filledBlob(count: kvDim, value: value)),
                ("@model_path/weights/\(prefix)_rms2.bin", filledBlob(count: dim, value: 1)),
                ("@model_path/weights/\(prefix)_w1.bin", filledBlob(rows: hidden, cols: dim, value: value)),
                ("@model_path/weights/\(prefix)_w3.bin", filledBlob(rows: hidden, cols: dim, value: value)),
                ("@model_path/weights/\(prefix)_w2.bin", filledBlob(rows: dim, cols: hidden, value: value)),
            ])
        }
        return blobs
    }

    public static func loadPackedQwen15BWeightBlobs(
        layerCount: Int,
        nativeDir: String
    ) throws -> [(path: String, data: Data)] {
        var blobs: [(path: String, data: Data)] = []
        blobs.reserveCapacity(layerCount * 11)
        for layer in 0..<layerCount {
            let layerDir = URL(fileURLWithPath: nativeDir, isDirectory: true)
                .appendingPathComponent("layers", isDirectory: true)
                .appendingPathComponent(String(layer), isDirectory: true)
            let prefix = "l\(layer)"
            let files: [(String, String)] = [
                ("\(prefix)_rms1.bin", "rms_att.bin"),
                ("\(prefix)_wq.bin", "wq.bin"),
                ("\(prefix)_wk.bin", "wk.bin"),
                ("\(prefix)_wv.bin", "wv.bin"),
                ("\(prefix)_bq.bin", "bq.bin"),
                ("\(prefix)_bk.bin", "bk.bin"),
                ("\(prefix)_bv.bin", "bv.bin"),
                ("\(prefix)_rms2.bin", "rms_ffn.bin"),
                ("\(prefix)_w1.bin", "w1.bin"),
                ("\(prefix)_w3.bin", "w3.bin"),
                ("\(prefix)_w2.bin", "w2.bin"),
            ]
            for (blobName, fileName) in files {
                let url = layerDir.appendingPathComponent(fileName)
                let data = try Data(contentsOf: url)
                guard !data.isEmpty else {
                    throw ANEError.invalidArguments("Empty packed weight \(url.path)")
                }
                blobs.append(("@model_path/weights/\(blobName)", data))
            }
        }
        return blobs
    }

    public static func compile(_ spec: CompileSpec) -> FusedHybridDecodeBlockCompileReport {
        let start = DispatchTime.now()
        do {
            let kernel = try ANEKernel(
                milText: spec.milText,
                weights: spec.weights,
                inputSizes: spec.inputSizes,
                outputSizes: spec.outputSizes,
                compileLabel: "fused.hybrid.block.n\(spec.layerCount)"
            )
            let compileMs = milliseconds(since: start)
            let hexId = kernel.hexId
            let programCount = countANEPrograms(hexId: hexId)
            return FusedHybridDecodeBlockCompileReport(
                layerCount: spec.layerCount,
                passed: true,
                compileMs: compileMs,
                programCount: programCount,
                weightBlobBytes: spec.weightBlobBytes,
                hexId: hexId,
                milDumpPath: nil,
                failingOp: nil,
                usedRealWeights: spec.usedRealWeights,
                errorDescription: nil
            )
        } catch {
            let compileMs = milliseconds(since: start)
            let milPath = dumpFailedMIL(spec)
            return FusedHybridDecodeBlockCompileReport(
                layerCount: spec.layerCount,
                passed: false,
                compileMs: compileMs,
                programCount: 0,
                weightBlobBytes: spec.weightBlobBytes,
                hexId: "",
                milDumpPath: milPath,
                failingOp: failingOpHint(from: error, milText: spec.milText),
                usedRealWeights: spec.usedRealWeights,
                errorDescription: "\(error)"
            )
        }
    }

    private static func filledBlob(count: Int, value: Float) -> Data {
        WeightBlob.build(from: [Float](repeating: value, count: count), rows: 1, cols: count)
    }

    private static func filledBlob(rows: Int, cols: Int, value: Float) -> Data {
        WeightBlob.build(from: [Float](repeating: value, count: rows * cols), rows: rows, cols: cols)
    }

    private static func milliseconds(since start: DispatchTime) -> Double {
        Double(DispatchTime.now().uptimeNanoseconds - start.uptimeNanoseconds) / 1_000_000
    }

    /// One MIL program compiled to one ANE kernel. Extra files in the hex-tmp
    /// tree (net.plist, compiled weights) are not extra hops.
    private static func countANEPrograms(hexId: String) -> Int {
        guard !hexId.isEmpty else { return 0 }
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent(hexId)
        guard let items = try? FileManager.default.contentsOfDirectory(atPath: dir.path) else {
            return 1
        }
        let programs = items.filter { name in
            name.hasSuffix(".hwx") || name == "model.hwx" || name.hasPrefix("program")
        }
        return max(1, programs.count)
    }

    private static func dumpFailedMIL(_ spec: CompileSpec) -> String {
        let stamp = Int(Date().timeIntervalSince1970)
        let filename = "espresso-fused-hybrid-n\(spec.layerCount)-\(stamp).mil"
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(filename)
        try? spec.milText.write(to: url, atomically: true, encoding: .utf8)
        return url.path
    }

    private static func failingOpHint(from error: Error, milText: String) -> String {
        let text = "\(error)"
        if text.contains("InvalidMIL") || text.lowercased().contains("invalid mil") {
            return "InvalidMILProgram"
        }
        if text.contains("ANE kernel compilation failed") {
            if milText.contains("8960") {
                return "ANE kernel compilation failed (likely weight/SRAM or fused QKV+FFN at 1536/8960)"
            }
            return "ANE kernel compilation failed"
        }
        return text
    }
}
