import Darwin
import EdgeRunnerIO
import EspressoEdgeRunner
import Foundation
import Metal
import ModelSupport
import Testing
@testable import RealModelInference

@Suite("QwenGGUFSidecarComparison")
struct QwenGGUFSidecarComparisonTests {
    private struct IndexedDifference: Sendable, Equatable {
        let index: Int
        let rawValue: Float
        let artifactValue: Float
        let absDiff: Float
    }

    private struct TensorComparisonStats: Sendable, Equatable {
        let rawShape: [Int]
        let rawCount: Int
        let artifactCount: Int
        let maxAbsDiff: Float
        let meanAbsDiff: Float
        let cosineSimilarity: Float
        let firstMismatches: [IndexedDifference]
    }

    @Test("Tensor comparison stats report first mismatches and cosine")
    func tensorComparisonStatsReportExpectedMetrics() {
        let stats = compareTensorData(
            rawShape: [2, 2],
            raw: [1, 2, 3, 4],
            artifact: [1, 2.25, 2.5, 4],
            mismatchThreshold: 0.2,
            maxReportedMismatches: 4
        )

        #expect(stats.rawShape == [2, 2])
        #expect(stats.rawCount == 4)
        #expect(stats.artifactCount == 4)
        #expect(abs(stats.maxAbsDiff - 0.5) < 1e-6)
        #expect(abs(stats.meanAbsDiff - 0.1875) < 1e-6)
        #expect(abs(stats.cosineSimilarity - 0.9950577) < 1e-5)
        #expect(stats.firstMismatches.count == 2)
        #expect(stats.firstMismatches[0] == IndexedDifference(index: 1, rawValue: 2, artifactValue: 2.25, absDiff: 0.25))
        #expect(stats.firstMismatches[1] == IndexedDifference(index: 2, rawValue: 3, artifactValue: 2.5, absDiff: 0.5))
    }

    @Test("GGUF tensor names map to exact float32 sidecar paths")
    func ggufTensorMapsToExactSidecarPath() throws {
        let root = URL(fileURLWithPath: "/tmp/example-artifact", isDirectory: true)

        let layerPath = try exactFloat32SidecarURL(
            weightDir: root,
            ggufName: "blk.27.attn_v.weight",
            architecture: "qwen3"
        )
        #expect(layerPath.path == "/tmp/example-artifact/layers/27/wv.float32.bin")

        let topLevelPath = try exactFloat32SidecarURL(
            weightDir: root,
            ggufName: "output.weight",
            architecture: "qwen3"
        )
        #expect(topLevelPath.path == "/tmp/example-artifact/lm_head.float32.bin")
    }

    @Test("Raw GGUF LM head selection prefers output weight when present")
    func rawGGUFLMHeadSelectionPrefersExplicitOutputWeight() throws {
        let device = try makeDevice()
        let buffer = device.makeBuffer(length: MemoryLayout<Float>.stride, options: .storageModeShared)!

        var weightMap = WeightMap()
        weightMap["output.weight"] = TensorStorage(
            buffer: buffer,
            byteOffset: 0,
            dataType: .float32,
            shape: [1],
            name: "output.weight"
        )
        weightMap["token_embd.weight"] = TensorStorage(
            buffer: buffer,
            byteOffset: 0,
            dataType: .float32,
            shape: [1],
            name: "token_embd.weight"
        )

        #expect(QwenGGUFVerificationSupport.rawGGUFLMHeadTensorName(from: weightMap) == "output.weight")
    }

    @Test("Raw GGUF LM head selection falls back to tied token embeddings")
    func rawGGUFLMHeadSelectionFallsBackToTokenEmbeddings() throws {
        let device = try makeDevice()
        let buffer = device.makeBuffer(length: MemoryLayout<Float>.stride, options: .storageModeShared)!

        var weightMap = WeightMap()
        weightMap["token_embd.weight"] = TensorStorage(
            buffer: buffer,
            byteOffset: 0,
            dataType: .float32,
            shape: [1],
            name: "token_embd.weight"
        )

        #expect(QwenGGUFVerificationSupport.rawGGUFLMHeadTensorName(from: weightMap) == "token_embd.weight")
    }

    @Test("Exact CPU decode keeps fp32 intermediates by default")
    func exactCPUDecodeKeepsFP32IntermediatesByDefault() {
        #expect(!RealModelInferenceEngine.shouldRoundCPUExactDecodeIntermediatesToFP16(env: [:]))
        #expect(RealModelInferenceEngine.shouldRoundCPUExactDecodeIntermediatesToFP16(env: [
            "ESPRESSO_DEBUG_CPU_EXACT_DECODE_KEEP_FP32_INTERMEDIATES": "0"
        ]))
    }

    @Test("Exact CPU decode env accepts explicit fp32 keep values")
    func exactCPUDecodeEnvAcceptsExplicitFP32KeepValues() {
        #expect(!RealModelInferenceEngine.shouldRoundCPUExactDecodeIntermediatesToFP16(env: [
            "ESPRESSO_DEBUG_CPU_EXACT_DECODE_KEEP_FP32_INTERMEDIATES": "1"
        ]))
        #expect(!RealModelInferenceEngine.shouldRoundCPUExactDecodeIntermediatesToFP16(env: [
            "ESPRESSO_DEBUG_CPU_EXACT_DECODE_KEEP_FP32_INTERMEDIATES": "true"
        ]))
    }

    @Test("Exact CPU testing helper rejects empty prompt tokens")
    func exactCPUNextTokenTestingHelperRejectsEmptyPrompt() {
        let config = MultiModelConfig(
            name: "qwen3",
            nLayer: 1,
            nHead: 1,
            nKVHead: 1,
            dModel: 4,
            headDim: 4,
            hiddenDim: 8,
            vocab: 16,
            maxSeq: 16,
            normEps: 1e-5,
            ropeTheta: 10_000,
            eosToken: nil,
            architecture: .llama
        )

        #expect(throws: RealModelInferenceError.self) {
            _ = try RealModelInferenceEngine.generateNextTokenExactCPUForTesting(
                config: config,
                weightDir: "/tmp/does-not-matter",
                promptTokens: []
            )
        }
    }

    @Test("Exact CPU token-sequence helper rejects non-positive max tokens")
    func exactCPUTokenSequenceTestingHelperRejectsNonPositiveMaxTokens() {
        let config = MultiModelConfig(
            name: "qwen3",
            nLayer: 1,
            nHead: 1,
            nKVHead: 1,
            dModel: 4,
            headDim: 4,
            hiddenDim: 8,
            vocab: 16,
            maxSeq: 16,
            normEps: 1e-5,
            ropeTheta: 10_000,
            eosToken: nil,
            architecture: .llama
        )

        #expect(throws: RealModelInferenceError.self) {
            _ = try RealModelInferenceEngine.generateTokensExactCPUForTesting(
                config: config,
                weightDir: "/tmp/does-not-matter",
                promptTokens: [9707],
                maxTokens: 0
            )
        }
    }

    @Test(
        "Debug compare raw GGUF tensors against artifact float32 sidecars",
        .enabled(
            if: ProcessInfo.processInfo.environment["ESPRESSO_DEBUG_COMPARE_RAW_GGUF_SIDECARS"] == "1"
                && ProcessInfo.processInfo.environment["ESPRESSO_DEBUG_GGUF_MODEL"] != nil
                && ProcessInfo.processInfo.environment["ESPRESSO_DEBUG_WEIGHT_DIR"] != nil
        )
    )
    func debugCompareRawGGUFTensorsAgainstArtifactSidecars() async throws {
        let env = ProcessInfo.processInfo.environment
        guard let ggufModel = env["ESPRESSO_DEBUG_GGUF_MODEL"], !ggufModel.isEmpty else {
            Issue.record("ESPRESSO_DEBUG_GGUF_MODEL is required")
            return
        }
        guard let weightDir = env["ESPRESSO_DEBUG_WEIGHT_DIR"], !weightDir.isEmpty else {
            Issue.record("ESPRESSO_DEBUG_WEIGHT_DIR is required")
            return
        }
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw NSError(domain: "QwenGGUFSidecarComparisonTests", code: 1, userInfo: [
                NSLocalizedDescriptionKey: "Metal device unavailable for GGUF dequantization"
            ])
        }

        let ggufURL = URL(fileURLWithPath: ggufModel)
        let artifactURL = URL(fileURLWithPath: weightDir, isDirectory: true)
        let loader = try GGUFLoader(url: ggufURL)
        let weightMap = try await loader.load(from: ggufURL)
        let architecture = loader.modelConfig.architectureName
        let tensorNames = debugTensorNames(
            env: env,
            defaultNames: [
                "blk.27.ffn_down.weight",
                "blk.27.attn_q.weight",
                "blk.27.attn_k.weight",
                "blk.27.attn_v.weight",
            ]
        )

        for tensorName in tensorNames {
            guard let rawTensor = weightMap[tensorName] else {
                Issue.record("Missing raw GGUF tensor: \(tensorName)")
                continue
            }

            let raw = try await DequantDispatcher.dequantize(tensor: rawTensor, device: device)
            let sidecarURL = try exactFloat32SidecarURL(
                weightDir: artifactURL,
                ggufName: tensorName,
                architecture: architecture
            )
            let artifact = try readFloat32Array(at: sidecarURL)
            let stats = compareTensorData(rawShape: rawTensor.shape, raw: raw, artifact: artifact)

            #expect(stats.rawCount == rawTensor.elementCount)
            #expect(stats.artifactCount == artifact.count)

            let mismatchSummary = if stats.firstMismatches.isEmpty {
                "[]"
            } else {
                "[" + stats.firstMismatches.map {
                    "{index=\($0.index), raw=\($0.rawValue), artifact=\($0.artifactValue), absDiff=\($0.absDiff)}"
                }.joined(separator: ", ") + "]"
            }

            fputs(
                "[qwen-sidecar-compare] tensor=\(tensorName) shape=\(stats.rawShape) rawCount=\(stats.rawCount) artifactCount=\(stats.artifactCount) maxAbsDiff=\(stats.maxAbsDiff) meanAbsDiff=\(stats.meanAbsDiff) cosine=\(stats.cosineSimilarity) firstMismatches=\(mismatchSummary)\n",
                stderr
            )
        }
    }

    private func debugTensorNames(
        env: [String: String],
        defaultNames: [String]
    ) -> [String] {
        guard let raw = env["ESPRESSO_DEBUG_TENSOR_NAMES"], !raw.isEmpty else {
            return defaultNames
        }
        let parsed = raw
            .split(separator: ",")
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
        return parsed.isEmpty ? defaultNames : parsed
    }

    private func exactFloat32SidecarURL(
        weightDir: URL,
        ggufName: String,
        architecture: String
    ) throws -> URL {
        guard let relativePath = EspressoTensorNameMapper.espressoPath(for: ggufName, architecture: architecture) else {
            throw NSError(domain: "QwenGGUFSidecarComparisonTests", code: 2, userInfo: [
                NSLocalizedDescriptionKey: "Unmapped GGUF tensor name: \(ggufName)"
            ])
        }

        let sidecarRelativePath: String
        if relativePath.hasSuffix(".bin") {
            sidecarRelativePath = String(relativePath.dropLast(4)) + ".float32.bin"
        } else {
            sidecarRelativePath = relativePath + ".float32"
        }
        return weightDir.appendingPathComponent(sidecarRelativePath)
    }

    private func readFloat32Array(at url: URL) throws -> [Float] {
        let data = try Data(contentsOf: url)
        let scalarSize = MemoryLayout<UInt32>.stride
        guard data.count.isMultiple(of: scalarSize) else {
            throw NSError(domain: "QwenGGUFSidecarComparisonTests", code: 3, userInfo: [
                NSLocalizedDescriptionKey: "Invalid float32 sidecar byte count at \(url.path)"
            ])
        }
        return data.withUnsafeBytes { raw in
            stride(from: 0, to: data.count, by: scalarSize).map { index in
                let bits = raw.loadUnaligned(fromByteOffset: index, as: UInt32.self)
                return Float(bitPattern: UInt32(littleEndian: bits))
            }
        }
    }

    private func makeDevice() throws -> MTLDevice {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw NSError(domain: "QwenGGUFSidecarComparisonTests", code: 4, userInfo: [
                NSLocalizedDescriptionKey: "Metal device unavailable"
            ])
        }
        return device
    }

    private func compareTensorData(
        rawShape: [Int],
        raw: [Float],
        artifact: [Float],
        mismatchThreshold: Float = 1e-5,
        maxReportedMismatches: Int = 8
    ) -> TensorComparisonStats {
        let overlapCount = min(raw.count, artifact.count)
        var maxAbsDiff: Float = 0
        var sumAbsDiff: Double = 0
        var dot: Double = 0
        var rawNorm: Double = 0
        var artifactNorm: Double = 0
        var firstMismatches: [IndexedDifference] = []

        if overlapCount > 0 {
            for index in 0..<overlapCount {
                let rawValue = raw[index]
                let artifactValue = artifact[index]
                let absDiff = abs(rawValue - artifactValue)
                maxAbsDiff = max(maxAbsDiff, absDiff)
                sumAbsDiff += Double(absDiff)
                dot += Double(rawValue) * Double(artifactValue)
                rawNorm += Double(rawValue) * Double(rawValue)
                artifactNorm += Double(artifactValue) * Double(artifactValue)

                if absDiff > mismatchThreshold, firstMismatches.count < maxReportedMismatches {
                    firstMismatches.append(
                        IndexedDifference(
                            index: index,
                            rawValue: rawValue,
                            artifactValue: artifactValue,
                            absDiff: absDiff
                        )
                    )
                }
            }
        }

        let cosineSimilarity: Float
        if rawNorm == 0 || artifactNorm == 0 {
            cosineSimilarity = rawNorm == artifactNorm ? 1 : 0
        } else {
            cosineSimilarity = Float(dot / (sqrt(rawNorm) * sqrt(artifactNorm)))
        }

        return TensorComparisonStats(
            rawShape: rawShape,
            rawCount: raw.count,
            artifactCount: artifact.count,
            maxAbsDiff: maxAbsDiff,
            meanAbsDiff: overlapCount == 0 ? 0 : Float(sumAbsDiff / Double(overlapCount)),
            cosineSimilarity: cosineSimilarity,
            firstMismatches: firstMismatches
        )
    }
}
