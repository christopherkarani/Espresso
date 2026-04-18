import ANEGraphIR
import ANETypes
import Darwin
import EdgeRunnerIO
import EspressoEdgeRunner
import Foundation
import Metal
import ModelSupport
import Testing
@testable import RealModelInference

@Test func test_buildPipelineConfigValidation() throws {
    let config = makeTinyGPT2Config()
    let tokenizerDir = try makeGPT2TokenizerDirectory()
    let validWeightDir = try makeMinimalGPT2WeightDirectory(config: config)
    let missingWeightDir = FileManager.default.temporaryDirectory
        .appendingPathComponent(UUID().uuidString)
        .path
    let missingTokenizerDir = FileManager.default.temporaryDirectory
        .appendingPathComponent(UUID().uuidString)
        .path

    try expectRealModelInferenceError(containing: missingWeightDir) {
        _ = try RealModelInferenceEngine.build(
            config: config,
            weightDir: missingWeightDir,
            tokenizerDir: tokenizerDir.path
        )
    }

    try expectRealModelInferenceError(containing: missingTokenizerDir) {
        _ = try RealModelInferenceEngine.build(
            config: config,
            weightDir: validWeightDir.path,
            tokenizerDir: missingTokenizerDir
        )
    }

    let missingHeadDir = try makeMinimalGPT2WeightDirectory(
        config: config,
        topLevelNaming: .canonical,
        omitTopLevelFiles: ["lmHead"]
    )
    try expectRealModelInferenceError(containing: "lm_head") {
        _ = try RealModelInferenceEngine.build(
            config: config,
            weightDir: missingHeadDir.path,
            tokenizerDir: tokenizerDir.path
        )
    }

    let invalidConfig = MultiModelConfig(
        name: "invalid-gpt2",
        nLayer: 1,
        nHead: 2,
        nKVHead: 2,
        dModel: 9,
        headDim: 4,
        hiddenDim: 32,
        vocab: 64,
        maxSeq: 8,
        normEps: 1e-5,
        architecture: .gpt2
    )
    try expectRealModelInferenceError(containing: "dmodel must equal nhead * headdim") {
        _ = try RealModelInferenceEngine.build(
            config: invalidConfig,
            weightDir: validWeightDir.path,
            tokenizerDir: tokenizerDir.path
        )
    }

    // Llama is now supported. With a weight dir that lacks llama-specific files,
    // the build should fail on missing layer weights (rms_att.bin etc) or succeed
    // if the GPT-2 dir happens to have compatible top-level files.
    // With a completely empty weight dir, it should fail on missing token embedding.
    let llamaTokenizerDir = try makeSentencePieceTokenizerDirectory()
    let llamaConfig = MultiModelConfig(
        name: "tiny-llama",
        nLayer: 1,
        nHead: 2,
        nKVHead: 2,
        dModel: 8,
        headDim: 4,
        hiddenDim: 16,
        vocab: 64,
        maxSeq: 8,
        normEps: 1e-5,
        architecture: .llama
    )
    let emptyLlamaDir = try makeTempDirectory()
    try expectRealModelInferenceError(containing: "token embedding") {
        _ = try RealModelInferenceEngine.build(
            config: llamaConfig,
            weightDir: emptyLlamaDir.path,
            tokenizerDir: llamaTokenizerDir.deletingLastPathComponent().path
        )
    }
}

@Test func test_buildRejectsLFM2WithoutLayerMetadata() throws {
    let tokenizerDir = try makeSentencePieceTokenizerDirectory()
    let weightDir = try makeTempDirectory()
    defer {
        try? FileManager.default.removeItem(at: tokenizerDir.deletingLastPathComponent())
        try? FileManager.default.removeItem(at: weightDir)
    }

    let config = MultiModelConfig(
        name: "invalid-lfm2",
        nLayer: 16,
        nHead: 16,
        nKVHead: 8,
        dModel: 1024,
        headDim: 64,
        hiddenDim: 4608,
        vocab: 65_536,
        maxSeq: 4_096,
        normEps: 1e-5,
        architecture: .lfm2
    )

    try expectRealModelInferenceError(containing: "layerTypes") {
        _ = try RealModelInferenceEngine.build(
            config: config,
            weightDir: weightDir.path,
            tokenizerDir: tokenizerDir.deletingLastPathComponent().path
        )
    }
}

@Test func test_spatialBucketSelection() {
    #expect(RealModelInferenceEngine.spatialBucket(for: 1, maxSeq: 16) == 1)
    #expect(RealModelInferenceEngine.spatialBucket(for: 2, maxSeq: 16) == 2)
    #expect(RealModelInferenceEngine.spatialBucket(for: 3, maxSeq: 16) == 4)
    #expect(RealModelInferenceEngine.spatialBucket(for: 7, maxSeq: 16) == 8)
    #expect(RealModelInferenceEngine.spatialBucket(for: 9, maxSeq: 16) == 16)
    #expect(RealModelInferenceEngine.spatialBucket(for: 17, maxSeq: 16) == 16)
    #expect(RealModelInferenceEngine.spatialBucket(for: 513, maxSeq: 1_024) == 1_024)
    #expect(RealModelInferenceEngine.minimumCompileSpatial(channels: 768) == 32)
    #expect(RealModelInferenceEngine.minimumCompileSpatial(channels: 1_536) == 16)
    #expect(RealModelInferenceEngine.minimumCompileSpatial(channels: 4_096) == 8)
    #expect(RealModelInferenceEngine.incrementalHeadSpatial(channels: 768) == 32)
}

@Test func test_milDeploymentTargetDefaultsToIOS18() {
    #expect(RealModelInferenceEngine.milDeploymentTarget(environment: [:]) == "ios18")
}

@Test func test_requireANEHardwareTestsEnabledUsesInjectedEnvironment() throws {
    try expectRealModelInferenceError(containing: "ANE_HARDWARE_TESTS=1") {
        try RealModelInferenceEngine.requireANEHardwareTestsEnabled(environment: [:])
    }
}

@Test func test_milDeploymentTargetReadsEnvironmentOverride() {
    #expect(
        RealModelInferenceEngine.milDeploymentTarget(
            environment: ["ESPRESSO_MIL_DEPLOYMENT_TARGET": "macos26"]
        ) == "macos26"
    )
}

@Test func test_gpt2NormKindDefaultsToLayerNorm() {
    #expect(RealModelInferenceEngine.gpt2NormKind(environment: [:]) == .layerNorm)
}

@Test func test_gpt2NormKindSupportsRMSNormOverrides() {
    #expect(
        RealModelInferenceEngine.gpt2NormKind(
            environment: ["ESPRESSO_GPT2_NORM": "rmsnorm"]
        ) == .rmsNorm
    )
    #expect(
        RealModelInferenceEngine.gpt2NormKind(
            environment: ["ESPRESSO_GPT2_USE_RMS_NORM": "1"]
        ) == .rmsNorm
    )
}

@Test func test_hybridGreedyHeadModeKeepsGPT2OnNormThenClassifier() {
    #expect(
        RealModelInferenceEngine.hybridGreedyHeadMode(
            config: makeTinyGPT2Config(),
            hasFactoredOutputHead: true,
            environment: ["ESPRESSO_ENABLE_LLAMA_HYBRID_FUSED_EXACT_HEAD": "1"]
        ) == .normThenClassifier
    )
}

@Test func test_hybridGreedyHeadModeSelectsLlamaFactoredAndFusedModes() {
    let llamaConfig = MultiModelConfig(
        name: "tiny-llama",
        nLayer: 2,
        nHead: 2,
        nKVHead: 2,
        dModel: 8,
        headDim: 4,
        hiddenDim: 16,
        vocab: 64,
        maxSeq: 8,
        normEps: 1e-5,
        architecture: .llama
    )

    #expect(
        RealModelInferenceEngine.hybridGreedyHeadMode(
            config: llamaConfig,
            hasFactoredOutputHead: true,
            environment: [:]
        ) == .classifierOnlyFactored
    )
    #expect(
        RealModelInferenceEngine.hybridGreedyHeadMode(
            config: llamaConfig,
            hasFactoredOutputHead: false,
            environment: ["ESPRESSO_ENABLE_LLAMA_HYBRID_FUSED_EXACT_HEAD": "1"]
        ) == .classifierOnlyFused
    )
    #expect(
        RealModelInferenceEngine.hybridGreedyHeadMode(
            config: llamaConfig,
            hasFactoredOutputHead: false,
            environment: [:]
        ) == .normThenClassifier
    )
}

@Test func test_emitGPT2AttentionMILForTestingSupportsDeploymentTargetOverride() throws {
    let config = makeTinyGPT2Config()
    let weightDir = try makeMinimalGPT2WeightDirectory(config: config)
    defer { try? FileManager.default.removeItem(at: weightDir) }

    let mil = try RealModelInferenceEngine.emitGPT2AttentionMILForTesting(
        config: config,
        weightDir: weightDir.path,
        spatial: 8,
        environment: ["ESPRESSO_MIL_DEPLOYMENT_TARGET": "ios19"]
    )

    #expect(mil.contains("func main<ios19>("))
}

@Test func test_emitGPT2AttentionMILForTestingSupportsRMSNormExperiment() throws {
    let config = makeTinyGPT2Config()
    let weightDir = try makeMinimalGPT2WeightDirectory(config: config)
    defer { try? FileManager.default.removeItem(at: weightDir) }

    let mil = try RealModelInferenceEngine.emitGPT2AttentionMILForTesting(
        config: config,
        weightDir: weightDir.path,
        spatial: 8,
        environment: ["ESPRESSO_GPT2_NORM": "rmsnorm"]
    )

    #expect(mil.contains("_ln1_ss = reduce_sum("))
    #expect(!mil.contains("_ln1_mean = reduce_mean("))
}

@Test func test_debugReferenceSiluMatchesExactTanhForm() {
    let samples: [Float] = [-8, -1.25, 0, 0.75, 6]
    for sample in samples {
        let expected = 0.5 * sample * (1 + tanh(0.5 * sample))
        #expect(abs(silu(sample) - expected) < 1e-6)
    }
}

@Test func test_generateNextTokenForTestingRejectsEmptyPromptTokens() throws {
    let config = makeTinyGPT2Config()
    try expectRealModelInferenceError(containing: "must not be empty") {
        _ = try RealModelInferenceEngine.generateNextTokenForTesting(
            config: config,
            weightDir: "/nonexistent",
            promptTokens: []
        )
    }
}

@Test func test_debugPromptTokensPrefersEnvironmentOverride() {
    #expect(
        debugPromptTokens(
            env: ["ESPRESSO_DEBUG_PROMPT_TOKENS": "1, 2,3"],
            defaultTokens: [9707, 21806]
        ) == [1, 2, 3]
    )
    #expect(debugPromptTokens(env: [:], defaultTokens: [9707, 21806]) == [9707, 21806])
}

@Test func test_resolveExactTwoTokenDraftWeightDirForTestingLoadsBundleRelativeDraftDescriptor() throws {
    let bundleRoot = try makeTempDirectory()
    defer { try? FileManager.default.removeItem(at: bundleRoot) }

    let weightDir = bundleRoot.appendingPathComponent("weights", isDirectory: true)
    let draftDir = weightDir.appendingPathComponent("draft/student", isDirectory: true)
    try FileManager.default.createDirectory(at: draftDir, withIntermediateDirectories: true)

    let config = MultiModelConfig(
        name: "stories110m",
        nLayer: 12,
        nHead: 12,
        nKVHead: 12,
        dModel: 768,
        headDim: 64,
        hiddenDim: 2048,
        vocab: 32_000,
        maxSeq: 256,
        normEps: 1e-5,
        architecture: .llama
    )
    let draftMetadata: [String: Any] = [
        "name": "stories110m-stable-copy",
        "nLayer": 12,
        "nHead": 12,
        "nKVHead": 12,
        "dModel": 768,
        "headDim": 64,
        "hiddenDim": 2048,
        "vocab": 32_000,
        "maxSeq": 256,
        "normEps": 1e-5,
        "ropeTheta": 10_000.0,
        "architecture": "llama",
    ]
    let draftMetadataData = try JSONSerialization.data(withJSONObject: draftMetadata, options: [.sortedKeys])
    try draftMetadataData.write(to: draftDir.appendingPathComponent("metadata.json"))

    let descriptor: [String: Any] = [
        "model_dir": "draft/student",
        "tokenizer_dir": "tokenizer",
        "model_id": "stories110m-stable-copy",
    ]
    let descriptorData = try JSONSerialization.data(withJSONObject: descriptor, options: [.sortedKeys])
    let descriptorURL = weightDir.appendingPathComponent("draft/exact-two-token.json")
    try FileManager.default.createDirectory(at: descriptorURL.deletingLastPathComponent(), withIntermediateDirectories: true)
    try descriptorData.write(to: descriptorURL)

    let resolved = try RealModelInferenceEngine.resolveExactTwoTokenDraftWeightDirForTesting(
        config: config,
        weightDirURL: weightDir,
        environment: [
            "ESPRESSO_BUNDLE_DRAFT_KIND": "exact_two_token",
            "ESPRESSO_BUNDLE_DRAFT_HORIZON": "2",
            "ESPRESSO_BUNDLE_DRAFT_ARTIFACT_REF": "weights/draft/exact-two-token.json",
        ]
    )
    #expect(resolved == draftDir.path)
}

@Test func test_exactTwoTokenDraftCPUFallbackReportsPassMetrics() throws {
    let bundleRoot = try makeTempDirectory()
    let tokenizerModel = try makeSentencePieceTokenizerDirectory()
    defer {
        try? FileManager.default.removeItem(at: bundleRoot)
        try? FileManager.default.removeItem(at: tokenizerModel.deletingLastPathComponent())
    }

    let baseConfig = makeTinyLlamaConfig()
    let config = MultiModelConfig(
        name: baseConfig.name,
        nLayer: baseConfig.nLayer,
        nHead: baseConfig.nHead,
        nKVHead: baseConfig.nKVHead,
        dModel: baseConfig.dModel,
        headDim: baseConfig.headDim,
        hiddenDim: baseConfig.hiddenDim,
        vocab: baseConfig.vocab,
        maxSeq: 4_096,
        normEps: baseConfig.normEps,
        ropeTheta: baseConfig.ropeTheta,
        eosToken: baseConfig.eosToken,
        architecture: baseConfig.architecture
    )
    let weightDir = bundleRoot.appendingPathComponent("weights", isDirectory: true)
    let draftDir = weightDir.appendingPathComponent("draft/student", isDirectory: true)
    try populateMinimalLlamaWeightDirectory(at: weightDir, config: config)
    try populateMinimalLlamaWeightDirectory(at: draftDir, config: config)

    let descriptor: [String: Any] = [
        "model_dir": "draft/student",
        "tokenizer_dir": "tokenizer",
        "model_id": "tiny-llama-draft",
    ]
    let descriptorURL = bundleRoot.appendingPathComponent("weights/draft/exact-two-token.json")
    try FileManager.default.createDirectory(
        at: descriptorURL.deletingLastPathComponent(),
        withIntermediateDirectories: true
    )
    try JSONSerialization.data(withJSONObject: descriptor, options: [.sortedKeys]).write(to: descriptorURL)

    let keys = [
        "ESPRESSO_USE_CPU_EXACT_DECODE",
        "ESPRESSO_FORCE_HYBRID_DECODE",
        "ESPRESSO_BUNDLE_DRAFT_KIND",
        "ESPRESSO_BUNDLE_DRAFT_HORIZON",
        "ESPRESSO_BUNDLE_DRAFT_ARTIFACT_REF",
    ]
    let previous = Dictionary(uniqueKeysWithValues: keys.map { ($0, ProcessInfo.processInfo.environment[$0]) })
    setenv("ESPRESSO_USE_CPU_EXACT_DECODE", "1", 1)
    unsetenv("ESPRESSO_FORCE_HYBRID_DECODE")
    setenv("ESPRESSO_BUNDLE_DRAFT_KIND", "exact_two_token", 1)
    setenv("ESPRESSO_BUNDLE_DRAFT_HORIZON", "2", 1)
    setenv("ESPRESSO_BUNDLE_DRAFT_ARTIFACT_REF", "weights/draft/exact-two-token.json", 1)
    defer {
        for (key, value) in previous {
            if let value {
                setenv(key, value, 1)
            } else {
                unsetenv(key)
            }
        }
    }

    var engine = try RealModelInferenceEngine.build(
        config: config,
        weightDir: weightDir.path,
        tokenizerDir: tokenizerModel.deletingLastPathComponent().path
    )
    let result = try engine.generate(prompt: "Hello", maxTokens: 2, temperature: 0.0)

    #expect(result.exactHeadBackend == "cpu_exact_two_token_draft")
    #expect(result.committedExactTokensPerPass != nil)
    #expect(result.acceptedFutureTokensPerPass != nil)
}

private func shouldRunLegacyQwenExperimentTests(
    env: [String: String] = ProcessInfo.processInfo.environment
) -> Bool {
    env["ESPRESSO_ENABLE_QWEN_EXPERIMENT_TESTS"] == "1"
}

@Test func test_loadWeightTablePreferringFloat32SidecarUsesSidecarWhenPresent() throws {
    let root = try makeTempDirectory()
    defer { try? FileManager.default.removeItem(at: root) }

    let blobPath = root.appendingPathComponent("lm_head.bin")
    try writeBlob(values: [1, 2, 3], to: blobPath)

    let exactValues: [Float] = [1.0003, -2.0007, 3.14159]
    let sidecarPath = root.appendingPathComponent("lm_head.float32.bin")
    var data = Data(capacity: exactValues.count * MemoryLayout<UInt32>.stride)
    for value in exactValues {
        var bits = value.bitPattern.littleEndian
        withUnsafeBytes(of: &bits) { data.append(contentsOf: $0) }
    }
    try data.write(to: sidecarPath)

    let loaded = try RealModelInferenceEngine.loadWeightTablePreferringFloat32Sidecar(
        at: blobPath.path,
        expectedCount: exactValues.count
    )
    #expect(loaded == exactValues)
}

@Test func test_loadWeightTablePreferringFloat32SidecarFallsBackToBlob() throws {
    let root = try makeTempDirectory()
    defer { try? FileManager.default.removeItem(at: root) }

    let blobValues: [Float] = [1, 2, 3]
    let blobPath = root.appendingPathComponent("rms_final.bin")
    try writeBlob(values: blobValues, to: blobPath)

    let loaded = try RealModelInferenceEngine.loadWeightTablePreferringFloat32Sidecar(
        at: blobPath.path,
        expectedCount: blobValues.count
    )
    #expect(loaded.elementsEqual(blobValues, by: { abs($0 - $1) < 1e-3 }))
}

@Test func test_loadRawFP16WeightTableIfNoExactFloat32SidecarReadsBlobPayload() throws {
    let root = try makeTempDirectory()
    defer { try? FileManager.default.removeItem(at: root) }

    let values: [Float] = [1.0, -2.5, 3.25]
    let blobPath = root.appendingPathComponent("lm_head.bin")
    try writeBlob(values: values, to: blobPath)

    let raw = try #require(
        try RealModelInferenceEngine.loadRawFP16WeightTableIfNoExactFloat32Sidecar(
            at: blobPath.path,
            expectedCount: values.count
        )
    )
    #expect(raw.map { Float(Float16(bitPattern: $0)) }.elementsEqual(values, by: { abs($0 - $1) < 1e-3 }))
}

@Test func test_loadRawFP16WeightTableIfNoExactFloat32SidecarDefersToExactSidecar() throws {
    let root = try makeTempDirectory()
    defer { try? FileManager.default.removeItem(at: root) }

    let blobPath = root.appendingPathComponent("lm_head.bin")
    try writeBlob(values: [1.0, 2.0, 3.0], to: blobPath)
    let sidecarPath = root.appendingPathComponent("lm_head.float32.bin")
    let bits: [UInt32] = [1.0, 2.0, 3.0].map { Float32($0).bitPattern }
    let data = bits.withUnsafeBytes { Data($0) }
    try data.write(to: sidecarPath)

    let raw = try RealModelInferenceEngine.loadRawFP16WeightTableIfNoExactFloat32Sidecar(
        at: blobPath.path,
        expectedCount: 3
    )
    #expect(raw == nil)
}

@Test func test_effectiveExactHeadBackendLabelAllowsLFM2FP16TiledHead() {
    #expect(
        RealModelInferenceEngine.effectiveExactHeadBackendLabelForTesting(
            strategy: .cpuFP16Tiled,
            architecture: .lfm2,
            hasRawFP16Head: true
        ) == "cpu_fp16_tiled"
    )
    #expect(
        RealModelInferenceEngine.effectiveExactHeadBackendLabelForTesting(
            strategy: .cpuFP16Tiled,
            architecture: .lfm2,
            hasRawFP16Head: false
        ) == "cpu_partitioned_fp32"
    )
    #expect(
        RealModelInferenceEngine.effectiveExactHeadBackendLabelForTesting(
            strategy: .cpuFP16Tiled,
            architecture: .gpt2,
            hasRawFP16Head: true
        ) == "cpu_partitioned_fp32"
    )
}

@Test func test_classifierArgmaxBlockSizeDefaultsTo4000() {
    #expect(RealModelInferenceEngine.classifierArgmaxBlockSize(environment: [:]) == 4_000)
    #expect(
        RealModelInferenceEngine.classifierArgmaxBlockSize(
            environment: ["ESPRESSO_CLASSIFIER_ARGMAX_BLOCK_SIZE": "0"]
        ) == 4_000
    )
    #expect(
        RealModelInferenceEngine.classifierArgmaxBlockSize(
            environment: ["ESPRESSO_CLASSIFIER_ARGMAX_BLOCK_SIZE": "nope"]
        ) == 4_000
    )
}

@Test func test_classifierArgmaxBlockSizeReadsEnvironmentOverride() {
    #expect(
        RealModelInferenceEngine.classifierArgmaxBlockSize(
            environment: ["ESPRESSO_CLASSIFIER_ARGMAX_BLOCK_SIZE": "8192"]
        ) == 8_192
    )
}

@Test func test_decodeContextFromCachesMatchesNaiveReference() {
    let heads = 4
    let kvHeads = 2
    let headDim = 8
    let visibleTokenCount = 5
    let cacheStride = 8
    let qOut = (0..<(heads * headDim)).map { Float(($0 % 7) - 3) * 0.125 }
    let kCache = (0..<(kvHeads * headDim * cacheStride)).map { Float(($0 % 11) - 5) * 0.0625 }
    let vCache = (0..<(kvHeads * headDim * cacheStride)).map { Float(($0 % 13) - 6) * 0.05 }

    let actual = RealModelInferenceEngine.decodeContextFromCachesForTesting(
        qOut: qOut,
        kCache: kCache,
        vCache: vCache,
        heads: heads,
        kvHeads: kvHeads,
        headDim: headDim,
        visibleTokenCount: visibleTokenCount,
        cacheStride: cacheStride
    )
    let expected = naiveDecodeContextFromCaches(
        qOut: qOut,
        kCache: kCache,
        vCache: vCache,
        heads: heads,
        kvHeads: kvHeads,
        headDim: headDim,
        visibleTokenCount: visibleTokenCount,
        cacheStride: cacheStride
    )

    #expect(maxAbsoluteDifference(actual, expected) < 1e-4)
}

@Test func test_fusedFFNGateUpProjectionMatchesSeparateSwiGLUPath() {
    let cols = 3
    let hiddenDim = 2
    let gate: [Float] = [
        1, 2, 3,
        4, 5, 6,
    ]
    let up: [Float] = [
        7, 8, 9,
        10, 11, 12,
    ]
    let vector: [Float] = [0.25, -0.5, 1.5]

    let fused = RealModelInferenceEngine.concatenateRowMajorMatricesVertically(
        upper: gate,
        lower: up,
        cols: cols
    )
    let fusedProjection = RealModelInferenceEngine.multiplyRowMajorMatrix(
        matrix: fused,
        rows: hiddenDim * 2,
        cols: cols,
        vector: vector
    )
    let activated = RealModelInferenceEngine.swiGLUActivatedFromFusedProjection(
        fusedProjection,
        hiddenDim: hiddenDim
    )

    let separateGate = RealModelInferenceEngine.multiplyRowMajorMatrix(
        matrix: gate,
        rows: hiddenDim,
        cols: cols,
        vector: vector
    )
    let separateUp = RealModelInferenceEngine.multiplyRowMajorMatrix(
        matrix: up,
        rows: hiddenDim,
        cols: cols,
        vector: vector
    )
    let expected = zip(separateGate, separateUp).map { gateValue, upValue in
        let silu = gateValue / (1 + Float(Darwin.exp(Double(-gateValue))))
        return silu * upValue
    }

    #expect(activated.elementsEqual(expected, by: { abs($0 - $1) < 1e-5 }))
}

@Test func test_evalHybridSingleLayerRawQKVOutputsForTestingRejectsUnsupportedArchitecture() throws {
    let config = makeTinyGPT2Config()
    try expectRealModelInferenceError(containing: "llama-family artifacts only") {
        _ = try RealModelInferenceEngine.evalHybridSingleLayerRawQKVOutputsForTesting(
            config: config,
            weightDir: "/nonexistent",
            layer: 0,
            token: 0
        )
    }
}

@Test func test_evalHybridSingleLayerDecodeProjectionForTestingRejectsUnsupportedArchitecture() throws {
    let config = makeTinyGPT2Config()
    try expectRealModelInferenceError(containing: "llama-family artifacts only") {
        _ = try RealModelInferenceEngine.evalHybridSingleLayerDecodeProjectionForTesting(
            config: config,
            weightDir: "/nonexistent",
            layer: 0,
            context: [0, 1, 2, 3, 4, 5, 6, 7]
        )
    }
}

@Test func test_evalHybridSingleLayerMetalContextForTestingRejectsUnsupportedArchitecture() throws {
    let config = makeTinyGPT2Config()
    try expectRealModelInferenceError(containing: "llama-family artifacts only") {
        _ = try RealModelInferenceEngine.evalHybridSingleLayerMetalContextForTesting(
            config: config,
            weightDir: "/nonexistent",
            layer: 0,
            token: 0
        )
    }
}

@Test func test_resolvedSpeculativeDraftLayerCountRequiresGreedyGPT2AndEnv() {
    let gpt2Config = MultiModelConfig(
        name: "spec-gpt2",
        nLayer: 12,
        nHead: 2,
        nKVHead: 2,
        dModel: 8,
        headDim: 4,
        hiddenDim: 32,
        vocab: 64,
        maxSeq: 8,
        normEps: 1e-5,
        architecture: .gpt2
    )
    let llamaConfig = MultiModelConfig(
        name: "spec-llama",
        nLayer: 12,
        nHead: 2,
        nKVHead: 2,
        dModel: 8,
        headDim: 4,
        hiddenDim: 32,
        vocab: 64,
        maxSeq: 8,
        normEps: 1e-5,
        architecture: .llama
    )

    #expect(
        RealModelInferenceEngine.resolvedSpeculativeDraftLayerCount(
            config: gpt2Config,
            temperature: 0,
            environment: [:]
        ) == nil
    )
    #expect(
        RealModelInferenceEngine.resolvedSpeculativeDraftLayerCount(
            config: gpt2Config,
            temperature: 0.8,
            environment: ["ESPRESSO_ENABLE_GPT2_SPECULATIVE": "1"]
        ) == nil
    )
    #expect(
        RealModelInferenceEngine.resolvedSpeculativeDraftLayerCount(
            config: llamaConfig,
            temperature: 0,
            environment: ["ESPRESSO_ENABLE_GPT2_SPECULATIVE": "1"]
        ) == nil
    )
}

@Test func test_resolvedSpeculativeDraftLayerCountDefaultsAndClamps() {
    let config = MultiModelConfig(
        name: "spec-gpt2",
        nLayer: 12,
        nHead: 2,
        nKVHead: 2,
        dModel: 8,
        headDim: 4,
        hiddenDim: 32,
        vocab: 64,
        maxSeq: 8,
        normEps: 1e-5,
        architecture: .gpt2
    )

    #expect(
        RealModelInferenceEngine.resolvedSpeculativeDraftLayerCount(
            config: config,
            temperature: 0,
            environment: ["ESPRESSO_ENABLE_GPT2_SPECULATIVE": "1"]
        ) == 1
    )
    #expect(
        RealModelInferenceEngine.resolvedSpeculativeDraftLayerCount(
            config: config,
            temperature: 0,
            environment: [
                "ESPRESSO_ENABLE_GPT2_SPECULATIVE": "1",
                "ESPRESSO_GPT2_SPECULATIVE_DRAFT_LAYERS": "0",
            ]
        ) == 1
    )
    #expect(
        RealModelInferenceEngine.resolvedSpeculativeDraftLayerCount(
            config: config,
            temperature: 0,
            environment: [
                "ESPRESSO_ENABLE_GPT2_SPECULATIVE": "1",
                "ESPRESSO_GPT2_SPECULATIVE_DRAFT_LAYERS": "99",
            ]
        ) == 11
    )
    #expect(
        RealModelInferenceEngine.resolvedSpeculativeDraftLayerCount(
            config: config,
            temperature: 0,
            environment: [
                "ESPRESSO_ENABLE_GPT2_SPECULATIVE": "1",
                "ESPRESSO_GPT2_SPECULATIVE_DRAFT_LAYERS": "junk",
            ]
        ) == 1
    )
}

@Test func test_usesHybridLayerInputRebinding_defaultsOffForLlama() {
    #expect(
        RealModelInferenceEngine.usesHybridLayerInputRebinding(
            architecture: .llama,
            environment: [:]
        ) == false
    )
    #expect(
        RealModelInferenceEngine.usesHybridLayerInputRebinding(
            architecture: .gpt2,
            environment: [:]
        ) == true
    )
}

@Test func test_usesHybridLayerInputRebinding_envOverridesForLlama() {
    #expect(
        RealModelInferenceEngine.usesHybridLayerInputRebinding(
            architecture: .llama,
            environment: ["ESPRESSO_ENABLE_LLAMA_HYBRID_LAYER_INPUT_REBIND": "1"]
        ) == true
    )
    #expect(
        RealModelInferenceEngine.usesHybridLayerInputRebinding(
            architecture: .llama,
            environment: [
                "ESPRESSO_ENABLE_LLAMA_HYBRID_LAYER_INPUT_REBIND": "1",
                "ESPRESSO_DISABLE_HYBRID_LAYER_INPUT_REBIND": "1",
            ]
        ) == false
    )
}

@Test func test_prefersCPUDecodeAttention_defaultsOnForQwenLlama() {
    let qwenConfig = MultiModelConfig(
        name: "qwen3",
        nLayer: 1,
        nHead: 2,
        nKVHead: 2,
        dModel: 8,
        headDim: 4,
        hiddenDim: 16,
        vocab: 64,
        maxSeq: 8,
        normEps: 1e-5,
        architecture: .llama
    )
    let gpt2Config = MultiModelConfig(
        name: "gpt2",
        nLayer: 1,
        nHead: 2,
        nKVHead: 2,
        dModel: 8,
        headDim: 4,
        hiddenDim: 16,
        vocab: 64,
        maxSeq: 8,
        normEps: 1e-5,
        architecture: .gpt2
    )

    #expect(RealModelInferenceEngine.prefersCPUDecodeAttention(config: qwenConfig, environment: [:]) == true)
    #expect(RealModelInferenceEngine.prefersCPUDecodeAttention(config: gpt2Config, environment: [:]) == false)
}

@Test func test_prefersCPUDecodeAttention_envOverridesDefault() {
    let qwenConfig = MultiModelConfig(
        name: "qwen3",
        nLayer: 1,
        nHead: 2,
        nKVHead: 2,
        dModel: 8,
        headDim: 4,
        hiddenDim: 16,
        vocab: 64,
        maxSeq: 8,
        normEps: 1e-5,
        architecture: .llama
    )
    let llamaConfig = MultiModelConfig(
        name: "llama3",
        nLayer: 1,
        nHead: 2,
        nKVHead: 2,
        dModel: 8,
        headDim: 4,
        hiddenDim: 16,
        vocab: 64,
        maxSeq: 8,
        normEps: 1e-5,
        architecture: .llama
    )

    #expect(
        RealModelInferenceEngine.prefersCPUDecodeAttention(
            config: qwenConfig,
            environment: ["ESPRESSO_FORCE_METAL_DECODE_ATTENTION": "1"]
        ) == false
    )
    #expect(
        RealModelInferenceEngine.prefersCPUDecodeAttention(
            config: llamaConfig,
            environment: ["ESPRESSO_USE_CPU_DECODE_ATTENTION": "1"]
        ) == true
    )
}

@Test func test_prefersCPUExactQKV_defaultsOffUnlessEnabled() {
    let qwenConfig = MultiModelConfig(
        name: "qwen3",
        nLayer: 1,
        nHead: 2,
        nKVHead: 2,
        dModel: 8,
        headDim: 4,
        hiddenDim: 16,
        vocab: 64,
        maxSeq: 8,
        normEps: 1e-5,
        architecture: .llama
    )
    let gpt2Config = MultiModelConfig(
        name: "gpt2",
        nLayer: 1,
        nHead: 2,
        nKVHead: 2,
        dModel: 8,
        headDim: 4,
        hiddenDim: 16,
        vocab: 64,
        maxSeq: 8,
        normEps: 1e-5,
        architecture: .gpt2
    )

    #expect(RealModelInferenceEngine.prefersCPUExactQKV(config: qwenConfig, environment: [:]) == false)
    #expect(RealModelInferenceEngine.prefersCPUExactQKV(config: gpt2Config, environment: [:]) == false)
}

@Test func test_prefersCPUExactQKV_envOverridesDefault() {
    let qwenConfig = MultiModelConfig(
        name: "qwen3",
        nLayer: 1,
        nHead: 2,
        nKVHead: 2,
        dModel: 8,
        headDim: 4,
        hiddenDim: 16,
        vocab: 64,
        maxSeq: 8,
        normEps: 1e-5,
        architecture: .llama
    )
    let llamaConfig = MultiModelConfig(
        name: "llama3",
        nLayer: 1,
        nHead: 2,
        nKVHead: 2,
        dModel: 8,
        headDim: 4,
        hiddenDim: 16,
        vocab: 64,
        maxSeq: 8,
        normEps: 1e-5,
        architecture: .llama
    )

    #expect(
        RealModelInferenceEngine.prefersCPUExactQKV(
            config: qwenConfig,
            environment: ["ESPRESSO_FORCE_ANE_QKV": "1"]
        ) == false
    )
    #expect(
        RealModelInferenceEngine.prefersCPUExactQKV(
            config: llamaConfig,
            environment: ["ESPRESSO_USE_CPU_EXACT_QKV": "1"]
        ) == true
    )
}

@Test func test_shouldUseBatchedMetalAttentionRequiresCachedBindings() {
    let enabledPolicy = RealModelExecutionPolicy(
        environment: ["ESPRESSO_BATCHED_METAL_ATTENTION": "1"]
    )

    #expect(
        RealModelInferenceEngine.shouldUseBatchedMetalAttention(
            policy: enabledPolicy,
            cachedBindingsAvailable: true,
            hasMetalRoPEConfig: false,
            hasCPUExactQKV: false
        )
    )
    #expect(
        !RealModelInferenceEngine.shouldUseBatchedMetalAttention(
            policy: enabledPolicy,
            cachedBindingsAvailable: false,
            hasMetalRoPEConfig: false,
            hasCPUExactQKV: false
        )
    )
    #expect(
        !RealModelInferenceEngine.shouldUseBatchedMetalAttention(
            policy: enabledPolicy,
            cachedBindingsAvailable: true,
            hasMetalRoPEConfig: true,
            hasCPUExactQKV: false
        )
    )
    #expect(
        !RealModelInferenceEngine.shouldUseBatchedMetalAttention(
            policy: enabledPolicy,
            cachedBindingsAvailable: true,
            hasMetalRoPEConfig: false,
            hasCPUExactQKV: true
        )
    )
}

@Test func test_prefersCPUExactDecode_defaultsOnForQwenLlama() {
    let qwenConfig = MultiModelConfig(
        name: "qwen3",
        nLayer: 1,
        nHead: 2,
        nKVHead: 2,
        dModel: 8,
        headDim: 4,
        hiddenDim: 16,
        vocab: 64,
        maxSeq: 8,
        normEps: 1e-5,
        architecture: .llama
    )
    let gpt2Config = MultiModelConfig(
        name: "gpt2",
        nLayer: 1,
        nHead: 2,
        nKVHead: 2,
        dModel: 8,
        headDim: 4,
        hiddenDim: 16,
        vocab: 64,
        maxSeq: 8,
        normEps: 1e-5,
        architecture: .gpt2
    )

    #expect(RealModelInferenceEngine.prefersCPUExactDecode(config: qwenConfig, environment: [:]) == true)
    #expect(RealModelInferenceEngine.prefersCPUExactDecode(config: gpt2Config, environment: [:]) == false)
}

@Test func test_prefersCPUExactDecode_envOverridesDefault() {
    let qwenConfig = MultiModelConfig(
        name: "qwen3",
        nLayer: 1,
        nHead: 2,
        nKVHead: 2,
        dModel: 8,
        headDim: 4,
        hiddenDim: 16,
        vocab: 64,
        maxSeq: 8,
        normEps: 1e-5,
        architecture: .llama
    )
    let llamaConfig = MultiModelConfig(
        name: "llama3",
        nLayer: 1,
        nHead: 2,
        nKVHead: 2,
        dModel: 8,
        headDim: 4,
        hiddenDim: 16,
        vocab: 64,
        maxSeq: 8,
        normEps: 1e-5,
        architecture: .llama
    )

    #expect(
        RealModelInferenceEngine.prefersCPUExactDecode(
            config: qwenConfig,
            environment: ["ESPRESSO_FORCE_HYBRID_DECODE": "1"]
        ) == false
    )
    #expect(
        RealModelInferenceEngine.prefersCPUExactDecode(
            config: llamaConfig,
            environment: ["ESPRESSO_USE_CPU_EXACT_DECODE": "1"]
        ) == true
    )
}

@Test func test_llamaGenerationPath_skipsHybridForQwenExactCPU() {
    let qwenConfig = MultiModelConfig(
        name: "qwen3",
        nLayer: 1,
        nHead: 2,
        nKVHead: 2,
        dModel: 8,
        headDim: 4,
        hiddenDim: 16,
        vocab: 64,
        maxSeq: 8,
        normEps: 1e-5,
        architecture: .llama
    )
    let llamaConfig = MultiModelConfig(
        name: "llama3",
        nLayer: 1,
        nHead: 2,
        nKVHead: 2,
        dModel: 8,
        headDim: 4,
        hiddenDim: 16,
        vocab: 64,
        maxSeq: 8,
        normEps: 1e-5,
        architecture: .llama
    )

    #expect(
        RealModelInferenceEngine.llamaGenerationPath(
            config: qwenConfig,
            environment: [:]
        ) == .exactCPU
    )
    #expect(
        RealModelInferenceEngine.llamaGenerationPath(
            config: qwenConfig,
            environment: ["ESPRESSO_FORCE_HYBRID_DECODE": "1"]
        ) == .hybrid
    )
    #expect(
        RealModelInferenceEngine.llamaGenerationPath(
            config: llamaConfig,
            environment: [:]
        ) == .hybrid
    )
}

@Test func test_decodeContextFromCaches_respectsVisibleTokenCount() {
    let qOut: [Float] = [1, 0]
    let kCache: [Float] = [
        1, 100,
        0, 0,
    ]
    let vCache: [Float] = [
        5, 99,
        7, 123,
    ]

    let causal = decodeContextFromCaches(
        qOut: qOut,
        kCache: kCache,
        vCache: vCache,
        heads: 1,
        kvHeads: 1,
        headDim: 2,
        mapping: .groupedContiguous,
        visibleTokenCount: 1
    )
    let nonCausal = decodeContextFromCaches(
        qOut: qOut,
        kCache: kCache,
        vCache: vCache,
        heads: 1,
        kvHeads: 1,
        headDim: 2,
        mapping: .groupedContiguous
    )

    #expect(causal[0] == 5)
    #expect(causal[1] == 7)
    #expect(abs(nonCausal[0] - 99) < 0.01)
    #expect(abs(nonCausal[1] - 123) < 0.01)
}

@Test func test_boundedSpeculativeCacheOrderEvictsLeastRecentlyUsed() {
    let insert = RealModelInferenceEngine.boundedSpeculativeCacheOrder(
        currentOrder: [1, 2, 3],
        accessedKey: 4,
        limit: 3,
        insertingNewEntry: true
    )
    #expect(insert.order == [2, 3, 4])
    #expect(insert.evictedKey == 1)

    let hit = RealModelInferenceEngine.boundedSpeculativeCacheOrder(
        currentOrder: insert.order,
        accessedKey: 3,
        limit: 3,
        insertingNewEntry: false
    )
    #expect(hit.order == [2, 4, 3])
    #expect(hit.evictedKey == nil)
}

@Test func test_weightPathResolution() throws {
    let root = "/tmp/real-model-inference"

    let gpt2Paths = LayerWeightPaths.forLayer(3, config: ModelRegistry.gpt2_124m, blobDir: root)
    #expect(gpt2Paths.rmsAtt == "\(root)/layers/3/ln_1_gamma.bin")
    #expect(gpt2Paths.bq == "\(root)/layers/3/bq.bin")
    #expect(gpt2Paths.rmsFfn == "\(root)/layers/3/ln_2_gamma.bin")
    #expect(gpt2Paths.b2 == "\(root)/layers/3/b2.bin")

    let llamaPaths = LayerWeightPaths.forLayer(5, config: ModelRegistry.stories110m, blobDir: root)
    #expect(llamaPaths.rmsAtt == "\(root)/layers/5/rms_att.bin")
    #expect(llamaPaths.qNorm == "\(root)/layers/5/q_norm.bin")
    #expect(llamaPaths.kNorm == "\(root)/layers/5/k_norm.bin")
    #expect(llamaPaths.bq == nil)
    #expect(llamaPaths.rmsFfn == "\(root)/layers/5/rms_ffn.bin")
    #expect(llamaPaths.w3 == "\(root)/layers/5/w3.bin")

    let canonicalDir = try makeMinimalGPT2WeightDirectory(config: makeTinyGPT2Config(), topLevelNaming: .canonical)
    let canonicalTop = try RealModelInferenceEngine.resolveTopLevelWeightPaths(
        config: makeTinyGPT2Config(),
        weightDir: canonicalDir.path
    )
    #expect(canonicalTop.tokenEmbedding == canonicalDir.appendingPathComponent("embeddings/token.bin").path)
    #expect(canonicalTop.positionEmbedding == canonicalDir.appendingPathComponent("embeddings/position.bin").path)
    #expect(canonicalTop.finalNormGamma == canonicalDir.appendingPathComponent("final_norm_gamma.bin").path)
    #expect(canonicalTop.finalNormBeta == canonicalDir.appendingPathComponent("final_norm_beta.bin").path)
    #expect(canonicalTop.lmHead == canonicalDir.appendingPathComponent("lm_head.bin").path)

    let aliasDir = try makeMinimalGPT2WeightDirectory(config: makeTinyGPT2Config(), topLevelNaming: .aliases)
    let aliasTop = try RealModelInferenceEngine.resolveTopLevelWeightPaths(
        config: makeTinyGPT2Config(),
        weightDir: aliasDir.path
    )
    #expect(aliasTop.tokenEmbedding == aliasDir.appendingPathComponent("embeddings/token_embeddings.bin").path)
    #expect(aliasTop.positionEmbedding == aliasDir.appendingPathComponent("embeddings/position_embeddings.bin").path)
    #expect(aliasTop.finalNormGamma == aliasDir.appendingPathComponent("ln_f_gamma.bin").path)
    #expect(aliasTop.finalNormBeta == aliasDir.appendingPathComponent("ln_f_beta.bin").path)
    #expect(aliasTop.lmHead == aliasDir.appendingPathComponent("classifier.bin").path)
}

@Test func test_loadHybridLayerWeightsLlamaLoadsOptionalQKNormWeightsWhenPresent() throws {
    let config = makeTinyLlamaConfig()
    let qNorm: [Float] = [1.0, 0.5, 1.5, 2.0]
    let kNorm: [Float] = [0.25, 1.25, 0.75, 1.75]
    let weightDir = try makeMinimalLlamaLayerWeightDirectory(
        config: config,
        qNorm: qNorm,
        kNorm: kNorm
    )

    let weights = try RealModelInferenceEngine.loadHybridLayerWeightsLlamaForTesting(
        config: config,
        weightDir: weightDir.path,
        layer: 0
    )

    let hasQNorm = weights.hasQNorm
    let hasKNorm = weights.hasKNorm
    let hasQKNorm = weights.hasQKNorm
    let loadedQNorm = weights.qNorm.withUnsafeBufferPointer { Array($0) }
    let loadedKNorm = weights.kNorm.withUnsafeBufferPointer { Array($0) }
    #expect(hasQNorm)
    #expect(hasKNorm)
    #expect(hasQKNorm)
    #expect(loadedQNorm == qNorm)
    #expect(loadedKNorm == kNorm)
}

@Test func test_loadHybridLayerWeightsLlamaLeavesQKNormAbsentWhenFilesMissing() throws {
    let config = makeTinyLlamaConfig()
    let weightDir = try makeMinimalLlamaLayerWeightDirectory(config: config)

    let weights = try RealModelInferenceEngine.loadHybridLayerWeightsLlamaForTesting(
        config: config,
        weightDir: weightDir.path,
        layer: 0
    )

    let hasQNorm = weights.hasQNorm
    let hasKNorm = weights.hasKNorm
    let hasQKNorm = weights.hasQKNorm
    let qNormCount = weights.qNorm.count
    let kNormCount = weights.kNorm.count
    #expect(hasQNorm == false)
    #expect(hasKNorm == false)
    #expect(hasQKNorm == false)
    #expect(qNormCount == 0)
    #expect(kNormCount == 0)
}

@Test func test_loadHybridLayerWeightsLlamaRequiresBothQKNormWeights() throws {
    let config = makeTinyLlamaConfig()
    let weightDir = try makeMinimalLlamaLayerWeightDirectory(
        config: config,
        qNorm: [1.0, 1.0, 1.0, 1.0]
    )

    try expectRealModelInferenceError(containing: "Mismatched llama Q/K norm weights") {
        _ = try RealModelInferenceEngine.loadHybridLayerWeightsLlamaForTesting(
            config: config,
            weightDir: weightDir.path,
            layer: 0
        )
    }
}

@Test func test_transformerLayerGraphUsesMILWeightOffset64() throws {
    let config = makeTinyGPT2Config()
    let paths = LayerWeightPaths.forLayer(0, config: config, blobDir: "/tmp/real-model-inference")
    let graph = TransformerLayerGraphBuilder.forwardLayer(
        layer: 0,
        config: config,
        paths: paths,
        spatial: config.maxSeq
    )

    let offsets = graph.nodes.compactMap { node -> UInt64? in
        guard case let .weight(_, offset) = node.attrs else { return nil }
        return offset
    }

    #expect(!offsets.isEmpty)
    #expect(offsets.allSatisfy { $0 == 64 })
}

@Test func test_tokenizerRoundtrip() throws {
    let gpt2TokenizerDir = try makeGPT2TokenizerDirectory()
    let gpt2 = try GPT2BPETokenizer(
        vocabURL: gpt2TokenizerDir.appendingPathComponent("vocab.json"),
        mergesURL: gpt2TokenizerDir.appendingPathComponent("merges.txt")
    )
    let gpt2Text = "Hello, world!\n"
    #expect(gpt2.decode(gpt2.encode(gpt2Text)) == gpt2Text)

    let sentencePieceModelURL = try makeSentencePieceTokenizerDirectory()
    let sentencePiece = try SentencePieceTokenizer(modelURL: sentencePieceModelURL)
    let sentencePieceText = "Hello, world!\n"
    #expect(sentencePiece.decode(sentencePiece.encode(sentencePieceText)) == sentencePieceText)
}

@Test func test_singleLayerCompileAndEval() throws {
    guard shouldRunANEHardwareTests() else { return }

    let config = makeHardwareGPT2Config()
    let weightDir = try makeMinimalGPT2WeightDirectory(config: config)
    let elementCount = config.dModel * config.maxSeq
    var values = [Float](repeating: 0, count: elementCount)
    for index in values.indices {
        values[index] = Float((index % 11) - 5) * 0.03125
    }
    let result = try RealModelInferenceEngine.compileAndEvalSingleLayerForTesting(
        config: config,
        weightDir: weightDir.path,
        layer: 0,
        spatial: config.maxSeq,
        input: values
    )

    #expect(result.allSatisfy { $0.isFinite })
    #expect(result.contains { abs($0) > 0.0001 })
}

@Test func test_gpt2HeadCompile() throws {
    guard shouldRunANEHardwareTests() else { return }

    let config = makeHardwareGPT2Config()
    let weightDir = try makeMinimalGPT2WeightDirectory(config: config)

    try RealModelInferenceEngine.compileHeadForTesting(
        config: config,
        weightDir: weightDir.path
    )
}

@Test func test_hybridSingleLayerMatchesFullSequenceLastTokenForActualGPT2Weights() throws {
    guard shouldRunANEHardwareTests() else { return }
    guard let weightsDir = ProcessInfo.processInfo.environment["GPT2_WEIGHTS_DIR"], !weightsDir.isEmpty else {
        return
    }

    let config = ModelRegistry.gpt2_124m
    let tokens: [TokenID] = [15496, 11]
    let packedInput = try RealModelInferenceEngine.composeEmbeddingInputForTesting(
        config: config,
        weightDir: weightsDir,
        tokens: tokens
    )
    let referenceSpatial = max(tokens.count, RealModelInferenceEngine.minimumCompileSpatial(channels: config.dModel))
    var fullInput = [Float](repeating: 0, count: config.dModel * referenceSpatial)
    for channel in 0..<config.dModel {
        for tokenIndex in tokens.indices {
            fullInput[channel * referenceSpatial + tokenIndex] =
                packedInput[channel * tokens.count + tokenIndex]
        }
    }
    let fullOutput = try RealModelInferenceEngine.compileAndEvalSingleLayerForTesting(
        config: config,
        weightDir: weightsDir,
        layer: 0,
        spatial: referenceSpatial,
        input: fullInput
    )
    let fullAttentionOutputs = try RealModelInferenceEngine.compileAndEvalSingleLayerAttentionOutputsForTesting(
        config: config,
        weightDir: weightsDir,
        layer: 0,
        spatial: referenceSpatial,
        input: fullInput
    )
    let hybridOutput = try RealModelInferenceEngine.evalHybridSingleLayerForTesting(
        config: config,
        weightDir: weightsDir,
        layer: 0,
        tokens: tokens
    )
    let hybridAttentionOutputs = try RealModelInferenceEngine.evalHybridSingleLayerAttentionOutputsForTesting(
        config: config,
        weightDir: weightsDir,
        layer: 0,
        tokens: tokens
    )

    let fullLastToken = extractSpatialSlice(
        from: fullOutput,
        channels: config.dModel,
        spatial: referenceSpatial,
        spatialIndex: tokens.count - 1
    )
    let fullAttentionLastToken = extractSpatialSlice(
        from: fullAttentionOutputs.hidden,
        channels: config.dModel,
        spatial: referenceSpatial,
        spatialIndex: tokens.count - 1
    )
    let fullKLastToken = extractSpatialSlice(
        from: fullAttentionOutputs.kCache,
        channels: config.dModel,
        spatial: referenceSpatial,
        spatialIndex: tokens.count - 1
    )
    let fullVLastToken = extractSpatialSlice(
        from: fullAttentionOutputs.vCache,
        channels: config.dModel,
        spatial: referenceSpatial,
        spatialIndex: tokens.count - 1
    )
    let maxDiff = maxAbsoluteDifference(fullLastToken, hybridOutput)
    let attentionMaxDiff = maxAbsoluteDifference(fullAttentionLastToken, hybridAttentionOutputs.hidden)
    let kMaxDiff = maxAbsoluteDifference(
        fullKLastToken,
        extractSpatialSlice(
            from: hybridAttentionOutputs.kCache,
            channels: config.dModel,
            spatial: tokens.count,
            spatialIndex: tokens.count - 1
        )
    )
    let vMaxDiff = maxAbsoluteDifference(
        fullVLastToken,
        extractSpatialSlice(
            from: hybridAttentionOutputs.vCache,
            channels: config.dModel,
            spatial: tokens.count,
            spatialIndex: tokens.count - 1
        )
    )
    #expect(kMaxDiff < 0.01)
    #expect(vMaxDiff < 0.01)
    #expect(attentionMaxDiff < 0.01)
    #expect(maxDiff < 0.01)
}

@Test func test_hybridSingleLayerRunsForActualLFM2AttentionWeights() throws {
    guard shouldRunANEHardwareTests() else { return }
    guard let weightsDir = lfm2ProbeWeightDirectory() else { return }

    let metadataURL = URL(fileURLWithPath: weightsDir, isDirectory: true).appendingPathComponent("metadata.json")
    let metadata = try JSONDecoder().decode(DebugWeightMetadataFile.self, from: Data(contentsOf: metadataURL))
    let config = try metadata.asConfig()
    let attentionLayer = try #require(config.resolvedLFM2LayerTypes.firstIndex(of: .fullAttention))

    let output = try RealModelInferenceEngine.evalHybridSingleLayerForTesting(
        config: config,
        weightDir: weightsDir,
        layer: attentionLayer,
        tokens: [11, 22]
    )

    #expect(output.count == config.dModel)
    #expect(output.allSatisfy { $0.isFinite })
    #expect(output.contains(where: { abs($0) > 0.0001 }))
}

@Test func test_fullModelGeneration() throws {
    guard shouldRunANEHardwareTests() else { return }

    guard let weightsDir = ProcessInfo.processInfo.environment["GPT2_WEIGHTS_DIR"], !weightsDir.isEmpty else {
        return
    }

    let tokenizerDir = {
        if let explicit = ProcessInfo.processInfo.environment["GPT2_TOKENIZER_DIR"], !explicit.isEmpty {
            return explicit
        }
        return weightsDir
    }()

    var engine = try RealModelInferenceEngine.build(
        config: ModelRegistry.gpt2_124m,
        weightDir: weightsDir,
        tokenizerDir: tokenizerDir
    )
    let result = try engine.generate(prompt: "Hello", maxTokens: 10, temperature: 0.0)

    #expect(!result.text.isEmpty)
    #expect(!result.promptTokens.isEmpty)
    #expect(!result.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
}

@Test func test_gpt2HybridGenerationWithMinimalWeights() throws {
    guard shouldRunANEHardwareTests() else { return }

    let config = makeHardwareGPT2Config()
    let weightDir = try makeMinimalGPT2WeightDirectory(config: config)
    let tokenizerDir = try makeGPT2TokenizerDirectory()
    defer {
        try? FileManager.default.removeItem(at: weightDir)
        try? FileManager.default.removeItem(at: tokenizerDir)
    }

    var engine = try RealModelInferenceEngine.build(
        config: config,
        weightDir: weightDir.path,
        tokenizerDir: tokenizerDir.path
    )
    let result = try engine.generate(prompt: "Hello", maxTokens: 1, temperature: 0.0)

    #expect(!result.promptTokens.isEmpty)
    #expect(result.text.hasPrefix("Hello"))
}

@Test func test_llama32GreedyNextTokenPrefixesMatchHFReference() throws {
    guard shouldRunANEHardwareTests() else { return }

    let repoRoot = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .deletingLastPathComponent()
    let weightsDir = repoRoot.appendingPathComponent(".artifacts/llama3_2_1b", isDirectory: true)
    let tokenizerDir = repoRoot.appendingPathComponent(".artifacts/llama3_2_1b_tokenizer", isDirectory: true)
    guard FileManager.default.fileExists(atPath: weightsDir.appendingPathComponent("metadata.json").path),
          FileManager.default.fileExists(atPath: tokenizerDir.path) else {
        return
    }

    let cases: [(prompt: String, expectedToken: TokenID, expectedSuffix: String)] = [
        ("Hello,", 358, " I"),
        ("Hello, I", 1097, " am"),
        ("Hello, I am", 8173, " interested"),
        ("Hello, I am interested", 922, " about"),
    ]
    let base = ModelRegistry.llama3_2_1b
    let config = MultiModelConfig(
        name: base.name,
        nLayer: base.nLayer,
        nHead: base.nHead,
        nKVHead: base.nKVHead,
        dModel: base.dModel,
        headDim: base.headDim,
        hiddenDim: base.hiddenDim,
        vocab: base.vocab,
        maxSeq: 512,
        normEps: base.normEps,
        ropeTheta: base.ropeTheta,
        eosToken: base.eosToken,
        architecture: base.architecture
    )

    var engine = try RealModelInferenceEngine.build(
        config: config,
        weightDir: weightsDir.path,
        tokenizerDir: tokenizerDir.path
    )

    for testcase in cases {
        let result = try engine.generate(prompt: testcase.prompt, maxTokens: 1, temperature: 0.0)
        #expect(result.tokens.count == 1, "Expected one generated token for prompt '\(testcase.prompt)'")
        #expect(result.tokens.first == testcase.expectedToken, "Prompt '\(testcase.prompt)' selected token \(String(describing: result.tokens.first))")
        #expect(result.text.hasSuffix(testcase.expectedSuffix), "Prompt '\(testcase.prompt)' produced text '\(result.text)'")
    }
}

private struct DebugWeightMetadataFile: Decodable {
    let name: String
    let nLayer: Int
    let nHead: Int
    let nKVHead: Int
    let dModel: Int
    let headDim: Int
    let hiddenDim: Int
    let vocab: Int
    let maxSeq: Int
    let normEps: Float
    let ropeTheta: Float?
    let eosToken: UInt32?
    let architecture: String
    let layerTypes: [LFM2LayerType]?
    let convCacheLength: Int?
    let tieEmbedding: Bool?

    func asConfig() throws -> MultiModelConfig {
        let parsedArchitecture: MultiModelConfig.Architecture
        switch architecture.lowercased() {
        case "gpt2":
            parsedArchitecture = .gpt2
        case "llama":
            parsedArchitecture = .llama
        case "lfm2":
            parsedArchitecture = .lfm2
        default:
            throw RealModelInferenceError.runtimeFailure("Unsupported architecture in metadata.json: \(architecture)")
        }
        return MultiModelConfig(
            name: name,
            nLayer: nLayer,
            nHead: nHead,
            nKVHead: nKVHead,
            dModel: dModel,
            headDim: headDim,
            hiddenDim: hiddenDim,
            vocab: vocab,
            maxSeq: maxSeq,
            normEps: normEps,
            ropeTheta: ropeTheta ?? 10_000.0,
            eosToken: eosToken,
            architecture: parsedArchitecture,
            lfm2LayerTypes: layerTypes,
            lfm2ConvCacheLength: convCacheLength,
            tieEmbedding: tieEmbedding ?? false
        )
    }
}

@Test func test_debugQwenRawPromptNextTokenFromWeightDir() throws {
    guard shouldRunLegacyQwenExperimentTests() else { return }
    guard shouldRunANEHardwareTests() else { return }
    guard let weightDir = ProcessInfo.processInfo.environment["ESPRESSO_DEBUG_WEIGHT_DIR"], !weightDir.isEmpty else {
        return
    }

    let metadataURL = URL(fileURLWithPath: weightDir, isDirectory: true).appendingPathComponent("metadata.json")
    let metadata = try JSONDecoder().decode(DebugWeightMetadataFile.self, from: Data(contentsOf: metadataURL))
    let config = try metadata.asConfig()
    let promptTokens: [TokenID]
    if let rawPromptTokens = ProcessInfo.processInfo.environment["ESPRESSO_DEBUG_PROMPT_TOKENS"], !rawPromptTokens.isEmpty {
        promptTokens = rawPromptTokens
            .split(separator: ",")
            .compactMap { TokenID($0.trimmingCharacters(in: .whitespacesAndNewlines)) }
    } else {
        promptTokens = [9707]
    }
    #expect(!promptTokens.isEmpty)

    let nextToken = try RealModelInferenceEngine.generateNextTokenForTesting(
        config: config,
        weightDir: weightDir,
        promptTokens: promptTokens
    )
    fputs("[qwen-debug-next-token] weightDir=\(weightDir) prompt=\(promptTokens) token=\(nextToken)\n", stderr)

    if ProcessInfo.processInfo.environment["ESPRESSO_DEBUG_DUMP_LAYER0_VCACHE"] == "1" {
        let outputs = try RealModelInferenceEngine.evalHybridSingleLayerAttentionOutputsForTesting(
            config: config,
            weightDir: weightDir,
            layer: 0,
            tokens: promptTokens
        )
        let prefix = outputs.vCache.prefix(32).map { String(format: "%.4f", $0) }.joined(separator: ",")
        fputs("[qwen-debug-layer0-vcache] prefix=[\(prefix)]\n", stderr)
    }

    if let expectedRaw = ProcessInfo.processInfo.environment["ESPRESSO_DEBUG_EXPECT_TOKEN"],
       let expected = TokenID(expectedRaw) {
        #expect(nextToken == expected)
    }
}

@Test func test_debugQwenCPUArtifactPromptNextTokenFromWeightDir() throws {
    guard shouldRunLegacyQwenExperimentTests() else { return }
    guard let weightDir = ProcessInfo.processInfo.environment["ESPRESSO_DEBUG_WEIGHT_DIR"], !weightDir.isEmpty else {
        return
    }
    guard ProcessInfo.processInfo.environment["ESPRESSO_DEBUG_CHECK_CPU_ARTIFACT_NEXT_TOKEN"] == "1" else {
        return
    }

    let metadataURL = URL(fileURLWithPath: weightDir, isDirectory: true).appendingPathComponent("metadata.json")
    let metadata = try JSONDecoder().decode(DebugWeightMetadataFile.self, from: Data(contentsOf: metadataURL))
    let config = try metadata.asConfig()
    let promptTokens: [TokenID]
    if let rawPromptTokens = ProcessInfo.processInfo.environment["ESPRESSO_DEBUG_PROMPT_TOKENS"], !rawPromptTokens.isEmpty {
        promptTokens = rawPromptTokens
            .split(separator: ",")
            .compactMap { TokenID($0.trimmingCharacters(in: .whitespacesAndNewlines)) }
    } else {
        promptTokens = [9707]
    }
    #expect(!promptTokens.isEmpty)

    let nextToken = try cpuArtifactLlamaNextToken(
        weightDir: URL(fileURLWithPath: weightDir, isDirectory: true),
        config: config,
        tokens: promptTokens.map(Int.init)
    )
    fputs("[qwen-debug-cpu-artifact-next-token] weightDir=\(weightDir) prompt=\(promptTokens) token=\(nextToken)\n", stderr)

    if let expectedRaw = ProcessInfo.processInfo.environment["ESPRESSO_DEBUG_EXPECT_TOKEN"],
       let expected = TokenID(expectedRaw) {
        #expect(TokenID(nextToken) == expected)
    }
}

@Test func test_debugQwenRuntimeFinalHiddenPromptNextTokenFromWeightDir() throws {
    guard shouldRunLegacyQwenExperimentTests() else { return }
    guard shouldRunANEHardwareTests() else { return }
    guard let weightDir = ProcessInfo.processInfo.environment["ESPRESSO_DEBUG_WEIGHT_DIR"], !weightDir.isEmpty else {
        return
    }
    guard ProcessInfo.processInfo.environment["ESPRESSO_DEBUG_CHECK_RUNTIME_FINAL_HIDDEN_NEXT_TOKEN"] == "1" else {
        return
    }

    let metadataURL = URL(fileURLWithPath: weightDir, isDirectory: true).appendingPathComponent("metadata.json")
    let metadata = try JSONDecoder().decode(DebugWeightMetadataFile.self, from: Data(contentsOf: metadataURL))
    let config = try metadata.asConfig()
    let promptTokens: [TokenID]
    if let rawPromptTokens = ProcessInfo.processInfo.environment["ESPRESSO_DEBUG_PROMPT_TOKENS"], !rawPromptTokens.isEmpty {
        promptTokens = rawPromptTokens
            .split(separator: ",")
            .compactMap { TokenID($0.trimmingCharacters(in: .whitespacesAndNewlines)) }
    } else {
        promptTokens = [9707]
    }
    #expect(!promptTokens.isEmpty)

    let runtime = try RealModelInferenceEngine.evalHybridLlamaLayerHiddenLineageForTesting(
        config: config,
        weightDir: weightDir,
        tokens: promptTokens
    )
    guard let finalHidden = runtime.layerHiddenStates.last else {
        Issue.record("Runtime lineage did not produce a final hidden state")
        return
    }

    let nextToken = try cpuArtifactLlamaNextTokenFromFinalHidden(
        finalHidden,
        weightDir: URL(fileURLWithPath: weightDir, isDirectory: true),
        config: config
    )
    fputs(
        "[qwen-debug-runtime-final-hidden-next-token] weightDir=\(weightDir) prompt=\(promptTokens) token=\(nextToken)\n",
        stderr
    )

    if let expectedRaw = ProcessInfo.processInfo.environment["ESPRESSO_DEBUG_EXPECT_TOKEN"],
       let expected = TokenID(expectedRaw) {
        #expect(TokenID(nextToken) == expected)
    }
}

@Test func test_debugQwenCPUExactLayer0ThenRuntimeRemainingLayersPromptNextTokenFromWeightDir() throws {
    guard shouldRunLegacyQwenExperimentTests() else { return }
    guard shouldRunANEHardwareTests() else { return }
    guard let weightDir = ProcessInfo.processInfo.environment["ESPRESSO_DEBUG_WEIGHT_DIR"], !weightDir.isEmpty else {
        return
    }
    guard ProcessInfo.processInfo.environment["ESPRESSO_DEBUG_CHECK_CPU_LAYER0_RUNTIME_REST_NEXT_TOKEN"] == "1" else {
        return
    }

    let metadataURL = URL(fileURLWithPath: weightDir, isDirectory: true).appendingPathComponent("metadata.json")
    let metadata = try JSONDecoder().decode(DebugWeightMetadataFile.self, from: Data(contentsOf: metadataURL))
    let config = try metadata.asConfig()
    let promptTokens = debugPromptTokens(
        env: ProcessInfo.processInfo.environment,
        defaultTokens: [9707]
    )
    #expect(!promptTokens.isEmpty)

    let weightURL = URL(fileURLWithPath: weightDir, isDirectory: true)
    let tokenEmbedding = try readBlobFloat16File(
        at: weightURL.appendingPathComponent("embeddings/token.bin"),
        expectedCount: config.vocab * config.dModel
    )
    var hiddenStates = try cpuSingleLlamaLayerStepOutputs(
        weightDir: weightURL,
        config: config,
        layerIndex: 0,
        inputs: promptTokens.map { token in
            Array(tokenEmbedding[token * config.dModel..<(token + 1) * config.dModel])
        }
    ).map(\.hidden)

    if config.nLayer > 1 {
        for layerIndex in 1..<config.nLayer {
            hiddenStates = try RealModelInferenceEngine.evalHybridSingleLlamaLayerOutputsFromInputsForTesting(
                config: config,
                weightDir: weightDir,
                layer: layerIndex,
                inputs: hiddenStates
            )
        }
    }

    guard let finalHidden = hiddenStates.last else {
        Issue.record("CPU-layer0/runtime-rest path did not produce a final hidden state")
        return
    }

    let nextToken = try cpuArtifactLlamaNextTokenFromFinalHidden(
        finalHidden,
        weightDir: weightURL,
        config: config
    )
    fputs(
        "[qwen-debug-cpu-layer0-runtime-rest-next-token] weightDir=\(weightDir) prompt=\(promptTokens) token=\(nextToken)\n",
        stderr
    )

    if let expectedRaw = ProcessInfo.processInfo.environment["ESPRESSO_DEBUG_EXPECT_TOKEN"],
       let expected = TokenID(expectedRaw) {
        #expect(TokenID(nextToken) == expected)
    }
}

@Test func test_debugQwenRawGGUFTopWeightsPromptNextTokenFromWeightDir() async throws {
    guard shouldRunLegacyQwenExperimentTests() else { return }
    guard shouldRunANEHardwareTests() else { return }
    guard let weightDir = ProcessInfo.processInfo.environment["ESPRESSO_DEBUG_WEIGHT_DIR"], !weightDir.isEmpty else {
        return
    }
    guard let ggufModel = ProcessInfo.processInfo.environment["ESPRESSO_DEBUG_GGUF_MODEL"], !ggufModel.isEmpty else {
        return
    }
    guard ProcessInfo.processInfo.environment["ESPRESSO_DEBUG_CHECK_RAW_GGUF_TOP_WEIGHTS_NEXT_TOKEN"] == "1" else {
        return
    }

    let metadataURL = URL(fileURLWithPath: weightDir, isDirectory: true).appendingPathComponent("metadata.json")
    let metadata = try JSONDecoder().decode(DebugWeightMetadataFile.self, from: Data(contentsOf: metadataURL))
    let config = try metadata.asConfig()
    let promptTokens: [TokenID]
    if let rawPromptTokens = ProcessInfo.processInfo.environment["ESPRESSO_DEBUG_PROMPT_TOKENS"], !rawPromptTokens.isEmpty {
        promptTokens = rawPromptTokens
            .split(separator: ",")
            .compactMap { TokenID($0.trimmingCharacters(in: .whitespacesAndNewlines)) }
    } else {
        promptTokens = [9707]
    }
    #expect(!promptTokens.isEmpty)

    let runtime = try RealModelInferenceEngine.evalHybridLlamaLayerHiddenLineageForTesting(
        config: config,
        weightDir: weightDir,
        tokens: promptTokens
    )
    guard let finalHidden = runtime.layerHiddenStates.last else {
        Issue.record("Runtime lineage did not produce a final hidden state")
        return
    }

    let artifactToken = try cpuArtifactLlamaNextTokenFromFinalHidden(
        finalHidden,
        weightDir: URL(fileURLWithPath: weightDir, isDirectory: true),
        config: config
    )
    let rawGGUFToken = try await rawGGUFLlamaNextTokenFromFinalHidden(
        finalHidden,
        ggufURL: URL(fileURLWithPath: ggufModel),
        config: config
    )
    fputs(
        "[qwen-debug-raw-gguf-top-weights-next-token] weightDir=\(weightDir) gguf=\(ggufModel) prompt=\(promptTokens) artifactToken=\(artifactToken) rawGGUFToken=\(rawGGUFToken)\n",
        stderr
    )

    if let expectedRaw = ProcessInfo.processInfo.environment["ESPRESSO_DEBUG_EXPECT_TOKEN"],
       let expected = TokenID(expectedRaw) {
        #expect(TokenID(rawGGUFToken) == expected)
    }
}

@Test func test_debugQwenCPUArtifactRawGGUFTopWeightsPromptNextTokenFromWeightDir() async throws {
    guard shouldRunLegacyQwenExperimentTests() else { return }
    guard let weightDir = ProcessInfo.processInfo.environment["ESPRESSO_DEBUG_WEIGHT_DIR"], !weightDir.isEmpty else {
        return
    }
    guard let ggufModel = ProcessInfo.processInfo.environment["ESPRESSO_DEBUG_GGUF_MODEL"], !ggufModel.isEmpty else {
        return
    }
    guard ProcessInfo.processInfo.environment["ESPRESSO_DEBUG_CHECK_CPU_ARTIFACT_RAW_GGUF_TOP_WEIGHTS_NEXT_TOKEN"] == "1" else {
        return
    }

    let metadataURL = URL(fileURLWithPath: weightDir, isDirectory: true).appendingPathComponent("metadata.json")
    let metadata = try JSONDecoder().decode(DebugWeightMetadataFile.self, from: Data(contentsOf: metadataURL))
    let config = try metadata.asConfig()
    let promptTokens: [Int]
    if let rawPromptTokens = ProcessInfo.processInfo.environment["ESPRESSO_DEBUG_PROMPT_TOKENS"], !rawPromptTokens.isEmpty {
        promptTokens = rawPromptTokens
            .split(separator: ",")
            .compactMap { Int($0.trimmingCharacters(in: .whitespacesAndNewlines)) }
    } else {
        promptTokens = [9707]
    }
    #expect(!promptTokens.isEmpty)

    let finalStates = try cpuArtifactLlamaLayerHiddenLineage(
        weightDir: URL(fileURLWithPath: weightDir, isDirectory: true),
        config: config,
        tokens: promptTokens
    )
    guard let finalHidden = finalStates.last else {
        Issue.record("CPU artifact lineage did not produce a final hidden state")
        return
    }

    let artifactToken = try cpuArtifactLlamaNextTokenFromFinalHidden(
        finalHidden,
        weightDir: URL(fileURLWithPath: weightDir, isDirectory: true),
        config: config
    )
    let rawGGUFToken = try await rawGGUFLlamaNextTokenFromFinalHidden(
        finalHidden,
        ggufURL: URL(fileURLWithPath: ggufModel),
        config: config
    )
    fputs(
        "[qwen-debug-cpu-artifact-raw-gguf-top-weights-next-token] weightDir=\(weightDir) gguf=\(ggufModel) prompt=\(promptTokens) artifactToken=\(artifactToken) rawGGUFToken=\(rawGGUFToken)\n",
        stderr
    )

    if let expectedRaw = ProcessInfo.processInfo.environment["ESPRESSO_DEBUG_EXPECT_TOKEN"],
       let expected = Int(expectedRaw) {
        #expect(rawGGUFToken == expected)
    }
}

@Test func test_debugQwenRawLayer0QKVFromWeightDir() throws {
    guard shouldRunLegacyQwenExperimentTests() else { return }
    guard shouldRunANEHardwareTests() else { return }
    guard let weightDir = ProcessInfo.processInfo.environment["ESPRESSO_DEBUG_WEIGHT_DIR"], !weightDir.isEmpty else {
        return
    }
    guard ProcessInfo.processInfo.environment["ESPRESSO_DEBUG_DUMP_LAYER0_RAW_QKV"] == "1" else {
        return
    }

    let metadataURL = URL(fileURLWithPath: weightDir, isDirectory: true).appendingPathComponent("metadata.json")
    let metadata = try JSONDecoder().decode(DebugWeightMetadataFile.self, from: Data(contentsOf: metadataURL))
    let config = try metadata.asConfig()
    let outputs = try RealModelInferenceEngine.evalHybridSingleLayerRawQKVOutputsForTesting(
        config: config,
        weightDir: weightDir,
        layer: 0,
        token: 9707
    )

    let qPrefix = outputs.qOut.prefix(32).map { String(format: "%.4f", $0) }.joined(separator: ",")
    let kPrefix = outputs.kOut.prefix(32).map { String(format: "%.4f", $0) }.joined(separator: ",")
    let vPrefix = outputs.vOut.prefix(32).map { String(format: "%.4f", $0) }.joined(separator: ",")
    fputs("[qwen-debug-layer0-qout] prefix=[\(qPrefix)]\n", stderr)
    fputs("[qwen-debug-layer0-kout] prefix=[\(kPrefix)]\n", stderr)
    fputs("[qwen-debug-layer0-vout] prefix=[\(vPrefix)]\n", stderr)

    if let dumpPath = ProcessInfo.processInfo.environment["ESPRESSO_DEBUG_DUMP_LAYER0_RAW_QKV_PATH"],
       !dumpPath.isEmpty {
        let payload: [String: [Float]] = [
            "qOut": outputs.qOut,
            "kOut": outputs.kOut,
            "vOut": outputs.vOut,
        ]
        let data = try JSONSerialization.data(withJSONObject: payload, options: [.sortedKeys])
        try data.write(to: URL(fileURLWithPath: dumpPath))
        fputs("[qwen-debug-layer0-qkv-path] path=\(dumpPath)\n", stderr)
    }
}

@Test func test_debugQwenLayer0DecodeProjectionContextContractFromWeightDir() throws {
    guard shouldRunLegacyQwenExperimentTests() else { return }
    guard shouldRunANEHardwareTests() else { return }
    guard let weightDir = ProcessInfo.processInfo.environment["ESPRESSO_DEBUG_WEIGHT_DIR"], !weightDir.isEmpty else {
        return
    }
    guard ProcessInfo.processInfo.environment["ESPRESSO_DEBUG_CHECK_LAYER0_PROJECTION_CONTEXT"] == "1" else {
        return
    }
    guard ProcessInfo.processInfo.environment["ESPRESSO_DISABLE_HYBRID_FUSED_POST_ATTENTION"] == "1" else {
        return
    }

    let metadataURL = URL(fileURLWithPath: weightDir, isDirectory: true).appendingPathComponent("metadata.json")
    let metadata = try JSONDecoder().decode(DebugWeightMetadataFile.self, from: Data(contentsOf: metadataURL))
    let config = try metadata.asConfig()
    let context = (0..<config.attentionDim).map { index in
        Float((index % 97) - 48) / 97.0
    }

    let projection = try RealModelInferenceEngine.evalHybridSingleLayerDecodeProjectionForTesting(
        config: config,
        weightDir: weightDir,
        layer: 0,
        context: context
    )

    let woURL = URL(fileURLWithPath: weightDir, isDirectory: true)
        .appendingPathComponent("layers/0/wo.bin")
    let wo = try readBlobFloat16File(at: woURL, expectedCount: config.dModel * config.attentionDim)
    let woRows = stride(from: 0, to: wo.count, by: config.attentionDim).map {
        Array(wo[$0..<$0 + config.attentionDim])
    }

    let headMajorCPU = multiplyRowMajorMatrix(
        rows: woRows,
        vector: context
    )
    let dimMajorContext = headMajorToDimMajorContext(
        context,
        heads: config.nHead,
        headDim: config.headDim
    )
    let dimMajorCPU = multiplyRowMajorMatrix(
        rows: woRows,
        vector: dimMajorContext
    )

    let headMajorMaxDiff = zip(projection.output, headMajorCPU).map { abs($0 - $1) }.max() ?? .infinity
    let dimMajorMaxDiff = zip(projection.output, dimMajorCPU).map { abs($0 - $1) }.max() ?? .infinity
    let headMajorMeanDiff = zip(projection.output, headMajorCPU).map { abs($0 - $1) }.reduce(0, +) / Float(config.dModel)
    let dimMajorMeanDiff = zip(projection.output, dimMajorCPU).map { abs($0 - $1) }.reduce(0, +) / Float(config.dModel)

    fputs(
        "[qwen-debug-layer0-projection] headMajorMaxDiff=\(headMajorMaxDiff) headMajorMeanDiff=\(headMajorMeanDiff) dimMajorMaxDiff=\(dimMajorMaxDiff) dimMajorMeanDiff=\(dimMajorMeanDiff)\n",
        stderr
    )

    #expect(headMajorMaxDiff < 0.01)
    #expect(headMajorMeanDiff < 0.001)
    #expect(headMajorMaxDiff < dimMajorMaxDiff)
    #expect(headMajorMeanDiff < dimMajorMeanDiff)
}

@Test func test_debugQwenLayer0MetalContextContractFromWeightDir() throws {
    guard shouldRunLegacyQwenExperimentTests() else { return }
    guard shouldRunANEHardwareTests() else { return }
    guard let weightDir = ProcessInfo.processInfo.environment["ESPRESSO_DEBUG_WEIGHT_DIR"], !weightDir.isEmpty else {
        return
    }
    guard ProcessInfo.processInfo.environment["ESPRESSO_DEBUG_CHECK_LAYER0_METAL_CONTEXT"] == "1" else {
        return
    }

    let metadataURL = URL(fileURLWithPath: weightDir, isDirectory: true).appendingPathComponent("metadata.json")
    let metadata = try JSONDecoder().decode(DebugWeightMetadataFile.self, from: Data(contentsOf: metadataURL))
    let config = try metadata.asConfig()
    let outputs = try RealModelInferenceEngine.evalHybridSingleLayerMetalContextForTesting(
        config: config,
        weightDir: weightDir,
        layer: 0,
        token: 9707
    )

    let groupedExpected = groupedContextFromVOut(
        outputs.vOut,
        heads: config.nHead,
        kvHeads: config.nKVHead,
        headDim: config.headDim
    )
    let moduloExpected = moduloContextFromVOut(
        outputs.vOut,
        heads: config.nHead,
        kvHeads: config.nKVHead,
        headDim: config.headDim
    )

    let groupedMaxDiff = zip(outputs.context, groupedExpected).map { abs($0 - $1) }.max() ?? .infinity
    let groupedMeanDiff = zip(outputs.context, groupedExpected).map { abs($0 - $1) }.reduce(0, +) / Float(config.attentionDim)
    let moduloMaxDiff = zip(outputs.context, moduloExpected).map { abs($0 - $1) }.max() ?? .infinity
    let moduloMeanDiff = zip(outputs.context, moduloExpected).map { abs($0 - $1) }.reduce(0, +) / Float(config.attentionDim)

    fputs(
        "[qwen-debug-layer0-metal-context] groupedMaxDiff=\(groupedMaxDiff) groupedMeanDiff=\(groupedMeanDiff) moduloMaxDiff=\(moduloMaxDiff) moduloMeanDiff=\(moduloMeanDiff)\n",
        stderr
    )

    #expect(groupedMaxDiff < 0.01)
    #expect(groupedMeanDiff < 0.001)
    #expect(groupedMaxDiff < moduloMaxDiff)
    #expect(groupedMeanDiff < moduloMeanDiff)
}

@Test func test_debugQwenLayer0Token1MetalContextMatchesCPUFromWeightDir() throws {
    guard shouldRunLegacyQwenExperimentTests() else { return }
    guard shouldRunANEHardwareTests() else { return }
    guard let weightDir = ProcessInfo.processInfo.environment["ESPRESSO_DEBUG_WEIGHT_DIR"], !weightDir.isEmpty else {
        return
    }
    guard ProcessInfo.processInfo.environment["ESPRESSO_DEBUG_CHECK_LAYER0_TOKEN1_METAL_CONTEXT"] == "1" else {
        return
    }

    let metadataURL = URL(fileURLWithPath: weightDir, isDirectory: true).appendingPathComponent("metadata.json")
    let metadata = try JSONDecoder().decode(DebugWeightMetadataFile.self, from: Data(contentsOf: metadataURL))
    let config = try metadata.asConfig()
    let promptTokens = debugPromptTokens(
        env: ProcessInfo.processInfo.environment,
        defaultTokens: [9707, 21806]
    )
    #expect(promptTokens.count >= 2)

    let outputs = try RealModelInferenceEngine.evalHybridSingleLayerHookedLlamaMetalContextForTesting(
        config: config,
        weightDir: weightDir,
        layer: 0,
        tokens: promptTokens.map(TokenID.init)
    )

    let groupedExpected = decodeContextFromCaches(
        qOut: outputs.qOut,
        kCache: outputs.kCache,
        vCache: outputs.vCache,
        heads: config.nHead,
        kvHeads: config.nKVHead,
        headDim: config.headDim,
        mapping: .groupedContiguous,
        visibleTokenCount: promptTokens.count
    )
    let moduloExpected = decodeContextFromCaches(
        qOut: outputs.qOut,
        kCache: outputs.kCache,
        vCache: outputs.vCache,
        heads: config.nHead,
        kvHeads: config.nKVHead,
        headDim: config.headDim,
        mapping: .moduloInterleaved,
        visibleTokenCount: promptTokens.count
    )

    let groupedMaxDiff = zip(outputs.context, groupedExpected).map { abs($0 - $1) }.max() ?? .infinity
    let groupedMeanDiff = zip(outputs.context, groupedExpected).map { abs($0 - $1) }.reduce(0, +) / Float(config.attentionDim)
    let moduloMaxDiff = zip(outputs.context, moduloExpected).map { abs($0 - $1) }.max() ?? .infinity
    let moduloMeanDiff = zip(outputs.context, moduloExpected).map { abs($0 - $1) }.reduce(0, +) / Float(config.attentionDim)

    fputs(
        "[qwen-debug-layer0-token1-metal-context] groupedMaxDiff=\(groupedMaxDiff) groupedMeanDiff=\(groupedMeanDiff) moduloMaxDiff=\(moduloMaxDiff) moduloMeanDiff=\(moduloMeanDiff)\n",
        stderr
    )

    #expect(groupedMaxDiff < 0.01)
    #expect(groupedMeanDiff < 0.001)
    #expect(groupedMaxDiff < moduloMaxDiff)
    #expect(groupedMeanDiff < moduloMeanDiff)
}

@Test func test_debugQwenToken1LayerLineageMatchesCPUArtifactFromWeightDir() throws {
    guard shouldRunLegacyQwenExperimentTests() else { return }
    guard shouldRunANEHardwareTests() else { return }
    guard let weightDir = ProcessInfo.processInfo.environment["ESPRESSO_DEBUG_WEIGHT_DIR"], !weightDir.isEmpty else {
        return
    }
    guard ProcessInfo.processInfo.environment["ESPRESSO_DEBUG_CHECK_TOKEN1_LAYER_LINEAGE"] == "1" else {
        return
    }

    let metadataURL = URL(fileURLWithPath: weightDir, isDirectory: true).appendingPathComponent("metadata.json")
    let metadata = try JSONDecoder().decode(DebugWeightMetadataFile.self, from: Data(contentsOf: metadataURL))
    let config = try metadata.asConfig()
    let tokens: [Int]
    if let rawPromptTokens = ProcessInfo.processInfo.environment["ESPRESSO_DEBUG_PROMPT_TOKENS"], !rawPromptTokens.isEmpty {
        tokens = rawPromptTokens
            .split(separator: ",")
            .compactMap { Int($0.trimmingCharacters(in: .whitespacesAndNewlines)) }
    } else {
        tokens = [9707, 21806]
    }
    #expect(!tokens.isEmpty)

    let runtime = try RealModelInferenceEngine.evalHybridLlamaLayerHiddenLineageForTesting(
        config: config,
        weightDir: weightDir,
        tokens: tokens.map(TokenID.init)
    )
    let cpu = try cpuArtifactLlamaLayerHiddenLineage(
        weightDir: URL(fileURLWithPath: weightDir, isDirectory: true),
        config: config,
        tokens: tokens
    )

    var firstBadLayer: Int?
    for layerIndex in 0..<config.nLayer {
        let diff = zip(runtime.layerHiddenStates[layerIndex], cpu[layerIndex]).map { abs($0 - $1) }
        let maxDiff = diff.max() ?? .infinity
        let meanDiff = diff.reduce(0, +) / Float(diff.count)
        fputs("[qwen-debug-token1-layer-lineage] layer=\(layerIndex) maxDiff=\(maxDiff) meanDiff=\(meanDiff)\n", stderr)
        if firstBadLayer == nil, meanDiff > 0.01 {
            firstBadLayer = layerIndex
        }
    }

    #expect(firstBadLayer == nil)
}

@Test func test_debugQwenLayer1MatchesCPUWhenFedExactCPUInputs() throws {
    guard shouldRunLegacyQwenExperimentTests() else { return }
    guard shouldRunANEHardwareTests() else { return }
    guard let weightDir = ProcessInfo.processInfo.environment["ESPRESSO_DEBUG_WEIGHT_DIR"], !weightDir.isEmpty else {
        return
    }
    guard ProcessInfo.processInfo.environment["ESPRESSO_DEBUG_CHECK_LAYER1_LOCAL_RUNTIME"] == "1" else {
        return
    }

    let metadataURL = URL(fileURLWithPath: weightDir, isDirectory: true).appendingPathComponent("metadata.json")
    let metadata = try JSONDecoder().decode(DebugWeightMetadataFile.self, from: Data(contentsOf: metadataURL))
    let config = try metadata.asConfig()
    let weightURL = URL(fileURLWithPath: weightDir, isDirectory: true)

    let cpuT0 = try cpuArtifactLlamaLayerHiddenLineage(
        weightDir: weightURL,
        config: config,
        tokens: [9707],
        throughLayerIndex: 0
    )
    let cpuT1 = try cpuArtifactLlamaLayerHiddenLineage(
        weightDir: weightURL,
        config: config,
        tokens: [9707, 21806],
        throughLayerIndex: 0
    )

    let runtime = try RealModelInferenceEngine.evalHybridSingleLlamaLayerFromInputsForTesting(
        config: config,
        weightDir: weightDir,
        layer: 1,
        inputs: [cpuT0[0], cpuT1[0]]
    )
    let expected = cpuT1[1]
    let diff = zip(runtime, expected).map { abs($0 - $1) }
    let maxDiff = diff.max() ?? .infinity
    let meanDiff = diff.reduce(0, +) / Float(diff.count)

    fputs("[qwen-debug-layer1-local-runtime] maxDiff=\(maxDiff) meanDiff=\(meanDiff)\n", stderr)

    #expect(maxDiff < 0.02)
    #expect(meanDiff < 0.005)
}

@Test func test_debugQwenLayer1MatchesCPUWhenFedRuntimeLayer0Inputs() throws {
    guard shouldRunLegacyQwenExperimentTests() else { return }
    guard shouldRunANEHardwareTests() else { return }
    guard let weightDir = ProcessInfo.processInfo.environment["ESPRESSO_DEBUG_WEIGHT_DIR"], !weightDir.isEmpty else {
        return
    }
    guard ProcessInfo.processInfo.environment["ESPRESSO_DEBUG_CHECK_LAYER1_RUNTIME_INPUTS"] == "1" else {
        return
    }

    let metadataURL = URL(fileURLWithPath: weightDir, isDirectory: true).appendingPathComponent("metadata.json")
    let metadata = try JSONDecoder().decode(DebugWeightMetadataFile.self, from: Data(contentsOf: metadataURL))
    let config = try metadata.asConfig()
    let weightURL = URL(fileURLWithPath: weightDir, isDirectory: true)

    let runtimeT0 = try RealModelInferenceEngine.evalHybridLlamaLayerHiddenLineageForTesting(
        config: config,
        weightDir: weightDir,
        tokens: [9707]
    )
    let runtimeT1 = try RealModelInferenceEngine.evalHybridLlamaLayerHiddenLineageForTesting(
        config: config,
        weightDir: weightDir,
        tokens: [9707, 21806]
    )

    let cpu = try cpuSingleLlamaLayerStepOutputs(
        weightDir: weightURL,
        config: config,
        layerIndex: 1,
        inputs: [runtimeT0.layerHiddenStates[0], runtimeT1.layerHiddenStates[0]]
    )[1]

    let diff = zip(runtimeT1.layerHiddenStates[1], cpu.hidden).map { abs($0 - $1) }
    let maxDiff = diff.max() ?? .infinity
    let meanDiff = diff.reduce(0, +) / Float(diff.count)

    fputs("[qwen-debug-layer1-runtime-inputs] maxDiff=\(maxDiff) meanDiff=\(meanDiff)\n", stderr)

    #expect(maxDiff < 0.02)
    #expect(meanDiff < 0.005)
}

@Test func test_debugQwenLayer0ExactStagesMatchCPUForPromptToken1InSplitPath() throws {
    guard shouldRunLegacyQwenExperimentTests() else { return }
    guard shouldRunANEHardwareTests() else { return }
    guard let weightDir = ProcessInfo.processInfo.environment["ESPRESSO_DEBUG_WEIGHT_DIR"], !weightDir.isEmpty else {
        return
    }
    guard ProcessInfo.processInfo.environment["ESPRESSO_DEBUG_CHECK_LAYER0_EXACT_STAGES"] == "1" else {
        return
    }
    guard ProcessInfo.processInfo.environment["ESPRESSO_DISABLE_HYBRID_FUSED_POST_ATTENTION"] == "1" else {
        return
    }

    let metadataURL = URL(fileURLWithPath: weightDir, isDirectory: true).appendingPathComponent("metadata.json")
    let metadata = try JSONDecoder().decode(DebugWeightMetadataFile.self, from: Data(contentsOf: metadataURL))
    let config = try metadata.asConfig()
    let weightURL = URL(fileURLWithPath: weightDir, isDirectory: true)
    let tokenEmbedding = try readBlobFloat16File(
        at: weightURL.appendingPathComponent("embeddings/token.bin"),
        expectedCount: config.vocab * config.dModel
    )
    let promptTokens = [9707, 21806]
    let inputs = promptTokens.map { token in
        Array(tokenEmbedding[token * config.dModel..<(token + 1) * config.dModel])
    }

    let runtime = try RealModelInferenceEngine.evalHybridSingleLlamaLayerDetailedFromInputsForTesting(
        config: config,
        weightDir: weightDir,
        layer: 0,
        inputs: inputs
    )
    let cpu = try cpuSingleLlamaLayerStepOutputs(
        weightDir: weightURL,
        config: config,
        layerIndex: 0,
        inputs: inputs
    )[1]

    let contextDiff = zip(runtime.context, cpu.context).map { abs($0 - $1) }
    let contextMaxDiff = contextDiff.max() ?? .infinity
    let contextMeanDiff = contextDiff.reduce(0, +) / Float(contextDiff.count)

    let projectionDiff = zip(runtime.projectionOut, cpu.projected).map { abs($0 - $1) }
    let projectionMaxDiff = projectionDiff.max() ?? .infinity
    let projectionMeanDiff = projectionDiff.reduce(0, +) / Float(projectionDiff.count)

    let hiddenDiff = zip(runtime.hidden, cpu.hidden).map { abs($0 - $1) }
    let hiddenMaxDiff = hiddenDiff.max() ?? .infinity
    let hiddenMeanDiff = hiddenDiff.reduce(0, +) / Float(hiddenDiff.count)

    fputs(
        "[qwen-debug-layer0-exact-stages] contextMaxDiff=\(contextMaxDiff) contextMeanDiff=\(contextMeanDiff) projectionMaxDiff=\(projectionMaxDiff) projectionMeanDiff=\(projectionMeanDiff) hiddenMaxDiff=\(hiddenMaxDiff) hiddenMeanDiff=\(hiddenMeanDiff)\n",
        stderr
    )

    #expect(contextMaxDiff < 0.01)
    #expect(contextMeanDiff < 0.001)
    #expect(projectionMaxDiff < 0.02)
    #expect(projectionMeanDiff < 0.005)
    #expect(hiddenMaxDiff < 0.02)
    #expect(hiddenMeanDiff < 0.005)
}

@Test func test_debugQwenPromptToken1ExactStagesMatchCPUForLayerFromEnv() throws {
    guard shouldRunLegacyQwenExperimentTests() else { return }
    guard shouldRunANEHardwareTests() else { return }
    guard let weightDir = ProcessInfo.processInfo.environment["ESPRESSO_DEBUG_WEIGHT_DIR"], !weightDir.isEmpty else {
        return
    }
    guard ProcessInfo.processInfo.environment["ESPRESSO_DEBUG_CHECK_PROMPT_TOKEN1_LAYER_STAGES"] == "1" else {
        return
    }
    guard let rawLayer = ProcessInfo.processInfo.environment["ESPRESSO_DEBUG_LAYER_INDEX"],
          let layerIndex = Int(rawLayer) else {
        Issue.record("Set ESPRESSO_DEBUG_LAYER_INDEX for the exact-stage layer probe")
        return
    }

    let metadataURL = URL(fileURLWithPath: weightDir, isDirectory: true).appendingPathComponent("metadata.json")
    let metadata = try JSONDecoder().decode(DebugWeightMetadataFile.self, from: Data(contentsOf: metadataURL))
    let config = try metadata.asConfig()
    #expect(layerIndex >= 0)
    #expect(layerIndex < config.nLayer)

    let weightURL = URL(fileURLWithPath: weightDir, isDirectory: true)
    let promptTokens = debugPromptTokens(
        env: ProcessInfo.processInfo.environment,
        defaultTokens: [9707, 21806]
    )
    let inputs: [[Float]]
    if layerIndex == 0 {
        let tokenEmbedding = try readBlobFloat16File(
            at: weightURL.appendingPathComponent("embeddings/token.bin"),
            expectedCount: config.vocab * config.dModel
        )
        inputs = promptTokens.map { token in
            Array(tokenEmbedding[token * config.dModel..<(token + 1) * config.dModel])
        }
    } else {
        let cpuT0 = try cpuArtifactLlamaLayerHiddenLineage(
            weightDir: weightURL,
            config: config,
            tokens: [promptTokens[0]],
            throughLayerIndex: layerIndex - 1
        )
        let cpuT1 = try cpuArtifactLlamaLayerHiddenLineage(
            weightDir: weightURL,
            config: config,
            tokens: promptTokens,
            throughLayerIndex: layerIndex - 1
        )
        inputs = [cpuT0[layerIndex - 1], cpuT1[layerIndex - 1]]
    }

    let runtime = try RealModelInferenceEngine.evalHybridSingleLlamaLayerDetailedFromInputsForTesting(
        config: config,
        weightDir: weightDir,
        layer: layerIndex,
        inputs: inputs
    )
    let cpu = try cpuSingleLlamaLayerStepOutputs(
        weightDir: weightURL,
        config: config,
        layerIndex: layerIndex,
        inputs: inputs
    )[1]

    func summarize(_ label: String, _ actual: [Float], _ expected: [Float]) -> (Float, Float) {
        let diff = zip(actual, expected).map { abs($0 - $1) }
        let maxDiff = diff.max() ?? .infinity
        let meanDiff = diff.reduce(0, +) / Float(diff.count)
        return (maxDiff, meanDiff)
    }

    let (contextMaxDiff, contextMeanDiff) = summarize("context", runtime.context, cpu.context)
    let (projectionMaxDiff, projectionMeanDiff) = summarize("projection", runtime.projectionOut, cpu.projected)
    let (hiddenMaxDiff, hiddenMeanDiff) = summarize("hidden", runtime.hidden, cpu.hidden)

    fputs(
        "[qwen-debug-prompt-token1-layer-exact-stages] layer=\(layerIndex) contextMaxDiff=\(contextMaxDiff) contextMeanDiff=\(contextMeanDiff) projectionMaxDiff=\(projectionMaxDiff) projectionMeanDiff=\(projectionMeanDiff) hiddenMaxDiff=\(hiddenMaxDiff) hiddenMeanDiff=\(hiddenMeanDiff)\n",
        stderr
    )
}

@Test func test_debugQwenLayer1QKVInputSurfaceIsStable() throws {
    guard shouldRunLegacyQwenExperimentTests() else { return }
    guard shouldRunANEHardwareTests() else { return }
    guard let weightDir = ProcessInfo.processInfo.environment["ESPRESSO_DEBUG_WEIGHT_DIR"], !weightDir.isEmpty else {
        return
    }
    guard ProcessInfo.processInfo.environment["ESPRESSO_DEBUG_CHECK_LAYER1_QKV_INPUT_STABILITY"] == "1" else {
        return
    }

    let metadataURL = URL(fileURLWithPath: weightDir, isDirectory: true).appendingPathComponent("metadata.json")
    let metadata = try JSONDecoder().decode(DebugWeightMetadataFile.self, from: Data(contentsOf: metadataURL))
    let config = try metadata.asConfig()
    let input = (0..<config.dModel).map { index in
        let centered = Float((index % 17) - 8) / 8.0
        return centered + sin(Float(index) * 0.1) * 0.25
    }

    let outputs = try RealModelInferenceEngine.evalHybridSingleLayerQKVInputStabilityForTesting(
        config: config,
        weightDir: weightDir,
        layer: 1,
        input: input
    )
    let diff = zip(outputs.inputBeforeQKV, outputs.inputAfterQKV).map { abs($0 - $1) }
    let maxDiff = diff.max() ?? .infinity
    let meanDiff = diff.reduce(0, +) / Float(diff.count)

    fputs("[qwen-debug-layer1-qkv-input-stability] maxDiff=\(maxDiff) meanDiff=\(meanDiff)\n", stderr)

    #expect(maxDiff < 0.0005)
    #expect(meanDiff < 0.0001)
}

@Test func test_debugQwenLayer1SyntheticLocalReferenceMatchesCPU() throws {
    guard shouldRunLegacyQwenExperimentTests() else { return }
    guard shouldRunANEHardwareTests() else { return }
    guard let weightDir = ProcessInfo.processInfo.environment["ESPRESSO_DEBUG_WEIGHT_DIR"], !weightDir.isEmpty else {
        return
    }
    guard ProcessInfo.processInfo.environment["ESPRESSO_DEBUG_CHECK_LAYER1_SYNTHETIC_LOCAL"] == "1" else {
        return
    }

    let metadataURL = URL(fileURLWithPath: weightDir, isDirectory: true).appendingPathComponent("metadata.json")
    let metadata = try JSONDecoder().decode(DebugWeightMetadataFile.self, from: Data(contentsOf: metadataURL))
    let config = try metadata.asConfig()
    let weightURL = URL(fileURLWithPath: weightDir, isDirectory: true)
    let inputs = makeDeterministicLayerInputs(config: config, tokenCount: 2)

    let runtime = try RealModelInferenceEngine.evalHybridSingleLlamaLayerDetailedFromInputsForTesting(
        config: config,
        weightDir: weightDir,
        layer: 1,
        inputs: inputs
    )
    let cpu = try cpuSingleLlamaLayerStepOutputs(
        weightDir: weightURL,
        config: config,
        layerIndex: 1,
        inputs: inputs
    )
    let expected = cpu[1]

    let contextDiff = zip(runtime.context, expected.context).map { abs($0 - $1) }
    let contextMaxDiff = contextDiff.max() ?? .infinity
    let contextMeanDiff = contextDiff.reduce(0, +) / Float(contextDiff.count)

    let projectionDiff = zip(runtime.projectionOut, expected.projected).map { abs($0 - $1) }
    let projectionMaxDiff = projectionDiff.max() ?? .infinity
    let projectionMeanDiff = projectionDiff.reduce(0, +) / Float(projectionDiff.count)

    let hiddenDiff = zip(runtime.hidden, expected.hidden).map { abs($0 - $1) }
    let hiddenMaxDiff = hiddenDiff.max() ?? .infinity
    let hiddenMeanDiff = hiddenDiff.reduce(0, +) / Float(hiddenDiff.count)

    fputs(
        "[qwen-debug-layer1-synth-local] contextMaxDiff=\(contextMaxDiff) contextMeanDiff=\(contextMeanDiff) projectionMaxDiff=\(projectionMaxDiff) projectionMeanDiff=\(projectionMeanDiff) hiddenMaxDiff=\(hiddenMaxDiff) hiddenMeanDiff=\(hiddenMeanDiff)\n",
        stderr
    )

    #expect(contextMaxDiff < 0.01)
    #expect(contextMeanDiff < 0.001)
    #expect(projectionMaxDiff < 0.02)
    #expect(projectionMeanDiff < 0.005)
    #expect(hiddenMaxDiff < 0.02)
    #expect(hiddenMeanDiff < 0.005)
}

@Test func test_debugQwenLayer1ContextMatchesCPUWhenFedExactCPUInputs() throws {
    guard shouldRunLegacyQwenExperimentTests() else { return }
    guard shouldRunANEHardwareTests() else { return }
    guard let weightDir = ProcessInfo.processInfo.environment["ESPRESSO_DEBUG_WEIGHT_DIR"], !weightDir.isEmpty else {
        return
    }
    guard ProcessInfo.processInfo.environment["ESPRESSO_DEBUG_CHECK_LAYER1_CONTEXT"] == "1" else {
        return
    }

    let metadataURL = URL(fileURLWithPath: weightDir, isDirectory: true).appendingPathComponent("metadata.json")
    let metadata = try JSONDecoder().decode(DebugWeightMetadataFile.self, from: Data(contentsOf: metadataURL))
    let config = try metadata.asConfig()
    let weightURL = URL(fileURLWithPath: weightDir, isDirectory: true)

    let cpuT0 = try cpuArtifactLlamaLayerHiddenLineage(
        weightDir: weightURL,
        config: config,
        tokens: [9707],
        throughLayerIndex: 0
    )
    let cpuT1 = try cpuArtifactLlamaLayerHiddenLineage(
        weightDir: weightURL,
        config: config,
        tokens: [9707, 21806],
        throughLayerIndex: 0
    )

    let outputs = try RealModelInferenceEngine.evalHybridSingleLlamaLayerDetailedFromInputsForTesting(
        config: config,
        weightDir: weightDir,
        layer: 1,
        inputs: [cpuT0[0], cpuT1[0]]
    )
    let groupedExpected = decodeContextFromCaches(
        qOut: outputs.qOut,
        kCache: outputs.kCache,
        vCache: outputs.vCache,
        heads: config.nHead,
        kvHeads: config.nKVHead,
        headDim: config.headDim,
        mapping: .groupedContiguous
    )
    let moduloExpected = decodeContextFromCaches(
        qOut: outputs.qOut,
        kCache: outputs.kCache,
        vCache: outputs.vCache,
        heads: config.nHead,
        kvHeads: config.nKVHead,
        headDim: config.headDim,
        mapping: .moduloInterleaved
    )

    let groupedDiff = zip(outputs.context, groupedExpected).map { abs($0 - $1) }
    let groupedMaxDiff = groupedDiff.max() ?? .infinity
    let groupedMeanDiff = groupedDiff.reduce(0, +) / Float(groupedDiff.count)
    let moduloDiff = zip(outputs.context, moduloExpected).map { abs($0 - $1) }
    let moduloMaxDiff = moduloDiff.max() ?? .infinity
    let moduloMeanDiff = moduloDiff.reduce(0, +) / Float(moduloDiff.count)

    fputs(
        "[qwen-debug-layer1-context] groupedMaxDiff=\(groupedMaxDiff) groupedMeanDiff=\(groupedMeanDiff) moduloMaxDiff=\(moduloMaxDiff) moduloMeanDiff=\(moduloMeanDiff)\n",
        stderr
    )

    #expect(groupedMaxDiff < 0.01)
    #expect(groupedMeanDiff < 0.001)
    #expect(groupedMaxDiff < moduloMaxDiff)
    #expect(groupedMeanDiff < moduloMeanDiff)
}

@Test func test_debugQwenLayer1ExactStagesMatchCPUInSplitPath() throws {
    guard shouldRunLegacyQwenExperimentTests() else { return }
    guard shouldRunANEHardwareTests() else { return }
    guard let weightDir = ProcessInfo.processInfo.environment["ESPRESSO_DEBUG_WEIGHT_DIR"], !weightDir.isEmpty else {
        return
    }
    guard ProcessInfo.processInfo.environment["ESPRESSO_DEBUG_CHECK_LAYER1_EXACT_STAGES"] == "1" else {
        return
    }
    guard ProcessInfo.processInfo.environment["ESPRESSO_DISABLE_HYBRID_FUSED_POST_ATTENTION"] == "1" else {
        return
    }

    let metadataURL = URL(fileURLWithPath: weightDir, isDirectory: true).appendingPathComponent("metadata.json")
    let metadata = try JSONDecoder().decode(DebugWeightMetadataFile.self, from: Data(contentsOf: metadataURL))
    let config = try metadata.asConfig()
    let weightURL = URL(fileURLWithPath: weightDir, isDirectory: true)

    let cpuT0 = try cpuArtifactLlamaLayerHiddenLineage(
        weightDir: weightURL,
        config: config,
        tokens: [9707],
        throughLayerIndex: 0
    )
    let cpuT1 = try cpuArtifactLlamaLayerHiddenLineage(
        weightDir: weightURL,
        config: config,
        tokens: [9707, 21806],
        throughLayerIndex: 0
    )
    let inputs = [cpuT0[0], cpuT1[0]]

    let runtime = try RealModelInferenceEngine.evalHybridSingleLlamaLayerDetailedFromInputsForTesting(
        config: config,
        weightDir: weightDir,
        layer: 1,
        inputs: inputs
    )
    let cpu = try cpuSingleLlamaLayerStepOutputs(
        weightDir: weightURL,
        config: config,
        layerIndex: 1,
        inputs: inputs
    )[1]

    let contextDiff = zip(runtime.context, cpu.context).map { abs($0 - $1) }
    let contextMaxDiff = contextDiff.max() ?? .infinity
    let contextMeanDiff = contextDiff.reduce(0, +) / Float(contextDiff.count)

    let projectionDiff = zip(runtime.projectionOut, cpu.projected).map { abs($0 - $1) }
    let projectionMaxDiff = projectionDiff.max() ?? .infinity
    let projectionMeanDiff = projectionDiff.reduce(0, +) / Float(projectionDiff.count)

    let hiddenDiff = zip(runtime.hidden, cpu.hidden).map { abs($0 - $1) }
    let hiddenMaxDiff = hiddenDiff.max() ?? .infinity
    let hiddenMeanDiff = hiddenDiff.reduce(0, +) / Float(hiddenDiff.count)

    fputs(
        "[qwen-debug-layer1-exact-stages] contextMaxDiff=\(contextMaxDiff) contextMeanDiff=\(contextMeanDiff) projectionMaxDiff=\(projectionMaxDiff) projectionMeanDiff=\(projectionMeanDiff) hiddenMaxDiff=\(hiddenMaxDiff) hiddenMeanDiff=\(hiddenMeanDiff)\n",
        stderr
    )

    #expect(contextMaxDiff < 0.01)
    #expect(contextMeanDiff < 0.001)
    #expect(projectionMaxDiff < 0.02)
    #expect(projectionMeanDiff < 0.005)
    #expect(hiddenMaxDiff < 0.02)
    #expect(hiddenMeanDiff < 0.005)
}

@Test func test_debugQwenLayer1ProjectionMatchesCPUWhenFedExactContext() throws {
    guard shouldRunLegacyQwenExperimentTests() else { return }
    guard shouldRunANEHardwareTests() else { return }
    guard let weightDir = ProcessInfo.processInfo.environment["ESPRESSO_DEBUG_WEIGHT_DIR"], !weightDir.isEmpty else {
        return
    }
    guard ProcessInfo.processInfo.environment["ESPRESSO_DEBUG_CHECK_LAYER1_PROJECTION"] == "1" else {
        return
    }
    guard ProcessInfo.processInfo.environment["ESPRESSO_DISABLE_HYBRID_FUSED_POST_ATTENTION"] == "1" else {
        return
    }

    let metadataURL = URL(fileURLWithPath: weightDir, isDirectory: true).appendingPathComponent("metadata.json")
    let metadata = try JSONDecoder().decode(DebugWeightMetadataFile.self, from: Data(contentsOf: metadataURL))
    let config = try metadata.asConfig()
    let weightURL = URL(fileURLWithPath: weightDir, isDirectory: true)

    let cpuT0 = try cpuArtifactLlamaLayerHiddenLineage(
        weightDir: weightURL,
        config: config,
        tokens: [9707],
        throughLayerIndex: 0
    )
    let cpuT1 = try cpuArtifactLlamaLayerHiddenLineage(
        weightDir: weightURL,
        config: config,
        tokens: [9707, 21806],
        throughLayerIndex: 0
    )
    let detailed = try RealModelInferenceEngine.evalHybridSingleLlamaLayerDetailedFromInputsForTesting(
        config: config,
        weightDir: weightDir,
        layer: 1,
        inputs: [cpuT0[0], cpuT1[0]]
    )
    let projection = try RealModelInferenceEngine.evalHybridSingleLayerDecodeProjectionForTesting(
        config: config,
        weightDir: weightDir,
        layer: 1,
        context: detailed.context
    )

    let woURL = weightURL.appendingPathComponent("layers/1/wo.bin")
    let wo = try readBlobFloat16File(at: woURL, expectedCount: config.dModel * config.attentionDim)
    let expected = multiplyRowMajorFlatMatrix(
        matrix: wo,
        rows: config.dModel,
        cols: config.attentionDim,
        vector: detailed.context
    )
    let diff = zip(projection.output, expected).map { abs($0 - $1) }
    let maxDiff = diff.max() ?? .infinity
    let meanDiff = diff.reduce(0, +) / Float(diff.count)

    fputs("[qwen-debug-layer1-projection] maxDiff=\(maxDiff) meanDiff=\(meanDiff)\n", stderr)

    #expect(maxDiff < 0.01)
    #expect(meanDiff < 0.001)
}

@Test func test_debugQwenLayer1ProjectionWithResidualMatchesCPU() throws {
    guard shouldRunLegacyQwenExperimentTests() else { return }
    guard shouldRunANEHardwareTests() else { return }
    guard let weightDir = ProcessInfo.processInfo.environment["ESPRESSO_DEBUG_WEIGHT_DIR"], !weightDir.isEmpty else {
        return
    }
    guard ProcessInfo.processInfo.environment["ESPRESSO_DEBUG_CHECK_LAYER1_PROJECTION_RESIDUAL"] == "1" else {
        return
    }
    guard ProcessInfo.processInfo.environment["ESPRESSO_DISABLE_HYBRID_FUSED_POST_ATTENTION"] == "1" else {
        return
    }

    let metadataURL = URL(fileURLWithPath: weightDir, isDirectory: true).appendingPathComponent("metadata.json")
    let metadata = try JSONDecoder().decode(DebugWeightMetadataFile.self, from: Data(contentsOf: metadataURL))
    let config = try metadata.asConfig()
    let weightURL = URL(fileURLWithPath: weightDir, isDirectory: true)

    let cpuT0 = try cpuArtifactLlamaLayerHiddenLineage(
        weightDir: weightURL,
        config: config,
        tokens: [9707],
        throughLayerIndex: 0
    )
    let cpuT1 = try cpuArtifactLlamaLayerHiddenLineage(
        weightDir: weightURL,
        config: config,
        tokens: [9707, 21806],
        throughLayerIndex: 0
    )
    let detailed = try RealModelInferenceEngine.evalHybridSingleLlamaLayerDetailedFromInputsForTesting(
        config: config,
        weightDir: weightDir,
        layer: 1,
        inputs: [cpuT0[0], cpuT1[0]]
    )
    let projection = try RealModelInferenceEngine.evalHybridSingleLayerDecodeProjectionForTesting(
        config: config,
        weightDir: weightDir,
        layer: 1,
        context: detailed.context,
        residual: cpuT1[0]
    )

    let woURL = weightURL.appendingPathComponent("layers/1/wo.bin")
    let wo = try readBlobFloat16File(at: woURL, expectedCount: config.dModel * config.attentionDim)
    let expected = zip(
        cpuT1[0],
        multiplyRowMajorFlatMatrix(
            matrix: wo,
            rows: config.dModel,
            cols: config.attentionDim,
            vector: detailed.context
        )
    ).map(+)

    let diff = zip(projection.output, expected).map { abs($0 - $1) }
    let maxDiff = diff.max() ?? .infinity
    let meanDiff = diff.reduce(0, +) / Float(diff.count)

    fputs("[qwen-debug-layer1-projection-residual] maxDiff=\(maxDiff) meanDiff=\(meanDiff)\n", stderr)

    #expect(maxDiff < 0.01)
    #expect(meanDiff < 0.001)
}

@Test func test_debugQwenLayer1FFNMatchesCPU() throws {
    guard shouldRunLegacyQwenExperimentTests() else { return }
    guard shouldRunANEHardwareTests() else { return }
    guard let weightDir = ProcessInfo.processInfo.environment["ESPRESSO_DEBUG_WEIGHT_DIR"], !weightDir.isEmpty else {
        return
    }
    guard ProcessInfo.processInfo.environment["ESPRESSO_DEBUG_CHECK_LAYER1_FFN"] == "1" else {
        return
    }

    let metadataURL = URL(fileURLWithPath: weightDir, isDirectory: true).appendingPathComponent("metadata.json")
    let metadata = try JSONDecoder().decode(DebugWeightMetadataFile.self, from: Data(contentsOf: metadataURL))
    let config = try metadata.asConfig()
    let weightURL = URL(fileURLWithPath: weightDir, isDirectory: true)

    let cpuT0 = try cpuArtifactLlamaLayerHiddenLineage(
        weightDir: weightURL,
        config: config,
        tokens: [9707],
        throughLayerIndex: 0
    )
    let cpuT1 = try cpuArtifactLlamaLayerHiddenLineage(
        weightDir: weightURL,
        config: config,
        tokens: [9707, 21806],
        throughLayerIndex: 0
    )
    let detailed = try RealModelInferenceEngine.evalHybridSingleLlamaLayerDetailedFromInputsForTesting(
        config: config,
        weightDir: weightDir,
        layer: 1,
        inputs: [cpuT0[0], cpuT1[0]]
    )
    let woURL = weightURL.appendingPathComponent("layers/1/wo.bin")
    let wo = try readBlobFloat16File(at: woURL, expectedCount: config.dModel * config.attentionDim)
    let projected = zip(
        cpuT1[0],
        multiplyRowMajorFlatMatrix(
            matrix: wo,
            rows: config.dModel,
            cols: config.attentionDim,
            vector: detailed.context
        )
    ).map(+)

    let ffn = try RealModelInferenceEngine.evalHybridSingleLayerDecodeFFNForTesting(
        config: config,
        weightDir: weightDir,
        layer: 1,
        input: projected
    )

    let layerURL = weightURL.appendingPathComponent("layers/1", isDirectory: true)
    let rmsFfn = try readBlobFloat16File(at: layerURL.appendingPathComponent("rms_ffn.bin"), expectedCount: config.dModel)
    let w1 = try readBlobFloat16File(at: layerURL.appendingPathComponent("w1.bin"), expectedCount: config.hiddenDim * config.dModel)
    let w2 = try readBlobFloat16File(at: layerURL.appendingPathComponent("w2.bin"), expectedCount: config.dModel * config.hiddenDim)
    let w3 = try readBlobFloat16File(at: layerURL.appendingPathComponent("w3.bin"), expectedCount: config.hiddenDim * config.dModel)
    let ffnNormed = rmsNorm(projected, weight: rmsFfn, eps: Float(config.normEps))
    let gate = multiplyRowMajorFlatMatrix(matrix: w1, rows: config.hiddenDim, cols: config.dModel, vector: ffnNormed)
    let up = multiplyRowMajorFlatMatrix(matrix: w3, rows: config.hiddenDim, cols: config.dModel, vector: ffnNormed)
    let down = multiplyRowMajorFlatMatrix(
        matrix: w2,
        rows: config.dModel,
        cols: config.hiddenDim,
        vector: zip(gate, up).map { silu($0) * $1 }
    )
    let expected = zip(projected, down).map(+)

    let diff = zip(ffn.output, expected).map { abs($0 - $1) }
    let maxDiff = diff.max() ?? .infinity
    let meanDiff = diff.reduce(0, +) / Float(diff.count)

    fputs("[qwen-debug-layer1-ffn] maxDiff=\(maxDiff) meanDiff=\(meanDiff)\n", stderr)

    #expect(maxDiff < 0.02)
    #expect(meanDiff < 0.005)
}

@Test func test_debugQwenLayer0FFNMatchesCPUForPromptToken1() throws {
    guard shouldRunLegacyQwenExperimentTests() else { return }
    guard shouldRunANEHardwareTests() else { return }
    guard let weightDir = ProcessInfo.processInfo.environment["ESPRESSO_DEBUG_WEIGHT_DIR"], !weightDir.isEmpty else {
        return
    }
    guard ProcessInfo.processInfo.environment["ESPRESSO_DEBUG_CHECK_LAYER0_FFN"] == "1" else {
        return
    }

    let metadataURL = URL(fileURLWithPath: weightDir, isDirectory: true).appendingPathComponent("metadata.json")
    let metadata = try JSONDecoder().decode(DebugWeightMetadataFile.self, from: Data(contentsOf: metadataURL))
    let config = try metadata.asConfig()
    let weightURL = URL(fileURLWithPath: weightDir, isDirectory: true)
    let tokenEmbedding = try readBlobFloat16File(
        at: weightURL.appendingPathComponent("embeddings/token.bin"),
        expectedCount: config.vocab * config.dModel
    )
    let promptTokens = [9707, 21806]
    let inputs = promptTokens.map { token in
        Array(tokenEmbedding[token * config.dModel..<(token + 1) * config.dModel])
    }
    let cpu = try cpuSingleLlamaLayerStepOutputs(
        weightDir: weightURL,
        config: config,
        layerIndex: 0,
        inputs: inputs
    )[1]

    let ffn = try RealModelInferenceEngine.evalHybridSingleLayerDecodeFFNForTesting(
        config: config,
        weightDir: weightDir,
        layer: 0,
        input: cpu.projected
    )
    let diff = zip(ffn.output, cpu.hidden).map { abs($0 - $1) }
    let maxDiff = diff.max() ?? .infinity
    let meanDiff = diff.reduce(0, +) / Float(diff.count)

    fputs("[qwen-debug-layer0-ffn] maxDiff=\(maxDiff) meanDiff=\(meanDiff)\n", stderr)

    #expect(maxDiff < 0.02)
    #expect(meanDiff < 0.005)
}

@Test func test_debugQwenLayer0FFNMatchesStrictRoundedCPUForPromptToken1() throws {
    guard shouldRunLegacyQwenExperimentTests() else { return }
    guard shouldRunANEHardwareTests() else { return }
    guard let weightDir = ProcessInfo.processInfo.environment["ESPRESSO_DEBUG_WEIGHT_DIR"], !weightDir.isEmpty else {
        return
    }
    guard ProcessInfo.processInfo.environment["ESPRESSO_DEBUG_CHECK_LAYER0_FFN_STRICT"] == "1" else {
        return
    }

    let metadataURL = URL(fileURLWithPath: weightDir, isDirectory: true).appendingPathComponent("metadata.json")
    let metadata = try JSONDecoder().decode(DebugWeightMetadataFile.self, from: Data(contentsOf: metadataURL))
    let config = try metadata.asConfig()
    let weightURL = URL(fileURLWithPath: weightDir, isDirectory: true)
    let tokenEmbedding = try readBlobFloat16File(
        at: weightURL.appendingPathComponent("embeddings/token.bin"),
        expectedCount: config.vocab * config.dModel
    )
    let promptTokens = [9707, 21806]
    let inputs = promptTokens.map { token in
        Array(tokenEmbedding[token * config.dModel..<(token + 1) * config.dModel])
    }
    let cpu = try cpuSingleLlamaLayerStepOutputs(
        weightDir: weightURL,
        config: config,
        layerIndex: 0,
        inputs: inputs,
        roundFFNIntermediatesToFP16: true
    )[1]

    let ffn = try RealModelInferenceEngine.evalHybridSingleLayerDecodeFFNForTesting(
        config: config,
        weightDir: weightDir,
        layer: 0,
        input: cpu.projected
    )
    let diff = zip(ffn.output, cpu.hidden).map { abs($0 - $1) }
    let maxDiff = diff.max() ?? .infinity
    let meanDiff = diff.reduce(0, +) / Float(diff.count)

    fputs("[qwen-debug-layer0-ffn-strict] maxDiff=\(maxDiff) meanDiff=\(meanDiff)\n", stderr)

    #expect(maxDiff < 0.005)
    #expect(meanDiff < 0.001)
}

@Test func test_debugQwenLayer0FFNPostNormMatchesCPUForPromptToken1() throws {
    guard shouldRunLegacyQwenExperimentTests() else { return }
    guard shouldRunANEHardwareTests() else { return }
    guard let weightDir = ProcessInfo.processInfo.environment["ESPRESSO_DEBUG_WEIGHT_DIR"], !weightDir.isEmpty else {
        return
    }
    guard ProcessInfo.processInfo.environment["ESPRESSO_DEBUG_CHECK_LAYER0_FFN_POST_NORM"] == "1" else {
        return
    }

    let metadataURL = URL(fileURLWithPath: weightDir, isDirectory: true).appendingPathComponent("metadata.json")
    let metadata = try JSONDecoder().decode(DebugWeightMetadataFile.self, from: Data(contentsOf: metadataURL))
    let config = try metadata.asConfig()
    let weightURL = URL(fileURLWithPath: weightDir, isDirectory: true)
    let tokenEmbedding = try readBlobFloat16File(
        at: weightURL.appendingPathComponent("embeddings/token.bin"),
        expectedCount: config.vocab * config.dModel
    )
    let promptTokens = [9707, 21806]
    let inputs = promptTokens.map { token in
        Array(tokenEmbedding[token * config.dModel..<(token + 1) * config.dModel])
    }
    let cpu = try cpuSingleLlamaLayerStepOutputs(
        weightDir: weightURL,
        config: config,
        layerIndex: 0,
        inputs: inputs
    )[1]
    let rmsFfn = try readBlobFloat16File(
        at: weightURL.appendingPathComponent("layers/0/rms_ffn.bin"),
        expectedCount: config.dModel
    )
    let normalized = rmsNorm(cpu.projected, weight: rmsFfn, eps: Float(config.normEps))

    let ffn = try RealModelInferenceEngine.evalHybridSingleLayerDecodeFFNPostNormForTesting(
        config: config,
        weightDir: weightDir,
        layer: 0,
        normalizedInput: normalized,
        residual: cpu.projected
    )
    let diff = zip(ffn.output, cpu.hidden).map { abs($0 - $1) }
    let maxDiff = diff.max() ?? .infinity
    let meanDiff = diff.reduce(0, +) / Float(diff.count)

    fputs("[qwen-debug-layer0-ffn-post-norm] maxDiff=\(maxDiff) meanDiff=\(meanDiff)\n", stderr)

    #expect(maxDiff < 0.005)
    #expect(meanDiff < 0.001)
}

@Test func test_debugQwenLayer0FFNStagesMatchCPUForPromptToken1() throws {
    guard shouldRunLegacyQwenExperimentTests() else { return }
    guard shouldRunANEHardwareTests() else { return }
    guard let weightDir = ProcessInfo.processInfo.environment["ESPRESSO_DEBUG_WEIGHT_DIR"], !weightDir.isEmpty else {
        return
    }
    guard ProcessInfo.processInfo.environment["ESPRESSO_DEBUG_CHECK_LAYER0_FFN_STAGES"] == "1" else {
        return
    }

    let metadataURL = URL(fileURLWithPath: weightDir, isDirectory: true).appendingPathComponent("metadata.json")
    let metadata = try JSONDecoder().decode(DebugWeightMetadataFile.self, from: Data(contentsOf: metadataURL))
    let config = try metadata.asConfig()
    let weightURL = URL(fileURLWithPath: weightDir, isDirectory: true)
    let tokenEmbedding = try readBlobFloat16File(
        at: weightURL.appendingPathComponent("embeddings/token.bin"),
        expectedCount: config.vocab * config.dModel
    )
    let promptTokens = [9707, 21806]
    let inputs = promptTokens.map { token in
        Array(tokenEmbedding[token * config.dModel..<(token + 1) * config.dModel])
    }
    let cpu = try cpuSingleLlamaLayerStepOutputs(
        weightDir: weightURL,
        config: config,
        layerIndex: 0,
        inputs: inputs
    )[1]

    let ffnStages = try RealModelInferenceEngine.evalHybridSingleLayerDecodeFFNStagesForTesting(
        config: config,
        weightDir: weightDir,
        layer: 0,
        normalizedInput: cpu.ffnNormed
    )

    func summarize(_ label: String, _ actual: [Float], _ expected: [Float]) {
        let diff = zip(actual, expected).map { abs($0 - $1) }
        let maxDiff = diff.max() ?? .infinity
        let meanDiff = diff.reduce(0, +) / Float(diff.count)
        fputs("[qwen-debug-layer0-ffn-stage] \(label) maxDiff=\(maxDiff) meanDiff=\(meanDiff)\n", stderr)
    }

    summarize("gate", ffnStages.gateLinear, cpu.gate)
    summarize("up", ffnStages.upLinear, cpu.up)
    summarize("silu", ffnStages.siluGate, cpu.gate.map(silu))
    summarize("gated", ffnStages.gated, cpu.activated)
    summarize("down", ffnStages.down, cpu.down)

    summarize("silu_local", ffnStages.siluGate, ffnStages.gateLinear.map(silu))
    summarize(
        "gated_local",
        ffnStages.gated,
        zip(ffnStages.siluGate, ffnStages.upLinear).map(*)
    )
    let layerURL = weightURL.appendingPathComponent("layers/0", isDirectory: true)
    let w2 = try readBlobFloat16File(
        at: layerURL.appendingPathComponent("w2.bin"),
        expectedCount: config.dModel * config.hiddenDim
    )
    summarize(
        "down_local",
        ffnStages.down,
        multiplyRowMajorFlatMatrix(
            matrix: w2,
            rows: config.dModel,
            cols: config.hiddenDim,
            vector: ffnStages.gated
        )
    )
}

@Test func test_debugQwenPromptToken1FFNStagesMatchCPUForLayerFromEnv() throws {
    guard shouldRunLegacyQwenExperimentTests() else { return }
    guard shouldRunANEHardwareTests() else { return }
    guard let weightDir = ProcessInfo.processInfo.environment["ESPRESSO_DEBUG_WEIGHT_DIR"], !weightDir.isEmpty else {
        return
    }
    guard ProcessInfo.processInfo.environment["ESPRESSO_DEBUG_CHECK_PROMPT_TOKEN1_FFN_STAGES"] == "1" else {
        return
    }
    guard let rawLayer = ProcessInfo.processInfo.environment["ESPRESSO_DEBUG_LAYER_INDEX"],
          let layerIndex = Int(rawLayer) else {
        Issue.record("Set ESPRESSO_DEBUG_LAYER_INDEX for the FFN stage layer probe")
        return
    }

    let metadataURL = URL(fileURLWithPath: weightDir, isDirectory: true).appendingPathComponent("metadata.json")
    let metadata = try JSONDecoder().decode(DebugWeightMetadataFile.self, from: Data(contentsOf: metadataURL))
    let config = try metadata.asConfig()
    #expect(layerIndex >= 0)
    #expect(layerIndex < config.nLayer)

    let weightURL = URL(fileURLWithPath: weightDir, isDirectory: true)
    let promptTokens = debugPromptTokens(
        env: ProcessInfo.processInfo.environment,
        defaultTokens: [9707, 21806]
    )
    let inputs: [[Float]]
    if layerIndex == 0 {
        let tokenEmbedding = try readBlobFloat16File(
            at: weightURL.appendingPathComponent("embeddings/token.bin"),
            expectedCount: config.vocab * config.dModel
        )
        inputs = promptTokens.map { token in
            Array(tokenEmbedding[token * config.dModel..<(token + 1) * config.dModel])
        }
    } else {
        let cpuT0 = try cpuArtifactLlamaLayerHiddenLineage(
            weightDir: weightURL,
            config: config,
            tokens: [promptTokens[0]],
            throughLayerIndex: layerIndex - 1
        )
        let cpuT1 = try cpuArtifactLlamaLayerHiddenLineage(
            weightDir: weightURL,
            config: config,
            tokens: promptTokens,
            throughLayerIndex: layerIndex - 1
        )
        inputs = [cpuT0[layerIndex - 1], cpuT1[layerIndex - 1]]
    }

    let cpu = try cpuSingleLlamaLayerStepOutputs(
        weightDir: weightURL,
        config: config,
        layerIndex: layerIndex,
        inputs: inputs
    )[1]

    let ffnStages = try RealModelInferenceEngine.evalHybridSingleLayerDecodeFFNStagesForTesting(
        config: config,
        weightDir: weightDir,
        layer: layerIndex,
        normalizedInput: cpu.ffnNormed
    )

    func summarize(_ label: String, _ actual: [Float], _ expected: [Float]) {
        let diff = zip(actual, expected).map { abs($0 - $1) }
        let maxDiff = diff.max() ?? .infinity
        let meanDiff = diff.reduce(0, +) / Float(diff.count)
        fputs(
            "[qwen-debug-prompt-token1-ffn-stage] layer=\(layerIndex) \(label) maxDiff=\(maxDiff) meanDiff=\(meanDiff)\n",
            stderr
        )
    }

    summarize("gate", ffnStages.gateLinear, cpu.gate)
    summarize("up", ffnStages.upLinear, cpu.up)
    summarize("silu", ffnStages.siluGate, cpu.gate.map(silu))
    summarize("gated", ffnStages.gated, cpu.activated)
    summarize("down", ffnStages.down, cpu.down)
    summarize("silu_local", ffnStages.siluGate, ffnStages.gateLinear.map(silu))
    summarize("gated_local", ffnStages.gated, zip(ffnStages.siluGate, ffnStages.upLinear).map(*))

    let layerURL = weightURL.appendingPathComponent("layers/\(layerIndex)", isDirectory: true)
    let w2 = try readBlobFloat16File(
        at: layerURL.appendingPathComponent("w2.bin"),
        expectedCount: config.dModel * config.hiddenDim
    )
    summarize(
        "down_local",
        ffnStages.down,
        multiplyRowMajorFlatMatrix(
            matrix: w2,
            rows: config.dModel,
            cols: config.hiddenDim,
            vector: ffnStages.gated
        )
    )
}

@Test func test_debugQwenLayer0HybridOutputMatchesCPUFromWeightDir() throws {
    guard shouldRunLegacyQwenExperimentTests() else { return }
    guard shouldRunANEHardwareTests() else { return }
    guard let weightDir = ProcessInfo.processInfo.environment["ESPRESSO_DEBUG_WEIGHT_DIR"], !weightDir.isEmpty else {
        return
    }
    guard ProcessInfo.processInfo.environment["ESPRESSO_DEBUG_CHECK_LAYER0_HYBRID_OUTPUT"] == "1" else {
        return
    }

    let metadataURL = URL(fileURLWithPath: weightDir, isDirectory: true).appendingPathComponent("metadata.json")
    let metadata = try JSONDecoder().decode(DebugWeightMetadataFile.self, from: Data(contentsOf: metadataURL))
    let config = try metadata.asConfig()
    let token: TokenID = 9707

    let hybrid = try RealModelInferenceEngine.evalHybridSingleLayerForTesting(
        config: config,
        weightDir: weightDir,
        layer: 0,
        tokens: [token]
    )

    let rootURL = URL(fileURLWithPath: weightDir, isDirectory: true)
    let tokenEmb = try readBlobFloat16File(
        at: rootURL.appendingPathComponent("embeddings/token.bin"),
        expectedCount: config.vocab * config.dModel
    )
    let layerURL = rootURL.appendingPathComponent("layers/0", isDirectory: true)
    let rmsAtt = try readBlobFloat16File(at: layerURL.appendingPathComponent("rms_att.bin"), expectedCount: config.dModel)
    let wv = try readBlobFloat16File(at: layerURL.appendingPathComponent("wv.bin"), expectedCount: config.kvDim * config.dModel)
    let wo = try readBlobFloat16File(at: layerURL.appendingPathComponent("wo.bin"), expectedCount: config.dModel * config.attentionDim)
    let rmsFfn = try readBlobFloat16File(at: layerURL.appendingPathComponent("rms_ffn.bin"), expectedCount: config.dModel)
    let w1 = try readBlobFloat16File(at: layerURL.appendingPathComponent("w1.bin"), expectedCount: config.hiddenDim * config.dModel)
    let w2 = try readBlobFloat16File(at: layerURL.appendingPathComponent("w2.bin"), expectedCount: config.dModel * config.hiddenDim)
    let w3 = try readBlobFloat16File(at: layerURL.appendingPathComponent("w3.bin"), expectedCount: config.hiddenDim * config.dModel)

    let x = Array(tokenEmb[Int(token) * config.dModel..<(Int(token) + 1) * config.dModel])
    let attnNormed = rmsNorm(x, weight: rmsAtt, eps: Float(config.normEps))
    let v = multiplyRowMajorFlatMatrix(
        matrix: wv,
        rows: config.kvDim,
        cols: config.dModel,
        vector: attnNormed
    )
    let context = groupedContextFromVOut(v, heads: config.nHead, kvHeads: config.nKVHead, headDim: config.headDim)
    let projected = zip(
        multiplyRowMajorFlatMatrix(
            matrix: wo,
            rows: config.dModel,
            cols: config.attentionDim,
            vector: context
        ),
        x
    ).map { $0 + $1 }
    let ffnNormed = rmsNorm(projected, weight: rmsFfn, eps: Float(config.normEps))
    let gate = multiplyRowMajorFlatMatrix(
        matrix: w1,
        rows: config.hiddenDim,
        cols: config.dModel,
        vector: ffnNormed
    )
    let up = multiplyRowMajorFlatMatrix(
        matrix: w3,
        rows: config.hiddenDim,
        cols: config.dModel,
        vector: ffnNormed
    )
    let activated = zip(gate, up).map { silu($0) * $1 }
    let down = multiplyRowMajorFlatMatrix(
        matrix: w2,
        rows: config.dModel,
        cols: config.hiddenDim,
        vector: activated
    )
    let cpu = zip(projected, down).map { $0 + $1 }

    let maxDiff = zip(hybrid, cpu).map { abs($0 - $1) }.max() ?? .infinity
    let meanDiff = zip(hybrid, cpu).map { abs($0 - $1) }.reduce(0, +) / Float(config.dModel)
    fputs("[qwen-debug-layer0-hybrid-output] maxDiff=\(maxDiff) meanDiff=\(meanDiff)\n", stderr)

    #expect(maxDiff < 0.02)
    #expect(meanDiff < 0.005)
}

@Test func test_debugQwenLayer0HookedKCacheMatchesCPUFromWeightDir() throws {
    guard shouldRunLegacyQwenExperimentTests() else { return }
    guard shouldRunANEHardwareTests() else { return }
    guard let weightDir = ProcessInfo.processInfo.environment["ESPRESSO_DEBUG_WEIGHT_DIR"], !weightDir.isEmpty else {
        return
    }
    guard ProcessInfo.processInfo.environment["ESPRESSO_DEBUG_CHECK_LAYER0_HOOKED_KCACHE"] == "1" else {
        return
    }

    let metadataURL = URL(fileURLWithPath: weightDir, isDirectory: true).appendingPathComponent("metadata.json")
    let metadata = try JSONDecoder().decode(DebugWeightMetadataFile.self, from: Data(contentsOf: metadataURL))
    let config = try metadata.asConfig()
    let promptTokens = debugPromptTokens(
        env: ProcessInfo.processInfo.environment,
        defaultTokens: [9707, 21806]
    )
    #expect(promptTokens.count >= 2)

    let outputs = try RealModelInferenceEngine.evalHybridSingleLayerHookedLlamaKCacheForTesting(
        config: config,
        weightDir: weightDir,
        layer: 0,
        tokens: promptTokens.map(TokenID.init)
    )

    let layerURL = URL(fileURLWithPath: weightDir, isDirectory: true).appendingPathComponent("layers/0", isDirectory: true)
    let kNorm = try readBlobFloat16File(
        at: layerURL.appendingPathComponent("k_norm.bin"),
        expectedCount: config.headDim
    )
    let tokenIndex = promptTokens.count - 1
    let expectedHookedK = applyHalfSplitRoPEPerHead(
        perHeadRMSNorm(outputs.rawKOut, weights: kNorm, heads: config.nKVHead, headDim: config.headDim, eps: Float(config.normEps)),
        heads: config.nKVHead,
        headDim: config.headDim,
        position: tokenIndex,
        theta: config.ropeTheta
    )
    let expectedHookedKFP16 = expectedHookedK.map { Float(Float16($0)) }
    let hookedMaxDiff = zip(outputs.hookedKOut, expectedHookedK).map { abs($0 - $1) }.max() ?? .infinity
    let hookedMeanDiff = zip(outputs.hookedKOut, expectedHookedK).map { abs($0 - $1) }.reduce(0, +) / Float(expectedHookedK.count)
    let hookedSurfaceMaxDiff = zip(outputs.hookedKOutSurface, expectedHookedKFP16).map { abs($0 - $1) }.max() ?? .infinity
    let hookedSurfaceMeanDiff = zip(outputs.hookedKOutSurface, expectedHookedKFP16).map { abs($0 - $1) }.reduce(0, +) / Float(expectedHookedKFP16.count)

    let expectedTokenSlice = expectedHookedKFP16
    let cacheStride = promptTokens.count
    let cacheTokenSlice = stride(from: 0, to: outputs.kCache.count, by: cacheStride).map { outputs.kCache[$0 + tokenIndex] }
    let cacheMaxDiff = zip(cacheTokenSlice, expectedTokenSlice).map { abs($0 - $1) }.max() ?? .infinity
    let cacheMeanDiff = zip(cacheTokenSlice, expectedTokenSlice).map { abs($0 - $1) }.reduce(0, +) / Float(expectedTokenSlice.count)

    fputs(
        "[qwen-debug-layer0-hooked-kcache] hookedMaxDiff=\(hookedMaxDiff) hookedMeanDiff=\(hookedMeanDiff) hookedSurfaceMaxDiff=\(hookedSurfaceMaxDiff) hookedSurfaceMeanDiff=\(hookedSurfaceMeanDiff) cacheMaxDiff=\(cacheMaxDiff) cacheMeanDiff=\(cacheMeanDiff)\n",
        stderr
    )

    #expect(hookedMaxDiff < 0.01)
    #expect(hookedMeanDiff < 0.001)
    #expect(hookedSurfaceMaxDiff < 0.01)
    #expect(hookedSurfaceMeanDiff < 0.001)
    #expect(cacheMaxDiff < 0.01)
    #expect(cacheMeanDiff < 0.001)
}

@Test func test_debugQwenLayer0AttentionStateMatchesCPUFromWeightDir() throws {
    guard shouldRunLegacyQwenExperimentTests() else { return }
    guard shouldRunANEHardwareTests() else { return }
    guard let weightDir = ProcessInfo.processInfo.environment["ESPRESSO_DEBUG_WEIGHT_DIR"], !weightDir.isEmpty else {
        return
    }
    guard ProcessInfo.processInfo.environment["ESPRESSO_DEBUG_CHECK_LAYER0_ATTENTION_STATE"] == "1" else {
        return
    }

    let metadataURL = URL(fileURLWithPath: weightDir, isDirectory: true).appendingPathComponent("metadata.json")
    let metadata = try JSONDecoder().decode(DebugWeightMetadataFile.self, from: Data(contentsOf: metadataURL))
    let config = try metadata.asConfig()
    let weightURL = URL(fileURLWithPath: weightDir, isDirectory: true)
    let promptTokens = debugPromptTokens(
        env: ProcessInfo.processInfo.environment,
        defaultTokens: [9707, 21806]
    )
    #expect(promptTokens.count >= 2)

    let tokenEmbedding = try readBlobFloat16File(
        at: weightURL.appendingPathComponent("embeddings/token.bin"),
        expectedCount: config.vocab * config.dModel
    )
    let inputs = promptTokens.map { token in
        Array(tokenEmbedding[token * config.dModel..<(token + 1) * config.dModel])
    }

    let runtime = try RealModelInferenceEngine.evalHybridSingleLlamaLayerDetailedFromInputsForTesting(
        config: config,
        weightDir: weightDir,
        layer: 0,
        inputs: inputs
    )
    let cpu = try cpuSingleLlamaLayerAttentionState(
        weightDir: weightURL,
        config: config,
        layerIndex: 0,
        inputs: inputs
    )

    func summarize(_ label: String, _ actual: [Float], _ expected: [Float]) {
        let diff = zip(actual, expected).map { abs($0 - $1) }
        let maxDiff = diff.max() ?? .infinity
        let meanDiff = diff.reduce(0, +) / Float(diff.count)
        fputs("[qwen-debug-layer0-attention-state] \(label) maxDiff=\(maxDiff) meanDiff=\(meanDiff)\n", stderr)
    }

    summarize("qOut", runtime.qOut, cpu.qOut)
    summarize("kCache", runtime.kCache, cpu.kCache)
    summarize("vCache", runtime.vCache, cpu.vCache)
}

private enum TopLevelNaming {
    case canonical
    case aliases
}

private func expectRealModelInferenceError(
    containing needle: String,
    _ body: () throws -> Void
) throws {
    do {
        try body()
        Issue.record("Expected RealModelInferenceError containing '\(needle)'")
    } catch let error as RealModelInferenceError {
        #expect(error.localizedDescription.localizedCaseInsensitiveContains(needle))
    } catch {
        Issue.record("Expected RealModelInferenceError, got \(type(of: error)): \(error)")
    }
}

private func makeTinyGPT2Config() -> MultiModelConfig {
    MultiModelConfig(
        name: "tiny-gpt2",
        nLayer: 1,
        nHead: 2,
        nKVHead: 2,
        dModel: 8,
        headDim: 4,
        hiddenDim: 32,
        vocab: 64,
        maxSeq: 8,
        normEps: 1e-5,
        architecture: .gpt2
    )
}

private func makeHardwareGPT2Config() -> MultiModelConfig {
    MultiModelConfig(
        name: "hardware-gpt2",
        nLayer: 1,
        nHead: 12,
        nKVHead: 12,
        dModel: 768,
        headDim: 64,
        hiddenDim: 3_072,
        vocab: 512,
        maxSeq: 32,
        normEps: 1e-5,
        architecture: .gpt2
    )
}

private func makeMinimalGPT2WeightDirectory(
    config: MultiModelConfig,
    topLevelNaming: TopLevelNaming = .canonical,
    omitTopLevelFiles: Set<String> = []
) throws -> URL {
    let root = try makeTempDirectory()
    let embeddingsDir = root.appendingPathComponent("embeddings", isDirectory: true)
    let layerDir = root
        .appendingPathComponent("layers", isDirectory: true)
        .appendingPathComponent("0", isDirectory: true)
    let maskDir = root.appendingPathComponent("masks", isDirectory: true)

    try FileManager.default.createDirectory(at: embeddingsDir, withIntermediateDirectories: true)
    try FileManager.default.createDirectory(at: layerDir, withIntermediateDirectories: true)
    try FileManager.default.createDirectory(at: maskDir, withIntermediateDirectories: true)

    let topLevelNames: [String: String] = switch topLevelNaming {
    case .canonical:
        [
            "tokenEmbedding": "embeddings/token.bin",
            "positionEmbedding": "embeddings/position.bin",
            "finalNormGamma": "final_norm_gamma.bin",
            "finalNormBeta": "final_norm_beta.bin",
            "lmHead": "lm_head.bin",
        ]
    case .aliases:
        [
            "tokenEmbedding": "embeddings/token_embeddings.bin",
            "positionEmbedding": "embeddings/position_embeddings.bin",
            "finalNormGamma": "ln_f_gamma.bin",
            "finalNormBeta": "ln_f_beta.bin",
            "lmHead": "classifier.bin",
        ]
    }

    if !omitTopLevelFiles.contains("tokenEmbedding") {
        try writeBlob(
            repeating: 0.03125,
            count: config.vocab * config.dModel,
            to: root.appendingPathComponent(topLevelNames["tokenEmbedding"]!)
        )
    }
    if !omitTopLevelFiles.contains("positionEmbedding") {
        try writeBlob(
            repeating: 0.015625,
            count: config.maxSeq * config.dModel,
            to: root.appendingPathComponent(topLevelNames["positionEmbedding"]!)
        )
    }
    if !omitTopLevelFiles.contains("finalNormGamma") {
        try writeBlob(repeating: 1.0, count: config.dModel, to: root.appendingPathComponent(topLevelNames["finalNormGamma"]!))
    }
    if !omitTopLevelFiles.contains("finalNormBeta") {
        try writeBlob(repeating: 0.0, count: config.dModel, to: root.appendingPathComponent(topLevelNames["finalNormBeta"]!))
    }
    if !omitTopLevelFiles.contains("lmHead") {
        try writeBlob(
            repeating: 0.0078125,
            count: config.vocab * config.dModel,
            to: root.appendingPathComponent(topLevelNames["lmHead"]!)
        )
    }

    try writeBlob(repeating: 1.0, count: config.dModel, to: layerDir.appendingPathComponent("ln_1_gamma.bin"))
    try writeBlob(repeating: 0.0, count: config.dModel, to: layerDir.appendingPathComponent("ln_1_beta.bin"))
    try writeBlob(repeating: 0.03125, count: config.dModel * config.dModel, to: layerDir.appendingPathComponent("wq.bin"))
    try writeBlob(repeating: 0.015625, count: config.dModel * config.dModel, to: layerDir.appendingPathComponent("wk.bin"))
    try writeBlob(repeating: 0.0234375, count: config.dModel * config.dModel, to: layerDir.appendingPathComponent("wv.bin"))
    try writeBlob(repeating: 0.02734375, count: config.dModel * config.dModel, to: layerDir.appendingPathComponent("wo.bin"))
    try writeBlob(repeating: 0.0, count: config.dModel, to: layerDir.appendingPathComponent("bq.bin"))
    try writeBlob(repeating: 0.0, count: config.dModel, to: layerDir.appendingPathComponent("bk.bin"))
    try writeBlob(repeating: 0.0, count: config.dModel, to: layerDir.appendingPathComponent("bv.bin"))
    try writeBlob(repeating: 0.0, count: config.dModel, to: layerDir.appendingPathComponent("bo.bin"))
    try writeBlob(repeating: 1.0, count: config.dModel, to: layerDir.appendingPathComponent("ln_2_gamma.bin"))
    try writeBlob(repeating: 0.0, count: config.dModel, to: layerDir.appendingPathComponent("ln_2_beta.bin"))
    try writeBlob(repeating: 0.01953125, count: config.hiddenDim * config.dModel, to: layerDir.appendingPathComponent("w1.bin"))
    try writeBlob(repeating: 0.0, count: config.hiddenDim, to: layerDir.appendingPathComponent("b1.bin"))
    try writeBlob(repeating: 0.01171875, count: config.dModel * config.hiddenDim, to: layerDir.appendingPathComponent("w2.bin"))
    try writeBlob(repeating: 0.0, count: config.dModel, to: layerDir.appendingPathComponent("b2.bin"))

    var bucket = 1
    while bucket <= config.maxSeq {
        try writeCausalMask(spatial: bucket, to: maskDir.appendingPathComponent("causal_\(bucket).bin"))
        bucket *= 2
    }

    return root
}

private func makeTinyLlamaConfig() -> MultiModelConfig {
    MultiModelConfig(
        name: "tiny-llama",
        nLayer: 1,
        nHead: 2,
        nKVHead: 1,
        dModel: 8,
        headDim: 4,
        hiddenDim: 16,
        vocab: 32,
        maxSeq: 16,
        normEps: 1e-6,
        architecture: .llama
    )
}

private func makeMinimalLlamaLayerWeightDirectory(
    config: MultiModelConfig,
    qNorm: [Float]? = nil,
    kNorm: [Float]? = nil
) throws -> URL {
    let root = try makeTempDirectory()
    let layerDir = root
        .appendingPathComponent("layers", isDirectory: true)
        .appendingPathComponent("0", isDirectory: true)
    try FileManager.default.createDirectory(at: layerDir, withIntermediateDirectories: true)

    let qDim = config.attentionDim
    let kvDim = config.kvDim
    try writeBlob(repeating: 1.0, count: config.dModel, to: layerDir.appendingPathComponent("rms_att.bin"))
    try writeBlob(repeating: 0.03125, count: config.dModel * qDim, to: layerDir.appendingPathComponent("wq.bin"))
    try writeBlob(repeating: 0.015625, count: config.dModel * kvDim, to: layerDir.appendingPathComponent("wk.bin"))
    try writeBlob(repeating: 0.0234375, count: config.dModel * kvDim, to: layerDir.appendingPathComponent("wv.bin"))
    try writeBlob(repeating: 0.02734375, count: config.dModel * qDim, to: layerDir.appendingPathComponent("wo.bin"))
    try writeBlob(repeating: 1.0, count: config.dModel, to: layerDir.appendingPathComponent("rms_ffn.bin"))
    try writeBlob(repeating: 0.01953125, count: config.hiddenDim * config.dModel, to: layerDir.appendingPathComponent("w1.bin"))
    try writeBlob(repeating: 0.01171875, count: config.dModel * config.hiddenDim, to: layerDir.appendingPathComponent("w2.bin"))
    try writeBlob(repeating: 0.017578125, count: config.hiddenDim * config.dModel, to: layerDir.appendingPathComponent("w3.bin"))
    if let qNorm {
        try writeBlob(values: qNorm, to: layerDir.appendingPathComponent("q_norm.bin"))
    }
    if let kNorm {
        try writeBlob(values: kNorm, to: layerDir.appendingPathComponent("k_norm.bin"))
    }

    return root
}

private func populateMinimalLlamaWeightDirectory(
    at root: URL,
    config: MultiModelConfig
) throws {
    try FileManager.default.createDirectory(
        at: root.appendingPathComponent("embeddings", isDirectory: true),
        withIntermediateDirectories: true
    )
    try writeBlob(
        repeating: 0.03125,
        count: config.vocab * config.dModel,
        to: root.appendingPathComponent("embeddings/token.bin")
    )
    try writeBlob(
        repeating: 1.0,
        count: config.dModel,
        to: root.appendingPathComponent("rms_final.bin")
    )
    try writeBlob(
        repeating: 0.0078125,
        count: config.vocab * config.dModel,
        to: root.appendingPathComponent("lm_head.bin")
    )

    let metadata: [String: Any] = [
        "name": config.name,
        "nLayer": config.nLayer,
        "nHead": config.nHead,
        "nKVHead": config.nKVHead,
        "dModel": config.dModel,
        "headDim": config.headDim,
        "hiddenDim": config.hiddenDim,
        "vocab": config.vocab,
        "maxSeq": config.maxSeq,
        "normEps": config.normEps,
        "ropeTheta": config.ropeTheta,
        "architecture": "llama",
    ]
    try JSONSerialization.data(withJSONObject: metadata, options: [.sortedKeys]).write(
        to: root.appendingPathComponent("metadata.json")
    )

    let layerDir = root
        .appendingPathComponent("layers", isDirectory: true)
        .appendingPathComponent("0", isDirectory: true)
    try FileManager.default.createDirectory(at: layerDir, withIntermediateDirectories: true)

    let qDim = config.attentionDim
    let kvDim = config.kvDim
    try writeBlob(repeating: 1.0, count: config.dModel, to: layerDir.appendingPathComponent("rms_att.bin"))
    try writeBlob(repeating: 0.03125, count: config.dModel * qDim, to: layerDir.appendingPathComponent("wq.bin"))
    try writeBlob(repeating: 0.015625, count: config.dModel * kvDim, to: layerDir.appendingPathComponent("wk.bin"))
    try writeBlob(repeating: 0.0234375, count: config.dModel * kvDim, to: layerDir.appendingPathComponent("wv.bin"))
    try writeBlob(repeating: 0.02734375, count: config.dModel * qDim, to: layerDir.appendingPathComponent("wo.bin"))
    try writeBlob(repeating: 1.0, count: config.dModel, to: layerDir.appendingPathComponent("rms_ffn.bin"))
    try writeBlob(repeating: 0.01953125, count: config.hiddenDim * config.dModel, to: layerDir.appendingPathComponent("w1.bin"))
    try writeBlob(repeating: 0.01171875, count: config.dModel * config.hiddenDim, to: layerDir.appendingPathComponent("w2.bin"))
    try writeBlob(repeating: 0.017578125, count: config.hiddenDim * config.dModel, to: layerDir.appendingPathComponent("w3.bin"))
}

private func writeBlob(repeating value: Float, count: Int, to url: URL) throws {
    let data = makeBlobData(repeating: value, count: count)
    try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
    try data.write(to: url)
}

private func writeBlob(values: [Float], to url: URL) throws {
    let data = makeBlobData(from: values)
    try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
    try data.write(to: url)
}

private func readBlobFloat16File(at url: URL, expectedCount: Int) throws -> [Float] {
    let data = try Data(contentsOf: url)
    let byteOffset = 128
    let scalarSize = MemoryLayout<UInt16>.stride
    let expectedBytes = byteOffset + expectedCount * scalarSize
    guard data.count >= expectedBytes else {
        Issue.record("Blob at \(url.path) is too small: \(data.count) < \(expectedBytes)")
        return []
    }
    return data.withUnsafeBytes { raw in
        var result = [Float](repeating: 0, count: expectedCount)
        for index in 0..<expectedCount {
            let bitPattern = raw.loadUnaligned(
                fromByteOffset: byteOffset + index * scalarSize,
                as: UInt16.self
            )
            result[index] = Float(Float16(bitPattern: UInt16(littleEndian: bitPattern)))
        }
        return result
    }
}

private func multiplyRowMajorMatrix(rows: [[Float]], vector: [Float]) -> [Float] {
    rows.map { row in
        zip(row, vector).reduce(into: Float(0)) { partial, pair in
            partial += pair.0 * pair.1
        }
    }
}

private func headMajorToDimMajorContext(_ context: [Float], heads: Int, headDim: Int) -> [Float] {
    precondition(context.count == heads * headDim)
    var result = [Float](repeating: 0, count: context.count)
    for head in 0..<heads {
        for dim in 0..<headDim {
            result[dim * heads + head] = context[head * headDim + dim]
        }
    }
    return result
}

private func groupedContextFromVOut(_ vOut: [Float], heads: Int, kvHeads: Int, headDim: Int) -> [Float] {
    precondition(vOut.count == kvHeads * headDim)
    precondition(heads % kvHeads == 0)
    let groupSize = heads / kvHeads
    var result = [Float](repeating: 0, count: heads * headDim)
    for head in 0..<heads {
        let kvHead = head / groupSize
        let srcBase = kvHead * headDim
        let dstBase = head * headDim
        result[dstBase..<dstBase + headDim] = vOut[srcBase..<srcBase + headDim]
    }
    return result
}

private func moduloContextFromVOut(_ vOut: [Float], heads: Int, kvHeads: Int, headDim: Int) -> [Float] {
    precondition(vOut.count == kvHeads * headDim)
    var result = [Float](repeating: 0, count: heads * headDim)
    for head in 0..<heads {
        let kvHead = head % kvHeads
        let srcBase = kvHead * headDim
        let dstBase = head * headDim
        result[dstBase..<dstBase + headDim] = vOut[srcBase..<srcBase + headDim]
    }
    return result
}

private enum KVHeadMappingMode {
    case groupedContiguous
    case moduloInterleaved
}

private func decodeContextFromCaches(
    qOut: [Float],
    kCache: [Float],
    vCache: [Float],
    heads: Int,
    kvHeads: Int,
    headDim: Int,
    mapping: KVHeadMappingMode,
    visibleTokenCount: Int? = nil
) -> [Float] {
    precondition(qOut.count == heads * headDim)
    precondition(kCache.count == vCache.count)
    precondition(kCache.count % (kvHeads * headDim) == 0)
    precondition(heads % kvHeads == 0)

    let allocatedTokens = kCache.count / (kvHeads * headDim)
    let visibleTokens = visibleTokenCount ?? allocatedTokens
    precondition(visibleTokens > 0)
    precondition(visibleTokens <= allocatedTokens)
    let scale = 1.0 / sqrt(Float(headDim))
    let groupSize = heads / kvHeads
    var context = [Float](repeating: 0, count: heads * headDim)

    for head in 0..<heads {
        let kvHead: Int
        switch mapping {
        case .groupedContiguous:
            kvHead = head / groupSize
        case .moduloInterleaved:
            kvHead = head % kvHeads
        }

        let qBase = head * headDim
        let kvBase = kvHead * headDim
        var scores = [Float](repeating: 0, count: visibleTokens)
        for token in 0..<visibleTokens {
            var dot: Float = 0
            for dim in 0..<headDim {
                dot += qOut[qBase + dim] * kCache[(kvBase + dim) * allocatedTokens + token]
            }
            scores[token] = dot * scale
        }

        let maxScore = scores.max() ?? 0
        var denom: Float = 0
        for token in 0..<visibleTokens {
            scores[token] = exp(scores[token] - maxScore)
            denom += scores[token]
        }
        let invDenom: Float = denom > 0 ? 1 / denom : 0

        for dim in 0..<headDim {
            var accum: Float = 0
            for token in 0..<visibleTokens {
                let weight = scores[token] * invDenom
                accum += weight * vCache[(kvBase + dim) * allocatedTokens + token]
            }
            context[qBase + dim] = accum
        }
    }

    return context
}

private func cpuArtifactLlamaLayerHiddenLineage(
    weightDir: URL,
    config: MultiModelConfig,
    tokens: [Int],
    throughLayerIndex: Int? = nil
) throws -> [[Float]] {
    precondition(!tokens.isEmpty)
    let maxLayerIndex = throughLayerIndex ?? (config.nLayer - 1)
    precondition(maxLayerIndex >= 0 && maxLayerIndex < config.nLayer)

    let tokenEmbedding = try readBlobFloat16File(
        at: weightDir.appendingPathComponent("embeddings/token.bin"),
        expectedCount: config.vocab * config.dModel
    )

    struct CPUArtifactLayerWeights {
        let rmsAtt: [Float]
        let wq: [Float]
        let wk: [Float]
        let wv: [Float]
        let wo: [Float]
        let rmsFfn: [Float]
        let w1: [Float]
        let w2: [Float]
        let w3: [Float]
        let qNorm: [Float]?
        let kNorm: [Float]?
    }

    let layers: [CPUArtifactLayerWeights] = try (0...maxLayerIndex).map { layerIndex in
        let layerURL = weightDir.appendingPathComponent("layers/\(layerIndex)", isDirectory: true)
        let qNormURL = layerURL.appendingPathComponent("q_norm.bin")
        let kNormURL = layerURL.appendingPathComponent("k_norm.bin")
        let qNorm = FileManager.default.fileExists(atPath: qNormURL.path)
            ? try readBlobFloat16File(at: qNormURL, expectedCount: config.headDim)
            : nil
        let kNorm = FileManager.default.fileExists(atPath: kNormURL.path)
            ? try readBlobFloat16File(at: kNormURL, expectedCount: config.headDim)
            : nil
        return CPUArtifactLayerWeights(
            rmsAtt: try readBlobFloat16File(at: layerURL.appendingPathComponent("rms_att.bin"), expectedCount: config.dModel),
            wq: try readBlobFloat16File(at: layerURL.appendingPathComponent("wq.bin"), expectedCount: config.attentionDim * config.dModel),
            wk: try readBlobFloat16File(at: layerURL.appendingPathComponent("wk.bin"), expectedCount: config.kvDim * config.dModel),
            wv: try readBlobFloat16File(at: layerURL.appendingPathComponent("wv.bin"), expectedCount: config.kvDim * config.dModel),
            wo: try readBlobFloat16File(at: layerURL.appendingPathComponent("wo.bin"), expectedCount: config.dModel * config.attentionDim),
            rmsFfn: try readBlobFloat16File(at: layerURL.appendingPathComponent("rms_ffn.bin"), expectedCount: config.dModel),
            w1: try readBlobFloat16File(at: layerURL.appendingPathComponent("w1.bin"), expectedCount: config.hiddenDim * config.dModel),
            w2: try readBlobFloat16File(at: layerURL.appendingPathComponent("w2.bin"), expectedCount: config.dModel * config.hiddenDim),
            w3: try readBlobFloat16File(at: layerURL.appendingPathComponent("w3.bin"), expectedCount: config.hiddenDim * config.dModel),
            qNorm: qNorm,
            kNorm: kNorm
        )
    }

    var kCaches = Array(
        repeating: [Float](repeating: 0, count: config.kvDim * tokens.count),
        count: layers.count
    )
    var vCaches = Array(
        repeating: [Float](repeating: 0, count: config.kvDim * tokens.count),
        count: layers.count
    )
    var finalLayerStates = Array(
        repeating: [Float](repeating: 0, count: config.dModel),
        count: layers.count
    )

    for (position, token) in tokens.enumerated() {
        var hidden = Array(tokenEmbedding[token * config.dModel..<(token + 1) * config.dModel])
        for layerIndex in layers.indices {
            let layer = layers[layerIndex]
            let attnNormed = rmsNorm(hidden, weight: layer.rmsAtt, eps: Float(config.normEps))
            var q = multiplyRowMajorFlatMatrix(
                matrix: layer.wq,
                rows: config.attentionDim,
                cols: config.dModel,
                vector: attnNormed
            )
            let k = multiplyRowMajorFlatMatrix(
                matrix: layer.wk,
                rows: config.kvDim,
                cols: config.dModel,
                vector: attnNormed
            )
            let v = multiplyRowMajorFlatMatrix(
                matrix: layer.wv,
                rows: config.kvDim,
                cols: config.dModel,
                vector: attnNormed
            )
            q = roundFloat16Vector(q)
            var kRounded = roundFloat16Vector(k)
            let vRounded = roundFloat16Vector(v)
            if let qNorm = layer.qNorm {
                q = perHeadRMSNorm(q, weights: qNorm, heads: config.nHead, headDim: config.headDim, eps: Float(config.normEps))
            }
            if let kNorm = layer.kNorm {
                kRounded = perHeadRMSNorm(kRounded, weights: kNorm, heads: config.nKVHead, headDim: config.headDim, eps: Float(config.normEps))
            }
            q = applyHalfSplitRoPEPerHead(
                q,
                heads: config.nHead,
                headDim: config.headDim,
                position: position,
                theta: config.ropeTheta
            )
            kRounded = applyHalfSplitRoPEPerHead(
                kRounded,
                heads: config.nKVHead,
                headDim: config.headDim,
                position: position,
                theta: config.ropeTheta
            )
            q = roundFloat16Vector(q)
            kRounded = roundFloat16Vector(kRounded)

            for channel in 0..<config.kvDim {
                kCaches[layerIndex][channel * tokens.count + position] = kRounded[channel]
                vCaches[layerIndex][channel * tokens.count + position] = vRounded[channel]
            }

            let context = decodeContextFromCaches(
                qOut: q,
                kCache: kCaches[layerIndex],
                vCache: vCaches[layerIndex],
                heads: config.nHead,
                kvHeads: config.nKVHead,
                headDim: config.headDim,
                mapping: .groupedContiguous,
                visibleTokenCount: position + 1
            )

            hidden = zip(
                hidden,
                multiplyRowMajorFlatMatrix(
                    matrix: layer.wo,
                    rows: config.dModel,
                    cols: config.attentionDim,
                    vector: context
                )
            ).map(+)
            hidden = roundFloat16Vector(hidden)

            let ffnNormed = rmsNorm(hidden, weight: layer.rmsFfn, eps: Float(config.normEps))
            let gate = multiplyRowMajorFlatMatrix(
                matrix: layer.w1,
                rows: config.hiddenDim,
                cols: config.dModel,
                vector: ffnNormed
            )
            let up = multiplyRowMajorFlatMatrix(
                matrix: layer.w3,
                rows: config.hiddenDim,
                cols: config.dModel,
                vector: ffnNormed
            )
            let down = multiplyRowMajorFlatMatrix(
                matrix: layer.w2,
                rows: config.dModel,
                cols: config.hiddenDim,
                vector: zip(gate, up).map { silu($0) * $1 }
            )
            hidden = zip(hidden, down).map(+)
            hidden = roundFloat16Vector(hidden)
            finalLayerStates[layerIndex] = hidden
        }
    }

    return finalLayerStates
}

private struct CPUSingleLlamaLayerStepOutputs {
    let context: [Float]
    let projected: [Float]
    let ffnNormed: [Float]
    let gate: [Float]
    let up: [Float]
    let activated: [Float]
    let down: [Float]
    let hidden: [Float]
}

private struct CPUSingleLlamaLayerAttentionState {
    let qOut: [Float]
    let kCache: [Float]
    let vCache: [Float]
}

private func cpuArtifactLlamaNextToken(
    weightDir: URL,
    config: MultiModelConfig,
    tokens: [Int]
) throws -> Int {
    let finalStates = try cpuArtifactLlamaLayerHiddenLineage(
        weightDir: weightDir,
        config: config,
        tokens: tokens
    )
    guard let finalHidden = finalStates.last else {
        throw NSError(domain: "RealModelInferenceTests", code: 1, userInfo: [
            NSLocalizedDescriptionKey: "CPU artifact lineage did not produce a final hidden state"
        ])
    }

    return try cpuArtifactLlamaNextTokenFromFinalHidden(
        finalHidden,
        weightDir: weightDir,
        config: config
    )
}

private func cpuArtifactLlamaNextTokenFromFinalHidden(
    _ finalHidden: [Float],
    weightDir: URL,
    config: MultiModelConfig
) throws -> Int {
    let rmsFinal = try readBlobFloat16File(
        at: weightDir.appendingPathComponent("rms_final.bin"),
        expectedCount: config.dModel
    )
    let lmHead = try readBlobFloat16File(
        at: weightDir.appendingPathComponent("lm_head.bin"),
        expectedCount: config.vocab * config.dModel
    )
    let normalized = rmsNorm(finalHidden, weight: rmsFinal, eps: Float(config.normEps))
    let logits = multiplyRowMajorFlatMatrix(
        matrix: lmHead,
        rows: config.vocab,
        cols: config.dModel,
        vector: normalized
    )
    guard let nextToken = logits.enumerated().max(by: { $0.element < $1.element })?.offset else {
        throw NSError(domain: "RealModelInferenceTests", code: 2, userInfo: [
            NSLocalizedDescriptionKey: "CPU artifact logits were empty"
        ])
    }
    return nextToken
}

private func rawGGUFLlamaNextTokenFromFinalHidden(
    _ finalHidden: [Float],
    ggufURL: URL,
    config: MultiModelConfig
) async throws -> Int {
    let topWeights = try await loadRawGGUFLlamaTopWeights(ggufURL: ggufURL, config: config)
    let normalized = rmsNorm(finalHidden, weight: topWeights.rmsFinal, eps: Float(config.normEps))
    let logits = multiplyRowMajorFlatMatrix(
        matrix: topWeights.lmHead,
        rows: config.vocab,
        cols: config.dModel,
        vector: normalized
    )
    guard let nextToken = logits.enumerated().max(by: { $0.element < $1.element })?.offset else {
        throw NSError(domain: "RealModelInferenceTests", code: 3, userInfo: [
            NSLocalizedDescriptionKey: "Raw GGUF logits were empty"
        ])
    }
    return nextToken
}

private struct RawGGUFLlamaTopWeights {
    let rmsFinal: [Float]
    let lmHead: [Float]
}

private func loadRawGGUFLlamaTopWeights(
    ggufURL: URL,
    config: MultiModelConfig
) async throws -> RawGGUFLlamaTopWeights {
    guard let device = MTLCreateSystemDefaultDevice() else {
        throw NSError(domain: "RealModelInferenceTests", code: 4, userInfo: [
            NSLocalizedDescriptionKey: "Metal device unavailable for GGUF dequantization"
        ])
    }

    let loader = try GGUFLoader(url: ggufURL)
    let weightMap = try await loader.load(from: ggufURL)

    guard let rmsTensor = weightMap["output_norm.weight"] else {
        throw NSError(domain: "RealModelInferenceTests", code: 5, userInfo: [
            NSLocalizedDescriptionKey: "Missing output_norm.weight in raw GGUF"
        ])
    }
    guard let lmHeadTensorName = QwenGGUFVerificationSupport.rawGGUFLMHeadTensorName(from: weightMap),
          let lmHeadTensor = weightMap[lmHeadTensorName] else {
        throw NSError(domain: "RealModelInferenceTests", code: 6, userInfo: [
            NSLocalizedDescriptionKey: "Missing output.weight/token_embd.weight in raw GGUF"
        ])
    }

    let rmsFinal = try await DequantDispatcher.dequantize(tensor: rmsTensor, device: device)
    let lmHead = try await DequantDispatcher.dequantize(tensor: lmHeadTensor, device: device)

    #expect(rmsFinal.count == config.dModel)
    #expect(lmHead.count == config.vocab * config.dModel)

    return RawGGUFLlamaTopWeights(rmsFinal: rmsFinal, lmHead: lmHead)
}

private func makeDeterministicLayerInputs(
    config: MultiModelConfig,
    tokenCount: Int
) -> [[Float]] {
    (0..<tokenCount).map { tokenIndex in
        (0..<config.dModel).map { channel in
            let base = Float(((channel + tokenIndex * 7) % 23) - 11) / 7.0
            let wave = sin(Float(channel + tokenIndex * 13) * 0.07) * 0.2
            return base + wave
        }
    }
}

private func debugPromptTokens(
    env: [String: String],
    defaultTokens: [Int]
) -> [Int] {
    guard let raw = env["ESPRESSO_DEBUG_PROMPT_TOKENS"], !raw.isEmpty else {
        return defaultTokens
    }
    let parsed = raw
        .split(separator: ",")
        .compactMap { Int($0.trimmingCharacters(in: .whitespacesAndNewlines)) }
    return parsed.isEmpty ? defaultTokens : parsed
}

private func cpuSingleLlamaLayerAttentionState(
    weightDir: URL,
    config: MultiModelConfig,
    layerIndex: Int,
    inputs: [[Float]]
) throws -> CPUSingleLlamaLayerAttentionState {
    precondition(!inputs.isEmpty)

    let layerURL = weightDir.appendingPathComponent("layers/\(layerIndex)", isDirectory: true)
    let qNorm = try readBlobFloat16File(
        at: layerURL.appendingPathComponent("q_norm.bin"),
        expectedCount: config.headDim
    )
    let kNorm = try readBlobFloat16File(
        at: layerURL.appendingPathComponent("k_norm.bin"),
        expectedCount: config.headDim
    )
    let rmsAtt = try readBlobFloat16File(
        at: layerURL.appendingPathComponent("rms_att.bin"),
        expectedCount: config.dModel
    )
    let wq = try readBlobFloat16File(
        at: layerURL.appendingPathComponent("wq.bin"),
        expectedCount: config.attentionDim * config.dModel
    )
    let wk = try readBlobFloat16File(
        at: layerURL.appendingPathComponent("wk.bin"),
        expectedCount: config.kvDim * config.dModel
    )
    let wv = try readBlobFloat16File(
        at: layerURL.appendingPathComponent("wv.bin"),
        expectedCount: config.kvDim * config.dModel
    )

    var qOut = [Float](repeating: 0, count: config.attentionDim)
    var kCache = [Float](repeating: 0, count: config.kvDim * inputs.count)
    var vCache = [Float](repeating: 0, count: config.kvDim * inputs.count)

    for (position, input) in inputs.enumerated() {
        let hidden = roundFloat16Vector(input)
        let attnNormed = rmsNorm(hidden, weight: rmsAtt, eps: Float(config.normEps))
        qOut = multiplyRowMajorFlatMatrix(
            matrix: wq,
            rows: config.attentionDim,
            cols: config.dModel,
            vector: attnNormed
        )
        var k = multiplyRowMajorFlatMatrix(
            matrix: wk,
            rows: config.kvDim,
            cols: config.dModel,
            vector: attnNormed
        )
        var v = multiplyRowMajorFlatMatrix(
            matrix: wv,
            rows: config.kvDim,
            cols: config.dModel,
            vector: attnNormed
        )

        qOut = roundFloat16Vector(qOut)
        k = roundFloat16Vector(k)
        v = roundFloat16Vector(v)

        qOut = perHeadRMSNorm(qOut, weights: qNorm, heads: config.nHead, headDim: config.headDim, eps: Float(config.normEps))
        k = perHeadRMSNorm(k, weights: kNorm, heads: config.nKVHead, headDim: config.headDim, eps: Float(config.normEps))

        qOut = applyHalfSplitRoPEPerHead(
            qOut,
            heads: config.nHead,
            headDim: config.headDim,
            position: position,
            theta: config.ropeTheta
        )
        k = applyHalfSplitRoPEPerHead(
            k,
            heads: config.nKVHead,
            headDim: config.headDim,
            position: position,
            theta: config.ropeTheta
        )
        qOut = roundFloat16Vector(qOut)
        k = roundFloat16Vector(k)

        for channel in 0..<config.kvDim {
            kCache[channel * inputs.count + position] = k[channel]
            vCache[channel * inputs.count + position] = v[channel]
        }
    }

    return CPUSingleLlamaLayerAttentionState(
        qOut: qOut,
        kCache: kCache,
        vCache: vCache
    )
}

private func cpuSingleLlamaLayerStepOutputs(
    weightDir: URL,
    config: MultiModelConfig,
    layerIndex: Int,
    inputs: [[Float]],
    roundFFNIntermediatesToFP16: Bool = false
) throws -> [CPUSingleLlamaLayerStepOutputs] {
    precondition(!inputs.isEmpty)

    let layerURL = weightDir.appendingPathComponent("layers/\(layerIndex)", isDirectory: true)
    let qNormURL = layerURL.appendingPathComponent("q_norm.bin")
    let kNormURL = layerURL.appendingPathComponent("k_norm.bin")
    let qNorm = FileManager.default.fileExists(atPath: qNormURL.path)
        ? try readBlobFloat16File(at: qNormURL, expectedCount: config.headDim)
        : nil
    let kNorm = FileManager.default.fileExists(atPath: kNormURL.path)
        ? try readBlobFloat16File(at: kNormURL, expectedCount: config.headDim)
        : nil

    let rmsAtt = try readBlobFloat16File(at: layerURL.appendingPathComponent("rms_att.bin"), expectedCount: config.dModel)
    let wq = try readBlobFloat16File(at: layerURL.appendingPathComponent("wq.bin"), expectedCount: config.attentionDim * config.dModel)
    let wk = try readBlobFloat16File(at: layerURL.appendingPathComponent("wk.bin"), expectedCount: config.kvDim * config.dModel)
    let wv = try readBlobFloat16File(at: layerURL.appendingPathComponent("wv.bin"), expectedCount: config.kvDim * config.dModel)
    let wo = try readBlobFloat16File(at: layerURL.appendingPathComponent("wo.bin"), expectedCount: config.dModel * config.attentionDim)
    let rmsFfn = try readBlobFloat16File(at: layerURL.appendingPathComponent("rms_ffn.bin"), expectedCount: config.dModel)
    let w1 = try readBlobFloat16File(at: layerURL.appendingPathComponent("w1.bin"), expectedCount: config.hiddenDim * config.dModel)
    let w2 = try readBlobFloat16File(at: layerURL.appendingPathComponent("w2.bin"), expectedCount: config.dModel * config.hiddenDim)
    let w3 = try readBlobFloat16File(at: layerURL.appendingPathComponent("w3.bin"), expectedCount: config.hiddenDim * config.dModel)

    var kCache = [Float](repeating: 0, count: config.kvDim * inputs.count)
    var vCache = [Float](repeating: 0, count: config.kvDim * inputs.count)
    var outputs: [CPUSingleLlamaLayerStepOutputs] = []
    outputs.reserveCapacity(inputs.count)

    for (position, rawInput) in inputs.enumerated() {
        var hidden = roundFloat16Vector(rawInput)
        let attnNormed = rmsNorm(hidden, weight: rmsAtt, eps: Float(config.normEps))
        var q = multiplyRowMajorFlatMatrix(
            matrix: wq,
            rows: config.attentionDim,
            cols: config.dModel,
            vector: attnNormed
        )
        var k = multiplyRowMajorFlatMatrix(
            matrix: wk,
            rows: config.kvDim,
            cols: config.dModel,
            vector: attnNormed
        )
        let v = multiplyRowMajorFlatMatrix(
            matrix: wv,
            rows: config.kvDim,
            cols: config.dModel,
            vector: attnNormed
        )
        q = roundFloat16Vector(q)
        k = roundFloat16Vector(k)
        let vRounded = roundFloat16Vector(v)

        if let qNorm {
            q = perHeadRMSNorm(q, weights: qNorm, heads: config.nHead, headDim: config.headDim, eps: Float(config.normEps))
        }
        if let kNorm {
            k = perHeadRMSNorm(k, weights: kNorm, heads: config.nKVHead, headDim: config.headDim, eps: Float(config.normEps))
        }

        q = applyHalfSplitRoPEPerHead(
            q,
            heads: config.nHead,
            headDim: config.headDim,
            position: position,
            theta: config.ropeTheta
        )
        k = applyHalfSplitRoPEPerHead(
            k,
            heads: config.nKVHead,
            headDim: config.headDim,
            position: position,
            theta: config.ropeTheta
        )
        q = roundFloat16Vector(q)
        k = roundFloat16Vector(k)

        for channel in 0..<config.kvDim {
            kCache[channel * inputs.count + position] = k[channel]
            vCache[channel * inputs.count + position] = vRounded[channel]
        }

        let context = decodeContextFromCaches(
            qOut: q,
            kCache: kCache,
            vCache: vCache,
            heads: config.nHead,
            kvHeads: config.nKVHead,
            headDim: config.headDim,
            mapping: .groupedContiguous,
            visibleTokenCount: position + 1
        )

        let projected = roundFloat16Vector(
            zip(
                hidden,
                multiplyRowMajorFlatMatrix(
                    matrix: wo,
                    rows: config.dModel,
                    cols: config.attentionDim,
                    vector: context
                )
            ).map(+)
        )

        let ffnNormedRaw = rmsNorm(projected, weight: rmsFfn, eps: Float(config.normEps))
        let ffnNormed = roundFFNIntermediatesToFP16 ? roundFloat16Vector(ffnNormedRaw) : ffnNormedRaw
        let gateRaw = multiplyRowMajorFlatMatrix(
            matrix: w1,
            rows: config.hiddenDim,
            cols: config.dModel,
            vector: ffnNormed
        )
        let gate = roundFFNIntermediatesToFP16 ? roundFloat16Vector(gateRaw) : gateRaw
        let upRaw = multiplyRowMajorFlatMatrix(
            matrix: w3,
            rows: config.hiddenDim,
            cols: config.dModel,
            vector: ffnNormed
        )
        let up = roundFFNIntermediatesToFP16 ? roundFloat16Vector(upRaw) : upRaw
        let activatedRaw = zip(gate, up).map { silu($0) * $1 }
        let activated = roundFFNIntermediatesToFP16 ? roundFloat16Vector(activatedRaw) : activatedRaw
        let downRaw = multiplyRowMajorFlatMatrix(
            matrix: w2,
            rows: config.dModel,
            cols: config.hiddenDim,
            vector: activated
        )
        let down = roundFFNIntermediatesToFP16 ? roundFloat16Vector(downRaw) : downRaw
        hidden = roundFloat16Vector(zip(projected, down).map(+))

        outputs.append(
            CPUSingleLlamaLayerStepOutputs(
                context: context,
                projected: projected,
                ffnNormed: ffnNormed,
                gate: gate,
                up: up,
                activated: activated,
                down: down,
                hidden: hidden
            )
        )
    }

    return outputs
}

private func roundFloat16Vector(_ values: [Float]) -> [Float] {
    values.map { Float(Float16($0)) }
}

private func rmsNorm(_ input: [Float], weight: [Float], eps: Float) -> [Float] {
    precondition(input.count == weight.count)
    let meanSquare = input.reduce(into: Float(0)) { partial, value in
        partial += value * value
    } / Float(input.count)
    let scale = 1.0 / sqrt(meanSquare + eps)
    return zip(input, weight).map { $0 * scale * $1 }
}

private func perHeadRMSNorm(
    _ input: [Float],
    weights: [Float],
    heads: Int,
    headDim: Int,
    eps: Float
) -> [Float] {
    precondition(input.count == heads * headDim)
    precondition(weights.count == headDim)
    var result = input
    for head in 0..<heads {
        let base = head * headDim
        let slice = Array(result[base..<base + headDim])
        let normalized = rmsNorm(slice, weight: weights, eps: eps)
        result[base..<base + headDim] = normalized[0..<headDim]
    }
    return result
}

private func applyHalfSplitRoPEPerHead(
    _ input: [Float],
    heads: Int,
    headDim: Int,
    position: Int,
    theta: Float
) -> [Float] {
    precondition(input.count == heads * headDim)
    precondition(headDim % 2 == 0)
    let halfDim = headDim / 2
    var result = input
    for head in 0..<heads {
        let base = head * headDim
        for dimPair in 0..<halfDim {
            let frequency = 1.0 / pow(theta, Float(2 * dimPair) / Float(headDim))
            let angle = Float(position) * frequency
            let cosv = cos(angle)
            let sinv = sin(angle)
            let i0 = base + dimPair
            let i1 = base + dimPair + halfDim
            let v0 = result[i0]
            let v1 = result[i1]
            result[i0] = v0 * cosv - v1 * sinv
            result[i1] = v0 * sinv + v1 * cosv
        }
    }
    return result
}

private func multiplyRowMajorFlatMatrix(matrix: [Float], rows: Int, cols: Int, vector: [Float]) -> [Float] {
    precondition(matrix.count == rows * cols)
    precondition(vector.count == cols)
    return (0..<rows).map { row in
        let base = row * cols
        return zip(matrix[base..<base + cols], vector).reduce(into: Float(0)) { partial, pair in
            partial += pair.0 * pair.1
        }
    }
}

private func silu(_ value: Float) -> Float {
    0.5 * value * (1 + tanh(0.5 * value))
}

private func writeCausalMask(spatial: Int, to url: URL) throws {
    var values = [Float](repeating: 0.0, count: spatial * spatial)
    for row in 0..<spatial {
        for column in 0..<spatial where column > row {
            values[row * spatial + column] = -10_000
        }
    }
    let data = makeBlobData(from: values)
    try data.write(to: url)
}

private func makeBlobData(repeating value: Float, count: Int) -> Data {
    makeBlobData(from: [Float](repeating: value, count: count))
}

private func makeBlobData(from values: [Float]) -> Data {
    var data = Data(count: 128 + values.count * MemoryLayout<UInt16>.size)
    data.withUnsafeMutableBytes { raw in
        guard let base = raw.baseAddress else { return }
        let bytes = base.assumingMemoryBound(to: UInt8.self)
        bytes[0] = 0x01
        bytes[4] = 0x02
        bytes[64] = 0xEF
        bytes[65] = 0xBE
        bytes[66] = 0xAD
        bytes[67] = 0xDE
        bytes[68] = 0x01

        raw.storeBytes(of: UInt32(values.count * 2).littleEndian, toByteOffset: 72, as: UInt32.self)
        raw.storeBytes(of: UInt32(128).littleEndian, toByteOffset: 80, as: UInt32.self)

        for (index, value) in values.enumerated() {
            raw.storeBytes(
                of: Float16(value).bitPattern.littleEndian,
                toByteOffset: 128 + (index * MemoryLayout<UInt16>.stride),
                as: UInt16.self
            )
        }
    }
    return data
}

private func makeGPT2TokenizerDirectory() throws -> URL {
    let directory = try makeTempDirectory()
    let newlinePiece = String(gpt2ByteUnicodeMap()[10]!)
    let spacePiece = String(gpt2ByteUnicodeMap()[32]!)
    let vocab: [String: Int] = [
        "Hello": 0,
        ",": 1,
        "\(spacePiece)world": 2,
        "!": 3,
        newlinePiece: 4,
    ]
    let vocabData = try JSONSerialization.data(withJSONObject: vocab, options: [.sortedKeys])
    try vocabData.write(to: directory.appendingPathComponent("vocab.json"))

    let merges = [
        "#version: 0.2",
        "H e",
        "He l",
        "Hel l",
        "Hell o",
        "\(spacePiece) w",
        "\(spacePiece)w o",
        "\(spacePiece)wo r",
        "\(spacePiece)wor l",
        "\(spacePiece)worl d",
    ].joined(separator: "\n")
    try merges.write(to: directory.appendingPathComponent("merges.txt"), atomically: true, encoding: .utf8)
    return directory
}

private func makeSentencePieceTokenizerDirectory() throws -> URL {
    let directory = try makeTempDirectory()
    let modelURL = directory.appendingPathComponent("tokenizer.model")
    let pieces: [(String, Float)] = [
        ("▁", 0.0),
        ("H", 0.0),
        ("e", 0.0),
        ("l", 0.0),
        ("o", 0.0),
        (",", 0.0),
        ("w", 0.0),
        ("r", 0.0),
        ("d", 0.0),
        ("!", 0.0),
        ("▁H", 1.0),
        ("▁He", 2.0),
        ("▁Hel", 3.0),
        ("▁Hell", 4.0),
        ("▁Hello", 5.0),
        ("▁w", 1.0),
        ("▁wo", 2.0),
        ("▁wor", 3.0),
        ("▁worl", 4.0),
        ("▁world", 5.0),
        ("<0x0A>", 0.0),
    ]

    var data = Data()
    append(Int32(16), to: &data)
    for (piece, score) in pieces {
        append(score, to: &data)
        let bytes = Array(piece.utf8)
        append(Int32(bytes.count), to: &data)
        data.append(contentsOf: bytes)
    }
    try data.write(to: modelURL)
    return modelURL
}

private func append(_ value: Int32, to data: inout Data) {
    var littleEndian = value.littleEndian
    withUnsafeBytes(of: &littleEndian) { data.append(contentsOf: $0) }
}

private func append(_ value: Float, to data: inout Data) {
    var bits = value.bitPattern.littleEndian
    withUnsafeBytes(of: &bits) { data.append(contentsOf: $0) }
}

private func makeTempDirectory() throws -> URL {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    return directory
}

private func gpt2ByteUnicodeMap() -> [UInt8: UnicodeScalar] {
    var printable = Array(33...126) + Array(161...172) + Array(174...255)
    var mapped = printable
    let printableSet = Set(printable)
    var extra = 0
    for value in 0..<256 where !printableSet.contains(value) {
        printable.append(value)
        mapped.append(256 + extra)
        extra += 1
    }
    var result: [UInt8: UnicodeScalar] = [:]
    for (index, byte) in printable.enumerated() {
        result[UInt8(byte)] = UnicodeScalar(mapped[index])!
    }
    return result
}

private func extractSpatialSlice(
    from values: [Float],
    channels: Int,
    spatial: Int,
    spatialIndex: Int
) -> [Float] {
    precondition(values.count == channels * spatial)
    precondition(spatialIndex >= 0 && spatialIndex < spatial)
    var output = [Float](repeating: 0, count: channels)
    for channel in 0..<channels {
        output[channel] = values[channel * spatial + spatialIndex]
    }
    return output
}

private func maxAbsoluteDifference(_ lhs: [Float], _ rhs: [Float]) -> Float {
    precondition(lhs.count == rhs.count)
    var maxValue: Float = 0
    for index in lhs.indices {
        maxValue = max(maxValue, abs(lhs[index] - rhs[index]))
    }
    return maxValue
}

private func naiveDecodeContextFromCaches(
    qOut: [Float],
    kCache: [Float],
    vCache: [Float],
    heads: Int,
    kvHeads: Int,
    headDim: Int,
    visibleTokenCount: Int,
    cacheStride: Int
) -> [Float] {
    let queriesPerKVHead = max(heads / max(kvHeads, 1), 1)
    let scale: Float = 1.0 / sqrt(Float(headDim))
    var context = [Float](repeating: 0, count: heads * headDim)

    for head in 0..<heads {
        let kvHead = min(head / queriesPerKVHead, kvHeads - 1)
        let qBase = head * headDim
        let kvBase = kvHead * headDim
        var scores = [Float](repeating: 0, count: visibleTokenCount)
        for token in 0..<visibleTokenCount {
            var dot: Float = 0
            for dim in 0..<headDim {
                dot += qOut[qBase + dim] * kCache[(kvBase + dim) * cacheStride + token]
            }
            scores[token] = dot * scale
        }

        let maxScore = scores.max() ?? 0
        var denom: Float = 0
        for token in 0..<visibleTokenCount {
            scores[token] = exp(scores[token] - maxScore)
            denom += scores[token]
        }
        let invDenom: Float = denom > 0 ? 1 / denom : 0

        for dim in 0..<headDim {
            var accum: Float = 0
            for token in 0..<visibleTokenCount {
                accum += scores[token] * invDenom * vCache[(kvBase + dim) * cacheStride + token]
            }
            context[qBase + dim] = accum
        }
    }

    return context
}

private func shouldRunANEHardwareTests() -> Bool {
    ProcessInfo.processInfo.environment["ANE_HARDWARE_TESTS"] == "1" && aneIsAvailable()
}

private func lfm2ProbeWeightDirectory() -> String? {
    let environment = ProcessInfo.processInfo.environment
    if let explicit = environment["ESPRESSO_LFM2_WEIGHT_DIR"], !explicit.isEmpty {
        return FileManager.default.fileExists(atPath: explicit) ? explicit : nil
    }
    let defaultPath = "/tmp/lfm2-350m.esp/weights"
    return FileManager.default.fileExists(atPath: defaultPath) ? defaultPath : nil
}

private func aneIsAvailable() -> Bool {
    let handle = dlopen(
        "/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine",
        RTLD_NOW
    )
    if handle == nil {
        return false
    }
    dlclose(handle)

    let requiredClasses = [
        "_ANEInMemoryModelDescriptor",
        "_ANEInMemoryModel",
        "_ANERequest",
        "_ANEIOSurfaceObject",
    ]
    return requiredClasses.allSatisfy { NSClassFromString($0) != nil }
}
