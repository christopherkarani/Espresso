import Foundation
import Testing
import ANETypes
import ModelSupport
@testable import RealModelInference

@Test func test_llamaTopLevelWeightPathsStruct() throws {
    let paths = RealModelInferenceEngine.LlamaTopLevelWeightPaths(
        tokenEmbedding: "/weights/embeddings/token.bin",
        finalNormGamma: "/weights/rms_final.bin",
        lmHead: "/weights/lm_head.bin"
    )
    #expect(paths.tokenEmbedding == "/weights/embeddings/token.bin")
    #expect(paths.finalNormGamma == "/weights/rms_final.bin")
    #expect(paths.lmHead == "/weights/lm_head.bin")
}

@Test func test_resolveTopLevelWeightPathsLlama() throws {
    let root = FileManager.default.temporaryDirectory
        .appendingPathComponent("llama-weight-test-\(UUID().uuidString)")
    let embeddingsDir = root.appendingPathComponent("embeddings", isDirectory: true)
    try FileManager.default.createDirectory(at: embeddingsDir, withIntermediateDirectories: true)

    let config = MultiModelConfig(
        name: "test-llama",
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

    // Write minimal weight files
    let blobSize = 64 * 8 * MemoryLayout<Float>.size
    let smallBlobSize = 8 * MemoryLayout<Float>.size
    let blob = Data(repeating: 0, count: blobSize)
    let smallBlob = Data(repeating: 0, count: smallBlobSize)

    try blob.write(to: embeddingsDir.appendingPathComponent("token.bin"))
    try smallBlob.write(to: root.appendingPathComponent("rms_final.bin"))
    try blob.write(to: root.appendingPathComponent("lm_head.bin"))

    let paths = try RealModelInferenceEngine.resolveLlamaTopLevelWeightPaths(
        config: config,
        weightDir: root.path
    )

    #expect(paths.tokenEmbedding == embeddingsDir.appendingPathComponent("token.bin").path)
    #expect(paths.finalNormGamma == root.appendingPathComponent("rms_final.bin").path)
    #expect(paths.lmHead == root.appendingPathComponent("lm_head.bin").path)

    try? FileManager.default.removeItem(at: root)
}

@Test func test_resolveTopLevelWeightPathsLlamaAcceptsFinalNormBin() throws {
    let root = FileManager.default.temporaryDirectory
        .appendingPathComponent("llama-weight-final-norm-test-\(UUID().uuidString)")
    let embeddingsDir = root.appendingPathComponent("embeddings", isDirectory: true)
    try FileManager.default.createDirectory(at: embeddingsDir, withIntermediateDirectories: true)

    let config = MultiModelConfig(
        name: "test-llama",
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

    let blobSize = 64 * 8 * MemoryLayout<Float>.size
    let smallBlobSize = 8 * MemoryLayout<Float>.size
    let blob = Data(repeating: 0, count: blobSize)
    let smallBlob = Data(repeating: 0, count: smallBlobSize)

    try blob.write(to: embeddingsDir.appendingPathComponent("token.bin"))
    try smallBlob.write(to: root.appendingPathComponent("final_norm.bin"))
    try blob.write(to: root.appendingPathComponent("lm_head.bin"))

    let paths = try RealModelInferenceEngine.resolveLlamaTopLevelWeightPaths(
        config: config,
        weightDir: root.path
    )

    #expect(paths.finalNormGamma == root.appendingPathComponent("final_norm.bin").path)

    try? FileManager.default.removeItem(at: root)
}

@Test func test_ropeTheta_defaultsTo10000() {
    let config = MultiModelConfig(
        name: "test",
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
    #expect(config.ropeTheta == 10_000.0)
}

@Test func test_ropeTheta_customValue() {
    let config = MultiModelConfig(
        name: "test",
        nLayer: 1,
        nHead: 2,
        nKVHead: 2,
        dModel: 8,
        headDim: 4,
        hiddenDim: 16,
        vocab: 64,
        maxSeq: 8,
        normEps: 1e-5,
        ropeTheta: 500_000.0,
        architecture: .llama
    )
    #expect(config.ropeTheta == 500_000.0)
}

@Test func test_llamaRegistryEntriesHaveCorrectRopeTheta() {
    #expect(ModelRegistry.llama3_2_1b.ropeTheta == 500_000.0)
    #expect(ModelRegistry.llama3_2_1b_ctx512.ropeTheta == 500_000.0)
    #expect(ModelRegistry.llama3_2_1b_ctx512.maxSeq == 512)
    #expect(ModelRegistry.llama3_2_3b.ropeTheta == 500_000.0)
    #expect(ModelRegistry.stories110m.ropeTheta == 10_000.0)
    #expect(ModelRegistry.gpt2_124m.ropeTheta == 10_000.0)
}

@Test func test_llamaMetalRoPEFastPathRejectsKVCacheBindings() {
    #expect(
        RealModelInferenceEngine.supportsLlamaMetalRoPEFastPath(
            cachedBindingsAvailable: true,
            kBindingContainsKVCache: true
        ) == false
    )
    #expect(
        RealModelInferenceEngine.supportsLlamaMetalRoPEFastPath(
            cachedBindingsAvailable: true,
            kBindingContainsKVCache: false
        ) == true
    )
    #expect(
        RealModelInferenceEngine.supportsLlamaMetalRoPEFastPath(
            cachedBindingsAvailable: false,
            kBindingContainsKVCache: false
        ) == false
    )
}

@Test func test_hybridCachedBindingsCanBeDisabledByEnvironment() {
    let gpt2Config = ModelRegistry.gpt2_124m
    let storiesConfig = ModelRegistry.stories110m
    let storiesBundleConfig = MultiModelConfig(
        name: "llama2.c-stories110M",
        nLayer: 12,
        nHead: 12,
        nKVHead: 12,
        dModel: 768,
        headDim: 64,
        hiddenDim: 2_048,
        vocab: 32_000,
        maxSeq: 256,
        normEps: 1e-5,
        architecture: .llama
    )
    let otherLlamaConfig = ModelRegistry.tinyLlama_1_1b
    #expect(
        RealModelInferenceEngine.supportsHybridCachedBindings(
            config: gpt2Config,
            environment: [:]
        ) == true
    )
    #expect(
        RealModelInferenceEngine.supportsHybridCachedBindings(
            config: storiesConfig,
            environment: [:]
        ) == true
    )
    #expect(
        RealModelInferenceEngine.supportsHybridCachedBindings(
            config: storiesBundleConfig,
            environment: [:]
        ) == true
    )
    #expect(
        RealModelInferenceEngine.supportsHybridCachedBindings(
            config: otherLlamaConfig,
            environment: [:]
        ) == false
    )
    #expect(
        RealModelInferenceEngine.supportsHybridCachedBindings(
            config: otherLlamaConfig,
            environment: ["ESPRESSO_ENABLE_LLAMA_HYBRID_CACHED_BINDINGS": "1"]
        ) == true
    )
    #expect(
        RealModelInferenceEngine.supportsHybridCachedBindings(
            config: gpt2Config,
            environment: ["ESPRESSO_DISABLE_HYBRID_CACHED_BINDINGS": "1"]
        ) == false
    )
    #expect(
        RealModelInferenceEngine.supportsHybridCachedBindings(
            config: storiesConfig,
            environment: [
                "ESPRESSO_ENABLE_LLAMA_HYBRID_CACHED_BINDINGS": "1",
                "ESPRESSO_DISABLE_HYBRID_CACHED_BINDINGS": "1",
            ]
        ) == false
    )
}

@Test func test_hybridDonorDeltaDefaultsOnForStoriesAndAllowsDisableOverride() {
    let gpt2Config = ModelRegistry.gpt2_124m
    let storiesConfig = ModelRegistry.stories110m
    let otherLlamaConfig = ModelRegistry.tinyLlama_1_1b
    #expect(
        RealModelInferenceEngine.supportsHybridDonorDelta(
            config: gpt2Config,
            environment: [:]
        ) == true
    )
    #expect(
        RealModelInferenceEngine.supportsHybridDonorDelta(
            config: otherLlamaConfig,
            environment: [:]
        ) == true
    )
    #expect(
        RealModelInferenceEngine.supportsHybridDonorDelta(
            config: storiesConfig,
            environment: [:]
        ) == true
    )
    #expect(
        RealModelInferenceEngine.supportsHybridDonorDelta(
            config: storiesConfig,
            environment: ["ESPRESSO_DISABLE_HYBRID_DONOR_DELTA": "1"]
        ) == false
    )
    #expect(
        RealModelInferenceEngine.supportsHybridDonorDelta(
            config: gpt2Config,
            environment: ["ESPRESSO_DISABLE_HYBRID_DONOR_DELTA": "1"]
        ) == false
    )
}

@Test func test_llamaHybridFusedExactHeadDefaultsOnForStoriesAndAllowsDisableOverride() {
    let gpt2Config = ModelRegistry.gpt2_124m
    let storiesConfig = ModelRegistry.stories110m

    #expect(
        RealModelInferenceEngine.usesLlamaHybridFusedExactHead(
            config: gpt2Config,
            environment: ["ESPRESSO_ENABLE_LLAMA_HYBRID_FUSED_EXACT_HEAD": "1"]
        ) == false
    )
    #expect(
        RealModelInferenceEngine.usesLlamaHybridFusedExactHead(
            config: storiesConfig,
            environment: [:]
        ) == true
    )
    #expect(
        RealModelInferenceEngine.usesLlamaHybridFusedExactHead(
            config: storiesConfig,
            environment: ["ESPRESSO_DISABLE_LLAMA_HYBRID_FUSED_EXACT_HEAD": "1"]
        ) == false
    )
}

@Test func test_llamaHybridFusedExactHeadEnablesForStoriesWhenRequested() {
    #expect(
        RealModelInferenceEngine.usesLlamaHybridFusedExactHead(
            config: ModelRegistry.stories110m,
            environment: ["ESPRESSO_ENABLE_LLAMA_HYBRID_FUSED_EXACT_HEAD": "1"]
        ) == true
    )
}

@Test func test_forceExactHeadBackendOverrideParsesKnownValues() {
    #expect(
        RealModelInferenceEngine.forcedExactHeadBackend(
            environment: ["ESPRESSO_FORCE_EXACT_HEAD_BACKEND": "cpu_fp16_tiled"]
        ) == .cpuFP16Tiled
    )
    #expect(
        RealModelInferenceEngine.forcedExactHeadBackend(
            environment: ["ESPRESSO_FORCE_EXACT_HEAD_BACKEND": "partitioned"]
        ) == .cpuPartitionedFP32
    )
    #expect(
        RealModelInferenceEngine.forcedExactHeadBackend(
            environment: ["ESPRESSO_FORCE_EXACT_HEAD_BACKEND": "ane"]
        ) == .ane
    )
    #expect(
        RealModelInferenceEngine.forcedExactHeadBackend(
            environment: ["ESPRESSO_FORCE_EXACT_HEAD_BACKEND": "bogus"]
        ) == nil
    )
}

@Test func test_resolveLlamaWeightPathsMissingFile() throws {
    let root = FileManager.default.temporaryDirectory
        .appendingPathComponent("llama-missing-test-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)

    let config = MultiModelConfig(
        name: "test-llama",
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

    do {
        _ = try RealModelInferenceEngine.resolveLlamaTopLevelWeightPaths(
            config: config,
            weightDir: root.path
        )
        Issue.record("Expected error for missing token embedding")
    } catch {
        #expect(error.localizedDescription.localizedCaseInsensitiveContains("token embedding"))
    }

    try? FileManager.default.removeItem(at: root)
}

@Test func test_loadHybridLayerWeightsLlamaLoadsOptionalQKNorms() throws {
    let root = try makeHybridLlamaLayerDirectory(
        config: makeTinyLlamaConfig(),
        includeQKNorms: true
    )
    defer { try? FileManager.default.removeItem(at: root) }

    let config = makeTinyLlamaConfig()
    let paths = LayerWeightPaths.forLayer(0, config: config, blobDir: root.path)
    let weights = try RealModelInferenceEngine.loadHybridLayerWeightsLlama(config: config, paths: paths)

    let hasQNorm = weights.hasQNorm
    let hasKNorm = weights.hasKNorm
    let loadedQNorm = materialize(weights.qNorm)
    let loadedKNorm = materialize(weights.kNorm)
    #expect(hasQNorm)
    #expect(hasKNorm)
    #expect(loadedQNorm == [0.25, 0.5, 0.75, 1.0])
    #expect(loadedKNorm == [1.0, 0.75, 0.5, 0.25])
}

@Test func test_loadHybridLayerWeightsLlamaKeepsMissingQKNormsOptional() throws {
    let root = try makeHybridLlamaLayerDirectory(
        config: makeTinyLlamaConfig(),
        includeQKNorms: false
    )
    defer { try? FileManager.default.removeItem(at: root) }

    let config = makeTinyLlamaConfig()
    let paths = LayerWeightPaths.forLayer(0, config: config, blobDir: root.path)
    let weights = try RealModelInferenceEngine.loadHybridLayerWeightsLlama(config: config, paths: paths)

    let hasQNorm = weights.hasQNorm
    let hasKNorm = weights.hasKNorm
    #expect(!hasQNorm)
    #expect(!hasKNorm)
    #expect(materialize(weights.qNorm).isEmpty)
    #expect(materialize(weights.kNorm).isEmpty)
}

@Test func test_applyPerHeadRMSNormInPlaceMatchesReference() {
    var values: [Float] = [1, 2, 3, 4, 2, 0, -2, 0]
    let weights: [Float] = [1, 0.5, 2, 1.5]
    let expected = referencePerHeadRMSNorm(
        values: values,
        weights: weights,
        headCount: 2,
        headDim: 4,
        epsilon: 1e-5
    )

    values.withUnsafeMutableBufferPointer { valuesBuffer in
        weights.withUnsafeBufferPointer { weightsBuffer in
            RealModelInferenceEngine.applyPerHeadRMSNormInPlace(
                values: valuesBuffer,
                weights: weightsBuffer,
                headCount: 2,
                headDim: 4,
                epsilon: 1e-5
            )
        }
    }

    for (actual, expectedValue) in zip(values, expected) {
        #expect(abs(actual - expectedValue) < 1e-5)
    }
}

private func makeTinyLlamaConfig() -> MultiModelConfig {
    MultiModelConfig(
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
}

private func makeHybridLlamaLayerDirectory(
    config: MultiModelConfig,
    includeQKNorms: Bool
) throws -> URL {
    let root = FileManager.default.temporaryDirectory
        .appendingPathComponent("hybrid-llama-layer-\(UUID().uuidString)", isDirectory: true)
    let layerDir = root
        .appendingPathComponent("layers", isDirectory: true)
        .appendingPathComponent("0", isDirectory: true)
    try FileManager.default.createDirectory(at: layerDir, withIntermediateDirectories: true)

    try writeBlob(values: Array(repeating: 1, count: config.dModel), to: layerDir.appendingPathComponent("rms_att.bin"))
    try writeBlob(values: Array(0..<(config.dModel * config.attentionDim)).map(Float.init), to: layerDir.appendingPathComponent("wq.bin"))
    try writeBlob(values: Array(0..<(config.dModel * config.kvDim)).map(Float.init), to: layerDir.appendingPathComponent("wk.bin"))
    try writeBlob(values: Array(0..<(config.dModel * config.kvDim)).map { Float($0) * 0.5 }, to: layerDir.appendingPathComponent("wv.bin"))
    try writeBlob(values: Array(0..<(config.dModel * config.attentionDim)).map { Float($0) * 0.25 }, to: layerDir.appendingPathComponent("wo.bin"))
    try writeBlob(values: Array(repeating: 1, count: config.dModel), to: layerDir.appendingPathComponent("rms_ffn.bin"))
    try writeBlob(values: Array(0..<(config.hiddenDim * config.dModel)).map(Float.init), to: layerDir.appendingPathComponent("w1.bin"))
    try writeBlob(values: Array(0..<(config.dModel * config.hiddenDim)).map(Float.init), to: layerDir.appendingPathComponent("w2.bin"))
    try writeBlob(values: Array(0..<(config.hiddenDim * config.dModel)).map { Float($0) * 0.125 }, to: layerDir.appendingPathComponent("w3.bin"))

    if includeQKNorms {
        try writeBlob(values: [0.25, 0.5, 0.75, 1.0], to: layerDir.appendingPathComponent("q_norm.bin"))
        try writeBlob(values: [1.0, 0.75, 0.5, 0.25], to: layerDir.appendingPathComponent("k_norm.bin"))
    }

    return root
}

private func writeBlob(values: [Float], to url: URL) throws {
    try WeightBlob.build(from: values, rows: 1, cols: values.count).write(to: url)
}

private func materialize(_ buffer: borrowing TensorBuffer) -> [Float] {
    buffer.withUnsafeBufferPointer { Array($0) }
}

private func referencePerHeadRMSNorm(
    values: [Float],
    weights: [Float],
    headCount: Int,
    headDim: Int,
    epsilon: Float
) -> [Float] {
    var output = values
    for head in 0..<headCount {
        let base = head * headDim
        var sumSq: Float = 0
        for lane in 0..<headDim {
            let value = output[base + lane]
            sumSq += value * value
        }
        let invRms = 1.0 / sqrtf(sumSq / Float(headDim) + epsilon)
        for lane in 0..<headDim {
            output[base + lane] *= invRms * weights[lane]
        }
    }
    return output
}
