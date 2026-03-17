import Darwin
import Foundation
import Testing
import ANEGraphIR
import ModelSupport
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
    let tokens: [UInt32] = [15496, 11]
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

private func writeBlob(repeating value: Float, count: Int, to url: URL) throws {
    let data = makeBlobData(repeating: value, count: count)
    try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
    try data.write(to: url)
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

private func shouldRunANEHardwareTests() -> Bool {
    ProcessInfo.processInfo.environment["ANE_HARDWARE_TESTS"] == "1" && aneIsAvailable()
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
