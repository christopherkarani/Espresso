import Foundation
import Testing
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
    #expect(ModelRegistry.llama3_2_3b.ropeTheta == 500_000.0)
    #expect(ModelRegistry.stories110m.ropeTheta == 10_000.0)
    #expect(ModelRegistry.gpt2_124m.ropeTheta == 10_000.0)
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
