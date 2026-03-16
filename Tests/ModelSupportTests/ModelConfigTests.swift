import Testing
import ANETypes
@testable import ModelSupport

@Test func registryLookupReturnsExpectedConfigs() throws {
    let gpt2 = try #require(ModelRegistry.config(named: "gpt2_124m"))
    #expect(gpt2.vocab == 50_257)
    #expect(gpt2.hiddenDim == 3_072)
    #expect(gpt2.architecture == .gpt2)

    let stories = try #require(ModelRegistry.config(named: "stories110m"))
    #expect(stories.vocab == 32_000)
    #expect(stories.hiddenDim == 2_048)
    #expect(stories.architecture == .llama)
}

@Test func registryContainsAllSixModels() {
    #expect(ModelRegistry.all.count == 6)
    #expect(ModelRegistry.all["smolLM_135m"]?.nKVHead == 3)
    #expect(ModelRegistry.all["tinyLlama_1_1b"]?.nHead == 32)
}

@Test func llama3_2_1bConfigIsCorrect() throws {
    let cfg = try #require(ModelRegistry.config(named: "llama3_2_1b"))
    #expect(cfg.nLayer == 16)
    #expect(cfg.nHead == 32)
    #expect(cfg.nKVHead == 8)
    #expect(cfg.dModel == 2048)
    #expect(cfg.headDim == 64)
    #expect(cfg.hiddenDim == 8192)
    #expect(cfg.vocab == 128_256)
    #expect(cfg.maxSeq == 2048)
    #expect(cfg.architecture == .llama)
    // dModel constraint: nHead * headDim
    #expect(cfg.dModel == cfg.nHead * cfg.headDim)
}

@Test func llama3_2_3bConfigIsCorrect() throws {
    let cfg = try #require(ModelRegistry.config(named: "llama3_2_3b"))
    #expect(cfg.nLayer == 28)
    #expect(cfg.nHead == 24)
    #expect(cfg.nKVHead == 8)
    #expect(cfg.dModel == 3072)
    #expect(cfg.headDim == 128)
    #expect(cfg.hiddenDim == 8192)
    #expect(cfg.vocab == 128_256)
    #expect(cfg.maxSeq == 2048)
    #expect(cfg.architecture == .llama)
    // dModel constraint: nHead * headDim
    #expect(cfg.dModel == cfg.nHead * cfg.headDim)
}

@Test func multiModelConfigCoexistsWithANETypesModelConfig() {
    let config = MultiModelConfig(
        name: "test",
        nLayer: 1,
        nHead: 1,
        nKVHead: 1,
        dModel: 64,
        headDim: 64,
        hiddenDim: 128,
        vocab: 256,
        maxSeq: 64,
        normEps: 1e-5,
        architecture: .llama
    )

    #expect(config.dModel == 64)
    #expect(ModelConfig.dim == 768)
}
