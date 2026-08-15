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
    #expect(ModelRegistry.all.count == 7)
    #expect(ModelRegistry.all["smolLM_135m"]?.nKVHead == 3)
    #expect(ModelRegistry.all["tinyLlama_1_1b"]?.nHead == 32)
    #expect(ModelRegistry.all["llama3_2_1b_ctx512"]?.maxSeq == 512)
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

@Test func multiModelConfigSupportsExpandedAttentionDimension() {
    let config = MultiModelConfig(
        name: "qwen3-shape",
        nLayer: 28,
        nHead: 16,
        nKVHead: 8,
        dModel: 1024,
        headDim: 128,
        hiddenDim: 3072,
        vocab: 151_936,
        maxSeq: 40960,
        normEps: 1e-6,
        architecture: .llama
    )

    #expect(config.attentionDim == 2048)
    #expect(config.kvDim == 1024)
    #expect(config.dModel != config.attentionDim)
}

@Test func modelFamilyRecognizesStories110MVariants() {
    #expect(ModelFamily.isStories110MVariant(name: "stories110m"))
    #expect(ModelFamily.isStories110MVariant(name: "stories110m-ctx256"))
    #expect(ModelFamily.isStories110MVariant(name: "llama2.c-stories110M"))
    #expect(ModelFamily.isStories110MVariant(name: "  Stories110M  "))
    #expect(!ModelFamily.isStories110MVariant(name: "gpt2_124m"))
    #expect(!ModelFamily.isStories110MVariant(name: "tinyLlama_1_1b"))
    #expect(!ModelFamily.isStories110MVariant(name: "qwen3-0.6b"))
    #expect(ModelFamily.isStories110MVariant(ModelRegistry.stories110m))
}

@Test func modelFamilyRecognizesQwenVariants() {
    #expect(ModelFamily.isQwenVariant(name: "qwen2.5"))
    #expect(ModelFamily.isQwenVariant(name: "Qwen2.5-0.5B-Instruct"))
    #expect(ModelFamily.isQwenVariant(name: "  Qwen2.5-0.5B-Instruct  "))
    #expect(!ModelFamily.isQwenVariant(name: "llama3"))
    #expect(!ModelFamily.isQwenVariant(name: "stories110m"))
    #expect(!ModelFamily.isQwenVariant(name: "gpt2_124m"))
    #expect(ModelFamily.isQwenVariant(
        MultiModelConfig(
            name: "Qwen2.5-0.5B-Instruct",
            nLayer: 24,
            nHead: 14,
            nKVHead: 2,
            dModel: 896,
            headDim: 64,
            hiddenDim: 4864,
            vocab: 151_936,
            maxSeq: 4096,
            normEps: 1e-6,
            architecture: .llama
        )
    ))
}

@Test func modelFamilyRecognizesQwen15BVariants() {
    #expect(ModelFamily.isQwen15BVariant(name: "Qwen2.5-1.5B-Instruct"))
    #expect(ModelFamily.isQwen15BVariant(name: "  Qwen2.5-1.5B-Instruct  "))
    #expect(ModelFamily.isQwen15BVariant(name: "qwen2.5-1.5b"))
    #expect(!ModelFamily.isQwen15BVariant(name: "Qwen2.5-0.5B-Instruct"))
    #expect(!ModelFamily.isQwen15BVariant(name: "qwen2.5"))
    #expect(!ModelFamily.isQwen15BVariant(name: "stories110m"))
    #expect(!ModelFamily.isQwen15BVariant(name: "tinyLlama_1_1b"))
    #expect(ModelFamily.isQwen15BVariant(
        MultiModelConfig(
            name: "Qwen2.5-1.5B-Instruct",
            nLayer: 28,
            nHead: 12,
            nKVHead: 2,
            dModel: 1536,
            headDim: 128,
            hiddenDim: 8960,
            vocab: 151_936,
            maxSeq: 1024,
            normEps: 1e-6,
            architecture: .llama
        )
    ))
}

@Test func preferredDecodePathParseTrimsAndLowercases() throws {
    #expect(try MultiModelConfig.PreferredDecodePath.parse("hybrid") == .hybrid)
    #expect(try MultiModelConfig.PreferredDecodePath.parse(" EXACT_CPU ") == .exactCPU)
    #expect(try MultiModelConfig.PreferredDecodePath.parse("Hybrid") == .hybrid)
    #expect(throws: MultiModelConfig.PreferredDecodePath.ParseError.unsupported("metal")) {
        try MultiModelConfig.PreferredDecodePath.parse("metal")
    }
    #expect(throws: MultiModelConfig.PreferredDecodePath.ParseError.unsupported("  ")) {
        try MultiModelConfig.PreferredDecodePath.parse("  ")
    }
}
