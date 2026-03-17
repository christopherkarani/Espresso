public enum ModelRegistry {
    public static let gpt2_124m = MultiModelConfig(
        name: "gpt2_124m",
        nLayer: 12,
        nHead: 12,
        nKVHead: 12,
        dModel: 768,
        headDim: 64,
        hiddenDim: 3_072,
        vocab: 50_257,
        maxSeq: 1_024,
        normEps: 1e-5,
        architecture: .gpt2
    )

    public static let stories110m = MultiModelConfig(
        name: "stories110m",
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

    public static let smolLM_135m = MultiModelConfig(
        name: "smolLM_135m",
        nLayer: 30,
        nHead: 9,
        nKVHead: 3,
        dModel: 576,
        headDim: 64,
        hiddenDim: 1_536,
        vocab: 49_152,
        maxSeq: 2_048,
        normEps: 1e-5,
        architecture: .llama
    )

    public static let tinyLlama_1_1b = MultiModelConfig(
        name: "tinyLlama_1_1b",
        nLayer: 22,
        nHead: 32,
        nKVHead: 4,
        dModel: 2_048,
        headDim: 64,
        hiddenDim: 5_632,
        vocab: 32_000,
        maxSeq: 2_048,
        normEps: 1e-5,
        architecture: .llama
    )

    /// Llama 3.2 1B — GQA (nKVHead=8), SwiGLU, RMSNorm, RoPE theta=500000.
    public static let llama3_2_1b = MultiModelConfig(
        name: "llama3_2_1b",
        nLayer: 16,
        nHead: 32,
        nKVHead: 8,
        dModel: 2_048,
        headDim: 64,
        hiddenDim: 8_192,
        vocab: 128_256,
        maxSeq: 2_048,
        normEps: 1e-5,
        ropeTheta: 500_000.0,
        architecture: .llama
    )

    /// Llama 3.2 3B — GQA (nKVHead=8), SwiGLU, RMSNorm, RoPE theta=500000.
    public static let llama3_2_3b = MultiModelConfig(
        name: "llama3_2_3b",
        nLayer: 28,
        nHead: 24,
        nKVHead: 8,
        dModel: 3_072,
        headDim: 128,
        hiddenDim: 8_192,
        vocab: 128_256,
        maxSeq: 2_048,
        normEps: 1e-5,
        ropeTheta: 500_000.0,
        architecture: .llama
    )

    public static let all: [String: MultiModelConfig] = [
        gpt2_124m.name: gpt2_124m,
        stories110m.name: stories110m,
        smolLM_135m.name: smolLM_135m,
        tinyLlama_1_1b.name: tinyLlama_1_1b,
        llama3_2_1b.name: llama3_2_1b,
        llama3_2_3b.name: llama3_2_3b,
    ]

    public static func config(named name: String) -> MultiModelConfig? {
        all[name]
    }
}
