import Testing
import ANEGraphIR
import ANEPasses
@testable import ModelSupport

@Test func gpt2LayerGraphUsesLayerNormGeluAndBiases() throws {
    let config = ModelRegistry.gpt2_124m
    let paths = LayerWeightPaths.forLayer(0, config: config, blobDir: "/tmp/gpt2")
    let graph = TransformerLayerGraphBuilder.forwardLayer(
        layer: 0,
        config: config,
        paths: paths,
        spatial: config.maxSeq
    )

    #expect(graph.graphInputs.map(\.name) == ["x"])
    #expect(graph.graphOutputs.map(\.name) == ["out"])
    #expect(graph.nodes.contains { $0.name == "layer0_ln1_gamma" && $0.op == .const })
    #expect(graph.nodes.contains { $0.name == "layer0_ln1_beta" && $0.op == .const })
    #expect(graph.nodes.contains { $0.name == "layer0_ffn_act_tanh" && $0.op == .tanh })
    #expect(graph.nodes.contains { $0.name == "layer0_q_bias" })
    #expect(graph.nodes.contains { $0.name == "layer0_attn_proj_bias" })
    #expect(graph.nodes.contains { $0.name == "layer0_ffn_up_bias" })
    #expect(graph.nodes.contains { $0.name == "layer0_ffn_down_bias" })
    #expect(allWeightOffsets(in: graph).allSatisfy { $0 == 64 })
    #expect(!ANEOptimizationPipeline.validate(graph).contains { $0.severity == .error })
}

@Test func llamaLayerGraphUsesRmsNormSwiGLUAndNoBiases() throws {
    let config = ModelRegistry.stories110m
    let paths = LayerWeightPaths.forLayer(0, config: config, blobDir: "/tmp/llama")
    let graph = TransformerLayerGraphBuilder.forwardLayer(
        layer: 0,
        config: config,
        paths: paths,
        spatial: config.maxSeq
    )

    #expect(graph.nodes.contains { $0.name == "layer0_rms1_weight" && $0.op == .const })
    #expect(graph.nodes.contains { $0.name == "layer0_ffn_gate_act_sigmoid" && $0.op == .sigmoid })
    #expect(graph.nodes.contains { $0.name == "layer0_ffn_gated" && $0.op == .mul })
    // Llama-family paths always offer q/k/v bias; the graph emits those consts when
    // the strings are non-nil. o_proj and the MLP stay bias-free.
    #expect(paths.bq != nil && paths.bk != nil && paths.bv != nil)
    #expect(paths.bo == nil && paths.b1 == nil && paths.b2 == nil)
    #expect(graph.nodes.contains { $0.name == "layer0_q_bias" && $0.op == .const })
    #expect(graph.nodes.contains { $0.name == "layer0_k_bias" && $0.op == .const })
    #expect(graph.nodes.contains { $0.name == "layer0_v_bias" && $0.op == .const })
    #expect(!graph.nodes.contains { $0.name == "layer0_attn_proj_bias" })
    #expect(!graph.nodes.contains { $0.name.contains("ffn") && $0.name.contains("bias") })
    #expect(allWeightOffsets(in: graph).allSatisfy { $0 == 64 })
    #expect(!ANEOptimizationPipeline.validate(graph).contains { $0.severity == .error })
}

@Test func llamaLayerGraphIncludesQKVBiasWhenBiasPathsOffered() throws {
    let config = ModelRegistry.stories110m
    let offered = LayerWeightPaths.forLayer(0, config: config, blobDir: "/tmp/qwen-qkv-bias")
    #expect(offered.bq != nil)
    let graph = TransformerLayerGraphBuilder.forwardLayer(
        layer: 0,
        config: config,
        paths: offered,
        spatial: config.maxSeq
    )

    #expect(graph.nodes.contains { $0.name == "layer0_q_bias" && $0.op == .const })
    #expect(graph.nodes.contains { $0.name == "layer0_k_bias" && $0.op == .const })
    #expect(graph.nodes.contains { $0.name == "layer0_v_bias" && $0.op == .const })
    #expect(!graph.nodes.contains { $0.name == "layer0_attn_proj_bias" })
}

@Test func llamaLayerPathsExposeOptionalQKNormArtifacts() {
    let config = ModelRegistry.llama3_2_1b_ctx512
    let paths = LayerWeightPaths.forLayer(3, config: config, blobDir: "/tmp/qwen")

    #expect(paths.qNorm == "/tmp/qwen/layers/3/q_norm.bin")
    #expect(paths.kNorm == "/tmp/qwen/layers/3/k_norm.bin")
    // Q/K/V bias paths are always offered on the llama family; on-disk presence decides
    // whether they are used, because Qwen2 biases q/k/v and plain llama does not.
    #expect(paths.bq == "/tmp/qwen/layers/3/bq.bin")
    #expect(paths.bk == "/tmp/qwen/layers/3/bk.bin")
    #expect(paths.bv == "/tmp/qwen/layers/3/bv.bin")
    // o_proj and the MLP are bias-free in every llama-family checkpoint we serve.
    #expect(paths.bo == nil)
    #expect(paths.b1 == nil)
    #expect(paths.b2 == nil)
}

@Test func supportedMaskBucketsMatchConverterContract() {
    #expect(TransformerLayerGraphBuilder.isSupportedMaskBucket(spatial: 1, maxSeq: 256))
    #expect(TransformerLayerGraphBuilder.isSupportedMaskBucket(spatial: 16, maxSeq: 256))
    #expect(TransformerLayerGraphBuilder.isSupportedMaskBucket(spatial: 256, maxSeq: 256))
    #expect(!TransformerLayerGraphBuilder.isSupportedMaskBucket(spatial: 17, maxSeq: 256))
    #expect(!TransformerLayerGraphBuilder.isSupportedMaskBucket(spatial: 512, maxSeq: 256))
}

private func allWeightOffsets(in graph: ANEGraph) -> [UInt64] {
    graph.nodes.compactMap { node in
        guard case let .weight(_, offset) = node.attrs else {
            return nil
        }
        return offset
    }
}
