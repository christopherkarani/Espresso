import Testing
import ANECodegen
import ANEGraphIR
import ANEPasses
@testable import ModelSupport

@Test func preRoPEGraphCompiles() throws {
    let config = ModelRegistry.stories110m
    let paths = LayerWeightPaths.forLayer(0, config: config, blobDir: "/tmp/llama")
    let graph = TransformerLayerGraphBuilder.preRoPEForwardLayer(
        layer: 0,
        config: config,
        paths: paths,
        spatial: config.maxSeq
    )

    #expect(graph.graphInputs.map(\.name) == ["x"])
    #expect(!graph.nodes.isEmpty)

    var optimized = graph
    ANEOptimizationPipeline.optimize(&optimized)
    let mil = ANECodegen.emit(optimized)
    #expect(!mil.isEmpty)
    #expect(mil.contains("func main"))
}

@Test func postRoPEGraphCompiles() throws {
    let config = ModelRegistry.stories110m
    let paths = LayerWeightPaths.forLayer(0, config: config, blobDir: "/tmp/llama")
    let graph = TransformerLayerGraphBuilder.postRoPEForwardLayer(
        layer: 0,
        config: config,
        paths: paths,
        spatial: config.maxSeq
    )

    #expect(!graph.nodes.isEmpty)

    var optimized = graph
    ANEOptimizationPipeline.optimize(&optimized)
    let mil = ANECodegen.emit(optimized)
    #expect(!mil.isEmpty)
    #expect(mil.contains("func main"))
}

@Test func preRoPEGraphOutputNames() throws {
    let config = ModelRegistry.stories110m
    let paths = LayerWeightPaths.forLayer(0, config: config, blobDir: "/tmp/llama")
    let graph = TransformerLayerGraphBuilder.preRoPEForwardLayer(
        layer: 0,
        config: config,
        paths: paths,
        spatial: config.maxSeq
    )

    // Outputs should be in alphabetical order: k, q, v
    let outputNames = graph.graphOutputs.map(\.name)
    #expect(outputNames == ["k", "q", "v"])
}

@Test func preRoPEGraphUsesRmsNormNoBias() throws {
    let config = ModelRegistry.stories110m
    let paths = LayerWeightPaths.forLayer(0, config: config, blobDir: "/tmp/llama")
    let graph = TransformerLayerGraphBuilder.preRoPEForwardLayer(
        layer: 0,
        config: config,
        paths: paths,
        spatial: config.maxSeq
    )

    // Should use RMSNorm (has _weight, no _beta)
    #expect(graph.nodes.contains { $0.name == "layer0_rms1_weight" && $0.op == .const })
    #expect(!graph.nodes.contains { $0.name.contains("bias") || $0.name.contains("beta") })
}

@Test func postRoPEGraphInputsAreAlphabetical() throws {
    let config = ModelRegistry.stories110m
    let paths = LayerWeightPaths.forLayer(0, config: config, blobDir: "/tmp/llama")
    let graph = TransformerLayerGraphBuilder.postRoPEForwardLayer(
        layer: 0,
        config: config,
        paths: paths,
        spatial: config.maxSeq
    )

    let inputNames = graph.graphInputs.map(\.name)
    #expect(inputNames == ["k_rotated", "q_rotated", "v", "x_residual"])
}
