import Testing
@testable import ANEBuilder
import ANEGraphIR

@Test func linearUsesConv1x1WeightShapeAndNoMatmul() throws {
    var graph = ANEGraph()
    let x = try graph.input("x", dtype: .fp16, shape: try ANEShape(channels: 16, spatial: 8))
    let out = try graph.linear(
        "proj",
        input: x,
        inDim: 16,
        outDim: 32,
        spatial: 8,
        weightPath: "@w",
        biasPath: nil
    )
    let expectedWeightShape = try ANEShape(batch: 32, channels: 16, height: 1, spatial: 1)

    #expect(out == 2)
    #expect(graph.nodes[1].op == .const)
    #expect(graph.nodes[2].op == .conv1x1)
    #expect(graph.nodes[1].shape == expectedWeightShape)
    #expect(!graph.nodes.contains { $0.op == .matmul })
}

@Test func linearBiasPathUsesSeparateAddNode() throws {
    var graph = ANEGraph()
    let x = try graph.input("x", dtype: .fp16, shape: try ANEShape(channels: 8, spatial: 4))
    let out = try graph.linear(
        "proj",
        input: x,
        inDim: 8,
        outDim: 12,
        spatial: 4,
        weightPath: "@w",
        biasPath: "@b"
    )

    #expect(graph.nodes[out].op == .add)
    #expect(graph.nodes[out].name == "proj_out")
    #expect(graph.nodes[3].attrs == .weight(blobPath: "@b", offset: 64))
}

@Test func rmsNormUsesReduceSumScalarMeanAndPowNegativeHalf() throws {
    var graph = ANEGraph()
    let x = try graph.input("x", dtype: .fp16, shape: try ANEShape(channels: 16, spatial: 8))
    let out = try graph.rmsNorm("rms", input: x, dim: 16, spatial: 8, eps: 1e-5, weightPath: "@gamma")

    let ops = graph.nodes.map(\.op)
    #expect(ops.contains(.reduceSum))
    #expect(!ops.contains(.reduceMean))
    #expect(!ops.contains(.rsqrt))
    #expect(graph.nodes[out].name == "rms_out")
    #expect(graph.nodes.contains { $0.name == "rms_invd" && $0.attrs == .scalar(1.0 / 16.0) })
    #expect(graph.nodes.contains { $0.name == "rms_nhalf" && $0.attrs == .scalar(-0.5) })
}

@Test func layerNormCentersThenAppliesGammaAndBeta() throws {
    var graph = ANEGraph()
    let x = try graph.input("x", dtype: .fp16, shape: try ANEShape(channels: 32, spatial: 4))
    let out = try graph.layerNorm(
        "ln",
        input: x,
        dim: 32,
        spatial: 4,
        eps: 1e-5,
        gammaPath: "@gamma",
        betaPath: "@beta"
    )

    #expect(graph.nodes[out].op == .add)
    #expect(graph.nodes.contains { $0.name == "ln_centered" && $0.op == .sub })
    #expect(graph.nodes.contains { $0.name == "ln_gamma" && $0.attrs == .weight(blobPath: "@gamma", offset: 64) })
    #expect(graph.nodes.contains { $0.name == "ln_beta" && $0.attrs == .weight(blobPath: "@beta", offset: 64) })
}

@Test func geluExpandsToTanhApproximation() throws {
    var graph = ANEGraph()
    let x = try graph.input("x", dtype: .fp16, shape: try ANEShape(channels: 8, spatial: 8))
    let out = try graph.gelu("gelu", input: x)

    #expect(graph.nodes[out].name == "gelu_out")
    #expect(graph.nodes.contains { $0.name == "gelu_tanh" && $0.op == .tanh })
    #expect(graph.nodes.contains { $0.name == "gelu_cubic" && $0.attrs == .scalar(0.044715) })
    #expect(graph.nodes.contains { $0.name == "gelu_scale" && $0.attrs == .scalar(0.7978846) })
}

@Test func siluIsSigmoidThenMultiply() throws {
    var graph = ANEGraph()
    let x = try graph.input("x", dtype: .fp16, shape: try ANEShape(channels: 8, spatial: 8))
    let out = try graph.silu("silu", input: x)

    #expect(graph.nodes[out].op == .mul)
    #expect(graph.nodes.contains { $0.name == "silu_sigmoid" && $0.op == .sigmoid })
}

@Test func residualWrapsSingleAdd() throws {
    var graph = ANEGraph()
    let x = try graph.input("x", dtype: .fp16, shape: try ANEShape(channels: 8, spatial: 2))
    let y = try graph.input("y", dtype: .fp16, shape: try ANEShape(channels: 8, spatial: 2))
    let out = try graph.residual("res", x: x, sublayer: y)

    #expect(graph.nodes[out].op == .add)
    #expect(graph.nodes[out].name == "res_out")
}

@Test func castWrappersUsePrimitiveCast() throws {
    var graph = ANEGraph()
    let x = try graph.input("x", dtype: .fp32, shape: try ANEShape(channels: 4, spatial: 4))
    let fp16 = try graph.castToFP16("to16", input: x)
    let fp32 = try graph.castToFP32("to32", input: fp16)

    #expect(graph.nodes[fp16].dtype == .fp16)
    #expect(graph.nodes[fp32].dtype == .fp32)
}

@Test func differentPrefixesPreventNameCollisions() throws {
    var graph = ANEGraph()
    let x = try graph.input("x", dtype: .fp16, shape: try ANEShape(channels: 16, spatial: 8))
    _ = try graph.rmsNorm("block0", input: x, dim: 16, spatial: 8, eps: 1e-5, weightPath: "@g0")
    _ = try graph.rmsNorm("block1", input: x, dim: 16, spatial: 8, eps: 1e-5, weightPath: "@g1")

    let names = graph.nodes.map(\.name)
    #expect(Set(names).count == names.count)
    #expect(names.contains("block0_sq"))
    #expect(names.contains("block1_sq"))
}
