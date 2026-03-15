import Testing
@testable import ANEBuilder
import ANEGraphIR

@Test func ffnWithGeluUsesTwoLinearStagesAndCompositeActivation() throws {
    var graph = ANEGraph()
    let x = try graph.input("x", dtype: .fp16, shape: try ANEShape(channels: 16, spatial: 8))

    let out = try graph.ffn(
        "ffn",
        input: x,
        inDim: 16,
        hiddenDim: 32,
        spatial: 8,
        w1Path: "@w1",
        b1Path: "@b1",
        w2Path: "@w2",
        b2Path: "@b2",
        activation: .gelu
    )

    #expect(graph.nodes[out].name == "ffn_down_out")
    #expect(graph.nodes.filter { $0.op == .conv1x1 }.count == 2)
    #expect(graph.nodes.contains { $0.name == "ffn_act_tanh" && $0.op == .tanh })
    #expect(graph.nodes.contains { $0.name == "ffn_up_bias" })
    #expect(graph.nodes.contains { $0.name == "ffn_down_bias" })
}

@Test func ffnWithReluUsesPrimitiveReluMiddleStage() throws {
    var graph = ANEGraph()
    let x = try graph.input("x", dtype: .fp16, shape: try ANEShape(channels: 12, spatial: 4))

    _ = try graph.ffn(
        "ffn",
        input: x,
        inDim: 12,
        hiddenDim: 24,
        spatial: 4,
        w1Path: "@w1",
        w2Path: "@w2",
        activation: .relu
    )

    #expect(graph.nodes.contains { $0.name == "ffn_act" && $0.op == .relu })
    #expect(!graph.nodes.contains { $0.name == "ffn_act_tanh" })
}

@Test func swigluFfnBuildsThreeLinearLayersAndSiluGate() throws {
    var graph = ANEGraph()
    let x = try graph.input("x", dtype: .fp16, shape: try ANEShape(channels: 10, spatial: 3))

    let out = try graph.swigluFFN(
        "swi",
        input: x,
        inDim: 10,
        hiddenDim: 20,
        spatial: 3,
        w1Path: "@w1",
        w3Path: "@w3",
        w2Path: "@w2"
    )

    #expect(graph.nodes[out].name == "swi_down_conv")
    #expect(graph.nodes.filter { $0.op == .conv1x1 }.count == 3)
    #expect(graph.nodes.contains { $0.name == "swi_gate_act_sigmoid" && $0.op == .sigmoid })
    #expect(graph.nodes.contains { $0.name == "swi_gated" && $0.op == .mul })
}

@Test func loraLinearDeclaresAdaptersAsInputsAndNotConstWeights() throws {
    var graph = ANEGraph()
    let x = try graph.input("x", dtype: .fp16, shape: try ANEShape(channels: 16, spatial: 5))

    let out = try graph.loraLinear(
        "proj",
        input: x,
        inDim: 16,
        outDim: 24,
        spatial: 5,
        weightPath: "@w",
        loraAName: "proj_loraA",
        loraBName: "proj_loraB",
        rank: 4,
        alpha: 8
    )

    #expect(graph.nodes[out].name == "proj_out")
    #expect(graph.graphInputs.map(\.name).contains("proj_loraA"))
    #expect(graph.graphInputs.map(\.name).contains("proj_loraB"))
    #expect(!graph.nodes.contains { node in
        switch node.attrs {
        case let .weight(blobPath, _):
            return blobPath == "proj_loraA" || blobPath == "proj_loraB"
        default:
            return false
        }
    })
    #expect(graph.nodes.contains { $0.name == "proj_lora_low_rank" && $0.op == .matmul })
    #expect(graph.nodes.contains { $0.name == "proj_lora_projected" && $0.op == .matmul })
    #expect(graph.nodes.contains { $0.name == "proj_lora_scale" && $0.attrs == .scalar(2.0) })
}

@Test func loraLinearUsesExpectedAdapterShapes() throws {
    var graph = ANEGraph()
    let x = try graph.input("x", dtype: .fp16, shape: try ANEShape(channels: 6, spatial: 7))

    _ = try graph.loraLinear(
        "proj",
        input: x,
        inDim: 6,
        outDim: 10,
        spatial: 7,
        weightPath: "@w",
        loraAName: "proj_loraA",
        loraBName: "proj_loraB",
        rank: 3,
        alpha: 6
    )

    let aNode = try #require(graph.nodes.first { $0.name == "proj_loraA" })
    let bNode = try #require(graph.nodes.first { $0.name == "proj_loraB" })
    let expectedAShape = try ANEShape(batch: 1, channels: 1, height: 6, spatial: 3)
    let expectedBShape = try ANEShape(batch: 1, channels: 1, height: 10, spatial: 3)
    #expect(aNode.op == .input)
    #expect(bNode.op == .input)
    #expect(aNode.shape == expectedAShape)
    #expect(bNode.shape == expectedBShape)
}
