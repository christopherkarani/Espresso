import Testing
@testable import ANEPasses
import ANEGraphIR

@Test func pipelineOptimizesMixedGraphToInputAndOutput() throws {
    var graph = ANEGraph()
    let x = try graph.addNode(inputNode(name: "x"))
    let identity = try graph.addNode(
        ANENode(op: .identity, name: "identity", dtype: .fp16, shape: shape(), inputs: [x])
    )
    let castToFP32 = try graph.addNode(
        ANENode(op: .cast, name: "to_fp32", dtype: .fp32, shape: shape(), inputs: [identity], attrs: .cast(target: .fp32))
    )
    let castToFP16 = try graph.addNode(
        ANENode(op: .cast, name: "to_fp16", dtype: .fp16, shape: shape(), inputs: [castToFP32], attrs: .cast(target: .fp16))
    )
    let relu = try graph.addNode(
        ANENode(op: .relu, name: "relu", dtype: .fp16, shape: shape(), inputs: [castToFP16], isOutput: true)
    )
    let dead = try graph.addNode(
        ANENode(op: .sigmoid, name: "dead", dtype: .fp16, shape: shape(), inputs: [x])
    )
    _ = dead
    try graph.setGraphOutputs([GraphPort(name: "relu", nodeIndex: relu)])

    let changed = ANEOptimizationPipeline.optimize(&graph)

    #expect(changed)
    #expect(graph.nodes[relu].inputs == [x])
    #expect(graph.liveNodeCount == 2)
    #expect(graph.nodes[x].isLive)
    #expect(graph.nodes[relu].isLive)
    #expect(graph.graphOutputs == [GraphPort(name: "relu", nodeIndex: relu)])
}

@Test func alreadyOptimalGraphDoesNotChange() throws {
    var graph = ANEGraph()
    let x = try graph.addNode(inputNode(name: "x"))
    let relu = try graph.addNode(
        ANENode(op: .relu, name: "relu", dtype: .fp16, shape: shape(), inputs: [x], isOutput: true)
    )
    try graph.setGraphOutputs([GraphPort(name: "relu", nodeIndex: relu)])

    let changed = ANEOptimizationPipeline.optimize(&graph)

    #expect(!changed)
    #expect(graph.liveNodeCount == 2)
    #expect(graph.nodes[relu].inputs == [x])
}

@Test func optimizeReachesFixpoint() throws {
    var graph = ANEGraph()
    let x = try graph.addNode(inputNode(name: "x"))
    let identity = try graph.addNode(
        ANENode(op: .identity, name: "identity", dtype: .fp16, shape: shape(), inputs: [x])
    )
    let relu = try graph.addNode(
        ANENode(op: .relu, name: "relu", dtype: .fp16, shape: shape(), inputs: [identity], isOutput: true)
    )
    try graph.setGraphOutputs([GraphPort(name: "relu", nodeIndex: relu)])

    #expect(ANEOptimizationPipeline.optimize(&graph))
    #expect(!ANEOptimizationPipeline.optimize(&graph))
    #expect(graph.liveNodeCount == 2)
    #expect(graph.nodes[relu].inputs == [x])
}

@Test func pipelineHandlesMultipleInteractingOptimizations() throws {
    var graph = ANEGraph()
    let x = try graph.addNode(inputNode(name: "x"))
    let castToFP32 = try graph.addNode(
        ANENode(op: .cast, name: "to_fp32", dtype: .fp32, shape: shape(), inputs: [x], attrs: .cast(target: .fp32))
    )
    let castToFP16 = try graph.addNode(
        ANENode(op: .cast, name: "to_fp16", dtype: .fp16, shape: shape(), inputs: [castToFP32], attrs: .cast(target: .fp16))
    )
    let reshape = try graph.addNode(
        ANENode(op: .reshape, name: "reshape", dtype: .fp16, shape: shape(), inputs: [castToFP16])
    )
    let identity = try graph.addNode(
        ANENode(op: .identity, name: "identity", dtype: .fp16, shape: shape(), inputs: [reshape])
    )
    let output = try graph.addNode(
        ANENode(op: .relu, name: "output", dtype: .fp16, shape: shape(), inputs: [identity], isOutput: true)
    )
    try graph.setGraphOutputs([GraphPort(name: "output", nodeIndex: output)])

    let changed = ANEOptimizationPipeline.optimize(&graph)

    #expect(changed)
    #expect(graph.nodes[output].inputs == [x])
    #expect(graph.liveNodeCount == 2)
}

private func shape() -> ANEShape {
    try! ANEShape(channels: 256, spatial: 128)
}

private func inputNode(name: String) -> ANENode {
    ANENode(op: .input, name: name, dtype: .fp16, shape: shape())
}
