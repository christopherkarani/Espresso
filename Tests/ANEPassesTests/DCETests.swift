import Testing
@testable import ANEPasses
import ANEGraphIR

@Test func deadBranchIsKilled() throws {
    var graph = ANEGraph()
    let x = try graph.addNode(inputNode(name: "x"))
    let live = try graph.addNode(
        ANENode(op: .relu, name: "live", dtype: .fp16, shape: shape(), inputs: [x], isOutput: true)
    )
    let dead = try graph.addNode(
        ANENode(op: .sigmoid, name: "dead", dtype: .fp16, shape: shape(), inputs: [x])
    )
    try graph.setGraphOutputs([GraphPort(name: "live", nodeIndex: live)])

    var pass = DeadCodeEliminationPass()
    let changed = pass.run(on: &graph)

    #expect(changed)
    #expect(graph.nodes[x].isLive)
    #expect(graph.nodes[live].isLive)
    #expect(!graph.nodes[dead].isLive)
}

@Test func transitiveInputsAreKeptLive() throws {
    var graph = ANEGraph()
    let x = try graph.addNode(inputNode(name: "x"))
    let a = try graph.addNode(
        ANENode(op: .relu, name: "a", dtype: .fp16, shape: shape(), inputs: [x])
    )
    let b = try graph.addNode(
        ANENode(op: .sigmoid, name: "b", dtype: .fp16, shape: shape(), inputs: [a], isOutput: true)
    )
    let dead = try graph.addNode(
        ANENode(op: .tanh, name: "dead", dtype: .fp16, shape: shape(), inputs: [x])
    )
    try graph.setGraphOutputs([GraphPort(name: "b", nodeIndex: b)])

    var pass = DeadCodeEliminationPass()
    _ = pass.run(on: &graph)

    #expect(graph.nodes[x].isLive)
    #expect(graph.nodes[a].isLive)
    #expect(graph.nodes[b].isLive)
    #expect(!graph.nodes[dead].isLive)
}

@Test func multipleDeadBranchesAreKilled() throws {
    var graph = ANEGraph()
    let x = try graph.addNode(inputNode(name: "x"))
    let output = try graph.addNode(
        ANENode(op: .relu, name: "out", dtype: .fp16, shape: shape(), inputs: [x], isOutput: true)
    )
    let deadA = try graph.addNode(
        ANENode(op: .sigmoid, name: "dead_a", dtype: .fp16, shape: shape(), inputs: [x])
    )
    let deadB = try graph.addNode(
        ANENode(op: .mul, name: "dead_b", dtype: .fp16, shape: shape(), inputs: [x, deadA])
    )
    try graph.setGraphOutputs([GraphPort(name: "out", nodeIndex: output)])

    var pass = DeadCodeEliminationPass()
    _ = pass.run(on: &graph)

    #expect(!graph.nodes[deadA].isLive)
    #expect(!graph.nodes[deadB].isLive)
    #expect(graph.liveNodeCount == 2)
}

@Test func alreadyDeadNodesStayDead() throws {
    var graph = ANEGraph()
    let x = try graph.addNode(inputNode(name: "x"))
    let output = try graph.addNode(
        ANENode(op: .relu, name: "out", dtype: .fp16, shape: shape(), inputs: [x], isOutput: true)
    )
    graph.setNodeLiveness(at: x, isLive: false)
    try graph.setGraphOutputs([GraphPort(name: "out", nodeIndex: output)])

    var pass = DeadCodeEliminationPass()
    _ = pass.run(on: &graph)

    #expect(!graph.nodes[x].isLive)
    #expect(graph.nodes[output].isLive)
}

private func shape() -> ANEShape {
    try! ANEShape(channels: 192, spatial: 96)
}

private func inputNode(name: String) -> ANENode {
    ANENode(op: .input, name: name, dtype: .fp16, shape: shape())
}
