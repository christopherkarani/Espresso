import Testing
@testable import ANEPasses
import ANEGraphIR

@Test func identityNodeIsBypassed() throws {
    var graph = ANEGraph()
    let x = try graph.addNode(inputNode(name: "x"))
    let identity = try graph.addNode(
        ANENode(op: .identity, name: "identity", dtype: .fp16, shape: shape(), inputs: [x])
    )
    let relu = try graph.addNode(
        ANENode(op: .relu, name: "relu", dtype: .fp16, shape: shape(), inputs: [identity], isOutput: true)
    )
    try graph.setGraphOutputs([GraphPort(name: "relu", nodeIndex: relu)])

    var pass = IdentityEliminationPass()
    let changed = pass.run(on: &graph)

    #expect(changed)
    #expect(graph.nodes[relu].inputs == [x])
    #expect(!graph.nodes[identity].isLive)
    #expect(graph.nodes[relu].isLive)
}

@Test func identityOutputPortIsRewritten() throws {
    var graph = ANEGraph()
    let x = try graph.addNode(inputNode(name: "x"))
    let identity = try graph.addNode(
        ANENode(op: .identity, name: "identity", dtype: .fp16, shape: shape(), inputs: [x], isOutput: true)
    )
    try graph.setGraphOutputs([GraphPort(name: "identity", nodeIndex: identity)])

    var pass = IdentityEliminationPass()
    let changed = pass.run(on: &graph)

    #expect(changed)
    #expect(graph.graphOutputs == [GraphPort(name: "identity", nodeIndex: x)])
    #expect(graph.nodes[x].isOutput)
    #expect(!graph.nodes[identity].isLive)
    #expect(!graph.nodes[identity].isOutput)
}

@Test func sameDTypeCastIsBypassed() throws {
    var graph = ANEGraph()
    let x = try graph.addNode(inputNode(name: "x"))
    let cast = try graph.addNode(
        ANENode(
            op: .cast,
            name: "cast",
            dtype: .fp16,
            shape: shape(),
            inputs: [x],
            attrs: .cast(target: .fp16)
        )
    )
    let relu = try graph.addNode(
        ANENode(op: .relu, name: "relu", dtype: .fp16, shape: shape(), inputs: [cast], isOutput: true)
    )
    try graph.setGraphOutputs([GraphPort(name: "relu", nodeIndex: relu)])

    var pass = IdentityEliminationPass()
    _ = pass.run(on: &graph)

    #expect(graph.nodes[relu].inputs == [x])
    #expect(!graph.nodes[cast].isLive)
}

@Test func sameShapeReshapeIsBypassed() throws {
    var graph = ANEGraph()
    let x = try graph.addNode(inputNode(name: "x"))
    let reshape = try graph.addNode(
        ANENode(op: .reshape, name: "reshape", dtype: .fp16, shape: shape(), inputs: [x])
    )
    let relu = try graph.addNode(
        ANENode(op: .relu, name: "relu", dtype: .fp16, shape: shape(), inputs: [reshape], isOutput: true)
    )
    try graph.setGraphOutputs([GraphPort(name: "relu", nodeIndex: relu)])

    var pass = IdentityEliminationPass()
    _ = pass.run(on: &graph)

    #expect(graph.nodes[relu].inputs == [x])
    #expect(!graph.nodes[reshape].isLive)
}

@Test func identityTransposeIsBypassed() throws {
    var graph = ANEGraph()
    let x = try graph.addNode(inputNode(name: "x", shape: shape(channels: 32, spatial: 64)))
    let transpose = try graph.addNode(
        ANENode(
            op: .transpose,
            name: "transpose",
            dtype: .fp16,
            shape: shape(channels: 32, spatial: 64),
            inputs: [x],
            attrs: .transpose(perm: [0, 1, 2, 3])
        )
    )
    let relu = try graph.addNode(
        ANENode(op: .relu, name: "relu", dtype: .fp16, shape: shape(channels: 32, spatial: 64), inputs: [transpose], isOutput: true)
    )
    try graph.setGraphOutputs([GraphPort(name: "relu", nodeIndex: relu)])

    var pass = IdentityEliminationPass()
    _ = pass.run(on: &graph)

    #expect(graph.nodes[relu].inputs == [x])
    #expect(!graph.nodes[transpose].isLive)
}

@Test func allConsumersAreRewritten() throws {
    var graph = ANEGraph()
    let x = try graph.addNode(inputNode(name: "x"))
    let identity = try graph.addNode(
        ANENode(op: .identity, name: "identity", dtype: .fp16, shape: shape(), inputs: [x])
    )
    let relu = try graph.addNode(
        ANENode(op: .relu, name: "relu", dtype: .fp16, shape: shape(), inputs: [identity], isOutput: true)
    )
    let sigmoid = try graph.addNode(
        ANENode(op: .sigmoid, name: "sigmoid", dtype: .fp16, shape: shape(), inputs: [identity], isOutput: true)
    )
    try graph.setGraphOutputs([
        GraphPort(name: "relu", nodeIndex: relu),
        GraphPort(name: "sigmoid", nodeIndex: sigmoid),
    ])

    var pass = IdentityEliminationPass()
    _ = pass.run(on: &graph)

    #expect(graph.nodes[relu].inputs == [x])
    #expect(graph.nodes[sigmoid].inputs == [x])
}

@Test func differentDTypeCastIsPreserved() throws {
    var graph = ANEGraph()
    let x = try graph.addNode(inputNode(name: "x"))
    let cast = try graph.addNode(
        ANENode(
            op: .cast,
            name: "cast",
            dtype: .fp32,
            shape: shape(),
            inputs: [x],
            attrs: .cast(target: .fp32)
        )
    )
    try graph.setGraphOutputs([GraphPort(name: "cast", nodeIndex: cast)])

    var pass = IdentityEliminationPass()
    let changed = pass.run(on: &graph)

    #expect(!changed)
    #expect(graph.nodes[cast].isLive)
}

@Test func nonIdentityTransposeIsPreserved() throws {
    var graph = ANEGraph()
    let x = try graph.addNode(inputNode(name: "x", shape: shape(channels: 16, height: 2, spatial: 8)))
    let transpose = try graph.addNode(
        ANENode(
            op: .transpose,
            name: "transpose",
            dtype: .fp16,
            shape: shape(channels: 16, height: 8, spatial: 2),
            inputs: [x],
            attrs: .transpose(perm: [0, 1, 3, 2]),
            isOutput: true
        )
    )
    try graph.setGraphOutputs([GraphPort(name: "transpose", nodeIndex: transpose)])

    var pass = IdentityEliminationPass()
    let changed = pass.run(on: &graph)

    #expect(!changed)
    #expect(graph.graphOutputs[0].nodeIndex == transpose)
    #expect(graph.nodes[transpose].isLive)
}

private func shape(channels: Int = 256, height: Int = 1, spatial: Int = 128) -> ANEShape {
    try! ANEShape(channels: channels, height: height, spatial: spatial)
}

private func inputNode(name: String, dtype: ANEDType = .fp16, shape: ANEShape = shape()) -> ANENode {
    ANENode(op: .input, name: name, dtype: dtype, shape: shape)
}
