import Testing
@testable import ANEPasses
import ANEGraphIR

@Test func fp16RoundTripCastIsBypassed() throws {
    var graph = ANEGraph()
    let x = try graph.addNode(inputNode(name: "x", dtype: .fp16))
    let castToFP32 = try graph.addNode(
        ANENode(op: .cast, name: "to_fp32", dtype: .fp32, shape: shape(), inputs: [x], attrs: .cast(target: .fp32))
    )
    let castToFP16 = try graph.addNode(
        ANENode(op: .cast, name: "to_fp16", dtype: .fp16, shape: shape(), inputs: [castToFP32], attrs: .cast(target: .fp16))
    )
    let relu = try graph.addNode(
        ANENode(op: .relu, name: "relu", dtype: .fp16, shape: shape(), inputs: [castToFP16], isOutput: true)
    )
    try graph.setGraphOutputs([GraphPort(name: "relu", nodeIndex: relu)])

    var pass = CastEliminationPass()
    let changed = pass.run(on: &graph)

    #expect(changed)
    #expect(graph.nodes[relu].inputs == [x])
    #expect(!graph.nodes[castToFP16].isLive)
    #expect(graph.nodes[castToFP32].isLive)
}

@Test func fp32RoundTripCastIsBypassed() throws {
    var graph = ANEGraph()
    let x = try graph.addNode(inputNode(name: "x", dtype: .fp32))
    let castToFP16 = try graph.addNode(
        ANENode(op: .cast, name: "to_fp16", dtype: .fp16, shape: shape(dtype: .fp16), inputs: [x], attrs: .cast(target: .fp16))
    )
    let castToFP32 = try graph.addNode(
        ANENode(op: .cast, name: "to_fp32", dtype: .fp32, shape: shape(dtype: .fp16), inputs: [castToFP16], attrs: .cast(target: .fp32))
    )
    let relu = try graph.addNode(
        ANENode(op: .relu, name: "relu", dtype: .fp32, shape: shape(dtype: .fp32), inputs: [castToFP32], isOutput: true)
    )
    try graph.setGraphOutputs([GraphPort(name: "relu", nodeIndex: relu)])

    var pass = CastEliminationPass()
    _ = pass.run(on: &graph)

    #expect(graph.nodes[relu].inputs == [x])
    #expect(!graph.nodes[castToFP32].isLive)
}

@Test func nonRoundTripCastPairIsPreserved() throws {
    var graph = ANEGraph()
    let x = try graph.addNode(inputNode(name: "x", dtype: .fp16))
    let castToInt = try graph.addNode(
        ANENode(op: .cast, name: "to_int", dtype: .int32, shape: shape(), inputs: [x], attrs: .cast(target: .int32))
    )
    let castBack = try graph.addNode(
        ANENode(op: .cast, name: "to_fp16", dtype: .fp16, shape: shape(), inputs: [castToInt], attrs: .cast(target: .fp16), isOutput: true)
    )
    try graph.setGraphOutputs([GraphPort(name: "to_fp16", nodeIndex: castBack)])

    var pass = CastEliminationPass()
    let changed = pass.run(on: &graph)

    #expect(!changed)
    #expect(graph.graphOutputs[0].nodeIndex == castBack)
    #expect(graph.nodes[castBack].isLive)
}

@Test func graphOutputRewritesToOriginalSource() throws {
    var graph = ANEGraph()
    let x = try graph.addNode(inputNode(name: "x", dtype: .fp16))
    let castToFP32 = try graph.addNode(
        ANENode(op: .cast, name: "to_fp32", dtype: .fp32, shape: shape(), inputs: [x], attrs: .cast(target: .fp32))
    )
    let castToFP16 = try graph.addNode(
        ANENode(op: .cast, name: "to_fp16", dtype: .fp16, shape: shape(), inputs: [castToFP32], attrs: .cast(target: .fp16), isOutput: true)
    )
    try graph.setGraphOutputs([GraphPort(name: "to_fp16", nodeIndex: castToFP16)])

    var pass = CastEliminationPass()
    _ = pass.run(on: &graph)

    #expect(graph.graphOutputs == [GraphPort(name: "to_fp16", nodeIndex: x)])
    #expect(graph.nodes[x].isOutput)
    #expect(!graph.nodes[castToFP16].isOutput)
}

private func shape(dtype: ANEDType = .fp16) -> ANEShape {
    switch dtype {
    case .fp32:
        return try! ANEShape(channels: 192, spatial: 96)
    default:
        return try! ANEShape(channels: 192, spatial: 96)
    }
}

private func inputNode(name: String, dtype: ANEDType) -> ANENode {
    ANENode(op: .input, name: name, dtype: dtype, shape: shape(dtype: dtype))
}
