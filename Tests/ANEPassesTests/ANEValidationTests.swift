import Testing
@testable import ANEPasses
import ANEGraphIR

@Test func concatBannedProducesError() throws {
    var graph = ANEGraph()
    let x = try graph.addNode(inputNode(name: "x", shape: validBoundaryShape()))
    let concat = try graph.addNode(
        ANENode(op: .concatBanned, name: "concat", dtype: .fp16, shape: validBoundaryShape(), inputs: [x], isOutput: true)
    )
    try graph.setGraphOutputs([GraphPort(name: "concat", nodeIndex: concat)])

    let diagnostics = ANEValidationPass().run(on: graph)

    #expect(diagnostics.contains(ANEConstraint(id: 1, severity: .error, message: "concat is banned by the ANE compiler", nodeIndex: concat)))
}

@Test func mismatchedOutputBufferSizesProduceWarning() throws {
    var graph = ANEGraph()
    let x = try graph.addNode(inputNode(name: "x", shape: validBoundaryShape()))
    let a = try graph.addNode(
        ANENode(op: .relu, name: "a", dtype: .fp16, shape: try ANEShape(channels: 256, spatial: 128), inputs: [x], isOutput: true)
    )
    let b = try graph.addNode(
        ANENode(op: .sigmoid, name: "b", dtype: .fp16, shape: try ANEShape(channels: 512, spatial: 128), inputs: [x], isOutput: true)
    )
    try graph.setGraphOutputs([
        GraphPort(name: "a", nodeIndex: a),
        GraphPort(name: "b", nodeIndex: b),
    ])

    let diagnostics = ANEValidationPass().run(on: graph)

    #expect(diagnostics.contains(where: { $0.id == 2 && $0.severity == .warning }))
}

@Test func unsortedOutputsDoNotProduceWarning() throws {
    var graph = ANEGraph()
    let x = try graph.addNode(inputNode(name: "x", shape: validBoundaryShape()))
    let a = try graph.addNode(
        ANENode(op: .relu, name: "a", dtype: .fp16, shape: validBoundaryShape(), inputs: [x], isOutput: true)
    )
    let z = try graph.addNode(
        ANENode(op: .sigmoid, name: "z", dtype: .fp16, shape: validBoundaryShape(), inputs: [x], isOutput: true)
    )
    try graph.setGraphOutputs([
        GraphPort(name: "a", nodeIndex: a),
        GraphPort(name: "z", nodeIndex: z),
    ])
    unsafelySetGraphOutputs(
        in: &graph,
        to: [GraphPort(name: "z", nodeIndex: z), GraphPort(name: "a", nodeIndex: a)]
    )

    let diagnostics = ANEValidationPass().run(on: graph)

    #expect(!diagnostics.contains(where: { $0.id == 3 }))
}

@Test func tinyBoundaryTensorProducesError() throws {
    var graph = ANEGraph()
    let tiny = try graph.addNode(inputNode(name: "tiny", shape: try ANEShape(channels: 4, spatial: 4)))
    try graph.setGraphOutputs([GraphPort(name: "tiny", nodeIndex: tiny)])

    let diagnostics = ANEValidationPass().run(on: graph)

    #expect(diagnostics.contains(where: { $0.id == 4 && $0.severity == .error && $0.nodeIndex == tiny }))
}

@Test func oversizedComputeNodeProducesWarning() throws {
    var graph = ANEGraph()
    let x = try graph.addNode(inputNode(name: "x", shape: try ANEShape(channels: 32768, spatial: 1024)))
    let relu = try graph.addNode(
        ANENode(op: .relu, name: "relu", dtype: .fp16, shape: try ANEShape(channels: 32768, spatial: 1024), inputs: [x], isOutput: true)
    )
    try graph.setGraphOutputs([GraphPort(name: "relu", nodeIndex: relu)])

    let diagnostics = ANEValidationPass().run(on: graph)

    #expect(diagnostics.contains(where: { $0.id == 6 && $0.severity == .warning && $0.nodeIndex == relu }))
}

@Test func nonPowerOfTwoSoftmaxProducesError() throws {
    var graph = ANEGraph()
    let x = try graph.addNode(inputNode(name: "x", shape: try ANEShape(channels: 257, spatial: 96)))
    let softmax = try graph.addNode(
        ANENode(
            op: .softmax,
            name: "softmax",
            dtype: .fp16,
            shape: try ANEShape(channels: 257, spatial: 96),
            inputs: [x],
            attrs: .softmax(axis: 1),
            isOutput: true
        )
    )
    try graph.setGraphOutputs([GraphPort(name: "softmax", nodeIndex: softmax)])

    let diagnostics = ANEValidationPass().run(on: graph)

    #expect(diagnostics.contains(where: { $0.id == 7 && $0.severity == .error && $0.nodeIndex == softmax }))
}

@Test func cleanGraphProducesNoDiagnostics() throws {
    var graph = ANEGraph()
    let x = try graph.addNode(inputNode(name: "x", shape: validBoundaryShape()))
    let relu = try graph.addNode(
        ANENode(op: .relu, name: "relu", dtype: .fp16, shape: validBoundaryShape(), inputs: [x], isOutput: true)
    )
    try graph.setGraphOutputs([GraphPort(name: "relu", nodeIndex: relu)])

    let diagnostics = ANEValidationPass().run(on: graph)

    #expect(diagnostics.isEmpty)
}

@Test func powerOfTwoSoftmaxPassesValidation() throws {
    var graph = ANEGraph()
    let x = try graph.addNode(inputNode(name: "x", shape: validBoundaryShape(channels: 256)))
    let softmax = try graph.addNode(
        ANENode(
            op: .softmax,
            name: "softmax",
            dtype: .fp16,
            shape: validBoundaryShape(channels: 256),
            inputs: [x],
            attrs: .softmax(axis: 1),
            isOutput: true
        )
    )
    try graph.setGraphOutputs([GraphPort(name: "softmax", nodeIndex: softmax)])

    let diagnostics = ANEValidationPass().run(on: graph)

    #expect(!diagnostics.contains(where: { $0.id == 7 }))
}

private func validBoundaryShape(channels: Int = 256, spatial: Int = 128) -> ANEShape {
    try! ANEShape(channels: channels, spatial: spatial)
}

private func inputNode(name: String, shape: ANEShape) -> ANENode {
    ANENode(op: .input, name: name, dtype: .fp16, shape: shape)
}

private struct UnsafeANEGraphLayout {
    var nodes: [ANENode]
    var graphInputs: [GraphPort]
    var graphOutputs: [GraphPort]
}

private func unsafelySetGraphOutputs(in graph: inout ANEGraph, to ports: [GraphPort]) {
    withUnsafeMutablePointer(to: &graph) { pointer in
        pointer.withMemoryRebound(to: UnsafeANEGraphLayout.self, capacity: 1) { rebound in
            rebound.pointee.graphOutputs = ports
        }
    }
}
