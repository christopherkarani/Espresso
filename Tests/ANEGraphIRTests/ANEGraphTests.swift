import Testing
@testable import ANEGraphIR

@Test func emptyGraph() {
    let g = ANEGraph()
    #expect(g.nodes.isEmpty)
    #expect(g.graphInputs.isEmpty)
    #expect(g.graphOutputs.isEmpty)
    #expect(g.liveNodeCount == 0)
}

@Test func addNodeReturnsSequentialIndices() {
    var g = ANEGraph()
    let i0 = g.addNode(ANENode(op: .input, name: "x", dtype: .fp16,
                                shape: ANEShape(channels: 4, spatial: 4)))
    let i1 = g.addNode(ANENode(op: .relu, name: "y", dtype: .fp16,
                                shape: ANEShape(channels: 4, spatial: 4), inputs: [i0]))
    #expect(i0 == 0)
    #expect(i1 == 1)
    #expect(g.nodes.count == 2)
    #expect(g.liveNodeCount == 2)
}

@Test func topoSortEmptyGraph() {
    let g = ANEGraph()
    let order = g.topoSort()
    #expect(order != nil)
    #expect(order!.isEmpty)
}

@Test func topoSortLinearChain() {
    // x → relu(x) → output
    var g = ANEGraph()
    let x = g.addNode(ANENode(op: .input, name: "x", dtype: .fp16,
                               shape: ANEShape(channels: 4, spatial: 4)))
    let r = g.addNode(ANENode(op: .relu, name: "r", dtype: .fp16,
                               shape: ANEShape(channels: 4, spatial: 4),
                               inputs: [x], isOutput: true))
    g.graphOutputs = [GraphPort(name: "r", nodeIndex: r)]

    let order = g.topoSort()
    #expect(order != nil)
    #expect(order! == [x, r])
}

@Test func topoSortSelfReference() {
    // x → add(x, x) → output (same node referenced twice)
    var g = ANEGraph()
    let x = g.addNode(ANENode(op: .input, name: "x", dtype: .fp16,
                               shape: ANEShape(channels: 4, spatial: 4)))
    let sum = g.addNode(ANENode(op: .add, name: "sum", dtype: .fp16,
                                 shape: ANEShape(channels: 4, spatial: 4),
                                 inputs: [x, x], isOutput: true))
    g.graphOutputs = [GraphPort(name: "sum", nodeIndex: sum)]

    let order = g.topoSort()
    #expect(order != nil)
    #expect(order!.count == 2)
    #expect(order![0] == x)
    #expect(order![1] == sum)
}

@Test func topoSortDiamondGraph() {
    // x → a(relu), x → b(sigmoid), a + b → c(add)
    var g = ANEGraph()
    let x = g.addNode(ANENode(op: .input, name: "x", dtype: .fp16,
                               shape: ANEShape(channels: 4, spatial: 4)))
    let a = g.addNode(ANENode(op: .relu, name: "a", dtype: .fp16,
                               shape: ANEShape(channels: 4, spatial: 4), inputs: [x]))
    let b = g.addNode(ANENode(op: .sigmoid, name: "b", dtype: .fp16,
                               shape: ANEShape(channels: 4, spatial: 4), inputs: [x]))
    let c = g.addNode(ANENode(op: .add, name: "c", dtype: .fp16,
                               shape: ANEShape(channels: 4, spatial: 4),
                               inputs: [a, b], isOutput: true))
    g.graphOutputs = [GraphPort(name: "c", nodeIndex: c)]

    let order = g.topoSort()
    #expect(order != nil)
    #expect(order!.count == 4)
    #expect(order![0] == x)
    #expect(order!.last == c)
    let aPos = order!.firstIndex(of: a)!
    let bPos = order!.firstIndex(of: b)!
    let cPos = order!.firstIndex(of: c)!
    #expect(aPos < cPos)
    #expect(bPos < cPos)
}

@Test func topoSortSkipsDeadNodes() {
    var g = ANEGraph()
    let x = g.addNode(ANENode(op: .input, name: "x", dtype: .fp16,
                               shape: ANEShape(channels: 4, spatial: 4)))
    var deadNode = ANENode(op: .relu, name: "dead", dtype: .fp16,
                            shape: ANEShape(channels: 4, spatial: 4), inputs: [x])
    deadNode.isLive = false
    let _ = g.addNode(deadNode)

    let out = g.addNode(ANENode(op: .sigmoid, name: "out", dtype: .fp16,
                                 shape: ANEShape(channels: 4, spatial: 4),
                                 inputs: [x], isOutput: true))
    g.graphOutputs = [GraphPort(name: "out", nodeIndex: out)]

    let order = g.topoSort()
    #expect(order != nil)
    #expect(order!.count == 2)
    #expect(!order!.contains(1))
}

@Test func topoSortDetectsCycle() {
    // Manually create a cycle: a references b, b references a
    var g = ANEGraph()
    let a = g.addNode(ANENode(op: .relu, name: "a", dtype: .fp16,
                               shape: ANEShape(channels: 4, spatial: 4),
                               inputs: [1]))
    let b = g.addNode(ANENode(op: .relu, name: "b", dtype: .fp16,
                               shape: ANEShape(channels: 4, spatial: 4),
                               inputs: [a], isOutput: true))
    g.graphOutputs = [GraphPort(name: "b", nodeIndex: b)]

    let order = g.topoSort()
    #expect(order == nil)
}

@Test func liveNodeCountAfterKilling() {
    var g = ANEGraph()
    let _ = g.addNode(ANENode(op: .input, name: "x", dtype: .fp16,
                               shape: ANEShape(channels: 4, spatial: 4)))
    let _ = g.addNode(ANENode(op: .relu, name: "y", dtype: .fp16,
                               shape: ANEShape(channels: 4, spatial: 4), inputs: [0]))

    #expect(g.liveNodeCount == 2)
    g.nodes[1].isLive = false
    #expect(g.liveNodeCount == 1)
}

@Test func graphPortEquality() {
    let a = GraphPort(name: "x", nodeIndex: 0)
    let b = GraphPort(name: "x", nodeIndex: 0)
    let c = GraphPort(name: "y", nodeIndex: 0)
    #expect(a == b)
    #expect(a != c)
}

@Test func nodeEquality() {
    let a = ANENode(op: .relu, name: "r", dtype: .fp16,
                     shape: ANEShape(channels: 4, spatial: 4), inputs: [0])
    let b = ANENode(op: .relu, name: "r", dtype: .fp16,
                     shape: ANEShape(channels: 4, spatial: 4), inputs: [0])
    #expect(a == b)
}

@Test func attrsExhaustiveMatch() {
    let cases: [ANEAttrs] = [
        .none,
        .conv(groups: 1, biasInput: nil),
        .matmul(transposeX: false, transposeY: true),
        .transpose(perm: [0, 1, 3, 2]),
        .reduce(axis: 1, keepDims: true),
        .softmax(axis: -1),
        .cast(target: .fp32),
        .slice(begin: [0, 0, 0, 0], end: [1, 768, 1, 256]),
        .scalar(0.5),
        .weight(blobPath: "@model_path/w.bin", offset: 64),
        .intTensor([1, 1]),
        .boolValue(true),
    ]
    #expect(cases.count == 12)

    for attr in cases {
        switch attr {
        case .none: break
        case .conv: break
        case .matmul: break
        case .transpose: break
        case .reduce: break
        case .softmax: break
        case .cast: break
        case .slice: break
        case .scalar: break
        case .weight: break
        case .intTensor: break
        case .boolValue: break
        }
    }
}

@Test func dtypeByteWidth() {
    #expect(ANEDType.fp16.byteWidth == 2)
    #expect(ANEDType.fp32.byteWidth == 4)
    #expect(ANEDType.int32.byteWidth == 4)
    #expect(ANEDType.bool.byteWidth == 1)
}

@Test func dtypeDescription() {
    #expect(ANEDType.fp16.description == "fp16")
    #expect(ANEDType.fp32.description == "fp32")
    #expect(ANEDType.int32.description == "int32")
    #expect(ANEDType.bool.description == "bool")
}
