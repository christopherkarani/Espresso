import Testing
@testable import ANEBuilder
import ANEGraphIR

@Test func inputRegistersGraphPort() throws {
    var graph = ANEGraph()
    let shape = try ANEShape(channels: 16, spatial: 32)
    let node = try graph.input("tokens", dtype: .fp16, shape: shape)

    #expect(node == 0)
    #expect(graph.nodes.count == 1)
    #expect(graph.nodes[0].op == .input)
    #expect(graph.graphInputs == [GraphPort(name: "tokens", nodeIndex: 0)])
}

@Test func constScalarDefaultsToFP16() throws {
    var graph = ANEGraph()
    let index = try graph.constScalar("half", 0.5)
    let expectedShape = try ANEShape(channels: 1, spatial: 1)

    #expect(graph.nodes[index].dtype == .fp16)
    #expect(graph.nodes[index].shape == expectedShape)
    #expect(graph.nodes[index].attrs == .scalar(0.5))
}

@Test func constWeightUsesDefaultOffset() throws {
    var graph = ANEGraph()
    let shape = try ANEShape(batch: 32, channels: 16, height: 1, spatial: 1)
    let index = try graph.constWeight("weight", shape: shape, blobPath: "@model_path/w.bin")

    #expect(graph.nodes[index].dtype == .fp16)
    #expect(graph.nodes[index].attrs == .weight(blobPath: "@model_path/w.bin", offset: 64))
}

@Test func constIntInfersShapeFromValueCount() throws {
    var graph = ANEGraph()
    let index = try graph.constInt("perm", values: [0, 1, 3, 2])
    let expectedShape = try ANEShape(channels: 4, spatial: 1)

    #expect(graph.nodes[index].dtype == .int32)
    #expect(graph.nodes[index].shape == expectedShape)
    #expect(graph.nodes[index].attrs == .intTensor([0, 1, 3, 2]))
}

@Test func conv1x1EncodesBiasInputIndex() throws {
    var graph = ANEGraph()
    let x = try graph.input("x", dtype: .fp16, shape: try ANEShape(channels: 16, spatial: 8))
    let w = try graph.constWeight("w", shape: try ANEShape(batch: 32, channels: 16, height: 1, spatial: 1), blobPath: "@w")
    let b = try graph.constWeight("b", shape: try ANEShape(channels: 32, spatial: 1), blobPath: "@b")
    let y = try graph.conv1x1("conv", input: x, weight: w, bias: b, outShape: try ANEShape(channels: 32, spatial: 8))

    #expect(graph.nodes[y].op == .conv1x1)
    #expect(graph.nodes[y].inputs == [x, w, b])
    #expect(graph.nodes[y].attrs == .conv(groups: 1, biasInput: 2))
}

@Test func matmulStoresTransposeFlags() throws {
    var graph = ANEGraph()
    let x = try graph.input("x", dtype: .fp16, shape: try ANEShape(channels: 2, height: 8, spatial: 4))
    let y = try graph.input("y", dtype: .fp16, shape: try ANEShape(channels: 2, height: 4, spatial: 8))
    let out = try graph.matmul(
        "mm",
        x: x,
        y: y,
        transposeX: true,
        transposeY: false,
        outShape: try ANEShape(channels: 2, height: 8, spatial: 8)
    )

    #expect(graph.nodes[out].attrs == .matmul(transposeX: true, transposeY: false))
}

@Test func elementwiseOpsUseLeftInputShapeAndDType() throws {
    var graph = ANEGraph()
    let x = try graph.input("x", dtype: .fp32, shape: try ANEShape(channels: 16, spatial: 8))
    let y = try graph.input("y", dtype: .fp16, shape: try ANEShape(channels: 16, spatial: 8))
    let addNode = try graph.add("sum", x: x, y: y)
    let subNode = try graph.sub("diff", x: x, y: y)
    let mulNode = try graph.mul("prod", x: x, y: y)
    let expectedShape = try ANEShape(channels: 16, spatial: 8)

    for index in [addNode, subNode, mulNode] {
        #expect(graph.nodes[index].dtype == .fp32)
        #expect(graph.nodes[index].shape == expectedShape)
    }
}

@Test func powUsesBaseShape() throws {
    var graph = ANEGraph()
    let base = try graph.input("base", dtype: .fp16, shape: try ANEShape(channels: 1, height: 1, spatial: 8))
    let exp = try graph.constScalar("exp", -0.5)
    let out = try graph.pow("pow", base: base, exp: exp)
    let expectedShape = try ANEShape(channels: 1, height: 1, spatial: 8)

    #expect(graph.nodes[out].shape == expectedShape)
}

@Test func reduceSumKeepDimsCollapsesAxisToOne() throws {
    var graph = ANEGraph()
    let input = try graph.input("x", dtype: .fp16, shape: try ANEShape(channels: 12, height: 3, spatial: 9))
    let out = try graph.reduceSum("rs", input: input, axis: 1, keepDims: true)
    let expectedShape = try ANEShape(channels: 1, height: 3, spatial: 9)

    #expect(graph.nodes[out].shape == expectedShape)
    #expect(graph.nodes[out].attrs == .reduce(axis: 1, keepDims: true))
}

@Test func reduceMaxWithoutKeepDimsCompactsToFourDShape() throws {
    var graph = ANEGraph()
    let input = try graph.input("x", dtype: .fp16, shape: try ANEShape(batch: 2, channels: 6, height: 4, spatial: 8))
    let out = try graph.reduceMax("rm", input: input, axis: 2, keepDims: false)
    let expectedShape = try ANEShape(batch: 2, channels: 6, height: 8, spatial: 1)

    #expect(graph.nodes[out].shape == expectedShape)
    #expect(graph.nodes[out].attrs == .reduce(axis: 2, keepDims: false))
}

@Test func transposePermutesDimensions() throws {
    var graph = ANEGraph()
    let input = try graph.input("x", dtype: .fp16, shape: try ANEShape(batch: 1, channels: 4, height: 8, spatial: 16))
    let out = try graph.transpose("t", input: input, perm: [0, 1, 3, 2])
    let expectedShape = try ANEShape(batch: 1, channels: 4, height: 16, spatial: 8)

    #expect(graph.nodes[out].shape == expectedShape)
    #expect(graph.nodes[out].attrs == .transpose(perm: [0, 1, 3, 2]))
}

@Test func castPreservesShapeAndChangesDType() throws {
    var graph = ANEGraph()
    let input = try graph.input("x", dtype: .fp32, shape: try ANEShape(channels: 8, spatial: 8))
    let out = try graph.cast("cast", input: input, to: .fp16)
    let expectedShape = try ANEShape(channels: 8, spatial: 8)

    #expect(graph.nodes[out].shape == expectedShape)
    #expect(graph.nodes[out].dtype == .fp16)
    #expect(graph.nodes[out].attrs == .cast(target: .fp16))
}

@Test func outputMarksNodeAndSortsGraphOutputs() throws {
    var graph = ANEGraph()
    let a = try graph.input("a", dtype: .fp16, shape: try ANEShape(channels: 2, spatial: 2))
    let b = try graph.relu("b_relu", input: a)
    let c = try graph.relu("c_relu", input: a)

    _ = try graph.output(c, name: "z_out")
    _ = try graph.output(b, name: "a_out")

    #expect(graph.nodes[b].isOutput)
    #expect(graph.nodes[c].isOutput)
    #expect(graph.graphOutputs.map(\.name) == ["a_out", "z_out"])
}
