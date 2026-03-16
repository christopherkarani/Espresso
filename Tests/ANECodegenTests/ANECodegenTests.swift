import Testing
import ANECodegen
import ANEGraphIR

@Test func programStructureForAddKernel() throws {
    let graph = try makeAddGraph()
    let mil = ANECodegen.emit(graph)

    #expect(mil.contains("program(1.3)"))
    #expect(mil.contains("func main<ios18>(tensor<fp16, [1, 4, 1, 8]> lhs, tensor<fp16, [1, 4, 1, 8]> rhs)"))
    #expect(mil.contains("tensor<fp16, [1, 4, 1, 8]> output = add(x=lhs, y=rhs)[name=string(\"output\")];"))
    #expect(mil.contains("} -> (output);"))
}

@Test func convEmissionIncludesBlobfileAndInlineParams() throws {
    var graph = ANEGraph()
    let x = try graph.addNode(inputNode(name: "x"))
    let w = try graph.addNode(weightNode(name: "W", shape: shape(channels: 4, spatial: 1), path: "@model_path/w.bin"))
    let conv = try graph.addNode(
        ANENode(
            op: .conv1x1,
            name: "convOut",
            dtype: .fp16,
            shape: shape(channels: 4, spatial: 8),
            inputs: [x, w],
            attrs: .conv(groups: 1, biasInput: nil),
            isOutput: true
        )
    )
    try graph.setGraphInputs([GraphPort(name: "x", nodeIndex: x)])
    try graph.setGraphOutputs([GraphPort(name: "convOut", nodeIndex: conv)])

    let mil = ANECodegen.emit(graph)
    #expect(mil.contains("BLOBFILE(path=string(\"@model_path/w.bin\"), offset=uint64(64))"))
    #expect(mil.contains("convOut_pt"))
    #expect(mil.contains("convOut_st"))
    #expect(mil.contains("convOut_pd"))
    #expect(mil.contains("convOut_dl"))
    #expect(mil.contains("convOut_gr"))
    #expect(mil.contains("conv(dilations=convOut_dl, groups=convOut_gr, pad=convOut_pd, pad_type=convOut_pt, strides=convOut_st, weight=W, x=x)"))
}

@Test func matmulEmissionUsesNamedTransposeConsts() throws {
    var graph = ANEGraph()
    let x = try graph.addNode(inputNode(name: "x"))
    let y = try graph.addNode(inputNode(name: "y"))
    let out = try graph.addNode(
        ANENode(
            op: .matmul,
            name: "mm",
            dtype: .fp16,
            shape: shape(channels: 4, spatial: 8),
            inputs: [x, y],
            attrs: .matmul(transposeX: false, transposeY: true),
            isOutput: true
        )
    )
    try graph.setGraphInputs([GraphPort(name: "x", nodeIndex: x), GraphPort(name: "y", nodeIndex: y)])
    try graph.setGraphOutputs([GraphPort(name: "mm", nodeIndex: out)])

    let mil = ANECodegen.emit(graph)
    #expect(mil.contains("bool mm_tx = const()[name=string(\"mm_tx\"), val=bool(false)];"))
    #expect(mil.contains("bool mm_ty = const()[name=string(\"mm_ty\"), val=bool(true)];"))
    #expect(mil.contains("matmul(transpose_x=mm_tx, transpose_y=mm_ty, x=x, y=y)"))
}

@Test func softmaxUsesAxisConst() throws {
    var graph = ANEGraph()
    let x = try graph.addNode(inputNode(name: "x"))
    let out = try graph.addNode(
        ANENode(
            op: .softmax,
            name: "sm",
            dtype: .fp16,
            shape: shape(channels: 4, spatial: 8),
            inputs: [x],
            attrs: .softmax(axis: 3),
            isOutput: true
        )
    )
    try graph.setGraphInputs([GraphPort(name: "x", nodeIndex: x)])
    try graph.setGraphOutputs([GraphPort(name: "sm", nodeIndex: out)])

    let mil = ANECodegen.emit(graph)
    #expect(mil.contains("int32 sm_axis = const()[name=string(\"sm_axis\"), val=int32(3)];"))
    #expect(mil.contains("softmax(axis=sm_axis, x=x)"))
}

@Test func reduceOpsUseAxesTensorAndKeepDimsBool() throws {
    var graph = ANEGraph()
    let x = try graph.addNode(inputNode(name: "x"))
    let sum = try graph.addNode(
        ANENode(
            op: .reduceSum,
            name: "sum",
            dtype: .fp16,
            shape: shape(channels: 1, spatial: 8),
            inputs: [x],
            attrs: .reduce(axis: 1, keepDims: true)
        )
    )
    let mean = try graph.addNode(
        ANENode(
            op: .reduceMean,
            name: "mean",
            dtype: .fp16,
            shape: shape(channels: 1, spatial: 8),
            inputs: [x],
            attrs: .reduce(axis: 1, keepDims: true)
        )
    )
    let max = try graph.addNode(
        ANENode(
            op: .reduceMax,
            name: "max",
            dtype: .fp16,
            shape: shape(channels: 1, spatial: 8),
            inputs: [x],
            attrs: .reduce(axis: 1, keepDims: true),
            isOutput: true
        )
    )
    try graph.setGraphInputs([GraphPort(name: "x", nodeIndex: x)])
    try graph.setGraphOutputs([GraphPort(name: "max", nodeIndex: max)])

    let mil = ANECodegen.emit(graph)
    #expect(mil.contains("tensor<int32, [1]> sum_axes"))
    #expect(mil.contains("bool sum_keep_dims"))
    #expect(mil.contains("reduce_sum(x=x, axes=sum_axes, keep_dims=sum_keep_dims)"))
    #expect(mil.contains("reduce_mean(x=x, axes=mean_axes, keep_dims=mean_keep_dims)"))
    #expect(mil.contains("reduce_max(x=x, axes=max_axes, keep_dims=max_keep_dims)"))
    _ = mean
    _ = sum
}

@Test func castEmissionUsesStringConst() throws {
    var graph = ANEGraph()
    let x = try graph.addNode(ANENode(op: .input, name: "x", dtype: .fp32, shape: shape(channels: 4, spatial: 8)))
    let cast = try graph.addNode(
        ANENode(
            op: .cast,
            name: "x16",
            dtype: .fp16,
            shape: shape(channels: 4, spatial: 8),
            inputs: [x],
            attrs: .cast(target: .fp16),
            isOutput: true
        )
    )
    try graph.setGraphInputs([GraphPort(name: "x", nodeIndex: x)])
    try graph.setGraphOutputs([GraphPort(name: "x16", nodeIndex: cast)])

    let mil = ANECodegen.emit(graph)
    #expect(mil.contains("string x16_dtype = const()[name=string(\"x16_dtype\"), val=string(\"fp16\")];"))
    #expect(mil.contains("tensor<fp16, [1, 4, 1, 8]> x16 = cast(dtype=x16_dtype, x=x)[name=string(\"x16\")];"))
}

@Test func reshapeAndTransposeEmissionUseShapeHelpers() throws {
    var graph = ANEGraph()
    let x = try graph.addNode(inputNode(name: "x"))
    let targetShape = try graph.addNode(
        ANENode(
            op: .const,
            name: "targetShape",
            dtype: .int32,
            shape: shape(batch: 1, channels: 4, height: 1, spatial: 1),
            attrs: .intTensor([1, 2, 2, 8])
        )
    )
    let reshaped = try graph.addNode(
        ANENode(
            op: .reshape,
            name: "reshaped",
            dtype: .fp16,
            shape: shape(batch: 1, channels: 2, height: 2, spatial: 8),
            inputs: [x, targetShape]
        )
    )
    let transposed = try graph.addNode(
        ANENode(
            op: .transpose,
            name: "transposed",
            dtype: .fp16,
            shape: shape(batch: 1, channels: 2, height: 8, spatial: 2),
            inputs: [reshaped],
            attrs: .transpose(perm: [0, 1, 3, 2]),
            isOutput: true
        )
    )
    try graph.setGraphInputs([GraphPort(name: "x", nodeIndex: x)])
    try graph.setGraphOutputs([GraphPort(name: "transposed", nodeIndex: transposed)])

    let mil = ANECodegen.emit(graph)
    #expect(mil.contains("tensor<int32, [4]> targetShape = const()"))
    #expect(mil.contains("reshape(shape=targetShape, x=x)"))
    #expect(mil.contains("tensor<int32, [4]> transposed_perm = const()"))
    #expect(mil.contains("transpose(perm=transposed_perm, x=reshaped)"))
}

@Test func returnTupleIsAlphabeticallySortedAndUsesOutputAliases() throws {
    var graph = ANEGraph()
    let x = try graph.addNode(inputNode(name: "x"))
    let y = try graph.addNode(inputNode(name: "y"))
    let zeta = try graph.addNode(
        ANENode(
            op: .add,
            name: "zetaNode",
            dtype: .fp16,
            shape: shape(channels: 4, spatial: 8),
            inputs: [x, y]
        )
    )
    let alpha = try graph.addNode(
        ANENode(
            op: .sub,
            name: "alphaNode",
            dtype: .fp16,
            shape: shape(channels: 4, spatial: 8),
            inputs: [x, y]
        )
    )
    try graph.setGraphInputs([GraphPort(name: "x", nodeIndex: x), GraphPort(name: "y", nodeIndex: y)])
    try graph.setGraphOutputs([
        GraphPort(name: "alpha", nodeIndex: alpha),
        GraphPort(name: "zeta", nodeIndex: zeta),
    ])

    let mil = ANECodegen.emit(graph)
    #expect(mil.contains("tensor<fp16, [1, 4, 1, 8]> alpha = identity(x=alphaNode)[name=string(\"alpha\")];"))
    #expect(mil.contains("tensor<fp16, [1, 4, 1, 8]> zeta = identity(x=zetaNode)[name=string(\"zeta\")];"))
    #expect(mil.contains("} -> (alpha, zeta);"))
}

@Test func deadNodesAreSkippedIncludingDeadConcatBanned() throws {
    var graph = try makeAddGraph()
    let dead = ANENode(
        op: .concatBanned,
        name: "deadConcat",
        dtype: .fp16,
        shape: shape(channels: 4, spatial: 8),
        inputs: [0, 1],
        isLive: false
    )
    _ = try graph.addNode(dead)

    let mil = ANECodegen.emit(graph)
    #expect(!mil.contains("deadConcat"))
    #expect(!mil.contains("concat("))
}

@Test func scalarConstUsesDecimalFormattingForIntegers() throws {
    var graph = ANEGraph()
    let x = try graph.addNode(inputNode(name: "x"))
    let one = try graph.addNode(
        ANENode(
            op: .const,
            name: "one",
            dtype: .fp16,
            shape: shape(channels: 1, spatial: 1),
            attrs: .scalar(1.0)
        )
    )
    let out = try graph.addNode(
        ANENode(
            op: .add,
            name: "out",
            dtype: .fp16,
            shape: shape(channels: 4, spatial: 8),
            inputs: [x, one],
            isOutput: true
        )
    )
    try graph.setGraphInputs([GraphPort(name: "x", nodeIndex: x)])
    try graph.setGraphOutputs([GraphPort(name: "out", nodeIndex: out)])

    let mil = ANECodegen.emit(graph)
    #expect(mil.contains("fp16 one = const()[name=string(\"one\"), val=fp16(1.0)];"))
}

@Test func sliceEmissionUsesBeginEndStrideAndMasks() throws {
    var graph = ANEGraph()
    let x = try graph.addNode(inputNode(name: "x"))
    let slice = try graph.addNode(
        ANENode(
            op: .slice,
            name: "sliceOut",
            dtype: .fp16,
            shape: shape(channels: 4, spatial: 4),
            inputs: [x],
            attrs: .slice(begin: [0, 0, 0, 0], end: [1, 4, 1, 4]),
            isOutput: true
        )
    )
    try graph.setGraphInputs([GraphPort(name: "x", nodeIndex: x)])
    try graph.setGraphOutputs([GraphPort(name: "sliceOut", nodeIndex: slice)])

    let mil = ANECodegen.emit(graph)
    #expect(mil.contains("sliceOut_begin"))
    #expect(mil.contains("sliceOut_end"))
    #expect(mil.contains("sliceOut_stride"))
    #expect(mil.contains("sliceOut_begin_mask"))
    #expect(mil.contains("sliceOut_end_mask"))
    #expect(mil.contains("sliceOut_squeeze_mask"))
    #expect(mil.contains("slice_by_index(begin=sliceOut_begin"))
}

@Test func customFunctionNameIsUsed() throws {
    let graph = try makeAddGraph()
    let mil = ANECodegen.emit(graph, functionName: "decode")
    #expect(mil.contains("func decode<ios18>("))
}

@Test func unaryOpsAreEmitted() throws {
    var graph = ANEGraph()
    let x = try graph.addNode(inputNode(name: "x"))
    let neg = try graph.addNode(ANENode(op: .neg, name: "negOut", dtype: .fp16, shape: shape(channels: 4, spatial: 8), inputs: [x]))
    let relu = try graph.addNode(ANENode(op: .relu, name: "reluOut", dtype: .fp16, shape: shape(channels: 4, spatial: 8), inputs: [neg]))
    let tanh = try graph.addNode(ANENode(op: .tanh, name: "tanhOut", dtype: .fp16, shape: shape(channels: 4, spatial: 8), inputs: [relu]))
    let sigmoid = try graph.addNode(ANENode(op: .sigmoid, name: "sigmoidOut", dtype: .fp16, shape: shape(channels: 4, spatial: 8), inputs: [tanh]))
    let exp = try graph.addNode(ANENode(op: .exp, name: "expOut", dtype: .fp16, shape: shape(channels: 4, spatial: 8), inputs: [sigmoid]))
    let sqrt = try graph.addNode(ANENode(op: .sqrt, name: "sqrtOut", dtype: .fp16, shape: shape(channels: 4, spatial: 8), inputs: [exp]))
    let rsqrt = try graph.addNode(ANENode(op: .rsqrt, name: "rsqrtOut", dtype: .fp16, shape: shape(channels: 4, spatial: 8), inputs: [sqrt]))
    let identity = try graph.addNode(ANENode(op: .identity, name: "identityOut", dtype: .fp16, shape: shape(channels: 4, spatial: 8), inputs: [rsqrt], isOutput: true))
    try graph.setGraphInputs([GraphPort(name: "x", nodeIndex: x)])
    try graph.setGraphOutputs([GraphPort(name: "identityOut", nodeIndex: identity)])

    let mil = ANECodegen.emit(graph)
    #expect(mil.contains("neg(x=x)"))
    #expect(mil.contains("relu(x=negOut)"))
    #expect(mil.contains("tanh(x=reluOut)"))
    #expect(mil.contains("sigmoid(x=tanhOut)"))
    #expect(mil.contains("exp(x=sigmoidOut)"))
    #expect(mil.contains("sqrt(x=expOut)"))
    #expect(mil.contains("rsqrt(x=sqrtOut)"))
    #expect(mil.contains("identity(x=rsqrtOut)"))
}

@Test func binaryOpsAndScalarConstsAreEmitted() throws {
    var graph = ANEGraph()
    let x = try graph.addNode(inputNode(name: "x"))
    let y = try graph.addNode(inputNode(name: "y"))
    let scalar = try graph.addNode(
        ANENode(
            op: .const,
            name: "scalar",
            dtype: .fp16,
            shape: shape(channels: 1, spatial: 1),
            attrs: .scalar(1.25)
        )
    )
    let add = try graph.addNode(ANENode(op: .add, name: "addOut", dtype: .fp16, shape: shape(channels: 4, spatial: 8), inputs: [x, y]))
    let sub = try graph.addNode(ANENode(op: .sub, name: "subOut", dtype: .fp16, shape: shape(channels: 4, spatial: 8), inputs: [add, y]))
    let mul = try graph.addNode(ANENode(op: .mul, name: "mulOut", dtype: .fp16, shape: shape(channels: 4, spatial: 8), inputs: [sub, scalar]))
    let pow = try graph.addNode(ANENode(op: .pow, name: "powOut", dtype: .fp16, shape: shape(channels: 4, spatial: 8), inputs: [mul, scalar], isOutput: true))
    try graph.setGraphInputs([GraphPort(name: "x", nodeIndex: x), GraphPort(name: "y", nodeIndex: y)])
    try graph.setGraphOutputs([GraphPort(name: "powOut", nodeIndex: pow)])

    let mil = ANECodegen.emit(graph)
    #expect(mil.contains("val=fp16(1.25)"))
    #expect(mil.contains("add(x=x, y=y)"))
    #expect(mil.contains("sub(x=addOut, y=y)"))
    #expect(mil.contains("mul(x=subOut, y=scalar)"))
    #expect(mil.contains("pow(x=mulOut, y=scalar)"))
}

@Test func convBiasIsLoweredToConvPlusAdd() throws {
    var graph = ANEGraph()
    let x = try graph.addNode(inputNode(name: "x"))
    let w = try graph.addNode(weightNode(name: "W", shape: shape(channels: 4, spatial: 1), path: "@model_path/wb.bin"))
    let bias = try graph.addNode(
        ANENode(
            op: .const,
            name: "bias",
            dtype: .fp16,
            shape: shape(channels: 4, spatial: 8),
            attrs: .scalar(0.5)
        )
    )
    let out = try graph.addNode(
        ANENode(
            op: .conv1x1,
            name: "biased",
            dtype: .fp16,
            shape: shape(channels: 4, spatial: 8),
            inputs: [x, w, bias],
            attrs: .conv(groups: 1, biasInput: 2),
            isOutput: true
        )
    )
    try graph.setGraphInputs([GraphPort(name: "x", nodeIndex: x)])
    try graph.setGraphOutputs([GraphPort(name: "biased", nodeIndex: out)])

    let mil = ANECodegen.emit(graph)
    #expect(mil.contains("tensor<fp16, [1, 4, 1, 8]> biased_core = conv("))
    #expect(mil.contains("tensor<fp16, [1, 4, 1, 8]> biased = add(x=biased_core, y=bias)[name=string(\"biased\")];"))
}

private func makeAddGraph() throws -> ANEGraph {
    var graph = ANEGraph()
    let lhs = try graph.addNode(inputNode(name: "lhs"))
    let rhs = try graph.addNode(inputNode(name: "rhs"))
    let output = try graph.addNode(
        ANENode(
            op: .add,
            name: "output",
            dtype: .fp16,
            shape: shape(channels: 4, spatial: 8),
            inputs: [lhs, rhs],
            isOutput: true
        )
    )
    try graph.setGraphInputs([GraphPort(name: "lhs", nodeIndex: lhs), GraphPort(name: "rhs", nodeIndex: rhs)])
    try graph.setGraphOutputs([GraphPort(name: "output", nodeIndex: output)])
    return graph
}

private func inputNode(name: String) -> ANENode {
    ANENode(op: .input, name: name, dtype: .fp16, shape: shape(channels: 4, spatial: 8))
}

private func weightNode(name: String, shape: ANEShape, path: String) -> ANENode {
    ANENode(op: .const, name: name, dtype: .fp16, shape: shape, attrs: .weight(blobPath: path, offset: 64))
}

private func shape(batch: Int = 1, channels: Int, height: Int = 1, spatial: Int) -> ANEShape {
    try! ANEShape(batch: batch, channels: channels, height: height, spatial: spatial)
}
