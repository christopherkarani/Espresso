import Testing
import ANECodegen

@Test func identicalProgramsHaveNoDiff() {
    let mil = sampleMIL(op: "add")
    #expect(MILDiff.textDiff(mil, mil).isEmpty)
}

@Test func whitespaceDifferencesAreIgnored() {
    let lhs = """
    program(1.3)
    {
        tensor<fp16, [1, 4, 1, 8]> y = add(x=x, y=y)[name=string("y")];
    }
    """
    let rhs = """
    program(1.3)
    {
    tensor<fp16,[1,4,1,8]> y=add(x=x,y=y)[name=string("y")];
    }
    """

    #expect(MILDiff.textDiff(lhs, rhs).isEmpty)
}

@Test func textDiffDetectsChangedLines() {
    let lhs = sampleMIL(op: "add")
    let rhs = sampleMIL(op: "mul")
    let diff = MILDiff.textDiff(lhs, rhs)

    #expect(diff.count == 2)
    #expect(diff[0].contains("=add("))
    #expect(diff[1].contains("=mul("))
}

@Test func structuralEquivalenceMatchesSameOpSequence() {
    let lhs = sampleMIL(op: "add")
    let rhs = sampleMIL(op: "add").replacingOccurrences(of: "y = add", with: "z = add")
    #expect(MILDiff.structuralEquiv(lhs, rhs))
}

@Test func structuralEquivalenceRejectsDifferentOps() {
    #expect(!MILDiff.structuralEquiv(sampleMIL(op: "add"), sampleMIL(op: "sub")))
}

@Test func structuralEquivalenceIgnoresConstLines() {
    let lhs = """
    fp16 scalar = const()[name=string("scalar"), val=fp16(1.0)];
    tensor<fp16, [1, 4, 1, 8]> y = add(x=x, y=scalar)[name=string("y")];
    """
    let rhs = """
    fp16 scalar = const()[name=string("scalar"), val=fp16(2.0)];
    tensor<fp16, [1, 4, 1, 8]> y = add(x=x, y=scalar)[name=string("y")];
    """

    #expect(MILDiff.structuralEquiv(lhs, rhs))
}

@Test func reorderedOpsAreNotStructurallyEquivalent() {
    let lhs = """
    tensor<fp16, [1, 4, 1, 8]> a = add(x=x, y=y)[name=string("a")];
    tensor<fp16, [1, 4, 1, 8]> b = mul(x=a, y=y)[name=string("b")];
    """
    let rhs = """
    tensor<fp16, [1, 4, 1, 8]> a = mul(x=x, y=y)[name=string("a")];
    tensor<fp16, [1, 4, 1, 8]> b = add(x=a, y=y)[name=string("b")];
    """

    #expect(!MILDiff.structuralEquiv(lhs, rhs))
}

@Test func extractOpsFindsAllNonConstOps() {
    let mil = """
    fp16 scalar = const()[name=string("scalar"), val=fp16(1.0)];
    tensor<fp16, [1, 4, 1, 8]> c = conv(dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, weight=W, x=x)[name=string("c")];
    tensor<fp16, [1, 4, 1, 8]> m = matmul(transpose_x=tx, transpose_y=ty, x=c, y=y)[name=string("m")];
    tensor<fp16, [1, 1, 1, 8]> r = reduce_sum(x=m, axes=ax, keep_dims=kd)[name=string("r")];
    tensor<fp16, [1, 4, 1, 4]> s = slice_by_index(begin=b, begin_mask=bm, end=e, end_mask=em, squeeze_mask=sm, stride=st, x=m)[name=string("s")];
    """

    #expect(MILDiff.extractOps(mil) == ["conv", "matmul", "reduce_sum", "slice_by_index"])
}

private func sampleMIL(op: String) -> String {
    """
    program(1.3)
    {
        tensor<fp16, [1, 4, 1, 8]> y = \(op)(x=x, y=y)[name=string("y")];
    }
    """
}
