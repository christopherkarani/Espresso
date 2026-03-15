import Testing
@testable import ANEGraphIR

@Test func allOpsPresent() {
    #expect(ANEOp.allCases.count == 27)
}

@Test func isComputeForIONodes() {
    #expect(!ANEOp.input.isCompute)
    #expect(!ANEOp.const.isCompute)
}

@Test func isComputeForComputeNodes() {
    let computeOps: [ANEOp] = [
        .conv1x1, .matmul, .add, .sub, .mul, .neg,
        .relu, .tanh, .sigmoid, .softmax, .exp,
        .pow, .sqrt, .rsqrt,
        .reduceMean, .reduceSum, .reduceMax,
        .reshape, .transpose, .cast, .slice, .sliceBySize, .concat, .identity,
        .concatBanned,
    ]
    for op in computeOps {
        #expect(op.isCompute, "Expected \(op) to be compute")
    }
}

@Test func isSourceForIONodes() {
    #expect(ANEOp.input.isSource)
    #expect(ANEOp.const.isSource)
    #expect(!ANEOp.add.isSource)
    #expect(!ANEOp.conv1x1.isSource)
}
