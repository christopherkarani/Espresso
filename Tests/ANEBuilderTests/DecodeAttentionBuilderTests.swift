import Testing
@testable import ANEBuilder
import ANEGraphIR

@Test func decodeGQAAttentionUsesHeadRepeatSoftmaxAndMask() throws {
    var graph = ANEGraph()
    let q = try graph.input("q", dtype: .fp16, shape: try ANEShape(channels: 12, spatial: 1))
    let k = try graph.input("k", dtype: .fp16, shape: try ANEShape(channels: 4, spatial: 8))
    let v = try graph.input("v", dtype: .fp16, shape: try ANEShape(channels: 4, spatial: 8))
    let mask = try graph.input("mask", dtype: .fp16, shape: try ANEShape(channels: 1, spatial: 8))

    let out = try graph.decodeGQAAttention(
        "attn",
        q: q,
        kCache: k,
        vCache: v,
        mask: mask,
        nHeads: 3,
        nKVHeads: 1,
        headDim: 4,
        maxSeq: 8
    )

    #expect(graph.nodes[out].name == "attn_out")
    #expect(graph.nodes[out].shape == (try ANEShape(channels: 12, spatial: 1)))
    #expect(graph.nodes.contains { $0.name == "attn_k_h0" && $0.op == .sliceBySize })
    #expect(graph.nodes.contains { $0.name == "attn_k_g0" && $0.op == .concat })
    #expect(graph.nodes.contains { $0.name == "attn_softmax" && $0.op == .softmax })
    let softmax = try #require(graph.nodes.first { $0.name == "attn_softmax" })
    #expect(softmax.attrs == .softmax(axis: 3))
    let scores = try #require(graph.nodes.first { $0.name == "attn_scores" })
    #expect(scores.shape == (try ANEShape(batch: 1, channels: 3, height: 1, spatial: 8)))
}

@Test func ropeHalfSplitEmitsCrossPairMulAndConcat() throws {
    var graph = ANEGraph()
    let x = try graph.input("x", dtype: .fp16, shape: try ANEShape(channels: 8, spatial: 1))
    let cos = try graph.input("cos", dtype: .fp16, shape: try ANEShape(channels: 4, spatial: 1))
    let sin = try graph.input("sin", dtype: .fp16, shape: try ANEShape(channels: 4, spatial: 1))

    let out = try graph.ropeHalfSplit(
        "rope",
        input: x,
        cos: cos,
        sin: sin,
        nHeads: 2,
        headDim: 4,
        spatial: 1
    )

    #expect(graph.nodes[out].shape == (try ANEShape(channels: 8, spatial: 1)))
    #expect(graph.nodes.contains { $0.name == "rope_x0" && $0.op == .sliceBySize })
    #expect(graph.nodes.contains { $0.name == "rope_x1" && $0.op == .sliceBySize })
    #expect(graph.nodes.contains { $0.name == "rope_y0" && $0.op == .sub })
    #expect(graph.nodes.contains { $0.name == "rope_y1" && $0.op == .add })
    #expect(graph.nodes.contains { $0.name == "rope_cat" && $0.op == .concat })
}

@Test func scatterCurrentIntoCacheIsOuterProductAdd() throws {
    var graph = ANEGraph()
    let cache = try graph.input("cache", dtype: .fp16, shape: try ANEShape(channels: 4, spatial: 8))
    let current = try graph.input("cur", dtype: .fp16, shape: try ANEShape(channels: 4, spatial: 1))
    let pos = try graph.input("pos", dtype: .fp16, shape: try ANEShape(channels: 1, spatial: 8))

    let out = try graph.scatterCurrentIntoCache(
        "sc",
        cache: cache,
        current: current,
        posMask: pos,
        channels: 4,
        maxSeq: 8
    )

    #expect(graph.nodes[out].shape == (try ANEShape(channels: 4, spatial: 8)))
    #expect(graph.nodes.contains { $0.name == "sc_outer" && $0.op == .matmul })
    #expect(graph.nodes.contains { $0.name == "sc_out" && $0.op == .add })
}
