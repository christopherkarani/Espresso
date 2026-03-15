import Testing
@testable import ANEBuilder
import ANEGraphIR

@Test func causalAttentionBuildsFullyDecomposedPath() throws {
    var graph = ANEGraph()
    let q = try graph.input("q", dtype: .fp16, shape: try ANEShape(channels: 16, spatial: 8))
    let k = try graph.input("k", dtype: .fp16, shape: try ANEShape(channels: 16, spatial: 8))
    let v = try graph.input("v", dtype: .fp16, shape: try ANEShape(channels: 16, spatial: 8))

    let out = try graph.causalAttention(
        "attn",
        q: q,
        k: k,
        v: v,
        nHeads: 4,
        headDim: 4,
        spatial: 8,
        maskPath: "@mask"
    )

    #expect(graph.nodes[out].name == "attn_out")
    #expect(graph.nodes.contains { $0.name == "attn_q_reshape" && $0.op == .reshape })
    #expect(graph.nodes.contains { $0.name == "attn_q_transpose" && $0.op == .transpose })
    #expect(graph.nodes.contains { $0.name == "attn_scores" && $0.op == .matmul })
    #expect(graph.nodes.contains { $0.name == "attn_scaled" && $0.op == .mul })
    #expect(graph.nodes.contains { $0.name == "attn_masked" && $0.op == .add })
    #expect(graph.nodes.contains { $0.name == "attn_softmax" && $0.op == .softmax })
    #expect(graph.nodes.contains { $0.name == "attn_context" && $0.op == .matmul })
    #expect(graph.nodes.contains { $0.name == "attn_context_transpose" && $0.op == .transpose })
}

@Test func causalAttentionUsesMaskConstAndExpectedOutputShape() throws {
    var graph = ANEGraph()
    let q = try graph.input("q", dtype: .fp16, shape: try ANEShape(channels: 24, spatial: 6))
    let k = try graph.input("k", dtype: .fp16, shape: try ANEShape(channels: 24, spatial: 6))
    let v = try graph.input("v", dtype: .fp16, shape: try ANEShape(channels: 24, spatial: 6))

    let out = try graph.causalAttention(
        "attn",
        q: q,
        k: k,
        v: v,
        nHeads: 6,
        headDim: 4,
        spatial: 6,
        maskPath: "@mask"
    )

    let maskNode = try #require(graph.nodes.first { $0.name == "attn_mask" })
    let expectedMaskShape = try ANEShape(batch: 1, channels: 1, height: 6, spatial: 6)
    let expectedOutputShape = try ANEShape(channels: 24, spatial: 6)
    #expect(maskNode.attrs == .weight(blobPath: "@mask", offset: 64))
    #expect(maskNode.shape == expectedMaskShape)
    #expect(graph.nodes[out].shape == expectedOutputShape)
}

@Test func causalAttentionUsesExpectedTransposeAndSoftmaxAxis() throws {
    var graph = ANEGraph()
    let q = try graph.input("q", dtype: .fp16, shape: try ANEShape(channels: 12, spatial: 4))
    let k = try graph.input("k", dtype: .fp16, shape: try ANEShape(channels: 12, spatial: 4))
    let v = try graph.input("v", dtype: .fp16, shape: try ANEShape(channels: 12, spatial: 4))

    _ = try graph.causalAttention(
        "attn",
        q: q,
        k: k,
        v: v,
        nHeads: 3,
        headDim: 4,
        spatial: 4,
        maskPath: "@mask"
    )

    let scoreNode = try #require(graph.nodes.first { $0.name == "attn_scores" })
    let softmaxNode = try #require(graph.nodes.first { $0.name == "attn_softmax" })
    #expect(scoreNode.attrs == .matmul(transposeX: false, transposeY: true))
    #expect(softmaxNode.attrs == .softmax(axis: -1))
    #expect(graph.nodes.contains { $0.name == "attn_q_transpose" && $0.attrs == .transpose(perm: [0, 1, 3, 2]) })
    #expect(graph.nodes.contains { $0.name == "attn_context_transpose" && $0.attrs == .transpose(perm: [0, 1, 3, 2]) })
}

@Test func causalAttentionPrefixScopesEveryInternalNode() throws {
    var graph = ANEGraph()
    let q = try graph.input("q", dtype: .fp16, shape: try ANEShape(channels: 8, spatial: 4))
    let k = try graph.input("k", dtype: .fp16, shape: try ANEShape(channels: 8, spatial: 4))
    let v = try graph.input("v", dtype: .fp16, shape: try ANEShape(channels: 8, spatial: 4))

    _ = try graph.causalAttention(
        "blk0_attn",
        q: q,
        k: k,
        v: v,
        nHeads: 2,
        headDim: 4,
        spatial: 4,
        maskPath: "@mask"
    )

    let internalNames = graph.nodes.map(\.name).filter { !["q", "k", "v"].contains($0) }
    #expect(internalNames.allSatisfy { $0.hasPrefix("blk0_attn_") })
}
