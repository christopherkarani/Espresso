import Foundation
import ANETypes
import ANEBuilder
import ANEGraphIR

/// Compile-only fused QKV+FFN block for measuring ANE graph depth at Qwen widths.
///
/// Attention / RoPE stay off this graph (Metal). Each stacked layer is:
/// RMSNorm → QKV(+bias) → zero-scale keep-alive → RMSNorm → SwiGLU FFN → residual.
/// Stories recurrent fusion is a different graph and does not count as a pass.
/// Shape and hops live on `FusedHybridDecodeLayerGenerator`.
public struct FusedHybridDecodeBlockGenerator: MILProgramGenerator {
    public let layerCount: Int
    public let dim: Int
    public let qDim: Int
    public let kvDim: Int
    public let hiddenDim: Int
    public let laneSpatial: Int
    public let normEps: Float
    public let hasQKVBias: Bool

    public init(
        layerCount: Int,
        dim: Int,
        qDim: Int,
        kvDim: Int,
        hiddenDim: Int,
        laneSpatial: Int = FusedHybridDecodeLayerGenerator.Qwen15BShape.laneSpatial,
        normEps: Float = FusedHybridDecodeLayerGenerator.Qwen15BShape.normEps,
        hasQKVBias: Bool = true
    ) {
        precondition(layerCount > 0)
        precondition(dim > 0 && qDim > 0 && kvDim > 0 && hiddenDim > 0)
        precondition(laneSpatial > 0)
        self.layerCount = layerCount
        self.dim = dim
        self.qDim = qDim
        self.kvDim = kvDim
        self.hiddenDim = hiddenDim
        self.laneSpatial = laneSpatial
        self.normEps = normEps
        self.hasQKVBias = hasQKVBias
    }

    public static func qwen15B(
        layerCount: Int,
        laneSpatial: Int = FusedHybridDecodeLayerGenerator.Qwen15BShape.laneSpatial
    ) -> Self {
        let shape = FusedHybridDecodeLayerGenerator.Qwen15BShape.self
        return Self(
            layerCount: layerCount,
            dim: shape.dModel,
            qDim: shape.qDim,
            kvDim: shape.kvDim,
            hiddenDim: shape.hiddenDim,
            laneSpatial: laneSpatial,
            normEps: shape.normEps,
            hasQKVBias: true
        )
    }

    public var inputBytes: Int { dim * laneSpatial * 2 }

    public var inputByteSizes: [Int] { [inputBytes] }

    public var outputByteSizes: [Int] { [dim * laneSpatial * 2] }

    public var milText: String {
        LegacyGraphSupport.emitGraph { graph in
            var x = try LegacyGraphSupport.input(&graph, name: "x", channels: dim, spatial: laneSpatial)
            for layer in 0..<layerCount {
                x = try emitLayer(&graph, input: x, layer: layer)
            }
            try LegacyGraphSupport.setOutputs(&graph, [("out", x)])
        }
    }

    private func emitLayer(_ graph: inout ANEGraph, input: Int, layer: Int) throws -> Int {
        let prefix = "l\(layer)"
        let normalizedQKV = try graph.rmsNorm(
            "\(prefix)_attn_norm",
            input: input,
            dim: dim,
            spatial: laneSpatial,
            eps: normEps,
            weightPath: "@model_path/weights/\(prefix)_rms1.bin"
        )
        let qOut = try graph.linear(
            "\(prefix)_q",
            input: normalizedQKV,
            inDim: dim,
            outDim: qDim,
            spatial: laneSpatial,
            weightPath: "@model_path/weights/\(prefix)_wq.bin",
            biasPath: hasQKVBias ? "@model_path/weights/\(prefix)_bq.bin" : nil
        )
        let kNew = try graph.linear(
            "\(prefix)_k",
            input: normalizedQKV,
            inDim: dim,
            outDim: kvDim,
            spatial: laneSpatial,
            weightPath: "@model_path/weights/\(prefix)_wk.bin",
            biasPath: hasQKVBias ? "@model_path/weights/\(prefix)_bk.bin" : nil
        )
        let vNew = try graph.linear(
            "\(prefix)_v",
            input: normalizedQKV,
            inDim: dim,
            outDim: kvDim,
            spatial: laneSpatial,
            weightPath: "@model_path/weights/\(prefix)_wv.bin",
            biasPath: hasQKVBias ? "@model_path/weights/\(prefix)_bv.bin" : nil
        )

        // Keep QKV live without putting Metal attention on ANE. Zero-scale
        // residual matches the Stories fused-decode probe: compile size, not serve.
        let qCh = try graph.reduceSum("\(prefix)_q_ch", input: qOut, axis: 1, keepDims: true)
        let kCh = try graph.reduceSum("\(prefix)_k_ch", input: kNew, axis: 1, keepDims: true)
        let vCh = try graph.reduceSum("\(prefix)_v_ch", input: vNew, axis: 1, keepDims: true)
        let qS = try graph.reduceSum("\(prefix)_q_s", input: qCh, axis: 3, keepDims: true)
        let kS = try graph.reduceSum("\(prefix)_k_s", input: kCh, axis: 3, keepDims: true)
        let vS = try graph.reduceSum("\(prefix)_v_s", input: vCh, axis: 3, keepDims: true)
        let kv = try graph.add("\(prefix)_kv", x: kS, y: vS)
        let qkv = try graph.add("\(prefix)_qkv", x: kv, y: qS)
        let zero = try graph.constScalar("\(prefix)_zc", 0)
        let z = try graph.mul("\(prefix)_z", x: qkv, y: zero)
        let xMid = try graph.add("\(prefix)_res", x: input, y: z)

        let normalizedFFN = try graph.rmsNorm(
            "\(prefix)_ffn_norm",
            input: xMid,
            dim: dim,
            spatial: laneSpatial,
            eps: normEps,
            weightPath: "@model_path/weights/\(prefix)_rms2.bin"
        )
        let y = try graph.swigluFFN(
            "\(prefix)_ffn",
            input: normalizedFFN,
            inDim: dim,
            hiddenDim: hiddenDim,
            spatial: laneSpatial,
            w1Path: "@model_path/weights/\(prefix)_w1.bin",
            w3Path: "@model_path/weights/\(prefix)_w3.bin",
            w2Path: "@model_path/weights/\(prefix)_w2.bin"
        )
        return try graph.add("\(prefix)_out", x: xMid, y: y)
    }
}
