import Foundation
import ANETypes
import ANEBuilder
import ANEGraphIR

/// Serving fused decode for Phase 11 `max_N = 1`: one transformer layer per MIL
/// program, attention inside the graph. That is 28 hops/token and deletes the
/// Metal QKV↔FFN sync. Do not stack N=2 — Phase 11 rejected it.
///
/// Inputs (alphabetical): kCache, mask, posMask, ropePack, vCache, x
/// Outputs (alphabetical): kNew, vNew, xOut
public struct FusedHybridDecodeLayerGenerator: MILProgramGenerator {
    /// Qwen2.5-1.5B-Instruct widths for the serve Layer graph. The Block
    /// compile probe reads this catalog; it does not own it.
    public enum Qwen15BShape {
        public static let nLayer = 28
        public static let dModel = 1536
        public static let nHead = 12
        public static let nKVHead = 2
        public static let headDim = 128
        public static let hiddenDim = 8960
        public static let qDim = dModel
        public static let kvDim = nKVHead * headDim
        public static let ropeTheta: Float = 1_000_000
        public static let normEps: Float = 1e-6
        public static let laneSpatial = 32
    }

    public static let phase11MaxN = 1
    public static let decodePathLabel = "fused"

    public let dim: Int
    public let qDim: Int
    public let kvDim: Int
    public let hiddenDim: Int
    public let nHeads: Int
    public let nKVHeads: Int
    public let headDim: Int
    public let maxSeq: Int
    public let laneSpatial: Int
    public let normEps: Float
    public let hasQKVBias: Bool

    public init(
        dim: Int,
        qDim: Int,
        kvDim: Int,
        hiddenDim: Int,
        nHeads: Int,
        nKVHeads: Int,
        headDim: Int,
        maxSeq: Int,
        laneSpatial: Int = Qwen15BShape.laneSpatial,
        normEps: Float = Qwen15BShape.normEps,
        hasQKVBias: Bool = true
    ) {
        precondition(dim > 0 && qDim > 0 && kvDim > 0 && hiddenDim > 0)
        precondition(nHeads > 0 && nKVHeads > 0 && nHeads % nKVHeads == 0)
        precondition(headDim > 0 && qDim == nHeads * headDim && kvDim == nKVHeads * headDim)
        precondition(maxSeq > 0 && laneSpatial > 0)
        self.dim = dim
        self.qDim = qDim
        self.kvDim = kvDim
        self.hiddenDim = hiddenDim
        self.nHeads = nHeads
        self.nKVHeads = nKVHeads
        self.headDim = headDim
        self.maxSeq = maxSeq
        self.laneSpatial = laneSpatial
        self.normEps = normEps
        self.hasQKVBias = hasQKVBias
    }

    public static func qwen15B(
        maxSeq: Int,
        laneSpatial: Int = Qwen15BShape.laneSpatial
    ) -> Self {
        let shape = Qwen15BShape.self
        return Self(
            dim: shape.dModel,
            qDim: shape.qDim,
            kvDim: shape.kvDim,
            hiddenDim: shape.hiddenDim,
            nHeads: shape.nHead,
            nKVHeads: shape.nKVHead,
            headDim: shape.headDim,
            maxSeq: maxSeq,
            laneSpatial: laneSpatial,
            normEps: shape.normEps,
            hasQKVBias: true
        )
    }

    public static func hopsPerToken(
        nLayer: Int = Qwen15BShape.nLayer,
        blockSize: Int = phase11MaxN
    ) -> Int {
        precondition(nLayer > 0 && blockSize > 0)
        return (nLayer + blockSize - 1) / blockSize
    }

    public var inputBytes: Int { dim * laneSpatial * 2 }

    public var inputByteSizes: [Int] {
        let cacheBytes = dim * maxSeq * 2
        let laneBytes = dim * laneSpatial * 2
        return [cacheBytes, cacheBytes, cacheBytes, laneBytes, cacheBytes, laneBytes]
    }

    public var outputByteSizes: [Int] {
        let laneBytes = dim * laneSpatial * 2
        return [laneBytes, laneBytes, laneBytes]
    }

    public var milText: String {
        let dim = self.dim
        let qDim = self.qDim
        let kvDim = self.kvDim
        let hiddenDim = self.hiddenDim
        let nHeads = self.nHeads
        let nKVHeads = self.nKVHeads
        let headDim = self.headDim
        let maxSeq = self.maxSeq
        let laneSpatial = self.laneSpatial
        let normEps = self.normEps
        let hasQKVBias = self.hasQKVBias
        let halfDim = headDim / 2

        return LegacyGraphSupport.emitGraph { graph in
            let kCache = try graph.input(
                "kCache",
                dtype: .fp16,
                shape: try ANEShape(channels: dim, spatial: maxSeq)
            )
            let mask = try graph.input(
                "mask",
                dtype: .fp16,
                shape: try ANEShape(channels: dim, spatial: maxSeq)
            )
            let posMask = try graph.input(
                "posMask",
                dtype: .fp16,
                shape: try ANEShape(channels: dim, spatial: maxSeq)
            )
            let ropePack = try graph.input(
                "ropePack",
                dtype: .fp16,
                shape: try ANEShape(channels: dim, spatial: laneSpatial)
            )
            let vCache = try graph.input(
                "vCache",
                dtype: .fp16,
                shape: try ANEShape(channels: dim, spatial: maxSeq)
            )
            let x = try graph.input(
                "x",
                dtype: .fp16,
                shape: try ANEShape(channels: dim, spatial: laneSpatial)
            )

            let normalized = try graph.rmsNorm(
                "attn_norm",
                input: x,
                dim: dim,
                spatial: laneSpatial,
                eps: normEps,
                weightPath: "@model_path/weights/rms1.bin"
            )
            let qFull = try graph.linear(
                "q",
                input: normalized,
                inDim: dim,
                outDim: qDim,
                spatial: laneSpatial,
                weightPath: "@model_path/weights/wq.bin",
                biasPath: hasQKVBias ? "@model_path/weights/bq.bin" : nil
            )
            let kFull = try graph.linear(
                "k",
                input: normalized,
                inDim: dim,
                outDim: kvDim,
                spatial: laneSpatial,
                weightPath: "@model_path/weights/wk.bin",
                biasPath: hasQKVBias ? "@model_path/weights/bk.bin" : nil
            )
            let vFull = try graph.linear(
                "v",
                input: normalized,
                inDim: dim,
                outDim: kvDim,
                spatial: laneSpatial,
                weightPath: "@model_path/weights/wv.bin",
                biasPath: hasQKVBias ? "@model_path/weights/bv.bin" : nil
            )

            let qTok = try graph.sliceBySize(
                "q_tok",
                input: qFull,
                begin: [0, 0, 0, 0],
                size: [1, qDim, 1, 1],
                outShape: try ANEShape(channels: qDim, spatial: 1)
            )
            let kTok = try graph.sliceBySize(
                "k_tok",
                input: kFull,
                begin: [0, 0, 0, 0],
                size: [1, kvDim, 1, 1],
                outShape: try ANEShape(channels: kvDim, spatial: 1)
            )
            let vTok = try graph.sliceBySize(
                "v_tok",
                input: vFull,
                begin: [0, 0, 0, 0],
                size: [1, kvDim, 1, 1],
                outShape: try ANEShape(channels: kvDim, spatial: 1)
            )

            let cos = try graph.sliceBySize(
                "rope_cos",
                input: ropePack,
                begin: [0, 0, 0, 0],
                size: [1, halfDim, 1, 1],
                outShape: try ANEShape(channels: halfDim, spatial: 1)
            )
            let sin = try graph.sliceBySize(
                "rope_sin",
                input: ropePack,
                begin: [0, halfDim, 0, 0],
                size: [1, halfDim, 1, 1],
                outShape: try ANEShape(channels: halfDim, spatial: 1)
            )
            let qRot = try graph.ropeHalfSplit(
                "q_rope",
                input: qTok,
                cos: cos,
                sin: sin,
                nHeads: nHeads,
                headDim: headDim,
                spatial: 1
            )
            let kRot = try graph.ropeHalfSplit(
                "k_rope",
                input: kTok,
                cos: cos,
                sin: sin,
                nHeads: nKVHeads,
                headDim: headDim,
                spatial: 1
            )

            let kPast = try graph.sliceBySize(
                "k_past",
                input: kCache,
                begin: [0, 0, 0, 0],
                size: [1, kvDim, 1, maxSeq],
                outShape: try ANEShape(channels: kvDim, spatial: maxSeq)
            )
            let vPast = try graph.sliceBySize(
                "v_past",
                input: vCache,
                begin: [0, 0, 0, 0],
                size: [1, kvDim, 1, maxSeq],
                outShape: try ANEShape(channels: kvDim, spatial: maxSeq)
            )
            let pos = try graph.sliceBySize(
                "pos",
                input: posMask,
                begin: [0, 0, 0, 0],
                size: [1, 1, 1, maxSeq],
                outShape: try ANEShape(channels: 1, spatial: maxSeq)
            )
            let attnMask = try graph.sliceBySize(
                "attn_mask",
                input: mask,
                begin: [0, 0, 0, 0],
                size: [1, 1, 1, maxSeq],
                outShape: try ANEShape(channels: 1, spatial: maxSeq)
            )
            let kUpdated = try graph.scatterCurrentIntoCache(
                "k_sc",
                cache: kPast,
                current: kRot,
                posMask: pos,
                channels: kvDim,
                maxSeq: maxSeq
            )
            let vUpdated = try graph.scatterCurrentIntoCache(
                "v_sc",
                cache: vPast,
                current: vTok,
                posMask: pos,
                channels: kvDim,
                maxSeq: maxSeq
            )
            let context = try graph.decodeGQAAttention(
                "attn",
                q: qRot,
                kCache: kUpdated,
                vCache: vUpdated,
                mask: attnMask,
                nHeads: nHeads,
                nKVHeads: nKVHeads,
                headDim: headDim,
                maxSeq: maxSeq
            )
            let projected = try graph.linear(
                "o",
                input: context,
                inDim: qDim,
                outDim: dim,
                spatial: 1,
                weightPath: "@model_path/weights/wo.bin"
            )
            let zero = try graph.constScalar("lane_z", 0)
            let zeroLane = try graph.mul("lane_zero", x: x, y: zero)
            let attnLane = try Self.placeTokenAtLane0(
                &graph,
                prefix: "attn_lane",
                token: projected,
                tokenChannels: dim,
                dim: dim,
                laneSpatial: laneSpatial,
                zeroLane: zeroLane
            )
            let xMid = try graph.add("attn_res", x: x, y: attnLane)

            let ffnNorm = try graph.rmsNorm(
                "ffn_norm",
                input: xMid,
                dim: dim,
                spatial: laneSpatial,
                eps: normEps,
                weightPath: "@model_path/weights/rms2.bin"
            )
            let ffn = try graph.swigluFFN(
                "ffn",
                input: ffnNorm,
                inDim: dim,
                hiddenDim: hiddenDim,
                spatial: laneSpatial,
                w1Path: "@model_path/weights/w1.bin",
                w3Path: "@model_path/weights/w3.bin",
                w2Path: "@model_path/weights/w2.bin"
            )
            let xOut = try graph.add("xOut", x: xMid, y: ffn)
            let kNew = try Self.placeTokenAtLane0(
                &graph,
                prefix: "kNew",
                token: kRot,
                tokenChannels: kvDim,
                dim: dim,
                laneSpatial: laneSpatial,
                zeroLane: zeroLane
            )
            let vNew = try Self.placeTokenAtLane0(
                &graph,
                prefix: "vNew",
                token: vTok,
                tokenChannels: kvDim,
                dim: dim,
                laneSpatial: laneSpatial,
                zeroLane: zeroLane
            )
            try LegacyGraphSupport.setOutputs(&graph, [("kNew", kNew), ("vNew", vNew), ("xOut", xOut)])
        }
    }

    private static func placeTokenAtLane0(
        _ graph: inout ANEGraph,
        prefix: String,
        token: Int,
        tokenChannels: Int,
        dim: Int,
        laneSpatial: Int,
        zeroLane: Int
    ) throws -> Int {
        let zeroTok = try graph.sliceBySize(
            "\(prefix)_ztok",
            input: zeroLane,
            begin: [0, 0, 0, 0],
            size: [1, tokenChannels, 1, 1],
            outShape: try ANEShape(channels: tokenChannels, spatial: 1)
        )
        let tokenLane: Int
        if laneSpatial == 1 {
            tokenLane = token
        } else {
            tokenLane = try graph.concat(
                "\(prefix)_lane",
                values: [token] + Array(repeating: zeroTok, count: laneSpatial - 1),
                axis: 3,
                interleave: false,
                outShape: try ANEShape(channels: tokenChannels, spatial: laneSpatial)
            )
        }
        if tokenChannels == dim {
            return tokenLane
        }
        let pad = try graph.sliceBySize(
            "\(prefix)_chpad",
            input: zeroLane,
            begin: [0, tokenChannels, 0, 0],
            size: [1, dim - tokenChannels, 1, laneSpatial],
            outShape: try ANEShape(channels: dim - tokenChannels, spatial: laneSpatial)
        )
        return try graph.concat(
            prefix,
            values: [tokenLane, pad],
            axis: 1,
            interleave: false,
            outShape: try ANEShape(channels: dim, spatial: laneSpatial)
        )
    }
}
