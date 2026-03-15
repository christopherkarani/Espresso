import ANEGraphIR

extension ANEGraph {
    public mutating func causalAttention(
        _ prefix: String,
        q: Int,
        k: Int,
        v: Int,
        nHeads: Int,
        headDim: Int,
        spatial: Int,
        maskPath: String
    ) throws -> Int {
        let modelDim = nHeads * headDim
        let headShape = try ANEShape(batch: 1, channels: nHeads, height: headDim, spatial: spatial)
        let transposedShape = try ANEShape(batch: 1, channels: nHeads, height: spatial, spatial: headDim)
        let scoresShape = try ANEShape(batch: 1, channels: nHeads, height: spatial, spatial: spatial)

        let q4 = try reshape("\(prefix)_q_reshape", input: q, shape: headShape)
        let k4 = try reshape("\(prefix)_k_reshape", input: k, shape: headShape)
        let v4 = try reshape("\(prefix)_v_reshape", input: v, shape: headShape)

        let qT = try transpose("\(prefix)_q_transpose", input: q4, perm: [0, 1, 3, 2])
        let kT = try transpose("\(prefix)_k_transpose", input: k4, perm: [0, 1, 3, 2])
        let vT = try transpose("\(prefix)_v_transpose", input: v4, perm: [0, 1, 3, 2])

        let scores = try matmul(
            "\(prefix)_scores",
            x: qT,
            y: kT,
            transposeX: false,
            transposeY: true,
            outShape: scoresShape
        )
        let scale = try constScalar("\(prefix)_scale", 1.0 / Float(headDim).squareRoot())
        let scaled = try mul("\(prefix)_scaled", x: scores, y: scale)
        let mask = try constWeight(
            "\(prefix)_mask",
            shape: try ANEShape(batch: 1, channels: 1, height: spatial, spatial: spatial),
            blobPath: maskPath
        )
        let masked = try add("\(prefix)_masked", x: scaled, y: mask)
        let attn = try softmax("\(prefix)_softmax", input: masked, axis: -1)
        let context = try matmul(
            "\(prefix)_context",
            x: attn,
            y: vT,
            transposeX: false,
            transposeY: false,
            outShape: transposedShape
        )
        let contextT = try transpose("\(prefix)_context_transpose", input: context, perm: [0, 1, 3, 2])
        return try reshape(
            "\(prefix)_out",
            input: contextT,
            shape: try ANEShape(channels: modelDim, spatial: spatial)
        )
    }
}
