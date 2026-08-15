import ANEGraphIR

extension ANEGraph {
    /// Half-split RoPE on a packed `[1, nHeads*headDim, 1, spatial]` tensor.
    ///
    /// `cos` / `sin` are `[1, halfDim, 1, 1]` (or broadcastable) and apply the
    /// Llama/Qwen pairs `(i, i+halfDim)`.
    public mutating func ropeHalfSplit(
        _ prefix: String,
        input: Int,
        cos: Int,
        sin: Int,
        nHeads: Int,
        headDim: Int,
        spatial: Int
    ) throws -> Int {
        precondition(nHeads > 0 && headDim > 0 && headDim % 2 == 0 && spatial > 0)
        let halfDim = headDim / 2
        let packed = nHeads * headDim
        let headsShape = try ANEShape(batch: 1, channels: nHeads, height: headDim, spatial: spatial)
        let halfShape = try ANEShape(batch: 1, channels: nHeads, height: halfDim, spatial: spatial)
        let cosShape = try ANEShape(batch: 1, channels: 1, height: halfDim, spatial: 1)

        let x4 = try reshape("\(prefix)_x4", input: input, shape: headsShape)
        let x0 = try sliceBySize(
            "\(prefix)_x0",
            input: x4,
            begin: [0, 0, 0, 0],
            size: [1, nHeads, halfDim, spatial],
            outShape: halfShape
        )
        let x1 = try sliceBySize(
            "\(prefix)_x1",
            input: x4,
            begin: [0, 0, halfDim, 0],
            size: [1, nHeads, halfDim, spatial],
            outShape: halfShape
        )
        let cos4 = try reshape("\(prefix)_cos4", input: cos, shape: cosShape)
        let sin4 = try reshape("\(prefix)_sin4", input: sin, shape: cosShape)
        let x0c = try mul("\(prefix)_x0c", x: x0, y: cos4)
        let x1s = try mul("\(prefix)_x1s", x: x1, y: sin4)
        let y0 = try sub("\(prefix)_y0", x: x0c, y: x1s)
        let x0s = try mul("\(prefix)_x0s", x: x0, y: sin4)
        let x1c = try mul("\(prefix)_x1c", x: x1, y: cos4)
        let y1 = try add("\(prefix)_y1", x: x0s, y: x1c)
        let cat = try concat(
            "\(prefix)_cat",
            values: [y0, y1],
            axis: 2,
            interleave: false,
            outShape: headsShape
        )
        return try reshape(
            "\(prefix)_out",
            input: cat,
            shape: try ANEShape(channels: packed, spatial: spatial)
        )
    }

    /// `cache + current ⊗ posMask` with static shapes. `current` is one token.
    public mutating func scatterCurrentIntoCache(
        _ prefix: String,
        cache: Int,
        current: Int,
        posMask: Int,
        channels: Int,
        maxSeq: Int
    ) throws -> Int {
        precondition(channels > 0 && maxSeq > 0)
        let col = try reshape(
            "\(prefix)_col",
            input: current,
            shape: try ANEShape(batch: 1, channels: 1, height: channels, spatial: 1)
        )
        let outer = try matmul(
            "\(prefix)_outer",
            x: col,
            y: posMask,
            transposeX: false,
            transposeY: false,
            outShape: try ANEShape(batch: 1, channels: 1, height: channels, spatial: maxSeq)
        )
        let scattered = try reshape(
            "\(prefix)_scattered",
            input: outer,
            shape: try ANEShape(channels: channels, spatial: maxSeq)
        )
        return try add("\(prefix)_out", x: cache, y: scattered)
    }

    /// Decode-time GQA: Q is one token, K/V caches are `maxSeq`.
    public mutating func decodeGQAAttention(
        _ prefix: String,
        q: Int,
        kCache: Int,
        vCache: Int,
        mask: Int,
        nHeads: Int,
        nKVHeads: Int,
        headDim: Int,
        maxSeq: Int
    ) throws -> Int {
        precondition(nHeads > 0 && nKVHeads > 0 && nHeads % nKVHeads == 0)
        precondition(headDim > 0 && maxSeq > 0)
        let qDim = nHeads * headDim

        let qHeads = try reshape(
            "\(prefix)_q4",
            input: q,
            shape: try ANEShape(batch: 1, channels: nHeads, height: headDim, spatial: 1)
        )
        let kHeads = try reshape(
            "\(prefix)_k4",
            input: kCache,
            shape: try ANEShape(batch: 1, channels: nKVHeads, height: headDim, spatial: maxSeq)
        )
        let vHeads = try reshape(
            "\(prefix)_v4",
            input: vCache,
            shape: try ANEShape(batch: 1, channels: nKVHeads, height: headDim, spatial: maxSeq)
        )
        let kRep = try repeatHeads(
            "\(prefix)_k",
            input: kHeads,
            nHeads: nHeads,
            nKVHeads: nKVHeads,
            headDim: headDim,
            spatial: maxSeq
        )
        let vRep = try repeatHeads(
            "\(prefix)_v",
            input: vHeads,
            nHeads: nHeads,
            nKVHeads: nKVHeads,
            headDim: headDim,
            spatial: maxSeq
        )

        let qT = try transpose("\(prefix)_q_transpose", input: qHeads, perm: [0, 1, 3, 2])
        let kT = try transpose("\(prefix)_k_transpose", input: kRep, perm: [0, 1, 3, 2])
        let vT = try transpose("\(prefix)_v_transpose", input: vRep, perm: [0, 1, 3, 2])
        let scores = try matmul(
            "\(prefix)_scores",
            x: qT,
            y: kT,
            transposeX: false,
            transposeY: true,
            outShape: try ANEShape(batch: 1, channels: nHeads, height: 1, spatial: maxSeq)
        )
        let scale = try constScalar("\(prefix)_scale", 1.0 / Float(headDim).squareRoot())
        let scaled = try mul("\(prefix)_scaled", x: scores, y: scale)
        let masked = try add("\(prefix)_masked", x: scaled, y: mask)
        let attn = try softmax("\(prefix)_softmax", input: masked, axis: 3)
        let context = try matmul(
            "\(prefix)_context",
            x: attn,
            y: vT,
            transposeX: false,
            transposeY: false,
            outShape: try ANEShape(batch: 1, channels: nHeads, height: 1, spatial: headDim)
        )
        let contextT = try transpose("\(prefix)_context_transpose", input: context, perm: [0, 1, 3, 2])
        return try reshape(
            "\(prefix)_out",
            input: contextT,
            shape: try ANEShape(channels: qDim, spatial: 1)
        )
    }

    /// Repeat each KV head `nHeads/nKVHeads` times in order: `[h0×R, h1×R, ...]`.
    /// Concatenating the packed KV tensor R times would interleave heads and break GQA.
    private mutating func repeatHeads(
        _ prefix: String,
        input: Int,
        nHeads: Int,
        nKVHeads: Int,
        headDim: Int,
        spatial: Int
    ) throws -> Int {
        let repeatCount = nHeads / nKVHeads
        if repeatCount == 1 {
            return input
        }
        let oneHead = try ANEShape(batch: 1, channels: 1, height: headDim, spatial: spatial)
        let groupShape = try ANEShape(batch: 1, channels: repeatCount, height: headDim, spatial: spatial)
        var groups: [Int] = []
        groups.reserveCapacity(nKVHeads)
        for kvIndex in 0..<nKVHeads {
            let head = try sliceBySize(
                "\(prefix)_h\(kvIndex)",
                input: input,
                begin: [0, kvIndex, 0, 0],
                size: [1, 1, headDim, spatial],
                outShape: oneHead
            )
            let group = try concat(
                "\(prefix)_g\(kvIndex)",
                values: Array(repeating: head, count: repeatCount),
                axis: 1,
                interleave: false,
                outShape: groupShape
            )
            groups.append(group)
        }
        if groups.count == 1 {
            return groups[0]
        }
        return try concat(
            "\(prefix)_repeat",
            values: groups,
            axis: 1,
            interleave: false,
            outShape: try ANEShape(batch: 1, channels: nHeads, height: headDim, spatial: spatial)
        )
    }
}
