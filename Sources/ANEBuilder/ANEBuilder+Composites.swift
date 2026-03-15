import ANEGraphIR

extension ANEGraph {
    public mutating func linear(
        _ prefix: String,
        input: Int,
        inDim: Int,
        outDim: Int,
        spatial: Int,
        weightPath: String,
        biasPath: String? = nil
    ) throws -> Int {
        let weight = try constWeight(
            "\(prefix)_weight",
            shape: try ANEShape(batch: outDim, channels: inDim, height: 1, spatial: 1),
            blobPath: weightPath
        )
        let conv = try conv1x1(
            "\(prefix)_conv",
            input: input,
            weight: weight,
            bias: nil,
            outShape: try ANEShape(channels: outDim, spatial: spatial)
        )

        guard let biasPath else {
            return conv
        }

        let bias = try constWeight(
            "\(prefix)_bias",
            shape: try ANEShape(channels: outDim, spatial: 1),
            blobPath: biasPath
        )
        return try add("\(prefix)_out", x: conv, y: bias)
    }

    public mutating func rmsNorm(
        _ prefix: String,
        input: Int,
        dim: Int,
        spatial: Int,
        eps: Float,
        weightPath: String
    ) throws -> Int {
        let sq = try mul("\(prefix)_sq", x: input, y: input)
        let ss = try reduceSum("\(prefix)_ss", input: sq, axis: 1, keepDims: true)
        let invd = try constScalar("\(prefix)_invd", 1.0 / Float(dim))
        let ms = try mul("\(prefix)_ms", x: ss, y: invd)
        let epsNode = try constScalar("\(prefix)_eps", eps)
        let ss3 = try add("\(prefix)_ss3", x: ms, y: epsNode)
        let nhalf = try constScalar("\(prefix)_nhalf", -0.5)
        let rrms = try pow("\(prefix)_rrms", base: ss3, exp: nhalf)
        let xr = try mul("\(prefix)_xr", x: input, y: rrms)
        let weight = try constWeight(
            "\(prefix)_weight",
            shape: try ANEShape(channels: dim, spatial: 1),
            blobPath: weightPath
        )
        return try mul("\(prefix)_out", x: xr, y: weight)
    }

    public mutating func layerNorm(
        _ prefix: String,
        input: Int,
        dim: Int,
        spatial: Int,
        eps: Float,
        gammaPath: String,
        betaPath: String
    ) throws -> Int {
        let sum = try reduceSum("\(prefix)_sum", input: input, axis: 1, keepDims: true)
        let invd = try constScalar("\(prefix)_invd", 1.0 / Float(dim))
        let mean = try mul("\(prefix)_mean", x: sum, y: invd)
        let centered = try sub("\(prefix)_centered", x: input, y: mean)
        let sq = try mul("\(prefix)_sq", x: centered, y: centered)
        let ss = try reduceSum("\(prefix)_ss", input: sq, axis: 1, keepDims: true)
        let variance = try mul("\(prefix)_var", x: ss, y: invd)
        let epsNode = try constScalar("\(prefix)_eps", eps)
        let varEps = try add("\(prefix)_var_eps", x: variance, y: epsNode)
        let nhalf = try constScalar("\(prefix)_nhalf", -0.5)
        let invStd = try pow("\(prefix)_inv_std", base: varEps, exp: nhalf)
        let normalized = try mul("\(prefix)_normalized", x: centered, y: invStd)
        let gamma = try constWeight(
            "\(prefix)_gamma",
            shape: try ANEShape(channels: dim, spatial: 1),
            blobPath: gammaPath
        )
        let scaled = try mul("\(prefix)_scaled", x: normalized, y: gamma)
        let beta = try constWeight(
            "\(prefix)_beta",
            shape: try ANEShape(channels: dim, spatial: 1),
            blobPath: betaPath
        )
        return try add("\(prefix)_out", x: scaled, y: beta)
    }

    public mutating func gelu(
        _ prefix: String,
        input: Int
    ) throws -> Int {
        let x2 = try mul("\(prefix)_x2", x: input, y: input)
        let x3 = try mul("\(prefix)_x3", x: x2, y: input)
        let cubic = try constScalar("\(prefix)_cubic", 0.044715)
        let cx3 = try mul("\(prefix)_cx3", x: x3, y: cubic)
        let inner = try add("\(prefix)_inner", x: input, y: cx3)
        let scale = try constScalar("\(prefix)_scale", 0.797_884_6)
        let scaled = try mul("\(prefix)_scaled", x: inner, y: scale)
        let tanhNode = try tanh("\(prefix)_tanh", input: scaled)
        let one = try constScalar("\(prefix)_one", 1.0)
        let onePlus = try add("\(prefix)_one_plus", x: tanhNode, y: one)
        let half = try constScalar("\(prefix)_half", 0.5)
        let halfX = try mul("\(prefix)_half_x", x: input, y: half)
        return try mul("\(prefix)_out", x: halfX, y: onePlus)
    }

    public mutating func silu(
        _ prefix: String,
        input: Int
    ) throws -> Int {
        let sigmoidNode = try sigmoid("\(prefix)_sigmoid", input: input)
        return try mul("\(prefix)_out", x: input, y: sigmoidNode)
    }

    public mutating func residual(
        _ prefix: String,
        x: Int,
        sublayer: Int
    ) throws -> Int {
        try add("\(prefix)_out", x: x, y: sublayer)
    }

    public mutating func castToFP16(
        _ prefix: String,
        input: Int
    ) throws -> Int {
        try cast("\(prefix)_out", input: input, to: .fp16)
    }

    public mutating func castToFP32(
        _ prefix: String,
        input: Int
    ) throws -> Int {
        try cast("\(prefix)_out", input: input, to: .fp32)
    }
}
