import Foundation
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
        if ProcessInfo.processInfo.environment["ESPRESSO_RMSNORM_USE_FP32"] == "1" {
            let input32 = try castToFP32("\(prefix)_input32", input: input)
            let sq32 = try mul("\(prefix)_sq32", x: input32, y: input32)
            let ss32 = try reduceSum("\(prefix)_ss32", input: sq32, axis: 1, keepDims: true)
            let invd32 = try constScalar("\(prefix)_invd32", 1.0 / Float(dim))
            let ms32 = try mul("\(prefix)_ms32", x: ss32, y: invd32)
            let eps32 = try constScalar("\(prefix)_eps32", eps)
            let ss332 = try add("\(prefix)_ss332", x: ms32, y: eps32)
            let nhalf32 = try constScalar("\(prefix)_nhalf32", -0.5)
            let rrms32 = try pow("\(prefix)_rrms32", base: ss332, exp: nhalf32)
            let xr32 = try mul("\(prefix)_xr32", x: input32, y: rrms32)
            let xr16 = try castToFP16("\(prefix)_xr16", input: xr32)
            let weight = try constWeight(
                "\(prefix)_weight",
                shape: try ANEShape(channels: dim, spatial: 1),
                blobPath: weightPath
            )
            return try mul("\(prefix)_out", x: xr16, y: weight)
        }
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
        if ProcessInfo.processInfo.environment["ESPRESSO_SILU_USE_FP32"] == "1" {
            let input32 = try castToFP32("\(prefix)_input32", input: input)
            let sigmoid32 = try sigmoid("\(prefix)_sigmoid", input: input32)
            let out32 = try mul("\(prefix)_out32", x: input32, y: sigmoid32)
            return try castToFP16("\(prefix)_out", input: out32)
        }
        // Default is true SiLU (x * sigmoid(x)): it matches the training-side
        // generators and the retained migration fixtures. The tanh identity
        // (mathematically equal, differently rounded in fp16) stays available
        // as an experiment behind ESPRESSO_SILU_USE_TANH=1.
        if ProcessInfo.processInfo.environment["ESPRESSO_SILU_USE_TANH"] == "1" {
            let half = try constScalar("\(prefix)_half", 0.5)
            let halfInput = try mul("\(prefix)_half_input", x: input, y: half)
            let tanhNode = try tanh("\(prefix)_tanh", input: halfInput)
            let one = try constScalar("\(prefix)_one", 1.0)
            let onePlus = try add("\(prefix)_one_plus", x: tanhNode, y: one)
            let halfX = try mul("\(prefix)_half_x", x: input, y: half)
            return try mul("\(prefix)_out", x: halfX, y: onePlus)
        }
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
