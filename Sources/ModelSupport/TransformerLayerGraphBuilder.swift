import ANEBuilder
import ANEGraphIR

public struct TransformerLayerGraphBuilder {
    static func isSupportedMaskBucket(spatial: Int, maxSeq: Int) -> Bool {
        guard spatial > 0, spatial <= maxSeq else { return false }
        return (spatial & (spatial - 1)) == 0
    }

    public static func forwardLayer(
        layer: Int,
        config: MultiModelConfig,
        paths: LayerWeightPaths,
        spatial: Int
    ) -> ANEGraph {
        precondition(config.dModel == config.nHead * config.headDim, "dModel must equal nHead * headDim")
        precondition(
            isSupportedMaskBucket(spatial: spatial, maxSeq: config.maxSeq),
            "spatial must be a power-of-two mask bucket within model context"
        )

        var graph = ANEGraph()
        let input = try! graph.input(
            "x",
            dtype: .fp16,
            shape: try! ANEShape(channels: config.dModel, spatial: spatial)
        )
        let prefix = "layer\(layer)"

        let output: Int
        switch config.architecture {
        case .gpt2:
            let ln1 = try! graph.layerNorm128(
                "\(prefix)_ln1",
                input: input,
                dim: config.dModel,
                spatial: spatial,
                eps: config.normEps,
                gammaPath: paths.rmsAtt,
                betaPath: paths.attentionNormBiasPath!
            )
            let q = try! graph.linear128(
                "\(prefix)_q",
                input: ln1,
                inDim: config.dModel,
                outDim: config.dModel,
                spatial: spatial,
                weightPath: paths.wq,
                biasPath: paths.bq
            )
            let k = try! graph.linear128(
                "\(prefix)_k",
                input: ln1,
                inDim: config.dModel,
                outDim: config.dModel,
                spatial: spatial,
                weightPath: paths.wk,
                biasPath: paths.bk
            )
            let v = try! graph.linear128(
                "\(prefix)_v",
                input: ln1,
                inDim: config.dModel,
                outDim: config.dModel,
                spatial: spatial,
                weightPath: paths.wv,
                biasPath: paths.bv
            )
            let attn = try! graph.causalAttention128(
                "\(prefix)_attn",
                q: q,
                k: k,
                v: v,
                nHeads: config.nHead,
                headDim: config.headDim,
                spatial: spatial,
                maskPath: paths.causalMaskPath(spatial: spatial)
            )
            let projected = try! graph.linear128(
                "\(prefix)_attn_proj",
                input: attn,
                inDim: config.dModel,
                outDim: config.dModel,
                spatial: spatial,
                weightPath: paths.wo,
                biasPath: paths.bo
            )
            let residual1 = try! graph.add("\(prefix)_res1_out", x: input, y: projected)
            let ln2 = try! graph.layerNorm128(
                "\(prefix)_ln2",
                input: residual1,
                dim: config.dModel,
                spatial: spatial,
                eps: config.normEps,
                gammaPath: paths.rmsFfn,
                betaPath: paths.ffnNormBiasPath!
            )
            let ffn = try! graph.ffn128(
                "\(prefix)_ffn",
                input: ln2,
                inDim: config.dModel,
                hiddenDim: config.hiddenDim,
                spatial: spatial,
                w1Path: paths.w1,
                b1Path: paths.b1,
                w2Path: paths.w2,
                b2Path: paths.b2,
                activation: .gelu
            )
            output = try! graph.add("\(prefix)_res2_out", x: residual1, y: ffn)

        case .llama:
            let norm1 = try! graph.rmsNorm128(
                "\(prefix)_rms1",
                input: input,
                dim: config.dModel,
                spatial: spatial,
                eps: config.normEps,
                weightPath: paths.rmsAtt
            )
            let q = try! graph.linear128(
                "\(prefix)_q",
                input: norm1,
                inDim: config.dModel,
                outDim: config.dModel,
                spatial: spatial,
                weightPath: paths.wq
            )
            // GQA is handled offline by repeating KV head blocks in the converter.
            let k = try! graph.linear128(
                "\(prefix)_k",
                input: norm1,
                inDim: config.dModel,
                outDim: config.dModel,
                spatial: spatial,
                weightPath: paths.wk
            )
            let v = try! graph.linear128(
                "\(prefix)_v",
                input: norm1,
                inDim: config.dModel,
                outDim: config.dModel,
                spatial: spatial,
                weightPath: paths.wv
            )
            let attn = try! graph.causalAttention128(
                "\(prefix)_attn",
                q: q,
                k: k,
                v: v,
                nHeads: config.nHead,
                headDim: config.headDim,
                spatial: spatial,
                maskPath: paths.causalMaskPath(spatial: spatial)
            )
            let projected = try! graph.linear128(
                "\(prefix)_attn_proj",
                input: attn,
                inDim: config.dModel,
                outDim: config.dModel,
                spatial: spatial,
                weightPath: paths.wo
            )
            let residual1 = try! graph.add("\(prefix)_res1_out", x: input, y: projected)
            let norm2 = try! graph.rmsNorm128(
                "\(prefix)_rms2",
                input: residual1,
                dim: config.dModel,
                spatial: spatial,
                eps: config.normEps,
                weightPath: paths.rmsFfn
            )
            let ffn = try! graph.swigluFFN128(
                "\(prefix)_ffn",
                input: norm2,
                inDim: config.dModel,
                hiddenDim: config.hiddenDim,
                spatial: spatial,
                w1Path: paths.w1,
                w3Path: paths.w3!,
                w2Path: paths.w2
            )
            output = try! graph.add("\(prefix)_res2_out", x: residual1, y: ffn)
        }

        _ = try! graph.output(output, name: "out")
        return graph
    }

    /// Pre-RoPE graph: RMSNorm -> Wq, Wk, Wv projections.
    /// Outputs: k, q, v (alphabetical order).
    public static func preRoPEForwardLayer(
        layer: Int,
        config: MultiModelConfig,
        paths: LayerWeightPaths,
        spatial: Int
    ) -> ANEGraph {
        precondition(config.architecture == .llama, "preRoPEForwardLayer only supports llama architecture")
        precondition(config.dModel == config.nHead * config.headDim, "dModel must equal nHead * headDim")
        precondition(
            isSupportedMaskBucket(spatial: spatial, maxSeq: config.maxSeq),
            "spatial must be a power-of-two mask bucket within model context"
        )

        var graph = ANEGraph()
        let input = try! graph.input(
            "x",
            dtype: .fp16,
            shape: try! ANEShape(channels: config.dModel, spatial: spatial)
        )
        let prefix = "layer\(layer)"

        let norm = try! graph.rmsNorm128(
            "\(prefix)_rms1",
            input: input,
            dim: config.dModel,
            spatial: spatial,
            eps: config.normEps,
            weightPath: paths.rmsAtt
        )
        let q = try! graph.linear128(
            "\(prefix)_q",
            input: norm,
            inDim: config.dModel,
            outDim: config.dModel,
            spatial: spatial,
            weightPath: paths.wq
        )
        let k = try! graph.linear128(
            "\(prefix)_k",
            input: norm,
            inDim: config.dModel,
            outDim: config.dModel,
            spatial: spatial,
            weightPath: paths.wk
        )
        let v = try! graph.linear128(
            "\(prefix)_v",
            input: norm,
            inDim: config.dModel,
            outDim: config.dModel,
            spatial: spatial,
            weightPath: paths.wv
        )

        // Outputs in alphabetical order: k, q, v
        _ = try! graph.output(k, name: "k")
        _ = try! graph.output(q, name: "q")
        _ = try! graph.output(v, name: "v")
        return graph
    }

    /// Post-RoPE graph: causalAttention(Q,K,V) -> Wo -> residual -> RMSNorm -> SwiGLU FFN -> residual.
    /// Inputs: k_rotated, q_rotated, v, x_residual (alphabetical order).
    public static func postRoPEForwardLayer(
        layer: Int,
        config: MultiModelConfig,
        paths: LayerWeightPaths,
        spatial: Int
    ) -> ANEGraph {
        precondition(config.architecture == .llama, "postRoPEForwardLayer only supports llama architecture")
        precondition(config.dModel == config.nHead * config.headDim, "dModel must equal nHead * headDim")
        precondition(
            isSupportedMaskBucket(spatial: spatial, maxSeq: config.maxSeq),
            "spatial must be a power-of-two mask bucket within model context"
        )

        var graph = ANEGraph()
        let prefix = "layer\(layer)"

        let kRotated = try! graph.input(
            "k_rotated",
            dtype: .fp16,
            shape: try! ANEShape(channels: config.dModel, spatial: spatial)
        )
        let qRotated = try! graph.input(
            "q_rotated",
            dtype: .fp16,
            shape: try! ANEShape(channels: config.dModel, spatial: spatial)
        )
        let v = try! graph.input(
            "v",
            dtype: .fp16,
            shape: try! ANEShape(channels: config.dModel, spatial: spatial)
        )
        let xResidual = try! graph.input(
            "x_residual",
            dtype: .fp16,
            shape: try! ANEShape(channels: config.dModel, spatial: spatial)
        )

        let attn = try! graph.causalAttention128(
            "\(prefix)_attn",
            q: qRotated,
            k: kRotated,
            v: v,
            nHeads: config.nHead,
            headDim: config.headDim,
            spatial: spatial,
            maskPath: paths.causalMaskPath(spatial: spatial)
        )
        let projected = try! graph.linear128(
            "\(prefix)_attn_proj",
            input: attn,
            inDim: config.dModel,
            outDim: config.dModel,
            spatial: spatial,
            weightPath: paths.wo
        )
        let residual1 = try! graph.add("\(prefix)_res1_out", x: xResidual, y: projected)
        let norm2 = try! graph.rmsNorm128(
            "\(prefix)_rms2",
            input: residual1,
            dim: config.dModel,
            spatial: spatial,
            eps: config.normEps,
            weightPath: paths.rmsFfn
        )
        let ffn = try! graph.swigluFFN128(
            "\(prefix)_ffn",
            input: norm2,
            inDim: config.dModel,
            hiddenDim: config.hiddenDim,
            spatial: spatial,
            w1Path: paths.w1,
            w3Path: paths.w3!,
            w2Path: paths.w2
        )
        let output = try! graph.add("\(prefix)_res2_out", x: residual1, y: ffn)
        _ = try! graph.output(output, name: "out")
        return graph
    }
}

private extension ANEGraph {
    mutating func constWeight128(_ name: String, shape: ANEShape, blobPath: String) throws -> Int {
        try constWeight(name, shape: shape, blobPath: blobPath, offset: 64)
    }

    mutating func linear128(
        _ prefix: String,
        input: Int,
        inDim: Int,
        outDim: Int,
        spatial: Int,
        weightPath: String,
        biasPath: String? = nil
    ) throws -> Int {
        let weight = try constWeight128(
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

        let bias = try constWeight128(
            "\(prefix)_bias",
            shape: try ANEShape(channels: outDim, spatial: 1),
            blobPath: biasPath
        )
        return try add("\(prefix)_out", x: conv, y: bias)
    }

    mutating func rmsNorm128(
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
        let varEps = try add("\(prefix)_var_eps", x: ms, y: epsNode)
        let nhalf = try constScalar("\(prefix)_nhalf", -0.5)
        let invStd = try pow("\(prefix)_inv_std", base: varEps, exp: nhalf)
        let normalized = try mul("\(prefix)_normalized", x: input, y: invStd)
        let weight = try constWeight128(
            "\(prefix)_weight",
            shape: try ANEShape(channels: dim, spatial: 1),
            blobPath: weightPath
        )
        return try mul("\(prefix)_out", x: normalized, y: weight)
    }

    mutating func layerNorm128(
        _ prefix: String,
        input: Int,
        dim: Int,
        spatial: Int,
        eps: Float,
        gammaPath: String,
        betaPath: String
    ) throws -> Int {
        let mean = try reduceMean("\(prefix)_mean", input: input, axis: 1, keepDims: true)
        let centered = try sub("\(prefix)_centered", x: input, y: mean)
        let sq = try mul("\(prefix)_sq", x: centered, y: centered)
        let variance = try reduceMean("\(prefix)_var", input: sq, axis: 1, keepDims: true)
        let epsNode = try constScalar("\(prefix)_eps", eps)
        let varEps = try add("\(prefix)_var_eps", x: variance, y: epsNode)
        let nhalf = try constScalar("\(prefix)_nhalf", -0.5)
        let invStd = try pow("\(prefix)_inv_std", base: varEps, exp: nhalf)
        let normalized = try mul("\(prefix)_normalized", x: centered, y: invStd)
        let gamma = try constWeight128(
            "\(prefix)_gamma",
            shape: try ANEShape(channels: dim, spatial: 1),
            blobPath: gammaPath
        )
        let scaled = try mul("\(prefix)_scaled", x: normalized, y: gamma)
        let beta = try constWeight128(
            "\(prefix)_beta",
            shape: try ANEShape(channels: dim, spatial: 1),
            blobPath: betaPath
        )
        return try add("\(prefix)_out", x: scaled, y: beta)
    }

    mutating func gelu128(
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

    mutating func silu128(
        _ prefix: String,
        input: Int
    ) throws -> Int {
        let sigmoidNode = try sigmoid("\(prefix)_sigmoid", input: input)
        return try mul("\(prefix)_out", x: input, y: sigmoidNode)
    }

    mutating func ffn128(
        _ prefix: String,
        input: Int,
        inDim: Int,
        hiddenDim: Int,
        spatial: Int,
        w1Path: String,
        b1Path: String?,
        w2Path: String,
        b2Path: String?,
        activation: Activation
    ) throws -> Int {
        let up = try linear128(
            "\(prefix)_up",
            input: input,
            inDim: inDim,
            outDim: hiddenDim,
            spatial: spatial,
            weightPath: w1Path,
            biasPath: b1Path
        )
        let activated: Int
        switch activation {
        case .gelu:
            activated = try gelu128("\(prefix)_act", input: up)
        case .silu:
            activated = try silu128("\(prefix)_act", input: up)
        case .relu:
            activated = try relu("\(prefix)_act", input: up)
        }
        return try linear128(
            "\(prefix)_down",
            input: activated,
            inDim: hiddenDim,
            outDim: inDim,
            spatial: spatial,
            weightPath: w2Path,
            biasPath: b2Path
        )
    }

    mutating func swigluFFN128(
        _ prefix: String,
        input: Int,
        inDim: Int,
        hiddenDim: Int,
        spatial: Int,
        w1Path: String,
        w3Path: String,
        w2Path: String
    ) throws -> Int {
        let gateLinear = try linear128(
            "\(prefix)_gate",
            input: input,
            inDim: inDim,
            outDim: hiddenDim,
            spatial: spatial,
            weightPath: w1Path
        )
        let gate = try silu128("\(prefix)_gate_act", input: gateLinear)
        let up = try linear128(
            "\(prefix)_up",
            input: input,
            inDim: inDim,
            outDim: hiddenDim,
            spatial: spatial,
            weightPath: w3Path
        )
        let gated = try mul("\(prefix)_gated", x: gate, y: up)
        return try linear128(
            "\(prefix)_down",
            input: gated,
            inDim: hiddenDim,
            outDim: inDim,
            spatial: spatial,
            weightPath: w2Path
        )
    }

    mutating func causalAttention128(
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
        let mask = try constWeight128(
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
