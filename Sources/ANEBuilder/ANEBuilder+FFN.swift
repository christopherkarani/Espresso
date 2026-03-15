import ANEGraphIR

extension ANEGraph {
    public mutating func ffn(
        _ prefix: String,
        input: Int,
        inDim: Int,
        hiddenDim: Int,
        spatial: Int,
        w1Path: String,
        b1Path: String? = nil,
        w2Path: String,
        b2Path: String? = nil,
        activation: Activation
    ) throws -> Int {
        let up = try linear(
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
            activated = try gelu("\(prefix)_act", input: up)
        case .silu:
            activated = try silu("\(prefix)_act", input: up)
        case .relu:
            activated = try relu("\(prefix)_act", input: up)
        }

        return try linear(
            "\(prefix)_down",
            input: activated,
            inDim: hiddenDim,
            outDim: inDim,
            spatial: spatial,
            weightPath: w2Path,
            biasPath: b2Path
        )
    }

    public mutating func swigluFFN(
        _ prefix: String,
        input: Int,
        inDim: Int,
        hiddenDim: Int,
        spatial: Int,
        w1Path: String,
        w3Path: String,
        w2Path: String
    ) throws -> Int {
        let gateLinear = try linear(
            "\(prefix)_gate",
            input: input,
            inDim: inDim,
            outDim: hiddenDim,
            spatial: spatial,
            weightPath: w1Path
        )
        let gate = try silu("\(prefix)_gate_act", input: gateLinear)
        let up = try linear(
            "\(prefix)_up",
            input: input,
            inDim: inDim,
            outDim: hiddenDim,
            spatial: spatial,
            weightPath: w3Path
        )
        let gated = try mul("\(prefix)_gated", x: gate, y: up)
        return try linear(
            "\(prefix)_down",
            input: gated,
            inDim: hiddenDim,
            outDim: inDim,
            spatial: spatial,
            weightPath: w2Path
        )
    }
}
