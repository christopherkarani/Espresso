import ANEGraphIR

extension ANEGraph {
    public mutating func loraLinear(
        _ prefix: String,
        input: Int,
        inDim: Int,
        outDim: Int,
        spatial: Int,
        weightPath: String,
        loraAName: String,
        loraBName: String,
        rank: Int,
        alpha: Float
    ) throws -> Int {
        let baseWeight = try constWeight(
            "\(prefix)_base_weight",
            shape: try ANEShape(batch: outDim, channels: inDim, height: 1, spatial: 1),
            blobPath: weightPath
        )
        let base = try conv1x1(
            "\(prefix)_base_conv",
            input: input,
            weight: baseWeight,
            bias: nil,
            outShape: try ANEShape(channels: outDim, spatial: spatial)
        )

        let loraA = try self.input(
            loraAName,
            dtype: .fp16,
            shape: try ANEShape(batch: 1, channels: 1, height: inDim, spatial: rank)
        )
        let loraB = try self.input(
            loraBName,
            dtype: .fp16,
            shape: try ANEShape(batch: 1, channels: 1, height: outDim, spatial: rank)
        )

        let xReshaped = try reshape(
            "\(prefix)_lora_input",
            input: input,
            shape: try ANEShape(batch: 1, channels: 1, height: inDim, spatial: spatial)
        )
        let lowRank = try matmul(
            "\(prefix)_lora_low_rank",
            x: loraA,
            y: xReshaped,
            transposeX: true,
            transposeY: false,
            outShape: try ANEShape(batch: 1, channels: 1, height: rank, spatial: spatial)
        )
        let projected = try matmul(
            "\(prefix)_lora_projected",
            x: loraB,
            y: lowRank,
            transposeX: false,
            transposeY: false,
            outShape: try ANEShape(batch: 1, channels: 1, height: outDim, spatial: spatial)
        )
        let loraReshaped = try reshape(
            "\(prefix)_lora_reshape",
            input: projected,
            shape: try ANEShape(channels: outDim, spatial: spatial)
        )
        let scale = try constScalar("\(prefix)_lora_scale", alpha / Float(rank))
        let scaled = try mul("\(prefix)_lora_scaled", x: loraReshaped, y: scale)
        return try add("\(prefix)_out", x: base, y: scaled)
    }
}
