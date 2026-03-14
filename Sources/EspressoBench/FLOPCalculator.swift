import ANETypes

enum FLOPCalculator {
    static func forwardPassFLOPs(
        dim: Int = ModelConfig.dim,
        hidden: Int = ModelConfig.hidden,
        seqLen: Int = ModelConfig.seqLen,
        heads: Int = ModelConfig.heads
    ) -> Double {
        let headDim = dim / heads
        let dimValue = Double(dim)
        let hiddenValue = Double(hidden)
        let seqValue = Double(seqLen)
        let headValue = Double(heads)
        let headDimValue = Double(headDim)

        let qkvProjections = 3.0 * dimValue * dimValue * seqValue * 2.0
        let attentionScores = headValue * seqValue * seqValue * headDimValue * 2.0
        let attentionValue = headValue * seqValue * seqValue * headDimValue * 2.0
        let outputProjection = dimValue * dimValue * seqValue * 2.0
        let swigluFFN = (
            (hiddenValue * dimValue * seqValue * 2.0) +
            (hiddenValue * dimValue * seqValue * 2.0) +
            (dimValue * hiddenValue * seqValue * 2.0)
        )
        let siluActivation = hiddenValue * seqValue * 5.0
        let softmax = headValue * seqValue * seqValue * 5.0

        return qkvProjections
            + attentionScores
            + attentionValue
            + outputProjection
            + swigluFFN
            + siluActivation
            + softmax
    }

    static func sustainedTFLOPS(flops: Double, latencyMs: Double) -> Double {
        guard latencyMs > 0 else { return 0 }
        return flops / (latencyMs / 1_000.0) / 1_000_000_000_000.0
    }

    static func aneUtilization(sustainedTFLOPS: Double, peakTFLOPS: Double = 18.0) -> Double {
        guard peakTFLOPS > 0 else { return 0 }
        return (sustainedTFLOPS / peakTFLOPS) * 100.0
    }
}
