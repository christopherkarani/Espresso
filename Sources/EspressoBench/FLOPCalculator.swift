import ANETypes

enum FLOPCalculator {
    /// Total FLOPs for a single-layer transformer forward pass.
    /// Counts multiply-accumulate as 2 FLOPs (1 mul + 1 add).
    static func forwardPassFLOPs(
        dim: Int = ModelConfig.dim,
        hidden: Int = ModelConfig.hidden,
        seqLen: Int = ModelConfig.seqLen,
        heads: Int = ModelConfig.heads
    ) -> Double {
        let headDim = dim / heads

        // QKV projections: 3 x (dim x dim x seqLen x 2)
        let qkvFLOPs = 3.0 * Double(dim) * Double(dim) * Double(seqLen) * 2.0

        // Attention scores: Q @ K^T per head
        let attnScoreFLOPs = Double(heads) * Double(seqLen) * Double(seqLen) * Double(headDim) * 2.0

        // Attention x V per head
        let attnValueFLOPs = Double(heads) * Double(seqLen) * Double(headDim) * Double(seqLen) * 2.0

        // Output projection
        let outputProjFLOPs = Double(dim) * Double(dim) * Double(seqLen) * 2.0

        // FFN SwiGLU: W1, W3, W2
        let ffnFLOPs = 3.0 * Double(hidden) * Double(dim) * Double(seqLen) * 2.0

        // SiLU + Softmax (small but counted)
        let siluFLOPs = 5.0 * Double(hidden) * Double(seqLen)
        let softmaxFLOPs = 5.0 * Double(heads) * Double(seqLen) * Double(seqLen)

        return qkvFLOPs + attnScoreFLOPs + attnValueFLOPs + outputProjFLOPs
            + ffnFLOPs + siluFLOPs + softmaxFLOPs
    }

    /// Convert to TFLOPS given latency in milliseconds.
    static func sustainedTFLOPS(flops: Double, latencyMs: Double) -> Double {
        guard latencyMs > 0 else { return 0 }
        return flops / (latencyMs / 1000.0) / 1e12
    }

    /// ANE utilization percentage (M-series peak defaults to 18.0 TFLOPS).
    static func aneUtilization(sustainedTFLOPS: Double, peakTFLOPS: Double = 18.0) -> Double {
        guard peakTFLOPS > 0 else { return 0 }
        return (sustainedTFLOPS / peakTFLOPS) * 100.0
    }
}
