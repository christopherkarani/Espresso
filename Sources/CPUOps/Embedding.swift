public enum Embedding {
    /// Channel-first lookup: output[d*seq + t] = embedding[tok*dim + d].
    /// `vocabSize` is used to validate token ids before indexing.
    public static func lookup(
        output: UnsafeMutablePointer<Float>,
        embedding: UnsafePointer<Float>,
        tokens: UnsafePointer<UInt16>,
        vocabSize: Int,
        dim: Int,
        seqLen: Int
    ) {
        precondition(vocabSize > 0)
        precondition(dim > 0)
        precondition(seqLen > 0)

        for t in 0..<seqLen {
            let token = Int(tokens[t])
            precondition(token < vocabSize, "Token id out of range")
            for d in 0..<dim {
                output[d * seqLen + t] = embedding[token * dim + d]
            }
        }
    }

    /// Channel-first backward: dEmbedding[tok*dim + d] += dx[d*seq + t].
    /// `vocabSize` is used to validate token ids before indexing.
    /// ACCUMULATES - does not zero dEmbedding first.
    public static func backward(
        dEmbedding: UnsafeMutablePointer<Float>,
        dx: UnsafePointer<Float>,
        tokens: UnsafePointer<UInt16>,
        vocabSize: Int,
        dim: Int,
        seqLen: Int
    ) {
        precondition(vocabSize > 0)
        precondition(dim > 0)
        precondition(seqLen > 0)

        for t in 0..<seqLen {
            let token = Int(tokens[t])
            precondition(token < vocabSize, "Token id out of range")
            for d in 0..<dim {
                dEmbedding[token * dim + d] += dx[d * seqLen + t]
            }
        }
    }
}
