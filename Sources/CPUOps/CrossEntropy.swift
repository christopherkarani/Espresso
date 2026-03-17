import Accelerate
import ANETypes

public enum CrossEntropy {
    public struct Workspace: ~Copyable {
        public let vocabSize: Int
        public let seqLen: Int
        fileprivate let transposed: UnsafeMutablePointer<Float>

        public init(vocabSize: Int, seqLen: Int) {
            precondition(vocabSize > 0)
            precondition(seqLen > 0)
            self.vocabSize = vocabSize
            self.seqLen = seqLen
            self.transposed = .allocate(capacity: vocabSize * seqLen)
        }

        deinit {
            transposed.deallocate()
        }
    }

    /// Column-major logits [vocab, seq]. Returns mean CE loss. Writes gradient into dlogits.
    public static func lossAndGradient(
        dlogits: UnsafeMutablePointer<Float>,
        logits: UnsafePointer<Float>,
        targets: UnsafePointer<TokenID>,
        vocabSize: Int,
        seqLen: Int
    ) -> Float {
        let workspace = Workspace(vocabSize: vocabSize, seqLen: seqLen)
        return lossAndGradient(
            dlogits: dlogits,
            logits: logits,
            targets: targets,
            vocabSize: vocabSize,
            seqLen: seqLen,
            workspace: workspace
        )
    }

    /// Column-major logits [vocab, seq]. Returns mean CE loss. Writes gradient into dlogits.
    /// Uses caller-provided workspace to avoid per-call heap allocation.
    public static func lossAndGradient(
        dlogits: UnsafeMutablePointer<Float>,
        logits: UnsafePointer<Float>,
        targets: UnsafePointer<TokenID>,
        vocabSize: Int,
        seqLen: Int,
        workspace: borrowing Workspace
    ) -> Float {
        precondition(vocabSize > 0)
        precondition(seqLen > 0)
        precondition(workspace.vocabSize == vocabSize)
        precondition(workspace.seqLen == seqLen)

        let buf = workspace.transposed

        // [V, S] -> [S, V]
        vDSP_mtrans(logits, 1, buf, 1, vDSP_Length(seqLen), vDSP_Length(vocabSize))

        var totalLoss: Float = 0
        var invS = 1.0 / Float(seqLen)

        for t in 0..<seqLen {
            let row = buf + (t * vocabSize)

            var maxv: Float = 0
            vDSP_maxv(row, 1, &maxv, vDSP_Length(vocabSize))

            var negMax = -maxv
            vDSP_vsadd(row, 1, &negMax, row, 1, vDSP_Length(vocabSize))

            var count32 = Int32(vocabSize)
            vvexpf(row, row, &count32)

            var sum: Float = 0
            vDSP_sve(row, 1, &sum, vDSP_Length(vocabSize))
            var invSum = 1.0 / sum
            vDSP_vsmul(row, 1, &invSum, row, 1, vDSP_Length(vocabSize))

            let tgt = Int(targets[t])
            precondition(tgt < vocabSize, "Target token id out of range")
            totalLoss -= logf(row[tgt] + 1e-10)
            row[tgt] -= 1.0
            vDSP_vsmul(row, 1, &invS, row, 1, vDSP_Length(vocabSize))
        }

        // [S, V] -> [V, S]
        vDSP_mtrans(buf, 1, dlogits, 1, vDSP_Length(vocabSize), vDSP_Length(seqLen))

        return totalLoss / Float(seqLen)
    }
}
