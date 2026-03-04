import Accelerate

public enum RoPE {
    /// Row-major layout: data[t * nHeads * headDim + h * headDim + i]
    /// Mutates q and k IN PLACE.
    public static func apply(
        q: UnsafeMutablePointer<Float>,
        k: UnsafeMutablePointer<Float>,
        seqLen: Int,
        nHeads: Int,
        headDim: Int
    ) {
        precondition(seqLen > 0)
        precondition(nHeads > 0)
        precondition(headDim > 0)
        precondition(headDim % 2 == 0)

        for t in 0..<seqLen {
            for h in 0..<nHeads {
                for i in stride(from: 0, to: headDim, by: 2) {
                    let freq = 1.0 / powf(10000.0, Float(i) / Float(headDim))
                    let value = Float(t) * freq
                    let cosv = cosf(value)
                    let sinv = sinf(value)

                    let off = t * nHeads * headDim + h * headDim + i

                    let q0 = q[off]
                    let q1 = q[off + 1]
                    q[off] = q0 * cosv - q1 * sinv
                    q[off + 1] = q0 * sinv + q1 * cosv

                    let k0 = k[off]
                    let k1 = k[off + 1]
                    k[off] = k0 * cosv - k1 * sinv
                    k[off + 1] = k0 * sinv + k1 * cosv
                }
            }
        }
    }

    /// Transposed rotation backward. Mutates dq and dk IN PLACE.
    public static func backward(
        dq: UnsafeMutablePointer<Float>,
        dk: UnsafeMutablePointer<Float>,
        seqLen: Int,
        nHeads: Int,
        headDim: Int
    ) {
        precondition(seqLen > 0)
        precondition(nHeads > 0)
        precondition(headDim > 0)
        precondition(headDim % 2 == 0)

        for t in 0..<seqLen {
            for h in 0..<nHeads {
                for i in stride(from: 0, to: headDim, by: 2) {
                    let freq = 1.0 / powf(10000.0, Float(i) / Float(headDim))
                    let value = Float(t) * freq
                    let cosv = cosf(value)
                    let sinv = sinf(value)

                    let off = t * nHeads * headDim + h * headDim + i

                    let dq0 = dq[off]
                    let dq1 = dq[off + 1]
                    dq[off] = dq0 * cosv + dq1 * sinv
                    dq[off + 1] = -dq0 * sinv + dq1 * cosv

                    let dk0 = dk[off]
                    let dk1 = dk[off + 1]
                    dk[off] = dk0 * cosv + dk1 * sinv
                    dk[off + 1] = -dk0 * sinv + dk1 * cosv
                }
            }
        }
    }
}
