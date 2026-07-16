import Accelerate

public enum RoPE {
    /// Row-major layout: data[t * nHeads * headDim + h * headDim + i]
    /// Llama-family/Qwen decode expects half-split rotation:
    /// (0, halfDim), (1, halfDim+1), ...
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

        let halfDim = headDim / 2
        var freqs = [Float](repeating: 0, count: halfDim)
        for idx in 0..<halfDim {
            freqs[idx] = 1.0 / powf(10000.0, Float(2 * idx) / Float(headDim))
        }

        for t in 0..<seqLen {
            for h in 0..<nHeads {
                for idx in 0..<halfDim {
                    let i0 = idx
                    let i1 = idx + halfDim
                    let value = Float(t) * freqs[idx]
                    let cosv = cosf(value)
                    let sinv = sinf(value)

                    let base = t * nHeads * headDim + h * headDim

                    let q0 = q[base + i0]
                    let q1 = q[base + i1]
                    q[base + i0] = q0 * cosv - q1 * sinv
                    q[base + i1] = q0 * sinv + q1 * cosv

                    let k0 = k[base + i0]
                    let k1 = k[base + i1]
                    k[base + i0] = k0 * cosv - k1 * sinv
                    k[base + i1] = k0 * sinv + k1 * cosv
                }
            }
        }
    }

    /// Single-token RoPE for decode step. Applies rotation at the given
    /// `position` to Q (`nHeads` heads) and K (`nKVHeads` heads).
    /// Row-major layout: `h * headDim + i`. Llama-family/Qwen decode
    /// expects half-split rotation: (0, halfDim), (1, halfDim+1), ...
    /// Mutates q and k IN PLACE.
    public static func applyDecodeStep(
        q: UnsafeMutablePointer<Float>,
        k: UnsafeMutablePointer<Float>,
        nHeads: Int,
        nKVHeads: Int,
        headDim: Int,
        position: Int,
        theta: Float = 10_000.0
    ) {
        precondition(nHeads > 0)
        precondition(nKVHeads > 0)
        precondition(headDim > 0)
        precondition(headDim % 2 == 0)
        precondition(position >= 0)

        let halfDim = headDim / 2
        var freqs = [Float](repeating: 0, count: halfDim)
        for idx in 0..<halfDim {
            freqs[idx] = 1.0 / powf(theta, Float(2 * idx) / Float(headDim))
        }

        // Rotate Q heads
        for h in 0..<nHeads {
            for idx in 0..<halfDim {
                let i0 = idx
                let i1 = idx + halfDim
                let angle = Float(position) * freqs[idx]
                let cosv = cosf(angle)
                let sinv = sinf(angle)

                let base = h * headDim
                let q0 = q[base + i0]
                let q1 = q[base + i1]
                q[base + i0] = q0 * cosv - q1 * sinv
                q[base + i1] = q0 * sinv + q1 * cosv
            }
        }

        // Rotate K heads (may be fewer due to GQA)
        for h in 0..<nKVHeads {
            for idx in 0..<halfDim {
                let i0 = idx
                let i1 = idx + halfDim
                let angle = Float(position) * freqs[idx]
                let cosv = cosf(angle)
                let sinv = sinf(angle)

                let base = h * headDim
                let k0 = k[base + i0]
                let k1 = k[base + i1]
                k[base + i0] = k0 * cosv - k1 * sinv
                k[base + i1] = k0 * sinv + k1 * cosv
            }
        }
    }

    /// Half-split rotation backward. Mutates dq and dk IN PLACE.
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

        let halfDim = headDim / 2
        var freqs = [Float](repeating: 0, count: halfDim)
        for idx in 0..<halfDim {
            freqs[idx] = 1.0 / powf(10000.0, Float(2 * idx) / Float(headDim))
        }

        for t in 0..<seqLen {
            for h in 0..<nHeads {
                for idx in 0..<halfDim {
                    let i0 = idx
                    let i1 = idx + halfDim
                    let value = Float(t) * freqs[idx]
                    let cosv = cosf(value)
                    let sinv = sinf(value)

                    let base = t * nHeads * headDim + h * headDim

                    let dq0 = dq[base + i0]
                    let dq1 = dq[base + i1]
                    dq[base + i0] = dq0 * cosv + dq1 * sinv
                    dq[base + i1] = -dq0 * sinv + dq1 * cosv

                    let dk0 = dk[base + i0]
                    let dk1 = dk[base + i1]
                    dk[base + i0] = dk0 * cosv + dk1 * sinv
                    dk[base + i1] = -dk0 * sinv + dk1 * cosv
                }
            }
        }
    }
}
