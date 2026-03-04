import Accelerate
import ANETypes

public enum GradientScaling {
    /// In-place `buffer *= factor`.
    @inline(__always)
    public static func scale(_ buffer: borrowing TensorBuffer, by factor: Float) {
        precondition(buffer.count >= 0)
        var f = factor
        buffer.withUnsafeMutablePointer { ptr in
            vDSP_vsmul(ptr, 1, &f, ptr, 1, vDSP_Length(buffer.count))
        }
    }

    /// Scale all per-layer gradient buffers by `factor`.
    @inline(__always)
    public static func scaleLayer(
        Wq: borrowing TensorBuffer,
        Wk: borrowing TensorBuffer,
        Wv: borrowing TensorBuffer,
        Wo: borrowing TensorBuffer,
        W1: borrowing TensorBuffer,
        W2: borrowing TensorBuffer,
        W3: borrowing TensorBuffer,
        rmsAtt: borrowing TensorBuffer,
        rmsFfn: borrowing TensorBuffer,
        by factor: Float
    ) {
        scale(Wq, by: factor)
        scale(Wk, by: factor)
        scale(Wv, by: factor)
        scale(Wo, by: factor)
        scale(W1, by: factor)
        scale(W2, by: factor)
        scale(W3, by: factor)
        scale(rmsAtt, by: factor)
        scale(rmsFfn, by: factor)
    }
}

