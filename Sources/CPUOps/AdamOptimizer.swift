import Accelerate

public enum AdamOptimizer {
    /// In-place Adam update with bias correction. Mutates w, m, v.
    public static func update(
        weights: UnsafeMutablePointer<Float>,
        gradients: UnsafePointer<Float>,
        m: UnsafeMutablePointer<Float>,
        v: UnsafeMutablePointer<Float>,
        count: Int,
        timestep: Int, // starts at 1, NOT 0
        lr: Float,
        beta1: Float,
        beta2: Float,
        eps: Float
    ) {
        precondition(count >= 0)
        precondition(timestep >= 1)

        let t = Float(timestep)
        let bc1 = 1.0 - powf(beta1, t)
        let bc2 = 1.0 - powf(beta2, t)

        for i in 0..<count {
            let g = gradients[i]
            m[i] = beta1 * m[i] + (1.0 - beta1) * g
            v[i] = beta2 * v[i] + (1.0 - beta2) * g * g
            let mh = m[i] / bc1
            let vh = v[i] / bc2
            weights[i] -= lr * mh / (sqrtf(vh) + eps)
        }
    }
}
