import Accelerate

public enum SiLU {
    /// silu(x) = x / (1 + exp(-x)) = x * sigmoid(x)
    @inlinable
    public static func forward(_ x: Float) -> Float {
        x / (1.0 + expf(-x))
    }

    /// silu'(x) = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
    @inlinable
    public static func backward(_ x: Float) -> Float {
        let s = 1.0 / (1.0 + expf(-x))
        return s * (1.0 + x * (1.0 - s))
    }
}
