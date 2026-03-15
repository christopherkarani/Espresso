/// Operations supported by the ANE MIL compiler.
/// This enum is exhaustive — ANE rejects any op not in this list.
public enum ANEOp: Sendable, Equatable, CaseIterable {
    // I/O nodes
    case input
    case const

    // Linear algebra (conv1x1 is ~3x faster than matmul on ANE)
    case conv1x1
    case matmul

    // Elementwise arithmetic
    case add, sub, mul, neg

    // Activations
    case relu, tanh, sigmoid, softmax, exp

    // Math (rsqrt not available in MIL text — use pow(x, -0.5) instead)
    case pow, sqrt, rsqrt

    // Reductions (reduce_mean doesn't exist on ANE — use reduce_sum + scalar)
    case reduceMean, reduceSum, reduceMax

    // Shape manipulation
    case reshape, transpose, cast, slice, sliceBySize, concat, identity

    // Exists solely for the validation pass to detect and reject
    case concatBanned

    /// True for ops that perform computation (not I/O or structural).
    public var isCompute: Bool {
        switch self {
        case .input, .const:
            false
        default:
            true
        }
    }

    /// True for ops that reference weight data via BLOBFILE.
    public var isWeightOp: Bool {
        self == .const
    }

    /// True for ops that have no inputs (source nodes in the graph).
    public var isSource: Bool {
        switch self {
        case .input, .const:
            true
        default:
            false
        }
    }
}
