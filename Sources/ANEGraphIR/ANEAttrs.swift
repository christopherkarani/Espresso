/// Op-specific attributes for graph nodes.
///
/// Each ANE operation may need additional parameters beyond its inputs.
/// Using an enum (not a struct/protocol) ensures exhaustive pattern matching.
public enum ANEAttrs: Sendable, Equatable {
    /// No attributes needed (elementwise ops, activations).
    case none

    /// Conv parameters. ANE conv doesn't support inline bias — bias is a separate add.
    /// - groups: Always 1 for standard linear layers
    /// - biasInput: Index of bias node in inputs array, or nil
    case conv(groups: Int, biasInput: Int?)

    /// Matrix multiplication flags.
    /// ANE MIL requires these as named bool constants, not inline literals.
    case matmul(transposeX: Bool, transposeY: Bool)

    /// Transpose permutation. Always 4 elements for 4D tensors.
    /// Common: [0, 1, 3, 2] swaps height and spatial.
    case transpose(perm: [Int])

    /// Reduction parameters.
    /// - axis: Dimension to reduce (0=batch, 1=channels, 2=height, 3=spatial)
    /// - keepDims: If true, reduced dimension becomes 1 (not removed)
    case reduce(axis: Int, keepDims: Bool)

    /// Softmax axis. WARNING: softmax on non-power-of-2 channel dims causes InvalidMILProgram.
    case softmax(axis: Int)

    /// Cast to target data type.
    case cast(target: ANEDType)

    /// Slice with begin/end indices per dimension.
    case slice(begin: [Int], end: [Int])

    /// Scalar floating-point constant (for eps, scale factors, etc.)
    case scalar(Float)

    /// Weight constant via BLOBFILE reference.
    /// - blobPath: Path like "@model_path/weights/wq.bin"
    /// - offset: Byte offset into the BLOBFILE (typically 64 for data start)
    case weight(blobPath: String, offset: UInt64)

    /// Integer tensor constant (for shape arrays, perm arrays, axis arrays).
    case intTensor([Int])

    /// Boolean constant (for matmul transpose flags, reduce keepDims, etc.)
    case boolValue(Bool)
}
