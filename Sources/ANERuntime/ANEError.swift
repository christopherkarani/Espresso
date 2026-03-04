import Foundation

public enum ANEError: Error, Sendable, CustomStringConvertible, LocalizedError {
    /// Swift-side argument validation failed before C interop call.
    case invalidArguments(String)
    /// ane_interop_compile returned NULL
    case compilationFailed
    /// ane_interop_eval returned false
    case evaluationFailed
    /// Compile count >= MAX_COMPILES — exec() restart needed
    case compileBudgetExhausted
    /// IOSurface allocation returned nil
    case surfaceAllocationFailed
    /// Requested IOSurface index is out of bounds for this kernel.
    case invalidSurfaceIndex(Int)
    /// Input surface unavailable for the requested index.
    case inputSurfaceUnavailable(Int)
    /// Output surface unavailable for the requested index.
    case outputSurfaceUnavailable(Int)

    public var description: String {
        switch self {
        case let .invalidArguments(message):
            return "Invalid ANE arguments: \(message)"
        case .compilationFailed:
            return "ANE kernel compilation failed"
        case .evaluationFailed:
            return "ANE kernel evaluation failed"
        case .compileBudgetExhausted:
            return "ANE compile budget exhausted — exec() restart required"
        case .surfaceAllocationFailed:
            return "IOSurface allocation failed"
        case let .invalidSurfaceIndex(index):
            return "ANE surface index out of range: \(index)"
        case let .inputSurfaceUnavailable(index):
            return "ANE input surface unavailable at index \(index)"
        case let .outputSurfaceUnavailable(index):
            return "ANE output surface unavailable at index \(index)"
        }
    }

    public var errorDescription: String? {
        description
    }
}
