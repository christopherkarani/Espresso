/// Data types supported by ANE MIL programs.
public enum ANEDType: Sendable, Equatable, CustomStringConvertible {
    case fp16
    case fp32
    case int32
    case bool

    /// Bytes per element.
    public var byteWidth: Int {
        switch self {
        case .fp16:
            2
        case .fp32, .int32:
            4
        case .bool:
            1
        }
    }

    /// MIL text representation.
    public var description: String {
        switch self {
        case .fp16:
            "fp16"
        case .fp32:
            "fp32"
        case .int32:
            "int32"
        case .bool:
            "bool"
        }
    }
}
