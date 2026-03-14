public struct ANEConstraint: Sendable, Equatable {
    public enum Severity: Sendable, Equatable {
        case warning
        case error
    }

    public let id: Int
    public let severity: Severity
    public let message: String
    public let nodeIndex: Int?

    public init(id: Int, severity: Severity, message: String, nodeIndex: Int? = nil) {
        self.id = id
        self.severity = severity
        self.message = message
        self.nodeIndex = nodeIndex
    }
}
