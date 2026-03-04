public protocol MILProgramGenerator: Sendable {
    var milText: String { get }
    var inputBytes: Int { get }
    var outputByteSizes: [Int] { get }
}

public extension MILProgramGenerator {
    var outputBytes: Int {
        precondition(outputByteSizes.count == 1, "Multiple outputs require outputByteSizes")
        return outputByteSizes[0]
    }
}
