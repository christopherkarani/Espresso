/// A named connection point on the graph boundary (input or output).
///
/// For inputs: name becomes the MIL function parameter name.
/// For outputs: name becomes the MIL return tuple element name.
/// Output names MUST be alphabetically sorted (ANE constraint #3).
public struct GraphPort: Sendable, Equatable {
    /// The MIL-visible name of this port.
    public var name: String

    /// Index of the node this port connects to.
    public var nodeIndex: Int

    public init(name: String, nodeIndex: Int) {
        self.name = name
        self.nodeIndex = nodeIndex
    }
}
