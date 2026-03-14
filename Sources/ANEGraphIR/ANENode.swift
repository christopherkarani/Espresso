/// A single node in the ANE computation graph.
///
/// Nodes reference their inputs by integer index into the graph's node array.
/// This flat-array-with-indices design (instead of pointers) is cache-friendly
/// and proven in production by the Orion reference implementation.
public struct ANENode: Sendable, Equatable {
    /// The operation this node performs.
    public var op: ANEOp

    /// Unique name within the graph. Used as the MIL SSA variable name.
    /// Must be unique — collisions cause MIL compile failures.
    public var name: String

    /// Output data type of this node.
    public var dtype: ANEDType

    /// Output tensor shape.
    public var shape: ANEShape

    /// Indices of input nodes in the graph's node array.
    /// Order matters: conv expects [x, weight] or [x, weight, bias].
    public var inputs: [Int]

    /// Op-specific attributes (weight paths, transpose perms, etc.)
    public var attrs: ANEAttrs

    /// Whether this node is a graph output (included in return tuple).
    public var isOutput: Bool

    /// Whether this node is live (reachable from outputs).
    /// Dead code elimination sets this to false for unreachable nodes.
    public var isLive: Bool

    public init(
        op: ANEOp,
        name: String,
        dtype: ANEDType,
        shape: ANEShape,
        inputs: [Int] = [],
        attrs: ANEAttrs = .none,
        isOutput: Bool = false,
        isLive: Bool = true
    ) {
        self.op = op
        self.name = name
        self.dtype = dtype
        self.shape = shape
        self.inputs = inputs
        self.attrs = attrs
        self.isOutput = isOutput
        self.isLive = isLive
    }
}
