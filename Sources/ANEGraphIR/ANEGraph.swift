/// A computation graph for ANE MIL programs.
///
/// Nodes are stored in a flat array and reference each other by index.
/// This design is cache-friendly and allows simple topological sort.
///
/// Usage:
/// ```swift
/// var g = ANEGraph()
/// let x = g.addNode(ANENode(op: .input, name: "x", dtype: .fp16,
///                            shape: ANEShape(channels: 768, spatial: 256)))
/// let y = g.addNode(ANENode(op: .relu, name: "y", dtype: .fp16,
///                            shape: ANEShape(channels: 768, spatial: 256),
///                            inputs: [x], isOutput: true))
/// g.graphInputs = [GraphPort(name: "x", nodeIndex: x)]
/// g.graphOutputs = [GraphPort(name: "y", nodeIndex: y)]
/// let order = g.topoSort()  // [0, 1]
/// ```
public struct ANEGraph: Sendable {
    /// All nodes in the graph, indexed by position.
    public var nodes: [ANENode] = []

    /// Named graph inputs (become MIL function parameters).
    public var graphInputs: [GraphPort] = []

    /// Named graph outputs (become MIL return tuple elements).
    /// Must be alphabetically sorted by name for ANE compatibility.
    public var graphOutputs: [GraphPort] = []

    public init() {}

    /// Append a node to the graph. Returns its index.
    @discardableResult
    public mutating func addNode(_ node: ANENode) -> Int {
        let idx = nodes.count
        nodes.append(node)
        return idx
    }

    /// Number of live (non-eliminated) nodes.
    public var liveNodeCount: Int {
        nodes.reduce(into: 0) { count, node in
            if node.isLive {
                count += 1
            }
        }
    }

    /// Topological sort via Kahn's algorithm.
    ///
    /// Returns node indices in valid execution order (inputs before consumers),
    /// or nil if a cycle is detected (which shouldn't happen with normal builder usage).
    ///
    /// Only considers live nodes. Dead nodes (isLive == false) are skipped.
    public func topoSort() -> [Int]? {
        let n = nodes.count
        guard n > 0 else { return [] }

        // Compute in-degree for each live node.
        var inDegree = [Int](repeating: 0, count: n)
        var liveCount = 0

        for i in 0..<n where nodes[i].isLive {
            liveCount += 1
            for inputIdx in nodes[i].inputs {
                guard inputIdx >= 0, inputIdx < n, nodes[inputIdx].isLive else { continue }
                inDegree[i] += 1
            }
        }

        // Seed queue with zero in-degree live nodes.
        var queue: [Int] = []
        queue.reserveCapacity(liveCount)
        for i in 0..<n where nodes[i].isLive && inDegree[i] == 0 {
            queue.append(i)
        }

        // Process queue.
        var result: [Int] = []
        result.reserveCapacity(liveCount)
        var head = 0

        while head < queue.count {
            let current = queue[head]
            head += 1
            result.append(current)

            // Decrement in-degree by multiplicity for every consumer of current.
            for i in 0..<n where nodes[i].isLive {
                let matchCount = nodes[i].inputs.reduce(into: 0) { count, inputIdx in
                    if inputIdx == current {
                        count += 1
                    }
                }
                if matchCount > 0 {
                    inDegree[i] -= matchCount
                    if inDegree[i] == 0 {
                        queue.append(i)
                    }
                }
            }
        }

        // If we didn't visit all live nodes, there's a cycle.
        return result.count == liveCount ? result : nil
    }
}
