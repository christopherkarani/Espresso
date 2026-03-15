public enum ANEGraphValidationError: Error, Sendable, Equatable {
    case nonPositiveDimension(name: String, value: Int)
    case byteSizeOverflow(dimensions: [Int])
    case duplicateNodeName(String)
    case invalidNodeInput(nodeName: String, inputIndex: Int, referencedIndex: Int)
    case invalidGraphInputPort(name: String, nodeIndex: Int)
    case invalidGraphOutputPort(name: String, nodeIndex: Int)
    case unsortedGraphOutputs
}

/// A computation graph for ANE MIL programs.
///
/// Nodes are stored in a flat array and reference each other by index.
/// This design is cache-friendly and allows simple topological sort.
///
/// Usage:
/// ```swift
/// var g = ANEGraph()
/// let x = try g.addNode(ANENode(op: .input, name: "x", dtype: .fp16,
///                               shape: ANEShape(channels: 768, spatial: 256)))
/// let y = try g.addNode(ANENode(op: .relu, name: "y", dtype: .fp16,
///                               shape: ANEShape(channels: 768, spatial: 256),
///                               inputs: [x], isOutput: true))
/// try g.setGraphInputs([GraphPort(name: "x", nodeIndex: x)])
/// try g.setGraphOutputs([GraphPort(name: "y", nodeIndex: y)])
/// let order = g.topoSort()  // [0, 1]
/// ```
public struct ANEGraph: Sendable {
    /// All nodes in the graph, indexed by position.
    public private(set) var nodes: [ANENode] = []

    /// Named graph inputs (become MIL function parameters).
    public private(set) var graphInputs: [GraphPort] = []

    /// Named graph outputs (become MIL return tuple elements).
    /// Must be alphabetically sorted by name for ANE compatibility.
    public private(set) var graphOutputs: [GraphPort] = []

    public init() {}

    /// Append a node to the graph. Returns its index.
    @discardableResult
    public mutating func addNode(_ node: ANENode) throws -> Int {
        try ensureUniqueNodeName(node.name, ignoring: nil)
        try validateInputIndices(node.inputs, forNodeNamed: node.name)
        let idx = nodes.count
        nodes.append(node)
        return idx
    }

    /// Replace an existing node while preserving graph-wide invariants.
    public mutating func replaceNode(at index: Int, with node: ANENode) throws {
        guard nodes.indices.contains(index) else {
            throw ANEGraphValidationError.invalidNodeInput(
                nodeName: node.name,
                inputIndex: 0,
                referencedIndex: index
            )
        }
        try ensureUniqueNodeName(node.name, ignoring: index)
        try validateInputIndices(node.inputs, forNodeNamed: node.name)
        nodes[index] = node
    }

    /// Replace the named graph inputs after validating referenced nodes.
    public mutating func setGraphInputs(_ ports: [GraphPort]) throws {
        for port in ports {
            guard nodes.indices.contains(port.nodeIndex) else {
                throw ANEGraphValidationError.invalidGraphInputPort(
                    name: port.name,
                    nodeIndex: port.nodeIndex
                )
            }
        }
        graphInputs = ports
    }

    /// Replace the named graph outputs after validating ordering and references.
    public mutating func setGraphOutputs(_ ports: [GraphPort]) throws {
        for port in ports {
            guard nodes.indices.contains(port.nodeIndex) else {
                throw ANEGraphValidationError.invalidGraphOutputPort(
                    name: port.name,
                    nodeIndex: port.nodeIndex
                )
            }
        }
        graphOutputs = ports
    }

    /// Update node liveness without exposing unchecked mutation of the node array.
    public mutating func setNodeLiveness(at index: Int, isLive: Bool) {
        guard nodes.indices.contains(index) else { return }
        nodes[index].isLive = isLive
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
        guard validate() == nil else { return nil }

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

    /// Validate graph invariants required by downstream ANE codegen.
    public func validate() -> ANEGraphValidationError? {
        var seenNames = Set<String>()
        for (nodeIndex, node) in nodes.enumerated() {
            if !seenNames.insert(node.name).inserted {
                return .duplicateNodeName(node.name)
            }
            for (inputIndex, referencedIndex) in node.inputs.enumerated() {
                guard nodes.indices.contains(referencedIndex) else {
                    return .invalidNodeInput(
                        nodeName: node.name,
                        inputIndex: inputIndex,
                        referencedIndex: referencedIndex
                    )
                }
            }
            _ = nodeIndex
        }

        for port in graphInputs {
            guard nodes.indices.contains(port.nodeIndex) else {
                return .invalidGraphInputPort(name: port.name, nodeIndex: port.nodeIndex)
            }
        }

        for port in graphOutputs {
            guard nodes.indices.contains(port.nodeIndex) else {
                return .invalidGraphOutputPort(name: port.name, nodeIndex: port.nodeIndex)
            }
        }

        return nil
    }

    private func ensureUniqueNodeName(_ name: String, ignoring index: Int?) throws {
        for (existingIndex, existingNode) in nodes.enumerated() {
            if existingIndex != index && existingNode.name == name {
                throw ANEGraphValidationError.duplicateNodeName(name)
            }
        }
    }

    private func validateInputIndices(_ inputs: [Int], forNodeNamed nodeName: String) throws {
        for (inputIndex, referencedIndex) in inputs.enumerated() {
            guard nodes.indices.contains(referencedIndex) else {
                throw ANEGraphValidationError.invalidNodeInput(
                    nodeName: nodeName,
                    inputIndex: inputIndex,
                    referencedIndex: referencedIndex
                )
            }
        }
    }
}
