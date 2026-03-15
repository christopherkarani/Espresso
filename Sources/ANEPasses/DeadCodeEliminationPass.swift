import ANEGraphIR

public struct DeadCodeEliminationPass: Sendable, ANEPass {
    public init() {}

    public mutating func run(on graph: inout ANEGraph) -> Bool {
        var reachable = [Bool](repeating: false, count: graph.nodes.count)
        var stack: [Int] = []
        stack.reserveCapacity(graph.graphOutputs.count)

        for port in graph.graphOutputs where graph.nodes.indices.contains(port.nodeIndex) {
            stack.append(port.nodeIndex)
        }

        while let index = stack.popLast() {
            guard graph.nodes.indices.contains(index), !reachable[index] else { continue }
            reachable[index] = true

            for inputIndex in graph.nodes[index].inputs where graph.nodes.indices.contains(inputIndex) {
                stack.append(inputIndex)
            }
        }

        var changed = false

        for index in graph.nodes.indices where graph.nodes[index].isLive && !reachable[index] {
            aneReplaceNode(in: &graph, at: index) { node in
                node.isLive = false
                node.isOutput = false
            }
            changed = true
        }

        if changed {
            aneSynchronizeOutputFlags(in: &graph)
        }

        return changed
    }
}
