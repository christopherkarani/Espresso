import ANEGraphIR

public struct IdentityEliminationPass: Sendable, ANEPass {
    public init() {}

    public mutating func run(on graph: inout ANEGraph) -> Bool {
        var changed = false

        for index in graph.nodes.indices {
            let node = graph.nodes[index]
            guard node.isLive, node.inputs.count == 1 else { continue }

            let inputIndex = node.inputs[0]
            guard graph.nodes.indices.contains(inputIndex) else { continue }
            let inputNode = graph.nodes[inputIndex]

            guard isBypassable(node: node, inputNode: inputNode) else { continue }

            let didRewrite = aneRewriteUses(in: &graph, replacing: index, with: inputIndex)
            aneEliminateNode(in: &graph, at: index)
            if didRewrite || node.isLive {
                changed = true
            }
        }

        if changed {
            aneSynchronizeOutputFlags(in: &graph)
        }

        return changed
    }

    private func isBypassable(node: ANENode, inputNode: ANENode) -> Bool {
        switch node.op {
        case .identity:
            return true
        case .cast:
            guard case let .cast(target) = node.attrs else { return false }
            return target == inputNode.dtype && node.dtype == inputNode.dtype
        case .reshape:
            return node.shape == inputNode.shape && node.dtype == inputNode.dtype
        case .transpose:
            guard case let .transpose(perm) = node.attrs else { return false }
            return perm == [0, 1, 2, 3] && node.shape == inputNode.shape && node.dtype == inputNode.dtype
        default:
            return false
        }
    }
}
