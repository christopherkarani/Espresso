import ANEGraphIR

public struct CastEliminationPass: Sendable, ANEPass {
    public init() {}

    public mutating func run(on graph: inout ANEGraph) -> Bool {
        var changed = false

        for index in graph.nodes.indices {
            let node = graph.nodes[index]
            guard node.isLive, node.op == .cast, node.inputs.count == 1 else { continue }

            let firstCastIndex = node.inputs[0]
            guard graph.nodes.indices.contains(firstCastIndex) else { continue }

            let firstCast = graph.nodes[firstCastIndex]
            guard firstCast.isLive, firstCast.op == .cast, firstCast.inputs.count == 1 else { continue }

            let sourceIndex = firstCast.inputs[0]
            guard graph.nodes.indices.contains(sourceIndex) else { continue }

            let sourceNode = graph.nodes[sourceIndex]
            guard isRoundTrip(secondCast: node, firstCast: firstCast, sourceNode: sourceNode) else { continue }

            let didRewrite = aneRewriteUses(in: &graph, replacing: index, with: sourceIndex)
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

    private func isRoundTrip(secondCast: ANENode, firstCast: ANENode, sourceNode: ANENode) -> Bool {
        guard
            case let .cast(intermediateDType) = firstCast.attrs,
            case let .cast(finalDType) = secondCast.attrs
        else {
            return false
        }

        let sourceDType = sourceNode.dtype
        let allowedPair = Set([sourceDType, intermediateDType]) == Set([ANEDType.fp16, ANEDType.fp32])

        return allowedPair
            && intermediateDType != sourceDType
            && firstCast.dtype == intermediateDType
            && finalDType == sourceDType
            && secondCast.dtype == sourceDType
    }
}
