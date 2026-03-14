import ANEGraphIR

public protocol ANEPass: Sendable {
    mutating func run(on graph: inout ANEGraph) -> Bool
}

@inline(__always)
func aneReplaceNode(in graph: inout ANEGraph, at index: Int, update: (inout ANENode) -> Void) {
    guard graph.nodes.indices.contains(index) else { return }

    let originalNode = graph.nodes[index]
    var updatedNode = originalNode
    update(&updatedNode)

    guard updatedNode != originalNode else { return }

    do {
        try graph.replaceNode(at: index, with: updatedNode)
    } catch {
        preconditionFailure("ANEPasses failed to replace node at index \(index): \(error)")
    }
}

@inline(__always)
func aneSetGraphOutputs(in graph: inout ANEGraph, to ports: [GraphPort]) {
    do {
        try graph.setGraphOutputs(ports)
    } catch {
        preconditionFailure("ANEPasses failed to update graph outputs: \(error)")
    }
}

func aneSynchronizeOutputFlags(in graph: inout ANEGraph) {
    let liveOutputIndices = Set<Int>(
        graph.graphOutputs.compactMap { port -> Int? in
            guard graph.nodes.indices.contains(port.nodeIndex) else { return nil }
            return graph.nodes[port.nodeIndex].isLive ? port.nodeIndex : nil
        }
    )

    for index in graph.nodes.indices {
        let shouldBeOutput = liveOutputIndices.contains(index)
        if graph.nodes[index].isOutput != shouldBeOutput {
            aneReplaceNode(in: &graph, at: index) { node in
                node.isOutput = shouldBeOutput
            }
        }
    }
}

@discardableResult
func aneRewriteUses(in graph: inout ANEGraph, replacing replacedIndex: Int, with replacementIndex: Int) -> Bool {
    guard replacedIndex != replacementIndex else { return false }

    var changed = false

    for index in graph.nodes.indices {
        var rewrittenInputs = graph.nodes[index].inputs
        var didRewriteNode = false

        for inputIndex in rewrittenInputs.indices where rewrittenInputs[inputIndex] == replacedIndex {
            rewrittenInputs[inputIndex] = replacementIndex
            didRewriteNode = true
        }

        if didRewriteNode {
            aneReplaceNode(in: &graph, at: index) { node in
                node.inputs = rewrittenInputs
            }
            changed = true
        }
    }

    var rewrittenOutputs = graph.graphOutputs
    var didRewriteOutputs = false

    for outputIndex in rewrittenOutputs.indices where rewrittenOutputs[outputIndex].nodeIndex == replacedIndex {
        rewrittenOutputs[outputIndex].nodeIndex = replacementIndex
        didRewriteOutputs = true
    }

    if didRewriteOutputs {
        aneSetGraphOutputs(in: &graph, to: rewrittenOutputs)
        changed = true
    }

    if changed {
        aneSynchronizeOutputFlags(in: &graph)
    }

    return changed
}

func aneEliminateNode(in graph: inout ANEGraph, at index: Int) {
    guard graph.nodes.indices.contains(index) else { return }

    aneReplaceNode(in: &graph, at: index) { node in
        node.isLive = false
        node.isOutput = false
    }
}
