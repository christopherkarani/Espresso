import ANEGraphIR

public struct ANEValidationPass: Sendable {
    public init() {}

    public func run(on graph: ANEGraph) -> [ANEConstraint] {
        var diagnostics: [ANEConstraint] = []

        diagnostics.append(contentsOf: validateConcatBan(in: graph))
        diagnostics.append(contentsOf: validateUniformOutputSizes(in: graph))
        diagnostics.append(contentsOf: validateOutputOrdering(in: graph))
        diagnostics.append(contentsOf: validateMinimumIOSurfaceSize(in: graph))
        diagnostics.append(contentsOf: validateSRAMBudget(in: graph))
        diagnostics.append(contentsOf: validateSoftmaxDimensions(in: graph))

        return diagnostics
    }

    private func validateConcatBan(in graph: ANEGraph) -> [ANEConstraint] {
        graph.nodes.enumerated().compactMap { index, node in
            guard node.isLive, node.op == .concatBanned else { return nil }
            return ANEConstraint(
                id: 1,
                severity: .error,
                message: "concat is banned by the ANE compiler",
                nodeIndex: index
            )
        }
    }

    private func validateUniformOutputSizes(in graph: ANEGraph) -> [ANEConstraint] {
        let outputSizes = graph.graphOutputs.compactMap { port -> Int? in
            guard graph.nodes.indices.contains(port.nodeIndex) else { return nil }
            let node = graph.nodes[port.nodeIndex]
            return node.shape.channels * node.shape.spatial
        }

        guard let firstSize = outputSizes.first else { return [] }
        guard outputSizes.dropFirst().contains(where: { $0 != firstSize }) else { return [] }

        return [
            ANEConstraint(
                id: 2,
                severity: .warning,
                message: "graph outputs should use uniform channels * spatial buffer sizes",
                nodeIndex: nil
            )
        ]
    }

    private func validateOutputOrdering(in graph: ANEGraph) -> [ANEConstraint] {
        let outputNames = graph.graphOutputs.map(\.name)
        guard outputNames != outputNames.sorted() else { return [] }

        return [
            ANEConstraint(
                id: 3,
                severity: .warning,
                message: "graph outputs must be alphabetically sorted",
                nodeIndex: nil
            )
        ]
    }

    private func validateMinimumIOSurfaceSize(in graph: ANEGraph) -> [ANEConstraint] {
        var boundaryIndices: [Int] = []
        var seenIndices = Set<Int>()

        for (index, node) in graph.nodes.enumerated() where node.isLive && node.op == .input {
            if seenIndices.insert(index).inserted {
                boundaryIndices.append(index)
            }
        }

        for port in graph.graphOutputs where seenIndices.insert(port.nodeIndex).inserted {
            boundaryIndices.append(port.nodeIndex)
        }

        return boundaryIndices.compactMap { index in
            guard graph.nodes.indices.contains(index) else { return nil }
            let node = graph.nodes[index]
            guard !node.shape.meetsMinimumIOSurfaceSize(for: node.dtype) else { return nil }

            return ANEConstraint(
                id: 4,
                severity: .error,
                message: "input/output tensors must allocate at least 49,152 bytes",
                nodeIndex: index
            )
        }
    }

    private func validateSRAMBudget(in graph: ANEGraph) -> [ANEConstraint] {
        graph.nodes.enumerated().compactMap { index, node in
            guard node.isLive, node.op.isCompute, node.shape.exceedsSRAMBudget(for: .fp16) else { return nil }
            return ANEConstraint(
                id: 6,
                severity: .warning,
                message: "compute nodes should stay within the 32MB fp16 SRAM budget",
                nodeIndex: index
            )
        }
    }

    private func validateSoftmaxDimensions(in graph: ANEGraph) -> [ANEConstraint] {
        graph.nodes.enumerated().compactMap { index, node in
            guard node.isLive, node.op == .softmax else { return nil }
            guard case let .softmax(axis) = node.attrs, axis == 1 else { return nil }
            let channels = node.shape.channels
            guard !(channels > 0 && (channels & (channels - 1)) == 0) else { return nil }

            return ANEConstraint(
                id: 7,
                severity: .error,
                message: "softmax over axis 1 requires a power-of-two channel dimension",
                nodeIndex: index
            )
        }
    }
}
