import Foundation
import ANEGraphIR

public struct ANECodegen: Sendable {
    public init() {}

    public static func emit(_ graph: ANEGraph, functionName: String = "main") -> String {
        Emitter(graph: graph, functionName: functionName).emit()
    }
}

private struct Emitter: Sendable {
    private static let headerLines = [
        "program(1.3)",
        "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, {\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, {\"coremltools-version\", \"9.0\"}})]",
        "{",
    ]

    private let graph: ANEGraph
    private let functionName: String
    private let locale = Locale(identifier: "en_US_POSIX")

    init(graph: ANEGraph, functionName: String) {
        self.graph = graph
        self.functionName = functionName
    }

    private var outputPorts: [GraphPort] {
        let explicit = graph.graphOutputs
        if !explicit.isEmpty {
            return explicit
        }

        return graph.nodes.enumerated()
            .filter { $0.element.isLive && $0.element.isOutput }
            .map { GraphPort(name: $0.element.name, nodeIndex: $0.offset) }
    }

    private var inputPorts: [GraphPort] {
        let explicit = graph.graphInputs
        if !explicit.isEmpty {
            return explicit
        }

        return graph.nodes.enumerated()
            .filter { $0.element.isLive && $0.element.op == .input }
            .map { GraphPort(name: $0.element.name, nodeIndex: $0.offset) }
            .sorted { $0.name < $1.name }
    }

    func emit() -> String {
        guard graph.validate() == nil else {
            preconditionFailure("ANECodegen requires a valid graph")
        }
        let topoOrder = graph.nodes.indices.filter { graph.nodes[$0].isLive }

        let inputsByIndex = Dictionary(uniqueKeysWithValues: inputPorts.map { ($0.nodeIndex, $0) })
        let outputs = outputPorts
        precondition(!outputs.isEmpty, "ANECodegen requires at least one graph output")

        var lines = Self.headerLines
        let parameterList = inputPorts.map { port in
            let node = graph.nodes[port.nodeIndex]
            precondition(node.op == .input, "Graph input \(port.name) must reference an input node")
            return "\(tensorType(dtype: node.dtype, shape: node.shape)) \(port.name)"
        }.joined(separator: ", ")
        lines.append("    func \(functionName)<ios18>(\(parameterList)) {")

        var valueNames = [Int: String]()
        for port in inputPorts {
            valueNames[port.nodeIndex] = port.name
        }

        for nodeIndex in topoOrder {
            let node = graph.nodes[nodeIndex]
            guard node.isLive else { continue }

            switch node.op {
            case .input:
                guard valueNames[nodeIndex] != nil || inputsByIndex[nodeIndex] != nil else {
                    preconditionFailure("Live input node \(node.name) is missing a graph input port")
                }
                valueNames[nodeIndex] = valueNames[nodeIndex] ?? node.name
            case .const:
                lines.append(contentsOf: emitConst(node))
                valueNames[nodeIndex] = node.name
            case .conv1x1:
                lines.append(contentsOf: emitConv(node, valueNames: valueNames))
                valueNames[nodeIndex] = node.name
            case .matmul:
                lines.append(contentsOf: emitMatmul(node, valueNames: valueNames))
                valueNames[nodeIndex] = node.name
            case .add, .sub, .mul, .pow:
                lines.append(emitBinary(node, valueNames: valueNames))
                valueNames[nodeIndex] = node.name
            case .neg, .relu, .tanh, .sigmoid, .exp, .sqrt, .rsqrt, .identity:
                lines.append(emitUnary(node, valueNames: valueNames))
                valueNames[nodeIndex] = node.name
            case .softmax:
                lines.append(contentsOf: emitSoftmax(node, valueNames: valueNames))
                valueNames[nodeIndex] = node.name
            case .reduceMean, .reduceSum, .reduceMax:
                lines.append(contentsOf: emitReduce(node, valueNames: valueNames))
                valueNames[nodeIndex] = node.name
            case .reshape:
                lines.append(contentsOf: emitReshape(node, valueNames: valueNames))
                valueNames[nodeIndex] = node.name
            case .transpose:
                lines.append(contentsOf: emitTranspose(node, valueNames: valueNames))
                valueNames[nodeIndex] = node.name
            case .cast:
                lines.append(contentsOf: emitCast(node, valueNames: valueNames))
                valueNames[nodeIndex] = node.name
            case .slice:
                lines.append(contentsOf: emitSlice(node, valueNames: valueNames))
                valueNames[nodeIndex] = node.name
            case .sliceBySize:
                lines.append(contentsOf: emitSliceBySize(node, valueNames: valueNames))
                valueNames[nodeIndex] = node.name
            case .concat:
                lines.append(contentsOf: emitConcat(node, valueNames: valueNames))
                valueNames[nodeIndex] = node.name
            case .concatBanned:
                preconditionFailure("concatBanned nodes must not reach ANECodegen")
            }
        }

        var returnNames: [String] = []
        for port in outputs {
            let node = graph.nodes[port.nodeIndex]
            guard let sourceName = valueNames[port.nodeIndex] else {
                preconditionFailure("Output \(port.name) references an unevaluated node")
            }

            if sourceName == port.name {
                returnNames.append(sourceName)
                continue
            }

            lines.append(
                "        \(tensorType(dtype: node.dtype, shape: node.shape)) \(port.name) = identity(x=\(sourceName))[name=string(\"\(port.name)\")];"
            )
            returnNames.append(port.name)
        }

        lines.append("    } -> (\(returnNames.joined(separator: ", ")));")
        lines.append("}")
        return lines.joined(separator: "\n") + "\n"
    }

    private func emitConst(_ node: ANENode) -> [String] {
        switch node.attrs {
        case .scalar(let value):
            return [
                "        \(node.dtype.description) \(node.name) = const()[name=string(\"\(node.name)\"), val=\(node.dtype.description)(\(formatFloat(value)))];"
            ]
        case .weight(let blobPath, let offset):
            let type = tensorType(dtype: node.dtype, shape: node.shape)
            return [
                "        \(type) \(node.name) = const()[name=string(\"\(node.name)\"), val=\(type)(BLOBFILE(path=string(\"\(blobPath)\"), offset=uint64(\(offset))))];"
            ]
        case .intTensor(let values):
            return [
                "        tensor<int32, [\(values.count)]> \(node.name) = const()[name=string(\"\(node.name)\"), val=tensor<int32, [\(values.count)]>([\(values.map(String.init).joined(separator: ", "))])];"
            ]
        case .boolValue(let value):
            return [
                "        bool \(node.name) = const()[name=string(\"\(node.name)\"), val=bool(\(value ? "true" : "false"))];"
            ]
        case .none:
            preconditionFailure("Const node \(node.name) is missing attrs")
        case .conv, .matmul, .transpose, .reduce, .softmax, .cast, .slice, .sliceBySize, .concat:
            preconditionFailure("Const node \(node.name) has invalid attrs \(node.attrs)")
        }
    }

    private func emitConv(_ node: ANENode, valueNames: [Int: String]) -> [String] {
        guard case .conv(let groups, let biasInput) = node.attrs else {
            preconditionFailure("Conv node \(node.name) must have conv attrs")
        }
        precondition(node.inputs.count >= 2, "Conv node \(node.name) requires x and weight inputs")

        let xName = requireValueName(for: node.inputs[0], valueNames: valueNames, nodeName: node.name)
        let weightName = requireValueName(for: node.inputs[1], valueNames: valueNames, nodeName: node.name)

        let ptName = "\(node.name)_pt"
        let stName = "\(node.name)_st"
        let pdName = "\(node.name)_pd"
        let dlName = "\(node.name)_dl"
        let grName = "\(node.name)_gr"
        let convOutputName = biasInput == nil ? node.name : "\(node.name)_core"

        var lines = [
            "        string \(ptName) = const()[name=string(\"\(ptName)\"), val=string(\"valid\")];",
            "        tensor<int32, [2]> \(stName) = const()[name=string(\"\(stName)\"), val=tensor<int32, [2]>([1, 1])];",
            "        tensor<int32, [4]> \(pdName) = const()[name=string(\"\(pdName)\"), val=tensor<int32, [4]>([0, 0, 0, 0])];",
            "        tensor<int32, [2]> \(dlName) = const()[name=string(\"\(dlName)\"), val=tensor<int32, [2]>([1, 1])];",
            "        int32 \(grName) = const()[name=string(\"\(grName)\"), val=int32(\(groups))];",
            "        \(tensorType(dtype: node.dtype, shape: node.shape)) \(convOutputName) = conv(dilations=\(dlName), groups=\(grName), pad=\(pdName), pad_type=\(ptName), strides=\(stName), weight=\(weightName), x=\(xName))[name=string(\"\(convOutputName)\")];",
        ]

        if let biasInput {
            precondition(node.inputs.indices.contains(biasInput), "Conv bias input for \(node.name) is out of range")
            let biasName = requireValueName(for: node.inputs[biasInput], valueNames: valueNames, nodeName: node.name)
            lines.append(
                "        \(tensorType(dtype: node.dtype, shape: node.shape)) \(node.name) = add(x=\(convOutputName), y=\(biasName))[name=string(\"\(node.name)\")];"
            )
        }

        return lines
    }

    private func emitMatmul(_ node: ANENode, valueNames: [Int: String]) -> [String] {
        guard case .matmul(let transposeX, let transposeY) = node.attrs else {
            preconditionFailure("Matmul node \(node.name) must have matmul attrs")
        }
        precondition(node.inputs.count == 2, "Matmul node \(node.name) requires two inputs")

        let xName = requireValueName(for: node.inputs[0], valueNames: valueNames, nodeName: node.name)
        let yName = requireValueName(for: node.inputs[1], valueNames: valueNames, nodeName: node.name)
        let txName = "\(node.name)_tx"
        let tyName = "\(node.name)_ty"

        return [
            "        bool \(txName) = const()[name=string(\"\(txName)\"), val=bool(\(transposeX ? "true" : "false"))];",
            "        bool \(tyName) = const()[name=string(\"\(tyName)\"), val=bool(\(transposeY ? "true" : "false"))];",
            "        \(tensorType(dtype: node.dtype, shape: node.shape)) \(node.name) = matmul(transpose_x=\(txName), transpose_y=\(tyName), x=\(xName), y=\(yName))[name=string(\"\(node.name)\")];",
        ]
    }

    private func emitBinary(_ node: ANENode, valueNames: [Int: String]) -> String {
        precondition(node.inputs.count == 2, "Binary node \(node.name) requires two inputs")
        let xName = requireValueName(for: node.inputs[0], valueNames: valueNames, nodeName: node.name)
        let yName = requireValueName(for: node.inputs[1], valueNames: valueNames, nodeName: node.name)
        return "        \(tensorType(dtype: node.dtype, shape: node.shape)) \(node.name) = \(milOpName(node.op))(x=\(xName), y=\(yName))[name=string(\"\(node.name)\")];"
    }

    private func emitUnary(_ node: ANENode, valueNames: [Int: String]) -> String {
        precondition(node.inputs.count == 1, "Unary node \(node.name) requires one input")
        let xName = requireValueName(for: node.inputs[0], valueNames: valueNames, nodeName: node.name)
        return "        \(tensorType(dtype: node.dtype, shape: node.shape)) \(node.name) = \(milOpName(node.op))(x=\(xName))[name=string(\"\(node.name)\")];"
    }

    private func emitSoftmax(_ node: ANENode, valueNames: [Int: String]) -> [String] {
        guard case .softmax(let axis) = node.attrs else {
            preconditionFailure("Softmax node \(node.name) must have softmax attrs")
        }
        precondition(node.inputs.count == 1, "Softmax node \(node.name) requires one input")

        let xName = requireValueName(for: node.inputs[0], valueNames: valueNames, nodeName: node.name)
        let axisName = "\(node.name)_axis"
        return [
            "        int32 \(axisName) = const()[name=string(\"\(axisName)\"), val=int32(\(axis))];",
            "        \(tensorType(dtype: node.dtype, shape: node.shape)) \(node.name) = softmax(axis=\(axisName), x=\(xName))[name=string(\"\(node.name)\")];",
        ]
    }

    private func emitReduce(_ node: ANENode, valueNames: [Int: String]) -> [String] {
        guard case .reduce(let axis, let keepDims) = node.attrs else {
            preconditionFailure("Reduce node \(node.name) must have reduce attrs")
        }
        precondition(node.inputs.count == 1, "Reduce node \(node.name) requires one input")

        let xName = requireValueName(for: node.inputs[0], valueNames: valueNames, nodeName: node.name)
        let axesName = "\(node.name)_axes"
        let keepDimsName = "\(node.name)_keep_dims"

        return [
            "        tensor<int32, [1]> \(axesName) = const()[name=string(\"\(axesName)\"), val=tensor<int32, [1]>([\(axis)])];",
            "        bool \(keepDimsName) = const()[name=string(\"\(keepDimsName)\"), val=bool(\(keepDims ? "true" : "false"))];",
            "        \(tensorType(dtype: node.dtype, shape: node.shape)) \(node.name) = \(milOpName(node.op))(x=\(xName), axes=\(axesName), keep_dims=\(keepDimsName))[name=string(\"\(node.name)\")];",
        ]
    }

    private func emitReshape(_ node: ANENode, valueNames: [Int: String]) -> [String] {
        precondition(!node.inputs.isEmpty, "Reshape node \(node.name) requires an input")
        let xName = requireValueName(for: node.inputs[0], valueNames: valueNames, nodeName: node.name)

        var lines: [String] = []
        let shapeName: String
        if node.inputs.count >= 2 {
            shapeName = requireValueName(for: node.inputs[1], valueNames: valueNames, nodeName: node.name)
        } else if case .intTensor(let dims) = node.attrs {
            shapeName = "\(node.name)_shape"
            lines.append(
                "        tensor<int32, [\(dims.count)]> \(shapeName) = const()[name=string(\"\(shapeName)\"), val=tensor<int32, [\(dims.count)]>([\(dims.map(String.init).joined(separator: ", "))])];"
            )
        } else {
            preconditionFailure("Reshape node \(node.name) requires a shape input or intTensor attrs")
        }

        lines.append(
            "        \(tensorType(dtype: node.dtype, shape: node.shape)) \(node.name) = reshape(shape=\(shapeName), x=\(xName))[name=string(\"\(node.name)\")];"
        )
        return lines
    }

    private func emitTranspose(_ node: ANENode, valueNames: [Int: String]) -> [String] {
        guard case .transpose(let perm) = node.attrs else {
            preconditionFailure("Transpose node \(node.name) must have transpose attrs")
        }
        precondition(node.inputs.count == 1, "Transpose node \(node.name) requires one input")

        let xName = requireValueName(for: node.inputs[0], valueNames: valueNames, nodeName: node.name)
        let permName = "\(node.name)_perm"
        return [
            "        tensor<int32, [\(perm.count)]> \(permName) = const()[name=string(\"\(permName)\"), val=tensor<int32, [\(perm.count)]>([\(perm.map(String.init).joined(separator: ", "))])];",
            "        \(tensorType(dtype: node.dtype, shape: node.shape)) \(node.name) = transpose(perm=\(permName), x=\(xName))[name=string(\"\(node.name)\")];",
        ]
    }

    private func emitCast(_ node: ANENode, valueNames: [Int: String]) -> [String] {
        guard case .cast(let target) = node.attrs else {
            preconditionFailure("Cast node \(node.name) must have cast attrs")
        }
        precondition(node.inputs.count == 1, "Cast node \(node.name) requires one input")

        let xName = requireValueName(for: node.inputs[0], valueNames: valueNames, nodeName: node.name)
        let dtypeName = "\(node.name)_dtype"
        return [
            "        string \(dtypeName) = const()[name=string(\"\(dtypeName)\"), val=string(\"\(target.description)\")];",
            "        \(tensorType(dtype: node.dtype, shape: node.shape)) \(node.name) = cast(dtype=\(dtypeName), x=\(xName))[name=string(\"\(node.name)\")];",
        ]
    }

    private func emitSlice(_ node: ANENode, valueNames: [Int: String]) -> [String] {
        guard case .slice(let begin, let end) = node.attrs else {
            preconditionFailure("Slice node \(node.name) must have slice attrs")
        }
        precondition(node.inputs.count == 1, "Slice node \(node.name) requires one input")
        precondition(begin.count == end.count, "Slice node \(node.name) begin/end rank mismatch")

        let xName = requireValueName(for: node.inputs[0], valueNames: valueNames, nodeName: node.name)
        let rank = begin.count
        let stride = Array(repeating: "1", count: rank)
        let falseMask = Array(repeating: "false", count: rank)
        let beginName = "\(node.name)_begin"
        let endName = "\(node.name)_end"
        let strideName = "\(node.name)_stride"
        let beginMaskName = "\(node.name)_begin_mask"
        let endMaskName = "\(node.name)_end_mask"
        let squeezeMaskName = "\(node.name)_squeeze_mask"

        return [
            "        tensor<int32, [\(rank)]> \(beginName) = const()[name=string(\"\(beginName)\"), val=tensor<int32, [\(rank)]>([\(begin.map(String.init).joined(separator: ", "))])];",
            "        tensor<int32, [\(rank)]> \(endName) = const()[name=string(\"\(endName)\"), val=tensor<int32, [\(rank)]>([\(end.map(String.init).joined(separator: ", "))])];",
            "        tensor<int32, [\(rank)]> \(strideName) = const()[name=string(\"\(strideName)\"), val=tensor<int32, [\(rank)]>([\(stride.joined(separator: ", "))])];",
            "        tensor<bool, [\(rank)]> \(beginMaskName) = const()[name=string(\"\(beginMaskName)\"), val=tensor<bool, [\(rank)]>([\(falseMask.joined(separator: ", "))])];",
            "        tensor<bool, [\(rank)]> \(endMaskName) = const()[name=string(\"\(endMaskName)\"), val=tensor<bool, [\(rank)]>([\(falseMask.joined(separator: ", "))])];",
            "        tensor<bool, [\(rank)]> \(squeezeMaskName) = const()[name=string(\"\(squeezeMaskName)\"), val=tensor<bool, [\(rank)]>([\(falseMask.joined(separator: ", "))])];",
            "        \(tensorType(dtype: node.dtype, shape: node.shape)) \(node.name) = slice_by_index(begin=\(beginName), begin_mask=\(beginMaskName), end=\(endName), end_mask=\(endMaskName), squeeze_mask=\(squeezeMaskName), stride=\(strideName), x=\(xName))[name=string(\"\(node.name)\")];",
        ]
    }

    private func emitSliceBySize(_ node: ANENode, valueNames: [Int: String]) -> [String] {
        guard case .sliceBySize(let begin, let size) = node.attrs else {
            preconditionFailure("SliceBySize node \(node.name) must have sliceBySize attrs")
        }
        precondition(node.inputs.count == 1, "SliceBySize node \(node.name) requires one input")

        let xName = requireValueName(for: node.inputs[0], valueNames: valueNames, nodeName: node.name)
        let rank = begin.count
        let beginName = "\(node.name)_begin"
        let sizeName = "\(node.name)_size"

        return [
            "        tensor<int32, [\(rank)]> \(beginName) = const()[name=string(\"\(beginName)\"), val=tensor<int32, [\(rank)]>([\(begin.map(String.init).joined(separator: ", "))])];",
            "        tensor<int32, [\(rank)]> \(sizeName) = const()[name=string(\"\(sizeName)\"), val=tensor<int32, [\(rank)]>([\(size.map(String.init).joined(separator: ", "))])];",
            "        \(tensorType(dtype: node.dtype, shape: node.shape)) \(node.name) = slice_by_size(begin=\(beginName), size=\(sizeName), x=\(xName))[name=string(\"\(node.name)\")];",
        ]
    }

    private func emitConcat(_ node: ANENode, valueNames: [Int: String]) -> [String] {
        guard case .concat(let axis, let interleave) = node.attrs else {
            preconditionFailure("Concat node \(node.name) must have concat attrs")
        }
        precondition(!node.inputs.isEmpty, "Concat node \(node.name) requires inputs")

        let axisName = "\(node.name)_axis"
        let interleaveName = "\(node.name)_interleave"
        let values = node.inputs.map { requireValueName(for: $0, valueNames: valueNames, nodeName: node.name) }

        return [
            "        int32 \(axisName) = const()[name=string(\"\(axisName)\"), val=int32(\(axis))];",
            "        bool \(interleaveName) = const()[name=string(\"\(interleaveName)\"), val=bool(\(interleave ? "true" : "false"))];",
            "        \(tensorType(dtype: node.dtype, shape: node.shape)) \(node.name) = concat(axis=\(axisName), interleave=\(interleaveName), values=(\(values.joined(separator: ","))))[name=string(\"\(node.name)\")];",
        ]
    }

    private func requireValueName(for nodeIndex: Int, valueNames: [Int: String], nodeName: String) -> String {
        guard let valueName = valueNames[nodeIndex] else {
            preconditionFailure("Node \(nodeName) references unavailable input index \(nodeIndex)")
        }
        return valueName
    }

    private func milOpName(_ op: ANEOp) -> String {
        switch op {
        case .reduceMean:
            "reduce_mean"
        case .reduceSum:
            "reduce_sum"
        case .reduceMax:
            "reduce_max"
        case .conv1x1:
            "conv"
        case .slice:
            "slice_by_index"
        case .sliceBySize:
            "slice_by_size"
        case .concat:
            "concat"
        case .input, .const, .concatBanned:
            preconditionFailure("No MIL op mapping for \(op)")
        default:
            String(describing: op).lowercased()
        }
    }

    private func tensorType(dtype: ANEDType, shape: ANEShape) -> String {
        "tensor<\(dtype.description), [\(shape.dimensions.map(String.init).joined(separator: ", "))]>"
    }

    private func tensorValueLiteral(dtype: ANEDType, shape: ANEShape, values: [String]) -> String {
        "\(tensorType(dtype: dtype, shape: shape))([\(values.joined(separator: ", "))])"
    }

    private func formatFloat(_ value: Float) -> String {
        if value.isFinite && value.rounded() == value {
            return String(format: "%.1f", locale: locale, Double(value))
        }

        var formatted = String(format: "%.6f", locale: locale, Double(value))
        while formatted.contains(".") && formatted.last == "0" {
            formatted.removeLast()
        }
        if formatted.last == "." {
            formatted.append("0")
        }
        return formatted
    }
}
