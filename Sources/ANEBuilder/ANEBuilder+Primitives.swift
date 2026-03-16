import ANEGraphIR

extension ANEGraph {
    public mutating func input(
        _ name: String,
        dtype: ANEDType,
        shape: ANEShape
    ) throws -> Int {
        let nodeIndex = try addNode(
            ANENode(op: .input, name: name, dtype: dtype, shape: shape)
        )
        var ports = graphInputs
        ports.append(GraphPort(name: name, nodeIndex: nodeIndex))
        try setGraphInputs(ports)
        return nodeIndex
    }

    public mutating func constScalar(
        _ name: String,
        _ value: Float,
        dtype: ANEDType = .fp16
    ) throws -> Int {
        try addNode(
            ANENode(
                op: .const,
                name: name,
                dtype: dtype,
                shape: try ANEShape(channels: 1, spatial: 1),
                attrs: .scalar(value)
            )
        )
    }

    public mutating func constWeight(
        _ name: String,
        shape: ANEShape,
        blobPath: String,
        offset: UInt64 = 64
    ) throws -> Int {
        try addNode(
            ANENode(
                op: .const,
                name: name,
                dtype: .fp16,
                shape: shape,
                attrs: .weight(blobPath: blobPath, offset: offset)
            )
        )
    }

    public mutating func constInt(
        _ name: String,
        values: [Int]
    ) throws -> Int {
        try addNode(
            ANENode(
                op: .const,
                name: name,
                dtype: .int32,
                shape: try ANEShape(channels: values.count, spatial: 1),
                attrs: .intTensor(values)
            )
        )
    }

    public mutating func conv1x1(
        _ name: String,
        input: Int,
        weight: Int,
        bias: Int? = nil,
        outShape: ANEShape
    ) throws -> Int {
        _ = try node(at: input, referencedBy: name)
        _ = try node(at: weight, referencedBy: name)
        let inputs = [input, weight] + (bias.map { [$0] } ?? [])
        return try addNode(
            ANENode(
                op: .conv1x1,
                name: name,
                dtype: try node(at: input, referencedBy: name).dtype,
                shape: outShape,
                inputs: inputs,
                attrs: .conv(groups: 1, biasInput: bias == nil ? nil : 2)
            )
        )
    }

    public mutating func matmul(
        _ name: String,
        x: Int,
        y: Int,
        transposeX: Bool = false,
        transposeY: Bool = false,
        outShape: ANEShape
    ) throws -> Int {
        let xNode = try node(at: x, referencedBy: name)
        _ = try node(at: y, referencedBy: name)
        return try addNode(
            ANENode(
                op: .matmul,
                name: name,
                dtype: xNode.dtype,
                shape: outShape,
                inputs: [x, y],
                attrs: .matmul(transposeX: transposeX, transposeY: transposeY)
            )
        )
    }

    public mutating func add(_ name: String, x: Int, y: Int) throws -> Int {
        try binaryElementwise(.add, name: name, x: x, y: y)
    }

    public mutating func sub(_ name: String, x: Int, y: Int) throws -> Int {
        try binaryElementwise(.sub, name: name, x: x, y: y)
    }

    public mutating func mul(_ name: String, x: Int, y: Int) throws -> Int {
        try binaryElementwise(.mul, name: name, x: x, y: y)
    }

    public mutating func pow(_ name: String, base: Int, exp: Int) throws -> Int {
        let baseNode = try node(at: base, referencedBy: name)
        _ = try node(at: exp, referencedBy: name)
        return try addNode(
            ANENode(
                op: .pow,
                name: name,
                dtype: baseNode.dtype,
                shape: baseNode.shape,
                inputs: [base, exp]
            )
        )
    }

    public mutating func sigmoid(_ name: String, input: Int) throws -> Int {
        try unarySameShape(.sigmoid, name: name, input: input)
    }

    public mutating func relu(_ name: String, input: Int) throws -> Int {
        try unarySameShape(.relu, name: name, input: input)
    }

    public mutating func tanh(_ name: String, input: Int) throws -> Int {
        try unarySameShape(.tanh, name: name, input: input)
    }

    public mutating func softmax(
        _ name: String,
        input: Int,
        axis: Int
    ) throws -> Int {
        let inputNode = try node(at: input, referencedBy: name)
        return try addNode(
            ANENode(
                op: .softmax,
                name: name,
                dtype: inputNode.dtype,
                shape: inputNode.shape,
                inputs: [input],
                attrs: .softmax(axis: axis)
            )
        )
    }

    public mutating func reduceSum(
        _ name: String,
        input: Int,
        axis: Int,
        keepDims: Bool
    ) throws -> Int {
        try reduction(
            .reduceSum,
            name: name,
            input: input,
            axis: axis,
            keepDims: keepDims
        )
    }

    public mutating func reduceMean(
        _ name: String,
        input: Int,
        axis: Int,
        keepDims: Bool
    ) throws -> Int {
        try reduction(
            .reduceMean,
            name: name,
            input: input,
            axis: axis,
            keepDims: keepDims
        )
    }

    public mutating func reduceMax(
        _ name: String,
        input: Int,
        axis: Int,
        keepDims: Bool
    ) throws -> Int {
        try reduction(
            .reduceMax,
            name: name,
            input: input,
            axis: axis,
            keepDims: keepDims
        )
    }

    public mutating func reshape(
        _ name: String,
        input: Int,
        shape: ANEShape
    ) throws -> Int {
        let inputNode = try node(at: input, referencedBy: name)
        return try addNode(
            ANENode(
                op: .reshape,
                name: name,
                dtype: inputNode.dtype,
                shape: shape,
                inputs: [input],
                attrs: .intTensor(shape.dimensions)
            )
        )
    }

    public mutating func transpose(
        _ name: String,
        input: Int,
        perm: [Int]
    ) throws -> Int {
        let inputNode = try node(at: input, referencedBy: name)
        precondition(perm.count == 4, "ANE transpose requires a 4D permutation")
        let inputDimensions = inputNode.shape.dimensions
        let outputDimensions = perm.map { inputDimensions[$0] }
        return try addNode(
            ANENode(
                op: .transpose,
                name: name,
                dtype: inputNode.dtype,
                shape: try Self.shape(from: outputDimensions),
                inputs: [input],
                attrs: .transpose(perm: perm)
            )
        )
    }

    public mutating func cast(
        _ name: String,
        input: Int,
        to dtype: ANEDType
    ) throws -> Int {
        let inputNode = try node(at: input, referencedBy: name)
        return try addNode(
            ANENode(
                op: .cast,
                name: name,
                dtype: dtype,
                shape: inputNode.shape,
                inputs: [input],
                attrs: .cast(target: dtype)
            )
        )
    }

    public mutating func output(
        _ input: Int,
        name: String
    ) throws -> Int {
        var outputNode = try node(at: input, referencedBy: name)
        outputNode.isOutput = true
        try replaceNode(at: input, with: outputNode)

        var ports = graphOutputs
        ports.append(GraphPort(name: name, nodeIndex: input))
        ports.sort { $0.name < $1.name }
        try setGraphOutputs(ports)
        return input
    }

    private func node(at index: Int, referencedBy name: String) throws -> ANENode {
        guard nodes.indices.contains(index) else {
            throw ANEGraphValidationError.invalidNodeInput(
                nodeName: name,
                inputIndex: 0,
                referencedIndex: index
            )
        }
        return nodes[index]
    }

    private static func shape(from dimensions: [Int]) throws -> ANEShape {
        precondition(dimensions.count == 4, "ANEGraph shapes must stay 4D")
        return try ANEShape(
            batch: dimensions[0],
            channels: dimensions[1],
            height: dimensions[2],
            spatial: dimensions[3]
        )
    }

    private mutating func binaryElementwise(
        _ op: ANEOp,
        name: String,
        x: Int,
        y: Int
    ) throws -> Int {
        let xNode = try node(at: x, referencedBy: name)
        _ = try node(at: y, referencedBy: name)
        return try addNode(
            ANENode(
                op: op,
                name: name,
                dtype: xNode.dtype,
                shape: xNode.shape,
                inputs: [x, y]
            )
        )
    }

    private mutating func unarySameShape(
        _ op: ANEOp,
        name: String,
        input: Int
    ) throws -> Int {
        let inputNode = try node(at: input, referencedBy: name)
        return try addNode(
            ANENode(
                op: op,
                name: name,
                dtype: inputNode.dtype,
                shape: inputNode.shape,
                inputs: [input]
            )
        )
    }

    private mutating func reduction(
        _ op: ANEOp,
        name: String,
        input: Int,
        axis: Int,
        keepDims: Bool
    ) throws -> Int {
        let inputNode = try node(at: input, referencedBy: name)
        let normalizedAxis = Self.normalize(axis: axis)
        var outputDimensions = inputNode.shape.dimensions
        if keepDims {
            outputDimensions[normalizedAxis] = 1
        } else {
            outputDimensions.remove(at: normalizedAxis)
            while outputDimensions.count < 4 {
                outputDimensions.append(1)
            }
        }

        return try addNode(
            ANENode(
                op: op,
                name: name,
                dtype: inputNode.dtype,
                shape: try Self.shape(from: outputDimensions),
                inputs: [input],
                attrs: .reduce(axis: normalizedAxis, keepDims: keepDims)
            )
        )
    }

    private static func normalize(axis: Int) -> Int {
        let normalized = axis < 0 ? axis + 4 : axis
        precondition((0..<4).contains(normalized), "ANE reductions require a valid 4D axis")
        return normalized
    }
}
