import ANEGraphIR

extension ANEGraph {
    public mutating func conv1x1(
        _ name: String,
        input: Int,
        weight: Int,
        bias: Int? = nil,
        groups: Int,
        outShape: ANEShape
    ) throws -> Int {
        precondition(nodes.indices.contains(input), "Invalid input index \(input) for \(name)")
        precondition(nodes.indices.contains(weight), "Invalid weight index \(weight) for \(name)")
        let inputs = [input, weight] + (bias.map { [$0] } ?? [])
        return try addNode(
            ANENode(
                op: .conv1x1,
                name: name,
                dtype: nodes[input].dtype,
                shape: outShape,
                inputs: inputs,
                attrs: .conv(groups: groups, biasInput: bias == nil ? nil : 2)
            )
        )
    }

    public mutating func sliceBySize(
        _ name: String,
        input: Int,
        begin: [Int],
        size: [Int],
        outShape: ANEShape
    ) throws -> Int {
        precondition(nodes.indices.contains(input), "Invalid input index \(input) for \(name)")
        let inputNode = nodes[input]
        return try addNode(
            ANENode(
                op: .sliceBySize,
                name: name,
                dtype: inputNode.dtype,
                shape: outShape,
                inputs: [input],
                attrs: .sliceBySize(begin: begin, size: size)
            )
        )
    }

    public mutating func concat(
        _ name: String,
        values: [Int],
        axis: Int,
        interleave: Bool,
        outShape: ANEShape
    ) throws -> Int {
        precondition(!values.isEmpty)
        precondition(nodes.indices.contains(values[0]), "Invalid first concat input for \(name)")
        let dtype = nodes[values[0]].dtype
        for value in values.dropFirst() {
            precondition(nodes.indices.contains(value), "Invalid concat input index \(value) for \(name)")
        }
        return try addNode(
            ANENode(
                op: .concat,
                name: name,
                dtype: dtype,
                shape: outShape,
                inputs: values,
                attrs: .concat(axis: axis, interleave: interleave)
            )
        )
    }
}
