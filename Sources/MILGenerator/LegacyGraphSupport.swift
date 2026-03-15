import Foundation
import ANEGraphIR
import ANEBuilder
import ANECodegen
import ANEPasses

enum LegacyGraphSupport {
    static func emitGraph(_ build: (inout ANEGraph) throws -> Void) -> String {
        var graph = ANEGraph()
        do {
            try build(&graph)
            ANEOptimizationPipeline.optimize(&graph)
            return ANECodegen.emit(graph)
        } catch {
            preconditionFailure("Failed to build legacy graph: \(error)")
        }
    }

    @discardableResult
    static func input(
        _ graph: inout ANEGraph,
        name: String,
        channels: Int,
        spatial: Int,
        dtype: ANEDType = .fp16
    ) throws -> Int {
        try graph.input(name, dtype: dtype, shape: try ANEShape(channels: channels, spatial: spatial))
    }

    @discardableResult
    static func weight(
        _ graph: inout ANEGraph,
        name: String,
        channels: Int,
        inputChannels: Int,
        path: String
    ) throws -> Int {
        try graph.constWeight(
            name,
            shape: try ANEShape(batch: channels, channels: inputChannels, height: 1, spatial: 1),
            blobPath: path
        )
    }

    @discardableResult
    static func vectorWeight(
        _ graph: inout ANEGraph,
        name: String,
        channels: Int,
        path: String
    ) throws -> Int {
        try graph.constWeight(name, shape: try ANEShape(channels: channels, spatial: 1), blobPath: path)
    }

    @discardableResult
    static func scalar(
        _ graph: inout ANEGraph,
        name: String,
        value: Float
    ) throws -> Int {
        try graph.constScalar(name, value)
    }

    @discardableResult
    static func axisTensor(
        _ graph: inout ANEGraph,
        name: String,
        _ value: Int
    ) throws -> Int {
        try graph.constInt(name, values: [value])
    }

    @discardableResult
    static func boolScalar(
        _ graph: inout ANEGraph,
        name: String,
        _ value: Bool
    ) throws -> Int {
        try graph.addNode(
            ANENode(
                op: .const,
                name: name,
                dtype: .bool,
                shape: try ANEShape(channels: 1, spatial: 1),
                attrs: .boolValue(value)
            )
        )
    }

    @discardableResult
    static func intTensor(
        _ graph: inout ANEGraph,
        name: String,
        values: [Int]
    ) throws -> Int {
        try graph.constInt(name, values: values)
    }

    static func setOutputs(
        _ graph: inout ANEGraph,
        _ outputs: [(String, Int)]
    ) throws {
        try graph.setGraphOutputs(outputs.map { GraphPort(name: $0.0, nodeIndex: $0.1) })
    }

    @discardableResult
    static func rmsNorm(
        _ graph: inout ANEGraph,
        input: Int,
        dim: Int,
        spatial: Int,
        sq: String,
        axisName: String,
        keepDimsName: String,
        ss: String,
        invdName: String,
        ss2: String,
        epsName: String,
        ss3: String,
        nhalfName: String,
        rrms: String,
        xr: String,
        weightName: String,
        weightPath: String,
        output: String
    ) throws -> Int {
        let sqNode = try graph.mul(sq, x: input, y: input)
        _ = try axisTensor(&graph, name: axisName, 1)
        _ = try boolScalar(&graph, name: keepDimsName, true)
        let ssNode = try graph.reduceSum(ss, input: sqNode, axis: 1, keepDims: true)
        let invd = try scalar(&graph, name: invdName, value: 1.0 / Float(dim))
        let ss2Node = try graph.mul(ss2, x: ssNode, y: invd)
        let eps = try scalar(&graph, name: epsName, value: 0.00001)
        let ss3Node = try graph.add(ss3, x: ss2Node, y: eps)
        let nhalf = try scalar(&graph, name: nhalfName, value: -0.5)
        let rrmsNode = try graph.pow(rrms, base: ss3Node, exp: nhalf)
        let xrNode = try graph.mul(xr, x: input, y: rrmsNode)
        let weight = try vectorWeight(&graph, name: weightName, channels: dim, path: weightPath)
        return try graph.mul(output, x: xrNode, y: weight)
    }

    @discardableResult
    static func conv(
        _ graph: inout ANEGraph,
        name: String,
        input: Int,
        weightName: String,
        outChannels: Int,
        inChannelsPerGroup: Int,
        groups: Int = 1,
        spatial: Int,
        weightPath: String
    ) throws -> Int {
        let weight = try graph.constWeight(
            weightName,
            shape: try ANEShape(batch: outChannels, channels: inChannelsPerGroup, height: 1, spatial: 1),
            blobPath: weightPath
        )
        return try graph.conv1x1(
            name,
            input: input,
            weight: weight,
            groups: groups,
            outShape: try ANEShape(channels: outChannels, spatial: spatial)
        )
    }

    @discardableResult
    static func swigluFFN(
        _ graph: inout ANEGraph,
        input: Int,
        dim: Int,
        hidden: Int,
        spatial: Int,
        w1Name: String = "W1",
        w1Path: String,
        h1Name: String = "h1",
        w3Name: String = "W3",
        w3Path: String,
        h3Name: String = "h3",
        sigName: String = "sig",
        siluName: String = "silu",
        gateName: String = "gate",
        w2Name: String = "W2",
        w2Path: String,
        outputName: String = "y"
    ) throws -> Int {
        let h1 = try conv(&graph, name: h1Name, input: input, weightName: w1Name, outChannels: hidden, inChannelsPerGroup: dim, spatial: spatial, weightPath: w1Path)
        let h3 = try conv(&graph, name: h3Name, input: input, weightName: w3Name, outChannels: hidden, inChannelsPerGroup: dim, spatial: spatial, weightPath: w3Path)
        let sig = try graph.sigmoid(sigName, input: h1)
        let silu = try graph.mul(siluName, x: h1, y: sig)
        let gate = try graph.mul(gateName, x: silu, y: h3)
        return try conv(&graph, name: outputName, input: gate, weightName: w2Name, outChannels: dim, inChannelsPerGroup: hidden, spatial: spatial, weightPath: w2Path)
    }

    static func recurrentLayer(
        _ graph: inout ANEGraph,
        dim: Int,
        lane: Int,
        groups: Int,
        layerIndex: Int,
        prefix: String,
        inputX: Int,
        inputState: Int,
        useIndexedWeights: Bool = true,
        includeRMSNorm: Bool = true,
        weightPrefix: String? = nil,
        outputXName: String,
        outputStateName: String
    ) throws -> (x: Int, state: Int) {
        let chPerGroup = dim / groups
        let layerSuffix = useIndexedWeights ? "\(layerIndex)" : ""
        let sq = "\(prefix)sq"
        let ss = "\(prefix)ss"
        let ss2 = "\(prefix)ss2"
        let ss3 = "\(prefix)ss3"
        let rrms = "\(prefix)rrms"
        let xr = "\(prefix)xr"
        let rw = "\(prefix)rw"
        let xn: Int
        if includeRMSNorm {
            xn = try rmsNorm(
                &graph,
                input: inputX,
                dim: dim,
                spatial: lane,
                sq: sq,
                axisName: "\(prefix)raxCh",
                keepDimsName: "\(prefix)kd",
                ss: ss,
                invdName: "\(prefix)invd",
                ss2: ss2,
                epsName: "\(prefix)eps",
                ss3: ss3,
                nhalfName: "\(prefix)nhalf",
                rrms: rrms,
                xr: xr,
                weightName: rw,
                weightPath: "@model_path/weights/rwkv_rms\(layerSuffix).bin",
                output: "\(prefix)xn"
            )
        } else {
            xn = inputX
        }
        let stem = weightPrefix ?? ""
        let xMix = try conv(&graph, name: "\(prefix)x_mix", input: xn, weightName: "\(prefix)Wx", outChannels: dim, inChannelsPerGroup: chPerGroup, groups: groups, spatial: lane, weightPath: "@model_path/weights/\(stem)wx\(layerSuffix).bin")
        let sMix = try conv(&graph, name: "\(prefix)s_mix", input: inputState, weightName: "\(prefix)Ws", outChannels: dim, inChannelsPerGroup: chPerGroup, groups: groups, spatial: lane, weightPath: "@model_path/weights/\(stem)ws\(layerSuffix).bin")
        let carry = try conv(&graph, name: "\(prefix)carry", input: inputState, weightName: "\(prefix)Wd", outChannels: dim, inChannelsPerGroup: chPerGroup, groups: groups, spatial: lane, weightPath: "@model_path/weights/\(stem)wd\(layerSuffix).bin")
        let mixPre = try graph.add("\(prefix)mix_pre", x: xMix, y: sMix)
        let gate = try graph.sigmoid("\(prefix)gate", input: mixPre)
        let gatedCarry = try graph.mul("\(prefix)gated_carry", x: carry, y: gate)
        let state = try graph.add(outputStateName, x: xMix, y: gatedCarry)
        let proj = try conv(&graph, name: "\(prefix)proj", input: state, weightName: "\(prefix)Wo", outChannels: dim, inChannelsPerGroup: chPerGroup, groups: groups, spatial: lane, weightPath: "@model_path/weights/\(stem)wo\(layerSuffix).bin")
        let x = try graph.add(outputXName, x: inputX, y: proj)
        return (x, state)
    }
}
