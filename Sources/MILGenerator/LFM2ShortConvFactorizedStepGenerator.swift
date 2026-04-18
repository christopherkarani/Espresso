import Foundation
import ANEGraphIR
import ANEBuilder

public struct LFM2ShortConvFactorizedStepGenerator: MILProgramGenerator {
    public let dim: Int
    public let laneSpatial: Int
    public let kernelWidth: Int
    public let groups: Int

    public init(
        dim: Int,
        laneSpatial: Int = 32,
        kernelWidth: Int = 3,
        groups: Int? = nil
    ) {
        precondition(dim > 0)
        precondition(laneSpatial > 0)
        precondition(kernelWidth > 1)
        let resolvedGroups = groups ?? dim
        precondition(resolvedGroups > 0)
        precondition(dim.isMultiple(of: resolvedGroups))
        self.dim = dim
        self.laneSpatial = laneSpatial
        self.kernelWidth = kernelWidth
        self.groups = resolvedGroups
    }

    public var stateTailSpatial: Int { kernelWidth - 1 }
    public var stateSpatial: Int { laneSpatial }
    public var inputChannelsPerGroup: Int { dim / groups }

    public var inputBytes: Int { dim * laneSpatial * 2 }

    public var inputByteSizes: [Int] {
        let xBytes = dim * laneSpatial * 2
        let stateBytes = dim * stateSpatial * 2
        return [xBytes, stateBytes]
    }

    public var outputByteSizes: [Int] {
        let xBytes = dim * laneSpatial * 2
        let stateBytes = dim * stateSpatial * 2
        return [xBytes, stateBytes]
    }

    public var milText: String {
        LegacyGraphSupport.emitGraph { graph in
            let x = try LegacyGraphSupport.input(
                &graph,
                name: "x",
                channels: dim,
                spatial: laneSpatial
            )
            let convStateIn = try LegacyGraphSupport.input(
                &graph,
                name: "convStateIn",
                channels: dim,
                spatial: stateSpatial
            )

            let convStateTail = try graph.sliceBySize(
                "convStateTail",
                input: convStateIn,
                begin: [0, 0, 0, laneSpatial - stateTailSpatial],
                size: [1, dim, 1, stateTailSpatial],
                outShape: try ANEShape(channels: dim, spatial: stateTailSpatial)
            )

            let historySpatial = laneSpatial + stateTailSpatial
            let historyShape = try ANEShape(channels: dim, spatial: historySpatial)
            let xHistory = try graph.concat(
                "xHistory",
                values: [convStateTail, x],
                axis: 3,
                interleave: false,
                outShape: historyShape
            )

            var tapOutputs: [Int] = []
            tapOutputs.reserveCapacity(kernelWidth)

            for tapIndex in 0..<kernelWidth {
                let tap = try graph.sliceBySize(
                    "tap\(tapIndex)",
                    input: xHistory,
                    begin: [0, 0, 0, tapIndex],
                    size: [1, dim, 1, laneSpatial],
                    outShape: try ANEShape(channels: dim, spatial: laneSpatial)
                )
                let weight = try graph.constWeight(
                    "lfm2_short_conv_tap\(tapIndex)",
                    shape: try ANEShape(
                        batch: dim,
                        channels: inputChannelsPerGroup,
                        height: 1,
                        spatial: 1
                    ),
                    blobPath: "@model_path/weights/lfm2_short_conv_tap\(tapIndex).bin"
                )
                let projected = try graph.addNode(
                    ANENode(
                        op: .conv1x1,
                        name: "tap\(tapIndex)_proj",
                        dtype: .fp16,
                        shape: try ANEShape(channels: dim, spatial: laneSpatial),
                        inputs: [tap, weight],
                        attrs: .conv(groups: groups, biasInput: nil)
                    )
                )
                tapOutputs.append(projected)
            }

            let xNext = try accumulatedSum(
                in: &graph,
                values: tapOutputs,
                outputShape: try ANEShape(channels: dim, spatial: laneSpatial)
            )

            try LegacyGraphSupport.setOutputs(
                &graph,
                [("xNext", xNext), ("convStateOut", x)]
            )
        }
    }

    private func accumulatedSum(
        in graph: inout ANEGraph,
        values: [Int],
        outputShape: ANEShape
    ) throws -> Int {
        precondition(!values.isEmpty)
        var running = values[0]
        for index in 1..<values.count {
            let sumName = index == values.count - 1 ? "xNext" : "tap_sum\(index)"
            running = try graph.addNode(
                ANENode(
                    op: .add,
                    name: sumName,
                    dtype: .fp16,
                    shape: outputShape,
                    inputs: [running, values[index]]
                )
            )
        }
        return running
    }
}
