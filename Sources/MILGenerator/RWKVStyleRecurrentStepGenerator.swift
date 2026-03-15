import Foundation
import ANETypes

public struct RWKVStyleRecurrentStepGenerator: MILProgramGenerator {
    public let laneSpatial: Int

    public init(laneSpatial: Int = 32) {
        precondition(laneSpatial > 0)
        self.laneSpatial = laneSpatial
    }

    public var inputBytes: Int { ModelConfig.dim * laneSpatial * 2 }

    public var inputByteSizes: [Int] {
        let bytes = ModelConfig.dim * laneSpatial * 2
        return [bytes, bytes]
    }

    public var outputByteSizes: [Int] {
        let bytes = ModelConfig.dim * laneSpatial * 2
        return [bytes, bytes]
    }

    public var milText: String {
        LegacyGraphSupport.emitGraph { graph in
            let x = try LegacyGraphSupport.input(&graph, name: "x", channels: ModelConfig.dim, spatial: laneSpatial)
            let stateIn = try LegacyGraphSupport.input(&graph, name: "stateIn", channels: ModelConfig.dim, spatial: laneSpatial)
            let layer = try LegacyGraphSupport.recurrentLayer(
                &graph,
                dim: ModelConfig.dim,
                lane: laneSpatial,
                groups: 1,
                layerIndex: 0,
                prefix: "",
                inputX: x,
                inputState: stateIn,
                useIndexedWeights: false,
                weightPrefix: "",
                outputXName: "xNext",
                outputStateName: "stateOut"
            )
            try LegacyGraphSupport.setOutputs(&graph, [("xNext", layer.x), ("stateOut", layer.state)])
        }
    }
}
