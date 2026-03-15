import Foundation
import ANETypes

public struct RWKVStyleTwoStepRecurrentGenerator: MILProgramGenerator {
    public let laneSpatial: Int

    public init(laneSpatial: Int = 32) {
        precondition(laneSpatial > 0)
        self.laneSpatial = laneSpatial
    }

    public var inputBytes: Int {
        inputByteSizes.reduce(0, +)
    }

    public var inputByteSizes: [Int] {
        let bytes = ModelConfig.dim * laneSpatial * 2
        return [bytes, bytes, bytes]
    }

    public var outputByteSizes: [Int] {
        let bytes = ModelConfig.dim * laneSpatial * 2
        return [bytes, bytes, bytes, bytes]
    }

    public var milText: String {
        LegacyGraphSupport.emitGraph { graph in
            let x0 = try LegacyGraphSupport.input(&graph, name: "x0", channels: ModelConfig.dim, spatial: laneSpatial)
            let x1 = try LegacyGraphSupport.input(&graph, name: "x1", channels: ModelConfig.dim, spatial: laneSpatial)
            let stateIn = try LegacyGraphSupport.input(&graph, name: "stateIn", channels: ModelConfig.dim, spatial: laneSpatial)
            let step0 = try LegacyGraphSupport.recurrentLayer(
                &graph, dim: ModelConfig.dim, lane: laneSpatial, groups: 1, layerIndex: 0,
                prefix: "s0_", inputX: x0, inputState: stateIn, useIndexedWeights: false, weightPrefix: "",
                outputXName: "x0Next", outputStateName: "stateMid"
            )
            let step1 = try LegacyGraphSupport.recurrentLayer(
                &graph, dim: ModelConfig.dim, lane: laneSpatial, groups: 1, layerIndex: 0,
                prefix: "s1_", inputX: x1, inputState: step0.state, useIndexedWeights: false, weightPrefix: "",
                outputXName: "x1Next", outputStateName: "stateOut"
            )
            try LegacyGraphSupport.setOutputs(&graph, [("x0Next", step0.x), ("x1Next", step1.x), ("stateMid", step0.state), ("stateOut", step1.state)])
        }
    }
}
