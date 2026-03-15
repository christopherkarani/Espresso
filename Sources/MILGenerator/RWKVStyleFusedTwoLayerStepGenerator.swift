import Foundation
import ANETypes

public struct RWKVStyleFusedTwoLayerStepGenerator: MILProgramGenerator {
    public let laneSpatial: Int

    public init(laneSpatial: Int = 32) {
        precondition(laneSpatial > 0)
        self.laneSpatial = laneSpatial
    }

    public var inputBytes: Int { ModelConfig.dim * laneSpatial * 2 }
    public var inputByteSizes: [Int] { let b = ModelConfig.dim * laneSpatial * 2; return [b, b, b] }
    public var outputByteSizes: [Int] { let b = ModelConfig.dim * laneSpatial * 2; return [b, b, b] }

    public var milText: String {
        LegacyGraphSupport.emitGraph { graph in
            let x = try LegacyGraphSupport.input(&graph, name: "x", channels: ModelConfig.dim, spatial: laneSpatial)
            let stateIn0 = try LegacyGraphSupport.input(&graph, name: "stateIn0", channels: ModelConfig.dim, spatial: laneSpatial)
            let stateIn1 = try LegacyGraphSupport.input(&graph, name: "stateIn1", channels: ModelConfig.dim, spatial: laneSpatial)
            let l0 = try LegacyGraphSupport.recurrentLayer(&graph, dim: ModelConfig.dim, lane: laneSpatial, groups: 1, layerIndex: 0, prefix: "l0_", inputX: x, inputState: stateIn0, weightPrefix: "", outputXName: "l0_xNext", outputStateName: "stateOut0")
            let l1 = try LegacyGraphSupport.recurrentLayer(&graph, dim: ModelConfig.dim, lane: laneSpatial, groups: 1, layerIndex: 1, prefix: "l1_", inputX: l0.x, inputState: stateIn1, weightPrefix: "", outputXName: "xNext", outputStateName: "stateOut1")
            try LegacyGraphSupport.setOutputs(&graph, [("xNext", l1.x), ("stateOut0", l0.state), ("stateOut1", l1.state)])
        }
    }
}
