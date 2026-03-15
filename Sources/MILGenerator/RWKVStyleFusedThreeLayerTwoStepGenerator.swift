import Foundation
import ANETypes

public struct RWKVStyleFusedThreeLayerTwoStepGenerator: MILProgramGenerator {
    public let laneSpatial: Int

    public init(laneSpatial: Int = 32) {
        precondition(laneSpatial > 0)
        self.laneSpatial = laneSpatial
    }

    public var inputBytes: Int { ModelConfig.dim * laneSpatial * 2 }
    public var inputByteSizes: [Int] { let b = ModelConfig.dim * laneSpatial * 2; return [b, b, b, b, b] }
    public var outputByteSizes: [Int] { let b = ModelConfig.dim * laneSpatial * 2; return [b, b, b, b, b, b, b, b] }

    public var milText: String {
        LegacyGraphSupport.emitGraph { graph in
            let x0 = try LegacyGraphSupport.input(&graph, name: "x0", channels: ModelConfig.dim, spatial: laneSpatial)
            let x1 = try LegacyGraphSupport.input(&graph, name: "x1", channels: ModelConfig.dim, spatial: laneSpatial)
            let stateIn0 = try LegacyGraphSupport.input(&graph, name: "stateIn0", channels: ModelConfig.dim, spatial: laneSpatial)
            let stateIn1 = try LegacyGraphSupport.input(&graph, name: "stateIn1", channels: ModelConfig.dim, spatial: laneSpatial)
            let stateIn2 = try LegacyGraphSupport.input(&graph, name: "stateIn2", channels: ModelConfig.dim, spatial: laneSpatial)
            let l0s0 = try LegacyGraphSupport.recurrentLayer(&graph, dim: ModelConfig.dim, lane: laneSpatial, groups: 1, layerIndex: 0, prefix: "l0s0_", inputX: x0, inputState: stateIn0, weightPrefix: "", outputXName: "l0_x0Next", outputStateName: "stateMid0")
            let l1s0 = try LegacyGraphSupport.recurrentLayer(&graph, dim: ModelConfig.dim, lane: laneSpatial, groups: 1, layerIndex: 1, prefix: "l1s0_", inputX: l0s0.x, inputState: stateIn1, weightPrefix: "", outputXName: "l1_x0Next", outputStateName: "stateMid1")
            let l2s0 = try LegacyGraphSupport.recurrentLayer(&graph, dim: ModelConfig.dim, lane: laneSpatial, groups: 1, layerIndex: 2, prefix: "l2s0_", inputX: l1s0.x, inputState: stateIn2, weightPrefix: "", outputXName: "x0Next", outputStateName: "stateMid2")
            let l0s1 = try LegacyGraphSupport.recurrentLayer(&graph, dim: ModelConfig.dim, lane: laneSpatial, groups: 1, layerIndex: 0, prefix: "l0s1_", inputX: x1, inputState: l0s0.state, weightPrefix: "", outputXName: "l0_x1Next", outputStateName: "stateOut0")
            let l1s1 = try LegacyGraphSupport.recurrentLayer(&graph, dim: ModelConfig.dim, lane: laneSpatial, groups: 1, layerIndex: 1, prefix: "l1s1_", inputX: l0s1.x, inputState: l1s0.state, weightPrefix: "", outputXName: "l1_x1Next", outputStateName: "stateOut1")
            let l2s1 = try LegacyGraphSupport.recurrentLayer(&graph, dim: ModelConfig.dim, lane: laneSpatial, groups: 1, layerIndex: 2, prefix: "l2s1_", inputX: l1s1.x, inputState: l2s0.state, weightPrefix: "", outputXName: "x1Next", outputStateName: "stateOut2")
            try LegacyGraphSupport.setOutputs(&graph, [("x0Next", l2s0.x), ("x1Next", l2s1.x), ("stateMid0", l0s0.state), ("stateMid1", l1s0.state), ("stateMid2", l2s0.state), ("stateOut0", l0s1.state), ("stateOut1", l1s1.state), ("stateOut2", l2s1.state)])
        }
    }
}
