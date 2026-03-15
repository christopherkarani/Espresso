import Foundation
import ANETypes

public struct RWKVStyleFusedThreeLayerStepGenerator: MILProgramGenerator {
    public let laneSpatial: Int
    public let groups: Int
    public let includeRMSNorm: Bool

    public init(laneSpatial: Int = 32, groups: Int = 1, includeRMSNorm: Bool = true) {
        precondition(laneSpatial > 0)
        precondition(groups > 0 && ModelConfig.dim % groups == 0)
        self.laneSpatial = laneSpatial
        self.groups = groups
        self.includeRMSNorm = includeRMSNorm
    }

    public var inputBytes: Int { ModelConfig.dim * laneSpatial * 2 }
    public var inputByteSizes: [Int] { let b = ModelConfig.dim * laneSpatial * 2; return [b, b, b, b] }
    public var outputByteSizes: [Int] { let b = ModelConfig.dim * laneSpatial * 2; return [b, b, b, b] }

    public var milText: String {
        LegacyGraphSupport.emitGraph { graph in
            let x = try LegacyGraphSupport.input(&graph, name: "x", channels: ModelConfig.dim, spatial: laneSpatial)
            let stateIn0 = try LegacyGraphSupport.input(&graph, name: "stateIn0", channels: ModelConfig.dim, spatial: laneSpatial)
            let stateIn1 = try LegacyGraphSupport.input(&graph, name: "stateIn1", channels: ModelConfig.dim, spatial: laneSpatial)
            let stateIn2 = try LegacyGraphSupport.input(&graph, name: "stateIn2", channels: ModelConfig.dim, spatial: laneSpatial)
            let l0 = try LegacyGraphSupport.recurrentLayer(&graph, dim: ModelConfig.dim, lane: laneSpatial, groups: groups, layerIndex: 0, prefix: "l0_", inputX: x, inputState: stateIn0, includeRMSNorm: includeRMSNorm, weightPrefix: "", outputXName: "l0_xNext", outputStateName: "stateOut0")
            let l1 = try LegacyGraphSupport.recurrentLayer(&graph, dim: ModelConfig.dim, lane: laneSpatial, groups: groups, layerIndex: 1, prefix: "l1_", inputX: l0.x, inputState: stateIn1, includeRMSNorm: includeRMSNorm, weightPrefix: "", outputXName: "l1_xNext", outputStateName: "stateOut1")
            let l2 = try LegacyGraphSupport.recurrentLayer(&graph, dim: ModelConfig.dim, lane: laneSpatial, groups: groups, layerIndex: 2, prefix: "l2_", inputX: l1.x, inputState: stateIn2, includeRMSNorm: includeRMSNorm, weightPrefix: "", outputXName: "xNext", outputStateName: "stateOut2")
            try LegacyGraphSupport.setOutputs(&graph, [("xNext", l2.x), ("stateOut0", l0.state), ("stateOut1", l1.state), ("stateOut2", l2.state)])
        }
    }
}
