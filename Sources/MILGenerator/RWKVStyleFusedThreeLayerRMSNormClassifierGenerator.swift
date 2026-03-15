import Foundation
import ANETypes

public struct RWKVStyleFusedThreeLayerRMSNormClassifierGenerator: MILProgramGenerator {
    public let vocabSize: Int
    public let laneSpatial: Int

    public init(vocabSize: Int, laneSpatial: Int = 32) {
        precondition(vocabSize > 0)
        precondition(laneSpatial > 0)
        self.vocabSize = vocabSize
        self.laneSpatial = laneSpatial
    }

    public var inputBytes: Int { ModelConfig.dim * laneSpatial * 2 }
    public var inputByteSizes: [Int] { let b = ModelConfig.dim * laneSpatial * 2; return [b, b, b, b] }
    public var outputByteSizes: [Int] { let b = ModelConfig.dim * laneSpatial * 2; return [b, b, b, b, vocabSize * laneSpatial * 2, laneSpatial * 2] }

    public var milText: String {
        LegacyGraphSupport.emitGraph { graph in
            let x = try LegacyGraphSupport.input(&graph, name: "x", channels: ModelConfig.dim, spatial: laneSpatial)
            let stateIn0 = try LegacyGraphSupport.input(&graph, name: "stateIn0", channels: ModelConfig.dim, spatial: laneSpatial)
            let stateIn1 = try LegacyGraphSupport.input(&graph, name: "stateIn1", channels: ModelConfig.dim, spatial: laneSpatial)
            let stateIn2 = try LegacyGraphSupport.input(&graph, name: "stateIn2", channels: ModelConfig.dim, spatial: laneSpatial)
            let l0 = try LegacyGraphSupport.recurrentLayer(&graph, dim: ModelConfig.dim, lane: laneSpatial, groups: 1, layerIndex: 0, prefix: "l0_", inputX: x, inputState: stateIn0, weightPrefix: "", outputXName: "l0_xNext", outputStateName: "stateOut0")
            let l1 = try LegacyGraphSupport.recurrentLayer(&graph, dim: ModelConfig.dim, lane: laneSpatial, groups: 1, layerIndex: 1, prefix: "l1_", inputX: l0.x, inputState: stateIn1, weightPrefix: "", outputXName: "l1_xNext", outputStateName: "stateOut1")
            let l2 = try LegacyGraphSupport.recurrentLayer(&graph, dim: ModelConfig.dim, lane: laneSpatial, groups: 1, layerIndex: 2, prefix: "l2_", inputX: l1.x, inputState: stateIn2, weightPrefix: "", outputXName: "xNext", outputStateName: "stateOut2")
            let hxn = try LegacyGraphSupport.rmsNorm(&graph, input: l2.x, dim: ModelConfig.dim, spatial: laneSpatial, sq: "h_sq", axisName: "h_rax_ch", keepDimsName: "h_kd", ss: "h_ss", invdName: "h_invd", ss2: "h_ss2", epsName: "h_eps", ss3: "h_ss3", nhalfName: "h_nhalf", rrms: "h_rrms", xr: "h_xr", weightName: "h_rw", weightPath: "@model_path/weights/rms_final.bin", output: "h_xn")
            let logits = try LegacyGraphSupport.conv(&graph, name: "logits", input: hxn, weightName: "Wcls", outChannels: vocabSize, inChannelsPerGroup: ModelConfig.dim, spatial: laneSpatial, weightPath: "@model_path/weights/classifier.bin")
            let maxVal = try graph.reduceMax("maxVal", input: logits, axis: 1, keepDims: true)
            try LegacyGraphSupport.setOutputs(&graph, [("xNext", l2.x), ("stateOut0", l0.state), ("stateOut1", l1.state), ("stateOut2", l2.state), ("logits", logits), ("maxVal", maxVal)])
        }
    }
}
