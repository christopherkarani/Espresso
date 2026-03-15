import Foundation
import ANETypes

public struct FactoredGenerationRMSNormClassifierGenerator: MILProgramGenerator {
    public let vocabSize: Int
    public let bottleneck: Int
    public let laneSpatial: Int
    public let groups: Int

    public init(vocabSize: Int, bottleneck: Int = 128, laneSpatial: Int = 32, groups: Int = 1) {
        precondition(vocabSize > 0)
        precondition(bottleneck > 0)
        precondition(laneSpatial > 0)
        precondition(groups > 0)
        precondition(bottleneck % groups == 0)
        precondition(ModelConfig.dim % groups == 0)
        self.vocabSize = vocabSize
        self.bottleneck = bottleneck
        self.laneSpatial = laneSpatial
        self.groups = groups
    }

    public var inputBytes: Int { ModelConfig.dim * laneSpatial * 2 }
    public var inputByteSizes: [Int] { [inputBytes] }
    public var outputByteSizes: [Int] { [vocabSize * laneSpatial * 2] }

    public var milText: String {
        LegacyGraphSupport.emitGraph { graph in
            let x = try LegacyGraphSupport.input(&graph, name: "x", channels: ModelConfig.dim, spatial: laneSpatial)
            let xn = try LegacyGraphSupport.rmsNorm(
                &graph,
                input: x,
                dim: ModelConfig.dim,
                spatial: laneSpatial,
                sq: "sq",
                axisName: "rax_ch",
                keepDimsName: "kd",
                ss: "ss",
                invdName: "invd",
                ss2: "ss2",
                epsName: "eps",
                ss3: "ss3",
                nhalfName: "nhalf",
                rrms: "rrms",
                xr: "xr",
                weightName: "rw",
                weightPath: "@model_path/weights/rms_final.bin",
                output: "xn"
            )
            let proj = try LegacyGraphSupport.conv(
                &graph,
                name: "proj",
                input: xn,
                weightName: "Wproj",
                outChannels: bottleneck,
                inChannelsPerGroup: ModelConfig.dim / groups,
                groups: groups,
                spatial: laneSpatial,
                weightPath: "@model_path/weights/cls_proj.bin"
            )
            let logits = try LegacyGraphSupport.conv(
                &graph,
                name: "logits",
                input: proj,
                weightName: "Wexp",
                outChannels: vocabSize,
                inChannelsPerGroup: bottleneck / groups,
                groups: groups,
                spatial: laneSpatial,
                weightPath: "@model_path/weights/cls_expand.bin"
            )
            try LegacyGraphSupport.setOutputs(&graph, [("logits", logits)])
        }
    }
}
