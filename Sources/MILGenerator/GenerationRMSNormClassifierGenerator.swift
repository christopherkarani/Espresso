import Foundation
import ANETypes

public struct GenerationRMSNormClassifierGenerator: MILProgramGenerator {
    public let vocabSize: Int
    public let laneSpatial: Int

    public init(vocabSize: Int, laneSpatial: Int = 32) {
        precondition(vocabSize > 0)
        precondition(laneSpatial > 0)
        self.vocabSize = vocabSize
        self.laneSpatial = laneSpatial
    }

    public var inputBytes: Int {
        ModelConfig.dim * laneSpatial * 2
    }

    public var inputByteSizes: [Int] {
        [inputBytes]
    }

    public var outputByteSizes: [Int] {
        [vocabSize * laneSpatial * 2, 1 * laneSpatial * 2]
    }

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
            let logits = try LegacyGraphSupport.conv(
                &graph,
                name: "logits",
                input: xn,
                weightName: "Wcls",
                outChannels: vocabSize,
                inChannelsPerGroup: ModelConfig.dim,
                spatial: laneSpatial,
                weightPath: "@model_path/weights/classifier.bin"
            )
            let maxVal = try graph.reduceMax("maxVal", input: logits, axis: 1, keepDims: true)
            try LegacyGraphSupport.setOutputs(&graph, [("logits", logits), ("maxVal", maxVal)])
        }
    }
}
