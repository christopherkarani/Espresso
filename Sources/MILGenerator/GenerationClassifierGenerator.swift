import Foundation
import ANETypes

public struct GenerationClassifierGenerator: MILProgramGenerator {
    public let vocabSize: Int
    public let laneSpatial: Int

    public init(vocabSize: Int, laneSpatial: Int = 1) {
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
        [vocabSize * laneSpatial * 2]
    }

    public var milText: String {
        LegacyGraphSupport.emitGraph { graph in
            let x = try LegacyGraphSupport.input(&graph, name: "x", channels: ModelConfig.dim, spatial: laneSpatial)
            let logits = try LegacyGraphSupport.conv(
                &graph,
                name: "logits",
                input: x,
                weightName: "Wcls",
                outChannels: vocabSize,
                inChannelsPerGroup: ModelConfig.dim,
                spatial: laneSpatial,
                weightPath: "@model_path/weights/classifier.bin"
            )
            try LegacyGraphSupport.setOutputs(&graph, [("logits", logits)])
        }
    }
}
