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

    public var inputBytes: Int {
        ModelConfig.dim * laneSpatial * 2
    }

    public var inputByteSizes: [Int] {
        let bytes = inputBytes
        return [bytes, bytes, bytes, bytes]
    }

    public var outputByteSizes: [Int] {
        let stateBytes = inputBytes
        return [stateBytes, stateBytes, stateBytes, stateBytes, vocabSize * laneSpatial * 2]
    }

    public var milText: String {
        let stepText = RWKVStyleFusedThreeLayerStepGenerator(laneSpatial: laneSpatial).milText
        let headText = GenerationRMSNormClassifierGenerator(
            vocabSize: vocabSize,
            laneSpatial: laneSpatial
        ).milText

        let stepReturn = "    } -> (xNext,stateOut0,stateOut1,stateOut2);"
        let headStart = "        tensor<fp16, [1,\(ModelConfig.dim),1,\(laneSpatial)]> sq = mul(x=x,y=x)[name=string(\"sq\")];"
        let headReturn = "    } -> (logits);"

        guard let stepReturnRange = stepText.range(of: stepReturn),
              let headStartRange = headText.range(of: headStart),
              let headReturnRange = headText.range(of: headReturn) else {
            preconditionFailure("failed to compose fused triplet+head MIL")
        }

        var headBody = String(headText[headStartRange.lowerBound..<headReturnRange.lowerBound])
        headBody = headBody.replacingOccurrences(of: "mul(x=x,y=x)", with: "mul(x=xNext,y=xNext)")
        headBody = headBody.replacingOccurrences(of: "mul(x=x,y=rw)", with: "mul(x=xNext,y=rw)")
        headBody = headBody.replacingOccurrences(
            of: "[1, \(vocabSize), 1, \(laneSpatial)]",
            with: "[1,\(vocabSize),1,\(laneSpatial)]"
        )

        var combined = stepText
        combined.replaceSubrange(
            stepReturnRange,
            with: headBody + "    } -> (xNext,stateOut0,stateOut1,stateOut2,logits);"
        )
        return combined
    }
}
