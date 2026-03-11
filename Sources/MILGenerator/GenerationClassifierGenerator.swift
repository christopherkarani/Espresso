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
        let dim = ModelConfig.dim
        let vocab = vocabSize
        let lane = laneSpatial

        var b = MILBuilder(reserveCapacity: 2_048)
        b.append(MILText.header)
        b.appendLine("    func main<ios18>(tensor<fp16, [1, \(dim), 1, \(lane)]> x) {")
        b.append(MILText.convConst)
        b.appendLine(
            "        tensor<fp16, [\(vocab), \(dim), 1, 1]> Wcls = const()[name=string(\"Wcls\"), val=tensor<fp16, [\(vocab), \(dim), 1, 1]>(BLOBFILE(path=string(\"@model_path/weights/classifier.bin\"), offset=uint64(64)))];"
        )
        b.appendLine(
            "        tensor<fp16, [1, \(vocab), 1, \(lane)]> logits = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=Wcls,x=x)[name=string(\"cls\")];"
        )
        b.appendLine("    } -> (logits);")
        b.appendLine("}")
        return b.text
    }
}
