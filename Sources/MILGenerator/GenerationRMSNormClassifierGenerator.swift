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
        let dim = ModelConfig.dim
        let vocab = vocabSize
        let lane = laneSpatial
        let invd: Float = 1.0 / Float(dim)

        var b = MILBuilder(reserveCapacity: 4_096)
        b.append(MILText.header)
        b.appendLine("    func main<ios18>(tensor<fp16, [1, \(dim), 1, \(lane)]> x) {")
        b.appendLine("        tensor<fp16, [1,\(dim),1,\(lane)]> sq = mul(x=x,y=x)[name=string(\"sq\")];")
        b.appendLine("        tensor<int32, [1]> raxCh = const()[name=string(\"rax_ch\"), val=tensor<int32, [1]>([1])];")
        b.appendLine("        bool kd = const()[name=string(\"kd\"), val=bool(true)];")
        b.appendLine("        tensor<fp16, [1,1,1,\(lane)]> ss = reduce_sum(x=sq,axes=raxCh,keep_dims=kd)[name=string(\"ss\")];")
        b.append("        fp16 invd = const()[name=string(\"invd\"), val=fp16(")
        b.appendFP16(invd)
        b.appendLine(")];")
        b.appendLine("        tensor<fp16, [1,1,1,\(lane)]> ss2 = mul(x=ss,y=invd)[name=string(\"ss2\")];")
        b.appendLine("        fp16 eps = const()[name=string(\"eps\"), val=fp16(0.00001)];")
        b.appendLine("        tensor<fp16, [1,1,1,\(lane)]> ss3 = add(x=ss2,y=eps)[name=string(\"ss3\")];")
        b.appendLine("        fp16 nhalf = const()[name=string(\"nhalf\"), val=fp16(-0.5)];")
        b.appendLine("        tensor<fp16, [1,1,1,\(lane)]> rrms = pow(x=ss3,y=nhalf)[name=string(\"rrms\")];")
        b.appendLine("        tensor<fp16, [1,\(dim),1,\(lane)]> xr = mul(x=x,y=rrms)[name=string(\"xr\")];")
        b.appendLine(
            "        tensor<fp16, [1,\(dim),1,1]> rw = const()[name=string(\"rw\"), val=tensor<fp16, [1,\(dim),1,1]>(BLOBFILE(path=string(\"@model_path/weights/rms_final.bin\"), offset=uint64(64)))];"
        )
        b.appendLine("        tensor<fp16, [1,\(dim),1,\(lane)]> xn = mul(x=xr,y=rw)[name=string(\"xn\")];")
        b.append(MILText.convConst)
        b.appendLine(
            "        tensor<fp16, [\(vocab), \(dim), 1, 1]> Wcls = const()[name=string(\"Wcls\"), val=tensor<fp16, [\(vocab), \(dim), 1, 1]>(BLOBFILE(path=string(\"@model_path/weights/classifier.bin\"), offset=uint64(64)))];"
        )
        b.appendLine(
            "        tensor<fp16, [1, \(vocab), 1, \(lane)]> logits = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=Wcls,x=xn)[name=string(\"cls\")];"
        )
        b.appendLine("        tensor<int32, [1]> rmaxCh = const()[name=string(\"rmax_ch\"), val=tensor<int32, [1]>([1])];")
        b.appendLine("        bool rmaxKd = const()[name=string(\"rmax_kd\"), val=bool(true)];")
        b.appendLine("        tensor<fp16, [1,1,1,\(lane)]> maxVal = reduce_max(x=logits,axes=rmaxCh,keep_dims=rmaxKd)[name=string(\"maxval\")];")
        b.appendLine("    } -> (logits,maxVal);")
        b.appendLine("}")
        return b.text
    }
}
