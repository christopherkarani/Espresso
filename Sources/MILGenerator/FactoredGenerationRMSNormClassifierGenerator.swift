import Foundation
import ANETypes

public struct FactoredGenerationRMSNormClassifierGenerator: MILProgramGenerator {
    public let vocabSize: Int
    public let bottleneck: Int
    public let laneSpatial: Int

    public init(vocabSize: Int, bottleneck: Int = 128, laneSpatial: Int = 32) {
        precondition(vocabSize > 0)
        precondition(bottleneck > 0)
        precondition(laneSpatial > 0)
        self.vocabSize = vocabSize
        self.bottleneck = bottleneck
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
        let k = bottleneck
        let invd: Float = 1.0 / Float(dim)

        var b = MILBuilder(reserveCapacity: 4_096)
        b.append(MILText.header)
        b.appendLine("    func main<ios18>(tensor<fp16, [1, \(dim), 1, \(lane)]> x) {")
        // RMSNorm
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
        // Factored conv 1: [dim → k]
        b.appendLine(
            "        tensor<fp16, [\(k), \(dim), 1, 1]> Wproj = const()[name=string(\"Wproj\"), val=tensor<fp16, [\(k), \(dim), 1, 1]>(BLOBFILE(path=string(\"@model_path/weights/cls_proj.bin\"), offset=uint64(64)))];"
        )
        b.appendLine(
            "        tensor<fp16, [1, \(k), 1, \(lane)]> proj = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=Wproj,x=xn)[name=string(\"proj\")];"
        )
        // Factored conv 2: [k → vocab]
        b.appendLine(
            "        tensor<fp16, [\(vocab), \(k), 1, 1]> Wexp = const()[name=string(\"Wexp\"), val=tensor<fp16, [\(vocab), \(k), 1, 1]>(BLOBFILE(path=string(\"@model_path/weights/cls_expand.bin\"), offset=uint64(64)))];"
        )
        b.appendLine(
            "        tensor<fp16, [1, \(vocab), 1, \(lane)]> logits = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=Wexp,x=proj)[name=string(\"expand\")];"
        )
        b.appendLine("    } -> (logits);")
        b.appendLine("}")
        return b.text
    }
}
