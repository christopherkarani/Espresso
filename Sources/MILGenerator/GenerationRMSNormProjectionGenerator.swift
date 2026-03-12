import Foundation
import ANETypes

/// RMSNorm + projection conv [dim → bottleneck] only. No expansion conv.
/// Output is a small projected surface for CPU-side expansion + argmax.
///
/// Input:  [1, dim, 1, laneSpatial]
/// Output: [1, bottleneck, 1, laneSpatial]
public struct GenerationRMSNormProjectionGenerator: MILProgramGenerator {
    public let bottleneck: Int
    public let laneSpatial: Int
    public let groups: Int

    public init(bottleneck: Int = 128, laneSpatial: Int = 32, groups: Int = 1) {
        precondition(bottleneck > 0)
        precondition(laneSpatial > 0)
        precondition(groups > 0)
        precondition(ModelConfig.dim % groups == 0)
        precondition(bottleneck % groups == 0)
        self.bottleneck = bottleneck
        self.laneSpatial = laneSpatial
        self.groups = groups
    }

    public var inputBytes: Int {
        ModelConfig.dim * laneSpatial * 2
    }

    public var inputByteSizes: [Int] {
        [inputBytes]
    }

    public var outputByteSizes: [Int] {
        [bottleneck * laneSpatial * 2]
    }

    public var milText: String {
        let dim = ModelConfig.dim
        let lane = laneSpatial
        let k = bottleneck
        let g = groups
        let projInPerGroup = dim / g
        let invd: Float = 1.0 / Float(dim)

        var b = MILBuilder(reserveCapacity: 2_048)
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
        // Projection conv: [dim → bottleneck] with groups
        b.appendLine("        string pt = const()[name=string(\"pt\"), val=string(\"valid\")];")
        b.appendLine("        tensor<int32, [2]> st = const()[name=string(\"st\"), val=tensor<int32, [2]>([1,1])];")
        b.appendLine("        tensor<int32, [4]> pd = const()[name=string(\"pd\"), val=tensor<int32, [4]>([0,0,0,0])];")
        b.appendLine("        tensor<int32, [2]> dl = const()[name=string(\"dl\"), val=tensor<int32, [2]>([1,1])];")
        b.appendLine("        int32 gr = const()[name=string(\"gr\"), val=int32(\(g))];")
        b.appendLine(
            "        tensor<fp16, [\(k), \(projInPerGroup), 1, 1]> Wproj = const()[name=string(\"Wproj\"), val=tensor<fp16, [\(k), \(projInPerGroup), 1, 1]>(BLOBFILE(path=string(\"@model_path/weights/cls_proj.bin\"), offset=uint64(64)))];"
        )
        b.appendLine(
            "        tensor<fp16, [1, \(k), 1, \(lane)]> proj = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=Wproj,x=xn)[name=string(\"proj\")];"
        )
        b.appendLine("    } -> (proj);")
        b.appendLine("}")
        return b.text
    }
}
