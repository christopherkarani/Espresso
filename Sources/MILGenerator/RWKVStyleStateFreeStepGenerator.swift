import Foundation
import ANETypes

/// One-input variant of the minimal RWKV-style recurrent decode step.
///
/// This is exact when the recurrent weights ignore incoming state (`Ws == 0`
/// and `Wd == 0`). In that case the recurrent cell degenerates into a pure
/// activation transform, which avoids the problematic two-input same-shape ANE
/// contract entirely.
public struct RWKVStyleStateFreeStepGenerator: MILProgramGenerator {
    public let laneSpatial: Int

    public init(laneSpatial: Int = 32) {
        precondition(laneSpatial > 0)
        self.laneSpatial = laneSpatial
    }

    public var inputBytes: Int { ModelConfig.dim * laneSpatial * 2 }

    public var inputByteSizes: [Int] {
        [inputBytes]
    }

    public var outputByteSizes: [Int] {
        [inputBytes]
    }

    public var milText: String {
        let dim = ModelConfig.dim
        let lane = laneSpatial
        let invd: Float = 1.0 / Float(dim)

        var builder = MILBuilder(reserveCapacity: 8_192)
        builder.append(MILText.header)
        builder.appendLine("    func main<ios18>(tensor<fp16, [1, \(dim), 1, \(lane)]> x) {")
        builder.appendLine("        tensor<int32, [1]> raxCh = const()[name=string(\"rax_ch\"), val=tensor<int32, [1]>([1])];")
        builder.appendLine("        bool kd = const()[name=string(\"kd\"), val=bool(true)];")
        builder.append("        fp16 invd = const()[name=string(\"invd\"), val=fp16(")
        builder.appendFP16(invd)
        builder.appendLine(")];")
        builder.appendLine("        fp16 eps = const()[name=string(\"eps\"), val=fp16(0.00001)];")
        builder.appendLine("        fp16 nhalf = const()[name=string(\"nhalf\"), val=fp16(-0.5)];")
        builder.appendLine("        tensor<fp16, [1,\(dim),1,\(lane)]> sq = mul(x=x,y=x)[name=string(\"sq\")];")
        builder.appendLine("        tensor<fp16, [1,1,1,\(lane)]> ss = reduce_sum(x=sq,axes=raxCh,keep_dims=kd)[name=string(\"ss\")];")
        builder.appendLine("        tensor<fp16, [1,1,1,\(lane)]> ss2 = mul(x=ss,y=invd)[name=string(\"ss2\")];")
        builder.appendLine("        tensor<fp16, [1,1,1,\(lane)]> ss3 = add(x=ss2,y=eps)[name=string(\"ss3\")];")
        builder.appendLine("        tensor<fp16, [1,1,1,\(lane)]> rrms = pow(x=ss3,y=nhalf)[name=string(\"rrms\")];")
        builder.appendLine("        tensor<fp16, [1,\(dim),1,\(lane)]> xr = mul(x=x,y=rrms)[name=string(\"xr\")];")
        builder.appendLine("        tensor<fp16, [1,\(dim),1,1]> rw = const()[name=string(\"rw\"), val=tensor<fp16, [1,\(dim),1,1]>(BLOBFILE(path=string(\"@model_path/weights/rwkv_rms.bin\"), offset=uint64(64)))];")
        builder.appendLine("        tensor<fp16, [1,\(dim),1,\(lane)]> xn = mul(x=xr,y=rw)[name=string(\"xn\")];")

        builder.append(MILText.convConst)
        builder.appendLine("        tensor<fp16, [\(dim),\(dim),1,1]> Wx = const()[name=string(\"Wx\"), val=tensor<fp16, [\(dim),\(dim),1,1]>(BLOBFILE(path=string(\"@model_path/weights/wx.bin\"), offset=uint64(64)))];")
        builder.appendLine("        tensor<fp16, [\(dim),\(dim),1,1]> Wo = const()[name=string(\"Wo\"), val=tensor<fp16, [\(dim),\(dim),1,1]>(BLOBFILE(path=string(\"@model_path/weights/wo.bin\"), offset=uint64(64)))];")
        builder.appendLine("        tensor<fp16, [1,\(dim),1,\(lane)]> xMix = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=Wx,x=xn)[name=string(\"x_mix\")];")
        builder.appendLine("        tensor<fp16, [1,\(dim),1,\(lane)]> proj = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=Wo,x=xMix)[name=string(\"proj\")];")
        builder.appendLine("        tensor<fp16, [1,\(dim),1,\(lane)]> xNext = add(x=x,y=proj)[name=string(\"xNext\")];")
        builder.appendLine("    } -> (xNext);")
        builder.appendLine("}")
        return builder.text
    }
}
