import Foundation
import ANETypes

/// Decode-time QKV projection kernel.
///
/// This isolates the ANE work needed before Metal-owned attention:
/// RMSNorm(x) -> Wq/Wk/Wv, with no cache reads, mask input, or output projection.
///
/// Input:
/// - `x`: `[1, dim, 1, laneSpatial]`
///
/// Outputs:
/// - `qOut`: `[1, dim, 1, laneSpatial]`
/// - `kNew`: `[1, dim, 1, laneSpatial]`
/// - `vNew`: `[1, dim, 1, laneSpatial]`
public struct DecodeQKVOnlyGenerator: MILProgramGenerator {
    public let laneSpatial: Int

    public init(laneSpatial: Int = 32) {
        precondition(laneSpatial > 0)
        self.laneSpatial = laneSpatial
    }

    public var inputBytes: Int { ModelConfig.dim * laneSpatial * 2 }

    public var inputByteSizes: [Int] {
        [ModelConfig.dim * laneSpatial * 2]
    }

    public var outputByteSizes: [Int] {
        [
            ModelConfig.dim * laneSpatial * 2,
            ModelConfig.dim * laneSpatial * 2,
            ModelConfig.dim * laneSpatial * 2,
        ]
    }

    public var milText: String {
        let dim = ModelConfig.dim
        let lane = self.laneSpatial
        let invd: Float = 1.0 / Float(dim)

        var b = MILBuilder(reserveCapacity: 8_192)
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
        b.appendLine("        tensor<fp16, [1,\(dim),1,1]> rw = const()[name=string(\"rw\"), val=tensor<fp16, [1,\(dim),1,1]>(BLOBFILE(path=string(\"@model_path/weights/rms1.bin\"), offset=uint64(64)))];")
        b.appendLine("        tensor<fp16, [1,\(dim),1,\(lane)]> xn = mul(x=xr,y=rw)[name=string(\"xn\")];")

        // QKV projections only.
        b.append(MILText.convConst)
        b.appendLine("        tensor<fp16, [\(dim),\(dim),1,1]> Wq = const()[name=string(\"Wq\"), val=tensor<fp16, [\(dim),\(dim),1,1]>(BLOBFILE(path=string(\"@model_path/weights/wq.bin\"), offset=uint64(64)))];")
        b.appendLine("        tensor<fp16, [\(dim),\(dim),1,1]> Wk = const()[name=string(\"Wk\"), val=tensor<fp16, [\(dim),\(dim),1,1]>(BLOBFILE(path=string(\"@model_path/weights/wk.bin\"), offset=uint64(64)))];")
        b.appendLine("        tensor<fp16, [\(dim),\(dim),1,1]> Wv = const()[name=string(\"Wv\"), val=tensor<fp16, [\(dim),\(dim),1,1]>(BLOBFILE(path=string(\"@model_path/weights/wv.bin\"), offset=uint64(64)))];")
        b.appendLine("        tensor<fp16, [1,\(dim),1,\(lane)]> qOut = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=Wq,x=xn)[name=string(\"cq\")];")
        b.appendLine("        tensor<fp16, [1,\(dim),1,\(lane)]> kNew = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=Wk,x=xn)[name=string(\"ck\")];")
        b.appendLine("        tensor<fp16, [1,\(dim),1,\(lane)]> vNew = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=Wv,x=xn)[name=string(\"cv\")];")

        b.appendLine("    } -> (qOut,kNew,vNew);")
        b.appendLine("}")
        return b.text
    }
}
