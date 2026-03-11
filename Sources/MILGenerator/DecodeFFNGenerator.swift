import Foundation
import ANETypes

/// Decode-time FFN kernel with fused residual.
///
/// Input:
/// - `x`: `[1, dim, 1, laneSpatial]` (token packed at lane 0, remaining lanes zero)
///
/// Output:
/// - `x + ffn(x)`: `[1, dim, 1, laneSpatial]`
public struct DecodeFFNGenerator: MILProgramGenerator {
    public let laneSpatial: Int

    public init(laneSpatial: Int = 32) {
        precondition(laneSpatial > 0)
        self.laneSpatial = laneSpatial
    }

    public var inputBytes: Int { ModelConfig.dim * laneSpatial * 2 }
    public var outputByteSizes: [Int] { [ModelConfig.dim * laneSpatial * 2] }

    public var milText: String {
        let dim = ModelConfig.dim
        let hidden = ModelConfig.hidden
        let invd: Float = 1.0 / Float(dim)
        let lane = self.laneSpatial

        var b = MILBuilder(reserveCapacity: 8_192)
        b.append(MILText.header)
        b.appendLine("    func main<ios18>(tensor<fp16, [1, \(dim), 1, \(lane)]> x) {")

        // RMSNorm
        b.appendLine("        tensor<fp16, [1,\(dim),1,\(lane)]> sq = mul(x=x,y=x)[name=string(\"sq\")];")
        b.appendLine("        tensor<int32, [1]> raxCh = const()[name=string(\"rax_ch\"), val=tensor<int32, [1]>([1])];")
        b.appendLine("        tensor<int32, [1]> raxSp = const()[name=string(\"rax_sp\"), val=tensor<int32, [1]>([3])];")
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
        b.appendLine("        tensor<fp16, [1,\(dim),1,1]> rw = const()[name=string(\"rw\"), val=tensor<fp16, [1,\(dim),1,1]>(BLOBFILE(path=string(\"@model_path/weights/rms2.bin\"), offset=uint64(64)))];")
        b.appendLine("        tensor<fp16, [1,\(dim),1,\(lane)]> xn = mul(x=xr,y=rw)[name=string(\"xn\")];")

        // FFN
        b.append(MILText.convConst)
        b.appendLine("        tensor<fp16, [\(hidden),\(dim),1,1]> W1 = const()[name=string(\"W1\"), val=tensor<fp16, [\(hidden),\(dim),1,1]>(BLOBFILE(path=string(\"@model_path/weights/w1.bin\"), offset=uint64(64)))];")
        b.appendLine("        tensor<fp16, [\(hidden),\(dim),1,1]> W3 = const()[name=string(\"W3\"), val=tensor<fp16, [\(hidden),\(dim),1,1]>(BLOBFILE(path=string(\"@model_path/weights/w3.bin\"), offset=uint64(64)))];")
        b.appendLine("        tensor<fp16, [\(dim),\(hidden),1,1]> W2 = const()[name=string(\"W2\"), val=tensor<fp16, [\(dim),\(hidden),1,1]>(BLOBFILE(path=string(\"@model_path/weights/w2.bin\"), offset=uint64(64)))];")
        b.appendLine("        tensor<fp16, [1,\(hidden),1,\(lane)]> h1 = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=W1,x=xn)[name=string(\"c1\")];")
        b.appendLine("        tensor<fp16, [1,\(hidden),1,\(lane)]> h3 = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=W3,x=xn)[name=string(\"c3\")];")
        b.appendLine("        tensor<fp16, [1,\(hidden),1,\(lane)]> sig = sigmoid(x=h1)[name=string(\"sg\")];")
        b.appendLine("        tensor<fp16, [1,\(hidden),1,\(lane)]> silu = mul(x=h1,y=sig)[name=string(\"si\")];")
        b.appendLine("        tensor<fp16, [1,\(hidden),1,\(lane)]> gate = mul(x=silu,y=h3)[name=string(\"gt\")];")
        b.appendLine("        tensor<fp16, [1,\(dim),1,\(lane)]> y = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=W2,x=gate)[name=string(\"c2\")];")

        // Residual
        b.appendLine("        tensor<fp16, [1,\(dim),1,\(lane)]> out = add(x=x,y=y)[name=string(\"res\")];")
        b.appendLine("    } -> (out);")
        b.appendLine("}")
        return b.text
    }
}
