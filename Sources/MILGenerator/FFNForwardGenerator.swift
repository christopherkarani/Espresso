import Foundation
import ANETypes

public struct FFNForwardGenerator: MILProgramGenerator {
    public init() {}

    public var inputBytes: Int { ModelConfig.dim * ModelConfig.seqLen * 2 }
    public var outputByteSizes: [Int] {
        // concat(y, h1, h3, gate, x2norm)
        [(2 * ModelConfig.dim + 3 * ModelConfig.hidden) * ModelConfig.seqLen * 2]
    }

    public var milText: String {
        let invd: Float = 1.0 / Float(ModelConfig.dim)

        var b = MILBuilder(reserveCapacity: 12_288)
        b.append(MILText.header)
        b.appendLine("    func main<ios18>(tensor<fp16, [1, \(ModelConfig.dim), 1, \(ModelConfig.seqLen)]> x) {")
        b.appendLine("        tensor<fp16, [1,\(ModelConfig.dim),1,\(ModelConfig.seqLen)]> sq = mul(x=x,y=x)[name=string(\"sq\")];")
        b.appendLine("        tensor<int32, [1]> rax = const()[name=string(\"rax\"), val=tensor<int32, [1]>([1])];")
        b.appendLine("        bool kd = const()[name=string(\"kd\"), val=bool(true)];")
        b.appendLine("        tensor<fp16, [1,1,1,\(ModelConfig.seqLen)]> ss = reduce_sum(x=sq,axes=rax,keep_dims=kd)[name=string(\"ss\")];")
        b.append("        fp16 invd = const()[name=string(\"invd\"), val=fp16(")
        b.appendFP16(invd)
        b.appendLine(")];")
        b.appendLine("        tensor<fp16, [1,1,1,\(ModelConfig.seqLen)]> ss2 = mul(x=ss,y=invd)[name=string(\"ss2\")];")
        b.appendLine("        fp16 eps = const()[name=string(\"eps\"), val=fp16(0.00001)];")
        b.appendLine("        tensor<fp16, [1,1,1,\(ModelConfig.seqLen)]> ss3 = add(x=ss2,y=eps)[name=string(\"ss3\")];")
        b.appendLine("        fp16 nhalf = const()[name=string(\"nhalf\"), val=fp16(-0.5)];")
        b.appendLine("        tensor<fp16, [1,1,1,\(ModelConfig.seqLen)]> rrms = pow(x=ss3,y=nhalf)[name=string(\"rrms\")];")
        b.appendLine("        tensor<fp16, [1,\(ModelConfig.dim),1,\(ModelConfig.seqLen)]> xr = mul(x=x,y=rrms)[name=string(\"xr\")];")
        b.appendLine("        tensor<fp16, [1,\(ModelConfig.dim),1,1]> rw = const()[name=string(\"rw\"), val=tensor<fp16, [1,\(ModelConfig.dim),1,1]>(BLOBFILE(path=string(\"@model_path/weights/rms2.bin\"), offset=uint64(64)))];")
        b.appendLine("        tensor<fp16, [1,\(ModelConfig.dim),1,\(ModelConfig.seqLen)]> xn = mul(x=xr,y=rw)[name=string(\"xn\")];")
        b.append(MILText.convConst)
        b.appendLine("        tensor<fp16, [\(ModelConfig.hidden),\(ModelConfig.dim),1,1]> W1 = const()[name=string(\"W1\"), val=tensor<fp16, [\(ModelConfig.hidden),\(ModelConfig.dim),1,1]>(BLOBFILE(path=string(\"@model_path/weights/w1.bin\"), offset=uint64(64)))];")
        b.appendLine("        tensor<fp16, [\(ModelConfig.hidden),\(ModelConfig.dim),1,1]> W3 = const()[name=string(\"W3\"), val=tensor<fp16, [\(ModelConfig.hidden),\(ModelConfig.dim),1,1]>(BLOBFILE(path=string(\"@model_path/weights/w3.bin\"), offset=uint64(64)))];")
        b.appendLine("        tensor<fp16, [\(ModelConfig.dim),\(ModelConfig.hidden),1,1]> W2 = const()[name=string(\"W2\"), val=tensor<fp16, [\(ModelConfig.dim),\(ModelConfig.hidden),1,1]>(BLOBFILE(path=string(\"@model_path/weights/w2.bin\"), offset=uint64(64)))];")
        b.appendLine("        tensor<fp16, [1,\(ModelConfig.hidden),1,\(ModelConfig.seqLen)]> h1 = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=W1,x=xn)[name=string(\"c1\")];")
        b.appendLine("        tensor<fp16, [1,\(ModelConfig.hidden),1,\(ModelConfig.seqLen)]> h3 = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=W3,x=xn)[name=string(\"c3\")];")
        b.appendLine("        tensor<fp16, [1,\(ModelConfig.hidden),1,\(ModelConfig.seqLen)]> sig = sigmoid(x=h1)[name=string(\"sg\")];")
        b.appendLine("        tensor<fp16, [1,\(ModelConfig.hidden),1,\(ModelConfig.seqLen)]> silu = mul(x=h1,y=sig)[name=string(\"si\")];")
        b.appendLine("        tensor<fp16, [1,\(ModelConfig.hidden),1,\(ModelConfig.seqLen)]> gate = mul(x=silu,y=h3)[name=string(\"gt\")];")
        b.appendLine("        tensor<fp16, [1,\(ModelConfig.dim),1,\(ModelConfig.seqLen)]> y = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=W2,x=gate)[name=string(\"c2\")];")
        b.appendLine("        int32 cax = const()[name=string(\"cax\"), val=int32(1)];")
        b.appendLine("        bool cid = const()[name=string(\"cid\"), val=bool(false)];")
        b.appendLine("        tensor<fp16, [1,\(2 * ModelConfig.dim + 3 * ModelConfig.hidden),1,\(ModelConfig.seqLen)]> out = concat(axis=cax,interleave=cid,values=(y,h1,h3,gate,xn))[name=string(\"cat\")];")
        b.appendLine("    } -> (out);")
        b.appendLine("}")
        return b.text
    }
}
