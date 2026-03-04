import Foundation
import ANETypes

public struct FFNBackwardGenerator: MILProgramGenerator {
    public init() {}

    public var inputBytes: Int { (ModelConfig.dim + 2 * ModelConfig.hidden) * ModelConfig.seqLen * 2 }
    public var outputByteSizes: [Int] { [(ModelConfig.dim + 2 * ModelConfig.hidden) * ModelConfig.seqLen * 2] }

    public var milText: String {
        var b = MILBuilder(reserveCapacity: 12_288)
        b.append(MILText.header)
        b.appendLine("    func main<ios18>(tensor<fp16, [1, \(ModelConfig.dim + 2 * ModelConfig.hidden), 1, \(ModelConfig.seqLen)]> x) {")
        b.append(MILText.convConst)
        b.appendLine("        tensor<int32, [4]> bd = const()[name=string(\"bd\"), val=tensor<int32, [4]>([0,0,0,0])];")
        b.appendLine("        tensor<int32, [4]> sd = const()[name=string(\"sd\"), val=tensor<int32, [4]>([1,\(ModelConfig.dim),1,\(ModelConfig.seqLen)])];")
        b.appendLine("        tensor<fp16, [1,\(ModelConfig.dim),1,\(ModelConfig.seqLen)]> dffn = slice_by_size(x=x,begin=bd,size=sd)[name=string(\"s0\")];")
        b.appendLine("        tensor<int32, [4]> b1 = const()[name=string(\"b1\"), val=tensor<int32, [4]>([0,\(ModelConfig.dim),0,0])];")
        b.appendLine("        tensor<int32, [4]> s1 = const()[name=string(\"s1\"), val=tensor<int32, [4]>([1,\(ModelConfig.hidden),1,\(ModelConfig.seqLen)])];")
        b.appendLine("        tensor<fp16, [1,\(ModelConfig.hidden),1,\(ModelConfig.seqLen)]> h1 = slice_by_size(x=x,begin=b1,size=s1)[name=string(\"s1x\")];")
        b.appendLine("        tensor<int32, [4]> b3 = const()[name=string(\"b3\"), val=tensor<int32, [4]>([0,\(ModelConfig.dim + ModelConfig.hidden),0,0])];")
        b.appendLine("        tensor<fp16, [1,\(ModelConfig.hidden),1,\(ModelConfig.seqLen)]> h3 = slice_by_size(x=x,begin=b3,size=s1)[name=string(\"s3x\")];")
        b.appendLine("        tensor<fp16, [\(ModelConfig.hidden),\(ModelConfig.dim),1,1]> W2t = const()[name=string(\"W2t\"), val=tensor<fp16, [\(ModelConfig.hidden),\(ModelConfig.dim),1,1]>(BLOBFILE(path=string(\"@model_path/weights/w2t.bin\"), offset=uint64(64)))];")
        b.appendLine("        tensor<fp16, [1,\(ModelConfig.hidden),1,\(ModelConfig.seqLen)]> dsilu = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=W2t,x=dffn)[name=string(\"cw2\")];")
        b.appendLine("        tensor<fp16, [1,\(ModelConfig.hidden),1,\(ModelConfig.seqLen)]> sig = sigmoid(x=h1)[name=string(\"sg\")];")
        b.appendLine("        fp16 one = const()[name=string(\"one\"), val=fp16(1.0)];")
        b.appendLine("        tensor<fp16, [1,\(ModelConfig.hidden),1,\(ModelConfig.seqLen)]> oms = sub(x=one,y=sig)[name=string(\"oms\")];")
        b.appendLine("        tensor<fp16, [1,\(ModelConfig.hidden),1,\(ModelConfig.seqLen)]> homs = mul(x=h1,y=oms)[name=string(\"homs\")];")
        b.appendLine("        tensor<fp16, [1,\(ModelConfig.hidden),1,\(ModelConfig.seqLen)]> brk = add(x=one,y=homs)[name=string(\"brk\")];")
        b.appendLine("        tensor<fp16, [1,\(ModelConfig.hidden),1,\(ModelConfig.seqLen)]> dsd = mul(x=sig,y=brk)[name=string(\"dsd\")];")
        b.appendLine("        tensor<fp16, [1,\(ModelConfig.hidden),1,\(ModelConfig.seqLen)]> t1 = mul(x=dsilu,y=h3)[name=string(\"t1\")];")
        b.appendLine("        tensor<fp16, [1,\(ModelConfig.hidden),1,\(ModelConfig.seqLen)]> dh1 = mul(x=t1,y=dsd)[name=string(\"dh1\")];")
        b.appendLine("        tensor<fp16, [1,\(ModelConfig.hidden),1,\(ModelConfig.seqLen)]> slh = mul(x=h1,y=sig)[name=string(\"slh\")];")
        b.appendLine("        tensor<fp16, [1,\(ModelConfig.hidden),1,\(ModelConfig.seqLen)]> dh3 = mul(x=dsilu,y=slh)[name=string(\"dh3\")];")
        b.appendLine("        tensor<fp16, [\(ModelConfig.dim),\(ModelConfig.hidden),1,1]> W1t = const()[name=string(\"W1t\"), val=tensor<fp16, [\(ModelConfig.dim),\(ModelConfig.hidden),1,1]>(BLOBFILE(path=string(\"@model_path/weights/w1t.bin\"), offset=uint64(64)))];")
        b.appendLine("        tensor<fp16, [\(ModelConfig.dim),\(ModelConfig.hidden),1,1]> W3t = const()[name=string(\"W3t\"), val=tensor<fp16, [\(ModelConfig.dim),\(ModelConfig.hidden),1,1]>(BLOBFILE(path=string(\"@model_path/weights/w3t.bin\"), offset=uint64(64)))];")
        b.appendLine("        tensor<fp16, [1,\(ModelConfig.dim),1,\(ModelConfig.seqLen)]> dx1 = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=W1t,x=dh1)[name=string(\"cw1\")];")
        b.appendLine("        tensor<fp16, [1,\(ModelConfig.dim),1,\(ModelConfig.seqLen)]> dx3 = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=W3t,x=dh3)[name=string(\"cw3\")];")
        b.appendLine("        tensor<fp16, [1,\(ModelConfig.dim),1,\(ModelConfig.seqLen)]> dx = add(x=dx1,y=dx3)[name=string(\"adx\")];")
        b.appendLine("        int32 cax = const()[name=string(\"cax\"), val=int32(1)];")
        b.appendLine("        bool cid = const()[name=string(\"cid\"), val=bool(false)];")
        b.appendLine("        tensor<fp16, [1,\(ModelConfig.dim + 2 * ModelConfig.hidden),1,\(ModelConfig.seqLen)]> out = concat(axis=cax,interleave=cid,values=(dx,dh1,dh3))[name=string(\"cat\")];")
        b.appendLine("    } -> (out);")
        b.appendLine("}")
        return b.text
    }
}
