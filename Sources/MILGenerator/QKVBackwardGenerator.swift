import Foundation
import ANETypes

public struct QKVBackwardGenerator: MILProgramGenerator {
    public init() {}

    public var inputBytes: Int { 3 * ModelConfig.dim * ModelConfig.seqLen * 2 }
    public var outputByteSizes: [Int] { [ModelConfig.dim * ModelConfig.seqLen * 2] }

    public var milText: String {
        var b = MILBuilder(reserveCapacity: 8_192)
        b.append(MILText.header)
        b.appendLine("    func main<ios18>(tensor<fp16, [1, \(3 * ModelConfig.dim), 1, \(ModelConfig.seqLen)]> x) {")
        b.append(MILText.convConst)
        b.appendLine("        tensor<int32, [4]> sz = const()[name=string(\"sz\"), val=tensor<int32, [4]>([1,\(ModelConfig.dim),1,\(ModelConfig.seqLen)])];")
        b.appendLine("        tensor<int32, [4]> b0 = const()[name=string(\"b0\"), val=tensor<int32, [4]>([0,0,0,0])];")
        b.appendLine("        tensor<fp16, [1,\(ModelConfig.dim),1,\(ModelConfig.seqLen)]> dq = slice_by_size(x=x,begin=b0,size=sz)[name=string(\"s0\")];")
        b.appendLine("        tensor<int32, [4]> b1 = const()[name=string(\"b1\"), val=tensor<int32, [4]>([0,\(ModelConfig.dim),0,0])];")
        b.appendLine("        tensor<fp16, [1,\(ModelConfig.dim),1,\(ModelConfig.seqLen)]> dk = slice_by_size(x=x,begin=b1,size=sz)[name=string(\"s1\")];")
        b.appendLine("        tensor<int32, [4]> b2 = const()[name=string(\"b2\"), val=tensor<int32, [4]>([0,\(2 * ModelConfig.dim),0,0])];")
        b.appendLine("        tensor<fp16, [1,\(ModelConfig.dim),1,\(ModelConfig.seqLen)]> dv = slice_by_size(x=x,begin=b2,size=sz)[name=string(\"s2\")];")
        b.appendLine("        tensor<fp16, [\(ModelConfig.dim),\(ModelConfig.dim),1,1]> Wqt = const()[name=string(\"Wqt\"), val=tensor<fp16, [\(ModelConfig.dim),\(ModelConfig.dim),1,1]>(BLOBFILE(path=string(\"@model_path/weights/wqt.bin\"), offset=uint64(64)))];")
        b.appendLine("        tensor<fp16, [\(ModelConfig.dim),\(ModelConfig.dim),1,1]> Wkt = const()[name=string(\"Wkt\"), val=tensor<fp16, [\(ModelConfig.dim),\(ModelConfig.dim),1,1]>(BLOBFILE(path=string(\"@model_path/weights/wkt.bin\"), offset=uint64(64)))];")
        b.appendLine("        tensor<fp16, [\(ModelConfig.dim),\(ModelConfig.dim),1,1]> Wvt = const()[name=string(\"Wvt\"), val=tensor<fp16, [\(ModelConfig.dim),\(ModelConfig.dim),1,1]>(BLOBFILE(path=string(\"@model_path/weights/wvt.bin\"), offset=uint64(64)))];")
        b.appendLine("        tensor<fp16, [1,\(ModelConfig.dim),1,\(ModelConfig.seqLen)]> dxq = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=Wqt,x=dq)[name=string(\"cq\")];")
        b.appendLine("        tensor<fp16, [1,\(ModelConfig.dim),1,\(ModelConfig.seqLen)]> dxk = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=Wkt,x=dk)[name=string(\"ck\")];")
        b.appendLine("        tensor<fp16, [1,\(ModelConfig.dim),1,\(ModelConfig.seqLen)]> dxv = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=Wvt,x=dv)[name=string(\"cv\")];")
        b.appendLine("        tensor<fp16, [1,\(ModelConfig.dim),1,\(ModelConfig.seqLen)]> dxqk = add(x=dxq,y=dxk)[name=string(\"aqk\")];")
        b.appendLine("        tensor<fp16, [1,\(ModelConfig.dim),1,\(ModelConfig.seqLen)]> out = add(x=dxqk,y=dxv)[name=string(\"out\")];")
        b.appendLine("    } -> (out);")
        b.appendLine("}")
        return b.text
    }
}
