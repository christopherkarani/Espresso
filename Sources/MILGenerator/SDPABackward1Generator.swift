import Foundation
import ANETypes

public struct SDPABackward1Generator: MILProgramGenerator {
    public init() {}

    public var inputBytes: Int { 4 * ModelConfig.dim * ModelConfig.seqLen * 2 }
    public var outputByteSizes: [Int] { [(ModelConfig.dim + 2 * ModelConfig.scoreCh) * ModelConfig.seqLen * 2] }

    public var milText: String {
        let sc: Float = 1.0 / sqrt(Float(ModelConfig.headDim))

        var b = MILBuilder(reserveCapacity: 14_336)
        b.append(MILText.header)
        b.appendLine("    func main<ios18>(tensor<fp16, [1, \(4 * ModelConfig.dim), 1, \(ModelConfig.seqLen)]> x) {")
        b.append(MILText.convConst)
        b.appendLine("        tensor<int32, [4]> sz = const()[name=string(\"sz\"), val=tensor<int32, [4]>([1,\(ModelConfig.dim),1,\(ModelConfig.seqLen)])];")
        b.appendLine("        tensor<int32, [4]> b0 = const()[name=string(\"b0\"), val=tensor<int32, [4]>([0,0,0,0])];")
        b.appendLine("        tensor<fp16, [1,\(ModelConfig.dim),1,\(ModelConfig.seqLen)]> qf = slice_by_size(x=x,begin=b0,size=sz)[name=string(\"s0\")];")
        b.appendLine("        tensor<int32, [4]> b1 = const()[name=string(\"b1\"), val=tensor<int32, [4]>([0,\(ModelConfig.dim),0,0])];")
        b.appendLine("        tensor<fp16, [1,\(ModelConfig.dim),1,\(ModelConfig.seqLen)]> kf = slice_by_size(x=x,begin=b1,size=sz)[name=string(\"s1\")];")
        b.appendLine("        tensor<int32, [4]> b2 = const()[name=string(\"b2\"), val=tensor<int32, [4]>([0,\(2 * ModelConfig.dim),0,0])];")
        b.appendLine("        tensor<fp16, [1,\(ModelConfig.dim),1,\(ModelConfig.seqLen)]> vf = slice_by_size(x=x,begin=b2,size=sz)[name=string(\"s2\")];")
        b.appendLine("        tensor<int32, [4]> b3 = const()[name=string(\"b3\"), val=tensor<int32, [4]>([0,\(3 * ModelConfig.dim),0,0])];")
        b.appendLine("        tensor<fp16, [1,\(ModelConfig.dim),1,\(ModelConfig.seqLen)]> dx2f = slice_by_size(x=x,begin=b3,size=sz)[name=string(\"s3\")];")
        b.appendLine("        tensor<fp16, [\(ModelConfig.dim),\(ModelConfig.dim),1,1]> Wot = const()[name=string(\"Wot\"), val=tensor<fp16, [\(ModelConfig.dim),\(ModelConfig.dim),1,1]>(BLOBFILE(path=string(\"@model_path/weights/wot.bin\"), offset=uint64(64)))];")
        b.appendLine("        tensor<fp16, [1,\(ModelConfig.dim),1,\(ModelConfig.seqLen)]> df = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=Wot,x=dx2f)[name=string(\"cwo\")];")
        b.appendLine("        tensor<int32, [4]> rsh = const()[name=string(\"rsh\"), val=tensor<int32, [4]>([1,\(ModelConfig.heads),\(ModelConfig.headDim),\(ModelConfig.seqLen)])];")
        b.appendLine("        tensor<int32, [4]> pm = const()[name=string(\"pm\"), val=tensor<int32, [4]>([0,1,3,2])];")
        b.appendLine("        tensor<fp16, [1,\(ModelConfig.heads),\(ModelConfig.headDim),\(ModelConfig.seqLen)]> qr = reshape(shape=rsh,x=qf)[name=string(\"rq\")];")
        b.appendLine("        tensor<fp16, [1,\(ModelConfig.heads),\(ModelConfig.seqLen),\(ModelConfig.headDim)]> q = transpose(perm=pm,x=qr)[name=string(\"tq\")];")
        b.appendLine("        tensor<fp16, [1,\(ModelConfig.heads),\(ModelConfig.headDim),\(ModelConfig.seqLen)]> kr = reshape(shape=rsh,x=kf)[name=string(\"rk\")];")
        b.appendLine("        tensor<fp16, [1,\(ModelConfig.heads),\(ModelConfig.seqLen),\(ModelConfig.headDim)]> k = transpose(perm=pm,x=kr)[name=string(\"tk\")];")
        b.appendLine("        tensor<fp16, [1,\(ModelConfig.heads),\(ModelConfig.headDim),\(ModelConfig.seqLen)]> vr = reshape(shape=rsh,x=vf)[name=string(\"rv\")];")
        b.appendLine("        tensor<fp16, [1,\(ModelConfig.heads),\(ModelConfig.seqLen),\(ModelConfig.headDim)]> v = transpose(perm=pm,x=vr)[name=string(\"tv\")];")
        b.appendLine("        tensor<fp16, [1,\(ModelConfig.heads),\(ModelConfig.headDim),\(ModelConfig.seqLen)]> dr = reshape(shape=rsh,x=df)[name=string(\"rd\")];")
        b.appendLine("        tensor<fp16, [1,\(ModelConfig.heads),\(ModelConfig.seqLen),\(ModelConfig.headDim)]> da = transpose(perm=pm,x=dr)[name=string(\"td\")];")
        b.appendLine("        bool bF = const()[name=string(\"bF\"), val=bool(false)];")
        b.appendLine("        bool bT = const()[name=string(\"bT\"), val=bool(true)];")
        b.appendLine("        tensor<fp16, [1,\(ModelConfig.heads),\(ModelConfig.seqLen),\(ModelConfig.seqLen)]> sc1 = matmul(transpose_x=bF,transpose_y=bT,x=q,y=k)[name=string(\"mm1\")];")
        b.append("        fp16 scv = const()[name=string(\"scv\"), val=fp16(")
        b.appendFP16(sc)
        b.appendLine(")];")
        b.appendLine("        tensor<fp16, [1,\(ModelConfig.heads),\(ModelConfig.seqLen),\(ModelConfig.seqLen)]> sc2 = mul(x=sc1,y=scv)[name=string(\"scl\")];")
        b.appendLine("        tensor<fp16, [1,1,\(ModelConfig.seqLen),\(ModelConfig.seqLen)]> cm = const()[name=string(\"cm\"), val=tensor<fp16, [1,1,\(ModelConfig.seqLen),\(ModelConfig.seqLen)]>(BLOBFILE(path=string(\"@model_path/weights/mask.bin\"), offset=uint64(64)))];")
        b.appendLine("        tensor<fp16, [1,\(ModelConfig.heads),\(ModelConfig.seqLen),\(ModelConfig.seqLen)]> ms = add(x=sc2,y=cm)[name=string(\"msk\")];")
        b.appendLine("        int32 sax = const()[name=string(\"sax\"), val=int32(-1)];")
        b.appendLine("        tensor<fp16, [1,\(ModelConfig.heads),\(ModelConfig.seqLen),\(ModelConfig.seqLen)]> probs = softmax(axis=sax,x=ms)[name=string(\"sm\")];")
        b.appendLine("        tensor<fp16, [1,\(ModelConfig.heads),\(ModelConfig.seqLen),\(ModelConfig.headDim)]> dv4 = matmul(transpose_x=bT,transpose_y=bF,x=probs,y=da)[name=string(\"dv\")];")
        b.appendLine("        tensor<fp16, [1,\(ModelConfig.heads),\(ModelConfig.seqLen),\(ModelConfig.seqLen)]> dp4 = matmul(transpose_x=bF,transpose_y=bT,x=da,y=v)[name=string(\"dp\")];")
        b.appendLine("        tensor<fp16, [1,\(ModelConfig.heads),\(ModelConfig.headDim),\(ModelConfig.seqLen)]> dvt = transpose(perm=pm,x=dv4)[name=string(\"dvt\")];")
        b.appendLine("        tensor<int32, [4]> dvs = const()[name=string(\"dvs\"), val=tensor<int32, [4]>([1,\(ModelConfig.dim),1,\(ModelConfig.seqLen)])];")
        b.appendLine("        tensor<fp16, [1,\(ModelConfig.dim),1,\(ModelConfig.seqLen)]> dvf = reshape(shape=dvs,x=dvt)[name=string(\"dvf\")];")
        b.appendLine("        tensor<int32, [4]> scs = const()[name=string(\"scs\"), val=tensor<int32, [4]>([1,\(ModelConfig.scoreCh),1,\(ModelConfig.seqLen)])];")
        b.appendLine("        tensor<fp16, [1,\(ModelConfig.scoreCh),1,\(ModelConfig.seqLen)]> pf = reshape(shape=scs,x=probs)[name=string(\"pf\")];")
        b.appendLine("        tensor<fp16, [1,\(ModelConfig.scoreCh),1,\(ModelConfig.seqLen)]> dpf = reshape(shape=scs,x=dp4)[name=string(\"dpf\")];")
        b.appendLine("        int32 cax = const()[name=string(\"cax\"), val=int32(1)];")
        b.appendLine("        bool cid = const()[name=string(\"cid\"), val=bool(false)];")
        b.appendLine("        tensor<fp16, [1,\(ModelConfig.dim + 2 * ModelConfig.scoreCh),1,\(ModelConfig.seqLen)]> out = concat(axis=cax,interleave=cid,values=(dvf,pf,dpf))[name=string(\"cat\")];")
        b.appendLine("    } -> (out);")
        b.appendLine("}")
        return b.text
    }
}
