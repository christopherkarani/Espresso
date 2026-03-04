import Foundation
import ANETypes

public struct SDPABackward2Generator: MILProgramGenerator {
    public init() {}

    public var inputBytes: Int { (2 * ModelConfig.scoreCh + 2 * ModelConfig.dim) * ModelConfig.seqLen * 2 }
    public var outputByteSizes: [Int] { [2 * ModelConfig.dim * ModelConfig.seqLen * 2] }

    public var milText: String {
        let sc: Float = 1.0 / sqrt(Float(ModelConfig.headDim))
        let bwd2In = 2 * ModelConfig.scoreCh + 2 * ModelConfig.dim

        var b = MILBuilder(reserveCapacity: 13_312)
        b.append(MILText.header)
        b.appendLine("    func main<ios18>(tensor<fp16, [1, \(bwd2In), 1, \(ModelConfig.seqLen)]> x) {")
        b.appendLine("        tensor<int32, [4]> sz_sc = const()[name=string(\"szsc\"), val=tensor<int32, [4]>([1,\(ModelConfig.scoreCh),1,\(ModelConfig.seqLen)])];")
        b.appendLine("        tensor<int32, [4]> b0 = const()[name=string(\"b0\"), val=tensor<int32, [4]>([0,0,0,0])];")
        b.appendLine("        tensor<fp16, [1,\(ModelConfig.scoreCh),1,\(ModelConfig.seqLen)]> pf = slice_by_size(x=x,begin=b0,size=sz_sc)[name=string(\"s0\")];")
        b.appendLine("        tensor<int32, [4]> b1 = const()[name=string(\"b1\"), val=tensor<int32, [4]>([0,\(ModelConfig.scoreCh),0,0])];")
        b.appendLine("        tensor<fp16, [1,\(ModelConfig.scoreCh),1,\(ModelConfig.seqLen)]> dpf = slice_by_size(x=x,begin=b1,size=sz_sc)[name=string(\"s1\")];")
        b.appendLine("        tensor<int32, [4]> sz_d = const()[name=string(\"szd\"), val=tensor<int32, [4]>([1,\(ModelConfig.dim),1,\(ModelConfig.seqLen)])];")
        b.appendLine("        tensor<int32, [4]> b2 = const()[name=string(\"b2\"), val=tensor<int32, [4]>([0,\(2 * ModelConfig.scoreCh),0,0])];")
        b.appendLine("        tensor<fp16, [1,\(ModelConfig.dim),1,\(ModelConfig.seqLen)]> qf = slice_by_size(x=x,begin=b2,size=sz_d)[name=string(\"s2\")];")
        b.appendLine("        tensor<int32, [4]> b3 = const()[name=string(\"b3\"), val=tensor<int32, [4]>([0,\(2 * ModelConfig.scoreCh + ModelConfig.dim),0,0])];")
        b.appendLine("        tensor<fp16, [1,\(ModelConfig.dim),1,\(ModelConfig.seqLen)]> kf = slice_by_size(x=x,begin=b3,size=sz_d)[name=string(\"s3\")];")
        b.appendLine("        tensor<int32, [4]> ssh = const()[name=string(\"ssh\"), val=tensor<int32, [4]>([1,\(ModelConfig.heads),\(ModelConfig.seqLen),\(ModelConfig.seqLen)])];")
        b.appendLine("        tensor<fp16, [1,\(ModelConfig.heads),\(ModelConfig.seqLen),\(ModelConfig.seqLen)]> probs = reshape(shape=ssh,x=pf)[name=string(\"rp\")];")
        b.appendLine("        tensor<fp16, [1,\(ModelConfig.heads),\(ModelConfig.seqLen),\(ModelConfig.seqLen)]> dp = reshape(shape=ssh,x=dpf)[name=string(\"rdp\")];")
        b.appendLine("        tensor<int32, [4]> rsh = const()[name=string(\"rsh\"), val=tensor<int32, [4]>([1,\(ModelConfig.heads),\(ModelConfig.headDim),\(ModelConfig.seqLen)])];")
        b.appendLine("        tensor<int32, [4]> pm = const()[name=string(\"pm\"), val=tensor<int32, [4]>([0,1,3,2])];")
        b.appendLine("        tensor<fp16, [1,\(ModelConfig.heads),\(ModelConfig.headDim),\(ModelConfig.seqLen)]> qr = reshape(shape=rsh,x=qf)[name=string(\"rq\")];")
        b.appendLine("        tensor<fp16, [1,\(ModelConfig.heads),\(ModelConfig.seqLen),\(ModelConfig.headDim)]> q = transpose(perm=pm,x=qr)[name=string(\"tq\")];")
        b.appendLine("        tensor<fp16, [1,\(ModelConfig.heads),\(ModelConfig.headDim),\(ModelConfig.seqLen)]> kr = reshape(shape=rsh,x=kf)[name=string(\"rk\")];")
        b.appendLine("        tensor<fp16, [1,\(ModelConfig.heads),\(ModelConfig.seqLen),\(ModelConfig.headDim)]> k = transpose(perm=pm,x=kr)[name=string(\"tk\")];")
        b.appendLine("        tensor<fp16, [1,\(ModelConfig.heads),\(ModelConfig.seqLen),\(ModelConfig.seqLen)]> pdp = mul(x=probs,y=dp)[name=string(\"pdp\")];")
        b.appendLine("        tensor<int32, [1]> rax = const()[name=string(\"rax\"), val=tensor<int32, [1]>([-1])];")
        b.appendLine("        bool kd = const()[name=string(\"kd\"), val=bool(true)];")
        b.appendLine("        tensor<fp16, [1,\(ModelConfig.heads),\(ModelConfig.seqLen),1]> spdp = reduce_sum(x=pdp,axes=rax,keep_dims=kd)[name=string(\"rs\")];")
        b.appendLine("        tensor<fp16, [1,\(ModelConfig.heads),\(ModelConfig.seqLen),\(ModelConfig.seqLen)]> dps = sub(x=dp,y=spdp)[name=string(\"dps\")];")
        b.appendLine("        tensor<fp16, [1,\(ModelConfig.heads),\(ModelConfig.seqLen),\(ModelConfig.seqLen)]> ds0 = mul(x=probs,y=dps)[name=string(\"ds0\")];")
        b.append("        fp16 scv = const()[name=string(\"scv\"), val=fp16(")
        b.appendFP16(sc)
        b.appendLine(")];")
        b.appendLine("        tensor<fp16, [1,\(ModelConfig.heads),\(ModelConfig.seqLen),\(ModelConfig.seqLen)]> ds = mul(x=ds0,y=scv)[name=string(\"ds\")];")
        b.appendLine("        bool bF = const()[name=string(\"bF\"), val=bool(false)];")
        b.appendLine("        bool bT = const()[name=string(\"bT\"), val=bool(true)];")
        b.appendLine("        tensor<fp16, [1,\(ModelConfig.heads),\(ModelConfig.seqLen),\(ModelConfig.headDim)]> dq4 = matmul(transpose_x=bF,transpose_y=bF,x=ds,y=k)[name=string(\"dq\")];")
        b.appendLine("        tensor<fp16, [1,\(ModelConfig.heads),\(ModelConfig.seqLen),\(ModelConfig.headDim)]> dk4 = matmul(transpose_x=bT,transpose_y=bF,x=ds,y=q)[name=string(\"dk\")];")
        b.appendLine("        tensor<fp16, [1,\(ModelConfig.heads),\(ModelConfig.headDim),\(ModelConfig.seqLen)]> dqt = transpose(perm=pm,x=dq4)[name=string(\"dqt\")];")
        b.appendLine("        tensor<fp16, [1,\(ModelConfig.heads),\(ModelConfig.headDim),\(ModelConfig.seqLen)]> dkt = transpose(perm=pm,x=dk4)[name=string(\"dkt\")];")
        b.appendLine("        tensor<int32, [4]> fs = const()[name=string(\"fs\"), val=tensor<int32, [4]>([1,\(ModelConfig.dim),1,\(ModelConfig.seqLen)])];")
        b.appendLine("        tensor<fp16, [1,\(ModelConfig.dim),1,\(ModelConfig.seqLen)]> dqf = reshape(shape=fs,x=dqt)[name=string(\"dqf\")];")
        b.appendLine("        tensor<fp16, [1,\(ModelConfig.dim),1,\(ModelConfig.seqLen)]> dkf = reshape(shape=fs,x=dkt)[name=string(\"dkf\")];")
        b.appendLine("        int32 cax = const()[name=string(\"cax\"), val=int32(1)];")
        b.appendLine("        bool cid = const()[name=string(\"cid\"), val=bool(false)];")
        b.appendLine("        tensor<fp16, [1,\(2 * ModelConfig.dim),1,\(ModelConfig.seqLen)]> out = concat(axis=cax,interleave=cid,values=(dqf,dkf))[name=string(\"cat\")];")
        b.appendLine("    } -> (out);")
        b.appendLine("}")
        return b.text
    }
}
