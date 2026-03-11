import Foundation
import ANETypes

/// Inference-only SDPA forward kernel with fused residual addition.
///
/// Unlike `SDPAForwardGenerator` which outputs 6 concatenated tensors
/// (oo, qf, kf, vf, af, xn) for backward pass, this outputs only the
/// residual-fused result: `x2 = x + oo` — a single `[1, dim, 1, seqLen]` tensor.
///
/// This eliminates ~5x output data (no Q/K/V/attnOut/xnorm written to surface)
/// and removes the CPU-side `vDSP_vadd` residual addition.
public struct SDPAForwardInferenceGenerator: MILProgramGenerator {
    public init() {}

    public var inputBytes: Int { ModelConfig.dim * ModelConfig.seqLen * 2 }
    public var outputByteSizes: [Int] { [ModelConfig.dim * ModelConfig.seqLen * 2] }

    public var milText: String {
        let sc: Float = 1.0 / sqrt(Float(ModelConfig.headDim))
        let invd: Float = 1.0 / Float(ModelConfig.dim)

        var b = MILBuilder(reserveCapacity: 16_384)
        b.append(MILText.header)
        b.appendLine("    func main<ios18>(tensor<fp16, [1, \(ModelConfig.dim), 1, \(ModelConfig.seqLen)]> x) {")

        // RMSNorm
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
        b.appendLine("        tensor<fp16, [1,\(ModelConfig.dim),1,1]> rw = const()[name=string(\"rw\"), val=tensor<fp16, [1,\(ModelConfig.dim),1,1]>(BLOBFILE(path=string(\"@model_path/weights/rms1.bin\"), offset=uint64(64)))];")
        b.appendLine("        tensor<fp16, [1,\(ModelConfig.dim),1,\(ModelConfig.seqLen)]> xn = mul(x=xr,y=rw)[name=string(\"xn\")];")

        // Conv constants + QKV projections
        b.append(MILText.convConst)
        b.appendLine("        tensor<fp16, [\(ModelConfig.dim),\(ModelConfig.dim),1,1]> Wq = const()[name=string(\"Wq\"), val=tensor<fp16, [\(ModelConfig.dim),\(ModelConfig.dim),1,1]>(BLOBFILE(path=string(\"@model_path/weights/wq.bin\"), offset=uint64(64)))];")
        b.appendLine("        tensor<fp16, [\(ModelConfig.dim),\(ModelConfig.dim),1,1]> Wk = const()[name=string(\"Wk\"), val=tensor<fp16, [\(ModelConfig.dim),\(ModelConfig.dim),1,1]>(BLOBFILE(path=string(\"@model_path/weights/wk.bin\"), offset=uint64(64)))];")
        b.appendLine("        tensor<fp16, [\(ModelConfig.dim),\(ModelConfig.dim),1,1]> Wv = const()[name=string(\"Wv\"), val=tensor<fp16, [\(ModelConfig.dim),\(ModelConfig.dim),1,1]>(BLOBFILE(path=string(\"@model_path/weights/wv.bin\"), offset=uint64(64)))];")
        b.appendLine("        tensor<fp16, [\(ModelConfig.dim),\(ModelConfig.dim),1,1]> Wo = const()[name=string(\"Wo\"), val=tensor<fp16, [\(ModelConfig.dim),\(ModelConfig.dim),1,1]>(BLOBFILE(path=string(\"@model_path/weights/wo.bin\"), offset=uint64(64)))];")
        b.appendLine("        tensor<fp16, [1,\(ModelConfig.dim),1,\(ModelConfig.seqLen)]> qf = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=Wq,x=xn)[name=string(\"cq\")];")
        b.appendLine("        tensor<fp16, [1,\(ModelConfig.dim),1,\(ModelConfig.seqLen)]> kf = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=Wk,x=xn)[name=string(\"ck\")];")
        b.appendLine("        tensor<fp16, [1,\(ModelConfig.dim),1,\(ModelConfig.seqLen)]> vf = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=Wv,x=xn)[name=string(\"cv\")];")

        // Attention: reshape → transpose → matmul → softmax → matmul → transpose → reshape → Wo
        b.appendLine("        tensor<int32, [4]> qsh = const()[name=string(\"qsh\"), val=tensor<int32, [4]>([1,\(ModelConfig.heads),\(ModelConfig.headDim),\(ModelConfig.seqLen)])];")
        b.appendLine("        tensor<int32, [4]> pm = const()[name=string(\"pm\"), val=tensor<int32, [4]>([0,1,3,2])];")
        b.appendLine("        tensor<fp16, [1,\(ModelConfig.heads),\(ModelConfig.headDim),\(ModelConfig.seqLen)]> q4 = reshape(shape=qsh,x=qf)[name=string(\"rq\")];")
        b.appendLine("        tensor<fp16, [1,\(ModelConfig.heads),\(ModelConfig.seqLen),\(ModelConfig.headDim)]> q = transpose(perm=pm,x=q4)[name=string(\"tq\")];")
        b.appendLine("        tensor<fp16, [1,\(ModelConfig.heads),\(ModelConfig.headDim),\(ModelConfig.seqLen)]> k4 = reshape(shape=qsh,x=kf)[name=string(\"rk\")];")
        b.appendLine("        tensor<fp16, [1,\(ModelConfig.heads),\(ModelConfig.seqLen),\(ModelConfig.headDim)]> k = transpose(perm=pm,x=k4)[name=string(\"tk\")];")
        b.appendLine("        tensor<fp16, [1,\(ModelConfig.heads),\(ModelConfig.headDim),\(ModelConfig.seqLen)]> v4 = reshape(shape=qsh,x=vf)[name=string(\"rv\")];")
        b.appendLine("        tensor<fp16, [1,\(ModelConfig.heads),\(ModelConfig.seqLen),\(ModelConfig.headDim)]> v = transpose(perm=pm,x=v4)[name=string(\"tv\")];")
        b.appendLine("        bool tx = const()[name=string(\"tx\"), val=bool(false)];")
        b.appendLine("        bool ty = const()[name=string(\"ty\"), val=bool(true)];")
        b.appendLine("        tensor<fp16, [1,\(ModelConfig.heads),\(ModelConfig.seqLen),\(ModelConfig.seqLen)]> sc1 = matmul(transpose_x=tx,transpose_y=ty,x=q,y=k)[name=string(\"mm1\")];")
        b.append("        fp16 scv = const()[name=string(\"scv\"), val=fp16(")
        b.appendFP16(sc)
        b.appendLine(")];")
        b.appendLine("        tensor<fp16, [1,\(ModelConfig.heads),\(ModelConfig.seqLen),\(ModelConfig.seqLen)]> sc2 = mul(x=sc1,y=scv)[name=string(\"scl\")];")
        b.appendLine("        tensor<fp16, [1,1,\(ModelConfig.seqLen),\(ModelConfig.seqLen)]> cm = const()[name=string(\"cm\"), val=tensor<fp16, [1,1,\(ModelConfig.seqLen),\(ModelConfig.seqLen)]>(BLOBFILE(path=string(\"@model_path/weights/mask.bin\"), offset=uint64(64)))];")
        b.appendLine("        tensor<fp16, [1,\(ModelConfig.heads),\(ModelConfig.seqLen),\(ModelConfig.seqLen)]> ms = add(x=sc2,y=cm)[name=string(\"msk\")];")
        b.appendLine("        int32 sax = const()[name=string(\"sax\"), val=int32(-1)];")
        b.appendLine("        tensor<fp16, [1,\(ModelConfig.heads),\(ModelConfig.seqLen),\(ModelConfig.seqLen)]> aw = softmax(axis=sax,x=ms)[name=string(\"sm\")];")
        b.appendLine("        tensor<fp16, [1,\(ModelConfig.heads),\(ModelConfig.seqLen),\(ModelConfig.headDim)]> a4 = matmul(transpose_x=tx,transpose_y=tx,x=aw,y=v)[name=string(\"mm2\")];")
        b.appendLine("        tensor<fp16, [1,\(ModelConfig.heads),\(ModelConfig.headDim),\(ModelConfig.seqLen)]> at = transpose(perm=pm,x=a4)[name=string(\"ta\")];")
        b.appendLine("        tensor<int32, [4]> os = const()[name=string(\"os\"), val=tensor<int32, [4]>([1,\(ModelConfig.dim),1,\(ModelConfig.seqLen)])];")
        b.appendLine("        tensor<fp16, [1,\(ModelConfig.dim),1,\(ModelConfig.seqLen)]> af = reshape(shape=os,x=at)[name=string(\"ra\")];")
        b.appendLine("        tensor<fp16, [1,\(ModelConfig.dim),1,\(ModelConfig.seqLen)]> oo = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=Wo,x=af)[name=string(\"co\")];")

        // Fused residual: x2 = x + oo (instead of concat for backward)
        b.appendLine("        tensor<fp16, [1,\(ModelConfig.dim),1,\(ModelConfig.seqLen)]> out = add(x=x,y=oo)[name=string(\"res\")];")
        b.appendLine("    } -> (out);")
        b.appendLine("}")
        return b.text
    }
}
