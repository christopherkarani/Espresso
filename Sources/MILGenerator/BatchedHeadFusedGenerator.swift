import Foundation
import ANETypes

/// Fused single-program transformer layer for inference using batched-head attention.
///
/// Identical to `FusedLayerInferenceGenerator` except the attention section uses
/// reshape → transpose → batched matmul instead of per-head slice_by_index + concat.
///
/// The batched-head path issues a single Q@K^T matmul over all 12 heads simultaneously
/// (`[1,12,256,64] × [1,12,64,256] → [1,12,256,256]`) rather than 12 independent
/// `[1,1,S,S]` matmuls. Whether the ANE scheduler exploits this more efficiently than
/// split-head is what the benchmark is measuring.
///
/// - Input:  `x`   — `[1, dim, 1, seqLen]` fp16 residual stream
/// - Output: `out` — `[1, dim, 1, seqLen]` fp16 updated residual stream
public struct BatchedHeadFusedGenerator: MILProgramGenerator {
    public init() {}

    public var inputBytes: Int { ModelConfig.dim * ModelConfig.seqLen * 2 }
    public var outputByteSizes: [Int] { [ModelConfig.dim * ModelConfig.seqLen * 2] }

    public var milText: String {
        let sc: Float = 1.0 / sqrt(Float(ModelConfig.headDim))
        let invd: Float = 1.0 / Float(ModelConfig.dim)
        let D   = ModelConfig.dim
        let H   = ModelConfig.hidden
        let nH  = ModelConfig.heads
        let hD  = ModelConfig.headDim
        let S   = ModelConfig.seqLen

        var b = MILBuilder(reserveCapacity: 65_536)
        b.append(MILText.header)
        b.appendLine("    func main<ios18>(tensor<fp16, [1, \(D), 1, \(S)]> x) {")

        // ── RMSNorm 1 ───────────────────────────────────────────────────────────
        b.appendLine("        tensor<fp16, [1,\(D),1,\(S)]> r1_sq = mul(x=x,y=x)[name=string(\"r1_sq\")];")
        b.appendLine("        tensor<int32, [1]> r1_rax = const()[name=string(\"r1_rax\"), val=tensor<int32, [1]>([1])];")
        b.appendLine("        bool r1_kd = const()[name=string(\"r1_kd\"), val=bool(true)];")
        b.appendLine("        tensor<fp16, [1,1,1,\(S)]> r1_ss = reduce_sum(x=r1_sq,axes=r1_rax,keep_dims=r1_kd)[name=string(\"r1_ss\")];")
        b.append("        fp16 r1_invd = const()[name=string(\"r1_invd\"), val=fp16(")
        b.appendFP16(invd)
        b.appendLine(")];")
        b.appendLine("        tensor<fp16, [1,1,1,\(S)]> r1_ss2 = mul(x=r1_ss,y=r1_invd)[name=string(\"r1_ss2\")];")
        b.appendLine("        fp16 r1_eps = const()[name=string(\"r1_eps\"), val=fp16(0.00001)];")
        b.appendLine("        tensor<fp16, [1,1,1,\(S)]> r1_ss3 = add(x=r1_ss2,y=r1_eps)[name=string(\"r1_ss3\")];")
        b.appendLine("        fp16 r1_nhalf = const()[name=string(\"r1_nhalf\"), val=fp16(-0.5)];")
        b.appendLine("        tensor<fp16, [1,1,1,\(S)]> r1_rrms = pow(x=r1_ss3,y=r1_nhalf)[name=string(\"r1_rrms\")];")
        b.appendLine("        tensor<fp16, [1,\(D),1,\(S)]> r1_xr = mul(x=x,y=r1_rrms)[name=string(\"r1_xr\")];")
        b.appendLine("        tensor<fp16, [1,\(D),1,1]> r1_rw = const()[name=string(\"r1_rw\"), val=tensor<fp16, [1,\(D),1,1]>(BLOBFILE(path=string(\"@model_path/weights/rms1.bin\"), offset=uint64(64)))];")
        b.appendLine("        tensor<fp16, [1,\(D),1,\(S)]> r1_xn = mul(x=r1_xr,y=r1_rw)[name=string(\"r1_xn\")];")

        // ── Conv constants (shared across all projections) ─────────────────────
        b.append(MILText.convConst)

        // ── QKV projections ────────────────────────────────────────────────────
        b.appendLine("        tensor<fp16, [\(D),\(D),1,1]> a_Wq = const()[name=string(\"a_Wq\"), val=tensor<fp16, [\(D),\(D),1,1]>(BLOBFILE(path=string(\"@model_path/weights/wq.bin\"), offset=uint64(64)))];")
        b.appendLine("        tensor<fp16, [\(D),\(D),1,1]> a_Wk = const()[name=string(\"a_Wk\"), val=tensor<fp16, [\(D),\(D),1,1]>(BLOBFILE(path=string(\"@model_path/weights/wk.bin\"), offset=uint64(64)))];")
        b.appendLine("        tensor<fp16, [\(D),\(D),1,1]> a_Wv = const()[name=string(\"a_Wv\"), val=tensor<fp16, [\(D),\(D),1,1]>(BLOBFILE(path=string(\"@model_path/weights/wv.bin\"), offset=uint64(64)))];")
        b.appendLine("        tensor<fp16, [\(D),\(D),1,1]> a_Wo = const()[name=string(\"a_Wo\"), val=tensor<fp16, [\(D),\(D),1,1]>(BLOBFILE(path=string(\"@model_path/weights/wo.bin\"), offset=uint64(64)))];")
        b.appendLine("        tensor<fp16, [1,\(D),1,\(S)]> a_qf = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=a_Wq,x=r1_xn)[name=string(\"a_qf\")];")
        b.appendLine("        tensor<fp16, [1,\(D),1,\(S)]> a_kf = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=a_Wk,x=r1_xn)[name=string(\"a_kf\")];")
        b.appendLine("        tensor<fp16, [1,\(D),1,\(S)]> a_vf = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=a_Wv,x=r1_xn)[name=string(\"a_vf\")];")

        // ── Shared attention constants ──────────────────────────────────────────
        b.append("        fp16 a_sc = const()[name=string(\"a_sc\"), val=fp16(")
        b.appendFP16(sc)
        b.appendLine(")];")
        b.appendLine("        tensor<fp16, [1,1,\(S),\(S)]> a_cm = const()[name=string(\"a_cm\"), val=tensor<fp16, [1,1,\(S),\(S)]>(BLOBFILE(path=string(\"@model_path/weights/mask.bin\"), offset=uint64(64)))];")
        b.appendLine("        int32 a_sax = const()[name=string(\"a_sax\"), val=int32(-1)];")
        b.appendLine("        tensor<int32, [4]> a_pm = const()[name=string(\"a_pm\"), val=tensor<int32, [4]>([0,1,3,2])];")
        b.appendLine("        bool a_fy = const()[name=string(\"a_fy\"), val=bool(false)];")
        b.appendLine("        bool a_ty = const()[name=string(\"a_ty\"), val=bool(true)];")

        // ── Batched-head attention ──────────────────────────────────────────────
        // Reshape: [1, D, 1, S] → [1, nH, hD, S]
        b.appendLine("        tensor<int32, [4]> a_hsh = const()[name=string(\"a_hsh\"), val=tensor<int32, [4]>([1,\(nH),\(hD),\(S)])];")
        b.appendLine("        tensor<fp16, [1,\(nH),\(hD),\(S)]> a_q4 = reshape(shape=a_hsh,x=a_qf)[name=string(\"a_q4\")];")
        b.appendLine("        tensor<fp16, [1,\(nH),\(hD),\(S)]> a_k4 = reshape(shape=a_hsh,x=a_kf)[name=string(\"a_k4\")];")
        b.appendLine("        tensor<fp16, [1,\(nH),\(hD),\(S)]> a_v4 = reshape(shape=a_hsh,x=a_vf)[name=string(\"a_v4\")];")

        // Transpose: [1, nH, hD, S] → [1, nH, S, hD]  (swap last two dims)
        b.appendLine("        tensor<fp16, [1,\(nH),\(S),\(hD)]> a_qt = transpose(perm=a_pm,x=a_q4)[name=string(\"a_qt\")];")
        b.appendLine("        tensor<fp16, [1,\(nH),\(S),\(hD)]> a_kt = transpose(perm=a_pm,x=a_k4)[name=string(\"a_kt\")];")
        b.appendLine("        tensor<fp16, [1,\(nH),\(S),\(hD)]> a_vt = transpose(perm=a_pm,x=a_v4)[name=string(\"a_vt\")];")

        // Q @ K^T → [1, nH, S, S]  (transpose_x=false, transpose_y=true)
        b.appendLine("        tensor<fp16, [1,\(nH),\(S),\(S)]> a_qk = matmul(transpose_x=a_fy,transpose_y=a_ty,x=a_qt,y=a_kt)[name=string(\"a_qk\")];")

        // Scale + causal mask + softmax
        b.appendLine("        tensor<fp16, [1,\(nH),\(S),\(S)]> a_ss = mul(x=a_qk,y=a_sc)[name=string(\"a_ss\")];")
        b.appendLine("        tensor<fp16, [1,\(nH),\(S),\(S)]> a_ms = add(x=a_ss,y=a_cm)[name=string(\"a_ms\")];")
        b.appendLine("        tensor<fp16, [1,\(nH),\(S),\(S)]> a_aw = softmax(axis=a_sax,x=a_ms)[name=string(\"a_aw\")];")

        // Scores @ V → [1, nH, S, hD]  (transpose_x=false, transpose_y=false)
        b.appendLine("        tensor<fp16, [1,\(nH),\(S),\(hD)]> a_av = matmul(transpose_x=a_fy,transpose_y=a_fy,x=a_aw,y=a_vt)[name=string(\"a_av\")];")

        // Transpose back: [1, nH, S, hD] → [1, nH, hD, S]
        b.appendLine("        tensor<fp16, [1,\(nH),\(hD),\(S)]> a_at = transpose(perm=a_pm,x=a_av)[name=string(\"a_at\")];")

        // Reshape back: [1, nH, hD, S] → [1, D, 1, S]
        b.appendLine("        tensor<int32, [4]> a_osh = const()[name=string(\"a_osh\"), val=tensor<int32, [4]>([1,\(D),1,\(S)])];")
        b.appendLine("        tensor<fp16, [1,\(D),1,\(S)]> a_af = reshape(shape=a_osh,x=a_at)[name=string(\"a_af\")];")

        // ── Wo projection + residual 1 ─────────────────────────────────────────
        b.appendLine("        tensor<fp16, [1,\(D),1,\(S)]> a_oo = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=a_Wo,x=a_af)[name=string(\"a_oo\")];")
        b.appendLine("        tensor<fp16, [1,\(D),1,\(S)]> a_res = add(x=x,y=a_oo)[name=string(\"a_res\")];")

        // ── RMSNorm 2 ───────────────────────────────────────────────────────────
        b.appendLine("        tensor<fp16, [1,\(D),1,\(S)]> r2_sq = mul(x=a_res,y=a_res)[name=string(\"r2_sq\")];")
        b.appendLine("        tensor<int32, [1]> r2_rax = const()[name=string(\"r2_rax\"), val=tensor<int32, [1]>([1])];")
        b.appendLine("        bool r2_kd = const()[name=string(\"r2_kd\"), val=bool(true)];")
        b.appendLine("        tensor<fp16, [1,1,1,\(S)]> r2_ss = reduce_sum(x=r2_sq,axes=r2_rax,keep_dims=r2_kd)[name=string(\"r2_ss\")];")
        b.append("        fp16 r2_invd = const()[name=string(\"r2_invd\"), val=fp16(")
        b.appendFP16(invd)
        b.appendLine(")];")
        b.appendLine("        tensor<fp16, [1,1,1,\(S)]> r2_ss2 = mul(x=r2_ss,y=r2_invd)[name=string(\"r2_ss2\")];")
        b.appendLine("        fp16 r2_eps = const()[name=string(\"r2_eps\"), val=fp16(0.00001)];")
        b.appendLine("        tensor<fp16, [1,1,1,\(S)]> r2_ss3 = add(x=r2_ss2,y=r2_eps)[name=string(\"r2_ss3\")];")
        b.appendLine("        fp16 r2_nhalf = const()[name=string(\"r2_nhalf\"), val=fp16(-0.5)];")
        b.appendLine("        tensor<fp16, [1,1,1,\(S)]> r2_rrms = pow(x=r2_ss3,y=r2_nhalf)[name=string(\"r2_rrms\")];")
        b.appendLine("        tensor<fp16, [1,\(D),1,\(S)]> r2_xr = mul(x=a_res,y=r2_rrms)[name=string(\"r2_xr\")];")
        b.appendLine("        tensor<fp16, [1,\(D),1,1]> r2_rw = const()[name=string(\"r2_rw\"), val=tensor<fp16, [1,\(D),1,1]>(BLOBFILE(path=string(\"@model_path/weights/rms2.bin\"), offset=uint64(64)))];")
        b.appendLine("        tensor<fp16, [1,\(D),1,\(S)]> r2_xn = mul(x=r2_xr,y=r2_rw)[name=string(\"r2_xn\")];")

        // ── SiLU-gated FFN ─────────────────────────────────────────────────────
        b.appendLine("        tensor<fp16, [\(H),\(D),1,1]> f_W1 = const()[name=string(\"f_W1\"), val=tensor<fp16, [\(H),\(D),1,1]>(BLOBFILE(path=string(\"@model_path/weights/w1.bin\"), offset=uint64(64)))];")
        b.appendLine("        tensor<fp16, [\(H),\(D),1,1]> f_W3 = const()[name=string(\"f_W3\"), val=tensor<fp16, [\(H),\(D),1,1]>(BLOBFILE(path=string(\"@model_path/weights/w3.bin\"), offset=uint64(64)))];")
        b.appendLine("        tensor<fp16, [\(D),\(H),1,1]> f_W2 = const()[name=string(\"f_W2\"), val=tensor<fp16, [\(D),\(H),1,1]>(BLOBFILE(path=string(\"@model_path/weights/w2.bin\"), offset=uint64(64)))];")
        b.appendLine("        tensor<fp16, [1,\(H),1,\(S)]> f_h1 = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=f_W1,x=r2_xn)[name=string(\"f_h1\")];")
        b.appendLine("        tensor<fp16, [1,\(H),1,\(S)]> f_h3 = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=f_W3,x=r2_xn)[name=string(\"f_h3\")];")
        b.appendLine("        tensor<fp16, [1,\(H),1,\(S)]> f_sig = sigmoid(x=f_h1)[name=string(\"f_sig\")];")
        b.appendLine("        tensor<fp16, [1,\(H),1,\(S)]> f_silu = mul(x=f_h1,y=f_sig)[name=string(\"f_silu\")];")
        b.appendLine("        tensor<fp16, [1,\(H),1,\(S)]> f_gate = mul(x=f_silu,y=f_h3)[name=string(\"f_gate\")];")
        b.appendLine("        tensor<fp16, [1,\(D),1,\(S)]> f_y = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=f_W2,x=f_gate)[name=string(\"f_y\")];")

        // ── Residual 2 ─────────────────────────────────────────────────────────
        b.appendLine("        tensor<fp16, [1,\(D),1,\(S)]> out = add(x=a_res,y=f_y)[name=string(\"f_res\")];")
        b.appendLine("    } -> (out);")
        b.appendLine("}")
        return b.text
    }
}
