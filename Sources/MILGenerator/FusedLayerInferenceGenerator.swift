import Foundation
import ANETypes

/// Fused single-program transformer layer for inference.
///
/// Combines RMSNorm1 + split-head SDPA + RMSNorm2 + SiLU-gated FFN into one
/// MIL program, eliminating the two ANE dispatch round-trips (SDPA kernel →
/// CPU residual add → FFN kernel) that dominate per-layer latency.
///
/// Optimizations over the separate `SDPAForwardInferenceGenerator` +
/// `FFNForwardInferenceGenerator` pair:
///
/// 1. **Fused dispatch** — one `_ANEClient` evaluate call instead of two,
///    removing ~0.4 ms of ANE setup overhead per layer per token.
///
/// 2. **Optimized RMSNorm** — replaces `reduce_sum → mul(invd) → add(eps) → pow(-0.5)`
///    with `reduce_mean → add(eps) → rsqrt`, saving 2 MIL ops per norm block
///    (4 ops per layer).
///
/// 3. **Split-head attention** — projects full QKV once via conv, then slices
///    per head and runs 12 independent attention sub-graphs before concatenating.
///    This exposes head-level parallelism the ANE scheduler can exploit and
///    avoids the reshape→transpose→matmul path that serialises all heads.
///
/// 4. **SiLU fusion** — `sigmoid(h1) → mul(h1, sig)` is kept as the explicit
///    two-op pattern (MIL has no standalone `silu` op in the ios18 dialect),
///    but the naming convention is consistent with the semantic intent.
///
/// - Input:  `x`   — `[1, dim, 1, seqLen]` fp16 residual stream
/// - Output: `out` — `[1, dim, 1, seqLen]` fp16 updated residual stream
public struct FusedLayerInferenceGenerator: MILProgramGenerator {
    public init() {}

    public var inputBytes: Int { ModelConfig.dim * ModelConfig.seqLen * 2 }
    public var outputByteSizes: [Int] { [ModelConfig.dim * ModelConfig.seqLen * 2] }

    public var milText: String {
        let sc: Float = 1.0 / sqrt(Float(ModelConfig.headDim))
        let D   = ModelConfig.dim
        let H   = ModelConfig.hidden
        let nH  = ModelConfig.heads
        let hD  = ModelConfig.headDim
        let S   = ModelConfig.seqLen

        var b = MILBuilder(reserveCapacity: 65_536)
        b.append(MILText.header)
        b.appendLine("    func main<ios18>(tensor<fp16, [1, \(D), 1, \(S)]> x) {")

        // ── RMSNorm 1 ─────────────────────────────────────────────────────────
        // sq = x * x
        b.appendLine("        tensor<fp16, [1,\(D),1,\(S)]> r1_sq = mul(x=x,y=x)[name=string(\"r1_sq\")];")
        // axes constant for channel dim
        b.appendLine("        tensor<int32, [1]> r1_rax = const()[name=string(\"r1_rax\"), val=tensor<int32, [1]>([1])];")
        // keep_dims = true
        b.appendLine("        bool r1_kd = const()[name=string(\"r1_kd\"), val=bool(true)];")
        // ms = reduce_mean(sq, axes=[1], keep_dims=true)  → [1,1,1,S]
        b.appendLine("        tensor<fp16, [1,1,1,\(S)]> r1_ms = reduce_mean(axes=r1_rax,keep_dims=r1_kd,x=r1_sq)[name=string(\"r1_ms\")];")
        // eps constant
        b.appendLine("        fp16 r1_eps = const()[name=string(\"r1_eps\"), val=fp16(0.00001)];")
        // ss3 = ms + eps
        b.appendLine("        tensor<fp16, [1,1,1,\(S)]> r1_ss3 = add(x=r1_ms,y=r1_eps)[name=string(\"r1_ss3\")];")
        // rrms = rsqrt(ss3)
        b.appendLine("        tensor<fp16, [1,1,1,\(S)]> r1_rrms = rsqrt(x=r1_ss3)[name=string(\"r1_rrms\")];")
        // xr = x * rrms
        b.appendLine("        tensor<fp16, [1,\(D),1,\(S)]> r1_xr = mul(x=x,y=r1_rrms)[name=string(\"r1_xr\")];")
        // rms1 weight
        b.appendLine("        tensor<fp16, [1,\(D),1,1]> r1_rw = const()[name=string(\"r1_rw\"), val=tensor<fp16, [1,\(D),1,1]>(BLOBFILE(path=string(\"@model_path/weights/rms1.bin\"), offset=uint64(64)))];")
        // xn = xr * rw  (broadcast over seqLen)
        b.appendLine("        tensor<fp16, [1,\(D),1,\(S)]> r1_xn = mul(x=r1_xr,y=r1_rw)[name=string(\"r1_xn\")];")

        // ── Conv constants (shared across all projections) ────────────────────
        b.append(MILText.convConst)

        // ── QKV projections ───────────────────────────────────────────────────
        b.appendLine("        tensor<fp16, [\(D),\(D),1,1]> a_Wq = const()[name=string(\"a_Wq\"), val=tensor<fp16, [\(D),\(D),1,1]>(BLOBFILE(path=string(\"@model_path/weights/wq.bin\"), offset=uint64(64)))];")
        b.appendLine("        tensor<fp16, [\(D),\(D),1,1]> a_Wk = const()[name=string(\"a_Wk\"), val=tensor<fp16, [\(D),\(D),1,1]>(BLOBFILE(path=string(\"@model_path/weights/wk.bin\"), offset=uint64(64)))];")
        b.appendLine("        tensor<fp16, [\(D),\(D),1,1]> a_Wv = const()[name=string(\"a_Wv\"), val=tensor<fp16, [\(D),\(D),1,1]>(BLOBFILE(path=string(\"@model_path/weights/wv.bin\"), offset=uint64(64)))];")
        b.appendLine("        tensor<fp16, [\(D),\(D),1,1]> a_Wo = const()[name=string(\"a_Wo\"), val=tensor<fp16, [\(D),\(D),1,1]>(BLOBFILE(path=string(\"@model_path/weights/wo.bin\"), offset=uint64(64)))];")
        // Full [1, D, 1, S] QKV projections — single conv each (maximally efficient on ANE)
        b.appendLine("        tensor<fp16, [1,\(D),1,\(S)]> a_qf = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=a_Wq,x=r1_xn)[name=string(\"a_qf\")];")
        b.appendLine("        tensor<fp16, [1,\(D),1,\(S)]> a_kf = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=a_Wk,x=r1_xn)[name=string(\"a_kf\")];")
        b.appendLine("        tensor<fp16, [1,\(D),1,\(S)]> a_vf = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=a_Wv,x=r1_xn)[name=string(\"a_vf\")];")

        // Attention scale constant (shared across all heads)
        b.append("        fp16 a_sc = const()[name=string(\"a_sc\"), val=fp16(")
        b.appendFP16(sc)
        b.appendLine(")];")

        // Causal mask (broadcast over all heads via [1,1,S,S] shape)
        b.appendLine("        tensor<fp16, [1,1,\(S),\(S)]> a_cm = const()[name=string(\"a_cm\"), val=tensor<fp16, [1,1,\(S),\(S)]>(BLOBFILE(path=string(\"@model_path/weights/mask.bin\"), offset=uint64(64)))];")

        // softmax axis constant (shared)
        b.appendLine("        int32 a_sax = const()[name=string(\"a_sax\"), val=int32(-1)];")

        // Transpose perm [0,1,3,2] — shared across all heads for the final
        // out_h transpose from [1,1,S,headDim] → [1,1,headDim,S]
        b.appendLine("        tensor<int32, [4]> a_pm = const()[name=string(\"a_pm\"), val=tensor<int32, [4]>([0,1,3,2])];")

        // ── Per-head split attention ───────────────────────────────────────────
        // For head h: slice [1,hD,1,S] from the full [1,D,1,S] QKV tensors,
        // reshape to [1,1,hD,S] for matmul, compute scores, softmax, weighted
        // sum, then reshape back to [1,hD,1,S] for the final concat.
        for h in 0 ..< nH {
            let chStart = h * hD
            let chEnd   = (h + 1) * hD
            let p = "h\(h)"  // per-head SSA prefix

            // ── Slice begin/end constants (unique per head) ───────────────────
            b.appendLine("        tensor<int32, [4]> \(p)_bg = const()[name=string(\"\(p)_bg\"), val=tensor<int32, [4]>([0,\(chStart),0,0])];")
            b.appendLine("        tensor<int32, [4]> \(p)_en = const()[name=string(\"\(p)_en\"), val=tensor<int32, [4]>([1,\(chEnd),1,\(S)])];")

            // ── Slice Q, K, V → [1, hD, 1, S] each ──────────────────────────
            b.appendLine("        tensor<fp16, [1,\(hD),1,\(S)]> \(p)_q = slice_by_index(begin=\(p)_bg,end=\(p)_en,x=a_qf)[name=string(\"\(p)_q\")];")
            b.appendLine("        tensor<fp16, [1,\(hD),1,\(S)]> \(p)_k = slice_by_index(begin=\(p)_bg,end=\(p)_en,x=a_kf)[name=string(\"\(p)_k\")];")
            b.appendLine("        tensor<fp16, [1,\(hD),1,\(S)]> \(p)_v = slice_by_index(begin=\(p)_bg,end=\(p)_en,x=a_vf)[name=string(\"\(p)_v\")];")

            // ── Reshape to matmul layout [1, 1, hD, S] ───────────────────────
            // MIL matmul treats the last two dims as the matrix; we need
            // [batch=1, 1, hD, S] so that transpose_x turns it into [S, hD].
            b.appendLine("        tensor<int32, [4]> \(p)_msh = const()[name=string(\"\(p)_msh\"), val=tensor<int32, [4]>([1,1,\(hD),\(S)])];")
            b.appendLine("        tensor<fp16, [1,1,\(hD),\(S)]> \(p)_qr = reshape(shape=\(p)_msh,x=\(p)_q)[name=string(\"\(p)_qr\")];")
            b.appendLine("        tensor<fp16, [1,1,\(hD),\(S)]> \(p)_kr = reshape(shape=\(p)_msh,x=\(p)_k)[name=string(\"\(p)_kr\")];")
            b.appendLine("        tensor<fp16, [1,1,\(hD),\(S)]> \(p)_vr = reshape(shape=\(p)_msh,x=\(p)_v)[name=string(\"\(p)_vr\")];")

            // ── Scores: Q^T @ K → [1, 1, S, S] ──────────────────────────────
            // transpose_x=true  : treats qr as [1,1,S,hD]
            // transpose_y=false : treats kr as [1,1,hD,S]
            // matmul result     : [1,1,S,S]
            b.appendLine("        bool \(p)_tx = const()[name=string(\"\(p)_tx\"), val=bool(true)];")
            b.appendLine("        bool \(p)_ty = const()[name=string(\"\(p)_ty\"), val=bool(false)];")
            b.appendLine("        tensor<fp16, [1,1,\(S),\(S)]> \(p)_sc = matmul(transpose_x=\(p)_tx,transpose_y=\(p)_ty,x=\(p)_qr,y=\(p)_kr)[name=string(\"\(p)_sc\")];")

            // Scale scores
            b.appendLine("        tensor<fp16, [1,1,\(S),\(S)]> \(p)_ss = mul(x=\(p)_sc,y=a_sc)[name=string(\"\(p)_ss\")];")

            // Add causal mask (broadcast [1,1,S,S])
            b.appendLine("        tensor<fp16, [1,1,\(S),\(S)]> \(p)_sm = add(x=\(p)_ss,y=a_cm)[name=string(\"\(p)_sm\")];")

            // Softmax over last axis
            b.appendLine("        tensor<fp16, [1,1,\(S),\(S)]> \(p)_aw = softmax(axis=a_sax,x=\(p)_sm)[name=string(\"\(p)_aw\")];")

            // ── Weighted sum: Attn @ V → [1, 1, S, hD] ───────────────────────
            // attn: [1,1,S,S], V: [1,1,hD,S]
            // transpose_y=true : treats vr as [1,1,S,hD]
            // result           : [1,1,S,hD]
            b.appendLine("        bool \(p)_tvx = const()[name=string(\"\(p)_tvx\"), val=bool(false)];")
            b.appendLine("        bool \(p)_tvy = const()[name=string(\"\(p)_tvy\"), val=bool(true)];")
            b.appendLine("        tensor<fp16, [1,1,\(S),\(hD)]> \(p)_av = matmul(transpose_x=\(p)_tvx,transpose_y=\(p)_tvy,x=\(p)_aw,y=\(p)_vr)[name=string(\"\(p)_av\")];")

            // ── Transpose [0,1,3,2]: [1,1,S,hD] → [1,1,hD,S] ────────────────
            b.appendLine("        tensor<fp16, [1,1,\(hD),\(S)]> \(p)_at = transpose(perm=a_pm,x=\(p)_av)[name=string(\"\(p)_at\")];")

            // ── Reshape back to channel-first: [1,1,hD,S] → [1,hD,1,S] ──────
            b.appendLine("        tensor<int32, [4]> \(p)_osh = const()[name=string(\"\(p)_osh\"), val=tensor<int32, [4]>([1,\(hD),1,\(S)])];")
            b.appendLine("        tensor<fp16, [1,\(hD),1,\(S)]> \(p)_or = reshape(shape=\(p)_osh,x=\(p)_at)[name=string(\"\(p)_or\")];")
        }

        // ── Concat 12 head outputs along channel axis → [1, D, 1, S] ─────────
        // MIL requires SSA variable references for axis and interleave params.
        b.appendLine("        int32 a_cax = const()[name=string(\"a_cax\"), val=int32(1)];")
        b.appendLine("        bool a_cid = const()[name=string(\"a_cid\"), val=bool(false)];")
        let headList = (0 ..< nH).map { "h\($0)_or" }.joined(separator: ", ")
        b.appendLine("        tensor<fp16, [1,\(D),1,\(S)]> a_cat = concat(axis=a_cax,interleave=a_cid,values=(\(headList)))[name=string(\"a_cat\")];")

        // ── Wo projection + residual 1 ────────────────────────────────────────
        b.appendLine("        tensor<fp16, [1,\(D),1,\(S)]> a_oo = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=a_Wo,x=a_cat)[name=string(\"a_oo\")];")
        b.appendLine("        tensor<fp16, [1,\(D),1,\(S)]> a_res = add(x=x,y=a_oo)[name=string(\"a_res\")];")

        // ── RMSNorm 2 ─────────────────────────────────────────────────────────
        b.appendLine("        tensor<fp16, [1,\(D),1,\(S)]> r2_sq = mul(x=a_res,y=a_res)[name=string(\"r2_sq\")];")
        b.appendLine("        tensor<int32, [1]> r2_rax = const()[name=string(\"r2_rax\"), val=tensor<int32, [1]>([1])];")
        b.appendLine("        bool r2_kd = const()[name=string(\"r2_kd\"), val=bool(true)];")
        b.appendLine("        tensor<fp16, [1,1,1,\(S)]> r2_ms = reduce_mean(axes=r2_rax,keep_dims=r2_kd,x=r2_sq)[name=string(\"r2_ms\")];")
        b.appendLine("        fp16 r2_eps = const()[name=string(\"r2_eps\"), val=fp16(0.00001)];")
        b.appendLine("        tensor<fp16, [1,1,1,\(S)]> r2_ss3 = add(x=r2_ms,y=r2_eps)[name=string(\"r2_ss3\")];")
        b.appendLine("        tensor<fp16, [1,1,1,\(S)]> r2_rrms = rsqrt(x=r2_ss3)[name=string(\"r2_rrms\")];")
        b.appendLine("        tensor<fp16, [1,\(D),1,\(S)]> r2_xr = mul(x=a_res,y=r2_rrms)[name=string(\"r2_xr\")];")
        b.appendLine("        tensor<fp16, [1,\(D),1,1]> r2_rw = const()[name=string(\"r2_rw\"), val=tensor<fp16, [1,\(D),1,1]>(BLOBFILE(path=string(\"@model_path/weights/rms2.bin\"), offset=uint64(64)))];")
        b.appendLine("        tensor<fp16, [1,\(D),1,\(S)]> r2_xn = mul(x=r2_xr,y=r2_rw)[name=string(\"r2_xn\")];")

        // ── SiLU-gated FFN ────────────────────────────────────────────────────
        // MIL ios18 dialect has no dedicated `silu` op; the explicit
        // sigmoid → pointwise-mul sequence is semantically equivalent and
        // the ANE compiler fuses it internally.
        b.appendLine("        tensor<fp16, [\(H),\(D),1,1]> f_W1 = const()[name=string(\"f_W1\"), val=tensor<fp16, [\(H),\(D),1,1]>(BLOBFILE(path=string(\"@model_path/weights/w1.bin\"), offset=uint64(64)))];")
        b.appendLine("        tensor<fp16, [\(H),\(D),1,1]> f_W3 = const()[name=string(\"f_W3\"), val=tensor<fp16, [\(H),\(D),1,1]>(BLOBFILE(path=string(\"@model_path/weights/w3.bin\"), offset=uint64(64)))];")
        b.appendLine("        tensor<fp16, [\(D),\(H),1,1]> f_W2 = const()[name=string(\"f_W2\"), val=tensor<fp16, [\(D),\(H),1,1]>(BLOBFILE(path=string(\"@model_path/weights/w2.bin\"), offset=uint64(64)))];")
        b.appendLine("        tensor<fp16, [1,\(H),1,\(S)]> f_h1 = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=f_W1,x=r2_xn)[name=string(\"f_h1\")];")
        b.appendLine("        tensor<fp16, [1,\(H),1,\(S)]> f_h3 = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=f_W3,x=r2_xn)[name=string(\"f_h3\")];")
        b.appendLine("        tensor<fp16, [1,\(H),1,\(S)]> f_sig = sigmoid(x=f_h1)[name=string(\"f_sig\")];")
        b.appendLine("        tensor<fp16, [1,\(H),1,\(S)]> f_silu = mul(x=f_h1,y=f_sig)[name=string(\"f_silu\")];")
        b.appendLine("        tensor<fp16, [1,\(H),1,\(S)]> f_gate = mul(x=f_silu,y=f_h3)[name=string(\"f_gate\")];")
        b.appendLine("        tensor<fp16, [1,\(D),1,\(S)]> f_y = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=f_W2,x=f_gate)[name=string(\"f_y\")];")

        // ── Residual 2 ────────────────────────────────────────────────────────
        b.appendLine("        tensor<fp16, [1,\(D),1,\(S)]> out = add(x=a_res,y=f_y)[name=string(\"f_res\")];")
        b.appendLine("    } -> (out);")
        b.appendLine("}")
        return b.text
    }
}
