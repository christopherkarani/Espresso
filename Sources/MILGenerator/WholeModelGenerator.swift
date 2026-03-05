import Foundation
import ANETypes

/// Whole-model inference generator.
///
/// Produces a single MIL program with all `nLayers` transformer layers chained
/// together, feeding `L{i}_out` as the input to layer `i+1`. This eliminates
/// the per-layer ANE dispatch round-trips and intermediate IOSurface copies,
/// reducing the per-token latency to a single `eval()` call.
///
/// **Program structure:**
/// - Shared constants emitted once (conv params, attention constants, per-head
///   slice indices, causal mask, RMSNorm scalars)
/// - Per-layer weight blobs prefixed `L{i}_` (9 blobs × N layers + 1 shared mask)
/// - Layer chain: `x` → `L0_out` → `L1_out` → … → `out`
///
/// - Input:  `x`   — `[1, dim, 1, seqLen]` fp16 residual stream
/// - Output: `out` — `[1, dim, 1, seqLen]` fp16 updated residual stream after
///                   all `nLayers` transformer blocks
public struct WholeModelGenerator: MILProgramGenerator {
    public let nLayers: Int

    public init(nLayers: Int = ModelConfig.nLayers) {
        self.nLayers = nLayers
    }

    public var inputBytes: Int { ModelConfig.dim * ModelConfig.seqLen * 2 }
    public var outputByteSizes: [Int] { [ModelConfig.dim * ModelConfig.seqLen * 2] }

    public var milText: String {
        let sc: Float   = 1.0 / sqrt(Float(ModelConfig.headDim))
        let invd: Float = 1.0 / Float(ModelConfig.dim)
        let D   = ModelConfig.dim
        let H   = ModelConfig.hidden
        let nH  = ModelConfig.heads
        let hD  = ModelConfig.headDim
        let S   = ModelConfig.seqLen

        // Reserve generously — a 12-layer program is ~70 KB of MIL text.
        var b = MILBuilder(reserveCapacity: 131_072)

        b.append(MILText.header)
        b.appendLine("    func main<ios18>(tensor<fp16, [1, \(D), 1, \(S)]> x) {")

        // ── Shared conv params (emitted once, reused by every conv in all layers) ─
        b.append(MILText.convConst)

        // ── Shared attention constants ─────────────────────────────────────────
        b.append("        fp16 a_sc = const()[name=string(\"a_sc\"), val=fp16(")
        b.appendFP16(sc)
        b.appendLine(")];")
        b.appendLine("        tensor<fp16, [1,1,\(S),\(S)]> a_cm = const()[name=string(\"a_cm\"), val=tensor<fp16, [1,1,\(S),\(S)]>(BLOBFILE(path=string(\"@model_path/weights/mask.bin\"), offset=uint64(64)))];")
        b.appendLine("        int32 a_sax = const()[name=string(\"a_sax\"), val=int32(-1)];")
        b.appendLine("        tensor<int32, [4]> a_pm = const()[name=string(\"a_pm\"), val=tensor<int32, [4]>([0,1,3,2])];")
        b.appendLine("        bool a_tx = const()[name=string(\"a_tx\"), val=bool(true)];")
        b.appendLine("        bool a_fy = const()[name=string(\"a_fy\"), val=bool(false)];")
        b.appendLine("        bool a_tvy = const()[name=string(\"a_tvy\"), val=bool(true)];")
        b.appendLine("        tensor<bool, [4]> a_bm = const()[name=string(\"a_bm\"), val=tensor<bool, [4]>([false, false, false, false])];")
        b.appendLine("        tensor<bool, [4]> a_em = const()[name=string(\"a_em\"), val=tensor<bool, [4]>([false, false, false, false])];")
        b.appendLine("        tensor<bool, [4]> a_sm = const()[name=string(\"a_sm\"), val=tensor<bool, [4]>([false, false, false, false])];")
        b.appendLine("        int32 a_cax = const()[name=string(\"a_cax\"), val=int32(1)];")
        b.appendLine("        bool a_cid = const()[name=string(\"a_cid\"), val=bool(false)];")

        // ── Shared RMSNorm scalars ─────────────────────────────────────────────
        b.append("        fp16 r_invd = const()[name=string(\"r_invd\"), val=fp16(")
        b.appendFP16(invd)
        b.appendLine(")];")
        b.appendLine("        fp16 r_eps = const()[name=string(\"r_eps\"), val=fp16(0.00001)];")
        b.appendLine("        fp16 r_nhalf = const()[name=string(\"r_nhalf\"), val=fp16(-0.5)];")
        b.appendLine("        tensor<int32, [1]> r_rax = const()[name=string(\"r_rax\"), val=tensor<int32, [1]>([1])];")
        b.appendLine("        bool r_kd = const()[name=string(\"r_kd\"), val=bool(true)];")

        // ── Shared per-head slice indices and reshape shapes (same for all layers) ─
        for h in 0 ..< nH {
            let chStart = h * hD
            let chEnd   = (h + 1) * hD
            let p = "h\(h)"
            b.appendLine("        tensor<int32, [4]> \(p)_bg = const()[name=string(\"\(p)_bg\"), val=tensor<int32, [4]>([0,\(chStart),0,0])];")
            b.appendLine("        tensor<int32, [4]> \(p)_en = const()[name=string(\"\(p)_en\"), val=tensor<int32, [4]>([1,\(chEnd),1,\(S)])];")
            b.appendLine("        tensor<int32, [4]> \(p)_msh = const()[name=string(\"\(p)_msh\"), val=tensor<int32, [4]>([1,1,\(hD),\(S)])];")
            b.appendLine("        tensor<int32, [4]> \(p)_osh = const()[name=string(\"\(p)_osh\"), val=tensor<int32, [4]>([1,\(hD),1,\(S)])];")
        }

        // ── Layer chain ────────────────────────────────────────────────────────
        for i in 0 ..< nLayers {
            let inputVar  = i == 0 ? "x" : "L\(i - 1)_out"
            let outputVar = i == nLayers - 1 ? "out" : "L\(i)_out"
            emitLayer(
                b: &b,
                layerIndex: i,
                inputVar: inputVar,
                outputVar: outputVar,
                D: D, H: H, S: S, nH: nH, hD: hD
            )
        }

        b.appendLine("    } -> (out);")
        b.appendLine("}")
        return b.text
    }

    // ── Private layer emitter ────────────────────────────────────────────────

    private func emitLayer(
        b: inout MILBuilder,
        layerIndex i: Int,
        inputVar: String,
        outputVar: String,
        D: Int, H: Int, S: Int, nH: Int, hD: Int
    ) {
        let lp = "L\(i)"   // layer prefix, e.g. "L0"

        // ── Per-layer weight blobs ───────────────────────────────────────────
        b.appendLine("        tensor<fp16, [1,\(D),1,1]> \(lp)_r1_rw = const()[name=string(\"\(lp)_r1_rw\"), val=tensor<fp16, [1,\(D),1,1]>(BLOBFILE(path=string(\"@model_path/weights/L\(i)_rms1.bin\"), offset=uint64(64)))];")
        b.appendLine("        tensor<fp16, [\(D),\(D),1,1]> \(lp)_a_Wq = const()[name=string(\"\(lp)_a_Wq\"), val=tensor<fp16, [\(D),\(D),1,1]>(BLOBFILE(path=string(\"@model_path/weights/L\(i)_wq.bin\"), offset=uint64(64)))];")
        b.appendLine("        tensor<fp16, [\(D),\(D),1,1]> \(lp)_a_Wk = const()[name=string(\"\(lp)_a_Wk\"), val=tensor<fp16, [\(D),\(D),1,1]>(BLOBFILE(path=string(\"@model_path/weights/L\(i)_wk.bin\"), offset=uint64(64)))];")
        b.appendLine("        tensor<fp16, [\(D),\(D),1,1]> \(lp)_a_Wv = const()[name=string(\"\(lp)_a_Wv\"), val=tensor<fp16, [\(D),\(D),1,1]>(BLOBFILE(path=string(\"@model_path/weights/L\(i)_wv.bin\"), offset=uint64(64)))];")
        b.appendLine("        tensor<fp16, [\(D),\(D),1,1]> \(lp)_a_Wo = const()[name=string(\"\(lp)_a_Wo\"), val=tensor<fp16, [\(D),\(D),1,1]>(BLOBFILE(path=string(\"@model_path/weights/L\(i)_wo.bin\"), offset=uint64(64)))];")
        b.appendLine("        tensor<fp16, [1,\(D),1,1]> \(lp)_r2_rw = const()[name=string(\"\(lp)_r2_rw\"), val=tensor<fp16, [1,\(D),1,1]>(BLOBFILE(path=string(\"@model_path/weights/L\(i)_rms2.bin\"), offset=uint64(64)))];")
        b.appendLine("        tensor<fp16, [\(H),\(D),1,1]> \(lp)_f_W1 = const()[name=string(\"\(lp)_f_W1\"), val=tensor<fp16, [\(H),\(D),1,1]>(BLOBFILE(path=string(\"@model_path/weights/L\(i)_w1.bin\"), offset=uint64(64)))];")
        b.appendLine("        tensor<fp16, [\(H),\(D),1,1]> \(lp)_f_W3 = const()[name=string(\"\(lp)_f_W3\"), val=tensor<fp16, [\(H),\(D),1,1]>(BLOBFILE(path=string(\"@model_path/weights/L\(i)_w3.bin\"), offset=uint64(64)))];")
        b.appendLine("        tensor<fp16, [\(D),\(H),1,1]> \(lp)_f_W2 = const()[name=string(\"\(lp)_f_W2\"), val=tensor<fp16, [\(D),\(H),1,1]>(BLOBFILE(path=string(\"@model_path/weights/L\(i)_w2.bin\"), offset=uint64(64)))];")

        // ── RMSNorm 1 ────────────────────────────────────────────────────────
        b.appendLine("        tensor<fp16, [1,\(D),1,\(S)]> \(lp)_r1_sq = mul(x=\(inputVar),y=\(inputVar))[name=string(\"\(lp)_r1_sq\")];")
        b.appendLine("        tensor<fp16, [1,1,1,\(S)]> \(lp)_r1_ss = reduce_sum(x=\(lp)_r1_sq,axes=r_rax,keep_dims=r_kd)[name=string(\"\(lp)_r1_ss\")];")
        b.appendLine("        tensor<fp16, [1,1,1,\(S)]> \(lp)_r1_ss2 = mul(x=\(lp)_r1_ss,y=r_invd)[name=string(\"\(lp)_r1_ss2\")];")
        b.appendLine("        tensor<fp16, [1,1,1,\(S)]> \(lp)_r1_ss3 = add(x=\(lp)_r1_ss2,y=r_eps)[name=string(\"\(lp)_r1_ss3\")];")
        b.appendLine("        tensor<fp16, [1,1,1,\(S)]> \(lp)_r1_rrms = pow(x=\(lp)_r1_ss3,y=r_nhalf)[name=string(\"\(lp)_r1_rrms\")];")
        b.appendLine("        tensor<fp16, [1,\(D),1,\(S)]> \(lp)_r1_xr = mul(x=\(inputVar),y=\(lp)_r1_rrms)[name=string(\"\(lp)_r1_xr\")];")
        b.appendLine("        tensor<fp16, [1,\(D),1,\(S)]> \(lp)_r1_xn = mul(x=\(lp)_r1_xr,y=\(lp)_r1_rw)[name=string(\"\(lp)_r1_xn\")];")

        // ── QKV projections ──────────────────────────────────────────────────
        b.appendLine("        tensor<fp16, [1,\(D),1,\(S)]> \(lp)_a_qf = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=\(lp)_a_Wq,x=\(lp)_r1_xn)[name=string(\"\(lp)_a_qf\")];")
        b.appendLine("        tensor<fp16, [1,\(D),1,\(S)]> \(lp)_a_kf = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=\(lp)_a_Wk,x=\(lp)_r1_xn)[name=string(\"\(lp)_a_kf\")];")
        b.appendLine("        tensor<fp16, [1,\(D),1,\(S)]> \(lp)_a_vf = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=\(lp)_a_Wv,x=\(lp)_r1_xn)[name=string(\"\(lp)_a_vf\")];")

        // ── Split-head attention ─────────────────────────────────────────────
        for h in 0 ..< nH {
            let hp = "h\(h)"    // shared head prefix
            let hv = "\(lp)_\(hp)"  // per-layer-per-head variable prefix

            // Slice Q, K, V (using shared slice constants)
            b.appendLine("        tensor<fp16, [1,\(hD),1,\(S)]> \(hv)_q = slice_by_index(begin=\(hp)_bg,begin_mask=a_bm,end=\(hp)_en,end_mask=a_em,squeeze_mask=a_sm,x=\(lp)_a_qf)[name=string(\"\(hv)_q\")];")
            b.appendLine("        tensor<fp16, [1,\(hD),1,\(S)]> \(hv)_k = slice_by_index(begin=\(hp)_bg,begin_mask=a_bm,end=\(hp)_en,end_mask=a_em,squeeze_mask=a_sm,x=\(lp)_a_kf)[name=string(\"\(hv)_k\")];")
            b.appendLine("        tensor<fp16, [1,\(hD),1,\(S)]> \(hv)_v = slice_by_index(begin=\(hp)_bg,begin_mask=a_bm,end=\(hp)_en,end_mask=a_em,squeeze_mask=a_sm,x=\(lp)_a_vf)[name=string(\"\(hv)_v\")];")

            // Reshape [1, hD, 1, S] → [1, 1, hD, S] (using shared shape consts)
            b.appendLine("        tensor<fp16, [1,1,\(hD),\(S)]> \(hv)_qr = reshape(shape=\(hp)_msh,x=\(hv)_q)[name=string(\"\(hv)_qr\")];")
            b.appendLine("        tensor<fp16, [1,1,\(hD),\(S)]> \(hv)_kr = reshape(shape=\(hp)_msh,x=\(hv)_k)[name=string(\"\(hv)_kr\")];")
            b.appendLine("        tensor<fp16, [1,1,\(hD),\(S)]> \(hv)_vr = reshape(shape=\(hp)_msh,x=\(hv)_v)[name=string(\"\(hv)_vr\")];")

            // Q^T @ K → [1, 1, S, S]
            b.appendLine("        tensor<fp16, [1,1,\(S),\(S)]> \(hv)_sc = matmul(transpose_x=a_tx,transpose_y=a_fy,x=\(hv)_qr,y=\(hv)_kr)[name=string(\"\(hv)_sc\")];")

            // Scale + causal mask + softmax
            b.appendLine("        tensor<fp16, [1,1,\(S),\(S)]> \(hv)_ss = mul(x=\(hv)_sc,y=a_sc)[name=string(\"\(hv)_ss\")];")
            b.appendLine("        tensor<fp16, [1,1,\(S),\(S)]> \(hv)_ms = add(x=\(hv)_ss,y=a_cm)[name=string(\"\(hv)_ms\")];")
            b.appendLine("        tensor<fp16, [1,1,\(S),\(S)]> \(hv)_aw = softmax(axis=a_sax,x=\(hv)_ms)[name=string(\"\(hv)_aw\")];")

            // Attn @ V^T → [1, 1, S, hD]
            b.appendLine("        tensor<fp16, [1,1,\(S),\(hD)]> \(hv)_av = matmul(transpose_x=a_fy,transpose_y=a_tvy,x=\(hv)_aw,y=\(hv)_vr)[name=string(\"\(hv)_av\")];")

            // Transpose [1,1,S,hD] → [1,1,hD,S] + reshape → [1, hD, 1, S]
            b.appendLine("        tensor<fp16, [1,1,\(hD),\(S)]> \(hv)_at = transpose(perm=a_pm,x=\(hv)_av)[name=string(\"\(hv)_at\")];")
            b.appendLine("        tensor<fp16, [1,\(hD),1,\(S)]> \(hv)_or = reshape(shape=\(hp)_osh,x=\(hv)_at)[name=string(\"\(hv)_or\")];")
        }

        // ── Concat all heads → [1, D, 1, S] ─────────────────────────────────
        let headList = (0 ..< nH).map { "L\(i)_h\($0)_or" }.joined(separator: ", ")
        b.appendLine("        tensor<fp16, [1,\(D),1,\(S)]> \(lp)_a_cat = concat(axis=a_cax,interleave=a_cid,values=(\(headList)))[name=string(\"\(lp)_a_cat\")];")

        // ── Wo projection + residual 1 ────────────────────────────────────────
        b.appendLine("        tensor<fp16, [1,\(D),1,\(S)]> \(lp)_a_oo = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=\(lp)_a_Wo,x=\(lp)_a_cat)[name=string(\"\(lp)_a_oo\")];")
        b.appendLine("        tensor<fp16, [1,\(D),1,\(S)]> \(lp)_a_res = add(x=\(inputVar),y=\(lp)_a_oo)[name=string(\"\(lp)_a_res\")];")

        // ── RMSNorm 2 ────────────────────────────────────────────────────────
        b.appendLine("        tensor<fp16, [1,\(D),1,\(S)]> \(lp)_r2_sq = mul(x=\(lp)_a_res,y=\(lp)_a_res)[name=string(\"\(lp)_r2_sq\")];")
        b.appendLine("        tensor<fp16, [1,1,1,\(S)]> \(lp)_r2_ss = reduce_sum(x=\(lp)_r2_sq,axes=r_rax,keep_dims=r_kd)[name=string(\"\(lp)_r2_ss\")];")
        b.appendLine("        tensor<fp16, [1,1,1,\(S)]> \(lp)_r2_ss2 = mul(x=\(lp)_r2_ss,y=r_invd)[name=string(\"\(lp)_r2_ss2\")];")
        b.appendLine("        tensor<fp16, [1,1,1,\(S)]> \(lp)_r2_ss3 = add(x=\(lp)_r2_ss2,y=r_eps)[name=string(\"\(lp)_r2_ss3\")];")
        b.appendLine("        tensor<fp16, [1,1,1,\(S)]> \(lp)_r2_rrms = pow(x=\(lp)_r2_ss3,y=r_nhalf)[name=string(\"\(lp)_r2_rrms\")];")
        b.appendLine("        tensor<fp16, [1,\(D),1,\(S)]> \(lp)_r2_xr = mul(x=\(lp)_a_res,y=\(lp)_r2_rrms)[name=string(\"\(lp)_r2_xr\")];")
        b.appendLine("        tensor<fp16, [1,\(D),1,\(S)]> \(lp)_r2_xn = mul(x=\(lp)_r2_xr,y=\(lp)_r2_rw)[name=string(\"\(lp)_r2_xn\")];")

        // ── SiLU-gated FFN ───────────────────────────────────────────────────
        b.appendLine("        tensor<fp16, [1,\(H),1,\(S)]> \(lp)_f_h1 = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=\(lp)_f_W1,x=\(lp)_r2_xn)[name=string(\"\(lp)_f_h1\")];")
        b.appendLine("        tensor<fp16, [1,\(H),1,\(S)]> \(lp)_f_h3 = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=\(lp)_f_W3,x=\(lp)_r2_xn)[name=string(\"\(lp)_f_h3\")];")
        b.appendLine("        tensor<fp16, [1,\(H),1,\(S)]> \(lp)_f_sig = sigmoid(x=\(lp)_f_h1)[name=string(\"\(lp)_f_sig\")];")
        b.appendLine("        tensor<fp16, [1,\(H),1,\(S)]> \(lp)_f_silu = mul(x=\(lp)_f_h1,y=\(lp)_f_sig)[name=string(\"\(lp)_f_silu\")];")
        b.appendLine("        tensor<fp16, [1,\(H),1,\(S)]> \(lp)_f_gate = mul(x=\(lp)_f_silu,y=\(lp)_f_h3)[name=string(\"\(lp)_f_gate\")];")
        b.appendLine("        tensor<fp16, [1,\(D),1,\(S)]> \(lp)_f_y = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=\(lp)_f_W2,x=\(lp)_f_gate)[name=string(\"\(lp)_f_y\")];")

        // ── Residual 2 → layer output ─────────────────────────────────────────
        b.appendLine("        tensor<fp16, [1,\(D),1,\(S)]> \(outputVar) = add(x=\(lp)_a_res,y=\(lp)_f_y)[name=string(\"\(lp)_f_res\")];")
    }
}
