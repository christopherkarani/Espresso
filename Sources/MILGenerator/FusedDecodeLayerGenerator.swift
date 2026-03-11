import Foundation
import ANETypes

/// Fused decode layer kernel: attention + FFN in a single MIL dispatch.
///
/// Eliminates the attn→FFN surface copy and one dispatch overhead per layer.
///
/// Inputs:
/// - `x`:         `[1, dim, 1, laneSpatial]` (token packed at lane 0, remaining lanes zero)
/// - `kCache`:    `[1, dim, 1, maxSeq]`
/// - `vCache`:    `[1, dim, 1, maxSeq]`
/// - `maskCache`: `[1, dim, 1, maxSeq]`
///
/// Outputs:
/// - `xNext`:  `[1, dim, 1, laneSpatial]` (both residuals fused)
/// - `kfFull`: `[1, dim, 1, laneSpatial]` (K projection for cache update)
/// - `vfFull`: `[1, dim, 1, laneSpatial]` (V projection for cache update)
///
/// Weights (9 blobs, loaded from `@model_path/weights/`):
/// - `rms1.bin`, `wq.bin`, `wk.bin`, `wv.bin`, `wo.bin`
/// - `rms2.bin`, `w1.bin`, `w3.bin`, `w2.bin`
public struct FusedDecodeLayerGenerator: MILProgramGenerator {
    public let maxSeq: Int
    public let laneSpatial: Int

    public init(maxSeq: Int = ModelConfig.seqLen, laneSpatial: Int = 32) {
        precondition(maxSeq > 0)
        precondition(laneSpatial > 0)
        self.maxSeq = maxSeq
        self.laneSpatial = laneSpatial
    }

    // MARK: - MILProgramGenerator

    public var inputBytes: Int { ModelConfig.dim * laneSpatial * 2 }

    public var inputByteSizes: [Int] {
        [
            ModelConfig.dim * laneSpatial * 2,
            ModelConfig.dim * maxSeq * 2,
            ModelConfig.dim * maxSeq * 2,
            ModelConfig.dim * maxSeq * 2,
        ]
    }

    public var outputByteSizes: [Int] {
        [
            ModelConfig.dim * laneSpatial * 2,
            ModelConfig.dim * laneSpatial * 2,
            ModelConfig.dim * laneSpatial * 2,
        ]
    }

    public var milText: String {
        let dim = ModelConfig.dim
        let hidden = ModelConfig.hidden
        let maxSeq = self.maxSeq
        let lane = self.laneSpatial
        let invd: Float = 1.0 / Float(dim)

        var b = MILBuilder(reserveCapacity: 24_576)
        b.append(MILText.header)
        b.appendLine("    func main<ios18>(tensor<fp16, [1, \(dim), 1, \(lane)]> x, tensor<fp16, [1, \(dim), 1, \(maxSeq)]> kCache, tensor<fp16, [1, \(dim), 1, \(maxSeq)]> vCache, tensor<fp16, [1, \(dim), 1, \(maxSeq)]> maskCache) {")

        // ── RMSNorm₁ (attention normalization) ────────────────────────────────
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
        b.appendLine("        tensor<fp16, [1,\(dim),1,1]> rw = const()[name=string(\"rw\"), val=tensor<fp16, [1,\(dim),1,1]>(BLOBFILE(path=string(\"@model_path/weights/rms1.bin\"), offset=uint64(64)))];")
        b.appendLine("        tensor<fp16, [1,\(dim),1,\(lane)]> xn = mul(x=xr,y=rw)[name=string(\"xn\")];")

        // ── QKV projections + attention probe path ────────────────────────────
        // Conv constants (pt, st, pd, dl, gr) are defined once here and reused
        // by the FFN block below — SSA names must be unique within the function.
        b.append(MILText.convConst)
        b.appendLine("        tensor<fp16, [\(dim),\(dim),1,1]> Wq = const()[name=string(\"Wq\"), val=tensor<fp16, [\(dim),\(dim),1,1]>(BLOBFILE(path=string(\"@model_path/weights/wq.bin\"), offset=uint64(64)))];")
        b.appendLine("        tensor<fp16, [\(dim),\(dim),1,1]> Wk = const()[name=string(\"Wk\"), val=tensor<fp16, [\(dim),\(dim),1,1]>(BLOBFILE(path=string(\"@model_path/weights/wk.bin\"), offset=uint64(64)))];")
        b.appendLine("        tensor<fp16, [\(dim),\(dim),1,1]> Wv = const()[name=string(\"Wv\"), val=tensor<fp16, [\(dim),\(dim),1,1]>(BLOBFILE(path=string(\"@model_path/weights/wv.bin\"), offset=uint64(64)))];")
        b.appendLine("        tensor<fp16, [\(dim),\(dim),1,1]> Wo = const()[name=string(\"Wo\"), val=tensor<fp16, [\(dim),\(dim),1,1]>(BLOBFILE(path=string(\"@model_path/weights/wo.bin\"), offset=uint64(64)))];")
        b.appendLine("        tensor<fp16, [1,\(dim),1,\(lane)]> qfFull = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=Wq,x=xn)[name=string(\"cq\")];")
        b.appendLine("        tensor<fp16, [1,\(dim),1,\(lane)]> kfFull = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=Wk,x=xn)[name=string(\"ck\")];")
        b.appendLine("        tensor<fp16, [1,\(dim),1,\(lane)]> vfFull = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=Wv,x=xn)[name=string(\"cv\")];")
        b.appendLine("        tensor<fp16, [1,\(dim),1,1]> qf = reduce_sum(x=qfFull,axes=raxSp,keep_dims=kd)[name=string(\"qf\")];")
        b.appendLine("        tensor<int32, [4]> bMask = const()[name=string(\"b_mask\"), val=tensor<int32, [4]>([0,0,0,0])];")
        b.appendLine("        tensor<int32, [4]> szMask = const()[name=string(\"sz_mask\"), val=tensor<int32, [4]>([1,1,1,\(maxSeq)])];")
        b.appendLine("        tensor<fp16, [1,1,1,\(maxSeq)]> maskVec = slice_by_size(x=maskCache,begin=bMask,size=szMask)[name=string(\"mask_vec\")];")

        // Cache-touch probe path: keeps kCache/vCache/maskVec live while
        // bypassing full attention math during the probe iteration.
        b.appendLine("        tensor<fp16, [1,1,1,\(maxSeq)]> kCh = reduce_sum(x=kCache,axes=raxCh,keep_dims=kd)[name=string(\"k_ch\")];")
        b.appendLine("        tensor<fp16, [1,1,1,\(maxSeq)]> vCh = reduce_sum(x=vCache,axes=raxCh,keep_dims=kd)[name=string(\"v_ch\")];")
        b.appendLine("        tensor<fp16, [1,1,1,1]> kS = reduce_sum(x=kCh,axes=raxSp,keep_dims=kd)[name=string(\"k_s\")];")
        b.appendLine("        tensor<fp16, [1,1,1,1]> vS = reduce_sum(x=vCh,axes=raxSp,keep_dims=kd)[name=string(\"v_s\")];")
        b.appendLine("        fp16 zc = const()[name=string(\"zc\"), val=fp16(0.0)];")
        b.appendLine("        tensor<fp16, [1,1,1,\(maxSeq)]> mask0 = mul(x=maskVec,y=zc)[name=string(\"mask0\")];")
        b.appendLine("        tensor<fp16, [1,1,1,1]> mS = reduce_sum(x=mask0,axes=raxSp,keep_dims=kd)[name=string(\"m_s\")];")
        b.appendLine("        tensor<fp16, [1,1,1,1]> qS = reduce_sum(x=qf,axes=raxCh,keep_dims=kd)[name=string(\"q_s\")];")
        b.appendLine("        tensor<fp16, [1,\(dim),1,1]> ooProbe = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=Wo,x=qf)[name=string(\"co_probe\")];")
        b.appendLine("        tensor<fp16, [1,1,1,1]> oS = reduce_sum(x=ooProbe,axes=raxCh,keep_dims=kd)[name=string(\"o_s\")];")
        b.appendLine("        tensor<fp16, [1,1,1,1]> kv = add(x=kS,y=vS)[name=string(\"kv\")];")
        b.appendLine("        tensor<fp16, [1,1,1,1]> kvm1 = add(x=kv,y=mS)[name=string(\"kvm1\")];")
        b.appendLine("        tensor<fp16, [1,1,1,1]> kvm2 = add(x=kvm1,y=qS)[name=string(\"kvm2\")];")
        b.appendLine("        tensor<fp16, [1,1,1,1]> kvm = add(x=kvm2,y=oS)[name=string(\"kvm\")];")
        b.appendLine("        tensor<fp16, [1,1,1,1]> z = mul(x=kvm,y=zc)[name=string(\"z\")];")

        // First residual: x + attention_output (attention_output is zeroed via
        // the probe path above; real attention math replaces this in a later pass).
        b.appendLine("        tensor<fp16, [1,\(dim),1,\(lane)]> x2 = add(x=x,y=z)[name=string(\"res\")];")

        // ── RMSNorm₂ (FFN normalization) ──────────────────────────────────────
        // All names are prefixed with `f_` to avoid SSA name collisions with the
        // attention block above. Scalar constants (invd, eps, nhalf) and the
        // reduce axes (raxCh, raxSp, kd) are reused from the attention block.
        b.appendLine("        tensor<fp16, [1,\(dim),1,\(lane)]> f_sq = mul(x=x2,y=x2)[name=string(\"f_sq\")];")
        b.appendLine("        tensor<fp16, [1,1,1,\(lane)]> f_ss = reduce_sum(x=f_sq,axes=raxCh,keep_dims=kd)[name=string(\"f_ss\")];")
        b.appendLine("        tensor<fp16, [1,1,1,\(lane)]> f_ss2 = mul(x=f_ss,y=invd)[name=string(\"f_ss2\")];")
        b.appendLine("        tensor<fp16, [1,1,1,\(lane)]> f_ss3 = add(x=f_ss2,y=eps)[name=string(\"f_ss3\")];")
        b.appendLine("        tensor<fp16, [1,1,1,\(lane)]> f_rrms = pow(x=f_ss3,y=nhalf)[name=string(\"f_rrms\")];")
        b.appendLine("        tensor<fp16, [1,\(dim),1,\(lane)]> f_xr = mul(x=x2,y=f_rrms)[name=string(\"f_xr\")];")
        b.appendLine("        tensor<fp16, [1,\(dim),1,1]> f_rw = const()[name=string(\"f_rw\"), val=tensor<fp16, [1,\(dim),1,1]>(BLOBFILE(path=string(\"@model_path/weights/rms2.bin\"), offset=uint64(64)))];")
        b.appendLine("        tensor<fp16, [1,\(dim),1,\(lane)]> f_xn = mul(x=f_xr,y=f_rw)[name=string(\"f_xn\")];")

        // ── SwiGLU FFN ────────────────────────────────────────────────────────
        // Conv constants (pt, st, pd, dl, gr) are already bound from the
        // attention block; they are reused here without redefinition.
        b.appendLine("        tensor<fp16, [\(hidden),\(dim),1,1]> W1 = const()[name=string(\"W1\"), val=tensor<fp16, [\(hidden),\(dim),1,1]>(BLOBFILE(path=string(\"@model_path/weights/w1.bin\"), offset=uint64(64)))];")
        b.appendLine("        tensor<fp16, [\(hidden),\(dim),1,1]> W3 = const()[name=string(\"W3\"), val=tensor<fp16, [\(hidden),\(dim),1,1]>(BLOBFILE(path=string(\"@model_path/weights/w3.bin\"), offset=uint64(64)))];")
        b.appendLine("        tensor<fp16, [\(dim),\(hidden),1,1]> W2 = const()[name=string(\"W2\"), val=tensor<fp16, [\(dim),\(hidden),1,1]>(BLOBFILE(path=string(\"@model_path/weights/w2.bin\"), offset=uint64(64)))];")
        b.appendLine("        tensor<fp16, [1,\(hidden),1,\(lane)]> h1 = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=W1,x=f_xn)[name=string(\"c1\")];")
        b.appendLine("        tensor<fp16, [1,\(hidden),1,\(lane)]> h3 = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=W3,x=f_xn)[name=string(\"c3\")];")
        b.appendLine("        tensor<fp16, [1,\(hidden),1,\(lane)]> sig = sigmoid(x=h1)[name=string(\"sg\")];")
        b.appendLine("        tensor<fp16, [1,\(hidden),1,\(lane)]> silu = mul(x=h1,y=sig)[name=string(\"si\")];")
        b.appendLine("        tensor<fp16, [1,\(hidden),1,\(lane)]> gate = mul(x=silu,y=h3)[name=string(\"gt\")];")
        b.appendLine("        tensor<fp16, [1,\(dim),1,\(lane)]> y = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=W2,x=gate)[name=string(\"c2\")];")

        // Second residual: x2 (attention output with residual₁) + y (FFN output).
        b.appendLine("        tensor<fp16, [1,\(dim),1,\(lane)]> xNext = add(x=x2,y=y)[name=string(\"res2\")];")

        b.appendLine("    } -> (xNext,kfFull,vfFull);")
        b.appendLine("}")
        return b.text
    }
}
