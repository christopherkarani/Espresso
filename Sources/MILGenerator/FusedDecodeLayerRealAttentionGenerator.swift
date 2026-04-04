import Foundation
import ANETypes

/// Fused decode layer kernel with REAL attention (not probe mode).
///
/// This kernel computes the full transformer layer including attention:
/// RMSNorm → QKV → Attention(Q,K_cache,V_cache) → Wo → residual → RMSNorm → SwiGLU FFN → residual
///
/// For stories110m: dim=768, heads=12, headDim=64, hidden=2048
public struct FusedDecodeLayerRealAttentionGenerator: MILProgramGenerator {
    public let maxSeq: Int
    public let laneSpatial: Int
    public let nHeads: Int
    public let headDim: Int

    public init(maxSeq: Int = ModelConfig.seqLen, laneSpatial: Int = 32, nHeads: Int = 12, headDim: Int = 64) {
        precondition(maxSeq > 0)
        precondition(laneSpatial > 0)
        self.maxSeq = maxSeq
        self.laneSpatial = laneSpatial
        self.nHeads = nHeads
        self.headDim = headDim
    }

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
        let heads = self.nHeads
        let headDim = self.headDim
        let invd: Float = 1.0 / Float(dim)
        let attnScale: Float = 1.0 / Float(headDim).squareRoot()

        var b = MILBuilder(reserveCapacity: 40_960)
        b.append(MILText.header)
        b.appendLine(MILText.functionLine(deploymentTarget: MILText.currentDeploymentTarget(), parameters: "tensor<fp16, [1, \(dim), 1, \(lane)]> x, tensor<fp16, [1, \(dim), 1, \(maxSeq)]> kCache, tensor<fp16, [1, \(dim), 1, \(maxSeq)]> vCache, tensor<fp16, [1, \(dim), 1, \(maxSeq)]> maskCache"))

        // ── RMSNorm₁ ──────────────────────────────────────────────────────────
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

        // ── QKV projections ───────────────────────────────────────────────────
        b.append(MILText.convConst)
        b.appendLine("        tensor<fp16, [\(dim),\(dim),1,1]> Wq = const()[name=string(\"Wq\"), val=tensor<fp16, [\(dim),\(dim),1,1]>(BLOBFILE(path=string(\"@model_path/weights/wq.bin\"), offset=uint64(64)))];")
        b.appendLine("        tensor<fp16, [\(dim),\(dim),1,1]> Wk = const()[name=string(\"Wk\"), val=tensor<fp16, [\(dim),\(dim),1,1]>(BLOBFILE(path=string(\"@model_path/weights/wk.bin\"), offset=uint64(64)))];")
        b.appendLine("        tensor<fp16, [\(dim),\(dim),1,1]> Wv = const()[name=string(\"Wv\"), val=tensor<fp16, [\(dim),\(dim),1,1]>(BLOBFILE(path=string(\"@model_path/weights/wv.bin\"), offset=uint64(64)))];")
        b.appendLine("        tensor<fp16, [1,\(dim),1,\(lane)]> qfFull = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=Wq,x=xn)[name=string(\"cq\")];")
        b.appendLine("        tensor<fp16, [1,\(dim),1,\(lane)]> kfFull = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=Wk,x=xn)[name=string(\"ck\")];")
        b.appendLine("        tensor<fp16, [1,\(dim),1,\(lane)]> vfFull = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=Wv,x=xn)[name=string(\"cv\")];")
        b.appendLine("        tensor<fp16, [1,\(dim),1,1]> qf = reduce_sum(x=qfFull,axes=raxSp,keep_dims=kd)[name=string(\"qf\")];")

        // ── REAL Attention: Q × K^T → softmax → × V ──────────────────────────
        // Reshape Q [1,dim,1,1] → [1,heads,headDim,1] → [1,heads,1,headDim] for matmul
        b.appendLine("        tensor<int32, [4]> qReshapeShape = const()[name=string(\"q_reshape_shape\"), val=tensor<int32, [4]>([1,\(heads),\(headDim),1])];")
        b.appendLine("        tensor<fp16, [1,\(heads),\(headDim),1]> qHeads = reshape(x=qf,new_shape=qReshapeShape)[name=string(\"q_heads\")];")
        b.appendLine("        tensor<int32, [4]> qTransPerm = const()[name=string(\"q_trans_perm\"), val=tensor<int32, [4]>([0,1,3,2])];")
        b.appendLine("        tensor<fp16, [1,\(heads),1,\(headDim)]> qT = transpose(x=qHeads,perm=qTransPerm)[name=string(\"q_trans\")];")

        // Reshape K cache [1,dim,1,maxSeq] → [1,heads,headDim,maxSeq] → [1,heads,maxSeq,headDim] for matmul
        b.appendLine("        tensor<int32, [4]> kReshapeShape = const()[name=string(\"k_reshape_shape\"), val=tensor<int32, [4]>([1,\(heads),\(headDim),\(maxSeq)])];")
        b.appendLine("        tensor<fp16, [1,\(heads),\(headDim),\(maxSeq)]> kHeads = reshape(x=kCache,new_shape=kReshapeShape)[name=string(\"k_heads\")];")
        b.appendLine("        tensor<int32, [4]> kTransPerm = const()[name=string(\"k_trans_perm\"), val=tensor<int32, [4]>([0,1,3,2])];")
        b.appendLine("        tensor<fp16, [1,\(heads),\(maxSeq),\(headDim)]> kT = transpose(x=kHeads,perm=kTransPerm)[name=string(\"k_trans\")];")

        // Scores = Q × K^T: [1,heads,1,headDim] × [1,heads,headDim,maxSeq] → [1,heads,1,maxSeq]
        b.appendLine("        tensor<int32, [4]> scoresShape = const()[name=string(\"scores_shape\"), val=tensor<int32, [4]>([1,\(heads),1,\(maxSeq)])];")
        b.appendLine("        tensor<fp16, [1,\(heads),1,\(maxSeq)]> scores = matmul(x=qT,y=kT,transpose_x=false,transpose_y=false,out_shape=scoresShape)[name=string(\"scores\")];")

        // Scale scores by 1/sqrt(headDim)
        b.append("        fp16 attnScale = const()[name=string(\"attn_scale\"), val=fp16(")
        b.appendFP16(attnScale)
        b.appendLine(")];")
        b.appendLine("        tensor<fp16, [1,\(heads),1,\(maxSeq)]> scaledScores = mul(x=scores,y=attnScale)[name=string(\"scaled_scores\")];")

        // Add causal mask
        b.appendLine("        tensor<int32, [4]> bMask = const()[name=string(\"b_mask\"), val=tensor<int32, [4]>([0,0,0,0])];")
        b.appendLine("        tensor<int32, [4]> szMask = const()[name=string(\"sz_mask\"), val=tensor<int32, [4]>([1,1,1,\(maxSeq)])];")
        b.appendLine("        tensor<fp16, [1,1,1,\(maxSeq)]> maskVec = slice_by_size(x=maskCache,begin=bMask,size=szMask)[name=string(\"mask_vec\")];")
        b.appendLine("        tensor<fp16, [1,\(heads),1,\(maxSeq)]> maskBroadcast = reshape(x=maskVec,new_shape=tensor<int32, [4]>([1,\(heads),1,\(maxSeq)]))[name=string(\"mask_bc\")];")
        b.appendLine("        tensor<fp16, [1,\(heads),1,\(maxSeq)]> maskedScores = add(x=scaledScores,y=maskBroadcast)[name=string(\"masked_scores\")];")

        // Softmax over the maxSeq dimension (axis 3)
        b.appendLine("        tensor<int32, [1]> raxAxis3 = const()[name=string(\"rax_axis3\"), val=tensor<int32, [1]>([3])];")
        b.appendLine("        tensor<fp16, [1,\(heads),1,\(maxSeq)]> attnWeights = softmax(x=maskedScores,axis=raxAxis3)[name=string(\"attn_weights\")];")

        // Reshape V cache [1,dim,1,maxSeq] → [1,heads,headDim,maxSeq]
        // Context = attnWeights × V: [1,heads,1,maxSeq] × [1,heads,headDim,maxSeq]^T → [1,heads,headDim,1]
        b.appendLine("        tensor<fp16, [1,\(heads),\(headDim),\(maxSeq)]> vHeads = reshape(x=vCache,new_shape=kReshapeShape)[name=string(\"v_heads\")];")
        b.appendLine("        tensor<int32, [4]> ctxShape = const()[name=string(\"ctx_shape\"), val=tensor<int32, [4]>([1,\(heads),\(headDim),1])];")
        b.appendLine("        tensor<fp16, [1,\(heads),\(headDim),1]> context = matmul(x=attnWeights,y=vHeads,transpose_x=false,transpose_y=true,out_shape=ctxShape)[name=string(\"context\")];")

        // Reshape context back: [1,heads,headDim,1] → [1,heads,1,headDim] → [1,dim,1,1]
        b.appendLine("        tensor<fp16, [1,\(heads),1,\(headDim)]> contextT = transpose(x=context,perm=qTransPerm)[name=string(\"context_trans\")];")
        b.appendLine("        tensor<fp16, [1,\(dim),1,1]> attnOut = reshape(x=contextT,new_shape=tensor<int32, [4]>([1,\(dim),1,1]))[name=string(\"attn_out\")];")

        // ── Output projection (Wo) ────────────────────────────────────────────
        b.appendLine("        tensor<fp16, [\(dim),\(dim),1,1]> Wo = const()[name=string(\"Wo\"), val=tensor<fp16, [\(dim),\(dim),1,1]>(BLOBFILE(path=string(\"@model_path/weights/wo.bin\"), offset=uint64(64)))];")
        b.appendLine("        tensor<fp16, [1,\(dim),1,1]> woOut = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=Wo,x=attnOut)[name=string(\"co_wo\")];")

        // Broadcast woOut back to lane width and add residual
        b.appendLine("        tensor<fp16, [1,\(dim),1,\(lane)]> woBroadcast = add(x=woOut,y=const()[name=string(\"zc\"), val=tensor<fp16, [1,\(dim),1,\(lane)]>(0.0))[name=string(\"wo_bc\")];")
        b.appendLine("        tensor<fp16, [1,\(dim),1,\(lane)]> x2 = add(x=x,y=woBroadcast)[name=string(\"res1\")];")

        // ── RMSNorm₂ (FFN normalization) ──────────────────────────────────────
        b.appendLine("        tensor<fp16, [1,\(dim),1,\(lane)]> f_sq = mul(x=x2,y=x2)[name=string(\"f_sq\")];")
        b.appendLine("        tensor<fp16, [1,1,1,\(lane)]> f_ss = reduce_sum(x=f_sq,axes=raxCh,keep_dims=kd)[name=string(\"f_ss\")];")
        b.appendLine("        tensor<fp16, [1,1,1,\(lane)]> f_ss2 = mul(x=f_ss,y=invd)[name=string(\"f_ss2\")];")
        b.appendLine("        tensor<fp16, [1,1,1,\(lane)]> f_ss3 = add(x=f_ss2,y=eps)[name=string(\"f_ss3\")];")
        b.appendLine("        tensor<fp16, [1,1,1,\(lane)]> f_rrms = pow(x=f_ss3,y=nhalf)[name=string(\"f_rrms\")];")
        b.appendLine("        tensor<fp16, [1,\(dim),1,\(lane)]> f_xr = mul(x=x2,y=f_rrms)[name=string(\"f_xr\")];")
        b.appendLine("        tensor<fp16, [1,\(dim),1,1]> f_rw = const()[name=string(\"f_rw\"), val=tensor<fp16, [1,\(dim),1,1]>(BLOBFILE(path=string(\"@model_path/weights/rms2.bin\"), offset=uint64(64)))];")
        b.appendLine("        tensor<fp16, [1,\(dim),1,\(lane)]> f_xn = mul(x=f_xr,y=f_rw)[name=string(\"f_xn\")];")

        // ── SwiGLU FFN ────────────────────────────────────────────────────────
        b.appendLine("        tensor<fp16, [\(hidden),\(dim),1,1]> W1 = const()[name=string(\"W1\"), val=tensor<fp16, [\(hidden),\(dim),1,1]>(BLOBFILE(path=string(\"@model_path/weights/w1.bin\"), offset=uint64(64)))];")
        b.appendLine("        tensor<fp16, [\(hidden),\(dim),1,1]> W3 = const()[name=string(\"W3\"), val=tensor<fp16, [\(hidden),\(dim),1,1]>(BLOBFILE(path=string(\"@model_path/weights/w3.bin\"), offset=uint64(64)))];")
        b.appendLine("        tensor<fp16, [\(dim),\(hidden),1,1]> W2 = const()[name=string(\"W2\"), val=tensor<fp16, [\(dim),\(hidden),1,1]>(BLOBFILE(path=string(\"@model_path/weights/w2.bin\"), offset=uint64(64)))];")
        b.appendLine("        tensor<fp16, [1,\(hidden),1,\(lane)]> h1 = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=W1,x=f_xn)[name=string(\"c1\")];")
        b.appendLine("        tensor<fp16, [1,\(hidden),1,\(lane)]> h3 = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=W3,x=f_xn)[name=string(\"c3\")];")
        b.appendLine("        tensor<fp16, [1,\(hidden),1,\(lane)]> sig = sigmoid(x=h1)[name=string(\"sg\")];")
        b.appendLine("        tensor<fp16, [1,\(hidden),1,\(lane)]> silu = mul(x=h1,y=sig)[name=string(\"si\")];")
        b.appendLine("        tensor<fp16, [1,\(hidden),1,\(lane)]> gate = mul(x=silu,y=h3)[name=string(\"gt\")];")
        b.appendLine("        tensor<fp16, [1,\(dim),1,\(lane)]> y = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=W2,x=gate)[name=string(\"c2\")];")

        // Second residual
        b.appendLine("        tensor<fp16, [1,\(dim),1,\(lane)]> xNext = add(x=x2,y=y)[name=string(\"res2\")];")

        b.appendLine("    } -> (xNext,kfFull,vfFull);")
        b.appendLine("}")
        return b.text
    }
}
