import Foundation
import ANETypes

/// Fused two-layer decode kernel: two transformer layers in a single MIL dispatch.
///
/// This is the Avenue 1 fallback after the fully packed K/V/mask input produced
/// `InvalidMILProgram` on ANE hardware. K/V caches stay packed; masks stay separate.
///
/// Inputs:
/// - `x`:             `[1, dim, 1, laneSpatial]`
/// - `packedKVCache`: `[1, 4*dim, 1, maxSeq]`
///   - channels `0..<dim`: layer 0 K cache
///   - channels `dim..<(2*dim)`: layer 0 V cache
///   - channels `2*dim..<(3*dim)`: layer 1 K cache
///   - channels `3*dim..<(4*dim)`: layer 1 V cache
/// - `maskCache0`:    `[1, dim, 1, maxSeq]`
/// - `maskCache1`:    `[1, dim, 1, maxSeq]`
///
/// Outputs:
/// - `xNext`:   `[1, dim, 1, laneSpatial]`
/// - `kPacked`: `[1, 2*dim, 1, laneSpatial]`  (layer 0 K || layer 1 K)
/// - `vPacked`: `[1, 2*dim, 1, laneSpatial]`  (layer 0 V || layer 1 V)
public struct FusedTwoLayerDecodeGenerator: MILProgramGenerator {
    public let maxSeq: Int
    public let laneSpatial: Int

    public init(maxSeq: Int = ModelConfig.seqLen, laneSpatial: Int = 32) {
        precondition(maxSeq > 0)
        precondition(laneSpatial > 0)
        self.maxSeq = maxSeq
        self.laneSpatial = laneSpatial
    }

    public var inputBytes: Int { ModelConfig.dim * laneSpatial * 2 }

    public var inputByteSizes: [Int] {
        [
            ModelConfig.dim * laneSpatial * 2,
            4 * ModelConfig.dim * maxSeq * 2,
            ModelConfig.dim * maxSeq * 2,
            ModelConfig.dim * maxSeq * 2,
        ]
    }

    public var outputByteSizes: [Int] {
        [
            ModelConfig.dim * laneSpatial * 2,
            2 * ModelConfig.dim * laneSpatial * 2,
            2 * ModelConfig.dim * laneSpatial * 2,
        ]
    }

    public var milText: String {
        let dim = ModelConfig.dim
        let hidden = ModelConfig.hidden
        let maxSeq = self.maxSeq
        let lane = self.laneSpatial
        let packedKVChannels = 4 * dim
        let invd: Float = 1.0 / Float(dim)

        var b = MILBuilder(reserveCapacity: 49_152)
        b.append(MILText.header)
        b.appendLine("    func main<ios18>(tensor<fp16, [1, \(dim), 1, \(lane)]> x, tensor<fp16, [1, \(packedKVChannels), 1, \(maxSeq)]> packedKVCache, tensor<fp16, [1, \(dim), 1, \(maxSeq)]> maskCache0, tensor<fp16, [1, \(dim), 1, \(maxSeq)]> maskCache1) {")

        b.appendLine("        tensor<int32, [1]> raxCh = const()[name=string(\"rax_ch\"), val=tensor<int32, [1]>([1])];")
        b.appendLine("        tensor<int32, [1]> raxSp = const()[name=string(\"rax_sp\"), val=tensor<int32, [1]>([3])];")
        b.appendLine("        bool kd = const()[name=string(\"kd\"), val=bool(true)];")
        b.append("        fp16 invd = const()[name=string(\"invd\"), val=fp16(")
        b.appendFP16(invd)
        b.appendLine(")];")
        b.appendLine("        fp16 eps = const()[name=string(\"eps\"), val=fp16(0.00001)];")
        b.appendLine("        fp16 nhalf = const()[name=string(\"nhalf\"), val=fp16(-0.5)];")
        b.append(MILText.convConst)

        b.appendLine("        tensor<int32, [4]> kv_b0 = const()[name=string(\"kv_b0\"), val=tensor<int32, [4]>([0,0,0,0])];")
        b.appendLine("        tensor<int32, [4]> kv_b1 = const()[name=string(\"kv_b1\"), val=tensor<int32, [4]>([0,\(dim),0,0])];")
        b.appendLine("        tensor<int32, [4]> kv_b2 = const()[name=string(\"kv_b2\"), val=tensor<int32, [4]>([0,\(2 * dim),0,0])];")
        b.appendLine("        tensor<int32, [4]> kv_b3 = const()[name=string(\"kv_b3\"), val=tensor<int32, [4]>([0,\(3 * dim),0,0])];")
        b.appendLine("        tensor<int32, [4]> kv_sz = const()[name=string(\"kv_sz\"), val=tensor<int32, [4]>([1,\(dim),1,\(maxSeq)])];")
        b.appendLine("        tensor<fp16, [1,\(dim),1,\(maxSeq)]> kCache = slice_by_size(x=packedKVCache,begin=kv_b0,size=kv_sz)[name=string(\"kv_l0_k\")];")
        b.appendLine("        tensor<fp16, [1,\(dim),1,\(maxSeq)]> vCache = slice_by_size(x=packedKVCache,begin=kv_b1,size=kv_sz)[name=string(\"kv_l0_v\")];")
        b.appendLine("        tensor<fp16, [1,\(dim),1,\(maxSeq)]> l1_kCache = slice_by_size(x=packedKVCache,begin=kv_b2,size=kv_sz)[name=string(\"kv_l1_k\")];")
        b.appendLine("        tensor<fp16, [1,\(dim),1,\(maxSeq)]> l1_vCache = slice_by_size(x=packedKVCache,begin=kv_b3,size=kv_sz)[name=string(\"kv_l1_v\")];")

        b.appendLine("        tensor<fp16, [1,\(dim),1,\(lane)]> sq = mul(x=x,y=x)[name=string(\"sq\")];")
        b.appendLine("        tensor<fp16, [1,1,1,\(lane)]> ss = reduce_sum(x=sq,axes=raxCh,keep_dims=kd)[name=string(\"ss\")];")
        b.appendLine("        tensor<fp16, [1,1,1,\(lane)]> ss2 = mul(x=ss,y=invd)[name=string(\"ss2\")];")
        b.appendLine("        tensor<fp16, [1,1,1,\(lane)]> ss3 = add(x=ss2,y=eps)[name=string(\"ss3\")];")
        b.appendLine("        tensor<fp16, [1,1,1,\(lane)]> rrms = pow(x=ss3,y=nhalf)[name=string(\"rrms\")];")
        b.appendLine("        tensor<fp16, [1,\(dim),1,\(lane)]> xr = mul(x=x,y=rrms)[name=string(\"xr\")];")
        b.appendLine("        tensor<fp16, [1,\(dim),1,1]> rw = const()[name=string(\"rw\"), val=tensor<fp16, [1,\(dim),1,1]>(BLOBFILE(path=string(\"@model_path/weights/l0_rms1.bin\"), offset=uint64(64)))];")
        b.appendLine("        tensor<fp16, [1,\(dim),1,\(lane)]> xn = mul(x=xr,y=rw)[name=string(\"xn\")];")
        b.appendLine("        tensor<fp16, [\(dim),\(dim),1,1]> Wq = const()[name=string(\"Wq\"), val=tensor<fp16, [\(dim),\(dim),1,1]>(BLOBFILE(path=string(\"@model_path/weights/l0_wq.bin\"), offset=uint64(64)))];")
        b.appendLine("        tensor<fp16, [\(dim),\(dim),1,1]> Wk = const()[name=string(\"Wk\"), val=tensor<fp16, [\(dim),\(dim),1,1]>(BLOBFILE(path=string(\"@model_path/weights/l0_wk.bin\"), offset=uint64(64)))];")
        b.appendLine("        tensor<fp16, [\(dim),\(dim),1,1]> Wv = const()[name=string(\"Wv\"), val=tensor<fp16, [\(dim),\(dim),1,1]>(BLOBFILE(path=string(\"@model_path/weights/l0_wv.bin\"), offset=uint64(64)))];")
        b.appendLine("        tensor<fp16, [\(dim),\(dim),1,1]> Wo = const()[name=string(\"Wo\"), val=tensor<fp16, [\(dim),\(dim),1,1]>(BLOBFILE(path=string(\"@model_path/weights/l0_wo.bin\"), offset=uint64(64)))];")
        b.appendLine("        tensor<fp16, [1,\(dim),1,\(lane)]> qfFull = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=Wq,x=xn)[name=string(\"cq\")];")
        b.appendLine("        tensor<fp16, [1,\(dim),1,\(lane)]> kfL0 = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=Wk,x=xn)[name=string(\"ck\")];")
        b.appendLine("        tensor<fp16, [1,\(dim),1,\(lane)]> vfL0 = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=Wv,x=xn)[name=string(\"cv\")];")
        b.appendLine("        tensor<fp16, [1,\(dim),1,1]> qf = reduce_sum(x=qfFull,axes=raxSp,keep_dims=kd)[name=string(\"qf\")];")
        b.appendLine("        tensor<int32, [4]> bMask = const()[name=string(\"b_mask\"), val=tensor<int32, [4]>([0,0,0,0])];")
        b.appendLine("        tensor<int32, [4]> szMask = const()[name=string(\"sz_mask\"), val=tensor<int32, [4]>([1,1,1,\(maxSeq)])];")
        b.appendLine("        tensor<fp16, [1,1,1,\(maxSeq)]> maskVec = slice_by_size(x=maskCache0,begin=bMask,size=szMask)[name=string(\"mask_vec\")];")
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
        b.appendLine("        tensor<fp16, [1,\(dim),1,\(lane)]> x2 = add(x=x,y=z)[name=string(\"res\")];")
        b.appendLine("        tensor<fp16, [1,\(dim),1,\(lane)]> f_sq = mul(x=x2,y=x2)[name=string(\"f_sq\")];")
        b.appendLine("        tensor<fp16, [1,1,1,\(lane)]> f_ss = reduce_sum(x=f_sq,axes=raxCh,keep_dims=kd)[name=string(\"f_ss\")];")
        b.appendLine("        tensor<fp16, [1,1,1,\(lane)]> f_ss2 = mul(x=f_ss,y=invd)[name=string(\"f_ss2\")];")
        b.appendLine("        tensor<fp16, [1,1,1,\(lane)]> f_ss3 = add(x=f_ss2,y=eps)[name=string(\"f_ss3\")];")
        b.appendLine("        tensor<fp16, [1,1,1,\(lane)]> f_rrms = pow(x=f_ss3,y=nhalf)[name=string(\"f_rrms\")];")
        b.appendLine("        tensor<fp16, [1,\(dim),1,\(lane)]> f_xr = mul(x=x2,y=f_rrms)[name=string(\"f_xr\")];")
        b.appendLine("        tensor<fp16, [1,\(dim),1,1]> f_rw = const()[name=string(\"f_rw\"), val=tensor<fp16, [1,\(dim),1,1]>(BLOBFILE(path=string(\"@model_path/weights/l0_rms2.bin\"), offset=uint64(64)))];")
        b.appendLine("        tensor<fp16, [1,\(dim),1,\(lane)]> f_xn = mul(x=f_xr,y=f_rw)[name=string(\"f_xn\")];")
        b.appendLine("        tensor<fp16, [\(hidden),\(dim),1,1]> W1 = const()[name=string(\"W1\"), val=tensor<fp16, [\(hidden),\(dim),1,1]>(BLOBFILE(path=string(\"@model_path/weights/l0_w1.bin\"), offset=uint64(64)))];")
        b.appendLine("        tensor<fp16, [\(hidden),\(dim),1,1]> W3 = const()[name=string(\"W3\"), val=tensor<fp16, [\(hidden),\(dim),1,1]>(BLOBFILE(path=string(\"@model_path/weights/l0_w3.bin\"), offset=uint64(64)))];")
        b.appendLine("        tensor<fp16, [\(dim),\(hidden),1,1]> W2 = const()[name=string(\"W2\"), val=tensor<fp16, [\(dim),\(hidden),1,1]>(BLOBFILE(path=string(\"@model_path/weights/l0_w2.bin\"), offset=uint64(64)))];")
        b.appendLine("        tensor<fp16, [1,\(hidden),1,\(lane)]> h1 = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=W1,x=f_xn)[name=string(\"c1\")];")
        b.appendLine("        tensor<fp16, [1,\(hidden),1,\(lane)]> h3 = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=W3,x=f_xn)[name=string(\"c3\")];")
        b.appendLine("        tensor<fp16, [1,\(hidden),1,\(lane)]> sig = sigmoid(x=h1)[name=string(\"sg\")];")
        b.appendLine("        tensor<fp16, [1,\(hidden),1,\(lane)]> silu = mul(x=h1,y=sig)[name=string(\"si\")];")
        b.appendLine("        tensor<fp16, [1,\(hidden),1,\(lane)]> gate = mul(x=silu,y=h3)[name=string(\"gt\")];")
        b.appendLine("        tensor<fp16, [1,\(dim),1,\(lane)]> y = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=W2,x=gate)[name=string(\"c2\")];")
        b.appendLine("        tensor<fp16, [1,\(dim),1,\(lane)]> xMid = add(x=x2,y=y)[name=string(\"res2\")];")

        b.appendLine("        tensor<fp16, [1,\(dim),1,\(lane)]> l1_sq = mul(x=xMid,y=xMid)[name=string(\"l1_sq\")];")
        b.appendLine("        tensor<fp16, [1,1,1,\(lane)]> l1_ss = reduce_sum(x=l1_sq,axes=raxCh,keep_dims=kd)[name=string(\"l1_ss\")];")
        b.appendLine("        tensor<fp16, [1,1,1,\(lane)]> l1_ss2 = mul(x=l1_ss,y=invd)[name=string(\"l1_ss2\")];")
        b.appendLine("        tensor<fp16, [1,1,1,\(lane)]> l1_ss3 = add(x=l1_ss2,y=eps)[name=string(\"l1_ss3\")];")
        b.appendLine("        tensor<fp16, [1,1,1,\(lane)]> l1_rrms = pow(x=l1_ss3,y=nhalf)[name=string(\"l1_rrms\")];")
        b.appendLine("        tensor<fp16, [1,\(dim),1,\(lane)]> l1_xr = mul(x=xMid,y=l1_rrms)[name=string(\"l1_xr\")];")
        b.appendLine("        tensor<fp16, [1,\(dim),1,1]> l1_rw = const()[name=string(\"l1_rw\"), val=tensor<fp16, [1,\(dim),1,1]>(BLOBFILE(path=string(\"@model_path/weights/l1_rms1.bin\"), offset=uint64(64)))];")
        b.appendLine("        tensor<fp16, [1,\(dim),1,\(lane)]> l1_xn = mul(x=l1_xr,y=l1_rw)[name=string(\"l1_xn\")];")
        b.appendLine("        tensor<fp16, [\(dim),\(dim),1,1]> l1_Wq = const()[name=string(\"l1_Wq\"), val=tensor<fp16, [\(dim),\(dim),1,1]>(BLOBFILE(path=string(\"@model_path/weights/l1_wq.bin\"), offset=uint64(64)))];")
        b.appendLine("        tensor<fp16, [\(dim),\(dim),1,1]> l1_Wk = const()[name=string(\"l1_Wk\"), val=tensor<fp16, [\(dim),\(dim),1,1]>(BLOBFILE(path=string(\"@model_path/weights/l1_wk.bin\"), offset=uint64(64)))];")
        b.appendLine("        tensor<fp16, [\(dim),\(dim),1,1]> l1_Wv = const()[name=string(\"l1_Wv\"), val=tensor<fp16, [\(dim),\(dim),1,1]>(BLOBFILE(path=string(\"@model_path/weights/l1_wv.bin\"), offset=uint64(64)))];")
        b.appendLine("        tensor<fp16, [\(dim),\(dim),1,1]> l1_Wo = const()[name=string(\"l1_Wo\"), val=tensor<fp16, [\(dim),\(dim),1,1]>(BLOBFILE(path=string(\"@model_path/weights/l1_wo.bin\"), offset=uint64(64)))];")
        b.appendLine("        tensor<fp16, [1,\(dim),1,\(lane)]> l1_qfFull = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=l1_Wq,x=l1_xn)[name=string(\"l1_cq\")];")
        b.appendLine("        tensor<fp16, [1,\(dim),1,\(lane)]> l1_kf = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=l1_Wk,x=l1_xn)[name=string(\"l1_ck\")];")
        b.appendLine("        tensor<fp16, [1,\(dim),1,\(lane)]> l1_vf = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=l1_Wv,x=l1_xn)[name=string(\"l1_cv\")];")
        b.appendLine("        tensor<fp16, [1,\(dim),1,1]> l1_qf = reduce_sum(x=l1_qfFull,axes=raxSp,keep_dims=kd)[name=string(\"l1_qf\")];")
        b.appendLine("        tensor<int32, [4]> l1_bMask = const()[name=string(\"l1_b_mask\"), val=tensor<int32, [4]>([0,0,0,0])];")
        b.appendLine("        tensor<int32, [4]> l1_szMask = const()[name=string(\"l1_sz_mask\"), val=tensor<int32, [4]>([1,1,1,\(maxSeq)])];")
        b.appendLine("        tensor<fp16, [1,1,1,\(maxSeq)]> l1_maskVec = slice_by_size(x=maskCache1,begin=l1_bMask,size=l1_szMask)[name=string(\"l1_mask_vec\")];")
        b.appendLine("        tensor<fp16, [1,1,1,\(maxSeq)]> l1_kCh = reduce_sum(x=l1_kCache,axes=raxCh,keep_dims=kd)[name=string(\"l1_k_ch\")];")
        b.appendLine("        tensor<fp16, [1,1,1,\(maxSeq)]> l1_vCh = reduce_sum(x=l1_vCache,axes=raxCh,keep_dims=kd)[name=string(\"l1_v_ch\")];")
        b.appendLine("        tensor<fp16, [1,1,1,1]> l1_kS = reduce_sum(x=l1_kCh,axes=raxSp,keep_dims=kd)[name=string(\"l1_k_s\")];")
        b.appendLine("        tensor<fp16, [1,1,1,1]> l1_vS = reduce_sum(x=l1_vCh,axes=raxSp,keep_dims=kd)[name=string(\"l1_v_s\")];")
        b.appendLine("        tensor<fp16, [1,1,1,\(maxSeq)]> l1_mask0 = mul(x=l1_maskVec,y=zc)[name=string(\"l1_mask0\")];")
        b.appendLine("        tensor<fp16, [1,1,1,1]> l1_mS = reduce_sum(x=l1_mask0,axes=raxSp,keep_dims=kd)[name=string(\"l1_m_s\")];")
        b.appendLine("        tensor<fp16, [1,1,1,1]> l1_qS = reduce_sum(x=l1_qf,axes=raxCh,keep_dims=kd)[name=string(\"l1_q_s\")];")
        b.appendLine("        tensor<fp16, [1,\(dim),1,1]> l1_ooProbe = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=l1_Wo,x=l1_qf)[name=string(\"l1_co_probe\")];")
        b.appendLine("        tensor<fp16, [1,1,1,1]> l1_oS = reduce_sum(x=l1_ooProbe,axes=raxCh,keep_dims=kd)[name=string(\"l1_o_s\")];")
        b.appendLine("        tensor<fp16, [1,1,1,1]> l1_kv = add(x=l1_kS,y=l1_vS)[name=string(\"l1_kv\")];")
        b.appendLine("        tensor<fp16, [1,1,1,1]> l1_kvm1 = add(x=l1_kv,y=l1_mS)[name=string(\"l1_kvm1\")];")
        b.appendLine("        tensor<fp16, [1,1,1,1]> l1_kvm2 = add(x=l1_kvm1,y=l1_qS)[name=string(\"l1_kvm2\")];")
        b.appendLine("        tensor<fp16, [1,1,1,1]> l1_kvm = add(x=l1_kvm2,y=l1_oS)[name=string(\"l1_kvm\")];")
        b.appendLine("        tensor<fp16, [1,1,1,1]> l1_z = mul(x=l1_kvm,y=zc)[name=string(\"l1_z\")];")
        b.appendLine("        tensor<fp16, [1,\(dim),1,\(lane)]> l1_x2 = add(x=xMid,y=l1_z)[name=string(\"l1_res\")];")
        b.appendLine("        tensor<fp16, [1,\(dim),1,\(lane)]> l1_f_sq = mul(x=l1_x2,y=l1_x2)[name=string(\"l1_f_sq\")];")
        b.appendLine("        tensor<fp16, [1,1,1,\(lane)]> l1_f_ss = reduce_sum(x=l1_f_sq,axes=raxCh,keep_dims=kd)[name=string(\"l1_f_ss\")];")
        b.appendLine("        tensor<fp16, [1,1,1,\(lane)]> l1_f_ss2 = mul(x=l1_f_ss,y=invd)[name=string(\"l1_f_ss2\")];")
        b.appendLine("        tensor<fp16, [1,1,1,\(lane)]> l1_f_ss3 = add(x=l1_f_ss2,y=eps)[name=string(\"l1_f_ss3\")];")
        b.appendLine("        tensor<fp16, [1,1,1,\(lane)]> l1_f_rrms = pow(x=l1_f_ss3,y=nhalf)[name=string(\"l1_f_rrms\")];")
        b.appendLine("        tensor<fp16, [1,\(dim),1,\(lane)]> l1_f_xr = mul(x=l1_x2,y=l1_f_rrms)[name=string(\"l1_f_xr\")];")
        b.appendLine("        tensor<fp16, [1,\(dim),1,1]> l1_f_rw = const()[name=string(\"l1_f_rw\"), val=tensor<fp16, [1,\(dim),1,1]>(BLOBFILE(path=string(\"@model_path/weights/l1_rms2.bin\"), offset=uint64(64)))];")
        b.appendLine("        tensor<fp16, [1,\(dim),1,\(lane)]> l1_f_xn = mul(x=l1_f_xr,y=l1_f_rw)[name=string(\"l1_f_xn\")];")
        b.appendLine("        tensor<fp16, [\(hidden),\(dim),1,1]> l1_W1 = const()[name=string(\"l1_W1\"), val=tensor<fp16, [\(hidden),\(dim),1,1]>(BLOBFILE(path=string(\"@model_path/weights/l1_w1.bin\"), offset=uint64(64)))];")
        b.appendLine("        tensor<fp16, [\(hidden),\(dim),1,1]> l1_W3 = const()[name=string(\"l1_W3\"), val=tensor<fp16, [\(hidden),\(dim),1,1]>(BLOBFILE(path=string(\"@model_path/weights/l1_w3.bin\"), offset=uint64(64)))];")
        b.appendLine("        tensor<fp16, [\(dim),\(hidden),1,1]> l1_W2 = const()[name=string(\"l1_W2\"), val=tensor<fp16, [\(dim),\(hidden),1,1]>(BLOBFILE(path=string(\"@model_path/weights/l1_w2.bin\"), offset=uint64(64)))];")
        b.appendLine("        tensor<fp16, [1,\(hidden),1,\(lane)]> l1_h1 = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=l1_W1,x=l1_f_xn)[name=string(\"l1_c1\")];")
        b.appendLine("        tensor<fp16, [1,\(hidden),1,\(lane)]> l1_h3 = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=l1_W3,x=l1_f_xn)[name=string(\"l1_c3\")];")
        b.appendLine("        tensor<fp16, [1,\(hidden),1,\(lane)]> l1_sig = sigmoid(x=l1_h1)[name=string(\"l1_sg\")];")
        b.appendLine("        tensor<fp16, [1,\(hidden),1,\(lane)]> l1_silu = mul(x=l1_h1,y=l1_sig)[name=string(\"l1_si\")];")
        b.appendLine("        tensor<fp16, [1,\(hidden),1,\(lane)]> l1_gate = mul(x=l1_silu,y=l1_h3)[name=string(\"l1_gt\")];")
        b.appendLine("        tensor<fp16, [1,\(dim),1,\(lane)]> l1_y = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=l1_W2,x=l1_gate)[name=string(\"l1_c2\")];")
        b.appendLine("        tensor<fp16, [1,\(dim),1,\(lane)]> xNext = add(x=l1_x2,y=l1_y)[name=string(\"l1_res2\")];")

        b.appendLine("        int32 cax = const()[name=string(\"cax\"), val=int32(1)];")
        b.appendLine("        bool cid = const()[name=string(\"cid\"), val=bool(false)];")
        b.appendLine("        tensor<fp16, [1,\(2 * dim),1,\(lane)]> kPacked = concat(axis=cax,interleave=cid,values=(kfL0,l1_kf))[name=string(\"k_cat\")];")
        b.appendLine("        tensor<fp16, [1,\(2 * dim),1,\(lane)]> vPacked = concat(axis=cax,interleave=cid,values=(vfL0,l1_vf))[name=string(\"v_cat\")];")
        b.appendLine("    } -> (xNext,kPacked,vPacked);")
        b.appendLine("}")
        return b.text
    }
}
