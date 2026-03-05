import Foundation
import ANETypes

/// Decode-time attention kernel with in-kernel self-attention score and KV emission.
///
/// Inputs:
/// - `x`:        `[1, dim, 1, laneSpatial]` (token packed at lane 0, remaining lanes zero)
/// - `kCache`:   `[1, dim, 1, maxSeq]`
/// - `vCache`:   `[1, dim, 1, maxSeq]`
/// - `maskCache`: `[1, dim, 1, maxSeq]` (per-position mask expanded across channels)
///
/// Outputs:
/// - `x2`: `[1, dim, 1, laneSpatial]`
/// - `k`:  `[1, dim, 1, laneSpatial]`
/// - `v`:  `[1, dim, 1, laneSpatial]`
public struct DecodeAttentionQKVGenerator: MILProgramGenerator {
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
        let maxSeq = self.maxSeq
        let lane = self.laneSpatial
        let invd: Float = 1.0 / Float(dim)
        let probeMode = ProcessInfo.processInfo.environment["ESPRESSO_DECODE_ATTN_PROBE_MODE"] ?? "cache-touch"

        var b = MILBuilder(reserveCapacity: 16_384)
        b.append(MILText.header)
        b.appendLine("    func main<ios18>(tensor<fp16, [1, \(dim), 1, \(lane)]> x, tensor<fp16, [1, \(dim), 1, \(maxSeq)]> kCache, tensor<fp16, [1, \(dim), 1, \(maxSeq)]> vCache, tensor<fp16, [1, \(dim), 1, \(maxSeq)]> maskCache) {")

        if probeMode == "passthrough" {
            b.appendLine("        tensor<fp16, [1,\(dim),1,\(lane)]> outX = add(x=x,y=x)[name=string(\"out_x\")];")
            b.appendLine("        tensor<fp16, [1,\(dim),1,\(lane)]> outK = add(x=x,y=x)[name=string(\"out_k\")];")
            b.appendLine("        tensor<fp16, [1,\(dim),1,\(lane)]> outV = add(x=x,y=x)[name=string(\"out_v\")];")
            b.appendLine("    } -> (outX,outK,outV);")
            b.appendLine("}")
            return b.text
        }
        if probeMode == "passthrough-touch" {
            b.appendLine("        fp16 zc = const()[name=string(\"zc\"), val=fp16(0.0)];")
            b.appendLine("        tensor<int32, [4]> b0 = const()[name=string(\"b0\"), val=tensor<int32, [4]>([0,0,0,0])];")
            b.appendLine("        tensor<int32, [4]> szK = const()[name=string(\"sz_k\"), val=tensor<int32, [4]>([1,\(dim),1,\(lane)])];")
            b.appendLine("        tensor<int32, [4]> szM = const()[name=string(\"sz_m\"), val=tensor<int32, [4]>([1,1,1,\(lane)])];")
            b.appendLine("        tensor<fp16, [1,\(dim),1,\(lane)]> kLane = slice_by_size(x=kCache,begin=b0,size=szK)[name=string(\"k_lane\")];")
            b.appendLine("        tensor<fp16, [1,1,1,\(lane)]> maskLane = slice_by_size(x=maskCache,begin=b0,size=szM)[name=string(\"m_lane\")];")
            b.appendLine("        tensor<fp16, [1,\(dim),1,\(lane)]> mix = mul(x=kLane,y=maskLane)[name=string(\"mix\")];")
            b.appendLine("        tensor<fp16, [1,\(dim),1,\(lane)]> z = mul(x=mix,y=zc)[name=string(\"z\")];")
            b.appendLine("        tensor<fp16, [1,\(dim),1,\(lane)]> outX = add(x=x,y=z)[name=string(\"out_x\")];")
            b.appendLine("        tensor<fp16, [1,\(dim),1,\(lane)]> outK = add(x=x,y=z)[name=string(\"out_k\")];")
            b.appendLine("        tensor<fp16, [1,\(dim),1,\(lane)]> outV = add(x=x,y=z)[name=string(\"out_v\")];")
            b.appendLine("    } -> (outX,outK,outV);")
            b.appendLine("}")
            return b.text
        }

        // RMSNorm(x lane-pack)
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

        // Q/K/V projections + output projection.
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

        // Temporary probe path: keep cache/mask inputs live while bypassing attention math.
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

        b.appendLine("    } -> (x2,kfFull,vfFull);")
        b.appendLine("}")
        return b.text
    }
}
