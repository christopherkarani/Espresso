import Foundation
import ANETypes

/// Minimal RWKV-style recurrent decode step.
///
/// The goal is not to faithfully reproduce RWKV yet. The goal is to answer the
/// architectural scaling question with the smallest compile-safe recurrent cell:
/// token input plus constant-size state, updated in one ANE eval.
///
/// Inputs:
/// - `x`:       `[1, dim, 1, laneSpatial]`
/// - `stateIn`: `[1, dim, 1, laneSpatial]`
///
/// Outputs:
/// - `xNext`:   `[1, dim, 1, laneSpatial]`
/// - `stateOut` `[1, dim, 1, laneSpatial]`
///
/// Weights:
/// - `rwkv_rms.bin`, `wx.bin`, `ws.bin`, `wd.bin`, `wo.bin`
public struct RWKVStyleRecurrentStepGenerator: MILProgramGenerator {
    public let laneSpatial: Int

    public init(laneSpatial: Int = 32) {
        precondition(laneSpatial > 0)
        self.laneSpatial = laneSpatial
    }

    public var inputBytes: Int { ModelConfig.dim * laneSpatial * 2 }

    public var inputByteSizes: [Int] {
        let bytes = ModelConfig.dim * laneSpatial * 2
        return [bytes, bytes]
    }

    public var outputByteSizes: [Int] {
        let bytes = ModelConfig.dim * laneSpatial * 2
        return [bytes, bytes]
    }

    public var milText: String {
        let dim = ModelConfig.dim
        let lane = self.laneSpatial
        let invd: Float = 1.0 / Float(dim)

        var b = MILBuilder(reserveCapacity: 8_192)
        b.append(MILText.header)
        b.appendLine(
            "    func main<ios18>(tensor<fp16, [1, \(dim), 1, \(lane)]> x, tensor<fp16, [1, \(dim), 1, \(lane)]> stateIn) {"
        )

        // RMSNorm over the input token lane.
        b.appendLine("        tensor<fp16, [1,\(dim),1,\(lane)]> sq = mul(x=x,y=x)[name=string(\"sq\")];")
        b.appendLine("        tensor<int32, [1]> raxCh = const()[name=string(\"rax_ch\"), val=tensor<int32, [1]>([1])];")
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
        b.appendLine("        tensor<fp16, [1,\(dim),1,1]> rw = const()[name=string(\"rw\"), val=tensor<fp16, [1,\(dim),1,1]>(BLOBFILE(path=string(\"@model_path/weights/rwkv_rms.bin\"), offset=uint64(64)))];")
        b.appendLine("        tensor<fp16, [1,\(dim),1,\(lane)]> xn = mul(x=xr,y=rw)[name=string(\"xn\")];")

        b.append(MILText.convConst)
        b.appendLine("        tensor<fp16, [\(dim),\(dim),1,1]> Wx = const()[name=string(\"Wx\"), val=tensor<fp16, [\(dim),\(dim),1,1]>(BLOBFILE(path=string(\"@model_path/weights/wx.bin\"), offset=uint64(64)))];")
        b.appendLine("        tensor<fp16, [\(dim),\(dim),1,1]> Ws = const()[name=string(\"Ws\"), val=tensor<fp16, [\(dim),\(dim),1,1]>(BLOBFILE(path=string(\"@model_path/weights/ws.bin\"), offset=uint64(64)))];")
        b.appendLine("        tensor<fp16, [\(dim),\(dim),1,1]> Wd = const()[name=string(\"Wd\"), val=tensor<fp16, [\(dim),\(dim),1,1]>(BLOBFILE(path=string(\"@model_path/weights/wd.bin\"), offset=uint64(64)))];")
        b.appendLine("        tensor<fp16, [\(dim),\(dim),1,1]> Wo = const()[name=string(\"Wo\"), val=tensor<fp16, [\(dim),\(dim),1,1]>(BLOBFILE(path=string(\"@model_path/weights/wo.bin\"), offset=uint64(64)))];")
        b.appendLine("        tensor<fp16, [1,\(dim),1,\(lane)]> xMix = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=Wx,x=xn)[name=string(\"x_mix\")];")
        b.appendLine("        tensor<fp16, [1,\(dim),1,\(lane)]> sMix = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=Ws,x=stateIn)[name=string(\"s_mix\")];")
        b.appendLine("        tensor<fp16, [1,\(dim),1,\(lane)]> carry = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=Wd,x=stateIn)[name=string(\"carry\")];")
        b.appendLine("        tensor<fp16, [1,\(dim),1,\(lane)]> mixPre = add(x=xMix,y=sMix)[name=string(\"mix_pre\")];")
        b.appendLine("        tensor<fp16, [1,\(dim),1,\(lane)]> gate = sigmoid(x=mixPre)[name=string(\"gate\")];")
        b.appendLine("        tensor<fp16, [1,\(dim),1,\(lane)]> gatedCarry = mul(x=carry,y=gate)[name=string(\"gated_carry\")];")
        b.appendLine("        tensor<fp16, [1,\(dim),1,\(lane)]> stateOut = add(x=xMix,y=gatedCarry)[name=string(\"state_out\")];")
        b.appendLine("        tensor<fp16, [1,\(dim),1,\(lane)]> proj = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=Wo,x=stateOut)[name=string(\"proj\")];")
        b.appendLine("        tensor<fp16, [1,\(dim),1,\(lane)]> xNext = add(x=x,y=proj)[name=string(\"x_next\")];")

        b.appendLine("    } -> (xNext,stateOut);")
        b.appendLine("}")
        return b.text
    }
}
