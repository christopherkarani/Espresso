import Foundation
import ANETypes

/// Two sequential minimal RWKV-style recurrent decode steps fused into one MIL program.
///
/// Inputs:
/// - `x`:        `[1, dim, 1, laneSpatial]`
/// - `stateIn0`: `[1, dim, 1, laneSpatial]`
/// - `stateIn1`: `[1, dim, 1, laneSpatial]`
///
/// Outputs:
/// - `xNext`:    `[1, dim, 1, laneSpatial]`
/// - `stateOut0` `[1, dim, 1, laneSpatial]`
/// - `stateOut1` `[1, dim, 1, laneSpatial]`
///
/// Weights:
/// - layer 0: `rwkv_rms0.bin`, `wx0.bin`, `ws0.bin`, `wd0.bin`, `wo0.bin`
/// - layer 1: `rwkv_rms1.bin`, `wx1.bin`, `ws1.bin`, `wd1.bin`, `wo1.bin`
public struct RWKVStyleFusedTwoLayerStepGenerator: MILProgramGenerator {
    public let laneSpatial: Int

    public init(laneSpatial: Int = 32) {
        precondition(laneSpatial > 0)
        self.laneSpatial = laneSpatial
    }

    public var inputBytes: Int { ModelConfig.dim * laneSpatial * 2 }

    public var inputByteSizes: [Int] {
        let bytes = ModelConfig.dim * laneSpatial * 2
        return [bytes, bytes, bytes]
    }

    public var outputByteSizes: [Int] {
        let bytes = ModelConfig.dim * laneSpatial * 2
        return [bytes, bytes, bytes]
    }

    public var milText: String {
        let dim = ModelConfig.dim
        let lane = laneSpatial
        let invd: Float = 1.0 / Float(dim)

        var b = MILBuilder(reserveCapacity: 16_384)
        b.append(MILText.header)
        b.appendLine(
            "    func main<ios18>(tensor<fp16, [1, \(dim), 1, \(lane)]> x, tensor<fp16, [1, \(dim), 1, \(lane)]> stateIn0, tensor<fp16, [1, \(dim), 1, \(lane)]> stateIn1) {"
        )

        b.appendLine("        tensor<int32, [1]> raxCh = const()[name=string(\"rax_ch\"), val=tensor<int32, [1]>([1])];")
        b.appendLine("        bool kd = const()[name=string(\"kd\"), val=bool(true)];")
        b.append("        fp16 invd = const()[name=string(\"invd\"), val=fp16(")
        b.appendFP16(invd)
        b.appendLine(")];")
        b.appendLine("        fp16 eps = const()[name=string(\"eps\"), val=fp16(0.00001)];")
        b.appendLine("        fp16 nhalf = const()[name=string(\"nhalf\"), val=fp16(-0.5)];")
        b.append(MILText.convConst)

        appendLayer(
            builder: &b,
            dim: dim,
            lane: lane,
            layerIndex: 0,
            prefix: "l0_",
            inputX: "x",
            inputState: "stateIn0",
            outputX: "l0_xNext",
            outputState: "stateOut0"
        )
        appendLayer(
            builder: &b,
            dim: dim,
            lane: lane,
            layerIndex: 1,
            prefix: "l1_",
            inputX: "l0_xNext",
            inputState: "stateIn1",
            outputX: "xNext",
            outputState: "stateOut1"
        )

        b.appendLine("    } -> (xNext,stateOut0,stateOut1);")
        b.appendLine("}")
        return b.text
    }

    private func appendLayer(
        builder b: inout MILBuilder,
        dim: Int,
        lane: Int,
        layerIndex: Int,
        prefix: String,
        inputX: String,
        inputState: String,
        outputX: String,
        outputState: String
    ) {
        b.appendLine("        tensor<fp16, [1,\(dim),1,\(lane)]> \(prefix)sq = mul(x=\(inputX),y=\(inputX))[name=string(\"\(prefix)sq\")];")
        b.appendLine("        tensor<fp16, [1,1,1,\(lane)]> \(prefix)ss = reduce_sum(x=\(prefix)sq,axes=raxCh,keep_dims=kd)[name=string(\"\(prefix)ss\")];")
        b.appendLine("        tensor<fp16, [1,1,1,\(lane)]> \(prefix)ss2 = mul(x=\(prefix)ss,y=invd)[name=string(\"\(prefix)ss2\")];")
        b.appendLine("        tensor<fp16, [1,1,1,\(lane)]> \(prefix)ss3 = add(x=\(prefix)ss2,y=eps)[name=string(\"\(prefix)ss3\")];")
        b.appendLine("        tensor<fp16, [1,1,1,\(lane)]> \(prefix)rrms = pow(x=\(prefix)ss3,y=nhalf)[name=string(\"\(prefix)rrms\")];")
        b.appendLine("        tensor<fp16, [1,\(dim),1,\(lane)]> \(prefix)xr = mul(x=\(inputX),y=\(prefix)rrms)[name=string(\"\(prefix)xr\")];")
        b.appendLine("        tensor<fp16, [1,\(dim),1,1]> \(prefix)rw = const()[name=string(\"\(prefix)rw\"), val=tensor<fp16, [1,\(dim),1,1]>(BLOBFILE(path=string(\"@model_path/weights/rwkv_rms\(layerIndex).bin\"), offset=uint64(64)))];")
        b.appendLine("        tensor<fp16, [1,\(dim),1,\(lane)]> \(prefix)xn = mul(x=\(prefix)xr,y=\(prefix)rw)[name=string(\"\(prefix)xn\")];")
        b.appendLine("        tensor<fp16, [\(dim),\(dim),1,1]> \(prefix)Wx = const()[name=string(\"\(prefix)Wx\"), val=tensor<fp16, [\(dim),\(dim),1,1]>(BLOBFILE(path=string(\"@model_path/weights/wx\(layerIndex).bin\"), offset=uint64(64)))];")
        b.appendLine("        tensor<fp16, [\(dim),\(dim),1,1]> \(prefix)Ws = const()[name=string(\"\(prefix)Ws\"), val=tensor<fp16, [\(dim),\(dim),1,1]>(BLOBFILE(path=string(\"@model_path/weights/ws\(layerIndex).bin\"), offset=uint64(64)))];")
        b.appendLine("        tensor<fp16, [\(dim),\(dim),1,1]> \(prefix)Wd = const()[name=string(\"\(prefix)Wd\"), val=tensor<fp16, [\(dim),\(dim),1,1]>(BLOBFILE(path=string(\"@model_path/weights/wd\(layerIndex).bin\"), offset=uint64(64)))];")
        b.appendLine("        tensor<fp16, [\(dim),\(dim),1,1]> \(prefix)Wo = const()[name=string(\"\(prefix)Wo\"), val=tensor<fp16, [\(dim),\(dim),1,1]>(BLOBFILE(path=string(\"@model_path/weights/wo\(layerIndex).bin\"), offset=uint64(64)))];")
        b.appendLine("        tensor<fp16, [1,\(dim),1,\(lane)]> \(prefix)xMix = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=\(prefix)Wx,x=\(prefix)xn)[name=string(\"\(prefix)x_mix\")];")
        b.appendLine("        tensor<fp16, [1,\(dim),1,\(lane)]> \(prefix)sMix = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=\(prefix)Ws,x=\(inputState))[name=string(\"\(prefix)s_mix\")];")
        b.appendLine("        tensor<fp16, [1,\(dim),1,\(lane)]> \(prefix)carry = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=\(prefix)Wd,x=\(inputState))[name=string(\"\(prefix)carry\")];")
        b.appendLine("        tensor<fp16, [1,\(dim),1,\(lane)]> \(prefix)mixPre = add(x=\(prefix)xMix,y=\(prefix)sMix)[name=string(\"\(prefix)mix_pre\")];")
        b.appendLine("        tensor<fp16, [1,\(dim),1,\(lane)]> \(prefix)gate = sigmoid(x=\(prefix)mixPre)[name=string(\"\(prefix)gate\")];")
        b.appendLine("        tensor<fp16, [1,\(dim),1,\(lane)]> \(prefix)gatedCarry = mul(x=\(prefix)carry,y=\(prefix)gate)[name=string(\"\(prefix)gated_carry\")];")
        b.appendLine("        tensor<fp16, [1,\(dim),1,\(lane)]> \(outputState) = add(x=\(prefix)xMix,y=\(prefix)gatedCarry)[name=string(\"\(prefix)state_out\")];")
        b.appendLine("        tensor<fp16, [1,\(dim),1,\(lane)]> \(prefix)proj = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=\(prefix)Wo,x=\(outputState))[name=string(\"\(prefix)proj\")];")
        b.appendLine("        tensor<fp16, [1,\(dim),1,\(lane)]> \(outputX) = add(x=\(inputX),y=\(prefix)proj)[name=string(\"\(prefix)x_next\")];")
    }
}
