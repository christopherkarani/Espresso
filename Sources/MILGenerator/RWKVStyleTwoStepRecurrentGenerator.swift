import Foundation
import ANETypes

/// Unrolled two-step variant of the minimal RWKV-style recurrent decode cell.
///
/// This kernel accepts two token activations and one incoming state, then emits:
/// - the activation after token 0
/// - the activation after token 1
/// - the intermediate state after token 0
/// - the final state after token 1
///
/// The goal is to let the generation harness promote either the intermediate or final
/// state after exact prefix verification, without paying a second recurrent decode.
public struct RWKVStyleTwoStepRecurrentGenerator: MILProgramGenerator {
    public let laneSpatial: Int

    public init(laneSpatial: Int = 32) {
        precondition(laneSpatial > 0)
        self.laneSpatial = laneSpatial
    }

    public var inputBytes: Int {
        inputByteSizes.reduce(0, +)
    }

    public var inputByteSizes: [Int] {
        let bytes = ModelConfig.dim * laneSpatial * 2
        return [bytes, bytes, bytes]
    }

    public var outputByteSizes: [Int] {
        let bytes = ModelConfig.dim * laneSpatial * 2
        return [bytes, bytes, bytes, bytes]
    }

    public var milText: String {
        let dim = ModelConfig.dim
        let lane = laneSpatial
        let invd: Float = 1.0 / Float(dim)

        var builder = MILBuilder(reserveCapacity: 16_384)
        builder.append(MILText.header)
        builder.appendLine(
            "    func main<ios18>(tensor<fp16, [1, \(dim), 1, \(lane)]> x0, tensor<fp16, [1, \(dim), 1, \(lane)]> x1, tensor<fp16, [1, \(dim), 1, \(lane)]> stateIn) {"
        )

        builder.appendLine("        tensor<int32, [1]> raxCh = const()[name=string(\"rax_ch\"), val=tensor<int32, [1]>([1])];")
        builder.appendLine("        bool kd = const()[name=string(\"kd\"), val=bool(true)];")
        builder.append("        fp16 invd = const()[name=string(\"invd\"), val=fp16(")
        builder.appendFP16(invd)
        builder.appendLine(")];")
        builder.appendLine("        fp16 eps = const()[name=string(\"eps\"), val=fp16(0.00001)];")
        builder.appendLine("        fp16 nhalf = const()[name=string(\"nhalf\"), val=fp16(-0.5)];")
        builder.append(MILText.convConst)
        builder.appendLine("        tensor<fp16, [1,\(dim),1,1]> rw = const()[name=string(\"rw\"), val=tensor<fp16, [1,\(dim),1,1]>(BLOBFILE(path=string(\"@model_path/weights/rwkv_rms.bin\"), offset=uint64(64)))];")
        builder.appendLine("        tensor<fp16, [\(dim),\(dim),1,1]> Wx = const()[name=string(\"Wx\"), val=tensor<fp16, [\(dim),\(dim),1,1]>(BLOBFILE(path=string(\"@model_path/weights/wx.bin\"), offset=uint64(64)))];")
        builder.appendLine("        tensor<fp16, [\(dim),\(dim),1,1]> Ws = const()[name=string(\"Ws\"), val=tensor<fp16, [\(dim),\(dim),1,1]>(BLOBFILE(path=string(\"@model_path/weights/ws.bin\"), offset=uint64(64)))];")
        builder.appendLine("        tensor<fp16, [\(dim),\(dim),1,1]> Wd = const()[name=string(\"Wd\"), val=tensor<fp16, [\(dim),\(dim),1,1]>(BLOBFILE(path=string(\"@model_path/weights/wd.bin\"), offset=uint64(64)))];")
        builder.appendLine("        tensor<fp16, [\(dim),\(dim),1,1]> Wo = const()[name=string(\"Wo\"), val=tensor<fp16, [\(dim),\(dim),1,1]>(BLOBFILE(path=string(\"@model_path/weights/wo.bin\"), offset=uint64(64)))];")

        appendStepBlock(
            to: &builder,
            prefix: "s0",
            dim: dim,
            lane: lane,
            inputName: "x0",
            stateName: "stateIn",
            outputName: "x0Next",
            outputStateName: "stateMid"
        )
        appendStepBlock(
            to: &builder,
            prefix: "s1",
            dim: dim,
            lane: lane,
            inputName: "x1",
            stateName: "stateMid",
            outputName: "x1Next",
            outputStateName: "stateOut"
        )

        builder.appendLine("    } -> (x0Next,x1Next,stateMid,stateOut);")
        builder.appendLine("}")
        return builder.text
    }

    private func appendStepBlock(
        to builder: inout MILBuilder,
        prefix: String,
        dim: Int,
        lane: Int,
        inputName: String,
        stateName: String,
        outputName: String,
        outputStateName: String
    ) {
        let p = prefix + "_"
        builder.appendLine("        tensor<fp16, [1,\(dim),1,\(lane)]> \(p)sq = mul(x=\(inputName),y=\(inputName))[name=string(\"\(p)sq\")];")
        builder.appendLine("        tensor<fp16, [1,1,1,\(lane)]> \(p)ss = reduce_sum(x=\(p)sq,axes=raxCh,keep_dims=kd)[name=string(\"\(p)ss\")];")
        builder.appendLine("        tensor<fp16, [1,1,1,\(lane)]> \(p)ss2 = mul(x=\(p)ss,y=invd)[name=string(\"\(p)ss2\")];")
        builder.appendLine("        tensor<fp16, [1,1,1,\(lane)]> \(p)ss3 = add(x=\(p)ss2,y=eps)[name=string(\"\(p)ss3\")];")
        builder.appendLine("        tensor<fp16, [1,1,1,\(lane)]> \(p)rrms = pow(x=\(p)ss3,y=nhalf)[name=string(\"\(p)rrms\")];")
        builder.appendLine("        tensor<fp16, [1,\(dim),1,\(lane)]> \(p)xr = mul(x=\(inputName),y=\(p)rrms)[name=string(\"\(p)xr\")];")
        builder.appendLine("        tensor<fp16, [1,\(dim),1,\(lane)]> \(p)xn = mul(x=\(p)xr,y=rw)[name=string(\"\(p)xn\")];")
        builder.appendLine("        tensor<fp16, [1,\(dim),1,\(lane)]> \(p)xMix = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=Wx,x=\(p)xn)[name=string(\"\(p)x_mix\")];")
        builder.appendLine("        tensor<fp16, [1,\(dim),1,\(lane)]> \(p)sMix = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=Ws,x=\(stateName))[name=string(\"\(p)s_mix\")];")
        builder.appendLine("        tensor<fp16, [1,\(dim),1,\(lane)]> \(p)carry = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=Wd,x=\(stateName))[name=string(\"\(p)carry\")];")
        builder.appendLine("        tensor<fp16, [1,\(dim),1,\(lane)]> \(p)mixPre = add(x=\(p)xMix,y=\(p)sMix)[name=string(\"\(p)mix_pre\")];")
        builder.appendLine("        tensor<fp16, [1,\(dim),1,\(lane)]> \(p)gate = sigmoid(x=\(p)mixPre)[name=string(\"\(p)gate\")];")
        builder.appendLine("        tensor<fp16, [1,\(dim),1,\(lane)]> \(p)gatedCarry = mul(x=\(p)carry,y=\(p)gate)[name=string(\"\(p)gated_carry\")];")
        builder.appendLine("        tensor<fp16, [1,\(dim),1,\(lane)]> \(outputStateName) = add(x=\(p)xMix,y=\(p)gatedCarry)[name=string(\"\(outputStateName)\")];")
        builder.appendLine("        tensor<fp16, [1,\(dim),1,\(lane)]> \(p)proj = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=Wo,x=\(outputStateName))[name=string(\"\(p)proj\")];")
        builder.appendLine("        tensor<fp16, [1,\(dim),1,\(lane)]> \(outputName) = add(x=\(inputName),y=\(p)proj)[name=string(\"\(outputName)\")];")
    }
}
