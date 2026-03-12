import Foundation
import ANETypes

/// Three sequential state-free RWKV-style layers fused into a single MIL program.
///
/// This is exact when each recurrent layer ignores incoming state (`Ws == 0`
/// and `Wd == 0`). The fused program keeps the safe one-input/one-output ANE
/// contract while collapsing three recurrent layer dispatches into one eval.
public struct RWKVStyleFusedThreeLayerStateFreeStepGenerator: MILProgramGenerator {
    public let laneSpatial: Int

    public init(laneSpatial: Int = 32) {
        precondition(laneSpatial > 0)
        self.laneSpatial = laneSpatial
    }

    public var inputBytes: Int { ModelConfig.dim * laneSpatial * 2 }

    public var inputByteSizes: [Int] {
        [inputBytes]
    }

    public var outputByteSizes: [Int] {
        [inputBytes]
    }

    public var milText: String {
        let dim = ModelConfig.dim
        let lane = laneSpatial
        let invd: Float = 1.0 / Float(dim)

        var builder = MILBuilder(reserveCapacity: 24_576)
        builder.append(MILText.header)
        builder.appendLine("    func main<ios18>(tensor<fp16, [1, \(dim), 1, \(lane)]> x) {")
        builder.appendLine("        tensor<int32, [1]> raxCh = const()[name=string(\"rax_ch\"), val=tensor<int32, [1]>([1])];")
        builder.appendLine("        bool kd = const()[name=string(\"kd\"), val=bool(true)];")
        builder.append("        fp16 invd = const()[name=string(\"invd\"), val=fp16(")
        builder.appendFP16(invd)
        builder.appendLine(")];")
        builder.appendLine("        fp16 eps = const()[name=string(\"eps\"), val=fp16(0.00001)];")
        builder.appendLine("        fp16 nhalf = const()[name=string(\"nhalf\"), val=fp16(-0.5)];")
        builder.append(MILText.convConst)

        appendLayer(
            builder: &builder,
            dim: dim,
            lane: lane,
            layerIndex: 0,
            prefix: "l0_",
            inputX: "x",
            outputX: "l0_xNext"
        )
        appendLayer(
            builder: &builder,
            dim: dim,
            lane: lane,
            layerIndex: 1,
            prefix: "l1_",
            inputX: "l0_xNext",
            outputX: "l1_xNext"
        )
        appendLayer(
            builder: &builder,
            dim: dim,
            lane: lane,
            layerIndex: 2,
            prefix: "l2_",
            inputX: "l1_xNext",
            outputX: "xNext"
        )

        builder.appendLine("    } -> (xNext);")
        builder.appendLine("}")
        return builder.text
    }

    private func appendLayer(
        builder: inout MILBuilder,
        dim: Int,
        lane: Int,
        layerIndex: Int,
        prefix: String,
        inputX: String,
        outputX: String
    ) {
        builder.appendLine("        tensor<fp16, [1,\(dim),1,\(lane)]> \(prefix)sq = mul(x=\(inputX),y=\(inputX))[name=string(\"\(prefix)sq\")];")
        builder.appendLine("        tensor<fp16, [1,1,1,\(lane)]> \(prefix)ss = reduce_sum(x=\(prefix)sq,axes=raxCh,keep_dims=kd)[name=string(\"\(prefix)ss\")];")
        builder.appendLine("        tensor<fp16, [1,1,1,\(lane)]> \(prefix)ss2 = mul(x=\(prefix)ss,y=invd)[name=string(\"\(prefix)ss2\")];")
        builder.appendLine("        tensor<fp16, [1,1,1,\(lane)]> \(prefix)ss3 = add(x=\(prefix)ss2,y=eps)[name=string(\"\(prefix)ss3\")];")
        builder.appendLine("        tensor<fp16, [1,1,1,\(lane)]> \(prefix)rrms = pow(x=\(prefix)ss3,y=nhalf)[name=string(\"\(prefix)rrms\")];")
        builder.appendLine("        tensor<fp16, [1,\(dim),1,\(lane)]> \(prefix)xr = mul(x=\(inputX),y=\(prefix)rrms)[name=string(\"\(prefix)xr\")];")
        builder.appendLine("        tensor<fp16, [1,\(dim),1,1]> \(prefix)rw = const()[name=string(\"\(prefix)rw\"), val=tensor<fp16, [1,\(dim),1,1]>(BLOBFILE(path=string(\"@model_path/weights/rwkv_rms\(layerIndex).bin\"), offset=uint64(64)))];")
        builder.appendLine("        tensor<fp16, [1,\(dim),1,\(lane)]> \(prefix)xn = mul(x=\(prefix)xr,y=\(prefix)rw)[name=string(\"\(prefix)xn\")];")
        builder.appendLine("        tensor<fp16, [\(dim),\(dim),1,1]> \(prefix)Wx = const()[name=string(\"\(prefix)Wx\"), val=tensor<fp16, [\(dim),\(dim),1,1]>(BLOBFILE(path=string(\"@model_path/weights/wx\(layerIndex).bin\"), offset=uint64(64)))];")
        builder.appendLine("        tensor<fp16, [\(dim),\(dim),1,1]> \(prefix)Wo = const()[name=string(\"\(prefix)Wo\"), val=tensor<fp16, [\(dim),\(dim),1,1]>(BLOBFILE(path=string(\"@model_path/weights/wo\(layerIndex).bin\"), offset=uint64(64)))];")
        builder.appendLine("        tensor<fp16, [1,\(dim),1,\(lane)]> \(prefix)xMix = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=\(prefix)Wx,x=\(prefix)xn)[name=string(\"\(prefix)x_mix\")];")
        builder.appendLine("        tensor<fp16, [1,\(dim),1,\(lane)]> \(prefix)proj = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=\(prefix)Wo,x=\(prefix)xMix)[name=string(\"\(prefix)proj\")];")
        builder.appendLine("        tensor<fp16, [1,\(dim),1,\(lane)]> \(outputX) = add(x=\(inputX),y=\(prefix)proj)[name=string(\"\(outputX)\")];")
    }
}
