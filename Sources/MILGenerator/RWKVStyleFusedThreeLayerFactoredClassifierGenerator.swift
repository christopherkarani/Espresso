import Foundation
import ANETypes

/// Three sequential RWKV-style recurrent layers + factored RMSNorm + [dim→k→vocab] classifier,
/// all in a single MIL program / ANE dispatch.
///
/// Inputs:  x [1,dim,1,lane], stateIn0..2 [1,dim,1,lane]
/// Outputs: xNext [1,dim,1,lane], stateOut0..2 [1,dim,1,lane], logits [1,vocab,1,lane]
public struct RWKVStyleFusedThreeLayerFactoredClassifierGenerator: MILProgramGenerator {
    public let vocabSize: Int
    public let bottleneck: Int
    public let laneSpatial: Int
    public let groups: Int

    public init(vocabSize: Int, bottleneck: Int = 128, laneSpatial: Int = 32, groups: Int = 1) {
        precondition(vocabSize > 0)
        precondition(bottleneck > 0)
        precondition(laneSpatial > 0)
        precondition(groups > 0)
        precondition(ModelConfig.dim % groups == 0)
        precondition(bottleneck % groups == 0)
        self.vocabSize = vocabSize
        self.bottleneck = bottleneck
        self.laneSpatial = laneSpatial
        self.groups = groups
    }

    public var inputBytes: Int {
        ModelConfig.dim * laneSpatial * 2
    }

    public var inputByteSizes: [Int] {
        let bytes = inputBytes
        return [bytes, bytes, bytes, bytes]
    }

    public var outputByteSizes: [Int] {
        let stateBytes = inputBytes
        return [stateBytes, stateBytes, stateBytes, stateBytes, vocabSize * laneSpatial * 2]
    }

    public var milText: String {
        let dim = ModelConfig.dim
        let lane = laneSpatial
        let k = bottleneck
        let vocab = vocabSize
        let g = groups
        let chPerGroup = dim / g
        let invd: Float = 1.0 / Float(dim)

        var b = MILBuilder(reserveCapacity: 32_768)
        b.append(MILText.header)
        b.appendLine(
            "    func main<ios18>(tensor<fp16, [1, \(dim), 1, \(lane)]> x, tensor<fp16, [1, \(dim), 1, \(lane)]> stateIn0, tensor<fp16, [1, \(dim), 1, \(lane)]> stateIn1, tensor<fp16, [1, \(dim), 1, \(lane)]> stateIn2) {"
        )

        // Shared constants
        b.appendLine("        tensor<int32, [1]> raxCh = const()[name=string(\"rax_ch\"), val=tensor<int32, [1]>([1])];")
        b.appendLine("        bool kd = const()[name=string(\"kd\"), val=bool(true)];")
        b.append("        fp16 invd = const()[name=string(\"invd\"), val=fp16(")
        b.appendFP16(invd)
        b.appendLine(")];")
        b.appendLine("        fp16 eps = const()[name=string(\"eps\"), val=fp16(0.00001)];")
        b.appendLine("        fp16 nhalf = const()[name=string(\"nhalf\"), val=fp16(-0.5)];")
        b.appendLine("        string pt = const()[name=string(\"pt\"), val=string(\"valid\")];")
        b.appendLine("        tensor<int32, [2]> st = const()[name=string(\"st\"), val=tensor<int32, [2]>([1,1])];")
        b.appendLine("        tensor<int32, [4]> pd = const()[name=string(\"pd\"), val=tensor<int32, [4]>([0,0,0,0])];")
        b.appendLine("        tensor<int32, [2]> dl = const()[name=string(\"dl\"), val=tensor<int32, [2]>([1,1])];")
        b.appendLine("        int32 gr = const()[name=string(\"gr\"), val=int32(\(g))];")
        // Head convs always use groups=1 (dense projection)
        b.appendLine("        int32 gr1 = const()[name=string(\"gr1\"), val=int32(1)];")

        // Three recurrent layers (with groups)
        appendLayer(builder: &b, dim: dim, lane: lane, chPerGroup: chPerGroup,
                    layerIndex: 0, prefix: "l0_",
                    inputX: "x", inputState: "stateIn0", outputX: "l0_xNext", outputState: "stateOut0")
        appendLayer(builder: &b, dim: dim, lane: lane, chPerGroup: chPerGroup,
                    layerIndex: 1, prefix: "l1_",
                    inputX: "l0_xNext", inputState: "stateIn1", outputX: "l1_xNext", outputState: "stateOut1")
        appendLayer(builder: &b, dim: dim, lane: lane, chPerGroup: chPerGroup,
                    layerIndex: 2, prefix: "l2_",
                    inputX: "l1_xNext", inputState: "stateIn2", outputX: "xNext", outputState: "stateOut2")

        // Factored RMSNorm + classifier on xNext (dense, groups=1)
        b.appendLine("        tensor<fp16, [1,\(dim),1,\(lane)]> h_sq = mul(x=xNext,y=xNext)[name=string(\"h_sq\")];")
        b.appendLine("        tensor<fp16, [1,1,1,\(lane)]> h_ss = reduce_sum(x=h_sq,axes=raxCh,keep_dims=kd)[name=string(\"h_ss\")];")
        b.appendLine("        tensor<fp16, [1,1,1,\(lane)]> h_ss2 = mul(x=h_ss,y=invd)[name=string(\"h_ss2\")];")
        b.appendLine("        tensor<fp16, [1,1,1,\(lane)]> h_ss3 = add(x=h_ss2,y=eps)[name=string(\"h_ss3\")];")
        b.appendLine("        tensor<fp16, [1,1,1,\(lane)]> h_rrms = pow(x=h_ss3,y=nhalf)[name=string(\"h_rrms\")];")
        b.appendLine("        tensor<fp16, [1,\(dim),1,\(lane)]> h_xr = mul(x=xNext,y=h_rrms)[name=string(\"h_xr\")];")
        b.appendLine(
            "        tensor<fp16, [1,\(dim),1,1]> h_rw = const()[name=string(\"h_rw\"), val=tensor<fp16, [1,\(dim),1,1]>(BLOBFILE(path=string(\"@model_path/weights/rms_final.bin\"), offset=uint64(64)))];"
        )
        b.appendLine("        tensor<fp16, [1,\(dim),1,\(lane)]> h_xn = mul(x=h_xr,y=h_rw)[name=string(\"h_xn\")];")
        // Factored conv 1: [dim → k] dense
        b.appendLine(
            "        tensor<fp16, [\(k), \(dim), 1, 1]> Wproj = const()[name=string(\"Wproj\"), val=tensor<fp16, [\(k), \(dim), 1, 1]>(BLOBFILE(path=string(\"@model_path/weights/cls_proj.bin\"), offset=uint64(64)))];"
        )
        b.appendLine(
            "        tensor<fp16, [1, \(k), 1, \(lane)]> h_proj = conv(dilations=dl,groups=gr1,pad=pd,pad_type=pt,strides=st,weight=Wproj,x=h_xn)[name=string(\"h_proj\")];"
        )
        // Factored conv 2: [k → vocab] dense
        b.appendLine(
            "        tensor<fp16, [\(vocab), \(k), 1, 1]> Wexp = const()[name=string(\"Wexp\"), val=tensor<fp16, [\(vocab), \(k), 1, 1]>(BLOBFILE(path=string(\"@model_path/weights/cls_expand.bin\"), offset=uint64(64)))];"
        )
        b.appendLine(
            "        tensor<fp16, [1, \(vocab), 1, \(lane)]> logits = conv(dilations=dl,groups=gr1,pad=pd,pad_type=pt,strides=st,weight=Wexp,x=h_proj)[name=string(\"h_expand\")];"
        )

        b.appendLine("    } -> (xNext,stateOut0,stateOut1,stateOut2,logits);")
        b.appendLine("}")
        return b.text
    }

    private func appendLayer(
        builder b: inout MILBuilder,
        dim: Int,
        lane: Int,
        chPerGroup: Int,
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
        b.appendLine("        tensor<fp16, [\(dim),\(chPerGroup),1,1]> \(prefix)Wx = const()[name=string(\"\(prefix)Wx\"), val=tensor<fp16, [\(dim),\(chPerGroup),1,1]>(BLOBFILE(path=string(\"@model_path/weights/wx\(layerIndex).bin\"), offset=uint64(64)))];")
        b.appendLine("        tensor<fp16, [\(dim),\(chPerGroup),1,1]> \(prefix)Ws = const()[name=string(\"\(prefix)Ws\"), val=tensor<fp16, [\(dim),\(chPerGroup),1,1]>(BLOBFILE(path=string(\"@model_path/weights/ws\(layerIndex).bin\"), offset=uint64(64)))];")
        b.appendLine("        tensor<fp16, [\(dim),\(chPerGroup),1,1]> \(prefix)Wd = const()[name=string(\"\(prefix)Wd\"), val=tensor<fp16, [\(dim),\(chPerGroup),1,1]>(BLOBFILE(path=string(\"@model_path/weights/wd\(layerIndex).bin\"), offset=uint64(64)))];")
        b.appendLine("        tensor<fp16, [\(dim),\(chPerGroup),1,1]> \(prefix)Wo = const()[name=string(\"\(prefix)Wo\"), val=tensor<fp16, [\(dim),\(chPerGroup),1,1]>(BLOBFILE(path=string(\"@model_path/weights/wo\(layerIndex).bin\"), offset=uint64(64)))];")
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
