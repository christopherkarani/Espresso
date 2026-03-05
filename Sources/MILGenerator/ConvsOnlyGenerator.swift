import Foundation
import ANETypes

/// Convolutions-only baseline for Experiment 1.
///
/// Measures pure convolution throughput of the 7 weight matrices
/// (Wq, Wk, Wv, Wo, W1, W3, W2) stripped of all attention logic —
/// no RMSNorm, no split-head SDPA, no softmax, no SiLU, no residuals.
///
/// DAG:
/// ```
/// x [1,768,1,256]
/// ├── SDPA path: x→Wq→Wk→Wv→Wo → c_o [1,768,1,256]
/// ├── FFN  path: W1(x) + W3(x) → W2 → c_2 [1,768,1,256]
/// └── add(c_o, c_2) → out [1,768,1,256]
/// ```
///
/// - Input:  `x`   — `[1, dim, 1, seqLen]` fp16
/// - Output: `out` — `[1, dim, 1, seqLen]` fp16
public struct ConvsOnlyGenerator: MILProgramGenerator {
    public init() {}

    public var inputBytes: Int { ModelConfig.dim * ModelConfig.seqLen * 2 }
    public var outputByteSizes: [Int] { [ModelConfig.dim * ModelConfig.seqLen * 2] }

    public var milText: String {
        let D = ModelConfig.dim     // 768
        let H = ModelConfig.hidden  // 2048
        let S = ModelConfig.seqLen  // 256

        var b = MILBuilder(reserveCapacity: 8_192)
        b.append(MILText.header)
        b.appendLine("    func main<ios18>(tensor<fp16, [1, \(D), 1, \(S)]> x) {")

        // ── Conv constants ────────────────────────────────────────────────────
        b.append(MILText.convConst)

        // ── SDPA path weight consts ───────────────────────────────────────────
        b.appendLine("        tensor<fp16, [\(D),\(D),1,1]> co_Wq = const()[name=string(\"co_Wq\"), val=tensor<fp16, [\(D),\(D),1,1]>(BLOBFILE(path=string(\"@model_path/weights/wq.bin\"), offset=uint64(64)))];")
        b.appendLine("        tensor<fp16, [\(D),\(D),1,1]> co_Wk = const()[name=string(\"co_Wk\"), val=tensor<fp16, [\(D),\(D),1,1]>(BLOBFILE(path=string(\"@model_path/weights/wk.bin\"), offset=uint64(64)))];")
        b.appendLine("        tensor<fp16, [\(D),\(D),1,1]> co_Wv = const()[name=string(\"co_Wv\"), val=tensor<fp16, [\(D),\(D),1,1]>(BLOBFILE(path=string(\"@model_path/weights/wv.bin\"), offset=uint64(64)))];")
        b.appendLine("        tensor<fp16, [\(D),\(D),1,1]> co_Wo = const()[name=string(\"co_Wo\"), val=tensor<fp16, [\(D),\(D),1,1]>(BLOBFILE(path=string(\"@model_path/weights/wo.bin\"), offset=uint64(64)))];")

        // ── FFN path weight consts ────────────────────────────────────────────
        b.appendLine("        tensor<fp16, [\(H),\(D),1,1]> co_W1 = const()[name=string(\"co_W1\"), val=tensor<fp16, [\(H),\(D),1,1]>(BLOBFILE(path=string(\"@model_path/weights/w1.bin\"), offset=uint64(64)))];")
        b.appendLine("        tensor<fp16, [\(H),\(D),1,1]> co_W3 = const()[name=string(\"co_W3\"), val=tensor<fp16, [\(H),\(D),1,1]>(BLOBFILE(path=string(\"@model_path/weights/w3.bin\"), offset=uint64(64)))];")
        b.appendLine("        tensor<fp16, [\(D),\(H),1,1]> co_W2 = const()[name=string(\"co_W2\"), val=tensor<fp16, [\(D),\(H),1,1]>(BLOBFILE(path=string(\"@model_path/weights/w2.bin\"), offset=uint64(64)))];")

        // ── SDPA path: sequential chain x → Wq → Wk → Wv → Wo ───────────────
        b.appendLine("        tensor<fp16, [1,\(D),1,\(S)]> co_qf = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=co_Wq,x=x)[name=string(\"co_qf\")];")
        b.appendLine("        tensor<fp16, [1,\(D),1,\(S)]> co_kf = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=co_Wk,x=co_qf)[name=string(\"co_kf\")];")
        b.appendLine("        tensor<fp16, [1,\(D),1,\(S)]> co_vf = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=co_Wv,x=co_kf)[name=string(\"co_vf\")];")
        b.appendLine("        tensor<fp16, [1,\(D),1,\(S)]> c_o   = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=co_Wo,x=co_vf)[name=string(\"c_o\")];")

        // ── FFN path: parallel W1(x) and W3(x), then add → W2 ────────────────
        b.appendLine("        tensor<fp16, [1,\(H),1,\(S)]> co_h1 = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=co_W1,x=x)[name=string(\"co_h1\")];")
        b.appendLine("        tensor<fp16, [1,\(H),1,\(S)]> co_h3 = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=co_W3,x=x)[name=string(\"co_h3\")];")
        b.appendLine("        tensor<fp16, [1,\(H),1,\(S)]> co_hg = add(x=co_h1,y=co_h3)[name=string(\"co_hg\")];")
        b.appendLine("        tensor<fp16, [1,\(D),1,\(S)]> c_2   = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=co_W2,x=co_hg)[name=string(\"c_2\")];")

        // ── Final merge ───────────────────────────────────────────────────────
        b.appendLine("        tensor<fp16, [1,\(D),1,\(S)]> out = add(x=c_o,y=c_2)[name=string(\"out\")];")

        b.appendLine("    } -> (out);")
        b.appendLine("}")
        return b.text
    }
}
