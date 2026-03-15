import Foundation
import ANETypes
import ANEGraphIR

public struct FFNForwardGenerator: MILProgramGenerator {
    public init() {}

    public var inputBytes: Int { ModelConfig.dim * ModelConfig.seqLen * 2 }
    public var outputByteSizes: [Int] {
        [(2 * ModelConfig.dim + 3 * ModelConfig.hidden) * ModelConfig.seqLen * 2]
    }

    public var milText: String {
        LegacyGraphSupport.emitGraph { graph in
            let x = try LegacyGraphSupport.input(&graph, name: "x", channels: ModelConfig.dim, spatial: ModelConfig.seqLen)
            let xn = try LegacyGraphSupport.rmsNorm(
                &graph,
                input: x,
                dim: ModelConfig.dim,
                spatial: ModelConfig.seqLen,
                sq: "sq",
                axisName: "rax",
                keepDimsName: "kd",
                ss: "ss",
                invdName: "invd",
                ss2: "ss2",
                epsName: "eps",
                ss3: "ss3",
                nhalfName: "nhalf",
                rrms: "rrms",
                xr: "xr",
                weightName: "rw",
                weightPath: "@model_path/weights/rms2.bin",
                output: "xn"
            )
            let h1 = try LegacyGraphSupport.conv(&graph, name: "h1", input: xn, weightName: "W1", outChannels: ModelConfig.hidden, inChannelsPerGroup: ModelConfig.dim, spatial: ModelConfig.seqLen, weightPath: "@model_path/weights/w1.bin")
            let h3 = try LegacyGraphSupport.conv(&graph, name: "h3", input: xn, weightName: "W3", outChannels: ModelConfig.hidden, inChannelsPerGroup: ModelConfig.dim, spatial: ModelConfig.seqLen, weightPath: "@model_path/weights/w3.bin")
            let sig = try graph.sigmoid("sig", input: h1)
            let silu = try graph.mul("silu", x: h1, y: sig)
            let gate = try graph.mul("gate", x: silu, y: h3)
            let y = try LegacyGraphSupport.conv(&graph, name: "y", input: gate, weightName: "W2", outChannels: ModelConfig.dim, inChannelsPerGroup: ModelConfig.hidden, spatial: ModelConfig.seqLen, weightPath: "@model_path/weights/w2.bin")
            let out = try graph.concat("out", values: [y, h1, h3, gate, xn], axis: 1, interleave: false, outShape: try ANEShape(channels: 2 * ModelConfig.dim + 3 * ModelConfig.hidden, spatial: ModelConfig.seqLen))
            try LegacyGraphSupport.setOutputs(&graph, [("out", out)])
        }
    }
}
