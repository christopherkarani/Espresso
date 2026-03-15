import Foundation
import ANETypes

/// Decode-time FFN kernel with fused residual.
///
/// Input:
/// - `x`: `[1, dim, 1, laneSpatial]` (token packed at lane 0, remaining lanes zero)
///
/// Output:
/// - `x + ffn(x)`: `[1, dim, 1, laneSpatial]`
public struct DecodeFFNGenerator: MILProgramGenerator {
    public let laneSpatial: Int

    public init(laneSpatial: Int = 32) {
        precondition(laneSpatial > 0)
        self.laneSpatial = laneSpatial
    }

    public var inputBytes: Int { ModelConfig.dim * laneSpatial * 2 }
    public var outputByteSizes: [Int] { [ModelConfig.dim * laneSpatial * 2] }

    public var milText: String {
        LegacyGraphSupport.emitGraph { graph in
            let x = try LegacyGraphSupport.input(&graph, name: "x", channels: ModelConfig.dim, spatial: laneSpatial)
            let xn = try LegacyGraphSupport.rmsNorm(
                &graph,
                input: x,
                dim: ModelConfig.dim,
                spatial: laneSpatial,
                sq: "sq",
                axisName: "rax_ch",
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
            let y = try LegacyGraphSupport.swigluFFN(
                &graph,
                input: xn,
                dim: ModelConfig.dim,
                hidden: ModelConfig.hidden,
                spatial: laneSpatial,
                w1Path: "@model_path/weights/w1.bin",
                h1Name: "h1",
                w3Path: "@model_path/weights/w3.bin",
                h3Name: "h3",
                sigName: "sig",
                siluName: "silu",
                gateName: "gate",
                w2Path: "@model_path/weights/w2.bin",
                outputName: "y"
            )
            let out = try graph.add("out", x: x, y: y)
            try LegacyGraphSupport.setOutputs(&graph, [("out", out)])
        }
    }
}
