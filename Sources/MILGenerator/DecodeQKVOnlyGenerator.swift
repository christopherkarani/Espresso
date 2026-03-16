import Foundation
import ANETypes
import ANEBuilder
import ANEGraphIR

/// Decode-time QKV projection kernel.
///
/// This isolates the ANE work needed before Metal-owned attention:
/// RMSNorm(x) -> Wq/Wk/Wv, with no cache reads, mask input, or output projection.
///
/// Input:
/// - `x`: `[1, dim, 1, laneSpatial]`
///
/// Outputs:
/// - `qOut`: `[1, dim, 1, laneSpatial]`
/// - `kNew`: `[1, dim, 1, laneSpatial]`
/// - `vNew`: `[1, dim, 1, laneSpatial]`
public struct DecodeQKVOnlyGenerator: MILProgramGenerator {
    public let dim: Int
    public let laneSpatial: Int
    public let architecture: LayerWeightsArchitecture

    public init(
        dim: Int = ModelConfig.dim,
        laneSpatial: Int = 32,
        architecture: LayerWeightsArchitecture = .rmsNormSwiGLU
    ) {
        precondition(dim > 0)
        precondition(laneSpatial > 0)
        self.dim = dim
        self.laneSpatial = laneSpatial
        self.architecture = architecture
    }

    public var inputBytes: Int { dim * laneSpatial * 2 }

    public var inputByteSizes: [Int] {
        [dim * laneSpatial * 2]
    }

    public var outputByteSizes: [Int] {
        [
            dim * laneSpatial * 2,
            dim * laneSpatial * 2,
            dim * laneSpatial * 2,
        ]
    }

    public var milText: String {
        LegacyGraphSupport.emitGraph { graph in
            let x = try LegacyGraphSupport.input(&graph, name: "x", channels: dim, spatial: laneSpatial)

            let normalized: Int
            switch architecture {
            case .rmsNormSwiGLU:
                normalized = try graph.rmsNorm(
                    "norm",
                    input: x,
                    dim: dim,
                    spatial: laneSpatial,
                    eps: 0.00001,
                    weightPath: "@model_path/weights/rms1.bin"
                )
            case .gpt2:
                normalized = try graph.layerNorm(
                    "norm",
                    input: x,
                    dim: dim,
                    spatial: laneSpatial,
                    eps: 0.00001,
                    gammaPath: "@model_path/weights/rms1.bin",
                    betaPath: "@model_path/weights/rms1_beta.bin"
                )
            }

            let qOut = try graph.linear(
                "q",
                input: normalized,
                inDim: dim,
                outDim: dim,
                spatial: laneSpatial,
                weightPath: "@model_path/weights/wq.bin",
                biasPath: architecture == .gpt2 ? "@model_path/weights/bq.bin" : nil
            )
            let kNew = try graph.linear(
                "k",
                input: normalized,
                inDim: dim,
                outDim: dim,
                spatial: laneSpatial,
                weightPath: "@model_path/weights/wk.bin",
                biasPath: architecture == .gpt2 ? "@model_path/weights/bk.bin" : nil
            )
            let vNew = try graph.linear(
                "v",
                input: normalized,
                inDim: dim,
                outDim: dim,
                spatial: laneSpatial,
                weightPath: "@model_path/weights/wv.bin",
                biasPath: architecture == .gpt2 ? "@model_path/weights/bv.bin" : nil
            )
            try LegacyGraphSupport.setOutputs(&graph, [("qOut", qOut), ("kNew", kNew), ("vNew", vNew)])
        }
    }
}
