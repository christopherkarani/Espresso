import Foundation
import ANETypes
import ANEBuilder
import ANEGraphIR

/// Decode-time output projection + FFN kernel with both residuals fused.
///
/// Inputs:
/// - `context`: `[1, dim, 1, laneSpatial]` fp32 attention context for the current token
/// - `residual`: `[1, dim, 1, laneSpatial]` fp16 residual input for the current token
///
/// Output:
/// - `x + ffn(norm(x))` where `x = Wo(context) + bias + residual`: `[1, dim, 1, laneSpatial]` fp16
public struct DecodeProjectionFFNGenerator: MILProgramGenerator {
    public let dim: Int
    public let hiddenDim: Int
    public let laneSpatial: Int
    public let architecture: LayerWeightsArchitecture

    public init(
        dim: Int = ModelConfig.dim,
        hiddenDim: Int = ModelConfig.hidden,
        laneSpatial: Int = 32,
        architecture: LayerWeightsArchitecture = .rmsNormSwiGLU
    ) {
        precondition(dim > 0)
        precondition(hiddenDim > 0)
        precondition(laneSpatial > 0)
        self.dim = dim
        self.hiddenDim = hiddenDim
        self.laneSpatial = laneSpatial
        self.architecture = architecture
    }

    public var inputBytes: Int {
        dim * laneSpatial * MemoryLayout<Float>.stride
    }

    public var inputByteSizes: [Int] {
        [
            inputBytes,
            dim * laneSpatial * 2,
        ]
    }

    public var outputByteSizes: [Int] {
        [dim * laneSpatial * 2]
    }

    public var milText: String {
        LegacyGraphSupport.emitGraph { graph in
            let context = try LegacyGraphSupport.input(
                &graph,
                name: "context",
                channels: dim,
                spatial: laneSpatial,
                dtype: .fp32
            )
            let residual = try LegacyGraphSupport.input(
                &graph,
                name: "residual",
                channels: dim,
                spatial: laneSpatial
            )
            let context16 = try graph.cast("context16", input: context, to: .fp16)
            let projected = try graph.linear(
                "proj",
                input: context16,
                inDim: dim,
                outDim: dim,
                spatial: laneSpatial,
                weightPath: "@model_path/weights/wo.bin",
                biasPath: architecture == .gpt2 ? "@model_path/weights/bo.bin" : nil
            )
            let x = try graph.add("x", x: projected, y: residual)

            let normalized: Int
            let y: Int

            switch architecture {
            case .rmsNormSwiGLU:
                normalized = try graph.rmsNorm(
                    "norm",
                    input: x,
                    dim: dim,
                    spatial: laneSpatial,
                    eps: 0.00001,
                    weightPath: "@model_path/weights/rms2.bin"
                )
                y = try graph.swigluFFN(
                    "ffn",
                    input: normalized,
                    inDim: dim,
                    hiddenDim: hiddenDim,
                    spatial: laneSpatial,
                    w1Path: "@model_path/weights/w1.bin",
                    w3Path: "@model_path/weights/w3.bin",
                    w2Path: "@model_path/weights/w2.bin"
                )
            case .gpt2:
                normalized = try graph.layerNorm(
                    "norm",
                    input: x,
                    dim: dim,
                    spatial: laneSpatial,
                    eps: 0.00001,
                    gammaPath: "@model_path/weights/rms2.bin",
                    betaPath: "@model_path/weights/rms2_beta.bin"
                )
                y = try graph.ffn(
                    "ffn",
                    input: normalized,
                    inDim: dim,
                    hiddenDim: hiddenDim,
                    spatial: laneSpatial,
                    w1Path: "@model_path/weights/w1.bin",
                    b1Path: "@model_path/weights/b1.bin",
                    w2Path: "@model_path/weights/w2.bin",
                    b2Path: "@model_path/weights/b2.bin",
                    activation: .gelu
                )
            }

            let out = try graph.add("out", x: x, y: y)
            try LegacyGraphSupport.setOutputs(&graph, [("out", out)])
        }
    }
}
