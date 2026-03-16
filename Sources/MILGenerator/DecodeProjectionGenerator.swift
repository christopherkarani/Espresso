import Foundation
import ANETypes
import ANEBuilder
import ANEGraphIR

/// Decode-time output projection kernel.
///
/// Inputs:
/// - `context`: `[1, dim, 1, 1]` fp32 attention context for the current token
/// - `residual`: `[1, dim, 1, 1]` fp16 residual input for the current token
///
/// Output:
/// - `Wo(context) + bias + residual`: `[1, dim, 1, 1]` fp16
public struct DecodeProjectionGenerator: MILProgramGenerator {
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
            let out = try graph.add("out", x: projected, y: residual)
            try LegacyGraphSupport.setOutputs(&graph, [("out", out)])
        }
    }
}
