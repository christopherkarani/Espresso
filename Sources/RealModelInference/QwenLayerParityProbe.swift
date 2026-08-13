import Foundation
import ModelSupport

/// Per-layer parity measurement surface for llama-family artifacts.
///
/// The CPU entry point deliberately calls the same
/// `RealModelInferenceEngine.exactCPULlamaLayerForward` used by the exact-CPU decode
/// path, and the ANE entry point calls the same single-layer hybrid kernel used by the
/// hybrid decode path. Neither re-implements the math, so a parity number measured here
/// describes what is actually served.
public enum QwenLayerParityProbe {
    /// Layer outputs for every position in `inputs`, plus which backend produced them.
    public struct LayerOutputs: Sendable {
        public let backend: String
        public let outputs: [[Float]]

        public init(backend: String, outputs: [[Float]]) {
            self.backend = backend
            self.outputs = outputs
        }
    }

    /// Reads `metadata.json` from a native weight directory.
    public static func loadConfig(nativeDir: String) throws -> MultiModelConfig {
        let metadataURL = URL(fileURLWithPath: nativeDir, isDirectory: true)
            .appendingPathComponent("metadata.json")
        return try RealModelInferenceEngine.loadConfigFromMetadataFile(at: metadataURL)
    }

    /// Runs one transformer layer on the CPU over `inputs`, treating element `i` as the
    /// hidden state entering the layer at position `i`.
    ///
    /// The KV cache starts empty and grows as positions are consumed, so position `i`
    /// attends over positions `0...i` exactly as incremental decode would.
    ///
    /// - Parameter roundIntermediatesToFP16: when true, intermediates are rounded to
    ///   fp16 after each projection, matching the ANE's storage precision.
    public static func evalCPULayer(
        config: MultiModelConfig,
        nativeDir: String,
        layer: Int,
        inputs: [[Float]],
        roundIntermediatesToFP16: Bool
    ) throws -> LayerOutputs {
        try validate(config: config, layer: layer, inputs: inputs)
        let paths = LayerWeightPaths.forLayer(
            layer,
            config: config,
            blobDir: URL(fileURLWithPath: nativeDir, isDirectory: true).path
        )
        let weights = try RealModelInferenceEngine.loadExactCPULlamaLayerWeights(
            config: config,
            paths: paths
        )

        let stride = inputs.count
        var kCache = [Float](repeating: 0, count: config.kvDim * stride)
        var vCache = [Float](repeating: 0, count: config.kvDim * stride)
        var outputs: [[Float]] = []
        outputs.reserveCapacity(inputs.count)
        for (position, input) in inputs.enumerated() {
            outputs.append(
                RealModelInferenceEngine.exactCPULlamaLayerForward(
                    hidden: input,
                    layer: weights,
                    config: config,
                    position: position,
                    kCache: &kCache,
                    vCache: &vCache,
                    cacheStride: stride,
                    roundIntermediatesToFP16: roundIntermediatesToFP16
                )
            )
        }
        return LayerOutputs(
            backend: roundIntermediatesToFP16 ? "cpu_exact_fp16_rounded" : "cpu_exact_fp32",
            outputs: outputs
        )
    }

    /// Runs one transformer layer through the ANE hybrid kernel over `inputs`, using the
    /// same position/cache semantics as ``evalCPULayer(config:nativeDir:layer:inputs:roundIntermediatesToFP16:)``.
    public static func evalANELayer(
        config: MultiModelConfig,
        nativeDir: String,
        layer: Int,
        inputs: [[Float]]
    ) throws -> LayerOutputs {
        try validate(config: config, layer: layer, inputs: inputs)
        let outputs = try RealModelInferenceEngine.evalHybridSingleLlamaLayerOutputsFromInputsForTesting(
            config: config,
            weightDir: nativeDir,
            layer: layer,
            inputs: inputs
        )
        return LayerOutputs(backend: "ane_hybrid", outputs: outputs)
    }

    private static func validate(config: MultiModelConfig, layer: Int, inputs: [[Float]]) throws {
        guard config.architecture == .llama else {
            throw RealModelInferenceError.unsupportedArchitecture(
                "Per-layer parity probe supports llama-family artifacts only, got \(config.architecture)"
            )
        }
        guard layer >= 0, layer < config.nLayer else {
            throw RealModelInferenceError.invalidGenerationParameters(
                "Layer \(layer) out of range for nLayer \(config.nLayer)"
            )
        }
        guard !inputs.isEmpty else {
            throw RealModelInferenceError.invalidGenerationParameters("Probe input list must not be empty")
        }
        for (index, input) in inputs.enumerated() where input.count != config.dModel {
            throw RealModelInferenceError.invalidGenerationParameters(
                "Probe input \(index) has \(input.count) values, expected dModel \(config.dModel)"
            )
        }
    }
}
