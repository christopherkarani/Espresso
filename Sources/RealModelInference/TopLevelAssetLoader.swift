import Foundation
import ANERuntime
import ANETypes
import ModelSupport

/// Top-level weight artifacts shared by every serving trunk: token embedding,
/// final norm, and LM head.
///
/// One module, one interface (`TopLevelAssetLoader.load`). Callers never touch
/// weight paths or sidecar rules directly; fixing a loading rule happens here
/// once and holds for GPT-2 dispatch, llama serving, and exact-CPU alike.
struct GPT2TopLevelAssets {
    let tokenEmbedding: [Float]
    let positionEmbedding: [Float]
    let finalNormGamma: [Float]
    let finalNormBeta: [Float]
    let lmHead: [Float]
    let finalNormGammaPath: String
    let finalNormBetaPath: String
    let finalNormGammaCompilePath: String
    let finalNormBetaCompilePath: String
    let finalNormGammaData: Data
    let finalNormBetaData: Data
}

struct LlamaTopLevelAssets {
    struct FactoredOutputHead: Sendable, Equatable {
        let projection: [Float]
        let expansion: [Float]
        let bottleneck: Int
        let groups: Int
    }

    let tokenEmbedding: [Float]
    let finalNormGamma: [Float]
    let lmHead: [Float]
    let lmHeadFP16: [UInt16]?
    let lmHeadHasExactFloat32Sidecar: Bool
    let factoredOutputHead: FactoredOutputHead?
    let finalNormGammaPath: String
    let finalNormGammaCompilePath: String
    let finalNormGammaData: Data
}

enum TopLevelAssets {
    case gpt2(GPT2TopLevelAssets)
    case llama(LlamaTopLevelAssets)
}

/// The llama top-level triple every consumer needs; `lmHeadFP16` is loaded
/// separately by consumers that actually run the fp16 head.
struct LlamaCoreWeights {
    let tokenEmbedding: [Float]
    let finalNormGamma: [Float]
    let lmHead: [Float]
}

/// Single production load path for an engine's top-level weights.
///
/// Historically this lived on the engine as `loadTestingTopLevelAssets` — a
/// misleading name for code that backs `RealModelInferenceEngine.build`.
enum TopLevelAssetLoader {
    static func load(
        config: MultiModelConfig,
        weightDir: String,
        weightDirURL: URL,
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) throws -> TopLevelAssets {
        switch config.architecture {
        case .gpt2:
            let topLevelPaths = try RealModelInferenceEngine.resolveTopLevelWeightPaths(config: config, weightDir: weightDir)
            let tokenEmbedding = try RealModelInferenceEngine.loadWeightTablePreferringFloat32Sidecar(
                at: topLevelPaths.tokenEmbedding,
                expectedCount: config.vocab * config.dModel
            )
            let positionEmbedding = try RealModelInferenceEngine.loadWeightTable(
                at: topLevelPaths.positionEmbedding,
                expectedCount: config.maxSeq * config.dModel
            )
            let finalNormGamma = try RealModelInferenceEngine.loadWeightTablePreferringFloat32Sidecar(
                at: topLevelPaths.finalNormGamma,
                expectedCount: config.dModel
            )
            let finalNormBeta = try RealModelInferenceEngine.loadWeightTable(
                at: topLevelPaths.finalNormBeta,
                expectedCount: config.dModel
            )
            let lmHead = try RealModelInferenceEngine.loadWeightTablePreferringFloat32Sidecar(
                at: topLevelPaths.lmHead,
                expectedCount: config.vocab * config.dModel
            )
            return .gpt2(GPT2TopLevelAssets(
                tokenEmbedding: tokenEmbedding,
                positionEmbedding: positionEmbedding,
                finalNormGamma: finalNormGamma,
                finalNormBeta: finalNormBeta,
                lmHead: lmHead,
                finalNormGammaPath: topLevelPaths.finalNormGamma,
                finalNormBetaPath: topLevelPaths.finalNormBeta,
                finalNormGammaCompilePath: RealModelInferenceEngine.compileBlobPath(actualPath: topLevelPaths.finalNormGamma, rootDir: weightDirURL),
                finalNormBetaCompilePath: RealModelInferenceEngine.compileBlobPath(actualPath: topLevelPaths.finalNormBeta, rootDir: weightDirURL),
                finalNormGammaData: WeightBlob.build(from: finalNormGamma, rows: 1, cols: finalNormGamma.count),
                finalNormBetaData: WeightBlob.build(from: finalNormBeta, rows: 1, cols: finalNormBeta.count)
            ))
        case .llama:
            let topLevelPaths = try RealModelInferenceEngine.resolveLlamaTopLevelWeightPaths(config: config, weightDir: weightDir)
            return .llama(try loadLlamaTopLevelAssets(
                config: config,
                topLevelPaths: topLevelPaths,
                weightDirURL: weightDirURL,
                environment: environment
            ))
        }
    }

    /// The llama triple (token embedding, final-norm gamma, exact-float32 LM
    /// head) shared by full asset loads, the CPU-exact runtime, and the
    /// lazily-cached exact-CPU weights.
    static func loadLlamaCoreWeights(
        config: MultiModelConfig,
        topLevelPaths: RealModelInferenceEngine.LlamaTopLevelWeightPaths
    ) throws -> LlamaCoreWeights {
        LlamaCoreWeights(
            tokenEmbedding: try RealModelInferenceEngine.loadWeightTablePreferringFloat32Sidecar(
                at: topLevelPaths.tokenEmbedding,
                expectedCount: config.vocab * config.dModel
            ),
            finalNormGamma: try RealModelInferenceEngine.loadWeightTablePreferringFloat32Sidecar(
                at: topLevelPaths.finalNormGamma,
                expectedCount: config.dModel
            ),
            lmHead: try RealModelInferenceEngine.loadWeightTablePreferringFloat32Sidecar(
                at: topLevelPaths.lmHead,
                expectedCount: config.vocab * config.dModel
            )
        )
    }

    static func loadLlamaTopLevelAssets(
        config: MultiModelConfig,
        topLevelPaths: RealModelInferenceEngine.LlamaTopLevelWeightPaths,
        weightDirURL: URL,
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) throws -> LlamaTopLevelAssets {
        let core = try loadLlamaCoreWeights(config: config, topLevelPaths: topLevelPaths)
        let lmHeadFP16 = try RealModelInferenceEngine.loadRawFP16WeightTableIfNoExactFloat32Sidecar(
            at: topLevelPaths.lmHead,
            expectedCount: config.vocab * config.dModel
        )
        let factoredOutputHead = try loadLlamaFactoredOutputHead(
            config: config,
            weightDirURL: weightDirURL,
            environment: environment
        )
        return LlamaTopLevelAssets(
            tokenEmbedding: core.tokenEmbedding,
            finalNormGamma: core.finalNormGamma,
            lmHead: core.lmHead,
            lmHeadFP16: lmHeadFP16,
            lmHeadHasExactFloat32Sidecar: lmHeadFP16 == nil,
            factoredOutputHead: factoredOutputHead,
            finalNormGammaPath: topLevelPaths.finalNormGamma,
            finalNormGammaCompilePath: RealModelInferenceEngine.compileBlobPath(actualPath: topLevelPaths.finalNormGamma, rootDir: weightDirURL),
            finalNormGammaData: WeightBlob.build(from: core.finalNormGamma, rows: 1, cols: core.finalNormGamma.count)
        )
    }

    private static func loadLlamaFactoredOutputHead(
        config: MultiModelConfig,
        weightDirURL: URL,
        environment: [String: String]
    ) throws -> LlamaTopLevelAssets.FactoredOutputHead? {
        guard config.architecture == .llama,
              environment["ESPRESSO_BUNDLE_OUTPUT_HEAD_KIND"] == "factored" else {
            return nil
        }

        guard let bottleneckRaw = environment["ESPRESSO_BUNDLE_OUTPUT_HEAD_BOTTLENECK"],
              let bottleneck = Int(bottleneckRaw),
              bottleneck > 0 else {
            throw RealModelInferenceError.invalidConfig(
                "Factored output head requires ESPRESSO_BUNDLE_OUTPUT_HEAD_BOTTLENECK > 0"
            )
        }
        guard let groupsRaw = environment["ESPRESSO_BUNDLE_OUTPUT_HEAD_GROUPS"],
              let groups = Int(groupsRaw),
              groups > 0 else {
            throw RealModelInferenceError.invalidConfig(
                "Factored output head requires ESPRESSO_BUNDLE_OUTPUT_HEAD_GROUPS > 0"
            )
        }

        let projectionPath = try RealModelInferenceEngine.resolveBundleWeightReference(
            environment["ESPRESSO_BUNDLE_OUTPUT_HEAD_PROJECTION_REF"] ?? "cls_proj.bin",
            weightDirURL: weightDirURL
        )
        let expansionPath = try RealModelInferenceEngine.resolveBundleWeightReference(
            environment["ESPRESSO_BUNDLE_OUTPUT_HEAD_EXPANSION_REF"] ?? "cls_expand.bin",
            weightDirURL: weightDirURL
        )

        let projectionCompactCount = bottleneck * (config.dModel / groups)
        let projectionDenseCount = bottleneck * config.dModel
        let expansionCompactCount = config.vocab * (bottleneck / groups)
        let expansionDenseCount = config.vocab * bottleneck
        let projection = try RealModelInferenceEngine.loadWeightTable(
            at: projectionPath,
            allowedCounts: [projectionCompactCount, projectionDenseCount]
        )
        let expansion = try RealModelInferenceEngine.loadWeightTable(
            at: expansionPath,
            allowedCounts: [expansionCompactCount, expansionDenseCount]
        )

        return LlamaTopLevelAssets.FactoredOutputHead(
            projection: projection,
            expansion: expansion,
            bottleneck: bottleneck,
            groups: groups
        )
    }
}
