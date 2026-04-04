import Foundation
import IOSurface
import Darwin
import Accelerate
import ANEInterop
import ANEBuilder
import ANECodegen
import ANEGraphIR
import ANEPasses
import ANERuntime
import ANETypes
import CPUOps
import Espresso
import MILGenerator
import ModelSupport

public struct GenerationResult: Sendable {
    public let text: String
    public let tokens: [TokenID]
    public let promptTokens: [TokenID]
    public let tokenLatenciesMs: [Double]
    public let tokensPerSecond: Double
    public let compileTimeMs: Double
    public let firstTokenLatencyMs: Double
    public let exactHeadBackend: String
    public let cachedBindingsEnabled: Bool
    public let committedExactTokensPerPass: Double?
    public let acceptedFutureTokensPerPass: Double?

    public init(
        text: String,
        tokens: [TokenID],
        promptTokens: [TokenID],
        tokenLatenciesMs: [Double] = [],
        tokensPerSecond: Double,
        compileTimeMs: Double,
        firstTokenLatencyMs: Double,
        exactHeadBackend: String = "unknown",
        cachedBindingsEnabled: Bool = false,
        committedExactTokensPerPass: Double? = nil,
        acceptedFutureTokensPerPass: Double? = nil
    ) {
        self.text = text
        self.tokens = tokens
        self.promptTokens = promptTokens
        self.tokenLatenciesMs = tokenLatenciesMs
        self.tokensPerSecond = tokensPerSecond
        self.compileTimeMs = compileTimeMs
        self.firstTokenLatencyMs = firstTokenLatencyMs
        self.exactHeadBackend = exactHeadBackend
        self.cachedBindingsEnabled = cachedBindingsEnabled
        self.committedExactTokensPerPass = committedExactTokensPerPass
        self.acceptedFutureTokensPerPass = acceptedFutureTokensPerPass
    }
}

public struct GenerationStep: Sendable {
    public let token: TokenID
    public let generatedTokens: [TokenID]
    public let text: String
    public let tokenLatencyMs: Double
    public let elapsedMs: Double
    public let firstTokenLatencyMs: Double
    public let tokensPerSecond: Double

    public init(
        token: TokenID,
        generatedTokens: [TokenID],
        text: String,
        tokenLatencyMs: Double,
        elapsedMs: Double,
        firstTokenLatencyMs: Double,
        tokensPerSecond: Double
    ) {
        self.token = token
        self.generatedTokens = generatedTokens
        self.text = text
        self.tokenLatencyMs = tokenLatencyMs
        self.elapsedMs = elapsedMs
        self.firstTokenLatencyMs = firstTokenLatencyMs
        self.tokensPerSecond = tokensPerSecond
    }
}

public enum RealModelInferenceError: Error, Sendable, Equatable, LocalizedError {
    case invalidConfig(String)
    case unsupportedArchitecture(String)
    case missingPath(String)
    case invalidMetadata(field: String, expected: String, actual: String)
    case invalidWeightCount(path: String, expected: Int, actual: Int)
    case invalidPrompt(String)
    case invalidGenerationParameters(String)
    case runtimeFailure(String)

    public var errorDescription: String? {
        switch self {
        case let .invalidConfig(message):
            return "Invalid model config: \(message)"
        case let .unsupportedArchitecture(message):
            return message
        case let .missingPath(path):
            return "Missing required path: \(path)"
        case let .invalidMetadata(field, expected, actual):
            return "metadata.json mismatch for \(field): expected \(expected), got \(actual)"
        case let .invalidWeightCount(path, expected, actual):
            return "Unexpected weight count for \(path): expected \(expected), got \(actual)"
        case let .invalidPrompt(message):
            return message
        case let .invalidGenerationParameters(message):
            return message
        case let .runtimeFailure(message):
            return message
        }
    }
}

public struct RealModelInferenceEngine: ~Copyable {
    private static let minimumANEIOSurfaceBytes = 49_152
    private static let classifierArgmaxBlockSize = 4_000

    public struct GPT2AttentionCompileProbeResult: Sendable, Equatable {
        public let spatial: Int
        public let deploymentTarget: String
        public let normKind: String
        public let diagnostics: [String]
        public let compileSucceeded: Bool
        public let error: String?

        public init(
            spatial: Int,
            deploymentTarget: String,
            normKind: String,
            diagnostics: [String],
            compileSucceeded: Bool,
            error: String?
        ) {
            self.spatial = spatial
            self.deploymentTarget = deploymentTarget
            self.normKind = normKind
            self.diagnostics = diagnostics
            self.compileSucceeded = compileSucceeded
            self.error = error
        }
    }

    enum GPT2NormKind: String, Sendable {
        case layerNorm = "layernorm"
        case rmsNorm = "rmsnorm"
    }

    enum HybridGreedyHeadMode: Sendable, Equatable {
        case normThenClassifier
        case classifierOnlyFactored
        case classifierOnlyFused
    }

    static func supportsLlamaMetalRoPEFastPath(
        cachedBindingsAvailable: Bool,
        kBindingContainsKVCache: Bool
    ) -> Bool {
        cachedBindingsAvailable && !kBindingContainsKVCache
    }

    static func supportsHybridCachedBindings(
        config: MultiModelConfig,
        environment: [String: String]
    ) -> Bool {
        if environment["ESPRESSO_DISABLE_HYBRID_CACHED_BINDINGS"] == "1" {
            return false
        }
        if config.architecture == .llama {
            if isStories110MVariant(config) {
                return true
            }
            return environment["ESPRESSO_ENABLE_LLAMA_HYBRID_CACHED_BINDINGS"] == "1"
        }
        return true
    }

    static func supportsHybridDonorDelta(
        config: MultiModelConfig,
        environment: [String: String]
    ) -> Bool {
        if environment["ESPRESSO_DISABLE_HYBRID_DONOR_DELTA"] == "1" {
            return false
        }
        if environment["ESPRESSO_ENABLE_HYBRID_DONOR_DELTA"] == "1" {
            return true
        }
        return true
    }

    static func isStories110MVariant(_ config: MultiModelConfig) -> Bool {
        let normalizedName = config.name
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
        return normalizedName == "stories110m" || normalizedName.contains("stories110m")
    }

    static func resolveClassifierStrategy(
        config: MultiModelConfig,
        hasExactFloat32LMHead: Bool,
        environment: [String: String]
    ) -> ClassifierStrategy {
        if let forced = forcedExactHeadBackend(environment: environment) {
            return forced
        }
        return ClassifierStrategy.select(
            for: config,
            hasExactFloat32LMHead: hasExactFloat32LMHead
        )
    }

    static func forcedExactHeadBackend(environment: [String: String]) -> ClassifierStrategy? {
        guard let rawValue = environment["ESPRESSO_FORCE_EXACT_HEAD_BACKEND"]?
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased(),
              !rawValue.isEmpty else {
            return nil
        }

        switch rawValue {
        case "ane", "ane_classifier":
            return .ane
        case "cpu_partitioned_fp32", "partitioned", "fp32":
            return .cpuPartitionedFP32
        case "cpu_fp16_tiled", "fp16_tiled", "fp16":
            return .cpuFP16Tiled
        default:
            return nil
        }
    }

    static func usesHybridLayerInputRebinding(
        architecture: MultiModelConfig.Architecture,
        environment: [String: String]
    ) -> Bool {
        if environment["ESPRESSO_DISABLE_HYBRID_LAYER_INPUT_REBIND"] == "1" {
            return false
        }
        return architecture != .llama || environment["ESPRESSO_ENABLE_LLAMA_HYBRID_LAYER_INPUT_REBIND"] == "1"
    }

    static func usesLlamaHybridFusedExactHead(
        config: MultiModelConfig,
        environment: [String: String]
    ) -> Bool {
        guard config.architecture == .llama else {
            return false
        }
        if environment["ESPRESSO_DISABLE_LLAMA_HYBRID_FUSED_EXACT_HEAD"] == "1" {
            return false
        }
        if environment["ESPRESSO_ENABLE_LLAMA_HYBRID_FUSED_EXACT_HEAD"] == "1" {
            return true
        }
        return Self.isStories110MVariant(config)
    }

    static func prefersCPUDecodeAttention(
        config: MultiModelConfig,
        environment: [String: String]
    ) -> Bool {
        if environment["ESPRESSO_FORCE_METAL_DECODE_ATTENTION"] == "1" {
            return false
        }
        if environment["ESPRESSO_USE_CPU_DECODE_ATTENTION"] == "1" {
            return true
        }
        guard config.architecture == .llama else {
            return false
        }
        return config.name.trimmingCharacters(in: .whitespacesAndNewlines).lowercased().contains("qwen")
    }

    static func prefersCPUExactQKV(
        config: MultiModelConfig,
        environment: [String: String]
    ) -> Bool {
        if environment["ESPRESSO_FORCE_ANE_QKV"] == "1" {
            return false
        }
        if environment["ESPRESSO_USE_CPU_EXACT_QKV"] == "1" {
            return true
        }
        return false
    }

    static func shouldRoundCPUExactDecodeIntermediatesToFP16(
        env: [String: String] = ProcessInfo.processInfo.environment
    ) -> Bool {
        guard let rawValue = env["ESPRESSO_DEBUG_CPU_EXACT_DECODE_KEEP_FP32_INTERMEDIATES"] else {
            return false
        }
        switch rawValue.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() {
        case "0", "false", "no", "off":
            return true
        default:
            return false
        }
    }

    static func prefersCPUExactDecode(
        config: MultiModelConfig,
        environment: [String: String]
    ) -> Bool {
        if environment["ESPRESSO_FORCE_HYBRID_DECODE"] == "1" {
            return false
        }
        if environment["ESPRESSO_USE_CPU_EXACT_DECODE"] == "1" {
            return true
        }
        guard config.architecture == .llama else {
            return false
        }
        return config.name.trimmingCharacters(in: .whitespacesAndNewlines).lowercased().contains("qwen")
    }

    static func milDeploymentTarget(
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> String {
        let rawValue = environment["ESPRESSO_MIL_DEPLOYMENT_TARGET"]?
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return rawValue?.isEmpty == false ? rawValue! : "ios18"
    }

    static func gpt2NormKind(
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> GPT2NormKind {
        if environment["ESPRESSO_GPT2_USE_RMS_NORM"] == "1" {
            return .rmsNorm
        }
        let rawValue = environment["ESPRESSO_GPT2_NORM"]?
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
        switch rawValue {
        case "rms", "rmsnorm":
            return .rmsNorm
        default:
            return .layerNorm
        }
    }

    static func hybridGreedyHeadMode(
        config: MultiModelConfig,
        hasFactoredOutputHead: Bool,
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> HybridGreedyHeadMode {
        guard config.architecture == .llama else {
            return .normThenClassifier
        }
        if hasFactoredOutputHead {
            return .classifierOnlyFactored
        }
        if usesLlamaHybridFusedExactHead(config: config, environment: environment) {
            return .classifierOnlyFused
        }
        return .normThenClassifier
    }

    enum LlamaGenerationPath: Sendable, Equatable {
        case hybrid
        case exactCPU
    }

    static func llamaGenerationPath(
        config: MultiModelConfig,
        environment: [String: String]
    ) -> LlamaGenerationPath {
        prefersCPUExactDecode(config: config, environment: environment) ? .exactCPU : .hybrid
    }

    struct TopLevelWeightPaths: Sendable, Equatable {
        let tokenEmbedding: String
        let positionEmbedding: String
        let finalNormGamma: String
        let finalNormBeta: String
        let lmHead: String
    }

    private struct GPT2TopLevelAssets {
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

    private struct LlamaTopLevelAssets {
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

    private enum TopLevelAssets {
        case gpt2(GPT2TopLevelAssets)
        case llama(LlamaTopLevelAssets)
    }

    struct AttentionTestingOutputs {
        let hidden: [Float]
        let kCache: [Float]
        let vCache: [Float]
    }

    struct RawQKVTestingOutputs {
        let qOut: [Float]
        let kOut: [Float]
        let vOut: [Float]
    }

    struct QKVInputStabilityTestingOutputs {
        let inputBeforeQKV: [Float]
        let inputAfterQKV: [Float]
    }

    struct HookedKCacheTestingOutputs {
        let rawKOut: [Float]
        let hookedKOut: [Float]
        let hookedKOutSurface: [Float]
        let kCache: [Float]
    }

    struct DecodeProjectionTestingOutputs {
        let output: [Float]
    }

    struct DecodeFFNTestingOutputs {
        let output: [Float]
    }

    struct DecodeFFNStagesTestingOutputs {
        let gateLinear: [Float]
        let upLinear: [Float]
        let siluGate: [Float]
        let gated: [Float]
        let down: [Float]
    }

    struct HybridMetalContextTestingOutputs {
        let context: [Float]
        let qOut: [Float]
        let kOut: [Float]
        let vOut: [Float]
    }

    struct HookedHybridMetalContextTestingOutputs {
        let context: [Float]
        let qOut: [Float]
        let kCache: [Float]
        let vCache: [Float]
    }

    struct LayerHiddenLineageTestingOutputs {
        let layerHiddenStates: [[Float]]
    }

    struct SingleLayerDetailedTestingOutputs {
        let hidden: [Float]
        let context: [Float]
        let projectionOut: [Float]
        let qOut: [Float]
        let kCache: [Float]
        let vCache: [Float]
    }

    struct LlamaQKNormWeights: Sendable {
        let q: [Float]
        let k: [Float]
    }

    struct LlamaCPUQKVWeights: Sendable {
        let rmsAtt: [Float]
        let wq: [Float]
        let wk: [Float]
        let wv: [Float]
        let qNorm: [Float]?
        let kNorm: [Float]?
    }

    struct ExactCPULlamaLayerWeights: Sendable {
        let rmsAtt: [Float]
        let wq: [Float]
        let wk: [Float]
        let wv: [Float]
        let wo: [Float]
        let rmsFfn: [Float]
        let w1: [Float]
        let w2: [Float]
        let w3: [Float]
        let qNorm: [Float]?
        let kNorm: [Float]?
    }

    private struct CachedExactCPULlamaWeights: Sendable {
        let tokenEmbedding: [Float]
        let finalNormGamma: [Float]
        let lmHead: [Float]
        let lmHeadFP16: [UInt16]?
        let layers: [ExactCPULlamaLayerWeights]
    }

    private struct ExactTwoTokenDraftDescriptor: Sendable, Decodable {
        let modelDir: String
        let tokenizerDir: String?
        let modelID: String?

        enum CodingKeys: String, CodingKey {
            case modelDir = "model_dir"
            case tokenizerDir = "tokenizer_dir"
            case modelID = "model_id"
        }
    }

    private struct ResolvedExactTwoTokenDraft: Sendable {
        let descriptor: ExactTwoTokenDraftDescriptor
        let descriptorURL: URL
        let weightDirURL: URL
        let config: MultiModelConfig
    }

    private struct CPUExactLlamaCheckpoint: Sendable {
        let visibleTokenCount: Int
        let lastHidden: [Float]
        let kCaches: [[Float]]
        let vCaches: [[Float]]
    }

    private struct CPUExactLlamaRuntime: Sendable {
        let config: MultiModelConfig
        let roundIntermediatesToFP16: Bool
        let tokenEmbedding: [Float]
        let finalNormGamma: [Float]
        let lmHead: [Float]
        let layers: [ExactCPULlamaLayerWeights]
        let classifierBlockMaxNorms: [Float]
        var classifierLogitsScratch: [Float]
        var kCaches: [[Float]]
        var vCaches: [[Float]]
        var lastHidden: [Float]
        var visibleTokenCount: Int

        init(config: MultiModelConfig, weightDirURL: URL) throws {
            let topLevelPaths = try RealModelInferenceEngine.resolveLlamaTopLevelWeightPaths(
                config: config,
                weightDir: weightDirURL.path
            )
            self.config = config
            self.roundIntermediatesToFP16 = RealModelInferenceEngine.shouldRoundCPUExactDecodeIntermediatesToFP16()
            self.tokenEmbedding = try RealModelInferenceEngine.loadWeightTablePreferringFloat32Sidecar(
                at: topLevelPaths.tokenEmbedding,
                expectedCount: config.vocab * config.dModel
            )
            self.finalNormGamma = try RealModelInferenceEngine.loadWeightTablePreferringFloat32Sidecar(
                at: topLevelPaths.finalNormGamma,
                expectedCount: config.dModel
            )
            self.lmHead = try RealModelInferenceEngine.loadWeightTablePreferringFloat32Sidecar(
                at: topLevelPaths.lmHead,
                expectedCount: config.vocab * config.dModel
            )
            self.layers = try (0..<config.nLayer).map { layerIndex in
                let paths = LayerWeightPaths.forLayer(
                    layerIndex,
                    config: config,
                    blobDir: weightDirURL.path
                )
                return try RealModelInferenceEngine.loadExactCPULlamaLayerWeights(
                    config: config,
                    paths: paths
                )
            }
            self.classifierBlockMaxNorms = lmHead.withUnsafeBufferPointer { weightBuffer in
                RealModelInferenceEngine.precomputeClassifierBlockMaxNorms(
                    classifier: weightBuffer.baseAddress!,
                    vocabSize: config.vocab,
                    dim: config.dModel,
                    blockSize: RealModelInferenceEngine.classifierArgmaxBlockSize
                )
            }
            self.classifierLogitsScratch = [Float](
                repeating: 0,
                count: min(RealModelInferenceEngine.classifierArgmaxBlockSize, config.vocab)
            )
            self.kCaches = Array(
                repeating: [Float](repeating: 0, count: config.kvDim * config.maxSeq),
                count: config.nLayer
            )
            self.vCaches = Array(
                repeating: [Float](repeating: 0, count: config.kvDim * config.maxSeq),
                count: config.nLayer
            )
            self.lastHidden = [Float](repeating: 0, count: config.dModel)
            self.visibleTokenCount = 0
        }

        mutating func reset() {
            for layerIndex in 0..<config.nLayer {
                kCaches[layerIndex].withUnsafeMutableBufferPointer { pointer in
                    for index in pointer.indices {
                        pointer[index] = 0
                    }
                }
                vCaches[layerIndex].withUnsafeMutableBufferPointer { pointer in
                    for index in pointer.indices {
                        pointer[index] = 0
                    }
                }
            }
            lastHidden.withUnsafeMutableBufferPointer { pointer in
                for index in pointer.indices {
                    pointer[index] = 0
                }
            }
            visibleTokenCount = 0
        }

        mutating func prefill(promptTokens: [TokenID]) throws {
            guard !promptTokens.isEmpty else {
                throw RealModelInferenceError.invalidGenerationParameters("Prompt tokens must not be empty")
            }
            reset()
            for token in promptTokens {
                try advance(token: token)
            }
        }

        mutating func captureCheckpoint() -> CPUExactLlamaCheckpoint {
            CPUExactLlamaCheckpoint(
                visibleTokenCount: visibleTokenCount,
                lastHidden: lastHidden,
                kCaches: kCaches,
                vCaches: vCaches
            )
        }

        mutating func rollback(to checkpoint: CPUExactLlamaCheckpoint) {
            visibleTokenCount = checkpoint.visibleTokenCount
            lastHidden = checkpoint.lastHidden
            kCaches = checkpoint.kCaches
            vCaches = checkpoint.vCaches
        }

        mutating func selectGreedyToken() -> TokenID {
            let normalized = RealModelInferenceEngine.rmsNorm(
                lastHidden,
                weight: finalNormGamma,
                eps: Float(config.normEps)
            )
            return TokenID(exactClassifierArgmax(normalized))
        }

        mutating func advance(token: TokenID) throws {
            guard visibleTokenCount < config.maxSeq else {
                throw RealModelInferenceError.runtimeFailure(
                    "Draft runtime position \(visibleTokenCount) exceeds context \(config.maxSeq)"
                )
            }
            lastHidden = maybeRound(try forwardToken(token, position: visibleTokenCount))
            visibleTokenCount += 1
        }

        private func maybeRound(_ values: [Float]) -> [Float] {
            roundIntermediatesToFP16 ? RealModelInferenceEngine.roundFloat16Vector(values) : values
        }

        private mutating func forwardToken(_ token: TokenID, position: Int) throws -> [Float] {
            guard Int(token) >= 0, Int(token) < config.vocab else {
                throw RealModelInferenceError.runtimeFailure("Draft runtime token \(token) is outside vocab \(config.vocab)")
            }
            var hidden = Array(tokenEmbedding[Int(token) * config.dModel..<(Int(token) + 1) * config.dModel])
            for layerIndex in 0..<config.nLayer {
                let layer = layers[layerIndex]
                let attnNormed = RealModelInferenceEngine.rmsNorm(
                    hidden,
                    weight: layer.rmsAtt,
                    eps: Float(config.normEps)
                )
                var q = maybeRound(
                    RealModelInferenceEngine.multiplyRowMajorMatrix(
                        matrix: layer.wq,
                        rows: config.attentionDim,
                        cols: config.dModel,
                        vector: attnNormed
                    )
                )
                var k = maybeRound(
                    RealModelInferenceEngine.multiplyRowMajorMatrix(
                        matrix: layer.wk,
                        rows: config.kvDim,
                        cols: config.dModel,
                        vector: attnNormed
                    )
                )
                let vRounded = maybeRound(
                    RealModelInferenceEngine.multiplyRowMajorMatrix(
                        matrix: layer.wv,
                        rows: config.kvDim,
                        cols: config.dModel,
                        vector: attnNormed
                    )
                )

                if let qNorm = layer.qNorm {
                    q.withUnsafeMutableBufferPointer { values in
                        qNorm.withUnsafeBufferPointer { weights in
                            RealModelInferenceEngine.applyPerHeadRMSNormInPlace(
                                values: values,
                                weights: weights,
                                headCount: config.nHead,
                                headDim: config.headDim,
                                epsilon: Float(config.normEps)
                            )
                        }
                    }
                }
                if let kNorm = layer.kNorm {
                    k.withUnsafeMutableBufferPointer { values in
                        kNorm.withUnsafeBufferPointer { weights in
                            RealModelInferenceEngine.applyPerHeadRMSNormInPlace(
                                values: values,
                                weights: weights,
                                headCount: config.nKVHead,
                                headDim: config.headDim,
                                epsilon: Float(config.normEps)
                            )
                        }
                    }
                }

                q = maybeRound(
                    RealModelInferenceEngine.applyHalfSplitRoPEPerHead(
                        q,
                        heads: config.nHead,
                        headDim: config.headDim,
                        position: position,
                        theta: config.ropeTheta
                    )
                )
                k = maybeRound(
                    RealModelInferenceEngine.applyHalfSplitRoPEPerHead(
                        k,
                        heads: config.nKVHead,
                        headDim: config.headDim,
                        position: position,
                        theta: config.ropeTheta
                    )
                )

                for channel in 0..<config.kvDim {
                    kCaches[layerIndex][channel * config.maxSeq + position] = k[channel]
                    vCaches[layerIndex][channel * config.maxSeq + position] = vRounded[channel]
                }

                let context = RealModelInferenceEngine.decodeContextFromCaches(
                    qOut: q,
                    kCache: kCaches[layerIndex],
                    vCache: vCaches[layerIndex],
                    heads: config.nHead,
                    kvHeads: config.nKVHead,
                    headDim: config.headDim,
                    visibleTokenCount: position + 1,
                    cacheStride: config.maxSeq
                )

                let projected = maybeRound(
                    zip(
                        hidden,
                        RealModelInferenceEngine.multiplyRowMajorMatrix(
                            matrix: layer.wo,
                            rows: config.dModel,
                            cols: config.attentionDim,
                            vector: context
                        )
                    ).map(+)
                )
                let ffnNormed = RealModelInferenceEngine.rmsNorm(
                    projected,
                    weight: layer.rmsFfn,
                    eps: Float(config.normEps)
                )
                let gate = RealModelInferenceEngine.multiplyRowMajorMatrix(
                    matrix: layer.w1,
                    rows: config.hiddenDim,
                    cols: config.dModel,
                    vector: ffnNormed
                )
                let up = RealModelInferenceEngine.multiplyRowMajorMatrix(
                    matrix: layer.w3,
                    rows: config.hiddenDim,
                    cols: config.dModel,
                    vector: ffnNormed
                )
                let activated = zip(gate, up).map { RealModelInferenceEngine.silu($0) * $1 }
                let down = RealModelInferenceEngine.multiplyRowMajorMatrix(
                    matrix: layer.w2,
                    rows: config.dModel,
                    cols: config.hiddenDim,
                    vector: activated
                )
                hidden = maybeRound(zip(projected, down).map(+))
            }
            return hidden
        }

        private mutating func exactClassifierArgmax(_ hidden: [Float]) -> Int {
            hidden.withUnsafeBufferPointer { hiddenBuffer in
                lmHead.withUnsafeBufferPointer { weightBuffer in
                    classifierBlockMaxNorms.withUnsafeBufferPointer { normsBuffer in
                        classifierLogitsScratch.withUnsafeMutableBufferPointer { scratchBuffer in
                            guard let hiddenBase = hiddenBuffer.baseAddress,
                                  let weightBase = weightBuffer.baseAddress,
                                  let normsBase = normsBuffer.baseAddress,
                                  let scratchBase = scratchBuffer.baseAddress else {
                                return 0
                            }
                            return RealModelInferenceEngine.partitionedArgmax(
                                classifier: weightBase,
                                input: hiddenBase,
                                logitsScratch: scratchBase,
                                blockMaxNorms: normsBase,
                                vocabSize: config.vocab,
                                dim: config.dModel,
                                blockSize: RealModelInferenceEngine.classifierArgmaxBlockSize
                            )
                        }
                    }
                }
            }
        }
    }

    private enum LoadedTokenizer {
        case gpt2(GPT2BPETokenizer)
        case sentencePiece(SentencePieceTokenizer)
        case debugIdentity

        func encode(_ text: String) -> [Int] {
            switch self {
            case let .gpt2(tokenizer):
                return tokenizer.encode(text)
            case let .sentencePiece(tokenizer):
                return tokenizer.encode(text)
            case .debugIdentity:
                return text
                    .split(whereSeparator: \.isWhitespace)
                    .compactMap { Int($0) }
            }
        }

        func decode(_ tokens: [Int]) -> String {
            switch self {
            case let .gpt2(tokenizer):
                return tokenizer.decode(tokens)
            case let .sentencePiece(tokenizer):
                return tokenizer.decode(tokens)
            case .debugIdentity:
                return tokens.map(String.init).joined(separator: " ")
            }
        }
    }

    struct CompiledLayer: ~Copyable {
        let attentionKernel: ANEKernel
        let attentionOutputSurface: IOSurfaceRef
        let ffnKernel: ANEKernel
        let outputSurface: IOSurfaceRef

        init(
            attentionKernel: consuming ANEKernel,
            attentionOutputSurface: IOSurfaceRef,
            ffnKernel: consuming ANEKernel,
            outputSurface: IOSurfaceRef
        ) {
            self.attentionKernel = attentionKernel
            self.attentionOutputSurface = attentionOutputSurface
            self.ffnKernel = ffnKernel
            self.outputSurface = outputSurface
        }
    }

    struct CompiledHead: ~Copyable {
        let kernel: ANEKernel
        let inputSurface: IOSurfaceRef
        let outputSurface: IOSurfaceRef

        init(kernel: consuming ANEKernel, inputSurface: IOSurfaceRef, outputSurface: IOSurfaceRef) {
            self.kernel = kernel
            self.inputSurface = inputSurface
            self.outputSurface = outputSurface
        }
    }

    struct CompiledClassifier: ~Copyable {
        let kernel: ANEKernel
        let inputSurface: IOSurfaceRef
        let outputSurface: IOSurfaceRef
        let maxValueSurface: IOSurfaceRef?

        init(
            kernel: consuming ANEKernel,
            inputSurface: IOSurfaceRef,
            outputSurface: IOSurfaceRef,
            maxValueSurface: IOSurfaceRef?
        ) {
            self.kernel = kernel
            self.inputSurface = inputSurface
            self.outputSurface = outputSurface
            self.maxValueSurface = maxValueSurface
        }
    }

    private struct SpeculativeRuntimeKey: Hashable {
        let draftLayerCount: Int
        let maxSeq: Int
    }

    private final class CachedSpeculativeRuntimePair {
        let key: SpeculativeRuntimeKey
        var draftRuntime: HybridLayerRangeRuntime
        var verifierRuntime: HybridLayerRangeRuntime

        init(
            key: SpeculativeRuntimeKey,
            config: MultiModelConfig,
            weightDirURL: URL,
            assets: GPT2TopLevelAssets
        ) throws {
            self.key = key
            self.draftRuntime = try HybridLayerRangeRuntime(
                config: config,
                weightDirURL: weightDirURL,
                assets: assets,
                layerRange: 0..<key.draftLayerCount,
                maxSeq: key.maxSeq
            )
            self.verifierRuntime = try HybridLayerRangeRuntime(
                config: config,
                weightDirURL: weightDirURL,
                assets: assets,
                layerRange: key.draftLayerCount..<config.nLayer,
                maxSeq: key.maxSeq
            )
        }

        func resetAll(dim: Int) {
            draftRuntime.reset(dim: dim)
            verifierRuntime.reset(dim: dim)
        }
    }

    private struct HybridRuntimeCheckpoint: Sendable {
        let step: Int
    }

    private struct HybridLayerRangeRuntime: ~Copyable {
        let layerRange: Range<Int>
        let maxSeq: Int
        let laneSpatial: Int
        let headSpatial: Int
        let layers: LayerStorage<HybridDecodeKernelSet>
        let surfaceHandles: [HybridDecodeSurfaceHandles]
        let greedyNorm: LayerStorage<CompiledHead>
        let greedyClassifier: LayerStorage<CompiledClassifier>
        let checkpointSurface: IOSurfaceRef
        let zeroSlice: TensorBuffer
        let preferCPUDecodeAttention: Bool
        var decodeState: DecodeState

        init(
            config: MultiModelConfig,
            weightDirURL: URL,
            assets: GPT2TopLevelAssets,
            layerRange: Range<Int>,
            maxSeq: Int
        ) throws {
            precondition(!layerRange.isEmpty)

            let layers = try RealModelInferenceEngine.compileHybridLayers(
                config: config,
                weightDirURL: weightDirURL,
                sourceLayerRange: layerRange,
                maxSeq: maxSeq
            )

            var surfaceHandles: [HybridDecodeSurfaceHandles] = []
            surfaceHandles.reserveCapacity(layers.count)
            for localLayerIndex in 0..<layers.count {
                do {
                    surfaceHandles.append(
                        try HybridDecodeSurfaceHandles(
                            kernels: layers[localLayerIndex],
                            logicalMaxSeq: maxSeq
                        )
                    )
                } catch {
                    let sourceLayerIndex = layerRange.lowerBound + localLayerIndex
                    throw RealModelInferenceError.runtimeFailure(
                        "Hybrid speculative surfaces unavailable for layer \(sourceLayerIndex): \(error)"
                    )
                }
            }

            if layers.count > 1,
               RealModelInferenceEngine.usesHybridLayerInputRebinding(
                   architecture: config.architecture,
                   environment: ProcessInfo.processInfo.environment
               ) {
                for localLayerIndex in 1..<layers.count {
                    do {
                        try layers[localLayerIndex].decodeQKVOnly.rebindInput(
                            at: 0,
                            to: surfaceHandles[localLayerIndex - 1].ffnOut
                        )
                    } catch {
                        let sourceLayerIndex = layerRange.lowerBound + localLayerIndex
                        throw RealModelInferenceError.runtimeFailure(
                            "Hybrid speculative chaining unavailable for layer \(sourceLayerIndex): \(error)"
                        )
                    }
                }
            }

            let headSpatial = RealModelInferenceEngine.incrementalHeadSpatial(channels: config.dModel)
            let greedyNorm = try LayerStorage<CompiledHead>(count: 1, throwingInitializer: { _ in
                try RealModelInferenceEngine.compileHead(
                    config: config,
                    weightDirURL: weightDirURL,
                    assets: assets,
                    spatial: headSpatial,
                    inputDType: .fp16,
                    outputDType: .fp16
                )
            })
            let greedyClassifier = try LayerStorage<CompiledClassifier>(count: 1, throwingInitializer: { _ in
                try RealModelInferenceEngine.compileClassifier(
                    config: config,
                    assets: assets,
                    spatial: headSpatial
                )
            })
            try greedyClassifier[0].kernel.rebindInput(
                at: 0,
                to: greedyNorm[0].outputSurface
            )
            if let finalSurface = surfaceHandles.last?.ffnOut {
                try greedyNorm[0].kernel.rebindInput(at: 0, to: finalSurface)
                try greedyClassifier[0].kernel.rebindInput(
                    at: 0,
                    to: greedyNorm[0].outputSurface
                )
            }

            guard let checkpointSurface = ane_interop_create_surface(config.dModel * surfaceHandles[0].laneSpatial * 2) else {
                throw RealModelInferenceError.runtimeFailure("Hybrid speculative checkpoint surface allocation failed")
            }

            var decodeState = try DecodeState(maxSeq: maxSeq)
            ForwardPass.initializeHybridDecodeCaches(
                surfaceHandles: surfaceHandles,
                dim: config.dModel
            )
            decodeState.reset()
            let zeroSlice = TensorBuffer(count: config.dModel, zeroed: true)

            self.layerRange = layerRange
            self.maxSeq = maxSeq
            self.laneSpatial = surfaceHandles[0].laneSpatial
            self.headSpatial = headSpatial
            self.layers = layers
            self.surfaceHandles = surfaceHandles
            self.greedyNorm = greedyNorm
            self.greedyClassifier = greedyClassifier
            self.checkpointSurface = checkpointSurface
            self.zeroSlice = zeroSlice
            self.preferCPUDecodeAttention = RealModelInferenceEngine.prefersCPUDecodeAttention(
                config: config,
                environment: ProcessInfo.processInfo.environment
            )
            self.decodeState = decodeState
        }

        var finalSurface: IOSurfaceRef {
            surfaceHandles[surfaceHandles.count - 1].ffnOut
        }

        var step: Int {
            decodeState.visibleTokenCount
        }

        mutating func reset(dim: Int) {
            ForwardPass.initializeHybridDecodeCaches(
                surfaceHandles: surfaceHandles,
                dim: dim
            )
            decodeState.reset()
        }

        mutating func captureCheckpoint(dim: Int) throws -> HybridRuntimeCheckpoint {
            try RealModelInferenceEngine.copyFullFP16Surface(
                dst: checkpointSurface,
                src: finalSurface,
                channels: dim,
                spatial: laneSpatial
            )
            return HybridRuntimeCheckpoint(step: step)
        }

        mutating func rollback(
            to checkpoint: HybridRuntimeCheckpoint,
            mutatedTokenCount: Int,
            dim: Int
        ) throws {
            decodeState = try DecodeState(maxSeq: maxSeq, step: checkpoint.step)
            try RealModelInferenceEngine.copyFullFP16Surface(
                dst: finalSurface,
                src: checkpointSurface,
                channels: dim,
                spatial: laneSpatial
            )
            guard mutatedTokenCount > 0 else { return }
            for handles in surfaceHandles {
                for offset in 0..<mutatedTokenCount {
                    let spatialIndex = checkpoint.step + offset
                    guard spatialIndex < maxSeq else { continue }
                    try zeroSlice.withUnsafeBufferPointer { zeroBuffer in
                        try SurfaceIO.writeFP16SpatialSlice(
                            to: handles.kCacheFull,
                            channelOffset: 0,
                            spatialIndex: spatialIndex,
                            spatial: maxSeq,
                            data: zeroBuffer,
                            channels: dim
                        )
                        try SurfaceIO.writeFP16SpatialSlice(
                            to: handles.vCacheFull,
                            channelOffset: 0,
                            spatialIndex: spatialIndex,
                            spatial: maxSeq,
                            data: zeroBuffer,
                            channels: dim
                        )
                    }
                }
            }
        }

        mutating func copyState(
            from other: borrowing HybridLayerRangeRuntime,
            dim: Int
        ) throws {
            precondition(layerRange == other.layerRange)
            precondition(maxSeq == other.maxSeq)
            precondition(laneSpatial == other.laneSpatial)

            decodeState = try DecodeState(maxSeq: maxSeq, step: other.step)
            for index in surfaceHandles.indices {
                try RealModelInferenceEngine.copyFullFP16Surface(
                    dst: surfaceHandles[index].kCacheFull,
                    src: other.surfaceHandles[index].kCacheFull,
                    channels: dim,
                    spatial: maxSeq
                )
                try RealModelInferenceEngine.copyFullFP16Surface(
                    dst: surfaceHandles[index].vCacheFull,
                    src: other.surfaceHandles[index].vCacheFull,
                    channels: dim,
                    spatial: maxSeq
                )
            }
            try RealModelInferenceEngine.copyFullFP16Surface(
                dst: finalSurface,
                src: other.finalSurface,
                channels: dim,
                spatial: laneSpatial
            )
        }

        mutating func selectGreedyToken(vocab: Int) throws -> TokenID {
            try RealModelInferenceEngine.evaluateGreedyClassifier(
                norm: greedyNorm[0],
                classifier: greedyClassifier[0],
                headSpatial: headSpatial,
                vocab: vocab
            )
        }

        mutating func advanceFromBuffer(
            _ inputBuffer: borrowing TensorBuffer,
            metalAttention: MetalAttentionKernel,
            dim: Int
        ) throws {
            var timings = HybridDecodeTimingBreakdown()
            try ForwardPass.runHybridDecodeTimed(
                xCur: inputBuffer,
                kernels: layers,
                surfaceHandles: surfaceHandles,
                metalAttention: metalAttention,
                decodeState: &decodeState,
                dim: dim,
                preferCPUDecodeAttention: preferCPUDecodeAttention,
                readFinalOutputIntoXCur: false,
                timings: &timings
            )
        }

        mutating func advanceFromSurface(
            _ sourceSurface: IOSurfaceRef,
            metalAttention: MetalAttentionKernel,
            dim: Int
        ) throws {
            let firstHandles = surfaceHandles[0]
            try RealModelInferenceEngine.copyFullFP16Surface(
                dst: firstHandles.qkvIn,
                src: sourceSurface,
                channels: dim,
                spatial: laneSpatial
            )
            var timings = HybridDecodeTimingBreakdown()
            try ForwardPass.runHybridDecodeTimedFromPreparedInput(
                kernels: layers,
                surfaceHandles: surfaceHandles,
                metalAttention: metalAttention,
                decodeState: &decodeState,
                dim: dim,
                preferCPUDecodeAttention: preferCPUDecodeAttention,
                timings: &timings
            )
        }
    }

    private static let gpt2EOSToken: TokenID = 50_256
    private static let speculativeRuntimeCacheLimit = 4

    private let config: MultiModelConfig
    private let weightDirURL: URL
    private let tokenizer: LoadedTokenizer
    private let assets: TopLevelAssets

    private var gpt2Assets: GPT2TopLevelAssets {
        guard case let .gpt2(a) = assets else {
            preconditionFailure("Attempted to access GPT-2 assets on a non-GPT-2 model")
        }
        return a
    }

    private var llamaAssets: LlamaTopLevelAssets {
        guard case let .llama(a) = assets else {
            preconditionFailure("Attempted to access Llama assets on a non-Llama model")
        }
        return a
    }

    private func hybridGreedyHeadMode(
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> HybridGreedyHeadMode {
        let hasFactoredOutputHead: Bool
        if config.architecture == .llama {
            hasFactoredOutputHead = llamaAssets.factoredOutputHead != nil
        } else {
            hasFactoredOutputHead = false
        }
        return Self.hybridGreedyHeadMode(
            config: config,
            hasFactoredOutputHead: hasFactoredOutputHead,
            environment: environment
        )
    }

    private var lmHeadWeights: [Float] {
        switch assets {
        case let .gpt2(a):
            a.lmHead
        case let .llama(a):
            a.lmHead
        }
    }

    private var compiledBucket: Int
    private var compiledLayers: LayerStorage<CompiledLayer>
    private var firstLayerInputSurface: IOSurfaceRef?
    private var compiledHead: LayerStorage<CompiledHead>
    private var compiledHybridBucket: Int
    private var compiledHybridLayers: LayerStorage<HybridDecodeKernelSet>
    private var compiledHybridSurfaceHandles: [HybridDecodeSurfaceHandles]
    private var compiledHybridLlamaQKNormWeights: [LlamaQKNormWeights?]
    private var compiledHybridHead: LayerStorage<CompiledHead>
    private var compiledHybridHeadSpatial: Int
    private var compiledHybridGreedyNorm: LayerStorage<CompiledHead>
    private var compiledHybridGreedyClassifier: LayerStorage<CompiledClassifier>
    private var compiledHybridGreedySpatial: Int
    private var hybridMetalAttention: MetalAttentionKernel?
    private var speculativeRuntimeCache: [SpeculativeRuntimeKey: CachedSpeculativeRuntimePair]
    private var speculativeRuntimeCacheOrder: [SpeculativeRuntimeKey]
    private let classifierBlockMaxNorms: [Float]
    private var classifierLogitsScratch: [Float]
    private let classifierStrategy: ClassifierStrategy
    private var cachedExactCPULlamaWeights: CachedExactCPULlamaWeights?

    private init(
        config: MultiModelConfig,
        weightDirURL: URL,
        tokenizer: LoadedTokenizer,
        assets: TopLevelAssets,
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) {
        let lmHead: [Float]
        let hasExactFloat32LMHead: Bool
        switch assets {
        case let .gpt2(a):
            lmHead = a.lmHead
            hasExactFloat32LMHead = true
        case let .llama(a):
            lmHead = a.lmHead
            hasExactFloat32LMHead = a.lmHeadHasExactFloat32Sidecar
        }
        let classifierBlockMaxNorms = lmHead.withUnsafeBufferPointer { weightBuffer in
            Self.precomputeClassifierBlockMaxNorms(
                classifier: weightBuffer.baseAddress!,
                vocabSize: config.vocab,
                dim: config.dModel,
                blockSize: Self.classifierArgmaxBlockSize
            )
        }
        self.config = config
        self.weightDirURL = weightDirURL
        self.tokenizer = tokenizer
        self.assets = assets
        self.compiledBucket = 0
        self.compiledLayers = Self.emptyStorage(CompiledLayer.self)
        self.firstLayerInputSurface = nil
        self.compiledHead = Self.emptyStorage(CompiledHead.self)
        self.compiledHybridBucket = 0
        self.compiledHybridLayers = Self.emptyStorage(HybridDecodeKernelSet.self)
        self.compiledHybridSurfaceHandles = []
        self.compiledHybridLlamaQKNormWeights = []
        self.compiledHybridHead = Self.emptyStorage(CompiledHead.self)
        self.compiledHybridHeadSpatial = 0
        self.compiledHybridGreedyNorm = Self.emptyStorage(CompiledHead.self)
        self.compiledHybridGreedyClassifier = Self.emptyStorage(CompiledClassifier.self)
        self.compiledHybridGreedySpatial = 0
        self.hybridMetalAttention = nil
        self.speculativeRuntimeCache = [:]
        self.speculativeRuntimeCacheOrder = []
        self.classifierBlockMaxNorms = classifierBlockMaxNorms
        self.classifierLogitsScratch = [Float](
            repeating: 0,
            count: min(Self.classifierArgmaxBlockSize, config.vocab)
        )
        self.classifierStrategy = Self.resolveClassifierStrategy(
            config: config,
            hasExactFloat32LMHead: hasExactFloat32LMHead,
            environment: environment
        )
        self.cachedExactCPULlamaWeights = nil
    }

    public static func build(
        config: MultiModelConfig,
        weightDir: String,
        tokenizerDir: String
    ) throws -> RealModelInferenceEngine {
        try validateConfig(config)

        let weightDirURL = URL(fileURLWithPath: weightDir, isDirectory: true)
        let tokenizerDirURL = URL(fileURLWithPath: tokenizerDir, isDirectory: true)
        try validateDirectory(weightDirURL)
        try validateDirectory(tokenizerDirURL)
        try validateMetadataIfPresent(config: config, weightDirURL: weightDirURL)

        let tokenizer = try loadTokenizer(config: config, tokenizerDirURL: tokenizerDirURL)

        let topLevelAssets = try loadTestingTopLevelAssets(
            config: config,
            weightDir: weightDir,
            weightDirURL: weightDirURL
        )
        return RealModelInferenceEngine(
            config: config,
            weightDirURL: weightDirURL,
            tokenizer: tokenizer,
            assets: topLevelAssets
        )
    }

    public mutating func generate(
        prompt: String,
        maxTokens: Int = 128,
        temperature: Float = 0.0,
        onStep: ((GenerationStep) -> Void)? = nil
    ) throws -> GenerationResult {
        guard maxTokens >= 0 else {
            throw RealModelInferenceError.invalidGenerationParameters("maxTokens must be >= 0")
        }
        guard temperature.isFinite, temperature >= 0 else {
            throw RealModelInferenceError.invalidGenerationParameters("temperature must be finite and >= 0")
        }

        let promptTokens = try encodePrompt(prompt)
        guard promptTokens.count < config.maxSeq else {
            throw RealModelInferenceError.invalidPrompt(
                "Prompt token count \(promptTokens.count) exceeds model context \(config.maxSeq - 1)"
            )
        }

        let remainingContext = config.maxSeq - promptTokens.count
        let effectiveMaxTokens = min(maxTokens, max(remainingContext, 0))
        if effectiveMaxTokens == 0 {
            let text = tokenizer.decode(promptTokens.map(Int.init))
            return GenerationResult(
                text: text,
                tokens: [],
                promptTokens: promptTokens,
                tokensPerSecond: 0,
                compileTimeMs: 0,
                firstTokenLatencyMs: 0
            )
        }

        let targetTokenCount = min(config.maxSeq, promptTokens.count + effectiveMaxTokens)
        let bucket = try Self.compileBucket(
            for: targetTokenCount,
            channels: config.dModel,
            maxSeq: config.maxSeq
        )
        let environment = ProcessInfo.processInfo.environment

        if config.architecture == .llama {
            if temperature == 0,
               let draft = try Self.resolveExactTwoTokenDraft(
                   config: config,
                   weightDirURL: weightDirURL,
                   environment: environment
               ) {
                return try generateIncrementalExactTwoTokenDraftLlama(
                    promptTokens: promptTokens,
                    effectiveMaxTokens: effectiveMaxTokens,
                    compileTimeMs: 0,
                    draft: draft,
                    onStep: onStep
                )
            }
            switch Self.llamaGenerationPath(
                config: config,
                environment: environment
            ) {
            case .exactCPU:
                return try generateIncrementalExactCPULlama(
                    promptTokens: promptTokens,
                    effectiveMaxTokens: effectiveMaxTokens,
                    temperature: temperature,
                    compileTimeMs: 0,
                    maxSeq: bucket,
                    onStep: onStep
                )
            case .hybrid:
                let compileStart = DispatchTime.now().uptimeNanoseconds
                let compileDidRun = try ensureHybridCompiledLlama(bucket: bucket)
                guard let metalAttention = hybridMetalAttention else {
                    throw RealModelInferenceError.runtimeFailure("Hybrid Metal attention unavailable for llama")
                }
                let compileEnd = DispatchTime.now().uptimeNanoseconds
                let compileTimeMs = compileDidRun ? Self.milliseconds(from: compileEnd - compileStart) : 0
                return try generateIncrementalHybridLlama(
                    promptTokens: promptTokens,
                    effectiveMaxTokens: effectiveMaxTokens,
                    temperature: temperature,
                    compileTimeMs: compileTimeMs,
                    maxSeq: bucket,
                    metalAttention: metalAttention,
                    onStep: onStep
                )
            }
        }

        let compileStart = DispatchTime.now().uptimeNanoseconds
        var compileDidRun = false
        var useHybridFastPath = false

        do {
            let hybridDidRun = try ensureHybridCompiled(bucket: bucket)
            compileDidRun = compileDidRun || hybridDidRun
            useHybridFastPath =
                compiledHybridLayers.count == config.nLayer &&
                compiledHybridSurfaceHandles.count == config.nLayer &&
                compiledHybridHead.count == 1 &&
                hybridMetalAttention != nil
        } catch {
            useHybridFastPath = false
        }

        if useHybridFastPath, let metalAttention = hybridMetalAttention {
            let compileEnd = DispatchTime.now().uptimeNanoseconds
            let compileTimeMs = compileDidRun ? Self.milliseconds(from: compileEnd - compileStart) : 0
            if let speculativeDraftLayerCount = Self.resolvedSpeculativeDraftLayerCount(
                config: config,
                temperature: temperature
            ) {
                var speculativeAttemptCompileTimeMs = 0.0
                do {
                    let (cachedRuntimePair, speculativeCompileTimeMs) = try cachedSpeculativeRuntimePair(
                        draftLayerCount: speculativeDraftLayerCount,
                        maxSeq: bucket
                    )
                    speculativeAttemptCompileTimeMs = speculativeCompileTimeMs
                    return try generateIncrementalHybridSpeculative(
                        promptTokens: promptTokens,
                        effectiveMaxTokens: effectiveMaxTokens,
                        compileTimeMs: compileTimeMs + speculativeCompileTimeMs,
                        metalAttention: metalAttention,
                        cachedRuntimePair: cachedRuntimePair,
                        onStep: onStep
                    )
                } catch {
                    if ProcessInfo.processInfo.environment["ESPRESSO_REALMODEL_DISABLE_HYBRID_FALLBACK"] == "1" {
                        throw RealModelInferenceError.runtimeFailure("Hybrid speculative fast path failed: \(error)")
                    }
                    fputs(
                        "[RealModelInference] Hybrid speculative fast path failed; falling back to non-speculative hybrid decode: \(String(describing: error))\n",
                        stderr
                    )
                    let fallbackCompileTimeMs = compileTimeMs + speculativeAttemptCompileTimeMs
                    return try generateIncrementalHybrid(
                        promptTokens: promptTokens,
                        effectiveMaxTokens: effectiveMaxTokens,
                        temperature: temperature,
                        compileTimeMs: fallbackCompileTimeMs,
                        maxSeq: bucket,
                        metalAttention: metalAttention,
                        onStep: onStep
                    )
                }
            }
            do {
                return try generateIncrementalHybrid(
                    promptTokens: promptTokens,
                    effectiveMaxTokens: effectiveMaxTokens,
                    temperature: temperature,
                    compileTimeMs: compileTimeMs,
                    maxSeq: bucket,
                    metalAttention: metalAttention,
                    onStep: onStep
                )
            } catch {
                if ProcessInfo.processInfo.environment["ESPRESSO_REALMODEL_DISABLE_HYBRID_FALLBACK"] == "1" {
                    throw RealModelInferenceError.runtimeFailure("Hybrid fast path failed: \(error)")
                }
                useHybridFastPath = false
            }
        }

        let baselineDidRun = try ensureCompiled(bucket: bucket)
        compileDidRun = compileDidRun || baselineDidRun
        let compileEnd = DispatchTime.now().uptimeNanoseconds
        let compileTimeMs = compileDidRun ? Self.milliseconds(from: compileEnd - compileStart) : 0

        guard let inputSurface = firstLayerInputSurface, compiledLayers.count == config.nLayer, compiledHead.count == 1 else {
            throw RealModelInferenceError.runtimeFailure("Compiled ANE surfaces are unavailable")
        }

        var allTokens = promptTokens
        var generatedTokens: [TokenID] = []
        var tokenLatenciesMs: [Double] = []
        generatedTokens.reserveCapacity(effectiveMaxTokens)
        tokenLatenciesMs.reserveCapacity(effectiveMaxTokens)

        let generationStart = DispatchTime.now().uptimeNanoseconds
        var firstTokenLatencyMs = 0.0
        var firstTokenRecorded = false
        var rng = SystemRandomNumberGenerator()
        let activeBucket = compiledBucket

        for _ in 0..<effectiveMaxTokens {
            let stepStart = DispatchTime.now().uptimeNanoseconds
            let sequenceLength = allTokens.count
            let activation = composeEmbeddingInput(tokens: allTokens, spatial: activeBucket)
            try activation.withUnsafeBufferPointer { buffer in
                try Self.writeFP32(to: inputSurface, data: buffer)
            }

            for layerIndex in 0..<compiledLayers.count {
                do {
                    try compiledLayers[layerIndex].attentionKernel.eval()
                    try compiledLayers[layerIndex].ffnKernel.eval()
                } catch {
                    throw RealModelInferenceError.runtimeFailure("Layer \(layerIndex) eval failed: \(error)")
                }
            }

            do {
                try compiledHead[0].kernel.eval()
            } catch {
                throw RealModelInferenceError.runtimeFailure("Final norm eval failed: \(error)")
            }

            var normalized = [Float](repeating: 0, count: config.dModel * activeBucket)
            try normalized.withUnsafeMutableBufferPointer { buffer in
                try Self.readFP32(from: compiledHead[0].outputSurface, into: buffer)
            }
            let lastHidden = Self.extractSpatialSlice(
                from: normalized,
                channels: config.dModel,
                spatial: activeBucket,
                spatialIndex: sequenceLength - 1
            )
            let nextToken = selectTokenFromNormalizedHidden(
                lastHidden,
                temperature: temperature,
                using: &rng
            )

            if !firstTokenRecorded {
                firstTokenLatencyMs = Self.milliseconds(from: DispatchTime.now().uptimeNanoseconds - generationStart)
                firstTokenRecorded = true
            }

            if nextToken == Self.gpt2EOSToken {
                break
            }

            generatedTokens.append(nextToken)
            allTokens.append(nextToken)
            let elapsedMs = Self.milliseconds(from: DispatchTime.now().uptimeNanoseconds - generationStart)
            let tokenLatencyMs = Self.milliseconds(from: DispatchTime.now().uptimeNanoseconds - stepStart)
            tokenLatenciesMs.append(tokenLatencyMs)
            let tokensPerSecond = Double(generatedTokens.count) / max(elapsedMs / 1_000, 1e-9)
            onStep?(
                GenerationStep(
                    token: nextToken,
                    generatedTokens: generatedTokens,
                    text: tokenizer.decode(allTokens.map(Int.init)),
                    tokenLatencyMs: tokenLatencyMs,
                    elapsedMs: elapsedMs,
                    firstTokenLatencyMs: firstTokenLatencyMs,
                    tokensPerSecond: tokensPerSecond
                )
            )
            if allTokens.count >= config.maxSeq {
                break
            }
        }

        let generationEnd = DispatchTime.now().uptimeNanoseconds
        let generationTimeMs = Self.milliseconds(from: generationEnd - generationStart)
        let tokensPerSecond = generatedTokens.isEmpty
            ? 0
            : Double(generatedTokens.count) / max(generationTimeMs / 1_000, 1e-9)

        return GenerationResult(
            text: tokenizer.decode(allTokens.map(Int.init)),
            tokens: generatedTokens,
            promptTokens: promptTokens,
            tokenLatenciesMs: tokenLatenciesMs,
            tokensPerSecond: tokensPerSecond,
            compileTimeMs: compileTimeMs,
            firstTokenLatencyMs: firstTokenLatencyMs,
            exactHeadBackend: classifierStrategy.exactHeadBackendLabel,
            cachedBindingsEnabled: false
        )
    }

    public static func generateNextTokenForTesting(
        config: MultiModelConfig,
        weightDir: String,
        promptTokens: [TokenID]
    ) throws -> TokenID {
        guard !promptTokens.isEmpty else {
            throw RealModelInferenceError.invalidGenerationParameters("Testing prompt token list must not be empty")
        }

        try validateConfig(config)
        let weightDirURL = URL(fileURLWithPath: weightDir, isDirectory: true)
        try validateDirectory(weightDirURL)
        try validateMetadataIfPresent(config: config, weightDirURL: weightDirURL)

        let topLevelAssets: TopLevelAssets
        switch config.architecture {
        case .gpt2:
            let topLevelPaths = try resolveTopLevelWeightPaths(config: config, weightDir: weightDir)
            let tokenEmbedding = try loadWeightTablePreferringFloat32Sidecar(
                at: topLevelPaths.tokenEmbedding,
                expectedCount: config.vocab * config.dModel
            )
            let positionEmbedding = try loadWeightTable(
                at: topLevelPaths.positionEmbedding,
                expectedCount: config.maxSeq * config.dModel
            )
            let finalNormGamma = try loadWeightTablePreferringFloat32Sidecar(
                at: topLevelPaths.finalNormGamma,
                expectedCount: config.dModel
            )
            let finalNormBeta = try loadWeightTable(
                at: topLevelPaths.finalNormBeta,
                expectedCount: config.dModel
            )
            let lmHead = try loadWeightTablePreferringFloat32Sidecar(
                at: topLevelPaths.lmHead,
                expectedCount: config.vocab * config.dModel
            )
            topLevelAssets = .gpt2(GPT2TopLevelAssets(
                tokenEmbedding: tokenEmbedding,
                positionEmbedding: positionEmbedding,
                finalNormGamma: finalNormGamma,
                finalNormBeta: finalNormBeta,
                lmHead: lmHead,
                finalNormGammaPath: topLevelPaths.finalNormGamma,
                finalNormBetaPath: topLevelPaths.finalNormBeta,
                finalNormGammaCompilePath: compileBlobPath(actualPath: topLevelPaths.finalNormGamma, rootDir: weightDirURL),
                finalNormBetaCompilePath: compileBlobPath(actualPath: topLevelPaths.finalNormBeta, rootDir: weightDirURL),
                finalNormGammaData: WeightBlob.build(from: finalNormGamma, rows: 1, cols: finalNormGamma.count),
                finalNormBetaData: WeightBlob.build(from: finalNormBeta, rows: 1, cols: finalNormBeta.count)
            ))
        case .llama:
            let topLevelPaths = try resolveLlamaTopLevelWeightPaths(config: config, weightDir: weightDir)
            topLevelAssets = .llama(try loadLlamaTopLevelAssets(
                config: config,
                topLevelPaths: topLevelPaths,
                weightDirURL: weightDirURL
            ))
        }

        var engine = RealModelInferenceEngine(
            config: config,
            weightDirURL: weightDirURL,
            tokenizer: .debugIdentity,
            assets: topLevelAssets
        )

        let targetTokenCount = min(config.maxSeq, promptTokens.count + 1)
        let bucket = try compileBucket(
            for: targetTokenCount,
            channels: config.dModel,
            maxSeq: config.maxSeq
        )

        guard config.architecture == .llama else {
            throw RealModelInferenceError.unsupportedArchitecture(
                "generateNextTokenForTesting currently supports llama-family artifacts only"
            )
        }

        let result: GenerationResult
        switch llamaGenerationPath(
            config: config,
            environment: ProcessInfo.processInfo.environment
        ) {
        case .exactCPU:
            result = try engine.generateIncrementalExactCPULlama(
                promptTokens: promptTokens,
                effectiveMaxTokens: 1,
                temperature: 0,
                compileTimeMs: 0,
                maxSeq: bucket,
                onStep: nil
            )
        case .hybrid:
            let compileStart = DispatchTime.now().uptimeNanoseconds
            let compileDidRun = try engine.ensureHybridCompiledLlama(bucket: bucket)
            guard let metalAttention = engine.hybridMetalAttention else {
                throw RealModelInferenceError.runtimeFailure("Hybrid Metal attention unavailable for llama testing helper")
            }
            let compileEnd = DispatchTime.now().uptimeNanoseconds
            let compileTimeMs = compileDidRun ? milliseconds(from: compileEnd - compileStart) : 0
            result = try engine.generateIncrementalHybridLlama(
                promptTokens: promptTokens,
                effectiveMaxTokens: 1,
                temperature: 0,
                compileTimeMs: compileTimeMs,
                maxSeq: bucket,
                metalAttention: metalAttention,
                onStep: nil
            )
        }
        guard let token = result.tokens.first else {
            throw RealModelInferenceError.runtimeFailure("Testing helper did not emit a next token")
        }
        return token
    }

    public static func generateNextTokenExactCPUForTesting(
        config: MultiModelConfig,
        weightDir: String,
        promptTokens: [TokenID]
    ) throws -> TokenID {
        guard !promptTokens.isEmpty else {
            throw RealModelInferenceError.invalidGenerationParameters("Testing prompt token list must not be empty")
        }

        try validateConfig(config)
        let weightDirURL = URL(fileURLWithPath: weightDir, isDirectory: true)
        try validateDirectory(weightDirURL)
        try validateMetadataIfPresent(config: config, weightDirURL: weightDirURL)

        let topLevelAssets = try loadTestingTopLevelAssets(
            config: config,
            weightDir: weightDir,
            weightDirURL: weightDirURL
        )
        var engine = RealModelInferenceEngine(
            config: config,
            weightDirURL: weightDirURL,
            tokenizer: .debugIdentity,
            assets: topLevelAssets
        )
        let result = try engine.generateIncrementalExactCPULlama(
            promptTokens: promptTokens,
            effectiveMaxTokens: 1,
            temperature: 0,
            compileTimeMs: 0,
            maxSeq: config.maxSeq,
            onStep: nil
        )
        guard let token = result.tokens.first else {
            throw RealModelInferenceError.runtimeFailure("Exact CPU testing helper did not emit a next token")
        }
        return token
    }

    public static func generateTokensExactCPUForTesting(
        config: MultiModelConfig,
        weightDir: String,
        promptTokens: [TokenID],
        maxTokens: Int
    ) throws -> [TokenID] {
        guard !promptTokens.isEmpty else {
            throw RealModelInferenceError.invalidGenerationParameters("Testing prompt token list must not be empty")
        }
        guard maxTokens > 0 else {
            throw RealModelInferenceError.invalidGenerationParameters("Testing max token count must be positive")
        }

        try validateConfig(config)
        let weightDirURL = URL(fileURLWithPath: weightDir, isDirectory: true)
        try validateDirectory(weightDirURL)
        try validateMetadataIfPresent(config: config, weightDirURL: weightDirURL)

        let topLevelAssets = try loadTestingTopLevelAssets(
            config: config,
            weightDir: weightDir,
            weightDirURL: weightDirURL
        )
        var engine = RealModelInferenceEngine(
            config: config,
            weightDirURL: weightDirURL,
            tokenizer: .debugIdentity,
            assets: topLevelAssets
        )
        let result = try engine.generateIncrementalExactCPULlama(
            promptTokens: promptTokens,
            effectiveMaxTokens: maxTokens,
            temperature: 0,
            compileTimeMs: 0,
            maxSeq: config.maxSeq,
            onStep: nil
        )
        return result.tokens
    }

    private static func loadTestingTopLevelAssets(
        config: MultiModelConfig,
        weightDir: String,
        weightDirURL: URL
    ) throws -> TopLevelAssets {
        switch config.architecture {
        case .gpt2:
            let topLevelPaths = try resolveTopLevelWeightPaths(config: config, weightDir: weightDir)
            let tokenEmbedding = try loadWeightTablePreferringFloat32Sidecar(
                at: topLevelPaths.tokenEmbedding,
                expectedCount: config.vocab * config.dModel
            )
            let positionEmbedding = try loadWeightTable(
                at: topLevelPaths.positionEmbedding,
                expectedCount: config.maxSeq * config.dModel
            )
            let finalNormGamma = try loadWeightTablePreferringFloat32Sidecar(
                at: topLevelPaths.finalNormGamma,
                expectedCount: config.dModel
            )
            let finalNormBeta = try loadWeightTable(
                at: topLevelPaths.finalNormBeta,
                expectedCount: config.dModel
            )
            let lmHead = try loadWeightTablePreferringFloat32Sidecar(
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
                finalNormGammaCompilePath: compileBlobPath(actualPath: topLevelPaths.finalNormGamma, rootDir: weightDirURL),
                finalNormBetaCompilePath: compileBlobPath(actualPath: topLevelPaths.finalNormBeta, rootDir: weightDirURL),
                finalNormGammaData: WeightBlob.build(from: finalNormGamma, rows: 1, cols: finalNormGamma.count),
                finalNormBetaData: WeightBlob.build(from: finalNormBeta, rows: 1, cols: finalNormBeta.count)
            ))
        case .llama:
            let topLevelPaths = try resolveLlamaTopLevelWeightPaths(config: config, weightDir: weightDir)
            return .llama(try loadLlamaTopLevelAssets(
                config: config,
                topLevelPaths: topLevelPaths,
                weightDirURL: weightDirURL
            ))
        }
    }

    private static func loadLlamaTopLevelAssets(
        config: MultiModelConfig,
        topLevelPaths: LlamaTopLevelWeightPaths,
        weightDirURL: URL,
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) throws -> LlamaTopLevelAssets {
        let tokenEmbedding = try loadWeightTablePreferringFloat32Sidecar(
            at: topLevelPaths.tokenEmbedding,
            expectedCount: config.vocab * config.dModel
        )
        let finalNormGamma = try loadWeightTablePreferringFloat32Sidecar(
            at: topLevelPaths.finalNormGamma,
            expectedCount: config.dModel
        )
        let lmHead = try loadWeightTablePreferringFloat32Sidecar(
            at: topLevelPaths.lmHead,
            expectedCount: config.vocab * config.dModel
        )
        let lmHeadFP16 = try loadRawFP16WeightTableIfNoExactFloat32Sidecar(
            at: topLevelPaths.lmHead,
            expectedCount: config.vocab * config.dModel
        )
        let factoredOutputHead = try loadLlamaFactoredOutputHead(
            config: config,
            weightDirURL: weightDirURL,
            environment: environment
        )
        return LlamaTopLevelAssets(
            tokenEmbedding: tokenEmbedding,
            finalNormGamma: finalNormGamma,
            lmHead: lmHead,
            lmHeadFP16: lmHeadFP16,
            lmHeadHasExactFloat32Sidecar: lmHeadFP16 == nil,
            factoredOutputHead: factoredOutputHead,
            finalNormGammaPath: topLevelPaths.finalNormGamma,
            finalNormGammaCompilePath: compileBlobPath(actualPath: topLevelPaths.finalNormGamma, rootDir: weightDirURL),
            finalNormGammaData: WeightBlob.build(from: finalNormGamma, rows: 1, cols: finalNormGamma.count)
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

        let projectionPath = try resolveBundleWeightReference(
            environment["ESPRESSO_BUNDLE_OUTPUT_HEAD_PROJECTION_REF"] ?? "cls_proj.bin",
            weightDirURL: weightDirURL
        )
        let expansionPath = try resolveBundleWeightReference(
            environment["ESPRESSO_BUNDLE_OUTPUT_HEAD_EXPANSION_REF"] ?? "cls_expand.bin",
            weightDirURL: weightDirURL
        )

        let projectionCompactCount = bottleneck * (config.dModel / groups)
        let projectionDenseCount = bottleneck * config.dModel
        let expansionCompactCount = config.vocab * (bottleneck / groups)
        let expansionDenseCount = config.vocab * bottleneck
        let projection = try loadWeightTable(
            at: projectionPath,
            allowedCounts: [projectionCompactCount, projectionDenseCount]
        )
        let expansion = try loadWeightTable(
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

    static func spatialBucket(for tokenCount: Int, maxSeq: Int) -> Int {
        let clamped = min(max(tokenCount, 1), maxSeq)
        var bucket = 1
        while bucket < clamped {
            bucket &*= 2
        }
        return min(bucket, maxSeq)
    }

    static func minimumCompileSpatial(channels: Int) -> Int {
        precondition(channels > 0)
        let bytesPerSpatial = channels * ANEDType.fp16.byteWidth
        let requiredSpatial = (minimumANEIOSurfaceBytes + bytesPerSpatial - 1) / bytesPerSpatial
        var bucket = 1
        while bucket < requiredSpatial {
            bucket &*= 2
        }
        return bucket
    }

    static func incrementalHeadSpatial(channels: Int) -> Int {
        minimumCompileSpatial(channels: channels)
    }

    static func resolvedSpeculativeDraftLayerCount(
        config: MultiModelConfig,
        temperature: Float,
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> Int? {
        guard config.architecture == .gpt2,
              temperature == 0,
              config.nLayer > 1,
              environment["ESPRESSO_ENABLE_GPT2_SPECULATIVE"] == "1" else {
            return nil
        }

        let defaultDraftLayerCount = 1
        let requestedDraftLayerCount = environment["ESPRESSO_GPT2_SPECULATIVE_DRAFT_LAYERS"].flatMap(Int.init)
            ?? defaultDraftLayerCount
        return min(max(requestedDraftLayerCount, 1), config.nLayer - 1)
    }

    static func resolveTopLevelWeightPaths(
        config: MultiModelConfig,
        weightDir: String
    ) throws -> TopLevelWeightPaths {
        let root = URL(fileURLWithPath: weightDir, isDirectory: true)
        try validateDirectory(root)

        switch config.architecture {
        case .gpt2:
            return TopLevelWeightPaths(
                tokenEmbedding: try requiredFile(
                    root: root,
                    candidates: ["embeddings/token.bin", "embeddings/token_embeddings.bin"],
                    label: "token embedding"
                ),
                positionEmbedding: try requiredFile(
                    root: root,
                    candidates: ["embeddings/position.bin", "embeddings/position_embeddings.bin"],
                    label: "position embedding"
                ),
                finalNormGamma: try requiredFile(
                    root: root,
                    candidates: ["final_norm_gamma.bin", "ln_f_gamma.bin", "rms_final.bin"],
                    label: "final norm gamma"
                ),
                finalNormBeta: try requiredFile(
                    root: root,
                    candidates: ["final_norm_beta.bin", "ln_f_beta.bin", "rms_final_beta.bin"],
                    label: "final norm beta"
                ),
                lmHead: try requiredFile(
                    root: root,
                    candidates: ["lm_head.bin", "classifier.bin"],
                    label: "lm head"
                )
            )
        case .llama:
            return TopLevelWeightPaths(
                tokenEmbedding: try requiredFile(
                    root: root,
                    candidates: ["embeddings/token.bin", "embeddings/token_embeddings.bin"],
                    label: "token embedding"
                ),
                positionEmbedding: "",
                finalNormGamma: try requiredFile(
                    root: root,
                    candidates: ["rms_final.bin", "final_norm_gamma.bin"],
                    label: "final norm gamma"
                ),
                finalNormBeta: "",
                lmHead: try requiredFile(
                    root: root,
                    candidates: ["lm_head.bin", "classifier.bin"],
                    label: "lm head"
                )
            )
        }
    }

    struct LlamaTopLevelWeightPaths: Sendable, Equatable {
        let tokenEmbedding: String
        let finalNormGamma: String
        let lmHead: String
    }

    static func resolveLlamaTopLevelWeightPaths(
        config: MultiModelConfig,
        weightDir: String
    ) throws -> LlamaTopLevelWeightPaths {
        let root = URL(fileURLWithPath: weightDir, isDirectory: true)
        try validateDirectory(root)
        return LlamaTopLevelWeightPaths(
            tokenEmbedding: try requiredFile(
                root: root,
                candidates: ["embeddings/token.bin", "embeddings/token_embeddings.bin"],
                label: "token embedding"
            ),
            finalNormGamma: try requiredFile(
                root: root,
                candidates: ["rms_final.bin", "final_norm_gamma.bin", "final_norm.bin"],
                label: "final norm gamma"
            ),
            lmHead: try requiredFile(
                root: root,
                candidates: ["lm_head.bin", "classifier.bin"],
                label: "lm head"
            )
        )
    }

    static func compileAndEvalSingleLayerForTesting(
        config: MultiModelConfig,
        weightDir: String,
        layer: Int,
        spatial: Int,
        input: [Float],
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) throws -> [Float] {
        let weightDirURL = URL(fileURLWithPath: weightDir, isDirectory: true)
        try validateDirectory(weightDirURL)
        let compiled = try compileLayer(
            layerIndex: layer,
            config: config,
            weightDirURL: weightDirURL,
            spatial: spatial,
            environment: environment
        )
        let inputSurface: IOSurfaceRef
        do {
            inputSurface = try compiled.attentionKernel.inputSurface(at: 0)
        } catch {
            throw RealModelInferenceError.runtimeFailure("Layer input surface unavailable: \(error)")
        }

        guard input.count == config.dModel * spatial else {
            throw RealModelInferenceError.invalidGenerationParameters(
                "Single-layer test input must have \(config.dModel * spatial) floats"
            )
        }

        try input.withUnsafeBufferPointer { buffer in
            try Self.writeFP32(to: inputSurface, data: buffer)
        }

        do {
            try compiled.attentionKernel.eval()
            try compiled.ffnKernel.eval()
        } catch {
            throw RealModelInferenceError.runtimeFailure("Single-layer eval failed: \(error)")
        }

        var output = [Float](repeating: 0, count: config.dModel * spatial)
        try output.withUnsafeMutableBufferPointer { buffer in
            try Self.readFP32(from: compiled.outputSurface, into: buffer)
        }
        return output
    }

    static func compileAndEvalSingleLayerAttentionForTesting(
        config: MultiModelConfig,
        weightDir: String,
        layer: Int,
        spatial: Int,
        input: [Float],
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) throws -> [Float] {
        try compileAndEvalSingleLayerAttentionOutputsForTesting(
            config: config,
            weightDir: weightDir,
            layer: layer,
            spatial: spatial,
            input: input,
            environment: environment
        ).hidden
    }

    static func compileAndEvalSingleLayerAttentionOutputsForTesting(
        config: MultiModelConfig,
        weightDir: String,
        layer: Int,
        spatial: Int,
        input: [Float],
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) throws -> AttentionTestingOutputs {
        let weightDirURL = URL(fileURLWithPath: weightDir, isDirectory: true)
        try validateDirectory(weightDirURL)
        let compiled = try compileLayer(
            layerIndex: layer,
            config: config,
            weightDirURL: weightDirURL,
            spatial: spatial,
            environment: environment
        )
        let inputSurface: IOSurfaceRef
        do {
            inputSurface = try compiled.attentionKernel.inputSurface(at: 0)
        } catch {
            throw RealModelInferenceError.runtimeFailure("Layer input surface unavailable: \(error)")
        }

        guard input.count == config.dModel * spatial else {
            throw RealModelInferenceError.invalidGenerationParameters(
                "Single-layer attention test input must have \(config.dModel * spatial) floats"
            )
        }

        try input.withUnsafeBufferPointer { buffer in
            try Self.writeFP32(to: inputSurface, data: buffer)
        }

        do {
            try compiled.attentionKernel.eval()
        } catch {
            throw RealModelInferenceError.runtimeFailure("Single-layer attention eval failed: \(error)")
        }

        var hidden = [Float](repeating: 0, count: config.dModel * spatial)
        try hidden.withUnsafeMutableBufferPointer { buffer in
            try Self.readFP32(from: compiled.attentionOutputSurface, into: buffer)
        }
        let kSurface = try compiled.attentionKernel.outputSurface(at: 1)
        let vSurface = try compiled.attentionKernel.outputSurface(at: 2)
        var kCache = [Float](repeating: 0, count: config.dModel * spatial)
        var vCache = [Float](repeating: 0, count: config.dModel * spatial)
        try kCache.withUnsafeMutableBufferPointer { buffer in
            try Self.readFP32(from: kSurface, into: buffer)
        }
        try vCache.withUnsafeMutableBufferPointer { buffer in
            try Self.readFP32(from: vSurface, into: buffer)
        }
        return AttentionTestingOutputs(hidden: hidden, kCache: kCache, vCache: vCache)
    }

    public static func probeGPT2AttentionCompilation(
        config: MultiModelConfig,
        weightDir: String,
        layer: Int = 0,
        spatials: [Int] = [64, 128, 256],
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) throws -> [GPT2AttentionCompileProbeResult] {
        guard config.architecture == .gpt2 else {
            throw RealModelInferenceError.unsupportedArchitecture(
                "GPT-2 attention compile probe supports GPT-2 architectures only."
            )
        }

        let weightDirURL = URL(fileURLWithPath: weightDir, isDirectory: true)
        try validateDirectory(weightDirURL)
        let paths = LayerWeightPaths.forLayer(layer, config: config, blobDir: weightDirURL.path)
        let deploymentTarget = milDeploymentTarget(environment: environment)
        let normKind = gpt2NormKind(environment: environment)

        return spatials.map { spatial in
            let graph = buildGPT2AttentionBlockGraph(
                layerIndex: layer,
                config: config,
                paths: paths,
                spatial: spatial,
                environment: environment
            )
            var optimized = graph
            ANEOptimizationPipeline.optimize(&optimized)
            let diagnostics = ANEValidationPass().run(on: optimized).map(\.message)
            do {
                let ioBytes = try ANEShape(channels: config.dModel, spatial: spatial).byteSize(for: .fp32)
                _ = try compileLayerBlock(
                    layerIndex: layer,
                    kind: .attention,
                    graph: graph,
                    weights: try attentionWeights(
                        config: config,
                        diskPaths: paths,
                        weightDirURL: weightDirURL,
                        spatial: spatial
                    ),
                    inputBytes: ioBytes,
                    outputBytes: [ioBytes, ioBytes, ioBytes],
                    weightDirURL: weightDirURL,
                    spatial: spatial,
                    environment: environment
                )
                return GPT2AttentionCompileProbeResult(
                    spatial: spatial,
                    deploymentTarget: deploymentTarget,
                    normKind: normKind.rawValue,
                    diagnostics: diagnostics,
                    compileSucceeded: true,
                    error: nil
                )
            } catch {
                return GPT2AttentionCompileProbeResult(
                    spatial: spatial,
                    deploymentTarget: deploymentTarget,
                    normKind: normKind.rawValue,
                    diagnostics: diagnostics,
                    compileSucceeded: false,
                    error: (error as? LocalizedError)?.errorDescription ?? String(describing: error)
                )
            }
        }
    }

    static func emitGPT2AttentionMILForTesting(
        config: MultiModelConfig,
        weightDir: String,
        layer: Int = 0,
        spatial: Int,
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) throws -> String {
        let weightDirURL = URL(fileURLWithPath: weightDir, isDirectory: true)
        try validateDirectory(weightDirURL)
        let paths = LayerWeightPaths.forLayer(layer, config: config, blobDir: weightDirURL.path)
        var graph = buildGPT2AttentionBlockGraph(
            layerIndex: layer,
            config: config,
            paths: paths,
            spatial: spatial,
            environment: environment
        )
        ANEOptimizationPipeline.optimize(&graph)
        return rewriteMILWeightPaths(
            ANECodegen.emit(graph, deploymentTarget: milDeploymentTarget(environment: environment)),
            rootDir: weightDirURL
        )
    }

    static func composeEmbeddingInputForTesting(
        config: MultiModelConfig,
        weightDir: String,
        tokens: [TokenID]
    ) throws -> [Float] {
        let weightDirURL = URL(fileURLWithPath: weightDir, isDirectory: true)
        try validateDirectory(weightDirURL)
        guard !tokens.isEmpty else {
            throw RealModelInferenceError.invalidGenerationParameters("Testing token list must not be empty")
        }
        guard tokens.count <= config.maxSeq else {
            throw RealModelInferenceError.invalidGenerationParameters(
                "Testing token count \(tokens.count) exceeds context \(config.maxSeq)"
            )
        }

        let tokenEmbedding: [Float]
        let positionEmbedding: [Float]
        switch config.architecture {
        case .gpt2:
            let topLevelPaths = try resolveTopLevelWeightPaths(config: config, weightDir: weightDir)
            tokenEmbedding = try loadWeightTablePreferringFloat32Sidecar(
                at: topLevelPaths.tokenEmbedding,
                expectedCount: config.vocab * config.dModel
            )
            positionEmbedding = try loadWeightTable(
                at: topLevelPaths.positionEmbedding,
                expectedCount: config.maxSeq * config.dModel
            )
        case .llama:
            let topLevelPaths = try resolveLlamaTopLevelWeightPaths(config: config, weightDir: weightDir)
            tokenEmbedding = try loadWeightTablePreferringFloat32Sidecar(
                at: topLevelPaths.tokenEmbedding,
                expectedCount: config.vocab * config.dModel
            )
            positionEmbedding = []
        }

        return composeTestingEmbeddingInput(
            config: config,
            tokens: tokens,
            tokenEmbedding: tokenEmbedding,
            positionEmbedding: positionEmbedding
        )
    }

    static func evalHybridSingleLayerForTesting(
        config: MultiModelConfig,
        weightDir: String,
        layer: Int,
        tokens: [TokenID]
    ) throws -> [Float] {
        let weightDirURL = URL(fileURLWithPath: weightDir, isDirectory: true)
        try validateDirectory(weightDirURL)
        guard !tokens.isEmpty else {
            throw RealModelInferenceError.invalidGenerationParameters("Testing token list must not be empty")
        }
        guard tokens.count <= config.maxSeq else {
            throw RealModelInferenceError.invalidGenerationParameters(
                "Testing token count \(tokens.count) exceeds context \(config.maxSeq)"
            )
        }

        let tokenEmbedding: [Float]
        let positionEmbedding: [Float]
        switch config.architecture {
        case .gpt2:
            let topLevelPaths = try resolveTopLevelWeightPaths(config: config, weightDir: weightDir)
            tokenEmbedding = try loadWeightTablePreferringFloat32Sidecar(
                at: topLevelPaths.tokenEmbedding,
                expectedCount: config.vocab * config.dModel
            )
            positionEmbedding = try loadWeightTable(
                at: topLevelPaths.positionEmbedding,
                expectedCount: config.maxSeq * config.dModel
            )
        case .llama:
            let topLevelPaths = try resolveLlamaTopLevelWeightPaths(config: config, weightDir: weightDir)
            tokenEmbedding = try loadWeightTablePreferringFloat32Sidecar(
                at: topLevelPaths.tokenEmbedding,
                expectedCount: config.vocab * config.dModel
            )
            positionEmbedding = []
        }
        let paths = LayerWeightPaths.forLayer(layer, config: config, blobDir: weightDirURL.path)
        let weights = try loadHybridLayerWeightsLlama(config: config, paths: paths)
        let maxSeq = max(tokens.count, 1)
        let kernels = try LayerStorage<HybridDecodeKernelSet>(count: 1, throwingInitializer: { _ in
            try HybridDecodeKernelSet(weights: weights, maxSeq: maxSeq)
        })
        let handles = [try HybridDecodeSurfaceHandles(
            kernels: kernels[0],
            logicalMaxSeq: maxSeq,
            dim: config.dModel,
            qDim: config.attentionDim,
            kvDim: config.kvDim
        )]
        let metalAttention = try MetalAttentionKernel()
        let xCur = TensorBuffer(count: config.dModel, zeroed: true)
        var decodeState = try DecodeState(maxSeq: maxSeq)

        ForwardPass.initializeHybridDecodeCaches(surfaceHandles: handles, dim: config.dModel)

        for (position, token) in tokens.enumerated() {
            writeTestingIncrementalEmbedding(
                config: config,
                token: token,
                position: position,
                tokenEmbedding: tokenEmbedding,
                positionEmbedding: positionEmbedding,
                into: xCur
            )
            var timings = HybridDecodeTimingBreakdown()
            try ForwardPass.runHybridDecodeTimed(
                xCur: xCur,
                kernels: kernels,
                surfaceHandles: handles,
                metalAttention: metalAttention,
                decodeState: &decodeState,
                dim: config.dModel,
                nHeads: config.nHead,
                nKVHeads: config.nKVHead,
                headDim: config.headDim,
                preferCPUDecodeAttention: Self.prefersCPUDecodeAttention(
                    config: config,
                    environment: ProcessInfo.processInfo.environment
                ),
                timings: &timings
            )
        }

        return xCur.withUnsafeBufferPointer { Array($0) }
    }

    static func evalHybridSingleLayerAttentionForTesting(
        config: MultiModelConfig,
        weightDir: String,
        layer: Int,
        tokens: [TokenID]
    ) throws -> [Float] {
        try evalHybridSingleLayerAttentionOutputsForTesting(
            config: config,
            weightDir: weightDir,
            layer: layer,
            tokens: tokens
        ).hidden
    }

    static func evalHybridSingleLayerAttentionOutputsForTesting(
        config: MultiModelConfig,
        weightDir: String,
        layer: Int,
        tokens: [TokenID]
    ) throws -> AttentionTestingOutputs {
        let weightDirURL = URL(fileURLWithPath: weightDir, isDirectory: true)
        try validateDirectory(weightDirURL)
        guard !tokens.isEmpty else {
            throw RealModelInferenceError.invalidGenerationParameters("Testing token list must not be empty")
        }
        guard tokens.count <= config.maxSeq else {
            throw RealModelInferenceError.invalidGenerationParameters(
                "Testing token count \(tokens.count) exceeds context \(config.maxSeq)"
            )
        }

        let topLevelPaths = try resolveTopLevelWeightPaths(config: config, weightDir: weightDir)
        let tokenEmbedding = try loadWeightTablePreferringFloat32Sidecar(
            at: topLevelPaths.tokenEmbedding,
            expectedCount: config.vocab * config.dModel
        )
        let positionEmbedding = try loadWeightTable(
            at: topLevelPaths.positionEmbedding,
            expectedCount: config.maxSeq * config.dModel
        )
        let paths = LayerWeightPaths.forLayer(layer, config: config, blobDir: weightDirURL.path)
        let weights = try loadHybridLayerWeightsLlama(config: config, paths: paths)
        let maxSeq = max(tokens.count, 1)
        let kernels = try LayerStorage<HybridDecodeKernelSet>(count: 1, throwingInitializer: { _ in
            try HybridDecodeKernelSet(weights: weights, maxSeq: maxSeq)
        })
        let handles = [try HybridDecodeSurfaceHandles(
            kernels: kernels[0],
            logicalMaxSeq: maxSeq,
            dim: config.dModel,
            qDim: config.attentionDim,
            kvDim: config.kvDim
        )]
        let metalAttention = try MetalAttentionKernel()
        let xCur = TensorBuffer(count: config.dModel, zeroed: true)
        var decodeState = try DecodeState(maxSeq: maxSeq)

        ForwardPass.initializeHybridDecodeCaches(surfaceHandles: handles, dim: config.dModel)

        for (position, token) in tokens.enumerated() {
            writeTestingIncrementalEmbedding(
                config: config,
                token: token,
                position: position,
                tokenEmbedding: tokenEmbedding,
                positionEmbedding: positionEmbedding,
                into: xCur
            )
            var timings = HybridDecodeTimingBreakdown()
            try ForwardPass.runHybridDecodeTimed(
                xCur: xCur,
                kernels: kernels,
                surfaceHandles: handles,
                metalAttention: metalAttention,
                decodeState: &decodeState,
                dim: config.dModel,
                nHeads: config.nHead,
                nKVHeads: config.nKVHead,
                headDim: config.headDim,
                preferCPUDecodeAttention: Self.prefersCPUDecodeAttention(
                    config: config,
                    environment: ProcessInfo.processInfo.environment
                ),
                timings: &timings
            )
        }

        var hidden = [Float](repeating: 0, count: config.dModel)
        try hidden.withUnsafeMutableBufferPointer { buffer in
            try SurfaceIO.readFP16SpatialSlice(
                from: handles[0].ffnIn,
                channelOffset: 0,
                spatialIndex: 0,
                spatial: handles[0].laneSpatial,
                into: buffer,
                channels: config.dModel
            )
        }
        let kvDim = config.kvDim
        var kCache = [Float](repeating: 0, count: kvDim * maxSeq)
        var vCache = [Float](repeating: 0, count: kvDim * maxSeq)
        kCache.withUnsafeMutableBufferPointer { buffer in
            SurfaceIO.readFP16(
                from: handles[0].kCacheFull,
                into: buffer,
                channelOffset: 0,
                channels: kvDim,
                spatial: maxSeq
            )
        }
        vCache.withUnsafeMutableBufferPointer { buffer in
            SurfaceIO.readFP16(
                from: handles[0].vCacheFull,
                into: buffer,
                channelOffset: 0,
                channels: kvDim,
                spatial: maxSeq
            )
        }
        return AttentionTestingOutputs(hidden: hidden, kCache: kCache, vCache: vCache)
    }

    static func evalHybridSingleLayerHookedLlamaKCacheForTesting(
        config: MultiModelConfig,
        weightDir: String,
        layer: Int,
        tokens: [TokenID]
    ) throws -> HookedKCacheTestingOutputs {
        guard config.architecture == .llama else {
            throw RealModelInferenceError.unsupportedArchitecture(
                "Hooked llama K-cache testing helper currently supports llama-family artifacts only"
            )
        }
        let weightDirURL = URL(fileURLWithPath: weightDir, isDirectory: true)
        try validateDirectory(weightDirURL)
        guard !tokens.isEmpty else {
            throw RealModelInferenceError.invalidGenerationParameters("Testing token list must not be empty")
        }
        guard tokens.count <= config.maxSeq else {
            throw RealModelInferenceError.invalidGenerationParameters(
                "Testing token count \(tokens.count) exceeds context \(config.maxSeq)"
            )
        }

        let topLevelPaths = try resolveLlamaTopLevelWeightPaths(config: config, weightDir: weightDir)
        let tokenEmbedding = try loadWeightTablePreferringFloat32Sidecar(
            at: topLevelPaths.tokenEmbedding,
            expectedCount: config.vocab * config.dModel
        )
        let paths = LayerWeightPaths.forLayer(layer, config: config, blobDir: weightDirURL.path)
        let weights = try loadHybridLayerWeightsLlama(config: config, paths: paths)
        let qkNormWeights = try loadLlamaQKNormWeights(config: config, paths: paths)
        let maxSeq = max(tokens.count, 1)
        let kernels = try LayerStorage<HybridDecodeKernelSet>(count: 1, throwingInitializer: { _ in
            try HybridDecodeKernelSet(weights: weights, maxSeq: maxSeq)
        })
        let handles = [try HybridDecodeSurfaceHandles(
            kernels: kernels[0],
            logicalMaxSeq: maxSeq,
            dim: config.dModel,
            qDim: config.attentionDim,
            kvDim: config.kvDim
        )]
        let metalAttention = try MetalAttentionKernel()
        let xCur = TensorBuffer(count: config.dModel, zeroed: true)
        var decodeState = try DecodeState(maxSeq: maxSeq)
        let qBufSize = config.attentionDim
        let kBufSize = config.kvDim
        let ropeQBuf = UnsafeMutableBufferPointer<Float>.allocate(capacity: qBufSize)
        let ropeKBuf = UnsafeMutableBufferPointer<Float>.allocate(capacity: kBufSize)
        defer {
            ropeQBuf.deallocate()
            ropeKBuf.deallocate()
        }
        var lastRawKOut = [Float](repeating: 0, count: kBufSize)
        var lastHookedKOut = [Float](repeating: 0, count: kBufSize)
        var lastHookedKOutSurface = [Float](repeating: 0, count: kBufSize)

        let ropeHook: (Int, IOSurfaceRef, IOSurfaceRef, Int, Int) throws -> Void = { _, qSurf, kSurf, laneSp, tokenIndex in
            do {
                try SurfaceIO.readFP16SpatialSlice(
                    from: qSurf,
                    channelOffset: 0,
                    spatialIndex: 0,
                    spatial: laneSp,
                    into: ropeQBuf,
                    channels: qBufSize
                )
                try SurfaceIO.readFP16SpatialSlice(
                    from: kSurf,
                    channelOffset: 0,
                    spatialIndex: 0,
                    spatial: laneSp,
                    into: ropeKBuf,
                    channels: kBufSize
                )
            } catch {
                throw ANEError.invalidArguments("Hooked llama K-cache helper surface read failed: \(error)")
            }

            lastRawKOut = Array(ropeKBuf)

            if let qkNormWeights {
                qkNormWeights.q.withUnsafeBufferPointer { weights in
                    RMSNorm.applyPerHeadSingleTokenInPlace(
                        values: ropeQBuf.baseAddress!,
                        headCount: config.nHead,
                        headDim: config.headDim,
                        weights: weights.baseAddress!,
                        epsilon: Float(config.normEps)
                    )
                }
                qkNormWeights.k.withUnsafeBufferPointer { weights in
                    RMSNorm.applyPerHeadSingleTokenInPlace(
                        values: ropeKBuf.baseAddress!,
                        headCount: config.nKVHead,
                        headDim: config.headDim,
                        weights: weights.baseAddress!,
                        epsilon: Float(config.normEps)
                    )
                }
            }

            RoPE.applyDecodeStep(
                q: ropeQBuf.baseAddress!,
                k: ropeKBuf.baseAddress!,
                nHeads: config.nHead,
                nKVHeads: config.nKVHead,
                headDim: config.headDim,
                position: tokenIndex,
                theta: config.ropeTheta
            )

            lastHookedKOut = Array(ropeKBuf)

            do {
                try SurfaceIO.writeFP16SpatialSlice(
                    to: qSurf,
                    channelOffset: 0,
                    spatialIndex: 0,
                    spatial: laneSp,
                    data: UnsafeBufferPointer(ropeQBuf),
                    channels: qBufSize
                )
                try SurfaceIO.writeFP16SpatialSlice(
                    to: kSurf,
                    channelOffset: 0,
                    spatialIndex: 0,
                    spatial: laneSp,
                    data: UnsafeBufferPointer(ropeKBuf),
                    channels: kBufSize
                )
                try lastHookedKOutSurface.withUnsafeMutableBufferPointer { out in
                    try SurfaceIO.readFP16SpatialSlice(
                        from: kSurf,
                        channelOffset: 0,
                        spatialIndex: 0,
                        spatial: laneSp,
                        into: out,
                        channels: kBufSize
                    )
                }
            } catch {
                throw ANEError.invalidArguments("Hooked llama K-cache helper surface write failed: \(error)")
            }
        }

        ForwardPass.initializeHybridDecodeCaches(surfaceHandles: handles, dim: config.dModel)

        for (position, token) in tokens.enumerated() {
            writeTestingIncrementalEmbedding(
                config: config,
                token: token,
                position: position,
                tokenEmbedding: tokenEmbedding,
                positionEmbedding: [],
                into: xCur
            )
            var timings = HybridDecodeTimingBreakdown()
            try ForwardPass.runHybridDecodeTimed(
                xCur: xCur,
                kernels: kernels,
                surfaceHandles: handles,
                metalAttention: metalAttention,
                decodeState: &decodeState,
                dim: config.dModel,
                nHeads: config.nHead,
                nKVHeads: config.nKVHead,
                headDim: config.headDim,
                preferCPUDecodeAttention: Self.prefersCPUDecodeAttention(
                    config: config,
                    environment: ProcessInfo.processInfo.environment
                ),
                postQKVHook: ropeHook,
                timings: &timings
            )
        }

        var kCache = [Float](repeating: 0, count: config.kvDim * maxSeq)
        kCache.withUnsafeMutableBufferPointer { buffer in
            SurfaceIO.readFP16(
                from: handles[0].kCacheFull,
                into: buffer,
                channelOffset: 0,
                channels: config.kvDim,
                spatial: maxSeq
            )
        }

        return HookedKCacheTestingOutputs(
            rawKOut: lastRawKOut,
            hookedKOut: lastHookedKOut,
            hookedKOutSurface: lastHookedKOutSurface,
            kCache: kCache
        )
    }

    static func evalHybridSingleLayerRawQKVOutputsForTesting(
        config: MultiModelConfig,
        weightDir: String,
        layer: Int,
        token: TokenID,
        position: Int = 0
    ) throws -> RawQKVTestingOutputs {
        guard config.architecture == .llama else {
            throw RealModelInferenceError.unsupportedArchitecture(
                "Raw hybrid QKV testing helper currently supports llama-family artifacts only"
            )
        }

        let weightDirURL = URL(fileURLWithPath: weightDir, isDirectory: true)
        try validateDirectory(weightDirURL)
        guard position >= 0, position < config.maxSeq else {
            throw RealModelInferenceError.invalidGenerationParameters(
                "Testing token position \(position) exceeds context \(config.maxSeq)"
            )
        }

        let topLevelPaths = try resolveLlamaTopLevelWeightPaths(config: config, weightDir: weightDir)
        let tokenEmbedding = try loadWeightTablePreferringFloat32Sidecar(
            at: topLevelPaths.tokenEmbedding,
            expectedCount: config.vocab * config.dModel
        )
        let tokenBase = Int(token) * config.dModel
        guard tokenBase >= 0, tokenBase + config.dModel <= tokenEmbedding.count else {
            throw RealModelInferenceError.invalidGenerationParameters(
                "Testing token \(token) is outside embedding table bounds"
            )
        }

        let paths = LayerWeightPaths.forLayer(layer, config: config, blobDir: weightDirURL.path)
        let weights = try loadHybridLayerWeightsLlama(config: config, paths: paths)
        let kernels = try LayerStorage<HybridDecodeKernelSet>(count: 1, throwingInitializer: { _ in
            try HybridDecodeKernelSet(weights: weights, maxSeq: 1)
        })
        let handles = [try HybridDecodeSurfaceHandles(
            kernels: kernels[0],
            logicalMaxSeq: 1,
            dim: config.dModel,
            qDim: config.attentionDim,
            kvDim: config.kvDim
        )]

        let xCur = TensorBuffer(count: config.dModel, zeroed: true)
        xCur.withUnsafeMutableBufferPointer { dst in
            for channel in 0..<config.dModel {
                dst[channel] = tokenEmbedding[tokenBase + channel]
            }
        }
        try xCur.withUnsafeBufferPointer { xBuf in
            try SurfaceIO.writeFP16SpatialSlice(
                to: handles[0].qkvIn,
                channelOffset: 0,
                spatialIndex: 0,
                spatial: handles[0].laneSpatial,
                data: xBuf,
                channels: config.dModel
            )
        }

        do {
            try kernels[0].decodeQKVOnly.eval()
        } catch {
            throw RealModelInferenceError.runtimeFailure("Single-layer raw decodeQKVOnly eval failed: \(error)")
        }

        let qDim = config.attentionDim
        let kvDim = config.kvDim
        var qOut = [Float](repeating: 0, count: qDim)
        var kOut = [Float](repeating: 0, count: kvDim)
        var vOut = [Float](repeating: 0, count: kvDim)
        try qOut.withUnsafeMutableBufferPointer { buffer in
            try SurfaceIO.readFP16SpatialSlice(
                from: handles[0].qOut,
                channelOffset: 0,
                spatialIndex: 0,
                spatial: handles[0].laneSpatial,
                into: buffer,
                channels: qDim
            )
        }
        try kOut.withUnsafeMutableBufferPointer { buffer in
            try SurfaceIO.readFP16SpatialSlice(
                from: handles[0].kOut,
                channelOffset: 0,
                spatialIndex: 0,
                spatial: handles[0].laneSpatial,
                into: buffer,
                channels: kvDim
            )
        }
        try vOut.withUnsafeMutableBufferPointer { buffer in
            try SurfaceIO.readFP16SpatialSlice(
                from: handles[0].vOut,
                channelOffset: 0,
                spatialIndex: 0,
                spatial: handles[0].laneSpatial,
                into: buffer,
                channels: kvDim
            )
        }

        return RawQKVTestingOutputs(qOut: qOut, kOut: kOut, vOut: vOut)
    }

    static func evalHybridSingleLayerQKVInputStabilityForTesting(
        config: MultiModelConfig,
        weightDir: String,
        layer: Int,
        input: [Float]
    ) throws -> QKVInputStabilityTestingOutputs {
        guard config.architecture == .llama else {
            throw RealModelInferenceError.unsupportedArchitecture(
                "Hybrid decode QKV input stability helper currently supports llama-family artifacts only"
            )
        }
        guard input.count == config.dModel else {
            throw RealModelInferenceError.invalidGenerationParameters(
                "Testing input count \(input.count) does not match dModel \(config.dModel)"
            )
        }

        let weightDirURL = URL(fileURLWithPath: weightDir, isDirectory: true)
        try validateDirectory(weightDirURL)

        let paths = LayerWeightPaths.forLayer(layer, config: config, blobDir: weightDirURL.path)
        let weights = try loadHybridLayerWeightsLlama(config: config, paths: paths)
        let kernels = try LayerStorage<HybridDecodeKernelSet>(count: 1, throwingInitializer: { _ in
            try HybridDecodeKernelSet(weights: weights, maxSeq: 1)
        })
        let handles = [try HybridDecodeSurfaceHandles(
            kernels: kernels[0],
            logicalMaxSeq: 1,
            dim: config.dModel,
            qDim: config.attentionDim,
            kvDim: config.kvDim
        )]

        try input.withUnsafeBufferPointer { source in
            try SurfaceIO.writeFP16SpatialSlice(
                to: handles[0].qkvIn,
                channelOffset: 0,
                spatialIndex: 0,
                spatial: handles[0].laneSpatial,
                data: source,
                channels: config.dModel
            )
        }

        var inputBeforeQKV = [Float](repeating: 0, count: config.dModel)
        try inputBeforeQKV.withUnsafeMutableBufferPointer { buffer in
            try SurfaceIO.readFP16SpatialSlice(
                from: handles[0].qkvIn,
                channelOffset: 0,
                spatialIndex: 0,
                spatial: handles[0].laneSpatial,
                into: buffer,
                channels: config.dModel
            )
        }

        do {
            try kernels[0].decodeQKVOnly.eval()
        } catch {
            throw RealModelInferenceError.runtimeFailure("Single-layer QKV stability eval failed: \(error)")
        }

        var inputAfterQKV = [Float](repeating: 0, count: config.dModel)
        try inputAfterQKV.withUnsafeMutableBufferPointer { buffer in
            try SurfaceIO.readFP16SpatialSlice(
                from: handles[0].qkvIn,
                channelOffset: 0,
                spatialIndex: 0,
                spatial: handles[0].laneSpatial,
                into: buffer,
                channels: config.dModel
            )
        }

        return QKVInputStabilityTestingOutputs(
            inputBeforeQKV: inputBeforeQKV,
            inputAfterQKV: inputAfterQKV
        )
    }

    static func evalHybridSingleLayerDecodeProjectionForTesting(
        config: MultiModelConfig,
        weightDir: String,
        layer: Int,
        context: [Float],
        residual: [Float]? = nil
    ) throws -> DecodeProjectionTestingOutputs {
        guard config.architecture == .llama else {
            throw RealModelInferenceError.unsupportedArchitecture(
                "Hybrid decode projection testing helper currently supports llama-family artifacts only"
            )
        }
        guard context.count == config.attentionDim else {
            throw RealModelInferenceError.invalidGenerationParameters(
                "Testing context count \(context.count) does not match attention dim \(config.attentionDim)"
            )
        }
        if let residual, residual.count != config.dModel {
            throw RealModelInferenceError.invalidGenerationParameters(
                "Testing residual count \(residual.count) does not match dModel \(config.dModel)"
            )
        }

        let weightDirURL = URL(fileURLWithPath: weightDir, isDirectory: true)
        try validateDirectory(weightDirURL)

        let paths = LayerWeightPaths.forLayer(layer, config: config, blobDir: weightDirURL.path)
        let weights = try loadHybridLayerWeightsLlama(config: config, paths: paths)
        let kernels = try LayerStorage<HybridDecodeKernelSet>(count: 1, throwingInitializer: { _ in
            try HybridDecodeKernelSet(weights: weights, maxSeq: 1)
        })
        let handles = [try HybridDecodeSurfaceHandles(
            kernels: kernels[0],
            logicalMaxSeq: 1,
            dim: config.dModel,
            qDim: config.attentionDim,
            kvDim: config.kvDim
        )]

        try context.withUnsafeBufferPointer { source in
            try writeFP32SpatialSlice(
                to: handles[0].projectionContextIn,
                spatialIndex: 0,
                spatial: handles[0].laneSpatial,
                data: source,
                channels: config.attentionDim
            )
        }
        let projectionResidual = residual ?? [Float](repeating: 0, count: config.dModel)
        try projectionResidual.withUnsafeBufferPointer { source in
            try SurfaceIO.writeFP16SpatialSlice(
                to: handles[0].projectionResidualIn,
                channelOffset: 0,
                spatialIndex: 0,
                spatial: handles[0].laneSpatial,
                data: source,
                channels: config.dModel
            )
        }

        do {
            try kernels[0].decodeProjection.eval()
        } catch {
            throw RealModelInferenceError.runtimeFailure("Single-layer decodeProjection eval failed: \(error)")
        }

        var output = [Float](repeating: 0, count: config.dModel)
        try output.withUnsafeMutableBufferPointer { buffer in
            try SurfaceIO.readFP16SpatialSlice(
                from: handles[0].projectionOut,
                channelOffset: 0,
                spatialIndex: 0,
                spatial: handles[0].laneSpatial,
                into: buffer,
                channels: config.dModel
            )
        }
        return DecodeProjectionTestingOutputs(output: output)
    }

    static func evalHybridSingleLayerDecodeFFNForTesting(
        config: MultiModelConfig,
        weightDir: String,
        layer: Int,
        input: [Float]
    ) throws -> DecodeFFNTestingOutputs {
        guard config.architecture == .llama else {
            throw RealModelInferenceError.unsupportedArchitecture(
                "Hybrid decode FFN testing helper currently supports llama-family artifacts only"
            )
        }
        guard input.count == config.dModel else {
            throw RealModelInferenceError.invalidGenerationParameters(
                "Testing FFN input count \(input.count) does not match dModel \(config.dModel)"
            )
        }

        let weightDirURL = URL(fileURLWithPath: weightDir, isDirectory: true)
        try validateDirectory(weightDirURL)

        let paths = LayerWeightPaths.forLayer(layer, config: config, blobDir: weightDirURL.path)
        let weights = try loadHybridLayerWeightsLlama(config: config, paths: paths)
        let kernels = try LayerStorage<HybridDecodeKernelSet>(count: 1, throwingInitializer: { _ in
            try HybridDecodeKernelSet(weights: weights, maxSeq: 1)
        })
        let ffnIn = try kernels[0].decodeFFN.inputSurface(at: 0)
        let ffnOut = try kernels[0].decodeFFN.outputSurface(at: 0)
        let laneSpatial = kernels[0].laneSpatial

        try input.withUnsafeBufferPointer { source in
            try SurfaceIO.writeFP16SpatialSlice(
                to: ffnIn,
                channelOffset: 0,
                spatialIndex: 0,
                spatial: laneSpatial,
                data: source,
                channels: config.dModel
            )
        }

        do {
            try kernels[0].decodeFFN.eval()
        } catch {
            throw RealModelInferenceError.runtimeFailure("Single-layer decodeFFN eval failed: \(error)")
        }

        var output = [Float](repeating: 0, count: config.dModel)
        try output.withUnsafeMutableBufferPointer { buffer in
            try SurfaceIO.readFP16SpatialSlice(
                from: ffnOut,
                channelOffset: 0,
                spatialIndex: 0,
                spatial: laneSpatial,
                into: buffer,
                channels: config.dModel
            )
        }
        return DecodeFFNTestingOutputs(output: output)
    }

    static func evalHybridSingleLayerDecodeFFNPostNormForTesting(
        config: MultiModelConfig,
        weightDir: String,
        layer: Int,
        normalizedInput: [Float],
        residual: [Float]
    ) throws -> DecodeFFNTestingOutputs {
        guard config.architecture == .llama else {
            throw RealModelInferenceError.unsupportedArchitecture(
                "Hybrid decode post-norm FFN testing helper currently supports llama-family artifacts only"
            )
        }
        guard normalizedInput.count == config.dModel else {
            throw RealModelInferenceError.invalidGenerationParameters(
                "Testing normalized input count \(normalizedInput.count) does not match dModel \(config.dModel)"
            )
        }
        guard residual.count == config.dModel else {
            throw RealModelInferenceError.invalidGenerationParameters(
                "Testing residual count \(residual.count) does not match dModel \(config.dModel)"
            )
        }

        let weightDirURL = URL(fileURLWithPath: weightDir, isDirectory: true)
        try validateDirectory(weightDirURL)

        let paths = LayerWeightPaths.forLayer(layer, config: config, blobDir: weightDirURL.path)
        let weights = try loadHybridLayerWeightsLlama(config: config, paths: paths)
        let generator = DecodeFFNPostNormGenerator(
            dim: weights.dim,
            hiddenDim: weights.hiddenDim,
            laneSpatial: HybridDecodeKernelSet.resolvedLaneSpatialForCurrentProcess(),
            architecture: weights.architecture
        )
        let w1Blob = WeightBlob.build(from: weights.W1.withUnsafeBufferPointer { Array($0) }, rows: weights.hiddenDim, cols: weights.dim)
        let w3Blob = WeightBlob.build(from: weights.W3.withUnsafeBufferPointer { Array($0) }, rows: weights.hiddenDim, cols: weights.dim)
        let w2Blob = WeightBlob.build(from: weights.W2.withUnsafeBufferPointer { Array($0) }, rows: weights.dim, cols: weights.hiddenDim)
        let kernel = try ANEKernel(
            milText: generator.milText,
            weights: [
                (path: "@model_path/weights/w1.bin", data: w1Blob),
                (path: "@model_path/weights/w3.bin", data: w3Blob),
                (path: "@model_path/weights/w2.bin", data: w2Blob),
            ],
            inputSizes: generator.inputByteSizes,
            outputSizes: generator.outputByteSizes
        )

        let normalizedSurface = try kernel.inputSurface(at: 0)
        let residualSurface = try kernel.inputSurface(at: 1)
        let outputSurface = try kernel.outputSurface(at: 0)
        let laneSpatial = HybridDecodeKernelSet.resolvedLaneSpatialForCurrentProcess()

        try normalizedInput.withUnsafeBufferPointer { source in
            try SurfaceIO.writeFP16SpatialSlice(
                to: normalizedSurface,
                channelOffset: 0,
                spatialIndex: 0,
                spatial: laneSpatial,
                data: source,
                channels: config.dModel
            )
        }
        try residual.withUnsafeBufferPointer { source in
            try SurfaceIO.writeFP16SpatialSlice(
                to: residualSurface,
                channelOffset: 0,
                spatialIndex: 0,
                spatial: laneSpatial,
                data: source,
                channels: config.dModel
            )
        }

        do {
            try kernel.eval()
        } catch {
            throw RealModelInferenceError.runtimeFailure("Single-layer decodeFFN post-norm eval failed: \(error)")
        }

        var output = [Float](repeating: 0, count: config.dModel)
        try output.withUnsafeMutableBufferPointer { buffer in
            try SurfaceIO.readFP16SpatialSlice(
                from: outputSurface,
                channelOffset: 0,
                spatialIndex: 0,
                spatial: laneSpatial,
                into: buffer,
                channels: config.dModel
            )
        }
        return DecodeFFNTestingOutputs(output: output)
    }

    static func evalHybridSingleLayerDecodeFFNStagesForTesting(
        config: MultiModelConfig,
        weightDir: String,
        layer: Int,
        normalizedInput: [Float]
    ) throws -> DecodeFFNStagesTestingOutputs {
        guard config.architecture == .llama else {
            throw RealModelInferenceError.unsupportedArchitecture(
                "Hybrid decode FFN stage testing helper currently supports llama-family artifacts only"
            )
        }
        guard normalizedInput.count == config.dModel else {
            throw RealModelInferenceError.invalidGenerationParameters(
                "Testing normalized input count \(normalizedInput.count) does not match dModel \(config.dModel)"
            )
        }

        let weightDirURL = URL(fileURLWithPath: weightDir, isDirectory: true)
        try validateDirectory(weightDirURL)

        let paths = LayerWeightPaths.forLayer(layer, config: config, blobDir: weightDirURL.path)
        let weights = try loadHybridLayerWeightsLlama(config: config, paths: paths)
        let laneSpatial = HybridDecodeKernelSet.resolvedLaneSpatialForCurrentProcess()
        let w1Blob = WeightBlob.build(from: weights.W1.withUnsafeBufferPointer { Array($0) }, rows: weights.hiddenDim, cols: weights.dim)
        let w3Blob = WeightBlob.build(from: weights.W3.withUnsafeBufferPointer { Array($0) }, rows: weights.hiddenDim, cols: weights.dim)
        let w2Blob = WeightBlob.build(from: weights.W2.withUnsafeBufferPointer { Array($0) }, rows: weights.dim, cols: weights.hiddenDim)

        func runStage(_ stage: DecodeFFNStagesGenerator.Stage, channels: Int) throws -> [Float] {
            let generator = DecodeFFNStagesGenerator(
                dim: weights.dim,
                hiddenDim: weights.hiddenDim,
                laneSpatial: laneSpatial,
                stage: stage
            )
            let kernel = try ANEKernel(
                milText: generator.milText,
                weights: [
                    (path: "@model_path/weights/w1.bin", data: w1Blob),
                    (path: "@model_path/weights/w3.bin", data: w3Blob),
                    (path: "@model_path/weights/w2.bin", data: w2Blob),
                ],
                inputSizes: generator.inputByteSizes,
                outputSizes: generator.outputByteSizes
            )
            let normalizedSurface = try kernel.inputSurface(at: 0)
            let outputSurface = try kernel.outputSurface(at: 0)
            try normalizedInput.withUnsafeBufferPointer { source in
                try SurfaceIO.writeFP16SpatialSlice(
                    to: normalizedSurface,
                    channelOffset: 0,
                    spatialIndex: 0,
                    spatial: laneSpatial,
                    data: source,
                    channels: config.dModel
                )
            }
            do {
                try kernel.eval()
            } catch {
                throw RealModelInferenceError.runtimeFailure("Single-layer decodeFFN \(stage) eval failed: \(error)")
            }
            var output = [Float](repeating: 0, count: channels)
            try output.withUnsafeMutableBufferPointer { buffer in
                try SurfaceIO.readFP16SpatialSlice(
                    from: outputSurface,
                    channelOffset: 0,
                    spatialIndex: 0,
                    spatial: laneSpatial,
                    into: buffer,
                    channels: channels
                )
            }
            return output
        }

        return DecodeFFNStagesTestingOutputs(
            gateLinear: try runStage(.gateLinear, channels: config.hiddenDim),
            upLinear: try runStage(.upLinear, channels: config.hiddenDim),
            siluGate: try runStage(.siluGate, channels: config.hiddenDim),
            gated: try runStage(.gated, channels: config.hiddenDim),
            down: try runStage(.down, channels: config.dModel)
        )
    }

    static func evalHybridSingleLayerMetalContextForTesting(
        config: MultiModelConfig,
        weightDir: String,
        layer: Int,
        token: TokenID,
        useFusedSDPA: Bool = true
    ) throws -> HybridMetalContextTestingOutputs {
        guard config.architecture == .llama else {
            throw RealModelInferenceError.unsupportedArchitecture(
                "Hybrid Metal context testing helper currently supports llama-family artifacts only"
            )
        }

        let weightDirURL = URL(fileURLWithPath: weightDir, isDirectory: true)
        try validateDirectory(weightDirURL)
        let topLevelPaths = try resolveLlamaTopLevelWeightPaths(config: config, weightDir: weightDir)
        let tokenEmbedding = try loadWeightTablePreferringFloat32Sidecar(
            at: topLevelPaths.tokenEmbedding,
            expectedCount: config.vocab * config.dModel
        )
        let tokenBase = Int(token) * config.dModel
        guard tokenBase >= 0, tokenBase + config.dModel <= tokenEmbedding.count else {
            throw RealModelInferenceError.invalidGenerationParameters(
                "Testing token \(token) is outside embedding table bounds"
            )
        }

        let paths = LayerWeightPaths.forLayer(layer, config: config, blobDir: weightDirURL.path)
        let weights = try loadHybridLayerWeightsLlama(config: config, paths: paths)
        let kernels = try LayerStorage<HybridDecodeKernelSet>(count: 1, throwingInitializer: { _ in
            try HybridDecodeKernelSet(weights: weights, maxSeq: 1)
        })
        let handles = [try HybridDecodeSurfaceHandles(
            kernels: kernels[0],
            logicalMaxSeq: 1,
            dim: config.dModel,
            qDim: config.attentionDim,
            kvDim: config.kvDim
        )]
        let metalAttention = try MetalAttentionKernel()

        let xCur = TensorBuffer(count: config.dModel, zeroed: true)
        xCur.withUnsafeMutableBufferPointer { dst in
            for channel in 0..<config.dModel {
                dst[channel] = tokenEmbedding[tokenBase + channel]
            }
        }
        try xCur.withUnsafeBufferPointer { xBuf in
            try SurfaceIO.writeFP16SpatialSlice(
                to: handles[0].qkvIn,
                channelOffset: 0,
                spatialIndex: 0,
                spatial: handles[0].laneSpatial,
                data: xBuf,
                channels: config.dModel
            )
        }

        do {
            try kernels[0].decodeQKVOnly.eval()
        } catch {
            throw RealModelInferenceError.runtimeFailure("Single-layer Metal-context decodeQKVOnly eval failed: \(error)")
        }

        let qDim = config.attentionDim
        let kvDim = config.kvDim
        var qOut = [Float](repeating: 0, count: qDim)
        var kOut = [Float](repeating: 0, count: kvDim)
        var vOut = [Float](repeating: 0, count: kvDim)
        try qOut.withUnsafeMutableBufferPointer { buffer in
            try SurfaceIO.readFP16SpatialSlice(
                from: handles[0].qOut,
                channelOffset: 0,
                spatialIndex: 0,
                spatial: handles[0].laneSpatial,
                into: buffer,
                channels: qDim
            )
        }
        try kOut.withUnsafeMutableBufferPointer { buffer in
            try SurfaceIO.readFP16SpatialSlice(
                from: handles[0].kOut,
                channelOffset: 0,
                spatialIndex: 0,
                spatial: handles[0].laneSpatial,
                into: buffer,
                channels: kvDim
            )
        }
        try vOut.withUnsafeMutableBufferPointer { buffer in
            try SurfaceIO.readFP16SpatialSlice(
                from: handles[0].vOut,
                channelOffset: 0,
                spatialIndex: 0,
                spatial: handles[0].laneSpatial,
                into: buffer,
                channels: kvDim
            )
        }

        do {
            try SurfaceIO.copyFP16SpatialSlice(
                dst: handles[0].kCacheFull,
                dstChannelOffset: 0,
                dstSpatialIndex: 0,
                dstSpatial: 1,
                src: handles[0].kOut,
                srcChannelOffset: 0,
                srcSpatialIndex: 0,
                srcSpatial: handles[0].laneSpatial,
                channels: kvDim
            )
            try SurfaceIO.copyFP16SpatialSlice(
                dst: handles[0].vCacheFull,
                dstChannelOffset: 0,
                dstSpatialIndex: 0,
                dstSpatial: 1,
                src: handles[0].vOut,
                srcChannelOffset: 0,
                srcSpatialIndex: 0,
                srcSpatial: handles[0].laneSpatial,
                channels: kvDim
            )
        } catch {
            throw RealModelInferenceError.runtimeFailure("Single-layer Metal-context KV cache write failed: \(error)")
        }

        let metalShape = try MetalDecodeAttentionShape(
            heads: config.nHead,
            kvHeads: config.nKVHead,
            headDim: config.headDim,
            visibleTokens: 1,
            cacheStride: 1,
            laneStride: handles[0].laneSpatial
        )
        do {
            if useFusedSDPA {
                try metalAttention.runFusedDecodeSDPAIntoSurface(
                    qSurface: handles[0].qOut,
                    kCacheSurface: handles[0].kCacheFull,
                    vCacheSurface: handles[0].vCacheFull,
                    contextSurface: handles[0].projectionContextIn,
                    shape: metalShape
                )
            } else {
                try metalAttention.runDecodeContextIntoSurface(
                    qSurface: handles[0].qOut,
                    kCacheSurface: handles[0].kCacheFull,
                    vCacheSurface: handles[0].vCacheFull,
                    contextSurface: handles[0].projectionContextIn,
                    shape: metalShape
                )
            }
        } catch {
            throw RealModelInferenceError.runtimeFailure("Single-layer Metal-context SDPA eval failed: \(error)")
        }

        var context = [Float](repeating: 0, count: qDim)
        try context.withUnsafeMutableBufferPointer { buffer in
            try readFP32SpatialSlice(
                from: handles[0].projectionContextIn,
                spatialIndex: 0,
                spatial: handles[0].laneSpatial,
                into: buffer,
                channels: qDim
            )
        }

        return HybridMetalContextTestingOutputs(
            context: context,
            qOut: qOut,
            kOut: kOut,
            vOut: vOut
        )
    }

    static func evalHybridSingleLayerHookedLlamaMetalContextForTesting(
        config: MultiModelConfig,
        weightDir: String,
        layer: Int,
        tokens: [TokenID],
        useFusedSDPA: Bool = true
    ) throws -> HookedHybridMetalContextTestingOutputs {
        guard config.architecture == .llama else {
            throw RealModelInferenceError.unsupportedArchitecture(
                "Hooked llama Metal context testing helper currently supports llama-family artifacts only"
            )
        }

        let weightDirURL = URL(fileURLWithPath: weightDir, isDirectory: true)
        try validateDirectory(weightDirURL)
        guard !tokens.isEmpty else {
            throw RealModelInferenceError.invalidGenerationParameters("Testing token list must not be empty")
        }
        guard tokens.count <= config.maxSeq else {
            throw RealModelInferenceError.invalidGenerationParameters(
                "Testing token count \(tokens.count) exceeds context \(config.maxSeq)"
            )
        }

        let topLevelPaths = try resolveLlamaTopLevelWeightPaths(config: config, weightDir: weightDir)
        let tokenEmbedding = try loadWeightTablePreferringFloat32Sidecar(
            at: topLevelPaths.tokenEmbedding,
            expectedCount: config.vocab * config.dModel
        )
        let paths = LayerWeightPaths.forLayer(layer, config: config, blobDir: weightDirURL.path)
        let weights = try loadHybridLayerWeightsLlama(config: config, paths: paths)
        let qkNormWeights = try loadLlamaQKNormWeights(config: config, paths: paths)
        let maxSeq = max(tokens.count, 1)
        let kernels = try LayerStorage<HybridDecodeKernelSet>(count: 1, throwingInitializer: { _ in
            try HybridDecodeKernelSet(weights: weights, maxSeq: maxSeq)
        })
        let handles = [try HybridDecodeSurfaceHandles(
            kernels: kernels[0],
            logicalMaxSeq: maxSeq,
            dim: config.dModel,
            qDim: config.attentionDim,
            kvDim: config.kvDim
        )]
        let metalAttention = try MetalAttentionKernel()
        let xCur = TensorBuffer(count: config.dModel, zeroed: true)
        var decodeState = try DecodeState(maxSeq: maxSeq)
        let qBufSize = config.attentionDim
        let kBufSize = config.kvDim
        let ropeQBuf = UnsafeMutableBufferPointer<Float>.allocate(capacity: qBufSize)
        let ropeKBuf = UnsafeMutableBufferPointer<Float>.allocate(capacity: kBufSize)
        defer {
            ropeQBuf.deallocate()
            ropeKBuf.deallocate()
        }

        let ropeHook: (Int, IOSurfaceRef, IOSurfaceRef, Int, Int) throws -> Void = { _, qSurf, kSurf, laneSp, tokenIndex in
            do {
                try SurfaceIO.readFP16SpatialSlice(
                    from: qSurf,
                    channelOffset: 0,
                    spatialIndex: 0,
                    spatial: laneSp,
                    into: ropeQBuf,
                    channels: qBufSize
                )
                try SurfaceIO.readFP16SpatialSlice(
                    from: kSurf,
                    channelOffset: 0,
                    spatialIndex: 0,
                    spatial: laneSp,
                    into: ropeKBuf,
                    channels: kBufSize
                )
            } catch {
                throw ANEError.invalidArguments("Hooked llama Metal-context helper surface read failed: \(error)")
            }

            if let qkNormWeights {
                qkNormWeights.q.withUnsafeBufferPointer { weights in
                    RMSNorm.applyPerHeadSingleTokenInPlace(
                        values: ropeQBuf.baseAddress!,
                        headCount: config.nHead,
                        headDim: config.headDim,
                        weights: weights.baseAddress!,
                        epsilon: Float(config.normEps)
                    )
                }
                qkNormWeights.k.withUnsafeBufferPointer { weights in
                    RMSNorm.applyPerHeadSingleTokenInPlace(
                        values: ropeKBuf.baseAddress!,
                        headCount: config.nKVHead,
                        headDim: config.headDim,
                        weights: weights.baseAddress!,
                        epsilon: Float(config.normEps)
                    )
                }
            }

            RoPE.applyDecodeStep(
                q: ropeQBuf.baseAddress!,
                k: ropeKBuf.baseAddress!,
                nHeads: config.nHead,
                nKVHeads: config.nKVHead,
                headDim: config.headDim,
                position: tokenIndex,
                theta: config.ropeTheta
            )

            do {
                try SurfaceIO.writeFP16SpatialSlice(
                    to: qSurf,
                    channelOffset: 0,
                    spatialIndex: 0,
                    spatial: laneSp,
                    data: UnsafeBufferPointer(ropeQBuf),
                    channels: qBufSize
                )
                try SurfaceIO.writeFP16SpatialSlice(
                    to: kSurf,
                    channelOffset: 0,
                    spatialIndex: 0,
                    spatial: laneSp,
                    data: UnsafeBufferPointer(ropeKBuf),
                    channels: kBufSize
                )
            } catch {
                throw ANEError.invalidArguments("Hooked llama Metal-context helper surface write failed: \(error)")
            }
        }

        ForwardPass.initializeHybridDecodeCaches(surfaceHandles: handles, dim: config.dModel)

        for (position, token) in tokens.enumerated() {
            writeTestingIncrementalEmbedding(
                config: config,
                token: token,
                position: position,
                tokenEmbedding: tokenEmbedding,
                positionEmbedding: [],
                into: xCur
            )
            var timings = HybridDecodeTimingBreakdown()
            try ForwardPass.runHybridDecodeTimed(
                xCur: xCur,
                kernels: kernels,
                surfaceHandles: handles,
                metalAttention: metalAttention,
                decodeState: &decodeState,
                dim: config.dModel,
                nHeads: config.nHead,
                nKVHeads: config.nKVHead,
                headDim: config.headDim,
                preferCPUDecodeAttention: Self.prefersCPUDecodeAttention(
                    config: config,
                    environment: ProcessInfo.processInfo.environment
                ),
                postQKVHook: ropeHook,
                timings: &timings
            )
        }

        var qOut = [Float](repeating: 0, count: config.attentionDim)
        try qOut.withUnsafeMutableBufferPointer { buffer in
            try SurfaceIO.readFP16SpatialSlice(
                from: handles[0].qOut,
                channelOffset: 0,
                spatialIndex: 0,
                spatial: handles[0].laneSpatial,
                into: buffer,
                channels: config.attentionDim
            )
        }

        var kCache = [Float](repeating: 0, count: config.kvDim * maxSeq)
        var vCache = [Float](repeating: 0, count: config.kvDim * maxSeq)
        kCache.withUnsafeMutableBufferPointer { buffer in
            SurfaceIO.readFP16(
                from: handles[0].kCacheFull,
                into: buffer,
                channelOffset: 0,
                channels: config.kvDim,
                spatial: maxSeq
            )
        }
        vCache.withUnsafeMutableBufferPointer { buffer in
            SurfaceIO.readFP16(
                from: handles[0].vCacheFull,
                into: buffer,
                channelOffset: 0,
                channels: config.kvDim,
                spatial: maxSeq
            )
        }

        var context = [Float](repeating: 0, count: config.attentionDim)
        try context.withUnsafeMutableBufferPointer { buffer in
            try readFP32SpatialSlice(
                from: handles[0].projectionContextIn,
                spatialIndex: 0,
                spatial: handles[0].laneSpatial,
                into: buffer,
                channels: config.attentionDim
            )
        }

        return HookedHybridMetalContextTestingOutputs(
            context: context,
            qOut: qOut,
            kCache: kCache,
            vCache: vCache
        )
    }

    static func evalHybridLlamaLayerHiddenLineageForTesting(
        config: MultiModelConfig,
        weightDir: String,
        tokens: [TokenID]
    ) throws -> LayerHiddenLineageTestingOutputs {
        guard config.architecture == .llama else {
            throw RealModelInferenceError.unsupportedArchitecture(
                "Llama layer lineage testing helper currently supports llama-family artifacts only"
            )
        }
        let weightDirURL = URL(fileURLWithPath: weightDir, isDirectory: true)
        try validateDirectory(weightDirURL)
        guard !tokens.isEmpty else {
            throw RealModelInferenceError.invalidGenerationParameters("Testing token list must not be empty")
        }
        guard tokens.count <= config.maxSeq else {
            throw RealModelInferenceError.invalidGenerationParameters(
                "Testing token count \(tokens.count) exceeds context \(config.maxSeq)"
            )
        }

        let topLevelPaths = try resolveLlamaTopLevelWeightPaths(config: config, weightDir: weightDir)
        let tokenEmbedding = try loadWeightTablePreferringFloat32Sidecar(
            at: topLevelPaths.tokenEmbedding,
            expectedCount: config.vocab * config.dModel
        )
        let maxSeq = max(tokens.count, 1)
        let kernels = try Self.compileHybridLayers(
            config: config,
            weightDirURL: weightDirURL,
            maxSeq: maxSeq
        )
        let handles = try (0..<config.nLayer).map { layerIndex in
            try HybridDecodeSurfaceHandles(
                kernels: kernels[layerIndex],
                logicalMaxSeq: maxSeq,
                dim: config.dModel,
                qDim: config.attentionDim,
                kvDim: config.kvDim
            )
        }
        let layerPaths = (0..<config.nLayer).map { LayerWeightPaths.forLayer($0, config: config, blobDir: weightDirURL.path) }
        let layerQKNormWeights = try layerPaths.map { try loadLlamaQKNormWeights(config: config, paths: $0) }
        let metalAttention = try MetalAttentionKernel()
        let xCur = TensorBuffer(count: config.dModel, zeroed: true)
        var decodeState = try DecodeState(maxSeq: maxSeq)

        let qBufSize = config.attentionDim
        let kBufSize = config.kvDim
        let ropeQBuf = UnsafeMutableBufferPointer<Float>.allocate(capacity: qBufSize)
        let ropeKBuf = UnsafeMutableBufferPointer<Float>.allocate(capacity: kBufSize)
        defer {
            ropeQBuf.deallocate()
            ropeKBuf.deallocate()
        }

        let ropeHook: (Int, IOSurfaceRef, IOSurfaceRef, Int, Int) throws -> Void = { layerIndex, qSurf, kSurf, laneSp, tokenIndex in
            do {
                try SurfaceIO.readFP16SpatialSlice(
                    from: qSurf,
                    channelOffset: 0,
                    spatialIndex: 0,
                    spatial: laneSp,
                    into: ropeQBuf,
                    channels: qBufSize
                )
                try SurfaceIO.readFP16SpatialSlice(
                    from: kSurf,
                    channelOffset: 0,
                    spatialIndex: 0,
                    spatial: laneSp,
                    into: ropeKBuf,
                    channels: kBufSize
                )
            } catch {
                throw ANEError.invalidArguments("Llama lineage helper surface read failed: \(error)")
            }

            if let norms = layerQKNormWeights[layerIndex] {
                norms.q.withUnsafeBufferPointer { weights in
                    RMSNorm.applyPerHeadSingleTokenInPlace(
                        values: ropeQBuf.baseAddress!,
                        headCount: config.nHead,
                        headDim: config.headDim,
                        weights: weights.baseAddress!,
                        epsilon: Float(config.normEps)
                    )
                }
                norms.k.withUnsafeBufferPointer { weights in
                    RMSNorm.applyPerHeadSingleTokenInPlace(
                        values: ropeKBuf.baseAddress!,
                        headCount: config.nKVHead,
                        headDim: config.headDim,
                        weights: weights.baseAddress!,
                        epsilon: Float(config.normEps)
                    )
                }
            }

            RoPE.applyDecodeStep(
                q: ropeQBuf.baseAddress!,
                k: ropeKBuf.baseAddress!,
                nHeads: config.nHead,
                nKVHeads: config.nKVHead,
                headDim: config.headDim,
                position: tokenIndex,
                theta: config.ropeTheta
            )

            do {
                try SurfaceIO.writeFP16SpatialSlice(
                    to: qSurf,
                    channelOffset: 0,
                    spatialIndex: 0,
                    spatial: laneSp,
                    data: UnsafeBufferPointer(ropeQBuf),
                    channels: qBufSize
                )
                try SurfaceIO.writeFP16SpatialSlice(
                    to: kSurf,
                    channelOffset: 0,
                    spatialIndex: 0,
                    spatial: laneSp,
                    data: UnsafeBufferPointer(ropeKBuf),
                    channels: kBufSize
                )
            } catch {
                throw ANEError.invalidArguments("Llama lineage helper surface write failed: \(error)")
            }
        }

        ForwardPass.initializeHybridDecodeCaches(surfaceHandles: handles, dim: config.dModel)

        for (position, token) in tokens.enumerated() {
            writeTestingIncrementalEmbedding(
                config: config,
                token: token,
                position: position,
                tokenEmbedding: tokenEmbedding,
                positionEmbedding: [],
                into: xCur
            )
            var timings = HybridDecodeTimingBreakdown()
            try ForwardPass.runHybridDecodeTimed(
                xCur: xCur,
                kernels: kernels,
                surfaceHandles: handles,
                metalAttention: metalAttention,
                decodeState: &decodeState,
                dim: config.dModel,
                nHeads: config.nHead,
                nKVHeads: config.nKVHead,
                headDim: config.headDim,
                preferCPUDecodeAttention: Self.prefersCPUDecodeAttention(
                    config: config,
                    environment: ProcessInfo.processInfo.environment
                ),
                postQKVHook: ropeHook,
                timings: &timings
            )
        }

        let layerHiddenStates = try handles.map { handle in
            var hidden = [Float](repeating: 0, count: config.dModel)
            try hidden.withUnsafeMutableBufferPointer { buffer in
                try SurfaceIO.readFP16SpatialSlice(
                    from: handle.ffnOut,
                    channelOffset: 0,
                    spatialIndex: 0,
                    spatial: handle.laneSpatial,
                    into: buffer,
                    channels: config.dModel
                )
            }
            return hidden
        }

        return LayerHiddenLineageTestingOutputs(layerHiddenStates: layerHiddenStates)
    }

    static func evalHybridSingleLlamaLayerFromInputsForTesting(
        config: MultiModelConfig,
        weightDir: String,
        layer: Int,
        inputs: [[Float]]
    ) throws -> [Float] {
        let outputs = try evalHybridSingleLlamaLayerOutputsFromInputsForTesting(
            config: config,
            weightDir: weightDir,
            layer: layer,
            inputs: inputs
        )
        guard let last = outputs.last else {
            throw RealModelInferenceError.invalidGenerationParameters(
                "Testing input list must not be empty"
            )
        }
        return last
    }

    static func evalHybridSingleLlamaLayerOutputsFromInputsForTesting(
        config: MultiModelConfig,
        weightDir: String,
        layer: Int,
        inputs: [[Float]]
    ) throws -> [[Float]] {
        guard config.architecture == .llama else {
            throw RealModelInferenceError.unsupportedArchitecture(
                "Single-layer llama input helper currently supports llama-family artifacts only"
            )
        }
        let weightDirURL = URL(fileURLWithPath: weightDir, isDirectory: true)
        try validateDirectory(weightDirURL)
        guard !inputs.isEmpty else {
            throw RealModelInferenceError.invalidGenerationParameters("Testing input list must not be empty")
        }
        guard inputs.count <= config.maxSeq else {
            throw RealModelInferenceError.invalidGenerationParameters(
                "Testing input count \(inputs.count) exceeds context \(config.maxSeq)"
            )
        }
        for input in inputs {
            guard input.count == config.dModel else {
                throw RealModelInferenceError.invalidGenerationParameters(
                    "Testing input count \(input.count) must equal dModel \(config.dModel)"
                )
            }
        }

        let paths = LayerWeightPaths.forLayer(layer, config: config, blobDir: weightDirURL.path)
        let weights = try loadHybridLayerWeightsLlama(config: config, paths: paths)
        let qkNormWeights = try loadLlamaQKNormWeights(config: config, paths: paths)
        let maxSeq = max(inputs.count, 1)
        let kernels = try LayerStorage<HybridDecodeKernelSet>(count: 1, throwingInitializer: { _ in
            try HybridDecodeKernelSet(weights: weights, maxSeq: maxSeq)
        })
        let handles = [try HybridDecodeSurfaceHandles(
            kernels: kernels[0],
            logicalMaxSeq: maxSeq,
            dim: config.dModel,
            qDim: config.attentionDim,
            kvDim: config.kvDim
        )]
        let metalAttention = try MetalAttentionKernel()
        let xCur = TensorBuffer(count: config.dModel, zeroed: true)
        var decodeState = try DecodeState(maxSeq: maxSeq)

        let qBufSize = config.attentionDim
        let kBufSize = config.kvDim
        let ropeQBuf = UnsafeMutableBufferPointer<Float>.allocate(capacity: qBufSize)
        let ropeKBuf = UnsafeMutableBufferPointer<Float>.allocate(capacity: kBufSize)
        defer {
            ropeQBuf.deallocate()
            ropeKBuf.deallocate()
        }

        let ropeHook: (Int, IOSurfaceRef, IOSurfaceRef, Int, Int) throws -> Void = { _, qSurf, kSurf, laneSp, tokenIndex in
            do {
                try SurfaceIO.readFP16SpatialSlice(
                    from: qSurf,
                    channelOffset: 0,
                    spatialIndex: 0,
                    spatial: laneSp,
                    into: ropeQBuf,
                    channels: qBufSize
                )
                try SurfaceIO.readFP16SpatialSlice(
                    from: kSurf,
                    channelOffset: 0,
                    spatialIndex: 0,
                    spatial: laneSp,
                    into: ropeKBuf,
                    channels: kBufSize
                )
            } catch {
                throw ANEError.invalidArguments("Single-layer llama input helper surface read failed: \(error)")
            }

            if let qkNormWeights {
                qkNormWeights.q.withUnsafeBufferPointer { weights in
                    RMSNorm.applyPerHeadSingleTokenInPlace(
                        values: ropeQBuf.baseAddress!,
                        headCount: config.nHead,
                        headDim: config.headDim,
                        weights: weights.baseAddress!,
                        epsilon: Float(config.normEps)
                    )
                }
                qkNormWeights.k.withUnsafeBufferPointer { weights in
                    RMSNorm.applyPerHeadSingleTokenInPlace(
                        values: ropeKBuf.baseAddress!,
                        headCount: config.nKVHead,
                        headDim: config.headDim,
                        weights: weights.baseAddress!,
                        epsilon: Float(config.normEps)
                    )
                }
            }

            RoPE.applyDecodeStep(
                q: ropeQBuf.baseAddress!,
                k: ropeKBuf.baseAddress!,
                nHeads: config.nHead,
                nKVHeads: config.nKVHead,
                headDim: config.headDim,
                position: tokenIndex,
                theta: config.ropeTheta
            )

            do {
                try SurfaceIO.writeFP16SpatialSlice(
                    to: qSurf,
                    channelOffset: 0,
                    spatialIndex: 0,
                    spatial: laneSp,
                    data: UnsafeBufferPointer(ropeQBuf),
                    channels: qBufSize
                )
                try SurfaceIO.writeFP16SpatialSlice(
                    to: kSurf,
                    channelOffset: 0,
                    spatialIndex: 0,
                    spatial: laneSp,
                    data: UnsafeBufferPointer(ropeKBuf),
                    channels: kBufSize
                )
            } catch {
                throw ANEError.invalidArguments("Single-layer llama input helper surface write failed: \(error)")
            }
        }

        ForwardPass.initializeHybridDecodeCaches(surfaceHandles: handles, dim: config.dModel)

        var outputs: [[Float]] = []
        outputs.reserveCapacity(inputs.count)
        for input in inputs {
            xCur.withUnsafeMutableBufferPointer { dst in
                for index in 0..<config.dModel {
                    dst[index] = input[index]
                }
            }
            var timings = HybridDecodeTimingBreakdown()
            try ForwardPass.runHybridDecodeTimed(
                xCur: xCur,
                kernels: kernels,
                surfaceHandles: handles,
                metalAttention: metalAttention,
                decodeState: &decodeState,
                dim: config.dModel,
                nHeads: config.nHead,
                nKVHeads: config.nKVHead,
                headDim: config.headDim,
                preferCPUDecodeAttention: Self.prefersCPUDecodeAttention(
                    config: config,
                    environment: ProcessInfo.processInfo.environment
                ),
                postQKVHook: ropeHook,
                timings: &timings
            )
            outputs.append(xCur.withUnsafeBufferPointer { Array($0) })
        }

        return outputs
    }

    static func evalHybridSingleLlamaLayerDetailedFromInputsForTesting(
        config: MultiModelConfig,
        weightDir: String,
        layer: Int,
        inputs: [[Float]]
    ) throws -> SingleLayerDetailedTestingOutputs {
        guard config.architecture == .llama else {
            throw RealModelInferenceError.unsupportedArchitecture(
                "Single-layer llama detailed helper currently supports llama-family artifacts only"
            )
        }
        let hidden = try evalHybridSingleLlamaLayerFromInputsForTesting(
            config: config,
            weightDir: weightDir,
            layer: layer,
            inputs: inputs
        )

        let weightDirURL = URL(fileURLWithPath: weightDir, isDirectory: true)
        let paths = LayerWeightPaths.forLayer(layer, config: config, blobDir: weightDirURL.path)
        let weights = try loadHybridLayerWeightsLlama(config: config, paths: paths)
        let qkNormWeights = try loadLlamaQKNormWeights(config: config, paths: paths)
        let maxSeq = max(inputs.count, 1)
        let kernels = try LayerStorage<HybridDecodeKernelSet>(count: 1, throwingInitializer: { _ in
            try HybridDecodeKernelSet(weights: weights, maxSeq: maxSeq)
        })
        let handles = [try HybridDecodeSurfaceHandles(
            kernels: kernels[0],
            logicalMaxSeq: maxSeq,
            dim: config.dModel,
            qDim: config.attentionDim,
            kvDim: config.kvDim
        )]
        let metalAttention = try MetalAttentionKernel()
        let xCur = TensorBuffer(count: config.dModel, zeroed: true)
        var decodeState = try DecodeState(maxSeq: maxSeq)

        let qBufSize = config.attentionDim
        let kBufSize = config.kvDim
        let ropeQBuf = UnsafeMutableBufferPointer<Float>.allocate(capacity: qBufSize)
        let ropeKBuf = UnsafeMutableBufferPointer<Float>.allocate(capacity: kBufSize)
        defer {
            ropeQBuf.deallocate()
            ropeKBuf.deallocate()
        }

        let ropeHook: (Int, IOSurfaceRef, IOSurfaceRef, Int, Int) throws -> Void = { _, qSurf, kSurf, laneSp, tokenIndex in
            do {
                try SurfaceIO.readFP16SpatialSlice(
                    from: qSurf,
                    channelOffset: 0,
                    spatialIndex: 0,
                    spatial: laneSp,
                    into: ropeQBuf,
                    channels: qBufSize
                )
                try SurfaceIO.readFP16SpatialSlice(
                    from: kSurf,
                    channelOffset: 0,
                    spatialIndex: 0,
                    spatial: laneSp,
                    into: ropeKBuf,
                    channels: kBufSize
                )
            } catch {
                throw ANEError.invalidArguments("Single-layer llama detailed helper surface read failed: \(error)")
            }

            if let qkNormWeights {
                qkNormWeights.q.withUnsafeBufferPointer { weights in
                    RMSNorm.applyPerHeadSingleTokenInPlace(
                        values: ropeQBuf.baseAddress!,
                        headCount: config.nHead,
                        headDim: config.headDim,
                        weights: weights.baseAddress!,
                        epsilon: Float(config.normEps)
                    )
                }
                qkNormWeights.k.withUnsafeBufferPointer { weights in
                    RMSNorm.applyPerHeadSingleTokenInPlace(
                        values: ropeKBuf.baseAddress!,
                        headCount: config.nKVHead,
                        headDim: config.headDim,
                        weights: weights.baseAddress!,
                        epsilon: Float(config.normEps)
                    )
                }
            }

            RoPE.applyDecodeStep(
                q: ropeQBuf.baseAddress!,
                k: ropeKBuf.baseAddress!,
                nHeads: config.nHead,
                nKVHeads: config.nKVHead,
                headDim: config.headDim,
                position: tokenIndex,
                theta: config.ropeTheta
            )

            do {
                try SurfaceIO.writeFP16SpatialSlice(
                    to: qSurf,
                    channelOffset: 0,
                    spatialIndex: 0,
                    spatial: laneSp,
                    data: UnsafeBufferPointer(ropeQBuf),
                    channels: qBufSize
                )
                try SurfaceIO.writeFP16SpatialSlice(
                    to: kSurf,
                    channelOffset: 0,
                    spatialIndex: 0,
                    spatial: laneSp,
                    data: UnsafeBufferPointer(ropeKBuf),
                    channels: kBufSize
                )
            } catch {
                throw ANEError.invalidArguments("Single-layer llama detailed helper surface write failed: \(error)")
            }
        }

        ForwardPass.initializeHybridDecodeCaches(surfaceHandles: handles, dim: config.dModel)

        for input in inputs {
            xCur.withUnsafeMutableBufferPointer { dst in
                for index in 0..<config.dModel {
                    dst[index] = input[index]
                }
            }
            var timings = HybridDecodeTimingBreakdown()
            try ForwardPass.runHybridDecodeTimed(
                xCur: xCur,
                kernels: kernels,
                surfaceHandles: handles,
                metalAttention: metalAttention,
                decodeState: &decodeState,
                dim: config.dModel,
                nHeads: config.nHead,
                nKVHeads: config.nKVHead,
                headDim: config.headDim,
                preferCPUDecodeAttention: Self.prefersCPUDecodeAttention(
                    config: config,
                    environment: ProcessInfo.processInfo.environment
                ),
                postQKVHook: ropeHook,
                timings: &timings
            )
        }

        var context = [Float](repeating: 0, count: config.attentionDim)
        try context.withUnsafeMutableBufferPointer { buffer in
            try readFP32SpatialSlice(
                from: handles[0].projectionContextIn,
                spatialIndex: 0,
                spatial: handles[0].laneSpatial,
                into: buffer,
                channels: config.attentionDim
            )
        }

        var projectionOut = [Float](repeating: 0, count: config.dModel)
        try projectionOut.withUnsafeMutableBufferPointer { buffer in
            try SurfaceIO.readFP16SpatialSlice(
                from: handles[0].projectionOut,
                channelOffset: 0,
                spatialIndex: 0,
                spatial: handles[0].laneSpatial,
                into: buffer,
                channels: config.dModel
            )
        }

        var qOut = [Float](repeating: 0, count: config.attentionDim)
        try qOut.withUnsafeMutableBufferPointer { buffer in
            try SurfaceIO.readFP16SpatialSlice(
                from: handles[0].qOut,
                channelOffset: 0,
                spatialIndex: 0,
                spatial: handles[0].laneSpatial,
                into: buffer,
                channels: config.attentionDim
            )
        }

        var kCache = [Float](repeating: 0, count: config.kvDim * maxSeq)
        var vCache = [Float](repeating: 0, count: config.kvDim * maxSeq)
        kCache.withUnsafeMutableBufferPointer { buffer in
            SurfaceIO.readFP16(
                from: handles[0].kCacheFull,
                into: buffer,
                channelOffset: 0,
                channels: config.kvDim,
                spatial: maxSeq
            )
        }
        vCache.withUnsafeMutableBufferPointer { buffer in
            SurfaceIO.readFP16(
                from: handles[0].vCacheFull,
                into: buffer,
                channelOffset: 0,
                channels: config.kvDim,
                spatial: maxSeq
            )
        }

        return SingleLayerDetailedTestingOutputs(
            hidden: hidden,
            context: context,
            projectionOut: projectionOut,
            qOut: qOut,
            kCache: kCache,
            vCache: vCache
        )
    }

    static func compileHeadForTesting(
        config: MultiModelConfig,
        weightDir: String
    ) throws {
        let weightDirURL = URL(fileURLWithPath: weightDir, isDirectory: true)
        try validateDirectory(weightDirURL)
        let topLevelPaths = try resolveTopLevelWeightPaths(config: config, weightDir: weightDir)
        let finalNormGamma = try loadWeightTablePreferringFloat32Sidecar(
            at: topLevelPaths.finalNormGamma,
            expectedCount: config.dModel
        )
        let finalNormBeta = try loadWeightTable(at: topLevelPaths.finalNormBeta, expectedCount: config.dModel)
        let assets = GPT2TopLevelAssets(
            tokenEmbedding: [],
            positionEmbedding: [],
            finalNormGamma: finalNormGamma,
            finalNormBeta: finalNormBeta,
            lmHead: [],
            finalNormGammaPath: topLevelPaths.finalNormGamma,
            finalNormBetaPath: topLevelPaths.finalNormBeta,
            finalNormGammaCompilePath: compileBlobPath(actualPath: topLevelPaths.finalNormGamma, rootDir: weightDirURL),
            finalNormBetaCompilePath: compileBlobPath(actualPath: topLevelPaths.finalNormBeta, rootDir: weightDirURL),
            finalNormGammaData: WeightBlob.build(from: finalNormGamma, rows: 1, cols: finalNormGamma.count),
            finalNormBetaData: WeightBlob.build(from: finalNormBeta, rows: 1, cols: finalNormBeta.count)
        )
        let spatial = try compileBucket(for: config.maxSeq, channels: config.dModel, maxSeq: config.maxSeq)
        _ = try compileHead(
            config: config,
            weightDirURL: weightDirURL,
            assets: assets,
            spatial: spatial
        )
    }

    static func requireANEHardwareTestsEnabled() throws {
        guard ProcessInfo.processInfo.environment["ANE_HARDWARE_TESTS"] == "1" else {
            throw RealModelInferenceError.runtimeFailure("Set ANE_HARDWARE_TESTS=1 to run ANE hardware tests")
        }
        let handle = dlopen(
            "/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine",
            RTLD_NOW
        )
        guard handle != nil else {
            throw RealModelInferenceError.runtimeFailure("AppleNeuralEngine.framework unavailable")
        }
        dlclose(handle)
        ane_interop_init()
    }

    private mutating func ensureCompiled(bucket: Int) throws -> Bool {
        guard compiledBucket < bucket else {
            return false
        }

        let newLayers = try Self.compileLayers(
            config: config,
            weightDirURL: weightDirURL,
            bucket: bucket
        )
        let newInputSurface = try Self.firstInputSurface(from: newLayers)
        let newHead = try LayerStorage<CompiledHead>(count: 1, throwingInitializer: { _ in
            try Self.compileHead(
                config: config,
                weightDirURL: weightDirURL,
                assets: gpt2Assets,
                spatial: bucket
            )
        })
        do {
            try newHead[0].kernel.rebindInput(at: 0, to: newLayers[newLayers.count - 1].outputSurface)
        } catch {
            throw RealModelInferenceError.runtimeFailure("Failed to chain final norm input: \(error)")
        }
        compiledLayers = newLayers
        compiledHead = newHead
        firstLayerInputSurface = newInputSurface
        compiledBucket = bucket
        return true
    }

    private mutating func ensureHybridCompiled(bucket: Int) throws -> Bool {
        var didCompile = false

        if compiledHybridBucket < bucket {
            let newLayers = try Self.compileHybridLayers(
                config: config,
                weightDirURL: weightDirURL,
                maxSeq: bucket
            )
            let newQKNormWeights = try Self.loadHybridLlamaQKNormWeights(
                config: config,
                weightDirURL: weightDirURL
            )
            var newSurfaceHandles: [HybridDecodeSurfaceHandles] = []
            newSurfaceHandles.reserveCapacity(newLayers.count)
            for layerIndex in 0..<newLayers.count {
                do {
                    newSurfaceHandles.append(
                        try HybridDecodeSurfaceHandles(
                            kernels: newLayers[layerIndex],
                            logicalMaxSeq: bucket
                        )
                    )
                } catch {
                    throw RealModelInferenceError.runtimeFailure(
                        "Hybrid decode surfaces unavailable for layer \(layerIndex): \(error)"
                    )
                }
            }
            if newLayers.count > 1,
               Self.usesHybridLayerInputRebinding(
                   architecture: config.architecture,
                   environment: ProcessInfo.processInfo.environment
               ) {
                for layerIndex in 1..<newLayers.count {
                    do {
                        try newLayers[layerIndex].decodeQKVOnly.rebindInput(
                            at: 0,
                            to: newSurfaceHandles[layerIndex - 1].ffnOut
                        )
                    } catch {
                        throw RealModelInferenceError.runtimeFailure(
                            "Hybrid decode chaining unavailable for layer \(layerIndex): \(error)"
                        )
                    }
                }
            }

            compiledHybridLayers = newLayers
            compiledHybridSurfaceHandles = newSurfaceHandles
            compiledHybridLlamaQKNormWeights = newQKNormWeights
            compiledHybridBucket = bucket
            didCompile = true
        }

        if compiledHybridLlamaQKNormWeights.count != config.nLayer {
            compiledHybridLlamaQKNormWeights = try Self.loadHybridLlamaQKNormWeights(
                config: config,
                weightDirURL: weightDirURL
            )
        }

        if hybridMetalAttention == nil {
            do {
                hybridMetalAttention = try MetalAttentionKernel()
            } catch {
                throw RealModelInferenceError.runtimeFailure("Hybrid Metal attention initialization failed: \(error)")
            }
            didCompile = true
        }

        let hybridHeadSpatial = Self.incrementalHeadSpatial(channels: config.dModel)
        if compiledHybridHead.count != 1 || compiledHybridHeadSpatial != hybridHeadSpatial {
            compiledHybridHead = try LayerStorage<CompiledHead>(count: 1, throwingInitializer: { _ in
                try Self.compileHead(
                    config: config,
                    weightDirURL: weightDirURL,
                    assets: gpt2Assets,
                    spatial: hybridHeadSpatial
                )
            })
            compiledHybridHeadSpatial = hybridHeadSpatial
            try Self.zeroSurface(compiledHybridHead[0].inputSurface)
            didCompile = true
        }

        if classifierStrategy.usesANEClassifier {
            if compiledHybridGreedyNorm.count != 1 ||
                compiledHybridGreedyClassifier.count != 1 ||
                compiledHybridGreedySpatial != hybridHeadSpatial {
                compiledHybridGreedyNorm = try LayerStorage<CompiledHead>(count: 1, throwingInitializer: { _ in
                    try Self.compileHead(
                        config: config,
                        weightDirURL: weightDirURL,
                        assets: gpt2Assets,
                        spatial: hybridHeadSpatial,
                        inputDType: .fp16,
                        outputDType: .fp16
                    )
                })
                compiledHybridGreedyClassifier = try LayerStorage<CompiledClassifier>(count: 1, throwingInitializer: { _ in
                    try Self.compileClassifier(
                        config: config,
                        assets: gpt2Assets,
                        spatial: hybridHeadSpatial
                    )
                })
                compiledHybridGreedySpatial = hybridHeadSpatial
                try compiledHybridGreedyClassifier[0].kernel.rebindInput(
                    at: 0,
                    to: compiledHybridGreedyNorm[0].outputSurface
                )
                didCompile = true
            }

            if compiledHybridGreedyNorm.count == 1,
               compiledHybridGreedyClassifier.count == 1,
               let finalSurface = compiledHybridSurfaceHandles.last?.ffnOut {
                try compiledHybridGreedyNorm[0].kernel.rebindInput(at: 0, to: finalSurface)
                try compiledHybridGreedyClassifier[0].kernel.rebindInput(
                    at: 0,
                    to: compiledHybridGreedyNorm[0].outputSurface
                )
            }
        }

        return didCompile
    }

    private mutating func ensureHybridCompiledLlama(bucket: Int) throws -> Bool {
        var didCompile = false

        if compiledHybridBucket < bucket {
            let newLayers = try Self.compileHybridLayers(
                config: config,
                weightDirURL: weightDirURL,
                maxSeq: bucket
            )
            let newQKNormWeights = try Self.loadHybridLlamaQKNormWeights(
                config: config,
                weightDirURL: weightDirURL
            )
            var newSurfaceHandles: [HybridDecodeSurfaceHandles] = []
            newSurfaceHandles.reserveCapacity(newLayers.count)
            for layerIndex in 0..<newLayers.count {
                do {
                    newSurfaceHandles.append(
                        try HybridDecodeSurfaceHandles(
                            kernels: newLayers[layerIndex],
                            logicalMaxSeq: bucket,
                            dim: config.dModel,
                            qDim: config.attentionDim,
                            kvDim: config.nKVHead * config.headDim
                        )
                    )
                } catch {
                    throw RealModelInferenceError.runtimeFailure(
                        "Llama hybrid decode surfaces unavailable for layer \(layerIndex): \(error)"
                    )
                }
            }
            if newLayers.count > 1,
               Self.usesHybridLayerInputRebinding(
                   architecture: config.architecture,
                   environment: ProcessInfo.processInfo.environment
               ) {
                for layerIndex in 1..<newLayers.count {
                    do {
                        try newLayers[layerIndex].decodeQKVOnly.rebindInput(
                            at: 0,
                            to: newSurfaceHandles[layerIndex - 1].ffnOut
                        )
                    } catch {
                        throw RealModelInferenceError.runtimeFailure(
                            "Llama hybrid decode chaining unavailable for layer \(layerIndex): \(error)"
                        )
                    }
                }
            }

            compiledHybridLayers = newLayers
            compiledHybridSurfaceHandles = newSurfaceHandles
            compiledHybridLlamaQKNormWeights = newQKNormWeights
            compiledHybridBucket = bucket
            didCompile = true
        }

        if compiledHybridLlamaQKNormWeights.count != config.nLayer {
            compiledHybridLlamaQKNormWeights = try Self.loadHybridLlamaQKNormWeights(
                config: config,
                weightDirURL: weightDirURL
            )
        }

        if hybridMetalAttention == nil {
            do {
                hybridMetalAttention = try MetalAttentionKernel()
            } catch {
                throw RealModelInferenceError.runtimeFailure("Llama hybrid Metal attention initialization failed: \(error)")
            }
            didCompile = true
        }

        let hybridHeadSpatial = Self.incrementalHeadSpatial(channels: config.dModel)
        let greedyHeadMode = hybridGreedyHeadMode()

        // Compile RMSNorm head (no beta) for llama
        if compiledHybridHead.count != 1 || compiledHybridHeadSpatial != hybridHeadSpatial {
            compiledHybridHead = try LayerStorage<CompiledHead>(count: 1, throwingInitializer: { _ in
                try Self.compileLlamaHead(
                    config: config,
                    weightDirURL: weightDirURL,
                    assets: llamaAssets,
                    spatial: hybridHeadSpatial
                )
            })
            compiledHybridHeadSpatial = hybridHeadSpatial
            try Self.zeroSurface(compiledHybridHead[0].inputSurface)
            didCompile = true
        }

        if classifierStrategy.usesANEClassifier {
            switch greedyHeadMode {
            case .classifierOnlyFactored:
                if compiledHybridGreedyNorm.count != 0 {
                    compiledHybridGreedyNorm = Self.emptyStorage(CompiledHead.self)
                    didCompile = true
                }
                if compiledHybridGreedyClassifier.count != 1 {
                    do {
                        compiledHybridGreedyClassifier = try LayerStorage<CompiledClassifier>(count: 1, throwingInitializer: { _ in
                            try Self.compileLlamaFactoredClassifier(
                                config: config,
                                assets: llamaAssets,
                                spatial: hybridHeadSpatial
                            )
                        })
                    } catch {
                        fputs(
                            "[RealModelInference] Llama factored classifier compile failed; falling back to dense ANE classifier: \(error)\n",
                            stderr
                        )
                        compiledHybridGreedyClassifier = Self.emptyStorage(CompiledClassifier.self)
                    }
                    didCompile = true
                }
                if compiledHybridGreedyClassifier.count == 1,
                   let finalSurface = compiledHybridSurfaceHandles.last?.ffnOut {
                    try compiledHybridGreedyClassifier[0].kernel.rebindInput(at: 0, to: finalSurface)
                }
            case .classifierOnlyFused:
                if compiledHybridGreedyNorm.count != 0 {
                    compiledHybridGreedyNorm = Self.emptyStorage(CompiledHead.self)
                    didCompile = true
                }
                if compiledHybridGreedyClassifier.count != 1 {
                    compiledHybridGreedyClassifier = try LayerStorage<CompiledClassifier>(count: 1, throwingInitializer: { _ in
                        try Self.compileLlamaRMSNormClassifier(
                            config: config,
                            assets: llamaAssets,
                            spatial: hybridHeadSpatial
                        )
                    })
                    didCompile = true
                }
                if compiledHybridGreedyClassifier.count == 1,
                   let finalSurface = compiledHybridSurfaceHandles.last?.ffnOut {
                    try compiledHybridGreedyClassifier[0].kernel.rebindInput(at: 0, to: finalSurface)
                }
            case .normThenClassifier:
                if compiledHybridGreedyNorm.count != 1 || compiledHybridGreedySpatial != hybridHeadSpatial {
                    compiledHybridGreedyNorm = try LayerStorage<CompiledHead>(count: 1, throwingInitializer: { _ in
                        try Self.compileLlamaHead(
                            config: config,
                            weightDirURL: weightDirURL,
                            assets: llamaAssets,
                            spatial: hybridHeadSpatial,
                            inputDType: .fp16,
                            outputDType: .fp16
                        )
                    })
                    compiledHybridGreedySpatial = hybridHeadSpatial
                    didCompile = true
                }

                if compiledHybridGreedyClassifier.count != 1 {
                    compiledHybridGreedyClassifier = try LayerStorage<CompiledClassifier>(count: 1, throwingInitializer: { _ in
                        try Self.compileLlamaClassifier(
                            config: config,
                            assets: llamaAssets,
                            spatial: hybridHeadSpatial
                        )
                    })
                    try compiledHybridGreedyClassifier[0].kernel.rebindInput(
                        at: 0,
                        to: compiledHybridGreedyNorm[0].outputSurface
                    )
                    didCompile = true
                }

                if compiledHybridGreedyClassifier.count == 1 {
                    try compiledHybridGreedyClassifier[0].kernel.rebindInput(
                        at: 0,
                        to: compiledHybridGreedyNorm[0].outputSurface
                    )
                }
            }
        }

        if compiledHybridGreedyNorm.count == 1,
           let finalSurface = compiledHybridSurfaceHandles.last?.ffnOut {
            try compiledHybridGreedyNorm[0].kernel.rebindInput(at: 0, to: finalSurface)
        }

        return didCompile
    }

    private static func loadHybridLlamaQKNormWeights(
        config: MultiModelConfig,
        weightDirURL: URL
    ) throws -> [LlamaQKNormWeights?] {
        try (0..<config.nLayer).map { layerIndex in
            let paths = LayerWeightPaths.forLayer(layerIndex, config: config, blobDir: weightDirURL.path)
            return try Self.loadLlamaQKNormWeights(config: config, paths: paths)
        }
    }

    private mutating func generateIncrementalHybrid(
        promptTokens: [TokenID],
        effectiveMaxTokens: Int,
        temperature: Float,
        compileTimeMs: Double,
        maxSeq: Int,
        metalAttention: MetalAttentionKernel,
        onStep: ((GenerationStep) -> Void)?
    ) throws -> GenerationResult {
        guard compiledHybridLayers.count == config.nLayer,
              compiledHybridSurfaceHandles.count == config.nLayer,
              compiledHybridHead.count == 1,
              compiledHybridHeadSpatial > 0 else {
            throw RealModelInferenceError.runtimeFailure("Hybrid decode state is unavailable")
        }

        ForwardPass.initializeHybridDecodeCaches(
            surfaceHandles: compiledHybridSurfaceHandles,
            dim: config.dModel
        )

        // Pre-create cached Metal bindings for all layers (GPT-2 path)
        let cachedBindings: [MetalAttentionKernel.CachedLayerBindings]? = Self.supportsHybridCachedBindings(
            config: config,
            environment: ProcessInfo.processInfo.environment
        ) ? {
            var bindings: [MetalAttentionKernel.CachedLayerBindings] = []
            bindings.reserveCapacity(compiledHybridSurfaceHandles.count)
            for handles in compiledHybridSurfaceHandles {
                do {
                    let binding = try metalAttention.createCachedLayerBindings(
                        qSurface: handles.qOut,
                        kOutputSurface: handles.kOut,
                        vOutputSurface: handles.vOut,
                        kCacheSurface: handles.kCacheFull,
                        vCacheSurface: handles.vCacheFull,
                        contextSurface: handles.projectionContextIn,
                        dim: handles.qDim,
                        kvDim: handles.kvDim,
                        laneStride: handles.laneSpatial,
                        cacheStride: maxSeq
                    )
                    bindings.append(binding)
                } catch {
                    return nil
                }
            }
            return bindings
        }() : nil

        if ProcessInfo.processInfo.environment["ESPRESSO_REALMODEL_DEBUG_HYBRID_CACHE"] == "1",
           let firstHandles = compiledHybridSurfaceHandles.first {
            fputs(
                "[hybrid-surface] qkvIn_row=\(IOSurfaceGetBytesPerRow(firstHandles.qkvIn)) qOut_row=\(IOSurfaceGetBytesPerRow(firstHandles.qOut)) ffnIn_row=\(IOSurfaceGetBytesPerRow(firstHandles.ffnIn)) ffnOut_row=\(IOSurfaceGetBytesPerRow(firstHandles.ffnOut)) laneSpatial=\(firstHandles.laneSpatial) maxSeq=\(firstHandles.maxSeq)\n",
                stderr
            )
        }
        let shouldDebugHybridCache = ProcessInfo.processInfo.environment["ESPRESSO_REALMODEL_DEBUG_HYBRID_CACHE"] == "1"

        let xCur = TensorBuffer(count: config.dModel, zeroed: true)
        var decodeState: DecodeState
        do {
            decodeState = try DecodeState(maxSeq: maxSeq)
        } catch {
            throw RealModelInferenceError.runtimeFailure("Hybrid decode state initialization failed: \(error)")
        }
        var timings = HybridDecodeTimingBreakdown()
        let greedyHeadMode = hybridGreedyHeadMode()
        let useANEGreedyHead =
            temperature == 0 &&
            classifierStrategy.usesANEClassifier &&
            compiledHybridGreedyClassifier.count == 1 &&
            (greedyHeadMode == .normThenClassifier
                ? compiledHybridGreedyNorm.count == 1
                : compiledHybridGreedyNorm.count == 0)

        for (position, token) in promptTokens.enumerated() {
            try writeIncrementalEmbedding(token: token, position: position, into: xCur)
            let debugInput: [Float]?
            if shouldDebugHybridCache, position < 2 {
                debugInput = xCur.withUnsafeBufferPointer { Array($0) }
            } else {
                debugInput = nil
            }
            do {
                try ForwardPass.runHybridDecodeTimed(
                    xCur: xCur,
                    kernels: compiledHybridLayers,
                    surfaceHandles: compiledHybridSurfaceHandles,
                    metalAttention: metalAttention,
                    decodeState: &decodeState,
                    dim: config.dModel,
                    preferCPUDecodeAttention: Self.prefersCPUDecodeAttention(
                        config: config,
                        environment: ProcessInfo.processInfo.environment
                    ),
                    readFinalOutputIntoXCur: !useANEGreedyHead,
                    cachedBindings: cachedBindings,
                    timings: &timings
                )
            } catch {
                throw RealModelInferenceError.runtimeFailure(
                    "Hybrid prefill failed at prompt position \(position): \(error)"
                )
            }
            if shouldDebugHybridCache,
               position < 2,
               let firstHandles = compiledHybridSurfaceHandles.first {
                try Self.debugLogHybridCache(
                    label: "prefill_\(position)",
                    surface: firstHandles.kCacheFull,
                    maxSeq: maxSeq,
                    channels: min(8, config.dModel),
                    tokenCount: min(position + 1, 2)
                )
                if let debugInput {
                    let layer0Paths = LayerWeightPaths.forLayer(0, config: config, blobDir: weightDirURL.path)
                    let debugLayer0Weights = try Self.loadHybridLayerWeights(config: config, paths: layer0Paths)
                    let expectedK = Self.debugExpectedGPT2KPrefix(
                        input: debugInput,
                        weights: debugLayer0Weights,
                        eps: config.normEps,
                        prefixChannels: min(8, config.dModel)
                    )
                    let expectedKTransposed = Self.debugExpectedGPT2KPrefixTransposed(
                        input: debugInput,
                        weights: debugLayer0Weights,
                        eps: config.normEps,
                        prefixChannels: min(8, config.dModel)
                    )
                    let values = expectedK.map { String(format: "%.4f", $0) }.joined(separator: ",")
                    let transposedValues = expectedKTransposed.map { String(format: "%.4f", $0) }.joined(separator: ",")
                    fputs("[hybrid-kref] prefill_\(position) [\(values)]\n", stderr)
                    fputs("[hybrid-kref-t] prefill_\(position) [\(transposedValues)]\n", stderr)
                }
            }
        }

        var allTokens = promptTokens
        var generatedTokens: [TokenID] = []
        var tokenLatenciesMs: [Double] = []
        generatedTokens.reserveCapacity(effectiveMaxTokens)
        tokenLatenciesMs.reserveCapacity(effectiveMaxTokens)

        let generationStart = DispatchTime.now().uptimeNanoseconds
        var emissionStart = generationStart
        var firstTokenLatencyMs = 0.0
        var firstTokenRecorded = false
        var rng = SystemRandomNumberGenerator()
        var normalized = [Float](repeating: 0, count: config.dModel)
        let headSpatial = compiledHybridHeadSpatial

        while generatedTokens.count < effectiveMaxTokens {
            let nextToken: TokenID
            if useANEGreedyHead {
                do {
                    if greedyHeadMode == .normThenClassifier {
                        try compiledHybridGreedyNorm[0].kernel.eval()
                    }
                    try compiledHybridGreedyClassifier[0].kernel.eval()
                    let argmax = try Self.greedyArgmax(
                        classifier: compiledHybridGreedyClassifier[0],
                        headSpatial: headSpatial,
                        vocab: config.vocab
                    )
                    guard let token = TokenID(exactly: argmax.index) else {
                        throw RealModelInferenceError.runtimeFailure(
                            "Greedy ANE classifier selected out-of-range token \(argmax.index)"
                        )
                    }
                    nextToken = token
                } catch let error as RealModelInferenceError {
                    throw error
                } catch {
                    throw RealModelInferenceError.runtimeFailure("Hybrid greedy ANE head evaluation failed: \(error)")
                }
            } else {
                do {
                    try xCur.withUnsafeBufferPointer { buffer in
                        try Self.writeFP32SpatialSlice(
                            to: compiledHybridHead[0].inputSurface,
                            spatialIndex: 0,
                            spatial: headSpatial,
                            data: buffer,
                            channels: config.dModel
                        )
                    }
                    try compiledHybridHead[0].kernel.eval()
                    try normalized.withUnsafeMutableBufferPointer { buffer in
                        try Self.readFP32SpatialSlice(
                            from: compiledHybridHead[0].outputSurface,
                            spatialIndex: 0,
                            spatial: headSpatial,
                            into: buffer,
                            channels: config.dModel
                        )
                    }
                } catch {
                    throw RealModelInferenceError.runtimeFailure("Hybrid step head evaluation failed: \(error)")
                }

                nextToken = selectTokenFromNormalizedHidden(
                    normalized,
                    temperature: temperature,
                    using: &rng
                )
            }
            let emissionNow = DispatchTime.now().uptimeNanoseconds
            let tokenLatencyMs = Self.milliseconds(from: emissionNow - emissionStart)

            if !firstTokenRecorded {
                firstTokenLatencyMs = Self.milliseconds(from: emissionNow - generationStart)
                firstTokenRecorded = true
            }

            if nextToken == Self.gpt2EOSToken {
                break
            }

            generatedTokens.append(nextToken)
            allTokens.append(nextToken)
            let elapsedMs = Self.milliseconds(from: emissionNow - generationStart)
            let tokensPerSecond = Double(generatedTokens.count) / max(elapsedMs / 1_000, 1e-9)
            tokenLatenciesMs.append(tokenLatencyMs)
            onStep?(
                GenerationStep(
                    token: nextToken,
                    generatedTokens: generatedTokens,
                    text: tokenizer.decode(allTokens.map(Int.init)),
                    tokenLatencyMs: tokenLatencyMs,
                    elapsedMs: elapsedMs,
                    firstTokenLatencyMs: firstTokenLatencyMs,
                    tokensPerSecond: tokensPerSecond
                )
            )

            if generatedTokens.count >= effectiveMaxTokens || allTokens.count >= config.maxSeq {
                break
            }

            try writeIncrementalEmbedding(token: nextToken, position: allTokens.count - 1, into: xCur)
            do {
                try ForwardPass.runHybridDecodeTimed(
                    xCur: xCur,
                    kernels: compiledHybridLayers,
                    surfaceHandles: compiledHybridSurfaceHandles,
                    metalAttention: metalAttention,
                    decodeState: &decodeState,
                    dim: config.dModel,
                    preferCPUDecodeAttention: Self.prefersCPUDecodeAttention(
                        config: config,
                        environment: ProcessInfo.processInfo.environment
                    ),
                    readFinalOutputIntoXCur: !useANEGreedyHead,
                    cachedBindings: cachedBindings,
                    timings: &timings
                )
            } catch {
                throw RealModelInferenceError.runtimeFailure(
                    "Hybrid decode failed at generated token \(generatedTokens.count - 1): \(error)"
                )
            }
            emissionStart = emissionNow
        }

        let generationEnd = DispatchTime.now().uptimeNanoseconds
        let generationTimeMs = Self.milliseconds(from: generationEnd - generationStart)
        let tokensPerSecond = generatedTokens.isEmpty
            ? 0
            : Double(generatedTokens.count) / max(generationTimeMs / 1_000, 1e-9)

        return GenerationResult(
            text: tokenizer.decode(allTokens.map(Int.init)),
            tokens: generatedTokens,
            promptTokens: promptTokens,
            tokenLatenciesMs: tokenLatenciesMs,
            tokensPerSecond: tokensPerSecond,
            compileTimeMs: compileTimeMs,
            firstTokenLatencyMs: firstTokenLatencyMs,
            exactHeadBackend: classifierStrategy.exactHeadBackendLabel,
            cachedBindingsEnabled: false
        )
    }

    private mutating func generateIncrementalHybridSpeculative(
        promptTokens: [TokenID],
        effectiveMaxTokens: Int,
        compileTimeMs: Double,
        metalAttention: MetalAttentionKernel,
        cachedRuntimePair: CachedSpeculativeRuntimePair,
        onStep: ((GenerationStep) -> Void)?
    ) throws -> GenerationResult {
        cachedRuntimePair.resetAll(dim: config.dModel)

        let xCur = TensorBuffer(count: config.dModel, zeroed: true)
        for (position, token) in promptTokens.enumerated() {
            try writeIncrementalEmbedding(token: token, position: position, into: xCur)
            do {
                try cachedRuntimePair.draftRuntime.advanceFromBuffer(
                    xCur,
                    metalAttention: metalAttention,
                    dim: config.dModel
                )
                try cachedRuntimePair.verifierRuntime.advanceFromSurface(
                    cachedRuntimePair.draftRuntime.finalSurface,
                    metalAttention: metalAttention,
                    dim: config.dModel
                )
            } catch {
                throw RealModelInferenceError.runtimeFailure(
                    "Hybrid speculative prefill failed at prompt position \(position): \(error)"
                )
            }
        }

        var allTokens = promptTokens
        var generatedTokens: [TokenID] = []
        var tokenLatenciesMs: [Double] = []
        generatedTokens.reserveCapacity(effectiveMaxTokens)
        tokenLatenciesMs.reserveCapacity(effectiveMaxTokens)

        let generationStart = DispatchTime.now().uptimeNanoseconds
        var emissionStart = generationStart
        var firstTokenLatencyMs = 0.0
        var firstTokenRecorded = false

        func emitToken(_ token: TokenID, at emissionNow: UInt64) {
            generatedTokens.append(token)
            allTokens.append(token)
            let elapsedMs = Self.milliseconds(from: emissionNow - generationStart)
            let tokenLatencyMs = Self.milliseconds(from: emissionNow - emissionStart)
            tokenLatenciesMs.append(tokenLatencyMs)
            if !firstTokenRecorded {
                firstTokenLatencyMs = Self.milliseconds(from: emissionNow - generationStart)
                firstTokenRecorded = true
            }
            let tokensPerSecond = Double(generatedTokens.count) / max(elapsedMs / 1_000, 1e-9)
            onStep?(
                GenerationStep(
                    token: token,
                    generatedTokens: generatedTokens,
                    text: tokenizer.decode(allTokens.map(Int.init)),
                    tokenLatencyMs: tokenLatencyMs,
                    elapsedMs: elapsedMs,
                    firstTokenLatencyMs: firstTokenLatencyMs,
                    tokensPerSecond: tokensPerSecond
                )
            )
        }

        while generatedTokens.count < effectiveMaxTokens {
            let checkpoint = try cachedRuntimePair.draftRuntime.captureCheckpoint(dim: config.dModel)
            let proposedToken0: TokenID
            do {
                proposedToken0 = try cachedRuntimePair.draftRuntime.selectGreedyToken(vocab: config.vocab)
                try writeIncrementalEmbedding(token: proposedToken0, position: allTokens.count, into: xCur)
                try cachedRuntimePair.draftRuntime.advanceFromBuffer(
                    xCur,
                    metalAttention: metalAttention,
                    dim: config.dModel
                )
            } catch {
                throw RealModelInferenceError.runtimeFailure(
                    "Hybrid speculative draft proposal-0 failed at generated token \(generatedTokens.count): \(error)"
                )
            }

            let proposedToken1: TokenID
            do {
                proposedToken1 = try cachedRuntimePair.draftRuntime.selectGreedyToken(vocab: config.vocab)
            } catch {
                throw RealModelInferenceError.runtimeFailure(
                    "Hybrid speculative draft proposal-1 failed at generated token \(generatedTokens.count): \(error)"
                )
            }

            let exactToken0: TokenID
            do {
                exactToken0 = try cachedRuntimePair.verifierRuntime.selectGreedyToken(vocab: config.vocab)
            } catch {
                throw RealModelInferenceError.runtimeFailure(
                    "Hybrid speculative verifier token-0 failed at generated token \(generatedTokens.count): \(error)"
                )
            }
            if exactToken0 == Self.gpt2EOSToken {
                break
            }

            if exactToken0 != proposedToken0 {
                do {
                    try cachedRuntimePair.draftRuntime.rollback(
                        to: checkpoint,
                        mutatedTokenCount: 1,
                        dim: config.dModel
                    )
                    try writeIncrementalEmbedding(token: exactToken0, position: allTokens.count, into: xCur)
                    try cachedRuntimePair.draftRuntime.advanceFromBuffer(
                        xCur,
                        metalAttention: metalAttention,
                        dim: config.dModel
                    )
                    try cachedRuntimePair.verifierRuntime.advanceFromSurface(
                        cachedRuntimePair.draftRuntime.finalSurface,
                        metalAttention: metalAttention,
                        dim: config.dModel
                    )
                } catch {
                    throw RealModelInferenceError.runtimeFailure(
                        "Hybrid speculative verifier rollback failed at generated token \(generatedTokens.count): \(error)"
                    )
                }

                let emissionNow = DispatchTime.now().uptimeNanoseconds
                emitToken(exactToken0, at: emissionNow)
                emissionStart = emissionNow
                if generatedTokens.count >= effectiveMaxTokens || allTokens.count >= config.maxSeq {
                    break
                }
                continue
            }

            do {
                try cachedRuntimePair.verifierRuntime.advanceFromSurface(
                    cachedRuntimePair.draftRuntime.finalSurface,
                    metalAttention: metalAttention,
                    dim: config.dModel
                )
            } catch {
                throw RealModelInferenceError.runtimeFailure(
                    "Hybrid speculative verifier promotion failed at generated token \(generatedTokens.count): \(error)"
                )
            }

            let emissionAfterFirst = DispatchTime.now().uptimeNanoseconds
            emitToken(exactToken0, at: emissionAfterFirst)
            emissionStart = emissionAfterFirst

            if generatedTokens.count >= effectiveMaxTokens || allTokens.count >= config.maxSeq {
                break
            }

            let exactToken1: TokenID
            do {
                exactToken1 = try cachedRuntimePair.verifierRuntime.selectGreedyToken(vocab: config.vocab)
            } catch {
                throw RealModelInferenceError.runtimeFailure(
                    "Hybrid speculative verifier token-1 failed at generated token \(generatedTokens.count): \(error)"
                )
            }
            if exactToken1 == Self.gpt2EOSToken {
                break
            }

            do {
                let committedSecondToken = exactToken1 == proposedToken1 ? proposedToken1 : exactToken1
                try writeIncrementalEmbedding(token: committedSecondToken, position: allTokens.count, into: xCur)
                try cachedRuntimePair.draftRuntime.advanceFromBuffer(
                    xCur,
                    metalAttention: metalAttention,
                    dim: config.dModel
                )
                try cachedRuntimePair.verifierRuntime.advanceFromSurface(
                    cachedRuntimePair.draftRuntime.finalSurface,
                    metalAttention: metalAttention,
                    dim: config.dModel
                )
            } catch {
                throw RealModelInferenceError.runtimeFailure(
                    "Hybrid speculative commit failed at generated token \(generatedTokens.count): \(error)"
                )
            }

            let emissionAfterSecond = DispatchTime.now().uptimeNanoseconds
            emitToken(exactToken1, at: emissionAfterSecond)
            emissionStart = emissionAfterSecond

            if allTokens.count >= config.maxSeq {
                break
            }
        }

        let generationEnd = DispatchTime.now().uptimeNanoseconds
        let generationTimeMs = Self.milliseconds(from: generationEnd - generationStart)
        let tokensPerSecond = generatedTokens.isEmpty
            ? 0
            : Double(generatedTokens.count) / max(generationTimeMs / 1_000, 1e-9)

        return GenerationResult(
            text: tokenizer.decode(allTokens.map(Int.init)),
            tokens: generatedTokens,
            promptTokens: promptTokens,
            tokenLatenciesMs: tokenLatenciesMs,
            tokensPerSecond: tokensPerSecond,
            compileTimeMs: compileTimeMs,
            firstTokenLatencyMs: firstTokenLatencyMs
        )
    }

    private mutating func cachedSpeculativeRuntimePair(
        draftLayerCount: Int,
        maxSeq: Int
    ) throws -> (CachedSpeculativeRuntimePair, Double) {
        let key = SpeculativeRuntimeKey(
            draftLayerCount: draftLayerCount,
            maxSeq: maxSeq
        )
        if let cached = speculativeRuntimeCache[key] {
            let orderUpdate = Self.boundedSpeculativeCacheOrder(
                currentOrder: speculativeRuntimeCacheOrder,
                accessedKey: key,
                limit: Self.speculativeRuntimeCacheLimit,
                insertingNewEntry: false
            )
            speculativeRuntimeCacheOrder = orderUpdate.order
            return (cached, 0)
        }

        let compileStart = DispatchTime.now().uptimeNanoseconds
        let cached = try CachedSpeculativeRuntimePair(
            key: key,
            config: config,
            weightDirURL: weightDirURL,
            assets: gpt2Assets
        )
        let orderUpdate = Self.boundedSpeculativeCacheOrder(
            currentOrder: speculativeRuntimeCacheOrder,
            accessedKey: key,
            limit: Self.speculativeRuntimeCacheLimit,
            insertingNewEntry: true
        )
        if let evictedKey = orderUpdate.evictedKey {
            speculativeRuntimeCache.removeValue(forKey: evictedKey)
        }
        speculativeRuntimeCache[key] = cached
        speculativeRuntimeCacheOrder = orderUpdate.order
        let compileTimeMs = Self.milliseconds(from: DispatchTime.now().uptimeNanoseconds - compileStart)
        return (cached, compileTimeMs)
    }

    static func boundedSpeculativeCacheOrder<Key: Equatable>(
        currentOrder: [Key],
        accessedKey: Key,
        limit: Int,
        insertingNewEntry: Bool
    ) -> (order: [Key], evictedKey: Key?) {
        precondition(limit > 0)

        var order = currentOrder.filter { $0 != accessedKey }
        var evictedKey: Key?
        if insertingNewEntry, order.count >= limit {
            evictedKey = order.removeFirst()
        }
        order.append(accessedKey)
        return (order, evictedKey)
    }

    private static func loadConfigFromMetadataFile(at metadataURL: URL) throws -> MultiModelConfig {
        let data: Data
        do {
            data = try Data(contentsOf: metadataURL)
        } catch {
            throw RealModelInferenceError.runtimeFailure("Failed to read metadata.json: \(error)")
        }

        let object: Any
        do {
            object = try JSONSerialization.jsonObject(with: data)
        } catch {
            throw RealModelInferenceError.runtimeFailure("metadata.json is not valid JSON: \(error)")
        }

        guard let metadata = object as? [String: Any] else {
            throw RealModelInferenceError.runtimeFailure("metadata.json must be a JSON object")
        }

        func requiredInt(_ key: String) throws -> Int {
            guard let number = metadata[key] as? NSNumber else {
                throw RealModelInferenceError.runtimeFailure("metadata.json missing numeric field \(key)")
            }
            return number.intValue
        }

        func requiredDouble(_ key: String) throws -> Double {
            guard let number = metadata[key] as? NSNumber else {
                throw RealModelInferenceError.runtimeFailure("metadata.json missing numeric field \(key)")
            }
            return number.doubleValue
        }

        guard let name = metadata["name"] as? String, !name.isEmpty else {
            throw RealModelInferenceError.runtimeFailure("metadata.json missing string field name")
        }
        let architecture: MultiModelConfig.Architecture
        switch (metadata["architecture"] as? String)?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() {
        case "gpt2":
            architecture = .gpt2
        case "llama":
            architecture = .llama
        default:
            throw RealModelInferenceError.runtimeFailure("metadata.json missing supported architecture")
        }

        return MultiModelConfig(
            name: name,
            nLayer: try requiredInt("nLayer"),
            nHead: try requiredInt("nHead"),
            nKVHead: try requiredInt("nKVHead"),
            dModel: try requiredInt("dModel"),
            headDim: try requiredInt("headDim"),
            hiddenDim: try requiredInt("hiddenDim"),
            vocab: try requiredInt("vocab"),
            maxSeq: try requiredInt("maxSeq"),
            normEps: Float(try requiredDouble("normEps")),
            ropeTheta: Float((metadata["ropeTheta"] as? NSNumber)?.doubleValue ?? 10_000),
            eosToken: (metadata["eosToken"] as? NSNumber)?.uint32Value,
            architecture: architecture
        )
    }

    private static func resolveExactTwoTokenDraft(
        config: MultiModelConfig,
        weightDirURL: URL,
        environment: [String: String]
    ) throws -> ResolvedExactTwoTokenDraft? {
        guard environment["ESPRESSO_BUNDLE_DRAFT_KIND"] == "exact_two_token" else {
            return nil
        }
        if let rawHorizon = environment["ESPRESSO_BUNDLE_DRAFT_HORIZON"],
           Int(rawHorizon) != 2 {
            throw RealModelInferenceError.runtimeFailure(
                "exact two-token draft requires horizon == 2, got \(rawHorizon)"
            )
        }
        guard let artifactRef = environment["ESPRESSO_BUNDLE_DRAFT_ARTIFACT_REF"],
              !artifactRef.isEmpty else {
            throw RealModelInferenceError.runtimeFailure("exact two-token draft requires ESPRESSO_BUNDLE_DRAFT_ARTIFACT_REF")
        }

        let bundleRootURL = weightDirURL.deletingLastPathComponent()
        let descriptorURL = bundleRootURL.appendingPathComponent(artifactRef).standardizedFileURL
        let bundleRootPath = bundleRootURL.path
        guard descriptorURL.path == bundleRootPath || descriptorURL.path.hasPrefix(bundleRootPath + "/") else {
            throw RealModelInferenceError.runtimeFailure("Draft artifact ref escapes bundle root: \(artifactRef)")
        }
        guard FileManager.default.fileExists(atPath: descriptorURL.path) else {
            throw RealModelInferenceError.runtimeFailure("Draft artifact file is missing: \(descriptorURL.path)")
        }

        let descriptorData = try Data(contentsOf: descriptorURL)
        let descriptor: ExactTwoTokenDraftDescriptor
        do {
            descriptor = try JSONDecoder().decode(ExactTwoTokenDraftDescriptor.self, from: descriptorData)
        } catch {
            throw RealModelInferenceError.runtimeFailure("Failed to decode draft descriptor \(descriptorURL.path): \(error)")
        }
        guard !descriptor.modelDir.isEmpty else {
            throw RealModelInferenceError.runtimeFailure("Draft descriptor is missing model_dir")
        }

        let draftWeightDirURL = weightDirURL.appendingPathComponent(
            descriptor.modelDir,
            isDirectory: true
        ).standardizedFileURL
        let weightRootPath = weightDirURL.path
        guard draftWeightDirURL.path == weightRootPath || draftWeightDirURL.path.hasPrefix(weightRootPath + "/") else {
            throw RealModelInferenceError.runtimeFailure("Draft model_dir escapes weights root: \(descriptor.modelDir)")
        }
        try validateDirectory(draftWeightDirURL)
        let draftConfig = try loadConfigFromMetadataFile(
            at: draftWeightDirURL.appendingPathComponent("metadata.json")
        )
        guard draftConfig.architecture == .llama else {
            throw RealModelInferenceError.runtimeFailure("exact two-token draft currently supports llama draft models only")
        }
        guard draftConfig.vocab == config.vocab else {
            throw RealModelInferenceError.runtimeFailure(
                "draft/full vocab mismatch: draft=\(draftConfig.vocab) full=\(config.vocab)"
            )
        }
        return ResolvedExactTwoTokenDraft(
            descriptor: descriptor,
            descriptorURL: descriptorURL,
            weightDirURL: draftWeightDirURL,
            config: draftConfig
        )
    }

    static func resolveExactTwoTokenDraftWeightDirForTesting(
        config: MultiModelConfig,
        weightDirURL: URL,
        environment: [String: String]
    ) throws -> String? {
        try resolveExactTwoTokenDraft(
            config: config,
            weightDirURL: weightDirURL,
            environment: environment
        )?.weightDirURL.path
    }

    private func encodePrompt(_ prompt: String) throws -> [TokenID] {
        let rawTokens = tokenizer.encode(prompt)
        guard !rawTokens.isEmpty else {
            throw RealModelInferenceError.invalidPrompt("Prompt produced no tokens")
        }
        var tokens: [TokenID] = []
        tokens.reserveCapacity(rawTokens.count)
        for token in rawTokens {
            guard token >= 0, token <= Int(TokenID.max) else {
                throw RealModelInferenceError.invalidPrompt("Token \(token) does not fit TokenID")
            }
            tokens.append(TokenID(token))
        }
        return tokens
    }

    private func composeEmbeddingInput(tokens: [TokenID], spatial: Int) -> [Float] {
        var output = [Float](repeating: 0, count: config.dModel * spatial)
        for tokenIndex in 0..<tokens.count {
            let token = Int(tokens[tokenIndex])
            let tokenBase = token * config.dModel
            let positionBase = tokenIndex * config.dModel
            for channel in 0..<config.dModel {
                output[channel * spatial + tokenIndex] =
                    gpt2Assets.tokenEmbedding[tokenBase + channel] +
                    gpt2Assets.positionEmbedding[positionBase + channel]
            }
        }
        return output
    }

    private func writeIncrementalEmbedding(
        token: TokenID,
        position: Int,
        into buffer: borrowing TensorBuffer
    ) throws {
        guard position >= 0, position < config.maxSeq else {
            throw RealModelInferenceError.runtimeFailure("Position \(position) exceeds context \(config.maxSeq)")
        }

        let tokenBase = Int(token) * config.dModel
        let positionBase = position * config.dModel
        buffer.withUnsafeMutableBufferPointer { dst in
            for channel in 0..<config.dModel {
                dst[channel] =
                    gpt2Assets.tokenEmbedding[tokenBase + channel] +
                    gpt2Assets.positionEmbedding[positionBase + channel]
            }
        }
    }

    private func writeIncrementalEmbeddingLlama(
        token: TokenID,
        into buffer: borrowing TensorBuffer
    ) throws {
        let tokenBase = Int(token) * config.dModel
        guard tokenBase + config.dModel <= llamaAssets.tokenEmbedding.count else {
            throw RealModelInferenceError.runtimeFailure(
                "Llama embedding OOB: token=\(token), base=\(tokenBase), embeddingCount=\(llamaAssets.tokenEmbedding.count), dModel=\(config.dModel)"
            )
        }
        buffer.withUnsafeMutableBufferPointer { dst in
            for channel in 0..<config.dModel {
                dst[channel] = llamaAssets.tokenEmbedding[tokenBase + channel]
            }
        }
    }

    private mutating func generateIncrementalHybridLlama(
        promptTokens: [TokenID],
        effectiveMaxTokens: Int,
        temperature: Float,
        compileTimeMs: Double,
        maxSeq: Int,
        metalAttention: MetalAttentionKernel,
        onStep: ((GenerationStep) -> Void)?
    ) throws -> GenerationResult {
        guard compiledHybridLayers.count == config.nLayer,
              compiledHybridSurfaceHandles.count == config.nLayer,
              compiledHybridLlamaQKNormWeights.count == config.nLayer,
              compiledHybridHead.count == 1,
              compiledHybridHeadSpatial > 0 else {
            throw RealModelInferenceError.runtimeFailure(
                "Llama hybrid decode state is unavailable: layers=\(compiledHybridLayers.count)/\(config.nLayer) surfaces=\(compiledHybridSurfaceHandles.count)/\(config.nLayer) qkNorms=\(compiledHybridLlamaQKNormWeights.count)/\(config.nLayer) head=\(compiledHybridHead.count) headSpatial=\(compiledHybridHeadSpatial)"
            )
        }

        if Self.llamaGenerationPath(
            config: config,
            environment: ProcessInfo.processInfo.environment
        ) == .exactCPU {
            return try generateIncrementalExactCPULlama(
                promptTokens: promptTokens,
                effectiveMaxTokens: effectiveMaxTokens,
                temperature: temperature,
                compileTimeMs: compileTimeMs,
                maxSeq: maxSeq,
                onStep: onStep
            )
        }

        ForwardPass.initializeHybridDecodeCaches(
            surfaceHandles: compiledHybridSurfaceHandles,
            dim: config.dModel
        )

        // Pre-create cached decode-surface metadata for all layers.
        let cachedBindings: [MetalAttentionKernel.CachedLayerBindings]? = Self.supportsHybridCachedBindings(
            config: config,
            environment: ProcessInfo.processInfo.environment
        ) ? {
            var bindings: [MetalAttentionKernel.CachedLayerBindings] = []
            bindings.reserveCapacity(compiledHybridSurfaceHandles.count)
            for handles in compiledHybridSurfaceHandles {
                do {
                    let binding = try metalAttention.createCachedLayerBindings(
                        qSurface: handles.qOut,
                        kOutputSurface: handles.kOut,
                        vOutputSurface: handles.vOut,
                        kCacheSurface: handles.kCacheFull,
                        vCacheSurface: handles.vCacheFull,
                        contextSurface: handles.projectionContextIn,
                        dim: handles.qDim,
                        kvDim: handles.kvDim,
                        laneStride: handles.laneSpatial,
                        cacheStride: maxSeq
                    )
                    bindings.append(binding)
                } catch {
                    return nil
                }
            }
            return bindings
        }() : nil

        let xCur = TensorBuffer(count: config.dModel, zeroed: true)
        var decodeState: DecodeState
        do {
            decodeState = try DecodeState(maxSeq: maxSeq)
        } catch {
            throw RealModelInferenceError.runtimeFailure("Llama hybrid decode state initialization failed: \(error)")
        }
        var timings = HybridDecodeTimingBreakdown()
        let greedyHeadMode = hybridGreedyHeadMode()
        let useANEGreedyHead =
            temperature == 0 &&
            classifierStrategy.usesANEClassifier &&
            compiledHybridGreedyClassifier.count == 1 &&
            (greedyHeadMode == .normThenClassifier
                ? compiledHybridGreedyNorm.count == 1
                : compiledHybridGreedyNorm.count == 0)

        let useCPUExactGreedyHead =
            temperature == 0 &&
            classifierStrategy.usesCPUExactClassifier

        // Build the RoPE hook closure that rotates Q and K between ANE QKV eval and Metal attention
        let nHeads = config.nHead
        let nKVHeads = config.nKVHead
        let headDim = config.headDim
        let qDim = config.attentionDim
        let theta = config.ropeTheta
        let layerQKNormWeights = compiledHybridLlamaQKNormWeights
        let normEps = Float(config.normEps)
        let hasAnyQKNorm = layerQKNormWeights.contains { $0 != nil }

        let qBufSize = qDim
        let kBufSize = nKVHeads * headDim
        // Pre-allocate RoPE scratch buffers once, reused across all layers and tokens.
        let ropeQBuf = UnsafeMutableBufferPointer<Float>.allocate(capacity: qBufSize)
        let ropeKBuf = UnsafeMutableBufferPointer<Float>.allocate(capacity: kBufSize)
        defer { ropeQBuf.deallocate() }
        defer { ropeKBuf.deallocate() }
        func applyRoPEHook(
            layerIndex: Int,
            qSurf: IOSurfaceRef,
            kSurf: IOSurfaceRef,
            laneSp: Int,
            tokenIndex: Int
        ) throws(ANEError) {
            do {
                try SurfaceIO.readFP16SpatialSlice(
                    from: qSurf, channelOffset: 0, spatialIndex: 0, spatial: laneSp,
                    into: ropeQBuf, channels: qBufSize
                )
                try SurfaceIO.readFP16SpatialSlice(
                    from: kSurf, channelOffset: 0, spatialIndex: 0, spatial: laneSp,
                    into: ropeKBuf, channels: kBufSize
                )
            } catch {
                throw ANEError.invalidArguments("RoPE hook surface read failed: \(error)")
            }

            if let norms = layerQKNormWeights[layerIndex] {
                norms.q.withUnsafeBufferPointer { weights in
                    RMSNorm.applyPerHeadSingleTokenInPlace(
                        values: ropeQBuf.baseAddress!,
                        headCount: nHeads,
                        headDim: headDim,
                        weights: weights.baseAddress!,
                        epsilon: normEps
                    )
                }
                norms.k.withUnsafeBufferPointer { weights in
                    RMSNorm.applyPerHeadSingleTokenInPlace(
                        values: ropeKBuf.baseAddress!,
                        headCount: nKVHeads,
                        headDim: headDim,
                        weights: weights.baseAddress!,
                        epsilon: normEps
                    )
                }
            }

            RoPE.applyDecodeStep(
                q: ropeQBuf.baseAddress!,
                k: ropeKBuf.baseAddress!,
                nHeads: nHeads,
                nKVHeads: nKVHeads,
                headDim: headDim,
                position: tokenIndex,
                theta: theta
            )

            do {
                try SurfaceIO.writeFP16SpatialSlice(
                    to: qSurf, channelOffset: 0, spatialIndex: 0, spatial: laneSp,
                    data: UnsafeBufferPointer(ropeQBuf), channels: qBufSize
                )
                try SurfaceIO.writeFP16SpatialSlice(
                    to: kSurf, channelOffset: 0, spatialIndex: 0, spatial: laneSp,
                    data: UnsafeBufferPointer(ropeKBuf), channels: kBufSize
                )
            } catch {
                throw ANEError.invalidArguments("RoPE hook surface write failed: \(error)")
            }
        }
        let ropeHook: (Int, IOSurfaceRef, IOSurfaceRef, Int, Int) throws -> Void = { layerIndex, qSurf, kSurf, laneSp, tokenIndex in
            try applyRoPEHook(
                layerIndex: layerIndex,
                qSurf: qSurf,
                kSurf: kSurf,
                laneSp: laneSp,
                tokenIndex: tokenIndex
            )
        }

        let metalRoPEConfig: MetalAttentionKernel.MetalRoPEConfig? = Self.supportsLlamaMetalRoPEFastPath(
            cachedBindingsAvailable: cachedBindings != nil && !hasAnyQKNorm,
            kBindingContainsKVCache: false
        )
            ? MetalAttentionKernel.MetalRoPEConfig(
                nHeads: nHeads, nKVHeads: nKVHeads, headDim: headDim, theta: theta
            )
            : nil
        let decodeDModel = config.dModel
        let decodeQDim = config.attentionDim
        let decodeKVDim = config.kvDim

        let useCPUExactQKV = Self.prefersCPUExactQKV(
            config: config,
            environment: ProcessInfo.processInfo.environment
        )
        let cpuExactQKVLayerWeights: [LlamaCPUQKVWeights]? = if useCPUExactQKV {
            try (0..<config.nLayer).map { layerIndex in
                let paths = LayerWeightPaths.forLayer(layerIndex, config: config, blobDir: weightDirURL.path)
                return try Self.loadLlamaCPUQKVWeights(config: config, paths: paths)
            }
        } else {
            nil
        }
        let cpuQKVHiddenBuf = useCPUExactQKV
            ? UnsafeMutableBufferPointer<Float>.allocate(capacity: decodeDModel)
            : nil
        let cpuQKVAttnNormedBuf = useCPUExactQKV
            ? UnsafeMutableBufferPointer<Float>.allocate(capacity: decodeDModel)
            : nil
        let cpuQBuf = useCPUExactQKV
            ? UnsafeMutableBufferPointer<Float>.allocate(capacity: decodeQDim)
            : nil
        let cpuKBuf = useCPUExactQKV
            ? UnsafeMutableBufferPointer<Float>.allocate(capacity: decodeKVDim)
            : nil
        let cpuVBuf = useCPUExactQKV
            ? UnsafeMutableBufferPointer<Float>.allocate(capacity: decodeKVDim)
            : nil
        defer { cpuQKVHiddenBuf?.deallocate() }
        defer { cpuQKVAttnNormedBuf?.deallocate() }
        defer { cpuQBuf?.deallocate() }
        defer { cpuKBuf?.deallocate() }
        defer { cpuVBuf?.deallocate() }
        let cpuExactQKV: ((Int, HybridDecodeSurfaceHandles, Int, Int) throws -> Void)? = if let layerWeights = cpuExactQKVLayerWeights,
                                                                                           let hiddenBuf = cpuQKVHiddenBuf,
                                                                                           let attnNormedBuf = cpuQKVAttnNormedBuf,
                                                                                           let qBuf = cpuQBuf,
                                                                                           let kBuf = cpuKBuf,
                                                                                           let vBuf = cpuVBuf {
            { layerIndex, handles, laneSp, _ in
                let weights = layerWeights[layerIndex]
                do {
                    try SurfaceIO.readFP16SpatialSlice(
                        from: handles.qkvIn,
                        channelOffset: 0,
                        spatialIndex: 0,
                        spatial: laneSp,
                        into: hiddenBuf,
                        channels: decodeDModel
                    )
                } catch {
                    throw ANEError.invalidArguments("CPU exact QKV input read failed: \(error)")
                }

                var sumSq: Float = 0
                vDSP_dotpr(hiddenBuf.baseAddress!, 1, hiddenBuf.baseAddress!, 1, &sumSq, vDSP_Length(decodeDModel))
                var invRms = 1.0 / sqrtf(sumSq / Float(decodeDModel) + normEps)
                vDSP_vsmul(hiddenBuf.baseAddress!, 1, &invRms, attnNormedBuf.baseAddress!, 1, vDSP_Length(decodeDModel))
                weights.rmsAtt.withUnsafeBufferPointer { gamma in
                    vDSP_vmul(attnNormedBuf.baseAddress!, 1, gamma.baseAddress!, 1, attnNormedBuf.baseAddress!, 1, vDSP_Length(decodeDModel))
                }

                Self.multiplyRowMajorMatrix(
                    matrix: weights.wq,
                    rows: decodeQDim,
                    cols: decodeDModel,
                    vector: UnsafeBufferPointer(attnNormedBuf),
                    into: qBuf
                )
                Self.multiplyRowMajorMatrix(
                    matrix: weights.wk,
                    rows: decodeKVDim,
                    cols: decodeDModel,
                    vector: UnsafeBufferPointer(attnNormedBuf),
                    into: kBuf
                )
                Self.multiplyRowMajorMatrix(
                    matrix: weights.wv,
                    rows: decodeKVDim,
                    cols: decodeDModel,
                    vector: UnsafeBufferPointer(attnNormedBuf),
                    into: vBuf
                )

                do {
                    try SurfaceIO.writeFP16SpatialSlice(
                        to: handles.qOut,
                        channelOffset: 0,
                        spatialIndex: 0,
                        spatial: laneSp,
                        data: UnsafeBufferPointer(qBuf),
                        channels: decodeQDim
                    )
                    try SurfaceIO.writeFP16SpatialSlice(
                        to: handles.kOut,
                        channelOffset: 0,
                        spatialIndex: 0,
                        spatial: laneSp,
                        data: UnsafeBufferPointer(kBuf),
                        channels: decodeKVDim
                    )
                    try SurfaceIO.writeFP16SpatialSlice(
                        to: handles.vOut,
                        channelOffset: 0,
                        spatialIndex: 0,
                        spatial: laneSp,
                        data: UnsafeBufferPointer(vBuf),
                        channels: decodeKVDim
                    )
                } catch {
                    throw ANEError.invalidArguments("CPU exact QKV surface write failed: \(error)")
                }
            }
        } else {
            nil
        }

        // Prefill: process prompt tokens
        for (position, token) in promptTokens.enumerated() {
            try writeIncrementalEmbeddingLlama(token: token, into: xCur)
            do {
                try ForwardPass.runHybridDecodeTimed(
                    xCur: xCur,
                    kernels: compiledHybridLayers,
                    surfaceHandles: compiledHybridSurfaceHandles,
                    metalAttention: metalAttention,
                    decodeState: &decodeState,
                    dim: config.dModel,
                    nHeads: nHeads,
                    nKVHeads: nKVHeads,
                    headDim: headDim,
                    preferCPUDecodeAttention: Self.prefersCPUDecodeAttention(
                        config: config,
                        environment: ProcessInfo.processInfo.environment
                    ),
                    qkvOverride: cpuExactQKV,
                    postQKVHook: metalRoPEConfig != nil ? nil : ropeHook,
                    readFinalOutputIntoXCur: !useANEGreedyHead,
                    cachedBindings: cachedBindings,
                    metalRoPEConfig: metalRoPEConfig,
                    timings: &timings
                )
            } catch {
                throw RealModelInferenceError.runtimeFailure(
                    "Llama hybrid prefill failed at prompt position \(position): \(error)"
                )
            }
        }

        var allTokens = promptTokens
        var generatedTokens: [TokenID] = []
        var tokenLatenciesMs: [Double] = []
        generatedTokens.reserveCapacity(effectiveMaxTokens)
        tokenLatenciesMs.reserveCapacity(effectiveMaxTokens)

        let generationStart = DispatchTime.now().uptimeNanoseconds
        var emissionStart = generationStart
        var firstTokenLatencyMs = 0.0
        var firstTokenRecorded = false
        var rng = SystemRandomNumberGenerator()
        var normalized = [Float](repeating: 0, count: config.dModel)
        let headSpatial = compiledHybridHeadSpatial

        while generatedTokens.count < effectiveMaxTokens {
            let nextToken: TokenID
            if useANEGreedyHead {
                do {
                    if greedyHeadMode != .normThenClassifier {
                        try compiledHybridGreedyClassifier[0].kernel.eval()
                    } else {
                        try compiledHybridGreedyNorm[0].kernel.eval()
                        try compiledHybridGreedyClassifier[0].kernel.eval()
                    }
                    let argmax = try Self.greedyArgmax(
                        classifier: compiledHybridGreedyClassifier[0],
                        headSpatial: headSpatial,
                        vocab: config.vocab
                    )
                    guard let token = TokenID(exactly: argmax.index) else {
                        throw RealModelInferenceError.runtimeFailure(
                            "Llama greedy ANE classifier selected out-of-range token \(argmax.index)"
                        )
                    }
                    nextToken = token
                } catch {
                    throw RealModelInferenceError.runtimeFailure("Llama hybrid greedy ANE head evaluation failed: \(error)")
                }
            } else if useCPUExactGreedyHead {
                normalized = xCur.withUnsafeBufferPointer {
                    Self.rmsNorm(Array($0), weight: llamaAssets.finalNormGamma, eps: Float(config.normEps))
                }
                nextToken = TokenID(exactClassifierArgmax(normalized))
            } else {
                do {
                    try xCur.withUnsafeBufferPointer { buffer in
                        try Self.writeFP32SpatialSlice(
                            to: compiledHybridHead[0].inputSurface,
                            spatialIndex: 0,
                            spatial: headSpatial,
                            data: buffer,
                            channels: config.dModel
                        )
                    }
                    try compiledHybridHead[0].kernel.eval()
                    try normalized.withUnsafeMutableBufferPointer { buffer in
                        try Self.readFP32SpatialSlice(
                            from: compiledHybridHead[0].outputSurface,
                            spatialIndex: 0,
                            spatial: headSpatial,
                            into: buffer,
                            channels: config.dModel
                        )
                    }
                } catch {
                    throw RealModelInferenceError.runtimeFailure("Llama hybrid step head evaluation failed: \(error)")
                }

                nextToken = selectTokenFromNormalizedHidden(
                    normalized,
                    temperature: temperature,
                    using: &rng
                )
            }
            let emissionNow = DispatchTime.now().uptimeNanoseconds
            let tokenLatencyMs = Self.milliseconds(from: emissionNow - emissionStart)

            if !firstTokenRecorded {
                firstTokenLatencyMs = Self.milliseconds(from: emissionNow - generationStart)
                firstTokenRecorded = true
            }

            if let eosToken = config.eosToken, nextToken == eosToken {
                generatedTokens.append(nextToken)
                break
            }
            generatedTokens.append(nextToken)
            allTokens.append(nextToken)
            let elapsedMs = Self.milliseconds(from: emissionNow - generationStart)
            let tokensPerSecond = Double(generatedTokens.count) / max(elapsedMs / 1_000, 1e-9)
            tokenLatenciesMs.append(tokenLatencyMs)
            onStep?(
                GenerationStep(
                    token: nextToken,
                    generatedTokens: generatedTokens,
                    text: tokenizer.decode(allTokens.map(Int.init)),
                    tokenLatencyMs: tokenLatencyMs,
                    elapsedMs: elapsedMs,
                    firstTokenLatencyMs: firstTokenLatencyMs,
                    tokensPerSecond: tokensPerSecond
                )
            )

            if generatedTokens.count >= effectiveMaxTokens || allTokens.count >= config.maxSeq {
                break
            }

            try writeIncrementalEmbeddingLlama(token: nextToken, into: xCur)
            do {
                if ProcessInfo.processInfo.environment["ESPRESSO_BATCHED_METAL_ATTENTION"] == "1" && metalRoPEConfig == nil && cpuExactQKV == nil {
                    try ForwardPass.runHybridDecodeBatchedMetalTimed(
                        xCur: xCur,
                        kernels: compiledHybridLayers,
                        surfaceHandles: compiledHybridSurfaceHandles,
                        metalAttention: metalAttention,
                        decodeState: &decodeState,
                        dim: config.dModel,
                        nHeads: nHeads,
                        nKVHeads: nKVHeads,
                        headDim: headDim,
                        postQKVHook: ropeHook,
                        cachedBindings: cachedBindings,
                        timings: &timings
                    )
                } else {
                    try ForwardPass.runHybridDecodeTimed(
                        xCur: xCur,
                        kernels: compiledHybridLayers,
                        surfaceHandles: compiledHybridSurfaceHandles,
                        metalAttention: metalAttention,
                        decodeState: &decodeState,
                        dim: config.dModel,
                        nHeads: nHeads,
                        nKVHeads: nKVHeads,
                        headDim: headDim,
                        preferCPUDecodeAttention: Self.prefersCPUDecodeAttention(
                            config: config,
                            environment: ProcessInfo.processInfo.environment
                        ),
                        qkvOverride: cpuExactQKV,
                        postQKVHook: metalRoPEConfig != nil ? nil : ropeHook,
                        readFinalOutputIntoXCur: !useANEGreedyHead,
                        cachedBindings: cachedBindings,
                        metalRoPEConfig: metalRoPEConfig,
                        timings: &timings
                    )
                }
            } catch {
                throw RealModelInferenceError.runtimeFailure(
                    "Llama hybrid decode failed at generated token \(generatedTokens.count - 1): \(error)"
                )
            }
            emissionStart = emissionNow
        }

        let generationEnd = DispatchTime.now().uptimeNanoseconds
        let generationTimeMs = Self.milliseconds(from: generationEnd - generationStart)
        let tokensPerSecond = generatedTokens.isEmpty
            ? 0
            : Double(generatedTokens.count) / max(generationTimeMs / 1_000, 1e-9)

        return GenerationResult(
            text: tokenizer.decode(allTokens.map(Int.init)),
            tokens: generatedTokens,
            promptTokens: promptTokens,
            tokenLatenciesMs: tokenLatenciesMs,
            tokensPerSecond: tokensPerSecond,
            compileTimeMs: compileTimeMs,
            firstTokenLatencyMs: firstTokenLatencyMs,
            exactHeadBackend: greedyHeadMode == .classifierOnlyFactored && useANEGreedyHead
                ? "ane_factored_classifier"
                : classifierStrategy.exactHeadBackendLabel,
            cachedBindingsEnabled: cachedBindings != nil
        )
    }

    private mutating func loadCachedExactCPULlamaWeights() throws -> CachedExactCPULlamaWeights {
        if let cachedExactCPULlamaWeights {
            return cachedExactCPULlamaWeights
        }

        let topLevelPaths = try Self.resolveLlamaTopLevelWeightPaths(
            config: config,
            weightDir: weightDirURL.path
        )
        let tokenEmbedding = try Self.loadWeightTablePreferringFloat32Sidecar(
            at: topLevelPaths.tokenEmbedding,
            expectedCount: config.vocab * config.dModel
        )
        let finalNormGamma = try Self.loadWeightTablePreferringFloat32Sidecar(
            at: topLevelPaths.finalNormGamma,
            expectedCount: config.dModel
        )
        let lmHead = try Self.loadWeightTablePreferringFloat32Sidecar(
            at: topLevelPaths.lmHead,
            expectedCount: config.vocab * config.dModel
        )
        let lmHeadFP16 = try Self.loadRawFP16WeightTableIfNoExactFloat32Sidecar(
            at: topLevelPaths.lmHead,
            expectedCount: config.vocab * config.dModel
        )
        let layers = try (0..<config.nLayer).map { layerIndex in
            let paths = LayerWeightPaths.forLayer(
                layerIndex,
                config: config,
                blobDir: weightDirURL.path
            )
            return try Self.loadExactCPULlamaLayerWeights(config: config, paths: paths)
        }
        let loadedWeights = CachedExactCPULlamaWeights(
            tokenEmbedding: tokenEmbedding,
            finalNormGamma: finalNormGamma,
            lmHead: lmHead,
            lmHeadFP16: lmHeadFP16,
            layers: layers
        )
        cachedExactCPULlamaWeights = loadedWeights
        return loadedWeights
    }

    private mutating func generateIncrementalExactTwoTokenDraftLlama(
        promptTokens: [TokenID],
        effectiveMaxTokens: Int,
        compileTimeMs: Double,
        draft: ResolvedExactTwoTokenDraft,
        onStep: ((GenerationStep) -> Void)?
    ) throws -> GenerationResult {
        guard !promptTokens.isEmpty else {
            throw RealModelInferenceError.invalidGenerationParameters("Prompt tokens must not be empty")
        }
        var fullRuntime = try CPUExactLlamaRuntime(config: config, weightDirURL: weightDirURL)
        var draftRuntime = try CPUExactLlamaRuntime(config: draft.config, weightDirURL: draft.weightDirURL)

        try fullRuntime.prefill(promptTokens: promptTokens)
        try draftRuntime.prefill(promptTokens: promptTokens)

        var allTokens = promptTokens
        var generatedTokens: [TokenID] = []
        var tokenLatenciesMs: [Double] = []
        generatedTokens.reserveCapacity(effectiveMaxTokens)
        tokenLatenciesMs.reserveCapacity(effectiveMaxTokens)

        let generationStart = DispatchTime.now().uptimeNanoseconds
        var emissionStart = generationStart
        var firstTokenLatencyMs = 0.0
        var firstTokenRecorded = false
        var committedExactTokensTotal = 0
        var acceptedFutureTokensTotal = 0
        var speculativePassCount = 0

        func emit(_ token: TokenID, at emissionNow: UInt64) {
            generatedTokens.append(token)
            allTokens.append(token)
            let elapsedMs = Self.milliseconds(from: emissionNow - generationStart)
            let tokenLatencyMs = Self.milliseconds(from: emissionNow - emissionStart)
            tokenLatenciesMs.append(tokenLatencyMs)
            if !firstTokenRecorded {
                firstTokenLatencyMs = Self.milliseconds(from: emissionNow - generationStart)
                firstTokenRecorded = true
            }
            let tokensPerSecond = Double(generatedTokens.count) / max(elapsedMs / 1_000, 1e-9)
            onStep?(
                GenerationStep(
                    token: token,
                    generatedTokens: generatedTokens,
                    text: tokenizer.decode(allTokens.map(Int.init)),
                    tokenLatencyMs: tokenLatencyMs,
                    elapsedMs: elapsedMs,
                    firstTokenLatencyMs: firstTokenLatencyMs,
                    tokensPerSecond: tokensPerSecond
                )
            )
            emissionStart = emissionNow
        }

        let firstToken = fullRuntime.selectGreedyToken()
        if let eosToken = config.eosToken, firstToken == eosToken {
            generatedTokens.append(firstToken)
            return GenerationResult(
                text: tokenizer.decode((promptTokens + generatedTokens).map(Int.init)),
                tokens: generatedTokens,
                promptTokens: promptTokens,
                tokenLatenciesMs: tokenLatenciesMs,
                tokensPerSecond: 0,
                compileTimeMs: compileTimeMs,
                firstTokenLatencyMs: 0,
                exactHeadBackend: "cpu_exact_two_token_draft",
                cachedBindingsEnabled: false,
                committedExactTokensPerPass: nil,
                acceptedFutureTokensPerPass: nil
            )
        }
        let firstEmission = DispatchTime.now().uptimeNanoseconds
        emit(firstToken, at: firstEmission)
        if generatedTokens.count >= effectiveMaxTokens || allTokens.count >= config.maxSeq {
            let totalMs = Self.milliseconds(from: DispatchTime.now().uptimeNanoseconds - generationStart)
            let tps = generatedTokens.isEmpty ? 0 : Double(generatedTokens.count) / max(totalMs / 1_000, 1e-9)
            return GenerationResult(
                text: tokenizer.decode(allTokens.map(Int.init)),
                tokens: generatedTokens,
                promptTokens: promptTokens,
                tokenLatenciesMs: tokenLatenciesMs,
                tokensPerSecond: tps,
                compileTimeMs: compileTimeMs,
                firstTokenLatencyMs: firstTokenLatencyMs,
                exactHeadBackend: "cpu_exact_two_token_draft",
                cachedBindingsEnabled: false,
                committedExactTokensPerPass: nil,
                acceptedFutureTokensPerPass: nil
            )
        }

        try fullRuntime.advance(token: firstToken)
        try draftRuntime.advance(token: firstToken)

        while generatedTokens.count < effectiveMaxTokens, allTokens.count < config.maxSeq {
            speculativePassCount += 1
            let remainingTokenBudget = min(effectiveMaxTokens - generatedTokens.count, config.maxSeq - allTokens.count)
            let draftCheckpoint = draftRuntime.captureCheckpoint()
            let proposedToken0 = draftRuntime.selectGreedyToken()
            try draftRuntime.advance(token: proposedToken0)

            let exactToken0 = fullRuntime.selectGreedyToken()
            var acceptedInPass = 0
            var committedInPass = 0

            if exactToken0 == proposedToken0 {
                acceptedInPass += 1
            } else {
                draftRuntime.rollback(to: draftCheckpoint)
            }

            let firstRoundEmission = DispatchTime.now().uptimeNanoseconds
            emit(exactToken0, at: firstRoundEmission)
            committedInPass += 1
            if let eosToken = config.eosToken, exactToken0 == eosToken {
                committedExactTokensTotal += committedInPass
                acceptedFutureTokensTotal += acceptedInPass
                break
            }
            try fullRuntime.advance(token: exactToken0)
            if exactToken0 != proposedToken0 {
                try draftRuntime.advance(token: exactToken0)
            }

            if generatedTokens.count >= effectiveMaxTokens || allTokens.count >= config.maxSeq || remainingTokenBudget <= 1 {
                committedExactTokensTotal += committedInPass
                acceptedFutureTokensTotal += acceptedInPass
                continue
            }

            let proposedToken1 = draftRuntime.selectGreedyToken()
            let exactToken1 = fullRuntime.selectGreedyToken()
            if exactToken1 == proposedToken1 {
                acceptedInPass += 1
            }
            let secondRoundEmission = DispatchTime.now().uptimeNanoseconds
            emit(exactToken1, at: secondRoundEmission)
            committedInPass += 1
            if let eosToken = config.eosToken, exactToken1 == eosToken {
                committedExactTokensTotal += committedInPass
                acceptedFutureTokensTotal += acceptedInPass
                break
            }
            try fullRuntime.advance(token: exactToken1)
            try draftRuntime.advance(token: exactToken1)

            committedExactTokensTotal += committedInPass
            acceptedFutureTokensTotal += acceptedInPass
        }

        let generationEnd = DispatchTime.now().uptimeNanoseconds
        let generationTimeMs = Self.milliseconds(from: generationEnd - generationStart)
        let tokensPerSecond = generatedTokens.isEmpty
            ? 0
            : Double(generatedTokens.count) / max(generationTimeMs / 1_000, 1e-9)
        let committedExactTokensPerPass = speculativePassCount == 0
            ? nil
            : Double(committedExactTokensTotal) / Double(speculativePassCount)
        let acceptedFutureTokensPerPass = speculativePassCount == 0
            ? nil
            : Double(acceptedFutureTokensTotal) / Double(speculativePassCount)

        return GenerationResult(
            text: tokenizer.decode(allTokens.map(Int.init)),
            tokens: generatedTokens,
            promptTokens: promptTokens,
            tokenLatenciesMs: tokenLatenciesMs,
            tokensPerSecond: tokensPerSecond,
            compileTimeMs: compileTimeMs,
            firstTokenLatencyMs: firstTokenLatencyMs,
            exactHeadBackend: "cpu_exact_two_token_draft",
            cachedBindingsEnabled: false,
            committedExactTokensPerPass: committedExactTokensPerPass,
            acceptedFutureTokensPerPass: acceptedFutureTokensPerPass
        )
    }

    private mutating func generateIncrementalExactCPULlama(
        promptTokens: [TokenID],
        effectiveMaxTokens: Int,
        temperature: Float,
        compileTimeMs: Double,
        maxSeq: Int,
        onStep: ((GenerationStep) -> Void)?
    ) throws -> GenerationResult {
        guard !promptTokens.isEmpty else {
            throw RealModelInferenceError.invalidGenerationParameters("Prompt tokens must not be empty")
        }
        let roundIntermediatesToFP16 = Self.shouldRoundCPUExactDecodeIntermediatesToFP16()
        let maybeRound: ([Float]) -> [Float] = { values in
            roundIntermediatesToFP16 ? Self.roundFloat16Vector(values) : values
        }
        let exactWeights = try loadCachedExactCPULlamaWeights()
        let tokenEmbedding = exactWeights.tokenEmbedding
        let finalNormGamma = exactWeights.finalNormGamma
        let layers = exactWeights.layers

        var kCaches = Array(
            repeating: [Float](repeating: 0, count: config.kvDim * maxSeq),
            count: config.nLayer
        )
        var vCaches = Array(
            repeating: [Float](repeating: 0, count: config.kvDim * maxSeq),
            count: config.nLayer
        )

        func forwardToken(_ token: TokenID, position: Int) throws -> [Float] {
            var hidden = Array(tokenEmbedding[Int(token) * config.dModel..<(Int(token) + 1) * config.dModel])
            for layerIndex in 0..<config.nLayer {
                let layer = layers[layerIndex]
                let attnNormed = Self.rmsNorm(hidden, weight: layer.rmsAtt, eps: Float(config.normEps))
                var q = maybeRound(
                    Self.multiplyRowMajorMatrix(
                        matrix: layer.wq,
                        rows: config.attentionDim,
                        cols: config.dModel,
                        vector: attnNormed
                    )
                )
                var k = maybeRound(
                    Self.multiplyRowMajorMatrix(
                        matrix: layer.wk,
                        rows: config.kvDim,
                        cols: config.dModel,
                        vector: attnNormed
                    )
                )
                let vRounded = maybeRound(
                    Self.multiplyRowMajorMatrix(
                        matrix: layer.wv,
                        rows: config.kvDim,
                        cols: config.dModel,
                        vector: attnNormed
                    )
                )

                if let qNorm = layer.qNorm {
                    q.withUnsafeMutableBufferPointer { values in
                        qNorm.withUnsafeBufferPointer { weights in
                            Self.applyPerHeadRMSNormInPlace(
                                values: values,
                                weights: weights,
                                headCount: config.nHead,
                                headDim: config.headDim,
                                epsilon: Float(config.normEps)
                            )
                        }
                    }
                }
                if let kNorm = layer.kNorm {
                    k.withUnsafeMutableBufferPointer { values in
                        kNorm.withUnsafeBufferPointer { weights in
                            Self.applyPerHeadRMSNormInPlace(
                                values: values,
                                weights: weights,
                                headCount: config.nKVHead,
                                headDim: config.headDim,
                                epsilon: Float(config.normEps)
                            )
                        }
                    }
                }

                q = maybeRound(
                    Self.applyHalfSplitRoPEPerHead(
                        q,
                        heads: config.nHead,
                        headDim: config.headDim,
                        position: position,
                        theta: config.ropeTheta
                    )
                )
                k = maybeRound(
                    Self.applyHalfSplitRoPEPerHead(
                        k,
                        heads: config.nKVHead,
                        headDim: config.headDim,
                        position: position,
                        theta: config.ropeTheta
                    )
                )

                for channel in 0..<config.kvDim {
                    kCaches[layerIndex][channel * maxSeq + position] = k[channel]
                    vCaches[layerIndex][channel * maxSeq + position] = vRounded[channel]
                }

                let context = Self.decodeContextFromCaches(
                    qOut: q,
                    kCache: kCaches[layerIndex],
                    vCache: vCaches[layerIndex],
                    heads: config.nHead,
                    kvHeads: config.nKVHead,
                    headDim: config.headDim,
                    visibleTokenCount: position + 1,
                    cacheStride: maxSeq
                )

                let projected = maybeRound(
                    zip(
                        hidden,
                        Self.multiplyRowMajorMatrix(
                            matrix: layer.wo,
                            rows: config.dModel,
                            cols: config.attentionDim,
                            vector: context
                        )
                    ).map(+)
                )
                let ffnNormed = Self.rmsNorm(projected, weight: layer.rmsFfn, eps: Float(config.normEps))
                let gate = Self.multiplyRowMajorMatrix(
                    matrix: layer.w1,
                    rows: config.hiddenDim,
                    cols: config.dModel,
                    vector: ffnNormed
                )
                let up = Self.multiplyRowMajorMatrix(
                    matrix: layer.w3,
                    rows: config.hiddenDim,
                    cols: config.dModel,
                    vector: ffnNormed
                )
                let activated = zip(gate, up).map { Self.silu($0) * $1 }
                let down = Self.multiplyRowMajorMatrix(
                    matrix: layer.w2,
                    rows: config.dModel,
                    cols: config.hiddenDim,
                    vector: activated
                )
                hidden = maybeRound(zip(projected, down).map(+))
            }
            return hidden
        }

        var allTokens = promptTokens
        var lastHidden = [Float](repeating: 0, count: config.dModel)
        for (position, token) in promptTokens.enumerated() {
            lastHidden = try forwardToken(token, position: position)
        }

        let generationStart = DispatchTime.now().uptimeNanoseconds
        var emissionStart = generationStart
        var firstTokenLatencyMs = 0.0
        var firstTokenRecorded = false
        var generatedTokens: [TokenID] = []
        var tokenLatenciesMs: [Double] = []
        generatedTokens.reserveCapacity(effectiveMaxTokens)
        tokenLatenciesMs.reserveCapacity(effectiveMaxTokens)
        var rng = SystemRandomNumberGenerator()

        while generatedTokens.count < effectiveMaxTokens {
            let normalized = Self.rmsNorm(lastHidden, weight: finalNormGamma, eps: Float(config.normEps))
            let nextToken: TokenID
            if temperature == 0 {
                nextToken = TokenID(exactClassifierArgmax(normalized))
            } else {
                nextToken = selectTokenFromNormalizedHidden(
                    normalized,
                    temperature: temperature,
                    using: &rng
                )
            }

            let emissionNow = DispatchTime.now().uptimeNanoseconds
            let tokenLatencyMs = Self.milliseconds(from: emissionNow - emissionStart)
            if !firstTokenRecorded {
                firstTokenLatencyMs = Self.milliseconds(from: emissionNow - generationStart)
                firstTokenRecorded = true
            }

            if let eosToken = config.eosToken, nextToken == eosToken {
                generatedTokens.append(nextToken)
                break
            }

            generatedTokens.append(nextToken)
            allTokens.append(nextToken)
            let elapsedMs = Self.milliseconds(from: emissionNow - generationStart)
            let tokensPerSecond = Double(generatedTokens.count) / max(elapsedMs / 1_000, 1e-9)
            tokenLatenciesMs.append(tokenLatencyMs)
            onStep?(
                GenerationStep(
                    token: nextToken,
                    generatedTokens: generatedTokens,
                    text: tokenizer.decode(allTokens.map(Int.init)),
                    tokenLatencyMs: tokenLatencyMs,
                    elapsedMs: elapsedMs,
                    firstTokenLatencyMs: firstTokenLatencyMs,
                    tokensPerSecond: tokensPerSecond
                )
            )

            if generatedTokens.count >= effectiveMaxTokens || allTokens.count >= maxSeq {
                break
            }

            lastHidden = try forwardToken(nextToken, position: allTokens.count - 1)
            emissionStart = emissionNow
        }

        let generationEnd = DispatchTime.now().uptimeNanoseconds
        let generationTimeMs = Self.milliseconds(from: generationEnd - generationStart)
        let tokensPerSecond = generatedTokens.isEmpty
            ? 0
            : Double(generatedTokens.count) / max(generationTimeMs / 1_000, 1e-9)

        return GenerationResult(
            text: tokenizer.decode(allTokens.map(Int.init)),
            tokens: generatedTokens,
            promptTokens: promptTokens,
            tokenLatenciesMs: tokenLatenciesMs,
            tokensPerSecond: tokensPerSecond,
            compileTimeMs: compileTimeMs,
            firstTokenLatencyMs: firstTokenLatencyMs,
            exactHeadBackend: classifierStrategy.exactHeadBackendLabel
        )
    }

    private static func compileHybridLayers(
        config: MultiModelConfig,
        weightDirURL: URL,
        sourceLayerRange: Range<Int>? = nil,
        maxSeq: Int
    ) throws -> LayerStorage<HybridDecodeKernelSet> {
        let layerRange = sourceLayerRange ?? (0..<config.nLayer)
        let useDonorDelta = supportsHybridDonorDelta(
            config: config,
            environment: ProcessInfo.processInfo.environment
        )
        var donorHexIDs: HybridDecodeKernelSet.DonorHexIDs? = nil
        return try LayerStorage<HybridDecodeKernelSet>(count: layerRange.count, throwingInitializer: { localLayerIndex in
            let layerIndex = layerRange.lowerBound + localLayerIndex
            let paths = LayerWeightPaths.forLayer(layerIndex, config: config, blobDir: weightDirURL.path)
            let weights: LayerWeights = switch config.architecture {
            case .gpt2: try loadHybridLayerWeights(config: config, paths: paths)
            case .llama: try loadHybridLayerWeightsLlama(config: config, paths: paths)
            }
            do {
                let kernels = try HybridDecodeKernelSet(
                    weights: weights,
                    maxSeq: maxSeq,
                    donorHexIDs: useDonorDelta ? donorHexIDs : nil
                )
                if useDonorDelta {
                    donorHexIDs = kernels.donorHexIDs
                }
                return kernels
            } catch {
                throw RealModelInferenceError.runtimeFailure(
                    "Hybrid decode compilation failed for layer \(layerIndex): \(error)"
                )
            }
        })
    }

    private static func compileLayers(
        config: MultiModelConfig,
        weightDirURL: URL,
        bucket: Int,
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) throws -> LayerStorage<CompiledLayer> {
        try LayerStorage<CompiledLayer>(count: config.nLayer, throwingInitializer: { layerIndex in
            try compileLayer(
                layerIndex: layerIndex,
                config: config,
                weightDirURL: weightDirURL,
                spatial: bucket,
                environment: environment
            )
        })
    }

    private static func loadHybridLayerWeights(
        config: MultiModelConfig,
        paths: LayerWeightPaths
    ) throws -> LayerWeights {
        // GPT-2 is always MHA (nKVHeads == nHeads), so kvDim defaults to dim
        let weights = LayerWeights(
            architecture: .gpt2,
            dim: config.dModel,
            hiddenDim: config.hiddenDim,
            normEps: config.normEps
        )

        let layerDirectory = URL(fileURLWithPath: paths.wq).deletingLastPathComponent()
        let attentionNormBiasPath = replacingGammaSuffix(in: paths.rmsAtt)
        let ffnNormBiasPath = replacingGammaSuffix(in: paths.rmsFfn)

        try loadTensor(weights.rmsAtt, from: paths.rmsAtt, expectedCount: config.dModel)
        try loadTensor(weights.attentionNormBeta, from: attentionNormBiasPath, expectedCount: config.dModel)
        try loadTensor(weights.Wq, from: paths.wq, expectedCount: config.dModel * config.dModel)
        try loadTensor(weights.Wk, from: paths.wk, expectedCount: config.dModel * config.dModel)
        try loadTensor(weights.Wv, from: paths.wv, expectedCount: config.dModel * config.dModel)
        try loadTensor(weights.Wo, from: paths.wo, expectedCount: config.dModel * config.dModel)
        guard let bqPath = paths.bq,
              let bkPath = paths.bk,
              let bvPath = paths.bv,
              let boPath = paths.bo else {
            throw RealModelInferenceError.runtimeFailure("Missing GPT-2 QKV bias weights for \(layerDirectory.path)")
        }
        try loadTensor(weights.bq, from: bqPath, expectedCount: config.dModel)
        try loadTensor(weights.bk, from: bkPath, expectedCount: config.dModel)
        try loadTensor(weights.bv, from: bvPath, expectedCount: config.dModel)
        try loadTensor(weights.bo, from: boPath, expectedCount: config.dModel)

        try loadTensor(weights.rmsFfn, from: paths.rmsFfn, expectedCount: config.dModel)
        try loadTensor(weights.ffnNormBeta, from: ffnNormBiasPath, expectedCount: config.dModel)
        try loadTensor(weights.W1, from: paths.w1, expectedCount: config.hiddenDim * config.dModel)
        try loadTensor(weights.W2, from: paths.w2, expectedCount: config.dModel * config.hiddenDim)
        guard let b1Path = paths.b1, let b2Path = paths.b2 else {
            throw RealModelInferenceError.runtimeFailure("Missing GPT-2 FFN bias weights for \(layerDirectory.path)")
        }
        try loadTensor(weights.b1, from: b1Path, expectedCount: config.hiddenDim)
        try loadTensor(weights.b2, from: b2Path, expectedCount: config.dModel)

        return weights
    }

    static func loadHybridLayerWeightsLlama(
        config: MultiModelConfig,
        paths: LayerWeightPaths
    ) throws -> LayerWeights {
        let qkNormWeights = try loadLlamaQKNormWeights(config: config, paths: paths)
        let qDim = config.attentionDim
        let kvDim = config.kvDim
        let weights = LayerWeights(
            architecture: .rmsNormSwiGLU,
            dim: config.dModel,
            hiddenDim: config.hiddenDim,
            qDim: qDim,
            kvDim: kvDim,
            normEps: config.normEps,
            qNormDim: qkNormWeights == nil ? nil : config.headDim,
            kNormDim: qkNormWeights == nil ? nil : config.headDim
        )

        try loadTensor(weights.rmsAtt, from: paths.rmsAtt, expectedCount: config.dModel)
        try loadTensor(weights.Wq, from: paths.wq, expectedCount: config.dModel * qDim)
        try loadTensor(weights.Wk, from: paths.wk, expectedCount: config.dModel * kvDim)
        try loadTensor(weights.Wv, from: paths.wv, expectedCount: config.dModel * kvDim)
        try loadTensor(weights.Wo, from: paths.wo, expectedCount: config.dModel * qDim)
        if let qkNormWeights {
            weights.qNorm.withUnsafeMutableBufferPointer { dst in
                _ = dst.initialize(from: qkNormWeights.q)
            }
            weights.kNorm.withUnsafeMutableBufferPointer { dst in
                _ = dst.initialize(from: qkNormWeights.k)
            }
        }
        try loadTensor(weights.rmsFfn, from: paths.rmsFfn, expectedCount: config.dModel)
        try loadTensor(weights.W1, from: paths.w1, expectedCount: config.hiddenDim * config.dModel)
        try loadTensor(weights.W2, from: paths.w2, expectedCount: config.dModel * config.hiddenDim)
        guard let w3Path = paths.w3 else {
            let layerDirectory = URL(fileURLWithPath: paths.wq).deletingLastPathComponent()
            throw RealModelInferenceError.runtimeFailure("Missing llama W3 (gate) weight for \(layerDirectory.path)")
        }
        try loadTensor(weights.W3, from: w3Path, expectedCount: config.hiddenDim * config.dModel)

        return weights
    }

    static func loadHybridLayerWeightsLlamaForTesting(
        config: MultiModelConfig,
        weightDir: String,
        layer: Int
    ) throws -> LayerWeights {
        let paths = LayerWeightPaths.forLayer(layer, config: config, blobDir: weightDir)
        return try loadHybridLayerWeightsLlama(config: config, paths: paths)
    }

    private static func loadLlamaQKNormWeights(
        config: MultiModelConfig,
        paths: LayerWeightPaths
    ) throws -> LlamaQKNormWeights? {
        let qNormExists = fileExists(at: paths.qNorm)
        let kNormExists = fileExists(at: paths.kNorm)
        guard qNormExists == kNormExists else {
            let layerDirectory = URL(fileURLWithPath: paths.wq).deletingLastPathComponent()
            throw RealModelInferenceError.runtimeFailure(
                "Mismatched llama Q/K norm weights for \(layerDirectory.path); expected both q_norm.bin and k_norm.bin"
            )
        }
        guard qNormExists else {
            return nil
        }
        guard let qNormPath = paths.qNorm, let kNormPath = paths.kNorm else {
            return nil
        }
        return LlamaQKNormWeights(
            q: try loadWeightTablePreferringFloat32Sidecar(at: qNormPath, expectedCount: config.headDim),
            k: try loadWeightTablePreferringFloat32Sidecar(at: kNormPath, expectedCount: config.headDim)
        )
    }

    private static func loadLlamaCPUQKVWeights(
        config: MultiModelConfig,
        paths: LayerWeightPaths
    ) throws -> LlamaCPUQKVWeights {
        let qkNormWeights = try loadLlamaQKNormWeights(config: config, paths: paths)
        return LlamaCPUQKVWeights(
            rmsAtt: try loadWeightTablePreferringFloat32Sidecar(at: paths.rmsAtt, expectedCount: config.dModel),
            wq: try loadWeightTablePreferringFloat32Sidecar(at: paths.wq, expectedCount: config.dModel * config.attentionDim),
            wk: try loadWeightTablePreferringFloat32Sidecar(at: paths.wk, expectedCount: config.dModel * config.kvDim),
            wv: try loadWeightTablePreferringFloat32Sidecar(at: paths.wv, expectedCount: config.dModel * config.kvDim),
            qNorm: qkNormWeights?.q,
            kNorm: qkNormWeights?.k
        )
    }

    private static func loadExactCPULlamaLayerWeights(
        config: MultiModelConfig,
        paths: LayerWeightPaths
    ) throws -> ExactCPULlamaLayerWeights {
        let qkNormWeights = try loadLlamaQKNormWeights(config: config, paths: paths)
        guard let w3Path = paths.w3 else {
            let layerDirectory = URL(fileURLWithPath: paths.wq).deletingLastPathComponent()
            throw RealModelInferenceError.runtimeFailure("Missing llama W3 (gate) weight for \(layerDirectory.path)")
        }
        return ExactCPULlamaLayerWeights(
            rmsAtt: try loadWeightTablePreferringFloat32Sidecar(at: paths.rmsAtt, expectedCount: config.dModel),
            wq: try loadWeightTablePreferringFloat32Sidecar(at: paths.wq, expectedCount: config.dModel * config.attentionDim),
            wk: try loadWeightTablePreferringFloat32Sidecar(at: paths.wk, expectedCount: config.dModel * config.kvDim),
            wv: try loadWeightTablePreferringFloat32Sidecar(at: paths.wv, expectedCount: config.dModel * config.kvDim),
            wo: try loadWeightTablePreferringFloat32Sidecar(at: paths.wo, expectedCount: config.dModel * config.attentionDim),
            rmsFfn: try loadWeightTablePreferringFloat32Sidecar(at: paths.rmsFfn, expectedCount: config.dModel),
            w1: try loadWeightTablePreferringFloat32Sidecar(at: paths.w1, expectedCount: config.hiddenDim * config.dModel),
            w2: try loadWeightTablePreferringFloat32Sidecar(at: paths.w2, expectedCount: config.dModel * config.hiddenDim),
            w3: try loadWeightTablePreferringFloat32Sidecar(at: w3Path, expectedCount: config.hiddenDim * config.dModel),
            qNorm: qkNormWeights?.q,
            kNorm: qkNormWeights?.k
        )
    }

    static func applyPerHeadRMSNormInPlace(
        values: UnsafeMutableBufferPointer<Float>,
        weights: UnsafeBufferPointer<Float>,
        headCount: Int,
        headDim: Int,
        epsilon: Float
    ) {
        precondition(headCount >= 0)
        precondition(headDim > 0)
        precondition(values.count == headCount * headDim)
        precondition(weights.count == headDim)

        for head in 0..<headCount {
            let base = head * headDim
            var sumSq: Float = 0
            for lane in 0..<headDim {
                let value = values[base + lane]
                sumSq += value * value
            }
            let invRms = 1.0 / sqrtf(sumSq / Float(headDim) + epsilon)
            for lane in 0..<headDim {
                values[base + lane] *= invRms * weights[lane]
            }
        }
    }

    private static func multiplyRowMajorMatrix(
        matrix: [Float],
        rows: Int,
        cols: Int,
        vector: UnsafeBufferPointer<Float>,
        into output: UnsafeMutableBufferPointer<Float>
    ) {
        precondition(matrix.count == rows * cols)
        precondition(vector.count == cols)
        precondition(output.count == rows)
        matrix.withUnsafeBufferPointer { matrixBuffer in
            vDSP_mmul(
                matrixBuffer.baseAddress!,
                1,
                vector.baseAddress!,
                1,
                output.baseAddress!,
                1,
                vDSP_Length(rows),
                1,
                vDSP_Length(cols)
            )
        }
    }

    private static func multiplyRowMajorMatrix(
        matrix: [Float],
        rows: Int,
        cols: Int,
        vector: [Float]
    ) -> [Float] {
        var output = [Float](repeating: 0, count: rows)
        output.withUnsafeMutableBufferPointer { outputBuffer in
            vector.withUnsafeBufferPointer { vectorBuffer in
                multiplyRowMajorMatrix(
                    matrix: matrix,
                    rows: rows,
                    cols: cols,
                    vector: vectorBuffer,
                    into: outputBuffer
                )
            }
        }
        return output
    }

    private static func roundFloat16Vector(_ values: [Float]) -> [Float] {
        values.map { Float(Float16($0)) }
    }

    private static func rmsNorm(_ input: [Float], weight: [Float], eps: Float) -> [Float] {
        precondition(input.count == weight.count)
        var normalized = [Float](repeating: 0, count: input.count)
        var sumSq: Float = 0
        input.withUnsafeBufferPointer { inputBuffer in
            vDSP_dotpr(inputBuffer.baseAddress!, 1, inputBuffer.baseAddress!, 1, &sumSq, vDSP_Length(input.count))
        }
        var invRms = 1.0 / sqrtf(sumSq / Float(input.count) + eps)
        input.withUnsafeBufferPointer { inputBuffer in
            normalized.withUnsafeMutableBufferPointer { normalizedBuffer in
                vDSP_vsmul(inputBuffer.baseAddress!, 1, &invRms, normalizedBuffer.baseAddress!, 1, vDSP_Length(input.count))
            }
        }
        weight.withUnsafeBufferPointer { weightBuffer in
            normalized.withUnsafeMutableBufferPointer { normalizedBuffer in
                vDSP_vmul(normalizedBuffer.baseAddress!, 1, weightBuffer.baseAddress!, 1, normalizedBuffer.baseAddress!, 1, vDSP_Length(input.count))
            }
        }
        return normalized
    }

    private static func applyHalfSplitRoPEPerHead(
        _ input: [Float],
        heads: Int,
        headDim: Int,
        position: Int,
        theta: Float
    ) -> [Float] {
        precondition(input.count == heads * headDim)
        precondition(headDim % 2 == 0)
        let halfDim = headDim / 2
        var output = input
        for head in 0..<heads {
            let base = head * headDim
            for dimPair in 0..<halfDim {
                let frequency = 1.0 / pow(theta, Float(2 * dimPair) / Float(headDim))
                let angle = Float(position) * frequency
                let cosv = cos(angle)
                let sinv = sin(angle)
                let i0 = base + dimPair
                let i1 = base + dimPair + halfDim
                let v0 = output[i0]
                let v1 = output[i1]
                output[i0] = v0 * cosv - v1 * sinv
                output[i1] = v0 * sinv + v1 * cosv
            }
        }
        return output
    }

    private static func decodeContextFromCaches(
        qOut: [Float],
        kCache: [Float],
        vCache: [Float],
        heads: Int,
        kvHeads: Int,
        headDim: Int,
        visibleTokenCount: Int,
        cacheStride: Int
    ) -> [Float] {
        precondition(qOut.count == heads * headDim)
        precondition(kCache.count == kvHeads * headDim * cacheStride)
        precondition(vCache.count == kvHeads * headDim * cacheStride)
        precondition(visibleTokenCount > 0 && visibleTokenCount <= cacheStride)
        let queriesPerKVHead = max(heads / max(kvHeads, 1), 1)
        let scale = 1.0 / sqrt(Float(headDim))
        var context = [Float](repeating: 0, count: heads * headDim)

        for head in 0..<heads {
            let kvHead = min(head / queriesPerKVHead, kvHeads - 1)
            let qBase = head * headDim
            let kvBase = kvHead * headDim
            var scores = [Float](repeating: 0, count: visibleTokenCount)
            for token in 0..<visibleTokenCount {
                var dot: Float = 0
                for dim in 0..<headDim {
                    dot += qOut[qBase + dim] * kCache[(kvBase + dim) * cacheStride + token]
                }
                scores[token] = dot * scale
            }

            let maxScore = scores.max() ?? 0
            var denom: Float = 0
            for token in 0..<visibleTokenCount {
                scores[token] = exp(scores[token] - maxScore)
                denom += scores[token]
            }
            let invDenom: Float = denom > 0 ? 1 / denom : 0

            for dim in 0..<headDim {
                var accum: Float = 0
                for token in 0..<visibleTokenCount {
                    accum += scores[token] * invDenom * vCache[(kvBase + dim) * cacheStride + token]
                }
                context[qBase + dim] = accum
            }
        }

        return context
    }

    private static func silu(_ value: Float) -> Float {
        0.5 * value * (1 + tanh(0.5 * value))
    }

    private enum LayerBlockKind: String {
        case attention = "attn"
        case ffn
    }

    private static func compileLayer(
        layerIndex: Int,
        config: MultiModelConfig,
        weightDirURL: URL,
        spatial: Int,
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) throws -> CompiledLayer {
        let paths = LayerWeightPaths.forLayer(layerIndex, config: config, blobDir: weightDirURL.path)
        let ioBytes = try ANEShape(channels: config.dModel, spatial: spatial).byteSize(for: .fp32)

        let attentionGraph = buildGPT2AttentionBlockGraph(
            layerIndex: layerIndex,
            config: config,
            paths: paths,
            spatial: spatial,
            environment: environment
        )
        let attentionKernel = try compileLayerBlock(
            layerIndex: layerIndex,
            kind: .attention,
            graph: attentionGraph,
            weights: try attentionWeights(
                config: config,
                diskPaths: paths,
                weightDirURL: weightDirURL,
                spatial: spatial
            ),
            inputBytes: ioBytes,
            outputBytes: [ioBytes, ioBytes, ioBytes],
            weightDirURL: weightDirURL,
            spatial: spatial,
            environment: environment
        )

        let attentionOutputSurface: IOSurfaceRef
        do {
            attentionOutputSurface = try attentionKernel.outputSurface(at: 0)
        } catch {
            throw RealModelInferenceError.runtimeFailure("Layer \(layerIndex) attention output surface unavailable: \(error)")
        }

        let ffnGraph = buildGPT2FFNBlockGraph(
            layerIndex: layerIndex,
            config: config,
            paths: paths,
            spatial: spatial,
            environment: environment
        )
        let ffnKernel = try compileLayerBlock(
            layerIndex: layerIndex,
            kind: .ffn,
            graph: ffnGraph,
            weights: try ffnWeights(
                config: config,
                diskPaths: paths,
                weightDirURL: weightDirURL
            ),
            inputBytes: ioBytes,
            outputBytes: [ioBytes],
            weightDirURL: weightDirURL,
            spatial: spatial,
            environment: environment
        )

        let outputSurface: IOSurfaceRef
        do {
            outputSurface = try ffnKernel.outputSurface(at: 0)
        } catch {
            throw RealModelInferenceError.runtimeFailure("Layer \(layerIndex) FFN output surface unavailable: \(error)")
        }
        do {
            try ffnKernel.rebindInput(at: 0, to: attentionOutputSurface)
        } catch {
            throw RealModelInferenceError.runtimeFailure("Layer \(layerIndex) failed to chain attention into FFN: \(error)")
        }
        return CompiledLayer(
            attentionKernel: attentionKernel,
            attentionOutputSurface: attentionOutputSurface,
            ffnKernel: ffnKernel,
            outputSurface: outputSurface
        )
    }

    private static func compileLayerBlock(
        layerIndex: Int,
        kind: LayerBlockKind,
        graph: ANEGraph,
        weights: [(path: String, data: Data)],
        inputBytes: Int,
        outputBytes: [Int],
        weightDirURL: URL,
        spatial: Int,
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) throws -> ANEKernel {
        var optimized = graph
        ANEOptimizationPipeline.optimize(&optimized)
        let mil = rewriteMILWeightPaths(
            ANECodegen.emit(optimized, deploymentTarget: milDeploymentTarget(environment: environment)),
            rootDir: weightDirURL
        )
        let diagnostics = ANEValidationPass().run(on: optimized)
        do {
            return try ANEKernel(
                milText: mil,
                weights: weights,
                inputSizes: [inputBytes],
                outputSizes: outputBytes
            )
        } catch {
            let milPath = dumpDebugMIL(
                mil,
                filename: "real-model-layer-\(layerIndex)-\(kind.rawValue)-s\(spatial).mil"
            )
            let validation = diagnostics.isEmpty
                ? "none"
                : diagnostics.map(\.message).joined(separator: " | ")
            throw RealModelInferenceError.runtimeFailure(
                "Layer \(layerIndex) \(kind.rawValue) compilation failed: \(error). Validation diagnostics: \(validation). MIL dump: \(milPath)"
            )
        }
    }

    private static func attentionWeights(
        config: MultiModelConfig,
        diskPaths: LayerWeightPaths,
        weightDirURL: URL,
        spatial: Int
    ) throws -> [(path: String, data: Data)] {
        func addPath(actualPath: String?, into values: inout [(path: String, data: Data)]) throws {
            guard let actualPath else { return }
            let compilePath = compileBlobPath(actualPath: actualPath, rootDir: weightDirURL)
            values.append((path: compilePath, data: try canonicalBlobData(at: actualPath)))
        }

        var values: [(path: String, data: Data)] = []
        switch config.architecture {
        case .gpt2:
            let diskAttnBeta = replacingGammaSuffix(in: diskPaths.rmsAtt)
            let diskFfnBeta = replacingGammaSuffix(in: diskPaths.rmsFfn)
            try addPath(actualPath: diskPaths.rmsAtt, into: &values)
            try addPath(actualPath: diskAttnBeta, into: &values)
            try addPath(actualPath: diskPaths.wq, into: &values)
            try addPath(actualPath: diskPaths.wk, into: &values)
            try addPath(actualPath: diskPaths.wv, into: &values)
            try addPath(actualPath: diskPaths.wo, into: &values)
            try addPath(actualPath: diskPaths.bq, into: &values)
            try addPath(actualPath: diskPaths.bk, into: &values)
            try addPath(actualPath: diskPaths.bv, into: &values)
            try addPath(actualPath: diskPaths.bo, into: &values)
            _ = diskFfnBeta
        case .llama:
            throw RealModelInferenceError.unsupportedArchitecture(
                "Llama full-sequence path is not supported; use the hybrid decode path instead."
            )
        }

        let maskActualPath = weightDirURL
            .appendingPathComponent("masks", isDirectory: true)
            .appendingPathComponent("causal_\(spatial).bin")
            .path
        let maskCompilePath = compileBlobPath(actualPath: maskActualPath, rootDir: weightDirURL)
        values.append((path: maskCompilePath, data: causalMaskBlob(seqLen: spatial)))
        return values
    }

    private static func ffnWeights(
        config: MultiModelConfig,
        diskPaths: LayerWeightPaths,
        weightDirURL: URL
    ) throws -> [(path: String, data: Data)] {
        func addPath(actualPath: String?, into values: inout [(path: String, data: Data)]) throws {
            guard let actualPath else { return }
            let compilePath = compileBlobPath(actualPath: actualPath, rootDir: weightDirURL)
            values.append((path: compilePath, data: try canonicalBlobData(at: actualPath)))
        }

        var values: [(path: String, data: Data)] = []
        switch config.architecture {
        case .gpt2:
            let diskFfnBeta = replacingGammaSuffix(in: diskPaths.rmsFfn)
            try addPath(actualPath: diskPaths.rmsFfn, into: &values)
            try addPath(actualPath: diskFfnBeta, into: &values)
            try addPath(actualPath: diskPaths.w1, into: &values)
            try addPath(actualPath: diskPaths.w2, into: &values)
            try addPath(actualPath: diskPaths.b1, into: &values)
            try addPath(actualPath: diskPaths.b2, into: &values)
        case .llama:
            throw RealModelInferenceError.unsupportedArchitecture(
                "Llama full-sequence path is not supported; use the hybrid decode path instead."
            )
        }
        return values
    }

    private static func compileHead(
        config: MultiModelConfig,
        weightDirURL: URL,
        assets: GPT2TopLevelAssets,
        spatial: Int,
        inputDType: ANEDType = .fp32,
        outputDType: ANEDType = .fp32,
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) throws -> CompiledHead {
        var graph = buildGPT2HeadGraph(
            config: config,
            assets: assets,
            spatial: spatial,
            inputDType: inputDType,
            outputDType: outputDType,
            environment: environment
        )
        ANEOptimizationPipeline.optimize(&graph)
        let mil = rewriteMILWeightPaths(
            ANECodegen.emit(graph, deploymentTarget: milDeploymentTarget(environment: environment)),
            rootDir: weightDirURL
        )
        let inputBytes = try ANEShape(channels: config.dModel, spatial: spatial).byteSize(for: inputDType)
        let outputBytes = try ANEShape(channels: config.dModel, spatial: spatial).byteSize(for: outputDType)
        let kernel: ANEKernel
        do {
            kernel = try ANEKernel(
                milText: mil,
                weights: [
                    (path: assets.finalNormGammaCompilePath, data: assets.finalNormGammaData),
                    (path: assets.finalNormBetaCompilePath, data: assets.finalNormBetaData),
                ],
                inputBytes: inputBytes,
                outputBytes: outputBytes
            )
        } catch {
            throw RealModelInferenceError.runtimeFailure("Final norm compilation failed: \(error)")
        }

        let inputSurface: IOSurfaceRef
        let outputSurface: IOSurfaceRef
        do {
            inputSurface = try kernel.inputSurface(at: 0)
            outputSurface = try kernel.outputSurface(at: 0)
        } catch {
            throw RealModelInferenceError.runtimeFailure("Final norm surfaces unavailable: \(error)")
        }
        return CompiledHead(kernel: kernel, inputSurface: inputSurface, outputSurface: outputSurface)
    }

    private static func compileClassifier(
        config: MultiModelConfig,
        assets: GPT2TopLevelAssets,
        spatial: Int
    ) throws -> CompiledClassifier {
        let generator = GenerationClassifierWithMaxGenerator(vocabSize: config.vocab, laneSpatial: spatial)
        let classifierBlob = WeightBlob.build(from: assets.lmHead, rows: config.vocab, cols: config.dModel)
        let kernel: ANEKernel
        do {
            kernel = try ANEKernel(
                milText: generator.milText,
                weights: [
                    (path: "@model_path/weights/classifier.bin", data: classifierBlob),
                ],
                inputSizes: generator.inputByteSizes,
                outputSizes: generator.outputByteSizes
            )
        } catch {
            throw RealModelInferenceError.runtimeFailure("Hybrid classifier compilation failed: \(error)")
        }

        let inputSurface: IOSurfaceRef
        let outputSurface: IOSurfaceRef
        let maxValueSurface: IOSurfaceRef
        do {
            inputSurface = try kernel.inputSurface(at: 0)
            outputSurface = try kernel.outputSurface(at: 0)
            maxValueSurface = try kernel.outputSurface(at: 1)
        } catch {
            throw RealModelInferenceError.runtimeFailure("Hybrid classifier surfaces unavailable: \(error)")
        }
        return CompiledClassifier(
            kernel: kernel,
            inputSurface: inputSurface,
            outputSurface: outputSurface,
            maxValueSurface: maxValueSurface
        )
    }

    private static func compileLlamaHead(
        config: MultiModelConfig,
        weightDirURL: URL,
        assets: LlamaTopLevelAssets,
        spatial: Int,
        inputDType: ANEDType = .fp32,
        outputDType: ANEDType = .fp32
    ) throws -> CompiledHead {
        var graph = buildLlamaHeadGraph(
            config: config,
            assets: assets,
            spatial: spatial,
            inputDType: inputDType,
            outputDType: outputDType
        )
        ANEOptimizationPipeline.optimize(&graph)
        let mil = rewriteMILWeightPaths(ANECodegen.emit(graph), rootDir: weightDirURL)
        let inputBytes = try ANEShape(channels: config.dModel, spatial: spatial).byteSize(for: inputDType)
        let outputBytes = try ANEShape(channels: config.dModel, spatial: spatial).byteSize(for: outputDType)
        let kernel: ANEKernel
        do {
            kernel = try ANEKernel(
                milText: mil,
                weights: [
                    (path: assets.finalNormGammaCompilePath, data: assets.finalNormGammaData),
                ],
                inputBytes: inputBytes,
                outputBytes: outputBytes
            )
        } catch {
            throw RealModelInferenceError.runtimeFailure("Llama final RMSNorm compilation failed: \(error)")
        }

        let inputSurface: IOSurfaceRef
        let outputSurface: IOSurfaceRef
        do {
            inputSurface = try kernel.inputSurface(at: 0)
            outputSurface = try kernel.outputSurface(at: 0)
        } catch {
            throw RealModelInferenceError.runtimeFailure("Llama final RMSNorm surfaces unavailable: \(error)")
        }
        return CompiledHead(kernel: kernel, inputSurface: inputSurface, outputSurface: outputSurface)
    }

    private static func compileLlamaClassifier(
        config: MultiModelConfig,
        assets: LlamaTopLevelAssets,
        spatial: Int
    ) throws -> CompiledClassifier {
        let generator = GenerationClassifierWithMaxGenerator(vocabSize: config.vocab, laneSpatial: spatial)
        let classifierBlob = if let lmHeadFP16 = assets.lmHeadFP16 {
            WeightBlob.buildFP16(from: lmHeadFP16)
        } else {
            WeightBlob.build(from: assets.lmHead, rows: config.vocab, cols: config.dModel)
        }
        let kernel: ANEKernel
        do {
            kernel = try ANEKernel(
                milText: generator.milText,
                weights: [
                    (path: "@model_path/weights/classifier.bin", data: classifierBlob),
                ],
                inputSizes: generator.inputByteSizes,
                outputSizes: generator.outputByteSizes
            )
        } catch {
            throw RealModelInferenceError.runtimeFailure("Llama classifier compilation failed: \(error)")
        }

        let inputSurface: IOSurfaceRef
        let outputSurface: IOSurfaceRef
        let maxValueSurface: IOSurfaceRef
        do {
            inputSurface = try kernel.inputSurface(at: 0)
            outputSurface = try kernel.outputSurface(at: 0)
            maxValueSurface = try kernel.outputSurface(at: 1)
        } catch {
            throw RealModelInferenceError.runtimeFailure("Llama classifier surfaces unavailable: \(error)")
        }
        return CompiledClassifier(
            kernel: kernel,
            inputSurface: inputSurface,
            outputSurface: outputSurface,
            maxValueSurface: maxValueSurface
        )
    }

    private static func compileLlamaRMSNormClassifier(
        config: MultiModelConfig,
        assets: LlamaTopLevelAssets,
        spatial: Int
    ) throws -> CompiledClassifier {
        guard config.dModel == ModelConfig.dim else {
            throw RealModelInferenceError.runtimeFailure(
                "Llama fused RMSNorm classifier currently requires dModel \(ModelConfig.dim), got \(config.dModel)"
            )
        }
        let generator = GenerationRMSNormClassifierGenerator(vocabSize: config.vocab, laneSpatial: spatial)
        let rmsBlob = assets.finalNormGamma.withUnsafeBufferPointer { ptr in
            WeightBlob.build(from: ptr, rows: 1, cols: config.dModel)
        }
        let classifierBlob = if let lmHeadFP16 = assets.lmHeadFP16 {
            WeightBlob.buildFP16(from: lmHeadFP16)
        } else {
            WeightBlob.build(from: assets.lmHead, rows: config.vocab, cols: config.dModel)
        }
        let kernel: ANEKernel
        do {
            kernel = try ANEKernel(
                milText: generator.milText,
                weights: [
                    (path: "@model_path/weights/rms_final.bin", data: rmsBlob),
                    (path: "@model_path/weights/classifier.bin", data: classifierBlob),
                ],
                inputSizes: generator.inputByteSizes,
                outputSizes: generator.outputByteSizes
            )
        } catch {
            throw RealModelInferenceError.runtimeFailure("Llama fused RMSNorm classifier compilation failed: \(error)")
        }

        let inputSurface: IOSurfaceRef
        let outputSurface: IOSurfaceRef
        let maxValueSurface: IOSurfaceRef
        do {
            inputSurface = try kernel.inputSurface(at: 0)
            outputSurface = try kernel.outputSurface(at: 0)
            maxValueSurface = try kernel.outputSurface(at: 1)
        } catch {
            throw RealModelInferenceError.runtimeFailure("Llama fused RMSNorm classifier surfaces unavailable: \(error)")
        }
        return CompiledClassifier(
            kernel: kernel,
            inputSurface: inputSurface,
            outputSurface: outputSurface,
            maxValueSurface: maxValueSurface
        )
    }

    private static func compileLlamaFactoredClassifier(
        config: MultiModelConfig,
        assets: LlamaTopLevelAssets,
        spatial: Int
    ) throws -> CompiledClassifier {
        guard let factoredOutputHead = assets.factoredOutputHead else {
            throw RealModelInferenceError.runtimeFailure("Factored llama classifier requested without factorized head weights")
        }
        guard config.dModel == ModelConfig.dim else {
            throw RealModelInferenceError.runtimeFailure(
                "Factored llama classifier currently requires dModel \(ModelConfig.dim), got \(config.dModel)"
            )
        }

        let projColsPerGroup = config.dModel / factoredOutputHead.groups
        let expColsPerGroup = factoredOutputHead.bottleneck / factoredOutputHead.groups
        let generator = FactoredGenerationRMSNormClassifierGenerator(
            vocabSize: config.vocab,
            bottleneck: factoredOutputHead.bottleneck,
            laneSpatial: spatial,
            groups: factoredOutputHead.groups
        )
        let rmsBlob = assets.finalNormGamma.withUnsafeBufferPointer { ptr in
            WeightBlob.build(from: ptr, rows: 1, cols: config.dModel)
        }
        let projBlob = buildGroupedWeightBlob(
            from: factoredOutputHead.projection,
            rows: factoredOutputHead.bottleneck,
            colsPerGroup: projColsPerGroup,
            groups: factoredOutputHead.groups
        )
        let expBlob = buildGroupedWeightBlob(
            from: factoredOutputHead.expansion,
            rows: config.vocab,
            colsPerGroup: expColsPerGroup,
            groups: factoredOutputHead.groups
        )
        let kernel: ANEKernel
        do {
            kernel = try ANEKernel(
                milText: generator.milText,
                weights: [
                    (path: "@model_path/weights/rms_final.bin", data: rmsBlob),
                    (path: "@model_path/weights/cls_proj.bin", data: projBlob),
                    (path: "@model_path/weights/cls_expand.bin", data: expBlob),
                ],
                inputSizes: generator.inputByteSizes,
                outputSizes: generator.outputByteSizes
            )
        } catch {
            throw RealModelInferenceError.runtimeFailure("Llama factored classifier compilation failed: \(error)")
        }

        let inputSurface: IOSurfaceRef
        let outputSurface: IOSurfaceRef
        do {
            inputSurface = try kernel.inputSurface(at: 0)
            outputSurface = try kernel.outputSurface(at: 0)
        } catch {
            throw RealModelInferenceError.runtimeFailure("Llama factored classifier surfaces unavailable: \(error)")
        }

        return CompiledClassifier(
            kernel: kernel,
            inputSurface: inputSurface,
            outputSurface: outputSurface,
            maxValueSurface: nil
        )
    }

    private static func buildLlamaHeadGraph(
        config: MultiModelConfig,
        assets: LlamaTopLevelAssets,
        spatial: Int,
        inputDType: ANEDType = .fp32,
        outputDType: ANEDType = .fp32
    ) -> ANEGraph {
        var graph = ANEGraph()
        let input = try! graph.input(
            "x",
            dtype: inputDType,
            shape: try! ANEShape(channels: config.dModel, spatial: spatial)
        )
        let x16 = inputDType == .fp16 ? input : try! graph.cast("final_rms_x16", input: input, to: .fp16)
        let norm = try! graph.rmsNorm128(
            "final_rms",
            input: x16,
            dim: config.dModel,
            spatial: spatial,
            eps: config.normEps,
            weightPath: assets.finalNormGammaPath
        )
        let output = outputDType == .fp16 ? norm : try! graph.cast("hidden", input: norm, to: .fp32)
        _ = try! graph.output(output, name: "hidden")
        return graph
    }

    private static func buildGPT2AttentionBlockGraph(
        layerIndex: Int,
        config: MultiModelConfig,
        paths: LayerWeightPaths,
        spatial: Int,
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> ANEGraph {
        var graph = ANEGraph()
        let prefix = "layer\(layerIndex)"
        let input = try! graph.input(
            "x",
            dtype: .fp32,
            shape: try! ANEShape(channels: config.dModel, spatial: spatial)
        )
        let x16 = try! graph.cast("\(prefix)_x16", input: input, to: .fp16)
        let ln1: Int
        switch gpt2NormKind(environment: environment) {
        case .layerNorm:
            ln1 = try! graph.layerNorm128(
                "\(prefix)_ln1",
                input: x16,
                dim: config.dModel,
                spatial: spatial,
                eps: config.normEps,
                gammaPath: paths.rmsAtt,
                betaPath: replacingGammaSuffix(in: paths.rmsAtt)
            )
        case .rmsNorm:
            ln1 = try! graph.rmsNorm128(
                "\(prefix)_ln1",
                input: x16,
                dim: config.dModel,
                spatial: spatial,
                eps: config.normEps,
                weightPath: paths.rmsAtt
            )
        }
        let q = try! graph.linear128(
            "\(prefix)_q",
            input: ln1,
            inDim: config.dModel,
            outDim: config.dModel,
            spatial: spatial,
            weightPath: paths.wq,
            biasPath: paths.bq
        )
        let k = try! graph.linear128(
            "\(prefix)_k",
            input: ln1,
            inDim: config.dModel,
            outDim: config.dModel,
            spatial: spatial,
            weightPath: paths.wk,
            biasPath: paths.bk
        )
        let v = try! graph.linear128(
            "\(prefix)_v",
            input: ln1,
            inDim: config.dModel,
            outDim: config.dModel,
            spatial: spatial,
            weightPath: paths.wv,
            biasPath: paths.bv
        )
        let attn = try! graph.causalAttention128(
            "\(prefix)_attn",
            q: q,
            k: k,
            v: v,
            nHeads: config.nHead,
            headDim: config.headDim,
            spatial: spatial,
            maskPath: layerMaskPath(for: paths, spatial: spatial)
        )
        let projected = try! graph.linear128(
            "\(prefix)_attn_proj",
            input: attn,
            inDim: config.dModel,
            outDim: config.dModel,
            spatial: spatial,
            weightPath: paths.wo,
            biasPath: paths.bo
        )
        let residual = try! graph.add("\(prefix)_res1_out", x: x16, y: projected)
        let hidden = try! graph.cast("hidden", input: residual, to: .fp32)
        let kCache = try! graph.cast("k_cache", input: k, to: .fp32)
        let vCache = try! graph.cast("v_cache", input: v, to: .fp32)
        _ = try! graph.output(hidden, name: "hidden")
        _ = try! graph.output(kCache, name: "k_cache")
        _ = try! graph.output(vCache, name: "v_cache")
        return graph
    }

    private static func buildGPT2FFNBlockGraph(
        layerIndex: Int,
        config: MultiModelConfig,
        paths: LayerWeightPaths,
        spatial: Int,
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> ANEGraph {
        var graph = ANEGraph()
        let prefix = "layer\(layerIndex)"
        let input = try! graph.input(
            "x",
            dtype: .fp32,
            shape: try! ANEShape(channels: config.dModel, spatial: spatial)
        )
        let x16 = try! graph.cast("\(prefix)_x16", input: input, to: .fp16)
        let ln2: Int
        switch gpt2NormKind(environment: environment) {
        case .layerNorm:
            ln2 = try! graph.layerNorm128(
                "\(prefix)_ln2",
                input: x16,
                dim: config.dModel,
                spatial: spatial,
                eps: config.normEps,
                gammaPath: paths.rmsFfn,
                betaPath: replacingGammaSuffix(in: paths.rmsFfn)
            )
        case .rmsNorm:
            ln2 = try! graph.rmsNorm128(
                "\(prefix)_ln2",
                input: x16,
                dim: config.dModel,
                spatial: spatial,
                eps: config.normEps,
                weightPath: paths.rmsFfn
            )
        }
        let ffn = try! graph.ffn128(
            "\(prefix)_ffn",
            input: ln2,
            inDim: config.dModel,
            hiddenDim: config.hiddenDim,
            spatial: spatial,
            w1Path: paths.w1,
            b1Path: paths.b1,
            w2Path: paths.w2,
            b2Path: paths.b2,
            activation: .gelu
        )
        let residual = try! graph.add("\(prefix)_res2_out", x: x16, y: ffn)
        let hidden = try! graph.cast("hidden", input: residual, to: .fp32)
        _ = try! graph.output(hidden, name: "hidden")
        return graph
    }

    private static func buildGPT2HeadGraph(
        config: MultiModelConfig,
        assets: GPT2TopLevelAssets,
        spatial: Int,
        inputDType: ANEDType = .fp32,
        outputDType: ANEDType = .fp32,
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> ANEGraph {
        var graph = ANEGraph()
        let input = try! graph.input(
            "x",
            dtype: inputDType,
            shape: try! ANEShape(channels: config.dModel, spatial: spatial)
        )
        let x16 = inputDType == .fp16 ? input : try! graph.cast("final_ln_x16", input: input, to: .fp16)
        let norm: Int
        switch gpt2NormKind(environment: environment) {
        case .layerNorm:
            norm = try! graph.layerNorm128(
                "final_ln",
                input: x16,
                dim: config.dModel,
                spatial: spatial,
                eps: config.normEps,
                gammaPath: assets.finalNormGammaPath,
                betaPath: assets.finalNormBetaPath
            )
        case .rmsNorm:
            norm = try! graph.rmsNorm128(
                "final_ln",
                input: x16,
                dim: config.dModel,
                spatial: spatial,
                eps: config.normEps,
                weightPath: assets.finalNormGammaPath
            )
        }
        let output = outputDType == .fp16 ? norm : try! graph.cast("hidden", input: norm, to: .fp32)
        _ = try! graph.output(output, name: "hidden")
        return graph
    }

    private static func firstInputSurface(from layers: borrowing LayerStorage<CompiledLayer>) throws -> IOSurfaceRef {
        guard layers.count > 0 else {
            throw RealModelInferenceError.runtimeFailure("No compiled layers were produced")
        }
        do {
            let inputSurface = try layers[0].attentionKernel.inputSurface(at: 0)
            for layerIndex in 1..<layers.count {
                try layers[layerIndex].attentionKernel.rebindInput(at: 0, to: layers[layerIndex - 1].outputSurface)
            }
            return inputSurface
        } catch {
            throw RealModelInferenceError.runtimeFailure("Failed to chain layer surfaces: \(error)")
        }
    }

    private static func loadTokenizer(
        config: MultiModelConfig,
        tokenizerDirURL: URL
    ) throws -> LoadedTokenizer {
        switch config.architecture {
        case .gpt2:
            let vocabURL = tokenizerDirURL.appendingPathComponent("vocab.json")
            let mergesURL = tokenizerDirURL.appendingPathComponent("merges.txt")
            guard FileManager.default.fileExists(atPath: vocabURL.path) else {
                throw RealModelInferenceError.missingPath(vocabURL.path)
            }
            guard FileManager.default.fileExists(atPath: mergesURL.path) else {
                throw RealModelInferenceError.missingPath(mergesURL.path)
            }
            do {
                return .gpt2(try GPT2BPETokenizer(vocabURL: vocabURL, mergesURL: mergesURL))
            } catch {
                throw RealModelInferenceError.runtimeFailure("Failed to load GPT-2 tokenizer: \(error)")
            }
        case .llama:
            // Try SentencePiece first (Llama, Mistral)
            let spCandidates = ["tokenizer.model", "tokenizer.bin"]
            for candidate in spCandidates {
                let url = tokenizerDirURL.appendingPathComponent(candidate)
                if FileManager.default.fileExists(atPath: url.path) {
                    do {
                        return .sentencePiece(try SentencePieceTokenizer(modelURL: url))
                    } catch {
                        throw RealModelInferenceError.runtimeFailure("Failed to load SentencePiece tokenizer: \(error)")
                    }
                }
            }
            let tokenizerJSONURL = tokenizerDirURL.appendingPathComponent("tokenizer.json")
            if FileManager.default.fileExists(atPath: tokenizerJSONURL.path) {
                do {
                    return .gpt2(try GPT2BPETokenizer(tokenizerJSONURL: tokenizerJSONURL))
                } catch {
                    throw RealModelInferenceError.runtimeFailure("Failed to load tokenizer.json BPE tokenizer: \(error)")
                }
            }
            // Fallback to GPT-2 BPE (Qwen uses BPE with llama-family architecture)
            let vocabURL = tokenizerDirURL.appendingPathComponent("vocab.json")
            let mergesURL = tokenizerDirURL.appendingPathComponent("merges.txt")
            if FileManager.default.fileExists(atPath: vocabURL.path),
               FileManager.default.fileExists(atPath: mergesURL.path) {
                do {
                    return .gpt2(try GPT2BPETokenizer(vocabURL: vocabURL, mergesURL: mergesURL))
                } catch {
                    throw RealModelInferenceError.runtimeFailure("Failed to load GPT-2 BPE tokenizer: \(error)")
                }
            }
            throw RealModelInferenceError.missingPath(
                "No tokenizer found in \(tokenizerDirURL.path) — tried tokenizer.model, tokenizer.bin, tokenizer.json, vocab.json+merges.txt"
            )
        }
    }

    private static func validateConfig(_ config: MultiModelConfig) throws {
        guard config.nLayer > 0 else {
            throw RealModelInferenceError.invalidConfig("nLayer must be > 0")
        }
        guard config.nHead > 0, config.nKVHead > 0 else {
            throw RealModelInferenceError.invalidConfig("nHead and nKVHead must be > 0")
        }
        guard config.dModel > 0, config.headDim > 0, config.hiddenDim > 0 else {
            throw RealModelInferenceError.invalidConfig("dModel, headDim, and hiddenDim must be > 0")
        }
        guard config.vocab > 0, config.maxSeq > 0 else {
            throw RealModelInferenceError.invalidConfig("vocab and maxSeq must be > 0")
        }
        guard config.nHead * config.headDim > 0, config.nKVHead * config.headDim > 0 else {
            throw RealModelInferenceError.invalidConfig("attention dimensions must be > 0")
        }
        guard config.nHead % config.nKVHead == 0 else {
            throw RealModelInferenceError.invalidConfig("nHead must be divisible by nKVHead")
        }
    }

    private static func requireCompileSpatialCapacity(channels: Int, maxSeq: Int) throws -> Int {
        let minimumSpatial = minimumCompileSpatial(channels: channels)
        guard minimumSpatial <= maxSeq else {
            throw RealModelInferenceError.invalidConfig(
                "maxSeq \(maxSeq) is too small for ANE minimum boundary size with dModel \(channels); requires at least \(minimumSpatial)"
            )
        }
        return minimumSpatial
    }

    private static func compileBucket(for tokenCount: Int, channels: Int, maxSeq: Int) throws -> Int {
        let minimumSpatial = try requireCompileSpatialCapacity(channels: channels, maxSeq: maxSeq)
        return max(spatialBucket(for: tokenCount, maxSeq: maxSeq), minimumSpatial)
    }

    private static func validateDirectory(_ url: URL) throws {
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory), isDirectory.boolValue else {
            throw RealModelInferenceError.missingPath(url.path)
        }
    }

    private static func validateMetadataIfPresent(
        config: MultiModelConfig,
        weightDirURL: URL
    ) throws {
        let metadataURL = weightDirURL.appendingPathComponent("metadata.json")
        guard FileManager.default.fileExists(atPath: metadataURL.path) else {
            return
        }

        let data: Data
        do {
            data = try Data(contentsOf: metadataURL)
        } catch {
            throw RealModelInferenceError.runtimeFailure("Failed to read metadata.json: \(error)")
        }

        let object: Any
        do {
            object = try JSONSerialization.jsonObject(with: data)
        } catch {
            throw RealModelInferenceError.runtimeFailure("metadata.json is not valid JSON: \(error)")
        }

        guard let metadata = object as? [String: Any] else {
            throw RealModelInferenceError.runtimeFailure("metadata.json must be a JSON object")
        }

        try requireMetadata(metadata, key: "architecture", expected: architectureName(config.architecture))
        try requireMetadata(metadata, key: "nLayer", expected: config.nLayer)
        try requireMetadata(metadata, key: "nHead", expected: config.nHead)
        try requireMetadata(metadata, key: "nKVHead", expected: config.nKVHead)
        try requireMetadata(metadata, key: "dModel", expected: config.dModel)
        try requireMetadata(metadata, key: "headDim", expected: config.headDim)
        try requireMetadata(metadata, key: "hiddenDim", expected: config.hiddenDim)
        try requireMetadata(metadata, key: "vocab", expected: config.vocab)
        try requireMetadata(metadata, key: "maxSeq", expected: config.maxSeq)
    }

    private static func requireMetadata(
        _ metadata: [String: Any],
        key: String,
        expected: String
    ) throws {
        guard let actual = metadata[key] else { return }
        let actualString: String
        if let number = actual as? NSNumber {
            actualString = number.stringValue
        } else {
            actualString = String(describing: actual)
        }
        guard actualString == expected else {
            throw RealModelInferenceError.invalidMetadata(field: key, expected: expected, actual: actualString)
        }
    }

    private static func requireMetadata(
        _ metadata: [String: Any],
        key: String,
        expected: Int
    ) throws {
        try requireMetadata(metadata, key: key, expected: String(expected))
    }

    private static func requiredFile(
        root: URL,
        candidates: [String],
        label: String
    ) throws -> String {
        for candidate in candidates {
            let path = root.appendingPathComponent(candidate).path
            if FileManager.default.fileExists(atPath: path) {
                return path
            }
        }
        throw RealModelInferenceError.missingPath(
            "\(root.path)/<\(label): \(candidates.joined(separator: " | "))>"
        )
    }

    static func loadWeightTable(at path: String, expectedCount: Int) throws -> [Float] {
        let values: [Float]
        do {
            values = try BlobWeightLoader.load(from: path)
        } catch {
            throw RealModelInferenceError.runtimeFailure("Failed to load weight blob \(path): \(error)")
        }
        guard values.count == expectedCount else {
            throw RealModelInferenceError.invalidWeightCount(path: path, expected: expectedCount, actual: values.count)
        }
        return values
    }

    static func loadWeightTable(at path: String, allowedCounts: [Int]) throws -> [Float] {
        let values: [Float]
        do {
            values = try BlobWeightLoader.load(from: path)
        } catch {
            throw RealModelInferenceError.runtimeFailure("Failed to load weight blob \(path): \(error)")
        }
        guard allowedCounts.contains(values.count) else {
            let expected = allowedCounts.map(String.init).joined(separator: " or ")
            throw RealModelInferenceError.runtimeFailure(
                "Unexpected weight count for \(path): expected \(expected), got \(values.count)"
            )
        }
        return values
    }

    static func loadRawFP16WeightTableIfNoExactFloat32Sidecar(
        at path: String,
        expectedCount: Int
    ) throws -> [UInt16]? {
        let sidecarPath = exactFloat32SidecarPath(forBlobPath: path)
        guard !FileManager.default.fileExists(atPath: sidecarPath) else {
            return nil
        }

        let header: BlobWeightLoader.Header
        do {
            header = try BlobWeightLoader.readHeader(from: path)
        } catch {
            throw RealModelInferenceError.runtimeFailure("Failed to read weight blob header \(path): \(error)")
        }

        let expectedBytes = expectedCount * MemoryLayout<UInt16>.stride
        guard Int(header.dataSize) == expectedBytes else {
            throw RealModelInferenceError.invalidWeightCount(
                path: path,
                expected: expectedCount,
                actual: Int(header.dataSize) / MemoryLayout<UInt16>.stride
            )
        }

        let handle: FileHandle
        do {
            handle = try FileHandle(forReadingFrom: URL(fileURLWithPath: path))
        } catch {
            throw RealModelInferenceError.runtimeFailure("Failed to open weight blob \(path): \(error)")
        }
        defer { try? handle.close() }

        do {
            try handle.seek(toOffset: UInt64(header.dataOffset))
        } catch {
            throw RealModelInferenceError.runtimeFailure("Failed to seek weight blob \(path): \(error)")
        }

        let payload: Data
        do {
            payload = try handle.read(upToCount: expectedBytes) ?? Data()
        } catch {
            throw RealModelInferenceError.runtimeFailure("Failed to read weight blob payload \(path): \(error)")
        }
        guard payload.count == expectedBytes else {
            throw RealModelInferenceError.invalidWeightCount(
                path: path,
                expected: expectedCount,
                actual: payload.count / MemoryLayout<UInt16>.stride
            )
        }

        return payload.withUnsafeBytes { raw in
            (0..<expectedCount).map { index in
                let bits = raw.loadUnaligned(
                    fromByteOffset: index * MemoryLayout<UInt16>.stride,
                    as: UInt16.self
                )
                return UInt16(littleEndian: bits)
            }
        }
    }

    static func exactFloat32SidecarPath(forBlobPath path: String) -> String {
        if path.hasSuffix(".bin") {
            return String(path.dropLast(4)) + ".float32.bin"
        }
        return path + ".float32"
    }

    static func loadExactFloat32WeightTable(
        at path: String,
        expectedCount: Int
    ) throws -> [Float]? {
        let sidecarPath = exactFloat32SidecarPath(forBlobPath: path)
        guard FileManager.default.fileExists(atPath: sidecarPath) else {
            return nil
        }

        let data: Data
        do {
            data = try Data(contentsOf: URL(fileURLWithPath: sidecarPath))
        } catch {
            throw RealModelInferenceError.runtimeFailure("Failed to read float32 sidecar \(sidecarPath): \(error)")
        }

        let scalarSize = MemoryLayout<UInt32>.stride
        let expectedBytes = expectedCount * scalarSize
        guard data.count == expectedBytes else {
            throw RealModelInferenceError.invalidWeightCount(
                path: sidecarPath,
                expected: expectedCount,
                actual: data.count / scalarSize
            )
        }

        return data.withUnsafeBytes { raw in
            (0..<expectedCount).map { index in
                let bits = raw.loadUnaligned(fromByteOffset: index * scalarSize, as: UInt32.self)
                return Float(bitPattern: UInt32(littleEndian: bits))
            }
        }
    }

    static func loadWeightTablePreferringFloat32Sidecar(
        at path: String,
        expectedCount: Int
    ) throws -> [Float] {
        if let exactValues = try loadExactFloat32WeightTable(at: path, expectedCount: expectedCount) {
            return exactValues
        }
        return try loadWeightTable(at: path, expectedCount: expectedCount)
    }

    private static func loadTensor(
        _ tensor: borrowing TensorBuffer,
        from path: String,
        expectedCount: Int
    ) throws {
        let values = try loadWeightTable(at: path, expectedCount: expectedCount)
        tensor.withUnsafeMutableBufferPointer { dst in
            values.withUnsafeBufferPointer { src in
                guard let dstBase = dst.baseAddress, let srcBase = src.baseAddress else {
                    return
                }
                dstBase.update(from: srcBase, count: expectedCount)
            }
        }
    }

    private static func buildGroupedWeightBlob(
        from weights: [Float],
        rows: Int,
        colsPerGroup: Int,
        groups: Int
    ) -> Data {
        let compactCount = rows * colsPerGroup
        let repacked: [Float] = weights.withUnsafeBufferPointer { buffer in
            if groups == 1 || buffer.count == compactCount {
                return Array(buffer)
            }

            let denseCols = colsPerGroup * groups
            precondition(rows.isMultiple(of: groups))
            precondition(buffer.count == rows * denseCols)

            let rowsPerGroup = rows / groups
            var compact = [Float](repeating: 0, count: compactCount)
            for row in 0..<rows {
                let group = row / rowsPerGroup
                let srcStart = row * denseCols + group * colsPerGroup
                let dstStart = row * colsPerGroup
                for col in 0..<colsPerGroup {
                    compact[dstStart + col] = buffer[srcStart + col]
                }
            }
            return compact
        }
        return WeightBlob.build(from: repacked, rows: rows, cols: colsPerGroup)
    }

    private static func fileExists(at path: String?) -> Bool {
        guard let path else { return false }
        return FileManager.default.fileExists(atPath: path)
    }

    private static func resolveBundleWeightReference(
        _ reference: String,
        weightDirURL: URL
    ) throws -> String {
        let normalized = reference.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !normalized.isEmpty else {
            throw RealModelInferenceError.invalidConfig("Bundle output-head reference must not be empty")
        }
        let relative = normalized.hasPrefix("weights/")
            ? String(normalized.dropFirst("weights/".count))
            : normalized
        let resolved = weightDirURL.appendingPathComponent(relative).standardizedFileURL.path
        guard FileManager.default.fileExists(atPath: resolved) else {
            throw RealModelInferenceError.missingPath(resolved)
        }
        return resolved
    }

    private static func compileBlobPath(actualPath: String, rootDir: URL) -> String {
        let rootPath = rootDir.standardizedFileURL.path
        let filePath = URL(fileURLWithPath: actualPath).standardizedFileURL.path
        let relativePath: String
        if filePath.hasPrefix(rootPath + "/") {
            relativePath = String(filePath.dropFirst(rootPath.count + 1))
        } else {
            relativePath = URL(fileURLWithPath: filePath).lastPathComponent
        }
        return "@model_path/weights/\(relativePath)"
    }

    private static func rewriteMILWeightPaths(_ mil: String, rootDir: URL) -> String {
        mil.replacingOccurrences(
            of: rootDir.standardizedFileURL.path,
            with: "@model_path/weights"
        )
    }

    private static func canonicalBlobData(
        at path: String,
        expectedCount: Int? = nil
    ) throws -> Data {
        let values = try loadWeightTable(at: path, expectedCount: expectedCount ?? loadWeightCount(at: path))
        return WeightBlob.build(from: values, rows: 1, cols: values.count)
    }

    private static func loadWeightCount(at path: String) -> Int {
        if let values = try? BlobWeightLoader.load(from: path) {
            return values.count
        }
        return 0
    }

    private static func replacingGammaSuffix(in path: String) -> String {
        path.replacingOccurrences(of: "_gamma.bin", with: "_beta.bin")
    }

    private static func layerMaskPath(for paths: LayerWeightPaths, spatial: Int) -> String {
        URL(fileURLWithPath: paths.wq)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("masks", isDirectory: true)
            .appendingPathComponent("causal_\(spatial).bin")
            .path
    }

    private static func causalMaskBlob(seqLen: Int) -> Data {
        let minFP16: Float = -65_504
        var values = [Float](repeating: 0, count: seqLen * seqLen)
        for row in 0..<seqLen {
            for column in (row + 1)..<seqLen {
                values[row * seqLen + column] = minFP16
            }
        }
        return WeightBlob.build(from: values, rows: seqLen, cols: seqLen)
    }

    private static func architectureName(_ architecture: MultiModelConfig.Architecture) -> String {
        switch architecture {
        case .gpt2:
            return "gpt2"
        case .llama:
            return "llama"
        }
    }

    private static func milliseconds(from nanoseconds: UInt64) -> Double {
        Double(nanoseconds) / 1_000_000
    }

    @discardableResult
    private static func dumpDebugMIL(_ mil: String, filename: String) -> String {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(filename)
        try? mil.write(to: url, atomically: true, encoding: .utf8)
        return url.path
    }

    private static func emptyStorage<Element: ~Copyable>(_: Element.Type = Element.self) -> LayerStorage<Element> {
        LayerStorage<Element>(count: 0) { _ in
            fatalError("unreachable empty storage initializer")
        }
    }

    private func sampleToken<R: RandomNumberGenerator>(
        from logits: [Float],
        temperature: Float,
        using rng: inout R
    ) -> TokenID {
        if temperature <= 0 {
            let index = logits.enumerated().max(by: { $0.element < $1.element })?.offset ?? 0
            return TokenID(index)
        }

        let maxLogit = logits.max() ?? 0
        var scaled = [Double](repeating: 0, count: logits.count)
        scaled.reserveCapacity(logits.count)
        var total = 0.0
        for index in logits.indices {
            let value = exp(Double((logits[index] - maxLogit) / temperature))
            scaled[index] = value
            total += value
        }
        if !total.isFinite || total <= 0 {
            let index = logits.enumerated().max(by: { $0.element < $1.element })?.offset ?? 0
            return TokenID(index)
        }

        var threshold = Double.random(in: 0..<total, using: &rng)
        for index in scaled.indices {
            threshold -= scaled[index]
            if threshold <= 0 {
                return TokenID(index)
            }
        }
        return TokenID(max(0, scaled.count - 1))
    }

    private mutating func selectTokenFromNormalizedHidden<R: RandomNumberGenerator>(
        _ hidden: [Float],
        temperature: Float,
        using rng: inout R
    ) -> TokenID {
        if temperature <= 0 {
            let index = exactClassifierArgmax(hidden)
            return TokenID(index)
        }
        let logits = projectLogits(hidden)
        return sampleToken(from: logits, temperature: temperature, using: &rng)
    }

    private mutating func exactClassifierArgmax(_ hidden: [Float]) -> Int {
        precondition(hidden.count == config.dModel)
        switch classifierStrategy {
        case .ane, .cpuPartitionedFP32:
            let blockSize = Self.classifierArgmaxBlockSize
            return hidden.withUnsafeBufferPointer { hiddenBuffer in
                lmHeadWeights.withUnsafeBufferPointer { weightBuffer in
                    classifierBlockMaxNorms.withUnsafeBufferPointer { normsBuffer in
                        classifierLogitsScratch.withUnsafeMutableBufferPointer { scratchBuffer in
                            guard let hiddenBase = hiddenBuffer.baseAddress,
                                  let weightBase = weightBuffer.baseAddress,
                                  let normsBase = normsBuffer.baseAddress,
                                  let scratchBase = scratchBuffer.baseAddress else {
                                return 0
                            }
                            return Self.partitionedArgmax(
                                classifier: weightBase,
                                input: hiddenBase,
                                logitsScratch: scratchBase,
                                blockMaxNorms: normsBase,
                                vocabSize: config.vocab,
                                dim: config.dModel,
                                blockSize: blockSize
                            )
                        }
                    }
                }
            }
        case .cpuFP16Tiled:
            guard case let .llama(assets) = assets, let lmHeadFP16 = assets.lmHeadFP16 else {
                return hidden.withUnsafeBufferPointer { hiddenBuffer in
                    lmHeadWeights.withUnsafeBufferPointer { weightBuffer in
                        classifierBlockMaxNorms.withUnsafeBufferPointer { normsBuffer in
                            classifierLogitsScratch.withUnsafeMutableBufferPointer { scratchBuffer in
                                guard let hiddenBase = hiddenBuffer.baseAddress,
                                      let weightBase = weightBuffer.baseAddress,
                                      let normsBase = normsBuffer.baseAddress,
                                      let scratchBase = scratchBuffer.baseAddress else {
                                    return 0
                                }
                                return Self.partitionedArgmax(
                                    classifier: weightBase,
                                    input: hiddenBase,
                                    logitsScratch: scratchBase,
                                    blockMaxNorms: normsBase,
                                    vocabSize: config.vocab,
                                    dim: config.dModel,
                                    blockSize: Self.classifierArgmaxBlockSize
                                )
                            }
                        }
                    }
                }
            }
            return hidden.withUnsafeBufferPointer { hiddenBuffer in
                lmHeadFP16.withUnsafeBufferPointer { weightBuffer in
                    guard let hiddenBase = hiddenBuffer.baseAddress,
                          let weightBase = weightBuffer.baseAddress else {
                        return 0
                    }
                    return FP16TiledClassifier.tiledMatvecArgmax(
                        weights: weightBase,
                        input: hiddenBase,
                        vocabSize: config.vocab,
                        dim: config.dModel
                    )
                }
            }
        }
    }

    private func projectLogits(_ hidden: [Float]) -> [Float] {
        precondition(hidden.count == config.dModel)
        var logits = [Float](repeating: 0, count: config.vocab)
        logits.withUnsafeMutableBufferPointer { logitsBuffer in
            lmHeadWeights.withUnsafeBufferPointer { weightBuffer in
                hidden.withUnsafeBufferPointer { hiddenBuffer in
                    guard let logitsBase = logitsBuffer.baseAddress,
                          let weightBase = weightBuffer.baseAddress,
                          let hiddenBase = hiddenBuffer.baseAddress else {
                        return
                    }
                    vDSP_mmul(
                        weightBase,
                        1,
                        hiddenBase,
                        1,
                        logitsBase,
                        1,
                        vDSP_Length(config.vocab),
                        1,
                        vDSP_Length(config.dModel)
                    )
                }
            }
        }
        return logits
    }

    private static func precomputeClassifierBlockMaxNorms(
        classifier: UnsafePointer<Float>,
        vocabSize: Int,
        dim: Int,
        blockSize: Int
    ) -> [Float] {
        precondition(vocabSize > 0)
        precondition(dim > 0)
        precondition(blockSize > 0)

        let numBlocks = (vocabSize + blockSize - 1) / blockSize
        var blockMaxNorms = [Float](repeating: 0, count: numBlocks)

        var blockIndex = 0
        var blockStart = 0
        while blockStart < vocabSize {
            let blockEnd = min(blockStart + blockSize, vocabSize)
            var blockMax: Float = 0

            for rowIndex in blockStart..<blockEnd {
                let rowBase = rowIndex * dim
                var sumOfSquares: Float = 0
                vDSP_svesq(classifier.advanced(by: rowBase), 1, &sumOfSquares, vDSP_Length(dim))
                let rowNorm = sqrtf(sumOfSquares)
                if rowNorm > blockMax {
                    blockMax = rowNorm
                }
            }

            blockMaxNorms[blockIndex] = blockMax
            blockIndex += 1
            blockStart = blockEnd
        }

        return blockMaxNorms
    }

    private static func partitionedArgmax(
        classifier: UnsafePointer<Float>,
        input: UnsafePointer<Float>,
        logitsScratch: UnsafeMutablePointer<Float>,
        blockMaxNorms: UnsafePointer<Float>,
        vocabSize: Int,
        dim: Int,
        blockSize: Int
    ) -> Int {
        var inputNormSquared: Float = 0
        vDSP_svesq(input, 1, &inputNormSquared, vDSP_Length(dim))
        let inputNorm = sqrtf(inputNormSquared)

        var bestIndex = 0
        var bestValue: Float = -.infinity
        var blockIndex = 0
        var blockStart = 0

        while blockStart < vocabSize {
            let blockEnd = min(blockStart + blockSize, vocabSize)
            let blockCount = blockEnd - blockStart

            if blockIndex > 0, bestValue > -.infinity {
                let upperBound = blockMaxNorms[blockIndex] * inputNorm
                if upperBound < bestValue {
                    blockIndex += 1
                    blockStart = blockEnd
                    continue
                }
            }

            vDSP_mmul(
                classifier.advanced(by: blockStart * dim),
                1,
                input,
                1,
                logitsScratch,
                1,
                vDSP_Length(blockCount),
                1,
                vDSP_Length(dim)
            )

            var blockMaxValue: Float = 0
            var blockMaxIndex: vDSP_Length = 0
            vDSP_maxvi(logitsScratch, 1, &blockMaxValue, &blockMaxIndex, vDSP_Length(blockCount))
            if blockMaxValue > bestValue {
                bestValue = blockMaxValue
                bestIndex = blockStart + Int(blockMaxIndex)
            }

            blockIndex += 1
            blockStart = blockEnd
        }

        return bestIndex
    }

    private static func extractSpatialSlice(
        from values: [Float],
        channels: Int,
        spatial: Int,
        spatialIndex: Int
    ) -> [Float] {
        precondition(values.count == channels * spatial)
        precondition(spatialIndex >= 0 && spatialIndex < spatial)
        var output = [Float](repeating: 0, count: channels)
        for channel in 0..<channels {
            output[channel] = values[channel * spatial + spatialIndex]
        }
        return output
    }

    private static func composeTestingEmbeddingInput(
        config: MultiModelConfig,
        tokens: [TokenID],
        tokenEmbedding: [Float],
        positionEmbedding: [Float]
    ) -> [Float] {
        var output = [Float](repeating: 0, count: config.dModel * tokens.count)
        for tokenIndex in tokens.indices {
            let token = Int(tokens[tokenIndex])
            precondition(token >= 0 && token < config.vocab)
            let tokenBase = token * config.dModel
            let positionBase = tokenIndex * config.dModel
            for channel in 0..<config.dModel {
                let positionValue = positionEmbedding.isEmpty ? 0 : positionEmbedding[positionBase + channel]
                output[channel * tokens.count + tokenIndex] =
                    tokenEmbedding[tokenBase + channel] +
                    positionValue
            }
        }
        return output
    }

    private static func writeTestingIncrementalEmbedding(
        config: MultiModelConfig,
        token: TokenID,
        position: Int,
        tokenEmbedding: [Float],
        positionEmbedding: [Float],
        into buffer: borrowing TensorBuffer
    ) {
        precondition(position >= 0 && position < config.maxSeq)
        let tokenBase = Int(token) * config.dModel
        let positionBase = position * config.dModel
        buffer.withUnsafeMutableBufferPointer { dst in
            for channel in 0..<config.dModel {
                let positionValue = positionEmbedding.isEmpty ? 0 : positionEmbedding[positionBase + channel]
                dst[channel] =
                    tokenEmbedding[tokenBase + channel] +
                    positionValue
            }
        }
    }

    private static func writeFP32(
        to surface: IOSurfaceRef,
        data: UnsafeBufferPointer<Float>
    ) throws {
        let byteCount = data.count * MemoryLayout<Float>.stride
        guard IOSurfaceGetAllocSize(surface) >= byteCount else {
            throw RealModelInferenceError.runtimeFailure("IOSurface too small for \(byteCount)-byte fp32 write")
        }
        guard IOSurfaceLock(surface, [], nil) == kIOReturnSuccess else {
            throw RealModelInferenceError.runtimeFailure("IOSurface lock failed for fp32 write")
        }
        defer { IOSurfaceUnlock(surface, [], nil) }
        guard let source = data.baseAddress else {
            throw RealModelInferenceError.runtimeFailure("IOSurface base address unavailable for fp32 write")
        }
        let baseAddress = IOSurfaceGetBaseAddress(surface)
        memcpy(baseAddress, source, byteCount)
    }

    private static func writeFP32SpatialSlice(
        to surface: IOSurfaceRef,
        spatialIndex: Int,
        spatial: Int,
        data: UnsafeBufferPointer<Float>,
        channels: Int
    ) throws {
        precondition(spatial > 0)
        precondition(spatialIndex >= 0 && spatialIndex < spatial)
        precondition(data.count == channels)
        let requiredBytes = channels * spatial * MemoryLayout<Float>.stride
        guard IOSurfaceGetAllocSize(surface) >= requiredBytes else {
            throw RealModelInferenceError.runtimeFailure("IOSurface too small for fp32 spatial-slice write")
        }
        guard IOSurfaceLock(surface, [], nil) == kIOReturnSuccess else {
            throw RealModelInferenceError.runtimeFailure("IOSurface lock failed for fp32 spatial-slice write")
        }
        defer { IOSurfaceUnlock(surface, [], nil) }
        guard let source = data.baseAddress else {
            throw RealModelInferenceError.runtimeFailure("IOSurface base address unavailable for fp32 spatial-slice write")
        }
        let baseAddress = IOSurfaceGetBaseAddress(surface).assumingMemoryBound(to: Float.self)
        for channel in 0..<channels {
            baseAddress[channel * spatial + spatialIndex] = source[channel]
        }
    }

    private static func readFP32(
        from surface: IOSurfaceRef,
        into buffer: UnsafeMutableBufferPointer<Float>
    ) throws {
        let byteCount = buffer.count * MemoryLayout<Float>.stride
        guard IOSurfaceGetAllocSize(surface) >= byteCount else {
            throw RealModelInferenceError.runtimeFailure("IOSurface too small for \(byteCount)-byte fp32 read")
        }
        guard IOSurfaceLock(surface, .readOnly, nil) == kIOReturnSuccess else {
            throw RealModelInferenceError.runtimeFailure("IOSurface lock failed for fp32 read")
        }
        defer { IOSurfaceUnlock(surface, .readOnly, nil) }
        guard let destination = buffer.baseAddress else {
            throw RealModelInferenceError.runtimeFailure("IOSurface base address unavailable for fp32 read")
        }
        let baseAddress = IOSurfaceGetBaseAddress(surface)
        memcpy(destination, baseAddress, byteCount)
    }

    private static func readFP32SpatialSlice(
        from surface: IOSurfaceRef,
        spatialIndex: Int,
        spatial: Int,
        into buffer: UnsafeMutableBufferPointer<Float>,
        channels: Int
    ) throws {
        precondition(spatial > 0)
        precondition(spatialIndex >= 0 && spatialIndex < spatial)
        precondition(buffer.count == channels)
        let requiredBytes = channels * spatial * MemoryLayout<Float>.stride
        guard IOSurfaceGetAllocSize(surface) >= requiredBytes else {
            throw RealModelInferenceError.runtimeFailure("IOSurface too small for fp32 spatial-slice read")
        }
        guard IOSurfaceLock(surface, .readOnly, nil) == kIOReturnSuccess else {
            throw RealModelInferenceError.runtimeFailure("IOSurface lock failed for fp32 spatial-slice read")
        }
        defer { IOSurfaceUnlock(surface, .readOnly, nil) }
        guard let destination = buffer.baseAddress else {
            throw RealModelInferenceError.runtimeFailure("IOSurface base address unavailable for fp32 spatial-slice read")
        }
        let baseAddress = IOSurfaceGetBaseAddress(surface).assumingMemoryBound(to: Float.self)
        for channel in 0..<channels {
            destination[channel] = baseAddress[channel * spatial + spatialIndex]
        }
    }

    private static func copyFullFP16Surface(
        dst: IOSurfaceRef,
        src: IOSurfaceRef,
        channels: Int,
        spatial: Int
    ) throws {
        try SurfaceIO.copyFP16(
            dst: dst,
            dstChannelOffset: 0,
            src: src,
            srcChannelOffset: 0,
            channels: channels,
            spatial: spatial
        )
    }

    private static func evaluateGreedyClassifier(
        norm: borrowing CompiledHead,
        classifier: borrowing CompiledClassifier,
        headSpatial: Int,
        vocab: Int
    ) throws -> TokenID {
        do {
            try norm.kernel.eval()
            try classifier.kernel.eval()
            let argmax = try greedyArgmax(
                classifier: classifier,
                headSpatial: headSpatial,
                vocab: vocab
            )
            guard let token = TokenID(exactly: argmax.index) else {
                throw RealModelInferenceError.runtimeFailure(
                    "Greedy ANE classifier selected out-of-range token \(argmax.index)"
                )
            }
            return token
        } catch let error as RealModelInferenceError {
            throw error
        } catch {
            throw RealModelInferenceError.runtimeFailure("Hybrid greedy ANE head evaluation failed: \(error)")
        }
    }

    private static func greedyArgmax(
        classifier: borrowing CompiledClassifier,
        headSpatial: Int,
        vocab: Int
    ) throws -> SurfaceIO.FP16ArgmaxResult {
        if let maxValueSurface = classifier.maxValueSurface {
            return try SurfaceIO.argmaxFP16SpatialSliceWithHint(
                from: classifier.outputSurface,
                channelOffset: 0,
                spatialIndex: 0,
                spatial: headSpatial,
                channels: vocab,
                hintSurface: maxValueSurface,
                hintSpatialIndex: 0,
                hintSpatial: headSpatial
            )
        }
        return try SurfaceIO.argmaxFP16SpatialSlice(
            from: classifier.outputSurface,
            channelOffset: 0,
            spatialIndex: 0,
            spatial: headSpatial,
            channels: vocab
        )
    }

    private static func zeroSurface(_ surface: IOSurfaceRef) throws {
        guard IOSurfaceLock(surface, [], nil) == kIOReturnSuccess else {
            throw RealModelInferenceError.runtimeFailure("IOSurface lock failed for zero initialization")
        }
        defer { IOSurfaceUnlock(surface, [], nil) }
        memset(IOSurfaceGetBaseAddress(surface), 0, IOSurfaceGetAllocSize(surface))
    }

    private static func debugLogHybridCache(
        label: String,
        surface: IOSurfaceRef,
        maxSeq: Int,
        channels: Int,
        tokenCount: Int
    ) throws {
        var parts: [String] = []
        for tokenIndex in 0..<tokenCount {
            var slice = [Float](repeating: 0, count: channels)
            try slice.withUnsafeMutableBufferPointer { dst in
                try SurfaceIO.readFP16SpatialSlice(
                    from: surface,
                    channelOffset: 0,
                    spatialIndex: tokenIndex,
                    spatial: maxSeq,
                    into: dst,
                    channels: channels
                )
            }
            let values = slice.map { String(format: "%.4f", $0) }.joined(separator: ",")
            parts.append("t\(tokenIndex)=[\(values)]")
        }
        fputs("[hybrid-cache] \(label) \(parts.joined(separator: " "))\n", stderr)
    }

    private static func debugExpectedGPT2KPrefix(
        input: [Float],
        weights: borrowing LayerWeights,
        eps: Float,
        prefixChannels: Int
    ) -> [Float] {
        let dim = weights.dim
        precondition(input.count == dim)
        precondition(prefixChannels <= dim)

        let mean = input.reduce(0, +) / Float(dim)
        var variance: Float = 0
        for value in input {
            let centered = value - mean
            variance += centered * centered
        }
        variance /= Float(dim)
        let invStd = 1 / sqrt(variance + eps)

        var normalized = [Float](repeating: 0, count: dim)
        weights.rmsAtt.withUnsafeBufferPointer { gamma in
            weights.attentionNormBeta.withUnsafeBufferPointer { beta in
                for channel in 0..<dim {
                    normalized[channel] = ((input[channel] - mean) * invStd) * gamma[channel] + beta[channel]
                }
            }
        }

        var output = [Float](repeating: 0, count: prefixChannels)
        weights.Wk.withUnsafeBufferPointer { wk in
            weights.bk.withUnsafeBufferPointer { bias in
                for row in 0..<prefixChannels {
                    var accum = bias[row]
                    let rowBase = row * dim
                    for column in 0..<dim {
                        accum += wk[rowBase + column] * normalized[column]
                    }
                    output[row] = accum
                }
            }
        }
        return output
    }

    private static func debugExpectedGPT2KPrefixTransposed(
        input: [Float],
        weights: borrowing LayerWeights,
        eps: Float,
        prefixChannels: Int
    ) -> [Float] {
        let dim = weights.dim
        precondition(input.count == dim)
        precondition(prefixChannels <= dim)

        let mean = input.reduce(0, +) / Float(dim)
        var variance: Float = 0
        for value in input {
            let centered = value - mean
            variance += centered * centered
        }
        variance /= Float(dim)
        let invStd = 1 / sqrt(variance + eps)

        var normalized = [Float](repeating: 0, count: dim)
        weights.rmsAtt.withUnsafeBufferPointer { gamma in
            weights.attentionNormBeta.withUnsafeBufferPointer { beta in
                for channel in 0..<dim {
                    normalized[channel] = ((input[channel] - mean) * invStd) * gamma[channel] + beta[channel]
                }
            }
        }

        var output = [Float](repeating: 0, count: prefixChannels)
        weights.Wk.withUnsafeBufferPointer { wk in
            weights.bk.withUnsafeBufferPointer { bias in
                for row in 0..<prefixChannels {
                    var accum = bias[row]
                    for column in 0..<dim {
                        accum += wk[column * dim + row] * normalized[column]
                    }
                    output[row] = accum
                }
            }
        }
        return output
    }
}

private extension ANEGraph {
    mutating func constWeight128(_ name: String, shape: ANEShape, blobPath: String) throws -> Int {
        try constWeight(name, shape: shape, blobPath: blobPath, offset: 64)
    }

    mutating func linear128(
        _ prefix: String,
        input: Int,
        inDim: Int,
        outDim: Int,
        spatial: Int,
        weightPath: String,
        biasPath: String? = nil
    ) throws -> Int {
        let weight = try constWeight128(
            "\(prefix)_weight",
            shape: try ANEShape(batch: outDim, channels: inDim, height: 1, spatial: 1),
            blobPath: weightPath
        )
        let conv = try conv1x1(
            "\(prefix)_conv",
            input: input,
            weight: weight,
            bias: nil,
            outShape: try ANEShape(channels: outDim, spatial: spatial)
        )
        guard let biasPath else {
            return conv
        }
        let bias = try constWeight128(
            "\(prefix)_bias",
            shape: try ANEShape(channels: outDim, spatial: 1),
            blobPath: biasPath
        )
        return try add("\(prefix)_out", x: conv, y: bias)
    }

    mutating func layerNorm128(
        _ prefix: String,
        input: Int,
        dim: Int,
        spatial: Int,
        eps: Float,
        gammaPath: String,
        betaPath: String
    ) throws -> Int {
        let mean = try reduceMean("\(prefix)_mean", input: input, axis: 1, keepDims: true)
        let centered = try sub("\(prefix)_centered", x: input, y: mean)
        let sq = try mul("\(prefix)_sq", x: centered, y: centered)
        let variance = try reduceMean("\(prefix)_var", input: sq, axis: 1, keepDims: true)
        let epsNode = try constScalar("\(prefix)_eps", eps)
        let varEps = try add("\(prefix)_var_eps", x: variance, y: epsNode)
        let nhalf = try constScalar("\(prefix)_nhalf", -0.5)
        let invStd = try pow("\(prefix)_inv_std", base: varEps, exp: nhalf)
        let normalized = try mul("\(prefix)_normalized", x: centered, y: invStd)
        let gamma = try constWeight128(
            "\(prefix)_gamma",
            shape: try ANEShape(channels: dim, spatial: 1),
            blobPath: gammaPath
        )
        let scaled = try mul("\(prefix)_scaled", x: normalized, y: gamma)
        let beta = try constWeight128(
            "\(prefix)_beta",
            shape: try ANEShape(channels: dim, spatial: 1),
            blobPath: betaPath
        )
        return try add("\(prefix)_out", x: scaled, y: beta)
    }

    mutating func rmsNorm128(
        _ prefix: String,
        input: Int,
        dim: Int,
        spatial: Int,
        eps: Float,
        weightPath: String
    ) throws -> Int {
        let sq = try mul("\(prefix)_sq", x: input, y: input)
        let ss = try reduceSum("\(prefix)_ss", input: sq, axis: 1, keepDims: true)
        let invd = try constScalar("\(prefix)_invd", 1.0 / Float(dim))
        let ms = try mul("\(prefix)_ms", x: ss, y: invd)
        let epsNode = try constScalar("\(prefix)_eps", eps)
        let varEps = try add("\(prefix)_var_eps", x: ms, y: epsNode)
        let nhalf = try constScalar("\(prefix)_nhalf", -0.5)
        let invStd = try pow("\(prefix)_inv_std", base: varEps, exp: nhalf)
        let normalized = try mul("\(prefix)_normalized", x: input, y: invStd)
        let weight = try constWeight128(
            "\(prefix)_weight",
            shape: try ANEShape(channels: dim, spatial: 1),
            blobPath: weightPath
        )
        return try mul("\(prefix)_out", x: normalized, y: weight)
    }

    mutating func gelu128(
        _ prefix: String,
        input: Int
    ) throws -> Int {
        let x2 = try mul("\(prefix)_x2", x: input, y: input)
        let x3 = try mul("\(prefix)_x3", x: x2, y: input)
        let cubic = try constScalar("\(prefix)_cubic", 0.044715)
        let cx3 = try mul("\(prefix)_cx3", x: x3, y: cubic)
        let inner = try add("\(prefix)_inner", x: input, y: cx3)
        let scale = try constScalar("\(prefix)_scale", 0.797_884_6)
        let scaled = try mul("\(prefix)_scaled", x: inner, y: scale)
        let tanhNode = try tanh("\(prefix)_tanh", input: scaled)
        let one = try constScalar("\(prefix)_one", 1.0)
        let onePlus = try add("\(prefix)_one_plus", x: tanhNode, y: one)
        let half = try constScalar("\(prefix)_half", 0.5)
        let halfX = try mul("\(prefix)_half_x", x: input, y: half)
        return try mul("\(prefix)_out", x: halfX, y: onePlus)
    }

    mutating func ffn128(
        _ prefix: String,
        input: Int,
        inDim: Int,
        hiddenDim: Int,
        spatial: Int,
        w1Path: String,
        b1Path: String?,
        w2Path: String,
        b2Path: String?,
        activation: Activation
    ) throws -> Int {
        let up = try linear128(
            "\(prefix)_up",
            input: input,
            inDim: inDim,
            outDim: hiddenDim,
            spatial: spatial,
            weightPath: w1Path,
            biasPath: b1Path
        )
        let activated: Int
        switch activation {
        case .gelu:
            activated = try gelu128("\(prefix)_act", input: up)
        case .silu:
            let sigmoidNode = try sigmoid("\(prefix)_act_sigmoid", input: up)
            activated = try mul("\(prefix)_act_out", x: up, y: sigmoidNode)
        case .relu:
            activated = try relu("\(prefix)_act_out", input: up)
        }
        return try linear128(
            "\(prefix)_down",
            input: activated,
            inDim: hiddenDim,
            outDim: inDim,
            spatial: spatial,
            weightPath: w2Path,
            biasPath: b2Path
        )
    }

    mutating func causalAttention128(
        _ prefix: String,
        q: Int,
        k: Int,
        v: Int,
        nHeads: Int,
        headDim: Int,
        spatial: Int,
        maskPath: String
    ) throws -> Int {
        let modelDim = nHeads * headDim
        let headShape = try ANEShape(batch: 1, channels: nHeads, height: headDim, spatial: spatial)
        let transposedShape = try ANEShape(batch: 1, channels: nHeads, height: spatial, spatial: headDim)
        let scoresShape = try ANEShape(batch: 1, channels: nHeads, height: spatial, spatial: spatial)

        let q4 = try reshape("\(prefix)_q_reshape", input: q, shape: headShape)
        let k4 = try reshape("\(prefix)_k_reshape", input: k, shape: headShape)
        let v4 = try reshape("\(prefix)_v_reshape", input: v, shape: headShape)
        let qT = try transpose("\(prefix)_q_transpose", input: q4, perm: [0, 1, 3, 2])
        let kT = try transpose("\(prefix)_k_transpose", input: k4, perm: [0, 1, 3, 2])
        let vT = try transpose("\(prefix)_v_transpose", input: v4, perm: [0, 1, 3, 2])

        let scores = try matmul(
            "\(prefix)_scores",
            x: qT,
            y: kT,
            transposeX: false,
            transposeY: true,
            outShape: scoresShape
        )
        let scale = try constScalar("\(prefix)_scale", 1.0 / Float(headDim).squareRoot())
        let scaled = try mul("\(prefix)_scaled", x: scores, y: scale)
        let mask = try constWeight128(
            "\(prefix)_mask",
            shape: try ANEShape(batch: 1, channels: 1, height: spatial, spatial: spatial),
            blobPath: maskPath
        )
        let masked = try add("\(prefix)_masked", x: scaled, y: mask)
        let attn = try softmax("\(prefix)_softmax", input: masked, axis: -1)
        let context = try matmul(
            "\(prefix)_context",
            x: attn,
            y: vT,
            transposeX: false,
            transposeY: false,
            outShape: transposedShape
        )
        let contextT = try transpose("\(prefix)_context_transpose", input: context, perm: [0, 1, 3, 2])
        return try reshape(
            "\(prefix)_out",
            input: contextT,
            shape: try ANEShape(channels: modelDim, spatial: spatial)
        )
    }
}
