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
    /// Wall from prompt submit through the first emitted token, **including prefill**.
    public let firstTokenLatencyMs: Double
    /// Wall of the prefill phase only (submit → last prefill step). Compile is excluded.
    public let prefillMs: Double
    public let exactHeadBackend: String
    public let cachedBindingsEnabled: Bool
    public let committedExactTokensPerPass: Double?
    public let acceptedFutureTokensPerPass: Double?
    /// Resolved llama serving trunk. `nil` when the run is not on a typed llama trunk
    /// (for example GPT-2), which telemetry still reports as `"unknown"`.
    public let trunk: Trunk?
    /// ANE hops per generated token. Fused N=1 is `nLayer`; split hybrid is `2 * nLayer`.
    public let hopsPerToken: Int?
    /// Hybrid per-token bucket report from `HybridDecodeTokenProfile.formatReport()`.
    public let decodeProfileReport: String?

    /// Stable telemetry label: `Trunk.telemetryLabel`, or `"unknown"` when `trunk` is nil.
    public var decodePath: String {
        trunk?.telemetryLabel ?? "unknown"
    }

    /// Published TTFT: submit through first token, including prefill.
    public var ttftIncludingPrefillMs: Double { firstTokenLatencyMs }

    public init(
        text: String,
        tokens: [TokenID],
        promptTokens: [TokenID],
        tokenLatenciesMs: [Double] = [],
        tokensPerSecond: Double,
        compileTimeMs: Double,
        firstTokenLatencyMs: Double,
        prefillMs: Double = 0,
        exactHeadBackend: String = "unknown",
        cachedBindingsEnabled: Bool = false,
        committedExactTokensPerPass: Double? = nil,
        acceptedFutureTokensPerPass: Double? = nil,
        trunk: Trunk? = nil,
        hopsPerToken: Int? = nil,
        decodeProfileReport: String? = nil
    ) {
        self.text = text
        self.tokens = tokens
        self.promptTokens = promptTokens
        self.tokenLatenciesMs = tokenLatenciesMs
        self.tokensPerSecond = tokensPerSecond
        self.compileTimeMs = compileTimeMs
        self.firstTokenLatencyMs = firstTokenLatencyMs
        self.prefillMs = prefillMs
        self.exactHeadBackend = exactHeadBackend
        self.cachedBindingsEnabled = cachedBindingsEnabled
        self.committedExactTokensPerPass = committedExactTokensPerPass
        self.acceptedFutureTokensPerPass = acceptedFutureTokensPerPass
        self.trunk = trunk
        self.hopsPerToken = hopsPerToken
        self.decodeProfileReport = decodeProfileReport
    }

    /// Compatibility initializer that accepts a telemetry path label.
    ///
    /// - Throws: ``Trunk/ParseError`` when `decodePath` is not `"unknown"` and not a known
    ///   trunk label. Prefer the `trunk:` initializer for new code.
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
        acceptedFutureTokensPerPass: Double? = nil,
        decodePath: String,
        hopsPerToken: Int? = nil,
        decodeProfileReport: String? = nil
    ) throws {
        let resolvedTrunk: Trunk?
        if decodePath == "unknown" {
            resolvedTrunk = nil
        } else {
            resolvedTrunk = try Trunk.parseTelemetryLabel(decodePath)
        }
        self.init(
            text: text,
            tokens: tokens,
            promptTokens: promptTokens,
            tokenLatenciesMs: tokenLatenciesMs,
            tokensPerSecond: tokensPerSecond,
            compileTimeMs: compileTimeMs,
            firstTokenLatencyMs: firstTokenLatencyMs,
            exactHeadBackend: exactHeadBackend,
            cachedBindingsEnabled: cachedBindingsEnabled,
            committedExactTokensPerPass: committedExactTokensPerPass,
            acceptedFutureTokensPerPass: acceptedFutureTokensPerPass,
            trunk: resolvedTrunk,
            hopsPerToken: hopsPerToken,
            decodeProfileReport: decodeProfileReport
        )
    }

    public func withTrunk(_ trunk: Trunk?) -> GenerationResult {
        GenerationResult(
            text: text,
            tokens: tokens,
            promptTokens: promptTokens,
            tokenLatenciesMs: tokenLatenciesMs,
            tokensPerSecond: tokensPerSecond,
            compileTimeMs: compileTimeMs,
            firstTokenLatencyMs: firstTokenLatencyMs,
            prefillMs: prefillMs,
            exactHeadBackend: exactHeadBackend,
            cachedBindingsEnabled: cachedBindingsEnabled,
            committedExactTokensPerPass: committedExactTokensPerPass,
            acceptedFutureTokensPerPass: acceptedFutureTokensPerPass,
            trunk: trunk,
            hopsPerToken: hopsPerToken,
            decodeProfileReport: decodeProfileReport
        )
    }

    /// Compatibility helper that parses a telemetry path label into ``trunk``.
    ///
    /// - Throws: ``Trunk/ParseError`` when `path` is not `"unknown"` and not a known trunk label.
    public func withDecodePath(_ path: String) throws -> GenerationResult {
        if path == "unknown" {
            return withTrunk(nil)
        }
        return withTrunk(try Trunk.parseTelemetryLabel(path))
    }
}

public struct GenerationStep: Sendable {
    public let token: TokenID
    public let generatedTokens: [TokenID]
    public let text: String
    public let tokenLatencyMs: Double
    public let elapsedMs: Double
    public let firstTokenLatencyMs: Double
    public let prefillMs: Double
    public let tokensPerSecond: Double

    public init(
        token: TokenID,
        generatedTokens: [TokenID],
        text: String,
        tokenLatencyMs: Double,
        elapsedMs: Double,
        firstTokenLatencyMs: Double,
        prefillMs: Double = 0,
        tokensPerSecond: Double
    ) {
        self.token = token
        self.generatedTokens = generatedTokens
        self.text = text
        self.tokenLatencyMs = tokenLatencyMs
        self.elapsedMs = elapsedMs
        self.firstTokenLatencyMs = firstTokenLatencyMs
        self.prefillMs = prefillMs
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
    /// Raised when `ESPRESSO_REALMODEL_DISABLE_HYBRID_FALLBACK=1` and work would otherwise
    /// leave the ANE hybrid decode path. `stage` names where the fallback was about to
    /// happen and `reason` names the op, kernel, or policy responsible.
    case hybridFallbackDisabled(stage: String, reason: String)
    case cancelled

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
        case let .hybridFallbackDisabled(stage, reason):
            return """
                ANE hybrid fallback is disabled \
                (ESPRESSO_REALMODEL_DISABLE_HYBRID_FALLBACK=1) but \(stage) would fall back \
                off the ANE: \(reason)
                """
        case .cancelled:
            return "Generation cancelled"
        }
    }
}

extension GenerationResult {
    /// Copy of the result with the decode profile report dropped, used by the
    /// token-suite helper so a profile string cannot outlive its ANE surfaces.
    func strippingDecodeProfileReport() -> GenerationResult {
        GenerationResult(
            text: text,
            tokens: tokens,
            promptTokens: promptTokens,
            tokenLatenciesMs: tokenLatenciesMs,
            tokensPerSecond: tokensPerSecond,
            compileTimeMs: compileTimeMs,
            firstTokenLatencyMs: firstTokenLatencyMs,
            exactHeadBackend: exactHeadBackend,
            cachedBindingsEnabled: cachedBindingsEnabled,
            committedExactTokensPerPass: committedExactTokensPerPass,
            acceptedFutureTokensPerPass: acceptedFutureTokensPerPass,
            trunk: trunk,
            hopsPerToken: hopsPerToken,
            decodeProfileReport: nil
        )
    }
}

public struct RealModelInferenceEngine: ~Copyable {
    private static let minimumANEIOSurfaceBytes = 49_152
    static let classifierArgmaxBlockSize = 4_000

    /// Single process-environment seam for this file: every read flows through
    /// here so global-state access stays auditable in one place.
    static var processEnvironment: [String: String] {
        ProcessInfo.processInfo.environment
    }

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
            // Default-on for the retained Stories demo family and the Qwen
            // 1.5B hybrid serve. Other Llama models stay opt-in.
            if ModelFamily.isStories110MVariant(config) {
                return true
            }
            if ModelFamily.isQwen15BVariant(config) {
                return true
            }
            return environment["ESPRESSO_ENABLE_LLAMA_HYBRID_CACHED_BINDINGS"] == "1"
        }
        return true
    }

    /// Qwen 1.5B must not silently fall back to fresh Metal bindings.
    /// Stories and other Llama opt-ins keep the historical serial fallback.
    static func requiresHybridCachedBindings(
        config: MultiModelConfig,
        environment: [String: String]
    ) -> Bool {
        supportsHybridCachedBindings(config: config, environment: environment)
            && ModelFamily.isQwen15BVariant(config)
    }

    /// Phase 11 compiled `max_N = 1` only. Serve that N: one fused program per
    /// layer with attention in-graph (28 hops, no Metal QKV↔FFN sync).
    static let fusedDecodePathLabel = Trunk.fusedHybrid.telemetryLabel
    static let fusedHybridFallbackStage = FusedHybridDecodeLayerKernelSet.fallbackStage

    static func prefersFusedHybridDecode(
        config: MultiModelConfig,
        environment: [String: String]
    ) -> Bool {
        if environment["ESPRESSO_DISABLE_FUSED_HYBRID_DECODE"] == "1" {
            return false
        }
        if environment["ESPRESSO_ENABLE_FUSED_HYBRID_DECODE"] == "1" {
            return config.architecture == .llama
        }
        return config.architecture == .llama && ModelFamily.isQwen15BVariant(config)
    }

    static func fusedHopsPerToken(nLayer: Int) -> Int {
        FusedHybridDecodeLayerKernelSet.hopsPerToken(nLayer: nLayer)
    }

    static func fusedHybridFallbackError(reason: String) -> RealModelInferenceError {
        .hybridFallbackDisabled(stage: fusedHybridFallbackStage, reason: reason)
    }

    static func makeHybridCachedBindingsOrFallback<Bindings>(
        config: MultiModelConfig,
        environment: [String: String],
        create: () throws -> Bindings
    ) throws -> Bindings? {
        guard supportsHybridCachedBindings(config: config, environment: environment) else {
            return nil
        }
        do {
            return try create()
        } catch {
            if requiresHybridCachedBindings(config: config, environment: environment) {
                throw RealModelInferenceError.hybridFallbackDisabled(
                    stage: "hybrid_cached_bindings",
                    reason: "cached Metal bindings failed: \(error)"
                )
            }
            return nil
        }
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
        // Donor delta copies a prior layer's net.plist and swaps weights.
        // At Qwen2.5-1.5B FFN width (8960×1536 = 13.7M elements) that path
        // misses more often than it hits; keep it for 0.5B-scale FFNs.
        if config.architecture == .llama,
           config.hiddenDim * config.dModel > Self.hybridDonorDeltaFFNElementLimit {
            return false
        }
        return true
    }

    /// FFN weight elements (`hiddenDim * dModel`) above this skip donor delta.
    /// 0.5B is 4864×896 = 4.36M and TinyLlama is 5632×2048 = 11.5M (keep);
    /// 1.5B is 8960×1536 = 13.8M (skip — donor misses more than it hits).
    static let hybridDonorDeltaFFNElementLimit = 12_000_000

    /// Stories 110M family recognition — delegates to `ModelFamily`.
    static func isStories110MVariant(_ config: MultiModelConfig) -> Bool {
        ModelFamily.isStories110MVariant(config)
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
        return ModelFamily.isQwenVariant(config)
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
        env: [String: String] = Self.processEnvironment
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
        DecodePathPolicy.prefersCPUExactDecode(
            config: config,
            options: DecodePathPolicy.optionsFromEnvironment(environment)
        )
    }

    /// Resolves the llama serving ``Trunk``, refusing to leave the ANE silently.
    ///
    /// With `ESPRESSO_REALMODEL_DISABLE_HYBRID_FALLBACK=1`, landing on the pure-CPU trunk is
    /// a failure rather than a quiet downgrade, and the thrown error names which policy
    /// chose CPU so the cause is actionable instead of mysterious.
    static func resolvedTrunk(
        config: MultiModelConfig,
        environment: [String: String]
    ) throws -> Trunk {
        try DecodePathPolicy.resolvedTrunk(
            config: config,
            fusedHybridPreferred: prefersFusedHybridDecode(config: config, environment: environment),
            options: DecodePathPolicy.optionsFromEnvironment(environment)
        )
    }

    static func milDeploymentTarget(
        environment: [String: String] = Self.processEnvironment
    ) -> String {
        let rawValue = environment["ESPRESSO_MIL_DEPLOYMENT_TARGET"]?
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return rawValue?.isEmpty == false ? rawValue! : "ios18"
    }

    static func gpt2NormKind(
        environment: [String: String] = Self.processEnvironment
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
        environment: [String: String] = Self.processEnvironment
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

    /// Select the serving trunk for a llama-family config (no fallback-disabled gate).
    static func selectTrunk(
        config: MultiModelConfig,
        environment: [String: String]
    ) -> Trunk {
        DecodePathPolicy.resolve(
            config: config,
            fusedHybridPreferred: prefersFusedHybridDecode(config: config, environment: environment),
            options: DecodePathPolicy.optionsFromEnvironment(environment)
        ).trunk
    }

    struct TopLevelWeightPaths: Sendable, Equatable {
        let tokenEmbedding: String
        let positionEmbedding: String
        let finalNormGamma: String
        let finalNormBeta: String
        let lmHead: String
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



    struct ExactTwoTokenDraftDescriptor: Sendable, Decodable {
        let modelDir: String
        let tokenizerDir: String?
        let modelID: String?

        enum CodingKeys: String, CodingKey {
            case modelDir = "model_dir"
            case tokenizerDir = "tokenizer_dir"
            case modelID = "model_id"
        }
    }

    struct ResolvedExactTwoTokenDraft: Sendable {
        let descriptor: ExactTwoTokenDraftDescriptor
        let descriptorURL: URL
        let weightDirURL: URL
        let config: MultiModelConfig
    }


    enum LoadedTokenizer {
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

    struct SpeculativeRuntimeKey: Hashable {
        let draftLayerCount: Int
        let maxSeq: Int
    }

    final class CachedSpeculativeRuntimePair {
        let key: SpeculativeRuntimeKey
        var draftRuntime: HybridLayerRangeRuntime
        var verifierRuntime: HybridLayerRangeRuntime

        init(
            key: SpeculativeRuntimeKey,
            config: MultiModelConfig,
            weightDirURL: URL,
            assets: GPT2TopLevelAssets,
            environment: [String: String]
        ) throws {
            self.key = key
            self.draftRuntime = try HybridLayerRangeRuntime(
                config: config,
                weightDirURL: weightDirURL,
                assets: assets,
                layerRange: 0..<key.draftLayerCount,
                maxSeq: key.maxSeq,
                environment: environment
            )
            self.verifierRuntime = try HybridLayerRangeRuntime(
                config: config,
                weightDirURL: weightDirURL,
                assets: assets,
                layerRange: key.draftLayerCount..<config.nLayer,
                maxSeq: key.maxSeq,
                environment: environment
            )
        }

        func resetAll(dim: Int) throws(ANEError) {
            try draftRuntime.reset(dim: dim)
            try verifierRuntime.reset(dim: dim)
        }
    }

    struct HybridRuntimeCheckpoint: Sendable {
        let step: Int
    }

    struct HybridLayerRangeRuntime: ~Copyable {
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
            maxSeq: Int,
            environment: [String: String]
        ) throws {
            precondition(!layerRange.isEmpty)

            let layers = try RealModelInferenceEngine.compileHybridLayers(
                config: config,
                weightDirURL: weightDirURL,
                sourceLayerRange: layerRange,
                maxSeq: maxSeq,
                environment: environment
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
                   environment: environment
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
                    outputDType: .fp16,
                    environment: environment
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
            try ForwardPass.initializeHybridDecodeCaches(
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
                environment: environment
            )
            self.decodeState = decodeState
        }

        var finalSurface: IOSurfaceRef {
            surfaceHandles[surfaceHandles.count - 1].ffnOut
        }

        var step: Int {
            decodeState.visibleTokenCount
        }

        mutating func reset(dim: Int) throws(ANEError) {
            try ForwardPass.initializeHybridDecodeCaches(
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

    static let gpt2EOSToken: TokenID = 50_256
    static let speculativeRuntimeCacheLimit = 4

    let config: MultiModelConfig
    var weightDirURL: URL
    let tokenizer: LoadedTokenizer
    private let assets: TopLevelAssets

    var gpt2Assets: GPT2TopLevelAssets {
        guard case let .gpt2(a) = assets else {
            preconditionFailure("Attempted to access GPT-2 assets on a non-GPT-2 model")
        }
        return a
    }

    var llamaAssets: LlamaTopLevelAssets {
        guard case let .llama(a) = assets else {
            preconditionFailure("Attempted to access Llama assets on a non-Llama model")
        }
        return a
    }

    func hybridGreedyHeadMode(
        environment: [String: String] = Self.processEnvironment
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

    /// Per-trunk compile readiness; the ensure functions below are the only writers.
    var baselineReadiness = CompiledReadiness<BaselineCompiledRuntime>.notCompiled
    var splitHybridReadiness = CompiledReadiness<SplitHybridCompiledRuntime>.notCompiled
    var fusedHybridReadiness = CompiledReadiness<FusedHybridCompiledRuntime>.notCompiled
    /// Bucket of the last fully resident split-hybrid layer-program set.
    ///
    /// Incremental-compile watermark consulted only by ``ensureHybridCompiled``
    /// (so a mid-compile failure never recompiles finished layers) and by the
    /// dispatch max-sequence defaults; loop readiness itself is carried by
    /// ``splitHybridReadiness``.
    var splitHybridLayerBucket = 0
    var compiledLayers: LayerStorage<CompiledLayer>
    var compiledHead: LayerStorage<CompiledHead>
    var compiledHybridLayers: LayerStorage<HybridDecodeKernelSet>
    var compiledHybridSurfaceHandles: [HybridDecodeSurfaceHandles]
    var compiledHybridLlamaQKNormWeights: [LlamaQKNormWeights?]
    var compiledHybridHead: LayerStorage<CompiledHead>
    var compiledHybridHeadSpatial: Int
    var compiledHybridGreedyNorm: LayerStorage<CompiledHead>
    var compiledHybridGreedyClassifier: LayerStorage<CompiledClassifier>
    var compiledHybridGreedySpatial: Int
    var compiledFusedHybridLayers: LayerStorage<FusedHybridDecodeLayerKernelSet>
    var compiledFusedHybridSurfaceHandles: [FusedHybridDecodeSurfaceHandles]
    var hybridMetalAttention: MetalAttentionKernel?
    var speculativeRuntimeCache: [SpeculativeRuntimeKey: CachedSpeculativeRuntimePair]
    var speculativeRuntimeCacheOrder: [SpeculativeRuntimeKey]
    private let classifierBlockMaxNorms: [Float]
    private var classifierLogitsScratch: [Float]
    let classifierStrategy: ClassifierStrategy
    let policies: EnginePolicies
    private var cachedExactCPULlamaWeights: CachedExactCPULlamaWeights?

    private init(
        config: MultiModelConfig,
        weightDirURL: URL,
        tokenizer: LoadedTokenizer,
        assets: TopLevelAssets,
        policies: EnginePolicies = .resolve()
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
        self.compiledLayers = Self.emptyStorage(CompiledLayer.self)
        self.compiledHead = Self.emptyStorage(CompiledHead.self)
        self.compiledHybridLayers = Self.emptyStorage(HybridDecodeKernelSet.self)
        self.compiledHybridSurfaceHandles = []
        self.compiledHybridLlamaQKNormWeights = []
        self.compiledHybridHead = Self.emptyStorage(CompiledHead.self)
        self.compiledHybridHeadSpatial = 0
        self.compiledHybridGreedyNorm = Self.emptyStorage(CompiledHead.self)
        self.compiledHybridGreedyClassifier = Self.emptyStorage(CompiledClassifier.self)
        self.compiledHybridGreedySpatial = 0
        self.compiledFusedHybridLayers = Self.emptyStorage(FusedHybridDecodeLayerKernelSet.self)
        self.compiledFusedHybridSurfaceHandles = []
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
            environment: policies.environment
        )
        self.policies = policies
        self.cachedExactCPULlamaWeights = nil
    }

    /// Builds an engine. `environment` is the single configuration seam:
    /// everything the process environment would have steered is resolved from
    /// this dictionary once, here. Callers embedding Espresso pass values;
    /// only the default reads the live process environment.
    public static func build(
        config: MultiModelConfig,
        weightDir: String,
        tokenizerDir: String,
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) throws -> RealModelInferenceEngine {
        try validateConfig(config)

        let weightDirURL = URL(fileURLWithPath: weightDir, isDirectory: true)
        let tokenizerDirURL = URL(fileURLWithPath: tokenizerDir, isDirectory: true)
        try validateDirectory(weightDirURL)
        try validateDirectory(tokenizerDirURL)
        try validateMetadataIfPresent(config: config, weightDirURL: weightDirURL)

        let tokenizer = try loadTokenizer(config: config, tokenizerDirURL: tokenizerDirURL)
        let policies = EnginePolicies(environment: environment)

        let topLevelAssets = try TopLevelAssetLoader.load(
            config: config,
            weightDir: weightDir,
            weightDirURL: weightDirURL,
            environment: environment
        )
        return RealModelInferenceEngine(
            config: config,
            weightDirURL: weightDirURL,
            tokenizer: tokenizer,
            assets: topLevelAssets,
            policies: policies
        )
    }

    /// The decode-path bootstrap seam: resolves the serving plan from explicit options
    /// when provided, otherwise from the bounded decode-path environment variables.
    private static func resolveDecodePlanSeam(
        config: MultiModelConfig,
        options: DecodePathOptions?,
        environment: [String: String] = Self.processEnvironment
    ) -> (plan: ResolvedDecodePlan, options: DecodePathOptions, fusedHybridPreferred: Bool) {
        let decodeOptions = options ?? DecodePathPolicy.optionsFromEnvironment(environment)
        let fusedHybridPreferred = Self.prefersFusedHybridDecode(config: config, environment: environment)
        let plan = DecodePathPolicy.resolve(
            config: config,
            fusedHybridPreferred: fusedHybridPreferred,
            options: decodeOptions
        )
        return (plan, decodeOptions, fusedHybridPreferred)
    }

    /// Compile-and-prepare state for one llama trunk's decode programs.
    private struct LlamaTrunkPreparation {
        let compileTimeMs: Double
        let metalAttention: MetalAttentionKernel?
    }

    /// One unit of llama serving work routed through the single ``Trunk`` dispatch point.
    private struct LlamaDecodeRequest {
        let trunk: Trunk
        let bucket: Int
        /// Names the caller in attention-unavailability errors ("llama" vs "llama testing helper").
        let sessionKind: String
        let promptTokens: [TokenID]
        let effectiveMaxTokens: Int
        let temperature: Float
        var topP: Float = 1.0
        var compileTimeMs: Double = 0
        /// Precomputed `prepareLlamaTrunkDecode` output; when nil the dispatcher prepares.
        var preparation: LlamaTrunkPreparation?
        /// Explicit decode-session context length; per-trunk defaults apply when nil.
        var maxSeq: Int? = nil
        var onStep: ((GenerationStep) -> Void)? = nil
        var isCancelled: (() -> Bool)? = nil
        var dropsDecodeProfileReport = false
    }

    /// Compiles decode programs for a selected llama trunk without decoding anything.
    ///
    /// Shared by chat precompile and the token-suite warmup; the fused error-wrapping and
    /// Metal-attention guards live here so every entry point behaves identically.
    private mutating func prepareLlamaTrunkDecode(
        _ trunk: Trunk,
        bucket: Int,
        sessionKind: String
    ) throws -> LlamaTrunkPreparation {
        if trunk == .fusedHybrid {
            let compileStart = DispatchTime.now().uptimeNanoseconds
            let compileDidRun: Bool
            do {
                compileDidRun = try ensureFusedHybridCompiled(bucket: bucket)
            } catch let error as RealModelInferenceError {
                if case .hybridFallbackDisabled = error { throw error }
                throw Self.fusedHybridFallbackError(
                    reason: error.errorDescription ?? "\(error)"
                )
            } catch {
                throw Self.fusedHybridFallbackError(reason: "\(error)")
            }
            let compileEnd = DispatchTime.now().uptimeNanoseconds
            return LlamaTrunkPreparation(
                compileTimeMs: compileDidRun ? Self.milliseconds(from: compileEnd - compileStart) : 0,
                metalAttention: nil
            )
        }
        if trunk == .splitHybrid {
            let compileStart = DispatchTime.now().uptimeNanoseconds
            let compileDidRun = try ensureHybridCompiled(bucket: bucket)
            guard let metalAttention = hybridMetalAttention else {
                throw RealModelInferenceError.runtimeFailure("Hybrid Metal attention unavailable for \(sessionKind)")
            }
            let compileEnd = DispatchTime.now().uptimeNanoseconds
            return LlamaTrunkPreparation(
                compileTimeMs: compileDidRun ? Self.milliseconds(from: compileEnd - compileStart) : 0,
                metalAttention: metalAttention
            )
        }
        return LlamaTrunkPreparation(compileTimeMs: 0, metalAttention: nil)
    }

    /// THE single ``Trunk`` dispatch point for llama serving sessions (REQ-004): every
    /// decode path funnels its per-trunk preparation and session work through this switch.
    private mutating func dispatchLlamaTrunkDecode(
        _ request: LlamaDecodeRequest
    ) throws -> GenerationResult {
        switch request.trunk {
        case .exactCPU:
            return try generateIncrementalExactCPULlama(
                promptTokens: request.promptTokens,
                effectiveMaxTokens: request.effectiveMaxTokens,
                temperature: request.temperature,
                topP: request.topP,
                compileTimeMs: request.compileTimeMs,
                maxSeq: request.maxSeq ?? request.bucket,
                onStep: request.onStep,
                isCancelled: request.isCancelled
            )
        case .fusedHybrid:
            let prepared: LlamaTrunkPreparation
            let sessionCompileTimeMs: Double
            if let provided = request.preparation {
                prepared = provided
                sessionCompileTimeMs = request.compileTimeMs
            } else {
                prepared = try prepareLlamaTrunkDecode(
                    .fusedHybrid,
                    bucket: request.bucket,
                    sessionKind: request.sessionKind
                )
                sessionCompileTimeMs = prepared.compileTimeMs
            }
            let result = try generateIncrementalFusedHybridLlama(
                promptTokens: request.promptTokens,
                effectiveMaxTokens: request.effectiveMaxTokens,
                temperature: request.temperature,
                topP: request.topP,
                compileTimeMs: sessionCompileTimeMs,
                maxSeq: request.maxSeq ?? max(request.bucket, fusedHybridReadiness.runtime?.bucket ?? request.bucket),
                onStep: request.onStep,
                isCancelled: request.isCancelled
            )
            if request.dropsDecodeProfileReport {
                return result.strippingDecodeProfileReport()
            }
            return result
        case .splitHybrid:
            let prepared: LlamaTrunkPreparation
            let sessionCompileTimeMs: Double
            if let provided = request.preparation {
                prepared = provided
                sessionCompileTimeMs = request.compileTimeMs
            } else {
                prepared = try prepareLlamaTrunkDecode(
                    .splitHybrid,
                    bucket: request.bucket,
                    sessionKind: request.sessionKind
                )
                sessionCompileTimeMs = prepared.compileTimeMs
            }
            guard let metalAttention = prepared.metalAttention else {
                throw RealModelInferenceError.runtimeFailure("Hybrid Metal attention unavailable for \(request.sessionKind)")
            }
            return try generateIncrementalHybridLlama(
                promptTokens: request.promptTokens,
                effectiveMaxTokens: request.effectiveMaxTokens,
                temperature: request.temperature,
                topP: request.topP,
                compileTimeMs: sessionCompileTimeMs,
                maxSeq: request.maxSeq ?? max(request.bucket, splitHybridLayerBucket),
                metalAttention: metalAttention,
                onStep: request.onStep,
                isCancelled: request.isCancelled
            )
        }
    }

    /// Compile hybrid decode kernels once for a context that covers later turns.
    /// Chat re-prefills growing history; compiling per-turn at a larger bucket
    /// exhausts the per-process ANE compile budget.
    public mutating func precompileHybridDecode(
        covering tokenCount: Int? = nil,
        options decodeOptions: DecodePathOptions? = nil
    ) throws {
        let target = min(max(tokenCount ?? config.maxSeq, 1), config.maxSeq)
        let bucket = try Self.compileBucket(
            for: target,
            channels: config.dModel,
            maxSeq: config.maxSeq
        )
        switch config.architecture {
        case .llama:
            let seam = Self.resolveDecodePlanSeam(
                config: config,
                options: decodeOptions,
                environment: policies.environment
            )
            _ = try prepareLlamaTrunkDecode(seam.plan.trunk, bucket: bucket, sessionKind: "llama")
        case .gpt2:
            _ = try ensureHybridCompiled(bucket: bucket)
        }
    }

    public mutating func generate(
        prompt: String,
        maxTokens: Int = 128,
        temperature: Float = 0.0,
        topP: Float = 1.0,
        onStep: ((GenerationStep) -> Void)? = nil,
        isCancelled: (() -> Bool)? = nil,
        options decodeOptions: DecodePathOptions? = nil
    ) throws -> GenerationResult {
        guard maxTokens >= 0 else {
            throw RealModelInferenceError.invalidGenerationParameters("maxTokens must be >= 0")
        }
        guard temperature.isFinite, temperature >= 0 else {
            throw RealModelInferenceError.invalidGenerationParameters("temperature must be finite and >= 0")
        }
        guard topP.isFinite, topP > 0, topP <= 1 else {
            throw RealModelInferenceError.invalidGenerationParameters("topP must be in (0, 1]")
        }
        try Self.throwIfCancelled(isCancelled)

        let promptTokens = try encodePrompt(prompt)
        guard promptTokens.count < config.maxSeq else {
            throw RealModelInferenceError.invalidPrompt(
                "Prompt token count \(promptTokens.count) exceeds model context \(config.maxSeq - 1)"
            )
        }

        let remainingContext = config.maxSeq - promptTokens.count
        let effectiveMaxTokens = min(maxTokens, max(remainingContext, 0))
        let environment = policies.environment
        let seam = Self.resolveDecodePlanSeam(config: config, options: decodeOptions, environment: environment)
        let plan = seam.plan
        if effectiveMaxTokens == 0 {
            let tokenizer = self.tokenizer
            let eosPolicy: EOSPolicy = config.architecture == .llama
                ? .fromConfig(config.eosToken.map(Int.init))
                : .fixed(Int(Self.gpt2EOSToken))
            let emission = EmissionCore(
                promptTokens: promptTokens,
                capacity: 0,
                eos: eosPolicy,
                onStep: nil,
                decodeText: { tokenizer.decode($0) }
            )
            var trunk: Trunk?
            var hopsPerToken: Int?
            if config.architecture == .llama {
                trunk = plan.trunk
                hopsPerToken = plan.trunk == .fusedHybrid
                    ? Self.fusedHopsPerToken(nLayer: config.nLayer)
                    : nil
            }
            return emission.makeResult(compileTimeMs: 0, trunk: trunk, hopsPerToken: hopsPerToken)
        }

        let targetTokenCount = min(config.maxSeq, promptTokens.count + effectiveMaxTokens)
        let bucket = try Self.compileBucket(
            for: targetTokenCount,
            channels: config.dModel,
            maxSeq: config.maxSeq
        )

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
            let trunk = try DecodePathPolicy.resolvedTrunk(
                config: config,
                fusedHybridPreferred: seam.fusedHybridPreferred,
                options: seam.options
            )
            return try dispatchLlamaTrunkDecode(LlamaDecodeRequest(
                trunk: trunk,
                bucket: bucket,
                sessionKind: "llama",
                promptTokens: promptTokens,
                effectiveMaxTokens: effectiveMaxTokens,
                temperature: temperature,
                topP: topP,
                onStep: onStep,
                isCancelled: isCancelled
            ))
        }

        let compileStart = DispatchTime.now().uptimeNanoseconds
        var compileDidRun = false
        var useHybridFastPath = false

        do {
            let hybridDidRun = try ensureHybridCompiled(bucket: bucket)
            compileDidRun = compileDidRun || hybridDidRun
            switch splitHybridReadiness {
            case .compiled:
                useHybridFastPath = true
            case .notCompiled:
                useHybridFastPath = false
                guard plan.allowsHybridFallback else {
                    throw RealModelInferenceError.hybridFallbackDisabled(
                        stage: "hybrid decode compile",
                        reason: """
                            hybrid decode state is incomplete: layers=\(compiledHybridLayers.count)/\(config.nLayer) \
                            surfaces=\(compiledHybridSurfaceHandles.count)/\(config.nLayer) \
                            head=\(compiledHybridHead.count)/1 \
                            metalAttention=\(hybridMetalAttention != nil)
                            """
                    )
                }
            }
        } catch let error as RealModelInferenceError {
            // A disabled-fallback error is the answer, not something to recover from.
            if case .hybridFallbackDisabled = error {
                throw error
            }
            if !plan.allowsHybridFallback {
                throw RealModelInferenceError.hybridFallbackDisabled(
                    stage: "hybrid decode compile",
                    reason: "\(error.errorDescription ?? "\(error)")"
                )
            }
            useHybridFastPath = false
        } catch {
            if !plan.allowsHybridFallback {
                throw RealModelInferenceError.hybridFallbackDisabled(
                    stage: "hybrid decode compile",
                    reason: "\(error)"
                )
            }
            useHybridFastPath = false
        }

        if useHybridFastPath, let metalAttention = hybridMetalAttention {
            let compileEnd = DispatchTime.now().uptimeNanoseconds
            let compileTimeMs = compileDidRun ? Self.milliseconds(from: compileEnd - compileStart) : 0
            if let speculativeDraftLayerCount = Self.resolvedSpeculativeDraftLayerCount(
                config: config,
                temperature: temperature,
                environment: environment
            ) {
                var speculativeAttemptCompileTimeMs = 0.0
                do {
                    let (cachedRuntimePair, speculativeCompileTimeMs) = try cachedSpeculativeRuntimePair(
                        draftLayerCount: speculativeDraftLayerCount,
                        maxSeq: bucket,
                        environment: environment
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
                    if !plan.allowsHybridFallback {
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
                        topP: topP,
                        compileTimeMs: fallbackCompileTimeMs,
                        maxSeq: bucket,
                        metalAttention: metalAttention,
                        onStep: onStep,
                        isCancelled: isCancelled
                    )
                }
            }
            do {
                return try generateIncrementalHybrid(
                    promptTokens: promptTokens,
                    effectiveMaxTokens: effectiveMaxTokens,
                    temperature: temperature,
                    topP: topP,
                    compileTimeMs: compileTimeMs,
                    maxSeq: bucket,
                    metalAttention: metalAttention,
                    onStep: onStep,
                    isCancelled: isCancelled
                )
            } catch {
                if !plan.allowsHybridFallback {
                    throw RealModelInferenceError.runtimeFailure("Hybrid fast path failed: \(error)")
                }
                useHybridFastPath = false
            }
        }

        let baselineDidRun = try ensureCompiled(bucket: bucket)
        compileDidRun = compileDidRun || baselineDidRun
        let compileEnd = DispatchTime.now().uptimeNanoseconds
        let compileTimeMs = compileDidRun ? Self.milliseconds(from: compileEnd - compileStart) : 0

        let baseline: BaselineCompiledRuntime
        switch baselineReadiness {
        case .compiled(let runtime):
            baseline = runtime
        case .notCompiled:
            throw RealModelInferenceError.runtimeFailure("Compiled ANE surfaces are unavailable")
        }
        let inputSurface = baseline.inputSurface

        let generationStart = DispatchTime.now().uptimeNanoseconds
        let tokenizer = self.tokenizer
        var emission = EmissionCore(
            promptTokens: promptTokens,
            capacity: effectiveMaxTokens,
            eos: .fixed(Int(Self.gpt2EOSToken)),
            onStep: onStep,
            decodeText: { tokenizer.decode($0) },
            startNanos: generationStart
        )
        var rng = SystemRandomNumberGenerator()
        let activeBucket = baseline.bucket

        for _ in 0..<effectiveMaxTokens {
            let sequenceLength = emission.allTokensCount
            let activation = composeEmbeddingInput(tokens: emission.allTokens, spatial: activeBucket)
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
            try Self.throwIfCancelled(isCancelled)
            let nextToken = selectTokenFromNormalizedHidden(
                lastHidden,
                temperature: temperature,
                topP: topP,
                using: &rng
            )

            let emissionNow = DispatchTime.now().uptimeNanoseconds
            emission.recordFirstTokenIfFirst(at: emissionNow)

            if emission.terminatesDecoding(nextToken) {
                break
            }

            emission.emit(nextToken, at: emissionNow)
            if emission.allTokensCount >= config.maxSeq {
                break
            }
        }

        return emission.makeResult(
            compileTimeMs: compileTimeMs,
            exactHeadBackend: classifierStrategy.exactHeadBackendLabel,
            cachedBindingsEnabled: false
        )
    }

    public static func generateNextTokenForTesting(
        config: MultiModelConfig,
        weightDir: String,
        promptTokens: [TokenID]
    ) throws -> TokenID {
        let results = try generateFromTokenSuiteForTesting(
            config: config,
            weightDir: weightDir,
            promptTokenSuite: [promptTokens],
            maxTokens: 1
        )
        guard let token = results.first?.tokens.first else {
            throw RealModelInferenceError.runtimeFailure("Testing helper did not emit a next token")
        }
        return token
    }

    /// Greedy-decodes every prompt in `promptTokenSuite` through a single engine instance.
    ///
    /// One engine means one ANE compile for the whole suite. Compiling per prompt would
    /// exhaust the finite per-process ANE compile budget long before a multi-prompt parity
    /// suite finished.
    public static func generateFromTokenSuiteForTesting(
        config: MultiModelConfig,
        weightDir: String,
        promptTokenSuite: [[TokenID]],
        maxTokens: Int,
        options decodeOptions: DecodePathOptions? = nil
    ) throws -> [GenerationResult] {
        guard !promptTokenSuite.isEmpty else {
            throw RealModelInferenceError.invalidGenerationParameters("Testing prompt suite must not be empty")
        }
        for promptTokens in promptTokenSuite where promptTokens.isEmpty {
            throw RealModelInferenceError.invalidGenerationParameters("Testing prompt token list must not be empty")
        }
        guard maxTokens > 0 else {
            throw RealModelInferenceError.invalidGenerationParameters("Testing max token count must be positive")
        }
        let promptTokens = promptTokenSuite[0]

        try validateConfig(config)
        let weightDirURL = URL(fileURLWithPath: weightDir, isDirectory: true)
        try validateDirectory(weightDirURL)
        try validateMetadataIfPresent(config: config, weightDirURL: weightDirURL)

        let topLevelAssets = try TopLevelAssetLoader.load(
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

        guard config.architecture == .llama else {
            throw RealModelInferenceError.unsupportedArchitecture(
                "generateFromTokenSuiteForTesting currently supports llama-family artifacts only"
            )
        }

        // Compile once for the longest prompt so no prompt triggers a second compile.
        let longestPromptCount = promptTokenSuite.map(\.count).max() ?? promptTokens.count
        let bucket = try compileBucket(
            for: min(config.maxSeq, longestPromptCount + maxTokens),
            channels: config.dModel,
            maxSeq: config.maxSeq
        )

        let seam = Self.resolveDecodePlanSeam(config: config, options: decodeOptions)
        let trunk = try DecodePathPolicy.resolvedTrunk(
            config: config,
            fusedHybridPreferred: seam.fusedHybridPreferred,
            options: seam.options
        )
        var compileTimeMs = 0.0
        var preparation: LlamaTrunkPreparation?
        if trunk != .exactCPU {
            // Compile once for the longest prompt so no prompt triggers a second compile.
            preparation = try engine.prepareLlamaTrunkDecode(
                trunk,
                bucket: bucket,
                sessionKind: "llama testing helper"
            )
            compileTimeMs = preparation?.compileTimeMs ?? 0
        }

        var results: [GenerationResult] = []
        results.reserveCapacity(promptTokenSuite.count)
        for prompt in promptTokenSuite {
            let effectiveMaxTokens = min(maxTokens, max(config.maxSeq - prompt.count, 0))
            guard effectiveMaxTokens > 0 else {
                throw RealModelInferenceError.invalidGenerationParameters(
                    "Prompt of \(prompt.count) tokens leaves no room to generate within context \(config.maxSeq)"
                )
            }
            // Drop the profile string so the next ANE eval cannot smash a
            // heap object that must survive the whole suite.
            results.append(
                try engine.dispatchLlamaTrunkDecode(LlamaDecodeRequest(
                    trunk: trunk,
                    bucket: bucket,
                    sessionKind: "llama testing helper",
                    promptTokens: prompt,
                    effectiveMaxTokens: effectiveMaxTokens,
                    temperature: 0,
                    compileTimeMs: results.isEmpty ? compileTimeMs : 0,
                    preparation: preparation,
                    maxSeq: trunk == .splitHybrid ? bucket : nil,
                    dropsDecodeProfileReport: true
                ))
            )
        }
        return results
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

        let topLevelAssets = try TopLevelAssetLoader.load(
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

        let topLevelAssets = try TopLevelAssetLoader.load(
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
        environment: [String: String]
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

    static func compileAndEvalSingleLayerForTesting(
        config: MultiModelConfig,
        weightDir: String,
        layer: Int,
        spatial: Int,
        input: [Float],
        environment: [String: String] = Self.processEnvironment
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
        environment: [String: String] = Self.processEnvironment
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
        environment: [String: String] = Self.processEnvironment
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
        environment: [String: String]? = nil
    ) throws -> [GPT2AttentionCompileProbeResult] {
        // Public declarations cannot reference internal symbols in default arguments,
        // so the bootstrap seam is applied here instead of in the parameter default.
        let environment = environment ?? Self.processEnvironment
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
        environment: [String: String] = Self.processEnvironment
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

        try ForwardPass.initializeHybridDecodeCaches(surfaceHandles: handles, dim: config.dModel)

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
                    environment: Self.processEnvironment
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

        try ForwardPass.initializeHybridDecodeCaches(surfaceHandles: handles, dim: config.dModel)

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
                    environment: Self.processEnvironment
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
        try mapSurfaceIOToRealModelError { try kCache.withUnsafeMutableBufferPointer { buffer in
            try SurfaceIO.readFP16(
                from: handles[0].kCacheFull,
                into: buffer,
                channelOffset: 0,
                channels: kvDim,
                spatial: maxSeq
            )
        } }
        try mapSurfaceIOToRealModelError { try vCache.withUnsafeMutableBufferPointer { buffer in
            try SurfaceIO.readFP16(
                from: handles[0].vCacheFull,
                into: buffer,
                channelOffset: 0,
                channels: kvDim,
                spatial: maxSeq
            )
        } }
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

        try ForwardPass.initializeHybridDecodeCaches(surfaceHandles: handles, dim: config.dModel)

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
                    environment: Self.processEnvironment
                ),
                postQKVHook: ropeHook,
                timings: &timings
            )
        }

        var kCache = [Float](repeating: 0, count: config.kvDim * maxSeq)
        try mapSurfaceIOToRealModelError { try kCache.withUnsafeMutableBufferPointer { buffer in
            try SurfaceIO.readFP16(
                from: handles[0].kCacheFull,
                into: buffer,
                channelOffset: 0,
                channels: config.kvDim,
                spatial: maxSeq
            )
        } }

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

        try ForwardPass.initializeHybridDecodeCaches(surfaceHandles: handles, dim: config.dModel)

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
                    environment: Self.processEnvironment
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
        try mapSurfaceIOToRealModelError { try kCache.withUnsafeMutableBufferPointer { buffer in
            try SurfaceIO.readFP16(
                from: handles[0].kCacheFull,
                into: buffer,
                channelOffset: 0,
                channels: config.kvDim,
                spatial: maxSeq
            )
        } }
        try mapSurfaceIOToRealModelError { try vCache.withUnsafeMutableBufferPointer { buffer in
            try SurfaceIO.readFP16(
                from: handles[0].vCacheFull,
                into: buffer,
                channelOffset: 0,
                channels: config.kvDim,
                spatial: maxSeq
            )
        } }

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
            maxSeq: maxSeq,
            environment: Self.processEnvironment
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

        try ForwardPass.initializeHybridDecodeCaches(surfaceHandles: handles, dim: config.dModel)

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
                    environment: Self.processEnvironment
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

        try ForwardPass.initializeHybridDecodeCaches(surfaceHandles: handles, dim: config.dModel)

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
                    environment: Self.processEnvironment
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

        try ForwardPass.initializeHybridDecodeCaches(surfaceHandles: handles, dim: config.dModel)

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
                    environment: Self.processEnvironment
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
        try mapSurfaceIOToRealModelError { try kCache.withUnsafeMutableBufferPointer { buffer in
            try SurfaceIO.readFP16(
                from: handles[0].kCacheFull,
                into: buffer,
                channelOffset: 0,
                channels: config.kvDim,
                spatial: maxSeq
            )
        } }
        try mapSurfaceIOToRealModelError { try vCache.withUnsafeMutableBufferPointer { buffer in
            try SurfaceIO.readFP16(
                from: handles[0].vCacheFull,
                into: buffer,
                channelOffset: 0,
                channels: config.kvDim,
                spatial: maxSeq
            )
        } }

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
        weightDir: String,
        environment: [String: String] = Self.processEnvironment
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
            spatial: spatial,
            environment: environment
        )
    }

    static func requireANEHardwareTestsEnabled() throws {
        guard Self.processEnvironment["ANE_HARDWARE_TESTS"] == "1" else {
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

    private final class FusedHybridStepper: LlamaStepping {

        let contextLimit: Int
        let tracksDecodeProfile = true

        private let expectedNLayer: Int
        private let dModel: Int
        private let headDim: Int
        private let ropeTheta: Float
        private let normEps: Float
        private let decodeMaxSeq: Int
        private let tokenEmbedding: [Float]
        private let finalNormGamma: [Float]
        private let selector: LlamaTokenSelector
        private let tokenizerRef: LoadedTokenizer
        private let isCancelled: (() -> Bool)?

        private let xCur: TensorBuffer
        private var decodeState: DecodeState
        private var pendingTimings = HybridDecodeTimingBreakdown()
        private var stepTimings = HybridDecodeTimingBreakdown()

        init(
            config: MultiModelConfig,
            decodeMaxSeq: Int,
            tokenEmbedding: [Float],
            finalNormGamma: [Float],
            tokenizerRef: LoadedTokenizer,
            selector: LlamaTokenSelector,
            isCancelled: (() -> Bool)?
        ) throws {
            self.expectedNLayer = config.nLayer
            self.dModel = config.dModel
            self.headDim = config.headDim
            self.ropeTheta = config.ropeTheta
            self.normEps = Float(config.normEps)
            self.contextLimit = config.maxSeq
            self.decodeMaxSeq = decodeMaxSeq
            self.tokenEmbedding = tokenEmbedding
            self.finalNormGamma = finalNormGamma
            self.tokenizerRef = tokenizerRef
            self.selector = selector
            self.isCancelled = isCancelled
            self.xCur = TensorBuffer(count: config.dModel, zeroed: true)
            do {
                self.decodeState = try DecodeState(maxSeq: decodeMaxSeq)
            } catch {
                throw RealModelInferenceEngine.fusedHybridFallbackError(reason: "fused decode state initialization failed: \(error)")
            }
        }

        func begin(host: inout RealModelInferenceEngine, promptTokens: [TokenID]) throws {
            switch host.fusedHybridReadiness {
            case .compiled:
                break
            case .notCompiled:
                throw RealModelInferenceEngine.fusedHybridFallbackError(
                    reason: """
                        fused N=1 state is incomplete: \
                        layers=\(host.compiledFusedHybridLayers.count)/\(expectedNLayer) \
                        surfaces=\(host.compiledFusedHybridSurfaceHandles.count)/\(expectedNLayer)
                        """
                )
            }

            do {
                try ForwardPass.initializeFusedHybridDecodeCaches(
                    surfaceHandles: host.compiledFusedHybridSurfaceHandles
                )
            } catch {
                throw RealModelInferenceEngine.fusedHybridFallbackError(reason: "fused N=1 cache init failed: \(error)")
            }

            for (position, token) in promptTokens.enumerated() {
                try writeEmbeddingLlama(token: token, into: xCur)
                do {
                    try ForwardPass.runFusedHybridDecodeTimed(
                        xCur: xCur,
                        kernels: host.compiledFusedHybridLayers,
                        surfaceHandles: host.compiledFusedHybridSurfaceHandles,
                        decodeState: &decodeState,
                        headDim: headDim,
                        ropeTheta: ropeTheta,
                        timings: &stepTimings
                    )
                } catch {
                    throw RealModelInferenceEngine.fusedHybridFallbackError(
                        reason: "fused N=1 prefill failed at prompt position \(position): \(error)"
                    )
                }
            }
        }

        func proposal(host: inout RealModelInferenceEngine) throws -> LlamaDecodeProposal {
            let normalized = xCur.withUnsafeBufferPointer {
                RealModelInferenceEngine.rmsNorm(Array($0), weight: finalNormGamma, eps: normEps)
            }
            return .normalizedHidden(normalized)
        }

        func advance(host: inout RealModelInferenceEngine, consuming token: TokenID, generatedCount: Int) throws {
            try writeEmbeddingLlama(token: token, into: xCur)
            stepTimings.reset()
            do {
                try ForwardPass.runFusedHybridDecodeTimed(
                    xCur: xCur,
                    kernels: host.compiledFusedHybridLayers,
                    surfaceHandles: host.compiledFusedHybridSurfaceHandles,
                    decodeState: &decodeState,
                    headDim: headDim,
                    ropeTheta: ropeTheta,
                    timings: &stepTimings
                )
            } catch {
                throw RealModelInferenceEngine.fusedHybridFallbackError(
                    reason: "fused N=1 decode failed at generated token \(generatedCount - 1): \(error)"
                )
            }
            pendingTimings = stepTimings
        }

        func takePendingTimings() -> HybridDecodeTimingBreakdown? { pendingTimings }

        func resolveToken(hidden: [Float], temperature: Float, topP: Float) -> TokenID {
            selector.selectToken(hidden: hidden, temperature: temperature, topP: topP)
        }

        func decodeText(_ tokens: [Int]) -> String {
            tokenizerRef.decode(tokens)
        }

        func throwIfCancelled() throws {
            try RealModelInferenceEngine.throwIfCancelled(isCancelled)
        }

        private func writeEmbeddingLlama(token: TokenID, into buffer: borrowing TensorBuffer) throws {
            let tokenBase = Int(token) * dModel
            guard tokenBase + dModel <= tokenEmbedding.count else {
                throw RealModelInferenceError.runtimeFailure(
                    "Llama embedding OOB: token=\(token), base=\(tokenBase), embeddingCount=\(tokenEmbedding.count), dModel=\(dModel)"
                )
            }
            buffer.withUnsafeMutableBufferPointer { dst in
                for channel in 0..<dModel {
                    dst[channel] = tokenEmbedding[tokenBase + channel]
                }
            }
        }
    }

    private mutating func generateIncrementalFusedHybridLlama(
        promptTokens: [TokenID],
        effectiveMaxTokens: Int,
        temperature: Float,
        topP: Float = 1.0,
        compileTimeMs: Double,
        maxSeq: Int,
        onStep: ((GenerationStep) -> Void)?,
        isCancelled: (() -> Bool)? = nil
    ) throws -> GenerationResult {
        let stepper = try FusedHybridStepper(
            config: config,
            decodeMaxSeq: maxSeq,
            tokenEmbedding: llamaAssets.tokenEmbedding,
            finalNormGamma: llamaAssets.finalNormGamma,
            tokenizerRef: tokenizer,
            selector: makeLlamaTokenSelector(),
            isCancelled: isCancelled
        )
        let session = LlamaServingSession(
            stepper: stepper,
            effectiveMaxTokens: effectiveMaxTokens,
            endOfSequenceToken: config.eosToken,
            temperature: temperature,
            topP: topP,
            onStep: onStep
        )
        let (emission, decodeProfileReport) = try session.run(host: &self, promptTokens: promptTokens)
        return emission.makeResult(
            compileTimeMs: compileTimeMs,
            exactHeadBackend: classifierStrategy.exactHeadBackendLabel,
            cachedBindingsEnabled: false,
            trunk: .fusedHybrid,
            hopsPerToken: Self.fusedHopsPerToken(nLayer: config.nLayer),
            decodeProfileReport: decodeProfileReport
        )
    }

    /// Split hybrid Trunk stepper: ANE QKV, host attention, ANE FFN per decode step.
    ///
    /// Owns the per-run hidden buffer, decode state, RoPE scratch, and CPU-exact-QKV
    /// buffers; compiled programs and surfaces stay on the host engine.
    private final class SplitHybridStepper: LlamaStepping {

        let contextLimit: Int
        let tracksDecodeProfile = true

        private let expectedNLayer: Int
        private let dModel: Int
        private let nHeads: Int
        private let nKVHeads: Int
        private let headDim: Int
        private let ropeTheta: Float
        private let normEps: Float
        private let vocabSize: Int
        private let decodeMaxSeq: Int
        private let metalAttention: MetalAttentionKernel
        private let cachedBindings: [MetalAttentionKernel.CachedLayerBindings]?
        private let preferCPUDecodeAttention: Bool
        private let greedyHeadMode: HybridGreedyHeadMode
        private let useANEGreedyHead: Bool
        private let useCPUExactGreedyHead: Bool
        private let cpuExactQKVLayerWeights: [LlamaCPUQKVWeights]?
        private let tokenEmbedding: [Float]
        private let finalNormGamma: [Float]
        private let selector: LlamaTokenSelector
        private let tokenizerRef: LoadedTokenizer
        private let isCancelled: (() -> Bool)?

        private let xCur: TensorBuffer
        private var decodeState: DecodeState
        private var pendingTimings = HybridDecodeTimingBreakdown()
        private var stepTimings = HybridDecodeTimingBreakdown()

        // RoPE scratch, allocated once per session run.
        private var ropeQBuf: UnsafeMutableBufferPointer<Float>?
        private var ropeKBuf: UnsafeMutableBufferPointer<Float>?
        private var layerQKNormWeights: [LlamaQKNormWeights?] = []

        // CPU exact QKV scratch, allocated when the override is active.
        private var cpuQKVHiddenBuf: UnsafeMutableBufferPointer<Float>?
        private var cpuQKVAttnNormedBuf: UnsafeMutableBufferPointer<Float>?
        private var cpuQBuf: UnsafeMutableBufferPointer<Float>?
        private var cpuKBuf: UnsafeMutableBufferPointer<Float>?
        private var cpuVBuf: UnsafeMutableBufferPointer<Float>?

        init(
            config: MultiModelConfig,
            decodeMaxSeq: Int,
            metalAttention: MetalAttentionKernel,
            cachedBindings: [MetalAttentionKernel.CachedLayerBindings]?,
            preferCPUDecodeAttention: Bool,
            greedyHeadMode: HybridGreedyHeadMode,
            useANEGreedyHead: Bool,
            useCPUExactGreedyHead: Bool,
            cpuExactQKVLayerWeights: [LlamaCPUQKVWeights]?,
            tokenEmbedding: [Float],
            finalNormGamma: [Float],
            tokenizerRef: LoadedTokenizer,
            selector: LlamaTokenSelector,
            isCancelled: (() -> Bool)?
        ) throws {
            self.expectedNLayer = config.nLayer
            self.dModel = config.dModel
            self.nHeads = config.nHead
            self.nKVHeads = config.nKVHead
            self.headDim = config.headDim
            self.ropeTheta = config.ropeTheta
            self.normEps = Float(config.normEps)
            self.vocabSize = config.vocab
            self.contextLimit = config.maxSeq
            self.decodeMaxSeq = decodeMaxSeq
            self.metalAttention = metalAttention
            self.cachedBindings = cachedBindings
            self.preferCPUDecodeAttention = preferCPUDecodeAttention
            self.greedyHeadMode = greedyHeadMode
            self.useANEGreedyHead = useANEGreedyHead
            self.useCPUExactGreedyHead = useCPUExactGreedyHead
            self.cpuExactQKVLayerWeights = cpuExactQKVLayerWeights
            self.tokenEmbedding = tokenEmbedding
            self.finalNormGamma = finalNormGamma
            self.tokenizerRef = tokenizerRef
            self.selector = selector
            self.isCancelled = isCancelled
            self.xCur = TensorBuffer(count: config.dModel, zeroed: true)
            do {
                self.decodeState = try DecodeState(maxSeq: decodeMaxSeq)
            } catch {
                throw RealModelInferenceError.runtimeFailure("Llama hybrid decode state initialization failed: \(error)")
            }
        }

        deinit {
            ropeQBuf?.deallocate()
            ropeKBuf?.deallocate()
            cpuQKVHiddenBuf?.deallocate()
            cpuQKVAttnNormedBuf?.deallocate()
            cpuQBuf?.deallocate()
            cpuKBuf?.deallocate()
            cpuVBuf?.deallocate()
        }

        func begin(host: inout RealModelInferenceEngine, promptTokens: [TokenID]) throws {
            switch host.splitHybridReadiness {
            case .compiled:
                break
            case .notCompiled:
                throw RealModelInferenceError.runtimeFailure(
                    "Llama hybrid decode state is unavailable: layers=\(host.compiledHybridLayers.count)/\(expectedNLayer) surfaces=\(host.compiledHybridSurfaceHandles.count)/\(expectedNLayer) qkNorms=\(host.compiledHybridLlamaQKNormWeights.count)/\(expectedNLayer) head=\(host.compiledHybridHead.count) headSpatial=\(host.compiledHybridHeadSpatial)"
                )
            }

            try ForwardPass.initializeHybridDecodeCaches(
                surfaceHandles: host.compiledHybridSurfaceHandles,
                dim: dModel
            )

            layerQKNormWeights = host.compiledHybridLlamaQKNormWeights
            allocateRoPEScratchIfNeeded()
            allocateCPUQKVScratchIfNeeded()

            for (position, token) in promptTokens.enumerated() {
                try writeEmbeddingLlama(token: token, into: xCur)
                do {
                    try ForwardPass.runHybridDecodeTimed(
                        xCur: xCur,
                        kernels: host.compiledHybridLayers,
                        surfaceHandles: host.compiledHybridSurfaceHandles,
                        metalAttention: metalAttention,
                        decodeState: &decodeState,
                        dim: dModel,
                        nHeads: nHeads,
                        nKVHeads: nKVHeads,
                        headDim: headDim,
                        preferCPUDecodeAttention: preferCPUDecodeAttention,
                        qkvOverride: makeCPUExactQKVOverride(),
                        postQKVHook: currentMetalRoPEConfig != nil ? nil : try makeRopeHook(),
                        readFinalOutputIntoXCur: !useANEGreedyHead,
                        cachedBindings: cachedBindings,
                        metalRoPEConfig: currentMetalRoPEConfig,
                        timings: &stepTimings
                    )
                } catch {
                    throw RealModelInferenceError.runtimeFailure(
                        "Llama hybrid prefill failed at prompt position \(position): \(error)"
                    )
                }
            }
        }

        func proposal(host: inout RealModelInferenceEngine) throws -> LlamaDecodeProposal {
            let nextToken: TokenID
            if useANEGreedyHead {
                do {
                    if greedyHeadMode != .normThenClassifier {
                        try host.compiledHybridGreedyClassifier[0].kernel.eval()
                    } else {
                        try host.compiledHybridGreedyNorm[0].kernel.eval()
                        try host.compiledHybridGreedyClassifier[0].kernel.eval()
                    }
                    let argmax = try RealModelInferenceEngine.greedyArgmax(
                        classifier: host.compiledHybridGreedyClassifier[0],
                        headSpatial: host.compiledHybridHeadSpatial,
                        vocab: vocabSize
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
                return .selected(nextToken)
            }

            if useCPUExactGreedyHead {
                let normalized = xCur.withUnsafeBufferPointer {
                    RealModelInferenceEngine.rmsNorm(Array($0), weight: finalNormGamma, eps: normEps)
                }
                return .normalizedHidden(normalized)
            }

            let headSpatial = host.compiledHybridHeadSpatial
            do {
                try xCur.withUnsafeBufferPointer { buffer in
                    try RealModelInferenceEngine.writeFP32SpatialSlice(
                        to: host.compiledHybridHead[0].inputSurface,
                        spatialIndex: 0,
                        spatial: headSpatial,
                        data: buffer,
                        channels: dModel
                    )
                }
                try host.compiledHybridHead[0].kernel.eval()
                var normalized = [Float](repeating: 0, count: dModel)
                try normalized.withUnsafeMutableBufferPointer { buffer in
                    try RealModelInferenceEngine.readFP32SpatialSlice(
                        from: host.compiledHybridHead[0].outputSurface,
                        spatialIndex: 0,
                        spatial: headSpatial,
                        into: buffer,
                        channels: dModel
                    )
                }
                return .normalizedHidden(normalized)
            } catch {
                throw RealModelInferenceError.runtimeFailure("Llama hybrid step head evaluation failed: \(error)")
            }
        }

        func advance(host: inout RealModelInferenceEngine, consuming token: TokenID, generatedCount: Int) throws {
            try writeEmbeddingLlama(token: token, into: xCur)
            stepTimings.reset()
            do {
                try ForwardPass.runHybridDecodeTimed(
                    xCur: xCur,
                    kernels: host.compiledHybridLayers,
                    surfaceHandles: host.compiledHybridSurfaceHandles,
                    metalAttention: metalAttention,
                    decodeState: &decodeState,
                    dim: dModel,
                    nHeads: nHeads,
                    nKVHeads: nKVHeads,
                    headDim: headDim,
                    preferCPUDecodeAttention: preferCPUDecodeAttention,
                    qkvOverride: makeCPUExactQKVOverride(),
                    postQKVHook: currentMetalRoPEConfig != nil ? nil : try makeRopeHook(),
                    readFinalOutputIntoXCur: !useANEGreedyHead,
                    cachedBindings: cachedBindings,
                    metalRoPEConfig: currentMetalRoPEConfig,
                    timings: &stepTimings
                )
            } catch {
                throw RealModelInferenceError.runtimeFailure(
                    "Llama hybrid decode failed at generated token \(generatedCount - 1): \(error)"
                )
            }
            pendingTimings = stepTimings
        }

        func takePendingTimings() -> HybridDecodeTimingBreakdown? { pendingTimings }

        func resolveToken(hidden: [Float], temperature: Float, topP: Float) -> TokenID {
            selector.selectToken(hidden: hidden, temperature: temperature, topP: topP)
        }

        func decodeText(_ tokens: [Int]) -> String {
            tokenizerRef.decode(tokens)
        }

        func throwIfCancelled() throws {
            try RealModelInferenceEngine.throwIfCancelled(isCancelled)
        }

        // MARK: RoPE hook machinery

        private var currentMetalRoPEConfig: MetalAttentionKernel.MetalRoPEConfig?

        private func allocateRoPEScratchIfNeeded() {
            guard ropeQBuf == nil else { return }
            let qBufSize = nHeads * headDim
            let kBufSize = nKVHeads * headDim
            ropeQBuf = UnsafeMutableBufferPointer<Float>.allocate(capacity: qBufSize)
            ropeKBuf = UnsafeMutableBufferPointer<Float>.allocate(capacity: kBufSize)

            let hasAnyQKNorm = layerQKNormWeights.contains { $0 != nil }
            currentMetalRoPEConfig =
                RealModelInferenceEngine.supportsLlamaMetalRoPEFastPath(
                    cachedBindingsAvailable: cachedBindings != nil && !hasAnyQKNorm,
                    kBindingContainsKVCache: false
                )
                ? MetalAttentionKernel.MetalRoPEConfig(
                    nHeads: nHeads, nKVHeads: nKVHeads, headDim: headDim, theta: ropeTheta
                )
                : nil
        }

        private func makeRopeHook() throws -> (Int, IOSurfaceRef, IOSurfaceRef, Int, Int) throws -> Void {
            guard let ropeQBuf, let ropeKBuf else {
                throw RealModelInferenceError.runtimeFailure("Llama hybrid RoPE scratch unavailable")
            }
            return { [weak self] layerIndex, qSurf, kSurf, laneSp, tokenIndex in
                guard let self else { return }
                do {
                    try SurfaceIO.readFP16SpatialSlice(
                        from: qSurf, channelOffset: 0, spatialIndex: 0, spatial: laneSp,
                        into: ropeQBuf, channels: ropeQBuf.count
                    )
                    try SurfaceIO.readFP16SpatialSlice(
                        from: kSurf, channelOffset: 0, spatialIndex: 0, spatial: laneSp,
                        into: ropeKBuf, channels: ropeKBuf.count
                    )
                } catch {
                    throw ANEError.invalidArguments("RoPE hook surface read failed: \(error)")
                }

                if let norms = self.layerQKNormWeights[layerIndex] {
                    norms.q.withUnsafeBufferPointer { weights in
                        RMSNorm.applyPerHeadSingleTokenInPlace(
                            values: ropeQBuf.baseAddress!,
                            headCount: self.nHeads,
                            headDim: self.headDim,
                            weights: weights.baseAddress!,
                            epsilon: self.normEps
                        )
                    }
                    norms.k.withUnsafeBufferPointer { weights in
                        RMSNorm.applyPerHeadSingleTokenInPlace(
                            values: ropeKBuf.baseAddress!,
                            headCount: self.nKVHeads,
                            headDim: self.headDim,
                            weights: weights.baseAddress!,
                            epsilon: self.normEps
                        )
                    }
                }

                RoPE.applyDecodeStep(
                    q: ropeQBuf.baseAddress!,
                    k: ropeKBuf.baseAddress!,
                    nHeads: self.nHeads,
                    nKVHeads: self.nKVHeads,
                    headDim: self.headDim,
                    position: tokenIndex,
                    theta: self.ropeTheta
                )

                do {
                    try SurfaceIO.writeFP16SpatialSlice(
                        to: qSurf, channelOffset: 0, spatialIndex: 0, spatial: laneSp,
                        data: UnsafeBufferPointer(ropeQBuf), channels: ropeQBuf.count
                    )
                    try SurfaceIO.writeFP16SpatialSlice(
                        to: kSurf, channelOffset: 0, spatialIndex: 0, spatial: laneSp,
                        data: UnsafeBufferPointer(ropeKBuf), channels: ropeKBuf.count
                    )
                } catch {
                    throw ANEError.invalidArguments("RoPE hook surface write failed: \(error)")
                }
            }
        }

        // MARK: CPU exact QKV override machinery

        private func allocateCPUQKVScratchIfNeeded() {
            guard cpuExactQKVLayerWeights != nil, cpuQKVHiddenBuf == nil else { return }
            cpuQKVHiddenBuf = UnsafeMutableBufferPointer<Float>.allocate(capacity: dModel)
            cpuQKVAttnNormedBuf = UnsafeMutableBufferPointer<Float>.allocate(capacity: dModel)
            cpuQBuf = UnsafeMutableBufferPointer<Float>.allocate(capacity: nHeads * headDim)
            cpuKBuf = UnsafeMutableBufferPointer<Float>.allocate(capacity: nKVHeads * headDim)
            cpuVBuf = UnsafeMutableBufferPointer<Float>.allocate(capacity: nKVHeads * headDim)
        }

        private func makeCPUExactQKVOverride() -> ((Int, HybridDecodeSurfaceHandles, Int, Int) throws -> Void)? {
            guard let layerWeights = cpuExactQKVLayerWeights,
                  let hiddenBuf = cpuQKVHiddenBuf,
                  let attnNormedBuf = cpuQKVAttnNormedBuf,
                  let qBuf = cpuQBuf,
                  let kBuf = cpuKBuf,
                  let vBuf = cpuVBuf else {
                return nil
            }
            return { layerIndex, handles, laneSp, _ in
                let weights = layerWeights[layerIndex]
                do {
                    try SurfaceIO.readFP16SpatialSlice(
                        from: handles.qkvIn,
                        channelOffset: 0,
                        spatialIndex: 0,
                        spatial: laneSp,
                        into: hiddenBuf,
                        channels: self.dModelForOverride
                    )
                } catch {
                    throw ANEError.invalidArguments("CPU exact QKV input read failed: \(error)")
                }

                var sumSq: Float = 0
                vDSP_dotpr(hiddenBuf.baseAddress!, 1, hiddenBuf.baseAddress!, 1, &sumSq, vDSP_Length(self.dModelForOverride))
                var invRms = 1.0 / sqrtf(sumSq / Float(self.dModelForOverride) + self.normEps)
                vDSP_vsmul(hiddenBuf.baseAddress!, 1, &invRms, attnNormedBuf.baseAddress!, 1, vDSP_Length(self.dModelForOverride))
                weights.rmsAtt.withUnsafeBufferPointer { gamma in
                    vDSP_vmul(attnNormedBuf.baseAddress!, 1, gamma.baseAddress!, 1, attnNormedBuf.baseAddress!, 1, vDSP_Length(self.dModelForOverride))
                }

                RealModelInferenceEngine.multiplyRowMajorMatrix(
                    matrix: weights.wq,
                    rows: self.qDimForOverride,
                    cols: self.dModelForOverride,
                    vector: UnsafeBufferPointer(attnNormedBuf),
                    into: qBuf
                )
                RealModelInferenceEngine.multiplyRowMajorMatrix(
                    matrix: weights.wk,
                    rows: self.kvDimForOverride,
                    cols: self.dModelForOverride,
                    vector: UnsafeBufferPointer(attnNormedBuf),
                    into: kBuf
                )
                RealModelInferenceEngine.multiplyRowMajorMatrix(
                    matrix: weights.wv,
                    rows: self.kvDimForOverride,
                    cols: self.dModelForOverride,
                    vector: UnsafeBufferPointer(attnNormedBuf),
                    into: vBuf
                )
                if let qkvBias = weights.qkvBias {
                    RealModelInferenceEngine.addBiasInPlace(qkvBias.q, into: qBuf)
                    RealModelInferenceEngine.addBiasInPlace(qkvBias.k, into: kBuf)
                    RealModelInferenceEngine.addBiasInPlace(qkvBias.v, into: vBuf)
                }

                do {
                    try SurfaceIO.writeFP16SpatialSlice(
                        to: handles.qOut,
                        channelOffset: 0,
                        spatialIndex: 0,
                        spatial: laneSp,
                        data: UnsafeBufferPointer(qBuf),
                        channels: self.qDimForOverride
                    )
                    try SurfaceIO.writeFP16SpatialSlice(
                        to: handles.kOut,
                        channelOffset: 0,
                        spatialIndex: 0,
                        spatial: laneSp,
                        data: UnsafeBufferPointer(kBuf),
                        channels: self.kvDimForOverride
                    )
                    try SurfaceIO.writeFP16SpatialSlice(
                        to: handles.vOut,
                        channelOffset: 0,
                        spatialIndex: 0,
                        spatial: laneSp,
                        data: UnsafeBufferPointer(vBuf),
                        channels: self.kvDimForOverride
                    )
                } catch {
                    throw ANEError.invalidArguments("CPU exact QKV surface write failed: \(error)")
                }
            }
        }

        private var dModelForOverride: Int { dModel }
        private var qDimForOverride: Int { nHeads * headDim }
        private var kvDimForOverride: Int { nKVHeads * headDim }

        // MARK: Shared helpers

        private func writeEmbeddingLlama(token: TokenID, into buffer: borrowing TensorBuffer) throws {
            let tokenBase = Int(token) * dModel
            guard tokenBase + dModel <= tokenEmbedding.count else {
                throw RealModelInferenceError.runtimeFailure(
                    "Llama embedding OOB: token=\(token), base=\(tokenBase), embeddingCount=\(tokenEmbedding.count), dModel=\(dModel)"
                )
            }
            buffer.withUnsafeMutableBufferPointer { dst in
                for channel in 0..<dModel {
                    dst[channel] = tokenEmbedding[tokenBase + channel]
                }
            }
        }
    }

    private mutating func generateIncrementalHybridLlama(
        promptTokens: [TokenID],
        effectiveMaxTokens: Int,
        temperature: Float,
        topP: Float = 1.0,
        compileTimeMs: Double,
        maxSeq: Int,
        metalAttention: MetalAttentionKernel,
        onStep: ((GenerationStep) -> Void)?,
        isCancelled: (() -> Bool)? = nil
    ) throws -> GenerationResult {
        let cachedBindings: [MetalAttentionKernel.CachedLayerBindings]? = try Self.makeHybridCachedBindingsOrFallback(
            config: config,
            environment: policies.environment
        ) {
            try compiledHybridSurfaceHandles.map { handles in
                try metalAttention.createCachedLayerBindings(
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
            }
        }

        let environment = policies.environment
        let greedyHeadMode = hybridGreedyHeadMode(environment: environment)
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

        let useCPUExactQKV = Self.prefersCPUExactQKV(config: config, environment: environment)
        var cpuExactQKVLayerWeights: [LlamaCPUQKVWeights]? = nil
        if useCPUExactQKV {
            cpuExactQKVLayerWeights = try (0..<config.nLayer).map { layerIndex in
                let paths = LayerWeightPaths.forLayer(layerIndex, config: config, blobDir: weightDirURL.path)
                return try Self.loadLlamaCPUQKVWeights(config: config, paths: paths)
            }
        }

        let stepper = try SplitHybridStepper(
            config: config,
            decodeMaxSeq: maxSeq,
            metalAttention: metalAttention,
            cachedBindings: cachedBindings,
            preferCPUDecodeAttention: Self.prefersCPUDecodeAttention(config: config, environment: environment),
            greedyHeadMode: greedyHeadMode,
            useANEGreedyHead: useANEGreedyHead,
            useCPUExactGreedyHead: useCPUExactGreedyHead,
            cpuExactQKVLayerWeights: cpuExactQKVLayerWeights,
            tokenEmbedding: llamaAssets.tokenEmbedding,
            finalNormGamma: llamaAssets.finalNormGamma,
            tokenizerRef: tokenizer,
            selector: makeLlamaTokenSelector(),
            isCancelled: isCancelled
        )
        let session = LlamaServingSession(
            stepper: stepper,
            effectiveMaxTokens: effectiveMaxTokens,
            endOfSequenceToken: config.eosToken,
            temperature: temperature,
            topP: topP,
            onStep: onStep
        )
        let (emission, decodeProfileReport) = try session.run(host: &self, promptTokens: promptTokens)
        return emission.makeResult(
            compileTimeMs: compileTimeMs,
            exactHeadBackend: greedyHeadMode == .classifierOnlyFactored && useANEGreedyHead
                ? "ane_factored_classifier"
                : classifierStrategy.exactHeadBackendLabel,
            cachedBindingsEnabled: cachedBindings != nil,
            trunk: .splitHybrid,
            decodeProfileReport: decodeProfileReport
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
        let coreWeights = try TopLevelAssetLoader.loadLlamaCoreWeights(
            config: config,
            topLevelPaths: topLevelPaths
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
            tokenEmbedding: coreWeights.tokenEmbedding,
            finalNormGamma: coreWeights.finalNormGamma,
            lmHead: coreWeights.lmHead,
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

        var clock = GenerateClock()
        try fullRuntime.prefill(promptTokens: promptTokens)
        try draftRuntime.prefill(promptTokens: promptTokens)
        clock.markPrefillEnd()
        let prefillMs = clock.prefillMs()
        let submitNS = clock.submitNS

        let generationStart = DispatchTime.now().uptimeNanoseconds
        let tokenizer = self.tokenizer
        var emission = EmissionCore(
            promptTokens: promptTokens,
            capacity: effectiveMaxTokens,
            eos: .fromConfig(config.eosToken.map(Int.init)),
            onStep: onStep,
            decodeText: { tokenizer.decode($0) },
            startNanos: generationStart
        )
        var committedExactTokensTotal = 0
        var acceptedFutureTokensTotal = 0
        var speculativePassCount = 0

        let firstToken = fullRuntime.selectGreedyToken()
        if emission.terminatesDecoding(firstToken) {
            emission.recordTerminalToken(firstToken)
            return emission.makeResult(
                compileTimeMs: compileTimeMs,
                exactHeadBackend: "cpu_exact_two_token_draft",
                trunk: .exactCPU,
                tokensPerSecondOverride: 0,
                textOverride: tokenizer.decode((promptTokens + [firstToken]).map(Int.init))
            )
        }
        let firstEmission = DispatchTime.now().uptimeNanoseconds
        emission.emit(firstToken, at: firstEmission)
        if emission.generatedTokenCount >= effectiveMaxTokens || emission.allTokensCount >= config.maxSeq {
            return emission.makeResult(
                compileTimeMs: compileTimeMs,
                exactHeadBackend: "cpu_exact_two_token_draft",
                trunk: .exactCPU
            )
        }

        try fullRuntime.advance(token: firstToken)
        try draftRuntime.advance(token: firstToken)

        while emission.generatedTokenCount < effectiveMaxTokens, emission.allTokensCount < config.maxSeq {
            speculativePassCount += 1
            let remainingTokenBudget = min(
                effectiveMaxTokens - emission.generatedTokenCount,
                config.maxSeq - emission.allTokensCount
            )
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
            emission.emit(exactToken0, at: firstRoundEmission)
            committedInPass += 1
            if emission.terminatesDecoding(exactToken0) {
                committedExactTokensTotal += committedInPass
                acceptedFutureTokensTotal += acceptedInPass
                break
            }
            try fullRuntime.advance(token: exactToken0)
            if exactToken0 != proposedToken0 {
                try draftRuntime.advance(token: exactToken0)
            }

            if emission.generatedTokenCount >= effectiveMaxTokens || emission.allTokensCount >= config.maxSeq || remainingTokenBudget <= 1 {
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
            emission.emit(exactToken1, at: secondRoundEmission)
            committedInPass += 1
            if emission.terminatesDecoding(exactToken1) {
                committedExactTokensTotal += committedInPass
                acceptedFutureTokensTotal += acceptedInPass
                break
            }
            try fullRuntime.advance(token: exactToken1)
            try draftRuntime.advance(token: exactToken1)

            committedExactTokensTotal += committedInPass
            acceptedFutureTokensTotal += acceptedInPass
        }

        let committedExactTokensPerPass = speculativePassCount == 0
            ? nil
            : Double(committedExactTokensTotal) / Double(speculativePassCount)
        let acceptedFutureTokensPerPass = speculativePassCount == 0
            ? nil
            : Double(acceptedFutureTokensTotal) / Double(speculativePassCount)

        return emission.makeResult(
            compileTimeMs: compileTimeMs,
            exactHeadBackend: "cpu_exact_two_token_draft",
            committedExactTokensPerPass: committedExactTokensPerPass,
            acceptedFutureTokensPerPass: acceptedFutureTokensPerPass,
            trunk: .exactCPU
        )
    }

    /// Exact-CPU Trunk stepper: transformer layers on the CPU, also the Qwen oracle.
    ///
    /// Owns the KV caches and rolling hidden state for one session run.
    private final class ExactCPUStepper: LlamaStepping {

        let contextLimit: Int
        let tracksDecodeProfile = false

        private let config: MultiModelConfig
        private let weights: CachedExactCPULlamaWeights
        private let roundIntermediatesToFP16: Bool
        private let selector: LlamaTokenSelector
        private let tokenizerRef: LoadedTokenizer
        private let isCancelled: (() -> Bool)?

        private var kCaches: [[Float]] = []
        private var vCaches: [[Float]] = []
        private var lastHidden: [Float] = []
        private var nextPosition = 0

        init(
            config: MultiModelConfig,
            maxSeq: Int,
            weights: CachedExactCPULlamaWeights,
            tokenizerRef: LoadedTokenizer,
            selector: LlamaTokenSelector,
            isCancelled: (() -> Bool)?
        ) {
            self.config = config
            self.contextLimit = maxSeq
            self.weights = weights
            self.roundIntermediatesToFP16 =
                RealModelInferenceEngine.shouldRoundCPUExactDecodeIntermediatesToFP16()
            self.tokenizerRef = tokenizerRef
            self.selector = selector
            self.isCancelled = isCancelled
        }

        func begin(host: inout RealModelInferenceEngine, promptTokens: [TokenID]) throws {
            kCaches = Array(
                repeating: [Float](repeating: 0, count: config.kvDim * contextLimit),
                count: config.nLayer
            )
            vCaches = Array(
                repeating: [Float](repeating: 0, count: config.kvDim * contextLimit),
                count: config.nLayer
            )
            lastHidden = [Float](repeating: 0, count: config.dModel)
            for (position, token) in promptTokens.enumerated() {
                lastHidden = try forwardToken(token, position: position)
            }
            nextPosition = promptTokens.count
        }

        func proposal(host: inout RealModelInferenceEngine) throws -> LlamaDecodeProposal {
            let normalized = RealModelInferenceEngine.rmsNorm(
                lastHidden,
                weight: weights.finalNormGamma,
                eps: Float(config.normEps)
            )
            return .normalizedHidden(normalized)
        }

        func advance(host: inout RealModelInferenceEngine, consuming token: TokenID, generatedCount: Int) throws {
            lastHidden = try forwardToken(token, position: nextPosition)
            nextPosition += 1
        }

        func takePendingTimings() -> HybridDecodeTimingBreakdown? { nil }

        func resolveToken(hidden: [Float], temperature: Float, topP: Float) -> TokenID {
            selector.selectToken(hidden: hidden, temperature: temperature, topP: topP)
        }

        func decodeText(_ tokens: [Int]) -> String {
            tokenizerRef.decode(tokens)
        }

        func throwIfCancelled() throws {
            try RealModelInferenceEngine.throwIfCancelled(isCancelled)
        }

        private func forwardToken(_ token: TokenID, position: Int) throws -> [Float] {
            let tokenIndex = Int(token)
            let tokenEnd = (tokenIndex + 1) * config.dModel
            guard tokenIndex < config.vocab, tokenEnd <= weights.tokenEmbedding.count else {
                throw RealModelInferenceError.runtimeFailure(
                    "Llama embedding OOB: token=\(token), base=\(tokenIndex * config.dModel), embeddingCount=\(weights.tokenEmbedding.count), dModel=\(config.dModel)"
                )
            }
            var hidden = Array(weights.tokenEmbedding[tokenIndex * config.dModel..<tokenEnd])
            for layerIndex in 0..<config.nLayer {
                hidden = RealModelInferenceEngine.exactCPULlamaLayerForward(
                    hidden: hidden,
                    layer: weights.layers[layerIndex],
                    config: config,
                    position: position,
                    kCache: &kCaches[layerIndex],
                    vCache: &vCaches[layerIndex],
                    cacheStride: contextLimit,
                    roundIntermediatesToFP16: roundIntermediatesToFP16
                )
            }
            return hidden
        }
    }

    private mutating func generateIncrementalExactCPULlama(
        promptTokens: [TokenID],
        effectiveMaxTokens: Int,
        temperature: Float,
        topP: Float = 1.0,
        compileTimeMs: Double,
        maxSeq: Int,
        onStep: ((GenerationStep) -> Void)?,
        isCancelled: (() -> Bool)? = nil
    ) throws -> GenerationResult {
        guard !promptTokens.isEmpty else {
            throw RealModelInferenceError.invalidGenerationParameters("Prompt tokens must not be empty")
        }
        let weights = try loadCachedExactCPULlamaWeights()
        let stepper = ExactCPUStepper(
            config: config,
            maxSeq: maxSeq,
            weights: weights,
            tokenizerRef: tokenizer,
            selector: makeLlamaTokenSelector(),
            isCancelled: isCancelled
        )
        let session = LlamaServingSession(
            stepper: stepper,
            effectiveMaxTokens: effectiveMaxTokens,
            endOfSequenceToken: config.eosToken,
            temperature: temperature,
            topP: topP,
            onStep: onStep
        )
        let (emission, _) = try session.run(host: &self, promptTokens: promptTokens)
        return emission.makeResult(
            compileTimeMs: compileTimeMs,
            exactHeadBackend: classifierStrategy.exactHeadBackendLabel,
            trunk: .exactCPU
        )
    }

    static func compileLayers(
        config: MultiModelConfig,
        weightDirURL: URL,
        bucket: Int,
        environment: [String: String] = Self.processEnvironment
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

    static func loadHybridLayerWeights(
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
        let qkvBias = try loadLlamaQKVBiasWeights(config: config, paths: paths)
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
            kNormDim: qkNormWeights == nil ? nil : config.headDim,
            hasQKVBias: qkvBias != nil
        )

        try loadTensor(weights.rmsAtt, from: paths.rmsAtt, expectedCount: config.dModel)
        try loadTensor(weights.Wq, from: paths.wq, expectedCount: config.dModel * qDim)
        try loadTensor(weights.Wk, from: paths.wk, expectedCount: config.dModel * kvDim)
        try loadTensor(weights.Wv, from: paths.wv, expectedCount: config.dModel * kvDim)
        try loadTensor(weights.Wo, from: paths.wo, expectedCount: config.dModel * qDim)
        if let qkvBias {
            weights.bq.withUnsafeMutableBufferPointer { dst in
                _ = dst.initialize(from: qkvBias.q)
            }
            weights.bk.withUnsafeMutableBufferPointer { dst in
                _ = dst.initialize(from: qkvBias.k)
            }
            weights.bv.withUnsafeMutableBufferPointer { dst in
                _ = dst.initialize(from: qkvBias.v)
            }
        }
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

    static func loadLlamaQKNormWeights(
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

    /// Load `bq`/`bk`/`bv` when the layer has them. Qwen2-family checkpoints bias q/k/v;
    /// plain llama checkpoints ship no bias files. A partial set is a converter bug and
    /// must fail loudly rather than silently drop a bias term.
    static func loadLlamaQKVBiasWeights(
        config: MultiModelConfig,
        paths: LayerWeightPaths
    ) throws -> LlamaQKVBiasWeights? {
        let presence = [paths.bq, paths.bk, paths.bv].map { fileExists(at: $0) }
        guard presence.contains(true) else {
            return nil
        }
        guard !presence.contains(false),
              let bqPath = paths.bq,
              let bkPath = paths.bk,
              let bvPath = paths.bv else {
            let layerDirectory = URL(fileURLWithPath: paths.wq).deletingLastPathComponent()
            throw RealModelInferenceError.runtimeFailure(
                "Incomplete llama Q/K/V bias weights for \(layerDirectory.path); expected all of bq.bin, bk.bin, bv.bin"
            )
        }
        return LlamaQKVBiasWeights(
            q: try loadWeightTablePreferringFloat32Sidecar(at: bqPath, expectedCount: config.attentionDim),
            k: try loadWeightTablePreferringFloat32Sidecar(at: bkPath, expectedCount: config.kvDim),
            v: try loadWeightTablePreferringFloat32Sidecar(at: bvPath, expectedCount: config.kvDim)
        )
    }

    static func loadLlamaCPUQKVWeights(
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
            kNorm: qkNormWeights?.k,
            qkvBias: try loadLlamaQKVBiasWeights(config: config, paths: paths)
        )
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
        environment: [String: String] = Self.processEnvironment
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
        environment: [String: String] = Self.processEnvironment
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

    static func compileHead(
        config: MultiModelConfig,
        weightDirURL: URL,
        assets: GPT2TopLevelAssets,
        spatial: Int,
        inputDType: ANEDType = .fp32,
        outputDType: ANEDType = .fp32,
        environment: [String: String]
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

    static func compileClassifier(
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

    static func compileLlamaHead(
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

    static func compileLlamaClassifier(
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

    static func compileLlamaRMSNormClassifier(
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

    static func compileLlamaFactoredClassifier(
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
        environment: [String: String] = Self.processEnvironment
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
        environment: [String: String] = Self.processEnvironment
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
        environment: [String: String] = Self.processEnvironment
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

    static func firstInputSurface(from layers: borrowing LayerStorage<CompiledLayer>) throws -> IOSurfaceRef {
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
        guard config.dModel == config.nHead * config.headDim else {
            throw RealModelInferenceError.invalidConfig("dModel must equal nHead * headDim")
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

    static func validateDirectory(_ url: URL) throws {
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

    static func resolveBundleWeightReference(
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

    static func compileBlobPath(actualPath: String, rootDir: URL) -> String {
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

    static func milliseconds(from nanoseconds: UInt64) -> Double {
        Double(nanoseconds) / 1_000_000
    }

    @discardableResult
    private static func dumpDebugMIL(_ mil: String, filename: String) -> String {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(filename)
        try? mil.write(to: url, atomically: true, encoding: .utf8)
        return url.path
    }

    static func emptyStorage<Element: ~Copyable>(_: Element.Type = Element.self) -> LayerStorage<Element> {
        LayerStorage<Element>(count: 0) { _ in
            fatalError("unreachable empty storage initializer")
        }
    }

    static func throwIfCancelled(_ isCancelled: (() -> Bool)?) throws {
        if isCancelled?() == true {
            throw RealModelInferenceError.cancelled
        }
    }

    private func sampleToken<R: RandomNumberGenerator>(
        from logits: [Float],
        temperature: Float,
        topP: Float,
        using rng: inout R
    ) -> TokenID {
        TokenID(
            NucleusSampler.sample(
                logits: logits,
                temperature: temperature,
                topP: topP,
                using: &rng
            )
        )
    }

    mutating func selectTokenFromNormalizedHidden<R: RandomNumberGenerator>(
        _ hidden: [Float],
        temperature: Float,
        topP: Float = 1.0,
        using rng: inout R
    ) -> TokenID {
        if temperature <= 0 {
            let index = exactClassifierArgmax(hidden)
            return TokenID(index)
        }
        let logits = projectLogits(hidden)
        return sampleToken(from: logits, temperature: temperature, topP: topP, using: &rng)
    }

    mutating func exactClassifierArgmax(_ hidden: [Float]) -> Int {
        precondition(hidden.count == config.dModel)
        if let dumpPath = policies.lmHeadHiddenDumpPath,
           !dumpPath.isEmpty {
            Self.appendLMHeadHiddenDump(hidden, to: dumpPath)
        }
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

    static func precomputeClassifierBlockMaxNorms(
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

    private static func appendLMHeadHiddenDump(_ hidden: [Float], to path: String) {
        var values = hidden
        values.withUnsafeBytes { raw in
            guard let bytes = raw.baseAddress else { return }
            let handle: FileHandle
            do {
                if !FileManager.default.fileExists(atPath: path) {
                    FileManager.default.createFile(atPath: path, contents: nil)
                }
                handle = try FileHandle(forWritingTo: URL(fileURLWithPath: path))
                try handle.seekToEnd()
                try handle.write(contentsOf: UnsafeRawBufferPointer(start: bytes, count: raw.count))
                try handle.close()
            } catch {
                return
            }
        }
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

    static func writeFP32SpatialSlice(
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

    static func readFP32SpatialSlice(
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


    static func zeroSurface(_ surface: IOSurfaceRef) throws {
        guard IOSurfaceLock(surface, [], nil) == kIOReturnSuccess else {
            throw RealModelInferenceError.runtimeFailure("IOSurface lock failed for zero initialization")
        }
        defer { IOSurfaceUnlock(surface, [], nil) }
        memset(IOSurfaceGetBaseAddress(surface), 0, IOSurfaceGetAllocSize(surface))
    }

    static func debugLogHybridCache(
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

    static func debugExpectedGPT2KPrefix(
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

    static func debugExpectedGPT2KPrefixTransposed(
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

    // MARK: - Llama serving session loop

    /// One Trunk's decode-step implementation behind the serving-session seam.
    ///
    /// Steppers own per-run hidden-state and KV-cache state; compiled ANE programs
    /// stay on the host engine and are reached through it. Host flows concretely
    /// because the engine is noncopyable and cannot cross an existential.
    protocol LlamaStepping: AnyObject {
        var contextLimit: Int { get }
        var tracksDecodeProfile: Bool { get }

        func begin(host: inout RealModelInferenceEngine, promptTokens: [TokenID]) throws
        func proposal(host: inout RealModelInferenceEngine) throws -> LlamaDecodeProposal
        func advance(
            host: inout RealModelInferenceEngine,
            consuming token: TokenID,
            generatedCount: Int
        ) throws
        func takePendingTimings() -> HybridDecodeTimingBreakdown?
        func resolveToken(hidden: [Float], temperature: Float, topP: Float) -> TokenID
        func decodeText(_ tokens: [Int]) -> String
        func throwIfCancelled() throws
    }

    /// The one decode-step loop for every llama serving session: sampling policy,
    /// cancellation, EOS, timing, telemetry, streaming, and context limits live here
    /// exactly once; Trunks sit behind ``LlamaStepping``.
    struct LlamaServingSession {
        private let stepper: any LlamaStepping
        private let effectiveMaxTokens: Int
        private let endOfSequenceToken: TokenID?
        private let temperature: Float
        private let topP: Float
        private let onStep: ((GenerationStep) -> Void)?

        init(
            stepper: any LlamaStepping,
            effectiveMaxTokens: Int,
            endOfSequenceToken: TokenID?,
            temperature: Float,
            topP: Float,
            onStep: ((GenerationStep) -> Void)?
        ) {
            self.stepper = stepper
            self.effectiveMaxTokens = effectiveMaxTokens
            self.endOfSequenceToken = endOfSequenceToken
            self.temperature = temperature
            self.topP = topP
            self.onStep = onStep
        }

        func run(
            host: inout RealModelInferenceEngine,
            promptTokens: [TokenID]
        ) throws -> (emission: EmissionCore, decodeProfileReport: String?) {
            try stepper.begin(host: &host, promptTokens: promptTokens)

            let generationStart = DispatchTime.now().uptimeNanoseconds
            let stepperBox = stepper
            var emission = EmissionCore(
                promptTokens: promptTokens,
                capacity: effectiveMaxTokens,
                eos: .fromConfig(endOfSequenceToken.map(Int.init)),
                onStep: onStep,
                decodeText: { stepperBox.decodeText($0) },
                startNanos: generationStart
            )

            var decodeProfileTokens: [HybridDecodeTimingBreakdown] = []
            if stepper.tracksDecodeProfile {
                decodeProfileTokens.reserveCapacity(effectiveMaxTokens)
            }

            while emission.generatedTokenCount < effectiveMaxTokens {
                try stepper.throwIfCancelled()

                let headStart = DispatchTime.now().uptimeNanoseconds
                let proposal = try stepper.proposal(host: &host)
                let nextToken: TokenID
                switch proposal {
                case .selected(let token):
                    nextToken = token
                case .normalizedHidden(let hidden):
                    nextToken = stepper.resolveToken(hidden: hidden, temperature: temperature, topP: topP)
                }

                if stepper.tracksDecodeProfile {
                    var entry = stepper.takePendingTimings() ?? HybridDecodeTimingBreakdown()
                    entry.tLMHead = RealModelInferenceEngine.milliseconds(
                        from: DispatchTime.now().uptimeNanoseconds &- headStart
                    )
                    decodeProfileTokens.append(entry)
                }

                let emissionNow = DispatchTime.now().uptimeNanoseconds
                emission.recordFirstTokenIfFirst(at: emissionNow)

                if emission.terminatesDecoding(nextToken) {
                    emission.recordTerminalToken(nextToken)
                    break
                }

                emission.emit(nextToken, at: emissionNow)

                if emission.generatedTokenCount >= effectiveMaxTokens
                    || emission.allTokensCount >= stepper.contextLimit {
                    break
                }

                try stepper.advance(
                    host: &host,
                    consuming: nextToken,
                    generatedCount: emission.generatedTokenCount
                )
            }

            let decodeProfileReport = decodeProfileTokens.isEmpty
                ? nil
                : HybridDecodeTokenProfile(tokens: decodeProfileTokens).formatReport()
            if let decodeProfileReport {
                fputs(decodeProfileReport + "\n", stderr)
            }

            return (emission, decodeProfileReport)
        }
    }

    // MARK: - Llama serving session steppers

    /// Shared token selection over normalized hidden state.
    ///
    /// One instance per llama serving session; owns the sampling RNG and the
    /// classifier logits scratch so selection behavior matches the engine's own
    /// path draw-for-draw and byte-for-byte.
    final class LlamaTokenSelector {
        private let strategy: ClassifierStrategy
        private let lmHeadWeights: [Float]
        private let lmHeadFP16: [UInt16]?
        private let classifierBlockMaxNorms: [Float]
        private var logitsScratch: [Float]
        private let vocabSize: Int
        private let dim: Int
        private var rng = SystemRandomNumberGenerator()

        init(
            strategy: ClassifierStrategy,
            lmHeadWeights: [Float],
            lmHeadFP16: [UInt16]?,
            classifierBlockMaxNorms: [Float],
            vocab: Int,
            dModel: Int
        ) {
            self.strategy = strategy
            self.lmHeadWeights = lmHeadWeights
            self.lmHeadFP16 = lmHeadFP16
            self.classifierBlockMaxNorms = classifierBlockMaxNorms
            self.logitsScratch = [Float](repeating: 0, count: vocab)
            self.vocabSize = vocab
            self.dim = dModel
        }

        func selectToken(hidden: [Float], temperature: Float, topP: Float) -> TokenID {
            if temperature <= 0 {
                return TokenID(exactClassifierArgmax(hidden))
            }
            let logits = projectLogits(hidden)
            return TokenID(
                NucleusSampler.sample(
                    logits: logits,
                    temperature: temperature,
                    topP: topP,
                    using: &rng
                )
            )
        }

        private func exactClassifierArgmax(_ hidden: [Float]) -> Int {
            precondition(hidden.count == dim)
            if let dumpPath = ProcessInfo.processInfo.environment["ESPRESSO_DUMP_LM_HEAD_HIDDEN"],
               !dumpPath.isEmpty {
                RealModelInferenceEngine.appendLMHeadHiddenDump(hidden, to: dumpPath)
            }
            switch strategy {
            case .ane, .cpuPartitionedFP32:
                return hidden.withUnsafeBufferPointer { hiddenBuffer in
                    lmHeadWeights.withUnsafeBufferPointer { weightBuffer in
                        classifierBlockMaxNorms.withUnsafeBufferPointer { normsBuffer in
                            logitsScratch.withUnsafeMutableBufferPointer { scratchBuffer in
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
                                    vocabSize: vocabSize,
                                    dim: dim,
                                    blockSize: RealModelInferenceEngine.classifierArgmaxBlockSize
                                )
                            }
                        }
                    }
                }
            case .cpuFP16Tiled:
                guard let lmHeadFP16 else {
                    return hidden.withUnsafeBufferPointer { hiddenBuffer in
                        lmHeadWeights.withUnsafeBufferPointer { weightBuffer in
                            classifierBlockMaxNorms.withUnsafeBufferPointer { normsBuffer in
                                logitsScratch.withUnsafeMutableBufferPointer { scratchBuffer in
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
                                        vocabSize: vocabSize,
                                        dim: dim,
                                        blockSize: RealModelInferenceEngine.classifierArgmaxBlockSize
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
                            vocabSize: vocabSize,
                            dim: dim
                        )
                    }
                }
            }
        }

        private func projectLogits(_ hidden: [Float]) -> [Float] {
            precondition(hidden.count == dim)
            var logits = [Float](repeating: 0, count: vocabSize)
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
                            vDSP_Length(vocabSize),
                            1,
                            vDSP_Length(dim)
                        )
                    }
                }
            }
            return logits
        }
    }

    private func makeLlamaTokenSelector() -> LlamaTokenSelector {
        let fp16Head: [UInt16]?
        if case let .llama(assets) = assets {
            fp16Head = assets.lmHeadFP16
        } else {
            fp16Head = nil
        }
        return LlamaTokenSelector(
            strategy: classifierStrategy,
            lmHeadWeights: lmHeadWeights,
            lmHeadFP16: fp16Head,
            classifierBlockMaxNorms: classifierBlockMaxNorms,
            vocab: config.vocab,
            dModel: config.dModel
        )
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
