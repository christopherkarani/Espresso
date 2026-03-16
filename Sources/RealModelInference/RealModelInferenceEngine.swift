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
    public let tokens: [UInt16]
    public let promptTokens: [UInt16]
    public let tokensPerSecond: Double
    public let compileTimeMs: Double
    public let firstTokenLatencyMs: Double

    public init(
        text: String,
        tokens: [UInt16],
        promptTokens: [UInt16],
        tokensPerSecond: Double,
        compileTimeMs: Double,
        firstTokenLatencyMs: Double
    ) {
        self.text = text
        self.tokens = tokens
        self.promptTokens = promptTokens
        self.tokensPerSecond = tokensPerSecond
        self.compileTimeMs = compileTimeMs
        self.firstTokenLatencyMs = firstTokenLatencyMs
    }
}

public struct GenerationStep: Sendable {
    public let token: UInt16
    public let generatedTokens: [UInt16]
    public let text: String
    public let tokenLatencyMs: Double
    public let elapsedMs: Double
    public let firstTokenLatencyMs: Double
    public let tokensPerSecond: Double

    public init(
        token: UInt16,
        generatedTokens: [UInt16],
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

    struct AttentionTestingOutputs {
        let hidden: [Float]
        let kCache: [Float]
        let vCache: [Float]
    }

    private enum LoadedTokenizer {
        case gpt2(GPT2BPETokenizer)
        case sentencePiece(SentencePieceTokenizer)

        func encode(_ text: String) -> [Int] {
            switch self {
            case let .gpt2(tokenizer):
                return tokenizer.encode(text)
            case let .sentencePiece(tokenizer):
                return tokenizer.encode(text)
            }
        }

        func decode(_ tokens: [Int]) -> String {
            switch self {
            case let .gpt2(tokenizer):
                return tokenizer.decode(tokens)
            case let .sentencePiece(tokenizer):
                return tokenizer.decode(tokens)
            }
        }
    }

    private struct CompiledLayer: ~Copyable {
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

    private struct CompiledHead: ~Copyable {
        let kernel: ANEKernel
        let inputSurface: IOSurfaceRef
        let outputSurface: IOSurfaceRef

        init(kernel: consuming ANEKernel, inputSurface: IOSurfaceRef, outputSurface: IOSurfaceRef) {
            self.kernel = kernel
            self.inputSurface = inputSurface
            self.outputSurface = outputSurface
        }
    }

    private struct CompiledClassifier: ~Copyable {
        let kernel: ANEKernel
        let inputSurface: IOSurfaceRef
        let outputSurface: IOSurfaceRef
        let maxValueSurface: IOSurfaceRef

        init(
            kernel: consuming ANEKernel,
            inputSurface: IOSurfaceRef,
            outputSurface: IOSurfaceRef,
            maxValueSurface: IOSurfaceRef
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

            if layers.count > 1 {
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

        mutating func selectGreedyToken(vocab: Int) throws -> UInt16 {
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
                timings: &timings
            )
        }
    }

    private static let gpt2EOSToken: UInt16 = 50_256
    private static let speculativeRuntimeCacheLimit = 4

    private let config: MultiModelConfig
    private let weightDirURL: URL
    private let tokenizer: LoadedTokenizer
    private let gpt2Assets: GPT2TopLevelAssets
    private var compiledBucket: Int
    private var compiledLayers: LayerStorage<CompiledLayer>
    private var firstLayerInputSurface: IOSurfaceRef?
    private var compiledHead: LayerStorage<CompiledHead>
    private var compiledHybridBucket: Int
    private var compiledHybridLayers: LayerStorage<HybridDecodeKernelSet>
    private var compiledHybridSurfaceHandles: [HybridDecodeSurfaceHandles]
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

    private init(
        config: MultiModelConfig,
        weightDirURL: URL,
        tokenizer: LoadedTokenizer,
        gpt2Assets: GPT2TopLevelAssets
    ) {
        let classifierBlockMaxNorms = gpt2Assets.lmHead.withUnsafeBufferPointer { weightBuffer in
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
        self.gpt2Assets = gpt2Assets
        self.compiledBucket = 0
        self.compiledLayers = Self.emptyStorage(CompiledLayer.self)
        self.firstLayerInputSurface = nil
        self.compiledHead = Self.emptyStorage(CompiledHead.self)
        self.compiledHybridBucket = 0
        self.compiledHybridLayers = Self.emptyStorage(HybridDecodeKernelSet.self)
        self.compiledHybridSurfaceHandles = []
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

        guard config.architecture == .gpt2 else {
            throw RealModelInferenceError.unsupportedArchitecture(
                "Architecture \(architectureName(config.architecture)) is not supported yet: llama-family models require RoPE in the ANE graph."
            )
        }

        let tokenizer = try loadTokenizer(config: config, tokenizerDirURL: tokenizerDirURL)

        let topLevelPaths = try resolveTopLevelWeightPaths(config: config, weightDir: weightDir)
        let tokenEmbedding = try loadWeightTable(
            at: topLevelPaths.tokenEmbedding,
            expectedCount: config.vocab * config.dModel
        )
        let positionEmbedding = try loadWeightTable(
            at: topLevelPaths.positionEmbedding,
            expectedCount: config.maxSeq * config.dModel
        )
        let finalNormGamma = try loadWeightTable(
            at: topLevelPaths.finalNormGamma,
            expectedCount: config.dModel
        )
        let finalNormBeta = try loadWeightTable(
            at: topLevelPaths.finalNormBeta,
            expectedCount: config.dModel
        )
        let lmHead = try loadWeightTable(
            at: topLevelPaths.lmHead,
            expectedCount: config.vocab * config.dModel
        )

        let assets = GPT2TopLevelAssets(
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
        )
        return RealModelInferenceEngine(
            config: config,
            weightDirURL: weightDirURL,
            tokenizer: tokenizer,
            gpt2Assets: assets
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
        var generatedTokens: [UInt16] = []
        generatedTokens.reserveCapacity(effectiveMaxTokens)

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
            tokensPerSecond: tokensPerSecond,
            compileTimeMs: compileTimeMs,
            firstTokenLatencyMs: firstTokenLatencyMs
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
            throw RealModelInferenceError.unsupportedArchitecture(
                "Architecture llama is not supported yet: llama-family models require RoPE in the ANE graph."
            )
        }
    }

    static func compileAndEvalSingleLayerForTesting(
        config: MultiModelConfig,
        weightDir: String,
        layer: Int,
        spatial: Int,
        input: [Float]
    ) throws -> [Float] {
        let weightDirURL = URL(fileURLWithPath: weightDir, isDirectory: true)
        try validateDirectory(weightDirURL)
        let compiled = try compileLayer(
            layerIndex: layer,
            config: config,
            weightDirURL: weightDirURL,
            spatial: spatial
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
        input: [Float]
    ) throws -> [Float] {
        try compileAndEvalSingleLayerAttentionOutputsForTesting(
            config: config,
            weightDir: weightDir,
            layer: layer,
            spatial: spatial,
            input: input
        ).hidden
    }

    static func compileAndEvalSingleLayerAttentionOutputsForTesting(
        config: MultiModelConfig,
        weightDir: String,
        layer: Int,
        spatial: Int,
        input: [Float]
    ) throws -> AttentionTestingOutputs {
        let weightDirURL = URL(fileURLWithPath: weightDir, isDirectory: true)
        try validateDirectory(weightDirURL)
        let compiled = try compileLayer(
            layerIndex: layer,
            config: config,
            weightDirURL: weightDirURL,
            spatial: spatial
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

    static func composeEmbeddingInputForTesting(
        config: MultiModelConfig,
        weightDir: String,
        tokens: [UInt16]
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

        let topLevelPaths = try resolveTopLevelWeightPaths(config: config, weightDir: weightDir)
        let tokenEmbedding = try loadWeightTable(
            at: topLevelPaths.tokenEmbedding,
            expectedCount: config.vocab * config.dModel
        )
        let positionEmbedding = try loadWeightTable(
            at: topLevelPaths.positionEmbedding,
            expectedCount: config.maxSeq * config.dModel
        )

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
        tokens: [UInt16]
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

        let topLevelPaths = try resolveTopLevelWeightPaths(config: config, weightDir: weightDir)
        let tokenEmbedding = try loadWeightTable(
            at: topLevelPaths.tokenEmbedding,
            expectedCount: config.vocab * config.dModel
        )
        let positionEmbedding = try loadWeightTable(
            at: topLevelPaths.positionEmbedding,
            expectedCount: config.maxSeq * config.dModel
        )
        let paths = LayerWeightPaths.forLayer(layer, config: config, blobDir: weightDirURL.path)
        let weights = try loadHybridLayerWeights(config: config, paths: paths)
        let maxSeq = max(tokens.count, 1)
        let kernels = try LayerStorage<HybridDecodeKernelSet>(count: 1, throwingInitializer: { _ in
            try HybridDecodeKernelSet(weights: weights, maxSeq: maxSeq)
        })
        let handles = [try HybridDecodeSurfaceHandles(kernels: kernels[0], logicalMaxSeq: maxSeq)]
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
                timings: &timings
            )
        }

        return xCur.withUnsafeBufferPointer { Array($0) }
    }

    static func evalHybridSingleLayerAttentionForTesting(
        config: MultiModelConfig,
        weightDir: String,
        layer: Int,
        tokens: [UInt16]
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
        tokens: [UInt16]
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
        let tokenEmbedding = try loadWeightTable(
            at: topLevelPaths.tokenEmbedding,
            expectedCount: config.vocab * config.dModel
        )
        let positionEmbedding = try loadWeightTable(
            at: topLevelPaths.positionEmbedding,
            expectedCount: config.maxSeq * config.dModel
        )
        let paths = LayerWeightPaths.forLayer(layer, config: config, blobDir: weightDirURL.path)
        let weights = try loadHybridLayerWeights(config: config, paths: paths)
        let maxSeq = max(tokens.count, 1)
        let kernels = try LayerStorage<HybridDecodeKernelSet>(count: 1, throwingInitializer: { _ in
            try HybridDecodeKernelSet(weights: weights, maxSeq: maxSeq)
        })
        let handles = [try HybridDecodeSurfaceHandles(kernels: kernels[0], logicalMaxSeq: maxSeq)]
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
        var kCache = [Float](repeating: 0, count: config.dModel * maxSeq)
        var vCache = [Float](repeating: 0, count: config.dModel * maxSeq)
        kCache.withUnsafeMutableBufferPointer { buffer in
            SurfaceIO.readFP16(
                from: handles[0].kCacheFull,
                into: buffer,
                channelOffset: 0,
                channels: config.dModel,
                spatial: maxSeq
            )
        }
        vCache.withUnsafeMutableBufferPointer { buffer in
            SurfaceIO.readFP16(
                from: handles[0].vCacheFull,
                into: buffer,
                channelOffset: 0,
                channels: config.dModel,
                spatial: maxSeq
            )
        }
        return AttentionTestingOutputs(hidden: hidden, kCache: kCache, vCache: vCache)
    }

    static func compileHeadForTesting(
        config: MultiModelConfig,
        weightDir: String
    ) throws {
        let weightDirURL = URL(fileURLWithPath: weightDir, isDirectory: true)
        try validateDirectory(weightDirURL)
        let topLevelPaths = try resolveTopLevelWeightPaths(config: config, weightDir: weightDir)
        let finalNormGamma = try loadWeightTable(at: topLevelPaths.finalNormGamma, expectedCount: config.dModel)
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
            if newLayers.count > 1 {
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
            compiledHybridBucket = bucket
            didCompile = true
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

        return didCompile
    }

    private mutating func generateIncrementalHybrid(
        promptTokens: [UInt16],
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
        let useANEGreedyHead =
            temperature == 0 &&
            compiledHybridGreedyNorm.count == 1 &&
            compiledHybridGreedyClassifier.count == 1

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
                    readFinalOutputIntoXCur: !useANEGreedyHead,
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
        var generatedTokens: [UInt16] = []
        generatedTokens.reserveCapacity(effectiveMaxTokens)

        let generationStart = DispatchTime.now().uptimeNanoseconds
        var emissionStart = generationStart
        var firstTokenLatencyMs = 0.0
        var firstTokenRecorded = false
        var rng = SystemRandomNumberGenerator()
        var normalized = [Float](repeating: 0, count: config.dModel)
        let headSpatial = compiledHybridHeadSpatial

        while generatedTokens.count < effectiveMaxTokens {
            let nextToken: UInt16
            if useANEGreedyHead {
                do {
                    try compiledHybridGreedyNorm[0].kernel.eval()
                    try compiledHybridGreedyClassifier[0].kernel.eval()
                    let argmax = try SurfaceIO.argmaxFP16SpatialSliceWithHint(
                        from: compiledHybridGreedyClassifier[0].outputSurface,
                        channelOffset: 0,
                        spatialIndex: 0,
                        spatial: headSpatial,
                        channels: config.vocab,
                        hintSurface: compiledHybridGreedyClassifier[0].maxValueSurface,
                        hintSpatialIndex: 0,
                        hintSpatial: headSpatial
                    )
                    guard let token = UInt16(exactly: argmax.index) else {
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
                    readFinalOutputIntoXCur: !useANEGreedyHead,
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
            tokensPerSecond: tokensPerSecond,
            compileTimeMs: compileTimeMs,
            firstTokenLatencyMs: firstTokenLatencyMs
        )
    }

    private mutating func generateIncrementalHybridSpeculative(
        promptTokens: [UInt16],
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
        var generatedTokens: [UInt16] = []
        generatedTokens.reserveCapacity(effectiveMaxTokens)

        let generationStart = DispatchTime.now().uptimeNanoseconds
        var emissionStart = generationStart
        var firstTokenLatencyMs = 0.0
        var firstTokenRecorded = false

        func emitToken(_ token: UInt16, at emissionNow: UInt64) {
            generatedTokens.append(token)
            allTokens.append(token)
            let elapsedMs = Self.milliseconds(from: emissionNow - generationStart)
            let tokenLatencyMs = Self.milliseconds(from: emissionNow - emissionStart)
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
            let proposedToken0: UInt16
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

            let proposedToken1: UInt16
            do {
                proposedToken1 = try cachedRuntimePair.draftRuntime.selectGreedyToken(vocab: config.vocab)
            } catch {
                throw RealModelInferenceError.runtimeFailure(
                    "Hybrid speculative draft proposal-1 failed at generated token \(generatedTokens.count): \(error)"
                )
            }

            let exactToken0: UInt16
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

            let exactToken1: UInt16
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

    private func encodePrompt(_ prompt: String) throws -> [UInt16] {
        let rawTokens = tokenizer.encode(prompt)
        guard !rawTokens.isEmpty else {
            throw RealModelInferenceError.invalidPrompt("Prompt produced no tokens")
        }
        var tokens: [UInt16] = []
        tokens.reserveCapacity(rawTokens.count)
        for token in rawTokens {
            guard token >= 0, token <= Int(UInt16.max) else {
                throw RealModelInferenceError.invalidPrompt("Token \(token) does not fit UInt16")
            }
            tokens.append(UInt16(token))
        }
        return tokens
    }

    private func composeEmbeddingInput(tokens: [UInt16], spatial: Int) -> [Float] {
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
        token: UInt16,
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

    private static func compileHybridLayers(
        config: MultiModelConfig,
        weightDirURL: URL,
        sourceLayerRange: Range<Int>? = nil,
        maxSeq: Int
    ) throws -> LayerStorage<HybridDecodeKernelSet> {
        let layerRange = sourceLayerRange ?? (0..<config.nLayer)
        return try LayerStorage<HybridDecodeKernelSet>(count: layerRange.count, throwingInitializer: { localLayerIndex in
            let layerIndex = layerRange.lowerBound + localLayerIndex
            let paths = LayerWeightPaths.forLayer(layerIndex, config: config, blobDir: weightDirURL.path)
            let weights = try loadHybridLayerWeights(config: config, paths: paths)
            do {
                return try HybridDecodeKernelSet(weights: weights, maxSeq: maxSeq)
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
        bucket: Int
    ) throws -> LayerStorage<CompiledLayer> {
        try LayerStorage<CompiledLayer>(count: config.nLayer, throwingInitializer: { layerIndex in
            try compileLayer(
                layerIndex: layerIndex,
                config: config,
                weightDirURL: weightDirURL,
                spatial: bucket
            )
        })
    }

    private static func loadHybridLayerWeights(
        config: MultiModelConfig,
        paths: LayerWeightPaths
    ) throws -> LayerWeights {
        let weights = LayerWeights(
            architecture: .gpt2,
            dim: config.dModel,
            hiddenDim: config.hiddenDim
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

    private enum LayerBlockKind: String {
        case attention = "attn"
        case ffn
    }

    private static func compileLayer(
        layerIndex: Int,
        config: MultiModelConfig,
        weightDirURL: URL,
        spatial: Int
    ) throws -> CompiledLayer {
        let paths = LayerWeightPaths.forLayer(layerIndex, config: config, blobDir: weightDirURL.path)
        let ioBytes = try ANEShape(channels: config.dModel, spatial: spatial).byteSize(for: .fp32)

        let attentionGraph = buildGPT2AttentionBlockGraph(
            layerIndex: layerIndex,
            config: config,
            paths: paths,
            spatial: spatial
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
            spatial: spatial
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
            spatial: spatial
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
            spatial: spatial
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
        spatial: Int
    ) throws -> ANEKernel {
        var optimized = graph
        ANEOptimizationPipeline.optimize(&optimized)
        let mil = rewriteMILWeightPaths(ANECodegen.emit(optimized), rootDir: weightDirURL)
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
                "Architecture llama is not supported yet: llama-family models require RoPE in the ANE graph."
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
                "Architecture llama is not supported yet: llama-family models require RoPE in the ANE graph."
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
        outputDType: ANEDType = .fp32
    ) throws -> CompiledHead {
        var graph = buildGPT2HeadGraph(
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

    private static func buildGPT2AttentionBlockGraph(
        layerIndex: Int,
        config: MultiModelConfig,
        paths: LayerWeightPaths,
        spatial: Int
    ) -> ANEGraph {
        var graph = ANEGraph()
        let prefix = "layer\(layerIndex)"
        let input = try! graph.input(
            "x",
            dtype: .fp32,
            shape: try! ANEShape(channels: config.dModel, spatial: spatial)
        )
        let x16 = try! graph.cast("\(prefix)_x16", input: input, to: .fp16)
        let ln1 = try! graph.layerNorm128(
            "\(prefix)_ln1",
            input: x16,
            dim: config.dModel,
            spatial: spatial,
            eps: config.normEps,
            gammaPath: paths.rmsAtt,
            betaPath: replacingGammaSuffix(in: paths.rmsAtt)
        )
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
        spatial: Int
    ) -> ANEGraph {
        var graph = ANEGraph()
        let prefix = "layer\(layerIndex)"
        let input = try! graph.input(
            "x",
            dtype: .fp32,
            shape: try! ANEShape(channels: config.dModel, spatial: spatial)
        )
        let x16 = try! graph.cast("\(prefix)_x16", input: input, to: .fp16)
        let ln2 = try! graph.layerNorm128(
            "\(prefix)_ln2",
            input: x16,
            dim: config.dModel,
            spatial: spatial,
            eps: config.normEps,
            gammaPath: paths.rmsFfn,
            betaPath: replacingGammaSuffix(in: paths.rmsFfn)
        )
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
        outputDType: ANEDType = .fp32
    ) -> ANEGraph {
        var graph = ANEGraph()
        let input = try! graph.input(
            "x",
            dtype: inputDType,
            shape: try! ANEShape(channels: config.dModel, spatial: spatial)
        )
        let x16 = inputDType == .fp16 ? input : try! graph.cast("final_ln_x16", input: input, to: .fp16)
        let norm = try! graph.layerNorm128(
            "final_ln",
            input: x16,
            dim: config.dModel,
            spatial: spatial,
            eps: config.normEps,
            gammaPath: assets.finalNormGammaPath,
            betaPath: assets.finalNormBetaPath
        )
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
            let candidates = ["tokenizer.model", "tokenizer.bin"]
            for candidate in candidates {
                let url = tokenizerDirURL.appendingPathComponent(candidate)
                if FileManager.default.fileExists(atPath: url.path) {
                    do {
                        return .sentencePiece(try SentencePieceTokenizer(modelURL: url))
                    } catch {
                        throw RealModelInferenceError.runtimeFailure("Failed to load SentencePiece tokenizer: \(error)")
                    }
                }
            }
            throw RealModelInferenceError.missingPath(
                tokenizerDirURL.appendingPathComponent("tokenizer.model").path
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
        guard config.dModel == config.nHead * config.headDim else {
            throw RealModelInferenceError.invalidConfig("dModel must equal nHead * headDim")
        }
        guard config.vocab <= Int(UInt16.max) else {
            throw RealModelInferenceError.invalidConfig("vocab \(config.vocab) exceeds UInt16 token capacity")
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

    private static func loadWeightTable(at path: String, expectedCount: Int) throws -> [Float] {
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
    ) -> UInt16 {
        if temperature <= 0 {
            let index = logits.enumerated().max(by: { $0.element < $1.element })?.offset ?? 0
            return UInt16(index)
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
            return UInt16(index)
        }

        var threshold = Double.random(in: 0..<total, using: &rng)
        for index in scaled.indices {
            threshold -= scaled[index]
            if threshold <= 0 {
                return UInt16(index)
            }
        }
        return UInt16(max(0, scaled.count - 1))
    }

    private mutating func selectTokenFromNormalizedHidden<R: RandomNumberGenerator>(
        _ hidden: [Float],
        temperature: Float,
        using rng: inout R
    ) -> UInt16 {
        if temperature <= 0 {
            let index = exactClassifierArgmax(hidden)
            return UInt16(index)
        }
        let logits = projectLogits(hidden)
        return sampleToken(from: logits, temperature: temperature, using: &rng)
    }

    private mutating func exactClassifierArgmax(_ hidden: [Float]) -> Int {
        precondition(hidden.count == config.dModel)
        let blockSize = Self.classifierArgmaxBlockSize
        return hidden.withUnsafeBufferPointer { hiddenBuffer in
            gpt2Assets.lmHead.withUnsafeBufferPointer { weightBuffer in
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
    }

    private func projectLogits(_ hidden: [Float]) -> [Float] {
        precondition(hidden.count == config.dModel)
        var logits = [Float](repeating: 0, count: config.vocab)
        logits.withUnsafeMutableBufferPointer { logitsBuffer in
            gpt2Assets.lmHead.withUnsafeBufferPointer { weightBuffer in
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

            cblas_sgemm(
                CblasRowMajor,
                CblasNoTrans,
                CblasNoTrans,
                Int32(blockCount),
                1,
                Int32(dim),
                1.0,
                classifier.advanced(by: blockStart * dim),
                Int32(dim),
                input,
                1,
                0.0,
                logitsScratch,
                1
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
        tokens: [UInt16],
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
                output[channel * tokens.count + tokenIndex] =
                    tokenEmbedding[tokenBase + channel] +
                    positionEmbedding[positionBase + channel]
            }
        }
        return output
    }

    private static func writeTestingIncrementalEmbedding(
        config: MultiModelConfig,
        token: UInt16,
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
                dst[channel] =
                    tokenEmbedding[tokenBase + channel] +
                    positionEmbedding[positionBase + channel]
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
    ) throws -> UInt16 {
        do {
            try norm.kernel.eval()
            try classifier.kernel.eval()
            let argmax = try SurfaceIO.argmaxFP16SpatialSliceWithHint(
                from: classifier.outputSurface,
                channelOffset: 0,
                spatialIndex: 0,
                spatial: headSpatial,
                channels: vocab,
                hintSurface: classifier.maxValueSurface,
                hintSpatialIndex: 0,
                hintSpatial: headSpatial
            )
            guard let token = UInt16(exactly: argmax.index) else {
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
