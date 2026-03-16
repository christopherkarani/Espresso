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

    private static let gpt2EOSToken: UInt16 = 50_256

    private let config: MultiModelConfig
    private let weightDirURL: URL
    private let tokenizer: LoadedTokenizer
    private let gpt2Assets: GPT2TopLevelAssets
    private var compiledBucket: Int
    private var compiledLayers: LayerStorage<CompiledLayer>
    private var firstLayerInputSurface: IOSurfaceRef?
    private var compiledHead: LayerStorage<CompiledHead>

    private init(
        config: MultiModelConfig,
        weightDirURL: URL,
        tokenizer: LoadedTokenizer,
        gpt2Assets: GPT2TopLevelAssets
    ) {
        self.config = config
        self.weightDirURL = weightDirURL
        self.tokenizer = tokenizer
        self.gpt2Assets = gpt2Assets
        self.compiledBucket = 0
        self.compiledLayers = Self.emptyStorage(CompiledLayer.self)
        self.firstLayerInputSurface = nil
        self.compiledHead = Self.emptyStorage(CompiledHead.self)
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
        let compileDidRun = try ensureCompiled(bucket: bucket)
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
            let logits = projectLogits(lastHidden)
            let nextToken = sampleToken(from: logits, temperature: temperature, using: &rng)

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
        spatial: Int
    ) throws -> CompiledHead {
        var graph = buildGPT2HeadGraph(config: config, assets: assets, spatial: spatial)
        ANEOptimizationPipeline.optimize(&graph)
        let mil = rewriteMILWeightPaths(ANECodegen.emit(graph), rootDir: weightDirURL)
        let inputBytes = try ANEShape(channels: config.dModel, spatial: spatial).byteSize(for: .fp32)
        let outputBytes = try ANEShape(channels: config.dModel, spatial: spatial).byteSize(for: .fp32)
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
        spatial: Int
    ) -> ANEGraph {
        var graph = ANEGraph()
        let input = try! graph.input(
            "x",
            dtype: .fp32,
            shape: try! ANEShape(channels: config.dModel, spatial: spatial)
        )
        let x16 = try! graph.cast("final_ln_x16", input: input, to: .fp16)
        let norm = try! graph.layerNorm128(
            "final_ln",
            input: x16,
            dim: config.dModel,
            spatial: spatial,
            eps: config.normEps,
            gammaPath: assets.finalNormGammaPath,
            betaPath: assets.finalNormBetaPath
        )
        let hidden = try! graph.cast("hidden", input: norm, to: .fp32)
        _ = try! graph.output(hidden, name: "hidden")
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
