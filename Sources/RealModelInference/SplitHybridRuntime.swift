import ANERuntime
import ANETypes
import CPUOps
import Espresso
import IOSurface
import Accelerate
import Darwin
import Foundation
import ModelSupport

// Split-hybrid trunk runtime (extracted from RealModelInferenceEngine).
//
// ANE QKV -> host attention -> ANE FFN, shared by GPT-2 and llama sessions,
// plus the speculative fast path that rides on it. The baseline (exact-CPU-
// routed ANE eval) ensure lives here too because the hybrid loops fall back
// to it. State stays in flat engine fields; only behavior moved.

extension RealModelInferenceEngine {
    mutating func ensureCompiled(bucket: Int) throws -> Bool {
        switch baselineReadiness {
        case .compiled(let runtime) where runtime.bucket >= bucket:
            return false
        case .compiled, .notCompiled:
            break
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
                spatial: bucket,
                environment: policies.environment
            )
        })
        do {
            try newHead[0].kernel.rebindInput(at: 0, to: newLayers[newLayers.count - 1].outputSurface)
        } catch {
            throw RealModelInferenceError.runtimeFailure("Failed to chain final norm input: \(error)")
        }
        compiledLayers = newLayers
        compiledHead = newHead
        baselineReadiness = .compiled(BaselineCompiledRuntime(bucket: bucket, inputSurface: newInputSurface))
        return true
    }

    /// Compiles split-hybrid decode programs when the resident set covers less context.
    ///
    /// One parameterized implementation serves both model families: the former
    /// GPT-2/llama ensure twins differed only in surface-handle geometry,
    /// output-head compilation, and error prefixes. Split-hybrid readiness
    /// transitions to `.compiled` after the output-head section, before the
    /// family-specific greedy-classifier section — a greedy-head compile failure
    /// must not unready the trunk's decode programs, matching the former flag
    /// semantics where such failures still served hybrid decode via CPU logits.
    mutating func ensureHybridCompiled(bucket: Int) throws -> Bool {
        var didCompile = false

        // Both flavors share every program up to the output head; only the
        // surface-handle geometry and head compilation differ.
        let isLlama = config.architecture == .llama
        let surfaceDim = isLlama ? config.dModel : ModelConfig.dim
        let surfaceQDim: Int? = isLlama ? config.attentionDim : nil
        let surfaceKVDim: Int? = isLlama ? config.nKVHead * config.headDim : nil

        if splitHybridLayerBucket < bucket {
            let newLayers = try Self.compileHybridLayers(
                config: config,
                weightDirURL: weightDirURL,
                maxSeq: bucket,
                environment: policies.environment
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
                            dim: surfaceDim,
                            qDim: surfaceQDim,
                            kvDim: surfaceKVDim
                        )
                    )
                } catch {
                    throw RealModelInferenceError.runtimeFailure(
                        "\(isLlama ? "Llama hybrid" : "Hybrid") decode surfaces unavailable for layer \(layerIndex): \(error)"
                    )
                }
            }
            if newLayers.count > 1,
               Self.usesHybridLayerInputRebinding(
                   architecture: config.architecture,
                   environment: policies.environment
               ) {
                for layerIndex in 1..<newLayers.count {
                    do {
                        try newLayers[layerIndex].decodeQKVOnly.rebindInput(
                            at: 0,
                            to: newSurfaceHandles[layerIndex - 1].ffnOut
                        )
                    } catch {
                        throw RealModelInferenceError.runtimeFailure(
                            "\(isLlama ? "Llama hybrid" : "Hybrid") decode chaining unavailable for layer \(layerIndex): \(error)"
                        )
                    }
                }
            }

            compiledHybridLayers = newLayers
            compiledHybridSurfaceHandles = newSurfaceHandles
            compiledHybridLlamaQKNormWeights = newQKNormWeights
            splitHybridLayerBucket = bucket
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
                throw RealModelInferenceError.runtimeFailure(
                    "\(isLlama ? "Llama hybrid" : "Hybrid") Metal attention initialization failed: \(error)"
                )
            }
            didCompile = true
        }

        let hybridHeadSpatial = Self.incrementalHeadSpatial(channels: config.dModel)
        if compiledHybridHead.count != 1 || compiledHybridHeadSpatial != hybridHeadSpatial {
            compiledHybridHead = try LayerStorage<CompiledHead>(count: 1, throwingInitializer: { _ in
                if isLlama {
                    return try Self.compileLlamaHead(
                        config: config,
                        weightDirURL: weightDirURL,
                        assets: llamaAssets,
                        spatial: hybridHeadSpatial
                    )
                }
                return try Self.compileHead(
                    config: config,
                    weightDirURL: weightDirURL,
                    assets: gpt2Assets,
                    spatial: hybridHeadSpatial,
                    environment: policies.environment
                )
            })
            compiledHybridHeadSpatial = hybridHeadSpatial
            try Self.zeroSurface(compiledHybridHead[0].inputSurface)
            didCompile = true
        }

        // Readiness transition: every program the split-hybrid decode loop
        // requires is now resident, independent of the greedy head below.
        if let runtime = SplitHybridCompiledRuntime(
            bucket: max(bucket, splitHybridLayerBucket),
            layerCount: compiledHybridLayers.count,
            surfaceHandleCount: compiledHybridSurfaceHandles.count,
            expectedLayerCount: config.nLayer,
            headCount: compiledHybridHead.count,
            qKNormCount: isLlama ? compiledHybridLlamaQKNormWeights.count : nil,
            headSpatial: compiledHybridHeadSpatial
        ) {
            splitHybridReadiness = .compiled(runtime)
        } else {
            splitHybridReadiness = .notCompiled
        }

        if classifierStrategy.usesANEClassifier {
            if isLlama {
                switch hybridGreedyHeadMode() {
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
            } else {
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
                            outputDType: .fp16,
                            environment: policies.environment
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
        }

        if isLlama,
           compiledHybridGreedyNorm.count == 1,
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

    mutating func generateIncrementalHybrid(
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
        switch splitHybridReadiness {
        case .compiled:
            break
        case .notCompiled:
            throw RealModelInferenceError.runtimeFailure("Hybrid decode state is unavailable")
        }

        try ForwardPass.initializeHybridDecodeCaches(
            surfaceHandles: compiledHybridSurfaceHandles,
            dim: config.dModel
        )

        // Pre-create cached Metal bindings for all layers (GPT-2 path)
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

        if policies.debugHybridCacheDumps,
           let firstHandles = compiledHybridSurfaceHandles.first {
            fputs(
                "[hybrid-surface] qkvIn_row=\(IOSurfaceGetBytesPerRow(firstHandles.qkvIn)) qOut_row=\(IOSurfaceGetBytesPerRow(firstHandles.qOut)) ffnIn_row=\(IOSurfaceGetBytesPerRow(firstHandles.ffnIn)) ffnOut_row=\(IOSurfaceGetBytesPerRow(firstHandles.ffnOut)) laneSpatial=\(firstHandles.laneSpatial) maxSeq=\(firstHandles.maxSeq)\n",
                stderr
            )
        }
        let shouldDebugHybridCache = policies.debugHybridCacheDumps

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
                        environment: policies.environment
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
        var normalized = [Float](repeating: 0, count: config.dModel)
        let headSpatial = compiledHybridHeadSpatial

        while emission.generatedTokenCount < effectiveMaxTokens {
            try Self.throwIfCancelled(isCancelled)
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
                    topP: topP,
                    using: &rng
                )
            }
            let emissionNow = DispatchTime.now().uptimeNanoseconds
            emission.recordFirstTokenIfFirst(at: emissionNow)

            if emission.terminatesDecoding(nextToken) {
                break
            }

            emission.emit(nextToken, at: emissionNow)

            if emission.generatedTokenCount >= effectiveMaxTokens || emission.allTokensCount >= config.maxSeq {
                break
            }

            try writeIncrementalEmbedding(token: nextToken, position: emission.allTokensCount - 1, into: xCur)
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
                        environment: policies.environment
                    ),
                    readFinalOutputIntoXCur: !useANEGreedyHead,
                    cachedBindings: cachedBindings,
                    timings: &timings
                )
            } catch {
                throw RealModelInferenceError.runtimeFailure(
                    "Hybrid decode failed at generated token \(emission.generatedTokenCount - 1): \(error)"
                )
            }
        }

        return emission.makeResult(
            compileTimeMs: compileTimeMs,
            exactHeadBackend: classifierStrategy.exactHeadBackendLabel,
            cachedBindingsEnabled: false
        )
    }

    mutating func generateIncrementalHybridSpeculative(
        promptTokens: [TokenID],
        effectiveMaxTokens: Int,
        compileTimeMs: Double,
        metalAttention: MetalAttentionKernel,
        cachedRuntimePair: CachedSpeculativeRuntimePair,
        onStep: ((GenerationStep) -> Void)?
    ) throws -> GenerationResult {
        try cachedRuntimePair.resetAll(dim: config.dModel)

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

        while emission.generatedTokenCount < effectiveMaxTokens {
            let checkpoint = try cachedRuntimePair.draftRuntime.captureCheckpoint(dim: config.dModel)
            let proposedToken0: TokenID
            do {
                proposedToken0 = try cachedRuntimePair.draftRuntime.selectGreedyToken(vocab: config.vocab)
                try writeIncrementalEmbedding(token: proposedToken0, position: emission.allTokensCount, into: xCur)
                try cachedRuntimePair.draftRuntime.advanceFromBuffer(
                    xCur,
                    metalAttention: metalAttention,
                    dim: config.dModel
                )
            } catch {
                throw RealModelInferenceError.runtimeFailure(
                    "Hybrid speculative draft proposal-0 failed at generated token \(emission.generatedTokenCount): \(error)"
                )
            }

            let proposedToken1: TokenID
            do {
                proposedToken1 = try cachedRuntimePair.draftRuntime.selectGreedyToken(vocab: config.vocab)
            } catch {
                throw RealModelInferenceError.runtimeFailure(
                    "Hybrid speculative draft proposal-1 failed at generated token \(emission.generatedTokenCount): \(error)"
                )
            }

            let exactToken0: TokenID
            do {
                exactToken0 = try cachedRuntimePair.verifierRuntime.selectGreedyToken(vocab: config.vocab)
            } catch {
                throw RealModelInferenceError.runtimeFailure(
                    "Hybrid speculative verifier token-0 failed at generated token \(emission.generatedTokenCount): \(error)"
                )
            }
            if emission.terminatesDecoding(exactToken0) {
                break
            }

            if exactToken0 != proposedToken0 {
                do {
                    try cachedRuntimePair.draftRuntime.rollback(
                        to: checkpoint,
                        mutatedTokenCount: 1,
                        dim: config.dModel
                    )
                    try writeIncrementalEmbedding(token: exactToken0, position: emission.allTokensCount, into: xCur)
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
                        "Hybrid speculative verifier rollback failed at generated token \(emission.generatedTokenCount): \(error)"
                    )
                }

                let emissionNow = DispatchTime.now().uptimeNanoseconds
                emission.emit(exactToken0, at: emissionNow)
                if emission.generatedTokenCount >= effectiveMaxTokens || emission.allTokensCount >= config.maxSeq {
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
                    "Hybrid speculative verifier promotion failed at generated token \(emission.generatedTokenCount): \(error)"
                )
            }

            let emissionAfterFirst = DispatchTime.now().uptimeNanoseconds
            emission.emit(exactToken0, at: emissionAfterFirst)

            if emission.generatedTokenCount >= effectiveMaxTokens || emission.allTokensCount >= config.maxSeq {
                break
            }

            let exactToken1: TokenID
            do {
                exactToken1 = try cachedRuntimePair.verifierRuntime.selectGreedyToken(vocab: config.vocab)
            } catch {
                throw RealModelInferenceError.runtimeFailure(
                    "Hybrid speculative verifier token-1 failed at generated token \(emission.generatedTokenCount): \(error)"
                )
            }
            if emission.terminatesDecoding(exactToken1) {
                break
            }

            do {
                let committedSecondToken = exactToken1 == proposedToken1 ? proposedToken1 : exactToken1
                try writeIncrementalEmbedding(token: committedSecondToken, position: emission.allTokensCount, into: xCur)
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
                    "Hybrid speculative commit failed at generated token \(emission.generatedTokenCount): \(error)"
                )
            }

            let emissionAfterSecond = DispatchTime.now().uptimeNanoseconds
            emission.emit(exactToken1, at: emissionAfterSecond)

            if emission.allTokensCount >= config.maxSeq {
                break
            }
        }

        return emission.makeResult(compileTimeMs: compileTimeMs)
    }

    mutating func cachedSpeculativeRuntimePair(
        draftLayerCount: Int,
        maxSeq: Int,
        environment: [String: String]
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
            assets: gpt2Assets,
            environment: environment
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


    static func resolveExactTwoTokenDraft(
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

    func encodePrompt(_ prompt: String) throws -> [TokenID] {
        let environment = policies.environment
        let textToEncode: String
        // CLI `preparedGeneratePrompt` may already apply the same wrap.
        if environment["ESPRESSO_RAW_PROMPT"] == "1" || prompt.hasPrefix("<|im_start|>") {
            textToEncode = prompt
        } else if QwenInstructPrompt.shouldWrap(config: config) {
            textToEncode = QwenInstructPrompt.wrapUserTurn(prompt)
        } else {
            textToEncode = prompt
        }
        let rawTokens = tokenizer.encode(textToEncode)
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

    func composeEmbeddingInput(tokens: [TokenID], spatial: Int) -> [Float] {
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

    func writeIncrementalEmbeddingLlama(
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




    static func compileHybridLayers(
        config: MultiModelConfig,
        weightDirURL: URL,
        sourceLayerRange: Range<Int>? = nil,
        maxSeq: Int,
        environment: [String: String]
    ) throws -> LayerStorage<HybridDecodeKernelSet> {
        let layerRange = sourceLayerRange ?? (0..<config.nLayer)
        let kernelOptions = HybridDecodeKernelOptions.resolve(environment: environment)
        let useDonorDelta = supportsHybridDonorDelta(
            config: config,
            environment: environment
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
                    donorHexIDs: useDonorDelta ? donorHexIDs : nil,
                    options: kernelOptions
                )
                if useDonorDelta {
                    donorHexIDs = kernels.donorHexIDs
                }
                return kernels
            } catch {
                throw RealModelInferenceError.runtimeFailure(
                    "Hybrid decode compilation failed for layer \(layerIndex): \(error). Failing-kernel MIL is dumped to $TMPDIR/espresso-hybrid-<kernel>-<unix-seconds>.mil"
                )
            }
        })
    }

}
