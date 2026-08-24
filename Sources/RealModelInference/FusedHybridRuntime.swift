import ANERuntime
import ANETypes
import Espresso
import IOSurface
import Darwin
import Foundation
import ModelSupport

// Fused-hybrid trunk runtime (extracted from RealModelInferenceEngine).
//
// One ANE program per transformer layer, attention included. The session owns
// the trunk's resident programs and surface handles; readiness follows
// ``CompiledReadiness`` and only the ensure function writes it.

extension RealModelInferenceEngine {
    mutating func ensureFusedHybridCompiled(bucket: Int) throws -> Bool {
        switch fusedHybridReadiness {
        case .compiled(let runtime) where runtime.bucket >= bucket:
            return false
        case .compiled, .notCompiled:
            break
        }

        let newLayers: LayerStorage<FusedHybridDecodeLayerKernelSet>
        do {
            newLayers = try Self.compileFusedHybridLayers(
                config: config,
                weightDirURL: weightDirURL,
                maxSeq: bucket,
                environment: policies.environment
            )
        } catch let error as RealModelInferenceError {
            if case .hybridFallbackDisabled = error { throw error }
            throw Self.fusedHybridFallbackError(reason: error.errorDescription ?? "\(error)")
        } catch {
            throw Self.fusedHybridFallbackError(reason: "\(error)")
        }

        var newSurfaceHandles: [FusedHybridDecodeSurfaceHandles] = []
        newSurfaceHandles.reserveCapacity(newLayers.count)
        for layerIndex in 0..<newLayers.count {
            do {
                newSurfaceHandles.append(
                    try FusedHybridDecodeSurfaceHandles(kernels: newLayers[layerIndex])
                )
            } catch {
                throw Self.fusedHybridFallbackError(
                    reason: "fused N=1 surfaces unavailable for layer \(layerIndex): \(error)"
                )
            }
        }

        compiledFusedHybridLayers = newLayers
        compiledFusedHybridSurfaceHandles = newSurfaceHandles
        if let runtime = FusedHybridCompiledRuntime(
            bucket: bucket,
            layerCount: compiledFusedHybridLayers.count,
            surfaceHandleCount: compiledFusedHybridSurfaceHandles.count,
            expectedLayerCount: config.nLayer
        ) {
            fusedHybridReadiness = .compiled(runtime)
        } else {
            fusedHybridReadiness = .notCompiled
        }
        return true
    }

    static func compileFusedHybridLayers(
        config: MultiModelConfig,
        weightDirURL: URL,
        maxSeq: Int,
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) throws -> LayerStorage<FusedHybridDecodeLayerKernelSet> {
        guard config.nHead > 0, config.nKVHead > 0, config.headDim > 0,
              config.nHead % config.nKVHead == 0 else {
            throw fusedHybridFallbackError(
                reason: "invalid fused N=1 head geometry nHead=\(config.nHead) nKVHead=\(config.nKVHead) headDim=\(config.headDim)"
            )
        }
        var donor: FusedHybridDecodeLayerKernelSet.DonorHexIDs?
        return try LayerStorage(count: config.nLayer, throwingInitializer: { layerIndex in
            fputs(
                "[FusedHybridDecode] compiling layer \(layerIndex)/\(config.nLayer) maxSeq=\(maxSeq) n=1\n",
                stderr
            )
            let paths = LayerWeightPaths.forLayer(
                layerIndex,
                config: config,
                blobDir: weightDirURL.path
            )
            let weights = try loadHybridLayerWeightsLlama(config: config, paths: paths)
            let compiled = try FusedHybridDecodeLayerKernelSet(
                weights: weights,
                maxSeq: maxSeq,
                nHeads: config.nHead,
                nKVHeads: config.nKVHead,
                headDim: config.headDim,
                donorHexIDs: donor,
                options: HybridDecodeKernelOptions.resolve(environment: environment)
            )
            donor = compiled.donorHexIDs
            return compiled
        })
    }

    mutating func generateIncrementalFusedHybridLlama(
        promptTokens: [TokenID],
        effectiveMaxTokens: Int,
        temperature: Float,
        topP: Float = 1.0,
        compileTimeMs: Double,
        maxSeq: Int,
        onStep: ((GenerationStep) -> Void)?,
        isCancelled: (() -> Bool)? = nil
    ) throws -> GenerationResult {
        switch fusedHybridReadiness {
        case .compiled:
            break
        case .notCompiled:
            throw Self.fusedHybridFallbackError(
                reason: """
                    fused N=1 state is incomplete: \
                    layers=\(compiledFusedHybridLayers.count)/\(config.nLayer) \
                    surfaces=\(compiledFusedHybridSurfaceHandles.count)/\(config.nLayer)
                    """
            )
        }

        do {
            try ForwardPass.initializeFusedHybridDecodeCaches(
                surfaceHandles: compiledFusedHybridSurfaceHandles
            )
        } catch {
            throw Self.fusedHybridFallbackError(reason: "fused N=1 cache init failed: \(error)")
        }

        let xCur = TensorBuffer(count: config.dModel, zeroed: true)
        var decodeState: DecodeState
        do {
            decodeState = try DecodeState(maxSeq: maxSeq)
        } catch {
            throw Self.fusedHybridFallbackError(reason: "fused decode state initialization failed: \(error)")
        }
        var timings = HybridDecodeTimingBreakdown()
        let hopsPerToken = Self.fusedHopsPerToken(nLayer: config.nLayer)

        for (position, token) in promptTokens.enumerated() {
            try writeIncrementalEmbeddingLlama(token: token, into: xCur)
            do {
                try ForwardPass.runFusedHybridDecodeTimed(
                    xCur: xCur,
                    kernels: compiledFusedHybridLayers,
                    surfaceHandles: compiledFusedHybridSurfaceHandles,
                    decodeState: &decodeState,
                    headDim: config.headDim,
                    ropeTheta: config.ropeTheta,
                    timings: &timings
                )
            } catch {
                throw Self.fusedHybridFallbackError(
                    reason: "fused N=1 prefill failed at prompt position \(position): \(error)"
                )
            }
        }

        var decodeProfileTokens: [HybridDecodeTimingBreakdown] = []
        decodeProfileTokens.reserveCapacity(effectiveMaxTokens)
        var pendingDecode = HybridDecodeTimingBreakdown()

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
        var rng = SystemRandomNumberGenerator()
        var normalized = [Float](repeating: 0, count: config.dModel)

        while emission.generatedTokenCount < effectiveMaxTokens {
            try Self.throwIfCancelled(isCancelled)
            let headStart = DispatchTime.now().uptimeNanoseconds
            normalized = xCur.withUnsafeBufferPointer {
                Self.rmsNorm(Array($0), weight: llamaAssets.finalNormGamma, eps: Float(config.normEps))
            }
            let nextToken = selectTokenFromNormalizedHidden(
                normalized,
                temperature: temperature,
                topP: topP,
                using: &rng
            )
            var tokenProfile = pendingDecode
            tokenProfile.tLMHead = Self.milliseconds(
                from: DispatchTime.now().uptimeNanoseconds - headStart
            )
            decodeProfileTokens.append(tokenProfile)
            let emissionNow = DispatchTime.now().uptimeNanoseconds
            emission.recordFirstTokenIfFirst(at: emissionNow)

            if emission.terminatesDecoding(nextToken) {
                emission.recordTerminalToken(nextToken)
                break
            }
            emission.emit(nextToken, at: emissionNow)

            if emission.generatedTokenCount >= effectiveMaxTokens || emission.allTokensCount >= config.maxSeq {
                break
            }

            try writeIncrementalEmbeddingLlama(token: nextToken, into: xCur)
            timings.reset()
            do {
                try ForwardPass.runFusedHybridDecodeTimed(
                    xCur: xCur,
                    kernels: compiledFusedHybridLayers,
                    surfaceHandles: compiledFusedHybridSurfaceHandles,
                    decodeState: &decodeState,
                    headDim: config.headDim,
                    ropeTheta: config.ropeTheta,
                    timings: &timings
                )
            } catch {
                throw Self.fusedHybridFallbackError(
                    reason: "fused N=1 decode failed at generated token \(emission.generatedTokenCount - 1): \(error)"
                )
            }
            pendingDecode = timings
        }

        let decodeProfileReport = decodeProfileTokens.isEmpty
            ? nil
            : HybridDecodeTokenProfile(tokens: decodeProfileTokens).formatReport()
        if let decodeProfileReport {
            fputs(decodeProfileReport + "\n", stderr)
        }

        return emission.makeResult(
            compileTimeMs: compileTimeMs,
            exactHeadBackend: classifierStrategy.exactHeadBackendLabel,
            trunk: .fusedHybrid,
            hopsPerToken: hopsPerToken,
            decodeProfileReport: decodeProfileReport
        )
    }
}
