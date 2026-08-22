import ANETypes
import Foundation
import ModelSupport

struct LlamaQKNormWeights: Sendable {
    let q: [Float]
    let k: [Float]
}

/// Q/K/V projection biases for llama-family layers that have them (Qwen2 does; plain
/// llama does not). Always all three or none.
struct LlamaQKVBiasWeights: Sendable {
    let q: [Float]
    let k: [Float]
    let v: [Float]
}

struct LlamaCPUQKVWeights: Sendable {
    let rmsAtt: [Float]
    let wq: [Float]
    let wk: [Float]
    let wv: [Float]
    let qNorm: [Float]?
    let kNorm: [Float]?
    let qkvBias: LlamaQKVBiasWeights?
}

// Exact-CPU trunk runtime (extracted from RealModelInferenceEngine).
//
// Owns the pure-host decode state for one llama serving session on the
// exact-CPU trunk: loaded weights, KV caches, checkpoint/rollback, and the
// per-token forward math. The layer-forward kernel is shared with
// QwenLayerParityProbe through the engine extension below, whose call
// surface is frozen for tests.

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
    let qkvBias: LlamaQKVBiasWeights?
}

struct CachedExactCPULlamaWeights: Sendable {
    let tokenEmbedding: [Float]
    let finalNormGamma: [Float]
    let lmHead: [Float]
    let lmHeadFP16: [UInt16]?
    let layers: [ExactCPULlamaLayerWeights]
}

struct CPUExactLlamaCheckpoint: Sendable {
    let visibleTokenCount: Int
    let lastHidden: [Float]
    let kCaches: [[Float]]
    let vCaches: [[Float]]
}

struct CPUExactLlamaRuntime: Sendable {
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
        let coreWeights = try TopLevelAssetLoader.loadLlamaCoreWeights(
            config: config,
            topLevelPaths: topLevelPaths
        )
        self.tokenEmbedding = coreWeights.tokenEmbedding
        self.finalNormGamma = coreWeights.finalNormGamma
        self.lmHead = coreWeights.lmHead
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
                RealModelInferenceEngine.projectRowMajorMatrix(
                    matrix: layer.wq,
                    rows: config.attentionDim,
                    cols: config.dModel,
                    vector: attnNormed,
                    bias: layer.qkvBias?.q
                )
            )
            var k = maybeRound(
                RealModelInferenceEngine.projectRowMajorMatrix(
                    matrix: layer.wk,
                    rows: config.kvDim,
                    cols: config.dModel,
                    vector: attnNormed,
                    bias: layer.qkvBias?.k
                )
            )
            let vRounded = maybeRound(
                RealModelInferenceEngine.projectRowMajorMatrix(
                    matrix: layer.wv,
                    rows: config.kvDim,
                    cols: config.dModel,
                    vector: attnNormed,
                    bias: layer.qkvBias?.v
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

extension RealModelInferenceEngine {
    static func loadExactCPULlamaLayerWeights(
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
            kNorm: qkNormWeights?.k,
            qkvBias: try loadLlamaQKVBiasWeights(config: config, paths: paths)
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

    static func roundFloat16Vector(_ values: [Float]) -> [Float] {
        values.map { Float(Float16($0)) }
    }

    /// Runs one Llama-family transformer layer on the CPU for a single token position.
    ///
    /// This is the single definition of the exact-CPU layer math: the incremental decode
    /// path and the per-layer parity probe both call it, so a parity measurement cannot
    /// drift away from what is actually served.
    ///
    /// `kCache`/`vCache` are channel-major (`channel * cacheStride + position`) and are
    /// updated in place for `position`.
    static func exactCPULlamaLayerForward(
        hidden: [Float],
        layer: ExactCPULlamaLayerWeights,
        config: MultiModelConfig,
        position: Int,
        kCache: inout [Float],
        vCache: inout [Float],
        cacheStride: Int,
        roundIntermediatesToFP16: Bool
    ) -> [Float] {
        let maybeRound: ([Float]) -> [Float] = { values in
            roundIntermediatesToFP16 ? Self.roundFloat16Vector(values) : values
        }
        let attnNormed = Self.rmsNorm(hidden, weight: layer.rmsAtt, eps: Float(config.normEps))
        var q = maybeRound(
            Self.projectRowMajorMatrix(
                matrix: layer.wq,
                rows: config.attentionDim,
                cols: config.dModel,
                vector: attnNormed,
                bias: layer.qkvBias?.q
            )
        )
        var k = maybeRound(
            Self.projectRowMajorMatrix(
                matrix: layer.wk,
                rows: config.kvDim,
                cols: config.dModel,
                vector: attnNormed,
                bias: layer.qkvBias?.k
            )
        )
        let vRounded = maybeRound(
            Self.projectRowMajorMatrix(
                matrix: layer.wv,
                rows: config.kvDim,
                cols: config.dModel,
                vector: attnNormed,
                bias: layer.qkvBias?.v
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
            kCache[channel * cacheStride + position] = k[channel]
            vCache[channel * cacheStride + position] = vRounded[channel]
        }

        let context = Self.decodeContextFromCaches(
            qOut: q,
            kCache: kCache,
            vCache: vCache,
            heads: config.nHead,
            kvHeads: config.nKVHead,
            headDim: config.headDim,
            visibleTokenCount: position + 1,
            cacheStride: cacheStride
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
        return maybeRound(zip(projected, down).map(+))
    }

    static func decodeContextFromCaches(
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
}
