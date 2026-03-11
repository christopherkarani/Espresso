import Foundation
import ANETypes

public enum LocalBigramArtifactBuilderError: Error, Equatable, Sendable {
    case notEnoughTokens
    case unsupportedToken(UInt16, vocabSize: Int)
    case unsupportedFeatureWidth(maxToken: UInt16, dim: Int)
    case studentContract(String)
}

public struct LocalBigramArtifacts: ~Copyable {
    public let generationWeights: GenerationWeights
    public let recurrentWeights: RecurrentGenerationWeights
    public let futureSidecar: TwoStepStudentSidecar

    public init(
        generationWeights: consuming GenerationWeights,
        recurrentWeights: consuming RecurrentGenerationWeights,
        futureSidecar: consuming TwoStepStudentSidecar
    ) {
        self.generationWeights = generationWeights
        self.recurrentWeights = recurrentWeights
        self.futureSidecar = futureSidecar
    }
}

public enum LocalBigramArtifactBuilder {
    public static func buildRecurrentWeights(
        tokens: [UInt16],
        layerCount: Int,
        vocabSize: Int = ModelConfig.vocab
    ) throws(LocalBigramArtifactBuilderError) -> RecurrentGenerationWeights {
        let artifacts = try build(tokens: tokens, layerCount: layerCount, vocabSize: vocabSize)
        return artifacts.recurrentWeights
    }

    public static func buildFutureSidecar(
        tokens: [UInt16],
        layerCount: Int,
        vocabSize: Int = ModelConfig.vocab
    ) throws(LocalBigramArtifactBuilderError) -> TwoStepStudentSidecar {
        let artifacts = try build(tokens: tokens, layerCount: layerCount, vocabSize: vocabSize)
        return artifacts.futureSidecar
    }

    public static func build(
        tokens: [UInt16],
        layerCount: Int,
        vocabSize: Int = ModelConfig.vocab
    ) throws(LocalBigramArtifactBuilderError) -> LocalBigramArtifacts {
        guard tokens.count >= 2 else {
            throw .notEnoughTokens
        }
        guard layerCount > 0 else {
            throw .unsupportedFeatureWidth(maxToken: 0, dim: ModelConfig.dim)
        }

        let maxToken = tokens.max() ?? 0
        guard Int(maxToken) < vocabSize else {
            throw .unsupportedToken(maxToken, vocabSize: vocabSize)
        }
        guard Int(maxToken) < ModelConfig.dim else {
            throw .unsupportedFeatureWidth(maxToken: maxToken, dim: ModelConfig.dim)
        }

        let nextByToken = mostLikelyNextTokenByCurrentToken(tokens: tokens)
        let futureByToken = mostLikelyFutureTokenByCurrentToken(nextByToken: nextByToken)

        let generationLayers = LayerStorage<LayerWeights>(count: layerCount) { _ in
            let weights = LayerWeights()
            zeroFill(weights.Wq)
            zeroFill(weights.Wk)
            zeroFill(weights.Wv)
            zeroFill(weights.Wo)
            zeroFill(weights.W1)
            zeroFill(weights.W2)
            zeroFill(weights.W3)
            fill(weights.rmsAtt, value: 1)
            fill(weights.rmsFfn, value: 1)
            return weights
        }

        let recurrentLayers = LayerStorage<RWKVStyleRecurrentWeights>(count: layerCount) { _ in
            let weights = RWKVStyleRecurrentWeights()
            fill(weights.rms, value: 1)
            zeroFill(weights.Wx)
            zeroFill(weights.Ws)
            zeroFill(weights.Wd)
            zeroFill(weights.Wo)
            return weights
        }

        let rmsFinal = TensorBuffer(count: ModelConfig.dim, zeroed: false)
        fill(rmsFinal, value: 1)

        let embedding = TensorBuffer(count: vocabSize * ModelConfig.dim, zeroed: true)
        let classifier = TensorBuffer(count: vocabSize * ModelConfig.dim, zeroed: true)
        fillBigramEmbeddingAndClassifier(
            embedding: embedding,
            classifier: classifier,
            nextByToken: nextByToken,
            vocabSize: vocabSize
        )

        let generationWeights = GenerationWeights(
            layers: generationLayers,
            rmsFinal: cloneTensor(rmsFinal),
            embedding: cloneTensor(embedding),
            classifier: cloneTensor(classifier),
            sharedClassifier: false,
            vocabSize: vocabSize
        )
        let recurrentWeights = RecurrentGenerationWeights(
            layers: recurrentLayers,
            rmsFinal: cloneTensor(rmsFinal),
            embedding: cloneTensor(embedding),
            classifier: cloneTensor(classifier),
            sharedClassifier: false,
            vocabSize: vocabSize
        )
        let futureSidecar: TwoStepStudentSidecar
        let futureClassifier = TensorBuffer(count: vocabSize * ModelConfig.dim, zeroed: true)
        fillFutureClassifier(
            futureClassifier: futureClassifier,
            futureByToken: futureByToken,
            vocabSize: vocabSize
        )

        do {
            futureSidecar = try TwoStepStudentSidecar(
                contract: try TwoStepStudentContract(
                    dim: ModelConfig.dim,
                    vocabSize: vocabSize,
                    layerCount: layerCount,
                    teacherClassifierWasShared: false
                ),
                futureRMS: cloneTensor(rmsFinal),
                futureClassifier: futureClassifier
            )
        } catch {
            throw .studentContract("\(error)")
        }

        return LocalBigramArtifacts(
            generationWeights: generationWeights,
            recurrentWeights: recurrentWeights,
            futureSidecar: futureSidecar
        )
    }

    public static func mostLikelyNextTokenByCurrentToken(tokens: [UInt16]) -> [UInt16: UInt16] {
        var counts: [UInt16: [UInt16: Int]] = [:]
        for idx in 0..<(tokens.count - 1) {
            let current = tokens[idx]
            let next = tokens[idx + 1]
            counts[current, default: [:]][next, default: 0] += 1
        }

        var result: [UInt16: UInt16] = [:]
        for (current, nextCounts) in counts {
            let best = nextCounts.max { lhs, rhs in
                if lhs.value == rhs.value {
                    return lhs.key > rhs.key
                }
                return lhs.value < rhs.value
            }
            result[current] = best?.key ?? 0
        }
        return result
    }

    public static func mostLikelyFutureTokenByCurrentToken(
        nextByToken: [UInt16: UInt16]
    ) -> [UInt16: UInt16] {
        var result: [UInt16: UInt16] = [:]
        result.reserveCapacity(nextByToken.count)

        for (current, next) in nextByToken {
            if let future = nextByToken[next] {
                result[current] = future
            } else {
                result[current] = next
            }
        }
        return result
    }

    private static func fillBigramEmbeddingAndClassifier(
        embedding: borrowing TensorBuffer,
        classifier: borrowing TensorBuffer,
        nextByToken: [UInt16: UInt16],
        vocabSize: Int
    ) {
        embedding.withUnsafeMutablePointer { embeddingPtr in
            classifier.withUnsafeMutablePointer { classifierPtr in
                for token in 0..<vocabSize where token < ModelConfig.dim {
                    embeddingPtr[token * ModelConfig.dim + token] = 1
                }
                for (current, next) in nextByToken {
                    let currentIndex = Int(current)
                    let nextIndex = Int(next)
                    guard currentIndex < ModelConfig.dim, nextIndex < vocabSize else { continue }
                    classifierPtr[nextIndex * ModelConfig.dim + currentIndex] = 10
                }
            }
        }
    }

    private static func fillFutureClassifier(
        futureClassifier: borrowing TensorBuffer,
        futureByToken: [UInt16: UInt16],
        vocabSize: Int
    ) {
        futureClassifier.withUnsafeMutablePointer { classifierPtr in
            for (current, future) in futureByToken {
                let currentIndex = Int(current)
                let futureIndex = Int(future)
                guard currentIndex < ModelConfig.dim, futureIndex < vocabSize else { continue }
                classifierPtr[futureIndex * ModelConfig.dim + currentIndex] = 10
            }
        }
    }
}

@inline(__always)
private func fill(_ buffer: borrowing TensorBuffer, value: Float) {
    buffer.withUnsafeMutablePointer { ptr in
        for idx in 0..<buffer.count {
            ptr[idx] = value
        }
    }
}

@inline(__always)
private func zeroFill(_ buffer: borrowing TensorBuffer) {
    fill(buffer, value: 0)
}

@inline(__always)
private func cloneTensor(_ source: borrowing TensorBuffer) -> TensorBuffer {
    let copy = TensorBuffer(count: source.count, zeroed: false)
    source.withUnsafePointer { src in
        copy.withUnsafeMutablePointer { dst in
            dst.update(from: src, count: source.count)
        }
    }
    return copy
}
