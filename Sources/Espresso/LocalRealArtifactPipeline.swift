import Foundation
import ANETypes

public enum LocalRealArtifactPipelineError: Error, Sendable, Equatable {
    case invalidPrefix(String)
    case manifestEncodeFailed(String)
    case manifestWriteFailed(String)
    case modelLoadFailed(String)
}

public struct LocalRealArtifactManifest: Sendable, Equatable, Codable {
    public let datasetPath: String
    public let generationModelPath: String
    public let recurrentCheckpointPath: String
    public let futureSidecarPath: String
    public let manifestPath: String
    public let promptToken: UInt16
    public let tokenCount: Int
    public let layerCount: Int
    public let vocabSize: Int

    public init(
        datasetPath: String,
        generationModelPath: String,
        recurrentCheckpointPath: String,
        futureSidecarPath: String,
        manifestPath: String,
        promptToken: UInt16,
        tokenCount: Int,
        layerCount: Int,
        vocabSize: Int
    ) {
        self.datasetPath = datasetPath
        self.generationModelPath = generationModelPath
        self.recurrentCheckpointPath = recurrentCheckpointPath
        self.futureSidecarPath = futureSidecarPath
        self.manifestPath = manifestPath
        self.promptToken = promptToken
        self.tokenCount = tokenCount
        self.layerCount = layerCount
        self.vocabSize = vocabSize
    }
}

public enum LocalRealArtifactPipeline {
    public static func exportLocalBigramArtifacts(
        datasetPath: String,
        prefix: String,
        layerCount: Int = ModelConfig.nLayers,
        vocabSize: Int = ModelConfig.vocab
    ) throws -> LocalRealArtifactManifest {
        guard !prefix.isEmpty else {
            throw LocalRealArtifactPipelineError.invalidPrefix(prefix)
        }

        let dataset: TokenDataset
        do {
            dataset = try TokenDataset(path: datasetPath, seqLen: 0)
        } catch {
            throw LocalRealArtifactPipelineError.modelLoadFailed("\(error)")
        }

        let tokens = Array(UnsafeBufferPointer(start: dataset.tokensBase, count: dataset.nTokens))
        let artifacts = try LocalBigramArtifactBuilder.build(
            tokens: tokens,
            layerCount: layerCount,
            vocabSize: vocabSize
        )

        let generationModelPath = "\(prefix).generation.bin"
        let recurrentCheckpointPath = "\(prefix).recurrent.bin"
        let futureSidecarPath = "\(prefix).future-sidecar.bin"
        let manifestPath = "\(prefix).manifest.json"

        do {
            try GenerationModelWeightStore.save(artifacts.generationWeights, to: generationModelPath)
            try RecurrentGenerationWeightStore.save(artifacts.recurrentWeights, to: recurrentCheckpointPath)
            try TwoStepStudentCheckpoint.save(path: futureSidecarPath, sidecar: artifacts.futureSidecar)
        } catch {
            throw LocalRealArtifactPipelineError.modelLoadFailed("\(error)")
        }

        let manifest = LocalRealArtifactManifest(
            datasetPath: datasetPath,
            generationModelPath: generationModelPath,
            recurrentCheckpointPath: recurrentCheckpointPath,
            futureSidecarPath: futureSidecarPath,
            manifestPath: manifestPath,
            promptToken: tokens.first ?? 0,
            tokenCount: tokens.count,
            layerCount: layerCount,
            vocabSize: vocabSize
        )
        do {
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            let data = try encoder.encode(manifest)
            try data.write(to: URL(fileURLWithPath: manifestPath), options: .atomic)
        } catch {
            throw LocalRealArtifactPipelineError.manifestWriteFailed("\(error)")
        }

        return manifest
    }

    public static func offlineAcceptanceGate(
        recurrentCheckpointPath: String,
        futureSidecarPath: String,
        promptTokens: [UInt16],
        maxNewTokens: Int,
        strategy: TokenSelectionStrategy = .argmax
    ) throws -> OfflineExactAcceptanceTrace {
        do {
            let weights = try RecurrentGenerationWeightStore.load(from: recurrentCheckpointPath)
            let sidecar = try TwoStepStudentCheckpoint.load(path: futureSidecarPath)

            var teacher = try CPURecurrentGenerationModel(
                weights: weights,
                layerCount: weights.layers.count
            )
            var student = try CPURecurrentGenerationModel(
                weights: weights,
                layerCount: weights.layers.count,
                futureSidecar: sidecar
            )

            return try OfflineExactAcceptanceEvaluator.evaluate(
                teacher: &teacher,
                student: &student,
                promptTokens: promptTokens,
                maxNewTokens: maxNewTokens,
                strategy: strategy
            )
        } catch {
            throw LocalRealArtifactPipelineError.modelLoadFailed("\(error)")
        }
    }
}
