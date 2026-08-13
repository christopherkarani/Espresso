import Foundation
import Testing
@testable import ESPBundle
@testable import ESPCompiler
@testable import ESPConvert

@Test func convertSupportMatrixMarksSupportedFamiliesAsTierA() {
    #expect(ESPConvertSupportMatrix.support(for: .qwen) == .tierA)
    #expect(ESPConvertSupportMatrix.support(for: .llama) == .tierA)
    #expect(ESPConvertSupportMatrix.support(for: .gpt2) == .tierA)
}

@Test func convertMatrixUsesExplicitNativeImportSources() {
    #expect(ESPConvertSupportMatrix.recommendedImportSource(for: .qwen) == .nativeModelDirectory)
    #expect(ESPConvertSupportMatrix.recommendedImportSource(for: .llama) == .nativeModelDirectory)
    #expect(
        ESPConvertSupportMatrix.recommendedImportSource(for: .gpt2)
            == .nativeModelDirectoryWithExternalTokenizer
    )
}

@Test func nativeExporterInfersFamilyAndBackends() throws {
    let metadataURL = try writeMetadata(
        """
        {
          "name": "Qwen3-0.6B",
          "nLayer": 28,
          "nHead": 16,
          "nKVHead": 8,
          "dModel": 1024,
          "headDim": 128,
          "hiddenDim": 3072,
          "vocab": 151936,
          "maxSeq": 4096,
          "normEps": 0.000001,
          "ropeTheta": 10000,
          "eosToken": 151643,
          "architecture": "llama"
        }
        """
    )

    let config = try ESPModelConfigIO.load(fromMetadataFile: metadataURL)
    let manifest = try ESPNativeModelBundleExporter.makeManifest(from: config)
    #expect(manifest.modelFamily == .qwen)
    #expect(manifest.supportedBackends == [.anePrivate, .cpuSafe])
    #expect(manifest.supportedProfiles.contains(.prefill2048))
    #expect(manifest.modelTier == .compat)
    #expect(manifest.behaviorClass == .exact)
    #expect(manifest.contextTargetTokens == 4096)
    #expect(manifest.optimization.recipe == "native-baseline")
}

@Test func nativeExporterAllowsExplicitContextTargetAndLineageOverrides() throws {
    let metadataURL = try writeMetadata(
        """
        {
          "name": "llama2.c-stories110M",
          "nLayer": 12,
          "nHead": 12,
          "nKVHead": 12,
          "dModel": 768,
          "headDim": 64,
          "hiddenDim": 2048,
          "vocab": 32000,
          "maxSeq": 1024,
          "normEps": 0.00001,
          "architecture": "llama"
        }
        """
    )

    let config = try ESPModelConfigIO.load(fromMetadataFile: metadataURL)
    let manifest = try ESPNativeModelBundleExporter.makeManifest(
        from: config,
        options: .init(
            contextTargetTokens: 256,
            modelTier: .optimized,
            behaviorClass: .exact,
            optimization: .init(
                recipe: "stories-ctx256",
                qualityGate: "short-long-prompt-parity",
                teacherModel: nil,
                draftModel: nil,
                performanceTarget: "105 tok/s"
            ),
            outputHead: .init(
                kind: .factored,
                behaviorClass: .nearExact,
                bottleneck: 128,
                groups: 1,
                projectionRef: "weights/cls_proj.bin",
                expansionRef: "weights/cls_expand.bin"
            ),
            draft: .init(
                kind: .exactTwoToken,
                behaviorClass: .exact,
                horizon: 2,
                verifier: "exact",
                rollback: "exact_replay",
                artifactRef: "weights/future-sidecar.bin",
                acceptanceMetric: "accepted_future_tokens"
            )
        )
    )

    #expect(manifest.modelID == "llama2.c-stories110M-ctx256")
    #expect(manifest.maxContext == 1024)
    #expect(manifest.contextTargetTokens == 256)
    #expect(manifest.modelTier == .optimized)
    #expect(manifest.optimization.recipe == "stories-ctx256")
    #expect(manifest.optimization.performanceTarget == "105 tok/s")
    #expect(manifest.outputHead?.kind == .factored)
    #expect(manifest.draft?.kind == .exactTwoToken)
}

@Test func nativeExporterRejectsContextTargetAboveModelContext() throws {
    let metadataURL = try writeMetadata(
        """
        {
          "name": "llama2.c-stories110M",
          "nLayer": 12,
          "nHead": 12,
          "nKVHead": 12,
          "dModel": 768,
          "headDim": 64,
          "hiddenDim": 2048,
          "vocab": 32000,
          "maxSeq": 256,
          "normEps": 0.00001,
          "architecture": "llama"
        }
        """
    )

    let config = try ESPModelConfigIO.load(fromMetadataFile: metadataURL)

    do {
        _ = try ESPNativeModelBundleExporter.makeManifest(
            from: config,
            options: .init(contextTargetTokens: 512)
        )
        Issue.record("Expected invalid context target rejection")
    } catch let error as ESPBundleValidationError {
        #expect(error == .invalidContextTarget(512))
    } catch {
        Issue.record("Unexpected error: \(error)")
    }
}

@Test func nativeExporterBuildsBundleFromCoLocatedTokenizerAssets() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
    let modelDirectory = root.appendingPathComponent("stories110m", isDirectory: true)
    let bundleURL = root.appendingPathComponent("stories110m.esp", isDirectory: true)
    try FileManager.default.createDirectory(at: modelDirectory, withIntermediateDirectories: true)
    try """
    {
      "name": "llama2.c-stories110M",
      "nLayer": 12,
      "nHead": 12,
      "nKVHead": 12,
      "dModel": 768,
      "headDim": 64,
      "hiddenDim": 2048,
      "vocab": 32000,
      "maxSeq": 256,
      "normEps": 0.00001,
      "architecture": "llama"
    }
    """.write(to: modelDirectory.appendingPathComponent("metadata.json"), atomically: true, encoding: .utf8)
    try Data("weights".utf8).write(to: modelDirectory.appendingPathComponent("lm_head.bin"))
    try Data("tokenizer".utf8).write(to: modelDirectory.appendingPathComponent("tokenizer.model"))

    let archive = try ESPNativeModelBundleExporter.exportModel(
        at: modelDirectory,
        outputBundleURL: bundleURL
    )

    #expect(archive.manifest.modelFamily == .llama)
    #expect(FileManager.default.fileExists(atPath: archive.weightsURL.appendingPathComponent("metadata.json").path))
    #expect(FileManager.default.fileExists(atPath: archive.tokenizerURL.appendingPathComponent("tokenizer.model").path))
    #expect(!FileManager.default.fileExists(atPath: archive.weightsURL.appendingPathComponent("tokenizer.model").path))
    // Copying the co-located model directory verbatim would duplicate every weight blob
    // under tokenizer/ and double the bundle on disk.
    #expect(!FileManager.default.fileExists(atPath: archive.tokenizerURL.appendingPathComponent("lm_head.bin").path))
    #expect(!FileManager.default.fileExists(atPath: archive.tokenizerURL.appendingPathComponent("metadata.json").path))
}

@Test func nativeExporterBuildsBundleWithExternalTokenizerDirectory() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
    let modelDirectory = root.appendingPathComponent("gpt2_124m", isDirectory: true)
    let tokenizerDirectory = root.appendingPathComponent("gpt2_tokenizer", isDirectory: true)
    let bundleURL = root.appendingPathComponent("gpt2_124m.esp", isDirectory: true)
    try FileManager.default.createDirectory(at: modelDirectory, withIntermediateDirectories: true)
    try FileManager.default.createDirectory(at: tokenizerDirectory, withIntermediateDirectories: true)
    try """
    {
      "name": "gpt2_124m",
      "nLayer": 12,
      "nHead": 12,
      "nKVHead": 12,
      "dModel": 768,
      "headDim": 64,
      "hiddenDim": 3072,
      "vocab": 50257,
      "maxSeq": 1024,
      "normEps": 0.00001,
      "architecture": "gpt2"
    }
    """.write(to: modelDirectory.appendingPathComponent("metadata.json"), atomically: true, encoding: .utf8)
    try Data("weights".utf8).write(to: modelDirectory.appendingPathComponent("lm_head.bin"))
    try Data("{}".utf8).write(to: tokenizerDirectory.appendingPathComponent("tokenizer.json"))

    let archive = try ESPNativeModelBundleExporter.exportModel(
        at: modelDirectory,
        tokenizerDirectory: tokenizerDirectory,
        outputBundleURL: bundleURL
    )

    #expect(archive.manifest.modelFamily == .gpt2)
    #expect(archive.manifest.supportedBackends == [.anePrivate])
    #expect(FileManager.default.fileExists(atPath: archive.tokenizerURL.appendingPathComponent("tokenizer.json").path))
}

private func writeMetadata(_ metadata: String) throws -> URL {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    let url = directory.appendingPathComponent("metadata.json")
    try metadata.write(to: url, atomically: true, encoding: .utf8)
    return url
}
