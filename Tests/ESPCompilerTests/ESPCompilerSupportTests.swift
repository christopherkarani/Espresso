import Testing
import Foundation
@testable import ESPBundle
@testable import ESPCompiler
import ModelSupport

@Test func compilerSupportMatrixMatchesPrivateFirstV1Scope() {
    #expect(ESPCompilerSupportMatrix.supportedModelFamilies == [.gpt2, .llama, .qwen])
    #expect(ESPCompilerSupportMatrix.defaultBackends == [.anePrivate, .cpuSafe])
    #expect(ESPCompilerSupportMatrix.defaultShippingProfiles == [.prefill256, .prefill2048, .decode1])
    #expect(ESPCompilerSupportMatrix.experimentalProfiles == [.decode2])
}

@Test func compilerRejectsDynamicControlFlowArchitectures() {
    let result = ESPCompilerSupportMatrix.classifyArchitecture(
        hasDynamicControlFlow: true,
        hasMixtureOfExperts: false
    )

    #expect(result == .unsupported(.dynamicControlFlow))
}

@Test func compilerRejectsMixtureOfExpertsArchitectures() {
    let result = ESPCompilerSupportMatrix.classifyArchitecture(
        hasDynamicControlFlow: false,
        hasMixtureOfExperts: true
    )

    #expect(result == .unsupported(.mixtureOfExperts))
}

@Test func compilerLoadsModelConfigFromPreparedMetadata() throws {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    let url = directory.appendingPathComponent("metadata.json")
    let metadata = """
    {
      "name": "qwen3",
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
    try metadata.write(to: url, atomically: true, encoding: .utf8)

    let config = try ESPModelConfigIO.load(fromMetadataFile: url)
    #expect(config.name == "qwen3")
    #expect(config.architecture == .llama)
    #expect(config.maxSeq == 4096)
    #expect(config.preferredDecodePath == nil)
}

/// An artifact that declares the ANE hybrid path must keep that declaration through the
/// bundle loader; dropping it routes the model to the pure-CPU oracle instead.
@Test func compilerPreservesDeclaredDecodePath() throws {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    let url = directory.appendingPathComponent("metadata.json")

    func metadata(decodePath: String) -> String {
        """
        {
          "name": "Qwen2.5-0.5B-Instruct",
          "nLayer": 24,
          "nHead": 14,
          "nKVHead": 2,
          "dModel": 896,
          "headDim": 64,
          "hiddenDim": 4864,
          "vocab": 151936,
          "maxSeq": 4096,
          "normEps": 0.000001,
          "ropeTheta": 1000000,
          "eosToken": 151645,
          "architecture": "llama",
          "preferredDecodePath": "\(decodePath)"
        }
        """
    }

    try metadata(decodePath: "hybrid").write(to: url, atomically: true, encoding: .utf8)
    #expect(try ESPModelConfigIO.load(fromMetadataFile: url).preferredDecodePath == .hybrid)

    try metadata(decodePath: "exact_cpu").write(to: url, atomically: true, encoding: .utf8)
    #expect(try ESPModelConfigIO.load(fromMetadataFile: url).preferredDecodePath == .exactCPU)

    try metadata(decodePath: " Hybrid ").write(to: url, atomically: true, encoding: .utf8)
    #expect(try ESPModelConfigIO.load(fromMetadataFile: url).preferredDecodePath == .hybrid)

    try metadata(decodePath: "metal").write(to: url, atomically: true, encoding: .utf8)
    #expect(throws: MultiModelConfig.PreferredDecodePath.ParseError.unsupported("metal")) {
        try ESPModelConfigIO.load(fromMetadataFile: url)
    }
}

/// Pack-native loads metadata through ESPModelConfigIO. Dropping preferredDecodePath
/// on the 1.5B shape would send the packed bundle back to the Qwen name-heuristic CPU path.
@Test func compilerPreservesQwen15BHybridDecodePath() throws {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    let url = directory.appendingPathComponent("metadata.json")
    let metadata = """
        {
          "name": "Qwen2.5-1.5B-Instruct",
          "nLayer": 28,
          "nHead": 12,
          "nKVHead": 2,
          "dModel": 1536,
          "headDim": 128,
          "hiddenDim": 8960,
          "vocab": 151936,
          "maxSeq": 1024,
          "normEps": 0.000001,
          "ropeTheta": 1000000,
          "eosToken": 151645,
          "architecture": "llama",
          "preferredDecodePath": "hybrid"
        }
        """
    try metadata.write(to: url, atomically: true, encoding: .utf8)

    let config = try ESPModelConfigIO.load(fromMetadataFile: url)
    #expect(config.name == "Qwen2.5-1.5B-Instruct")
    #expect(config.nLayer == 28)
    #expect(config.dModel == 1536)
    #expect(config.hiddenDim == 8960)
    #expect(config.vocab == 151936)
    #expect(config.preferredDecodePath == .hybrid)
    #expect(config.architecture == .llama)
}
