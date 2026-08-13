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

    try metadata(decodePath: "metal").write(to: url, atomically: true, encoding: .utf8)
    #expect(throws: (any Error).self) {
        try ESPModelConfigIO.load(fromMetadataFile: url)
    }
}
