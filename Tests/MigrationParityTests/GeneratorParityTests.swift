import Foundation
import Testing
import MILGenerator
import ANETypes
import ANECodegen

@Suite("Generator parity")
struct GeneratorParityTests {
    @Test func generationClassifierParity() throws { try assertParity("GenerationClassifierGenerator", GenerationClassifierGenerator(vocabSize: ModelConfig.vocab, laneSpatial: 1)) }
    @Test func generationRMSNormClassifierParity() throws { try assertParity("GenerationRMSNormClassifierGenerator", GenerationRMSNormClassifierGenerator(vocabSize: ModelConfig.vocab, laneSpatial: 32)) }
    @Test func generationRMSNormProjectionParity() throws { try assertParity("GenerationRMSNormProjectionGenerator", GenerationRMSNormProjectionGenerator(bottleneck: 128, laneSpatial: 32, groups: 1)) }
    @Test func ffnForwardInferenceParity() throws { try assertParity("FFNForwardInferenceGenerator", FFNForwardInferenceGenerator()) }
    @Test func sdpaForwardParity() throws { try assertParity("SDPAForwardGenerator", SDPAForwardGenerator()) }
    @Test func sdpaForwardInferenceParity() throws { try assertParity("SDPAForwardInferenceGenerator", SDPAForwardInferenceGenerator()) }
    @Test func ffnForwardParity() throws { try assertParity("FFNForwardGenerator", FFNForwardGenerator()) }
    @Test func decodeAttentionQKVParity() throws { try assertDecodeParity("DecodeAttentionQKVGenerator") { DecodeAttentionQKVGenerator(maxSeq: ModelConfig.seqLen, laneSpatial: 32) } }
    @Test func decodeFFNParity() throws { try assertParity("DecodeFFNGenerator", DecodeFFNGenerator(laneSpatial: 32)) }
    @Test func decodeQKVOnlyParity() throws { try assertParity("DecodeQKVOnlyGenerator", DecodeQKVOnlyGenerator(laneSpatial: 32)) }
    @Test func fusedDecodeLayerParity() throws { try assertParity("FusedDecodeLayerGenerator", FusedDecodeLayerGenerator(maxSeq: ModelConfig.seqLen, laneSpatial: 32)) }
    @Test func fusedTwoLayerDecodeParity() throws { try assertParity("FusedTwoLayerDecodeGenerator", FusedTwoLayerDecodeGenerator(maxSeq: ModelConfig.seqLen, laneSpatial: 32)) }
    @Test func factoredGenerationRMSNormClassifierParity() throws { try assertParity("FactoredGenerationRMSNormClassifierGenerator", FactoredGenerationRMSNormClassifierGenerator(vocabSize: ModelConfig.vocab, bottleneck: 128, laneSpatial: 32, groups: 1)) }
    @Test func sdpaBackward1Parity() throws { try assertParity("SDPABackward1Generator", SDPABackward1Generator()) }
    @Test func sdpaBackward2Parity() throws { try assertParity("SDPABackward2Generator", SDPABackward2Generator()) }
    @Test func ffnBackwardParity() throws { try assertParity("FFNBackwardGenerator", FFNBackwardGenerator()) }
    @Test func qkvBackwardParity() throws { try assertParity("QKVBackwardGenerator", QKVBackwardGenerator()) }
    @Test func rwkvRecurrentStepParity() throws { try assertParity("RWKVStyleRecurrentStepGenerator", RWKVStyleRecurrentStepGenerator(laneSpatial: 32)) }
    @Test func rwkvTwoStepRecurrentParity() throws { try assertParity("RWKVStyleTwoStepRecurrentGenerator", RWKVStyleTwoStepRecurrentGenerator(laneSpatial: 32)) }
    @Test func rwkvFusedTwoLayerStepParity() throws { try assertParity("RWKVStyleFusedTwoLayerStepGenerator", RWKVStyleFusedTwoLayerStepGenerator(laneSpatial: 32)) }
    @Test func rwkvFusedTwoLayerTwoStepParity() throws { try assertParity("RWKVStyleFusedTwoLayerTwoStepGenerator", RWKVStyleFusedTwoLayerTwoStepGenerator(laneSpatial: 32)) }
    @Test func rwkvFusedThreeLayerStepParity() throws { try assertParity("RWKVStyleFusedThreeLayerStepGenerator", RWKVStyleFusedThreeLayerStepGenerator(laneSpatial: 32, groups: 1, includeRMSNorm: true)) }
    @Test func rwkvFusedThreeLayerTwoStepParity() throws { try assertParity("RWKVStyleFusedThreeLayerTwoStepGenerator", RWKVStyleFusedThreeLayerTwoStepGenerator(laneSpatial: 32)) }
    @Test func rwkvFusedThreeLayerRMSNormClassifierParity() throws { try assertParity("RWKVStyleFusedThreeLayerRMSNormClassifierGenerator", RWKVStyleFusedThreeLayerRMSNormClassifierGenerator(vocabSize: ModelConfig.vocab, laneSpatial: 32)) }
    @Test func rwkvFusedThreeLayerFactoredClassifierParity() throws { try assertParity("RWKVStyleFusedThreeLayerFactoredClassifierGenerator", RWKVStyleFusedThreeLayerFactoredClassifierGenerator(vocabSize: ModelConfig.vocab, bottleneck: 128, laneSpatial: 32, groups: 1)) }
}

private func assertDecodeParity<T: MILProgramGenerator>(_ fixtureName: String, _ make: () -> T) throws {
    let mode = ProcessInfo.processInfo.environment["ESPRESSO_DECODE_ATTN_PROBE_MODE"]
    #expect(mode == nil || mode == "cache-touch", "Decode parity fixtures require default cache-touch probe mode")
    try assertParity(fixtureName, make())
}

private func assertParity(_ fixtureName: String, _ generator: some MILProgramGenerator) throws {
    let goldenMIL = try loadFixture(fixtureName)
    let newMIL = generator.milText

    #expect(
        MILDiff.structuralEquiv(goldenMIL, newMIL),
        "Op sequence mismatch for \(fixtureName): \(MILDiff.extractOps(goldenMIL)) vs \(MILDiff.extractOps(newMIL))"
    )

    for path in extractBlobPaths(goldenMIL) {
        #expect(newMIL.contains(path), "Missing weight path \(path) in \(fixtureName)")
    }

    #expect(extractReturnTuple(goldenMIL) == extractReturnTuple(newMIL), "Return tuple mismatch for \(fixtureName)")
    #expect(extractInputSignature(goldenMIL) == extractInputSignature(newMIL), "Input signature mismatch for \(fixtureName)")
}

private func loadFixture(_ name: String) throws -> String {
    let url = try #require(
        Bundle.module.url(forResource: name, withExtension: "mil")
        ?? Bundle.module.url(forResource: name, withExtension: "mil", subdirectory: "Fixtures")
    )
    return try String(contentsOf: url, encoding: .utf8)
}

private func extractBlobPaths(_ mil: String) -> [String] {
    let regex = try! NSRegularExpression(pattern: #"BLOBFILE\(path=string\("([^"]+)"\)"#)
    let range = NSRange(mil.startIndex..<mil.endIndex, in: mil)
    return regex.matches(in: mil, range: range).compactMap { match in
        guard let r = Range(match.range(at: 1), in: mil) else { return nil }
        return String(mil[r])
    }
}

private func extractReturnTuple(_ mil: String) -> [String] {
    guard let line = mil.split(whereSeparator: \.isNewline).first(where: { $0.contains("} -> (") }) else { return [] }
    guard let start = line.firstIndex(of: "("), let end = line.lastIndex(of: ")"), start < end else { return [] }
    return line[line.index(after: start)..<end]
        .split(separator: ",")
        .map { $0.replacingOccurrences(of: " ", with: "") }
}

private func extractInputSignature(_ mil: String) -> String {
    guard let line = mil.split(whereSeparator: \.isNewline).first(where: { $0.contains("func main<ios18>(") }) else { return "" }
    return line.replacingOccurrences(of: " ", with: "")
}
