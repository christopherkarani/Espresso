import Testing
@testable import ModelSupport

@Test func trunkTelemetryLabelsMatchCONTEXTVocabulary() {
    #expect(Trunk.fusedHybrid.rawValue == "fused")
    #expect(Trunk.splitHybrid.rawValue == "hybrid")
    #expect(Trunk.exactCPU.rawValue == "exact_cpu")
    #expect(Trunk.fusedHybrid.isHybrid)
    #expect(Trunk.splitHybrid.isHybrid)
    #expect(!Trunk.exactCPU.isHybrid)
}

@Test func trunkParseTelemetryLabelTrimsAndLowercases() throws {
    #expect(try Trunk.parseTelemetryLabel("fused") == .fusedHybrid)
    #expect(try Trunk.parseTelemetryLabel(" HYBRID ") == .splitHybrid)
    #expect(try Trunk.parseTelemetryLabel("Exact_CPU") == .exactCPU)
    #expect(throws: Trunk.ParseError.unsupported("metal")) {
        try Trunk.parseTelemetryLabel("metal")
    }
}

@Test func preferredDecodePathMapsOntoTrunkFamilyOnly() {
    #expect(MultiModelConfig.PreferredDecodePath.hybrid.prefersHybridFamily)
    #expect(MultiModelConfig.PreferredDecodePath.hybrid.exactCPUTrunk == nil)
    #expect(MultiModelConfig.PreferredDecodePath.exactCPU.exactCPUTrunk == .exactCPU)
    #expect(!MultiModelConfig.PreferredDecodePath.exactCPU.prefersHybridFamily)
}
