import Testing
@testable import ModelSupport

@Test func nucleusSamplerGreedyIgnoresTopP() {
    let logits: [Float] = [1, 5, 2]
    #expect(NucleusSampler.selectIndex(logits: logits, temperature: 0, topP: 0.1, unitSample: 0.99) == 1)
}

@Test func nucleusSamplerTopPDropsLowMassTail() {
    // After temperature=1 softmax the first logit dominates; top-p 0.6 keeps only index 0.
    let logits: [Float] = [8, 0, 0]
    #expect(NucleusSampler.selectIndex(logits: logits, temperature: 1, topP: 0.6, unitSample: 0.99) == 0)
}

@Test func nucleusSamplerFullDistributionCanSelectSecondMass() {
    let logits: [Float] = [0, 0]
    #expect(NucleusSampler.selectIndex(logits: logits, temperature: 1, topP: 1, unitSample: 0.75) == 1)
}
