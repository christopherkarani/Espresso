import Testing
import ModelSupport
@testable import RealModelInference

private func makeDecodePathPolicyConfig(
    name: String = "llama3",
    architecture: MultiModelConfig.Architecture = .llama,
    preferredDecodePath: MultiModelConfig.PreferredDecodePath? = nil,
    nLayer: Int = 4
) -> MultiModelConfig {
    MultiModelConfig(
        name: name,
        nLayer: nLayer,
        nHead: 2,
        nKVHead: 2,
        dModel: 8,
        headDim: 4,
        hiddenDim: 16,
        vocab: 64,
        maxSeq: 8,
        normEps: 1e-5,
        architecture: architecture,
        preferredDecodePath: preferredDecodePath
    )
}

struct DecodePathPolicyTests {
    @Test func optionsFromEnvironmentParsesBoundedVariables() {
        let options = DecodePathPolicy.optionsFromEnvironment([
            "ESPRESSO_FORCE_HYBRID_DECODE": "1",
            "ESPRESSO_USE_CPU_EXACT_DECODE": "1",
            "ESPRESSO_REALMODEL_DISABLE_HYBRID_FALLBACK": "1",
            "DECODE_EVAL_FFN_ONLY": "1"
        ])
        #expect(options == DecodePathOptions(
            forceHybridDecode: true,
            useCPUExactDecode: true,
            disableHybridFallback: true,
            ffnOnlyEval: true
        ))
    }

    @Test func optionsFromEnvironmentIgnoresUnsetAndOtherValues() {
        #expect(DecodePathPolicy.optionsFromEnvironment([:]) == DecodePathOptions())
        for rawValue in ["0", "", "true", "2"] {
            let options = DecodePathPolicy.optionsFromEnvironment([
                "ESPRESSO_FORCE_HYBRID_DECODE": rawValue,
                "ESPRESSO_USE_CPU_EXACT_DECODE": rawValue,
                "ESPRESSO_REALMODEL_DISABLE_HYBRID_FALLBACK": rawValue,
                "DECODE_EVAL_FFN_ONLY": rawValue
            ])
            #expect(!options.forceHybridDecode)
            #expect(!options.useCPUExactDecode)
            #expect(!options.disableHybridFallback)
            #expect(!options.ffnOnlyEval)
        }
    }

    @Test func resolveSelectsTrunkPerExplicitOptions() {
        let llama = makeDecodePathPolicyConfig(name: "llama3")
        let qwenLegacy = makeDecodePathPolicyConfig(name: "Qwen3-0.6B")

        var cases: [(
            options: DecodePathOptions,
            fusedPreferred: Bool,
            expected: Trunk,
            config: MultiModelConfig
        )] = [
            // Plain llama: split hybrid by default.
            (DecodePathOptions(), false, .splitHybrid, llama),
            // Fused preference promotes the trunk.
            (DecodePathOptions(), true, .fusedHybrid, llama),
            // CPU-exact beats fused preference.
            (DecodePathOptions(useCPUExactDecode: true), true, .exactCPU, llama),
            // Force-hybrid rescues legacy Qwen off the exact-CPU oracle.
            (DecodePathOptions(forceHybridDecode: true), false, .splitHybrid, qwenLegacy),
            // Legacy Qwen routing lands on exact-CPU without overrides.
            (DecodePathOptions(), false, .exactCPU, qwenLegacy),
            // Declared artifact path wins over the name heuristic.
            (
                DecodePathOptions(),
                false,
                .splitHybrid,
                makeDecodePathPolicyConfig(name: "Qwen2.5-0.5B-Instruct", preferredDecodePath: .hybrid)
            ),
            (
                DecodePathOptions(),
                true,
                .fusedHybrid,
                makeDecodePathPolicyConfig(name: "llama-fused", preferredDecodePath: .hybrid)
            )
        ]

        for testCase in cases {
            let plan = DecodePathPolicy.resolve(
                config: testCase.config,
                fusedHybridPreferred: testCase.fusedPreferred,
                options: testCase.options
            )
            #expect(plan.trunk == testCase.expected)
        }
    }

    @Test func forceHybridBeatsCPUExactOption() {
        let plan = DecodePathPolicy.resolve(
            config: makeDecodePathPolicyConfig(),
            fusedHybridPreferred: true,
            options: DecodePathOptions(forceHybridDecode: true, useCPUExactDecode: true)
        )
        #expect(plan.trunk == .fusedHybrid)
    }

    @Test func resolveCarriesFallbackPolicyAndDispatchOption() {
        let config = makeDecodePathPolicyConfig()
        let rerunPlan = DecodePathPolicy.resolve(
            config: config,
            fusedHybridPreferred: false,
            options: DecodePathOptions(ffnOnlyEval: true)
        )
        #expect(rerunPlan.fallbackPolicy == .rerunOnBaseline)
        #expect(rerunPlan.allowsHybridFallback)
        #expect(rerunPlan.ffnOnlyEval)

        let disabledPlan = DecodePathPolicy.resolve(
            config: config,
            fusedHybridPreferred: false,
            options: DecodePathOptions(disableHybridFallback: true)
        )
        #expect(disabledPlan.fallbackPolicy == .disabled)
        #expect(!disabledPlan.allowsHybridFallback)
        #expect(!disabledPlan.ffnOnlyEval)
    }

    @Test func cpuExactRoutingSourceIsClassified() {
        let operatorPlan = DecodePathPolicy.resolve(
            config: makeDecodePathPolicyConfig(name: "llama3"),
            fusedHybridPreferred: false,
            options: DecodePathOptions(useCPUExactDecode: true)
        )
        #expect(operatorPlan.cpuExactRoutingSource == .operatorRequest)

        let artifactPlan = DecodePathPolicy.resolve(
            config: makeDecodePathPolicyConfig(preferredDecodePath: .exactCPU),
            fusedHybridPreferred: false,
            options: DecodePathOptions()
        )
        #expect(artifactPlan.cpuExactRoutingSource == .artifactDeclaration)

        let legacyPlan = DecodePathPolicy.resolve(
            config: makeDecodePathPolicyConfig(name: "Qwen3-0.6B"),
            fusedHybridPreferred: false,
            options: DecodePathOptions()
        )
        #expect(legacyPlan.cpuExactRoutingSource == .legacyQwenNameRouting)

        let hybridPlan = DecodePathPolicy.resolve(
            config: makeDecodePathPolicyConfig(name: "Qwen3-0.6B"),
            fusedHybridPreferred: false,
            options: DecodePathOptions(forceHybridDecode: true)
        )
        #expect(hybridPlan.cpuExactRoutingSource == nil)
    }

    @Test func explicitOptionsWinOverEnvironmentSeededDefaults() {
        // Edge case §9.1: env set AND explicit option passed → explicit wins;
        // env remains the fallback when no option is passed.
        let env = ["ESPRESSO_FORCE_HYBRID_DECODE": "1"]
        let seeded = DecodePathPolicy.optionsFromEnvironment(env)
        #expect(seeded.forceHybridDecode)

        let qwenLegacy = makeDecodePathPolicyConfig(name: "Qwen3-0.6B")
        #expect(
            DecodePathPolicy.resolve(config: qwenLegacy, fusedHybridPreferred: false, options: seeded).trunk != .exactCPU
        )

        let explicitOff = DecodePathOptions()
        #expect(
            DecodePathPolicy.resolve(config: qwenLegacy, fusedHybridPreferred: false, options: explicitOff).trunk == .exactCPU
        )
    }

    @Test func resolvedTrunkThrowsWhenFallbackDisabledOnExactCPU() throws {
        let qwenLegacy = makeDecodePathPolicyConfig(name: "Qwen3-0.6B")
        do {
            _ = try DecodePathPolicy.resolvedTrunk(
                config: qwenLegacy,
                fusedHybridPreferred: false,
                options: DecodePathOptions(disableHybridFallback: true)
            )
            Issue.record("Expected hybridFallbackDisabled")
        } catch let error as RealModelInferenceError {
            guard case let .hybridFallbackDisabled(stage, reason) = error else {
                Issue.record("Unexpected error shape: \(error)")
                return
            }
            #expect(stage == "llama trunk selection")
            #expect(reason.contains("legacy Qwen CPU-exact routing policy"))
            #expect(error.errorDescription?.contains("ESPRESSO_REALMODEL_DISABLE_HYBRID_FALLBACK=1") == true)
        }
    }

    @Test func resolvedTrunkReasonNamesCPUExactOption() throws {
        do {
            _ = try DecodePathPolicy.resolvedTrunk(
                config: makeDecodePathPolicyConfig(name: "llama3"),
                fusedHybridPreferred: false,
                options: DecodePathOptions(useCPUExactDecode: true, disableHybridFallback: true)
            )
            Issue.record("Expected hybridFallbackDisabled")
        } catch let error as RealModelInferenceError {
            guard case let .hybridFallbackDisabled(_, reason) = error else {
                Issue.record("Unexpected error shape: \(error)")
                return
            }
            #expect(reason.contains("ESPRESSO_USE_CPU_EXACT_DECODE"))
        }
    }

    @Test func resolvedTrunkReasonNamesDeclaredArtifactPath() throws {
        do {
            _ = try DecodePathPolicy.resolvedTrunk(
                config: makeDecodePathPolicyConfig(preferredDecodePath: .exactCPU),
                fusedHybridPreferred: false,
                options: DecodePathOptions(disableHybridFallback: true)
            )
            Issue.record("Expected hybridFallbackDisabled")
        } catch let error as RealModelInferenceError {
            guard case let .hybridFallbackDisabled(_, reason) = error else {
                Issue.record("Unexpected error shape: \(error)")
                return
            }
            #expect(reason.contains("preferredDecodePath=exact_cpu in metadata.json"))
        }
    }

    @Test func resolvedTrunkKeepsHybridWhenFallbackDisabled() throws {
        let trunk = try DecodePathPolicy.resolvedTrunk(
            config: makeDecodePathPolicyConfig(name: "llama3"),
            fusedHybridPreferred: true,
            options: DecodePathOptions(disableHybridFallback: true)
        )
        #expect(trunk.isHybrid)
    }
}
