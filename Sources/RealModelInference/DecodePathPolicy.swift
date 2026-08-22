import ModelSupport

/// Typed steering inputs for decode-path decisions.
///
/// Replaces direct process-environment reads scattered through the engine:
/// every production decode-path decision starts from one snapshot of these
/// options, produced either explicitly by an embedding host (CLI, benchmarks)
/// or once at the bootstrap seam via ``DecodePathPolicy/optionsFromEnvironment(_:)``.
public struct DecodePathOptions: Sendable, Equatable {
    /// `ESPRESSO_FORCE_HYBRID_DECODE=1` — steer legacy CPU-routed artifacts onto an ANE trunk.
    public var forceHybridDecode: Bool
    /// `ESPRESSO_USE_CPU_EXACT_DECODE=1` — request the exact-CPU trunk outright.
    public var useCPUExactDecode: Bool
    /// `ESPRESSO_REALMODEL_DISABLE_HYBRID_FALLBACK=1` — refuse to leave the selected ANE
    /// trunk; throw instead of downgrading.
    public var disableHybridFallback: Bool
    /// `DECODE_EVAL_FFN_ONLY=1` — decode-step kernels skip attention work (eval harness).
    public var ffnOnlyEval: Bool

    public init(
        forceHybridDecode: Bool = false,
        useCPUExactDecode: Bool = false,
        disableHybridFallback: Bool = false,
        ffnOnlyEval: Bool = false
    ) {
        self.forceHybridDecode = forceHybridDecode
        self.useCPUExactDecode = useCPUExactDecode
        self.disableHybridFallback = disableHybridFallback
        self.ffnOnlyEval = ffnOnlyEval
    }
}

/// What happens when a selected trunk cannot be served.
public enum HybridFallbackPolicy: Sendable, Equatable {
    /// Rerun the work on the lower trunk (hybrid fast-path failure falls back to baseline).
    case rerunOnBaseline
    /// Refuse to leave the selected trunk; raise `hybridFallbackDisabled` instead.
    case disabled
}

/// Why a resolution landed on the exact-CPU trunk. Drives actionable errors when
/// fallback to CPU is forbidden.
public enum ExactCPURoutingSource: Sendable, Equatable {
    /// `ESPRESSO_USE_CPU_EXACT_DECODE=1` asked for it.
    case operatorRequest
    /// The artifact declares `preferredDecodePath = exact_cpu` in metadata.json.
    case artifactDeclaration
    /// Legacy name-based routing for early Qwen artifacts without a declared path.
    case legacyQwenNameRouting
}

/// The outcome of resolving decode-path policy for one llama serving session:
/// the selected trunk, its fallback behavior, and the dispatch options consumed
/// while running decode steps.
public struct ResolvedDecodePlan: Sendable, Equatable {
    /// Selected trunk: fused hybrid, split hybrid, or exact-CPU.
    public let trunk: Trunk
    /// Fallback behavior when the selected trunk cannot be served.
    public let fallbackPolicy: HybridFallbackPolicy
    /// Dispatch option handed to decode-step kernels.
    public let ffnOnlyEval: Bool
    /// Set when `trunk` is `.exactCPU`: which routing rule chose it.
    public let cpuExactRoutingSource: ExactCPURoutingSource?

    init(
        trunk: Trunk,
        fallbackPolicy: HybridFallbackPolicy,
        ffnOnlyEval: Bool,
        cpuExactRoutingSource: ExactCPURoutingSource?
    ) {
        self.trunk = trunk
        self.fallbackPolicy = fallbackPolicy
        self.ffnOnlyEval = ffnOnlyEval
        self.cpuExactRoutingSource = cpuExactRoutingSource
    }

    public var allowsHybridFallback: Bool {
        fallbackPolicy == .rerunOnBaseline
    }
}

/// Pure decode-path decision logic: `(config, options)` → selected trunk +
/// fallback policy + dispatch options. No process-global state is consulted;
/// callers bootstrap options once via ``optionsFromEnvironment(_:)`` and pass
/// explicit values wherever a host knows better than the environment.
public enum DecodePathPolicy {
    /// The single bootstrap seam: parse decode-path steering from raw
    /// environment values. Explicit options beat these values; these values
    /// remain the fallback when no option is set.
    public static func optionsFromEnvironment(_ environment: [String: String]) -> DecodePathOptions {
        DecodePathOptions(
            forceHybridDecode: environment["ESPRESSO_FORCE_HYBRID_DECODE"] == "1",
            useCPUExactDecode: environment["ESPRESSO_USE_CPU_EXACT_DECODE"] == "1",
            disableHybridFallback: environment["ESPRESSO_REALMODEL_DISABLE_HYBRID_FALLBACK"] == "1",
            ffnOnlyEval: environment["DECODE_EVAL_FFN_ONLY"] == "1"
        )
    }

    /// Whether the exact-CPU trunk is preferred for this configuration.
    ///
    /// Precedence mirrors the serving contract: an explicit hybrid force wins,
    /// then an explicit CPU request, then the artifact's declared
    /// `preferredDecodePath`, then legacy name-based Qwen routing.
    public static func prefersCPUExactDecode(
        config: MultiModelConfig,
        options: DecodePathOptions
    ) -> Bool {
        if options.forceHybridDecode {
            return false
        }
        if options.useCPUExactDecode {
            return true
        }
        guard config.architecture == .llama else {
            return false
        }
        // An artifact that states where it decodes is trusted over the name heuristic below.
        if let declared = config.preferredDecodePath {
            return declared == .exactCPU
        }
        // Legacy routing: early Qwen artifacts predate a working ANE path at these widths
        // and are kept on the CPU oracle so old bundles do not change behaviour.
        return ModelFamily.isQwenVariant(config)
    }

    /// Resolve the serving plan for a configuration under the given options.
    ///
    /// `fusedHybridPreferred` carries the fused-vs-split hybrid preference whose
    /// own steering variables live outside this module's scope; the policy owns
    /// the selection order (exact-CPU first, then fused hybrid, then split hybrid),
    /// the fallback policy, and the dispatch options.
    public static func resolve(
        config: MultiModelConfig,
        fusedHybridPreferred: Bool,
        options: DecodePathOptions
    ) -> ResolvedDecodePlan {
        let fallbackPolicy: HybridFallbackPolicy = options.disableHybridFallback ? .disabled : .rerunOnBaseline
        if let cpuSource = cpuExactRoutingSource(config: config, options: options) {
            return ResolvedDecodePlan(
                trunk: .exactCPU,
                fallbackPolicy: fallbackPolicy,
                ffnOnlyEval: options.ffnOnlyEval,
                cpuExactRoutingSource: cpuSource
            )
        }
        return ResolvedDecodePlan(
            trunk: fusedHybridPreferred ? .fusedHybrid : .splitHybrid,
            fallbackPolicy: fallbackPolicy,
            ffnOnlyEval: options.ffnOnlyEval,
            cpuExactRoutingSource: nil
        )
    }

    private static func cpuExactRoutingSource(
        config: MultiModelConfig,
        options: DecodePathOptions
    ) -> ExactCPURoutingSource? {
        guard prefersCPUExactDecode(config: config, options: options) else {
            return nil
        }
        if options.useCPUExactDecode {
            return .operatorRequest
        }
        if config.preferredDecodePath == .exactCPU {
            return .artifactDeclaration
        }
        return .legacyQwenNameRouting
    }

    /// Resolves the llama serving ``Trunk``, refusing to leave the ANE silently.
    ///
    /// With fallback disabled, landing on the pure-CPU trunk is a failure rather than a
    /// quiet downgrade, and the thrown error names which policy chose CPU so the cause is
    /// actionable instead of mysterious.
    public static func resolvedTrunk(
        config: MultiModelConfig,
        fusedHybridPreferred: Bool,
        options: DecodePathOptions
    ) throws -> Trunk {
        let plan = resolve(config: config, fusedHybridPreferred: fusedHybridPreferred, options: options)
        guard plan.trunk == .exactCPU, plan.fallbackPolicy == .disabled else {
            return plan.trunk
        }
        let reason: String
        switch plan.cpuExactRoutingSource {
        case .operatorRequest:
            reason = "ESPRESSO_USE_CPU_EXACT_DECODE=1 explicitly requests the pure-CPU decode path"
        case .artifactDeclaration:
            reason = "the artifact declares preferredDecodePath=exact_cpu in metadata.json"
        case .legacyQwenNameRouting, nil:
            reason = """
                model name "\(config.name)" matches the legacy Qwen CPU-exact routing policy \
                and metadata.json does not declare preferredDecodePath; set \
                preferredDecodePath=hybrid or ESPRESSO_FORCE_HYBRID_DECODE=1 to decode on the ANE
                """
        }
        throw RealModelInferenceError.hybridFallbackDisabled(
            stage: "llama trunk selection",
            reason: reason
        )
    }
}
