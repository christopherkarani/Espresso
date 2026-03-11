import Foundation

public struct ANEOptionSnapshot: Equatable {
    let compileCachePolicy: String
    let evalPath: String
    let strictOptions: Bool
    let disablePowerSaving: Bool
    let keepModelWired: Bool
    let enableLateLatch: Bool
    let skipPrepare: Bool
    let disableIOFences: Bool
    let enableFWToFWSignal: Bool
    let useCompilerOptions: Bool
    let perfStatsRequested: Bool
    let queueDepth: String?
    let memoryPoolID: String?
    let perfStatsMask: String?
    let decodeLaneSpatial: Int
    let benchSeed: String

    public static func fromEnvironment(_ env: [String: String]) -> Self {
        Self(
            compileCachePolicy: env["ANE_COMPILE_CACHE_POLICY"] ?? "auto",
            evalPath: env["ANE_EVAL_PATH"] ?? "inmem",
            strictOptions: env["ANE_STRICT_OPTIONS"] == "1",
            disablePowerSaving: env["ANE_DISABLE_POWER_SAVING"] == "1",
            keepModelWired: env["ANE_KEEP_MODEL_WIRED"] == "1",
            enableLateLatch: env["ANE_ENABLE_LATE_LATCH"] == "1",
            skipPrepare: env["ANE_SKIP_PREPARE"] == "1",
            disableIOFences: env["ANE_DISABLE_IO_FENCES"] == "1",
            enableFWToFWSignal: env["ANE_ENABLE_FW_TO_FW_SIGNAL"] == "1",
            useCompilerOptions: env["ANE_USE_COMPILER_OPTIONS"] == "1",
            perfStatsRequested: env["ANE_PERF_STATS"] == "1",
            queueDepth: nonEmpty(env["ANE_QUEUE_DEPTH"]),
            memoryPoolID: nonEmpty(env["ANE_MEMORY_POOL_ID"]),
            perfStatsMask: nonEmpty(env["ANE_PERF_STATS_MASK"]),
            decodeLaneSpatial: max(32, env["ESPRESSO_DECODE_LANE_SPATIAL"].flatMap(Int.init) ?? 32),
            benchSeed: env["ESPRESSO_BENCH_SEED"] ?? ""
        )
    }

    public func asJSON() -> [String: Any] {
        var out: [String: Any] = [
            "compile_cache_policy": compileCachePolicy,
            "eval_path": evalPath,
            "strict_options": strictOptions,
            "disable_power_saving": disablePowerSaving,
            "keep_model_wired": keepModelWired,
            "enable_late_latch": enableLateLatch,
            "skip_prepare": skipPrepare,
            "disable_io_fences": disableIOFences,
            "enable_fw_to_fw_signal": enableFWToFWSignal,
            "use_compiler_options": useCompilerOptions,
            "perf_stats_requested": perfStatsRequested,
            "decode_lane_spatial": decodeLaneSpatial,
            "bench_seed": benchSeed,
        ]
        if let queueDepth {
            out["queue_depth"] = queueDepth
        }
        if let memoryPoolID {
            out["memory_pool_id"] = memoryPoolID
        }
        if let perfStatsMask {
            out["perf_stats_mask"] = perfStatsMask
        }
        return out
    }

    private static func nonEmpty(_ value: String?) -> String? {
        guard let value, !value.isEmpty else { return nil }
        return value
    }
}
