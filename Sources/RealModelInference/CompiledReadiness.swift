import IOSurface

/// Per-trunk compile readiness for the current engine session.
///
/// Replaces the former `compiled*` flag-and-conjunction family: a trunk is
/// either `.notCompiled`, or `.compiled(runtime)` where the runtime was
/// captured only after every program that trunk's decode loop requires became
/// resident. Readiness checks switch over this enum exhaustively instead of
/// re-deriving flag counts at each use site, and engine state transitions go
/// through it — the ensure functions are the only writers.
enum CompiledReadiness<Runtime> {
    /// No programs are resident for this trunk yet.
    case notCompiled
    /// The trunk's validated runtime is resident and covers `runtime`'s bucket.
    case compiled(Runtime)

    /// The resident runtime, or `nil` when nothing is compiled.
    var runtime: Runtime? {
        switch self {
        case .compiled(let runtime): runtime
        case .notCompiled: nil
        }
    }
}

/// Programs required by the baseline (exact-CPU-routed ANE eval) decode loop.
///
/// Constructed only after every program the loop needs is resident, so a
/// `.compiled` value proves the loop's former guard conjunction.
struct BaselineCompiledRuntime {
    /// Context bucket the programs were compiled for.
    let bucket: Int
    /// First-layer input surface the loop writes embeddings into.
    let inputSurface: IOSurfaceRef
}

/// Programs required by the split-hybrid decode loop
/// (ANE QKV → host attention → ANE FFN), shared by GPT-2 and llama sessions.
struct SplitHybridCompiledRuntime {
    /// Context bucket the programs were compiled for.
    let bucket: Int
    /// Output-head lane spatial; `> 0` once the head program is resident.
    let headSpatial: Int

    /// Captures the resident split-hybrid program facts, or `nil` unless every
    /// program the loop requires is present. Llama sessions additionally
    /// require one QK-norm weight entry per layer; pass `qKNormCount: nil` for
    /// architectures without them.
    init?(
        bucket: Int,
        layerCount: Int,
        surfaceHandleCount: Int,
        expectedLayerCount: Int,
        headCount: Int,
        qKNormCount: Int? = nil,
        headSpatial: Int
    ) {
        guard layerCount == expectedLayerCount,
              surfaceHandleCount == expectedLayerCount,
              headCount == 1,
              headSpatial > 0 else {
            return nil
        }
        if let qKNormCount, qKNormCount != expectedLayerCount {
            return nil
        }
        self.bucket = bucket
        self.headSpatial = headSpatial
    }
}

/// Programs required by the fused-hybrid decode loop (one ANE program per layer).
struct FusedHybridCompiledRuntime {
    /// Context bucket the programs were compiled for.
    let bucket: Int

    /// Captures the resident fused-hybrid program facts, or `nil` unless every
    /// program the loop requires is present.
    init?(
        bucket: Int,
        layerCount: Int,
        surfaceHandleCount: Int,
        expectedLayerCount: Int
    ) {
        guard layerCount == expectedLayerCount,
              surfaceHandleCount == expectedLayerCount else {
            return nil
        }
        self.bucket = bucket
    }
}
