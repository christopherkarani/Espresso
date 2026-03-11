import Foundation
import ANETypes
import Espresso

enum ResultsFormatter {
    static func chipName() -> String {
        func readSysctl(_ name: String) -> String? {
            var size: size_t = 0
            guard sysctlbyname(name, nil, &size, nil, 0) == 0, size > 0 else { return nil }
            var buf = [UInt8](repeating: 0, count: size)
            guard sysctlbyname(name, &buf, &size, nil, 0) == 0 else { return nil }
            // Trim null terminator
            if let nullIdx = buf.firstIndex(of: 0) { buf = Array(buf[..<nullIdx]) }
            return String(decoding: buf, as: UTF8.self)
        }
        // Intel path
        if let brand = readSysctl("machdep.cpu.brand_string") { return brand }
        // Apple Silicon: hw.model gives Mac model identifier
        if let model = readSysctl("hw.model") { return model }
        return "Unknown"
    }

    static func formatReport(
        aneResult: BenchmarkResult,
        aneTimingBreakdown: (ane: Double, io: Double, elem: Double)?,
        compileTimeMs: Double?,
        inferenceResult: BenchmarkResult? = nil,
        inferenceTimingBreakdown: (ane: Double, io: Double, elem: Double)? = nil,
        inferenceCompileTimeMs: Double? = nil,
        coreMLResults: [(label: String, result: BenchmarkResult)]?,
        coreMLLoadTimeMs: Double?,
        thermalBefore: String?,
        thermalAfter: String?,
        flopsPerPass: Double,
        nLayers: Int
    ) -> String {
        var out = ""
        let chip = chipName()
        let peakTFLOPS = 18.0

        out += "=== ANE DIRECT BENCHMARK (TRAINING FORWARD) ===\n"
        out += "Chip: \(chip)\n"
        out += String(format: "ANE Peak: %.1f TFLOPS\n", peakTFLOPS)
        out += "Workload: \(nLayers)-layer transformer, dim=\(ModelConfig.dim), "
        out += "seq=\(ModelConfig.seqLen), heads=\(ModelConfig.heads), hidden=\(ModelConfig.hidden)\n"
        out += String(format: "FLOPs per forward pass: %.2f GFLOPs\n", flopsPerPass / 1e9)
        if let compileMs = compileTimeMs {
            out += String(format: "Kernel compilation: %.1f ms (%d kernels)\n", compileMs, nLayers * 5)
        }
        out += "\n"

        out += formatLatencySection(aneResult, flopsPerPass: flopsPerPass, peakTFLOPS: peakTFLOPS)

        if let breakdown = aneTimingBreakdown {
            let total = breakdown.ane + breakdown.io + breakdown.elem
            if total > 0 {
                out += "--- Time Breakdown (avg per forward pass) ---\n"
                out += String(format: "ANE kernel:    %.3f ms (%.1f%%)\n", breakdown.ane, breakdown.ane / total * 100)
                out += String(format: "Surface I/O:   %.3f ms (%.1f%%)\n", breakdown.io, breakdown.io / total * 100)
                out += String(format: "CPU element:   %.3f ms (%.1f%%)\n\n", breakdown.elem, breakdown.elem / total * 100)
            }
        }

        // Inference-optimized results
        if let infResult = inferenceResult {
            out += "=== ANE DIRECT BENCHMARK (INFERENCE, FUSED RESIDUALS) ===\n"
            if let compileMs = inferenceCompileTimeMs {
                out += String(format: "Kernel compilation: %.1f ms (%d kernels)\n", compileMs, nLayers * 2)
            }
            out += "\n"

            out += formatLatencySection(infResult, flopsPerPass: flopsPerPass, peakTFLOPS: peakTFLOPS)

            if let breakdown = inferenceTimingBreakdown {
                let total = breakdown.ane + breakdown.io + breakdown.elem
                if total > 0 {
                    out += "--- Time Breakdown (avg per forward pass) ---\n"
                    out += String(format: "ANE kernel:    %.3f ms (%.1f%%)\n", breakdown.ane, breakdown.ane / total * 100)
                    out += String(format: "Surface I/O:   %.3f ms (%.1f%%)\n", breakdown.io, breakdown.io / total * 100)
                    out += String(format: "CPU element:   %.3f ms (%.1f%%)\n\n", breakdown.elem, breakdown.elem / total * 100)
                }
            }

            // Comparison: training vs inference
            let speedup = aneResult.median / infResult.median
            let savings = aneResult.median - infResult.median
            out += "--- Training vs Inference ---\n"
            out += String(format: "Training median:  %.3f ms\n", aneResult.median)
            out += String(format: "Inference median: %.3f ms\n", infResult.median)
            out += String(format: "Speedup: %.2fx (%.3f ms saved)\n\n", speedup, savings)
        }

        if let coreMLResults {
            for (label, result) in coreMLResults {
                out += "=== CORE ML BASELINE (\(label)) ===\n"
                if let loadTime = coreMLLoadTimeMs, label.contains("all") {
                    out += String(format: "Model load time: %.1f ms\n", loadTime)
                }
                out += formatLatencySection(result, flopsPerPass: flopsPerPass, peakTFLOPS: peakTFLOPS)

                let trainingSpeedup = result.median / aneResult.median
                out += "--- vs ANE Direct (Training) ---\n"
                out += String(format: "Speedup (ANE Training vs this): %.2fx\n", trainingSpeedup)
                if let infResult = inferenceResult {
                    let inferenceSpeedup = result.median / infResult.median
                    out += String(format: "Speedup (ANE Inference vs this): %.2fx\n", inferenceSpeedup)
                }
                out += "\n"
            }
        }

        if let before = thermalBefore, let after = thermalAfter {
            out += "=== POWER & THERMAL ===\n"
            out += "Thermal state: \(before) -> \(after)\n\n"
        }

        return out
    }

    static func formatInferenceOnlyReport(
        inferenceResult: BenchmarkResult,
        inferenceTimingBreakdown: (ane: Double, io: Double, elem: Double)?,
        inferenceCompileTimeMs: Double?,
        coreMLResults: [(label: String, result: BenchmarkResult)]?,
        coreMLLoadTimeMs: Double?,
        flopsPerPass: Double,
        nLayers: Int,
        thermalBefore: String? = nil,
        thermalAfter: String? = nil
    ) -> String {
        var out = ""
        let chip = chipName()
        let peakTFLOPS = 18.0

        out += "=== ANE DIRECT BENCHMARK (INFERENCE ONLY) ===\n"
        out += "Chip: \(chip)\n"
        out += String(format: "ANE Peak: %.1f TFLOPS\n", peakTFLOPS)
        out += "Workload: \(nLayers)-layer transformer, dim=\(ModelConfig.dim), "
        out += "seq=\(ModelConfig.seqLen), heads=\(ModelConfig.heads), hidden=\(ModelConfig.hidden)\n"
        out += String(format: "FLOPs per forward pass: %.2f GFLOPs\n", flopsPerPass / 1e9)
        if let compileMs = inferenceCompileTimeMs {
            out += String(format: "Kernel compilation: %.1f ms (%d kernels)\n", compileMs, nLayers * 2)
        }
        out += "\n"

        out += formatLatencySection(inferenceResult, flopsPerPass: flopsPerPass, peakTFLOPS: peakTFLOPS)

        if let breakdown = inferenceTimingBreakdown {
            let total = breakdown.ane + breakdown.io + breakdown.elem
            if total > 0 {
                out += "--- Time Breakdown (avg per forward pass) ---\n"
                out += String(format: "ANE kernel:    %.3f ms (%.1f%%)\n", breakdown.ane, breakdown.ane / total * 100)
                out += String(format: "Surface I/O:   %.3f ms (%.1f%%)\n", breakdown.io, breakdown.io / total * 100)
                out += String(format: "CPU element:   %.3f ms (%.1f%%)\n\n", breakdown.elem, breakdown.elem / total * 100)
            }
        }

        if let coreMLResults {
            for (label, result) in coreMLResults {
                out += "=== CORE ML BASELINE (\(label)) ===\n"
                if let loadTime = coreMLLoadTimeMs, label.contains("all") {
                    out += String(format: "Model load time: %.1f ms\n", loadTime)
                }
                out += formatLatencySection(result, flopsPerPass: flopsPerPass, peakTFLOPS: peakTFLOPS)
                let speedup = result.median / inferenceResult.median
                out += "--- vs ANE Direct (Inference) ---\n"
                out += String(format: "Speedup (ANE vs this): %.2fx\n\n", speedup)
            }
        }

        if let before = thermalBefore, let after = thermalAfter {
            out += "=== POWER & THERMAL ===\n"
            out += "Thermal state: \(before) -> \(after)\n\n"
        }

        return out
    }

    static func formatDecodeReport(
        decodeResult: BenchmarkResult,
        decodeTimingBreakdown: (ane: Double, io: Double, elem: Double)?,
        decodeCompileTimeMs: Double?,
        decodeTokensPerSecond: Double?,
        coreMLDecodeResults: [(label: String, result: BenchmarkResult)]?,
        coreMLLoadTimeMs: Double?,
        nLayers: Int,
        decodeSteps: Int,
        decodeMaxSeq: Int
    ) -> String {
        var out = ""
        let chip = chipName()

        out += "=== ANE DIRECT BENCHMARK (AUTOREGRESSIVE DECODE) ===\n"
        out += "Chip: \(chip)\n"
        out += "Workload: \(nLayers)-layer transformer decode, dim=\(ModelConfig.dim), heads=\(ModelConfig.heads), hidden=\(ModelConfig.hidden)\n"
        out += "Decode schedule: steps=\(decodeSteps), maxSeq=\(decodeMaxSeq)\n"
        if let compileMs = decodeCompileTimeMs {
            out += String(format: "Kernel compilation: %.1f ms (%d kernels)\n", compileMs, nLayers * 2)
        }
        out += "\n"

        out += "--- Token Latency (\(decodeResult.iterationCount) tokens, \(decodeResult.warmupCount) warmup tokens) ---\n"
        out += String(format: "Mean:    %.3f ms\n", decodeResult.mean)
        out += String(format: "Median:  %.3f ms\n", decodeResult.median)
        out += String(format: "P95:     %.3f ms\n", decodeResult.p95)
        out += String(format: "P99:     %.3f ms\n", decodeResult.p99)
        out += String(format: "Stddev:  %.3f ms\n", decodeResult.stddev)
        out += String(format: "Min:     %.3f ms\n", decodeResult.min)
        out += String(format: "Max:     %.3f ms\n\n", decodeResult.max)

        let aneTPS = decodeTokensPerSecond ?? (decodeResult.mean > 0 ? 1000.0 / decodeResult.mean : 0)
        out += "--- Throughput ---\n"
        out += String(format: "Tokens/sec: %.1f\n\n", aneTPS)

        if let breakdown = decodeTimingBreakdown {
            let total = breakdown.ane + breakdown.io + breakdown.elem
            if total > 0 {
                out += "--- Time Breakdown (avg per token) ---\n"
                out += String(format: "ANE kernel:    %.3f ms (%.1f%%)\n", breakdown.ane, breakdown.ane / total * 100)
                out += String(format: "Surface I/O:   %.3f ms (%.1f%%)\n", breakdown.io, breakdown.io / total * 100)
                out += String(format: "CPU element:   %.3f ms (%.1f%%)\n\n", breakdown.elem, breakdown.elem / total * 100)
            }
        }

        if let coreMLDecodeResults {
            let fastest = coreMLDecodeResults.min { $0.result.median < $1.result.median }?.result
            for (label, result) in coreMLDecodeResults {
                out += "=== CORE ML NAIVE DECODE (\(label)) ===\n"
                if let loadTime = coreMLLoadTimeMs, label.contains(".all") {
                    out += String(format: "Model load time: %.1f ms\n", loadTime)
                }
                out += String(format: "Mean:    %.3f ms/token\n", result.mean)
                out += String(format: "Median:  %.3f ms/token\n", result.median)
                out += String(format: "P95:     %.3f ms/token\n", result.p95)
                out += String(format: "Tokens/sec: %.1f\n", result.mean > 0 ? 1000.0 / result.mean : 0)
                let speedup = result.median / decodeResult.median
                out += String(format: "Speedup (ANE vs this): %.2fx\n\n", speedup)
            }
            if let fastest {
                let strictSpeedup = fastest.median / decodeResult.median
                out += "--- Strict Gate (Fastest Core ML Naive Decode) ---\n"
                out += String(format: "Fastest Core ML median: %.3f ms/token\n", fastest.median)
                out += String(format: "ANE median: %.3f ms/token\n", decodeResult.median)
                out += String(format: "Speedup: %.2fx\n\n", strictSpeedup)
            }
        }

        return out
    }

    private static func formatLatencySection(
        _ result: BenchmarkResult,
        flopsPerPass: Double,
        peakTFLOPS: Double
    ) -> String {
        var out = ""
        let sustained = FLOPCalculator.sustainedTFLOPS(flops: flopsPerPass, latencyMs: result.median)
        let utilPct = FLOPCalculator.aneUtilization(sustainedTFLOPS: sustained, peakTFLOPS: peakTFLOPS)
        let fwdPerSec = 1000.0 / result.median

        out += "--- Latency (\(result.iterationCount) iterations, \(result.warmupCount) warmup) ---\n"
        out += String(format: "Mean:    %.3f ms\n", result.mean)
        out += String(format: "Median:  %.3f ms\n", result.median)
        out += String(format: "P95:     %.3f ms\n", result.p95)
        out += String(format: "P99:     %.3f ms\n", result.p99)
        out += String(format: "Stddev:  %.3f ms\n", result.stddev)
        out += String(format: "Min:     %.3f ms\n", result.min)
        out += String(format: "Max:     %.3f ms\n\n", result.max)

        out += "--- Throughput ---\n"
        out += String(format: "Sustained TFLOPS:   %.4f\n", sustained)
        out += String(format: "ANE Utilization:    %.1f%%\n", utilPct)
        out += String(format: "Forward passes/sec: %.0f\n\n", fwdPerSec)

        return out
    }

    static func writeCSV(latencies: [Double], to path: String) throws {
        let header = "iteration,latency_ms\n"
        let rows = latencies.enumerated().map { "\($0.offset),\(String(format: "%.6f", $0.element))" }
        let content = header + rows.joined(separator: "\n") + "\n"
        try content.write(toFile: path, atomically: true, encoding: .utf8)
    }

    static func writeInferenceKernelProfileCSV(profile: InferenceKernelProfile, to path: String) throws {
        let posix = Locale(identifier: "en_US_POSIX")
        var out = "layer,iteration,attn_write_us,attn_write_lock_us,attn_write_body_us,attn_write_unlock_us,attn_eval_us,attn_hw_ns,attn_hw_us,attn_host_overhead_us,attn_read_us,attn_read_lock_us,attn_read_body_us,attn_read_unlock_us,gap_attn_to_ffn_us,ffn_write_us,ffn_write_lock_us,ffn_write_body_us,ffn_write_unlock_us,ffn_copy_us,ffn_eval_us,ffn_hw_ns,ffn_hw_us,ffn_host_overhead_us,ffn_read_us,ffn_read_lock_us,ffn_read_body_us,ffn_read_unlock_us\n"

        for (layerIdx, layer) in profile.layers.enumerated() {
            let n = layer.attnWriteUS.count
            precondition(layer.attnWriteLockUS.count == n)
            precondition(layer.attnWriteBodyUS.count == n)
            precondition(layer.attnWriteUnlockUS.count == n)
            precondition(layer.attnEvalUS.count == n)
            precondition(layer.attnHwNS.count == n)
            precondition(layer.attnHostOverheadUS.count == n)
            precondition(layer.attnReadUS.count == n)
            precondition(layer.attnReadLockUS.count == n)
            precondition(layer.attnReadBodyUS.count == n)
            precondition(layer.attnReadUnlockUS.count == n)
            precondition(layer.ffnWriteUS.count == n)
            precondition(layer.ffnWriteLockUS.count == n)
            precondition(layer.ffnWriteBodyUS.count == n)
            precondition(layer.ffnWriteUnlockUS.count == n)
            precondition(layer.ffnCopyUS.count == n)
            precondition(layer.ffnEvalUS.count == n)
            precondition(layer.ffnHwNS.count == n)
            precondition(layer.ffnHostOverheadUS.count == n)
            precondition(layer.ffnReadUS.count == n)
            precondition(layer.ffnReadLockUS.count == n)
            precondition(layer.ffnReadBodyUS.count == n)
            precondition(layer.ffnReadUnlockUS.count == n)
            precondition(layer.gapAttnToFfnUS.count == n)

            for i in 0..<n {
                let attnHwUS = Double(layer.attnHwNS[i]) / 1_000.0
                let ffnHwUS = Double(layer.ffnHwNS[i]) / 1_000.0
                out += String(
                    format: "%d,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%llu,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%llu,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n",
                    locale: posix,
                    layerIdx,
                    i,
                    layer.attnWriteUS[i],
                    layer.attnWriteLockUS[i],
                    layer.attnWriteBodyUS[i],
                    layer.attnWriteUnlockUS[i],
                    layer.attnEvalUS[i],
                    layer.attnHwNS[i],
                    attnHwUS,
                    layer.attnHostOverheadUS[i],
                    layer.attnReadUS[i],
                    layer.attnReadLockUS[i],
                    layer.attnReadBodyUS[i],
                    layer.attnReadUnlockUS[i],
                    layer.gapAttnToFfnUS[i],
                    layer.ffnWriteUS[i],
                    layer.ffnWriteLockUS[i],
                    layer.ffnWriteBodyUS[i],
                    layer.ffnWriteUnlockUS[i],
                    layer.ffnCopyUS[i],
                    layer.ffnEvalUS[i],
                    layer.ffnHwNS[i],
                    ffnHwUS,
                    layer.ffnHostOverheadUS[i],
                    layer.ffnReadUS[i],
                    layer.ffnReadLockUS[i],
                    layer.ffnReadBodyUS[i],
                    layer.ffnReadUnlockUS[i]
                )
            }
        }

        try out.write(toFile: path, atomically: true, encoding: .utf8)
    }

    static func formatInferenceKernelProfileSummaryTable(
        profile: InferenceKernelProfile,
        handoff: ForwardPass.InferenceInterKernelHandoff
    ) -> String {
        guard !profile.layers.isEmpty else { return "" }
        let hasAnyHWExecutionTime = profile.layers.contains { layer in
            layer.attnHwNS.contains(where: { $0 > 0 }) || layer.ffnHwNS.contains(where: { $0 > 0 })
        }
        let handoffLabel: String
        switch handoff {
        case .cpuRoundTrip:
            handoffLabel = "cpuRoundTrip"
        case .fp16SurfaceCopy:
            handoffLabel = "fp16SurfaceCopy"
        }
        var out = ""
        out += "--- Inference Per-Layer Breakdown (avg us) ---\n"
        out += "Columns: host_eval_us, hw_exec_us, host_overhead_us, io_lock_us, io_body_us, io_unlock_us\n"
        out += "Handoff mode: \(handoffLabel)\n"
        if !hasAnyHWExecutionTime {
            out += "hwExecutionTime unavailable on this host (perf stats not provided by driver/runtime); hw columns remain 0.0\n"
        }
        out += "layer | attn(host/hw/over) | attn_io(lock/body/unlock) | gap | handoff(cpu/copy) | ffn(host/hw/over) | ffn_io(lock/body/unlock)\n"

        for idx in profile.layers.indices {
            let mean = profile.averageLayerMetrics(layerIndex: idx)
            out += String(
                format: "L%-3d | %.1f / %.1f / %.1f | %.1f / %.1f / %.1f | %.1f | %.1f / %.1f | %.1f / %.1f / %.1f | %.1f / %.1f / %.1f\n",
                idx,
                mean.attnEvalUS,
                mean.attnHwUS,
                mean.attnHostOverheadUS,
                mean.attnIOLockUS,
                mean.attnIOBodyUS,
                mean.attnIOUnlockUS,
                mean.gapAttnToFfnUS,
                mean.handoffCPUUS,
                mean.handoffFP16CopyUS,
                mean.ffnEvalUS,
                mean.ffnHwUS,
                mean.ffnHostOverheadUS,
                mean.ffnIOLockUS,
                mean.ffnIOBodyUS,
                mean.ffnIOUnlockUS
            )
        }

        out += "\n"
        return out
    }

    static func writeDecodeKernelProfileCSV(profile: DecodeKernelProfile, to path: String) throws {
        let posix = Locale(identifier: "en_US_POSIX")
        var out = "layer,sample,attn_eval_us,attn_hw_ns,attn_host_overhead_us,self_mask_update_us,k_cache_update_us,v_cache_update_us,mask_update_us,x2_to_ffn_copy_us,ffn_eval_us,ffn_hw_ns,ffn_host_overhead_us,ffn_to_next_attn_copy_us\n"

        for (layerIdx, layer) in profile.layers.enumerated() {
            let n = layer.attnEvalUS.count
            precondition(layer.attnHwNS.count == n)
            precondition(layer.attnHostOverheadUS.count == n)
            precondition(layer.selfMaskUpdateUS.count == n)
            precondition(layer.kCacheUpdateUS.count == n)
            precondition(layer.vCacheUpdateUS.count == n)
            precondition(layer.maskUpdateUS.count == n)
            precondition(layer.x2ToFfnCopyUS.count == n)
            precondition(layer.ffnEvalUS.count == n)
            precondition(layer.ffnHwNS.count == n)
            precondition(layer.ffnHostOverheadUS.count == n)
            precondition(layer.ffnToNextAttnCopyUS.count == n)

            for i in 0..<n {
                out += String(
                    format: "%d,%d,%.3f,%llu,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%llu,%.3f,%.3f\n",
                    locale: posix,
                    layerIdx,
                    i,
                    layer.attnEvalUS[i],
                    layer.attnHwNS[i],
                    layer.attnHostOverheadUS[i],
                    layer.selfMaskUpdateUS[i],
                    layer.kCacheUpdateUS[i],
                    layer.vCacheUpdateUS[i],
                    layer.maskUpdateUS[i],
                    layer.x2ToFfnCopyUS[i],
                    layer.ffnEvalUS[i],
                    layer.ffnHwNS[i],
                    layer.ffnHostOverheadUS[i],
                    layer.ffnToNextAttnCopyUS[i]
                )
            }
        }

        try out.write(toFile: path, atomically: true, encoding: .utf8)
    }
}
