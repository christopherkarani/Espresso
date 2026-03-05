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
        nLayers: Int
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
        var out = "layer,iteration,attn_write_us,attn_eval_us,attn_read_us,gap_attn_to_ffn_us,ffn_write_us,ffn_copy_us,ffn_eval_us,ffn_read_us\n"

        for (layerIdx, layer) in profile.layers.enumerated() {
            let n = layer.attnWriteUS.count
            precondition(layer.attnEvalUS.count == n)
            precondition(layer.attnReadUS.count == n)
            precondition(layer.ffnWriteUS.count == n)
            precondition(layer.ffnCopyUS.count == n)
            precondition(layer.ffnEvalUS.count == n)
            precondition(layer.ffnReadUS.count == n)
            precondition(layer.gapAttnToFfnUS.count == n)

            for i in 0..<n {
                out += String(
                    format: "%d,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n",
                    locale: posix,
                    layerIdx,
                    i,
                    layer.attnWriteUS[i],
                    layer.attnEvalUS[i],
                    layer.attnReadUS[i],
                    layer.gapAttnToFfnUS[i],
                    layer.ffnWriteUS[i],
                    layer.ffnCopyUS[i],
                    layer.ffnEvalUS[i],
                    layer.ffnReadUS[i]
                )
            }
        }

        try out.write(toFile: path, atomically: true, encoding: .utf8)
    }
}
