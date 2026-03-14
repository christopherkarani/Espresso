import Darwin
import Foundation
import ANETypes
import Espresso

enum ResultsFormatter {
    private static let locale = Locale(identifier: "en_US_POSIX")

    static func chipName() -> String {
        var length: size_t = 0
        guard sysctlbyname("machdep.cpu.brand_string", nil, &length, nil, 0) == 0, length > 0 else {
            return "Unknown"
        }

        var bytes = [CChar](repeating: 0, count: length)
        guard sysctlbyname("machdep.cpu.brand_string", &bytes, &length, nil, 0) == 0 else {
            return "Unknown"
        }

        let trimmed = bytes.prefix { $0 != 0 }
        return String(decoding: trimmed.map { UInt8(bitPattern: $0) }, as: UTF8.self)
    }

    static func formatReport(
        aneResult: BenchmarkResult,
        aneTimingBreakdown: (ane: Double, io: Double, elem: Double),
        coreMLResults: [(label: String, result: BenchmarkResult)],
        coreMLLoadTimeMs: Double?,
        thermalBefore: String?,
        thermalAfter: String?,
        flopsPerPass: Double,
        nLayers: Int
    ) -> String {
        let aneTFLOPS = FLOPCalculator.sustainedTFLOPS(flops: flopsPerPass, latencyMs: aneResult.median)
        let aneUtilization = FLOPCalculator.aneUtilization(sustainedTFLOPS: aneTFLOPS)
        let aneForwardPassesPerSecond = aneResult.median > 0 ? 1_000.0 / aneResult.median : 0

        var lines: [String] = []
        lines.append("EspressoBench Report")
        lines.append("====================")
        lines.append(
            "Chip: \(chipName()) | Config: layers=\(nLayers) dim=\(ModelConfig.dim) hidden=\(ModelConfig.hidden) heads=\(ModelConfig.heads) seq=\(ModelConfig.seqLen)"
        )
        lines.append(
            formatted("FLOPs per forward pass: %.3f GFLOPs", flopsPerPass / 1_000_000_000.0)
        )
        lines.append("")
        lines.append("Latency Stats (ms)")
        lines.append("------------------")
        lines.append(tableHeader())
        lines.append(tableRow(label: aneResult.label, result: aneResult))
        for entry in coreMLResults {
            lines.append(tableRow(label: entry.label, result: entry.result))
        }
        lines.append("")
        lines.append("Throughput")
        lines.append("----------")
        lines.append(
            formatted(
                "%@: %.3f TFLOPS | %.2f%% peak utilization | %.2f forward passes/sec",
                aneResult.label as NSString,
                aneTFLOPS,
                aneUtilization,
                aneForwardPassesPerSecond
            )
        )
        for entry in coreMLResults {
            let tflops = FLOPCalculator.sustainedTFLOPS(flops: flopsPerPass, latencyMs: entry.result.median)
            let utilization = FLOPCalculator.aneUtilization(sustainedTFLOPS: tflops)
            let forwardPassesPerSecond = entry.result.median > 0 ? 1_000.0 / entry.result.median : 0
            lines.append(
                formatted(
                    "%@: %.3f TFLOPS | %.2f%% peak utilization | %.2f forward passes/sec",
                    entry.label,
                    tflops,
                    utilization,
                    forwardPassesPerSecond
                )
            )
        }
        lines.append("")
        lines.append("Time Breakdown (\(aneResult.label), avg ms)")
        lines.append(String(repeating: "-", count: max(20, 20 + aneResult.label.count)))
        let totalTime = aneTimingBreakdown.ane + aneTimingBreakdown.io + aneTimingBreakdown.elem
        lines.append(
            timeBreakdownRow(label: "ANE compute", value: aneTimingBreakdown.ane, total: totalTime)
        )
        lines.append(
            timeBreakdownRow(label: "IO", value: aneTimingBreakdown.io, total: totalTime)
        )
        lines.append(
            timeBreakdownRow(label: "CPU", value: aneTimingBreakdown.elem, total: totalTime)
        )
        if !coreMLResults.isEmpty {
            lines.append("")
            lines.append("Core ML Comparison")
            lines.append("------------------")
            if let coreMLLoadTimeMs {
                lines.append(formatted("Core ML load time (.all): %.3f ms", coreMLLoadTimeMs))
            }
            for entry in coreMLResults {
                let ratio = entry.result.median > 0 ? entry.result.median / aneResult.median : 0
                lines.append(
                    formatted(
                        "%@: median %.3f ms | vs %@ %.2fx",
                        entry.label,
                        entry.result.median,
                        aneResult.label as NSString,
                        ratio
                    )
                )
            }
        }
        if let thermalBefore, let thermalAfter {
            lines.append("")
            lines.append("Thermal State")
            lines.append("-------------")
            lines.append("Before: \(thermalBefore)")
            lines.append("After:  \(thermalAfter)")
        }

        return lines.joined(separator: "\n") + "\n"
    }

    static func writeCSV(latencies: [Double], to path: String) throws {
        var rows = ["iteration,latency_ms"]
        rows.reserveCapacity(latencies.count + 1)
        for (index, latency) in latencies.enumerated() {
            rows.append(formatted("%d,%.6f", index + 1, latency))
        }
        let output = rows.joined(separator: "\n") + "\n"
        try output.write(toFile: path, atomically: true, encoding: .utf8)
    }

    static func formatReport(
        aneResult: BenchmarkResult,
        aneTimingBreakdown: (ane: Double, io: Double, elem: Double),
        compileTimeMs: Double?,
        inferenceResult: BenchmarkResult?,
        inferenceTimingBreakdown: (ane: Double, io: Double, elem: Double)?,
        inferenceCompileTimeMs: Double?,
        coreMLResults: [(label: String, result: BenchmarkResult)]?,
        coreMLLoadTimeMs: Double?,
        thermalBefore: String?,
        thermalAfter: String?,
        flopsPerPass: Double,
        nLayers: Int
    ) -> String {
        var report = formatReport(
            aneResult: aneResult,
            aneTimingBreakdown: aneTimingBreakdown,
            coreMLResults: coreMLResults ?? [],
            coreMLLoadTimeMs: coreMLLoadTimeMs,
            thermalBefore: thermalBefore,
            thermalAfter: thermalAfter,
            flopsPerPass: flopsPerPass,
            nLayers: nLayers
        )
        if let compileTimeMs {
            report += formatted("ANE direct compile time: %.3f ms\n", compileTimeMs)
        }
        if let inferenceResult {
            report += "\n"
            report += formatInferenceComparisonSection(
                trainingResult: aneResult,
                inferenceResult: inferenceResult,
                inferenceTimingBreakdown: inferenceTimingBreakdown,
                inferenceCompileTimeMs: inferenceCompileTimeMs,
                coreMLResults: coreMLResults,
                coreMLLoadTimeMs: coreMLLoadTimeMs,
                flopsPerPass: flopsPerPass
            )
        }
        return report
    }

    private static func tableHeader() -> String {
        "Label                        Mean(ms) Median(ms)    P50(ms)    P95(ms)    P99(ms)    Min(ms)    Max(ms) StdDev(ms)"
    }

    private static func tableRow(label: String, result: BenchmarkResult) -> String {
        formatted(
            "%-28@ %10.3f %10.3f %10.3f %10.3f %10.3f %10.3f %10.3f %10.3f",
            label as NSString,
            result.mean,
            result.median,
            result.p50,
            result.p95,
            result.p99,
            result.min,
            result.max,
            result.stddev
        )
    }

    private static func timeBreakdownRow(label: String, value: Double, total: Double) -> String {
        let percentage = total > 0 ? (value / total) * 100.0 : 0
        return formatted("%-12@ %10.3f ms %8.2f%%", label as NSString, value, percentage)
    }

    private static func formatted(_ format: String, _ arguments: CVarArg...) -> String {
        String(format: format, locale: locale, arguments: arguments)
    }

    // Compatibility shims for the pre-existing bench CLI. main.swift is rewritten later in this task.
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
        var report = formatReport(
            aneResult: inferenceResult,
            aneTimingBreakdown: inferenceTimingBreakdown ?? (0, 0, 0),
            coreMLResults: coreMLResults ?? [],
            coreMLLoadTimeMs: coreMLLoadTimeMs,
            thermalBefore: thermalBefore,
            thermalAfter: thermalAfter,
            flopsPerPass: flopsPerPass,
            nLayers: nLayers
        )
        if let inferenceCompileTimeMs {
            report += formatted("Inference compile time: %.3f ms\n", inferenceCompileTimeMs)
        }
        return report
    }

    private static func formatInferenceComparisonSection(
        trainingResult: BenchmarkResult,
        inferenceResult: BenchmarkResult,
        inferenceTimingBreakdown: (ane: Double, io: Double, elem: Double)?,
        inferenceCompileTimeMs: Double?,
        coreMLResults: [(label: String, result: BenchmarkResult)]?,
        coreMLLoadTimeMs: Double?,
        flopsPerPass: Double
    ) -> String {
        let inferenceTFLOPS = FLOPCalculator.sustainedTFLOPS(flops: flopsPerPass, latencyMs: inferenceResult.median)
        let inferenceUtilization = FLOPCalculator.aneUtilization(sustainedTFLOPS: inferenceTFLOPS)
        let inferenceForwardPassesPerSecond = inferenceResult.median > 0 ? 1_000.0 / inferenceResult.median : 0
        let trainingToInferenceSpeedup = inferenceResult.median > 0 ? trainingResult.median / inferenceResult.median : 0

        var lines: [String] = []
        lines.append("Inference Comparison")
        lines.append("--------------------")
        if let inferenceCompileTimeMs {
            lines.append(formatted("Inference compile time: %.3f ms", inferenceCompileTimeMs))
        }
        lines.append(tableHeader())
        lines.append(tableRow(label: inferenceResult.label, result: inferenceResult))
        for entry in coreMLResults ?? [] {
            lines.append(tableRow(label: entry.label, result: entry.result))
        }
        lines.append("")
        lines.append(
            formatted(
                "%@: %.3f TFLOPS | %.2f%% peak utilization | %.2f forward passes/sec",
                inferenceResult.label as NSString,
                inferenceTFLOPS,
                inferenceUtilization,
                inferenceForwardPassesPerSecond
            )
        )
        lines.append(
            formatted(
                "Training vs inference speedup: %.2fx",
                trainingToInferenceSpeedup
            )
        )
        if let breakdown = inferenceTimingBreakdown {
            let total = breakdown.ane + breakdown.io + breakdown.elem
            lines.append("")
            lines.append("Time Breakdown (\(inferenceResult.label), avg ms)")
            lines.append(String(repeating: "-", count: max(20, 20 + inferenceResult.label.count)))
            lines.append(timeBreakdownRow(label: "ANE compute", value: breakdown.ane, total: total))
            lines.append(timeBreakdownRow(label: "IO", value: breakdown.io, total: total))
            lines.append(timeBreakdownRow(label: "CPU", value: breakdown.elem, total: total))
        }
        if let coreMLResults, !coreMLResults.isEmpty {
            lines.append("")
            lines.append("Core ML Comparison (vs inference)")
            lines.append("-------------------------------")
            if let coreMLLoadTimeMs {
                lines.append(formatted("Core ML load time (.all): %.3f ms", coreMLLoadTimeMs))
            }
            for entry in coreMLResults {
                let ratio = entry.result.median > 0 ? entry.result.median / inferenceResult.median : 0
                lines.append(
                    formatted(
                        "%@: median %.3f ms | vs %@ %.2fx",
                        entry.label,
                        entry.result.median,
                        inferenceResult.label as NSString,
                        ratio
                    )
                )
            }
        }
        return lines.joined(separator: "\n") + "\n"
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
        var lines: [String] = []
        lines.append("EspressoBench Decode Report")
        lines.append("===========================")
        lines.append("Chip: \(chipName()) | layers=\(nLayers) | decodeSteps=\(decodeSteps) | decodeMaxSeq=\(decodeMaxSeq)")
        if let decodeCompileTimeMs {
            lines.append(formatted("Decode compile time: %.3f ms", decodeCompileTimeMs))
        }
        if let decodeTokensPerSecond {
            lines.append(formatted("ANE tokens/sec: %.3f", decodeTokensPerSecond))
        }
        lines.append(tableHeader())
        lines.append(tableRow(label: decodeResult.label, result: decodeResult))
        for entry in coreMLDecodeResults ?? [] {
            lines.append(tableRow(label: entry.label, result: entry.result))
        }
        if let breakdown = decodeTimingBreakdown {
            let total = breakdown.ane + breakdown.io + breakdown.elem
            lines.append("")
            lines.append(timeBreakdownRow(label: "ANE compute", value: breakdown.ane, total: total))
            lines.append(timeBreakdownRow(label: "IO", value: breakdown.io, total: total))
            lines.append(timeBreakdownRow(label: "CPU", value: breakdown.elem, total: total))
        }
        if let coreMLLoadTimeMs {
            lines.append("")
            lines.append(formatted("Core ML decode load time: %.3f ms", coreMLLoadTimeMs))
        }
        if let coreMLDecodeResults, !coreMLDecodeResults.isEmpty {
            lines.append("")
            lines.append("Core ML Decode Comparison")
            lines.append("------------------------")
            for entry in coreMLDecodeResults {
                let speedup = entry.result.median > 0 ? entry.result.median / decodeResult.median : 0
                lines.append(
                    formatted(
                        "%@: median %.3f ms/token | %@ speedup %.2fx",
                        entry.label,
                        entry.result.median,
                        decodeResult.label as NSString,
                        speedup
                    )
                )
            }
            if let fastest = coreMLDecodeResults.min(by: { $0.result.median < $1.result.median })?.result {
                let strictSpeedup = fastest.median > 0 ? fastest.median / decodeResult.median : 0
                lines.append(
                    formatted(
                        "Fastest Core ML decode median %.3f ms/token | %@ speedup %.2fx",
                        fastest.median,
                        decodeResult.label as NSString,
                        strictSpeedup
                    )
                )
            }
        }
        return lines.joined(separator: "\n") + "\n"
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
