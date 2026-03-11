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
                "ANE Direct: %.3f TFLOPS | %.2f%% peak utilization | %.2f forward passes/sec",
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
        lines.append("Time Breakdown (ANE Direct, avg ms)")
        lines.append("-----------------------------------")
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
                        "%@: median %.3f ms | ANE speedup %.2fx",
                        entry.label,
                        entry.result.median,
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
            report += formatInferenceOnlyReport(
                inferenceResult: inferenceResult,
                inferenceTimingBreakdown: inferenceTimingBreakdown,
                inferenceCompileTimeMs: inferenceCompileTimeMs,
                coreMLResults: coreMLResults,
                coreMLLoadTimeMs: coreMLLoadTimeMs,
                flopsPerPass: flopsPerPass,
                nLayers: nLayers,
                thermalBefore: nil,
                thermalAfter: nil
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
        return lines.joined(separator: "\n") + "\n"
    }

    static func writeInferenceKernelProfileCSV(profile: InferenceKernelProfile, to path: String) throws {
        var rows = ["layer,iteration,attn_eval_us,attn_hw_ns,attn_host_overhead_us,ffn_eval_us,ffn_hw_ns,ffn_host_overhead_us"]
        for (layerIndex, layer) in profile.layers.enumerated() {
            for iteration in layer.attnEvalUS.indices {
                rows.append(
                    formatted(
                        "%d,%d,%.3f,%llu,%.3f,%.3f,%llu,%.3f",
                        layerIndex,
                        iteration,
                        layer.attnEvalUS[iteration],
                        layer.attnHwNS[iteration],
                        layer.attnHostOverheadUS[iteration],
                        layer.ffnEvalUS[iteration],
                        layer.ffnHwNS[iteration],
                        layer.ffnHostOverheadUS[iteration]
                    )
                )
            }
        }
        try (rows.joined(separator: "\n") + "\n").write(toFile: path, atomically: true, encoding: .utf8)
    }

    static func formatInferenceKernelProfileSummaryTable(
        profile: InferenceKernelProfile,
        handoff: ForwardPass.InferenceInterKernelHandoff
    ) -> String {
        let handoffLabel = switch handoff {
        case .cpuRoundTrip: "cpuRoundTrip"
        case .fp16SurfaceCopy: "fp16SurfaceCopy"
        }
        var lines: [String] = []
        lines.append("")
        lines.append("Inference Kernel Profile")
        lines.append("------------------------")
        lines.append("Handoff: \(handoffLabel)")
        for layerIndex in profile.layers.indices {
            let averages = profile.averageLayerMetrics(layerIndex: layerIndex)
            lines.append(
                formatted(
                    "L%d attn %.3f us | ffn %.3f us | gap %.3f us",
                    layerIndex,
                    averages.attnEvalUS,
                    averages.ffnEvalUS,
                    averages.gapAttnToFfnUS
                )
            )
        }
        return lines.joined(separator: "\n") + "\n"
    }

    static func writeDecodeKernelProfileCSV(profile: DecodeKernelProfile, to path: String) throws {
        var rows = ["layer,iteration,attn_eval_us,attn_hw_ns,attn_host_overhead_us,ffn_eval_us,ffn_hw_ns,ffn_host_overhead_us"]
        for (layerIndex, layer) in profile.layers.enumerated() {
            for iteration in layer.attnEvalUS.indices {
                rows.append(
                    formatted(
                        "%d,%d,%.3f,%llu,%.3f,%.3f,%llu,%.3f",
                        layerIndex,
                        iteration,
                        layer.attnEvalUS[iteration],
                        layer.attnHwNS[iteration],
                        layer.attnHostOverheadUS[iteration],
                        layer.ffnEvalUS[iteration],
                        layer.ffnHwNS[iteration],
                        layer.ffnHostOverheadUS[iteration]
                    )
                )
            }
        }
        try (rows.joined(separator: "\n") + "\n").write(toFile: path, atomically: true, encoding: .utf8)
    }
}
