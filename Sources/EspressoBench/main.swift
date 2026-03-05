import Foundation
import ANETypes
import ANERuntime
import Espresso
import Darwin

// MARK: - CLI Argument Parsing

struct BenchmarkOptions {
    var aneOnly: Bool = false
    var inference: Bool = false
    var decode: Bool = false
    var inferenceFP16Handoff: Bool = false
    var inferenceOnly: Bool = false
    var profileKernels: Bool = false
    var sustained: Bool = false
    var warmup: Int = 50
    var iterations: Int = 1000
    var decodeSteps: Int = 32
    var decodeMaxSeq: Int = 32
    var outputDir: String? = nil
    var coreMLModelPath: String = "benchmarks/models/transformer_layer.mlpackage"
    var nLayers: Int = 1
    var dumpANE: Bool = false
    var dumpANEFilter: String? = nil
    var perfStats: Bool = false

    static func parse(_ args: [String]) -> BenchmarkOptions {
        var opts = BenchmarkOptions()
        var i = 1  // skip program name
        while i < args.count {
            switch args[i] {
            case "--ane-only":
                opts.aneOnly = true
            case "--inference":
                opts.inference = true
            case "--decode":
                opts.decode = true
            case "--inference-fp16-handoff":
                opts.inferenceFP16Handoff = true
            case "--inference-only":
                opts.inference = true
                opts.inferenceOnly = true
            case "--profile-kernels":
                opts.profileKernels = true
            case "--sustained":
                opts.sustained = true
            case "--warmup":
                i += 1
                guard i < args.count, let v = Int(args[i]) else {
                    printStderr("--warmup requires an integer argument")
                    exit(1)
                }
                opts.warmup = v
            case "--iterations":
                i += 1
                guard i < args.count, let v = Int(args[i]) else {
                    printStderr("--iterations requires an integer argument")
                    exit(1)
                }
                opts.iterations = v
            case "--decode-steps":
                i += 1
                guard i < args.count, let v = Int(args[i]), v > 0 else {
                    printStderr("--decode-steps requires a positive integer argument")
                    exit(1)
                }
                opts.decodeSteps = v
            case "--decode-max-seq":
                i += 1
                guard i < args.count, let v = Int(args[i]), v > 1 else {
                    printStderr("--decode-max-seq requires an integer >= 2")
                    exit(1)
                }
                opts.decodeMaxSeq = v
            case "--output":
                i += 1
                guard i < args.count else {
                    printStderr("--output requires a path argument")
                    exit(1)
                }
                opts.outputDir = args[i]
            case "--model":
                i += 1
                guard i < args.count else {
                    printStderr("--model requires a path argument")
                    exit(1)
                }
                opts.coreMLModelPath = args[i]
            case "--layers":
                i += 1
                guard i < args.count, let v = Int(args[i]) else {
                    printStderr("--layers requires an integer argument")
                    exit(1)
                }
                opts.nLayers = v
            case "--dump-ane":
                opts.dumpANE = true
            case "--dump-ane-filter":
                i += 1
                guard i < args.count else {
                    printStderr("--dump-ane-filter requires a string argument")
                    exit(1)
                }
                opts.dumpANEFilter = args[i]
            case "--perf-stats":
                opts.perfStats = true
            case "--help", "-h":
                printUsage()
                exit(0)
            default:
                printStderr("Unknown argument: \(args[i])")
                printUsage()
                exit(1)
            }
            i += 1
        }
        return opts
    }

    static func printUsage() {
        print("""
        EspressoBench — ANE Runtime Benchmark Suite

        Usage: espresso-bench [OPTIONS]

        Options:
          --ane-only         Skip Core ML benchmarks
          --inference        Run inference-optimized forward pass (fused residuals)
          --decode           Run autoregressive decode benchmark with KV cache
          --inference-fp16-handoff  Use FP16 surface-to-surface handoff between attn -> ffn
          --inference-only   Run inference benchmark only (skips training benchmark to save compile budget)
          --decode-steps N   Decode tokens per sequence (default: 32)
          --decode-max-seq N Decode KV-cache max sequence length (default: 32)
                            (current ANE decode path requires decode-max-seq == 32)
          --profile-kernels  Record per-kernel stage timing (us) and save CSV
          --sustained        Run 60-second sustained thermal test
          --warmup N         Warmup iterations (default: 50)
          --iterations N     Measured iterations (default: 1000)
          --output DIR       Output directory for results
          --model PATH       Path to Core ML .mlpackage
          --layers N         Number of transformer layers (default: 1)
          --dump-ane         Dump ObjC runtime methods/properties for private ANE classes
          --dump-ane-filter S  Only show methods/properties containing substring S
          --perf-stats       Enable `_ANEPerformanceStats` collection for direct ANE eval (sets ANE_PERF_STATS=1)
          -h, --help         Show this help
        """)
    }
}

// MARK: - Main

let opts = BenchmarkOptions.parse(CommandLine.arguments)

if opts.dumpANE {
    _ = dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW)
    let classes = [
        "_ANEInMemoryModelDescriptor",
        "_ANEInMemoryModel",
        "_ANERequest",
        "_ANEIOSurfaceObject",
        "_ANEPerformanceStats",
        "_ANEPerformanceCounters",
        "_ANEClient",
        "_ANEVirtualClient",
        "_ANEChainingRequest",
    ]
    for c in classes {
        print(ANEIntrospector.dump(className: c, filter: opts.dumpANEFilter))
    }
    exit(0)
}

if opts.perfStats {
    setenv("ANE_PERF_STATS", "1", 1)
}

let runner = BenchmarkRunner(warmup: opts.warmup, iterations: opts.iterations)

let flopsPerPass = FLOPCalculator.forwardPassFLOPs() * Double(opts.nLayers)
printStderr("Espresso Benchmark Suite")
printStderr("========================")
printStderr(String(format: "Config: dim=%d, hidden=%d, seq=%d, heads=%d, layers=%d",
                   ModelConfig.dim, ModelConfig.hidden, ModelConfig.seqLen, ModelConfig.heads, opts.nLayers))
printStderr(String(format: "FLOPs per forward pass: %.2f GFLOPs", flopsPerPass / 1e9))
printStderr(String(format: "Iterations: %d warmup + %d measured", opts.warmup, opts.iterations))
printStderr("")

let handoff: ForwardPass.InferenceInterKernelHandoff = opts.inferenceFP16Handoff ? .fp16SurfaceCopy : .cpuRoundTrip

func benchmarkStats(_ result: BenchmarkResult) -> [String: Any] {
    [
        "mean_ms": result.mean,
        "median_ms": result.median,
        "p95_ms": result.p95,
        "p99_ms": result.p99,
        "min_ms": result.min,
        "max_ms": result.max,
        "stddev_ms": result.stddev,
        "warmup_count": result.warmupCount,
        "iteration_count": result.iterationCount,
        "tokens_per_second": result.mean > 0 ? 1000.0 / result.mean : 0,
    ]
}

func inferenceProfileAverages(_ profile: InferenceKernelProfile) -> [[String: Any]] {
    profile.layers.indices.map { layerIdx in
        let mean = profile.averageLayerMetrics(layerIndex: layerIdx)
        return [
            "layer": layerIdx,
            "samples": mean.sampleCount,
            "attn_eval_host_us": mean.attnEvalUS,
            "attn_hw_us": mean.attnHwUS,
            "attn_host_overhead_us": mean.attnHostOverheadUS,
            "attn_io_lock_us": mean.attnIOLockUS,
            "attn_io_body_us": mean.attnIOBodyUS,
            "attn_io_unlock_us": mean.attnIOUnlockUS,
            "gap_attn_to_ffn_us": mean.gapAttnToFfnUS,
            "handoff_cpu_roundtrip_us": mean.handoffCPUUS,
            "handoff_fp16_copy_us": mean.handoffFP16CopyUS,
            "ffn_eval_host_us": mean.ffnEvalUS,
            "ffn_hw_us": mean.ffnHwUS,
            "ffn_host_overhead_us": mean.ffnHostOverheadUS,
            "ffn_io_lock_us": mean.ffnIOLockUS,
            "ffn_io_body_us": mean.ffnIOBodyUS,
            "ffn_io_unlock_us": mean.ffnIOUnlockUS,
        ] as [String: Any]
    }
}

if opts.decode {
    let decodeResult: ANEDirectBench.Result
    do {
        decodeResult = try ANEDirectBench.runDecode(
            warmup: opts.warmup,
            iterations: opts.iterations,
            decodeSteps: opts.decodeSteps,
            decodeMaxSeq: opts.decodeMaxSeq,
            nLayers: opts.nLayers,
            profileKernels: opts.profileKernels
        )
    } catch {
        printStderr("ANE decode benchmark failed: \(error)")
        exit(1)
    }

    var coreMLDecodeResult: CoreMLBench.Result? = nil
    if !opts.aneOnly {
        do {
            coreMLDecodeResult = try CoreMLBench.runNaiveDecode(
                warmup: opts.warmup,
                iterations: opts.iterations,
                decodeSteps: opts.decodeSteps,
                decodeMaxSeq: opts.decodeMaxSeq,
                modelPath: opts.coreMLModelPath
            )
        } catch {
            printStderr("Core ML decode benchmark failed: \(error)")
            printStderr("Continuing with ANE-only decode results...")
        }
    }

    let report = ResultsFormatter.formatDecodeReport(
        decodeResult: decodeResult.benchmarkResult,
        decodeTimingBreakdown: decodeResult.avgTimingBreakdown,
        decodeCompileTimeMs: decodeResult.compileTimeMs,
        decodeTokensPerSecond: decodeResult.tokensPerSecond,
        coreMLDecodeResults: coreMLDecodeResult?.results,
        coreMLLoadTimeMs: coreMLDecodeResult?.modelLoadTimeMs,
        nLayers: opts.nLayers,
        decodeSteps: opts.decodeSteps,
        decodeMaxSeq: opts.decodeMaxSeq
    )
    print(report)

    let outputDir: String
    if let dir = opts.outputDir {
        outputDir = dir
    } else {
        let dateFormatter = DateFormatter()
        dateFormatter.dateFormat = "yyyy-MM-dd-HHmmss"
        dateFormatter.locale = Locale(identifier: "en_US_POSIX")
        let timestamp = dateFormatter.string(from: Date())
        outputDir = "benchmarks/results/\(timestamp)"
    }

    do {
        try FileManager.default.createDirectory(atPath: outputDir, withIntermediateDirectories: true)
        try ResultsFormatter.writeCSV(
            latencies: decodeResult.benchmarkResult.latencies,
            to: "\(outputDir)/ane_decode_token_latencies.csv"
        )
        if let profile = decodeResult.decodeKernelProfile {
            try ResultsFormatter.writeDecodeKernelProfileCSV(
                profile: profile,
                to: "\(outputDir)/ane_decode_kernel_profile.csv"
            )
        }
        if let coreML = coreMLDecodeResult {
            for (label, result) in coreML.results {
                let filename = label.lowercased()
                    .replacingOccurrences(of: " ", with: "_")
                    .replacingOccurrences(of: "(", with: "")
                    .replacingOccurrences(of: ")", with: "")
                    .replacingOccurrences(of: ".", with: "")
                try ResultsFormatter.writeCSV(
                    latencies: result.latencies,
                    to: "\(outputDir)/\(filename)_token_latencies.csv"
                )
            }
        }
        try report.write(toFile: "\(outputDir)/summary.txt", atomically: true, encoding: .utf8)
        var summaryJSON = RunMetadata.base(mode: "decode", options: opts)
        summaryJSON["ane_decode"] = [
            "compile_time_ms": decodeResult.compileTimeMs,
            "timing_breakdown_ms": [
                "ane": decodeResult.avgTimingBreakdown.ane,
                "io": decodeResult.avgTimingBreakdown.io,
                "elem": decodeResult.avgTimingBreakdown.elem,
            ],
            "metrics": benchmarkStats(decodeResult.benchmarkResult),
        ]
        if let coreML = coreMLDecodeResult {
            let entries = coreML.results.map { (label, result) in
                [
                    "label": label,
                    "metrics": benchmarkStats(result),
                ] as [String: Any]
            }
            summaryJSON["coreml_decode"] = [
                "model_load_time_ms": coreML.modelLoadTimeMs,
                "results": entries,
            ]
            if let fastest = coreML.results.min(by: { $0.result.median < $1.result.median })?.result {
                summaryJSON["speedup_vs_fastest_coreml_decode"] = fastest.median / decodeResult.benchmarkResult.median
            }
        }
        try RunMetadata.writeJSON(summaryJSON, to: "\(outputDir)/summary.json")
        printStderr("\nResults saved to: \(outputDir)/")
    } catch {
        printStderr("Failed to save decode results: \(error)")
    }

    exit(0)
}

if opts.inferenceOnly {
    // --- Inference-only flow (skips training compile budget) ---
    let thermalBefore = ThermalMonitor.currentState()

    let inferenceResult: ANEDirectBench.Result
    do {
        inferenceResult = try ANEDirectBench.runInference(
            warmup: opts.warmup,
            iterations: opts.iterations,
            nLayers: opts.nLayers,
            handoff: handoff,
            profileKernels: opts.profileKernels
        )
    } catch {
        printStderr("ANE Inference benchmark failed: \(error)")
        exit(1)
    }

    // Core ML (optional)
    var coreMLResult: CoreMLBench.Result? = nil
    if !opts.aneOnly {
        do {
            coreMLResult = try CoreMLBench.run(runner: runner, modelPath: opts.coreMLModelPath)
        } catch {
            printStderr("Core ML benchmark failed: \(error)")
            printStderr("Continuing with ANE-only results...")
        }
    }

    let thermalAfter = ThermalMonitor.currentState()

    var report = ResultsFormatter.formatInferenceOnlyReport(
        inferenceResult: inferenceResult.benchmarkResult,
        inferenceTimingBreakdown: inferenceResult.avgTimingBreakdown,
        inferenceCompileTimeMs: inferenceResult.compileTimeMs,
        coreMLResults: coreMLResult?.results,
        coreMLLoadTimeMs: coreMLResult?.modelLoadTimeMs,
        flopsPerPass: flopsPerPass,
        nLayers: opts.nLayers,
        thermalBefore: thermalBefore,
        thermalAfter: thermalAfter
    )
    if let profile = inferenceResult.kernelProfile {
        report += ResultsFormatter.formatInferenceKernelProfileSummaryTable(profile: profile, handoff: handoff)
    }
    print(report)

    // Save results
    let outputDir: String
    if let dir = opts.outputDir {
        outputDir = dir
    } else {
        let dateFormatter = DateFormatter()
        dateFormatter.dateFormat = "yyyy-MM-dd-HHmmss"
        dateFormatter.locale = Locale(identifier: "en_US_POSIX")
        let timestamp = dateFormatter.string(from: Date())
        outputDir = "benchmarks/results/\(timestamp)"
    }

    do {
        try FileManager.default.createDirectory(atPath: outputDir, withIntermediateDirectories: true)

        try ResultsFormatter.writeCSV(
            latencies: inferenceResult.benchmarkResult.latencies,
            to: "\(outputDir)/ane_inference_latencies.csv"
        )
        if let profile = inferenceResult.kernelProfile {
            try ResultsFormatter.writeInferenceKernelProfileCSV(
                profile: profile,
                to: "\(outputDir)/ane_inference_kernel_profile.csv"
            )
        }
        if let coreML = coreMLResult {
            for (label, result) in coreML.results {
                let filename = label.lowercased()
                    .replacingOccurrences(of: " ", with: "_")
                    .replacingOccurrences(of: "(", with: "")
                    .replacingOccurrences(of: ")", with: "")
                    .replacingOccurrences(of: ".", with: "")
                try ResultsFormatter.writeCSV(
                    latencies: result.latencies,
                    to: "\(outputDir)/\(filename)_latencies.csv"
                )
            }
        }

        try report.write(toFile: "\(outputDir)/summary.txt", atomically: true, encoding: .utf8)
        var summaryJSON = RunMetadata.base(mode: "inference-only", options: opts)
        var inferenceEntry: [String: Any] = [
            "compile_time_ms": inferenceResult.compileTimeMs,
            "timing_breakdown_ms": [
                "ane": inferenceResult.avgTimingBreakdown.ane,
                "io": inferenceResult.avgTimingBreakdown.io,
                "elem": inferenceResult.avgTimingBreakdown.elem,
            ],
            "metrics": benchmarkStats(inferenceResult.benchmarkResult),
        ]
        if let profile = inferenceResult.kernelProfile {
            inferenceEntry["kernel_profile_layer_averages_us"] = inferenceProfileAverages(profile)
        }
        summaryJSON["ane_inference"] = inferenceEntry
        if let coreML = coreMLResult {
            summaryJSON["coreml"] = [
                "model_load_time_ms": coreML.modelLoadTimeMs,
                "results": coreML.results.map { entry in
                    [
                        "label": entry.label,
                        "metrics": benchmarkStats(entry.result),
                    ] as [String: Any]
                },
            ]
        }
        try RunMetadata.writeJSON(summaryJSON, to: "\(outputDir)/summary.json")
        printStderr("\nResults saved to: \(outputDir)/")
    } catch {
        printStderr("Failed to save results: \(error)")
    }

    exit(0)
}

// --- Default flow: training + (optional) inference + (optional) coreml ---

// Benchmark 1: ANE Direct (training forward pass)
let aneResult: ANEDirectBench.Result
do {
    aneResult = try ANEDirectBench.run(warmup: opts.warmup, iterations: opts.iterations, nLayers: opts.nLayers)
} catch {
    printStderr("ANE Direct benchmark failed: \(error)")
    exit(1)
}

// Benchmark 1b: ANE Direct Inference (optional, fused residuals)
var inferenceResult: ANEDirectBench.Result? = nil
if opts.inference {
    do {
        inferenceResult = try ANEDirectBench.runInference(
            warmup: opts.warmup,
            iterations: opts.iterations,
            nLayers: opts.nLayers,
            handoff: handoff,
            profileKernels: opts.profileKernels
        )
    } catch {
        printStderr("ANE Inference benchmark failed: \(error)")
        printStderr("Continuing without inference results...")
    }
}

// Benchmark 2: Core ML (optional)
var coreMLResult: CoreMLBench.Result? = nil
if !opts.aneOnly {
    do {
        coreMLResult = try CoreMLBench.run(runner: runner, modelPath: opts.coreMLModelPath)
    } catch {
        printStderr("Core ML benchmark failed: \(error)")
        printStderr("Continuing with ANE-only results...")
    }
}

// Benchmark 3: Sustained thermal test (optional)
var thermalBefore: String? = nil
var thermalAfter: String? = nil
if opts.sustained {
    printStderr("\n=== Sustained Thermal Test (60 seconds) ===")
    do {
        let thermal = try ANEDirectBench.runSustained(duration: 60.0, nLayers: opts.nLayers)
        thermalBefore = thermal.before
        thermalAfter = thermal.after
        for sample in thermal.samples {
            printStderr(String(format: "    t=%.0fs: %@", sample.time, sample.state))
        }
        printStderr("  Total forward passes: \(thermal.iterations)")
    } catch {
        printStderr("  Thermal test failed: \(error)")
    }
}

// Output report
let inferenceReportData: (result: BenchmarkResult, breakdown: (ane: Double, io: Double, elem: Double), compileMs: Double)?
if let inf = inferenceResult {
    inferenceReportData = (result: inf.benchmarkResult, breakdown: inf.avgTimingBreakdown, compileMs: inf.compileTimeMs)
} else {
    inferenceReportData = nil
}

var report = ResultsFormatter.formatReport(
    aneResult: aneResult.benchmarkResult,
    aneTimingBreakdown: aneResult.avgTimingBreakdown,
    compileTimeMs: aneResult.compileTimeMs,
    inferenceResult: inferenceReportData?.result,
    inferenceTimingBreakdown: inferenceReportData?.breakdown,
    inferenceCompileTimeMs: inferenceReportData?.compileMs,
    coreMLResults: coreMLResult?.results,
    coreMLLoadTimeMs: coreMLResult?.modelLoadTimeMs,
    thermalBefore: thermalBefore,
    thermalAfter: thermalAfter,
    flopsPerPass: flopsPerPass,
    nLayers: opts.nLayers
)
if let profile = inferenceResult?.kernelProfile {
    report += ResultsFormatter.formatInferenceKernelProfileSummaryTable(profile: profile, handoff: handoff)
}
print(report)

// Save results
let outputDir: String
if let dir = opts.outputDir {
    outputDir = dir
} else {
    let dateFormatter = DateFormatter()
    dateFormatter.dateFormat = "yyyy-MM-dd-HHmmss"
    dateFormatter.locale = Locale(identifier: "en_US_POSIX")
    let timestamp = dateFormatter.string(from: Date())
    outputDir = "benchmarks/results/\(timestamp)"
}

do {
    try FileManager.default.createDirectory(atPath: outputDir, withIntermediateDirectories: true)

    try ResultsFormatter.writeCSV(
        latencies: aneResult.benchmarkResult.latencies,
        to: "\(outputDir)/ane_direct_latencies.csv"
    )

    if let inf = inferenceResult {
        try ResultsFormatter.writeCSV(
            latencies: inf.benchmarkResult.latencies,
            to: "\(outputDir)/ane_inference_latencies.csv"
        )
        if let profile = inf.kernelProfile {
            try ResultsFormatter.writeInferenceKernelProfileCSV(
                profile: profile,
                to: "\(outputDir)/ane_inference_kernel_profile.csv"
            )
        }
    }

    if let coreML = coreMLResult {
        for (label, result) in coreML.results {
            let filename = label.lowercased()
                .replacingOccurrences(of: " ", with: "_")
                .replacingOccurrences(of: "(", with: "")
                .replacingOccurrences(of: ")", with: "")
                .replacingOccurrences(of: ".", with: "")
            try ResultsFormatter.writeCSV(
                latencies: result.latencies,
                to: "\(outputDir)/\(filename)_latencies.csv"
            )
        }
    }

    try report.write(toFile: "\(outputDir)/summary.txt", atomically: true, encoding: .utf8)
    var summaryJSON = RunMetadata.base(mode: "full", options: opts)
    summaryJSON["ane_direct"] = [
        "compile_time_ms": aneResult.compileTimeMs,
        "timing_breakdown_ms": [
            "ane": aneResult.avgTimingBreakdown.ane,
            "io": aneResult.avgTimingBreakdown.io,
            "elem": aneResult.avgTimingBreakdown.elem,
        ],
        "metrics": benchmarkStats(aneResult.benchmarkResult),
    ]
    if let inf = inferenceResult {
        var inferenceEntry: [String: Any] = [
            "compile_time_ms": inf.compileTimeMs,
            "timing_breakdown_ms": [
                "ane": inf.avgTimingBreakdown.ane,
                "io": inf.avgTimingBreakdown.io,
                "elem": inf.avgTimingBreakdown.elem,
            ],
            "metrics": benchmarkStats(inf.benchmarkResult),
        ]
        if let profile = inf.kernelProfile {
            inferenceEntry["kernel_profile_layer_averages_us"] = inferenceProfileAverages(profile)
        }
        summaryJSON["ane_inference"] = inferenceEntry
    }
    if let coreML = coreMLResult {
        summaryJSON["coreml"] = [
            "model_load_time_ms": coreML.modelLoadTimeMs,
            "results": coreML.results.map { entry in
                [
                    "label": entry.label,
                    "metrics": benchmarkStats(entry.result),
                ] as [String: Any]
            },
        ]
    }
    try RunMetadata.writeJSON(summaryJSON, to: "\(outputDir)/summary.json")
    printStderr("\nResults saved to: \(outputDir)/")
} catch {
    printStderr("Failed to save results: \(error)")
}
