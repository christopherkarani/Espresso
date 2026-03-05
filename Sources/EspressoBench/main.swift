import Foundation
import ANETypes
import ANERuntime
import Espresso

// MARK: - CLI Argument Parsing

struct BenchmarkOptions {
    var aneOnly: Bool = false
    var inference: Bool = false
    var diagnostic: Bool = false
    var sustained: Bool = false
    var warmup: Int = 50
    var iterations: Int = 1000
    var outputDir: String? = nil
    var coreMLModelPath: String = "benchmarks/models/transformer_layer.mlpackage"
    var nLayers: Int = 1

    static func parse(_ args: [String]) -> BenchmarkOptions {
        var opts = BenchmarkOptions()
        var i = 1  // skip program name
        while i < args.count {
            switch args[i] {
            case "--ane-only":
                opts.aneOnly = true
            case "--inference":
                opts.inference = true
            case "--diagnostic":
                opts.diagnostic = true
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
          --diagnostic       Run diagnostic experiments (convs-only, batched-head, whole-model)
          --sustained        Run 60-second sustained thermal test
          --warmup N         Warmup iterations (default: 50)
          --iterations N     Measured iterations (default: 1000)
          --output DIR       Output directory for results
          --model PATH       Path to Core ML .mlpackage
          --layers N         Number of transformer layers (default: 1)
          -h, --help         Show this help
        """)
    }
}

// MARK: - Main

let opts = BenchmarkOptions.parse(CommandLine.arguments)
let runner = BenchmarkRunner(warmup: opts.warmup, iterations: opts.iterations)

let flopsPerPass = FLOPCalculator.forwardPassFLOPs() * Double(opts.nLayers)
printStderr("Espresso Benchmark Suite")
printStderr("========================")
printStderr(String(format: "Config: dim=%d, hidden=%d, seq=%d, heads=%d, layers=%d",
                   ModelConfig.dim, ModelConfig.hidden, ModelConfig.seqLen, ModelConfig.heads, opts.nLayers))
printStderr(String(format: "FLOPs per forward pass: %.2f GFLOPs", flopsPerPass / 1e9))
printStderr(String(format: "Iterations: %d warmup + %d measured", opts.warmup, opts.iterations))
printStderr("")

// --- Benchmark 1: ANE Direct (training forward pass) ---
let aneResult: ANEDirectBench.Result
do {
    aneResult = try ANEDirectBench.run(warmup: opts.warmup, iterations: opts.iterations, nLayers: opts.nLayers)
} catch {
    printStderr("ANE Direct benchmark failed: \(error)")
    exit(1)
}

// --- Benchmark 1b: ANE Direct Inference (optional, fused residuals) ---
var inferenceResult: ANEDirectBench.Result? = nil
if opts.inference {
    do {
        inferenceResult = try ANEDirectBench.runInference(warmup: opts.warmup, iterations: opts.iterations, nLayers: opts.nLayers)
    } catch {
        printStderr("ANE Inference benchmark failed: \(error)")
        printStderr("Continuing without inference results...")
    }
}

// --- Benchmark 1c: ANE Fused Inference (split-head + kernel fusion) ---
var fusedResult: ANEDirectBench.Result? = nil
if opts.inference {
    do {
        fusedResult = try ANEDirectBench.runFusedInference(warmup: opts.warmup, iterations: opts.iterations, nLayers: opts.nLayers)
    } catch {
        printStderr("ANE Fused Inference benchmark failed: \(error)")
        printStderr("Continuing without fused results...")
    }
}

// --- Benchmark 1d: Diagnostic Experiments (optional) ---
var diagnosticConvsOnly: ANEDirectBench.Result? = nil
var diagnosticBatchedHead: ANEDirectBench.Result? = nil
var diagnosticWholeModel: ANEDirectBench.Result? = nil

if opts.diagnostic {
    printStderr("\n=== Phase 2 Diagnostic Experiments ===")
    printStderr("Running fused baseline + 3 diagnostic experiments...")

    // Exp 1: Convs-Only
    do {
        diagnosticConvsOnly = try DiagnosticBench.runConvsOnly(
            warmup: opts.warmup, iterations: opts.iterations, nLayers: opts.nLayers)
    } catch {
        printStderr("Diagnostic Exp 1 (Convs-Only) failed: \(error)")
    }

    // Exp 2: Batched-Head
    do {
        diagnosticBatchedHead = try DiagnosticBench.runBatchedHead(
            warmup: opts.warmup, iterations: opts.iterations, nLayers: opts.nLayers)
    } catch {
        printStderr("Diagnostic Exp 2 (Batched-Head) failed: \(error)")
    }

    // Exp 3: Whole-Model
    do {
        diagnosticWholeModel = try DiagnosticBench.runWholeModel(
            warmup: opts.warmup, iterations: opts.iterations, nLayers: opts.nLayers)
    } catch {
        printStderr("Diagnostic Exp 3 (Whole-Model) failed: \(error)")
    }

    // Print comparative table
    printStderr("\n=== Diagnostic Results Summary ===")
    printStderr(String(format: "%-25s %10s %10s %10s %10s",
        "Experiment", "Median", "Mean", "P95", "ms/layer"))

    let nL = Double(opts.nLayers)

    if let fused = fusedResult {
        let r = fused.benchmarkResult
        printStderr(String(format: "%-25s %10.3f %10.3f %10.3f %10.3f",
            "Fused (baseline)", r.median, r.mean, r.p95, r.median / nL))
    }

    if let co = diagnosticConvsOnly {
        let r = co.benchmarkResult
        printStderr(String(format: "%-25s %10.3f %10.3f %10.3f %10.3f",
            "Convs-Only", r.median, r.mean, r.p95, r.median / nL))
    }

    if let bh = diagnosticBatchedHead {
        let r = bh.benchmarkResult
        printStderr(String(format: "%-25s %10.3f %10.3f %10.3f %10.3f",
            "Batched-Head", r.median, r.mean, r.p95, r.median / nL))
    }

    if let wm = diagnosticWholeModel {
        let r = wm.benchmarkResult
        printStderr(String(format: "%-25s %10.3f %10.3f %10.3f %10.3f",
            "Whole-Model", r.median, r.mean, r.p95, r.median / nL))
    }

    // Overhead decomposition
    if let fused = fusedResult, let co = diagnosticConvsOnly {
        let fusedPerLayer = fused.benchmarkResult.median / nL
        let convsPerLayer = co.benchmarkResult.median / nL
        let attnOverhead = fusedPerLayer - convsPerLayer
        printStderr(String(format: "\n  Per-layer budget decomposition:"))
        printStderr(String(format: "    Pure convolution compute: %.3f ms/layer", convsPerLayer))
        printStderr(String(format: "    Attention + norm overhead: %.3f ms/layer", attnOverhead))
        printStderr(String(format: "    Hardware floor: 0.213 ms/layer"))
    }

    if let fused = fusedResult, let wm = diagnosticWholeModel {
        let dispatchOverhead = (fused.benchmarkResult.median - wm.benchmarkResult.median) / max(1.0, nL - 1.0)
        printStderr(String(format: "    Dispatch overhead per layer: %.3f ms", dispatchOverhead))
    }

    if let fused = fusedResult, let bh = diagnosticBatchedHead {
        let delta = bh.benchmarkResult.median - fused.benchmarkResult.median
        printStderr(String(format: "    Batched vs Split-Head delta: %+.3f ms", delta))
    }

    printStderr("")
}

// --- Benchmark 2: Core ML (optional) ---
var coreMLResult: CoreMLBench.Result? = nil
if !opts.aneOnly {
    do {
        coreMLResult = try CoreMLBench.run(runner: runner, modelPath: opts.coreMLModelPath)
    } catch {
        printStderr("Core ML benchmark failed: \(error)")
        printStderr("Continuing with ANE-only results...")
    }
}

// --- Benchmark 3: Sustained thermal test (optional) ---
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

// --- Output Report ---
let inferenceReportData: (result: BenchmarkResult, breakdown: (ane: Double, io: Double, elem: Double), compileMs: Double)?
if let inf = inferenceResult {
    inferenceReportData = (result: inf.benchmarkResult, breakdown: inf.avgTimingBreakdown, compileMs: inf.compileTimeMs)
} else {
    inferenceReportData = nil
}

let report = ResultsFormatter.formatReport(
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
print(report)

// --- Save Results ---
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

    // ANE Direct CSV
    try ResultsFormatter.writeCSV(
        latencies: aneResult.benchmarkResult.latencies,
        to: "\(outputDir)/ane_direct_latencies.csv"
    )

    // ANE Inference CSV
    if let inf = inferenceResult {
        try ResultsFormatter.writeCSV(
            latencies: inf.benchmarkResult.latencies,
            to: "\(outputDir)/ane_inference_latencies.csv"
        )
    }

    // Core ML CSVs
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

    // Diagnostic CSVs
    if let co = diagnosticConvsOnly {
        try ResultsFormatter.writeCSV(
            latencies: co.benchmarkResult.latencies,
            to: "\(outputDir)/diagnostic_convs_only_latencies.csv"
        )
    }
    if let bh = diagnosticBatchedHead {
        try ResultsFormatter.writeCSV(
            latencies: bh.benchmarkResult.latencies,
            to: "\(outputDir)/diagnostic_batched_head_latencies.csv"
        )
    }
    if let wm = diagnosticWholeModel {
        try ResultsFormatter.writeCSV(
            latencies: wm.benchmarkResult.latencies,
            to: "\(outputDir)/diagnostic_whole_model_latencies.csv"
        )
    }

    // Summary report
    try report.write(toFile: "\(outputDir)/summary.txt", atomically: true, encoding: .utf8)

    printStderr("\nResults saved to: \(outputDir)/")
} catch {
    printStderr("Failed to save results: \(error)")
}
