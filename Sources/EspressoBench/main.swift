import Foundation
import ANETypes
import ANERuntime
import Espresso

// MARK: - CLI Argument Parsing

struct BenchmarkOptions {
    var aneOnly: Bool = false
    var inference: Bool = false
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

    // Summary report
    try report.write(toFile: "\(outputDir)/summary.txt", atomically: true, encoding: .utf8)

    printStderr("\nResults saved to: \(outputDir)/")
} catch {
    printStderr("Failed to save results: \(error)")
}
