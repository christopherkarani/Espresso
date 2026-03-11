import Foundation
import ANETypes

struct BenchmarkOptions {
    var aneOnly = false
    var sustained = false
    var warmup = 50
    var iterations = 1_000
    var outputDirectory: String?
    var modelPath = "benchmarks/models/transformer_layer.mlpackage"
    var layers = ModelConfig.nLayers

    static func parse(arguments: [String]) throws -> BenchmarkOptions {
        var options = BenchmarkOptions()
        var index = 1

        while index < arguments.count {
            switch arguments[index] {
            case "--ane-only":
                options.aneOnly = true
            case "--sustained":
                options.sustained = true
            case "--warmup":
                index += 1
                options.warmup = try parseInt(arguments, index: index, flag: "--warmup")
            case "--iterations":
                index += 1
                options.iterations = try parseInt(arguments, index: index, flag: "--iterations")
            case "--output":
                index += 1
                guard index < arguments.count else {
                    throw CLIError.missingValue("--output")
                }
                options.outputDirectory = arguments[index]
            case "--model":
                index += 1
                guard index < arguments.count else {
                    throw CLIError.missingValue("--model")
                }
                options.modelPath = arguments[index]
            case "--layers":
                index += 1
                options.layers = try parseInt(arguments, index: index, flag: "--layers")
            case "-h", "--help":
                throw CLIError.helpRequested
            default:
                throw CLIError.unknownFlag(arguments[index])
            }
            index += 1
        }

        guard options.warmup >= 0 else {
            throw CLIError.invalidValue("--warmup", "must be >= 0")
        }
        guard options.iterations > 0 else {
            throw CLIError.invalidValue("--iterations", "must be > 0")
        }
        guard options.layers > 0 else {
            throw CLIError.invalidValue("--layers", "must be > 0")
        }

        return options
    }

    static func usage() -> String {
        """
        Usage: espresso-bench [options]

          --ane-only           Skip Core ML baselines
          --sustained          Run a 60s sustained thermal pass after the benchmark
          --warmup N           Warmup iterations (default: 50)
          --iterations N       Measured iterations (default: 1000)
          --output DIR         Output directory for CSV and summary.txt
          --model PATH         Core ML model path (default: benchmarks/models/transformer_layer.mlpackage)
          --layers N           Number of transformer layers to benchmark (default: \(ModelConfig.nLayers))
          -h, --help           Show this help
        """
    }

    private static func parseInt(_ arguments: [String], index: Int, flag: String) throws -> Int {
        guard index < arguments.count else {
            throw CLIError.missingValue(flag)
        }
        guard let value = Int(arguments[index]) else {
            throw CLIError.invalidValue(flag, "expected integer")
        }
        return value
    }
}

enum CLIError: Error, CustomStringConvertible {
    case helpRequested
    case missingValue(String)
    case invalidValue(String, String)
    case unknownFlag(String)

    var description: String {
        switch self {
        case .helpRequested:
            return BenchmarkOptions.usage()
        case .missingValue(let flag):
            return "Missing value for \(flag)\n\n\(BenchmarkOptions.usage())"
        case .invalidValue(let flag, let message):
            return "Invalid value for \(flag): \(message)\n\n\(BenchmarkOptions.usage())"
        case .unknownFlag(let flag):
            return "Unknown argument: \(flag)\n\n\(BenchmarkOptions.usage())"
        }
    }
}

enum EspressoBenchMain {
    static func run(options: BenchmarkOptions) throws {
        let runner = BenchmarkRunner(warmup: options.warmup, iterations: options.iterations)
        let aneResult = try ANEDirectBench.run(runner: runner, nLayers: options.layers)

        var coreMLResult: CoreMLBench.Result? = nil
        if !options.aneOnly {
            do {
                coreMLResult = try CoreMLBench.run(runner: runner, modelPath: options.modelPath)
            } catch {
                printStderr("Skipping Core ML baseline: \(error)")
            }
        }

        var thermalBefore: String?
        var thermalAfter: String?
        if options.sustained {
            let sustained = try ANEDirectBench.runSustained(duration: 60.0, nLayers: options.layers)
            thermalBefore = sustained.before
            thermalAfter = sustained.after
        }

        let flopsPerPass = FLOPCalculator.forwardPassFLOPs() * Double(options.layers)
        let report = ResultsFormatter.formatReport(
            aneResult: aneResult.benchmarkResult,
            aneTimingBreakdown: aneResult.avgTimingBreakdown,
            coreMLResults: coreMLResult?.results ?? [],
            coreMLLoadTimeMs: coreMLResult?.modelLoadTimeMs,
            thermalBefore: thermalBefore,
            thermalAfter: thermalAfter,
            flopsPerPass: flopsPerPass,
            nLayers: options.layers
        )
        print(report, terminator: "")

        let outputDirectory = try resolvedOutputDirectory(from: options.outputDirectory)
        try FileManager.default.createDirectory(
            at: outputDirectory,
            withIntermediateDirectories: true
        )

        try ResultsFormatter.writeCSV(
            latencies: aneResult.benchmarkResult.latencies,
            to: outputDirectory.appendingPathComponent("ane_direct.csv").path
        )

        if let coreMLResult {
            for (label, result) in coreMLResult.results {
                try ResultsFormatter.writeCSV(
                    latencies: result.latencies,
                    to: outputDirectory.appendingPathComponent(csvFileName(for: label)).path
                )
            }
        }

        try report.write(
            to: outputDirectory.appendingPathComponent("summary.txt"),
            atomically: true,
            encoding: String.Encoding.utf8
        )
    }

    private static func resolvedOutputDirectory(from explicitPath: String?) throws -> URL {
        if let explicitPath {
            return URL(fileURLWithPath: explicitPath, isDirectory: true)
        }

        let formatter = DateFormatter()
        formatter.calendar = Calendar(identifier: .gregorian)
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.timeZone = TimeZone.current
        formatter.dateFormat = "yyyy-MM-dd-HHmmss"

        let timestamp = formatter.string(from: Date())
        return URL(fileURLWithPath: "benchmarks/results/\(timestamp)", isDirectory: true)
    }

    private static func csvFileName(for label: String) -> String {
        let lowered = label.lowercased()
        let sanitized = lowered.map { character -> Character in
            if character.isLetter || character.isNumber {
                return character
            }
            return "_"
        }
        let collapsed = String(sanitized).replacingOccurrences(
            of: "_+",
            with: "_",
            options: .regularExpression
        )
        let trimmed = collapsed.trimmingCharacters(in: CharacterSet(charactersIn: "_"))
        return "\(trimmed).csv"
    }
}

do {
    let options = try BenchmarkOptions.parse(arguments: CommandLine.arguments)
    try EspressoBenchMain.run(options: options)
} catch CLIError.helpRequested {
    print(BenchmarkOptions.usage())
} catch {
    printStderr(String(describing: error))
    Foundation.exit(1)
}
