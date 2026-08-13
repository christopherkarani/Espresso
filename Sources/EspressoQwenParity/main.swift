import Foundation
import ModelSupport
import RealModelInference

// Per-layer parity driver.
//
// Reads a flat little-endian float32 file of layer inputs produced by the PyTorch
// reference (`scripts/qwen25_pytorch_reference.py`), runs each requested transformer
// layer through Espresso's CPU oracle or the ANE hybrid kernel, and writes the layer
// outputs back in the same layout. The comparison and report live in Python; this tool
// only produces Espresso's numbers.
//
// Layout for both input and output files, layer-major then position-major:
//   value(layerSlot, position, channel)
//     = file[((layerSlot * positions) + position) * dModel + channel]
// where `layerSlot` indexes the `--layers` list, not the absolute layer number.

private enum Backend: String {
    case cpuFP32 = "cpu-fp32"
    case cpuFP16 = "cpu-fp16"
    case ane
}

private struct Options {
    var nativeDir: String
    var inputsPath: String
    var outputPath: String
    var positions: Int
    var backend: Backend
    var layers: [Int]?
}

private func fail(_ message: String) -> Never {
    FileHandle.standardError.write(Data("error: \(message)\n".utf8))
    exit(1)
}

private let usage = """
usage: espresso-qwen-parity --native-dir <dir> --inputs <file.f32> --out <file.f32> \
--positions <n> --backend cpu-fp32|cpu-fp16|ane [--layers 0,1,2]
"""

private func parseOptions(_ argv: [String]) -> Options {
    var nativeDir: String?
    var inputsPath: String?
    var outputPath: String?
    var positions: Int?
    var backend: Backend?
    var layers: [Int]?

    var index = 1
    while index < argv.count {
        let flag = argv[index]
        func value() -> String {
            index += 1
            guard index < argv.count else { fail("\(flag) requires a value\n\(usage)") }
            return argv[index]
        }
        switch flag {
        case "--native-dir": nativeDir = value()
        case "--inputs": inputsPath = value()
        case "--out": outputPath = value()
        case "--positions":
            let raw = value()
            guard let parsed = Int(raw), parsed > 0 else { fail("--positions must be a positive integer, got \(raw)") }
            positions = parsed
        case "--backend":
            let raw = value()
            guard let parsed = Backend(rawValue: raw) else { fail("unknown --backend \(raw)\n\(usage)") }
            backend = parsed
        case "--layers":
            let raw = value()
            let parsed = raw.split(separator: ",").map { entry -> Int in
                guard let layer = Int(entry.trimmingCharacters(in: .whitespaces)) else {
                    fail("--layers expects comma-separated integers, got \(raw)")
                }
                return layer
            }
            guard !parsed.isEmpty else { fail("--layers must name at least one layer") }
            layers = parsed
        case "-h", "--help":
            print(usage)
            exit(0)
        default:
            fail("unknown flag \(flag)\n\(usage)")
        }
        index += 1
    }

    guard let nativeDir else { fail("--native-dir is required\n\(usage)") }
    guard let inputsPath else { fail("--inputs is required\n\(usage)") }
    guard let outputPath else { fail("--out is required\n\(usage)") }
    guard let positions else { fail("--positions is required\n\(usage)") }
    guard let backend else { fail("--backend is required\n\(usage)") }

    return Options(
        nativeDir: nativeDir,
        inputsPath: inputsPath,
        outputPath: outputPath,
        positions: positions,
        backend: backend,
        layers: layers
    )
}

private func readFloat32File(at path: String, expectedCount: Int) -> [Float] {
    guard let data = FileManager.default.contents(atPath: path) else {
        fail("failed to read \(path)")
    }
    let expectedBytes = expectedCount * 4
    guard data.count == expectedBytes else {
        fail("\(path) has \(data.count) bytes, expected \(expectedBytes) (\(expectedCount) float32)")
    }
    var values = [Float](repeating: 0, count: expectedCount)
    _ = values.withUnsafeMutableBytes { destination in
        data.copyBytes(to: destination)
    }
    return values
}

private func writeFloat32File(_ values: [Float], to path: String) {
    let data = values.withUnsafeBufferPointer { buffer in
        Data(buffer: buffer)
    }
    do {
        try data.write(to: URL(fileURLWithPath: path))
    } catch {
        fail("failed to write \(path): \(error)")
    }
}

private let options = parseOptions(CommandLine.arguments)

private let config: MultiModelConfig
do {
    config = try QwenLayerParityProbe.loadConfig(nativeDir: options.nativeDir)
} catch {
    fail("failed to load metadata.json from \(options.nativeDir): \(error)")
}

let requestedLayers = options.layers ?? Array(0..<config.nLayer)
for layer in requestedLayers where layer < 0 || layer >= config.nLayer {
    fail("layer \(layer) out of range for nLayer \(config.nLayer)")
}

let perLayerCount = options.positions * config.dModel
let flatInputs = readFloat32File(
    at: options.inputsPath,
    expectedCount: requestedLayers.count * perLayerCount
)

FileHandle.standardError.write(
    Data(
        """
        espresso-qwen-parity: model=\(config.name) backend=\(options.backend.rawValue) \
        layers=\(requestedLayers.count) positions=\(options.positions) dModel=\(config.dModel)

        """.utf8
    )
)

var flatOutputs = [Float]()
flatOutputs.reserveCapacity(requestedLayers.count * perLayerCount)

for (slot, layer) in requestedLayers.enumerated() {
    let base = slot * perLayerCount
    var inputs: [[Float]] = []
    inputs.reserveCapacity(options.positions)
    for position in 0..<options.positions {
        let start = base + position * config.dModel
        inputs.append(Array(flatInputs[start..<(start + config.dModel)]))
    }

    let result: QwenLayerParityProbe.LayerOutputs
    do {
        switch options.backend {
        case .cpuFP32:
            result = try QwenLayerParityProbe.evalCPULayer(
                config: config,
                nativeDir: options.nativeDir,
                layer: layer,
                inputs: inputs,
                roundIntermediatesToFP16: false
            )
        case .cpuFP16:
            result = try QwenLayerParityProbe.evalCPULayer(
                config: config,
                nativeDir: options.nativeDir,
                layer: layer,
                inputs: inputs,
                roundIntermediatesToFP16: true
            )
        case .ane:
            result = try QwenLayerParityProbe.evalANELayer(
                config: config,
                nativeDir: options.nativeDir,
                layer: layer,
                inputs: inputs
            )
        }
    } catch {
        fail("layer \(layer) failed on backend \(options.backend.rawValue): \(error)")
    }

    guard result.outputs.count == options.positions else {
        fail("layer \(layer) returned \(result.outputs.count) positions, expected \(options.positions)")
    }
    for (position, output) in result.outputs.enumerated() {
        guard output.count == config.dModel else {
            fail("layer \(layer) position \(position) returned \(output.count) values, expected \(config.dModel)")
        }
        flatOutputs.append(contentsOf: output)
    }
    FileHandle.standardError.write(Data("  layer \(layer): ok (\(result.backend))\n".utf8))
}

writeFloat32File(flatOutputs, to: options.outputPath)
FileHandle.standardError.write(Data("wrote \(flatOutputs.count) float32 to \(options.outputPath)\n".utf8))
