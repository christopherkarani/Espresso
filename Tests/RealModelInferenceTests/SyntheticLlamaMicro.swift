import Foundation
import ModelSupport

/// Deterministic micro llama artifact for golden decode traces.
///
/// Builds a valid llama-family weight directory from a fixed seed so every machine
/// materializes byte-identical weights. The exact-CPU trunk decodes it without any
/// ANE dependency, which makes recorded traces replayable in hosted CI.
enum SyntheticLlamaMicro {
    static let name = "synthetic-llama-micro"

    static var config: MultiModelConfig {
        MultiModelConfig(
            name: name,
            nLayer: 2,
            nHead: 4,
            nKVHead: 2,
            dModel: 64,
            headDim: 16,
            hiddenDim: 176,
            vocab: 256,
            maxSeq: 512,
            normEps: 1e-5,
            ropeTheta: 10_000.0,
            eosToken: nil,
            architecture: .llama
        )
    }

    /// Fixed-seed uniform values in [-scale, scale].
    private static func values(_ count: Int, seed: UInt64, scale: Float) -> [Float] {
        var state = seed &+ 0x9E3779B97F4A7C15
        func next() -> Float {
            state &+= 0x9E3779B97F4A7C15
            var z = state
            z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
            z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
            z ^= z >> 31
            let unit = Float(z >> 40) / Float(1 << 24)
            return (unit * 2 - 1) * scale
        }
        return (0..<count).map { _ in next() }
    }

    private static func normValues(_ count: Int, seed: UInt64) -> [Float] {
        // Near-one gammas keep activations stable while still varying per lane.
        values(count, seed: seed, scale: 0.05).map { 1.0 + $0 }
    }

    /// Writes one fp16 blob with the 128-byte header BlobWeightLoader expects.
    private static func writeBlob(_ floats: [Float], to path: URL) throws {
        var payload = Data(capacity: floats.count * 2)
        for value in floats {
            var bits = Float16(value).bitPattern.littleEndian
            withUnsafeBytes(of: &bits) { payload.append(contentsOf: $0) }
        }

        var header = Data(repeating: 0, count: 128)
        header.withUnsafeMutableBytes { raw in
            func store(_ value: UInt32, at offset: Int) {
                var le = value.littleEndian
                raw.storeBytes(of: le, toByteOffset: offset, as: UInt32.self)
            }
            store(0xDEADBEEF, at: 64)
            store(UInt32(payload.count), at: 72)
            store(128, at: 80)
        }

        try FileManager.default.createDirectory(
            at: path.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        header.append(payload)
        try header.write(to: path)
    }

    /// Materializes the weight directory and returns its path.
    static func makeBundle(in root: URL) throws -> URL {
        let weightDir = root.appendingPathComponent("weights", isDirectory: true)
        let fm = FileManager.default
        try fm.createDirectory(at: weightDir, withIntermediateDirectories: true)

        let c = config
        try writeBlob(values(c.vocab * c.dModel, seed: 1, scale: 0.05), to: weightDir.appendingPathComponent("embeddings/token.bin"))
        try writeBlob(normValues(c.dModel, seed: 2), to: weightDir.appendingPathComponent("rms_final.bin"))
        try writeBlob(values(c.vocab * c.dModel, seed: 3, scale: 0.05), to: weightDir.appendingPathComponent("lm_head.bin"))

        for layer in 0..<c.nLayer {
            let layerSeed = UInt64(100 + layer * 10)
            let dir = weightDir.appendingPathComponent("layers/\(layer)", isDirectory: true)
            try writeBlob(normValues(c.dModel, seed: layerSeed + 1), to: dir.appendingPathComponent("rms_att.bin"))
            try writeBlob(values(c.dModel * c.attentionDim, seed: layerSeed + 2, scale: 0.08), to: dir.appendingPathComponent("wq.bin"))
            try writeBlob(values(c.dModel * c.kvDim, seed: layerSeed + 3, scale: 0.08), to: dir.appendingPathComponent("wk.bin"))
            try writeBlob(values(c.dModel * c.kvDim, seed: layerSeed + 4, scale: 0.08), to: dir.appendingPathComponent("wv.bin"))
            try writeBlob(values(c.dModel * c.attentionDim, seed: layerSeed + 5, scale: 0.08), to: dir.appendingPathComponent("wo.bin"))
            try writeBlob(normValues(c.dModel, seed: layerSeed + 6), to: dir.appendingPathComponent("rms_ffn.bin"))
            try writeBlob(values(c.hiddenDim * c.dModel, seed: layerSeed + 7, scale: 0.08), to: dir.appendingPathComponent("w1.bin"))
            try writeBlob(values(c.dModel * c.hiddenDim, seed: layerSeed + 8, scale: 0.08), to: dir.appendingPathComponent("w2.bin"))
            try writeBlob(values(c.hiddenDim * c.dModel, seed: layerSeed + 9, scale: 0.08), to: dir.appendingPathComponent("w3.bin"))
        }
        return weightDir
    }
}
