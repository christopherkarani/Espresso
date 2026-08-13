import Foundation

public struct LayerWeightPaths: Sendable, Equatable {
    public let rmsAtt: String
    public let qNorm: String?
    public let kNorm: String?
    public let wq: String
    public let wk: String
    public let wv: String
    public let wo: String
    public let bq: String?
    public let bk: String?
    public let bv: String?
    public let bo: String?
    public let rmsFfn: String
    public let w1: String
    public let w2: String
    public let w3: String?
    public let b1: String?
    public let b2: String?

    public static func forLayer(
        _ layer: Int,
        config: MultiModelConfig,
        blobDir: String
    ) -> LayerWeightPaths {
        let layerDir = URL(fileURLWithPath: blobDir, isDirectory: true)
            .appendingPathComponent("layers", isDirectory: true)
            .appendingPathComponent(String(layer), isDirectory: true)

        func path(_ name: String) -> String {
            layerDir.appendingPathComponent(name).path
        }

        switch config.architecture {
        case .gpt2:
            return LayerWeightPaths(
                rmsAtt: path("ln_1_gamma.bin"),
                qNorm: nil,
                kNorm: nil,
                wq: path("wq.bin"),
                wk: path("wk.bin"),
                wv: path("wv.bin"),
                wo: path("wo.bin"),
                bq: path("bq.bin"),
                bk: path("bk.bin"),
                bv: path("bv.bin"),
                bo: path("bo.bin"),
                rmsFfn: path("ln_2_gamma.bin"),
                w1: path("w1.bin"),
                w2: path("w2.bin"),
                w3: nil,
                b1: path("b1.bin"),
                b2: path("b2.bin")
            )
        case .llama:
            return LayerWeightPaths(
                rmsAtt: path("rms_att.bin"),
                qNorm: path("q_norm.bin"),
                kNorm: path("k_norm.bin"),
                wq: path("wq.bin"),
                wk: path("wk.bin"),
                wv: path("wv.bin"),
                wo: path("wo.bin"),
                // Qwen2-family checkpoints bias q/k/v (and only q/k/v). Plain llama
                // checkpoints have no bias files, so presence decides: loaders treat
                // bq/bk/bv as an all-or-nothing group, like q_norm/k_norm.
                bq: path("bq.bin"),
                bk: path("bk.bin"),
                bv: path("bv.bin"),
                bo: nil,
                rmsFfn: path("rms_ffn.bin"),
                w1: path("w1.bin"),
                w2: path("w2.bin"),
                w3: path("w3.bin"),
                b1: nil,
                b2: nil
            )
        }
    }

    var layerDirectory: URL {
        URL(fileURLWithPath: wq).deletingLastPathComponent()
    }

    var blobRootDirectory: URL {
        layerDirectory.deletingLastPathComponent().deletingLastPathComponent()
    }

    var attentionNormBiasPath: String? {
        replacingGammaSuffix(in: rmsAtt)
    }

    var ffnNormBiasPath: String? {
        replacingGammaSuffix(in: rmsFfn)
    }

    func causalMaskPath(spatial: Int) -> String {
        blobRootDirectory
            .appendingPathComponent("masks", isDirectory: true)
            .appendingPathComponent("causal_\(spatial).bin")
            .path
    }

    private func replacingGammaSuffix(in path: String) -> String? {
        guard path.hasSuffix("_gamma.bin") else { return nil }
        return path.replacingOccurrences(of: "_gamma.bin", with: "_beta.bin")
    }
}
