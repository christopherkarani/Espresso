import Darwin
import Foundation
import ANETypes
import MILGenerator

/// One compiled Phase-11 `max_N = 1` fused layer (QKV + attention + FFN).
public struct FusedHybridDecodeLayerKernelSet: ~Copyable {
    public static let phase11MaxN = 1
    /// Telemetry label; must stay aligned with `Trunk.fusedHybrid.rawValue` in ModelSupport.
    public static let decodePathLabel = "fused"
    public static let fallbackStage = "fused_hybrid_decode"

    public static func hopsPerToken(nLayer: Int, blockSize: Int = phase11MaxN) -> Int {
        precondition(nLayer > 0 && blockSize > 0)
        return (nLayer + blockSize - 1) / blockSize
    }

    public struct DonorHexIDs: Sendable {
        public let fusedLayer: String
    }

    public let fusedLayer: ANEKernel
    public let maxSeq: Int
    public let laneSpatial: Int
    public let dim: Int
    public let kvDim: Int
    public let donorHexIDs: DonorHexIDs

    public init(
        weights: borrowing LayerWeights,
        maxSeq: Int,
        nHeads: Int,
        nKVHeads: Int,
        headDim: Int,
        donorHexIDs: DonorHexIDs? = nil
    ) throws(ANEError) {
        guard maxSeq > 0 else {
            throw .invalidArguments("fused hybrid decode maxSeq must be > 0")
        }
        guard nHeads > 0, nKVHeads > 0, nHeads % nKVHeads == 0, headDim > 0 else {
            throw .invalidArguments("fused hybrid decode head geometry is invalid")
        }
        let laneSpatial = HybridDecodeKernelSet.resolvedLaneSpatialForCurrentProcess()
        let compiled = try Self.compile(
            weights: weights,
            maxSeq: maxSeq,
            laneSpatial: laneSpatial,
            nHeads: nHeads,
            nKVHeads: nKVHeads,
            headDim: headDim,
            donorHexId: donorHexIDs?.fusedLayer
        )
        let hexId = compiled.hexId
        self.fusedLayer = compiled
        self.maxSeq = maxSeq
        self.laneSpatial = laneSpatial
        self.dim = weights.dim
        self.kvDim = weights.kvDim
        self.donorHexIDs = DonorHexIDs(fusedLayer: hexId)
    }

    private static func compile(
        weights: borrowing LayerWeights,
        maxSeq: Int,
        laneSpatial: Int,
        nHeads: Int,
        nKVHeads: Int,
        headDim: Int,
        donorHexId: String?
    ) throws(ANEError) -> ANEKernel {
        let spec = makeSpec(
            weights: weights,
            maxSeq: maxSeq,
            laneSpatial: laneSpatial,
            nHeads: nHeads,
            nKVHeads: nKVHeads,
            headDim: headDim
        )
        let donorDisabled = ProcessInfo.processInfo.environment["ESPRESSO_DISABLE_HYBRID_DONOR_DELTA"] == "1"
        if !donorDisabled, let donorHexId, !donorHexId.isEmpty {
            do {
                return try ANEKernel(
                    milText: spec.milText,
                    weights: spec.weights,
                    inputSizes: spec.inputSizes,
                    outputSizes: spec.outputSizes,
                    compileLabel: "fused.hybrid.layer.n1.delta",
                    donorHexId: donorHexId
                )
            } catch {
                // Delta is best-effort; cold compile is the real path.
            }
        }
        do {
            return try ANEKernel(
                milText: spec.milText,
                weights: spec.weights,
                inputSizes: spec.inputSizes,
                outputSizes: spec.outputSizes,
                compileLabel: "fused.hybrid.layer.n1.cold"
            )
        } catch {
            let milPath = dumpFailedMIL(spec)
            fputs("[FusedHybridDecode] N=1 serve compile failed. MIL dump: \(milPath)\n", stderr)
            throw error
        }
    }

    private struct CompileSpec {
        let milText: String
        let weights: [(path: String, data: Data)]
        let inputSizes: [Int]
        let outputSizes: [Int]
    }

    private static func makeSpec(
        weights: borrowing LayerWeights,
        maxSeq: Int,
        laneSpatial: Int,
        nHeads: Int,
        nKVHeads: Int,
        headDim: Int
    ) -> CompileSpec {
        let generator = FusedHybridDecodeLayerGenerator(
            dim: weights.dim,
            qDim: weights.qDim,
            kvDim: weights.kvDim,
            hiddenDim: weights.hiddenDim,
            nHeads: nHeads,
            nKVHeads: nKVHeads,
            headDim: headDim,
            maxSeq: maxSeq,
            laneSpatial: laneSpatial,
            normEps: weights.normEps,
            hasQKVBias: weights.hasQKVBias
        )
        return CompileSpec(
            milText: generator.milText,
            weights: weightBlobs(weights),
            inputSizes: generator.inputByteSizes,
            outputSizes: generator.outputByteSizes
        )
    }

    private static func weightBlobs(_ weights: borrowing LayerWeights) -> [(path: String, data: Data)] {
        let dim = weights.dim
        let qDim = weights.qDim
        let kvDim = weights.kvDim
        let hidden = weights.hiddenDim
        var blobs: [(path: String, data: Data)] = [
            ("@model_path/weights/rms1.bin", buildBlob(from: weights.rmsAtt, rows: 1, cols: dim)),
            ("@model_path/weights/wq.bin", buildBlob(from: weights.Wq, rows: qDim, cols: dim)),
            ("@model_path/weights/wk.bin", buildBlob(from: weights.Wk, rows: kvDim, cols: dim)),
            ("@model_path/weights/wv.bin", buildBlob(from: weights.Wv, rows: kvDim, cols: dim)),
            ("@model_path/weights/wo.bin", buildBlob(from: weights.Wo, rows: dim, cols: qDim)),
            ("@model_path/weights/rms2.bin", buildBlob(from: weights.rmsFfn, rows: 1, cols: dim)),
            ("@model_path/weights/w1.bin", buildBlob(from: weights.W1, rows: hidden, cols: dim)),
            ("@model_path/weights/w3.bin", buildBlob(from: weights.W3, rows: hidden, cols: dim)),
            ("@model_path/weights/w2.bin", buildBlob(from: weights.W2, rows: dim, cols: hidden)),
        ]
        if weights.hasQKVBias {
            blobs += [
                ("@model_path/weights/bq.bin", buildBlob(from: weights.bq, rows: 1, cols: qDim)),
                ("@model_path/weights/bk.bin", buildBlob(from: weights.bk, rows: 1, cols: kvDim)),
                ("@model_path/weights/bv.bin", buildBlob(from: weights.bv, rows: 1, cols: kvDim)),
            ]
        }
        return blobs
    }

    private static func buildBlob(from buffer: borrowing TensorBuffer, rows: Int, cols: Int) -> Data {
        buffer.withUnsafeBufferPointer { ptr in
            WeightBlob.build(from: ptr, rows: rows, cols: cols)
        }
    }

    private static func dumpFailedMIL(_ spec: CompileSpec) -> String {
        let stamp = Int(Date().timeIntervalSince1970)
        let filename = "espresso-fused-hybrid-serve-n1-\(stamp).mil"
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(filename)
        try? spec.milText.write(to: url, atomically: true, encoding: .utf8)
        return url.path
    }
}
