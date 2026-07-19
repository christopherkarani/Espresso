import Foundation
import ANETypes
import MILGenerator

/// Fused decode kernel set with REAL attention computed on ANE.
///
/// Single kernel per layer: RMSNorm → QKV → Attention → Wo → residual → RMSNorm → SwiGLU FFN → residual
/// Outputs: xNext, K projection, V projection (for cache update)
public struct FusedDecodeRealAttentionKernelSet: ~Copyable {
    public static let defaultLaneSpatial = 32

    internal enum KernelKind: String, CaseIterable {
        case fusedDecodeRealAttention
    }

    internal struct CompileSpec {
        internal let kind: KernelKind
        internal let milText: String
        internal let weights: [(path: String, data: Data)]
        internal let inputSizes: [Int]
        internal let outputSizes: [Int]
    }

    public let fusedLayer: ANEKernel
    public let maxSeq: Int
    public let kernelMaxSeq: Int
    public let laneSpatial: Int

    private init(
        fusedLayer: consuming ANEKernel,
        logicalMaxSeq: Int,
        kernelMaxSeq: Int,
        laneSpatial: Int
    ) {
        self.fusedLayer = fusedLayer
        self.maxSeq = logicalMaxSeq
        self.kernelMaxSeq = kernelMaxSeq
        self.laneSpatial = laneSpatial
    }

    public init(weights: borrowing LayerWeights, maxSeq: Int = ModelConfig.seqLen) throws(ANEError) {
        guard maxSeq > 0 else {
            throw .invalidArguments("fused decode real attention maxSeq must be > 0")
        }
        let laneSpatial = Self.resolvedLaneSpatialForCurrentProcess()
        guard maxSeq >= laneSpatial else {
            throw .invalidArguments(
                "fused decode real attention maxSeq (\(maxSeq)) must be >= laneSpatial (\(laneSpatial))"
            )
        }
        guard maxSeq % laneSpatial == 0 else {
            throw .invalidArguments(
                "fused decode real attention maxSeq (\(maxSeq)) must be a multiple of laneSpatial (\(laneSpatial))"
            )
        }
        let kernelMaxSeq = laneSpatial
        let compiled = try Self.compileFusedDecodeRealAttention(
            weights: weights,
            maxSeq: kernelMaxSeq,
            laneSpatial: laneSpatial
        )
        self.init(
            fusedLayer: compiled,
            logicalMaxSeq: maxSeq,
            kernelMaxSeq: kernelMaxSeq,
            laneSpatial: laneSpatial
        )
    }

    internal static func compileSpecs(weights: borrowing LayerWeights, maxSeq: Int) -> [CompileSpec] {
        let laneSpatial = resolvedLaneSpatialForCurrentProcess()
        precondition(maxSeq > 0)
        precondition(maxSeq >= laneSpatial)
        return [
            makeFusedDecodeRealAttentionSpec(weights: weights, maxSeq: laneSpatial, laneSpatial: laneSpatial),
        ]
    }

    private static func compileFusedDecodeRealAttention(
        weights: borrowing LayerWeights,
        maxSeq: Int,
        laneSpatial: Int
    ) throws(ANEError) -> ANEKernel {
        let spec = makeFusedDecodeRealAttentionSpec(weights: weights, maxSeq: maxSeq, laneSpatial: laneSpatial)
        return try ANEKernel(
            milText: spec.milText,
            weights: spec.weights,
            inputSizes: spec.inputSizes,
            outputSizes: spec.outputSizes
        )
    }

    private static func makeFusedDecodeRealAttentionSpec(
        weights: borrowing LayerWeights,
        maxSeq: Int,
        laneSpatial: Int
    ) -> CompileSpec {
        let dim = ModelConfig.dim
        let hidden = ModelConfig.hidden
        let heads = ModelConfig.dim / ModelConfig.headDim
        let headDim = ModelConfig.headDim

        let generator = FusedDecodeLayerRealAttentionGenerator(
            maxSeq: maxSeq, laneSpatial: laneSpatial, nHeads: heads, headDim: headDim
        )

        let rms1Blob = buildBlob(from: weights.rmsAtt, rows: 1, cols: dim)
        let wqBlob   = buildBlob(from: weights.Wq,     rows: dim, cols: dim)
        let wkBlob   = buildBlob(from: weights.Wk,     rows: dim, cols: dim)
        let wvBlob   = buildBlob(from: weights.Wv,     rows: dim, cols: dim)
        let woBlob   = buildBlob(from: weights.Wo,     rows: dim, cols: dim)
        let rms2Blob = buildBlob(from: weights.rmsFfn, rows: 1,   cols: dim)
        let w1Blob   = buildBlob(from: weights.W1,     rows: hidden, cols: dim)
        let w3Blob   = buildBlob(from: weights.W3,     rows: hidden, cols: dim)
        let w2Blob   = buildBlob(from: weights.W2,     rows: dim,    cols: hidden)

        return CompileSpec(
            kind: .fusedDecodeRealAttention,
            milText: generator.milText,
            weights: [
                (path: "@model_path/weights/rms1.bin", data: rms1Blob),
                (path: "@model_path/weights/wq.bin",   data: wqBlob),
                (path: "@model_path/weights/wk.bin",   data: wkBlob),
                (path: "@model_path/weights/wv.bin",   data: wvBlob),
                (path: "@model_path/weights/wo.bin",   data: woBlob),
                (path: "@model_path/weights/rms2.bin", data: rms2Blob),
                (path: "@model_path/weights/w1.bin",   data: w1Blob),
                (path: "@model_path/weights/w3.bin",   data: w3Blob),
                (path: "@model_path/weights/w2.bin",   data: w2Blob),
            ],
            inputSizes: generator.inputByteSizes,
            outputSizes: generator.outputByteSizes
        )
    }

    @inline(__always)
    private static func buildBlob(from buffer: borrowing TensorBuffer, rows: Int, cols: Int) -> Data {
        buffer.withUnsafeBufferPointer { ptr in
            WeightBlob.build(from: ptr, rows: rows, cols: cols)
        }
    }

    @inline(__always)
    public static func resolvedLaneSpatialForCurrentProcess() -> Int {
        let envSpatial = ProcessInfo.processInfo
            .environment["ESPRESSO_DECODE_LANE_SPATIAL"]
            .flatMap(Int.init)
        return max(defaultLaneSpatial, envSpatial ?? defaultLaneSpatial)
    }
}
