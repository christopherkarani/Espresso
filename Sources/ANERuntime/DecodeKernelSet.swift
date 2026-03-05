import Foundation
import ANETypes
import MILGenerator

/// Owns decode-time kernels for one transformer layer.
///
/// Kernel A: decode attention + QKV emit (`x2_t`, `k_t`, `v_t`)  
/// Kernel B: decode FFN (`x3_t`)
public struct DecodeKernelSet: ~Copyable {
    public static let defaultLaneSpatial = 32

    internal enum KernelKind: String, CaseIterable {
        case decodeAttnQKV
        case decodeFFN
    }

    internal struct CompileSpec {
        internal let kind: KernelKind
        internal let milText: String
        internal let weights: [(path: String, data: Data)]
        internal let inputSizes: [Int]
        internal let outputSizes: [Int]
    }

    public let decodeAttnQKV: ANEKernel
    public let decodeFFN: ANEKernel

    public let maxSeq: Int
    public let laneSpatial: Int

    @inline(__always)
    private static func buildBlob(from buffer: borrowing TensorBuffer, rows: Int, cols: Int) -> Data {
        buffer.withUnsafeBufferPointer { ptr in
            WeightBlob.build(from: ptr, rows: rows, cols: cols)
        }
    }

    private init(decodeAttnQKV: consuming ANEKernel, decodeFFN: consuming ANEKernel, maxSeq: Int, laneSpatial: Int) {
        self.decodeAttnQKV = decodeAttnQKV
        self.decodeFFN = decodeFFN
        self.maxSeq = maxSeq
        self.laneSpatial = laneSpatial
    }

    public init(weights: borrowing LayerWeights, maxSeq: Int = ModelConfig.seqLen) throws(ANEError) {
        guard maxSeq > 0 else {
            throw .invalidArguments("decode maxSeq must be > 0")
        }
        let laneSpatial = Self.resolvedLaneSpatialForCurrentProcess()
        guard maxSeq == laneSpatial else {
            throw .invalidArguments("decode maxSeq (\(maxSeq)) must equal laneSpatial (\(laneSpatial)) on current ANE path")
        }
        let compiledAttn = try Self.compileDecodeAttnQKV(weights: weights, maxSeq: maxSeq, laneSpatial: laneSpatial)
        let compiledFFN = try Self.compileDecodeFFN(weights: weights, laneSpatial: laneSpatial)
        self.init(decodeAttnQKV: compiledAttn, decodeFFN: compiledFFN, maxSeq: maxSeq, laneSpatial: laneSpatial)
    }

    internal static func compileSpecs(weights: borrowing LayerWeights, maxSeq: Int) -> [CompileSpec] {
        let laneSpatial = resolvedLaneSpatialForCurrentProcess()
        precondition(maxSeq == laneSpatial)
        return [
            makeDecodeAttnQKVSpec(weights: weights, maxSeq: maxSeq, laneSpatial: laneSpatial),
            makeDecodeFFNSpec(weights: weights, laneSpatial: laneSpatial),
        ]
    }

    private static func compileDecodeAttnQKV(weights: borrowing LayerWeights, maxSeq: Int, laneSpatial: Int) throws(ANEError) -> ANEKernel {
        let spec = makeDecodeAttnQKVSpec(weights: weights, maxSeq: maxSeq, laneSpatial: laneSpatial)
        return try ANEKernel(
            milText: spec.milText,
            weights: spec.weights,
            inputSizes: spec.inputSizes,
            outputSizes: spec.outputSizes
        )
    }

    private static func makeDecodeAttnQKVSpec(weights: borrowing LayerWeights, maxSeq: Int, laneSpatial: Int) -> CompileSpec {
        let dim = ModelConfig.dim
        let generator = DecodeAttentionQKVGenerator(maxSeq: maxSeq, laneSpatial: laneSpatial)

        let rms1Blob = buildBlob(from: weights.rmsAtt, rows: 1, cols: dim)
        let wqBlob = buildBlob(from: weights.Wq, rows: dim, cols: dim)
        let wkBlob = buildBlob(from: weights.Wk, rows: dim, cols: dim)
        let wvBlob = buildBlob(from: weights.Wv, rows: dim, cols: dim)
        let woBlob = buildBlob(from: weights.Wo, rows: dim, cols: dim)

        return CompileSpec(
            kind: .decodeAttnQKV,
            milText: generator.milText,
            weights: [
                (path: "@model_path/weights/rms1.bin", data: rms1Blob),
                (path: "@model_path/weights/wq.bin", data: wqBlob),
                (path: "@model_path/weights/wk.bin", data: wkBlob),
                (path: "@model_path/weights/wv.bin", data: wvBlob),
                (path: "@model_path/weights/wo.bin", data: woBlob),
            ],
            inputSizes: generator.inputByteSizes,
            // Output surfaces:
            // 0: x2_t  [1, dim, 1, laneSpatial]
            // 1: k_t   [1, dim, 1, laneSpatial]
            // 2: v_t   [1, dim, 1, laneSpatial]
            outputSizes: generator.outputByteSizes
        )
    }

    private static func compileDecodeFFN(weights: borrowing LayerWeights, laneSpatial: Int) throws(ANEError) -> ANEKernel {
        let spec = makeDecodeFFNSpec(weights: weights, laneSpatial: laneSpatial)
        return try ANEKernel(
            milText: spec.milText,
            weights: spec.weights,
            inputSizes: spec.inputSizes,
            outputSizes: spec.outputSizes
        )
    }

    private static func makeDecodeFFNSpec(weights: borrowing LayerWeights, laneSpatial: Int) -> CompileSpec {
        let dim = ModelConfig.dim
        let hidden = ModelConfig.hidden
        let generator = DecodeFFNGenerator(laneSpatial: laneSpatial)

        let rms2Blob = buildBlob(from: weights.rmsFfn, rows: 1, cols: dim)
        let w1Blob = buildBlob(from: weights.W1, rows: hidden, cols: dim)
        let w3Blob = buildBlob(from: weights.W3, rows: hidden, cols: dim)
        let w2Blob = buildBlob(from: weights.W2, rows: dim, cols: hidden)

        return CompileSpec(
            kind: .decodeFFN,
            milText: generator.milText,
            weights: [
                (path: "@model_path/weights/rms2.bin", data: rms2Blob),
                (path: "@model_path/weights/w1.bin", data: w1Blob),
                (path: "@model_path/weights/w3.bin", data: w3Blob),
                (path: "@model_path/weights/w2.bin", data: w2Blob),
            ],
            inputSizes: [generator.inputBytes],
            outputSizes: generator.outputByteSizes
        )
    }

    @inline(__always)
    public static func resolvedLaneSpatialForCurrentProcess() -> Int {
        let envSpatial = ProcessInfo.processInfo.environment["ESPRESSO_DECODE_LANE_SPATIAL"].flatMap(Int.init)
        return max(defaultLaneSpatial, envSpatial ?? defaultLaneSpatial)
    }
}
