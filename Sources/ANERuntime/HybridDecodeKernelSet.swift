import Foundation
import ANETypes
import MILGenerator

public struct HybridOutputProjectionWeights: Sendable {
    public let cacheKey: String
    public let rowMajorWeights: [Float]

    public init(cacheKey: String, rowMajorWeights: [Float]) {
        self.cacheKey = cacheKey
        self.rowMajorWeights = rowMajorWeights
    }
}

/// Owns the split decode path for ANE QKV-only + Metal attention + ANE FFN.
public struct HybridDecodeKernelSet: ~Copyable {
    internal enum KernelKind: String, CaseIterable {
        case decodeQKVOnly
        case decodeFFN
    }

    internal struct CompileSpec {
        internal let kind: KernelKind
        internal let milText: String
        internal let weights: [(path: String, data: Data)]
        internal let inputSizes: [Int]
        internal let outputSizes: [Int]
    }

    public let decodeQKVOnly: ANEKernel
    public let decodeFFN: ANEKernel
    public let outputProjection: HybridOutputProjectionWeights
    public let maxSeq: Int
    public let laneSpatial: Int

    @inline(__always)
    private static func buildBlob(from buffer: borrowing TensorBuffer, rows: Int, cols: Int) -> Data {
        buffer.withUnsafeBufferPointer { ptr in
            WeightBlob.build(from: ptr, rows: rows, cols: cols)
        }
    }

    private static func copyRowMajorWeights(from buffer: borrowing TensorBuffer) -> [Float] {
        buffer.withUnsafeBufferPointer { ptr in
            Array(ptr)
        }
    }

    private init(
        decodeQKVOnly: consuming ANEKernel,
        decodeFFN: consuming ANEKernel,
        outputProjection: HybridOutputProjectionWeights,
        maxSeq: Int,
        laneSpatial: Int
    ) {
        self.decodeQKVOnly = decodeQKVOnly
        self.decodeFFN = decodeFFN
        self.outputProjection = outputProjection
        self.maxSeq = maxSeq
        self.laneSpatial = laneSpatial
    }

    public init(weights: borrowing LayerWeights, maxSeq: Int = ModelConfig.seqLen) throws(ANEError) {
        guard maxSeq > 0 else {
            throw .invalidArguments("hybrid decode maxSeq must be > 0")
        }
        let laneSpatial = Self.resolvedLaneSpatialForCurrentProcess()
        let compiledQKV = try Self.compileDecodeQKVOnly(weights: weights, laneSpatial: laneSpatial)
        let compiledFFN = try Self.compileDecodeFFN(weights: weights, laneSpatial: laneSpatial)
        let outputProjection = HybridOutputProjectionWeights(
            cacheKey: UUID().uuidString,
            rowMajorWeights: Self.copyRowMajorWeights(from: weights.Wo)
        )
        self.init(
            decodeQKVOnly: compiledQKV,
            decodeFFN: compiledFFN,
            outputProjection: outputProjection,
            maxSeq: maxSeq,
            laneSpatial: laneSpatial
        )
    }

    internal static func compileSpecs(weights: borrowing LayerWeights, maxSeq: Int) -> [CompileSpec] {
        precondition(maxSeq > 0)
        let laneSpatial = resolvedLaneSpatialForCurrentProcess()
        return [
            makeDecodeQKVOnlySpec(weights: weights, laneSpatial: laneSpatial),
            makeDecodeFFNSpec(weights: weights, laneSpatial: laneSpatial),
        ]
    }

    private static func compileDecodeQKVOnly(
        weights: borrowing LayerWeights,
        laneSpatial: Int
    ) throws(ANEError) -> ANEKernel {
        let spec = makeDecodeQKVOnlySpec(weights: weights, laneSpatial: laneSpatial)
        return try ANEKernel(
            milText: spec.milText,
            weights: spec.weights,
            inputSizes: spec.inputSizes,
            outputSizes: spec.outputSizes
        )
    }

    private static func makeDecodeQKVOnlySpec(
        weights: borrowing LayerWeights,
        laneSpatial: Int
    ) -> CompileSpec {
        let dim = ModelConfig.dim
        let generator = DecodeQKVOnlyGenerator(laneSpatial: laneSpatial)

        let rms1Blob = buildBlob(from: weights.rmsAtt, rows: 1, cols: dim)
        let wqBlob = buildBlob(from: weights.Wq, rows: dim, cols: dim)
        let wkBlob = buildBlob(from: weights.Wk, rows: dim, cols: dim)
        let wvBlob = buildBlob(from: weights.Wv, rows: dim, cols: dim)

        return CompileSpec(
            kind: .decodeQKVOnly,
            milText: generator.milText,
            weights: [
                (path: "@model_path/weights/rms1.bin", data: rms1Blob),
                (path: "@model_path/weights/wq.bin", data: wqBlob),
                (path: "@model_path/weights/wk.bin", data: wkBlob),
                (path: "@model_path/weights/wv.bin", data: wvBlob),
            ],
            inputSizes: generator.inputByteSizes,
            outputSizes: generator.outputByteSizes
        )
    }

    private static func compileDecodeFFN(
        weights: borrowing LayerWeights,
        laneSpatial: Int
    ) throws(ANEError) -> ANEKernel {
        let spec = makeDecodeFFNSpec(weights: weights, laneSpatial: laneSpatial)
        return try ANEKernel(
            milText: spec.milText,
            weights: spec.weights,
            inputSizes: spec.inputSizes,
            outputSizes: spec.outputSizes
        )
    }

    private static func makeDecodeFFNSpec(
        weights: borrowing LayerWeights,
        laneSpatial: Int
    ) -> CompileSpec {
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
        return max(DecodeKernelSet.defaultLaneSpatial, envSpatial ?? DecodeKernelSet.defaultLaneSpatial)
    }
}
