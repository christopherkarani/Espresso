import Foundation
import ANETypes
import MILGenerator

public struct HybridOutputProjectionWeights: Sendable {
    public let cacheKey: String
    public let rowMajorWeights: [Float]
    public let rowMajorBias: [Float]

    public init(cacheKey: String, rowMajorWeights: [Float], rowMajorBias: [Float]) {
        self.cacheKey = cacheKey
        self.rowMajorWeights = rowMajorWeights
        self.rowMajorBias = rowMajorBias
    }
}

/// Owns the split decode path for ANE QKV-only + Metal attention + ANE FFN.
public struct HybridDecodeKernelSet: ~Copyable {
    internal enum KernelKind: String, CaseIterable {
        case decodeQKVOnly
        case decodeProjectionFFN
        case decodeProjection
        case decodeFFN
    }

    internal struct CompileSpec {
        internal let kind: KernelKind
        internal let milText: String
        internal let weights: [(path: String, data: Data)]
        internal let inputSizes: [Int]
        internal let outputSizes: [Int]
    }

    internal struct CompiledPostAttention: ~Copyable {
        internal let decodeProjection: ANEKernel
        internal let decodeFFN: ANEKernel
        internal let usesFusedPostAttention: Bool
    }

    public let decodeQKVOnly: ANEKernel
    public let decodeProjection: ANEKernel
    public let decodeFFN: ANEKernel
    public let usesFusedPostAttention: Bool
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

    private static func copyTransposedWeights(from buffer: borrowing TensorBuffer, dim: Int) -> [Float] {
        buffer.withUnsafeBufferPointer { ptr in
            var result = [Float](repeating: 0, count: dim * dim)
            for row in 0..<dim {
                for column in 0..<dim {
                    result[row * dim + column] = ptr[column * dim + row]
                }
            }
            return result
        }
    }

    private init(
        decodeQKVOnly: consuming ANEKernel,
        decodeProjection: consuming ANEKernel,
        decodeFFN: consuming ANEKernel,
        usesFusedPostAttention: Bool,
        outputProjection: HybridOutputProjectionWeights,
        maxSeq: Int,
        laneSpatial: Int
    ) {
        self.decodeQKVOnly = decodeQKVOnly
        self.decodeProjection = decodeProjection
        self.decodeFFN = decodeFFN
        self.usesFusedPostAttention = usesFusedPostAttention
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
        let compiledPostAttention = try Self.compilePostAttention(weights: weights, laneSpatial: laneSpatial)
        let outputProjection = HybridOutputProjectionWeights(
            cacheKey: UUID().uuidString,
            rowMajorWeights: Self.copyRowMajorWeights(from: weights.Wo),
            rowMajorBias: weights.architecture == .gpt2
                ? Self.copyRowMajorWeights(from: weights.bo)
                : [Float](repeating: 0, count: weights.dim)
        )
        self.init(
            decodeQKVOnly: compiledQKV,
            decodeProjection: compiledPostAttention.decodeProjection,
            decodeFFN: compiledPostAttention.decodeFFN,
            usesFusedPostAttention: compiledPostAttention.usesFusedPostAttention,
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
            makeDecodeProjectionSpec(weights: weights, laneSpatial: laneSpatial),
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
        let dim = weights.dim
        let kvDim = weights.kvDim
        let generator = DecodeQKVOnlyGenerator(
            dim: dim,
            kvDim: kvDim,
            laneSpatial: laneSpatial,
            architecture: weights.architecture
        )

        let rms1Blob = buildBlob(from: weights.rmsAtt, rows: 1, cols: dim)
        let wqBlob = buildBlob(from: weights.Wq, rows: dim, cols: dim)
        let wkBlob = buildBlob(from: weights.Wk, rows: kvDim, cols: dim)
        let wvBlob = buildBlob(from: weights.Wv, rows: kvDim, cols: dim)
        let rms1BetaBlob = buildBlob(from: weights.attentionNormBeta, rows: 1, cols: dim)
        let bqBlob = buildBlob(from: weights.bq, rows: 1, cols: dim)
        let bkBlob = buildBlob(from: weights.bk, rows: 1, cols: kvDim)
        let bvBlob = buildBlob(from: weights.bv, rows: 1, cols: kvDim)

        let qkvWeights: [(path: String, data: Data)]
        switch weights.architecture {
        case .rmsNormSwiGLU:
            qkvWeights = [
                (path: "@model_path/weights/rms1.bin", data: rms1Blob),
                (path: "@model_path/weights/wq.bin", data: wqBlob),
                (path: "@model_path/weights/wk.bin", data: wkBlob),
                (path: "@model_path/weights/wv.bin", data: wvBlob),
            ]
        case .gpt2:
            qkvWeights = [
                (path: "@model_path/weights/rms1.bin", data: rms1Blob),
                (path: "@model_path/weights/rms1_beta.bin", data: rms1BetaBlob),
                (path: "@model_path/weights/wq.bin", data: wqBlob),
                (path: "@model_path/weights/wk.bin", data: wkBlob),
                (path: "@model_path/weights/wv.bin", data: wvBlob),
                (path: "@model_path/weights/bq.bin", data: bqBlob),
                (path: "@model_path/weights/bk.bin", data: bkBlob),
                (path: "@model_path/weights/bv.bin", data: bvBlob),
            ]
        }

        return CompileSpec(
            kind: .decodeQKVOnly,
            milText: generator.milText,
            weights: qkvWeights,
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

    private static func compileDecodeProjectionFFN(
        weights: borrowing LayerWeights,
        laneSpatial: Int
    ) throws(ANEError) -> ANEKernel {
        let spec = makeDecodeProjectionFFNSpec(weights: weights, laneSpatial: laneSpatial)
        return try ANEKernel(
            milText: spec.milText,
            weights: spec.weights,
            inputSizes: spec.inputSizes,
            outputSizes: spec.outputSizes
        )
    }

    private static func compilePostAttention(
        weights: borrowing LayerWeights,
        laneSpatial: Int
    ) throws(ANEError) -> CompiledPostAttention {
        let fusionEnabled = ProcessInfo.processInfo.environment["ESPRESSO_ENABLE_HYBRID_FUSED_POST_ATTENTION"] == "1"
        if fusionEnabled {
            do {
                return CompiledPostAttention(
                    decodeProjection: try compileDecodeProjectionFFN(weights: weights, laneSpatial: laneSpatial),
                    decodeFFN: try compileDecodeFFN(weights: weights, laneSpatial: laneSpatial),
                    usesFusedPostAttention: true
                )
            } catch {
            }
        }

        return CompiledPostAttention(
            decodeProjection: try compileDecodeProjection(weights: weights, laneSpatial: laneSpatial),
            decodeFFN: try compileDecodeFFN(weights: weights, laneSpatial: laneSpatial),
            usesFusedPostAttention: false
        )
    }

    private static func compileDecodeProjection(
        weights: borrowing LayerWeights,
        laneSpatial: Int
    ) throws(ANEError) -> ANEKernel {
        let spec = makeDecodeProjectionSpec(weights: weights, laneSpatial: laneSpatial)
        return try ANEKernel(
            milText: spec.milText,
            weights: spec.weights,
            inputSizes: spec.inputSizes,
            outputSizes: spec.outputSizes
        )
    }

    private static func makeDecodeProjectionSpec(
        weights: borrowing LayerWeights,
        laneSpatial: Int
    ) -> CompileSpec {
        let dim = weights.dim
        let generator = DecodeProjectionGenerator(
            dim: dim,
            laneSpatial: laneSpatial,
            architecture: weights.architecture
        )
        let woBlob = buildBlob(from: weights.Wo, rows: dim, cols: dim)
        let boBlob = buildBlob(from: weights.bo, rows: 1, cols: dim)

        let projectionWeights: [(path: String, data: Data)]
        switch weights.architecture {
        case .rmsNormSwiGLU:
            projectionWeights = [
                (path: "@model_path/weights/wo.bin", data: woBlob),
            ]
        case .gpt2:
            projectionWeights = [
                (path: "@model_path/weights/wo.bin", data: woBlob),
                (path: "@model_path/weights/bo.bin", data: boBlob),
            ]
        }

        return CompileSpec(
            kind: .decodeProjection,
            milText: generator.milText,
            weights: projectionWeights,
            inputSizes: generator.inputByteSizes,
            outputSizes: generator.outputByteSizes
        )
    }

    private static func makeDecodeProjectionFFNSpec(
        weights: borrowing LayerWeights,
        laneSpatial: Int
    ) -> CompileSpec {
        let dim = weights.dim
        let hidden = weights.hiddenDim
        let generator = DecodeProjectionFFNGenerator(
            dim: dim,
            hiddenDim: hidden,
            laneSpatial: laneSpatial,
            architecture: weights.architecture
        )
        let woBlob = buildBlob(from: weights.Wo, rows: dim, cols: dim)
        let boBlob = buildBlob(from: weights.bo, rows: 1, cols: dim)
        let rms2Blob = buildBlob(from: weights.rmsFfn, rows: 1, cols: dim)
        let w1Blob = buildBlob(from: weights.W1, rows: hidden, cols: dim)
        let w3Blob = buildBlob(from: weights.W3, rows: hidden, cols: dim)
        let w2Blob = buildBlob(from: weights.W2, rows: dim, cols: hidden)
        let rms2BetaBlob = buildBlob(from: weights.ffnNormBeta, rows: 1, cols: dim)
        let b1Blob = buildBlob(from: weights.b1, rows: 1, cols: hidden)
        let b2Blob = buildBlob(from: weights.b2, rows: 1, cols: dim)

        let fusedWeights: [(path: String, data: Data)]
        switch weights.architecture {
        case .rmsNormSwiGLU:
            fusedWeights = [
                (path: "@model_path/weights/wo.bin", data: woBlob),
                (path: "@model_path/weights/rms2.bin", data: rms2Blob),
                (path: "@model_path/weights/w1.bin", data: w1Blob),
                (path: "@model_path/weights/w3.bin", data: w3Blob),
                (path: "@model_path/weights/w2.bin", data: w2Blob),
            ]
        case .gpt2:
            fusedWeights = [
                (path: "@model_path/weights/wo.bin", data: woBlob),
                (path: "@model_path/weights/bo.bin", data: boBlob),
                (path: "@model_path/weights/rms2.bin", data: rms2Blob),
                (path: "@model_path/weights/rms2_beta.bin", data: rms2BetaBlob),
                (path: "@model_path/weights/w1.bin", data: w1Blob),
                (path: "@model_path/weights/w2.bin", data: w2Blob),
                (path: "@model_path/weights/b1.bin", data: b1Blob),
                (path: "@model_path/weights/b2.bin", data: b2Blob),
            ]
        }

        return CompileSpec(
            kind: .decodeProjectionFFN,
            milText: generator.milText,
            weights: fusedWeights,
            inputSizes: generator.inputByteSizes,
            outputSizes: generator.outputByteSizes
        )
    }

    private static func makeDecodeFFNSpec(
        weights: borrowing LayerWeights,
        laneSpatial: Int
    ) -> CompileSpec {
        let dim = weights.dim
        let hidden = weights.hiddenDim
        let generator = DecodeFFNGenerator(
            dim: dim,
            hiddenDim: hidden,
            laneSpatial: laneSpatial,
            architecture: weights.architecture
        )

        let rms2Blob = buildBlob(from: weights.rmsFfn, rows: 1, cols: dim)
        let w1Blob = buildBlob(from: weights.W1, rows: hidden, cols: dim)
        let w3Blob = buildBlob(from: weights.W3, rows: hidden, cols: dim)
        let w2Blob = buildBlob(from: weights.W2, rows: dim, cols: hidden)
        let rms2BetaBlob = buildBlob(from: weights.ffnNormBeta, rows: 1, cols: dim)
        let b1Blob = buildBlob(from: weights.b1, rows: 1, cols: hidden)
        let b2Blob = buildBlob(from: weights.b2, rows: 1, cols: dim)

        let ffnWeights: [(path: String, data: Data)]
        switch weights.architecture {
        case .rmsNormSwiGLU:
            ffnWeights = [
                (path: "@model_path/weights/rms2.bin", data: rms2Blob),
                (path: "@model_path/weights/w1.bin", data: w1Blob),
                (path: "@model_path/weights/w3.bin", data: w3Blob),
                (path: "@model_path/weights/w2.bin", data: w2Blob),
            ]
        case .gpt2:
            ffnWeights = [
                (path: "@model_path/weights/rms2.bin", data: rms2Blob),
                (path: "@model_path/weights/rms2_beta.bin", data: rms2BetaBlob),
                (path: "@model_path/weights/w1.bin", data: w1Blob),
                (path: "@model_path/weights/w2.bin", data: w2Blob),
                (path: "@model_path/weights/b1.bin", data: b1Blob),
                (path: "@model_path/weights/b2.bin", data: b2Blob),
            ]
        }

        return CompileSpec(
            kind: .decodeFFN,
            milText: generator.milText,
            weights: ffnWeights,
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
