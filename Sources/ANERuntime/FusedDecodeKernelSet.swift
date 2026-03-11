import Foundation
import ANETypes
import MILGenerator

/// Owns a single fused decode kernel per transformer layer (attention + FFN combined).
///
/// The fused kernel replaces the separate `decodeAttnQKV` + `decodeFFN` kernels from
/// `DecodeKernelSet` with one MIL program, eliminating the inter-kernel IOSurface round-trip.
///
/// Inputs:
/// - `x`:         `[1, dim, 1, laneSpatial]`
/// - `kCache`:    `[1, dim, 1, kernelMaxSeq]`
/// - `vCache`:    `[1, dim, 1, kernelMaxSeq]`
/// - `maskCache`: `[1, dim, 1, kernelMaxSeq]`
///
/// Outputs:
/// - `x_out`: `[1, dim, 1, laneSpatial]`
/// - `k_t`:   `[1, dim, 1, laneSpatial]`
/// - `v_t`:   `[1, dim, 1, laneSpatial]`
public struct FusedDecodeKernelSet: ~Copyable {

    // MARK: - Public constants

    /// Default spatial lane width used when the environment override is absent.
    public static let defaultLaneSpatial = 32

    // MARK: - Internal types

    internal enum KernelKind: String, CaseIterable {
        case fusedDecodeLayer
    }

    internal struct CompileSpec {
        internal let kind: KernelKind
        internal let milText: String
        internal let weights: [(path: String, data: Data)]
        internal let inputSizes: [Int]
        internal let outputSizes: [Int]
    }

    // MARK: - Public stored properties

    /// The single compiled fused layer kernel (attention + FFN in one MIL program).
    public let fusedLayer: ANEKernel

    /// Logical decode context size requested by the caller.
    public let maxSeq: Int

    /// Max sequence width baked into the decode-attention portion of the kernel.
    /// Always equal to `laneSpatial` (the compile-time lane tile width).
    public let kernelMaxSeq: Int

    /// Spatial tile width used for this kernel set.
    public let laneSpatial: Int

    // MARK: - Private init

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

    // MARK: - Public init

    /// Compile the fused decode-layer kernel for a given set of layer weights.
    ///
    /// - Parameters:
    ///   - weights: The layer weights to bake into the kernel as BLOBFILE constants.
    ///   - maxSeq:  Logical KV-cache sequence length for this model (must be a positive
    ///              multiple of `resolvedLaneSpatialForCurrentProcess()`).
    ///
    /// - Throws: `ANEError.invalidArguments` if `maxSeq` fails the alignment checks,
    ///           or any `ANEError` from the underlying `ANEKernel` compile.
    public init(weights: borrowing LayerWeights, maxSeq: Int = ModelConfig.seqLen) throws(ANEError) {
        guard maxSeq > 0 else {
            throw .invalidArguments("fused decode maxSeq must be > 0")
        }
        let laneSpatial = Self.resolvedLaneSpatialForCurrentProcess()
        guard maxSeq >= laneSpatial else {
            throw .invalidArguments(
                "fused decode maxSeq (\(maxSeq)) must be >= laneSpatial (\(laneSpatial))"
            )
        }
        guard maxSeq % laneSpatial == 0 else {
            throw .invalidArguments(
                "fused decode maxSeq (\(maxSeq)) must be a multiple of laneSpatial (\(laneSpatial))"
            )
        }
        let kernelMaxSeq = laneSpatial
        let compiled = try Self.compileFusedDecodeLayer(
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

    // MARK: - Internal compile-spec helpers

    /// Returns an array of `CompileSpec` values suitable for parallel pre-compilation.
    ///
    /// Callers that want to batch-compile kernel sets across multiple layers can call this
    /// without constructing an `ANEKernel` directly, then feed each spec into `ANEKernel.init`.
    internal static func compileSpecs(weights: borrowing LayerWeights, maxSeq: Int) -> [CompileSpec] {
        let laneSpatial = resolvedLaneSpatialForCurrentProcess()
        precondition(maxSeq > 0)
        precondition(maxSeq >= laneSpatial)
        return [
            makeFusedDecodeLayerSpec(weights: weights, maxSeq: laneSpatial, laneSpatial: laneSpatial),
        ]
    }

    // MARK: - Private compilation

    private static func compileFusedDecodeLayer(
        weights: borrowing LayerWeights,
        maxSeq: Int,
        laneSpatial: Int
    ) throws(ANEError) -> ANEKernel {
        let spec = makeFusedDecodeLayerSpec(weights: weights, maxSeq: maxSeq, laneSpatial: laneSpatial)
        return try ANEKernel(
            milText: spec.milText,
            weights: spec.weights,
            inputSizes: spec.inputSizes,
            outputSizes: spec.outputSizes
        )
    }

    private static func makeFusedDecodeLayerSpec(
        weights: borrowing LayerWeights,
        maxSeq: Int,
        laneSpatial: Int
    ) -> CompileSpec {
        let dim = ModelConfig.dim
        let hidden = ModelConfig.hidden

        let generator = FusedDecodeLayerGenerator(maxSeq: maxSeq, laneSpatial: laneSpatial)

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
            kind: .fusedDecodeLayer,
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

    // MARK: - Weight blob builder

    /// Converts a `TensorBuffer` of Float32 weights into an ANE BLOBFILE-compatible `Data` value.
    ///
    /// The returned `Data` includes the 128-byte ANE blob header followed by fp16-converted
    /// weight values, matching the format expected by MIL BLOBFILE constants.
    @inline(__always)
    private static func buildBlob(from buffer: borrowing TensorBuffer, rows: Int, cols: Int) -> Data {
        buffer.withUnsafeBufferPointer { ptr in
            WeightBlob.build(from: ptr, rows: rows, cols: cols)
        }
    }

    // MARK: - Lane spatial resolution

    /// Returns the lane spatial width to use for the current process.
    ///
    /// Reads `ESPRESSO_DECODE_LANE_SPATIAL` from the environment to allow per-process override.
    /// Falls back to `defaultLaneSpatial` (32) when the variable is absent or non-numeric.
    /// The resolved value is always at least `defaultLaneSpatial`.
    @inline(__always)
    public static func resolvedLaneSpatialForCurrentProcess() -> Int {
        let envSpatial = ProcessInfo.processInfo
            .environment["ESPRESSO_DECODE_LANE_SPATIAL"]
            .flatMap(Int.init)
        return max(defaultLaneSpatial, envSpatial ?? defaultLaneSpatial)
    }
}
