import Foundation
import IOSurface
import ANETypes
import MILGenerator

/// Owns the 5 weight-bearing ANE kernels for a single transformer layer.
/// Recompiled each batch when weights change. ~Copyable: deinit frees all 5 kernels.
/// Different lifecycle from StaticKernel (sdpaBwd2).
public struct LayerKernelSet: ~Copyable {
    internal enum KernelKind: String, CaseIterable {
        case fwdAttn
        case fwdFFN
        case ffnBwd
        case sdpaBwd1
        case qkvBwd
    }

    internal struct CompileSpec {
        internal let kind: KernelKind
        internal let milText: String
        internal let weights: [(path: String, data: Data)]
        internal let inputBytes: Int
        internal let outputBytes: Int

        internal var weightPaths: [String] {
            weights.map(\.path)
        }
    }

    public let fwdAttn: ANEKernel
    public let fwdFFN: ANEKernel
    public let ffnBwd: ANEKernel
    public let sdpaBwd1: ANEKernel
    public let qkvBwd: ANEKernel

    @inline(__always)
    private static func buildBlob(from buffer: borrowing TensorBuffer, rows: Int, cols: Int) -> Data {
        buffer.withUnsafeBufferPointer { ptr in
            WeightBlob.build(from: ptr, rows: rows, cols: cols)
        }
    }

    @inline(__always)
    private static func buildTransposedBlob(from buffer: borrowing TensorBuffer, rows: Int, cols: Int) -> Data {
        buffer.withUnsafeBufferPointer { ptr in
            WeightBlob.buildTransposed(from: ptr, rows: rows, cols: cols)
        }
    }

    @inline(__always)
    private static func compile(from spec: CompileSpec) throws(ANEError) -> ANEKernel {
        try ANEKernel(
            milText: spec.milText,
            weights: spec.weights,
            inputBytes: spec.inputBytes,
            outputBytes: spec.outputBytes
        )
    }

    private init(
        fwdAttn: consuming ANEKernel,
        fwdFFN: consuming ANEKernel,
        ffnBwd: consuming ANEKernel,
        sdpaBwd1: consuming ANEKernel,
        qkvBwd: consuming ANEKernel
    ) {
        self.fwdAttn = fwdAttn
        self.fwdFFN = fwdFFN
        self.ffnBwd = ffnBwd
        self.sdpaBwd1 = sdpaBwd1
        self.qkvBwd = qkvBwd
    }

    /// Compile all 5 weight-bearing kernels for one layer from its weights.
    /// Uses `borrowing` to avoid copying the ~324 MiB LayerWeights.
    public init(weights: borrowing LayerWeights) throws(ANEError) {
        let compiledFwdAttn = try Self.compileFwdAttn(weights: weights)
        let compiledFwdFFN = try Self.compileFwdFFN(weights: weights)
        let compiledFFNBwd = try Self.compileFFNBwd(weights: weights)
        let compiledSDPABwd1 = try Self.compileSDPABwd1(weights: weights)
        let compiledQKVBwd = try Self.compileQKVBwd(weights: weights)

        self.init(
            fwdAttn: compiledFwdAttn,
            fwdFFN: compiledFwdFFN,
            ffnBwd: compiledFFNBwd,
            sdpaBwd1: compiledSDPABwd1,
            qkvBwd: compiledQKVBwd
        )
    }

    internal static func compileSpecs(weights: borrowing LayerWeights) -> [CompileSpec] {
        [
            makeFwdAttnSpec(weights: weights),
            makeFwdFFNSpec(weights: weights),
            makeFFNBwdSpec(weights: weights),
            makeSDPABwd1Spec(weights: weights),
            makeQKVBwdSpec(weights: weights),
        ]
    }

    private static func compileFwdAttn(weights: borrowing LayerWeights) throws(ANEError) -> ANEKernel {
        let spec = makeFwdAttnSpec(weights: weights)
        return try compile(from: spec)
    }

    private static func makeFwdAttnSpec(weights: borrowing LayerWeights) -> CompileSpec {
        let dim = ModelConfig.dim
        let generator = SDPAForwardGenerator()

        let rms1Blob = buildBlob(from: weights.rmsAtt, rows: 1, cols: dim)
        let wqBlob = buildBlob(from: weights.Wq, rows: dim, cols: dim)
        let wkBlob = buildBlob(from: weights.Wk, rows: dim, cols: dim)
        let wvBlob = buildBlob(from: weights.Wv, rows: dim, cols: dim)
        let woBlob = buildBlob(from: weights.Wo, rows: dim, cols: dim)
        let maskBlob = CausalMask.blob(seqLen: ModelConfig.seqLen)

        return CompileSpec(
            kind: .fwdAttn,
            milText: generator.milText,
            weights: [
                (path: "@model_path/weights/rms1.bin", data: rms1Blob),
                (path: "@model_path/weights/wq.bin", data: wqBlob),
                (path: "@model_path/weights/wk.bin", data: wkBlob),
                (path: "@model_path/weights/wv.bin", data: wvBlob),
                (path: "@model_path/weights/wo.bin", data: woBlob),
                (path: "@model_path/weights/mask.bin", data: maskBlob),
            ],
            inputBytes: generator.inputBytes,
            outputBytes: generator.outputBytes
        )
    }

    private static func compileFwdFFN(weights: borrowing LayerWeights) throws(ANEError) -> ANEKernel {
        let spec = makeFwdFFNSpec(weights: weights)
        return try compile(from: spec)
    }

    private static func makeFwdFFNSpec(weights: borrowing LayerWeights) -> CompileSpec {
        let dim = ModelConfig.dim
        let hidden = ModelConfig.hidden
        let generator = FFNForwardGenerator()

        let rms2Blob = buildBlob(from: weights.rmsFfn, rows: 1, cols: dim)
        let w1Blob = buildBlob(from: weights.W1, rows: hidden, cols: dim)
        let w3Blob = buildBlob(from: weights.W3, rows: hidden, cols: dim)
        let w2Blob = buildBlob(from: weights.W2, rows: dim, cols: hidden)

        return CompileSpec(
            kind: .fwdFFN,
            milText: generator.milText,
            weights: [
                (path: "@model_path/weights/rms2.bin", data: rms2Blob),
                (path: "@model_path/weights/w1.bin", data: w1Blob),
                (path: "@model_path/weights/w3.bin", data: w3Blob),
                (path: "@model_path/weights/w2.bin", data: w2Blob),
            ],
            inputBytes: generator.inputBytes,
            outputBytes: generator.outputBytes
        )
    }

    private static func compileFFNBwd(weights: borrowing LayerWeights) throws(ANEError) -> ANEKernel {
        let spec = makeFFNBwdSpec(weights: weights)
        return try compile(from: spec)
    }

    private static func makeFFNBwdSpec(weights: borrowing LayerWeights) -> CompileSpec {
        let dim = ModelConfig.dim
        let hidden = ModelConfig.hidden
        let generator = FFNBackwardGenerator()

        let w2tBlob = buildTransposedBlob(from: weights.W2, rows: dim, cols: hidden)
        let w1tBlob = buildTransposedBlob(from: weights.W1, rows: hidden, cols: dim)
        let w3tBlob = buildTransposedBlob(from: weights.W3, rows: hidden, cols: dim)

        return CompileSpec(
            kind: .ffnBwd,
            milText: generator.milText,
            weights: [
                (path: "@model_path/weights/w2t.bin", data: w2tBlob),
                (path: "@model_path/weights/w1t.bin", data: w1tBlob),
                (path: "@model_path/weights/w3t.bin", data: w3tBlob),
            ],
            inputBytes: generator.inputBytes,
            outputBytes: generator.outputBytes
        )
    }

    private static func compileSDPABwd1(weights: borrowing LayerWeights) throws(ANEError) -> ANEKernel {
        let spec = makeSDPABwd1Spec(weights: weights)
        return try compile(from: spec)
    }

    private static func makeSDPABwd1Spec(weights: borrowing LayerWeights) -> CompileSpec {
        let dim = ModelConfig.dim
        let generator = SDPABackward1Generator()

        let maskBlob = CausalMask.blob(seqLen: ModelConfig.seqLen)
        let wotBlob = buildTransposedBlob(from: weights.Wo, rows: dim, cols: dim)

        return CompileSpec(
            kind: .sdpaBwd1,
            milText: generator.milText,
            weights: [
                (path: "@model_path/weights/mask.bin", data: maskBlob),
                (path: "@model_path/weights/wot.bin", data: wotBlob),
            ],
            inputBytes: generator.inputBytes,
            outputBytes: generator.outputBytes
        )
    }

    private static func compileQKVBwd(weights: borrowing LayerWeights) throws(ANEError) -> ANEKernel {
        let spec = makeQKVBwdSpec(weights: weights)
        return try compile(from: spec)
    }

    private static func makeQKVBwdSpec(weights: borrowing LayerWeights) -> CompileSpec {
        let dim = ModelConfig.dim
        let generator = QKVBackwardGenerator()

        let wqtBlob = buildTransposedBlob(from: weights.Wq, rows: dim, cols: dim)
        let wktBlob = buildTransposedBlob(from: weights.Wk, rows: dim, cols: dim)
        let wvtBlob = buildTransposedBlob(from: weights.Wv, rows: dim, cols: dim)

        return CompileSpec(
            kind: .qkvBwd,
            milText: generator.milText,
            weights: [
                (path: "@model_path/weights/wqt.bin", data: wqtBlob),
                (path: "@model_path/weights/wkt.bin", data: wktBlob),
                (path: "@model_path/weights/wvt.bin", data: wvtBlob),
            ],
            inputBytes: generator.inputBytes,
            outputBytes: generator.outputBytes
        )
    }
}
