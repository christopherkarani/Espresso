import Foundation
import ANETypes
import MILGenerator

/// Owns a single fused two-layer decode kernel.
///
/// The kernel collapses two transformer layers into one MIL program with a packed cache input.
public struct FusedTwoLayerDecodeKernelSet: ~Copyable {

    public static let defaultLaneSpatial = 32

    internal enum KernelKind: String, CaseIterable {
        case fusedTwoLayerDecode
    }

    internal struct CompileSpec {
        internal let kind: KernelKind
        internal let milText: String
        internal let weights: [(path: String, data: Data)]
        internal let inputSizes: [Int]
        internal let outputSizes: [Int]
    }

    public let fusedPair: ANEKernel
    public let maxSeq: Int
    public let kernelMaxSeq: Int
    public let laneSpatial: Int

    private init(
        fusedPair: consuming ANEKernel,
        logicalMaxSeq: Int,
        kernelMaxSeq: Int,
        laneSpatial: Int
    ) {
        self.fusedPair = fusedPair
        self.maxSeq = logicalMaxSeq
        self.kernelMaxSeq = kernelMaxSeq
        self.laneSpatial = laneSpatial
    }

    public init(
        layer0Weights: borrowing LayerWeights,
        layer1Weights: borrowing LayerWeights,
        maxSeq: Int = ModelConfig.seqLen
    ) throws(ANEError) {
        guard maxSeq > 0 else {
            throw .invalidArguments("fused two-layer decode maxSeq must be > 0")
        }
        let laneSpatial = Self.resolvedLaneSpatialForCurrentProcess()
        guard maxSeq >= laneSpatial else {
            throw .invalidArguments(
                "fused two-layer decode maxSeq (\(maxSeq)) must be >= laneSpatial (\(laneSpatial))"
            )
        }
        guard maxSeq % laneSpatial == 0 else {
            throw .invalidArguments(
                "fused two-layer decode maxSeq (\(maxSeq)) must be a multiple of laneSpatial (\(laneSpatial))"
            )
        }
        let kernelMaxSeq = laneSpatial
        let compiled = try Self.compileFusedTwoLayerDecode(
            layer0Weights: layer0Weights,
            layer1Weights: layer1Weights,
            maxSeq: kernelMaxSeq,
            laneSpatial: laneSpatial
        )
        self.init(
            fusedPair: compiled,
            logicalMaxSeq: maxSeq,
            kernelMaxSeq: kernelMaxSeq,
            laneSpatial: laneSpatial
        )
    }

    internal static func compileSpecs(
        layer0Weights: borrowing LayerWeights,
        layer1Weights: borrowing LayerWeights,
        maxSeq: Int
    ) -> [CompileSpec] {
        let laneSpatial = resolvedLaneSpatialForCurrentProcess()
        precondition(maxSeq > 0)
        precondition(maxSeq >= laneSpatial)
        return [
            makeFusedTwoLayerDecodeSpec(
                layer0Weights: layer0Weights,
                layer1Weights: layer1Weights,
                maxSeq: laneSpatial,
                laneSpatial: laneSpatial
            ),
        ]
    }

    private static func compileFusedTwoLayerDecode(
        layer0Weights: borrowing LayerWeights,
        layer1Weights: borrowing LayerWeights,
        maxSeq: Int,
        laneSpatial: Int
    ) throws(ANEError) -> ANEKernel {
        let spec = makeFusedTwoLayerDecodeSpec(
            layer0Weights: layer0Weights,
            layer1Weights: layer1Weights,
            maxSeq: maxSeq,
            laneSpatial: laneSpatial
        )
        return try ANEKernel(
            milText: spec.milText,
            weights: spec.weights,
            inputSizes: spec.inputSizes,
            outputSizes: spec.outputSizes
        )
    }

    private static func makeFusedTwoLayerDecodeSpec(
        layer0Weights: borrowing LayerWeights,
        layer1Weights: borrowing LayerWeights,
        maxSeq: Int,
        laneSpatial: Int
    ) -> CompileSpec {
        let dim = ModelConfig.dim
        let hidden = ModelConfig.hidden
        let generator = FusedTwoLayerDecodeGenerator(maxSeq: maxSeq, laneSpatial: laneSpatial)

        return CompileSpec(
            kind: .fusedTwoLayerDecode,
            milText: generator.milText,
            weights: [
                (path: "@model_path/weights/l0_rms1.bin", data: buildBlob(from: layer0Weights.rmsAtt, rows: 1, cols: dim)),
                (path: "@model_path/weights/l0_wq.bin", data: buildBlob(from: layer0Weights.Wq, rows: dim, cols: dim)),
                (path: "@model_path/weights/l0_wk.bin", data: buildBlob(from: layer0Weights.Wk, rows: dim, cols: dim)),
                (path: "@model_path/weights/l0_wv.bin", data: buildBlob(from: layer0Weights.Wv, rows: dim, cols: dim)),
                (path: "@model_path/weights/l0_wo.bin", data: buildBlob(from: layer0Weights.Wo, rows: dim, cols: dim)),
                (path: "@model_path/weights/l0_rms2.bin", data: buildBlob(from: layer0Weights.rmsFfn, rows: 1, cols: dim)),
                (path: "@model_path/weights/l0_w1.bin", data: buildBlob(from: layer0Weights.W1, rows: hidden, cols: dim)),
                (path: "@model_path/weights/l0_w3.bin", data: buildBlob(from: layer0Weights.W3, rows: hidden, cols: dim)),
                (path: "@model_path/weights/l0_w2.bin", data: buildBlob(from: layer0Weights.W2, rows: dim, cols: hidden)),
                (path: "@model_path/weights/l1_rms1.bin", data: buildBlob(from: layer1Weights.rmsAtt, rows: 1, cols: dim)),
                (path: "@model_path/weights/l1_wq.bin", data: buildBlob(from: layer1Weights.Wq, rows: dim, cols: dim)),
                (path: "@model_path/weights/l1_wk.bin", data: buildBlob(from: layer1Weights.Wk, rows: dim, cols: dim)),
                (path: "@model_path/weights/l1_wv.bin", data: buildBlob(from: layer1Weights.Wv, rows: dim, cols: dim)),
                (path: "@model_path/weights/l1_wo.bin", data: buildBlob(from: layer1Weights.Wo, rows: dim, cols: dim)),
                (path: "@model_path/weights/l1_rms2.bin", data: buildBlob(from: layer1Weights.rmsFfn, rows: 1, cols: dim)),
                (path: "@model_path/weights/l1_w1.bin", data: buildBlob(from: layer1Weights.W1, rows: hidden, cols: dim)),
                (path: "@model_path/weights/l1_w3.bin", data: buildBlob(from: layer1Weights.W3, rows: hidden, cols: dim)),
                (path: "@model_path/weights/l1_w2.bin", data: buildBlob(from: layer1Weights.W2, rows: dim, cols: hidden)),
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
