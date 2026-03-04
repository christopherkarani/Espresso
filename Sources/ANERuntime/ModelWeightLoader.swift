import Foundation
import Darwin
import ANETypes

/// Errors specific to model weight loading.
public enum ModelLoadError: Error, Sendable, Equatable {
    /// File could not be opened.
    case fileNotFound(String)
    /// Header dimensions do not match ModelConfig.
    case configMismatch(expected: String, got: String)
    /// File is truncated — fewer bytes than expected.
    case truncatedFile(expectedBytes: Int, actualBytes: Int)
}

/// Result of loading pretrained weights.
public struct PretrainedWeights: ~Copyable {
    public let layers: LayerStorage<LayerWeights>
    public let rmsFinal: TensorBuffer
    public let embed: TensorBuffer
    /// Classifier weights when vocab_size < 0 (unshared classifier); empty when shared.
    public let classifier: TensorBuffer
    /// True when vocab_size > 0 (classifier shares embed weights).
    public let sharedClassifier: Bool
}

/// Loads pretrained weights from the llama2.c binary format.
public enum ModelWeightLoader {
    private static let headerFieldCount = 7

    internal enum LayerParameterKind: String, CaseIterable {
        case rmsAtt = "rms_att"
        case wq
        case wk
        case wv
        case wo
        case rmsFfn = "rms_ffn"
        case w1
        case w2
        case w3

        @inline(__always)
        internal var floatsPerLayer: Int {
            switch self {
            case .rmsAtt, .rmsFfn:
                return ModelConfig.dim
            case .wq, .wk, .wv, .wo:
                return ModelConfig.dim * ModelConfig.dim
            case .w1, .w3:
                return ModelConfig.hidden * ModelConfig.dim
            case .w2:
                return ModelConfig.dim * ModelConfig.hidden
            }
        }
    }

    internal struct PayloadSegment: Equatable {
        internal let name: String
        internal let floatCount: Int

        internal var byteCount: Int {
            floatCount * MemoryLayout<Float>.stride
        }
    }

    /// Payload segment layout for llama2.c weights, in strict on-disk order.
    internal static func payloadLayout(vocabSize: Int32) -> [PayloadSegment] {
        let vocabMagnitude = Int(vocabSize.magnitude)
        var segments: [PayloadSegment] = [
            PayloadSegment(name: "embed", floatCount: vocabMagnitude * ModelConfig.dim),
        ]

        for parameter in LayerParameterKind.allCases {
            segments.append(
                PayloadSegment(
                    name: "\(parameter.rawValue)[all]",
                    floatCount: parameter.floatsPerLayer * ModelConfig.nLayers
                )
            )
        }

        segments.append(PayloadSegment(name: "rms_final", floatCount: ModelConfig.dim))
        if vocabSize < 0 {
            segments.append(PayloadSegment(name: "wcls", floatCount: vocabMagnitude * ModelConfig.dim))
        }
        return segments
    }

    /// Load weights from a `.bin` file at the given path.
    /// Validates header dimensions against ModelConfig before reading payload.
    public static func load(from path: String) throws(ModelLoadError) -> PretrainedWeights {
        guard let file = fopen(path, "rb") else {
            throw .fileNotFound(path)
        }
        defer { fclose(file) }

        let header = try parseHeader(from: file)
        guard Int(header.dim) == ModelConfig.dim,
              Int(header.hiddenDim) == ModelConfig.hidden,
              Int(header.nLayers) == ModelConfig.nLayers,
              Int(header.nHeads) == ModelConfig.heads,
              Int(header.nKvHeads) == ModelConfig.heads,
              Int(header.vocabSize.magnitude) == ModelConfig.vocab,
              Int(header.seqLen) == ModelConfig.seqLen else {
            let expected = "dim=\(ModelConfig.dim) hidden=\(ModelConfig.hidden) layers=\(ModelConfig.nLayers) heads=\(ModelConfig.heads) kvHeads=\(ModelConfig.heads) vocab=±\(ModelConfig.vocab) seq=\(ModelConfig.seqLen)"
            let got = "dim=\(header.dim) hidden=\(header.hiddenDim) layers=\(header.nLayers) heads=\(header.nHeads) kvHeads=\(header.nKvHeads) vocab=\(header.vocabSize) seq=\(header.seqLen)"
            throw .configMismatch(expected: expected, got: got)
        }

        let vocabMagnitude = Int(header.vocabSize.magnitude)
        let sharedClassifier = header.vocabSize > 0

        let embed = TensorBuffer(count: vocabMagnitude * ModelConfig.dim, zeroed: false)
        let rmsFinal = TensorBuffer(count: ModelConfig.dim, zeroed: false)
        let classifier = TensorBuffer(
            count: sharedClassifier ? 0 : vocabMagnitude * ModelConfig.dim,
            zeroed: false
        )
        let layers = LayerStorage<LayerWeights>(count: ModelConfig.nLayers) { _ in
            LayerWeights()
        }

        try readInto(embed, from: file)

        for parameter in LayerParameterKind.allCases {
            for layer in 0..<ModelConfig.nLayers {
                try readLayerParameter(parameter, layer: layer, from: layers, file: file)
            }
        }

        try readInto(rmsFinal, from: file)
        if !sharedClassifier {
            try readInto(classifier, from: file)
        }

        return PretrainedWeights(
            layers: layers,
            rmsFinal: rmsFinal,
            embed: embed,
            classifier: classifier,
            sharedClassifier: sharedClassifier
        )
    }

    internal static func parseHeader(
        from file: UnsafeMutablePointer<FILE>
    ) throws(ModelLoadError) -> (
        dim: Int32,
        hiddenDim: Int32,
        nLayers: Int32,
        nHeads: Int32,
        nKvHeads: Int32,
        vocabSize: Int32,
        seqLen: Int32
    ) {
        var fields = [Int32](repeating: 0, count: headerFieldCount)
        let fieldCount = fields.count
        let readCount = fields.withUnsafeMutableBufferPointer { ptr in
            fread(ptr.baseAddress, MemoryLayout<Int32>.stride, fieldCount, file)
        }

        guard readCount == fieldCount else {
            throw .truncatedFile(
                expectedBytes: fieldCount * MemoryLayout<Int32>.stride,
                actualBytes: readCount * MemoryLayout<Int32>.stride
            )
        }

        return (
            dim: Int32(littleEndian: fields[0]),
            hiddenDim: Int32(littleEndian: fields[1]),
            nLayers: Int32(littleEndian: fields[2]),
            nHeads: Int32(littleEndian: fields[3]),
            nKvHeads: Int32(littleEndian: fields[4]),
            vocabSize: Int32(littleEndian: fields[5]),
            seqLen: Int32(littleEndian: fields[6])
        )
    }

    @inline(__always)
    private static func readLayerParameter(
        _ parameter: LayerParameterKind,
        layer: Int,
        from layers: borrowing LayerStorage<LayerWeights>,
        file: UnsafeMutablePointer<FILE>
    ) throws(ModelLoadError) {
        switch parameter {
        case .rmsAtt:
            try readInto(layers[layer].rmsAtt, from: file)
        case .wq:
            try readInto(layers[layer].Wq, from: file)
        case .wk:
            try readInto(layers[layer].Wk, from: file)
        case .wv:
            try readInto(layers[layer].Wv, from: file)
        case .wo:
            try readInto(layers[layer].Wo, from: file)
        case .rmsFfn:
            try readInto(layers[layer].rmsFfn, from: file)
        case .w1:
            try readInto(layers[layer].W1, from: file)
        case .w2:
            try readInto(layers[layer].W2, from: file)
        case .w3:
            try readInto(layers[layer].W3, from: file)
        }
    }

    @inline(__always)
    private static func readInto(
        _ buffer: borrowing TensorBuffer,
        from file: UnsafeMutablePointer<FILE>
    ) throws(ModelLoadError) {
        let readCount = buffer.withUnsafeMutablePointer { ptr in
            fread(ptr, MemoryLayout<Float>.stride, buffer.count, file)
        }

        guard readCount == buffer.count else {
            throw .truncatedFile(
                expectedBytes: buffer.count * MemoryLayout<Float>.stride,
                actualBytes: readCount * MemoryLayout<Float>.stride
            )
        }
    }
}
