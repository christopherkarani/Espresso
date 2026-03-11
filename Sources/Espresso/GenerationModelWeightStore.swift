import Foundation
import Darwin
import ANETypes

public enum GenerationModelWeightStoreError: Error, Equatable, Sendable {
    case fileOpenFailed(String)
    case shortWrite(expectedBytes: Int, actualBytes: Int)
    case truncatedFile(expectedBytes: Int, actualBytes: Int)
    case configMismatch(expected: String, got: String)
}

public enum GenerationModelWeightStore {
    public static func save(
        layers: borrowing LayerStorage<LayerWeights>,
        rmsFinal: borrowing TensorBuffer,
        embed: borrowing TensorBuffer,
        classifier: borrowing TensorBuffer,
        sharedClassifier: Bool,
        vocabSize: Int = ModelConfig.vocab,
        to path: String
    ) throws(GenerationModelWeightStoreError) {
        guard let file = fopen(path, "wb") else {
            throw .fileOpenFailed(path)
        }
        defer { fclose(file) }

        try writeHeader(
            dim: ModelConfig.dim,
            hidden: ModelConfig.hidden,
            nLayers: layers.count,
            nHeads: ModelConfig.heads,
            nKvHeads: ModelConfig.heads,
            vocabSize: sharedClassifier ? vocabSize : -vocabSize,
            seqLen: ModelConfig.seqLen,
            to: file
        )

        try writeTensor(embed, to: file)
        for idx in 0..<layers.count { try writeTensor(layers[idx].rmsAtt, to: file) }
        for idx in 0..<layers.count { try writeTensor(layers[idx].Wq, to: file) }
        for idx in 0..<layers.count { try writeTensor(layers[idx].Wk, to: file) }
        for idx in 0..<layers.count { try writeTensor(layers[idx].Wv, to: file) }
        for idx in 0..<layers.count { try writeTensor(layers[idx].Wo, to: file) }
        for idx in 0..<layers.count { try writeTensor(layers[idx].rmsFfn, to: file) }
        for idx in 0..<layers.count { try writeTensor(layers[idx].W1, to: file) }
        for idx in 0..<layers.count { try writeTensor(layers[idx].W2, to: file) }
        for idx in 0..<layers.count { try writeTensor(layers[idx].W3, to: file) }
        try writeTensor(rmsFinal, to: file)
        if !sharedClassifier {
            try writeTensor(classifier, to: file)
        }
    }

    public static func save(
        _ weights: borrowing GenerationWeights,
        to path: String
    ) throws(GenerationModelWeightStoreError) {
        try save(
            layers: weights.layers,
            rmsFinal: weights.rmsFinal,
            embed: weights.embedding,
            classifier: weights.classifier,
            sharedClassifier: weights.sharedClassifier,
            vocabSize: weights.vocabSize,
            to: path
        )
    }

    public static func load(
        path: String
    ) throws(GenerationModelWeightStoreError) -> GenerationWeights {
        guard let file = fopen(path, "rb") else {
            throw .fileOpenFailed(path)
        }
        defer { fclose(file) }

        let header = try readHeader(from: file)
        guard header.dim == ModelConfig.dim,
              header.hidden == ModelConfig.hidden,
              header.nHeads == ModelConfig.heads,
              header.nKvHeads == ModelConfig.heads,
              header.seqLen == ModelConfig.seqLen else {
            let expected = "dim=\(ModelConfig.dim) hidden=\(ModelConfig.hidden) heads=\(ModelConfig.heads) kvHeads=\(ModelConfig.heads) seq=\(ModelConfig.seqLen)"
            let got = "dim=\(header.dim) hidden=\(header.hidden) heads=\(header.nHeads) kvHeads=\(header.nKvHeads) seq=\(header.seqLen)"
            throw .configMismatch(expected: expected, got: got)
        }

        let sharedClassifier = header.vocabSize > 0
        let vocabSize = abs(header.vocabSize)
        let embed = TensorBuffer(count: vocabSize * ModelConfig.dim, zeroed: false)
        try readTensor(embed, from: file)
        let layers = LayerStorage<LayerWeights>(count: header.nLayers) { _ in LayerWeights() }
        for idx in 0..<header.nLayers { try readTensor(layers[idx].rmsAtt, from: file) }
        for idx in 0..<header.nLayers { try readTensor(layers[idx].Wq, from: file) }
        for idx in 0..<header.nLayers { try readTensor(layers[idx].Wk, from: file) }
        for idx in 0..<header.nLayers { try readTensor(layers[idx].Wv, from: file) }
        for idx in 0..<header.nLayers { try readTensor(layers[idx].Wo, from: file) }
        for idx in 0..<header.nLayers { try readTensor(layers[idx].rmsFfn, from: file) }
        for idx in 0..<header.nLayers { try readTensor(layers[idx].W1, from: file) }
        for idx in 0..<header.nLayers { try readTensor(layers[idx].W2, from: file) }
        for idx in 0..<header.nLayers { try readTensor(layers[idx].W3, from: file) }
        let rmsFinal = TensorBuffer(count: ModelConfig.dim, zeroed: false)
        try readTensor(rmsFinal, from: file)
        let classifier = TensorBuffer(count: sharedClassifier ? 0 : vocabSize * ModelConfig.dim, zeroed: false)
        if !sharedClassifier {
            try readTensor(classifier, from: file)
        }

        return GenerationWeights(
            layers: layers,
            rmsFinal: rmsFinal,
            embedding: embed,
            classifier: classifier,
            sharedClassifier: sharedClassifier,
            vocabSize: vocabSize
        )
    }

    internal struct TinyLayerWeights: ~Copyable {
        internal let Wq, Wk, Wv, Wo: TensorBuffer
        internal let W1, W2, W3: TensorBuffer
        internal let rmsAtt, rmsFfn: TensorBuffer

        internal init(dim: Int, hidden: Int) {
            let wqSize = dim * dim
            let woSize = dim * dim
            let w1Size = hidden * dim
            let w2Size = dim * hidden
            let w3Size = hidden * dim
            self.Wq = TensorBuffer(count: wqSize, zeroed: false)
            self.Wk = TensorBuffer(count: wqSize, zeroed: false)
            self.Wv = TensorBuffer(count: wqSize, zeroed: false)
            self.Wo = TensorBuffer(count: woSize, zeroed: false)
            self.W1 = TensorBuffer(count: w1Size, zeroed: false)
            self.W2 = TensorBuffer(count: w2Size, zeroed: false)
            self.W3 = TensorBuffer(count: w3Size, zeroed: false)
            self.rmsAtt = TensorBuffer(count: dim, zeroed: false)
            self.rmsFfn = TensorBuffer(count: dim, zeroed: false)
        }
    }

    internal struct TinyWeights: ~Copyable {
        internal let layers: LayerStorage<TinyLayerWeights>
        internal let rmsFinal: TensorBuffer
        internal let embed: TensorBuffer
        internal let classifier: TensorBuffer
    }

    internal static func _saveTiny(
        path: String,
        dim: Int,
        hidden: Int,
        nLayers: Int,
        nHeads: Int,
        seqLen: Int,
        vocab: Int,
        sharedClassifier: Bool,
        layers: borrowing LayerStorage<TinyLayerWeights>,
        rmsFinal: borrowing TensorBuffer,
        embed: borrowing TensorBuffer,
        classifier: borrowing TensorBuffer
    ) throws(GenerationModelWeightStoreError) {
        guard let file = fopen(path, "wb") else {
            throw .fileOpenFailed(path)
        }
        defer { fclose(file) }

        try writeHeader(
            dim: dim,
            hidden: hidden,
            nLayers: nLayers,
            nHeads: nHeads,
            nKvHeads: nHeads,
            vocabSize: sharedClassifier ? vocab : -vocab,
            seqLen: seqLen,
            to: file
        )

        try writeTensor(embed, to: file)
        for idx in 0..<nLayers { try writeTensor(layers[idx].rmsAtt, to: file) }
        for idx in 0..<nLayers { try writeTensor(layers[idx].Wq, to: file) }
        for idx in 0..<nLayers { try writeTensor(layers[idx].Wk, to: file) }
        for idx in 0..<nLayers { try writeTensor(layers[idx].Wv, to: file) }
        for idx in 0..<nLayers { try writeTensor(layers[idx].Wo, to: file) }
        for idx in 0..<nLayers { try writeTensor(layers[idx].rmsFfn, to: file) }
        for idx in 0..<nLayers { try writeTensor(layers[idx].W1, to: file) }
        for idx in 0..<nLayers { try writeTensor(layers[idx].W2, to: file) }
        for idx in 0..<nLayers { try writeTensor(layers[idx].W3, to: file) }
        try writeTensor(rmsFinal, to: file)
        if !sharedClassifier {
            try writeTensor(classifier, to: file)
        }
    }

    internal static func _loadTiny(
        path: String,
        dim: Int,
        hidden: Int,
        nLayers: Int,
        nHeads: Int,
        seqLen: Int,
        vocab: Int,
        sharedClassifier: Bool
    ) throws(GenerationModelWeightStoreError) -> TinyWeights {
        guard let file = fopen(path, "rb") else {
            throw .fileOpenFailed(path)
        }
        defer { fclose(file) }

        let header = try readHeader(from: file)
        let expectedVocab = sharedClassifier ? vocab : -vocab
        guard header.dim == dim,
              header.hidden == hidden,
              header.nLayers == nLayers,
              header.nHeads == nHeads,
              header.nKvHeads == nHeads,
              header.vocabSize == expectedVocab,
              header.seqLen == seqLen else {
            let expected = "dim=\(dim) hidden=\(hidden) layers=\(nLayers) heads=\(nHeads) kvHeads=\(nHeads) vocab=\(expectedVocab) seq=\(seqLen)"
            let got = "dim=\(header.dim) hidden=\(header.hidden) layers=\(header.nLayers) heads=\(header.nHeads) kvHeads=\(header.nKvHeads) vocab=\(header.vocabSize) seq=\(header.seqLen)"
            throw .configMismatch(expected: expected, got: got)
        }

        let embed = TensorBuffer(count: vocab * dim, zeroed: false)
        try readTensor(embed, from: file)
        let layers = LayerStorage<TinyLayerWeights>(count: nLayers) { _ in
            TinyLayerWeights(dim: dim, hidden: hidden)
        }
        for idx in 0..<nLayers { try readTensor(layers[idx].rmsAtt, from: file) }
        for idx in 0..<nLayers { try readTensor(layers[idx].Wq, from: file) }
        for idx in 0..<nLayers { try readTensor(layers[idx].Wk, from: file) }
        for idx in 0..<nLayers { try readTensor(layers[idx].Wv, from: file) }
        for idx in 0..<nLayers { try readTensor(layers[idx].Wo, from: file) }
        for idx in 0..<nLayers { try readTensor(layers[idx].rmsFfn, from: file) }
        for idx in 0..<nLayers { try readTensor(layers[idx].W1, from: file) }
        for idx in 0..<nLayers { try readTensor(layers[idx].W2, from: file) }
        for idx in 0..<nLayers { try readTensor(layers[idx].W3, from: file) }
        let rmsFinal = TensorBuffer(count: dim, zeroed: false)
        try readTensor(rmsFinal, from: file)
        let classifier = TensorBuffer(count: sharedClassifier ? 0 : vocab * dim, zeroed: false)
        if !sharedClassifier {
            try readTensor(classifier, from: file)
        }

        return TinyWeights(
            layers: layers,
            rmsFinal: rmsFinal,
            embed: embed,
            classifier: classifier
        )
    }

    private static func writeHeader(
        dim: Int,
        hidden: Int,
        nLayers: Int,
        nHeads: Int,
        nKvHeads: Int,
        vocabSize: Int,
        seqLen: Int,
        to file: UnsafeMutablePointer<FILE>
    ) throws(GenerationModelWeightStoreError) {
        try writeInt32(dim, to: file)
        try writeInt32(hidden, to: file)
        try writeInt32(nLayers, to: file)
        try writeInt32(nHeads, to: file)
        try writeInt32(nKvHeads, to: file)
        try writeInt32(vocabSize, to: file)
        try writeInt32(seqLen, to: file)
    }

    private static func readHeader(
        from file: UnsafeMutablePointer<FILE>
    ) throws(GenerationModelWeightStoreError) -> (
        dim: Int,
        hidden: Int,
        nLayers: Int,
        nHeads: Int,
        nKvHeads: Int,
        vocabSize: Int,
        seqLen: Int
    ) {
        var fields = [Int32](repeating: 0, count: 7)
        let expectedBytes = fields.count * MemoryLayout<Int32>.stride
        let readCount = fields.withUnsafeMutableBufferPointer { ptr in
            fread(ptr.baseAddress, MemoryLayout<Int32>.stride, ptr.count, file)
        }
        guard readCount == fields.count else {
            throw .truncatedFile(
                expectedBytes: expectedBytes,
                actualBytes: readCount * MemoryLayout<Int32>.stride
            )
        }

        return (
            dim: Int(Int32(littleEndian: fields[0])),
            hidden: Int(Int32(littleEndian: fields[1])),
            nLayers: Int(Int32(littleEndian: fields[2])),
            nHeads: Int(Int32(littleEndian: fields[3])),
            nKvHeads: Int(Int32(littleEndian: fields[4])),
            vocabSize: Int(Int32(littleEndian: fields[5])),
            seqLen: Int(Int32(littleEndian: fields[6]))
        )
    }

    private static func writeInt32(
        _ value: Int,
        to file: UnsafeMutablePointer<FILE>
    ) throws(GenerationModelWeightStoreError) {
        var littleEndian = Int32(value).littleEndian
        let written = withUnsafePointer(to: &littleEndian) { ptr in
            fwrite(ptr, MemoryLayout<Int32>.stride, 1, file)
        }
        guard written == 1 else {
            throw .shortWrite(
                expectedBytes: MemoryLayout<Int32>.stride,
                actualBytes: written * MemoryLayout<Int32>.stride
            )
        }
    }

    private static func writeTensor(
        _ buffer: borrowing TensorBuffer,
        to file: UnsafeMutablePointer<FILE>
    ) throws(GenerationModelWeightStoreError) {
        let written = buffer.withUnsafePointer { ptr in
            fwrite(ptr, MemoryLayout<Float>.stride, buffer.count, file)
        }
        guard written == buffer.count else {
            throw .shortWrite(
                expectedBytes: buffer.count * MemoryLayout<Float>.stride,
                actualBytes: written * MemoryLayout<Float>.stride
            )
        }
    }

    private static func readTensor(
        _ buffer: borrowing TensorBuffer,
        from file: UnsafeMutablePointer<FILE>
    ) throws(GenerationModelWeightStoreError) {
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
