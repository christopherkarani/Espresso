import Foundation
import ANETypes

public enum RecurrentGenerationWeightStoreError: Error, Equatable, Sendable {
    case fileOpenFailed(String)
    case invalidMagic(UInt32)
    case unsupportedVersion(UInt32)
    case configMismatch(expected: String, got: String)
    case truncatedFile(expectedBytes: Int, actualBytes: Int)
    case shortWrite(expectedBytes: Int, actualBytes: Int)
}

public enum RecurrentGenerationWeightStore {
    private static let magic: UInt32 = 0x3147_5752 // "RGW1"
    private static let version: UInt32 = 1
    private static let headerFieldCount = 10

    public static func save(
        _ weights: borrowing RecurrentGenerationWeights,
        to path: String
    ) throws(RecurrentGenerationWeightStoreError) {
        guard let file = fopen(path, "wb") else {
            throw .fileOpenFailed(path)
        }
        defer { fclose(file) }

        try write(UInt32(littleEndian: magic), to: file)
        try write(UInt32(littleEndian: version), to: file)
        try write(Int32(ModelConfig.dim), to: file)
        try write(Int32(ModelConfig.wqSize), to: file)
        try write(Int32(ModelConfig.woSize), to: file)
        try write(Int32(weights.layers.count), to: file)
        try write(Int32(weights.vocabSize), to: file)
        try write(Int32(ModelConfig.seqLen), to: file)
        try write(Int32(weights.sharedClassifier ? 1 : 0), to: file)
        try write(Int32(0), to: file) // reserved

        try writeTensor(weights.rmsFinal, to: file)
        try writeTensor(weights.embedding, to: file)
        if !weights.sharedClassifier {
            try writeTensor(weights.classifier, to: file)
        }

        for idx in 0..<weights.layers.count {
            try writeTensor(weights.layers[idx].rms, to: file)
            try writeTensor(weights.layers[idx].Wx, to: file)
            try writeTensor(weights.layers[idx].Ws, to: file)
            try writeTensor(weights.layers[idx].Wd, to: file)
            try writeTensor(weights.layers[idx].Wo, to: file)
        }
    }

    public static func load(
        from path: String
    ) throws(RecurrentGenerationWeightStoreError) -> RecurrentGenerationWeights {
        guard let file = fopen(path, "rb") else {
            throw .fileOpenFailed(path)
        }
        defer { fclose(file) }

        let header = try parseHeader(from: file)
        try validate(header: header)

        let sharedClassifier = header.sharedClassifier != 0
        let rmsFinal = TensorBuffer(count: ModelConfig.dim, zeroed: false)
        try readTensor(rmsFinal, from: file)

        let embedding = TensorBuffer(count: header.vocabSize * ModelConfig.dim, zeroed: false)
        try readTensor(embedding, from: file)

        let classifier = TensorBuffer(
            count: sharedClassifier ? 0 : header.vocabSize * ModelConfig.dim,
            zeroed: false
        )
        if !sharedClassifier {
            try readTensor(classifier, from: file)
        }

        let layers = LayerStorage<RWKVStyleRecurrentWeights>(count: header.layerCount) { _ in
            RWKVStyleRecurrentWeights()
        }
        for idx in 0..<header.layerCount {
            try readTensor(layers[idx].rms, from: file)
            try readTensor(layers[idx].Wx, from: file)
            try readTensor(layers[idx].Ws, from: file)
            try readTensor(layers[idx].Wd, from: file)
            try readTensor(layers[idx].Wo, from: file)
        }

        return RecurrentGenerationWeights(
            layers: layers,
            rmsFinal: rmsFinal,
            embedding: embedding,
            classifier: classifier,
            sharedClassifier: sharedClassifier,
            vocabSize: header.vocabSize
        )
    }

    private static func parseHeader(
        from file: UnsafeMutablePointer<FILE>
    ) throws(RecurrentGenerationWeightStoreError) -> (
        magic: UInt32,
        version: UInt32,
        dim: Int32,
        wqSize: Int32,
        woSize: Int32,
        layerCount: Int,
        vocabSize: Int,
        seqLen: Int32,
        sharedClassifier: Int32
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
            magic: UInt32(bitPattern: Int32(littleEndian: fields[0])),
            version: UInt32(bitPattern: Int32(littleEndian: fields[1])),
            dim: Int32(littleEndian: fields[2]),
            wqSize: Int32(littleEndian: fields[3]),
            woSize: Int32(littleEndian: fields[4]),
            layerCount: Int(Int32(littleEndian: fields[5])),
            vocabSize: Int(Int32(littleEndian: fields[6])),
            seqLen: Int32(littleEndian: fields[7]),
            sharedClassifier: Int32(littleEndian: fields[8])
        )
    }

    private static func validate(
        header: (
            magic: UInt32,
            version: UInt32,
            dim: Int32,
            wqSize: Int32,
            woSize: Int32,
            layerCount: Int,
            vocabSize: Int,
            seqLen: Int32,
            sharedClassifier: Int32
        )
    ) throws(RecurrentGenerationWeightStoreError) {
        guard header.magic == magic else {
            throw .invalidMagic(header.magic)
        }
        guard header.version == version else {
            throw .unsupportedVersion(header.version)
        }
        guard header.layerCount > 0 else {
            let expected = expectedConfigDescription()
            let got = "layerCount=\(header.layerCount)"
            throw .configMismatch(expected: expected, got: got)
        }
        guard Int(header.dim) == ModelConfig.dim,
              Int(header.wqSize) == ModelConfig.wqSize,
              Int(header.woSize) == ModelConfig.woSize,
              header.vocabSize == ModelConfig.vocab,
              Int(header.seqLen) == ModelConfig.seqLen else {
            let expected = expectedConfigDescription()
            let got = "dim=\(header.dim) wqSize=\(header.wqSize) woSize=\(header.woSize) vocab=\(header.vocabSize) seq=\(header.seqLen)"
            throw .configMismatch(expected: expected, got: got)
        }
    }

    private static func expectedConfigDescription() -> String {
        "dim=\(ModelConfig.dim) wqSize=\(ModelConfig.wqSize) woSize=\(ModelConfig.woSize) vocab=\(ModelConfig.vocab) seq=\(ModelConfig.seqLen)"
    }

    private static func write(_ value: UInt32, to file: UnsafeMutablePointer<FILE>) throws(RecurrentGenerationWeightStoreError) {
        var littleEndian = value.littleEndian
        let written = withUnsafePointer(to: &littleEndian) { ptr in
            fwrite(ptr, MemoryLayout<UInt32>.stride, 1, file)
        }
        guard written == 1 else {
            throw .shortWrite(expectedBytes: MemoryLayout<UInt32>.stride, actualBytes: written * MemoryLayout<UInt32>.stride)
        }
    }

    private static func write(_ value: Int32, to file: UnsafeMutablePointer<FILE>) throws(RecurrentGenerationWeightStoreError) {
        var littleEndian = value.littleEndian
        let written = withUnsafePointer(to: &littleEndian) { ptr in
            fwrite(ptr, MemoryLayout<Int32>.stride, 1, file)
        }
        guard written == 1 else {
            throw .shortWrite(expectedBytes: MemoryLayout<Int32>.stride, actualBytes: written * MemoryLayout<Int32>.stride)
        }
    }

    private static func writeTensor(
        _ buffer: borrowing TensorBuffer,
        to file: UnsafeMutablePointer<FILE>
    ) throws(RecurrentGenerationWeightStoreError) {
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
    ) throws(RecurrentGenerationWeightStoreError) {
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
