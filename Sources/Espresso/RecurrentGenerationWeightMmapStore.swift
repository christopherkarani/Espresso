import ANETypes
import Darwin
import Foundation

/// Errors thrown by `RecurrentGenerationWeightMmapStore`.
public enum RecurrentGenerationWeightMmapStoreError: Error, Equatable, Sendable {
    /// The flat-weight file could not be mmap'd or was too small.
    case mmapFailed(TensorBufferMmapError)
    /// A tensor could not be written to the flat-weight file.
    case writeFailed(TensorBufferWriteError)
}

/// A zero-copy mmap-backed loader for `RecurrentGenerationWeights`.
///
/// ## File Format
///
/// The store writes a flat, headerless sequence of `Float32` values (little-endian, host
/// byte order) in the following fixed order:
///
/// ```
/// rmsFinal            [dim]
/// embedding           [vocab * dim]
/// classifier          [vocab * dim]   (written as zeros when sharedClassifier=true)
/// layer[0].rms        [dim]
/// layer[0].Wx         [wqSize]
/// layer[0].Ws         [wqSize]
/// layer[0].Wd         [wqSize]
/// layer[0].Wo         [woSize]
/// layer[1].rms        [dim]
/// ...
/// layer[N-1].Wo       [woSize]
/// ```
///
/// Total element count:
/// `dim + 2*(vocab*dim) + layerCount*(dim + 3*wqSize + woSize)`
///
/// The classifier slot is always present so the layout is fully determined by `layerCount`
/// and the compile-time `ModelConfig` constants regardless of `sharedClassifier`.
public enum RecurrentGenerationWeightMmapStore {

    // MARK: - Layout

    /// Returns the total number of `Float` elements in the flat layout for `layerCount` layers.
    public static func totalFloatCount(layerCount: Int) -> Int {
        let headRegion = ModelConfig.dim                       // rmsFinal
            + ModelConfig.vocab * ModelConfig.dim              // embedding
            + ModelConfig.vocab * ModelConfig.dim              // classifier (placeholder)
        let perLayer = ModelConfig.dim                         // rms
            + ModelConfig.wqSize                               // Wx
            + ModelConfig.wqSize                               // Ws
            + ModelConfig.wqSize                               // Wd
            + ModelConfig.woSize                               // Wo
        return headRegion + layerCount * perLayer
    }

    // MARK: - Serialize (write)

    /// Writes all weight tensors to a flat binary file at `path`.
    ///
    /// When `weights.sharedClassifier` is `true`, the classifier slot is written as
    /// all-zeros (same element count as the embedding) to keep the layout invariant.
    public static func write(
        _ weights: borrowing RecurrentGenerationWeights,
        to path: String
    ) throws(RecurrentGenerationWeightMmapStoreError) {
        guard let file = Darwin.fopen(path, "wb") else {
            throw .writeFailed(.fileOpenFailed(path: path, errno: Darwin.errno))
        }
        defer { Darwin.fclose(file) }

        try append(weights.rmsFinal, name: "rmsFinal", to: file, path: path)
        try append(weights.embedding, name: "embedding", to: file, path: path)

        if weights.sharedClassifier {
            let placeholder = TensorBuffer(count: weights.vocabSize * ModelConfig.dim, zeroed: true)
            try append(placeholder, name: "classifier(placeholder)", to: file, path: path)
        } else {
            try append(weights.classifier, name: "classifier", to: file, path: path)
        }

        for idx in 0..<weights.layers.count {
            try append(weights.layers[idx].rms, name: "layer[\(idx)].rms", to: file, path: path)
            try append(weights.layers[idx].Wx,  name: "layer[\(idx)].Wx",  to: file, path: path)
            try append(weights.layers[idx].Ws,  name: "layer[\(idx)].Ws",  to: file, path: path)
            try append(weights.layers[idx].Wd,  name: "layer[\(idx)].Wd",  to: file, path: path)
            try append(weights.layers[idx].Wo,  name: "layer[\(idx)].Wo",  to: file, path: path)
        }
    }

    // MARK: - Load (mmap)

    /// Maps a flat weight file into memory and returns `RecurrentGenerationWeights` whose
    /// tensor fields are non-owning slice views into the single mmap'd region.
    ///
    /// No `Float` data is copied. The OS page cache drives warm/cold transitions
    /// transparently. The mmap lifetime is tied to the returned weights object.
    ///
    /// - Parameters:
    ///   - path:            Path to the file produced by `write(_:to:)`.
    ///   - layerCount:      Number of recurrent layers the file was written with.
    ///   - sharedClassifier: When `true`, `RecurrentGenerationWeights.classifier` is set to
    ///                       an empty buffer and the embedding matrix is reused. The classifier
    ///                       slot in the file is still present but ignored.
    public static func load(
        from path: String,
        layerCount: Int,
        sharedClassifier: Bool = true
    ) throws(RecurrentGenerationWeightMmapStoreError) -> RecurrentGenerationWeights {
        precondition(layerCount > 0)

        let totalFloats = totalFloatCount(layerCount: layerCount)

        // Single mmap for the entire file.
        let backing = try mmapBacking(path: path, count: totalFloats)

        // Carve non-owning slice views into the flat layout.
        var cursor = 0

        let rmsFinal = backing.nonOwningSlice(offset: cursor, count: ModelConfig.dim)
        cursor += ModelConfig.dim

        let embeddingCount = ModelConfig.vocab * ModelConfig.dim
        let embedding = backing.nonOwningSlice(offset: cursor, count: embeddingCount)
        cursor += embeddingCount

        let classifierCount = ModelConfig.vocab * ModelConfig.dim
        let classifier = backing.nonOwningSlice(offset: cursor, count: classifierCount)
        cursor += classifierCount

        let perLayer = ModelConfig.dim + ModelConfig.wqSize * 3 + ModelConfig.woSize
        let layers = LayerStorage<RWKVStyleRecurrentWeights>(count: layerCount) { idx in
            let base = cursor + idx * perLayer
            return RWKVStyleRecurrentWeights(
                rms: backing.nonOwningSlice(offset: base,
                                            count: ModelConfig.dim),
                Wx:  backing.nonOwningSlice(offset: base + ModelConfig.dim,
                                            count: ModelConfig.wqSize),
                Ws:  backing.nonOwningSlice(offset: base + ModelConfig.dim + ModelConfig.wqSize,
                                            count: ModelConfig.wqSize),
                Wd:  backing.nonOwningSlice(offset: base + ModelConfig.dim + 2 * ModelConfig.wqSize,
                                            count: ModelConfig.wqSize),
                Wo:  backing.nonOwningSlice(offset: base + ModelConfig.dim + 3 * ModelConfig.wqSize,
                                            count: ModelConfig.woSize)
            )
        }

        // When sharedClassifier=true the classifier data is not used; pass an empty buffer.
        let classifierForInit: TensorBuffer = sharedClassifier
            ? TensorBuffer(count: 0, zeroed: true)
            : classifier

        return RecurrentGenerationWeights(
            layers: layers,
            rmsFinal: rmsFinal,
            embedding: embedding,
            classifier: classifierForInit,
            sharedClassifier: sharedClassifier,
            vocabSize: ModelConfig.vocab,
            mmapBacking: backing
        )
    }

    // MARK: - Private helpers

    /// Wraps TensorBuffer mmap init to convert its error type.
    /// Avoids do/catch on ~Copyable types which triggers compiler bugs.
    private static func mmapBacking(
        path: String,
        count: Int
    ) throws(RecurrentGenerationWeightMmapStoreError) -> TensorBuffer {
        do throws(TensorBufferMmapError) {
            return try TensorBuffer(mmapFrom: path, offset: 0, count: count)
        } catch {
            throw .mmapFailed(error)
        }
    }

    private static func append(
        _ buffer: borrowing TensorBuffer,
        name: String,
        to file: UnsafeMutablePointer<Darwin.FILE>,
        path: String
    ) throws(RecurrentGenerationWeightMmapStoreError) {
        let written = buffer.withUnsafePointer { ptr in
            Darwin.fwrite(ptr, MemoryLayout<Float>.stride, buffer.count, file)
        }
        guard written == buffer.count else {
            throw .writeFailed(.writeFailed(
                path: path,
                written: written * MemoryLayout<Float>.stride,
                expected: buffer.count * MemoryLayout<Float>.stride,
                errno: Darwin.errno
            ))
        }
    }
}
