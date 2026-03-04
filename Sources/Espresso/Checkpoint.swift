import ANETypes
import Darwin
import Foundation

public struct CheckpointMeta: Sendable, Equatable {
    public var step: Int
    public var totalSteps: Int
    public var lr: Float
    public var loss: Float

    public var cumCompile: Double
    public var cumTrain: Double
    public var cumWall: Double

    public var cumSteps: Int
    public var cumBatches: Int
    public var adamT: Int

    public init() {
        self.step = 0
        self.totalSteps = 0
        self.lr = 0
        self.loss = 0
        self.cumCompile = 0
        self.cumTrain = 0
        self.cumWall = 0
        self.cumSteps = 0
        self.cumBatches = 0
        self.adamT = 0
    }
}

public enum CheckpointError: Error, Sendable, Equatable {
    case openFailed(path: String, errno: Int32)
    case ioWriteFailed(segment: String, expectedCount: Int, actualCount: Int)
    case ioReadFailed(segment: String, expectedCount: Int, actualCount: Int)
    case invalidMagic(Int32)
    case unsupportedVersion(Int32)
    case configMismatch(expected: String, got: String)
    case argumentOutOfRange(String)
}

public enum Checkpoint {
    private static let expectedMagic: Int32 = 0x424C5A54
    private static let expectedVersion: Int32 = 2

    public static func save(
        path: String,
        meta: CheckpointMeta,
        layers: borrowing LayerStorage<LayerWeights>,
        layerAdam: borrowing LayerStorage<LayerAdam>,
        rmsFinal: borrowing TensorBuffer,
        adamRmsFinal: borrowing AdamState,
        embed: borrowing TensorBuffer,
        adamEmbed: borrowing AdamState
    ) throws(CheckpointError) {
        var header = try makeHeader(
            dim: ModelConfig.dim,
            hidden: ModelConfig.hidden,
            nLayers: ModelConfig.nLayers,
            nHeads: ModelConfig.heads,
            seqLen: ModelConfig.seqLen,
            vocab: ModelConfig.vocab,
            meta: meta
        )

        guard let file = fopen(path, "wb") else {
            throw .openFailed(path: path, errno: errno)
        }
        defer { fclose(file) }

        try writeHeader(file, header: &header)
        try writeAllLayers(
            file,
            nLayers: ModelConfig.nLayers,
            layers: layers,
            layerAdam: layerAdam
        )
        try writeGlobals(
            file,
            rmsFinal: rmsFinal,
            adamRmsFinal: adamRmsFinal,
            embed: embed,
            adamEmbed: adamEmbed
        )
    }

    public static func load(
        path: String,
        intoLayers layers: borrowing LayerStorage<LayerWeights>,
        intoLayerAdam layerAdam: borrowing LayerStorage<LayerAdam>,
        intoRmsFinal rmsFinal: borrowing TensorBuffer,
        intoAdamRmsFinal adamRmsFinal: borrowing AdamState,
        intoEmbed embed: borrowing TensorBuffer,
        intoAdamEmbed adamEmbed: borrowing AdamState
    ) throws(CheckpointError) -> CheckpointMeta {
        guard let file = fopen(path, "rb") else {
            throw .openFailed(path: path, errno: errno)
        }
        defer { fclose(file) }

        let header = try readAndValidateHeader(file)
        try validateHeaderConfig(
            header,
            dim: ModelConfig.dim,
            hidden: ModelConfig.hidden,
            nLayers: ModelConfig.nLayers,
            nHeads: ModelConfig.heads,
            seqLen: ModelConfig.seqLen,
            vocab: ModelConfig.vocab
        )

        try readAllLayers(
            file,
            nLayers: ModelConfig.nLayers,
            layers: layers,
            layerAdam: layerAdam
        )
        try readGlobals(
            file,
            rmsFinal: rmsFinal,
            adamRmsFinal: adamRmsFinal,
            embed: embed,
            adamEmbed: adamEmbed
        )

        return meta(from: header)
    }

    // MARK: - Tiny (Test-only)

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

    internal struct TinyLayerAdam: ~Copyable {
        internal let Wq, Wk, Wv, Wo: AdamState
        internal let W1, W2, W3: AdamState
        internal let rmsAtt, rmsFfn: AdamState

        internal init(dim: Int, hidden: Int) {
            let wqSize = dim * dim
            let woSize = dim * dim
            let w1Size = hidden * dim
            let w2Size = dim * hidden
            let w3Size = hidden * dim
            self.Wq = AdamState(count: wqSize)
            self.Wk = AdamState(count: wqSize)
            self.Wv = AdamState(count: wqSize)
            self.Wo = AdamState(count: woSize)
            self.W1 = AdamState(count: w1Size)
            self.W2 = AdamState(count: w2Size)
            self.W3 = AdamState(count: w3Size)
            self.rmsAtt = AdamState(count: dim)
            self.rmsFfn = AdamState(count: dim)
        }
    }

    internal struct LayoutSegment: Equatable {
        internal let name: String
        internal let byteOffset: Int
        internal let byteCount: Int
    }

    internal static func _tinyLayout(dim: Int, hidden: Int, nLayers: Int, vocab: Int) -> [LayoutSegment] {
        precondition(dim >= 0 && hidden >= 0 && nLayers >= 0 && vocab >= 0)

        let wqSize = dim * dim
        let woSize = dim * dim
        let w1Size = hidden * dim
        let w2Size = dim * hidden
        let w3Size = hidden * dim

        func bytes(_ floatCount: Int) -> Int { floatCount * MemoryLayout<Float>.stride }

        var segments: [LayoutSegment] = []
        segments.reserveCapacity(1 + nLayers * 27 + 6)

        var off = 0
        segments.append(LayoutSegment(name: "header", byteOffset: off, byteCount: MemoryLayout<CheckpointHeader>.stride))
        off += MemoryLayout<CheckpointHeader>.stride

        for L in 0..<nLayers {
            func add(_ name: String, _ count: Int) {
                segments.append(LayoutSegment(name: name, byteOffset: off, byteCount: bytes(count)))
                off += bytes(count)
            }

            // Weights (9 segments)
            add("L\(L).Wq", wqSize)
            add("L\(L).Wk", wqSize)
            add("L\(L).Wv", wqSize)
            add("L\(L).Wo", woSize)
            add("L\(L).W1", w1Size)
            add("L\(L).W2", w2Size)
            add("L\(L).W3", w3Size)
            add("L\(L).rmsAtt", dim)
            add("L\(L).rmsFfn", dim)

            // Adam m/v pairs (18 segments)
            add("L\(L).Wq.m", wqSize); add("L\(L).Wq.v", wqSize)
            add("L\(L).Wk.m", wqSize); add("L\(L).Wk.v", wqSize)
            add("L\(L).Wv.m", wqSize); add("L\(L).Wv.v", wqSize)
            add("L\(L).Wo.m", woSize); add("L\(L).Wo.v", woSize)
            add("L\(L).W1.m", w1Size); add("L\(L).W1.v", w1Size)
            add("L\(L).W2.m", w2Size); add("L\(L).W2.v", w2Size)
            add("L\(L).W3.m", w3Size); add("L\(L).W3.v", w3Size)
            add("L\(L).rmsAtt.m", dim); add("L\(L).rmsAtt.v", dim)
            add("L\(L).rmsFfn.m", dim); add("L\(L).rmsFfn.v", dim)
        }

        // Globals
        segments.append(LayoutSegment(name: "rmsFinal", byteOffset: off, byteCount: bytes(dim))); off += bytes(dim)
        segments.append(LayoutSegment(name: "adamRmsFinal.m", byteOffset: off, byteCount: bytes(dim))); off += bytes(dim)
        segments.append(LayoutSegment(name: "adamRmsFinal.v", byteOffset: off, byteCount: bytes(dim))); off += bytes(dim)
        segments.append(LayoutSegment(name: "embed", byteOffset: off, byteCount: bytes(vocab * dim))); off += bytes(vocab * dim)
        segments.append(LayoutSegment(name: "adamEmbed.m", byteOffset: off, byteCount: bytes(vocab * dim))); off += bytes(vocab * dim)
        segments.append(LayoutSegment(name: "adamEmbed.v", byteOffset: off, byteCount: bytes(vocab * dim))); off += bytes(vocab * dim)

        return segments
    }

    internal static func _saveTiny(
        path: String,
        dim: Int,
        hidden: Int,
        nLayers: Int,
        vocab: Int,
        meta: CheckpointMeta,
        layers: borrowing LayerStorage<TinyLayerWeights>,
        layerAdam: borrowing LayerStorage<TinyLayerAdam>,
        rmsFinal: borrowing TensorBuffer,
        adamRmsFinal: borrowing AdamState,
        embed: borrowing TensorBuffer,
        adamEmbed: borrowing AdamState
    ) throws(CheckpointError) {
        var header = try makeHeader(
            dim: dim,
            hidden: hidden,
            nLayers: nLayers,
            nHeads: 1,
            seqLen: 2,
            vocab: vocab,
            meta: meta
        )

        guard let file = fopen(path, "wb") else {
            throw .openFailed(path: path, errno: errno)
        }
        defer { fclose(file) }

        try writeHeader(file, header: &header)
        try writeAllLayers(file, nLayers: nLayers, layers: layers, layerAdam: layerAdam)
        try writeGlobals(file, rmsFinal: rmsFinal, adamRmsFinal: adamRmsFinal, embed: embed, adamEmbed: adamEmbed)
    }

    internal static func _loadTiny(
        path: String,
        dim: Int,
        hidden: Int,
        nLayers: Int,
        vocab: Int
    ) throws(CheckpointError) -> CheckpointMeta {
        guard let file = fopen(path, "rb") else {
            throw .openFailed(path: path, errno: errno)
        }
        defer { fclose(file) }

        let header = try readAndValidateHeader(file)
        try validateHeaderConfig(header, dim: dim, hidden: hidden, nLayers: nLayers, nHeads: Int(header.nHeads), seqLen: Int(header.seqLen), vocab: vocab)
        return meta(from: header)
    }

    internal static func _loadTiny(
        path: String,
        dim: Int,
        hidden: Int,
        nLayers: Int,
        vocab: Int,
        intoLayers layers: borrowing LayerStorage<TinyLayerWeights>,
        intoLayerAdam layerAdam: borrowing LayerStorage<TinyLayerAdam>,
        intoRmsFinal rmsFinal: borrowing TensorBuffer,
        intoAdamRmsFinal adamRmsFinal: borrowing AdamState,
        intoEmbed embed: borrowing TensorBuffer,
        intoAdamEmbed adamEmbed: borrowing AdamState
    ) throws(CheckpointError) -> CheckpointMeta {
        guard let file = fopen(path, "rb") else {
            throw .openFailed(path: path, errno: errno)
        }
        defer { fclose(file) }

        let header = try readAndValidateHeader(file)
        try validateHeaderConfig(header, dim: dim, hidden: hidden, nLayers: nLayers, nHeads: Int(header.nHeads), seqLen: Int(header.seqLen), vocab: vocab)
        try readAllLayers(file, nLayers: nLayers, layers: layers, layerAdam: layerAdam)
        try readGlobals(file, rmsFinal: rmsFinal, adamRmsFinal: adamRmsFinal, embed: embed, adamEmbed: adamEmbed)
        return meta(from: header)
    }

    // MARK: - Header

    @inline(__always)
    private static func checkedInt32(_ value: Int, field: String) throws(CheckpointError) -> Int32 {
        guard value >= Int(Int32.min), value <= Int(Int32.max) else {
            throw .argumentOutOfRange("Field \(field) out of Int32 range: \(value)")
        }
        return Int32(value)
    }

    private static func makeHeader(
        dim: Int,
        hidden: Int,
        nLayers: Int,
        nHeads: Int,
        seqLen: Int,
        vocab: Int,
        meta: CheckpointMeta
    ) throws(CheckpointError) -> CheckpointHeader {
        var h = CheckpointHeader()
        h.step = try checkedInt32(meta.step, field: "step")
        h.totalSteps = try checkedInt32(meta.totalSteps, field: "totalSteps")
        h.nLayers = try checkedInt32(nLayers, field: "nLayers")
        h.vocabSize = try checkedInt32(vocab, field: "vocabSize")
        h.dim = try checkedInt32(dim, field: "dim")
        h.hiddenDim = try checkedInt32(hidden, field: "hiddenDim")
        h.nHeads = try checkedInt32(nHeads, field: "nHeads")
        h.seqLen = try checkedInt32(seqLen, field: "seqLen")
        h.lr = meta.lr
        h.loss = meta.loss
        h.cumCompile = meta.cumCompile
        h.cumTrain = meta.cumTrain
        h.cumWall = meta.cumWall
        h.cumSteps = try checkedInt32(meta.cumSteps, field: "cumSteps")
        h.cumBatches = try checkedInt32(meta.cumBatches, field: "cumBatches")
        h.adamT = try checkedInt32(meta.adamT, field: "adamT")
        return h
    }

    private static func meta(from header: CheckpointHeader) -> CheckpointMeta {
        var m = CheckpointMeta()
        m.step = Int(header.step)
        m.totalSteps = Int(header.totalSteps)
        m.lr = header.lr
        m.loss = header.loss
        m.cumCompile = header.cumCompile
        m.cumTrain = header.cumTrain
        m.cumWall = header.cumWall
        m.cumSteps = Int(header.cumSteps)
        m.cumBatches = Int(header.cumBatches)
        m.adamT = Int(header.adamT)
        return m
    }

    private static func writeHeader(_ file: UnsafeMutablePointer<FILE>, header: inout CheckpointHeader) throws(CheckpointError) {
        let wroteBytes = withUnsafeBytes(of: &header) { raw -> Int in
            fwrite(raw.baseAddress, 1, raw.count, file)
        }
        guard wroteBytes == MemoryLayout<CheckpointHeader>.stride else {
            throw .ioWriteFailed(segment: "header", expectedCount: MemoryLayout<CheckpointHeader>.stride, actualCount: wroteBytes)
        }
    }

    private static func readAndValidateHeader(_ file: UnsafeMutablePointer<FILE>) throws(CheckpointError) -> CheckpointHeader {
        var header = CheckpointHeader()
        let readBytes = withUnsafeMutableBytes(of: &header) { raw -> Int in
            fread(raw.baseAddress, 1, raw.count, file)
        }
        guard readBytes == MemoryLayout<CheckpointHeader>.stride else {
            throw .ioReadFailed(segment: "header", expectedCount: MemoryLayout<CheckpointHeader>.stride, actualCount: readBytes)
        }
        guard header.magic == expectedMagic else {
            throw .invalidMagic(header.magic)
        }
        guard header.version == expectedVersion else {
            throw .unsupportedVersion(header.version)
        }
        return header
    }

    private static func validateHeaderConfig(
        _ header: CheckpointHeader,
        dim: Int,
        hidden: Int,
        nLayers: Int,
        nHeads: Int,
        seqLen: Int,
        vocab: Int
    ) throws(CheckpointError) {
        guard Int(header.dim) == dim,
              Int(header.hiddenDim) == hidden,
              Int(header.nLayers) == nLayers,
              Int(header.vocabSize) == vocab,
              Int(header.nHeads) == nHeads,
              Int(header.seqLen) == seqLen else {
            let expected = "dim=\(dim) hidden=\(hidden) layers=\(nLayers) heads=\(nHeads) seq=\(seqLen) vocab=\(vocab)"
            let got = "dim=\(header.dim) hidden=\(header.hiddenDim) layers=\(header.nLayers) heads=\(header.nHeads) seq=\(header.seqLen) vocab=\(header.vocabSize)"
            throw .configMismatch(expected: expected, got: got)
        }
    }

    // MARK: - Payload I/O

    @inline(__always)
    private static func writeFloats(
        _ file: UnsafeMutablePointer<FILE>,
        segment: String,
        buffer: borrowing TensorBuffer
    ) throws(CheckpointError) {
        let wrote = buffer.withUnsafePointer { ptr in
            fwrite(ptr, MemoryLayout<Float>.stride, buffer.count, file)
        }
        guard wrote == buffer.count else {
            throw .ioWriteFailed(segment: segment, expectedCount: buffer.count, actualCount: wrote)
        }
    }

    @inline(__always)
    private static func readFloats(
        _ file: UnsafeMutablePointer<FILE>,
        segment: String,
        into buffer: borrowing TensorBuffer
    ) throws(CheckpointError) {
        let read = buffer.withUnsafeMutablePointer { ptr in
            fread(ptr, MemoryLayout<Float>.stride, buffer.count, file)
        }
        guard read == buffer.count else {
            throw .ioReadFailed(segment: segment, expectedCount: buffer.count, actualCount: read)
        }
    }

    private static func writeLayerPayload(
        _ file: UnsafeMutablePointer<FILE>,
        layer: Int,
        Wq: borrowing TensorBuffer,
        Wk: borrowing TensorBuffer,
        Wv: borrowing TensorBuffer,
        Wo: borrowing TensorBuffer,
        W1: borrowing TensorBuffer,
        W2: borrowing TensorBuffer,
        W3: borrowing TensorBuffer,
        rmsAtt: borrowing TensorBuffer,
        rmsFfn: borrowing TensorBuffer,
        aWq: borrowing AdamState,
        aWk: borrowing AdamState,
        aWv: borrowing AdamState,
        aWo: borrowing AdamState,
        aW1: borrowing AdamState,
        aW2: borrowing AdamState,
        aW3: borrowing AdamState,
        aRmsAtt: borrowing AdamState,
        aRmsFfn: borrowing AdamState
    ) throws(CheckpointError) {
        // Per-layer weights (ObjC order)
        try writeFloats(file, segment: "L\(layer).Wq", buffer: Wq)
        try writeFloats(file, segment: "L\(layer).Wk", buffer: Wk)
        try writeFloats(file, segment: "L\(layer).Wv", buffer: Wv)
        try writeFloats(file, segment: "L\(layer).Wo", buffer: Wo)
        try writeFloats(file, segment: "L\(layer).W1", buffer: W1)
        try writeFloats(file, segment: "L\(layer).W2", buffer: W2)
        try writeFloats(file, segment: "L\(layer).W3", buffer: W3)
        try writeFloats(file, segment: "L\(layer).rmsAtt", buffer: rmsAtt)
        try writeFloats(file, segment: "L\(layer).rmsFfn", buffer: rmsFfn)

        // Per-layer Adam state (m then v per parameter, same order as weights)
        try writeFloats(file, segment: "L\(layer).Wq.m", buffer: aWq.m)
        try writeFloats(file, segment: "L\(layer).Wq.v", buffer: aWq.v)
        try writeFloats(file, segment: "L\(layer).Wk.m", buffer: aWk.m)
        try writeFloats(file, segment: "L\(layer).Wk.v", buffer: aWk.v)
        try writeFloats(file, segment: "L\(layer).Wv.m", buffer: aWv.m)
        try writeFloats(file, segment: "L\(layer).Wv.v", buffer: aWv.v)
        try writeFloats(file, segment: "L\(layer).Wo.m", buffer: aWo.m)
        try writeFloats(file, segment: "L\(layer).Wo.v", buffer: aWo.v)
        try writeFloats(file, segment: "L\(layer).W1.m", buffer: aW1.m)
        try writeFloats(file, segment: "L\(layer).W1.v", buffer: aW1.v)
        try writeFloats(file, segment: "L\(layer).W2.m", buffer: aW2.m)
        try writeFloats(file, segment: "L\(layer).W2.v", buffer: aW2.v)
        try writeFloats(file, segment: "L\(layer).W3.m", buffer: aW3.m)
        try writeFloats(file, segment: "L\(layer).W3.v", buffer: aW3.v)
        try writeFloats(file, segment: "L\(layer).rmsAtt.m", buffer: aRmsAtt.m)
        try writeFloats(file, segment: "L\(layer).rmsAtt.v", buffer: aRmsAtt.v)
        try writeFloats(file, segment: "L\(layer).rmsFfn.m", buffer: aRmsFfn.m)
        try writeFloats(file, segment: "L\(layer).rmsFfn.v", buffer: aRmsFfn.v)
    }

    private static func readLayerPayload(
        _ file: UnsafeMutablePointer<FILE>,
        layer: Int,
        Wq: borrowing TensorBuffer,
        Wk: borrowing TensorBuffer,
        Wv: borrowing TensorBuffer,
        Wo: borrowing TensorBuffer,
        W1: borrowing TensorBuffer,
        W2: borrowing TensorBuffer,
        W3: borrowing TensorBuffer,
        rmsAtt: borrowing TensorBuffer,
        rmsFfn: borrowing TensorBuffer,
        aWq: borrowing AdamState,
        aWk: borrowing AdamState,
        aWv: borrowing AdamState,
        aWo: borrowing AdamState,
        aW1: borrowing AdamState,
        aW2: borrowing AdamState,
        aW3: borrowing AdamState,
        aRmsAtt: borrowing AdamState,
        aRmsFfn: borrowing AdamState
    ) throws(CheckpointError) {
        try readFloats(file, segment: "L\(layer).Wq", into: Wq)
        try readFloats(file, segment: "L\(layer).Wk", into: Wk)
        try readFloats(file, segment: "L\(layer).Wv", into: Wv)
        try readFloats(file, segment: "L\(layer).Wo", into: Wo)
        try readFloats(file, segment: "L\(layer).W1", into: W1)
        try readFloats(file, segment: "L\(layer).W2", into: W2)
        try readFloats(file, segment: "L\(layer).W3", into: W3)
        try readFloats(file, segment: "L\(layer).rmsAtt", into: rmsAtt)
        try readFloats(file, segment: "L\(layer).rmsFfn", into: rmsFfn)

        try readFloats(file, segment: "L\(layer).Wq.m", into: aWq.m)
        try readFloats(file, segment: "L\(layer).Wq.v", into: aWq.v)
        try readFloats(file, segment: "L\(layer).Wk.m", into: aWk.m)
        try readFloats(file, segment: "L\(layer).Wk.v", into: aWk.v)
        try readFloats(file, segment: "L\(layer).Wv.m", into: aWv.m)
        try readFloats(file, segment: "L\(layer).Wv.v", into: aWv.v)
        try readFloats(file, segment: "L\(layer).Wo.m", into: aWo.m)
        try readFloats(file, segment: "L\(layer).Wo.v", into: aWo.v)
        try readFloats(file, segment: "L\(layer).W1.m", into: aW1.m)
        try readFloats(file, segment: "L\(layer).W1.v", into: aW1.v)
        try readFloats(file, segment: "L\(layer).W2.m", into: aW2.m)
        try readFloats(file, segment: "L\(layer).W2.v", into: aW2.v)
        try readFloats(file, segment: "L\(layer).W3.m", into: aW3.m)
        try readFloats(file, segment: "L\(layer).W3.v", into: aW3.v)
        try readFloats(file, segment: "L\(layer).rmsAtt.m", into: aRmsAtt.m)
        try readFloats(file, segment: "L\(layer).rmsAtt.v", into: aRmsAtt.v)
        try readFloats(file, segment: "L\(layer).rmsFfn.m", into: aRmsFfn.m)
        try readFloats(file, segment: "L\(layer).rmsFfn.v", into: aRmsFfn.v)
    }

    private static func writeAllLayers(
        _ file: UnsafeMutablePointer<FILE>,
        nLayers: Int,
        layers: borrowing LayerStorage<LayerWeights>,
        layerAdam: borrowing LayerStorage<LayerAdam>
    ) throws(CheckpointError) {
        for L in 0..<nLayers {
            try writeLayerPayload(
                file,
                layer: L,
                Wq: layers[L].Wq,
                Wk: layers[L].Wk,
                Wv: layers[L].Wv,
                Wo: layers[L].Wo,
                W1: layers[L].W1,
                W2: layers[L].W2,
                W3: layers[L].W3,
                rmsAtt: layers[L].rmsAtt,
                rmsFfn: layers[L].rmsFfn,
                aWq: layerAdam[L].Wq,
                aWk: layerAdam[L].Wk,
                aWv: layerAdam[L].Wv,
                aWo: layerAdam[L].Wo,
                aW1: layerAdam[L].W1,
                aW2: layerAdam[L].W2,
                aW3: layerAdam[L].W3,
                aRmsAtt: layerAdam[L].rmsAtt,
                aRmsFfn: layerAdam[L].rmsFfn
            )
        }
    }

    private static func writeAllLayers(
        _ file: UnsafeMutablePointer<FILE>,
        nLayers: Int,
        layers: borrowing LayerStorage<TinyLayerWeights>,
        layerAdam: borrowing LayerStorage<TinyLayerAdam>
    ) throws(CheckpointError) {
        for L in 0..<nLayers {
            try writeLayerPayload(
                file,
                layer: L,
                Wq: layers[L].Wq,
                Wk: layers[L].Wk,
                Wv: layers[L].Wv,
                Wo: layers[L].Wo,
                W1: layers[L].W1,
                W2: layers[L].W2,
                W3: layers[L].W3,
                rmsAtt: layers[L].rmsAtt,
                rmsFfn: layers[L].rmsFfn,
                aWq: layerAdam[L].Wq,
                aWk: layerAdam[L].Wk,
                aWv: layerAdam[L].Wv,
                aWo: layerAdam[L].Wo,
                aW1: layerAdam[L].W1,
                aW2: layerAdam[L].W2,
                aW3: layerAdam[L].W3,
                aRmsAtt: layerAdam[L].rmsAtt,
                aRmsFfn: layerAdam[L].rmsFfn
            )
        }
    }

    private static func readAllLayers(
        _ file: UnsafeMutablePointer<FILE>,
        nLayers: Int,
        layers: borrowing LayerStorage<LayerWeights>,
        layerAdam: borrowing LayerStorage<LayerAdam>
    ) throws(CheckpointError) {
        for L in 0..<nLayers {
            try readLayerPayload(
                file,
                layer: L,
                Wq: layers[L].Wq,
                Wk: layers[L].Wk,
                Wv: layers[L].Wv,
                Wo: layers[L].Wo,
                W1: layers[L].W1,
                W2: layers[L].W2,
                W3: layers[L].W3,
                rmsAtt: layers[L].rmsAtt,
                rmsFfn: layers[L].rmsFfn,
                aWq: layerAdam[L].Wq,
                aWk: layerAdam[L].Wk,
                aWv: layerAdam[L].Wv,
                aWo: layerAdam[L].Wo,
                aW1: layerAdam[L].W1,
                aW2: layerAdam[L].W2,
                aW3: layerAdam[L].W3,
                aRmsAtt: layerAdam[L].rmsAtt,
                aRmsFfn: layerAdam[L].rmsFfn
            )
        }
    }

    private static func readAllLayers(
        _ file: UnsafeMutablePointer<FILE>,
        nLayers: Int,
        layers: borrowing LayerStorage<TinyLayerWeights>,
        layerAdam: borrowing LayerStorage<TinyLayerAdam>
    ) throws(CheckpointError) {
        for L in 0..<nLayers {
            try readLayerPayload(
                file,
                layer: L,
                Wq: layers[L].Wq,
                Wk: layers[L].Wk,
                Wv: layers[L].Wv,
                Wo: layers[L].Wo,
                W1: layers[L].W1,
                W2: layers[L].W2,
                W3: layers[L].W3,
                rmsAtt: layers[L].rmsAtt,
                rmsFfn: layers[L].rmsFfn,
                aWq: layerAdam[L].Wq,
                aWk: layerAdam[L].Wk,
                aWv: layerAdam[L].Wv,
                aWo: layerAdam[L].Wo,
                aW1: layerAdam[L].W1,
                aW2: layerAdam[L].W2,
                aW3: layerAdam[L].W3,
                aRmsAtt: layerAdam[L].rmsAtt,
                aRmsFfn: layerAdam[L].rmsFfn
            )
        }
    }

    private static func writeGlobals(
        _ file: UnsafeMutablePointer<FILE>,
        rmsFinal: borrowing TensorBuffer,
        adamRmsFinal: borrowing AdamState,
        embed: borrowing TensorBuffer,
        adamEmbed: borrowing AdamState
    ) throws(CheckpointError) {
        try writeFloats(file, segment: "rmsFinal", buffer: rmsFinal)
        try writeFloats(file, segment: "adamRmsFinal.m", buffer: adamRmsFinal.m)
        try writeFloats(file, segment: "adamRmsFinal.v", buffer: adamRmsFinal.v)
        try writeFloats(file, segment: "embed", buffer: embed)
        try writeFloats(file, segment: "adamEmbed.m", buffer: adamEmbed.m)
        try writeFloats(file, segment: "adamEmbed.v", buffer: adamEmbed.v)
    }

    private static func readGlobals(
        _ file: UnsafeMutablePointer<FILE>,
        rmsFinal: borrowing TensorBuffer,
        adamRmsFinal: borrowing AdamState,
        embed: borrowing TensorBuffer,
        adamEmbed: borrowing AdamState
    ) throws(CheckpointError) {
        try readFloats(file, segment: "rmsFinal", into: rmsFinal)
        try readFloats(file, segment: "adamRmsFinal.m", into: adamRmsFinal.m)
        try readFloats(file, segment: "adamRmsFinal.v", into: adamRmsFinal.v)
        try readFloats(file, segment: "embed", into: embed)
        try readFloats(file, segment: "adamEmbed.m", into: adamEmbed.m)
        try readFloats(file, segment: "adamEmbed.v", into: adamEmbed.v)
    }
}
