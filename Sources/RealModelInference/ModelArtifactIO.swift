import ANETypes
import Foundation
import ModelSupport

// Model artifact IO (extracted from RealModelInferenceEngine).
//
// Everything about reading an artifact from disk: metadata.json parsing,
// weight-blob and float32-sidecar loading, top-level weight-path resolution,
// tokenizer discovery. Call surface is unchanged; only the file moved.

extension RealModelInferenceEngine {
    static func loadConfigFromMetadataFile(at metadataURL: URL) throws -> MultiModelConfig {
        let data: Data
        do {
            data = try Data(contentsOf: metadataURL)
        } catch {
            throw RealModelInferenceError.runtimeFailure("Failed to read metadata.json: \(error)")
        }

        let object: Any
        do {
            object = try JSONSerialization.jsonObject(with: data)
        } catch {
            throw RealModelInferenceError.runtimeFailure("metadata.json is not valid JSON: \(error)")
        }

        guard let metadata = object as? [String: Any] else {
            throw RealModelInferenceError.runtimeFailure("metadata.json must be a JSON object")
        }

        func requiredInt(_ key: String) throws -> Int {
            guard let number = metadata[key] as? NSNumber else {
                throw RealModelInferenceError.runtimeFailure("metadata.json missing numeric field \(key)")
            }
            return number.intValue
        }

        func requiredDouble(_ key: String) throws -> Double {
            guard let number = metadata[key] as? NSNumber else {
                throw RealModelInferenceError.runtimeFailure("metadata.json missing numeric field \(key)")
            }
            return number.doubleValue
        }

        guard let name = metadata["name"] as? String, !name.isEmpty else {
            throw RealModelInferenceError.runtimeFailure("metadata.json missing string field name")
        }
        let architecture: MultiModelConfig.Architecture
        switch (metadata["architecture"] as? String)?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() {
        case "gpt2":
            architecture = .gpt2
        case "llama":
            architecture = .llama
        default:
            throw RealModelInferenceError.runtimeFailure("metadata.json missing supported architecture")
        }

        let preferredDecodePath: MultiModelConfig.PreferredDecodePath?
        if let value = metadata["preferredDecodePath"] {
            let raw: String
            if let string = value as? String {
                raw = string
            } else {
                raw = String(describing: value)
            }
            do {
                preferredDecodePath = try MultiModelConfig.PreferredDecodePath.parse(raw)
            } catch {
                throw RealModelInferenceError.runtimeFailure(
                    "metadata.json has unsupported preferredDecodePath \"\(raw)\" (expected \"hybrid\" or \"exact_cpu\")"
                )
            }
        } else {
            preferredDecodePath = nil
        }

        return MultiModelConfig(
            name: name,
            nLayer: try requiredInt("nLayer"),
            nHead: try requiredInt("nHead"),
            nKVHead: try requiredInt("nKVHead"),
            dModel: try requiredInt("dModel"),
            headDim: try requiredInt("headDim"),
            hiddenDim: try requiredInt("hiddenDim"),
            vocab: try requiredInt("vocab"),
            maxSeq: try requiredInt("maxSeq"),
            normEps: Float(try requiredDouble("normEps")),
            ropeTheta: Float((metadata["ropeTheta"] as? NSNumber)?.doubleValue ?? 10_000),
            eosToken: (metadata["eosToken"] as? NSNumber)?.uint32Value,
            architecture: architecture,
            preferredDecodePath: preferredDecodePath
        )
    }

    static func resolveTopLevelWeightPaths(
        config: MultiModelConfig,
        weightDir: String
    ) throws -> TopLevelWeightPaths {
        let root = URL(fileURLWithPath: weightDir, isDirectory: true)
        try validateDirectory(root)

        switch config.architecture {
        case .gpt2:
            return TopLevelWeightPaths(
                tokenEmbedding: try requiredFile(
                    root: root,
                    candidates: ["embeddings/token.bin", "embeddings/token_embeddings.bin"],
                    label: "token embedding"
                ),
                positionEmbedding: try requiredFile(
                    root: root,
                    candidates: ["embeddings/position.bin", "embeddings/position_embeddings.bin"],
                    label: "position embedding"
                ),
                finalNormGamma: try requiredFile(
                    root: root,
                    candidates: ["final_norm_gamma.bin", "ln_f_gamma.bin", "rms_final.bin"],
                    label: "final norm gamma"
                ),
                finalNormBeta: try requiredFile(
                    root: root,
                    candidates: ["final_norm_beta.bin", "ln_f_beta.bin", "rms_final_beta.bin"],
                    label: "final norm beta"
                ),
                lmHead: try requiredFile(
                    root: root,
                    candidates: ["lm_head.bin", "classifier.bin"],
                    label: "lm head"
                )
            )
        case .llama:
            return TopLevelWeightPaths(
                tokenEmbedding: try requiredFile(
                    root: root,
                    candidates: ["embeddings/token.bin", "embeddings/token_embeddings.bin"],
                    label: "token embedding"
                ),
                positionEmbedding: "",
                finalNormGamma: try requiredFile(
                    root: root,
                    candidates: ["rms_final.bin", "final_norm_gamma.bin"],
                    label: "final norm gamma"
                ),
                finalNormBeta: "",
                lmHead: try requiredFile(
                    root: root,
                    candidates: ["lm_head.bin", "classifier.bin"],
                    label: "lm head"
                )
            )
        }
    }

    struct LlamaTopLevelWeightPaths: Sendable, Equatable {
        let tokenEmbedding: String
        let finalNormGamma: String
        let lmHead: String
    }

    static func resolveLlamaTopLevelWeightPaths(
        config: MultiModelConfig,
        weightDir: String
    ) throws -> LlamaTopLevelWeightPaths {
        let root = URL(fileURLWithPath: weightDir, isDirectory: true)
        try validateDirectory(root)
        return LlamaTopLevelWeightPaths(
            tokenEmbedding: try requiredFile(
                root: root,
                candidates: ["embeddings/token.bin", "embeddings/token_embeddings.bin"],
                label: "token embedding"
            ),
            finalNormGamma: try requiredFile(
                root: root,
                candidates: ["rms_final.bin", "final_norm_gamma.bin", "final_norm.bin"],
                label: "final norm gamma"
            ),
            lmHead: try requiredFile(
                root: root,
                candidates: ["lm_head.bin", "classifier.bin"],
                label: "lm head"
            )
        )
    }


    static func loadTokenizer(
        config: MultiModelConfig,
        tokenizerDirURL: URL
    ) throws -> LoadedTokenizer {
        switch config.architecture {
        case .gpt2:
            let vocabURL = tokenizerDirURL.appendingPathComponent("vocab.json")
            let mergesURL = tokenizerDirURL.appendingPathComponent("merges.txt")
            guard FileManager.default.fileExists(atPath: vocabURL.path) else {
                throw RealModelInferenceError.missingPath(vocabURL.path)
            }
            guard FileManager.default.fileExists(atPath: mergesURL.path) else {
                throw RealModelInferenceError.missingPath(mergesURL.path)
            }
            do {
                return .gpt2(try GPT2BPETokenizer(vocabURL: vocabURL, mergesURL: mergesURL))
            } catch {
                throw RealModelInferenceError.runtimeFailure("Failed to load GPT-2 tokenizer: \(error)")
            }
        case .llama:
            // Try SentencePiece first (Llama, Mistral)
            let spCandidates = ["tokenizer.model", "tokenizer.bin"]
            for candidate in spCandidates {
                let url = tokenizerDirURL.appendingPathComponent(candidate)
                if FileManager.default.fileExists(atPath: url.path) {
                    do {
                        return .sentencePiece(try SentencePieceTokenizer(modelURL: url))
                    } catch {
                        throw RealModelInferenceError.runtimeFailure("Failed to load SentencePiece tokenizer: \(error)")
                    }
                }
            }
            let tokenizerJSONURL = tokenizerDirURL.appendingPathComponent("tokenizer.json")
            if FileManager.default.fileExists(atPath: tokenizerJSONURL.path) {
                do {
                    return .gpt2(try GPT2BPETokenizer(tokenizerJSONURL: tokenizerJSONURL))
                } catch {
                    throw RealModelInferenceError.runtimeFailure("Failed to load tokenizer.json BPE tokenizer: \(error)")
                }
            }
            // Fallback to GPT-2 BPE (Qwen uses BPE with llama-family architecture)
            let vocabURL = tokenizerDirURL.appendingPathComponent("vocab.json")
            let mergesURL = tokenizerDirURL.appendingPathComponent("merges.txt")
            if FileManager.default.fileExists(atPath: vocabURL.path),
               FileManager.default.fileExists(atPath: mergesURL.path) {
                do {
                    return .gpt2(try GPT2BPETokenizer(vocabURL: vocabURL, mergesURL: mergesURL))
                } catch {
                    throw RealModelInferenceError.runtimeFailure("Failed to load GPT-2 BPE tokenizer: \(error)")
                }
            }
            throw RealModelInferenceError.missingPath(
                "No tokenizer found in \(tokenizerDirURL.path) — tried tokenizer.model, tokenizer.bin, tokenizer.json, vocab.json+merges.txt"
            )
        }
    }


    private static func requiredFile(
        root: URL,
        candidates: [String],
        label: String
    ) throws -> String {
        for candidate in candidates {
            let path = root.appendingPathComponent(candidate).path
            if FileManager.default.fileExists(atPath: path) {
                return path
            }
        }
        throw RealModelInferenceError.missingPath(
            "\(root.path)/<\(label): \(candidates.joined(separator: " | "))>"
        )
    }

    static func loadWeightTable(at path: String, expectedCount: Int) throws -> [Float] {
        let values: [Float]
        do {
            values = try BlobWeightLoader.load(from: path)
        } catch {
            throw RealModelInferenceError.runtimeFailure("Failed to load weight blob \(path): \(error)")
        }
        guard values.count == expectedCount else {
            throw RealModelInferenceError.invalidWeightCount(path: path, expected: expectedCount, actual: values.count)
        }
        return values
    }

    static func loadWeightTable(at path: String, allowedCounts: [Int]) throws -> [Float] {
        let values: [Float]
        do {
            values = try BlobWeightLoader.load(from: path)
        } catch {
            throw RealModelInferenceError.runtimeFailure("Failed to load weight blob \(path): \(error)")
        }
        guard allowedCounts.contains(values.count) else {
            let expected = allowedCounts.map(String.init).joined(separator: " or ")
            throw RealModelInferenceError.runtimeFailure(
                "Unexpected weight count for \(path): expected \(expected), got \(values.count)"
            )
        }
        return values
    }

    static func loadRawFP16WeightTableIfNoExactFloat32Sidecar(
        at path: String,
        expectedCount: Int
    ) throws -> [UInt16]? {
        let sidecarPath = exactFloat32SidecarPath(forBlobPath: path)
        guard !FileManager.default.fileExists(atPath: sidecarPath) else {
            return nil
        }

        let header: BlobWeightLoader.Header
        do {
            header = try BlobWeightLoader.readHeader(from: path)
        } catch {
            throw RealModelInferenceError.runtimeFailure("Failed to read weight blob header \(path): \(error)")
        }

        let expectedBytes = expectedCount * MemoryLayout<UInt16>.stride
        guard Int(header.dataSize) == expectedBytes else {
            throw RealModelInferenceError.invalidWeightCount(
                path: path,
                expected: expectedCount,
                actual: Int(header.dataSize) / MemoryLayout<UInt16>.stride
            )
        }

        let handle: FileHandle
        do {
            handle = try FileHandle(forReadingFrom: URL(fileURLWithPath: path))
        } catch {
            throw RealModelInferenceError.runtimeFailure("Failed to open weight blob \(path): \(error)")
        }
        defer { try? handle.close() }

        do {
            try handle.seek(toOffset: UInt64(header.dataOffset))
        } catch {
            throw RealModelInferenceError.runtimeFailure("Failed to seek weight blob \(path): \(error)")
        }

        let payload: Data
        do {
            payload = try handle.read(upToCount: expectedBytes) ?? Data()
        } catch {
            throw RealModelInferenceError.runtimeFailure("Failed to read weight blob payload \(path): \(error)")
        }
        guard payload.count == expectedBytes else {
            throw RealModelInferenceError.invalidWeightCount(
                path: path,
                expected: expectedCount,
                actual: payload.count / MemoryLayout<UInt16>.stride
            )
        }

        return payload.withUnsafeBytes { raw in
            (0..<expectedCount).map { index in
                let bits = raw.loadUnaligned(
                    fromByteOffset: index * MemoryLayout<UInt16>.stride,
                    as: UInt16.self
                )
                return UInt16(littleEndian: bits)
            }
        }
    }

    static func exactFloat32SidecarPath(forBlobPath path: String) -> String {
        if path.hasSuffix(".bin") {
            return String(path.dropLast(4)) + ".float32.bin"
        }
        return path + ".float32"
    }

    static func loadExactFloat32WeightTable(
        at path: String,
        expectedCount: Int
    ) throws -> [Float]? {
        let sidecarPath = exactFloat32SidecarPath(forBlobPath: path)
        guard FileManager.default.fileExists(atPath: sidecarPath) else {
            return nil
        }

        let data: Data
        do {
            data = try Data(contentsOf: URL(fileURLWithPath: sidecarPath))
        } catch {
            throw RealModelInferenceError.runtimeFailure("Failed to read float32 sidecar \(sidecarPath): \(error)")
        }

        let scalarSize = MemoryLayout<UInt32>.stride
        let expectedBytes = expectedCount * scalarSize
        guard data.count == expectedBytes else {
            throw RealModelInferenceError.invalidWeightCount(
                path: sidecarPath,
                expected: expectedCount,
                actual: data.count / scalarSize
            )
        }

        return data.withUnsafeBytes { raw in
            (0..<expectedCount).map { index in
                let bits = raw.loadUnaligned(fromByteOffset: index * scalarSize, as: UInt32.self)
                return Float(bitPattern: UInt32(littleEndian: bits))
            }
        }
    }

    static func loadWeightTablePreferringFloat32Sidecar(
        at path: String,
        expectedCount: Int
    ) throws -> [Float] {
        if let exactValues = try loadExactFloat32WeightTable(at: path, expectedCount: expectedCount) {
            return exactValues
        }
        return try loadWeightTable(at: path, expectedCount: expectedCount)
    }

    static func loadTensor(
        _ tensor: borrowing TensorBuffer,
        from path: String,
        expectedCount: Int
    ) throws {
        let values = try loadWeightTable(at: path, expectedCount: expectedCount)
        tensor.withUnsafeMutableBufferPointer { dst in
            values.withUnsafeBufferPointer { src in
                guard let dstBase = dst.baseAddress, let srcBase = src.baseAddress else {
                    return
                }
                dstBase.update(from: srcBase, count: expectedCount)
            }
        }
    }

    static func buildGroupedWeightBlob(
        from weights: [Float],
        rows: Int,
        colsPerGroup: Int,
        groups: Int
    ) -> Data {
        let compactCount = rows * colsPerGroup
        let repacked: [Float] = weights.withUnsafeBufferPointer { buffer in
            if groups == 1 || buffer.count == compactCount {
                return Array(buffer)
            }

            let denseCols = colsPerGroup * groups
            precondition(rows.isMultiple(of: groups))
            precondition(buffer.count == rows * denseCols)

            let rowsPerGroup = rows / groups
            var compact = [Float](repeating: 0, count: compactCount)
            for row in 0..<rows {
                let group = row / rowsPerGroup
                let srcStart = row * denseCols + group * colsPerGroup
                let dstStart = row * colsPerGroup
                for col in 0..<colsPerGroup {
                    compact[dstStart + col] = buffer[srcStart + col]
                }
            }
            return compact
        }
        return WeightBlob.build(from: repacked, rows: rows, cols: colsPerGroup)
    }

    static func fileExists(at path: String?) -> Bool {
        guard let path else { return false }
        return FileManager.default.fileExists(atPath: path)
    }

}
