import Foundation

public enum LocalTextTokenDatasetBuilderError: Error, Equatable, Sendable {
    case noRoots
    case noFilesFound
    case invalidUTF8Path(String)
    case ioFailure(String)
}

public enum LocalTextTokenDatasetBuilder {
    public static let fileSeparatorToken: UInt16 = 256

    public static func collectTokens(
        roots: [String],
        allowedExtensions: Set<String> = ["swift", "md", "txt", "py", "sh", "m", "h", "c"],
        maxFiles: Int? = nil,
        maxBytes: Int? = nil
    ) throws(LocalTextTokenDatasetBuilderError) -> [UInt16] {
        let cleanedRoots = roots.filter { !$0.isEmpty }
        guard !cleanedRoots.isEmpty else {
            throw .noRoots
        }

        let fileManager = FileManager.default
        var filePaths: [String] = []
        for root in cleanedRoots.sorted() {
            guard let enumerator = fileManager.enumerator(
                at: URL(fileURLWithPath: root, isDirectory: true),
                includingPropertiesForKeys: [.isRegularFileKey],
                options: [.skipsHiddenFiles, .skipsPackageDescendants]
            ) else {
                throw .ioFailure("failed to enumerate \(root)")
            }

            for case let url as URL in enumerator {
                let values = try? url.resourceValues(forKeys: [.isRegularFileKey])
                guard values?.isRegularFile == true else { continue }
                let ext = url.pathExtension.lowercased()
                guard allowedExtensions.contains(ext) else { continue }
                filePaths.append(url.path)
                if let maxFiles, filePaths.count >= maxFiles {
                    break
                }
            }
            if let maxFiles, filePaths.count >= maxFiles {
                break
            }
        }

        filePaths.sort()
        guard !filePaths.isEmpty else {
            throw .noFilesFound
        }

        var tokens: [UInt16] = []
        tokens.reserveCapacity(16_384)
        var consumedBytes = 0
        for (index, path) in filePaths.enumerated() {
            let data: Data
            do {
                data = try Data(contentsOf: URL(fileURLWithPath: path), options: [.mappedIfSafe])
            } catch {
                throw .ioFailure("failed to read \(path): \(error)")
            }

            for byte in data {
                if let maxBytes, consumedBytes >= maxBytes {
                    return tokens
                }
                tokens.append(UInt16(byte))
                consumedBytes += 1
            }

            if index + 1 < filePaths.count, maxBytes == nil || consumedBytes < maxBytes! {
                tokens.append(fileSeparatorToken)
            }
        }

        return tokens
    }

    public static func writeUInt16Dataset(
        tokens: [UInt16],
        to path: String
    ) throws(LocalTextTokenDatasetBuilderError) {
        var data = Data(capacity: tokens.count * MemoryLayout<UInt16>.stride)
        for token in tokens {
            var littleEndian = token.littleEndian
            withUnsafeBytes(of: &littleEndian) { bytes in
                data.append(bytes.bindMemory(to: UInt8.self))
            }
        }

        do {
            try data.write(to: URL(fileURLWithPath: path), options: .atomic)
        } catch {
            throw .ioFailure("failed to write \(path): \(error)")
        }
    }
}
