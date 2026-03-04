import Darwin
import Foundation
import ANETypes

public enum TokenDatasetError: Error, Sendable, Equatable {
    case fileNotFound(String)
    case statFailed(errno: Int32)
    case invalidByteCount(Int)
    case mmapFailed(errno: Int32)
    case tooFewTokens(expectedAtLeast: Int, actual: Int)
}

/// Read-only mmap of a pre-tokenized UInt16 dataset.
///
/// The on-disk format is a flat array of UInt16 tokens in native endianness.
/// Maps to `train_large.m:274-281`.
public struct TokenDataset: ~Copyable {
    private let fd: Int32
    private let mappedBytes: Int
    private let mapping: UnsafeMutableRawPointer

    public let nTokens: Int

    // Internal hook for tests: verify CLOEXEC is set so exec-restarts don't leak fds.
    @usableFromInline
    internal var _debugFileDescriptor: Int32 { fd }

    public init(path: String, seqLen: Int = ModelConfig.seqLen) throws(TokenDatasetError) {
        precondition(seqLen >= 0)

        // Use O_CLOEXEC so file descriptors don't leak across exec() restarts.
        let fd = open(path, O_RDONLY | O_CLOEXEC)
        guard fd >= 0 else {
            throw .fileNotFound(path)
        }
        _ = fcntl(fd, F_SETFD, FD_CLOEXEC)

        var st = stat()
        guard fstat(fd, &st) == 0 else {
            let e = errno
            close(fd)
            throw .statFailed(errno: e)
        }

        let byteCount = Int(st.st_size)
        guard byteCount >= 0 else {
            close(fd)
            throw .invalidByteCount(byteCount)
        }
        guard byteCount % MemoryLayout<UInt16>.stride == 0 else {
            close(fd)
            throw .invalidByteCount(byteCount)
        }

        let map = mmap(nil, byteCount, PROT_READ, MAP_PRIVATE, fd, 0)
        guard let map, map != MAP_FAILED else {
            let e = errno
            close(fd)
            throw .mmapFailed(errno: e)
        }

        let nTokens = byteCount / MemoryLayout<UInt16>.stride
        let minTokens = seqLen + 1
        guard nTokens >= minTokens else {
            munmap(map, byteCount)
            close(fd)
            throw .tooFewTokens(expectedAtLeast: minTokens, actual: nTokens)
        }

        self.fd = fd
        self.mappedBytes = byteCount
        self.mapping = map
        self.nTokens = nTokens
    }

    public var tokensBase: UnsafePointer<UInt16> {
        let p: UnsafeMutablePointer<UInt16> = mapping.bindMemory(to: UInt16.self, capacity: nTokens)
        return UnsafePointer(p)
    }

    /// Returns a pointer into the mmap region at the given token offset.
    public subscript(offset: Int) -> UnsafePointer<UInt16> {
        precondition(offset >= 0 && offset < nTokens)
        return tokensBase.advanced(by: offset)
    }

    deinit {
        munmap(mapping, mappedBytes)
        close(fd)
    }
}
