import Darwin
import Foundation

// MARK: - Error Types

public enum TensorBufferMmapError: Error, Equatable, Sendable, CustomStringConvertible {
    case fileOpenFailed(path: String, errno: Int32)
    case fileStatFailed(path: String, errno: Int32)
    case fileTooSmall(path: String, requiredBytes: Int, actualBytes: Int)
    case mmapFailed(path: String, errno: Int32)

    public var description: String {
        switch self {
        case .fileOpenFailed(let path, let errno):
            return "TensorBuffer mmap: failed to open '\(path)': errno \(errno) (\(String(cString: strerror(errno))))"
        case .fileStatFailed(let path, let errno):
            return "TensorBuffer mmap: failed to stat '\(path)': errno \(errno) (\(String(cString: strerror(errno))))"
        case .fileTooSmall(let path, let requiredBytes, let actualBytes):
            return "TensorBuffer mmap: '\(path)' has \(actualBytes) bytes, need \(requiredBytes)"
        case .mmapFailed(let path, let errno):
            return "TensorBuffer mmap: mmap failed for '\(path)': errno \(errno) (\(String(cString: strerror(errno))))"
        }
    }
}

public enum TensorBufferWriteError: Error, Equatable, Sendable, CustomStringConvertible {
    case fileOpenFailed(path: String, errno: Int32)
    case writeFailed(path: String, written: Int, expected: Int, errno: Int32)

    public var description: String {
        switch self {
        case .fileOpenFailed(let path, let errno):
            return "TensorBuffer write: failed to open '\(path)': errno \(errno) (\(String(cString: strerror(errno))))"
        case .writeFailed(let path, let written, let expected, let errno):
            return "TensorBuffer write: wrote \(written) of \(expected) bytes to '\(path)': errno \(errno) (\(String(cString: strerror(errno))))"
        }
    }
}

// MARK: - TensorBuffer

public struct TensorBuffer: ~Copyable {
    public static let allocationAlignment: Int = 64

    public let count: Int
    private let rawStorage: UnsafeMutableRawPointer
    private let baseAddress: UnsafeMutablePointer<Float>
    private let storageKind: StorageKind

    // MARK: Storage kind

    private enum StorageKind {
        case owned
        case nonOwning
        case mmapped(mappedSize: Int)
    }

    // MARK: Helpers

    @inline(__always)
    private static func byteCount(for elementCount: Int) -> Int {
        let bytes = elementCount.multipliedReportingOverflow(by: MemoryLayout<Float>.stride)
        precondition(!bytes.overflow)
        return bytes.partialValue
    }

    // MARK: Initializers

    /// Allocates a new buffer of `count` Float32 elements, optionally zeroed.
    public init(count: Int, zeroed: Bool) {
        precondition(count >= 0)
        self.count = count
        self.storageKind = .owned

        let requestedBytes = Self.byteCount(for: count)
        let allocatedBytes = max(requestedBytes, 1)

        self.rawStorage = UnsafeMutableRawPointer.allocate(
            byteCount: allocatedBytes,
            alignment: Self.allocationAlignment
        )
        self.baseAddress = rawStorage.bindMemory(to: Float.self, capacity: count)
        if zeroed {
            rawStorage.initializeMemory(as: UInt8.self, repeating: 0, count: requestedBytes)
        }
    }

    /// Creates a non-owning view into an existing buffer's memory.
    ///
    /// The caller must guarantee the source buffer outlives this view.
    @inline(__always)
    public init(nonOwningViewOf source: borrowing TensorBuffer) {
        self.count = source.count
        self.rawStorage = source.rawStorage
        self.baseAddress = source.baseAddress
        self.storageKind = .nonOwning
    }

    /// Creates a non-owning view over an arbitrary Float pointer.
    ///
    /// The caller must guarantee the pointer remains valid
    /// for the lifetime of the returned TensorBuffer.
    @inline(__always)
    public init(nonOwningPointer ptr: UnsafeMutablePointer<Float>, count: Int) {
        precondition(count >= 0)
        self.count = count
        self.rawStorage = UnsafeMutableRawPointer(ptr)
        self.baseAddress = ptr
        self.storageKind = .nonOwning
    }

    /// Memory-maps a binary file of packed Float32 values.
    ///
    /// - Parameters:
    ///   - path: Path to the file containing raw IEEE 754 Float32 values.
    ///   - offset: Byte offset into the file at which the tensor data starts.
    ///             This is rounded down to the nearest page boundary; the
    ///             `baseAddress` is then adjusted to account for the remainder.
    ///   - count: Number of Float32 elements to map.
    ///
    /// The mapping uses `PROT_READ`, `MAP_PRIVATE`, and `MADV_SEQUENTIAL`.
    /// The caller should ensure the file is not deleted while the buffer is live.
    public init(mmapFrom path: String, offset: Int, count: Int) throws(TensorBufferMmapError) {
        precondition(offset >= 0)
        precondition(count >= 0)

        let fd = Darwin.open(path, O_RDONLY)
        guard fd >= 0 else {
            throw .fileOpenFailed(path: path, errno: Darwin.errno)
        }
        defer { Darwin.close(fd) }

        var statBuf = Darwin.stat()
        guard Darwin.fstat(fd, &statBuf) == 0 else {
            throw .fileStatFailed(path: path, errno: Darwin.errno)
        }

        let requiredBytes = offset + Self.byteCount(for: count)
        let fileBytes = Int(statBuf.st_size)
        guard fileBytes >= requiredBytes else {
            throw .fileTooSmall(path: path, requiredBytes: requiredBytes, actualBytes: fileBytes)
        }

        // mmap requires a page-aligned offset
        let pageSize = Int(Darwin.getpagesize())
        let alignedOffset = (offset / pageSize) * pageSize
        let offsetRemainder = offset - alignedOffset
        let mappedSize = Self.byteCount(for: count) + offsetRemainder

        let rawPtr = Darwin.mmap(
            nil,
            mappedSize,
            PROT_READ,
            MAP_PRIVATE,
            fd,
            off_t(alignedOffset)
        )
        guard let rawPtr, rawPtr != MAP_FAILED else {
            throw .mmapFailed(path: path, errno: Darwin.errno)
        }

        // Advise sequential access to allow kernel read-ahead
        Darwin.madvise(rawPtr, mappedSize, MADV_SEQUENTIAL)

        let mutableRaw = UnsafeMutableRawPointer(mutating: rawPtr)
        self.count = count
        self.rawStorage = mutableRaw
        self.baseAddress = mutableRaw
            .advanced(by: offsetRemainder)
            .bindMemory(to: Float.self, capacity: count)
        self.storageKind = .mmapped(mappedSize: mappedSize)
    }

    // MARK: Deinit

    deinit {
        switch storageKind {
        case .owned:
            rawStorage.deallocate()
        case .nonOwning:
            break
        case .mmapped(let mappedSize):
            _ = Darwin.munmap(rawStorage, mappedSize)
        }
    }

    // MARK: Slice

    /// Returns a non-owning sub-range view starting at `offset` with `count` elements.
    ///
    /// The returned buffer shares memory with `self`. The caller must ensure `self`
    /// (or whatever backing allocation it points into) outlives the returned slice.
    public func nonOwningSlice(offset: Int, count: Int) -> TensorBuffer {
        precondition(offset >= 0)
        precondition(count >= 0)
        precondition(offset + count <= self.count)
        return TensorBuffer(
            nonOwningPointer: baseAddress.advanced(by: offset),
            count: count
        )
    }

    // MARK: Write

    /// Writes the buffer contents to a binary file as packed Float32 values.
    ///
    /// The file is created or truncated at `path`.
    public func writeTo(path: String) throws(TensorBufferWriteError) {
        let fp = Darwin.fopen(path, "wb")
        guard let fp else {
            throw .fileOpenFailed(path: path, errno: Darwin.errno)
        }
        defer { Darwin.fclose(fp) }

        let elementCount = count
        let written = Darwin.fwrite(baseAddress, MemoryLayout<Float>.stride, elementCount, fp)
        if written != elementCount {
            throw .writeFailed(path: path, written: written, expected: elementCount, errno: Darwin.errno)
        }
    }

    // MARK: Pointer accessors

    @inline(__always)
    public func withUnsafeMutablePointer<R>(_ body: (UnsafeMutablePointer<Float>) throws -> R) rethrows -> R {
        try body(baseAddress)
    }

    @inline(__always)
    public func withUnsafePointer<R>(_ body: (UnsafePointer<Float>) throws -> R) rethrows -> R {
        try body(UnsafePointer(baseAddress))
    }

    @inline(__always)
    public func withUnsafeMutableBufferPointer<R>(_ body: (UnsafeMutableBufferPointer<Float>) throws -> R) rethrows -> R {
        try withUnsafeMutablePointer { ptr in
            try body(UnsafeMutableBufferPointer(start: ptr, count: count))
        }
    }

    @inline(__always)
    public func withUnsafeBufferPointer<R>(_ body: (UnsafeBufferPointer<Float>) throws -> R) rethrows -> R {
        try withUnsafePointer { ptr in
            try body(UnsafeBufferPointer(start: ptr, count: count))
        }
    }

    // MARK: Mutation

    public func zero() {
        rawStorage.initializeMemory(as: UInt8.self, repeating: 0, count: Self.byteCount(for: count))
    }
}
