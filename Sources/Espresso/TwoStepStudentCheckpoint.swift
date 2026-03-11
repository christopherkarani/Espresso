import ANETypes
import Darwin
import Foundation

public enum TwoStepStudentCheckpointError: Error, Sendable, Equatable {
    case openFailed(path: String, errno: Int32)
    case ioWriteFailed(segment: String, expectedCount: Int, actualCount: Int)
    case ioReadFailed(segment: String, expectedCount: Int, actualCount: Int)
    case invalidMagic(Int32)
    case unsupportedVersion(Int32)
    case configMismatch(expected: String, got: String)
    case argumentOutOfRange(String)
}

public struct TwoStepStudentContract: Sendable, Equatable {
    public let dim: Int
    public let vocabSize: Int
    public let layerCount: Int
    public let horizon: Int
    public let exactPrefixOnly: Bool
    public let promotesPreparedState: Bool
    public let teacherClassifierWasShared: Bool

    public init(
        dim: Int,
        vocabSize: Int,
        layerCount: Int,
        horizon: Int = 2,
        exactPrefixOnly: Bool = true,
        promotesPreparedState: Bool = true,
        teacherClassifierWasShared: Bool
    ) throws(TwoStepStudentCheckpointError) {
        guard dim > 0 else {
            throw .argumentOutOfRange("dim must be > 0")
        }
        guard vocabSize > 0 else {
            throw .argumentOutOfRange("vocabSize must be > 0")
        }
        guard layerCount > 0 else {
            throw .argumentOutOfRange("layerCount must be > 0")
        }
        guard horizon == 2 else {
            throw .argumentOutOfRange("two-step student contract requires horizon == 2")
        }
        guard exactPrefixOnly else {
            throw .argumentOutOfRange("two-step student contract requires exactPrefixOnly == true")
        }
        guard promotesPreparedState else {
            throw .argumentOutOfRange("two-step student contract requires promotesPreparedState == true")
        }

        self.dim = dim
        self.vocabSize = vocabSize
        self.layerCount = layerCount
        self.horizon = horizon
        self.exactPrefixOnly = exactPrefixOnly
        self.promotesPreparedState = promotesPreparedState
        self.teacherClassifierWasShared = teacherClassifierWasShared
    }
}

public struct TwoStepStudentSidecar: ~Copyable {
    public let contract: TwoStepStudentContract
    public let futureRMS: TensorBuffer
    public let futureClassifier: TensorBuffer

    public init(
        contract: TwoStepStudentContract,
        futureRMS: consuming TensorBuffer,
        futureClassifier: consuming TensorBuffer
    ) throws(TwoStepStudentCheckpointError) {
        guard futureRMS.count == contract.dim else {
            throw .configMismatch(
                expected: "futureRMS.count=\(contract.dim)",
                got: "futureRMS.count=\(futureRMS.count)"
            )
        }
        let expectedClassifierCount = contract.dim * contract.vocabSize
        guard futureClassifier.count == expectedClassifierCount else {
            throw .configMismatch(
                expected: "futureClassifier.count=\(expectedClassifierCount)",
                got: "futureClassifier.count=\(futureClassifier.count)"
            )
        }

        self.contract = contract
        self.futureRMS = futureRMS
        self.futureClassifier = futureClassifier
    }
}

public enum TwoStepStudentCheckpoint {
    private static let expectedMagic: Int32 = 0x32535446 // "FTS2"
    private static let expectedVersion: Int32 = 1
    private static let flagExactPrefixOnly: UInt32 = 1 << 0
    private static let flagPromotesPreparedState: UInt32 = 1 << 1
    private static let flagTeacherClassifierWasShared: UInt32 = 1 << 2

    private struct Header {
        var magic: Int32
        var version: Int32
        var dim: Int32
        var vocabSize: Int32
        var layerCount: Int32
        var horizon: Int32
        var flags: UInt32
        var reserved: UInt32
    }

    public static func modelConfigContract(
        layerCount: Int = ModelConfig.nLayers,
        teacherClassifierWasShared: Bool
    ) throws(TwoStepStudentCheckpointError) -> TwoStepStudentContract {
        try TwoStepStudentContract(
            dim: ModelConfig.dim,
            vocabSize: ModelConfig.vocab,
            layerCount: layerCount,
            teacherClassifierWasShared: teacherClassifierWasShared
        )
    }

    public static func seedFromTeacher(
        dim: Int,
        vocabSize: Int,
        layerCount: Int,
        teacherRMS: borrowing TensorBuffer,
        teacherEmbedding: borrowing TensorBuffer,
        teacherClassifier: borrowing TensorBuffer,
        teacherClassifierWasShared: Bool
    ) throws(TwoStepStudentCheckpointError) -> TwoStepStudentSidecar {
        let contract = try TwoStepStudentContract(
            dim: dim,
            vocabSize: vocabSize,
            layerCount: layerCount,
            teacherClassifierWasShared: teacherClassifierWasShared
        )
        guard teacherRMS.count == dim else {
            throw .configMismatch(
                expected: "teacherRMS.count=\(dim)",
                got: "teacherRMS.count=\(teacherRMS.count)"
            )
        }

        let futureRMS = cloneTensor(teacherRMS)
        let expectedClassifierCount = dim * vocabSize
        let futureClassifier: TensorBuffer
        if teacherClassifierWasShared {
            guard teacherEmbedding.count == expectedClassifierCount else {
                throw .configMismatch(
                    expected: "teacherEmbedding.count=\(expectedClassifierCount)",
                    got: "teacherEmbedding.count=\(teacherEmbedding.count)"
                )
            }
            futureClassifier = cloneTensor(teacherEmbedding)
        } else {
            guard teacherClassifier.count == expectedClassifierCount else {
                throw .configMismatch(
                    expected: "teacherClassifier.count=\(expectedClassifierCount)",
                    got: "teacherClassifier.count=\(teacherClassifier.count)"
                )
            }
            futureClassifier = cloneTensor(teacherClassifier)
        }

        return try TwoStepStudentSidecar(
            contract: contract,
            futureRMS: futureRMS,
            futureClassifier: futureClassifier
        )
    }

    public static func save(
        path: String,
        sidecar: borrowing TwoStepStudentSidecar
    ) throws(TwoStepStudentCheckpointError) {
        guard let file = fopen(path, "wb") else {
            throw .openFailed(path: path, errno: errno)
        }
        defer { fclose(file) }

        var header = try makeHeader(contract: sidecar.contract)
        try writeHeader(file, header: &header)
        try writeBuffer(file, buffer: sidecar.futureRMS, segmentName: "futureRMS")
        try writeBuffer(file, buffer: sidecar.futureClassifier, segmentName: "futureClassifier")
    }

    public static func load(
        path: String,
        expectedContract: TwoStepStudentContract? = nil
    ) throws(TwoStepStudentCheckpointError) -> TwoStepStudentSidecar {
        guard let file = fopen(path, "rb") else {
            throw .openFailed(path: path, errno: errno)
        }
        defer { fclose(file) }

        let header = try readAndValidateHeader(file)
        let contract = try contract(from: header)
        if let expectedContract, expectedContract != contract {
            throw .configMismatch(
                expected: describe(contract: expectedContract),
                got: describe(contract: contract)
            )
        }

        let futureRMS = TensorBuffer(count: contract.dim, zeroed: false)
        let futureClassifier = TensorBuffer(count: contract.dim * contract.vocabSize, zeroed: false)
        try readBuffer(file, into: futureRMS, segmentName: "futureRMS")
        try readBuffer(file, into: futureClassifier, segmentName: "futureClassifier")

        return try TwoStepStudentSidecar(
            contract: contract,
            futureRMS: futureRMS,
            futureClassifier: futureClassifier
        )
    }

    @inline(__always)
    private static func cloneTensor(_ source: borrowing TensorBuffer) -> TensorBuffer {
        let copy = TensorBuffer(count: source.count, zeroed: false)
        source.withUnsafePointer { src in
            copy.withUnsafeMutablePointer { dst in
                dst.update(from: src, count: source.count)
            }
        }
        return copy
    }

    @inline(__always)
    private static func checkedInt32(_ value: Int, field: String) throws(TwoStepStudentCheckpointError) -> Int32 {
        guard value >= 0, value <= Int(Int32.max) else {
            throw .argumentOutOfRange("\(field) must fit Int32")
        }
        return Int32(value)
    }

    private static func makeHeader(
        contract: TwoStepStudentContract
    ) throws(TwoStepStudentCheckpointError) -> Header {
        var flags: UInt32 = 0
        if contract.exactPrefixOnly {
            flags |= flagExactPrefixOnly
        }
        if contract.promotesPreparedState {
            flags |= flagPromotesPreparedState
        }
        if contract.teacherClassifierWasShared {
            flags |= flagTeacherClassifierWasShared
        }

        return Header(
            magic: expectedMagic.littleEndian,
            version: expectedVersion.littleEndian,
            dim: try checkedInt32(contract.dim, field: "dim").littleEndian,
            vocabSize: try checkedInt32(contract.vocabSize, field: "vocabSize").littleEndian,
            layerCount: try checkedInt32(contract.layerCount, field: "layerCount").littleEndian,
            horizon: try checkedInt32(contract.horizon, field: "horizon").littleEndian,
            flags: flags.littleEndian,
            reserved: 0
        )
    }

    private static func contract(from header: Header) throws(TwoStepStudentCheckpointError) -> TwoStepStudentContract {
        let flags = UInt32(littleEndian: header.flags)
        return try TwoStepStudentContract(
            dim: Int(Int32(littleEndian: header.dim)),
            vocabSize: Int(Int32(littleEndian: header.vocabSize)),
            layerCount: Int(Int32(littleEndian: header.layerCount)),
            horizon: Int(Int32(littleEndian: header.horizon)),
            exactPrefixOnly: flags & flagExactPrefixOnly != 0,
            promotesPreparedState: flags & flagPromotesPreparedState != 0,
            teacherClassifierWasShared: flags & flagTeacherClassifierWasShared != 0
        )
    }

    private static func describe(contract: TwoStepStudentContract) -> String {
        "dim=\(contract.dim) vocab=\(contract.vocabSize) layers=\(contract.layerCount) horizon=\(contract.horizon) exactPrefixOnly=\(contract.exactPrefixOnly) promotesPreparedState=\(contract.promotesPreparedState) teacherClassifierWasShared=\(contract.teacherClassifierWasShared)"
    }

    private static func writeHeader(
        _ file: UnsafeMutablePointer<FILE>,
        header: inout Header
    ) throws(TwoStepStudentCheckpointError) {
        let wroteBytes = withUnsafePointer(to: &header) {
            fwrite($0, 1, MemoryLayout<Header>.stride, file)
        }
        guard wroteBytes == MemoryLayout<Header>.stride else {
            throw .ioWriteFailed(
                segment: "header",
                expectedCount: MemoryLayout<Header>.stride,
                actualCount: wroteBytes
            )
        }
    }

    private static func readAndValidateHeader(
        _ file: UnsafeMutablePointer<FILE>
    ) throws(TwoStepStudentCheckpointError) -> Header {
        var header = Header(
            magic: 0,
            version: 0,
            dim: 0,
            vocabSize: 0,
            layerCount: 0,
            horizon: 0,
            flags: 0,
            reserved: 0
        )
        let readBytes = withUnsafeMutablePointer(to: &header) {
            fread($0, 1, MemoryLayout<Header>.stride, file)
        }
        guard readBytes == MemoryLayout<Header>.stride else {
            throw .ioReadFailed(
                segment: "header",
                expectedCount: MemoryLayout<Header>.stride,
                actualCount: readBytes
            )
        }

        let magic = Int32(littleEndian: header.magic)
        guard magic == expectedMagic else {
            throw .invalidMagic(magic)
        }
        let version = Int32(littleEndian: header.version)
        guard version == expectedVersion else {
            throw .unsupportedVersion(version)
        }
        return header
    }

    private static func writeBuffer(
        _ file: UnsafeMutablePointer<FILE>,
        buffer: borrowing TensorBuffer,
        segmentName: String
    ) throws(TwoStepStudentCheckpointError) {
        let wroteCount = buffer.withUnsafePointer { ptr in
            fwrite(ptr, MemoryLayout<Float>.stride, buffer.count, file)
        }
        guard wroteCount == buffer.count else {
            throw .ioWriteFailed(
                segment: segmentName,
                expectedCount: buffer.count,
                actualCount: wroteCount
            )
        }
    }

    private static func readBuffer(
        _ file: UnsafeMutablePointer<FILE>,
        into buffer: borrowing TensorBuffer,
        segmentName: String
    ) throws(TwoStepStudentCheckpointError) {
        let readCount = buffer.withUnsafeMutablePointer { ptr in
            fread(ptr, MemoryLayout<Float>.stride, buffer.count, file)
        }
        guard readCount == buffer.count else {
            throw .ioReadFailed(
                segment: segmentName,
                expectedCount: buffer.count,
                actualCount: readCount
            )
        }
    }
}
