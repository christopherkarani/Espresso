import ANEInterop
import ANERuntime
import Darwin
import Foundation

public struct DeltaCompilationHandle: ~Copyable {
    let handle: OpaquePointer
    public let hexId: String

    deinit {
        ane_interop_free(handle)
    }
}

public enum DeltaCompilation {
    private enum CompileGate {
        static let lock = NSLock()
    }

    private enum RetryPolicy {
        static let maxAttempts = 5
        static let initialDelayMicroseconds: useconds_t = 100_000
        static let maxDelayMicroseconds: useconds_t = 1_000_000

        static func shouldRetry(lastCompileError: Int32, attemptIndex: Int) -> Bool {
            guard attemptIndex >= 0 else { return false }
            guard lastCompileError == ANE_INTEROP_COMPILE_ERROR_COMPILER_FAILURE else {
                return false
            }
            return attemptIndex + 1 < maxAttempts
        }

        static func delayMicroseconds(afterFailedAttempt attemptIndex: Int) -> useconds_t {
            guard attemptIndex > 0 else {
                return initialDelayMicroseconds
            }

            var delay = initialDelayMicroseconds
            for _ in 0..<attemptIndex {
                if delay >= maxDelayMicroseconds {
                    return maxDelayMicroseconds
                }
                delay = min(delay * 2, maxDelayMicroseconds)
            }
            return delay
        }

        static func sleepAfterFailedAttempt(_ attemptIndex: Int) {
            usleep(delayMicroseconds(afterFailedAttempt: attemptIndex))
        }
    }

    private static let hexIdBufferLength = 257

    @inline(__always)
    private static func mapInteropCompileError() -> ANEError {
        switch ane_interop_last_compile_error() {
        case ANE_INTEROP_COMPILE_ERROR_INVALID_ARGUMENTS:
            return .invalidArguments("ANE interop compile contract rejected arguments")
        case ANE_INTEROP_COMPILE_ERROR_DUPLICATE_WEIGHT_PATH:
            return .invalidArguments("Duplicate weight path detected")
        case ANE_INTEROP_COMPILE_ERROR_SURFACE_ALLOCATION_FAILED:
            return .surfaceAllocationFailed
        default:
            return .compilationFailed
        }
    }

    private static func checkedByteSizes(
        _ sizes: [Int],
        label: String
    ) throws(ANEError) -> [Int] {
        guard sizes.count <= Int(Int32.max),
              sizes.allSatisfy({ $0 >= 0 }) else {
            throw .invalidArguments("\(label) counts must fit Int32 and sizes must be non-negative")
        }
        return sizes
    }

    private static func checkedWeightTuples(
        _ weights: [(path: String, data: Data)]
    ) throws(ANEError) -> [(path: String, data: Data)] {
        guard weights.count <= Int(Int32.max) else {
            throw .invalidArguments("Weight count must fit Int32")
        }

        for (path, _) in weights {
            guard !path.isEmpty else {
                throw .invalidArguments("Weight paths must be non-empty")
            }
        }
        return weights
    }

    private static func validatedMILData(_ milText: String) throws(ANEError) -> Data {
        guard let milData = milText.data(using: .utf8), !milData.isEmpty else {
            throw .invalidArguments("MIL text must be valid, non-empty UTF-8")
        }
        return milData
    }

    private static func withPreparedBuffers<Result>(
        milData: Data,
        weights: [(path: String, data: Data)],
        inputSizes: [Int],
        outputSizes: [Int],
        _ body: (
            UnsafePointer<UInt8>,
            Int,
            UnsafeMutablePointer<UnsafePointer<CChar>?>?,
            UnsafeMutablePointer<UnsafePointer<UInt8>?>?,
            UnsafePointer<Int>?,
            Int32,
            UnsafePointer<Int>?,
            Int32,
            UnsafePointer<Int>?,
            Int32
        ) -> Result
    ) -> Result {
        let weightPaths = weights.map(\.path)
        let weightDatas = weights.map(\.data)
        var pathPointers = [UnsafePointer<CChar>?](repeating: nil, count: weights.count)
        var dataPointers = [UnsafePointer<UInt8>?](repeating: nil, count: weights.count)
        let dataLengths = weightDatas.map(\.count)

        func withWeightPathPointers(_ index: Int, _ nested: () -> Result) -> Result {
            guard index < weightPaths.count else {
                return nested()
            }
            return weightPaths[index].withCString { cPath in
                pathPointers[index] = cPath
                return withWeightPathPointers(index + 1, nested)
            }
        }

        func withWeightDataPointers(_ index: Int, _ nested: () -> Result) -> Result {
            guard index < weightDatas.count else {
                return nested()
            }
            if weightDatas[index].isEmpty {
                dataPointers[index] = nil
                return withWeightDataPointers(index + 1, nested)
            }
            return weightDatas[index].withUnsafeBytes { raw in
                dataPointers[index] = raw.bindMemory(to: UInt8.self).baseAddress
                return withWeightDataPointers(index + 1, nested)
            }
        }

        return milData.withUnsafeBytes { milRaw in
            let milBuffer = milRaw.bindMemory(to: UInt8.self)
            let milBase = milBuffer.baseAddress!

            return withWeightPathPointers(0) {
                withWeightDataPointers(0) {
                    pathPointers.withUnsafeMutableBufferPointer { pathBuf in
                        dataPointers.withUnsafeMutableBufferPointer { dataBuf in
                            dataLengths.withUnsafeBufferPointer { lenBuf in
                                inputSizes.withUnsafeBufferPointer { inputBuf in
                                    outputSizes.withUnsafeBufferPointer { outputBuf in
                                        body(
                                            milBase,
                                            milBuffer.count,
                                            weights.isEmpty ? nil : pathBuf.baseAddress,
                                            weights.isEmpty ? nil : dataBuf.baseAddress,
                                            weights.isEmpty ? nil : lenBuf.baseAddress,
                                            Int32(weights.count),
                                            inputSizes.isEmpty ? nil : inputBuf.baseAddress,
                                            Int32(inputSizes.count),
                                            outputSizes.isEmpty ? nil : outputBuf.baseAddress,
                                            Int32(outputSizes.count)
                                        )
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    private static func stringFromNullTerminatedUTF8(_ buffer: [CChar]) -> String {
        let prefix = buffer.prefix { $0 != 0 }
        return String(decoding: prefix.map(UInt8.init(bitPattern:)), as: UTF8.self)
    }

    private static func readHexId(from handle: OpaquePointer) throws(ANEError) -> String {
        var buffer = [CChar](repeating: 0, count: hexIdBufferLength)
        let ok = buffer.withUnsafeMutableBufferPointer { buf in
            ane_interop_get_hex_id(handle, buf.baseAddress, buf.count)
        }
        guard ok else { throw .compilationFailed }
        let hexId = stringFromNullTerminatedUTF8(buffer)
        guard !hexId.isEmpty else {
            throw .compilationFailed
        }
        return hexId
    }

    private static func ensureCompileCountDelta(
        startCount: Int,
        expectedDelta: Int,
        cleanupHandle: OpaquePointer? = nil
    ) throws(ANEError) {
        let endCount = CompileBudget.currentCount
        guard endCount - startCount == expectedDelta else {
            if let cleanupHandle {
                ane_interop_free(cleanupHandle)
            }
            throw .compilationFailed
        }
    }

    /// Standard compile (used at startup). Increments compile budget.
    public static func compileInitial(
        milText: String,
        weights: [(path: String, data: Data)],
        inputSizes: [Int],
        outputSizes: [Int]
    ) throws(ANEError) -> DeltaCompilationHandle {
        let milData = try validatedMILData(milText)
        let checkedWeights = try checkedWeightTuples(weights)
        let checkedInputs = try checkedByteSizes(inputSizes, label: "Input")
        let checkedOutputs = try checkedByteSizes(outputSizes, label: "Output")

        ane_interop_init()

        CompileGate.lock.lock()
        defer { CompileGate.lock.unlock() }

        let startCount = CompileBudget.currentCount
        guard !CompileBudget.isExhausted else {
            throw .compileBudgetExhausted
        }

        var attemptIndex = 0
        while true {
            var hexIdBuffer = [CChar](repeating: 0, count: hexIdBufferLength)
            let rawHandle = withPreparedBuffers(
                milData: milData,
                weights: checkedWeights,
                inputSizes: checkedInputs,
                outputSizes: checkedOutputs
            ) { milBase, milLen, pathBase, dataBase, lenBase, weightCount, inputBase, inputCount, outputBase, outputCount in
                hexIdBuffer.withUnsafeMutableBufferPointer { hexBuf in
                    ane_interop_compile_with_id(
                        milBase,
                        milLen,
                        pathBase,
                        dataBase,
                        lenBase,
                        weightCount,
                        inputCount,
                        inputBase,
                        outputCount,
                        outputBase,
                        hexBuf.baseAddress,
                        hexBuf.count
                    )
                }
            }

            if let rawHandle {
                try ensureCompileCountDelta(startCount: startCount, expectedDelta: 1, cleanupHandle: rawHandle)
                let hexId = stringFromNullTerminatedUTF8(hexIdBuffer)
                guard !hexId.isEmpty else {
                    ane_interop_free(rawHandle)
                    throw .compilationFailed
                }
                return DeltaCompilationHandle(handle: rawHandle, hexId: hexId)
            }

            let lastError = ane_interop_last_compile_error()
            guard RetryPolicy.shouldRetry(lastCompileError: lastError, attemptIndex: attemptIndex) else {
                throw mapInteropCompileError()
            }

            RetryPolicy.sleepAfterFailedAttempt(attemptIndex)
            attemptIndex += 1
        }
    }

    /// Delta reload using donor's compiled artifact. Does NOT increment compile budget.
    public static func reloadWeights(
        milText: String,
        weights: [(path: String, data: Data)],
        inputSizes: [Int],
        outputSizes: [Int],
        donorHexId: String
    ) throws(ANEError) -> DeltaCompilationHandle {
        guard !donorHexId.isEmpty else {
            throw .invalidArguments("donorHexId must be non-empty")
        }

        let milData = try validatedMILData(milText)
        let checkedWeights = try checkedWeightTuples(weights)
        let checkedInputs = try checkedByteSizes(inputSizes, label: "Input")
        let checkedOutputs = try checkedByteSizes(outputSizes, label: "Output")
        let startCount = CompileBudget.currentCount

        ane_interop_init()

        let rawHandle = withPreparedBuffers(
            milData: milData,
            weights: checkedWeights,
            inputSizes: checkedInputs,
            outputSizes: checkedOutputs
        ) { milBase, milLen, pathBase, dataBase, lenBase, weightCount, inputBase, inputCount, outputBase, outputCount in
            donorHexId.withCString { donorCString in
                ane_interop_delta_reload(
                    milBase,
                    milLen,
                    pathBase,
                    dataBase,
                    lenBase,
                    weightCount,
                    inputCount,
                    inputBase,
                    outputCount,
                    outputBase,
                    donorCString
                )
            }
        }

        guard let rawHandle else {
            throw mapInteropCompileError()
        }

        try ensureCompileCountDelta(startCount: startCount, expectedDelta: 0, cleanupHandle: rawHandle)
        let hexId = try readHexId(from: rawHandle)
        return DeltaCompilationHandle(handle: rawHandle, hexId: hexId)
    }

    /// Fastest path: unload, replace weight files, reload in-place.
    /// Does NOT increment compile budget.
    public static func fastReload(
        handle: borrowing DeltaCompilationHandle,
        weights: [(path: String, data: Data)]
    ) throws(ANEError) {
        let checkedWeights = try checkedWeightTuples(weights)
        let startCount = CompileBudget.currentCount

        let ok = checkedWeights.isEmpty
            ? ane_interop_fast_reload(handle.handle, nil, nil, nil, 0)
            : checkedWeights.map(\.path).withUnsafeTemporaryPathAndDataPointers(checkedWeights.map(\.data)) { pathBase, dataBase, lenBase in
                ane_interop_fast_reload(
                    handle.handle,
                    pathBase,
                    dataBase,
                    lenBase,
                    Int32(checkedWeights.count)
                )
            }

        guard ok else {
            throw .compilationFailed
        }
        try ensureCompileCountDelta(startCount: startCount, expectedDelta: 0)
    }
}

private extension Array where Element == String {
    func withUnsafeTemporaryPathAndDataPointers<Result>(
        _ datas: [Data],
        _ body: (
            UnsafeMutablePointer<UnsafePointer<CChar>?>,
            UnsafeMutablePointer<UnsafePointer<UInt8>?>,
            UnsafePointer<Int>
        ) -> Result
    ) -> Result {
        precondition(count == datas.count)

        var pathPointers = [UnsafePointer<CChar>?](repeating: nil, count: count)
        var dataPointers = [UnsafePointer<UInt8>?](repeating: nil, count: count)
        let lengths = datas.map(\.count)

        func withPaths(_ index: Int, _ nested: () -> Result) -> Result {
            guard index < count else { return nested() }
            return self[index].withCString { cPath in
                pathPointers[index] = cPath
                return withPaths(index + 1, nested)
            }
        }

        func withDatas(_ index: Int, _ nested: () -> Result) -> Result {
            guard index < datas.count else { return nested() }
            if datas[index].isEmpty {
                dataPointers[index] = nil
                return withDatas(index + 1, nested)
            }
            return datas[index].withUnsafeBytes { raw in
                dataPointers[index] = raw.bindMemory(to: UInt8.self).baseAddress
                return withDatas(index + 1, nested)
            }
        }

        return withPaths(0) {
            withDatas(0) {
                pathPointers.withUnsafeMutableBufferPointer { pathBuf in
                    dataPointers.withUnsafeMutableBufferPointer { dataBuf in
                        lengths.withUnsafeBufferPointer { lenBuf in
                            body(pathBuf.baseAddress!, dataBuf.baseAddress!, lenBuf.baseAddress!)
                        }
                    }
                }
            }
        }
    }
}
