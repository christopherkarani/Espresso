import Foundation
import ANEInterop
import IOSurface

public struct ANEKernel: ~Copyable {
    private enum CompileGate {
        static let lock = NSLock()
    }

    private let handle: OpaquePointer

    @inline(__always)
    private static func checkedSurfaceIndex(_ index: Int) throws(ANEError) -> Int32 {
        guard index >= 0 && index <= Int(Int32.max) else {
            throw .invalidSurfaceIndex(index)
        }
        return Int32(index)
    }

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

    /// Compile a MIL program with optional weight blobs into an ANE kernel.
    ///
    /// - Parameters:
    ///   - milText: MIL program text (will be UTF-8 encoded)
    ///   - weights: Array of (path, data) pairs. Paths follow "@model_path/weights/<name>.bin" convention.
    ///   - inputSizes: Byte sizes for each input IOSurface (typically 1 element for single-input kernels)
    ///   - outputSizes: Byte sizes for each output IOSurface (typically 1 element)
    ///   - checkBudget: If true, checks CompileBudget before compiling. Default true.
    ///
    /// - Throws: `ANEError.compileBudgetExhausted` if budget exceeded (and checkBudget is true).
    ///           `ANEError.compilationFailed` if ANE compilation returns NULL.
    public init(
        milText: String,
        weights: [(path: String, data: Data)],
        inputSizes: [Int],
        outputSizes: [Int],
        checkBudget: Bool = true
    ) throws(ANEError) {
        guard weights.count <= Int(Int32.max),
              inputSizes.count <= Int(Int32.max),
              outputSizes.count <= Int(Int32.max),
              inputSizes.allSatisfy({ $0 >= 0 }),
              outputSizes.allSatisfy({ $0 >= 0 }) else {
            throw .invalidArguments("Counts must fit Int32 and byte sizes must be non-negative")
        }

        guard let milData = milText.data(using: .utf8), !milData.isEmpty else {
            throw .invalidArguments("MIL text must be valid, non-empty UTF-8")
        }

        let weightPaths = weights.map(\.path)
        let weightDatas = weights.map(\.data)
        var pathPointers = [UnsafePointer<CChar>?](repeating: nil, count: weights.count)
        var dataPointers = [UnsafePointer<UInt8>?](repeating: nil, count: weights.count)
        let dataLengths = weightDatas.map(\.count)

        func compileHandle() -> OpaquePointer? {
            ane_interop_init()

            return milData.withUnsafeBytes { milRaw in
                let milBuffer = milRaw.bindMemory(to: UInt8.self)
                guard let milBase = milBuffer.baseAddress else {
                    return nil
                }

                func withWeightPathPointers<R>(_ index: Int, _ body: () -> R) -> R {
                    guard index < weightPaths.count else {
                        return body()
                    }
                    return weightPaths[index].withCString { cPath in
                        pathPointers[index] = cPath
                        return withWeightPathPointers(index + 1, body)
                    }
                }

                func withWeightDataPointers<R>(_ index: Int, _ body: () -> R) -> R {
                    guard index < weightDatas.count else {
                        return body()
                    }
                    if weightDatas[index].isEmpty {
                        dataPointers[index] = nil
                        return withWeightDataPointers(index + 1, body)
                    }
                    return weightDatas[index].withUnsafeBytes { raw in
                        dataPointers[index] = raw.bindMemory(to: UInt8.self).baseAddress
                        return withWeightDataPointers(index + 1, body)
                    }
                }

                return withWeightPathPointers(0) {
                    withWeightDataPointers(0) {
                        pathPointers.withUnsafeMutableBufferPointer { pathBuf in
                            dataPointers.withUnsafeMutableBufferPointer { dataBuf in
                                dataLengths.withUnsafeBufferPointer { lenBuf in
                                    inputSizes.withUnsafeBufferPointer { inputBuf in
                                        outputSizes.withUnsafeBufferPointer { outputBuf in
                                            ane_interop_compile(
                                                milBase,
                                                milBuffer.count,
                                                weights.isEmpty ? nil : pathBuf.baseAddress,
                                                weights.isEmpty ? nil : dataBuf.baseAddress,
                                                weights.isEmpty ? nil : lenBuf.baseAddress,
                                                Int32(weights.count),
                                                Int32(inputSizes.count),
                                                inputSizes.isEmpty ? nil : inputBuf.baseAddress,
                                                Int32(outputSizes.count),
                                                outputSizes.isEmpty ? nil : outputBuf.baseAddress
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

        let rawHandle: OpaquePointer?
        if checkBudget {
            CompileGate.lock.lock()
            defer { CompileGate.lock.unlock() }
            if CompileBudget.isExhausted {
                throw .compileBudgetExhausted
            }
            rawHandle = compileHandle()
        } else {
            rawHandle = compileHandle()
        }

        guard let rawHandle else {
            throw Self.mapInteropCompileError()
        }
        self.handle = rawHandle
    }

    /// Convenience: single-input, single-output kernel (most common case).
    public init(
        milText: String,
        weights: [(path: String, data: Data)],
        inputBytes: Int,
        outputBytes: Int,
        checkBudget: Bool = true
    ) throws(ANEError) {
        try self.init(
            milText: milText,
            weights: weights,
            inputSizes: [inputBytes],
            outputSizes: [outputBytes],
            checkBudget: checkBudget
        )
    }

    /// Run the compiled kernel on ANE.
    /// Input data must be written to inputSurface before calling.
    /// Output data is available on outputSurface after return.
    public func eval() throws(ANEError) {
        guard ane_interop_eval(handle) else {
            throw .evaluationFailed
        }
    }

    /// Access input IOSurface (retained; safe to hold independently of kernel lifetime).
    public func inputSurface(at index: Int) throws(ANEError) -> IOSurfaceRef {
        let checkedIndex = try Self.checkedSurfaceIndex(index)
        guard let surface = ane_interop_copy_input(handle, checkedIndex) else {
            throw .inputSurfaceUnavailable(index)
        }
        return surface
    }

    /// Access output IOSurface (retained; safe to hold independently of kernel lifetime).
    public func outputSurface(at index: Int) throws(ANEError) -> IOSurfaceRef {
        let checkedIndex = try Self.checkedSurfaceIndex(index)
        guard let surface = ane_interop_copy_output(handle, checkedIndex) else {
            throw .outputSurfaceUnavailable(index)
        }
        return surface
    }

    deinit {
        ane_interop_free(handle)
    }
}
