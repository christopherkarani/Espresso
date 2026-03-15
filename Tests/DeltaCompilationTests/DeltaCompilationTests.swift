import Darwin
import Foundation
import Testing
import XCTest
@testable import DeltaCompilation

private let minimalMIL = """
program(1.3)
[buildInfo = dict<string, string>({{"coremlc-component-MIL", "3510.2.1"}, {"coremlc-version", "3505.4.1"}, {"coremltools-component-milinternal", ""}, {"coremltools-version", "9.0"}})]
{
    func main<ios18>(tensor<fp32, [1, 1, 1, 1]> x) {
        string to16 = const()[name=string("to16"), val=string("fp16")];
        tensor<fp16, [1,1,1,1]> x16 = cast(dtype=to16, x=x)[name=string("x16")];
        string to32 = const()[name=string("to32"), val=string("fp32")];
        tensor<fp32, [1,1,1,1]> y = cast(dtype=to32, x=x16)[name=string("y")];
    } -> (y);
}
"""

private let minimalInputSizes = [MemoryLayout<Float>.stride]
private let minimalOutputSizes = [MemoryLayout<Float>.stride]

private func aneIsAvailable() -> Bool {
    let handle = dlopen(
        "/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine",
        RTLD_NOW
    )
    if handle == nil {
        return false
    }
    dlclose(handle)

    let requiredClasses = [
        "_ANEInMemoryModelDescriptor",
        "_ANEInMemoryModel",
        "_ANERequest",
        "_ANEIOSurfaceObject",
    ]
    for c in requiredClasses where NSClassFromString(c) == nil {
        return false
    }
    return true
}

private func shouldRunANEHardwareTests() -> Bool {
    ProcessInfo.processInfo.environment["ANE_HARDWARE_TESTS"] == "1" && aneIsAvailable()
}

private func borrowedHexId(_ handle: borrowing DeltaCompilationHandle) -> String {
    handle.hexId
}

@Test func deltaCompilationHandleCompilesInBorrowingContexts() {
    let borrow: (borrowing DeltaCompilationHandle) -> String = { borrowedHexId($0) }
    _ = borrow
    let sampleHexId = "abc123"
    #expect(sampleHexId == "abc123")
}

@Test func compileInitialRejectsEmptyMIL() {
    do {
        _ = try DeltaCompilation.compileInitial(
            milText: "",
            weights: [],
            inputSizes: minimalInputSizes,
            outputSizes: minimalOutputSizes
        )
        Issue.record("Expected compileInitial to reject empty MIL")
    } catch {
        #expect(String(describing: error).contains("MIL text"))
    }
}

@Test func reloadWeightsRejectsEmptyDonorHexId() {
    do {
        _ = try DeltaCompilation.reloadWeights(
            milText: minimalMIL,
            weights: [],
            inputSizes: minimalInputSizes,
            outputSizes: minimalOutputSizes,
            donorHexId: ""
        )
        Issue.record("Expected reloadWeights to reject an empty donor hex id")
    } catch {
        #expect(String(describing: error).contains("donorHexId"))
    }
}

@Test func compileInitialReturnsHandleWithNonEmptyHexId() throws {
    guard shouldRunANEHardwareTests() else { return }
    let handle = try DeltaCompilation.compileInitial(
        milText: minimalMIL,
        weights: [],
        inputSizes: minimalInputSizes,
        outputSizes: minimalOutputSizes
    )
    #expect(!borrowedHexId(handle).isEmpty)
}

@Test func reloadWeightsProducesAValidHandle() throws {
    guard shouldRunANEHardwareTests() else { return }
    let donor = try DeltaCompilation.compileInitial(
        milText: minimalMIL,
        weights: [],
        inputSizes: minimalInputSizes,
        outputSizes: minimalOutputSizes
    )
    let reloaded = try DeltaCompilation.reloadWeights(
        milText: minimalMIL,
        weights: [],
        inputSizes: minimalInputSizes,
        outputSizes: minimalOutputSizes,
        donorHexId: borrowedHexId(donor)
    )
    #expect(!borrowedHexId(reloaded).isEmpty)
}

@Test func fastReloadSucceedsForAnExistingHandle() throws {
    guard shouldRunANEHardwareTests() else { return }
    let handle = try DeltaCompilation.compileInitial(
        milText: minimalMIL,
        weights: [],
        inputSizes: minimalInputSizes,
        outputSizes: minimalOutputSizes
    )
    let initialHexId = borrowedHexId(handle)
    try DeltaCompilation.fastReload(handle: handle, weights: [])
    #expect(borrowedHexId(handle) == initialHexId)
}
