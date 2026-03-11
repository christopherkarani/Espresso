import XCTest
import Darwin
import ANEInterop
import ANETypes
import MILGenerator
@testable import ANERuntime

private let vcProbeChannels = 4
private let vcProbeSpatial = 8
private let vcProbeBytes = vcProbeChannels * vcProbeSpatial * MemoryLayout<UInt16>.stride

private func requireANEAvailable(file: StaticString = #filePath, line: UInt = #line) throws {
    let handle = dlopen(
        "/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine",
        RTLD_NOW
    )
    if handle == nil {
        throw XCTSkip("AppleNeuralEngine.framework unavailable", file: file, line: line)
    }
    dlclose(handle)
}

private func requireANEHardwareTests(file: StaticString = #filePath, line: UInt = #line) throws {
    guard ProcessInfo.processInfo.environment["ANE_HARDWARE_TESTS"] == "1" else {
        throw XCTSkip("Set ANE_HARDWARE_TESTS=1 to run ANE hardware tests", file: file, line: line)
    }
    try requireANEAvailable(file: file, line: line)
}

private func vcIdentityWeightBlob(channels: Int) -> Data {
    var weights = [Float](repeating: 0, count: channels * channels)
    for i in 0..<channels {
        weights[i * channels + i] = 1
    }
    return WeightBlob.build(from: weights, rows: channels, cols: channels)
}

private func vcIdentityKernel() throws -> ANEKernel {
    let mil = GenericMIL.conv(
        inCh: vcProbeChannels,
        outCh: vcProbeChannels,
        spatial: vcProbeSpatial
    )
    return try ANEKernel(
        milText: mil,
        weights: [(
            path: "@model_path/weights/weight.bin",
            data: vcIdentityWeightBlob(channels: vcProbeChannels)
        )],
        inputBytes: vcProbeBytes,
        outputBytes: vcProbeBytes
    )
}

// MARK: - Non-hardware tests (always run)

final class VirtualClientEvalProbeTests: XCTestCase {

    func test_runtime_has_virtual_client_returns_bool() throws {
        try requireANEAvailable()
        // Just verifying the capability check doesn't crash
        let result = ane_interop_runtime_has_virtual_client()
        // Result is platform-dependent; we just verify it returns cleanly
        _ = result
    }

    func test_runtime_has_shared_events_request_returns_bool() throws {
        try requireANEAvailable()
        let result = ane_interop_runtime_has_shared_events_request()
        _ = result
    }

    // MARK: - Hardware-gated tests

    func test_vc_probe_discovers_all_classes() throws {
        try requireANEHardwareTests()
        let kernel = try vcIdentityKernel()
        let probe = kernel.virtualClientProbe(skipEval: true)

        // Class discovery should work regardless of whether eval succeeds
        // These are informational — log findings for diagnostics
        print("VCProbe class discovery:")
        print("  hasVirtualClientClass: \(probe.hasVirtualClientClass)")
        print("  hasVirtualClientProperty: \(probe.hasVirtualClientProperty)")
        print("  hasSharedEventsClass: \(probe.hasSharedEventsClass)")
        print("  hasSharedWaitEventClass: \(probe.hasSharedWaitEventClass)")
        print("  hasSharedSignalEventClass: \(probe.hasSharedSignalEventClass)")
        print("  hasIOSurfaceSharedEventClass: \(probe.hasIOSurfaceSharedEventClass)")
        print("  hasDoEvaluateCompletionEvent: \(probe.hasDoEvaluateCompletionEvent)")
        print("  hasStandardEvaluate: \(probe.hasStandardEvaluate)")
        print("  hasMapIOSurfaces: \(probe.hasMapIOSurfaces)")
        print("  hasLoadModel: \(probe.hasLoadModel)")
        print("  hasRequestSharedEventsFactory: \(probe.hasRequestSharedEventsFactory)")
        print("  hasSetSharedEvents: \(probe.hasSetSharedEvents)")
        print("  hasSetCompletionHandler: \(probe.hasSetCompletionHandler)")

        // At minimum, the VirtualClient class should exist on Apple Silicon macOS
        XCTAssertTrue(
            probe.hasVirtualClientClass,
            "_ANEVirtualClient class should be present"
        )
    }

    func test_vc_probe_obtains_virtual_client_from_client() throws {
        try requireANEHardwareTests()
        let kernel = try vcIdentityKernel()
        let probe = kernel.virtualClientProbe(skipEval: true)

        print("VCProbe virtualClient acquisition:")
        print("  obtainedVirtualClient: \(probe.obtainedVirtualClient)")
        print("  stage: \(probe.stage)")

        // virtualClient property may return nil from the InMemoryModel's sharedConnection.
        // This is a probe finding — log it and record. The property exists but the
        // _ANEClient obtained via sharedConnection may not support virtualClient.
        if !probe.obtainedVirtualClient {
            print("  NOTE: virtualClient returned nil — property exists but returns nil from InMemoryModel's _ANEClient")
            print("  This suggests virtualClient requires a different _ANEClient instantiation path")
        }
        // Assert the property at least exists (class discovery)
        XCTAssertTrue(probe.hasVirtualClientProperty,
                      "_ANEClient should have virtualClient property")
    }

    func test_vc_probe_builds_shared_events_container() throws {
        try requireANEHardwareTests()
        let kernel = try vcIdentityKernel()
        let probe = kernel.virtualClientProbe(
            useSharedEvents: true,
            useWaitEvent: true,
            skipEval: true
        )

        print("VCProbe shared events construction:")
        print("  builtIOSurfaceSharedEvent: \(probe.builtIOSurfaceSharedEvent)")
        print("  builtWaitEvent: \(probe.builtWaitEvent)")
        print("  builtSignalEvent: \(probe.builtSignalEvent)")
        print("  builtSharedEventsContainer: \(probe.builtSharedEventsContainer)")
        print("  stage: \(probe.stage)")

        // IOSurfaceSharedEvent +new returns nil (golden output confirmed).
        // Factory methods are required instead. Log the finding.
        if probe.hasIOSurfaceSharedEventClass && !probe.builtIOSurfaceSharedEvent {
            print("  NOTE: IOSurfaceSharedEvent +new returns nil — must use factory method (e.g., from MTLSharedEvent)")
        }
        // Just verify class was found
        XCTAssertTrue(probe.hasIOSurfaceSharedEventClass,
                      "IOSurfaceSharedEvent class should be present")
    }

    func test_vc_probe_standard_eval_on_virtual_client() throws {
        try requireANEHardwareTests()
        let kernel = try vcIdentityKernel()

        // Write known input
        let input: [Float] = [1, 2, 3, 4, 5, 6, 7, 8,
                              9, 10, 11, 12, 13, 14, 15, 16,
                              17, 18, 19, 20, 21, 22, 23, 24,
                              25, 26, 27, 28, 29, 30, 31, 32]
        let inputSurface = try kernel.inputSurface(at: 0)
        XCTAssertTrue(
            ane_interop_io_write_fp16(inputSurface, input, Int32(vcProbeChannels), Int32(vcProbeSpatial))
        )

        let probe = kernel.virtualClientProbe()

        print("VCProbe standard eval:")
        print("  standardEvalSucceeded: \(probe.standardEvalSucceeded)")
        print("  stage: \(probe.stage)")

        if probe.hasStandardEvaluate && probe.obtainedVirtualClient {
            // If eval succeeded, verify output correctness (identity kernel)
            if probe.standardEvalSucceeded {
                var output = [Float](repeating: 0, count: vcProbeChannels * vcProbeSpatial)
                let outputSurface = try kernel.outputSurface(at: 0)
                XCTAssertTrue(
                    ane_interop_io_read_fp16(outputSurface, 0, &output, Int32(vcProbeChannels), Int32(vcProbeSpatial))
                )
                for i in 0..<input.count {
                    XCTAssertEqual(
                        output[i], input[i],
                        accuracy: 0.1,
                        "Identity kernel output[\(i)] should match input"
                    )
                }
            }
        }
    }

    func test_vc_probe_completion_event_eval_nil() throws {
        try requireANEHardwareTests()
        let kernel = try vcIdentityKernel()
        let probe = kernel.virtualClientProbe(useCompletionEvent: true)

        print("VCProbe completionEvent eval (nil event):")
        print("  completionEventEvalSucceeded: \(probe.completionEventEvalSucceeded)")
        print("  stage: \(probe.stage)")
    }

    func test_vc_probe_completion_event_eval_with_shared_event() throws {
        try requireANEHardwareTests()
        let kernel = try vcIdentityKernel()
        let probe = kernel.virtualClientProbe(
            useCompletionEvent: true,
            useSharedEvents: true
        )

        print("VCProbe completionEvent eval (with IOSurfaceSharedEvent):")
        print("  builtIOSurfaceSharedEvent: \(probe.builtIOSurfaceSharedEvent)")
        print("  completionEventEvalSucceeded: \(probe.completionEventEvalSucceeded)")
        print("  stage: \(probe.stage)")
    }

    func test_vc_probe_completion_handler_on_request() throws {
        try requireANEHardwareTests()
        let kernel = try vcIdentityKernel()
        let probe = kernel.virtualClientProbe(useCompletionHandler: true)

        print("VCProbe completionHandler:")
        print("  completionHandlerFired: \(probe.completionHandlerFired)")
        print("  stage: \(probe.stage)")
    }

    func test_vc_probe_with_map_surfaces() throws {
        try requireANEHardwareTests()
        let kernel = try vcIdentityKernel()
        let probe = kernel.virtualClientProbe(mapSurfaces: true)

        print("VCProbe with mapSurfaces:")
        print("  mappedSurfaces: \(probe.mappedSurfaces)")
        print("  standardEvalSucceeded: \(probe.standardEvalSucceeded)")
        print("  stage: \(probe.stage)")
    }

    func test_vc_probe_with_load_on_virtual_client() throws {
        try requireANEHardwareTests()
        let kernel = try vcIdentityKernel()
        let probe = kernel.virtualClientProbe(loadOnVirtualClient: true)

        print("VCProbe with loadOnVirtualClient:")
        print("  loadedOnVirtualClient: \(probe.loadedOnVirtualClient)")
        print("  standardEvalSucceeded: \(probe.standardEvalSucceeded)")
        print("  stage: \(probe.stage)")
    }

    func test_vc_probe_full_pipeline() throws {
        try requireANEHardwareTests()
        let kernel = try vcIdentityKernel()

        // Write known input
        let input: [Float] = (0..<(vcProbeChannels * vcProbeSpatial)).map { Float($0) + 1 }
        let inputSurface = try kernel.inputSurface(at: 0)
        XCTAssertTrue(
            ane_interop_io_write_fp16(inputSurface, input, Int32(vcProbeChannels), Int32(vcProbeSpatial))
        )

        let probe = kernel.virtualClientProbe(
            useCompletionEvent: true,
            useCompletionHandler: true,
            useSharedEvents: true,
            useWaitEvent: true,
            mapSurfaces: true,
            loadOnVirtualClient: true
        )

        print("VCProbe full pipeline:")
        print("  obtainedVirtualClient: \(probe.obtainedVirtualClient)")
        print("  builtIOSurfaceSharedEvent: \(probe.builtIOSurfaceSharedEvent)")
        print("  builtWaitEvent: \(probe.builtWaitEvent)")
        print("  builtSignalEvent: \(probe.builtSignalEvent)")
        print("  builtSharedEventsContainer: \(probe.builtSharedEventsContainer)")
        print("  builtRequest: \(probe.builtRequest)")
        print("  mappedSurfaces: \(probe.mappedSurfaces)")
        print("  loadedOnVirtualClient: \(probe.loadedOnVirtualClient)")
        print("  standardEvalSucceeded: \(probe.standardEvalSucceeded)")
        print("  completionEventEvalSucceeded: \(probe.completionEventEvalSucceeded)")
        print("  completionHandlerFired: \(probe.completionHandlerFired)")
        print("  stage: \(probe.stage)")
    }

    // MARK: - Direct VirtualClient instantiation tests

    func test_vc_probe_direct_shared_connection() throws {
        try requireANEHardwareTests()
        let kernel = try vcIdentityKernel()
        let probe = kernel.virtualClientProbe(
            skipEval: true,
            useDirectInstantiation: true
        )

        print("VCProbe direct instantiation (skipEval):")
        print("  triedPropertyOnClient: \(probe.triedPropertyOnClient)")
        print("  triedDirectSharedConnection: \(probe.triedDirectSharedConnection)")
        print("  triedInitWithSingletonAccess: \(probe.triedInitWithSingletonAccess)")
        print("  triedNew: \(probe.triedNew)")
        print("  directConnectSucceeded: \(probe.directConnectSucceeded)")
        print("  obtainedVirtualClient: \(probe.obtainedVirtualClient)")
        print("  stage: \(probe.stage)")

        // At minimum we should try the property path and then fallbacks
        XCTAssertTrue(probe.triedPropertyOnClient || probe.triedDirectSharedConnection,
                      "Should try at least one acquisition path")
    }

    func test_vc_probe_direct_instantiation_with_eval() throws {
        try requireANEHardwareTests()
        let kernel = try vcIdentityKernel()

        // Write known input
        let input: [Float] = [1, 2, 3, 4, 5, 6, 7, 8,
                              9, 10, 11, 12, 13, 14, 15, 16,
                              17, 18, 19, 20, 21, 22, 23, 24,
                              25, 26, 27, 28, 29, 30, 31, 32]
        let inputSurface = try kernel.inputSurface(at: 0)
        XCTAssertTrue(
            ane_interop_io_write_fp16(inputSurface, input, Int32(vcProbeChannels), Int32(vcProbeSpatial))
        )

        let probe = kernel.virtualClientProbe(
            useDirectInstantiation: true
        )

        print("VCProbe direct instantiation (with eval):")
        print("  triedPropertyOnClient: \(probe.triedPropertyOnClient)")
        print("  triedDirectSharedConnection: \(probe.triedDirectSharedConnection)")
        print("  triedInitWithSingletonAccess: \(probe.triedInitWithSingletonAccess)")
        print("  triedNew: \(probe.triedNew)")
        print("  directConnectSucceeded: \(probe.directConnectSucceeded)")
        print("  obtainedVirtualClient: \(probe.obtainedVirtualClient)")
        print("  standardEvalSucceeded: \(probe.standardEvalSucceeded)")
        print("  stage: \(probe.stage)")

        if probe.obtainedVirtualClient && probe.standardEvalSucceeded {
            var output = [Float](repeating: 0, count: vcProbeChannels * vcProbeSpatial)
            let outputSurface = try kernel.outputSurface(at: 0)
            XCTAssertTrue(
                ane_interop_io_read_fp16(outputSurface, 0, &output, Int32(vcProbeChannels), Int32(vcProbeSpatial))
            )
            for i in 0..<input.count {
                XCTAssertEqual(
                    output[i], input[i],
                    accuracy: 0.1,
                    "Identity kernel output[\(i)] should match input"
                )
            }
            print("  OUTPUT VERIFIED: identity kernel correct via direct VirtualClient")
        }
    }

    func test_vc_probe_direct_full_pipeline() throws {
        try requireANEHardwareTests()
        let kernel = try vcIdentityKernel()

        let input: [Float] = (0..<(vcProbeChannels * vcProbeSpatial)).map { Float($0) + 1 }
        let inputSurface = try kernel.inputSurface(at: 0)
        XCTAssertTrue(
            ane_interop_io_write_fp16(inputSurface, input, Int32(vcProbeChannels), Int32(vcProbeSpatial))
        )

        let probe = kernel.virtualClientProbe(
            useCompletionEvent: true,
            useCompletionHandler: true,
            useSharedEvents: true,
            useWaitEvent: true,
            mapSurfaces: true,
            loadOnVirtualClient: true,
            useDirectInstantiation: true
        )

        print("VCProbe direct full pipeline:")
        print("  triedPropertyOnClient: \(probe.triedPropertyOnClient)")
        print("  triedDirectSharedConnection: \(probe.triedDirectSharedConnection)")
        print("  triedInitWithSingletonAccess: \(probe.triedInitWithSingletonAccess)")
        print("  triedNew: \(probe.triedNew)")
        print("  directConnectSucceeded: \(probe.directConnectSucceeded)")
        print("  obtainedVirtualClient: \(probe.obtainedVirtualClient)")
        print("  builtIOSurfaceSharedEvent: \(probe.builtIOSurfaceSharedEvent)")
        print("  builtSharedEventsContainer: \(probe.builtSharedEventsContainer)")
        print("  mappedSurfaces: \(probe.mappedSurfaces)")
        print("  loadedOnVirtualClient: \(probe.loadedOnVirtualClient)")
        print("  standardEvalSucceeded: \(probe.standardEvalSucceeded)")
        print("  completionEventEvalSucceeded: \(probe.completionEventEvalSucceeded)")
        print("  completionHandlerFired: \(probe.completionHandlerFired)")
        print("  stage: \(probe.stage)")
    }

    // MARK: - Code Signing Identity Probe

    func test_vc_probe_code_signing_identity() throws {
        try requireANEHardwareTests()
        let probe = ANEKernel.codeSigningProbe()

        print("VCProbe code signing:")
        print("  hasGetCodeSigningIdentity: \(probe.hasGetCodeSigningIdentity)")
        print("  hasSetCodeSigningIdentity: \(probe.hasSetCodeSigningIdentity)")
        print("  gotIdentityString: \(probe.gotIdentityString)")
        print("  identityString: '\(probe.identityString)'")
        print("  setIdentityBeforeInstantiation: \(probe.setIdentityBeforeInstantiation)")
        print("  instantiationSucceededAfterSet: \(probe.instantiationSucceededAfterSet)")

        XCTAssertTrue(probe.hasGetCodeSigningIdentity,
                      "_ANEVirtualClient should have +getCodeSigningIdentity")
    }

    // MARK: - Standard Eval CompletionHandler Probe

    func test_vc_probe_standard_completion_handler() throws {
        try requireANEHardwareTests()
        let kernel = try vcIdentityKernel()

        let input: [Float] = [1, 2, 3, 4, 5, 6, 7, 8,
                              9, 10, 11, 12, 13, 14, 15, 16,
                              17, 18, 19, 20, 21, 22, 23, 24,
                              25, 26, 27, 28, 29, 30, 31, 32]
        let inputSurface = try kernel.inputSurface(at: 0)
        XCTAssertTrue(
            ane_interop_io_write_fp16(inputSurface, input, Int32(vcProbeChannels), Int32(vcProbeSpatial))
        )

        let probe = kernel.standardCompletionProbe()

        print("Standard completion handler probe:")
        print("  requestHasCompletionHandler: \(probe.requestHasCompletionHandler)")
        print("  completionHandlerSet: \(probe.completionHandlerSet)")
        print("  evalSucceeded: \(probe.evalSucceeded)")
        print("  completionHandlerFired: \(probe.completionHandlerFired)")
        print("  evalTimeMS: \(probe.evalTimeMS)")

        XCTAssertTrue(probe.requestHasCompletionHandler,
                      "_ANERequest should support setCompletionHandler:")
        // Eval may fail on some hosts (known instability), but the handler should still fire
        if !probe.evalSucceeded {
            print("  NOTE: eval failed (known host instability) but handler fired=\(probe.completionHandlerFired)")
        }
    }

    func test_vc_probe_standard_completion_handler_with_metal_shared_event() throws {
        throw XCTSkip(
            "Avenue 2 abandoned: attaching Metal-backed shared events via setSharedEvents: hangs the standard eval path on hardware before completion or event advancement."
        )
    }
}
