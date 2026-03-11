import XCTest
import Darwin
import ANEInterop
import ANETypes
import MILGenerator
@testable import ANERuntime

private let chainingIdentityChannels = 4
private let chainingIdentitySpatial = 8
private let chainingIdentityBytes = chainingIdentityChannels * chainingIdentitySpatial * MemoryLayout<UInt16>.stride

private struct ChainingProbeProcessResult {
    let status: Int32
    let stdout: String
    let stderr: String
    let timedOut: Bool
}

private func requireANEAvailableForChaining(file: StaticString = #filePath, line: UInt = #line) throws {
    let handle = dlopen(
        "/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine",
        RTLD_NOW
    )
    if handle == nil {
        throw XCTSkip("AppleNeuralEngine.framework unavailable", file: file, line: line)
    }
    dlclose(handle)
}

private func requireANEHardwareTestsEnabledForChaining(file: StaticString = #filePath, line: UInt = #line) throws {
    guard ProcessInfo.processInfo.environment["ANE_HARDWARE_TESTS"] == "1" else {
        throw XCTSkip("Set ANE_HARDWARE_TESTS=1 to run ANE hardware tests", file: file, line: line)
    }
    try requireANEAvailableForChaining(file: file, line: line)
}

private func withEnvironmentValue<T>(
    key: String,
    value: String?,
    file: StaticString = #filePath,
    line: UInt = #line,
    _ body: () throws -> T
) rethrows -> T {
    let previous = getenv(key)
    let previousValue = previous.map { String(cString: $0) }
    if let value {
        XCTAssertEqual(setenv(key, value, 1), 0, file: file, line: line)
    } else {
        XCTAssertEqual(unsetenv(key), 0, file: file, line: line)
    }
    defer {
        if let previousValue {
            XCTAssertEqual(setenv(key, previousValue, 1), 0, file: file, line: line)
        } else {
            XCTAssertEqual(unsetenv(key), 0, file: file, line: line)
        }
    }
    return try body()
}

private func chainingIdentityWeightBlob(channels: Int) -> Data {
    var weights = [Float](repeating: 0, count: channels * channels)
    for i in 0..<channels {
        weights[i * channels + i] = 1
    }
    return WeightBlob.build(from: weights, rows: channels, cols: channels)
}

private func chainingIdentityKernel() throws -> ANEKernel {
    let mil = GenericMIL.conv(inCh: chainingIdentityChannels, outCh: chainingIdentityChannels, spatial: chainingIdentitySpatial)
    return try ANEKernel(
        milText: mil,
        weights: [(path: "@model_path/weights/weight.bin", data: chainingIdentityWeightBlob(channels: chainingIdentityChannels))],
        inputBytes: chainingIdentityBytes,
        outputBytes: chainingIdentityBytes
    )
}

private func repoRootURL() -> URL {
    URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .deletingLastPathComponent()
}

private func espressoBenchExecutableURL() throws -> URL {
    let repoRoot = repoRootURL()
    let candidates = [
        repoRoot.appendingPathComponent(".build/debug/espresso-bench"),
        repoRoot.appendingPathComponent(".build/arm64-apple-macosx/debug/espresso-bench"),
    ]
    if let found = candidates.first(where: { FileManager.default.isExecutableFile(atPath: $0.path) }) {
        return found
    }
    throw XCTSkip("Swift espresso-bench not found/executable under \(repoRoot.path)")
}

private func runProbeProcess(
    executableURL: URL,
    arguments: [String],
    currentDirectoryURL: URL,
    timeoutSeconds: TimeInterval
) throws -> ChainingProbeProcessResult {
    let stdoutPipe = Pipe()
    let stderrPipe = Pipe()

    let process = Process()
    process.executableURL = executableURL
    process.arguments = arguments
    process.currentDirectoryURL = currentDirectoryURL
    var environment = ProcessInfo.processInfo.environment
    environment["ESPRESSO_BENCH_SEED"] = "1"
    environment["ANE_COMPILE_CACHE_POLICY"] = "preferCached"
    process.environment = environment
    process.standardOutput = stdoutPipe
    process.standardError = stderrPipe

    let exited = DispatchSemaphore(value: 0)
    process.terminationHandler = { _ in exited.signal() }

    try process.run()

    let timedOut = exited.wait(timeout: .now() + timeoutSeconds) != .success
    if timedOut {
        process.terminate()
        _ = exited.wait(timeout: .now() + 5)
    }

    let stdoutData = stdoutPipe.fileHandleForReading.readDataToEndOfFile()
    let stderrData = stderrPipe.fileHandleForReading.readDataToEndOfFile()
    return ChainingProbeProcessResult(
        status: process.terminationStatus,
        stdout: String(decoding: stdoutData, as: UTF8.self),
        stderr: String(decoding: stderrData, as: UTF8.self),
        timedOut: timedOut
    )
}

final class DecodeChainingInteropTests: XCTestCase {
    func test_chaining_probe_identity_kernel_returns_controlled_status() throws {
        try requireANEHardwareTestsEnabledForChaining()

        let kernel = try chainingIdentityKernel()
        let probe = withEnvironmentValue(
            key: "ANE_INTEROP_CHAINING_PROBE_STATS_SURFACE",
            value: "null"
        ) {
            kernel.chainingProbe()
        }

        if probe.hasChainingRequestClass && probe.hasPrepareSelector {
            XCTAssertNotEqual(
                probe.stage,
                Int(ANE_INTEROP_CHAINING_STAGE_UNAVAILABLE.rawValue),
                "A supported runtime should advance the probe beyond unavailable"
            )
        }

        if probe.hasOutputSetsFactory {
            XCTAssertFalse(
                probe.builtOutputSet,
                "safe in-process probe must stay on the NULL stats-surface path"
            )
            XCTAssertEqual(
                probe.stage,
                Int(ANE_INTEROP_CHAINING_STAGE_OUTPUT_SETS_BUILD_FAILED.rawValue),
                "safe in-process probe should stop at output-set construction failure"
            )
        }

        if probe.prepared {
            XCTAssertTrue(probe.builtRequest)
            XCTAssertEqual(probe.stage, Int(ANE_INTEROP_CHAINING_STAGE_PREPARE_SUCCEEDED.rawValue))
        }
    }

    func test_chaining_probe_can_build_real_stats_output_set_without_prepare() throws {
        try requireANEHardwareTestsEnabledForChaining()

        let kernel = try chainingIdentityKernel()
        let probe = kernel.chainingProbe(
            useRealStatsSurface: true,
            skipPrepare: true,
            validateRequest: true
        )

        guard probe.hasChainingRequestClass, probe.hasOutputSetsFactory else {
            throw XCTSkip("Chaining output-set builder unavailable on this host")
        }

        XCTAssertTrue(
            probe.usedRealStatsSurface,
            "object-only probe should record that it used a real stats surface"
        )
        XCTAssertTrue(
            probe.builtOutputSet,
            "real stats-surface object-only probe should build _ANEIOSurfaceOutputSets"
        )
        XCTAssertTrue(
            probe.builtRequest,
            "object-only probe should still build the chaining request"
        )
        XCTAssertTrue(
            probe.requestValidated,
            "object-only probe should validate the request when asked"
        )
        XCTAssertEqual(
            probe.stage,
            Int(ANE_INTEROP_CHAINING_STAGE_PREPARE_SKIPPED.rawValue),
            "object-only probe should skip prepare inside xctest"
        )
    }

    func test_chaining_probe_buffers_ready_returns_controlled_status() throws {
        try requireANEHardwareTestsEnabledForChaining()

        let kernel = try chainingIdentityKernel()
        let probe = kernel.chainingProbe(
            useRealStatsSurface: true,
            skipPrepare: true,
            validateRequest: true,
            callBuffersReady: true
        )

        guard probe.hasChainingRequestClass, probe.hasInputBuffersReadyClass else {
            throw XCTSkip("Chaining buffersReady probe unavailable on this host")
        }

        XCTAssertTrue(probe.builtInputBuffersReady)
        XCTAssertTrue(probe.requestValidated)
        XCTAssertTrue(
            probe.calledBuffersReady,
            "buffersReady probe should attempt the client sequencing call"
        )
        XCTAssertFalse(probe.prepared)
        XCTAssertTrue(
            probe.stage == Int(ANE_INTEROP_CHAINING_STAGE_INPUT_BUFFERS_READY_CALL_FAILED.rawValue) ||
            probe.stage == Int(ANE_INTEROP_CHAINING_STAGE_INPUT_BUFFERS_READY_CALL_SUCCEEDED.rawValue),
            "buffersReady probe must report a controlled client-call result"
        )
        XCTAssertEqual(
            probe.buffersReadySucceeded,
            probe.stage == Int(ANE_INTEROP_CHAINING_STAGE_INPUT_BUFFERS_READY_CALL_SUCCEEDED.rawValue)
        )
    }

    func test_chaining_probe_enqueue_sets_returns_controlled_status() throws {
        try requireANEHardwareTestsEnabledForChaining()

        let kernel = try chainingIdentityKernel()
        let probe = kernel.chainingProbe(
            useRealStatsSurface: true,
            skipPrepare: true,
            validateRequest: true,
            callEnqueueSets: true
        )

        guard probe.hasChainingRequestClass, probe.hasOutputSetEnqueueClass else {
            throw XCTSkip("Chaining enqueueSets probe unavailable on this host")
        }

        XCTAssertTrue(probe.builtOutputSet)
        XCTAssertTrue(probe.builtOutputSetEnqueue)
        XCTAssertTrue(probe.requestValidated)
        XCTAssertTrue(
            probe.calledEnqueueSets,
            "enqueueSets probe should attempt the client sequencing call"
        )
        XCTAssertFalse(probe.prepared)
        XCTAssertTrue(
            probe.stage == Int(ANE_INTEROP_CHAINING_STAGE_ENQUEUE_SETS_CALL_FAILED.rawValue) ||
            probe.stage == Int(ANE_INTEROP_CHAINING_STAGE_ENQUEUE_SETS_CALL_SUCCEEDED.rawValue),
            "enqueueSets probe must report a controlled client-call result"
        )
        XCTAssertEqual(
            probe.enqueueSetsSucceeded,
            probe.stage == Int(ANE_INTEROP_CHAINING_STAGE_ENQUEUE_SETS_CALL_SUCCEEDED.rawValue)
        )
    }

    func test_chaining_probe_enqueue_metadata_overrides_return_controlled_status() throws {
        try requireANEHardwareTestsEnabledForChaining()

        let kernel = try chainingIdentityKernel()
        let probe = kernel.chainingProbe(
            useRealStatsSurface: true,
            skipPrepare: true,
            validateRequest: true,
            callEnqueueSets: true,
            requestProcedureIndex: 1,
            requestTransactionHandle: 7,
            requestFWEnqueueDelay: 3,
            requestMemoryPoolId: 11,
            enqueueProcedureIndex: 1,
            enqueueSetIndex: 1,
            enqueueSignalValue: 5,
            enqueueSignalNotRequired: false,
            enqueueOpenLoop: true
        )

        guard probe.hasChainingRequestClass, probe.hasOutputSetEnqueueClass else {
            throw XCTSkip("Chaining enqueue metadata probe unavailable on this host")
        }

        XCTAssertTrue(probe.builtOutputSet)
        XCTAssertTrue(probe.builtOutputSetEnqueue)
        XCTAssertTrue(probe.builtRequest)
        XCTAssertTrue(probe.requestValidated)
        XCTAssertTrue(
            probe.calledEnqueueSets,
            "enqueue metadata probe should attempt the client sequencing call"
        )
        XCTAssertFalse(probe.prepared)
        XCTAssertTrue(
            probe.stage == Int(ANE_INTEROP_CHAINING_STAGE_ENQUEUE_SETS_CALL_FAILED.rawValue) ||
            probe.stage == Int(ANE_INTEROP_CHAINING_STAGE_ENQUEUE_SETS_CALL_SUCCEEDED.rawValue),
            "enqueue metadata probe must report a controlled client-call result"
        )
        XCTAssertEqual(
            probe.enqueueSetsSucceeded,
            probe.stage == Int(ANE_INTEROP_CHAINING_STAGE_ENQUEUE_SETS_CALL_SUCCEEDED.rawValue)
        )
    }

    func test_chaining_probe_shared_signal_event_enqueue_returns_controlled_status() throws {
        try requireANEHardwareTestsEnabledForChaining()

        let kernel = try chainingIdentityKernel()
        let probe = kernel.chainingProbe(
            useRealStatsSurface: true,
            skipPrepare: true,
            validateRequest: true,
            callEnqueueSets: true,
            enqueueSignalValue: 5,
            enqueueSignalNotRequired: false,
            useSharedSignalEvent: true,
            sharedSignalEventValue: 5,
            sharedSignalEventSymbolIndex: 0,
            sharedSignalEventType: 0
        )

        guard probe.hasChainingRequestClass, probe.hasOutputSetEnqueueClass, probe.hasSharedSignalEventClass else {
            throw XCTSkip("Chaining shared-signal-event probe unavailable on this host")
        }

        XCTAssertTrue(probe.builtSharedSignalEvent)
        XCTAssertTrue(probe.builtRequest)
        XCTAssertTrue(probe.requestValidated)
        XCTAssertTrue(
            probe.calledEnqueueSets,
            "shared-signal-event probe should attempt the enqueueSets client call"
        )
        XCTAssertTrue(
            probe.stage == Int(ANE_INTEROP_CHAINING_STAGE_ENQUEUE_SETS_CALL_FAILED.rawValue) ||
            probe.stage == Int(ANE_INTEROP_CHAINING_STAGE_ENQUEUE_SETS_CALL_SUCCEEDED.rawValue),
            "shared-signal-event probe must report a controlled client-call result"
        )
    }

    func test_chaining_probe_scalar_loopback_ids_return_controlled_status() throws {
        try requireANEHardwareTestsEnabledForChaining()

        let kernel = try chainingIdentityKernel()
        let probe = kernel.chainingProbe(
            useRealStatsSurface: true,
            skipPrepare: true,
            validateRequest: true,
            useScalarLoopbackSymbolIndices: true
        )

        guard probe.hasChainingRequestClass else {
            throw XCTSkip("Chaining request probe unavailable on this host")
        }

        XCTAssertFalse(
            probe.usedArrayLoopbackSymbolIndices,
            "scalar-loopback probe should pass singular loopback ids"
        )
        XCTAssertTrue(
            probe.stage == Int(ANE_INTEROP_CHAINING_STAGE_PREPARE_SKIPPED.rawValue) ||
            probe.stage == Int(ANE_INTEROP_CHAINING_STAGE_REQUEST_VALIDATE_FAILED.rawValue) ||
            probe.stage == Int(ANE_INTEROP_CHAINING_STAGE_REQUEST_BUILD_FAILED.rawValue),
            "scalar-loopback probe must report a controlled pre-prepare result"
        )
    }

    func test_chaining_probe_scalar_loopback_buffers_ready_returns_controlled_status() throws {
        try requireANEHardwareTestsEnabledForChaining()

        let kernel = try chainingIdentityKernel()
        let probe = kernel.chainingProbe(
            useRealStatsSurface: true,
            skipPrepare: true,
            validateRequest: true,
            useScalarLoopbackSymbolIndices: true,
            callBuffersReady: true
        )

        guard probe.hasChainingRequestClass, probe.hasInputBuffersReadyClass else {
            throw XCTSkip("Scalar-loopback buffersReady probe unavailable on this host")
        }

        XCTAssertFalse(probe.usedArrayLoopbackSymbolIndices)
        XCTAssertTrue(
            probe.stage == Int(ANE_INTEROP_CHAINING_STAGE_REQUEST_VALIDATE_FAILED.rawValue) ||
            probe.stage == Int(ANE_INTEROP_CHAINING_STAGE_INPUT_BUFFERS_READY_CALL_FAILED.rawValue) ||
            probe.stage == Int(ANE_INTEROP_CHAINING_STAGE_INPUT_BUFFERS_READY_CALL_SUCCEEDED.rawValue),
            "scalar-loopback buffersReady probe must report a controlled request/client-call result"
        )
        XCTAssertEqual(
            probe.calledBuffersReady,
            probe.stage == Int(ANE_INTEROP_CHAINING_STAGE_INPUT_BUFFERS_READY_CALL_FAILED.rawValue) ||
            probe.stage == Int(ANE_INTEROP_CHAINING_STAGE_INPUT_BUFFERS_READY_CALL_SUCCEEDED.rawValue)
        )
    }

    func test_external_prepare_probe_isolated_from_test_harness() throws {
        try requireANEHardwareTestsEnabledForChaining()

        let executableURL = try espressoBenchExecutableURL()
        let outputDir = URL(fileURLWithPath: NSTemporaryDirectory(), isDirectory: true)
            .appendingPathComponent("decode_d5b_prepare_probe_\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: outputDir, withIntermediateDirectories: true)
        let processResult = try runProbeProcess(
            executableURL: executableURL,
            arguments: ["--probe-chaining-prepare", "--output", outputDir.path],
            currentDirectoryURL: repoRootURL(),
            timeoutSeconds: 20
        )

        let summaryURL = outputDir.appendingPathComponent("summary.json", isDirectory: false)
        XCTAssertTrue(
            FileManager.default.fileExists(atPath: summaryURL.path),
            "probe should emit summary metadata before entering prepare"
        )

        let summaryData = try Data(contentsOf: summaryURL)
        let summaryObject = try XCTUnwrap(JSONSerialization.jsonObject(with: summaryData) as? [String: Any])
        XCTAssertTrue(
            processResult.timedOut || processResult.status == 0,
            "isolated prepare probe must either time out under external control or exit cleanly. status=\(processResult.status) stderr=\n\(processResult.stderr)"
        )

        XCTAssertEqual(summaryObject["mode"] as? String, "chaining-prepare-probe")

        let probeOptions = try XCTUnwrap(summaryObject["probe_options"] as? [String: Any])
        XCTAssertEqual(probeOptions["stats_surface_mode"] as? String, "output0")
        XCTAssertEqual(probeOptions["use_real_stats_surface"] as? Bool, true)
        XCTAssertEqual(probeOptions["skip_prepare"] as? Bool, false)
        XCTAssertEqual(probeOptions["call_buffers_ready"] as? Bool, false)
        XCTAssertEqual(probeOptions["call_enqueue_sets"] as? Bool, false)
        XCTAssertEqual(probeOptions["validate_request"] as? Bool, true)

        let resultEntry = try XCTUnwrap(summaryObject["result"] as? [String: Any])
        if processResult.timedOut {
            let status = try XCTUnwrap(resultEntry["status"] as? String)
            XCTAssertTrue(
                ["started", "kernel_compiled"].contains(status),
                "timed-out external probe should stop at a known checkpoint"
            )
            if status == "kernel_compiled" {
                XCTAssertNotNil(resultEntry["compile_elapsed_ms"] as? Double)
            }
        } else {
            XCTAssertEqual(resultEntry["status"] as? String, "completed")
            XCTAssertNotNil(resultEntry["compile_elapsed_ms"] as? Double)
            XCTAssertNotNil(resultEntry["probe_elapsed_ms"] as? Double)
            let probe = try XCTUnwrap(summaryObject["probe"] as? [String: Any])
            XCTAssertEqual(probe["has_prepare_selector"] as? Bool, true)
            XCTAssertEqual(probe["built_output_set"] as? Bool, true)
            XCTAssertEqual(probe["built_request"] as? Bool, true)
            XCTAssertNotEqual(probe["stage"] as? Int, Int(ANE_INTEROP_CHAINING_STAGE_UNAVAILABLE.rawValue))
        }
    }

    func test_external_prepare_probe_records_skip_prepare_null_stats_surface_options() throws {
        try requireANEHardwareTestsEnabledForChaining()

        let executableURL = try espressoBenchExecutableURL()
        let outputDir = URL(fileURLWithPath: NSTemporaryDirectory(), isDirectory: true)
            .appendingPathComponent("decode_d5b_skip_prepare_probe_\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: outputDir, withIntermediateDirectories: true)
        let processResult = try runProbeProcess(
            executableURL: executableURL,
            arguments: [
                "--probe-chaining-prepare",
                "--probe-chaining-skip-prepare",
                "--probe-chaining-stats-surface", "null",
                "--output", outputDir.path,
            ],
            currentDirectoryURL: repoRootURL(),
            timeoutSeconds: 20
        )

        let summaryURL = outputDir.appendingPathComponent("summary.json", isDirectory: false)
        XCTAssertTrue(FileManager.default.fileExists(atPath: summaryURL.path))

        let summaryData = try Data(contentsOf: summaryURL)
        let summaryObject = try XCTUnwrap(JSONSerialization.jsonObject(with: summaryData) as? [String: Any])
        XCTAssertEqual(summaryObject["mode"] as? String, "chaining-prepare-probe")

        let probeOptions = try XCTUnwrap(summaryObject["probe_options"] as? [String: Any])
        XCTAssertEqual(probeOptions["stats_surface_mode"] as? String, "null")
        XCTAssertEqual(probeOptions["use_real_stats_surface"] as? Bool, false)
        XCTAssertEqual(probeOptions["skip_prepare"] as? Bool, true)
        XCTAssertEqual(probeOptions["call_buffers_ready"] as? Bool, false)
        XCTAssertEqual(probeOptions["call_enqueue_sets"] as? Bool, false)
        XCTAssertEqual(probeOptions["validate_request"] as? Bool, true)

        let resultEntry = try XCTUnwrap(summaryObject["result"] as? [String: Any])
        if processResult.timedOut {
            let status = try XCTUnwrap(resultEntry["status"] as? String)
            XCTAssertTrue(
                ["started", "kernel_compiled"].contains(status),
                "timed-out skip-prepare probe should stop at a known checkpoint"
            )
            if status == "kernel_compiled" {
                XCTAssertNotNil(resultEntry["compile_elapsed_ms"] as? Double)
            }
        } else {
            XCTAssertEqual(processResult.status, 0)
            XCTAssertEqual(resultEntry["status"] as? String, "completed")
            XCTAssertNotNil(resultEntry["compile_elapsed_ms"] as? Double)
            XCTAssertNotNil(resultEntry["probe_elapsed_ms"] as? Double)
            let probe = try XCTUnwrap(summaryObject["probe"] as? [String: Any])
            XCTAssertEqual(probe["built_output_set"] as? Bool, false)
            XCTAssertEqual(
                probe["stage"] as? Int,
                Int(ANE_INTEROP_CHAINING_STAGE_OUTPUT_SETS_BUILD_FAILED.rawValue)
            )
        }
    }

    func test_external_buffers_ready_probe_records_controlled_client_call() throws {
        try requireANEHardwareTestsEnabledForChaining()

        let executableURL = try espressoBenchExecutableURL()
        let outputDir = URL(fileURLWithPath: NSTemporaryDirectory(), isDirectory: true)
            .appendingPathComponent("decode_d5b_buffers_ready_probe_\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: outputDir, withIntermediateDirectories: true)
        let processResult = try runProbeProcess(
            executableURL: executableURL,
            arguments: [
                "--probe-chaining-prepare",
                "--probe-chaining-skip-prepare",
                "--probe-chaining-call-buffers-ready",
                "--output", outputDir.path,
            ],
            currentDirectoryURL: repoRootURL(),
            timeoutSeconds: 20
        )

        XCTAssertFalse(
            processResult.timedOut,
            "isolated buffersReady probe should return a controlled status instead of hanging. stderr=\n\(processResult.stderr)"
        )
        XCTAssertEqual(processResult.status, 0)

        let summaryURL = outputDir.appendingPathComponent("summary.json", isDirectory: false)
        XCTAssertTrue(FileManager.default.fileExists(atPath: summaryURL.path))

        let summaryData = try Data(contentsOf: summaryURL)
        let summaryObject = try XCTUnwrap(JSONSerialization.jsonObject(with: summaryData) as? [String: Any])
        XCTAssertEqual(summaryObject["mode"] as? String, "chaining-prepare-probe")

        let probeOptions = try XCTUnwrap(summaryObject["probe_options"] as? [String: Any])
        XCTAssertEqual(probeOptions["stats_surface_mode"] as? String, "output0")
        XCTAssertEqual(probeOptions["use_real_stats_surface"] as? Bool, true)
        XCTAssertEqual(probeOptions["skip_prepare"] as? Bool, true)
        XCTAssertEqual(probeOptions["call_buffers_ready"] as? Bool, true)
        XCTAssertEqual(probeOptions["call_enqueue_sets"] as? Bool, false)
        XCTAssertEqual(probeOptions["validate_request"] as? Bool, true)
        let env = try XCTUnwrap(summaryObject["env"] as? [String: String])
        XCTAssertEqual(env["ESPRESSO_BENCH_SEED"], "1")
        XCTAssertEqual(env["ANE_COMPILE_CACHE_POLICY"], "preferCached")

        let resultEntry = try XCTUnwrap(summaryObject["result"] as? [String: Any])
        XCTAssertEqual(resultEntry["status"] as? String, "completed")
        XCTAssertNotNil(resultEntry["compile_elapsed_ms"] as? Double)
        XCTAssertNotNil(resultEntry["probe_elapsed_ms"] as? Double)

        let probe = try XCTUnwrap(summaryObject["probe"] as? [String: Any])
        XCTAssertEqual(probe["has_prepare_selector"] as? Bool, true)
        XCTAssertEqual(probe["built_output_set"] as? Bool, true)
        XCTAssertEqual(probe["built_input_buffers_ready"] as? Bool, true)
        XCTAssertEqual(probe["built_request"] as? Bool, true)
        XCTAssertEqual(probe["request_validated"] as? Bool, true)
        XCTAssertEqual(probe["called_buffers_ready"] as? Bool, true)
        XCTAssertFalse(probe["prepared"] as? Bool ?? true)
        XCTAssertTrue(
            (probe["stage"] as? Int) == Int(ANE_INTEROP_CHAINING_STAGE_INPUT_BUFFERS_READY_CALL_FAILED.rawValue) ||
            (probe["stage"] as? Int) == Int(ANE_INTEROP_CHAINING_STAGE_INPUT_BUFFERS_READY_CALL_SUCCEEDED.rawValue),
            "external buffersReady probe must report a controlled client-call stage"
        )
        XCTAssertEqual(
            probe["buffers_ready_succeeded"] as? Bool,
            (probe["stage"] as? Int) == Int(ANE_INTEROP_CHAINING_STAGE_INPUT_BUFFERS_READY_CALL_SUCCEEDED.rawValue)
        )
    }

    func test_external_scalar_loopback_buffers_ready_probe_records_controlled_client_call() throws {
        try requireANEHardwareTestsEnabledForChaining()

        let executableURL = try espressoBenchExecutableURL()
        let outputDir = URL(fileURLWithPath: NSTemporaryDirectory(), isDirectory: true)
            .appendingPathComponent("decode_d5b_scalar_loopback_buffers_ready_probe_\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: outputDir, withIntermediateDirectories: true)
        let processResult = try runProbeProcess(
            executableURL: executableURL,
            arguments: [
                "--probe-chaining-prepare",
                "--probe-chaining-skip-prepare",
                "--probe-chaining-call-buffers-ready",
                "--probe-chaining-scalar-loopback",
                "--output", outputDir.path,
            ],
            currentDirectoryURL: repoRootURL(),
            timeoutSeconds: 20
        )

        XCTAssertFalse(
            processResult.timedOut,
            "isolated scalar-loopback buffersReady probe should return a controlled status instead of hanging. stderr=\n\(processResult.stderr)"
        )
        XCTAssertEqual(processResult.status, 0)

        let summaryURL = outputDir.appendingPathComponent("summary.json", isDirectory: false)
        XCTAssertTrue(FileManager.default.fileExists(atPath: summaryURL.path))

        let summaryData = try Data(contentsOf: summaryURL)
        let summaryObject = try XCTUnwrap(JSONSerialization.jsonObject(with: summaryData) as? [String: Any])
        XCTAssertEqual(summaryObject["mode"] as? String, "chaining-prepare-probe")

        let probeOptions = try XCTUnwrap(summaryObject["probe_options"] as? [String: Any])
        XCTAssertEqual(probeOptions["stats_surface_mode"] as? String, "output0")
        XCTAssertEqual(probeOptions["use_real_stats_surface"] as? Bool, true)
        XCTAssertEqual(probeOptions["skip_prepare"] as? Bool, true)
        XCTAssertEqual(probeOptions["call_buffers_ready"] as? Bool, true)
        XCTAssertEqual(probeOptions["call_enqueue_sets"] as? Bool, false)
        XCTAssertEqual(probeOptions["scalar_loopback"] as? Bool, true)
        XCTAssertEqual(probeOptions["validate_request"] as? Bool, true)

        let resultEntry = try XCTUnwrap(summaryObject["result"] as? [String: Any])
        XCTAssertEqual(resultEntry["status"] as? String, "completed")
        XCTAssertNotNil(resultEntry["compile_elapsed_ms"] as? Double)
        XCTAssertNotNil(resultEntry["probe_elapsed_ms"] as? Double)

        let probe = try XCTUnwrap(summaryObject["probe"] as? [String: Any])
        XCTAssertEqual(probe["used_array_loopback_symbol_indices"] as? Bool, false)
        XCTAssertTrue(
            (probe["stage"] as? Int) == Int(ANE_INTEROP_CHAINING_STAGE_REQUEST_VALIDATE_FAILED.rawValue) ||
            (probe["stage"] as? Int) == Int(ANE_INTEROP_CHAINING_STAGE_INPUT_BUFFERS_READY_CALL_FAILED.rawValue) ||
            (probe["stage"] as? Int) == Int(ANE_INTEROP_CHAINING_STAGE_INPUT_BUFFERS_READY_CALL_SUCCEEDED.rawValue),
            "external scalar-loopback buffersReady probe must report a controlled request/client-call stage"
        )
        XCTAssertEqual(
            probe["called_buffers_ready"] as? Bool,
            (probe["stage"] as? Int) == Int(ANE_INTEROP_CHAINING_STAGE_INPUT_BUFFERS_READY_CALL_FAILED.rawValue) ||
            (probe["stage"] as? Int) == Int(ANE_INTEROP_CHAINING_STAGE_INPUT_BUFFERS_READY_CALL_SUCCEEDED.rawValue)
        )
    }

    func test_external_enqueue_sets_probe_records_controlled_client_call() throws {
        try requireANEHardwareTestsEnabledForChaining()

        let executableURL = try espressoBenchExecutableURL()
        let outputDir = URL(fileURLWithPath: NSTemporaryDirectory(), isDirectory: true)
            .appendingPathComponent("decode_d5b_enqueue_sets_probe_\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: outputDir, withIntermediateDirectories: true)
        let processResult = try runProbeProcess(
            executableURL: executableURL,
            arguments: [
                "--probe-chaining-prepare",
                "--probe-chaining-skip-prepare",
                "--probe-chaining-call-enqueue-sets",
                "--output", outputDir.path,
            ],
            currentDirectoryURL: repoRootURL(),
            timeoutSeconds: 20
        )

        XCTAssertFalse(
            processResult.timedOut,
            "isolated enqueueSets probe should return a controlled status instead of hanging. stderr=\n\(processResult.stderr)"
        )
        XCTAssertEqual(processResult.status, 0)

        let summaryURL = outputDir.appendingPathComponent("summary.json", isDirectory: false)
        XCTAssertTrue(FileManager.default.fileExists(atPath: summaryURL.path))

        let summaryData = try Data(contentsOf: summaryURL)
        let summaryObject = try XCTUnwrap(JSONSerialization.jsonObject(with: summaryData) as? [String: Any])
        XCTAssertEqual(summaryObject["mode"] as? String, "chaining-prepare-probe")

        let probeOptions = try XCTUnwrap(summaryObject["probe_options"] as? [String: Any])
        XCTAssertEqual(probeOptions["stats_surface_mode"] as? String, "output0")
        XCTAssertEqual(probeOptions["use_real_stats_surface"] as? Bool, true)
        XCTAssertEqual(probeOptions["skip_prepare"] as? Bool, true)
        XCTAssertEqual(probeOptions["call_buffers_ready"] as? Bool, false)
        XCTAssertEqual(probeOptions["call_enqueue_sets"] as? Bool, true)
        XCTAssertEqual(probeOptions["validate_request"] as? Bool, true)

        let resultEntry = try XCTUnwrap(summaryObject["result"] as? [String: Any])
        XCTAssertEqual(resultEntry["status"] as? String, "completed")
        XCTAssertNotNil(resultEntry["compile_elapsed_ms"] as? Double)
        XCTAssertNotNil(resultEntry["probe_elapsed_ms"] as? Double)

        let probe = try XCTUnwrap(summaryObject["probe"] as? [String: Any])
        XCTAssertEqual(probe["has_prepare_selector"] as? Bool, true)
        XCTAssertEqual(probe["built_output_set"] as? Bool, true)
        XCTAssertEqual(probe["built_output_set_enqueue"] as? Bool, true)
        XCTAssertEqual(probe["built_request"] as? Bool, true)
        XCTAssertEqual(probe["request_validated"] as? Bool, true)
        XCTAssertEqual(probe["called_enqueue_sets"] as? Bool, true)
        XCTAssertFalse(probe["prepared"] as? Bool ?? true)
        XCTAssertTrue(
            (probe["stage"] as? Int) == Int(ANE_INTEROP_CHAINING_STAGE_ENQUEUE_SETS_CALL_FAILED.rawValue) ||
            (probe["stage"] as? Int) == Int(ANE_INTEROP_CHAINING_STAGE_ENQUEUE_SETS_CALL_SUCCEEDED.rawValue),
            "external enqueueSets probe must report a controlled client-call stage"
        )
        XCTAssertEqual(
            probe["enqueue_sets_succeeded"] as? Bool,
            (probe["stage"] as? Int) == Int(ANE_INTEROP_CHAINING_STAGE_ENQUEUE_SETS_CALL_SUCCEEDED.rawValue)
        )
    }

    func test_external_enqueue_sets_probe_records_metadata_overrides() throws {
        try requireANEHardwareTestsEnabledForChaining()

        let executableURL = try espressoBenchExecutableURL()
        let outputDir = URL(fileURLWithPath: NSTemporaryDirectory(), isDirectory: true)
            .appendingPathComponent("decode_d5b_enqueue_metadata_probe_\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: outputDir, withIntermediateDirectories: true)
        let processResult = try runProbeProcess(
            executableURL: executableURL,
            arguments: [
                "--probe-chaining-prepare",
                "--probe-chaining-skip-prepare",
                "--probe-chaining-call-enqueue-sets",
                "--probe-chaining-request-procedure-index", "1",
                "--probe-chaining-request-transaction-handle", "7",
                "--probe-chaining-request-fw-enqueue-delay", "3",
                "--probe-chaining-request-memory-pool-id", "11",
                "--probe-chaining-enqueue-procedure-index", "1",
                "--probe-chaining-enqueue-set-index", "1",
                "--probe-chaining-enqueue-signal-value", "5",
                "--probe-chaining-enqueue-require-signal",
                "--probe-chaining-enqueue-open-loop",
                "--probe-chaining-ready-procedure-index", "1",
                "--probe-chaining-ready-execution-delay", "2",
                "--output", outputDir.path,
            ],
            currentDirectoryURL: repoRootURL(),
            timeoutSeconds: 20
        )

        XCTAssertFalse(
            processResult.timedOut,
            "isolated enqueue metadata probe should return a controlled status instead of hanging. stderr=\n\(processResult.stderr)"
        )
        XCTAssertEqual(processResult.status, 0)

        let summaryURL = outputDir.appendingPathComponent("summary.json", isDirectory: false)
        XCTAssertTrue(FileManager.default.fileExists(atPath: summaryURL.path))

        let summaryData = try Data(contentsOf: summaryURL)
        let summaryObject = try XCTUnwrap(JSONSerialization.jsonObject(with: summaryData) as? [String: Any])
        XCTAssertEqual(summaryObject["mode"] as? String, "chaining-prepare-probe")

        let probeOptions = try XCTUnwrap(summaryObject["probe_options"] as? [String: Any])
        XCTAssertEqual(probeOptions["request_procedure_index"] as? UInt32, 1)
        XCTAssertEqual(probeOptions["request_transaction_handle"] as? UInt64, 7)
        XCTAssertEqual(probeOptions["request_fw_enqueue_delay"] as? UInt64, 3)
        XCTAssertEqual(probeOptions["request_memory_pool_id"] as? UInt64, 11)
        XCTAssertEqual(probeOptions["enqueue_procedure_index"] as? UInt32, 1)
        XCTAssertEqual(probeOptions["enqueue_set_index"] as? UInt32, 1)
        XCTAssertEqual(probeOptions["enqueue_signal_value"] as? UInt64, 5)
        XCTAssertEqual(probeOptions["enqueue_signal_not_required"] as? Bool, false)
        XCTAssertEqual(probeOptions["enqueue_open_loop"] as? Bool, true)
        XCTAssertEqual(probeOptions["ready_procedure_index"] as? UInt32, 1)
        XCTAssertEqual(probeOptions["ready_execution_delay"] as? UInt64, 2)

        let resultEntry = try XCTUnwrap(summaryObject["result"] as? [String: Any])
        XCTAssertEqual(resultEntry["status"] as? String, "completed")
        XCTAssertNotNil(resultEntry["compile_elapsed_ms"] as? Double)
        XCTAssertNotNil(resultEntry["probe_elapsed_ms"] as? Double)

        let probe = try XCTUnwrap(summaryObject["probe"] as? [String: Any])
        XCTAssertEqual(probe["called_enqueue_sets"] as? Bool, true)
        XCTAssertTrue(
            (probe["stage"] as? Int) == Int(ANE_INTEROP_CHAINING_STAGE_ENQUEUE_SETS_CALL_FAILED.rawValue) ||
            (probe["stage"] as? Int) == Int(ANE_INTEROP_CHAINING_STAGE_ENQUEUE_SETS_CALL_SUCCEEDED.rawValue),
            "external enqueue metadata probe must report a controlled client-call stage"
        )
    }

    func test_external_shared_signal_event_probe_records_metadata_overrides() throws {
        try requireANEHardwareTestsEnabledForChaining()

        let executableURL = try espressoBenchExecutableURL()
        let outputDir = URL(fileURLWithPath: NSTemporaryDirectory(), isDirectory: true)
            .appendingPathComponent("decode_d5b_shared_signal_event_probe_\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: outputDir, withIntermediateDirectories: true)
        let processResult = try runProbeProcess(
            executableURL: executableURL,
            arguments: [
                "--probe-chaining-prepare",
                "--probe-chaining-skip-prepare",
                "--probe-chaining-call-enqueue-sets",
                "--probe-chaining-use-shared-signal-event",
                "--probe-chaining-shared-signal-value", "5",
                "--probe-chaining-shared-signal-symbol-index", "0",
                "--probe-chaining-shared-signal-event-type", "0",
                "--probe-chaining-enqueue-signal-value", "5",
                "--probe-chaining-enqueue-require-signal",
                "--output", outputDir.path,
            ],
            currentDirectoryURL: repoRootURL(),
            timeoutSeconds: 20
        )

        XCTAssertFalse(
            processResult.timedOut,
            "isolated shared-signal-event probe should return a controlled status instead of hanging. stderr=\n\(processResult.stderr)"
        )
        XCTAssertEqual(processResult.status, 0)

        let summaryURL = outputDir.appendingPathComponent("summary.json", isDirectory: false)
        XCTAssertTrue(FileManager.default.fileExists(atPath: summaryURL.path))

        let summaryData = try Data(contentsOf: summaryURL)
        let summaryObject = try XCTUnwrap(JSONSerialization.jsonObject(with: summaryData) as? [String: Any])
        XCTAssertEqual(summaryObject["mode"] as? String, "chaining-prepare-probe")

        let probeOptions = try XCTUnwrap(summaryObject["probe_options"] as? [String: Any])
        XCTAssertEqual(probeOptions["use_shared_signal_event"] as? Bool, true)
        XCTAssertEqual(probeOptions["shared_signal_event_value"] as? UInt64, 5)
        XCTAssertEqual(probeOptions["shared_signal_event_symbol_index"] as? UInt32, 0)
        XCTAssertEqual(probeOptions["shared_signal_event_type"] as? Int64, 0)
        XCTAssertEqual(probeOptions["enqueue_signal_value"] as? UInt64, 5)
        XCTAssertEqual(probeOptions["enqueue_signal_not_required"] as? Bool, false)

        let resultEntry = try XCTUnwrap(summaryObject["result"] as? [String: Any])
        XCTAssertEqual(resultEntry["status"] as? String, "completed")

        let probe = try XCTUnwrap(summaryObject["probe"] as? [String: Any])
        XCTAssertEqual(probe["has_shared_signal_event_class"] as? Bool, true)
        XCTAssertEqual(probe["built_shared_signal_event"] as? Bool, true)
        XCTAssertEqual(probe["called_enqueue_sets"] as? Bool, true)
        XCTAssertTrue(
            (probe["stage"] as? Int) == Int(ANE_INTEROP_CHAINING_STAGE_ENQUEUE_SETS_CALL_FAILED.rawValue) ||
            (probe["stage"] as? Int) == Int(ANE_INTEROP_CHAINING_STAGE_ENQUEUE_SETS_CALL_SUCCEEDED.rawValue),
            "external shared-signal-event probe must report a controlled client-call stage"
        )
    }
}
