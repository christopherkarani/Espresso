import Foundation
import XCTest
import Accelerate
import Darwin
import ANETypes
import ANERuntime
import CPUOps
@testable import Espresso

final class EspressoTests: XCTestCase {
    private static func makeTempBinaryPath(prefix: String) -> String {
        let fileName = "\(prefix)-\(UUID().uuidString).bin"
        return FileManager.default.temporaryDirectory.appendingPathComponent(fileName).path
    }

    private static func writeUInt16File(_ values: [UInt16], to path: String) throws {
        let le = values.map { $0.littleEndian }
        let data = le.withUnsafeBytes { raw in Data(raw) }
        try data.write(to: URL(fileURLWithPath: path), options: .atomic)
    }

    private static func readFileBytes(path: String) throws -> Data {
        try Data(contentsOf: URL(fileURLWithPath: path), options: .mappedIfSafe)
    }

    private static func readFloatBitPattern(_ data: Data, offset: Int) -> UInt32 {
        precondition(offset >= 0)
        precondition(offset + 4 <= data.count)
        let b0 = UInt32(data[offset + 0])
        let b1 = UInt32(data[offset + 1]) << 8
        let b2 = UInt32(data[offset + 2]) << 16
        let b3 = UInt32(data[offset + 3]) << 24
        return b0 | b1 | b2 | b3
    }

    private static func makeTempDirectory(prefix: String) throws -> URL {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("\(prefix)-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    private static func repoRootURL() -> URL {
        // .../Tests/EspressoTests/EspressoTests.swift -> repo root (3 levels up).
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
    }

    private struct ProcessResult {
        let status: Int32
        let stdout: String
        let stderr: String
    }

    private static func runProcess(
        executableURL: URL,
        arguments: [String],
        currentDirectoryURL: URL,
        timeoutSeconds: TimeInterval
    ) throws -> ProcessResult {
        let stdoutPipe = Pipe()
        let stderrPipe = Pipe()

        let p = Process()
        p.executableURL = executableURL
        p.arguments = arguments
        p.currentDirectoryURL = currentDirectoryURL
        p.standardOutput = stdoutPipe
        p.standardError = stderrPipe

        let exited = DispatchSemaphore(value: 0)
        p.terminationHandler = { _ in exited.signal() }

        try p.run()

        if exited.wait(timeout: .now() + timeoutSeconds) != .success {
            p.terminate()
            throw NSError(domain: "EspressoTests.ProcessTimeout", code: 1)
        }

        let stdoutData = stdoutPipe.fileHandleForReading.readDataToEndOfFile()
        let stderrData = stderrPipe.fileHandleForReading.readDataToEndOfFile()

        return ProcessResult(
            status: p.terminationStatus,
            stdout: String(decoding: stdoutData, as: UTF8.self),
            stderr: String(decoding: stderrData, as: UTF8.self)
        )
    }

    private struct StepEvent {
        let step: Int
        let loss: Double
        let ms: Double?
    }

    private static func parseStepEvents(stderr: String) -> [StepEvent] {
        stderr
            .split(whereSeparator: \.isNewline)
            .compactMap { lineSub in
                let line = String(lineSub)
                guard line.hasPrefix("{"), line.contains("\"type\":\"step\"") else { return nil }
                guard let data = line.data(using: .utf8) else { return nil }
                guard
                    let obj = try? JSONSerialization.jsonObject(with: data),
                    let dict = obj as? [String: Any],
                    (dict["type"] as? String) == "step"
                else { return nil }

                guard let step = dict["step"] as? Int else { return nil }
                guard let loss = (dict["loss"] as? NSNumber)?.doubleValue else { return nil }
                let ms = (dict["ms"] as? NSNumber)?.doubleValue
                return StepEvent(step: step, loss: loss, ms: ms)
            }
    }

    private func requireIntegrationTestsEnabled(file: StaticString = #filePath, line: UInt = #line) throws {
        guard ProcessInfo.processInfo.environment["ESPRESSO_INTEGRATION_TESTS"] == "1" else {
            throw XCTSkip("Integration tests disabled; set ESPRESSO_INTEGRATION_TESTS=1", file: file, line: line)
        }
    }

    private func requirePerfTestsEnabled(file: StaticString = #filePath, line: UInt = #line) throws {
        guard ProcessInfo.processInfo.environment["ESPRESSO_PERF_TESTS"] == "1" else {
            throw XCTSkip("Perf tests disabled; set ESPRESSO_PERF_TESTS=1", file: file, line: line)
        }
    }

    private static func cpuBrandString() -> String? {
        var size: size_t = 0
        guard sysctlbyname("machdep.cpu.brand_string", nil, &size, nil, 0) == 0 else { return nil }
        var buf = [CChar](repeating: 0, count: Int(size))
        guard sysctlbyname("machdep.cpu.brand_string", &buf, &size, nil, 0) == 0 else { return nil }
        let n = buf.firstIndex(of: 0) ?? buf.count
        let bytes = buf.prefix(n).map { UInt8(bitPattern: $0) }
        return String(decoding: bytes, as: UTF8.self)
    }

    // MARK: - ANE gating helpers

    /// Skip if ANE runtime is unavailable.
    private func requireANEAvailable(file: StaticString = #filePath, line: UInt = #line) throws {
        let handle = dlopen(
            "/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine",
            RTLD_NOW
        )
        if handle == nil {
            throw XCTSkip("AppleNeuralEngine.framework unavailable", file: file, line: line)
        }
        dlclose(handle)

        let requiredClasses = [
            "_ANEInMemoryModelDescriptor",
            "_ANEInMemoryModel",
            "_ANERequest",
            "_ANEIOSurfaceObject",
        ]
        for c in requiredClasses where NSClassFromString(c) == nil {
            throw XCTSkip("ANE private class missing: \(c)", file: file, line: line)
        }
    }

    /// Skip tests that require ANE hardware unless ANE_HARDWARE_TESTS=1.
    private func requireANEHardwareTestsEnabled(file: StaticString = #filePath, line: UInt = #line) throws {
        guard ProcessInfo.processInfo.environment["ANE_HARDWARE_TESTS"] == "1" else {
            throw XCTSkip("ANE hardware tests disabled", file: file, line: line)
        }
        try requireANEAvailable(file: file, line: line)
    }

    private func fillTensor(_ buffer: borrowing TensorBuffer, value: Float) {
        buffer.withUnsafeMutableBufferPointer { ptr in
            for i in 0..<ptr.count { ptr[i] = value }
        }
    }

    private func fillLayerWeights(_ weights: borrowing LayerWeights, value: Float) {
        fillTensor(weights.Wq, value: value)
        fillTensor(weights.Wk, value: value)
        fillTensor(weights.Wv, value: value)
        fillTensor(weights.Wo, value: value)
        fillTensor(weights.W1, value: value)
        fillTensor(weights.W2, value: value)
        fillTensor(weights.W3, value: value)
        fillTensor(weights.rmsAtt, value: value)
        fillTensor(weights.rmsFfn, value: value)
    }

    final class LockedCounter: @unchecked Sendable {
        private let lock = NSLock()
        private var value: Int = 0

        func increment() {
            lock.lock()
            value += 1
            lock.unlock()
        }

        func load() -> Int {
            lock.lock()
            let v = value
            lock.unlock()
            return v
        }
    }

    func test_ane_option_snapshot_defaults_when_env_empty() {
        let snapshot = ANEOptionSnapshot.fromEnvironment([:])
        XCTAssertEqual(snapshot.compileCachePolicy, "auto")
        XCTAssertEqual(snapshot.evalPath, "inmem")
        XCTAssertFalse(snapshot.strictOptions)
        XCTAssertFalse(snapshot.disablePowerSaving)
        XCTAssertFalse(snapshot.keepModelWired)
        XCTAssertFalse(snapshot.enableLateLatch)
        XCTAssertFalse(snapshot.skipPrepare)
        XCTAssertFalse(snapshot.disableIOFences)
        XCTAssertFalse(snapshot.enableFWToFWSignal)
        XCTAssertFalse(snapshot.useCompilerOptions)
        XCTAssertFalse(snapshot.perfStatsRequested)
        XCTAssertNil(snapshot.queueDepth)
        XCTAssertNil(snapshot.memoryPoolID)
        XCTAssertNil(snapshot.perfStatsMask)
        XCTAssertEqual(snapshot.decodeLaneSpatial, 32)
        XCTAssertEqual(snapshot.benchSeed, "")

        let json = snapshot.asJSON()
        XCTAssertEqual(json["compile_cache_policy"] as? String, "auto")
        XCTAssertEqual(json["eval_path"] as? String, "inmem")
        XCTAssertEqual(json["strict_options"] as? Bool, false)
        XCTAssertEqual(json["decode_lane_spatial"] as? Int, 32)
        XCTAssertNil(json["queue_depth"])
        XCTAssertNil(json["memory_pool_id"])
        XCTAssertNil(json["perf_stats_mask"])
    }

    func test_ane_option_snapshot_reads_overrides_and_normalizes_lane_spatial() {
        let env: [String: String] = [
            "ANE_COMPILE_CACHE_POLICY": "preferCached",
            "ANE_EVAL_PATH": "clientDirect",
            "ANE_STRICT_OPTIONS": "1",
            "ANE_DISABLE_POWER_SAVING": "1",
            "ANE_KEEP_MODEL_WIRED": "1",
            "ANE_ENABLE_LATE_LATCH": "1",
            "ANE_SKIP_PREPARE": "1",
            "ANE_DISABLE_IO_FENCES": "1",
            "ANE_ENABLE_FW_TO_FW_SIGNAL": "1",
            "ANE_USE_COMPILER_OPTIONS": "1",
            "ANE_PERF_STATS": "1",
            "ANE_QUEUE_DEPTH": "16",
            "ANE_MEMORY_POOL_ID": "2",
            "ANE_PERF_STATS_MASK": "0xF",
            "ESPRESSO_DECODE_LANE_SPATIAL": "16",
            "ESPRESSO_BENCH_SEED": "1",
        ]
        let snapshot = ANEOptionSnapshot.fromEnvironment(env)
        XCTAssertEqual(snapshot.compileCachePolicy, "preferCached")
        XCTAssertEqual(snapshot.evalPath, "clientDirect")
        XCTAssertTrue(snapshot.strictOptions)
        XCTAssertTrue(snapshot.disablePowerSaving)
        XCTAssertTrue(snapshot.keepModelWired)
        XCTAssertTrue(snapshot.enableLateLatch)
        XCTAssertTrue(snapshot.skipPrepare)
        XCTAssertTrue(snapshot.disableIOFences)
        XCTAssertTrue(snapshot.enableFWToFWSignal)
        XCTAssertTrue(snapshot.useCompilerOptions)
        XCTAssertTrue(snapshot.perfStatsRequested)
        XCTAssertEqual(snapshot.queueDepth, "16")
        XCTAssertEqual(snapshot.memoryPoolID, "2")
        XCTAssertEqual(snapshot.perfStatsMask, "0xF")
        XCTAssertEqual(snapshot.decodeLaneSpatial, 32)
        XCTAssertEqual(snapshot.benchSeed, "1")

        var envLargeLane = env
        envLargeLane["ESPRESSO_DECODE_LANE_SPATIAL"] = "128"
        let largeLane = ANEOptionSnapshot.fromEnvironment(envLargeLane)
        XCTAssertEqual(largeLane.decodeLaneSpatial, 128)

        let json = snapshot.asJSON()
        XCTAssertEqual(json["queue_depth"] as? String, "16")
        XCTAssertEqual(json["memory_pool_id"] as? String, "2")
        XCTAssertEqual(json["perf_stats_mask"] as? String, "0xF")
    }

    func test_gradient_accumulator_enqueue_and_barrier() {
        let accumulator = GradientAccumulator()
        let counter = LockedCounter()

        for _ in 0..<3 {
            accumulator.enqueue {
                counter.increment()
            }
        }

        accumulator.barrier()
        XCTAssertEqual(counter.load(), 3)
    }

    func test_gradient_accumulator_wait_all() {
        let accumulator = GradientAccumulator()
        let counter = LockedCounter()

        for _ in 0..<5 {
            accumulator.enqueue {
                // Small delay to ensure blocks are truly async.
                usleep(2_000)
                counter.increment()
            }
        }

        accumulator.waitAll()
        XCTAssertEqual(counter.load(), 5)
    }

    func test_token_dataset_small_file() throws {
        let path = Self.makeTempBinaryPath(prefix: "token-dataset-small")
        defer { try? FileManager.default.removeItem(atPath: path) }

        let values = (0..<100).map { UInt16($0) }
        try Self.writeUInt16File(values, to: path)

        let dataset = try TokenDataset(path: path, seqLen: 10)
        XCTAssertEqual(dataset.nTokens, 100)
        XCTAssertEqual(dataset[0].pointee, 0)
        XCTAssertEqual(dataset[99].pointee, 99)
    }

    func test_token_dataset_validates_minimum_size() throws {
        let path = Self.makeTempBinaryPath(prefix: "token-dataset-too-small")
        defer { try? FileManager.default.removeItem(atPath: path) }

        let seqLen = 16
        let values = (0..<seqLen).map { UInt16($0) } // fewer than seqLen+1
        try Self.writeUInt16File(values, to: path)

        do {
            _ = try TokenDataset(path: path, seqLen: seqLen)
            XCTFail("Expected TokenDataset to reject files with < seqLen+1 tokens")
        } catch {
            // Expected.
        }
    }

    func test_token_dataset_sets_close_on_exec() throws {
        let path = Self.makeTempBinaryPath(prefix: "token-dataset-cloexec")
        defer { try? FileManager.default.removeItem(atPath: path) }

        let values = (0..<32).map { UInt16($0) }
        try Self.writeUInt16File(values, to: path)

        let dataset = try TokenDataset(path: path, seqLen: 8)
        let flags = fcntl(dataset._debugFileDescriptor, F_GETFD)
        XCTAssertNotEqual(flags, -1, "fcntl(F_GETFD) failed")
        XCTAssertNotEqual(flags & FD_CLOEXEC, 0, "Expected FD_CLOEXEC to be set on dataset fd")
    }

    func test_sampler_deterministic_sequence() {
        let maxPos = 1000

        Sampler.seed(startStep: 0)
        let seq1 = (0..<10).map { _ in Sampler.samplePosition(maxPos: maxPos) }

        Sampler.seed(startStep: 0)
        let seq2 = (0..<10).map { _ in Sampler.samplePosition(maxPos: maxPos) }

        XCTAssertEqual(seq1, seq2)
    }

    func test_sampler_range_valid() {
        let maxPos = 123

        Sampler.seed(startStep: 1)
        for _ in 0..<100 {
            let pos = Sampler.samplePosition(maxPos: maxPos)
            XCTAssertGreaterThanOrEqual(pos, 0)
            XCTAssertLessThan(pos, maxPos)
        }
    }

    func test_exec_restart_formats_message() {
        XCTAssertEqual(
            ExecRestart.message(step: 7, compileCount: 42, loss: 1.23456),
            "[exec() restart step 7, 42 compiles, loss=1.2346]"
        )
    }

    func test_exec_restart_resolved_executable_path_is_valid() {
        let path = ExecRestart.resolvedExecutablePath()
        XCTAssertFalse(path.isEmpty)
        XCTAssertTrue(path.contains("/"), "Expected an executable path containing '/' (got: \(path))")
        XCTAssertTrue(FileManager.default.fileExists(atPath: path), "Expected resolved executable path to exist: \(path)")
    }

    func test_exec_restart_argv_preserves_args_and_includes_resume_once() {
        let execPath = "/abs/path/espresso-train"

        do {
            let argv = ExecRestart.restartArgv(
                currentArguments: ["espresso-train", "--data", "d.bin", "--steps", "10"],
                resolvedExecPath: execPath
            )
            XCTAssertEqual(argv, [execPath, "--data", "d.bin", "--steps", "10", "--resume"])
        }

        do {
            let argv = ExecRestart.restartArgv(
                currentArguments: ["espresso-train", "--resume", "--steps", "10", "--resume"],
                resolvedExecPath: execPath
            )
            XCTAssertEqual(argv, [execPath, "--steps", "10", "--resume"])
        }
    }

    func test_exec_restart_matches_objc_contract() {
        let execPath = "/abs/path/espresso-train"

        let argv = ExecRestart.restartArgv(
            currentArguments: ["./espresso-train", "--steps", "100", "--lr", "0.0003"],
            resolvedExecPath: execPath
        )
        XCTAssertEqual(argv, [execPath, "--steps", "100", "--lr", "0.0003", "--resume"])

        let idempotent = ExecRestart.restartArgv(
            currentArguments: ["./espresso-train", "--resume", "--steps", "100", "--resume"],
            resolvedExecPath: execPath
        )
        XCTAssertEqual(idempotent, [execPath, "--steps", "100", "--resume"])

        // tiny_train.m uses execl(argv[0], argv[0], "--resume", NULL):
        // when no user args are present, Swift restartArgv is behaviorally identical.
        let objcEquivalent = ExecRestart.restartArgv(
            currentArguments: ["./espresso-train"],
            resolvedExecPath: execPath
        )
        XCTAssertEqual(objcEquivalent, [execPath, "--resume"])
    }

    func test_checkpoint_header_validation() throws {
        let dim = 4
        let hidden = 8
        let nLayers = 1
        let vocab = 10

        let path = Self.makeTempBinaryPath(prefix: "ckpt-bad-header")
        defer { try? FileManager.default.removeItem(atPath: path) }

        var header = CheckpointHeader()
        header.dim = Int32(dim)
        header.hiddenDim = Int32(hidden)
        header.nLayers = Int32(nLayers)
        header.vocabSize = Int32(vocab)
        header.nHeads = 1
        header.seqLen = 2

        // Write header only, with wrong magic.
        header.magic = Int32(bitPattern: 0xDEADBEEF)
        let headerData = withUnsafeBytes(of: &header) { raw in Data(raw) }
        try headerData.write(to: URL(fileURLWithPath: path), options: .atomic)

        do {
            _ = try Checkpoint._loadTiny(path: path, dim: dim, hidden: hidden, nLayers: nLayers, vocab: vocab)
            XCTFail("Expected invalid magic to be rejected")
        } catch CheckpointError.invalidMagic(_) {
            // Expected.
        } catch {
            XCTFail("Unexpected error: \(error)")
        }

        // Wrong version.
        header = CheckpointHeader()
        header.dim = Int32(dim)
        header.hiddenDim = Int32(hidden)
        header.nLayers = Int32(nLayers)
        header.vocabSize = Int32(vocab)
        header.nHeads = 1
        header.seqLen = 2
        header.version = 99
        let badVersionData = withUnsafeBytes(of: &header) { raw in Data(raw) }
        try badVersionData.write(to: URL(fileURLWithPath: path), options: .atomic)

        do {
            _ = try Checkpoint._loadTiny(path: path, dim: dim, hidden: hidden, nLayers: nLayers, vocab: vocab)
            XCTFail("Expected unsupported version to be rejected")
        } catch CheckpointError.unsupportedVersion(_) {
            // Expected.
        } catch {
            XCTFail("Unexpected error: \(error)")
        }
    }

    func test_checkpoint_segment_order_small() throws {
        let dim = 4
        let hidden = 8
        let nLayers = 1
        let vocab = 10

        let path = Self.makeTempBinaryPath(prefix: "ckpt-segment-order")
        defer { try? FileManager.default.removeItem(atPath: path) }

        let layerWeights = LayerStorage<Checkpoint.TinyLayerWeights>(count: nLayers) { _ in
            Checkpoint.TinyLayerWeights(dim: dim, hidden: hidden)
        }
        let layerAdam = LayerStorage<Checkpoint.TinyLayerAdam>(count: nLayers) { _ in
            Checkpoint.TinyLayerAdam(dim: dim, hidden: hidden)
        }
        let rmsFinal = TensorBuffer(count: dim, zeroed: false)
        let adamRmsFinal = AdamState(count: dim)
        let embed = TensorBuffer(count: vocab * dim, zeroed: false)
        let adamEmbed = AdamState(count: vocab * dim)

        func fill(_ buffer: borrowing TensorBuffer, value: Float) {
            buffer.withUnsafeMutableBufferPointer { ptr in
                for i in 0..<ptr.count { ptr[i] = value }
            }
        }

        // Per-layer weights.
        fill(layerWeights[0].Wq, value: 1)
        fill(layerWeights[0].Wk, value: 2)
        fill(layerWeights[0].Wv, value: 3)
        fill(layerWeights[0].Wo, value: 4)
        fill(layerWeights[0].W1, value: 5)
        fill(layerWeights[0].W2, value: 6)
        fill(layerWeights[0].W3, value: 7)
        fill(layerWeights[0].rmsAtt, value: 8)
        fill(layerWeights[0].rmsFfn, value: 9)

        // Per-layer adam (m/v pairs), same order.
        fill(layerAdam[0].Wq.m, value: 11); fill(layerAdam[0].Wq.v, value: 12)
        fill(layerAdam[0].Wk.m, value: 13); fill(layerAdam[0].Wk.v, value: 14)
        fill(layerAdam[0].Wv.m, value: 15); fill(layerAdam[0].Wv.v, value: 16)
        fill(layerAdam[0].Wo.m, value: 17); fill(layerAdam[0].Wo.v, value: 18)
        fill(layerAdam[0].W1.m, value: 19); fill(layerAdam[0].W1.v, value: 20)
        fill(layerAdam[0].W2.m, value: 21); fill(layerAdam[0].W2.v, value: 22)
        fill(layerAdam[0].W3.m, value: 23); fill(layerAdam[0].W3.v, value: 24)
        fill(layerAdam[0].rmsAtt.m, value: 25); fill(layerAdam[0].rmsAtt.v, value: 26)
        fill(layerAdam[0].rmsFfn.m, value: 27); fill(layerAdam[0].rmsFfn.v, value: 28)

        // Globals.
        fill(rmsFinal, value: 101)
        fill(adamRmsFinal.m, value: 102); fill(adamRmsFinal.v, value: 103)
        fill(embed, value: 104)
        fill(adamEmbed.m, value: 105); fill(adamEmbed.v, value: 106)

        var meta = CheckpointMeta()
        meta.step = 1
        meta.totalSteps = 2
        meta.lr = 0.001
        meta.loss = 9.0
        meta.cumCompile = 1.0
        meta.cumTrain = 2.0
        meta.cumWall = 3.0
        meta.cumSteps = 4
        meta.cumBatches = 5
        meta.adamT = 6

        try Checkpoint._saveTiny(
            path: path,
            dim: dim,
            hidden: hidden,
            nLayers: nLayers,
            vocab: vocab,
            meta: meta,
            layers: layerWeights,
            layerAdam: layerAdam,
            rmsFinal: rmsFinal,
            adamRmsFinal: adamRmsFinal,
            embed: embed,
            adamEmbed: adamEmbed
        )

        let bytes = try Self.readFileBytes(path: path)
        let segments = Checkpoint._tinyLayout(dim: dim, hidden: hidden, nLayers: nLayers, vocab: vocab)

        let expected: [String: Float] = [
            "L0.Wq": 1, "L0.Wk": 2, "L0.Wv": 3, "L0.Wo": 4, "L0.W1": 5, "L0.W2": 6, "L0.W3": 7,
            "L0.rmsAtt": 8, "L0.rmsFfn": 9,
            "L0.Wq.m": 11, "L0.Wq.v": 12,
            "L0.Wk.m": 13, "L0.Wk.v": 14,
            "L0.Wv.m": 15, "L0.Wv.v": 16,
            "L0.Wo.m": 17, "L0.Wo.v": 18,
            "L0.W1.m": 19, "L0.W1.v": 20,
            "L0.W2.m": 21, "L0.W2.v": 22,
            "L0.W3.m": 23, "L0.W3.v": 24,
            "L0.rmsAtt.m": 25, "L0.rmsAtt.v": 26,
            "L0.rmsFfn.m": 27, "L0.rmsFfn.v": 28,
            "rmsFinal": 101,
            "adamRmsFinal.m": 102, "adamRmsFinal.v": 103,
            "embed": 104,
            "adamEmbed.m": 105, "adamEmbed.v": 106,
        ]

        for seg in segments {
            guard seg.name != "header" else { continue }
            guard let exp = expected[seg.name] else {
                XCTFail("Missing expected marker for segment \(seg.name)")
                continue
            }
            let gotBits = Self.readFloatBitPattern(bytes, offset: seg.byteOffset)
            XCTAssertEqual(gotBits, exp.bitPattern, "Segment \(seg.name) at offset \(seg.byteOffset) mismatch")
        }
    }

    func test_checkpoint_save_load_roundtrip() throws {
        let dim = 4
        let hidden = 8
        let nLayers = 1
        let vocab = 10

        let path1 = Self.makeTempBinaryPath(prefix: "ckpt-roundtrip-1")
        let path2 = Self.makeTempBinaryPath(prefix: "ckpt-roundtrip-2")
        defer {
            try? FileManager.default.removeItem(atPath: path1)
            try? FileManager.default.removeItem(atPath: path2)
        }

        let layers1 = LayerStorage<Checkpoint.TinyLayerWeights>(count: nLayers) { _ in
            Checkpoint.TinyLayerWeights(dim: dim, hidden: hidden)
        }
        let adam1 = LayerStorage<Checkpoint.TinyLayerAdam>(count: nLayers) { _ in
            Checkpoint.TinyLayerAdam(dim: dim, hidden: hidden)
        }
        let rmsFinal1 = TensorBuffer(count: dim, zeroed: false)
        let adamRmsFinal1 = AdamState(count: dim)
        let embed1 = TensorBuffer(count: vocab * dim, zeroed: false)
        let adamEmbed1 = AdamState(count: vocab * dim)

        func fillRamp(_ buffer: borrowing TensorBuffer, base: Float) {
            buffer.withUnsafeMutableBufferPointer { ptr in
                for i in 0..<ptr.count { ptr[i] = base + Float(i) * 0.001 }
            }
        }

        // Fill with non-uniform data to ensure exact roundtrip.
        fillRamp(layers1[0].Wq, base: 1.0)
        fillRamp(layers1[0].Wk, base: 2.0)
        fillRamp(layers1[0].Wv, base: 3.0)
        fillRamp(layers1[0].Wo, base: 4.0)
        fillRamp(layers1[0].W1, base: 5.0)
        fillRamp(layers1[0].W2, base: 6.0)
        fillRamp(layers1[0].W3, base: 7.0)
        fillRamp(layers1[0].rmsAtt, base: 8.0)
        fillRamp(layers1[0].rmsFfn, base: 9.0)

        fillRamp(adam1[0].Wq.m, base: 11.0); fillRamp(adam1[0].Wq.v, base: 12.0)
        fillRamp(adam1[0].Wk.m, base: 13.0); fillRamp(adam1[0].Wk.v, base: 14.0)
        fillRamp(adam1[0].Wv.m, base: 15.0); fillRamp(adam1[0].Wv.v, base: 16.0)
        fillRamp(adam1[0].Wo.m, base: 17.0); fillRamp(adam1[0].Wo.v, base: 18.0)
        fillRamp(adam1[0].W1.m, base: 19.0); fillRamp(adam1[0].W1.v, base: 20.0)
        fillRamp(adam1[0].W2.m, base: 21.0); fillRamp(adam1[0].W2.v, base: 22.0)
        fillRamp(adam1[0].W3.m, base: 23.0); fillRamp(adam1[0].W3.v, base: 24.0)
        fillRamp(adam1[0].rmsAtt.m, base: 25.0); fillRamp(adam1[0].rmsAtt.v, base: 26.0)
        fillRamp(adam1[0].rmsFfn.m, base: 27.0); fillRamp(adam1[0].rmsFfn.v, base: 28.0)

        fillRamp(rmsFinal1, base: 101.0)
        fillRamp(adamRmsFinal1.m, base: 102.0); fillRamp(adamRmsFinal1.v, base: 103.0)
        fillRamp(embed1, base: 104.0)
        fillRamp(adamEmbed1.m, base: 105.0); fillRamp(adamEmbed1.v, base: 106.0)

        var meta = CheckpointMeta()
        meta.step = 123
        meta.totalSteps = 456
        meta.lr = 0.001
        meta.loss = 9.0
        meta.cumCompile = 1.0
        meta.cumTrain = 2.0
        meta.cumWall = 3.0
        meta.cumSteps = 4
        meta.cumBatches = 5
        meta.adamT = 6

        try Checkpoint._saveTiny(
            path: path1,
            dim: dim,
            hidden: hidden,
            nLayers: nLayers,
            vocab: vocab,
            meta: meta,
            layers: layers1,
            layerAdam: adam1,
            rmsFinal: rmsFinal1,
            adamRmsFinal: adamRmsFinal1,
            embed: embed1,
            adamEmbed: adamEmbed1
        )

        // Load into a fresh state and save again; bytes must match.
        let layers2 = LayerStorage<Checkpoint.TinyLayerWeights>(count: nLayers) { _ in
            Checkpoint.TinyLayerWeights(dim: dim, hidden: hidden)
        }
        let adam2 = LayerStorage<Checkpoint.TinyLayerAdam>(count: nLayers) { _ in
            Checkpoint.TinyLayerAdam(dim: dim, hidden: hidden)
        }
        let rmsFinal2 = TensorBuffer(count: dim, zeroed: true)
        let adamRmsFinal2 = AdamState(count: dim)
        let embed2 = TensorBuffer(count: vocab * dim, zeroed: true)
        let adamEmbed2 = AdamState(count: vocab * dim)

        let loadedMeta = try Checkpoint._loadTiny(
            path: path1,
            dim: dim,
            hidden: hidden,
            nLayers: nLayers,
            vocab: vocab,
            intoLayers: layers2,
            intoLayerAdam: adam2,
            intoRmsFinal: rmsFinal2,
            intoAdamRmsFinal: adamRmsFinal2,
            intoEmbed: embed2,
            intoAdamEmbed: adamEmbed2
        )
        XCTAssertEqual(loadedMeta, meta)

        try Checkpoint._saveTiny(
            path: path2,
            dim: dim,
            hidden: hidden,
            nLayers: nLayers,
            vocab: vocab,
            meta: meta,
            layers: layers2,
            layerAdam: adam2,
            rmsFinal: rmsFinal2,
            adamRmsFinal: adamRmsFinal2,
            embed: embed2,
            adamEmbed: adamEmbed2
        )

        let b1 = try Self.readFileBytes(path: path1)
        let b2 = try Self.readFileBytes(path: path2)
        XCTAssertEqual(b1, b2)
    }

    func test_forward_single_layer_output_nonzero_finite() throws {
        try requireANEHardwareTestsEnabled()

        // Ensure budget is available for compilation within this test target.
        try? CompileBudget.setCount(0)

        let dim = ModelConfig.dim
        let seq = ModelConfig.seqLen

        let weights = LayerWeights()
        fillLayerWeights(weights, value: 0.01)
        let kernelStorage = try LayerStorage<LayerKernelSet>(count: 1, throwingInitializer: { _ in
            try LayerKernelSet(weights: weights)
        })
        let actStorage = LayerStorage<LayerActivations>(count: 1) { _ in LayerActivations() }

        let xCur = TensorBuffer(count: dim * seq, zeroed: false)
        fillTensor(xCur, value: 0.02)

        let accumulator = GradientAccumulator()
        try ForwardPass.run(xCur: xCur, acts: actStorage, kernels: kernelStorage, accumulator: accumulator)

        // Output is written back into xCur.
        let stats = xCur.withUnsafeBufferPointer { ptr -> (hasNonZero: Bool, allFinite: Bool) in
            let hasNonZero = ptr.contains(where: { $0 != 0 })
            let allFinite = ptr.allSatisfy { $0.isFinite }
            return (hasNonZero, allFinite)
        }
        XCTAssertTrue(stats.hasNonZero)
        XCTAssertTrue(stats.allFinite)
    }

    func test_forward_12_layers_no_nan() throws {
        try requireANEHardwareTestsEnabled()
        try? CompileBudget.setCount(0)

        let dim = ModelConfig.dim
        let seq = ModelConfig.seqLen

        let layers = LayerStorage<LayerWeights>(count: ModelConfig.nLayers) { _ in
            let w = LayerWeights()
            fillLayerWeights(w, value: 0.01)
            return w
        }

        let kernelStorage = try LayerStorage<LayerKernelSet>(count: ModelConfig.nLayers, throwingInitializer: { i in
            try LayerKernelSet(weights: layers[i])
        })
        let actStorage = LayerStorage<LayerActivations>(count: ModelConfig.nLayers) { _ in LayerActivations() }

        let xCur = TensorBuffer(count: dim * seq, zeroed: false)
        fillTensor(xCur, value: 0.02)

        let accumulator = GradientAccumulator()
        try ForwardPass.run(xCur: xCur, acts: actStorage, kernels: kernelStorage, accumulator: accumulator)

        func assertAllFinite(_ buffer: borrowing TensorBuffer, _ name: String, file: StaticString = #filePath, line: UInt = #line) {
            buffer.withUnsafeBufferPointer { ptr in
                XCTAssertTrue(ptr.allSatisfy { $0.isFinite }, "\(name) contains NaN/Inf", file: file, line: line)
            }
        }

        assertAllFinite(xCur, "xCur")
        for L in 0..<actStorage.count {
            // Q/K/V are intentionally not read back to CPU in ForwardPass (they persist on IOSurface for backward).
            assertAllFinite(actStorage[L].layerIn, "acts[\(L)].layerIn")
            assertAllFinite(actStorage[L].xnorm, "acts[\(L)].xnorm")
            assertAllFinite(actStorage[L].attnOut, "acts[\(L)].attnOut")
            assertAllFinite(actStorage[L].oOut, "acts[\(L)].oOut")
            assertAllFinite(actStorage[L].x2, "acts[\(L)].x2")
            assertAllFinite(actStorage[L].x2norm, "acts[\(L)].x2norm")
            assertAllFinite(actStorage[L].h1, "acts[\(L)].h1")
            assertAllFinite(actStorage[L].h3, "acts[\(L)].h3")
            assertAllFinite(actStorage[L].siluOut, "acts[\(L)].siluOut")
            assertAllFinite(actStorage[L].ffnOut, "acts[\(L)].ffnOut")
        }
    }

    func test_backward_produces_nonzero_gradients() throws {
        try requireANEHardwareTestsEnabled()
        try? CompileBudget.setCount(0)

        let dim = ModelConfig.dim
        let hidden = ModelConfig.hidden
        let seq = ModelConfig.seqLen

        let weightsStorage = LayerStorage<LayerWeights>(count: 1) { _ in
            let w = LayerWeights()
            fillLayerWeights(w, value: 0.01)
            return w
        }
        let kernelStorage = try LayerStorage<LayerKernelSet>(count: 1, throwingInitializer: { i in
            try LayerKernelSet(weights: weightsStorage[i])
        })
        let staticStorage = try LayerStorage<StaticKernel>(count: 1, throwingInitializer: { _ in
            try StaticKernel()
        })
        let actStorage = LayerStorage<LayerActivations>(count: 1) { _ in LayerActivations() }
        let gradStorage = LayerStorage<LayerGradients>(count: 1) { _ in LayerGradients() }

        let xCur = TensorBuffer(count: dim * seq, zeroed: false)
        fillTensor(xCur, value: 0.02)

        let accumulator = GradientAccumulator()
        try ForwardPass.run(xCur: xCur, acts: actStorage, kernels: kernelStorage, accumulator: accumulator)

        let dy = TensorBuffer(count: dim * seq, zeroed: false)
        fillTensor(dy, value: 0.1)

        let scratch = BackwardScratch(dim: dim, hidden: hidden, seqLen: seq)
        let rmsWorkspace = RMSNorm.Workspace(seqLen: seq)
        try BackwardPass.run(
            dy: dy,
            acts: actStorage,
            kernels: kernelStorage,
            staticKernels: staticStorage,
            grads: gradStorage,
            weights: weightsStorage,
            scratch: scratch,
            rmsWorkspace: rmsWorkspace,
            accumulator: accumulator
        )

        // dW blocks complete due to barrier inside BackwardPass, but waitAll is cheap and makes this test robust.
        accumulator.waitAll()

        let rmsAttMax = gradStorage[0].rmsAtt.withUnsafeBufferPointer { ptr -> Float in
            var m: Float = 0
            for v in ptr { m = max(m, abs(v)) }
            return m
        }
        let w1Max = gradStorage[0].W1.withUnsafeBufferPointer { ptr -> Float in
            var m: Float = 0
            for i in 0..<min(ptr.count, 4096) { m = max(m, abs(ptr[i])) }
            return m
        }

        XCTAssertGreaterThan(rmsAttMax, 0, "Expected nonzero RMSNorm gradients")
        XCTAssertGreaterThan(w1Max, 0, "Expected nonzero weight gradients")
    }

    func test_backward_residual_gradient_flow() throws {
        try requireANEHardwareTestsEnabled()
        try? CompileBudget.setCount(0)

        let dim = ModelConfig.dim
        let hidden = ModelConfig.hidden
        let seq = ModelConfig.seqLen

        let weightsStorage = LayerStorage<LayerWeights>(count: 1) { _ in
            let w = LayerWeights()
            fillLayerWeights(w, value: 0.01)
            return w
        }
        let kernelStorage = try LayerStorage<LayerKernelSet>(count: 1, throwingInitializer: { i in
            try LayerKernelSet(weights: weightsStorage[i])
        })
        let staticStorage = try LayerStorage<StaticKernel>(count: 1, throwingInitializer: { _ in
            try StaticKernel()
        })
        let actStorage = LayerStorage<LayerActivations>(count: 1) { _ in LayerActivations() }
        let gradStorage = LayerStorage<LayerGradients>(count: 1) { _ in LayerGradients() }

        let xCur = TensorBuffer(count: dim * seq, zeroed: false)
        fillTensor(xCur, value: 0.02)

        let accumulator = GradientAccumulator()
        try ForwardPass.run(xCur: xCur, acts: actStorage, kernels: kernelStorage, accumulator: accumulator)

        let dy = TensorBuffer(count: dim * seq, zeroed: false)
        fillTensor(dy, value: 0.1)

        let scratch = BackwardScratch(dim: dim, hidden: hidden, seqLen: seq)
        let rmsWorkspace = RMSNorm.Workspace(seqLen: seq)
        try BackwardPass.run(
            dy: dy,
            acts: actStorage,
            kernels: kernelStorage,
            staticKernels: staticStorage,
            grads: gradStorage,
            weights: weightsStorage,
            scratch: scratch,
            rmsWorkspace: rmsWorkspace,
            accumulator: accumulator
        )

        // Single-layer: dy must equal dxRms1 + dx2 from that layer.
        dy.withUnsafeBufferPointer { dyPtr in
            scratch.dx2.withUnsafeBufferPointer { dx2Ptr in
                scratch.dxRms1.withUnsafeBufferPointer { dxrPtr in
                    for i in 0..<min(dyPtr.count, 4096) {
                        let exp = dx2Ptr[i] + dxrPtr[i]
                        XCTAssertEqual(dyPtr[i], exp, accuracy: 1e-5, "dy[\(i)] mismatch")
                    }
                }
            }
        }
    }

    func test_backward_iosurface_copy_chain() throws {
        try requireANEHardwareTestsEnabled()
        try? CompileBudget.setCount(0)

        let dim = ModelConfig.dim
        let seq = ModelConfig.seqLen
        let scoreCh = ModelConfig.scoreCh

        func maxAbsDiff(_ a: [Float], _ b: [Float]) -> Float {
            precondition(a.count == b.count)
            var m: Float = 0
            for i in a.indices { m = max(m, abs(a[i] - b[i])) }
            return m
        }

        let weights = LayerWeights()
        fillLayerWeights(weights, value: 0.01)
        let kernels = try LayerKernelSet(weights: weights)
        let staticKernel = try StaticKernel()

        let fwdAttnIn = try kernels.fwdAttn.inputSurface(at: 0)
        let fwdAttnOut = try kernels.fwdAttn.outputSurface(at: 0)
        let sdpa1In = try kernels.sdpaBwd1.inputSurface(at: 0)
        let sdpa1Out = try kernels.sdpaBwd1.outputSurface(at: 0)
        let sdpa2In = try staticKernel.kernel.inputSurface(at: 0)
        let sdpa2Out = try staticKernel.kernel.outputSurface(at: 0)
        let qkvIn = try kernels.qkvBwd.inputSurface(at: 0)
        let qkvOut = try kernels.qkvBwd.outputSurface(at: 0)

        var xCur = [Float](repeating: 0, count: dim * seq)
        for i in xCur.indices { xCur[i] = Float(i % 128 + 1) * 0.005 }
        xCur.withUnsafeBufferPointer { buf in
            SurfaceIO.writeFP16(to: fwdAttnIn, data: buf, channels: dim, spatial: seq)
        }
        try kernels.fwdAttn.eval()

        // 1) Q|K|V copy: fwdAttn out @DIM -> sdpaBwd1 in @0
        try SurfaceIO.copyFP16(
            dst: sdpa1In,
            dstChannelOffset: 0,
            src: fwdAttnOut,
            srcChannelOffset: dim,
            channels: 3 * dim,
            spatial: seq
        )
        var srcQKV = [Float](repeating: 0, count: 3 * dim * seq)
        var dstQKV = [Float](repeating: 0, count: 3 * dim * seq)
        srcQKV.withUnsafeMutableBufferPointer { out in
            SurfaceIO.readFP16(from: fwdAttnOut, into: out, channelOffset: dim, channels: 3 * dim, spatial: seq)
        }
        dstQKV.withUnsafeMutableBufferPointer { out in
            SurfaceIO.readFP16(from: sdpa1In, into: out, channelOffset: 0, channels: 3 * dim, spatial: seq)
        }
        XCTAssertLessThan(maxAbsDiff(srcQKV, dstQKV), 1e-2)
        XCTAssertTrue(dstQKV.contains(where: { $0 != 0 }))

        // 2) dx2 -> sdpaBwd1 @3*DIM
        var dy = [Float](repeating: 0, count: dim * seq)
        for i in dy.indices { dy[i] = Float(i % 64 + 1) * 0.01 }
        try dy.withUnsafeBufferPointer { buf in
            try SurfaceIO.writeFP16At(
                to: sdpa1In,
                channelOffset: 3 * dim,
                data: buf,
                channels: dim,
                spatial: seq
            )
        }
        var roundtripDy = [Float](repeating: 0, count: dim * seq)
        roundtripDy.withUnsafeMutableBufferPointer { out in
            SurfaceIO.readFP16(from: sdpa1In, into: out, channelOffset: 3 * dim, channels: dim, spatial: seq)
        }
        XCTAssertLessThan(maxAbsDiff(dy, roundtripDy), 1e-2)

        // 3) sdpaBwd1 eval
        try kernels.sdpaBwd1.eval()

        // 4) dscores copy: sdpaBwd1 out @DIM -> sdpaBwd2 in @0
        try SurfaceIO.copyFP16(
            dst: sdpa2In,
            dstChannelOffset: 0,
            src: sdpa1Out,
            srcChannelOffset: dim,
            channels: 2 * scoreCh,
            spatial: seq
        )
        var dscoresSrc = [Float](repeating: 0, count: 2 * scoreCh * seq)
        var dscoresDst = [Float](repeating: 0, count: 2 * scoreCh * seq)
        dscoresSrc.withUnsafeMutableBufferPointer { out in
            SurfaceIO.readFP16(from: sdpa1Out, into: out, channelOffset: dim, channels: 2 * scoreCh, spatial: seq)
        }
        dscoresDst.withUnsafeMutableBufferPointer { out in
            SurfaceIO.readFP16(from: sdpa2In, into: out, channelOffset: 0, channels: 2 * scoreCh, spatial: seq)
        }
        XCTAssertLessThan(maxAbsDiff(dscoresSrc, dscoresDst), 1e-2)
        XCTAssertTrue(dscoresDst.contains(where: { $0 != 0 }))

        // 5) Q|K copy: fwdAttn out @DIM -> sdpaBwd2 in @2*SCORE_CH
        try SurfaceIO.copyFP16(
            dst: sdpa2In,
            dstChannelOffset: 2 * scoreCh,
            src: fwdAttnOut,
            srcChannelOffset: dim,
            channels: 2 * dim,
            spatial: seq
        )
        var qkSrc = [Float](repeating: 0, count: 2 * dim * seq)
        var qkDst = [Float](repeating: 0, count: 2 * dim * seq)
        qkSrc.withUnsafeMutableBufferPointer { out in
            SurfaceIO.readFP16(from: fwdAttnOut, into: out, channelOffset: dim, channels: 2 * dim, spatial: seq)
        }
        qkDst.withUnsafeMutableBufferPointer { out in
            SurfaceIO.readFP16(from: sdpa2In, into: out, channelOffset: 2 * scoreCh, channels: 2 * dim, spatial: seq)
        }
        XCTAssertLessThan(maxAbsDiff(qkSrc, qkDst), 1e-2)

        // 6) sdpaBwd2 eval
        try staticKernel.kernel.eval()

        // 7) dv from sdpaBwd1 output @0
        var dv = [Float](repeating: 0, count: dim * seq)
        dv.withUnsafeMutableBufferPointer { out in
            SurfaceIO.readFP16(from: sdpa1Out, into: out, channelOffset: 0, channels: dim, spatial: seq)
        }
        XCTAssertTrue(dv.allSatisfy(\.isFinite))
        XCTAssertTrue(dv.contains(where: { $0 != 0 }))

        // 8) dq|dk from sdpaBwd2 output @0
        var dqdk = [Float](repeating: 0, count: 2 * dim * seq)
        dqdk.withUnsafeMutableBufferPointer { out in
            SurfaceIO.readFP16(from: sdpa2Out, into: out, channelOffset: 0, channels: 2 * dim, spatial: seq)
        }
        XCTAssertTrue(dqdk.allSatisfy(\.isFinite))
        XCTAssertTrue(dqdk.contains(where: { $0 != 0 }))

        // 9) qkvBwd setup: dq|dk from sdpaBwd2 + dv from sdpaBwd1
        try SurfaceIO.copyFP16(
            dst: qkvIn,
            dstChannelOffset: 0,
            src: sdpa2Out,
            srcChannelOffset: 0,
            channels: 2 * dim,
            spatial: seq
        )
        try SurfaceIO.copyFP16(
            dst: qkvIn,
            dstChannelOffset: 2 * dim,
            src: sdpa1Out,
            srcChannelOffset: 0,
            channels: dim,
            spatial: seq
        )

        var qkvDv = [Float](repeating: 0, count: dim * seq)
        qkvDv.withUnsafeMutableBufferPointer { out in
            SurfaceIO.readFP16(from: qkvIn, into: out, channelOffset: 2 * dim, channels: dim, spatial: seq)
        }
        XCTAssertLessThan(maxAbsDiff(dv, qkvDv), 1e-2, "dv must come from sdpaBwd1 output")

        try kernels.qkvBwd.eval()
        var dxAttn = [Float](repeating: 0, count: dim * seq)
        dxAttn.withUnsafeMutableBufferPointer { out in
            SurfaceIO.readFP16(from: qkvOut, into: out, channelOffset: 0, channels: dim, spatial: seq)
        }
        XCTAssertTrue(dxAttn.allSatisfy(\.isFinite))
        XCTAssertTrue(dxAttn.contains(where: { $0 != 0 }))
    }

    func test_gradient_accumulation_averages() {
        let stepsBatch = 2
        let gsc = 1.0 / Float(stepsBatch)

        let buf = TensorBuffer(count: 16, zeroed: false)
        fillTensor(buf, value: 2.0) // sum of two 1.0 micro-step grads

        GradientScaling.scale(buf, by: gsc)

        buf.withUnsafeBufferPointer { ptr in
            XCTAssertTrue(ptr.allSatisfy { $0 == 1.0 })
        }
    }

    func test_gradient_accumulation_scaling() {
        let stepsBatch = 2
        let gsc = 1.0 / Float(stepsBatch)

        // Small synthetic buffers to validate scaling is applied to every gradient group.
        let Wq = TensorBuffer(count: 8, zeroed: false)
        let Wk = TensorBuffer(count: 8, zeroed: false)
        let Wv = TensorBuffer(count: 8, zeroed: false)
        let Wo = TensorBuffer(count: 8, zeroed: false)
        let W1 = TensorBuffer(count: 8, zeroed: false)
        let W2 = TensorBuffer(count: 8, zeroed: false)
        let W3 = TensorBuffer(count: 8, zeroed: false)
        let rmsAtt = TensorBuffer(count: 8, zeroed: false)
        let rmsFfn = TensorBuffer(count: 8, zeroed: false)

        let grmsFinal = TensorBuffer(count: 8, zeroed: false)
        let gembed = TensorBuffer(count: 8, zeroed: false)

        // Unique markers per buffer.
        fillTensor(Wq, value: 2); fillTensor(Wk, value: 4); fillTensor(Wv, value: 6); fillTensor(Wo, value: 8)
        fillTensor(W1, value: 10); fillTensor(W2, value: 12); fillTensor(W3, value: 14)
        fillTensor(rmsAtt, value: 16); fillTensor(rmsFfn, value: 18)
        fillTensor(grmsFinal, value: 20)
        fillTensor(gembed, value: 22)

        GradientScaling.scaleLayer(
            Wq: Wq, Wk: Wk, Wv: Wv, Wo: Wo,
            W1: W1, W2: W2, W3: W3,
            rmsAtt: rmsAtt, rmsFfn: rmsFfn,
            by: gsc
        )
        GradientScaling.scale(grmsFinal, by: gsc)
        GradientScaling.scale(gembed, by: gsc)

        func assertAllEqual(_ buffer: borrowing TensorBuffer, _ expected: Float, file: StaticString = #filePath, line: UInt = #line) {
            buffer.withUnsafeBufferPointer { ptr in
                XCTAssertTrue(ptr.allSatisfy { $0 == expected }, file: file, line: line)
            }
        }

        assertAllEqual(Wq, 1); assertAllEqual(Wk, 2); assertAllEqual(Wv, 3); assertAllEqual(Wo, 4)
        assertAllEqual(W1, 5); assertAllEqual(W2, 6); assertAllEqual(W3, 7)
        assertAllEqual(rmsAtt, 8); assertAllEqual(rmsFfn, 9)
        assertAllEqual(grmsFinal, 10)
        assertAllEqual(gembed, 11)
    }

    func test_exec_restart_checkpoint_roundtrip() throws {
        // This validates the "--resume" metadata restore path without requiring ANE hardware.
        let dim = 4
        let hidden = 8
        let nLayers = 1
        let vocab = 10

        let path = Self.makeTempBinaryPath(prefix: "ckpt-exec-roundtrip")
        defer { try? FileManager.default.removeItem(atPath: path) }

        let layers = LayerStorage<Checkpoint.TinyLayerWeights>(count: nLayers) { _ in
            Checkpoint.TinyLayerWeights(dim: dim, hidden: hidden)
        }
        let adam = LayerStorage<Checkpoint.TinyLayerAdam>(count: nLayers) { _ in
            Checkpoint.TinyLayerAdam(dim: dim, hidden: hidden)
        }
        let rmsFinal = TensorBuffer(count: dim, zeroed: false)
        let adamRmsFinal = AdamState(count: dim)
        let embed = TensorBuffer(count: vocab * dim, zeroed: false)
        let adamEmbed = AdamState(count: vocab * dim)

        fillTensor(rmsFinal, value: 1.0)
        fillTensor(embed, value: 2.0)

        var meta = CheckpointMeta()
        meta.step = 123
        meta.totalSteps = 456
        meta.lr = 3e-4
        meta.loss = 9.876
        meta.cumCompile = 11.0
        meta.cumTrain = 22.0
        meta.cumWall = 33.0
        meta.cumSteps = 44
        meta.cumBatches = 55
        meta.adamT = 66

        try Checkpoint._saveTiny(
            path: path,
            dim: dim,
            hidden: hidden,
            nLayers: nLayers,
            vocab: vocab,
            meta: meta,
            layers: layers,
            layerAdam: adam,
            rmsFinal: rmsFinal,
            adamRmsFinal: adamRmsFinal,
            embed: embed,
            adamEmbed: adamEmbed
        )

        let loaded = try Checkpoint._loadTiny(path: path, dim: dim, hidden: hidden, nLayers: nLayers, vocab: vocab)
        XCTAssertEqual(loaded, meta)
    }

    // MARK: - Group 4B: Integration Tests (ANE-gated)

    func test_single_step_loss_matches_objc() throws {
        try requireANEHardwareTestsEnabled()
        try requireIntegrationTestsEnabled()

        let repoRoot = Self.repoRootURL()
        let objcExe = repoRoot.appendingPathComponent("training/train_large")
        let swiftExe = repoRoot.appendingPathComponent(".build/debug/espresso-train")

        guard FileManager.default.isExecutableFile(atPath: objcExe.path) else {
            throw XCTSkip("ObjC train_large not found/executable at \(objcExe.path)")
        }
        guard FileManager.default.isExecutableFile(atPath: swiftExe.path) else {
            throw XCTSkip("Swift espresso-train not found/executable at \(swiftExe.path)")
        }

        let dir = try Self.makeTempDirectory(prefix: "espresso-int-loss1")
        defer { try? FileManager.default.removeItem(at: dir) }

        // Force pos=0 always: nTokens = seqLen+1 => maxPos = 0.
        let tokens = [UInt16](repeating: 0, count: ModelConfig.seqLen + 1)
        try Self.writeUInt16File(tokens, to: dir.appendingPathComponent("tinystories_data00.bin").path)

        let objc = try Self.runProcess(
            executableURL: objcExe,
            arguments: ["--steps", "1"],
            currentDirectoryURL: dir,
            timeoutSeconds: 600
        )
        XCTAssertEqual(objc.status, 0, "ObjC process failed. stderr:\n\(objc.stderr)")

        let objcEvents = Self.parseStepEvents(stderr: objc.stderr)
        guard let objcLoss0 = objcEvents.first(where: { $0.step == 0 })?.loss else {
            XCTFail("ObjC did not emit step 0 JSON telemetry. stderr:\n\(objc.stderr)")
            return
        }

        let swift = try Self.runProcess(
            executableURL: swiftExe,
            arguments: ["--steps", "1", "--ckpt", "/dev/null"],
            currentDirectoryURL: dir,
            timeoutSeconds: 600
        )
        XCTAssertEqual(swift.status, 0, "Swift process failed. stderr:\n\(swift.stderr)")

        let swiftEvents = Self.parseStepEvents(stderr: swift.stderr)
        guard let swiftLoss0 = swiftEvents.first(where: { $0.step == 0 })?.loss else {
            XCTFail("Swift did not emit step 0 JSON telemetry. stderr:\n\(swift.stderr)")
            return
        }

        XCTAssertLessThan(abs(swiftLoss0 - objcLoss0), 0.01, "Loss mismatch. objc=\(objcLoss0) swift=\(swiftLoss0)")
    }

    func test_10_steps_loss_decreases() throws {
        try requireANEHardwareTestsEnabled()
        try requireIntegrationTestsEnabled()

        guard let modelPath = ProcessInfo.processInfo.environment["STORIES_MODEL_PATH"] else {
            throw XCTSkip("STORIES_MODEL_PATH not set")
        }
        guard FileManager.default.fileExists(atPath: modelPath) else {
            throw XCTSkip("STORIES_MODEL_PATH missing: \(modelPath)")
        }

        let repoRoot = Self.repoRootURL()
        let swiftExe = repoRoot.appendingPathComponent(".build/debug/espresso-train")
        guard FileManager.default.isExecutableFile(atPath: swiftExe.path) else {
            throw XCTSkip("Swift espresso-train not found/executable at \(swiftExe.path)")
        }

        let dir = try Self.makeTempDirectory(prefix: "espresso-int-loss10")
        defer { try? FileManager.default.removeItem(at: dir) }

        let tokens = [UInt16](repeating: 0, count: ModelConfig.seqLen + 1)
        try Self.writeUInt16File(tokens, to: dir.appendingPathComponent("tinystories_data00.bin").path)

        let res = try Self.runProcess(
            executableURL: swiftExe,
            arguments: ["--steps", "10", "--model", modelPath, "--ckpt", "/dev/null"],
            currentDirectoryURL: dir,
            timeoutSeconds: 1_800
        )
        XCTAssertEqual(res.status, 0, "Swift process failed. stderr:\n\(res.stderr)")

        let events = Self.parseStepEvents(stderr: res.stderr)
        guard let loss0 = events.first(where: { $0.step == 0 })?.loss else {
            XCTFail("Missing step 0 telemetry. stderr:\n\(res.stderr)")
            return
        }
        guard let loss9 = events.first(where: { $0.step == 9 })?.loss else {
            XCTFail("Missing step 9 telemetry. stderr:\n\(res.stderr)")
            return
        }

        XCTAssertLessThan(loss9, loss0, "Expected loss to decrease over 10 steps. loss0=\(loss0) loss9=\(loss9)")
    }

    func test_checkpoint_binary_compatible_with_objc() throws {
        try requireANEHardwareTestsEnabled()
        try requireIntegrationTestsEnabled()

        guard ProcessInfo.processInfo.environment["ESPRESSO_CKPT_COMPAT_TESTS"] == "1" else {
            throw XCTSkip("Checkpoint compat test disabled; set ESPRESSO_CKPT_COMPAT_TESTS=1")
        }

        let repoRoot = Self.repoRootURL()
        let objcExe = repoRoot.appendingPathComponent("training/train_large")
        let swiftExe = repoRoot.appendingPathComponent(".build/debug/espresso-train")

        guard FileManager.default.isExecutableFile(atPath: objcExe.path) else {
            throw XCTSkip("ObjC train_large not found/executable at \(objcExe.path)")
        }
        guard FileManager.default.isExecutableFile(atPath: swiftExe.path) else {
            throw XCTSkip("Swift espresso-train not found/executable at \(swiftExe.path)")
        }

        let dir = try Self.makeTempDirectory(prefix: "espresso-int-ckpt-compat")
        defer { try? FileManager.default.removeItem(at: dir) }

        let tokens = [UInt16](repeating: 0, count: ModelConfig.seqLen + 1)
        try Self.writeUInt16File(tokens, to: dir.appendingPathComponent("tinystories_data00.bin").path)

        // ObjC writes the checkpoint only on compile-budget restart; --steps 11 forces a restart at step 10.
        let objc = try Self.runProcess(
            executableURL: objcExe,
            arguments: ["--steps", "11"],
            currentDirectoryURL: dir,
            timeoutSeconds: 3_600
        )
        XCTAssertEqual(objc.status, 0, "ObjC process failed. stderr:\n\(objc.stderr)")

        let ckptPath = dir.appendingPathComponent("ane_stories110M_ckpt.bin").path
        guard FileManager.default.fileExists(atPath: ckptPath) else {
            XCTFail("Expected ObjC checkpoint at \(ckptPath)")
            return
        }

        let objcEvents = Self.parseStepEvents(stderr: objc.stderr)
        guard let objcLoss10 = objcEvents.first(where: { $0.step == 10 })?.loss else {
            XCTFail("ObjC did not emit step 10 telemetry. stderr:\n\(objc.stderr)")
            return
        }

        // Swift should be able to resume from the ObjC checkpoint.
        let swift = try Self.runProcess(
            executableURL: swiftExe,
            arguments: ["--resume"],
            currentDirectoryURL: dir,
            timeoutSeconds: 3_600
        )
        XCTAssertEqual(swift.status, 0, "Swift resume failed. stderr:\n\(swift.stderr)")

        let swiftEvents = Self.parseStepEvents(stderr: swift.stderr)
        guard let swiftLoss10 = swiftEvents.first(where: { $0.step == 10 })?.loss else {
            XCTFail("Swift did not emit step 10 telemetry. stderr:\n\(swift.stderr)")
            return
        }

        XCTAssertLessThan(abs(swiftLoss10 - objcLoss10), 0.01, "Resume loss mismatch. objc=\(objcLoss10) swift=\(swiftLoss10)")
    }

    func test_1_step_gradients_match_objc() throws {
        try requireANEHardwareTestsEnabled()
        try requireIntegrationTestsEnabled()

        guard ProcessInfo.processInfo.environment["ESPRESSO_GRADIENT_PARITY_TESTS"] == "1" else {
            throw XCTSkip("Gradient parity test disabled; set ESPRESSO_GRADIENT_PARITY_TESTS=1")
        }

        let repoRoot = Self.repoRootURL()
        let objcExe = repoRoot.appendingPathComponent("training/train_large")
        let swiftExe = repoRoot.appendingPathComponent(".build/debug/espresso-train")

        guard FileManager.default.isExecutableFile(atPath: objcExe.path) else {
            throw XCTSkip("ObjC train_large not found/executable at \(objcExe.path)")
        }
        guard FileManager.default.isExecutableFile(atPath: swiftExe.path) else {
            throw XCTSkip("Swift espresso-train not found/executable at \(swiftExe.path)")
        }

        let objcDir = try Self.makeTempDirectory(prefix: "espresso-int-grad-objc")
        let swiftDir = try Self.makeTempDirectory(prefix: "espresso-int-grad-swift")
        defer {
            try? FileManager.default.removeItem(at: objcDir)
            try? FileManager.default.removeItem(at: swiftDir)
        }

        let tokens = [UInt16](repeating: 0, count: ModelConfig.seqLen + 1)
        try Self.writeUInt16File(tokens, to: objcDir.appendingPathComponent("tinystories_data00.bin").path)
        try Self.writeUInt16File(tokens, to: swiftDir.appendingPathComponent("tinystories_data00.bin").path)

        // ObjC: force a compile-budget restart so a checkpoint is written at step 10.
        let objc = try Self.runProcess(
            executableURL: objcExe,
            arguments: ["--steps", "11"],
            currentDirectoryURL: objcDir,
            timeoutSeconds: 3_600
        )
        XCTAssertEqual(objc.status, 0, "ObjC process failed. stderr:\n\(objc.stderr)")

        // Swift: run exactly one accumulation batch (10 steps) so the final checkpoint is written at step 10.
        let swift = try Self.runProcess(
            executableURL: swiftExe,
            arguments: ["--steps", "10"],
            currentDirectoryURL: swiftDir,
            timeoutSeconds: 3_600
        )
        XCTAssertEqual(swift.status, 0, "Swift process failed. stderr:\n\(swift.stderr)")

        let objcCkpt = objcDir.appendingPathComponent("ane_stories110M_ckpt.bin").path
        let swiftCkpt = swiftDir.appendingPathComponent("ane_stories110M_ckpt.bin").path
        guard FileManager.default.fileExists(atPath: objcCkpt) else {
            XCTFail("Missing ObjC checkpoint at \(objcCkpt)")
            return
        }
        guard FileManager.default.fileExists(atPath: swiftCkpt) else {
            XCTFail("Missing Swift checkpoint at \(swiftCkpt)")
            return
        }

        func layerMNorms(path: String) throws -> [Double] {
            guard let f = fopen(path, "rb") else { throw NSError(domain: "EspressoTests.CkptOpen", code: 1) }
            defer { fclose(f) }

            var header = CheckpointHeader()
            let hsz = MemoryLayout<CheckpointHeader>.size
            let got = withUnsafeMutableBytes(of: &header) { raw -> Int in
                fread(raw.baseAddress, 1, raw.count, f)
            }
            guard got == hsz else { throw NSError(domain: "EspressoTests.CkptReadHeader", code: 1) }
            guard header.magic == 0x424C5A54, header.version == 2 else {
                throw NSError(domain: "EspressoTests.CkptBadHeader", code: 1)
            }
            guard header.step == 10 else { throw NSError(domain: "EspressoTests.CkptBadStep", code: Int(header.step)) }
            guard header.adamT == 1 else { throw NSError(domain: "EspressoTests.CkptBadAdamT", code: Int(header.adamT)) }

            let wq = ModelConfig.wqSize
            let wo = ModelConfig.woSize
            let w1 = ModelConfig.w1Size
            let w2 = ModelConfig.w2Size
            let w3 = ModelConfig.w3Size
            let dim = ModelConfig.dim

            @inline(__always)
            func skipFloats(_ count: Int) throws {
                let bytes = off_t(count) * off_t(MemoryLayout<Float>.stride)
                if fseeko(f, bytes, SEEK_CUR) != 0 {
                    throw NSError(domain: "EspressoTests.CkptSeek", code: 1)
                }
            }

            @inline(__always)
            func readSumSquares(_ count: Int) throws -> Double {
                let chunk = 16_384
                let buf = UnsafeMutablePointer<Float>.allocate(capacity: chunk)
                defer { buf.deallocate() }

                var remaining = count
                var sum: Double = 0
                while remaining > 0 {
                    let n = min(chunk, remaining)
                    let got = fread(buf, MemoryLayout<Float>.stride, n, f)
                    guard got == n else { throw NSError(domain: "EspressoTests.CkptRead", code: 1) }
                    for i in 0..<n {
                        let v = Double(buf[i])
                        sum += v * v
                    }
                    remaining -= n
                }
                return sum
            }

            let nLayers = Int(header.nLayers)
            var norms: [Double] = []
            norms.reserveCapacity(nLayers)

            for _ in 0..<nLayers {
                // Skip per-layer weights (9 segments).
                try skipFloats(wq) // Wq
                try skipFloats(wq) // Wk
                try skipFloats(wq) // Wv
                try skipFloats(wo) // Wo
                try skipFloats(w1) // W1
                try skipFloats(w2) // W2
                try skipFloats(w3) // W3
                try skipFloats(dim) // rmsAtt
                try skipFloats(dim) // rmsFfn

                // Adam: m then v for each parameter (same order as weights). Compute L2 norm of m only.
                var sumSquares: Double = 0

                sumSquares += try readSumSquares(wq); try skipFloats(wq) // Wq m/v
                sumSquares += try readSumSquares(wq); try skipFloats(wq) // Wk m/v
                sumSquares += try readSumSquares(wq); try skipFloats(wq) // Wv m/v
                sumSquares += try readSumSquares(wo); try skipFloats(wo) // Wo m/v
                sumSquares += try readSumSquares(w1); try skipFloats(w1) // W1 m/v
                sumSquares += try readSumSquares(w2); try skipFloats(w2) // W2 m/v
                sumSquares += try readSumSquares(w3); try skipFloats(w3) // W3 m/v
                sumSquares += try readSumSquares(dim); try skipFloats(dim) // rmsAtt m/v
                sumSquares += try readSumSquares(dim); try skipFloats(dim) // rmsFfn m/v

                norms.append(sqrt(sumSquares))
            }
            return norms
        }

        let objcNorms = try layerMNorms(path: objcCkpt)
        let swiftNorms = try layerMNorms(path: swiftCkpt)
        XCTAssertEqual(objcNorms.count, swiftNorms.count)

        for i in 0..<objcNorms.count {
            let a = objcNorms[i]
            let b = swiftNorms[i]
            let rel = abs(a - b) / max(1e-12, abs(a))
            XCTAssertLessThan(rel, 0.05, "Layer \(i) m-norm rel error too high: objc=\(a) swift=\(b) rel=\(rel)")
        }
    }

    // MARK: - Group 4D: Performance Test (ANE-gated)

    func test_100_steps_benchmark() throws {
        try requireANEHardwareTestsEnabled()
        try requirePerfTestsEnabled()

        let brand = Self.cpuBrandString() ?? "unknown"
        guard brand.contains("M4") else {
            throw XCTSkip("Perf target is M4; got \(brand)")
        }

        // Keep compilation budget deterministic across runs.
        try? CompileBudget.setCount(0)

        let dim = ModelConfig.dim
        let hidden = ModelConfig.hidden
        let seqLen = ModelConfig.seqLen
        let nLayers = ModelConfig.nLayers
        let vocab = ModelConfig.vocab

        // Fixed weights to avoid any variability from RNG.
        let weights = LayerStorage<LayerWeights>(count: nLayers) { _ in
            let w = LayerWeights()
            fillTensor(w.Wq, value: 0.01)
            fillTensor(w.Wk, value: 0.01)
            fillTensor(w.Wv, value: 0.01)
            fillTensor(w.Wo, value: 0.01)
            fillTensor(w.W1, value: 0.01)
            fillTensor(w.W2, value: 0.01)
            fillTensor(w.W3, value: 0.01)
            fillTensor(w.rmsAtt, value: 1.0)
            fillTensor(w.rmsFfn, value: 1.0)
            return w
        }

        let kernels = try LayerStorage<LayerKernelSet>(count: nLayers, throwingInitializer: { i in
            try LayerKernelSet(weights: weights[i])
        })
        let staticKernels = try LayerStorage<StaticKernel>(count: nLayers, throwingInitializer: { _ in
            try StaticKernel()
        })

        let acts = LayerStorage<LayerActivations>(count: nLayers) { _ in LayerActivations() }
        let grads = LayerStorage<LayerGradients>(count: nLayers) { _ in LayerGradients() }

        let rmsFinal = TensorBuffer(count: dim, zeroed: false)
        fillTensor(rmsFinal, value: 1.0)
        let embed = TensorBuffer(count: vocab * dim, zeroed: false)
        fillTensor(embed, value: 0.02)

        let grmsFinal = TensorBuffer(count: dim, zeroed: true)
        let gembed = TensorBuffer(count: vocab * dim, zeroed: true)

        let xCur = TensorBuffer(count: dim * seqLen, zeroed: false)
        let xFinal = TensorBuffer(count: dim * seqLen, zeroed: false)
        let logits = TensorBuffer(count: vocab * seqLen, zeroed: false)
        let dlogits = TensorBuffer(count: vocab * seqLen, zeroed: false)
        let dy = TensorBuffer(count: dim * seqLen, zeroed: false)
        let scratch = BackwardScratch(dim: dim, hidden: hidden, seqLen: seqLen)
        let rmsWorkspace = RMSNorm.Workspace(seqLen: seqLen)

        let accumulator = GradientAccumulator()

        var tb = mach_timebase_info_data_t()
        mach_timebase_info(&tb)

        @inline(__always)
        func ms(_ delta: UInt64) -> Double {
            let nanos = (Double(delta) * Double(tb.numer)) / Double(tb.denom)
            return nanos / 1_000_000.0
        }

        let tokens = [UInt16](repeating: 0, count: seqLen + 1)
        let nSteps = 100

        var stepTimes: [Double] = []
        stepTimes.reserveCapacity(nSteps)

        try tokens.withUnsafeBufferPointer { tokBuf in
            guard let base = tokBuf.baseAddress else { throw NSError(domain: "EspressoTests.Tokens", code: 1) }
            let inputTokens = base
            let targetTokens = base.advanced(by: 1)

            for _ in 0..<nSteps {
                let t0 = mach_absolute_time()

                // Embedding lookup.
                xCur.withUnsafeMutablePointer { xPtr in
                    embed.withUnsafePointer { ePtr in
                        Embedding.lookup(
                            output: xPtr,
                            embedding: ePtr,
                            tokens: inputTokens,
                            vocabSize: vocab,
                            dim: dim,
                            seqLen: seqLen
                        )
                    }
                }

                // Forward transformer.
                try ForwardPass.run(
                    xCur: xCur,
                    acts: acts,
                    kernels: kernels,
                    accumulator: accumulator,
                    dim: dim,
                    hidden: hidden,
                    seqLen: seqLen
                )

                // Final RMSNorm.
                xFinal.withUnsafeMutablePointer { outPtr in
                    xCur.withUnsafePointer { inPtr in
                        rmsFinal.withUnsafePointer { wPtr in
                            RMSNorm.forward(output: outPtr, input: inPtr, weights: wPtr, dim: dim, seqLen: seqLen)
                        }
                    }
                }

                // Classifier: logits = embed @ xFinal.
                logits.withUnsafeMutablePointer { logitsPtr in
                    embed.withUnsafePointer { ePtr in
                        xFinal.withUnsafePointer { xPtr in
                            BLAS.sgemm(
                                CblasRowMajor, CblasNoTrans, CblasNoTrans,
                                m: Int32(vocab), n: Int32(seqLen), k: Int32(dim),
                                alpha: 1.0,
                                a: ePtr, lda: Int32(dim),
                                b: xPtr, ldb: Int32(seqLen),
                                beta: 0.0,
                                c: logitsPtr, ldc: Int32(seqLen)
                            )
                        }
                    }
                }

                // Cross-entropy loss + dlogits.
                _ = dlogits.withUnsafeMutablePointer { dlogitsPtr in
                    logits.withUnsafePointer { logitsPtr in
                        CrossEntropy.lossAndGradient(
                            dlogits: dlogitsPtr,
                            logits: logitsPtr,
                            targets: targetTokens,
                            vocabSize: vocab,
                            seqLen: seqLen
                        )
                    }
                }

                // Classifier backward: dy = embed^T @ dlogits.
                dy.withUnsafeMutablePointer { dyPtr in
                    embed.withUnsafePointer { ePtr in
                        dlogits.withUnsafePointer { dlogitsPtr in
                            BLAS.sgemm(
                                CblasRowMajor, CblasTrans, CblasNoTrans,
                                m: Int32(dim), n: Int32(seqLen), k: Int32(vocab),
                                alpha: 1.0,
                                a: ePtr, lda: Int32(dim),
                                b: dlogitsPtr, ldb: Int32(seqLen),
                                beta: 0.0,
                                c: dyPtr, ldc: Int32(seqLen)
                            )
                        }
                    }
                }

                // dembed += dlogits @ xFinal^T (async, accumulate).
                let gembedPtr = gembed.withUnsafeMutablePointer { SendablePointer($0) }
                let captDlogits = dlogits.withUnsafePointer { SendableConstPointer($0) }
                let captXFinal = xFinal.withUnsafePointer { SendableConstPointer($0) }
                accumulator.enqueue { [captDlogits, captXFinal, gembedPtr] in
                    BLAS.sgemm(
                        CblasRowMajor, CblasNoTrans, CblasTrans,
                        m: Int32(vocab), n: Int32(dim), k: Int32(seqLen),
                        alpha: 1.0,
                        a: captDlogits.pointer, lda: Int32(seqLen),
                        b: captXFinal.pointer, ldb: Int32(seqLen),
                        beta: 1.0,
                        c: gembedPtr.pointer, ldc: Int32(dim)
                    )
                }

                // Final RMSNorm backward: dx -> dy, accumulate dw into grmsFinal.
                scratch.dxRms1.withUnsafeMutablePointer { dxPtr in
                    grmsFinal.withUnsafeMutablePointer { dwPtr in
                        dy.withUnsafePointer { dyPtr in
                            xCur.withUnsafePointer { xPtr in
                                rmsFinal.withUnsafePointer { wPtr in
                                    RMSNorm.backward(
                                        dx: dxPtr,
                                        dw: dwPtr,
                                        dy: dyPtr,
                                        x: xPtr,
                                        weights: wPtr,
                                        dim: dim,
                                        seqLen: seqLen
                                    )
                                }
                            }
                        }
                    }
                }
                dy.withUnsafeMutablePointer { dyPtr in
                    scratch.dxRms1.withUnsafePointer { dxPtr in
                        dyPtr.update(from: dxPtr, count: dim * seqLen)
                    }
                }

                // Backward transformer.
                try BackwardPass.run(
                    dy: dy,
                    acts: acts,
                    kernels: kernels,
                    staticKernels: staticKernels,
                    grads: grads,
                    weights: weights,
                    scratch: scratch,
                    rmsWorkspace: rmsWorkspace,
                    accumulator: accumulator,
                    dim: dim,
                    hidden: hidden,
                    seqLen: seqLen
                )

                // Embedding backward.
                accumulator.barrier()
                gembed.withUnsafeMutablePointer { gPtr in
                    dy.withUnsafePointer { dyPtr in
                        Embedding.backward(
                            dEmbedding: gPtr,
                            dx: dyPtr,
                            tokens: inputTokens,
                            vocabSize: vocab,
                            dim: dim,
                            seqLen: seqLen
                        )
                    }
                }

                let t1 = mach_absolute_time()
                stepTimes.append(ms(t1 - t0))
            }
        }

        accumulator.waitAll()

        // Use a warm-up window to stabilize caches and ANE scheduling.
        let warmup = min(10, stepTimes.count)
        let sample = stepTimes.dropFirst(warmup)
        let avg = sample.reduce(0.0, +) / Double(sample.count)

        XCTAssertLessThanOrEqual(avg, 9.3, "Avg step time too high: \(avg) ms/step over \(sample.count) steps")
    }
}
