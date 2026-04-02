import Darwin
import Foundation

public enum ESPBenchmarkThermalState: String, Sendable, Codable, CaseIterable {
    case nominal
    case fair
    case serious
    case critical
    case unknown

    public static func current(processInfo: ProcessInfo = .processInfo) -> ESPBenchmarkThermalState {
        switch processInfo.thermalState {
        case .nominal:
            .nominal
        case .fair:
            .fair
        case .serious:
            .serious
        case .critical:
            .critical
        @unknown default:
            .unknown
        }
    }

    public var permitsBenchmarkStart: Bool {
        switch self {
        case .nominal, .fair, .unknown:
            true
        case .serious, .critical:
            false
        }
    }
}

public enum ESPBenchmarkGuardError: Error, Equatable, LocalizedError {
    case benchmarkAlreadyRunning(lockPath: String)
    case thermalStateTooHigh(state: String)
    case lockOpenFailed(lockPath: String, errno: Int32)

    public var errorDescription: String? {
        switch self {
        case let .benchmarkAlreadyRunning(lockPath):
            return "Another ANE benchmark is already running. Lock path: \(lockPath)"
        case let .thermalStateTooHigh(state):
            return "Benchmark start rejected because the thermal state is \(state). Wait for the machine to cool down."
        case let .lockOpenFailed(lockPath, code):
            let reason = String(cString: strerror(code))
            return "Failed to open benchmark lock \(lockPath): \(reason)"
        }
    }
}

public final class ESPBenchmarkExecutionGuard {
    public static let defaultLockPath = "/tmp/espresso-ane-benchmark.lock"

    public let lockPath: String
    public let thermalBefore: String

    private let fileDescriptor: Int32

    public init(
        lockPath: String = ESPBenchmarkExecutionGuard.defaultLockPath,
        processInfo: ProcessInfo = .processInfo,
        fileManager: FileManager = .default
    ) throws {
        self.lockPath = lockPath

        if !fileManager.fileExists(atPath: lockPath) {
            fileManager.createFile(atPath: lockPath, contents: Data(), attributes: nil)
        }

        let descriptor = open(lockPath, O_CREAT | O_RDWR, S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH)
        guard descriptor >= 0 else {
            throw ESPBenchmarkGuardError.lockOpenFailed(lockPath: lockPath, errno: errno)
        }

        if flock(descriptor, LOCK_EX | LOCK_NB) != 0 {
            let code = errno
            close(descriptor)
            if code == EWOULDBLOCK {
                throw ESPBenchmarkGuardError.benchmarkAlreadyRunning(lockPath: lockPath)
            }
            throw ESPBenchmarkGuardError.lockOpenFailed(lockPath: lockPath, errno: code)
        }

        fileDescriptor = descriptor

        let thermalState = ESPBenchmarkThermalState.current(processInfo: processInfo)
        thermalBefore = thermalState.rawValue
        guard thermalState.permitsBenchmarkStart else {
            flock(fileDescriptor, LOCK_UN)
            close(fileDescriptor)
            throw ESPBenchmarkGuardError.thermalStateTooHigh(state: thermalState.rawValue)
        }
    }

    deinit {
        flock(fileDescriptor, LOCK_UN)
        close(fileDescriptor)
    }

    public func thermalAfter(processInfo: ProcessInfo = .processInfo) -> String {
        ESPBenchmarkThermalState.current(processInfo: processInfo).rawValue
    }
}
