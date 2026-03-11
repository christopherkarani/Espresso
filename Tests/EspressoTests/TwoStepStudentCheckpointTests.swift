import XCTest
import ANETypes
@testable import Espresso

final class TwoStepStudentCheckpointTests: XCTestCase {
    private static func makeTempBinaryPath(prefix: String) -> String {
        let dir = FileManager.default.temporaryDirectory
        return dir.appendingPathComponent("\(prefix)-\(UUID().uuidString).bin").path
    }

    private static func fillRamp(_ buffer: borrowing TensorBuffer, base: Float) {
        buffer.withUnsafeMutableBufferPointer { ptr in
            for idx in 0..<ptr.count {
                ptr[idx] = base + Float(idx) * 0.125
            }
        }
    }

    private static func floats(from buffer: borrowing TensorBuffer) -> [Float] {
        buffer.withUnsafeBufferPointer { Array($0) }
    }

    func test_two_step_student_sidecar_roundtrip_preserves_contract_and_weights() throws {
        let path = Self.makeTempBinaryPath(prefix: "two-step-student-roundtrip")
        defer { try? FileManager.default.removeItem(atPath: path) }

        let contract = try TwoStepStudentContract(
            dim: 4,
            vocabSize: 6,
            layerCount: 3,
            teacherClassifierWasShared: true
        )
        let futureRMS = TensorBuffer(count: contract.dim, zeroed: false)
        let futureClassifier = TensorBuffer(count: contract.dim * contract.vocabSize, zeroed: false)
        Self.fillRamp(futureRMS, base: 1.0)
        Self.fillRamp(futureClassifier, base: 10.0)
        let expectedRMS = Self.floats(from: futureRMS)
        let expectedClassifier = Self.floats(from: futureClassifier)

        let sidecar = try TwoStepStudentSidecar(
            contract: contract,
            futureRMS: futureRMS,
            futureClassifier: futureClassifier
        )
        try TwoStepStudentCheckpoint.save(path: path, sidecar: sidecar)

        let loaded = try TwoStepStudentCheckpoint.load(path: path, expectedContract: contract)
        XCTAssertEqual(loaded.contract, contract)
        XCTAssertEqual(Self.floats(from: loaded.futureRMS), expectedRMS)
        XCTAssertEqual(Self.floats(from: loaded.futureClassifier), expectedClassifier)
    }

    func test_two_step_student_sidecar_seeds_future_classifier_from_shared_teacher_embedding() throws {
        let teacherRMS = TensorBuffer(count: 4, zeroed: false)
        let teacherEmbedding = TensorBuffer(count: 24, zeroed: false)
        let teacherClassifier = TensorBuffer(count: 0, zeroed: true)
        Self.fillRamp(teacherRMS, base: 2.0)
        Self.fillRamp(teacherEmbedding, base: 20.0)

        let sidecar = try TwoStepStudentCheckpoint.seedFromTeacher(
            dim: 4,
            vocabSize: 6,
            layerCount: 3,
            teacherRMS: teacherRMS,
            teacherEmbedding: teacherEmbedding,
            teacherClassifier: teacherClassifier,
            teacherClassifierWasShared: true
        )

        XCTAssertEqual(sidecar.contract.teacherClassifierWasShared, true)
        XCTAssertEqual(Self.floats(from: sidecar.futureRMS), Self.floats(from: teacherRMS))
        XCTAssertEqual(Self.floats(from: sidecar.futureClassifier), Self.floats(from: teacherEmbedding))
    }

    func test_two_step_student_sidecar_seeds_future_classifier_from_unshared_teacher_classifier() throws {
        let teacherRMS = TensorBuffer(count: 4, zeroed: false)
        let teacherEmbedding = TensorBuffer(count: 24, zeroed: false)
        let teacherClassifier = TensorBuffer(count: 24, zeroed: false)
        Self.fillRamp(teacherRMS, base: 3.0)
        Self.fillRamp(teacherEmbedding, base: 30.0)
        Self.fillRamp(teacherClassifier, base: 40.0)

        let sidecar = try TwoStepStudentCheckpoint.seedFromTeacher(
            dim: 4,
            vocabSize: 6,
            layerCount: 3,
            teacherRMS: teacherRMS,
            teacherEmbedding: teacherEmbedding,
            teacherClassifier: teacherClassifier,
            teacherClassifierWasShared: false
        )

        XCTAssertEqual(sidecar.contract.teacherClassifierWasShared, false)
        XCTAssertEqual(Self.floats(from: sidecar.futureRMS), Self.floats(from: teacherRMS))
        XCTAssertEqual(Self.floats(from: sidecar.futureClassifier), Self.floats(from: teacherClassifier))
    }

    func test_two_step_student_sidecar_load_rejects_contract_mismatch() throws {
        let path = Self.makeTempBinaryPath(prefix: "two-step-student-mismatch")
        defer { try? FileManager.default.removeItem(atPath: path) }

        let contract = try TwoStepStudentContract(
            dim: 4,
            vocabSize: 6,
            layerCount: 3,
            teacherClassifierWasShared: true
        )
        let futureRMS = TensorBuffer(count: contract.dim, zeroed: false)
        let futureClassifier = TensorBuffer(count: contract.dim * contract.vocabSize, zeroed: false)
        Self.fillRamp(futureRMS, base: 1.0)
        Self.fillRamp(futureClassifier, base: 10.0)

        let sidecar = try TwoStepStudentSidecar(
            contract: contract,
            futureRMS: futureRMS,
            futureClassifier: futureClassifier
        )
        try TwoStepStudentCheckpoint.save(path: path, sidecar: sidecar)

        let mismatched = try TwoStepStudentContract(
            dim: 5,
            vocabSize: 6,
            layerCount: 3,
            teacherClassifierWasShared: true
        )

        do {
            _ = try TwoStepStudentCheckpoint.load(path: path, expectedContract: mismatched)
            XCTFail("Expected config mismatch")
        } catch let error as TwoStepStudentCheckpointError {
            guard case let .configMismatch(expected, got) = error else {
                return XCTFail("Unexpected error: \(error)")
            }
            XCTAssertTrue(expected.contains("dim=5"))
            XCTAssertTrue(got.contains("dim=4"))
        }
    }
}
