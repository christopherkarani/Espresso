import XCTest
@testable import Espresso

final class InferenceKernelProfileTests: XCTestCase {
    func test_record_tracks_lock_unlock_and_host_overhead_fields() {
        var profile = InferenceKernelProfile(layerCount: 1, reservedSamplesPerLayer: 0)
        profile.record(
            layerIndex: 0,
            attnWriteUS: 12,
            attnWriteLockUS: 1,
            attnWriteBodyUS: 10,
            attnWriteUnlockUS: 1,
            attnEvalUS: 20,
            attnHwNS: 15_000,
            attnHostOverheadUS: 5,
            attnReadUS: 8,
            attnReadLockUS: 1,
            attnReadBodyUS: 6,
            attnReadUnlockUS: 1,
            ffnWriteUS: 9,
            ffnWriteLockUS: 1,
            ffnWriteBodyUS: 7,
            ffnWriteUnlockUS: 1,
            ffnCopyUS: 0,
            ffnEvalUS: 30,
            ffnHwNS: 22_000,
            ffnHostOverheadUS: 8,
            ffnReadUS: 11,
            ffnReadLockUS: 2,
            ffnReadBodyUS: 7,
            ffnReadUnlockUS: 2,
            gapAttnToFfnUS: 3
        )

        let layer = profile.layers[0]
        XCTAssertEqual(layer.attnWriteUS[0], 12, accuracy: 1e-9)
        XCTAssertEqual(layer.attnWriteLockUS[0], 1, accuracy: 1e-9)
        XCTAssertEqual(layer.attnWriteBodyUS[0], 10, accuracy: 1e-9)
        XCTAssertEqual(layer.attnWriteUnlockUS[0], 1, accuracy: 1e-9)
        XCTAssertEqual(layer.attnHostOverheadUS[0], 5, accuracy: 1e-9)
        XCTAssertEqual(layer.ffnWriteUS[0], 9, accuracy: 1e-9)
        XCTAssertEqual(layer.ffnWriteLockUS[0], 1, accuracy: 1e-9)
        XCTAssertEqual(layer.ffnWriteBodyUS[0], 7, accuracy: 1e-9)
        XCTAssertEqual(layer.ffnWriteUnlockUS[0], 1, accuracy: 1e-9)
        XCTAssertEqual(layer.ffnHostOverheadUS[0], 8, accuracy: 1e-9)
        XCTAssertEqual(layer.ffnReadUnlockUS[0], 2, accuracy: 1e-9)
    }

    func test_layerAverages_exposes_host_vs_hw_and_handoff_breakdown() {
        var profile = InferenceKernelProfile(layerCount: 1, reservedSamplesPerLayer: 2)
        profile.record(
            layerIndex: 0,
            attnWriteUS: 10,
            attnWriteLockUS: 1,
            attnWriteBodyUS: 8,
            attnWriteUnlockUS: 1,
            attnEvalUS: 40,
            attnHwNS: 30_000,
            attnHostOverheadUS: 10,
            attnReadUS: 12,
            attnReadLockUS: 2,
            attnReadBodyUS: 8,
            attnReadUnlockUS: 2,
            ffnWriteUS: 14,
            ffnWriteLockUS: 3,
            ffnWriteBodyUS: 8,
            ffnWriteUnlockUS: 3,
            ffnCopyUS: 0,
            ffnEvalUS: 50,
            ffnHwNS: 35_000,
            ffnHostOverheadUS: 15,
            ffnReadUS: 16,
            ffnReadLockUS: 4,
            ffnReadBodyUS: 8,
            ffnReadUnlockUS: 4,
            gapAttnToFfnUS: 5
        )
        profile.record(
            layerIndex: 0,
            attnWriteUS: 14,
            attnWriteLockUS: 3,
            attnWriteBodyUS: 8,
            attnWriteUnlockUS: 3,
            attnEvalUS: 44,
            attnHwNS: 36_000,
            attnHostOverheadUS: 8,
            attnReadUS: 10,
            attnReadLockUS: 2,
            attnReadBodyUS: 6,
            attnReadUnlockUS: 2,
            ffnWriteUS: 0,
            ffnWriteLockUS: 0,
            ffnWriteBodyUS: 0,
            ffnWriteUnlockUS: 0,
            ffnCopyUS: 9,
            ffnEvalUS: 46,
            ffnHwNS: 34_000,
            ffnHostOverheadUS: 12,
            ffnReadUS: 14,
            ffnReadLockUS: 2,
            ffnReadBodyUS: 10,
            ffnReadUnlockUS: 2,
            gapAttnToFfnUS: 7
        )

        let avg = profile.averageLayerMetrics(layerIndex: 0)
        XCTAssertEqual(avg.sampleCount, 2)
        XCTAssertEqual(avg.attnEvalUS, 42, accuracy: 1e-9)
        XCTAssertEqual(avg.attnHwUS, 33, accuracy: 1e-9)
        XCTAssertEqual(avg.attnHostOverheadUS, 9, accuracy: 1e-9)
        XCTAssertEqual(avg.attnIOLockUS, 4, accuracy: 1e-9)
        XCTAssertEqual(avg.attnIOBodyUS, 15, accuracy: 1e-9)
        XCTAssertEqual(avg.attnIOUnlockUS, 4, accuracy: 1e-9)
        XCTAssertEqual(avg.handoffCPUUS, 18, accuracy: 1e-9)
        XCTAssertEqual(avg.handoffFP16CopyUS, 4.5, accuracy: 1e-9)
        XCTAssertEqual(avg.ffnEvalUS, 48, accuracy: 1e-9)
        XCTAssertEqual(avg.ffnHwUS, 34.5, accuracy: 1e-9)
        XCTAssertEqual(avg.ffnHostOverheadUS, 13.5, accuracy: 1e-9)
        XCTAssertEqual(avg.ffnIOLockUS, 4.5, accuracy: 1e-9)
        XCTAssertEqual(avg.ffnIOBodyUS, 13, accuracy: 1e-9)
        XCTAssertEqual(avg.ffnIOUnlockUS, 4.5, accuracy: 1e-9)
        XCTAssertEqual(avg.gapAttnToFfnUS, 6, accuracy: 1e-9)
    }
}
