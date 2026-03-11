import XCTest
@testable import Espresso

final class DecodeStateTests: XCTestCase {
    func test_decode_state_rejects_non_positive_max_seq() {
        XCTAssertThrowsError(try DecodeState(maxSeq: 0))
        XCTAssertThrowsError(try DecodeState(maxSeq: -4))
    }

    func test_decode_state_step_visibility_and_commit_order() throws {
        var state = try DecodeState(maxSeq: 4)
        XCTAssertEqual(state.step, 0)
        XCTAssertEqual(state.visibleTokenCount, 0)

        let t0 = try state.beginTokenStep()
        XCTAssertEqual(t0, 0)
        XCTAssertEqual(state.visibleTokenCount, 0)
        try state.commitTokenStep(expectedIndex: t0)
        XCTAssertEqual(state.step, 1)
        XCTAssertEqual(state.visibleTokenCount, 1)

        let t1 = try state.beginTokenStep()
        XCTAssertEqual(t1, 1)
        XCTAssertEqual(state.visibleTokenCount, 1)
        try state.commitTokenStep(expectedIndex: t1)
        XCTAssertEqual(state.step, 2)
        XCTAssertEqual(state.visibleTokenCount, 2)
    }

    func test_decode_state_detects_overflow_and_mismatched_commit() throws {
        var overflow = try DecodeState(maxSeq: 1)
        let t0 = try overflow.beginTokenStep()
        try overflow.commitTokenStep(expectedIndex: t0)
        XCTAssertEqual(overflow.step, 1)
        XCTAssertThrowsError(try overflow.beginTokenStep())

        var mismatch = try DecodeState(maxSeq: 2)
        XCTAssertThrowsError(try mismatch.commitTokenStep(expectedIndex: 1))
    }

    func test_decode_tiling_window_base_and_local_index() {
        let lane = 32
        XCTAssertEqual(DecodeTiling.windowBase(for: 0, laneSpatial: lane), 0)
        XCTAssertEqual(DecodeTiling.windowBase(for: 31, laneSpatial: lane), 0)
        XCTAssertEqual(DecodeTiling.windowBase(for: 32, laneSpatial: lane), 32)
        XCTAssertEqual(DecodeTiling.windowBase(for: 63, laneSpatial: lane), 32)
        XCTAssertEqual(DecodeTiling.windowBase(for: 64, laneSpatial: lane), 64)

        XCTAssertEqual(DecodeTiling.localIndex(for: 0, laneSpatial: lane), 0)
        XCTAssertEqual(DecodeTiling.localIndex(for: 31, laneSpatial: lane), 31)
        XCTAssertEqual(DecodeTiling.localIndex(for: 32, laneSpatial: lane), 0)
        XCTAssertEqual(DecodeTiling.localIndex(for: 63, laneSpatial: lane), 31)
        XCTAssertEqual(DecodeTiling.localIndex(for: 64, laneSpatial: lane), 0)
    }

    func test_decode_tiling_covers_full_range_for_logical_max_seq() {
        let lane = 32
        let maxSeq = lane * 4

        for tokenIndex in 0..<maxSeq {
            let base = DecodeTiling.windowBase(for: tokenIndex, laneSpatial: lane)
            let local = DecodeTiling.localIndex(for: tokenIndex, laneSpatial: lane)
            XCTAssertGreaterThanOrEqual(base, 0)
            XCTAssertEqual(base % lane, 0)
            XCTAssertGreaterThanOrEqual(local, 0)
            XCTAssertLessThan(local, lane)
            XCTAssertEqual(base + local, tokenIndex)
        }
    }

    func test_decode_tiling_window_sync_only_on_tile_boundaries_after_first_token() {
        let lane = 32
        XCTAssertFalse(DecodeTiling.shouldSyncWindow(for: 0, laneSpatial: lane))
        XCTAssertFalse(DecodeTiling.shouldSyncWindow(for: 1, laneSpatial: lane))
        XCTAssertFalse(DecodeTiling.shouldSyncWindow(for: 31, laneSpatial: lane))
        XCTAssertTrue(DecodeTiling.shouldSyncWindow(for: 32, laneSpatial: lane))
        XCTAssertFalse(DecodeTiling.shouldSyncWindow(for: 33, laneSpatial: lane))
        XCTAssertFalse(DecodeTiling.shouldSyncWindow(for: 63, laneSpatial: lane))
        XCTAssertTrue(DecodeTiling.shouldSyncWindow(for: 64, laneSpatial: lane))
    }

    func test_decode_tiling_window_sync_pattern_matches_modulo_rule() {
        let lane = 32
        let maxSeq = lane * 8
        for tokenIndex in 0..<maxSeq {
            let expected = tokenIndex > 0 && tokenIndex % lane == 0
            XCTAssertEqual(
                DecodeTiling.shouldSyncWindow(for: tokenIndex, laneSpatial: lane),
                expected,
                "unexpected sync decision at token \(tokenIndex)"
            )
        }
    }

    func test_decode_runtime_options_force_full_window_sync_defaults_off() {
        XCTAssertFalse(DecodeRuntimeOptions.forceFullWindowSync(env: [:]))
        XCTAssertFalse(DecodeRuntimeOptions.forceFullWindowSync(env: ["ESPRESSO_DECODE_FORCE_FULL_WINDOW_SYNC": "0"]))
        XCTAssertFalse(DecodeRuntimeOptions.forceFullWindowSync(env: ["ESPRESSO_DECODE_FORCE_FULL_WINDOW_SYNC": "true"]))
    }

    func test_decode_runtime_options_force_full_window_sync_enabled_by_one() {
        XCTAssertTrue(DecodeRuntimeOptions.forceFullWindowSync(env: ["ESPRESSO_DECODE_FORCE_FULL_WINDOW_SYNC": "1"]))
    }
}
