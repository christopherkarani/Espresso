import XCTest
@testable import Espresso

final class HybridDecodeTimingBreakdownTests: XCTestCase {
    func test_reset_zeros_rope_and_lm_head_with_existing_buckets() {
        var timings = HybridDecodeTimingBreakdown(
            tAneQKV: 1,
            tRoPE: 2,
            tMetal: 3,
            tAneFFN: 4,
            tLMHead: 5,
            tIO: 6
        )

        timings.reset()

        XCTAssertEqual(timings.tAneQKV, 0)
        XCTAssertEqual(timings.tRoPE, 0)
        XCTAssertEqual(timings.tMetal, 0)
        XCTAssertEqual(timings.tAneFFN, 0)
        XCTAssertEqual(timings.tLMHead, 0)
        XCTAssertEqual(timings.tIO, 0)
    }

    func test_token_profile_mean_excludes_ttft_and_formats_named_buckets() {
        let ttft = HybridDecodeTimingBreakdown(tLMHead: 90)
        let steady = HybridDecodeTimingBreakdown(
            tAneQKV: 100,
            tRoPE: 50,
            tMetal: 200,
            tAneFFN: 180,
            tLMHead: 60,
            tIO: 36
        )
        let profile = HybridDecodeTokenProfile(tokens: [ttft, steady, steady])

        let mean = profile.meanExcludingFirst
        XCTAssertEqual(mean.tAneQKV, 100, accuracy: 1e-9)
        XCTAssertEqual(mean.tRoPE, 50, accuracy: 1e-9)
        XCTAssertEqual(mean.tMetal, 200, accuracy: 1e-9)
        XCTAssertEqual(mean.tAneFFN, 180, accuracy: 1e-9)
        XCTAssertEqual(mean.tLMHead, 60, accuracy: 1e-9)
        XCTAssertEqual(mean.tIO, 36, accuracy: 1e-9)
        XCTAssertEqual(mean.totalMs, 626, accuracy: 1e-9)

        let report = profile.formatReport()
        XCTAssertTrue(report.contains("decode_profile_mean_ms/token"), report)
        XCTAssertTrue(report.contains("qkv=100.00"), report)
        XCTAssertTrue(report.contains("rope=50.00"), report)
        XCTAssertTrue(report.contains("attn=200.00"), report)
        XCTAssertTrue(report.contains("ffn=180.00"), report)
        XCTAssertTrue(report.contains("lm_head=60.00"), report)
        XCTAssertTrue(report.contains("io=36.00"), report)
        XCTAssertTrue(report.contains("n=2"), report)
        XCTAssertTrue(report.contains("decode_profile_ttft_ms=90.00"), report)
        XCTAssertTrue(report.contains("decode_profile_token i=0"), report)
        XCTAssertTrue(report.contains("decode_profile_token i=1"), report)
        XCTAssertTrue(report.contains("decode_profile_token i=2"), report)
    }
}
