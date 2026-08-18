import XCTest
import ANETypes
@testable import MILGenerator

final class FusedHybridDecodeBlockGeneratorTests: XCTestCase {
    func test_qwen15b_n1_emits_ios18_qkv_bias_and_swiglu_ffn() {
        let gen = FusedHybridDecodeBlockGenerator.qwen15B(layerCount: 1)
        let mil = gen.milText

        XCTAssertTrue(mil.contains("program(1.3)"))
        XCTAssertTrue(mil.contains("func main<ios18>"))
        XCTAssertTrue(mil.contains("tensor<fp16, [1, 1536, 1, 32]> x"))
        XCTAssertTrue(mil.contains("tensor<fp16, [1536, 1536, 1, 1]>"))
        XCTAssertTrue(mil.contains("tensor<fp16, [256, 1536, 1, 1]>"))
        XCTAssertTrue(mil.contains("tensor<fp16, [8960, 1536, 1, 1]>"))
        XCTAssertTrue(mil.contains("l0_rms1.bin"))
        XCTAssertTrue(mil.contains("l0_wq.bin"))
        XCTAssertTrue(mil.contains("l0_wk.bin"))
        XCTAssertTrue(mil.contains("l0_wv.bin"))
        XCTAssertTrue(mil.contains("l0_bq.bin"))
        XCTAssertTrue(mil.contains("l0_bk.bin"))
        XCTAssertTrue(mil.contains("l0_bv.bin"))
        XCTAssertTrue(mil.contains("l0_rms2.bin"))
        XCTAssertTrue(mil.contains("l0_w1.bin"))
        XCTAssertTrue(mil.contains("l0_w3.bin"))
        XCTAssertTrue(mil.contains("l0_w2.bin"))
        XCTAssertFalse(mil.contains("wo.bin"), "Fusion is QKV+FFN only; Wo stays on Metal")
        XCTAssertTrue(mil.contains("sigmoid"), "SiLU default stays sigmoid")
        XCTAssertFalse(mil.contains("tanh"), "Tanh SiLU identity must stay opt-in")
        XCTAssertEqual(gen.inputByteSizes, [1536 * 32 * 2])
        XCTAssertEqual(gen.outputByteSizes, [1536 * 32 * 2])
    }

    func test_qwen15b_n2_emits_two_layer_weight_prefixes() {
        let gen = FusedHybridDecodeBlockGenerator.qwen15B(layerCount: 2)
        let mil = gen.milText

        XCTAssertTrue(mil.contains("l0_w1.bin"))
        XCTAssertTrue(mil.contains("l1_w1.bin"))
        XCTAssertTrue(mil.contains("l0_bq.bin"))
        XCTAssertTrue(mil.contains("l1_bq.bin"))
    }

    func test_ssa_names_are_unique_for_n4() {
        let mil = FusedHybridDecodeBlockGenerator.qwen15B(layerCount: 4).milText
        var names: [String] = []
        var scanner = mil[mil.startIndex...]
        let namePrefix = "name=string(\""
        let nameSuffix = "\")"
        while let range = scanner.range(of: namePrefix) {
            let afterPrefix = range.upperBound
            if let endRange = scanner[afterPrefix...].range(of: nameSuffix) {
                names.append(String(scanner[afterPrefix..<endRange.lowerBound]))
                scanner = scanner[endRange.upperBound...]
            } else {
                break
            }
        }
        XCTAssertEqual(names.count, Set(names).count, "Duplicate SSA names in N=4 fused block")
    }
}
