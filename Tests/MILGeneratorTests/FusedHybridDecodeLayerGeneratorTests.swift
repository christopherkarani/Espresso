import XCTest
import ANETypes
@testable import MILGenerator

final class FusedHybridDecodeLayerGeneratorTests: XCTestCase {
    func test_qwen15b_n1_serving_graph_has_attention_rope_wo_and_ios18() {
        let gen = FusedHybridDecodeLayerGenerator.qwen15B(maxSeq: 128)
        let mil = gen.milText

        XCTAssertTrue(mil.contains("program(1.3)"))
        XCTAssertTrue(mil.contains("func main<ios18>"))
        XCTAssertTrue(mil.contains("tensor<fp16, [1, 1536, 1, 32]> x"))
        XCTAssertTrue(mil.contains("tensor<fp16, [1, 1536, 1, 128]> kCache"))
        XCTAssertTrue(mil.contains("tensor<fp16, [1, 1536, 1, 128]> vCache"))
        XCTAssertTrue(mil.contains("wo.bin"), "Serving N=1 includes Wo; Phase 11 probe omitted it")
        XCTAssertTrue(mil.contains("bq.bin"))
        XCTAssertTrue(mil.contains("sigmoid"), "SiLU default stays sigmoid")
        XCTAssertFalse(mil.contains("tanh"), "Tanh SiLU identity must stay opt-in")
        XCTAssertTrue(mil.contains("softmax"), "Attention lives inside the N=1 graph")
        XCTAssertEqual(gen.inputByteSizes.count, 6)
        XCTAssertEqual(gen.outputByteSizes.count, 3)
        XCTAssertEqual(FusedHybridDecodeLayerGenerator.hopsPerToken(nLayer: 28), 28)
        XCTAssertEqual(FusedHybridDecodeLayerGenerator.phase11MaxN, 1)
    }

    func test_input_and_output_names_are_alphabetical() {
        let mil = FusedHybridDecodeLayerGenerator.qwen15B(maxSeq: 64).milText
        let header = mil.split(separator: "\n").first { $0.contains("func main<ios18>") }.map(String.init) ?? mil
        let names = ["kCache", "mask", "posMask", "ropePack", "vCache", "x"]
        let positions = names.compactMap { header.range(of: $0)?.lowerBound }
        XCTAssertEqual(positions.count, names.count, header)
        for (lhs, rhs) in zip(positions, positions.dropFirst()) {
            XCTAssertLessThan(lhs, rhs, header)
        }
        XCTAssertTrue(mil.contains("} -> (kNew,vNew,xOut);") || mil.contains("-> (kNew, vNew, xOut)"), mil)
    }

    func test_ssa_names_are_unique() {
        let mil = FusedHybridDecodeLayerGenerator.qwen15B(maxSeq: 32).milText
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
        XCTAssertEqual(names.count, Set(names).count, "Duplicate SSA names in serving N=1 graph")
    }
}
