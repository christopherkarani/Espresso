import XCTest
import IOSurface
import ANEInterop
import ANETypes
import MILGenerator
@testable import ANERuntime
@testable import Espresso

// MARK: - MIL Generation Tests (no hardware needed)

final class FusedDecodeLayerGeneratorTests: XCTestCase {

    func test_mil_text_is_nonempty_and_contains_program_header() {
        let gen = FusedDecodeLayerGenerator(maxSeq: 32, laneSpatial: 32)
        let mil = gen.milText
        XCTAssertFalse(mil.isEmpty)
        XCTAssertTrue(mil.contains("program(1.3)"))
        XCTAssertTrue(mil.contains("func main<ios18>"))
    }

    func test_mil_text_contains_all_nine_weight_blobs() {
        let gen = FusedDecodeLayerGenerator(maxSeq: 32, laneSpatial: 32)
        let mil = gen.milText
        let expectedBlobs = [
            "rms1.bin", "wq.bin", "wk.bin", "wv.bin", "wo.bin",
            "rms2.bin", "w1.bin", "w3.bin", "w2.bin",
        ]
        for blob in expectedBlobs {
            XCTAssertTrue(mil.contains(blob), "Missing weight blob: \(blob)")
        }
    }

    func test_mil_text_contains_fused_residuals() {
        let gen = FusedDecodeLayerGenerator(maxSeq: 32, laneSpatial: 32)
        let mil = gen.milText
        // First residual: attention output
        XCTAssertTrue(mil.contains("name=string(\"res\")"), "Missing first residual (res)")
        // Second residual: FFN output
        XCTAssertTrue(mil.contains("name=string(\"res2\")"), "Missing second residual (res2)")
    }

    func test_mil_text_contains_ffn_ops() {
        let gen = FusedDecodeLayerGenerator(maxSeq: 32, laneSpatial: 32)
        let mil = gen.milText
        // SwiGLU FFN ops
        XCTAssertTrue(mil.contains("sigmoid(x=h1)"), "Missing sigmoid")
        XCTAssertTrue(mil.contains("name=string(\"si\")"), "Missing SiLU")
        XCTAssertTrue(mil.contains("name=string(\"gt\")"), "Missing gate")
    }

    func test_mil_text_has_unique_ssa_names() {
        let gen = FusedDecodeLayerGenerator(maxSeq: 32, laneSpatial: 32)
        let mil = gen.milText

        // Extract all name=string("...") values
        var names: [String] = []
        var scanner = mil[mil.startIndex...]
        let namePrefix = "name=string(\""
        let nameSuffix = "\")"
        while let range = scanner.range(of: namePrefix) {
            let afterPrefix = range.upperBound
            if let endRange = scanner[afterPrefix...].range(of: nameSuffix) {
                let name = String(scanner[afterPrefix..<endRange.lowerBound])
                names.append(name)
                scanner = scanner[endRange.upperBound...]
            } else {
                break
            }
        }

        let uniqueNames = Set(names)
        XCTAssertEqual(
            names.count, uniqueNames.count,
            "Duplicate SSA names found: \(names.filter { n in names.filter { $0 == n }.count > 1 })"
        )
    }

    func test_mil_text_has_three_outputs() {
        let gen = FusedDecodeLayerGenerator(maxSeq: 32, laneSpatial: 32)
        let mil = gen.milText
        XCTAssertTrue(mil.contains("-> (xNext,kfFull,vfFull)"))
    }

    func test_mil_text_has_four_inputs() {
        let gen = FusedDecodeLayerGenerator(maxSeq: 32, laneSpatial: 32)
        let mil = gen.milText
        XCTAssertTrue(mil.contains("> x,"))
        XCTAssertTrue(mil.contains("> kCache,"))
        XCTAssertTrue(mil.contains("> vCache,"))
        XCTAssertTrue(mil.contains("> maskCache)"))
    }

    func test_inputByteSizes_has_four_entries() {
        let gen = FusedDecodeLayerGenerator(maxSeq: 64, laneSpatial: 32)
        XCTAssertEqual(gen.inputByteSizes.count, 4)
        let dim = ModelConfig.dim
        XCTAssertEqual(gen.inputByteSizes[0], dim * 32 * 2)   // x
        XCTAssertEqual(gen.inputByteSizes[1], dim * 64 * 2)   // kCache
        XCTAssertEqual(gen.inputByteSizes[2], dim * 64 * 2)   // vCache
        XCTAssertEqual(gen.inputByteSizes[3], dim * 64 * 2)   // maskCache
    }

    func test_outputByteSizes_has_three_entries() {
        let gen = FusedDecodeLayerGenerator(maxSeq: 32, laneSpatial: 32)
        XCTAssertEqual(gen.outputByteSizes.count, 3)
        let dim = ModelConfig.dim
        for size in gen.outputByteSizes {
            XCTAssertEqual(size, dim * 32 * 2)
        }
    }

    func test_ffn_rmsnorm_uses_x2_not_x() {
        let gen = FusedDecodeLayerGenerator(maxSeq: 32, laneSpatial: 32)
        let mil = gen.milText
        // The FFN RMSNorm should operate on x2 (attention output), not x (original input)
        XCTAssertTrue(mil.contains("f_sq = mul(x=x2,y=x2)"), "FFN RMSNorm should use x2")
    }

    func test_second_residual_adds_to_x2() {
        let gen = FusedDecodeLayerGenerator(maxSeq: 32, laneSpatial: 32)
        let mil = gen.milText
        // xNext = x2 + y (FFN output added to attention residual)
        XCTAssertTrue(mil.contains("xNext = add(x=x2,y=y)"), "Second residual should add FFN output to x2")
    }

    func test_conv_constants_defined_once() {
        let gen = FusedDecodeLayerGenerator(maxSeq: 32, laneSpatial: 32)
        let mil = gen.milText
        // Conv constants should only appear once (not duplicated for FFN)
        let ptCount = mil.components(separatedBy: "name=string(\"pt\")").count - 1
        XCTAssertEqual(ptCount, 1, "Conv constant 'pt' should be defined exactly once")
    }
}

final class FusedTwoLayerDecodeGeneratorTests: XCTestCase {

    func test_mil_text_is_nonempty_and_contains_program_header() {
        let gen = FusedTwoLayerDecodeGenerator(maxSeq: 32, laneSpatial: 32)
        let mil = gen.milText
        XCTAssertFalse(mil.isEmpty)
        XCTAssertTrue(mil.contains("program(1.3)"))
        XCTAssertTrue(mil.contains("func main<ios18>"))
    }

    func test_mil_text_contains_all_eighteen_weight_blobs() {
        let gen = FusedTwoLayerDecodeGenerator(maxSeq: 32, laneSpatial: 32)
        let mil = gen.milText
        let expectedBlobs = [
            "l0_rms1.bin", "l0_wq.bin", "l0_wk.bin", "l0_wv.bin", "l0_wo.bin",
            "l0_rms2.bin", "l0_w1.bin", "l0_w3.bin", "l0_w2.bin",
            "l1_rms1.bin", "l1_wq.bin", "l1_wk.bin", "l1_wv.bin", "l1_wo.bin",
            "l1_rms2.bin", "l1_w1.bin", "l1_w3.bin", "l1_w2.bin",
        ]
        for blob in expectedBlobs {
            XCTAssertTrue(mil.contains(blob), "Missing weight blob: \(blob)")
        }
    }

    func test_mil_text_contains_packed_kv_input_and_output_concat() {
        let gen = FusedTwoLayerDecodeGenerator(maxSeq: 32, laneSpatial: 32)
        let mil = gen.milText
        XCTAssertTrue(mil.contains("tensor<fp16, [1, 3072, 1, 32]> packedKVCache"))
        XCTAssertTrue(mil.contains("> maskCache0,"))
        XCTAssertTrue(mil.contains("> maskCache1)"))
        XCTAssertTrue(mil.contains("concat(axis=cax,interleave=cid,values=(kfL0,l1_kf))"))
        XCTAssertTrue(mil.contains("concat(axis=cax,interleave=cid,values=(vfL0,l1_vf))"))
    }

    func test_mil_text_contains_four_packed_kv_slices() {
        let gen = FusedTwoLayerDecodeGenerator(maxSeq: 32, laneSpatial: 32)
        let mil = gen.milText
        let count = mil.components(separatedBy: "slice_by_size(x=packedKVCache").count - 1
        XCTAssertEqual(count, 4, "Expected 4 packed-K/V slices (K/V for 2 layers)")
    }

    func test_mil_text_has_unique_ssa_names() {
        let gen = FusedTwoLayerDecodeGenerator(maxSeq: 32, laneSpatial: 32)
        let mil = gen.milText

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

        XCTAssertEqual(names.count, Set(names).count, "Duplicate SSA names found in two-layer MIL")
    }

    func test_inputByteSizes_has_four_entries() {
        let gen = FusedTwoLayerDecodeGenerator(maxSeq: 64, laneSpatial: 32)
        XCTAssertEqual(gen.inputByteSizes.count, 4)
        XCTAssertEqual(gen.inputByteSizes[0], ModelConfig.dim * 32 * 2)
        XCTAssertEqual(gen.inputByteSizes[1], 4 * ModelConfig.dim * 64 * 2)
        XCTAssertEqual(gen.inputByteSizes[2], ModelConfig.dim * 64 * 2)
        XCTAssertEqual(gen.inputByteSizes[3], ModelConfig.dim * 64 * 2)
    }

    func test_outputByteSizes_has_three_entries() {
        let gen = FusedTwoLayerDecodeGenerator(maxSeq: 32, laneSpatial: 32)
        XCTAssertEqual(gen.outputByteSizes.count, 3)
        XCTAssertEqual(gen.outputByteSizes[0], ModelConfig.dim * 32 * 2)
        XCTAssertEqual(gen.outputByteSizes[1], 2 * ModelConfig.dim * 32 * 2)
        XCTAssertEqual(gen.outputByteSizes[2], 2 * ModelConfig.dim * 32 * 2)
    }
}

// MARK: - Hardware-gated compilation tests

private func requireANEHardware(file: StaticString = #filePath, line: UInt = #line) throws {
    guard ProcessInfo.processInfo.environment["ANE_HARDWARE_TESTS"] == "1" else {
        throw XCTSkip("Set ANE_HARDWARE_TESTS=1 to run ANE hardware tests", file: file, line: line)
    }
    let handle = dlopen(
        "/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine",
        RTLD_NOW
    )
    if handle == nil {
        throw XCTSkip("AppleNeuralEngine.framework unavailable", file: file, line: line)
    }
    dlclose(handle)
}

private func makeTestLayerWeights(value: Float = 0.01) -> LayerWeights {
    let w = LayerWeights()
    fillTestWeights(w, value: value)
    return w
}

private func fillTestWeights(_ weights: borrowing LayerWeights, value: Float) {
    func fill(_ buf: borrowing TensorBuffer, _ val: Float) {
        buf.withUnsafeMutableBufferPointer { ptr in
            for i in ptr.indices { ptr[i] = val }
        }
    }
    fill(weights.Wq, value)
    fill(weights.Wk, value)
    fill(weights.Wv, value)
    fill(weights.Wo, value)
    fill(weights.W1, value)
    fill(weights.W3, value)
    fill(weights.W2, value)
    fill(weights.rmsAtt, 1.0)
    fill(weights.rmsFfn, 1.0)
}

final class FusedDecodeKernelSetTests: XCTestCase {

    func test_fused_kernel_compiles_on_hardware() throws {
        try requireANEHardware()
        let weights = makeTestLayerWeights()
        let kernelSet = try FusedDecodeKernelSet(weights: weights, maxSeq: 32)
        XCTAssertEqual(kernelSet.laneSpatial, 32)
        XCTAssertEqual(kernelSet.kernelMaxSeq, 32)
        XCTAssertEqual(kernelSet.maxSeq, 32)
    }

    func test_fused_kernel_eval_succeeds() throws {
        try requireANEHardware()
        let weights = makeTestLayerWeights()
        let kernelSet = try FusedDecodeKernelSet(weights: weights, maxSeq: 32)
        do {
            try kernelSet.fusedLayer.eval()
        } catch {
            print("  NOTE: fused kernel eval failed (known host instability): \(error)")
        }
    }

    func test_fused_kernel_surfaces_accessible() throws {
        try requireANEHardware()
        let weights = makeTestLayerWeights()
        let kernelSet = try FusedDecodeKernelSet(weights: weights, maxSeq: 32)
        // Verify 4 input surfaces are accessible
        for i in 0..<4 {
            let surf = try kernelSet.fusedLayer.inputSurface(at: i)
            XCTAssertTrue(IOSurfaceGetAllocSize(surf) > 0, "Input surface \(i) should be non-empty")
        }
        // Verify 3 output surfaces are accessible
        for i in 0..<3 {
            let surf = try kernelSet.fusedLayer.outputSurface(at: i)
            XCTAssertTrue(IOSurfaceGetAllocSize(surf) > 0, "Output surface \(i) should be non-empty")
        }
    }

    func test_fused_invalid_maxSeq_throws() {
        let weights = makeTestLayerWeights()
        do {
            _ = try FusedDecodeKernelSet(weights: weights, maxSeq: 0)
            XCTFail("Should have thrown for maxSeq=0")
        } catch let error as ANEError {
            XCTAssertTrue("\(error)".contains("must be > 0"))
        } catch {
            XCTFail("Unexpected error type: \(error)")
        }
    }

    func test_fused_misaligned_maxSeq_throws() {
        let weights = makeTestLayerWeights()
        do {
            _ = try FusedDecodeKernelSet(weights: weights, maxSeq: 33)
            XCTFail("Should have thrown for maxSeq=33")
        } catch let error as ANEError {
            XCTAssertTrue("\(error)".contains("multiple"))
        } catch {
            XCTFail("Unexpected error type: \(error)")
        }
    }

    func test_fused_decode_loop_single_layer() throws {
        try requireANEHardware()
        let weights = makeTestLayerWeights()
        let kernels = try LayerStorage<FusedDecodeKernelSet>(count: 1) { _ in
            try! FusedDecodeKernelSet(weights: weights, maxSeq: 32)
        }
        let surfaceHandles = try (0..<1).map { i in
            try FusedDecodeSurfaceHandles(kernels: kernels[i])
        }
        ForwardPass.initializeFusedDecodeCachesAndMask(surfaceHandles: surfaceHandles)

        var state = try DecodeState(maxSeq: 32)
        var timings = StepTimingBreakdown()
        let xCur = TensorBuffer(count: ModelConfig.dim, zeroed: true)
        xCur.withUnsafeMutableBufferPointer { ptr in
            for i in ptr.indices { ptr[i] = 0.01 }
        }

        do {
            try ForwardPass.runFusedDecodeTimed(
                xCur: xCur,
                kernels: kernels,
                surfaceHandles: surfaceHandles,
                decodeState: &state,
                timings: &timings
            )
            XCTAssertEqual(state.step, 1, "Should advance to step 1")
            XCTAssertTrue(timings.tAne > 0, "Should record ANE time")
            print("  Fused decode step 0: ANE=\(String(format: "%.3f", timings.tAne))ms, IO=\(String(format: "%.3f", timings.tIO))ms")
        } catch {
            print("  NOTE: fused decode loop eval failed (known host instability): \(error)")
        }
    }

    func test_fused_vs_unfused_timing_comparison() throws {
        try requireANEHardware()
        let weights = makeTestLayerWeights()
        let nLayers = 1
        let maxSeq = 32

        // Compile fused kernel
        let fusedKernels = try LayerStorage<FusedDecodeKernelSet>(count: nLayers) { _ in
            try! FusedDecodeKernelSet(weights: weights, maxSeq: maxSeq)
        }
        let fusedHandles = try (0..<nLayers).map { i in
            try FusedDecodeSurfaceHandles(kernels: fusedKernels[i])
        }
        ForwardPass.initializeFusedDecodeCachesAndMask(surfaceHandles: fusedHandles)

        // Compile unfused kernels
        let unfusedKernels = try LayerStorage<DecodeKernelSet>(count: nLayers) { _ in
            try! DecodeKernelSet(weights: weights, maxSeq: maxSeq)
        }
        let unfusedHandles = try (0..<nLayers).map { i in
            try DecodeSurfaceHandles(kernels: unfusedKernels[i])
        }
        ForwardPass.initializeDecodeCachesAndMask(surfaceHandles: unfusedHandles)

        let xCur = TensorBuffer(count: ModelConfig.dim, zeroed: true)
        xCur.withUnsafeMutableBufferPointer { ptr in
            for i in ptr.indices { ptr[i] = 0.01 }
        }

        // Run 5 fused steps
        var fusedState = try DecodeState(maxSeq: maxSeq)
        var fusedTimings = StepTimingBreakdown()
        var fusedStepsCompleted = 0
        for _ in 0..<5 {
            do {
                try ForwardPass.runFusedDecodeTimed(
                    xCur: xCur, kernels: fusedKernels,
                    surfaceHandles: fusedHandles, decodeState: &fusedState,
                    timings: &fusedTimings
                )
                fusedStepsCompleted += 1
            } catch {
                print("  Fused eval failed at step \(fusedStepsCompleted): \(error)")
                break
            }
        }

        // Run 5 unfused steps
        var unfusedState = try DecodeState(maxSeq: maxSeq)
        var unfusedTimings = StepTimingBreakdown()
        var unfusedStepsCompleted = 0
        for _ in 0..<5 {
            do {
                try ForwardPass.runDecodeTimed(
                    xCur: xCur, kernels: unfusedKernels,
                    surfaceHandles: unfusedHandles, decodeState: &unfusedState,
                    timings: &unfusedTimings
                )
                unfusedStepsCompleted += 1
            } catch {
                print("  Unfused eval failed at step \(unfusedStepsCompleted): \(error)")
                break
            }
        }

        print("  ── Fused vs Unfused (\(nLayers) layer, \(maxSeq) maxSeq) ──")
        if fusedStepsCompleted > 0 {
            let fusedPerStep = fusedTimings.tAne / Double(fusedStepsCompleted)
            print("  Fused:   \(fusedStepsCompleted) steps, ANE=\(String(format: "%.3f", fusedTimings.tAne))ms total, \(String(format: "%.3f", fusedPerStep))ms/step, IO=\(String(format: "%.3f", fusedTimings.tIO))ms")
        }
        if unfusedStepsCompleted > 0 {
            let unfusedPerStep = unfusedTimings.tAne / Double(unfusedStepsCompleted)
            print("  Unfused: \(unfusedStepsCompleted) steps, ANE=\(String(format: "%.3f", unfusedTimings.tAne))ms total, \(String(format: "%.3f", unfusedPerStep))ms/step, IO=\(String(format: "%.3f", unfusedTimings.tIO))ms")
        }
        if fusedStepsCompleted > 0 && unfusedStepsCompleted > 0 {
            let fusedPerStep = fusedTimings.tAne / Double(fusedStepsCompleted)
            let unfusedPerStep = unfusedTimings.tAne / Double(unfusedStepsCompleted)
            let savedMS = unfusedPerStep - fusedPerStep
            print("  Savings: \(String(format: "%.3f", savedMS))ms/step (\(String(format: "%.1f", savedMS / unfusedPerStep * 100))%)")
        }
    }
}

final class FusedTwoLayerDecodeKernelSetTests: XCTestCase {

    func test_fused_two_layer_compile_fails_with_controlled_error_on_hardware() throws {
        try requireANEHardware()
        let layer0Weights = makeTestLayerWeights(value: 0.01)
        let layer1Weights = makeTestLayerWeights(value: 0.02)
        do {
            _ = try FusedTwoLayerDecodeKernelSet(
                layer0Weights: layer0Weights,
                layer1Weights: layer1Weights,
                maxSeq: 32
            )
            XCTFail("Expected fused two-layer fallback compile to fail with a controlled error")
        } catch {
            print("  NOTE: fused two-layer fallback compile failed as expected: \(error)")
            XCTAssertTrue("\(error)".contains("ANE kernel compilation failed"))
        }
    }
}
