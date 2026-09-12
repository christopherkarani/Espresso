import Testing
import ModelSupport
import Espresso
@testable import RealModelInference

@Test func smallVocabSelectsANE() {
    let config = MultiModelConfig(
        name: "tiny-test",
        nLayer: 1,
        nHead: 1,
        nKVHead: 1,
        dModel: 64,
        headDim: 64,
        hiddenDim: 128,
        vocab: 256,
        maxSeq: 32,
        normEps: 1e-5,
        architecture: .llama
    )
    #expect(ClassifierStrategy.select(for: config) == .ane)
}

@Test func largeVocabSelectsMetalGEMV() {
    let config = MultiModelConfig(
        name: "tinyllama-test",
        nLayer: 22,
        nHead: 32,
        nKVHead: 4,
        dModel: 2048,
        headDim: 64,
        hiddenDim: 5632,
        vocab: 32_000,
        maxSeq: 2048,
        normEps: 1e-5,
        architecture: .llama
    )
    // 32000 * 2048 = 65_536_000 elements > 16_000_000
    #expect(ClassifierStrategy.select(for: config) == .metalFP16GEMV)
    #expect(ClassifierStrategy.metalFP16GEMV.runtimeFallback == .cpuFP16Tiled)
}

@Test func stories110mUsesANEAllowlist() {
    #expect(ClassifierStrategy.select(for: ModelRegistry.stories110m) == .ane)
}

@Test func storiesBundleStyleNameUsesANEAllowlist() {
    let config = MultiModelConfig(
        name: "llama2.c-stories110M",
        nLayer: 12,
        nHead: 12,
        nKVHead: 12,
        dModel: 768,
        headDim: 64,
        hiddenDim: 2_048,
        vocab: 32_000,
        maxSeq: 256,
        normEps: 1e-5,
        architecture: .llama
    )
    #expect(ClassifierStrategy.select(for: config) == .ane)
}

@Test func llamaLargeVocabWithExactSidecarSelectsPartitionedCPU() {
    let config = MultiModelConfig(
        name: "tinyllama-test",
        nLayer: 22,
        nHead: 32,
        nKVHead: 4,
        dModel: 2048,
        headDim: 64,
        hiddenDim: 5632,
        vocab: 32_000,
        maxSeq: 2048,
        normEps: 1e-5,
        architecture: .llama
    )
    #expect(ClassifierStrategy.select(for: config, hasExactFloat32LMHead: true) == .cpuPartitionedFP32)
}

@Test func qwen3VocabWithoutSidecarSelectsMetalGEMV() {
    let config = MultiModelConfig(
        name: "qwen3-0.6b-test",
        nLayer: 28,
        nHead: 16,
        nKVHead: 8,
        dModel: 1024,
        headDim: 64,
        hiddenDim: 3072,
        vocab: 151_936,
        maxSeq: 4096,
        normEps: 1e-5,
        architecture: .llama
    )
    // 151936 * 1024 = 155_582_464 elements >> 16_000_000
    #expect(ClassifierStrategy.select(for: config) == .metalFP16GEMV)
}

@Test func qwen25_15bVocabSelectsMetalGEMV() {
    let config = MultiModelConfig(
        name: "Qwen2.5-1.5B-Instruct",
        nLayer: 28,
        nHead: 12,
        nKVHead: 2,
        dModel: 1536,
        headDim: 128,
        hiddenDim: 8960,
        vocab: 151_936,
        maxSeq: 1024,
        normEps: 1e-6,
        architecture: .llama,
        preferredDecodePath: .hybrid
    )
    // 151936 * 1536 = 233_373_696 fp16 elements (~467 MB) vs ~16M SRAM (~32 MB).
    // The head cannot live in ANE SRAM; it streams once per token through the Metal GPU
    // GEMV (≈2 ms on M3 Max vs ≈23 ms on the CPU tiled path). cpu_fp16_tiled stays as
    // the forced/runtime-fallback backend.
    #expect(config.vocab * config.dModel > 16_000_000)
    #expect(ClassifierStrategy.select(for: config) == .metalFP16GEMV)
    #expect(ClassifierStrategy.select(for: config).exactHeadBackendLabel == "metal_fp16_gemv")
    #expect(ClassifierStrategy.select(for: config).usesCPUExactClassifier)
}

@Test func gpt2LargeVocabKeepsPartitionedCPUPath() {
    let config = MultiModelConfig(
        name: "gpt2-large-vocab",
        nLayer: 12,
        nHead: 12,
        nKVHead: 12,
        dModel: 768,
        headDim: 64,
        hiddenDim: 3072,
        vocab: 50_257,
        maxSeq: 1024,
        normEps: 1e-5,
        architecture: .gpt2
    )
    #expect(ClassifierStrategy.select(for: config) == .cpuPartitionedFP32)
}

@Test func exactThresholdSelectsANE() {
    // 250_000 * 64 == 16_000_000 == limit → should be .ane
    let config = MultiModelConfig(
        name: "boundary-test",
        nLayer: 1,
        nHead: 1,
        nKVHead: 1,
        dModel: 64,
        headDim: 64,
        hiddenDim: 128,
        vocab: 250_000,
        maxSeq: 32,
        normEps: 1e-5,
        architecture: .llama
    )
    #expect(ClassifierStrategy.select(for: config) == .ane)
}

@Test func oneOverThresholdSelectsCPU() {
    // 250_001 * 64 == 16_000_064 > 16_000_000 → leaves the ANE; dModel 64 is below the
    // Metal GEMV lane width (256), so the CPU tiled head is the exact backend.
    let config = MultiModelConfig(
        name: "boundary-plus-one",
        nLayer: 1,
        nHead: 1,
        nKVHead: 1,
        dModel: 64,
        headDim: 64,
        hiddenDim: 128,
        vocab: 250_001,
        maxSeq: 32,
        normEps: 1e-5,
        architecture: .llama
    )
    #expect(ClassifierStrategy.select(for: config) == .cpuFP16Tiled)
}

@Test func cpuTiledArgmaxCorrectness() {
    let vocabSize = 16
    let dim = 8
    let winnerRow = 7

    // Build row-major FP16 weight matrix [vocabSize x dim]
    // All rows have small values except row 7, which has 5.0 in every column.
    var fp16Weights = [UInt16](repeating: Float16(0.1).bitPattern, count: vocabSize * dim)
    for col in 0..<dim {
        fp16Weights[winnerRow * dim + col] = Float16(5.0).bitPattern
    }

    // Input: all ones (FP32)
    let input = [Float](repeating: 1.0, count: dim)

    let result = fp16Weights.withUnsafeBufferPointer { wBuf in
        input.withUnsafeBufferPointer { iBuf in
            FP16TiledClassifier.tiledMatvecArgmax(
                weights: wBuf.baseAddress!,
                input: iBuf.baseAddress!,
                vocabSize: vocabSize,
                dim: dim,
                tileRows: 4
            )
        }
    }

    #expect(result == winnerRow)
}

@Test func cpuTiledLargeVocabIntegration() {
    let vocabSize = 50_000
    let dim = 64
    let winnerToken = 42_000

    // Build row-major FP16 weight matrix — all zeros except winner row
    var fp16Weights = [UInt16](repeating: Float16(0.0).bitPattern, count: vocabSize * dim)
    for col in 0..<dim {
        fp16Weights[winnerToken * dim + col] = Float16(1.0).bitPattern
    }

    // Input: all ones (FP32)
    let input = [Float](repeating: 1.0, count: dim)

    let result = fp16Weights.withUnsafeBufferPointer { wBuf in
        input.withUnsafeBufferPointer { iBuf in
            FP16TiledClassifier.tiledMatvecArgmax(
                weights: wBuf.baseAddress!,
                input: iBuf.baseAddress!,
                vocabSize: vocabSize,
                dim: dim
            )
        }
    }

    #expect(result == winnerToken)
}
