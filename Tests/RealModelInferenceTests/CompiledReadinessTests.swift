import Testing
import IOSurface
import ANETypes
@testable import Espresso
@testable import RealModelInference

private func makeTestSurface() -> IOSurfaceRef {
    let surface = IOSurfaceCreate([
        kIOSurfaceWidth: 1,
        kIOSurfaceHeight: 1,
        kIOSurfaceBytesPerElement: 4,
    ] as CFDictionary)!
    return surface
}

@Suite struct CompiledReadinessTests {
    // MARK: CompiledReadiness transitions

    @Test func readinessTransitionsBetweenNotCompiledAndCompiled() {
        var readiness = CompiledReadiness<BaselineCompiledRuntime>.notCompiled
        #expect(readiness.runtime == nil)

        readiness = .compiled(BaselineCompiledRuntime(bucket: 256, inputSurface: makeTestSurface()))
        switch readiness {
        case .compiled(let runtime):
            #expect(runtime.bucket == 256)
        case .notCompiled:
            Issue.record("readiness should be compiled after the transition")
        }
    }

    @Test func runtimeAccessorExposesOnlyTheResidentRuntime() {
        let notCompiled = CompiledReadiness<FusedHybridCompiledRuntime>.notCompiled
        #expect(notCompiled.runtime == nil)

        guard let runtime = FusedHybridCompiledRuntime(
            bucket: 512,
            layerCount: 4,
            surfaceHandleCount: 4,
            expectedLayerCount: 4
        ) else {
            Issue.record("a complete fused program set must validate")
            return
        }
        let resident = CompiledReadiness<FusedHybridCompiledRuntime>.compiled(runtime)
        #expect(resident.runtime?.bucket == 512)
    }

    // MARK: SplitHybridCompiledRuntime validation

    @Test func splitHybridRuntimeRejectsIncompleteProgramSets() {
        // Missing layers
        #expect(SplitHybridCompiledRuntime(
            bucket: 256, layerCount: 3, surfaceHandleCount: 4,
            expectedLayerCount: 4, headCount: 1, headSpatial: 32
        ) == nil)
        // Missing surfaces
        #expect(SplitHybridCompiledRuntime(
            bucket: 256, layerCount: 4, surfaceHandleCount: 3,
            expectedLayerCount: 4, headCount: 1, headSpatial: 32
        ) == nil)
        // Missing head
        #expect(SplitHybridCompiledRuntime(
            bucket: 256, layerCount: 4, surfaceHandleCount: 4,
            expectedLayerCount: 4, headCount: 0, headSpatial: 32
        ) == nil)
        // Degenerate head spatial
        #expect(SplitHybridCompiledRuntime(
            bucket: 256, layerCount: 4, surfaceHandleCount: 4,
            expectedLayerCount: 4, headCount: 1, headSpatial: 0
        ) == nil)
    }

    @Test func splitHybridRuntimeValidatesQKNormWeightsOnlyWhenRequested() {
        // GPT-2 flavor: no QK-norm requirement.
        #expect(SplitHybridCompiledRuntime(
            bucket: 256, layerCount: 4, surfaceHandleCount: 4,
            expectedLayerCount: 4, headCount: 1, qKNormCount: nil, headSpatial: 32
        ) != nil)
        // Llama flavor: one entry per layer is required.
        #expect(SplitHybridCompiledRuntime(
            bucket: 256, layerCount: 4, surfaceHandleCount: 4,
            expectedLayerCount: 4, headCount: 1, qKNormCount: 4, headSpatial: 32
        ) != nil)
        #expect(SplitHybridCompiledRuntime(
            bucket: 256, layerCount: 4, surfaceHandleCount: 4,
            expectedLayerCount: 4, headCount: 1, qKNormCount: 3, headSpatial: 32
        ) == nil)
    }

    // MARK: FusedHybridCompiledRuntime validation

    @Test func fusedHybridRuntimeRequiresEveryLayerAndSurface() {
        #expect(FusedHybridCompiledRuntime(
            bucket: 128, layerCount: 4, surfaceHandleCount: 4, expectedLayerCount: 4
        ) != nil)
        #expect(FusedHybridCompiledRuntime(
            bucket: 128, layerCount: 2, surfaceHandleCount: 4, expectedLayerCount: 4
        ) == nil)
        #expect(FusedHybridCompiledRuntime(
            bucket: 128, layerCount: 4, surfaceHandleCount: 0, expectedLayerCount: 4
        ) == nil)
    }

    // MARK: GenerationOutputHeadSelection backend derivation

    @Test func cpuSgemmSelectionCarriesItsExactBackend() {
        for backend in [GenerationOutputHeadBackend.cpu, .cpuThenANE, .cpuPartitionedArgmax, .cpuFP16Tiled] {
            let selection = GenerationOutputHeadSelection.cpuSgemm(backend)
            #expect(selection.backend == backend)
        }
    }

    @Test func stagedCPUSelectionIsPresenceTypedAgainstTheStagedHead() throws {
        let classifierWeights = TensorBuffer(count: ModelConfig.dim * 8, zeroed: true)
        let stagedHead = try CPUStagedExactGenerationOutputHead(
            classifierWeights: classifierWeights,
            vocabSize: 8,
            layoutStrategy: .contiguous(shardSize: 4)
        )
        let selection = GenerationOutputHeadSelection.stagedCPU(stagedHead)
        switch selection {
        case .stagedCPU:
            // Presence-typed: no guard-throw needed to reach the head.
            #expect(true)
        case .cpuSgemm, .aneClassifier, .aneRMSNormClassifier:
            Issue.record("staged selection must stay in the stagedCPU case")
        }
    }
}
