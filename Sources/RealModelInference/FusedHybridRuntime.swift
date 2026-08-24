import ANERuntime
import ANETypes
import Espresso
import IOSurface
import Darwin
import Foundation
import ModelSupport

// Fused-hybrid trunk runtime (extracted from RealModelInferenceEngine).
//
// One ANE program per transformer layer, attention included. The session owns
// the trunk's resident programs and surface handles; readiness follows
// ``CompiledReadiness`` and only the ensure function writes it.

extension RealModelInferenceEngine {
    mutating func ensureFusedHybridCompiled(bucket: Int) throws -> Bool {
        switch fusedHybridReadiness {
        case .compiled(let runtime) where runtime.bucket >= bucket:
            return false
        case .compiled, .notCompiled:
            break
        }

        let newLayers: LayerStorage<FusedHybridDecodeLayerKernelSet>
        do {
            newLayers = try Self.compileFusedHybridLayers(
                config: config,
                weightDirURL: weightDirURL,
                maxSeq: bucket,
                environment: policies.environment
            )
        } catch let error as RealModelInferenceError {
            if case .hybridFallbackDisabled = error { throw error }
            throw Self.fusedHybridFallbackError(reason: error.errorDescription ?? "\(error)")
        } catch {
            throw Self.fusedHybridFallbackError(reason: "\(error)")
        }

        var newSurfaceHandles: [FusedHybridDecodeSurfaceHandles] = []
        newSurfaceHandles.reserveCapacity(newLayers.count)
        for layerIndex in 0..<newLayers.count {
            do {
                newSurfaceHandles.append(
                    try FusedHybridDecodeSurfaceHandles(kernels: newLayers[layerIndex])
                )
            } catch {
                throw Self.fusedHybridFallbackError(
                    reason: "fused N=1 surfaces unavailable for layer \(layerIndex): \(error)"
                )
            }
        }

        compiledFusedHybridLayers = newLayers
        compiledFusedHybridSurfaceHandles = newSurfaceHandles
        if let runtime = FusedHybridCompiledRuntime(
            bucket: bucket,
            layerCount: compiledFusedHybridLayers.count,
            surfaceHandleCount: compiledFusedHybridSurfaceHandles.count,
            expectedLayerCount: config.nLayer
        ) {
            fusedHybridReadiness = .compiled(runtime)
        } else {
            fusedHybridReadiness = .notCompiled
        }
        return true
    }

    static func compileFusedHybridLayers(
        config: MultiModelConfig,
        weightDirURL: URL,
        maxSeq: Int,
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) throws -> LayerStorage<FusedHybridDecodeLayerKernelSet> {
        guard config.nHead > 0, config.nKVHead > 0, config.headDim > 0,
              config.nHead % config.nKVHead == 0 else {
            throw fusedHybridFallbackError(
                reason: "invalid fused N=1 head geometry nHead=\(config.nHead) nKVHead=\(config.nKVHead) headDim=\(config.headDim)"
            )
        }
        var donor: FusedHybridDecodeLayerKernelSet.DonorHexIDs?
        return try LayerStorage(count: config.nLayer, throwingInitializer: { layerIndex in
            fputs(
                "[FusedHybridDecode] compiling layer \(layerIndex)/\(config.nLayer) maxSeq=\(maxSeq) n=1\n",
                stderr
            )
            let paths = LayerWeightPaths.forLayer(
                layerIndex,
                config: config,
                blobDir: weightDirURL.path
            )
            let weights = try loadHybridLayerWeightsLlama(config: config, paths: paths)
            let compiled = try FusedHybridDecodeLayerKernelSet(
                weights: weights,
                maxSeq: maxSeq,
                nHeads: config.nHead,
                nKVHeads: config.nKVHead,
                headDim: config.headDim,
                donorHexIDs: donor,
                options: HybridDecodeKernelOptions.resolve(environment: environment)
            )
            donor = compiled.donorHexIDs
            return compiled
        })
    }

}
