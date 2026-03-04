import Accelerate
import CPUOps
import Darwin
import Foundation
import ANERuntime
import ANETypes
import Espresso

@main
enum EspressoTrainMain {
    private static let defaultCheckpointPath = "ane_stories110M_ckpt.bin"
    private static let defaultModelPath = "../../assets/models/stories110M.bin"
    private static let defaultDataPath = "tinystories_data00.bin"

    private struct Args {
        var resume: Bool = false
        var totalSteps: Int = 10_000
        var lr: Float = 3e-4

        var checkpointPath: String = defaultCheckpointPath
        var modelPath: String = defaultModelPath
        var dataPath: String = defaultDataPath
    }

    private enum TrainExit {
        case finished
        case execRestart(step: Int, compileCount: Int, loss: Float)
    }

    static func main() {
        do {
            let exitReason = try train(args: parseArgs(CommandLine.arguments))
            switch exitReason {
            case .finished:
                return
            case let .execRestart(step, compileCount, loss):
                ExecRestart.restart(step: step, compileCount: compileCount, loss: loss)
            }
        } catch {
            fputs("espresso-train error: \(error)\n", stderr)
            exit(1)
        }
    }

    private static func parseArgs(_ argv: [String]) -> Args {
        var a = Args()
        var i = 1
        while i < argv.count {
            let arg = argv[i]
            switch arg {
            case "--resume":
                a.resume = true
            case "--steps":
                if i + 1 < argv.count { a.totalSteps = Int(argv[i + 1]) ?? a.totalSteps; i += 1 }
            case "--lr":
                if i + 1 < argv.count { a.lr = Float(argv[i + 1]) ?? a.lr; i += 1 }
            case "--ckpt":
                if i + 1 < argv.count { a.checkpointPath = argv[i + 1]; i += 1 }
            case "--model":
                if i + 1 < argv.count { a.modelPath = argv[i + 1]; i += 1 }
            case "--data":
                if i + 1 < argv.count { a.dataPath = argv[i + 1]; i += 1 }
            default:
                break
            }
            i += 1
        }
        return a
    }

    // MARK: - Timing

    private enum MachTime {
        private static let tb: mach_timebase_info_data_t = {
            var tb = mach_timebase_info_data_t()
            mach_timebase_info(&tb)
            return tb
        }()

        @inline(__always)
        static func now() -> UInt64 { mach_absolute_time() }

        @inline(__always)
        static func ms(_ delta: UInt64) -> Double {
            let nanos = (Double(delta) * Double(tb.numer)) / Double(tb.denom)
            return nanos / 1_000_000.0
        }
    }

    // MARK: - Training

    private static func train(args: Args) throws -> TrainExit {
        setbuf(stdout, nil)
        CheckpointHeader.validateLayout()

        let dim = ModelConfig.dim
        let hidden = ModelConfig.hidden
        let heads = ModelConfig.heads
        let seqLen = ModelConfig.seqLen
        let nLayers = ModelConfig.nLayers
        let vocab = ModelConfig.vocab

        let beta1: Float = 0.9
        let beta2: Float = 0.999
        let eps: Float = 1e-8
        let posix = Locale(identifier: "en_US_POSIX")

        // Allocate per-layer state.
        let layers = LayerStorage<LayerWeights>(count: nLayers) { _ in LayerWeights() }
        let layerAdam = LayerStorage<LayerAdam>(count: nLayers) { _ in LayerAdam() }
        let acts = LayerStorage<LayerActivations>(count: nLayers) { _ in LayerActivations() }
        let grads = LayerStorage<LayerGradients>(count: nLayers) { _ in LayerGradients() }

        // Globals: final RMSNorm + embedding.
        let rmsFinal = TensorBuffer(count: dim, zeroed: false)
        let embed = TensorBuffer(count: vocab * dim, zeroed: false) // [vocab, dim] row-major
        let grmsFinal = TensorBuffer(count: dim, zeroed: true)
        let gembed = TensorBuffer(count: vocab * dim, zeroed: true)
        let adamRmsFinal = AdamState(count: dim)
        let adamEmbed = AdamState(count: vocab * dim)

        var totalSteps = args.totalSteps
        var lr = args.lr

        var cumCompile: Double = 0
        var cumTrain: Double = 0
        var cumWall: Double = 0
        var cumSteps: Int = 0
        var cumBatches: Int = 0
        var adamT: Int = 0
        var startStep: Int = 0
        var lastLoss: Float = 999.0
        var resuming: Bool = false

        if args.resume {
            do {
                let meta = try Checkpoint.load(
                    path: args.checkpointPath,
                    intoLayers: layers,
                    intoLayerAdam: layerAdam,
                    intoRmsFinal: rmsFinal,
                    intoAdamRmsFinal: adamRmsFinal,
                    intoEmbed: embed,
                    intoAdamEmbed: adamEmbed
                )
                startStep = meta.step
                totalSteps = meta.totalSteps
                lr = meta.lr
                lastLoss = meta.loss
                cumCompile = meta.cumCompile
                cumTrain = meta.cumTrain
                cumWall = meta.cumWall
                cumSteps = meta.cumSteps
                cumBatches = meta.cumBatches
                adamT = meta.adamT
                resuming = true
                print(String(format: "[RESUMED step %d, loss=%.4f]", startStep, lastLoss))
            } catch {
                print("[resume failed: \(error)]")
            }
        }

        if !resuming {
            // Load pretrained weights. The checkpoint format assumes a shared embed/classifier.
            do {
                let pretrained = try ModelWeightLoader.load(from: args.modelPath)
                precondition(pretrained.sharedClassifier, "Checkpoint format assumes shared embed/classifier weights")

                for L in 0..<nLayers {
                    layers[L].Wq.withUnsafeMutablePointer { dst in
                        pretrained.layers[L].Wq.withUnsafePointer { src in
                            dst.update(from: src, count: ModelConfig.wqSize)
                        }
                    }
                    layers[L].Wk.withUnsafeMutablePointer { dst in
                        pretrained.layers[L].Wk.withUnsafePointer { src in
                            dst.update(from: src, count: ModelConfig.wqSize)
                        }
                    }
                    layers[L].Wv.withUnsafeMutablePointer { dst in
                        pretrained.layers[L].Wv.withUnsafePointer { src in
                            dst.update(from: src, count: ModelConfig.wqSize)
                        }
                    }
                    layers[L].Wo.withUnsafeMutablePointer { dst in
                        pretrained.layers[L].Wo.withUnsafePointer { src in
                            dst.update(from: src, count: ModelConfig.woSize)
                        }
                    }
                    layers[L].W1.withUnsafeMutablePointer { dst in
                        pretrained.layers[L].W1.withUnsafePointer { src in
                            dst.update(from: src, count: ModelConfig.w1Size)
                        }
                    }
                    layers[L].W2.withUnsafeMutablePointer { dst in
                        pretrained.layers[L].W2.withUnsafePointer { src in
                            dst.update(from: src, count: ModelConfig.w2Size)
                        }
                    }
                    layers[L].W3.withUnsafeMutablePointer { dst in
                        pretrained.layers[L].W3.withUnsafePointer { src in
                            dst.update(from: src, count: ModelConfig.w3Size)
                        }
                    }
                    layers[L].rmsAtt.withUnsafeMutablePointer { dst in
                        pretrained.layers[L].rmsAtt.withUnsafePointer { src in
                            dst.update(from: src, count: dim)
                        }
                    }
                    layers[L].rmsFfn.withUnsafeMutablePointer { dst in
                        pretrained.layers[L].rmsFfn.withUnsafePointer { src in
                            dst.update(from: src, count: dim)
                        }
                    }
                }

                rmsFinal.withUnsafeMutablePointer { dst in
                    pretrained.rmsFinal.withUnsafePointer { src in
                        dst.update(from: src, count: dim)
                    }
                }
                embed.withUnsafeMutablePointer { dst in
                    pretrained.embed.withUnsafePointer { src in
                        dst.update(from: src, count: vocab * dim)
                    }
                }
            } catch {
                // Random init (deterministic), then reseed for sampling below.
                print("[pretrained load failed, random init: \(error)]")
                srand48(42)

                let scaleD = 1.0 / sqrtf(Float(dim))
                let scaleH = 1.0 / sqrtf(Float(hidden))
                let scaleDD = Double(scaleD)
                let scaleHD = Double(scaleH)

                @inline(__always)
                func randSymmetric(scale: Double) -> Float {
                    // Match ObjC ordering/precision: (float scale) promoted to double, multiplied in double,
                    // then cast once on assignment to Float.
                    Float(scale * (2.0 * drand48() - 1.0))
                }

                for L in 0..<nLayers {
                    layers[L].Wq.withUnsafeMutablePointer { wq in
                        layers[L].Wk.withUnsafeMutablePointer { wk in
                            for i in 0..<ModelConfig.wqSize {
                                wq[i] = randSymmetric(scale: scaleDD)
                                wk[i] = randSymmetric(scale: scaleDD)
                            }
                        }
                    }
                    layers[L].Wv.withUnsafeMutablePointer { wv in
                        layers[L].Wo.withUnsafeMutablePointer { wo in
                            for i in 0..<ModelConfig.wqSize {
                                wv[i] = randSymmetric(scale: scaleDD)
                                wo[i] = randSymmetric(scale: scaleDD)
                            }
                        }
                    }
                    layers[L].W1.withUnsafeMutablePointer { ptr in
                        for i in 0..<ModelConfig.w1Size { ptr[i] = randSymmetric(scale: scaleHD) }
                    }
                    layers[L].W2.withUnsafeMutablePointer { ptr in
                        for i in 0..<ModelConfig.w2Size { ptr[i] = randSymmetric(scale: scaleDD) }
                    }
                    layers[L].W3.withUnsafeMutablePointer { ptr in
                        for i in 0..<ModelConfig.w3Size { ptr[i] = randSymmetric(scale: scaleHD) }
                    }
                    layers[L].rmsAtt.withUnsafeMutablePointer { ptr in
                        for i in 0..<dim { ptr[i] = 1.0 }
                    }
                    layers[L].rmsFfn.withUnsafeMutablePointer { ptr in
                        for i in 0..<dim { ptr[i] = 1.0 }
                    }
                }
                rmsFinal.withUnsafeMutablePointer { ptr in
                    for i in 0..<dim { ptr[i] = 1.0 }
                }
                let escale: Float = 0.02
                let escaleD = Double(escale)
                embed.withUnsafeMutablePointer { ptr in
                    for i in 0..<(vocab * dim) { ptr[i] = randSymmetric(scale: escaleD) }
                }
            }
        }

        // mmap token data.
        let dataset = try TokenDataset(path: args.dataPath, seqLen: seqLen)
        let maxPos = dataset.nTokens - seqLen - 1

        // Compile static sdpaBwd2 kernels (no weights).
        let staticKernels = try LayerStorage<StaticKernel>(count: nLayers, throwingInitializer: { _ in
            try StaticKernel()
        })

        let accumulator = GradientAccumulator()
        Sampler.seed(startStep: startStep)

        // Scratch buffers (allocate once, reuse).
        let xCur = TensorBuffer(count: dim * seqLen, zeroed: false)
        let xFinal = TensorBuffer(count: dim * seqLen, zeroed: false)
        let logits = TensorBuffer(count: vocab * seqLen, zeroed: false)
        let dlogits = TensorBuffer(count: vocab * seqLen, zeroed: false)
        let dy = TensorBuffer(count: dim * seqLen, zeroed: false)
        let bwdScratch = BackwardScratch(dim: dim, hidden: hidden, seqLen: seqLen)

        var totalCompileMs: Double = 0
        var totalTrainMs: Double = 0
        var stepsDoneThisRun: Int = 0
        var batchesThisRun: Int = 0

        let wallStart = MachTime.now()

        @inline(__always)
        func stderrLine(_ line: String) {
            line.withCString { cstr in
                _ = fputs(cstr, stderr)
                _ = fputc(0x0A, stderr)
            }
        }

        @inline(__always)
        func adamUpdate(
            weights w: borrowing TensorBuffer,
            grads g: borrowing TensorBuffer,
            state s: borrowing AdamState,
            timestep: Int
        ) {
            w.withUnsafeMutablePointer { wPtr in
                g.withUnsafePointer { gPtr in
                    s.m.withUnsafeMutablePointer { mPtr in
                        s.v.withUnsafeMutablePointer { vPtr in
                            AdamOptimizer.update(
                                weights: wPtr,
                                gradients: gPtr,
                                m: mPtr,
                                v: vPtr,
                                count: s.count,
                                timestep: timestep,
                                lr: lr,
                                beta1: beta1,
                                beta2: beta2,
                                eps: eps
                            )
                        }
                    }
                }
            }
        }

        var step = startStep
        while step < totalSteps {
            // Compile budget: exec restart if we can't compile another full weight-bearing batch.
            if CompileBudget.currentCount + ModelConfig.totalWeightKernels > ModelConfig.maxCompiles {
                let wallMs = MachTime.ms(MachTime.now() - wallStart)
                var meta = CheckpointMeta()
                meta.step = step
                meta.totalSteps = totalSteps
                meta.lr = lr
                meta.loss = lastLoss
                meta.cumCompile = cumCompile + totalCompileMs
                meta.cumTrain = cumTrain + totalTrainMs
                meta.cumWall = cumWall + wallMs
                meta.cumSteps = cumSteps + stepsDoneThisRun
                meta.cumBatches = cumBatches + batchesThisRun
                meta.adamT = adamT

                try Checkpoint.save(
                    path: args.checkpointPath,
                    meta: meta,
                    layers: layers,
                    layerAdam: layerAdam,
                    rmsFinal: rmsFinal,
                    adamRmsFinal: adamRmsFinal,
                    embed: embed,
                    adamEmbed: adamEmbed
                )

                return .execRestart(step: step, compileCount: CompileBudget.currentCount, loss: lastLoss)
            }

            // Compile all layers' weight-bearing kernels.
            let tc0 = MachTime.now()
            let kernelStorage: LayerStorage<LayerKernelSet>
            do {
                kernelStorage = try LayerStorage<LayerKernelSet>(count: nLayers, throwingInitializer: { i in
                    try LayerKernelSet(weights: layers[i])
                })
            } catch {
                // Compile failed: force budget exhaustion and restart on next iteration.
                try? CompileBudget.setCount(ModelConfig.maxCompiles)
                continue
            }
            let cms = MachTime.ms(MachTime.now() - tc0)
            totalCompileMs += cms

            // Zero gradient accumulators (accumulate across micro-steps).
            for L in 0..<nLayers { grads[L].zero() }
            grmsFinal.zero()
            gembed.zero()

            var stepsBatch = 0
            let tt0 = MachTime.now()

            for _ in 0..<ModelConfig.accumSteps where step < totalSteps {
                let stepT0 = MachTime.now()
                let currentStep = step

                let pos = Sampler.samplePosition(maxPos: maxPos)
                let inputTokens = dataset[pos]
                let targetTokens = dataset[pos + 1]

                // Embedding lookup -> xCur.
                xCur.withUnsafeMutablePointer { xPtr in
                    embed.withUnsafePointer { ePtr in
                        Embedding.lookup(
                            output: xPtr,
                            embedding: ePtr,
                            tokens: inputTokens,
                            vocabSize: vocab,
                            dim: dim,
                            seqLen: seqLen
                        )
                    }
                }

                // Forward pass (12 layers).
                try ForwardPass.run(xCur: xCur, acts: acts, kernels: kernelStorage, accumulator: accumulator)

                // Final RMSNorm.
                xFinal.withUnsafeMutablePointer { outPtr in
                    xCur.withUnsafePointer { inPtr in
                        rmsFinal.withUnsafePointer { wPtr in
                            RMSNorm.forward(output: outPtr, input: inPtr, weights: wPtr, dim: dim, seqLen: seqLen)
                        }
                    }
                }

                // Classifier: logits = embed @ xFinal.
                logits.withUnsafeMutablePointer { logitsPtr in
                    embed.withUnsafePointer { ePtr in
                        xFinal.withUnsafePointer { xPtr in
                            BLAS.sgemm(
                                CblasRowMajor, CblasNoTrans, CblasNoTrans,
                                m: Int32(vocab), n: Int32(seqLen), k: Int32(dim),
                                alpha: 1.0,
                                a: ePtr, lda: Int32(dim),
                                b: xPtr, ldb: Int32(seqLen),
                                beta: 0.0,
                                c: logitsPtr, ldc: Int32(seqLen)
                            )
                        }
                    }
                }

                // Cross-entropy loss + dlogits.
                let loss = dlogits.withUnsafeMutablePointer { dlogitsPtr in
                    logits.withUnsafePointer { logitsPtr in
                        CrossEntropy.lossAndGradient(
                            dlogits: dlogitsPtr,
                            logits: logitsPtr,
                            targets: targetTokens,
                            vocabSize: vocab,
                            seqLen: seqLen
                        )
                    }
                }
                lastLoss = loss

                // Classifier backward: dy = embed^T @ dlogits.
                dy.withUnsafeMutablePointer { dyPtr in
                    embed.withUnsafePointer { ePtr in
                        dlogits.withUnsafePointer { dlogitsPtr in
                            BLAS.sgemm(
                                CblasRowMajor, CblasTrans, CblasNoTrans,
                                m: Int32(dim), n: Int32(seqLen), k: Int32(vocab),
                                alpha: 1.0,
                                a: ePtr, lda: Int32(dim),
                                b: dlogitsPtr, ldb: Int32(seqLen),
                                beta: 0.0,
                                c: dyPtr, ldc: Int32(seqLen)
                            )
                        }
                    }
                }

                // dembed += dlogits @ xFinal^T (async, accumulate).
                let gembedPtr = gembed.withUnsafeMutablePointer { SendablePointer($0) }
                let captDlogits = dlogits.withUnsafePointer { SendableConstPointer($0) }
                let captXFinal = xFinal.withUnsafePointer { SendableConstPointer($0) }
                accumulator.enqueue { [captDlogits, captXFinal] in
                    BLAS.sgemm(
                        CblasRowMajor, CblasNoTrans, CblasTrans,
                        m: Int32(vocab), n: Int32(dim), k: Int32(seqLen),
                        alpha: 1.0,
                        a: captDlogits.pointer, lda: Int32(seqLen),
                        b: captXFinal.pointer, ldb: Int32(seqLen),
                        beta: 1.0,
                        c: gembedPtr.pointer, ldc: Int32(dim)
                    )
                }

                // Final RMSNorm backward: dx_rms_final -> dy (in-place overwrite).
                bwdScratch.dxRms1.withUnsafeMutablePointer { dxPtr in
                    grmsFinal.withUnsafeMutablePointer { dwPtr in
                        dy.withUnsafePointer { dyPtr in
                            xCur.withUnsafePointer { xPtr in
                                rmsFinal.withUnsafePointer { wPtr in
                                    RMSNorm.backward(
                                        dx: dxPtr,
                                        dw: dwPtr,
                                        dy: dyPtr,
                                        x: xPtr,
                                        weights: wPtr,
                                        dim: dim,
                                        seqLen: seqLen
                                    )
                                }
                            }
                        }
                    }
                }
                dy.withUnsafeMutablePointer { dyPtr in
                    bwdScratch.dxRms1.withUnsafePointer { dxPtr in
                        dyPtr.update(from: dxPtr, count: dim * seqLen)
                    }
                }

                // Backward pass (12 layers, reverse).
                try BackwardPass.run(
                    dy: dy,
                    acts: acts,
                    kernels: kernelStorage,
                    staticKernels: staticKernels,
                    grads: grads,
                    weights: layers,
                    scratch: bwdScratch,
                    accumulator: accumulator,
                    dim: dim,
                    hidden: hidden,
                    seqLen: seqLen,
                    heads: heads
                )

                // Embedding backward (accumulates into gembed).
                accumulator.barrier()
                gembed.withUnsafeMutablePointer { gPtr in
                    dy.withUnsafePointer { dyPtr in
                        Embedding.backward(
                            dEmbedding: gPtr,
                            dx: dyPtr,
                            tokens: inputTokens,
                            vocabSize: vocab,
                            dim: dim,
                            seqLen: seqLen
                        )
                    }
                }

                if currentStep % 10 == 0 || currentStep == startStep {
                    print(String(format: "step %-4d loss=%.4f", locale: posix, currentStep, lastLoss))
                }

                let stepMs = MachTime.ms(MachTime.now() - stepT0)
                stderrLine(
                    String(
                        format: "{\"type\":\"step\",\"step\":%d,\"loss\":%.6f,\"ms\":%.3f,\"compiles\":%d}",
                        locale: posix,
                        currentStep,
                        lastLoss,
                        stepMs,
                        CompileBudget.currentCount
                    )
                )

                stepsBatch += 1
                stepsDoneThisRun += 1
                step += 1
            }

            let tms = MachTime.ms(MachTime.now() - tt0)
            totalTrainMs += tms
            batchesThisRun += 1

            // Wait all async dW.
            accumulator.waitAll()

            // Scale gradients by mean over actual accumulated steps.
            let gsc = 1.0 / Float(stepsBatch)
            for L in 0..<nLayers {
                GradientScaling.scaleLayer(
                    Wq: grads[L].Wq, Wk: grads[L].Wk, Wv: grads[L].Wv, Wo: grads[L].Wo,
                    W1: grads[L].W1, W2: grads[L].W2, W3: grads[L].W3,
                    rmsAtt: grads[L].rmsAtt, rmsFfn: grads[L].rmsFfn,
                    by: gsc
                )
            }
            GradientScaling.scale(grmsFinal, by: gsc)
            GradientScaling.scale(gembed, by: gsc)

            // Adam update: increment timestep BEFORE update (t starts at 1).
            adamT += 1
            for L in 0..<nLayers {
                adamUpdate(weights: layers[L].Wq, grads: grads[L].Wq, state: layerAdam[L].Wq, timestep: adamT)
                adamUpdate(weights: layers[L].Wk, grads: grads[L].Wk, state: layerAdam[L].Wk, timestep: adamT)
                adamUpdate(weights: layers[L].Wv, grads: grads[L].Wv, state: layerAdam[L].Wv, timestep: adamT)
                adamUpdate(weights: layers[L].Wo, grads: grads[L].Wo, state: layerAdam[L].Wo, timestep: adamT)
                adamUpdate(weights: layers[L].W1, grads: grads[L].W1, state: layerAdam[L].W1, timestep: adamT)
                adamUpdate(weights: layers[L].W2, grads: grads[L].W2, state: layerAdam[L].W2, timestep: adamT)
                adamUpdate(weights: layers[L].W3, grads: grads[L].W3, state: layerAdam[L].W3, timestep: adamT)
                adamUpdate(weights: layers[L].rmsAtt, grads: grads[L].rmsAtt, state: layerAdam[L].rmsAtt, timestep: adamT)
                adamUpdate(weights: layers[L].rmsFfn, grads: grads[L].rmsFfn, state: layerAdam[L].rmsFfn, timestep: adamT)
            }
            adamUpdate(weights: rmsFinal, grads: grmsFinal, state: adamRmsFinal, timestep: adamT)
            adamUpdate(weights: embed, grads: gembed, state: adamEmbed, timestep: adamT)

            print(
                String(
                    format: "  [batch %d: compile=%.0fms train=%.1fms (%.1fms/step) compiles=%d]",
                    locale: posix,
                    stepsBatch,
                    cms,
                    tms,
                    tms / Double(stepsBatch),
                    CompileBudget.currentCount
                )
            )

            stderrLine(
                String(
                    format: "{\"type\":\"batch\",\"batch\":%d,\"compile_ms\":%.1f,\"train_ms\":%.1f,\"ms_per_step\":%.1f}",
                    locale: posix,
                    stepsBatch,
                    cms,
                    tms,
                    tms / Double(stepsBatch)
                )
            )

            do {
                // Perf summary (matches train_large.m formulas).
                let fwdFlops = Double(nLayers)
                    * (4.0 * 2.0 * Double(dim) * Double(dim) * Double(seqLen)
                        + 2.0 * 2.0 * Double(dim) * Double(hidden) * Double(seqLen)
                        + 2.0 * Double(hidden) * Double(dim) * Double(seqLen))
                let sdpaFlops = Double(nLayers)
                    * 2.0 * Double(heads) * 5.0 * Double(seqLen) * Double(seqLen) * Double(ModelConfig.headDim)
                let aneFBatch = (fwdFlops * 2.0 + sdpaFlops) * Double(stepsBatch)
                let aneTflops = aneFBatch / (tms * 1e9)
                let aneUtilPct = 100.0 * aneTflops / 15.8
                stderrLine(String(format: "{\"type\":\"perf\",\"ane_tflops\":%.3f,\"ane_util_pct\":%.2f}", locale: posix, aneTflops, aneUtilPct))
            }
        }

        // Save final checkpoint.
        let wallMs = MachTime.ms(MachTime.now() - wallStart)
        var meta = CheckpointMeta()
        meta.step = step
        meta.totalSteps = totalSteps
        meta.lr = lr
        meta.loss = lastLoss
        meta.cumCompile = cumCompile + totalCompileMs
        meta.cumTrain = cumTrain + totalTrainMs
        meta.cumWall = cumWall + wallMs
        meta.cumSteps = cumSteps + stepsDoneThisRun
        meta.cumBatches = cumBatches + batchesThisRun
        meta.adamT = adamT

        try Checkpoint.save(
            path: args.checkpointPath,
            meta: meta,
            layers: layers,
            layerAdam: layerAdam,
            rmsFinal: rmsFinal,
            adamRmsFinal: adamRmsFinal,
            embed: embed,
            adamEmbed: adamEmbed
        )

        // Efficiency report (matches train_large.m).
        do {
            let wall = cumWall + wallMs
            let compile = cumCompile + totalCompileMs
            let train = cumTrain + totalTrainMs
            let steps = cumSteps + stepsDoneThisRun

            let fwdFlops = Double(nLayers)
                * (4.0 * 2.0 * Double(dim) * Double(dim) * Double(seqLen)
                    + 2.0 * 2.0 * Double(dim) * Double(hidden) * Double(seqLen)
                    + 2.0 * Double(hidden) * Double(dim) * Double(seqLen))
            let sdpaFlops = Double(nLayers)
                * 2.0 * Double(heads) * 5.0 * Double(seqLen) * Double(seqLen) * Double(ModelConfig.headDim)
            let clsFlops = 2.0 * Double(vocab) * Double(dim) * Double(seqLen)
            let totalFlops = (fwdFlops * 3.0 + sdpaFlops + clsFlops * 3.0) * Double(steps)
            let aneFlops = (fwdFlops * 2.0 + sdpaFlops) * Double(steps)

            print("\n=== Efficiency Report ===")
            print(String(format: "Total steps:     %d", locale: posix, steps))
            print(String(format: "Wall time:       %.0f ms (%.1f s)", locale: posix, wall, wall / 1000.0))
            print(String(format: "Compile time:    %.0f ms (%.1f%%)", locale: posix, compile, 100.0 * compile / wall))
            print(String(format: "Train time:      %.0f ms (%.1f%%)", locale: posix, train, 100.0 * train / wall))
            print(String(format: "Avg train:       %.1f ms/step", locale: posix, train / Double(max(steps, 1))))
            print(String(format: "ANE TFLOPS:      %.2f sustained", locale: posix, aneFlops / (train * 1e9)))
            print(String(format: "Total TFLOPS:    %.2f (ANE+CPU)", locale: posix, totalFlops / (train * 1e9)))
            print(String(format: "ANE utilization: %.1f%% of 15.8 TFLOPS", locale: posix, 100.0 * (aneFlops / (train * 1e9)) / 15.8))
        }

        return .finished
    }
}
