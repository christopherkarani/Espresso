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
        var twoStepStudentSidecarPath: String?
        var generationModelExportPath: String?
        var localTextDatasetPath: String?
        var localBigramPrefix: String?
        var artifactLayerCount: Int = 6
        var offlineAcceptanceJSONPath: String?
        var offlineRecurrentCheckpointPath: String?
        var offlineFutureSidecarPath: String?
        var promptToken: UInt16?
        var gateMaxNewTokens: Int = 8
        var textRoots: [String] = []
        var textExtensions: Set<String> = ["swift", "md", "txt", "py", "sh", "m", "h", "c"]
        var maxCorpusFiles: Int?
        var maxCorpusBytes: Int?
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
            case "--export-two-step-student":
                if i + 1 < argv.count { a.twoStepStudentSidecarPath = argv[i + 1]; i += 1 }
            case "--export-generation-model":
                if i + 1 < argv.count { a.generationModelExportPath = argv[i + 1]; i += 1 }
            case "--build-local-text-dataset":
                if i + 1 < argv.count { a.localTextDatasetPath = argv[i + 1]; i += 1 }
            case "--export-local-bigram-prefix":
                if i + 1 < argv.count { a.localBigramPrefix = argv[i + 1]; i += 1 }
            case "--artifact-layer-count":
                if i + 1 < argv.count { a.artifactLayerCount = Int(argv[i + 1]) ?? a.artifactLayerCount; i += 1 }
            case "--offline-acceptance-json":
                if i + 1 < argv.count { a.offlineAcceptanceJSONPath = argv[i + 1]; i += 1 }
            case "--offline-recurrent-checkpoint":
                if i + 1 < argv.count { a.offlineRecurrentCheckpointPath = argv[i + 1]; i += 1 }
            case "--offline-future-sidecar":
                if i + 1 < argv.count { a.offlineFutureSidecarPath = argv[i + 1]; i += 1 }
            case "--prompt-token":
                if i + 1 < argv.count, let token = UInt16(argv[i + 1]) { a.promptToken = token; i += 1 }
            case "--gate-max-new-tokens":
                if i + 1 < argv.count { a.gateMaxNewTokens = Int(argv[i + 1]) ?? a.gateMaxNewTokens; i += 1 }
            case "--text-root":
                if i + 1 < argv.count { a.textRoots.append(argv[i + 1]); i += 1 }
            case "--text-ext":
                if i + 1 < argv.count {
                    let ext = argv[i + 1].trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
                    if !ext.isEmpty {
                        a.textExtensions.insert(ext)
                    }
                    i += 1
                }
            case "--max-corpus-files":
                if i + 1 < argv.count { a.maxCorpusFiles = Int(argv[i + 1]); i += 1 }
            case "--max-corpus-bytes":
                if i + 1 < argv.count { a.maxCorpusBytes = Int(argv[i + 1]); i += 1 }
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

        if let datasetPath = args.localTextDatasetPath {
            let roots = args.textRoots.isEmpty ? [FileManager.default.currentDirectoryPath] : args.textRoots
            let tokens = try LocalTextTokenDatasetBuilder.collectTokens(
                roots: roots,
                allowedExtensions: args.textExtensions,
                maxFiles: args.maxCorpusFiles,
                maxBytes: args.maxCorpusBytes
            )
            try LocalTextTokenDatasetBuilder.writeUInt16Dataset(tokens: tokens, to: datasetPath)
            print("[built local text dataset: \(datasetPath) tokens=\(tokens.count) roots=\(roots.count)]")
            return .finished
        }

        var offlineRecurrentCheckpointPath = args.offlineRecurrentCheckpointPath
        var offlineFutureSidecarPath = args.offlineFutureSidecarPath
        var offlinePromptToken = args.promptToken

        if let prefix = args.localBigramPrefix {
            let manifest = try LocalRealArtifactPipeline.exportLocalBigramArtifacts(
                datasetPath: args.dataPath,
                prefix: prefix,
                layerCount: args.artifactLayerCount,
                vocabSize: vocab
            )
            offlineRecurrentCheckpointPath = offlineRecurrentCheckpointPath ?? manifest.recurrentCheckpointPath
            offlineFutureSidecarPath = offlineFutureSidecarPath ?? manifest.futureSidecarPath
            offlinePromptToken = offlinePromptToken ?? manifest.promptToken
            print(
                "[exported local bigram artifacts: manifest=\(manifest.manifestPath) prompt=\(manifest.promptToken) tokens=\(manifest.tokenCount)]"
            )
            if args.offlineAcceptanceJSONPath == nil {
                return .finished
            }
        }

        if let offlineAcceptanceJSONPath = args.offlineAcceptanceJSONPath {
            guard let recurrentCheckpointPath = offlineRecurrentCheckpointPath else {
                throw GenerationError.invalidArguments("offline gate requires --offline-recurrent-checkpoint or --export-local-bigram-prefix")
            }
            guard let futureSidecarPath = offlineFutureSidecarPath else {
                throw GenerationError.invalidArguments("offline gate requires --offline-future-sidecar or --export-local-bigram-prefix")
            }
            guard let promptToken = offlinePromptToken else {
                throw GenerationError.invalidArguments("offline gate requires --prompt-token or an exported local-bigram manifest")
            }

            let trace = try LocalRealArtifactPipeline.offlineAcceptanceGate(
                recurrentCheckpointPath: recurrentCheckpointPath,
                futureSidecarPath: futureSidecarPath,
                promptTokens: [promptToken],
                maxNewTokens: args.gateMaxNewTokens
            )
            let payload: [String: Any] = [
                "prompt_tokens": [Int(promptToken)],
                "generated_tokens": trace.generatedTokens.map(Int.init),
                "committed_exact_token_counts": trace.committedExactTokenCounts,
                "accepted_future_token_counts": trace.acceptedFutureTokenCounts,
                "parity_status": trace.parityMatchedAllCommittedTokens ? "match" : "mismatch",
                "committed_exact_tokens_per_pass": trace.committedExactTokensPerPass,
                "accepted_future_tokens_per_pass": trace.acceptedFutureTokensPerPass,
                "recurrent_checkpoint": recurrentCheckpointPath,
                "future_sidecar": futureSidecarPath,
            ]
            let data = try JSONSerialization.data(withJSONObject: payload, options: [.prettyPrinted, .sortedKeys])
            try data.write(to: URL(fileURLWithPath: offlineAcceptanceJSONPath), options: .atomic)
            print("[wrote offline acceptance gate: \(offlineAcceptanceJSONPath)]")
            return .finished
        }

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

        if let generationModelPath = args.generationModelExportPath {
            let emptyClassifier = TensorBuffer(count: 0, zeroed: true)
            try GenerationModelWeightStore.save(
                layers: layers,
                rmsFinal: rmsFinal,
                embed: embed,
                classifier: emptyClassifier,
                sharedClassifier: true,
                vocabSize: vocab,
                to: generationModelPath
            )
            print("[exported generation model: \(generationModelPath)]")
            if args.twoStepStudentSidecarPath == nil {
                return .finished
            }
        }

        if let sidecarPath = args.twoStepStudentSidecarPath {
            let emptyClassifier = TensorBuffer(count: 0, zeroed: true)
            let sidecar = try TwoStepStudentCheckpoint.seedFromTeacher(
                dim: dim,
                vocabSize: vocab,
                layerCount: nLayers,
                teacherRMS: rmsFinal,
                teacherEmbedding: embed,
                teacherClassifier: emptyClassifier,
                teacherClassifierWasShared: true
            )
            try TwoStepStudentCheckpoint.save(path: sidecarPath, sidecar: sidecar)
            print("[exported two-step student sidecar: \(sidecarPath)]")
            return .finished
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
        let rmsWorkspace = RMSNorm.Workspace(seqLen: seqLen)
        let crossEntropyWorkspace = CrossEntropy.Workspace(vocabSize: vocab, seqLen: seqLen)

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

            let surfaceHandles: [LayerSurfaceHandles]
            do {
                surfaceHandles = try SurfaceHandleCache.build(kernels: kernelStorage, staticKernels: staticKernels)
            } catch {
                try? CompileBudget.setCount(ModelConfig.maxCompiles)
                continue
            }

            // Zero gradient accumulators (accumulate across micro-steps).
            for L in 0..<nLayers { grads[L].zero() }
            grmsFinal.zero()
            gembed.zero()

            var stepsBatch = 0
            let tt0 = MachTime.now()
            var batchTimings = StepTimingBreakdown()
            var batchStepMsAccum: Double = 0

            for _ in 0..<ModelConfig.accumSteps where step < totalSteps {
                let stepT0 = MachTime.now()
                let currentStep = step
                var stepTimings = StepTimingBreakdown()
                var t0 = MachTime.now()

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
                stepTimings.tElem += MachTime.ms(MachTime.now() - t0)

                // Wait for prior async dW work before touching layer IO for this step.
                t0 = MachTime.now()
                accumulator.barrier()
                stepTimings.tCblasWait += MachTime.ms(MachTime.now() - t0)

                // Forward pass (12 layers).
                try ForwardPass.runTimed(
                    xCur: xCur,
                    acts: acts,
                    kernels: kernelStorage,
                    accumulator: accumulator,
                    dim: dim,
                    hidden: hidden,
                    seqLen: seqLen,
                    surfaceHandles: surfaceHandles,
                    timings: &stepTimings
                )

                // Final RMSNorm.
                t0 = MachTime.now()
                xFinal.withUnsafeMutablePointer { outPtr in
                    xCur.withUnsafePointer { inPtr in
                        rmsFinal.withUnsafePointer { wPtr in
                            RMSNorm.forward(
                                output: outPtr,
                                input: inPtr,
                                weights: wPtr,
                                dim: dim,
                                seqLen: seqLen,
                                workspace: rmsWorkspace
                            )
                        }
                    }
                }
                stepTimings.tRms += MachTime.ms(MachTime.now() - t0)

                // Classifier: logits = embed @ xFinal.
                t0 = MachTime.now()
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
                stepTimings.tCls += MachTime.ms(MachTime.now() - t0)

                // Cross-entropy loss + dlogits.
                t0 = MachTime.now()
                let loss = dlogits.withUnsafeMutablePointer { dlogitsPtr in
                    logits.withUnsafePointer { logitsPtr in
                        CrossEntropy.lossAndGradient(
                            dlogits: dlogitsPtr,
                            logits: logitsPtr,
                            targets: targetTokens,
                            vocabSize: vocab,
                            seqLen: seqLen,
                            workspace: crossEntropyWorkspace
                        )
                    }
                }
                lastLoss = loss
                stepTimings.tElem += MachTime.ms(MachTime.now() - t0)

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
                                        seqLen: seqLen,
                                        workspace: rmsWorkspace
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
                try BackwardPass.runTimed(
                    dy: dy,
                    acts: acts,
                    kernels: kernelStorage,
                    staticKernels: staticKernels,
                    grads: grads,
                    weights: layers,
                    scratch: bwdScratch,
                    rmsWorkspace: rmsWorkspace,
                    accumulator: accumulator,
                    dim: dim,
                    hidden: hidden,
                    seqLen: seqLen,
                    heads: heads,
                    surfaceHandles: surfaceHandles,
                    timings: &stepTimings
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
                batchStepMsAccum += stepMs
                let completedSteps = stepsBatch + 1
                batchTimings.tAne += stepTimings.tAne
                batchTimings.tIO += stepTimings.tIO
                batchTimings.tCls += stepTimings.tCls
                batchTimings.tElem += stepTimings.tElem
                batchTimings.tRms += stepTimings.tRms
                batchTimings.tCblasWait += stepTimings.tCblasWait
                stderrLine(
                    String(
                        format: "{\"type\":\"step\",\"step\":%d,\"loss\":%.6f,\"ms\":%.3f,\"ms_per_step\":%.3f,\"t_ane\":%.3f,\"t_io\":%.3f,\"t_cls\":%.3f,\"t_elem\":%.3f,\"t_rms\":%.3f,\"t_cblas_wait\":%.3f,\"compiles\":%d}",
                        locale: posix,
                        currentStep,
                        lastLoss,
                        stepMs,
                        batchStepMsAccum / Double(completedSteps),
                        batchTimings.tAne / Double(completedSteps),
                        batchTimings.tIO / Double(completedSteps),
                        batchTimings.tCls / Double(completedSteps),
                        batchTimings.tElem / Double(completedSteps),
                        batchTimings.tRms / Double(completedSteps),
                        batchTimings.tCblasWait / Double(completedSteps),
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
            print(
                String(
                    format: "    ane=%.1f io=%.1f cls=%.1f elem=%.1f rms=%.1f cblas_wait=%.1f ms/step",
                    locale: posix,
                    batchTimings.tAne / Double(stepsBatch),
                    batchTimings.tIO / Double(stepsBatch),
                    batchTimings.tCls / Double(stepsBatch),
                    batchTimings.tElem / Double(stepsBatch),
                    batchTimings.tRms / Double(stepsBatch),
                    batchTimings.tCblasWait / Double(stepsBatch)
                )
            )

            stderrLine(
                String(
                    format: "{\"type\":\"batch\",\"batch\":%d,\"compile_ms\":%.1f,\"train_ms\":%.1f,\"ms_per_step\":%.1f,\"t_ane\":%.3f,\"t_io\":%.3f,\"t_cls\":%.3f,\"t_elem\":%.3f,\"t_rms\":%.3f,\"t_cblas_wait\":%.3f}",
                    locale: posix,
                    stepsBatch,
                    cms,
                    tms,
                    tms / Double(stepsBatch),
                    batchTimings.tAne / Double(stepsBatch),
                    batchTimings.tIO / Double(stepsBatch),
                    batchTimings.tCls / Double(stepsBatch),
                    batchTimings.tElem / Double(stepsBatch),
                    batchTimings.tRms / Double(stepsBatch),
                    batchTimings.tCblasWait / Double(stepsBatch)
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
