# Phase 6: Espresso — Forward, Backward, Training Loop, Checkpoint

<role>
You are a senior Swift 6.2 systems engineer completing the final phase of a bottom-up Swift rewrite of an ANE (Apple Neural Engine) training codebase (~6,100 lines Obj-C/C). Phases 1-5 are implemented and passing all 111 tests. Your task is Phase 6: rewrite the entire `train_large.m` training loop into Swift across 7 files in the `Espresso` target + 1 entry point in `EspressoTrain`. This is the largest phase — it ties every prior phase together into a production training pipeline.

You have deep expertise in:
- Swift 6.2 strict concurrency, `~Copyable` ownership, `borrowing`/`consuming` semantics
- Apple Neural Engine kernel compilation via private APIs (wrapped in `ANEInterop` C shim)
- IOSurface-based fp16 data transfer between CPU and ANE
- Accelerate/vDSP/cblas for CPU numerical operations
- GCD serial queue patterns for async gradient accumulation
- POSIX APIs (`mmap`, `execl`, `srand48`/`drand48`) via `import Darwin`
</role>

---

<context>

## Implementation Roadmap

**You MUST follow the todolist in `tasks/phase6-todolist.md` step-by-step.** This todolist is your progress tracker — mark items `[x]` as you complete them. Work through groups in order. Run the verification gate at the end of each group before proceeding to the next.

The todolist organizes work into 6 groups:
- **Group 0**: Pre-flight checks (verify baseline)
- **Group 1**: Leaf types (GradientAccumulator, TokenDataset, Sampler, ExecRestart, Checkpoint)
- **Group 2**: Forward pass
- **Group 3**: Backward pass (includes SendableBuffer helpers)
- **Group 4**: Training loop (main.swift) + integration tests
- **Group 5**: Final verification

---

## Project State

**Completed phases (all tested, all passing):**
- Phase 1: ANEInterop — C/ObjC shim for private ANE APIs (`ane_interop_compile`, `ane_interop_eval`, `ane_interop_free`, `ane_interop_io_copy`, `ane_interop_io_write_fp16_at`, fp16 conversion, compile counter)
- Phase 2: ANETypes — `ModelConfig`, `TensorBuffer: ~Copyable`, `LayerStorage<Element: ~Copyable>`, `LayerWeights`, `LayerActivations`, `LayerGradients`, `AdamState`, `LayerAdam`, `CheckpointHeader`, `SurfaceIO`, `WeightBlob`
- Phase 3: MILGenerator — 6 MIL program generators (`SDPAForwardGenerator`, `FFNForwardGenerator`, `FFNBackwardGenerator`, `SDPABackward1Generator`, `SDPABackward2Generator`, `QKVBackwardGenerator`) + `GenericMIL` + `CausalMask`
- Phase 4: CPUOps — `RMSNorm` (forward/backward), `CrossEntropy` (lossAndGradient), `AdamOptimizer` (update), `Embedding` (lookup/backward), `RoPE`, `SiLU`
- Phase 5: ANERuntime — `ANEKernel: ~Copyable`, `ANEError`, `CompileBudget`, `LayerKernelSet: ~Copyable` (5 weight-bearing kernels), `StaticKernel: ~Copyable` (weight-free sdpaBwd2), `ModelWeightLoader`

**Test status:** 111 tests executed, 20 skipped (ANE hardware-gated), 0 failures.

**SwiftPM targets:**
- `Espresso` (depends on `ANERuntime`, `CPUOps`, `ANETypes`) — your implementation target
- `EspressoTrain` (executable, depends on `Espresso`) — your entry point target
- `EspressoTests` — your test target

---

## Files to Implement

| # | File | Description | Maps to ObjC source |
|---|------|-------------|---------------------|
| 1 | `Sources/Espresso/GradientAccumulator.swift` | Serial DispatchQueue + DispatchGroup for async cblas dW | `train_large.m:309-310` (`dw_q`, `dw_grp`), used at lines 392, 448, 483, 503, 526, 578, 601 |
| 2 | `Sources/Espresso/TokenDataset.swift` | mmap binary token file as UInt16 array | `train_large.m:274-281` |
| 3 | `Sources/Espresso/Sampler.swift` | Deterministic sampling via `srand48`/`drand48` | `train_large.m:317, 374-377` |
| 4 | `Sources/Espresso/ExecRestart.swift` | `execl()` restart when compile budget exhausted | `train_large.m:322-333` |
| 5 | `Sources/Espresso/Checkpoint.swift` | Save/load binary checkpoint matching ObjC format exactly | `train_large.m:110-181` |
| 6 | `Sources/Espresso/ForwardPass.swift` | 12-layer forward: ANE eval + residual connections | `train_large.m:384-420` |
| 7 | `Sources/Espresso/BackwardPass.swift` | Reverse-order backward: ANE kernels + async cblas dW | `train_large.m:461-575` |
| 8 | `Sources/EspressoTrain/main.swift` | Entry point: arg parsing, training loop, telemetry | `train_large.m:184-687` |

</context>

---

<objective_c_source>

## ObjC Source Being Ported

Below is the complete `train_large.m` (688 lines). Every function has been annotated with the Swift API it maps to and the specific behavioral contract you must preserve.

### Checkpoint Save/Load (train_large.m:110-181)

**What this does**: Saves/loads the complete training state to a binary file. The checkpoint contains a 96-byte header (`CheckpointHeader`, already implemented in Phase 2), followed by per-layer weight+Adam data, followed by global parameters. The critical invariant is **segment order** — the checkpoint must be byte-compatible with the ObjC version for cross-loading.

**Why the segment order matters**: The checkpoint format writes **per-layer** (all of layer L's weights and Adam states, then layer L+1). This is DIFFERENT from the pretrained format (`ModelWeightLoader`) which writes **per-type** (all Wq across layers, then all Wk, etc.). Mixing these up is the #1 checkpoint bug.

```c
// train_large.m:110-181
static void save_checkpoint(const char *path, int step, int total_steps, float lr, float loss,
                            double cc, double ct, double cw, int cs, int cb, int adam_t,
                            LayerWeights *lw, LayerAdam *la, float *rms_final, AdamState *arms_final,
                            float *embed, AdamState *aembed) {
    FILE *f = fopen(path, "wb");
    CkptHdr h = {0};
    h.magic = 0x424C5A54; h.version = 2;
    h.step = step; h.total_steps = total_steps;
    h.n_layers = NLAYERS; h.vocab_size = VOCAB; h.dim = DIM;
    h.hidden_dim = HIDDEN; h.n_heads = HEADS; h.seq_len = SEQ;
    h.lr = lr; h.loss = loss;
    h.cum_compile = cc; h.cum_train = ct; h.cum_wall = cw;
    h.cum_steps = cs; h.cum_batches = cb; h.adam_t = adam_t;
    fwrite(&h, sizeof(h), 1, f);                    // 96 bytes
    // Per-layer: weights then adam (ALL of layer L before layer L+1)
    for (int L = 0; L < NLAYERS; L++) {
        // Weights: Wq, Wk, Wv, Wo, W1, W2, W3, rms_att, rms_ffn
        fwrite(lw[L].Wq,4,WQ_SZ,f); fwrite(lw[L].Wk,4,WQ_SZ,f);
        fwrite(lw[L].Wv,4,WQ_SZ,f); fwrite(lw[L].Wo,4,WO_SZ,f);
        fwrite(lw[L].W1,4,W1_SZ,f); fwrite(lw[L].W2,4,W2_SZ,f); fwrite(lw[L].W3,4,W3_SZ,f);
        fwrite(lw[L].rms_att,4,DIM,f); fwrite(lw[L].rms_ffn,4,DIM,f);
        // Adam: m then v for each parameter, same order as weights
        fwrite(la[L].Wq.m,4,WQ_SZ,f); fwrite(la[L].Wq.v,4,WQ_SZ,f);
        fwrite(la[L].Wk.m,4,WQ_SZ,f); fwrite(la[L].Wk.v,4,WQ_SZ,f);
        fwrite(la[L].Wv.m,4,WQ_SZ,f); fwrite(la[L].Wv.v,4,WQ_SZ,f);
        fwrite(la[L].Wo.m,4,WO_SZ,f); fwrite(la[L].Wo.v,4,WO_SZ,f);
        fwrite(la[L].W1.m,4,W1_SZ,f); fwrite(la[L].W1.v,4,W1_SZ,f);
        fwrite(la[L].W2.m,4,W2_SZ,f); fwrite(la[L].W2.v,4,W2_SZ,f);
        fwrite(la[L].W3.m,4,W3_SZ,f); fwrite(la[L].W3.v,4,W3_SZ,f);
        fwrite(la[L].rms_att.m,4,DIM,f); fwrite(la[L].rms_att.v,4,DIM,f);
        fwrite(la[L].rms_ffn.m,4,DIM,f); fwrite(la[L].rms_ffn.v,4,DIM,f);
    }
    // Global parameters (after all layers)
    fwrite(rms_final,4,DIM,f);
    fwrite(arms_final->m,4,DIM,f); fwrite(arms_final->v,4,DIM,f);
    fwrite(embed,4,VOCAB*DIM,f);
    fwrite(aembed->m,4,VOCAB*DIM,f); fwrite(aembed->v,4,VOCAB*DIM,f);
    fclose(f);
}

// Load is identical order — read header, validate magic+version, read per-layer, read global
static bool load_checkpoint(const char *path, int *step, ...) {
    FILE *f = fopen(path, "rb");
    if (!f) return false;
    CkptHdr h;
    fread(&h, sizeof(h), 1, f);
    if (h.magic != 0x424C5A54 || h.version != 2) { fclose(f); return false; }
    // ... restore header fields ...
    // ... read per-layer in same order as save ...
    // ... read global in same order as save ...
    fclose(f);
    return true;
}
```

**Swift mapping**: Use `fopen`/`fread`/`fwrite`/`fclose` from `import Darwin`. Use `TensorBuffer.withUnsafePointer { ptr in fwrite(ptr, 4, count, file) }` and `TensorBuffer.withUnsafeMutablePointer { ptr in fread(ptr, 4, count, file) }`. Write `CheckpointHeader` via `withUnsafeBytes(of: &header) { fwrite($0.baseAddress!, 1, $0.count, file) }`.

---

### Forward Pass Per Layer (train_large.m:384-420)

**What this does**: Runs 12 transformer layers sequentially. Each layer executes two fused ANE kernels (fwdAttn for attention, fwdFFN for feed-forward) with residual additions on CPU. Activations are saved to `acts[L]` for the backward pass.

**Why `dispatch_group_wait` before fwdAttn write (line 392)**: The previous layer's backward-pass dW blocks may still be running asynchronously on the serial GCD queue. Writing new data to the fwdAttn input IOSurface while the previous layer's async cblas is reading from the same surface would corrupt data. The barrier ensures safety.

**Why Q/K/V are read to activations in ObjC but NOT in this Swift forward pass**: In the ObjC code, Q/K/V are read at fwdAttn output offsets 1*DIM, 2*DIM, 3*DIM to save them for the backward pass. However, in our Swift implementation the backward pass reads Q/K/V directly from the IOSurface via `io_copy` (they persist on the surface). We still read oOut (offset 0), attnOut (offset 4*DIM), and xnorm (offset 5*DIM) to CPU buffers.

```c
// train_large.m:384-420
for (int L=0; L<NLAYERS; L++) {
    LayerActs *ac = &acts[L];
    memcpy(ac->layer_in, x_cur, SEQ*DIM*4);        // save for rmsnorm bwd

    dispatch_group_wait(dw_grp, DISPATCH_TIME_FOREVER); // BARRIER: wait for async dW
    io_write_fp16(kern[L].fwdAttn->ioIn, x_cur, DIM, SEQ);
    ane_eval(kern[L].fwdAttn);
    io_read_fp16(kern[L].fwdAttn->ioOut, ac->o_out,    0,     DIM, SEQ);     // offset 0
    io_read_fp16(kern[L].fwdAttn->ioOut, ac->attn_out, 4*DIM, DIM, SEQ);     // offset 4*DIM
    io_read_fp16(kern[L].fwdAttn->ioOut, ac->xnorm,    5*DIM, DIM, SEQ);     // offset 5*DIM
    // NOTE: Q(1*DIM), K(2*DIM), V(3*DIM) stay on IOSurface for backward pass

    vDSP_vadd(x_cur, 1, ac->o_out, 1, ac->x2, 1, (vDSP_Length)(SEQ*DIM));   // residual: x2 = x + o

    io_write_fp16(kern[L].fwdFFN->ioIn, ac->x2, DIM, SEQ);
    ane_eval(kern[L].fwdFFN);
    io_read_fp16(kern[L].fwdFFN->ioOut, ac->ffn_out,  0,              DIM,    SEQ);  // offset 0
    io_read_fp16(kern[L].fwdFFN->ioOut, ac->h1,       DIM,            HIDDEN, SEQ);  // offset DIM
    io_read_fp16(kern[L].fwdFFN->ioOut, ac->h3,       DIM+HIDDEN,     HIDDEN, SEQ);  // offset DIM+HIDDEN
    io_read_fp16(kern[L].fwdFFN->ioOut, ac->silu_out, DIM+2*HIDDEN,   HIDDEN, SEQ);  // offset DIM+2*HIDDEN
    io_read_fp16(kern[L].fwdFFN->ioOut, ac->x2norm,   DIM+3*HIDDEN,   DIM,    SEQ);  // offset DIM+3*HIDDEN

    vDSP_vadd(ac->x2, 1, ac->ffn_out, 1, x_cur, 1, (vDSP_Length)(SEQ*DIM)); // residual: x_next = x2 + ffn
}
```

---

### Backward Pass Per Layer (train_large.m:461-575)

**What this does**: Runs 12 layers in REVERSE order. Each layer computes gradients using 3 ANE backward kernels (ffnBwd, sdpaBwd1, sdpaBwd2) + 1 weight-only kernel (qkvBwd), plus CPU-side RMSNorm backward and async cblas weight gradient accumulation. The `dy` buffer carries the gradient backward through layers — at the end of each layer, it's updated to `dy = dx_rms1 + dx2` which combines both skip-connection gradient paths.

**Why async dW blocks use malloc+memcpy (lines 478-491)**: The cblas weight gradient operations (sgemm for dW2, dW1, dW3, dWo, dWq, dWk, dWv) are dispatched asynchronously to overlap with ANE computation. But the input buffers (like dffn, siluOut, etc.) will be overwritten in the next layer's computation. The ObjC code copies them to heap-allocated buffers that the GCD block owns and frees. In Swift, use `SendableBuffer: ~Copyable, @unchecked Sendable` which provides automatic cleanup via `deinit`.

**Why `dv` is read from sdpaBwd1 output, not sdpaBwd2**: The SDPA backward is split into two ANE kernels. sdpaBwd1 computes dv and dscores. sdpaBwd2 takes dscores + Q + K and computes dq and dk. So dv comes from sdpaBwd1.ioOut at offset 0, while dq and dk come from sdpaBwd2.ioOut.

```c
// train_large.m:461-575 (annotated)
for (int L=NLAYERS-1; L>=0; L--) {
    LayerActs *ac = &acts[L];
    LayerGrads *gr = &grads[L];
    memcpy(dffn, dy, SEQ*DIM*4);                     // dffn = dy (residual copy)

    // === STEP 1: FFN backward (ANE) ===
    io_write_fp16_at(kern[L].ffnBwd->ioIn, 0, dffn, DIM, SEQ);           // I/O #1
    io_copy(kern[L].ffnBwd->ioIn, DIM,                                   // I/O #2
            kern[L].fwdFFN->ioOut, DIM, 2*HIDDEN, SEQ);  // copy h1|h3 from fwd output
    ane_eval(kern[L].ffnBwd);
    io_read_fp16(kern[L].ffnBwd->ioOut, dx_ffn, 0,           DIM,    SEQ);  // I/O #3
    io_read_fp16(kern[L].ffnBwd->ioOut, dh1,    DIM,         HIDDEN, SEQ);  // I/O #4
    io_read_fp16(kern[L].ffnBwd->ioOut, dh3,    DIM+HIDDEN,  HIDDEN, SEQ);  // I/O #5

    // === STEP 1b: Async dW FFN (CPU, dispatched to serial queue) ===
    // malloc+memcpy 5 buffers, dispatch cblas: dW2 += dffn@silu^T, dW1 += dh1@x2norm^T, dW3 += dh3@x2norm^T
    float *capt_dffn = malloc(SEQ*DIM*4); memcpy(capt_dffn, dffn, SEQ*DIM*4);
    // ... (captures for silu_out, dh1, dh3, x2norm) ...
    dispatch_group_async(dw_grp, dw_q, ^{
        cblas_sgemm(..., capt_dffn, ..., capt_silu, ..., gr->W2, ...);  // dW2
        cblas_sgemm(..., capt_dh1, ..., capt_x2n, ..., gr->W1, ...);   // dW1
        cblas_sgemm(..., capt_dh3, ..., capt_x2n, ..., gr->W3, ...);   // dW3
        free(capt_dffn); free(capt_silu); free(capt_dh1); free(capt_dh3); free(capt_x2n);
    });

    // === STEP 2: RMSNorm2 backward (CPU) ===
    memset(dx2, 0, SEQ*DIM*4);
    rmsnorm_bwd(dx2, gr->rms_ffn, dx_ffn, ac->x2, lw[L].rms_ffn, DIM, SEQ);
    for(int i=0;i<SEQ*DIM;i++) dx2[i] += dy[i];     // residual: dx2 += dy

    // === STEP 3: Async dWo (CPU) ===
    // capt_do=copy(dx2), capt_attn=copy(attn_out)
    dispatch_group_async(dw_grp, dw_q, ^{
        cblas_sgemm(..., capt_do, ..., capt_attn, ..., gr->Wo, ...);  // dWo
        free(capt_do); free(capt_attn);
    });

    // === STEP 4: SDPA backward (ANE, two kernels) ===
    // sdpaBwd1: input = [Q|K|V|dx2], output = [dv|dscores]
    io_copy(kern[L].sdpaBwd1->ioIn, 0,                                  // I/O #6
            kern[L].fwdAttn->ioOut, DIM, 3*DIM, SEQ);    // copy Q|K|V from fwdAttn
    io_write_fp16_at(kern[L].sdpaBwd1->ioIn, 3*DIM, dx2, DIM, SEQ);    // I/O #7
    ane_eval(kern[L].sdpaBwd1);

    // sdpaBwd2: input = [dscores|Q|K], output = [dq|dk]
    io_copy(sdpaBwd2[L]->ioIn, 0,                                       // I/O #8
            kern[L].sdpaBwd1->ioOut, DIM, 2*SCORE_CH, SEQ);  // copy dscores
    io_copy(sdpaBwd2[L]->ioIn, 2*SCORE_CH,                              // I/O #9
            kern[L].fwdAttn->ioOut, DIM, 2*DIM, SEQ);        // copy Q|K
    ane_eval(sdpaBwd2[L]);

    io_read_fp16(sdpaBwd2[L]->ioOut, dq, 0,   DIM, SEQ);               // I/O #10
    io_read_fp16(sdpaBwd2[L]->ioOut, dk, DIM,  DIM, SEQ);               // I/O #11
    io_read_fp16(kern[L].sdpaBwd1->ioOut, dv, 0, DIM, SEQ);             // I/O #12 (from sdpaBwd1!)

    // === STEP 5: Async dWq/dWk/dWv (CPU) ===
    // capt_dq, capt_dk, capt_dv, capt_xnorm
    dispatch_group_async(dw_grp, dw_q, ^{
        cblas_sgemm(..., capt_dq, ..., capt_xn, ..., gr->Wq, ...);   // dWq
        cblas_sgemm(..., capt_dk, ..., capt_xn, ..., gr->Wk, ...);   // dWk
        cblas_sgemm(..., capt_dv, ..., capt_xn, ..., gr->Wv, ...);   // dWv
        free(capt_dq); free(capt_dk); free(capt_dv); free(capt_xn);
    });

    // === STEP 6: QKV backward (ANE) ===
    io_copy(kern[L].qkvBwd->ioIn, 0,                                    // I/O #13
            sdpaBwd2[L]->ioOut, 0, 2*DIM, SEQ);              // copy dq|dk
    io_copy(kern[L].qkvBwd->ioIn, 2*DIM,                                // I/O #14
            kern[L].sdpaBwd1->ioOut, 0, DIM, SEQ);           // copy dv
    ane_eval(kern[L].qkvBwd);
    io_read_fp16(kern[L].qkvBwd->ioOut, dx_attn, 0, DIM, SEQ);         // I/O #15

    // === STEP 7: RMSNorm1 backward (CPU) ===
    float *dx_rms1 = calloc(SEQ*DIM, 4);
    rmsnorm_bwd(dx_rms1, gr->rms_att, dx_attn, ac->layer_in, lw[L].rms_att, DIM, SEQ);
    for(int i=0;i<SEQ*DIM;i++) dy[i] = dx_rms1[i] + dx2[i];  // propagate to prev layer
    free(dx_rms1);
}
```

---

### Gradient Accumulation + Adam Update (train_large.m:601-628)

**What this does**: After ACCUM_STEPS inner steps, waits for all async dW to finish, scales all accumulated gradients by `1.0/steps_batch`, increments adam_t, then applies Adam to all parameters.

**Why scale by 1/steps_batch**: Gradients are accumulated (summed) over multiple micro-steps. Dividing by the step count gives the mean gradient, which is what Adam expects. If steps_batch < ACCUM_STEPS (because we hit total_steps limit), the divisor is the ACTUAL number of steps taken, not ACCUM_STEPS.

**Why adam_t starts at 1**: Adam bias correction uses `1 - beta^t`. At t=0, this produces division by zero. The ObjC code increments adam_t BEFORE calling adam_update. The Swift `AdamOptimizer.update` function has `precondition(timestep >= 1)`.

```c
// train_large.m:601-628
dispatch_group_wait(dw_grp, DISPATCH_TIME_FOREVER);  // wait ALL async dW

float gsc = 1.0f / steps_batch;      // mean over actual accumulated steps
adam_t++;                              // increment BEFORE update (adam_t starts at 0, first call uses 1)

for (int L=0; L<NLAYERS; L++) {
    // Scale all 9 gradient buffers
    for(size_t i=0;i<WQ_SZ;i++){g->Wq[i]*=gsc; g->Wk[i]*=gsc; g->Wv[i]*=gsc; g->Wo[i]*=gsc;}
    for(size_t i=0;i<W1_SZ;i++) g->W1[i]*=gsc;
    for(size_t i=0;i<W2_SZ;i++) g->W2[i]*=gsc;
    for(size_t i=0;i<W3_SZ;i++) g->W3[i]*=gsc;
    for(int i=0;i<DIM;i++){g->rms_att[i]*=gsc; g->rms_ffn[i]*=gsc;}
    // Adam update for all 9 parameters
    adam_update(lw[L].Wq, g->Wq, &la[L].Wq, adam_t, lr, adam_b1, adam_b2, adam_eps);
    // ... (Wk, Wv, Wo, W1, W2, W3, rms_att, rms_ffn) ...
}
// Global gradients: scale + Adam
for(int i=0;i<DIM;i++) grms_final[i]*=gsc;
adam_update(rms_final, grms_final, &arms_final, adam_t, lr, adam_b1, adam_b2, adam_eps);
for(size_t i=0;i<(size_t)VOCAB*DIM;i++) gembed[i]*=gsc;
adam_update(embed, gembed, &aembed, adam_t, lr, adam_b1, adam_b2, adam_eps);
```

---

### Token Data, Sampling, and Exec Restart

**Token Data (train_large.m:274-281)**: The training data is a flat binary file of pretokenized UInt16 values. Mapped read-only via `mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0)`. Token at position `pos` is `token_data[pos]`. Input tokens start at `pos`, target tokens at `pos+1` (teacher forcing).

```c
int data_fd = open(DATA_PATH, O_RDONLY);
struct stat st; fstat(data_fd, &st);
size_t data_len = st.st_size;
uint16_t *token_data = (uint16_t*)mmap(NULL, data_len, PROT_READ, MAP_PRIVATE, data_fd, 0);
size_t n_tokens = data_len / 2;
```

**Sampling (train_large.m:317, 374-377)**: The random sampling MUST use `srand48(42 + startStep)` and `drand48()` exactly — this ensures bit-for-bit reproducible training sequences that match the ObjC reference. The position is `Int(drand48() * Double(maxPos))` where `maxPos = nTokens - seqLen - 1`.

```c
srand48(42 + start_step);                    // seed once before training loop
// ... inside accumulation loop:
size_t max_pos = n_tokens - SEQ - 1;
size_t pos = (size_t)(drand48() * max_pos);
uint16_t *input_tokens = token_data + pos;
uint16_t *target_tokens = token_data + pos + 1;
```

**Exec Restart (train_large.m:322-333)**: The ANE has a compile budget (~100 compilations before the driver refuses). When budget is exhausted, the process saves a checkpoint and calls `execl()` to restart itself with `--resume`. This replaces the process image — NO Swift deinit runs, NO ARC cleanup. Free ANE kernels and save checkpoint BEFORE calling execl.

```c
if (g_compile_count + TOTAL_WEIGHT_KERNELS > MAX_COMPILES) {
    // 1. free all kernels (weight-bearing + static sdpaBwd2)
    for (int L=0; L<NLAYERS; L++) { free_layer_kernels(&kern[L]); free_kern(sdpaBwd2[L]); }
    // 2. save checkpoint
    save_checkpoint(CKPT_PATH, step, total_steps, lr, last_loss, ...);
    // 3. print + flush
    printf("[exec() restart step %d, %d compiles, loss=%.4f]\n", step, g_compile_count, last_loss);
    fflush(stdout);
    // 4. replace process image
    execl(argv[0], argv[0], "--resume", NULL);
    perror("execl"); return 1;
}
```

### Compile Failure Recovery (train_large.m:348)

**What this does**: If any layer's kernel fails to compile, the ObjC code forces the compile count to MAX_COMPILES and continues the loop. The next iteration will hit the compile-budget check, save a checkpoint, and exec-restart.

```c
if (!compile_ok) { g_compile_count = MAX_COMPILES; continue; }
```

**Swift equivalent**: Call `try CompileBudget.setCount(ModelConfig.maxCompiles)` and `continue` the outer loop.

</objective_c_source>

---

<swift_api_reference>

## Swift APIs Available (Implemented in Phases 1-5)

These are the ACTUAL signatures from the implemented codebase. Use these exactly — do not guess at APIs.

### ModelConfig (Sources/ANETypes/ModelConfig.swift)
```swift
public enum ModelConfig {
    public static let dim = 768
    public static let hidden = 2048        // Matches ObjC HIDDEN=2048 (stories_config.h:19)
    public static let heads = 12           // but ModelConfig.hidden = 2048 is what's implemented
    public static let seqLen = 256
    public static let nLayers = 12
    public static let vocab = 32_000
    public static let accumSteps = 10
    public static let maxCompiles = 100
    public static let kernelsPerLayer = 5
    public static let totalWeightKernels = kernelsPerLayer * nLayers  // 60
    public static let headDim = dim / heads     // 64
    public static let scoreCh = heads * seqLen  // 3072
    public static let wqSize = dim * dim        // 589,824
    public static let woSize = dim * dim
    public static let w1Size = hidden * dim     // 1,572,864
    public static let w2Size = dim * hidden
    public static let w3Size = hidden * dim
    public static let layerParams = 4 * wqSize + w1Size + w2Size + w3Size + 2 * dim
    public static let totalParams = nLayers * layerParams + dim + vocab * dim
}
```

### TensorBuffer (Sources/ANETypes/TensorBuffer.swift)
```swift
public struct TensorBuffer: ~Copyable {
    public let count: Int
    public init(count: Int, zeroed: Bool)
    public func withUnsafeMutablePointer<R>(_ body: (UnsafeMutablePointer<Float>) throws -> R) rethrows -> R
    public func withUnsafePointer<R>(_ body: (UnsafePointer<Float>) throws -> R) rethrows -> R
    public func withUnsafeMutableBufferPointer<R>(_ body: (UnsafeMutableBufferPointer<Float>) throws -> R) rethrows -> R
    public func withUnsafeBufferPointer<R>(_ body: (UnsafeBufferPointer<Float>) throws -> R) rethrows -> R
    public func zero()
    deinit  // deallocates underlying memory
}
```

### LayerStorage (Sources/ANETypes/LayerStorage.swift)
```swift
public struct LayerStorage<Element: ~Copyable>: ~Copyable {
    public let count: Int
    public init(count: Int, initializer: (Int) -> Element)
    public subscript(index: Int) -> Element {
        _read { yield storage[index] }
        _modify { yield &storage[index] }
    }
    deinit  // deinitializes all elements and deallocates
}
```

### LayerWeights / LayerGradients / LayerActivations / AdamState / LayerAdam
```swift
// LayerWeights: ~Copyable — 9 TensorBuffers (Wq, Wk, Wv, Wo, W1, W2, W3, rmsAtt, rmsFfn)
public struct LayerWeights: ~Copyable { public init() }

// LayerGradients: ~Copyable — 9 TensorBuffers, init zeroed, has zero() method
public struct LayerGradients: ~Copyable {
    public init()
    public func zero()  // zeros all 9 buffers
}

// LayerActivations: ~Copyable — 13 TensorBuffers
public struct LayerActivations: ~Copyable {
    public let layerIn, xnorm, Q, K, V, attnOut, oOut, x2, x2norm, h1, h3, siluOut, ffnOut: TensorBuffer
    public init()
}

// AdamState: ~Copyable — m and v TensorBuffers, init zeroed
public struct AdamState: ~Copyable {
    public let m, v: TensorBuffer
    public let count: Int
    public init(count: Int)
}

// LayerAdam: ~Copyable — 9 AdamStates matching LayerWeights
public struct LayerAdam: ~Copyable { public init() }
```

### CheckpointHeader (Sources/ANETypes/CheckpointHeader.swift)
```swift
@frozen public struct CheckpointHeader {
    // 96 bytes total, magic=0x424C5A54, version=2
    public var magic, version, step, totalSteps, nLayers, vocabSize, dim, hiddenDim, nHeads, seqLen: Int32
    public var lr, loss: Float
    public var cumCompile, cumTrain, cumWall: Double
    public var cumSteps, cumBatches, adamT, pad0, pad1, pad2: Int32
    public init()  // zeros all fields, sets magic and version
    public static func validateLayout()  // precondition checks all 20+ field offsets
}
```

### SurfaceIO (Sources/ANETypes/SurfaceIO.swift)
```swift
public enum SurfaceIO {
    // Write fp32 buffer → IOSurface as fp16 (full surface, offset 0)
    public static func writeFP16(to surface: IOSurfaceRef,
                                 data: UnsafeBufferPointer<Float>,
                                 channels: Int, spatial: Int)

    // Read fp16 from IOSurface at channel offset → fp32 buffer
    public static func readFP16(from surface: IOSurfaceRef,
                                into dst: UnsafeMutableBufferPointer<Float>,
                                channelOffset: Int, channels: Int, spatial: Int)

    // Write fp32 buffer → IOSurface at channel offset as fp16
    public static func writeFP16At(to surface: IOSurfaceRef,
                                   channelOffset: Int,
                                   data: UnsafeBufferPointer<Float>,
                                   channels: Int, spatial: Int) throws(SurfaceIOError)

    // Copy fp16 region between IOSurfaces
    public static func copyFP16(dst: IOSurfaceRef, dstChannelOffset: Int,
                                src: IOSurfaceRef, srcChannelOffset: Int,
                                channels: Int, spatial: Int) throws(SurfaceIOError)
}
```

### ANEKernel (Sources/ANERuntime/ANEKernel.swift)
```swift
public struct ANEKernel: ~Copyable {
    // Compile MIL program with weights → ANE kernel
    public init(milText: String, weights: [(path: String, data: Data)],
                inputBytes: Int, outputBytes: Int, checkBudget: Bool = true) throws(ANEError)
    public func eval() throws(ANEError)
    public func inputSurface(at index: Int) throws(ANEError) -> IOSurfaceRef   // retained copy
    public func outputSurface(at index: Int) throws(ANEError) -> IOSurfaceRef  // retained copy
    deinit  // calls ane_interop_free
}
```

### LayerKernelSet (Sources/ANERuntime/LayerKernelSet.swift)
```swift
public struct LayerKernelSet: ~Copyable {
    public let fwdAttn, fwdFFN, ffnBwd, sdpaBwd1, qkvBwd: ANEKernel
    public init(weights: borrowing LayerWeights) throws(ANEError)  // compiles all 5 kernels
}
```

### StaticKernel (Sources/ANERuntime/StaticKernel.swift)
```swift
public struct StaticKernel: ~Copyable {
    public let kernel: ANEKernel
    public init() throws(ANEError)  // compiles weight-free sdpaBwd2
}
```

### CompileBudget (Sources/ANERuntime/CompileBudget.swift)
```swift
public enum CompileBudget {
    public static let maxCompiles: Int
    public static var currentCount: Int { get }
    public static var isExhausted: Bool { get }
    public static func setCount(_ value: Int) throws(ANEError)
    public static var remaining: Int { get }
}
```

### ModelWeightLoader (Sources/ANERuntime/ModelWeightLoader.swift)
```swift
public struct PretrainedWeights: ~Copyable {
    public let layers: LayerStorage<LayerWeights>
    public let rmsFinal: TensorBuffer
    public let embed: TensorBuffer
    public let sharedClassifier: Bool
}
public enum ModelWeightLoader {
    public static func load(from path: String) throws(ModelLoadError) -> PretrainedWeights
}
```

### CPUOps
```swift
public enum RMSNorm {
    // Channel-first forward: x[dim, seq], w[dim] → out[dim, seq]
    public static func forward(output:input:weights: dim:seqLen:)
    // Channel-first backward: computes dx, ACCUMULATES into dw (dw[i] += ...)
    public static func backward(dx:dw:dy:x:weights: dim:seqLen:)
}
public enum CrossEntropy {
    // Column-major [vocab, seq]. Returns mean CE loss. Writes gradient into dlogits.
    public static func lossAndGradient(dlogits:logits:targets: vocabSize:seqLen:) -> Float
}
public enum AdamOptimizer {
    // timestep starts at 1, NOT 0
    public static func update(weights:gradients:m:v:count: timestep:lr:beta1:beta2:eps:)
}
public enum Embedding {
    // Channel-first: output[d*seq + t] = embedding[tok*dim + d]
    public static func lookup(output:embedding:tokens: vocabSize:dim:seqLen:)
    // Channel-first backward: ACCUMULATES into dEmbedding
    public static func backward(dEmbedding:dx:tokens: vocabSize:dim:seqLen:)
}
```

</swift_api_reference>

---

<iosurface_layout>

## IOSurface Channel Layouts

All IOSurface data is fp16, channel-first `[channels, spatial]` where `spatial = seqLen = 256`. Channel offsets are in units of channels (not bytes). `SCORE_CH = heads * seqLen = 12 * 256 = 3072`.

| Kernel | Input Layout | Input Channels | Output Layout | Output Channels |
|--------|-------------|----------------|---------------|-----------------|
| fwdAttn | `[x]` | DIM | `[o\|Q\|K\|V\|attn\|xnorm]` | 6*DIM |
| fwdFFN | `[x2]` | DIM | `[ffn\|h1\|h3\|silu\|x2norm]` | 2*DIM+3*HIDDEN |
| ffnBwd | `[dffn\|h1\|h3]` | DIM+2*HIDDEN | `[dx\|dh1\|dh3]` | DIM+2*HIDDEN |
| sdpaBwd1 | `[Q\|K\|V\|dx2]` | 4*DIM | `[dv\|dscores]` | DIM+2*SCORE_CH |
| sdpaBwd2 | `[dscores\|Q\|K]` | 2*SCORE_CH+2*DIM | `[dq\|dk]` | 2*DIM |
| qkvBwd | `[dq\|dk\|dv]` | 3*DIM | `[dx_attn]` | DIM |

### Backward Pass: Complete IOSurface I/O Table (15 Operations Per Layer)

This table maps EXACTLY to the C code. Every row is one IOSurface API call. Get ANY offset wrong and the backward pass produces garbage.

| # | C Function | Destination | Dst Ch Off | Source | Src Ch Off | Channels | Why |
|---|---|---|---|---|---|---|---|
| 1 | `io_write_fp16_at` | ffnBwd.ioIn | 0 | CPU:dffn | - | DIM | Write gradient to ffnBwd |
| 2 | `io_copy` | ffnBwd.ioIn | DIM | fwdFFN.ioOut | DIM | 2*HIDDEN | Copy h1\|h3 from forward (skip ffn_out at offset 0) |
| 3 | `io_read_fp16` | CPU:dx_ffn | - | ffnBwd.ioOut | 0 | DIM | Read FFN input gradient |
| 4 | `io_read_fp16` | CPU:dh1 | - | ffnBwd.ioOut | DIM | HIDDEN | Read h1 gradient |
| 5 | `io_read_fp16` | CPU:dh3 | - | ffnBwd.ioOut | DIM+HIDDEN | HIDDEN | Read h3 gradient |
| 6 | `io_copy` | sdpaBwd1.ioIn | 0 | fwdAttn.ioOut | DIM | 3*DIM | Copy Q\|K\|V from forward (skip o_out at offset 0) |
| 7 | `io_write_fp16_at` | sdpaBwd1.ioIn | 3*DIM | CPU:dx2 | - | DIM | Write attention gradient |
| 8 | `io_copy` | sdpaBwd2.ioIn | 0 | sdpaBwd1.ioOut | DIM | 2*SCORE_CH | Copy dscores (skip dv at offset 0) |
| 9 | `io_copy` | sdpaBwd2.ioIn | 2*SCORE_CH | fwdAttn.ioOut | DIM | 2*DIM | Copy Q\|K for sdpaBwd2 |
| 10 | `io_read_fp16` | CPU:dq | - | sdpaBwd2.ioOut | 0 | DIM | Read query gradient |
| 11 | `io_read_fp16` | CPU:dk | - | sdpaBwd2.ioOut | DIM | DIM | Read key gradient |
| 12 | `io_read_fp16` | CPU:dv | - | sdpaBwd1.ioOut | 0 | DIM | Read value gradient (from sdpaBwd1, NOT sdpaBwd2) |
| 13 | `io_copy` | qkvBwd.ioIn | 0 | sdpaBwd2.ioOut | 0 | 2*DIM | Copy dq\|dk |
| 14 | `io_copy` | qkvBwd.ioIn | 2*DIM | sdpaBwd1.ioOut | 0 | DIM | Copy dv |
| 15 | `io_read_fp16` | CPU:dx_attn | - | qkvBwd.ioOut | 0 | DIM | Read attention input gradient |

</iosurface_layout>

---

<async_dw_pattern>

## Async dW Dispatch Pattern

The ObjC code dispatches weight gradient (dW) computations asynchronously to overlap CPU cblas with ANE kernel evaluation. In Swift, use `SendableBuffer: ~Copyable, @unchecked Sendable` for heap buffer captures, and `SendablePointer: @unchecked Sendable` for gradient accumulator pointer captures.

### SendableBuffer Pattern
```swift
/// Owns an exclusive heap copy of a float buffer.
/// ~Copyable prevents aliasing. @unchecked Sendable because we guarantee
/// no aliasing after construction. deinit provides automatic cleanup.
struct SendableBuffer: ~Copyable, @unchecked Sendable {
    let pointer: UnsafeMutablePointer<Float>
    let count: Int
    init(copying source: UnsafePointer<Float>, count: Int) {
        self.count = count
        self.pointer = .allocate(capacity: count)
        self.pointer.initialize(from: source, count: count)
    }
    deinit { pointer.deinitialize(count: count); pointer.deallocate() }
}

/// Sendable pointer wrapper for gradient accumulators.
/// Safe because: serial queue ensures exclusive access within a batch.
struct SendablePointer: @unchecked Sendable {
    let pointer: UnsafeMutablePointer<Float>
    init(_ p: UnsafeMutablePointer<Float>) { self.pointer = p }
}

/// Sendable const pointer for read-only captures (e.g. dembed block: dlogits, x_final).
struct SendableConstPointer: @unchecked Sendable {
    let pointer: UnsafePointer<Float>
    init(_ p: UnsafePointer<Float>) { self.pointer = p }
}
```

### Usage in Backward Pass (maps to train_large.m:483-491)
```swift
// Snapshot buffers — heap copy, like ObjC malloc+memcpy
var captDffn = SendableBuffer(copying: dffnPtr, count: seqLen * dim)
var captSilu = SendableBuffer(copying: siluPtr, count: seqLen * hidden)
let grW2 = SendablePointer(grads[L].W2.withUnsafeMutablePointer { $0 })

accumulator.enqueue { [captDffn = consume captDffn, captSilu = consume captSilu] in
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                Int32(dim), Int32(hidden), Int32(seq), 1.0,
                captDffn.pointer, Int32(seq),
                captSilu.pointer, Int32(seq),
                1.0, grW2.pointer, Int32(hidden))
    // captDffn and captSilu deinit runs here automatically
}
```

### Async dW Blocks Summary (4 types, 3 barrier points)

| Block | Per Step | Captured Data | cblas Operations | Updates |
|---|---|---|---|---|
| dembed | 1x | dlogits + x_final (const ptrs, no copy) | dlogits @ x_final^T | gembed[VOCAB,DIM] |
| FFN dW | 12x | dffn, siluOut, dh1, dh3, x2norm (5 copies) | dW2, dW1, dW3 | grads[L].W2, W1, W3 |
| dWo | 12x | dx2, attnOut (2 copies) | dWo | grads[L].Wo |
| QKV dW | 12x | dq, dk, dv, xnorm (4 copies) | dWq, dWk, dWv | grads[L].Wq, Wk, Wv |

**Barrier points** (must call `accumulator.barrier()`):
1. Before each fwdAttn IOSurface write (line 392) — ensures prior layer's dW is done
2. Before `Embedding.backward()` (line 578) — ensures dembed block is done
3. Before Adam update (line 601) — ensures ALL dW blocks are done

</async_dw_pattern>

---

<checkpoint_format>

## Checkpoint Binary Format

### Segment Order (MUST match ObjC byte-for-byte)

```
[CheckpointHeader]                   96 bytes
FOR L = 0 ..< nLayers:
  [Weights]
    Wq          dim*dim floats
    Wk          dim*dim floats
    Wv          dim*dim floats
    Wo          dim*dim floats
    W1          hidden*dim floats
    W2          dim*hidden floats
    W3          hidden*dim floats
    rmsAtt      dim floats
    rmsFfn      dim floats
  [Adam m/v pairs]
    Wq.m, Wq.v       each dim*dim floats
    Wk.m, Wk.v       each dim*dim floats
    Wv.m, Wv.v       each dim*dim floats
    Wo.m, Wo.v       each dim*dim floats
    W1.m, W1.v       each hidden*dim floats
    W2.m, W2.v       each dim*hidden floats
    W3.m, W3.v       each hidden*dim floats
    rmsAtt.m, rmsAtt.v   each dim floats
    rmsFfn.m, rmsFfn.v   each dim floats
[Global]
  rmsFinal          dim floats
  adamRmsFinal.m    dim floats
  adamRmsFinal.v    dim floats
  embed             vocab*dim floats
  adamEmbed.m       vocab*dim floats
  adamEmbed.v       vocab*dim floats
```

**Contrast with pretrained format** (ModelWeightLoader reads this):
```
[Header]  7 × Int32
embed     V*dim floats
FOR each type (rmsAtt, Wq, Wk, Wv, Wo, rmsFfn, W1, W2, W3):  ← per-TYPE, not per-layer
  FOR L = 0 ..< nLayers:
    type[L]   size floats
rmsFinal    dim floats
```

</checkpoint_format>

---

<classifier_matmul>

## Classifier (CPU) — cblas_sgemm

The classifier projects the final hidden state through the embedding matrix to produce logits. This is NOT on ANE — it uses cblas_sgemm on CPU because the vocab dimension (32,000) doesn't fit the ANE kernel tile size.

### Forward: logits = embed @ x_final
```c
// train_large.m:428-431
// embed[VOCAB, DIM] row-major, x_final[DIM, SEQ] channel-first
// → logits[VOCAB, SEQ] column-major (what CrossEntropy expects)
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
            VOCAB, SEQ, DIM, 1.0f,
            embed, DIM, x_final, SEQ, 0.0f, logits, SEQ);
```

### Backward: dy = embed^T @ dlogits, dembed += dlogits @ x_final^T
```c
// train_large.m:443-452
// dx_final = embed^T @ dlogits → dy[DIM, SEQ]
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
            DIM, SEQ, VOCAB, 1.0f,
            embed, DIM, dlogits, SEQ, 0.0f, dy, SEQ);

// dembed accumulation (async on serial queue, beta=1.0 for accumulate)
dispatch_group_async(dw_grp, dw_q, ^{
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                VOCAB, DIM, SEQ, 1.0f,
                dlogits, SEQ, x_final, SEQ, 1.0f, gembed, DIM);
});
```

**Swift**: Use `import Accelerate` for `cblas_sgemm`. All Int parameters must be `Int32`. The `beta=1.0f` on the dembed accumulation is critical — it means "add to existing value" not "overwrite".

</classifier_matmul>

</context>

---

<instructions>

## Implementation Rules

### Ownership & Concurrency
1. All `~Copyable` types use `consuming`/`borrowing` parameter ownership where appropriate.
2. `GradientAccumulator` is a `final class` (reference semantics needed for shared state) with `@unchecked Sendable` (safety guaranteed by serial queue).
3. `SendableBuffer`, `SendablePointer`, `SendableConstPointer` are the ONLY types used to capture data into GCD blocks. Never capture raw pointers into closures.
4. Use `[captX = consume captX]` capture lists for `SendableBuffer` to transfer ownership into the closure.

### Numerical Correctness
5. All `cblas_sgemm` dimension parameters are `Int32`, not `Int`.
6. `AdamOptimizer.update` timestep starts at 1. Increment adam_t BEFORE calling update (matches ObjC line 605).
7. Gradient scaling: `1.0 / Float(steps_batch)` applied to ALL gradient buffers including rmsFinal and embed, BEFORE Adam. Not `1.0 / Float(ACCUM_STEPS)` — use actual steps taken.
8. `RMSNorm.backward` ACCUMULATES into dw — it adds to existing values. Zero gradient buffers at the start of each compilation batch (before the accumulation loop), not before each step.
9. `Embedding.backward` ACCUMULATES into dEmbedding — same pattern as RMSNorm.
10. `CrossEntropy.lossAndGradient` expects column-major `[vocab, seq]` layout — which is exactly what `cblas_sgemm` produces for the logits.

### IOSurface I/O
11. Channel offsets are in units of channels (e.g., "offset DIM" means DIM channels), not bytes.
12. `SurfaceIO.readFP16` takes a channel offset. `SurfaceIO.writeFP16` writes at offset 0. `SurfaceIO.writeFP16At` writes at a specified channel offset.
13. The fwdAttn output layout `[o|Q|K|V|attn|xnorm]` places Q at offset DIM (not 0). The backward pass copies Q|K|V starting from fwdAttn.ioOut at srcChannelOffset = DIM.
14. `ANEKernel.inputSurface(at: 0)` and `outputSurface(at: 0)` return retained IOSurfaceRef copies.

### Checkpoint
15. Checkpoint writes use `fopen`/`fwrite`/`fclose` via `import Darwin`. Read header first, validate magic (0x424C5A54) and version (2) before reading payload.
16. All Int fields in header are `Int32`. All weight payload is `Float` (not Double).

### Performance
17. Do NOT allocate memory inside the inner accumulation loop (except for dx_rms1 which must be zeroed each iteration — or use a pre-allocated buffer and zero it). The ObjC code allocates dx_rms_final and dx_rms1 with calloc inside the loop — in Swift you should pre-allocate and zero before use.
18. Scratch buffers (dy, dffn, dx_ffn, etc.) are allocated ONCE before the training loop and reused.

### Testing
19. Gate ANE hardware-dependent tests with:
```swift
guard ProcessInfo.processInfo.environment["ANE_HARDWARE_TESTS"] == "1" else {
    throw XCTSkip("ANE hardware tests disabled")
}
```
20. Gate data-file-dependent tests by checking file existence.

</instructions>

---

<tests>

## TDD Test Specifications

Write ALL tests FIRST in `Tests/EspressoTests/EspressoTests.swift`, then implement until green.

### Test 1: `test_gradient_accumulator_enqueue_and_barrier`
Enqueue 3 blocks that each increment a shared counter. Call barrier. Assert counter == 3.
**Verifies**: GradientAccumulator correctly wraps DispatchQueue + DispatchGroup.

### Test 2: `test_gradient_accumulator_wait_all`
Enqueue 5 blocks with small delays. Call waitAll. Assert all completed.
**Verifies**: waitAll blocks until all enqueued work finishes.

### Test 3: `test_token_dataset_small_file`
Create temp file with 100 UInt16 values (0...99). Open as TokenDataset. Verify nTokens == 100 and data[0] == 0, data[99] == 99.
**Verifies**: mmap-based token access works correctly.

### Test 4: `test_token_dataset_validates_minimum_size`
Create temp file with fewer than seqLen+1 tokens. Verify init throws/fails.
**Verifies**: TokenDataset rejects too-small files.

### Test 5: `test_sampler_deterministic_sequence`
Seed with 42. Call samplePosition 10 times. Seed with 42 again. Call 10 times. Verify same sequence.
**Verifies**: srand48/drand48 reproducibility.

### Test 6: `test_checkpoint_save_load_roundtrip`
Create small synthetic weights (dim=4, hidden=8, nLayers=1, vocab=10). Save to temp file. Load back. Verify all weights are byte-identical (not just approximately equal).
**Verifies**: Checkpoint format correctness.

### Test 7: `test_checkpoint_segment_order_small`
Save checkpoint with known values. Read file bytes directly. Verify header at offset 0, per-layer weights at offset 96, per-layer Adam after weights, global after all layers.
**Verifies**: Binary layout matches ObjC exactly.

### Test 8: `test_checkpoint_header_validation`
Write file with wrong magic (0xDEADBEEF). Verify load fails. Write with wrong version (99). Verify load fails.
**Verifies**: Header validation prevents loading corrupt checkpoints.

### Test 9: `test_forward_single_layer_output_nonzero_finite` (ANE-gated)
Compile 1 LayerKernelSet with random weights. Run ForwardPass for 1 layer. Verify output is nonzero and all values are finite.
**Verifies**: Forward pass ANE pipeline produces valid output.

### Test 10: `test_forward_12_layers_no_nan` (ANE-gated)
Full 12-layer forward pass. Verify all activation buffers contain no NaN or Inf.
**Verifies**: No numerical instability across full depth.

### Test 11: `test_backward_produces_nonzero_gradients` (ANE-gated)
Run forward + backward for 1 step. Verify at least some gradient buffers are nonzero.
**Verifies**: Backward pass gradient flow.

### Test 12: `test_gradient_accumulation_scaling`
Create synthetic gradients. Accumulate over 2 steps (ACCUM_STEPS=2). Verify gradient values are scaled by 0.5 before Adam.
**Verifies**: The `1.0/steps_batch` scaling is correct.

### Test 13: `test_10_steps_loss_decreases` (ANE-gated + data-gated)
Load pretrained weights. Run 10 training steps. Verify loss[10] < loss[0].
**Verifies**: Training loop actually learns.

### Test 14: `test_exec_restart_checkpoint_roundtrip`
Save checkpoint at step N with known state. Load with --resume logic. Verify step, lr, loss, adam_t, cum_* counters are all restored exactly.
**Verifies**: exec-restart recovery preserves full training state.

</tests>

---

<verification>

## Verification Criteria

| Criterion | Target | How to Measure |
|---|---|---|
| Compilation | Zero errors, zero warnings | `swift build 2>&1 \| grep -c error` == 0 |
| Prior tests | No regressions | `swift test` still shows 111+ tests, 0 failures |
| New tests | All Phase 6 tests pass | `swift test --filter EspressoTests` |
| Checkpoint binary compat | Byte-identical roundtrip | Save, load, save again — files identical |
| Forward output | Finite, nonzero | Check all LayerActivations buffers |
| Backward gradients | Nonzero, correct residual flow | Verify dy = dx_rms1 + dx2 per layer |
| Gradient scaling | Correct 1/steps_batch | Test with known accumulated values |
| Loss decrease | loss[10] < loss[0] | 10-step training run |
| Performance | <= 9.3 ms/step on M4 | 100-step benchmark (ANE-gated) |

</verification>

---

<workflow>

## Implementation Order

Follow the todolist groups in `tasks/phase6-todolist.md`:

1. **Group 0**: Pre-flight — verify baseline compiles and tests pass
2. **Group 1**: Leaf types — GradientAccumulator, TokenDataset, Sampler, ExecRestart, Checkpoint (tests first for each)
3. **Group 2**: ForwardPass (tests first, then implement)
4. **Group 3**: BackwardPass + SendableBuffer helpers (tests first, then implement)
5. **Group 4**: main.swift + integration tests
6. **Group 5**: Final verification — all tests green, no regressions

At each gate: `swift build && swift test --filter EspressoTests`. Fix failures before proceeding.

</workflow>

---

<agentic_guidance>

## Pitfalls (Ranked by Severity)

### 1. IOSurface Channel Offset Errors (CRITICAL)
The backward pass has 15 IOSurface I/O operations per layer with specific source/destination channel offsets. Getting ANY offset wrong produces garbage gradients. Cross-reference EVERY io_copy/io_write_fp16_at/io_read_fp16 call against the IOSurface I/O table above. The most common mistake: fwdAttn output Q starts at offset DIM (not 0), because o_out occupies offset 0.

### 2. Checkpoint vs Pretrained Format Confusion (CRITICAL)
Checkpoint writes per-LAYER (all of layer L, then L+1). Pretrained writes per-TYPE (all Wq across layers, then all Wk). Using ModelWeightLoader's read order for checkpoint save/load will produce a corrupt checkpoint that silently loads wrong weights.

### 3. Missing Gradient Scaling (HIGH)
Forgetting to scale gradients by `1.0/steps_batch` before Adam, or scaling only per-layer but not rmsFinal/embed, produces divergent training. ALL 11 gradient groups must be scaled: 9 per-layer + rmsFinal + embed.

### 4. Adam Timestep Off-by-One (HIGH)
adam_t must be incremented BEFORE calling AdamOptimizer.update (which has `precondition(timestep >= 1)`). Starting at 0 and forgetting to increment crashes. Starting at 1 and incrementing before the first call means first update uses timestep=2, which gives wrong bias correction.

### 5. SendableBuffer Ownership (MEDIUM)
Use `consume` in capture lists: `[captX = consume captX]`. Without `consume`, the compiler may copy the `~Copyable` type (or refuse to compile). The `consume` transfers ownership into the closure — the SendableBuffer's deinit runs after the closure body executes.

### 6. RMSNorm.backward ACCUMULATES (MEDIUM)
`RMSNorm.backward` adds to `dw`, it does NOT zero `dw` first. Zero all gradient buffers ONCE at the start of each compilation batch (`grads[L].zero()`), then let backward accumulate across steps.

### 7. Barrier Before fwdAttn Write (MEDIUM)
Without the barrier at the start of each forward layer, a previous layer's async dW block might still be using the same IOSurface, causing data corruption. This manifests as non-deterministic NaN.

### 8. exec() Skips Deinit (LOW)
`execl()` replaces the process image. No Swift deinit runs, no ARC cleanup. This is fine — the OS reclaims all memory. But you MUST save the checkpoint and flush output BEFORE calling execl.

</agentic_guidance>
