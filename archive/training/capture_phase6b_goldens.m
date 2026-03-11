#import <Foundation/Foundation.h>
#import <math.h>

#import "stories_io.h"
#import "stories_mil.h"

static void fill_const(float *dst, int n, float v) {
    for (int i = 0; i < n; i++) dst[i] = v;
}

static BOOL write_f32le(NSString *path, const float *data, size_t count, NSError **errOut) {
    FILE *f = fopen(path.fileSystemRepresentation, "wb");
    if (!f) {
        if (errOut) *errOut = [NSError errorWithDomain:@"capture_phase6b_goldens" code:1 userInfo:@{NSLocalizedDescriptionKey:@"fopen failed"}];
        return NO;
    }
    size_t w = fwrite(data, sizeof(float), count, f);
    fclose(f);
    if (w != count) {
        if (errOut) *errOut = [NSError errorWithDomain:@"capture_phase6b_goldens" code:2 userInfo:@{NSLocalizedDescriptionKey:@"fwrite failed"}];
        return NO;
    }
    return YES;
}

int main(void) {
    @autoreleasepool {
        ane_init();
        printf("ANE classes: D=%p I=%p R=%p IO=%p\n", g_D, g_I, g_AR, g_AIO);

        NSString *repo = [[[NSFileManager defaultManager] currentDirectoryPath] stringByStandardizingPath];
        NSString *fixturesDir = [repo stringByAppendingPathComponent:@"Tests/ANERuntimeTests/Fixtures"];
        [[NSFileManager defaultManager] createDirectoryAtPath:fixturesDir withIntermediateDirectories:YES attributes:nil error:nil];

        // Shared deterministic weights (matches Swift tests).
        float *Wq = (float *)malloc(sizeof(float) * WQ_SZ);
        float *Wk = (float *)malloc(sizeof(float) * WQ_SZ);
        float *Wv = (float *)malloc(sizeof(float) * WQ_SZ);
        float *Wo = (float *)malloc(sizeof(float) * WO_SZ);
        float *W1 = (float *)malloc(sizeof(float) * W1_SZ);
        float *W2 = (float *)malloc(sizeof(float) * W2_SZ);
        float *W3 = (float *)malloc(sizeof(float) * W3_SZ);
        float *rms1 = (float *)malloc(sizeof(float) * DIM);
        float *rms2 = (float *)malloc(sizeof(float) * DIM);
        if (!Wq || !Wk || !Wv || !Wo || !W1 || !W2 || !W3 || !rms1 || !rms2) {
            fprintf(stderr, "alloc failed\n");
            return 1;
        }
        fill_const(Wq, WQ_SZ, 0.01f);
        fill_const(Wk, WQ_SZ, 0.01f);
        fill_const(Wv, WQ_SZ, 0.01f);
        fill_const(Wo, WO_SZ, 0.01f);
        fill_const(W1, W1_SZ, 0.01f);
        fill_const(W2, W2_SZ, 0.01f);
        fill_const(W3, W3_SZ, 0.01f);
        fill_const(rms1, DIM, 0.01f);
        fill_const(rms2, DIM, 0.01f);

        NSDictionary *wAttn = @{
            @"@model_path/weights/rms1.bin": @{@"offset": @0, @"data": build_blob(rms1, 1, DIM)},
            @"@model_path/weights/wq.bin": @{@"offset": @0, @"data": build_blob(Wq, DIM, DIM)},
            @"@model_path/weights/wk.bin": @{@"offset": @0, @"data": build_blob(Wk, DIM, DIM)},
            @"@model_path/weights/wv.bin": @{@"offset": @0, @"data": build_blob(Wv, DIM, DIM)},
            @"@model_path/weights/wo.bin": @{@"offset": @0, @"data": build_blob(Wo, DIM, DIM)},
            @"@model_path/weights/mask.bin": @{@"offset": @0, @"data": get_mask_blob()},
        };

        NSDictionary *wFfn = @{
            @"@model_path/weights/rms2.bin": @{@"offset": @0, @"data": build_blob(rms2, 1, DIM)},
            @"@model_path/weights/w1.bin": @{@"offset": @0, @"data": build_blob(W1, HIDDEN, DIM)},
            @"@model_path/weights/w3.bin": @{@"offset": @0, @"data": build_blob(W3, HIDDEN, DIM)},
            @"@model_path/weights/w2.bin": @{@"offset": @0, @"data": build_blob(W2, DIM, HIDDEN)},
        };

        NSDictionary *wFfnBwd = @{
            @"@model_path/weights/w2t.bin": @{@"offset": @0, @"data": build_blob_t(W2, DIM, HIDDEN)},
            @"@model_path/weights/w1t.bin": @{@"offset": @0, @"data": build_blob_t(W1, HIDDEN, DIM)},
            @"@model_path/weights/w3t.bin": @{@"offset": @0, @"data": build_blob_t(W3, HIDDEN, DIM)},
        };

        NSString *milAttn = gen_sdpa_fwd_taps();
        NSString *milFfn = gen_ffn_fwd_taps();
        NSString *milFfnBwd = gen_ffn_bwd();
        printf("MIL lengths: attn=%lu ffn=%lu ffnBwd=%lu\n",
               (unsigned long)milAttn.length, (unsigned long)milFfn.length, (unsigned long)milFfnBwd.length);

        int fwdAttnInBytes = DIM * SEQ * 2;
        int fwdAttnOutBytes = 6 * DIM * SEQ * 2;
        int fwdFfnInBytes = DIM * SEQ * 2;
        int fwdFfnOutBytes = (DIM + 3 * HIDDEN + DIM) * SEQ * 2;
        int ffnBwdInBytes = (DIM + 2 * HIDDEN) * SEQ * 2;
        int ffnBwdOutBytes = (DIM + 2 * HIDDEN) * SEQ * 2;

        Kern *kAttn = compile_kern_mil_w(milAttn, wAttn, fwdAttnInBytes, fwdAttnOutBytes);
        Kern *kFfn = compile_kern_mil_w(milFfn, wFfn, fwdFfnInBytes, fwdFfnOutBytes);
        Kern *kFfnBwd = compile_kern_mil_w(milFfnBwd, wFfnBwd, ffnBwdInBytes, ffnBwdOutBytes);
        if (!kAttn || !kFfn || !kFfnBwd) {
            fprintf(stderr, "kernel compile failed (attn=%p ffn=%p ffnBwd=%p)\n", kAttn, kFfn, kFfnBwd);
            if (kAttn) free_kern(kAttn);
            if (kFfn) free_kern(kFfn);
            if (kFfnBwd) free_kern(kFfnBwd);
            free(Wq); free(Wk); free(Wv); free(Wo); free(W1); free(W2); free(W3); free(rms1); free(rms2);
            return 1;
        }

        float *x = (float *)calloc((size_t)DIM * SEQ, sizeof(float));
        float *oOut = (float *)calloc((size_t)DIM * SEQ, sizeof(float));
        float *ffnY = (float *)calloc((size_t)DIM * SEQ, sizeof(float));
        float *ffnBwdIn = (float *)calloc((size_t)(DIM + 2 * HIDDEN) * SEQ, sizeof(float));
        float *ffnDX = (float *)calloc((size_t)DIM * SEQ, sizeof(float));
        if (!x || !oOut || !ffnY || !ffnBwdIn || !ffnDX) {
            fprintf(stderr, "buffer alloc failed\n");
            free_kern(kAttn); free_kern(kFfn); free_kern(kFfnBwd);
            free(Wq); free(Wk); free(Wv); free(Wo); free(W1); free(W2); free(W3); free(rms1); free(rms2);
            free(x); free(oOut); free(ffnY); free(ffnBwdIn); free(ffnDX);
            return 1;
        }

        for (int i = 0; i < DIM * SEQ; i++) x[i] = (float)(i % 64 + 1) * 0.01f;
        io_write_fp16(kAttn->ioIn, x, DIM, SEQ);
        ane_eval(kAttn);
        io_read_fp16(kAttn->ioOut, oOut, 0, DIM, SEQ);

        io_write_fp16(kFfn->ioIn, oOut, DIM, SEQ);
        ane_eval(kFfn);
        io_read_fp16(kFfn->ioOut, ffnY, 0, DIM, SEQ);

        for (int i = 0; i < (DIM + 2 * HIDDEN) * SEQ; i++) {
            ffnBwdIn[i] = (float)(i % 128 + 1) * 0.005f;
        }
        io_write_fp16(kFfnBwd->ioIn, ffnBwdIn, DIM + 2 * HIDDEN, SEQ);
        ane_eval(kFfnBwd);
        io_read_fp16(kFfnBwd->ioOut, ffnDX, 0, DIM, SEQ);

        NSError *err = nil;
        if (!write_f32le([fixturesDir stringByAppendingPathComponent:@"fwd_attn_oOut_seq256_f32le.bin"], oOut, (size_t)DIM * SEQ, &err) ||
            !write_f32le([fixturesDir stringByAppendingPathComponent:@"fwd_ffn_y_seq256_f32le.bin"], ffnY, (size_t)DIM * SEQ, &err) ||
            !write_f32le([fixturesDir stringByAppendingPathComponent:@"ffn_bwd_dx_seq256_f32le.bin"], ffnDX, (size_t)DIM * SEQ, &err)) {
            fprintf(stderr, "write fixture failed: %s\n", err ? err.localizedDescription.UTF8String : "unknown");
            free_kern(kAttn); free_kern(kFfn); free_kern(kFfnBwd);
            free(Wq); free(Wk); free(Wv); free(Wo); free(W1); free(W2); free(W3); free(rms1); free(rms2);
            free(x); free(oOut); free(ffnY); free(ffnBwdIn); free(ffnDX);
            return 1;
        }

        printf("wrote fixtures in %s\n", fixturesDir.fileSystemRepresentation);

        free_kern(kAttn);
        free_kern(kFfn);
        free_kern(kFfnBwd);
        free(Wq); free(Wk); free(Wv); free(Wo); free(W1); free(W2); free(W3); free(rms1); free(rms2);
        free(x); free(oOut); free(ffnY); free(ffnBwdIn); free(ffnDX);
        return 0;
    }
}
