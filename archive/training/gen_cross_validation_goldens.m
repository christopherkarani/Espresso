#import <Foundation/Foundation.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <IOSurface/IOSurface.h>
#include <math.h>
#include <sys/stat.h>

#define DIM 768
#define HEADS 12
#define HD (DIM / HEADS)
#define HIDDEN 2048
#define SEQ 64

typedef struct {
    id model;
    NSString *tmpDir;
} Kern;

static Class g_D, g_I, g_AR, g_AIO;

static void ane_init(void) {
    dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);
    g_D = NSClassFromString(@"_ANEInMemoryModelDescriptor");
    g_I = NSClassFromString(@"_ANEInMemoryModel");
    g_AR = NSClassFromString(@"_ANERequest");
    g_AIO = NSClassFromString(@"_ANEIOSurfaceObject");
}

static IOSurfaceRef make_surface(size_t bytes) {
    return IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth: @(bytes),
        (id)kIOSurfaceHeight: @1,
        (id)kIOSurfaceBytesPerElement: @1,
        (id)kIOSurfaceBytesPerRow: @(bytes),
        (id)kIOSurfaceAllocSize: @(bytes),
        (id)kIOSurfacePixelFormat: @0,
    });
}

static NSData *build_blob(const float *w, int oc, int ic) {
    int wsize = oc * ic * 2;
    int total = 128 + wsize;
    uint8_t *buf = (uint8_t *)calloc((size_t)total, 1);

    buf[0] = 1;
    buf[4] = 2;
    buf[64] = 0xEF;
    buf[65] = 0xBE;
    buf[66] = 0xAD;
    buf[67] = 0xDE;
    buf[68] = 1;
    *(uint32_t *)(buf + 72) = (uint32_t)wsize;
    *(uint32_t *)(buf + 80) = 128;

    _Float16 *fp16 = (_Float16 *)(buf + 128);
    for (int i = 0; i < oc * ic; i++) {
        fp16[i] = (_Float16)w[i];
    }

    return [NSData dataWithBytesNoCopy:buf length:(NSUInteger)total freeWhenDone:YES];
}

static NSData *build_blob_t(const float *w, int rows, int cols) {
    int wsize = rows * cols * 2;
    int total = 128 + wsize;
    uint8_t *buf = (uint8_t *)calloc((size_t)total, 1);

    buf[0] = 1;
    buf[4] = 2;
    buf[64] = 0xEF;
    buf[65] = 0xBE;
    buf[66] = 0xAD;
    buf[67] = 0xDE;
    buf[68] = 1;
    *(uint32_t *)(buf + 72) = (uint32_t)wsize;
    *(uint32_t *)(buf + 80) = 128;

    _Float16 *fp16 = (_Float16 *)(buf + 128);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            fp16[j * rows + i] = (_Float16)w[i * cols + j];
        }
    }

    return [NSData dataWithBytesNoCopy:buf length:(NSUInteger)total freeWhenDone:YES];
}

static NSData *build_blob_fp16(const _Float16 *data, int count) {
    int wsize = count * 2;
    int total = 128 + wsize;
    uint8_t *buf = (uint8_t *)calloc((size_t)total, 1);

    buf[0] = 1;
    buf[4] = 2;
    buf[64] = 0xEF;
    buf[65] = 0xBE;
    buf[66] = 0xAD;
    buf[67] = 0xDE;
    buf[68] = 1;
    *(uint32_t *)(buf + 72) = (uint32_t)wsize;
    *(uint32_t *)(buf + 80) = 128;
    memcpy(buf + 128, data, (size_t)wsize);

    return [NSData dataWithBytesNoCopy:buf length:(NSUInteger)total freeWhenDone:YES];
}

static BOOL write_data(NSString *path, NSData *data) {
    NSError *err = nil;
    BOOL ok = [data writeToFile:path options:NSDataWritingAtomic error:&err];
    if (!ok) {
        fprintf(stderr, "write failed (%s): %s\n", path.fileSystemRepresentation, err.localizedDescription.UTF8String);
    }
    return ok;
}

static BOOL write_string(NSString *path, NSString *text) {
    NSError *err = nil;
    BOOL ok = [text writeToFile:path atomically:YES encoding:NSUTF8StringEncoding error:&err];
    if (!ok) {
        fprintf(stderr, "write text failed (%s): %s\n", path.fileSystemRepresentation, err.localizedDescription.UTF8String);
    }
    return ok;
}

static BOOL write_f32le(NSString *path, const float *data, size_t count) {
    FILE *f = fopen(path.fileSystemRepresentation, "wb");
    if (!f) {
        fprintf(stderr, "fopen failed (%s)\n", path.fileSystemRepresentation);
        return NO;
    }
    size_t n = fwrite(data, sizeof(float), count, f);
    fclose(f);
    if (n != count) {
        fprintf(stderr, "fwrite failed (%s): wrote %zu of %zu\n", path.fileSystemRepresentation, n, count);
        return NO;
    }
    return YES;
}

static Kern compile_mil(NSString *mil, NSDictionary *weightDict) {
    Kern k = {nil, nil};

    NSData *milData = [mil dataUsingEncoding:NSUTF8StringEncoding];
    id desc = ((id (*)(Class, SEL, id, id, id))objc_msgSend)(
        g_D,
        @selector(modelWithMILText:weights:optionsPlist:),
        milData,
        weightDict ?: @{},
        nil
    );
    if (!desc) {
        printf("  desc=NULL\n");
        return k;
    }

    id model = ((id (*)(Class, SEL, id))objc_msgSend)(g_I, @selector(inMemoryModelWithDescriptor:), desc);
    id hex = ((id (*)(id, SEL))objc_msgSend)(model, @selector(hexStringIdentifier));
    NSString *tmpDir = [NSTemporaryDirectory() stringByAppendingPathComponent:hex];
    [[NSFileManager defaultManager] createDirectoryAtPath:[tmpDir stringByAppendingPathComponent:@"weights"]
                              withIntermediateDirectories:YES
                                               attributes:nil
                                                    error:nil];
    [milData writeToFile:[tmpDir stringByAppendingPathComponent:@"model.mil"] atomically:YES];
    for (NSString *path in weightDict) {
        NSString *localPath = [tmpDir stringByAppendingPathComponent:[path stringByReplacingOccurrencesOfString:@"@model_path/" withString:@""]];
        NSData *blob = weightDict[path][@"data"];
        [blob writeToFile:localPath atomically:YES];
    }

    NSError *err = nil;
    BOOL ok = ((BOOL (*)(id, SEL, unsigned int, id, NSError **))objc_msgSend)(
        model,
        @selector(compileWithQoS:options:error:),
        21,
        @{},
        &err
    );
    if (!ok) {
        printf("  compile FAIL: %s\n", err ? err.localizedDescription.UTF8String : "");
        [[NSFileManager defaultManager] removeItemAtPath:tmpDir error:nil];
        return k;
    }

    ok = ((BOOL (*)(id, SEL, unsigned int, id, NSError **))objc_msgSend)(
        model,
        @selector(loadWithQoS:options:error:),
        21,
        @{},
        &err
    );
    if (!ok) {
        printf("  load FAIL: %s\n", err ? err.localizedDescription.UTF8String : "");
        [[NSFileManager defaultManager] removeItemAtPath:tmpDir error:nil];
        return k;
    }

    k.model = model;
    k.tmpDir = tmpDir;
    return k;
}

static BOOL ane_eval_io(Kern *k, IOSurfaceRef *ins, int nin, IOSurfaceRef *outs, int nout) {
    NSMutableArray *inArr = [NSMutableArray array];
    NSMutableArray *inIdx = [NSMutableArray array];
    NSMutableArray *outArr = [NSMutableArray array];
    NSMutableArray *outIdx = [NSMutableArray array];

    for (int i = 0; i < nin; i++) {
        id wrap = ((id (*)(Class, SEL, IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), ins[i]);
        [inArr addObject:wrap];
        [inIdx addObject:@(i)];
    }
    for (int i = 0; i < nout; i++) {
        id wrap = ((id (*)(Class, SEL, IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), outs[i]);
        [outArr addObject:wrap];
        [outIdx addObject:@(i)];
    }

    id req = ((id (*)(Class, SEL, id, id, id, id, id, id, id))objc_msgSend)(
        g_AR,
        @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
        inArr,
        inIdx,
        outArr,
        outIdx,
        nil,
        nil,
        @0
    );

    NSError *err = nil;
    BOOL ok = ((BOOL (*)(id, SEL, unsigned int, id, id, NSError **))objc_msgSend)(
        k->model,
        @selector(evaluateWithQoS:options:request:error:),
        21,
        @{},
        req,
        &err
    );
    if (!ok) {
        printf("  eval FAIL: %s\n", err ? err.localizedDescription.UTF8String : "");
    }
    return ok;
}

static void cleanup_kern(Kern *k) {
    if (!k->model) {
        return;
    }
    NSError *err = nil;
    ((BOOL (*)(id, SEL, unsigned int, NSError **))objc_msgSend)(k->model, @selector(unloadWithQoS:error:), 21, &err);
    [[NSFileManager defaultManager] removeItemAtPath:k->tmpDir error:nil];
}

static NSString *full_fused_mil(void) {
    float scaleVal = 1.0f / sqrtf((float)HD);

    return [NSString stringWithFormat:
        @"program(1.3)\n"
        "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, {\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, {\"coremltools-version\", \"9.0\"}})]\n"
        "{\n"
        "    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n"
        "        string pt = const()[name = string(\"pt\"), val = string(\"valid\")];\n"
        "        tensor<int32, [2]> st = const()[name = string(\"st\"), val = tensor<int32, [2]>([1, 1])];\n"
        "        tensor<int32, [4]> pd = const()[name = string(\"pd\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n"
        "        tensor<int32, [2]> dl = const()[name = string(\"dl\"), val = tensor<int32, [2]>([1, 1])];\n"
        "        int32 gr1 = const()[name = string(\"g1\"), val = int32(1)];\n"
        "        tensor<fp16, [%d, %d, 1, 1]> Wq = const()[name = string(\"Wq\"), val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/wq.bin\"), offset = uint64(64)))];\n"
        "        tensor<fp16, [%d, %d, 1, 1]> Wk = const()[name = string(\"Wk\"), val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/wk.bin\"), offset = uint64(64)))];\n"
        "        tensor<fp16, [%d, %d, 1, 1]> Wv = const()[name = string(\"Wv\"), val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/wv.bin\"), offset = uint64(64)))];\n"
        "        tensor<fp16, [%d, %d, 1, 1]> Wout = const()[name = string(\"Wo\"), val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/wo.bin\"), offset = uint64(64)))];\n"
        "        tensor<fp16, [1, %d, 1, %d]> q_flat = conv(dilations = dl, groups = gr1, pad = pd, pad_type = pt, strides = st, weight = Wq, x = x)[name = string(\"cq\")];\n"
        "        tensor<fp16, [1, %d, 1, %d]> k_flat = conv(dilations = dl, groups = gr1, pad = pd, pad_type = pt, strides = st, weight = Wk, x = x)[name = string(\"ck\")];\n"
        "        tensor<fp16, [1, %d, 1, %d]> v_flat = conv(dilations = dl, groups = gr1, pad = pd, pad_type = pt, strides = st, weight = Wv, x = x)[name = string(\"cv\")];\n"
        "        tensor<int32, [4]> qsh = const()[name = string(\"qsh\"), val = tensor<int32, [4]>([1, %d, %d, %d])];\n"
        "        tensor<fp16, [1, %d, %d, %d]> q_4d = reshape(shape = qsh, x = q_flat)[name = string(\"rq\")];\n"
        "        tensor<int32, [4]> perm = const()[name = string(\"pm\"), val = tensor<int32, [4]>([0, 1, 3, 2])];\n"
        "        tensor<fp16, [1, %d, %d, %d]> q = transpose(perm = perm, x = q_4d)[name = string(\"tq\")];\n"
        "        tensor<fp16, [1, %d, %d, %d]> k_4d = reshape(shape = qsh, x = k_flat)[name = string(\"rk\")];\n"
        "        tensor<fp16, [1, %d, %d, %d]> k = transpose(perm = perm, x = k_4d)[name = string(\"tk\")];\n"
        "        tensor<fp16, [1, %d, %d, %d]> v_4d = reshape(shape = qsh, x = v_flat)[name = string(\"rv\")];\n"
        "        tensor<fp16, [1, %d, %d, %d]> v = transpose(perm = perm, x = v_4d)[name = string(\"tv\")];\n"
        "        bool ty = const()[name = string(\"ty\"), val = bool(true)];\n"
        "        bool tx = const()[name = string(\"tx\"), val = bool(false)];\n"
        "        tensor<fp16, [1, %d, %d, %d]> scores = matmul(transpose_x = tx, transpose_y = ty, x = q, y = k)[name = string(\"mm1\")];\n"
        "        fp16 sc = const()[name = string(\"sc\"), val = fp16(%f)];\n"
        "        tensor<fp16, [1, %d, %d, %d]> scaled = mul(x = scores, y = sc)[name = string(\"scl\")];\n"
        "        tensor<fp16, [1, 1, %d, %d]> cmask = const()[name = string(\"cm\"), val = tensor<fp16, [1, 1, %d, %d]>(BLOBFILE(path = string(\"@model_path/weights/mask.bin\"), offset = uint64(64)))];\n"
        "        tensor<fp16, [1, %d, %d, %d]> masked = add(x = scaled, y = cmask)[name = string(\"msk\")];\n"
        "        int32 sax = const()[name = string(\"sax\"), val = int32(-1)];\n"
        "        tensor<fp16, [1, %d, %d, %d]> attn_w = softmax(axis = sax, x = masked)[name = string(\"sm\")];\n"
        "        tensor<fp16, [1, %d, %d, %d]> attn_4d = matmul(transpose_x = tx, transpose_y = tx, x = attn_w, y = v)[name = string(\"mm2\")];\n"
        "        tensor<fp16, [1, %d, %d, %d]> attn_t = transpose(perm = perm, x = attn_4d)[name = string(\"ta\")];\n"
        "        tensor<int32, [4]> osh = const()[name = string(\"osh\"), val = tensor<int32, [4]>([1, %d, 1, %d])];\n"
        "        tensor<fp16, [1, %d, 1, %d]> attn_flat = reshape(shape = osh, x = attn_t)[name = string(\"ra\")];\n"
        "        tensor<fp16, [1, %d, 1, %d]> out = conv(dilations = dl, groups = gr1, pad = pd, pad_type = pt, strides = st, weight = Wout, x = attn_flat)[name = string(\"co\")];\n"
        "    } -> (out);\n"
        "}\n",
        DIM, SEQ,
        DIM, DIM, DIM, DIM,
        DIM, DIM, DIM, DIM,
        DIM, DIM, DIM, DIM,
        DIM, DIM, DIM, DIM,
        DIM, SEQ,
        DIM, SEQ,
        DIM, SEQ,
        HEADS, HD, SEQ,
        HEADS, HD, SEQ,
        HEADS, SEQ, HD,
        HEADS, HD, SEQ,
        HEADS, SEQ, HD,
        HEADS, HD, SEQ,
        HEADS, SEQ, HD,
        HEADS, SEQ, SEQ,
        scaleVal,
        HEADS, SEQ, SEQ,
        SEQ, SEQ, SEQ, SEQ,
        HEADS, SEQ, SEQ,
        HEADS, SEQ, SEQ,
        HEADS, SEQ, HD,
        HEADS, HD, SEQ,
        DIM, SEQ,
        DIM, SEQ,
        DIM, SEQ
    ];
}

static NSString *fused_bwd_mil(void) {
    return [NSString stringWithFormat:
        @"program(1.3)\n"
        "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, {\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, {\"coremltools-version\", \"9.0\"}})]\n"
        "{\n"
        "    func main<ios18>(tensor<fp32, [1, %d, 1, %d]> x) {\n"
        "        string d1 = const()[name = string(\"d1\"), val = string(\"fp16\")];\n"
        "        tensor<fp16, [1, %d, 1, %d]> x16 = cast(dtype = d1, x = x)[name = string(\"cx\")];\n"
        "        tensor<int32, [4]> b1 = const()[name = string(\"b1\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n"
        "        tensor<int32, [4]> s1 = const()[name = string(\"s1\"), val = tensor<int32, [4]>([1, %d, 1, %d])];\n"
        "        tensor<fp16, [1, %d, 1, %d]> dh1 = slice_by_size(x = x16, begin = b1, size = s1)[name = string(\"sl1\")];\n"
        "        tensor<int32, [4]> b3 = const()[name = string(\"b3\"), val = tensor<int32, [4]>([0, %d, 0, 0])];\n"
        "        tensor<int32, [4]> s3 = const()[name = string(\"s3\"), val = tensor<int32, [4]>([1, %d, 1, %d])];\n"
        "        tensor<fp16, [1, %d, 1, %d]> dh3 = slice_by_size(x = x16, begin = b3, size = s3)[name = string(\"sl3\")];\n"
        "        string pt = const()[name = string(\"pt\"), val = string(\"valid\")];\n"
        "        tensor<int32, [2]> st = const()[name = string(\"st\"), val = tensor<int32, [2]>([1, 1])];\n"
        "        tensor<int32, [4]> pd = const()[name = string(\"pd\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n"
        "        tensor<int32, [2]> dl = const()[name = string(\"dl\"), val = tensor<int32, [2]>([1, 1])];\n"
        "        int32 gr = const()[name = string(\"gr\"), val = int32(1)];\n"
        "        tensor<fp16, [%d, %d, 1, 1]> W1t = const()[name = string(\"W1t\"), val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/w1t.bin\"), offset = uint64(64)))];\n"
        "        tensor<fp16, [%d, %d, 1, 1]> W3t = const()[name = string(\"W3t\"), val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/w3t.bin\"), offset = uint64(64)))];\n"
        "        tensor<fp16, [1, %d, 1, %d]> dx1 = conv(dilations = dl, groups = gr, pad = pd, pad_type = pt, strides = st, weight = W1t, x = dh1)[name = string(\"cv1\")];\n"
        "        tensor<fp16, [1, %d, 1, %d]> dx3 = conv(dilations = dl, groups = gr, pad = pd, pad_type = pt, strides = st, weight = W3t, x = dh3)[name = string(\"cv3\")];\n"
        "        tensor<fp16, [1, %d, 1, %d]> sum = add(x = dx1, y = dx3)[name = string(\"ad\")];\n"
        "        string d2 = const()[name = string(\"d2\"), val = string(\"fp32\")];\n"
        "        tensor<fp32, [1, %d, 1, %d]> y = cast(dtype = d2, x = sum)[name = string(\"co\")];\n"
        "    } -> (y);\n"
        "}\n",
        HIDDEN * 2, SEQ,
        HIDDEN * 2, SEQ,
        HIDDEN, SEQ,
        HIDDEN, SEQ,
        HIDDEN,
        HIDDEN, SEQ,
        HIDDEN, SEQ,
        DIM, HIDDEN, DIM, HIDDEN,
        DIM, HIDDEN, DIM, HIDDEN,
        DIM, SEQ,
        DIM, SEQ,
        DIM, SEQ,
        DIM, SEQ
    ];
}

static void print_file_info(NSString *path) {
    struct stat st;
    if (stat(path.fileSystemRepresentation, &st) == 0) {
        printf("  wrote %s (%lld bytes)\n", path.fileSystemRepresentation, (long long)st.st_size);
    } else {
        printf("  wrote %s\n", path.fileSystemRepresentation);
    }
}

int main(int argc, const char **argv) {
    @autoreleasepool {
        setbuf(stdout, NULL);
        ane_init();

        NSString *outDir = nil;
        if (argc >= 2) {
            outDir = [[NSString stringWithUTF8String:argv[1]] stringByStandardizingPath];
        } else {
            NSString *cwd = [[NSFileManager defaultManager] currentDirectoryPath];
            outDir = [[cwd stringByAppendingPathComponent:@"training/golden_outputs"] stringByStandardizingPath];
        }

        NSError *mkdirErr = nil;
        if (![[NSFileManager defaultManager] createDirectoryAtPath:outDir withIntermediateDirectories:YES attributes:nil error:&mkdirErr]) {
            fprintf(stderr, "mkdir failed: %s\n", mkdirErr.localizedDescription.UTF8String);
            return 1;
        }

        printf("Output directory: %s\n", outDir.fileSystemRepresentation);

        // CV-1: weight blob oracle (4x4 matrix [1..16]).
        float weight4x4[16];
        for (int i = 0; i < 16; i++) {
            weight4x4[i] = (float)(i + 1);
        }
        NSString *weightBlobPath = [outDir stringByAppendingPathComponent:@"weight_blob_4x4.bin"];
        if (!write_data(weightBlobPath, build_blob(weight4x4, 4, 4))) {
            return 1;
        }
        print_file_info(weightBlobPath);

        // CV-2: causal mask oracle (seq=8).
        _Float16 mask8[64];
        for (int t = 0; t < 8; t++) {
            for (int t2 = 0; t2 < 8; t2++) {
                mask8[t * 8 + t2] = (t2 <= t) ? (_Float16)0.0f : (_Float16)(-65504.0f);
            }
        }
        NSString *maskBlobPath = [outDir stringByAppendingPathComponent:@"causal_mask_seq8.bin"];
        if (!write_data(maskBlobPath, build_blob_fp16(mask8, 64))) {
            return 1;
        }
        print_file_info(maskBlobPath);

        // CV-3: full fused forward oracle.
        srand48(42);
        float scD = 1.0f / sqrtf((float)DIM);
        float scH = 1.0f / sqrtf((float)HIDDEN);

        float *Wq = (float *)malloc((size_t)DIM * DIM * sizeof(float));
        float *Wk = (float *)malloc((size_t)DIM * DIM * sizeof(float));
        float *Wv = (float *)malloc((size_t)DIM * DIM * sizeof(float));
        float *Wo = (float *)malloc((size_t)DIM * DIM * sizeof(float));
        float *W1 = (float *)malloc((size_t)HIDDEN * DIM * sizeof(float));
        float *W2 = (float *)malloc((size_t)DIM * HIDDEN * sizeof(float));
        float *W3 = (float *)malloc((size_t)HIDDEN * DIM * sizeof(float));
        if (!Wq || !Wk || !Wv || !Wo || !W1 || !W2 || !W3) {
            fprintf(stderr, "alloc failed for full_fused weights\n");
            free(Wq); free(Wk); free(Wv); free(Wo); free(W1); free(W2); free(W3);
            return 1;
        }

        for (int i = 0; i < DIM * DIM; i++) Wq[i] = scD * (float)(2.0 * drand48() - 1.0);
        for (int i = 0; i < DIM * DIM; i++) Wk[i] = scD * (float)(2.0 * drand48() - 1.0);
        for (int i = 0; i < DIM * DIM; i++) Wv[i] = scD * (float)(2.0 * drand48() - 1.0);
        for (int i = 0; i < DIM * DIM; i++) Wo[i] = scD * (float)(2.0 * drand48() - 1.0);
        for (int i = 0; i < HIDDEN * DIM; i++) W1[i] = scH * (float)(2.0 * drand48() - 1.0);
        for (int i = 0; i < DIM * HIDDEN; i++) W2[i] = scD * (float)(2.0 * drand48() - 1.0);
        for (int i = 0; i < HIDDEN * DIM; i++) W3[i] = scH * (float)(2.0 * drand48() - 1.0);

        _Float16 *mask64 = (_Float16 *)calloc((size_t)SEQ * SEQ, sizeof(_Float16));
        if (!mask64) {
            fprintf(stderr, "alloc failed for full_fused mask\n");
            free(Wq); free(Wk); free(Wv); free(Wo); free(W1); free(W2); free(W3);
            return 1;
        }
        for (int t = 0; t < SEQ; t++) {
            for (int t2 = 0; t2 < SEQ; t2++) {
                mask64[t * SEQ + t2] = (t2 <= t) ? (_Float16)0.0f : (_Float16)(-65504.0f);
            }
        }

        NSString *fullMil = full_fused_mil();
        NSString *fullMilPath = [outDir stringByAppendingPathComponent:@"full_fused.mil"];
        if (!write_string(fullMilPath, fullMil)) {
            free(mask64); free(Wq); free(Wk); free(Wv); free(Wo); free(W1); free(W2); free(W3);
            return 1;
        }
        print_file_info(fullMilPath);

        NSData *wqBlob = build_blob(Wq, DIM, DIM);
        NSData *wkBlob = build_blob(Wk, DIM, DIM);
        NSData *wvBlob = build_blob(Wv, DIM, DIM);
        NSData *woBlob = build_blob(Wo, DIM, DIM);
        NSData *maskBlob = build_blob_fp16(mask64, SEQ * SEQ);

        if (!write_data([outDir stringByAppendingPathComponent:@"full_fused_wq.bin"], wqBlob) ||
            !write_data([outDir stringByAppendingPathComponent:@"full_fused_wk.bin"], wkBlob) ||
            !write_data([outDir stringByAppendingPathComponent:@"full_fused_wv.bin"], wvBlob) ||
            !write_data([outDir stringByAppendingPathComponent:@"full_fused_wo.bin"], woBlob) ||
            !write_data([outDir stringByAppendingPathComponent:@"full_fused_mask.bin"], maskBlob)) {
            free(mask64); free(Wq); free(Wk); free(Wv); free(Wo); free(W1); free(W2); free(W3);
            return 1;
        }

        NSDictionary *fullWeights = @{
            @"@model_path/weights/wq.bin": @{@"offset": @0, @"data": wqBlob},
            @"@model_path/weights/wk.bin": @{@"offset": @0, @"data": wkBlob},
            @"@model_path/weights/wv.bin": @{@"offset": @0, @"data": wvBlob},
            @"@model_path/weights/wo.bin": @{@"offset": @0, @"data": woBlob},
            @"@model_path/weights/mask.bin": @{@"offset": @0, @"data": maskBlob},
        };

        Kern fullKernel = compile_mil(fullMil, fullWeights);
        if (!fullKernel.model) {
            fprintf(stderr, "full_fused compile/load failed\n");
            free(mask64); free(Wq); free(Wk); free(Wv); free(Wo); free(W1); free(W2); free(W3);
            return 1;
        }

        float *xTokenMajor = (float *)malloc((size_t)SEQ * DIM * sizeof(float));
        float *xChannelMajor = (float *)malloc((size_t)SEQ * DIM * sizeof(float));
        float *fullOut = (float *)malloc((size_t)SEQ * DIM * sizeof(float));
        if (!xTokenMajor || !xChannelMajor || !fullOut) {
            fprintf(stderr, "alloc failed for full_fused IO\n");
            cleanup_kern(&fullKernel);
            free(mask64); free(Wq); free(Wk); free(Wv); free(Wo); free(W1); free(W2); free(W3);
            free(xTokenMajor); free(xChannelMajor); free(fullOut);
            return 1;
        }

        for (int i = 0; i < SEQ * DIM; i++) {
            xTokenMajor[i] = 0.1f * (float)(2.0 * drand48() - 1.0);
        }
        for (int t = 0; t < SEQ; t++) {
            for (int c = 0; c < DIM; c++) {
                xChannelMajor[c * SEQ + t] = xTokenMajor[t * DIM + c];
            }
        }

        NSString *fullInputPath = [outDir stringByAppendingPathComponent:@"full_fused_input_seq64_f32le.bin"];
        if (!write_f32le(fullInputPath, xChannelMajor, (size_t)DIM * SEQ)) {
            cleanup_kern(&fullKernel);
            free(mask64); free(Wq); free(Wk); free(Wv); free(Wo); free(W1); free(W2); free(W3);
            free(xTokenMajor); free(xChannelMajor); free(fullOut);
            return 1;
        }
        print_file_info(fullInputPath);

        IOSurfaceRef fullIn = make_surface((size_t)DIM * SEQ * 2);
        IOSurfaceRef fullOutSurface = make_surface((size_t)DIM * SEQ * 2);
        IOSurfaceLock(fullIn, 0, NULL);
        _Float16 *fullInPtr = (_Float16 *)IOSurfaceGetBaseAddress(fullIn);
        for (int i = 0; i < DIM * SEQ; i++) {
            fullInPtr[i] = (_Float16)xChannelMajor[i];
        }
        IOSurfaceUnlock(fullIn, 0, NULL);

        IOSurfaceRef fullIns[] = {fullIn};
        IOSurfaceRef fullOuts[] = {fullOutSurface};
        if (!ane_eval_io(&fullKernel, fullIns, 1, fullOuts, 1)) {
            cleanup_kern(&fullKernel);
            CFRelease(fullIn); CFRelease(fullOutSurface);
            free(mask64); free(Wq); free(Wk); free(Wv); free(Wo); free(W1); free(W2); free(W3);
            free(xTokenMajor); free(xChannelMajor); free(fullOut);
            return 1;
        }

        IOSurfaceLock(fullOutSurface, kIOSurfaceLockReadOnly, NULL);
        _Float16 *fullOutPtr = (_Float16 *)IOSurfaceGetBaseAddress(fullOutSurface);
        for (int i = 0; i < DIM * SEQ; i++) {
            fullOut[i] = (float)fullOutPtr[i];
        }
        IOSurfaceUnlock(fullOutSurface, kIOSurfaceLockReadOnly, NULL);

        NSString *fullOutPath = [outDir stringByAppendingPathComponent:@"full_fused_out_seq64_f32le.bin"];
        if (!write_f32le(fullOutPath, fullOut, (size_t)DIM * SEQ)) {
            cleanup_kern(&fullKernel);
            CFRelease(fullIn); CFRelease(fullOutSurface);
            free(mask64); free(Wq); free(Wk); free(Wv); free(Wo); free(W1); free(W2); free(W3);
            free(xTokenMajor); free(xChannelMajor); free(fullOut);
            return 1;
        }
        print_file_info(fullOutPath);

        cleanup_kern(&fullKernel);
        CFRelease(fullIn);
        CFRelease(fullOutSurface);
        free(mask64);
        free(Wq); free(Wk); free(Wv); free(Wo); free(W1); free(W2); free(W3);
        free(xTokenMajor); free(xChannelMajor); free(fullOut);

        // CV-4: fused backward oracle.
        srand48(42);
        float sc = 1.0f / sqrtf((float)HIDDEN);

        float *BW1 = (float *)malloc((size_t)HIDDEN * DIM * sizeof(float));
        float *BW3 = (float *)malloc((size_t)HIDDEN * DIM * sizeof(float));
        if (!BW1 || !BW3) {
            fprintf(stderr, "alloc failed for fused_bwd weights\n");
            free(BW1); free(BW3);
            return 1;
        }
        for (int i = 0; i < HIDDEN * DIM; i++) {
            BW1[i] = sc * (float)(2.0 * drand48() - 1.0);
            BW3[i] = sc * (float)(2.0 * drand48() - 1.0);
        }

        NSString *bwdMil = fused_bwd_mil();
        NSString *bwdMilPath = [outDir stringByAppendingPathComponent:@"fused_bwd.mil"];
        if (!write_string(bwdMilPath, bwdMil)) {
            free(BW1); free(BW3);
            return 1;
        }
        print_file_info(bwdMilPath);

        NSData *w1tBlob = build_blob_t(BW1, HIDDEN, DIM);
        NSData *w3tBlob = build_blob_t(BW3, HIDDEN, DIM);
        if (!write_data([outDir stringByAppendingPathComponent:@"fused_bwd_w1t.bin"], w1tBlob) ||
            !write_data([outDir stringByAppendingPathComponent:@"fused_bwd_w3t.bin"], w3tBlob)) {
            free(BW1); free(BW3);
            return 1;
        }

        NSDictionary *bwdWeights = @{
            @"@model_path/weights/w1t.bin": @{@"offset": @0, @"data": w1tBlob},
            @"@model_path/weights/w3t.bin": @{@"offset": @0, @"data": w3tBlob},
        };

        Kern bwdKernel = compile_mil(bwdMil, bwdWeights);
        if (!bwdKernel.model) {
            fprintf(stderr, "fused_bwd compile/load failed\n");
            free(BW1); free(BW3);
            return 1;
        }

        float *dh1 = (float *)malloc((size_t)SEQ * HIDDEN * sizeof(float));
        float *dh3 = (float *)malloc((size_t)SEQ * HIDDEN * sizeof(float));
        float *bwdInput = (float *)calloc((size_t)SEQ * HIDDEN * 2, sizeof(float));
        float *bwdOutput = (float *)malloc((size_t)SEQ * DIM * sizeof(float));
        if (!dh1 || !dh3 || !bwdInput || !bwdOutput) {
            fprintf(stderr, "alloc failed for fused_bwd IO\n");
            cleanup_kern(&bwdKernel);
            free(BW1); free(BW3);
            free(dh1); free(dh3); free(bwdInput); free(bwdOutput);
            return 1;
        }

        for (int i = 0; i < SEQ * HIDDEN; i++) {
            dh1[i] = 0.01f * sinf((float)i * 0.007f);
            dh3[i] = 0.01f * cosf((float)i * 0.011f);
        }

        for (int t = 0; t < SEQ; t++) {
            for (int c = 0; c < HIDDEN; c++) {
                bwdInput[c * SEQ + t] = dh1[t * HIDDEN + c];
                bwdInput[(HIDDEN + c) * SEQ + t] = dh3[t * HIDDEN + c];
            }
        }

        NSString *bwdInputPath = [outDir stringByAppendingPathComponent:@"fused_bwd_input_seq64_f32le.bin"];
        if (!write_f32le(bwdInputPath, bwdInput, (size_t)(HIDDEN * 2 * SEQ))) {
            cleanup_kern(&bwdKernel);
            free(BW1); free(BW3);
            free(dh1); free(dh3); free(bwdInput); free(bwdOutput);
            return 1;
        }
        print_file_info(bwdInputPath);

        IOSurfaceRef bwdInSurface = make_surface((size_t)HIDDEN * 2 * SEQ * 4);
        IOSurfaceRef bwdOutSurface = make_surface((size_t)DIM * SEQ * 4);

        IOSurfaceLock(bwdInSurface, 0, NULL);
        float *bwdInPtr = (float *)IOSurfaceGetBaseAddress(bwdInSurface);
        memcpy(bwdInPtr, bwdInput, (size_t)HIDDEN * 2 * SEQ * sizeof(float));
        IOSurfaceUnlock(bwdInSurface, 0, NULL);

        IOSurfaceRef bwdIns[] = {bwdInSurface};
        IOSurfaceRef bwdOuts[] = {bwdOutSurface};
        if (!ane_eval_io(&bwdKernel, bwdIns, 1, bwdOuts, 1)) {
            cleanup_kern(&bwdKernel);
            CFRelease(bwdInSurface); CFRelease(bwdOutSurface);
            free(BW1); free(BW3);
            free(dh1); free(dh3); free(bwdInput); free(bwdOutput);
            return 1;
        }

        IOSurfaceLock(bwdOutSurface, kIOSurfaceLockReadOnly, NULL);
        float *bwdOutPtr = (float *)IOSurfaceGetBaseAddress(bwdOutSurface);
        memcpy(bwdOutput, bwdOutPtr, (size_t)DIM * SEQ * sizeof(float));
        IOSurfaceUnlock(bwdOutSurface, kIOSurfaceLockReadOnly, NULL);

        NSString *bwdOutPath = [outDir stringByAppendingPathComponent:@"fused_bwd_dx_seq64_f32le.bin"];
        if (!write_f32le(bwdOutPath, bwdOutput, (size_t)DIM * SEQ)) {
            cleanup_kern(&bwdKernel);
            CFRelease(bwdInSurface); CFRelease(bwdOutSurface);
            free(BW1); free(BW3);
            free(dh1); free(dh3); free(bwdInput); free(bwdOutput);
            return 1;
        }
        print_file_info(bwdOutPath);

        cleanup_kern(&bwdKernel);
        CFRelease(bwdInSurface);
        CFRelease(bwdOutSurface);
        free(BW1); free(BW3);
        free(dh1); free(dh3); free(bwdInput); free(bwdOutput);

        printf("Done. Generated binary ObjC cross-validation oracles.\n");
        return 0;
    }
}
