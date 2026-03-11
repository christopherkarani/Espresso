// golden_mil_gen.m — Generate golden MIL text fixtures from ObjC generators
// Usage: golden_mil_gen <output-dir>
#import <Foundation/Foundation.h>

#include "stories_mil.h"

int main(int argc, char *argv[]) {
    @autoreleasepool {
        if (argc < 2) {
            fprintf(stderr, "usage: golden_mil_gen <output-dir>\n");
            return 1;
        }

        NSString *outDir = [NSString stringWithUTF8String:argv[1]];
        [[NSFileManager defaultManager] createDirectoryAtPath:outDir
                                  withIntermediateDirectories:YES
                                                   attributes:nil
                                                        error:nil];

        NSDictionary<NSString *, NSString *> *files = @{
            @"sdpa_fwd_taps.mil": gen_sdpa_fwd_taps(),
            @"ffn_fwd_taps.mil": gen_ffn_fwd_taps(),
            @"ffn_bwd.mil": gen_ffn_bwd(),
            @"sdpa_bwd1.mil": gen_sdpa_bwd1(),
            @"sdpa_bwd2.mil": gen_sdpa_bwd2(),
            @"qkvb.mil": gen_qkvb(),
        };

        for (NSString *name in files) {
            NSString *path = [outDir stringByAppendingPathComponent:name];
            [files[name] writeToFile:path atomically:YES encoding:NSUTF8StringEncoding error:nil];
        }
    }
    return 0;
}

