// Exhaustive check of the BF16 fused-MinGRU projection store (leg 21): the
// matmul2d result stored straight into a bfloat threadgroup tile must equal
// from_float() of the fp32 cooperative-tensor result (the original store) for
// every fp32 accumulator pattern. Kernels: PUF_TEST_PROJ_STORE section of
// src/metal/shaders/mingru.metal. Usage: test_proj_store LIB.metallib [--quick]
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include <algorithm>
#include "proj_store_counters.h"

struct PstArgs {
    uint64_t x, W, cnt;
    uint32_t base, total, series, K, slots, rows;
};

// Number of requested patterns in [begin, end) that lie in [lo, hi).
static uint64_t overlap(uint64_t begin, uint64_t end, uint64_t lo, uint64_t hi) {
    return std::min(end, hi) > std::max(begin, lo)
        ? std::min(end, hi) - std::max(begin, lo) : 0;
}

int main(int argc, char** argv) {
    @autoreleasepool {
        if (argc < 2) { fprintf(stderr, "usage: %s LIB.metallib [--quick]\n", argv[0]); return 2; }
        bool quick = argc > 2 && !strcmp(argv[2], "--quick");
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        NSError* err = nil;
        id<MTLLibrary> lib = [dev newLibraryWithURL:[NSURL fileURLWithPath:@(argv[1])] error:&err];
        if (!lib) { fprintf(stderr, "library: %s\n", err.description.UTF8String); return 2; }
        auto pso = [&](const char* name) {
            id<MTLFunction> f = [lib newFunctionWithName:@(name)];
            if (!f) { fprintf(stderr, "missing %s\n", name); exit(2); }
            id<MTLComputePipelineState> p = [dev newComputePipelineStateWithFunction:f error:nil];
            if (!p) { fprintf(stderr, "pipeline %s failed\n", name); exit(2); }
            return p;
        };
        id<MTLComputePipelineState> gen = pso("pst_gen");
        struct Cfg { const char* kernel; int K, rows_per_tg, threads; } cfgs[] = {
            {"pst_check64", 32, 64, 128}, {"pst_check32", 32, 32, 64}, {"pst_check_h64", 64, 16, 64}};
        const uint32_t ROWS = 1u << 20;
        const uint32_t series_total[3] = {2u * (231u << 23), 2u * (24u << 23), 256u};
        id<MTLCommandQueue> q = [dev newCommandQueue];
        id<MTLBuffer> xb = [dev newBufferWithLength:(size_t)ROWS * 64 * 2 options:MTLResourceStorageModeShared];
        const int MAXB = 4096;
        const size_t counter_bytes = PST_COUNTERS * sizeof(uint32_t);
        id<MTLBuffer> cb = [dev newBufferWithLength:MAXB * counter_bytes options:MTLResourceStorageModeShared];
        int fails = 0;
        for (const Cfg& c : cfgs) {
            id<MTLComputePipelineState> chk = pso(c.kernel);
            const uint32_t slots = c.K / 3;
            // W (3K, K): W[n][3n + i] = w for n < slots. w = 1 (0x3f80) or 2^-64 (0x1f80).
            // The short fourth series exercises ALL output columns with +/- (n+1).
            // Its weights and target oracle are independent of the sparse triplets.
            id<MTLBuffer> wb[3];
            for (int s = 0; s < 3; s++) {
                wb[s] = [dev newBufferWithLength:3 * c.K * c.K * 2 options:MTLResourceStorageModeShared];
                uint16_t* w = (uint16_t*)wb[s].contents;
                memset(w, 0, 3 * c.K * c.K * 2);
                if (s == 2) {
                    for (int n = 0; n < 3 * c.K; n++) {
                        float value = n + 1;
                        uint32_t bits;
                        memcpy(&bits, &value, sizeof(bits));
                        w[n * c.K] = bits >> 16;  // integers 1..192 are exact BF16
                    }
                } else {
                    for (uint32_t n = 0; n < slots; n++)
                        for (int i = 0; i < 3; i++) w[n * c.K + 3 * n + i] = s ? 0x1f80 : 0x3f80;
                }
            }
            for (uint32_t series = 0; series < 4; series++) {
                uint64_t tot[PST_COUNTERS] = {}, expected[PST_COUNTERS] = {};
                uint32_t rows = series == 3 ? 64 : series == 2 ? 256 : ROWS;
                uint32_t total = series == 3 ? rows * 3 * c.K : series_total[series];
                uint32_t per_row = series == 3 ? 3 * c.K : series == 2 ? 1 : slots;
                uint64_t per_batch = (uint64_t)rows * per_row;
                uint32_t nb = (uint32_t)((total + per_batch - 1) / per_batch);
                uint32_t stride = 1;
                if (quick && nb > 8) { stride = nb / 8; }
                memset(cb.contents, 0, MAXB * counter_bytes);
                int used = 0;
                std::vector<id<MTLCommandBuffer>> commands;
                for (uint32_t b = 0; b < nb; b++) {
                    // Keep both FTZ signs and -0 in the quick sample as well.
                    if (b % stride && !(series == 1 && b == (24u << 23) / per_batch)) continue;
                    if (used == MAXB) { fprintf(stderr, "too many batches\n"); return 2; }
                    uint64_t begin = b * per_batch, end = std::min<uint64_t>(begin + per_batch, total);
                    uint64_t count = end - begin;
                    expected[PST_CONTROLLED] += count;
                    if (series != 2) {
                        uint64_t ftzp = 0, ftzn = 0, negzero = 0, rounded_zero = 0;
                        if (series == 1) {
                            const uint64_t sign = 24u << 23;
                            ftzp = overlap(begin, end, 1, 0x00800000);
                            ftzn = overlap(begin, end, sign + 1, sign + 0x00800000);
                            negzero = overlap(begin, end, sign, sign + 1);
                            // Only positive subnormals <= 0x00008000 round to +0.
                            rounded_zero = overlap(begin, end, 1, 0x00008001);
                        }
                        expected[PST_ACC_EXACT] += count - ftzp - ftzn - negzero;
                        expected[PST_ACC_FTZ_POS] += ftzp;
                        expected[PST_ACC_FTZ_NEG] += ftzn;
                        expected[PST_ACC_NEG_ZERO] += negzero;
                        expected[PST_ROUNDED_TARGET] += count - ftzp - ftzn - negzero + rounded_zero;
                        expected[PST_NONZERO] += count - ftzp - ftzn - negzero
                            - (series == 1 ? overlap(begin, end, 0, 1) : 0);
                    }
                    int wi = series == 3 ? 2 : series == 1 ? 1 : 0;
                    PstArgs a = {xb.gpuAddress, wb[wi].gpuAddress, cb.gpuAddress + used * counter_bytes,
                                 (uint32_t)begin, total, series, (uint32_t)c.K, slots, rows};
                    id<MTLCommandBuffer> cmd = [q commandBuffer];
                    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
                    [enc useResource:xb usage:MTLResourceUsageRead | MTLResourceUsageWrite];
                    [enc useResource:wb[wi] usage:MTLResourceUsageRead];
                    [enc useResource:cb usage:MTLResourceUsageRead | MTLResourceUsageWrite];
                    [enc setComputePipelineState:gen];
                    [enc setBytes:&a length:sizeof(a) atIndex:0];
                    [enc dispatchThreads:MTLSizeMake(rows * c.K, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
                    [enc setComputePipelineState:chk];
                    [enc dispatchThreadgroups:MTLSizeMake(rows / c.rows_per_tg, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(c.threads, 1, 1)];
                    [enc endEncoding];
                    [cmd commit];
                    commands.push_back(cmd);
                    used++;
                }
                id<MTLCommandBuffer> fin = [q commandBuffer];
                [fin commit];
                [fin waitUntilCompleted];
                for (id<MTLCommandBuffer> cmd : commands) {
                    if (cmd.status == MTLCommandBufferStatusError) {
                        fprintf(stderr, "GPU command: %s\n", cmd.error.description.UTF8String);
                        return 2;
                    }
                }
                const uint32_t* cnt = (const uint32_t*)cb.contents;
                for (int u = 0; u < used; u++)
                    for (int k = 0; k < PST_COUNTERS; k++) tot[k] += cnt[u * PST_COUNTERS + k];
                printf("%-14s series %u%s: patterns %llu  mismatches %llu  NaN bit diffs %llu  "
                       "reference==round(target) %llu  nonzero outputs %llu  bf16-subnormal outputs %llu\n",
                       c.kernel, series, stride > 1 ? " (quick sample)" : "",
                       (unsigned long long)tot[PST_CONTROLLED], (unsigned long long)tot[PST_MISMATCH],
                       (unsigned long long)tot[PST_NAN_BITS_DIFF], (unsigned long long)tot[PST_ROUNDED_TARGET],
                       (unsigned long long)tot[PST_NONZERO], (unsigned long long)tot[PST_SUBNORMAL]);
                if (series != 2) {
                    printf("  FP32 accumulators: exact %llu  FTZ +sub/-sub %llu/%llu  -0 -> +0 %llu  bad %llu\n",
                           (unsigned long long)tot[PST_ACC_EXACT], (unsigned long long)tot[PST_ACC_FTZ_POS],
                           (unsigned long long)tot[PST_ACC_FTZ_NEG], (unsigned long long)tot[PST_ACC_NEG_ZERO],
                           (unsigned long long)tot[PST_ACC_BAD]);
                }
                const char* labels[PST_COUNTERS] = {"bit mismatches", "NaN bit differences", "rounded targets",
                    "controlled patterns", "nonzero outputs", "BF16 subnormals", "exact FP32 accumulators",
                    "positive subnormal FTZ", "negative subnormal FTZ", "negative zero to positive zero",
                    "unexpected FP32 accumulators"};
                for (int k = 0; k < PST_COUNTERS; k++) {
                    // NaN canonicalization is allowed between target and reference,
                    // but ANY production/reference bit difference must fail.
                    if (series == 2 && k != PST_MISMATCH && k != PST_NAN_BITS_DIFF && k != PST_CONTROLLED) continue;
                    if (tot[k] != expected[k]) {
                        printf("  FAIL: %s: observed %llu, expected %llu\n", labels[k],
                               (unsigned long long)tot[k], (unsigned long long)expected[k]);
                        fails++;
                    }
                }
                if (series == 3) printf("  distinct nonzero targets in all %d output columns\n", 3 * c.K);
            }
        }
        printf(fails ? "FAIL (%d)\n" : "PASS\n", fails);
        return fails ? 1 : 0;
    }
}
