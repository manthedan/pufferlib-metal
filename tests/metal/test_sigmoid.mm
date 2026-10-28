// Exhaustive check of puf_sigmoid_exact in src/metal/shaders/common.h:
// compare every fp32 input (including all BF16 values and NaNs) against the
// device's original IEEE-division sigmoid in each translation unit. Require
// <= 1 ULP by default, or bit equality with PUF_SIG_REQUIRE_EXACT=1.
// An independent CPU oracle checks a strided sample using the GPU's own
// exp and rounded denominator, accepting only MSL's RN/RZ and optional FTZ
// outcomes for the IEEE reference, and the chosen ULP budget for the result.
// Kernels: PUF_SIG_TEST_KERNELS(SUF, FN) instances (PUF_TEST_SIGMOID builds).
// Usage: test_sigmoid LIB.metallib:SUF [LIB.metallib:SUF ...] [--quick]
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>

#include <string>
#include <vector>

struct SigArgs {
    uint64_t cnt, rec, dump;
    uint32_t base, per_thread, stride;
};

static float bits2f(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }
static uint32_t f2bits(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }

static uint32_t ulp_distance(uint32_t got, uint32_t want) {
    // Positive finite oracle values, including the denominator; either zero sign.
    if ((got & 0x7fffffffu) == 0) got = 0;
    if ((want & 0x7fffffffu) == 0) want = 0;
    if (got >= 0x7f800000u || want >= 0x7f800000u) return UINT32_MAX;
    return got > want ? got - want : want - got;
}

// Round positive binary64 values in the oracle's [2^-149, FLT_MAX] range by integer
// operations, independent of the host's rounding mode and subnormal settings.
// Double division of fp32 operands has enough precision for both fp32 RN/RZ.
static bool permitted_rounding(double value, uint32_t got, uint32_t budget = 0) {
    if (value == 0.0) return ulp_distance(got, 0) <= budget;
    uint64_t bits;
    memcpy(&bits, &value, sizeof(bits));
    int exponent = int((bits >> 52) & 0x7ffu) - 1023;
    int shift = exponent < -126 ? -97 - exponent : 29;
    uint64_t sig = (bits & 0xfffffffffffffull) | (1ull << 52);
    uint32_t rz = uint32_t(sig >> shift);
    if (exponent >= -126) rz += uint32_t(exponent + 126) << 23;
    uint64_t rem = sig & ((1ull << shift) - 1), half = 1ull << (shift - 1);
    uint32_t rn = rz + (rem > half || (rem == half && (rz & 1u)));
    return ulp_distance(got, rn) <= budget || ulp_distance(got, rz) <= budget ||
        (value < 0x1p-126 && ulp_distance(got, 0) <= budget);
}

static bool oracle_accepts(float x, float z, float den, float result, uint32_t budget = 0) {
    // Judge division using the device's exp, including any small overshoot
    // above 1 permitted by exp's error bound near zero.
    if (!(z >= 0.0f && std::isfinite(z) && den >= 1.0f && std::isfinite(den))) return false;
    if (!permitted_rounding(1.0 + double(z), f2bits(den))) return false;
    double numerator = x >= 0.0f ? 1.0 : double(z);
    return permitted_rounding(numerator / double(den), f2bits(result), budget);
}

static bool oracle_self_check() {
    // RN ties, RZ, either FTZ zero sign, preserved subnormals, and GPU-rounded
    // denominators. A two-ULP normal error must never acquire an FTZ exemption.
    float z = bits2f(0x3f7ffffeu), den = bits2f(0x3fffffffu);
    bool ok = permitted_rounding(0.5 + 0x1p-25, 0x3f000000u) &&
        !permitted_rounding(0.5 + 0x1p-25, 0x3f000001u) &&
        permitted_rounding(0.5 + 0x1.8p-24, 0x3f000001u) &&
        permitted_rounding(0.5 + 0x1.8p-24, 0x3f000002u) &&
        oracle_accepts(-90.0f, bits2f(1), 1.0f, bits2f(1)) &&
        oracle_accepts(-90.0f, bits2f(1), 1.0f, 0.0f) &&
        oracle_accepts(-90.0f, bits2f(1), 1.0f, -0.0f) &&
        !oracle_accepts(-90.0f, bits2f(1), 1.0f, bits2f(2)) &&
        !oracle_accepts(-1.0f, 0.5f, 1.5f, 0.0f) &&
        oracle_accepts(0x1p-23f, z, den, bits2f(0x3f000001u)) &&
        oracle_accepts(0x1p-23f, z, den, 0.5f) &&
        oracle_accepts(0x1p-24f, bits2f(0x3f7fffffu), den, bits2f(0x3f000001u)) &&
        oracle_accepts(-0x1p-24f, bits2f(0x3f800001u), 2.0f, bits2f(0x3f000001u)) &&
        !oracle_accepts(0.0f, 1.0f, 2.0f, bits2f(0x3f000001u)) &&
        oracle_accepts(0.0f, 1.0f, 2.0f, bits2f(0x3f000001u), 1) &&
        !oracle_accepts(0.0f, 1.0f, 2.0f, bits2f(0x3f000002u), 1) &&
        !oracle_accepts(0.0f, 1.0f, bits2f(0x40000001u), 0.5f);
    if (!ok) fprintf(stderr, "CPU rounding/FTZ oracle self-check failed\n");
    return ok;
}

int main(int argc, char** argv) {
    @autoreleasepool {
        setvbuf(stdout, nullptr, _IOLBF, 0);
        if (!oracle_self_check()) return 2;
        const char* exact_env = getenv("PUF_SIG_REQUIRE_EXACT");
        bool require_exact = exact_env && !strcmp(exact_env, "1");
        uint32_t budget = require_exact ? 0 : 1;
        printf("sigmoid tolerance: %s\n", require_exact ? "bit equality" : "<= 1 ULP");
        bool quick = false;
        std::vector<std::pair<std::string, std::string>> targets;
        for (int i = 1; i < argc; i++) {
            if (!strcmp(argv[i], "--quick")) { quick = true; continue; }
            std::string t = argv[i];
            size_t c = t.rfind(':');
            if (c == std::string::npos) { fprintf(stderr, "bad target %s\n", argv[i]); return 2; }
            targets.push_back({t.substr(0, c), t.substr(c + 1)});
        }
        if (targets.empty()) {
            fprintf(stderr, "usage: %s LIB.metallib:SUF [...] [--quick]\n", argv[0]);
            return 2;
        }
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> q = [dev newCommandQueue];
        id<MTLBuffer> cb = [dev newBufferWithLength:16 options:MTLResourceStorageModeShared];
        id<MTLBuffer> rb = [dev newBufferWithLength:64 * 3 * 4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> db = [dev newBufferWithLength:(size_t)(1u << 20) * 16 options:MTLResourceStorageModeShared];
        int fails = 0;
        for (auto& tg : targets) {
        std::string tag = tg.first.substr(tg.first.rfind('/') + 1) + ":" + tg.second;
        NSError* err = nil;
        id<MTLLibrary> lib = [dev newLibraryWithURL:[NSURL fileURLWithPath:@(tg.first.c_str())] error:&err];
        if (!lib) { fprintf(stderr, "library: %s\n", err.description.UTF8String); return 2; }
        auto pso = [&](const std::string& name) {
            id<MTLFunction> f = [lib newFunctionWithName:@(name.c_str())];
            if (!f) { fprintf(stderr, "missing %s\n", name.c_str()); exit(2); }
            id<MTLComputePipelineState> p = [dev newComputePipelineStateWithFunction:f error:nil];
            if (!p) { fprintf(stderr, "pipeline %s failed\n", name.c_str()); exit(2); }
            return p;
        };
        id<MTLComputePipelineState> chk = pso("sig_check_" + tg.second), dmp = pso("sig_dump_" + tg.second);
        auto run = [&](id<MTLComputePipelineState> p, SigArgs a, uint32_t threads) {
            id<MTLCommandBuffer> c = [q commandBuffer];
            id<MTLComputeCommandEncoder> e = [c computeCommandEncoder];
            [e setComputePipelineState:p];
            [e setBytes:&a length:sizeof(a) atIndex:0];
            [e useResource:cb usage:MTLResourceUsageRead | MTLResourceUsageWrite];
            [e useResource:rb usage:MTLResourceUsageWrite];
            [e useResource:db usage:MTLResourceUsageWrite];
            [e dispatchThreads:MTLSizeMake(threads, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [e endEncoding];
            [c commit];
            [c waitUntilCompleted];
            if (c.error) { fprintf(stderr, "GPU error: %s\n", c.error.description.UTF8String); exit(2); }
        };
        // 2^32 patterns in 16 batches of 2^28 (2^20 threads x 256 each).
        // --quick: batches 0, 4, 8, 12, 15 (including both signed zeros).
        const uint32_t per_thread = 256, threads = 1u << 20;
        uint64_t checked = 0, mismatches = 0, nan_mismatches = 0;
        uint32_t max_ulp = 0, printed = 0;
        for (uint32_t b = 0; b < 16; b++) {
            if (quick && b % 4 != 0 && b != 15) continue;
            // Per-batch counters cannot wrap, even if every input mismatches.
            memset(cb.contents, 0, 16);
            run(chk, {cb.gpuAddress, rb.gpuAddress, db.gpuAddress, b << 28, per_thread, 0}, threads);
            checked += 1ull << 28;
            const uint32_t* cnt = (const uint32_t*)cb.contents;
            const uint32_t* rec = (const uint32_t*)rb.contents;
            mismatches += cnt[0];
            nan_mismatches += cnt[1];
            if (cnt[3] > max_ulp) max_ulp = cnt[3];
            for (uint32_t i = 0; i < cnt[2] && i < 64 && printed < 8; i++, printed++)
                printf("  x=%08x (%g) shipped=%08x ref=%08x\n", rec[3 * i], bits2f(rec[3 * i]),
                    rec[3 * i + 1], rec[3 * i + 2]);
        }
        printf("[%s] sigmoid exhaustive: %llu patterns, mismatches %llu (+%llu with NaN input), max ULP %u\n",
            tag.c_str(), (unsigned long long)checked, (unsigned long long)mismatches,
            (unsigned long long)nan_mismatches, max_ulp);
        fails += require_exact ? mismatches != 0 || nan_mismatches != 0 : max_ulp > budget;
        // Independent oracle on a strided sample: 2^20 points spaced 4099
        // apart span all 2^32 patterns (both signs, every exponent).
        const uint32_t STRIDE = 4099, PTS = 1u << 20;
        uint64_t sampled = 0, sampled_neg = 0, bad = 0, bad_ref = 0;
        for (uint32_t base = 0; base < STRIDE; base += quick ? 257 : 13) {
            run(dmp, {cb.gpuAddress, rb.gpuAddress, db.gpuAddress, base, 0, STRIDE}, PTS);
            const float* d = (const float*)db.contents;
            for (uint32_t i = 0; i < PTS; i++) {
                uint64_t xb = (uint64_t)base + (uint64_t)i * STRIDE;
                if (xb > 0xffffffffull) break;
                float x = bits2f((uint32_t)xb), z = d[4 * i], den = d[4 * i + 1];
                float got = d[4 * i + 2], ref = d[4 * i + 3];
                if (std::isnan(x)) continue;
                sampled++;
                sampled_neg += x < 0.0f;
                bool ref_ok = oracle_accepts(x, z, den, ref);
                bool got_ok = oracle_accepts(x, z, den, got, budget);
                if (!ref_ok || !got_ok) {
                    if (bad + bad_ref < 8) printf("  oracle x=%08x z=%08x d=%08x shipped=%08x ref=%08x\n",
                        (uint32_t)xb, f2bits(z), f2bits(den), f2bits(got), f2bits(ref));
                    bad += !got_ok;
                    bad_ref += !ref_ok;
                }
            }
        }
        printf("[%s] sigmoid oracle: %llu sampled (%llu negative), %llu outside tolerance, %llu invalid IEEE references (RN/RZ, optional FTZ)\n",
            tag.c_str(), (unsigned long long)sampled, (unsigned long long)sampled_neg,
            (unsigned long long)bad, (unsigned long long)bad_ref);
        fails += bad != 0 || bad_ref != 0 || sampled_neg < sampled / 3 || sampled - sampled_neg < sampled / 3;
        }
        printf("%s\n", fails ? "FAIL" : "PASS");
        return fails ? 1 : 0;
    }
}
