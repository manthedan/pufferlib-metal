// Rollout-kernel test for the Metal backend: the real PUF_LAUNCH adapters in
// src/metal/kernels.h dispatching the MSL ports in src/metal/shaders/.
//
// Golden (CUDA FP32 dumps, see ../golden/REPORT.md):
//   gate_*    mingru_gate            out / state_out
//   sample_*  sample_logits          raw 64-byte Philox state out, actions,
//                                    env_actions (exact); logprobs/values
//   sample_*  rng_init               curand_init(73, agent, 0) followed by
//                                    agent%7 discarded uniforms (host Philox
//                                    stepper) must equal the dumped input state
//   sample_*  curand_uniform         draws from cloned input states (test hook
//                                    puf_test_curand_uniform), bit-exact
// Continuous sample_logits (curand_normal, Box-Muller cache reuse across
// launches) vs a host reference. Synthetic (host reference): snapshot_state,
// zero_term_state, cast, add, clamp, transpose_102, uniform_scale.
//
// Build/run: tests/metal/run_test_kernels.sh [golden_run_dir ...]
#import <Foundation/Foundation.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "compat/cuda_runtime.h"
#include "compat/curand_kernel.h"
#include "launch.h"
#include "runtime.h"

// Minimal stand-ins for the upstream types kernels.h adapts (src/pufferl.cu).
typedef float precision_t;
#define PUF_MAX_DIMS 8
typedef struct { float* data; int64_t shape[PUF_MAX_DIMS]; } Float;
typedef struct { precision_t* data; int64_t shape[PUF_MAX_DIMS]; } Prec;
static int64_t numel(int64_t* shape) {
    int64_t n = 1;
    for (int i = 0; i < PUF_MAX_DIMS && shape[i] != 0; i++) n *= shape[i];
    return n;
}
static int g_num_atns = 1;
#define NUM_ATNS g_num_atns

#include "kernels.h"

static int g_fail = 0;

template <typename T>
static T* dalloc(long n) {
    T* p = nullptr;
    if (cudaMalloc(&p, (size_t)(n > 0 ? n : 1) * sizeof(T)) != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed\n");
        exit(2);
    }
    return p;
}

static int grid(int n) { return (n + 255) / 256; }

struct Err {
    double max_abs = 0, max_rel = 0;
    long mismatches = 0;  // non-bit-identical elements
    long n = 0;
};

static Err compare_f32(const float* x, const float* ref, long n) {
    Err e;
    e.n = n;
    for (long i = 0; i < n; i++) {
        if (memcmp(&x[i], &ref[i], 4) != 0) e.mismatches++;
        double a = x[i], b = ref[i];
        if (std::isnan(a) && std::isnan(b)) continue;
        double d = fabs(a - b);
        if (!std::isfinite(d)) d = INFINITY;
        double den = fmax(fmax(fabs(a), fabs(b)), 2.2250738585072014e-308);
        e.max_abs = fmax(e.max_abs, d);
        e.max_rel = fmax(e.max_rel, d / den);
    }
    return e;
}

static void report(const char* cs, const char* what, const Err& e, double tol_abs) {
    bool ok = e.max_abs <= tol_abs;
    if (!ok) g_fail++;
    printf("  %-22s %-14s n=%-7ld bit-exact=%-7ld max_abs=%.3e max_rel=%.3e%s\n", cs, what,
        e.n, e.n - e.mismatches, e.max_abs, e.max_rel, ok ? "" : "  FAIL");
}

static void report_exact(const char* cs, const char* what, long n, long mismatches) {
    if (mismatches) g_fail++;
    printf("  %-22s %-14s n=%-7ld mismatches=%ld%s\n", cs, what, n, mismatches,
        mismatches ? "  FAIL" : "");
}

// ---------------------------------------------------------------------------
// Golden loading
// ---------------------------------------------------------------------------

struct Case {
    NSString* dir;
    NSDictionary* tensors;
    NSDictionary* scalars;
};

static Case load_case(NSString* dir) {
    NSData* mj = [NSData dataWithContentsOfFile:[dir stringByAppendingPathComponent:@"manifest.json"]];
    NSDictionary* man = [NSJSONSerialization JSONObjectWithData:mj options:0 error:nil];
    if (!man) {
        fprintf(stderr, "bad manifest in %s\n", dir.UTF8String);
        exit(2);
    }
    return {dir, man[@"tensors"], man[@"scalars"]};
}

static NSData* read_raw(const Case& c, NSString* name) {
    NSDictionary* spec = c.tensors[name];
    if (!spec) {
        fprintf(stderr, "%s: no tensor %s\n", c.dir.UTF8String, name.UTF8String);
        exit(2);
    }
    NSData* d = [NSData dataWithContentsOfFile:[c.dir stringByAppendingPathComponent:spec[@"file"]]];
    if (!d) {
        fprintf(stderr, "cannot read %s/%s\n", c.dir.UTF8String, name.UTF8String);
        exit(2);
    }
    return d;
}

template <typename T>
static std::vector<T> read_vec(const Case& c, NSString* name) {
    NSData* d = read_raw(c, name);
    std::vector<T> v(d.length / sizeof(T));
    memcpy(v.data(), d.bytes, d.length);
    return v;
}

template <typename T>
static T* upload(const std::vector<T>& v) {
    T* p = dalloc<T>((long)v.size());
    memcpy(p, v.data(), v.size() * sizeof(T));
    return p;
}

// ---------------------------------------------------------------------------
// Host Philox stepper (cuRAND curand() semantics) for the agent%7 discards.
// ---------------------------------------------------------------------------

static void philox_block(uint32_t c[4], uint32_t k0, uint32_t k1, uint32_t out[4]) {
    uint32_t x[4] = {c[0], c[1], c[2], c[3]};
    for (int r = 0; r < 10; r++) {
        uint64_t p0 = (uint64_t)0xD2511F53u * x[0];
        uint64_t p1 = (uint64_t)0xCD9E8D57u * x[2];
        uint32_t y[4] = {(uint32_t)(p1 >> 32) ^ x[1] ^ k0, (uint32_t)p1,
            (uint32_t)(p0 >> 32) ^ x[3] ^ k1, (uint32_t)p0};
        memcpy(x, y, sizeof(x));
        k0 += 0x9E3779B9u;
        k1 += 0xBB67AE85u;
    }
    memcpy(out, x, sizeof(x));
}

static uint32_t host_curand(curandStatePhilox4_32_10_t* s) {
    uint32_t ret = s->output[s->STATE & 3];
    s->STATE++;
    if (s->STATE == 4) {
        if (++s->ctr[0] == 0 && ++s->ctr[1] == 0 && ++s->ctr[2] == 0) ++s->ctr[3];
        philox_block(s->ctr, s->key[0], s->key[1], s->output);
        s->STATE = 0;
    }
    return ret;
}

static void host_curand_discard(curandStatePhilox4_32_10_t* s) { host_curand(s); }

// cuRAND curand_normal (Box-Muller pair, second value cached in the state).
static float host_curand_normal(curandStatePhilox4_32_10_t* s) {
    const float INV = 2.3283064e-10f, INV_2PI = 2.3283064e-10f * 6.2831855f;
    if (s->boxmuller_flag != 1) {
        uint32_t x = host_curand(s), y = host_curand(s);
        float u = fmaf((float)x, INV, INV / 2.0f);
        float v = fmaf((float)y, INV_2PI, INV_2PI / 2.0f);
        float r = sqrtf(-2.0f * logf(u));
        s->boxmuller_extra = cosf(v) * r;
        s->boxmuller_flag = 1;
        return sinf(v) * r;
    }
    s->boxmuller_flag = 0;
    return s->boxmuller_extra;
}

static float host_finite_or_clamp(float x, float lo, float hi) {
    if (std::isnan(x)) return 0.0f;
    if (std::isinf(x)) return x > 0.0f ? hi : lo;
    return fminf(hi, fmaxf(lo, x));
}

// ---------------------------------------------------------------------------
// Golden cases
// ---------------------------------------------------------------------------

static void run_gate(const Case& c, const char* cs) {
    int B = [c.scalars[@"B"] intValue], H = [c.scalars[@"H"] intValue];
    float* comb = upload(read_vec<float>(c, @"combined"));
    float* st = upload(read_vec<float>(c, @"state_in"));
    float* x = upload(read_vec<float>(c, @"x"));
    float* out = dalloc<float>((long)B * H);
    float* next = dalloc<float>((long)B * H);
    PUF_LAUNCH(mingru_gate, grid(B * H), 256, 0, 0)(out, next, comb, st, x, H, B);
    cudaDeviceSynchronize();
    std::vector<float> ro = read_vec<float>(c, @"out"), rs = read_vec<float>(c, @"state_out");
    report(cs, "out", compare_f32(out, ro.data(), (long)B * H), 1e-5);
    report(cs, "state_out", compare_f32(next, rs.data(), (long)B * H), 1e-5);
    cudaFree(comb); cudaFree(st); cudaFree(x); cudaFree(out); cudaFree(next);
}

static void run_sample(const Case& c, const char* cs) {
    int B = [c.scalars[@"B"] intValue];
    unsigned long long seed = [c.scalars[@"seed"] unsignedLongLongValue];
    std::vector<uint8_t> rng_in = read_vec<uint8_t>(c, @"rng_in");
    std::vector<uint8_t> rng_out_ref = read_vec<uint8_t>(c, @"rng_out");
    std::vector<int> act_sizes = read_vec<int>(c, @"act_sizes");
    std::vector<float> dec = read_vec<float>(c, @"decoder");
    std::vector<float> masks = read_vec<float>(c, @"masks");
    int num_atns = (int)act_sizes.size();
    int fused = (int)(dec.size() / B);
    int mask_stride = (int)(masks.size() / B);
    g_num_atns = num_atns;

    // rng_init + host discards vs the dumped input state.
    curandStatePhilox4_32_10_t* st = dalloc<curandStatePhilox4_32_10_t>(B);
    PUF_LAUNCH(rng_init, grid(B), 256, 0, 0)(st, (uint64_t)seed, B);
    cudaDeviceSynchronize();
    for (int i = 0; i < B; i++) {
        for (int d = 0; d < i % 7; d++) host_curand_discard(&st[i]);
    }
    long bad = 0;
    for (int i = 0; i < B; i++) {
        if (memcmp(&st[i], rng_in.data() + (size_t)i * 64, 64) != 0) bad++;
    }
    report_exact(cs, "rng_init", B, bad);

    // curand_uniform from cloned input states vs the dumped draws.
    memcpy(st, rng_in.data(), (size_t)B * 64);
    {
        struct { uint64_t states, out; int32_t n, draws; } pa;
        float* draws = dalloc<float>((long)B * num_atns);
        pa = {puf_mtl::dptr(st), puf_mtl::dptr(draws), B, num_atns};
        static puf_mtl::Pipeline* pp = puf_mtl::pipeline("puf_test_curand_uniform");
        puf_mtl::dispatch(0, pp, dim3(grid(B)), dim3(256), pa);
        cudaDeviceSynchronize();
        std::vector<float> ru = read_vec<float>(c, @"uniform_draws");
        Err eu = compare_f32(draws, ru.data(), (long)B * num_atns);
        report_exact(cs, "uniform_draws", eu.n, eu.mismatches);
        cudaFree(draws);
    }

    // sample_logits from the dumped state.
    Prec dec_p = {upload(dec), {B, fused}};
    Prec logstd = {};
    int* d_act = upload(act_sizes);
    float* d_mask = upload(masks);
    float* actions = dalloc<float>((long)B * num_atns);
    float* env_actions = dalloc<float>((long)B * num_atns);
    float* logprobs = dalloc<float>(B);
    float* values = dalloc<float>(B);
    PUF_LAUNCH(sample_logits, grid(B), 256, 0, 0)(dec_p, logstd, d_act, actions, env_actions,
        logprobs, values, st, d_mask, mask_stride);
    cudaDeviceSynchronize();
    bad = 0;
    for (int i = 0; i < B; i++) {
        if (memcmp(&st[i], rng_out_ref.data() + (size_t)i * 64, 64) != 0) bad++;
    }
    report_exact(cs, "rng_out", B, bad);
    std::vector<float> ra = read_vec<float>(c, @"actions"), rea = read_vec<float>(c, @"env_actions");
    Err ea = compare_f32(actions, ra.data(), (long)B * num_atns);
    report_exact(cs, "actions", ea.n, ea.mismatches);
    Err eea = compare_f32(env_actions, rea.data(), (long)B * num_atns);
    report_exact(cs, "env_actions", eea.n, eea.mismatches);
    std::vector<float> rl = read_vec<float>(c, @"logprobs"), rv = read_vec<float>(c, @"values");
    report(cs, "logprobs", compare_f32(logprobs, rl.data(), B), 1e-5);
    Err ev = compare_f32(values, rv.data(), B);
    report_exact(cs, "values", ev.n, ev.mismatches);
    cudaFree(st); cudaFree(dec_p.data); cudaFree(d_act); cudaFree(d_mask);
    cudaFree(actions); cudaFree(env_actions); cudaFree(logprobs); cudaFree(values);
}

static void run_golden(const char* run_dir) {
    NSString* root = [NSString stringWithUTF8String:run_dir];
    NSArray* names = [[[NSFileManager defaultManager] contentsOfDirectoryAtPath:root error:nil]
        sortedArrayUsingSelector:@selector(compare:)];
    printf("== golden (%s)\n", run_dir);
    int count = 0;
    for (NSString* name in names) {
        NSString* dir = [root stringByAppendingPathComponent:name];
        if ([name hasPrefix:@"gate_"]) {
            run_gate(load_case(dir), name.UTF8String);
        } else if ([name hasPrefix:@"sample_"]) {
            run_sample(load_case(dir), name.UTF8String);
        } else {
            continue;
        }
        count++;
    }
    printf("golden cases: %d\n", count);
    if (count == 0) g_fail++;
}

// ---------------------------------------------------------------------------
// Synthetic cases vs host reference
// ---------------------------------------------------------------------------

// Continuous sample_logits (curand_normal) vs a host reference: 3 heads (odd,
// so the Box-Muller cache is left populated between launches), two launches,
// some agents starting with a populated cache.
static void run_continuous() {
    printf("== continuous sample_logits vs host reference\n");
    const char* cs = "continuous";
    const int B = 37, NA = 3, fused = NA + 1;
    g_num_atns = NA;
    std::mt19937 rng(77);
    std::uniform_real_distribution<float> U(-1.5f, 1.5f);
    curandStatePhilox4_32_10_t* st = dalloc<curandStatePhilox4_32_10_t>(B);
    PUF_LAUNCH(rng_init, grid(B), 256, 0, 0)(st, (uint64_t)0x123456789abcull, B);
    cudaDeviceSynchronize();
    for (int i = 0; i < B; i++) {
        for (int d = 0; d < i % 5; d++) host_curand_discard(&st[i]);
        if (i % 4 == 0) {
            st[i].boxmuller_flag = 1;
            st[i].boxmuller_extra = 0.25f * (float)(i % 7) - 0.5f;
        }
    }
    std::vector<curandStatePhilox4_32_10_t> ref_st(st, st + B);
    Prec dec = {dalloc<float>((long)B * fused), {B, fused}};
    for (long i = 0; i < (long)B * fused; i++) dec.data[i] = U(rng);
    dec.data[3 * fused + 1] = NAN;        // finite_or_clamp paths
    dec.data[5 * fused + 2] = INFINITY;
    Prec logstd = {dalloc<float>(NA), {1, NA}};
    logstd.data[0] = -0.7f;
    logstd.data[1] = 0.3f;
    logstd.data[2] = 5.0f;  // clamped to 2
    int* act = dalloc<int>(NA);
    for (int h = 0; h < NA; h++) act[h] = 1;
    float* actions = dalloc<float>((long)B * NA);
    float* env_actions = dalloc<float>((long)B * NA);
    float* logprobs = dalloc<float>(B);
    float* values = dalloc<float>(B);
    std::vector<float> ref_act((size_t)B * NA), ref_lp(B);
    double max_act = 0, max_lp = 0, max_extra = 0;
    long st_bad = 0, env_bad = 0, val_bad = 0;
    for (int launch = 0; launch < 2; launch++) {
        PUF_LAUNCH(sample_logits, grid(B), 256, 0, 0)(dec, logstd, act, actions, env_actions,
            logprobs, values, st, (float*)nullptr, 0);
        cudaDeviceSynchronize();
        for (int i = 0; i < B; i++) {
            float total = 0.0f;
            for (int h = 0; h < NA; h++) {
                float mean = host_finite_or_clamp(dec.data[i * fused + h], -1.0e6f, 1.0e6f);
                float log_std = host_finite_or_clamp(logstd.data[h], -20.0f, 2.0f);
                float a = host_finite_or_clamp(
                    mean + expf(log_std) * host_curand_normal(&ref_st[i]), -1.0e6f, 1.0e6f);
                float nrm = (a - mean) / expf(log_std);
                total += -0.5f * nrm * nrm - 0.9189385332046727f - log_std;
                ref_act[(size_t)i * NA + h] = a;
            }
            ref_lp[i] = total;
            const curandStatePhilox4_32_10_t &g = st[i], &r = ref_st[i];
            bool same = !memcmp(g.ctr, r.ctr, 16) && !memcmp(g.output, r.output, 16)
                && !memcmp(g.key, r.key, 8) && g.STATE == r.STATE
                && g.boxmuller_flag == r.boxmuller_flag;
            st_bad += !same;
            max_extra = fmax(max_extra, fabs(g.boxmuller_extra - r.boxmuller_extra));
            ref_st[i].boxmuller_extra = g.boxmuller_extra;  // don't compound ulp drift
            if (values[i] != dec.data[i * fused + NA]) val_bad++;
        }
        for (long k = 0; k < (long)B * NA; k++) {
            max_act = fmax(max_act, fabs(actions[k] - ref_act[k]));
            if (memcmp(&actions[k], &env_actions[k], 4)) env_bad++;
        }
        for (int i = 0; i < B; i++) max_lp = fmax(max_lp, fabs(logprobs[i] - ref_lp[i]));
    }
    report_exact(cs, "rng state", 2L * B, st_bad);
    report_exact(cs, "env_actions==act", 2L * B * NA, env_bad);
    report_exact(cs, "values", 2L * B, val_bad);
    bool ok = max_act <= 1e-4 && max_lp <= 1e-3 && max_extra <= 1e-5;
    if (!ok) g_fail++;
    printf("  %-22s %-14s max_abs(actions)=%.3e max_abs(logprob)=%.3e max_abs(bm_extra)=%.3e%s\n",
        cs, "normal draws", max_act, max_lp, max_extra, ok ? "" : "  FAIL");
    cudaFree(st); cudaFree(dec.data); cudaFree(logstd.data); cudaFree(act);
    cudaFree(actions); cudaFree(env_actions); cudaFree(logprobs); cudaFree(values);
}

static void run_synthetic() {
    printf("== synthetic\n");
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> U(-2.0f, 2.0f);
    const char* cs = "synthetic";

    // State layout (L, agents, H): snapshot agents 0..count of src into
    // dst rows [dst_start, dst_start+count); zero rows whose terminal != 0.
    int L = 3, H = 5, src_agents = 7, dst_agents = 19, dst_start = 9, count = 6;
    Prec src = {dalloc<float>((long)L * src_agents * H), {L, src_agents, H}};
    Prec dst = {dalloc<float>((long)L * dst_agents * H), {L, dst_agents, H}};
    std::vector<float> ref_dst((size_t)L * dst_agents * H);
    for (long i = 0; i < L * src_agents * H; i++) src.data[i] = U(rng);
    for (long i = 0; i < L * dst_agents * H; i++) dst.data[i] = ref_dst[i] = U(rng);
    for (int l = 0; l < L; l++)
        for (int r = 0; r < count; r++)
            for (int h = 0; h < H; h++)
                ref_dst[((size_t)l * dst_agents + dst_start + r) * H + h] =
                    src.data[((size_t)l * src_agents + r) * H + h];
    PUF_LAUNCH(snapshot_state, grid(L * count * H), 256, 0, 0)(dst, src, dst_start, count);
    cudaDeviceSynchronize();
    Err e = compare_f32(dst.data, ref_dst.data(), (long)ref_dst.size());
    report_exact(cs, "snapshot_state", e.n, e.mismatches);

    int term_start = 4, state_start = 2;
    Float terms = {dalloc<float>(32), {32}};
    for (int i = 0; i < 32; i++) terms.data[i] = (i % 3 == 0) ? 1.0f : (i % 5 == 0 ? 0.5f : 0.0f);
    for (int l = 0; l < L; l++)
        for (int r = 0; r < count; r++)
            for (int h = 0; h < H; h++)
                if (terms.data[term_start + r] != 0.0f)
                    ref_dst[((size_t)l * dst_agents + state_start + r) * H + h] = 0.0f;
    PUF_LAUNCH(zero_term_state, grid(L * count * H), 256, 0, 0)(dst, terms, state_start,
        term_start, count);
    cudaDeviceSynchronize();
    e = compare_f32(dst.data, ref_dst.data(), (long)ref_dst.size());
    report_exact(cs, "zero_term_state", e.n, e.mismatches);

    // Elementwise ports.
    int n = 1000;
    float* a = dalloc<float>(n);
    float* b = dalloc<float>(n);
    unsigned char* u8 = dalloc<unsigned char>(n);
    std::vector<float> ref(n);
    for (int i = 0; i < n; i++) { a[i] = U(rng); b[i] = U(rng); ref[i] = a[i] + b[i]; }
    PUF_LAUNCH(add_kernel, 2, 128, 0, 0)(a, (const float*)b, n);
    cudaDeviceSynchronize();
    e = compare_f32(a, ref.data(), n);
    report_exact(cs, "add_kernel", e.n, e.mismatches);

    for (int i = 0; i < n; i++) ref[i] = fminf(fmaxf(a[i], -0.5f), 0.75f);
    PUF_LAUNCH(clamp_precision_kernel, 1, 64, 0, 0)(a, -0.5f, 0.75f, n);
    cudaDeviceSynchronize();
    e = compare_f32(a, ref.data(), n);
    report_exact(cs, "clamp", e.n, e.mismatches);

    for (int i = 0; i < n; i++) { b[i] = (float)i / n; ref[i] = b[i] * 2.0f * 0.3f - 0.3f; }
    PUF_LAUNCH(uniform_scale_kernel, grid(n), 256, 0, 0)(b, 0.3f, n);
    cudaDeviceSynchronize();
    e = compare_f32(b, ref.data(), n);
    report(cs, "uniform_scale", e, 1e-7);

    for (int i = 0; i < n; i++) { u8[i] = (unsigned char)(i * 7); ref[i] = (float)u8[i]; }
    PUF_LAUNCH(cast, grid(n), 256, 0, 0)(a, u8, n);
    cudaDeviceSynchronize();
    e = compare_f32(a, ref.data(), n);
    report_exact(cs, "cast_u8", e.n, e.mismatches);

    int A = 4, Bd = 10, C = 25;  // A*B*C == n
    for (int i = 0; i < n; i++) a[i] = U(rng);
    for (int ai = 0; ai < A; ai++)
        for (int bi = 0; bi < Bd; bi++)
            for (int ci = 0; ci < C; ci++)
                ref[(size_t)bi * A * C + ai * C + ci] = a[(size_t)ai * Bd * C + bi * C + ci];
    PUF_LAUNCH(transpose_102, grid(n), 256, 0, 0)(b, (const float*)a, A, Bd, C);
    cudaDeviceSynchronize();
    e = compare_f32(b, ref.data(), n);
    report_exact(cs, "transpose_102", e.n, e.mismatches);

    cudaFree(src.data); cudaFree(dst.data); cudaFree(terms.data);
    cudaFree(a); cudaFree(b); cudaFree(u8);
}

int main(int argc, char** argv) {
    @autoreleasepool {
        for (int i = 1; i < argc; i++) run_golden(argv[i]);
        run_continuous();
        run_synthetic();
        printf(g_fail ? "FAILED (%d)\n" : "ALL PASS\n", g_fail);
    }
    return g_fail ? 1 : 0;
}
