// BF16 self-consistency test for the Metal backend (bf16 metallib, no
// PRECISION_FLOAT). Everything here is an exact (bitwise) check:
//
// 1. Conversions: cast_f32_to_prec / cast_prec_to_f32 over special values,
//    denormals, rounding ties and random bit patterns vs the host RNE
//    (compat/cuda_bf16.h, same as CUDA's __float2bfloat16 for non-NaN input).
// 2. GEMM: every kernel path (tensor_ops nt/nn/tn, nn_add, tiled, split-K,
//    sg_tn with and without split-K, alpha == 0 scale) at Breakout train /
//    rollout / Muon shapes. The bf16 kernel must take the same path as the
//    f32 kernel and produce exactly RNE(f32 kernel on the widened inputs):
//    both accumulate in fp32 in the same order and the bf16 one rounds once
//    at the store (C = bf16(alpha * acc + beta * C)). nn_add must equal
//    bf16(bf16(A @ B) + D) (rounded GEMM store, then add_kernel).
// 3. Muon: muon_ns5 and muon_ns5_batch (one dispatch) vs the unfused bf16
//    kernel sequence muon_step runs (sum_sq partials/reduce, l2 normalize,
//    five NS steps of bf16 GEMMs + copies, store_update).
// 4. puff_advantage bf16 vs a host reference with the kernel's explicit fma
//    contractions (no exp/log involved, so bit-exact).
//
// Build/run: tests/metal/run_test_bf16.sh
#import <Foundation/Foundation.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "compat/cuda_bf16.h"
#include "gemm.h"
#include "runtime.h"

#ifdef PRECISION_FLOAT
#error "test_bf16 is built against the bf16 metallib (no PRECISION_FLOAT)"
#endif

typedef __nv_bfloat16 bf16;
static int g_fail = 0;

static float widen(bf16 v) { return puf_bf16_bits_to_float(v.x); }
static uint16_t bits(float f) { return puf_float_to_bf16_bits(f); }
static uint32_t fbits(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }
static float ffrom(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }

template <typename T>
static T* dalloc(long n) {
    T* p = nullptr;
    if (cudaMalloc((void**)&p, (size_t)(n > 0 ? n : 1) * sizeof(T)) != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed\n");
        exit(2);
    }
    return p;
}

static void check(bool ok, const char* what) {
    printf("%-60s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) g_fail++;
}

// ---------------------------------------------------------------------------
// 1. conversions
// ---------------------------------------------------------------------------
struct CastArgs { uint64_t dst, src; int32_t n; };

static void test_casts() {
    std::vector<uint32_t> pat = {
        0x00000000u, 0x80000000u, 0x00000001u, 0x80000001u, 0x007fffffu, 0x00400000u,
        0x00008000u, 0x00018000u, 0x00007fffu, 0x0000ffffu, 0x00800000u, 0x3f800000u,
        0x3f808000u, 0x3f818000u, 0x3f807fffu, 0x3f80ffffu, 0x7f7fffffu, 0x7f7f8000u,
        0x7f800000u, 0xff800000u, 0x7fc00000u, 0x7f800001u, 0xffc12345u, 0x7fffffffu};
    std::mt19937 rng(7);
    for (int i = 0; i < 1 << 20; i++) pat.push_back(rng());
    for (int i = 0; i < 1 << 16; i++) pat.push_back(rng() & 0x807fffffu);  // denormals
    long n = (long)pat.size();
    float* src = dalloc<float>(n);
    bf16* mid = dalloc<bf16>(n);
    float* back = dalloc<float>(n);
    for (long i = 0; i < n; i++) src[i] = ffrom(pat[i]);
    puf_mtl::Pipeline* p_to = puf_mtl::pipeline("cast_f32_to_prec");
    puf_mtl::Pipeline* p_from = puf_mtl::pipeline("cast_prec_to_f32");
    unsigned grid = (unsigned)((n + 255) / 256);
    puf_mtl::dispatch(0, p_to, dim3(grid), dim3(256), CastArgs{puf_mtl::gpu_addr(mid),
        puf_mtl::gpu_addr(src), (int32_t)n});
    puf_mtl::dispatch(0, p_from, dim3(grid), dim3(256), CastArgs{puf_mtl::gpu_addr(back),
        puf_mtl::gpu_addr(mid), (int32_t)n});
    cudaDeviceSynchronize();
    long bad_to = 0, bad_back = 0;
    for (long i = 0; i < n; i++) {
        if (mid[i].x != bits(src[i])) {
            if (bad_to < 6) printf("  in 0x%08x got 0x%04x want 0x%04x\n", fbits(src[i]), mid[i].x, bits(src[i]));
            bad_to++;
        }
        if (fbits(back[i]) != ((uint32_t)mid[i].x << 16)) bad_back++;
    }
    char msg[128];
    snprintf(msg, sizeof(msg), "cast f32->bf16 RNE (%ld values, %ld mismatches)", n, bad_to);
    check(bad_to == 0, msg);
    snprintf(msg, sizeof(msg), "cast bf16->f32 exact (%ld mismatches)", bad_back);
    check(bad_back == 0, msg);
    cudaFree(src); cudaFree(mid); cudaFree(back);
}

// ---------------------------------------------------------------------------
// 2. GEMM
// ---------------------------------------------------------------------------
struct GemmCase {
    const char* name;
    bool ta, tb;
    int M, N, K;
    float alpha, beta;
    puf_mtl::GemmPath want;
    bool tensor_ops, split_k, sg_tn;
    bool small_n = true;
};

static void fill_bf16(bf16* p, long n, std::mt19937& rng, float scale) {
    std::normal_distribution<float> d(0.0f, scale);
    for (long i = 0; i < n; i++) p[i] = __float2bfloat16(d(rng));
}

static void test_gemm_case(const GemmCase& c, std::mt19937& rng, void* ws, size_t ws_bytes) {
    long na = (long)c.M * c.K, nb = (long)c.N * c.K, nc = (long)c.M * c.N;
    int lda = c.ta ? c.M : c.K, ldb = c.tb ? c.K : c.N, ldc = c.N;
    bf16 *A = dalloc<bf16>(na), *B = dalloc<bf16>(nb), *C = dalloc<bf16>(nc);
    float *Af = dalloc<float>(na), *Bf = dalloc<float>(nb), *Cf = dalloc<float>(nc);
    fill_bf16(A, na, rng, 1.0f);
    fill_bf16(B, nb, rng, 0.5f);
    fill_bf16(C, nc, rng, 1.0f);
    for (long i = 0; i < na; i++) Af[i] = widen(A[i]);
    for (long i = 0; i < nb; i++) Bf[i] = widen(B[i]);
    for (long i = 0; i < nc; i++) Cf[i] = widen(C[i]);
    std::vector<uint16_t> c_in(nc);
    for (long i = 0; i < nc; i++) c_in[i] = C[i].x;
    puf_mtl::GemmOpts o;
    o.allow_tensor_ops = c.tensor_ops;
    o.allow_split_k = c.split_k;
    o.allow_sg_tn = c.sg_tn;
    o.allow_small_n = c.small_n;
    puf_mtl::GemmPath p32 = puf_mtl::gemm_rm(0, c.ta, c.tb, c.M, c.N, c.K, c.alpha, Af, lda,
        Bf, ldb, c.beta, Cf, ldc, ws, ws_bytes, &o);
    // NaN-poison the shared workspace so the bf16 reduction can't reuse the
    // FP32 reference's split-K partials.
    cudaDeviceSynchronize();
    if (ws) memset(ws, 0xff, ws_bytes);
    puf_mtl::GemmPath p16 = puf_mtl::gemm_rm(0, c.ta, c.tb, c.M, c.N, c.K, c.alpha, A, lda,
        B, ldb, c.beta, C, ldc, ws, ws_bytes, &o);
    cudaDeviceSynchronize();
    long bad = 0;
    for (long i = 0; i < nc; i++) {
        if (C[i].x != bits(Cf[i])) bad++;
    }
    char msg[160];
    snprintf(msg, sizeof(msg), "gemm %-22s path %d/%d (want %d) mismatches %ld/%ld", c.name,
        (int)p16, (int)p32, (int)c.want, bad, nc);
    check(bad == 0 && p16 == p32 && p16 == c.want, msg);
    cudaFree(A); cudaFree(B); cudaFree(C); cudaFree(Af); cudaFree(Bf); cudaFree(Cf);
}

static void test_gemm_nn_add(int M, int N, int K, std::mt19937& rng) {
    long na = (long)M * K, nb = (long)K * N, nc = (long)M * N;
    bf16 *A = dalloc<bf16>(na), *B = dalloc<bf16>(nb), *C = dalloc<bf16>(nc), *D = dalloc<bf16>(nc);
    float *Af = dalloc<float>(na), *Bf = dalloc<float>(nb), *Cf = dalloc<float>(nc);
    fill_bf16(A, na, rng, 1.0f);
    fill_bf16(B, nb, rng, 0.5f);
    fill_bf16(D, nc, rng, 1.0f);
    for (long i = 0; i < na; i++) Af[i] = widen(A[i]);
    for (long i = 0; i < nb; i++) Bf[i] = widen(B[i]);
    bool ok16 = puf_mtl::gemm_nn_add_rm(0, M, N, K, A, B, D, C);
    puf_mtl::GemmPath p32 = puf_mtl::gemm_rm(0, false, false, M, N, K, 1.0f, Af, K, Bf, N, 0.0f,
        Cf, N);
    cudaDeviceSynchronize();
    long bad = 0;
    for (long i = 0; i < nc; i++) {
        float want = widen(__float2bfloat16(Cf[i])) + widen(D[i]);
        if (C[i].x != bits(want)) bad++;
    }
    char msg[160];
    snprintf(msg, sizeof(msg), "gemm nn_add %dx%dx%d (f32 path %d) mismatches %ld/%ld", M, N, K,
        (int)p32, bad, nc);
    check(ok16 && p32 == puf_mtl::GEMM_TENSOR_OPS && bad == 0, msg);
    cudaFree(A); cudaFree(B); cudaFree(C); cudaFree(D); cudaFree(Af); cudaFree(Bf); cudaFree(Cf);
}

// BF16 small-N GEMM against an independent host oracle (fma chain over k
// ascending on widened inputs, alpha/beta epilogue, one bf16 rounding) and
// against the forced bf16 tiled kernel, bitwise. Outputs start NaN.
static void test_gemm_small_n(std::mt19937& rng) {
    // pad > 0: lda % 4 != 0 takes the kernel's scalar A loads.
    struct C { int M, N, K; bool tb; float alpha, beta; int pad; };
    const C cases[] = {{16384, 4, 64, true, 1, 0, 0}, {65536, 5, 32, true, 1, 0, 0},
        {1000, 8, 256, true, 1, 0, 1}, {517, 7, 48, false, 1.5f, 0.75f, 0},
        {333, 1, 16, true, 0.5f, -1.25f, 3}, {4096, 4, 32, true, 1, 0, 1}};
    for (const C& c : cases) {
        int M = c.M, N = c.N, K = c.K, ldb = c.tb ? K : N, lda = K + c.pad;
        long na = (long)M * lda, nb = (long)N * K, nc = (long)M * N;
        bf16 *A = dalloc<bf16>(na), *B = dalloc<bf16>(nb), *Cs = dalloc<bf16>(nc), *Ct = dalloc<bf16>(nc);
        fill_bf16(A, na, rng, 1.0f);
        fill_bf16(B, nb, rng, 0.5f);
        std::vector<uint16_t> c0(nc);
        for (long i = 0; i < nc; i++) {
            c0[i] = c.beta == 0.0f ? 0x7fc1 : bits(std::normal_distribution<float>(0, 1)(rng));
            Cs[i].x = Ct[i].x = c0[i];
        }
        puf_mtl::GemmOpts tiled;
        tiled.allow_tensor_ops = tiled.allow_split_k = tiled.allow_sg_tn = tiled.allow_small_n = false;
        puf_mtl::GemmPath ps = puf_mtl::gemm_rm(0, false, c.tb, M, N, K, c.alpha, A, lda, B, ldb,
            c.beta, Cs, N);
        puf_mtl::GemmPath pt = puf_mtl::gemm_rm(0, false, c.tb, M, N, K, c.alpha, A, lda, B, ldb,
            c.beta, Ct, N, nullptr, 0, &tiled);
        cudaDeviceSynchronize();
        long bad_ref = 0, bad_tiled = 0;
        for (int m = 0; m < M; m++)
            for (int n = 0; n < N; n++) {
                float acc = 0.0f;
                for (int k = 0; k < K; k++) {
                    float b = widen(c.tb ? B[(long)n * K + k] : B[(long)k * N + n]);
                    acc = std::fma(widen(A[(long)m * lda + k]), b, acc);
                }
                long ci = (long)m * N + n;
                float v = c.alpha * acc;
                if (c.beta != 0.0f) v = std::fma(c.beta, puf_bf16_bits_to_float(c0[ci]), v);
                bad_ref += Cs[ci].x != bits(v);
                bad_tiled += Cs[ci].x != Ct[ci].x;
            }
        char msg[160];
        snprintf(msg, sizeof(msg), "gemm small_n oracle %dx%dx%d %c lda %d path %d/%d mismatches ref %ld tiled %ld",
            M, N, K, c.tb ? 'T' : 'N', lda, (int)ps, (int)pt, bad_ref, bad_tiled);
        check(ps == puf_mtl::GEMM_SMALL_N && pt == puf_mtl::GEMM_TILED && bad_ref == 0
            && bad_tiled == 0, msg);
        cudaFree(A); cudaFree(B); cudaFree(Cs); cudaFree(Ct);
    }
}

// BF16 gemm_rm_deferred + gemm_finish == gemm_rm bit for bit, partials past
// ws_off, C untouched until the finish (split paths).
static void test_gemm_deferred(std::mt19937& rng) {
    struct C { const char* tag; int M, N, K; bool sg, tops; float alpha, beta;
        puf_mtl::GemmPath want; };
    const C cases[] = {
        {"tops_tn", 192, 64, 16384, true, true, 1, 0, puf_mtl::GEMM_TOPS_TN_SPLIT},
        {"sg_tn", 7, 64, 16384, true, true, 1, 0, puf_mtl::GEMM_SG_TN},
        {"split_k", 96, 32, 8192, false, false, 0.5f, 0.25f, puf_mtl::GEMM_SPLIT_K},
    };
    size_t ws_bytes = 4 << 20, off = 1 << 20;
    char* ws = dalloc<char>((long)ws_bytes);
    for (const C& c : cases) {
        long na = (long)c.K * c.M, nb = (long)c.K * c.N, nc = (long)c.M * c.N;
        bf16 *A = dalloc<bf16>(na), *B = dalloc<bf16>(nb), *Cd = dalloc<bf16>(nc), *Cr = dalloc<bf16>(nc);
        fill_bf16(A, na, rng, 1.0f);
        fill_bf16(B, nb, rng, 0.5f);
        std::vector<uint16_t> c0(nc);
        for (long i = 0; i < nc; i++) {
            c0[i] = c.beta == 0.0f ? 0x7fc1 : bits(std::normal_distribution<float>(0, 1)(rng));
            Cd[i].x = Cr[i].x = c0[i];
        }
        puf_mtl::GemmOpts o;
        o.allow_sg_tn = c.sg;
        o.allow_tensor_ops = c.tops;
        memset(ws, 0xff, ws_bytes);
        puf_mtl::GemmPath pr = puf_mtl::gemm_rm(0, true, false, c.M, c.N, c.K, c.alpha, A, c.M,
            B, c.N, c.beta, Cr, c.N, ws, ws_bytes, &o);
        cudaDeviceSynchronize();
        memset(ws, 0xff, ws_bytes);
        puf_mtl::GemmDeferred d;
        puf_mtl::GemmPath p = puf_mtl::gemm_rm_deferred(0, true, false, c.M, c.N, c.K, c.alpha,
            A, c.M, B, c.N, c.beta, Cd, c.N, ws, ws_bytes, off, &d, &o);
        cudaDeviceSynchronize();
        long touched = 0, below = 0;
        for (long i = 0; i < nc; i++) touched += Cd[i].x != c0[i];
        for (size_t i = 0; i < off; i++) below += (unsigned char)ws[i] != 0xff;
        bool pend = d.pipe != nullptr;
        puf_mtl::gemm_finish(0, &d);
        cudaDeviceSynchronize();
        long bad = 0;
        for (long i = 0; i < nc; i++) bad += Cd[i].x != Cr[i].x;
        char msg[160];
        snprintf(msg, sizeof(msg), "gemm deferred %s path %d/%d pending %d touched %ld below_off %ld "
            "mismatch %ld", c.tag, (int)p, (int)pr, pend, touched, below, bad);
        check(p == c.want && pr == c.want && pend && touched == 0 && below == 0 && bad == 0, msg);
        cudaFree(A); cudaFree(B); cudaFree(Cd); cudaFree(Cr);
    }
    cudaFree(ws);
}

static void test_gemm() {
    std::mt19937 rng(11);
    size_t ws_bytes = 4 << 20;
    void* ws = dalloc<char>((long)ws_bytes);
    using namespace puf_mtl;
    // Breakout: minibatch rows B*T = 8192 (train), 64-row rollout blocks,
    // hidden 32 (3H = 96), encoder K = obs size, decoder N = actions + 1.
    const GemmCase cases[] = {
        {"nt enc fwd", false, true, 8192, 32, 128, 1, 0, GEMM_TENSOR_OPS, true, true, true},
        {"nt mingru fwd", false, true, 8192, 96, 32, 1, 0, GEMM_TENSOR_OPS, true, true, true},
        {"nt ragged K", false, true, 512, 32, 118, 1, 0, GEMM_TENSOR_OPS, true, true, true},
        {"nn dX", false, false, 8192, 32, 96, 1, 0, GEMM_TENSOR_OPS, true, true, true},
        {"tn tensor", true, false, 64, 32, 1024, 1, 0, GEMM_TENSOR_OPS, true, false, true},
        {"nt dec (small_n)", false, true, 8192, 4, 32, 1, 0, GEMM_SMALL_N, true, true, true},
        {"nt dec h64 (small_n)", false, true, 16384, 4, 64, 1, 0, GEMM_SMALL_N, true, true, true},
        {"nn small_n alpha/beta", false, false, 517, 7, 48, 1.5f, 0.75f, GEMM_SMALL_N, true, true, true},
        {"nt dec forced tiled", false, true, 8192, 4, 32, 1, 0, GEMM_TILED, true, true, true, false},
        {"nt rollout (tiled)", false, true, 1000, 96, 32, 1, 0, GEMM_TILED, true, true, true},
        {"nn forced tiled", false, false, 512, 32, 96, 1, 0, GEMM_TILED, false, true, true},
        {"tn sg dW", true, false, 96, 32, 8192, 1, 0, GEMM_SG_TN, true, true, true},
        {"tn sg enc dW", true, false, 32, 118, 8192, 1, 0, GEMM_SG_TN, true, true, true},
        {"tn sg dec dW ragged", true, false, 4, 32, 8193, 1, 0, GEMM_SG_TN, true, true, true},
        {"tn tops split dW h64", true, false, 192, 64, 16384, 1, 0, GEMM_TOPS_TN_SPLIT, true, true, true},
        {"tn tops split ragged K", true, false, 128, 96, 5000, 0.5f, 0.25f, GEMM_TOPS_TN_SPLIT, true, true, true},
        {"tn tops split off (sg)", true, false, 192, 64, 16384, 1, 0, GEMM_SG_TN, false, true, true},
        {"tn split-K tiled", true, false, 96, 32, 8192, 1, 0, GEMM_SPLIT_K, true, true, false},
        {"tn no split-K (tiled)", true, false, 96, 32, 4096, 1, 0, GEMM_TILED, true, false, true},
        {"nn muon alpha/beta", false, false, 32, 32, 32, 2.0315f, -4.0848f, GEMM_TILED, true, true, true},
        {"tn muon gram", true, false, 32, 32, 96, 1, 0, GEMM_TILED, true, true, true},
        {"nt muon gram", false, true, 32, 32, 128, 1, 0, GEMM_TILED, true, true, true},
        {"tn split-K beta", true, false, 96, 32, 8192, 0.5f, 0.25f, GEMM_SPLIT_K, false, true, false},
        {"scale alpha=0", false, false, 32, 32, 32, 0.0f, 0.5f, GEMM_SCALE, true, true, true},
        // bf16 sg_tn loads: ragged M/N tiles, odd leading dims (2-byte aligned rows).
        {"tn sg ragged MxN", true, false, 37, 23, 8195, 1, 0, GEMM_SG_TN, true, true, true},
    };
    for (const GemmCase& c : cases) test_gemm_case(c, rng, ws, ws_bytes);
    // No workspace: one sg_tn slice writing C directly (alpha/beta epilogue).
    const GemmCase direct[] = {
        {"tn sg direct alpha/beta", true, false, 37, 23, 4099, 1.5f, 0.75f, GEMM_SG_TN, true, true, true},
        {"tn sg direct dW", true, false, 96, 32, 8192, 1, 0, GEMM_SG_TN, true, true, true},
    };
    for (const GemmCase& c : direct) test_gemm_case(c, rng, nullptr, 0);
    test_gemm_nn_add(8192, 32, 96, rng);
    test_gemm_nn_add(512, 64, 32, rng);
    test_gemm_small_n(rng);
    test_gemm_deferred(rng);
    cudaFree(ws);
}

// ---------------------------------------------------------------------------
// 3. Muon Newton-Schulz
// ---------------------------------------------------------------------------
static const double ns_coeffs[5][3] = {
    {4.0848, -6.8946, 2.9270},
    {3.9505, -6.3029, 2.6377},
    {3.7418, -5.5913, 2.3037},
    {2.8769, -3.1427, 1.2046},
    {2.8366, -3.0525, 1.2012},
};
struct SumArgs { uint64_t dst, src; int32_t n; };
struct NormArgs { uint64_t dst, sum_sq; float eps; int32_t n; };
struct StoreArgs { uint64_t dst, src; float scale; int32_t n; };

// muon_step's per-matrix tail with the generic kernels (algo.cu loop body).
static void muon_unfused(bf16* x, int R, int C, float scale, float* partials, float* norm,
        bf16* x_buf, bf16* gram, bf16* gram_buf) {
    using namespace puf_mtl;
    long ne = (long)R * C;
    int nblk = (int)std::min<long>((ne + 255) / 256, 256);
    dispatch(0, pipeline("muon_sum_sq_partials"), dim3(nblk), dim3(256),
        SumArgs{gpu_addr(partials), gpu_addr(x), (int32_t)ne});
    dispatch(0, pipeline("muon_sum_sq_reduce"), dim3(1), dim3(256),
        SumArgs{gpu_addr(norm), gpu_addr(partials), nblk});
    dispatch(0, pipeline("muon_l2_normalize"), dim3((unsigned)((ne + 255) / 256)), dim3(256),
        NormArgs{gpu_addr(x), gpu_addr(norm), 1e-7f, (int32_t)ne});
    bool tall = R > C;
    int M = tall ? C : R;
    for (int i = 0; i < 5; i++) {
        bf16* src = (i % 2 == 0) ? x : x_buf;
        bf16* dst = (i % 2 == 0) ? x_buf : x;
        // puf_mm_tn(src, src, gram) / puf_mm(src, src, gram)
        if (tall) gemm_rm(0, true, false, M, M, R, 1.0f, src, C, src, C, 0.0f, gram, M);
        else gemm_rm(0, false, true, M, M, C, 1.0f, src, C, src, C, 0.0f, gram, M);
        cudaMemcpyAsync(gram_buf, gram, (size_t)M * M * 2, cudaMemcpyDeviceToDevice, 0);
        gemm_rm(0, false, false, M, M, M, (float)ns_coeffs[i][2], gram, M, gram, M,
            (float)ns_coeffs[i][1], gram_buf, M);
        cudaMemcpyAsync(dst, src, (size_t)ne * 2, cudaMemcpyDeviceToDevice, 0);
        if (tall) gemm_rm(0, false, false, R, C, C, 1.0f, src, C, gram_buf, C,
            (float)ns_coeffs[i][0], dst, C);
        else gemm_rm(0, false, false, R, C, R, 1.0f, gram_buf, R, src, C,
            (float)ns_coeffs[i][0], dst, C);
    }
    dispatch(0, pipeline("muon_store_update"), dim3((unsigned)((ne + 255) / 256)), dim3(256),
        StoreArgs{gpu_addr(x), gpu_addr(x_buf), scale, (int32_t)ne});
}

static void test_muon() {
    std::mt19937 rng(5);
    // Multiples of 8 take the simdgroup-fragment stages (tall, wide, square).
    const int shapes[][2] = {{32, 118}, {96, 32}, {4, 32}, {32, 32}, {7, 13}, {8, 64},
        {40, 24}, {32, 128}};
    const int ns = sizeof(shapes) / sizeof(shapes[0]);
    float flat[15];
    for (int i = 0; i < 15; i++) flat[i] = (float)ns_coeffs[i / 3][i % 3];
    float* partials = dalloc<float>(256);
    float* norm = dalloc<float>(1);
    bf16 *x_buf = dalloc<bf16>(4096), *gram = dalloc<bf16>(1024), *gram_buf = dalloc<bf16>(1024);
    std::vector<bf16*> a(ns), b(ns), c(ns);
    puf_mtl::MuonNSMatrix mats[8];
    for (int s = 0; s < ns; s++) {
        int R = shapes[s][0], C = shapes[s][1];
        long ne = (long)R * C;
        a[s] = dalloc<bf16>(ne); b[s] = dalloc<bf16>(ne); c[s] = dalloc<bf16>(ne);
        fill_bf16(a[s], ne, rng, 0.01f);
        memcpy(b[s], a[s], ne * 2);
        memcpy(c[s], a[s], ne * 2);
        float scale = sqrtf(fmaxf(1.0f, (float)R / (float)C));
        mats[s] = {b[s], R, C, scale};
    }
    // Unfused reference, one matrix at a time.
    for (int s = 0; s < ns; s++) {
        int R = shapes[s][0], C = shapes[s][1];
        muon_unfused(a[s], R, C, sqrtf(fmaxf(1.0f, (float)R / (float)C)), partials, norm,
            x_buf, gram, gram_buf);
    }
    bool batched = puf_mtl::muon_ns5_batch_rm(0, mats, ns, flat, 1e-7f);
    cudaDeviceSynchronize();
    long bad = 0, total = 0;
    for (int s = 0; s < ns; s++) {
        long ne = (long)shapes[s][0] * shapes[s][1];
        for (long i = 0; i < ne; i++) bad += a[s][i].x != b[s][i].x;
        total += ne;
    }
    char msg[160];
    snprintf(msg, sizeof(msg), "muon_ns5_batch vs unfused bf16 sequence (%ld/%ld mismatches)",
        bad, total);
    check(batched && bad == 0, msg);

    // muon_ns5 (single matrix, after the generic normalize): compare x_buf.
    long bad1 = 0, total1 = 0;
    for (int s = 0; s < ns; s++) {
        int R = shapes[s][0], C = shapes[s][1];
        long ne = (long)R * C;
        bf16* ref = c[s];
        std::vector<uint16_t> orig(ne);
        for (long i = 0; i < ne; i++) orig[i] = ref[i].x;
        // reference: unfused NS only (no normalize / store) via the full
        // sequence minus store: run normalize + 5 steps, read x_buf.
        using namespace puf_mtl;
        int nblk = (int)std::min<long>((ne + 255) / 256, 256);
        dispatch(0, pipeline("muon_sum_sq_partials"), dim3(nblk), dim3(256),
            SumArgs{gpu_addr(partials), gpu_addr(ref), (int32_t)ne});
        dispatch(0, pipeline("muon_sum_sq_reduce"), dim3(1), dim3(256),
            SumArgs{gpu_addr(norm), gpu_addr(partials), nblk});
        dispatch(0, pipeline("muon_l2_normalize"), dim3((unsigned)((ne + 255) / 256)), dim3(256),
            NormArgs{gpu_addr(ref), gpu_addr(norm), 1e-7f, (int32_t)ne});
        cudaDeviceSynchronize();
        bf16* fx = dalloc<bf16>(ne);
        memcpy(fx, ref, ne * 2);
        bf16 *fxb = dalloc<bf16>(ne), *fg = dalloc<bf16>(1024), *fgb = dalloc<bf16>(1024);
        bool ok = muon_ns5_rm(0, fx, fxb, fg, fgb, R, C, flat);
        // unfused 5 steps on ref
        bool tall = R > C;
        int M = tall ? C : R;
        for (int i = 0; i < 5; i++) {
            bf16* src = (i % 2 == 0) ? ref : x_buf;
            bf16* dst = (i % 2 == 0) ? x_buf : ref;
            if (tall) gemm_rm(0, true, false, M, M, R, 1.0f, src, C, src, C, 0.0f, gram, M);
            else gemm_rm(0, false, true, M, M, C, 1.0f, src, C, src, C, 0.0f, gram, M);
            cudaMemcpyAsync(gram_buf, gram, (size_t)M * M * 2, cudaMemcpyDeviceToDevice, 0);
            gemm_rm(0, false, false, M, M, M, flat[3 * i + 2], gram, M, gram, M, flat[3 * i + 1],
                gram_buf, M);
            cudaMemcpyAsync(dst, src, (size_t)ne * 2, cudaMemcpyDeviceToDevice, 0);
            if (tall) gemm_rm(0, false, false, R, C, C, 1.0f, src, C, gram_buf, C, flat[3 * i],
                dst, C);
            else gemm_rm(0, false, false, R, C, R, 1.0f, gram_buf, R, src, C, flat[3 * i], dst, C);
        }
        cudaDeviceSynchronize();
        if (!ok) bad1 += ne;
        for (long i = 0; i < ne; i++) {
            bad1 += fxb[i].x != x_buf[i].x;   // step 4 result
            bad1 += fx[i].x != ref[i].x;      // step 3 result
        }
        for (long i = 0; i < (long)M * M; i++) {
            bad1 += fg[i].x != gram[i].x;
            bad1 += fgb[i].x != gram_buf[i].x;
        }
        total1 += 2 * ne + 2L * M * M;
        cudaFree(fx); cudaFree(fxb); cudaFree(fg); cudaFree(fgb);
    }
    snprintf(msg, sizeof(msg), "muon_ns5 vs unfused bf16 NS steps (%ld/%ld mismatches)", bad1,
        total1);
    check(bad1 == 0, msg);
    for (int s = 0; s < ns; s++) { cudaFree(a[s]); cudaFree(b[s]); cudaFree(c[s]); }
    cudaFree(partials); cudaFree(norm); cudaFree(x_buf); cudaFree(gram); cudaFree(gram_buf);
}

// ---------------------------------------------------------------------------
// 4. puff_advantage
// ---------------------------------------------------------------------------
struct AdvArgs {
    uint64_t values, rewards, dones, importance, advantages, returns;
    float gamma, lambda, rho_clip, c_clip;
    int32_t num_steps, horizon;
};

// muon_ns_wide_rm (matrices muon_ns5_batch rejects) vs the unfused bf16
// sequence: tensor_ops grams (tall / wide / square; x 16-byte aligned or
// not), tiled grams with K % 16 != 0, a 2047-row matrix.
static void test_muon_wide() {
    std::mt19937 rng(11);
    const int shapes[][3] = {{192, 64, 0}, {64, 192, 0}, {128, 128, 0}, {100, 40, 0},
        {40, 100, 1}, {2047, 3, 0}, {192, 64, 1}, {33, 64, 0}};
    const int ns = sizeof(shapes) / sizeof(shapes[0]);
    float flat[15];
    for (int i = 0; i < 15; i++) flat[i] = (float)ns_coeffs[i / 3][i % 3];
    float* partials = dalloc<float>(256);
    float* norm = dalloc<float>(1);
    bf16 *x_buf = dalloc<bf16>(2047 * 192), *gram = dalloc<bf16>(192 * 192),
        *gram_buf = dalloc<bf16>(192 * 192);
    std::vector<bf16*> a(ns), b(ns);
    puf_mtl::MuonNSMatrix mats[8];
    size_t ws_bytes = 0;
    for (int s = 0; s < ns; s++) {
        int R = shapes[s][0], C = shapes[s][1], off = shapes[s][2];
        long ne = (long)R * C;
        a[s] = dalloc<bf16>(ne + 1) + off; b[s] = dalloc<bf16>(ne + 1) + off;
        fill_bf16(a[s], ne, rng, 0.01f);
        memcpy(b[s], a[s], ne * 2);
        mats[s] = {b[s], R, C, sqrtf(fmaxf(1.0f, (float)R / (float)C))};
        ws_bytes += puf_mtl::muon_ns_wide_scratch_bytes(R, C);
    }
    void* ws = dalloc<unsigned char>((long)ws_bytes);
    for (int s = 0; s < ns; s++) {
        int R = shapes[s][0], C = shapes[s][1];
        muon_unfused(a[s], R, C, sqrtf(fmaxf(1.0f, (float)R / (float)C)), partials, norm,
            x_buf, gram, gram_buf);
    }
    bool ran = puf_mtl::muon_ns_wide_rm(0, mats, ns, flat, 1e-7f, ws, ws_bytes);
    cudaDeviceSynchronize();
    long bad = 0, total = 0;
    for (int s = 0; s < ns; s++) {
        long ne = (long)shapes[s][0] * shapes[s][1];
        for (long i = 0; i < ne; i++) bad += a[s][i].x != b[s][i].x;
        total += ne;
    }
    char msg[160];
    snprintf(msg, sizeof(msg), "muon_ns_wide vs unfused bf16 sequence (%ld/%ld mismatches)",
        bad, total);
    check(ran && bad == 0, msg);
}

static void test_advantage() {
    const int rows = 1000, H = 64;
    long n = (long)rows * H;
    std::mt19937 rng(3);
    bf16 *v = dalloc<bf16>(n), *r = dalloc<bf16>(n), *d = dalloc<bf16>(n), *imp = dalloc<bf16>(n);
    bf16 *adv = dalloc<bf16>(n), *ret = dalloc<bf16>(n);
    std::uniform_real_distribution<float> u(0.0f, 1.0f);
    for (long i = 0; i < n; i++) {
        v[i] = __float2bfloat16(u(rng) * 4 - 2);
        r[i] = __float2bfloat16(u(rng) < 0.1f ? 1.0f : 0.0f);
        d[i] = __float2bfloat16(u(rng) < 0.02f ? 1.0f : 0.0f);
        imp[i] = __float2bfloat16(0.5f + u(rng));
    }
    float gamma = 0.995f, lambda = 0.9f, rho = 1.0f, cc = 1.0f;
    puf_mtl::dispatch(0, puf_mtl::pipeline("puff_advantage"), dim3((rows + 255) / 256), dim3(256),
        AdvArgs{puf_mtl::gpu_addr(v), puf_mtl::gpu_addr(r), puf_mtl::gpu_addr(d),
            puf_mtl::gpu_addr(imp), puf_mtl::gpu_addr(adv), puf_mtl::gpu_addr(ret),
            gamma, lambda, rho, cc, rows, H});
    cudaDeviceSynchronize();
    long bad = 0;
    for (int row = 0; row < rows; row++) {
        long off = (long)row * H;
        float lastlam = 0.0f;
        float nv = widen(v[off + H - 1]), nd = widen(d[off + H - 1]), nr = widen(r[off + H - 1]);
        std::vector<float> a(H, 0.0f);
        for (int t = H - 2; t >= 0; t--) {
            float vi = widen(v[off + t]), ii = widen(imp[off + t]);
            float nnt = 1.0f - nd;
            float rh = fminf(ii, rho), c = fminf(ii, cc);
            float delta_inner = std::fma(gamma * nv, nnt, nr) - vi;
            float trace = gamma * lambda * c * lastlam * nnt;
            lastlam = std::fma(rh, delta_inner, trace);
            a[t] = lastlam;
            nv = vi; nd = widen(d[off + t]); nr = widen(r[off + t]);
        }
        for (int t = 0; t < H; t++) {
            bad += adv[off + t].x != bits(a[t]);
            bad += ret[off + t].x != bits(widen(v[off + t]) + a[t]);
        }
    }
    char msg[160];
    snprintf(msg, sizeof(msg), "puff_advantage bf16 vs host reference (%ld/%ld mismatches)", bad,
        2 * n);
    check(bad == 0, msg);
    cudaFree(v); cudaFree(r); cudaFree(d); cudaFree(imp); cudaFree(adv); cudaFree(ret);
}

int main() {
    @autoreleasepool {
        puf_mtl::init();
        test_casts();
        test_gemm();
        test_muon();
        test_muon_wide();
        test_advantage();
        printf("%s (%d failures)\n", g_fail ? "FAIL" : "PASS", g_fail);
        return g_fail ? 1 : 0;
    }
}
