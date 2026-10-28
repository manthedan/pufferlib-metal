// GEMM correctness test for the Metal backend's cuBLAS shim.
//
// 1. Golden: every gemm_* case in a CUDA FP32 golden dump (manifest.json +
//    raw little-endian .bin, produced by the real 5.0 puf_mm/puf_mm_tn/
//    puf_mm_nn wrappers on cuBLAS) is replayed through the same
//    cublasGemmExDense mapping algo.cu uses, and compared to the CUDA output.
// 2. Synthetic: every transpose combination at the real Breakout rollout/train
//    and Muon shapes (hidden 32 and 128), each with every applicable kernel path
//    forced (tensor_ops / tiled / split-K), plus alpha/beta, beta==0-with-NaN-C,
//    and strided (non-dense) leading dimensions, compared to Accelerate
//    cblas_sgemm. The unforced variant of every case is issued through
//    cublasGemmEx (column-major mapping), the forced ones through gemm_rm.
//    Fails if any kernel path was never exercised.
//
// Errors: max_abs = max|x-ref|; max_rel = max |x-ref| / max(|x|,|ref|) over
// elements (golden.py definition; blows up on cancellation near zero);
// norm = max_abs / max|ref| (pass criterion: norm <= 1e-4 for golden/tiled).
//
// Build/run: tests/metal/run_test_gemm.sh [golden_run_dir]
#import <Foundation/Foundation.h>
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cstring>
#include <random>
#include <chrono>
#include <string>
#include <vector>

#include "compat/cublas_v2.h"
#include "gemm.h"
#include "runtime.h"

static const char* path_name(puf_mtl::GemmPath p) {
    switch (p) {
        case puf_mtl::GEMM_TENSOR_OPS: return "tensor_ops";
        case puf_mtl::GEMM_TILED: return "tiled";
        case puf_mtl::GEMM_SPLIT_K: return "split_k";
        case puf_mtl::GEMM_SCALE: return "scale";
        case puf_mtl::GEMM_SG_TN: return "sg_tn";
        case puf_mtl::GEMM_TOPS_TN_SPLIT: return "tops_tn_sk";
        case puf_mtl::GEMM_SMALL_N: return "small_n";
        case puf_mtl::GEMM_WIDE_TN: return "wide_tn";
        default: return "none";
    }
}

struct Err {
    double max_abs = 0, max_rel = 0, max_ref = 0;
    long nonfinite = 0;
    double norm() const { return max_ref > 0 ? max_abs / max_ref : max_abs; }
};

static Err compare(const float* x, const float* ref, long n) {
    Err e;
    for (long i = 0; i < n; i++) {
        double a = x[i], b = ref[i];
        if (!std::isfinite(a) || !std::isfinite(b)) {
            if (!(std::isnan(a) && std::isnan(b)) && a != b) e.nonfinite++;
            continue;
        }
        double d = fabs(a - b);
        double den = fmax(fmax(fabs(a), fabs(b)), 2.2250738585072014e-308);
        e.max_abs = fmax(e.max_abs, d);
        e.max_rel = fmax(e.max_rel, d / den);
        e.max_ref = fmax(e.max_ref, fabs(b));
    }
    return e;
}

static float* dalloc(long n) {
    float* p = nullptr;
    if (cudaMalloc(&p, (size_t)n * sizeof(float)) != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed\n");
        exit(2);
    }
    return p;
}

static int g_fail = 0;
static long g_path_count[puf_mtl::GEMM_WIDE_TN + 1];  // per GemmPath, over the synthetic sweep
static cublasHandle_t g_handle;   // for replaying synthetic cases via cublasGemmEx

// ---------------------------------------------------------------------------
// Golden replay
// ---------------------------------------------------------------------------

static std::vector<float> read_bin(NSString* dir, NSDictionary* spec) {
    NSString* file = [dir stringByAppendingPathComponent:spec[@"file"]];
    NSData* d = [NSData dataWithContentsOfFile:file];
    if (!d) {
        fprintf(stderr, "cannot read %s\n", file.UTF8String);
        exit(2);
    }
    std::vector<float> v(d.length / sizeof(float));
    memcpy(v.data(), d.bytes, d.length);
    return v;
}

// Mirrors algo.cu cublasGemmExDense + puf_mm / puf_mm_tn / puf_mm_nn.
static void dense_gemm(cublasHandle_t h, cublasOperation_t op_a, cublasOperation_t op_b,
        int M, int N, int K, const float* A, const float* B, float* C,
        float alpha, float beta) {
    int lda = (op_a == CUBLAS_OP_N) ? K : M;
    int ldb = (op_b == CUBLAS_OP_N) ? N : K;
    cublasGemmEx(h, op_b, op_a, N, M, K, &alpha, B, CUDA_R_32F, ldb, A, CUDA_R_32F, lda,
        &beta, C, CUDA_R_32F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
}

static void run_golden(const char* run_dir, cublasHandle_t h) {
    NSString* root = [NSString stringWithUTF8String:run_dir];
    NSArray* names = [[[NSFileManager defaultManager] contentsOfDirectoryAtPath:root error:nil]
        sortedArrayUsingSelector:@selector(compare:)];
    printf("== golden (%s)\n", run_dir);
    printf("%-40s %-10s %6s %6s %6s %5s %5s %-10s %10s %10s %10s %10s\n", "case", "op", "M", "N",
        "K", "alpha", "beta", "path", "max_abs", "max_rel", "norm", "accel_norm");
    int count = 0;
    for (NSString* name in names) {
        if (![name hasPrefix:@"gemm_"]) continue;
        NSString* dir = [root stringByAppendingPathComponent:name];
        NSData* mj = [NSData dataWithContentsOfFile:[dir stringByAppendingPathComponent:@"manifest.json"]];
        NSDictionary* man = [NSJSONSerialization JSONObjectWithData:mj options:0 error:nil];
        NSDictionary* t = man[@"tensors"];
        NSDictionary* sc = man[@"scalars"];
        NSString* op = sc[@"operation"];
        int M = [sc[@"M"] intValue], N = [sc[@"N"] intValue], K = [sc[@"K"] intValue];
        float alpha = [sc[@"alpha"] floatValue], beta = [sc[@"beta"] floatValue];
        std::vector<float> a = read_bin(dir, t[@"a"]), b = read_bin(dir, t[@"b"]);
        std::vector<float> c_in = read_bin(dir, t[@"c_in"]), c_out = read_bin(dir, t[@"c_out"]);
        cublasOperation_t op_a = CUBLAS_OP_N, op_b = CUBLAS_OP_N;
        if ([op isEqualToString:@"puf_mm"]) op_b = CUBLAS_OP_T;
        else if ([op isEqualToString:@"puf_mm_tn"]) op_a = CUBLAS_OP_T;
        else if (![op isEqualToString:@"puf_mm_nn"]) {
            fprintf(stderr, "unknown op %s\n", op.UTF8String);
            exit(2);
        }
        long na = (long)a.size(), nb = (long)b.size(), nc = (long)M * N;
        if ((long)c_out.size() != nc || na != (long)M * K || nb != (long)N * K) {
            fprintf(stderr, "%s: size mismatch\n", name.UTF8String);
            exit(2);
        }
        float *dA = dalloc(na), *dB = dalloc(nb), *dC = dalloc(nc);
        memcpy(dA, a.data(), na * 4);
        memcpy(dB, b.data(), nb * 4);
        memcpy(dC, c_in.data(), nc * 4);
        dense_gemm(h, op_a, op_b, M, N, K, dA, dB, dC, alpha, beta);
        puf_mtl::GemmPath path = puf_mtl::gemm_last_path();
        cudaDeviceSynchronize();
        Err e = compare(dC, c_out.data(), nc);

        // Accelerate on the same inputs, for context.
        std::vector<float> ref = c_in;
        int lda = (op_a == CUBLAS_OP_N) ? K : M, ldb = (op_b == CUBLAS_OP_N) ? N : K;
        cblas_sgemm(CblasRowMajor, op_a == CUBLAS_OP_N ? CblasNoTrans : CblasTrans,
            op_b == CUBLAS_OP_N ? CblasNoTrans : CblasTrans, M, N, K, alpha, a.data(), lda,
            b.data(), ldb, beta, ref.data(), N);
        Err ea = compare(ref.data(), c_out.data(), nc);

        bool ok = e.nonfinite == 0 && e.norm() <= 1e-4;
        if (!ok) g_fail++;
        printf("%-40s %-10s %6d %6d %6d %5.2g %5.2g %-10s %10.3e %10.3e %10.3e %10.3e%s\n",
            name.UTF8String + 5, op.UTF8String, M, N, K, alpha, beta, path_name(path),
            e.max_abs, e.max_rel, e.norm(), ea.norm(), ok ? "" : "  FAIL");
        cudaFree(dA);
        cudaFree(dB);
        cudaFree(dC);
        count++;
    }
    printf("golden cases: %d\n", count);
    if (count == 0) {
        fprintf(stderr, "no gemm_* cases in %s (expected a run dir such as golden/run_a)\n",
            run_dir);
        g_fail++;
    }
}

// ---------------------------------------------------------------------------
// Synthetic sweep vs Accelerate
// ---------------------------------------------------------------------------

struct Shape { const char* tag; int M, N, K; };

static void run_synth_case(cudaStream_t s, void* ws, size_t ws_bytes, const Shape& sh,
        bool ta, bool tb, float alpha, float beta, int pad, std::mt19937& rng) {
    int M = sh.M, N = sh.N, K = sh.K;
    int lda = std::max(1, (ta ? M : K) + pad), ldb = std::max(1, (tb ? K : N) + pad);
    int ldc = N + pad;
    long na = (long)(ta ? K : M) * lda, nb = (long)(tb ? N : K) * ldb, nc = (long)M * ldc;
    std::uniform_real_distribution<float> U(-1.0f, 1.0f);
    std::vector<float> A(na), B(nb), C0(nc);
    for (auto& v : A) v = U(rng);
    for (auto& v : B) v = U(rng);
    for (auto& v : C0) v = beta == 0.0f ? NAN : U(rng);  // beta==0 must not read C
    std::vector<float> ref(nc);
    for (long i = 0; i < nc; i++) ref[i] = beta == 0.0f ? 0.0f : C0[i];
    cblas_sgemm(CblasRowMajor, ta ? CblasTrans : CblasNoTrans, tb ? CblasTrans : CblasNoTrans,
        M, N, K, alpha, A.data(), lda, B.data(), ldb, beta, ref.data(), ldc);

    float *dA = dalloc(na), *dB = dalloc(nb), *dC = dalloc(nc);
    memcpy(dA, A.data(), na * 4);
    memcpy(dB, B.data(), nb * 4);

    struct Variant { const char* name; puf_mtl::GemmOpts o; };
    Variant variants[] = {
        {"auto", {true, true, true}},
        {"tiled", {false, false, false, false}},
        {"split_k", {false, true, false, false}},
        {"sg_tn", {false, true, true, false}},
    };
    puf_mtl::GemmPath seen[4];
    for (int v = 0; v < 4; v++) {
        memcpy(dC, C0.data(), nc * 4);
        puf_mtl::GemmPath p;
        if (v == 0) {
            // "auto" goes through the cuBLAS shim with the column-major
            // mapping algo.cu uses: C^T = op(B)^T op(A)^T.
            cublasGemmEx(g_handle, tb ? CUBLAS_OP_T : CUBLAS_OP_N,
                ta ? CUBLAS_OP_T : CUBLAS_OP_N, N, M, K, &alpha, dB, CUDA_R_32F, ldb, dA,
                CUDA_R_32F, lda, &beta, dC, CUDA_R_32F, ldc, CUBLAS_COMPUTE_32F,
                CUBLAS_GEMM_DEFAULT);
            p = puf_mtl::gemm_last_path();
        } else {
            p = puf_mtl::gemm_rm(s, ta, tb, M, N, K, alpha, dA, lda, dB, ldb,
                beta, dC, ldc, ws, ws_bytes, &variants[v].o);
        }
        seen[v] = p;
        g_path_count[p]++;
        // Forcing must take effect: "tiled" never resolves elsewhere.
        if (v == 1 && p != puf_mtl::GEMM_TILED && alpha != 0.0f) {
            printf("%-22s forced tiled resolved to %s  FAIL\n", sh.tag, path_name(p));
            g_fail++;
        }
        // Skip forced variants that resolved to an already-tested path.
        bool dup = false;
        for (int u = 0; u < v; u++) dup |= seen[u] == p;
        cudaStreamSynchronize(s);
        if (dup) continue;
        // Compare only the logical M x N block; padding columns must be untouched.
        Err e;
        long pad_bad = 0;
        for (int m = 0; m < M; m++) {
            Err r = compare(dC + (long)m * ldc, ref.data() + (long)m * ldc, N);
            e.max_abs = fmax(e.max_abs, r.max_abs);
            e.max_rel = fmax(e.max_rel, r.max_rel);
            e.max_ref = fmax(e.max_ref, r.max_ref);
            e.nonfinite += r.nonfinite;
            for (int n = N; n < ldc; n++) {
                float got = dC[(long)m * ldc + n], want = C0[(long)m * ldc + n];
                if (memcmp(&got, &want, 4) != 0) pad_bad++;
            }
        }
        double tol = (p == puf_mtl::GEMM_TENSOR_OPS || p == puf_mtl::GEMM_TOPS_TN_SPLIT)
            ? 1e-3 : 1e-4;
        bool ok = e.nonfinite == 0 && pad_bad == 0 && e.norm() <= tol;
        if (!ok) g_fail++;
        printf("%-22s %c%c %6d %6d %6d %5.2g %5.2g %3d %-10s %10.3e %10.3e %10.3e%s\n", sh.tag,
            ta ? 'T' : 'N', tb ? 'T' : 'N', M, N, K, alpha, beta, pad, path_name(p), e.max_abs,
            e.max_rel, e.norm(), ok ? "" : "  FAIL");
    }
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
}

// alpha == 0: C = beta * C; NaN in A/B must not propagate, and beta == 0 must
// not read (NaN) C.
static void run_alpha_zero(cudaStream_t s, void* ws, size_t ws_bytes) {
    const int M = 37, N = 53, K = 29, ldc = N + 3;
    float *dA = dalloc((long)M * K), *dB = dalloc((long)K * N), *dC = dalloc((long)M * ldc);
    for (long i = 0; i < (long)M * K; i++) dA[i] = (i % 5 == 0) ? NAN : 1.0f;
    for (long i = 0; i < (long)K * N; i++) dB[i] = (i % 7 == 0) ? INFINITY : 1.0f;
    const float betas[] = {0.0f, -1.25f};
    for (float beta : betas) {
        std::vector<float> C0((size_t)M * ldc);
        for (long i = 0; i < (long)C0.size(); i++) C0[i] = beta == 0.0f ? NAN : (float)(i % 11) - 5;
        memcpy(dC, C0.data(), C0.size() * 4);
        puf_mtl::GemmPath p = puf_mtl::gemm_rm(s, false, false, M, N, K, 0.0f, dA, K, dB, N,
            beta, dC, ldc, ws, ws_bytes, nullptr);
        g_path_count[p]++;
        cudaStreamSynchronize(s);
        long bad = 0;
        for (int m = 0; m < M; m++)
            for (int n = 0; n < ldc; n++) {
                float want = n < N ? (beta == 0.0f ? 0.0f : beta * C0[(size_t)m * ldc + n])
                                   : C0[(size_t)m * ldc + n];
                float got = dC[(size_t)m * ldc + n];
                bool ok = n < N ? got == want : memcmp(&got, &want, 4) == 0;
                if (!ok) bad++;
            }
        if (bad) g_fail++;
        printf("%-22s NN %6d %6d %6d %5.2g %5.2g %3d %-10s mismatches=%ld%s\n", "alpha=0 NaN A/B",
            M, N, K, 0.0, beta, ldc - N, path_name(p), bad, bad ? "  FAIL" : "");
    }
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
}

// gemm_nn_add_rm must be bit-identical to the tensor_ops nn GEMM followed by
// C += D (MinGRU backward dX + highway gradient), and must decline shapes the
// tensor_ops path cannot take.
static void run_nn_add(cudaStream_t s, void* ws, size_t ws_bytes) {
    struct { int M, N, K; bool fused; } cases[] = {
        {65536, 32, 96, true},   // Breakout MinGRU backward (H=32)
        {4096, 128, 384, true},  // H=128
        {128, 64, 8, true},
        {96, 32, 96, false},     // M % 64
        {128, 48, 96, false},    // N % 32
    };
    std::mt19937 rng(11);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    for (auto& c : cases) {
        long mk = (long)c.M * c.K, kn = (long)c.K * c.N, mn = (long)c.M * c.N;
        float *dA = dalloc(mk), *dB = dalloc(kn), *dD = dalloc(mn), *dC = dalloc(mn), *dR = dalloc(mn);
        for (long i = 0; i < mk; i++) dA[i] = u(rng);
        for (long i = 0; i < kn; i++) dB[i] = u(rng);
        for (long i = 0; i < mn; i++) { dD[i] = u(rng); dC[i] = NAN; }
        bool fused = puf_mtl::gemm_nn_add_rm(s, c.M, c.N, c.K, dA, dB, dD, dC);
        puf_mtl::GemmPath p = puf_mtl::gemm_rm(s, false, false, c.M, c.N, c.K, 1.0f, dA, c.K,
            dB, c.N, 0.0f, dR, c.N, ws, ws_bytes, nullptr);
        cudaStreamSynchronize(s);
        long bad = 0;
        if (fused) {
            for (long i = 0; i < mn; i++) {
                float want = dR[i] + dD[i];
                if (memcmp(&want, &dC[i], 4)) bad++;
            }
        }
        bool ok = fused == c.fused && (!fused || (p == puf_mtl::GEMM_TENSOR_OPS && bad == 0));
        if (!ok) g_fail++;
        printf("%-22s NN %6d %6d %6d fused=%d ref=%-10s bit mismatches=%ld%s\n", "gemm + add epilogue",
            c.M, c.N, c.K, fused, path_name(p), bad, ok ? "" : "  FAIL");
        cudaFree(dA); cudaFree(dB); cudaFree(dD); cudaFree(dC); cudaFree(dR);
    }
}

// muon_step's Newton-Schulz loop (algo.cu) unfused vs muon_ns5_rm: every
// buffer it leaves behind (x, x_buf, gram, gram_buf) must match bit for bit.
static void run_muon_ns(cudaStream_t s, cublasHandle_t h) {
    static const double ns_coeffs[5][3] = {
        {4.0848, -6.8946, 2.9270}, {3.9505, -6.3029, 2.6377}, {3.7418, -5.5913, 2.3037},
        {2.8769, -3.1427, 1.2046}, {2.8366, -3.0525, 1.2012}};
    float coeffs[15];
    for (int i = 0; i < 15; i++) coeffs[i] = (float)ns_coeffs[i / 3][i % 3];
    struct { int R, C; } cases[] = {{96, 32}, {32, 118}, {4, 32}, {5, 32}, {32, 32},
        {64, 17}, {17, 64}, {1, 5}, {7, 1}, {32, 128}, {2047, 2}, {31, 129}, {4096, 1},
        {8, 8}, {8, 64}, {128, 32}, {40, 24}, {24, 40}, {512, 8},  // fragment path
        {33, 64}, {63, 1000}, {4097, 1}, {128, 64}, {3000, 8}};  // last six ineligible
    std::mt19937 rng(77);
    std::normal_distribution<float> g(0.0f, 1.0f);
    for (auto& c : cases) {
        int R = c.R, C = c.C, M = R < C ? R : C;
        bool tall = R > C;
        long n = (long)R * C;
        float *x0 = dalloc(n), *x = dalloc(n), *xb = dalloc(n), *gr = dalloc((long)M * M),
              *gb = dalloc((long)M * M);
        float *fx = dalloc(n), *fxb = dalloc(n), *fgr = dalloc((long)M * M), *fgb = dalloc((long)M * M);
        double ss = 0;
        for (long i = 0; i < n; i++) { x0[i] = g(rng); ss += (double)x0[i] * x0[i]; }
        for (long i = 0; i < n; i++) x0[i] = (float)(x0[i] / sqrt(ss));
        memcpy(x, x0, n * 4); memcpy(fx, x0, n * 4);
        for (float* p2 : {xb, fxb}) for (long i = 0; i < n; i++) p2[i] = NAN;
        for (float* p2 : {gr, gb, fgr, fgb}) for (long i = 0; i < (long)M * M; i++) p2[i] = NAN;
        auto run_ref = [&] {
            for (int i = 0; i < 5; ++i) {
                float* src = (i % 2 == 0) ? x : xb;
                float* dst = (i % 2 == 0) ? xb : x;
                if (tall) dense_gemm(h, CUBLAS_OP_T, CUBLAS_OP_N, C, C, R, src, src, gr, 1.0f, 0.0f);
                else dense_gemm(h, CUBLAS_OP_N, CUBLAS_OP_T, R, R, C, src, src, gr, 1.0f, 0.0f);
                cudaMemcpyAsync(gb, gr, (size_t)M * M * 4, cudaMemcpyDeviceToDevice, s);
                dense_gemm(h, CUBLAS_OP_N, CUBLAS_OP_N, M, M, M, gr, gr, gb,
                    (float)ns_coeffs[i][2], (float)ns_coeffs[i][1]);
                cudaMemcpyAsync(dst, src, (size_t)n * 4, cudaMemcpyDeviceToDevice, s);
                if (tall) dense_gemm(h, CUBLAS_OP_N, CUBLAS_OP_N, R, C, C, src, gb, dst,
                    1.0f, (float)ns_coeffs[i][0]);
                else dense_gemm(h, CUBLAS_OP_N, CUBLAS_OP_N, R, C, R, gb, src, dst,
                    1.0f, (float)ns_coeffs[i][0]);
            }
        };
        run_ref();
        bool elig = puf_mtl::muon_ns5_eligible(R, C);
        bool ran = puf_mtl::muon_ns5_rm(s, fx, fxb, fgr, fgb, R, C, coeffs);
        cudaStreamSynchronize(s);
        long bad = 0;
        if (ran) {
            bad += memcmp(x, fx, n * 4) != 0;
            bad += memcmp(xb, fxb, n * 4) != 0;
            bad += memcmp(gr, fgr, (size_t)M * M * 4) != 0;
            bad += memcmp(gb, fgb, (size_t)M * M * 4) != 0;
        }
        bool want = M <= 32 && n <= 4096 && (R > C ? R : C) <= 2047;
        bool ok = ran == want && elig == want && bad == 0;
        double t_ref = 0, t_fused = 0;
        if (ran) {
            auto t0 = std::chrono::steady_clock::now();
            for (int it = 0; it < 50; it++) { memcpy(x, x0, n * 4); run_ref(); cudaStreamSynchronize(s); }
            auto t1 = std::chrono::steady_clock::now();
            for (int it = 0; it < 50; it++) {
                memcpy(fx, x0, n * 4);
                puf_mtl::muon_ns5_rm(s, fx, fxb, fgr, fgb, R, C, coeffs);
                cudaStreamSynchronize(s);
            }
            auto t2 = std::chrono::steady_clock::now();
            t_ref = std::chrono::duration<double, std::milli>(t1 - t0).count() / 50;
            t_fused = std::chrono::duration<double, std::milli>(t2 - t1).count() / 50;
        }
        if (!ok) g_fail++;
        printf("%-22s R=%4d C=%4d fused=%d tensors_differing=%ld  (sync'd wall: ref %.3f ms, fused %.3f ms)%s\n",
            "muon NS5", R, C, (int)ran, bad, t_ref, t_fused, ok ? "" : "  FAIL");
        for (float* p2 : {x0, x, xb, gr, gb, fx, fxb, fgr, fgb}) cudaFree(p2);
    }
}

// GEMM_SMALL_N must equal the tiled kernel bit for bit. Independent oracle:
// a host fma chain over k ascending from +0, then alpha * acc and
// fma(beta, C, v) iff beta != 0 (the tiled kernel's per-element arithmetic).
// Every output starts as NaN (beta == 0 must not read C); padding columns
// and the rows past M must stay untouched.
static void run_small_n(cudaStream_t s, void* ws, size_t ws_bytes) {
    printf("== small-N GEMM vs host fma chain and forced tiled (bitwise)\n");
    struct C { int M, N, K; bool tb; float alpha, beta; int pad; bool want; };
    const C cases[] = {
        {16384, 4, 64, true, 1, 0, 0, true},    // Pong decoder forward
        {65536, 5, 32, true, 1, 0, 0, true},    // Breakout decoder forward
        {16384, 3, 32, true, 1, 0, 0, true},    // Cartpole / Flappy decoder
        {1024, 4, 64, true, 1, 0, 0, true},     // rollout decoder
        {1000, 8, 256, true, 1, 0, 3, true},    // ragged M, max N/K, strided
        {333, 1, 16, true, 0.5f, -1.25f, 2, true},
        {517, 7, 48, false, 1.5f, 0.75f, 1, true},  // NN, alpha/beta
        {129, 8, 128, false, 1, 0, 0, true},
        {64, 9, 64, true, 1, 0, 0, false},      // N > 8: not eligible
        {64, 4, 40, true, 1, 0, 0, false},      // K % 16 != 0: not eligible
        {64, 4, 272, true, 1, 0, 0, false},     // K > 256: not eligible
    };
    std::mt19937 rng(77);
    std::uniform_real_distribution<float> U(-1.0f, 1.0f);
    for (const C& c : cases) {
        int M = c.M, N = c.N, K = c.K;
        int lda = K + c.pad, ldb = (c.tb ? K : N) + c.pad, ldc = N + c.pad;
        long na = (long)M * lda, nb = (long)(c.tb ? N : K) * ldb, nc = (long)(M + 1) * ldc;
        std::vector<float> A(na), B(nb), C0(nc);
        for (auto& v : A) v = U(rng);
        for (auto& v : B) v = U(rng);
        // Exact zeros exercise the sign of zero sums.
        for (long i = 0; i < na; i += 97) A[i] = 0.0f;
        for (long i = 0; i < nc; i++) C0[i] = c.beta == 0.0f && i < (long)M * ldc
            && i % ldc < N ? NAN : U(rng);
        std::vector<float> ref(C0);
        for (int m = 0; m < M; m++)
            for (int n = 0; n < N; n++) {
                float acc = 0.0f;
                for (int k = 0; k < K; k++) {
                    float b = c.tb ? B[(long)n * ldb + k] : B[(long)k * ldb + n];
                    acc = std::fma(A[(long)m * lda + k], b, acc);
                }
                float v = c.alpha * acc;
                long ci = (long)m * ldc + n;
                if (c.beta != 0.0f) v = std::fma(c.beta, C0[ci], v);
                ref[ci] = v;
            }
        float *dA = dalloc(na), *dB = dalloc(nb), *dC = dalloc(nc), *dT = dalloc(nc);
        memcpy(dA, A.data(), na * 4);
        memcpy(dB, B.data(), nb * 4);
        memcpy(dC, C0.data(), nc * 4);
        memcpy(dT, C0.data(), nc * 4);
        puf_mtl::GemmOpts tiled = {false, false, false, false};
        puf_mtl::GemmPath p = puf_mtl::gemm_rm(s, false, c.tb, M, N, K, c.alpha, dA, lda, dB, ldb,
            c.beta, dC, ldc, ws, ws_bytes);
        puf_mtl::GemmPath pt = puf_mtl::gemm_rm(s, false, c.tb, M, N, K, c.alpha, dA, lda, dB, ldb,
            c.beta, dT, ldc, ws, ws_bytes, &tiled);
        cudaStreamSynchronize(s);
        g_path_count[p]++;
        long bad_ref = memcmp(dC, ref.data(), nc * 4) != 0;
        long bad_tiled = 0;
        for (long i = 0; i < nc; i++) bad_tiled += memcmp(dC + i, dT + i, 4) != 0;
        if (bad_ref) {
            bad_ref = 0;
            for (long i = 0; i < nc; i++) bad_ref += memcmp(dC + i, &ref[i], 4) != 0;
        }
        bool path_ok = c.want ? p == puf_mtl::GEMM_SMALL_N : p != puf_mtl::GEMM_SMALL_N;
        bool ok = path_ok && pt == puf_mtl::GEMM_TILED && bad_ref == 0 && bad_tiled == 0;
        if (!ok) g_fail++;
        printf("small_n %6d %2d %4d %c a=%5.2g b=%5.2g pad=%d path %-10s ref_mismatch %ld "
            "tiled_mismatch %ld%s\n", M, N, K, c.tb ? 'T' : 'N', c.alpha, c.beta, c.pad,
            path_name(p), bad_ref, bad_tiled, ok ? "" : "  FAIL");
        cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(dT);
    }
}

// gemm_rm_deferred + gemm_finish must equal gemm_rm bit for bit on every
// path (same split count: it follows ws_bytes, not the room past ws_off).
// Split paths must leave C untouched until gemm_finish, keep their partials
// inside [ws_off, ws_off + ws_used), and encode nothing when the partials do
// not fit; single-dispatch paths leave nothing pending. Two deferred GEMMs
// whose split slices share a concurrent window and whose reductions share
// the next one must not disturb each other's partials.
static void run_deferred(cudaStream_t s, void* ws, size_t ws_bytes) {
    printf("== deferred split-K reduction vs gemm_rm (bitwise)\n");
    struct C { const char* tag; bool ta, tb; int M, N, K; float alpha, beta; bool sg, tops;
        puf_mtl::GemmPath want; };
    const C cases[] = {
        {"tops_tn (H64 dW)", true, false, 192, 64, 16384, 1, 0, true, true,
            puf_mtl::GEMM_TOPS_TN_SPLIT},
        {"sg_tn (dec dW)", true, false, 7, 64, 16384, 1, 0, true, true, puf_mtl::GEMM_SG_TN},
        {"sg_tn (enc dW)", true, false, 64, 52, 16384, 1, 0, true, true, puf_mtl::GEMM_SG_TN},
        {"split_k tn", true, false, 96, 32, 8192, 1, 0, false, false, puf_mtl::GEMM_SPLIT_K},
        {"split_k nn a/b", false, false, 64, 48, 4096, 0.5f, -1.25f, false, false,
            puf_mtl::GEMM_SPLIT_K},
        {"tensor_ops nt", false, true, 256, 64, 64, 1, 0, true, true, puf_mtl::GEMM_TENSOR_OPS},
        {"sg_tn no split", true, false, 32, 32, 2048, 1, 0, true, true, puf_mtl::GEMM_SG_TN},
    };
    std::mt19937 rng(99);
    std::uniform_real_distribution<float> U(-1.0f, 1.0f);
    const size_t off = 1 << 20;
    struct Live { float *dA, *dB, *dC, *dR; long nc; };
    std::vector<Live> live;
    for (const C& c : cases) {
        int M = c.M, N = c.N, K = c.K;
        int lda = c.ta ? M : K, ldb = c.tb ? K : N, ldc = N;
        long na = (long)M * K, nb = (long)K * N, nc = (long)M * N;
        std::vector<float> A(na), B(nb), C0(nc);
        for (auto& v : A) v = U(rng);
        for (auto& v : B) v = U(rng);
        for (auto& v : C0) v = c.beta == 0.0f ? NAN : U(rng);
        float *dA = dalloc(na), *dB = dalloc(nb), *dC = dalloc(nc), *dR = dalloc(nc);
        memcpy(dA, A.data(), na * 4);
        memcpy(dB, B.data(), nb * 4);
        memcpy(dC, C0.data(), nc * 4);
        memcpy(dR, C0.data(), nc * 4);
        puf_mtl::GemmOpts o;
        o.allow_sg_tn = c.sg;
        o.allow_tensor_ops = c.tops;
        // The no-split sg_tn case: a workspace too small for two slices.
        size_t wsb = strstr(c.tag, "no split") ? (size_t)M * N * 4 : ws_bytes;
        // Poison the whole workspace so stale partials cannot pass by luck.
        memset(ws, 0xff, ws_bytes);
        puf_mtl::GemmPath pr = puf_mtl::gemm_rm(s, c.ta, c.tb, M, N, K, c.alpha, dA, lda, dB, ldb,
            c.beta, dR, ldc, ws, wsb, &o);
        cudaStreamSynchronize(s);
        memset(ws, 0xff, ws_bytes);
        puf_mtl::GemmDeferred d;
        uint64_t n0 = puf_mtl::stream_stats(s).dispatches;
        puf_mtl::GemmPath p = puf_mtl::gemm_rm_deferred(s, c.ta, c.tb, M, N, K, c.alpha,
            dA, lda, dB, ldb, c.beta, dC, ldc, ws, wsb, wsb == ws_bytes ? off : 0, &d, &o);
        cudaStreamSynchronize(s);
        uint64_t n1 = puf_mtl::stream_stats(s).dispatches;
        bool split = p == puf_mtl::GEMM_TOPS_TN_SPLIT || p == puf_mtl::GEMM_SPLIT_K
            || (p == puf_mtl::GEMM_SG_TN && d.pipe);
        // Split: C untouched before the finish; the partials sit past off.
        bool untouched = !split || memcmp(dC, C0.data(), nc * 4) == 0;
        bool region = !split || (d.ws_used > 0 && off + d.ws_used <= ws_bytes);
        long below = 0;
        if (split) {
            const unsigned* w = (const unsigned*)ws;
            for (size_t i = 0; i < off / 4; i++) below += w[i] != 0xffffffffu;
        }
        bool pend_ok = split ? d.pipe != nullptr : d.pipe == nullptr;
        bool disp_ok = n1 - n0 == 1;
        puf_mtl::gemm_finish(s, &d);
        bool cleared = d.pipe == nullptr;
        cudaStreamSynchronize(s);
        long bad = 0;
        for (long i = 0; i < nc; i++) bad += memcmp(dC + i, dR + i, 4) != 0;
        bool ok = p == pr && p == c.want && untouched && region && below == 0 && pend_ok
            && disp_ok && cleared && bad == 0;
        if (!ok) g_fail++;
        printf("deferred %-18s path %-10s pending %d untouched %d region %d below_off %ld "
            "mismatch %ld%s\n", c.tag, path_name(p), d.pipe != nullptr || split, untouched,
            region, below, bad, ok ? "" : "  FAIL");
        live.push_back({dA, dB, dC, dR, nc});
    }
    // Partials that do not fit past ws_off: nothing encoded, C untouched.
    {
        const Live& L = live[0];
        std::vector<float> before(L.nc);
        memcpy(before.data(), L.dC, L.nc * 4);
        puf_mtl::GemmDeferred d;
        uint64_t n0 = puf_mtl::stream_stats(s).dispatches;
        puf_mtl::GemmPath p = puf_mtl::gemm_rm_deferred(s, true, false, 192, 64, 16384, 1.0f,
            L.dA, 192, L.dB, 64, 0.0f, L.dC, 64, ws, ws_bytes, ws_bytes - 4096, &d);
        cudaStreamSynchronize(s);
        bool ok = p == puf_mtl::GEMM_NONE && d.pipe == nullptr
            && puf_mtl::stream_stats(s).dispatches == n0
            && memcmp(L.dC, before.data(), L.nc * 4) == 0;
        if (!ok) g_fail++;
        printf("deferred no room: path %s%s\n", path_name(p), ok ? "" : "  FAIL");
    }
    // Two splits in one concurrent window, both reductions in the next.
    // Integrated BwdOverlap exhaustion/reuse is covered by test_bwd_overlap.cpp
    // (both precisions), also run by run_test_gemm.sh.
    {
        const C& a = cases[0];
        const C& b = cases[3];
        const Live& La = live[0];
        const Live& Lb = live[3];
        for (long i = 0; i < La.nc; i++) La.dC[i] = NAN;
        for (long i = 0; i < Lb.nc; i++) Lb.dC[i] = NAN;
        memset(ws, 0xff, ws_bytes);
        puf_mtl::GemmOpts ob;
        ob.allow_sg_tn = false;
        ob.allow_tensor_ops = false;
        puf_mtl::GemmDeferred da, db;
        puf_mtl::concurrent_begin(s);
        puf_mtl::gemm_rm_deferred(s, a.ta, a.tb, a.M, a.N, a.K, 1.0f, La.dA, a.M, La.dB, a.N,
            0.0f, La.dC, a.N, ws, ws_bytes, 0, &da);
        size_t off_b = (da.ws_used + 255) & ~(size_t)255;
        puf_mtl::gemm_rm_deferred(s, b.ta, b.tb, b.M, b.N, b.K, 1.0f, Lb.dA, b.M, Lb.dB, b.N,
            0.0f, Lb.dC, b.N, ws, ws_bytes, off_b, &db, &ob);
        puf_mtl::concurrent_end(s);
        puf_mtl::concurrent_begin(s);
        puf_mtl::gemm_finish(s, &da);
        puf_mtl::gemm_finish(s, &db);
        puf_mtl::concurrent_end(s);
        cudaStreamSynchronize(s);
        long bad = 0;
        for (long i = 0; i < La.nc; i++) bad += memcmp(La.dC + i, La.dR + i, 4) != 0;
        for (long i = 0; i < Lb.nc; i++) bad += memcmp(Lb.dC + i, Lb.dR + i, 4) != 0;
        bool ok = bad == 0 && off_b > 0;
        if (!ok) g_fail++;
        printf("deferred pair in windows: mismatch %ld%s\n", bad, ok ? "" : "  FAIL");
    }
    for (auto& L : live) { cudaFree(L.dA); cudaFree(L.dB); cudaFree(L.dC); cudaFree(L.dR); }
}

static void run_synthetic(cudaStream_t s, void* ws, size_t ws_bytes) {
    printf("== synthetic vs Accelerate cblas_sgemm (row-major op flags)\n");
    printf("%-22s %-2s %6s %6s %6s %5s %5s %3s %-10s %10s %10s %10s\n", "shape", "op", "M", "N",
        "K", "alpha", "beta", "pad", "path", "max_abs", "max_rel", "norm");
    std::mt19937 rng(1234);
    // Breakout (hidden 32, obs 118, 1 discrete head of 3 -> 4 logits+value cols)
    const Shape nt[] = {
        {"rollout enc", 4096, 32, 118}, {"rollout mingru", 4096, 96, 32},
        {"rollout dec", 4096, 4, 32}, {"train enc", 65536, 32, 118},
        {"train mingru", 65536, 96, 32}, {"train dec", 65536, 4, 32},
        {"h128 train mingru", 65536, 384, 128},
    };
    const Shape tn[] = {
        {"dW dec", 4, 32, 65536}, {"dW mingru", 96, 32, 65536}, {"dW enc", 32, 118, 65536},
        {"h128 dW mingru", 384, 128, 65536}, {"h64 dW mingru", 192, 64, 16384},
        {"dW ragged K", 128, 96, 5000},
    };
    const Shape nn[] = {
        {"dX dec", 65536, 32, 4}, {"dX mingru", 65536, 32, 96},
        {"h128 dX mingru", 65536, 128, 384},
    };
    for (auto& sh : nt) run_synth_case(s, ws, ws_bytes, sh, false, true, 1, 0, 0, rng);
    for (auto& sh : tn) run_synth_case(s, ws, ws_bytes, sh, true, false, 1, 0, 0, rng);
    for (auto& sh : nn) run_synth_case(s, ws, ws_bytes, sh, false, false, 1, 0, 0, rng);
    // Muon Newton-Schulz (coeffs of the first iteration): gram, poly, iterate.
    const float c0 = 4.0848f, c1 = -6.8946f, c2 = 2.9270f;
    struct MuonP { const char* tag; int R, C; } mp[] = {
        {"muon enc 32x118", 32, 118}, {"muon dec 4x32", 4, 32}, {"muon mingru 96x32", 96, 32},
        {"muon h128 384x128", 384, 128}, {"muon h128 enc 128x118", 128, 118},
    };
    for (auto& p : mp) {
        bool tall = p.R > p.C;
        int Mg = tall ? p.C : p.R, Ng = tall ? p.R : p.C;
        Shape gram = {p.tag, Mg, Mg, Ng};
        run_synth_case(s, ws, ws_bytes, gram, tall, !tall, 1, 0, 0, rng);   // tall: TN, wide: NT
        Shape poly = {p.tag, Mg, Mg, Mg};
        run_synth_case(s, ws, ws_bytes, poly, false, false, c2, c1, 0, rng);
        Shape iter = tall ? Shape{p.tag, p.R, p.C, p.C} : Shape{p.tag, p.R, p.C, p.R};
        run_synth_case(s, ws, ws_bytes, iter, false, false, 1, c0, 0, rng);
    }
    // Every transpose combo, odd sizes, alpha/beta, strided operands.
    const Shape odd[] = {{"odd small", 37, 53, 29}, {"odd K-long", 70, 33, 5000},
        {"odd K-long ragged", 13, 21, 4099}, {"K-long no split", 400, 400, 2100},
        {"aligned 128", 128, 64, 96}, {"K=0", 16, 16, 0}};
    for (auto& sh : odd)
        for (int ta = 0; ta < 2; ta++)
            for (int tb = 0; tb < 2; tb++) {
                run_synth_case(s, ws, ws_bytes, sh, ta, tb, 1, 0, 0, rng);
                run_synth_case(s, ws, ws_bytes, sh, ta, tb, 0.5f, -1.25f, 0, rng);
                run_synth_case(s, ws, ws_bytes, sh, ta, tb, 1.5f, 0, 3, rng);
            }
    run_alpha_zero(s, ws, ws_bytes);
    run_nn_add(s, ws, ws_bytes);
    run_small_n(s, ws, ws_bytes);
    run_deferred(s, ws, ws_bytes);
    // Every implementation must actually have been exercised.
    const puf_mtl::GemmPath need[] = {puf_mtl::GEMM_TENSOR_OPS, puf_mtl::GEMM_TILED,
        puf_mtl::GEMM_SPLIT_K, puf_mtl::GEMM_SCALE, puf_mtl::GEMM_SG_TN,
        puf_mtl::GEMM_TOPS_TN_SPLIT, puf_mtl::GEMM_SMALL_N};
    for (auto p : need) {
        printf("path %-10s exercised %ld times%s\n", path_name(p), g_path_count[p],
            g_path_count[p] ? "" : "  FAIL");
        if (!g_path_count[p]) g_fail++;
    }
}

int main(int argc, char** argv) {
    @autoreleasepool {
        cudaStream_t s;
        cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
        const size_t ws_bytes = 4 * 1024 * 1024;  // same as algo.cu cublas_init_stream
        void* ws = nullptr;
        cudaMalloc(&ws, ws_bytes);
        cublasHandle_t h;
        cublasCreate(&h);
        cublasSetStream(h, s);
        cublasSetWorkspace(h, ws, ws_bytes);
        g_handle = h;
        for (int i = 1; i < argc; i++) run_golden(argv[i], h);
        run_synthetic(s, ws, ws_bytes);
        run_muon_ns(s, h);
        puf_mtl::Stats st = puf_mtl::stats();
        printf("dispatches=%llu commits=%llu host_syncs=%llu\n",
            (unsigned long long)st.dispatches, (unsigned long long)st.commits,
            (unsigned long long)st.host_syncs);
        printf(g_fail ? "FAILED: %d checks\n" : "ALL PASS\n", g_fail);
        return g_fail ? 1 : 0;
    }
}
