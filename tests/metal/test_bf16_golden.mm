// BF16 counterpart of the learner golden gates (test_gemm / test_kernels /
// test_mingru / test_ppo_muon) against CUDA BF16 dumps (manifest precision
// "BF16", tests/metal_ref on CUDA-reference branch golden-bf16). Built against the bf16
// metallib with precision_t = __nv_bfloat16, through the real PUF_LAUNCH
// adapters and the cuBLAS shim, mirroring the CUDA call sequence of each case:
//
//   gemm_*      cublasGemmEx 16BF in/out, COMPUTE_32F (puf_mm/_tn/_nn mapping)
//   cast_bf16_probe  cast_f32_to_prec / cast_prec_to_f32 (bits, NaN payloads)
//   adv_*       puff_advantage (bf16 W=8 segments)
//   gate_*      mingru_gate
//   scan_*      mingru_scan_forward(+_row) / backward (and the other variant)
//   policy_*    encoder/MinGRU/decoder forward + backward (bf16 GEMMs, scans,
//               assemble_decoder_grad, add_kernel), as algo.cu
//   ppo_*       cache_imp_and_v, ppo_loss_compute, ppo_loss_reduce
//   muon_*      muon_step: sum_sq, clip_nesterov (fp32 momentum), NS5 tail on
//               bf16 (unfused GEMMs, and muon_ns5_batch_rm bit-identical to
//               it), fp32 master update, bf16 param refresh (cast)
//   sample_*    rng_init, curand_uniform, sample_logits
//
// Tolerances. The CUDA BF16 goldens are bit-reproducible run to run
// (golden/bf16_reports/summary.json: 0 differing tensors over two fresh
// generator processes and two replays), so the CUDA noise floor is zero and
// every allowed difference is a cross-implementation one. Both sides
// accumulate GEMMs in fp32 and round to bf16 once; the order of the fp32
// accumulation differs (cuBLAS tensor-core tiles vs Metal simdgroup /
// tensor_ops tiles), which flips the final bf16 rounding of an element when
// the two fp32 sums straddle a rounding boundary. A direct GEMM output can
// therefore differ by 1 bf16 ulp (2^-8 relative); everything downstream of a
// bf16 GEMM inherits a few ulps. Each check reports bit-exact count, max
// ulps (in the output dtype) and norm = max_abs / max|ref|; the pass bound per
// tensor family is stated where it is checked.
//
// Build/run: tests/metal/run_test_bf16_golden.sh GOLDEN_BF16_RUN_DIR ...
#import <Foundation/Foundation.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <type_traits>
#include <vector>

#include "compat/cublas_v2.h"
#include "compat/cuda_bf16.h"
#include "compat/cuda_runtime.h"
#include "compat/curand_kernel.h"
#include "gemm.h"
#include "launch.h"
#include "runtime.h"

#ifdef PRECISION_FLOAT
#error "test_bf16_golden is built against the bf16 metallib (no PRECISION_FLOAT)"
#endif

// Stand-ins for the upstream types kernels.h adapts (src/pufferl.cu, algo.cu).
typedef __nv_bfloat16 precision_t;
typedef __nv_bfloat16 bf16;
#define PUF_MAX_DIMS 8
#define BLOCK_SIZE 256
typedef struct { float* data; int64_t shape[PUF_MAX_DIMS]; } Float;
typedef struct { precision_t* data; int64_t shape[PUF_MAX_DIMS]; } Prec;
static int64_t numel(const int64_t* shape) {
    int64_t n = 1;
    for (int i = 0; i < PUF_MAX_DIMS && shape[i] != 0; i++) n *= shape[i];
    return n;
}
static int g_num_atns = 1;
#define NUM_ATNS g_num_atns
struct PrefixScan {
    precision_t* combined_ptr = NULL;
    precision_t* state_ptr = NULL;
    precision_t* input_ptr = NULL;
    precision_t* terminals_ptr = NULL;
    int B = 0, T = 0, H = 0;
    Prec scan_h;
    Prec out, next_state;
    Prec grad_combined, grad_state;
    Prec grad_input;
};
#define MINGRU_TILE 32
struct PPOGraphArgs {
    precision_t* imp;
    const float* actions;
    const precision_t *old_logprobs, *advantages, *values, *returns;
};
struct PPOKernelArgs {
    float *grad_logits, *grad_logstd, *grad_values_pred;
    const precision_t *logits, *logstd, *values_pred;
    const int* act_sizes;
    const precision_t* action_mask;
    int num_atns;
    float clip_coef, vf_clip_coef, vf_coef;
    const float* ent_coef;
    int T_seq, A_total, N;
    bool is_continuous;
};
#include "kernels.h"

static int g_fail = 0;
static int grid(long n) { return (int)((n + 255) / 256); }

template <typename T>
static T* dalloc(long n) {
    T* p = nullptr;
    if (cudaMalloc((void**)&p, (size_t)std::max(n, 1L) * sizeof(T)) != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed\n");
        exit(2);
    }
    return p;
}
static void poison(bf16* p, long n) { for (long i = 0; i < n; i++) p[i].x = 0x7fc1; }
static void poison(float* p, long n) { for (long i = 0; i < n; i++) p[i] = NAN; }

// ---------------------------------------------------------------------------
// Golden reading
// ---------------------------------------------------------------------------
struct Case {
    NSString* dir;
    NSDictionary *tensors, *scalars;
    int I(NSString* k) const { return [scalars[k] intValue]; }
    float F(NSString* k) const { return [scalars[k] floatValue]; }
    bool has(NSString* k) const { return tensors[k] != nil; }
    NSString* dtype(NSString* k) const { return tensors[k][@"dtype"]; }
    NSArray* shape(NSString* k) const { return tensors[k][@"shape"]; }
    size_t count(NSString* k) const {
        size_t n = 1;
        for (NSNumber* d in tensors[k][@"shape"]) n *= d.unsignedLongLongValue;
        return n;
    }
    // Raw elements of a tensor whose dtype must have sizeof(T) bytes.
    template <class T> std::vector<T> raw(NSString* k) const {
        NSDictionary* spec = tensors[k];
        if (!spec) { fprintf(stderr, "%s: no tensor %s\n", dir.UTF8String, k.UTF8String); exit(2); }
        NSString* dt = spec[@"dtype"];
        size_t es = [dt isEqual:@"bf16"] ? 2 : [dt isEqual:@"u8"] ? 1 : [dt isEqual:@"u64"] ? 8 : 4;
        if (es != sizeof(T)) {
            fprintf(stderr, "%s/%s: dtype %s read as %zu-byte elements\n", dir.UTF8String,
                k.UTF8String, dt.UTF8String, sizeof(T));
            exit(2);
        }
        NSData* d = [NSData dataWithContentsOfFile:[dir stringByAppendingPathComponent:spec[@"file"]]];
        if (!d || d.length != count(k) * es) {
            fprintf(stderr, "%s/%s: bad tensor file\n", dir.UTF8String, k.UTF8String);
            exit(2);
        }
        std::vector<T> v(count(k));
        memcpy(v.data(), d.bytes, d.length);
        return v;
    }
    template <class T> T* put(NSString* k) const {
        std::vector<T> v = raw<T>(k);
        T* p = dalloc<T>((long)v.size());
        memcpy(p, v.data(), v.size() * sizeof(T));
        return p;
    }
};
static Case load_case(NSString* dir) {
    NSData* mj = [NSData dataWithContentsOfFile:[dir stringByAppendingPathComponent:@"manifest.json"]];
    NSDictionary* m = mj ? [NSJSONSerialization JSONObjectWithData:mj options:0 error:nil] : nil;
    if (!m || [m[@"schema_version"] intValue] != 1 || ![m[@"byte_order"] isEqual:@"little"]) {
        fprintf(stderr, "bad manifest in %s\n", dir.UTF8String);
        exit(2);
    }
    if (![m[@"precision"] isEqual:@"BF16"]) {
        fprintf(stderr, "%s: precision %s, expected BF16 goldens\n", dir.UTF8String,
            [m[@"precision"] description].UTF8String);
        exit(2);
    }
    return {dir, m[@"tensors"], m[@"scalars"]};
}

// ---------------------------------------------------------------------------
// Comparison
// ---------------------------------------------------------------------------
static int64_t ord16(uint16_t b) { return (b & 0x8000) ? -(int64_t)(b & 0x7fff) : (int64_t)b; }
static int64_t ord32(uint32_t b) {
    return (b & 0x80000000u) ? -(int64_t)(b & 0x7fffffffu) : (int64_t)b;
}
static float f32_of(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }

// Pass: every element within max_ulp (output dtype ulps) OR within
// norm_tol * max|ref| (absolute floor for cancellation near zero) OR within
// rel_tol * |ref|. bits: also NaN payloads must match (EXACT).
struct Tol {
    int64_t max_ulp;
    double norm_tol;
    double rel_tol = 0, abs_tol = 0;
    bool bits = false;
};
static const Tol EXACT = {0, 0.0, 0.0, 0.0, true};
// Exact except NaN payloads (Metal quiets as (bits >> 16) | 0x40, CUDA
// returns 0x7FFF; see src/metal/shaders/common.h).
static const Tol EXACT_NAN_ANY = {0, 0.0, 0.0, 0.0, false};

struct Stats {
    long n = 0, exact = 0, bad = 0, first_bad = -1;
    int64_t max_ulp = 0;
    double max_abs = 0, max_ref = 0;
};

template <class T>
static Stats stats_of(const T* got, const std::vector<T>& ref, Tol tol) {
    Stats s;
    s.n = (long)ref.size();
    std::vector<double> a(ref.size()), b(ref.size());
    for (size_t i = 0; i < ref.size(); i++) {
        if constexpr (std::is_same_v<T, bf16>) {
            a[i] = puf_bf16_bits_to_float(got[i].x); b[i] = puf_bf16_bits_to_float(ref[i].x);
        } else {
            a[i] = got[i]; b[i] = ref[i];
        }
        if (std::isfinite(b[i])) s.max_ref = std::max(s.max_ref, fabs(b[i]));
    }
    for (size_t i = 0; i < ref.size(); i++) {
        int64_t ulp;
        bool same_bits;
        if constexpr (std::is_same_v<T, bf16>) {
            same_bits = got[i].x == ref[i].x;
            ulp = std::llabs(ord16(got[i].x) - ord16(ref[i].x));
        } else {
            uint32_t ga, rb;
            memcpy(&ga, &got[i], 4); memcpy(&rb, &ref[i], 4);
            same_bits = ga == rb;
            ulp = std::llabs(ord32(ga) - ord32(rb));
        }
        s.exact += same_bits;
        if (same_bits) continue;
        if (std::isnan(a[i]) && std::isnan(b[i])) {
            if (tol.bits && s.first_bad < 0) s.first_bad = (long)i;
            s.bad += tol.bits;
            continue;
        }
        bool finite = std::isfinite(a[i]) && std::isfinite(b[i]);
        double d = finite ? fabs(a[i] - b[i]) : INFINITY;
        s.max_abs = std::max(s.max_abs, d);
        s.max_ulp = std::max(s.max_ulp, ulp);
        if (!finite || (ulp > tol.max_ulp && d > tol.norm_tol * s.max_ref &&
                           d > tol.rel_tol * fabs(b[i]) && d > tol.abs_tol)) {
            if (s.first_bad < 0) s.first_bad = (long)i;
            s.bad++;
        }
    }
    return s;
}

template <class T>
static void check(const Case& c, const std::string& cs, NSString* name, const T* got, Tol tol,
        const std::vector<T>* ref_override = nullptr) {
    std::vector<T> ref = ref_override ? *ref_override : c.raw<T>(name);
    Stats s = stats_of(got, ref, tol);
    if (s.bad) g_fail++;
    double norm = s.max_ref > 0 ? s.max_abs / s.max_ref : s.max_abs;
    printf("  %-30s %-24s %-4s n=%-8ld bit-exact=%-8ld max_ulp=%-6lld norm=%.3e%s\n",
        cs.c_str(), name.UTF8String, std::is_same_v<T, bf16> ? "bf16" : "f32", s.n, s.exact,
        (long long)s.max_ulp, norm, s.bad ? "  FAIL" : "");
    if (s.bad) {
        long i = s.first_bad;
        double g, r;
        if constexpr (std::is_same_v<T, bf16>) {
            g = puf_bf16_bits_to_float(got[i].x); r = puf_bf16_bits_to_float(ref[i].x);
            printf("    %ld bad; first at %ld: got 0x%04x (%.9g) ref 0x%04x (%.9g)\n", s.bad, i,
                got[i].x, g, ref[i].x, r);
        } else {
            uint32_t gb, rb;
            memcpy(&gb, &got[i], 4); memcpy(&rb, &ref[i], 4);
            printf("    %ld bad; first at %ld: got 0x%08x (%.9g) ref 0x%08x (%.9g)\n", s.bad, i,
                gb, (double)got[i], rb, (double)ref[i]);
        }
    }
}

// ---------------------------------------------------------------------------
// GEMM (algo.cu cublasGemmExDense + puf_mm / puf_mm_tn / puf_mm_nn)
// ---------------------------------------------------------------------------
static cublasHandle_t g_h;
static void dense_gemm(cublasOperation_t op_a, cublasOperation_t op_b, int M, int N, int K,
        const bf16* A, const bf16* B, bf16* C, float alpha = 1.0f, float beta = 0.0f) {
    int lda = (op_a == CUBLAS_OP_N) ? K : M;
    int ldb = (op_b == CUBLAS_OP_N) ? N : K;
    cublasStatus_t st = cublasGemmEx(g_h, op_b, op_a, N, M, K, &alpha, B, CUDA_R_16BF, ldb, A,
        CUDA_R_16BF, lda, &beta, C, CUDA_R_16BF, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    if (st != CUBLAS_STATUS_SUCCESS) { fprintf(stderr, "cublasGemmEx bf16 failed\n"); exit(2); }
}
static void mm_nt(const bf16* a, const bf16* b, bf16* out, int M, int N, int K) {
    dense_gemm(CUBLAS_OP_N, CUBLAS_OP_T, M, N, K, a, b, out);
}
static void mm_tn(const bf16* a, const bf16* b, bf16* out, int M, int N, int K) {
    dense_gemm(CUBLAS_OP_T, CUBLAS_OP_N, M, N, K, a, b, out);
}
static void mm_nn(const bf16* a, const bf16* b, bf16* out, int M, int N, int K) {
    dense_gemm(CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, a, b, out);
}

// Direct GEMM outputs. Metal must be within 1 bf16 ulp of the exactly
// rounded product (fp64 host reference on the same bf16 inputs; 2^-12 x
// max|C| floor for cancellation, where an fp32 sum near zero carries
// accumulation-order error above its own bf16 ulp). Metal vs CUDA: the same
// bound, except where CUDA itself is further from exact math: cuBLAS's bf16
// weight-gradient GEMMs (tn, K = 65536) are off by up to ~0.005 x max|C|
// (golden/bf16_reports; validate.py bounds them by 2^-7 x max|C|), so the
// Metal-vs-CUDA bound there is CUDA's own measured error vs exact + 2^-12.
static const Tol GEMM_TOL = {1, 1.0 / 4096};

static void run_gemm(const Case& c, const std::string& cs) {
    NSString* op = c.scalars[@"operation"];
    int M = c.I(@"M"), N = c.I(@"N"), K = c.I(@"K");
    float alpha = c.F(@"alpha"), beta = c.F(@"beta");
    cublasOperation_t op_a = CUBLAS_OP_N, op_b = CUBLAS_OP_N;
    if ([op isEqualToString:@"puf_mm"]) op_b = CUBLAS_OP_T;
    else if ([op isEqualToString:@"puf_mm_tn"]) op_a = CUBLAS_OP_T;
    else if (![op isEqualToString:@"puf_mm_nn"]) { fprintf(stderr, "unknown op\n"); exit(2); }
    bf16 *A = c.put<bf16>(@"a"), *B = c.put<bf16>(@"b"), *C = c.put<bf16>(@"c_in");
    if ((long)c.count(@"c_out") != (long)M * N) { fprintf(stderr, "%s: size\n", cs.c_str()); exit(2); }
    // fp64 reference: alpha * op(A) op(B) + beta * C_in, rounded once to bf16.
    std::vector<bf16> exact((size_t)M * N);
    {
        std::vector<double> a((size_t)M * K), b((size_t)K * N);
        for (int m = 0; m < M; m++)
            for (int k = 0; k < K; k++)
                a[(size_t)m * K + k] = puf_bf16_bits_to_float(
                    (op_a == CUBLAS_OP_N ? A[(size_t)m * K + k] : A[(size_t)k * M + m]).x);
        for (int k = 0; k < K; k++)
            for (int n = 0; n < N; n++)
                b[(size_t)k * N + n] = puf_bf16_bits_to_float(
                    (op_b == CUBLAS_OP_N ? B[(size_t)k * N + n] : B[(size_t)n * K + k]).x);
        std::vector<double> acc(N);
        for (int m = 0; m < M; m++) {
            std::fill(acc.begin(), acc.end(), 0.0);
            for (int k = 0; k < K; k++) {
                double av = a[(size_t)m * K + k];
                if (av == 0.0) continue;
                const double* br = &b[(size_t)k * N];
                for (int n = 0; n < N; n++) acc[n] += av * br[n];
            }
            for (int n = 0; n < N; n++) {
                double v = (double)alpha * acc[n];
                if (beta != 0.0f) v += (double)beta * puf_bf16_bits_to_float(C[(size_t)m * N + n].x);
                exact[(size_t)m * N + n] = __float2bfloat16((float)v);
            }
        }
    }
    dense_gemm(op_a, op_b, M, N, K, A, B, C, alpha, beta);
    cudaDeviceSynchronize();
    std::vector<bf16> cuda = c.raw<bf16>(@"c_out");
    Stats ce = stats_of(cuda.data(), exact, GEMM_TOL);
    double cuda_norm = ce.max_ref > 0 ? ce.max_abs / ce.max_ref : 0;
    check<bf16>(c, cs + " [vs exact]", @"c_out", C, GEMM_TOL, &exact);
    Tol vs_cuda = GEMM_TOL;
    if (ce.bad) vs_cuda.norm_tol = cuda_norm + 1.0 / 4096;
    printf("  %-30s cuda vs exact: bit-exact=%ld/%ld norm=%.3e%s\n", cs.c_str(), ce.exact, ce.n,
        cuda_norm, ce.bad ? "  (beyond 1 ulp: Metal-vs-CUDA bound widened to it)" : "");
    check<bf16>(c, cs, @"c_out", C, vs_cuda);
    cudaFree(A); cudaFree(B); cudaFree(C);
}

// ---------------------------------------------------------------------------
// Conversions: bit-exact, including NaN payload quieting.
// ---------------------------------------------------------------------------
static void run_cast_probe(const Case& c, const std::string& cs) {
    float* in = c.put<float>(@"input");
    long n = (long)c.count(@"input");
    bf16* mid = dalloc<bf16>(n);
    float* back = dalloc<float>(n);
    PUF_LAUNCH(cast, grid(n), 256, 0, 0)(mid, in, (int)n);
    PUF_LAUNCH(cast, grid(n), 256, 0, 0)(back, mid, (int)n);
    cudaDeviceSynchronize();
    check<bf16>(c, cs, @"bf16", mid, EXACT_NAN_ANY);
    check<float>(c, cs, @"roundtrip", back, EXACT_NAN_ANY);
    // The host conversion (compat/cuda_bf16.h) vs CUDA's host __float2bfloat16.
    std::vector<float> hin = c.raw<float>(@"input");
    std::vector<bf16> host(n);
    for (long i = 0; i < n; i++) host[i] = __float2bfloat16(hin[i]);
    check<bf16>(c, cs, @"bf16_host", host.data(), EXACT_NAN_ANY);
    // Metal device == Metal host bit for bit (NaN payloads included), and
    // every non-bit-exact element vs CUDA is a NaN.
    std::vector<bf16> ref = c.raw<bf16>(@"bf16");
    long dev_host = 0, nan_payload = 0;
    for (long i = 0; i < n; i++) {
        dev_host += mid[i].x != host[i].x;
        if (mid[i].x != ref[i].x) nan_payload += std::isnan(hin[i]) ? 1 : 1000000;
    }
    bool ok = dev_host == 0 && nan_payload < 1000000;
    printf("  %-30s device==host mismatches=%ld; vs CUDA only NaN payloads differ: %ld%s\n",
        cs.c_str(), dev_host, nan_payload % 1000000, ok ? "" : "  FAIL");
    if (!ok) g_fail++;
    cudaFree(in); cudaFree(mid); cudaFree(back);
}

// ---------------------------------------------------------------------------
// Advantage: no GEMM, explicit fma contractions on both sides -> bit-exact.
// ---------------------------------------------------------------------------
static void run_advantage(const Case& c, const std::string& cs) {
    int B = c.I(@"B"), T = c.I(@"T");
    bf16 *v = c.put<bf16>(@"values"), *r = c.put<bf16>(@"rewards"), *d = c.put<bf16>(@"terminals");
    bf16* imp = c.I(@"importance_is_null") ? nullptr : c.put<bf16>(@"importance");
    bf16 *adv = dalloc<bf16>((long)B * T), *ret = dalloc<bf16>((long)B * T);
    poison(adv, (long)B * T); poison(ret, (long)B * T);
    PUF_LAUNCH(puff_advantage, (B + 63) / 64, 64, 0, 0)(v, r, d, imp, adv, ret, c.F(@"gamma"),
        c.F(@"lambda"), c.F(@"rho_clip"), c.F(@"c_clip"), B, T);
    cudaDeviceSynchronize();
    check<bf16>(c, cs, @"advantages", adv, EXACT);
    check<bf16>(c, cs, @"returns", ret, EXACT);
    // In place (train writes returns over values).
    PUF_LAUNCH(puff_advantage, (B + 63) / 64, 64, 0, 0)(v, r, d, imp, adv, v, c.F(@"gamma"),
        c.F(@"lambda"), c.F(@"rho_clip"), c.F(@"c_clip"), B, T);
    cudaDeviceSynchronize();
    std::vector<bf16> rr = c.raw<bf16>(@"returns");
    check<bf16>(c, cs, @"returns", v, EXACT, &rr);
    cudaFree(v); cudaFree(r); cudaFree(d); if (imp) cudaFree(imp); cudaFree(adv); cudaFree(ret);
}

// ---------------------------------------------------------------------------
// MinGRU gate / scans: elementwise transcendental math on bf16 inputs; MSL
// vs CUDA libm (exp/log1p/sigmoid) differ in the last fp32 bits, which can
// flip one bf16 rounding; the scans carry h_t (rounded each step) along T.
// ---------------------------------------------------------------------------
static const Tol ELEM_TOL = {1, 0.0};
static const Tol SCAN_TOL = {2, 1.0 / 4096};

// Round binary64 directly to BF16, independent of the host rounding mode.
// BF16 values and their midpoints are exactly representable in binary64.
// Searching the positive finite encodings avoids an intermediate FP32 round.
static bf16 double_to_bf16_rne(double value) {
    uint16_t sign = std::signbit(value) ? 0x8000 : 0;
    double a = fabs(value);
    if (std::isnan(a)) return bf16::from_bits(sign | 0x7fc0);
    if (a >= 0x1.ffp127) return bf16::from_bits(sign | 0x7f80);
    auto decode = [](uint16_t bits) { return (double)puf_bf16_bits_to_float(bits); };
    if (a >= decode(0x7f7f)) return bf16::from_bits(sign | 0x7f7f);
    uint16_t lo = 0, hi = 0x7f7f;
    while (lo < hi) {
        uint16_t mid = lo + (hi - lo) / 2;
        if (decode(mid) < a) lo = mid + 1;
        else hi = mid;
    }
    if (lo && a != decode(lo)) {
        double midpoint = (decode(lo - 1) + decode(lo)) * .5;
        if (a < midpoint || (a == midpoint && (lo & 1))) --lo;
    }
    return bf16::from_bits(sign | lo);
}

static void check_gate_state(const Case& c, const std::string& cs, int B, int H,
        const bf16* comb, const bf16* st, const bf16* next, std::vector<bf16> expected) {
    // Near cancellation, FP32 sigmoid/lerp roundoff can span several BF16
    // output ulps. Keep the 1-ulp gate everywhere else. For an exception,
    // independently require BOTH CUDA and Metal to be within the same
    // one-ulp limit of correctly rounded float64. No absolute floor is added.
    long cancellation_checks=0;
    for(long j=0;j<(long)B*H;j++) {
        if (std::llabs(ord16(next[j].x)-ord16(expected[j].x))<=1) continue;
        int b=(int)(j/H), h=(int)(j%H), k=b*3*H+h;
        auto f=[](bf16 v) { return (double)puf_bf16_bits_to_float(v.x); };
        double hidden=f(comb[k]), z=1/(1+exp(-f(comb[k+H]))), initial=f(st[j]);
        double ht=hidden>=0 ? hidden+.5 : 1/(1+exp(-hidden));
        double exact=(1-z)*initial+z*ht;
        double condition=fabs((1-z)*initial)+fabs(z*ht);
        bf16 rounded=double_to_bf16_rne(exact);
        long metal_ulp=std::llabs(ord16(next[j].x)-ord16(rounded.x));
        long cuda_ulp=std::llabs(ord16(expected[j].x)-ord16(rounded.x));
        if (std::isfinite(f(next[j])) && std::isfinite(f(expected[j])) &&
                fabs(exact)<0x1p-8*condition &&
                metal_ulp<=1 && cuda_ulp<=1) {
            printf("  %s state_out[%ld] f64 cancellation: Metal=%.12e CUDA=%.12e exact=%.12e rounded=%.12e ulps=%ld/%ld\n",
                cs.c_str(),j,f(next[j]),f(expected[j]),exact,f(rounded),metal_ulp,cuda_ulp);
            expected[j]=rounded;
            cancellation_checks++;
        }
    }
    check<bf16>(c, cs+(cancellation_checks ? " [f64 cancellation checked]" : ""),
        @"state_out", next, ELEM_TOL, &expected);
}

static void run_gate(const Case& c, const std::string& cs) {
    int B = c.I(@"B"), H = c.I(@"H");
    bf16 *comb = c.put<bf16>(@"combined"), *st = c.put<bf16>(@"state_in"), *x = c.put<bf16>(@"x");
    bf16 *out = dalloc<bf16>((long)B * H), *next = dalloc<bf16>((long)B * H);
    PUF_LAUNCH(mingru_gate, grid((long)B * H), 256, 0, 0)(out, next, comb, st, x, H, B);
    cudaDeviceSynchronize();
    check<bf16>(c, cs, @"out", out, ELEM_TOL);
    check_gate_state(c, cs, B, H, comb, st, next, c.raw<bf16>(@"state_out"));
    cudaFree(comb); cudaFree(st); cudaFree(x); cudaFree(out); cudaFree(next);
}

static bool use_row(const PrefixScan& s) {
    return (s.T >= 256 && s.B * s.H <= 32768) || (s.T >= 128 && s.H <= 32);
}
static void scan_forward(PrefixScan& scan, bool row) {
    if (row) {
        dim3 block(MINGRU_TILE, MINGRU_TILE), g(scan.B, (scan.H + MINGRU_TILE - 1) / MINGRU_TILE);
        PUF_LAUNCH(mingru_scan_forward_row, g, block, 0, 0)(scan);
    } else {
        PUF_LAUNCH(mingru_scan_forward, grid((long)scan.B * scan.H), BLOCK_SIZE, 0, 0)(scan);
    }
}
static void scan_backward(PrefixScan& scan, bool row, bf16* grad, bf16* grad_next) {
    if (row) {
        dim3 block(MINGRU_TILE, MINGRU_TILE), g(scan.B, (scan.H + MINGRU_TILE - 1) / MINGRU_TILE);
        PUF_LAUNCH(mingru_scan_backward_row, g, block, 0, 0)(scan, grad, grad_next);
    } else {
        PUF_LAUNCH(mingru_scan_backward, grid((long)scan.B * scan.H), BLOCK_SIZE, 0, 0)(
            scan, grad, grad_next);
    }
}
static PrefixScan alloc_scan(int B, int T, int H) {
    PrefixScan s;
    s.B = B; s.T = T; s.H = H;
    long BTH = (long)B * T * H;
    s.scan_h = {dalloc<bf16>(BTH), {B, T, H}};
    s.out = {dalloc<bf16>(BTH), {B, T, H}};
    s.next_state = {dalloc<bf16>((long)B * H), {B, 1, H}};
    s.grad_combined = {dalloc<bf16>(3 * BTH), {B, T, 3 * H}};
    s.grad_state = {dalloc<bf16>((long)B * H), {B, 1, H}};
    s.grad_input = {dalloc<bf16>(BTH), {B, T, H}};
    poison(s.scan_h.data, BTH); poison(s.out.data, BTH); poison(s.next_state.data, (long)B * H);
    poison(s.grad_combined.data, 3 * BTH); poison(s.grad_state.data, (long)B * H);
    poison(s.grad_input.data, BTH);
    return s;
}
static void free_scan(PrefixScan& s) {
    cudaFree(s.scan_h.data); cudaFree(s.out.data); cudaFree(s.next_state.data);
    cudaFree(s.grad_combined.data); cudaFree(s.grad_state.data); cudaFree(s.grad_input.data);
}

static void run_scan(const Case& c, const std::string& cs, bool row) {
    int B = c.I(@"B"), T = c.I(@"T"), H = c.I(@"H");
    bf16 *comb = c.put<bf16>(@"combined"), *st = c.put<bf16>(@"state_in"), *x = c.put<bf16>(@"x");
    bf16 *term = c.put<bf16>(@"terminals"), *gout = c.put<bf16>(@"grad_out");
    bf16* gnext = c.put<bf16>(@"grad_next_state");
    PrefixScan s = alloc_scan(B, T, H);
    s.combined_ptr = comb; s.state_ptr = st; s.input_ptr = x; s.terminals_ptr = term;
    scan_forward(s, row);
    scan_backward(s, row, gout, gnext);
    cudaDeviceSynchronize();
    check<bf16>(c, cs, @"out", s.out.data, SCAN_TOL);
    check<bf16>(c, cs, @"next_state", s.next_state.data, SCAN_TOL);
    check<bf16>(c, cs, @"scan_h", s.scan_h.data, SCAN_TOL);
    check<bf16>(c, cs, @"grad_combined", s.grad_combined.data, SCAN_TOL);
    check<bf16>(c, cs, @"grad_input", s.grad_input.data, SCAN_TOL);
    check<bf16>(c, cs, @"grad_state", s.grad_state.data, SCAN_TOL);
    free_scan(s);
    for (bf16* p : {comb, st, x, term, gout, gnext}) cudaFree(p);
}

// ---------------------------------------------------------------------------
// Policy forward + backward (algo.cu arch_forward_train + arch_backward).
// Four MinGRU layers of bf16 GEMMs feeding scans: rounding flips compound
// layer to layer, so outputs are held to a norm bound (2^-6 of max|ref|, i.e.
// a few bf16 ulps of the largest element) rather than per-element ulps;
// weight gradients reduce over B*T = 65536 rows of bf16 products.
// ---------------------------------------------------------------------------
static const Tol POLICY_TOL = {4, 1.0 / 64};

static void run_policy(const Case& c, const std::string& cs) {
    int B = c.I(@"B"), TT = c.I(@"T"), H = c.I(@"H"), L = c.I(@"layers");
    int in_dim = c.I(@"input_dim"), agent_off = c.I(@"agent_off");
    long B_TT = (long)B * TT;
    int od1 = (int)(c.count(@"param_1") / H), od = od1 - 1;
    int A = [c.shape(@"initial_state")[1] intValue];
    bf16 *obs = c.put<bf16>(@"obs"), *state = c.put<bf16>(@"initial_state");
    bf16* term = c.put<bf16>(@"terminals");
    float *dlogits = c.put<float>(@"dlogits"), *dvalue = c.put<float>(@"dvalue");
    bf16* gnext = c.put<bf16>(@"grad_next_state");
    bf16 *W_enc = c.put<bf16>(@"param_0"), *W_dec = c.put<bf16>(@"param_1");
    std::vector<bf16*> W(L);
    for (int i = 0; i < L; i++) W[i] = c.put<bf16>([NSString stringWithFormat:@"param_%d", 2 + i]);

    bf16* enc_out = dalloc<bf16>(B_TT * H);
    bf16* enc_saved = dalloc<bf16>(B_TT * in_dim);
    bf16* enc_wgrad = dalloc<bf16>((long)H * in_dim);
    bf16* dec_out = dalloc<bf16>(B_TT * od1);
    bf16* dec_grad_out = dalloc<bf16>(B_TT * od1);
    bf16* dec_saved = dalloc<bf16>(B_TT * H);
    bf16* dec_grad_in = dalloc<bf16>(B_TT * H);
    bf16* dec_wgrad = dalloc<bf16>((long)od1 * H);
    bf16* grad_input_buf = dalloc<bf16>(B_TT * H);
    std::vector<bf16*> saved(L), combined(L), wgrad(L);
    std::vector<PrefixScan> scans(L);
    for (int i = 0; i < L; i++) {
        saved[i] = dalloc<bf16>(B_TT * H);
        combined[i] = dalloc<bf16>(B_TT * 3 * H);
        wgrad[i] = dalloc<bf16>((long)3 * H * H);
        scans[i] = alloc_scan(B, TT, H);
    }
    bool want_fused = getenv("PUF_GOLDWIDE_FUSED") && atoi(getenv("PUF_GOLDWIDE_FUSED"));
    bool fused = want_fused && puf_mtl::mingru_fused_eligible(B, TT, H, enc_out, W[0]);
    if (want_fused && (H == 128 || H == 256) && TT == 64 && !fused) {
        fprintf(stderr, "required H128/H256 golden fused path unavailable\n"); exit(2);
    }
    printf("  %s H=%d T=%d policy path=%s\n", cs.c_str(), H, TT, fused ? "fused" : "generic");
    cudaStream_t stream = 0;
    cudaMemcpyAsync(enc_saved, obs, (size_t)B_TT * in_dim * 2, cudaMemcpyDeviceToDevice, 0);
    mm_nt(obs, W_enc, enc_out, (int)B_TT, H, in_dim);
    bf16* x = enc_out;
    for (int i = 0; i < L; i++) {
        cudaMemcpyAsync(saved[i], x, (size_t)B_TT * H * 2, cudaMemcpyDeviceToDevice, 0);
        bf16* state_i = state + ((long)i * A + agent_off) * H;
        PrefixScan& s = scans[i];
        s.combined_ptr = combined[i]; s.state_ptr = state_i; s.input_ptr = saved[i];
        s.terminals_ptr = term;
        if (fused) {
            if (!puf_mtl::mingru_fused_forward_rm(stream, B, TT, H, saved[i], W[i], state_i,
                    term, s.scan_h.data, s.out.data, s.next_state.data, combined[i])) {
                fprintf(stderr, "eligible golden fused forward rejected\n"); exit(2);
            }
        } else {
            mm_nt(saved[i], W[i], combined[i], B_TT, 3 * H, H);
            scan_forward(s, use_row(s));
        }
        x = s.out.data;
    }
    cudaMemcpyAsync(dec_saved, x, (size_t)B_TT * H * 2, cudaMemcpyDeviceToDevice, 0);
    mm_nt(x, W_dec, dec_out, (int)B_TT, od1, H);
    PUF_LAUNCH(assemble_decoder_grad, grid(B_TT * od1), BLOCK_SIZE, 0, 0)(
        dec_grad_out, dlogits, dvalue, (int)B_TT, od, od1);
    mm_tn(dec_grad_out, dec_saved, dec_wgrad, od1, H, (int)B_TT);
    mm_nn(dec_grad_out, W_dec, dec_grad_in, (int)B_TT, H, od1);
    bf16* grad = dec_grad_in;
    for (int i = L - 1; i >= 0; i--) {
        PrefixScan& s = scans[i];
        if (fused) {
            if (!puf_mtl::mingru_fused_backward_rm(stream, B, TT, H, saved[i], W[i], term,
                    s.scan_h.data, grad, gnext, s.grad_combined.data, s.grad_state.data,
                    s.grad_input.data, grad_input_buf, nullptr, combined[i])) {
                fprintf(stderr, "eligible golden fused backward rejected\n"); exit(2);
            }
            mm_tn(s.grad_combined.data, saved[i], wgrad[i], 3 * H, H, B_TT);
        } else {
            scan_backward(s, use_row(s), grad, gnext);
            mm_tn(s.grad_combined.data, saved[i], wgrad[i], 3 * H, H, (int)B_TT);
            mm_nn(s.grad_combined.data, W[i], grad_input_buf, (int)B_TT, H, 3 * H);
            int n = (int)numel(s.grad_input.shape);
            PUF_LAUNCH(add_kernel, grid(n), BLOCK_SIZE, 0, 0)(grad_input_buf, s.grad_input.data, n);
        }
        grad = grad_input_buf;
    }
    mm_tn(grad, enc_saved, enc_wgrad, H, in_dim, (int)B_TT);
    cudaDeviceSynchronize();

    check<bf16>(c, cs, @"decoder", dec_out, POLICY_TOL);
    for (int i = 0; i < L; i++) {
        check<bf16>(c, cs, [NSString stringWithFormat:@"layer_%d_out", i], scans[i].out.data, POLICY_TOL);
        check<bf16>(c, cs, [NSString stringWithFormat:@"layer_%d_next_state", i],
            scans[i].next_state.data, POLICY_TOL);
    }
    check<bf16>(c, cs, @"param_0_grad", enc_wgrad, POLICY_TOL);
    check<bf16>(c, cs, @"param_1_grad", dec_wgrad, POLICY_TOL);
    for (int i = 0; i < L; i++) {
        check<bf16>(c, cs, [NSString stringWithFormat:@"param_%d_grad", 2 + i], wgrad[i], POLICY_TOL);
        check<bf16>(c, cs, [NSString stringWithFormat:@"layer_%d_grad_state", i],
            scans[i].grad_state.data, POLICY_TOL);
    }
    check<bf16>(c, cs, @"encoder_grad", grad, POLICY_TOL);
    for (int i = 0; i < L; i++) {
        cudaFree(saved[i]); cudaFree(combined[i]); cudaFree(wgrad[i]); cudaFree(W[i]);
        free_scan(scans[i]);
    }
    for (void* p : {(void*)obs, (void*)state, (void*)term, (void*)dlogits, (void*)dvalue,
             (void*)gnext, (void*)W_enc, (void*)W_dec, (void*)enc_out, (void*)enc_saved,
             (void*)enc_wgrad, (void*)dec_out, (void*)dec_grad_out, (void*)dec_saved,
             (void*)dec_grad_in, (void*)dec_wgrad, (void*)grad_input_buf})
        cudaFree(p);
}

// ---------------------------------------------------------------------------
// PPO: bf16 inputs widened, fp32 math (exp/log), bf16 importance/values out.
// fp32 outputs carry MSL-vs-CUDA libm differences (FP32 gate bounds: ~1e-6
// relative); bf16 outputs at most one rounding flip.
// ---------------------------------------------------------------------------
static void run_ppo(const Case& c, const std::string& cs) {
    std::vector<int> sizes = c.raw<int>(@"act_sizes");
    int B = c.I(@"B"), T = c.I(@"T"), NT = B * T, A = [c.shape(@"decoder")[2] intValue] - 1;
    g_num_atns = (int)sizes.size();
    bf16 *logits = c.put<bf16>(@"decoder"), *oldlp = c.put<bf16>(@"old_logprobs");
    bf16* mask = c.put<bf16>(@"mask");
    float* act = c.put<float>(@"actions");
    float *lp = dalloc<float>((long)NT * A), *newlp = dalloc<float>(NT);
    bf16 *imp = dalloc<bf16>(NT), *v = dalloc<bf16>(NT);
    int* sz = c.put<int>(@"act_sizes");
    Prec decoder{logits, {B, T, A + 1}}, empty{};
    PUF_LAUNCH(cache_imp_and_v, grid(NT), 256, 0, 0)(decoder, act, oldlp, mask, empty, sz, imp, v, lp, newlp);
    cudaDeviceSynchronize();
    const Tol F32_TOL = {64, 1e-6};  // 64 fp32 ulps = 2^-17 relative
    check<float>(c, cs, @"cached_logps", lp, F32_TOL);
    check<float>(c, cs, @"cached_new_logprobs", newlp, F32_TOL);
    check<bf16>(c, cs, @"importance", imp, ELEM_TOL);
    check<bf16>(c, cs, @"live_values", v, EXACT);
    float* parts = dalloc<float>((long)grid(NT) * 8);
    float* loss = c.put<float>(@"losses_in");
    float* ec = dalloc<float>(1);
    ec[0] = c.F(@"ent_coef");
    bf16 *advs = c.put<bf16>(@"advantages"), *oldv = c.put<bf16>(@"old_values");
    bf16* rets = c.put<bf16>(@"returns");
    PPOKernelArgs a{lp, nullptr, newlp, logits, nullptr, logits + A, sz, mask, g_num_atns,
        c.F(@"clip_coef"), c.F(@"vf_clip_coef"), c.F(@"vf_coef"), ec, T, A, B, false};
    PPOGraphArgs g{imp, act, oldlp, advs, oldv, rets};
    PUF_LAUNCH(ppo_loss_compute, grid(NT), 256, 0, 0)(parts, a, g);
    PUF_LAUNCH(ppo_loss_reduce, 1, 8, 0, 0)(loss, parts, grid(NT));
    cudaDeviceSynchronize();
    // dlogits / dvalue: fp32 math on the above. An element whose bf16
    // importance flipped by one ulp moves by up to 2^-8 of itself (rel_tol);
    // otherwise libm-level differences, as in the FP32 gate (1e-6 x max).
    // abs 1e-9: the FP32 gate's dlogits atol (NT > 256), for near-zero
    // entries where fp32 libm differences in the softmax cancel.
    const Tol GRAD_TOL = {256, 1e-6, 1.0 / 256, 1e-9};
    check<float>(c, cs, @"dlogits", lp, GRAD_TOL);
    check<float>(c, cs, @"dvalue", newlp, GRAD_TOL);
    check<float>(c, cs, @"loss_partials", parts, GRAD_TOL);
    check<float>(c, cs, @"losses", loss, GRAD_TOL);
    for (void* p : {(void*)logits, (void*)oldlp, (void*)mask, (void*)act, (void*)lp, (void*)newlp,
             (void*)imp, (void*)v, (void*)sz, (void*)parts, (void*)loss, (void*)ec, (void*)advs,
             (void*)oldv, (void*)rets})
        cudaFree(p);
}

// ---------------------------------------------------------------------------
// Muon (algo.cu muon_step in BF16: bf16 gradient / NS buffers, fp32 momentum
// and master weights; then train's cast(param, master)).
// ---------------------------------------------------------------------------
static void ref_ns_tail(bf16* x, int R, int C, const float* coeff, float eps, float* partials,
        float* nsnorm, std::vector<void*>& owned) {
    int ne = R * C, M = std::min(R, C);
    bf16 *xb = dalloc<bf16>(ne), *gram = dalloc<bf16>((long)M * M), *gb = dalloc<bf16>((long)M * M);
    owned.push_back(xb); owned.push_back(gram); owned.push_back(gb);
    int nblk = std::min(grid(ne), 256);
    PUF_LAUNCH(muon_sum_sq_partials, nblk, 256, 0, 0)(partials, x, ne);
    PUF_LAUNCH(muon_sum_sq_reduce, 1, 256, 0, 0)(nsnorm, partials, nblk);
    PUF_LAUNCH(muon_l2_normalize, grid(ne), 256, 0, 0)(x, nsnorm, eps, ne);
    for (int i = 0; i < 5; i++) {
        bf16 *src = i % 2 ? xb : x, *dst = i % 2 ? x : xb;
        if (R > C) dense_gemm(CUBLAS_OP_T, CUBLAS_OP_N, C, C, R, src, src, gram);
        else dense_gemm(CUBLAS_OP_N, CUBLAS_OP_T, R, R, C, src, src, gram);
        cudaMemcpyAsync(gb, gram, (size_t)M * M * 2, cudaMemcpyDeviceToDevice, 0);
        dense_gemm(CUBLAS_OP_N, CUBLAS_OP_N, M, M, M, gram, gram, gb, coeff[i * 3 + 2], coeff[i * 3 + 1]);
        cudaMemcpyAsync(dst, src, (size_t)ne * 2, cudaMemcpyDeviceToDevice, 0);
        if (R > C) dense_gemm(CUBLAS_OP_N, CUBLAS_OP_N, R, C, C, src, gb, dst, 1, coeff[i * 3]);
        else dense_gemm(CUBLAS_OP_N, CUBLAS_OP_N, R, C, R, gb, src, dst, 1, coeff[i * 3]);
    }
    float scale = sqrtf(fmaxf(1, (float)R / (float)C));
    PUF_LAUNCH(muon_store_update, grid(ne), 256, 0, 0)(x, xb, scale, ne);
}

static void run_muon(const Case& c, const std::string& cs) {
    std::vector<void*> owned;
    long n = (long)c.count(@"grads");
    int nb = std::min(grid(n), 256);
    bf16* gc = c.put<bf16>(@"grads");
    float *wb = c.put<float>(@"params_in"), *mb = c.put<float>(@"momentum_in");
    bf16* param = c.put<bf16>(@"param_bf16_in");
    float *partials = dalloc<float>(256), *norm = dalloc<float>(1), *nsnorm = dalloc<float>(1);
    float* lr = dalloc<float>(1);
    lr[0] = c.F(@"lr");
    owned.insert(owned.end(), {gc, wb, mb, param, partials, norm, nsnorm, lr});
    // Weight kernel isolated from NS GEMM drift: CUDA's own bf16 update.
    float* iso_w = c.put<float>(@"params_in");
    bf16* cuda_update = c.put<bf16>(@"update");
    owned.push_back(iso_w); owned.push_back(cuda_update);
    PUF_LAUNCH(muon_weight_update, grid(n), 256, 0, 0)(iso_w, cuda_update, lr, c.F(@"weight_decay"), (int)n);
    cudaDeviceSynchronize();
    std::vector<float> pout = c.raw<float>(@"params_out");
    check<float>(c, cs + " [cuda update]", @"params_out", iso_w, EXACT);

    PUF_LAUNCH(muon_sum_sq_partials, nb, 256, 0, 0)(partials, gc, (int)n);
    PUF_LAUNCH(muon_sum_sq_reduce, 1, 256, 0, 0)(norm, partials, nb);
    PUF_LAUNCH(muon_clip_nesterov, grid(n), 256, 0, 0)(mb, gc, norm, c.F(@"max_grad_norm"),
        c.F(@"clip_eps"), c.F(@"momentum"), (int)n);
    std::vector<float> coeff = c.raw<float>(@"ns_coefficients");
    cudaDeviceSynchronize();
    bf16* gc_batch = dalloc<bf16>(n);
    owned.push_back(gc_batch);
    memcpy(gc_batch, gc, n * 2);
    // Clipped Nesterov output (bf16 gradient), before NS: no GEMM involved.
    std::vector<puf_mtl::MuonNSMatrix> mats;
    bool all_eligible = true;
    for (int p = 0;; p++) {
        NSString* name = [NSString stringWithFormat:@"param_%d_in", p];
        if (!c.has(name)) break;
        NSArray* shape = c.shape(name);
        if (shape.count < 2) continue;
        int R = [shape[0] intValue], C = 1;
        for (NSUInteger d = 1; d < shape.count; d++) C *= [shape[d] intValue];
        int off = c.I([NSString stringWithFormat:@"param_%d_offset", p]);
        ref_ns_tail(gc + off, R, C, coeff.data(), c.F(@"ns_eps"), partials, nsnorm, owned);
        all_eligible = all_eligible && puf_mtl::muon_ns5_eligible(R, C);
        mats.push_back({gc_batch + off, R, C, sqrtf(fmaxf(1, (float)R / (float)C))});
    }
    PUF_LAUNCH(muon_weight_update, grid(n), 256, 0, 0)(wb, gc, lr, c.F(@"weight_decay"), (int)n);
    PUF_LAUNCH(cast, grid(n), 256, 0, 0)(param, wb, (int)n);
    cudaDeviceSynchronize();
    if (all_eligible && mats.size() <= 8) {
        float cf[15];
        for (int i = 0; i < 15; i++) cf[i] = coeff[i];
        bool ran = puf_mtl::muon_ns5_batch_rm(nullptr, mats.data(), (int)mats.size(), cf, c.F(@"ns_eps"));
        cudaDeviceSynchronize();
        long bad = 0;
        for (long i = 0; i < n; i++) bad += gc[i].x != gc_batch[i].x;
        printf("  %-30s muon_ns5_batch_rm vs unfused bf16 tail: ran=%d differing=%ld%s\n",
            cs.c_str(), (int)ran, bad, ran && bad == 0 ? "" : "  FAIL");
        if (!ran || bad) g_fail++;
    }
    // grad_sum_sq: fp32 reduction of bf16 squares, same tree -> exact.
    bool all_wide = !all_eligible && !mats.empty() && mats.size() <= 8; size_t wide_bytes=0;
    for (const auto& mat : mats) {
        all_wide = all_wide && puf_mtl::muon_ns_wide_eligible(mat.R, mat.C);
        wide_bytes += puf_mtl::muon_ns_wide_scratch_bytes(mat.R, mat.C);
    }
    if (all_wide) {
        void* scratch=dalloc<unsigned char>(wide_bytes); owned.push_back(scratch);
        bool ran=puf_mtl::muon_ns_wide_rm(nullptr,mats.data(),(int)mats.size(),coeff.data(),
            c.F(@"ns_eps"),scratch,wide_bytes);
        cudaDeviceSynchronize();
        if (!ran || memcmp(gc,gc_batch,n*2)!=0) {
            fprintf(stderr,"golden wide Muon rejected or differs from generic tail\n"); g_fail++;
        }
        check<bf16>(c,cs+" [wide CUDA]",@"update",gc_batch, Tol{8, 1.0/32});
    }
    check<float>(c, cs, @"grad_sum_sq", norm, EXACT);
    check<float>(c, cs, @"momentum_out", mb, EXACT);
    // update: five NS iterations of bf16 GEMMs (each rounding to bf16)
    // amplify GEMM rounding flips; held to 2^-5 of max|update| (the update is
    // orthogonalized: entries O(1), and lr * 2^-5 ~ 2e-3 absolute on weights).
    const Tol NS_TOL = {8, 1.0 / 32};
    check<bf16>(c, cs, @"update", gc, NS_TOL);
    // Weights move by lr * update: the update tolerance scaled by lr.
    double lr_v = c.F(@"lr");
    std::vector<float> ref_up(n);
    {
        std::vector<bf16> u = c.raw<bf16>(@"update");
        for (long i = 0; i < n; i++) ref_up[i] = puf_bf16_bits_to_float(u[i].x);
    }
    double max_up = 0;
    for (float u : ref_up) max_up = std::max(max_up, (double)fabsf(u));
    {
        double max_w = 0;
        for (float w : pout) max_w = std::max(max_w, (double)fabsf(w));
        Tol W_TOL = {2, NS_TOL.norm_tol * lr_v * max_up / std::max(max_w, 1e-30)};
        check<float>(c, cs, @"params_out", wb, W_TOL);
        for (int p = 0;; p++) {
            NSString* key = [NSString stringWithFormat:@"param_%d_out", p];
            if (!c.has(key)) break;
            int off = c.I([NSString stringWithFormat:@"param_%d_offset", p]);
            std::vector<float> rp = c.raw<float>(key);
            double mw = 0;
            for (float w : rp) mw = std::max(mw, (double)fabsf(w));
            Tol PT = {2, NS_TOL.norm_tol * lr_v * max_up / std::max(mw, 1e-30)};
            check<float>(c, cs, key, wb + off, PT);
        }
        // bf16 refresh of the master: one more rounding (+1 ulp).
        Tol P16 = {3, W_TOL.norm_tol};
        check<bf16>(c, cs, @"param_bf16_out", param, P16);
    }
    for (void* p : owned) cudaFree(p);
}

// ---------------------------------------------------------------------------
// Sampling: Philox / actions exact; bf16 logprob (fp32 log-softmax of bf16
// logits, rounded once) within one flip; value is a copy -> exact.
// ---------------------------------------------------------------------------
static void run_sample(const Case& c, const std::string& cs) {
    int B = c.I(@"B");
    unsigned long long seed = [c.scalars[@"seed"] unsignedLongLongValue];
    std::vector<uint8_t> rng_in = c.raw<uint8_t>(@"rng_in"), rng_out = c.raw<uint8_t>(@"rng_out");
    std::vector<int> act_sizes = c.raw<int>(@"act_sizes");
    int num_atns = (int)act_sizes.size();
    int fused = (int)(c.count(@"decoder") / B), mask_stride = (int)(c.count(@"masks") / B);
    g_num_atns = num_atns;
    curandStatePhilox4_32_10_t* st = dalloc<curandStatePhilox4_32_10_t>(B);
    PUF_LAUNCH(rng_init, grid(B), 256, 0, 0)(st, (uint64_t)seed, B);
    cudaDeviceSynchronize();
    // The dumped input state is curand_init + agent%7 discarded uniforms: the
    // discards are checked in test_kernels; here the state is loaded as dumped.
    memcpy(st, rng_in.data(), (size_t)B * 64);
    Prec dec_p = {c.put<bf16>(@"decoder"), {B, fused}}, logstd = {};
    int* d_act = c.put<int>(@"act_sizes");
    bf16* d_mask = c.put<bf16>(@"masks");
    float *actions = dalloc<float>((long)B * num_atns), *env_actions = dalloc<float>((long)B * num_atns);
    bf16 *logprobs = dalloc<bf16>(B), *values = dalloc<bf16>(B);
    PUF_LAUNCH(sample_logits, grid(B), 256, 0, 0)(dec_p, logstd, d_act, actions, env_actions,
        logprobs, values, st, d_mask, mask_stride);
    cudaDeviceSynchronize();
    long bad = 0;
    for (int i = 0; i < B; i++) bad += memcmp(&st[i], rng_out.data() + (size_t)i * 64, 64) != 0;
    printf("  %-30s %-24s n=%-8d mismatches=%ld%s\n", cs.c_str(), "rng_out", B, bad, bad ? "  FAIL" : "");
    if (bad) g_fail++;
    check<float>(c, cs, @"actions", actions, EXACT);
    check<float>(c, cs, @"env_actions", env_actions, EXACT);
    check<bf16>(c, cs, @"logprobs", logprobs, ELEM_TOL);
    check<bf16>(c, cs, @"values", values, EXACT);
    for (void* p : {(void*)st, (void*)dec_p.data, (void*)d_act, (void*)d_mask, (void*)actions,
             (void*)env_actions, (void*)logprobs, (void*)values})
        cudaFree(p);
}

static void run_dir(const char* run_dir) {
    NSString* root = [NSString stringWithUTF8String:run_dir];
    NSArray* names = [[[NSFileManager defaultManager] contentsOfDirectoryAtPath:root error:nil]
        sortedArrayUsingSelector:@selector(compare:)];
    printf("== BF16 golden (%s)\n", run_dir);
    int count = 0;
    for (NSString* name in names) {
        NSString* dir = [root stringByAppendingPathComponent:name];
        if (![[NSFileManager defaultManager] fileExistsAtPath:[dir stringByAppendingPathComponent:@"manifest.json"]])
            continue;
        std::string cs = name.UTF8String;
        if ([name isEqual:@"train_epoch3"]) continue;  // tests/metal/run_train_replay_bf16.sh
        Case c = load_case(dir);
        if ([name hasPrefix:@"gemm_"]) run_gemm(c, cs);
        else if ([name isEqual:@"cast_bf16_probe"]) run_cast_probe(c, cs);
        else if ([name hasPrefix:@"adv_"]) run_advantage(c, cs);
        else if ([name hasPrefix:@"gate_"]) run_gate(c, cs);
        else if ([name hasPrefix:@"scan_"]) {
            bool row = [c.scalars[@"operation"] hasPrefix:@"mingru_scan_forward_row"];
            run_scan(c, cs, row);
            run_scan(c, cs + (row ? " [plain]" : " [row]"), !row);
        } else if ([name hasPrefix:@"policy_"]) {
            if (!c.has(@"dlogits")) { printf("  %-30s skipped (no dlogits)\n", cs.c_str()); continue; }
            run_policy(c, cs);
        } else if ([name hasPrefix:@"ppo_"]) run_ppo(c, cs);
        else if ([name hasPrefix:@"muon_"]) run_muon(c, cs);
        else if ([name hasPrefix:@"sample_"]) run_sample(c, cs);
        else { printf("  %-30s UNHANDLED  FAIL\n", cs.c_str()); g_fail++; continue; }
        count++;
    }
    printf("cases: %d\n", count);
    if (count == 0) g_fail++;
}

int main(int argc, char** argv) {
    @autoreleasepool {
        if (argc < 2) {
            fprintf(stderr, "usage: test_bf16_golden GOLDEN_BF16_RUN_DIR ...\n");
            return 2;
        }
        cudaStream_t s = 0;
        const size_t ws_bytes = 4 * 1024 * 1024;
        void* ws = nullptr;
        cudaMalloc(&ws, ws_bytes);
        cublasCreate(&g_h);
        cublasSetStream(g_h, s);
        cublasSetWorkspace(g_h, ws, ws_bytes);
        for (int i = 1; i < argc; i++) run_dir(argv[i]);
        printf(g_fail ? "FAILED: %d checks\n" : "ALL PASS\n", g_fail);
        return g_fail ? 1 : 0;
    }
}
