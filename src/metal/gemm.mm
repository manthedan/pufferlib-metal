// Metal backend GEMM: cuBLAS shim (column-major cublasGemmEx) on top of a
// row-major dispatcher choosing between Metal 4 tensor_ops (matmul2d), a
// bounds-checked tiled kernel, and split-K for long reductions.
// Kernels: shaders/gemm.metal.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <set>
#include <string>

#include "gemm.h"
#include "shaders/wide_config.h"
#include "compat/cublas_v2.h"

namespace puf_mtl {

struct GemmArgs {
    uint64_t A, B, C, partial;
    int32_t M, N, K;
    int32_t lda, ldb, ldc;
    int32_t ta, tb;
    int32_t k_chunk;
    float alpha, beta;
};
static_assert(sizeof(GemmArgs) == 80, "GemmArgs layout must match gemm.metal");

struct GemmReduceArgs {
    uint64_t partial, C;
    int32_t M, N, ldc, splits;
    float alpha, beta;
};

struct TensorGemmArgs {
    uint64_t A, B, C;
    int32_t M, N, K, pad;
};

static bool aligned16(const void* p) { return ((uintptr_t)p & 15) == 0; }

static thread_local GemmPath g_last_path = GEMM_NONE;
GemmPath gemm_last_path() { return g_last_path; }

// Kernel name for storage type dt (0 = f32, 1 = bf16): "<base>_f32" / "<base>_bf16".
static Pipeline* gemm_pipeline(const char* base, int dt) {
    char name[96];
    snprintf(name, sizeof(name), "%s_%s", base, dt ? "bf16" : "f32");
    return pipeline(name);
}

static bool large_linear_enabled() {
    static const bool enabled = [] {
        const char* e = getenv("PUF_METAL_GEMM_LARGE_LINEAR");
        return !(e && strcmp(e, "0") == 0);
    }();
    return enabled;
}

// dt selects the storage type of A, B and C (cublasGemmEx CUDA_R_32F or
// CUDA_R_16BF); accumulation and split-K partials are fp32 in both.
// With `defer` (gemm_rm_deferred): split paths place their partials at
// workspace + ws_off and store the reduction in *defer instead of encoding
// it; single-dispatch paths that use no workspace are encoded as usual. A
// split path whose partials do not fit in [ws_off, ws_bytes) encodes
// nothing and returns GEMM_NONE. Split counts always follow ws_bytes, so
// every path and result is the same as without `defer`.
static GemmPath gemm_rm_impl(cudaStream_t s, int dt, bool ta, bool tb, int M, int N, int K,
        float alpha, const void* A, int lda, const void* B, int ldb,
        float beta, void* C, int ldc,
        void* workspace, size_t ws_bytes, const GemmOpts* opts,
        size_t ws_off = 0, GemmDeferred* defer = nullptr) {
    GemmOpts def;
    if (!opts) opts = &def;
    if (defer) *defer = GemmDeferred();
    if (M <= 0 || N <= 0) return GEMM_NONE;
    if (K < 0) fatal("gemm: negative K");
    // Partials of `splits` slices at workspace + ws_off; false: do not fit.
    auto partials_at = [&](long splits, long mn, uint64_t* addr) {
        if (ws_off % 16) fatal("gemm: workspace offset %zu not 16-byte aligned", ws_off);
        if (ws_off + (size_t)splits * (size_t)mn * sizeof(float) > ws_bytes) return false;
        *addr = gpu_addr((char*)workspace + ws_off, "gemm workspace");
        if (defer) defer->ws_used = (size_t)splits * (size_t)mn * sizeof(float);
        return true;
    };
    // Encodes the split-K reduction now, or leaves it in *defer.
    auto reduce = [&](Pipeline* p, const GemmReduceArgs& r, long mn) {
        unsigned grid = (unsigned)((mn + 255) / 256);
        if (defer) {
            defer->pipe = p;
            defer->grid = grid;
            static_assert(sizeof(GemmReduceArgs) == sizeof(defer->args), "GemmDeferred args");
            memcpy(defer->args, &r, sizeof(r));
            return;
        }
        dispatch(s, p, dim3(grid), dim3(256), r);
    };

    // alpha == 0: BLAS does not evaluate op(A) @ op(B), so NaN/Inf in A or B
    // must not reach C. The reduce kernel with zero splits computes
    // C = beta * C (and never reads C when beta == 0).
    if (alpha == 0.0f) {
        static Pipeline* p_red0_dt[2] = {gemm_pipeline("puf_gemm_reduce", 0),
            gemm_pipeline("puf_gemm_reduce", 1)};
        Pipeline* p_red0 = p_red0_dt[dt];
        long mn0 = (long)M * N;
        GemmReduceArgs r = {0, gpu_addr(C, "gemm C"), M, N, ldc, 0, 0.0f, beta};
        dispatch(s, p_red0, dim3((unsigned)((mn0 + 255) / 256)), dim3(256), r);
        return GEMM_SCALE;
    }

    static const bool wide_tn = [] {
        const char* e = getenv("PUF_METAL_GEMM_WIDE_TN");
        return !(e && strcmp(e, "0") == 0);
    }();
    const bool wide_dw = wide_tn && opts->allow_wide_tn && ta && !tb && (N == 128 || N == 256)
        && M == 3 * N && K > 0 && K % 16 == 0;
    const int wide_tm = dt ? PUF_WIDE_TN_M_BF16 : PUF_WIDE_TN_M_F32;
    static const bool large_tn = [] {
        const char* e = getenv("PUF_METAL_GEMM_LARGE_TN");
        return !(e && strcmp(e, "0") == 0);
    }();
    const bool large_dw = large_tn && opts->allow_large_tn && ta && !tb
        && (N == 512 || N == 1024) && M == 3 * N && K > 0 && K % 16 == 0;
    const bool large_ragged = large_tn && opts->allow_large_tn && opts->allow_tensor_ops
        && ta && !tb && (M == 512 || M == 1024) && N >= 32 && N % 32 != 0
        && K > 0 && K % 16 == 0 && lda == M && ldb == N && ldc == N
        && aligned16(A) && aligned16(B) && aligned16(C);
    const int large_tm = dt ? PUF_LARGE_TN_M_BF16 : PUF_LARGE_TN_M_F32;
    const int large_n = large_dw ? (N == 512 ?
        (dt ? PUF_LARGE_TN_N512_BF16 : PUF_LARGE_TN_N512_F32) : PUF_LARGE_TN_N1024) : PUF_LARGE_TN_N;

    // Weight-gradient shape with tensor_ops-aligned output (M % 64, N % 32,
    // e.g. H=64 MinGRU dW (192, 64)): matmul2d split-K partials + fixed-order
    // reduce. ~2x faster than sg_tn there; a different summation order than
    // sg_tn, so PUF_METAL_TOPS_TN_SPLIT=0 restores it.
    static const bool tops_tn_split = [] {
        const char* e = getenv("PUF_METAL_TOPS_TN_SPLIT");
        return !(e && strcmp(e, "0") == 0);
    }();
    if (tops_tn_split && opts->allow_split_k && opts->allow_sg_tn && opts->allow_tensor_ops
            && ta && !tb && K >= 2048 && M % 64 == 0 && N % 32 == 0
            && lda == M && ldb == N && workspace
            && aligned16(A) && aligned16(B) && aligned16(workspace)) {
        constexpr long kTgs = 128, kMinK = 512;
        long tiles = (long)(N / 32) * (M / 64);
        long mn = (long)M * N;
        long splits = kTgs / tiles;
        long by_k = K / kMinK;
        long by_ws = (long)(ws_bytes / sizeof(float)) / mn;
        if (by_k < splits) splits = by_k;
        if (by_ws < splits) splits = by_ws;
        if (splits >= 2) {
            int chunk = (int)((K + splits - 1) / splits);
            chunk = (chunk + 63) / 64 * 64;
            splits = (K + chunk - 1) / chunk;
            struct { uint64_t A, B, partial; int32_t M, N, K, k_chunk; } ta_args = {
                gpu_addr(A, "gemm A"), gpu_addr(B, "gemm B"), 0, M, N, K, chunk};
            if (!partials_at(splits, mn, &ta_args.partial)) return GEMM_NONE;
            static Pipeline* p_tt_dt[2] = {gemm_pipeline("tensor_ops_gemm_tn_split", 0),
                gemm_pipeline("tensor_ops_gemm_tn_split", 1)};
            if (wide_dw) {
                Pipeline* p = gemm_pipeline("wide_gemm_tn_split", dt);
                dispatch(s, p, dim3(N / PUF_WIDE_TN_N, M / wide_tm, (unsigned)splits),
                    dim3(PUF_WIDE_TN_THREADS), ta_args);
            } else {
            dispatch(s, p_tt_dt[dt], dim3(N / 32, M / 64, (unsigned)splits), dim3(128), ta_args);
            }
            static Pipeline* p_red_dt[2] = {gemm_pipeline("puf_gemm_reduce", 0),
                gemm_pipeline("puf_gemm_reduce", 1)};
            GemmReduceArgs r = {ta_args.partial, gpu_addr(C, "gemm C"), M, N, ldc,
                (int32_t)splits, alpha, beta};
            reduce(p_red_dt[dt], r, mn);
            return GEMM_TOPS_TN_SPLIT;
        }
    }

    // Weight-gradient shape: opA = A^T with a long K and a small output.
    // The 64x64 tiled split-K below wastes most of each tile on M, N <= 128
    // and runs far below bandwidth; simdgroup MMAs on 16x16 tiles do not.
    if (opts->allow_split_k && opts->allow_sg_tn && ta && !tb && K >= 2048) {
        // Tuned on M4 (tests/metal/run_bench_gemm.sh): 16x16 tiles beat 32x32
        // (register pressure); ~128 threadgroups, >= 1024 K rows per slice.
        constexpr int kSgTnTile = 16;
        constexpr long kTgs = 512, kMinK = 1024;
        long tiles = (long)((N + kSgTnTile - 1) / kSgTnTile) * ((M + kSgTnTile - 1) / kSgTnTile);
        long mn = (long)M * N;
        if (tiles <= 256) {
            static Pipeline* p_sg_dt[2] = {gemm_pipeline("puf_gemm_tn_sg16", 0),
                gemm_pipeline("puf_gemm_tn_sg16", 1)};
            Pipeline* p_sg = p_sg_dt[dt];
            long splits = kTgs / tiles;
            long by_k = K / kMinK;
            long by_ws = workspace ? (long)(ws_bytes / sizeof(float)) / mn : 0;
            if (by_k < splits) splits = by_k;
            if (by_ws < splits) splits = by_ws;
            GemmArgs a = {gpu_addr(A, "gemm A"), gpu_addr(B, "gemm B"), gpu_addr(C, "gemm C"),
                0, M, N, K, lda, ldb, ldc, 1, 0, K, alpha, beta};
            unsigned gx = (unsigned)((N + kSgTnTile - 1) / kSgTnTile);
            unsigned gy = (unsigned)((M + kSgTnTile - 1) / kSgTnTile);
            if (splits < 2) {
                dispatch(s, p_sg, dim3(gx, gy, 1), dim3(128), a);
                return GEMM_SG_TN;
            }
            int chunk = (int)((K + splits - 1) / splits);
            chunk = (chunk + 31) / 32 * 32;
            splits = (K + chunk - 1) / chunk;
            if (!partials_at(splits, mn, &a.partial)) return GEMM_NONE;
            a.k_chunk = chunk;
            dispatch(s, p_sg, dim3(gx, gy, (unsigned)splits), dim3(128), a);
            static Pipeline* p_red_dt[2] = {gemm_pipeline("puf_gemm_reduce", 0),
                gemm_pipeline("puf_gemm_reduce", 1)};
            Pipeline* p_red = p_red_dt[dt];
            GemmReduceArgs r = {a.partial, a.C, M, N, ldc, (int32_t)splits, alpha, beta};
            reduce(p_red, r, mn);
            return GEMM_SG_TN;
        }
    }

    // Small N (decoder forward: actions + 1 columns): one thread per output
    // row, bit-identical to the tiled kernel below, which every eligible
    // shape would otherwise take (N < 32 rules out tensor_ops, K <= 256 rules
    // out split-K). PUF_METAL_GEMM_SMALL_N=0 restores the tiled kernel.
    static const bool small_n = [] {
        const char* e = getenv("PUF_METAL_GEMM_SMALL_N");
        return !(e && strcmp(e, "0") == 0);
    }();
    if (small_n && opts->allow_small_n && !ta && N <= 8 && K >= 16 && K <= 256 && K % 16 == 0) {
        static Pipeline* p_sn[2] = {gemm_pipeline("puf_gemm_small_n", 0),
            gemm_pipeline("puf_gemm_small_n", 1)};
        GemmArgs a = {gpu_addr(A, "gemm A"), gpu_addr(B, "gemm B"), gpu_addr(C, "gemm C"), 0,
            M, N, K, lda, ldb, ldc, 0, tb ? 1 : 0, 1, alpha, beta};
        // k_chunk selects the kernel's four-element A loads.
        if (aligned16(A) && lda % 4 == 0) a.k_chunk = 4;
        dispatch(s, p_sn[dt], dim3((unsigned)(((long)M * N + 127) / 128)), dim3(128), a);
        return GEMM_SMALL_N;
    }

    // Metal 4 tensor_ops fast path.
    bool dense = lda == (ta ? M : K) && ldb == (tb ? K : N) && ldc == N;
    if (opts->allow_tensor_ops && K > 0 && alpha == 1.0f && beta == 0.0f && dense
            && M % 64 == 0 && N % 32 == 0 && !(ta && tb)
            && aligned16(A) && aligned16(B) && aligned16(C)) {
        static Pipeline* p_nt[2] = {gemm_pipeline("tensor_ops_gemm_nt", 0),
            gemm_pipeline("tensor_ops_gemm_nt", 1)};
        static Pipeline* p_nn[2] = {gemm_pipeline("tensor_ops_gemm_nn", 0),
            gemm_pipeline("tensor_ops_gemm_nn", 1)};
        static Pipeline* p_tn[2] = {gemm_pipeline("tensor_ops_gemm_tn", 0),
            gemm_pipeline("tensor_ops_gemm_tn", 1)};
        if (large_dw) {
            TensorGemmArgs a = {gpu_addr(A, "gemm A"), gpu_addr(B, "gemm B"),
                gpu_addr(C, "gemm C"), M, N, K, 0};
            dispatch(s, gemm_pipeline("wide_large_tn", dt),
                dim3((N + large_n - 1) / large_n, (M + large_tm - 1) / large_tm),
                dim3(PUF_LARGE_TN_THREADS), a);
            return GEMM_WIDE_TN;
        }
        if (wide_dw) {
            TensorGemmArgs a = {gpu_addr(A, "gemm A"), gpu_addr(B, "gemm B"),
                gpu_addr(C, "gemm C"), M, N, K, 0};
            dispatch(s, gemm_pipeline("wide_gemm_tn", dt),
                dim3(N / PUF_WIDE_TN_N, M / wide_tm), dim3(PUF_WIDE_TN_THREADS), a);
            return GEMM_WIDE_TN;
        }
        // Learner-size recurrent projections only; small rollout tiles keep
        // their original pipelines and scheduling.
        if (large_linear_enabled() && opts->allow_large_linear && !ta && tb && M >= 4096
                && (K == 512 || K == 1024) && N == 3 * K) {
            TensorGemmArgs a = {gpu_addr(A, "gemm A"), gpu_addr(B, "gemm B"),
                gpu_addr(C, "gemm C"), M, N, K, 0};
            dispatch(s, gemm_pipeline("wide_large_nt", dt),
                dim3((N + PUF_LARGE_LINEAR_N - 1) / PUF_LARGE_LINEAR_N,
                    (M + PUF_LARGE_LINEAR_M - 1) / PUF_LARGE_LINEAR_M),
                dim3(PUF_LARGE_LINEAR_THREADS), a);
            return GEMM_LARGE_LINEAR;
        }
        Pipeline* p = ta ? p_tn[dt] : (tb ? p_nt[dt] : p_nn[dt]);
        TensorGemmArgs a = {gpu_addr(A, "gemm A"), gpu_addr(B, "gemm B"),
            gpu_addr(C, "gemm C"), M, N, K, 0};
        dispatch(s, p, dim3(N / 32, M / 64), dim3(128), a);
        return GEMM_TENSOR_OPS;
    }

    static Pipeline* p_gemm_dt[2] = {gemm_pipeline("puf_gemm", 0), gemm_pipeline("puf_gemm", 1)};
    Pipeline* p_gemm = p_gemm_dt[dt];
    GemmArgs a = {gpu_addr(A, "gemm A"), gpu_addr(B, "gemm B"), gpu_addr(C, "gemm C"), 0,
        M, N, K, lda, ldb, ldc, ta ? 1 : 0, tb ? 1 : 0, K > 0 ? K : 1, alpha, beta};
    unsigned gx = (unsigned)((N + 63) / 64), gy = (unsigned)((M + 63) / 64);

    // Split-K: few output tiles, long reduction (e.g. weight grads with
    // K = minibatch rows). Partials live in the cuBLAS workspace.
    long tiles = (long)gx * gy;
    long mn = (long)M * N;
    if (opts->allow_split_k && workspace && K >= 2048 && tiles < 64) {
        long by_tiles = 256 / tiles;
        long by_k = K / 512;
        long by_ws = (long)(ws_bytes / sizeof(float)) / mn;
        long splits = by_tiles < by_k ? by_tiles : by_k;
        if (by_ws < splits) splits = by_ws;
        if (splits >= 2) {
            int chunk = (int)((K + splits - 1) / splits);
            chunk = (chunk + 15) / 16 * 16;
            splits = (K + chunk - 1) / chunk;
            if (!partials_at(splits, mn, &a.partial)) return GEMM_NONE;
            a.k_chunk = chunk;
            if (large_ragged) {
                // Keep the scalar path's split count, k_chunk, FP32 partial
                // layout and reducer. Only the independent output tile changes.
                struct { uint64_t A, B, partial; int32_t M, N, K, k_chunk; } wa = {
                    a.A, a.B, a.partial, M, N, K, chunk};
                dispatch(s, gemm_pipeline("wide_large_tn_split", dt),
                    dim3((N + PUF_LARGE_TN_N - 1) / PUF_LARGE_TN_N,
                         (M + large_tm - 1) / large_tm, (unsigned)splits),
                    dim3(PUF_LARGE_TN_THREADS), wa);
            } else {
                dispatch(s, p_gemm, dim3(gx, gy, (unsigned)splits), dim3(256), a);
            }
            static Pipeline* p_red_dt[2] = {gemm_pipeline("puf_gemm_reduce", 0),
                gemm_pipeline("puf_gemm_reduce", 1)};
            Pipeline* p_red = p_red_dt[dt];
            GemmReduceArgs r = {a.partial, a.C, M, N, ldc, (int32_t)splits, alpha, beta};
            reduce(p_red, r, mn);
            return GEMM_SPLIT_K;
        }
    }
    // Only replace the unsplit scalar fallback: earlier split choices and
    // their accumulation order are unchanged. Full ascending-K fma chain.
    static const bool ragged = [] {
        const char* e = getenv("PUF_METAL_GEMM_RAGGED");
        return !(e && strcmp(e, "0") == 0);
    }();
    if (ragged && opts->allow_wide_tn && opts->allow_tensor_ops && ta && !tb && (M == 128 || M == 256)
            && N % 32 != 0 && N >= 32 && K > 0 && K % 16 == 0
            && alpha == 1.0f && beta == 0.0f && dense
            && aligned16(A) && aligned16(B) && aligned16(C)) {
        Pipeline* pr = gemm_pipeline("wide_gemm_tn", dt);
        TensorGemmArgs r = {a.A, a.B, a.C, M, N, K, 0};
        dispatch(s, pr, dim3((N + PUF_WIDE_TN_N - 1) / PUF_WIDE_TN_N, M / (dt ? PUF_WIDE_TN_M_BF16 : PUF_WIDE_TN_M_F32)), dim3(PUF_WIDE_TN_THREADS), r);
        return GEMM_WIDE_TN;
    }
    if (large_ragged && alpha == 1.0f && beta == 0.0f) {
        TensorGemmArgs r = {a.A, a.B, a.C, M, N, K, 0};
        dispatch(s, gemm_pipeline("wide_large_tn", dt),
            dim3((N + large_n - 1) / large_n, (M + large_tm - 1) / large_tm),
            dim3(PUF_LARGE_TN_THREADS), r);
        return GEMM_WIDE_TN;
    }
    dispatch(s, p_gemm, dim3(gx, gy, 1), dim3(256), a);
    return GEMM_TILED;
}

struct TensorGemmAddArgs {
    uint64_t A, B, C, D;
    int32_t M, N, K, pad;
};

static bool gemm_nn_add_impl(cudaStream_t s, int dt, int M, int N, int K,
        const void* A, const void* B, const void* D, void* C) {
    // Same eligibility as the tensor_ops branch of gemm_rm_impl (nn, dense).
    if (K <= 0 || M <= 0 || N <= 0 || M % 64 || N % 32 || D == C
            || !aligned16(A) || !aligned16(B) || !aligned16(C) || !aligned16(D)) {
        return false;
    }
    if (large_linear_enabled() && M >= 4096 && (N == 512 || N == 1024) && K == 3 * N) {
        TensorGemmAddArgs a = {gpu_addr(A, "gemm A"), gpu_addr(B, "gemm B"),
            gpu_addr(C, "gemm C"), gpu_addr(D, "gemm D"), M, N, K, 0};
        dispatch(s, gemm_pipeline("wide_large_nn_add", dt),
            dim3((N + PUF_LARGE_LINEAR_N - 1) / PUF_LARGE_LINEAR_N,
                (M + PUF_LARGE_LINEAR_M - 1) / PUF_LARGE_LINEAR_M),
            dim3(PUF_LARGE_LINEAR_THREADS), a);
        return true;
    }
    static Pipeline* p_dt[2] = {gemm_pipeline("tensor_ops_gemm_nn_add", 0),
        gemm_pipeline("tensor_ops_gemm_nn_add", 1)};
    Pipeline* p = p_dt[dt];
    TensorGemmAddArgs a = {gpu_addr(A, "gemm A"), gpu_addr(B, "gemm B"),
        gpu_addr(C, "gemm C"), gpu_addr(D, "gemm D"), M, N, K, 0};
    dispatch(s, p, dim3(N / 32, M / 64), dim3(128), a);
    return true;
}

bool gemm_nn_add_rm(cudaStream_t s, int M, int N, int K,
        const float* A, const float* B, const float* D, float* C) {
    return gemm_nn_add_impl(s, 0, M, N, K, A, B, D, C);
}

bool gemm_nn_add_rm(cudaStream_t s, int M, int N, int K,
        const __nv_bfloat16* A, const __nv_bfloat16* B, const __nv_bfloat16* D,
        __nv_bfloat16* C) {
    return gemm_nn_add_impl(s, 1, M, N, K, A, B, D, C);
}

struct MingruFusedFwdArgs {
    uint64_t x, W, state, terminals, scan_h, out, next_state;
    int32_t B, T, H, pad;
};
static_assert(sizeof(MingruFusedFwdArgs) == 72, "MingruFusedFwdArgs layout");

struct MingruFusedBwdArgs {
    uint64_t x, W, terminals, scan_h, grad_out, grad_next_state,
        grad_combined, grad_state, grad_input, grad_x;
    int32_t B, T, H, pad;
};
static_assert(sizeof(MingruFusedBwdArgs) == 96, "MingruFusedBwdArgs layout");

bool mingru_fused_eligible(int B, int T, int H, const prec_t* x, const prec_t* W) {
    static int enabled = -1;
    if (enabled < 0) {
        const char* e = getenv("PUF_METAL_FUSED_MINGRU");
        enabled = !(e && strcmp(e, "0") == 0);
    }
    // H=64 (mingru_fused_*_h64): PUF_METAL_FUSED_MINGRU64=0 disables it alone.
    static const bool enabled64 = [] {
        const char* e = getenv("PUF_METAL_FUSED_MINGRU64");
        return !(e && strcmp(e, "0") == 0);
    }();
    static const bool enabled_wide = [] {
        const char* e = getenv("PUF_METAL_FUSED_MINGRU_WIDE");
        return !(e && strcmp(e, "0") == 0);
    }();
    // 64-row tiles of whole sequences; the tensor_ops nt GEMM it replaces
    // needs M % 64 == 0 and N = 3H % 32 == 0. Indices are int: 3H * B*T fits.
    return enabled && (H == 32 || (H == 64 && enabled64) || ((H == 128 || H == 256) && enabled_wide)) && T > 0 && 64 % T == 0 && B > 0
        && ((long)B * T) % 64 == 0 && (long)B * T <= 0x7fffffffL / (3 * H)
        && aligned16(x) && aligned16(W);
}

static uint64_t addr_or_null(const void* p, const char* what) {
    return p ? gpu_addr(p, what) : 0;
}

struct MingruFused64FwdArgs {
    uint64_t x, W, state, terminals, scan_h, out, next_state, combined;
    int32_t B, T, H, pad;
};
static_assert(sizeof(MingruFused64FwdArgs) == 80, "MingruFused64FwdArgs layout");

struct MingruFused64BwdArgs {
    uint64_t x, W, terminals, scan_h, grad_out, grad_next_state,
        grad_combined, grad_state, grad_input, grad_x, combined;
    int32_t B, T, H, pad;
};
static_assert(sizeof(MingruFused64BwdArgs) == 104, "MingruFused64BwdArgs layout");

bool mingru_fused_forward_rm(cudaStream_t s, int B, int T, int H,
        const prec_t* x, const prec_t* W, const prec_t* state, const prec_t* terminals,
        prec_t* scan_h, prec_t* out, prec_t* next_state, prec_t* combined) {
    if (!mingru_fused_eligible(B, T, H, x, W)) return false;
    if (H >= 64) {
        if (!combined) return false;
#ifndef PRECISION_FLOAT
        // BF16 H=64 copies each projection chunk to `combined` in 16-byte words.
        if (H == 64 && !aligned16(combined)) return false;
#endif
        Pipeline* p64 = nullptr;
        if (H == 64) { static Pipeline* p = pipeline("mingru_fused_forward_h64"); p64 = p; }
        else p64 = pipeline(H == 128 ? "wide_mingru_forward_h128" : "wide_mingru_forward_h256");
        MingruFused64FwdArgs a = {gpu_addr(x, "mingru x"), gpu_addr(W, "mingru W"),
            gpu_addr(state, "mingru state"), addr_or_null(terminals, "mingru terminals"),
            gpu_addr(scan_h, "mingru scan_h"), gpu_addr(out, "mingru out"),
            gpu_addr(next_state, "mingru next_state"), gpu_addr(combined, "mingru combined"),
            B, T, H, 0};
        dispatch(s, p64, dim3((unsigned)((long)B * T / 64)), dim3(H == 64 ? 64 : (H == 128 ? PUF_WIDE_128_THREADS : PUF_WIDE_256_THREADS)), a);
        return true;
    }
    static Pipeline* p = pipeline("mingru_fused_forward");
    int rows = 64, threads = 128;
    Pipeline* selected = p;
#ifndef PRECISION_FLOAT
    static const bool small = [] {
        const char* e = getenv("PUF_METAL_MINGRU_FWD32");
        return !(e && strcmp(e, "0") == 0);
    }();
    if (small && T == 32) {
        static Pipeline* p32 = pipeline("mingru_fused_forward32");
        selected = p32; rows = 32; threads = 64;
    }
#endif
    MingruFusedFwdArgs a = {gpu_addr(x, "mingru x"), gpu_addr(W, "mingru W"),
        gpu_addr(state, "mingru state"), addr_or_null(terminals, "mingru terminals"),
        gpu_addr(scan_h, "mingru scan_h"), gpu_addr(out, "mingru out"),
        gpu_addr(next_state, "mingru next_state"), B, T, H, 0};
    dispatch(s, selected, dim3((unsigned)((long)B * T / rows)), dim3(threads), a);
    return true;
}

struct MingruDWReduceArgs { uint64_t partial, out; int32_t tiles, pad; };

// (tiles, 96, 32) partials -> (96, 32), ascending tile order per weight.
static void dw_reduce(cudaStream_t s, Pipeline* p, const float* partial, prec_t* out, int tiles) {
    MingruDWReduceArgs r = {gpu_addr(partial, "mingru dW partials"),
        gpu_addr(out, "mingru dW"), tiles, 0};
    dispatch(s, p, dim3(3072 / 64), dim3(64), r);
}

void mingru_dw_reduce_rm(cudaStream_t s, const float* partial, prec_t* out, int tiles,
        bool serial_oracle) {
    static Pipeline* p_fast = pipeline("mingru_dw_reduce");
    static Pipeline* p_serial = pipeline("mingru_dw_reduce_serial");
    if (tiles > 0) dw_reduce(s, serial_oracle ? p_serial : p_fast, partial, out, tiles);
}

bool mingru_fused_backward_rm(cudaStream_t s, int B, int T, int H,
        const prec_t* x, const prec_t* W, const prec_t* terminals, const prec_t* scan_h,
        const prec_t* grad_out, const prec_t* grad_next_state,
        prec_t* grad_combined, prec_t* grad_state, prec_t* grad_input, prec_t* grad_x,
        prec_t* grad_weight, const prec_t* combined, bool defer_dw_reduce) {
    if (!mingru_fused_eligible(B, T, H, x, W)) return false;
    // H>=64 keeps grad_input in threadgroup memory; the device copy is
    // optional there (null skips it). H=32 always writes it.
    if (!grad_x || grad_x == grad_input || !aligned16(grad_x) || !aligned16(grad_input)
            || (!grad_input && H < 64)) {
        return false;
    }
    if (H >= 64) {
        // No fused dW: grad_combined is written for the generic dW GEMM.
        // The scan reads the forward's combined, written as grad_combined is.
        if (grad_weight || !combined || combined == grad_combined) return false;
        Pipeline* p64 = nullptr;
        if (H == 64) { static Pipeline* p = pipeline("mingru_fused_backward_h64"); p64 = p; }
        else p64 = pipeline(H == 128 ? "wide_mingru_backward_h128" : "wide_mingru_backward_h256");
        MingruFused64BwdArgs a = {gpu_addr(x, "mingru x"), gpu_addr(W, "mingru W"),
            addr_or_null(terminals, "mingru terminals"), gpu_addr(scan_h, "mingru scan_h"),
            gpu_addr(grad_out, "mingru grad_out"), gpu_addr(grad_next_state, "mingru grad_next"),
            gpu_addr(grad_combined, "mingru grad_combined"), gpu_addr(grad_state, "mingru grad_state"),
            addr_or_null(grad_input, "mingru grad_input"), gpu_addr(grad_x, "mingru grad_x"),
            gpu_addr(combined, "mingru combined"), B, T, H, 0};
        dispatch(s, p64, dim3((unsigned)((long)B * T / 64)), dim3(H == 64 ? 64 : (H == 128 ? PUF_WIDE_128_THREADS : PUF_WIDE_256_THREADS)), a);
        return true;
    }
#ifndef PRECISION_FLOAT
    // BF16 data may be only two-byte aligned, but the dW scratch is a
    // device-float tensor consumed/stored by tensor_ops.
    if (grad_weight && !aligned16(grad_combined)) return false;
#endif
    static Pipeline* p_plain = pipeline("mingru_fused_backward");
    static Pipeline* p_dw = pipeline("mingru_fused_backward_dw");
    Pipeline* p = grad_weight ? p_dw : p_plain;
    MingruFusedBwdArgs a = {gpu_addr(x, "mingru x"), gpu_addr(W, "mingru W"),
        addr_or_null(terminals, "mingru terminals"), gpu_addr(scan_h, "mingru scan_h"),
        gpu_addr(grad_out, "mingru grad_out"), gpu_addr(grad_next_state, "mingru grad_next"),
        gpu_addr(grad_combined, "mingru grad_combined"), gpu_addr(grad_state, "mingru grad_state"),
        gpu_addr(grad_input, "mingru grad_input"), gpu_addr(grad_x, "mingru grad_x"),
        B, T, H, 0};
    dispatch(s, p, dim3((unsigned)((long)B * T / 64)), dim3(128), a);
    if (grad_weight && !defer_dw_reduce) {
        static Pipeline* p_reduce = pipeline("mingru_dw_reduce");
        dw_reduce(s, p_reduce, (const float*)grad_combined, grad_weight,
            (int)((long)B * T / 64));
    }
    return true;
}

// The small NS scalar/MMA paths share a pipeline. Log the actual dispatch
// argument (not just the environment setting) for process-level coverage.
static void muon_ns_log(int R, int C, int mma) {
    static const bool enabled = [] {
        const char* e = getenv("PUF_METAL_GEMM_LOG");
        return e && *e && strcmp(e, "0") != 0;
    }();
    if (!enabled) return;
    static std::mutex mu;
    static std::set<std::string> seen;
    char key[96];
    snprintf(key, sizeof(key), "R=%d C=%d mma=%d", R, C, mma);
    std::lock_guard<std::mutex> lk(mu);
    if (seen.insert(key).second) fprintf(stderr, "muon_ns %s\n", key);
}

struct MuonNSArgs {
    uint64_t x, x_buf, gram, gram_buf;
    float coeffs[15];
    int32_t R, C, mma;
};
static_assert(sizeof(MuonNSArgs) == 104, "MuonNSArgs layout");

bool muon_ns5_eligible(int R, int C) {
    static int enabled = -1;
    if (enabled < 0) {
        const char* e = getenv("PUF_METAL_FUSED_NS");
        enabled = !(e && strcmp(e, "0") == 0);
    }
    // Keeps every unfused GEMM on puf_gemm_f32's plain tiled path: the gram
    // GEMM (alpha 1, beta 0) would take tensor_ops at M % 64 == 0 and split-K
    // at K >= 2048; the other two have alpha or beta != {1, 0}.
    // Threadgroup staging (shaders/muon.metal MUON_NS_MAX_*) adds M <= 32 and
    // R * C <= 4096 (1024 threads x 4 elements).
    int lo = R < C ? R : C, hi = R < C ? C : R;
    return enabled && lo >= 1 && lo <= 32 && hi <= 2047 && (long)R * C <= 4096;
}

// The fused NS steps' GEMM stages on simdgroup 8x8 fragments (same per-element
// ascending-K fma chains, bit-identical) when R and C are multiples of 8;
// PUF_METAL_MUON_NS_MMA=0 keeps the scalar loops.
static bool muon_ns_mma(int R, int C) {
    static const bool enabled = [] {
        const char* e = getenv("PUF_METAL_MUON_NS_MMA");
        return !(e && strcmp(e, "0") == 0);
    }();
    return enabled && R % 8 == 0 && C % 8 == 0;
}

bool muon_ns5_rm(cudaStream_t s, prec_t* x, prec_t* x_buf, prec_t* gram, prec_t* gram_buf,
        int R, int C, const float coeffs[15]) {
    if (!muon_ns5_eligible(R, C)) return false;
    static Pipeline* p = pipeline("muon_ns5");
    MuonNSArgs a = {gpu_addr(x, "muon x"), gpu_addr(x_buf, "muon x_buf"),
        gpu_addr(gram, "muon gram"), gpu_addr(gram_buf, "muon gram_buf"), {}, R, C,
        muon_ns_mma(R, C)};
    for (int i = 0; i < 15; i++) a.coeffs[i] = coeffs[i];
    unsigned threads = pipeline_max_threads(p);
    if (threads > 1024) threads = 1024;
    threads &= ~31u;
    if (threads == 0 || (long)R * C > 4L * threads) return false;
    muon_ns_log(R, C, a.mma);
    dispatch(s, p, dim3(1), dim3(threads), a);
    return true;
}

struct MuonNSMatArgs { uint64_t x; float scale; int32_t R, C, mma; };
struct MuonNSBatchArgs { MuonNSMatArgs mats[8]; float coeffs[15]; float eps; int32_t count; };
static_assert(sizeof(MuonNSMatArgs) == 24 && sizeof(MuonNSBatchArgs) == 264,
    "MuonNSBatchArgs layout");

bool muon_ns5_batch_rm(cudaStream_t s, const MuonNSMatrix* mats, int count,
        const float coeffs[15], float eps) {
    static const bool enabled = [] {
        const char* e = getenv("PUF_METAL_MUON_BATCH");
        return !(e && strcmp(e, "0") == 0);
    }();
    if (!enabled || count < 1 || count > 8) return false;
    static Pipeline* p = pipeline("muon_ns5_batch");
    unsigned threads = pipeline_max_threads(p);
    if (threads > 1024) threads = 1024;
    threads &= ~31u;
    if (threads < 256) return false;
    MuonNSBatchArgs a = {};
    for (int i = 0; i < count; i++) {
        const MuonNSMatrix& m = mats[i];
        if (!muon_ns5_eligible(m.R, m.C) || (long)m.R * m.C > 4L * threads) return false;
        a.mats[i] = {gpu_addr(m.x, "muon batch x"), m.scale, m.R, m.C, muon_ns_mma(m.R, m.C)};
    }
    for (int i = 0; i < 15; i++) a.coeffs[i] = coeffs[i];
    a.eps = eps;
    a.count = count;
    for (int i = 0; i < count; i++) muon_ns_log(a.mats[i].R, a.mats[i].C, a.mats[i].mma);
    dispatch(s, p, dim3(count), dim3(threads), a);
    return true;
}

struct MuonWideMatArgs {
    uint64_t x, xb, g, gb, part;
    float scale;
    int32_t R, C, tops;
};
struct MuonWideArgs {
    MuonWideMatArgs mats[8];
    float c0, c1, c2, eps;
    int32_t count, step, pad0, pad1;
};
static_assert(sizeof(MuonWideMatArgs) == 56 && sizeof(MuonWideArgs) == 480,
    "MuonWideArgs layout");

static bool large_muon_enabled() {
    static const bool enabled = [] {
        const char* e = getenv("PUF_METAL_MUON_LARGE");
        return !(e && strcmp(e, "0") == 0);
    }();
    return enabled;
}

bool muon_ns_wide_eligible(int R, int C) {
    static const bool enabled = [] {
        const char* e = getenv("PUF_METAL_MUON_WIDE");
        return !(e && strcmp(e, "0") == 0);
    }();
    // Every unfused GEMM stays below gemm_rm's K >= 2048 split-K / sg_tn
    // branches (K is R or C for the gram, min(R, C) for the other two).
    int lo = R < C ? R : C, hi = R < C ? C : R;
    // At min512/1024 the output tile count also rules out every split-K
    // branch, even for K>=2048. Restrict the new cases to aligned recurrent
    // matrix dimensions; pointer alignment still selects each Gram's oracle.
    bool large = large_muon_enabled() && (lo == 512 || lo == 1024)
        && hi <= 3 * lo && R % 16 == 0 && C % 16 == 0;
    return enabled && lo >= 1 && (hi < 2048 || large);
}

static bool muon_large_shape(int R, int C) {
    int M = std::min(R, C);
    return large_muon_enabled() && (M == 512 || M == 1024) && R % 16 == 0 && C % 16 == 0;
}

size_t muon_ns_wide_scratch_bytes(int R, int C, bool aligned_input) {
    long M = R < C ? R : C;
    auto up = [](size_t b) { return (b + 15) & ~(size_t)15; };
    size_t x_bytes = up((size_t)R * C * sizeof(prec_t));
    return x_bytes + 2 * up((size_t)(M * M) * sizeof(prec_t)) + 256 * sizeof(float)
        + (!aligned_input && muon_large_shape(R, C) ? x_bytes : 0);
}

bool muon_ns_wide_rm(cudaStream_t s, const MuonNSMatrix* mats, int count,
        const float coeffs[15], float eps, void* scratch, size_t scratch_bytes) {
    if (count < 1 || count > 8 || !scratch || !aligned16(scratch)) return false;
    MuonWideArgs a = {};
    char* p = (char*)scratch;
    size_t used = 0;
    void* staged[8] = {};
    size_t staged_bytes[8] = {};
    int max_blk = 0, max_rc = 0, max_m = 0, max_mm = 0;
    for (int i = 0; i < count; i++) {
        const MuonNSMatrix& m = mats[i];
        if (!muon_ns_wide_eligible(m.R, m.C)) return false;
        int M = m.R < m.C ? m.R : m.C, RC = m.R * m.C;
        size_t need = muon_ns_wide_scratch_bytes(m.R, m.C, aligned16(m.x));
        if (used + need > scratch_bytes) return false;
        char* base = p + used;
        size_t xb_b = ((size_t)RC * sizeof(prec_t) + 15) & ~(size_t)15;
        size_t g_b = ((size_t)M * M * sizeof(prec_t) + 15) & ~(size_t)15;
        used += need;
        // Keep unaligned large gradients in aligned scratch for the whole NS
        // chain. Both copies preserve bits; normalization and every product
        // retain the same reduction and precision boundaries.
        const bool stage = !aligned16(m.x) && muon_large_shape(m.R, m.C);
        prec_t* x = m.x;
        if (stage) {
            staged[i] = base + xb_b + 2 * g_b + 256 * sizeof(float);
            staged_bytes[i] = (size_t)RC * sizeof(prec_t);
            x = (prec_t*)staged[i];
        }
        int tops = M % 64 == 0 ? ((aligned16(x) ? 1 : 0) | 2) : 0;
        // Bit2 enables large fragments per matrix, including mixed batches.
        // The old Gram reads only bits0/1; the opt-out remains effective even
        // when another H128/H256 matrix selects the auxiliary stage kernels.
        if (muon_large_shape(m.R, m.C)) tops |= 4;
        a.mats[i] = {gpu_addr(x, "muon wide x"), gpu_addr(base, "muon wide scratch"),
            gpu_addr(base + xb_b, "muon wide scratch"), gpu_addr(base + xb_b + g_b, "muon wide scratch"),
            gpu_addr(base + xb_b + 2 * g_b, "muon wide scratch"), m.scale, m.R, m.C, tops};
        int blk = (RC + 255) / 256;
        if (blk > max_blk) max_blk = blk;
        if (RC > max_rc) max_rc = RC;
        if (M > max_m) max_m = M;
        if (M * M > max_mm) max_mm = M * M;
    }
    // Validate the entire batch before encoding any work, including copies.
    for (int i = 0; i < count; ++i) if (staged[i])
        cudaMemcpyAsync(staged[i], mats[i].x, staged_bytes[i], cudaMemcpyDeviceToDevice, s);
    static Pipeline* p_sum = pipeline("muon_wide_sumsq");
    static Pipeline* p_norm = pipeline("muon_wide_normalize");
    static Pipeline* p_gram = pipeline("muon_wide_gram");
    static Pipeline* p_gbuf = pipeline("muon_wide_gram_buf");
    static Pipeline* p_upd = pipeline("muon_wide_update");
    static Pipeline* p_store = pipeline("muon_wide_store");
    static const bool wide_mma = [] {
        const char* e = getenv("PUF_METAL_MUON_WIDE_MMA");
        return !(e && strcmp(e, "0") == 0);
    }();
    bool use_wide_mma = false, use_large_gram = false;
    int gram_groups = 0, gbuf_groups = 0, update_groups = 0;
    if (wide_mma) for (int i = 0; i < count; ++i) {
        int lo = std::min(mats[i].R, mats[i].C);
        use_large_gram |= (a.mats[i].tops & 4) != 0;
        use_wide_mma |= (lo == 128 || lo == 256 || (a.mats[i].tops & 4)) && mats[i].R % 16 == 0 && mats[i].C % 16 == 0;
    }
    if (use_large_gram) for (int i = 0; i < count; ++i) {
        auto& m = a.mats[i]; int M = std::min(m.R, m.C);
        auto tiles = [](int R, int C, int TM, int TN) { return ((R + TM - 1) / TM) * ((C + TN - 1) / TN); };
        bool large = (m.tops & 4) != 0;
        gram_groups = std::max(gram_groups, large ? tiles(M, M, PUF_LARGE_MUON_GRAM_M, PUF_LARGE_MUON_GRAM_N)
            : (M * M + PUF_LARGE_MUON_GRAM_THREADS - 1) / PUF_LARGE_MUON_GRAM_THREADS);
        gbuf_groups = std::max(gbuf_groups, large ? tiles(M, M, PUF_LARGE_MUON_PRODUCT_M, PUF_LARGE_MUON_PRODUCT_N)
            : (M * M + PUF_LARGE_MUON_PRODUCT_THREADS - 1) / PUF_LARGE_MUON_PRODUCT_THREADS);
        update_groups = std::max(update_groups, large ? tiles(m.R, m.C, PUF_LARGE_MUON_PRODUCT_M, PUF_LARGE_MUON_PRODUCT_N)
            : (m.R * m.C + PUF_LARGE_MUON_PRODUCT_THREADS - 1) / PUF_LARGE_MUON_PRODUCT_THREADS);
    }
    Pipeline* gram = use_large_gram ? pipeline("wide_large_muon_gram") : p_gram;
    Pipeline* gbuf = use_large_gram ? pipeline("wide_large_muon_gram_buf") : (use_wide_mma ? pipeline("wide_muon_gram_buf") : p_gbuf);
    Pipeline* upd = use_large_gram ? pipeline("wide_large_muon_update") : (use_wide_mma ? pipeline("wide_muon_update") : p_upd);
    a.eps = eps;
    a.count = count;
    unsigned z = (unsigned)count, rc_blk = (unsigned)((max_rc + 255) / 256);
    dispatch(s, p_sum, dim3((unsigned)(max_blk < 256 ? max_blk : 256), 1, z), dim3(256), a);
    dispatch(s, p_norm, dim3(rc_blk, 1, z), dim3(256), a);
    for (int i = 0; i < 5; i++) {
        a.c0 = coeffs[3 * i]; a.c1 = coeffs[3 * i + 1]; a.c2 = coeffs[3 * i + 2];
        a.step = i;
        if (use_large_gram) {
            dispatch(s, gram, dim3(gram_groups, 1, z),
                dim3(PUF_LARGE_MUON_GRAM_THREADS), a);
        } else {
            dispatch(s, gram, dim3((max_m + 31) / 32, (max_m + 63) / 64, z), dim3(128), a);
        }
        if (use_large_gram) {
            dispatch(s, gbuf, dim3(gbuf_groups, 1, z), dim3(PUF_LARGE_MUON_PRODUCT_THREADS), a);
            dispatch(s, upd, dim3(update_groups, 1, z), dim3(PUF_LARGE_MUON_PRODUCT_THREADS), a);
        } else {
            dispatch(s, gbuf, dim3((unsigned)((max_mm + 255) / 256), 1, z), dim3(256), a);
            dispatch(s, upd, dim3(rc_blk, 1, z), dim3(256), a);
        }
    }
    dispatch(s, p_store, dim3(rc_blk, 1, z), dim3(256), a);
    for (int i = 0; i < count; ++i) if (staged[i])
        cudaMemcpyAsync(mats[i].x, staged[i], staged_bytes[i], cudaMemcpyDeviceToDevice, s);
    return true;
}

// PUF_METAL_GEMM_LOG=1: print each distinct GEMM shape / path once (stderr).
static void gemm_log(GemmPath path, bool ta, bool tb, int M, int N, int K,
        int lda, int ldb, int ldc, float alpha, float beta) {
    static int enabled = -1;
    if (enabled < 0) {
        const char* e = getenv("PUF_METAL_GEMM_LOG");
        enabled = e && *e && strcmp(e, "0") != 0;
    }
    if (!enabled) return;
    static std::mutex mu;
    static std::set<std::string> seen;
    char key[192];
    snprintf(key, sizeof(key), "path=%d ta=%d tb=%d M=%d N=%d K=%d lda=%d ldb=%d ldc=%d "
        "alpha=%g beta=%g", (int)path, ta, tb, M, N, K, lda, ldb, ldc, alpha, beta);
    std::lock_guard<std::mutex> lk(mu);
    if (seen.insert(key).second) fprintf(stderr, "gemm %s\n", key);
}

GemmPath gemm_rm_deferred(cudaStream_t s, bool ta, bool tb, int M, int N, int K,
        float alpha, const float* A, int lda, const float* B, int ldb,
        float beta, float* C, int ldc, void* workspace, size_t ws_bytes, size_t ws_off,
        GemmDeferred* defer, const GemmOpts* opts) {
    g_last_path = gemm_rm_impl(s, 0, ta, tb, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc,
        workspace, ws_bytes, opts, ws_off, defer);
    gemm_log(g_last_path, ta, tb, M, N, K, lda, ldb, ldc, alpha, beta);
    return g_last_path;
}

GemmPath gemm_rm_deferred(cudaStream_t s, bool ta, bool tb, int M, int N, int K,
        float alpha, const __nv_bfloat16* A, int lda, const __nv_bfloat16* B, int ldb,
        float beta, __nv_bfloat16* C, int ldc, void* workspace, size_t ws_bytes, size_t ws_off,
        GemmDeferred* defer, const GemmOpts* opts) {
    g_last_path = gemm_rm_impl(s, 1, ta, tb, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc,
        workspace, ws_bytes, opts, ws_off, defer);
    gemm_log(g_last_path, ta, tb, M, N, K, lda, ldb, ldc, alpha, beta);
    return g_last_path;
}

void gemm_finish(cudaStream_t s, GemmDeferred* defer) {
    if (!defer->pipe) return;
    GemmReduceArgs r;
    memcpy(&r, defer->args, sizeof(r));
    dispatch(s, defer->pipe, dim3(defer->grid), dim3(256), r);
    *defer = GemmDeferred();
}

GemmPath gemm_rm(cudaStream_t s, bool ta, bool tb, int M, int N, int K,
        float alpha, const float* A, int lda, const float* B, int ldb,
        float beta, float* C, int ldc,
        void* workspace, size_t ws_bytes, const GemmOpts* opts) {
    g_last_path = gemm_rm_impl(s, 0, ta, tb, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc,
        workspace, ws_bytes, opts);
    gemm_log(g_last_path, ta, tb, M, N, K, lda, ldb, ldc, alpha, beta);
    return g_last_path;
}

GemmPath gemm_rm(cudaStream_t s, bool ta, bool tb, int M, int N, int K,
        float alpha, const __nv_bfloat16* A, int lda, const __nv_bfloat16* B, int ldb,
        float beta, __nv_bfloat16* C, int ldc,
        void* workspace, size_t ws_bytes, const GemmOpts* opts) {
    g_last_path = gemm_rm_impl(s, 1, ta, tb, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc,
        workspace, ws_bytes, opts);
    gemm_log(g_last_path, ta, tb, M, N, K, lda, ldb, ldc, alpha, beta);
    return g_last_path;
}

}  // namespace puf_mtl

// ---------------------------------------------------------------------------
// cuBLAS shim
// ---------------------------------------------------------------------------

struct cublasContext {
    cudaStream_t stream = 0;
    void* workspace = nullptr;
    size_t ws_bytes = 0;
};

cublasStatus_t cublasCreate(cublasHandle_t* handle) {
    *handle = new cublasContext();
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDestroy(cublasHandle_t handle) {
    delete handle;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSetStream(cublasHandle_t handle, cudaStream_t stream) {
    handle->stream = stream;
    // Like cuBLAS, setting the stream resets the user workspace.
    handle->workspace = nullptr;
    handle->ws_bytes = 0;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSetWorkspace(cublasHandle_t handle, void* ws, size_t bytes) {
    handle->workspace = puf_mtl::is_gpu_visible(ws) ? ws : nullptr;
    handle->ws_bytes = handle->workspace ? bytes : 0;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSetMathMode(cublasHandle_t, cublasMath_t) {
    return CUBLAS_STATUS_SUCCESS;
}

// Column-major C(m x n) = alpha * op(A) @ op(B) + beta * C.
// Viewed row-major, a column-major matrix with leading dim ld is its
// transpose, so  C_rm(n x m) = op(B)_rm(n x k) @ op(A)_rm(k x m):
// a row-major GEMM with the operands swapped and the same transpose flags.
cublasStatus_t cublasGemmEx(cublasHandle_t h,
        cublasOperation_t transa, cublasOperation_t transb,
        int m, int n, int k, const void* alpha,
        const void* A, cudaDataType_t Atype, int lda,
        const void* B, cudaDataType_t Btype, int ldb,
        const void* beta, void* C, cudaDataType_t Ctype, int ldc,
        cublasComputeType_t, cublasGemmAlgo_t) {
    // Same storage type for A, B, C: CUDA_R_32F or CUDA_R_16BF (both with an
    // fp32 compute type; alpha/beta are float, as cuBLAS takes for COMPUTE_32F).
    bool f32 = Atype == CUDA_R_32F && Btype == CUDA_R_32F && Ctype == CUDA_R_32F;
    bool bf16 = Atype == CUDA_R_16BF && Btype == CUDA_R_16BF && Ctype == CUDA_R_16BF;
    if (!f32 && !bf16) {
        puf_mtl::fatal("cublasGemmEx: only all-CUDA_R_32F or all-CUDA_R_16BF operands");
    }
    if (transa == CUBLAS_OP_C || transb == CUBLAS_OP_C) {
        puf_mtl::fatal("cublasGemmEx: CUBLAS_OP_C unsupported");
    }
    if (f32) {
        puf_mtl::gemm_rm(h->stream, transb != CUBLAS_OP_N, transa != CUBLAS_OP_N,
            n, m, k, *(const float*)alpha,
            (const float*)B, ldb, (const float*)A, lda,
            *(const float*)beta, (float*)C, ldc, h->workspace, h->ws_bytes);
    } else {
        puf_mtl::gemm_rm(h->stream, transb != CUBLAS_OP_N, transa != CUBLAS_OP_N,
            n, m, k, *(const float*)alpha,
            (const __nv_bfloat16*)B, ldb, (const __nv_bfloat16*)A, lda,
            *(const float*)beta, (__nv_bfloat16*)C, ldc, h->workspace, h->ws_bytes);
    }
    return CUBLAS_STATUS_SUCCESS;
}
