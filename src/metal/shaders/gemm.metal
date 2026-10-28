// GEMM kernels for the cuBLAS shim (src/metal/gemm.mm). Row-major semantics:
//   C(M,N) = alpha * opA(A) @ opB(B) + beta * C
//   opA(A)(m,k) = ta ? A[k*lda + m] : A[m*lda + k]
//   opB(B)(k,n) = tb ? B[n*ldb + k] : B[k*ldb + n]
// beta == 0 never reads C (cuBLAS semantics: C may hold NaN/garbage).
// Every kernel exists for float (_f32) and bfloat (_bf16) storage; both
// accumulate in fp32 (cublasGemmEx CUBLAS_COMPUTE_32F) and the bf16 variants
// round once at the final store, C = bf16(alpha * acc + beta * C), like
// cuBLAS. Split-K partials are fp32 in both. The float instantiations are the
// original FP32 kernels (same arithmetic, same names).
#include "common.h"
#include <metal_tensor>
#include <MetalPerformancePrimitives/MPPTensorOpsMatMul2d.h>
using namespace mpp::tensor_ops;

template <typename T>
struct GemmArgsT {
    device const T* A;
    device const T* B;
    device T* C;
    device float* partial;  // split-K scratch (splits x M x N) or null
    int M, N, K;
    int lda, ldb, ldc;
    int ta, tb;
    int k_chunk;            // K range per grid.z slice
    float alpha, beta;
};

// Generic bounds-checked tiled GEMM: 64x64 output tile, BK=16, 256 threads,
// 4x4 outputs per thread (strided by 16 for conflict-free threadgroup reads).
// The tile loop structure follows valo-metal's sgemm_ksplit (MIT, Valtteri Valo,
// src/metal/shader_src.h @ 1e7e9dd1), widened and given full lda/ldb/ldc,
// transpose flags and alpha/beta.
#define GT_M 64
#define GT_N 64
#define GT_K 16

template <typename T>
static void gemm_tiled(constant GemmArgsT<T>& a, threadgroup float (*As)[GT_M + 4],
        threadgroup float (*Bs)[GT_N + 4], uint3 tgid, uint tid) {
    const int M = a.M, N = a.N, K = a.K;
    const int m0 = (int)tgid.y * GT_M;
    const int n0 = (int)tgid.x * GT_N;
    const int k_begin = (int)tgid.z * a.k_chunk;
    const int k_end = min(K, k_begin + a.k_chunk);
    const int tx = (int)tid % 16;
    const int ty = (int)tid / 16;
    device const T* A = a.A;
    device const T* B = a.B;

    float acc[4][4];
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++) acc[i][j] = 0.0f;

    for (int k0 = k_begin; k0 < k_end; k0 += GT_K) {
        for (int r = 0; r < 4; r++) {
            int e = (int)tid + r * 256;
            int mm, kk;
            if (a.ta) { mm = e % GT_M; kk = e / GT_M; }   // contiguous along m
            else      { kk = e % GT_K; mm = e / GT_K; }   // contiguous along k
            int gm = m0 + mm, gk = k0 + kk;
            float v = 0.0f;
            if (gm < M && gk < k_end) {
                v = a.ta ? puf_ld(A, (long)gk * a.lda + gm) : puf_ld(A, (long)gm * a.lda + gk);
            }
            As[kk][mm] = v;
        }
        for (int r = 0; r < 4; r++) {
            int e = (int)tid + r * 256;
            int nn, kk;
            if (a.tb) { kk = e % GT_K; nn = e / GT_K; }   // contiguous along k
            else      { nn = e % GT_N; kk = e / GT_N; }   // contiguous along n
            int gn = n0 + nn, gk = k0 + kk;
            float v = 0.0f;
            if (gn < N && gk < k_end) {
                v = a.tb ? puf_ld(B, (long)gn * a.ldb + gk) : puf_ld(B, (long)gk * a.ldb + gn);
            }
            Bs[kk][nn] = v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int kk = 0; kk < GT_K; kk++) {
            float av[4], bv[4];
            for (int i = 0; i < 4; i++) av[i] = As[kk][ty + 16 * i];
            for (int j = 0; j < 4; j++) bv[j] = Bs[kk][tx + 16 * j];
            for (int i = 0; i < 4; i++)
                for (int j = 0; j < 4; j++) acc[i][j] = fma(av[i], bv[j], acc[i][j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (a.partial) {
        device float* P = a.partial + (long)tgid.z * M * N;
        for (int i = 0; i < 4; i++) {
            int m = m0 + ty + 16 * i;
            if (m >= M) continue;
            for (int j = 0; j < 4; j++) {
                int n = n0 + tx + 16 * j;
                if (n < N) P[(long)m * N + n] = acc[i][j];
            }
        }
        return;
    }
    for (int i = 0; i < 4; i++) {
        int m = m0 + ty + 16 * i;
        if (m >= M) continue;
        for (int j = 0; j < 4; j++) {
            int n = n0 + tx + 16 * j;
            if (n >= N) continue;
            long ci = (long)m * a.ldc + n;
            float v = a.alpha * acc[i][j];
            if (a.beta != 0.0f) v = fma(a.beta, puf_ld(a.C, ci), v);
            puf_st(a.C, ci, v);
        }
    }
}

#define PUF_GEMM_TILED(SUF, T)                                                     \
kernel void puf_gemm_##SUF(constant GemmArgsT<T>& a [[buffer(0)]],                 \
        uint3 tgid [[threadgroup_position_in_grid]],                               \
        uint tid [[thread_index_in_threadgroup]]) {                                \
    threadgroup float As[GT_K][GT_M + 4];                                          \
    threadgroup float Bs[GT_K][GT_N + 4];                                          \
    gemm_tiled<T>(a, As, Bs, tgid, tid);                                           \
}
PUF_GEMM_TILED(f32, float)
PUF_GEMM_TILED(bf16, bfloat)

// Small-N GEMM (decoder forward: N = actions + 1 columns, K = hidden):
// opA = A (not transposed), N <= GSN_MAX_N, K % GT_K == 0, GSN_MIN_K <= K <=
// GSN_MAX_K, no split-K. The 64x64 tiled kernel above wastes all but N of
// its 64 columns and runs four barriers per 16-deep K step. Here one thread
// owns one output row; opB(B) is staged once per threadgroup as float. Each
// output is the tiled kernel's exact per-element arithmetic: acc starts at
// 0, fma(a, b, acc) over k ascending (with K % 16 == 0 the tiled kernel has
// no zero-padded k steps), then v = alpha * acc, v = fma(beta, C, v) iff
// beta != 0, one store rounding. Bit-identical to puf_gemm_* for every
// eligible shape (tests/metal/test_gemm.mm run_small_n, test_bf16.mm).
#define GSN_MAX_N 8
#define GSN_MAX_K 256
#define GSN_THREADS 128

// Four consecutive elements as float (bf16 widened exactly, no denormal flush).
static inline float4 gsn_ld4(device const float* p, int i) {
    return ((device const float4*)p)[i];
}
static inline float4 gsn_ld4(device const bfloat* p, int i) {
    return as_type<float4>(uint4(((device const ushort4*)p)[i]) << 16);
}

template <typename T>
static void gemm_small_n(constant GemmArgsT<T>& a, threadgroup float* Bs, uint gid, uint tid) {
    const int M = a.M, N = a.N, K = a.K;
    device const T* B = a.B;
    for (int e = (int)tid; e < N * K; e += GSN_THREADS) {
        int n = e / K, k = e % K;
        Bs[e] = a.tb ? puf_ld(B, (long)n * a.ldb + k) : puf_ld(B, (long)k * a.ldb + n);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // One thread per output element (row-major over (row, n)).
    const int row = (int)(gid / (uint)N), n = (int)(gid % (uint)N);
    if (row >= M) return;
    device const T* Ar = a.A + (long)row * a.lda;
    threadgroup const float* Bn = Bs + n * K;
    float acc = 0.0f;
    if (a.k_chunk == 4) {
        // Host: A 16-byte aligned, lda % 4 == 0 (K % 16 == 0). Same fma
        // order, four-element loads.
        for (int k4 = 0; k4 < K / 4; k4++) {
            float4 v = gsn_ld4(Ar, k4);
            acc = fma(v.x, Bn[4 * k4 + 0], acc);
            acc = fma(v.y, Bn[4 * k4 + 1], acc);
            acc = fma(v.z, Bn[4 * k4 + 2], acc);
            acc = fma(v.w, Bn[4 * k4 + 3], acc);
        }
    } else {
        for (int k = 0; k < K; k++) acc = fma(puf_ld(Ar, k), Bn[k], acc);
    }
    long ci = (long)row * a.ldc + n;
    float v = a.alpha * acc;
    if (a.beta != 0.0f) v = fma(a.beta, puf_ld(a.C, ci), v);
    puf_st(a.C, ci, v);
}

#define PUF_GEMM_SMALL_N(SUF, T)                                                   \
kernel void puf_gemm_small_n_##SUF(constant GemmArgsT<T>& a [[buffer(0)]],         \
        uint gid [[thread_position_in_grid]],                                      \
        uint tid [[thread_index_in_threadgroup]]) {                                \
    threadgroup float Bs[GSN_MAX_N * GSN_MAX_K];                                   \
    gemm_small_n<T>(a, Bs, gid, tid);                                              \
}
PUF_GEMM_SMALL_N(f32, float)
PUF_GEMM_SMALL_N(bf16, bfloat)

template <typename T>
struct GemmReduceArgsT {
    device const float* partial;
    device T* C;
    int M, N, ldc, splits;
    float alpha, beta;
};

// Sums split-K partials in a fixed order (deterministic), applies alpha/beta.
template <typename T>
static void gemm_reduce(constant GemmReduceArgsT<T>& a, uint gid) {
    long total = (long)a.M * a.N;
    if ((long)gid >= total) return;
    float s = 0.0f;
    for (int z = 0; z < a.splits; z++) s += a.partial[(long)z * total + gid];
    int m = (int)(gid / (uint)a.N), n = (int)(gid % (uint)a.N);
    long ci = (long)m * a.ldc + n;
    float v = a.alpha * s;
    if (a.beta != 0.0f) v = fma(a.beta, puf_ld(a.C, ci), v);
    puf_st(a.C, ci, v);
}
kernel void puf_gemm_reduce_f32(constant GemmReduceArgsT<float>& a [[buffer(0)]],
        uint gid [[thread_position_in_grid]]) {
    gemm_reduce<float>(a, gid);
}
kernel void puf_gemm_reduce_bf16(constant GemmReduceArgsT<bfloat>& a [[buffer(0)]],
        uint gid [[thread_position_in_grid]]) {
    gemm_reduce<bfloat>(a, gid);
}

// ---------------------------------------------------------------------------
// Metal 4 tensor_ops (MetalPerformancePrimitives matmul2d) fast paths.
// Adapted from valo-metal (MIT, Valtteri Valo) src/metal/platform.mm @ 1e7e9dd1
// (tensor_ops_gemm_{nt,nn,tn}_f32), switched to the struct-argument ABI.
// Preconditions (checked on the host): dense operands (lda/ldb/ldc natural),
// alpha == 1, beta == 0, M % 64 == 0, N % 32 == 0. 128 threads/threadgroup,
// grid (N/32, M/64).
// ---------------------------------------------------------------------------

struct TensorGemmArgs {
    device float* A;
    device float* B;
    device float* C;
    int M, N, K;
};

// C(M,N) = A(M,K) @ B(N,K)^T
kernel void tensor_ops_gemm_nt_f32(constant TensorGemmArgs& a [[buffer(0)]],
        uint2 tgid [[threadgroup_position_in_grid]]) {
    const int M = a.M, N = a.N, K = a.K;
    auto A = tensor<device float, dextents<int32_t, 2>, tensor_inline>(
        a.A, dextents<int32_t, 2>(K, M));
    auto B = tensor<device float, dextents<int32_t, 2>, tensor_inline>(
        a.B, dextents<int32_t, 2>(K, N));
    auto C = tensor<device float, dextents<int32_t, 2>, tensor_inline>(
        a.C, dextents<int32_t, 2>(N, M));
    constexpr auto desc = matmul2d_descriptor(64, 32,
        static_cast<int>(dynamic_extent), false, true, false);
    matmul2d<desc, execution_simdgroups<4>> op;
    auto mA = A.slice(0, tgid.y * 64);
    auto mB = B.slice(0, tgid.x * 32);
    auto mC = C.slice(tgid.x * 32, tgid.y * 64);
    op.run(mA, mB, mC);
}

// C(M,N) = A(M,K) @ B(K,N)
kernel void tensor_ops_gemm_nn_f32(constant TensorGemmArgs& a [[buffer(0)]],
        uint2 tgid [[threadgroup_position_in_grid]]) {
    const int M = a.M, N = a.N, K = a.K;
    auto A_cm = tensor<device float, dextents<int32_t, 2>, tensor_inline>(
        a.A, dextents<int32_t, 2>(K, M));
    auto B_cm = tensor<device float, dextents<int32_t, 2>, tensor_inline>(
        a.B, dextents<int32_t, 2>(N, K));
    auto C_cm = tensor<device float, dextents<int32_t, 2>, tensor_inline>(
        a.C, dextents<int32_t, 2>(N, M));
    constexpr auto desc = matmul2d_descriptor(64, 32,
        static_cast<int>(dynamic_extent), false, false, false);
    matmul2d<desc, execution_simdgroups<4>> op;
    auto mFirst = A_cm.slice(0, tgid.y * 64);
    auto mSecond = B_cm.slice(tgid.x * 32, 0);
    auto mResult = C_cm.slice(tgid.x * 32, tgid.y * 64);
    op.run(mFirst, mSecond, mResult);
}

// C(M,N) = A(M,K) @ B(K,N) + D(M,N), D dense like C (may not alias C).
// The product is rounded to FP32 exactly as tensor_ops_gemm_nn_f32 stores
// it, then D is added once: bit-identical to the GEMM followed by
// add_kernel (dst = C + D), with one pass over C instead of three.
struct TensorGemmAddArgs {
    device float* A;
    device float* B;
    device float* C;
    device float* D;
    int M, N, K;
};

kernel void tensor_ops_gemm_nn_add_f32(constant TensorGemmAddArgs& a [[buffer(0)]],
        uint2 tgid [[threadgroup_position_in_grid]]) {
    const int M = a.M, N = a.N, K = a.K;
    auto A_cm = tensor<device float, dextents<int32_t, 2>, tensor_inline>(
        a.A, dextents<int32_t, 2>(K, M));
    auto B_cm = tensor<device float, dextents<int32_t, 2>, tensor_inline>(
        a.B, dextents<int32_t, 2>(N, K));
    auto C_cm = tensor<device float, dextents<int32_t, 2>, tensor_inline>(
        a.C, dextents<int32_t, 2>(N, M));
    auto D_cm = tensor<device float, dextents<int32_t, 2>, tensor_inline>(
        a.D, dextents<int32_t, 2>(N, M));
    constexpr auto desc = matmul2d_descriptor(64, 32,
        static_cast<int>(dynamic_extent), false, false, false);
    matmul2d<desc, execution_simdgroups<4>> op;
    auto mFirst = A_cm.slice(0, tgid.y * 64);
    auto mSecond = B_cm.slice(tgid.x * 32, 0);
    auto mResult = C_cm.slice(tgid.x * 32, tgid.y * 64);
    auto mAdd = D_cm.slice(tgid.x * 32, tgid.y * 64);
    auto cT = op.get_destination_cooperative_tensor<decltype(mFirst),
        decltype(mSecond), float>();
    #pragma unroll
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
        if (cT.is_valid_element(i)) cT[i] = 0.0f;
    }
    op.run(mFirst, mSecond, cT);
    auto dT = op.get_destination_cooperative_tensor<decltype(mFirst),
        decltype(mSecond), float>();
    dT.load(mAdd);
    #pragma unroll
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
        if (cT.is_valid_element(i)) cT[i] += dT[i];
    }
    cT.store(mResult);
}

// C(M,N) = A(K,M)^T @ B(K,N)
kernel void tensor_ops_gemm_tn_f32(constant TensorGemmArgs& a [[buffer(0)]],
        uint2 tgid [[threadgroup_position_in_grid]]) {
    const int M = a.M, N = a.N, K = a.K;
    auto A_cm = tensor<device float, dextents<int32_t, 2>, tensor_inline>(
        a.A, dextents<int32_t, 2>(M, K));
    auto B_cm = tensor<device float, dextents<int32_t, 2>, tensor_inline>(
        a.B, dextents<int32_t, 2>(N, K));
    auto C_cm = tensor<device float, dextents<int32_t, 2>, tensor_inline>(
        a.C, dextents<int32_t, 2>(N, M));
    constexpr auto desc = matmul2d_descriptor(64, 32,
        static_cast<int>(dynamic_extent), true, false, false);
    matmul2d<desc, execution_simdgroups<4>> op;
    auto mFirst = A_cm.slice(tgid.y * 64, 0);
    auto mSecond = B_cm.slice(tgid.x * 32, 0);
    auto mResult = C_cm.slice(tgid.x * 32, tgid.y * 64);
    op.run(mFirst, mSecond, mResult);
}


// bf16 tensor_ops variants: bfloat operands, fp32 cooperative destination,
// rounded to bf16 at the store (cuBLAS CUDA_R_16BF out, COMPUTE_32F). The
// store is explicit (get_multidimensional_index: [0] = column within the
// 32-wide N tile, [1] = row within the 64-row M tile) so the rounding is
// puf_f32_to_bf16's (RNE, no denormal flush).
struct TensorGemmArgsBF16 {
    device bfloat* A;
    device bfloat* B;
    device bfloat* C;
    int M, N, K;
};

template <typename CT>
static inline void tensor_zero(thread CT& cT) {
    #pragma unroll
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
        if (cT.is_valid_element(i)) cT[i] = 0.0f;
    }
}

template <typename CT>
static inline void tensor_store_bf16(thread CT& cT, device bfloat* C, int N, int n0, int m0) {
    #pragma unroll
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
        if (!cT.is_valid_element(i)) continue;
        auto idx = cT.get_multidimensional_index(i);
        C[(long)(m0 + idx[1]) * N + n0 + idx[0]] = puf_f32_to_bf16(cT[i]);
    }
}

typedef tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> BF16Tensor2;

kernel void tensor_ops_gemm_nt_bf16(constant TensorGemmArgsBF16& a [[buffer(0)]],
        uint2 tgid [[threadgroup_position_in_grid]]) {
    const int M = a.M, N = a.N, K = a.K;
    auto A = BF16Tensor2(a.A, dextents<int32_t, 2>(K, M));
    auto B = BF16Tensor2(a.B, dextents<int32_t, 2>(K, N));
    constexpr auto desc = matmul2d_descriptor(64, 32,
        static_cast<int>(dynamic_extent), false, true, false);
    matmul2d<desc, execution_simdgroups<4>> op;
    auto mA = A.slice(0, tgid.y * 64);
    auto mB = B.slice(0, tgid.x * 32);
    auto cT = op.get_destination_cooperative_tensor<decltype(mA), decltype(mB), float>();
    tensor_zero(cT);
    op.run(mA, mB, cT);
    tensor_store_bf16(cT, a.C, N, tgid.x * 32, tgid.y * 64);
}

kernel void tensor_ops_gemm_nn_bf16(constant TensorGemmArgsBF16& a [[buffer(0)]],
        uint2 tgid [[threadgroup_position_in_grid]]) {
    const int M = a.M, N = a.N, K = a.K;
    auto A_cm = BF16Tensor2(a.A, dextents<int32_t, 2>(K, M));
    auto B_cm = BF16Tensor2(a.B, dextents<int32_t, 2>(N, K));
    constexpr auto desc = matmul2d_descriptor(64, 32,
        static_cast<int>(dynamic_extent), false, false, false);
    matmul2d<desc, execution_simdgroups<4>> op;
    auto mFirst = A_cm.slice(0, tgid.y * 64);
    auto mSecond = B_cm.slice(tgid.x * 32, 0);
    auto cT = op.get_destination_cooperative_tensor<decltype(mFirst), decltype(mSecond), float>();
    tensor_zero(cT);
    op.run(mFirst, mSecond, cT);
    tensor_store_bf16(cT, a.C, N, tgid.x * 32, tgid.y * 64);
}

// C = bf16(bf16(A @ B) + D): the GEMM's rounded store followed by
// add_kernel's dst = bf16(dst + src), in one pass.
struct TensorGemmAddArgsBF16 {
    device bfloat* A;
    device bfloat* B;
    device bfloat* C;
    device bfloat* D;
    int M, N, K;
};

kernel void tensor_ops_gemm_nn_add_bf16(constant TensorGemmAddArgsBF16& a [[buffer(0)]],
        uint2 tgid [[threadgroup_position_in_grid]]) {
    const int M = a.M, N = a.N, K = a.K;
    auto A_cm = BF16Tensor2(a.A, dextents<int32_t, 2>(K, M));
    auto B_cm = BF16Tensor2(a.B, dextents<int32_t, 2>(N, K));
    constexpr auto desc = matmul2d_descriptor(64, 32,
        static_cast<int>(dynamic_extent), false, false, false);
    matmul2d<desc, execution_simdgroups<4>> op;
    auto mFirst = A_cm.slice(0, tgid.y * 64);
    auto mSecond = B_cm.slice(tgid.x * 32, 0);
    auto cT = op.get_destination_cooperative_tensor<decltype(mFirst), decltype(mSecond), float>();
    tensor_zero(cT);
    op.run(mFirst, mSecond, cT);
    const int n0 = tgid.x * 32, m0 = tgid.y * 64;
    #pragma unroll
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
        if (!cT.is_valid_element(i)) continue;
        auto idx = cT.get_multidimensional_index(i);
        long ci = (long)(m0 + idx[1]) * N + n0 + idx[0];
        float prod = puf_bf16_to_f32(puf_f32_to_bf16(cT[i]));
        a.C[ci] = puf_f32_to_bf16(prod + puf_bf16_to_f32(a.D[ci]));
    }
}

kernel void tensor_ops_gemm_tn_bf16(constant TensorGemmArgsBF16& a [[buffer(0)]],
        uint2 tgid [[threadgroup_position_in_grid]]) {
    const int M = a.M, N = a.N, K = a.K;
    auto A_cm = BF16Tensor2(a.A, dextents<int32_t, 2>(M, K));
    auto B_cm = BF16Tensor2(a.B, dextents<int32_t, 2>(N, K));
    constexpr auto desc = matmul2d_descriptor(64, 32,
        static_cast<int>(dynamic_extent), true, false, false);
    matmul2d<desc, execution_simdgroups<4>> op;
    auto mFirst = A_cm.slice(tgid.y * 64, 0);
    auto mSecond = B_cm.slice(tgid.x * 32, 0);
    auto cT = op.get_destination_cooperative_tensor<decltype(mFirst), decltype(mSecond), float>();
    tensor_zero(cT);
    op.run(mFirst, mSecond, cT);
    tensor_store_bf16(cT, a.C, N, tgid.x * 32, tgid.y * 64);
}

// ---------------------------------------------------------------------------
// Split-K tensor_ops A^T B for weight gradients with M % 64 == 0, N % 32 == 0
// (gemm.mm GEMM_TOPS_TN_SPLIT): grid.z slice z multiplies rows
// [z * k_chunk, min(K, (z + 1) * k_chunk)) of A (K, M) and B (K, N) with the
// tensor_ops_gemm_tn matmul2d configuration and stores the fp32 tile to
// partial[z] (M x N, dense); puf_gemm_reduce sums the slices in z order.
// grid (N / 32, M / 64, splits), 128 threads.
// ---------------------------------------------------------------------------
struct TensorTnSplitArgs {
    device void* A;
    device void* B;
    device float* partial;
    int M, N, K, k_chunk;
};

template <typename T>
static void tops_tn_split(constant TensorTnSplitArgs& a, uint3 tgid) {
    const int M = a.M, N = a.N;
    const int k0 = (int)tgid.z * a.k_chunk;
    const int kc = min(a.k_chunk, a.K - k0);
    if (kc <= 0) return;
    auto A_cm = tensor<device T, dextents<int32_t, 2>, tensor_inline>(
        (device T*)a.A + (long)k0 * M, dextents<int32_t, 2>(M, kc));
    auto B_cm = tensor<device T, dextents<int32_t, 2>, tensor_inline>(
        (device T*)a.B + (long)k0 * N, dextents<int32_t, 2>(N, kc));
    auto P_cm = tensor<device float, dextents<int32_t, 2>, tensor_inline>(
        a.partial + (long)tgid.z * M * N, dextents<int32_t, 2>(N, M));
    constexpr auto desc = matmul2d_descriptor(64, 32,
        static_cast<int>(dynamic_extent), true, false, false);
    matmul2d<desc, execution_simdgroups<4>> op;
    auto mFirst = A_cm.slice(tgid.y * 64, 0);
    auto mSecond = B_cm.slice(tgid.x * 32, 0);
    auto mResult = P_cm.slice(tgid.x * 32, tgid.y * 64);
    op.run(mFirst, mSecond, mResult);
}
kernel void tensor_ops_gemm_tn_split_f32(constant TensorTnSplitArgs& a [[buffer(0)]],
        uint3 tgid [[threadgroup_position_in_grid]]) {
    tops_tn_split<float>(a, tgid);
}
kernel void tensor_ops_gemm_tn_split_bf16(constant TensorTnSplitArgs& a [[buffer(0)]],
        uint3 tgid [[threadgroup_position_in_grid]]) {
    tops_tn_split<bfloat>(a, tgid);
}

// ---------------------------------------------------------------------------
// Weight-gradient shape: C(M,N) = A(K,M)^T @ B(K,N) with small M, N and a long
// K (minibatch rows). simdgroup_matrix 8x8 MMAs fed straight from device
// memory; 8x8 blocks that cross M, N or the end of the K slice are staged
// through threadgroup memory with bounds checks (zero fill).
// Threadgroup = 4 simdgroups sharing one TxT output tile (T = 16); simdgroup s takes
// the 8-row K blocks k_begin + 8*s + 32*i of this grid.z slice, and the four
// partial tiles are summed in a fixed order (deterministic).
// Host preconditions: ta=1, tb=0, k_chunk % 8 == 0.
// Writes partial[z] (M x N, dense) when partial != null, else
// C = alpha * acc + beta * C.  grid (ceil(N/T), ceil(M/T), splits), 128 threads.
// ---------------------------------------------------------------------------
#include <metal_simdgroup_matrix>

// d[i][j] = A[(k + j) * lda + mb + i]  (an 8x8 block of A^T)
static inline void sg_load_at(thread simdgroup_float8x8& d, device const float* A, int lda,
        int k, int mb, int M, int k_end, threadgroup float* st, uint lane) {
    if (mb + 8 <= M && k + 8 <= k_end) {
        simdgroup_load(d, A + (long)k * lda + mb, (ulong)lda, ulong2(0, 0), true);
        return;
    }
    for (uint e = lane; e < 64; e += 32) {
        int i = (int)e / 8, j = (int)e % 8;
        st[e] = (mb + i < M && k + j < k_end) ? A[(long)(k + j) * lda + mb + i] : 0.0f;
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_load(d, st, 8);
    simdgroup_barrier(mem_flags::mem_threadgroup);
}

// d[i][j] = B[(k + i) * ldb + nb + j]
static inline void sg_load_b(thread simdgroup_float8x8& d, device const float* B, int ldb,
        int k, int nb, int N, int k_end, threadgroup float* st, uint lane) {
    if (nb + 8 <= N && k + 8 <= k_end) {
        simdgroup_load(d, B + (long)k * ldb + nb, (ulong)ldb);
        return;
    }
    for (uint e = lane; e < 64; e += 32) {
        int i = (int)e / 8, j = (int)e % 8;
        st[e] = (nb + j < N && k + i < k_end) ? B[(long)(k + i) * ldb + nb + j] : 0.0f;
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_load(d, st, 8);
    simdgroup_barrier(mem_flags::mem_threadgroup);
}

template <typename S, int T>
static void gemm_tn_sg(constant GemmArgsT<S>& a, threadgroup float (*red)[T][T],
        threadgroup float (*stage)[64], uint3 tgid, uint sg, uint lane, uint tid) {
    constexpr int NB = T / 8;
    const int M = a.M, N = a.N, K = a.K;
    const int m0 = (int)tgid.y * T, n0 = (int)tgid.x * T;
    const int k_begin = (int)tgid.z * a.k_chunk;
    const int k_end = min(K, k_begin + a.k_chunk);
    // Number of 8-blocks of this tile inside M / N (uniform per threadgroup).
    const int mb = min(NB, (M - m0 + 7) / 8), nb = min(NB, (N - n0 + 7) / 8);
    threadgroup float* st = stage[sg];

    simdgroup_float8x8 acc[NB][NB];
    for (int i = 0; i < NB; i++)
        for (int j = 0; j < NB; j++) acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    for (int k = k_begin + 8 * (int)sg; k < k_end; k += 32) {
        simdgroup_float8x8 av[NB], bv[NB];
        for (int i = 0; i < NB; i++)
            if (i < mb) sg_load_at(av[i], a.A, a.lda, k, m0 + 8 * i, M, k_end, st, lane);
        for (int j = 0; j < NB; j++)
            if (j < nb) sg_load_b(bv[j], a.B, a.ldb, k, n0 + 8 * j, N, k_end, st, lane);
        for (int i = 0; i < NB; i++)
            for (int j = 0; j < NB; j++)
                if (i < mb && j < nb) simdgroup_multiply_accumulate(acc[i][j], av[i], bv[j], acc[i][j]);
    }
    for (int i = 0; i < NB; i++)
        for (int j = 0; j < NB; j++) simdgroup_store(acc[i][j], &red[sg][8 * i][8 * j], T);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int e = (int)tid; e < T * T; e += 128) {
        int r = e / T, c = e % T;
        int m = m0 + r, n = n0 + c;
        if (m >= M || n >= N) continue;
        float v = ((red[0][r][c] + red[1][r][c]) + red[2][r][c]) + red[3][r][c];
        if (a.partial) {
            a.partial[(long)tgid.z * M * N + (long)m * N + n] = v;
        } else {
            long ci = (long)m * a.ldc + n;
            float o = a.alpha * v;
            if (a.beta != 0.0f) o = fma(a.beta, puf_ld(a.C, ci), o);
            puf_st(a.C, ci, o);
        }
    }
}

#define PUF_GEMM_TN_SG(T, SUF, S)                                                  \
kernel void puf_gemm_tn_sg##T##_##SUF(constant GemmArgsT<S>& a [[buffer(0)]],      \
        uint3 tgid [[threadgroup_position_in_grid]],                               \
        uint sg [[simdgroup_index_in_threadgroup]],                                \
        uint lane [[thread_index_in_simdgroup]],                                   \
        uint tid [[thread_index_in_threadgroup]]) {                                \
    threadgroup float red[4][T][T];                                                \
    threadgroup float stage[4][64];                                                \
    gemm_tn_sg<S, T>(a, red, stage, tgid, sg, lane, tid);                          \
}
// 32 (4x4 accumulators per simdgroup) measured ~1.7x slower on M4
PUF_GEMM_TN_SG(16, f32, float)

// bf16 storage. The f32 kernel's per-lane simdgroup_load gathers move half
// the bytes per load instruction in bf16 and ran at ~30 GB/s on M4 (vs ~75
// for f32). Here the threadgroup cooperatively loads each 32-row K chunk of
// the A and B tiles with vector loads (4 bf16 per thread: one 8-byte or two
// 4-byte loads as alignment allows, scalar at ragged edges), widens it
// exactly to fp32 in threadgroup memory, and the simdgroups take their 8-row
// blocks from there. Block assignment (simdgroup s: rows k0 + 8s), zero fill,
// skipped blocks, MMA order and the final 4-way sum are those of
// gemm_tn_sg<float, 16>, so the result is the f32 kernel's on widened inputs.
static inline void sg_coop_tile(threadgroup float (*dst)[16], device const bfloat* X, int ldx,
        int k0, int k_end, int c0, int C, uint tid) {
    const int r = (int)tid / 4, q = ((int)tid % 4) * 4;  // 128 threads: 32 rows x 4 quads
    const int k = k0 + r, c = c0 + q;
    threadgroup float* d = &dst[r][q];
    if (k < k_end && c + 4 <= C) {
        device const bfloat* p = X + (long)k * ldx + c;
        const ulong addr = (ulong)p;
        if ((addr & 7) == 0) {
            ushort4 v = *(device const ushort4*)p;
            d[0] = as_type<float>(uint(v.x) << 16);
            d[1] = as_type<float>(uint(v.y) << 16);
            d[2] = as_type<float>(uint(v.z) << 16);
            d[3] = as_type<float>(uint(v.w) << 16);
            return;
        }
        if ((addr & 3) == 0) {
            ushort2 v = *(device const ushort2*)p, w = *(device const ushort2*)(p + 2);
            d[0] = as_type<float>(uint(v.x) << 16);
            d[1] = as_type<float>(uint(v.y) << 16);
            d[2] = as_type<float>(uint(w.x) << 16);
            d[3] = as_type<float>(uint(w.y) << 16);
            return;
        }
    }
    for (int e = 0; e < 4; e++)
        d[e] = (k < k_end && c + e < C) ? puf_ld(X, (long)k * ldx + c + e) : 0.0f;
}

kernel void puf_gemm_tn_sg16_bf16(constant GemmArgsT<bfloat>& a [[buffer(0)]],
        uint3 tgid [[threadgroup_position_in_grid]],
        uint sg [[simdgroup_index_in_threadgroup]],
        uint lane [[thread_index_in_simdgroup]],
        uint tid [[thread_index_in_threadgroup]]) {
    constexpr int T = 16, NB = 2;
    threadgroup float red[4][T][T];
    threadgroup float As[32][T];
    threadgroup float Bs[32][T];
    const int M = a.M, N = a.N, K = a.K;
    const int m0 = (int)tgid.y * T, n0 = (int)tgid.x * T;
    const int k_begin = (int)tgid.z * a.k_chunk;
    const int k_end = min(K, k_begin + a.k_chunk);
    const int mb = min(NB, (M - m0 + 7) / 8), nb = min(NB, (N - n0 + 7) / 8);

    simdgroup_float8x8 acc[NB][NB];
    for (int i = 0; i < NB; i++)
        for (int j = 0; j < NB; j++) acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    // k_chunk is a multiple of 32 on the split path; k_begin = 0 otherwise.
    for (int k0 = k_begin; k0 < k_end; k0 += 32) {
        sg_coop_tile(As, a.A, a.lda, k0, k_end, m0, M, tid);
        sg_coop_tile(Bs, a.B, a.ldb, k0, k_end, n0, N, tid);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (k0 + 8 * (int)sg < k_end) {
            simdgroup_float8x8 av[NB], bv[NB];
            for (int i = 0; i < NB; i++)
                simdgroup_load(av[i], &As[8 * sg][8 * i], T, ulong2(0, 0), true);
            for (int j = 0; j < NB; j++) simdgroup_load(bv[j], &Bs[8 * sg][8 * j], T);
            for (int i = 0; i < NB; i++)
                for (int j = 0; j < NB; j++)
                    if (i < mb && j < nb) simdgroup_multiply_accumulate(acc[i][j], av[i], bv[j], acc[i][j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (int i = 0; i < NB; i++)
        for (int j = 0; j < NB; j++) simdgroup_store(acc[i][j], &red[sg][8 * i][8 * j], T);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int e = (int)tid; e < T * T; e += 128) {
        int r = e / T, c = e % T;
        int m = m0 + r, n = n0 + c;
        if (m >= M || n >= N) continue;
        float v = ((red[0][r][c] + red[1][r][c]) + red[2][r][c]) + red[3][r][c];
        if (a.partial) {
            a.partial[(long)tgid.z * M * N + (long)m * N + n] = v;
        } else {
            long ci = (long)m * a.ldc + n;
            float o = a.alpha * v;
            if (a.beta != 0.0f) o = fma(a.beta, puf_ld(a.C, ci), o);
            puf_st(a.C, ci, o);
        }
    }
}
