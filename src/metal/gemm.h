// Metal backend GEMM entry points (row-major) used by the cuBLAS shim and
// tests/metal/test_gemm.mm.
#pragma once
#include <cstddef>
#include "runtime.h"
#include "compat/cuda_bf16.h"

namespace puf_mtl {

// Storage type of the precision-dependent kernels below (compiled from
// shaders/*.metal with the same PRECISION flag as this translation unit;
// runtime.mm rejects a metallib built for the other precision).
#ifdef PRECISION_FLOAT
typedef float prec_t;
#else
typedef __nv_bfloat16 prec_t;
#endif

// GEMM_SCALE: alpha == 0, C = beta * C without reading A/B (BLAS semantics).
// GEMM_SG_TN: A^T B with long K (weight gradients), simdgroup-matrix split-K.
// GEMM_TOPS_TN_SPLIT: the same shape class with M % 64 == 0 and N % 32 == 0,
// tensor_ops split-K (PUF_METAL_TOPS_TN_SPLIT=0 disables).
enum GemmPath { GEMM_NONE = 0, GEMM_TENSOR_OPS = 1, GEMM_TILED = 2, GEMM_SPLIT_K = 3,
    GEMM_SCALE = 4, GEMM_SG_TN = 5, GEMM_TOPS_TN_SPLIT = 6, GEMM_SMALL_N = 7,
    GEMM_WIDE_TN = 8, GEMM_LARGE_LINEAR = 9 };

struct GemmOpts {
    bool allow_tensor_ops = true;
    bool allow_split_k = true;
    bool allow_sg_tn = true;  // also requires allow_split_k
    bool allow_small_n = true;  // GEMM_SMALL_N (see gemm_rm_impl)
    bool allow_large_linear = true; // recurrent NT, H512/H1024 and M>=4096
    bool allow_large_tn = true; // H512/H1024 exact spatial tiles
    bool allow_wide_tn = true; // order-preserving wide/ragged spatial tiles
};

// C(M,N) = alpha * opA(A) @ opB(B) + beta * C, row-major with explicit
// leading dimensions. opA(A) is A (M x K, stride lda) or, if ta, A^T where A
// is stored K x M. Likewise for B. `workspace` (GPU-visible, may be null)
// enables split-K for long-K shapes. Returns the path taken.
GemmPath gemm_rm(cudaStream_t s, bool ta, bool tb, int M, int N, int K,
    float alpha, const float* A, int lda, const float* B, int ldb,
    float beta, float* C, int ldc,
    void* workspace = nullptr, size_t ws_bytes = 0, const GemmOpts* opts = nullptr);
// bf16 storage (cublasGemmEx CUDA_R_16BF): same paths and fp32 accumulation,
// C = bf16(alpha * acc + beta * C) rounded once at the store.
GemmPath gemm_rm(cudaStream_t s, bool ta, bool tb, int M, int N, int K,
    float alpha, const __nv_bfloat16* A, int lda, const __nv_bfloat16* B, int ldb,
    float beta, __nv_bfloat16* C, int ldc,
    void* workspace = nullptr, size_t ws_bytes = 0, const GemmOpts* opts = nullptr);

// A split-K reduction left pending by gemm_rm_deferred (pipe == nullptr:
// nothing pending).
struct GemmDeferred {
    Pipeline* pipe = nullptr;
    unsigned grid = 0;
    size_t ws_used = 0;  // partial bytes at workspace + ws_off
    alignas(8) unsigned char args[40];
};
// gemm_rm with the split-K reduction deferred, for overlapping it with
// unrelated work (see concurrent_begin): split paths write their partials at
// workspace + ws_off and leave the reduction in *defer; the result is
// complete once gemm_finish(s, defer) has run. Paths, split counts and bits
// are those of gemm_rm(..., workspace, ws_bytes, opts) - ws_off only moves
// the partials, which must fit below ws_bytes, else nothing is encoded and
// GEMM_NONE is returned. Single-dispatch paths run as in gemm_rm and leave
// nothing pending.
GemmPath gemm_rm_deferred(cudaStream_t s, bool ta, bool tb, int M, int N, int K,
    float alpha, const float* A, int lda, const float* B, int ldb,
    float beta, float* C, int ldc, void* workspace, size_t ws_bytes, size_t ws_off,
    GemmDeferred* defer, const GemmOpts* opts = nullptr);
GemmPath gemm_rm_deferred(cudaStream_t s, bool ta, bool tb, int M, int N, int K,
    float alpha, const __nv_bfloat16* A, int lda, const __nv_bfloat16* B, int ldb,
    float beta, __nv_bfloat16* C, int ldc, void* workspace, size_t ws_bytes, size_t ws_off,
    GemmDeferred* defer, const GemmOpts* opts = nullptr);
// Encodes a pending reduction (if any) and clears it.
void gemm_finish(cudaStream_t s, GemmDeferred* defer);

// C(M,N) = A(M,K) @ B(K,N) + D(M,N), all dense row-major, D must not alias C.
// Bit-identical to gemm_rm(nn, alpha 1, beta 0) followed by C += D when the
// GEMM would take the tensor_ops path; returns false and encodes nothing
// for any other shape / alignment (the caller then runs the two-pass path).
bool gemm_nn_add_rm(cudaStream_t s, int M, int N, int K,
    const float* A, const float* B, const float* D, float* C);
// bf16: C = bf16(bf16(A @ B) + D), the rounded GEMM store then add_kernel.
bool gemm_nn_add_rm(cudaStream_t s, int M, int N, int K,
    const __nv_bfloat16* A, const __nv_bfloat16* B, const __nv_bfloat16* D,
    __nv_bfloat16* C);

// Fused MinGRU train layer (shaders/mingru.metal mingru_fused_*). Forward:
// the (B*T, 3H) projection x @ W^T is computed per 64-row tile in
// threadgroup memory with the tensor_ops nt GEMM's exact matmul2d
// configuration and consumed by the scan in the same kernel; `combined` is
// never written. BF16 T=32 uses a 32-row/two-simdgroup forward tile
// (PUF_METAL_MINGRU_FWD32=0 restores 64 rows). Backward: recomputes the projection from the saved input x,
// runs the reverse scan, and computes grad_x = grad_combined @ W + grad_input
// from the threadgroup tile (tensor_ops_gemm_nn_add_f32's configuration).
// Bit-identical to gemm_rm(nt) into combined + mingru_scan_forward, and
// mingru_scan_backward + gemm_nn_add_rm. grad_out may alias grad_x (in place).
// H >= 64 (H=64 in mingru.metal, H=128/256 in wide.metal) differs: its projection is compute-bound, so
// the backward does not recompute it. The forward also stores the projection
// to `combined` (bit-identical to the GEMM's store) and the backward reads it
// (combined is required, and must not be grad_combined); the backward writes
// grad_combined and has no fused dW (grad_weight must be null). Otherwise
// these calls return false and encode nothing. PUF_METAL_FUSED_MINGRU64=0
// disables H == 64 alone; PUF_METAL_FUSED_MINGRU_WIDE=0 disables H=128/256.
// H >= 64 permits null grad_input (kept on chip). H == 32 ignores `combined`.
// Eligible only for H in {32, 64, 128, 256}, 64 % T == 0, (B*T) % 64 == 0, 16-byte aligned
// x, W (and grad_x, grad_input != grad_x); otherwise returns false and encodes
// nothing, and the caller runs the unfused path. PUF_METAL_FUSED_MINGRU=0
// disables it. If grad_weight is non-null, the backward also computes dW:
// grad_combined becomes FP32 scratch for (B*T/64, 3H, H) partials (half
// its FP32 capacity, all its BF16 capacity), followed by a fixed tile-order
// reduction to grad_weight. BF16 rounds combined/scan/gradient stores as in
// the generic path, and rounds dW only after the FP32 partial reduction.
// Other gradients keep their exact arithmetic; dW changes summation order.
// grad_weight must be disjoint from the inputs and other outputs. BF16
// grad_combined must be 16-byte aligned when used as FP32 dW scratch.
// defer_dw_reduce leaves that reduction to the caller:
// mingru_dw_reduce_rm(s, (float*)grad_combined, grad_weight, B*T/64) after
// this kernel (bit-identical to the reduction encoded here).
bool mingru_fused_eligible(int B, int T, int H, const prec_t* x, const prec_t* W);
bool mingru_fused_forward_rm(cudaStream_t s, int B, int T, int H,
    const prec_t* x, const prec_t* W, const prec_t* state, const prec_t* terminals,
    prec_t* scan_h, prec_t* out, prec_t* next_state, prec_t* combined = nullptr);
// H=64 accepts grad_input == nullptr (grad_x is still complete).
bool mingru_fused_backward_rm(cudaStream_t s, int B, int T, int H,
    const prec_t* x, const prec_t* W, const prec_t* terminals, const prec_t* scan_h,
    const prec_t* grad_out, const prec_t* grad_next_state,
    prec_t* grad_combined, prec_t* grad_state, prec_t* grad_input, prec_t* grad_x,
    prec_t* grad_weight = nullptr, const prec_t* combined = nullptr,
    bool defer_dw_reduce = false);

// The fused backward's dW reduction alone: sums `tiles` (96, 32) partials in
// ascending tile order. serial_oracle selects the plain one-load-per-step loop
// (tests: the shipped batched-load kernel must match it bit for bit).
void mingru_dw_reduce_rm(cudaStream_t s, const float* partial, prec_t* out, int tiles,
    bool serial_oracle = false);

// muon_step's five Newton-Schulz steps for one (R, C) matrix in one
// threadgroup dispatch, bit-identical to the unfused puf_mm_tn / puf_mm /
// puf_copy / puf_mm_nn sequence (all of which take puf_gemm_f32's tiled path
// at these sizes). coeffs = ns_coeffs[5][3] flattened. Ineligible (returns
// false, encodes nothing) unless 1 <= min(R, C) <= 32 and R * C <= 4096
// (the matrix, gram and gram_buf are staged in threadgroup memory).
// PUF_METAL_FUSED_NS=0 disables.
bool muon_ns5_eligible(int R, int C);
// muon_step's per-matrix tail (norm partials/reduce, l2 normalize, five NS
// steps, store_update) for up to 8 matrices in one dispatch, one threadgroup
// each; x[i] (R[i] x C[i]) is overwritten with scale[i] * NS(x / |x|).
// Returns false (encodes nothing) unless count <= 8 and every matrix is
// muon_ns5_eligible. PUF_METAL_MUON_BATCH=0 disables.
struct MuonNSMatrix { prec_t* x; int R, C; float scale; };
bool muon_ns5_batch_rm(cudaStream_t s, const MuonNSMatrix* mats, int count,
    const float coeffs[15], float eps);
// The same per-matrix tail for up to 8 matrices of any shape with
// max(R, C) < 2048 (meant for those muon_ns5_eligible rejects), batched
// across matrices: 18 dispatches in all (shaders/muon.metal muon_wide_*),
// each stage bit-identical per element to the unfused kernels and GEMM paths
// it replaces (tensor_ops gram where gemm_rm would take it, tiled otherwise).
// `scratch` (16-byte aligned) must hold the sum of
// muon_ns_wide_scratch_bytes over the matrices. Returns false (encodes
// nothing) if any matrix is ineligible or scratch is short.
// PUF_METAL_MUON_WIDE=0 disables.
// Large aligned-dimension matrices with min512/1024 and max<=3*min are
// also exact: all original GEMMs are unsplit at these output sizes.
// PUF_METAL_MUON_LARGE=0 restores their old eligibility and scalar stages.
bool muon_ns_wide_eligible(int R, int C);
// The default reserves for an unaligned input. Callers that know the packed
// gradient offset can omit the aligned staging buffer when aligned_input=true.
size_t muon_ns_wide_scratch_bytes(int R, int C, bool aligned_input = false);
bool muon_ns_wide_rm(cudaStream_t s, const MuonNSMatrix* mats, int count,
    const float coeffs[15], float eps, void* scratch, size_t scratch_bytes);
bool muon_ns5_rm(cudaStream_t s, prec_t* x, prec_t* x_buf, prec_t* gram, prec_t* gram_buf,
    int R, int C, const float coeffs[15]);

// Path taken by the most recent gemm_rm call on this thread (for tests).
GemmPath gemm_last_path();

}  // namespace puf_mtl
