// Craftax category-embedding encoder (ocean/craftax/craftax.cu).
// The backward embedding scatter keeps CUDA's 2^24 fixed-point int64 sums
// (envs_fxp.h): bit-identical to CUDA for identical grad_concat inputs.
// precision_t storage, with FP32 arithmetic and explicit CUDA-compatible rounding.
#include "envs_fxp.h"

#define CX_NUM_CELLS 99
#define CX_CHANNELS 8
#define CX_MAP_FLOATS (CX_NUM_CELLS * CX_CHANNELS)
#define CX_NUM_SCALARS 51
#define CX_EMB_DIM 16
#define CX_MAP_EMB (CX_NUM_CELLS * CX_EMB_DIM)
#define CX_CONCAT (CX_MAP_EMB + CX_NUM_SCALARS)
#define CX_EMBED_N (154 * CX_EMB_DIM)

constant int CX_OFFSETS[CX_CHANNELS] = {0, 64, 72, 74, 90, 106, 122, 138};

struct CraftaxArgs {
    device precision_t* dst;
    device const precision_t* src;     // obs (embed/scalars/backward) or fxp sums
    device const precision_t* weight;  // embed_w (forward) or grad_concat (backward)
    int B, obs_size, n;
};

kernel void craftax_embed_bag(constant CraftaxArgs& a [[buffer(0)]],
        uint idx [[thread_position_in_grid]]) {
    if (idx >= uint(a.B * CX_NUM_CELLS * CX_EMB_DIM)) return;
    int d = idx % CX_EMB_DIM;
    int cell = (idx / CX_EMB_DIM) % CX_NUM_CELLS;
    int b = idx / (CX_NUM_CELLS * CX_EMB_DIM);
    device const precision_t* ids = a.src + (long)b * a.obs_size + cell * CX_CHANNELS;
    float sum = 0.0f;
    for (int ch = 0; ch < CX_CHANNELS; ch++) {
        int row = CX_OFFSETS[ch] + int(to_float(ids[ch]));
        sum += to_float(a.weight[row * CX_EMB_DIM + d]);
    }
    a.dst[(long)b * CX_CONCAT + cell * CX_EMB_DIM + d] = from_float(sum);
}

kernel void craftax_copy_scalars(constant CraftaxArgs& a [[buffer(0)]],
        uint idx [[thread_position_in_grid]]) {
    if (idx >= uint(a.B * CX_NUM_SCALARS)) return;
    int j = idx % CX_NUM_SCALARS;
    int b = idx / CX_NUM_SCALARS;
    a.dst[(long)b * CX_CONCAT + CX_MAP_EMB + j] = from_float(to_float(a.src[(long)b * a.obs_size + CX_MAP_FLOATS + j]));
}

// dst is the int64 [154*16] accumulator.
kernel void craftax_embed_bag_backward(constant CraftaxArgs& a [[buffer(0)]],
        uint3 blockIdx [[threadgroup_position_in_grid]],
        uint3 blockDim [[threads_per_threadgroup]],
        uint3 threadIdx [[thread_position_in_threadgroup]],
        uint3 gridDim [[threadgroups_per_grid]]) {
    threadgroup atomic_uint acc_lo[CX_EMBED_N];
    threadgroup atomic_uint acc_hi[CX_EMBED_N];
    fxp_tg_zero<CX_EMBED_N>(acc_lo, acc_hi, threadIdx.x, blockDim.x);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    int total = a.B * CX_NUM_CELLS * CX_EMB_DIM;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
            idx += blockDim.x * gridDim.x) {
        int d = idx % CX_EMB_DIM;
        int cell = (idx / CX_EMB_DIM) % CX_NUM_CELLS;
        int b = idx / (CX_NUM_CELLS * CX_EMB_DIM);
        device const precision_t* ids = a.src + (long)b * a.obs_size + cell * CX_CHANNELS;
        ulong g = fxp_quantize(to_float(a.weight[(long)b * CX_CONCAT + cell * CX_EMB_DIM + d]));
        for (int ch = 0; ch < CX_CHANNELS; ch++) {
            int row = CX_OFFSETS[ch] + int(to_float(ids[ch]));
            int i = row * CX_EMB_DIM + d;
            fxp_atomic_add64(&acc_lo[i], &acc_hi[i], g);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    fxp_tg_flush<CX_EMBED_N>((device long*)a.dst, acc_lo, acc_hi, threadIdx.x, blockDim.x);
}

kernel void craftax_fxp_to_precision(constant CraftaxArgs& a [[buffer(0)]],
        uint idx [[thread_position_in_grid]]) {
    if (idx >= uint(a.n)) return;
    a.dst[idx] = from_float(fxp_to_float(((device const long*)a.src)[idx]));
}
