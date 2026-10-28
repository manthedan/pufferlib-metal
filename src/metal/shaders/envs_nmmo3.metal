// NMMO3 conv encoder (ocean/nmmo3/nmmo3.cu, default non-N3_ATTN path).
// One thread per output element, same indexing as the CUDA kernels; the
// embedding scatter keeps CUDA's 2^24 fixed-point int64 sums (envs_fxp.h).
// precision_t storage, with FP32 arithmetic and explicit CUDA-compatible rounding.
#include "envs_fxp.h"

#define N3_MAP_H 11
#define N3_MAP_W 15
#define N3_NFEAT 10
#define N3_MAP_SIZE (N3_MAP_H * N3_MAP_W * N3_NFEAT)
#define N3_PLAYER 47
#define N3_REWARD 10
#define N3_EMBED_DIM 32
#define N3_EMBED_VOCAB 128
#define N3_PLAYER_EMBED (N3_PLAYER * N3_EMBED_DIM)
#define N3_C1_IC 59
#define N3_C1_OC 128
#define N3_C1_K 5
#define N3_C1_S 3
#define N3_C1_OH 3
#define N3_C1_OW 4
#define N3_C2_IC 128
#define N3_C2_OC 128
#define N3_C2_K 3
#define N3_C2_OH 1
#define N3_C2_OW 2
#define N3_CONV_FLAT (N3_C2_OC * N3_C2_OH * N3_C2_OW)
#define N3_CONCAT (N3_CONV_FLAT + N3_PLAYER_EMBED + N3_PLAYER + N3_REWARD)
#define N3_C1_COL_W (N3_C1_IC * N3_C1_K * N3_C1_K)
#define N3_C1_SPATIAL (N3_C1_OH * N3_C1_OW)
#define N3_C2_COL_W (N3_C2_IC * N3_C2_K * N3_C2_K)
#define N3_C2_SPATIAL (N3_C2_OH * N3_C2_OW)
#define N3_C2_IN_PLANE (N3_C2_IC * N3_C1_OH * N3_C1_OW)
#define N3_EMBED_N (N3_EMBED_VOCAB * N3_EMBED_DIM)

constant int N3_OFFSETS[10] = {0, 4, 8, 25, 30, 33, 38, 43, 48, 55};

struct N3Args {
    device precision_t* dst;
    device const precision_t* src;
    device const precision_t* src2;
    device const precision_t* src3;
    int B, obs_size, OC, spatial, flag, n;
};

// dst = col, src = obs.
kernel void n3_c1_im2col_obs(constant N3Args& a [[buffer(0)]],
        uint idx [[thread_position_in_grid]]) {
    if (idx >= uint(a.B * N3_C1_SPATIAL * N3_C1_COL_W)) return;
    int row = idx / N3_C1_COL_W;
    int c = idx % N3_C1_COL_W;
    int b = row / N3_C1_SPATIAL;
    int rem = row % N3_C1_SPATIAL;
    int oh = rem / N3_C1_OW;
    int ow = rem % N3_C1_OW;
    int ic = c / (N3_C1_K * N3_C1_K);
    int kk = c % (N3_C1_K * N3_C1_K);
    int kh = kk / N3_C1_K;
    int kw = kk % N3_C1_K;
    int ih = oh * N3_C1_S + kh;
    int iw = ow * N3_C1_S + kw;
    int f = 0;
    for (int i = 1; i < 10; i++) {
        if (ic >= N3_OFFSETS[i]) f = i;
    }
    int val = int(to_float(a.src[(long)b * a.obs_size + (long)(ih * N3_MAP_W + iw) * N3_NFEAT + f]));
    a.dst[idx] = from_float(val == ic - N3_OFFSETS[f] ? 1.0f : 0.0f);
}

// dst = col, src = conv1 output (NCHW).
kernel void n3_c2_im2col(constant N3Args& a [[buffer(0)]],
        uint idx [[thread_position_in_grid]]) {
    if (idx >= uint(a.B * N3_C2_SPATIAL * N3_C2_COL_W)) return;
    int row = idx / N3_C2_COL_W;
    int c = idx % N3_C2_COL_W;
    int b = row / N3_C2_SPATIAL;
    int ow = row % N3_C2_SPATIAL;
    int ic = c / (N3_C2_K * N3_C2_K);
    int kk = c % (N3_C2_K * N3_C2_K);
    int kh = kk / N3_C2_K;
    int kw = kk % N3_C2_K;
    a.dst[idx] = from_float(to_float(a.src[((long)b * N3_C2_IC + ic) * N3_C1_SPATIAL + kh * N3_C1_OW + (ow + kw)]));
}

// dst = grad_input (NCHW), src = col gradient.
kernel void n3_c2_col2im(constant N3Args& a [[buffer(0)]],
        uint idx [[thread_position_in_grid]]) {
    if (idx >= uint(a.B * N3_C2_IN_PLANE)) return;
    int iw = idx % N3_C1_OW;
    int ih = (idx / N3_C1_OW) % N3_C1_OH;
    int ic = (idx / N3_C1_SPATIAL) % N3_C2_IC;
    int b = idx / N3_C2_IN_PLANE;
    float sum = 0.0f;
    int base = b * N3_C2_SPATIAL * N3_C2_COL_W + ic * (N3_C2_K * N3_C2_K) + ih * N3_C2_K;
    for (int kw = 0; kw < N3_C2_K; kw++) {
        int ow = iw - kw;
        if (ow >= 0 && ow < N3_C2_OW) sum += to_float(a.src[base + ow * N3_C2_COL_W + kw]);
    }
    a.dst[idx] = from_float(sum);
}

// flag = relu.
kernel void n3_rows_to_nchw(constant N3Args& a [[buffer(0)]],
        uint idx [[thread_position_in_grid]]) {
    if (idx >= uint(a.B * a.OC * a.spatial)) return;
    int b = idx / (a.OC * a.spatial);
    int oc = (idx / a.spatial) % a.OC;
    int s = idx % a.spatial;
    float value = to_float(a.src[(b * a.spatial + s) * a.OC + oc]);
    if (a.flag) value = fmax(0.0f, value);
    a.dst[idx] = from_float(value);
}

kernel void n3_nchw_to_rows(constant N3Args& a [[buffer(0)]],
        uint idx [[thread_position_in_grid]]) {
    if (idx >= uint(a.B * a.OC * a.spatial)) return;
    int b = idx / (a.OC * a.spatial);
    int oc = (idx / a.spatial) % a.OC;
    int s = idx % a.spatial;
    a.dst[(b * a.spatial + s) * a.OC + oc] = from_float(to_float(a.src[idx]));
}

// One thread per (b, player feature): copies a 32-wide embedding row.
// dst = out, src = obs, src2 = embed_w.
kernel void n3_embedding(constant N3Args& a [[buffer(0)]],
        uint idx [[thread_position_in_grid]]) {
    if (idx >= uint(a.B * N3_PLAYER)) return;
    int b = idx / N3_PLAYER;
    int f = idx % N3_PLAYER;
    int val = int(to_float(a.src[b * a.obs_size + N3_MAP_SIZE + f]));
#ifdef PRECISION_FLOAT
    device const float4* src4 = (device const float4*)(a.src2 + val * N3_EMBED_DIM);
    device float4* dst4 = (device float4*)(a.dst + b * N3_PLAYER_EMBED + f * N3_EMBED_DIM);
    for (int i = 0; i < N3_EMBED_DIM / 4; i++) dst4[i] = src4[i];
#else
    for (int d = 0; d < N3_EMBED_DIM; d++)
        a.dst[b * N3_PLAYER_EMBED + f * N3_EMBED_DIM + d] = from_float(to_float(a.src2[val * N3_EMBED_DIM + d]));
#endif
}

// dst = concat, src = conv_flat, src2 = embed, src3 = obs.
kernel void n3_concat(constant N3Args& a [[buffer(0)]],
        uint idx [[thread_position_in_grid]]) {
    if (idx >= uint(a.B * N3_CONCAT)) return;
    int b = idx / N3_CONCAT;
    int c = idx % N3_CONCAT;
    float val;
    if (c < N3_CONV_FLAT) {
        val = to_float(a.src[b * N3_CONV_FLAT + c]);
    } else if (c < N3_CONV_FLAT + N3_PLAYER_EMBED) {
        val = to_float(a.src2[b * N3_PLAYER_EMBED + (c - N3_CONV_FLAT)]);
    } else if (c < N3_CONV_FLAT + N3_PLAYER_EMBED + N3_PLAYER) {
        val = to_float(a.src3[b * a.obs_size + N3_MAP_SIZE + (c - N3_CONV_FLAT - N3_PLAYER_EMBED)]);
    } else {
        val = to_float(a.src3[b * a.obs_size + a.obs_size - N3_REWARD
            + (c - N3_CONV_FLAT - N3_PLAYER_EMBED - N3_PLAYER)]);
    }
    a.dst[idx] = from_float(val);
}

kernel void n3_relu(constant N3Args& a [[buffer(0)]],
        uint idx [[thread_position_in_grid]]) {
    if (idx >= uint(a.n)) return;
    a.dst[idx] = from_float(fmax(0.0f, to_float(a.dst[idx])));
}

// dst = grad (in place), src = forward output.
kernel void n3_relu_backward(constant N3Args& a [[buffer(0)]],
        uint idx [[thread_position_in_grid]]) {
    if (idx >= uint(a.n)) return;
    if (to_float(a.src[idx]) <= 0.0f) a.dst[idx] = from_float(0.0f);
}

// dst = conv_grad, src = concat_grad.
kernel void n3_concat_backward_conv(constant N3Args& a [[buffer(0)]],
        uint idx [[thread_position_in_grid]]) {
    if (idx >= uint(a.B * N3_CONV_FLAT)) return;
    int b = idx / N3_CONV_FLAT;
    int c = idx % N3_CONV_FLAT;
    a.dst[b * N3_CONV_FLAT + c] = from_float(to_float(a.src[b * N3_CONCAT + c]));
}

// dst = int64 [128*32] accumulator, src = concat_grad, src2 = obs.
kernel void n3_embedding_backward(constant N3Args& a [[buffer(0)]],
        uint3 blockIdx [[threadgroup_position_in_grid]],
        uint3 blockDim [[threads_per_threadgroup]],
        uint3 threadIdx [[thread_position_in_threadgroup]],
        uint3 gridDim [[threadgroups_per_grid]]) {
    threadgroup atomic_uint acc_lo[N3_EMBED_N];
    threadgroup atomic_uint acc_hi[N3_EMBED_N];
    fxp_tg_zero<N3_EMBED_N>(acc_lo, acc_hi, threadIdx.x, blockDim.x);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    int total = a.B * N3_PLAYER;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
            idx += blockDim.x * gridDim.x) {
        int b = idx / N3_PLAYER;
        int f = idx % N3_PLAYER;
        int val = int(to_float(a.src2[b * a.obs_size + N3_MAP_SIZE + f]));
        device const precision_t* g = a.src + b * N3_CONCAT + N3_CONV_FLAT + f * N3_EMBED_DIM;
        for (int d = 0; d < N3_EMBED_DIM; d++) {
            int i = val * N3_EMBED_DIM + d;
            fxp_atomic_add64(&acc_lo[i], &acc_hi[i], fxp_quantize(to_float(g[d])));
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    fxp_tg_flush<N3_EMBED_N>((device long*)a.dst, acc_lo, acc_hi, threadIdx.x, blockDim.x);
}

kernel void n3_fxp_to_precision(constant N3Args& a [[buffer(0)]],
        uint idx [[thread_position_in_grid]]) {
    if (idx >= uint(a.n)) return;
    a.dst[idx] = from_float(fxp_to_float(((device const long*)a.src)[idx]));
}
