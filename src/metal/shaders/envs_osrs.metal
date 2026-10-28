// OSRS entity encoder (osrs_colosseum / inferno / pvp / zulrah).
// Mirrors ocean/osrs/osrs_entity_encoder.cu kernel for kernel, with the same
// grid/threadgroup geometry and per-thread accumulation order. Validated
// against CUDA goldens (tests/metal/run_test_env_golden.sh).
// precision_t storage, with FP32 arithmetic and explicit CUDA-compatible rounding.
#include "common.h"

#define OSRS_BOTTLENECK 16
#define OSRS_BATCH_TILE 8
#define OSRS_HIDDEN_TILE 32
#define OSRS_FC_THREADS (OSRS_BATCH_TILE * OSRS_HIDDEN_TILE)
// Static threadgroup tiles; the host adapter checks num_records against this.
#define OSRS_MAX_RECORDS 32
#define OSRS_GRAD_THREADS 256

static inline float osrs_gelu_fwd(float x) {
    float inner = 0.7978845608028654f * (x + 0.044715f * x * x * x);
    return 0.5f * x * (1.0f + tanh(inner));
}

static inline float osrs_gelu_grad(float x) {
    float x3 = x * x * x;
    float inner = 0.7978845608028654f * (x + 0.044715f * x3);
    float t = tanh(inner);
    float dinner = 0.7978845608028654f * (1.0f + 3.0f * 0.044715f * x * x);
    return 0.5f * (1.0f + t) + 0.5f * x * (1.0f - t * t) * dinner;
}

struct OsrsGatherArgs {
    device precision_t* flat;
    device const precision_t* obs;
    device const float* table;   // [table_rows][table_cols] item observation table
    int B, obs_size, obs_start, num_records, obs_features, type_onehot,
        type_code_scale, expansion, table_rows, table_cols;
};

kernel void osrs_entity_gather_branch(constant OsrsGatherArgs& a [[buffer(0)]],
        uint idx [[thread_position_in_grid]]) {
    bool item_table = a.expansion == 1;
    int features = item_table ? a.table_cols : a.type_onehot + a.obs_features - 1;
    int record_block = a.num_records * features;
    if (idx >= uint(a.B * record_block)) return;
    int b = idx / record_block;
    int off = idx - b * record_block;
    int record = off / features;
    int feature = off - record * features;
    device const precision_t* src = a.obs + (long)b * a.obs_size + a.obs_start
        + record * a.obs_features;
    int code = int(rint(to_float(src[0]) * float(a.type_code_scale)));
    // CUDA asserts on out-of-range codes; Metal has no device assert, so the
    // record becomes NaN instead (never an out-of-bounds read).
    if (item_table) {
        a.flat[idx] = from_float((code >= 0 && code < a.table_rows)
            ? a.table[code * a.table_cols + feature] : NAN);
    } else if (code < 0 || code > a.type_onehot) {
        a.flat[idx] = from_float(NAN);
    } else if (feature < a.type_onehot) {
        a.flat[idx] = from_float(code == feature + 1 ? 1.0f : 0.0f);
    } else {
        a.flat[idx] = from_float(to_float(src[1 + feature - a.type_onehot]));
    }
}

struct OsrsPoolArgs {
    device precision_t* out;
    device int* argmax;          // nullable (rollout)
    device precision_t* h1;            // nullable (rollout)
    device const precision_t* z1;
    device const precision_t* record_flat;
    device const precision_t* l2_w;
    int B, H, num_records, record_features, active_width;
};

// Threadgroup (32 hidden, 8 batch), grid (ceil(B/8), ceil(H/32)).
kernel void osrs_entity_fused_pool_fwd(constant OsrsPoolArgs& a [[buffer(0)]],
        uint3 blockIdx [[threadgroup_position_in_grid]],
        uint3 threadIdx [[thread_position_in_threadgroup]]) {
    threadgroup float h1_tile[OSRS_BATCH_TILE * OSRS_MAX_RECORDS * OSRS_BOTTLENECK];
    threadgroup float mask_tile[OSRS_BATCH_TILE * OSRS_MAX_RECORDS];
    threadgroup float w_tile[OSRS_HIDDEN_TILE * OSRS_BOTTLENECK];
    int R = a.num_records;
    int batch_base = blockIdx.x * OSRS_BATCH_TILE;
    int hidden_base = blockIdx.y * OSRS_HIDDEN_TILE;
    int tx = threadIdx.x, ty = threadIdx.y;
    int tid = ty * OSRS_HIDDEN_TILE + tx;
    int h1_values = OSRS_BATCH_TILE * R * OSRS_BOTTLENECK;
    for (int idx = tid; idx < h1_values; idx += OSRS_FC_THREADS) {
        int bt = idx / (R * OSRS_BOTTLENECK);
        int rem = idx - bt * R * OSRS_BOTTLENECK;
        int gb = batch_base + bt;
        float value = gb < a.B
            ? osrs_gelu_fwd(to_float(a.z1[(long)gb * R * OSRS_BOTTLENECK + rem])) : 0.0f;
        h1_tile[idx] = value;
        if (a.h1 && blockIdx.y == 0 && gb < a.B)
            a.h1[(long)gb * R * OSRS_BOTTLENECK + rem] = from_float(value);
    }
    int mask_values = OSRS_BATCH_TILE * R;
    for (int idx = tid; idx < mask_values; idx += OSRS_FC_THREADS) {
        int bt = idx / R;
        int record = idx - bt * R;
        int gb = batch_base + bt;
        float active = 0.0f;
        if (gb < a.B) {
            device const precision_t* rec = a.record_flat
                + ((long)gb * R + record) * a.record_features;
            for (int f = 0; f < a.active_width; f++) active += to_float(rec[f]);
        }
        mask_tile[idx] = active;
    }
    int w_values = OSRS_HIDDEN_TILE * OSRS_BOTTLENECK;
    for (int idx = tid; idx < w_values; idx += OSRS_FC_THREADS) {
        int d = idx / OSRS_HIDDEN_TILE;
        int th = idx - d * OSRS_HIDDEN_TILE;
        int gh = hidden_base + th;
        w_tile[idx] = gh < a.H ? to_float(a.l2_w[(long)gh * OSRS_BOTTLENECK + d]) : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    int b = batch_base + ty;
    int h = hidden_base + tx;
    if (b >= a.B || h >= a.H) return;
    float best = -3.4028234663852886e38f;
    int best_record = -1;
    for (int record = 0; record < R; record++) {
        if (mask_tile[ty * R + record] <= 0.0f) continue;
        threadgroup const float* hp = h1_tile + (ty * R + record) * OSRS_BOTTLENECK;
        float sum = 0.0f;
        for (int d = 0; d < OSRS_BOTTLENECK; d++)
            sum = fma(w_tile[d * OSRS_HIDDEN_TILE + tx], to_float(hp[d]), sum);
        if (sum > best) { best = sum; best_record = record; }
    }
    long out_idx = (long)b * a.H + h;
    a.out[out_idx] = from_float(to_float(a.out[out_idx]) + (best_record < 0 ? 0.0f : best));
    if (a.argmax) a.argmax[out_idx] = best_record;
}

struct OsrsBwdArgs {
    device precision_t* dst;           // l2 wgrad [H,16] or grad_z1 [B*R,16]
    device const precision_t* grad;    // [B,H]
    device const precision_t* src;     // h1 (wgrad) or l2_w (grad_z1)
    device const precision_t* z1;
    device const int* argmax;
    int B, H, num_records;
};

// One threadgroup per hidden unit; 256 threads stride over the batch.
kernel void osrs_entity_fused_l2_wgrad(constant OsrsBwdArgs& a [[buffer(0)]],
        uint3 blockIdx [[threadgroup_position_in_grid]],
        uint3 threadIdx [[thread_position_in_threadgroup]],
        uint3 blockDim [[threads_per_threadgroup]],
        uint lane [[thread_index_in_simdgroup]],
        uint warp [[simdgroup_index_in_threadgroup]],
        uint num_warps [[simdgroups_per_threadgroup]]) {
    threadgroup float warp_sums[OSRS_BOTTLENECK * 32];
    int h = blockIdx.x;
    float sum[OSRS_BOTTLENECK];
    for (int k = 0; k < OSRS_BOTTLENECK; k++) sum[k] = 0.0f;
    for (int b = threadIdx.x; b < a.B; b += blockDim.x) {
        int record = a.argmax[(long)b * a.H + h];
        if (record < 0) continue;
        float g = to_float(a.grad[(long)b * a.H + h]);
        device const precision_t* hp = a.src + ((long)b * a.num_records + record) * OSRS_BOTTLENECK;
        for (int k = 0; k < OSRS_BOTTLENECK; k++) sum[k] = fma(g, to_float(hp[k]), sum[k]);
    }
    for (int k = 0; k < OSRS_BOTTLENECK; k++) {
        float value = simd_sum(sum[k]);
        if (lane == 0) warp_sums[k * 32 + warp] = value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (threadIdx.x < OSRS_BOTTLENECK) {
        int k = threadIdx.x;
        float value = 0.0f;
        for (uint w = 0; w < num_warps; w++) value += warp_sums[k * 32 + w];
        a.dst[(long)h * OSRS_BOTTLENECK + k] = from_float(value);
    }
}

// One threadgroup (256 threads) per batch row; hidden units are processed in
// 256-wide tiles exactly as the CUDA kernel does.
kernel void osrs_entity_fused_grad_z1(constant OsrsBwdArgs& a [[buffer(0)]],
        uint3 blockIdx [[threadgroup_position_in_grid]],
        uint3 threadIdx [[thread_position_in_threadgroup]]) {
    threadgroup float accum[OSRS_MAX_RECORDS * OSRS_BOTTLENECK];
    threadgroup int arg_s[OSRS_GRAD_THREADS];
    threadgroup float grad_s[OSRS_GRAD_THREADS];
    int b = blockIdx.x;
    int t = threadIdx.x;
    int n = a.num_records * OSRS_BOTTLENECK;
    for (int idx = t; idx < n; idx += OSRS_GRAD_THREADS) accum[idx] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int base = 0; base < a.H; base += OSRS_GRAD_THREADS) {
        int h = base + t;
        if (h < a.H) {
            arg_s[t] = a.argmax[(long)b * a.H + h];
            grad_s[t] = to_float(a.grad[(long)b * a.H + h]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        int tile = min(a.H - base, OSRS_GRAD_THREADS);
        for (int idx = t; idx < n; idx += OSRS_GRAD_THREADS) {
            int record = idx / OSRS_BOTTLENECK;
            int k = idx - record * OSRS_BOTTLENECK;
            float sum = 0.0f;
            for (int j = 0; j < tile; j++) {
                if (arg_s[j] != record) continue;
                sum = fma(grad_s[j], to_float(a.src[(long)(base + j) * OSRS_BOTTLENECK + k]), sum);
            }
            accum[idx] += sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (int idx = t; idx < n; idx += OSRS_GRAD_THREADS) {
        long out_idx = (long)b * n + idx;
        a.dst[out_idx] = from_float(accum[idx] * osrs_gelu_grad(to_float(a.z1[out_idx])));
    }
}
