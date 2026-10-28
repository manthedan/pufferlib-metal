// src/pufferl.cu log_reduce: Env/Log layout is supplied by the host adapter.
#include "common.h"

struct LogReduceArgs {
    device uchar* envs;
    device float* out;
    int total_agents, clear, env_stride, log_offset, num_agents_offset, n_offset, nf;
};
kernel void log_reduce(constant LogReduceArgs& a [[buffer(0)]], PUF_KERNEL_IDX,
        threadgroup float* sh [[threadgroup(0)]]) {
    uint tid = threadIdx.x;
    int count = a.total_agents / *reinterpret_cast<device int*>(a.envs + a.num_agents_offset);
    // Channel-major traversal has the same per-channel sum order as CUDA's
    // local[LOG_NF]. Clear n only after all channels have tested it.
    for (int c = 0; c < a.nf; c++) {
        float sum = 0.0f;
        for (int i = tid; i < count; i += blockDim.x) {
            device float* log = reinterpret_cast<device float*>(a.envs + i * a.env_stride + a.log_offset);
            if (log[a.n_offset / 4] != 0.0f) sum += log[c];
            if (a.clear && c != a.n_offset / 4) log[c] = 0.0f;
        }
        sh[c * blockDim.x + tid] = sum;
    }
    if (a.clear) {
        for (int i = tid; i < count; i += blockDim.x) {
            device float* log = reinterpret_cast<device float*>(a.envs + i * a.env_stride + a.log_offset);
            log[a.n_offset / 4] = 0.0f;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = blockDim.x / 2; s; s >>= 1) {
        if (tid < s) {
            for (int c = 0; c < a.nf; c++)
                sh[c * blockDim.x + tid] += sh[c * blockDim.x + tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) for (int c = 0; c < a.nf; c++) a.out[c] = sh[c * blockDim.x];
}
