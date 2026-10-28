// Metal backend kernel adapters. Included by src/metal/pufferl_metal.cpp AFTER
// src/pufferl.cu so upstream types (Prec, PrefixScan, PPOKernelArgs, ...) are
// visible. One overload per (kernel, argument types) that has an MSL port in
// src/metal/shaders/kernels.metal; anything else hits the catch-all in
// launch.mm and aborts with "metal: unimplemented kernel <name>".
//
// Host twins of the MSL argument structs use uint64_t for device pointers
// (GPU virtual addresses from puf_mtl::gpu_addr).
#pragma once

#include "launch.h"
#include "runtime.h"

namespace puf_mtl {

template <typename T>
inline void launch(const LaunchCfg& c, const char* fn, const T& args) {
    // Function-local cache keyed by MSL function name.
    struct Entry { const char* fn; Pipeline* p; };
    static thread_local Entry cache[16];
    static thread_local int n = 0;
    Pipeline* p = nullptr;
    for (int i = 0; i < n; i++) {
        if (cache[i].fn == fn) { p = cache[i].p; break; }
    }
    if (!p) {
        p = pipeline(fn);
        if (n < 16) cache[n++] = {fn, p};
    }
    dispatch(c.stream, p, c.grid, c.block, &args, sizeof(T), c.smem);
}

inline uint64_t dptr(const void* p) { return gpu_addr(p, "kernel argument"); }

}  // namespace puf_mtl

#define PUF_ADAPTER(kname, ...) \
    static inline void puf_mtl_kernel(puf_kid_##kname*, const puf_mtl::LaunchCfg& cfg, __VA_ARGS__)


// ---- cast overloads ----------------------------------------------------------

struct MtlCastPtrArgs { uint64_t dst, src; int32_t n; };

PUF_ADAPTER(cast, precision_t* dst, float* src, int n) {
    puf_mtl::launch(cfg, "cast_f32_to_prec", MtlCastPtrArgs{puf_mtl::dptr(dst), puf_mtl::dptr(src), n});
}
PUF_ADAPTER(cast, precision_t* dst, unsigned char* src, int n) {
    puf_mtl::launch(cfg, "cast_u8_to_prec", MtlCastPtrArgs{puf_mtl::dptr(dst), puf_mtl::dptr(src), n});
}
PUF_ADAPTER(cast, unsigned char* dst, precision_t* src, int n) {
    puf_mtl::launch(cfg, "cast_prec_to_u8", MtlCastPtrArgs{puf_mtl::dptr(dst), puf_mtl::dptr(src), n});
}

#ifndef PRECISION_FLOAT
PUF_ADAPTER(cast, float* dst, precision_t* src, int n) {
    puf_mtl::launch(cfg, "cast_prec_to_f32", MtlCastPtrArgs{puf_mtl::dptr(dst), puf_mtl::dptr(src), n});
}
PUF_ADAPTER(cast, precision_t* dst, precision_t* src, int n) {
    puf_mtl::launch(cfg, "cast_prec_to_prec", MtlCastPtrArgs{puf_mtl::dptr(dst), puf_mtl::dptr(src), n});
}
#endif

struct MtlCastRewTermArgs { uint64_t rew_dst, rew_src, term_dst, term_src; int32_t n; };
PUF_ADAPTER(cast_rew_term, precision_t* rew_dst, const float* rew_src,
        precision_t* term_dst, const float* term_src, int n) {
    puf_mtl::launch(cfg, "cast_rew_term", MtlCastRewTermArgs{puf_mtl::dptr(rew_dst),
        puf_mtl::dptr(rew_src), puf_mtl::dptr(term_dst), puf_mtl::dptr(term_src), n});
}

// ---- elementwise -------------------------------------------------------------

struct MtlTranspose102Args { uint64_t dst, src; int32_t A, B, C; };
PUF_ADAPTER(transpose_102, precision_t* dst, const precision_t* src, int A, int B, int C) {
    puf_mtl::launch(cfg, "transpose_102",
        MtlTranspose102Args{puf_mtl::dptr(dst), puf_mtl::dptr(src), A, B, C});
}

#ifndef PRECISION_FLOAT
PUF_ADAPTER(transpose_102, float* dst, const float* src, int A, int B, int C) {
    puf_mtl::launch(cfg, "transpose_102_f32",
        MtlTranspose102Args{puf_mtl::dptr(dst), puf_mtl::dptr(src), A, B, C});
}
#endif

struct MtlClampArgs { uint64_t dst; float lo, hi; int32_t n; };
PUF_ADAPTER(clamp_precision_kernel, precision_t* dst, float lo, float hi, int n) {
    puf_mtl::launch(cfg, "clamp_precision_kernel", MtlClampArgs{puf_mtl::dptr(dst), lo, hi, n});
}

struct MtlAddArgs { uint64_t dst, src; int32_t n; };
PUF_ADAPTER(add_kernel, precision_t* dst, const precision_t* src, int n) {
    puf_mtl::launch(cfg, "add_kernel", MtlAddArgs{puf_mtl::dptr(dst), puf_mtl::dptr(src), n});
}

struct MtlUniformScaleArgs { uint64_t data; float bound; int32_t n; };
PUF_ADAPTER(uniform_scale_kernel, float* data, float bound, int n) {
    puf_mtl::launch(cfg, "uniform_scale_kernel", MtlUniformScaleArgs{puf_mtl::dptr(data), bound, n});
}

// %globaltimer analogue: an MTL4CounterHeap timestamp (ns) resolved into *dst
// at the next host sync with the stream (train_impl syncs before reading).
PUF_ADAPTER(puf_stamp, unsigned long long* dst) {
    puf_mtl::stamp(cfg.stream, dst);
}

// ---- rollout path (pufferl.cu / algo.cu; shaders/rollout.metal) -------------

struct MtlRngInitArgs { uint64_t states; uint64_t seed; int32_t n; };
static_assert(sizeof(MtlRngInitArgs) == 24, "RngInitArgs layout");
static_assert(sizeof(curandStatePhilox4_32_10_t) == 64, "philox state layout");
PUF_ADAPTER(rng_init, curandStatePhilox4_32_10_t* states, uint64_t seed, int n) {
    puf_mtl::launch(cfg, "rng_init", MtlRngInitArgs{puf_mtl::dptr(states), seed, n});
}

#ifndef PUFFER_NETHACK  // NetHack variants live in kernels_nethack.h
struct MtlSampleLogitsArgs {
    uint64_t logits, logstd, act_sizes, actions, env_actions, logprobs, value_out,
        rng_states, action_mask;
    int32_t B, fused_cols, num_atns, mask_stride, is_continuous;
};
static_assert(sizeof(MtlSampleLogitsArgs) == 96, "SampleLogitsArgs layout");
PUF_ADAPTER(sample_logits, Prec dec_out, Prec logstd, int* act_sizes, float* actions,
        float* env_actions, precision_t* logprobs, precision_t* value_out,
        curandStatePhilox4_32_10_t* rng_states, precision_t* action_mask, int mask_stride) {
    bool is_continuous = logstd.data != NULL && numel(logstd.shape) > 0;
    puf_mtl::launch(cfg, "sample_logits", MtlSampleLogitsArgs{
        puf_mtl::dptr(dec_out.data), is_continuous ? puf_mtl::dptr(logstd.data) : 0,
        puf_mtl::dptr(act_sizes), puf_mtl::dptr(actions), puf_mtl::dptr(env_actions),
        puf_mtl::dptr(logprobs), puf_mtl::dptr(value_out), puf_mtl::dptr(rng_states),
        puf_mtl::dptr(action_mask), (int32_t)dec_out.shape[0], (int32_t)dec_out.shape[1],
        NUM_ATNS, mask_stride, is_continuous ? 1 : 0});
}
#endif

struct MtlSnapshotStateArgs {
    uint64_t dst, src;
    int32_t L, H, src_agents, dst_agents, dst_start, count;
};
static_assert(sizeof(MtlSnapshotStateArgs) == 40, "SnapshotStateArgs layout");
PUF_ADAPTER(snapshot_state, Prec dst, Prec src, int dst_start, int count) {
    puf_mtl::launch(cfg, "snapshot_state", MtlSnapshotStateArgs{
        puf_mtl::dptr(dst.data), puf_mtl::dptr(src.data), (int32_t)src.shape[0],
        (int32_t)src.shape[2], (int32_t)src.shape[1], (int32_t)dst.shape[1],
        dst_start, count});
}

struct MtlZeroTermStateArgs {
    uint64_t state, terminals;
    int32_t L, H, agents, state_start, terminal_start, count;
};
static_assert(sizeof(MtlZeroTermStateArgs) == 40, "ZeroTermStateArgs layout");
PUF_ADAPTER(zero_term_state, Prec state, Float terminals, int state_start,
        int terminal_start, int count) {
    puf_mtl::launch(cfg, "zero_term_state", MtlZeroTermStateArgs{
        puf_mtl::dptr(state.data), puf_mtl::dptr(terminals.data), (int32_t)state.shape[0],
        (int32_t)state.shape[2], (int32_t)state.shape[1], state_start, terminal_start,
        count});
}

struct MtlMingruGateArgs {
    uint64_t out, next_state, combined, state_in, x_in;
    int32_t H, B;
};
static_assert(sizeof(MtlMingruGateArgs) == 48, "MingruGateArgs layout");
PUF_ADAPTER(mingru_gate, precision_t* out, precision_t* next_state,
        const precision_t* combined, const precision_t* state_in,
        const precision_t* x_in, int H, int B) {
    puf_mtl::launch(cfg, "mingru_gate", MtlMingruGateArgs{puf_mtl::dptr(out),
        puf_mtl::dptr(next_state), puf_mtl::dptr(combined), puf_mtl::dptr(state_in),
        puf_mtl::dptr(x_in), H, B});
}

#include "kernels_mingru.h"  // MinGRU scans + decoder-backward helpers
#undef PUF_ADAPTER
#include "kernels_ppo.h"
#include "kernels_envs.h"
