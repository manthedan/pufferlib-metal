// Small dense encoder / MinGRU / dense decoder inference. Argument addresses
// are copied into the runtime's fenced constant ring, never a temporary GPU
// pointer table. Optional terminals reset the input state; initial_state saves
// that post-reset state before recurrent updates. Sampling follows unchanged.
#pragma once

#include "runtime.h"

namespace puf_mtl {

struct FusedRolloutArgs {
    uint64_t obs, encoder, recurrent[4], decoder, state, out;
    uint64_t terminals, initial_state; // optional compact terminals / post-reset snapshot
    int32_t initial_agents; // layer stride of initial_state, in agents
    int32_t B, input_dim, hidden, layers, output_dim;
};
static_assert(sizeof(FusedRolloutArgs) == 112, "FusedRolloutArgs layout");

inline bool fused_rollout_supported(int B, int input, int H, int L, int output) {
    // Low-level kernel shapes. arch_forward separately limits automatic use
    // to H=32, the only variant with a measured end-to-end speedup.
    return B > 0 && input > 0 && input <= 256 && (H == 32 || H == 64 || H == 128)
        && L > 0 && L <= 4 && output > 0 && output <= 32;
}

inline void fused_rollout(cudaStream_t stream, const FusedRolloutArgs& a) {
    if (a.initial_state && a.initial_agents < a.B) fatal("fused rollout: initial-state stride < batch");
    Pipeline* p = pipeline(a.hidden == 32 ? "rollout_forward_h32" :
        a.hidden == 64 ? "rollout_forward_h64" : "rollout_forward_h128");
    int groups = a.hidden == 32 ? 4 : a.hidden == 64 ? 2 : 1;
    dispatch(stream, p, dim3((a.B + 8 * groups - 1) / (8 * groups)), dim3(32 * groups), a);
}

// One rollout step in one dispatch (shaders/rollout.metal rollout_step_h32 /
// rollout_step_h64): env obs copy (precision_t, bitwise; u8 or float cast),
// reward / terminal / u8 action-mask casts, the H=32 or H=64 fused forward
// (with terminal reset and snapshot) and discrete sampling. smp mirrors the
// MSL SampleLogitsArgs (kernels.h MtlSampleLogitsArgs).
struct StepSampleArgs {
    uint64_t logits, logstd, act_sizes, actions, env_actions, logprobs, value_out,
        rng_states, action_mask;
    int32_t B, fused_cols, num_atns, mask_stride, is_continuous;
};
static_assert(sizeof(StepSampleArgs) == 96, "SampleLogitsArgs layout");
struct RolloutStepArgs {
    FusedRolloutArgs fwd;
    StepSampleArgs smp;
    uint64_t obs_src, rew_dst, rew_src, term_dst, term_src, mask_dst, mask_src;
    int32_t mask_size, obs_is_u8, obs_row_stride, pad; // obs_is_u8: 0 precision_t,
    // 1 u8, 2 float (cast to bf16); 0 stride means input_dim
};
static_assert(sizeof(RolloutStepArgs) == 280, "RolloutStepArgs layout");

inline bool rollout_step_supported(const RolloutStepArgs& a) {
    const FusedRolloutArgs& f = a.fwd;
    return (f.hidden == 32 || f.hidden == 64) && fused_rollout_supported(f.B, f.input_dim, f.hidden, f.layers, f.output_dim)
        && (!f.initial_state || f.initial_agents >= f.B)
        && (a.obs_row_stride == 0 || a.obs_row_stride >= f.input_dim)
        && a.smp.B == f.B && a.smp.fused_cols == f.output_dim && !a.smp.is_continuous
        && a.smp.logits == f.out && a.smp.action_mask == a.mask_dst
        && a.mask_size >= 0 && a.mask_size <= 256 && a.smp.mask_stride == a.mask_size;
}

inline void rollout_step(cudaStream_t stream, const RolloutStepArgs& a) {
    if (!rollout_step_supported(a)) fatal("rollout step: unsupported shape");
    static Pipeline* p32 = pipeline("rollout_step_h32");
    static Pipeline* p64 = pipeline("rollout_step_h64");
    if (a.fwd.hidden == 32) dispatch(stream, p32, dim3((a.fwd.B + 31) / 32), dim3(128), a);
    else dispatch(stream, p64, dim3((a.fwd.B + 15) / 16), dim3(64), a);
}

// Packed rollout weights (shaders/rollout.metal rollout_pack_weights): float
// 8x8 right-hand MMA tiles of the encoder (H/8 x ceil(K/8)), each recurrent
// layer (3H/8 x H/8) and the decoder (ceil(D/8) x H/8), in that order.
struct RolloutPackArgs {
    uint64_t encoder, recurrent[4], decoder, packed;
    int32_t K, H, L, D;
};
static_assert(sizeof(RolloutPackArgs) == 72, "RolloutPackArgs layout");

inline size_t rollout_packed_encoder(int K, int H) { return (size_t)(H / 8) * ((K + 7) / 8) * 64; }
inline size_t rollout_packed_layer(int H) { return (size_t)(3 * H / 8) * (H / 8) * 64; }
inline size_t rollout_packed_floats(int K, int H, int L, int D) {
    return rollout_packed_encoder(K, H) + (size_t)L * rollout_packed_layer(H)
        + (size_t)((D + 7) / 8) * (H / 8) * 64;
}

inline void rollout_pack(cudaStream_t stream, const RolloutPackArgs& a) {
    if ((a.H != 32 && a.H != 64) || a.L < 1 || a.L > 4 || a.K < 1 || a.D < 1) {
        fatal("rollout pack: unsupported shape");
    }
    static Pipeline* p = pipeline("rollout_pack_weights");
    size_t n = rollout_packed_floats(a.K, a.H, a.L, a.D);
    dispatch(stream, p, dim3((unsigned)((n + 255) / 256)), dim3(256), a);
}

// Points the step's weight arguments at the packed tiles of `packed`.
inline void rollout_use_packed(FusedRolloutArgs& f, uint64_t packed) {
    size_t off = rollout_packed_encoder(f.input_dim, f.hidden);
    f.encoder = packed;
    for (int l = 0; l < 4; l++) {
        f.recurrent[l] = l < f.layers ? packed + 4 * (off + (size_t)l * rollout_packed_layer(f.hidden)) : 0;
    }
    f.decoder = packed + 4 * (off + (size_t)f.layers * rollout_packed_layer(f.hidden));
}

// rollout_step with packed weights (see rollout_use_packed). split: simdgroups
// per 8-agent group (H=64: 1, 2 or 4; H=32: 1 or 2).
inline void rollout_step_packed(cudaStream_t stream, const RolloutStepArgs& a, int split) {
    if (!rollout_step_supported(a)) fatal("rollout step: unsupported shape");
    static Pipeline* p64[3] = {pipeline("rollout_step_packed_h64_p1"),
        pipeline("rollout_step_packed_h64_p2"), pipeline("rollout_step_packed_h64_p4")};
    static Pipeline* p32[2] = {pipeline("rollout_step_packed_h32_p1"),
        pipeline("rollout_step_packed_h32_p2")};
    int B = a.fwd.B;
    if (a.fwd.hidden == 64 && split == 1) dispatch(stream, p64[0], dim3((B + 15) / 16), dim3(64), a);
    else if (a.fwd.hidden == 64 && split == 2) dispatch(stream, p64[1], dim3((B + 15) / 16), dim3(128), a);
    else if (a.fwd.hidden == 64 && split == 4) dispatch(stream, p64[2], dim3((B + 7) / 8), dim3(128), a);
    else if (a.fwd.hidden == 32 && split == 1) dispatch(stream, p32[0], dim3((B + 31) / 32), dim3(128), a);
    else if (a.fwd.hidden == 32 && split == 2) dispatch(stream, p32[1], dim3((B + 15) / 16), dim3(128), a);
    else fatal("rollout step packed: unsupported split %d for H=%d", split, a.fwd.hidden);
}

// Independent buffer arguments along dispatch grid Y; per-buffer RNG/state
// and the arithmetic inside each threadgroup are unchanged.
inline void rollout_steps_packed(cudaStream_t stream, const RolloutStepArgs* args, int count, int split) {
    if (count < 2 || count > 4) fatal("rollout steps: unsupported buffer count");
    const RolloutStepArgs& a = args[0];
    for (int i = 0; i < count; i++) {
        if (!rollout_step_supported(args[i]) || args[i].fwd.B != a.fwd.B
                || args[i].fwd.hidden != a.fwd.hidden) fatal("rollout steps: unsupported shape");
    }
    static Pipeline* p64[3] = {pipeline("rollout_steps_packed_h64_p1"),
        pipeline("rollout_steps_packed_h64_p2"), pipeline("rollout_steps_packed_h64_p4")};
    static Pipeline* p32[2] = {pipeline("rollout_steps_packed_h32_p1"),
        pipeline("rollout_steps_packed_h32_p2")};
    int B = a.fwd.B;
    Pipeline* p = nullptr;
    int groups = 0, threads = 128;
    if (a.fwd.hidden == 64 && split == 1) { p = p64[0]; groups = 2; threads = 64; }
    else if (a.fwd.hidden == 64 && split == 2) { p = p64[1]; groups = 2; }
    else if (a.fwd.hidden == 64 && split == 4) { p = p64[2]; groups = 1; }
    else if (a.fwd.hidden == 32 && split == 1) { p = p32[0]; groups = 4; }
    else if (a.fwd.hidden == 32 && split == 2) { p = p32[1]; groups = 2; }
    else fatal("rollout step packed: unsupported split %d for H=%d", split, a.fwd.hidden);
    dispatch(stream, p, dim3((B + 8 * groups - 1) / (8 * groups), count), dim3(threads),
        (const void*)args, count * sizeof(*args));
}

}  // namespace puf_mtl
