// The CUDA policy_* golden cases exercise the training scan, not the
// sequential rollout state update. Compare the fused forward against the
// original Metal rollout operations for 256 steps, carrying state and RNG.
// On M4 the FP32 MMA accumulation matches tensor_ops/tiled GEMM exactly:
// require every float and all 64 Philox bytes bit-exact at every step.
// Inputs include ragged tiles, masks, terminal resets, odd continuous head
// counts (Box-Muller cache), and a non-default stream with ring rotations.
// H=32 discrete cases also run pufferl.cu's single-dispatch rollout step
// (rollout_step_h32: float or u8 obs copy, reward / terminal / u8 mask casts
// + forward +
// sampling) as a third path and compare it bit-exactly as well.
// Built twice: FP32 (-DPRECISION_FLOAT, float storage) and BF16 (default
// metallib, __nv_bfloat16 storage; the fused kernels round at the same points
// as the generic bf16 GEMM / gate stores): tests/metal/run_test_rollout.sh.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "compat/cuda_bf16.h"
#include "compat/curand_kernel.h"
#include "gemm.h"
#include "launch.h"
#include "rollout.h"

#ifdef PRECISION_FLOAT
typedef float precision_t;
#else
typedef __nv_bfloat16 precision_t;
#endif
typedef precision_t P;
#define PUF_MAX_DIMS 8
typedef struct { float* data; int64_t shape[PUF_MAX_DIMS]; } Float;
typedef struct { precision_t* data; int64_t shape[PUF_MAX_DIMS]; } Prec;
static int64_t numel(const int64_t* shape) {
    int64_t n = 1;
    for (int i = 0; i < PUF_MAX_DIMS && shape[i]; i++) n *= shape[i];
    return n;
}
static int g_heads;
#define NUM_ATNS g_heads
#include "kernels.h"

static int grid(int n) { return (n + 255) / 256; }
template <typename T> static T* alloc(size_t n) { return (T*)puf_mtl::alloc(n * sizeof(T)); }

// NaN in both float and bf16 (0x7fc1 halfwords): a store the step skips
// can't pass with a value left by an earlier step.
template <typename T> static void poison(T* p, size_t n) {
    std::fill_n((uint16_t*)p, n * sizeof(T) / 2, (uint16_t)0x7fc1);
}

template <typename T>
static void compare(const char* what, const T* a, const T* b, size_t n,
        double& max_error) {
    for (size_t i = 0; i < n; i++) {
        double x = (float)a[i], y = (float)b[i];
        double err = fabs(x - y);
        max_error = std::max(max_error, err);
        if (!std::isfinite(err) || memcmp(a + i, b + i, sizeof(T)) != 0) {
            fprintf(stderr, "%s[%zu]: fused %.9g original %.9g abs %.3g\n", what, i, x, y, err);
            exit(1);
        }
    }
}

// 0: the step path is rollout_step; s > 0: rollout_step_packed with split s
// over weights packed (rollout_pack) into a NaN-poisoned buffer.
static int g_step_split = 0;

static void run(int B, int K, int H, int L, std::vector<int> heads, bool continuous) {
    const int steps = 256;
    g_heads = (int)heads.size();
    int D = continuous ? g_heads + 1 : 1;
    if (!continuous) for (int h : heads) D += h;
    if (!puf_mtl::fused_rollout_supported(B, K, H, L, D)) abort();
    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    std::vector<void*> allocations;
    auto floats = [&](size_t n) {
        float* p = alloc<float>(n); allocations.push_back(p); return p;
    };
    auto precs = [&](size_t n) {
        P* p = alloc<P>(n); allocations.push_back(p); return p;
    };
    P* obs = precs((size_t)B * K);
    P* enc = precs((size_t)H * K);
    P* dec = precs((size_t)D * H);
    P* W[4] = {};
    for (int l = 0; l < L; l++) W[l] = precs(3 * H * H);
    const bool step = (H == 32 || H == 64) && !continuous;
    const int paths = step ? 3 : 2;
    P* st[3] = {precs((size_t)L * B * H), precs((size_t)L * B * H), precs((size_t)L * B * H)};
    P* out[3] = {precs((size_t)B * D), precs((size_t)B * D), precs((size_t)B * D)};
    float* act[3], *env_act[3];
    P* lp[3], *val[3];
    for (int i = 0; i < 3; i++) {
        act[i] = floats((size_t)B * g_heads); env_act[i] = floats((size_t)B * g_heads);
        lp[i] = precs(B); val[i] = precs(B);
    }
    unsigned char* obs_u8 = alloc<unsigned char>((size_t)B * K); allocations.push_back(obs_u8);
    unsigned char* mask_u8 = alloc<unsigned char>((size_t)B * (D + 3)); allocations.push_back(mask_u8);
    P* obs_f32 = precs((size_t)B * K);  // env observations of precision_t (bitwise copy)
    float* obs_float = alloc<float>((size_t)B * K); allocations.push_back(obs_float);  // obs_is_u8 2
    // Two 7-step slots, with head/tail guards. Keep the other slot and every
    // unselected time row intact while mixing compact and strided copies.
    const size_t step_slot = (size_t)B * 7 * K;
    const size_t step_n = 2 * step_slot + 32;
    P* step_storage = precs(step_n);
    std::fill_n(step_storage, step_n, P(17.0f));
    std::vector<P> step_expected(step_n, P(17.0f));
    P* step_mask = precs((size_t)B * (D + 3));
    float* rew_src = floats(B);
    P* rew_dst = precs(B), *term_dst = precs(B);
    P* x = precs((size_t)B * H);
    P* gate_out = precs((size_t)B * H);
    P* next = precs((size_t)B * H);
    P* combined = precs((size_t)B * 3 * H);
    P* mask = precs((size_t)B * (D + 3)); // nontrivial row stride
    P* logstd = continuous ? precs(g_heads) : nullptr;
    float* terminals = floats(B + 11);
    const int initial_agents = B + 7;
    const size_t initial_n = (size_t)L * initial_agents * H;
    P* initial[3] = {precs(initial_n), precs(initial_n), precs(initial_n)};
    std::fill_n(terminals, B + 11, 13.0f);
    for (auto* p : initial) std::fill_n(p, initial_n, P(17.0f));
    int* sizes = alloc<int>(g_heads); allocations.push_back(sizes);
    memcpy(sizes, heads.data(), g_heads * sizeof(int));
    auto* rng0 = alloc<curandStatePhilox4_32_10_t>(B); allocations.push_back(rng0);
    auto* rng1 = alloc<curandStatePhilox4_32_10_t>(B); allocations.push_back(rng1);
    auto* rng2 = alloc<curandStatePhilox4_32_10_t>(B); allocations.push_back(rng2);
    curandStatePhilox4_32_10_t* rng[3] = {rng0, rng1, rng2};
    std::mt19937 gen(73);
    std::uniform_real_distribution<float> uniform(-1.0f, 1.0f);
    auto fill = [&](auto* p, size_t n, float scale) {
        for (size_t i = 0; i < n; i++) p[i] = uniform(gen) * scale;
    };
    fill(enc, H * K, sqrtf(6.0f / K));
    fill(dec, D * H, sqrtf(3.0f / H));
    for (int l = 0; l < L; l++) fill(W[l], 3 * H * H, sqrtf(3.0f / H));
    fill(st[0], (size_t)L * B * H, 0.5f);
    memcpy(st[1], st[0], (size_t)L * B * H * sizeof(P));
    memcpy(st[2], st[0], (size_t)L * B * H * sizeof(P));
    if (continuous) fill(logstd, g_heads, 0.5f);
    for (int i = 0; i < 3; i++) {
        PUF_LAUNCH(rng_init, grid(B), 256, 0, stream)(rng[i], uint64_t(0x123456789abcdef0), B);
    }
    puf_mtl::FusedRolloutArgs a = {};
    a.obs = puf_mtl::gpu_addr(obs); a.encoder = puf_mtl::gpu_addr(enc);
    for (int l = 0; l < L; l++) a.recurrent[l] = puf_mtl::gpu_addr(W[l]);
    a.decoder = puf_mtl::gpu_addr(dec); a.state = puf_mtl::gpu_addr(st[1]);
    a.out = puf_mtl::gpu_addr(out[1]);
    a.B = B; a.input_dim = K; a.hidden = H; a.layers = L; a.output_dim = D;
    double state_err = 0, logits_err = 0, sample_err = 0;
    auto mm = [&](const P* A, const P* W, P* C, int N, int reduction) {
        puf_mtl::gemm_rm(stream, false, true, B, N, reduction, 1.0f,
            A, reduction, W, reduction, 0.0f, C, N);
    };
    uint64_t packed = 0;
    if (step && g_step_split) {
        size_t n = puf_mtl::rollout_packed_floats(K, H, L, D);
        float* p = alloc<float>(n + 64);
        allocations.push_back(p);
        poison(p, n + 64);
        packed = puf_mtl::gpu_addr(p);
        puf_mtl::RolloutPackArgs pa = {puf_mtl::gpu_addr(enc), {}, puf_mtl::gpu_addr(dec), packed, K, H, L, D};
        for (int l = 0; l < L; l++) pa.recurrent[l] = puf_mtl::gpu_addr(W[l]);
        puf_mtl::rollout_pack(stream, pa);
        cudaStreamSynchronize(stream);
        // Guard past the packed tiles stays poisoned.
        for (size_t i = n; i < n + 64; i++) if (((uint32_t*)p)[i] != 0x7fc17fc1u) {
            fprintf(stderr, "rollout_pack wrote past its tiles at %zu\n", i); exit(1);
        }
    }
    cudaStreamSynchronize(stream);
    for (int t = 0; t < steps; t++) {
        const int step_stride = t % 4 == 0 ? K : 7 * K;
        const size_t step_offset = 16 + ((t / 7) % 2) * step_slot
            + (step_stride == K ? 0 : (t % 7) * K);
        P* step_obs = step_storage + step_offset;
        // Prior iteration synchronized before any shared allocation is touched.
        if (step) {
            // t % 4 == 1: precision_t env observations (copied bitwise); 3:
            // float, cast (obs_is_u8 2). Even: bytes (full range / small
            // integers); the reference sees their casts.
            if (t % 4 == 3) {
                // float env observations, cast to precision_t (cast_f32_to_prec).
                for (size_t i = 0; i < (size_t)B * K; i++) {
                    obs_float[i] = ((float)(gen() >> 8) / 16777216.0f - 0.5f) * 3.0f;
                    obs[i] = P(obs_float[i]);
                }
            } else if (t % 2) {
                fill(obs, (size_t)B * K, 1.0f);
                memcpy(obs_f32, obs, (size_t)B * K * sizeof(P));
            } else {
                for (size_t i = 0; i < (size_t)B * K; i++) {
                    obs_u8[i] = (unsigned char)(t % 4 ? gen() >> 24 : gen() % 3);
                    obs[i] = P((float)obs_u8[i]);
                }
            }
            fill(rew_src, B, 2.0f);
        } else {
            fill(obs, (size_t)B * K, 1.0f);
        }
        for (int b = 0; b < B; b++) {
            terminals[b + 3] = ((b * 7 + t) % 31 == 0) ? (t % 2 ? -1.0f : 1.0f) : 0.0f;
            int offset = 0;
            for (int h : heads) {
                for (int j = 0; j < h; j++) mask[b * (D + 3) + offset + j] =
                    P(j == 0 || ((j + b + t) % 3) ? 1.0f : 0.0f);
                offset += h;
            }
            // Env masks are bytes; the step casts the whole (D + 3)-wide row.
            for (int j = 0; j < D + 3; j++)
                mask_u8[b * (D + 3) + j] = j < D - 1 ? (unsigned char)(float)mask[b * (D + 3) + j] : 2;
        }
        // Exercise reset/no-reset and snapshot/no-snapshot independently.
        Prec state = {st[0], {L, B, H}};
        if (t % 3) {
            Float term = {terminals, {B + 11}};
            PUF_LAUNCH(zero_term_state, grid(L * B * H), 256, 0, stream)(state, term, 0, 3, B);
        }
        if (t % 5 == 0) {
            Prec dst = {initial[0], {L, initial_agents, H}};
            PUF_LAUNCH(snapshot_state, grid(L * B * H), 256, 0, stream)(dst, state, 3, B);
        }
        a.terminals = t % 3 ? puf_mtl::gpu_addr(terminals + 3) : 0;
        a.initial_state = t % 5 == 0 ? puf_mtl::gpu_addr(initial[1] + 3 * H) : 0;
        a.initial_agents = initial_agents;
        // Original algo.cu arch_forward / mingru_forward.
        mm(obs, enc, x, H, K);
        P* input = x;
        for (int l = 0; l < L; l++) {
            mm(input, W[l], combined, 3 * H, H);
            PUF_LAUNCH(mingru_gate, grid(B * H), 256, 0, stream)(gate_out, next,
                combined, st[0] + (size_t)l * B * H, input, H, B);
            cudaMemcpyAsync(st[0] + (size_t)l * B * H, next,
                (size_t)B * H * sizeof(P), cudaMemcpyDeviceToDevice, stream);
            input = gate_out;
        }
        mm(input, dec, out[0], D, H);
        puf_mtl::fused_rollout(stream, a);
        if (step) {
            puf_mtl::RolloutStepArgs sa = {};
            sa.fwd = a;
            sa.fwd.obs = puf_mtl::gpu_addr(step_obs);
            sa.fwd.state = puf_mtl::gpu_addr(st[2]);
            sa.fwd.out = puf_mtl::gpu_addr(out[2]);
            sa.fwd.initial_state = t % 5 == 0 ? puf_mtl::gpu_addr(initial[2] + 3 * H) : 0;
            sa.smp = {sa.fwd.out, 0, puf_mtl::gpu_addr(sizes), puf_mtl::gpu_addr(act[2]),
                puf_mtl::gpu_addr(env_act[2]), puf_mtl::gpu_addr(lp[2]), puf_mtl::gpu_addr(val[2]),
                puf_mtl::gpu_addr(rng[2]), puf_mtl::gpu_addr(step_mask), B, D, g_heads, D + 3, 0};
            sa.obs_src = puf_mtl::gpu_addr(t % 4 == 3 ? (void*)obs_float
                : t % 2 ? (void*)obs_f32 : (void*)obs_u8);
            sa.obs_is_u8 = t % 4 == 3 ? 2 : t % 2 ? 0 : 1;
            sa.obs_row_stride = step_stride == K ? 0 : step_stride;
            for (int b = 0; b < B; ++b) {
                memcpy(step_expected.data() + step_offset + (size_t)b * step_stride,
                    obs + (size_t)b * K, K * sizeof(P));
            }
            sa.rew_dst = puf_mtl::gpu_addr(rew_dst); sa.rew_src = puf_mtl::gpu_addr(rew_src);
            sa.term_dst = puf_mtl::gpu_addr(term_dst);
            sa.term_src = puf_mtl::gpu_addr(terminals + 3);
            sa.mask_dst = puf_mtl::gpu_addr(step_mask); sa.mask_src = puf_mtl::gpu_addr(mask_u8);
            sa.mask_size = D + 3;
            if (!puf_mtl::rollout_step_supported(sa)) abort();
            poison(out[2], (size_t)B * D); poison(act[2], (size_t)B * g_heads);
            poison(env_act[2], (size_t)B * g_heads); poison(lp[2], B); poison(val[2], B);
            poison(rew_dst, B); poison(term_dst, B); poison(step_mask, (size_t)B * (D + 3));
            if (packed) {
                puf_mtl::rollout_use_packed(sa.fwd, packed);
                puf_mtl::rollout_step_packed(stream, sa, g_step_split);
            } else {
                puf_mtl::rollout_step(stream, sa);
            }
        }
        for (int i = 0; i < 2; i++) {
            Prec logits = {out[i], {B, D}}, logs = {logstd, {1, continuous ? g_heads : 0}};
            PUF_LAUNCH(sample_logits, grid(B), 256, 0, stream)(logits, logs, sizes,
                act[i], env_act[i], lp[i], val[i], rng[i], mask, D + 3);
        }
        cudaStreamSynchronize(stream);
        compare("post-reset snapshot and guards", initial[1], initial[0], initial_n, state_err);
        compare("state", st[1], st[0], (size_t)L * B * H, state_err);
        compare("logits", out[1], out[0], (size_t)B * D, logits_err);
        compare("actions", act[1], act[0], (size_t)B * g_heads, sample_err);
        compare("env_actions", env_act[1], env_act[0], (size_t)B * g_heads, sample_err);
        compare("logprobs", lp[1], lp[0], B, sample_err);
        compare("values", val[1], val[0], B, sample_err);
        if (memcmp(rng[0], rng[1], (size_t)B * sizeof(*rng[0]))) {
            fprintf(stderr, "Philox state mismatch at step %d\n", t); exit(1);
        }
        if (step) {
            // Mask rows are cast whole (stride D+3); the heads use D - 1 columns.
            for (int b = 0; b < B; b++) {
                compare("step mask", step_mask + b * (D + 3), mask + b * (D + 3), D - 1, sample_err);
                for (int j = D - 1; j < D + 3; j++) if ((float)step_mask[b * (D + 3) + j] != 2.0f) {
                    fprintf(stderr, "step mask tail [%d,%d]\n", b, j); exit(1);
                }
            }
            compare("step obs slots and guards", step_storage, step_expected.data(), step_n, state_err);
            std::vector<P> rew_ref(rew_src, rew_src + B), term_ref(terminals + 3, terminals + 3 + B);
            compare("step rewards", rew_dst, rew_ref.data(), B, sample_err);
            compare("step terminals", term_dst, term_ref.data(), B, sample_err);
            compare("step snapshot", initial[2], initial[0], initial_n, state_err);
            compare("step state", st[2], st[0], (size_t)L * B * H, state_err);
            compare("step logits", out[2], out[0], (size_t)B * D, logits_err);
            compare("step actions", act[2], act[0], (size_t)B * g_heads, sample_err);
            compare("step env_actions", env_act[2], env_act[0], (size_t)B * g_heads, sample_err);
            compare("step logprobs", lp[2], lp[0], B, sample_err);
            compare("step values", val[2], val[0], B, sample_err);
            if (memcmp(rng[0], rng[2], (size_t)B * sizeof(*rng[0]))) {
                fprintf(stderr, "step Philox state mismatch at step %d\n", t); exit(1);
            }
        }
    }
    printf("B=%d K=%d H=%d L=%d D=%d continuous=%d paths=%d%s steps=%d: state_abs=%.3g logits_abs=%.3g sample_abs=%.3g RNG exact PASS\n",
        B, K, H, L, D, continuous, paths, packed ? (g_step_split == 1 ? " (packed split 1)"
            : g_step_split == 2 ? " (packed split 2)" : " (packed split 4)") : "",
        steps, state_err, logits_err, sample_err);
    cudaStreamDestroy(stream);
    for (void* p : allocations) puf_mtl::free(p);
}

// A launch whose args disagree with the kernel's compile-time hidden size (or
// layer count) must be a no-op. Buffers are sized for H=128 so a missing guard
// shows up as modified sentinels rather than out-of-bounds writes.
static void run_mismatch_guard() {
    const int B = 16, K = 8, HM = 128, LM = 5, D = 4;
    std::vector<void*> allocations;
    auto precs = [&](size_t n, float v) {
        P* p = alloc<P>(n); allocations.push_back(p);
        for (size_t i = 0; i < n; i++) p[i] = v;
        return p;
    };
    P* obs = precs((size_t)B * K, 0.25f);
    P* enc = precs((size_t)HM * K, 0.01f);
    P* dec = precs((size_t)3 * HM * HM, 0.01f);
    P* rec = precs((size_t)3 * HM * HM, 0.01f);
    P* state = precs((size_t)LM * B * HM, 7.0f);
    P* out = precs((size_t)B * D, 9.0f);
    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    struct Case { const char* kernel; int groups, hidden, layers; } cases[] = {
        {"rollout_forward_h64", 2, 32, 4}, {"rollout_forward_h32", 4, 64, 4},
        {"rollout_forward_h128", 1, 64, 4}, {"rollout_forward_h32", 4, 32, 5},
        {"rollout_forward_h32", 4, 32, 0},
    };
    for (const Case& c : cases) {
        puf_mtl::FusedRolloutArgs a = {};
        a.obs = puf_mtl::gpu_addr(obs); a.encoder = puf_mtl::gpu_addr(enc);
        for (int l = 0; l < 4; l++) a.recurrent[l] = puf_mtl::gpu_addr(rec);
        a.decoder = puf_mtl::gpu_addr(dec); a.state = puf_mtl::gpu_addr(state);
        a.out = puf_mtl::gpu_addr(out);
        a.B = B; a.input_dim = K; a.hidden = c.hidden; a.layers = c.layers; a.output_dim = D;
        puf_mtl::dispatch(stream, puf_mtl::pipeline(c.kernel),
            dim3((B + 8 * c.groups - 1) / (8 * c.groups)), dim3(32 * c.groups), a);
    }
    // rollout_step_h32 with args its fixed shape or the sampler can't take.
    // Its obs/state/out/mask/reward destinations are all sentinel buffers.
    for (int c = 0; c < 6; c++) {
        puf_mtl::RolloutStepArgs a = {};
        a.fwd.obs = puf_mtl::gpu_addr(obs); a.fwd.encoder = puf_mtl::gpu_addr(enc);
        for (int l = 0; l < 4; l++) a.fwd.recurrent[l] = puf_mtl::gpu_addr(rec);
        a.fwd.decoder = puf_mtl::gpu_addr(dec); a.fwd.state = puf_mtl::gpu_addr(state);
        a.fwd.out = puf_mtl::gpu_addr(out);
        a.fwd.B = B; a.fwd.input_dim = K; a.fwd.hidden = c == 0 ? 64 : 32;
        a.fwd.layers = 4; a.fwd.output_dim = D;
        a.smp = {a.fwd.out, 0, a.fwd.state, a.fwd.state, a.fwd.state, a.fwd.state,
            a.fwd.state, a.fwd.state, a.fwd.state, B, D, 1, c == 3 ? 300 : 4, c == 1};
        a.obs_src = a.fwd.state; a.rew_dst = a.fwd.out; a.rew_src = a.fwd.state;
        a.term_dst = a.fwd.out; a.term_src = a.fwd.state;
        a.mask_dst = a.fwd.out; a.mask_src = a.fwd.state;
        a.mask_size = c == 2 ? 3 : c == 3 ? 300 : 4;
        a.obs_row_stride = c == 4 ? -1 : c == 5 ? K - 1 : 0;
        if (puf_mtl::rollout_step_supported(a)) abort();
        // Invalid strides are host-only rejection checks: never dispatch a
        // negative stride over a buffer whose base has no preceding guard.
        if (c >= 4) continue;
        puf_mtl::dispatch(stream, puf_mtl::pipeline("rollout_step_h32"), dim3((B + 31) / 32), dim3(128), a);
    }
    cudaStreamSynchronize(stream);
    for (size_t i = 0; i < (size_t)B * K; i++) if ((float)obs[i] != 0.25f) {
        fprintf(stderr, "mismatched launch wrote obs[%zu]\n", i); exit(1);
    }
    for (size_t i = 0; i < (size_t)LM * B * HM; i++) if ((float)state[i] != 7.0f) {
        fprintf(stderr, "mismatched launch wrote state[%zu]\n", i); exit(1);
    }
    for (size_t i = 0; i < (size_t)B * D; i++) if ((float)out[i] != 9.0f) {
        fprintf(stderr, "mismatched launch wrote out[%zu]\n", i); exit(1);
    }
    puts("mismatched kernel/args launches are no-ops PASS");
    cudaStreamDestroy(stream);
    for (void* p : allocations) puf_mtl::free(p);
}

int main() {
    run_mismatch_guard();
    run(4096, 118, 32, 4, {3}, false); // actual Breakout dimensions
    run(65, 127, 32, 4, {31}, false); // H32 ragged agents/input, maximum decoder width
    run(128, 118, 128, 4, {31}, false); // tensor_ops encoder, recurrence and decoder
    run(64, 128, 64, 4, {7}, false);    // tensor_ops encoder and recurrence
    run(1, 1, 32, 1, {2, 5}, false);
    run(7, 31, 64, 3, {3, 4, 5}, false);
    run(1024, 8, 64, 3, {3}, false);   // Pong-like H=64 rollout step
    run(513, 7, 64, 2, {2}, false);    // Flappy-like, ragged agent groups
    run(63, 127, 128, 4, {31}, false);
    run(64, 256, 32, 2, {7, 9}, false);
    run(65, 118, 32, 4, {1, 1, 1}, true);
    run(257, 129, 128, 4, {1, 1, 1}, true);
    // Packed-weight steps (rollout_step_packed) against the same references.
    for (int split : {1, 2, 4}) {
        g_step_split = split;
        if (split < 4) {
            run(4096, 118, 32, 4, {3}, false);
            run(65, 127, 32, 4, {31}, false);
            run(1, 1, 32, 1, {2, 5}, false);
            run(64, 256, 32, 2, {7, 9}, false);
            run(1024, 4, 32, 2, {2}, false);  // Cartpole
        }
        run(64, 128, 64, 4, {7}, false);
        run(7, 31, 64, 3, {3, 4, 5}, false);
        run(1024, 8, 64, 3, {3}, false);
        run(513, 7, 64, 2, {2}, false);
        run(9, 64, 64, 1, {12, 9, 10}, false);  // decoder spans 4 tiles
    }
    g_step_split = 0;
    if (puf_mtl::fused_rollout_supported(1, 257, 32, 4, 4)
            || puf_mtl::fused_rollout_supported(1, 118, 16, 4, 4)
            || puf_mtl::fused_rollout_supported(1, 118, 32, 5, 4)
            || puf_mtl::fused_rollout_supported(1, 118, 32, 4, 33)) abort();
    puts("ALL PASS");
}
