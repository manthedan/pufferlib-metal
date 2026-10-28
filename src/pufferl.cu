// CUDA
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_profiler_api.h>
#include <cublas_v2.h>
#include <curand.h>
#include <curand_kernel.h>
#include <nccl.h>
#include <nvml.h>
#include <nvtx3/nvToolsExt.h>

// C standard
#include <cassert>
#include <cmath>
#include <cstdint>
#include <errno.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// POSIX / threading
#include <dirent.h>
#include <fcntl.h>
#include <omp.h>
#include <pthread.h>
#include <signal.h>
#include <spawn.h>
#include <sys/file.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#ifdef PUF_METAL
#include <algorithm>
#include <sys/resource.h>
#include <sys/sysctl.h>
#endif
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

// Project
#include "ini.h"

// Kernel launch: PUF_LAUNCH(k, grid, block, smem, stream)(args) is k<<<...>>>(args).
// The Metal backend (src/metal/launch.h) predefines it to route launches.
#ifndef PUF_LAUNCH
#define PUF_LAUNCH(k, g, b, s, st) k<<<g, b, s, st>>>
#endif

#ifdef PRECISION_FLOAT
typedef float precision_t;
constexpr bool USE_BF16 = false;
constexpr cudaDataType_t CUBLAS_PRECISION = CUDA_R_32F;
constexpr cublasComputeType_t CUBLAS_COMPUTE = CUBLAS_COMPUTE_32F;
#define NCCL_PRECISION ncclFloat
#define to_float(x) (x)
#define from_float(x) (x)
#else
typedef __nv_bfloat16 precision_t;
constexpr bool USE_BF16 = true;
constexpr cudaDataType_t CUBLAS_PRECISION = CUDA_R_16BF;
constexpr cublasComputeType_t CUBLAS_COMPUTE = CUBLAS_COMPUTE_32F;
#define NCCL_PRECISION ncclBfloat16
#define to_float(x) __bfloat162float(x)
#define from_float(x) __float2bfloat16(x)
#endif

#define PUF_MAX_DIMS 8
#define BLOCK_SIZE 256
int grid_size(int N) {
    return (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
}

// Compile vs a single env: -DENV_HEADER=ocean/<env>/<env>.h or .cu (--cu)
#include ENV_HEADER

typedef struct {
    float* data;
    int64_t shape[PUF_MAX_DIMS];
} Float;

typedef struct {
    unsigned char* data;
    int64_t shape[PUF_MAX_DIMS];
} Byte;

typedef struct {
    long* data;
    int64_t shape[PUF_MAX_DIMS];
} Long;

typedef struct {
    int* data;
    int64_t shape[PUF_MAX_DIMS];
} Int;

typedef struct {
    precision_t* data;
    int64_t shape[PUF_MAX_DIMS];
} Prec;

__host__ __device__ int ndim(int64_t* shape) {
    int n = 0;
    while (n < PUF_MAX_DIMS && shape[n] != 0) {
        n++;
    }
    return n;
}

__host__ __device__ int64_t numel(int64_t* shape) {
    int64_t n = 1;
    for (int i = 0; i < PUF_MAX_DIMS && shape[i] != 0; i++) {
        n *= shape[i];
    }
    return n;
}

int64_t batch_size(int64_t* shape) {
    int n = ndim(shape);
    int64_t b = 1;
    for (int i = 0; i < n - 2; i++) {
        b *= shape[i];
    }
    return b;
}

void squeeze_shape(int64_t* shape, int dim) {
    int n = ndim(shape);
    shape[dim] *= shape[dim + 1];
    for (int i = dim + 1; i < n - 1; i++) {
        shape[i] = shape[i + 1];
    }
    shape[n - 1] = 0;
}

Prec* puf_squeeze(Prec* t, int dim) {
    squeeze_shape(t->shape, dim);
    return t;
}

Float* puf_squeeze(Float* t, int dim) {
    squeeze_shape(t->shape, dim);
    return t;
}

Prec* puf_unsqueeze(Prec* t, int dim, int64_t d0, int64_t d1) {
    int n = ndim(t->shape);
    assert(n + 1 <= PUF_MAX_DIMS);
    assert(t->shape[dim] == d0 * d1);
    for (int i = n; i > dim; i--) {
        t->shape[i] = t->shape[i - 1];
    }
    t->shape[dim] = d0;
    t->shape[dim + 1] = d1;
    return t;
}

void puf_copy(Prec* dst, Prec* src, cudaStream_t stream) {
    assert(numel(dst->shape) == numel(src->shape) && "puf_copy: size mismatch");
    cudaMemcpyAsync(dst->data, src->data,
        numel(dst->shape) * sizeof(precision_t),
        cudaMemcpyDeviceToDevice, stream);
}

#ifndef PUF_METAL  // device code: Metal ports live in src/metal/shaders
__global__ void cast(precision_t* dst,
        float* src, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = from_float(src[idx]);
    }
}

#ifdef PRECISION_FLOAT
// GPU envs fix obs as bf16; float train needs an explicit widen.
__global__ void cast(precision_t* dst,
        __nv_bfloat16* src, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = __bfloat162float(src[idx]);
    }
}
#else
// Identity overload so obs→rollout cast arm typechecks when obs_t is bf16.
__global__ void cast(precision_t* dst,
        precision_t* src, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = src[idx];
    }
}

__global__ void cast(float* dst,
        precision_t* src, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = to_float(src[idx]);
    }
}
#endif

__global__ void cast(precision_t* dst,
        unsigned char* src, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = from_float((float)src[idx]);
    }
}

__global__ void cast(unsigned char* dst,
        precision_t* src, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = to_float(src[idx]);
    }
}

// Fused rew+term cast
__global__ void cast_rew_term(
        precision_t* __restrict__ rew_dst, const float* __restrict__ rew_src,
        precision_t* __restrict__ term_dst, const float* __restrict__ term_src,
        int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        rew_dst[idx] = from_float(rew_src[idx]);
        term_dst[idx] = from_float(term_src[idx]);
    }
}
#endif  // PUF_METAL

// Allocates large buffers (params/acts/grads) in contiguous memory.
struct AllocEntry {
    void** data_ptr;
    int64_t* shape;
    int elem_size;
};

struct Allocator {
    AllocEntry* regs;
    int num_regs;
    void* mem;
    long total_elems;
    long total_bytes;
};

void _alloc_register(Allocator* alloc,
        void** data_ptr, int64_t* shape, int elem_size) {
    alloc->regs = (AllocEntry*)realloc(alloc->regs,
        (alloc->num_regs + 1) * sizeof(AllocEntry));
    alloc->regs[alloc->num_regs++] = {data_ptr, shape, elem_size};
    long n = numel(shape);
    alloc->total_elems += n;
    alloc->total_bytes = (alloc->total_bytes + 15) & ~15;
    alloc->total_bytes += n * elem_size;
}
void alloc_register(Allocator* a, Prec* t) {
    _alloc_register(a, (void**)&t->data, t->shape, sizeof(precision_t));
}
void alloc_register(Allocator* a, Float* t) {
    _alloc_register(a, (void**)&t->data, t->shape, sizeof(float));
}
void alloc_register(Allocator* a, Long* t) {
    _alloc_register(a, (void**)&t->data, t->shape, sizeof(long));
}
void alloc_register(Allocator* a, Int* t) {
    _alloc_register(a, (void**)&t->data, t->shape, sizeof(int));
}

void alloc_create(Allocator* alloc) {
    assert(cudaMalloc(&alloc->mem, alloc->total_bytes) == cudaSuccess
        && "alloc_create: cudaMalloc failed");
    cudaMemset(alloc->mem, 0, alloc->total_bytes);
    long offset = 0;
    for (int i = 0; i < alloc->num_regs; i++) {
        offset = (offset + 15) & ~15;
        *alloc->regs[i].data_ptr = (char*)alloc->mem + offset;
        offset += numel(alloc->regs[i].shape) * alloc->regs[i].elem_size;
    }
}

// Deterministic block tree-reduce. smem layout: smem[c * nthreads + tid].
// Caller fills smem; tid 0 writes nchan results to out[0..nchan).
#ifndef PUF_METAL  // device code: Metal ports live in src/metal/shaders
#ifndef PUF_BLOCK_REDUCE_SUM
#define PUF_BLOCK_REDUCE_SUM
__device__ __forceinline__ void block_reduce_sum(
        float* smem, float* out, int tid, int nthreads, int nchan) {
    __syncthreads();
    for (int s = nthreads / 2; s > 0; s >>= 1) {
        if (tid < s) {
            for (int c = 0; c < nchan; c++) {
                smem[c * nthreads + tid] += smem[c * nthreads + tid + s];
            }
        }
        __syncthreads();
    }
    if (tid == 0) {
        for (int c = 0; c < nchan; c++) {
            out[c] = smem[c * nthreads];
        }
    }
}
#endif
#endif  // PUF_METAL

// Algo + sweeps only depend on basic tensor types and utilities
#include "algo.cu"
#include "protein.cu"

typedef struct {
    int horizon;
    int total_agents;
    int num_buffers;
    int hidden_size;
    int num_layers;
    float lr;
    float min_lr_ratio;
    bool anneal_lr;
    float momentum;
    int minibatch_size;
    float replay_ratio;
    long total_timesteps;
    float max_grad_norm;
    float clip_coef;
    float vf_clip_coef;
    float vf_coef;
    float ent_coef;
    float min_ent_coef_ratio;
    bool anneal_ent_coef;
    float gamma;
    float gae_lambda;
    bool vtrace;
    float vtrace_rho_clip;
    float vtrace_c_clip;
    bool async;
    bool reset_every_horizon;
    bool cudagraphs;
    bool profile;
    int rank;
    int world_size;
    int gpu_id;
    int num_threads;
    int seed;
} Hypers;

// Rank / device context for one process in a multi-GPU train job.
typedef struct {
    int rank;
    int world_size;
    int gpu_id;
    int artifact_owner;
    ncclUniqueId* nccl_id;
} TrainContext;

typedef struct ObsTensor {
    obs_t* data;
    int64_t shape[8];
} ObsTensor;

// Each rollout buffer manages a constant subset of environments
// Simulation + inference overlap across buffers
// Note: experimental async path adds an extra slot dimension.
struct RolloutBuf {
    Prec observations;  // (horizon, agents, input_size)
    Prec initial_states;
    Float actions;      // (horizon, agents, num_atns) float32: large discrete IDs
    Prec values;        // (horizon, agents)
    Prec logprobs;      // ...
    Prec rewards;
    Prec terminals;
    Prec action_mask;   // (horizon, agents, mask_size)
};

// Buffers are initialized as raw structs with only shape information.
// alloc_register stores the shape and data pointer.
// Memory is only allocated after all buffers are registered.
void register_rollout_buffers(RolloutBuf* bufs, Allocator* alloc,
        int T, int B, int input_size, int num_atns, int mask_size) {
    memset(bufs, 0, sizeof(*bufs));
    bufs->observations = {.shape = {T, B, input_size}};
    bufs->actions      = {.shape = {T, B, num_atns}};
    bufs->values       = {.shape = {T, B}};
    bufs->logprobs     = {.shape = {T, B}};
    bufs->rewards      = {.shape = {T, B}};
    bufs->terminals    = {.shape = {T, B}};
    bufs->action_mask  = {.shape = {T, B, mask_size}};
    Prec* prec_fields[] = {
        &bufs->observations, &bufs->values, &bufs->logprobs,
        &bufs->rewards, &bufs->terminals, &bufs->action_mask,
    };
    for (int i = 0; i < (int)(sizeof(prec_fields) / sizeof(prec_fields[0])); i++) {
        alloc_register(alloc, prec_fields[i]);
    }
    alloc_register(alloc, &bufs->actions);
}

// Rank-2 or rank-3 time-major tensor. F==0 means rank-2 (zero-terminated shape);
// stride still multiplies by max(F, 1).
Prec puf_time_view(Prec p, int start_t, int T) {
    long B = p.shape[1];
    long F = p.shape[2];
    long stride_f = F > 1 ? F : 1;
    return {
        .data = p.data + (long)start_t * B * stride_f,
        .shape = {T, B, F},
    };
}

Float puf_time_view(Float p, int start_t, int T) {
    long B = p.shape[1];
    long F = p.shape[2];
    long stride_f = F > 1 ? F : 1;
    return {
        .data = p.data + (long)start_t * B * stride_f,
        .shape = {T, B, F},
    };
}

RolloutBuf rollout_time_view(RolloutBuf* base, int start_t, int T) {
    RolloutBuf view = *base;
    view.observations = puf_time_view(base->observations, start_t, T);
    view.actions      = puf_time_view(base->actions,      start_t, T);
    view.values       = puf_time_view(base->values,       start_t, T);
    view.logprobs     = puf_time_view(base->logprobs,     start_t, T);
    view.rewards      = puf_time_view(base->rewards,      start_t, T);
    view.terminals    = puf_time_view(base->terminals,    start_t, T);
    view.action_mask  = puf_time_view(base->action_mask,  start_t, T);
    return view;
}

// Env batch. Device IO is always PuffeRL.env (EnvBuf).
// CPU: host pins + workers. GPU: single-buffer device envs (no host pins).
struct VecEnv {
    Env* envs;  // CPU: host array. GPU: device base (batch).
    int size;
    int total_agents;
    int buffers;
    int agents_per_buf;
    int mask_size;
    int num_policies;
    int* policy_layout;
    // CPU pins / workers (unused on GPU)
    int* env_starts;
    int* env_counts;
    obs_t* observations;
    float* actions;
    float* rewards;
    float* terminals;
    unsigned char* action_mask;
    int* worker_state;
    int shutdown;
    pthread_t* threads;
    float* accum;
    int num_workers;
    // GPU log reduce scratch (unused on CPU)
    float* log_scratch;
};

struct EnvBuf {
    ObsTensor obs;    // (total_agents, obs_size)
    Float actions;    // (total_agents, num_atns)
    Float rewards;    // (total_agents,)
    Float terminals;  // (total_agents,)
    Byte action_mask; // (total_agents, mask_size); always allocated
};

// Owned runnable policy instance. policies[0] is trainable; policies[i>0] are
// frozen opponents (selfplay/match). Frozen policies are rollout-only (no train
// acts/grads/muon) and may use a different arch via hist_policy_hidden_size /
// hist_policy_num_layers. Primary registers rollout acts into PuffeRL.activ_alloc
// (shared with train); frozen policies use their own activ_alloc.
typedef struct {
    bool frozen;
    Arch arch;
    Weights weights;
    Allocator params_alloc;
    Allocator activ_alloc;
    Prec param;
    Float master_weights;
    Prec* buffer_states;    // [num_buffers]
    Activations* buf_acts;  // [num_buffers]
} Policy;

// PROF_* accum; MODEL_*/H2D_* worker+GPU rollout events; TE_* train_impl events.
enum {
    PROF_ROLLOUT,
    PROF_MODEL,
    PROF_ENV,
    PROF_COPY,
    PROF_TRAIN_MISC,
    PROF_TRAIN_MODEL,
    NUM_PROF,
    MODEL_START = 0,
    MODEL_END,
    COPY_END,
    ENV_END = COPY_END,
    H2D_START,
    H2D_END,
    NUM_EV,
    EV_T = ENV_END + 1,
    TE_S = 0,
    TE_MID,
    TE_E,
    NUM_TE,
};

// Index must match PROF_* order above (first NUM_PROF names).
const char* PROF_NAMES[] = {
    "rollout",
    "eval_model",
    "eval_env",
    "eval_copy",
    "train_misc",
    "train_model",
};

// Index must match LossIdx 0..LOSS_N-1 in algo.cu (not LOSS_N counter).
const char* LOSS_NAMES[] = {
    "loss/policy",
    "loss/value",
    "loss/entropy",
    "loss/total",
    "loss/old_kl",
    "loss/kl",
    "loss/clipfrac",
    "importance",
};

typedef struct {
    unsigned long long* stamps;  // device [2 * NUM_TE]; globaltimer ns
    cudaEvent_t* rollout_ev;  // GPU [EV_T * horizon]; null on CPU
    float accum[NUM_PROF];
    int skip_rollout_time;
} Profile;

typedef struct PuffeRL {
    Policy* policies;        // [num_policies]; policies[0] trainable, rest frozen
    int num_policies;
    Weights actor_weights; // async rollout snapshot of policies[0]; unused when async=0
    Activations train_activs;
    Allocator weight_alloc;      // async actor weights
    Allocator grads_alloc;
    Allocator activ_alloc;       // train + primary rollout (shared)
    VecEnv* vec;
    Muon muon;
    ncclComm_t nccl_comm;  // NCCL communicator for multi-GPU
    Hypers hypers;
    bool is_continuous;  // True if all action dimensions are continuous (size==1)
    RolloutBuf rollouts;
    RolloutBuf train_rollouts;  // Pre-allocated transposed copy for train_impl
#ifdef PUF_METAL
    // Each slot owns its observation layout until its learner has joined dW.
    // Set before releasing rollout workers; read only for the other ready slot.
    bool metal_obs_bt[2];
    // Rollout-step weights packed at rollout start (metal_rollout_pack);
    // metal_packed_split > 0 while they match the policy the steps read.
    float* metal_packed;
    size_t metal_packed_floats;
    int metal_packed_split;
    int metal_rollout_group;  // buffers per dispatch, fixed before workers wake
    int metal_rollout_gate;   // gated pipelined steps (metal_gated_horizon), like group
    bool metal_gate_streams;  // rollout streams are host-wait-only for gating
    // Four-buffer gated groups (rollout_start): 0 undecided, else 2 or 4;
    // decided once from the learner's rollout slack (metal_dw_policy).
    int metal_group_pref;
    int metal_slack_n;
    float metal_slack[8];
#endif
    EnvBuf env;
    TrainGraph train_buf;
    Prec train_state;  // (L, A, H) carry in env order; graph reads with dest_off
    cudaGraphExec_t* rollout_graphs;  // CPU: [slots][horizon][num_buffers]
    cudaGraphExec_t gpu_rollout_graph[2];  // GPU: full net+env horizon per slot
    cudaGraphExec_t train_cudagraph[2];  // per async slot; src pointers differ
    cudaStream_t* streams;  // per-buffer raw CUDA streams
    cudaStream_t default_stream;  // main-thread stream (captured once at init)
    cudaStream_t train_stream;    // dedicated learner stream (always non-default)
    int* act_sizes;        // device ACT_SIZES (NUM_ATNS ints)
    float* losses;         // device loss accumulator (NUM_LOSSES)
    PPOBufs ppo_bufs; // Pre-allocated buffers for ppo_loss_fwd_bwd
    Prec actor_param;      // async flat actor params
    Prec grad;
    long* rng_offset;      // device counters (num_buffers+1)
    Profile profile;
    nvmlDevice_t nvml_device;
    long epoch;
    long global_step;
    double start_time;
    double last_log_time;
    long last_log_step;
    int write_slot;
    int async_ready_slot;
    int async_write_slot;
    int async_num_slots;  // 2 when async (Cleanba), else 1
    bool async_boot;
    ulong seed;
    curandStatePhilox4_32_10_t** rng_states;  // per-buffer persistent RNG states [num_buffers]
    char env_name[64];  // For policy arch rebuild at create.
} PuffeRL;

#ifdef PUF_WARM_START_TEST
static void puf_warm_start_test_dump(const char* path, const void* data, size_t n) {
    if (!path || !*path) return;
    void* host = malloc(n);
    if (!host) exit(2);
    cudaMemcpy(host, data, n, cudaMemcpyDeviceToHost);
    FILE* fp = fopen(path, "wb");
    if (!fp || fwrite(host, 1, n, fp) != n) exit(2);
    fclose(fp);
    free(host);
}

static void puf_warm_start_test_dump_state(PuffeRL* p) {
    const char* weights = getenv("PUF_WARM_START_TEST_WEIGHTS");
    const char* momentum = getenv("PUF_WARM_START_TEST_MOMENTUM");
    puf_warm_start_test_dump(weights, p->policies[0].master_weights.data,
        numel(p->policies[0].master_weights.shape) * sizeof(float));
    puf_warm_start_test_dump(momentum, p->muon.mb.data,
        numel(p->muon.mb.shape) * sizeof(float));
#if defined(PUFFER_NETHACK) && defined(NH_MSG_FROZEN)
    auto* ew = (NethackEncoderWeights*)p->policies[0].weights.encoder;
    size_t bytes = (size_t)NH_MSG_VOCAB * NH_MSG_HID * sizeof(precision_t);
    puf_warm_start_test_dump(getenv("PUF_WARM_START_TEST_NH_TABLE"), ew->msg_w.data, bytes);
    puf_warm_start_test_dump(getenv("PUF_WARM_START_TEST_NH_CACHE"), ew->msg_w0_dev, bytes);
    if (p->hypers.async) {
        auto* actor = (NethackEncoderWeights*)p->actor_weights.encoder;
        puf_warm_start_test_dump(getenv("PUF_WARM_START_TEST_NH_ACTOR_CACHE"), actor->msg_w0_dev, bytes);
    }
#endif
}

static void puf_warm_start_test_dump_rollout(PuffeRL* p) {
    static bool dumped = false;
    const char* path = getenv("PUF_WARM_START_TEST_OUTPUT");
    if (dumped || !path || !*path) return;
    dumped = true;
    RolloutBuf r = rollout_time_view(&p->rollouts,
        p->write_slot * p->hypers.horizon, p->hypers.horizon);
    FILE* fp = fopen(path, "wb");
    if (!fp) exit(2);
    auto dump = [&](const void* data, size_t n) {
        void* host = malloc(n);
        if (!host) exit(2);
        cudaMemcpy(host, data, n, cudaMemcpyDeviceToHost);
        if (fwrite(host, 1, n, fp) != n) exit(2);
        free(host);
    };
    dump(r.actions.data, numel(r.actions.shape) * sizeof(float));
    dump(r.values.data, numel(r.values.shape) * sizeof(precision_t));
    dump(r.logprobs.data, numel(r.logprobs.shape) * sizeof(precision_t));
    fclose(fp);
}
#endif

// Infer path: sample + forward, then vec workers.
static void profile_begin(const char* tag, bool enable) {
    if (enable) {
        nvtxRangePushA(tag);
    }
}

static void profile_end(bool enable) {
    if (enable) {
        nvtxRangePop();
    }
}

#ifndef PUF_METAL  // device code: Metal ports live in src/metal/shaders
__global__ void rng_init(curandStatePhilox4_32_10_t* states, uint64_t seed, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        curand_init(seed, idx, 0, &states[idx]);
    }
}

// Action logits and value share one row: [logits..., value]. logstd empty ⇒ discrete.
// Discrete: always-cache logsumexp + inverse-CDF; mask always present (all-ones if env has none).
// Continuous: ignores mask.
__global__ void sample_logits(
        Prec dec_out,              // (B, logits_dim + 1)
        Prec logstd,               // (1, od) continuous only; .data null if discrete
        int* act_sizes,            // (NUM_ATNS,)
        float* actions,            // (B, num_atns) float32 rollout store
        float* env_actions,        // (B, num_atns) env dispatch
        precision_t* logprobs,     // (B,)
        precision_t* value_out,    // (B,)
        curandStatePhilox4_32_10_t* rng_states,
        precision_t* action_mask,  // (B, A_total); always allocated
        int mask_stride) {
    int B = dec_out.shape[0];
    int fused_cols = dec_out.shape[1];
    int num_atns = NUM_ATNS;
    precision_t* logits = dec_out.data;
    bool is_continuous = logstd.data != NULL && numel(logstd.shape) > 0;
    precision_t* logstd_data = logstd.data;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B) {
        return;
    }

    curandStatePhilox4_32_10_t state = rng_states[idx];
    int logits_base = idx * fused_cols;
    float total_log_prob = 0.0f;

    if (is_continuous) {
        for (int h = 0; h < num_atns; h++) {
            float mean = safe_continuous_mean(logits, logits_base + h);
            float log_std = safe_continuous_logstd(logstd_data, h);
            float std = expf(log_std);
            float action = finite_or_clamp(
                mean + std * curand_normal(&state), -1.0e6f, 1.0e6f);
            // Preserve reduced-precision continuous semantics for logprob.
            precision_t stored_p = from_float(action);
            float stored = to_float(stored_p);
            float lp, ent;
            ppo_continuous_head(mean, log_std, stored, &lp, &ent);
            total_log_prob += lp;
            int aidx = idx * num_atns + h;
            actions[aidx] = stored;
            env_actions[aidx] = stored;
        }
    } else {
        int logits_offset = 0;
        int mask_base = idx * mask_stride;
        for (int h = 0; h < num_atns; h++) {
            int A = act_sizes[h];
            float cache[PPO_MAX_HEAD_A];
            float logsumexp = ppo_discrete_logsumexp(
                logits, logits_base, logits_offset, A, action_mask, mask_base, cache);
#ifdef PUFFER_NETHACK
            float inv_K = 0.0f;
            float eps = h == 0 ? nethack_verb_eps_load(
                action_mask + mask_base + logits_offset, A, &inv_K) : 0.0f;
#endif
            float rand_val = curand_uniform(&state);
            float cumsum = 0.0f;
            int sampled = A - 1;
            for (int a = 0; a < A; a++) {
#ifdef PUFFER_NETHACK
                if (eps > 0.0f) {
                    cumsum += nethack_verb_eps_mix(expf(cache[a] - logsumexp),
                        action_mask[mask_base + logits_offset + a], eps, inv_K);
                } else
#endif
                {
                    cumsum += expf(cache[a] - logsumexp);
                }
                if (rand_val < cumsum) {
                    sampled = a;
                    break;
                }
            }
            // CDF fall-through (float rounding) lands on A - 1, which may be
            // masked; snap to the last legal action. A legitimate A - 1 pick
            // is always legal, so the snap is an exact no-op for it.
            if (sampled == A - 1) {
                for (int a = A - 1; a >= 0; a--) {
                    if (to_float(action_mask[mask_base + logits_offset + a]) != 0.0f) {
                        sampled = a;
                        break;
                    }
                }
            }
            // Float32 preserves large categorical IDs that BF16 cannot represent.
            int aidx = idx * num_atns + h;
            float action = (float)sampled;
            actions[aidx] = action;
            env_actions[aidx] = action;
#ifdef PUFFER_NETHACK
            int verb = (int)actions[idx * num_atns];
            int used = nethack_head_used(verb, h);
            if (used) {
                if (eps > 0.0f) {
                    total_log_prob += logf((1.0f - eps)
                        * expf(cache[sampled] - logsumexp) + eps * inv_K);
                } else {
                    total_log_prob += cache[sampled] - logsumexp;
                }
            }
#else
            total_log_prob += cache[sampled] - logsumexp;
#endif
            logits_offset += A;
        }
    }

    logprobs[idx] = from_float(total_log_prob);
    value_out[idx] = logits[logits_base + fused_cols - 1];
    rng_states[idx] = state;
}
#endif  // PUF_METAL

#ifndef PUF_METAL  // device code: Metal ports live in src/metal/shaders
// Index into (L, agents, H): element-parallel over L*count*H.
// state_row is agent index within the state tensor's agent dim.
static __device__ long state_elem_idx(
        int layer, int agents_stride, int state_row, int h, int H) {
    return ((long)layer * agents_stride + state_row) * H + h;
}

// Copy buffer RNN state into rollout initial_states at t=0 (carry path).
// Trainable policy only; src agents are 0..count, dst at dst_start.
__global__ void snapshot_state(Prec dst, Prec src,
        int dst_start, int count) {
    int L = src.shape[0];
    int H = src.shape[2];
    int total = L * count * H;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    int h = idx % H;
    int rel = (idx / H) % count;
    int layer = idx / (count * H);
    long src_i = state_elem_idx(layer, (int)src.shape[1], rel, h, H);
    long dst_i = state_elem_idx(layer, (int)dst.shape[1], dst_start + rel, h, H);
    dst.data[dst_i] = src.data[src_i];
}
#endif  // PUF_METAL

// View slot of full (slots, L, A, H) as (L, A, H).
static Prec init_slot(Prec full, int slot) {
    int L = (int)full.shape[1];
    int A = (int)full.shape[2];
    int H = (int)full.shape[3];
    long stride = (long)L * A * H;
    return {.data = full.data + (long)slot * stride, .shape = {L, A, H}};
}

#ifndef PUF_METAL  // device code: Metal ports live in src/metal/shaders
// Zero RNN state for agents that just terminated. Same grid as snapshot
// (L*count*H); non-terminal threads return after one terminal load.
__global__ void zero_term_state(Prec state, Float terminals,
        int state_start, int terminal_start, int count) {
    int L = state.shape[0];
    int H = state.shape[2];
    int total = L * count * H;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    int h = idx % H;
    int rel = (idx / H) % count;
    int layer = idx / (count * H);
    if (terminals.data[terminal_start + rel] == 0.0f) {
        return;
    }
    long i = state_elem_idx(layer, (int)state.shape[1], state_start + rel, h, H);
    state.data[i] = from_float(0.0f);
}
#endif  // PUF_METAL

// Select time t, then agents [start, start+count). Rank-2 has F==0 (zero-term shape);
// stride uses max(F, 1). Out shape {count, F} keeps ndim 1 when F==0.
Prec puf_slice(Prec p, int t, int start, int count) {
    long B = p.shape[1];
    long F = p.shape[2];
    long stride_f = F > 1 ? F : 1;
    return {
        .data = p.data + (long)(t * B + start) * stride_f,
        .shape = {count, F},
    };
}

Float puf_slice(Float p, int t, int start, int count) {
    long B = p.shape[1];
    long F = p.shape[2];
    long stride_f = F > 1 ? F : 1;
    return {
        .data = p.data + (long)(t * B + start) * stride_f,
        .shape = {count, F},
    };
}

#ifndef PUF_METAL
#define PUF_TRACE(tag, a, b) ((void)0)
#endif
#ifdef PUF_METAL
// Pinned env IO already uses Metal shared storage. The worker finishes all
// CPU writes before encoding forward, then synchronizes its rollout stream
// before reading actions or overwriting inputs. Each buffer owns its slice.
#define PUF_TRACE(tag, a, b) puf_mtl::trace_mark(tag, a, b)
static bool metal_shared_env() {
    static const bool enabled = [] {
        const char* e = getenv("PUF_METAL_SHARED_ENV");
        return !e || strcmp(e, "0") != 0;
    }();
    return enabled;
}
#endif

#ifdef PUF_METAL
// PUF_METAL_ROLLOUT_GATE=0 turns pipelined gated rollout steps off
// (metal_gated_horizon).
static bool metal_gate_env() {
    static const bool enabled = [] {
        const char* e = getenv("PUF_METAL_ROLLOUT_GATE");
        return !e || strcmp(e, "0") != 0;
    }();
    return enabled;
}

// PUF_ENV_THREADS=N: run the CPU envs on N threads in total instead of the
// config's vec.num_threads (split over buffers into OpenMP teams). Envs
// step independently with per-env RNG, so the count never changes results.
// Measured (leg 19): clamping Pong/Cartpole's 16 threads on the 10-core M4
// Air to 4-10 cuts env time but leaves SPS flat, so the default keeps the
// config's count.
static int metal_env_threads(int requested) {
    static const int forced = [] {
        const char* e = getenv("PUF_ENV_THREADS");
        return e ? atoi(e) : 0;
    }();
    return forced > 0 ? forced : requested;
}

// OpenMP team of a grouped rollout worker: the buffers' threads combined,
// capped at all cores but two (the main thread, the other workers) unless
// PUF_ENV_THREADS sets the count: Cartpole's 16 threads in four-buffer
// groups would otherwise run one 16-thread team on the 10-core Air
// (four-buffer gated groups: -2.4% at 16, +0.5% at 8 or 4 vs pairs).
// Thread counts never change results (per-env RNG).
static int metal_env_team(int workers, int group) {
    static const int cap = [] {
        if (getenv("PUF_ENV_THREADS")) return 1 << 20;
        int n = 0;
        size_t len = sizeof(n);
        if (sysctlbyname("hw.ncpu", &n, &len, NULL, 0) != 0) n = 0;
        return n > 4 ? n - 2 : 1 << 20;
    }();
    int team = workers * group;
    return group > 1 && team > cap ? cap : team;
}

// PUF_METAL_ROLLOUT_SPIN_US: host waits on rollout streams poll for up to
// this long before blocking (puf_mtl::stream_wait_spin). 0 = block at once.
static uint32_t metal_rollout_spin_us() {
    static const uint32_t v = [] {
        const char* e = getenv("PUF_METAL_ROLLOUT_SPIN_US");
        return e ? (uint32_t)atoi(e) : 0u;
    }();
    return v;
}

// PUF_METAL_ROLLOUT_PACKED: -1 unset (packed steps, default split), 0 off,
// else the forced split (metal_rollout_pack).
static int metal_rollout_packed_env() {
    static const int v = [] {
        const char* e = getenv("PUF_METAL_ROLLOUT_PACKED");
        return !e ? -1 : atoi(e);
    }();
    return v;
}

// The whole step (obs/reward/terminal/mask casts, H=32 fused forward with
// terminal reset and snapshot, discrete sampling) as one dispatch
// (shaders/rollout.metal rollout_step_h32). Only for one trainable-or-frozen
// policy spanning the buffer block, float or u8 observations and discrete
// actions;
// returns false (nothing enqueued) otherwise. PUF_METAL_ROLLOUT_STEP=0
// disables it. Same per-element arithmetic, so results are bit-identical.
// Observation element type: 0 precision_t (copied bitwise: float in FP32,
// bf16 in BF16), 1 u8, 2 float in a BF16 build (cast as the generic cast
// kernel does), -1 unsupported.
template <typename O>
static int metal_rollout_step_obs(O*) { return -1; }
static int metal_rollout_step_obs(precision_t*) { return 0; }
static int metal_rollout_step_obs(unsigned char*) { return 1; }
#ifndef PRECISION_FLOAT
static int metal_rollout_step_obs(float*) { return 2; }
#endif

static bool metal_rollout_step(PuffeRL* pufferl, int buf, int t, RolloutBuf& rollouts,
        EnvBuf* env, int start, int block_size, int graph_slot, cudaStream_t stream,
        bool obs_bt = false, bool args_only = false, puf_mtl::RolloutStepArgs* out = nullptr) {
    static const bool enabled = [] {
        const char* e = getenv("PUF_METAL_ROLLOUT_STEP");
        return !e || strcmp(e, "0") != 0;
    }();
    static const bool fused_reset = [] {
        const char* e = getenv("PUF_METAL_FUSED_RESET");
        return !e || strcmp(e, "0") != 0;
    }();
    int* layout = pufferl->vec->policy_layout;
    if (!enabled || !fused_reset || pufferl->num_policies != 1
            || layout[0] != 0 || layout[1] != block_size
            || metal_rollout_step_obs(env->obs.data) < 0) {
        return false;
    }
    Policy* pol = &pufferl->policies[0];
    Hypers* hypers = &pufferl->hypers;
    Weights* w = (!pol->frozen && hypers->async) ? &pufferl->actor_weights : &pol->weights;
    DecoderWeights* dw = (DecoderWeights*)w->decoder;
    if (dw->continuous) return false;
    Activations* acts = &pol->buf_acts[buf];
    Prec* st = &pol->buffer_states[buf];
    Prec obs_b = puf_slice(rollouts.observations, t, start, block_size);
    if (obs_b.shape[1] != env->obs.shape[1]) return false;
    if (obs_bt) {
        // The slot allocation is unchanged, but its rows are (agent,time,obs).
        obs_b.data = rollouts.observations.data
            + ((long)start * hypers->horizon + t) * obs_b.shape[1];
        if ((long)hypers->horizon * obs_b.shape[1] > 0x7fffffffL) return false;
    }
    Prec initial = {};
    if (!pol->frozen && t == 0 && rollouts.initial_states.data != NULL) {
        initial = init_slot(rollouts.initial_states, graph_slot);
    }
    puf_mtl::RolloutStepArgs a = {};
    Prec dec = arch_try_fused_rollout(&pol->arch, *w, *acts, obs_b, *st, stream,
        env->terminals.data + start,
        initial.data ? initial.data + (long)start * st->shape[2] : nullptr,
        initial.data ? (int)initial.shape[1] : 0, &a.fwd);
    if (!dec.data) return false;
    // H=64 (rollout_step_h64, 8 agents per simdgroup) trades ~12 dispatch
    // boundaries per step for more GPU time under learner contention: a win
    // when the per-buffer rollout threads are latency-bound (Flappy, 4
    // buffers: +6%) and for a single FP32 buffer, whose learner dominates
    // (Pong FP32: +2%), a loss for a single BF16 buffer (Pong BF16: -8%).
    // The packed step (rollout_step_packed) halves that GPU time, so with it
    // the single BF16 buffer takes the step too (Pong BF16 rollout wait
    // 41 -> 23 ms/epoch, SPS neutral: its learner is the critical path).
    // PUF_METAL_ROLLOUT_STEP_H64=1 / 0 forces it on / off.
    static const int step_h64 = [] {
        const char* e = getenv("PUF_METAL_ROLLOUT_STEP_H64");
        return !e ? -1 : strcmp(e, "0") != 0;
    }();
    bool step_h64_default = hypers->num_buffers > 1 || !USE_BF16
        || metal_rollout_packed_env() != 0;
    if (a.fwd.hidden == 64 && !(step_h64 == 1 || (step_h64 < 0 && step_h64_default))) {
        return false;
    }

    int mask_size = rollouts.action_mask.shape[2];
    Float act_b = puf_slice(rollouts.actions, t, start, block_size);
    Prec lp_b = puf_slice(rollouts.logprobs, t, start, block_size);
    Prec val_b = puf_slice(rollouts.values, t, start, block_size);
    Prec mask_b = puf_slice(rollouts.action_mask, t, start, block_size);
    Prec rew_dst = puf_slice(rollouts.rewards, t, start, block_size);
    Prec term_dst = puf_slice(rollouts.terminals, t, start, block_size);
    long act_cols = env->actions.shape[1];
    a.smp = {puf_mtl::gpu_addr(dec.data), 0, puf_mtl::gpu_addr(pufferl->act_sizes),
        puf_mtl::gpu_addr(act_b.data), puf_mtl::gpu_addr(env->actions.data + (long)start * act_cols),
        puf_mtl::gpu_addr(lp_b.data), puf_mtl::gpu_addr(val_b.data),
        puf_mtl::gpu_addr(pufferl->rng_states[buf]), puf_mtl::gpu_addr(mask_b.data),
        (int32_t)dec.shape[0], (int32_t)dec.shape[1], NUM_ATNS, mask_size, 0};
    a.obs_src = puf_mtl::gpu_addr(env->obs.data + (long)start * env->obs.shape[1]);
    a.obs_is_u8 = metal_rollout_step_obs(env->obs.data);
    a.rew_dst = puf_mtl::gpu_addr(rew_dst.data);
    a.rew_src = puf_mtl::gpu_addr(env->rewards.data + start);
    a.term_dst = puf_mtl::gpu_addr(term_dst.data);
    a.term_src = puf_mtl::gpu_addr(env->terminals.data + start);
    a.mask_dst = puf_mtl::gpu_addr(mask_b.data);
    a.mask_src = puf_mtl::gpu_addr(env->action_mask.data + (long)start * mask_size);
    a.mask_size = mask_size;
    a.obs_row_stride = obs_bt ? hypers->horizon * (int)obs_b.shape[1] : 0;
    if (!puf_mtl::rollout_step_supported(a)) return false;
    if (out) *out = a;
    if (args_only) return true;
    if (pufferl->metal_packed_split) {
        puf_mtl::rollout_use_packed(a.fwd, puf_mtl::gpu_addr(pufferl->metal_packed));
        puf_mtl::rollout_step_packed(stream, a, pufferl->metal_packed_split);
    } else {
        puf_mtl::rollout_step(stream, a);
    }
    return true;
}

// Allocation-free eligibility shared by preflight and the actual pack path.
// The step still validates its bound GPU addresses immediately before dispatch.
static int metal_rollout_pack_split(PuffeRL* p) {
    const char* flags[] = {"PUF_METAL_ROLLOUT_STEP", "PUF_METAL_FUSED_RESET",
        "PUF_METAL_FUSED_ROLLOUT"};
    for (const char* key : flags) {
        const char* v = getenv(key);
        if (v && strcmp(v, "0") == 0) return 0;
    }
    int split = metal_rollout_packed_env();
    int B = p->hypers.total_agents / p->hypers.num_buffers;
    int H = p->hypers.hidden_size;
    if (!split || p->num_policies != 1 || p->is_continuous
            || p->vec->policy_layout[0] != 0 || p->vec->policy_layout[1] != B
            || metal_rollout_step_obs(p->env.obs.data) < 0) return 0;
    Arch* arch = &p->policies[0].arch;
    if (arch->encoder.forward != encoder_forward || arch->network.forward != mingru_forward
            || arch->decoder.forward != decoder_forward) return 0;
    int K = arch->encoder.in_dim, D = arch->decoder.output_dim + 1;
    if ((H != 32 && H != 64) || arch->encoder.out_dim != H
            || arch->decoder.hidden_dim != H
            || !puf_mtl::fused_rollout_supported(B, K, H, arch->network.num_layers, D)
            || p->env.obs.shape[1] != K || p->env.action_mask.shape[1] > 256
            || (long)p->hypers.horizon * K > 0x7fffffffL) return 0;
    const char* h64 = getenv("PUF_METAL_ROLLOUT_STEP_H64");
    if (H == 64 && h64 && strcmp(h64, "0") == 0) return 0;
    if (split < 0) split = (H == 32 ? 1 : 2) * (USE_BF16 ? 1 : 2);
    if ((H == 32 && split > 2) || (split != 1 && split != 2 && split != 4)) return 0;
    return split;
}

// Test hook (tests/metal/run_test_pack_order.sh): PUF_METAL_TEST_PACK_HOLD=1
// holds the default stream ~3 ms before each pack, so a rollout whose
// workers did not wait for the pack would read stale packed weights.
static void* metal_pack_release(void* h) {
    usleep(3000);
    puf_mtl::test_release(h);
    return NULL;
}

// Packs the weights the rollout steps of this rollout will read (rollout.h
// rollout_pack: float MMA tiles, exact) on the default stream; rollout_start
// syncs that stream (with the async actor snapshot) before any worker steps.
// Called by rollout_start after the snapshot copy, i.e. whenever those
// weights may have changed. PUF_METAL_ROLLOUT_PACKED=0 keeps rollout_step;
// 1/2/4 forces the split (simdgroups per 8-agent group, rollout_step_packed).
static void metal_rollout_pack(PuffeRL* p, int slot, RolloutBuf& view) {
    const int split = metal_rollout_pack_split(p);
    p->metal_packed_split = 0;
    puf_mtl::RolloutStepArgs a = {};
    int block = p->hypers.total_agents / p->hypers.num_buffers;
    if (split == 0 || !metal_rollout_step(p, 0, 0, view, &p->env, 0, block, slot,
            p->streams[0], p->metal_obs_bt[slot], true, &a)) {
        return;
    }
    const puf_mtl::FusedRolloutArgs& f = a.fwd;
    size_t n = puf_mtl::rollout_packed_floats(f.input_dim, f.hidden, f.layers, f.output_dim);
    if (n > p->metal_packed_floats) {
        cudaFree(p->metal_packed);
        cudaMalloc((void**)&p->metal_packed, n * sizeof(float));
        p->metal_packed_floats = n;
    }
    puf_mtl::RolloutPackArgs pa = {f.encoder, {}, f.decoder, puf_mtl::gpu_addr(p->metal_packed),
        f.input_dim, f.hidden, f.layers, f.output_dim};
    for (int l = 0; l < 4; l++) pa.recurrent[l] = f.recurrent[l];
    static const bool hold = getenv("PUF_METAL_TEST_PACK_HOLD")
        && strcmp(getenv("PUF_METAL_TEST_PACK_HOLD"), "0") != 0;
    if (hold) {
        pthread_t th;
        pthread_create(&th, NULL, metal_pack_release, puf_mtl::test_hold(p->default_stream));
        pthread_detach(th);
    }
    puf_mtl::rollout_pack(p->default_stream, pa);
    p->metal_packed_split = split;
}

// Decoder dW placement for the next epoch (algo.cu g_metal_dw_side). On the
// side stream, each minibatch's learner work splits into two command buffers
// around a cross-queue hand-off that leaves the learner idle ~180 us; that
// loses whenever the learner is on the critical path (clean A/B, side ->
// train stream: Cartpole +7.6% BF16 / +6.4% FP32, Pong +8.3% / +2.7%,
// Flappy BF16 +1.5%). A single rollout buffer whose step chain is the
// critical path instead runs uncontended in those gaps (Breakout: -2.2% /
// -1.7% on the train stream). So: the side stream only with one buffer
// and a learner that waited for the rollout more than 5% of the last epoch
// (Breakout ~10-15%, Pong ~1%). PUF_METAL_DW_SIDE=0 / 1 pins it.
// The backward dW overlap (algo.cu g_metal_bwd_overlap) uses the same
// signal: its concurrent dW GEMMs shorten the learner but contend with an
// overlapping rollout, so it only runs while the learner is the critical
// path (slack <= 5%; sync training always). Clean A/B with it always on:
// Pong +2.8% BF16, but Flappy FP32 -4.9% and Breakout BF16 -2.8% (both
// rollout bound, 15-30% slack). PUF_METAL_BWD_OVERLAP=0 / 1 pins it.
static void metal_dw_policy(PuffeRL* p, double slack, double epoch) {
    static const int pinned = [] {
        const char* e = getenv("PUF_METAL_DW_SIDE");
        return !e ? -1 : strcmp(e, "0") != 0;
    }();
    g_metal_dw_side = pinned >= 0 ? pinned == 1
        : p->hypers.num_buffers == 1 && slack > 0.05 * epoch;
    g_metal_bwd_overlap = slack <= 0.05 * epoch;
    // Four-buffer gated groups: one rollout queue whose bigger steps start
    // ~25 us after gate_open instead of 70-100 us on two queues of pairs,
    // but contend harder with the learner. Clean n=6 vs pairs: Flappy
    // +6.5% BF16 / +4.5% FP32 (learner waits 14% / 9% of an epoch for the
    // rollout with pairs), Cartpole +1.3% / -1.2% (0.4% / 2%). So: decided
    // once, from the median slack of async epochs 3-8 under pairs (> 5%,
    // as for the dW policies: four-buffer groups).
    if (p->metal_group_pref == 0 && p->hypers.async && epoch > 0) {
        if (p->epoch > 2) p->metal_slack[p->metal_slack_n++] = (float)(slack / epoch);
        if (p->metal_slack_n == 6) {
            float s[6];
            memcpy(s, p->metal_slack, sizeof(s));
            std::sort(s, s + 6);
            float med = 0.5f * (s[2] + s[3]);
            p->metal_group_pref = med > 0.05f ? 4 : 2;
            if (p->hypers.num_buffers % 4 == 0 && p->metal_rollout_gate) {
                fprintf(stderr, "metal: learner slack %.1f%% -> rollout groups of %d\n",
                    100.0f * med, p->metal_group_pref);
            }
        }
    }
}
#else
static int metal_env_team(int workers, int group) { return workers * group; }
#endif

static void pufferl_forward_step(PuffeRL* pufferl, int buf, int t,
        cudaStream_t stream) {
    Hypers* hypers = &pufferl->hypers;
    int graph_slot = hypers->async ? pufferl->write_slot : 0;
    RolloutBuf rollouts = pufferl->rollouts;
    if (hypers->async) {
        rollouts = rollout_time_view(&pufferl->rollouts,
            pufferl->write_slot * hypers->horizon, hypers->horizon);
    }
    EnvBuf* env = &pufferl->env;
    VecEnv* vec = pufferl->vec;
#ifdef PUF_METAL
    EnvBuf shared_env = *env;
    if (metal_shared_env()) {
        shared_env.obs.data = vec->observations;
        shared_env.actions.data = vec->actions;
        shared_env.rewards.data = vec->rewards;
        shared_env.terminals.data = vec->terminals;
        shared_env.action_mask.data = vec->action_mask;
        env = &shared_env;
    }
#endif
    int block_size = hypers->total_agents / hypers->num_buffers;
    int start = buf * block_size;
    int* layout = vec->policy_layout;

#ifdef PUF_METAL
    if (metal_rollout_step(pufferl, buf, t, rollouts, env, block_size * buf, block_size,
            graph_slot, stream, pufferl->metal_obs_bt[graph_slot])) {
        return;
    }
    if (pufferl->metal_obs_bt[graph_slot]) {
        puf_mtl::fatal("rollout observation layout: eligibility changed within a horizon");
    }
#endif
    // Copy observations, rewards, terminals from GPU env buffers to rollout buffer
    ObsTensor* obs_env = &env->obs;
    int n = block_size * obs_env->shape[1];
    Prec obs_dst = puf_slice(rollouts.observations, t, start, block_size);
    // Env obs -> rollout: D2D if same type, else cast (float/uchar → precision_t).
    if (sizeof(obs_t) == sizeof(precision_t)) {
        cudaMemcpyAsync(obs_dst.data,
            obs_env->data + (long)start * obs_env->shape[1],
            n * sizeof(precision_t), cudaMemcpyDeviceToDevice, stream);
    } else {
        PUF_LAUNCH(cast, grid_size(n), BLOCK_SIZE, 0, stream)(
            obs_dst.data, obs_env->data + (long)start * obs_env->shape[1], n);
    }

    Prec rew_dst = puf_slice(rollouts.rewards, t, start, block_size);
    Prec term_dst = puf_slice(rollouts.terminals, t, start, block_size);
    PUF_LAUNCH(cast_rew_term, grid_size(block_size), BLOCK_SIZE, 0, stream)(
        rew_dst.data, env->rewards.data + start,
        term_dst.data, env->terminals.data + start, block_size);

    // Mask always allocated (env-written or synthetic all-ones). Continuous ignores it in sample.
    int mask_size = rollouts.action_mask.shape[2];
    int mask_stride = mask_size;
    Prec mask_slice = puf_slice(rollouts.action_mask, t, start, block_size);
    PUF_LAUNCH(cast, grid_size(block_size * mask_size), BLOCK_SIZE, 0, stream)(
        mask_slice.data,
        env->action_mask.data + (long)start * mask_size,
        block_size * mask_size);

    // Per-policy forward: layout[b]..layout[b+1) within each buffer chunk.
    long act_cols = env->actions.shape[1];
    for (int b = 0; b < pufferl->num_policies; b++) {
        int off = layout[b];
        int n = layout[b + 1] - off;
        if (n == 0) {
            continue;
        }

        Policy* pol = &pufferl->policies[b];
        Weights* w = (!pol->frozen && hypers->async)
            ? &pufferl->actor_weights : &pol->weights;
        Activations* acts = &pol->buf_acts[buf];
        Prec* st = &pol->buffer_states[buf];

        int sub = start + off;
        Prec obs_b  = puf_slice(rollouts.observations, t, sub, n);
        Float act_b = puf_slice(rollouts.actions,      t, sub, n);
        Prec lp_b   = puf_slice(rollouts.logprobs,     t, sub, n);
        Prec val_b  = puf_slice(rollouts.values,       t, sub, n);
        Prec mask_b = puf_slice(rollouts.action_mask,  t, sub, n);

        Prec dec = {};
#ifdef PUF_METAL
        static const bool fused_reset = [] {
            const char* e = getenv("PUF_METAL_FUSED_RESET");
            return !e || strcmp(e, "0") != 0;
        }();
        if (fused_reset) {
            // Snapshot the post-reset state before the fused forward updates it.
            // The destination keeps the full rollout's layer stride even when
            // this policy owns only a compact subset of the agents.
            Prec initial = {};
            if (!pol->frozen && t == 0 && rollouts.initial_states.data != NULL) {
                initial = init_slot(rollouts.initial_states, graph_slot);
            }
            dec = arch_try_fused_rollout(&pol->arch, *w, *acts, obs_b, *st, stream,
                env->terminals.data + sub,
                initial.data ? initial.data + (long)sub * st->shape[2] : nullptr,
                initial.data ? (int)initial.shape[1] : 0);
        }
#endif
        if (!dec.data) {
            // Per-policy state is compact (n agents); local index 0..n-1.
            int state_n = (int)st->shape[0] * n * (int)st->shape[2];
            PUF_LAUNCH(zero_term_state, grid_size(state_n), BLOCK_SIZE, 0, stream)(
                *st, env->terminals, 0, sub, n);

            // Carry path: snapshot trainable policy state into per-slot initial_states.
            if (!pol->frozen && t == 0 && rollouts.initial_states.data != NULL) {
                Prec slot_st = init_slot(rollouts.initial_states, graph_slot);
                PUF_LAUNCH(snapshot_state, grid_size(state_n), BLOCK_SIZE, 0, stream)(
                    slot_st, *st, sub, n);
            }

            dec = arch_forward(&pol->arch, *w, *acts, obs_b, *st, stream);
        }

        Prec p_logstd = {};
        DecoderWeights* dw = (DecoderWeights*)w->decoder;
        if (dw->continuous) {
            p_logstd = dw->logstd;
        }

        // Offset RNG by off so policies don't collide on per-buffer rng slots.
        PUF_LAUNCH(sample_logits, grid_size(n), BLOCK_SIZE, 0, stream)(
            dec, p_logstd, pufferl->act_sizes,
            act_b.data, env->actions.data + (long)sub * act_cols,
            lp_b.data, val_b.data,
            pufferl->rng_states[buf] + off,
            mask_b.data, mask_stride);
    }
}

void pufferl_forward(PuffeRL* pufferl, int buf, int t, cudaStream_t stream) {
    Hypers* hypers = &pufferl->hypers;
    // GPU rollout graphs the whole horizon; CPU still graphs one net step.
    bool step_graph = hypers->cudagraphs && pufferl->rollout_graphs != NULL;
    int graph_slot = hypers->async ? pufferl->write_slot : 0;
    int graph = (graph_slot * hypers->horizon + t) * hypers->num_buffers + buf;
    profile_begin("fused_rollout", hypers->profile);

    if (step_graph && pufferl->rollout_graphs[graph] != NULL) {
        assert(cudaGraphLaunch(
            pufferl->rollout_graphs[graph], stream) == cudaSuccess
            && "cudaGraphLaunch failed");
        profile_end(hypers->profile);
        return;
    }

    if (step_graph) {
        assert(cudaStreamBeginCapture(
            stream, cudaStreamCaptureModeThreadLocal) == cudaSuccess
            && "cudaStreamBeginCapture failed");
    }
    pufferl_forward_step(pufferl, buf, t, stream);
    if (step_graph) {
        cudaGraph_t _graph;
        assert(cudaStreamEndCapture(stream, &_graph) == cudaSuccess
                && "cudaStreamEndCapture failed");
        assert(cudaGraphInstantiate(&pufferl->rollout_graphs[graph], _graph, 0)
                == cudaSuccess && "cudaGraphInstantiate failed");
        cudaGraphDestroy(_graph);
        assert(cudaGraphLaunch(pufferl->rollout_graphs[graph], stream) == cudaSuccess
                && "cudaGraphLaunch failed");
    }
    profile_end(hypers->profile);
}

double wall_clock() {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// Log is a flat float struct (host + device).
constexpr int LOG_NF = (int)(sizeof(Log) / sizeof(float));

// Shared host/device: fold Log into float acc; optionally clear the source.
__host__ __device__ static inline void log_accum(float* acc, Log* log, int clear) {
    if (log->n != 0.0f) {
        const float* el = (const float*)log;
        for (int j = 0; j < LOG_NF; j++) {
            acc[j] += el[j];
        }
    }
    if (clear) {
        float* el = (float*)log;
        for (int j = 0; j < LOG_NF; j++) {
            el[j] = 0.0f;
        }
    }
}

#ifndef PUF_METAL  // device code: Metal ports live in src/metal/shaders
__global__ void log_reduce(Env* envs,
        float* out, int total_agents, int clear) {
    extern __shared__ float sh[];
    int tid = threadIdx.x;
    float local[LOG_NF] = {};
    for (int i = tid; i < total_agents / envs->num_agents; i += blockDim.x) {
        log_accum(local, &envs[i].log, clear);
    }
    for (int j = 0; j < LOG_NF; j++) {
        sh[j * blockDim.x + tid] = local[j];
    }
    block_reduce_sum(sh, out, tid, blockDim.x, LOG_NF);
}
#endif  // PUF_METAL

static void env_log_sum(VecEnv* vec, Log* out, int clear) {
    if (PUF_BACKEND == PUF_GPU) {
        PUF_LAUNCH(log_reduce, 1, 256, LOG_NF * 256 * sizeof(float), 0)(
            vec->envs, vec->log_scratch, vec->total_agents, clear);
        cudaMemcpy(out, vec->log_scratch, sizeof(Log), cudaMemcpyDeviceToHost);
        return;
    }
    float* acc = (float*)out;
    for (int i = 0; i < vec->size; i++) {
        log_accum(acc, &vec->envs[i].log, clear);
    }
}

#if PUF_BACKEND != PUF_GPU
static void puf_bind_stream(cudaStream_t) {}
static Env* puf_vec_create(int, Dict*, obs_t*, float*, float*, float*) {
    return NULL;
}
#endif

static void env_setup(PuffeRL* p, VecEnv* vec, Dict* vk, Dict* ek) {
    if (PUF_BACKEND == PUF_GPU) {
        assert(vec->buffers == 1 && "GPU env: num_buffers must be 1");
        vec->envs = puf_vec_create(vec->total_agents, ek,
            p->env.obs.data, p->env.actions.data,
            p->env.rewards.data, p->env.terminals.data);
        cudaMalloc((void**)&vec->log_scratch, sizeof(Log));
        cudaMemset(vec->log_scratch, 0, sizeof(Log));
        vec->policy_layout[0] = 0;
        vec->policy_layout[1] = vec->agents_per_buf;
        return;
    }
    int total_agents = vec->total_agents;
    int num_buffers = vec->buffers;
    int apb = vec->agents_per_buf;
    int num_threads = dict_get(vk, "num_threads");
#ifdef PUF_METAL
    num_threads = metal_env_threads(num_threads);
#endif
    vec->num_workers = num_threads / num_buffers;
    if (vec->num_workers < 1) {
        vec->num_workers = 1;
    }
    vec->env_starts = (int*)calloc(1, num_buffers * sizeof(int));
    vec->env_counts = (int*)calloc(1, num_buffers * sizeof(int));
    int num_envs = 0;
#ifdef MY_VEC_INIT
    vec->envs = my_vec_init(&num_envs, vec->env_starts,
        vec->env_counts, vk, ek);
#else
    Env* envs = (Env*)calloc(total_agents, sizeof(Env));
    int agents_created = 0;
    while (agents_created < total_agents) {
        envs[num_envs].rng = num_envs;
        puf_init(&envs[num_envs], ek);
        agents_created += envs[num_envs].num_agents;
        num_envs++;
    }
    envs = (Env*)realloc(envs, num_envs * sizeof(Env));
    int buf = 0, buf_agents = 0;
    for (int i = 0; i < num_envs; i++) {
        buf_agents += envs[i].num_agents;
        vec->env_counts[buf]++;
        if (buf_agents >= apb && buf < num_buffers - 1) {
            buf++;
            vec->env_starts[buf] = i + 1;
            buf_agents = 0;
        }
    }
    vec->envs = envs;
#endif
    vec->size = num_envs;

    size_t obs_bytes = total_agents * OBS_SIZE * sizeof(obs_t);
    size_t mask_bytes = total_agents * vec->mask_size * sizeof(unsigned char);
    cudaHostAlloc((void**)&vec->observations, obs_bytes, cudaHostAllocPortable);
    cudaHostAlloc((void**)&vec->actions, total_agents * NUM_ATNS * sizeof(float),
        cudaHostAllocPortable);
    cudaHostAlloc((void**)&vec->rewards, total_agents * sizeof(float),
        cudaHostAllocPortable);
    cudaHostAlloc((void**)&vec->terminals, total_agents * sizeof(float),
        cudaHostAllocPortable);
    cudaHostAlloc((void**)&vec->action_mask, mask_bytes, cudaHostAllocPortable);
    memset(vec->action_mask, 1, mask_bytes);

    float frozen_pct = dict_get(vk, "hist_policy_percent");
    for (int buf = 0; buf < num_buffers; buf++) {
        int buf_start = buf * apb;
        int env_start = vec->env_starts[buf];
        int env_count = vec->env_counts[buf];
        int frozen_start = env_count;
        if (vec->num_policies > 1 && frozen_pct > 0.0f) {
            frozen_start = env_count - (int)(frozen_pct * env_count);
        }
        int* counts = (int*)calloc(vec->num_policies, sizeof(int));
        for (int e = 0; e < env_count; e++) {
            Env* eptr = &vec->envs[env_start + e];
            for (int s = 0; s < eptr->num_agents; s++) {
                int policy = e < frozen_start ? 0 : eptr->agents[s].policy;
                assert(policy >= 0 && policy < vec->num_policies);
                counts[policy]++;
            }
        }
        int offset = 0;
        for (int b = 0; b <= vec->num_policies; b++) {
            if (buf == 0) {
                vec->policy_layout[b] = offset;
            } else {
                assert(vec->policy_layout[b] == offset);
            }
            if (b < vec->num_policies) {
                offset += counts[b];
            }
        }
        assert(offset == apb);
        int* cursors = (int*)calloc(vec->num_policies, sizeof(int));
        for (int b = 0; b < vec->num_policies; b++) {
            cursors[b] = buf_start + vec->policy_layout[b];
        }
        for (int e = 0; e < env_count; e++) {
            Env* eptr = &vec->envs[env_start + e];
            int tag = 0;
            for (int s = 0; s < eptr->num_agents; s++) {
                int policy = e < frozen_start ? 0 : eptr->agents[s].policy;
                if (policy > tag) {
                    tag = policy;
                }
                int phys = cursors[policy]++;
                Agent* a = &eptr->agents[s];
                a->observations = vec->observations + (size_t)phys * OBS_SIZE;
                a->actions = vec->actions + (size_t)phys * NUM_ATNS;
                a->rewards = vec->rewards + phys;
                a->terminals = vec->terminals + phys;
                a->action_mask = vec->action_mask + (size_t)phys * vec->mask_size;
            }
            eptr->tag = tag;
            eptr->boundary_reached = 0;
        }
        free(cursors);
        free(counts);
    }
}

static void cpu_upload(PuffeRL* p, int start, int n, cudaStream_t stream) {
#ifdef PUF_METAL
    if (metal_shared_env()) return;
#endif
    VecEnv* v = p->vec;
    EnvBuf* e = &p->env;
    long mask = v->mask_size;
    cudaMemcpyAsync(e->obs.data + (size_t)start * OBS_SIZE,
        v->observations + (size_t)start * OBS_SIZE,
        n * OBS_SIZE * sizeof(obs_t), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(e->rewards.data + start, v->rewards + start,
        n * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(e->terminals.data + start, v->terminals + start,
        n * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(e->action_mask.data + (size_t)start * mask,
        v->action_mask + (size_t)start * mask,
        n * mask * sizeof(unsigned char), cudaMemcpyHostToDevice, stream);
}

#ifdef PUF_METAL
static void metal_rollout_group_step(PuffeRL* p, int buf, int t, int count, cudaStream_t stream) {
    int slot = p->hypers.async ? p->write_slot : 0;
    RolloutBuf view = rollout_time_view(&p->rollouts, slot * p->hypers.horizon, p->hypers.horizon);
    EnvBuf env = p->env;
    if (metal_shared_env()) {
        env.obs.data = p->vec->observations;
        env.actions.data = p->vec->actions;
        env.rewards.data = p->vec->rewards;
        env.terminals.data = p->vec->terminals;
        env.action_mask.data = p->vec->action_mask;
    }
    puf_mtl::RolloutStepArgs args[4];
    int block = p->vec->agents_per_buf;
    if (count == 1) {  // the packed single-buffer step, as pufferl_forward_step encodes it
        if (!metal_rollout_step(p, buf, t, view, &env, buf * block, block, slot, stream,
                p->metal_obs_bt[slot])) {
            puf_mtl::fatal("rollout step eligibility changed within a horizon");
        }
        return;
    }
    for (int i = 0; i < count; i++) {
        if (!metal_rollout_step(p, buf + i, t, view, &env, (buf + i) * block, block,
                slot, stream, p->metal_obs_bt[slot], true, &args[i])) {
            puf_mtl::fatal("rollout group eligibility changed within a horizon");
        }
        puf_mtl::rollout_use_packed(args[i].fwd, puf_mtl::gpu_addr(p->metal_packed));
    }
    puf_mtl::rollout_steps_packed(stream, args, count, p->metal_packed_split);
}
#endif

#ifdef PUF_METAL
static void metal_read_step_times(cudaEvent_t* e, float* acc) {
    float ms;
    cudaEventElapsedTime(&ms, e[MODEL_START], e[MODEL_END]);
    acc[PROF_MODEL] += ms;
    cudaEventElapsedTime(&ms, e[MODEL_END], e[COPY_END]);
    acc[PROF_COPY] += ms;
}

// Test hook (run_test_rollout_gate.sh): PUF_METAL_TEST_GATE_HOLD=N sleeps
// 4.5 s in the env step of the N-th gated step of the process (counted over
// workers), so Metal releases the gated commit behind it before gate_open.
static bool metal_gate_hold_now() {
    static const long n = [] {
        const char* e = getenv("PUF_METAL_TEST_GATE_HOLD");
        return e ? atol(e) : 0L;
    }();
    static long count = 0;
    return n > 0 && __atomic_add_fetch(&count, 1, __ATOMIC_RELAXED) == n;
}

// Pipelined horizon (metal_rollout_gate): while step t runs, step t+1 is
// encoded and committed behind a host gate (runtime.h gate_arm) and released
// right after the env step, so Metal's ~100 us commit -> GPU start latency
// overlaps step t's GPU time, the wake and the env step; release -> start
// is ~30 us. The same single dispatch with the same arguments, in the same
// order per stream: bit-identical. If Metal released a gated step early (a
// queue wait expires ~4 s after commit, e.g. under a stopped process), it ran
// as an empty dispatch (gate_ran false): the step after it is cancelled and
// both are encoded again. No gated step is pending when the horizon ends.
// Timing events rotate over three sets (step t+1 is recorded before step
// t-1's are read); a step's model time is from the previous step's
// completion to its own (env step included), its copy time 0 (shared IO).
static void metal_gated_horizon(PuffeRL* p, int buf, int group, cudaStream_t stream,
        cudaEvent_t* ev, float* acc, int env_start, int env_end, int agent_start, int agents) {
    VecEnv* vec = p->vec;
    Env* envs = vec->envs;
    int H = p->hypers.horizon;
    static const bool hold_step = getenv("PUF_METAL_TEST_STEP_HOLD")
        && strcmp(getenv("PUF_METAL_TEST_STEP_HOLD"), "0") != 0;
    auto hold = [&]() {
        if (!hold_step) return;
        pthread_t th;
        pthread_create(&th, NULL, metal_pack_release, puf_mtl::test_hold(stream));
        pthread_detach(th);
    };
    auto set = [&](int t) { return &ev[(t % 3) * NUM_EV]; };
    // Ungated step t: encode, commit, record.
    auto plain = [&](int t) {
        cudaEvent_t* e = set(t);
        hold();
        cudaEventRecord(e[MODEL_START], stream);
        metal_rollout_group_step(p, buf, t, group, stream);
        cudaEventRecord(e[MODEL_END], stream);
        cudaEventRecord(e[COPY_END], stream);
    };
    bool have = false, cur_gated = false;
    uint64_t cur = 0;
    for (int t = 0; t < H; t++) {
        PUF_TRACE(1, buf, t);
        if (!have) {
            plain(t);
            have = true;
            cur_gated = false;
        }
        PUF_TRACE(2, buf, t);
        uint64_t next = 0;
        if (t + 1 < H) {
            cudaEvent_t* e = set(t + 1);
            hold();
            cudaEventRecord(e[MODEL_START], stream);  // = step t's commit
            puf_mtl::gate_arm(stream);
            metal_rollout_group_step(p, buf, t + 1, group, stream);
            next = puf_mtl::gate_commit(stream);
            cudaEventRecord(e[MODEL_END], stream);
            cudaEventRecord(e[COPY_END], stream);
        }
        PUF_TRACE(3, buf, t);
        cudaEventSynchronize(set(t)[COPY_END]);
        PUF_TRACE(4, buf, t);
        if (cur_gated && !puf_mtl::gate_ran(stream, cur)) {
            // Released before gate_open: nothing ran. Cancel step t+1 (it
            // would read step t's outputs), then redo step t ungated.
            fprintf(stderr, "metal: rollout buffer %d step %d: gated step released early, "
                "encoding it again\n", buf, t);
            if (next) {
                puf_mtl::gate_open(stream, false);
                cudaEventSynchronize(set(t + 1)[COPY_END]);
                if (puf_mtl::gate_ran(stream, next)) {
                    puf_mtl::fatal("cancelled gated rollout step ran");
                }
                next = 0;
            }
            plain(t);
            cudaEventSynchronize(set(t)[COPY_END]);
        }
        if (t > 0) metal_read_step_times(set(t - 1), acc);
        PUF_TRACE(5, buf, t);
        memset(&vec->rewards[agent_start], 0, agents * sizeof(float));
        memset(&vec->terminals[agent_start], 0, agents * sizeof(float));
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        #pragma omp parallel for schedule(static) num_threads(metal_env_team(vec->num_workers, group))
        for (int i = env_start; i < env_end; i++) {
            puf_step(&envs[i]);
        }
        if (next && metal_gate_hold_now()) usleep(4500000);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        PUF_TRACE(6, buf, t);
        acc[PROF_ENV] += (t1.tv_sec - t0.tv_sec) * 1000.0f + (t1.tv_nsec - t0.tv_nsec) / 1e6f;
        if (next) puf_mtl::gate_open(stream, true);
        have = next != 0;
        cur_gated = have;
        cur = next;
    }
    cudaStreamSynchronize(stream);
    metal_read_step_times(set(H - 1), acc);
}
#endif

// CPU worker handshake. Atomic on worker_state[]; calloc leaves BUF_STARTING.
enum {
    BUF_STARTING,
    BUF_WAITING,
    BUF_RUNNING,
};

typedef struct {
    PuffeRL* pufferl;
    int buf;
} VecThreadArg;

static void* vec_thread_main(void* arg) {
    VecThreadArg* a = (VecThreadArg*)arg;
    PuffeRL* pufferl = a->pufferl;
    VecEnv* vec = pufferl->vec;
    int buf = a->buf;
    int* state = &vec->worker_state[buf];
    int horizon = pufferl->hypers.horizon;
    cudaSetDevice(pufferl->hypers.gpu_id);
    int apb = vec->agents_per_buf;
    int agent_start = buf * apb;
    int env_start = vec->env_starts[buf];
    Env* envs = vec->envs;
    cudaStream_t stream = pufferl->streams[buf];
    cublas_init_stream(stream);
    // Two sets of timing events, alternating per step: a step's model/copy
    // times are read one step later, when their completion stamps have long
    // been taken, instead of waiting for them right after the sync.
    cudaEvent_t ev[3 * NUM_EV];  // the third set: metal_gated_horizon only
    for (int i = 0; i < 3 * NUM_EV; i++) {
        cudaEventCreate(&ev[i]);
    }
    __atomic_store_n(state, BUF_WAITING, __ATOMIC_SEQ_CST);
    float* my_accum = &vec->accum[buf * NUM_PROF];
    struct timespec t0, t1;
    float ms = 0.0f;
    while (true) {
        while (__atomic_load_n(state, __ATOMIC_SEQ_CST) != BUF_RUNNING) {
#ifdef PUF_METAL
            int group = __atomic_load_n(&pufferl->metal_rollout_group, __ATOMIC_RELAXED);
            if (group > 1 && buf % group) usleep(1000);
#endif
            if (__atomic_load_n(&vec->shutdown, __ATOMIC_SEQ_CST)) {
                for (int i = 0; i < 3 * NUM_EV; i++) {
                    cudaEventDestroy(ev[i]);
                }
                return NULL;
            }
        }
        if (__atomic_load_n(&vec->shutdown, __ATOMIC_SEQ_CST)) {
            for (int i = 0; i < 3 * NUM_EV; i++) {
                cudaEventDestroy(ev[i]);
            }
            return NULL;
        }
        int h2d_pending = 0;
        int group = 1;
#ifdef PUF_METAL
        group = pufferl->metal_rollout_group;
#endif
        int agents = apb * group;
        int last_buf = buf + group - 1;
        int env_end = vec->env_starts[last_buf] + vec->env_counts[last_buf];
#ifdef PUF_METAL
        if (pufferl->metal_rollout_gate) {
            metal_gated_horizon(pufferl, buf, group, stream, ev, my_accum, env_start, env_end,
                agent_start, agents);
            for (int i = 0; i < NUM_PROF; i++) my_accum[i] *= group;
            __atomic_store_n(state, BUF_WAITING, __ATOMIC_SEQ_CST);
            continue;
        }
#endif
        for (int t = 0; t < horizon; t++) {
            cudaEvent_t* tev = &ev[(t & 1) * NUM_EV];
            cudaEvent_t* pev = &ev[(~t & 1) * NUM_EV];
#ifdef PUF_METAL
            static const bool hold_step = getenv("PUF_METAL_TEST_STEP_HOLD")
                && strcmp(getenv("PUF_METAL_TEST_STEP_HOLD"), "0") != 0;
            if (hold_step) {
                pthread_t th;
                pthread_create(&th, NULL, metal_pack_release, puf_mtl::test_hold(stream));
                pthread_detach(th);
            }
#endif
            PUF_TRACE(1, buf, t);
            cudaEventRecord(tev[MODEL_START], stream);
#ifdef PUF_METAL
            if (group > 1) metal_rollout_group_step(pufferl, buf, t, group, stream);
            else
#endif
            pufferl_forward(pufferl, buf, t, stream);
            PUF_TRACE(2, buf, t);
            cudaEventRecord(tev[MODEL_END], stream);
            PUF_TRACE(3, buf, t);
#ifdef PUF_METAL
            if (!metal_shared_env()) {
#endif
            cudaMemcpyAsync(
                &vec->actions[agent_start * NUM_ATNS],
                &pufferl->env.actions.data[agent_start * NUM_ATNS],
                agents * NUM_ATNS * sizeof(float),
                cudaMemcpyDeviceToHost, stream);
#ifdef PUF_METAL
            }
#endif
            cudaEventRecord(tev[COPY_END], stream);
            cudaStreamSynchronize(stream);
            PUF_TRACE(4, buf, t);
            if (h2d_pending) {  // step t - 1's events
                cudaEventElapsedTime(&ms, pev[MODEL_START], pev[MODEL_END]);
                my_accum[PROF_MODEL] += ms;
                cudaEventElapsedTime(&ms, pev[MODEL_END], pev[COPY_END]);
                my_accum[PROF_COPY] += ms;
                cudaEventElapsedTime(&ms, ev[H2D_START], ev[H2D_END]);
                my_accum[PROF_COPY] += ms;
                h2d_pending = 0;
            }
            PUF_TRACE(5, buf, t);
            memset(&vec->rewards[agent_start], 0, agents * sizeof(float));
            memset(&vec->terminals[agent_start], 0, agents * sizeof(float));
            clock_gettime(CLOCK_MONOTONIC, &t0);
            #pragma omp parallel for schedule(static) num_threads(metal_env_team(vec->num_workers, group))
            for (int i = env_start; i < env_end; i++) {
                puf_step(&envs[i]);
            }
            clock_gettime(CLOCK_MONOTONIC, &t1);
            PUF_TRACE(6, buf, t);
            my_accum[PROF_ENV] += (t1.tv_sec - t0.tv_sec) * 1000.0f
                + (t1.tv_nsec - t0.tv_nsec) / 1e6f;
            cudaEventRecord(ev[H2D_START], stream);
            cpu_upload(pufferl, agent_start, agents, stream);
            cudaEventRecord(ev[H2D_END], stream);
            h2d_pending = 1;
        }
        cudaStreamSynchronize(stream);
        if (h2d_pending) {
            cudaEvent_t* lev = &ev[((horizon - 1) & 1) * NUM_EV];
            cudaEventElapsedTime(&ms, lev[MODEL_START], lev[MODEL_END]);
            my_accum[PROF_MODEL] += ms;
            cudaEventElapsedTime(&ms, lev[MODEL_END], lev[COPY_END]);
            my_accum[PROF_COPY] += ms;
            cudaEventElapsedTime(&ms, ev[H2D_START], ev[H2D_END]);
            my_accum[PROF_COPY] += ms;
        }
        for (int i = 0; i < NUM_PROF; i++) my_accum[i] *= group;
        __atomic_store_n(state, BUF_WAITING, __ATOMIC_SEQ_CST);
    }
}

// Fresh episodes + zero RNN carry. Workers sit in BUF_WAITING between rollouts.
static void env_restart(PuffeRL* p) {
    VecEnv* vec = p->vec;
    if (PUF_BACKEND == PUF_GPU) {
        puf_reset(vec->envs);
    } else {
        #pragma omp parallel for schedule(static) num_threads(vec->num_workers)
        for (int i = 0; i < vec->size; i++) {
            puf_reset(&vec->envs[i]);
        }
        cpu_upload(p, 0, vec->total_agents, p->default_stream);
    }
    for (int b = 0; b < p->num_policies; b++) {
        for (int i = 0; i < vec->buffers; i++) {
            Prec* st = &p->policies[b].buffer_states[i];
            cudaMemset(st->data, 0, numel(st->shape) * sizeof(precision_t));
        }
    }
    cudaDeviceSynchronize();
}

static void env_start(PuffeRL* p) {
    VecEnv* vec = p->vec;
    if (PUF_BACKEND == PUF_GPU) {
        env_restart(p);
        return;
    }
    vec->worker_state = (int*)calloc(1, vec->buffers * sizeof(int));
    vec->threads = (pthread_t*)calloc(1, vec->buffers * sizeof(pthread_t));
    VecThreadArg* args = (VecThreadArg*)calloc(1,
        vec->buffers * sizeof(VecThreadArg));
    vec->accum = (float*)calloc(1, vec->buffers * NUM_PROF * sizeof(float));
    env_restart(p);
    for (int i = 0; i < vec->buffers; i++) {
        args[i].pufferl = p;
        args[i].buf = i;
        pthread_create(&vec->threads[i], NULL, vec_thread_main, &args[i]);
    }
    for (int i = 0; i < vec->buffers; i++) {
        int* state = &vec->worker_state[i];
        while (__atomic_load_n(state, __ATOMIC_SEQ_CST) != BUF_WAITING) {}
    }
}

static void rollout_start(PuffeRL* p) {
    if (PUF_BACKEND == PUF_GPU) {
        cudaStream_t stream = p->streams[0];
        puf_bind_stream(stream);
        cudaEvent_t* ev = p->profile.rollout_ev;
        int slot = p->write_slot;
        bool first = p->hypers.cudagraphs && p->gpu_rollout_graph[slot] == NULL;
        p->profile.skip_rollout_time = first;
        if (p->hypers.cudagraphs && !first) {
            cudaEventRecord(ev[0], stream);
            cudaGraphLaunch(p->gpu_rollout_graph[slot], stream);
            cudaEventRecord(ev[1], stream);
            return;
        }
        double t_cap = 0;
        if (first) {
            t_cap = wall_clock();
            assert(cudaStreamBeginCapture(
                stream, cudaStreamCaptureModeThreadLocal) == cudaSuccess
                && "cudaStreamBeginCapture failed");
        }
        int H = p->hypers.horizon;
        for (int t = 0; t < H; t++) {
            int base = t * EV_T;
            cudaEventRecord(ev[base + MODEL_START], stream);
            pufferl_forward_step(p, 0, t, stream);
            cudaEventRecord(ev[base + MODEL_END], stream);
            puf_step(p->vec->envs);
            cudaEventRecord(ev[base + ENV_END], stream);
        }
        if (first) {
            cudaGraph_t graph;
            assert(cudaStreamEndCapture(stream, &graph) == cudaSuccess
                && "cudaStreamEndCapture failed");
            assert(cudaGraphInstantiate(
                &p->gpu_rollout_graph[slot], graph, 0)
                == cudaSuccess && "cudaGraphInstantiate failed");
            cudaGraphDestroy(graph);
            double dt = wall_clock() - t_cap;
            p->start_time += dt;
            p->last_log_time += dt;
            cudaGraphLaunch(p->gpu_rollout_graph[slot], stream);
        }
        return;
    }
    int group = 1;
#ifdef PUF_METAL
    group = p->metal_rollout_group;
#endif
    for (int buf = 0; buf < p->vec->buffers; buf += group) {
        int* state = &p->vec->worker_state[buf];
        __atomic_store_n(state, BUF_RUNNING, __ATOMIC_SEQ_CST);
    }
}

void rollout_finish(PuffeRL* p, double t0) {
    if (PUF_BACKEND == PUF_GPU) {
        cudaStreamSynchronize(p->streams[0]);
        float model_ms = 0.0f, env_ms = 0.0f, ms;
        cudaEvent_t* ev = p->profile.rollout_ev;
        if (p->profile.skip_rollout_time) {
            p->profile.skip_rollout_time = 0;
            return;
        }
        if (p->hypers.cudagraphs && p->gpu_rollout_graph[p->write_slot] != NULL) {
            cudaEventElapsedTime(&ms, ev[0], ev[1]);
            p->profile.accum[PROF_ROLLOUT] += ms;
            return;
        }
        int H = p->hypers.horizon;
        for (int t = 0; t < H; t++) {
            int base = t * EV_T;
            cudaEventElapsedTime(&ms, ev[base + MODEL_START], ev[base + MODEL_END]);
            model_ms += ms;
            cudaEventElapsedTime(&ms, ev[base + MODEL_END], ev[base + ENV_END]);
            env_ms += ms;
        }
        p->profile.accum[PROF_MODEL] += model_ms;
        p->profile.accum[PROF_ENV] += env_ms;
        p->profile.accum[PROF_ROLLOUT] += model_ms + env_ms;
        return;
    }
    for (int buf = 0; buf < p->vec->buffers; buf++) {
        int* state = &p->vec->worker_state[buf];
        while (__atomic_load_n(state, __ATOMIC_SEQ_CST) != BUF_WAITING) {}
    }
    float sec = (float)(wall_clock() - t0);
    p->profile.accum[PROF_ROLLOUT] += sec * 1000.0f;
    float eval_prof[NUM_PROF] = {0};
    for (int buf = 0; buf < p->vec->buffers; buf++) {
        float* src = &p->vec->accum[buf * NUM_PROF];
        for (int i = 0; i < NUM_PROF; i++) {
            eval_prof[i] += src[i];
        }
        memset(src, 0, NUM_PROF * sizeof(float));
    }
    p->profile.accum[PROF_MODEL] += eval_prof[PROF_MODEL] / p->vec->buffers;
    p->profile.accum[PROF_ENV] += eval_prof[PROF_ENV] / p->vec->buffers;
    p->profile.accum[PROF_COPY] += eval_prof[PROF_COPY] / p->vec->buffers;
#ifdef PUF_WARM_START_TEST
    puf_warm_start_test_dump_rollout(p);
#endif
}

static void env_close(VecEnv* vec) {
    if (PUF_BACKEND == PUF_GPU) {
        puf_close(vec->envs);
        cudaFree(vec->log_scratch);
        return;
    }
    __atomic_store_n(&vec->shutdown, 1, __ATOMIC_SEQ_CST);
    for (int i = 0; i < vec->buffers; i++) {
        pthread_join(vec->threads[i], NULL);
    }
    for (int i = 0; i < vec->size; i++) {
        puf_close(&vec->envs[i]);
    }
#ifdef MY_VEC_CLOSE
    my_vec_close(vec->envs);
#endif
}

void vec_log(VecEnv* vec, Dict* out, int clear) {
    Log aggregate = {0};
    float* acc = (float*)&aggregate;
    env_log_sum(vec, &aggregate, clear);

    float n = aggregate.n;
    Dict env_out = {0};
    if (n > 0.0f) {
        for (int j = 0; j < LOG_NF; j++) {
            acc[j] /= n;
        }
        puf_log(&aggregate, &env_out);
    }
    dict_set(&env_out, "n", n);
    for (int i = 0; i < env_out.size; i++) {
        char key[256];
        snprintf(key, sizeof(key), "env/%s", env_out.items[i].key);
        dict_set(out, key, env_out.items[i].value);
    }
    dict_clear(&env_out);
}

static Prec slice_rows(Prec p, int off, int n) {
    int row = 1;
    for (int i = 1; i < PUF_MAX_DIMS && p.shape[i]; i++) {
        row *= (int)p.shape[i];
    }
    Prec out = p;
    out.data = p.data + (int64_t)off * row;
    out.shape[0] = n;
    return out;
}

static Float slice_rows(Float p, int off, int n) {
    int row = 1;
    for (int i = 1; i < PUF_MAX_DIMS && p.shape[i]; i++) {
        row *= (int)p.shape[i];
    }
    Float out = p;
    out.data = p.data + (int64_t)off * row;
    out.shape[0] = n;
    return out;
}

#ifndef PUF_METAL  // device code: Metal ports live in src/metal/shaders
// Transpose (A, B, C) → (B, A, C). Sequential, coalesced on dest rows.
// Two types: actions are float32 (large discrete IDs); everything else is Prec.
__global__ void transpose_102(precision_t* dst, const precision_t* src,
        int A, int B, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = A * B * C;
    if (idx >= total) {
        return;
    }
    int a = idx / (B * C);
    int rem = idx % (B * C);
    int b = rem / C;
    int c = rem % C;
    dst[b * A * C + a * C + c] = src[idx];
}

#if !defined(PRECISION_FLOAT)
__global__ void transpose_102(float* dst, const float* src, int A, int B, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = A * B * C;
    if (idx >= total) {
        return;
    }
    int a = idx / (B * C);
    int rem = idx % (B * C);
    int b = rem / C;
    int c = rem % C;
    dst[b * A * C + a * C + c] = src[idx];
}
#endif
#endif  // PUF_METAL

// Cosine decay base → min over t in [0, T). Double for t/T (float loses
// precision past 2^24). Caller passes epoch and total train epochs.
float cosine_annealing(float base, float min_v, long t, long T) {
    double u = (double)t / (double)T;
    return min_v + 0.5f * (base - min_v) * (1.0f + (float)cos(M_PI * u));
}

#ifndef PUF_METAL  // device code: Metal ports live in src/metal/shaders
__global__ void clamp_precision_kernel(precision_t* dst, float lo, float hi, int n) {
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n;
            idx += blockDim.x * gridDim.x) {
        float v = to_float(dst[idx]);
        dst[idx] = from_float(fminf(fmaxf(v, lo), hi));
    }
}

// 1-thread graph node. cudaEventElapsedTime does not timestamp in-graph events.
__global__ void puf_stamp(unsigned long long* dst) {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    *dst = t;
}
#endif  // PUF_METAL

#ifdef PUF_GOLDEN_DUMP
#include "../tests/metal_ref/train_dump.h"
#endif

static void train_epoch_gpu(PuffeRL* pufferl, RolloutBuf src, int slot,
        cudaStream_t stream) {
    Hypers* hypers = &pufferl->hypers;
    RolloutBuf train = pufferl->train_rollouts;
    RolloutBuf* rollouts = &train;
    unsigned long long* st = pufferl->profile.stamps + slot * NUM_TE;
    PUF_LAUNCH(puf_stamp, 1, 1, 0, stream)(st + TE_S);

    int T = src.observations.shape[0];
    int B = src.observations.shape[1];
    int obs_size = (int)src.observations.shape[2];
    int num_atns = (int)src.actions.shape[2];
    int mask_c = src.action_mask.shape[2];
#ifdef PUF_METAL
    if (pufferl->metal_obs_bt[slot]) {
        // Alias only the completed ready slot. The producer writes the other
        // slot, and the standard train/dW join precedes this slot's reuse.
        rollouts->observations = {.data = src.observations.data, .shape = {B, T, obs_size}};
    } else
#endif
    {
        PUF_LAUNCH(transpose_102, grid_size(T * B * obs_size), BLOCK_SIZE, 0, stream)(
            rollouts->observations.data, src.observations.data, T, B, obs_size);
    }
    PUF_LAUNCH(transpose_102, grid_size(T * B * num_atns), BLOCK_SIZE, 0, stream)(
        rollouts->actions.data, src.actions.data, T, B, num_atns);
    PUF_LAUNCH(transpose_102, grid_size(T * B), BLOCK_SIZE, 0, stream)(
        rollouts->logprobs.data, src.logprobs.data, T, B, 1);
    PUF_LAUNCH(transpose_102, grid_size(T * B), BLOCK_SIZE, 0, stream)(
        rollouts->rewards.data, src.rewards.data, T, B, 1);
    PUF_LAUNCH(transpose_102, grid_size(T * B), BLOCK_SIZE, 0, stream)(
        rollouts->terminals.data, src.terminals.data, T, B, 1);
    PUF_LAUNCH(transpose_102, grid_size(T * B), BLOCK_SIZE, 0, stream)(
        rollouts->values.data, src.values.data, T, B, 1);
    PUF_LAUNCH(transpose_102, grid_size(T * B * mask_c), BLOCK_SIZE, 0, stream)(
        rollouts->action_mask.data, src.action_mask.data, T, B, mask_c);

    PUF_LAUNCH(clamp_precision_kernel, grid_size(
        numel(rollouts->rewards.shape)), BLOCK_SIZE, 0, stream)(
        rollouts->rewards.data, -1.0f, 1.0f, numel(rollouts->rewards.shape));

    if (hypers->reset_every_horizon || src.initial_states.data == NULL) {
        cudaMemsetAsync(pufferl->train_state.data, 0,
            numel(pufferl->train_state.shape) * sizeof(precision_t), stream);
    } else {
        Prec slot_st = init_slot(src.initial_states, slot);
        cudaMemcpyAsync(pufferl->train_state.data, slot_st.data,
            numel(pufferl->train_state.shape) * sizeof(precision_t),
            cudaMemcpyDeviceToDevice, stream);
    }
    PUF_LAUNCH(puf_stamp, 1, 1, 0, stream)(st + TE_MID);

    int batch_size = hypers->total_agents * hypers->horizon;
    int mb_segs = hypers->minibatch_size / hypers->horizon;
    int total_minibatches = hypers->replay_ratio * batch_size / hypers->minibatch_size;
    int n_rows = (int)rollouts->observations.shape[0];
    int Nmb = (int)pufferl->train_buf.mb_advantages.shape[0];
    int Tmb = (int)pufferl->train_buf.mb_advantages.shape[1];
    constexpr int ADV_THREADS = 64;
    int adv_grid = (Nmb + ADV_THREADS - 1) / ADV_THREADS;
    Policy* primary = &pufferl->policies[0];
    for (int mb = 0; mb < total_minibatches; ++mb) {
        int dest_off = (mb * mb_segs) % n_rows;
        TrainGraph graph = pufferl->train_buf;
        graph.mb_obs = slice_rows(rollouts->observations, dest_off, Nmb);
        graph.mb_actions = slice_rows(rollouts->actions, dest_off, Nmb);
        graph.mb_logprobs = slice_rows(rollouts->logprobs, dest_off, Nmb);
        graph.mb_terminals = slice_rows(rollouts->terminals, dest_off, Nmb);
        graph.mb_rewards = slice_rows(rollouts->rewards, dest_off, Nmb);
        graph.mb_values = slice_rows(rollouts->values, dest_off, Nmb);
        graph.mb_action_mask = slice_rows(rollouts->action_mask, dest_off, Nmb);
        graph.mb_state = pufferl->train_state;
        DecoderWeights* dw_train = (DecoderWeights*)primary->weights.decoder;
        Prec p_logstd = {};
        if (dw_train->continuous) {
            p_logstd = dw_train->logstd;
        }
        Prec dec = arch_forward_train(&primary->arch, primary->weights,
            pufferl->train_activs, graph.mb_obs, graph.mb_state,
            graph.mb_terminals, dest_off, graph, p_logstd,
            pufferl->act_sizes, pufferl->ppo_bufs.grad_logits.data,
            pufferl->ppo_bufs.grad_values.data, stream);
        PUF_LAUNCH(puff_advantage, adv_grid, ADV_THREADS, 0, stream)(
            graph.mb_gae_v.data, graph.mb_rewards.data,
            graph.mb_terminals.data,
            hypers->vtrace ? graph.mb_imp.data : NULL,
            graph.mb_advantages.data, graph.mb_gae_v.data,
            hypers->gamma, hypers->gae_lambda,
            hypers->vtrace_rho_clip, hypers->vtrace_c_clip, Nmb, Tmb);
        graph.mb_returns = graph.mb_gae_v;

        ppo_loss_fwd_bwd(dec, p_logstd, graph,
            pufferl->act_sizes, pufferl->losses,
            hypers->clip_coef, hypers->vf_clip_coef, hypers->vf_coef,
            pufferl->ppo_bufs.ent_coef,
            pufferl->ppo_bufs, pufferl->is_continuous, stream);

#ifdef PUF_GOLDEN_DUMP
        golden_minibatch(pufferl, mb, stream);
#endif
        Float grad_logits = pufferl->ppo_bufs.grad_logits;
        Float grad_logstd = pufferl->is_continuous
            ? pufferl->ppo_bufs.grad_logstd : Float();
        Float grad_values = pufferl->ppo_bufs.grad_values;
        arch_backward(&primary->arch, primary->weights, pufferl->train_activs,
            grad_logits, grad_logstd, grad_values, stream);

        if (pufferl->nccl_comm != NULL && hypers->world_size > 1) {
            ncclAllReduce(pufferl->grad.data, pufferl->grad.data,
                numel(pufferl->grad.shape), NCCL_PRECISION, ncclAvg,
                pufferl->nccl_comm, stream);
        }
        muon_step(&pufferl->muon, primary->master_weights,
            pufferl->grad, hypers->max_grad_norm, stream);
        if (USE_BF16) {
            int n = numel(primary->param.shape);
            PUF_LAUNCH(cast, grid_size(n), BLOCK_SIZE, 0, stream)(
                primary->param.data, primary->master_weights.data, n);
        }
    }
    PUF_LAUNCH(puf_stamp, 1, 1, 0, stream)(st + TE_E);
}

void train_impl(PuffeRL* pufferl, RolloutBuf* src_arg) {
    Hypers* hypers = &pufferl->hypers;
    RolloutBuf src = src_arg ? *src_arg : pufferl->rollouts;
    cudaStream_t train_stream = pufferl->train_stream;

    int batch_size = hypers->total_agents * hypers->horizon;
    bool anneal_lr = hypers->anneal_lr;
    int current_epoch = pufferl->epoch;
    Muon* muon = &pufferl->muon;

    // Schedule over this rank's train epochs (same as outer loop), not global
    // total_timesteps/batch — multi-GPU would otherwise only traverse 1/W of the
    // cosine and never reach min_lr / min_ent.
    int total_epochs = hypers->total_timesteps / hypers->world_size / batch_size;
    if (anneal_lr) {
        float lr_min = hypers->min_lr_ratio * hypers->lr;
        float lr = cosine_annealing(hypers->lr, lr_min, current_epoch, total_epochs);
        cudaMemcpy(muon->lr, &lr, sizeof(float), cudaMemcpyHostToDevice);
    }

    // Annealed entropy coefficient — same cosine shape as lr. With PG signal
    // alive, the entropy bonus that kept early-training exploratory becomes
    // load-bearing dead weight late in training; cosine-decay frees the policy
    // to commit harder on what it has already learned.
    float current_ent_coef = hypers->ent_coef;
    if (hypers->anneal_ent_coef) {
        float ent_min = hypers->min_ent_coef_ratio * hypers->ent_coef;
        current_ent_coef = cosine_annealing(
            hypers->ent_coef, ent_min, current_epoch, total_epochs);
    }
    // Host write + H2D stay outside the graph; device ptr is what kernels read.
    cudaMemcpyAsync(pufferl->ppo_bufs.ent_coef, &current_ent_coef,
        sizeof(float), cudaMemcpyHostToDevice, train_stream);

    int slot = hypers->async ? pufferl->async_ready_slot : 0;
#ifdef PUF_GOLDEN_DUMP
    golden_before_train(pufferl, src, slot);
#endif
    int total_minibatches = hypers->replay_ratio * batch_size / hypers->minibatch_size;
    bool first = hypers->cudagraphs && pufferl->train_cudagraph[slot] == NULL;
    profile_begin("train_forward_backward", hypers->profile);
    if (first) {
        double t_cap = wall_clock();
        assert(cudaStreamBeginCapture(
            train_stream, cudaStreamCaptureModeThreadLocal)
            == cudaSuccess && "cudaStreamBeginCapture failed");
        train_epoch_gpu(pufferl, src, slot, train_stream);
        cudaGraph_t graph;
        assert(cudaStreamEndCapture(train_stream, &graph)
            == cudaSuccess && "cudaStreamEndCapture failed");
        assert(cudaGraphInstantiate(
            &pufferl->train_cudagraph[slot], graph, 0)
            == cudaSuccess && "cudaGraphInstantiate failed");
        cudaGraphDestroy(graph);
        double dt = wall_clock() - t_cap;
        pufferl->start_time += dt;
        pufferl->last_log_time += dt;
    }
    if (hypers->cudagraphs) {
        cudaGraphLaunch(pufferl->train_cudagraph[slot], train_stream);
    } else {
        train_epoch_gpu(pufferl, src, slot, train_stream);
    }
    profile_end(hypers->profile);
    PUF_TRACE(12, 0, 0);

    cudaStreamSynchronize(train_stream);

    if (total_minibatches > 0) {
        unsigned long long h[NUM_TE];
        cudaMemcpy(h, pufferl->profile.stamps + slot * NUM_TE,
            sizeof(h), cudaMemcpyDeviceToHost);
        pufferl->profile.accum[PROF_TRAIN_MISC] +=
            (float)(h[TE_MID] - h[TE_S]) * 1e-6f;
        pufferl->profile.accum[PROF_TRAIN_MODEL] +=
            (float)(h[TE_E] - h[TE_MID]) * 1e-6f;
    }
#ifdef PUF_GOLDEN_DUMP
    golden_after_train(pufferl);
#endif
    pufferl->epoch += 1;
}

// Load policy weights (flat fp32 checkpoint). Safe between rollouts —
// graphs hold the pointer, not a copy of the data.
// Checkpoint I/O (load/save weights).
void mkdir_p(const char* path) {
    char tmp[1024];
    snprintf(tmp, sizeof(tmp), "%s", path);
    for (char* p = tmp + 1; *p; p++) {
        if (*p == '/') {
            *p = 0;
            assert((mkdir(tmp, 0777) == 0 || errno == EEXIST)
                && "failed to create directory");
            *p = '/';
        }
    }
    assert((mkdir(tmp, 0777) == 0 || errno == EEXIST)
        && "failed to create directory");
}

void puf_find_latest_checkpoint(const char* dir,
        char* out, size_t out_size, time_t* best_time) {
    DIR* dp = opendir(dir);
    if (!dp) {
        return;
    }

    struct dirent* ent = NULL;
    while ((ent = readdir(dp))) {
        if (strcmp(ent->d_name, ".") == 0 || strcmp(ent->d_name, "..") == 0) {
            continue;
        }

        char path[4096];
        snprintf(path, sizeof(path), "%s/%s", dir, ent->d_name);

        struct stat st;
        if (stat(path, &st) != 0) {
            continue;
        }

        if (S_ISDIR(st.st_mode)) {
            puf_find_latest_checkpoint(path, out, out_size, best_time);
        } else if (S_ISREG(st.st_mode)) {
            size_t plen = strlen(path);
            if (plen < 4 || strcmp(path + plen - 4, ".bin") != 0) continue;
            if (st.st_ctime < *best_time) continue;
            *best_time = st.st_ctime;
            snprintf(out, out_size, "%s", path);
        }
    }

    closedir(dp);
}

const char* puf_checkpoint_path_key(Ini* ini, const char* key,
        char* out, size_t out_size) {
    const char* load_path = puf_ini_get_str(ini, "base", key);
    if (!load_path || strcmp(load_path, "None") == 0) {
        return NULL;
    }

    if (strcmp(load_path, "latest") != 0) {
        return load_path;
    }

    char root[2048];
    snprintf(root, sizeof(root), "%s/%s",
        puf_ini_get_str(ini, "base", "checkpoint_dir"),
        puf_ini_get_str(ini, "base", "env_name"));

    out[0] = 0;
    time_t best_time = 0;
    puf_find_latest_checkpoint(root, out, out_size, &best_time);
    assert(out[0] && "no .bin checkpoints found");
    return out;
}

// Legacy checkpoint bytes are a flat master-weight array, including allocator
// padding. Equal element counts alone cannot identify the source precision.
static bool puf_checkpoint_layout_dependent(Allocator* alloc) {
    size_t bf = 0, fp = 0;
    for (int i = 0; i < alloc->num_regs; i++) {
        bf = (bf + 15) & ~size_t(15);
        fp = (fp + 15) & ~size_t(15);
        if (bf / 2 != fp / 4) return true;
        size_t n = numel(alloc->regs[i].shape);
        bf += n * 2;
        fp += n * 4;
    }
    return false;
}

static void puf_checkpoint_layout_path(char* out, size_t n, const char* path) {
    int len = snprintf(out, n, "%s.layout", path);
    assert(len >= 0 && (size_t)len < n && "checkpoint layout path too long");
}

static void puf_checkpoint_check_layout(Allocator* alloc, const char* path) {
    if (!puf_checkpoint_layout_dependent(alloc)) return;
    // Let the ordinary loader retain its missing-file/size diagnostics.
    struct stat st;
    if (stat(path, &st) != 0 || !S_ISREG(st.st_mode)
            || st.st_size != (off_t)(alloc->total_elems * sizeof(float))) return;
    char layout[4096];
    puf_checkpoint_layout_path(layout, sizeof(layout), path);
    FILE* fp = fopen(layout, "r");
    if (!fp) {
        // Legacy checkpoints (and upstream ones) carry no sidecar: keep loading
        // them as before, i.e. assume they were written by this build precision.
        fprintf(stderr,
            "warning: %s has precision-dependent padding and no .layout sidecar; "
            "assuming it was saved by this build's precision (%s). Loading a "
            "checkpoint from the other precision would misplace weights.\n",
            path, sizeof(precision_t) == 2 ? "BF16" : "FP32");
        return;
    }
    int source_bytes = 0;
    bool known = fscanf(fp, "PUF_LAYOUT_V1 %d", &source_bytes) == 1
        && (source_bytes == 2 || source_bytes == 4);
    fclose(fp);
    if (!known || source_bytes != sizeof(precision_t)) {
        fprintf(stderr,
            "checkpoint parameter layout mismatch: %s has precision-dependent "
            "padding. Cross-precision loading is unsupported for this layout; "
            "use the original build precision and retain its .layout sidecar.\n", path);
        exit(1);
    }
}

void puf_save_weights(PuffeRL* p, const char* path) {
    Float mw = p->policies[0].master_weights;
    int64_t nbytes = numel(mw.shape) * sizeof(float);
    char* buf = (char*)malloc(nbytes);
    cudaMemcpy(buf, mw.data, nbytes, cudaMemcpyDeviceToHost);
    char tmp[4096];
    snprintf(tmp, sizeof(tmp), "%s.tmp.%d", path, getpid());
    FILE* fp = fopen(tmp, "wb");
    assert(fp && "failed to open weights for writing");
    assert(fwrite(buf, 1, nbytes, fp) == (size_t)nbytes
        && "failed to write weights");
    fclose(fp);
    free(buf);
    assert(rename(tmp, path) == 0 && "failed to publish weights");
    // Preserve the raw checkpoint format and seeded hashes. Only layouts whose
    // padding differs need source-precision metadata alongside those bytes.
    if (puf_checkpoint_layout_dependent(&p->policies[0].params_alloc)) {
        char layout[4096], layout_tmp[4096];
        puf_checkpoint_layout_path(layout, sizeof(layout), path);
        snprintf(layout_tmp, sizeof(layout_tmp), "%s.tmp.%d", layout, getpid());
        FILE* meta = fopen(layout_tmp, "w");
        assert(meta && "failed to open checkpoint layout metadata");
        assert(fprintf(meta, "PUF_LAYOUT_V1 %zu\n", sizeof(precision_t)) > 0);
        assert(fclose(meta) == 0);
        assert(rename(layout_tmp, layout) == 0 && "failed to publish checkpoint layout metadata");
    }
}

void puf_load_weights_into(Float dst, Prec params,
        cudaStream_t stream, const char* path) {
    int64_t nbytes = numel(dst.shape) * sizeof(float);
    struct stat st;
    if (stat(path, &st) != 0) {
        fprintf(stderr, "checkpoint load failed: %s: %s\n", path, strerror(errno));
        exit(1);
    }
    if (!S_ISREG(st.st_mode) || st.st_size != (off_t)nbytes) {
        fprintf(stderr,
            "checkpoint shape/precision mismatch: %s has %lld bytes; policy expects "
            "%lld bytes (%lld serialized FP32 master weights). Check the environment, "
            "policy.hidden_size, and policy.num_layers; BF16 and FP32 builds both use "
            "FP32 checkpoint files.\n",
            path, (long long)st.st_size, (long long)nbytes,
            (long long)numel(dst.shape));
        exit(1);
    }
    FILE* fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "checkpoint load failed: %s: %s\n", path, strerror(errno));
        exit(1);
    }
    char* buf = (char*)malloc(nbytes);
    if (!buf) {
        fprintf(stderr, "checkpoint load failed: cannot allocate %lld host bytes\n",
            (long long)nbytes);
        fclose(fp);
        exit(1);
    }
    size_t nread = fread(buf, 1, nbytes, fp);
    fclose(fp);
    if ((int64_t)nread != nbytes) {
        fprintf(stderr, "checkpoint load failed: short read from %s (%zu/%lld bytes)\n",
            path, nread, (long long)nbytes);
        free(buf);
        exit(1);
    }
    cudaMemcpy(dst.data, buf, nbytes, cudaMemcpyHostToDevice);
    free(buf);
    if (USE_BF16) {
        int n = numel(params.shape);
        PUF_LAUNCH(cast, grid_size(n), BLOCK_SIZE, 0, stream)(params.data, dst.data, n);
    }
}

// Load weights into policies[i] (full index; 0 = trainable, i>0 = frozen).
void pufferl_load_policy(PuffeRL* pufferl, int i, const char* path) {
    assert(i >= 0 && i < pufferl->num_policies);
    Policy* pol = &pufferl->policies[i];
#ifndef PUF_TEST_SKIP_CHECKPOINT_LAYOUT
    puf_checkpoint_check_layout(&pol->params_alloc, path);
#endif
    puf_load_weights_into(pol->master_weights, pol->param,
        pufferl->default_stream, path);
#if defined(PUFFER_NETHACK) && defined(NH_MSG_FROZEN)
    nethack_refresh_weight_cache(pol->weights.encoder, pufferl->default_stream);
#endif
    cudaDeviceSynchronize();
}

// fp32 master weights: alias param buffer in float mode; separate fp32 copy in bf16.
// cast_now: copy param→master now (primary after init). Frozen policies load later.
static void master_weights_setup(Float* mw, Prec* param,
        bool cast_now, cudaStream_t stream) {
    long n = numel(param->shape);
    if (USE_BF16) {
        *mw = (Float){.shape = {n}};
        cudaMalloc((void**)&mw->data, n * sizeof(float));
        if (cast_now) {
            PUF_LAUNCH(cast, grid_size(n), BLOCK_SIZE, 0, stream)(mw->data, param->data, n);
        }
    } else {
        *mw = (Float){.data = (float*)param->data, .shape = {n}};
    }
}

PuffeRL* create_pufferl(Ini* ini, TrainContext* ctx) {
    Hypers hypers = {
        .horizon = puf_ini_get(ini, "train", "horizon"),
        .total_agents = puf_ini_get(ini, "vec", "total_agents"),
        .num_buffers = puf_ini_get(ini, "vec", "num_buffers"),
        .hidden_size = puf_ini_get(ini, "policy", "hidden_size"),
        .num_layers = puf_ini_get(ini, "policy", "num_layers"),
        .lr = puf_ini_get(ini, "train", "learning_rate"),
        .min_lr_ratio = puf_ini_get(ini, "train", "min_lr_ratio"),
        .anneal_lr = puf_ini_get(ini, "train", "anneal_lr") != 0,
        .momentum = puf_ini_get(ini, "train", "momentum"),
        .minibatch_size = puf_ini_get(ini, "train", "minibatch_size"),
        .replay_ratio = puf_ini_get(ini, "train", "replay_ratio"),
        .total_timesteps = puf_ini_get(ini, "train", "total_timesteps"),
        .max_grad_norm = puf_ini_get(ini, "train", "max_grad_norm"),
        .clip_coef = puf_ini_get(ini, "train", "clip_coef"),
        .vf_clip_coef = puf_ini_get(ini, "train", "vf_clip_coef"),
        .vf_coef = puf_ini_get(ini, "train", "vf_coef"),
        .ent_coef = puf_ini_get(ini, "train", "ent_coef"),
        .min_ent_coef_ratio = puf_ini_get(ini, "train", "min_ent_coef_ratio"),
        .anneal_ent_coef = puf_ini_get(ini, "train", "anneal_ent_coef") != 0,
        .gamma = puf_ini_get(ini, "train", "gamma"),
        .gae_lambda = puf_ini_get(ini, "train", "gae_lambda"),
        .vtrace = puf_ini_get(ini, "train", "vtrace") != 0,
        .vtrace_rho_clip = puf_ini_get(ini, "train", "vtrace_rho_clip"),
        .vtrace_c_clip = puf_ini_get(ini, "train", "vtrace_c_clip"),
        .async = puf_ini_get(ini, "base", "async") != 0,
        .reset_every_horizon = puf_ini_get(ini, "base", "reset_every_horizon") != 0,
#ifdef PUF_METAL
        .cudagraphs = false,  // CUDA graphs have no Metal equivalent
#else
        .cudagraphs = puf_ini_get(ini, "base", "cudagraphs") >= 0,
#endif
        .profile = puf_ini_get(ini, "base", "profile") != 0,
        .rank = ctx->rank,
        .world_size = ctx->world_size,
        .gpu_id = ctx->gpu_id,
        .num_threads = puf_ini_get(ini, "vec", "num_threads"),
        .seed = puf_ini_get(ini, "base", "seed"),
    };
    Dict vec_kwargs = {0};
    dict_copy(&vec_kwargs, puf_ini_section(ini, "vec", 0));
    Dict* env_kwargs = puf_ini_section(ini, "env", 0);
    ncclUniqueId* nccl_id = ctx->nccl_id;

    PuffeRL* pufferl = (PuffeRL*)calloc(1, sizeof(PuffeRL));
    pufferl->hypers = hypers;
    snprintf(pufferl->env_name, sizeof(pufferl->env_name), "%s", PUFFER_ENV_NAME);

    cudaSetDevice(hypers.gpu_id);
#ifdef PUF_METAL
    puf_mtl::MemoryInfo memory_at_create = puf_mtl::memory_info();
#endif
    cublas_init_handle();

    if (hypers.world_size > 1) {
        ncclCommInitRank(&pufferl->nccl_comm, hypers.world_size, *nccl_id, hypers.rank);
        printf("Rank %d/%d: NCCL initialized\n", hypers.rank, hypers.world_size);
    }

    pufferl->seed = (ulong)hypers.seed + hypers.rank;

    // GPU tensors allocated into pufferl->env; vec aliases them.
    int total_agents = dict_get(&vec_kwargs, "total_agents");
    int num_buffers = dict_get(&vec_kwargs, "num_buffers");
    VecEnv* vec = (VecEnv*)calloc(1, sizeof(VecEnv));
    vec->total_agents = total_agents;
    vec->buffers = num_buffers;
    vec->agents_per_buf = total_agents / num_buffers;
    // Total policies including trainable policies[0]. Default 1 = train only.
    int num_policies = dict_get(&vec_kwargs, "num_policies");
    if (num_policies < 1) {
        num_policies = 1;
    }
    // GPU envs have no per-env tags / frozen layout — selfplay and match stay CPU-only.
    assert(!(PUF_BACKEND == PUF_GPU
            && (num_policies > 1 || puf_ini_get(ini, "selfplay", "enabled")))
        && "GPU env backend does not support selfplay or multi-policy (match)");

    // Discrete action layout. Continuous dims are size 1. Mask width is act_n.
    int num_action_heads = NUM_ATNS;
    int act_sizes[] = ACT_SIZES;
    int act_n = 0;
    int n_cont = 0;
    int n_disc = 0;
    for (int i = 0; i < num_action_heads; i++) {
        if (act_sizes[i] == 1) {
            n_cont++;
        } else {
            n_disc++;
        }
        act_n += act_sizes[i];
    }
    assert(!(n_cont > 0 && n_disc > 0)
        && "mixed continuous/discrete action spaces not supported");
    bool is_continuous = n_cont > 0;
    pufferl->is_continuous = is_continuous;
    vec->num_policies = num_policies;
    vec->policy_layout = (int*)calloc(1, (vec->num_policies + 1) * sizeof(int));
    vec->mask_size = act_n;

    // Device env IO (EnvBuf).
    pufferl->env = {
        .obs =         {.shape = {total_agents, OBS_SIZE}},
        .actions =     {.shape = {total_agents, NUM_ATNS}},
        .rewards =     {.shape = {total_agents}},
        .terminals =   {.shape = {total_agents}},
        .action_mask = {.shape = {total_agents, act_n}},
    };
    EnvBuf* env = &pufferl->env;
    size_t mask_bytes = total_agents * act_n * sizeof(unsigned char);
    cudaMalloc((void**)&env->obs.data, total_agents * OBS_SIZE * sizeof(obs_t));
    cudaMalloc((void**)&env->actions.data, total_agents * NUM_ATNS * sizeof(float));
    cudaMalloc((void**)&env->rewards.data, total_agents * sizeof(float));
    cudaMalloc((void**)&env->terminals.data, total_agents * sizeof(float));
    cudaMalloc((void**)&env->action_mask.data, mask_bytes);
    cudaMemset(env->obs.data, 0, total_agents * OBS_SIZE * sizeof(obs_t));
    cudaMemset(env->actions.data, 0, total_agents * NUM_ATNS * sizeof(float));
    cudaMemset(env->rewards.data, 0, total_agents * sizeof(float));
    cudaMemset(env->terminals.data, 0, total_agents * sizeof(float));
    cudaMemset(env->action_mask.data, 1, mask_bytes);

    env_setup(pufferl, vec, &vec_kwargs, env_kwargs);
    pufferl->vec = vec;

    cudaMalloc((void**)&pufferl->profile.stamps,
        2 * NUM_TE * sizeof(unsigned long long));
    if (PUF_BACKEND == PUF_GPU) {
        int H = hypers.horizon;
        pufferl->profile.rollout_ev = (cudaEvent_t*)calloc(
            1, EV_T * H * sizeof(cudaEvent_t));
        for (int i = 0; i < EV_T * H; i++) {
            assert(cudaEventCreate(&pufferl->profile.rollout_ev[i]) == cudaSuccess);
        }
    }
    nvmlInit();
    nvmlDeviceGetHandleByIndex(hypers.gpu_id, &pufferl->nvml_device);

    int input_size = OBS_SIZE;
    int hidden_size = hypers.hidden_size;
    int num_layers = hypers.num_layers;
    int decoder_output_size = is_continuous ? num_action_heads : act_n;
    int minibatch_segments = hypers.minibatch_size / hypers.horizon;
    int B_TT = minibatch_segments * hypers.horizon;
    int horizon = hypers.horizon;
    int agents_per_buf = total_agents / num_buffers;

    // Dedicated learner stream (always non-default; nonblocking when async).
    if (hypers.async) {
        assert(cudaStreamCreateWithFlags(
            &pufferl->train_stream, cudaStreamNonBlocking) == cudaSuccess);
    } else {
        assert(cudaStreamCreate(&pufferl->train_stream) == cudaSuccess);
    }
    cublas_init_stream(pufferl->train_stream);

    Allocator* acts = &pufferl->activ_alloc;
    Allocator* grads = &pufferl->grads_alloc;

    // All policies: policies[0] trainable, rest historical/frozen (rollout-only).
    int hist_hidden = dict_get(&vec_kwargs, "hist_policy_hidden_size");
    int hist_layers = dict_get(&vec_kwargs, "hist_policy_num_layers");
    pufferl->num_policies = vec->num_policies;
    assert(!(pufferl->num_policies > 1 && (hist_hidden <= 0 || hist_layers <= 0))
        && "num_policies > 1 requires hist_policy_hidden_size and hist_policy_num_layers > 0");
    pufferl->policies = (Policy*)calloc(1, pufferl->num_policies * sizeof(Policy));

    for (int b = 0; b < pufferl->num_policies; b++) {
        Policy* pol = &pufferl->policies[b];
        pol->frozen = (b > 0);
        int h = pol->frozen ? hist_hidden : hidden_size;
        int L = pol->frozen ? hist_layers : num_layers;
        int slice = vec->policy_layout[b + 1] - vec->policy_layout[b];
        assert(slice > 0 && "policy has no agents");

        pol->arch = build_arch(input_size, h, L,
            decoder_output_size, is_continuous, hypers.horizon);
        pol->weights = weights_create(&pol->arch, &pol->params_alloc);
        Allocator* aalloc = pol->frozen ? &pol->activ_alloc : acts;
        pol->buf_acts = (Activations*)calloc(
            1, num_buffers * sizeof(Activations));
        pol->buffer_states = (Prec*)calloc(1, num_buffers * sizeof(Prec));
        for (int i = 0; i < num_buffers; i++) {
            pol->buf_acts[i] = arch_reg_rollout(
                &pol->arch, pol->weights, aalloc, slice);
            pol->buffer_states[i] = {.shape = {L, slice, h}};
            alloc_register(aalloc, &pol->buffer_states[i]);
        }
    }

    // Train-only extras on policies[0] (trainable).
    Policy* primary = &pufferl->policies[0];
    if (hypers.async) {
        pufferl->actor_weights = weights_create(
            &primary->arch, &pufferl->weight_alloc);
    }
    pufferl->train_activs = arch_reg_train(
        &primary->arch, primary->weights, acts, grads, B_TT);

    // Async is Cleanba 2-slot only (lag 1). Sync uses a single slot.
    int async_slots = hypers.async ? 2 : 1;
    pufferl->async_num_slots = async_slots;
    int rollout_horizon = async_slots * horizon;
    register_rollout_buffers(&pufferl->rollouts,
        acts, rollout_horizon, total_agents, input_size, num_action_heads, act_n);
    // Carry path: per-slot initial RNN states. reset_every_horizon zeros train_state.
    if (!hypers.reset_every_horizon) {
        pufferl->rollouts.initial_states = {
            .shape = {async_slots, num_layers, total_agents, hidden_size}};
        alloc_register(acts, &pufferl->rollouts.initial_states);
    }
    register_train_buffers(pufferl->train_buf, acts, minibatch_segments, horizon);
    register_rollout_buffers(&pufferl->train_rollouts,
        acts, total_agents, horizon, input_size, num_action_heads, act_n);
    register_ppo_buffers(pufferl->ppo_bufs, acts, minibatch_segments,
        hypers.horizon, decoder_output_size, is_continuous);
    pufferl->train_state = {.shape = {num_layers, total_agents, hidden_size}};
    alloc_register(acts, &pufferl->train_state);

    cudaMalloc((void**)&pufferl->rng_offset, (num_buffers + 1) * sizeof(long));
    cudaMemset(pufferl->rng_offset, 0, (num_buffers + 1) * sizeof(long));
    cudaMalloc((void**)&pufferl->act_sizes, num_action_heads * sizeof(int));
    cudaMalloc((void**)&pufferl->losses, NUM_LOSSES * sizeof(float));

    muon_init(&pufferl->muon, &primary->params_alloc, hypers.momentum, acts);

#ifdef PUF_METAL
#ifndef PUF_METAL_MEMORY_PREFLIGHT
#define PUF_METAL_MEMORY_PREFLIGHT 1
#endif
    if (PUF_METAL_MEMORY_PREFLIGHT) {
        // Conservative shared-buffer peak: registered pools, later rollout
        // streams/workspaces, eligible packed weights, and one init scratch.
        // Driver-private command/pipeline storage is not predictable here.
        auto rounded = [](size_t n) {
            const size_t page = 16 * 1024;
            n = n ? n : 1;
            return (n + page - 1) / page * page;
        };
        size_t future = 0, largest = 0, init_scratch = 0;
        auto planned = [&](size_t n) {
            size_t bytes = rounded(n);
            future += bytes;
            if (bytes > largest) largest = bytes;
        };
        for (int b = 0; b < pufferl->num_policies; b++) {
            Policy* pol = &pufferl->policies[b];
            planned((size_t)pol->params_alloc.total_bytes);
            if (pol->frozen) planned((size_t)pol->activ_alloc.total_bytes);
            if (USE_BF16) planned((size_t)pol->params_alloc.total_elems * sizeof(float));
            if (b == 0) {
                // Kaiming uses a full FP32 tensor even in FP32 builds;
                // normal init rounds an odd element count up to an even one.
                for (int j = 0; j < pol->params_alloc.num_regs; j++) {
                    size_t n = numel(pol->params_alloc.regs[j].shape);
                    init_scratch = (std::max)(init_scratch, rounded(((n + 1) & ~size_t(1)) * sizeof(float)));
                }
            }
        }
        if (hypers.async) planned((size_t)pufferl->weight_alloc.total_bytes);
        planned((size_t)grads->total_bytes);
        planned((size_t)acts->total_bytes);
        for (int i = 0; i < num_buffers; i++) {
            planned((size_t)agents_per_buf * sizeof(curandStatePhilox4_32_10_t));
            // stream_create's four constant rings and progress/gate words.
            future += puf_mtl::stream_buffer_bytes();
            if (PUF_BACKEND != PUF_GPU || i == 0) planned(kCublasWsBytes);
        }
#if defined(PUFFER_NETHACK) && defined(NH_MSG_FROZEN)
        for (int b = 0; b < pufferl->num_policies; b++) {
            planned((size_t)NH_MSG_VOCAB * NH_MSG_HID * sizeof(precision_t));
        }
#endif
        size_t packed_bytes = 0;
        if (metal_rollout_pack_split(pufferl)) {
            packed_bytes = puf_mtl::rollout_packed_floats(input_size, hidden_size,
                num_layers, decoder_output_size + 1) * sizeof(float);
            planned(packed_bytes);
        }
        largest = (std::max)(largest, init_scratch);
        puf_mtl::MemoryInfo now = puf_mtl::memory_info();
        size_t config_live = now.process_live >= memory_at_create.process_live
            ? now.process_live - memory_at_create.process_live : now.process_live;
        size_t outside = now.current_allocated > config_live
            ? now.current_allocated - config_live : 0;
        size_t available = now.recommended_max_working_set > outside
            ? now.recommended_max_working_set - outside : 0;
        size_t hard_limit = now.max_buffer_length;
#ifdef PUF_METAL_MEMORY_TEST
        const char* test_budget = getenv("PUF_METAL_TEST_MEMORY_BUDGET_BYTES");
        if (test_budget && *test_budget) available = (size_t)strtoull(test_budget, NULL, 10);
        const char* test_limit = getenv("PUF_METAL_TEST_MAX_BUFFER_BYTES");
        if (test_limit && *test_limit) hard_limit = (size_t)strtoull(test_limit, NULL, 10);
#endif
        size_t requested = config_live + future + init_scratch;
#ifdef PUF_METAL_MEMORY_TEST
        fprintf(stderr, "metal memory plan: live=%zu future=%zu init_scratch=%zu "
            "packed=%zu stream_buffers=%zu workspace=%zu buffers=%d peak=%zu largest=%zu recommended=%zu max_buffer=%zu\n",
            config_live, future, init_scratch, packed_bytes, puf_mtl::stream_buffer_bytes(),
            kCublasWsBytes, num_buffers, requested, largest, available, hard_limit);
        // Inspect real large geometries without allocating them on a shared GPU.
#endif
        if (largest > hard_limit || requested > available) {
            bool hard = largest > hard_limit;
            fprintf(stderr,
                "metal: %s: requested %.3f GB peak, recommended available %.3f GB "
                "(MTLDevice recommendedMaxWorkingSetSize %.3f GB; %.3f GB already used "
                "outside this trainer; largest page-rounded allocation %.3f GB, "
                "maxBufferLength %.3f GB).\n"
                "Reduce vec.total_agents, train.minibatch_size, train.horizon, or "
                "policy.hidden_size; try BF16 or base.async=0 to reduce memory.\n",
                hard ? "configuration memory preflight failed (hard buffer limit)"
                     : "WARNING: configuration exceeds the working-set recommendation; continuing",
                requested / 1e9, available / 1e9,
                now.recommended_max_working_set / 1e9, outside / 1e9,
                largest / 1e9, hard_limit / 1e9);
            if (hard) exit(1);
        }
#ifdef PUF_METAL_MEMORY_TEST
        if (getenv("PUF_METAL_TEST_PLAN_ONLY")) exit(0);
#endif
    }
#endif

    // Allocate all policy param/activ pools, then train grads + shared acts.
    for (int b = 0; b < pufferl->num_policies; b++) {
        Policy* pol = &pufferl->policies[b];
        alloc_create(&pol->params_alloc);
        if (pol->frozen) {
            alloc_create(&pol->activ_alloc);
        }
        pol->param = {
            .data = (precision_t*)pol->params_alloc.mem,
            .shape = {pol->params_alloc.total_elems},
        };
    }
    if (hypers.async) {
        alloc_create(&pufferl->weight_alloc);
    }
    alloc_create(grads);
    alloc_create(acts);

    pufferl->grad = {.data = (precision_t*)grads->mem, .shape = {grads->total_elems}};
    if (hypers.async) {
        pufferl->actor_param = {
            .data = (precision_t*)pufferl->weight_alloc.mem,
            .shape = {pufferl->weight_alloc.total_elems},
        };
    }

    ulong init_seed = hypers.seed;
    weights_init(&primary->arch,
        primary->weights, &init_seed, pufferl->default_stream);
#if defined(PUFFER_NETHACK) && defined(NH_MSG_FROZEN)
    if (hypers.async) {
        // Actor snapshots share policy 0's frozen table, not an uninitialized
        // cache in the separately created actor weight object.
        auto* actor = (NethackEncoderWeights*)pufferl->actor_weights.encoder;
        actor->msg_w0_dev = ((NethackEncoderWeights*)primary->weights.encoder)->msg_w0_dev;
    }
#endif
    // Primary: cast param→master now. Frozen: load later via pufferl_load_policy.
    for (int b = 0; b < pufferl->num_policies; b++) {
        Policy* pol = &pufferl->policies[b];
        master_weights_setup(&pol->master_weights, &pol->param,
            !pol->frozen, pufferl->default_stream);
    }
    if (hypers.async) {
        puf_copy(&pufferl->actor_param, &primary->param, pufferl->default_stream);
        cudaStreamSynchronize(pufferl->default_stream);
    }

    // Per-buffer persistent RNG states
    pufferl->rng_states = (curandStatePhilox4_32_10_t**)calloc(
        1, num_buffers * sizeof(curandStatePhilox4_32_10_t*));
    for (int i = 0; i < num_buffers; i++) {
        cudaMalloc((void**)&pufferl->rng_states[i],
            agents_per_buf * sizeof(curandStatePhilox4_32_10_t));
        cudaMemset(pufferl->rng_states[i], 0,
            agents_per_buf * sizeof(curandStatePhilox4_32_10_t));
        PUF_LAUNCH(rng_init, grid_size(agents_per_buf), BLOCK_SIZE, 0, 0)(
            pufferl->rng_states[i], pufferl->seed + i, agents_per_buf);
    }

    // Post-create initialization
    cudaMemcpy(pufferl->act_sizes, act_sizes,
        num_action_heads*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(pufferl->losses, 0, NUM_LOSSES * sizeof(float));
    cudaMemcpy(pufferl->muon.lr, &hypers.lr, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(pufferl->muon.mb.data, 0, numel(pufferl->muon.mb.shape) * sizeof(float));

#ifdef PUFFER_NETHACK
    nethack_policy_init(ini);
#endif

    // CPU: per-step net graphs. GPU: full-horizon. Both captured on first use.
    if (hypers.cudagraphs && PUF_BACKEND != PUF_GPU) {
        int rollout_graph_slots = pufferl->async_num_slots;
        pufferl->rollout_graphs = (cudaGraphExec_t*)calloc(1,
            rollout_graph_slots*horizon*num_buffers*sizeof(cudaGraphExec_t));
    }
    pufferl->streams = (cudaStream_t*)calloc(1, num_buffers * sizeof(cudaStream_t));
    for (int i = 0; i < num_buffers; i++) {
        if (hypers.async) {
            assert(cudaStreamCreateWithFlags(
                &pufferl->streams[i], cudaStreamNonBlocking) == cudaSuccess);
#ifdef PUF_METAL
            // Gated steps (metal_gated_horizon) need host-wait-only streams,
            // a single buffer's too (nothing on the GPU waits for it in
            // async mode). PUF_METAL_LAZY_PROGRESS=0 keeps progress words
            // and so also disables gating.
            const char* lazy = getenv("PUF_METAL_LAZY_PROGRESS");
            bool lazy_on = !lazy || strcmp(lazy, "0") != 0;
            pufferl->metal_gate_streams = lazy_on && metal_gate_env() && puf_mtl::gate_available();
            if (lazy_on && (num_buffers > 1 || pufferl->metal_gate_streams)) {
                puf_mtl::stream_host_wait_only(pufferl->streams[i]);
            }
            puf_mtl::stream_wait_spin(pufferl->streams[i], metal_rollout_spin_us());
#endif
        } else {
            assert(cudaStreamCreate(&pufferl->streams[i]) == cudaSuccess);
        }
    }
    // CPU rollout workers register their streams in vec_thread_main.
    if (PUF_BACKEND == PUF_GPU) {
        cublas_init_stream(pufferl->streams[0]);
    }

    env_start(pufferl);
#ifdef PUF_METAL_MEMORY_TEST
    fprintf(stderr, "metal memory actual: created=%zu\n",
        puf_mtl::memory_info().process_live - memory_at_create.process_live);
#endif

    if (hypers.profile) {
        cudaDeviceSynchronize();
        cudaProfilerStart();
    }

    double now = wall_clock();
    pufferl->start_time = now;
    pufferl->last_log_time = now;
    pufferl->last_log_step = 0;

    dict_clear(&vec_kwargs);
    return pufferl;
}

// OS reclaims memory/CUDA context on exit. This is the intended design.
// All memory is allocated up front and static across training.
// Skip cudaDeviceSynchronize/ncclCommDestroy: captured NCCL graphs deadlock DP.
void close_pufferl(PuffeRL* p) {
    if (p->hypers.profile) {
        cudaProfilerStop();
    }
    nvmlShutdown();
    env_close(p->vec);
}

// Dashboard
static int puf_dashboard_tty = 0;
static int puf_dashboard_last_rows = 0;
static int puf_dashboard_last_cols = 0;
static int puf_dashboard_frame = 0;

#define PUF_DASH_W 80
#define PUF_DASH_BASE_ROWS 12
#define PUF_DASH_MAX_USER_ROWS 15

// Colors no-op when not a TTY.
#define PUF_A  (puf_dashboard_tty ? "\033[96m" : "")
#define PUF_W  (puf_dashboard_tty ? "\033[97m" : "")
#define PUF_G  (puf_dashboard_tty ? "\033[90m" : "")
#define PUF_R  (puf_dashboard_tty ? "\033[0m" : "")

static void dash_eol(void) {
    if (puf_dashboard_tty) {
        printf("\033[K");
    }
    putchar('\n');
}

static void dash_end(void) {
    if (puf_dashboard_tty) {
        printf("\033[J\033[?2026l");
    }
    fflush(stdout);
}

// Right-align into w cols (truncate if long). Gray unit letters
static void dash_cell(const char* s, int w) {
    int n = (int)strlen(s);
    if (n > w) {
        n = w;
    }
    int numeric = 0;
    for (int i = 0; i < n; i++) {
        numeric |= (s[i] >= '0' && s[i] <= '9');
    }
    printf("%s%*s", PUF_W, w - n, "");
    for (int i = 0; i < n; i++) {
        int unit = numeric && strchr("%KMBTGdhms", s[i]);
        printf("%s%c", unit ? PUF_G : PUF_W, s[i]);
    }
    printf("%s", PUF_R);
}

static void dash_rule(const char* left, const char* right) {
    printf("%s%s", PUF_W, left);
    for (int i = 0; i < PUF_DASH_W - 2; i++) printf("─");
    printf("%s%s", right, PUF_R);
    dash_eol();
}

static void dash_blank(void) {
    printf("%s│%*s│%s", PUF_W, PUF_DASH_W - 2, "", PUF_R);
    dash_eol();
}

static void dash_abbrev(char* out, size_t n, double val) {
    const char* suf[] = {"", "K", "M", "B", "T"};
    int i = 0;
    while (val >= 1000.0 && i < 4) {
        val /= 1000.0;
        i++;
    }
    snprintf(out, n, "%.1f%s", val, suf[i]);
}

// <1s = ms; sub-hour keeps ms; longer → d/h/m/s.
static void dash_duration(char* out, size_t n, double sec) {
    if (sec < 0) {
        sec = 0;
    }
    long ms = (long)(sec * 1000.0 + 0.5);
    if (ms < 1000) {
        snprintf(out, n, "%ldms", ms);
    } else if (ms < 60000) {
        snprintf(out, n, "%lds %03ldms", ms / 1000, ms % 1000);
    } else if (ms < 3600000) {
        snprintf(out, n, "%ldm %02lds %03ldms",
            ms / 60000, (ms / 1000) % 60, ms % 1000);
    } else {
        long s = ms / 1000;
        snprintf(out, n, "%ldd %ldh %ldm %lds",
            s / 86400, (s / 3600) % 24, (s / 60) % 60, s % 60);
    }
}

// Missing keys keep last train values when merged into last_log.
static double dash_num(Dict* log, const char* key, double fallback) {
    DictItem* it = dict_find(log, key);
    return it ? it->value : fallback;
}

void puf_dashboard_print(Ini* ini, PuffeRL* p, Dict* log, int epoch) {
    puf_dashboard_tty = isatty(STDOUT_FILENO);
    int term_rows = 1000, term_cols = PUF_DASH_W;
    if (puf_dashboard_tty) {
        struct winsize ws;
        if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) == 0) {
            if (ws.ws_row > 0) term_rows = ws.ws_row;
            if (ws.ws_col > 0) term_cols = ws.ws_col;
        }
    }

    const char* env_name = puf_ini_get_str(ini, "base", "env_name");
    double steps = dash_num(log, "agent_steps",
        (double)p->global_step * p->hypers.world_size);
    double sps = dash_num(log, "SPS", 0);
    long configured = puf_ini_get(ini, "train", "total_timesteps");
    long local_batch = (long)p->hypers.total_agents * p->hypers.horizon;
    long local_steps = configured / p->hypers.world_size;
    double target = (double)((local_steps / local_batch) * local_batch * p->hypers.world_size);
    double remain_sec = sps > 0 && target > steps ? (target - steps) / sps : 0;
    double rollout = dash_num(log, "perf/rollout", 0);
    double train_time = dash_num(log, "perf/train", 0);
    double perf_total = rollout + train_time;

    char params[32], steps_s[32], sps_s[32], uptime[64], remaining[64], epoch_s[32];
    dash_abbrev(params, sizeof(params), (double)numel(p->policies[0].master_weights.shape));
    dash_abbrev(steps_s, sizeof(steps_s), steps);
    dash_abbrev(sps_s, sizeof(sps_s), sps);
    double up = dash_num(log, "uptime", wall_clock() - p->start_time);
    dash_duration(uptime, sizeof(uptime), up);
    dash_duration(remaining, sizeof(remaining), remain_sec);
    snprintf(epoch_s, sizeof(epoch_s), "%d", epoch);

    if (puf_dashboard_tty) {
        printf("\033[?2026h");
        if (term_rows != puf_dashboard_last_rows || term_cols != puf_dashboard_last_cols) {
            printf("\033[H\033[J");
        } else {
            printf("\033[H");
        }
        puf_dashboard_last_rows = term_rows;
        puf_dashboard_last_cols = term_cols;
    }
    // Tiny terminal: one-line compact summary.
    if (puf_dashboard_tty && (term_cols < PUF_DASH_W || term_rows <= PUF_DASH_BASE_ROWS)) {
        char compact[512];
        snprintf(compact, sizeof(compact),
            "PufferLib 5.0  env=%s  steps=%s  SPS=%s  score=%.3f  epoch=%s  to_go=%s",
            env_name, steps_s, sps_s, dash_num(log, "env/score", 0), epoch_s, remaining);
        printf("%.*s", term_cols > 1 ? term_cols - 1 : term_cols, compact);
        if (term_rows > 1) {
            dash_eol();
        } else if (puf_dashboard_tty) {
            printf("\033[K");
        }
        dash_end();
        return;
    }

    char gpu[16], vram[32], ram[16];
    snprintf(gpu, sizeof(gpu), "%3.0f%%", dict_get(log, "util/gpu_percent"));
    snprintf(vram, sizeof(vram), "%.1f/%.0fG",
        dict_get(log, "util/vram_used_gb"),
        dict_get(log, "util/vram_total_gb"));
    snprintf(ram, sizeof(ram), "%.1fG", dict_get(log, "util/cpu_mem_gb"));

    int fish_span = 18;
    int fish_pos = (fish_span - 3) - (puf_dashboard_frame++ % (fish_span - 2));

    dash_rule("╭", "╮");
    printf("%s│%s %sPufferLib %s5.0%s%*s%s🐡%s%*s%sGPU%s:%s ",
        PUF_W, PUF_R, PUF_A, PUF_W, PUF_R, fish_pos, "", PUF_A, PUF_R,
        fish_span - 2 - fish_pos, "", PUF_A, PUF_G, PUF_R);
    dash_cell(gpu, 4);
    printf("   %sVRAM%s:%s", PUF_A, PUF_G, PUF_R);
    dash_cell(vram, 10);
    printf("    %sRAM%s:%s", PUF_A, PUF_G, PUF_R);
    dash_cell(ram, 6);
    printf("     %s│%s", PUF_W, PUF_R);
    dash_eol();
    dash_blank();

    // Stats | Perf | Losses — fixed widths sum to 80 with borders/spacing.
    struct {
        const char* a;
        const char* av;
        const char* b;
        const char* perf_key;  // null → use perf_sec
        double perf_sec;
        const char* c;
        const char* loss_key;
    } rows[] = {
        {"Env", env_name, "Evaluate", NULL, rollout, "Losses", "loss/total"},
        {"Params", params, "  Model", "perf/eval_model", 0, "policy", "loss/policy"},
        {"Steps", steps_s, "  Env", "perf/eval_env", 0, "value", "loss/value"},
        {"SPS", sps_s, "  Copy", "perf/eval_copy", 0, "entropy", "loss/entropy"},
        {"Epoch", epoch_s, "Train", NULL, train_time, "old_kl", "loss/old_kl"},
        {"Uptime", uptime, "  Model", "perf/train_model", 0, "kl", "loss/kl"},
        {"To go", remaining, "  Misc", "perf/train_misc", 0, "clipfrac", "loss/clipfrac"},
    };
    for (int i = 0; i < (int)(sizeof(rows) / sizeof(rows[0])); i++) {
        char bt[64], bp[16], cv[32];
        double sec = rows[i].perf_key
            ? dash_num(log, rows[i].perf_key, 0) : rows[i].perf_sec;
        dash_duration(bt, sizeof(bt), sec);
        snprintf(bp, sizeof(bp), "%d%%",
            perf_total > 0 ? (int)(100.0 * sec / perf_total) : 0);
        DictItem* lit = dict_find(log, rows[i].loss_key);
        if (lit) {
            snprintf(cv, sizeof(cv), "%.3f", lit->value);
        } else {
            cv[0] = 0;
        }
        printf("%s│%s %s%-9.9s%s ", PUF_W, PUF_R, PUF_A, rows[i].a, PUF_R);
        dash_cell(rows[i].av, 13);
        printf("    %s%-12.12s%s ", PUF_A, rows[i].b, PUF_R);
        dash_cell(bt, 6);
        putchar(' ');
        dash_cell(bp, 4);
        printf("    %s%-10.10s%s ", PUF_A, rows[i].c, PUF_R);
        dash_cell(cv, 7);
        printf("    %s│%s", PUF_W, PUF_R);
        dash_eol();
    }
    dash_blank();

    int user_rows = PUF_DASH_MAX_USER_ROWS;
    if (puf_dashboard_tty) {
        user_rows = term_rows - PUF_DASH_BASE_ROWS - 1;
        if (user_rows < 0) user_rows = 0;
        if (user_rows > PUF_DASH_MAX_USER_ROWS) user_rows = PUF_DASH_MAX_USER_ROWS;
    }

    char pending_key[128];
    double pending_val = 0;
    int pending = 0, n = 0, max_items = 2 * user_rows;
    for (int i = 0; i < log->size && n < max_items; i++) {
        const char* key = log->items[i].key;
        if (strncmp(key, "env/", 4) != 0 || strcmp(key, "env/n") == 0) continue;
        const char* sk = key + 4;
        if (!pending) {
            snprintf(pending_key, sizeof(pending_key), "%s", sk);
            pending_val = log->items[i].value;
            pending = 1;
        } else {
            char ls[32], rs[32];
            snprintf(ls, sizeof(ls), "%.3f", pending_val);
            snprintf(rs, sizeof(rs), "%.3f", log->items[i].value);
            printf("%s│%s %s%-25.25s%s %s%9.9s%s   %s%-25.25s%s %s%9.9s%s    %s│%s",
                PUF_W, PUF_R, PUF_A, pending_key, PUF_R, PUF_W, ls, PUF_R,
                PUF_A, sk, PUF_R, PUF_W, rs, PUF_R, PUF_W, PUF_R);
            dash_eol();
            pending = 0;
        }
        n++;
    }
    if (pending) {
        char ls[32];
        snprintf(ls, sizeof(ls), "%.3f", pending_val);
        printf("%s│%s %s%-25.25s%s %s%9.9s%s   %-25.25s %9.9s    %s│%s",
            PUF_W, PUF_R, PUF_A, pending_key,
            PUF_R, PUF_W, ls, PUF_R, "", "", PUF_W, PUF_R);
        dash_eol();
    }
    dash_rule("╰", "╯");
    dash_end();
}

void log_util(PuffeRL* p, Dict* out) {
    nvmlUtilization_t util;
    nvmlDeviceGetUtilizationRates(p->nvml_device, &util);
    dict_set(out, "util/gpu_percent", (double)util.gpu);

    size_t cuda_free;
    size_t cuda_total;
    cudaMemGetInfo(&cuda_free, &cuda_total);
    dict_set(out, "util/vram_used_gb",
        (double)(cuda_total - cuda_free) / (1024.0 * 1024.0 * 1024.0));
    dict_set(out, "util/vram_total_gb",
        (double)cuda_total / (1024.0 * 1024.0 * 1024.0));

    long rss_kb = 0;
    FILE* status = fopen("/proc/self/status", "r");
    if (status) {
        char line[256];
        while (fgets(line, sizeof(line), status)) {
            if (sscanf(line, "VmRSS: %ld", &rss_kb) == 1) {
                break;
            }
        }
        fclose(status);
    }
    dict_set(out, "util/cpu_mem_gb", (double)rss_kb / (1024.0 * 1024.0));
}

void trainer_eval_log(PuffeRL* p, Dict* out) {
    p->last_log_time = wall_clock();
    p->last_log_step = p->global_step;
    log_util(p, out);
    vec_log(p->vec, out, 0);
}

// One dict copy per train log (+ optional final snapshot). Capacity fixed at
// train start: ≤ train_epochs + 1 final append.
typedef struct {
    Dict* items;
    int size;
    int capacity;
} PufLogHistory;

static void puf_log_history_init(PufLogHistory* h, int capacity) {
    h->size = 0;
    h->capacity = capacity;
    h->items = (Dict*)calloc(capacity, sizeof(Dict));
}

static void puf_log_history_add(PufLogHistory* h, Dict* log) {
    assert(h->size < h->capacity);
    dict_copy(&h->items[h->size], log);
    h->size++;
}

// Refresh trainer fields even when no episodes completed. Only absent fields
// (normally env/* episode statistics) carry forward from the previous log.
static void puf_log_snapshot_update(Dict* snapshot, Dict* log) {
    for (int i = 0; i < log->size; i++) {
        dict_set(snapshot, log->items[i].key, log->items[i].value);
    }
}

// points-1 equal-width bins [j*w, (j+1)*w), w=final_steps/(points-1),
// followed by the exact final snapshot. Empty bins carry the last non-empty
// bin's mean; leading empty bins use the first observed value of the key.
// Absent keys do not contribute samples. History has ordered agent_steps.
// points==1 returns only the final value (also valid for zero final_steps).
static void log_history_bin_mean(PufLogHistory* h, const char* key,
        int points, double* out) {
    assert(h->size > 0 && points >= 1);
    if (points == 1) {
        out[0] = dict_get(&h->items[h->size - 1], key);
        return;
    }
    double final_steps = dict_get(&h->items[h->size - 1], "agent_steps");
    int out_idx = 0;
    int bin_n = 0;
    double bin_sum = 0;
    double fallback = dict_get(&h->items[h->size - 1], key);
    for (int i = 0; i < h->size; i++) {
        DictItem* item = dict_find(&h->items[i], key);
        if (item) {
            fallback = item->value;
            break;
        }
    }
    double next_bin = final_steps / (points - 1);
    for (int i = 0; i < h->size; i++) {
        Dict* log = &h->items[i];
        DictItem* item = dict_find(log, key);
        if (!item) continue;
        double steps = dict_get(log, "agent_steps");
        // Close ALL preceding bins before assigning this sample. In particular,
        // a late sample must never be pulled backward into an earlier empty bin.
        while (out_idx < points - 1 && steps >= next_bin) {
            fallback = bin_n ? bin_sum / bin_n : fallback;
            out[out_idx++] = fallback;
            bin_n = 0;
            bin_sum = 0;
            next_bin = final_steps * (out_idx + 1) / (points - 1);
        }
        bin_sum += item->value;
        bin_n++;
    }
    out[points - 1] = dict_get(&h->items[h->size - 1], key);
    while (out_idx < points - 1) {
        out[out_idx++] = fallback;
    }
}

double rollout_start(PuffeRL* p, int slot) {
#ifdef PUFFER_NETHACK
    nethack_policy_on_rollout(p->global_step, p->hypers.total_timesteps);
#endif
    p->write_slot = slot;
#ifdef PUF_METAL
    static const bool obs_bt = [] {
#ifdef PUF_GOLDEN_DUMP
        // Golden capture externalizes the original time-major rollout tensor.
        return false;
#else
        const char* e = getenv("PUF_METAL_OBS_BT");
        return !e || strcmp(e, "0") != 0;
#endif
    }();
    // Preflight every block before selecting a slot's layout. All eligibility
    // inputs (architecture, shapes, policy layout and toggles) stay constant
    // within the horizon; t=0 also checks the optional state snapshot.
    bool bt = obs_bt;
    RolloutBuf view = rollout_time_view(&p->rollouts,
        slot * p->hypers.horizon, p->hypers.horizon);
    int block = p->hypers.total_agents / p->hypers.num_buffers;
    for (int b = 0; b < p->hypers.num_buffers && bt; ++b) {
        bt = metal_rollout_step(p, b, 0, view, &p->env, b * block, block,
            slot, p->streams[b], true, true);
    }
    p->metal_obs_bt[slot] = bt;
#endif
    if (p->hypers.async) {
        Prec* param = &p->policies[0].param;
        int64_t n = numel(param->shape);
        cudaMemcpyAsync(p->actor_param.data, param->data,
            n * sizeof(precision_t), cudaMemcpyDeviceToDevice, p->default_stream);
#ifdef PUF_METAL
        metal_rollout_pack(p, slot, view);  // behind the copy, same sync
#endif
        cudaStreamSynchronize(p->default_stream);
    }
#ifdef PUF_METAL
    else {
        metal_rollout_pack(p, slot, view);
        cudaStreamSynchronize(p->default_stream);
    }
    // Gated steps: one packed dispatch per step (grouped or single buffer)
    // reading the shared env IO, nothing else on the stream.
    bool gate = p->metal_gate_streams && p->hypers.async && metal_shared_env()
        && p->metal_packed_split && PUF_BACKEND != PUF_GPU;
    for (int b = 0; b < p->vec->buffers && gate; b++) {
        gate = metal_rollout_step(p, b, 0, view, &p->env, b * block, block,
            slot, p->streams[b], p->metal_obs_bt[slot], true);
    }
    const char* group_env = getenv("PUF_METAL_ROLLOUT_GROUP");
    // Two original buffers per dispatch reduce queue/wake round trips while
    // keeping each buffer's state, RNG and output layout. Clean n=6: Flappy
    // +5.8/10.0%, Cartpole +8.1/11.6% (BF16/FP32) over ungrouped packing.
    // Ungated, four buffers increase wake/env overhead; gated, they can win
    // (metal_dw_policy decides metal_group_pref). =1 restores ungrouped
    // steps, =2 pairs, =4 four-buffer groups.
    int group = group_env ? atoi(group_env)
        : !p->hypers.async ? 1 : gate && p->metal_group_pref == 4 ? 4 : 2;
    if (!group_env && group == 4 && p->vec->buffers % 4 != 0) group = 2;
    if ((group == 2 || group == 4) && p->metal_packed_split
            && p->vec->buffers % group == 0 && PUF_BACKEND != PUF_GPU) {
        bool eligible = true;
        for (int b = 0; b < p->vec->buffers && eligible; b++) {
            eligible = metal_rollout_step(p, b, 0, view, &p->env, b * block, block,
                slot, p->streams[b], p->metal_obs_bt[slot], true);
        }
        if (!eligible) group = 1;
    } else {
        group = 1;
    }
    __atomic_store_n(&p->metal_rollout_group, group, __ATOMIC_RELAXED);
    __atomic_store_n(&p->metal_rollout_gate, gate ? 1 : 0, __ATOMIC_RELAXED);
#endif
    if (p->hypers.reset_every_horizon) {
        for (int b = 0; b < p->num_policies; b++) {
            for (int i = 0; i < p->hypers.num_buffers; i++) {
                Prec* st = &p->policies[b].buffer_states[i];
                cudaMemsetAsync(st->data, 0, numel(st->shape) * sizeof(precision_t),
                    p->default_stream);
            }
        }
        cudaStreamSynchronize(p->default_stream);
    }

    double t0 = wall_clock();
    rollout_start(p);
    return t0;
}
void rollouts(PuffeRL* p) {
    double t0 = rollout_start(p, 0);
    rollout_finish(p, t0);
    p->global_step += p->hypers.horizon * p->hypers.total_agents;
}

typedef struct {
    float score;
    float perf;
    float draw;
    int games;
} EvalResult;

#define TRAIN_RESULT_MAX_POINTS 64
typedef struct {
    float score;
    float cost;
    float steps;
    int points;
    float scores[TRAIN_RESULT_MAX_POINTS];
    float costs[TRAIN_RESULT_MAX_POINTS];
    float step_points[TRAIN_RESULT_MAX_POINTS];
} TrainResult;

#define EVAL_RENDER 0
#define EVAL_SCORE 1
#define EVAL_MATCH 2

#define SELFPLAY_MAX_HIST 8
#define SELFPLAY_MAX_LADDER 16
#define SELFPLAY_PATH_MAX 4096
// One historical opponent ↔ policies[policy_idx] (env tag == policy_idx).
typedef struct {
    int policy_idx;
    long opp_started_step;
} SelfplayHist;

typedef struct {
    int num_hist;
    int max_size;
    long opp_timeout_steps;
    unsigned int rng;
    char (*pool)[SELFPLAY_PATH_MAX];
    int pool_size;
    SelfplayHist hist[SELFPLAY_MAX_HIST];
} Selfplay;

void selfplay_add_checkpoint(Selfplay* sp, const char* path) {
    for (int i = 0; i < sp->pool_size; i++) {
        if (strcmp(sp->pool[i], path) == 0) return;
    }
    if (sp->pool_size == sp->max_size) {
        memmove(sp->pool, sp->pool + 1, (sp->max_size - 1) * sizeof(*sp->pool));
        sp->pool_size--;
    }
    snprintf(sp->pool[sp->pool_size++], sizeof(sp->pool[0]), "%s", path);
}

const char* selfplay_sample(Selfplay* sp) {
    int idx = (int)(rand_r(&sp->rng) % (unsigned int)sp->pool_size);
    return sp->pool[idx];
}

typedef struct {
    char section[64];
    char key[64];
} SweepParam;

extern char** environ;

typedef struct {
    int run;
    int random;
    int gp_obs;
    int pareto;
    int fd;
    pid_t pid;
    float* sample;
    TrainResult result;
} SweepJob;

void run_sweep(Ini* ini, const char* exe_path) {
    // Build SweepSpace + param map from [sweep.<section>.<key>] sections.
    const char* goal = puf_ini_get_str(ini, "sweep", "goal");
    assert((strcmp(goal, "maximize") == 0 || strcmp(goal, "minimize") == 0)
        && "sweep.goal must be maximize or minimize");
    int direction = strcmp(goal, "minimize") == 0 ? -1 : 1;

    SweepParam* params = (SweepParam*)calloc(ini->num_sections, sizeof(SweepParam));
    SweepSpace* space = (SweepSpace*)calloc(1, sizeof(SweepSpace));
    space->spaces = (Space*)calloc((size_t)ini->num_sections, sizeof(Space));
    space->cost_idx = -1;
    space->optimize_direction = direction;
    int n_params = 0;
    for (int i = 0; i < ini->num_sections; i++) {
        Dict* dict = &ini->sections[i];
        if (strncmp(dict->name, "sweep.", 6) != 0) {
            continue;
        }
        const char* path = dict->name + 6;
        const char* dot = strrchr(path, '.');
        assert(dot && dot != path && dot[1]
            && "expected section [sweep.<section>.<key>]");
        snprintf(params[n_params].section, sizeof(params[n_params].section),
            "%.*s", (int)(dot - path), path);
        snprintf(params[n_params].key, sizeof(params[n_params].key), "%s", dot + 1);

        const char* dist = dict_get_str(dict, "distribution");
        SpaceType type = SPACE_LINEAR;
        int is_integer = 0;
        if (strcmp(dist, "uniform") == 0) {
            type = SPACE_LINEAR;
        } else if (strcmp(dist, "int_uniform") == 0) {
            type = SPACE_LINEAR;
            is_integer = 1;
        } else if (strcmp(dist, "uniform_pow2") == 0) {
            type = SPACE_POW2;
            is_integer = 1;
        } else if (strcmp(dist, "log_normal") == 0) {
            type = SPACE_LOG;
        } else if (strcmp(dist, "logit_normal") == 0) {
            type = SPACE_LOGIT;
        } else {
            assert(0 && "invalid sweep distribution (use uniform/int_uniform/"
                "uniform_pow2/log_normal/logit_normal)");
        }

        float min_v = dict_get(dict, "min");
        float max_v = dict_get(dict, "max");
        const char* scale_s = dict_get_str(dict, "scale");
        float scale;
        if (strcmp(scale_s, "auto") == 0) {
            scale = 0.5f;
        } else if (strcmp(scale_s, "time") == 0) {
            assert(min_v > 0 && max_v > 0
                && "scale=time requires positive min/max");
            scale = 1.0f / (log2f(max_v) - log2f(min_v));
        } else {
            scale = dict_get(dict, "scale");
        }

        space->spaces[n_params] = (Space){
            .type = type, .min = min_v, .max = max_v, .scale = scale, .is_integer = is_integer};
        if (strcmp(dict->name, "sweep.train.total_timesteps") == 0) {
            space->cost_idx = n_params;
        }
        n_params++;
    }
    space->num = n_params;

    int max_runs = puf_ini_get(ini, "sweep", "max_runs");
    int downsample = puf_ini_get(ini, "sweep", "downsample");
    int prune_pareto = puf_ini_get(ini, "sweep", "prune_pareto");
    const char* metric_dist = puf_ini_get_str(ini, "sweep", "metric_distribution");
    assert((strcmp(metric_dist, "linear") == 0 || strcmp(metric_dist, "logit") == 0)
        && "sweep.metric_distribution must be linear or logit");
    int use_logit = strcmp(metric_dist, "logit") == 0;
    float max_cost = puf_ini_get(ini, "sweep", "max_suggestion_cost");
    float early_stop_quantile = puf_ini_get(ini, "sweep", "early_stop_quantile");
    assert(max_runs >= 1 && "sweep.max_runs must be >= 1");
    assert(downsample >= 1 && downsample <= TRAIN_RESULT_MAX_POINTS
        && "sweep.downsample must be in [1, TRAIN_RESULT_MAX_POINTS]");
    int success_cap = (int)fmaxf((float)(max_runs * downsample * 2), 8192.0f);

    // GPU packing: each trial is a full train (launch_train) that may itself
    // use train.gpus for NCCL DP. Concurrent trials = sweep_gpus / train_gpus
    // on disjoint blocks [0,W), [W,2W), ...  e.g. 8 GPUs, train.gpus=2 → 4 trials.
    int total_gpus = 0;
    assert(cudaGetDeviceCount(&total_gpus) == cudaSuccess && total_gpus >= 1);
    int train_gpus = puf_ini_get(ini, "train", "gpus");
    int sweep_gpus = puf_ini_get(ini, "sweep", "gpus");
    if (sweep_gpus == -1) {
        sweep_gpus = total_gpus;
    }
    assert(train_gpus >= 1 && (float)sweep_gpus == fminf(
        fmaxf((float)sweep_gpus, (float)train_gpus), (float)total_gpus));

    ProteinSweep* protein = protein_sweep_create((ProteinSweep){
        .space = space,
        .num_random_samples = 10,
        .suggestions_per_pareto = 256,
        .gp_training_iter = 50,
        .gp_learning_rate = 0.001f,
        .optimizer_reset_frequency = 50,
        .gp_max_obs = 750,
        .infer_batch_size = 4096,
        .use_success_prob = downsample == 1,
        .prune_pareto = prune_pareto,
        .use_logit = use_logit,
        .global_search_scale = 1.0f,
        .max_suggestion_cost = max_cost,
        .expansion_rate = 0.1f,
        .cost_random_suggestion = -0.8f,
        .early_stop_quantile = early_stop_quantile,
        .success_cap = success_cap,
        .failure_cap = 1024,
        .top_k = 5,
        .rng_seed = 73ULL,
    });

    int parallel = sweep_gpus / train_gpus;
    // Row 0 = staging for suggest; rows 1..parallel = per-slot copies for observe.
    float* samples = (float*)calloc((parallel + 1) * space->num, sizeof(float));
    SweepJob* jobs = (SweepJob*)calloc(parallel, sizeof(SweepJob));
    int failed_workers = 0;
    int next_run_id = 0;
    int completed = 0;
    int active = 0;
    // Free-list: slot i owns GPUs [i*train_gpus, (i+1)*train_gpus). Refill as
    // soon as any trial exits (waitpid -1), so finished GPUs are never idle
    // while more runs remain.
    while (completed < max_runs || active > 0) {
        for (int i = 0; i < parallel; i++) {
            if (jobs[i].pid || completed + active >= max_runs) {
                continue;
            }

            ProteinSweepInfo info = {0};
            if (!next_run_id) {
                for (int j = 0; j < space->num; j++) {
                    float val = puf_ini_get(ini, params[j].section, params[j].key);
                    float norm = space_normalize(&space->spaces[j], val);
                    if (!(isfinite(norm) && norm >= -1.0f && norm <= 1.0f)) {
                        fprintf(stderr,
                            "default sweep value outside its sweep range: %s.%s\n",
                            params[j].section, params[j].key);
                        exit(1);
                    }
                    samples[j] = norm;
                }
            } else {
                // Invalid jobs die and produce "fail" obs
                info = protein_sweep_suggest(protein, samples, NAN);
            }

            SweepJob job = {
                .run = next_run_id++,
                .random = info.is_random,
                .gp_obs = info.n_gp_obs,
                .pareto = info.n_pareto,
                .sample = samples + (size_t)(i + 1) * space->num,
            };
            memcpy(job.sample, samples, space->num * sizeof(float));

            for (int p = 0; p < space->num; p++) {
                float val = space_unnormalize(&space->spaces[p], samples[p]);
                char buf[64];
                snprintf(buf, sizeof(buf), "%.9g", val);
                char key[256];
                snprintf(key, sizeof(key), "%s.%s",
                    params[p].section, params[p].key);
                puf_ini_put(ini, key, buf);
            }
            char run_id[128];
            snprintf(run_id, sizeof(run_id), "sweep_%ld_%04d",
                (long)(1000.0 * wall_clock()), job.run);
            puf_ini_put(ini, "base.run_id", run_id);

            // Spawn train: result pipe + full ini as section.key=value argv.
            // Child may NCCL-fork again if train.gpus > 1.
            int pipefd[2];
            assert(pipe(pipefd) == 0);
            char buf[64];
            snprintf(buf, sizeof(buf), "%d", i * train_gpus);
            puf_ini_put(ini, "base.gpu_offset", buf);
            snprintf(buf, sizeof(buf), "%d", pipefd[1]);
            puf_ini_put(ini, "base.result_fd", buf);

            int nkeys = 0;
            for (int s = 0; s < ini->num_sections; s++) {
                nkeys += ini->sections[s].size;
            }
            char** argv = (char**)calloc(nkeys + 3, sizeof(char*));
            argv[0] = (char*)exe_path;
            argv[1] = (char*)"train";
            int argc = 2;
            char full_key[PUF_DICT_MAX_KEY * 2];
            char val[128];
            for (int s = 0; s < ini->num_sections; s++) {
                Dict* dict = &ini->sections[s];
                for (int k = 0; k < dict->size; k++) {
                    DictItem* item = &dict->items[k];
                    const char* src = item->str;
                    if (!src) {
                        snprintf(val, sizeof(val), "%.17g", item->value);
                        src = val;
                    }
                    snprintf(full_key, sizeof(full_key), "%s.%s",
                        dict->name, item->key);
                    size_t n = strlen(full_key) + strlen(src) + 4;
                    argv[argc] = (char*)malloc(n);
                    snprintf(argv[argc], n, "--%s=%s", full_key, src);
                    argc++;
                }
            }
            argv[argc] = NULL;

            posix_spawn_file_actions_t actions;
            posix_spawn_file_actions_init(&actions);
            posix_spawn_file_actions_addclose(&actions, pipefd[0]);
            posix_spawn_file_actions_addopen(
                &actions, STDOUT_FILENO, "/dev/null", O_WRONLY, 0);
            assert(posix_spawnp(&job.pid, exe_path, &actions, NULL, argv, environ) == 0
                && "posix_spawn train failed");
            posix_spawn_file_actions_destroy(&actions);
            for (int a = 2; a < argc; a++) {
                free(argv[a]);
            }
            free(argv);
            close(pipefd[1]);
            job.fd = pipefd[0];
            jobs[i] = job;
            active++;
        }
        if (!active) {
            break;
        }

        // Reap one finished worker and feed protein (curve points or a fail obs).
        int status = 0;
        pid_t done = waitpid(-1, &status, 0);
        assert(done > 0 && "sweep waitpid failed");
        SweepJob* job = NULL;
        for (int j = 0; j < parallel; j++) {
            if (jobs[j].pid == done) {
                job = &jobs[j];
                break;
            }
        }
        assert(job && "waitpid reaped unknown child");
        int nread = (int)read(job->fd, &job->result, sizeof(job->result));
        close(job->fd);
        job->pid = 0;
        active--;

        if (nread != (int)sizeof(job->result)
                || !WIFEXITED(status) || WEXITSTATUS(status) != 0) {
            fprintf(stderr, "sweep worker run=%d failed; marking sample bad\n",
                job->run);
            protein_sweep_observe(protein, job->sample, NAN, max_cost, 1);
            assert(++failed_workers <= 1000 && "too many failed sweep workers");
            continue;
        }
        // points[]: learning-curve downsample, or 1 final score (e.g. selfplay).
        for (int pi = 0; pi < job->result.points; pi++) {
            protein_sweep_observe(protein, job->sample,
                job->result.scores[pi], job->result.costs[pi], 0);
        }
        printf("sweep run=%d score=%.4f cost=%.2f steps=%.0f random=%d gp_obs=%d pareto=%d\n",
            job->run, job->result.score, job->result.cost, job->result.steps,
            job->random, job->gp_obs, job->pareto);
        completed++;
    }
}

// Sweep objective: bare names → env/<name>; keys with '/' used as-is.
static void sweep_metric_key(Ini* ini, char* buf, size_t n) {
    const char* metric = puf_ini_get_str(ini, "sweep", "metric");
    snprintf(buf, n, "%s%s", strchr(metric, '/') ? "" : "env/", metric);
}

// board!=NULL: merge env/* into train last_log (uptime + util/* stay frozen).
static EvalResult eval_loop(Ini* ini, PuffeRL* p, int mode, int verbose,
        int render, long eval_episodes, Dict* board, int epoch) {
    int match = mode == EVAL_MATCH;
    char metric_key[128];
    sweep_metric_key(ini, metric_key, sizeof(metric_key));
    EvalResult result = {0};
    if (!render) {
        Dict wipe = {0};
        vec_log(p->vec, &wipe, 1);
        dict_clear(&wipe);
    }
    double last_dash = 0;
    while (true) {
        if (render) {
            puf_render(p->vec->envs);
            if (WindowShouldClose()) {
                return result;
            }
        }
        rollouts(p);
        Dict el = {0};
        trainer_eval_log(p, &el);
        if (board) {
            for (int i = 0; i < el.size; i++) {
                const char* k = el.items[i].key;
                if (strncmp(k, "util/", 5) == 0) {
                    continue;
                }
                dict_set(board, k, el.items[i].value);
            }
        }
        Dict* show = board ? board : &el;
        double now = wall_clock();
        if (!render && verbose && now - last_dash >= 0.6) {
            puf_dashboard_print(ini, p, show, board ? epoch : 0);
            last_dash = now;
        }
        if (render) {
            dict_clear(&el);
            continue;
        }
        long n = dict_get(&el, "env/n");
        if (n < eval_episodes) {
            dict_clear(&el);
            continue;
        }
        if (verbose) {
            puf_dashboard_print(ini, p, show, board ? epoch : 0);
        }
        DictItem* m = dict_find(&el, metric_key);
        result.score = match ? dict_get(&el, "env/policy_0_score")
            : (m ? m->value : dict_get(&el, "env/score"));
        result.perf = dict_get(&el, "env/perf");
        if (match) {
            result.draw = dict_get(&el, "env/draw_rate");
        }
        result.games = (int)n;
        dict_clear(&el);
        return result;
    }
}

static PuffeRL* eval_make(Ini* ini, TrainContext* ctx, int mode, int render) {
    int match = mode == EVAL_MATCH;
    long eval_agents = puf_ini_get(ini, "base", "eval_agents");
    if (render) {
        // One env on screen. Some envs (drone num_drones, boids num_boids) do
        // not use env.num_agents; default.ini's 1 then under-sizes the buffer
        // and env_setup's offset==apb assert fires.
        Env probe = {0};
        puf_init(&probe, puf_ini_section(ini, "env", 0));
        int n = probe.num_agents;
        puf_close(&probe);
        assert(n > 0 && "render eval: env reported 0 agents");
        char nb[32];
        snprintf(nb, sizeof(nb), "%d", n);
        puf_ini_put(ini, "vec.total_agents", nb);
        puf_ini_put(ini, "vec.num_buffers", "1");
        puf_ini_put(ini, "vec.num_threads", "1");
        puf_ini_put(ini, "train.verb_eps", "1");
    } else if (eval_agents != -1) {
        char buf[64];
        snprintf(buf, sizeof(buf), "%ld", eval_agents);
        puf_ini_put(ini, "vec.total_agents", buf);
        puf_ini_put(ini, "train.verb_eps", "0");
    }
    if (match) {
        int h = puf_ini_get(ini, "policy", "hidden_size");
        int L = puf_ini_get(ini, "policy", "num_layers");
        char hb[32], lb[32];
        snprintf(hb, sizeof(hb), "%d", h);
        snprintf(lb, sizeof(lb), "%d", L);
        puf_ini_put(ini, "vec.num_policies", "2");
        puf_ini_put(ini, "vec.hist_policy_percent", "1");
        puf_ini_put(ini, "vec.hist_policy_hidden_size", hb);
        puf_ini_put(ini, "vec.hist_policy_num_layers", lb);
        puf_ini_put(ini, "selfplay.enabled", "0");
    }
    puf_ini_put(ini, "base.reset_every_horizon", "0");
    if (render) {
        puf_ini_put(ini, "train.horizon", "1");
    }
    PuffeRL* p = create_pufferl(ini, ctx);
    if (match) {
        char a_buf[4096], b_buf[4096];
        const char* a = puf_checkpoint_path_key(ini, "load_model_path", a_buf, sizeof(a_buf));
        const char* b = puf_checkpoint_path_key(ini,
            "load_enemy_model_path", b_buf, sizeof(b_buf));
        assert(a && b && "match requires load_model_path and load_enemy_model_path");
        pufferl_load_policy(p, 0, a);
        pufferl_load_policy(p, 1, b);
    } else {
        char buf[4096];
        const char* path = puf_checkpoint_path_key(ini, "load_model_path", buf, sizeof(buf));
        if (path) {
            pufferl_load_policy(p, 0, path);
        }
    }
    return p;
}

EvalResult run_eval(Ini* ini, TrainContext* ctx, int mode, int verbose,
        int render) {
    long n = puf_ini_get(ini, "base", "eval_episodes");
    assert((render || n > 0) && "eval requires positive base.eval_episodes");
    PuffeRL* p = eval_make(ini, ctx, mode, render);
    EvalResult r = eval_loop(ini, p, mode, verbose, render, n, NULL, 0);
    if (!render) {
        long nparams = 0;
        if (p && p->policies) {
            nparams = numel(p->policies[0].master_weights.shape);
        }
#ifndef PUF_EVAL_BACKEND
#ifdef PUF_METAL
#define PUF_EVAL_BACKEND METAL
#else
#define PUF_EVAL_BACKEND CUDA
#endif
#endif
#define PUF_EVAL_STRINGIFY_INNER(x) #x
#define PUF_EVAL_STRINGIFY(x) PUF_EVAL_STRINGIFY_INNER(x)
        printf(PUF_EVAL_STRINGIFY(PUF_EVAL_BACKEND)
            "_EVAL env=%s score=%.6f perf=%.6f games=%d params=%ld\n",
            puf_ini_get_str(ini, "base", "env_name"),
            r.score, r.perf, r.games, nparams);
        fflush(stdout);
    }
    close_pufferl(p);
    return r;
}

#ifdef PUF_METAL
// Metal runtime counters per train epoch: logged as metal/* (ini metrics) at
// every log interval and summarized once after the epoch loop
// ("metal_summary ..." line, parsed by tests/metal/bench.sh).
typedef struct {
    puf_mtl::Stats all, train, roll;
    double wall;
    double cpu;  // process user + system CPU seconds (getrusage)
    long epoch, step;
    double train_ms, rollout_ms;  // profile.accum sums (stamps / events)
} MetalMark;

static MetalMark metal_mark(PuffeRL* p) {
    MetalMark m;
    m.all = puf_mtl::stats();
    m.train = puf_mtl::stream_stats(p->train_stream);
    m.roll = p->streams ? puf_mtl::stream_stats(p->streams[0]) : puf_mtl::Stats{};
    m.wall = wall_clock();
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    m.cpu = ru.ru_utime.tv_sec + ru.ru_stime.tv_sec
        + (ru.ru_utime.tv_usec + ru.ru_stime.tv_usec) * 1e-6;
    m.epoch = p->epoch;
    m.step = p->global_step;
    m.train_ms = 0;
    m.rollout_ms = 0;
    return m;
}

// Writes per-epoch deltas between marks a and b into out.
static void metal_log(const MetalMark* a, const MetalMark* b, Dict* out) {
    double ne = (double)(b->epoch - a->epoch);
    if (ne <= 0) return;
    dict_set(out, "metal/dispatches", (b->all.dispatches - a->all.dispatches) / ne);
    dict_set(out, "metal/commits", (b->all.commits - a->all.commits) / ne);
    dict_set(out, "metal/cpu_waits", (b->all.host_syncs - a->all.host_syncs) / ne);
    dict_set(out, "metal/gated_commits", (b->all.gated - a->all.gated) / ne);
    dict_set(out, "metal/spin_hits", (b->all.spin_hit - a->all.spin_hit) / ne);
    dict_set(out, "metal/spin_misses", (b->all.spin_miss - a->all.spin_miss) / ne);
    dict_set(out, "metal/strict_feedback_joins",
        (b->all.strict_feedback_joins - a->all.strict_feedback_joins) / ne);
    dict_set(out, "metal/barriers", (b->all.barriers - a->all.barriers) / ne);
    dict_set(out, "metal/cpu_wait_ms", (b->all.wait_ns - a->all.wait_ns) * 1e-6 / ne);
    dict_set(out, "metal/encode_ms", (b->all.encode_ns - a->all.encode_ns) * 1e-6 / ne);
    dict_set(out, "metal/train_dispatches", (b->train.dispatches - a->train.dispatches) / ne);
    dict_set(out, "metal/train_encode_ms", (b->train.encode_ns - a->train.encode_ns) * 1e-6 / ne);
    dict_set(out, "metal/train_wait_ms", (b->train.wait_ns - a->train.wait_ns) * 1e-6 / ne);
    dict_set(out, "metal/train_gpu_ms", (b->train.gpu_ns - a->train.gpu_ns) * 1e-6 / ne);
    dict_set(out, "metal/rollout_dispatches", (b->roll.dispatches - a->roll.dispatches) / ne);
    dict_set(out, "metal/rollout_encode_ms", (b->roll.encode_ns - a->roll.encode_ns) * 1e-6 / ne);
    dict_set(out, "metal/rollout_wait_ms", (b->roll.wait_ns - a->roll.wait_ns) * 1e-6 / ne);
    dict_set(out, "metal/rollout_gpu_ms", (b->roll.gpu_ns - a->roll.gpu_ns) * 1e-6 / ne);
    dict_set(out, "metal/epoch_ms", (b->wall - a->wall) * 1e3 / ne);
    // Average busy CPU cores over the interval (spin / busy-wait cost).
    double dt = b->wall - a->wall;
    dict_set(out, "metal/cpu_cores", dt > 0 ? (b->cpu - a->cpu) / dt : 0.0);
}
#endif

TrainResult run_train(Ini* ini, TrainContext* ctx) {
    int use_selfplay = puf_ini_get(ini, "selfplay", "enabled");
    if (!use_selfplay) {
        puf_ini_put(ini, "vec.num_policies", "1");
        puf_ini_put(ini, "vec.hist_policy_percent", "0");
    } else {
        int npol = puf_ini_get(ini, "vec", "num_policies");
        assert(npol >= 2 && npol <= SELFPLAY_MAX_HIST + 1
            && "selfplay requires vec.num_policies in 2..SELFPLAY_MAX_HIST+1");
        assert(puf_ini_get(ini, "vec", "hist_policy_percent") > 0
            && "selfplay requires vec.hist_policy_percent > 0");
        // Pool opps are this run's checkpoints; hist arch must match policy.
        char hb[32], lb[32];
        snprintf(hb, sizeof(hb), "%d",
            (int)puf_ini_get(ini, "policy", "hidden_size"));
        snprintf(lb, sizeof(lb), "%d",
            (int)puf_ini_get(ini, "policy", "num_layers"));
        puf_ini_put(ini, "vec.hist_policy_hidden_size", hb);
        puf_ini_put(ini, "vec.hist_policy_num_layers", lb);
        double ladder[SELFPLAY_MAX_LADDER];
        assert((puf_ini_get(ini, "selfplay", "eval_bot_games") <= 0
            || puf_ini_get_list(ini, "selfplay", "eval_bots", ladder,
                SELFPLAY_MAX_LADDER) > 0)
            && "selfplay.eval_bot_games requires selfplay.eval_bots");
    }

    char run_id[64];
    snprintf(run_id, sizeof(run_id), "%s", puf_ini_get_str(ini, "base", "run_id"));

    char checkpoint_dir[2048];
    char log_dir[2048];
    snprintf(checkpoint_dir, sizeof(checkpoint_dir), "%s/%s/%s",
        puf_ini_get_str(ini, "base", "checkpoint_dir"),
        puf_ini_get_str(ini, "base", "env_name"), run_id);
    snprintf(log_dir, sizeof(log_dir), "%s/%s",
        puf_ini_get_str(ini, "base", "log_dir"),
        puf_ini_get_str(ini, "base", "env_name"));
    if (ctx->artifact_owner) {
        mkdir_p(checkpoint_dir);
        mkdir_p(log_dir);
    }

    PuffeRL* pufferl = create_pufferl(ini, ctx);
    char warm_start_buf[4096];
    const char* warm_start = puf_checkpoint_path_key(
        ini, "load_model_path", warm_start_buf, sizeof(warm_start_buf));
#ifndef PUF_TEST_SKIP_TRAIN_WARM_START
    if (warm_start) {
        pufferl_load_policy(pufferl, 0, warm_start);
        if (ctx->artifact_owner) {
            printf("Warm-started policy weights from %s "
                "(optimizer, step, replay, and RNG state are fresh).\n", warm_start);
        }
    }
#endif
#ifdef PUF_WARM_START_TEST
    puf_warm_start_test_dump_state(pufferl);
#endif
    Selfplay selfplay = {0};
    if (use_selfplay) {
        char initial_checkpoint[4096];
        snprintf(initial_checkpoint, sizeof(initial_checkpoint),
            "%s/%016ld.bin", checkpoint_dir, pufferl->global_step);
        puf_save_weights(pufferl, initial_checkpoint);
        selfplay.num_hist = pufferl->num_policies - 1;
        assert(selfplay.num_hist > 0 && selfplay.num_hist <= SELFPLAY_MAX_HIST
            && "selfplay requires num_policies in 2..SELFPLAY_MAX_HIST+1");
        selfplay.max_size = puf_ini_get(ini, "selfplay", "max_size");
        assert(selfplay.max_size > 0 && "selfplay.max_size must be positive");
        selfplay.pool = (char (*)[SELFPLAY_PATH_MAX])calloc(
            selfplay.max_size, sizeof(*selfplay.pool));
        selfplay.opp_timeout_steps = puf_ini_get(ini, "selfplay", "opp_timeout_steps");
        selfplay.rng = puf_ini_get(ini, "selfplay", "seed") + pufferl->hypers.rank;
        long current_step = pufferl->global_step * pufferl->hypers.world_size;

        selfplay_add_checkpoint(&selfplay, initial_checkpoint);
        for (int s = 0; s < selfplay.num_hist; s++) {
            SelfplayHist* hist = &selfplay.hist[s];
            hist->policy_idx = s + 1;
            pufferl_load_policy(pufferl, hist->policy_idx, selfplay_sample(&selfplay));
            hist->opp_started_step = current_step;
        }
    }

    long total_timesteps = puf_ini_get(ini, "train", "total_timesteps");
    long batch_size = puf_ini_get(ini, "vec", "total_agents") *
        puf_ini_get(ini, "train", "horizon");
    long local_timesteps = total_timesteps / ctx->world_size;
    long train_epochs = local_timesteps / batch_size;
    long checkpoint_interval = puf_ini_get(ini, "base", "checkpoint_interval");
    char target_key[128];
    sweep_metric_key(ini, target_key, sizeof(target_key));
    const char* test_schedule = getenv("PUF_TEST_NO_INTERMEDIATE_LOG");
    bool test_no_intermediate_log = test_schedule
        && strcmp(test_schedule, "0") != 0;
    Dict last_log = {0};
    // At most one history entry per train epoch (+1 final snapshot for log dump).
    PufLogHistory log_history;
    puf_log_history_init(&log_history, (int)train_epochs + 1);
    TrainResult result = {0};
    char final_checkpoint[4096] = {0};

#ifdef PUF_METAL
    MetalMark metal_first = {}, metal_prev = {};
    double metal_train_ms = 0, metal_rollout_ms = 0;
    double metal_env_ms = 0, metal_model_ms = 0, metal_copy_ms = 0;
#endif
    for (long epoch = 0; epoch < train_epochs; epoch++) {
#ifdef PUF_METAL
        if (epoch == 1) {  // skip epoch 0 (pipeline creation, warmup)
            metal_first = metal_prev = metal_mark(pufferl);
            metal_train_ms = metal_rollout_ms = 0;
            metal_env_ms = metal_model_ms = metal_copy_ms = 0;
        }
#endif
        if (pufferl->hypers.async) {
            // Cleanba 2-slot: warmup fills slot 0; then collect into write
            // while training the other slot (exactly one epoch old).
            int prefetch_next = epoch + 1 < train_epochs;
            if (!pufferl->async_boot) {
                double t0 = rollout_start(pufferl, 0);
                rollout_finish(pufferl, t0);
                pufferl->async_ready_slot = 0;
                pufferl->async_write_slot = 1;
                pufferl->async_boot = true;
            }

            int ready_slot = pufferl->async_ready_slot;
            int write_slot = pufferl->async_write_slot;
            double t0 = 0.0;
            if (prefetch_next) {
                t0 = rollout_start(pufferl, write_slot);
            }

            pufferl->global_step += pufferl->hypers.horizon
                * pufferl->hypers.total_agents;
            RolloutBuf train_src = rollout_time_view(&pufferl->rollouts,
                ready_slot * pufferl->hypers.horizon, pufferl->hypers.horizon);
            PUF_TRACE(11, 0, (int)epoch);
#ifdef PUF_METAL
            double t_epoch = wall_clock();
#endif
            train_impl(pufferl, &train_src);
            PUF_TRACE(13, 0, (int)epoch);
#ifdef PUF_METAL
            double t_train = wall_clock();
#endif

            if (prefetch_next) {
                rollout_finish(pufferl, t0);
                PUF_TRACE(14, 0, (int)epoch);
                pufferl->async_ready_slot = 1 - ready_slot;
                pufferl->async_write_slot = 1 - write_slot;
            }
#ifdef PUF_METAL
            double t_done = wall_clock();
            metal_dw_policy(pufferl, t_done - t_train, t_done - t_epoch);
#endif
        } else {
#ifdef PUF_METAL
            metal_dw_policy(pufferl, 0.0, 1.0);
#endif
            rollouts(pufferl);
            train_impl(pufferl, NULL);
        }

        char saved_checkpoint[4096] = {0};
        if (epoch == train_epochs - 1 || (checkpoint_interval > 0
                && (epoch + 1) % checkpoint_interval == 0)) {
            snprintf(saved_checkpoint, sizeof(saved_checkpoint),
                "%s/%016ld.bin", checkpoint_dir, pufferl->global_step);
            if (ctx->artifact_owner || use_selfplay) {
                puf_save_weights(pufferl, saved_checkpoint);
            }
            if (ctx->artifact_owner) {
                snprintf(final_checkpoint, sizeof(final_checkpoint),
                    "%s", saved_checkpoint);
            }
        }
        if (use_selfplay && saved_checkpoint[0]) {
            selfplay_add_checkpoint(&selfplay, saved_checkpoint);
        }

        // Opponent swap mid-episode: treat as truncate + reset (not boundary wait).
        if (use_selfplay && selfplay.opp_timeout_steps > 0) {
            long step = pufferl->global_step * pufferl->hypers.world_size;
            for (int s = 0; s < selfplay.num_hist; s++) {
                SelfplayHist* hist = &selfplay.hist[s];
                if (step - hist->opp_started_step >= selfplay.opp_timeout_steps) {
                    pufferl_load_policy(pufferl, hist->policy_idx,
                        selfplay_sample(&selfplay));
                    hist->opp_started_step = step;
                }
            }
        }

        // Regression-only deterministic schedule: log every epoch until the
        // objective first appears, then leave the forced final epoch as the
        // next log. The default 0.6-second production cadence is unchanged.
        bool throttle_log = test_no_intermediate_log
            ? last_log.size && dict_find(&last_log, target_key)
            : last_log.size && wall_clock() < pufferl->last_log_time + 0.6;
        if (throttle_log && epoch < train_epochs - 1) {
            continue;
        }

        Dict new_log = {0};
        long global_step = pufferl->global_step;
        double now = wall_clock();
        double dt = now - pufferl->last_log_time;
        double sps = dt > 0 ? (double)(global_step -
            pufferl->last_log_step) / dt * pufferl->hypers.world_size : 0;
        pufferl->last_log_time = now;
        pufferl->last_log_step = global_step;

        dict_set(&new_log, "SPS", sps);
        dict_set(&new_log, "agent_steps", (double)global_step * pufferl->hypers.world_size);
        dict_set(&new_log, "uptime", now - pufferl->start_time);
        dict_set(&new_log, "epoch", (double)pufferl->epoch);

        vec_log(pufferl->vec, &new_log, 1);

        float losses_host[NUM_LOSSES];
        cudaMemcpy(losses_host, pufferl->losses, sizeof(losses_host),
            cudaMemcpyDeviceToHost);
        float inv_n = losses_host[LOSS_N] > 0 ? 1.0f / losses_host[LOSS_N] : 0.0f;
        for (int i = 0; i < LOSS_N; i++) {
            dict_set(&new_log, LOSS_NAMES[i], losses_host[i] * inv_n);
        }
        cudaMemset(pufferl->losses, 0, NUM_LOSSES * sizeof(float));

        log_util(pufferl, &new_log);

#ifdef PUF_METAL
        if (epoch >= 1) {
            MetalMark now_mark = metal_mark(pufferl);
            metal_log(&metal_prev, &now_mark, &new_log);
            metal_prev = now_mark;
            // The first log runs at epoch 0, so these sums start at epoch 1.
            metal_train_ms += pufferl->profile.accum[PROF_TRAIN_MISC]
                + pufferl->profile.accum[PROF_TRAIN_MODEL];
            metal_rollout_ms += pufferl->profile.accum[PROF_ROLLOUT];
            metal_env_ms += pufferl->profile.accum[PROF_ENV];
            metal_model_ms += pufferl->profile.accum[PROF_MODEL];
            metal_copy_ms += pufferl->profile.accum[PROF_COPY];
        } else {
            // Initialize metal/* at epoch 0 with zero deltas, before there
            // is a measured interval to report.
            MetalMark zero = metal_prev;
            zero.epoch = metal_prev.epoch + 1;
            metal_log(&metal_prev, &zero, &new_log);
        }
#endif
        float train_total = 0;
        for (int i = 0; i < NUM_PROF; i++) {
            float sec = pufferl->profile.accum[i] / 1000.0f;
            char key[256];
            snprintf(key, sizeof(key), "perf/%s", PROF_NAMES[i]);
            dict_set(&new_log, key, sec);
            if (i >= PROF_TRAIN_MISC) {
                train_total += sec;
            }
        }
        dict_set(&new_log, "perf/train", train_total);
        memset(pufferl->profile.accum, 0, sizeof(pufferl->profile.accum));

        if (use_selfplay) {
            dict_set(&new_log, "pool/size", selfplay.pool_size);
            dict_set(&new_log, "pool/num_hist", selfplay.num_hist);
            dict_set(&new_log, "pool/num_policies", pufferl->num_policies);
        }
        if (ctx->artifact_owner) {
            puf_dashboard_print(ini, pufferl, &new_log, (int)pufferl->epoch);
        }
        result.cost = dict_get(&new_log, "uptime");
        result.steps = dict_get(&new_log, "agent_steps");
        puf_log_snapshot_update(&last_log, &new_log);
        // Wait until the objective appears; do not treat negative values as missing.
        // The final epoch bypasses the time throttle above, so both the curve
        // and artifact history include current trainer state even with env/n=0.
        if (dict_find(&last_log, target_key)) {
            puf_log_history_add(&log_history, &last_log);
        }
        dict_clear(&new_log);
    }

#ifdef PUF_METAL
    if (ctx->artifact_owner && train_epochs > 1) {
        MetalMark end_mark = metal_mark(pufferl);
        Dict sum = {0};
        metal_log(&metal_first, &end_mark, &sum);
        double ne = (double)(end_mark.epoch - metal_first.epoch);
        double sps = (end_mark.step - metal_first.step) / (end_mark.wall - metal_first.wall);
        printf("metal_summary epochs=%.0f sps=%.0f train_ms=%.4g rollout_ms=%.4g",
            ne, sps, metal_train_ms / ne, metal_rollout_ms / ne);
        printf(" rollout_env_ms=%.4g rollout_model_ms=%.4g rollout_copy_ms=%.4g",
            metal_env_ms / ne, metal_model_ms / ne, metal_copy_ms / ne);
        for (int i = 0; i < sum.size; i++) {
            printf(" %s=%.4g", sum.items[i].key + 6, sum.items[i].value);
        }
        printf("\n");
        dict_clear(&sum);
    }
#endif
    // TrainResult curve: bin-mean over log_history (same as artifact metrics).
    DictItem* target = dict_find(&last_log, target_key);
    result.score = target ? (float)target->value : 0;

    int points = use_selfplay ? 1 : puf_ini_get(ini, "sweep", "downsample");
    assert(points >= 1 && points <= TRAIN_RESULT_MAX_POINTS
        && "sweep.downsample must be in [1, TRAIN_RESULT_MAX_POINTS]");
    result.points = points;

    if (log_history.size == 0 || points == 1) {
        // No observed objective: one explicit fallback, not spurious
        // additional Protein observations at zero cost/steps.
        result.points = 1;
        result.scores[0] = result.score;
        result.costs[0] = result.cost;
        result.step_points[0] = result.steps;
    } else {
        double tmp[TRAIN_RESULT_MAX_POINTS];
        log_history_bin_mean(&log_history, target_key, points, tmp);
        for (int p = 0; p < points; p++) {
            result.scores[p] = (float)tmp[p];
        }
        log_history_bin_mean(&log_history, "uptime", points, tmp);
        for (int p = 0; p < points; p++) {
            result.costs[p] = (float)tmp[p];
        }
        log_history_bin_mean(&log_history, "agent_steps", points, tmp);
        for (int p = 0; p < points; p++) {
            result.step_points[p] = (float)tmp[p];
        }
        result.scores[points - 1] = result.score;
        result.costs[points - 1] = result.cost;
        result.step_points[points - 1] = result.steps;
    }

    long bot_games = use_selfplay ? puf_ini_get(ini, "selfplay", "eval_bot_games") : 0;
    int max_opp = use_selfplay ? puf_ini_get(ini, "selfplay", "eval_pool_size") : 0;
    long pool_games = use_selfplay ? puf_ini_get(ini, "selfplay", "eval_games") : 0;
    int bot_ladder = use_selfplay && bot_games > 0 && final_checkpoint[0];
    int pool_eval = !bot_ladder && use_selfplay && max_opp > 0
        && pool_games > 0 && final_checkpoint[0];
    long eval_episodes = puf_ini_get(ini, "base", "eval_episodes");
    int eval_epoch = (int)pufferl->epoch;
    char eval_ckpt[4096] = {0};
    if (final_checkpoint[0]) {
        snprintf(eval_ckpt, sizeof(eval_ckpt), "%s", final_checkpoint);
    }
    // Close every rank before rank-0 eval. Eval on the live DP trainer leaves
    // child vec workers in BUF_RUNNING/BUF_WAITING while rank 0 waitpid()s
    // (and previously poisoned rank-0 CUDA graphs if children exited early).
    close_pufferl(pufferl);
    if (ctx->artifact_owner && !bot_ladder && !pool_eval && eval_episodes > 0
            && eval_ckpt[0]) {
        puf_ini_put(ini, "base.load_model_path", eval_ckpt);
        TrainContext eval_ctx = {
            .rank = 0,
            .world_size = 1,
            .gpu_id = ctx->gpu_id,
            .artifact_owner = 1,
            .nccl_id = NULL,
        };
        PuffeRL* ep = eval_make(ini, &eval_ctx, EVAL_SCORE, 0);
        EvalResult r = eval_loop(ini, ep, EVAL_SCORE, 1, 0, eval_episodes,
            &last_log, eval_epoch);
        result.score = result.scores[result.points - 1] = r.score;
        close_pufferl(ep);
    }

    char log_path[4096];
    snprintf(log_path, sizeof(log_path), "%s/%s.ini", log_dir, run_id);
    if (ctx->artifact_owner) {
        FILE* fp = fopen(log_path, "w");
        assert(fp && "failed to open log for writing");
        fprintf(fp, "# PufferLib log v1\n");
        puf_ini_write(fp, ini);
        fclose(fp);
    }

    // Final Protein point (points=1): bot ladder > pool match > train curve.
    if (bot_ladder && ctx->artifact_owner) {
        puf_ini_put(ini, "base.load_model_path", final_checkpoint);
        puf_ini_put(ini, "env.num_agents", "1");
        puf_ini_put(ini, "env.num_bots", "1");
        puf_ini_put(ini, "selfplay.enabled", "0");
        puf_ini_put(ini, "vec.num_policies", "1");
        puf_ini_put(ini, "vec.hist_policy_percent", "0");
        // [bot_eval] keys are [env] names (dr, not env.dr). Dotted keys break
        // sweep argv, which flattens as section.key and splits on the last dot.
        Dict* bot_eval = puf_ini_section(ini, "bot_eval", 1);
        for (int i = 0; i < bot_eval->size; i++) {
            DictItem* over = &bot_eval->items[i];
            char ek[PUF_DICT_MAX_KEY + 8];
            snprintf(ek, sizeof(ek), "env.%s", over->key);
            puf_ini_put(ini, ek, over->str);
        }
        // Fixed parallelism; ignore swept train total_agents. Each env plays
        // bot_games / ladder_envs games, so bots face a warmed-up kNN rather
        // than being re-measured cold once per env.
        char nbuf[32];
        long envs = puf_ini_get(ini, "selfplay", "eval_bot_envs");
        snprintf(nbuf, sizeof(nbuf), "%ld", envs);
        puf_ini_put(ini, "vec.total_agents", nbuf);
        long threads = puf_ini_get(ini, "selfplay", "eval_bot_threads");
        if (threads > 0) {
            snprintf(nbuf, sizeof(nbuf), "%ld", threads);
            puf_ini_put(ini, "vec.num_threads", nbuf);
        }
        double ladder[SELFPLAY_MAX_LADDER];
        int rungs = puf_ini_get_list(ini, "selfplay", "eval_bots", ladder,
            SELFPLAY_MAX_LADDER);
        // One PuffeRL for the whole ladder. close_pufferl frees nothing, so a
        // trainer per rung is a leak. Swap bot_policy and restart instead.
        PuffeRL* ep = eval_make(ini, ctx, EVAL_SCORE, 0);
        float sum = 0;
        for (int i = 0; i < rungs; i++) {
            if (PUF_BACKEND != PUF_GPU) {
                for (int e = 0; e < ep->vec->size; e++) {
                    puf_set_bot_policy(&ep->vec->envs[e], (int)ladder[i]);
                }
            }
            env_restart(ep);
            EvalResult r = eval_loop(ini, ep, EVAL_SCORE, 0, 0, bot_games, NULL, 0);
            sum += r.perf;
            printf("bot_eval policy=%d games=%d perf=%.4f\n",
                (int)ladder[i], r.games, r.perf);
        }
        close_pufferl(ep);
        result.score = result.scores[0] = sum / rungs;
        result.points = 1;
        result.costs[0] = result.cost;
        result.step_points[0] = result.steps;
        dict_set(&last_log, "selfplay/bot_ladder_perf", result.score);
        printf("bot_eval mean_perf=%.4f\n", result.score);
    } else if (pool_eval && ctx->artifact_owner) {
        puf_ini_put(ini, "base.load_model_path", final_checkpoint);
        TrainContext eval_ctx = {.world_size = 1, .artifact_owner = 1};
        int n_opp = 0;
        float sum = 0;
        for (int i = 0; i < selfplay.pool_size && n_opp < max_opp; i++) {
            if (strcmp(selfplay.pool[i], final_checkpoint) == 0) {
                continue;
            }
            puf_ini_put(ini, "base.load_enemy_model_path", selfplay.pool[i]);
            PuffeRL* ep = eval_make(ini, &eval_ctx, EVAL_MATCH, 0);
            EvalResult r = eval_loop(ini, ep, EVAL_MATCH, 0, 0, pool_games, NULL, 0);
            close_pufferl(ep);
            sum += r.score;
            n_opp++;
            printf("selfplay_eval vs %s games=%d score=%.4f draw=%.4f\n",
                selfplay.pool[i], r.games, r.score, r.draw);
        }
        if (n_opp) {
            result.score = result.scores[0] = sum / n_opp;
            result.points = 1;
            result.costs[0] = result.cost;
            result.step_points[0] = result.steps;
            dict_set(&last_log, "selfplay/pool_score", result.score);
            printf("selfplay_eval mean_score=%.4f n=%d\n", result.score, n_opp);
        }
    }

    if (ctx->artifact_owner) {
        assert(log_history.size == 0 || dict_find(&last_log, target_key));
        puf_log_history_add(&log_history, &last_log);
        FILE* fp = fopen(log_path, "a");
        assert(fp && "failed to open log for writing");
        fprintf(fp, "\n[metrics]\n");

        // Last snapshot keys (train + selfplay eval overlays).
        if (log_history.size > 0) {
            int metric_points = points;
            double* out = (double*)calloc(metric_points, sizeof(double));
            Dict* key_src = &log_history.items[log_history.size - 1];
            for (int k = 0; k < key_src->size; k++) {
                const char* key = key_src->items[k].key;
                if (strncmp(key, "loss/", 5) == 0) {
                    continue;
                }
                log_history_bin_mean(&log_history, key, metric_points, out);
                fprintf(fp, "%s = ", key);
                for (int i = 0; i < metric_points; i++) {
                    fprintf(fp, "%s%.17g", i ? "," : "", out[i]);
                }
                fputc('\n', fp);
            }
            free(out);
        }
        fclose(fp);
    }
    for (int i = 0; i < log_history.size; i++) {
        dict_clear(&log_history.items[i]);
    }
    free(log_history.items);
    free(selfplay.pool);
    return result;
}

// Fork DP workers before any CUDA/NCCL call (those start runtime threads;
// fork after that SIGSEGVs children and rank 0 hangs in ncclCommInitRank).
// Rank 0 generates the NCCL id after fork and pipes it. Sweep trials occupy
// contiguous GPU blocks; rank 0 owns the last GPU and writes TrainResult.
TrainResult launch_train(Ini* ini) {
    int mb = puf_ini_get(ini, "train", "minibatch_size");
    int horizon = puf_ini_get(ini, "train", "horizon");
    int agents = puf_ini_get(ini, "vec", "total_agents");
    int world_size = puf_ini_get(ini, "train", "gpus");
    int gpu_offset = puf_ini_get(ini, "base", "gpu_offset");
    assert(world_size >= 1 && "train.gpus must be >= 1");
    assert(horizon > 0 && mb % horizon == 0
        && "train.minibatch_size must be divisible by train.horizon");
    assert((long)mb <= (long)horizon * agents
        && "train.minibatch_size must be <= train.horizon * vec.total_agents");
    assert(agents % (mb / horizon) == 0
        && "vec.total_agents must be divisible by minibatch rows");
    assert(horizon % ADV_VEC_WIDTH == 0
        && "train.horizon must be a multiple of ADV_VEC_WIDTH (4 float / 8 bf16)");

    const char* run_id = puf_ini_get_str(ini, "base", "run_id");
    if (!run_id[0] || strcmp(run_id, "None") == 0) {
        char buf[64];
        snprintf(buf, sizeof(buf), "%ld", (long)(1000.0 * wall_clock()));
        puf_ini_put(ini, "base.run_id", buf);
    }

    int nccl_pipe[2] = {-1, -1};
    if (world_size > 1) {
        assert(pipe(nccl_pipe) == 0 && "pipe failed");
    }

    int n_workers = world_size - 1;
    pid_t* pids = (pid_t*)calloc(n_workers, sizeof(pid_t));
    for (int rank = world_size - 1; rank >= 1; rank--) {
        pid_t pid = fork();
        assert(pid >= 0 && "fork failed");
        if (pid == 0) {
            close(nccl_pipe[1]);
            ncclUniqueId nccl_id;
            assert(read(nccl_pipe[0], &nccl_id, sizeof(nccl_id)) == (ssize_t)sizeof(nccl_id)
                && "failed to read ncclUniqueId");
            assert(freopen("/dev/null", "w", stdout) == stdout);
            TrainContext child = {
                .rank = rank,
                .world_size = world_size,
                .gpu_id = gpu_offset + rank - 1,
                .artifact_owner = 0,
                .nccl_id = &nccl_id,
            };
            run_train(ini, &child);
            // Rank 0 may still be in post-train eval. Stay alive until it
            // closes the pipe; exiting earlier aborts NCCL and poisons rank 0
            // CUDA graphs (cudaGraphLaunch fails in pufferl_forward).
            char b;
            ssize_t n = read(nccl_pipe[0], &b, 1);
            assert(n >= 0);
            close(nccl_pipe[0]);
            puf_ini_free(ini);
            exit(0);
        }
        pids[rank - 1] = pid;
    }

    ncclUniqueId nccl_id;
    ncclUniqueId* nccl_ptr = NULL;
    if (world_size > 1) {
        close(nccl_pipe[0]);
        ncclGetUniqueId(&nccl_id);
        for (int i = 0; i < n_workers; i++) {
            assert(write(nccl_pipe[1], &nccl_id, sizeof(nccl_id)) == (ssize_t)sizeof(nccl_id)
                && "failed to write ncclUniqueId");
        }
        nccl_ptr = &nccl_id;
    }

    TrainContext host = {
        .rank = 0,
        .world_size = world_size,
        .gpu_id = gpu_offset + world_size - 1,
        .artifact_owner = 1,
        .nccl_id = nccl_ptr,
    };
    TrainResult result = run_train(ini, &host);
    if (world_size > 1) {
        close(nccl_pipe[1]);
    }
    for (int i = 0; i < n_workers; i++) {
        int status = 0;
        waitpid(pids[i], &status, 0);
        assert(WIFEXITED(status) && WEXITSTATUS(status) == 0
            && "train rank worker failed");
    }
    free(pids);

    // Sweep parent reads this over the pipe; CLI train ignores (result_fd=0).
    int result_fd = puf_ini_get(ini, "base", "result_fd");
    if (result_fd > 0) {
        assert(write(result_fd, &result, sizeof(result)) == sizeof(result));
        close(result_fd);
    }
    return result;
}

#ifdef PUFFERLIB_BUILD_MAIN
int main(int argc, char** argv) {
    setbuf(stdout, NULL);
    setbuf(stderr, NULL);
    if (argc < 2) {
        fprintf(stderr,
            "usage: %s train|eval|match|sweep [latest|MODEL.bin] [--headless] [--section.key=value ...]\n",
            argv[0]);
        exit(1);
    }
    const char* mode = argv[1];
    // Train forks DP workers; CUDA before that fork SIGSEGVs the children.
    if (strcmp(mode, "train") != 0) {
        int total_gpus = 0;
        assert(cudaGetDeviceCount(&total_gpus) == cudaSuccess && total_gpus >= 1
            && "no CUDA devices available");
    }
    int argi = 2;
    const char* model = NULL;
    if (strcmp(mode, "eval") == 0 && argi < argc && argv[argi][0] &&
            argv[argi][0] != '-' && strchr(argv[argi], '=') == NULL) {
        model = argv[argi++];
    }
    int headless = 0;
    char* ini_argv[argc > 0 ? (size_t)argc : 1];
    int ini_argc = 0;
    for (int i = argi; i < argc; i++) {
        if (strcmp(argv[i], "--headless") == 0) {
            headless = 1;
            continue;
        }
        ini_argv[ini_argc++] = argv[i];
    }
    Ini ini = {0};
    puf_ini_load_env(&ini, PUFFER_ENV_NAME, ini_argc, ini_argv);
    if (model) {
        puf_ini_put(&ini, "base.load_model_path", model);
    }
    TrainContext ctx = {.world_size = 1, .artifact_owner = 1};
    int render = !headless;

    if (strcmp(mode, "train") == 0) {
        launch_train(&ini);
    } else if (strcmp(mode, "sweep") == 0) {
        run_sweep(&ini, argv[0]);
    } else if (strcmp(mode, "eval") == 0) {
        run_eval(&ini, &ctx, EVAL_SCORE, 1, render);
    } else if (strcmp(mode, "match") == 0) {
        run_eval(&ini, &ctx, EVAL_MATCH, 1, render);
    } else {
        assert(0 && "unknown mode (train|eval|match|sweep)");
    }

    puf_ini_free(&ini);
    return 0;
}
#endif
