// Microbenchmark of the single-dispatch rollout step (rollout.h rollout_step)
// at the small envs' per-buffer shapes. Two timings per shape:
//   rt   one dispatch per commit + host wait, as a rollout worker does
//        (GPU span of the command buffer and host round trip, medians).
//        An otherwise idle GPU clocks down between such tiny commits, so
//        this overstates in-training latency.
//   tput 64 back-to-back steps in one command buffer (GPU span / step),
//        after 640 warm-up steps; the number to compare kernels by.
// Weights/state are random; only timing is reported (correctness is
// test_rollout's job). tests/metal/run_bench_rollout_step.sh builds it.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
#include <unistd.h>

#include "compat/cuda_bf16.h"
#include "compat/curand_kernel.h"
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
static int g_heads = 1;
#define NUM_ATNS g_heads
#include "kernels.h"

template <typename T> static T* alloc(size_t n) { return (T*)puf_mtl::alloc(n * sizeof(T)); }

static double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

static void bench(const char* name, int B, int K, int H, int L, int A, int iters) {
    const int D = A + 1, M = A;
    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    std::mt19937 gen(7);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    auto precs = [&](size_t n, float scale) {
        P* p = alloc<P>(n);
        for (size_t i = 0; i < n; i++) p[i] = u(gen) * scale;
        return p;
    };
    P* obs = precs((size_t)B * K * 64, 0.0f);
    P* enc = precs((size_t)H * K, 0.5f);
    P* dec = precs((size_t)D * H, 0.2f);
    P* W[4] = {};
    for (int l = 0; l < L; l++) W[l] = precs((size_t)3 * H * H, 0.2f);
    P* st = precs((size_t)L * B * H, 0.5f);
    P* out = precs((size_t)B * D, 0.0f);
    P* init = precs((size_t)L * B * H, 0.0f);
    float* act = alloc<float>(B);
    float* env_act = alloc<float>(B);
    P* lp = precs(B, 0.0f), *val = precs(B, 0.0f);
    P* rew_dst = precs(B, 0.0f), *term_dst = precs(B, 0.0f), *mask_dst = precs((size_t)B * M, 0.0f);
    float* obs_src = alloc<float>((size_t)B * K);
    for (size_t i = 0; i < (size_t)B * K; i++) obs_src[i] = u(gen);
    float* rew_src = alloc<float>(B), *term_src = alloc<float>(B);
    for (int b = 0; b < B; b++) term_src[b] = b % 97 == 0;
    unsigned char* mask_src = alloc<unsigned char>((size_t)B * M);
    memset(mask_src, 1, (size_t)B * M);
    int* sizes = alloc<int>(1);
    sizes[0] = A;
    auto* rng = alloc<curandStatePhilox4_32_10_t>(B);
    PUF_LAUNCH(rng_init, (B + 255) / 256, 256, 0, stream)(rng, uint64_t(42), B);
    cudaStreamSynchronize(stream);

    puf_mtl::RolloutStepArgs a = {};
    auto& f = a.fwd;
    f.obs = puf_mtl::gpu_addr(obs); f.encoder = puf_mtl::gpu_addr(enc);
    for (int l = 0; l < L; l++) f.recurrent[l] = puf_mtl::gpu_addr(W[l]);
    f.decoder = puf_mtl::gpu_addr(dec); f.state = puf_mtl::gpu_addr(st);
    f.out = puf_mtl::gpu_addr(out); f.terminals = puf_mtl::gpu_addr(term_src);
    f.B = B; f.input_dim = K; f.hidden = H; f.layers = L; f.output_dim = D;
    a.smp = {f.out, 0, puf_mtl::gpu_addr(sizes), puf_mtl::gpu_addr(act),
        puf_mtl::gpu_addr(env_act), puf_mtl::gpu_addr(lp), puf_mtl::gpu_addr(val),
        puf_mtl::gpu_addr(rng), puf_mtl::gpu_addr(mask_dst), B, D, 1, M, 0};
    a.obs_src = puf_mtl::gpu_addr(obs_src);
#ifdef PRECISION_FLOAT
    a.obs_is_u8 = 0;  // float env obs into float rows: bitwise copy
#else
    a.obs_is_u8 = 2;
#endif
    a.rew_dst = puf_mtl::gpu_addr(rew_dst); a.rew_src = puf_mtl::gpu_addr(rew_src);
    a.term_dst = puf_mtl::gpu_addr(term_dst); a.term_src = puf_mtl::gpu_addr(term_src);
    a.mask_dst = puf_mtl::gpu_addr(mask_dst); a.mask_src = puf_mtl::gpu_addr(mask_src);
    a.mask_size = M;
    a.obs_row_stride = 64 * K;  // time-interleaved rows, as PUF_METAL_OBS_BT
    if (!puf_mtl::rollout_step_supported(a)) { fprintf(stderr, "unsupported\n"); exit(1); }
    // BENCH_KERNEL=name: dispatch that pipeline (rollout_step's H=64 grid)
    // instead; BENCH_WF32=1 widens the weights to float first (exact).
    const char* kname = getenv("BENCH_KERNEL");
    int split = getenv("BENCH_PACKED") ? atoi(getenv("BENCH_PACKED")) : 0;
    if (split) {
        size_t n = puf_mtl::rollout_packed_floats(K, H, L, D);
        float* packed = alloc<float>(n);
        puf_mtl::RolloutPackArgs pa = {f.encoder, {}, f.decoder, puf_mtl::gpu_addr(packed), K, H, L, D};
        for (int l = 0; l < 4; l++) pa.recurrent[l] = f.recurrent[l];
        puf_mtl::rollout_pack(stream, pa);
        cudaStreamSynchronize(stream);
        puf_mtl::rollout_use_packed(f, puf_mtl::gpu_addr(packed));
    }
    if (getenv("BENCH_WF32")) {
        auto widen = [&](uint64_t src, size_t n) {
            float* d = alloc<float>(n);
            const P* p = (const P*)(src == f.encoder ? (void*)enc : src == f.decoder ? (void*)dec : nullptr);
            for (int l = 0; l < L && !p; l++) if (src == f.recurrent[l]) p = W[l];
            for (size_t i = 0; i < n; i++) d[i] = (float)p[i];
            return puf_mtl::gpu_addr(d);
        };
        f.encoder = widen(f.encoder, (size_t)H * K);
        f.decoder = widen(f.decoder, (size_t)D * H);
        for (int l = 0; l < L; l++) f.recurrent[l] = widen(f.recurrent[l], (size_t)3 * H * H);
    }
    auto step = [&]() {
        if (split) { puf_mtl::rollout_step_packed(stream, a, split); return; }
        if (!kname) { puf_mtl::rollout_step(stream, a); return; }
        static int tg = getenv("BENCH_TG") ? atoi(getenv("BENCH_TG")) : 64;
        static int apg = getenv("BENCH_APG") ? atoi(getenv("BENCH_APG")) : 16;
        puf_mtl::dispatch(stream, puf_mtl::pipeline(kname), dim3((B + apg - 1) / apg), dim3(tg), a);
    };

    std::vector<double> gpu, rt;
    for (int i = 0; i < iters + 20; i++) {
        uint64_t g0 = puf_mtl::stream_stats(stream).gpu_ns;
        auto t0 = std::chrono::steady_clock::now();
        a.fwd.obs = puf_mtl::gpu_addr(obs + (i % 64) * K);
        f.initial_state = i % 64 == 0 ? puf_mtl::gpu_addr(init) : 0;
        f.initial_agents = i % 64 == 0 ? B : 0;
        step();
        cudaStreamSynchronize(stream);
        auto t1 = std::chrono::steady_clock::now();
        // Feedback (GPU span) lags the event signal slightly.
        uint64_t g1;
        for (int w = 0; w < 1000 && (g1 = puf_mtl::stream_stats(stream).gpu_ns) == g0; w++) usleep(20);
        if (i >= 20) {
            gpu.push_back((g1 - g0) * 1e-3);
            rt.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
        }
    }
    std::vector<double> tput;
    for (int r = 0; r < 40; r++) {  // first 10: clock ramp-up
        cudaStreamSynchronize(stream);
        uint64_t g0 = puf_mtl::stream_stats(stream).gpu_ns;
        for (int t = 0; t < 64; t++) {
            a.fwd.obs = puf_mtl::gpu_addr(obs + t * K);
            f.initial_state = 0; f.initial_agents = 0;
            step();
        }
        cudaStreamSynchronize(stream);
        uint64_t g1;
        for (int w = 0; w < 1000 && (g1 = puf_mtl::stream_stats(stream).gpu_ns) == g0; w++) usleep(20);
        if (r >= 10) tput.push_back((g1 - g0) * 1e-3 / 64);
    }
    printf("%-14s B=%5d K=%3d H=%3d L=%d D=%d: rt gpu %7.1f us  host %7.1f us | tput %6.1f us/step\n",
        name, B, K, H, L, D, median(gpu), median(rt), median(tput));
    cudaStreamDestroy(stream);
}

int main(int argc, char** argv) {
    int iters = argc > 1 ? atoi(argv[1]) : 300;
    if (argc > 2) {  // single shape: iters B K H L A
        bench("custom", atoi(argv[2]), atoi(argv[3]), atoi(argv[4]), atoi(argv[5]), atoi(argv[6]), iters);
        return 0;
    }
    bench("flappy/buf", 512, 6, 64, 2, 2, iters);
    bench("flappy/all", 2048, 6, 64, 2, 2, iters);
    bench("cartpole/buf", 1024, 4, 32, 2, 2, iters);
    bench("pong", 1024, 8, 64, 3, 3, iters);
    bench("breakout", 4096, 118, 32, 4, 3, iters / 3);
    return 0;
}
