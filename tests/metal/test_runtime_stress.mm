// Multi-threaded stress test for the Metal CUDA stream/event emulation
// (runtime.mm). It mirrors the host patterns of src/pufferl.cu with
// base.async=1 so a deadlock, lost wakeup or misordered copy shows up as a
// watchdog abort or a wrong value:
//
//   rollout threads (vec_thread_main): one non-blocking stream each; per step
//     timing events around "model" GEMMs, a D2H action copy (GPU-visible or
//     pageable destination), cudaStreamSynchronize, cudaEventElapsedTime,
//     CPU "env" work, then a pageable H2D observation upload; workers park
//     between rollouts and are released by the main thread.
//   learner (main thread, train_impl): legacy cudaMemcpy, staged pageable
//     cudaMemcpyAsync, GEMMs on a non-blocking train stream with weight-grad
//     GEMMs on a side stream joined by cudaEventRecord/cudaStreamWaitEvent
//     (algo.cu puf_mm_tn_async_after / puf_dw_join), cudaStreamSynchronize.
//   every few epochs: NULL-stream and blocking-stream work (legacy default
//     stream dependencies) and cudaMalloc/cudaFree, whose device-wide drain
//     runs while the rollout threads are mid-step.
//
// All data are small integers, so every GEMM result is exact in FP32.
// Usage: test_runtime_stress [seconds=20] [rollout_threads=3] [stall_seconds=120]
// STRESS_DROP_WAIT=1 drops one cross-stream wait to show the checks catch it.
// Build/run: tests/metal/run_test_runtime_stress.sh
#import <Foundation/Foundation.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

#include "compat/cuda_runtime.h"
#include "gemm.h"
#include "runtime.h"

static float* dalloc(long n) {
    float* p = nullptr;
    if (cudaMalloc(&p, (size_t)n * sizeof(float)) != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed\n");
        exit(2);
    }
    return p;
}

static double now_s() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

static void gemm(cudaStream_t s, bool ta, int M, int N, int K, const float* A, int lda,
        const float* B, int ldb, float* C, void* ws = nullptr, size_t ws_bytes = 0,
        float beta = 0.0f) {
    puf_mtl::gemm_rm(s, ta, false, M, N, K, 1.0f, A, lda, B, ldb, beta, C, N, ws, ws_bytes);
}

enum { WAITING = 0, RUNNING = 1 };
enum { MODEL_START, MODEL_END, COPY_END, H2D_START, H2D_END, NUM_EV };

static std::atomic<bool> g_shutdown{false};
static std::atomic<long> g_fail{0};
static std::atomic<long> g_progress{0};
static std::atomic<long> g_staged{0};
// STRESS_DROP_WAIT=1 skips the learner's dW join (self-check: the run must FAIL).
static const bool g_drop_wait = getenv("STRESS_DROP_WAIT") != nullptr;  // busy-stream uploads that took the ring path

struct Rollout {
    int id = 0, B = 64, D = 64, horizon = 16;
    cudaStream_t stream = nullptr;
    float *obs, *eye, *hid, *out;  // device
    float* act_vis;                // cudaHostAlloc (GPU-visible) action buffer
    std::vector<float> act_page, obs_page;  // pageable host buffers
    std::atomic<int> state{WAITING};
    std::atomic<long> steps{0};
    std::atomic<const char*> phase{"init"};
    float model_ms = 0, copy_ms = 0;
    int value = 0;

    void run() {
        cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
        obs = dalloc((long)B * D);
        eye = dalloc((long)D * D);
        hid = dalloc((long)B * D);
        out = dalloc((long)B * D);
        cudaHostAlloc((void**)&act_vis, (size_t)B * D * sizeof(float), 0);
        act_page.assign((size_t)B * D, 0.0f);
        obs_page.assign((size_t)B * D, 0.0f);
        for (int i = 0; i < D; i++) eye[i * D + i] = 1.0f;
        value = id * 1000 + 1;
        for (float& x : obs_page) x = (float)value;
        cudaMemcpy(obs, obs_page.data(), obs_page.size() * 4, cudaMemcpyHostToDevice);
        cudaEvent_t ev[NUM_EV];
        for (int i = 0; i < NUM_EV; i++) cudaEventCreate(&ev[i]);
        while (true) {
            phase = "parked";
            while (state.load() != RUNNING) {
                if (g_shutdown.load()) {
                    for (int i = 0; i < NUM_EV; i++) cudaEventDestroy(ev[i]);
                    return;
                }
                std::this_thread::yield();
            }
            bool h2d_pending = false;
            for (int t = 0; t < horizon; t++) {
                long step = steps.load();
                phase = "model";
                cudaEventRecord(ev[MODEL_START], stream);
                gemm(stream, false, B, D, D, obs, D, eye, D, hid);
                gemm(stream, false, B, D, D, hid, D, eye, D, out);
                cudaEventRecord(ev[MODEL_END], stream);
                bool visible = (step & 1) == 0;
                float* act = visible ? act_vis : act_page.data();
                phase = "copy";
                cudaMemcpyAsync(act, out, (size_t)B * D * 4, cudaMemcpyDeviceToHost, stream);
                cudaEventRecord(ev[COPY_END], stream);
                phase = "stream_sync";
                cudaStreamSynchronize(stream);
                phase = "elapsed";
                float ms = 0;
                cudaEventElapsedTime(&ms, ev[MODEL_START], ev[MODEL_END]);
                model_ms += ms;
                cudaEventElapsedTime(&ms, ev[MODEL_END], ev[COPY_END]);
                copy_ms += ms;
                if (h2d_pending) {
                    cudaEventElapsedTime(&ms, ev[H2D_START], ev[H2D_END]);
                    copy_ms += ms;
                }
                phase = "env";
                long bad = 0;
                for (int i = 0; i < B * D; i++) bad += act[i] != (float)value;
                if (bad) {
                    fprintf(stderr, "rollout %d step %ld: %ld wrong actions (want %d, got %g)\n",
                        id, step, bad, value, act[0]);
                    g_fail++;
                }
                value = id * 1000 + 1 + (int)((step + 1) % 997);
                for (float& x : obs_page) x = (float)value;
                phase = "upload";
                cudaEventRecord(ev[H2D_START], stream);
                bool busy = step % 3 == 2;
                uint64_t d0 = 0;
                if (busy) {
                    // Busy-stream variant: the upload is staged in the ring
                    // behind queued GPU work (a GEMM that reads obs) instead
                    // of copied immediately on the host.
                    gemm(stream, false, B, D, D, obs, D, eye, D, hid);
                    d0 = puf_mtl::stream_stats(stream).dispatches;
                }
                cudaMemcpyAsync(obs, obs_page.data(), obs_page.size() * 4,
                    cudaMemcpyHostToDevice, stream);
                if (busy && puf_mtl::stream_stats(stream).dispatches > d0) g_staged++;
                cudaEventRecord(ev[H2D_END], stream);
                h2d_pending = true;
                steps++;
                g_progress++;
            }
            phase = "final_sync";
            cudaStreamSynchronize(stream);
            if (h2d_pending) {
                float ms = 0;
                cudaEventElapsedTime(&ms, ev[H2D_START], ev[H2D_END]);
                copy_ms += ms;
            }
            state.store(WAITING);
        }
    }
};

struct Learner {
    int R = 2048, D = 64;  // rows x features; dW = X^T Y uses the long-K paths
    cudaStream_t train = nullptr, dw = nullptr, blocking = nullptr;
    cudaEvent_t main_ready, dw_done;
    float *X, *eye[3], *Y, *G, *Z, *lr;
    void* ws = nullptr;
    size_t ws_bytes = 4 << 20;
    std::vector<float> Zref;
    static constexpr long kMaxChurn = 8192;
    float* scratch = nullptr;
    std::atomic<long> epochs{0};
    std::atomic<const char*> phase{"init"};

    void init() {
        cudaStreamCreateWithFlags(&train, cudaStreamNonBlocking);
        cudaStreamCreateWithFlags(&dw, cudaStreamNonBlocking);
        cudaStreamCreate(&blocking);
        cudaEventCreateWithFlags(&main_ready, cudaEventDisableTiming);
        cudaEventCreateWithFlags(&dw_done, cudaEventDisableTiming);
        X = dalloc((long)R * D);
        for (int c = 0; c < 3; c++) eye[c] = dalloc((long)D * D);
        Y = dalloc((long)R * D);
        G = dalloc((long)D * D);
        Z = dalloc((long)R * D);
        lr = dalloc(4);
        scratch = dalloc(4);
        cudaMalloc(&ws, ws_bytes);
        for (int c = 0; c < 3; c++)
            for (int i = 0; i < D; i++) eye[c][i * D + i] = (float)(c + 1);
        for (long i = 0; i < (long)R * D; i++) X[i] = (float)((i * 2654435761u >> 7) % 4);
        // Zref = X (X^T X); every minibatch adds c * Zref (c = 1..3), so
        // 6 * Zref must stay exact in FP32 (entries < 2^24).
        std::vector<double> g((size_t)D * D, 0.0);
        for (int r = 0; r < R; r++)
            for (int i = 0; i < D; i++)
                for (int j = 0; j < D; j++) g[i * D + j] += X[(long)r * D + i] * X[(long)r * D + j];
        Zref.assign((size_t)R * D, 0.0f);
        for (int r = 0; r < R; r++)
            for (int j = 0; j < D; j++) {
                double acc = 0;
                for (int k = 0; k < D; k++) acc += X[(long)r * D + k] * g[k * D + j];
                Zref[(size_t)r * D + j] = (float)acc;
            }
        float zmax = 0;
        for (float z : Zref) zmax = z > zmax ? z : zmax;
        if (6.0f * zmax >= 16777216.0f) {
            fprintf(stderr, "stress: reference not exact in FP32 (max %g)\n", zmax);
            exit(2);
        }
    }

    void epoch(long e) {
        phase = "legacy_memcpy";
        float h_lr = (float)(e % 100);
        cudaMemcpy(lr, &h_lr, 4, cudaMemcpyHostToDevice);
        phase = "train";
        // Each minibatch uses a different scale c, and consecutive epochs
        // rotate the order, so a GEMM that reads a stale Y or G (a lost
        // record/wait dependency) changes the accumulated sum.
        cudaMemsetAsync(Z, 0, (size_t)R * D * 4, train);
        for (int mb = 0; mb < 3; mb++) {
            int c = (int)((e + mb) % 3);
            float ent = (float)mb;
            cudaMemcpyAsync(lr, &ent, 4, cudaMemcpyHostToDevice, train);  // staged
            gemm(train, false, R, D, D, X, D, eye[c], D, Y);  // Y = (c+1) X
            cudaEventRecord(main_ready, train);
            cudaStreamWaitEvent(dw, main_ready, 0);
            gemm(dw, true, D, D, R, X, D, Y, D, G, ws, ws_bytes);  // G = X^T Y
            cudaEventRecord(dw_done, dw);
            if (!g_drop_wait) cudaStreamWaitEvent(train, dw_done, 0);
            gemm(train, false, R, D, D, X, D, G, D, Z, nullptr, 0, 1.0f);  // Z += X G
        }
        phase = "train_sync";
        cudaStreamSynchronize(train);
        long bad = 0;
        for (long i = 0; i < (long)R * D; i++) bad += Z[i] != 6.0f * Zref[i];
        if (bad || lr[0] != 2.0f) {
            fprintf(stderr, "learner epoch %ld: %ld wrong outputs, lr %g\n", e, bad, lr[0]);
            g_fail++;
        }
        if (e % 4 == 3) {
            // Legacy default stream: NULL-stream work orders after the blocking
            // stream's GEMM, and a blocking-stream copy after NULL-stream work.
            phase = "legacy_streams";
            gemm(blocking, false, R, D, D, X, D, eye[0], D, Y);
            cudaMemsetAsync(Y, 0, 16, 0);
            // The runtime's allocation registry never reuses slots
            // (kMaxAllocs), so cap the churn for long soak runs.
            bool churn = e < 4 * kMaxChurn;
            float* tmp = churn ? dalloc(4) : scratch;
            cudaMemcpyAsync(tmp, Y, 16, cudaMemcpyDeviceToDevice, blocking);
            cudaStreamSynchronize(blocking);
            if (tmp[0] != 0.0f || tmp[3] != 0.0f) {
                fprintf(stderr, "learner epoch %ld: legacy ordering broken\n", e);
                g_fail++;
            }
            phase = "free";
            if (churn) cudaFree(tmp);  // device-wide drain while rollouts run
            else cudaDeviceSynchronize();
        }
        epochs++;
        g_progress++;
    }
};

int main(int argc, char** argv) {
    double seconds = argc > 1 ? atof(argv[1]) : 20.0;
    int nroll = argc > 2 ? atoi(argv[2]) : 3;
    double stall_s = argc > 3 ? atof(argv[3]) : 120.0;
    @autoreleasepool {
        Learner L;
        L.init();
        std::vector<Rollout> rolls(nroll);
        std::vector<std::thread> threads;
        for (int i = 0; i < nroll; i++) {
            rolls[i].id = i;
            threads.emplace_back([&rolls, i] { @autoreleasepool { rolls[i].run(); } });
        }
        std::atomic<bool> done{false};
        std::thread watchdog([&] {
            long last = -1;
            double since = now_s();
            while (!done.load()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(200));
                long p = g_progress.load();
                if (p != last) {
                    last = p;
                    since = now_s();
                } else if (now_s() - since > stall_s) {
                    fprintf(stderr, "STALL: no progress for %.0f s. learner phase=%s epochs=%ld\n",
                        stall_s, L.phase.load(), L.epochs.load());
                    for (auto& r : rolls) {
                        fprintf(stderr, "  rollout %d phase=%s state=%d steps=%ld\n", r.id,
                            r.phase.load(), r.state.load(), r.steps.load());
                    }
                    fflush(stderr);
                    abort();
                }
            }
        });

        double t0 = now_s();
        long e = 0;
        while (now_s() - t0 < seconds) {
            // Release every parked buffer (pufferl: BUF_WAITING -> BUF_RUNNING),
            // then train concurrently with the rollouts.
            for (auto& r : rolls) {
                if (r.state.load() == WAITING) r.state.store(RUNNING);
            }
            L.epoch(e++);
        }
        L.phase = "drain";
        for (auto& r : rolls) {
            while (r.state.load() != WAITING) std::this_thread::yield();
        }
        g_shutdown = true;
        for (auto& t : threads) t.join();
        done = true;
        watchdog.join();
        cudaDeviceSynchronize();

        long steps = 0;
        for (auto& r : rolls) steps += r.steps.load();
        puf_mtl::Stats st = puf_mtl::stats();
        printf("stress: %.1f s, %d rollout threads, %ld rollout steps, %ld learner epochs, "
            "dispatches=%llu commits=%llu host_syncs=%llu\n", now_s() - t0, nroll, steps,
            L.epochs.load(), (unsigned long long)st.dispatches, (unsigned long long)st.commits,
            (unsigned long long)st.host_syncs);
        printf("busy-stream uploads staged on the GPU: %ld\n", g_staged.load());
        bool idle = steps == 0 || L.epochs.load() == 0 || (steps >= 30 && g_staged.load() == 0);
        if (idle) g_fail++;
        printf(g_fail ? "FAILED (%ld)\n" : "ALL PASS\n", g_fail.load());
    }
    return g_fail ? 1 : 0;
}
