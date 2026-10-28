// Stream-semantics test for the Metal CUDA-runtime emulation (runtime.mm).
//   1. legacy default stream: GEMM on a blocking stream, then memsetAsync on
//      the NULL stream -> the memset must land last.
//   2. wait-only submission: an idle stream that waits on a busy stream's
//      event and records its own event must not complete early.
//   3. pageable H2D staging across command-buffer rotations: many small
//      cudaMemcpyAsync from a reused stack buffer on a busy stream.
//   4. host fallback copy (large pageable source) stays ordered with the
//      stream.
//   5. dependency bookkeeping: settled and GPU waits repeated many times do
//      not accumulate records; a 24-stream chain is verified and retired by
//      one sync; a strict host call confirms all commit feedback; completed
//      blocking streams do not hit a stream limit.
//   1b. cudaStreamWaitEvent / cudaStreamSynchronize on the NULL stream (and a
//      blocking stream's wait-event) honor legacy default-stream ordering.
//   6. concurrent windows: exact barrier counts and producer -> window ->
//      consumer ordering.
// Build/run: tests/metal/run_test_runtime.sh
#import <Foundation/Foundation.h>

#include <cstdio>
#include <cstring>
#include <chrono>
#include <thread>
#include <vector>

#include "compat/cuda_runtime.h"
#include "gemm.h"
#include "runtime.h"

static int g_fail = 0;

static void check(const char* name, bool ok) {
    if (!ok) g_fail++;
    printf("  %-48s %s\n", name, ok ? "ok" : "FAIL");
}

static float* dalloc(long n) {
    float* p = nullptr;
    cudaMalloc(&p, (size_t)n * sizeof(float));
    return p;
}

// Keeps a stream busy for a while: a large tiled GEMM.
struct Busy {
    int M, N, K;
    float *A, *B, *C;
    explicit Busy(int n = 2048) : M(n), N(n), K(n) {
        A = dalloc((long)M * K);
        B = dalloc((long)K * N);
        C = dalloc((long)M * N);
        for (long i = 0; i < (long)M * K; i++) A[i] = 1.0f;
        for (long i = 0; i < (long)K * N; i++) B[i] = 1.0f;
    }
    void run(cudaStream_t s, float* out = nullptr) {
        puf_mtl::GemmOpts o;
        o.allow_tensor_ops = false;
        o.allow_split_k = false;
        puf_mtl::gemm_rm(s, false, false, M, N, K, 1.0f, A, K, B, N, 0.0f, out ? out : C, N,
            nullptr, 0, &o);
    }
};

// 6. concurrent window (puf_mtl::concurrent_begin/end): the window's first
// dispatch waits for the producer before it, later window dispatches skip
// the barrier, and the first dispatch after the window waits for all of it.
// Counted barriers are exact; the values catch a missing wait (each GEMM
// takes milliseconds, so a consumer started early reads NaN poison).
static void test_concurrent_window(Busy& busy, cudaStream_t s) {
    long n = (long)busy.M * busy.N;
    float* c1 = dalloc(n);
    float* c2 = dalloc(n);
    float* d1 = dalloc(n);
    float* d2 = dalloc(n);
    for (int rep = 0; rep < 3; rep++) {
        for (long i = 0; i < n; i++) c1[i] = c2[i] = d1[i] = d2[i] = NAN;
        puf_mtl::GemmOpts o;
        o.allow_tensor_ops = false;
        o.allow_split_k = false;
        auto times_ones = [&](const float* a, float* out) {  // out = a @ ones
            puf_mtl::gemm_rm(s, false, false, busy.M, busy.N, busy.K, 1.0f, a, busy.K,
                busy.B, busy.N, 0.0f, out, busy.N, nullptr, 0, &o);
        };
        busy.run(s, c1);              // c1 = K
        uint64_t b0 = puf_mtl::stream_stats(s).barriers;
        puf_mtl::concurrent_begin(s);
        times_ones(c1, c2);           // reads the producer: c2 = K^2
        busy.run(s, d1);              // independent, disjoint outputs: d1 = d2 = K
        busy.run(s, d2);
        puf_mtl::concurrent_end(s);
        times_ones(c2, c1);           // reads the window: c1 = K^3
        uint64_t b1 = puf_mtl::stream_stats(s).barriers;
        cudaStreamSynchronize(s);
        float k = (float)busy.K;
        long bad = 0;
        for (long i = 0; i < n; i += 1031) {
            bad += c2[i] != k * k;
            bad += d1[i] != k;
            bad += d2[i] != k;
            bad += c1[i] != k * k * k;
        }
        char name[64];
        snprintf(name, sizeof(name), "concurrent window values (rep %d)", rep);
        check(name, bad == 0);
        snprintf(name, sizeof(name), "concurrent window barriers %llu == 2",
            (unsigned long long)(b1 - b0));
        check(name, b1 - b0 == 2);
    }
    cudaFree(c1);
    cudaFree(c2);
    cudaFree(d1);
    cudaFree(d2);
}

// A held producer makes a missing host wait observable as stale NaNs, even
// on an otherwise idle GPU. Fresh poison before every enqueue; neither the
// source nor the destination can already contain the expected answer.
static void test_host_wait_only() {
    for (int mode = 0; mode < 3; mode++) {
        cudaStream_t prod, cons;
        if (mode == 2) cudaStreamCreate(&prod);  // legacy NULL-stream consumer
        else cudaStreamCreateWithFlags(&prod, cudaStreamNonBlocking);
        cudaStreamCreateWithFlags(&cons, cudaStreamNonBlocking);
        puf_mtl::stream_host_wait_only(prod);
        float* x = dalloc(4);
        float* y = dalloc(4);
        cudaEvent_t ev;
        cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);
        bool ok = true;
        for (int round = 0; round < 3; round++) {
            const uint32_t nan = 0x7fc10000;
            for (int i = 0; i < 4; i++) {
                memcpy(x + i, &nan, sizeof(nan));
                memcpy(y + i, &nan, sizeof(nan));
            }
            void* h = puf_mtl::test_hold(prod);
            // Force a GPU store: cudaMemsetAsync on an idle stream can
            // take its host shortcut, which would bypass the test hold.
            struct Fill { uint64_t dst, n; uint32_t value, aligned; };
            Fill fill = {puf_mtl::gpu_addr(x), 16, (uint32_t)(0x3e + round) * 0x01010101u, 1};
            puf_mtl::dispatch(prod, puf_mtl::pipeline("puf_fill_bytes"), dim3(1), dim3(256), fill);
            cudaEventRecord(ev, prod);
            std::thread release([h] {
                std::this_thread::sleep_for(std::chrono::milliseconds(30));
                puf_mtl::test_release(h);
            });
            if (mode == 0) {
                cudaStreamSynchronize(prod);
            } else {
                cudaStream_t dst = mode == 2 ? nullptr : cons;
                if (mode == 1) cudaStreamWaitEvent(dst, ev, 0);
                cudaMemcpyAsync(y, x, 16, cudaMemcpyDeviceToDevice, dst);
                cudaStreamSynchronize(dst);
            }
            uint32_t want = (uint32_t)(0x3e + round) * 0x01010101u;
            for (int i = 0; i < 4; i++) {
                uint32_t got;
                memcpy(&got, (mode == 0 ? x : y) + i, sizeof(got));
                ok &= got == want;
            }
            release.join();
            cudaDeviceSynchronize();
        }
        const char* names[] = {"host-only: sync sees fresh output",
            "host-only: event consumer sees fresh output",
            "host-only: legacy consumer sees fresh output"};
        check(names[mode], ok);
        cudaEventDestroy(ev);
        cudaFree(x);
        cudaFree(y);
    }
}

int main(int argc, char** argv) {
    @autoreleasepool {
        test_host_wait_only();
        if (argc > 1 && strcmp(argv[1], "host-only") == 0) return g_fail ? 1 : 0;
        Busy busy;
        cudaStream_t blk, nb1, nb2;
        cudaStreamCreate(&blk);
        cudaStreamCreateWithFlags(&nb1, cudaStreamNonBlocking);
        cudaStreamCreateWithFlags(&nb2, cudaStreamNonBlocking);

        // 1. legacy default-stream ordering.
        {
            busy.run(blk);
            cudaMemsetAsync(busy.C, 0, (size_t)busy.M * busy.N * 4, 0);
            cudaDeviceSynchronize();
            bool ok = true;
            for (long i = 0; i < (long)busy.M * busy.N; i += 4099) ok &= busy.C[i] == 0.0f;
            check("legacy: NULL-stream memset after blocking GEMM", ok);

            // And the reverse: NULL-stream work, then a blocking-stream copy.
            float* x = dalloc(4);
            busy.run(0);
            cudaMemcpyAsync(x, busy.C, 16, cudaMemcpyDeviceToDevice, blk);
            cudaStreamSynchronize(blk);
            check("legacy: blocking-stream copy after NULL-stream GEMM",
                x[0] == (float)busy.K);
            // Non-blocking streams are exempt (no assertion on timing; just
            // must not deadlock).
            busy.run(0);
            cudaMemsetAsync(x, 0, 16, nb1);
            cudaDeviceSynchronize();
            cudaFree(x);
        }

        // 1b. NULL-stream cudaStreamWaitEvent / cudaStreamSynchronize(0) are
        // ordered after earlier blocking-stream work (audit candidate 3). The
        // blocking writer is still uncommitted when the NULL stream waits on
        // an unrelated event, so a missing dependency fails deterministically.
        {
            cudaEvent_t ev;
            cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);
            float* x = dalloc(4);
            cudaMemsetAsync(x, 0, 16, nb2);  // unrelated producer
            cudaEventRecord(ev, nb2);
            cudaStreamSynchronize(nb2);
            memset(busy.C, 0, (size_t)busy.M * busy.N * 4);
            busy.run(blk);
            cudaStreamWaitEvent(0, ev, 0);
            cudaStreamSynchronize(0);
            check("legacy: NULL wait-event + NULL sync cover blocking work",
                busy.C[(long)busy.M * busy.N - 1] == (float)busy.K);
            cudaDeviceSynchronize();

            memset(busy.C, 0, (size_t)busy.M * busy.N * 4);
            busy.run(blk);
            cudaStreamSynchronize(0);
            check("legacy: NULL sync covers pending blocking work",
                busy.C[(long)busy.M * busy.N - 1] == (float)busy.K);
            cudaDeviceSynchronize();

            // Reverse: NULL-stream writer, then a blocking stream waits on an
            // unrelated event and is synchronized.
            memset(busy.C, 0, (size_t)busy.M * busy.N * 4);
            busy.run(0);
            cudaStreamWaitEvent(blk, ev, 0);
            cudaStreamSynchronize(blk);
            check("legacy: blocking wait-event + sync cover NULL work",
                busy.C[(long)busy.M * busy.N - 1] == (float)busy.K);
            cudaDeviceSynchronize();
            cudaEventDestroy(ev);
            cudaFree(x);
        }

        // 2. wait-only submission on an idle stream.
        {
            cudaEvent_t e1, e2;
            cudaEventCreateWithFlags(&e1, cudaEventDisableTiming);
            cudaEventCreateWithFlags(&e2, cudaEventDisableTiming);
            memset(busy.C, 0, (size_t)busy.M * busy.N * 4);
            busy.run(nb1);
            cudaEventRecord(e1, nb1);
            cudaStreamWaitEvent(nb2, e1, 0);
            cudaEventRecord(e2, nb2);
            cudaEventSynchronize(e2);
            check("wait-only: event after wait observes producer", busy.C[0] == (float)busy.K);
            memset(busy.C, 0, (size_t)busy.M * busy.N * 4);
            busy.run(nb1);
            cudaEventRecord(e1, nb1);
            cudaStreamWaitEvent(nb2, e1, 0);
            cudaStreamSynchronize(nb2);
            check("wait-only: stream sync observes producer",
                busy.C[(long)busy.M * busy.N - 1] == (float)busy.K);
            cudaEventDestroy(e1);
            cudaEventDestroy(e2);
        }

        // 3. pageable staging across rotations (ring + dispatch limit).
        {
            const int n = 3000, words = 1024;  // 4 KiB each -> > 1 MiB ring
            float* dst = dalloc((long)n * words);
            busy.run(nb1);
            float stage[words];
            for (int i = 0; i < n; i++) {
                for (int w = 0; w < words; w++) stage[w] = (float)(i * 7 + w);
                cudaMemcpyAsync(dst + (long)i * words, stage, sizeof(stage),
                    cudaMemcpyHostToDevice, nb1);
            }
            cudaStreamSynchronize(nb1);
            long bad = 0;
            for (int i = 0; i < n; i++)
                for (int w = 0; w < words; w++)
                    bad += dst[(long)i * words + w] != (float)(i * 7 + w);
            check("staged pageable H2D across rotations", bad == 0);
            cudaFree(dst);
        }

        test_concurrent_window(busy, nb2);

        // 4. host fallback (pageable > 64 KiB) after queued GPU work.
        {
            std::vector<float> big((size_t)busy.M * busy.N, -3.0f);
            busy.run(nb1);  // writes K into C
            cudaMemcpyAsync(busy.C, big.data(), big.size() * 4, cudaMemcpyHostToDevice, nb1);
            cudaStreamSynchronize(nb1);
            check("host fallback copy lands after queued GEMM", busy.C[12345] == -3.0f);
        }

        // 5. dependency bookkeeping.
        {
            cudaEvent_t e;
            cudaEventCreateWithFlags(&e, cudaEventDisableTiming);
            float* y = dalloc(1024);
            // 5a. wait-only consumer on a completed producer, many times:
            // nothing may accumulate.
            cudaStream_t w;
            cudaStreamCreateWithFlags(&w, cudaStreamNonBlocking);
            cudaMemsetAsync(y, 1, 4096, nb1);
            cudaEventRecord(e, nb1);
            cudaStreamSynchronize(nb1);
            size_t d0 = puf_mtl::pending_deps();
            for (int i = 0; i < 10000; i++) {
                cudaStreamWaitEvent(w, e, 0);
                cudaStreamSynchronize(w);
            }
            check("bookkeeping: 10k settled wait/sync stay flat", puf_mtl::pending_deps() == d0);
            // 5b. GPU waits, many times. The producer is held on the GPU
            // (test_hold) until the consumer has committed, so every round
            // records a GPU wait.
            Busy small(64);
            cudaEvent_t e2;
            cudaEventCreateWithFlags(&e2, cudaEventDisableTiming);
            bool flat = true;
            uint64_t c0 = puf_mtl::stream_stats(w).commits;
            for (int i = 0; i < 400; i++) {
                void* h = puf_mtl::test_hold(nb1);
                small.run(nb1);
                cudaEventRecord(e, nb1);
                cudaStreamWaitEvent(w, e, 0);
                cudaMemsetAsync(y + 512, 0, 16, w);
                cudaEventRecord(e2, w);  // commit the consumer
                if (puf_mtl::pending_deps() != d0 + 1) flat = false;
                puf_mtl::test_release(h);
                cudaStreamSynchronize(w);
                if (puf_mtl::pending_deps() != d0) flat = false;
            }
            cudaEventDestroy(e2);
            cudaDeviceSynchronize();
            check("bookkeeping: 400 GPU wait/sync stay flat", flat
                && puf_mtl::stream_stats(w).commits - c0 == 400 && puf_mtl::pending_deps() == d0);
            // 5c. dependency chain deeper than 16 streams behind a held
            // producer; syncing the tail verifies (and retires) all of it.
            cudaStream_t chain[24];
            for (int i = 0; i < 24; i++) {
                cudaStreamCreateWithFlags(&chain[i], cudaStreamNonBlocking);
            }
            void* h = puf_mtl::test_hold(nb1);
            busy.run(nb1);
            cudaStream_t prev = nb1;
            for (int i = 0; i < 24; i++) {
                cudaEventRecord(e, prev);
                cudaStreamWaitEvent(chain[i], e, 0);
                cudaMemsetAsync(y + i, 0, 4, chain[i]);
                cudaEventRecord(e, chain[i]);  // commit behind the GPU wait
                prev = chain[i];
            }
            size_t pend = puf_mtl::pending_deps();
            puf_mtl::test_release(h);
            cudaStreamSynchronize(chain[23]);
            check("bookkeeping: 24-stream chain verified and retired",
                pend == d0 + 24 && puf_mtl::pending_deps() == d0);
            // 5d. strict host call after a lagged sync joins the producer's
            // commit feedback.
            busy.run(nb1);
            cudaEventRecord(e, nb1);
            cudaStreamWaitEvent(nullptr, e, 0);
            cudaMemsetAsync(y, 0, 16, nullptr);
            cudaStreamSynchronize(nullptr);
            float tmp[4];
            cudaMemcpy(tmp, y, 16, cudaMemcpyDeviceToHost);
            check("feedback: strict copy after lagged sync confirms all",
                puf_mtl::unconfirmed_commits() == 0);
            // 5e. completed blocking streams do not count against a limit.
            for (int i = 0; i < 70; i++) {
                cudaStream_t t;
                cudaStreamCreate(&t);
                cudaMemsetAsync(y, i, 16, t);
                cudaStreamSynchronize(t);
                cudaStreamDestroy(t);
            }
            cudaMemsetAsync(y, 3, 16, nullptr);
            cudaStreamSynchronize(nullptr);
            check("legacy: 70 completed blocking streams", ((unsigned char*)y)[0] == 3);
            cudaEventDestroy(e);
            cudaFree(y);
        }

        puf_mtl::Stats st = puf_mtl::stats();
        printf("dispatches=%llu commits=%llu host_syncs=%llu\n",
            (unsigned long long)st.dispatches, (unsigned long long)st.commits,
            (unsigned long long)st.host_syncs);
        printf(g_fail ? "FAILED (%d)\n" : "ALL PASS\n", g_fail);
    }
    return g_fail ? 1 : 0;
}
