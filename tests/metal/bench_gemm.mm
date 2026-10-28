// GEMM microbenchmark: median GPU wall per call for one row-major shape.
//   build/metal/bench_gemm/bench_gemm M N K ta tb [iters] [bf16]
// Build: tests/metal/run_bench_gemm.sh M N K ta tb [iters] [bf16]
// bf16: __nv_bfloat16 A/B/C (fp32 accumulate) instead of float.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cstring>

#include "compat/cuda_bf16.h"
#include "gemm.h"
#include "runtime.h"

int main(int argc, char** argv) {
    if (argc < 6) { fprintf(stderr, "usage: bench_gemm M N K ta tb [iters]\n"); return 2; }
    int M = atoi(argv[1]), N = atoi(argv[2]), K = atoi(argv[3]);
    bool ta = atoi(argv[4]), tb = atoi(argv[5]);
    int iters = argc > 6 ? atoi(argv[6]) : 50;
    bool bf = argc > 7 && strcmp(argv[7], "bf16") == 0;
    size_t es = bf ? 2 : 4;
    cudaStream_t s;
    cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
    size_t ws_bytes = 4 << 20;
    void *ws, *A, *B, *C;
    cudaMalloc(&ws, ws_bytes);
    cudaMalloc(&A, (size_t)M * K * es);
    cudaMalloc(&B, (size_t)K * N * es);
    cudaMalloc(&C, (size_t)M * N * es);
    for (long i = 0; i < (long)M * K; i++) {
        if (bf) ((__nv_bfloat16*)A)[i] = (float)(i % 7) * 0.1f;
        else ((float*)A)[i] = (float)(i % 7) * 0.1f;
    }
    for (long i = 0; i < (long)K * N; i++) {
        if (bf) ((__nv_bfloat16*)B)[i] = (float)(i % 5) * 0.1f;
        else ((float*)B)[i] = (float)(i % 5) * 0.1f;
    }
    int lda = ta ? M : K, ldb = tb ? K : N;
    std::vector<double> t;
    for (int r = 0; r < 7; r++) {
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; i++) {
            if (bf)
                puf_mtl::gemm_rm(s, ta, tb, M, N, K, 1.0f, (__nv_bfloat16*)A, lda,
                    (__nv_bfloat16*)B, ldb, 0.0f, (__nv_bfloat16*)C, N, ws, ws_bytes);
            else
                puf_mtl::gemm_rm(s, ta, tb, M, N, K, 1.0f, (float*)A, lda, (float*)B, ldb,
                    0.0f, (float*)C, N, ws, ws_bytes);
        }
        cudaStreamSynchronize(s);
        t.push_back(std::chrono::duration<double, std::micro>(
            std::chrono::steady_clock::now() - t0).count() / iters);
    }
    std::sort(t.begin(), t.end());
    printf("%s M=%d N=%d K=%d ta=%d tb=%d path=%d  %.1f us/call (median of 7 x %d)\n",
        bf ? "bf16" : "f32 ", M, N, K, ta, tb, (int)puf_mtl::gemm_last_path(), t[3], iters);
    return 0;
}
