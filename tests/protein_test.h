// Shared production API harness for the Protein tests (CUDA or Metal CPU).
#pragma once
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifndef PUF_METAL
#include <cuda_runtime.h>
#define BLOCK_SIZE 256
__device__ __forceinline__ void block_reduce_sum(
        float *smem, float *out, int tid, int nthreads, int nchan) {
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
#include "protein.cu"
#ifdef PUF_METAL
#define srand protein_srand
#define rand protein_rand
#endif

static ProteinSweep *test_sweep(SweepSpace *space) {
    return protein_sweep_create((ProteinSweep){
        .space = space, .num_random_samples = 0, .suggestions_per_pareto = 32,
        .gp_training_iter = 15, .gp_learning_rate = 0.01f,
        .optimizer_reset_frequency = 5, .gp_max_obs = 64, .infer_batch_size = 128,
        .use_success_prob = 1, .prune_pareto = 1, .global_search_scale = 1.0f,
        .max_suggestion_cost = 3600, .expansion_rate = 0.1f,
        .cost_random_suggestion = -0.8f, .early_stop_quantile = 0.3f,
        .success_cap = 256, .failure_cap = 64, .top_k = 5, .rng_seed = 73});
}
