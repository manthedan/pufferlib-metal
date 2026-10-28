// Environment-only benchmark: same adapter, precision, and optimization as trainer.
#include <chrono>
#include <cmath>
#include <omp.h>
#ifndef PRECISION_FLOAT
#include "cuda_bf16.h"
typedef __nv_bfloat16 precision_t;
#define from_float(x) __float2bfloat16(x)
#else
typedef float precision_t;
#define from_float(x) (x)
#endif
#include "ocean/robot_arm/robot_arm_cpu.h"

int main(int argc, char** argv) {
    int n = argc > 1 ? atoi(argv[1]) : 4096;
    int steps = argc > 2 ? atoi(argv[2]) : 128;
    int threads = argc > 3 ? atoi(argv[3]) : 10;
    assert(n > 0 && steps > 0 && threads > 0);
    omp_set_num_threads(threads);
    Env* envs = (Env*)calloc(n, sizeof(Env));
    obs_t* obs = (obs_t*)calloc(n * OBS_SIZE, sizeof(obs_t));
    float* actions = (float*)calloc(n * RA_ACTIONS, sizeof(float));
    float* rewards = (float*)calloc(n, sizeof(float));
    float* terminals = (float*)calloc(n, sizeof(float));
    Dict kwargs = {0};
    for (int i = 0; i < n; ++i) {
        envs[i].rng = i;
        puf_init(&envs[i], &kwargs);
        envs[i].agents[0] = {obs + i * OBS_SIZE, actions + i * RA_ACTIONS,
            rewards + i, terminals + i, nullptr, 0};
        puf_reset(&envs[i]);
    }
    auto run = [&](int ticks) {
        for (int t = 0; t < ticks; ++t) {
            // Deterministic changing actions, independent of thread scheduling.
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < n; ++i) {
                for (int a = 0; a < RA_ACTIONS; ++a)
                    actions[i * RA_ACTIONS + a] =
                        ((i * 17 + a * 13 + t * 7) % 201 - 100) * 0.01f;
                puf_step(&envs[i]);
            }
        }
    };
    run(8);
    auto start = std::chrono::steady_clock::now();
    run(steps);
    double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    double checksum = 0;
    for (int i = 0; i < n * OBS_SIZE; ++i) {
        assert(std::isfinite((float)obs[i]));
        checksum += (float)obs[i];
    }
    for (int i = 0; i < n; ++i) assert(std::isfinite(rewards[i]));
    printf("env_bench obs_bytes=%zu env_bytes=%zu agents=%d steps=%d threads=%d seconds=%.6f sps=%.1f checksum=%.9g\n",
        sizeof(obs_t), sizeof(Env), n, steps, threads, sec, n * (double)steps / sec, checksum);
    free(envs); free(obs); free(actions); free(rewards); free(terminals);
}
