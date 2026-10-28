// Fixed-action capture of production CPU/CUDA environment code, no learner.
// CUDA: nvcc -x cu -DRA_TRAJECTORY_CUDA ... ; CPU: clang++ ...
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_bf16.h>
#ifdef RA_TRAJECTORY_CUDA
#include <cuda_runtime.h>
#endif
#ifdef PRECISION_FLOAT
typedef float precision_t;
#define from_float(x) (x)
#else
typedef __nv_bfloat16 precision_t;
#define from_float(x) __float2bfloat16(x)
#endif
#ifdef RA_TRAJECTORY_CUDA
#include "ocean/robot_arm/robot_arm.cu"
#define CUDA_OK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
    fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(e)); exit(1); } } while (0)
#else
#include "ocean/robot_arm/robot_arm_cpu.h"
#endif

static const unsigned seeds[] = {1, 2, 43, 999};
enum { ENVS = 4, MODES = 3, WIDTH = 2 * OBS_SIZE + 2 };
#ifdef RA_TRAJECTORY_CUDA
__global__ void capture_unquantized(Env* envs, float* obs) {
    int e = threadIdx.x;
    if (e < ENVS) ra_observe(&envs[e].world.state, obs + e * OBS_SIZE);
}
#endif

static float fixed_action(int t, int e, int a) {
    if (e == 0) return 0.0f;
    // Integer arithmetic makes the action stream identical across compilers.
    int phase = (t + 19 * a + 31 * e) % 200;
    int triangle = phase < 100 ? phase : 200 - phase;
    return (triangle - 50) * (e == 1 ? 0.005f : 0.02f);
}

int main(int argc, char** argv) {
    if (argc != 3 && argc != 4) {
        fprintf(stderr, "usage: %s output.bin steps [reset]\n", argv[0]); return 2;
    }
    bool reset = argc == 4;
    assert(!reset || strcmp(argv[3], "reset") == 0);
    int steps = atoi(argv[2]);
    assert(steps > 0);
    FILE* out = fopen(argv[1], "wb");
    assert(out);
    uint32_t header[] = {0x52415452, 1, MODES, ENVS, (uint32_t)steps + 1,
        WIDTH, sizeof(obs_t), 0};
    assert(fwrite(header, sizeof(header), 1, out) == 1);
    Env* host = (Env*)calloc(ENVS, sizeof(Env));
    obs_t obs[ENVS * OBS_SIZE];
    float actions[ENVS * RA_ACTIONS], rewards[ENVS], terminals[ENVS];
    float raw[ENVS * OBS_SIZE];
#ifdef RA_TRAJECTORY_CUDA
    Env* device;
    obs_t* dobs;
    float *dactions, *drewards, *dterminals, *draw;
    CUDA_OK(cudaMalloc(&device, ENVS * sizeof(Env)));
    CUDA_OK(cudaMalloc(&dobs, sizeof(obs)));
    CUDA_OK(cudaMalloc(&dactions, sizeof(actions)));
    CUDA_OK(cudaMalloc(&drewards, sizeof(rewards)));
    CUDA_OK(cudaMalloc(&dterminals, sizeof(terminals)));
    CUDA_OK(cudaMalloc(&draw, sizeof(raw)));
#endif
    for (int mode = 0; mode < MODES; ++mode) {
        memset(host, 0, ENVS * sizeof(Env));
        Dict kwargs = {0};
        dict_set(&kwargs, "stack", mode == 1);
        dict_set(&kwargs, "basketball", mode == 2);
        for (int e = 0; e < ENVS; ++e) {
#ifdef RA_TRAJECTORY_CUDA
            g_ra_stack = mode == 1;
            g_ra_basketball = mode == 2;
            ra_fill(host + e, seeds[e]);
#else
            host[e].rng = seeds[e] - 1;
            puf_init(host + e, &kwargs);
#ifdef RA_TRAJECTORY_PERTURB
            // Diagnostic only: sensitivity to a single-ULP initial joint change.
            host[e].world.state.q[0] = nextafterf(host[e].world.state.q[0], INFINITY);
#endif
            host[e].agents[0] = {obs + e * OBS_SIZE, actions + e * RA_ACTIONS,
                rewards + e, terminals + e, nullptr, 0};
#endif
            if (reset) host[e].world.state.step =
                (mode == 2 ? RA_BASKETBALL_MAX_STEPS : RA_MAX_STEPS) - 2;
#ifndef RA_TRAJECTORY_CUDA
            puf_reset(host + e);
#endif
        }
        dict_clear(&kwargs);
#ifdef RA_TRAJECTORY_CUDA
        CUDA_OK(cudaMemcpy(device, host, ENVS * sizeof(Env), cudaMemcpyHostToDevice));
        ra_kinit<<<1, RA_CUDA_BLOCK_SIZE>>>(device, dobs, drewards, dterminals, ENVS);
#endif
        for (int t = 0; t <= steps; ++t) {
            if (t > 0) {
                for (int e = 0; e < ENVS; ++e)
                    for (int a = 0; a < RA_ACTIONS; ++a)
                        actions[e * RA_ACTIONS + a] = fixed_action(t - 1, e, a);
#ifdef RA_TRAJECTORY_CUDA
                CUDA_OK(cudaMemcpy(dactions, actions, sizeof(actions), cudaMemcpyHostToDevice));
                ra_kbegin<<<1, RA_CUDA_BLOCK_SIZE>>>(device, 0, ENVS, dactions);
                ra_kphys<<<1, RA_CUDA_BLOCK_SIZE>>>(device, 0, ENVS);
                ra_kfin<<<1, RA_CUDA_BLOCK_SIZE>>>(device, 0, ENVS, dobs, drewards, dterminals);
#else
                for (int e = 0; e < ENVS; ++e) puf_step(host + e);
#endif
            }
#ifdef RA_TRAJECTORY_CUDA
            capture_unquantized<<<1, ENVS>>>(device, draw);
            CUDA_OK(cudaGetLastError());
            CUDA_OK(cudaMemcpy(obs, dobs, sizeof(obs), cudaMemcpyDeviceToHost));
            CUDA_OK(cudaMemcpy(rewards, drewards, sizeof(rewards), cudaMemcpyDeviceToHost));
            CUDA_OK(cudaMemcpy(terminals, dterminals, sizeof(terminals), cudaMemcpyDeviceToHost));
            CUDA_OK(cudaMemcpy(raw, draw, sizeof(raw), cudaMemcpyDeviceToHost));
#else
            for (int e = 0; e < ENVS; ++e)
                ra_observe(&host[e].world.state, raw + e * OBS_SIZE);
#endif
            for (int e = 0; e < ENVS; ++e) {
                float row[WIDTH];
                for (int f = 0; f < OBS_SIZE; ++f) row[f] = (float)obs[e * OBS_SIZE + f];
                row[OBS_SIZE] = rewards[e];
                row[OBS_SIZE + 1] = terminals[e];
                // Unquantized observation from captured state separates
                // physics drift from upstream's unconditional BF16 obs cast.
                memcpy(row + OBS_SIZE + 2, raw + e * OBS_SIZE, OBS_SIZE * sizeof(float));
                assert(fwrite(row, sizeof(row), 1, out) == 1);
            }
        }
    }
    assert(fclose(out) == 0);
#ifdef RA_TRAJECTORY_CUDA
    CUDA_OK(cudaFree(device)); CUDA_OK(cudaFree(dobs)); CUDA_OK(cudaFree(dactions));
    CUDA_OK(cudaFree(drewards)); CUDA_OK(cudaFree(dterminals));
    CUDA_OK(cudaFree(draw));
#endif
    free(host);
}
