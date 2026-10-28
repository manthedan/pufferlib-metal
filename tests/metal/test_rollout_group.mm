// Independent per-buffer packed calls vs one two-dimensional dispatch.
// Every output is freshly NaN-poisoned, every input differs across buffers,
// and recurrent state/RNG are carried unchanged through repeated steps.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>

#include "compat/cuda_bf16.h"
#include "compat/curand_kernel.h"
#include "launch.h"
#include "rollout.h"

#ifdef PRECISION_FLOAT
using P = float;
#else
using P = __nv_bfloat16;
#endif

struct Region {
    void* p;
    size_t bytes;
};

static void poison(void* p, size_t bytes) {
    std::fill_n((uint16_t*)p, bytes / 2, (uint16_t)0x7fc1);
}

static void run(int H, int split, int count, int B) {
    const int K = 7, L = 2, D = 4, M = 5, stride = 3 * K;
    std::mt19937 gen(41 + H + split + count + B);
    std::vector<void*> allocations;
    auto alloc = [&](size_t bytes) {
        void* p = puf_mtl::alloc(bytes);
        allocations.push_back(p);
        return p;
    };
    auto prec = [&](size_t n) {
        P* p = (P*)alloc(n * sizeof(P));
        for (size_t i = 0; i < n; i++) p[i] = P(((int)(gen() % 201) - 100) * 0.001f);
        return p;
    };
    auto addr = [](const void* p) { return puf_mtl::gpu_addr(p); };
    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    puf_mtl::stream_host_wait_only(stream);
    puf_mtl::RolloutStepArgs args[2][4] = {};
    std::vector<Region> outputs[2][4];
    std::vector<Region> carries[2][4];
    float* terms[4];
    for (int b = 0; b < count; b++) {
        P* enc = prec(H * K), *dec = prec(D * H);
        P* rec[4] = {prec(3 * H * H), prec(3 * H * H), nullptr, nullptr};
        size_t packed_n = puf_mtl::rollout_packed_floats(K, H, L, D);
        float* packed = (float*)alloc(packed_n * sizeof(float));
        poison(packed, packed_n * sizeof(float));
        puf_mtl::RolloutPackArgs pa = {addr(enc), {addr(rec[0]), addr(rec[1]), 0, 0},
            addr(dec), addr(packed), K, H, L, D};
        puf_mtl::rollout_pack(stream, pa);
        float* obs = (float*)alloc(B * K * sizeof(float));
        float* rewards = (float*)alloc(B * sizeof(float));
        terms[b] = (float*)alloc(B * sizeof(float));
        unsigned char* mask = (unsigned char*)alloc(B * M);
        for (int i = 0; i < B * K; i++) obs[i] = ((int)(gen() % 301) - 150) * 0.01f;
        for (int i = 0; i < B; i++) rewards[i] = b + i * 0.01f;
        for (int i = 0; i < B * M; i++) mask[i] = i % M == 0 || (i + b) % 3 != 0;
        int* sizes = (int*)alloc(sizeof(int));
        sizes[0] = D - 1;
        for (int path = 0; path < 2; path++) {
            auto output = [&](size_t bytes) {
                // Tail guard belongs to the compared region as well.
                void* p = alloc(bytes + 32);
                outputs[path][b].push_back({p, bytes + 32});
                memset((char*)p + bytes, 0xa5, 32);
                return addr(p);
            };
            auto carry = [&](size_t bytes) {
                void* p = alloc(bytes + 32);
                carries[path][b].push_back({p, bytes + 32});
                memset((char*)p + bytes, 0xa5, 32);
                return p;
            };
            P* state = (P*)carry(L * B * H * sizeof(P));
            auto* rng = (curandStatePhilox4_32_10_t*)carry(B * sizeof(curandStatePhilox4_32_10_t));
            if (path == 0) {
                for (int i = 0; i < L * B * H; i++) state[i] = P(((int)(gen() % 201) - 100) * 0.01f);
                memset(rng, 0, B * sizeof(*rng));
                for (int i = 0; i < B; i++) {
                    rng[i].key[0] = 912 + b;
                    rng[i].ctr[2] = i;
                }
            } else {
                for (size_t i = 0; i < carries[path][b].size(); i++) {
                    memcpy(carries[path][b][i].p, carries[0][b][i].p, carries[0][b][i].bytes);
                }
            }
            auto& a = args[path][b];
            a.fwd = {output(B * stride * sizeof(P)), addr(enc),
                {addr(rec[0]), addr(rec[1]), 0, 0}, addr(dec), addr(state),
                output(B * D * sizeof(P)), addr(terms[b]), output(L * (B + 3) * H * sizeof(P)),
                B + 3, B, K, H, L, D};
            a.smp = {a.fwd.out, 0, addr(sizes), output(B * sizeof(float)), output(B * sizeof(float)),
                output(B * sizeof(P)), output(B * sizeof(P)), addr(rng), output(B * M * sizeof(P)),
                B, D, 1, M, 0};
            a.obs_src = addr(obs); a.obs_is_u8 = 2; a.obs_row_stride = stride;
            a.rew_src = addr(rewards); a.rew_dst = output(B * sizeof(P));
            a.term_src = addr(terms[b]); a.term_dst = output(B * sizeof(P));
            a.mask_src = addr(mask); a.mask_dst = a.smp.action_mask; a.mask_size = M;
            puf_mtl::rollout_use_packed(a.fwd, addr(packed));
        }
    }
    cudaStreamSynchronize(stream);
    for (int t = 0; t < 17; t++) {
        for (int b = 0; b < count; b++) {
            for (int i = 0; i < B; i++) terms[b][i] = (i + b * 3 + t) % 7 == 0;
            for (int path = 0; path < 2; path++) {
                for (const Region& r : outputs[path][b]) poison(r.p, r.bytes - 32);
            }
            puf_mtl::rollout_step_packed(stream, args[0][b], split);
        }
        puf_mtl::rollout_steps_packed(stream, args[1], count, split);
        cudaStreamSynchronize(stream);
        for (int b = 0; b < count; b++) {
            for (int carry = 0; carry < 2; carry++) {
                auto& ref = carry ? carries[0][b] : outputs[0][b];
                auto& got = carry ? carries[1][b] : outputs[1][b];
                for (size_t i = 0; i < ref.size(); i++) {
                    if (memcmp(ref[i].p, got[i].p, ref[i].bytes)) {
                        fprintf(stderr, "group mismatch H=%d split=%d count=%d B=%d step=%d buffer=%d carry=%d region=%zu\n",
                            H, split, count, B, t, b, carry, i);
                        exit(1);
                    }
                    const unsigned char* tail = (const unsigned char*)got[i].p + got[i].bytes - 32;
                    for (int k = 0; k < 32; k++) if (tail[k] != 0xa5) abort();
                }
            }
            // Active observation rows must be finite; the skipped rows and
            // snapshot padding deliberately remain NaNs in both paths.
            P* obs = (P*)outputs[1][b][0].p;
            for (int i = 0; i < B; i++) for (int k = 0; k < K; k++) {
                if (!std::isfinite((float)obs[i * stride + k])) abort();
            }
            P* initial = (P*)outputs[1][b][2].p;
            for (int l = 0; l < L; l++) for (int i = 0; i < B * H; i++) {
                if (!std::isfinite((float)initial[l * (B + 3) * H + i])) abort();
            }
            for (size_t i = 1; i < outputs[1][b].size(); i++) {
                if (i == 2) continue;
                const Region& r = outputs[1][b][i];
                size_t n = (r.bytes - 32) / (i == 3 || i == 4 ? sizeof(float) : sizeof(P));
                for (size_t j = 0; j < n; j++) {
                    float v = i == 3 || i == 4 ? ((float*)r.p)[j] : (float)((P*)r.p)[j];
                    if (!std::isfinite(v)) abort();
                }
            }
        }
    }
    printf("group H=%d split=%d count=%d B=%d: outputs/state/RNG/guards exact PASS\n", H, split, count, B);
    cudaStreamDestroy(stream);
    for (void* p : allocations) puf_mtl::free(p);
}

static void throw_fatal(const char* msg) { throw std::runtime_error(msg); }

int main(int argc, char** argv) {
    if (argc == 2 && strcmp(argv[1], "primary-only") == 0) {
        puf_mtl::set_fatal_handler(throw_fatal);
        try {
            puf_mtl::pipeline("rollout_step_packed_h64_p4");
            puts("primary pipeline without auxiliary library PASS");
            return 0;
        } catch (const std::runtime_error&) {
            puts("primary pipeline without auxiliary library FAIL");
            return 1;
        }
    }
    if (argc == 2 && strcmp(argv[1], "reject-aux") == 0) {
        puf_mtl::set_fatal_handler(throw_fatal);
        bool rejected = false;
        try {
            run(64, 2, 2, 17);
        } catch (const std::runtime_error& e) {
            rejected = strstr(e.what(), "grouped rollout library was not compiled for this binary") != nullptr;
        }
        cudaDeviceSynchronize();
        printf("auxiliary precision rejection %s\n", rejected ? "PASS" : "FAIL");
        return rejected ? 0 : 1;
    }
    if (argc == 3) {
        int H = atoi(argv[1]), split = atoi(argv[2]);
        for (int count : {2, 4}) {
            run(H, split, count, 17);
            run(H, split, count, 64);
        }
        return 0;
    }
    for (int count : {2, 4}) for (int H : {32, 64}) for (int split : {1, 2, 4}) {
        if (H == 32 && split == 4) continue;
        run(H, split, count, 17);
        run(H, split, count, 64);
    }
    puts("rollout_group ALL PASS");
}
