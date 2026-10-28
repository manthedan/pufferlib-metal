// CPU compute backend for protein.cu. Accelerate uses column-major FP32 BLAS/LAPACK.
// Keep acquisition, observations, Adam, Pareto and cost logic in protein.cu.
#pragma once

// Public Joe--Kuo D(6) parameters; see tools/metal/gen_protein_sobol.py and
// NOTICE. Compact construction data expands to the same 20,000 x 33 uint32
// layout as the optional external cuRAND fixture. No NVIDIA table is embedded.
#include <CommonCrypto/CommonDigest.h>
#include <stdio.h>

#define PROTEIN_SOBOL_DIMENSIONS 20000
#define PROTEIN_CURAND_SOBOL_SHA256 "fce775e7962beda83fa73c64c628a0323a94eadf5d32bad97d1dc6159e52e5f0"
__asm__(".section __TEXT,__protein_sobol,regular\n"
    ".p2align 2\n_protein_sobol_public:\n"
    ".incbin \"src/metal/protein_sobol_public.bin\"\n.text\n");
extern const unsigned char protein_sobol_public[394264];

static void protein_sobol_error(const char *path, const char *reason) {
    fprintf(stderr, "Protein Sobol table '%s': %s\n", path, reason);
    exit(EXIT_FAILURE);
}

struct ProteinSobolTable {
    uint32_t rows[PROTEIN_SOBOL_DIMENSIONS][33];
    bool exact;
};

// Consume little-endian packed bits. The largest field is 17 bits.
static uint32_t protein_sobol_bits(const unsigned char *p, int *bit, int count) {
    uint32_t value = 0;
    for (int i = 0; i < count; i++, ++*bit) {
        value |= ((p[*bit / 8] >> (*bit % 8)) & 1u) << i;
    }
    return value;
}

static const ProteinSobolTable &protein_sobol_table(void) {
    // C++ static initialization is thread-safe. Select once per process, before
    // the first suggestion; changing the environment afterwards has no effect.
    static const ProteinSobolTable table = [] {
        ProteinSobolTable t = {};
        const char *path = getenv("PUF_PROTEIN_SOBOL_TABLE");
        if (path) {
            FILE *f = fopen(path, "rb");
            if (!f) protein_sobol_error(path, "cannot open external cuRAND table");
            size_t n = fread(t.rows, 1, sizeof(t.rows), f);
            int extra = fgetc(f), error = ferror(f);
            fclose(f);
            if (n != sizeof(t.rows) || extra != EOF || error) {
                protein_sobol_error(path, "expected exactly 2640000 bytes (20000 x 33 uint32)");
            }
            unsigned char digest[CC_SHA256_DIGEST_LENGTH];
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
            CC_SHA256(t.rows, (CC_LONG)sizeof(t.rows), digest);
#pragma clang diagnostic pop
            char hex[65];
            for (int i = 0; i < CC_SHA256_DIGEST_LENGTH; i++) {
                snprintf(hex + 2*i, 3, "%02x", digest[i]);
            }
            if (strcmp(hex, PROTEIN_CURAND_SOBOL_SHA256) != 0) {
                protein_sobol_error(path, "SHA256 mismatch; expected " PROTEIN_CURAND_SOBOL_SHA256);
            }
            t.exact = true;
            return t;
        }
        const unsigned char *p = protein_sobol_public;
        uint64_t state = 0x50554646534f424cULL; // PROTEIN_SOBOL_SHIFT_SEED
        for (int d = 0; d < PROTEIN_SOBOL_DIMENSIONS; d++) {
            uint32_t *v = t.rows[d];
            if (d == 0) {
                for (int b = 0; b < 32; b++) v[b] = 1u << (31-b);
            } else {
                int s = *p++, bit = 0;
                uint32_t a = protein_sobol_bits(p, &bit, s-1);
                for (int b = 0; b < s; b++) {
                    v[b] = ((protein_sobol_bits(p, &bit, b) << 1) | 1u) << (31-b);
                }
                p += (bit+7)/8;
                for (int b = s; b < 32; b++) {
                    v[b] = v[b-s] ^ (v[b-s] >> s);
                    for (int k = 1; k < s; k++) {
                        if (a & (1u << (s-1-k))) v[b] ^= v[b-k];
                    }
                }
            }
            uint64_t z = (state += 0x9e3779b97f4a7c15ULL);
            z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
            z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
            v[32] = (uint32_t)(z ^ (z >> 31));
        }
        assert(p == protein_sobol_public + sizeof(protein_sobol_public));
        return t;
    }();
    return table;
}

static void protein_sobol(uint32_t offset, float *out, int dim) {
    if (dim <= 0 || dim > PROTEIN_SOBOL_DIMENSIONS) {
        protein_sobol_error("dimensions", "expected 1..20000");
    }
    const ProteinSobolTable &table = protein_sobol_table();
    uint32_t gray = offset ^ (offset >> 1);
    for (int d = 0; d < dim; d++) {
        uint32_t v = table.rows[d][32];
        for (int b = 0; b < 32; b++) {
            if (gray & (1u << b)) v ^= table.rows[d][b];
        }
        out[d] = fmaf((float)v, 0x1p-32f, 0x1p-33f);
        // cuRAND conversion can round up to 1. Preserve it in exact mode;
        // public mode has the conventional half-open [0,1) endpoint contract.
        if (!table.exact) out[d] = fminf(out[d], 0x1.fffffep-1f);
    }
}

// glibc rand's additive generator. Darwin's Park-Miller rand changes cost
// noise, ratio shuffles and observation subsampling. This private stream has
// glibc's default seed (1) and never changes the trainer/environment RNG.
static uint32_t protein_rand_state[31];
static int protein_rand_pos = -1;

static int protein_rand(void) {
    int i = protein_rand_pos;
    if (i < 0) {
        uint32_t seed = 1;
        for (int j = 0; j < 31; j++) {
            protein_rand_state[j] = seed;
            seed = (uint64_t)seed * 16807 % 2147483647;
        }
        protein_rand_pos = 3;
        for (int j = 0; j < 310; j++) {
            protein_rand();
        }
        i = protein_rand_pos;
    }
    uint32_t v = protein_rand_state[i] + protein_rand_state[(i + 28) % 31];
    protein_rand_state[i] = v;
    protein_rand_pos = (i + 1) % 31;
    return v >> 1;
}

static void protein_srand(unsigned seed) {
    int32_t word = seed ? seed : 1;
    protein_rand_state[0] = word;
    for (int i = 1; i < 31; i++) {
        word = 16807 * (word % 127773) - 2836 * (word / 127773);
        if (word < 0) {
            word += 2147483647;
        }
        protein_rand_state[i] = word;
    }
    protein_rand_pos = 3;
    for (int i = 0; i < 310; i++) {
        protein_rand();
    }
}

// Same counter/key layout, rounds, subsequences and uniform conversion as
// shaders/rollout.metal; only scalar unsigned arithmetic is needed on CPU.
static void protein_philox(curandStatePhilox4_32_10_t *s) {
    uint32_t c[4], k0 = s->key[0], k1 = s->key[1];
    memcpy(c, s->ctr, sizeof(c));
    for (int r = 0; r < 10; r++) {
        uint64_t p0 = 0xD2511F53ull * c[0], p1 = 0xCD9E8D57ull * c[2];
        uint32_t next[] = {(uint32_t)(p1 >> 32) ^ c[1] ^ k0,
            (uint32_t)p1, (uint32_t)(p0 >> 32) ^ c[3] ^ k1, (uint32_t)p0};
        memcpy(c, next, sizeof(c));
        k0 += 0x9E3779B9u;
        k1 += 0xBB67AE85u;
    }
    memcpy(s->output, c, sizeof(c));
}

static float protein_uniform(curandStatePhilox4_32_10_t *s) {
    uint32_t v = s->output[s->STATE++];
    if (s->STATE == 4) {
        for (int i = 0; i < 4; i++) {
            if (++s->ctr[i]) {
                break;
            }
        }
        protein_philox(s);
        s->STATE = 0;
    }
    return fmaf((float)v, 0x1p-32f, 0x1p-33f);
}

static void protein_k_init_rng(curandStatePhilox4_32_10_t *states, int n,
        unsigned long long seed) {
    for (int i = 0; i < n; i++) {
        memset(states + i, 0, sizeof(*states));
        states[i].ctr[2] = i;
        states[i].key[0] = seed;
        states[i].key[1] = seed >> 32;
        protein_philox(states + i);
    }
}

static void protein_k_sample(float *candidates, const float *centers,
        const float *scales, curandStatePhilox4_32_10_t *rng, int n_candidates,
        int n_centers, int dim, float global_scale, int cost_dim, float fixed_cost) {
    for (int i = 0; i < n_candidates; i++) {
        int center = ((int)(protein_uniform(rng + i) * n_centers)) % n_centers;
        for (int d = 0; d < dim; d++) {
            float s = scales[d] * global_scale;
            float val = fmaf(s, fmaf(2.0f, protein_uniform(rng + i), -1.0f),
                centers[center * dim + d]);
            candidates[i * dim + d] = fmaxf(-1.0f, fminf(1.0f, val));
        }
        if (cost_dim >= 0 && !isnan(fixed_cost)) {
            candidates[i * dim + cost_dim] = fixed_cost;
        }
    }
}

typedef struct {
    int dim, n, cap, pred_cap;
    GPKernel *kernel;
    float raw_noise, mean;
    float *d_X, *d_y, *d_L, *d_alpha, *d_Kinv, *d_Ks, *h_kg;
} GaussianProcess;

static void gp_init(GaussianProcess *gp, int dim, int cap, int pred_m, float noise) {
    *gp = (GaussianProcess){.dim = dim, .cap = cap,
        .pred_cap = pred_m < 4096 ? pred_m : 4096};
    int np = dim + 2;
    gp->kernel = (GPKernel *)malloc(sizeof(GPKernel) + np * sizeof(float));
    gp->kernel->n_params = np;
    gp->kernel->raw_params = (float *)(gp->kernel + 1);
    for (int i = 0; i < np; i++) {
        gp->kernel->raw_params[i] = inv_softplus(1.0f);
    }
    gp->raw_noise = inv_softplus(noise);
    size_t count = cap * dim + 2 * cap + 2 * (size_t)cap * cap
        + (size_t)cap * gp->pred_cap + np;
    gp->d_X = (float *)malloc(count * sizeof(float));
    assert(gp->d_X);
    gp->d_y = gp->d_X + cap * dim;
    gp->d_L = gp->d_y + cap;
    gp->d_alpha = gp->d_L + cap * cap;
    gp->d_Kinv = gp->d_alpha + cap;
    gp->d_Ks = gp->d_Kinv + cap * cap;
    gp->h_kg = gp->d_Ks + cap * gp->pred_cap;
}

static void matern32lin_build_K(GaussianProcess *gp, const float *X1,
        const float *X2, int n, int m, float noise, float *K, cudaStream_t stream) {
    int d = gp->dim, np = gp->kernel->n_params;
    float inv[d];
    for (int k = 0; k < d; k++) {
        inv[k] = 1.0f / softplus(gp->kernel->raw_params[k]);
    }
    float sf = softplus(gp->kernel->raw_params[SF_IDX(np)]);
    float off = softplus(gp->kernel->raw_params[OFF_IDX(np)]);
    for (int col = 0; col < m; col++) {
        for (int row = 0; row < n; row++) {
            float dot = 0, r2 = 0;
            for (int k = 0; k < d; k++) {
                float xr = X1[row * d + k], xc = X2[col * d + k];
                dot = fmaf(xr, xc, dot);
                float diff = (xr - xc) * inv[k];
                r2 = fmaf(diff, diff, r2);
            }
            float u = sqrt(3.0) * sqrt((double)r2);
            float v = sf * (dot + off + (1.0 + u) * exp(-(double)u));
            if (noise != 0 && row == col) {
                v += noise;
            }
            K[col * n + row] = v;
        }
    }
}

static int gp_recompute(GaussianProcess *gp, cudaStream_t stream) {
    int n = gp->n, info;
    float noise = softplus(gp->raw_noise);
    for (int attempt = 0; attempt < 2; attempt++) {
        matern32lin_build_K(gp, gp->d_X, gp->d_X, n, n,
            noise + (attempt ? 1e-8f : 0), gp->d_L, stream);
        spotrf_("L", &n, gp->d_L, &n, &info);
        if (info == 0) {
            break;
        }
    }
    if (info != 0) {
        return -2;
    }
    for (int i = 0; i < n; i++) {
        gp->d_alpha[i] = gp->d_y[i] - gp->mean;
    }
    cblas_strsv(CblasColMajor, CblasLower, CblasNoTrans, CblasNonUnit,
        n, gp->d_L, n, gp->d_alpha, 1);
    cblas_strsv(CblasColMajor, CblasLower, CblasTrans, CblasNonUnit,
        n, gp->d_L, n, gp->d_alpha, 1);
    return 0;
}

static void gp_predict(GaussianProcess *gp, const float *X, float *means, int m,
        cudaStream_t stream) {
    int n = gp->n;
    for (int start = 0; start < m; start += gp->pred_cap) {
        int batch = m - start < gp->pred_cap ? m - start : gp->pred_cap;
        matern32lin_build_K(gp, gp->d_X, X + start * gp->dim,
            n, batch, 0, gp->d_Ks, stream);
        cblas_sgemv(CblasColMajor, CblasTrans, n, batch, 1, gp->d_Ks, n,
            gp->d_alpha, 1, 0, means + start, 1);
        for (int i = 0; i < batch; i++) {
            means[start + i] += gp->mean;
        }
    }
}

// Match CUDA's 16x16 tile, thread order, binary reduction and host fold.
static void protein_mll_grad(GaussianProcess *gp, float *sums) {
    int d = gp->dim, n = gp->n, np = d + 2;
    float inv[d];
    for (int k = 0; k < d; k++) {
        inv[k] = 1.0f / softplus(gp->kernel->raw_params[k]);
    }
    float sf = softplus(gp->kernel->raw_params[SF_IDX(np)]);
    float off = softplus(gp->kernel->raw_params[OFF_IDX(np)]);
    memset(sums, 0, np * sizeof(float));
    float sh[np * GP_BLOCK * GP_BLOCK];
    for (int by = 0; by < n; by += GP_BLOCK) {
        for (int bx = 0; bx < n; bx += GP_BLOCK) {
            memset(sh, 0, sizeof(sh));
            for (int y = 0; y < GP_BLOCK && by + y < n; y++) {
                for (int x = 0; x < GP_BLOCK && bx + x < n; x++) {
                    int row = by + y, col = bx + x, tid = y * GP_BLOCK + x;
                    float dot = 0, r2 = 0;
                    for (int k = 0; k < d; k++) {
                        float xr = gp->d_X[row * d + k], xc = gp->d_X[col * d + k];
                        dot = fmaf(xr, xc, dot);
                        float t = (xr - xc) * inv[k];
                        r2 = fmaf(t, t, r2);
                    }
                    float u = sqrtf(3.0f) * sqrtf(r2), e = expf(-u);
                    float W = fmaf(gp->d_alpha[row], gp->d_alpha[col],
                        -gp->d_Kinv[col * n + row]);
                    for (int k = 0; k < d; k++) {
                        float diff = gp->d_X[row * d + k] - gp->d_X[col * d + k];
                        sh[k * 256 + tid] = W * (sf * 3.0f * diff * diff
                            * inv[k] * inv[k] * inv[k] * e);
                    }
                    sh[d * 256 + tid] = W * fmaf(1.0f + u, e, dot + off);
                    sh[(d + 1) * 256 + tid] = W;
                }
            }
            for (int k = 0; k < np; k++) {
                for (int stride = 128; stride > 0; stride >>= 1) {
                    for (int tid = 0; tid < stride; tid++) {
                        sh[k * 256 + tid] += sh[k * 256 + tid + stride];
                    }
                }
                sums[k] += sh[k * 256];
            }
        }
    }
}

