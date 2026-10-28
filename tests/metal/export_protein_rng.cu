// Regenerate CUDA-only Protein RNG fixtures and the full scrambled Sobol table:
// nvcc -O2 tests/metal/export_protein_rng.cu -lcurand -o export_protein_rng
// ./export_protein_rng "$PUFFER_GOLDEN_DIR/protein/curand_sobol32.bin" [golden_protein_rng.txt]
// One output argument exports only the host table (no GPU work).
#include <assert.h>
#include <curand.h>
#include <curand_kernel.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

__global__ void draws(float *out, unsigned long long seed) {
    int subs[] = {0, 1, 17, 65535};
    int i = threadIdx.x;
    curandStatePhilox4_32_10_t s;
    curand_init(seed, subs[i], 0, &s);
    for (int j = 0; j < 40; j++) {
        out[i * 40 + j] = curand_uniform(&s);
    }
}

int main(int argc, char **argv) {
    assert(argc == 2 || argc == 3);
    const uint32_t endian_check = 1;
    assert(*(const unsigned char *)&endian_check == 1 && "export requires little endian");
    curandDirectionVectors32_t *v;
    unsigned *c;
    assert(curandGetDirectionVectors32(&v,
        CURAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6) == CURAND_STATUS_SUCCESS);
    assert(curandGetScrambleConstants32(&c) == CURAND_STATUS_SUCCESS);
    FILE *f = fopen(argv[1], "wb");
    assert(f);
    for (int d = 0; d < 20000; d++) {
        assert(fwrite(v[d], 4, 32, f) == 32);
        assert(fwrite(c + d, 4, 1, f) == 1);
    }
    assert(fclose(f) == 0);
    if (argc == 2) return 0;
    f = fopen(argv[2], "w");
    assert(f);
    fprintf(f, "# cuRAND scrambled Sobol32, Philox uniform, glibc rand fixtures\n");
    int dims[] = {1, 4, 21, 20000};
    uint32_t offsets[] = {0, 73, 4242, 0x7ffffffe};
    for (int di = 0; di < 4; di++) {
        int dim = dims[di];
        for (int oi = 0; oi < 4; oi++) {
            uint32_t offset = offsets[oi];
            curandGenerator_t g;
            assert(curandCreateGeneratorHost(&g,
                CURAND_RNG_QUASI_SCRAMBLED_SOBOL32) == CURAND_STATUS_SUCCESS);
            assert(curandSetQuasiRandomGeneratorDimensions(g, dim) == CURAND_STATUS_SUCCESS);
            assert(curandSetGeneratorOffset(g, offset) == CURAND_STATUS_SUCCESS);
            float *x = (float *)malloc(dim * sizeof(float));
            for (int step = 0; step < 3; step++) {
                assert(curandGenerateUniform(g, x, dim) == CURAND_STATUS_SUCCESS);
                for (int d = 0; d < dim; d++) {
                    if (d > 23 && d < dim - 2) {
                        continue;
                    }
                    uint32_t bits;
                    memcpy(&bits, x + d, 4);
                    fprintf(f, "S %u %d %d %d %08x\n", offset, dim, step, d, bits);
                }
            }
            free(x);
            curandDestroyGenerator(g);
        }
    }
    unsigned long long seeds[] = {0, 1, 73, 0xdeadbeef01234567ull};
    float *device;
    assert(cudaMalloc(&device, 160 * sizeof(float)) == cudaSuccess);
    for (int si = 0; si < 4; si++) {
        unsigned long long seed = seeds[si];
        draws<<<1, 4>>>(device, seed);
        float x[160];
        assert(cudaMemcpy(x, device, sizeof(x), cudaMemcpyDeviceToHost) == cudaSuccess);
        for (int i = 0; i < 160; i++) {
            uint32_t bits;
            memcpy(&bits, x + i, 4);
            fprintf(f, "P %llu %d %d %d %08x\n", seed, 4, i / 40, i % 40, bits);
        }
    }
    unsigned rseeds[] = {0, 1, 73, 2024, 0xffffffff};
    for (int si = 0; si < 5; si++) {
        unsigned seed = rseeds[si];
        srand(seed);
        for (int i = 0; i < 40; i++) {
            fprintf(f, "R %u %d %d %d %08x\n", seed, 1, 0, i, (unsigned)rand());
        }
    }
    fclose(f);
}
