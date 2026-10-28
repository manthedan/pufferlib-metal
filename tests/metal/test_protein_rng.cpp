#define PUF_METAL 1
#include "../protein_test.h"

int main(void) {
    FILE *f = fopen("tests/golden_protein_rng.txt", "r");
    assert(f);
    char line[256];
    float sobol[20000], draws[160];
    curandStatePhilox4_32_10_t *states =
        (curandStatePhilox4_32_10_t *)calloc(65536, sizeof(*states));
    int subs[] = {0, 1, 17, 65535};
    int count = 0;
    while (fgets(line, sizeof(line), f)) {
        if (line[0] == '#') {
            continue;
        }
        char kind;
        unsigned long long seed;
        int dim, step, index;
        uint32_t expected, got;
        assert(sscanf(line, "%c %llu %d %d %d %x",
            &kind, &seed, &dim, &step, &index, &expected) == 6);
        if (kind == 'S') {
            protein_sobol(seed + step, sobol, dim);
            memcpy(&got, sobol + index, 4);
        } else if (kind == 'P') {
            if (step == 0 && index == 0) {
                protein_k_init_rng(states, 65536, seed);
                for (int i = 0; i < 4; i++) {
                    for (int j = 0; j < 40; j++) {
                        draws[i * 40 + j] = protein_uniform(states + subs[i]);
                    }
                }
            }
            memcpy(&got, draws + step * 40 + index, 4);
        } else {
            assert(kind == 'R');
            if (index == 0) {
                srand(seed);
            }
            got = rand();
        }
        if (got != expected) {
            fprintf(stderr, "FAIL %s got=%08x\n", line, got);
            return 1;
        }
        count++;
    }
    fclose(f);
    free(states);
    assert(count == 1464);
    printf("PASS %d bitwise CUDA Sobol/Philox and glibc rand fixtures\n", count);
}
