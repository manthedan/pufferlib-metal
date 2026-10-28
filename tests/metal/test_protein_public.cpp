// Public mode: exercise the production table, RNG and suggestion API.
#define PUF_METAL 1
#include "../protein_test.h"
#include <set>

int main(int argc, char **argv) {
    const auto &table = protein_sobol_table();
    if (argc == 2 && strcmp(argv[1], "probe") == 0) return 0;
    assert(!table.exact);
    if (argc == 2 && strcmp(argv[1], "table") == 0) {
        assert(fwrite(table.rows, 1, sizeof(table.rows), stdout) == sizeof(table.rows));
        return 0;
    }
    if (argc == 2 && strcmp(argv[1], "suggest") == 0) {
        // Identical synthetic observations keep wall-clock training costs out
        // of this determinism check. Include both initial and GP suggestions.
        srand(2024);
        Space axes[21];
        for (auto &axis : axes) axis = (Space){SPACE_LINEAR, -1, 1, 0.5f};
        SweepSpace space = {axes, 21, -1, 1};
        auto *sw = test_sweep(&space);
        sw->num_random_samples = 10;
        int guided = 0;
        for (int i = 0; i < 16; i++) {
            float x[21];
            ProteinSweepInfo info = protein_sweep_suggest(sw, x, NAN);
            guided += !info.is_random;
            assert(fwrite(x, sizeof(float), 21, stdout) == 21);
            float score = 1 - x[0]*x[0];
            protein_sweep_observe(sw, x, score, 1, 0);
        }
        assert(guided > 0);
        return 0;
    }
    // Joe--Kuo first ten points (Gray-code order), before the digital shift.
    const float reference[10][3] = {
        {0, 0, 0}, {.5, .5, .5}, {.75, .25, .25}, {.25, .75, .75},
        {.375, .375, .625}, {.875, .875, .125}, {.625, .125, .875},
        {.125, .625, .375}, {.1875, .3125, .9375}, {.6875, .8125, .4375}
    };
    for (uint32_t i = 0; i < 10; i++) {
        for (int d = 0; d < 3; d++) {
            uint32_t v = 0, gray = i ^ (i >> 1);
            for (int b = 0; b < 32; b++) if (gray & (1u << b)) v ^= table.rows[d][b];
            assert((double)v * 0x1p-32 == reference[i][d]);
        }
    }
    // All 20,000 dimensions: nonsingular triangular construction and one
    // point per dyadic stratum for the first 256 unscrambled/shifted points.
    for (int d = 0; d < PROTEIN_SOBOL_DIMENSIONS; d++) {
        bool bins[256] = {}, shifted[256] = {};
        for (int b = 0; b < 32; b++) {
            uint32_t pivot = 1u << (31-b);
            assert((table.rows[d][b] & (pivot-1)) == 0);
            assert(table.rows[d][b] & pivot);
        }
        for (uint32_t i = 0; i < 256; i++) {
            uint32_t v = 0, gray = i ^ (i >> 1);
            for (int b = 0; b < 8; b++) if (gray & (1u << b)) v ^= table.rows[d][b];
            assert(!bins[v >> 24]);
            bins[v >> 24] = true;
            v ^= table.rows[d][32];
            assert(!shifted[v >> 24]);
            shifted[v >> 24] = true;
        }
    }
    std::set<float> unique;
    float x[PROTEIN_SOBOL_DIMENSIONS];
    for (uint32_t i = 0; i < 4096; i++) {
        protein_sobol(i, x, 35);
        assert(unique.insert(x[0]).second); // therefore no duplicate points
        for (int d = 0; d < 35; d++) assert(x[d] >= 0 && x[d] < 1);
    }
    for (uint32_t i : {0u, 73u, 0x7ffffffeu, 0xfffffffeu, 0xffffffffu}) {
        protein_sobol(i, x, PROTEIN_SOBOL_DIMENSIONS);
        for (float v : x) assert(v >= 0 && v < 1);
    }
    // Force the uint32 maximum in dimension 1: reverse its direction bits,
    // then invert Gray code. Float conversion would otherwise round to 1.
    uint32_t gray = 0, target = 0xffffffffu ^ table.rows[0][32];
    for (int b = 0; b < 32; b++) gray |= ((target >> (31-b)) & 1u) << b;
    uint32_t offset = gray;
    for (uint32_t rest = gray >> 1; rest; rest >>= 1) offset ^= rest;
    protein_sobol(offset, x, 1);
    assert(x[0] == 0x1.fffffep-1f);
    puts("PASS public Joe-Kuo reference points, stratification, uniqueness and endpoints");
}
