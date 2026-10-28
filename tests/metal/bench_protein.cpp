// CPU GP step timings. Standalone: clang++ -O2 -std=c++17 -Isrc
// tests/metal/bench_protein.cpp -framework Accelerate -o build/metal/protein/bench
#define PUF_METAL 1
#include "../protein_test.h"

static double seconds(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

int main(void) {
    const int d = 21, m = 65536;
    int sizes[] = {64, 256, 750, 1200};
    printf("n,gram_ms,recompute_ms,inverse_ms,gradient_ms,train50_ms,predict65536_ms\n");
    for (int si = 0; si < 4; si++) {
        int n = sizes[si];
        GaussianProcess gp;
        gp_init(&gp, d, n, m, 0.01f);
        float *X = (float *)malloc(n * d * sizeof(float));
        float *y = (float *)malloc(n * sizeof(float));
        float *test = (float *)malloc(m * d * sizeof(float));
        float *pred = (float *)malloc(m * sizeof(float));
        srand(73);
        for (int i = 0; i < n * d; i++) {
            X[i] = 2.0f * rand() / RAND_MAX - 1;
        }
        for (int i = 0; i < m * d; i++) {
            test[i] = 2.0f * rand() / RAND_MAX - 1;
        }
        for (int i = 0; i < n; i++) {
            y[i] = 0.5f + 0.25f * X[i * d];
        }
        int np = d + 4, t = 0;
        float opt[3 * np];
        memset(opt, 0, sizeof(opt));
        gp_train(&gp, opt, opt + np, opt + 2 * np, &t, 0.001f, X, y, n, 0, 0);
        double a = seconds();
        matern32lin_build_K(&gp, X, X, n, n, 0.01f, gp.d_L, 0);
        double b = seconds();
        assert(gp_recompute(&gp, 0) == 0);
        double c = seconds();
        memset(gp.d_Kinv, 0, n * n * sizeof(float));
        for (int i = 0; i < n; i++) {
            gp.d_Kinv[i * (n + 1)] = 1;
        }
        int info;
        spotrs_("L", &n, &n, gp.d_L, &n, gp.d_Kinv, &n, &info);
        assert(info == 0);
        double e = seconds();
        float sums[d + 2];
        protein_mll_grad(&gp, sums);
        double f = seconds();
        gp_train(&gp, opt, opt + np, opt + 2 * np, &t, 0.001f, X, y, n, 50, 0);
        double g = seconds();
        gp_predict(&gp, test, pred, m, 0);
        double h = seconds();
        printf("%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n", n, 1000 * (b - a),
            1000 * (c - b), 1000 * (e - c), 1000 * (f - e),
            1000 * (g - f), 1000 * (h - g));
        fflush(stdout);
        free(gp.kernel);
        free(gp.d_X);
        free(X);
        free(y);
        free(test);
        free(pred);
    }
}
