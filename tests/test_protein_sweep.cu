// Production-API adaptation of upstream's 21D synthetic sweep and GP holdout
// tests (test_sweep.py/test_sweep_hyper.py depend on removed Python modules).
#include "protein_test.h"

int main(void) {
    srand(2024);
    const int dim = 21;
    Space axes[dim];
    for (int i = 0; i < dim; i++) {
        axes[i] = (Space){SPACE_LINEAR, -1, 1, 0.5f};
    }
    axes[0] = (Space){SPACE_LOG, 3e7f, 1e11f, 0.1f};
    axes[2] = (Space){SPACE_LOG, 1e-5f, 0.1f, 0.5f};
    SweepSpace space = {axes, dim, 0, 1};
    ProteinSweep *sw = test_sweep(&space);
    sw->num_random_samples = 10;
    float random_best = 0, guided_best = 0;
    int guided = 0;
    for (int run = 0; run < 40; run++) {
        float x[dim];
        ProteinSweepInfo info = protein_sweep_suggest(sw, x, NAN);
        for (int i = 0; i < dim; i++) {
            assert(isfinite(x[i]) && x[i] >= -1 && x[i] <= 1);
        }
        float lr = space_unnormalize(axes + 2, x[2]);
        float ts = space_unnormalize(axes, x[0]);
        float basic = expf(-powf(log10f(lr) + 3, 2));
        float score = 0, cost = 0;
        for (int ds = 1; ds <= 5; ds++) {
            float frac_ts = ts * ds / 5;
            cost = frac_ts / 5e7f;
            score = basic * cost;
            float obs[dim];
            memcpy(obs, x, sizeof(obs));
            obs[0] = space_normalize(axes, frac_ts);
            protein_sweep_observe(sw, obs, score, cost, 0);
        }
        if (info.is_random) {
            random_best = fmaxf(random_best, score);
        } else {
            guided_best = fmaxf(guided_best, score);
            guided++;
            assert(info.n_pareto > 0 && info.n_gp_obs <= 64 && info.n_candidates > 0);
            assert(isfinite(info.score_loss) && isfinite(info.cost_loss));
        }
        printf("run=%d random=%d score=%.5f cost=%.3f obs=%d pareto=%d\n",
            run, info.is_random, score, cost, sw->succ_n, info.n_pareto);
    }
    assert(guided == 30 && sw->succ_n > 64);
    assert(guided_best >= 0.9f * random_best);
    printf("PASS 21D sweep random_best=%.5f guided_best=%.5f\n", random_best, guided_best);

    // Independent held-out linear score / exponential-cost data, as in
    // test_sweep_hyper.py. Validate prediction, not just finite losses.
    Space two_axes[2] = {{SPACE_LINEAR, -1, 1, 0.5f}, {SPACE_LINEAR, -1, 1, 0.5f}};
    SweepSpace two = {two_axes, 2, -1, 1};
    ProteinSweep *fit = test_sweep(&two);
    for (int i = 0; i < 49; i++) {
        float x[] = {-0.9f + 0.3f * (i % 7), -0.9f + 0.3f * (i / 7)};
        protein_sweep_observe(fit, x, 2 + x[0] + 0.5f * x[1], expf(2 + x[0]), 0);
    }
    float out[2];
    protein_sweep_suggest(fit, out, NAN);
    double mse = 0, cost_mse = 0;
    float single[9];
    for (int i = 0; i < 9; i++) {
        float x[] = {-0.75f + 0.6f * (i % 3), -0.75f + 0.6f * (i / 3)};
        float y = (2 + x[0] + 0.5f * x[1] - fit->min_score)
            / (fit->max_score - fit->min_score + PROTEIN_EPSILON);
        // Kernel test cross covariance and solve through the production API.
#ifdef PUF_METAL
        float pred, cpred;
        gp_predict(&fit->gp_score, x, &pred, 1, 0);
        gp_predict(&fit->gp_cost, x, &cpred, 1, 0);
#else
        float *dx, *dp;
        cudaMalloc(&dx, sizeof(x));
        cudaMalloc(&dp, 2 * sizeof(float));
        cudaMemcpy(dx, x, sizeof(x), cudaMemcpyHostToDevice);
        gp_predict(&fit->gp_score, dx, dp, 1, 0);
        gp_predict(&fit->gp_cost, dx, dp + 1, 1, 0);
        float pred, cpred;
        cudaMemcpy(&pred, dp, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&cpred, dp + 1, sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(dx);
        cudaFree(dp);
#endif
        single[i] = pred;
        float c = (2 + x[0] - fit->log_c_min)
            / (fit->log_c_max - fit->log_c_min + PROTEIN_EPSILON);
        mse += (pred - y) * (pred - y);
        cost_mse += (cpred - c) * (cpred - c);
    }
    assert(sqrt(mse / 9) < 0.02 && sqrt(cost_mse / 9) < 0.02);
    printf("PASS GP holdout score_rmse=%.6f cost_rmse=%.6f\n",
        sqrt(mse / 9), sqrt(cost_mse / 9));
#ifdef PUF_METAL
    // Prediction must preserve outputs across both 4096-candidate boundaries.
    const int m = 8193;
    float *batch = (float *)malloc(m * 2 * sizeof(float));
    float *pred = (float *)malloc(m * sizeof(float));
    for (int i = 0; i < m; i++) {
        batch[2 * i] = -0.75f + 0.6f * (i % 9 % 3);
        batch[2 * i + 1] = -0.75f + 0.6f * (i % 9 / 3);
    }
    gp_predict(&fit->gp_score, batch, pred, m, 0);
    for (int i = 0; i < m; i++) {
        assert(fabsf(pred[i] - single[i % 9]) < 1e-5f);
    }
    free(batch);
    free(pred);
    puts("PASS batched prediction across 4096 and 8192");
#endif
    return 0;
}
