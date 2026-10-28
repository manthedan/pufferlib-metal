// Production-API adaptation of the upstream Pareto, cost, classifier, sampling
// and acquisition tests. The old ProteinObs/ProteinAcq APIs were removed.
#include "protein_test.h"

static void test_pareto(void) {
    Space axes[2] = {{SPACE_LINEAR, -1, 1, 0.5f}, {SPACE_LINEAR, -1, 1, 0.5f}};
    SweepSpace space = {axes, 2, 1, 1};
    ProteinSweep *sw = test_sweep(&space);
    float scores[] = {1, 3, 2, 4, 2.5f}, costs[] = {10, 30, 20, 50, 40};
    for (int i = 0; i < 5; i++) {
        float x[] = {0.1f * i, 0.2f * i};
        protein_sweep_observe(sw, x, scores[i], costs[i], 0);
    }
    float out[2];
    ProteinSweepInfo info = protein_sweep_suggest(sw, out, NAN);
    int expected[] = {0, 2, 1, 3};
    assert(info.n_pareto == 4 && sw->n_top == 5);
    assert(memcmp(sw->pareto_buf, expected, sizeof(expected)) == 0);
    assert(sw->top_idx[0] == 3 && sw->upper_cost_threshold == 50);
    // Four Pareto centers and two cheaper shifts for each top observation.
    assert(fabsf(sw->centers_buf[4 * 2 + 1] - (-0.2f)) < 1e-6f);
    assert(fabsf(sw->centers_buf[9 * 2 + 1] - (0.6f - 1.6f / 3)) < 1e-6f);
    assert(info.n_candidates > 0 && isfinite(info.rating));
    for (int i = 0; i < info.n_candidates * 2; i++) {
        assert(sw->h_cands[i] >= -1 && sw->h_cands[i] <= 1);
    }
    info = protein_sweep_suggest(sw, out, 0.25f);
    assert(out[1] == 0.25f);
    float best = -FLT_MAX;
    for (int i = 0; i < info.n_candidates; i++) {
        best = fmaxf(best, sw->h_pred[i]);
    }
    assert(fabsf(info.rating - best) < 1e-6f);
    space.optimize_direction = -1;
    info = protein_sweep_suggest(sw, out, 0.25f);
    best = -FLT_MAX;
    for (int i = 0; i < info.n_candidates; i++) {
        best = fmaxf(best, -sw->h_pred[i]);
    }
    assert(fabsf(info.rating - best) < 1e-6f);
    // Cost cap masks every candidate in unconstrained acquisition.
    sw->max_suggestion_cost = 0;
    info = protein_sweep_suggest(sw, out, NAN);
    assert(info.rating == 0);
    puts("PASS Pareto, search centers, bounds, fixed cost, minimize and cost cap");

    ProteinSweep *pruned = test_sweep(&space);
    float ps[] = {1, 2, 2.5f, 2.51f}, pc[] = {10, 20, 30, 100};
    for (int i = 0; i < 4; i++) {
        float x[] = {0.1f * i, 0.2f * i};
        protein_sweep_observe(pruned, x, ps[i], pc[i], 0);
    }
    info = protein_sweep_suggest(pruned, out, NAN);
    assert(info.n_pareto == 3 && pruned->upper_cost_threshold == 30);
    pruned->prune_pareto = 0;
    info = protein_sweep_suggest(pruned, out, NAN);
    assert(info.n_pareto == 4);
    puts("PASS inefficient Pareto tail pruning");
}

static void test_observations(void) {
    Space axes[2] = {{SPACE_LINEAR, -1, 1, 0.5f}, {SPACE_LOG, 1, 100, 0.5f}};
    SweepSpace space = {axes, 2, 1, 1};
    ProteinSweep *sw = test_sweep(&space);
    float x[] = {0.25f, 0.5f};
    protein_sweep_observe(sw, x, 1, 10, 0);
    protein_sweep_observe(sw, x, 2, 20, 0);
    assert(sw->succ_n == 1 && sw->succ_scores[0] == 2 && sw->succ_costs[0] == 20);
    protein_sweep_observe(sw, x, NAN, 10, 0);
    protein_sweep_observe(sw, x, 1, 10, 1);
    assert(sw->fail_n == 2);
    x[1] = -1;
    protein_sweep_observe(sw, x, 1, 10, 0);
    assert(sw->succ_n == 1);
    sw->use_logit = 1;
    x[1] = 0;
    protein_sweep_observe(sw, x, 0.75f, 10, 0);
    assert(fabsf(sw->succ_scores[1] - logf(3)) < 1e-6f);
    sw->success_cap = sw->succ_n;
    x[0] = 0.9f;
    protein_sweep_observe(sw, x, 0.5f, 10, 0);
    assert(sw->succ_n == 2);
    puts("PASS duplicate replacement, failures, boundary, caps and logit");
}

static void test_models(void) {
    Space axes[2] = {{SPACE_LINEAR, -1, 1, 0.5f}, {SPACE_LOG, 1, 100, 0.5f}};
    SweepSpace space = {axes, 2, 1, 1};
    ProteinSweep *sw = test_sweep(&space);
    for (int i = 0; i < 50; i++) {
        float c = 10 + 90.0f * i / 50;
        float x[] = {0.1f + 0.8f * i / 50, space_normalize(axes + 1, c)};
        protein_sweep_observe(sw, x, 0.5f + 0.3f * logf(c), c, 0);
        x[0] = -x[0];
        protein_sweep_observe(sw, x, NAN, c, 1);
    }
    float out[2];
    ProteinSweepInfo info = protein_sweep_suggest(sw, out, NAN);
    assert(info.n_gp_obs == 64); // exercises recent/random subsampling
    assert(sw->cm_fitted && sw->clf_fitted);
    assert(fabsf(sw->cm_A - 0.5f) < 0.01f && fabsf(sw->cm_B - 0.3f) < 0.01f);
    assert(!protein_sweep_should_stop(sw, -100, 1));
    assert(protein_sweep_should_stop(sw, 0, 50));
    assert(!protein_sweep_should_stop(sw, 10, 50));
    assert(protein_sweep_should_stop(sw, 0, 200));
    assert(!protein_sweep_should_stop(sw, 10, 200));
    assert(sw->clf_w[0] > 0); // positive examples are on the positive x side
    assert(sw->opt_t_score == 15 && sw->opt_t_cost == 15);
    for (int i = 0; i < 4; i++) {
        protein_sweep_suggest(sw, out, NAN);
    }
    assert(sw->opt_t_score == 0 && sw->opt_t_cost == 0);
    puts("PASS cost fit/early stop, success classifier, GP cap and Adam reset");
}

static void test_duplicates(void) {
    Space axis = {SPACE_LINEAR, -1, 1, 0.5f};
    SweepSpace space = {&axis, 1, -1, 1};
    ProteinSweep *sw = test_sweep(&space);
    float x[] = {0, 0, 1e-7f, 0.25f, 0.5f, 0.5f};
    int kept[6];
    int n = filter_near_duplicates(sw, x, 6, 1, PROTEIN_EPSILON, kept);
    assert(n == 3 && kept[0] == 2 && kept[1] == 3 && kept[2] == 5);
    sw->num_random_samples = 2;
    float out;
    ProteinSweepInfo info = protein_sweep_suggest(sw, &out, NAN);
    assert(info.is_random && isfinite(out));
    protein_sweep_observe(sw, &out, 1, 1, 1);
    info = protein_sweep_suggest(sw, &out, NAN);
    assert(info.is_random && sw->succ_n == 0);
    protein_sweep_observe(sw, &out, 1, 1, 0);
    info = protein_sweep_suggest(sw, &out, NAN);
    assert(!info.is_random && info.n_gp_obs == 2);
    puts("PASS KD duplicates, no cost dimension, random and all-failure fallback");
}

int main(void) {
    srand(42);
    test_pareto();
    test_observations();
    test_models();
    test_duplicates();
    return 0;
}
