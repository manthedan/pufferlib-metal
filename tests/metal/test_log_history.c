// The runner extracts the production host-only block from pufferl.cu.
// Expected arrays are hand-calculated, independent of the binning algorithm.
#include <assert.h>
#include <math.h>
#include "ini.h"
#include "log_history_under_test.h"

static int failures;
static void check(const char* name, const double* steps, const double* values,
        int n, int points, const double* expected) {
    PufLogHistory h;
    puf_log_history_init(&h, n);
    for (int i = 0; i < n; i++) {
        Dict d = {0};
        dict_set(&d, "agent_steps", steps[i]);
        if (!isnan(values[i])) dict_set(&d, "value", values[i]);
        puf_log_history_add(&h, &d);
        dict_clear(&d);
    }
    double out[64];
    log_history_bin_mean(&h, "value", points, out);
    for (int i = 0; i < points; i++) {
        if (isnan(out[i]) || fabs(out[i] - expected[i]) > 1e-12) {
            fprintf(stderr, "%s[%d]: got %g, expected %g\n",
                name, i, out[i], expected[i]);
            failures++;
        }
    }
    for (int i = 0; i < n; i++) dict_clear(&h.items[i]);
    free(h.items);
}

int main(void) {
    check("dense + exact boundaries", (double[]){0, 5, 10, 15, 20, 25, 30},
        (double[]){2, 4, 10, 14, 20, 24, 99}, 7, 4,
        (double[]){3, 12, 22, 99});
    check("sparse", (double[]){1, 26, 40}, (double[]){2, 8, 99}, 3, 9,
        (double[]){2, 2, 2, 2, 2, 8, 8, 8, 99});
    check("skip + carry mean", (double[]){1, 2, 31, 32, 50},
        (double[]){2, 6, 20, 40, 99}, 5, 6,
        (double[]){4, 4, 4, 30, 30, 99});
    check("leading gaps", (double[]){21, 22, 40}, (double[]){4, 8, 99}, 3, 5,
        (double[]){4, 4, 6, 6, 99});
    check("single sample", (double[]){40}, (double[]){-7}, 1, 5,
        (double[]){-7, -7, -7, -7, -7});
    check("one point", (double[]){1, 2, 40}, (double[]){5, 6, -7}, 3, 1,
        (double[]){-7});
    check("zero steps", (double[]){0}, (double[]){3}, 1, 4,
        (double[]){3, 3, 3, 3});
    check("duplicate final overlay", (double[]){1, 20, 20},
        (double[]){2, 10, 99}, 3, 3, (double[]){2, 2, 99});
    check("late key + absent samples", (double[]){1, 10, 11, 20, 30},
        (double[]){NAN, 4, 8, NAN, 99}, 5, 4, (double[]){4, 6, 6, 99});

    // Completed episodes are intermittent. Negative objectives are valid;
    // trainer fields must advance while absent episode metrics carry forward.
    Dict last = {0}, fresh = {0};
    dict_set(&last, "env/score", -3);
    dict_set(&last, "env/n", 4);
    dict_set(&last, "epoch", 7);
    dict_set(&last, "metal/train_ms", 123);
    dict_set(&fresh, "env/n", 0);
    dict_set(&fresh, "epoch", 9);
    dict_set(&fresh, "agent_steps", 900);
    dict_set(&fresh, "uptime", 2);
    dict_set(&fresh, "metal/train_ms", 456);
    puf_log_snapshot_update(&last, &fresh);
    assert(dict_get(&last, "env/score") == -3);
    assert(dict_get(&last, "env/n") == 0);
    assert(dict_get(&last, "epoch") == 9);
    assert(dict_get(&last, "agent_steps") == 900);
    assert(dict_get(&last, "uptime") == 2);
    assert(dict_get(&last, "metal/train_ms") == 456);
    dict_clear(&last);
    dict_clear(&fresh);
    if (failures) return 1;
    puts("PASS log history: nine bin fixtures + no-episode snapshot");
    return 0;
}
