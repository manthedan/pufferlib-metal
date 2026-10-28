// Historical log binning from upstream PufferLib ea7f9893 src/pufferl.cu.
// Copyright (c) 2022 PufferAI. MIT License; see the repository LICENSE.

static void log_history_bin_mean(PufLogHistory* h, const char* key,
        int points, double* out) {
    assert(h->size > 0 && points >= 1);
    if (points == 1) {
        out[0] = dict_get(&h->items[h->size - 1], key);
        return;
    }
    double final_steps = dict_get(&h->items[h->size - 1], "agent_steps");
    int out_idx = 0;
    int bin_n = 0;
    double bin_sum = 0;
    double fallback = dict_get(&h->items[0], key);
    double next_bin = final_steps / (points - 1);
    for (int i = 0; i < h->size; i++) {
        Dict* log = &h->items[i];
        bin_sum += dict_get(log, key);
        bin_n++;
        double steps = dict_get(log, "agent_steps");
        if (steps < next_bin || out_idx >= points - 1) {
            continue;
        }
        fallback = bin_n ? bin_sum / bin_n : fallback;
        out[out_idx++] = fallback;
        bin_n = 0;
        bin_sum = 0;
        next_bin += final_steps / (points - 1);
    }
    out[points - 1] = dict_get(&h->items[h->size - 1], key);
    while (out_idx < points - 1) {
        out[out_idx++] = fallback;
    }
}
