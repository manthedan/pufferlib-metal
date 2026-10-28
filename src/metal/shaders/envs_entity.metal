// point embed + max-pool encoders shared by minimal and asteroids.
// Each output has one writer; reductions are serial and deterministic.
// precision_t storage, with FP32 arithmetic and explicit CUDA-compatible rounding.
#include "common.h"

struct EntityArgs {
    device precision_t* dst;
    device int* argmax;
    device const precision_t* input;
    device const precision_t* weight;
    device const precision_t* grad;
    int B, H, obs_size, self_dim, point_dim, num_points;
};

kernel void entity_materialize(constant EntityArgs& a [[buffer(0)]],
        uint i [[thread_position_in_grid]]) {
    int D = a.self_dim + a.point_dim;
    if (i >= uint(a.B * a.num_points * D)) return;
    int d = i % D, p = (i / D) % a.num_points, b = i / (D * a.num_points);
    int o = d < a.self_dim ? d : a.self_dim + p * a.point_dim + d - a.self_dim;
    a.dst[i] = from_float(to_float(a.input[b * a.obs_size + o]));
}

kernel void entity_linear_max(constant EntityArgs& a [[buffer(0)]],
        uint i [[thread_position_in_grid]]) {
    if (i >= uint(a.B * a.H)) return;
    int b = i / a.H, h = i % a.H, best = 0;
    float maximum = -MAXFLOAT;
    for (int p = 0; p < a.num_points; p++) {
        float s = 0;
        for (int d = 0; d < 16; d++)
            s = fma(to_float(a.weight[h * 16 + d]), max(0.0f, to_float(a.input[(b * a.num_points + p) * 16 + d])), s);
        // CUDA's strict comparison picks the first point on ties.
        if (s > maximum) { maximum = s; best = p; }
    }
    a.dst[i] = from_float(maximum);
    if (a.argmax) a.argmax[i] = best;
}

kernel void entity_output_wgrad(constant EntityArgs& a [[buffer(0)]],
        uint i [[thread_position_in_grid]]) {
    if (i >= uint(a.H * 16)) return;
    int h = i / 16, d = i % 16;
    float s = 0;
    for (int b = 0; b < a.B; b++) {
        int p = a.argmax[b * a.H + h];
        s = fma(to_float(a.grad[b * a.H + h]), max(0.0f, to_float(a.input[(b * a.num_points + p) * 16 + d])), s);
    }
    a.dst[i] = from_float(s);
}

kernel void entity_grad_input(constant EntityArgs& a [[buffer(0)]],
        uint i [[thread_position_in_grid]]) {
    if (i >= uint(a.B * a.num_points * 16)) return;
    int b = i / (a.num_points * 16), p = (i / 16) % a.num_points, d = i % 16;
    float s = 0;
    if (to_float(a.input[i]) > 0) {
        for (int h = 0; h < a.H; h++)
            if (a.argmax[b * a.H + h] == p)
                s = fma(to_float(a.grad[b * a.H + h]), to_float(a.weight[h * 16 + d]), s);
    }
    a.dst[i] = from_float(s);
}
