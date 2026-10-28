// Fused rollout forward kernels; the shared template is in rollout_forward.h.
#include "rollout_forward.h"

#define ROLLOUT_FORWARD(H, S) \
kernel void rollout_forward_h##H(constant FusedRolloutArgs& a [[buffer(0)]], \
        uint group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]], \
        uint sg [[simdgroup_index_in_threadgroup]]) { \
    threadgroup float x[S][8 * H], combined[S][8 * 3 * H], stage_a[S][64], stage_b[S][64]; \
    /* Weights and state are indexed with the compile-time H. A mismatched */ \
    /* launch (e.g. h64 over hidden=32 buffers) writes state out of bounds */ \
    /* and has wedged the GPU; make it a no-op. Uniform, so barrier-safe. */ \
    if (a.hidden != H || a.layers < 1 || a.layers > 4 \
            || (a.initial_state && a.initial_agents < a.B)) return; \
    rollout_forward<H>(a, group * S + sg, lane, x[sg], combined[sg], stage_a[sg], stage_b[sg]); \
}
ROLLOUT_FORWARD(64, 2)
ROLLOUT_FORWARD(128, 1)

// Keep inter-layer activations in shared memory, but load ragged MMA tiles
// and store logits through the matrix registers. This removes staging and
// its barriers without keeping all hidden tiles live across the projections.
kernel void rollout_forward_h32(constant FusedRolloutArgs& a [[buffer(0)]],
        uint group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
        uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float x[4][8 * 32], combined[4][8 * 3 * 32];
    if (a.hidden != 32 || a.layers < 1 || a.layers > 4 || a.B <= 0
            || a.input_dim < 1 || a.input_dim > 256 || a.output_dim < 1 || a.output_dim > 32
            || (a.initial_state && a.initial_agents < a.B)) return;
    rollout_forward<32, true>(a, group * 4 + sg, lane, x[sg], combined[sg], combined[sg], combined[sg] + 64);
}

#ifdef PUF_TEST_SIGMOID
PUF_SIG_TEST_KERNELS(rollout_fwd, rollout_sigmoid)
#endif
