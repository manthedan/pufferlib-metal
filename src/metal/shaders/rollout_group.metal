// Grouped entry points live outside the primary library so adding them
// cannot change the existing pipelines of single-buffer environments.
#include "rollout.metal"
#include "runtime.metal"  // matching precision marker for the auxiliary library

#define ROLLOUT_STEPS_PACKED(H, P, G) \
kernel void rollout_steps_packed_h##H##_p##P(constant RolloutStepArgs* a [[buffer(0)]], \
        uint2 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]], \
        uint sg [[simdgroup_index_in_threadgroup]]) { \
    threadgroup float xa[G][8 * H], xb[G][8 * H], stage[G * P][64]; \
    rollout_step_packed_body<H, P, G>(a[tg.y], tg.x, lane, sg, xa[sg / P], xb[sg / P], stage[sg]); \
}
ROLLOUT_STEPS_PACKED(64, 1, 2)
ROLLOUT_STEPS_PACKED(64, 2, 2)
ROLLOUT_STEPS_PACKED(64, 4, 1)
ROLLOUT_STEPS_PACKED(32, 1, 4)
ROLLOUT_STEPS_PACKED(32, 2, 2)
