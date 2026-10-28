// Metal backend: NVTX ranges are no-ops.
#pragma once
static inline int nvtxRangePushA(const char*) { return 0; }
static inline int nvtxRangePop(void) { return 0; }
