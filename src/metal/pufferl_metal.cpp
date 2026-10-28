// Metal backend translation unit for the native trainer (`puffer`).
//
// Compiles the upstream CUDA trainer source as plain C++: CUDA headers resolve
// to src/metal/compat (shims implemented in runtime.mm / gemm.mm), device
// code is compiled out by the PUF_METAL guards, and every <<<>>> launch goes
// through PUF_LAUNCH -> puf_mtl_kernel adapters (kernels.h) or aborts with
// "metal: unimplemented kernel <name>".
#define PUF_METAL 1
#ifdef PUFFER_NETHACK
#include "nethack_config.h"
#endif

#include "launch.h"
#include "runtime.h"  // puf_mtl::Stats for the metal/* profile counters
#include "gemm.h"
#include "rollout.h"

#if defined(PUF_METAL_TRAIN_REPLAY) || defined(PUF_ENV_GOLDEN)
#define main puf_upstream_main
#endif
#include "../pufferl.cu"
#if defined(PUF_METAL_TRAIN_REPLAY) || defined(PUF_ENV_GOLDEN)
#undef main
#endif

#include "kernels.h"

#ifdef PUF_METAL_TRAIN_REPLAY
#include "../../tests/metal/train_replay.h"
#endif

#ifdef PUF_ENV_GOLDEN
#include "../../tests/metal/env_golden.h"
#endif
