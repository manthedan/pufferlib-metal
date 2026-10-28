// Metal backend launch layer: catch-all for kernels without a Metal port.
#include <cstdarg>
#include <cstdio>
#include <cstdlib>

#include "launch.h"

void puf_mtl_kernel(puf_mtl::Unported, const puf_mtl::LaunchCfg& cfg, ...) {
    fprintf(stderr,
        "metal: unimplemented kernel %s (grid %ux%ux%u, block %ux%ux%u)\n",
        cfg.name, cfg.grid.x, cfg.grid.y, cfg.grid.z,
        cfg.block.x, cfg.block.y, cfg.block.z);
    fflush(stderr);
    abort();
}
