// Runtime helper kernels: stream-ordered memcpy/memset on unified memory.
#include "common.h"

struct CopyArgs {
    device uchar* dst;
    device const uchar* src;
    ulong n;
    uint aligned;
    uint pad;
};

kernel void puf_copy_bytes(constant CopyArgs& a [[buffer(0)]],
        uint gid [[thread_position_in_grid]]) {
    ulong i = (ulong)gid * 16;
    if (i >= a.n) return;
    if (a.aligned && i + 16 <= a.n) {
        *(device uint4*)(a.dst + i) = *(device const uint4*)(a.src + i);
        return;
    }
    ulong end = min(i + 16, a.n);
    for (ulong j = i; j < end; j++) a.dst[j] = a.src[j];
}

struct FillArgs {
    device uchar* dst;
    ulong n;
    uint value;  // byte replicated x4
    uint aligned;
};

kernel void puf_fill_bytes(constant FillArgs& a [[buffer(0)]],
        uint gid [[thread_position_in_grid]]) {
    ulong i = (ulong)gid * 16;
    if (i >= a.n) return;
    if (a.aligned && i + 16 <= a.n) {
        *(device uint4*)(a.dst + i) = uint4(a.value);
        return;
    }
    ulong end = min(i + 16, a.n);
    for (ulong j = i; j < end; j++) a.dst[j] = (uchar)(a.value & 0xff);
}

// Precision marker checked by runtime.mm load_library (never dispatched).
#ifdef PRECISION_FLOAT
kernel void puf_precision_marker_f32(device uint* out [[buffer(0)]]) { out[0] = 4; }
#else
kernel void puf_precision_marker_bf16(device uint* out [[buffer(0)]]) { out[0] = 2; }
#endif
