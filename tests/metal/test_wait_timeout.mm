// Do MTL4 queue event waits time out and release dependent work?
//
// Probe for the runtime audit's candidate 1 (Apple documents that a timed-out
// MTL4 queue wait unblocks subsequent work, without giving the timeout).
//
// mode "cpu" (default): the producer is the CPU, which withholds a shared-
// event signal; no GPU work is pending at all.
//   A (wait-only):  queue A: wait(PA, 1); signal(QA, 1)
//   B (with work):  queue B: wait(PB, 1); commit [tiny kernel writes 42];
//                   signal(QB, 1). The commit carries a feedback handler.
//   The host polls every 50 ms for up to LIMIT seconds (default 120), then
//   signals PA / PB itself so nothing stays blocked, and drains.
//
// mode "gpu": the producer is a GPU queue that is busy (making progress) for
// about TOTAL seconds (default 10): a chain of command buffers of a bounded
// ALU kernel (fixed iteration count, ~250 ms each), then signal(PC, 1).
//   C: queue C: wait(PC, 1); commit [tiny kernel writes 42]; signal(QC, 1).
//   Shows whether a wait on a slow-but-alive producer also times out.
//   Also checks two kinds of historical ordering evidence: (1) a GPU witness,
//   each producer command buffer ends by copying its index into a progress
//   word and the consumer's command buffer starts by copying that word
//   (timed-out wait -> stale witness); (2) commit-feedback GPU timestamps
//   (consumer GPUStartTime vs the last producer GPUEndTime; same clock).
//
// Every kernel has a hard iteration bound; nothing can spin forever.
// Build/run from the repo root:
//   tests/metal/run_test_wait_timeout.sh cpu [limit_s=120]
//   tests/metal/run_test_wait_timeout.sh gpu [total_s=10] [nwaits=1] [consumer_pre_cbs=0]
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <mach/mach_time.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>

static double now_s() {
    static mach_timebase_info_data_t tb = {0, 0};
    if (tb.denom == 0) mach_timebase_info(&tb);
    return (double)mach_absolute_time() * tb.numer / tb.denom * 1e-9;
}

static std::atomic<int> fb_done{0};
static double fb_t = 0;
static NSString* fb_err = nil;
static double g_t0 = 0;

static const char* kSrc =
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "kernel void sentinel(device uint* o [[buffer(0)]], uint i [[thread_position_in_grid]]) {\n"
    "    if (i == 0) o[0] = 42u;\n"
    "}\n"
    // Bounded ALU loop: iters is clamped to 1<<24 on the GPU as well.
    "struct SpinArgs { device float* out; uint iters; };\n"
    "kernel void spin(constant SpinArgs& a [[buffer(0)]], uint i [[thread_position_in_grid]]) {\n"
    "    uint n = min(a.iters, 1u << 24);\n"
    "    float x = float(i) * 1e-6f, y = 1.0001f;\n"
    "    for (uint k = 0; k < n; k++) { x = fma(x, y, 1e-7f); y = fma(y, 0.99999f, 1e-6f); }\n"
    "    if (x == 12345.0f) a.out[i & 1023] = y;\n"
    "}\n";

struct Ctx {
    id<MTLDevice> dev;
    id<MTLComputePipelineState> sentinel, spin;
    id<MTLBuffer> out, args;
    id<MTLResidencySet> rs;
};

struct SpinArgs { uint64_t out; uint32_t iters, pad; };

// One command buffer = one dispatch of `pso` with argument address `arg`.
static id<MTL4CommandBuffer> one_dispatch(Ctx& c, id<MTL4CommandAllocator> alloc,
        id<MTLComputePipelineState> pso, uint64_t arg, MTLSize grid) {
    NSError* err = nil;
    MTL4ArgumentTableDescriptor* atd = [MTL4ArgumentTableDescriptor new];
    atd.maxBufferBindCount = 1;
    id<MTL4ArgumentTable> table = [c.dev newArgumentTableWithDescriptor:atd error:&err];
    [table setAddress:arg atIndex:0];
    id<MTL4CommandBuffer> cb = [c.dev newCommandBuffer];
    [cb beginCommandBufferWithAllocator:alloc];
    id<MTL4ComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setArgumentTable:table];
    [enc setComputePipelineState:pso];
    [enc dispatchThreadgroups:grid threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [enc endEncoding];
    [cb endCommandBuffer];
    return cb;
}

// Optional 8-byte buffer copies around the single dispatch (GPU-side
// progress / witness words, see mode gpu).
struct Copy8 { id<MTLBuffer> src; size_t soff; id<MTLBuffer> dst; size_t doff; };

static id<MTL4CommandBuffer> dispatch_copies(Ctx& c, id<MTL4CommandAllocator> alloc,
        id<MTLComputePipelineState> pso, uint64_t arg, MTLSize grid, const Copy8* before,
        const Copy8* after) {
    NSError* err = nil;
    MTL4ArgumentTableDescriptor* atd = [MTL4ArgumentTableDescriptor new];
    atd.maxBufferBindCount = 1;
    id<MTL4ArgumentTable> table = [c.dev newArgumentTableWithDescriptor:atd error:&err];
    [table setAddress:arg atIndex:0];
    id<MTL4CommandBuffer> cb = [c.dev newCommandBuffer];
    [cb beginCommandBufferWithAllocator:alloc];
    id<MTL4ComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setArgumentTable:table];
    if (before) {
        [enc copyFromBuffer:before->src sourceOffset:before->soff toBuffer:before->dst
            destinationOffset:before->doff size:8];
        [enc barrierAfterEncoderStages:MTLStageBlit beforeEncoderStages:MTLStageDispatch
            visibilityOptions:MTL4VisibilityOptionDevice];
    }
    [enc setComputePipelineState:pso];
    [enc dispatchThreadgroups:grid threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    if (after) {
        [enc barrierAfterEncoderStages:MTLStageDispatch beforeEncoderStages:MTLStageBlit
            visibilityOptions:MTL4VisibilityOptionDevice];
        [enc copyFromBuffer:after->src sourceOffset:after->soff toBuffer:after->dst
            destinationOffset:after->doff size:8];
    }
    [enc endEncoding];
    [cb endCommandBuffer];
    return cb;
}

static double fb_start = 0, fb_end = 0, prod_end = 0;

static MTL4CommitOptions* feedback_opts() {
    MTL4CommitOptions* opts = [MTL4CommitOptions new];
    [opts addFeedbackHandler:^(id<MTL4CommitFeedback> fb) {
        fb_t = now_s() - g_t0;
        fb_start = fb.GPUStartTime;
        fb_end = fb.GPUEndTime;
        fb_err = fb.error ? [NSString stringWithFormat:@"%@ (domain %@ code %ld)",
            fb.error.localizedDescription, fb.error.domain, (long)fb.error.code] : nil;
        fb_done.store(1);
    }];
    return opts;
}

static uint32_t sentinel_value(Ctx& c) { return ((volatile uint32_t*)c.out.contents)[0]; }

static void report_feedback() {
    for (int i = 0; i < 200 && !fb_done.load(); i++) usleep(10000);
    printf("commit feedback: %s at %.2f s, error: %s\n", fb_done.load() ? "delivered" : "MISSING",
        fb_t, fb_err ? fb_err.UTF8String : "none");
}

static int mode_cpu(Ctx& c, double limit) {
    id<MTL4CommandQueue> qa = [c.dev newMTL4CommandQueue];
    id<MTL4CommandQueue> qb = [c.dev newMTL4CommandQueue];
    [qb addResidencySet:c.rs];
    id<MTLSharedEvent> PA = [c.dev newSharedEvent], QA = [c.dev newSharedEvent];
    id<MTLSharedEvent> PB = [c.dev newSharedEvent], QB = [c.dev newSharedEvent];
    id<MTL4CommandAllocator> alloc = [c.dev newCommandAllocator];
    id<MTL4CommandBuffer> cb = one_dispatch(c, alloc, c.sentinel, c.out.gpuAddress,
        MTLSizeMake(1, 1, 1));

    g_t0 = now_s();
    [qa waitForEvent:PA value:1];
    [qa signalEvent:QA value:1];
    [qb waitForEvent:PB value:1];
    id<MTL4CommandBuffer> bufs[] = {cb};
    [qb commit:bufs count:1 options:feedback_opts()];
    [qb signalEvent:QB value:1];

    printf("mode cpu, pid %d: consumers queued behind withheld CPU producers; "
        "polling up to %.0f s\n", getpid(), limit);
    fflush(stdout);
    double ta = -1, tb = -1;
    uint32_t sentinel_at_tb = 0;
    double next_report = 10;
    while (now_s() - g_t0 < limit && (ta < 0 || tb < 0)) {
        usleep(50000);
        double t = now_s() - g_t0;
        if (ta < 0 && QA.signaledValue >= 1) {
            ta = t;
            printf("  A: QA advanced at %.2f s while PA=%llu\n", t,
                (unsigned long long)PA.signaledValue);
        }
        if (tb < 0 && QB.signaledValue >= 1) {
            tb = t;
            sentinel_at_tb = sentinel_value(c);
            printf("  B: QB advanced at %.2f s while PB=%llu, sentinel=%u\n", t,
                (unsigned long long)PB.signaledValue, sentinel_at_tb);
        }
        if (t >= next_report) {
            printf("  ... %.0f s: QA=%llu QB=%llu sentinel=%u\n", t,
                (unsigned long long)QA.signaledValue, (unsigned long long)QB.signaledValue,
                sentinel_value(c));
            next_report += 10;
        }
        fflush(stdout);
    }
    double trel = now_s() - g_t0;
    PA.signaledValue = 1;
    PB.signaledValue = 1;
    bool da = [QA waitUntilSignaledValue:1 timeoutMS:10000];
    bool db = [QB waitUntilSignaledValue:1 timeoutMS:10000];
    printf("released producers at %.2f s; drained A=%d B=%d\n", trel, da, db);
    if (ta < 0) printf("result A (wait-only): wait held\n");
    else printf("result A (wait-only): TIMED OUT, consumer released after %.2f s\n", ta);
    if (tb < 0) printf("result B (wait+work): wait held\n");
    else printf("result B (wait+work): TIMED OUT, consumer ran after %.2f s (sentinel %u)\n",
        tb, sentinel_at_tb);
    report_feedback();
    return 0;
}

static int mode_gpu(Ctx& c, double total, int nwaits, int npre) {
    id<MTL4CommandQueue> qp = [c.dev newMTL4CommandQueue];
    id<MTL4CommandQueue> qc = [c.dev newMTL4CommandQueue];
    [qp addResidencySet:c.rs];
    [qc addResidencySet:c.rs];
    id<MTLSharedEvent> PC = [c.dev newSharedEvent], QC = [c.dev newSharedEvent];
    id<MTLSharedEvent> prog = [c.dev newSharedEvent];   // producer progress
    id<MTLSharedEvent> cprog = [c.dev newSharedEvent];  // consumer head-start progress
    const MTLSize grid = MTLSizeMake(64, 1, 1);  // 64 x 256 threads

    // Calibrate (warm): grow the iteration count until one spin command
    // buffer takes >= 50 ms, then scale to ~250 ms.
    SpinArgs* sa = (SpinArgs*)c.args.contents;
    uint32_t iters = 1u << 14;
    double dt = 0;
    id<MTL4CommandAllocator> cal = [c.dev newCommandAllocator];
    for (int round = 1; round <= 8; round++) {
        sa[0] = {c.out.gpuAddress + 64, iters, 0};
        [cal reset];
        id<MTL4CommandBuffer> cb0 = one_dispatch(c, cal, c.spin, c.args.gpuAddress, grid);
        id<MTL4CommandBuffer> b0[] = {cb0};
        double t = now_s();
        [qp commit:b0 count:1];
        [qp signalEvent:prog value:(uint64_t)round];
        if (![prog waitUntilSignaledValue:(uint64_t)round timeoutMS:10000]) {
            fprintf(stderr, "calibration did not finish in 10 s\n");
            return 2;
        }
        dt = std::max(now_s() - t, 1e-4);
        if (dt >= 0.05 || iters >= (1u << 24)) break;
        iters = std::min<uint32_t>(iters * 4, 1u << 24);
    }
    uint32_t cal_iters = iters;
    double scale = 0.25 / dt;
    iters = (uint32_t)std::min<double>((double)(1u << 24), iters * scale);
    int ncb = (int)std::min(80.0, std::max(1.0, total / 0.25));
    printf("mode gpu, pid %d: calibration %.1f ms for %u iters -> %u iters/cb (~250 ms), "
        "%d producer command buffers (~%.0f s)\n", getpid(), dt * 1e3, cal_iters, iters, ncb,
        ncb * 0.25);
    fflush(stdout);

    sa[1] = {c.out.gpuAddress + 64, iters, 0};
    id<MTL4CommandAllocator> pa = [c.dev newCommandAllocator];
    id<MTL4CommandAllocator> ca = [c.dev newCommandAllocator];
    // GPU-side ordering evidence: each producer command buffer ends by
    // copying its index (1..ncb) into a progress word; the consumer's command
    // buffer starts by copying that word into a witness word.
    const size_t kProgress = 1024, kWitness = 2048, kValues = 4096;
    uint64_t* vals = (uint64_t*)((char*)c.args.contents + kValues);
    for (int i = 0; i < 80; i++) vals[i] = (uint64_t)(i + 1);
    ((uint64_t*)((char*)c.out.contents + kProgress))[0] = 0;
    ((uint64_t*)((char*)c.out.contents + kWitness))[0] = 0;
    Copy8 snap = {c.out, kProgress, c.out, kWitness};
    id<MTL4CommandBuffer> sentinel_cb = dispatch_copies(c, ca, c.sentinel, c.out.gpuAddress,
        MTLSizeMake(1, 1, 1), &snap, nullptr);
    npre = std::min(npre, 40);
    id<MTL4CommandAllocator> prea = [c.dev newCommandAllocator];
    id<MTL4CommandBuffer> cpre[40];
    for (int i = 0; i < npre; i++) {
        cpre[i] = one_dispatch(c, prea, c.spin, c.args.gpuAddress + sizeof(SpinArgs), grid);
    }
    printf("consumer: %d spin command buffers ahead, then %d identical waits\n", npre, nwaits);
    // Encode all producer command buffers from one allocator up front.
    id<MTL4CommandBuffer> pcbs[80];
    for (int i = 0; i < ncb; i++) {
        Copy8 prog_write = {c.args, kValues + 8 * (size_t)i, c.out, kProgress};
        pcbs[i] = dispatch_copies(c, pa, c.spin, c.args.gpuAddress + sizeof(SpinArgs), grid,
            nullptr, &prog_write);
    }

    const uint64_t base = prog.signaledValue;
    g_t0 = now_s();
    for (int i = 0; i < ncb; i++) {
        id<MTL4CommandBuffer> b[] = {pcbs[i]};
        if (i == ncb - 1) {
            MTL4CommitOptions* po = [MTL4CommitOptions new];
            [po addFeedbackHandler:^(id<MTL4CommitFeedback> fb) { prod_end = fb.GPUEndTime; }];
            [qp commit:b count:1 options:po];
        } else {
            [qp commit:b count:1];
        }
        [qp signalEvent:prog value:base + 1 + (uint64_t)i];
    }
    [qp signalEvent:PC value:1];
    // Optional consumer-side head start: npre spin command buffers on the
    // consumer queue ahead of its wait (does the timeout start at submission
    // or when the wait reaches the head of the queue?).
    for (int i = 0; i < npre; i++) {
        id<MTL4CommandBuffer> b[] = {cpre[i]};
        [qc commit:b count:1];
        [qc signalEvent:cprog value:(uint64_t)(1 + i)];
    }
    // nwaits identical waits: does each one get its own timeout?
    for (int i = 0; i < nwaits; i++) [qc waitForEvent:PC value:1];
    id<MTL4CommandBuffer> bc[] = {sentinel_cb};
    [qc commit:bc count:1 options:feedback_opts()];
    [qc signalEvent:QC value:1];

    double tq = -1, tp = -1, tpre = npre ? -1 : 0;
    uint64_t prog_at_q = 0;
    uint32_t sent_at_q = 0;
    double limit = (ncb + npre) * 0.25 * 4 + 30;
    double next_report = 2;
    while (now_s() - g_t0 < limit && (tq < 0 || tp < 0)) {
        usleep(20000);
        double tt = now_s() - g_t0;
        if (tq < 0 && QC.signaledValue >= 1) {
            tq = tt;
            prog_at_q = prog.signaledValue;
            sent_at_q = sentinel_value(c);
            printf("  C: QC advanced at %.2f s, PC=%llu, producer at cb %llu/%d, sentinel=%u\n",
                tt, (unsigned long long)PC.signaledValue,
                (unsigned long long)(prog_at_q - base), ncb, sent_at_q);
        }
        if (tpre < 0 && cprog.signaledValue >= (uint64_t)npre) {
            tpre = tt;
            printf("  consumer head-start work finished at %.2f s (wait reaches queue head)\n",
                tt);
        }
        if (tp < 0 && PC.signaledValue >= 1) {
            tp = tt;
            printf("  producer finished at %.2f s\n", tt);
        }
        if (tt >= next_report) {
            printf("  ... %.0f s: producer cb %llu/%d, QC=%llu\n", tt,
                (unsigned long long)(prog.signaledValue - base), ncb,
                (unsigned long long)QC.signaledValue);
            next_report += 2;
        }
        fflush(stdout);
    }
    if (tp < 0) {
        // Bounded work: should never happen; wait (bounded) before exiting.
        printf("producer not finished after %.0f s; waiting up to 60 s more\n", limit);
        [PC waitUntilSignaledValue:1 timeoutMS:60000];
    }
    [QC waitUntilSignaledValue:1 timeoutMS:10000];
    if (tq >= 0 && tp >= 0 && tq + 0.05 < tp) {
        printf("result C: TIMED OUT, consumer ran at %.2f s, %.2f s before the producer "
            "finished (%.2f s)\n", tq, tp - tq, tp);
    } else {
        printf("result C: wait held (consumer %.2f s, producer %.2f s)\n", tq, tp);
    }
    report_feedback();
    for (int i = 0; i < 200 && prod_end == 0; i++) usleep(10000);
    uint64_t witness = ((volatile uint64_t*)((char*)c.out.contents + kWitness))[0];
    printf("witness: consumer saw producer progress %llu of %d -> %s\n",
        (unsigned long long)witness, ncb, witness >= (uint64_t)ncb ? "ordered" : "TIMED-OUT WAIT");
    printf("timestamps: consumer GPUStart - producer last GPUEnd = %+.3f ms "
        "(consumer %.6f..%.6f, producer end %.6f; host now %.6f)\n",
        (fb_start - prod_end) * 1e3, fb_start, fb_end, prod_end,
        (double)clock_gettime_nsec_np(CLOCK_UPTIME_RAW) * 1e-9);
    return 0;
}

int main(int argc, char** argv) {
    const char* mode = argc > 1 ? argv[1] : "cpu";
    double arg = argc > 2 ? atof(argv[2]) : (strcmp(mode, "gpu") == 0 ? 10.0 : 120.0);
    @autoreleasepool {
        Ctx c;
        c.dev = MTLCreateSystemDefaultDevice();
        if (!c.dev || ![c.dev supportsFamily:MTLGPUFamilyMetal4]) {
            fprintf(stderr, "Metal 4 device required\n");
            return 2;
        }
        NSError* err = nil;
        id<MTLLibrary> lib = [c.dev newLibraryWithSource:@(kSrc) options:nil error:&err];
        if (!lib) { fprintf(stderr, "compile: %s\n", err.localizedDescription.UTF8String); return 2; }
        c.sentinel = [c.dev newComputePipelineStateWithFunction:
            [lib newFunctionWithName:@"sentinel"] error:&err];
        c.spin = [c.dev newComputePipelineStateWithFunction:
            [lib newFunctionWithName:@"spin"] error:&err];
        if (!c.sentinel || !c.spin) { fprintf(stderr, "pipeline failed\n"); return 2; }
        c.out = [c.dev newBufferWithLength:16384 options:MTLResourceStorageModeShared];
        c.args = [c.dev newBufferWithLength:16384 options:MTLResourceStorageModeShared];
        memset(c.out.contents, 0, 16384);
        c.rs = [c.dev newResidencySetWithDescriptor:[MTLResidencySetDescriptor new] error:&err];
        [c.rs addAllocation:c.out];
        [c.rs addAllocation:c.args];
        [c.rs commit];
        int nwaits = argc > 3 ? atoi(argv[3]) : 1;
        int npre = argc > 4 ? atoi(argv[4]) : 0;
        int rc = strcmp(mode, "gpu") == 0 ? mode_gpu(c, arg, std::max(1, nwaits), npre)
                                          : mode_cpu(c, arg);
        printf("final sentinel=%u\n", sentinel_value(c));
        return rc;
    }
}
