// Metal backend runtime: unified-memory allocator, buffer registry, Metal 4
// stream/queue model and the CUDA runtime shims declared in
// compat/cuda_runtime.h. See runtime.h for the model.
//
// Stream structure, constant-ring argument passing, residency-set handling and
// the "drain every queue before freeing GPU-visible memory" rule are adapted
// from valo-metal (MIT, Valtteri Valo) src/metal/platform.{h,mm} @ 1e7e9dd1.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <dispatch/dispatch.h>
#include <mach-o/getsect.h>
#include <mach-o/ldsyms.h>
#include <mach-o/dyld.h>
#include <mach/mach_time.h>
#include <stdarg.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "runtime.h"
#include "compat/curand.h"

namespace puf_mtl {

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

static std::atomic<void (*)(const char*)> g_fatal_handler{nullptr};

void set_fatal_handler(void (*fn)(const char* msg)) { g_fatal_handler.store(fn); }

void fatal(const char* fmt, ...) {
    char msg[1024];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(msg, sizeof(msg), fmt, ap);
    va_end(ap);
    fprintf(stderr, "metal: %s\n", msg);
    fflush(stderr);
    if (auto fn = g_fatal_handler.load()) fn(msg);
    abort();
}

static double now_seconds() {
    static mach_timebase_info_data_t tb = {0, 0};
    if (tb.denom == 0) mach_timebase_info(&tb);
    return (double)mach_absolute_time() * tb.numer / tb.denom * 1e-9;
}

static uint64_t now_ns() {
    static mach_timebase_info_data_t tb = {0, 0};
    if (tb.denom == 0) mach_timebase_info(&tb);
    return mach_absolute_time() * tb.numer / tb.denom;
}

// PUF_METAL_TRACE=path: host/GPU timeline of every commit, blocking host wait
// and trace_mark() (profiling only; CSV written at exit).
struct TraceRec {
    int kind, sid, tag, n;  // kind: 0 commit, 1 wait, 2 mark
    uint64_t v;
    double t0, t1, g0, g1;
};
static TraceRec* g_trace = nullptr;
static std::atomic<size_t> g_trace_n{0};
static constexpr size_t kTraceMax = 1 << 22;
static const char* g_trace_path = nullptr;

static TraceRec* trace_new() {
    size_t i = g_trace_n.fetch_add(1, std::memory_order_relaxed);
    return i < kTraceMax ? &g_trace[i] : nullptr;
}

static void trace_dump() {
    size_t n = std::min(g_trace_n.load(), kTraceMax);
    FILE* f = fopen(g_trace_path, "w");
    if (!f) return;
    fprintf(f, "kind,sid,tag,n,v,t0,t1,g0,g1\n");
    for (size_t i = 0; i < n; i++) {
        const TraceRec& r = g_trace[i];
        fprintf(f, "%d,%d,%d,%d,%llu,%.7f,%.7f,%.7f,%.7f\n", r.kind, r.sid, r.tag, r.n,
            (unsigned long long)r.v, r.t0, r.t1, r.g0, r.g1);
    }
    fclose(f);
}

// ---------------------------------------------------------------------------
// Context
// ---------------------------------------------------------------------------

static constexpr int kSlots = 4;                         // command buffers in flight per stream
static constexpr size_t kRingBytes = 1 << 20;            // per-slot constant ring
static constexpr int kMaxDispatchPerCommit = 256;        // keep the GPU fed
static constexpr NSUInteger kWaitTimeoutMs = 300000;
static constexpr size_t kPage = 16384;
static constexpr int kMaxAllocs = 1 << 14;
static constexpr int kStampSlots = 256;                  // timestamp heap entries per stream
static constexpr int kFbRing = 1024;                     // commit-feedback ring per stream
static constexpr int kWitness = 1024;                    // witness slots per stream
// Per-stream GPU-visible word buffer (uint64 indices): progress word, per-slot
// value sources for it, then the witness slots.
static constexpr int kWordProgress = 0;
static constexpr int kWordValue = 1;
static constexpr int kWordWitness = 64;
static_assert(kWordValue + kSlots <= kWordWitness, "word layout");
// Host-gated commits (gate_arm): per slot, the grid the host releases in
// gate_open (stage, 2 words: x, y, z, 0) and the grid the gated dispatch
// reads (live, 2 words), copied from stage behind the gate's queue wait.
static constexpr int kWordGate = kWordWitness + kWitness;
static constexpr int kWordsTotal = kWordGate + 4 * kSlots;

struct Allocation {
    std::atomic<char*> base{nullptr};
    std::atomic<size_t> size{0};
    uint64_t gpu = 0;
    id<MTLBuffer> buffer = nil;
};

struct Ctx {
    id<MTLDevice> device = nil;
    id<MTLLibrary> library = nil;
    id<MTLLibrary> wide_library = nil;
    id<MTLLibrary> rollout_library = nil;  // lazy; grouped entry points only
    id<MTLResidencySet> residency = nil;
    MTLSharedEventListener* listener = nil;
    std::mutex residency_mu;

    Allocation allocs[kMaxAllocs];
    std::atomic<int> num_allocs{0};
    std::mutex alloc_mu;

    std::mutex pipe_mu;
    std::unordered_map<std::string, Pipeline*> pipes;

    std::mutex streams_mu;
    std::vector<Stream*> streams;
    Stream* default_stream = nullptr;

    double ts_ns_per_tick = 1.0;  // GPU timestamp (mach) ticks -> ns
};

// Counters kept per stream and summed for the global view.
struct Counters {
    std::atomic<uint64_t> dispatches{0}, commits{0}, syncs{0}, barriers{0};
    std::atomic<uint64_t> wait_ns{0}, encode_ns{0}, gpu_ns{0}, gated{0};
    std::atomic<uint64_t> spin_hit{0}, spin_miss{0};
    std::atomic<uint64_t> strict_feedback_joins{0};
};

static Ctx* g = nullptr;
static std::once_flag g_once;

struct Pipeline {
    id<MTLComputePipelineState> pso;
    std::string name;
    // PUF_METAL_KPROF=1: per-kernel GPU time (precise timestamps around
    // every dispatch; serializes encoders, so only for attribution).
    std::atomic<uint64_t> kp_ns{0}, kp_n{0};
};

static bool g_kprof = false;
static bool g_kprof_shapes = false;  // PUF_METAL_KPROF=2: also per (stream, grid)
static std::mutex g_kprof_mu;
static std::unordered_map<std::string, std::pair<uint64_t, uint64_t>> g_kprof_by_shape;
static constexpr int kKprofSlots = 4096;  // timestamp entries per stream (2 per dispatch)

// ---------------------------------------------------------------------------
// Streams
// ---------------------------------------------------------------------------

struct Slot {
    id<MTL4CommandAllocator> allocator = nil;
    id<MTL4CommandBuffer> cmd = nil;
    // Witness snapshots, committed ahead of `cmd` when it has queue waits.
    id<MTL4CommandAllocator> pre_allocator = nil;
    id<MTL4CommandBuffer> pre_cmd = nil;
    id<MTLBuffer> ring = nil;
    uint64_t done_value = 0;  // event value signaled when this slot's work completes
};

struct PendingStamp {
    unsigned long long* dst;
    uint64_t value;  // stream event value after which the heap entry is valid
    int index;
};

// A cross-stream dependency: producer stream p must reach `value`.
struct Dep {
    Stream* p;
    uint64_t value;
};

// A GPU queue wait of commit `at` of the owning stream on producer p reaching
// `value`; witness slot widx holds the producer's progress word as the
// consumer saw it right behind the wait.
struct DepAt {
    uint64_t at;
    Stream* p;
    uint64_t value;
    int widx;
};

struct Stream {
    std::mutex mu;
    id<MTL4CommandQueue> queue = nil;
    id<MTLSharedEvent> event = nil;
    id<MTL4ArgumentTable> table = nil;
    id<MTL4ComputeCommandEncoder> enc = nil;
    Slot slots[kSlots];
    int cur = 0;
    bool open = false;
    int ndispatch = 0;
    // concurrent_begin/end: dispatches after the window's first one skip the
    // inter-dispatch barrier (-1 = no window open).
    int conc_n = -1;
    size_t ring_off = 0;
    uint64_t committed = 0;  // last value signaled by a commit on this queue
    bool blocking = false;   // participates in legacy default-stream syncs
    bool is_default = false;
    bool host_wait_only = false;  // immutable after the first enqueue
    std::atomic<uint64_t> spin_ns{0};  // stream_wait_spin
    int sid = 0;  // creation order (0 = default stream)
    std::vector<Dep> waits;  // cross-stream dependencies of the next commit
    Counters ctr;
    id<MTL4CounterHeap> stamps = nil;  // lazily created by stamp()
    int stamp_next = 0;
    std::vector<PendingStamp> pending_stamps;
    id<MTL4CounterHeap> kheap = nil;  // PUF_METAL_KPROF
    int knext = 0;
    struct KPending { Pipeline* p; int idx; uint64_t value; dim3 grid; };
    std::vector<KPending> kpend;
    // GPU words (see kWord*): progress word written at the end of every
    // commit, witness slots written by the pre command buffers.
    id<MTLBuffer> words = nil;
    volatile uint64_t* wcpu = nullptr;
    // Dependency verification (verify) and stall diagnostics (dump_streams).
    // diag_mu is a leaf lock: it is never held while taking another lock or
    // waiting, so any thread may take it.
    std::mutex diag_mu;
    uint64_t diag_committed = 0;
    // GPU waits whose transitive verification has not finished. Entries are
    // removed only once their whole dependency closure verified, so every
    // observer either sees them or the verification is complete.
    std::vector<DepAt> deps;
    std::atomic<uint64_t> dep_min_at{UINT64_MAX};  // min deps[].at (fast path)
    uint64_t witness_at[kWitness] = {};  // commit owning each witness slot (0 = free)
    int wnext = 0;
    // Commit feedback: fb_done[v % kFbRing] == v once commit v reported
    // success; fb_confirmed is the highest v with every commit <= v
    // reported. A failed commit aborts in its handler, so it never becomes
    // confirmed.
    std::atomic<uint64_t> fb_done[kFbRing];
    std::atomic<uint64_t> fb_confirmed{0};
    std::atomic<uint64_t> fb_owed{0};  // highest value handed to the host unconfirmed
    std::atomic<int> fb_waiters{0};
    std::mutex fb_mu;
    std::condition_variable fb_cv;
    // Host-gated commits (gate_arm / gate_open / gate_ran). Under mu except
    // where noted.
    id<MTLSharedEvent> gate = nil;  // signaled by the host only
    uint64_t gate_target = 0;       // gate value the latest gated commit waits for
    bool gate_armed = false;        // the next commit is gated
    int gate_ndispatch = 0;         // dispatches encoded into the armed buffer
    int gate_slot = -1;             // slot of the gated commit waiting for gate_open
    uint64_t gate_value[kSlots] = {};    // commit value of the gated buffer in each slot
    uint32_t gate_grid[kSlots][3] = {};  // its real grid
    // Commit whose feedback MTL4 holds back while the gated commit behind it
    // waits for gate_open (0 = none); strict joins skip it (join_all_feedback).
    std::atomic<uint64_t> gate_hold{0};
    std::atomic<bool> gated{false};  // ever gated: lagged feedback one step deeper
    std::mutex owed_mu;              // gated streams: fb_owed / fb_owed_old
    uint64_t fb_owed_old = 0;

    uint64_t completed() const { return event.signaledValue; }
};

static void legacy_deps(Stream* s);

static Counters g_ctr;  // process-wide sums of every stream's counters

static inline void count(std::atomic<uint64_t> Counters::*field, Stream* s, uint64_t n = 1) {
    (s->ctr.*field).fetch_add(n, std::memory_order_relaxed);
    (g_ctr.*field).fetch_add(n, std::memory_order_relaxed);
}

// Sum of every stream's completed value: advances whenever any GPU work
// finishes, so a slow (e.g. contended) GPU is not mistaken for a deadlock.
static uint64_t gpu_progress() {
    std::lock_guard<std::mutex> lk(g->streams_mu);
    uint64_t sum = 0;
    for (Stream* o : g->streams) sum += o->completed();
    return sum;
}

// Prints every stream's committed/completed values and the cross-stream
// GPU waits whose verification has not finished.
static void dump_streams() {
    std::vector<Stream*> all;
    {
        std::lock_guard<std::mutex> lk(g->streams_mu);
        all = g->streams;
    }
    for (Stream* o : all) {
        std::lock_guard<std::mutex> lk(o->diag_mu);
        uint64_t done = o->completed();
        fprintf(stderr, "metal:   s%d%s committed %llu completed %llu feedback %llu\n", o->sid,
            o->is_default ? " (default)" : "", (unsigned long long)o->diag_committed,
            (unsigned long long)done, (unsigned long long)o->fb_confirmed.load());
        for (const DepAt& w : o->deps) {
            fprintf(stderr, "metal:     commit %llu waits for s%d >= %llu (now %llu, "
                "witness %llu)\n", (unsigned long long)w.at, w.p->sid,
                (unsigned long long)w.value, (unsigned long long)w.p->completed(),
                (unsigned long long)o->wcpu[kWordWitness + w.widx]);
        }
    }
    fflush(stderr);
}

// ---------------------------------------------------------------------------
// Host-side verification of GPU completion
//
// MTL4 queue waits (waitForEvent:value:) are not unconditional: measured on
// M4 / macOS 26.4 (tests/metal/run_test_wait_timeout.sh), a queue wait is
// released about 4.0 s after it was submitted, even while the producer queue
// is still busy making progress, and the work behind it runs. Nothing reports
// it: the commit feedback has no error and nothing is logged. On a slow or
// contended GPU a consumer could therefore read a producer's buffers early.
// Commit errors (fb.error) are delivered asynchronously on a feedback queue,
// possibly after the event value that the host waits on has advanced.
//
// Evidence of order is recorded on the GPU. GPU-waited commits end by writing their
// value to their stream's progress word (after a barrier behind all of their
// work). A commit with queue waits is preceded, behind the same waits, by a
// small command buffer that copies each producer's progress word into a
// witness slot of the consumer; the consumer's work starts only after that
// copy (queue barrier). So witness >= waited value iff the producer's work
// had finished when the consumer was released; a timed-out wait leaves an
// older value there, whatever the producer does afterwards.
// Host-only streams omit that word; add_wait_locked instead completes and
// verifies such a producer on the host before any consumer work is enqueued.
//
// Whenever the host observes that stream s completed value v (observe),
// before returning to the caller it
//   1. verifies (verify) every recorded GPU wait of the commits of s <= v:
//      its witness must have reached the waited value, else the wait timed
//      out -> fatal; then the same for the producers up to the waited values
//      (transitively, iteratively, revisits are skipped). Dependencies whose
//      producer had already completed when the consumer committed need no
//      GPU wait; their producers are verified right then and nothing is
//      recorded. Records are removed only after the whole closure verified,
//      so a concurrent observer either sees them or the work is done;
//   2. joins commit feedback (the handler aborts on error, so a failed
//      commit never becomes confirmed):
//      - strict (cudaDeviceSynchronize, blocking cudaMemcpy/cudaMemset,
//        cudaFree, cuRAND host calls, or every sync with
//        PUF_METAL_STRICT_FEEDBACK=1): waits until feedback for every
//        completed commit of every stream was delivered;
//      - lagged (cudaStreamSynchronize, event syncs, internal waits): feedback
//        arrives ~50 us after the event signal, a measurable cost on the
//        per-step rollout syncs, so these do not wait for it for v itself;
//        they first block until every value previously handed to the host
//        unconfirmed has been confirmed (when they observe a newer value),
//        for s and the producers visited in 1. A failed commit thus aborts
//        the process from its handler ~50 us after the host saw it complete,
//        and at the latest at the next strict sync; results leave the
//        process (checkpoints, logs) through strict paths.
// ---------------------------------------------------------------------------

[[noreturn]] static void wait_timed_out(Stream* s, const DepAt& d, uint64_t witness) {
    dump_streams();
    fatal("cross-stream wait timed out: s%d commit %llu ran before producer stream s%d "
        "reached value %llu (GPU witness saw %llu, now %llu). Metal releases a queue wait "
        "about 4 s after submission even if the producer is still running, so dependent GPU "
        "work read stale data; the GPU is too slow, contended or hung for this workload.",
        s->sid, (unsigned long long)d.at, d.p->sid, (unsigned long long)d.value,
        (unsigned long long)witness, (unsigned long long)d.p->completed());
}

// Advances s->fb_confirmed through consecutively reported commits (<= v).
static uint64_t fb_advance(Stream* s, uint64_t v) {
    uint64_t c0 = s->fb_confirmed.load(std::memory_order_acquire);
    uint64_t c = c0;
    while (c < v && s->fb_done[(c + 1) % kFbRing].load(std::memory_order_acquire) == c + 1) c++;
    if (c == c0) return c;
    uint64_t cur = c0;
    while (cur < c && !s->fb_confirmed.compare_exchange_weak(cur, c,
            std::memory_order_acq_rel, std::memory_order_acquire)) {
    }
    return std::max(c, cur);
}

// Returns once commit feedback for every commit of s <= v has been
// delivered without error. The commits must already have completed or be
// about to (the caller observed or waited for them).
static void join_feedback(Stream* s, uint64_t v, const char* where) {
    if (s->fb_confirmed.load(std::memory_order_acquire) >= v) return;
    if (fb_advance(s, v) >= v) return;
    uint64_t t0 = now_ns();
    bool warned = false;
    s->fb_waiters.fetch_add(1, std::memory_order_seq_cst);
    {
        std::unique_lock<std::mutex> lk(s->fb_mu);
        while (fb_advance(s, v) < v) {
            s->fb_cv.wait_for(lk, std::chrono::milliseconds(2));
            uint64_t dt = now_ns() - t0;
            if (!warned && dt >= 5000000000ull) {
                warned = true;
                fprintf(stderr, "metal: commit feedback for s%d value %llu (%s) not delivered "
                    "after 5 s (completed %llu, confirmed %llu)\n", s->sid,
                    (unsigned long long)v, where, (unsigned long long)s->completed(),
                    (unsigned long long)s->fb_confirmed.load());
            }
            if (dt >= kWaitTimeoutMs * 1000000ull) {
                lk.unlock();
                dump_streams();
                fatal("commit feedback for s%d value %llu (%s) not delivered within %lus",
                    s->sid, (unsigned long long)v, where,
                    (unsigned long)(kWaitTimeoutMs / 1000));
            }
        }
    }
    s->fb_waiters.fetch_sub(1, std::memory_order_seq_cst);
    count(&Counters::wait_ns, s, now_ns() - t0);
}

static bool g_strict_feedback = false;  // PUF_METAL_STRICT_FEEDBACK=1

// Lagged feedback (2. above) for s up to v.
static void settle_feedback(Stream* s, uint64_t v, const char* where) {
    if (s->gated.load(std::memory_order_acquire)) {
        // A gated stream's latest commit keeps its feedback until the next
        // gate opens (MTL4 holds a commit's feedback back while the queue's
        // next commit is blocked on a queue wait), i.e. until after the host
        // acted on it. Settle one observation later: when observing a newer
        // value, the value handed out before the previous one.
        std::lock_guard<std::mutex> lk(s->owed_mu);
        uint64_t owed = s->fb_owed.load(std::memory_order_acquire);
        if (v <= owed) return;
        if (s->fb_owed_old) join_feedback(s, s->fb_owed_old, where);
        if (s->fb_confirmed.load(std::memory_order_acquire) >= v || fb_advance(s, v) >= v) {
            s->fb_owed_old = 0;
            s->fb_owed.store(v, std::memory_order_release);
            return;
        }
        s->fb_owed_old = owed > s->fb_confirmed.load(std::memory_order_acquire) ? owed : 0;
        s->fb_owed.store(v, std::memory_order_release);
        return;
    }
    // Re-observing values already handed out (e.g. event queries right after
    // a stream sync) does not block; observing a newer value first settles
    // the previous hand-off, whose feedback has normally long arrived.
    uint64_t owed = s->fb_owed.load(std::memory_order_acquire);
    if (v <= owed) return;
    if (owed) join_feedback(s, owed, where);
    if (s->fb_confirmed.load(std::memory_order_acquire) >= v || fb_advance(s, v) >= v) return;
    while (owed < v && !s->fb_owed.compare_exchange_weak(owed, v,
            std::memory_order_acq_rel, std::memory_order_acquire)) {
    }
}

// Strict feedback (2. above): every completed commit of every stream.
static void join_all_feedback(const char* where, bool strict_override = false) {
    // Process-wide only: a join covers every stream, so attributing it to the
    // stream whose observation triggered it would be misleading. Keep the
    // default hot path unchanged; the opt-in increment is relaxed.
    if (strict_override) {
        g_ctr.strict_feedback_joins.fetch_add(1, std::memory_order_relaxed);
    }
    std::vector<Stream*> all;
    {
        std::lock_guard<std::mutex> lk(g->streams_mu);
        all = g->streams;
    }
    for (Stream* x : all) {
        uint64_t v = x->completed();
        // Held back behind a pending gate (the gate's opener may be the very
        // thread that needs this join): covered by the lagged settle of the
        // stream's next observation instead.
        uint64_t h = x->gate_hold.load(std::memory_order_acquire);
        if (h && v >= h) v = h - 1;
        join_feedback(x, v, where);
    }
}

static constexpr NSUInteger kWaitSliceMs = 5000;
static constexpr uint64_t kWaitWarnNs = 60ull * 1000000000ull;

// Blocks until s completes value v, without verification. Gives up only
// after the whole GPU has made no progress for kWaitTimeoutMs; reports long
// waits once.
static void raw_wait(Stream* s, uint64_t v, const char* where) {
    if (s->completed() >= v) return;
    uint64_t t0 = now_ns();
    if (uint64_t spin = s->spin_ns.load(std::memory_order_relaxed)) {
        // Bounded poll (stream_wait_spin); isb is the cheapest pause that
        // keeps the poll responsive on Apple cores.
        uint64_t end = t0 + spin;
        bool hit = false;
        for (;;) {
            for (int i = 0; i < 16; i++) __builtin_arm_isb(15);
            if (s->completed() >= v) { hit = true; break; }
            if (now_ns() >= end) break;
        }
        count(hit ? &Counters::spin_hit : &Counters::spin_miss, s);
    }
    uint64_t last_change = t0, progress = gpu_progress();
    bool warned = false;
    while (s->completed() < v && ![s->event waitUntilSignaledValue:v timeoutMS:kWaitSliceMs]) {
        uint64_t now = now_ns(), p = gpu_progress();
        if (p != progress) {
            progress = p;
            last_change = now;
        }
        if (!warned && now - t0 >= kWaitWarnNs) {
            warned = true;
            fprintf(stderr, "metal: host has waited %.0f s in %s for s%d >= %llu "
                "(completed %llu, GPU %s)\n", (now - t0) * 1e-9, where, s->sid,
                (unsigned long long)v, (unsigned long long)s->completed(),
                now - last_change < 2 * kWaitSliceMs * 1000000ull ? "progressing" : "idle");
            dump_streams();
        }
        if (now - last_change >= kWaitTimeoutMs * 1000000ull) {
            dump_streams();
            fatal("GPU made no progress for %lus while waiting in %s (s%d value %llu, "
                "completed %llu)", (unsigned long)(kWaitTimeoutMs / 1000), where, s->sid,
                (unsigned long long)v, (unsigned long long)s->completed());
        }
    }
    count(&Counters::syncs, s);
    uint64_t t1 = now_ns();
    count(&Counters::wait_ns, s, t1 - t0);
    if (g_trace) {
        if (TraceRec* r = trace_new()) *r = {1, s->sid, 0, 0, v, t0 * 1e-9, t1 * 1e-9, 0, 0};
    }
}

// 1. above: verifies the recorded GPU waits of the commits of s <= u (which
// must have completed) and, transitively, of their producers. lagged: also
// settles lagged feedback of the producers visited. Takes diag_mu of the
// streams visited (one at a time) and may wait for producers whose witness
// shows their work finished (their signal is imminent); callers may hold
// stream mutexes.
static void verify(Stream* s, uint64_t u, const char* where, bool lagged) {
    if (s->dep_min_at.load(std::memory_order_acquire) > u) return;
    struct Node { Stream* s; uint64_t upto; };
    struct Seen { DepAt d; uint64_t witness; };
    std::vector<Node> work{{s, u}}, done;
    std::vector<Seen> seen;
    while (!work.empty()) {
        Node n = work.back();
        work.pop_back();
        // Each stream once per value reached (cycles, diamonds).
        Node* m = nullptr;
        for (Node& x : done) {
            if (x.s == n.s) m = &x;
        }
        if (m && m->upto >= n.upto) continue;
        if (m) m->upto = n.upto;
        else done.push_back(n);
        if (n.s->dep_min_at.load(std::memory_order_acquire) > n.upto) continue;
        seen.clear();
        {
            // Witnesses are read while their records (and slots) are held.
            std::lock_guard<std::mutex> lk(n.s->diag_mu);
            for (const DepAt& d : n.s->deps) {
                if (d.at <= n.upto) seen.push_back({d, n.s->wcpu[kWordWitness + d.widx]});
            }
        }
        for (const Seen& x : seen) {
            if (x.witness < x.d.value) wait_timed_out(n.s, x.d, x.witness);
            // The producer's work up to the value finished before the
            // consumer started; only its signal may still be in flight.
            raw_wait(x.d.p, x.d.value, where);
            work.push_back({x.d.p, x.d.value});
        }
    }
    // The whole closure verified: retire the records.
    for (const Node& n : done) {
        if (lagged && n.s != s) settle_feedback(n.s, n.upto, where);
        if (n.s->dep_min_at.load(std::memory_order_acquire) > n.upto) continue;
        std::lock_guard<std::mutex> lk(n.s->diag_mu);
        uint64_t mn = UINT64_MAX;
        size_t keep = 0;
        for (size_t i = 0; i < n.s->deps.size(); i++) {
            const DepAt& d = n.s->deps[i];
            if (d.at <= n.upto) {
                if (n.s->witness_at[d.widx] == d.at) n.s->witness_at[d.widx] = 0;
                continue;
            }
            mn = std::min(mn, d.at);
            n.s->deps[keep++] = d;
        }
        n.s->deps.resize(keep);
        n.s->dep_min_at.store(mn, std::memory_order_release);
    }
}

// The host has seen s reach value v (and is about to act on it): verify
// dependencies and feedback as described above. Callers may hold stream
// mutexes.
static void observe(Stream* s, uint64_t v, const char* where, bool strict) {
    bool strict_override = g_strict_feedback && !strict;
    strict = strict || g_strict_feedback;
    verify(s, v, where, !strict);
    if (strict) {
#ifdef PUF_METAL_MUTATE_SKIP_STRICT_FEEDBACK
        // Test-only mutation: retain ordinary strict API joins, but remove the
        // joins added by PUF_METAL_STRICT_FEEDBACK. The alternate-path gate
        // must reject this build from its zero strict_feedback_joins count.
        if (!strict_override) join_all_feedback(where);
#else
        join_all_feedback(where, strict_override);
#endif
    }
    else settle_feedback(s, v, where);
}

// Blocks until s completes value v, then observes it.
static void wait_value(Stream* s, uint64_t v, const char* where, bool strict = false) {
    if (v == 0) {
        bool strict_override = g_strict_feedback && !strict;
#ifdef PUF_METAL_MUTATE_SKIP_STRICT_FEEDBACK
        if (strict && !strict_override) join_all_feedback(where);
#else
        if (strict || g_strict_feedback) join_all_feedback(where, strict_override);
#endif
        return;
    }
    raw_wait(s, v, where);
    observe(s, v, where, strict);
}

static void on_feedback(Stream* s, uint64_t v, id<MTL4CommitFeedback> fb) {
    if (fb.error) {
        fatal("GPU command buffer failed (s%d commit %llu): %s", s->sid,
            (unsigned long long)v, fb.error.localizedDescription.UTF8String);
    }
    double span = fb.GPUEndTime - fb.GPUStartTime;
    if (span > 0) count(&Counters::gpu_ns, s, (uint64_t)(span * 1e9));
    s->fb_done[v % kFbRing].store(v, std::memory_order_seq_cst);
    if (s->fb_waiters.load(std::memory_order_seq_cst) > 0) {
        std::lock_guard<std::mutex> lk(s->fb_mu);
        s->fb_cv.notify_all();
    }
}

static Stream* stream_create(bool blocking, bool is_default) {
    Stream* s = new Stream();
    s->blocking = blocking;
    s->is_default = is_default;
    for (int i = 0; i < kFbRing; i++) s->fb_done[i].store(0, std::memory_order_relaxed);
    s->queue = [g->device newMTL4CommandQueue];
    if (!s->queue) fatal("newMTL4CommandQueue failed (Metal 4 GPU required)");
    [s->queue addResidencySet:g->residency];
    s->event = [g->device newSharedEvent];
    s->event.signaledValue = 0;
    MTL4ArgumentTableDescriptor* atd = [MTL4ArgumentTableDescriptor new];
    atd.maxBufferBindCount = 31;
    NSError* err = nil;
    s->table = [g->device newArgumentTableWithDescriptor:atd error:&err];
    if (!s->table) fatal("newArgumentTableWithDescriptor failed");
    s->words = [g->device newBufferWithLength:kWordsTotal * sizeof(uint64_t)
                                      options:MTLResourceStorageModeShared];
    if (!s->words) fatal("failed to create stream word buffer");
    s->wcpu = (volatile uint64_t*)s->words.contents;
    memset((void*)s->wcpu, 0, s->words.length);
    {
        std::lock_guard<std::mutex> lk(g->residency_mu);
        for (int i = 0; i < kSlots; i++) {
            s->slots[i].allocator = [g->device newCommandAllocator];
            s->slots[i].cmd = [g->device newCommandBuffer];
            s->slots[i].pre_allocator = [g->device newCommandAllocator];
            s->slots[i].pre_cmd = [g->device newCommandBuffer];
            s->slots[i].ring = [g->device newBufferWithLength:kRingBytes
                                                      options:MTLResourceStorageModeShared];
            if (!s->slots[i].allocator || !s->slots[i].cmd || !s->slots[i].ring
                    || !s->slots[i].pre_allocator || !s->slots[i].pre_cmd) {
                fatal("failed to create command buffer slot");
            }
            [g->residency addAllocation:s->slots[i].ring];
        }
        [g->residency addAllocation:s->words];
        [g->residency commit];
    }
    std::lock_guard<std::mutex> lk(g->streams_mu);
    s->sid = (int)g->streams.size();
    g->streams.push_back(s);
    return s;
}

// Requires s->mu.
static void begin_locked(Stream* s) {
    if (s->open) return;
    Slot& slot = s->slots[s->cur];
    if (s->gate_slot == s->cur) {
        // Only the commit right after gate_arm may be gated, and one at a time.
        fatal("s%d: command buffer slot of a gated commit reused before gate_open", s->sid);
    }
    wait_value(s, slot.done_value, "command buffer slot reuse");
    s->gate_value[s->cur] = 0;
    [slot.allocator reset];
    [slot.cmd beginCommandBufferWithAllocator:slot.allocator];
    s->enc = [slot.cmd computeCommandEncoder];
    [s->enc setArgumentTable:s->table];
    // Order against everything previously committed to this queue, including
    // the witness copies (blit) of this commit.
    [s->enc barrierAfterQueueStages:MTLStageDispatch | MTLStageBlit
                       beforeStages:MTLStageDispatch | MTLStageBlit
                  visibilityOptions:MTL4VisibilityOptionDevice];
    count(&Counters::barriers, s);
    s->open = true;
    s->ndispatch = 0;
    s->ring_off = 0;
}

// Requires s->mu. Removes the queued dependencies whose producers have
// already completed (no GPU wait needed) after verifying those producers.
// Returns true if none are left.
static bool settle_waits_locked(Stream* s) {
    if (s->waits.empty()) return true;
    size_t keep = 0;
    for (size_t i = 0; i < s->waits.size(); i++) {
        Dep w = s->waits[i];
        if (w.p->completed() >= w.value) {
            verify(w.p, w.value, "cross-stream dependency", false);
        } else {
            s->waits[keep++] = w;
        }
    }
    s->waits.resize(keep);
    return keep == 0;
}

// Requires s->mu. Encodes GPU queue waits for the queued dependencies of
// commit `at` that are still pending, records them with a witness slot each
// and returns the command buffer that fills the witnesses (nil if none).
// Producers that completed meanwhile are verified instead.
static id<MTL4CommandBuffer> submit_waits_locked(Stream* s, uint64_t at, bool gated) {
    // Prune: verify the waits of commits that completed meanwhile.
    verify(s, s->completed(), "commit", false);
    if (s->waits.empty() && !gated) return nil;
    struct W { Stream* p; int widx; };
    thread_local std::vector<W> ws;
    thread_local std::vector<Dep> sat;
    ws.clear();
    sat.clear();
    {
        std::lock_guard<std::mutex> lk(s->diag_mu);
        s->diag_committed = at;
        for (const Dep& w : s->waits) {
            if (w.p->completed() >= w.value) {
                sat.push_back(w);
                continue;
            }
            int widx = s->wnext;
            if (s->witness_at[widx] != 0) {
                fatal("s%d: too many pending cross-stream dependencies (%d)", s->sid, kWitness);
            }
            s->wnext = (s->wnext + 1) % kWitness;
            s->witness_at[widx] = at;
            s->wcpu[kWordWitness + widx] = 0;
            [s->queue waitForEvent:w.p->event value:w.value];
            s->deps.push_back({at, w.p, w.value, widx});
            if (at < s->dep_min_at.load(std::memory_order_relaxed)) {
                s->dep_min_at.store(at, std::memory_order_release);
            }
            ws.push_back({w.p, widx});
        }
    }
    s->waits.clear();
    for (const Dep& w : sat) verify(w.p, w.value, "cross-stream dependency", false);
    // The gate goes last: the host opens it only after its own work, so
    // cross-stream producers are not held up behind it.
    if (gated) [s->queue waitForEvent:s->gate value:++s->gate_target];
    if (ws.empty()) return nil;
    Slot& slot = s->slots[s->cur];  // free: the main command buffer is in it
    [slot.pre_allocator reset];
    [slot.pre_cmd beginCommandBufferWithAllocator:slot.pre_allocator];
    id<MTL4ComputeCommandEncoder> e = [slot.pre_cmd computeCommandEncoder];
    for (const W& w : ws) {
        [e copyFromBuffer:w.p->words sourceOffset:kWordProgress * sizeof(uint64_t)
                 toBuffer:s->words destinationOffset:(kWordWitness + w.widx) * sizeof(uint64_t)
                     size:sizeof(uint64_t)];
    }

    [e endEncoding];
    [slot.pre_cmd endCommandBuffer];
    return slot.pre_cmd;
}

// Requires s->mu. Next commit value; keeps the feedback ring from lapping.
static uint64_t next_commit_locked(Stream* s) {
    if (s->committed == UINT64_MAX) fatal("s%d: commit counter overflow", s->sid);
    uint64_t v = s->committed + 1;
    if (v > (uint64_t)kFbRing) join_feedback(s, v - kFbRing, "feedback ring reuse");
    return v;
}

// Requires s->mu. Non-blocking: submits the open command buffer. Queued
// cross-stream waits whose producers have not completed get a commit of
// their own if nothing else is open, so later syncs and event records
// observe (and verify) them.
static void commit_locked(Stream* s) {
    if (!s->open) {
        if (settle_waits_locked(s)) return;
        begin_locked(s);
    }
    Slot& slot = s->slots[s->cur];
    uint64_t v = next_commit_locked(s);
    // GPU consumers need this witness; host-only streams use their event.
    if (!s->host_wait_only) {
        s->wcpu[kWordValue + s->cur] = v;
        [s->enc barrierAfterEncoderStages:MTLStageDispatch | MTLStageBlit
                  beforeEncoderStages:MTLStageBlit
                    visibilityOptions:MTL4VisibilityOptionDevice];
        [s->enc copyFromBuffer:s->words sourceOffset:(kWordValue + s->cur) * sizeof(uint64_t)
                  toBuffer:s->words destinationOffset:kWordProgress * sizeof(uint64_t)
                      size:sizeof(uint64_t)];
    }
    [s->enc endEncoding];
    s->enc = nil;
    [slot.cmd endCommandBuffer];
    // An armed stream gates the command buffer holding its dispatch (an
    // empty one committed meanwhile, e.g. for another thread's sync, is not).
    bool gated = s->gate_armed && s->gate_ndispatch > 0;
    if (gated) {
        s->gate_armed = false;
        s->gate_ndispatch = 0;
        count(&Counters::gated, s);
        s->gate_value[s->cur] = v;
        s->gate_slot = s->cur;
        if (s->committed) s->gate_hold.store(s->committed, std::memory_order_release);
    }
    id<MTL4CommandBuffer> pre = submit_waits_locked(s, v, gated);
    id<MTL4CommandBuffer> bufs[] = {pre, slot.cmd};
    MTL4CommitOptions* opts = [MTL4CommitOptions new];
    // Handlers fire once per options object; each carries its commit value.
    Stream* sp = s;
    TraceRec* tr = g_trace ? trace_new() : nullptr;
    if (tr) *tr = {0, s->sid, gated ? 2 : pre ? 1 : 0, s->ndispatch, v, now_seconds(), 0, 0, 0};
    [opts addFeedbackHandler:^(id<MTL4CommitFeedback> fb) {
        if (tr) {
            tr->g0 = fb.GPUStartTime;
            tr->g1 = fb.GPUEndTime;
        }
        on_feedback(sp, v, fb);
    }];
    if (pre) [s->queue commit:bufs count:2 options:opts];
    else [s->queue commit:bufs + 1 count:1 options:opts];
    s->committed = v;
    [s->queue signalEvent:s->event value:v];
    if (tr) tr->t1 = now_seconds();
    slot.done_value = v;
    s->cur = (s->cur + 1) % kSlots;
    s->open = false;
    count(&Counters::commits, s);
}

// Requires s->mu. True if nothing is queued on s or pending on the GPU, so
// the host may act for s right now (dependencies are then observed).
static bool idle_locked(Stream* s) {
    if (s->open || s->completed() < s->committed || !settle_waits_locked(s)) return false;
    observe(s, s->committed, "idle stream", false);
    return true;
}

// Opens a command buffer that can take `dispatches` more dispatches and
// `ring_bytes` more constant-ring bytes without rotating. Callers that stage
// data in the ring and then consume it from a dispatch must reserve both up
// front: a rotation in between would leave the payload in a slot whose reuse
// fence does not cover the consuming dispatch.
static void kprof_room_locked(Stream* s, int dispatches);

static void reserve_locked(Stream* s, int dispatches, size_t ring_bytes) {
    // The kernel profiler may also have to drain (commit + rotate) to reuse
    // heap entries; do that now, before anything is staged.
    if (g_kprof) kprof_room_locked(s, dispatches);
    begin_locked(s);
    if (s->ndispatch + dispatches > kMaxDispatchPerCommit
            || s->ring_off + ring_bytes > kRingBytes) {
        commit_locked(s);
        begin_locked(s);
    }
}

static size_t ring_round(size_t bytes) { return (bytes + 255) & ~(size_t)255; }

// Reserves `bytes` in the current slot's constant ring (opens a command
// buffer; commits and rotates when full). Returns CPU pointer + GPU address.
static void* ring_alloc_locked(Stream* s, size_t bytes, uint64_t* gpu) {
    size_t need = ring_round(bytes);
    if (need > kRingBytes) fatal("constant ring request too large (%zu bytes)", bytes);
    begin_locked(s);
    if (s->ring_off + need > kRingBytes) {
        commit_locked(s);
        begin_locked(s);
    }
    Slot& slot = s->slots[s->cur];
    char* cpu = (char*)slot.ring.contents + s->ring_off;
    *gpu = slot.ring.gpuAddress + s->ring_off;
    s->ring_off += need;
    return cpu;
}

static id<MTL4CounterHeap> new_timestamp_heap(int n) {
    MTL4CounterHeapDescriptor* d = [MTL4CounterHeapDescriptor new];
    d.type = MTL4CounterHeapTypeTimestamp;
    d.count = (NSUInteger)n;
    NSError* err = nil;
    id<MTL4CounterHeap> h = [g->device newCounterHeapWithDescriptor:d error:&err];
    if (!h) fatal("newCounterHeapWithDescriptor failed: %s",
        err ? err.localizedDescription.UTF8String : "?");
    return h;
}

// Requires s->mu. Folds completed kernel-profile pairs into the pipelines.
static void kprof_resolve_locked(Stream* s) {
    if (s->kpend.empty()) return;
    uint64_t done = s->completed();
    size_t keep = 0;
    for (size_t i = 0; i < s->kpend.size(); i++) {
        Stream::KPending& k = s->kpend[i];
        if (k.value > done) {
            s->kpend[keep++] = k;
            continue;
        }
        NSData* d = [s->kheap resolveCounterRange:NSMakeRange((NSUInteger)k.idx, 2)];
        if (d && d.length >= 2 * sizeof(MTL4TimestampHeapEntry)) {
            const MTL4TimestampHeapEntry* e = (const MTL4TimestampHeapEntry*)d.bytes;
            if (e[1].timestamp >= e[0].timestamp) {
                uint64_t ns = (uint64_t)((e[1].timestamp - e[0].timestamp) * g->ts_ns_per_tick);
                k.p->kp_ns += ns;
                k.p->kp_n++;
                if (g_kprof_shapes) {
                    char key[160];
                    snprintf(key, sizeof(key), "%s s%d g%ux%ux%u", k.p->name.c_str(),
                        s->sid, k.grid.x, k.grid.y, k.grid.z);
                    std::lock_guard<std::mutex> lk(g_kprof_mu);
                    auto& v = g_kprof_by_shape[key];
                    v.first += ns;
                    v.second++;
                }
            }
        }
    }
    s->kpend.resize(keep);
}

static void dispatch_locked(Stream* s, Pipeline* p, dim3 grid, dim3 block,
        const void* args, size_t args_size, size_t tg_mem);

// Requires s->mu. Reserves two heap entries, draining their previous user.
static int kprof_reserve_locked(Stream* s) {
    if (!s->kheap) s->kheap = new_timestamp_heap(kKprofSlots);
    int idx = s->knext;
    s->knext = (s->knext + 2) % kKprofSlots;
    for (const Stream::KPending& k : s->kpend) {
        if (k.idx == idx) {
            commit_locked(s);
            wait_value(s, s->committed, "kernel profiler heap reuse");
            kprof_resolve_locked(s);
            break;
        }
    }
    return idx;
}

// Requires s->mu. Guarantees the next `dispatches` kprof_reserve_locked calls
// will not commit, so staged ring payloads stay in the slot of their consumer.
static void kprof_room_locked(Stream* s, int dispatches) {
    for (int d = 0; d < dispatches; d++) {
        int idx = (s->knext + 2 * d) % kKprofSlots;
        for (const Stream::KPending& k : s->kpend) {
            if (k.idx == idx) {
                commit_locked(s);
                wait_value(s, s->committed, "kernel profiler heap reuse");
                kprof_resolve_locked(s);
                return;
            }
        }
    }
}

static void dispatch_locked(Stream* s, Pipeline* p, dim3 grid, dim3 block,
        const void* args, size_t args_size, size_t tg_mem) {
    if (grid.x == 0 || grid.y == 0 || grid.z == 0) return;
    if (block.x * block.y * block.z > p->pso.maxTotalThreadsPerThreadgroup) {
        fatal("kernel %s: block %ux%ux%u exceeds pipeline limit %lu", p->name.c_str(),
            block.x, block.y, block.z, (unsigned long)p->pso.maxTotalThreadsPerThreadgroup);
    }
    int kidx = g_kprof ? kprof_reserve_locked(s) : -1;
    begin_locked(s);
    if (s->ndispatch >= kMaxDispatchPerCommit) {
        commit_locked(s);
        begin_locked(s);
    }
    uint64_t arg_gpu = 0;
    void* arg_cpu = ring_alloc_locked(s, args_size ? args_size : 16, &arg_gpu);
    if (args_size) memcpy(arg_cpu, args, args_size);
    [s->table setAddress:arg_gpu atIndex:0];
    if (s->ndispatch > 0 && s->conc_n <= 0) {
        [s->enc barrierAfterEncoderStages:MTLStageDispatch
                      beforeEncoderStages:MTLStageDispatch
                        visibilityOptions:MTL4VisibilityOptionDevice];
        count(&Counters::barriers, s);
    }
    if (s->conc_n >= 0) s->conc_n++;
    [s->enc setComputePipelineState:p->pso];
    if (tg_mem) {
        [s->enc setThreadgroupMemoryLength:(tg_mem + 15) & ~(size_t)15 atIndex:0];
    }
    if (kidx >= 0) {
        [s->enc writeTimestampWithGranularity:MTL4TimestampGranularityPrecise
                                     intoHeap:s->kheap atIndex:(NSUInteger)kidx];
    }
    if (s->gate_armed) {
        // Gated: the grid comes from the slot's live words, which hold the
        // real grid only if the host released it (gate_open) before the
        // gate's queue wait ended; else zeros, an empty dispatch.
        if (s->gate_ndispatch++ > 0 || s->ndispatch > 0) {
            fatal("s%d: a gated command buffer holds exactly one dispatch", s->sid);
        }
        volatile uint32_t* w = (volatile uint32_t*)(s->wcpu + kWordGate + 4 * s->cur);
        for (int i = 0; i < 8; i++) w[i] = 0;
        s->gate_grid[s->cur][0] = grid.x;
        s->gate_grid[s->cur][1] = grid.y;
        s->gate_grid[s->cur][2] = grid.z;
        // Released grid -> the dispatch's indirect arguments, first thing
        // in the command buffer, i.e. behind the gate's queue wait: the stage
        // still holds zeros (an empty dispatch) unless gate_open wrote it.
        size_t stage = (size_t)(kWordGate + 4 * s->cur) * sizeof(uint64_t);
        [s->enc copyFromBuffer:s->words sourceOffset:stage toBuffer:s->words
             destinationOffset:stage + 2 * sizeof(uint64_t) size:2 * sizeof(uint64_t)];
        [s->enc barrierAfterEncoderStages:MTLStageBlit
                      beforeEncoderStages:MTLStageDispatch | MTLStageBlit
                        visibilityOptions:MTL4VisibilityOptionDevice];
        count(&Counters::barriers, s);
        [s->enc dispatchThreadgroupsWithIndirectBuffer:s->words.gpuAddress + stage
                + 2 * sizeof(uint64_t)
                                 threadsPerThreadgroup:MTLSizeMake(block.x, block.y, block.z)];
    } else {
        [s->enc dispatchThreadgroups:MTLSizeMake(grid.x, grid.y, grid.z)
               threadsPerThreadgroup:MTLSizeMake(block.x, block.y, block.z)];
    }
    if (kidx >= 0) {
        [s->enc writeTimestampWithGranularity:MTL4TimestampGranularityPrecise
                                     intoHeap:s->kheap atIndex:(NSUInteger)(kidx + 1)];
        s->kpend.push_back({p, kidx, s->committed + 1, grid});
    }
    s->ndispatch++;
    count(&Counters::dispatches, s);
}

// Requires s->mu. Writes every pending timestamp whose command buffer has
// completed to its destination (ns).
static void resolve_stamps_locked(Stream* s) {
    if (s->pending_stamps.empty()) return;
    uint64_t done = s->completed();
    size_t keep = 0;
    for (size_t i = 0; i < s->pending_stamps.size(); i++) {
        PendingStamp& p = s->pending_stamps[i];
        if (p.value > done) {
            s->pending_stamps[keep++] = p;
            continue;
        }
        NSData* d = [s->stamps resolveCounterRange:NSMakeRange((NSUInteger)p.index, 1)];
        uint64_t ticks = 0;
        if (d && d.length >= sizeof(MTL4TimestampHeapEntry)) {
            ticks = ((const MTL4TimestampHeapEntry*)d.bytes)->timestamp;
        }
        *p.dst = (unsigned long long)((double)ticks * g->ts_ns_per_tick);
    }
    s->pending_stamps.resize(keep);
}

// Requires s->mu.
static void stamp_locked(Stream* s, unsigned long long* dst) {
    if (!s->stamps) s->stamps = new_timestamp_heap(kStampSlots);
    int idx = s->stamp_next;
    s->stamp_next = (s->stamp_next + 1) % kStampSlots;
    // Never reuse a heap entry the GPU may still write or we have not read.
    for (const PendingStamp& p : s->pending_stamps) {
        if (p.index == idx) {
            commit_locked(s);
            wait_value(s, p.value, "timestamp heap reuse");
            resolve_stamps_locked(s);
            break;
        }
    }
    begin_locked(s);
    [s->enc writeTimestampWithGranularity:MTL4TimestampGranularityPrecise
                                 intoHeap:s->stamps
                                  atIndex:(NSUInteger)idx];
    s->pending_stamps.push_back({dst, s->committed + 1, idx});
}

Stream* resolve(cudaStream_t s) {
    init();
    return s ? (Stream*)s : g->default_stream;
}

// strict: also wait for the commit feedback of everything synchronized.
static void stream_sync(Stream* s, bool strict) {
    // Legacy default stream: synchronizing it also covers earlier work on
    // every blocking stream. (Work on a blocking stream already carries its
    // NULL-stream dependencies from enqueue time.)
    if (s->is_default) legacy_deps(s);
    uint64_t v;
    {
        std::lock_guard<std::mutex> lk(s->mu);
        commit_locked(s);
        v = s->committed;
    }
    wait_value(s, v, strict ? "host synchronization" : "cudaStreamSynchronize", strict);
    std::lock_guard<std::mutex> lk(s->mu);
    resolve_stamps_locked(s);
    kprof_resolve_locked(s);
}

void stream_sync(cudaStream_t cs) {
    stream_sync(resolve(cs), false);
}

void stream_host_wait_only(cudaStream_t cs) {
    Stream* s = resolve(cs);
    std::lock_guard<std::mutex> lk(s->mu);
    if (s->open || s->committed || !s->waits.empty()) {
        fatal("stream_host_wait_only requires an unused stream");
    }
    s->host_wait_only = true;
}

void stream_wait_spin(cudaStream_t cs, uint32_t us) {
    resolve(cs)->spin_ns.store((uint64_t)us * 1000ull, std::memory_order_relaxed);
}

bool gate_available() {
    init();
    // The kernel profiler may commit inside a dispatch, and strict feedback
    // would make the gate's opener wait for feedback only the gate releases.
    return !g_kprof && !g_strict_feedback;
}

uint64_t gate_arm(cudaStream_t cs) {
    Stream* s = resolve(cs);
    std::lock_guard<std::mutex> lk(s->mu);
    if (!gate_available()) fatal("gated commits are unavailable (kernel profiler / strict feedback)");
    if (!s->host_wait_only) fatal("s%d: gated commits need a host-wait-only stream", s->sid);
    if (s->gate_armed || s->gate_slot >= 0) fatal("s%d: gate already armed or pending", s->sid);
    if (!s->gate) {
        s->gate = [g->device newSharedEvent];
        s->gate.signaledValue = 0;
        s->gated.store(true, std::memory_order_release);
    }
    commit_locked(s);
    s->gate_armed = true;
    s->gate_ndispatch = 0;
    return s->committed;
}

uint64_t gate_commit(cudaStream_t cs) {
    Stream* s = resolve(cs);
    std::lock_guard<std::mutex> lk(s->mu);
    commit_locked(s);
    if (s->gate_armed || s->gate_slot < 0) fatal("s%d: gate_commit without a gated dispatch", s->sid);
    return s->gate_value[s->gate_slot];
}

void gate_open(cudaStream_t cs, bool run) {
    Stream* s = resolve(cs);
    std::lock_guard<std::mutex> lk(s->mu);
    int k = s->gate_slot;
    if (k < 0) fatal("s%d: gate_open without a pending gated commit", s->sid);
    if (run) {
        // The caller's writes (env outputs the dispatch reads) reach memory
        // before the grid: if Metal released the queue wait on its own while
        // we publish, the GPU may copy the stage before the signal below, and
        // a nonzero grid must then imply final inputs.
        std::atomic_thread_fence(std::memory_order_seq_cst);
        volatile uint32_t* w = (volatile uint32_t*)(s->wcpu + kWordGate + 4 * k);
        w[0] = s->gate_grid[k][0];
        w[1] = s->gate_grid[k][1];
        w[2] = s->gate_grid[k][2];
    }
    // The stage reaches memory before the signal.
    std::atomic_thread_fence(std::memory_order_seq_cst);
    s->gate.signaledValue = s->gate_target;
    s->gate_slot = -1;
    s->gate_hold.store(0, std::memory_order_release);
    if (g_trace) {
        double t = now_seconds();
        if (TraceRec* r = trace_new()) *r = {3, s->sid, run ? 1 : 0, 0, s->gate_value[k], t, t, 0, 0};
    }
}

bool gate_ran(cudaStream_t cs, uint64_t v) {
    Stream* s = resolve(cs);
    std::lock_guard<std::mutex> lk(s->mu);
    int k = -1;
    for (int i = 0; i < kSlots; i++) {
        if (s->gate_value[i] == v && v) k = i;
    }
    if (k < 0 || k == s->gate_slot) fatal("s%d: gate_ran(%llu): not an opened gated commit", s->sid,
        (unsigned long long)v);
    if (s->completed() < v) fatal("s%d: gate_ran(%llu) before completion", s->sid,
        (unsigned long long)v);
    volatile uint32_t* live = (volatile uint32_t*)(s->wcpu + kWordGate + 4 * k + 2);
    return live[0] == s->gate_grid[k][0] && live[1] == s->gate_grid[k][1]
        && live[2] == s->gate_grid[k][2];
}

void concurrent_begin(cudaStream_t cs) {
    Stream* s = resolve(cs);
    std::lock_guard<std::mutex> lk(s->mu);
    if (s->conc_n >= 0) fatal("s%d: nested concurrent_begin", s->sid);
    s->conc_n = 0;
}

void concurrent_end(cudaStream_t cs) {
    Stream* s = resolve(cs);
    std::lock_guard<std::mutex> lk(s->mu);
    if (s->conc_n < 0) fatal("s%d: concurrent_end without concurrent_begin", s->sid);
    s->conc_n = -1;
}

void stamp(cudaStream_t cs, unsigned long long* dst) {
    Stream* s = resolve(cs);
    legacy_deps(s);
    std::lock_guard<std::mutex> lk(s->mu);
    stamp_locked(s, dst);
}

void device_sync() {
    init();
    std::vector<Stream*> all;
    {
        std::lock_guard<std::mutex> lk(g->streams_mu);
        all = g->streams;
    }
    for (Stream* s : all) stream_sync(s, true);
}

// Legacy-default-stream approximation for blocking host calls (cudaMemcpy,
// cudaMemset): drain the default stream and every stream created without
// cudaStreamNonBlocking.
static void legacy_sync() {
    init();
    std::vector<Stream*> all;
    {
        std::lock_guard<std::mutex> lk(g->streams_mu);
        all = g->streams;
    }
    for (Stream* s : all) {
        if (s->is_default || s->blocking) stream_sync(s, true);
    }
}

// ---------------------------------------------------------------------------
// Init + library
// ---------------------------------------------------------------------------

// shaders/runtime.metal defines exactly one of these markers, from the
// PRECISION flag the metallib was compiled with. A float kernel bound to a
// bf16 buffer (or the reverse) would read the wrong bytes, so a library built
// for the other precision is rejected before any dispatch.
#ifdef PRECISION_FLOAT
static const char* kPrecisionMarker = "puf_precision_marker_f32";
#else
static const char* kPrecisionMarker = "puf_precision_marker_bf16";
#endif

static id<MTLLibrary> check_precision(id<MTLLibrary> lib, const char* what) {
    id<MTLFunction> fn = [lib newFunctionWithName:[NSString stringWithUTF8String:kPrecisionMarker]];
    if (!fn) fatal("%s was not compiled for this binary's precision (missing %s)", what,
        kPrecisionMarker);
    return lib;
}

static id<MTLLibrary> load_library(id<MTLDevice> dev) {
    NSError* err = nil;
    const char* path = getenv("PUF_METALLIB");
    if (path && *path) {
        NSURL* url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:path]];
        id<MTLLibrary> lib = [dev newLibraryWithURL:url error:&err];
        if (!lib) fatal("failed to load PUF_METALLIB=%s: %s", path,
            err ? err.localizedDescription.UTF8String : "?");
        return check_precision(lib, "PUF_METALLIB");
    }
    unsigned long size = 0;
    // Keep shader payload size independent of the host's __DATA/BSS layout.
    // Tests and older binaries may still embed in the legacy __DATA section.
    uint8_t* data = getsectiondata(&_mh_execute_header, "__PUF_METAL", "__metallib", &size);
    if (!data) data = getsectiondata(&_mh_execute_header, "__DATA", "__metallib", &size);
    if (!data || size == 0) {
        fatal("no embedded metallib (__PUF_METAL or legacy __DATA); build with tools/metal/build_native.sh "
            "or set PUF_METALLIB");
    }
    dispatch_data_t dd = dispatch_data_create(data, size, dispatch_get_main_queue(),
        DISPATCH_DATA_DESTRUCTOR_DEFAULT);
    id<MTLLibrary> lib = [dev newLibraryWithData:dd error:&err];
    if (!lib) fatal("failed to load embedded metallib: %s",
        err ? err.localizedDescription.UTF8String : "?");
    return check_precision(lib, "embedded metallib");
}

// Called under pipe_mu, only after a grouped function is missing from the
// primary library. Native trainers embed both libraries; standalone harnesses
// may keep the generated auxiliary file next to the executable instead.
static id<MTLLibrary> load_rollout_library() {
    NSError* err = nil;
    const char* override = getenv("PUF_METAL_ROLLOUT_LIB");
    id<MTLLibrary> lib = nil;
    if (override && *override) {
        NSURL* url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:override]];
        lib = [g->device newLibraryWithURL:url error:&err];
    } else {
        unsigned long size = 0;
        uint8_t* data = getsectiondata(&_mh_execute_header, "__PUF_METAL", "__rollgroup", &size);
        if (data && size) {
            dispatch_data_t dd = dispatch_data_create(data, size, dispatch_get_main_queue(),
                DISPATCH_DATA_DESTRUCTOR_DEFAULT);
            lib = [g->device newLibraryWithData:dd error:&err];
        } else {
            const char* primary = getenv("PUF_METALLIB");
            NSString* path;
            if (primary && *primary) {
                path = [[NSString stringWithUTF8String:primary] stringByDeletingPathExtension];
                path = [path stringByAppendingString:@".rollout.metallib"];
            } else {
                uint32_t n = 1024;
                std::vector<char> exe(n);
                if (_NSGetExecutablePath(exe.data(), &n) != 0) {
                    exe.resize(n);
                    if (_NSGetExecutablePath(exe.data(), &n) != 0) fatal("executable path unavailable");
                }
                path = [[NSString stringWithUTF8String:exe.data()] stringByDeletingLastPathComponent];
                path = [path stringByAppendingPathComponent:@"puffer.rollout.metallib"];
            }
            lib = [g->device newLibraryWithURL:[NSURL fileURLWithPath:path] error:&err];
        }
    }
    if (!lib) fatal("failed to load grouped rollout library: %s",
        err ? err.localizedDescription.UTF8String : "?");
    return check_precision(lib, "grouped rollout library");
}

static id<MTLLibrary> load_wide_library() {
    NSError* err = nil;
    const char* override = getenv("PUF_METAL_WIDE_LIB");
    id<MTLLibrary> lib = nil;
    if (override && *override) {
        NSURL* url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:override]];
        lib = [g->device newLibraryWithURL:url error:&err];
    } else {
        unsigned long size = 0;
        uint8_t* data = getsectiondata(&_mh_execute_header, "__PUF_METAL", "__wide", &size);
        if (data && size) {
            dispatch_data_t dd = dispatch_data_create(data, size, dispatch_get_main_queue(),
                DISPATCH_DATA_DESTRUCTOR_DEFAULT);
            lib = [g->device newLibraryWithData:dd error:&err];
        } else {
            const char* primary = getenv("PUF_METALLIB");
            NSString* path;
            if (primary && *primary) {
                path = [[NSString stringWithUTF8String:primary] stringByDeletingPathExtension];
                path = [path stringByAppendingString:@".wide.metallib"];
            } else {
                uint32_t n = 1024;
                std::vector<char> exe(n);
                if (_NSGetExecutablePath(exe.data(), &n) != 0) {
                    exe.resize(n);
                    if (_NSGetExecutablePath(exe.data(), &n) != 0) fatal("executable path unavailable");
                }
                path = [[NSString stringWithUTF8String:exe.data()] stringByDeletingLastPathComponent];
                path = [path stringByAppendingPathComponent:@"puffer.wide.metallib"];
            }
            lib = [g->device newLibraryWithURL:[NSURL fileURLWithPath:path] error:&err];
        }
    }
    if (!lib) fatal("failed to load wide learner library: %s",
        err ? err.localizedDescription.UTF8String : "?");
    return check_precision(lib, "wide learner library");
}

static void kprof_report() {
    if (!g) return;
    device_sync();
    std::vector<Pipeline*> ps;
    uint64_t total = 0;
    {
        std::lock_guard<std::mutex> lk(g->pipe_mu);
        for (auto& kv : g->pipes) {
            if (kv.second->kp_n) ps.push_back(kv.second);
            total += kv.second->kp_ns;
        }
    }
    std::sort(ps.begin(), ps.end(), [](Pipeline* a, Pipeline* b) { return a->kp_ns > b->kp_ns; });
    fprintf(stderr, "metal kernel profile (PUF_METAL_KPROF): total %.1f ms GPU\n", total * 1e-6);
    fprintf(stderr, "%-36s %10s %12s %10s %6s\n", "kernel", "count", "total_ms", "avg_us", "pct");
    for (Pipeline* p : ps) {
        fprintf(stderr, "%-36s %10llu %12.2f %10.2f %6.1f\n", p->name.c_str(),
            (unsigned long long)p->kp_n.load(), p->kp_ns * 1e-6,
            p->kp_ns * 1e-3 / (double)p->kp_n, total ? 100.0 * p->kp_ns / total : 0.0);
    }
    if (!g_kprof_shapes) return;
    std::vector<std::pair<std::string, std::pair<uint64_t, uint64_t>>> rows(
        g_kprof_by_shape.begin(), g_kprof_by_shape.end());
    std::sort(rows.begin(), rows.end(),
        [](const auto& a, const auto& b) { return a.second.first > b.second.first; });
    fprintf(stderr, "by kernel / stream id / grid:\n");
    for (size_t i = 0; i < rows.size() && i < 60; i++) {
        fprintf(stderr, "%-52s %8llu %10.2f %10.2f\n", rows[i].first.c_str(),
            (unsigned long long)rows[i].second.second, rows[i].second.first * 1e-6,
            rows[i].second.first * 1e-3 / (double)rows[i].second.second);
    }
}

static void init_once() {
    @autoreleasepool {
        g = new Ctx();
        g->device = MTLCreateSystemDefaultDevice();
        if (!g->device) fatal("no Metal device");
        if (![g->device supportsFamily:MTLGPUFamilyMetal4]) {
            fatal("device %s does not support the Metal 4 family", g->device.name.UTF8String);
        }
        g->library = load_library(g->device);
        const char* kp = getenv("PUF_METAL_KPROF");
        g_kprof = kp && *kp && strcmp(kp, "0") != 0;
        g_kprof_shapes = g_kprof && strcmp(kp, "2") == 0;
        const char* sf = getenv("PUF_METAL_STRICT_FEEDBACK");
        g_strict_feedback = sf && *sf && strcmp(sf, "0") != 0;
        if (g_kprof) atexit(kprof_report);
        g_trace_path = getenv("PUF_METAL_TRACE");
        if (g_trace_path && *g_trace_path) {
            g_trace = (TraceRec*)calloc(kTraceMax, sizeof(TraceRec));
            atexit(trace_dump);
        }
        // Resolved MTL4 timestamps on Apple silicon are mach_absolute_time
        // ticks (see tests/metal/test_ppo_muon.mm timing harness).
        mach_timebase_info_data_t tb;
        mach_timebase_info(&tb);
        g->ts_ns_per_tick = (double)tb.numer / tb.denom;
        MTLResidencySetDescriptor* rsd = [MTLResidencySetDescriptor new];
        rsd.initialCapacity = 64;
        NSError* err = nil;
        g->residency = [g->device newResidencySetWithDescriptor:rsd error:&err];
        if (!g->residency) fatal("newResidencySetWithDescriptor failed");
        [g->residency requestResidency];
        dispatch_queue_t q = dispatch_queue_create("puf.metal.events", DISPATCH_QUEUE_SERIAL);
        g->listener = [[MTLSharedEventListener alloc] initWithDispatchQueue:q];
        g->default_stream = stream_create(/*blocking=*/true, /*is_default=*/true);
    }
}

void init() {
    std::call_once(g_once, init_once);
}

void trace_mark(int tag, int a, int b) {
    if (!g_trace) return;
    double t = now_seconds();
    if (TraceRec* r = trace_new()) *r = {2, a, tag, b, 0, t, t, 0, 0};
}

Pipeline* pipeline(const char* name) {
    init();
    std::lock_guard<std::mutex> lk(g->pipe_mu);
    auto it = g->pipes.find(name);
    if (it != g->pipes.end()) return it->second;
    id<MTLFunction> fn = [g->library newFunctionWithName:[NSString stringWithUTF8String:name]];
    if (!fn && strncmp(name, "rollout_steps_packed_", 21) == 0) {
        if (!g->rollout_library) g->rollout_library = load_rollout_library();
        fn = [g->rollout_library newFunctionWithName:[NSString stringWithUTF8String:name]];
    }
    if (!fn && strncmp(name, "wide_", 5) == 0) {
        if (!g->wide_library) g->wide_library = load_wide_library();
        fn = [g->wide_library newFunctionWithName:[NSString stringWithUTF8String:name]];
    }
    if (!fn) fatal("shader function '%s' missing from metallib", name);
    NSError* err = nil;
    id<MTLComputePipelineState> pso =
        [g->device newComputePipelineStateWithFunction:fn error:&err];
    if (!pso) fatal("pipeline '%s' failed: %s", name,
        err ? err.localizedDescription.UTF8String : "?");
    if (getenv("PUF_METAL_PIPELINE_INFO")) {
        fprintf(stderr, "metal pipeline %s: threads_max=%lu simd=%lu scratch=%lu device_scratch_max=%lu\n",
            name, (unsigned long)pso.maxTotalThreadsPerThreadgroup,
            (unsigned long)pso.threadExecutionWidth, (unsigned long)pso.staticThreadgroupMemoryLength,
            (unsigned long)g->device.maxThreadgroupMemoryLength);
    }
    Pipeline* p = new Pipeline{pso, name};
    g->pipes[name] = p;
    return p;
}

uint32_t pipeline_max_threads(Pipeline* p) {
    return (uint32_t)p->pso.maxTotalThreadsPerThreadgroup;
}

// Requires s->mu. Queues a dependency of s's next commit on producer p
// reaching v (merged with an earlier one on p). At commit it becomes a
// GPU-side queue wait if p has not completed v by then; either way it is
// verified when the host observes the commit (observe()).
static void add_wait_locked(Stream* s, Stream* p, uint64_t v) {
    if (p->host_wait_only) {
        // No GPU progress word can witness this dependency. Finish and
        // verify the producer on the host before any consumer submission,
        // including its transitive GPU waits and lagged commit feedback.
        wait_value(p, v, "host-only stream dependency");
        return;
    }
    for (Dep& w : s->waits) {
        if (w.p == p) {
            if (w.value < v) w.value = v;
            return;
        }
    }
    s->waits.push_back({p, v});
}

// Legacy default-stream semantics (nvcc's default --default-stream legacy):
// work submitted to the NULL stream waits for all earlier work on blocking
// streams (cudaStreamCreate / flags without cudaStreamNonBlocking), and work
// on a blocking stream waits for earlier NULL-stream work. Call with no
// stream mutex held; it briefly locks the other streams to commit their
// pending work and then queues GPU-side waits on s.
static void legacy_deps(Stream* s) {
    if (!s->is_default && !s->blocking) return;
    thread_local std::vector<Dep> deps;
    deps.clear();
    // Producers that already completed still go through settle_waits_locked
    // (which verifies their history) unless nothing is left to verify.
    auto add = [&](Stream* o) {
        std::lock_guard<std::mutex> lk(o->mu);
        commit_locked(o);
        uint64_t c = o->committed;
        if (c == 0) return;
        if (o->completed() >= c && o->dep_min_at.load(std::memory_order_acquire) > c) return;
        deps.push_back({o, c});
    };
    if (s->is_default) {
        std::vector<Stream*> all;
        {
            std::lock_guard<std::mutex> lk(g->streams_mu);
            all = g->streams;
        }
        for (Stream* o : all) {
            if (o != s && o->blocking) add(o);
        }
    } else {
        add(g->default_stream);
    }
    if (deps.empty()) return;
    std::lock_guard<std::mutex> lk(s->mu);
    for (const Dep& d : deps) add_wait_locked(s, d.p, d.value);
}

void dispatch(cudaStream_t cs, Pipeline* p, dim3 grid, dim3 block,
        const void* args, size_t args_size, size_t tg_mem) {
    uint64_t t0 = now_ns();
    Stream* s = resolve(cs);
    legacy_deps(s);
    std::lock_guard<std::mutex> lk(s->mu);
    dispatch_locked(s, p, grid, block, args, args_size, tg_mem);
    count(&Counters::encode_ns, s, now_ns() - t0);
}

static Stats snapshot(const Counters& c) {
    auto ld = [](const std::atomic<uint64_t>& a) { return a.load(std::memory_order_relaxed); };
    return Stats{ld(c.dispatches), ld(c.commits), ld(c.syncs), ld(c.barriers),
        ld(c.wait_ns), ld(c.encode_ns), ld(c.gpu_ns), ld(c.gated), ld(c.spin_hit),
        ld(c.spin_miss), ld(c.strict_feedback_joins)};
}

Stats stats() {
    init();
    return snapshot(g_ctr);
}

size_t pending_deps() {
    init();
    std::vector<Stream*> all;
    {
        std::lock_guard<std::mutex> lk(g->streams_mu);
        all = g->streams;
    }
    size_t n = 0;
    for (Stream* o : all) {
        std::lock_guard<std::mutex> lk(o->diag_mu);
        n += o->deps.size();
    }
    return n;
}

void* test_hold(cudaStream_t cs) {
    Stream* s = resolve(cs);
    id<MTLSharedEvent> ev = [g->device newSharedEvent];
    ev.signaledValue = 0;
    std::lock_guard<std::mutex> lk(s->mu);
    commit_locked(s);
    [s->queue waitForEvent:ev value:1];
    return (__bridge_retained void*)ev;
}

void test_release(void* h) {
    id<MTLSharedEvent> ev = (__bridge_transfer id<MTLSharedEvent>)h;
    ev.signaledValue = 1;
}

uint64_t unconfirmed_commits() {
    init();
    std::vector<Stream*> all;
    {
        std::lock_guard<std::mutex> lk(g->streams_mu);
        all = g->streams;
    }
    uint64_t n = 0;
    for (Stream* o : all) {
        uint64_t c = o->completed(), f = o->fb_confirmed.load(std::memory_order_acquire);
        if (c > f) n += c - f;
    }
    return n;
}

Stats stream_stats(cudaStream_t cs) {
    return snapshot(resolve(cs)->ctr);
}

// ---------------------------------------------------------------------------
// Allocation registry
// ---------------------------------------------------------------------------

// Lock-free lookup: entries are fully written before num_allocs is published;
// freed entries get size 0 (and are never reused).
static Allocation* find_alloc(const void* p, bool allow_end) {
    if (!g) return nullptr;
    const char* c = (const char*)p;
    int n = g->num_allocs.load(std::memory_order_acquire);
    Allocation* end_hit = nullptr;
    for (int i = n - 1; i >= 0; i--) {
        Allocation& a = g->allocs[i];
        char* base = a.base.load(std::memory_order_acquire);
        size_t size = a.size.load(std::memory_order_acquire);
        if (!base || size == 0) continue;
        if (c >= base && c < base + size) return &a;
        if (allow_end && c == base + size) end_hit = &a;
    }
    return end_hit;
}

static std::atomic<size_t> g_live_bytes{0};
static std::atomic<int> g_live_count{0};

size_t stream_buffer_bytes() {
    return kSlots * kRingBytes
        + ((kWordsTotal * sizeof(uint64_t) + kPage - 1) / kPage) * kPage;
}

MemoryInfo memory_info() {
    init();
    std::lock_guard<std::mutex> lk(g->streams_mu);
    return {
        .recommended_max_working_set =
            (size_t)g->device.recommendedMaxWorkingSetSize,
        .current_allocated = (size_t)g->device.currentAllocatedSize,
        .process_live = g_live_bytes.load() + g->streams.size() * stream_buffer_bytes(),
        .max_buffer_length = (size_t)g->device.maxBufferLength,
    };
}

// Explains a failed cudaMalloc: Metal wires every GPU-visible allocation, and
// the kernel caps the device's total wired memory (iogpu wired limit, about
// recommendedMaxWorkingSetSize minus what other processes hold), well below
// physical RAM. PUF_METAL_ALLOC_LOG=1 also logs every allocation.
static void alloc_report(const char* what, size_t len) {
    fprintf(stderr, "metal: %s %.3f GB (live %.3f GB in %d allocations; "
        "device allocated %.3f GB, recommendedMaxWorkingSetSize %.3f GB, "
        "maxBufferLength %.3f GB)\n", what, len / 1e9, g_live_bytes.load() / 1e9,
        g_live_count.load(), (double)g->device.currentAllocatedSize / 1e9,
        (double)g->device.recommendedMaxWorkingSetSize / 1e9,
        (double)g->device.maxBufferLength / 1e9);
}

void* alloc(size_t bytes) {
    init();
    static const bool log = [] {
        const char* e = getenv("PUF_METAL_ALLOC_LOG");
        return e && *e && strcmp(e, "0") != 0;
    }();
    size_t len = ((bytes ? bytes : 1) + kPage - 1) / kPage * kPage;
    // Fail before touching the pages: zero-filling an oversized request
    // would first push up to len bytes of RAM through the pager.
    if (len > g->device.maxBufferLength) {
        alloc_report("cudaMalloc: request exceeds maxBufferLength:", len);
        return nullptr;
    }
    void* p = nullptr;
    if (posix_memalign(&p, kPage, len) != 0 || !p) {
        alloc_report("cudaMalloc: host allocation failed for", len);
        return nullptr;
    }
    memset(p, 0, len);
    id<MTLBuffer> buf = [g->device newBufferWithBytesNoCopy:p
                                                     length:len
                                                    options:MTLResourceStorageModeShared
                                                deallocator:nil];
    if (!buf) {
        alloc_report("cudaMalloc: Metal could not wire", len);
        ::free(p);
        return nullptr;
    }
    {
        std::lock_guard<std::mutex> lk(g->residency_mu);
        [g->residency addAllocation:buf];
        [g->residency commit];
    }
    std::lock_guard<std::mutex> lk(g->alloc_mu);
    int i = g->num_allocs.load(std::memory_order_relaxed);
    if (i >= kMaxAllocs) fatal("allocation registry full (%d)", kMaxAllocs);
    Allocation& a = g->allocs[i];
    a.buffer = buf;
    a.gpu = buf.gpuAddress;
    a.size.store(len, std::memory_order_relaxed);
    a.base.store((char*)p, std::memory_order_release);
    g->num_allocs.store(i + 1, std::memory_order_release);
    g_live_bytes += len;
    g_live_count++;
    if (log) alloc_report("cudaMalloc", len);
    return p;
}

void free(void* p) {
    if (!p) return;
    init();
    // Metal 4 command buffers do not retain resources: drain every queue
    // before the pages can go back to malloc (cf. valo-metal 1e7e9dd1).
    device_sync();
    Allocation* a = find_alloc(p, false);
    if (!a || a->base.load() != (char*)p) {
        fatal("cudaFree of pointer %p that is not the start of a live allocation", p);
    }
    id<MTLBuffer> buf;
    {
        std::lock_guard<std::mutex> lk(g->alloc_mu);
        buf = a->buffer;
        g_live_bytes -= a->size.load(std::memory_order_relaxed);
        g_live_count--;
        a->size.store(0, std::memory_order_release);
        a->buffer = nil;
    }
    {
        std::lock_guard<std::mutex> lk(g->residency_mu);
        [g->residency removeAllocation:buf];
        [g->residency commit];
    }
    buf = nil;
    ::free(p);
}

bool is_gpu_visible(const void* p) {
    return p && find_alloc(p, false) != nullptr;
}

uint64_t gpu_addr(const void* p, const char* what) {
    if (!p) return 0;
    Allocation* a = find_alloc(p, true);
    if (!a) {
        fatal("%s %p is not in any GPU-visible allocation (use cudaMalloc/cudaHostAlloc)",
            what, p);
    }
    return a->gpu + (uint64_t)((const char*)p - a->base.load(std::memory_order_acquire));
}

// ---------------------------------------------------------------------------
// Built-in copy / fill kernels (shaders/runtime.metal)
// ---------------------------------------------------------------------------

struct CopyArgs {
    uint64_t dst, src, n;
    uint32_t aligned, pad;
};
struct FillArgs {
    uint64_t dst, n;
    uint32_t value, aligned;
};

static dim3 bytes_grid(uint64_t n) {
    uint64_t threads = (n + 15) / 16;
    uint64_t groups = (threads + 255) / 256;
    if (groups > 0xffffffffull) fatal("copy too large (%llu bytes)", (unsigned long long)n);
    return dim3((unsigned)groups);
}

static void gpu_copy_locked(Stream* s, uint64_t dst, uint64_t src, uint64_t n) {
    static Pipeline* p = pipeline("puf_copy_bytes");
    CopyArgs a = {dst, src, n, (uint32_t)(((dst | src) & 15) == 0), 0};
    dispatch_locked(s, p, bytes_grid(n), dim3(256), &a, sizeof(a), 0);
}

static void gpu_fill_locked(Stream* s, uint64_t dst, uint8_t value, uint64_t n) {
    static Pipeline* p = pipeline("puf_fill_bytes");
    FillArgs a = {dst, n, value * 0x01010101u, (uint32_t)((dst & 15) == 0)};
    dispatch_locked(s, p, bytes_grid(n), dim3(256), &a, sizeof(a), 0);
}

}  // namespace puf_mtl

using namespace puf_mtl;

// ---------------------------------------------------------------------------
// CUDA runtime shims
// ---------------------------------------------------------------------------

struct CUevent_st {
    // (stream, value) of the last record; written and read under mu so a
    // concurrent record never exposes a new stream with a stale value.
    std::mutex mu;
    Stream* stream = nullptr;
    uint64_t value = 0;
    unsigned flags = 0;
    // Timing events only: host-observed completion time, written by a
    // shared-event listener. `gen` discards stale listeners after a
    // re-record; `pending` keeps the object alive until listeners finish.
    std::atomic<double> t_done{0.0};
    std::atomic<bool> stamped{false};
    std::atomic<uint64_t> gen{0};
    std::atomic<int> pending{0};
};

cudaError_t cudaMalloc(void** ptr, size_t size) {
    *ptr = puf_mtl::alloc(size);
    return *ptr ? cudaSuccess : cudaErrorMemoryAllocation;
}

cudaError_t cudaHostAlloc(void** ptr, size_t size, unsigned int) {
    return cudaMalloc(ptr, size);
}

cudaError_t cudaMallocHost(void** ptr, size_t size) {
    return cudaMalloc(ptr, size);
}

cudaError_t cudaFree(void* ptr) {
    puf_mtl::free(ptr);
    return cudaSuccess;
}

cudaError_t cudaFreeHost(void* ptr) {
    puf_mtl::free(ptr);
    return cudaSuccess;
}

cudaError_t cudaMemcpy(void* dst, const void* src, size_t n, cudaMemcpyKind) {
    if (n == 0) return cudaSuccess;
    legacy_sync();
    memcpy(dst, src, n);
    return cudaSuccess;
}

cudaError_t cudaMemset(void* dst, int value, size_t n) {
    if (n == 0) return cudaSuccess;
    legacy_sync();
    memset(dst, value, n);
    return cudaSuccess;
}

cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t n,
        cudaMemcpyKind, cudaStream_t cs) {
    if (n == 0) return cudaSuccess;
    Stream* s = resolve(cs);
    legacy_deps(s);
    std::unique_lock<std::mutex> lk(s->mu);
    if (idle_locked(s)) {
        // Nothing queued ahead of us: the copy can happen right now.
        memcpy(dst, src, n);
        return cudaSuccess;
    }
    bool dst_vis = is_gpu_visible(dst);
    bool src_vis = is_gpu_visible(src);
    if (dst_vis && src_vis) {
        gpu_copy_locked(s, gpu_addr(dst), gpu_addr(src), n);
        return cudaSuccess;
    }
    if (dst_vis && n <= 64 * 1024) {
        // Pageable source (possibly a stack variable): snapshot it now into
        // the constant ring, copy on the GPU in stream order.
        // Reserve payload + copy arguments together so no rotation separates
        // the staged bytes from the dispatch that reads them.
        reserve_locked(s, 1, ring_round(n) + ring_round(sizeof(CopyArgs)));
        uint64_t staged;
        void* cpu = ring_alloc_locked(s, n, &staged);
        memcpy(cpu, src, n);
        gpu_copy_locked(s, gpu_addr(dst), staged, n);
        return cudaSuccess;
    }
    // Fallback: drain the stream, then copy on the host. The stream mutex
    // stays held so no later work on this stream can start before the copy.
    commit_locked(s);
    wait_value(s, s->committed, "cudaMemcpyAsync (host fallback)");
    memcpy(dst, src, n);
    return cudaSuccess;
}

cudaError_t cudaMemsetAsync(void* dst, int value, size_t n, cudaStream_t cs) {
    if (n == 0) return cudaSuccess;
    Stream* s = resolve(cs);
    legacy_deps(s);
    std::unique_lock<std::mutex> lk(s->mu);
    if (idle_locked(s)) {
        memset(dst, value, n);
        return cudaSuccess;
    }
    if (is_gpu_visible(dst)) {
        gpu_fill_locked(s, gpu_addr(dst), (uint8_t)value, n);
        return cudaSuccess;
    }
    commit_locked(s);
    wait_value(s, s->committed, "cudaMemsetAsync (host fallback)");
    memset(dst, value, n);
    return cudaSuccess;
}

cudaError_t cudaMemGetInfo(size_t* free_bytes, size_t* total_bytes) {
    init();
    size_t total = (size_t)g->device.recommendedMaxWorkingSetSize;
    size_t used = (size_t)g->device.currentAllocatedSize;
    *total_bytes = total;
    *free_bytes = used < total ? total - used : 0;
    return cudaSuccess;
}

cudaError_t cudaStreamCreate(cudaStream_t* stream) {
    init();
    *stream = (cudaStream_t)stream_create(/*blocking=*/true, false);
    return cudaSuccess;
}

cudaError_t cudaStreamCreateWithFlags(cudaStream_t* stream, unsigned int flags) {
    init();
    *stream = (cudaStream_t)stream_create((flags & cudaStreamNonBlocking) == 0, false);
    return cudaSuccess;
}

cudaError_t cudaStreamDestroy(cudaStream_t cs) {
    if (!cs) return cudaSuccess;
    // Streams are few and long-lived; drain and keep the object (other
    // events may still reference its MTLSharedEvent).
    stream_sync(cs);
    return cudaSuccess;
}

cudaError_t cudaStreamSynchronize(cudaStream_t cs) {
    stream_sync(cs);
    return cudaSuccess;
}

cudaError_t cudaDeviceSynchronize(void) {
    device_sync();
    return cudaSuccess;
}

cudaError_t cudaEventCreate(cudaEvent_t* event) {
    return cudaEventCreateWithFlags(event, cudaEventDefault);
}

cudaError_t cudaEventCreateWithFlags(cudaEvent_t* event, unsigned int flags) {
    init();
    CUevent_st* e = new CUevent_st();
    e->flags = flags;
    *event = e;
    return cudaSuccess;
}

// Last recorded (stream, value) of e.
static Stream* event_get(cudaEvent_t e, uint64_t* v) {
    std::lock_guard<std::mutex> lk(e->mu);
    *v = e->value;
    return e->stream;
}

cudaError_t cudaEventDestroy(cudaEvent_t e) {
    if (!e) return cudaSuccess;
    // Pending completion listeners hold a raw pointer to e.
    if (e->pending.load() > 0) {
        uint64_t v;
        if (Stream* s = event_get(e, &v)) wait_value(s, v, "cudaEventDestroy");
        while (e->pending.load() > 0) usleep(10);
    }
    delete e;
    return cudaSuccess;
}

cudaError_t cudaEventRecord(cudaEvent_t e, cudaStream_t cs) {
    Stream* s = resolve(cs);
    legacy_deps(s);
    uint64_t v;
    {
        std::lock_guard<std::mutex> lk(s->mu);
        commit_locked(s);
        v = s->committed;
    }
    {
        std::lock_guard<std::mutex> lk(e->mu);
        e->stream = s;
        e->value = v;
    }
    if (e->flags & cudaEventDisableTiming) return cudaSuccess;
    // Never blocks: a listener from an earlier record of e that fires late
    // sees a newer generation and does nothing.
    uint64_t my_gen = e->gen.fetch_add(1) + 1;
    e->stamped.store(false);
    if (v == 0 || s->completed() >= v) {
        e->t_done.store(now_seconds());
        e->stamped.store(true);
    } else {
        CUevent_st* ep = e;
        ep->pending.fetch_add(1);
        [s->event notifyListener:g->listener atValue:v block:^(id<MTLSharedEvent>, uint64_t) {
            if (ep->gen.load() == my_gen) {
                ep->t_done.store(now_seconds());
                ep->stamped.store(true);
            }
            ep->pending.fetch_sub(1);
        }];
    }
    return cudaSuccess;
}

cudaError_t cudaEventSynchronize(cudaEvent_t e) {
    uint64_t v;
    if (Stream* s = event_get(e, &v)) wait_value(s, v, "cudaEventSynchronize");
    return cudaSuccess;
}

cudaError_t cudaEventElapsedTime(float* ms, cudaEvent_t a, cudaEvent_t b) {
    uint64_t va, vb;
    Stream* sa = event_get(a, &va);
    Stream* sb = event_get(b, &vb);
    if (!sa || !sb || ((a->flags | b->flags) & cudaEventDisableTiming)) {
        *ms = 0.0f;
        return cudaErrorInvalidValue;
    }
    wait_value(sa, va, "cudaEventElapsedTime");
    wait_value(sb, vb, "cudaEventElapsedTime");
    while (!a->stamped.load() || !b->stamped.load()) usleep(10);
    // Host-observed completion times (listener callbacks), not GPU timestamps.
    *ms = (float)((b->t_done.load() - a->t_done.load()) * 1e3);
    return cudaSuccess;
}

cudaError_t cudaStreamWaitEvent(cudaStream_t cs, cudaEvent_t e, unsigned int) {
    Stream* s = resolve(cs);
    // The wait is an operation enqueued on s: on the legacy default stream
    // (or a blocking stream) it is ordered after earlier blocking-stream
    // (NULL-stream) work like any other.
    legacy_deps(s);
    uint64_t v;
    Stream* p = event_get(e, &v);
    if (!p || p == s || v == 0) return cudaSuccess;
    std::lock_guard<std::mutex> lk(s->mu);
    // Applied as a GPU-side queue wait at s's next commit (unless p is done
    // by then). The producer was committed by cudaEventRecord, so this
    // cannot deadlock.
    add_wait_locked(s, p, v);
    return cudaSuccess;
}

cudaError_t cudaSetDevice(int device) {
    init();
    if (device != 0) fatal("cudaSetDevice(%d): only device 0 exists on the Metal backend", device);
    return cudaSuccess;
}

cudaError_t cudaGetDevice(int* device) {
    *device = 0;
    return cudaSuccess;
}

cudaError_t cudaGetDeviceCount(int* count) {
    init();
    *count = 1;
    return cudaSuccess;
}

cudaError_t cudaGetLastError(void) { return cudaSuccess; }

const char* cudaGetErrorString(cudaError_t err) {
    return err == cudaSuccess ? "no error" : "metal backend error";
}

[[noreturn]] static void no_graphs(const char* fn) {
    fatal("%s: CUDA graphs are not supported on the Metal backend (cudagraphs is forced off)", fn);
}
cudaError_t cudaStreamBeginCapture(cudaStream_t, cudaStreamCaptureMode) {
    no_graphs("cudaStreamBeginCapture");
}
cudaError_t cudaStreamEndCapture(cudaStream_t, cudaGraph_t*) { no_graphs("cudaStreamEndCapture"); }
cudaError_t cudaGraphInstantiate(cudaGraphExec_t*, cudaGraph_t, unsigned long long) {
    no_graphs("cudaGraphInstantiate");
}
cudaError_t cudaGraphLaunch(cudaGraphExec_t, cudaStream_t) { no_graphs("cudaGraphLaunch"); }
cudaError_t cudaGraphDestroy(cudaGraph_t) { no_graphs("cudaGraphDestroy"); }
cudaError_t cudaGraphExecDestroy(cudaGraphExec_t) { no_graphs("cudaGraphExecDestroy"); }

// ---------------------------------------------------------------------------
// cuRAND host API: CPU generation into unified memory
// ---------------------------------------------------------------------------

struct curandGenerator_st {
    uint64_t seed = 0;
    uint64_t offset = 0;
};

static inline uint64_t splitmix64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

// Uniform in (0, 1], like curandGenerateUniform.
static inline float u01(uint64_t seed, uint64_t i) {
    uint32_t bits = (uint32_t)(splitmix64(seed ^ splitmix64(i)) >> 40);  // 24 bits
    return ((float)bits + 1.0f) * (1.0f / 16777216.0f);
}

curandStatus_t curandCreateGenerator(curandGenerator_t* gen, curandRngType_t) {
    *gen = new curandGenerator_st();
    return CURAND_STATUS_SUCCESS;
}

curandStatus_t curandDestroyGenerator(curandGenerator_t gen) {
    delete gen;
    return CURAND_STATUS_SUCCESS;
}

curandStatus_t curandSetPseudoRandomGeneratorSeed(curandGenerator_t gen,
        unsigned long long seed) {
    gen->seed = seed;
    gen->offset = 0;
    return CURAND_STATUS_SUCCESS;
}

// The generator runs on the default stream: drain it so earlier GPU work on
// `out` is complete before the CPU writes.
curandStatus_t curandGenerateUniform(curandGenerator_t gen, float* out, size_t n) {
    stream_sync(resolve(0), true);
    for (size_t i = 0; i < n; i++) out[i] = u01(gen->seed, gen->offset + i);
    gen->offset += n;
    return CURAND_STATUS_SUCCESS;
}

curandStatus_t curandGenerateNormal(curandGenerator_t gen, float* out, size_t n,
        float mean, float stddev) {
    stream_sync(resolve(0), true);
    for (size_t i = 0; i + 1 < n + 1; i += 2) {
        float u1 = u01(gen->seed, gen->offset + i);
        float u2 = u01(gen->seed, gen->offset + i + 1);
        float r = sqrtf(-2.0f * logf(u1));
        float th = 6.28318530717958647692f * u2;
        out[i] = mean + stddev * r * cosf(th);
        if (i + 1 < n) out[i + 1] = mean + stddev * r * sinf(th);
    }
    gen->offset += (n + 1) & ~(size_t)1;
    return CURAND_STATUS_SUCCESS;
}
