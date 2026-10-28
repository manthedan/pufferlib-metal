// Metal backend runtime: plain C++ interface (no Objective-C types) shared by
// the CUDA shims, the launch layer, the GEMM path and tests.
//
// Model: every allocation is page-aligned shared memory wrapped NoCopy in an
// MTLBuffer and registered in a global table, so a host pointer can be
// translated to a GPU virtual address (CPU VA != GPU VA on Apple silicon).
// Each cudaStream_t owns one MTL4CommandQueue; work is encoded into rotating
// command buffers with a dispatch barrier before every kernel, which gives
// CUDA in-order stream semantics. Cross-stream ordering (cudaStreamWaitEvent)
// is a GPU-side queue wait on the producer's MTLSharedEvent.
//
// Metal 4 command buffers do NOT retain the resources they reference: memory
// may only be released after every queue that could touch it has drained.
// puf_mtl::free / cudaFree therefore synchronize the whole device first.
#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "compat/cuda_runtime.h"

namespace puf_mtl {

void init();

// Allocation registry ------------------------------------------------------
void* alloc(size_t bytes);           // zeroed, 16 KiB aligned, GPU-visible
void free(void* p);                  // drains all streams first
bool is_gpu_visible(const void* p);  // inside a registered allocation
struct MemoryInfo {
    size_t recommended_max_working_set;
    size_t current_allocated;
    size_t process_live;
    size_t max_buffer_length;
};
// Snapshot used by the trainer's configuration preflight. `process_live`
// counts this runtime's page-rounded live allocations, including stream
// buffers; `current_allocated` is Metal's currentAllocatedSize.
MemoryInfo memory_info();
// Page-rounded shared buffers created by each new stream (rings + words).
size_t stream_buffer_bytes();
// GPU address of p. NULL maps to 0. One-past-the-end pointers are accepted.
// Aborts with `what` in the message if p is not GPU-visible.
uint64_t gpu_addr(const void* p, const char* what = "pointer");

// Streams ------------------------------------------------------------------
struct Stream;
Stream* resolve(cudaStream_t s);     // NULL -> process-wide default stream
void stream_sync(cudaStream_t s);
void device_sync();
// Before the first enqueue, opt a stream into host-only completion waits.
// Omits its GPU progress-word stores. A later cross-stream dependency on it
// falls back to a verified host wait before the consumer can be submitted.
// Call only while the stream is unused (no concurrent enqueues).
void stream_host_wait_only(cudaStream_t s);
// Host waits on s (event/stream syncs) first poll its completion for up to
// `us` microseconds before blocking (0 = block at once, the default). For
// short GPU spans a blocking wake costs ~40-60 us; polling costs a CPU core
// for at most `us` per wait. Counts polls that ended in a block (spin_miss).
void stream_wait_spin(cudaStream_t s, uint32_t us);

// Host-gated commits ----------------------------------------------------------
// Lets a host thread commit work that must not start before the host has
// finished something (e.g. the rollout step after the CPU env step), so that
// Metal's submission latency (~100 us commit -> GPU start) overlaps the host
// work: once the host opens the gate the work starts ~30 us later.
//   gate_arm(s): commits what is open on s; s's next command buffer, which
//     must hold exactly one dispatch, is gated. Returns s's last commit.
//   gate_commit(s): commits it (behind a queue wait on s's gate event) and
//     returns its value. At most one gated commit waits per stream.
//   gate_open(s, run): releases it. run=false releases it as an empty
//     dispatch (cancel). Make the dispatch's inputs final before calling.
//   gate_ran(s, v): after the host observed gated commit v complete: whether
//     its dispatch ran with its real grid. Metal releases a queue wait ~4 s
//     after submission whether or not the host signaled; the dispatch reads
//     its grid from words gate_open fills, copied behind the wait, so a wait
//     released before gate_open runs an empty dispatch and returns false
//     here. The caller must then encode the work again.
// Requires a host-wait-only stream (nothing on the GPU may depend on a
// gated commit) and gate_available(). Commits before a pending gated commit
// get their feedback late (MTL4 holds a commit's feedback back while the
// next commit on its queue is blocked); strict joins skip that commit and the
// stream settles lagged feedback one observation later.
bool gate_available();
uint64_t gate_arm(cudaStream_t s);
uint64_t gate_commit(cudaStream_t s);
void gate_open(cudaStream_t s, bool run = true);
bool gate_ran(cudaStream_t s, uint64_t v);

// Kernels -----------------------------------------------------------------
struct Pipeline;
// Looks up (and caches) a compute pipeline from the embedded metallib.
// Aborts if the function does not exist.
Pipeline* pipeline(const char* name);
// Grouped rollout functions load lazily from __PUF_METAL,__rollgroup (or
// puffer.rollout.metallib beside a standalone harness). Kernel tests may
// override just that library with PUF_METAL_ROLLOUT_LIB.
uint32_t pipeline_max_threads(Pipeline* p);

// Encodes one dispatch on `s`: grid/block follow CUDA's <<<grid, block>>>
// (threadgroup counts x threads per threadgroup). `args` is copied into the
// stream's constant ring at encode time and bound at [[buffer(0)]].
// `tg_mem` bytes of threadgroup memory are bound at [[threadgroup(0)]].
void dispatch(cudaStream_t s, Pipeline* p, dim3 grid, dim3 block,
    const void* args, size_t args_size, size_t tg_mem = 0);

template <typename T>
inline void dispatch(cudaStream_t s, Pipeline* p, dim3 grid, dim3 block,
        const T& args, size_t tg_mem = 0) {
    // dispatch(s, p, g, b, &args, sizeof(args)) would bind here with T = a
    // pointer and sizeof(args) as tg_mem; pass the struct itself instead.
    static_assert(!std::is_pointer<T>::value, "pass the argument struct by value");
    dispatch(s, p, grid, block, &args, sizeof(T), tg_mem);
}

// Concurrent window: dispatches encoded on `s` between concurrent_begin and
// concurrent_end may execute concurrently with each other (no inter-dispatch
// barrier among them). The window's first dispatch still waits for every
// earlier dispatch, and the first dispatch after concurrent_end waits for
// every dispatch in the window. The caller guarantees the window's
// dispatches neither read what another writes nor write the same memory.
// A command-buffer boundary inside a window only adds ordering.
void concurrent_begin(cudaStream_t s);
void concurrent_end(cudaStream_t s);

[[noreturn]] void fatal(const char* fmt, ...);
// Tests only: fatal() passes its message to fn before aborting; fn may throw
// to recover (only meaningful for errors raised on the calling thread).
void set_fatal_handler(void (*fn)(const char* msg));

// GPU timestamp (nanoseconds, arbitrary epoch) written to *dst once the
// work encoded on `s` so far has executed. Backed by an MTL4CounterHeap; the
// value lands in *dst when the host next synchronizes with `s`
// (cudaStreamSynchronize / cudaDeviceSynchronize / blocking cudaMemcpy).
void stamp(cudaStream_t s, unsigned long long* dst);

// PUF_METAL_TRACE=path only (profiling): records a host timestamp with a
// caller-defined tag and two integers in the CSV timeline. No-op otherwise.
void trace_mark(int tag, int a = 0, int b = 0);

// Stats (tests, profiling). Monotonic counters since init.
struct Stats {
    uint64_t dispatches;
    uint64_t commits;     // command-buffer commits (+ wait/signal-only submits)
    uint64_t host_syncs;  // CPU waits that actually blocked on the GPU
    uint64_t barriers;    // encoder + queue barriers encoded
    uint64_t wait_ns;     // CPU time blocked in those waits
    uint64_t encode_ns;   // CPU time inside puf_mtl::dispatch (incl. locking)
    uint64_t gpu_ns;      // sum of command-buffer GPU spans (commit feedback)
    uint64_t gated;       // gated commits (gate_commit)
    uint64_t spin_hit;    // stream_wait_spin polls that saw completion
    uint64_t spin_miss;   // ... that ran out and blocked
    // join_all_feedback calls caused by PUF_METAL_STRICT_FEEDBACK. Ordinary
    // strict API operations are excluded, so this remains zero by default.
    uint64_t strict_feedback_joins;
};
Stats stats();
// Same counters restricted to one stream (NULL = default stream).
Stats stream_stats(cudaStream_t s);
// Tests only: cross-stream GPU waits recorded and not yet verified, summed
// over all streams.
size_t pending_deps();
// Tests only: completed commits whose feedback has not been confirmed, summed
// over all streams.
uint64_t unconfirmed_commits();
// Tests only: work committed to s after test_hold(s) waits on the GPU until
// test_release(h) (or until Metal releases the queue wait, ~4 s).
void* test_hold(cudaStream_t s);
void test_release(void* h);

}  // namespace puf_mtl
