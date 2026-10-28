> Historical development log; last recorded at commit `3ecb4081` (tree archived from `e68239d9`). Statements and gate counts describe that revision, not current support.

# Metal runtime debugging: NetHack async stall, default-geometry allocation, NetHack speed

Machine: Apple M4 (10-core GPU, 24 GB), macOS 26, Metal 4. Branch `metal-rt` (from `metal` @ 49c4b39a).
Evidence from the original report: `wt-envs/build/metal/nethack_training/` (`metal_async_wait.*`,
`metal_default_alloc.*`, `metal_serial_large.*`, `metal_run.json`, `sample*.txt`).

## 0. Confounder: the GPU was degraded system-wide during the original runs

From about 02:46 (local) the GPU delivered ~2% of its normal sustained throughput with **no
PufferLib process running**:

| probe (scratchpad `probe/`, run with nothing of ours alive) | healthy | degraded |
|---|---|---|
| sustained device copy | ~100 GB/s | 2.1-2.3 GB/s (bursts up to 45 GB/s) |
| FP32 FMA ALU | ~3.5 TFLOP/s | ~150 GFLOP/s |
| Breakout `./puffer train` | 1.8-1.98 M SPS, train ~56 ms/epoch | 50-200 K SPS, train 380-850 ms/epoch |

`ioreg` reports Device/Tiler/Renderer Utilization 100%. Per-client `AppUsage` cannot explain it:
it does not report MTL4 clients at all. `NSProcessInfo.thermalState` says nominal, low-power mode
is off and there is no memory pressure.

The kernel log does show a power cap. `ApplePPMPolicyCPMS::setDetailedThermalPowerBudget`
(clientId 2) logs the budget once a second while it is constrained:
- Before 02:52 it regularly returns to 65000 (unconstrained), with dips to 11-20 K.
- From 03:29 to 04:15 it sits at **2500-11000**, mostly at the 2500 floor.
- Throughput tracks it: Breakout 65-220 K SPS and the copy probe 2-20 GB/s in that window.

This is consistent with SoC power/thermal capping. Load from the two agents sharing the GPU
plausibly drives it, but it is not something the runtime can fix. The opt-relay baseline at ~02:46
was already 3-4x slow. All three NetHack reports (async run 02:48-02:53, serial run 02:53-02:56,
`metal_run` 02:56-02:58) were measured inside the degraded window.

## 1. "Async stall" (NetHack 128 agents, H128, mb1024, async on)

**Repro (original):** `bin_nethack train --train.total_timesteps=262144 --base.seed=73
--vec.total_agents=128 --vec.num_buffers=2 --vec.num_threads=8 --train.horizon=128
--train.minibatch_size=1024 --base.eval_episodes=0` (async default on). Epochs 1-4 took ~16 s
each, epoch 5 took 54 s, then nothing happened for ~3 min. The stack sample showed the learner in
`begin_locked` -> `waitUntilSignaledValue` and the rollout threads in `stream_sync`.

**Root cause: GPU throughput collapse, not a runtime deadlock.**
- `begin_locked` waiting in `wait_value(slot.done_value)` is ring back-pressure: the stream has
  `kSlots` command buffers in flight and waits for the oldest to finish. It always waits for
  committed work, never for a value nobody will signal.
- The **sync** run (`metal_serial_large`: `--base.async=0`, 1 buffer, same 128x128 geometry)
  also made no visible progress for 164 s before being killed. Its sample (`sample_sync.txt`)
  shows the main thread inside train (`gemm_rm`, `nh_idemb_grad`, `muon_l2_normalize`,
  `gpu_copy`), with every leaf in `begin_locked` back-pressure, so it was moving, just slowly.
  No async code is involved, so async is not the cause.
- At the measured degraded rate (`metal_run`: 64x16 geometry at 159 SPS), one 16,384-step epoch
  of the 128x128 config takes minutes. That matches "5 epochs, then nothing", given that the
  degradation was still deepening (2-4x at 02:46, ~50x by 02:55).
- Code audit of `runtime.mm` + pufferl's async protocol: every `cudaStreamWaitEvent` targets a
  value already committed by the matching `cudaEventRecord`. Rollout streams are non-blocking and
  never wait on the learner's streams. The learner only waits on its own thread-local dW stream
  (`puf_mm_tn_async_after` / `puf_dw_join`). Rollout/learner handoff uses host
  synchronisation (`cudaStreamSynchronize` + host flags). No cross-thread wait can name an
  unsignalled value.
- The old `wait_value` had a flat 300 s timeout per wait. Every individual wait completed well
  within that, so it never fired. That matches the slow-but-moving picture.

**Fixes / hardening (commits on `metal-rt`):**
- `40829e37` progress-aware waits. `wait_value` now waits in 5 s slices. After 60 s it prints one
  warning with a stream-state dump (`dump_streams`: per stream committed/completed values and
  pending cross-stream waits). It aborts only if **no stream in the process** completes anything
  for 300 s (`gpu_progress()`). A real deadlock now fails fast, with a diagnosis. A slow GPU no
  longer looks like a hang, and it no longer gets killed partway through a long wait.
- `09f0eda2` regression stress test `tests/metal/run_test_runtime_stress.sh [seconds] [threads]
  [stall_s]` (`test_runtime_stress.mm`). It mirrors pufferl's async host pattern:
  - N rollout threads on non-blocking streams: timing events, D2H to GPU-visible or pageable
    memory, `cudaStreamSynchronize`, `cudaEventElapsedTime`, pageable H2D. Workers park and are
    released between rollouts.
  - A learner: legacy `cudaMemcpy`, staged pageable copies, train-stream GEMMs with dW GEMMs on a
    side stream joined by record/wait.
  - Periodic NULL-stream / blocking-stream work and `cudaMalloc`/`cudaFree` device drains that
    run while the rollout threads are mid-step.

  All values are exact integers and are checked. A watchdog aborts when a step makes no progress
  for `stall_s`.

Latent issues seen in the audit but not fixed (pufferl does not hit them):
- `CUevent_st::{stream,value}` are not atomic, so recording an event on one thread while another
  thread waits on it is a data race.
- Timing events have a `gen`/`stamped` race that can return a stale `t_done`. This affects timing
  only.

**Re-run of the original repro on this branch (04:07-04:13, budget 2.5-11 K):** 347 s, killed by
me. It finished no epoch, so I have no stall-free completion to show.

The new diagnostics show what was happening. The 8 "host has waited 60 s" dumps report
committed/completed counters that keep advancing between dumps: s2 (train) 12/8 -> 57/53 -> 66/62,
rollout s3 286 -> 476. The only pending cross-stream waits were s1 (dW) commit 7 waiting for
s2 >= 58 and s2 commit 59 waiting for s1 >= 7. That is not a cycle: s2 commit 58 depends on
nothing.

Two stack samples show the learner in `muon_step` -> `gemm_rm` -> `begin_locked` back-pressure
and the rollout threads in `stream_sync`. That is the same signature as the original report, but
here it is demonstrably moving. Each 4-deep train command buffer (Muon Newton-Schulz on the scalar
GEMM path) took 20-60 s on the capped GPU, and the small rollout command buffers queue behind it.
There were windows of more than 10 s with no completion in the process while a separate probe
process still got 11-14 GB/s. So the capped GPU appears to run our queues one long train command
buffer at a time.

A complete long async run on an uncapped GPU was still outstanding at the end
of that investigation. The follow-up below closes it.

### 2026-10-26 follow-up: async run completes on the 16 GB M4

The rough-edges pass rebuilt NetHack FP32 on a Mac mini M4 (10-core GPU,
16 GB, macOS 27) and reran the same 128-agent, horizon-128 async geometry that
had appeared to stall, shortened from 16 to 8 epochs:

```sh
build/metal/rough_nethack/puffer train \
  --base.seed=73 --vec.total-agents=128 --vec.num-buffers=2 \
  --vec.num-threads=8 --train.horizon=128 --train.minibatch-size=1024 \
  --train.total-timesteps=131072 --base.eval-episodes=0
```

It completed all 131,072 steps and 8 epochs with exit status 0 in 2m43.8s.
Epochs remained steady at about 20.0-20.45s of learner time, including epochs
6-8 beyond the old apparent-stall point. The final summary reported about
804 SPS, 35,640 dispatches and 288 commits per measured epoch. There were no
60-second wait warnings, deadlock dumps, allocation failures or assertions.
The dashboard reported 1.8 GB live against the 12 GB displayed budget.

The existing independent async topology stress also passed for 60.1s with
three rollout threads, 446,352 rollout steps, 29,050 learner epochs and
1,841,932 dispatches; 148,783 busy-stream uploads took the intended staged
path. No NetHack or stress process remained afterward. This follow-up did not
reproduce a port defect, so it changes documentation only; the progress-aware
wait diagnostics and stress test described above remain the regression
coverage.

## 2. Default NetHack geometry fails to allocate (512 agents, 4 buffers, H512, mb4096)

`metal_default_alloc.log`: `cudaMalloc` of the activation arena fails. The kernel log at 02:47:21:
`IOGPUDevice::new_resource ... failed: try to wire down too much memory : 19160891392 > 14302248960`.

| | bytes |
|---|---|
| failing single allocation (pufferl activation arena) | **19.161 GB** |
| live allocations at that point | 0.205 GB (no leak) |
| `maxBufferLength` (per-buffer cap = the wire-down limit in the message) | 14.302 GB |
| `recommendedMaxWorkingSetSize` | 19.070 GB |

The arena is mostly FP32 observations: NetHack obs = 4782 floats per agent-step, action mask =
742 floats.

| | size |
|---|---|
| rollout obs `[2 async slots x 512, 512, 4782]` | 10.03 GB |
| train-layout obs `[512, 512, 4782]` | 5.01 GB |
| masks `[1024, 512, 742]` + `[512, 512, 742]` | 1.56 + 0.78 GB |

**Verdict: a genuine capacity limit, not a runtime bug.** The single allocation is bigger than
the per-buffer cap. Even split, the total exceeds the recommended working set. Splitting the arena
transparently is not possible either: separately allocated no-copy `MTLBuffer`s do not have
contiguous `gpuAddress` ranges, and pufferl indexes the arena as one range.

Options:
- `--base.async=0` (one rollout slot) needs ~11.6 GB and fits.
- Halve agents or horizon.
- Upstream: store obs as uint8 (4x smaller). The CUDA default build uses bf16, so it needs
  ~9.6 GB.

What changed: `1b6fb397` makes the failure self-explanatory. Before calling posix_memalign+memset,
`cudaMalloc` checks the size against `maxBufferLength` and prints the request, live bytes, device
allocated size, working set and max buffer length. `PUF_METAL_ALLOC_LOG=1` logs every allocation.

## 3. NetHack 72x slower than CUDA (159 vs 11,481 SPS)

The 159 SPS `metal_run` number (02:56-02:58) was taken on the degraded GPU (section 0), so the 72x
ratio is not a property of the port. The comparison config is also tiny: 64 agents x H16 = 1024
steps per epoch, and on CUDA train is 10 ms/epoch. The run is dominated by per-dispatch latency
and Muon's per-parameter Newton-Schulz loop.

Kernel profile (`PUF_METAL_KPROF=2`, 64 agents / H16 / mb1024 / async=0, degraded GPU, so only
the shares mean anything):

| kernel | share | notes |
|---|---|---|
| `puf_gemm_f32` (tiled scalar fallback) | 31% | see GEMM list below |
| `tensor_ops_gemm_nt` | 11% | |
| `puf_copy_bytes` | 10% | 720 one-threadgroup copies on the train stream: Muon NS `gram_buf`/`dst` copies (algo.cu, upstream structure) |
| `tensor_ops_gemm_tn` | 7% | |
| `nh_bias_relu` | 4% | |
| `nh_sample_logits` | 2.6% | |
| Muon elementwise (`weight_update`, `clip_nesterov`, `l2_normalize`, ...) | ~8% | |

GEMMs that fall off the tensor_ops path (`PUF_METAL_GEMM_LOG=1`). The fast path needs
alpha==1, beta==0, M%64==0, N%32==0 and dense layout.
- Muon Newton-Schulz: every `X X^T`, `G G` and `G X` with alpha/beta != 0/1. Examples:
  1024x1024x1024 alpha=2.93 beta=-6.89, and 1024x1698x1024 beta=4.08. These run on the scalar
  tiled kernel.
- Ragged N or M: N = 1698, 648, 592; M = 80, 208 (`ta=1`). Small encoder shapes with N = 16
  and 48.

These belong to the optimization relay. `wt-opt` is changing `gemm.mm` / `gemm.metal` / `algo.cu`
right now, so this branch does not touch them. Recommended work, in order:
1. tensor_ops epilogue with alpha/beta (C = alpha*AB + beta*C) so Muon NS runs on matmul2d.
2. Ragged edges: run tensor_ops on the 64x32-aligned interior, plus a scalar kernel on the edge
   strips (or pad K/N in the Muon buffers).
3. Fuse Muon's `gram_buf` / `dst` copies into the GEMM epilogue (beta path reading a separate C
   source), or batch the per-parameter NS loop.

**Same-time NetHack A/B, `metal_run` config** (64 agents, H16, sync, 16384 steps):

| build | SPS |
|---|---|
| `wt-envs` `bin_nethack` (d27b266d, the build that produced the 159 SPS report) | 89-142 |
| this branch, run back to back | 140 (summary) |
| this branch, a few minutes later | 243 |

CUDA does 11.1 K on this config. So the opt merges in 49c4b39a did not regress NetHack, and the
remaining gap needs a measurement on an uncapped GPU; I could not get one today.

**Bug fixed (`fceed9fa`):** NetHack runs exited 1 after training with
`missing key [?] metal/dispatches`. `log_history_bin_mean` reads every key of the last log from
every history entry. The epoch-0 log had no `metal/*` keys, and NetHack finishes episodes in epoch
0. The epoch-0 log now carries zero deltas, and the run exits 0.

## Numbers

All numbers below were measured while the GPU was still degraded (section 0).

**No SPS regression from this branch.** Interleaved A/B with
`tests/metal/bench.sh --no-build -n 3 --steps 5000000 --ab build/puffer_base build/puffer_head`.
`puffer_base` is breakout built from 49c4b39a (`runtime.mm` swapped back), `puffer_head` is this
branch:

| median of 3 | base 49c4b39a | head | delta |
|---|---|---|---|
| sps | 149.5 K | 165.4 K | +10.6% (noise; per-run spread 78-223 K) |
| train_ms | 673 | 525 | |
| dispatches / commits / cpu_waits per epoch | 868.8 / 81.27 / 44.14 | identical | 0 |

Both builds are ~13x below the healthy ~2 M SPS. The same probe that needs no PufferLib code shows
the same drop, so the slowdown is external.

**Gates on `metal-rt` HEAD:**
- `run_train_replay` PASS
- `run_test_runtime` ALL PASS
- `run_test_rollout` ALL PASS
- `run_test_env_golden` PASS (0 failing tensors)
- `run_test_envs` PASS
- `run_test_gemm`, `kernels`, `mingru`, `ppo_muon` (`../golden/run_a`) PASS
- Full Breakout `bench.sh --full -n 1`: eval score **784.0** (>= 700). It ran at 65 K SPS on the
  degraded GPU.
- After `367ad813` / `8fbe6f18` / `fceed9fa`, re-run: `run_test_runtime` ALL PASS, stress 60 s ALL
  PASS, `run_train_replay` PASS, `run_test_rollout` ALL PASS. A 3M-step Breakout run completes
  (95 K SPS, capped GPU). The NetHack small config exits 0.

**Stress test:**
- `run_test_runtime_stress.sh 60 3 120`: 60 s, 3 rollout threads, 31,648 rollout steps, 2,096
  learner epochs, 131 K dispatches, 10,548 ring-staged busy-stream uploads, ALL PASS.
- Self-check `STRESS_DROP_WAIT=1` (dW join removed): FAILED in 931 of 970 epochs, so the checks do
  detect a lost cross-stream dependency.

## Review (codex, `docs/review-rt.md`)

| finding | resolution |
|---|---|
| P2: busy-upload variant never staged anything, because the memset took the idle fast path | Queue a GEMM first, count staged uploads, and fail if there are none (`8fbe6f18`) |
| P2: alloc churn exhausts the 16 K allocation registry on long soaks (slots are never reused) | Cap the churn at 8 K cycles (`8fbe6f18`) |
| P2: identical minibatch inputs hide stale cross-stream reads | Rotating per-minibatch scales, accumulated result (`8fbe6f18`) |
| P3: "live allocations" printed registry usage | Track a live count (`367ad813`) |
