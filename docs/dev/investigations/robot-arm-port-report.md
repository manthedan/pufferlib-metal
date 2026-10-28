> Historical development log; last recorded at commit `e606becf` (tree archived from `e68239d9`). Statements and gate counts describe that revision, not current support.

# Robot-arm CPU physics with the Metal learner

Measured 2026-10-26 in `wt-robotarm`, branch `metal-robotarm`, based on
`b0f7bb84`. Mac mini M4, 10 CPU cores, 10 GPU cores, 16 GB, macOS 27.0.1
(26A434), Apple clang 21.0.0, Xcode selected through
`DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer`, bash 3.2.57.
Python tools used `<repo>/.venv`.

**Verdict:** CPU physics is fast enough to train: approximately **155–158k
environment steps/s** on ten threads and **110–116k training SPS with the
default four threads**, or **142–153k with ten threads** in the bounded
throughput runs. Early CUDA trajectories and reset behavior pass tight gates.
Long trajectories amplify rounding differences. The matched training runs
complete, but **learning-quality parity is not established**: neither solves
the task in 33.6M steps, and CUDA shows more grasp/lift progress.

## Implementation and upstream changes

`./build.sh robot_arm --metal` builds BF16; adding `--float` builds FP32.
`PUF_ENV_THREADS=10 ./puffer train ...` uses all ten CPU threads. The config
itself remains unchanged: 4096 agents, one buffer, four threads, H256/L2,
horizon 128, minibatch 32768, eight continuous action heads, seed 42.

The new `ocean/robot_arm/robot_arm_cpu.h` implements the ordinary CPU `puf_*`
API. The existing native vector worker supplies per-agent observation,
action, reward and terminal pointers and parallelizes independent environments.
The learner, vector worker implementation and Metal shaders are unchanged.
Environment seeds match CUDA's `i + 1` initialization; upstream environment
seeds are independent of the policy's `base.seed`.

Only two upstream robot-arm files change:

- `robot_arm.h`: `RA_CPU` guards erase device annotations and provide clang
  inlining. A guarded reset parameter makes basketball RNG order explicit.
- `robot_arm_cuda.cuh`: guard the CUDA header, abstract kernel annotation and
  index, and choose observation conversion. CPU calls each existing kernel
  body with one environment and index zero. CUDA expansions retain the
  original behavior.

`robot_arm.cu` is untouched. Build selection changes are limited to robot_arm
in `build.sh` and `tools/metal/build_native.sh`. CUDA remains the default
non-Metal backend; standalone `--cpu`/web support was not added.

The sequence is unchanged: initialization via `ra_kinit`, then `ra_kbegin`,
eight substeps of **prep → buildc → react → solve**, then `ra_kfin`.
No physics equations, collision limits, integration step, rewards, or action
scales were simplified. Each Env occupies 102,944 bytes; 4096 environments
use about 402 MiB before learner allocations.

Two portability details required explicit validation:

1. Upstream `ra_resetb` passes two RNG calls as arguments to `ra_v3`.
   Its GCC host evaluates z then x; its CUDA device evaluates x then z.
   Clang initially produced different ball positions at frame zero. The CPU
   adapter now requests host order during creation/render reset and device
   order during episode/ball resets. Both paths have CUDA fixtures. This is
   a real logic discrepancy that was fixed, separate from numerical drift.
2. Upstream calls `__float2bfloat16` even when `obs_t` is FP32. The CPU
   adapter uses `from_float`: BF16 round-to-nearest-even or full FP32.
   FP32 CUDA/CPU stored observations therefore differ by quantization even
   at initialization (maximum approximately 0.00191). The harness captures
   unquantized GPU observations as well, and verifies both physics and the
   appropriate stored-observation conversion.

## Throughput and time split

Environment-only benchmark: 4096 environments, eight warmup ticks, 256
measured ticks, deterministic changing actions, ten OpenMP threads, `-O2`
as in the trainer. These repetitions ran after our training measurements,
with no robot-arm learner running. The Mac was not exclusively reserved;
other sessions can contribute system variability. Three repetitions:

| Precision | Environment SPS, repetitions | Median |
|---|---|---:|
| BF16 observations | 153,417 / 157,806 / 167,209 | **157,806** |
| FP32 observations | 156,305 / 154,994 / 153,763 | **154,994** |

Physics is FP32 in both builds. Variation is run-to-run system variability,
not evidence that BF16 speeds up the physics. Earlier 128-tick spike runs
measured 175,100 / 177,052 SPS; their shorter action history encounters a
different workload, so the longer repetitions above are the reported result.

Training measurements use the default config with only total steps reduced
to 4,194,304 (eight epochs), profile enabled, evaluation disabled and output
paths relocated. `PUF_METAL_KPROF=1` records GPU timings. Seven measured
epochs exclude epoch-zero initialization; the final epoch drains the async
pipeline without launching another rollout.

| Precision / CPU threads | Summary SPS | CPU env ms | Rollout model ms | Learner ms | Learner GPU ms | Wall ms |
|---|---:|---:|---:|---:|---:|---:|
| BF16 / default 4 | 115,660 | 4,025 | 419.1 | 640.7 | 631.9 | 4,533 |
| FP32 / default 4 | 110,299 | 4,159 | 472.8 | 944.6 | 934.6 | 4,753 |
| BF16 / 10 | 152,775 | 2,929 | 412.8 | 669.9 | 659.7 | 3,432 |
| FP32 / 10 | 141,948 | 3,041 | 473.8 | 1,005 | 991.3 | 3,694 |

Times are trainer summary averages per measured epoch. CPU environment and
learner work overlap, so these columns must not be added to infer wall time.
CPU physics accounts for roughly 67–73% of the sum of environment, rollout
model and learner counters with ten threads. The throughput is well above
the requested 20k-SPS stopping threshold. The full 1B-step default would take
roughly 2.2 hours at the observed longer-run Metal rate.

Reproduce locally (substitute `f32`/`-DPRECISION_FLOAT`/`--float` as needed):

```sh
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
export PATH="$HOME/metal_puffer/.venv/bin:$PATH"
mkdir -p build/robot_arm
bash tests/metal/build_robot_arm_cpu.sh tests/metal/bench_robot_arm.cpp build/robot_arm/bench
build/robot_arm/bench 4096 256 10
./build.sh robot_arm build/robot_arm/train_bf16 --metal
PUF_ENV_THREADS=10 PUF_METAL_KPROF=1 build/robot_arm/train_bf16 train \
  --train.total_timesteps=4194304 --base.profile=1 --base.eval_episodes=0 \
  --base.log_dir=build/robot_arm/bench_logs --base.checkpoint_dir=build/robot_arm/bench_checkpoints
```

## CUDA numerical comparison

Reference: CUDA-reference RTX 3090, CUDA 12.8.93, unmodified environment source from
`920a88e92f333b00e6b201659129c041aaa6c10f`. All three robot-arm files match
the local base byte-for-byte. An isolated archive lives at
`<cuda-reference>/robot-arm-oojWWcdf`; no other worktree was changed.
Builds and runs used `nice -n 19`. GPU occupancy was checked before runs,
and work waited while another user's process occupied the GPU.

The capture harness calls the actual four CUDA kernels and the actual CPU
adapter. Four fixed seeds and action streams run for 640 steps in each of
pick, stack and basketball, in both precisions. It also captures a forced
timeout at step 2, including automatic reset observations. Fixture layout,
source hashes, flags and regeneration commands are in
`tests/metal/fixtures/robot_arm/README.md`.

Maximum/mean absolute **unquantized observation** error over all twelve
mode/seed combinations at each indicated step (same physics in both builds):

| Step | Max absolute error | Mean absolute error |
|---:|---:|---:|
| 0 | 1.79e-7 | 8.07e-9 |
| 1 | 6.43e-7 | 1.76e-8 |
| 2 | 1.72e-6 | 2.26e-8 |
| 3 | 8.41e-6 | 6.71e-8 |
| 4 | 2.88e-5 | 2.27e-7 |
| 6 | 0.00345 | 3.05e-5 |
| 8 | 0.0249 | 0.000242 |
| 16 | 0.224 | 0.00255 |
| 64 | 0.0613 | 0.00124 |
| 256 | 0.265 | 0.00273 |
| 640 | 0.262 | 0.00316 |

Define divergence as any raw observation differing by more than 1e-3: all
three modes first cross that threshold at **step 6**. They cross 1e-2 at
step 7 and 0.1 at step 15. The early largest differences occur in joint
velocity observations. Across the whole capture, max/mean errors are
0.378/0.00164 (pick), 0.378/0.00161 (stack), and 0.404/0.00176 (basketball).
All terminals match through step 640. Reward and stored-observation max/mean
errors at sampled steps, separately for every mode and precision, are
committed in `docs/dev/investigations/robot-arm-results.json`.

Reward error over the same twelve mode/seed combinations:

| Step | Max absolute error | Mean absolute error |
|---:|---:|---:|
| 1 | 3.21e-8 | 1.43e-8 |
| 4 | 3.21e-8 | 1.19e-8 |
| 16 | 3.04e-5 | 7.83e-6 |
| 64 | 0.000126 | 2.14e-5 |
| 256 | 0.000190 | 3.20e-5 |
| 640 | 3.48e-5 | 1.02e-5 |

As a sensitivity control, the same CPU executable was built with a diagnostic
that changes only initial q[0] by one representable float toward +infinity.
Initial observation error is at most 9.31e-10; the otherwise identical CPU
trajectories exceed 1e-3 at step 8, 1e-2 at step 11, and 0.1 at step 15.
Together with the tightly matching first steps and unchanged physics bodies,
this supports chaotic amplification of floating-point differences after
the separately corrected RNG-order bug. It is not a proof that every future
trajectory is equivalent. No float64 rewrite or relaxed full-trajectory
acceptance test was used.

## Matched training comparison

Both runs use BF16, seed 42, identical `default.ini` and `robot_arm.ini`,
33,554,432 steps (64 epochs), and a separate 1024-episode evaluation with
1024 evaluation agents. Architecture, optimizer, horizon, minibatch, replay
and annealing settings are unchanged. Metal uses ten CPU threads; CUDA uses
its GPU environment and retains CUDA graphs. Metal disables CUDA graphs as
usual. Total-step reduction also shortens the default annealing schedules.

The Metal command was:

```sh
PUF_ENV_THREADS=10 build/robot_arm/train_bf16 train \
  --train.total_timesteps=33554432 --base.eval_episodes=1024 --base.eval_agents=1024 \
  --base.profile=1 --base.seed=42 --base.run_id=robot_arm_compare \
  --base.log_dir=build/robot_arm/learning/logs \
  --base.checkpoint_dir=build/robot_arm/learning/checkpoints
```

CUDA used the same arguments with `nice -n 19` and its reference binary in
the remote scratch tree. `PUF_ENV_THREADS` applies only to the Metal CPU path.

| Backend | Training seconds | Mean SPS | Eval success score | Eval return | Eval grasp rate | Eval lift rate |
|---|---:|---:|---:|---:|---:|---:|
| Metal + CPU physics | 269.91 | 124,318 | 0 | -4.36512 | 0 | 0 |
| CUDA + GPU physics | 67.23 | 499,093 | 0 | -4.34715 | 0.00879 | 0.00293 |

The longer Metal run's warmup-excluded summary is 126,180 SPS, with 3,722 ms
CPU environment, 420 ms rollout model, and 681.8 ms learner per epoch. Rates
decline relative to the short spike as the action/contact workload evolves.

Training-curve bin means (before final evaluation):

| Backend | Mean steps, millions | Episode return | Grasp rate |
|---|---|---|---|
| Metal | 5.90 / 14.51 / 23.07 / 31.46 | -4.298 / -4.345 / -4.363 / -4.366 | 0 / 0 / 0 / 0 |
| CUDA | 5.90 / 14.94 / 21.61 / 29.83 | -4.300 / -4.143 / -4.383 / -4.291 | 0.00012 / 0.00067 / 0.00117 / 0.00247 |

These are the trainer's recorded history bins at their actual step positions,
not artificially aligned curves. Both success curves remain zero. CUDA
learns occasional grasps/lifts while Metal does not in this single short run;
the return difference is small, but that intermediate-skill gap is real.
The experiment is only 3.36% of the default 1B-step budget and has one seed.
It establishes execution and bounded numerical agreement, **not equal sample
efficiency, convergence, or task mastery**. A multi-seed, longer comparison
is needed before claiming learning parity.

The reference trainer was compiled directly with the normal native nvcc/link
flags: its existing `build.sh` literal-typedef check rejects `robot_arm.cu`
because `obs_t` is declared in its included header. No reference physics or
trainer math was patched to bypass that build-script check.

## Gates and limits

`tests/metal/run_test_robot_arm.sh` is registered in `tests/metal/run_all.sh`.
It uses fresh artifact directories and checks:

- CPU vs committed CUDA fixtures for initialization and the first four
  physics steps, in both precisions, three modes and four seeds;
- a separate BF16 timeout/reset fixture, including basketball's device RNG
  order, reward and terminal output;
- BF16 and FP32 builds plus 262,144 training steps with 256 agents, unchanged
  H256/L2/horizon/minibatch, finite checkpoints and persisted episode metrics.

The three fixtures total 100,896 bytes. Raw-observation absolute limits at
steps 0–4 are `[5e-7, 1e-6, 3e-6, 1.2e-5, 4e-5]`; stored observations
allow 4e-5, rewards 1e-7, and terminals must match exactly. The short window
is intentional; the later chaotic trajectories are measured, not hidden
behind broad tolerances.
The gate separately requires exact CPU storage semantics: FP32 observations
equal the unquantized observation vector, and BF16 observations equal its
round-to-nearest-even conversion. Thus CUDA's FP32 quantization quirk cannot
hide accidental BF16 quantization in the CPU float build.

`python tests/metal/mutate_robot_arm.py` removes the joint gravity term in a
temporary source copy and invokes the same trajectory gate. It fails at the
observation comparison as required. Production files are untouched, and the
clean BF16/FP32 trajectory and training smoke gates subsequently pass.
The full robot-arm gate also invokes this mutation check automatically.

Full regression command:

```sh
PUFFER_GOLDEN_DIR=<repo>/golden PATH=<repo>/.venv/bin:$PATH bash tests/metal/run_all.sh
```

**Full runner: 40/40 gates PASS, process exit status 0.** This includes all
39 existing gates and the new robot-arm gate with its physics mutation.
Pong, Flappy, Cartpole and Breakout retained their exact reference scores
and checkpoint hashes across the exercised precisions and overlap settings.
Evidence is retained in
`build/robot_arm/gates_summary.log`, `build/<gate>.log`, and
`build/robot_arm/`; quantitative summaries and fixture hashes are committed
in `docs/dev/investigations/robot-arm-results.json`.

Rendering assets/UI were not exercised. Stack and basketball have numerical
and reset coverage but no training-quality experiment. The CPU implementation
preserves this CUDA/compiler reference's reset order; an upstream change to
RNG sequencing should update both adapter and fixtures. A shader physics port
could remove the dominant CPU cost, but the approximately 100 KiB per-env
working state and contact solver make register spilling and memory layout
important design questions. The measured CPU route is viable without that
additional implementation.
