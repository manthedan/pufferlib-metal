# Robot arm on Metal

Metal uses the original robot-arm physics on the CPU and the BF16 or FP32
Metal learner. Early CUDA trajectories and resets pass tight comparisons;
long trajectories amplify rounding differences. At the full default 1B-step
budget, all three Metal and three CUDA runs completed, but neither backend
solved pick. No learning-quality gap was statistically detected with three
seeds per backend; equivalence is not established.

## Port and CPU physics

`./build.sh robot_arm --metal` selects BF16; add `--float` for FP32.
`PUF_ENV_THREADS=10` selects ten CPU threads instead of the configured four.
Default geometry is unchanged: 4096 agents, one buffer, H256/L2, horizon 128,
minibatch 32768, eight continuous action heads, policy seed 42. Environment
initialization uses agent index+1, independently of policy seed.

`ocean/robot_arm/robot_arm_cpu.h` implements the CPU puf_* interface using
the existing vector worker and per-agent pointers. RA_CPU guards in the
shared headers remove device annotations and adapt indexing/observation
conversion; robot_arm.cu remains unchanged. CUDA is still the default
non-Metal backend; standalone CPU/web support was not added.

Initialization calls ra_kinit, then each tick uses ra_kbegin, eight substeps
of prep→buildc→react→solve, and ra_kfin. Physics equations, collision limits,
integration, rewards and action scales are unchanged. State is 102,944 bytes
per environment, about 402 MiB for 4096 environments before learner buffers.
OpenMP workers step independent states with per-environment RNG. Terminal
handling logs the completed episode, resets, then returns reset observations
with that step's reward/terminal; there is no extra vector-worker reset.

Two portability differences were isolated:

- Basketball reset originally passed two RNG calls as function arguments.
  GCC host evaluates z then x; CUDA device evaluates x then z. The CPU adapter
  explicitly uses host order for creation/render reset and device order for
  episode/ball resets. Both have CUDA fixtures.
- CUDA converts observations through BF16 even in the FP32 build. CPU
  `from_float` uses RNE BF16 or full FP32. Initialization stored-observation
  differences can therefore be approximately 0.00191 in FP32. Tests compare
  unquantized physics and each backend's intended storage separately.

## CPU and training throughput

Measured 2026-10-26 on M4, 10 CPU/10 GPU cores, 16 GB, macOS 27.0.1,
Clang 21, with other system activity possible. Environment benchmark: 4096
environments, eight warmup ticks, 256 measured ticks, changing deterministic
actions, ten threads and -O2. Raw repetition rates from the results JSON:

| Observation precision | Environment SPS repetitions |
|---|---|
| bf16 | 153416.9 / 157806.0 / 167209.2 |
| f32 | 156305.0 / 154994.3 / 153763.2 |

Physics is FP32 in both builds; the variation does not show a BF16 physics
speedup. Earlier 128-tick spike rates 175,100/177,052 SPS use a different
contact history and are not the longer-run result.

Training uses default geometry, 4,194,304 steps (eight epochs), profiling,
evaluation off, and relocated outputs. Seven measured epochs exclude warmup;
the final epoch drains without a fresh rollout. Times are per-epoch averages:

| Precision / CPU threads | Summary SPS | CPU env ms | Rollout model ms | Learner ms | Learner GPU ms | Wall ms |
|---|---:|---:|---:|---:|---:|---:|
| BF16 / default 4 | 115,660 | 4,025 | 419.1 | 640.7 | 631.9 | 4,533 |
| FP32 / default 4 | 110,299 | 4,159 | 472.8 | 944.6 | 934.6 | 4,753 |
| BF16 / 10 | 152,775 | 2,929 | 412.8 | 669.9 | 659.7 | 3,432 |
| FP32 / 10 | 141,948 | 3,041 | 473.8 | 1,005 | 991.3 | 3,694 |

CPU environment, rollout and learner work overlap; do not sum these columns
into wall time. CPU physics is the dominant cost. Full-budget measurements
below supersede extrapolation from these short runs.

```sh
bash tests/metal/build_robot_arm_cpu.sh tests/metal/bench_robot_arm.cpp build/robot_arm/bench
build/robot_arm/bench 4096 256 10
./build.sh robot_arm build/robot_arm/train_bf16 --metal
PUF_ENV_THREADS=10 PUF_METAL_KPROF=1 build/robot_arm/train_bf16 train \
  --train.total_timesteps=4194304 --base.profile=1 --base.eval_episodes=0 \
  --base.log_dir=build/robot_arm/bench_logs --base.checkpoint_dir=build/robot_arm/bench_checkpoints
```

## CUDA trajectory validation

The reference is RTX 3090 / CUDA 12.8.93 and unmodified upstream environment
source. The harness calls actual CUDA kernels and the CPU adapter for four
fixed seeds/action streams, 640 steps, pick/stack/basketball, both precisions,
plus forced timeout/reset at step 2. The
[fixture README](../../tests/metal/fixtures/robot_arm/README.md) retains format,
source hashes and regeneration commands. The three tracked files are each
33,632 bytes, total 100,896 bytes.

Maximum/mean unquantized observation error across twelve mode/seed combinations
(same physics in both builds), at the original report's displayed precision:

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

All modes first exceed raw-observation errors 1e-5/1e-4/1e-3/1e-2/0.1 at
steps 4/5/6/7/15. All terminals match through step640. Whole-trajectory values
from the JSON, without additional rounding:

| Mode | Max raw error | Mean raw error |
|---|---|---|
| pick | 0.3775320053100586 | 0.0016413857008807636 |
| stack | 0.3775320053100586 | 0.001610084061898508 |
| basketball | 0.4043847918510437 | 0.0017591354040385442 |


A CPU sensitivity control changes only initial q[0] by one representable
float toward +infinity. Initial observation error is at most 9.31e-10;
it first exceeds 1e-3 at step 8, 1e-2 at step 11 and 0.1 at step 15. This
supports chaotic amplification after the corrected RNG-order bug; it does
not prove every long trajectory equivalent. No float64 physics rewrite or
broad long-trajectory tolerance was used.

Selected stored-observation/reward error records from the removed trajectory
JSON (max/mean over four seeds; FP32 storage intentionally differs from CUDA's
BF16 conversion). “Reset” is the separate forced-reset BF16 capture:

| Capture | Mode | Step | Stored obs max / mean | Reward max / mean |
|---|---|---|---|---|
| bf16 | pick | 0 | 0.0 / 0.0 | 0.0 / 0.0 |
| bf16 | pick | 4 | 3.0517578125e-05 / 1.1060734550483176e-07 | 3.213062882423401e-08 / 1.8102582544088364e-08 |
| bf16 | pick | 640 | 0.255859375 / 0.00280714728971058 | 3.475783159956336e-05 / 1.4092576748225838e-05 |
| bf16 | stack | 0 | 0.0 / 0.0 | 0.0 / 0.0 |
| bf16 | stack | 4 | 3.0517578125e-05 / 1.1062170702163845e-07 | 1.6007106751203537e-08 / 9.356881491839886e-09 |
| bf16 | stack | 640 | 0.255859375 / 0.002756948136578167 | 1.7881335224956274e-05 / 7.548758730990812e-06 |
| bf16 | basketball | 0 | 0.0 / 0.0 | 0.0 / 0.0 |
| bf16 | basketball | 4 | 3.0517578125e-05 / 2.211444821146042e-07 | 1.4202669262886047e-08 / 8.316419553011656e-09 |
| bf16 | basketball | 640 | 0.26171875 / 0.003938486132080117 | 2.727028913795948e-05 / 8.975259333965369e-06 |
| f32 | pick | 0 | 0.0019078254699707031 / 0.00012109472684813596 | 0.0 / 0.0 |
| f32 | pick | 4 | 0.0018750429153442383 / 0.00021171153840769344 | 3.213062882423401e-08 / 1.8102582544088364e-08 |
| f32 | pick | 640 | 0.255872517824173 / 0.0029064585736844986 | 3.475783159956336e-05 / 1.4092576748225838e-05 |
| f32 | stack | 0 | 0.0019078254699707031 / 9.326236091589595e-05 | 0.0 / 0.0 |
| f32 | stack | 4 | 0.0018750429153442383 / 0.00018294234562512846 | 1.6007106751203537e-08 / 9.356881491839886e-09 |
| f32 | stack | 640 | 0.255872517824173 / 0.002850198826985037 | 1.7881335224956274e-05 / 7.548758730990812e-06 |
| f32 | basketball | 0 | 0.0019078254699707031 / 0.00013312859553717198 | 0.0 / 0.0 |
| f32 | basketball | 4 | 0.0018750429153442383 / 0.00023023288379589346 | 1.4202669262886047e-08 / 8.316419553011656e-09 |
| f32 | basketball | 640 | 0.2624100148677826 / 0.004092821116371171 | 2.727028913795948e-05 / 8.975259333965369e-06 |
| forced_reset | pick | 2 | 0.0 / 0.0 | 2.1420419216156006e-08 / 5.529727786779404e-09 |
| forced_reset | stack | 2 | 0.0 / 0.0 | 1.0710209608078003e-08 / 2.7066562324762344e-09 |
| forced_reset | basketball | 2 | 0.0 / 0.0 | 1.4319084584712982e-08 / 7.101334631443024e-09 |

`bash tests/metal/run_test_robot_arm.sh` uses fresh outputs and gates only
initialization/steps 0–4 plus reset, both precisions, three modes/four seeds.
Raw absolute limits are [5e-7, 1e-6, 3e-6, 1.2e-5, 4e-5]; stored observations
allow 4e-5, rewards 1e-7 and terminals must be exact. CPU FP32 storage must
exactly equal raw observations; BF16 must exactly equal their RNE conversion.
Removing joint gravity or quantizing CPU FP32 observations incorrectly is
rejected. BF16/FP32 smoke training runs 262,144 steps with 256 agents and
unchanged H256/L2/horizon/minibatch, finite weights and episode metrics.
The historical complete port suite passed 40/40.

## Preliminary 33.6M-step comparison

BF16, seed 42, 33,554,432 steps/64 epochs, unchanged configs except budget,
ten CPU threads on Metal and GPU physics on CUDA, then 1024-agent/episode
evaluation. Reducing total steps also shortens annealing schedules, so this
run is separate from the default-budget comparison. Exact final JSON values:

| Backend | Training seconds | Mean SPS | Eval success | Eval return | Eval grasp | Eval lift |
|---|---|---|---|---|---|---|
| metal | 269.9084167480469 | 124317.84234176835 | 0.0 | -4.365119457244873 | 0.0 | 0.0 |
| cuda | 67.23078918457031 | 499093.23402231984 | 0.0 | -4.347153663635254 | 0.0087890625 | 0.0029296875 |

The four recorded training bins are snapshot means, not aligned episodes:

| Backend | Mean steps | Episode return | Grasp | Lift |
|---|---|---|---|---|
| metal | 5898240.0 | -4.298267960548401 | 0.0 | 0.0 |
| metal | 14505301.333333334 | -4.345358212788899 | 0.0 | 0.0 |
| metal | 23068672.0 | -4.362840533256531 | 0.0 | 0.0 |
| metal | 31457280.0 | -4.365726947784424 | 0.0 | 0.0 |
| cuda | 5898240.0 | -4.299702405929565 | 0.0001220703125 | 0.0 |
| cuda | 14942208.0 | -4.143436074256897 | 0.0006715954514220357 | 0.00012212994624860585 |
| cuda | 21612316.444444444 | -4.382853560977512 | 0.0011671123146596882 | 0.000380012892290122 |
| cuda | 29826161.777777776 | -4.290919886695014 | 0.0024706621447371114 | 0.000814538397308853 |

Success remained zero. CUDA showed occasional grasps/lifts while Metal did
not; this single short comparison established execution, not learning parity.

## Full-budget three-seed comparison

Measured October 26–27, 2026. Raw results, all curves, provenance, telemetry
and full-precision statistics remain in
[robot-arm-learning-results.json](robot-arm-learning-results.json).
Metal: M4, ten CPU cores, 16 GiB, macOS 27.0.1, CPU physics, BF16,
PUF_ENV_THREADS=10. CUDA: RTX 3090/CUDA 12.8, GPU physics, BF16, nice 19.
Seeds 42/43/44 ran sequentially per backend. Optional 45/46 were not run.
Two partial CUDA 42 attempts ending at 351,797,248 and 221,249,536 logged
steps had no final evaluation and are excluded.

The configured 1,000,000,000 steps floor to 1,907 epochs / 999,817,216 steps.
Training is 4096 agents, horizon 128, H256/L2, minibatch 32768, replay ratio 1,
learning rate 0.0003, entropy coefficient 0.002 and default schedules. Only pick is enabled.
Config bytes and completed INI train/policy/env sections match across hosts:

| File | SHA-256 |
|---|---|
| `config/default.ini` | `7f493977f49f2537a59c18c2a283f062e5d6063d9e130c2d15f6be075cda1084` |
| `config/robot_arm.ini` | `b207726ed51ea222f49d6f39bda90b854a3d60757e3ffccc5f7131c3e488a461` |

All six exit codes are zero and each final evaluation has exactly 1024
episodes on a new 1024-agent vector, final checkpoint and stochastic policy.
The INI's final agent count reflects evaluation, not training geometry.
Environment seeds remain agent index+1 independently of policy seed.

CUDA's separate reference source applied only the
[logging patch](../../tools/metal/robot_arm_learning_cuda_logfix.patch)
described in [training history](parity-investigations.md#training-log-history).
Physics, learner, RNG, CUDA graphs, checkpointing and cadence are unchanged.
Training hours are final INI uptime/3600; mean SPS is 999,817,216/uptime.
Evaluation preserves training uptime. CUDA excludes graph-capture setup.
Last drained-epoch SPS and arithmetic mean snapshot SPS are not training
throughput; runner timestamps include setup/evaluation/polling delays.

### Results and statistics

Values retain the original report's displayed precision; the JSON retains
full precision. Rates are episode rates and pick score equals success.

| Backend | Seed | Training hours | Mean SPS | Eval success | Eval return | Eval grasp | Eval lift |
|---|---:|---:|---:|---:|---:|---:|---:|
| Metal + CPU | 42 | 3.027 | 91,738 | 0.000% | -4.27456 | 0.2930% | 0.000% |
| Metal + CPU | 43 | 3.432 | 80,916 | 0.000% | -4.26959 | 3.9062% | 0.000% |
| Metal + CPU | 44 | 3.003 | 92,498 | 0.000% | -4.26379 | 9.7656% | 0.000% |
| CUDA + GPU | 42 | 1.725 | 161,032 | 0.000% | -4.26819 | 7.5195% | 0.000% |
| CUDA + GPU | 43 | 1.909 | 145,514 | 0.000% | -4.26715 | 10.0586% | 0.000% |
| CUDA + GPU | 44 | 1.851 | 150,060 | 0.000% | -4.27064 | 7.9102% | 0.000% |

| Backend | Metric | Mean | Sample SD | Range |
|---|---|---:|---:|---|
| Metal | Success (%) | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| Metal | Return | -4.26932 | 0.00539 | [-4.27456, -4.26379] |
| Metal | Grasp (%) | 4.6549 | 4.7805 | [0.2930, 9.7656] |
| Metal | Lift (%) | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| Metal | Training hours | 3.154 | 0.241 | [3.003, 3.432] |
| Metal | Mean SPS | 88,384 | 6,478 | [80,916, 92,498] |
| CUDA | Success (%) | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| CUDA | Return | -4.26866 | 0.00179 | [-4.27064, -4.26715] |
| CUDA | Grasp (%) | 8.4961 | 1.3672 | [7.5195, 10.0586] |
| CUDA | Lift (%) | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| CUDA | Training hours | 1.828 | 0.094 | [1.725, 1.909] |
| CUDA | Mean SPS | 152,202 | 7,978 | [145,514, 161,032] |

| Metric | Metal − CUDA mean | Welch t | df | Welch p | Exact Mann–Whitney U (Metal) | Exact p | Welch 95% CI for mean difference |
|---|---:|---:|---:|---:|---:|---:|---|
| Return | −0.000657 | −0.2002 | 2.4354 | 0.8568 | 4 | 1.0000 | [−0.012604, 0.011291] |
| Grasp (percentage points) | −3.8411 | −1.3381 | 2.3250 | 0.2967 | 2 | 0.4000 | [−14.6771, 6.9949] |

Grasp counts in seed order are 3, 40, 100 / 1024 on Metal and 77, 103, 81 / 1024
on CUDA. Metal's mean is 3.8411 percentage points lower with much greater
observed seed variation; return means differ by −0.000657. The replication
unit is training seed (n=3 per backend), not 3072 independent trained policies.
All evaluations share the same initial environment-seed set; equal policy
seed labels do not ensure paired random streams across backends.

Tests are exploratory, independent-sample, two-sided and unadjusted for
multiplicity, with no prespecified equivalence margin. Welch assumptions
cannot be assessed with n=3 and its approximate intervals are fragile for
bounded rates. Exact rank p-values were checked by enumerating all 20
assignments; the smallest possible two-sided p is 0.1, even for complete
rank separation. Success/lift tests are undefined at zero variance and
omitted. Non-rejection permits a meaningful gap; it is not a probability
of equivalence or evidence of equal population success rates.

All final success/lift rates are zero. Both backends exhibit some training
grasps/lifts; CUDA 44's first two success bins were 0.211416% and 0.000071%,
with none in the final half or final evaluation. Four training-bin snapshot
means (not episode-weighted) and a separate final evaluation overlay remain
in the JSON. The result supports comparable failure to solve pick at this
budget, with no statistically detected learning-quality regression.

### Throughput and validation limits

Aggregate CUDA training: 19,742.53 seconds (5.484 hours); Metal: 34,063.91
seconds (9.462 hours). Mean per-seed rates 152,202 versus 88,384 SPS give
approximately 1.72× for CUDA, comparing whole systems including GPU versus
CPU physics, not isolated learner speed or energy efficiency. Full Metal
runs took 3.003–3.432 hours, longer than short-run extrapolation.

Metal warmup-excluded summaries:

| Seed | Summary SPS | CPU env ms/epoch | Rollout model ms/epoch | Learner ms/epoch | Learner GPU ms/epoch | Thermal samples |
|---|---:|---:|---:|---:|---:|---:|
| 42 | 91,753 | 5,278 | 437.2 | 642.7 | 633.5 | 182 |
| 43 | 80,923 | 6,016 | 463.9 | 640.5 | 631.3 | 206 |
| 44 | 92,514 | 5,236 | 432.4 | 643.5 | 634.3 | 181 |

Counters overlap; do not sum CPU environment/model/learner time. Learner
cost remains near 641–644 ms while environment time varies. All 569 sampled
pmset observations reported no warnings, AC was used and low-power mode was
off. Power in watts was not measured; absence of warnings does not exclude
frequency/power caps. CUDA pre-run telemetry showed no compute processes,
but exclusivity throughout each run is not established and possible late
host activity was not removed from timing.

Metal checkpoints were independently rechecked for finite FP32 weights;
each is 1,652,768 bytes. CUDA logs/INIs establish configuration, exact
steps/epochs, finite arrays, evaluation count and final checkpoint reference.
CUDA checkpoint bytes/hashes/finite weights were not independently validated;
they stayed on the reference host. Do not infer that validation from exit 0.

Existing gates cover H256 primitives and discrete Freeway H128/Tetris H512
train replay. FP32 synthetic sampling/PPO covers continuous heads, but BF16
golden sampling/PPO uses empty logstd tensors and discrete heads. There is
no complete H256/L2 eight-head continuous BF16 learner replay. Stack and
basketball have trajectory/reset coverage only; rendering was not exercised.

Stronger parity evidence needs more independent seeds and a prespecified
practical equivalence margin. Cross-evaluating each policy on both backends,
identical-action physics replay across resets, then a matched continuous
BF16 learner update would help isolate any grasp gap. These experiments
have not been performed. Shader physics might reduce the dominant CPU cost,
but roughly 100 KiB state per environment and the contact solver make memory
layout/register spilling significant design constraints.

## Reproduction

From the repository root with dependencies and Python environment available:

```sh
./build.sh robot_arm build/ra/train_bf16 --metal
for seed in 42 43 44; do
    out="build/ra/metal/seed${seed}"
    mkdir -p "$out"
    PUF_ENV_THREADS=10 build/ra/train_bf16 train \
        --base.seed="$seed" --base.profile=1 \
        --base.eval_episodes=1024 --base.eval_agents=1024 \
        --base.run_id="metal_seed${seed}" \
        --base.log_dir="$out/logs" --base.checkpoint_dir="$out/checkpoints" \
        > "$out/train.log" 2>&1
done
```

The historical CUDA build compiled the trainer directly because its build
script rejected obs_t defined in robot_arm's included header. Configure
CUDA_HOME/NCCL_HOME and the normal native CUDA dependencies. Apply the
logging patch only to a separate source output of the historical reference
(the current Metal tree already has the fix), then compile:

```sh
mkdir -p build/ra
patch -o build/ra/pufferl-logfix.cu src/pufferl.cu < tools/metal/robot_arm_learning_cuda_logfix.patch
nice -n 19 "$CUDA_HOME/bin/nvcc" -O2 --threads 1 -arch=sm_86 -std=c++17 \
    -I. -Isrc -Iocean/robot_arm -Ivendor -Iraylib-5.5_linux_amd64/include \
    -I"$CUDA_HOME/include" -I"$CUDA_HOME/include/cccl" -I"$NCCL_HOME/include" \
    '-DENV_HEADER="ocean/robot_arm/robot_arm.cu"' -DENV_NAME=robot_arm \
    '-DPUFFER_ENV_NAME="robot_arm"' -DPUFFERLIB_BUILD_MAIN \
    -Xcompiler=-DPLATFORM_DESKTOP -Xcompiler=-fopenmp \
    -Xcompiler=-Wno-narrowing -Xcompiler=-Wno-unreachable-code \
    --diag-suppress=2361 --diag-suppress=111 --diag-suppress=128 \
    build/ra/pufferl-logfix.cu raylib-5.5_linux_amd64/lib/libraylib.a \
    -L"$CUDA_HOME/lib64" -L"$NCCL_HOME/lib" \
    -lcudart -lnccl -lnvidia-ml -lcublas -lcusolver -lcurand \
    -lm -Xlinker=-lpthread -lomp5 -lGL -o build/ra/train_bf16
```

Use the same training arguments, CUDA output paths and nice 19; omit
PUF_ENV_THREADS. With checkpoints in build/ra/{metal,cuda}/seed{seed}, run
`python3 tools/metal/robot_arm_learning_results.py --output build/ra/results-regenerated.json`.
The collector requires checkpoints; the copied CUDA folders without them
cannot be fully validated by it. Statistics used SciPy 1.18.1 Welch ttest_ind
(equal_var=False) and exact two-sided mannwhitneyu, plus independent rank
enumeration. The retained JSON records all full-precision inputs/results.
