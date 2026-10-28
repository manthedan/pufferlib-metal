# Robot-arm full-budget learning comparison

**Verdict: no detected learning-quality gap at this budget; equivalence is
not established. Neither backend solves the default task within 1B steps.**
All three Metal and all three CUDA seeds completed the full default budget.
Every final evaluation has zero success and lift. CUDA has higher mean grasp,
but the three-seed distributions overlap and exploratory tests do not reject
a common distribution. This is limited evidence of comparable unsuccessful
learning, not proof of backend equivalence or of a Metal regression.

Measured October 26–27, 2026 (America/Los_Angeles). Exact metrics, training
curves, sample statistics, tests and artifact hashes are in
[robot-arm-learning-results.json](robot-arm-learning-results.json).

## Protocol

The Metal branch is `metal-ra-learn`, starting at
`5f0bede592e67aeee04a7daf58c02dd8358a012c`. Hardware is a Mac mini M4,
10 CPU cores, 16 GiB, macOS 27.0.1 (26A434), Apple Clang 21, Bash 3.2.57.
Metal uses CPU physics, the BF16 learner and `PUF_ENV_THREADS=10`.
Seeds 42, 43 and 44 ran sequentially. No additional heavy jobs were launched
by this experiment on the Metal machine; OS background activity was possible.

The CUDA reference host uses an RTX 3090, CUDA 12.8, GPU physics and the BF16
learner. Seeds 42, 43 and 44 completed sequentially with nice level 19 and
`--base.eval_episodes=1024 --base.eval_agents=1024 --base.profile=1`.
All exit codes are zero. Each pre-run GPU snapshot reports no running
processes. The experiment operator reports that the CUDA reference host was
otherwise idle except possibly at the very end. Device telemetry cannot
establish exclusivity throughout each run; no timing intervals were removed.

Two earlier partial CUDA seed-42 attempts (last logged steps 351,797,248 and
221,249,536) were stopped without final evaluation. They remain excluded;
the completed runs supersede the previous pending status. Optional seeds
45–46 were not run.

Both config files were recorded as identical across hosts:

| File | SHA-256 |
|---|---|
| `config/default.ini` | `7f493977f49f2537a59c18c2a283f062e5d6063d9e130c2d15f6be075cda1084` |
| `config/robot_arm.ini` | `b207726ed51ea222f49d6f39bda90b854a3d60757e3ffccc5f7131c3e488a461` |

The completed INIs also match exactly in their `train`, `policy` and `env`
sections. The unchanged default budget is **1,000,000,000 steps**. With
4,096 training agents and horizon 128, the trainer floors this to
**1,907 epochs / 999,817,216 steps**. The policy is H256/L2 with eight
continuous action heads, minibatch 32,768, replay ratio 1, learning rate
0.0003, entropy coefficient 0.002 and the default annealing schedules.
Pick is the default task; stack and basketball are disabled. The earlier
33.6M-step experiment used a different budget and annealing schedule and
is not part of this comparison.

Final evaluation uses a newly created vector of 1,024 agents, the final
checkpoint and standard stochastic policy sampling. All six runs evaluated
**exactly 1,024 completed episodes**. The saved INI reflects evaluation's
mutation of `vec.total_agents` to 1,024; training used 4,096. Policy
initialization and sampling use `base.seed`; environment initialization uses
agent index + 1 on both backends, independently of `base.seed`.

## Logging and measurement

The CUDA reference predates the logging correction described in
[logfix-report.md](logfix-report.md). Its separate source copy and binary
under `build/ra/` apply only the corrected history-bin function and snapshot
update/admission logic. The patch is
[robot_arm_learning_cuda_logfix.patch](../../../tools/metal/robot_arm_learning_cuda_logfix.patch).
Reference physics, learner math, RNG, CUDA graphs, checkpointing and logging
cadence are unchanged. This update changes documentation and parsed data only.

The four training-curve points are arithmetic means of logged snapshots in
equal-width, half-open step intervals. Missing episode statistics carry
forward as specified by the logging fix. These are not episode-weighted
means. The fifth saved point is the final evaluation overlay, kept separate
from the training curves on both backends.

**Training hours = final INI `uptime` / 3,600; mean SPS = 999,817,216 / final
INI `uptime`.** Evaluation replaces environment metrics while preserving
training uptime. CUDA excludes graph-capture setup from that uptime. Metal
also reports a separate warmup-excluded `metal_summary` SPS. Neither the
last drained-epoch SPS (millions of steps/s on CUDA) nor the arithmetic mean
of snapshot/bin SPS is the mean training throughput. Runner timestamps
include setup, evaluation and up to 60 seconds of polling delay and are not
used for training hours or SPS.

Copied CUDA artifacts are under `build/ra_cuda_final/seed{42,43,44}/`.
The final CUDA checkpoints at step 999,817,216 remain on the CUDA reference
host, as reported by the experiment operator. The final INIs reference those
checkpoints; their bytes, hashes and finite weights were **not independently
validated in this update**. Metal checkpoints were revalidated locally.
Raw CUDA artifacts total 13,839,154 bytes (522,132 bytes when individually
gzipped). Only parsed summaries, curves and source hashes are committed in
the JSON; raw logs remain in the ignored build directory.

## Semantic audit and diagnostic scope


The CPU adapter invokes the same `ra_kbegin`, `ra_kphys`, and `ra_kfin`
bodies as CUDA. Action clamping precedes the same eight physics substeps.
On terminal, `ra_kfin` logs the completed episode, resets the physical state,
then writes the reset observation together with the terminal flag and the
completed step's reward. There is no extra vector-worker reset. Observation
normalization lives in the shared `ra_observe`; BF16 storage uses
round-to-nearest-even on both backends. The earlier basketball RNG-order
correction is present, although these runs exercise pick.

With one policy, one buffer and one agent per environment, CPU pointer
assignment preserves environment/agent order. OpenMP's static workers step
independent states; RNG is stored per environment. CUDA uses corresponding
contiguous agent offsets. CPU workers clear rewards and terminals before
stepping; the shared environment writes both values on every step. CPU/GPU
episode-log reductions use different summation orders, which can change
floating-point aggregate values but does not by itself imply different
episode events.

The existing wide gates include H256 GEMM, scan, gate and policy-layer
coverage. Their end-to-end learner replay fixtures use Freeway H128/L4 and
Tetris H512/L2, with discrete actions. FP32 synthetic kernel tests cover
continuous sampling and PPO, but the BF16 golden PPO/sampling readers pass
empty log-standard-deviation tensors and exercise discrete actions. Thus
the gates do **not** constitute a full H256/L2, eight-head continuous-action
BF16 learner parity test. This is a coverage limitation, not evidence of a
learner defect.

## Results and verdict

Success, grasp and lift are episode rates; `env/score` equals
`env/success_rate` in the default pick task.

| Backend | Seed | Training hours | Mean SPS | Eval success | Eval return | Eval grasp | Eval lift |
|---|---:|---:|---:|---:|---:|---:|---:|
| Metal + CPU | 42 | 3.027 | 91,738 | 0.000% | -4.27456 | 0.2930% | 0.000% |
| Metal + CPU | 43 | 3.432 | 80,916 | 0.000% | -4.26959 | 3.9062% | 0.000% |
| Metal + CPU | 44 | 3.003 | 92,498 | 0.000% | -4.26379 | 9.7656% | 0.000% |
| CUDA + GPU | 42 | 1.725 | 161,032 | 0.000% | -4.26819 | 7.5195% | 0.000% |
| CUDA + GPU | 43 | 1.909 | 145,514 | 0.000% | -4.26715 | 10.0586% | 0.000% |
| CUDA + GPU | 44 | 1.851 | 150,060 | 0.000% | -4.27064 | 7.9102% | 0.000% |

Across-seed summaries use arithmetic means and **sample SD (n−1)**, with
n=3 per backend. SDs for rates are in percentage points. Mean SPS below is
the arithmetic mean of the three per-run rates.

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

Grasp counts are **3, 40, 100 / 1,024** on Metal and **77, 103, 81 / 1,024**
on CUDA, in seed order 42–44. Metal's grasp mean is 3.8411 percentage points
lower, with much larger observed seed variation. Its highest seed overlaps
the CUDA range. Return means differ by only −0.000657 (Metal minus CUDA).

The replication unit is the **training seed (n=3 per backend)**, not 3,072
independent trained policies. Episodes share a learned policy, and all runs
use the same initial environment-seed set. Matching seed labels do not
ensure matched stochastic streams across backends, so these are exploratory
independent-sample comparisons. All tests are two-sided, without multiplicity
adjustment; no equivalence margin was specified.

| Metric | Metal − CUDA mean | Welch t | df | Welch p | Exact Mann–Whitney U (Metal) | Exact p | Welch 95% CI for mean difference |
|---|---:|---:|---:|---:|---:|---:|---|
| Return | −0.000657 | −0.2002 | 2.4354 | 0.8568 | 4 | 1.0000 | [−0.012604, 0.011291] |
| Grasp (percentage points) | −3.8411 | −1.3381 | 2.3250 | 0.2967 | 2 | 0.4000 | [−14.6771, 6.9949] |

Welch tests allow unequal variances but their distributional assumptions
cannot be assessed with three observations per group. The intervals are
approximate and especially fragile for bounded grasp rates. The exact rank
tests have no ties for return or grasp and were independently checked by
enumerating all 20 assignments of six observations to two groups of three.
Their smallest possible two-sided p-value is 0.1: even complete rank
separation could not reach 0.05. See the SciPy documentation for
[Welch's test](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html)
and the [exact Mann–Whitney test](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mannwhitneyu.html).

**Metal's grasp results plausibly come from the same distribution as CUDA's:**
these observations do not reject that explanation, while also allowing a
meaningful backend gap. A p-value is not the probability that the distributions
are equal. The differing sample spreads are too sparsely estimated to
establish a variance difference. Success and lift are identically zero;
Welch tests are undefined with zero variance and are omitted. Their observed
agreement does not establish equal population success or lift probabilities.

At this budget, the evidence supports **comparable failure to solve pick**,
with no statistically detected learning-quality regression. It does not
establish learning equivalence, and the grasp mean gap should remain visible
rather than being dismissed as proven noise. The unchanged default config
does not solve the task on either backend within the nominal 1B-step budget.

### Training curves

The four saved training-bin means are shown at their actual mean step
positions. CUDA seed 44 has nonzero success means in its first two bins
(0.21142% and 0.000071%); all other success-bin means are zero. These are
snapshot means, not pooled success rates or counts. Bins span approximately
0–250M, 250–500M, 500–750M and 750–999.8M steps and weight snapshots, not
completed episodes. Final evaluation values above are separate.

| Backend | Seed | Mean steps (M) | Success | Episode return | Grasp | Lift | Mean snapshot SPS |
|---|---:|---:|---:|---:|---:|---:|---:|
| Metal | 42 | 125.83 | 0.000000% | -4.36596 | 1.2637% | 0.08475% | 104.9k |
| Metal | 42 | 374.87 | 0.000000% | -4.40724 | 2.0036% | 0.02693% | 94.5k |
| Metal | 42 | 624.95 | 0.000000% | -4.28862 | 2.8931% | 0.00016% | 82.9k |
| Metal | 42 | 874.77 | 0.000000% | -4.27245 | 0.2410% | 0.00000% | 90.7k |
| Metal | 43 | 125.83 | 0.000000% | -4.28665 | 3.3578% | 0.10504% | 94.2k |
| Metal | 43 | 374.87 | 0.000000% | -4.38083 | 3.1796% | 0.00861% | 85.6k |
| Metal | 43 | 624.95 | 0.000000% | -4.27695 | 4.1023% | 0.00000% | 74.4k |
| Metal | 43 | 874.77 | 0.000000% | -4.26936 | 4.2110% | 0.00000% | 75.3k |
| Metal | 44 | 125.83 | 0.000000% | -4.30329 | 3.2455% | 0.19413% | 104.2k |
| Metal | 44 | 374.87 | 0.000000% | -4.30791 | 2.3178% | 0.02977% | 97.3k |
| Metal | 44 | 624.95 | 0.000000% | -4.26614 | 8.3346% | 0.00000% | 88.2k |
| Metal | 44 | 874.77 | 0.000000% | -4.26311 | 10.2856% | 0.00000% | 84.8k |
| CUDA | 42 | 125.83 | 0.000000% | -4.36303 | 3.8073% | 0.31193% | 228.4k |
| CUDA | 42 | 374.87 | 0.000000% | -4.35467 | 5.4889% | 0.11297% | 165.8k |
| CUDA | 42 | 624.95 | 0.000000% | -4.27672 | 5.1576% | 0.00276% | 156.5k |
| CUDA | 42 | 874.77 | 0.000000% | -4.26848 | 7.4870% | 0.00036% | 141.9k |
| CUDA | 43 | 125.83 | 0.000000% | -4.29229 | 5.3749% | 0.39729% | 208.3k |
| CUDA | 43 | 374.87 | 0.000000% | -4.34332 | 4.9891% | 0.15348% | 137.5k |
| CUDA | 43 | 624.95 | 0.000000% | -4.27568 | 6.4105% | 0.00109% | 151.2k |
| CUDA | 43 | 874.77 | 0.000000% | -4.26740 | 9.2040% | 0.00024% | 129.4k |
| CUDA | 44 | 125.83 | 0.211416% | -4.34124 | 4.5420% | 0.78043% | 189.9k |
| CUDA | 44 | 374.87 | 0.000071% | -4.33415 | 4.2809% | 0.09151% | 145.6k |
| CUDA | 44 | 624.95 | 0.000000% | -4.27270 | 7.0618% | 0.00141% | 156.2k |
| CUDA | 44 | 874.77 | 0.000000% | -4.26871 | 8.4651% | 0.00249% | 143.9k |

Both backends show some grasp behavior. CUDA seed 44 has transient early
training success, but no success persists into the final half of training or
any final evaluation. Rare training lifts likewise do not persist into final
evaluation. Improved returns and grasp rates should not be described as
solving pick-and-place.

### Throughput, power and thermal notes

CUDA consumed **19,742.53 training seconds (5.484 hours)** in aggregate,
versus Metal's **34,063.91 seconds (9.462 hours)**. CUDA trained each seed in
1.725–1.909 hours; its average per-seed mean SPS was 152,202 versus 88,384
for Metal, a ratio of approximately **1.72×**. This compares two complete
systems, including GPU versus CPU physics, and does not isolate learner
speed or energy efficiency. Possible late CUDA host activity and ordinary
background variation limit precise hardware attribution. CUDA device
telemetry summaries are in the JSON; they exclude zero-utilization samples
using the existing collector's convention.

The three Metal runs consumed **34,063.91 training seconds (9.462 hours)** in
aggregate, excluding final evaluation and runner polling delays. Full-budget
training took approximately 3.00–3.43 hours per seed, longer than the prior
short-run extrapolation. The default checkpoint interval of ten epochs was
retained. Warmup-excluded trainer summaries are:

| Seed | Summary SPS | CPU env ms/epoch | Rollout model ms/epoch | Learner ms/epoch | Learner GPU ms/epoch | Thermal samples |
|---|---:|---:|---:|---:|---:|---:|
| 42 | 91,753 | 5,278 | 437.2 | 642.7 | 633.5 | 182 |
| 43 | 80,923 | 6,016 | 463.9 | 640.5 | 631.3 | 206 |
| 44 | 92,514 | 5,236 | 432.4 | 643.5 | 634.3 | 181 |

CPU environment time dominates; actor/environment and learner work overlap,
so these counters must not be summed into wall time. Learner time stays near
641–644 ms while CPU environment time differs across seeds. Snapshot SPS
also varies within a run; the high final drained-epoch SPS is not the mean
training rate used in the results table. Seed 44 briefly dipped to about
55–58k SPS near its end, then recovered.

`pmset -g therm` reported no thermal or performance warnings in **569**
complete sampled observations. The machine used AC power with low-power
mode off. **Mac power in watts was not measured**: there was no external
wattmeter or privileged power sampling. The absence of `pmset` warnings does
not prove that power or frequency capping never occurred. OS background
activity remained possible; the postrun process snapshot showed an active
`ANECompilerService`. That snapshot does not establish the cause of the late
throughput dip. No additional heavy workload from this experiment ran on
the mini concurrently with a Metal seed.

### Validation and next discriminating experiment

All six processes exited 0. For CUDA, this update validates the full-budget
configuration, matching learning/environment sections, 1,907 epochs, exact
step count, finite five-point metric arrays, exactly 1,024 evaluated episodes,
pick score/success agreement and the final checkpoint reference. The same
final-INI extraction and timing formulas as the existing collector are used.
CUDA artifact hashes, sizes, runner timestamps and parsed telemetry are in
the JSON. Complete source INIs and logs remain available locally for audit.

The unchanged [collector](../../../tools/metal/robot_arm_learning_results.py)
reproduced every original Metal record and rechecked finite FP32 checkpoint
weights. Each Metal checkpoint is 1,652,768 bytes. The collector requires
checkpoint files, so it cannot directly validate the copied CUDA folders:
this report does not claim CUDA checkpoint hash or weight validation.
Statistical calculations use SciPy 1.18.1; exact rank p-values also match
independent enumeration. No training, production source changes or numerical
changes were made for this update.

No additional experiment is required to report these results. A stronger
parity claim would need more independent training seeds and a prespecified
practical equivalence margin. If investigating the grasp gap, first evaluate
each backend's final policies on both backends with the same 1,024-agent
setup. This tests policy transfer while still combining physics and
inference differences. Identical-action CPU/CUDA replay across resets can
then isolate environment semantics. If training remains suspect, replay one
H256/L2 eight-head continuous BF16 update with identical rollout, parameter,
optimizer and RNG state to address the golden-gate coverage gap. These
cross-backend diagnostics have not been performed.

## Reproduction

Run commands from the repository root with the project Python environment
activated. Metal (run the seeds sequentially):


```sh
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
export PATH="$VIRTUAL_ENV/bin:$PATH"
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

On the CUDA reference host, configure `CUDA_HOME` and `NCCL_HOME`, copy the
committed logging patch to `build/ra/cuda-logfix.patch`, and build a separate
output using the normal native flags:

```sh
mkdir -p build/ra
patch -o build/ra/pufferl-logfix.cu src/pufferl.cu < build/ra/cuda-logfix.patch
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

The direct compile avoids the existing reference `build.sh` rejection of
the `obs_t` typedef in robot_arm's included header. Before each run, inspect
`nvidia-smi` and wait for other compute processes to finish. Use the Metal
training arguments above with `cuda` in output paths/run IDs, no
`PUF_ENV_THREADS` override and `nice -n 19` before the reference binary.
Retain the default budget and config. Completed CUDA outputs were copied to
`build/ra_cuda_final/seed{seed}` without checkpoints.

With checkpoints present under `build/ra/{metal,cuda}/seed{seed}`, the
unchanged collector can regenerate the core numeric summary:

```sh
python tools/metal/robot_arm_learning_results.py --output build/ra/results-regenerated.json
```

For the supplied CUDA copies without checkpoints, the following reproduces
the final metric/timing extraction and the reported statistics (requires
NumPy and SciPy). It uses the collector for the locally available Metal
checkpoints; CUDA checkpoint validation remains unavailable:

```python
import configparser
from pathlib import Path
import statistics
from scipy import stats
from tools.metal.robot_arm_learning_results import METRICS, STEPS, read_run

runs = [read_run(Path(f"build/ra/metal/seed{s}"), "metal") for s in (42, 43, 44)]
for seed in (42, 43, 44):
    folder = Path(f"build/ra_cuda_final/seed{seed}")
    assert (folder / "exit_code.txt").read_text().strip() == "0"
    ini = configparser.ConfigParser(interpolation=None, strict=False)
    path, = (folder / "logs").rglob("*.ini")
    ini.read(path)
    last = {k: float(v.split(",")[-1]) for k, v in ini["metrics"].items()}
    assert last["agent_steps"] == STEPS and last["epoch"] == 1907
    assert last["env/n"] == 1024
    runs.append(dict(backend="cuda", seed=seed,
        training_seconds=last["uptime"], mean_sps=STEPS / last["uptime"],
        eval={k: last[f"env/{k}"] for k in METRICS}))
for r in runs:
    r["training_hours"] = r["training_seconds"] / 3600
    print(r["backend"], r["seed"], r["eval"], r["training_hours"], r["mean_sps"])
for backend in ("metal", "cuda"):
    rs = [r for r in runs if r["backend"] == backend]
    for key in (*METRICS, "training_hours", "mean_sps"):
        xs = [r["eval"][key] if key in METRICS else r[key] for r in rs]
        print(backend, key, statistics.mean(xs), statistics.stdev(xs), min(xs), max(xs))
for key in ("episode_return", "grasp_rate"):
    a, b = ([r["eval"][key] for r in runs if r["backend"] == backend]
            for backend in ("metal", "cuda"))
    t = stats.ttest_ind(a, b, equal_var=False)
    print(key, t, t.confidence_interval(),
          stats.mannwhitneyu(a, b, alternative="two-sided", method="exact"))
```
