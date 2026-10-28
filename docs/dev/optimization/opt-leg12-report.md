> Historical development log; last recorded at commit `8c866d2f` (tree archived from `e68239d9`). Statements and gate counts describe that revision, not current support.

# Leg 12 (Sol): BF16 custom Metal env encoders

Worktree `<repo>/wt-opt12`, branch
`metal-opt12`, based on `7538b089`. Nothing pushed. All nine requested envs
now build and train in default BF16; `--float` remains supported. The
FP32 fallback in `tools/metal/build_native.sh` is removed, and the sweep
builds default precision.

## Implementation

`kernels_envs.h` no longer rejects BF16. Entity, OSRS, Craftax and NMMO3
shaders use `precision_t` device storage with explicit widening to FP32 and
RNE conversion at CUDA's stores. Reductions retain FP32 accumulators.
NMMO3's BF16 embedding copy uses storage-typed elements; its FP32 float4
copy stays unchanged. OSRS's item table stays FP32. Its forward GELU tile
also stays FP32: only the saved backward `h1` is rounded, exactly as CUDA.
Craftax/NMMO3/NetHack retain the associative 2^24 int64 embedding sums.

NetHack's encoder/decoder already had storage-typed arguments and explicit
conversions except two backward stores. Those stores now use `from_float`.
The conditional sampling, cache and PPO shader pointers now follow the
upstream split: BF16 logits/masks/rollout values/advantages, FP32 actions,
logps, PPO gradients, losses and epsilon. Index/argmax scratch remains
FP32 or int32 as upstream declares it. Unused generic learner paths and
Breakout arithmetic are unchanged.

## Per-env status and CUDA goldens

CUDA: CUDA-reference `<cuda-reference>/pufferlib-envs-bf16`, branch
`golden-envs-bf16`, final reference commit `7f027ee0`. Started at
`golden-envs` `920a88e9`; cherry-picked the precision-generic harness from
`13497e22` and `da2516e4` as `177d287c` and `3f389444`. Added BF16 env harness
`bce6d7ca`, then Asteroids' configured hidden size in `7f027ee0`.

Both `run_a` and `run_b` are under `<cuda-reference>/golden_envs_bf16` and
were rsynced to `<repo>/golden/envs_bf16`.
Reference GPU: RTX 3090, driver 580.178.04. Always source
`<cuda-reference>/env.sh` first. No system packages or other users'
processes were changed. The relocated bison required
`M4="$PUFFER_REFERENCE_ROOT/bin/m4"`; native NetHack training on that older
checkout needed `NVCC_EXTRA="-include tests/metal_ref/cuda_launch.h"`.
Both are documented in `tests/metal_ref/README-env-bf16.md`.

The two CUDA runs are byte-identical for **all 18 cases** (every tensor):
**CUDA noise = 0**. `determinism.json`, `inventory.json`, `commit.txt` and
`gpu.txt` are beside the goldens. Metal and CUDA use the same harness
header (verbatim copy on CUDA-reference). Each env has a partial-tile case B=13,
H=40 and a configured-hidden case; default B=520 except NMMO3 B=1500.

| Env | Default golden H | BF16 default / learner smoke | Max Metal/CUDA norm error, both cases | Golden gate |
|---|---:|---|---:|---|
| minimal | 128 | OK | 0.003165 | PASS |
| asteroids | 256 | OK | 0.002174 | PASS (configured case entirely exact) |
| osrs_zulrah | 128 | OK | 0.005128 | PASS |
| osrs_pvp | 128 | OK | 0.003953 | PASS |
| osrs_colosseum | 512 | OK | 0.006369 | PASS |
| osrs_inferno | 1024 | OK | 0.004425 | PASS |
| craftax | 1024 | OK | 0.006944 | PASS |
| nmmo3 | 512 | OK | 0.006803 | PASS |
| nethack | 1024 | OK | 0.009346 | PASS |

Norm error is max absolute difference / max absolute CUDA tensor value.
Both CUDA runs pass separately: **1196 tensor comparisons per run**, plus
**290 independent exact/FP64 checks** on Metal replay intermediates.
NetHack includes the pointer decoder, key/spell gradients and conditional
sampling/PPO at both epsilon endpoints. Backward restores CUDA's saved
forward state and pooling choices, and checks decoder outputs before
restoring the encoder-backward boundary.

## Tolerances and stale-buffer protection

No tolerances were fitted to failing tensors. The full-network allowance
is the existing leg-9 policy bound: **4 BF16 ulp or 2^-5 tensor max**.
Direct operations additionally have a much stricter independent check:
**1 BF16 ulp or 2^-12 tensor max against FP64**. For NetHack GEMM+bias/ReLU,
the bound propagates one GEMM rounding interval through the second store;
it accounts for cancellation in the bias add. Exact copies, saved
observations, categorical materialization, integer indices, and fixed-point
scatters/flushes are checked exactly. Sampling actions/RNG and BF16
sampling/importance outputs are exact; FP32 conditional PPO outputs keep
the existing FP32 1e-5 norm gate. OSRS argmax is checked against FP64 scores
from each backend's actual z1 and the **unrounded** GELU tile, with the
existing 1e-5 FP32 score-gap bound. Inactive rows must choose -1; identical
records must preserve strict first-winner semantics.

The CUDA FP64 measurements are evidence, not a relaxed Metal gate. There
are 23 CUDA direct-operation measurements beyond the stricter one-ulp
budget, as with leg 9's cuBLAS BF16 reductions. The largest measured CUDA
norm deviation from FP64 is **0.007497** at NetHack's local GEMM+bias/ReLU.
Metal passes every direct FP64 bound, including these operations. Reports
preserve the CUDA measurements separately from Metal pass/fail results.

`run_test_env_bf16.sh` adds **64 poisoned primitive checks**, with two
changed-input trials: both entity geometries; OSRS pool/saved GELU/wgrad/
grad_z1 with H=40 and H=289; Craftax embedding/scalars/scatter/flush;
NMMO3 categorical im2col, copy, col2im, embedding, concat, scatter/flush;
NetHack bias/ReLU and pointer norms. Algebraic dyadic fixtures are exact;
transcendental outputs use one BF16 ulp against FP64. Every pure destination
is poisoned before use; accumulators are explicitly reset where required
by the API. Full BF16 replay allocations are poisoned before first use,
main FP32 forward outputs are poisoned, and conditional-head outputs are
re-poisoned at each epsilon endpoint. Replay output directories are removed
first and every requested env must produce both manifests.

Four mutations were rejected: missing entity materialization, missing
Craftax scalar copy, missing NMMO3 embedding copy, and incorrectly rounding
the OSRS GELU tile before pooling. Logs: `build/leg12/mutations/`.

## Training

Seed 1, same geometry/config overrides across the three backends, **64
learner epochs** each. Used `env_sweep.py`'s bounded geometry: H=64, two
layers, horizon 16, full-batch minibatch, two buffers/threads, one policy,
self-play and final evaluation disabled, replay ratio raised to 1 only
where a lower default would execute no learner. There are 262144 steps per
run except NMMO3's whole multi-agent env geometry requires 2097152 steps.
These are final logged **training** scores, not post-training evaluation.

| Env | Metal BF16 | Metal FP32 | CUDA BF16 |
|---|---:|---:|---:|
| minimal | -381.4280 | -382.1285 | -383.7610 |
| asteroids | 1.7962 | 1.8421 | 1.3295 |
| osrs_zulrah | 0.1592 | 0.1668 | 0.1839 |
| osrs_pvp | 0.004330 | 0.002232 | 0.002791 |
| osrs_colosseum | 0.001171 | 0.001296 | 0.001124 |
| osrs_inferno | 0.02697 | 0.02899 | 0.03158 |
| craftax | 5.0000 | 4.8727 | 4.8799 |
| nmmo3 | 1.0000 | 1.0000 | 1.0000 |
| nethack | 2.8276 | 3.6543 | 2.4070 |

All 27 runs completed 64 epochs, with finite losses and checkpoints.
They show comparable short-run behavior, not statistical equivalence or
convergence. NetHack's final windows differ appreciably across all three
backends, and NMMO3's score is flat at 1; neither is evidence of solved
learning. Longer multi-seed learning is still useful follow-up work.

Also ran two rollout epochs with **the actual default geometry/policy**
for the smaller four envs, changing only duration, output paths, seed and
final evaluation count:

| Env | Metal BF16 | Metal FP32 |
|---|---:|---:|
| minimal | -60.3013 | -60.5784 |
| asteroids | 0.82153 | 0.82102 |
| osrs_zulrah | 0.11951 | 0.11877 |
| osrs_pvp | 0.001449 | 0.001482 |

The larger envs use the sweep geometry for training smokes; their configured
hidden sizes are covered by the golden cases. In particular, NetHack's
512-agent, horizon-512, H=1024 default is not used for this fanless 24 GB
smoke. No out-of-memory default training was attempted.

Evidence: `build/leg12/training/{ENV}/{bf16,f32}.log`, per-run `*_out/logs/`
and checkpoints, `build/leg12/training/results.json`,
`build/leg12/default_smoke/`, `build/leg12/cuda_training_results.json`.
CUDA-reference logs/checkpoints are under
`pufferlib-envs-bf16/build/leg12/training/`; initial NetHack build failure
was resolved by the launch-header flag above and rerun successfully.

## FP32 and Breakout preservation

The existing FP32 env golden gate passes all 14 cases at its unchanged
1e-5 tolerance. More strongly, compiled `7538b089` shaders into an isolated
baseline metallib and replayed the same binaries/harness/inputs against it:
**1124 FP32 tensors across the seven large envs are byte-identical to the
ported shaders**. An additional **72 tensors** for Minimal/Asteroids are
byte-identical. Baseline extraction stayed under `build/leg12/`; no other
branch or worktree was checked out or edited.

Default-length Breakout seed 1, exact commands with only output binary
name and `--metal` / `--float` selection, followed by `X train --base.seed=1`:

| Precision | Final env/score | Checkpoint SHA256 |
|---|---|---|
| FP32 | 844.95672607421875 | 2ee4da82d356a44c8f22f24607ed320200ec3a6a28c4124d931fe67b3b05e841 |
| BF16 | 852.43658447265625 | f5dc197b8d7fde55… |

Both match the required baseline. They are repeated by the final gate
runner, which prints the complete hashes and fresh log paths in
`build/gates_leg12/breakout_seed1.log`.

## Final sweep and gates

`tools/metal/env_sweep.sh`, **without --out**, default BF16:
**81/85 OK**. Expected remaining entries: drive RUNTIME_FAIL (missing data),
matsci BUILD_FAIL (LAMMPS), robot_arm CUDA_ONLY, osrs SKIPPED (support dir).
Every recorded build command omits `--float`. Results and exact commands:
`build/metal/sweep/results.json`, `table.md`, per-env build/train logs.

The supplied runner was copied to `build/gates_leg12/run.sh`, with its cd
changed to this worktree, and the new primitive gate, BF16 env gate against
both CUDA runs, and exact Breakout seed-1 check added. **21/21 PASS**:
runtime, rollout, shared_env, wait_detect, obs_layout, kernels, mingru,
gemm, ppo_muon, replay, env_golden, bf16, rollout_bf16, bf16_golden_a,
bf16_golden_b, replay_bf16, mingru_bf16, env_bf16_math,
env_golden_bf16_a, env_golden_bf16_b, breakout_seed1.

Final verification runs at the report/source commit; the runner records
its exact hash in `build/gates_leg12/commit.txt`. Status and complete logs:
`build/gates_leg12/status.log`, `build/gates_leg12/*.log`. The full sweep
runs again after those gates at the same commit. No performance claims are
made from these correctness runs or their overlapping preliminary checks.

## Commits and remaining work

- `d7a1eb36`: BF16 custom shaders/adapters; remove fallback; default sweep.
- `fbc44a63`: CUDA BF16 replay, poisoned FP64 primitives/intermediates.
- `e537d03c`: require complete replays and exact BF16 sampling outputs.
- Report/source commit: this report, relay entry and configured Asteroids
  H=256 golden case. See `git log` and final gate `commit.txt` for its hash.

CUDA-reference commits: `177d287c`, `3f389444`, `bce6d7ca`, `7f027ee0` on
`golden-envs-bf16`; nothing pushed. No requested encoder remains FP32-only.
The pre-existing optional NMMO3 attention alternative is outside the
ported default convolution path. Longer multi-seed custom-env learning
and BF16 performance tuning remain open; correctness has no known blocker.
