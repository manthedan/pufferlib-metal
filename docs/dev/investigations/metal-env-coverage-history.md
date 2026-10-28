> Historical development log; last recorded at commit `e68239d9` (tree archived from `e68239d9`). Statements and gate counts describe that revision, not current support.

# Historical Metal environment coverage audit

Update 2026-10-26: `robot_arm --metal` now supports CPU physics with BF16 or
FP32 learners. See [the robot-arm port report](robot-arm-port-report.md) for
performance, CUDA trajectory gates and the limits of the training comparison.
The measurements and CUDA-only classification below describe the earlier audit.

Measured 2026-10-24 on branch `metal-envs` (first leg based on `94dfd4e3`; the custom-encoder leg starts at `feabd10b`), on an Apple M4 MacBook Air (24 GB), macOS 26.4.1, Xcode 26.6 (17F113). These historical measurements are FP32-only; the current Metal build defaults to BF16 and `--float` selects FP32. `/bin/bash` is 3.2.57. All 85 `ocean/` directories were attempted: 84 environments plus the shared `osrs/` support directory. No CUDA environment (`--cu`) was ported or run. Matsci (LAMMPS), drive (WOMD data) and robot_arm (CUDA-only) were deliberately left out of scope.

Final smoke counts: **81 OK, 1 BUILD_FAIL, 0 UNPORTED_KERNEL, 1 RUNTIME_FAIL, 1 CUDA_ONLY, 1 SKIPPED**. These describe the commands below, not full tuned-config convergence. Custom networks that still contain CUDA syntax fail compilation before reaching a launch, so they are BUILD_FAIL rather than UNPORTED_KERNEL.

## Reproduction and scope

```sh
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
/bin/bash tools/metal/env_sweep.sh
# Retry a subset (updates its rows); --resume skips all already recorded rows.
/bin/bash tools/metal/env_sweep.sh asteroids minimal moba impulse_wars
/bin/bash tests/metal/run_test_envs.sh
# Replay CUDA goldens for the custom encoders (expects ../golden/envs/run_a).
/bin/bash tests/metal/run_test_env_golden.sh
```

The sweep runs builds and training serially, with an independent `build/metal/bin_ENV` for each env. Commands, exit codes, durations, logs and the generated table live in `build/metal/sweep/{results.json,ENV.build.log,ENV.train.log,table.md}`. Build timeout: 300 seconds; train timeout: 45 seconds. A timeout kills and reaps the whole process group before moving on.

Each smoke run retains the environment configuration but uses hidden size 64, two layers, horizon 16, two updates, and a minibatch equal to the rollout batch. Agents start at 256, rounded up for whole multi-agent environments. Buffers are capped at two (single-buffer constraints retained), threads at two. MOBA derives five/ten agents from `script_opponents`. Self-play and final evaluation are disabled; configured replay ratios below one are clamped to one so every epoch executes a learner minibatch (other replay ratios are retained). The actual overrides are in `results.json`. A pass requires exit 0, epoch 2 in the dashboard, and no NaN/Inf token in output. It is a build/rollout/learner smoke check, not exhaustive numerical validation, self-play coverage, or an episode/convergence guarantee.

The script understands INI inline comments and bracketed numeric lists. It loads the compiled env's normal config; the `craftax_1m` config alias is not a separate ocean directory. Training budgets use the verified `--train.total_timesteps` spelling. The trainer truncates to whole `total_agents * horizon` batches; smoke budgets contain exactly two batches.

## Repairs and validation

- Ported the common point-embedding/max-pool encoder used by **minimal and asteroids**: materialization, linear + ReLU/max forward and argmax, output-weight gradient, and entity-input gradient. New shaders are in `src/metal/shaders/envs_entity.metal`, new adapters in `src/metal/kernels_envs.h`, with one include added at the end of `kernels.h`. CUDA launches keep their original behavior via `PUF_LAUNCH`; device bodies are guarded for Metal. The optional `MINIMAL_ATTN` PTX encoder still reports an explicit build error on Metal.
- **No CUDA goldens exist for these new kernels.** `tests/metal/test_envs.mm` compares the actual GPU adapters against independent double-precision CPU formulas for both layouts. It checks partial batches/hidden tiles (B=1/7/9, H=1/33/65), signed inputs/gradients, exact ReLU zeros, first-point ties, nullable argmax, padded observation strides, interior pointers and guard elements. All cases passed. Float tolerances are `2e-5 * (1 + abs(reference))`; observed maximum errors were below 2e-6. Both real envs also complete smoke training. The pre-existing rollout-kernel synthetic suite (`tests/metal/run_test_kernels.sh`) also passes after the new adapter include. An explicit `NVCC_EXTRA=-DMINIMAL_ATTN` build confirms the intended Metal rejection.
- Added the C API implementation to the **Impulse Wars** Metal build, fixing unresolved `puf_init/puf_step/...` symbols. Its smoke training passes with the downloaded macOS Box2D archive.
- Replaced MOBA BFS's Clang-incompatible C++ variable-length-array pointer with equivalent flat indexing. MOBA builds and trains after respecting its five-agent geometry.
- Replaced Linux-only `nproc` with `getconf _NPROCESSORS_ONLN` for the NetHack dependency build. On Darwin, `build.sh` also passes `-Wno-error=implicit-function-declaration` to the fast-nle CMake build (checkout `e7f6568568242745a7f7ebd8f730367123522cac`). `nle_lnc_bits` returns `int`, so the implicit declaration is ABI-correct, and `libnethack.so` now builds. The NetHack policy was subsequently ported in the NetHack leg below.
- Fixed sweep assumptions exposed by Admiral's required single buffer and Shared Pool's bracketed `num_agents` configuration. Admiral's CPU env trains on Metal; its optional CUDA simulator is a separate unsupported backend.

### Custom-encoder leg (`feabd10b..`)

- **OSRS (zulrah, pvp, colosseum, inferno)**: `ocean/osrs/osrs_entity_encoder.cu` guards its device code under `PUF_METAL` and launches through `PUF_LAUNCH`. `src/metal/shaders/envs_osrs.metal` implements the branch gather, the fused pool forward, the L2 weight gradient and the fused grad-z1 kernel. Records per entity are capped at 32 (the adapter aborts above that). An invalid item code writes NaN instead of triggering CUDA's device assert. The item table is uploaded once.
- **Craftax**: `ocean/craftax/craftax.cu` runs its category-embedding encoder through `src/metal/shaders/envs_craftax.metal`. The CUDA int64 fixed-point atomics for the embedding gradient are emulated with paired 32-bit atomics and an explicit carry (`src/metal/shaders/envs_fxp.h`, shared with NMMO3), so accumulation stays order-independent and deterministic. The sweep now forces `--train.gpus=1` for configs that ask for more GPUs (craftax asks for 8, which hit an NCCL assertion).
- **NMMO3**: `ocean/nmmo3/nmmo3.h` replaces three Clang-incompatible VLA pointer casts with flat indexing. This is a mechanical rewrite and the simulation logic is unchanged. `src/metal/shaders/envs_nmmo3.metal` has 12 kernels for the conv encoder (the embedding backward uses exactly the 32 KB threadgroup limit). The optional `N3_ATTN` encoder gives an explicit `#error` on Metal.
- Adapters for all three are in `src/metal/kernels_envs.h` behind each env's define. Kernel ids were added to `src/metal/launch.h`. No runtime, GEMM or core shader files were changed.
- **CUDA goldens.** A shared harness, `tests/metal/env_golden.h`, runs each Encoder vtable through a train forward, a rollout forward and a backward pass, then dumps the params, inputs, outputs, saved activations and gradients. On CUDA-reference (RTX 3090, branch `golden-envs` at `9d49661c` in `<cuda-reference>/pufferlib`) it generated 12 cases: a small case (B=13, H=40) and a default case (default hidden size; B=520, or B=1500 for NMMO3) for each of the 6 envs. Two CUDA runs were bitwise identical. The Metal-branch sources, built with nvcc on CUDA-reference, reproduce all 12 goldens bitwise, so CUDA behavior is unchanged. `tests/metal/run_test_env_golden.sh` replays the goldens on Metal, restoring the golden forward activations before the backward pass, and `tests/metal/compare_env_golden.py` checks max-normalised error against a tolerance of 1e-5. All 12 cases pass in about 30 s. The worst errors are 1.96e-6 (nmmo3_default), 1.71e-6 (craftax_default) and 1.45e-6 (colosseum_default). Many tensors are bitwise equal. inferno_default has 3 forward argmax flips out of about 1M entries. The comparator recomputes each flipped pooling score in float64 from the golden h1 and l2 weights, and accepts a flip only if both records are valid and active and within 1e-5 of the winner. These 3 are genuine ties (relative gap about 3.4e-8). A flip to `-2` or to a valid but non-tied record fails. The Mac copy of `env_golden.h` also writes a `pool_argmax_groups` note into its replay manifests for this check; otherwise the harness matches the CUDA-reference copy used to generate the goldens. The goldens (560 MB) live outside the repo in `../golden/envs/run_a`.
- The first-leg CPU-reference suite (`tests/metal/run_test_envs.sh`) still passes.

### NetHack leg (`97390ef6..`)

- Ported **50 kernels** from the active default policy and its weight-loading/linear-pool paths into `src/metal/shaders/envs_nethack.metal`, with host adapters in `src/metal/kernels_nethack.h` (included by `kernels_envs.h`). This covers factorized glyph embeddings, semantic crop and terrain features, inventory/discovered-item/state features, message trigram bags, monster/item token streams, identity embeddings, spell features, sum/max pools and their deterministic backward scatters, bias gradients, plus the inventory/spell pointer decoder and verb/direction/value assembly. The dormant legacy patch-encoder kernels are compiled out on Metal. Shader dimensions and every argument-struct ABI are asserted on the host.
- `envs_nethack_policy.metal` implements NetHack-specific sampling, importance caching and PPO gradients: only heads consumed by the selected verb contribute log probability, entropy or gradients, and the verb head uses the same annealed epsilon mixture and chain-rule scale as CUDA. RNG draws for unused heads are preserved. The upstream launches use `PUF_LAUNCH`; CUDA-only bodies and symbol copies have Metal guards. No runtime, GEMM, or core shader files changed.
- Extended the shared CUDA/Metal harness with small (B=13, H=40) and default-width (B=520, H=1024) cases, **328 tensors each**. Observations cover map corners, signed LE stats, empty/padded entities, unknown item states, spells and accessory slots. Both Encoder and Decoder vtables run training/rollout forward and backward. Independent head checks cover all 26 verbs, partial and all-zero masks, epsilon 0.005 and 0, exact sampled actions/RNG state, importance ratios and PPO gradients. Replay checks actual decoder key gradients before restoring CUDA gradients for the encoder backward boundary.
- On CUDA-reference, the original CUDA source matches bitwise across two runs, and the guarded source matches it bitwise for all **656 tensors**. Reference commits on `golden-envs`: `8c5d6436` (harness), `1d848678` (guarded sources), `75edf62a` (independent key-gradient recording). The source files before the port had identical SHA-256 hashes across machines. Missing flex, bison and m4 were installed from downloaded Debian packages under `<cuda-reference>/tools/parser-root`, with wrappers under `<cuda-reference>/bin`; no sudo or system writes were used.
- Metal replay passes the unchanged **1e-5** maximum-normalized tolerance. Worst errors: **3.4607e-6 default**, **1.8594e-6 small**; conditional-head tensors peak at **4.3164e-7**. No forward argmax flips. Goldens are in `../golden/envs/run_a/nethack_{small,default}`; determinism/source-equivalence metadata is in `../golden/envs/nethack-{determinism,source-equivalence}.json`. All 14 environment golden cases pass, the CPU-reference suite passes, and train replay passes with exact RNG state.
- `tools/metal/env_sweep.sh nethack` passes two updates (8,192 steps, H=64, two layers); final printed entropy 3.252. Training logs and exact commands are retained in `build/metal/sweep`.

## Per-directory results

The table records the first observed build error after repairs, or the runtime diagnostic. Local absolute paths in diagnostics are shortened to worktree-relative paths. Each OK row completed two learner updates with finite printed losses.

| Environment | Class | Observation |
|---|---|---|
| admiral | OK | Two learner updates completed; finite losses (small policy/batch; evaluation disabled). |
| affine_lock | OK | Two learner updates completed; finite losses (small policy/batch; evaluation disabled). |
| ants | OK | Two learner updates completed; finite losses (small policy/batch; evaluation disabled). |
| asteroids | OK | Two learner updates completed; finite losses (small policy/batch; evaluation disabled). |
| backgammon | OK | Two learner updates completed; finite losses (small policy/batch; evaluation disabled). |
| bat | OK | Two learner updates completed; finite losses (small policy/batch; evaluation disabled). |
| battle | OK | Two learner updates completed; finite losses (small policy/batch; evaluation disabled). |
| benchmark | OK | Two learner updates completed; finite losses (small policy/batch; evaluation disabled). |
| blastar | OK | Two learner updates completed; finite losses (small policy/batch; evaluation disabled). |
| boids | OK | Two learner updates completed; finite losses (small policy/batch; evaluation disabled). |
| boss_fight | OK | Two learner updates completed; finite losses (small policy/batch; evaluation disabled). |
| boxoban | OK | Two learner updates completed; finite losses (small policy/batch; evaluation disabled). |
| breakout | OK | Two learner updates completed; finite losses (small policy/batch; evaluation disabled). |
| cartpole | OK | Two learner updates completed; finite losses (small policy/batch; evaluation disabled). |
| chain_mdp | OK | Two learner updates completed; finite losses (small policy/batch; evaluation disabled). |
| chain_reaction | OK | Two learner updates completed; finite losses (small policy/batch; evaluation disabled). |
| checkers | OK | Two learner updates completed; finite losses (small policy/batch; evaluation disabled). |
| chess | OK | Two learner updates completed; finite losses (small policy/batch; evaluation disabled). |
| click | OK | Two learner updates completed; finite losses (small policy/batch; evaluation disabled). |
| clifford | OK | Two learner updates completed; finite losses (small policy/batch; evaluation disabled). |
| connect4 | OK | Two learner updates completed; finite losses (small policy/batch; evaluation disabled). |
| convert | OK | Two learner updates completed; finite losses (small policy/batch; evaluation disabled). |
| convert_circle | OK | Two learner updates completed; finite losses (small policy/batch; evaluation disabled). |
| craftax | OK | Two learner updates completed; finite losses (small policy/batch; evaluation disabled). |
| craftax_classic | OK | Two learner updates completed; finite losses (small policy/batch; evaluation disabled). |
| dino | OK | Two learner updates completed; finite losses (small policy/batch; evaluation disabled). |
| docking | OK | Two learner updates completed; finite losses (small policy/batch; evaluation disabled). |
| double_pendulum | OK | Two learner updates completed; finite losses (small policy/batch; evaluation disabled). |
| drive | RUNTIME_FAIL | ERROR: Cannot find map files at drive_data/binaries/ |
| drmario | OK | Two learner updates completed; finite losses (small policy/batch; evaluation disabled). |
| drone | OK | Two learner updates completed; finite losses (small policy/batch; evaluation disabled). |
| enduro | OK | Two learner updates completed; finite losses (small policy/batch; evaluation disabled). |
| flappy | OK | Two learner updates completed; finite losses (small policy/batch; evaluation disabled). |
| freeway | OK | Two learner updates completed; finite losses (small policy/batch; evaluation disabled). |
| g2048 | OK | Two learner updates completed; finite losses (small policy/batch; evaluation disabled). |
| go | OK | Two learner updates completed; finite losses (small policy/batch; evaluation disabled). |
| guerrillacheckers | OK | Two learner updates completed; finite losses (small policy/batch; evaluation disabled). |
| hex | OK | Two learner updates completed; finite losses (small policy/batch; evaluation disabled). |
| impulse_wars | OK | Two learner updates completed; finite losses (small policy/batch; evaluation disabled). |
| laser_puzzle | OK | Two learner updates completed; finite losses (small policy/batch; evaluation disabled). |
| lightsout | OK | Two learner updates completed; finite losses (small policy/batch; evaluation disabled). |
| matsci | BUILD_FAIL | ./ocean/matsci/matsci.h:5:10: fatal error: 'lammps/library.h' file not found |
| maze | OK | Two learner updates completed; finite losses (small policy/batch; evaluation disabled). |
| mazing_contest | OK | Two learner updates completed; finite losses (small policy/batch; evaluation disabled). |
| memory | OK | Two learner updates completed; finite losses (small policy/batch; evaluation disabled). |
| minimal | OK | Two learner updates completed; finite losses (small policy/batch; evaluation disabled). |
| moba | OK | Two learner updates completed; finite losses (small policy/batch; evaluation disabled). |
| nethack | OK | Two learner updates completed with finite losses; full custom policy and conditional heads ported. |
| nmmo3 | OK | Two learner updates completed; finite losses (small policy/batch; evaluation disabled). |
| onestateworld | OK | Two learner updates completed; finite losses (small policy/batch; evaluation disabled). |
| onlyfish | OK | Two learner updates completed; finite losses (small policy/batch; evaluation disabled). |
| osrs | SKIPPED | Shared OSRS support directory, no standalone environment/header/config; build attempted. |
| osrs_colosseum | OK | Two learner updates completed; finite losses (small policy/batch; evaluation disabled). |
| osrs_inferno | OK | Two learner updates completed; finite losses (small policy/batch; evaluation disabled). |
| osrs_pvp | OK | Two learner updates completed; finite losses (small policy/batch; evaluation disabled). |
| osrs_zulrah | OK | Two learner updates completed; finite losses (small policy/batch; evaluation disabled). |
| overcooked | OK | Two learner updates completed; finite losses (small policy/batch; evaluation disabled). |
| pacman | OK | Two learner updates completed; finite losses (small policy/batch; evaluation disabled). |
| pathfinder | OK | Two learner updates completed; finite losses (small policy/batch; evaluation disabled). |
| pong | OK | Two learner updates completed; finite losses (small policy/batch; evaluation disabled). |
| robocode | OK | Two learner updates completed; finite losses (small policy/batch; evaluation disabled). |
| robot_arm | CUDA_ONLY | Expected: robot_arm implies --cu; Metal rejects CUDA envs. |
| rware | OK | Two learner updates completed; finite losses (small policy/batch; evaluation disabled). |
| scape | OK | Two learner updates completed; finite losses (small policy/batch; evaluation disabled). |
| shared_pool | OK | Two learner updates completed; finite losses (small policy/batch; evaluation disabled). |
| slimevolley | OK | Two learner updates completed; finite losses (small policy/batch; evaluation disabled). |
| snake | OK | Two learner updates completed; finite losses (small policy/batch; evaluation disabled). |
| soccer | OK | Two learner updates completed; finite losses (small policy/batch; evaluation disabled). |
| space_invaders | OK | Two learner updates completed; finite losses (small policy/batch; evaluation disabled). |
| squared | OK | Two learner updates completed; finite losses (small policy/batch; evaluation disabled). |
| squared_continuous | OK | Two learner updates completed; finite losses (small policy/batch; evaluation disabled). |
| tactical | OK | Two learner updates completed; finite losses (small policy/batch; evaluation disabled). |
| target | OK | Two learner updates completed; finite losses (small policy/batch; evaluation disabled). |
| template | OK | Two learner updates completed; finite losses (small policy/batch; evaluation disabled). |
| terraform | OK | Two learner updates completed; finite losses (small policy/batch; evaluation disabled). |
| tetris | OK | Two learner updates completed; finite losses (small policy/batch; evaluation disabled). |
| tmaze | OK | Two learner updates completed; finite losses (small policy/batch; evaluation disabled). |
| tower_climb | OK | Two learner updates completed; finite losses (small policy/batch; evaluation disabled). |
| tower_defence | OK | Two learner updates completed; finite losses (small policy/batch; evaluation disabled). |
| trash_pickup | OK | Two learner updates completed; finite losses (small policy/batch; evaluation disabled). |
| tripletriad | OK | Two learner updates completed; finite losses (small policy/batch; evaluation disabled). |
| tron | OK | Two learner updates completed; finite losses (small policy/batch; evaluation disabled). |
| wef | OK | Two learner updates completed; finite losses (small policy/batch; evaluation disabled). |
| whackamole | OK | Two learner updates completed; finite losses (small policy/batch; evaluation disabled). |
| whisker_racer | OK | Two learner updates completed; finite losses (small policy/batch; evaluation disabled). |

## Default-length training and CUDA comparison

One default-length run per environment/backend, seed 73, with normal tuned configs and default `base.eval_episodes=10000`. Only output locations and `base.run_id` changed. CUDA was built with `--float` on CUDA-reference's RTX 3090 (driver 580.178.04), reference commit `4fa5e83377c15b2f084383a00be86a36ab3fc56a`. The four config files (`default`, `cartpole`, `pong`, `flappy`) match byte-for-byte between machines; SHA-256 hashes and exact summaries are saved in `docs/dev/investigations/metal-env-training-results.json`. Metal code was at `87038a42` (the later coverage commit changes documentation only).

| Env | Requested / executed steps (epochs) | Metal train / eval score | CUDA FP32 train / eval score | Metal mean / final SPS | CUDA mean / final SPS |
|---|---|---|---|---|---|
| cartpole | 15,000,000 / 14,942,208 (114) | 199.872 / 199.846 | 70.834 / 199.449 | 4.567M / 4.8M | 24.261M / 27.1M |
| pong | 8,000,000 / 7,995,392 (122) | 20.980 / 20.994 | 12.211 / 20.816 | 0.473M / 0.570M | 2.837M / 3.5M |
| flappy | 20,000,000 / 19,922,944 (152) | 46.707 / 51.043 | 45.514 / 50.645 | 2.016M / 2.7M | 14.546M / 15.8M |

Scores are `env/score`, not normalized `env/perf`. Train scores are the last printed completed-episode snapshot before evaluation (dashboard precision: three decimals); when the final train frame contains no episodes, the preceding completed-episode snapshot is used. CUDA's short runs log infrequently, so their train windows include much more early learning than Metal's; these train snapshots are not directly comparable learning curves. Evaluation scores come from the final saved INI metrics, with the dashboard confirming them. Default batched evaluation exceeds 10,000 episodes: Metal 12,288 / 10,148 / 10,466 and CUDA 12,288 / 10,037 / 10,539 (cartpole / pong / flappy). Cartpole evaluation `env/perf` is 0.994260 Metal versus 0.992271 CUDA.

Mean SPS is executed training steps divided by the trainer's final reported training uptime, excluding final evaluation. Final SPS is the final train dashboard interval, rounded by the dashboard. Metal training uptimes were 3.271 / 16.893 / 9.882 seconds; CUDA 0.616 / 2.818 / 1.370 seconds. Local wall times including initialization, final checkpoint and evaluation were 3.560 / 21.219 / 17.428 seconds. These are single runs on different machines; the Mac was shared with a separate performance investigation, so timings are observations, not isolated benchmarks. CUDA graphs retain their normal setting on CUDA and are forced off on Metal. Close evaluation scores demonstrate that these standard networks learn on both backends; they do not establish kernel-level parity or validate the newly ported encoders against CUDA.

Reproduce locally (each command is serial; retain default training and evaluation settings):

```sh
mkdir -p build/metal/full
for env in cartpole pong flappy; do
    ./build.sh "$env" "build/metal/bin_$env" --metal
    "build/metal/bin_$env" train --base.run_id=metal_full \
        --base.log_dir=build/metal/full/logs \
        --base.checkpoint_dir=build/metal/full/checkpoints \
        > "build/metal/full/$env.train.log" 2>&1
done
```

On CUDA-reference, source `<cuda-reference>/env.sh`, work inside `<cuda-reference>/pufferlib`, and use `./build.sh "$env" "build/metal_envs/bin_$env" --float`, then `build/metal_envs/bin_$env train --base.run_id=cuda_full --base.log_dir=build/metal_envs/logs --base.checkpoint_dir=build/metal_envs/checkpoints`. All remote artifacts stay below `<cuda-reference>`; pre-existing `puffer` and `puffer_f32` files were not changed. Local full logs/checkpoints/INIs are under `build/metal/full`; copied reference logs/INIs are under `build/metal/cuda-ref/build/metal_envs`.

### OSRS Zulrah (custom-encoder leg)

Default `osrs_zulrah` config, seed 73. Configs match byte-for-byte (SHA-256 values are in `docs/dev/investigations/metal-env-training-results-leg2.json`). Metal was at `08dbe4da`; CUDA used `--float` on CUDA-reference at `9d49661c`, with CUDA graphs at their default setting. Each run used 33,816,576 steps (129 epochs), and each evaluation covered 12,288 or 12,289 episodes.

| Backend | Seed | Eval score | Eval wins | Train uptime | Mean SPS |
|---|---|---|---|---|---|
| Metal | 73 | 3.131 | 2.221 | 398.5 s | 84.9K |
| CUDA FP32 | 73 | 3.297 | 2.321 | 56.4 s | 599K |
| CUDA FP32 | 74 | 3.197 | 2.253 | 56.3 s | 600K |
| CUDA FP32 | 75 | 3.179 | 2.243 | 52.6 s | 642K |

Metal learns the task, but its seed-73 eval score is about 1.5 CUDA standard deviations below the CUDA mean (3.225, sd 0.064 over 3 seeds). That comes from a single Metal seed on a GPU shared with a concurrent benchmark, so it is not a demonstrated regression. Metal training is about 7x slower than the 3090. Metal logs are in `build/metal/full/osrs_zulrah.*` and `build/metal/full/logs/osrs_zulrah`, and the copied CUDA INIs are in `build/metal/cuda-ref/osrs_zulrah`. A default-length craftax run was not attempted, because `craftax.ini` (1.2e11 steps) and `craftax_1m` (2e9) are far beyond a 15-minute budget.

### NetHack matched fixed-budget training

The unmodified default training geometry (512 agents, four buffers, horizon 512, minibatch 4,096) fails `alloc_create`/`cudaMalloc` on this M4 before the first epoch (log: `build/metal/nethack_training/metal_default_alloc.log`). A smaller asynchronous attempt with 128 agents/horizon 128/minibatch 1,024 reached five epochs (81,920 steps) and then stopped progressing; after another 161 seconds without an epoch it was killed. The process sample shows the learner in `puf_mtl::begin_locked` and both rollout threads in `stream_sync`, all waiting on IOSurface shared events (`sample.txt`). This is evidence of a stream-wait stall, not a proven root cause. A serialized large-batch attempt was stopped after 164 s before its first completed epoch to keep GPU use bounded. These attempts are not credited as completed training runs; no runtime code was changed.

The completed comparison retains the full **H=1024, four-layer default policy**, seed **73**, and all learning hyperparameters. Identical overrides on Metal and CUDA FP32: **64 agents, one buffer, two threads, horizon 16, minibatch 1,024, `base.async=0`, 16,384 total steps**, and 128 requested evaluation episodes. Each epoch executes one learner minibatch (`floor(1.78021395 * 1024 / 1024)`). Both completed **16 epochs**, with finite printed losses. This is a short functional training check, not convergence. Normal tuned config files match byte-for-byte across machines. Exact commands, metrics hashes, golden results and unsuccessful attempts are in `docs/dev/investigations/metal-env-nethack-results.json`.

| Backend | Steps / epochs | Final train score | Eval score (actual episodes) | Train uptime | Mean / final SPS |
|---|---|---|---|---|---|
| Metal M4 | 16,384 / 16 | 2.000 | 2.404255 (141) | 102.799 s | 159.4 / 120.9 |
| CUDA FP32 RTX 3090 | 16,384 / 16 | 1.840 | 2.470588 (136) | 1.427 s | 11,481.5 / 11,138.8 |

Mean SPS excludes evaluation; end-to-end wall times were 107.970 s and 2.423 s. Train scores use the first epoch-16 dashboard snapshot; eval scores use persisted final metrics. Batched evaluation overshoots the requested episode count. These are single-seed short runs on different hardware, with the M4 shared with the runtime work; they do not establish a learning-quality regression. The billion-step default budget cannot fit the requested approximately 20-minute window. Logs and checkpoints remain in `build/metal/nethack_training`; CUDA artifacts were copied from `<cuda-reference>/nethack_training` on CUDA-reference.

## Remaining blockers

- NetHack's exact default training geometry exceeds the Metal allocation budget on the measured M4. The full default policy is validated numerically and trains using the smaller serialized batch geometry documented above; the billion-step default budget was not run. The earlier longer async attempt appeared to stall in Metal stream-event waits while the GPU was severely power-capped; a later eight-epoch run of the same 128-agent/horizon-128 async geometry completed normally. See `docs/dev/investigations/metal-runtime-debug.md`. Compile-time variants changing message width/vocabulary, token count, or accessory-panel dimensions are rejected by a static assertion.
- The optional `N3_ATTN` (NMMO3) and `MINIMAL_ATTN` encoders use inline PTX and are rejected explicitly on Metal.
- Craftax was validated by goldens and smoke training only. Its default training budget is too long for a full-length run here.
- `matsci` requires external LAMMPS headers/library; these are not present in this checkout.
- Drive's native training path requires WOMD maps in `drive_data/binaries`; the shipped two demo maps in `resources/drive` do not satisfy its training dataset scan. No substitute dataset was silently used.
- `robot_arm` and optional `--cu` simulators remain CUDA-only. `osrs/` itself is shared support code.
- Optional attention encoders, alternate observation/action modes, self-play configurations and long training of every smoke-pass environment are outside this sweep's tested configurations.

## Review


Static review found no additional concrete regressions in the encoder ports, CPU-reference tests or build fixes. The follow-up review confirmed the clamp and rerun metrics, checked coverage counts, config hashes, training step budgets, scores and SPS against retained artifacts, and found no remaining concrete regressions. Review logs are retained in `build/metal/review.log` and `build/metal/review-followup.log`. Both reviews were read-only and launched no GPU jobs.

### Custom-encoder leg review


- P1: the lazy OSRS item-table upload in `kernels_envs.h` was not synchronized. With the default two rollout buffers, a second thread could read the table pointer before the upload finished and gather from a zero-filled table on its first forward, and two threads could also race on the allocation. Allocation and upload now happen in a C++ thread-safe static initializer, which aborts if either fails. Only a first rollout could have been affected, so the zulrah run above (from before the fix) is not materially changed. All four OSRS smoke rows pass again after the fix.
- P2: the comparator accepted up to 1e-4 of forward argmax entries differing, with no further check, so an invalid index passed. Flips are now validated as genuine ties (see above). All 12 golden cases still pass, and two negative tests (`-2`, and a valid but non-tied record) now fail.

### NetHack leg review


After review, reran all required gates serially with the project venv on PATH: **14 golden cases / 1,124 tensors pass** at 1e-5 (NetHack worst remains 3.4607e-6), **entity CPU-reference tests pass (0 failures)**, and **train replay passes, including exact RNG offsets and state**. Logs are `build/metal/nethack-post-review-{golden,envs,replay}.log`. The documented allocation and longer-async stream-wait limits remain; the completed serialized fixed-budget comparison and smoke pass are separate evidence. CUDA-reference `golden-envs` ends at `920a88e9` (the last change only aligns preprocessor whitespace); the shared harness matches the Mac copy.
