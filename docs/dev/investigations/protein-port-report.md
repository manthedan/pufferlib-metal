> Historical port report: the embedded cuRAND table described below has since
> been removed. Default Metal sweeps now use public Joe--Kuo directions and
> fixed SplitMix64 shifts; exact parity loads a private external fixture.
> See [the replacement report](../protein-sobol-report.md).

> Historical development log; last recorded at commit `e68239d9` (tree archived from `e68239d9`). Statements and gate counts describe that revision, not current support.

# Protein on Metal

Later environment update: robot_arm's CUDA-only limitation in the audit below
is superseded by the [CPU physics port for Metal](robot-arm-port-report.md),
including BF16/FP32 builds, measurements and CUDA trajectory validation.

2026-10-26 · MacBook Air M4, macOS 26.4 · worktree `wt-protein`, branch
`metal-protein`, starting at `ef62360b`. No other worktree was modified and
nothing was pushed or merged. The untracked classic_snake gate snapshots
are excluded from commits.

## Implementation

`puffer sweep` now includes the real `src/protein.cu` on Metal. The CUDA and
Metal builds share the search-space transforms, GP hyperparameter Adam/AMSGrad
updates (including the learned mean and noise prior), observation handling,
KD duplicate filtering, Pareto/pruning/search centers, cost quantile and pinball
fit, success classifier, acquisition scoring, caps, and optimizer resets.
Constants and the CUDA computation path are preserved.

`src/metal/protein_cpu.h` implements the compute backend. FP32 Accelerate
`spotrf`/`spotrs`, `cblas_strsv`, `cblas_sdot`, `cblas_sasum`, and `cblas_sgemv`
replace cuSOLVER/cuBLAS. Kernel gradients retain CUDA's 16×16 tile ordering,
256-lane binary reductions, and fixed host fold. Philox uses the existing
Metal sampler's round constants, state layout, subsequence placement, counter
advance and fused uniform conversion, expressed as scalar CPU arithmetic.
Ordinary heap buffers keep Protein independent of Metal command submission.

Prediction uses batches of at most 4096 candidates. This changes scratch
storage, not the candidate set or prediction formula. At 750 observations,
two prediction matrices occupy about 23.4 MiB instead of 375 MiB at 65536
candidates. Scratch is preallocated at creation; suggestions do not allocate
new large buffers. Accelerate is linked into the native trainer and the
standalone backward-test harness. Training kernels and training logic are
unchanged.

The complete cuRAND scrambled Joe–Kuo6 Sobol32 table is embedded as read-only
Mach-O data: 20000 rows of 32 direction words plus one scramble constant.
There is no runtime CUDA dependency or data-file lookup. The binary is
2,640,000 bytes (2.52 MiB), little-endian uint32, with SHA256
`fce775e7962beda83fa73c64c628a0323a94eadf5d32bad97d1dc6159e52e5f0`.
`tests/metal/export_protein_rng.cu` regenerates it through cuRAND's public APIs.
The existing builds already require invocation from the repository root;
the assembler `.incbin` follows that convention.

Darwin `rand()` differs from glibc. Protein has a private glibc-compatible
stream, initially seeded with 1 as in the native CUDA CLI. It supplies the
same cost noise, ratio-pool shuffles and observation subsampling without
changing the trainer's or environments' libc RNG. The test harness also uses
that stream for its seeded synthetic observation noise. Other application
calls to Darwin `srand`/`rand` do not alter Protein's private stream.

## Numerics and CUDA reference

**The original checked-in golden was stale.** Running the pristine source
blob `a9e8ca47fc2d0b67afc0ce22db083feb8d10949b` from the starting commit on
CUDA-reference's RTX 3090 / CUDA 12.8 failed **157 of 412 records** against that file.
For example, the first GP score loss was `0.329143047` on current CUDA versus
`0.331116319` in the old fixture. This is not a Metal tolerance issue.

The fixture was refreshed from that pristine CUDA source in an isolated copy
under `<cuda-reference>/protein-port`; the shared upstream checkout remained
read-only. GPU availability was checked and reference commands used
`nice -n 19`. The new fixture passes **412/412 bitwise on pristine CUDA**.
The comparison rule remains upstream's default: `atol=1e-5`, `rtol=1e-4`.
Metal passes **318 bitwise + 94 within tolerance, zero failures**. Maximum
absolute difference is `1.90735e-5` on a predicted cost. No tolerance was widened
and no Metal-generated values were used to update the golden.

**Suggestion identity:** all **116/116 normalized suggestion components** in
the 29-suggestion regression trajectory are bitwise identical to current
CUDA, including fixed-cost suggestions. All **1464 RNG fixture records** are
also bitwise identical: Sobol dimensions 1/4/21/20000, multiple offsets and
consecutive points; Philox seeds including a full-width 64-bit seed,
subsequences 0/1/17/65535 and 40 consecutive draws; glibc seeds including
2024 and UINT32_MAX. The 21-dimensional synthetic sweep produces the same
reported results on pristine CUDA and Metal.

This does not promise universal bitwise identity for arbitrary histories.
Accelerate versus cuBLAS/cuSOLVER reduction/factorization order, CPU versus
CUDA transcendental functions, and host floating-point contraction can differ
slightly; near an acquisition tie they can select different candidates.
Equal-key C-library sort ordering can also differ across operating systems.
Identical RNG state, seed, observation history and host-random consumption
are necessary. Real sweeps also observe measured wall-clock costs, so their
trajectories need not match across machines.

## Tests and commands

All commands below run from `wt-protein`:

```sh
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
export PATH=<repo>/.venv/bin:$PATH

tests/metal/run_test_protein.sh
python3 tests/test_sweep.py --metal
python3 tests/test_sweep_hyper.py --metal
python3 tests/metal/mutate_protein.py
tests/metal/run_test_sweep_cli.sh
bash tests/metal/run_all.sh

clang++ -std=c++17 -O2 -Isrc tests/metal/bench_protein.cpp \
    -framework Accelerate -o build/metal/protein/bench
build/metal/protein/bench
```

`test_protein.cu` and `test_protein_sweep.cu` referenced removed
`gp_cuda.cu`/ProteinObs/ProteinAcq APIs. They now exercise the production API:
Pareto order/pruning, cheaper centers, candidate bounds, fixed cost, minimize
acquisition, cost-cap masking, duplicate replacement, failures, boundary
rejection, capacity limits, logit, cost fitting/early-stop decisions, classifier,
GP observation subsampling, Adam reset, KD filtering, and fallback behavior.
The 21D synthetic task retains five observations per trial and checks guided
improvement. Random best `1.72937`; GP best `3.68107` over 40 trials. A separate
held-out prediction check covers the intent of `test_sweep_hyper.py`: normalized
score RMSE `0.000407`, log-cost RMSE `0.000575` (limits 0.02).

`test_sweep.py --metal` and `test_sweep_hyper.py --metal` now dispatch to the
native synthetic/holdout test before importing Python dependencies. Their
unflagged legacy modes still require the removed `pufferlib.sweep`/GPyTorch
stack. The Metal path validates production optimization and held-out accuracy;
it does not recreate the legacy plotting or GP hyperparameter-grid experiment.
The numerics harness uses the original record set and comparison logic.

Nine isolated-source mutations were all caught: changed kernel offset,
reversed kernel gradient, wrong prediction batch offset, shifted Sobol offset, shifted Philox subsequence,
inverted cost cap, disabled Pareto pruning, disabled early stop, and disabled
near-duplicate filtering. The script restores by deleting each isolated copy,
then reruns the clean baseline. Production source is never mutated. An
AddressSanitizer + UndefinedBehaviorSanitizer run of the behavior tests also
passes. Logs: `build/metal/protein/{tests,mutations,sanitize}.log`.

`tests/metal/run_all.sh` adds `protein` (three production tests plus RNG fixtures)
and `sweep_cli` to the existing 31 gates, and now returns failure if any gate
fails. The original harness depends on local golden assets and the existing
`build/gates_leg13`, `build/gates_leg17`, and `build/leg13` helper scripts.

**Final clean run: 33/33 gates PASS, process exit status 0** — all 31 original
gates plus `protein` and `sweep_cli`. The run used a frozen copy of the
runner (`build/gates_verify.sh`, identical gate commands); summary:
`build/gates_summary.log`. The final sweep gate completed in 1.30 seconds.

Independent extra Breakout check:

```sh
tests/metal/helpers/seedrun.sh build/gates_leg17/e2e/breakout_bf16 breakout 1
# score=852.43658447265625 sha=f5dc197b8d7fde55
```

All seeded score/checkpoint references exercised by the clean run remain
exact, including the default and opt-out modes (and forced overlap modes
for cartpole/breakout):

| Environment | Precision | Score | SHA256 prefix |
|---|---|---:|---|
| Pong | BF16 | 20.996614456176758 | b1425713d9e71433 |
| Pong | FP32 | 20.99445915222168 | 5ee574ef2f33c6dd |
| Flappy | BF16 | 50.560173034667969 | 91886134a50625e1 |
| Flappy | FP32 | 51.408348083496094 | 4eafd59e547e43e1 |
| Cartpole | BF16 | 199.96110534667969 | b10ccdb69f45c996 |
| Cartpole | FP32 | 200 | bf0330663b50b01e |
| Breakout | BF16 | 852.43658447265625 | f5dc197b8d7fde55 |
| Breakout | FP32 | 844.95672607421875 | 2ee4da82d356a44c |

The first pass found the test-only backward harness's missing Accelerate
link flag, now fixed and separately passed. Editing the runner during that
pass also disrupted one entry. The complete immutable-copy rerun passed every entry with no failures.

For independent CUDA reproduction, copy the pristine source and the test
harness into a scratch directory, then:

```sh
. <cuda-reference>/env.sh
nice -n 19 nvcc -O2 -Isrc tests/test_protein_numerics.cu \
    -lcublas -lcusolver -lcurand -o test_protein_numerics
nice -n 19 ./test_protein_numerics tests/golden_protein_numerics.txt
# To regenerate the RNG data, in that scratch directory:
nice -n 19 nvcc -O2 tests/metal/export_protein_rng.cu -lcurand -o export_protein_rng
nice -n 19 ./export_protein_rng protein_sobol32.bin golden_protein_rng.txt
```

## Real sweep

`tests/metal/run_test_sweep_cli.sh` builds cartpole BF16 and starts 12 real
child-process training trials: one configured default, ten random suggestions,
then one GP-guided suggestion. `sweep.gpus=-1` correctly selects one Apple GPU.
The actual command is saved in `build/metal/protein/demo/command.txt` and full
output in `build/metal/protein/cli.log`. Timesteps are bounded to 65536–262144,
with legal narrow architecture/batch ranges; runs do not require a long train.
This is an execution smoke test, not a learning-quality benchmark.

An observed BF16 run finished in **1.25 seconds**. An additional FP32 demo
passed the same twelve-trial checks in **10.34 seconds** while other gates
were running (`build/metal/protein/cli_f32.log`). Each worker wrote a checkpoint,
its actual suggestion and three learning-curve/cost points to INI logs. Example:

```text
suggestion ..._0001: timesteps=72893.9 lr=0.144285232
suggestion ..._0010: timesteps=67625.5 lr=0.490239948
suggestion ..._0011: timesteps=81691.2 lr=0.291195571
sweep run=11 score=0.0600 cost=0.03 steps=73728 random=0 gp_obs=11 pareto=5
```

The CLI test asserts all twelve results arrived, ten were random, the final
trial had GP observations and a nonempty Pareto front, actual steps stayed
under the cap, and saved curve/cost arrays have three points. Exact output
and final gate-run examples are included in the generated logs. Runtime cost
is variable, so GP choices vary between repetitions. The behavior/numerics
tests separately verify cost-cap masking and the fitted early-stop function.

## CPU timings

One GP, 21 dimensions, FP32, 50 training iterations and 65536 test candidates;
wall times in milliseconds. One measurement per shape, including ordinary
system variability; the laptop was also running the validation suite.
`recompute` includes Gram construction, Cholesky and two triangular solves.

| Observations | Gram | Recompute | Inverse solve | Gradient reduction | Train 50 | Predict 65536 |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | 0.072 | 0.074 | 0.039 | 0.114 | 9.639 | 74.393 |
| 256 | 0.813 | 0.848 | 0.076 | 1.292 | 122.178 | 230.811 |
| 750 | 7.001 | 7.192 | 2.773 | 11.173 | 1073.390 | 717.737 |
| 1200 | 17.928 | 18.941 | 3.601 | 29.397 | 3090.160 | 1147.943 |

The CLI uses 750 GP observations. Even with two GPs and a saturated candidate
pool this is seconds per suggestion, while the bounded CLI demo's small GP
is much cheaper. CPU is adequate; Metal GP kernels would add implementation
and synchronization cost without a demonstrated need. Log:
`build/metal/protein/bench.csv`.

## CLI/config parity audit

Scope: the native PufferLib 5 CLI in this checkout, not historical Python
PufferLib releases. Its complete command list is `train`, `eval`, `match`,
`sweep`. Render is the default mode of eval/match; there is no `render`,
`league`, `resume`, `play`, or `profile` native subcommand. Costs below are
rough engineering effort, not commitments.

| Command / major feature | Metal status and evidence | Gap / rough cost |
|---|---|---|
| `train` BF16 / `--float` FP32 | Works; learner/rollout/encoder replay gates and exact seeded end-to-end gates. | None for supported CPU environments. |
| `eval --headless` | Works from explicit `.bin` and positional `latest`; Metal reports finite `METAL_EVAL` results (`CUDA_EVAL` remains the CUDA label). | None identified. |
| `eval` render / interactive play | Raylib successfully opens 600×200 cartpole render surface, Apple M4 OpenGL 4.1 renderer, 60 FPS target; own audit process stopped after startup. | UI automation could not bind the unbundled command-line app; pixels/input were not visually verified. Existing per-env rendering assets/features vary. |
| `match [--headless]` | Works with `base.load_model_path` and `base.load_enemy_model_path`; slimevolley match: score/perf 0.5, 128 games. Rendering uses the same eval loop. | No independent match render/input smoke. |
| `sweep`, packing and worker failures | Works on one GPU with `gpus=-1`, child execution/result pipe, random/GP/Pareto/cost flow and INI artifacts. GP failures covered by tests. | Multiple concurrent GPU trials unavailable on this one-device backend. |
| Sweep distributions / goals / metrics | Shared linear, log, pow2, logit, integer handling, scale `auto`/`time`/numeric, metric path, logit score, maximize/minimize acquisition, pruning and cost cap. | Shared upstream issue: Pareto/top-k assume maximizing even when acquisition minimizes. Correct direction handling + regressions about 1 day. |
| Sweep early stop / resume | Cost fit, the configured `early_stop_quantile`, and `protein_sweep_should_stop` work in tests, but the CLI intentionally remains unchanged. | Skipped in the rough-edges pass: shared upstream blocks in `waitpid` and receives one final `TrainResult`, with no live-progress or stop channel. It does not define whether a stopped trial is a failure, censored observation, or successful partial curve, nor checkpoint/log finalization. The default remains current behavior (no early stop); inventing those semantics would be a larger upstream divergence. No sweep persistence/restart exists. |
| Self-play historical pool | Works; checkpoint creation, opponent refresh every 4096 steps, two-opponent final scoring, mean score 0.5 in bounded slimevolley smoke. | No separate league/rating server upstream. Cross-OS libc `rand_r` can change opponent selections; seeded sequence parity would need a shared RNG, about 1 day. |
| Bot ladder / `[bot_eval]` | Works; slimevolley final ladder policy 0 produced perf 0.5 over 128 games. A nonempty `[bot_eval] max_ticks=32` override also passed in an isolated config copy. | Additional ladder rungs/environment-specific bots not exhaustively tried. CLI only overrides existing INI keys; new bot override keys must be placed in config. |
| Checkpoints / `latest` | Portable FP32 master-weight `.bin` save/load works for train warm starts, native eval, self-play/match and standalone CPU eval of a Metal checkpoint. Train warm start deliberately keeps fresh optimizer/step/replay/RNG state. | Full exact resume is absent upstream. |
| INI/CLI configuration | Shared default→environment→`--section.key=value` precedence, numeric/list/string handling and hyphen aliases. Narrow sweep overrides exercised. Environment is compiled into native binary. | No runtime native environment selection, as upstream. |
| Logging / sweep visualization | Dashboard and INI config/metric arrays work; profile/Metal counters logged. `constellation` and `cache_data` compile natively on macOS. | No WandB/Neptune/TensorBoard/MLflow backend in this native upstream CLI; external bridge about 1–3 days each. Viewer was build-tested only. |
| Async/sync, RNN carry/reset, replay, V-trace, schedules | Shared training paths. Cheap sync + reset + V-trace + replay ratio 2 + entropy annealing + profile smoke passed; default async/carry covered by gates. | None identified for supported shapes. |
| Architecture, action heads, environments | Existing gates cover dense/recurrent, wide, action masks and custom encoders. Existing broad CPU-env port remains in place. | CUDA `.cu` environments and robot_arm are rejected by `--metal`; each needs a physics/kernel port, typically days to weeks. |
| `train.gpus>1`, GPU offsets, NCCL | Single Apple GPU works; runtime reports one device; NCCL shim explicitly rejects multi-GPU. | Multi-device/distributed Metal training is a substantial new backend path, several weeks and suitable hardware. |
| CUDA graphs / profiling / utilization | `base.cudagraphs` is deliberately disabled on Metal. `base.profile` and Metal timing counters work. | CUDA profiler hooks are no-ops; NVML utilization is unavailable/zero. Metal capture/Instruments integration and utilization reporting about 2–5 days. CUDA graph performance parity is separate optimization work. |
| CPU play/eval, web and build modes | `--cpu` cartpole builds and evaluates a Metal checkpoint (`match=1`, correct weight count). `--debug` is a build flag. Web remains CPU/Emscripten; `--profile` selects the CUDA build mode. | Web was not rebuilt; no Metal web backend is expected. No CUDA `--profile` build on macOS; use Metal profiling above. |

Audit logs and the exact bounded commands are in
`build/metal/protein/audit_cli.py`, `build/metal/protein/audit.log`, and
`build/metal/protein/audit/`. These are generated evidence, not committed
artifacts. The report is also copied to the requested shared docs path:
`<repo>/docs/dev/investigations/protein-port-report.md`.

Implementation commits on `metal-protein`: `6ce44420` (CUDA golden refresh),
`b3834bc5` (CPU backend), `8236e17c` (production/RNG/CLI gates), and
`55ef994c` (Python test entry points). All commits use the configured identity
and the requested Claude-Session trailer.

## Review follow-up

2026-10-26: addressed Sol's stale-worker-log gate finding and independently
regenerated the numerical fixture on CUDA-reference.

`tests/metal/test_sweep_cli.py` now creates a fresh `run-*` directory beneath
the requested output root (default `build/metal/protein/demo`) on every
invocation. It prints that directory, retains its command/output/checkpoints,
and requires exactly twelve worker INI files before checking every file's
suggestion and three-point learning curve. Earlier runs cannot satisfy the
current invocation's assertions.

Mutation check: `python3 build/metal/protein/review/mutate_sweep_cli.py`
copied `src` into an isolated build tree, disabled both worker INI-writing
blocks in `run_train` with `ctx->artifact_owner && false`, and built the real
cartpole Metal CLI there. With the original **48 stale INIs still present**
in the default output root, the mutant printed all **12 valid result rows**,
but the gate **failed (exit 1)** with `expected 12 worker logs ... got 0`.
After deleting the mutant tree, the restored production gate
`bash tests/metal/run_test_sweep_cli.sh` **passed (exit 0)** with twelve new
INIs; its sweep took 1.33 seconds. Production source was unchanged throughout.
Evidence: `build/metal/protein/review/worker-log-{mutation,restored}.log`,
`worker-log-mutation.diff`, and `worker-log-mutation-result.json`.

The independent CUDA run used an empty directory created by
`ssh -o BatchMode=yes CUDA-reference 'mktemp -d <cuda-reference>/protein-review-XXXXXXXX'`:
`<cuda-reference>/protein-review-xuGU8uNZ`. Only files beneath
`<cuda-reference>` were written on CUDA-reference. No previous source copies,
binaries, or generated fixtures were reused. `nvidia-smi` showed an idle
RTX 3090 (0% utilization, 1 MiB allocated, no running processes); process
checks immediately before compilation and execution were also empty. The
compiler was CUDA 12.8, V12.8.93; compilation and execution used `nice -n 19`.

The two source files came from an unmodified `git archive` export of commit
`ef62360be1c848efb6effc57a71b1da32cb45f52`:

| File | Git blob |
|---|---|
| `src/protein.cu` | `a9e8ca47fc2d0b67afc0ce22db083feb8d10949b` |
| `tests/test_protein_numerics.cu` | `9ce17fc72b9ec1cb09c59c7f1cdc6c18841338e8` |

Source export and the exact build/generator commands, run from `wt-protein`
and the remote scratch directory respectively:

```sh
git archive ef62360be1c848efb6effc57a71b1da32cb45f52 \
    src/protein.cu tests/test_protein_numerics.cu | \
    ssh -o BatchMode=yes CUDA-reference \
    'tar -xf - -C <cuda-reference>/protein-review-xuGU8uNZ'

# Executed through ssh -o BatchMode=yes CUDA-reference 'bash --noprofile --norc -s':
. <cuda-reference>/env.sh
cd <cuda-reference>/protein-review-xuGU8uNZ
nice -n 19 nvcc -O2 -Isrc tests/test_protein_numerics.cu \
    -lcublas -lcusolver -lcurand -o test_protein_numerics > build.log 2>&1
nice -n 19 ./test_protein_numerics --gen golden_protein_numerics.txt | tee generate.log
```

The generator produced **412 records**. The fresh file was copied back and
compared with `git show HEAD:tests/golden_protein_numerics.txt` (fixture blob
`82dfede63c6ac082c9452454d1429180337b9bc8`). The raw comparison:

```sh
diff -u build/metal/protein/review/committed-golden_protein_numerics.txt \
    build/metal/protein/review/CUDA-reference-golden_protein_numerics.txt
```

returned **1**, with exactly one difference: the committed file has two
additional leading provenance comments. All **412 names, decimal values and
hexadecimal float bits match exactly**. Removing exactly those two comment
lines from the comparison copy and rerunning `diff -u` returned **0**, with
an empty diff. Both generator payloads have SHA256
`61d2ab1ce405383ad9e78e418802db2a3b12e04d0a839dda1f51c075983ccdf5`.
The committed fixture was not modified. Full commands, GPU/compiler output,
fresh output and both diffs are retained in `build/metal/protein/review/`
(`CUDA-reference-generate.sh`, `CUDA-reference-generate.log`, `golden.diff`,
`golden-payload.diff`) and the remote scratch directory.

Final validation: **`bash tests/metal/run_all.sh` passed all 33/33 gates, process
exit status 0**, including `protein` and the revised `sweep_cli`. This run
executed the requested runner directly. Summary:
`build/metal/protein/review/gates-summary.log`; individual gate output remains
in `build/<gate>.log`.
