> Historical development log; last recorded at commit `0bee65d9` (tree archived from `e68239d9`). Statements and gate counts describe that revision, not current support.

# Metal rough-edges release pass

This pass prepares the Metal port of PufferLib 5 for publication as a
standalone fork. The comparison point is upstream PufferLib branch `5.0` at
`ea7f9893`; the work started from Metal commit `49e63975`. Changes that touch
the shared trainer are either backend-guarded or inactive under the existing
defaults.

## 1. Backend-correct evaluation label

Metal evaluation now prints `METAL_EVAL`; CUDA builds retain `CUDA_EVAL`.
A repository-wide search of scripts, sweep parsing, tests, Protein, and the
constellation/cache-data tools found no consumer of the old literal, so no
compatibility alias was necessary. The shared print site is selected by the
existing `PUF_METAL` compile-time guard.

Regression coverage: `tests/metal/run_test_eval_label.sh` performs a real
Metal evaluation, requires `METAL_EVAL`, rejects `CUDA_EVAL`, and proves the
check catches a deliberately restored-label mutant before rebuilding and
passing the unmodified source.

## 2. Metal precision help text

`build.sh --help` and its nearby user-facing comment now say that `--metal`
defaults to BF16 and that `--float` selects FP32. A historical coverage note
that could be mistaken for current build behavior was clarified.

This is help/documentation-only divergence. The build defaults and numerical
paths did not change. `tests/metal/run_test_build_help.sh` checks the current
wording, rejects the stale wording, and includes a stale-help mutation check.

## 3. Metal memory preflight

The working-set recommendation is advisory: an overrun prints a WARNING with
requested/recommended bytes and geometry keys to reduce, then continues. Only
a page-rounded single allocation exceeding `maxBufferLength` fails preflight.
The conservative shared-buffer peak now includes each later rollout stream's
four 1 MiB rings and word buffer, its 4 MiB workspace, FP32 initialization
scratch (also used by BF16 initialization), and packed weights only when the
actual packed path is eligible. Existing stream buffers are included in the
runtime snapshot. Driver-private allocations are not claimed as predictable.
No allocation layout or training arithmetic changes.

On the 24 GB M4 Air, the shipped g2048 FP32 geometry planned 14.098 GB total,
including a 13.736 GB largest buffer, against a 19.065 GB recommendation and
14.302 GB buffer limit. It completed one epoch (262,144 steps) and saved its
checkpoint. A safe small configuration one page above a forced recommendation
also completed training with the warning. These are correctness checks, not
performance measurements.

`run_test_memory_preflight.sh` now forces the hard limit for the oversized
rejection case, independent of host capacity. It compares the eight-stream
plan with actual initialized allocations, checks scratch/packed eligibility,
and catches isolated missing-ring, missing-workspace, missing-scratch,
ineligible-pack, raw-size-rounding, and disabled-preflight mutants. Focused
predicate tests also catch continuous/multi-policy/custom-encoder mutants.

## 4. NetHack async follow-up

The previously reported wait stall did not reproduce on the uncapped Mac mini
M4. NetHack FP32 ran with async enabled, seed 73, 128 agents, two buffers,
eight CPU threads, horizon 128, minibatch size 1024, no evaluation, and 131,072
training steps. All eight epochs completed with exit status 0 in 2m43.8s at
about 804 SPS; learner times remained steady at 20.0--20.45 seconds through
epochs 6--8. There were no 60-second wait warnings, allocation failures,
deadlock dumps, or assertions.

The independent async runtime stress test also completed 60.1 seconds with
three rollout threads, 446,352 rollout steps, 29,050 learner epochs, 1,841,932
dispatches, and 148,783 staged uploads. No process from either run was left
running. Because no port defect was reproduced, this item changes only
`docs/dev/investigations/metal-runtime-debug.md` and `docs/dev/investigations/metal-env-coverage-history.md`; the existing
progress-aware wait diagnostics and async stress test remain the regression
coverage.

## 5. Weights-only training warm start

`train` now honors `base.load_model_path` immediately after trainer creation
and before the first rollout. It loads policy 0's FP32 master weights while
leaving optimizer momentum, step/schedule state, replay state, environment
state, and RNG state fresh. The console calls this a warm start rather than a
resume.

Checkpoint loading requires an exact raw-file size and reports clear
missing-file and shape/precision mismatch errors. Both BF16 and FP32 builds
store FP32 master weights. Cross-precision loading also requires compatible
allocator layouts: equal shapes/counts alone are insufficient. Precision-dependent
layouts now save a `.bin.layout` sidecar and require the same build precision;
legacy files without that metadata (including upstream checkpoints) keep loading
as before with a warning, assuming the build's precision. Raw weight bytes
and default checkpoint hashes remain unchanged. This is an intentional small
divergence in the shared train entry point; when the default value remains `None`, only the existing
string lookup occurs and no device state or numerical path changes.

`tests/metal/run_test_warm_start.sh` creates a checkpoint and verifies that
loaded master weights and the first rollout's actions, values, and log
probabilities match exactly. It also verifies zeroed optimizer momentum,
clear shape/truncated-file errors, identical default-`None` and omitted-key
checkpoints, a skipped-load mutant, and the restored implementation.

The review follow-up also refreshes NetHack's `NH_MSG_FROZEN` message-table
cache on the loader's stream after checkpoint loading/casting and before the
final synchronization. The cache is per-policy, so loading an opponent cannot
replace the primary policy's table; the async actor aliases that primary cache.
A focused C++ test compiles the real cache
helper and policy loader with queued CUDA mocks, covering both random and `NH_MSG_INIT`
starting tables and two-policy isolation in both storage widths; omitted/premature
refresh and shared-cache mutants fail. Once the full runner built fast-nle,
`run_test_frozen_nethack.sh` also passed a real FP32 `NH_MSG_FROZEN` sync/async warm starts
(8 agents, H32/L1, 64 steps) with a conflicting `NH_MSG_INIT` table. The live
table and cache match after loading; the omitted-refresh mutant fails, and the
restored binary passes. Frozen BF16 cache ordering remains covered by the
focused queued-CUDA test rather than a full BF16 NetHack training run.

`run_test_checkpoint_layout.sh` exercises shipped Drone H64/L2 continuous
policies in both precisions: same-precision master weights load exactly,
both incompatible cross-precision directions and an invalid sidecar fail, a
legacy file without a sidecar loads exactly with a warning,
and an unchecked-flat-load mutant is caught. Compatible discrete layouts
still load across precisions in the guide smoke test.

## 6. Sweep early stop

Skipped deliberately. Upstream has `early_stop_quantile`, cost fitting, and a
tested `protein_sweep_should_stop`, which indicates intent, but the sweep
parent blocks in `waitpid` and receives only one final `TrainResult`. There is
no live worker-progress or stop channel, and upstream does not define whether
a terminated trial is a failure, a censored observation, or a successful
partial curve, nor how its checkpoint and log should be finalized.

Wiring only the final decision would never stop a live trial; inventing the
missing protocol and result semantics would be a substantial, rebase-costly
divergence. The CLI therefore retains its current behavior, and the limitation
is documented in the parity audit and user guide.

## 7. Metal user guide

`docs/metal-guide.md` now covers Apple Silicon/macOS 26/Metal 4 requirements,
Xcode and `DEVELOPER_DIR`, build/train/eval/match/sweep commands, BF16 versus
FP32, README M4 performance figures, memory sizing, environment and CUDA
feature differences, user-facing `PUF_*` controls, and toolchain/license/
Raylib troubleshooting. It identifies `robot_arm` as **CUDA-only for now** and
notes that internal test/debug environment variables exist without presenting
them as supported controls. The README top section links to the guide with one
short sentence.

The review fixes build SlimeVolley before match and set Cartpole's initial
sweep timesteps inside the narrowed range. `run_test_guide.sh` executes bounded
BF16/FP32 builds and training, seed overrides, warm start, headless latest/path
eval, compatible cross-precision eval, a SlimeVolley checkpoint match, and a
two-trial Cartpole sweep. A guide-content check catches missing-build, missing/
out-of-range sweep initial-value, and unqualified-portability mutants. Rendered eval, privileged Xcode setup, downloads,
and package installation are not rerun as part of these smoke checks.

## Final verification

The complete gate runner was rerun on the 24 GB M4 Air with the requested
environment:

```sh
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
PATH=<repo>/.venv/bin:$PATH bash tests/metal/run_all.sh
```

All 46 gates passed; the runner exited 0 with `GATES DONE`. The final log is
`build/gates_review_final.log`. This includes the memory, warm-start,
checkpoint-layout/frozen-NetHack, and guide gates and their mutation checks,
plus the existing FP32/BF16 goldens, replay, overlap, wide, large, and end-to-end
score/checkpoint-hash suites. The default seed references and checkpoint hashes
remain unchanged. Freeway's environment `max` macro is handled by macro-safe
maximum expressions in the preflight.
