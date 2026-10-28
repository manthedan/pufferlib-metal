# CUDA golden harness for the Metal learner

Upstream: `ea7f989371270d2948f96728bda8aec5e7634a4c`. All reference math is
called from the real `src/pufferl.cu`/`src/algo.cu` translation unit, as in
`tests/profile_kernels.cu`. Build uses `PRECISION_FLOAT`, `-O2`, CUDA 12.8, and
the existing cuBLAS default algorithm/math-mode selection. No replacement math
produces golden outputs. Python math is used only for independent validation.

## Regenerate on a CUDA reference host

The original captures used a Linux x86-64 host with an RTX 3090 (sm_86),
CUDA 12.8 and the upstream revision above. Install the CUDA toolkit, NCCL,
Raylib 5.5 Linux libraries, OpenMP, and Python >=3.9 with NumPy. From the
reference checkout, set `CUDA_HOME` and `NCCL_HOME` to those installations;
no sibling setup script or private host is required. Both builds use
`$CUDA_HOME/bin/nvcc`, CUDA headers/libraries under that root, and
`$NCCL_HOME/include` / `$NCCL_HOME/lib` (the same layout for the generator and
nested trainer build). The reference build prepends these roots to `PATH` and
`LD_LIBRARY_PATH`; set the library path likewise when running the generated
binaries if your loader does not already know these installations. `ccache` is
optional: the root CUDA build uses it only when installed. Normal CUDA builds
without `NCCL_HOME` retain their system/Python-wheel discovery.

```bash
export PUFFER_GOLDEN_DIR="$PWD/golden"
./tests/metal_ref/build.sh
./tests/metal_ref/run.sh "$PUFFER_GOLDEN_DIR"
```

`run.sh` defaults to `PUFFER_GOLDEN_DIR`, or `<repo>/golden` when unset.
It runs the operation generator twice, captures native Breakout twice,
replays the first input twice, validates CPU math, and records SHA-256
hashes, versions, comparison reports and configuration snapshots.
Generation is a separate CUDA workflow; it never updates Metal expectations.

Individual commands, from the same environment/repository:

```bash
./tests/metal_ref/generate $PUFFER_GOLDEN_DIR/run_a
PUF_GOLDEN_TRAIN="$PUFFER_GOLDEN_DIR/run_a/train_epoch3" \
PUF_GOLDEN_STOP=1 ./tests/metal_ref/puffer_golden train \
  --base.run_id=golden_a \
  --base.log_dir="$PUFFER_GOLDEN_DIR/native_logs" \
  --base.checkpoint_dir="$PUFFER_GOLDEN_DIR/native_checkpoints"
./tests/metal_ref/generate replay $PUFFER_GOLDEN_DIR/run_a/train_epoch3 $PUFFER_GOLDEN_DIR/replay_a/train_epoch3
python tests/metal_ref/golden.py $PUFFER_GOLDEN_DIR/run_a $PUFFER_GOLDEN_DIR/run_b
OPENBLAS_NUM_THREADS=1 python tests/metal_ref/validate.py $PUFFER_GOLDEN_DIR/run_a
```

No environment, learner, or schedule hyperparameters override the default
Breakout INIs. The native capture exits after the post-train dump at epoch 3;
it does not run the remaining 415 epochs or post-training evaluation.

## Layout and reader contract

`run_a/` is the reference tree; `run_b/` is the independently regenerated tree.
Each has 115 operation cases and `train_epoch3`, for 116 cases per full run.
`replay_a/train_epoch3` and `replay_b/train_epoch3` are two replays of the
**same** `run_a/train_epoch3` input. There are 234 case directories in total.

Each case contains `manifest.json` and named raw binary files, with no binary
headers, alignment gaps, strides, or compression. The manifest has:

* `schema_version: 1`, `byte_order: little`, upstream revision, precision;
* `tensors[name] = {dtype, shape, file, role}`;
* `scalars`: hyperparameters, operation, dimensions, and layout descriptions.

Dtypes are `f32` = IEEE float32, `i32` = signed 32-bit, `u8`, and `u64`.
Shapes use C order. Roles are `input`, `output`, `param`, or `state`.
Input/state/param tensors required by an operation are dumped before mutation.
Names ending `_in`/`_out` distinguish mutable values. Repeated parameter tensors
with both flat and named views are intentional; both use headerless files.

```python
from golden import load_case, compare
reference = load_case("golden/run_a/policy_ppo_breakout")
# Arrays are read-only NumPy memmaps, with explicitly little-endian dtypes.
report = compare(reference["decoder"], metal_decoder, atol=1e-6, rtol=1e-5)
```

`compare` returns bitwise equality, maximum absolute and symmetric relative
error, mismatch count, first index outside the requested tolerances, and both
values at that index. Defaults use zero tolerance. It also accepts dictionaries
of arrays. Equal NaNs compare numerically equal; bitwise comparison still checks
payloads. Integer comparisons preserve u64 differences beyond 2**53.

## Cases

| Case pattern | Count per run | Inputs and outputs |
|---|---:|---|
| `gemm_{tiny,breakout}_*` | 88 | A, B, initial C; output C; M/N/K/alpha/beta. Encoder, decoder, and MinGRU rollout/train NT; weight-gradient TN; used input-gradient NN; each parameter geometry and all five Muon Gram/polynomial/iteration alpha-beta pairs. Leading batch/time dimensions are retained to exercise wrapper folding. |
| `gate_{tiny,breakout}` | 2 | Combined gates, initial state, highway input; output and next state. |
| `scan_*` | 7 | Combined gates, highway input, initial state, terminals, grad-output and grad-next-state; output, next state, cached recurrent inputs, and all scan gradients. Plain/row Breakout pair, tiny pair, partial 35-step tile pair, and 129-step pair. Resets include t=0, middle, final, and nonzero fractional flags. |
| `sample_{tiny,breakout}` | 2 | Decoder outputs, masks, action sizes, raw Philox state and decoded counters/key/cache/index before and after; cloned-state uniform draws, actions, env actions, logprobs, values. Includes partial masks, one legal action, and all-masked rows. |
| `adv_{tiny,breakout}_{gae,vtrace}` | 4 | Values, rewards, terminals, optional importance; advantages and returns. GAE passes a null importance pointer; V-trace exercises rho/c clipping. Returns alias the value buffer, as in training. |
| `ppo_{tiny,breakout}` | 2 | Fused logits/value, actions, masks, old logprobs/values, returns and advantages; cached per-class/new logprobs, live values, importance, dlogits, dvalue, loss partials, reduced losses/count. Calls both real PPO kernels through their wrapper. |
| `muon_{tiny,breakout}_{matrix,wide,full}` | 6 | Flat and shaped params, gradients, momentum, coefficient table, lr/momentum/clip/eps values; updated params/momentum, final update and grad sum-of-squares. Single tall and wide matrices plus the entire policy parameter set. |
| `policy_{tiny,breakout}` | 2 | Params, obs, terminals, initial state, masks/actions/old logprobs, dlogits/dvalue and zero grad-next-state as inputs; decoder, layer outputs/next states, caches, every parameter gradient, layer initial-state gradients and encoder-output gradient. |
| `policy_ppo_{tiny,breakout}` | 2 | Full policy followed by real GAE, PPO and backward. Also dumps rewards/old values, advantages/returns, emitted dlogits/dvalue, losses/partials and every parameter gradient. |
| `train_epoch3` | 1 | Entire input required by one real train call plus post-train state and each minibatch loss; details below. |

Real dimensions: 4096 rollout agents, observation width 118, hidden width 32,
four MinGRU layers, three categorical actions, horizon 32, minibatch 65536 =
2048 agent sequences. Full real policy cases use the actual `(4,4096,32)` state
shape and agent offset 2048. The tiny policy uses H=8, two layers, B=2, T=8,
input width 12, and a nonzero agent offset. Odd H=5 tests cover gate/scan/GEMM
and single-matrix Muon edge shapes without violating upstream packed/aligned
parameter-buffer assumptions.

Parameter registration order is encoder weight `(32,118)`, decoder weight
`(4,32)`, then four complete MinGRU weights `(96,32)`. There are 16,192 FP32
parameters; MinGRU matrices are never split into three optimizer matrices.

## Real train capture and replay

The captured call enters with `epoch=3`, after three completed learner epochs,
and produces epoch 4 parameters. Global step is 524288 because the native loop
increments it immediately before calling train. Async ready slot is 1.
The 55M configured schedule is preserved: 419 complete train epochs.

The 39 tensors include all six parameters individually and flat, before/after;
momentum before/after; ready-slot time-major obs/actions/logprobs/values/rewards/
terminals/masks; both initial-state slots; train-state input/output; the zero
backward grad-next-state input; actor params/state; Philox states and the legacy
rng-offset buffer; action sizes; loss accumulator input/output; final optimizer
update; and the exact minibatch row ordering and per-minibatch loss matrix.

The input schedule is `lr=0.06270512938499451`,
`ent_coef=0.004893647041171789`. Minibatch row offsets are `[0,2048,0]`:
`int(1.88528275 * 131072 / 65536) == 3`. All other learner scalars are in the
manifest. Loss columns are PG, VF, ENT, TOTAL, OLD_APPROX_KL, APPROX_KL,
CLIPFRAC, IMP, count; each per-minibatch count is one.

The capture-only pre-hook waits for the already-started actor prefetch before
reading RNG and actor recurrent state. This changes overlap timing for the
captured call, while retaining the ready rollout and frozen actor snapshot.
The first three learner epochs use the normal async/cudagraph path. The extra
loss-recording kernel is included when each train graph is first captured;
it writes a separate buffer without modifying learner state. Training itself
consumes no RNG. Scratch buffers are recreated/overwritten; persistent inputs
are dumped explicitly. The CUDA replay executable loads the saved tensors into
a newly created default Breakout learner and calls `train_impl` once. The
prefetch workers stay idle for replay. No environment simulator state is needed
for replay of a train call.

Philox ABI on this build is 64 bytes: ctr at 0, cached output at 16, key at 32,
STATE at 40, normal flags at 44/48, cached float at 52, cached double at 56.
Sampling cases also expose 12 decoded raw 32-bit fields in the order recorded
in `scalars.rng_fields`; reinterpret their `i32` bits as u32. Inputs deliberately
discard `agent % 7` uniforms after `curand_init(73,agent,0)` to exercise cached
word indices and counter carry. Uniform probes copy the state before drawing,
so they expose exactly the draws used by the unmodified sampler.

## Determinism and validation

On CUDA-reference (RTX 3090, driver 580.178.04, CUDA 12.8), all 116 cases from the two
fresh runs are bitwise identical, including every input, output, parameter,
state file, and manifest. Both same-input train replays are bitwise identical
and reproduce the live capture exactly. Measured run-to-run max absolute and
relative errors are zero. No operation exhibited nondeterminism in these runs.
cuBLAS retains its default algorithm selection and upstream per-stream 4 MiB
workspaces; the tested reductions are deterministic two-pass/tree reductions,
with no atomic loss/norm accumulation. These are measurements on this toolchain,
not a promise of bit parity across GPU models or CUDA versions.

Plain versus row scan implementations have deterministic rounding differences:
max absolute output difference `4.76837158203125e-7` for Breakout and the
129-step pair; `2.384185791015625e-7` for the partial tiny pair. The per-variant
reference files are authoritative. This is a variant difference, not temporal
noise.

`validate.py` passes 166 independent numerical checks: every GEMM against
NumPy math, gate/scan forward and backward, GAE/V-trace, Muon, tiny policy
forward, finite differences for every tiny-policy parameter, PPO dlogits/dvalue
and total loss. It additionally checks sampling legality/value copies, RNG
word-index diversity, u64 comparison accuracy, and NaN/Inf comparison behavior.
`inventory.json` validates every shape/dtype/file byte count and records SHA-256
for each case file. Reports are `determinism_live.json`,
`determinism_replay.json`, `replay_vs_live.json`, `scan_variants.json`,
`run_a_validation.json`, `summary.json`, and `metadata.json`.

## Upstream hooks (all behind PUF_GOLDEN_DUMP)

Only 13 lines were added to upstream files; `src/algo.cu` is unchanged.

* `src/pufferl.cu:1461`: include `tests/metal_ref/train_dump.h`.
* `src/pufferl.cu:1552`: record each minibatch loss from actual PPO partials.
* `src/pufferl.cu:1613`: allocate recording storage before graph capture and
  capture pre-train input after schedules are uploaded.
* `src/pufferl.cu:1654`: capture outputs after train-stream synchronization.

## Porting-map corrections and resolved details

* Kernel-map rows 22-25 and sections 3.3/3.5 misattribute `cache_imp_and_v`,
  PPO and advantage kernels to `pufferl.cu`. Actual locations are
  `src/algo.cu:1407`, `:1496`, `:1628`, `:1643` (PPO wrapper), and `:1716`.
  The native train dispatch remains in `pufferl.cu`.
* Kernel-map row 27 describes `rng_offset` as device counters. That buffer is
  dead in this upstream revision, as the host map correctly states. Actual
  consumed counters/word index/key are inside persistent Philox states.
* Kernel-map statements that 5.0 GEMMs always store BF16 need a build qualifier:
  `PRECISION_FLOAT` uses `CUDA_R_32F` inputs/outputs and `CUBLAS_COMPUTE_32F`.
  No BF16 round-trip occurs in this dataset.
* Row scans have the same formulas but are not bit-identical to plain scans,
  as the measured differences above show. Treat their files independently.
* The host-map UNVERIFIED Philox ABI is now measured/exercised as described
  above. The packed/aligned parameter warning in host-map section 3.2 is valid;
  tiny full policies deliberately use aligned dimensions.

All work, builds, pip installs, caches, logs, and dumps remain under
`<cuda-reference>` on CUDA-reference. No sudo/apt was used. Existing baseline binaries
`puffer` and `puffer_f32` were left untouched and remain untracked.
