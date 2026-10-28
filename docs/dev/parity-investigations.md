# Metal parity investigations

This record separates local numerical agreement, execution smoke tests and
learning evidence. Current support and commands are in the [Metal guide](../metal-guide.md),
[environment coverage](../metal-env-coverage.md) and [test suite](../../tests/metal/README.md).
Historical tests used the stated configurations; gate counts are not current
coverage inventories.

## H1024 backward discrepancy

In the one-layer FP32 B4/T64/I36/H1024 fixture, only hidden element
(batch=1, time=48, hidden=817) crosses MinGRU's derivative discontinuity.
The activation is a+0.5 for a>=0 and sigmoid(a) otherwise. Values are
continuous at zero, but the one-sided derivatives are 1 and 0.25. A valid
FP32 projection can therefore change a backward gradient fourfold without
a material forward difference.

| Quantity | Value |
|---|---:|
| Float64 projection, entirely from golden inputs | `-2.384857582249378e-9` |
| Float64 dot product using Metal's rounded encoder output | `-3.251539544557591e-9` |
| Metal tensor_ops NT projection | `+1.147246408983449e-9` |
| Independent CPU ascending FP32 `fmaf` loop, same operands | `+1.147246408983449e-9` (bit-identical to Metal) |
| Independent CPU reverse FP32 `fmaf` loop | `+1.220488599074088e-9` |
| Sum of absolute products for that dot | `0.2044148581981533` |
| Standard `gamma_1024 * sum_abs_products` FP32 bound | `1.247725436111538e-5` |
| Float64 canonical `grad_combined[1,48,817]` | `-3.052075868061584e-6` |
| Metal `grad_combined[1,48,817]` | `-1.220830290549202e-5` |
| Independently predicted branch-gradient delta | `-9.156227604184751e-6` |

The independent ascending FP32 FMA loop exactly reproduces Metal. TN dW
and NN dX are correct on their actual operands. CUDA policy captures contain
final gradients but not the combined projection or scan-gradient intermediates;
those were reconstructed independently, not presented as observed CUDA dumps.
Standalone H1024 scans pass their direct comparisons.

| Comparison | Maximum absolute error | Normalized error |
|---|---:|---:|
| Metal scan combined gradient vs full float64, before branch accounting | `9.1562270e-6` | `6.4169e-2` |
| Same scan comparison, excluding the single crossing element | `2.2359e-11` | — |
| Metal dW vs float64 GEMM on the **same Metal operands** | `4.1599e-11` | `5.9333e-7` |
| Metal dX + highway vs float64 on the **same Metal operands** | `9.8561e-11` | `3.9725e-7` |
| CUDA dW vs canonical float64 policy backward | `4.1679e-11` | `5.9447e-7` |
| CUDA encoder gradient vs canonical float64 | `4.5205e-11` | `1.8220e-7` |
| CUDA encoder dW vs canonical float64 | `6.9739e-10` | `3.7558e-7` |

Propagating the single derivative delta through projection dW, dX and encoder
dW reduces their normalized Metal/CUDA errors from 1.205e-2, 5.762e-4 and
7.553e-5 to 2.08e-7, 4.55e-7 and 7.52e-7. H256/H512 controls have no branch
crossings. The three GEMMs use ordinary tensor operations: NT (256,3072,1024),
TN (3072,1024,256), NN (256,1024,3072). K=256 excludes split-K dW, and H1024
is ineligible for fused MinGRU. This was an ill-conditioned comparison, not
an allocation, indexing or kernel defect.

The reader restricts the exception to L1/H1024 FP32. It requires a sign
crossing within the ordinary FP32 dot-product error bound, propagated encoder
error included, and validates the complete forward/backward computation in
float64 at 1e-5. CUDA final gradients must independently match canonical
float64. Only the analytic branch delta adjusts the three expectations;
the existing 1e-4 policy bound and printed raw errors remain. Thirty dumped
H256/H512/H1024 tensors were byte-identical before and after reader changes.

### BF16 cancellation

The separate gate_h1024 state element 3915 (b=3,h=843) has hidden/gate/state
0.48828125 / -0.54296875 / -0.57421875. The exact float64 result is
-4.3031267678e-6. CUDA gives -4.321336746216e-6 and Metal
-4.261732101440e-6: a 5.960464477539e-8 difference, two output ULPs.
Both are one BF16 ULP from the directly rounded float64 reference
0xb690 = -4.291534423828125e-6.

The reader keeps the one-ULP direct comparison. An exception requires
abs(exact) < 2^-8 * (abs((1-z)*state)+abs(z*h_tilde)) and **both** backends
within one ULP of the independent result. No absolute floor or larger ULP
budget is added. Binary64 must round directly to BF16: the regression
(-3.65625, 1.875, -0.1640625) rounds to 0x378b, whereas passing through
FP32 incorrectly yields 0x378a. Tests cover all 32,639 adjacent positive
finite BF16 pairs, both signs, midpoints and neighboring doubles, plus
subnormals, overflow, zero, infinities and NaNs.

Committed host controls require valid cases to pass and six bad cases to
exit 1 with numerical diagnostics: out-of-bound sign crossing, bad scan dC,
dW or dX, and either backend two ULPs from the cancellation reference.
Three isolated reader mutations are caught, followed by restored passes.
The complete wide audit verifies 276 cases, 3,312 tensor files and 3,645
hashes; all eight Freeway/Tetris train comparisons and noise probes pass.
The final applicable historical suite passed 31/31; two then-unavailable
Protein gates were excluded. Golden data and production outputs were unchanged.
See [wide CUDA fixtures](../../tests/metal/README.md#wide-cuda-fixtures) for commands.

## Protein CPU backend and public Sobol data

Metal uses the shared `src/protein.cu` search-space, GP training, acquisition,
Pareto, pruning, cost-fit and classifier logic. `src/metal/protein_cpu.h`
uses FP32 Accelerate Cholesky, triangular solves and BLAS in place of
cuSOLVER/cuBLAS. Kernel gradients retain CUDA's 16x16 tile order, 256-lane
binary reductions and host fold. Prediction batches at most 4096 candidates;
at 750 observations, two matrices use about 23.4 MiB instead of 375 MiB for
65536 candidates, without changing the candidate set. Scratch is preallocated.
Philox preserves the CUDA state layout/subsequence/counter and fused uniform
conversion. A private glibc-compatible stream supplies host randomness without
changing the trainer/environment libc RNG.

The original numerical fixture was stale: pristine CUDA failed 157/412
records (first GP score loss 0.329143047 versus old 0.331116319). It was
regenerated from pristine source on RTX 3090 / CUDA 12.8, then independently
regenerated again: all 412 numeric payloads and bits matched. The payload
SHA256 is `61d2ab1ce405383ad9e78e418802db2a3b12e04d0a839dda1f51c075983ccdf5`.
No Metal output was used to update it.

| Check | Recorded result |
|---|---|
| CUDA numerical fixture | 412/412 bitwise |
| Metal numerical fixture | 318 bitwise + 94 within unchanged atol=1e-5, rtol=1e-4; zero failures; max absolute difference 1.90735e-5 |
| Exact-mode suggestion trajectory | 116/116 components in 29 suggestions bitwise CUDA-identical |
| Exact-mode RNG | 1,464 Sobol/Philox/glibc records bitwise; Sobol dimensions 1/4/21/20000 |
| 21D synthetic task, 40 trials | Random best 1.72937; GP best 3.68107 |
| Held-out accuracy | Normalized score RMSE 0.000407; log-cost RMSE 0.000575; limits 0.02 |
| Isolated source mutations | 9/9 caught; clean tests and ASan/UBSan pass |
| Real CLI smoke | 12 child trials: initial config, ten random, one GP-guided; every trial writes checkpoint and three-point INI curve |

The CLI gate uses a fresh directory and requires exactly twelve worker INIs.
A disabled-writer mutation failed with zero new logs despite 48 stale INIs
and twelve valid result rows. Near acquisition ties, BLAS/factorization,
transcendental, host contraction and equal-key sort differences can change
choices. Wall-clock costs also differ, so arbitrary real sweep trajectories
are not promised to match.

One-shot CPU diagnostic timings (ms), 21 dimensions, FP32, 50 iterations and
65536 candidates, measured while validation also ran; these are not isolated
performance benchmarks:

| Observations | Gram | Recompute | Inverse solve | Gradient reduction | Train 50 | Predict 65536 |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | 0.072 | 0.074 | 0.039 | 0.114 | 9.639 | 74.393 |
| 256 | 0.813 | 0.848 | 0.076 | 1.292 | 122.178 | 230.811 |
| 750 | 7.001 | 7.192 | 2.773 | 11.173 | 1073.390 | 717.737 |
| 1200 | 17.928 | 18.941 | 3.601 | 29.397 | 3090.160 | 1147.943 |

### Public Sobol replacement

Public builds embed **394,264 bytes** of compact Joe–Kuo construction data,
replacing the 2,640,000-byte cuRAND export (85.07% reduction). The committed
publisher-text excerpt covers dimensions 2–20,000; dimension 1 is implicit.
Keeping 20,000 dimensions preserves custom configuration axes beyond the 21D
test. Standard recurrence expands once per process to [20000][33] uint32
rows, 2,640,000 bytes. No build-time download is needed.

Each dimension uses a digital XOR shift: low 32 bits of successive SplitMix64
outputs from fixed seed 0x50554646534f424c, incrementing by
0x9e3779b97f4a7c15 before mixing with uint64 wrapping. Sweep seeds remain
uint32 Sobol offsets and draws use Gray-code XOR evaluation. Public conversion
clamps a rounded 1 to the largest FP32 below 1, preserving [0,1). Exact mode
keeps cuRAND conversion unchanged, including its possible rounded 1 endpoint.
The public sequence is digitally scrambled Sobol with the same dimensional
capacity; individual suggestions and optimization outcomes need not match CUDA.

| Data | SHA256 |
|---|---|
| Full publisher text (1,887,612 bytes) | `68eedd2a4e3b659b9695e7aff0f8ac68718bcf620730fc3d3a8c65df2a067441` |
| Committed publisher-text excerpt (1,771,202 bytes) | `af7492ff6eb0c4e1d968a15710d6496e61709d151821859841b0c16893d7b905` |
| Committed compact public parameters | `adca16e3a020a0ee099fd130b6cb5f46628c14fe629909b7fa850eb585d07210` |
| Expanded public rows, including shifts | `e6c1be592b7b514b41adb06d9f2e364471e8e4670825b1428e8974e2a2072749` |
| External cuRAND table | `fce775e7962beda83fa73c64c628a0323a94eadf5d32bad97d1dc6159e52e5f0` |

Source/license details remain next to the data in
[Joe–Kuo provenance](../../tools/metal/joe-kuo/README.md), [THIRD_PARTY](../../THIRD_PARTY.md)
and NOTICE. The live UNSW URL returned 404 on the validation date; the
publisher's repository supplied a byte-identical full source. Plain public
directions do not reproduce cuRAND's scrambled directions: 0/20,000 matching
rows, including reverse/byte-swap/XOR comparisons. Public code contains no
cuRAND directions or constants; exported expected RNG results remain tests.

Offline verification and regeneration:

```sh
python3 tools/metal/gen_protein_sobol.py --check
python3 tools/metal/gen_protein_sobol.py --from-compact src/metal/protein_sobol_public.bin \
  --check --expanded-output build/metal/protein/public-expanded.bin
# Optional full publisher-source verification:
python3 tools/metal/gen_protein_sobol.py --source /path/to/new-joe-kuo-6.21201 --check
python3 tools/metal/gen_protein_sobol.py --download --check
```

The public gate independently parses the pinned text, regenerates the compact
bytes and all expanded runtime bytes, and rejects a modified source row.
It checks the first ten 3D points, nonsingular directions, 256-point dyadic
stratification for all 20,000 dimensions before/after shifting, no duplicate
points in the first 4,096 draws, endpoints, deterministic fresh-process GP
suggestions and a real two-trial sweep. Public, guide and CLI gates explicitly
select public mode. The historical complete replacement suite passed 58/58;
with the private fixture absent, only exact Protein skips and public mode passes.

`PUF_PROTEIN_SOBOL_TABLE` selects an external exact-cuRAND fixture at first
use per process. Missing data is a skip in fixture-dependent gates; an explicit
empty/unreadable/truncated/oversized/wrong-hash path fails instead of falling
back. Export instructions, size and SHA are in the
[reference harness](../../tests/metal_ref/README.md#external-curand-protein-fixture).
The new host-only exporter option/wrapper was not executed on CUDA during
this work. The public bundle excludes this private export.

CLI scope remains native `train`, `eval`, `match`, `sweep`; there is no exact
resume, distributed Metal trainer or concurrent multi-GPU sweep. Warm start
loads weights only, subject to layout compatibility. Protein's early-stop
function is tested, but the CLI waits for one final worker result and has no
live-progress/stop protocol. Shared Pareto/top-k logic assumes maximization
even when acquisition minimizes. Rendering startup and visualization tools
were build/smoke tested, not exhaustively checked for pixels/input.

## Training log history

Two issues affected saved INIs and Protein observations. Upstream binning
added a sample before closing at most one crossed boundary, displacing sparse
samples and repeating stale means. The port additionally retained the previous
episode-bearing snapshot, refreshing only steps and uptime when no episode
finished. Thus epoch/timings/losses could remain stale; Freeway's saved epoch
1115 persisted even though training reached 1144. Upstream also omitted
objective-free final events and assumed dense metric keys.

For P>1 output points, final step F and w=F/(P−1), the first P−1 values
are arithmetic sample means in half-open intervals [j*w,(j+1)*w). Boundary
samples enter the following bin. Close all crossed bins before admission;
empty bins carry the last nonempty mean, leading empties use the first
observed value, and absent keys contribute no samples. The last value is
the exact final snapshot. P=1, one-sample and zero-step histories are defined.

Each logging event refreshes every present field; absent episode statistics
carry forward, while env/n becomes zero. Once the objective appears, every
logging event enters history, including the unthrottled final epoch. Ordinary
0.6-second logging cadence, training, RNG and checkpointing are unchanged.
Protein uses the same complete snapshot/binning as INI. Carried scores are
last-known statistics, not new episodes. This intentionally changes historical
bin means/costs and may change later sweep choices. If no objective ever
appears, return one zero-score point at actual final steps/cost; the old code
advertised multiple points with unpopulated entries. Evaluation/self-play
overlays and valid negative objectives remain supported.

`run_test_log_history.sh` compiles the actual host functions with ASan/UBSan,
checks hand-calculated dense/sparse/boundary/missing-key fixtures and rejects
the pinned upstream binning and NaN mutations. Real FP32 Freeway runs compare
the result-fd payload with saved INI, allowing FP32 serialization rounding.
The scheduled 60-epoch case (1,966,080 steps) has logging epochs 31 and 60,
final env/n=0 and fresh epoch/cost; the stale-snapshot mutant leaves all 32
outputs at 31 and fails. A two-epoch 65,536-step case tests the one-point
fallback; a non-episode objective tests late episode keys. Historical full
suite: 40/40, all 20 small seeded mode comparisons unchanged.

## Environment coverage and training evidence

The 2026-10-24 FP32 audit attempted 85 Ocean directories (84 environments
plus shared OSRS support): 81 OK, one build failure (LAMMPS), one runtime
failure (missing Drive maps), one then-CUDA-only Robot Arm, one support-directory
skip, zero unported-kernel results. Robot Arm was subsequently ported;
see [its evidence](robot-arm.md). These are historical counts, not current
support totals. Smoke geometry was H64/L2/T16, two learner updates, full-batch
minibatches, up to two buffers/threads, agents rounded for whole environments,
self-play/evaluation off, one GPU and replay ratio raised to one when needed
to ensure learning executes. Each pass required epoch 2, exit 0 and finite losses.

Repairs included entity encoders for Minimal/Asteroids; Impulse Wars' C API
linkage; flat indexing for Clang-incompatible MOBA/NMMO3 VLA casts; portable
NetHack build tools; OSRS gather/pool/backward; Craftax/NMMO3 deterministic
2^24 fixed-point embedding sums using paired uint32 atomics; and 50 active
NetHack kernels plus conditional sampling/PPO. The OSRS item upload is a
thread-safe static initialization. Pool argmax differences must be genuine
FP64-validated ties, never just a permitted fraction of changed indices.

| Numerical evidence | Result |
|---|---|
| Entity independent double reference | B=1/7/9, H=1/33/65; 2e-5*(1+abs(reference)), observed max below 2e-6 |
| Initial six custom encoders | 12 CUDA cases deterministic and guarded CUDA source bit-identical; Metal all pass at 1e-5. Worst NMMO3 1.96e-6, Craftax 1.71e-6, Colosseum 1.45e-6 |
| Inferno FP32 pooling | 3 flips among about 1M entries, relative gap about 3.4e-8; invalid/non-tied mutations rejected |
| NetHack extended audit | 14 total cases / 1,124 tensors pass at 1e-5; both NetHack cases have 328 tensors, no forward argmax flips; RNG/sampling exact |
| BF16 port | 18 cases, two CUDA runs byte-identical; 1196 comparisons + 290 independent exact/FP64 checks per run; 64 poisoned primitive checks |

BF16 full-network bounds are 4 ULP or 2^-5 tensor max; direct arithmetic
uses 1 ULP or 2^-12 tensor max against FP64. Copies, indices and fixed-point
scatters are exact. NetHack bias propagates prior GEMM rounding; OSRS pooling
uses the unrounded FP32 GELU tile. CUDA itself exceeds the stricter direct
budget in 23 measurements, worst norm 0.007497; these are reported separately
and do not relax Metal's gate. Worst Metal/CUDA BF16 network norm is 0.009346
(NetHack). FP32 tensors remained byte-identical in 1196 comparisons.

### BF16 bounded training

Seed 1, 64 epochs, H64/L2/T16 and the bounded sweep geometry; 262144 steps
except NMMO3's whole-agent geometry at 2097152. All 27 runs had finite losses
and checkpoints. Scores are final logged training values, not evaluation.
NetHack windows vary and NMMO3's flat score 1 is not evidence of solved learning.

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

Two-epoch actual-default-geometry smokes for the four smaller environments:

| Env | Metal BF16 | Metal FP32 |
|---|---:|---:|
| minimal | -60.3013 | -60.5784 |
| asteroids | 0.82153 | 0.82102 |
| osrs_zulrah | 0.11951 | 0.11877 |
| osrs_pvp | 0.001449 | 0.001482 |

### FP32 default-length runs

Seed 73, unchanged tuned configurations, default requested 10,000 evaluation
episodes; batched evaluation overshoots. The CUDA reference is RTX 3090,
CUDA 12.8, driver 580.178.04. Mean SPS is executed steps / final training
uptime, excluding evaluation; final SPS is the displayed interval. Training
scores are the last completed-episode snapshot; CUDA's less frequent logs
include more early learning, so these are not aligned learning curves.
The M4 was shared with other work; these single runs are not isolated hardware
benchmarks. Tables below preserve the JSON numbers without new rounding.

| Environment | Requested steps | Executed steps | Epochs |
|---|---|---|---|
| cartpole | 15000000 | 14942208 | 114 |
| pong | 8000000 | 7995392 | 122 |
| flappy | 20000000 | 19922944 | 152 |

| Backend | Environment | Train score | Train perf | Eval score | Eval perf | Eval episodes |
|---|---|---|---|---|---|---|
| Metal | cartpole | 199.872 | 0.994 | 199.8463592529297 | 0.994260311126709 | 12288 |
| Metal | pong | 20.98 | 0.999 | 20.993791580200195 | 0.999717652797699 | 10148 |
| Metal | flappy | 46.707 | 0.977 | 51.04318618774414 | 0.9716699123382568 | 10466 |
| CUDA | cartpole | 70.834 | 0.349 | 199.4488067626953 | 0.9922711253166199 | 12288 |
| CUDA | pong | 12.211 | 0.74 | 20.815980911254883 | 0.9918094873428345 | 10037 |
| CUDA | flappy | 45.514 | 0.951 | 50.64474868774414 | 0.9687778949737549 | 10539 |

| Backend | Environment | Training seconds | Mean SPS | Final SPS |
|---|---|---|---|---|
| Metal | cartpole | 3.271467924118042 | 4567432.218987226 | 4.8M |
| Metal | pong | 16.892578125 | 473307.97826338303 | 570.4K |
| Metal | flappy | 9.881886005401611 | 2016107.450451237 | 2.7M |
| CUDA | cartpole | 0.6158826351165771 | 24261453.640711382 | 27.1M |
| CUDA | pong | 2.8179337978363037 | 2837324.2856660113 | 3.5M |
| CUDA | flappy | 1.369692087173462 | 14545564.062586935 | 15.8M |

### OSRS Zulrah FP32

Requested 33,817,236 steps; executed 33,816,576 (129 epochs). Configurations
matched between backends. One Metal seed and three CUDA seeds establish
learning execution, not equivalence. The reported CUDA evaluation mean/SD
were 3.225/0.064; Metal seed 73 was about 1.5 SD lower. The Metal run preceded
the thread-safe item-upload fix; all OSRS smokes passed again after the fix.

| Backend | Seed | Train score | Train wins | Eval score | Eval wins |
|---|---|---|---|---|---|
| Metal | 73 | 3.071889117360115 | 2.1826618686318398 | 3.1314027309417725 | 2.2212727069854736 |
| CUDA FP32 | 73 | 3.275067001581192 | 2.304786577820778 | 3.2974071502685547 | 2.3212077617645264 |
| CUDA FP32 | 74 | 3.172569513320923 | 2.2284901291131973 | 3.1972880363464355 | 2.252990484237671 |
| CUDA FP32 | 75 | 3.1573547422885895 | 2.225034177303314 | 3.1794216632843018 | 2.2425944805145264 |

| Backend | Seed | Training seconds | Mean SPS | Eval episodes | Eval episode length |
|---|---|---|---|---|---|
| Metal | 73 | 398.5467360019684 | 84849.7125813445 | 12288 | 593.4961547851562 |
| CUDA FP32 | 73 | 56.4394965171814 | 599165.0898180055 | 12288 | 595.6897583007812 |
| CUDA FP32 | 74 | 56.32669115066528 | 600365.0367025081 | 12289 | 590.5281372070312 |
| CUDA FP32 | 75 | 52.63885998725891 | 642426.0709328661 | 12288 | 591.9645385742188 |

### NetHack fixed-budget FP32

Full default H1024/L4 policy and learning hyperparameters, seed 73; reduced
geometry: 64 agents, one buffer, two threads, T16, minibatch1024, async=0,
16,384 steps / 16 epochs, 128 requested evaluation episodes. Both runs exit 0
with finite losses. This is a short functional comparison, not convergence.
The Metal timings were captured during the system-wide GPU slowdown diagnosed
below; the apparent 72x gap is not a backend performance claim.

| Backend | Train score | Eval score | Eval episodes |
|---|---|---|---|
| Metal | 2.0 | 2.4042553901672363 | 141 |
| CUDA FP32 | 1.84 | 2.470588207244873 | 136 |

| Backend | Training seconds | Total wall seconds | Mean SPS | Final SPS |
|---|---|---|---|---|
| Metal | 102.7994339466095 | 107.97043187500094 | 159.37830950031582 | 120.8988121312734 |
| CUDA FP32 | 1.4269862174987793 | 2.4232790926471353 | 11481.540465553946 | 11138.756934652993 |

| NetHack case | Tensors | Worst norm error | Head worst norm error | Argmax flips |
|---|---|---|---|---|
| nethack_default | 328 | 3.4606729130164808e-06 | 4.316326820674441e-07 | 0 |
| nethack_small | 328 | 1.8593761272467772e-06 | 2.091378168019649e-07 | 0 |

Unsuccessful attempts, excluded from completed training evidence:

| Attempt | Requested steps | Exit | Wall seconds |
|---|---|---|---|
| metal_default_alloc | 1048576 | -6 | 1.9335964579950087 |
| metal_async_wait | 262144 | -9 | 288.39179616598994 |
| metal_serial_large | 65536 | -9 | 164.29405695898458 |

The default geometry failed allocation. The 128-agent/T128 async run reached
five epochs (81,920 steps) before termination during the slowdown; the sync
attempt stopped before its first epoch. A later eight-epoch async completion
is recorded below.

### Configuration provenance and coverage limits

The matched FP32 runs recorded identical configuration bytes on both hosts:

| Configuration | SHA256 |
|---|---|
| config/default.ini | `7f493977f49f2537a59c18c2a283f062e5d6063d9e130c2d15f6be075cda1084` |
| config/cartpole.ini | `5c67f29a8b2e53f9f64be36c28c50dbf2a4a653f03a4420aedc150693a14ecf2` |
| config/pong.ini | `fc4b26ef110b5bef75e69a39e86f8ead85140640c68932a855309dc41e3cd923` |
| config/flappy.ini | `f443679ac395f267d800f1f39321e558d5a404aec467192e6a1ce79a01731014` |
| config/osrs_zulrah.ini | `93d1288442a7807a57e2d5bad0eb514a172d2e3d6408f10fc74ff5dd01328de9` |
| config/nethack.ini | `db2ed9f49ec6d0c81f3f1a351368e9c656919795330cc7f815108277756e7344` |

Optional MINIMAL_ATTN/N3_ATTN PTX encoders remain unsupported. Metal rejects
NH_MSG_V4096, NH_MSG128, NH_NO_APANEL and defined NETHACK_V3_K values other
than 16; explicit 16 matches default preprocessing. Craftax long-budget
convergence, alternate observation/action modes, self-play variants and all
smoke-pass environments' long training are not covered. LAMMPS and Drive
training maps are external dependencies; two demo maps do not replace WOMD.
See [current coverage](../metal-env-coverage.md) for the supported configurations.

## Metal runtime debugging

### Distinguish back-pressure from a stall

A stack in `begin_locked` → `waitUntilSignaledValue` may mean the stream's
command-buffer ring is full and waiting on already committed work. Inspect
progress across multiple samples rather than infer deadlock from one stack.
The original NetHack investigation found both async and sync paths slow
while independent probes were degraded with no trainer process running:

| Probe | Healthy | Degraded |
|---|---|---|
| Sustained device copy | ~100 GB/s | 2.1–2.3 GB/s, bursts up to 45 GB/s |
| FP32 FMA | ~3.5 TFLOP/s | ~150 GFLOP/s |
| Breakout | 1.8–1.98M SPS, ~56 ms/epoch train | 50–200K SPS, 380–850 ms/epoch train |

Device utilization read 100%, while client accounting did not expose MTL4
clients in that investigation. Thermal state was nominal, low-power mode off
and memory pressure absent. Kernel ApplePPMPolicyCPMS power-budget logs fell
from an unconstrained 65000 to 2500–11000; throughput tracked the cap.
Nominal thermal status alone therefore does not establish healthy throughput.
Avoid attributing loaded-machine SPS or isolated profiles to code changes.

Runtime waits use 5-second slices. After 60 seconds a warning dumps each
stream's committed/completed values and pending dependencies. Abort occurs
after 300 seconds without **any process stream** completing work. In the
slow run train counters advanced 12/8→57/53→66/62; rollout completion
advanced 286→476. A train commit 59 waiting for dW 7, which waits for train
58, is not a cycle if train 58 has no dependency. Long buffers on a capped
GPU can delay small rollout work behind them.

Contributor workflow:

1. Compare independent bandwidth/ALU probes and inspect concurrent GPU load,
   power and memory pressure. Do not benchmark while other tests run.
2. Capture stream dumps and repeated stack samples; identify the committed
   producer for each waited value and whether completions advance.
3. Compare bounded async and sync runs with the same geometry. Use
   `PUF_METAL_TRACE=FILE` and `python3 tools/metal/trace_timeline.py FILE`
   for queue/wake attribution; `PUF_METAL_KPROF=2` serializes kernels and
   `PUF_METAL_GEMM_LOG=1` shows selected shapes/paths. Disable evaluation
   during KPROF diagnostics.
4. Run `bash tests/metal/run_test_runtime_stress.sh 60 3 120` for host handoff,
   staged uploads, learner/dW joins, legacy streams and allocation drains.
   `STRESS_DROP_WAIT=1` is a negative control, not a normal setting.

The stress test changes minibatch inputs to expose stale reads and queues a
GEMM before busy-stream uploads so the staging path really runs. The removed
join control failed in 931/970 epochs. Allocation churn is capped at 8K
cycles because the registry has 16K slots. Historical audit limitations:
concurrent record/wait on the same event's non-atomic stream/value fields and
a timing gen/stamped race were identified outside the trainer's usage; the
investigation did not establish those unsupported usages as safe.

A later FP32 NetHack run on an uncapped 16 GiB M4 completed the previously
problematic async geometry: seed73, 128 agents, two buffers, eight threads,
T128, minibatch1024, 131,072 steps, evaluation off. All eight epochs completed
in 2m43.8s, about 804 SPS, learner 20.0–20.45 s/epoch, with no 60-second
warnings, assertions or allocation failures. Reported live memory was 1.8 GB
against a displayed 12 GB budget. The independent 60.1-second stress passed
446,352 rollout steps, 29,050 learner epochs, 1,841,932 dispatches and 148,783
staged uploads. This closes the apparent async-stall report for that geometry.

### Allocation diagnostics

`PUF_METAL_ALLOC_LOG=1` logs allocations. Compare page-rounded requested
buffer size with `maxBufferLength` (hard per-buffer limit), and total planned
memory with `recommendedMaxWorkingSetSize` (advisory). Preflight warns and
continues above the recommendation; only exceeding the hard limit fails.
It includes rollout rings/workspaces, initialization scratch and eligible
packed weights, but cannot predict all driver-private allocation.

The historical FP32 NetHack arena requested 19,160,891,392 bytes against a
14,302,248,960-byte per-buffer limit, with only 0.205 GB already live.
Recommended working set was 19.070 GB. Dominant allocations were rollout
observations 10.03 GB, train observations 5.01 GB and masks 1.56+0.78 GB.
This is capacity, not a leak. Reduce agents/horizon or use sync (one slot,
~11.6 GB in that configuration); BF16 reduces storage. Separately allocated
no-copy buffers cannot transparently replace one contiguous GPU-address arena.

A 24 GB M4 g2048 FP32 check planned 14.098 GB total and a 13.736 GB largest
buffer, below its 19.065 GB recommendation and 14.302 GB hard limit, and
completed 262,144 steps plus checkpoint. This does not contradict the smaller
16 GiB device's budget limit. Always use the current device's reported limits.
