# Cleanup sprint leg 3

2026-10-27; branch `metal-cleanup3`, starting revision
`9cceb5c37ad833d33964011af746de53fafe767c`; upstream `ea7f9893`.
Validation host: Mac mini M4, 16 GB, macOS 27.0.1 (26A434), Xcode 27.1
(27A9275), Bash 3.2.57. The changes and this report are committed together with
the requested session trailer. No history rewrite or push was performed.

No kernel arithmetic, dispatch default, training configuration, fixed seed
expectation or external golden fixture was changed. Production changes are
Metal-only unsupported-configuration guards, removal of their unreachable
shader alternatives, an opt-in dispatch diagnostic, and the review follow-up's
strict-feedback diagnostic counter. No tracked files were deleted. Existing
owner decisions concerning asset rights, packaging and history cleanup remain
outside this leg.

## Verification

- Review follow-up schedule gate: **PASS** in BF16 and FP32, including exact
  checkpoint/score equality, default/lazy/spin strict-join counts of zero,
  strict-mode count **485.5/epoch** in both precisions, and rejection of the
  compiled skipped-join mutation. Logs/results
  `build/metal/alternate-schedule.rjPgb2/`. Direct Protein gate: **PASS**,
  including all **1,464** bitwise Sobol/Philox/glibc fixtures.
- Review follow-up full suite: **57/57 PASS**, zero failures/skips, **1849 s
  (30m 49s)**. Per-gate logs `build/metal/gates/run.BGkoPc/`; the final
  `optouts_schedule` gate includes the successful skipped-join mutation check.
  Root `./puffer` remained absent. This is the final validation for the review
  fixes described below.
- Initial unmodified runner: **50/50 PASS**, zero failures/skips, **1725.84 s
  (28m 45.84s)**. Summary `build/cleanup-leg3/baseline.log`; per-gate logs
  `build/metal/gates/run.tFCtvq/`. Existing production sources and the runner were not
  edited until this run finished.
- New alternate/configuration suite: **6/6 PASS**, zero failures/skips,
  **110.35 s (1m 50.35s)**. Summary `build/cleanup-leg3/alternate-final.log`;
  per-gate logs/results `build/metal/gates/run.xk12Fk/`.
- Optional mutation suite: **4/4 PASS**, zero failures/skips, **76.18 s
  (1m 16.18s)**. Summary `build/cleanup-leg3/mutation.log`; per-gate logs
  `build/metal/gates/run.IH5EpK/`.
- Final full suite: **57/57 PASS**, zero failures/skips, **1837.16 s
  (30m 37.16s)**. Summary `build/cleanup-leg3/full-final.log`; per-gate logs
  `build/metal/gates/run.rHVf0U/`. All existing NetHack gates and all six new
  alternate/configuration gates passed. Root `./puffer` stayed absent through
  the entire run. The final fused-dW score/loss deltas exactly repeated the
  alternate-only observations below.
- Dependency preflight: **57 entries**, no missing fixtures or invalid metadata;
  `build/cleanup-leg3/deps.log`. Changed/new scripts pass Bash 3.2 syntax checks.
  Runner lists 57 default, six alternate-only and four optional mutation gates.

All 64 logged score/hash records across the existing H64, overlap, wide and
large end-to-end tests match the initial baseline: 20 score/hash-prefix records
from H64/overlap and 44 score/full-SHA-256 records from wide/large.

The first alternate run correctly failed for two reasons: the new IEEE build-log
assertion used the kernel tag `rollout_fwd` in place of source filename
`rollout_forward`, and the exploratory 16-horizon fused-dW comparison exceeded
an initial one-point score budget. The filename assertion was corrected; the
fused-dW gate now explicitly tests early training with score **and loss** bounds,
as explained below. The failed run is retained in
`build/cleanup-leg3/alternate.log` / `build/metal/gates/run.wSDXJe/`; it is not
counted as passing validation.

## Runner and root build output

The initial baseline reproduced the stray root `./puffer`. Runner-selected
`run_test_shared_env.sh`, `run_test_obs_layout.sh`, `run_test_pack_order.sh` and
`run_test_rollout_gate.sh` omitted the output argument to `build.sh`; the nested
`run_test_rollout_group_order.sh` did likewise when used standalone. All five
now create isolated trainer paths, pass them explicitly, and remove their own
trainer on exit. The seed helper already changes to an isolated binary directory
before invoking `./puffer` and needed no change. The preexisting stray output
and the one reproduced by the baseline were deleted.

Every executed gate now snapshots root `puffer` before/after: presence, inode,
mode, size, modification/change times and SHA-256, including symlink targets.
A changed root output fails the gate even if its command succeeds. The new
host-only `root_output_guard` gate tests absent/present files, writes, deletion,
identical-byte replacement, dangling links and target changes, and executes the
actual runner function against disposable paths to verify failure accounting.
It is the seventh addition to the default inventory.

## New alternate/configuration gates

All six run by default; `--suite alternate` selects just these gates.

| Gate | Assertion |
|---|---|
| `sigmoid_ieee` | Builds with `PUF_SIG_IEEE_DIV`; all 2^32 FP32 input patterns in each of five sigmoid-bearing translation units, plus the sampled CPU rounding/FTZ oracle. Existing portable <=1 ULP tolerance; observed zero mismatches and max ULP 0 throughout. |
| `ieee_seed` | Builds ordinary Cartpole trainers in BF16 and FP32 with default versus IEEE shaders; seed-1 checkpoints and scores match exactly. A test-only `xcrun` shim injects the define into every shader compile and verifies both compilations of all five relevant source files. `NVCC_EXTRA` alone would only affect the host. |
| `strict_launch` | Ten compile/link-only standard trainer builds: Breakout, Pong, NetHack, continuous Drone, and NMMO3's custom encoder, each BF16/FP32, with `PUF_METAL_STRICT_LAUNCH`. No training launches. |
| `optouts_math` | Process-isolated small NS, MMA, batching, wide NS and fused-dW opt-outs, with positive default dispatch and alternate-path evidence. |
| `optouts_schedule` | Process-isolated strict feedback, lazy-progress opt-out and 50 us spin, using existing runtime counters; KPROF is deliberately off because profiling itself disables gating. |
| `nethack_variants` | Default versus explicit K=16 preprocessing equality; actual native builds and standalone shader preprocessing reject the three unsupported switches and K=0/32 with specific diagnostics. |

General seed geometry: Cartpole, seed 1, 524,288 transitions / 16 horizons,
1,024 agents, four buffers, four environment threads, two layers, T=32,
minibatch 8,192, 128 evaluation episodes and agents. H32 covers small NS;
H64 covers wide NS. Both BF16 and FP32 run. Fused dW uses the same geometry
with 131,072 transitions / four horizons. Existing
`PUF_TEST_NO_INTERMEDIATE_LOG=1` makes loss averaging independent of elapsed-time
log cadence. New processes clear inherited internal Metal/test overrides.

### Exact versus tolerance findings

Every row was run in both precisions. “Exact” means full checkpoint SHA-256
and evaluated score equality between the default and the opt-out process.

| Process override | Actual alternate-path evidence | Result |
|---|---|---|
| `PUF_METAL_FUSED_NS=0` | `muon_ns5_batch` and `muon_ns5` disappear; `muon_wide_sumsq` appears (small matrices move to the wide dispatcher) | Exact |
| `PUF_METAL_MUON_NS_MMA=0` | Default has actual small-NS launch arguments `mma=1`; opt-out logs only `mma=0` while small NS still dispatches | Exact |
| `PUF_METAL_MUON_BATCH=0` | `muon_ns5_batch` disappears and individual `muon_ns5` dispatches appear | Exact |
| `PUF_METAL_MUON_WIDE=0` | At H64, wide Muon pipelines disappear and generic GEMM dispatches remain | Exact |
| `PUF_METAL_FUSED_DW=0` | `mingru_fused_backward_dw` disappears; plain `mingru_fused_backward` dispatches | Non-bitwise; bounded four-horizon score/loss comparison below |
| `PUF_METAL_STRICT_FEEDBACK=1` | Default gated commits >0 and strict-feedback joins =0; opt-out has zero gated commits and positive joins, with no profiler or other gate-disabling override | Exact |
| `PUF_METAL_LAZY_PROGRESS=0` | Default gated commits >0; opt-out has zero | Exact |
| `PUF_METAL_ROLLOUT_SPIN_US=50` | Default spin hit/miss counters are both zero; nonzero spin counts in opt-out | Exact |

The small NS scalar/MMA implementations share a pipeline name. To distinguish
them, `PUF_METAL_GEMM_LOG` now also prints each distinct actual R/C/mma dispatch
argument once, after eligibility checks. This is opt-in logging only; no
pipeline, flag parsing, argument value or arithmetic was changed.

Representative alternate-suite H32 checkpoint prefixes (default, IEEE, three
small-NS overrides and all scheduling overrides): BF16
`da4a556ea882a6ba`, FP32 `c39766415b532f50`; scores
75.82080841064453 and 45.284671783447266. H64 wide-NS pairs:
BF16 `11516e8e86fc16d8`, score 84.15068817138672; FP32
`fc84f1090fd9999b1`, score 60.315067291259766. These are observed results,
not new fixed golden expectations. All full SHA-256 values and counters are
in per-precision `*-results.json` files under the alternate run directory.

The scheduling reference had 43.4 gated commits per epoch in both precisions;
strict/lazy opt-outs had zero. A new process-wide relaxed atomic counter records
only calls to `join_all_feedback()` caused by `PUF_METAL_STRICT_FEEDBACK`, not
ordinary strict API operations. It remains zero in default, lazy and spin runs
and is required positive in strict mode, proving the flag reaches the feedback
join rather than merely disabling the rollout gate. A test-only compiled
mutation skips those strict-mode joins while leaving gating disabled; the
positive-counter assertion rejects it after successful reference and mutant
training runs. The 50 us runs had positive spin hits and misses (BF16 2.6/42.2;
FP32 3.267/41.53 per epoch in the alternate-only run). Spin counts depend on
timing, so the gate requires a positive combined count, not these exact values.

### Fused dW scope and limits

`test_mingru.mm` documents that H32 fused dW emits one partial per 64 rows and
reduces in tile order, unlike the materialized GEMM; its existing local FP32
norm-error bound is 2e-5. BF16 also rounds the differently accumulated result.
It is therefore not an exact checkpoint contract, despite the audit's shorthand
“exact-comparison fallback.” The new gate requires, after four horizons:

- absolute evaluated score difference <= **0.25**;
- absolute displayed total-loss difference <= **0.001 + 0.005 × abs(reference
  loss)** (0.5% plus the trainer's three-decimal display resolution);
- positive fused dispatches in the reference, zero in the opt-out, and positive
  plain backward dispatches in the opt-out.

| Precision | Default / opt-out score | Score delta | Default / opt-out displayed loss | Loss delta | Checkpoint equality |
|---|---|---|---|---|---|
| BF16 | 9.465517044067383 / 9.504347801208496 | 0.03883075714111328 | 14.627 / 14.627 | 0 | No |
| FP32 | 9.571005821228027 / 9.571005821228027 | 0 | 14.951 / 14.922 | 0.029 | No |

This is an early-training numerical regression, **not** evidence of matching
long-training trajectories. In the exploratory 16-horizon run, the default /
opt-out scores were 75.820808 / 79.404762 (BF16) and 45.284672 / 52.601124
(FP32), deltas 3.583954 and 7.316452. Relative L2 checkpoint differences were
0.07160 and 0.05887. Four-horizon relative L2 differences were 0.003462 and
0.001832. Probe logs are under `build/cleanup-leg3/dw-probe/` and
`build/cleanup-leg3/dw-short-{bf16,f32}.log`. No arithmetic was changed to force
agreement, and the longer-run differences were not accepted under a widened
long-run score budget.

## Optional mutation suite

`--suite mutation` adds four runner entries and does not add these standalone
negative controls to the default duration:

- `sigmoid_mutations`: clean quick pass, then `PUF_SIG_MUTATE=1/2` must fail
  exact mode and mutation 3 must fail the portable one-ULP limit.
- `proj_store_mutation`: clean quick pass, then `PST_MUTATE_REF` must produce
  a bit-mismatch rejection.
- `goldwide_mutations`: fresh builds, then the existing `mutate_goldwide.py`
  three synthetic reader mutations and restored reference controls.
- `protein_mutations`: existing isolated-source `mutate_protein.py`, followed
  by the clean Protein tests (all nine mutations caught).

The shell wrappers require both exit 1 and the numerical failure diagnostic;
a compilation or unrelated tool failure is not a caught sigmoid/projection
mutation. Existing wide/large and nested mutation checks remain in the default
suite unchanged.

## Metal-only NetHack configuration

`src/metal/nethack_config.h` is included by the Metal trainer, adapter and
encoder shader. It emits a clear `#error` for `NH_MSG_V4096`, `NH_MSG128`,
`NH_NO_APANEL`, and any defined `NETHACK_V3_K != 16`. CUDA sources and their
variants are untouched. Only now-unreachable alternatives/conditionals were
removed from the Metal shader header/source; supported default operations
remain identical.

`xcrun -sdk macosx metal -std=metal4.0 -E -P -Isrc/metal/shaders` output was
compared with `cmp` before/after, with and without `-DPRECISION_FLOAT`.
Both NetHack translation units are byte-identical in both precisions:

| Translation unit | Precision | SHA-256, before and after |
|---|---|---|
| `envs_nethack.metal` | BF16 | `42555e83e7e33149fadd3b054f46bd7f1ec9957b431eec745f9a93c253875fc5` |
| `envs_nethack.metal` | FP32 | `f19ddf0b1203fd37fa45c230c4a9dc28b20862e3aa6729b84a5ea9d2995e3772` |
| `envs_nethack_policy.metal` | BF16 | `4eb1e889185c085e163b4ecc52921d7438d0688787a07f431b565a0fb516ac07` |
| `envs_nethack_policy.metal` | FP32 | `37f26f6df8aa9d5e23b84178a6bd5dcb0bbb281979b39f92a23868f7a9a7add6` |

Encoder output was captured before editing. The unchanged policy TU was also
compared against the starting Git version in a disposable include directory.
The permanent gate checks explicit K=16 versus unset and rejects unsupported
host/shader configurations; hashes above record this Xcode's preprocessing,
not a cross-toolchain hash requirement. Captures remain under
`build/cleanup-leg3/nethack-*.ii`.

## Third-party inventory and disk hygiene

[THIRD_PARTY.md](../../THIRD_PARTY.md), linked from README and NOTICE, covers all
vendor entries, every resources top-level set, embedded font notices, identified
derived Ocean code, fork Metal adaptations, and downloaded build dependencies.
It records file declarations and their notice locations, not a legal or
redistribution-rights determination. Unknown origins/licenses are explicitly
marked “not stated in repository.” Font metadata actually declares OFL 1.1
for all seven bundled fonts. Collections-C headers declare LGPL v3-or-later;
the dlmalloc header separately declares public domain. Existing notices and
assets were not changed.

Historical provenance finding (superseded by the public-table change below):
follow-up review found and added the omitted 2,640,000-byte
`src/metal/protein_sobol32.bin` inventory entry. It consists of 20,000
little-endian rows, each holding 32 direction words exported from cuRAND 12.8's
`CURAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6` set and one word from
`curandGetScrambleConstants32`. The public Joe-Kuo
`new-joe-kuo-6.21201` file (SHA-256
`68eedd2a4e3b659b9695e7aff0f8ac68718bcf620730fc3d3a8c65df2a067441`) was
downloaded under `build/cleanup-leg3-review/sobol/` via an archived capture of
the requested UNSW URL after the live URL returned 404. Its standard 32-bit
expansion, including the conventional dimension-1 row, exactly reproduces the
publisher's displayed first ten points in three dimensions.

That plain expansion does **not** reproduce the binary's scrambled vectors:
0/20,000 rows match the same dimension and zero match any of the 21,201 public
dimensions. Reversed word order, per-word byte swapping, and XOR by the stored
row constant (for the corresponding dimension) also yield zero matches. The
binary's SHA-256 is
`fce775e7962beda83fa73c64c628a0323a94eadf5d32bad97d1dc6159e52e5f0`.
The public file establishes the underlying Joe-Kuo6 direction-number family,
but cuRAND's exported scrambled-vector transform/data and all scramble constants
are not byte-reproducible from it; all 660,000 words were therefore identified
as NVIDIA-exported data.

The constants are used, not padding. Metal `protein_sobol()` initializes each
dimension from row word 33 before XORing selected direction words. Upstream
Protein contains no direct `curandGetScrambleConstants32` call, but creates a
`CURAND_RNG_QUASI_SCRAMBLED_SOBOL32` host generator, so cuRAND owns the
scrambled direction/initial-constant setup internally. That review preserved the existing binary because replacing only the constants
could not reproduce cuRAND's transformed vectors.

The owner subsequently approved removing that table from the tracked tree.
It now lives only in the external golden set as `protein/curand_sobol32.bin`;
its recorded hash remains for exact-mode validation. The default embeds
394,264 bytes of compact public Joe--Kuo construction parameters covering
Protein's existing 20,000-dimension capacity, with real fixed-seed SplitMix64
XOR shifts. Suggestions therefore change in default mode; explicit external
cuRAND mode preserves the original bitwise gates. The generator, license,
validation and final gate counts are documented in
[protein-sobol-report.md](protein-sobol-report.md).

The same follow-up corrected the chess source to the Wikimedia Commons URL
already present in `ocean/chess/chess.h`. Rechecking every other “not stated in
repository” entry against its cited files also recovered dlmalloc's project
description URL, the simplex sine/cosine discussion, Drive map conversion from
the preprocessed WOMD dataset, the included Impulse Wars texture generator and
shader notices/coefficient source, and Tower Climb's two GLSL source links.
Statements that remain unknown were narrowed to the specific assets or code for
which the cited files still provide no source or separate license.

Downloaded dependencies include the pinned raylib, Box2D, fast-nle and OSRS
archive, plus fast-nle's pybind11, deboost.context, bzip2, libtmt and converter.
The inventory explicitly distinguishes transitive `GIT_HASH` declarations from
verified effective revision pins (both inspected checkout HEADs differ from
the declared hashes); fixing those upstream build declarations or
resolving missing asset rights is not part of this inventory-only task.

Free space was checked before long runs: initially 53 GiB available on the
Data volume, approximately 52 GiB during the follow-up runs, and 49 GiB at
completion. Completed baseline and final replay/checkpoint payloads totaling
about 2.77 GiB were removed while preserving logs/manifests (each run: 839.8 MiB
of goldwide payloads plus 579.4 MiB of checkpoints). Exploratory dW trainers and
checkpoints were also removed. New seeded gates remove checkpoints after
hashing and trainer binaries after successful comparisons. No tracked artifact
or external golden file was removed.
