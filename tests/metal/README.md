# Metal regression suites

Run from any directory on Apple Silicon with macOS 26+, full Xcode 26 and its
Metal toolchain, Homebrew `libomp`, and Python >=3.9 with NumPy on `PATH`.
The full suite also builds NetHack (Git, CMake, make, flex, bison, m4) and OSRS
(Python, curl, tar). First builds download Raylib, fast-nle and OSRS assets;
see the environment READMEs for setup and asset provenance limitations.

```sh
export PUFFER_GOLDEN_DIR=/path/to/golden
# Optional: select a non-default full Xcode installation.
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
tests/metal/run_all.sh --list
tests/metal/run_all.sh --check-deps
tests/metal/run_all.sh
```

`PUFFER_GOLDEN_DIR` defaults to **<repo>/golden**. `DEVELOPER_DIR` defaults to
`/Applications/Xcode.app/Contents/Developer` only when unset. Scripts preserve
a caller-selected toolchain. The runner creates `build/metal/gates/run.XXXXXX/`
for each invocation's logs and seed-helper artifacts; goldwide uses a fresh
subdirectory there, so reruns do not require deleting replay output. Other
harness build directories are reused: run one aggregate suite at a time in a
worktree. The guide smoke uses isolated binaries under `build/metal/guide/`.

The authoritative `--list` has **59 default gates**:

- Host-only package/fetch controls (determinism, privacy, checksum, partial archive,
  contract validation, safe extraction, download and destination preservation).
- Host-only malformed-fixture controls (empty inventories, missing scalar/schema,
  empty tensor maps, zero dimensions and truncated files).
- Runtime, launch/stream waits, shared environment transport, observation layout,
  packing and rollout gating/recovery.
- Selected FP32/BF16 kernels, GEMM, MinGRU, PPO/Muon, custom encoders and CUDA
  train replay comparisons, including recorded tolerances and noise controls.
- H64 and overlap end-to-end seed checks: scores and SHA-256 prefixes copied
  unchanged from the original H64/overlap seed helpers. Wide/large end-to-end gates
  require exact score/checkpoint agreement between generic, default and opt-out
  paths. H256 always uses tracked Breakout with explicit H256/L2/T64 geometry;
  it has no stored fixed reference to regenerate. H128 uses Freeway.
- Projection storage, exhaustive M4 exact sigmoid comparison, mutation controls,
  Protein/sweep CLI, eval label, build help, memory preflight, warm start,
  checkpoint layouts, guide commands, log history and robot-arm trajectories.

These are selected configurations, not all-environment convergence or exhaustive
cross-device equivalence. The sigmoid gate requests bit equality on the validated
M4 toolchain; standalone portable mode has a different documented tolerance.
Benchmarks, environment smoke sweeps and long runtime stress remain standalone.
The additional negative controls are opt-in and are not included in the default
count. The existing wide/large and nested mutation gates remain in the default
suite.

## Alternate paths and optional mutation suite

```sh
tests/metal/run_all.sh --suite alternate --list
tests/metal/run_all.sh --suite alternate
tests/metal/run_all.sh --suite mutation --list
tests/metal/run_all.sh --suite mutation
```

`--suite alternate` selects the six new alternate/configuration gates below;
all six also run by default. `--suite mutation` selects four additional negative
control gates, with no external golden dependency. `--list`, `--check-deps` and
missing-fixture behavior apply to the selected suite.

| Gate | Required evidence |
|---|---|
| `sigmoid_ieee` | Builds every sigmoid-bearing shader with `PUF_SIG_IEEE_DIV`; exhaustive device IEEE comparison and sampled rounding/FTZ-aware CPU oracle with the documented <=1 ULP budget |
| `ieee_seed` | Standard Cartpole trainer, BF16 and FP32, with all shader compilations explicitly using IEEE division; short seed-1 checkpoint and score equality against the default build on the validated M4 toolchain |
| `strict_launch` | Compile/link only, `PUF_METAL_STRICT_LAUNCH`, both precisions: Breakout, Pong, NetHack, continuous Drone and custom-encoder NMMO3 |
| `optouts_math` | Fresh processes for small NS fusion, small NS MMA, small NS batching, wide NS and H32 fused dW opt-outs; actual kernel profiles and small-NS dispatch arguments under `PUF_METAL_GEMM_LOG` |
| `optouts_schedule` | Fresh processes with strict feedback, eager progress words and 50 us rollout spin; existing gated-commit/spin counters, with KPROF off because it disables gating |
| `nethack_variants` | Explicit K=16 equals default shader preprocessing; real native builds and shader preprocessing reject all unsupported variants with specific diagnostics |

Seed comparisons use Cartpole, seed 1, 524,288 transitions (16 horizons),
1,024 agents / four buffers, two layers, T=32, minibatch 8,192, 128 evaluation
episodes/agents, and both storage precisions. Math runs use H32, with a separate
H64 run to exercise wide Muon. The four Muon opt-outs and all three scheduling
controls require exact SHA-256 and score equality. H32 fused dW has a different
summation order: the existing kernel gate bounds its FP32 local norm error at
2e-5, while a separate four-horizon
(131,072 transition) early-training comparison requires score difference <=0.25
and displayed total-loss difference <=0.001 + 0.005 * abs(reference loss).
The absolute loss term accounts for the trainer's three-decimal display.
Checkpoint hashes are recorded but are not expected to match for this path.
All new seed runs use the existing `PUF_TEST_NO_INTERMEDIATE_LOG=1` control to
make loss averaging independent of wall-clock logging cadence. These are
selected finite-run checks, not long-training equivalence claims. The observed four-horizon results were:

| Precision | Default / opt-out score | Score delta | Default / opt-out loss | Loss delta |
|---|---|---|---|---|
| BF16 | 9.465517044067383 / 9.504347801208496 | 0.03883075714111328 | 14.627 / 14.627 | 0 |
| FP32 | 9.571005821228027 / 9.571005821228027 | 0 | 14.951 / 14.922 | 0.029 |

Neither checkpoint pair is bit-identical. An exploratory 16-horizon comparison
had score deltas 3.583954 (BF16) and 7.316452 (FP32), with relative L2 weight
differences 0.07160 / 0.05887; four-horizon differences were 0.003462 / 0.001832.
The shorter gate tests early numerical behavior; it does not establish long-run
agreement or accept the longer run under a widened tolerance.

Dispatch evidence distinguishes each alternate from a silent fallback:

| Override | Required path evidence |
|---|---|
| `PUF_METAL_FUSED_NS=0` | Small fused NS disappears; wide dispatcher appears |
| `PUF_METAL_MUON_NS_MMA=0` | Actual small-NS arguments change from mma=1 to mma=0 |
| `PUF_METAL_MUON_BATCH=0` | Individual NS replaces batched NS |
| `PUF_METAL_MUON_WIDE=0` | Generic GEMM replaces wide Muon at H64 |
| `PUF_METAL_FUSED_DW=0` | Plain backward replaces fused backward/dW |
| `PUF_METAL_STRICT_FEEDBACK=1` | Zero gated commits and positive strict-feedback joins; skipping joins fails a compiled mutation |
| `PUF_METAL_LAZY_PROGRESS=0` | Zero gated commits versus positive default count |
| `PUF_METAL_ROLLOUT_SPIN_US=50` | Positive combined spin hit/miss count versus zero default |

These counters prove selection; timing-dependent counts are not fixed golden
values. Strict-join counts exclude ordinary strict API operations.

The IEEE seed build uses a test-only `xcrun` shim to add the define to every
shader compile, and verifies the build log for all five sigmoid-bearing shader
files in both precisions. `NVCC_EXTRA` alone only affects host compilation and
would not exercise the shader fallback. No production build default changes.

Optional mutation gates run `PUF_SIG_MUTATE` variants 1/2 in exact mode and
variant 3 (two ULP at signed zero) in portable mode, `PST_MUTATE_REF`,
`mutate_goldwide.py` and `mutate_protein.py`. The sigmoid/projection wrappers
first require an unmutated pass, then require exit 1 and a numerical failure
diagnostic, so a compile failure cannot masquerade as a caught mutation.
Goldwide builds fresh synthetic reference-control readers; its existing mutation
script checks rejection diagnostics and restored controls. Protein uses isolated
source copies and reruns the clean tests.

Every gate fingerprints root `./puffer` before and after execution (existence,
identity, metadata and SHA-256, including symlink targets). Creation, removal or
modification fails that gate even if its command succeeds. The new host-only
`root_output_guard` gate tests this runner behavior using disposable paths.
Shared-env, observation-layout, pack-order, rollout-gate and standalone grouped
rollout-order helpers build isolated trainers and remove them on exit. New
short-seed gates delete checkpoints after recording hashes, and remove their
trainer binaries after successful checks; logs/results remain under the run
directory. Check `df -h /System/Volumes/Data` before repeated full runs and
remove completed generated tensor/checkpoint outputs when no longer needed.

## Missing fixtures and dependency checks

Thirteen gates require external captures: twelve numerical comparison gates and
the exact-cuRAND Protein gate. `--check-deps` performs no GPU work;
it checks tools, every required case manifest, and tensor-file sizes.
`fixture_cases.json` records 678 required case manifests plus the private
Protein binary and two wide audit records. `fixture_contracts.json`
pins SHA-256 of canonical JSON metadata (sorted keys, compact separators) from
the existing bundle, plus the 57 wide report files and 3,645-file SHA inventory
count. These are metadata contracts, not regenerated numerical expectations.
They pin every required scalar, tensor name, shape, dtype, role, precision and
schema field; JSON formatting/key order may differ. Shapes must be nonempty
and strictly positive. The wide SHA file list is reconstructed from validated
manifests and the pinned report list and must match exactly; its per-case
inventory must agree too. A deliberate capture metadata change requires review
of these contracts, never automatic regeneration during a test run.

Missing captures print **SKIP(missing fixture)** and
make either a normal run or dependency check exit nonzero. Explicit
`--allow-missing-fixtures` permits skips (never counts them as PASS); real test
failures still exit nonzero. Invalid manifests are errors. The wide suite also
verifies recorded hashes; normal fixture comparisons retain their own numeric
and structural validation. Dependency checks do not claim GPU execution or
verify every tensor's content/hash. `--list` needs neither fixtures nor Xcode.

## External golden bundle

The bundle is published as the
[`goldens-v1` release](https://github.com/manthedan/pufferlib-metal/releases/tag/goldens-v1).
GitHub caps release assets at 2 GiB, so the archive is split into two parts;
`golden_bundle.json` lists them and pins the combined archive SHA256, size and
public fixture contract. The fetcher downloads the parts in order into one
resumable partial file and verifies the combined archive before extracting.
The verified archive contains 10,059 files (678 manifests, 9,323 tensors and
58 wide audit files), totaling 5,294,267,886 uncompressed bytes. The archive is
3,190,932,528 bytes, SHA256
`90c6375dab77e4634868ad780474068768ce610a92e8550e1f376caa50cae1e5`.
The original package validation compared every installed file: 10,030 unchanged
and 29 sanitized (28 logs plus their SHA inventory), with no tensor or manifest
changes. There were 276 allowed prefix replacements; no actionable privacy
matches remained. The source capture was unchanged.

| Original bundle validation | Result | Time |
|---|---|---:|
| Package | 10,059 files | 531.550 s |
| Local fetch with pinned metadata | PASS | 15.296 s |
| Every installed file vs original/provenance | PASS | 9.456 s |
| Full default suite against public bundle | 59 total: 58 PASS, 0 FAIL, 1 SKIP (`protein`), exit 0 | 1,891.478 s |

The private cuRAND export is intentionally absent; public Protein runs. These
are recorded validation results, not a replacement for checking a new download.

```sh
# Defaults to <repo>/golden, just like run_all.sh.
tools/metal/fetch_goldens.sh
# Download a single-file copy from elsewhere (the pinned checksum still applies).
tools/metal/fetch_goldens.sh --url https://example.org/metal-goldens-v1.tar.gz
# Offline use; use the tracked metadata, or an explicitly trusted local sidecar.
tools/metal/fetch_goldens.sh --file /path/to/metal-goldens-v1.tar.gz \
  --metadata /path/to/metal-goldens-v1.tar.gz.json
PUFFER_GOLDEN_DIR=/path/to/golden tools/metal/fetch_goldens.sh --file /path/to/metal-goldens-v1.tar.gz
```

Package/fetch require Bash 3.2+ and Python 3.9+ (standard library only); URL
fetches also require `curl`. The format is USTAR tar with gzip compression,
extractable with stock macOS/Linux `tar -xzf`. Python `hashlib` supplies SHA256
on both platforms, without depending on `shasum` versus `sha256sum`.
Use the fetcher for checksum and contract verification before installation.
`--metadata` is a trust input: only use a sidecar obtained from a trusted source.

Downloads use `curl -fL`, retries, and resume from a sibling `.part` file named
with the expected archive digest; a failed download is retained for retry.
If the server cannot resume, or a cached download fails verification, remove
that named `.part` file to start again. The fetcher verifies archive size and
SHA256 before extracting into a temporary sibling directory. It rejects unsafe
paths, links, unexpected files, and the private cuRAND export, then runs
`check_fixtures.py` and verifies the wide file hashes. Only the intentionally
excluded cuRAND file may be missing. Installation renames the verified staging
directory into place. Concurrent fetchers are blocked by a sibling install lock;
after a terminated process, check that no installer is running before removing
a stale lock. `--force` moves an existing destination to a unique `.backup-*`
sibling; it never deletes the previous fixtures. Failed validation leaves the
existing destination untouched. Keep enough free space for the download,
extracted bundle, and any preserved backup.

Existing private golden directories no longer satisfy the public wide-report
contract. To migrate once, install the verified public bundle at the desired
destination with `--force`:

```sh
tools/metal/fetch_goldens.sh --file /path/to/metal-goldens-v1.tar.gz \
  --destination /path/to/golden --force
```

The old directory is moved intact to `/path/to/golden.backup-<unique-id>` and its
location is printed. Nothing in the old directory is deleted, including private
fixtures excluded from the public bundle. The tracked provenance hashes remain
available to verify its original logs. No migration of the source capture was
performed during bundle validation.

Without external captures the thirteen dependent default gates print
`SKIP(missing fixture)`; `--allow-missing-fixtures` explicitly permits those
skips. A public bundle intentionally leaves only `protein` skipped in the
default suite (`protein_mutations` is also skipped in the optional mutation
suite). The public Protein gate runs without the private export.

To build a candidate bundle locally:

```sh
PUF_BUNDLE_PRIVATE_TOKENS='private-user,private-host' \
  tools/metal/package_goldens.sh --source /path/to/private/golden \
  --output /path/outside/golden/metal-goldens-v1.tar.gz
```

Private tokens can instead (or additionally) come from an untracked comma- or
newline-separated file passed with `--private-tokens FILE`. Packaging prints a
prominent warning when neither source supplies any tokens. Generic macOS/Linux home-directory paths and email addresses are always scanned.

The packager derives the exact file set from gate groups, `fixture_cases.json`,
validated manifests and `fixture_contracts.json`, including all required wide
SHA inventory entries. It reports and excludes unused files. Files are sorted;
USTAR headers use mode 0644, zero timestamps/owners and empty owner names;
gzip uses level 9, timestamp zero and no filename. Repeated packaging with the
same Python/zlib implementation and input bytes produces identical archives;
different compressor versions should be checked before changing a release pin.
A small `.tar.gz.json` sidecar records format version, archive SHA256/size,
file count, uncompressed size and the contract-file SHA256. After successful
packaging, review and copy this metadata to `tests/metal/golden_bundle.json`.
To publish, `python3 tools/metal/golden_bundle.py split --archive <archive>
--base-url <release download URL>` writes `.part-NN` files under 2 GiB and
prints the `parts` list for the metadata; upload the parts as release assets.
These tools never publish.

The `.tar.gz.scan.json` audit lists included/excluded files, original matches,
remaining matches, and rewritten files. UTF-8 metadata, logs and source text
are scanned; matches inside 64-character hex digests are reported as false
positives. The only permitted sanitization is a CUDA reference checkout matching
the recorded Linux home-directory checkout prefix in `wide_reports/*.log`,
replaced with `<cuda-ref>/`. The exact regex and original/sanitized SHA256
for every rewritten file are tracked in
[`golden_sanitization.json`](golden_sanitization.json). These logs are capture
provenance; gates use them only for SHA inventory checking.

The packager changes only those prefix bytes and the corresponding digest bytes
in the wide SHA inventory, preserving all other bytes, ordering and formatting.
`inventory.json` has no log entries and remains byte-identical. The public
contract pins that unchanged inventory, the regenerated `SHA256.json`, and all
28 sanitized logs. Any other personal-data match stops packaging, including
matches in unprotected text. Source fixtures are never edited. Packaging accepts
either the recorded private originals or the already sanitized public files.
The NVIDIA cuRAND-exported `protein/curand_sobol32.bin` is always excluded and
must not be redistributed.
Generate it privately on a CUDA host using
[the Protein exporter](../metal_ref/README.md#external-curand-protein-fixture):

```sh
PUFFER_GOLDEN_DIR=/path/to/private/golden bash tests/metal_ref/export_protein_sobol.sh
```

Copy that private result to the same relative path on the test machine only if
you need exact-cuRAND gates. The exporter requires cuRAND 12.8 and `nvcc`; its
host-table export wrapper has not been tested on CUDA here.

The public bundle layout below is relative to `$PUFFER_GOLDEN_DIR`:

| Path | Capture consumed |
|---|---|
| `run_a/` | FP32 operation cases and native `train_epoch3/` input |
| `replay_a/train_epoch3/` | FP32 CUDA replay output |
| `run_bf16/`, `run_bf16_b/` | Independent BF16 operation cases; first includes train input |
| `replay_bf16/train_epoch3/` | BF16 CUDA replay output |
| `envs/run_a/` | FP32 custom-encoder cases, including NetHack |
| `envs_bf16/run_a/`, `envs_bf16/run_b/` | Independent BF16 custom-encoder captures |
| `wide_a/`, `wide_b/`, `wide_bf16/`, `wide_bf16_b/` | Wide operation/policy cases plus `train_{freeway,tetris}/train_epoch3/` and `replay_{freeway,tetris}/train_epoch3/` |
| `wide_reports/inventory.json`, `wide_reports/SHA256.json` | Required wide audit records (retain every additional file referenced by the hash inventory) |

Each case has `manifest.json` and the raw tensor files named in its `tensors`
map, including shapes, dtypes and roles. Private captures can also include independent
FP32 `run_b`, replay repeats and extra provenance; the public packager leaves
these out unless required by the contracts and wide hash inventory.
Robot-arm trajectory binaries and Protein's numeric text fixture are tracked
inside the repository and do not require this external bundle.

## Capture provenance and regeneration

The original fixtures were generated on **a Linux x86-64 host with an RTX 3090
(sm_86)**, CUDA 12.8, from upstream `ea7f9893` plus instrumentation hooks.
The harness calls real upstream operations and captures native epoch-3 training,
then repeats generation and replay in fresh processes to check determinism.
No Python replacement math generates the expected CUDA outputs. Manifests and
SHA inventories preserve the captured values; do not replace expected seed
scores or checkpoint hashes with output from the implementation under test.

The tracked [FP32 reference harness](../metal_ref/README.md) documents generation
with explicit `CUDA_HOME`/`NCCL_HOME`; no private sibling setup is required.
[BF16 encoder provenance](../metal_ref/README.md#cuda-bf16-custom-encoder-references) records historical
captures and their generation commands. Those branch-specific BF16,
encoder and wide generation helpers are not all shipped here: full regeneration of every
public bundle case is not yet available from this checkout alone. Packaging
existing captures does not remove that limitation.
[Wide fixture documentation](#wide-cuda-fixtures) describes the geometry, tolerances,
independent controls and hash audit. [Robot-arm fixture documentation](fixtures/robot_arm/README.md)
records the tracked trajectory capture procedure and precision variants.

## Wide CUDA fixtures

From the repository root, using the suite prerequisites above:

```bash
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
# NumPy is required by compare_train_replay.py.
export PUFFER_GOLDEN_DIR="${PUFFER_GOLDEN_DIR:-$PWD/golden}"
tests/metal/run_test_goldwide.sh $PUFFER_GOLDEN_DIR
```

CPU/compiler validation without executing a Metal binary:

```bash
tests/metal/run_test_goldwide.sh --build-only
python3 tests/metal/check_goldwide.py $PUFFER_GOLDEN_DIR
```

`--build-only` compiles both metallibs, five operation readers, two host control
binaries, and Freeway and Tetris replay executables in each precision at
nice 19. It exits before running any GPU work. The CPU audit checks all
SHA-256 hashes, byte counts, precision tags, reset classes, required cases,
independent-run equality and replay/live output equality. All golden files
are read-only inputs.

The bundle is `$PUFFER_GOLDEN_DIR/{wide_a,wide_b,wide_bf16,wide_bf16_b}`, with audits,
source/config snapshots and determinism reports in `$PUFFER_GOLDEN_DIR/wide_reports`.
The historical capture report is external to this repository; the bundle
manifests and reports are the machine-readable provenance.

The gate reuses existing numerical tolerances. FP32 gate/GEMM/Muon readers
cover every matching case; the BF16 reader covers all operation cases. Both
policy readers compare generic and eligible fused layers directly with CUDA.
`PUF_GOLDWIDE_FUSED=1` requires the shipped H128/H256 T64 paths to run; rejected
eligible launches fail. H512/H1024 fixtures automatically use fused functions
when those widths become eligible, and explicitly print `path=generic` until
then. Tetris H512/T128 uses the generic plain scan; the standalone scan
fixtures exercise both plain and row kernels. The full train replay also runs
with fused MinGRU enabled and disabled. Wide Muon cases additionally compare
the wide optimizer kernel directly with CUDA and bitwise with the generic tail.

Train inputs are `TAG/train_ENV/train_epoch3`; CUDA references are
`TAG/replay_ENV/train_epoch3`. Replay references contain output tensors only;
the full replayable inputs remain in both independent train captures. The new
`replay-wide` command loads the compiled environment config with the recorded
geometry overrides: 128 total agents and minibatch=128*horizon. Freeway is
H128/L4/T64, Tetris H512/L2/T128; the other default settings are preserved.
The original `replay` command and existing golden paths remain supported.

Train comparisons preserve the existing two-perturbation noise-floor contract:
float max error <= max(1e-5, 2*measured perturbation spread); integer/RNG tensors
must match exactly. This is a stability gate, not a bitwise cross-device claim.
Operation gates retain their existing bounds. CUDA run-to-run noise was zero.
The wrapper creates a fresh output directory by default. If setting
`PUFFER_GOLDWIDE_OUT` explicitly, choose a fresh directory: replay refuses
existing train output directories. `--build-only` and the CPU audit do not
establish a Metal numerical pass.

The [parity investigation](../../docs/dev/parity-investigations.md#h1024-backward-discrepancy)
records the measured diagnosis. The one-layer H1024 FP32 fixture lands within FP32 reduction error of the
MinGRU hidden activation's derivative discontinuity at zero. Its reader
reports raw CUDA gradient errors, then accounts for independently calculated
derivative branch changes. This requires float64 validation of the encoder,
projection, scan, each backward GEMM on its actual operands, and the CUDA
final gradients. Sign changes must satisfy an FP32 dot-product error bound.
The existing `1e-4` policy bound stays in effect after the analytic correction;
independent reference checks use `1e-5`. Other widths retain direct comparison.
`test_policy_kink` is a one-lane regression for identical forward values and
fourfold derivative changes across zero.

BF16 gate state comparisons retain the one-ulp bound, with a per-element
float64 check for severe cancellation: both CUDA and Metal must be within
one BF16 ulp of the correctly rounded independent result. No absolute floor
or larger ulp tolerance is added.

Before reading the goldens, the gate runs committed host-only controls against
the actual reader validators. Valid branch/cancellation cases must pass; an
out-of-bound FP32 sign crossing, corrupt scan/dW/dX, and Metal or CUDA BF16
values two ULPs from the independent reference must each exit 1 with the
expected numerical diagnostic. BF16 rounding goes directly from binary64 to
BF16 with ties to even. The rounding regression checks every finite adjacent
BF16 midpoint, both signs and neighboring doubles, plus the cancellation
example that double-rounds incorrectly through FP32.

Rerun just these controls, or prove that they catch weakened validators:

```bash
bash tests/metal/check_goldwide_controls.sh
python3 tests/metal/mutate_goldwide.py
```

The mutation runner removes the sign-crossing bound, bypasses float64
validation, and restores the intermediate float cast in isolated build
copies. Each bad input must escape its mutated validator, the control gate
must fail, and the unmodified control gate must pass again. Logs and
`results.json` are written under `build/metal/goldwide/mutations`.

To retain policy intermediates for an independent NumPy diagnosis:

```bash
# Select the output directory printed by a preceding wide-gate build/run.
export PUFFER_GOLDWIDE_OUT=/path/to/wide-gate-output
PUF_POLICY_DUMP_DIR=build/policy-dumps \
  "$PUFFER_GOLDWIDE_OUT/fp32/test_mingru" $PUFFER_GOLDEN_DIR/wide_a
python3 tests/metal/diagnose_policy_backward.py $PUFFER_GOLDEN_DIR/wide_a build/policy-dumps
```

The NumPy script writes `comparison.json` and float64 reference `.npz` files
beside the raw Metal dumps. CUDA policy cases contain final gradients but
omit projection and scan-gradient intermediates; those are reconstructed
from the golden inputs rather than attributed to an unrecorded CUDA dump.
