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
  unchanged from the original leg 13/17 helpers. Wide/large end-to-end gates
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
selected finite-run checks, not long-training equivalence claims. Full results
and observed differences are in [the leg 3 report](../../docs/dev/cleanup-leg3-report.md).

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

The public bundle is **ready, unpublished**. `golden_bundle.json` pins the
archive SHA256, size and public fixture contract; its URL is intentionally empty.
Use `--file` with the locally supplied archive until a download URL is published.
[The packaging report](../../docs/dev/golden-bundle-report.md) records the actual
archive, validation results and timings.

```sh
# Once a URL is published; defaults to <repo>/golden, just like run_all.sh.
tools/metal/fetch_goldens.sh
# Override the published location (the pinned checksum still applies).
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
prominent warning when neither source supplies any tokens. Generic `/Users/`
and `/home/` paths and email addresses are always scanned.

The packager derives the exact file set from gate groups, `fixture_cases.json`,
validated manifests and `fixture_contracts.json`, including all required wide
SHA inventory entries. It reports and excludes unused files. Files are sorted;
USTAR headers use mode 0644, zero timestamps/owners and empty owner names;
gzip uses level 9, timestamp zero and no filename. Repeated packaging with the
same Python/zlib implementation and input bytes produces identical archives;
different compressor versions should be checked before changing a release pin.
A small `.tar.gz.json` sidecar records format version, archive SHA256/size,
file count, uncompressed size and the contract-file SHA256. After successful
packaging, review and copy this metadata to `tests/metal/golden_bundle.json`;
its URL remains empty until the owner publishes. These tools never publish.

The `.tar.gz.scan.json` audit lists included/excluded files, original matches,
remaining matches, and rewritten files. UTF-8 metadata, logs and source text
are scanned; matches inside 64-character hex digests are reported as false
positives. The only permitted sanitization is a CUDA reference checkout matching
`/home/[A-Za-z0-9._-]+/metal_puffer_ref/` in `wide_reports/*.log`, replaced with
`<cuda-ref>/`. The regex and original/sanitized SHA256 for every rewritten file are tracked in
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
[the Protein exporter](../metal_ref/README-protein.md):

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
[BF16 encoder provenance](../metal_ref/README-env-bf16.md) records historical
reference branches and their generation commands. Those branch-specific BF16,
encoder and wide generation helpers are not all shipped here: full regeneration of every
public bundle case is not yet available from this checkout alone. Packaging
existing captures does not remove that limitation.
[Wide fixture documentation](GOLDWIDE.md) describes the geometry, tolerances,
independent controls and hash audit. [Robot-arm fixture documentation](fixtures/robot_arm/README.md)
records the tracked trajectory capture procedure and precision variants.
