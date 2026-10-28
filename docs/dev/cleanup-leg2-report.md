# Cleanup sprint leg 2

2026-10-27; branch `metal-cleanup`, starting revision `ea7945e7`; upstream
`ea7f9893`. Validation host: Mac mini M4, macOS 27.0.1, Xcode 27.1,
Bash 3.2.57. No numerical sources, kernels, dispatch defaults, configuration,
seed score/hash references or external golden files were changed.

## Commits

- `7ba4e3cb`: fixture hardening, CUDA root handling, approved artifact cleanup,
  reproducible dependency downloads and documentation.
- The final verification commit records this report, corrects the documented
  report-file count, and closes the malformed-path/error-classification corner case.

## Review fixes

**P1 — malformed fixtures:** the existing `fixture_cases.json` lists 678
required manifests. New `fixture_contracts.json` pins the SHA-256 of their
canonical JSON metadata, the wide bundle's 57 report files and its expected
3,645-file hash inventory. This records the existing capture contract without
regenerating tensor data or score expectations. Shared host validation checks
schema/version/byte order, precision, nonempty scalar and tensor maps, positive
integer tensor dimensions, dtype/role, safe relative filenames, duplicate JSON
keys, exact metadata contracts and tensor byte sizes. Root-relative `.` paths
are rejected, and all checker exceptions return the invalid-fixture status
rather than the missing-file status. The metadata digest pins
all scalar keys and values, including valid zero-valued flags, so missing
reader dimensions cannot silently default to zero. Formatting/key order may
change without changing the contract.

The wide SHA file list is derived from validated manifests plus the pinned
report list and must match exactly. The per-case inventory must have complete
coverage and matching tensor counts, bytes and hash maps. Both preflight and
`check_goldwide.py` use this validation. Invalid metadata fails even when
`--allow-missing-fixtures` is enabled; missing files retain the documented skip
behavior. Future intentional metadata changes require review of the pinned
contract, not automatic regeneration by tests.

The new `fixture_validation` host-only runner gate has five test methods with
multiple negative cases: empty SHA `{}` in both CLI entrypoints, truncated or
invalid hash inventory, missing scalar/schema, empty manifest/tensor map,
zero/empty tensor dimensions, missing tensor, changed dimension/precision,
truncated tensor bytes, wrong hash coverage and unsafe paths. All negative
cases must fail. The aggregate inventory increases from 48 to **49 gates**.

**P2 — CUDA reference roots:** the generator explicitly invokes
`$CUDA_HOME/bin/nvcc`; both compiler invocations use that toolkit's headers and
libraries and `$NCCL_HOME/include` / `$NCCL_HOME/lib`. The reference wrapper
exports the roots, prepends CUDA to `PATH`, and includes both library roots in
`LD_LIBRARY_PATH`. The nested root build honors explicit NCCL roots while
retaining its prior system/Python-wheel discovery when unset. `ccache` is
optional, with the prior cached invocation preserved when it is installed.
The reference README documents these requirements and loader setup.

## Owner-approved cleanup

- Extracted exactly `log_history_bin_mean` from `tests/pufferl.cu.bak` into
  `tests/metal/fixtures/log_history/upstream_log_history_bin_mean.cu.inc`.
  Its body was compared byte-for-byte with both the backup and upstream
  `ea7f9893:src/pufferl.cu`; the header identifies upstream and its MIT license.
  The updated test still rejects both historical binning and NaN mutations.
- Removed `tests/pufferl.cu.bak`, `tests/algo.cu.bak` and
  `tests/offpolicy_scaling_study.py.orig`. Search found no active consumers of
  the latter two. Updated the historical logfix report's reference to the
  extracted fixture; the leg 1 report remains a dated historical record.
- Confirmed and removed the two Linux x86-64 ELF outputs
  `resources/trailer/{trailer,architecture}`. Exact `.gitignore` entries prevent
  their re-addition. `build.sh trailer` still builds both from the retained C
  sources; existing run/build references remain applicable after building.
- Normalized the tracked robot-arm GLB to mode 0644. Its Git blob is unchanged.

These specific deletion/extraction/mode decisions are complete. Remaining owner
decisions from leg 1 concern other trailer/media packaging, inherited contributor
guidance, coordinated history privacy/size cleanup and third-party
licensing/provenance (Collections-C, NMMO, DrussGT/derived ports, fonts, chess
art and OSRS rights). Publishing the full golden bundle and expanding alternate
path coverage also remain separate work. This leg does not claim release
readiness. No history rewrite or push was performed.

## Reproducible downloads

`tools/build_dependencies.sh`, sourced by `build.sh` and NetHack's web helper,
centralizes the pins. Raylib retains the exact 5.5 release URLs previously used.
Box2D is pinned to **df25d747**, the release returned by the old `latest` URL on
2026-10-27. fast-nle is pinned to default HEAD observed on that date:
**e7f6568568242745a7f7ebd8f730367123522cac**. A fresh checkout fetches and checks
out this exact Git object. Its upstream transitive build dependencies are not
separately reworked here.

Archive hashes, measured from fresh downloads of the exact URLs:

| Archive | SHA-256 |
|---|---|
| `raylib-5.5_macos.tar.gz` | `930c67b676963c6cffbd965814664523081ecbf3d30fc9df4211d0064aa6ba39` |
| `raylib-5.5_linux_amd64.tar.gz` | `3d95ef03d5b38dfa55c0a16ca122d382134b078f0e5b270b52fe7eae0549c000` |
| `raylib-5.5_webassembly.zip` | `798b6bea650e78a60fe49f106a15d92ea4e33efd3aa1b3efa34b0438a14bbf2c` |
| `box2d-macos-arm64.tar.gz` | `a14b0ee5efb262436805c9d0bded5dea3e617d90a9585e8f19236dbef24d6aaf` |
| `box2d-linux-amd64.tar.gz` | `5d37f7921504c09bf8387638c0096cee2109b01be5deb9e1809ee60693325c30` |
| `box2d-web.tar.gz` | `518c31d1c34dcd6f6772a1cf513bf6372df29e4a1e0fb6f32bd55f85bacd4e78` |

All six archives were freshly downloaded and extracted through the production
helper under `build/cleanup-leg2/fresh.SsKUpn/`; fast-nle was fetched there too.
The Box2D hashes also match GitHub's release-asset digests. A corrupted archive
passed through the downloader produced an explicit expected/actual SHA-256
mismatch and failed before creating an extracted directory. HTTP errors also
fail via `curl -f`. Existing extracted directories and `vendor/fast-nle/src`
are deliberately reused without verification or checkout, preserving prior
cache behavior and local patches. Cache reuse was exercised with an invalid
URL, proving it does not re-download. The guide documents this policy.

## Verification

Full runner: **49/49 PASS, 0 FAIL, 0 SKIP, exit 0**. Summary:
`build/cleanup-leg2/full-gates.log`. All 20 H64/overlap seed-mode runs match the
original fixed score/checkpoint hash references. Wide/large end-to-end paths
also retain exact score/hash agreement; wide and large mutation gates caught
all 24 and 33 mutations respectively and restored production sources.
Per-gate logs and fresh goldwide output: `build/metal/gates/run.HvO50t/`.

- Dependency checks: 49 entries, zero missing fixtures/errors; no GPU work.
- Host negative fixture gate passes. The final malformed-path guard was added
  after the aggregate host gate had completed; its expanded negative suite and
  preflight over every fixture group were rerun successfully. Evidence:
  `build/cleanup-leg2/final-fixture-validation.log` and `deps-final.log`.
  A separate runner
  dependency check with a
  missing required scalar and `--allow-missing-fixtures` exits 1 and reports
  invalid-fixture FAIL, never PASS/SKIP for the malformed group.
- Host wide audit: 276 cases, 3,312 tensor files, 1,941,212,704 tensor bytes and
  **3,645 SHA-256 files verified**. The audit was rerun after the final host guard;
  evidence: `build/cleanup-leg2/goldwide-host-final.json`. Tensor data/expected
  hashes are unchanged.
- Log-history host/training tests pass; historical and NaN binning mutations
  remain distinguishable. Evidence: `build/cleanup-leg2/log-history.log`.
- CUDA command construction tested in a disposable worktree-local fixture:
  generator and nested trainer both select the declared compiler/include/lib
  roots with ccache absent and present. Evidence:
  `build/cleanup-leg2/cuda-command-roots.log`. Actual CUDA compilation/execution
  was not run: this Apple Silicon host has no CUDA toolkit/GPU.
- Bash 3.2 syntax checks pass for all five changed shell scripts. Diff whitespace
  checks pass; numerical source/configuration and seed-reference diffs are empty.

## Follow-up: dependency cache validation

Review follow-up replaces the permissive dependency-cache reuse described above.
Raylib and Box2D now extract in a temporary sibling and are renamed into place
only after checksum verification; a marker pins the source URL and SHA-256.
Missing or mismatched markers trigger a verified replacement, with the prior
cache moved aside until the new directory is installed. fast-nle reuse now
requires a complete, clean Git checkout at the pinned commit. Clean checkouts at
another revision are fetched and detached at the pin; modified or invalid
checkouts fail without reset or deletion. Fresh clones are installed atomically.
The offline `dependency_pins` host gate covers fresh/reused/replaced/corrupt
archive caches and pinned/outdated/modified/invalid fast-nle checkouts.
Bash 3.2 syntax checks, the offline gate and the requested breakout Metal build
pass. The full runner completed **50/50 PASS, 0 FAIL, 0 SKIP**; per-gate logs
are in `build/metal/gates/run.QB5VGo/`.

## Review follow-up (fast-nle reuse)

- The clean-tree check now considers tracked changes only, so untracked build
  outputs such as `build-web/dat/nhdat` no longer block a repeat NetHack web build.
- Moving a clean checkout to the pinned revision removes `build/` and
  `build-web/` so libraries built from the old revision are not reused.
- Partial-clone detection reads promisor settings with `git config --bool`, so
  `TRUE` and mixed true/false remotes are rejected.
- A fresh clone that loses an install race re-validates the winner instead of
  nesting inside it.

`run_test_dependency_pins.sh` covers each case; removing any one of the fixes
(or both race defenses) makes it fail. NetHack and Breakout Metal builds pass.
