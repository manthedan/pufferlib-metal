# Cleanup sprint leg 1

2026-10-27; branch `metal-cleanup`, base `e68239d9`; upstream `ea7f9893`.
Validation host: Mac mini M4, 16 GB, macOS 27.0.1, Xcode 27.1 (27A9275),
Bash 3.2.57. No numerical implementation, dispatch default, fixture value or
fixed score/checkpoint expectation was changed.

## Commits

- `b2fdf8d2`: tracked portable gates, recovered helpers, documentation/privacy/archive pass, build and comment fixes.
- `700870c4`: corrected NetHack/OSRS instructions and preserved archived JSON formatting.
- `207f6431`: malformed-fixture handling and clarified historical measurement provenance.
- This report is recorded in the final cleanup verification commit.

## Work packages

**A — clone-complete gates:** replaced the ignored-location runner with
`tests/metal/run_all.sh` and tracked H64/overlap/seed helpers. Preserved all eight
fixed score/SHA-prefix pairs. Added the 48-gate inventory, dependency/fixture
checks, explicit missing-fixture skips and opt-in allowance, per-run logs and
fresh goldwide output. Caller Xcode selection is preserved. H256 always uses
tracked Breakout with the existing exact generic/default/opt-out comparison;
no new expected values were generated. Guide smoke binaries are isolated.
The test README documents suites, capture layout/provenance and the external
bundle's current availability limits. CUDA reference setup no longer sources
a private sibling script.

**B — documentation/privacy:** rewrote the fork introduction, requirements,
validation scope and support routing while preserving upstream README content.
Added one dated benchmark table and one current environment coverage table;
archived reports/relay entries and result JSON under `docs/dev/`, with revision
banners/metadata and repaired links. Removed personal infrastructure strings and
the embedded review instruction; JSON numeric fields were compared against the
base and are unchanged. Documented verified trainer/platform differences,
CPU-backed robot-arm support and qualified throughput/parity claims. Updated
NOTICE's upstream base and MLX scope, and corrected NetHack/OSRS commands.

**C — quality:** corrected stale NetHack/Muon comments only; `all --metal`
now fails clearly for both default and `--float` requests. The CUDA all-build
branch is unchanged. Documented Python >=3.9 and optional build dependencies.

## Verification

Full runner: **48/48 PASS, 0 FAIL, 0 SKIP, exit 0**, including all 20 H64/overlap
seed-mode runs with the original score/hash references. Summary:
`build/cleanup-gates-final.log`; per-gate logs and fresh goldwide output:
`build/metal/gates/run.lrIGL5/`. The root `puffer` hash remained unchanged across
the guide test. A preliminary run was interrupted after an in-flight build-script
edit caused a shell-read failure; the complete restarted run above is the result
reported here. No failing result was accepted as a pass.

- Empty fixture directory: `--list` prints 48 entries; `--check-deps` reports
  12 `SKIP(missing fixture)` and exits 1; adding `--allow-missing-fixtures`
  exits 0 without marking any skipped gate PASS.
- Normal-run skip/exit control flow was separately exercised with GPU commands
  stubbed; malformed fixture metadata still fails with the allowance enabled.
  Those control tests are not included in the real GPU gate count.
- Fresh local clone at `207f6431`: 48 entries; dependency checks exit 0 with
  goldens and 1 with the empty directory (12 skips). Tracked privacy scan:
  zero matches. Only the disposable clone was removed afterward.
- Changed shell scripts pass Bash 3.2 syntax checks; guide contract and build-help
  checks pass. Current/archived Markdown link targets exist. Compiler-source
  changes are comments only. Original helper reference pairs, JSON numbers,
  upstream README text, backups and trailer executables were checked unchanged.

## Remaining data and owner decisions

No requested personal-name/home-path patterns remain in tracked files. Upstream
and vendor copyright/attribution text is retained, including attribution emails
in `vendor/collections-c/include/{cc_array.h,cc_common.h,dlmalloc.h}`. Historical
Git objects and commit metadata were not scrubbed or rewritten.

**Pending owner decision:** deletion/extraction of `tests/pufferl.cu.bak`,
`tests/algo.cu.bak` and `tests/offpolicy_scaling_study.py.orig`; removal of the
Linux executables `resources/trailer/{trailer,architecture}`; trailer/media and
other upstream asset packaging; curation of inherited contributor guidance;
coordinated history privacy/size cleanup. All specified files were retained.

Asset licensing/provenance inherited from upstream remains pending owner
review: Collections-C notices, NMMO assets, DrussGT and derived ports, fonts,
chess art, and OSRS download rights. This leg adds attribution context, not a
redistribution-rights determination, and does not establish release readiness.

Suggested next leg: publish/version the full golden bundle and generation
workflow with hashes, pin/checksum downloaded dependencies, finish the
third-party inventory and owner decisions, then address the audit's missing
alternate-path gates. Any history rewrite or tracked-file deletion requires a
separate authorized leg. No history rewrite or push was performed here.
