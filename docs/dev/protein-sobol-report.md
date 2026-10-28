# Protein public Sobol replacement

2026-10-27, `metal-cleanup3`, upstream PufferLib 5 `ea7f9893`; Mac mini M4,
Xcode selected through `/Applications/Xcode.app/Contents/Developer`, Bash 3.2.

## Design and provenance

The NVIDIA-exported `src/metal/protein_sobol32.bin` was copied, byte-for-byte,
to `$PUFFER_GOLDEN_DIR/protein/curand_sobol32.bin` and removed with `git rm`.
The default public artifact is `src/metal/protein_sobol_public.bin`:
**394,264 bytes** (385.02 KiB), down from **2,640,000 bytes**, an **85.07%**
reduction. It contains no cuRAND directions or constants.

Protein accepts up to 20,000 dimensions, with custom CLI axes discovered from
configuration. Although the synthetic gate uses 21, bundled configurations can
exceed 21. Capping the replacement at 21 would reduce existing functionality.
The compact artifact therefore retains exactly the supported 20,000 dimensions,
omitting the remaining 1,201 public dimensions. Dimension 1 is implicit; the
other dimensions store polynomial degrees and packed coefficients/odd initial
direction values. The standard recurrence expands these into 32 uint32 words
per dimension once per process. The runtime layout remains `[20000][33]`.
Expanded memory is 2,640,000 bytes; no build-time download or generation is needed.

The source is Joe and Kuo's `new-joe-kuo-6.21201`, from
https://web.maths.unsw.edu.au/~fkuo/sobol/. The live UNSW URL returned 404 on
the validation date. The [Joe--Kuo source repository](https://github.com/joe-kuo/sobol_data)
provided a byte-identical copy of the source retained from the prior review.
The full BSD-style license and attribution are reproduced in `THIRD_PARTY.md`
and `NOTICE`, from that repository's `LICENSE`. An exact publisher-text excerpt
is committed at `tools/metal/joe-kuo/new-joe-kuo-6.20000`, with the license and
attribution alongside it. The excerpt contains the header and only dimensions
2 through 20,000, the rows Protein uses.

| Data | SHA256 |
|---|---|
| Full publisher text (1,887,612 bytes) | `68eedd2a4e3b659b9695e7aff0f8ac68718bcf620730fc3d3a8c65df2a067441` |
| Committed publisher-text excerpt (1,771,202 bytes) | `af7492ff6eb0c4e1d968a15710d6496e61709d151821859841b0c16893d7b905` |
| Committed compact public parameters | `adca16e3a020a0ee099fd130b6cb5f46628c14fe629909b7fa850eb585d07210` |
| Expanded public rows, including shifts | `e6c1be592b7b514b41adb06d9f2e364471e8e4670825b1428e8974e2a2072749` |
| External cuRAND table | `fce775e7962beda83fa73c64c628a0323a94eadf5d32bad97d1dc6159e52e5f0` |

Scrambling is a digital XOR shift. Each dimension receives the low 32 bits of
the next SplitMix64 output from fixed seed `0x50554646534f424c` (`PUFFSOBL`).
The state is incremented by `0x9e3779b97f4a7c15` before the two multiply/xorshift
mixes, with 64-bit wrapping arithmetic. The generator documents the exact bit
packing and PRNG. Sweep seeds retain their existing meaning as uint32 Sobol
offsets; draws retain Gray-code XOR evaluation. Public float conversion retains
the fused scaling, with values rounded to 1 clamped to the largest float below
1 to satisfy `[0,1)`. Exact mode retains the original conversion unchanged.

Default sweeps remain statistically equivalent as digitally scrambled Sobol
sampling with the same dimensions and 32-bit construction. Individual
suggestions are no longer bitwise-identical to upstream CUDA, and this is not
a claim that optimization outcomes are identical. No non-Protein numerics or
defaults were changed.

## Regeneration and exact mode

```sh
# Offline regeneration/check from the committed, hash-pinned source excerpt.
python3 tools/metal/gen_protein_sobol.py --check

# Download from UNSW, falling back to the Joe--Kuo repository; verify the full
# SHA256, derive its excerpt, and compare that with the committed excerpt.
python3 tools/metal/gen_protein_sobol.py --download --check

# The same full-file validation without network access.
python3 tools/metal/gen_protein_sobol.py \
  --source /path/to/new-joe-kuo-6.21201 --check

# Offline expansion of the hash-pinned compact artifact.
python3 tools/metal/gen_protein_sobol.py \
  --from-compact src/metal/protein_sobol_public.bin --check \
  --expanded-output build/metal/protein/public-expanded.bin

export PUF_PROTEIN_SOBOL_TABLE="$PUFFER_GOLDEN_DIR/protein/curand_sobol32.bin"
# Run a sweep with exact cuRAND data; unset the variable for public mode.
unset PUF_PROTEIN_SOBOL_TABLE
```

The runtime caches table selection at first use. An explicitly set path must
be readable and contain exactly 2,640,000 bytes with the recorded SHA256.
Empty paths, missing files, truncation, trailing bytes and wrong contents fail
clearly; none silently fall back to public mode. CommonCrypto provides the
SHA256 implementation already available on macOS.

`fixture_cases.json` and `fixture_contracts.json` record the external table;
`check_fixtures.py` distinguishes absent data (exit 1) from malformed data
(exit 2). The exact Protein and Protein mutation gates select the fixture
automatically and use the runner's existing `SKIP(missing fixture)` convention.
The public, CLI sweep and guide gates explicitly select public mode.

`tests/metal_ref/README-protein.md` and `export_protein_sobol.sh` document the
CUDA-host export. A one-output option in the existing CUDA exporter writes
only the table, with no GPU work. **The new exporter option/wrapper has not
been run on CUDA**; the existing export was preserved and hash-checked here.

## Validation

`tests/metal/run_all.sh`: **58/58 PASS, 0 FAIL, 0 SKIP(missing fixture)**,
exit 0. Summary: `build/metal/protein/run-all.log`. Per-gate logs:
`build/metal/gates/run.kR2YLp/`. This includes all six alternate-path gates;
root `./puffer` remained absent. The full run used the final production code
and final public tests.

- Exact Protein gate: **PASS**, including all **1,464** bitwise CUDA
  Sobol/Philox/glibc fixtures, the 21D synthetic sweep, GP numerics/holdout and
  prediction batching. Logs: `build/metal/protein/exact.log` and the full-run
  `protein.log`.
- Protein mutation controls: **9/9 caught**, followed by a passing clean exact
  baseline. Log: `build/metal/protein/mutations.log`.
- Public gate: **PASS**, including the two-trial Cartpole sweep. The existing
  12-trial CLI sweep and the guide smoke (including its default two-trial
  sweep) also passed in the full run.
- Missing-fixture checks used an APFS clone of the complete golden set under
  `build/metal/protein/golden-without-protein/`, with only
  `protein/curand_sobol32.bin` removed. `run_all.sh --check-deps
  --allow-missing-fixtures` examined **58 gates**, reported exactly **1
  SKIP(missing fixture)** (`protein`), and exited 0. The public gate's
  dependencies remained available. Direct exact and mutation entry points
  reported `protein SKIP(missing fixture)` and `protein_mutations
  SKIP(missing fixture)` respectively. Logs: `missing-deps.log`,
  `missing-direct.log`, `missing-mutations.log` under `build/metal/protein/`.
  Dependency preflight does not execute numerical gates; its PASS counter is 0.
  The public gate was then executed against this incomplete fixture copy and
  **passed**, including its two-trial Cartpole sweep, even with an inherited
  `PUF_PROTEIN_SOBOL_TABLE` pointing at the missing file (the public gate
  explicitly unsets it). Log: `build/metal/protein/public-missing.log`.
- Download, full local source, offline source-excerpt and offline compact
  generator checks all passed.
  Bash 3.2 syntax and staged whitespace checks passed.

The public gate parses the committed raw publisher-text excerpt, regenerates
the compact table without calling the compact unpacker, and compares the
394,264 artifact bytes with the committed table. It independently expands
those parsed source rows in Python and compares all 2,640,000 runtime bytes,
including generated shifts, byte-for-byte. It also changes one valid direction
number in a temporary excerpt and proves generation fails its pinned-source
check. The full-source generator was run separately with `--check`; it verified
the complete publisher-file SHA256, derived and byte-compared the committed
excerpt, and reproduced the artifact exactly. All gate paths are offline.

Public checks cover the first ten three-dimensional Joe--Kuo reference points;
triangular/nonsingular directions and 256-point dyadic stratification for all
20,000 dimensions before and after digital shifting; no duplicate points in
the first 4,096 draws; endpoint/range checks including a constructed uint32
maximum; and identical suggestions across two fresh processes with the same
seed and synthetic observations, including GP suggestions. A real two-trial
Cartpole sweep also runs. Bad source hashes and external-table open/size/hash
errors have negative controls.

The first two-trial smoke run exposed a test expectation error: trial zero is
the initial configuration, so only trial one is random. The test now counts
`min(runs - 1, 10)` random suggestions; the existing 12-trial expectation is
unchanged. No production change was needed for this correction.

## Removal audit

The tracked-tree audit checks whole-file SHA256 and four distinctive sequences
of eight uint32 words from the beginning, dimensions 74 and 10,000, and end of
the external table, in raw bytes and formatted hexadecimal/decimal forms.
No table copy or matching sampled sequence was found. Text searches for the
old hash, filename and cuRAND export identifiers locate only the intended
validation pins, exporter/instructions and historical provenance reports.
The old port report now links to this replacement report. RNG expected-result
fixtures remain tests, not exported direction/scramble data.

No history rewrite or push was performed; this removes the data from the
current tracked tree, not from pre-existing Git history. Work outside this
worktree was limited to `$PUFFER_GOLDEN_DIR/protein/`.
