# Wide CUDA reference gates

From the repository root (see [suite setup](README.md)):

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
The script refuses existing train output directories under
`build/metal/goldwide`; move those result directories before another full run.
No Metal numerical pass is claimed until this script is actually run.

The one-layer H1024 FP32 fixture lands within FP32 reduction error of the
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
PUF_POLICY_DUMP_DIR=build/policy-dumps \
  build/metal/goldwide/fp32/test_mingru $PUFFER_GOLDEN_DIR/wide_a
python3 tests/metal/diagnose_policy_backward.py $PUFFER_GOLDEN_DIR/wide_a build/policy-dumps
```

The NumPy script writes `comparison.json` and float64 reference `.npz` files
beside the raw Metal dumps. CUDA policy cases contain final gradients but
omit projection and scan-gradient intermediates; those are reconstructed
from the golden inputs rather than attributed to an unrecorded CUDA dump.
