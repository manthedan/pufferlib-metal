# Metal user guide

This fork runs the native PufferLib 5 trainer on Apple GPUs. It is an
unofficial standalone fork, not an upstream-supported Metal build.

## Requirements

- Apple Silicon and macOS 26 or newer. The runtime requires the Metal 4 GPU
  family; Intel Macs and older macOS releases are unsupported.
- Full Xcode 26 or newer, including the Metal toolchain. Command Line Tools
  alone are not sufficient.
- Homebrew installed, plus `libomp` (`brew install libomp`). The first build also needs network
  access and `curl` to fetch the Raylib 5.5 binary archive. Some environments,
  notably NetHack, download and build additional dependencies.
- Python >=3.9 for tools; NumPy for numerical fixture comparisons. NetHack also
  needs Git, CMake, make, flex, bison and m4. OSRS asset setup uses Python, curl
  and tar; see its environment README for provenance limitations.

Select Xcode in every shell used to build:

```sh
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
xcrun -sdk macosx -f metal
```

## Dependency downloads and caches

Fresh builds verify Raylib **5.5** and Box2D **df25d747** release archives with
SHA-256 before extraction. The URLs and checksums for macOS, Linux and web
artifacts are pinned in [`tools/build_dependencies.sh`](../tools/build_dependencies.sh).
A checksum mismatch or HTTP error stops the build; macOS uses `shasum`, with
`sha256sum` supported on Linux. Web archives also require `unzip`.
Box2D's release was resolved from the previous `latest` URL on 2026-10-27.

Both native and web NetHack builds fetch fast-nle commit
`e7f6568568242745a7f7ebd8f730367123522cac` (default HEAD resolved on that date)
and check out that exact commit. This pins fast-nle itself; its CMake-managed
transitive dependencies retain that upstream revision's build configuration.

Extracted Raylib/Box2D directories are reused only when their pin marker records
the current URL and archive SHA-256. Legacy, stale, or interrupted caches are
replaced from a freshly verified archive. Existing fast-nle checkouts must be
complete, clean Git worktrees; a clean checkout is moved to the pinned commit,
while local modifications cause the build to stop without resetting them.
The first build still needs network access; valid pinned caches do not.

## Build and run

The environment is compiled into the executable. Metal defaults to BF16;
add `--float` for FP32.

```sh
# BF16 build and normal training
./build.sh breakout --metal
./puffer train

# A separate FP32 executable
./build.sh breakout build/puffer_breakout_f32 --metal --float
build/puffer_breakout_f32 train

# Override INI values from the CLI
./puffer train --base.seed=1 --train.total-timesteps=5000000
```

Checkpoints contain flat FP32 master weights in both build precisions. Training
can warm-start from one, but this is not a resume: optimizer momentum, step and
schedule counters, replay data, environment state, and RNG state all start
fresh. The environment, policy shape, and parameter layout must match.

```sh
./puffer train --base.load-model-path=checkpoints/breakout/old/0000000050000000.bin

# `latest` searches the configured checkpoint directory for this environment.
./puffer eval latest --headless
./puffer eval checkpoints/breakout/run/0000000050000000.bin --headless

# Omit --headless for the environment's Raylib renderer.
./puffer eval latest

./build.sh slimevolley --metal
./puffer match --headless \
  --base.load-model-path=checkpoints/slimevolley/a.bin \
  --base.load-enemy-model-path=checkpoints/slimevolley/b.bin
```

Protein sweeps run on CPU Accelerate and launch Metal training workers. The
default sweep is large (`max_runs=1200`), so bound exploratory runs explicitly:

```sh
./build.sh cartpole --metal
./puffer sweep --sweep.max-runs=12 --sweep.gpus=-1 \
  --base.eval-episodes=0 --train.total-timesteps=131072 \
  --sweep.train.total-timesteps.min=65536 \
  --sweep.train.total-timesteps.max=262144
```

By default, Metal Protein uses public Joe--Kuo D(6) Sobol directions with
fixed-seed SplitMix64 digital XOR shifts. The compact table is embedded, so
builds and sweeps need no download. Sweeps remain statistically equivalent in
the sense of using digitally scrambled, stratified Sobol points with the same
20,000-dimension capacity and 32-bit Gray-code construction; individual
suggestions are **no longer bitwise-identical to upstream CUDA Protein**.
This does not assert identical optimization outcomes. Public points are in
`[0,1)` (a float rounded to 1 is clamped to the largest float below 1).

For exact cuRAND compatibility, supply the private external fixture before
starting the sweep:

```sh
export PUF_PROTEIN_SOBOL_TABLE="$PUFFER_GOLDEN_DIR/protein/curand_sobol32.bin"
# Run the bounded sweep above with this variable set.
# Return to the default public table:
unset PUF_PROTEIN_SOBOL_TABLE
```

The loader requires exactly 2,640,000 bytes and the pinned cuRAND SHA256;
an invalid path or mismatched table fails clearly. Selection is cached at
first use for the process. [Export and validation instructions](../tests/metal_ref/README.md#external-curand-protein-fixture)
describe how to supply the table. The public generator, input/output checksums,
and BSD attribution are in [THIRD_PARTY.md](../THIRD_PARTY.md); the exact gates
report `SKIP(missing fixture)` when the external table is unavailable.

The Protein cost fit and `early_stop_quantile` exist, but the upstream CLI has
no live worker-progress/stop protocol, so sweeps do not currently terminate a
trial early.

## Precision and expected speed

BF16 is the recommended default: it reduces activation/rollout memory and is
usually faster. `--float` selects FP32 for both host code and Metal shaders.
The runtime rejects an external metallib compiled for the other precision.
Checkpoint files remain FP32. Cross-precision loading requires identical tensor
shapes **and compatible allocator layouts**. Some continuous policies (including
Drone H64/L2) have precision-dependent padding. Their checkpoints now include a
`.bin.layout` sidecar identifying the build precision; retain it with the weights
and load with that same precision. Cross-precision loads for these layouts are
rejected. Legacy files (including upstream checkpoints) without a sidecar still
load as before, with a warning: they are assumed to come from the current build
precision, which raw byte counts cannot verify. Compatible layouts (such as
standard discrete policies) still support cross-precision loading.

The [README benchmark link](../README.md) points to the single
[dated benchmark table](benchmarks.md), with machine, OS, power, precision,
commit and measurement caveats. Throughput depends on policy geometry,
environment cost, other GPU users and sustained SoC power limits.

## Memory sizing

Metal uses unified memory. `MTLDevice.recommendedMaxWorkingSetSize` is a soft
performance recommendation, not an allocation limit. Before creating the large
policy, gradient, and activation pools, the trainer estimates a conservative
shared-buffer peak, including later rollout stream rings/workspaces, eligible
packed weights, and initialization scratch. Exceeding the available recommendation
prints a **WARNING** with requested/recommended bytes and continues. A single
page-rounded buffer exceeding `maxBufferLength` is a hard preflight failure.
Allocation sizes and training numerics are not adjusted automatically. Actual
allocation can still fail under memory pressure; driver-private storage is not
included in the estimate.

The largest drivers are rollout observations/action masks and train
activations. Reduce these keys, usually in this order:

1. `vec.total_agents`
2. `train.horizon`
3. `train.minibatch_size`
4. `policy.hidden_size` (and, if appropriate, `policy.num_layers`)

`--base.async=0` removes the second rollout slot and can make a large
configuration fit, at the cost of learner/rollout overlap. BF16 also reduces
most activation storage. On the 24 GB M4 Air used for the review follow-up,
the shipped g2048 FP32 plan requested about 14.10 GB, with a 13.74 GB largest
buffer, below the reported 19.07 GB recommendation and 14.30 GB single-buffer
limit. Limits vary with hardware; the recommendation is advisory on every host.

## Environment and CUDA feature coverage

The single current environment table and links to historical measurements are in
[Metal environment coverage](metal-env-coverage.md). The smoke sweep trained
81 environments in FP32, including custom Metal policies for NetHack, Craftax, NMMO3,
OSRS encounters, Minimal, and Asteroids. Standard CPU environments are the
best-supported path.

| Area | Metal status |
|---|---|
| `train`, headless/rendered `eval`, `match` | Supported for ported CPU environments |
| Protein `sweep` | Supported on one Apple GPU; live trial early-stop is not wired |
| BF16 / FP32 | Supported; BF16 is default |
| Warm start | Weights only; not optimizer/RNG/exact resume |
| Multi-GPU / NCCL | Unsupported; one Metal device per run |
| CUDA graphs and CUDA profiler/NVML | CUDA graphs are disabled; Metal counters and profiling variables are available |
| CUDA `.cu` simulators | Unsupported; use CPU environment implementations where available |
| Optional `N3_ATTN` / `MINIMAL_ATTN` | CUDA/PTX-only and rejected explicitly |
| Matsci | Requires an external LAMMPS installation not bundled here |
| Drive | Requires WOMD map data under `drive_data/binaries` |
| Web build | Remains the CPU/Emscripten path, not a Metal backend |

Robot-arm uses the CPU implementation in `ocean/robot_arm/robot_arm_cpu.h`.
The [recorded Mac mini M4 measurements](dev/robot-arm.md#port-and-cpu-physics)
show environment-only throughput of ~155–158k steps/s on ten threads and bounded
training of ~110–116k SPS with the default four threads. Full default-config
1B-step runs on the Mac mini M4 averaged 81–93k SPS (88k mean over three seeds),
versus 146–161k SPS for CUDA GPU physics on an RTX 3090. In that
[three-seed comparison](dev/robot-arm.md#full-budget-three-seed-comparison) neither
backend solves the default task at 1B steps (0% success and lift everywhere);
final returns match (−4.269 mean on both) and grasp rates (Metal 0.3–9.8%,
CUDA 7.5–10.1%) show no significant difference, though three seeds per side
cannot establish equivalence.

## Trainer and platform differences from upstream

Relative to upstream `ea7f9893`, the core learning algorithm is retained, with
these observable trainer/platform changes (not an exhaustive internal diff):

- Metal evaluation prints `METAL_EVAL` rather than implying a CUDA device
  (`PUF_EVAL_BACKEND` in `src/pufferl.cu`; covered by the eval-label gate).
- Training honors `base.load_model_path` as a **weights-only warm start**,
  initializing optimizer, replay and RNG state afresh. NetHack's frozen lookup
  data is refreshed after loading (`puffer_train` in `src/pufferl.cu`).
- Log-history binning assigns each sample to one output bin and handles empty
  bins without producing NaNs. The final epoch bypasses logging throttles,
  and sparse/late metric observations no longer shift samples into earlier bins (`log_history_bin_mean` and the training loop).
- Precision-dependent checkpoint layouts carry a `.bin.layout` sidecar and
  reject incompatible cross-precision loads; raw FP32 weights remain unchanged
  (`puf_checkpoint_check_layout` / checkpoint save in `src/pufferl.cu`).
- Metal memory preflight warns on the recommended working-set budget and fails
  on a hard single-buffer limit without silently resizing training geometry.
- Robot-arm CPU initialization explicitly orders RNG draws to match the observed
  CUDA reference host/device orders. Upstream's argument evaluation order was
  unspecified (`ocean/robot_arm/robot_arm.h`, `RA_CPU` branch).
- Metal uses one device, CPU simulation, shared-memory transport and its own
  asynchronous rollout/learner scheduling. CUDA graphs/NCCL are unavailable;
  Protein sweeps use CPU Accelerate (`src/metal/protein_cpu.h`).

The [gate inventory](../tests/metal/README.md) identifies checks for these paths.
`./build.sh all --metal` fails clearly: build named environments with `--metal`
and optionally `--float`, or run the bounded environment smoke sweep.

## User-facing environment variables

These are the supported operational/diagnostic variables found in `src/`:

| Variable | Effect |
|---|---|
| `PUF_ENV_THREADS=N` | Override total CPU environment worker threads. Results are unchanged because environments have per-env RNG. |
| `PUF_METAL_ALLOC_LOG=1` | Log every page-rounded Metal allocation and live/device budget totals. |
| `PUF_METAL_KPROF=1` | Print per-kernel GPU time at exit. It adds precise timestamps and serializes encoders, so do not use its wall time as a benchmark. |
| `PUF_METAL_KPROF=2` | Also break the kernel profile down by stream and grid shape. |
| `PUF_METAL_GEMM_LOG=1` | Print each distinct GEMM shape and selected implementation once. |
| `PUF_METAL_PIPELINE_INFO=1` | Print Metal pipeline thread width and threadgroup-memory limits as pipelines are created. |
| `PUF_METAL_TRACE=FILE.csv` | Write a command-buffer/wait timeline at exit; summarize it with `python3 tools/metal/trace_timeline.py FILE.csv`. |
| `PUF_METALLIB=FILE.metallib` | Advanced: replace the embedded primary shader library. Its BF16/FP32 marker must match the executable. |

The source also contains optimizer-path overrides, auxiliary-library overrides,
and variables containing `TEST`; those are internal tuning, regression, or
debug controls rather than stable user settings.

## Testing and golden fixtures

The [Metal regression suite](../tests/metal/README.md) has 59 default gates.
Thirteen require external fixtures: twelve CUDA numerical comparisons and one
exact-cuRAND Protein gate. Missing fixtures report `SKIP(missing fixture)` and
make the suite fail unless `--allow-missing-fixtures` is explicitly selected.
Host controls, public Protein, and other fixture-independent gates still run.

The package/fetch tools use Bash 3.2+, Python 3.9+ and (for downloads) curl.
The public format is `.tar.gz`, readable by stock macOS/Linux tar. Fetch checks
the pinned archive SHA256, sizes, file list and fixture contracts before
installing at `$PUFFER_GOLDEN_DIR` (default `<repo>/golden`).

The bundle is published as the
[`goldens-v1` release](https://github.com/manthedan/pufferlib-metal/releases/tag/goldens-v1)
(two parts, because GitHub caps release assets at 2 GiB; the fetcher joins and
verifies them):

```sh
tools/metal/fetch_goldens.sh
# Or use an owner-provided local archive with the tracked digest:
tools/metal/fetch_goldens.sh --file /path/to/metal-goldens-v1.tar.gz
# The public bundle intentionally omits the exact-cuRAND fixture:
tests/metal/run_all.sh --allow-missing-fixtures
```

`--url` overrides the download location without overriding its expected hash.
`--force` preserves an existing installation as a `.backup-*` sibling. See the
[test README](../tests/metal/README.md#external-golden-bundle) for local sidecars,
resume behavior, packaging and the full audit procedure.

Regenerate CUDA operation/train captures on a CUDA host following
[`tests/metal_ref`](../tests/metal_ref/README.md), with explicit `CUDA_HOME` and
`NCCL_HOME`; historical BF16/encoder/wide helpers are not all shipped here.
The NVIDIA cuRAND-exported `protein/curand_sobol32.bin` is never redistributed.
For private exact-CUDA Protein parity, follow the
[cuRAND 12.8 exporter instructions](../tests/metal_ref/README.md#external-curand-protein-fixture) on a
CUDA host, then copy the table into your private fixture directory. Its absence
leaves `protein` (and optional `protein_mutations`) skipped; `protein_public`
continues to run. Do not generate numerical expectations from Metal outputs.

## Troubleshooting

**`Metal compiler not found` or `xcrun: error`:** select the full Xcode app,
then install the separately downloadable toolchain if needed:

```sh
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
xcodebuild -downloadComponent MetalToolchain
xcrun -sdk macosx -f metal
```

**Xcode license or first-launch failure:** complete Xcode setup once, then
retry from a new shell:

```sh
sudo xcodebuild -license accept
sudo xcodebuild -runFirstLaunch
```

**`brew --prefix libomp` fails or OpenMP headers/libraries are missing:**
install Homebrew `libomp` and rebuild: `brew install libomp`.

**Raylib archive/download errors:** the first build downloads
`raylib-5.5_macos`. Check network access and rerun from the repository root.
The build replaces an unmarked or mismatched cache only after a new archive has
verified and extracted successfully, so manual cache removal is not required.

**A render window cannot open:** rendered eval/match requires a logged-in macOS
GUI session and is the default when `--headless` is absent. Use `--headless`
over SSH, in CI, or when no WindowServer session is available. Run from the
repository root so environment-specific assets can be found.

**Memory warning or preflight failure:** use the requested/recommended values in the diagnostic,
then reduce the listed geometry keys or try BF16 / `--base.async=0`. Set
`PUF_METAL_ALLOC_LOG=1` only when a more detailed allocation trace is useful.

**A long NetHack epoch looks stalled:** large H=1024 policies can spend tens of
seconds in a learner command buffer. The runtime warns after a 60-second host
wait and prints stream progress; it aborts only after five minutes with no GPU
completion anywhere. See [the runtime investigation](dev/parity-investigations.md#metal-runtime-debugging).
