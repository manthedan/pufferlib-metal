# Metal environment coverage

Current support as of 2026-10-27. This is the authoritative
current environment summary. The [historical measured audit](dev/parity-investigations.md#environment-coverage-and-training-evidence)
summarizes measured configurations and historical statuses superseded by later ports.

| Environment or group | Current Metal support and evidence |
|---|---|
| Standard CPU environments | Native Metal learner; the historical sweep smoke-trained 81 environments in **FP32**, with two learner updates and finite-loss checks. This is not a convergence or all-BF16 claim. |
| NetHack, Craftax, NMMO3, OSRS encounters | Custom Metal policy adapters; selected FP32/BF16 CUDA encoder fixtures and replay gates. |
| Minimal, Asteroids | Default point encoders supported; CPU formula and selected BF16 CUDA fixture checks. Optional `MINIMAL_ATTN` is unsupported. |
| `robot_arm` | Supported through `ocean/robot_arm/robot_arm_cpu.h`, with BF16 or FP32 Metal learners. CPU physics trajectory fixtures are gated. A three-seed 1B-step comparison found no significant learning difference versus CUDA (neither backend solves the default task at that budget); CPU physics trains at ~88k SPS versus ~152k for CUDA. See the [port investigation](dev/robot-arm.md#port-and-cpu-physics) and [learning comparison](dev/robot-arm.md#full-budget-three-seed-comparison). |
| `matsci` | Requires an external LAMMPS installation; excluded from the successful smoke count. |
| `drive` | Requires external WOMD maps in `drive_data/binaries`; excluded from the successful smoke count. |
| `osrs/` | Shared support directory, not a standalone environment; build an encounter such as `osrs_colosseum`. |
| CUDA `.cu` simulators, `N3_ATTN`, `MINIMAL_ATTN` | Unsupported on Metal; use the default CPU simulator/policy where available. |

Run `tests/metal/run_all.sh --list` for the current regression inventory and read
[the test guide](../tests/metal/README.md) for fixtures and limits. The optional
`tools/metal/env_sweep.sh` performs bounded smoke training; it is not part of the
59-gate aggregate suite. Its current build command uses the default BF16 precision;
the archived 81-environment measurement explicitly used FP32.
