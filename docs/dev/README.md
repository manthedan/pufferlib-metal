# Development records

Technical evidence behind the Metal backend; start with the [Metal guide](../metal-guide.md)
for installation and use. Measurements describe their recorded configurations
and toolchains, not universal performance or learning guarantees.

- [Optimization history](optimization-history.md): benchmark method, retained changes,
  rejected experiments and remaining headroom.
- [Parity investigations](parity-investigations.md): numerical comparisons, environment
  training, Protein, logging and runtime debugging.
- [Robot arm](robot-arm.md): CPU physics, trajectory validation and learning comparison.
- [Robot-arm learning data](robot-arm-learning-results.json): raw results behind the
  three-seed, full-budget comparison.

Commit hashes cited in dev records refer to the private development history.
This repository publishes that work as a single commit on top of upstream
PufferLib 5.0 (`ea7f9893`), so those hashes do not resolve here.
