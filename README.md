# pufferlib-metal (unofficial)

An unofficial standalone fork of [PufferLib 5](https://github.com/PufferAI/PufferLib)
with a native Apple Metal 4 backend for training on Apple Silicon. The upstream
base is branch `5.0` at `ea7f989371270d2948f96728bda8aec5e7634a4c`.
It is not affiliated with or supported by PufferAI. Report fork-specific issues
in **this repository's issue tracker**. This fork carries its own platform changes;
it does not promise automatic synchronization with upstream.

The core learning algorithm is retained from upstream; platform/trainer behavior
changes are listed in the [Metal guide](docs/metal-guide.md#trainer-and-platform-differences-from-upstream).
The trainer compiles as C++ against CUDA-named compatibility shims backed by Metal.
BF16 is the default; `--float` selects FP32. Training uses one Apple GPU and CPU
environment implementations.

Requirements: Apple Silicon, macOS 26+, full Xcode 26 with the Metal toolchain,
Homebrew with `libomp`, and network access for the first build. Python tools need
Python >=3.9 (numerical tests also need NumPy). See the [setup guide](docs/metal-guide.md)
for optional-environment dependencies.

```sh
brew install libomp
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
./build.sh breakout --metal
./puffer train
```

Validation covers the configurations registered in [the Metal gate suite](tests/metal/README.md):
selected FP32/BF16 kernels, CUDA fixture comparisons and train replays, exact seeded
checkpoint comparisons, and platform/CLI checks. External goldens are required for
those comparison gates. The historical environment sweep smoke-trained **81 environments
in FP32**; it does not establish all-environment BF16 or CUDA learning parity.
See [current coverage](docs/metal-env-coverage.md) and the
[dated benchmark table](docs/benchmarks.md) for the measured configurations and limits.
Protein hyperparameter sweeps run on CPU Accelerate; see the guide for usage.
The CUDA-reference golden fixtures for the Metal tests are a [release download](https://github.com/manthedan/pufferlib-metal/releases/tag/goldens-v1); see the [fetch instructions](tests/metal/README.md#external-golden-bundle).

Credits: PufferAI and upstream contributors; [Valtteri Valo's Metal 4 backend](https://github.com/valtterivalo/PufferLib)
for PufferLib 4.0 (PR #532); and [MLX](https://github.com/ml-explore/mlx)
for adapted Muon kernel components. See [NOTICE](NOTICE), [LICENSE](LICENSE), and the
[third-party inventory](THIRD_PARTY.md).

## Original upstream README

The text below is retained from upstream PufferLib. Its support and marketing
statements describe upstream, not support for this standalone Metal fork.

---

[![Discord](https://dcbadge.limes.pink/api/server/puffer?style=plastic)](https://discord.gg/puffer)
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/cloudposse.svg?style=social&label=Follow%20%40jsuarez)](https://twitter.com/jsuarez)

PufferLib is a fast and sane reinforcement learning library that can train tiny, super-human models in seconds. The included learning algorithm, hyperparameter tuning, and simulation methods are the product of our own research. All our tools are free and open source. Need a high performance environment for your application? We build them professionally and offer training + extended support. Contact jsuarez🐡puffer🐡ai.

All of our documentation is hosted at [puffer.ai](https://puffer.ai "PufferLib Documentation"). @jsuarez5341 on [Discord](https://discord.gg/puffer) for support. Post there before opening issues. We're always looking for new contributors!

## Star to puff up the project!

<a href="https://star-history.com/#pufferai/pufferlib&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=pufferai/pufferlib&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=pufferai/pufferlib&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=pufferai/pufferlib&type=Date" />
 </picture>
</a>
