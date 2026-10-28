# External cuRAND Protein fixture

Public Metal builds embed only Joe--Kuo construction data. Exact CUDA parity
requires a user-supplied cuRAND 12.8 export, outside the repository. On a
little-endian CUDA host with `nvcc` and cuRAND installed:

```sh
export PUFFER_GOLDEN_DIR=/absolute/path/to/private/golden
bash tests/metal_ref/export_protein_sobol.sh
```

The wrapper and new host-only exporter option are **untested on CUDA** (developed
on a Mac). The existing two-output exporter also regenerates the RNG samples
and requires a GPU. The one-output option only calls cuRAND host table accessors;
it launches no kernels. Do not use `-DNDEBUG`: export assertions check API/I/O
errors. Only regenerate in a fixture directory you intend to update.

Copy the result to `$PUFFER_GOLDEN_DIR/protein/curand_sobol32.bin` on the Mac.
It must contain exactly 20,000 little-endian rows of 33 uint32 words: 32 words
from `CURAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6`, followed by the corresponding
`curandGetScrambleConstants32` word. Size: **2,640,000 bytes**. SHA256:

`fce775e7962beda83fa73c64c628a0323a94eadf5d32bad97d1dc6159e52e5f0`

The tracked `tests/metal/fixture_cases.json` and `fixture_contracts.json` pin this
external fixture; `check_fixtures.py ... protein` verifies its size and digest.
Missing data is a skip; malformed data is a failure. Never commit the export.

```sh
export PUF_PROTEIN_SOBOL_TABLE="$PUFFER_GOLDEN_DIR/protein/curand_sobol32.bin"
./puffer sweep --sweep.max-runs=2 --sweep.gpus=-1
```

Selection is cached on first Sobol use per process. An unset variable selects
public mode; an empty, unreadable, short, oversized or wrong-hash path fails
clearly. Exact mode retains cuRAND's original float conversion, including its
possible rounded endpoint of 1.0. `run_test_protein.sh` and `mutate_protein.py`
select the fixture automatically. `run_all.sh --allow-missing-fixtures` permits
`SKIP(missing fixture)` for unavailable exact gates while still running the
offline public-table gate, default sweep and guide smoke.
