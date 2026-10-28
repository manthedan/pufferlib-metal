# OSRS encounters

Use the [Metal prerequisites](../../docs/metal-guide.md#requirements), Python >=3.9,
`curl`, and `tar`. The first build runs the hash-checked asset setup in
`scripts/setup-data.sh`. Rights/provenance for the downloaded OSRS asset archive
remain pending owner review; upstream inheritance does not establish redistribution
permission. The environment is compiled into the executable; add `--float` for FP32.

## Build

```bash
./build.sh osrs_colosseum --metal
./build.sh osrs_inferno --metal
```

## Train and eval

```bash
./build.sh osrs_colosseum --metal
./puffer train
./puffer eval latest
./puffer eval checkpoints/osrs_colosseum/model.bin
./build.sh osrs_inferno --metal
./puffer sweep --sweep.max-runs=12 --sweep.gpus=-1
```

## Tests

```bash
cc -std=c11 -O2 -I. -o /tmp/t ocean/osrs/tests/test_colosseum_golden.c -lm && /tmp/t
```
