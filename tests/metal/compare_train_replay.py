#!/usr/bin/env python3
"""Compare a Metal train() replay dump against the CUDA replay golden.

Usage: tests/metal/compare_train_replay.py GOLDEN_REPLAY_CASE METAL_OUT_CASE [NOISE_A NOISE_B]
  e.g. ../golden/replay_a/train_epoch3 build/metal/train_replay

With NOISE_A/NOISE_B (two Metal replays whose inputs were nudged by +-1 ulp
with different seeds, see PUF_REPLAY_PERTURB_SEED), a float tensor passes when
its Metal-vs-CUDA max abs error is within NOISE_FACTOR x the max abs spread
between the two noisy replays: one train() call amplifies rounding noise far
beyond fixed tolerances, mostly through Muon's Newton-Schulz iterations.

Every tensor with role output in the golden must exist in the Metal dump.
Exit status is nonzero if any tensor exceeds the tolerances.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "metal_ref"))
import json  # noqa: E402

import numpy as np  # noqa: E402
from golden import load_case  # noqa: E402

# One train() call = several minibatches of forward/backward + Muon. Float
# summation order differs from cuBLAS/CUDA reductions, so require closeness,
# not bit-equality; integer/RNG state must match exactly.
ATOL = 1e-5
RTOL = 1e-4
NOISE_FACTOR = 2.0


def main():
    ref_dir, out_dir = sys.argv[1], sys.argv[2]
    roles = {k: t["role"] for k, t in json.load(open(os.path.join(ref_dir, "manifest.json")))["tensors"].items()}
    ref, out = load_case(ref_dir), load_case(out_dir)
    noise = (load_case(sys.argv[3]), load_case(sys.argv[4])) if len(sys.argv) == 5 else None
    failed = 0
    print(f"{'tensor':28} {'shape':>16} {'max_abs':>11} {'max_rel':>11} {'noise':>11}  status")
    for name, role in roles.items():
        if role != "output":
            continue
        if name not in out:
            print(f"{name:28} {'':>16} {'':>11} {'':>11} {'':>11}  MISSING")
            failed += 1
            continue
        a, b = np.asarray(ref[name]), np.asarray(out[name])
        if a.shape != b.shape:
            print(f"{name:28} shape {a.shape} vs {b.shape}  MISMATCH")
            failed += 1
            continue
        if a.dtype.kind in "iu":
            ok = np.array_equal(a, b)
            print(f"{name:28} {str(a.shape):>16} {'':>11} {'':>11} {'':>11}  {'exact' if ok else 'DIFF'}")
            failed += not ok
            continue
        a64, b64 = a.astype(np.float64), b.astype(np.float64)
        finite = np.isfinite(b64).all()
        diff = np.abs(a64 - b64)
        rel = diff / np.maximum(np.abs(a64), 1e-12)
        if noise:
            floor = np.abs(np.asarray(noise[0][name], np.float64) - np.asarray(noise[1][name], np.float64)).max()
            ok = finite and bool(diff.max() <= max(ATOL, NOISE_FACTOR * floor))
            nstr = f"{floor:11.3e}"
        else:
            ok = finite and bool(np.all(diff <= ATOL + RTOL * np.abs(a64)))
            nstr = f"{'':>11}"
        print(f"{name:28} {str(a.shape):>16} {diff.max():11.3e} {rel.max():11.3e} {nstr}  "
              f"{'ok' if ok else ('NONFINITE' if not finite else 'FAIL')}")
        failed += not ok
    print("PASS" if not failed else f"FAIL ({failed} tensors)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
