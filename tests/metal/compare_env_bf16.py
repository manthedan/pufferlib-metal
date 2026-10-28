"""BF16 env gate: BF16 policy bounds plus exact/fp64 primitive checks.

Full networks inherit rounded GEMM intermediates: 4 ulp OR 2^-5 tensor max,
unchanged from the BF16 policy gate. Simple transforms and fixed-point sums
are additionally checked exactly; direct arithmetic uses fp64 (1 ulp or
2^-12 tensor max). Integer backward state and all inputs/params are exact.
"""

import argparse
import json
from pathlib import Path
import sys
import numpy as np
from compare_env_golden import load, norm_err, argmax_groups
from env_bf16_math import spacing, validate

# Fill patterns for memory a kernel never wrote: CUDA goldens were dumped
# from 0xa5-filled allocations, Metal replays from 0x7fc1 halfwords
# (tests/metal/env_golden.h), which are NaN in bf16/f32. Unwritten positions
# must coincide exactly; a Metal store skipped where CUDA wrote is then a
# non-finite failure, never a value hiding inside a tolerance.
SIZE = {"f32": 4, "bf16": 2, "i32": 4, "u64": 8}
UINT = {2: "<u2", 4: "<u4", 8: "<u8"}


def raw(case, entry):
    n = SIZE[entry["dtype"]]
    return np.fromfile(case / entry["file"], dtype=UINT[n]).reshape(entry["shape"])


def filled(r, fill):
    return r == np.frombuffer(fill * (r.itemsize // len(fill)), dtype=r.dtype)[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("golden", type=Path)
    ap.add_argument("metal", type=Path)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    cases = sorted(
        p.name for p in args.metal.iterdir() if (p / "manifest.json").exists()
    )
    if not cases:
        sys.exit("no replay cases")
    report = {}
    failures = 0
    for case in cases:
        g, gm = load(args.golden / case)
        m, mm = load(args.metal / case)
        meta = json.loads((args.metal / case / "manifest.json").read_text())
        if meta["precision"] != "BF16":
            sys.exit(f"{case}: expected BF16 replay")
        env = meta["scalars"]["env"]
        groups = argmax_groups(args.metal / case / "manifest.json")
        rows = {}
        worst = 0
        for name, b in g.items():
            if mm[name] != gm[name]:
                sys.exit(f"{case}/{name}: tensor metadata mismatch")
            gr, mr = raw(args.golden / case, gm[name]), raw(args.metal / case, mm[name])
            gu = filled(gr, b"\xa5")
            # Backward replays restore CUDA's saved state first, so a region
            # neither backend writes can also hold CUDA's own fill bytes.
            mu = filled(mr, b"\xc1\x7f") | (gu & (mr == gr))
            if not np.array_equal(gu, mu):
                rows[name] = {"pass": False, "bitwise_equal": False,
                              "unwritten_cuda": int(gu.sum()), "unwritten_metal": int(mu.sum())}
                continue
            if gu.all():
                # Neither backend writes this dump (e.g. backward-only state).
                rows[name] = {"pass": True, "bitwise_equal": True, "unwritten": True}
                continue
            if gu.any():
                m[name] = np.where(mu, b, m[name])
            a = m[name]
            equal = a.tobytes() == b.tobytes()
            r = {"bitwise_equal": equal}
            if gm[name]["role"] in ("input", "param") or b.dtype == np.int32:
                # Forward argmax is checked against each backend's actual
                # scores below, rather than allowing arbitrary index errors.
                ok = equal or (name in groups and name.startswith("acts_fwd"))
            elif gm[name]["dtype"] == "f32":
                n, d, scale, nf = norm_err(a, b)
                worst = max(worst, n)
                ok = (
                    nf == 0
                    and n <= 1e-5
                    and bool(np.all(np.isfinite(a)) and np.all(np.isfinite(b)))
                )
                if name.endswith("actions"):
                    ok = equal
                r.update(norm_err=n, max_abs=d)
            elif name.startswith("heads_") and gm[name]["dtype"] == "bf16":
                ok = equal
            else:
                n, d, scale, nf = norm_err(a, b)
                worst = max(worst, n)
                bound = np.maximum(4 * spacing(b), 2.0**-5 * scale)
                ok = nf == 0 and bool(
                    np.all(np.abs(a.astype(np.float64) - b.astype(np.float64)) <= bound)
                )
                r.update(norm_err=n, max_abs=d)
            r["pass"] = bool(ok)
            rows[name] = r
        math = {
            side: validate(data, g, env, groups, reference=(side == "cuda"))
            for side, data in [("metal", m), ("cuda", g)]
        }
        # OSRS pooling uses an FP32 GELU tile, not the rounded saved h1.
        for name, (flat, h1, l2w, active_w) in groups.items():
            z1 = "acts_fwd_" + f"{int(h1.rsplit('_', 1)[1]) - 1:02d}"
            for side, data in [("metal", m), ("cuda", g)]:
                B, H = data[name].shape
                R = data[flat].shape[0] // B
                z = data[z1].astype(np.float64).reshape(B, R, 16)
                hp = 0.5 * z * (1 + np.tanh(0.7978845608028654 * (z + 0.044715 * z**3)))
                scores = hp @ data[l2w].astype(np.float64).T
                active = data[flat].reshape(B, R, -1)[:, :, :active_w].sum(axis=2) > 0
                scores = np.where(active[:, :, None], scores, -np.inf)
                am = data[name]
                valid = (am >= 0) & (am < R)
                chosen = np.take_along_axis(
                    scores, np.maximum(0, am)[:, None, :], axis=1
                )[:, 0, :]
                best = scores.max(axis=1)
                has = active.any(axis=1)[:, None]
                # fp32 transcendentals + 16-term dot; FP32 golden bound.
                with np.errstate(invalid="ignore"):
                    gap = np.where(has, best - chosen, 0)
                first = scores.argmax(axis=1)
                # Identical records have identical FP32 scores: strict first
                # winner semantics must hold even when a near-tie is allowed.
                first_z = z[np.arange(B)[:, None], first]
                chosen_z = z[np.arange(B)[:, None], np.maximum(0, am)]
                duplicate = np.all(first_z == chosen_z, axis=2) & has
                tie_ok = np.all(~duplicate | (am == first))
                ok = tie_ok and np.all(
                    np.where(
                        has,
                        valid
                        & (
                            gap
                            <= 1e-5
                            * np.maximum(
                                1,
                                np.max(
                                    np.where(np.isfinite(scores), np.abs(scores), 0),
                                    axis=1,
                                ),
                            )
                        ),
                        am == -1,
                    )
                )
                math[side]["pool " + name] = {
                    "pass": bool(ok),
                    "max_score_gap": float(gap.max()),
                }
        failures += sum(not r["pass"] for r in rows.values()) + sum(
            not r.get("pass", True) for checks in math.values() for r in checks.values()
        )
        for name, r in rows.items():
            if not r["pass"]:
                print("FAIL", case, name, r)
        for side, checks in math.items():
            for name, r in checks.items():
                if not r.get("pass", True):
                    print("FAIL", case, side, name, r)
        print(
            case,
            "worst_norm",
            f"{worst:.6g}",
            "exact",
            sum(r["bitwise_equal"] for r in rows.values()),
            "/",
            len(rows),
            "unwritten",
            sum(r.get("unwritten", False) for r in rows.values()),
        )
        report[case] = {"tensors": rows, "math": math}
    if args.json:
        args.json.write_text(json.dumps(report, indent=1) + "\n")
    print(
        "BF16 env goldens:",
        "PASS" if not failures else "FAIL",
        f"({failures} failures)",
    )
    sys.exit(bool(failures))


if __name__ == "__main__":
    main()
