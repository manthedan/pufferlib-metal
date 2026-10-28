"""Compare Metal custom-encoder replays against CUDA goldens (env_golden.h).

Usage: compare_env_golden.py GOLDEN_RUN METAL_OUT [--tol 1e-5] [--json OUT.json]

Float tensors pass when max|metal - cuda| / max|cuda| <= tol (error normalised
by the reference's max magnitude). Integer argmax tensors after backward must
match exactly (backward replays CUDA's post-forward activations). Forward
argmax may differ only where two records tie to rounding: at most 1e-4 of
entries, the pooled output must still pass, and every differing entry must
pick a valid active record whose score (recomputed in float64 from the golden
h1 and l2 weights) is within --tie-tol of the golden winner's. The metadata for
that check comes from the replay manifest's pool_argmax_groups note; an
integer tensor without it must match exactly. int64 fixed-point embedding
accumulators are compared like floats. Inputs must be bitwise identical.
"""
import argparse, json, pathlib, sys
import numpy as np

DT = {"f32": "<f4", "bf16": "<u2", "i32": "<i4", "u64": "<i8"}


def load(case):
    m = json.loads((case / "manifest.json").read_text())["tensors"]
    tensors = {}
    for k, v in m.items():
        a = np.fromfile(case / v["file"], dtype=DT[v["dtype"]]).reshape(v["shape"])
        if v["dtype"] == "bf16":
            a = (a.astype(np.uint32) << 16).view(np.float32)
        tensors[k] = a
    return tensors, m


def argmax_groups(manifest_path):
    note = json.loads(manifest_path.read_text()).get("scalars", {}).get("pool_argmax_groups", "")
    groups = {}
    for g in filter(None, note.split(";")):
        am, flat, h1, l2w, active = (int(x) for x in g.split(":"))
        groups[f"acts_fwd_{am:02d}"] = (f"acts_fwd_{flat:02d}", f"acts_fwd_{h1:02d}",
                                       f"param_{l2w:02d}", active)
    return groups


def bad_ties(name, metal_idx, gold, group, tie_tol):
    """Count differing argmax entries that are not genuine near-ties."""
    flat_k, h1_k, l2w_k, active_w = group
    am_gold = gold[name]
    B, H = am_gold.shape
    flat = gold[flat_k].astype(np.float64)
    R = flat.shape[0] // B
    active = flat.reshape(B, R, -1)[:, :, :active_w].sum(axis=2) > 0
    h1 = gold[h1_k].astype(np.float64).reshape(B, R, -1)
    l2w = gold[l2w_k].astype(np.float64)
    bad = 0
    for b, h in zip(*np.nonzero(metal_idx != am_gold)):
        m, g = int(metal_idx[b, h]), int(am_gold[b, h])
        if not (0 <= m < R and 0 <= g < R and active[b, m] and active[b, g]):
            bad += 1
            continue
        scores = h1[b] @ l2w[h]
        best = scores[active[b]].max()
        scale = max(1.0, float(np.abs(scores[active[b]]).max()))
        if best - scores[m] > tie_tol * scale or best - scores[g] > tie_tol * scale:
            bad += 1
    return bad


def norm_err(a, b):
    a, b = a.astype(np.float64), b.astype(np.float64)
    fin = np.isfinite(a) & np.isfinite(b)
    nonfinite = int(np.sum(~fin & ~(np.isnan(a) & np.isnan(b)) & (a != b)))
    ref = float(np.max(np.abs(b[fin]))) if fin.any() else 0.0
    d = float(np.max(np.abs(a[fin] - b[fin]))) if fin.any() else 0.0
    return d / ref if ref > 0 else d, d, ref, nonfinite


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("golden"); ap.add_argument("metal")
    ap.add_argument("--tol", type=float, default=1e-5)
    ap.add_argument("--tie-tol", type=float, default=1e-5,
                    help="max score gap, relative to max(1, max |score|), for an argmax flip")
    ap.add_argument("--json")
    args = ap.parse_args()
    gold, out = pathlib.Path(args.golden), pathlib.Path(args.metal)
    report, failures = {}, 0
    cases = sorted(p.name for p in out.iterdir() if (p / "manifest.json").exists())
    if not cases:
        sys.exit(f"no Metal cases in {out}")
    for case in cases:
        g, gm = load(gold / case)
        m, _ = load(out / case)
        groups = argmax_groups(out / case / "manifest.json")
        rows, worst = {}, 0.0
        for name in g:
            kind = gm[name]["role"]
            a, b = m[name], g[name]
            r = {"bitwise_equal": bool(a.tobytes() == b.tobytes()), "elements": int(b.size)}
            if kind in ("input", "param"):
                ok = r["bitwise_equal"]
            elif b.dtype == np.int32:
                mism = int(np.sum(a != b))
                r["mismatches"] = mism
                if name.startswith("acts_fwd") and name in groups and mism:
                    r["non_tie_mismatches"] = bad_ties(name, a, g, groups[name], args.tie_tol)
                    ok = mism <= 1e-4 * b.size and r["non_tie_mismatches"] == 0
                else:
                    ok = mism == 0
            else:
                n, d, ref, nonfin = norm_err(a, b)
                r.update({"norm_err": n, "max_abs": d, "max_ref": ref, "nonfinite_mismatch": nonfin})
                ok = nonfin == 0 and n <= args.tol
                worst = max(worst, n)
            r["pass"] = bool(ok)
            failures += not ok
            rows[name] = r
            if not ok:
                print(f"  FAIL {case}/{name}: {r}")
        argmax_flips = sum(v.get("mismatches", 0) for k, v in rows.items() if k.startswith("acts_fwd"))
        bitwise = sum(v["bitwise_equal"] for k, v in rows.items() if gm[k]["role"] not in ("input", "param"))
        outputs = sum(1 for k in rows if gm[k]["role"] not in ("input", "param"))
        print(f"{case:26s} tensors={len(rows):3d} worst_norm_err={worst:.3g} "
              f"bitwise={bitwise}/{outputs} fwd_argmax_flips={argmax_flips} "
              f"{'PASS' if all(v['pass'] for v in rows.values()) else 'FAIL'}")
        report[case] = rows
    if args.json:
        pathlib.Path(args.json).write_text(json.dumps(report, indent=1) + "\n")
    print(f"env goldens: {'PASS' if failures == 0 else 'FAIL'} ({failures} failing tensors, tol {args.tol:g})")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
