#!/usr/bin/env python3
"""Portable raw-tensor reader, numerical comparison, and two-tree determinism report.

manifest.json: schema_version=1, byte_order=little, tensors[name]={dtype,shape,
file,role}, scalars={...}. All tensors are C contiguous with no headers/padding.
Philox opaque u8 state is CUDA-ABI-specific; decoded fields are also supplied.
bf16 = raw IEEE bfloat16 bits (little-endian u16). load_case widens bf16 exactly to
float32 ((u16 << 16) viewed as f32, NaN payloads preserved); pass raw_bf16=True for
the uint16 bit memmaps. Bitwise equality of widened arrays equals bitwise equality
of the raw bits, so compare() works unchanged for bf16-vs-bf16.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

DTYPES = {"f32": np.dtype("<f4"), "i32": np.dtype("<i4"),
          "u8": np.dtype("u1"), "u64": np.dtype("<u8"), "bf16": np.dtype("<u2")}

def widen_bf16(bits):
    """Exact bf16 -> float32 widening of raw uint16 bits."""
    out = (np.asarray(bits, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)
    out.flags.writeable = False
    return out

def bf16_bits(x):
    """Raw bf16 bits of a float32 array that holds exactly-widened bf16 values."""
    u = np.asarray(x, dtype=np.float32).view(np.uint32)
    if (u & 0xFFFF).any():
        raise ValueError("array is not exactly representable in bf16")
    return (u >> 16).astype(np.uint16)

def round_bf16(x):
    """Round to nearest even bf16 (from float32), returned widened as float32."""
    u = np.asarray(x, dtype=np.float32).view(np.uint32).astype(np.uint64)
    nan = np.isnan(np.asarray(x, dtype=np.float32))
    r = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)
    r = np.where(nan, ((u >> 16) | 0x40).astype(np.uint16), r)
    return widen_bf16(r)

def ulp_diff(a, b, bits):
    """|ordered-int distance| between same-shaped float arrays at bf16 (16) or f32 (32)
    granularity; +0/-0 are 0 apart. Entries with any NaN are reported as 0 if both NaN."""
    def ordered(x):
        u = np.asarray(x, dtype=np.float32).view(np.uint32).astype(np.int64)
        if bits == 16:
            u >>= 16
        sign = u >> (bits - 1)
        mag = u & ((1 << (bits - 1)) - 1)
        return np.where(sign != 0, -mag, mag)
    d = np.abs(ordered(a) - ordered(b))
    return np.where(np.isnan(a) & np.isnan(b), 0, d)

def manifest_dtypes(directory):
    return {k: v["dtype"] for k, v in
            json.loads((Path(directory) / "manifest.json").read_text())["tensors"].items()}

def load_case(directory, raw_bf16=False):
    """Return {tensor_name: read-only numpy array}; validate shape and byte count.

    bf16 tensors come back as exactly-widened float32 arrays (or raw uint16 bit memmaps
    when raw_bf16=True); every other dtype is a read-only little-endian memmap."""
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("byte_order") != "little":
        raise ValueError("only little-endian cases are supported")
    arrays = {}
    for name, spec in manifest["tensors"].items():
        dtype = DTYPES[spec["dtype"]]
        shape = tuple(spec["shape"])
        if not shape or any(not isinstance(n, int) or n <= 0 for n in shape):
            raise ValueError(f"invalid shape for {name}: {shape}")
        path = directory / spec["file"]
        if path.parent.resolve() != directory.resolve():
            raise ValueError(f"tensor file must be in case directory: {path}")
        expected = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
        if path.stat().st_size != expected:
            raise ValueError(f"{name}: expected {expected} bytes, got {path.stat().st_size}")
        if spec["role"] not in {"input", "output", "param", "state"}:
            raise ValueError(f"invalid role for {name}")
        arrays[name] = np.memmap(path, dtype=dtype, mode="r", shape=shape)
        if spec["dtype"] == "bf16" and not raw_bf16:
            arrays[name] = widen_bf16(arrays[name])
    return arrays

def compare(a, b, *, atol=0.0, rtol=0.0, dtype=None):
    """Report max abs/relative error, bit parity, and first tolerance mismatch.

    Relative error = abs(a-b)/max(abs(a),abs(b),float64.tiny).
    Equal NaNs compare numerically equal; bitwise_equal still checks payloads.
    Dicts compare matching tensor names recursively. No printing side effects.
    dtype="bf16" (a manifest dtype, or a dict of them for dict inputs) accepts widened
    float32 or raw uint16 inputs and adds max_ulp in bf16 units; float32 inputs get
    max_ulp in f32 units. Arrays are compared as given otherwise.
    """
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            raise ValueError(f"tensor names differ: {a.keys() ^ b.keys()}")
        dt = dtype if isinstance(dtype, dict) else {k: dtype for k in a}
        return {k: compare(a[k], b[k], atol=atol, rtol=rtol, dtype=dt.get(k)) for k in a}
    if dtype == "bf16":
        a, b = (widen_bf16(x) if np.asarray(x).dtype == np.uint16 else x for x in (a, b))
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch {a.shape} != {b.shape}")
    if a.dtype != b.dtype:
        raise ValueError(f"dtype mismatch {a.dtype} != {b.dtype}")
    bitwise = a.tobytes() == b.tobytes()
    if a.dtype.kind in "iu":
        # Exact subtraction for u64 counters without float64 precision loss.
        mismatch = a != b
        indices = np.flatnonzero(mismatch)
        max_abs = max((abs(int(a.flat[i])-int(b.flat[i])) for i in indices), default=0)
        max_rel = max((abs(int(a.flat[i])-int(b.flat[i])) / max(abs(int(a.flat[i])),abs(int(b.flat[i])),1) for i in indices), default=0.0)
        rel = np.array([max_rel])
    else:
        with np.errstate(invalid="ignore"):  # signaling NaN payloads widen quietly
            aa, bb = a.astype(np.float64), b.astype(np.float64)
        equal = (aa == bb) | (np.isnan(aa) & np.isnan(bb))
        finite = np.isfinite(aa) & np.isfinite(bb)
        diff = np.zeros(a.shape, dtype=np.float64)
        np.subtract(aa, bb, out=diff, where=finite)
        np.abs(diff, out=diff)
        diff[~finite & ~equal] = np.inf
        denom = np.maximum(np.maximum(np.abs(aa), np.abs(bb)), np.finfo(float).tiny)
        rel = np.zeros(a.shape, dtype=np.float64)
        np.divide(diff, denom, out=rel, where=finite)
        rel[~finite & ~equal] = np.inf
        tolerance = atol + rtol * np.where(finite, np.abs(bb), 0)
        mismatch = ~equal & (~finite | (diff > tolerance))
        max_abs = float(diff.max(initial=0))
        indices = np.flatnonzero(mismatch)
        if a.dtype == np.float32:
            ulp_bits = 16 if dtype == "bf16" else 32
            ulps = np.where(finite | equal, ulp_diff(a, b, ulp_bits), np.iinfo(np.int64).max)
            max_ulp = int(ulps.max(initial=0))
    first = tuple(int(i) for i in np.unravel_index(int(indices[0]), a.shape)) if indices.size else None
    def value(x):
        v=x[first].item()
        return str(v) if isinstance(v, float) and not np.isfinite(v) else v
    extra = {}
    if a.dtype == np.float32:
        extra = {"ulp_unit": "bf16" if dtype == "bf16" else "f32", "max_ulp": max_ulp}
    return {"bitwise_equal": bitwise, "max_abs": max_abs, **extra,
            "max_rel": float(rel.max(initial=0)), "mismatch_count": int(indices.size),
            "first_mismatching_index": first,
            "a_at_first": value(a) if first is not None else None,
            "b_at_first": value(b) if first is not None else None}

def compare_trees(a, b):
    a, b = Path(a), Path(b)
    names_a = {p.parent.relative_to(a) for p in a.rglob("manifest.json")}
    names_b = {p.parent.relative_to(b) for p in b.rglob("manifest.json")}
    if names_a != names_b:
        raise ValueError(f"case directories differ: {names_a ^ names_b}")
    report = {}
    for name in sorted(names_a):
        ma=json.loads((a/name/"manifest.json").read_text())
        mb=json.loads((b/name/"manifest.json").read_text())
        result=compare(load_case(a/name), load_case(b/name),
                       dtype={k:v["dtype"] for k,v in ma["tensors"].items()})
        report[str(name)]={"manifest_equal": ma==mb,
                           "bitwise_equal": all(r["bitwise_equal"] for r in result.values()),
                           "max_abs": max(r["max_abs"] for r in result.values()),
                           "tensors": result}
    return report

def inventory(root):
    result={}
    for path in sorted(Path(root).rglob("manifest.json")):
        arrays=load_case(path.parent)
        result[str(path.parent.relative_to(root))]={
            "tensor_count":len(arrays), "bytes":sum(x.nbytes for x in arrays.values()),
            "nonfinite_float_tensors":[n for n,x in arrays.items() if x.dtype.kind=="f" and not np.isfinite(x).all()],
            "sha256":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(path.parent.iterdir()) if p.is_file()}}
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("a");p.add_argument("b",nargs="?");p.add_argument("--output")
    args=p.parse_args()
    report=compare_trees(args.a,args.b) if args.b else inventory(Path(args.a))
    text=json.dumps(report,indent=2,allow_nan=True)+"\n"
    if args.output:Path(args.output).write_text(text)
    else:print(text,end="")

if __name__=="__main__":main()
