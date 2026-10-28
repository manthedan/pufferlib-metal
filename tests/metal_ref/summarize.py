#!/usr/bin/env python3
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import numpy as np
from golden import compare,load_case

root=Path(sys.argv[1]);repo=Path(__file__).resolve().parents[2]
def capture(*args):return subprocess.check_output(args,cwd=repo,text=True).strip()
def write(name,value):(root/name).write_text(json.dumps(value,indent=2)+"\n")
live=json.loads((root/"determinism_live.json").read_text());replay=json.loads((root/"determinism_replay.json").read_text())
r=compare(load_case(root/"run_a/train_epoch3"),load_case(root/"replay_a/train_epoch3"));write("replay_vs_live.json",r)
variants={}
for a,b in (("scan_breakout","scan_row_breakout"),("scan_plain_long","scan_row_long"),("scan_plain_partial","scan_row_tiny")):
    variants[a+"_vs_"+b]=compare(load_case(root/"run_a"/a),load_case(root/"run_a"/b))
write("scan_variants.json",variants)
write("summary.json",{
 "cases_per_full_run":len(live),"operation_cases":len(live)-1,
 "live_bitwise_identical":all(v["bitwise_equal"] and v["manifest_equal"] for v in live.values()),
 "live_max_abs":max(v["max_abs"] for v in live.values()),
 "replay_bitwise_identical":all(v["bitwise_equal"] and v["manifest_equal"] for v in replay.values()),
 "replay_max_abs":max(v["max_abs"] for v in replay.values()),
 "replay_matches_live_bitwise":all(v["bitwise_equal"] for v in r.values()),
 "measured_nondeterministic_ops":[k for k,v in live.items() if not v["bitwise_equal"]],
 "determinism_scope":"Two fresh processes per generator, same GPU/toolchain. Train capture completes actor prefetch before snapshot; same-input replay compared separately. No guarantee across drivers or GPUs.",
 "scan_variant_max_abs":{pair:max(v["max_abs"] for v in tensors.values()) for pair,tensors in variants.items()},
 "minibatch_losses":load_case(root/"run_a/train_epoch3")["minibatch_losses"].tolist(),
 "loss_order":["PG","VF","ENT","TOTAL","OLD_APPROX_KL","APPROX_KL","CLIPFRAC","IMP","count"],
 "validation_checks":len(json.loads((root/"run_a_validation.json").read_text()))})
write("metadata.json",{
 "upstream_commit":"ea7f989371270d2948f96728bda8aec5e7634a4c",
 "harness_commit":capture("git","rev-parse","HEAD"),"branch":capture("git","branch","--show-current"),
 "gpu":capture("nvidia-smi","--query-gpu=name,driver_version","--format=csv,noheader"),
 "nvcc":capture("nvcc","--version"),"gcc":capture("gcc","--version"),"python":platform.python_version(),"numpy":np.__version__,
 "build":"tests/metal_ref/build.sh", "generate":"tests/metal_ref/run.sh",
 "source_sha256":{str(p.relative_to(repo)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted([repo/"src/pufferl.cu",repo/"src/algo.cu",*Path(__file__).parent.glob("*.cu"),*Path(__file__).parent.glob("*.h"),*Path(__file__).parent.glob("*.py"),*Path(__file__).parent.glob("*.sh")])}})
assert all(v["bitwise_equal"] for v in r.values()),"train replay differs from live capture"

print((root/"summary.json").read_text())
