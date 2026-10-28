#!/usr/bin/env python3
"""The production wrapper and direct references must each select all new stages."""
import argparse, os, re, subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('binary');p.add_argument('--log',required=True);a=p.parse_args()
r=subprocess.run([str(Path(a.binary).resolve())],env=dict(os.environ,PUF_METAL_KPROF='1'),capture_output=True,text=True)
s=r.stdout+r.stderr;Path(a.log).write_text(s)
assert r.returncode==0,s[-4000:]
for name in ['wide_large_muon_gram','wide_large_muon_gram_buf','wide_large_muon_update']:
 m=re.search(r'^'+name+r'\s+(\d+)\s',s,re.M)
 assert m and int(m[1])==30,('large wrapper kernel selection',name,m[1] if m else None)
m=re.search(r'^puf_copy_bytes\s+(\d+)\s',s,re.M)
assert m and int(m[1])==212,('large aligned staging selection',m[1] if m else None)
print('ALL PASS: large wrapper/direct stages and twelve aligned staging copies')
