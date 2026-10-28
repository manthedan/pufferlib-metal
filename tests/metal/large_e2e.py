#!/usr/bin/env python3
"""Seed-1 H512/H1024 fixed-default-geometry checkpoint/score comparison."""
import argparse, hashlib, os, re, subprocess, tempfile
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('binary');p.add_argument('env',choices=['tetris','wef']);p.add_argument('--reference');p.add_argument('--quick',action='store_true');a=p.parse_args()
root=Path.cwd();out=Path(tempfile.mkdtemp(prefix='large_e2e.',dir='build/metal')).resolve()
flags=['PUF_METAL_GEMM_LARGE_TN','PUF_METAL_MUON_LARGE','PUF_METAL_GEMM_LARGE_LINEAR']
steps=1048576 if a.env=='tetris' else 65536

def run(tag,binary,settings):
 d=out/tag;env=dict(os.environ)
 for key in flags+['PUF_METAL_KPROF']:env.pop(key,None)
 env.update(settings)
 cmd=[str(Path(binary).resolve()),'train','--base.seed=1',f'--train.total-timesteps={steps}',
  '--base.eval-episodes=128','--base.eval-agents=128','--base.checkpoint-interval=100000',
  f'--base.log-dir={d}/logs',f'--base.checkpoint-dir={d}/checkpoints']
 r=subprocess.run(cmd,env=env,capture_output=True,text=True)
 (out/(tag+'.log')).write_text(r.stdout+r.stderr)
 assert r.returncode==0,(tag,r.returncode,out)
 ck=list((d/'checkpoints').rglob('*.bin'));ini=list((d/'logs').rglob('*.ini'))
 assert len(ck)==len(ini)==1,(tag,ck,ini)
 scores=re.findall(r'^env/score\s*=\s*(.*)',ini[0].read_text(),re.M);assert scores,(tag,ini)
 score=scores[-1].split(',')[-1].strip();sha=hashlib.sha256(ck[0].read_bytes()).hexdigest()
 summary=re.search(r'^metal_summary.*',r.stdout,re.M);assert summary,tag
 dispatch=float(re.search(r'train_dispatches=([^ ]+)',summary[0])[1])
 print(f'{tag} score={score} sha={sha} train_dispatches={dispatch}',flush=True)
 return score,sha,dispatch
ref=run('generic',a.binary,dict.fromkeys(flags,'0'))
on=run('default',a.binary,{})
assert ref[:2]==on[:2],(ref,on,out)
if a.env=='wef':assert on[2]<ref[2],('large Muon not selected',ref,on)
if a.reference:
 saved=run('saved_base',a.reference,{})
 assert saved[:2]==ref[:2],(saved,ref,out)
if not a.quick:
 for flag in flags:
  off=run(flag+'_off',a.binary,{flag:'0'})
  assert off[:2]==ref[:2],(flag,ref,off,out)
print('ALL PASS',out,flush=True)
