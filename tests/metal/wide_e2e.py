#!/usr/bin/env python3
"""Short fixed-geometry seed gate: each wide optimization on/off vs generic."""
import argparse, hashlib, os, re, subprocess, tempfile
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('binary');p.add_argument('env');p.add_argument('--reference');p.add_argument('--steps',type=int,default=2097152);p.add_argument('--extra',action='append',default=[]);args=p.parse_args()
out=Path(tempfile.mkdtemp(prefix='wide_e2e.',dir='build/metal')).resolve()
flags=['PUF_METAL_GEMM_WIDE_TN','PUF_METAL_FUSED_MINGRU_WIDE','PUF_METAL_GEMM_RAGGED','PUF_METAL_MUON_WIDE_MMA']
def run(tag,binary,env):
 d=out/tag
 e=dict(os.environ)
 for k in flags+['PUF_METAL_KPROF']:e.pop(k,None)
 e.update(env)
 cmd=[str(Path(binary).resolve()),'train','--base.seed=1',f'--train.total-timesteps={args.steps}',
      '--base.eval-episodes=128','--base.eval-agents=128','--base.checkpoint-interval=100000',
      f'--base.log-dir={d}/logs',f'--base.checkpoint-dir={d}/checkpoints']+args.extra
 r=subprocess.run(cmd,env=e,capture_output=True,text=True)
 (out/(tag+'.log')).write_text(r.stdout+r.stderr)
 assert r.returncode==0,(tag,r.returncode,out)
 ck=list((d/'checkpoints').rglob('*.bin'));assert len(ck)==1,ck
 ini=list((d/'logs').rglob('*.ini'));assert len(ini)==1,ini
 scores=re.findall(r'^env/score\s*=\s*(.*)',ini[0].read_text(),re.M)
 assert scores,('no score',ini)
 score=scores[-1].split(',')[-1].strip();sha=hashlib.sha256(ck[0].read_bytes()).hexdigest()
 summary=re.search(r'^metal_summary.*',r.stdout,re.M);assert summary,tag
 dispatch=float(re.search(r'train_dispatches=([^ ]+)',summary[0])[1])
 print(f'{tag} score={score} sha={sha} train_dispatches={dispatch}',flush=True)
 return score,sha,dispatch
ref=run('generic',args.reference or args.binary,dict.fromkeys(flags,'0'))
on=run('default',args.binary,{})
assert on[:2]==ref[:2],(ref,on,out)
assert on[2]<ref[2],('fused learner not selected',on[2],ref[2])
for flag in flags:
 off=run(flag+'_off',args.binary,{flag:'0'})
 assert off[:2]==ref[:2],(flag,ref,off,out)
print('ALL PASS',out,flush=True)
