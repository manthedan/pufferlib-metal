#!/usr/bin/env python3
"""Order-preserving Muon spatial tile search using production-shaped batches.
Independent mixed/ragged stage tests must pass before timing. --bench contains
only wrapper dispatches, so direct reference timings cannot bias selection.
Source/defaults are never edited. End-to-end clean A/B decides retention.
"""
import argparse, itertools, json, os, re, subprocess, time
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('kind',choices=['muon','products'])
p.add_argument('--boundary',action='store_true',help='larger output tiles, at most 128 float accumulators per thread')
p.add_argument('--extended',action='store_true',help='128-row tiles with 32/64 columns')
p.add_argument('--precision',choices=['f32','bf16'])
p.add_argument('--output',default='build/metal/tune_large_stages');a=p.parse_args()
out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=True)
env=dict(os.environ,DEVELOPER_DIR=os.environ.get('DEVELOPER_DIR', '/Applications/Xcode.app/Contents/Developer'))
for key in ['PUF_METAL_WIDE_LIB','PUF_METAL_MUON_LARGE','PUF_METAL_MUON_WIDE_MMA','PUF_METAL_KPROF']:env.pop(key,None)
def run(cmd,log,prof=False):
 with log.open('w') as f:return subprocess.run(cmd,env=dict(env,**({'PUF_METAL_KPROF':'1'} if prof else {})),stdout=f,stderr=subprocess.STDOUT).returncode==0
records=[];start=time.monotonic()
for precision in ([a.precision] if a.precision else ['f32','bf16']):
 flags=['-DPRECISION_FLOAT'] if precision=='f32' else []
 d0=out/precision;d0.mkdir(exist_ok=True);primary=d0/'puffer.metallib'
 assert run(['bash','tools/metal/build_metallib.sh',str(primary),*flags],d0/'primary.log')
 prefix='PUF_LARGE_MUON_'+('GRAM' if a.kind=='muon' else 'PRODUCT')
 candidates=[(128,128,128),(256,32,64),(256,32,128),(256,64,128),(512,32,128),(64,256,128)] if a.boundary else itertools.product([128] if a.extended else [32,64],[32,64] if a.extended else [32,64,128],[64,128])
 for m,n,threads in candidates:
  tag=f'{precision}_m{m}_n{n}_s{threads}';d=out/tag;d.mkdir(exist_ok=True)
  defines={prefix+'_M':m,prefix+'_N':n,prefix+'_THREADS':threads};defs=[f'-D{k}={v}' for k,v in defines.items()]
  commands=[['xcrun','-sdk','macosx','metal','-std=metal4.0','-O2','-fmetal-math-mode=safe','-fmetal-math-fp32-functions=precise','-Isrc/metal/shaders',*flags,*defs,'-c','src/metal/shaders/wide.metal','-o',str(d/'wide.air')],
   ['xcrun','-sdk','macosx','metallib',str(d/'wide.air'),'-o',str(d/'puffer.wide.metallib')],
   ['clang++','-std=c++17','-O2','-fobjc-arc',*flags,*defs,'-Isrc/metal','-Isrc/metal/compat','tests/metal/test_large_muon.mm','src/metal/runtime.mm','src/metal/gemm.mm','-framework','Metal','-framework','Foundation',f'-Wl,-sectcreate,__DATA,__metallib,{primary}','-o',str(d/'test')]]
  ok=all(run(cmd,d/f'build{i}.log') for i,cmd in enumerate(commands))
  ok=ok and run([str(d/'test')],d/'check.log')
  rec=dict(tag=tag,exact=ok,defines=defines)
  if ok:
   for h in [512,1024]:
    assert run([str(d/'test'),'--bench',str(h)],d/f'bench{h}.log',True)
    txt=(d/f'bench{h}.log').read_text();timings={}
    for stage in ['gram','gram_buf','update']:
     match=re.search(r'^wide_large_muon_'+stage+r'\s+\d+\s+[\d.]+\s+([\d.]+)',txt,re.M)
     assert match,(tag,h,stage);timings[stage+'_us']=float(match[1])
    rec[str(h)]=timings
  rec['elapsed']=time.monotonic()-start;records.append(rec)
  (out/'results.json').write_text(json.dumps(records,indent=2));print(json.dumps(rec),flush=True)
print(f'Search took {time.monotonic()-start:.2f} seconds',flush=True)
for precision,h in itertools.product(['f32','bf16'],[512,1024]):
 good=[r for r in records if r['exact'] and r['tag'].startswith(precision)]
 if good:
  def score(r):
   v=r[str(h)];return v['gram_us'] if a.kind=='muon' else v['gram_buf_us']+v['update_us']
  winner=min(good,key=score)
  print('Winner',precision,h,winner['tag'],winner[str(h)],flush=True)
