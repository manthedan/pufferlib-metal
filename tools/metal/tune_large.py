#!/usr/bin/env python3
"""Offline H512/H1024 TN search: spatial tiles only, fixed ascending K and
unchanged split-K boundaries. Every candidate passes independent references
before timing. Does not edit defaults. End-to-end clean A/B decides retention.
"""
import argparse, itertools, json, os, re, subprocess, time
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',default='build/metal/tune_large')
p.add_argument('--linear',action='store_true',help='recurrent NT and dX tiles')
p.add_argument('--extended',action='store_true',help='larger TN tiles')
p.add_argument('--quick',action='store_true',help='only 32/64-row, 64/128-column, 64-thread candidates')
a=p.parse_args();out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=True)
env=dict(os.environ,DEVELOPER_DIR=os.environ.get('DEVELOPER_DIR', '/Applications/Xcode.app/Contents/Developer'))
for key in ['PUF_METAL_GEMM_LARGE_TN','PUF_METAL_GEMM_LARGE_LINEAR','PUF_METAL_WIDE_LIB','PUF_METALLIB']:env.pop(key,None)
def run(cmd,log):
 with log.open('w') as f:return subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT).returncode==0
primary=out/'puffer.metallib'
assert run(['bash','tools/metal/build_metallib.sh',str(primary),'-DPRECISION_FLOAT'],out/'primary.log')
records=[];start=time.monotonic()
choices=([(128,n,s) for n,s in itertools.product([32,64,128],[64,128])]+[(64,256,s) for s in [64,128]]) if a.extended else itertools.product([32,64] if a.quick else ([32,64,128] if a.linear else [16,32,64]),[64,128] if a.quick else [32,64,128],[64] if a.quick else [64,128])
for m,n,threads in choices:
 tag=f'm{m}_n{n}_s{threads}';d=out/tag;d.mkdir(exist_ok=True)
 defs=[f'-DPUF_LARGE_TN_M_F32={m}',f'-DPUF_LARGE_TN_M_BF16={m}',f'-DPUF_LARGE_TN_N={n}',f'-DPUF_LARGE_TN_THREADS={threads}',f'-DPUF_LARGE_TN_N512_F32={n}',f'-DPUF_LARGE_TN_N512_BF16={n}',f'-DPUF_LARGE_TN_N1024={n}']
 if a.linear:defs=[f'-DPUF_LARGE_LINEAR_M={m}',f'-DPUF_LARGE_LINEAR_N={n}',f'-DPUF_LARGE_LINEAR_THREADS={threads}']
 source='tests/metal/test_large_linear.mm' if a.linear else 'tests/metal/test_large_gemm.mm'
 commands=[['xcrun','-sdk','macosx','metal','-std=metal4.0','-O2','-fmetal-math-mode=safe','-fmetal-math-fp32-functions=precise','-Isrc/metal/shaders','-DPRECISION_FLOAT',*defs,'-c','src/metal/shaders/wide.metal','-o',str(d/'wide.air')],
 ['xcrun','-sdk','macosx','metallib',str(d/'wide.air'),'-o',str(d/'puffer.wide.metallib')],
 ['clang++','-std=c++17','-O2','-fobjc-arc','-DPRECISION_FLOAT',*defs,'-Isrc/metal','-Isrc/metal/compat',source,'src/metal/runtime.mm','src/metal/gemm.mm','-framework','Metal','-framework','Foundation',f'-Wl,-sectcreate,__DATA,__metallib,{primary}','-o',str(d/'test')]]
 ok=all(run(cmd,d/f'build{i}.log') for i,cmd in enumerate(commands))
 ok=ok and run([str(d/'test')],d/'check.log')
 rec=dict(tag=tag,exact=ok)
 if ok:
  for h,k in [(512,65536),(1024,4096)]:
   assert run([str(d/'test'),'--bench',str(h),str(k)],d/f'bench{h}.log')
   txt=(d/f'bench{h}.log').read_text()
   if a.linear:rec[str(h)]={f'{stage}_bytes{b}_mode{mode}':float(ms) for stage,b,mode,ms in re.findall(r'(nt|nn_add) H\d+ bytes(\d+) mode(\d) ([\d.]+)ms',txt)}
   else:rec[str(h)]={f'bytes{b}_mode{mode}':float(ms) for b,mode,ms in re.findall(r'bytes(\d+) mode(\d) ([\d.]+)ms',txt)}
 rec['elapsed']=time.monotonic()-start;records.append(rec)
 (out/'results.json').write_text(json.dumps(records,indent=2));print(json.dumps(rec),flush=True)
print(f'Search took {time.monotonic()-start:.2f} seconds',flush=True)
for h,b,stage in itertools.product([512,1024],[4,2],['nt_','nn_add_'] if a.linear else ['']):
 good=[r for r in records if r['exact']]
 if good:
  winner=min(good,key=lambda r:r[str(h)][f'{stage}bytes{b}_mode1'])
  print('Winner',h,b,stage,winner['tag'],winner[str(h)],flush=True)
