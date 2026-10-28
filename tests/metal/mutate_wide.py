#!/usr/bin/env python3
"""Mutation-check every wide entry point without editing production source.
Run the three run_test_wide*.sh unit gates first (MinGRU, GEMM and Muon).
Shader mutants use separate auxiliary libraries; NaN-poisoned independent
references and canaries must reject them. Host selector mutants compile a
separate gemm.mm copy and must fail the wrapper kernel-count assertion.
The review regressions also rerun the original after each caught mutation.
"""
from pathlib import Path
import argparse, os, subprocess as sp
root=Path.cwd();out=root/'build/metal/wide_mutations';out.mkdir(exist_ok=True)
source=Path('src/metal/shaders/wide.metal').read_text().replace('#include "wide_muon.h"',Path('src/metal/shaders/wide_muon.h').read_text())
env=dict(os.environ,DEVELOPER_DIR=os.environ.get('DEVELOPER_DIR', '/Applications/Xcode.app/Contents/Developer'))
mutations=[]
for p in ['f32','bf16']:
 for H in [128,256]:
  mutations += [(f'{p}_forward{H}',p,'test_wide_'+p+'/test_wide_mingru',['--check',str(H)],'a.combined[(long)(row0 + idx[1]) * H3 + n0 + idx[0]] = v;',f'if (H != {H}) a.combined[(long)(row0 + idx[1]) * H3 + n0 + idx[0]] = v;')]
  mutations += [(f'{p}_backward{H}',p,'test_wide_'+p+'/test_wide_mingru',['--check',str(H)],'a.grad_combined[c + 2 * H] = from_float(grad_proj);',f'if (H != {H}) a.grad_combined[c + 2 * H] = from_float(grad_proj);')]
 for kind in ['false','true']:
  mutations += [(f'{p}_muon_{kind}',p,'test_wide_muon_'+p+'/test_wide_muon',[],f'muon_wide_mma(a, tg, tid, {kind});','/* skipped */')]
 mutations += [(f'{p}_muon_ragged',p,'test_wide_muon_'+p+'/test_wide_muon',[],' && m.R % 16 == 0 && m.C % 16 == 0','')]
 for kind,old in [('fcarry','else carry[h / THREADS] = h_t;'),('bcarry','else carry[h / THREADS] = dh;')]:
  mutations += [(p+'_'+kind,p,'test_wide_'+p+'/test_wide_mingru',['--check','256'],old,'else carry[h / THREADS] = 0.0f;')]
for p,T in [('f32','float'),('bf16','bfloat')]:
 # Precision-specific entry point skips its body entirely; poison catches it.
 old=f'WIDE_GEMM({p}, {T}, PUF_WIDE_TN_M_{p.upper()})'
 new=f'kernel void wide_gemm_tn_{p}(constant WideGemmArgs<{T}>& a [[buffer(0)]], uint2 tg [[threadgroup_position_in_grid]]) {{ }}'
 mutations += [(p+'_gemm','f32','test_wide_gemm/test_wide_gemm',[],old,new)]
for p,T in [('f32','float'),('bf16','bfloat')]:
 old=f'WIDE_TN_SPLIT({p}, {T}, PUF_WIDE_TN_M_{p.upper()})'
 new=f'kernel void wide_gemm_tn_split_{p}(constant WideTnSplitArgs& a [[buffer(0)]], uint3 tg [[threadgroup_position_in_grid]]) {{ }}'
 mutations += [(p+'_tn_split','f32','test_wide_gemm/test_wide_gemm',[],old,new)]
host_mutations=[(f'{p}_muon_dispatch',p) for p in ['f32','bf16']]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--only',nargs='+',choices=[m[0] for m in mutations+host_mutations],help='run only these mutations')
selected=parser.parse_args().only
if selected:
 mutations=[m for m in mutations if m[0] in selected]
 host_mutations=[m for m in host_mutations if m[0] in selected]
for name,p,binary,args,old,new in mutations:
 assert old in source,(name,old)
 d=out/name;d.mkdir(exist_ok=True);src=d/'mut.metal';src.write_text(source.replace(old,new))
 flags=['-DPRECISION_FLOAT'] if p=='f32' else []
 cmds=[['xcrun','-sdk','macosx','metal','-std=metal4.0','-O2','-fmetal-math-mode=safe','-fmetal-math-fp32-functions=precise','-Isrc/metal/shaders',*flags,'-c',str(src),'-o',str(d/'mut.air')],['xcrun','-sdk','macosx','metallib',str(d/'mut.air'),'-o',str(d/'mut.metallib')]]
 with open(d/'build.log','w') as f:
  for cmd in cmds:sp.run(cmd,stdout=f,stderr=sp.STDOUT,env=env,check=True)
 with open(d/'test.log','w') as f:r=sp.run([str(root/'build/metal'/binary),*args],stdout=f,stderr=sp.STDOUT,env=dict(env,PUF_METAL_WIDE_LIB=str(d/'mut.metallib')))
 print(name,'CAUGHT' if r.returncode else 'SURVIVED',flush=True)
 if not r.returncode:raise SystemExit(1)
 if name.endswith('_muon_ragged'):
  # Require the named failures, not just an unrelated process error.
  log=(d/'test.log').read_text()
  assert r.returncode==1 and 'stage=update shape=256x257 differences=256' in log and 'stage=update buffer=18 guard_corruption' in log,log
  original=root/'build/metal'/Path(binary).parent/'puffer.wide.metallib'
  with open(d/'restored.log','w') as f:
   sp.run([str(root/'build/metal'/binary),*args],stdout=f,stderr=sp.STDOUT,
          env=dict(env,PUF_METAL_WIDE_LIB=str(original)),check=True)
  print(name,'RESTORED PASS',flush=True)
host_source=Path('src/metal/gemm.mm').read_text()
for name,p in host_mutations:
 d=out/name;d.mkdir(exist_ok=True)
 old='if (wide_mma) for (int i = 0; i < count; ++i)'
 assert host_source.count(old)==1
 src=d/'mut_gemm.mm';src.write_text(host_source.replace(old,'if (false && wide_mma) for (int i = 0; i < count; ++i)'))
 original=root/'build/metal'/('test_wide_muon_'+p)
 binary=d/'test_wide_muon'
 flags=['-DPRECISION_FLOAT'] if p=='f32' else []
 cmd=['clang++','-std=c++17','-O2','-fobjc-arc','-Wall',*flags,
      '-Isrc/metal','-Isrc/metal/compat','tests/metal/test_wide_muon.mm',
      'src/metal/runtime.mm',str(src),'-framework','Metal','-framework','Foundation',
      '-Wl,-sectcreate,__DATA,__metallib,'+str(original/'puffer.metallib'),'-o',str(binary)]
 with open(d/'build.log','w') as f:sp.run(cmd,stdout=f,stderr=sp.STDOUT,env=env,check=True)
 check=['python3','tests/metal/check_wide_muon_dispatch.py']
 test_env=dict(env,PUF_METAL_WIDE_LIB=str(original/'puffer.wide.metallib'))
 with open(d/'test.log','w') as f:
  r=sp.run([*check,str(binary),'--log-dir',str(d/'dispatch')],stdout=f,stderr=sp.STDOUT,env=test_env)
 print(name,'CAUGHT' if r.returncode else 'SURVIVED',flush=True)
 assert r.returncode and 'wrapper kernel selection enabled' in (d/'test.log').read_text(),d/'test.log'
 with open(d/'restored.log','w') as f:
  sp.run([*check,str(original/'test_wide_muon'),'--log-dir',str(d/'restored_dispatch')],
         stdout=f,stderr=sp.STDOUT,env=test_env,check=True)
 print(name,'RESTORED PASS',flush=True)
print('ALL',len(mutations)+len(host_mutations),'CAUGHT; production source unchanged')
