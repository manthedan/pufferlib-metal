#!/usr/bin/env python3
"""Private shader/host mutants must fail independent numerical or dispatch
checks; restored binaries must pass. Run the large unit gates first.
"""
from pathlib import Path
import argparse, os, subprocess as sp
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--component',choices=['all','tn','muon','linear'],default='all');args=p.parse_args()
root=Path.cwd();out=root/'build/metal/large_mutations';out.mkdir(parents=True,exist_ok=True)
env=dict(os.environ,DEVELOPER_DIR=os.environ.get('DEVELOPER_DIR', '/Applications/Xcode.app/Contents/Developer'))
source=Path('src/metal/shaders/wide.metal').read_text().replace('#include "wide_muon.h"',Path('src/metal/shaders/wide_muon.h').read_text())
source=source.replace('#include "large_linear.h"',Path('src/metal/shaders/large_linear.h').read_text())
source=source.replace('#include "large_muon_products.h"',Path('src/metal/shaders/large_muon_products.h').read_text())
mutations=[]
if args.component in ['all','tn']:
 for label,size in [('f32',4),('bf16',2)]:
  for stage,body,tile in [('tn512','wide_gemm_tn_body','N512'),('tn1024','wide_gemm_tn_body','PUF_LARGE_TN_N1024'),('ragged','wide_gemm_tn_body','PUF_LARGE_TN_N'),('split','wide_tn_split_body','PUF_LARGE_TN_N')]:
   old=f'{body}<TYPE, TM, {tile}, PUF_LARGE_TN_THREADS>(a, tg);'
   mutations.append((label+'_'+stage,'f32','test_large_gemm','test_large_gemm',old,f'{{ if (sizeof(TYPE) != {size}) '+old+' }'))
 old='    }\n}\n#define WIDE_GEMM'
 new='    }\n    if (tg.x == 0 && tg.y == 0) puf_st(a.C, (long)a.M * a.N, NAN);\n}\n#define WIDE_GEMM'
 mutations.append(('tn_edge_oob_nan','f32','test_large_gemm','test_large_gemm',old,new))
if args.component in ['all','muon']:
 for precision in ['f32','bf16']:
  for stage,kind in [('gram_buf','false'),('update','true')]:
   old=f'large_muon_product<{kind}>(a,tg,tid);'
   mutations.append((precision+'_large_'+stage,precision,'test_large_muon_'+precision,'test_large_muon',old,'/* skipped product */'))
  old='kernel void wide_large_muon_gram(constant MuonWideArgs& a [[buffer(0)]],\n        uint3 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {'
  mutations.append((precision+'_large_gram',precision,'test_large_muon_'+precision,'test_large_muon',old,old+' return;'))
  old='device precision_t* dst = UPDATE ? (a.step % 2 == 0 ? m.xb : m.x) : m.gb;\n    device precision_t* old = UPDATE ? src : m.g;'
  new='device precision_t* dst = UPDATE ? (a.step % 2 == 0 ? m.xb : m.x) : m.gb;\n    if (UPDATE && a.step == 4 && tg.x == 0 && tid == 0) dst[rows * N] = from_float(NAN);\n    device precision_t* old = UPDATE ? src : m.g;'
  mutations.append((precision+'_edge_oob_nan',precision,'test_large_muon_'+precision,'test_large_muon',old,new))
if args.component in ['all','linear']:
 for label,size in [('f32',4),('bf16',2)]:
  for stage in ['nt','nn_add']:
   old=f'large_{stage}_body<TYPE,PUF_LARGE_LINEAR_M,PUF_LARGE_LINEAR_N,PUF_LARGE_LINEAR_THREADS>(a,tg);'
   mutations.append((label+'_'+stage,'f32','test_large_linear','test_large_linear',old,f'if(sizeof(TYPE)!={size}) '+old))
 old='if constexpr(sizeof(T)==2)prod=puf_bf16_to_f32(puf_f32_to_bf16(prod));'
 mutations.append(('bf16_add_rounding','f32','test_large_linear','test_large_linear',old,''))
 old='large_nt_body<TYPE,PUF_LARGE_LINEAR_M,PUF_LARGE_LINEAR_N,PUF_LARGE_LINEAR_THREADS>(a,tg);'
 new=old+' puf_st(a.C,(long)a.M*a.N,NAN);'
 mutations.append(('linear_edge_oob_nan','f32','test_large_linear','test_large_linear',old,new))
for name,precision,folder,exe,old,new in mutations:
 assert source.count(old)==1,(name,old)
 d=out/name;d.mkdir(exist_ok=True);(d/'mut.metal').write_text(source.replace(old,new))
 flags=['-DPRECISION_FLOAT'] if precision=='f32' else []
 commands=[['xcrun','-sdk','macosx','metal','-std=metal4.0','-O2','-fmetal-math-mode=safe','-fmetal-math-fp32-functions=precise','-Isrc/metal/shaders',*flags,'-c',str(d/'mut.metal'),'-o',str(d/'mut.air')],
 ['xcrun','-sdk','macosx','metallib',str(d/'mut.air'),'-o',str(d/'mut.metallib')]]
 with (d/'build.log').open('w') as f:
  for cmd in commands:sp.run(cmd,env=env,stdout=f,stderr=sp.STDOUT,check=True)
 base=root/'build/metal'/folder;binary=base/exe
 with (d/'test.log').open('w') as f:
  r=sp.run([str(binary)],env=dict(env,PUF_METAL_WIDE_LIB=str(d/'mut.metallib')),stdout=f,stderr=sp.STDOUT,timeout=120)
 assert r.returncode==1 and 'FAIL' in (d/'test.log').read_text(),(name,r.returncode)
 with (d/'restored.log').open('w') as f:
  sp.run([str(binary)],env=dict(env,PUF_METAL_WIDE_LIB=str(base/'puffer.wide.metallib')),stdout=f,stderr=sp.STDOUT,check=True)
 print(name,'CAUGHT; RESTORED PASS',flush=True)
count=len(mutations)
if args.component in ['all','muon']:
 host=Path('src/metal/gemm.mm').read_text()
 old='use_large_gram |= (a.mats[i].tops & 4) != 0;';assert host.count(old)==1
 for precision in ['f32','bf16']:
  name=precision+'_large_gram_selector';d=out/name;d.mkdir(exist_ok=True)
  (d/'gemm.mm').write_text(host.replace(old,'use_large_gram = false;'))
  base=root/'build/metal'/('test_large_muon_'+precision)
  flags=['-DPRECISION_FLOAT'] if precision=='f32' else []
  cmd=['clang++','-std=c++17','-O2','-fobjc-arc',*flags,'-Isrc/metal','-Isrc/metal/compat',
   'tests/metal/test_large_muon.mm','src/metal/runtime.mm',str(d/'gemm.mm'),'-framework','Metal','-framework','Foundation',
   f'-Wl,-sectcreate,__DATA,__metallib,{base}/puffer.metallib','-o',str(d/'test')]
  with (d/'build.log').open('w') as f:sp.run(cmd,env=env,stdout=f,stderr=sp.STDOUT,check=True)
  test_env=dict(env,PUF_METAL_WIDE_LIB=str(base/'puffer.wide.metallib'))
  with (d/'test.log').open('w') as f:
   r=sp.run(['python3','tests/metal/check_large_muon_dispatch.py',str(d/'test'),'--log',str(d/'dispatch.log')],env=test_env,stdout=f,stderr=sp.STDOUT)
  assert r.returncode and 'large wrapper kernel selection' in (d/'test.log').read_text(),name
  with (d/'restored.log').open('w') as f:
   sp.run(['python3','tests/metal/check_large_muon_dispatch.py',str(base/'test_large_muon'),'--log',str(d/'restored_dispatch.log')],env=test_env,stdout=f,stderr=sp.STDOUT,check=True)
  count+=1;print(name,'CAUGHT; RESTORED PASS',flush=True)
if args.component in ['all','muon']:
 host=Path('src/metal/gemm.mm').read_text()
 cases=[
  ('stage_in','cudaMemcpyAsync(staged[i], mats[i].x, staged_bytes[i], cudaMemcpyDeviceToDevice, s);','(void)0;','FAIL'),
  ('stage_out','cudaMemcpyAsync(mats[i].x, staged[i], staged_bytes[i], cudaMemcpyDeviceToDevice, s);','(void)0;','FAIL'),
  ('stage_selector','const bool stage = !aligned16(m.x) && muon_large_shape(m.R, m.C);','const bool stage = false;','large aligned staging selection'),
  ('scratch_budget','for (int i = 0; i < count; ++i) if (staged[i])\n        cudaMemcpyAsync(mats[i].x, staged[i], staged_bytes[i], cudaMemcpyDeviceToDevice, s);\n    return true;',
   'for (int i = 0; i < count; ++i) if (staged[i])\n        cudaMemcpyAsync(mats[i].x, staged[i], staged_bytes[i], cudaMemcpyDeviceToDevice, s);\n    if (aligned16(mats[0].x)) ((unsigned char*)scratch)[scratch_bytes] = 0;\n    return true;','FAIL')]
 for precision in ['f32','bf16']:
  base=root/'build/metal'/('test_large_muon_'+precision)
  flags=['-DPRECISION_FLOAT'] if precision=='f32' else []
  for stage,old,new,reason in cases:
   assert host.count(old)==1
   name=precision+'_'+stage;d=out/name;d.mkdir(exist_ok=True)
   (d/'gemm.mm').write_text(host.replace(old,new))
   cmd=['clang++','-std=c++17','-O2','-fobjc-arc',*flags,'-Isrc/metal','-Isrc/metal/compat',
    'tests/metal/test_large_muon.mm','src/metal/runtime.mm',str(d/'gemm.mm'),'-framework','Metal','-framework','Foundation',
    f'-Wl,-sectcreate,__DATA,__metallib,{base}/puffer.metallib','-o',str(d/'test')]
   with (d/'build.log').open('w') as f:sp.run(cmd,env=env,stdout=f,stderr=sp.STDOUT,check=True)
   test_env=dict(env,PUF_METAL_WIDE_LIB=str(base/'puffer.wide.metallib'))
   with (d/'test.log').open('w') as f:
    r=sp.run(['python3','tests/metal/check_large_muon_dispatch.py',str(d/'test'),'--log',str(d/'dispatch.log')],env=test_env,stdout=f,stderr=sp.STDOUT)
   assert r.returncode and reason in (d/'test.log').read_text(),name
   with (d/'restored.log').open('w') as f:
    sp.run(['python3','tests/metal/check_large_muon_dispatch.py',str(base/'test_large_muon'),'--log',str(d/'restored_dispatch.log')],env=test_env,stdout=f,stderr=sp.STDOUT,check=True)
   count+=1;print(name,'CAUGHT; RESTORED PASS',flush=True)
print('ALL',count,'CAUGHT; production source unchanged')
