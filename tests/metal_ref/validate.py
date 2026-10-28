#!/usr/bin/env python3
"""Independent CPU math checks and tiny-policy finite-difference gradients."""
import json
from pathlib import Path
import sys
import numpy as np
from golden import load_case,compare

root=Path(sys.argv[1]);results={}
def check(label,actual,expected,atol=3e-6,rtol=3e-5):
    a=np.asarray(actual,dtype=np.float64);b=np.asarray(expected,dtype=np.float64)
    report=compare(a,b,atol=atol,rtol=rtol);results[label]=report
    if report["mismatch_count"]:raise AssertionError(f"{label}: {report}")
def sigmoid(x):return 1/(1+np.exp(-x))
def load(name):return {n:np.asarray(v,dtype=np.float64) for n,v in load_case(root/name).items()}

for path in sorted(root.glob("gemm_*/manifest.json")):
    d=load(path.parent.name);s=json.loads(path.read_text())["scalars"]
    a=d["a"].reshape((-1,d["a"].shape[-1]));b=d["b"].reshape((-1,d["b"].shape[-1]))
    op=s["operation"]
    y=a@b.T if op=="puf_mm" else a.T@b if op=="puf_mm_tn" else a@b
    y=s["alpha"]*y+s["beta"]*d["c_in"]
    check(path.parent.name,d["c_out"],y,atol=8e-5,rtol=1e-4)

for name in ("gate_tiny","gate_breakout"):
    d=load(name);hid,z,s=np.split(d["combined"],3,axis=-1);ht=np.where(hid>=0,hid+.5,sigmoid(hid));h=d["state_in"]+(ht-d["state_in"])*sigmoid(z)
    check(name+"/state",d["state_out"],h);check(name+"/out",d["out"],sigmoid(s)*h+(1-sigmoid(s))*d["x"])

for path in sorted(root.glob("scan_*/manifest.json")):
    name=path.parent.name;d=load(name);B,T,H=d["x"].shape
    h=d["state_in"].copy();cache=np.zeros((B,T,H));out=np.zeros_like(cache)
    for t in range(T):
        h=np.where(d["terminals"][:,t,None]!=0,0,h);cache[:,t]=h
        hidden,z,s=np.split(d["combined"][:,t],3,axis=-1);z,s=sigmoid(z),sigmoid(s);ht=np.where(hidden>=0,hidden+.5,sigmoid(hidden));h=h+(ht-h)*z;out[:,t]=s*h+(1-s)*d["x"][:,t];h=h.astype(np.float32).astype(np.float64)
    check(name+"/scan_h",d["scan_h"],cache);check(name+"/next",d["next_state"],h);check(name+"/out",d["out"],out)
    dh=d["grad_next_state"].copy();dc=np.zeros_like(d["combined"]);dx=np.zeros_like(cache)
    for t in reversed(range(T)):
        hidden,z,s=np.split(d["combined"][:,t],3,axis=-1);z,s=sigmoid(z),sigmoid(s);ht=np.where(hidden>=0,hidden+.5,sigmoid(hidden));hp=d["scan_h"][:,t];h=hp+(ht-hp)*z;go=d["grad_out"][:,t];total=dh+go*s
        dc[:,t]=np.concatenate((total*z*np.where(hidden>=0,1,ht*(1-ht)),total*(ht-hp)*z*(1-z),go*(h-d["x"][:,t])*s*(1-s)),axis=-1);dx[:,t]=go*(1-s);dh=np.where(d["terminals"][:,t,None]!=0,0,total*(1-z))
    check(name+"/grad_combined",d["grad_combined"],dc);check(name+"/grad_input",d["grad_input"],dx);check(name+"/grad_state",d["grad_state"],dh)

for path in sorted(root.glob("adv_*/manifest.json")):
    d=load(path.parent.name);s=json.loads(path.read_text())["scalars"];v,r,t=d["values"],d["rewards"],d["terminals"];imp=d.get("importance",np.ones_like(v));adv=np.zeros_like(v);last=np.zeros(v.shape[0])
    for i in reversed(range(v.shape[1]-1)):
        last=np.minimum(imp[:,i],s["rho_clip"])*(r[:,i+1]+s["gamma"]*v[:,i+1]*(1-t[:,i+1])-v[:,i])+s["gamma"]*s["lambda"]*np.minimum(imp[:,i],s["c_clip"])*last*(1-t[:,i+1]);adv[:,i]=last
    check(path.parent.name+"/adv",d["advantages"],adv);check(path.parent.name+"/returns",d["returns"],adv+v)

for path in sorted(root.glob("muon_*/manifest.json")):
    d=load(path.parent.name);s=json.loads(path.read_text())["scalars"];grad=d["grads"];clip=min(s["max_grad_norm"]/(np.linalg.norm(grad)+s["clip_eps"]),1);g=grad*clip;m=s["momentum"]*d["momentum_in"]+g;update=g+s["momentum"]*m
    for key in sorted(k for k in d if k.startswith("param_") and k.endswith("_in")):
        prefix=key[:-3];off=int(s[prefix+"_offset"]);R,C=d[key].shape;ne=R*C;x=update[off:off+ne].reshape(R,C);x=x/max(np.linalg.norm(x),s["ns_eps"])
        for a,b,c in d["ns_coefficients"]:
            gram=x.T@x if R>C else x@x.T;poly=b*gram+c*gram@gram;x=a*x+(x@poly if R>C else poly@x)
        update[off:off+ne]=x.ravel()*np.sqrt(max(1,R/C))
    check(path.parent.name+"/momentum",d["momentum_out"],m)
    check(path.parent.name+"/params",d["params_out"],d["params_in"]-s["lr"]*update,atol=2e-5,rtol=2e-4)

# Check PPO gradients by differentiating a scalar CPU loss (no CUDA derivative code).
d=load("ppo_tiny");s=json.loads((root/"ppo_tiny/manifest.json").read_text())["scalars"]
B,T,A=d["mask"].shape[0]//int(s["T"]),int(s["T"]),3
mask=d["mask"].reshape(-1,A);actions=d["actions"].reshape(-1).astype(int)
def loss(dec):
    flat=dec.reshape(-1,A+1);z=np.where(mask!=0,flat[:,:A],-1e4);mx=z.max(-1);lse=mx+np.log(np.exp(z-mx[:,None]).sum(-1));lp=z[np.arange(len(z)),actions]-lse
    ratio=np.exp(lp-d["old_logprobs"].ravel());adv=d["advantages"].ravel();pg=np.maximum(-adv*ratio,-adv*np.clip(ratio,1-s["clip_coef"],1+s["clip_coef"]))
    v=flat[:,A];old=d["old_values"].ravel();ret=d["returns"].ravel();vc=old+np.clip(v-old,-s["vf_clip_coef"],s["vf_clip_coef"]);vf=.5*np.maximum((v-ret)**2,(vc-ret)**2);logps=z-lse[:,None];ent=-(np.exp(logps)*logps).sum(-1)
    return (pg+s["vf_coef"]*vf-s["ent_coef"]*ent).mean()
def finite_gradient(fn,x,eps=1e-5):
    g=np.zeros_like(x)
    for idx in np.ndindex(x.shape):
        before=x[idx];x[idx]=before+eps;plus=fn(x);x[idx]=before-eps;minus=fn(x);x[idx]=before;g[idx]=(plus-minus)/(2*eps)
    return g
grad=finite_gradient(loss,d["decoder"].copy());check("ppo_tiny/dlogits_finite_difference",d["dlogits"],grad[...,:A]);check("ppo_tiny/dvalue_finite_difference",d["dvalue"].reshape(B,T),grad[...,A]);check("ppo_tiny/total_loss",d["losses"][3:4],np.array([loss(d["decoder"])]),atol=2e-5)

# Entire tiny recurrent network: finite differences for every parameter.
d=load("policy_tiny");s=json.loads((root/"policy_tiny/manifest.json").read_text())["scalars"];L=int(s["layers"]);B,T,I=d["obs"].shape;H=int(s["H"]);params=[d[f"param_{i}"] .copy() for i in range(L+2)]
def forward():
    x=d["obs"]@params[0].T
    for i in range(L):
        combined=x@params[i+2].T;out=np.zeros_like(x);h=d["initial_state"][i,1:1+B].copy()
        for t in range(T):
            h=np.where(d["terminals"][:,t,None]!=0,0,h);hid,z,p=np.split(combined[:,t],3,axis=-1);z,p=sigmoid(z),sigmoid(p);ht=np.where(hid>=0,hid+.5,sigmoid(hid));h=h+(ht-h)*z;out[:,t]=p*h+(1-p)*x[:,t]
        x=out
    return x@params[1].T
check("policy_tiny/forward",d["decoder"],forward())
def objective(_):
    out=forward();return (out[...,:3]*d["dlogits"]).sum()+(out[...,3]*d["dvalue"][...,0]).sum()
for i,param in enumerate(params):check(f"policy_tiny/param_{i}_finite_difference",d[f"param_{i}_grad"],finite_gradient(objective,param),atol=3e-8,rtol=3e-4)

# Sampling and ABI checks; uniform draws come from a cloned real cuRAND state.
for name in ("sample_tiny","sample_breakout"):
    d=load(name);check(name+"/env_actions",d["actions"],d["env_actions"],0,0);check(name+"/value",d["values"],d["decoder"][:,-1],0,0)
    assert ((d["uniform_draws"]>0)&(d["uniform_draws"]<=1)).all()
    for i,a in enumerate(d["actions"].ravel().astype(int)):
        if d["masks"][i].any():assert d["masks"][i,a]!=0
    fields=d["rng_fields_in"];assert len(set(fields[:,10].astype(int)))>1

# Counter comparisons must preserve differences above float64 exact-integer range.
r=compare(np.array([2**63],dtype=np.uint64),np.array([2**63+1],dtype=np.uint64))
assert r["max_abs"]==1 and r["max_rel"]>0 and r["first_mismatching_index"]==(0,)
assert compare(np.array([1.]),np.array([np.inf]))["mismatch_count"]==1
assert compare(np.array([np.nan]),np.array([np.nan]))["mismatch_count"]==0

out=root.parent/(root.name+"_validation.json");out.write_text(json.dumps(results,indent=2)+"\n");print(f"PASS: {len(results)} independent numerical checks; {out}")
