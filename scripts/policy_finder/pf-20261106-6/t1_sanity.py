import torch as th, yaml
r=yaml.safe_load(open('configs/managers/rule_based/pf-20261106-6.yml'))
def run(t, op, prm):
    T=t+1
    c=th.tensor([0,4,10,20, 5,5,5,5.]).view(1,8,1).repeat(1,1,T)
    valid=th.ones(1,8,T); ig=th.tensor([1,1,1,1,0,0,0,0.]).view(1,8,1).repeat(1,1,T)
    p=th.zeros(1,8,t); p[:,4:,:]=op
    env=dict(c=c,valid=valid,p=p,in_group=ig,t=th.tensor(float(t)),th=th,**{k:th.tensor(float(v)) for k,v in prm.items()})
    exec(r['code'],{},env); return env['punishment'].clamp(0,30).floor()
prm=dict(c_thr=8,k=1,m0=1,rho=1,n_end=2)
for t,op in [(0,0),(5,0),(5,3),(22,3),(21,3)]: print(t,op,run(t,op,prm).tolist())
