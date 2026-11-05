import torch as th, yaml
cfg = yaml.safe_load(open('/opt/algorithmic-institutions/configs/managers/rule_based/pf-20261106-2.yml'))
def run(t, opp_p, params):
    T1=t+1
    c = th.tensor([0,5,10,20, 0,5,10,20.]).view(1,8,1).repeat(1,1,T1)
    valid = th.ones(1,8,T1); in_g = th.tensor([1,1,1,1,0,0,0,0.]).view(1,8,1).repeat(1,1,T1)
    p = th.zeros(1,8,t); p[:,4:,:] = opp_p
    env = dict(c=c,valid=valid,p=p,in_group=in_g,t=th.tensor(float(t)),th=th, **{k:th.tensor(float(v)) for k,v in params.items()})
    exec(cfg['code'], {}, env); return env['punishment'][0,:4].clamp(0,30).floor().tolist()
P=dict(T=10,s=0.5,d=0.3,L=2,k=0.5)
for t,o in [(0,0),(5,0),(5,4),(7,0),(22,0),(21,4)]: print(t,o,run(t,o,P))
