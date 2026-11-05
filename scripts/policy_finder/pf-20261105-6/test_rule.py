"""Run the rule code on a toy history to check the punishments it deals."""
import torch as th, yaml
r=yaml.safe_load(open('configs/managers/rule_based/pf-20261105-6.yml'))
c=th.tensor([[[0.,0.],[10.,10.],[19.,19.],[20.,20.]]]); v=th.ones_like(c); v[0,1,-1]=0
for pr in [dict(target=20,slope=1,cap=30,half_life=100,quiet=2),dict(target=15,slope=3,cap=5,half_life=6,quiet=3)]:
    for t in [1,10,21,23]:
        env=dict(th=th,c=c,valid=v,p=th.zeros(1,4,1),t=th.tensor(float(t)),**{k:th.tensor(float(x)) for k,x in pr.items()})
        exec(r['code'],env); print(pr,t,env['punishment'].tolist())
