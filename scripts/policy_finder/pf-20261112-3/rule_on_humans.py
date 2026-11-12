"""Run the rule over the human games' histories at sample design points: how much it punishes."""
import sys; sys.path.insert(0,'scripts/policy_finder/pf-20261112-3'); sys.path.insert(0,'src')
import numpy as np, torch as th
from human_load import load
from aimanager.manager.rule import load_rule, sobol_design
import inspect
d=load()
games=sorted(d.game.unique())
C=np.zeros((len(games),8,24)); V=np.zeros_like(C); G=np.zeros_like(C); P=np.zeros_like(C)
for i,gm in enumerate(games):
    x=d[d.game==gm]
    for r in x.itertuples():
        C[i,r.player_id,r.round_number]=r.contribution; V[i,r.player_id,r.round_number]=1-r.player_no_input
        G[i,r.player_id,r.round_number]=(r.group_id==0); P[i,r.player_id,r.round_number]=r.punishment
code=open('configs/managers/rule_based/pf-20261112-3.yml').read().split('code: |\n')[1]
code='\n'.join(l[2:] for l in code.splitlines())
def run(params):
    tot=0; cnt=0
    for t in range(24):
        env=dict(th=th,c=th.tensor(C[...,:t+1]),valid=th.tensor(V[...,:t+1]),in_group=th.tensor(G[...,:t+1]),p=th.tensor(P[...,:t]),t=th.tensor(float(t)))
        env.update({k:th.tensor(float(v)) for k,v in params.items()})
        exec(code,env)
        pun=env['punishment'].clamp(0,30).floor()*th.tensor(G[...,t]*V[...,t])
        tot+=pun.sum().item(); cnt+=(G[...,t]*V[...,t]).sum()
    return tot/cnt
for prm in [dict(gap0=0,k=3,cap=15,w=4),dict(gap0=4,k=0.5,cap=5,w=2),dict(gap0=8,k=0.3,cap=3,w=1),dict(gap0=2,k=0.1,cap=5,w=4),dict(gap0=0,k=0.03,cap=15,w=4)]:
    print(prm, 'mean punishment per member-round', round(run(prm),3))
print('human managers', round((P*G*V).sum()/(G*V).sum(),3))
