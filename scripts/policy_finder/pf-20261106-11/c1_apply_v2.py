# v2: apply the v2 rule to human histories (group 0 view) in a cold process: per-head punishment by round
import sys; sys.path.insert(0,'/opt/algorithmic-institutions/scripts/policy_finder/pf-20261106-11')
sys.path.insert(0,'/opt/algorithmic-institutions/src')
from load import load
import numpy as np, torch as th, yaml
from aimanager.manager.rule import read_rule, run_rule, rule_inputs
df=load()
rule,code=read_rule('/opt/algorithmic-institutions/configs/managers/rule_based/pf-20261106-11-v2.yml','exp2_multi')
games=sorted(df.game.unique())
def arr(col): return th.tensor(np.stack([df[df.game==g].pivot(index='player_id',columns='round_number',values=col).values for g in games]))
C,V,P,G=arr('contribution'),1-arr('player_no_input'),arr('punishment'),arr('group_id')
for pt in [dict(rate=3,delta=0,floor=6,late=0.5,kappa=0),dict(rate=8,delta=2,floor=0,late=0,kappa=1),dict(rate=1,delta=-2,floor=10,late=1,kappa=2)]:
    out=[]
    for t in range(24):
        inp=rule_inputs(C[...,:t+1],V[...,:t+1],P[...,:t],G[...,:t+1],0,'exp2_multi')
        pun=run_rule(code,pt,inp).clamp(0,30).floor()*(G[...,t]==0)
        out.append((pun.sum()/(G[...,t]==0).sum()).item())
    print(pt, 'per head by round', np.round(out,2), 'mean', round(np.mean(out),2))
print('human per head', np.round(df[df.group_id==0].groupby('round_number').punishment.mean().values,2))
