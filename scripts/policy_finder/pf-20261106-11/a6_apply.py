# apply the rule to human histories (group 0's view) to see punishment levels per head
import sys; sys.path.insert(0,'/opt/algorithmic-institutions/scripts/policy_finder/pf-20261106-11')
sys.path.insert(0,'/opt/algorithmic-institutions/src')
from load import load
import numpy as np, torch as th, yaml
from aimanager.manager.rule import read_rule, run_rule, rule_inputs
df=load()
rule,code=read_rule('/opt/algorithmic-institutions/configs/managers/rule_based/pf-20261106-11.yml','exp2_multi')
games=sorted(df.game.unique())
def arr(col): return th.tensor(np.stack([df[df.game==g].pivot(index='player_id',columns='round_number',values=col).values for g in games]))
C,V,P,G=arr('contribution'),1-arr('player_no_input'),arr('punishment'),arr('group_id')
for pt in [dict(rate=0.5,w=0.5,anchor=10,lam=0.3,h=2),dict(rate=0.3,w=1,anchor=10,lam=0,h=2),dict(rate=1,w=0.7,anchor=12,lam=0.5,h=4)]:
    out=[]
    for t in range(24):
        inp=rule_inputs(C[...,:t+1],V[...,:t+1],P[...,:t],G[...,:t+1],0,'exp2_multi')
        pun=run_rule(code,pt,inp).clamp(0,30).floor()*(G[...,t]==0)
        out.append((pun.sum()/(G[...,t]==0).sum()).item())
    print(pt, 'per head by round', np.round(out,2), 'mean', round(np.mean(out),2))
print('human per head', np.round(df[df.group_id==0].groupby('round_number').punishment.mean().values,2))
