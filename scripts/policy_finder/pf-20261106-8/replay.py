# replay the rule on the human histories (open loop) to see the punishment it would deal
import sys; sys.path.insert(0,'/opt/algorithmic-institutions/src')
from load import load
import numpy as np, torch as th, yaml
from aimanager.manager.rule import load_rule, run_rule
df=load()
th.clamp(th.ones(2),min=1.0); (th.tensor(1.)<23).float()  # warm torch lazy imports before the builtin-free exec
games=sorted(df.game.unique())
C=np.zeros((len(games),8,24)); V=np.zeros_like(C); P=np.zeros_like(C); G=np.zeros_like(C)
for i,gm in enumerate(games):
    s=df[df.game==gm]
    for r in s.itertuples():
        C[i,r.player_id,r.round_number]=r.contribution; V[i,r.player_id,r.round_number]=1-r.player_no_input
        P[i,r.player_id,r.round_number]=r.punishment; G[i,r.player_id,r.round_number]=r.group_id
path='/opt/algorithmic-institutions/configs/managers/rule_based/pf-20261106-8.yml'
for prm in [dict(slope=1.0,dead_zone=1.0,cap=10.0,horizon=24.0,adapt=0.0),
            dict(slope=1.0,dead_zone=1.0,cap=10.0,horizon=24.0,adapt=0.5),
            dict(slope=0.5,dead_zone=2.0,cap=6.0,horizon=16.0,adapt=0.0),
            dict(slope=2.0,dead_zone=0.0,cap=30.0,horizon=40.0,adapt=1.0)]:
    code,params=load_rule(path,prm,'exp2_multi')
    tot=[];by=[]
    for t in range(24):
        inp=dict(c=th.tensor(C[...,:t+1]).float(),valid=th.tensor(V[...,:t+1]).float(),p=th.tensor(P[...,:t]).float(),
                 in_group=th.tensor(G[...,:t+1]==0).float(),t=th.tensor(float(t)))
        pu=run_rule(code,params,inp).clamp(0,30).floor().numpy()
        m=(G[...,t]==0)
        tot.append(pu[m].mean()); by.append((pu[m]>0).mean())
    print(prm,'mean p/member-round %.2f'%np.mean(tot),'frac punished %.2f'%np.mean(by),'rounds0-3 %.2f'%np.mean(tot[:4]),'rounds 12-23 %.2f'%np.mean(tot[12:]))
