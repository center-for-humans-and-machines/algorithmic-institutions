# Replay the rule on the human histories (group 0's seat) at a mid design point; compare with human punishment.
import sys; sys.path.insert(0,'scripts/policy_finder/pf-20261106-10')
import numpy as np, torch as th, yaml
from load import load
df=load()
games=sorted(df.game.unique())
C=np.zeros((len(games),8,24)); V=np.zeros_like(C); P=np.zeros_like(C); G=np.zeros_like(C)
for i,gm in enumerate(games):
    d=df[df.game==gm]
    for r in d.itertuples():
        C[i,r.player_id,r.round_number]=r.contribution; V[i,r.player_id,r.round_number]=1-r.player_no_input
        P[i,r.player_id,r.round_number]=r.punishment; G[i,r.player_id,r.round_number]=(r.group_id==0)
rule=yaml.safe_load(open('configs/managers/rule_based/pf-20261106-10.yml'))
params=dict(slope=float(sys.argv[1]) if len(sys.argv)>1 else 0.7, tol=2.0, cap=6.0, base=1.0, ratio=0.6)
T=lambda x: th.tensor(x,dtype=th.float)
out=[]
for t in range(24):
    env=dict(th=th, c=T(C[...,:t+1]), valid=T(V[...,:t+1]), p=T(P[...,:t]), in_group=T(G[...,:t+1]), t=th.tensor(float(t)))
    env.update({k:th.tensor(v) for k,v in params.items()})
    exec(rule['code'], env)
    pun=env['punishment'].clamp(0,30).floor().numpy()*G[...,t]
    out.append(pun)
out=np.stack(out,-1)
mem=G.sum()
print('rule pun per member-round', out.sum()/mem, 'human (group 0)', (P*G).sum()/mem)
print('rule per round', (out.sum((0,1))/G.sum((0,1))).round(2))
print('share punished', (out[G==1]>0).mean().round(3), 'max', out.max())
