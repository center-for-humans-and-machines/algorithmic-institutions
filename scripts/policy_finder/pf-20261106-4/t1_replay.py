"""Replay the rule on human game histories (group 0 seat): how much would it punish, whom, and when."""
import sys; sys.path.insert(0,'/opt/algorithmic-institutions/scripts/policy_finder/pf-20261106-4')
from load import load
import numpy as np, torch as th
from aimanager.manager.rule import load_rule, rule_inputs, run_rule
df, gs = load()
games = sorted(df.game.unique())
def arr(col):
    return th.tensor(np.stack([df[df.game==g].pivot(index='player_id', columns='round_number', values=col).values for g in games]))
C, V, P, G = arr('contribution'), 1-arr('player_no_input'), arr('punishment'), arr('group_id')
rule = '/opt/algorithmic-institutions/configs/managers/rule_based/pf-20261106-4.yml'
for params in [dict(k=0.5,t_stop=12,quiet=1,cap0=2.0,rel=0.75), dict(k=1.0,t_stop=20,quiet=0,cap0=8.0,rel=1.5), dict(k=0.0,t_stop=20,quiet=0,cap0=8.0,rel=1.5)]:
    code, prm = load_rule(rule, params, setting='exp2_multi')
    tot = []; per_round=[]
    for t in range(24):
        inp = rule_inputs(C[...,:t+1], V[...,:t+1], P[...,:t], G[...,:t+1], 0, setting='exp2_multi')
        pun = run_rule(code, prm, inp).clamp(0,30).floor() * (G[...,t]==0) * V[...,t]
        per_round.append(pun.sum().item()/ max((G[...,t]==0).sum().item(),1))
    print(params, 'mean punishment per member by round', np.round(per_round,2))
