# v2: replay the v2 rule (cold process) on the human histories: mean punishment per member of group 0 per round
import sys; sys.path.insert(0, '/opt/algorithmic-institutions/scripts/policy_finder/pf-20261106-7')
from load import load
import numpy as np, torch as th
from aimanager.manager.rule import read_rule, run_rule, rule_inputs
rule, code = read_rule('/opt/algorithmic-institutions/configs/managers/rule_based/pf-20261106-7-v2.yml', 'exp2_multi')
df = load()
games = sorted(df.game.unique())
def arr(col):
    return th.tensor(np.stack([df[df.game==g].pivot(index='player_id',columns='round_number',values=col).values for g in games])).float()
C, P, G, V = arr('contribution'), arr('punishment'), arr('group_id'), 1-arr('player_no_input')
P_own = th.zeros_like(P)  # punishment history if my rule had dealt group 0's (humans' contributions kept)
for params in [dict(c0=11,k=3.0,horizon=20.0,budget=20.0,dec=0.3), dict(c0=10,k=1.0,horizon=12.0,budget=5.0,dec=1.0), dict(c0=14,k=8.0,horizon=40.0,budget=100.0,dec=0.0)]:
    pass
    out, Pm = [], P.clone()
    for t in range(24):
        inp = rule_inputs(C[...,:t+1], V[...,:t+1], Pm[...,:t], G[...,:t+1], th.zeros(len(games),1,1,dtype=th.long), 'exp2_multi')
        pu = run_rule(code, params, inp).clamp(0,30).floor()
        m = inp['in_group'][...,-1]
        Pm[..., t] = th.where(m > 0, pu, P[..., t])
        out.append(float((pu*m).sum()/m.sum()))
    print(params, np.round(out,1))
