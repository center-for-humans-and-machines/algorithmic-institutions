# v4: replay the v5 rule (cold process) on the human histories: mean punishment per member of group 0 per round
import sys; sys.path.insert(0, '/opt/algorithmic-institutions/scripts/policy_finder/pf-20261106-7')
from load import load
import numpy as np, torch as th
from aimanager.manager.rule import read_rule, run_rule, rule_inputs
rule, code = read_rule('/opt/algorithmic-institutions/configs/managers/rule_based/pf-20261106-7-v5.yml', 'exp2_multi')
df = load()
games = sorted(df.game.unique())
def arr(col):
    return th.tensor(np.stack([df[df.game==g].pivot(index='player_id',columns='round_number',values=col).values for g in games])).float()
C, P, G, V = arr('contribution'), arr('punishment'), arr('group_id'), 1-arr('player_no_input')
P_own = th.zeros_like(P)  # punishment history if my rule had dealt group 0's (humans' contributions kept)
for params in [dict(c0=9,w=0.0,k=11.4,decay=0.2,follow=0.38), dict(c0=9,w=0.5,k=10.3,decay=0.2,follow=0.38), dict(c0=9,w=1.0,k=9.4,decay=0.2,follow=0.38), dict(c0=6,w=0.0,k=2.0,decay=0.05,follow=0.0), dict(c0=14,w=0.0,k=25.0,decay=1.0,follow=0.8)]:
    pass
    out, Pm = [], P.clone()
    for t in range(24):
        inp = rule_inputs(C[...,:t+1], V[...,:t+1], Pm[...,:t], G[...,:t+1], th.zeros(len(games),1,1,dtype=th.long), 'exp2_multi')
        pu = run_rule(code, params, inp).clamp(0,30).floor()
        m = inp['in_group'][...,-1]
        Pm[..., t] = th.where(m > 0, pu, P[..., t])
        out.append(float((pu*m).sum()/m.sum()))
    print(params, np.round(out,1))
