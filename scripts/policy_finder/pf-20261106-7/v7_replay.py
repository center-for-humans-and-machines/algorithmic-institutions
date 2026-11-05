# v7: replay the v7 rule (cold process) on the human histories: mean punishment per member of group 0 per round,
# and per newcomer / non-newcomer at block starts
import sys; sys.path.insert(0, '/opt/algorithmic-institutions/scripts/policy_finder/pf-20261106-7')
from load import load
import numpy as np, torch as th
from aimanager.manager.rule import read_rule, run_rule, rule_inputs
rule, code = read_rule('/opt/algorithmic-institutions/configs/managers/rule_based/pf-20261106-7-v7.yml', 'exp2_multi')
df = load()
games = sorted(df.game.unique())
def arr(col):
    return th.tensor(np.stack([df[df.game==g].pivot(index='player_id',columns='round_number',values=col).values for g in games])).float()
C, P, G, V = arr('contribution'), arr('punishment'), arr('group_id'), 1-arr('player_no_input')
v6 = dict(c0=11, w=0.468, k=25.76, gamma=1.6, decay=0.345, follow=0.633)
for params in [dict(v6, fresh=1.0), dict(v6, fresh=0.0), dict(v6, fresh=2.0), dict(v6, gamma=3.0, k=40, fresh=0.5)]:
    out, Pm, nb = [], P.clone(), []
    for t in range(24):
        inp = rule_inputs(C[...,:t+1], V[...,:t+1], Pm[...,:t], G[...,:t+1], th.zeros(len(games),1,1,dtype=th.long), 'exp2_multi')
        pu = run_rule(code, params, inp).clamp(0,30).floor()
        m = inp['in_group'][...,-1]
        Pm[..., t] = th.where(m > 0, pu, P[..., t])
        out.append(float((pu*m).sum()/m.sum()))
        if t in (4, 8, 12):
            a = m * (1 - inp['in_group'][..., t-1])
            nb.append('t%d new %.1f old %.1f' % (t, float((pu*a).sum()/a.sum()), float((pu*(m-a)).sum()/(m-a).sum())))
    print(params, np.round(out, 1), nb)
