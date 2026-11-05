# replay the rule on the human histories: what punishment level would it deal, per round, at a few param points
import sys; sys.path.insert(0, '/opt/algorithmic-institutions/scripts/policy_finder/pf-20261106-7')
from load import load
import numpy as np, torch as th
from aimanager.manager.rule import read_rule, run_rule, rule_inputs
rule, code = read_rule('/opt/algorithmic-institutions/configs/managers/rule_based/pf-20261106-7.yml', 'exp2_multi')
df = load()
games = sorted(df.game.unique())
def arr(col):
    return th.tensor(np.stack([df[df.game==g].pivot(index='player_id',columns='round_number',values=col).values for g in games]))
C, P, G, V = arr('contribution'), arr('punishment'), arr('group_id'), 1-arr('player_no_input')
for params in [dict(c0=10,k=0.5,horizon=30,slack=1.0), dict(c0=14,k=1.0,horizon=20,slack=2.0), dict(c0=6,k=0.3,horizon=60,slack=0.5)]:
    out=[]
    for t in range(24):
        inp = rule_inputs(C[...,:t+1], V[...,:t+1], P[...,:t], G[...,:t+1], th.zeros(len(games),1,1,dtype=th.long), 'exp2_multi')
        pu = run_rule(code, params, inp).clamp(0,30).floor()
        m = inp['in_group'][...,-1]
        out.append(float((pu*m).sum()/m.sum()))
    print(params, np.round(out,2))
