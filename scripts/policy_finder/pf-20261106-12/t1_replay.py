# replay human histories through the rule at a mid-range point; report mean punishment
import sys; sys.path.insert(0, '/opt/algorithmic-institutions/scripts/policy_finder/pf-20261106-12')
from load import load
import numpy as np, torch as th
from aimanager.manager.rule import load_rule, rule_inputs, run_rule
d, _ = load()
games = sorted(d.game.unique())
def arr(col):
    return th.tensor(np.stack([d[d.game==g].pivot(index='player_id', columns='round_number', values=col).values for g in games]))
C, V, P, G = arr('contribution'), arr('valid'), arr('punishment'), arr('group_id').long()
rule = '/opt/algorithmic-institutions/configs/managers/rule_based/pf-20261106-12.yml'
for pt in [dict(norm_frac=0.8, slope=1.0, pmax=6, allow=1.0, t_stop=21), dict(norm_frac=1.0, slope=0.5, pmax=4, allow=0.3, t_stop=20)]:
    code, params = load_rule(rule, pt, 'exp2_multi')
    out = []
    for t in range(24):
        inp = rule_inputs(C[..., :t+1], V[..., :t+1], P[..., :t], G[..., :t+1], 0, 'exp2_multi')
        r = run_rule(code, params, inp).clamp(0, 30).floor()
        m = (G[..., t] == 0) & (V[..., t] > 0)
        out.append(float(r[m].mean()))
    print(pt, 'mean p by round', np.round(out, 2))
