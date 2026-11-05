"""Replay the rule on the human histories (as group 0's manager) and compare
its punishments with the human managers'."""
import sys; sys.path.insert(0, '/opt/algorithmic-institutions/scripts/policy_finder/pf-20261106-5')
sys.path.insert(0, '/opt/algorithmic-institutions/src')
import numpy as np, torch as th
from load import load
from aimanager.manager.rule import read_rule, run_rule
rule, code = read_rule('/opt/algorithmic-institutions/configs/managers/rule_based/pf-20261106-5.yml', 'exp2_multi')
d = load()
games = sorted(d.game.unique())
def arr(col):
    return th.tensor(np.stack([d[d.game == g].pivot(index='player_id', columns='round_number', values=col).values for g in games]), dtype=th.float)
C, V, P, G = arr('contribution'), 1 - arr('player_no_input'), arr('punishment'), arr('group_id')
for params in [dict(slope=0.5, tol=3, cap0=5, follow=1, t_stop=16), dict(slope=1, tol=0, cap0=15, follow=0, t_stop=23), dict(slope=0.2, tol=6, cap0=2, follow=2, t_stop=12)]:
    out, hum = [], []
    for t in range(24):
        inp = {k: v.contiguous() for k, v in dict(c=C[..., :t+1], valid=V[..., :t+1], p=P[..., :t], in_group=(G[..., :t+1] == 0).float(), t=th.tensor(float(t))).items()}
        pun = run_rule(code, params, inp).clamp(0, 30).floor()
        m = (G[..., t] == 0) & (V[..., t] > 0)
        out.append(pun[m].mean().item()); hum.append(P[..., t][m].mean().item())
    print(params); print(' rule ', np.round(out, 2)); print(' human', np.round(hum, 2))
