"""Replay the v5 rule on the human histories (as group 0's manager), in a fresh
process: realised punishment per valid member per round, to calibrate ranges."""
import sys; sys.path.insert(0, '/opt/algorithmic-institutions/scripts/policy_finder/pf-20261106-5')
sys.path.insert(0, '/opt/algorithmic-institutions/src')
import numpy as np, torch as th
from aimanager.manager.rule import read_rule, run_rule
rule, code = read_rule('/opt/algorithmic-institutions/configs/managers/rule_based/pf-20261106-5-v5.yml', 'exp2_multi')
from load import load
d = load()
games = sorted(d.game.unique())
def arr(col):
    return th.tensor(np.stack([d[d.game == g].pivot(index='player_id', columns='round_number', values=col).values for g in games]), dtype=th.float)
C, V, P, G = arr('contribution'), 1 - arr('player_no_input'), arr('punishment'), arr('group_id')
sets = [dict(target0=9.9, target1=9.9, slope=5.8, target_late=6.9, n_late=3),   # ~ v4's peak in v5's terms
        dict(target0=9.9, target1=7, slope=6, target_late=6, n_late=3),
        dict(target0=7, target1=3, slope=2, target_late=2, n_late=0),     # low corner
        dict(target0=13, target1=13, slope=25, target_late=12, n_late=4)]  # high corner
for params in sets:
    params = {k: th.tensor(float(v)) for k, v in params.items()}
    out, hum, frac = [], [], []
    for t in range(24):
        inp = {k: v.contiguous() for k, v in dict(c=C[..., :t+1], valid=V[..., :t+1], p=P[..., :t], in_group=(G[..., :t+1] == 0).float(), t=th.tensor(float(t))).items()}
        pun = run_rule(code, params, inp).clamp(0, 30).floor()
        m = (G[..., t] == 0) & (V[..., t] > 0)
        out.append(pun[m].mean().item()); hum.append(P[..., t][m].mean().item()); frac.append((pun[m] > 0).float().mean().item())
    print({k: float(v) for k, v in params.items()}); print(' rule ', np.round(out, 1)); print(' share', np.round(frac, 2)); print(' human', np.round(hum, 1))
