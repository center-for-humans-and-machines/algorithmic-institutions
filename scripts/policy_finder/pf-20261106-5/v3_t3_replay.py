"""Replay the v3 rule on the human histories (as group 0's manager), in a fresh
process: realised punishment per valid member per round, to calibrate ranges."""
import sys; sys.path.insert(0, '/opt/algorithmic-institutions/scripts/policy_finder/pf-20261106-5')
sys.path.insert(0, '/opt/algorithmic-institutions/src')
import numpy as np, torch as th
from aimanager.manager.rule import read_rule, run_rule
rule, code = read_rule('/opt/algorithmic-institutions/configs/managers/rule_based/pf-20261106-5-v3.yml', 'exp2_multi')
from load import load
d = load()
games = sorted(d.game.unique())
def arr(col):
    return th.tensor(np.stack([d[d.game == g].pivot(index='player_id', columns='round_number', values=col).values for g in games]), dtype=th.float)
C, V, P, G = arr('contribution'), 1 - arr('player_no_input'), arr('punishment'), arr('group_id')
sets = [dict(target_open=9.7, n_open=3, target_late=0, window=4, slope=4.35, cap=22.4, t_end=9),
        dict(target_open=10, n_open=1, target_late=0, window=2, slope=4, cap=22, t_end=22),
        dict(target_open=10, n_open=2, target_late=6, window=2, slope=3, cap=25, t_end=16),
        dict(target_open=10, n_open=4, target_late=6, window=1, slope=3, cap=25, t_end=22),
        dict(target_open=5, n_open=1, target_late=0, window=1, slope=0.5, cap=5, t_end=8),
        dict(target_open=14, n_open=4, target_late=12, window=4, slope=12, cap=30, t_end=22)]
for params in sets:
    params = {k: th.tensor(float(v)) for k, v in params.items()}
    out, hum, frac = [], [], []
    for t in range(24):
        inp = {k: v.contiguous() for k, v in dict(c=C[..., :t+1], valid=V[..., :t+1], p=P[..., :t], in_group=(G[..., :t+1] == 0).float(), t=th.tensor(float(t))).items()}
        pun = run_rule(code, params, inp).clamp(0, 30).floor()
        m = (G[..., t] == 0) & (V[..., t] > 0)
        out.append(pun[m].mean().item()); hum.append(P[..., t][m].mean().item()); frac.append((pun[m] > 0).float().mean().item())
    print({k: float(v) for k, v in params.items()}); print(' rule ', np.round(out, 1)); print(' share', np.round(frac, 2)); print(' human', np.round(hum, 1))
