"""Replay the v4 rule on the human histories (as group 0's manager), in a fresh
process: realised punishment per valid member per round, to calibrate ranges."""
import sys; sys.path.insert(0, '/opt/algorithmic-institutions/scripts/policy_finder/pf-20261106-5')
sys.path.insert(0, '/opt/algorithmic-institutions/src')
import numpy as np, torch as th
from aimanager.manager.rule import read_rule, run_rule
rule, code = read_rule('/opt/algorithmic-institutions/configs/managers/rule_based/pf-20261106-5-v4.yml', 'exp2_multi')
from load import load
d = load()
games = sorted(d.game.unique())
def arr(col):
    return th.tensor(np.stack([d[d.game == g].pivot(index='player_id', columns='round_number', values=col).values for g in games]), dtype=th.float)
C, V, P, G = arr('contribution'), 1 - arr('player_no_input'), arr('punishment'), arr('group_id')
sets = [dict(target=8.8, slope=6.3, decay=0, window_open=2, window_late=2, n_blocks=3),   # ~ v3's peak (+ round 9)
        dict(target=8.8, slope=6.3, decay=1.5, window_open=2, window_late=1, n_blocks=3),
        dict(target=9, slope=20, decay=0, window_open=2, window_late=1, n_blocks=2),
        dict(target=5.5, slope=1.5, decay=3, window_open=1, window_late=1, n_blocks=1),   # low corner
        dict(target=12.5, slope=25, decay=-1, window_open=4, window_late=3, n_blocks=5)]  # high corner
for params in sets:
    params = {k: th.tensor(float(v)) for k, v in params.items()}
    out, hum, frac = [], [], []
    for t in range(24):
        inp = {k: v.contiguous() for k, v in dict(c=C[..., :t+1], valid=V[..., :t+1], p=P[..., :t], in_group=(G[..., :t+1] == 0).float(), t=th.tensor(float(t))).items()}
        pun = run_rule(code, params, inp).clamp(0, 30).floor()
        m = (G[..., t] == 0) & (V[..., t] > 0)
        out.append(pun[m].mean().item()); hum.append(P[..., t][m].mean().item()); frac.append((pun[m] > 0).float().mean().item())
    print({k: float(v) for k, v in params.items()}); print(' rule ', np.round(out, 1)); print(' share', np.round(frac, 2)); print(' human', np.round(hum, 1))
