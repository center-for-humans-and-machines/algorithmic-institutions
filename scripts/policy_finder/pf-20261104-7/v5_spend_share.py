"""v5: under v4's best sweep points, how much of my group's punishment goes to players I punished
the round before, per round; and the contribution of those players."""
import sys, json, numpy as np, pandas as pd
sys.path.insert(0, 'scripts/policy_finder/pf-20261104-7')
from v5_panel import load
V = 'pf-20261104-7-v4'
r = load(V)
sw = json.load(open(f'plots/simulation/policy_finder/{V}_sweep/sweep.json'))
pts = sorted(sw['points'], key=lambda x: -x['pool'])[:15]
idx = [int(x['name'].split('_s')[-1]) for x in pts]
print([(i, round(x['pool'], 1)) for i, x in zip(idx, pts)][:5])
m = np.isin(r['point'], idx)
c, p, g = r['c'][m].astype(float), r['p'][m].astype(float), r['g'][m]
pm = p * g
tot = pm.sum((0, 1))
prev = np.concatenate([np.zeros_like(pm[:, :, :1]), pm[:, :, :-1]], -1)
rep = (pm * (prev > 0)).sum((0, 1))
n = g.sum((0, 1))
print('round, spend per member, share to last-round-punished')
for t in range(12):
    print(t, round(tot[t] / n[t], 2), round(rep[t] / max(tot[t], 1), 2))
print('overall share', rep.sum() / tot.sum())
