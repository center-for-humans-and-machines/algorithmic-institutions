"""v6: which round's cg gap drives the first switch (round 3)? OLS of my group's size change at
round 3 on the per-capita cg gap in rounds 0..3 and on my group's mean own-punishment per round,
v5 sweep. Also the gap per round for the 15 best sweep points."""
import sys, json, numpy as np, pandas as pd
sys.path.insert(0, 'scripts/policy_finder/pf-20261104-7')
from v5_panel import load
V = sys.argv[1] if len(sys.argv) > 1 else 'pf-20261104-7-v5'
r = load(V)
c, p, v, g = r['c'].astype(float), r['p'].astype(float), r['v'].astype(float), r['g'].astype(float)
def cg(mask):
    n = (mask * v).sum(1)
    return np.where(n > 0, (1.6 * (c * mask * v).sum(1) - (p * mask * v).sum(1)) / np.maximum(n, 1), 0)
cm, co = cg(g), cg(1 - g)
gap = cm - co
size = g.sum(1)
dn = size[:, 4] - size[:, 3]
pm = (p * g).sum(1) / np.maximum(g.sum(1), 1)
cmean = (c * g).sum(1) / np.maximum(g.sum(1), 1)
X = np.column_stack([np.ones(len(dn))] + [gap[:, t] for t in range(4)])
b = np.linalg.lstsq(X, dn, rcond=None)[0]
print('dn ~ gap0..gap3:', b.round(3))
X2 = np.column_stack([np.ones(len(dn))] + [cm[:, t] for t in range(4)] + [co[:, t] for t in range(4)])
print('dn ~ cg_mine0..3, cg_other0..3:', np.linalg.lstsq(X2, dn, rcond=None)[0].round(3))
X3 = np.column_stack([np.ones(len(dn))] + [cmean[:, t] for t in range(4)] + [pm[:, t] for t in range(4)] + [co[:, t] for t in range(4)])
print('dn ~ c_mine0..3, p_mine0..3, cg_other0..3:', np.linalg.lstsq(X3, dn, rcond=None)[0].round(3))
sw = json.load(open(f'plots/simulation/policy_finder/{V}_sweep/sweep.json'))
pts = sw['points'] if 'points' in sw else sw
pool = {int(q['name'].split('_s')[-1]) if 'name' in q else i: q['pool'] for i, q in enumerate(pts)}
best = sorted(pool, key=pool.get)[-15:]
m = np.isin(r['point'], best)
print('best15: mean gap by round 0..7', gap[m, :8].mean(0).round(2))
print('best15: c mine', cmean[m, :8].mean(0).round(2), 'p mine', pm[m, :8].mean(0).round(2))
print('best15: cg other', co[m, :8].mean(0).round(2))
print('best15: size', size[m, :12].mean(0).round(2))
