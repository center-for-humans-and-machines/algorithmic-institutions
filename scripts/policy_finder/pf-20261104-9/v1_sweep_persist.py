"""How long does a punishment's lift last in sim? Members of group 0 at rounds 0-7 with
ratio to others' mean < 0.75: contribution k rounds later (k=1..12), stayers only, by p bin.
Also spillover: change in unpunished members' contribution vs punishment dealt to others."""
import numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
z = np.load('/tmp/v1sweep.npz'); C, V, P, G = [z[k].astype(np.float32) for k in 'CVPG']
mine = (G == 0) & (V > 0)
n = mine.sum(2, keepdims=True); tot = (C * mine).sum(2, keepdims=True)
oth = (tot - C * mine) / np.maximum(n - mine, 1)
ptot = (P * mine).sum(2, keepdims=True)
out = {}
for t0 in (1, 5, 9):
    sel = mine[..., t0] & (n[..., 0, t0][..., None] >= 2) & (C[..., t0] < 0.75 * oth[..., t0])
    # no punishment in round t0-1 to isolate a 'first' punishment
    sel &= (P[..., t0 - 1] == 0)
    pb = pd.cut(P[..., t0][sel], [-1, 0, 3, 8, 14, 30])
    res = {}
    for k in (1, 2, 3, 5, 8, 12):
        if t0 + k > 23: continue
        stay = np.all(G[..., t0:t0 + k + 1] == 0, axis=-1)[sel]
        dlt = (C[..., t0 + k] - C[..., t0])[sel]
        res[k] = pd.Series(np.where(stay, dlt, np.nan)).groupby(pb).mean()
    print('t0 =', t0, '(laggards <0.75x, unpunished the round before): change in c after k rounds, stayers')
    print(pd.DataFrame(res).round(2))
# spillover: unpunished members, ratio 0.75-1.5, change after 1 round vs punishment dealt to others / member
rows = []
for t in range(0, 20):
    sel = mine[..., t] & (P[..., t] == 0) & (G[..., t + 1] == 0) & (n[..., 0, t][..., None] >= 3)
    r = (C[..., t] / np.maximum(oth[..., t], .5))[sel]
    po = (ptot[..., 0, t][..., None] / np.maximum(n[..., 0, t][..., None] - 1, 1) * np.ones_like(C[..., t]))[sel]
    rows.append(pd.DataFrame({'r': r, 'po': po, 'd': (C[..., t + 1] - C[..., t])[sel]}))
d = pd.concat(rows)
d['rb'] = pd.cut(d.r, [-.01, .75, 1.0, 1.5, 100]); d['pob'] = pd.cut(d.po, [-1, 0, 1, 3, 6, 100])
print('unpunished members: change after 1 round by ratio x punishment per other member')
print(d.pivot_table('d', 'rb', 'pob', 'mean').round(2))
