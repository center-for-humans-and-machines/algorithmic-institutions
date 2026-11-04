"""Does the lift from a punishment depend on where in the 4-round phase it falls? Laggards
(< 0.75x others' mean) of group 0 unpunished the round before, change in c 4 rounds later,
by phase position of the punishment round and punishment bin; stayers through t+4 only,
plus the share still in my group at t+4."""
import numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
z = np.load('/tmp/v1sweep.npz'); C, V, P, G = [z[k].astype(np.float32) for k in 'CVPG']
mine = (G == 0) & (V > 0)
n = mine.sum(2, keepdims=True); tot = (C * mine).sum(2, keepdims=True)
oth = (tot - C * mine) / np.maximum(n - mine, 1)
rows = []
for t in range(1, 19):
    sel = mine[..., t] & (n[..., 0, t][..., None] >= 2) & (C[..., t] < 0.75 * oth[..., t]) & (P[..., t - 1] == 0)
    stay = np.all(G[..., t:t + 5] == 0, -1)[sel]
    rows.append(pd.DataFrame({'pos': t % 4, 'p': P[..., t][sel], 'stay': stay,
                              'd4': np.where(stay, (C[..., t + 4] - C[..., t])[sel], np.nan)}))
d = pd.concat(rows); d['pb'] = pd.cut(d.p, [-1, 0, 3, 8, 14, 30])
print('change in c after 4 rounds'); print(d.pivot_table('d4', 'pos', 'pb', 'mean').round(2))
print('share still in group 0 at t+4'); print(d.pivot_table('stay', 'pos', 'pb', 'mean').round(3))
