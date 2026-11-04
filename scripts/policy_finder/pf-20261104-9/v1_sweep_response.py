"""Micro response of artificial humans to my (group 0) manager's punishment in v1's sweep:
for members of group 0 at round t, by own c and ratio to others' mean, outcome by punishment:
change in c after 1 and 3 rounds (if still in group 0), and leaving at the next decision."""
import numpy as np, pandas as pd
z = np.load('/tmp/v1sweep.npz'); C, V, P, G = [z[k].astype(np.float32) for k in 'CVPG']
mine = (G == 0) & (V > 0)
n = mine.sum(2, keepdims=True); tot = (C * mine).sum(2, keepdims=True)
oth = (tot - C * mine) / np.maximum(n - mine, 1)
rows = []
for t in range(0, 20):
    sel = mine[..., t] & (n[..., 0, t][..., None] >= 2)
    dec = (t // 4 + 1) * 4  # first round after the next decision
    df = pd.DataFrame({'c': C[..., t][sel], 'o': oth[..., t][sel], 'p': P[..., t][sel],
                       'c1': C[..., t + 1][sel], 'c3': C[..., min(t + 3, 23)][sel],
                       's1': (G[..., t + 1] == 0)[sel], 's3': (G[..., min(t + 3, 23)] == 0).astype(bool)[sel] & (G[..., t + 1] == 0)[sel],
                       'left': (G[..., dec] != 0)[sel] if dec < 24 else np.zeros(sel.sum(), bool),
                       'decnext': np.full(sel.sum(), (t % 4 == 3) and t < 20), 't': t})
    rows.append(df)
d = pd.concat(rows)
d['ratio'] = pd.cut(d.c / np.maximum(d.o, 0.5), [-0.01, 0.25, 0.5, 0.75, 1.0, 1.5, 100])
d['pb'] = pd.cut(d.p, [-1, 0, 2, 5, 9, 14, 30])
d['d1'] = np.where(d.s1, d.c1 - d.c, np.nan); d['d3'] = np.where(d.s3, d.c3 - d.c, np.nan)
pd.set_option('display.width', 200)
print('n rows', len(d))
print('change after 1 round (stayers)'); print(d.pivot_table('d1', 'ratio', 'pb', 'mean').round(2))
print('change after 3 rounds'); print(d.pivot_table('d3', 'ratio', 'pb', 'mean').round(2))
print('counts'); print(d.pivot_table('d1', 'ratio', 'pb', 'count'))
print('P(leave at next decision), rounds 3,7,11,15,19 only'); dd = d[d.decnext]
print(dd.pivot_table('left', 'ratio', 'pb', 'mean').round(3))
print('P(leave at next decision) by round in phase of punishment, all ratios, p>0 vs 0')
d['ph'] = d.t % 4; d['pun'] = d.p > 0
print(d[d.t < 20].pivot_table('left', 'ph', 'pun', 'mean').round(3))

