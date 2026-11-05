# v2: across v1's design points, which punishment profile (by phase) goes with a larger pool, per opponent
import numpy as np, json
d = np.load('/tmp/v1sweep.npz'); C, P, G = [d[k].astype(np.float32) for k in 'CPG']
pt, opp, seat = d['point'], d['opp'], d['seat']
mine = (G == seat[:, None, None]).astype(np.float32)
n = mine.sum(1)  # [games, rounds]
pool = (1.6 * (C * mine).sum(1) - (P * mine).sum(1))  # per round
pm = (P * mine).sum(1) / np.maximum(n, 1)  # my mean punishment per member per round
po = (P * (1 - mine)).sum(1) / np.maximum(8 - n, 1)
cm = (C * mine).sum(1) / np.maximum(n, 1)
phases = {'r0': [0], 'r1-3': [1, 2, 3], 'blockstart4-16': [4, 8, 12, 16], 'decision3-19': [3, 7, 11, 15, 19], 'r4-11': list(range(4, 12)), 'r12-23': list(range(12, 24))}
rows = []
for o in range(1, 5):
    for p_ in np.unique(pt):
        m = (pt == p_) & (opp == o)
        r = dict(opp=o, pt=p_, pool=pool[m].mean(), n=n[m].mean(), c=cm[m].mean(), c_end=cm[m][:, 20:].mean())
        for k, rr in phases.items():
            r[k] = pm[m][:, rr].mean()
        rows.append(r)
import pandas as pd
df = pd.DataFrame(rows)
df.to_csv('/tmp/v1_points.csv', index=False)
feats = ['r0', 'r1-3', 'r4-11', 'r12-23']
for o in range(1, 5):
    s = df[df.opp == o]
    X = np.c_[np.ones(len(s)), s[feats].values, s['r0'].values**2]
    for y in ['pool', 'n', 'c']:
        b = np.linalg.lstsq(X, s[y].values, rcond=None)[0]
        print(o, y, dict(zip(['const'] + feats + ['r0^2'], np.round(b, 3))))
s = df.groupby('pt').mean()
X = np.c_[np.ones(len(s)), s[feats].values, s['r0'].values**2]
b = np.linalg.lstsq(X, s['pool'].values, rcond=None)[0]
print('all', dict(zip(['const'] + feats + ['r0^2'], np.round(b, 3))))
print(df.groupby('opp')[['pool','n','c','r0','r1-3','r4-11','r12-23']].describe().T.round(2).to_string())
