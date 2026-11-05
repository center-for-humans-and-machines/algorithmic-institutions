# v3: across v2's design points: punishment by phase vs pool, members, late contributions, per opponent
import numpy as np, pandas as pd, json, sys
src = sys.argv[1] if len(sys.argv)>1 else '/tmp/v2sweep.npz'
d = np.load(src); C, P, G = [d[k].astype(np.float32) for k in 'CPG']
pt, opp, seat = d['point'], d['opp'], d['seat']
mine = (G == seat[:, None, None]).astype(np.float32)
n = mine.sum(1)
pool = 1.6 * (C * mine).sum(1) - (P * mine).sum(1)
pm = (P * mine).sum(1) / np.maximum(n, 1)
cm = (C * mine).sum(1) / np.maximum(n, 1)
phases = {'r0': [0], 'r1-2': [1, 2], 'r3': [3], 'r4-7': [4,5,6,7], 'r8-23': list(range(8, 24))}
rows = []
for o in range(1, 5):
    for p_ in np.unique(pt):
        m = (pt == p_) & (opp == o)
        r = dict(opp=o, pt=p_, pool=pool[m].mean(), n=n[m].mean(), n_b1=n[m][:, 4:8].mean(), c=cm[m].mean(), c_late=cm[m][:, 16:].mean(), pool_late=pool[m][:,16:].mean())
        for k, rr in phases.items():
            r[k] = pm[m][:, rr].mean()
        rows.append(r)
df = pd.DataFrame(rows); df.to_csv(src.replace('.npz', '_points.csv'), index=False)
feats = list(phases)
for o in range(1, 5):
    s = df[df.opp == o]
    X = np.c_[np.ones(len(s)), s[feats].values, s['r0'].values**2]
    for y in ['pool', 'n', 'c_late', 'pool_late']:
        b = np.linalg.lstsq(X, s[y].values, rcond=None)[0]
        print(o, y, dict(zip(['const'] + feats + ['r0^2'], np.round(b, 3))))
print(df.groupby('opp')[['pool', 'n', 'c', 'c_late'] + feats].mean().round(2))
s = df.groupby('pt').mean(); top = s.sort_values('pool', ascending=False).head(10)
print(top[['pool', 'n', 'c', 'c_late'] + feats].round(2))
