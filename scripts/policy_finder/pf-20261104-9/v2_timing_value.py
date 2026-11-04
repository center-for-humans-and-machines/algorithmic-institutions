"""Across v2's 256 sweep points: regress the pool on the punishment per member dealt
in round buckets (and the share of it dealt to members at/above the others' mean), to
see where in the game a punishment point adds or costs pool."""
import numpy as np, json
d = np.load('/tmp/pfw/v2.npz'); C, V, P, G = [d[k].astype(np.float32) for k in 'CVPG']
g0 = G == 0
pool = (1.6 * (C * g0).sum(2) - (P * g0).sum(2)).mean((1, 2))
mem = g0.sum(2)
ppm = ((P * g0).sum(2) / np.maximum(mem, 1)).mean(1)  # point, round
tot = (C * g0).sum(2, keepdims=True); n = g0.sum(2, keepdims=True)
ref = (tot - C) / np.maximum(n - 1, 1)
hi = g0 & (C >= ref) & (n > 1)
phi = ((P * hi).sum(2) / np.maximum(mem, 1)).mean(1)
buckets = {'r0': [0], 'r1-2': [1, 2], 'r3': [3], 'r4': [4], 'r5-7': [5, 6, 7], 'r8': [8], 'r9-11': [9, 10, 11],
           'r12-15': [12, 13, 14, 15], 'r16-19': [16, 17, 18, 19], 'r20-23': [20, 21, 22, 23]}
X = np.stack([ppm[:, b].sum(1) for b in buckets.values()] + [phi.sum(1)], 1)
names = list(buckets) + ['to_above_mean(all r)']
Xc = np.c_[np.ones(len(pool)), X]
beta, *_ = np.linalg.lstsq(Xc, pool, rcond=None)
res = pool - Xc @ beta; s2 = res.var() * len(pool) / (len(pool) - Xc.shape[1])
se = np.sqrt(np.diag(s2 * np.linalg.inv(Xc.T @ Xc)))
print('R2', 1 - res.var() / pool.var())
for nm, b, e, m in zip(['const'] + names, beta, se, [1] + list(X.mean(0))):
    print(f'{nm:22s} {b:+7.2f} +- {e:5.2f}   (mean x {m:.2f})')
