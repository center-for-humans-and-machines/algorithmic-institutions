"""Point-level decomposition of the pool: members, contribution per member, punishment
per member, by parameter bins; and per-round contribution of the top points vs the rest."""
import sys, numpy as np, pandas as pd
z = np.load(sys.argv[1]); c, p, g, v = (z[k].astype(np.float32) for k in 'cpgv')
keys = list(z['keys']); P = pd.DataFrame(z['params'], columns=keys); P['pool'] = z['pool']
m = (g == 0)
n = m.sum(2)                                   # [pt, ep, round]
P['members'] = n.mean((1, 2))
P['cpm'] = (c*m).sum((1, 2, 3)) / m.sum((1, 2, 3))
P['ppm'] = (p*m).sum((1, 2, 3)) / m.sum((1, 2, 3))
P['c_late'] = (c*m)[..., 20:].sum((1, 2, 3)) / m[..., 20:].sum((1, 2, 3))
P['c_r0'] = c[:, :, :, 0][m[..., 0]].reshape(len(P), -1).mean(1)
P['bill'] = (p*m).sum(2).mean((1, 2))
P['gross'] = 1.6*(c*m).sum(2).mean((1, 2))
pd.set_option('display.width', 200)
print(P.corr().round(2)[['pool', 'members', 'cpm', 'bill', 'gross']])
print(P.sort_values('pool', ascending=False).head(15).round(2).to_string())
for k in keys:
    b = pd.qcut(P[k], 4, duplicates='drop') if P[k].nunique() > 6 else P[k]
    print(P.groupby(b, observed=True)[['pool', 'members', 'cpm', 'c_late', 'bill', 'gross']].mean().round(2))
np.save('/tmp/v1_points.npy', P.to_records(index=False))
