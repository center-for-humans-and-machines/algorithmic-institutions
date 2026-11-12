"""v6: v5's slot-adjusted surface (adj from v6_fe.py): bins per param and a quadratic fit."""
import numpy as np, pandas as pd
df = pd.read_csv('/tmp/pf/adj6.csv'); z = np.load('/tmp/pf/points.npz')
d = df[df.v == 'v5'].reset_index(drop=True)
P = z['v5_params']; names = ['D0', 'D4', 'D8', 's', 'hi']
assert (z['v5_pidx'] == d.slot.values).all()
for i, n in enumerate(names): d[n] = P[:, i]
pd.set_option('display.width', 200)
for n in ['D0', 'D4', 'D8', 's']:
    q = pd.qcut(d[n], 5)
    print(n, d.groupby(q, observed=True)[['adj', 'mem', 'own', 'oth']].mean().round(2).T.to_string(), '\n')
print('hi', d.groupby('hi')[['adj', 'mem']].mean().round(2).T.to_string())
cols = {}
for n in ['D0', 'D4', 'D8']:
    cols[n] = d[n] / 10; cols[n + '^2'] = (d[n] / 10) ** 2
cols['s'] = d.s; cols['s^2'] = d.s ** 2; cols['s*D4'] = d.s * d.D4 / 10; cols['s*D8'] = d.s * d.D8 / 10
for h in range(4, 9): cols[f'hi{h}'] = (d.hi == h).astype(float)
X = pd.DataFrame(cols); X['1'] = 1.0
for y in ['adj', 'mem']:
    b, *_ = np.linalg.lstsq(X.values, d[y].values, rcond=None)
    r = d[y].values - X.values @ b; s2 = (r ** 2).sum() / (len(r) - X.shape[1])
    se = np.sqrt(np.diag(s2 * np.linalg.inv(X.values.T @ X.values)))
    print(y, 'resid sd %.2f' % np.sqrt(s2)); print(' '.join(f'{c}:{x:+.2f}({e:.2f})' for c, x, e in zip(X.columns, b, se)))
print(d.sort_values('adj', ascending=False).head(15)[['slot', 'lead', 'adj'] + names].round(2).to_string())
