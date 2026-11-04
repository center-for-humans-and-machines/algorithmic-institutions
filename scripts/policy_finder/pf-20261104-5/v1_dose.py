"""v2: across v1 sweep points, how late contributions, members and score depend on the
punishment dose per member-round in each block (0-3, 4-7, 8-11, 12-23)."""
import pandas as pd, numpy as np
from numpy.linalg import lstsq
t = pd.read_csv('scripts/policy_finder/pf-20261104-5/v1_points.csv', index_col=0)
X = np.c_[np.ones(len(t)), t[['pm0', 'pm1', 'pm2', 'pm3']].values]
for y in ['cm1', 'cm2', 'cm3', 'n1', 'n2', 'n3', 'pool0', 'pool3', 'score']:
    b, *_ = lstsq(X, t[y].values, rcond=None)
    res = t[y].values - X @ b
    print(f'{y:6s} ~ 1 + pm0 pm1 pm2 pm3: {np.round(b, 2)}  R2={1 - res.var() / t[y].var():.2f}')
# with squares to see saturation in the first block
X2 = np.c_[X, t.pm0 ** 2, t.pm1 ** 2]
for y in ['cm3', 'score']:
    b, *_ = lstsq(X2, t[y].values, rcond=None)
    print(y, 'with pm0^2, pm1^2:', np.round(b, 3))
t['dose'] = pd.cut(t.pm0, [-1, 0.01, 0.5, 1.5, 3, 10])
print(t.groupby('dose', observed=True)[['pm0', 'pm1', 'pm2', 'pm3', 'cm0', 'cm1', 'cm3', 'n1', 'n3', 'score']].mean().round(2).assign(N=t.groupby('dose', observed=True).size()))
print(t.sort_values('score', ascending=False).head(12)[['k', 'thr', 'cap', 't_stop', 'pm0', 'pm1', 'pm2', 'pm3', 'cm3', 'n3', 'score']].round(2))
