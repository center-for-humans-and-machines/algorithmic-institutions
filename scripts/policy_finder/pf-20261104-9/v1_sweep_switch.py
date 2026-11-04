"""Switching in v1's sweep. (a) Leaving my group at a decision: logistic-ish table by
punishment received in each of the 4 rounds before the decision. (b) Who leaves and who joins,
by contribution in the last round before the decision, for light vs heavy punishing points.
(c) Joining rate of ah's members vs my group's mean punishment in the phase."""
import numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
z = np.load('/tmp/v1sweep.npz'); C, V, P, G = [z[k].astype(np.float32) for k in 'CVPG']
rows = []
for dec in (4, 8, 12, 16, 20):
    t = dec - 1
    df = pd.DataFrame({'g': G[..., t].ravel(), 'left': (G[..., dec] != G[..., t]).ravel(),
                       'c': C[..., t].ravel(), 'cph': C[..., dec - 4:dec].mean(-1).ravel(),
                       **{f'p{k}': P[..., dec - 4 + k].ravel() for k in range(4)},
                       'pt': np.repeat(np.arange(256), 500 * 8), 'dec': dec})
    # my group's mean punishment per member in the phase and its size
    mine = (G[..., dec - 4:dec] == 0)
    pm = (P[..., dec - 4:dec] * mine).sum((2, 3)) / np.maximum(mine.sum((2, 3)), 1)
    om = (P[..., dec - 4:dec] * ~mine).sum((2, 3)) / np.maximum((~mine).sum((2, 3)), 1)
    df['pm_mine'] = np.repeat(pm.ravel(), 8); df['pm_ah'] = np.repeat(om.ravel(), 8)
    rows.append(df)
d = pd.concat(rows)
m = d[d.g == 0]
print('base leave rate from my group %.3f, from ah %.3f' % (m.left.mean(), d[d.g == 1].left.mean()))
for k in range(4):
    print('my members, leave rate by punishment in round %d of the phase (others any):' % k,
          m.groupby(pd.cut(m[f'p{k}'], [-1, 0, 3, 8, 14, 30])).left.mean().round(3).to_dict())
# isolate: punished only in one round of the phase
for k in range(4):
    only = (m[f'p{k}'] > 0) & (m[[f'p{j}' for j in range(4) if j != k]].sum(1) == 0)
    none = m[['p0', 'p1', 'p2', 'p3']].sum(1) == 0
    print('punished only in round %d: leave %.3f (n=%d, mean p %.1f) vs never %.3f' % (
        k, m[only].left.mean(), only.sum(), m[only][f'p{k}'].mean(), m[none].left.mean()))
# who leaves / joins by c phase mean, light vs heavy points
s = d.groupby('pt').pm_mine.mean()
light = s.index[s < s.quantile(.3)]; heavy = s.index[s > s.quantile(.7)]
for nm, pts in (('light', light), ('heavy', heavy)):
    x = d[d.pt.isin(pts)]
    cb = pd.cut(x.cph, [-1, 3, 7, 11, 15, 20])
    print(nm, 'leave mine by phase-mean c', x[x.g == 0].groupby(cb[x.g == 0]).left.mean().round(3).to_dict())
    print(nm, 'join from ah by phase-mean c', x[x.g == 1].groupby(cb[x.g == 1]).left.mean().round(3).to_dict())
# joining from ah vs my mean punishment, ah's own members unpunished
a = d[(d.g == 1) & (d[['p0', 'p1', 'p2', 'p3']].sum(1) == 0)]
print('ah members unpunished: join rate by my group mean punishment/member-round',
      a.groupby(pd.cut(a.pm_mine, [-1, 0, .5, 1, 2, 4, 30])).left.mean().round(3).to_dict())
mm = m[m[['p0', 'p1', 'p2', 'p3']].sum(1) == 0]
print('my members unpunished: leave rate by my group mean punishment',
      mm.groupby(pd.cut(mm.pm_mine, [-1, 0, .5, 1, 2, 4, 30])).left.mean().round(3).to_dict())
