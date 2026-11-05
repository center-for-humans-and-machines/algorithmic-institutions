# The opponents' own groups as examples of other styles: their pool, members, contributions and
# punishment per head by round, averaged over all of v1's sweep games; also punishment by
# contribution level (what targets they use).
import numpy as np, pandas as pd
d = np.load('/tmp/v1sweep.npz')
c, p, v, g, ours, opp = d['c'].astype(np.float32), d['p'].astype(np.float32), d['v'], d['g'], d['ours'], d['opp']
ing = (g == ours[:, None, None])
for o in [1, 2, 3, 4]:
    m = opp == o
    res = {}
    for name, grp in [('ours', ing[m]), ('opp', ~ing[m])]:
        n = grp.sum(1)
        res[name] = dict(pool=((1.6 * c[m] - p[m]) * grp).sum(1).mean(0), mem=n.mean(0),
                         c=((c[m] * grp).sum(1) / np.maximum(n, 1)).mean(0),
                         p=((p[m] * grp).sum(1) / np.maximum(n, 1)).mean(0))
    print(f'== opponent {o}: mean pool ours {res["ours"]["pool"].mean():.1f}  theirs {res["opp"]["pool"].mean():.1f}')
    print(' theirs per round: p', np.round(res['opp']['p'], 1))
    print('                   c', np.round(res['opp']['c'], 1))
    print('                 mem', np.round(res['opp']['mem'], 2))
    # their punishment by own contribution and by deviation from their group's mean
    gg = ~ing[m]; cc = c[m]; pp = p[m]
    n = gg.sum(1, keepdims=True)
    gm = (cc * gg).sum(1, keepdims=True) / np.maximum(n, 1)
    sel = gg & v[m]
    dev = (cc - gm)[sel]; pv = pp[sel]; cv = cc[sel]
    tt = np.broadcast_to(np.arange(24), cc.shape)[sel]
    D = pd.DataFrame(dict(dev=dev, p=pv, c=cv, t=tt))
    D['cb'] = pd.cut(D.c, [-1, 0, 4, 8, 12, 16, 19, 20])
    D['db'] = pd.cut(D.dev, [-21, -8, -4, -1, 1, 4, 21])
    print(' their p by own c, rounds 0-1:', D[D.t < 2].groupby('cb', observed=True).p.mean().round(1).to_dict())
    print(' their p by own c, rounds 4-5:', D[(D.t >= 4) & (D.t < 6)].groupby('cb', observed=True).p.mean().round(1).to_dict())
    print(' their p by dev from mean, rounds 0-7:', D[D.t < 8].groupby('db', observed=True).p.mean().round(1).to_dict())
