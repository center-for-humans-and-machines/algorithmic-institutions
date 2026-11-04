"""Leave rate at a decision for members of the rule's group in points with quiet>=2
(punishment only in the block's first two rounds): by punishment in those two rounds and
mean contribution; and by the group's other members' punishment (spill-over); and the
join rate: share of the other group's members who move in, by the rule group's mean
punishment in the block."""
import sys, numpy as np, pandas as pd
z = np.load(sys.argv[1]); c, p, g, v = (z[k].astype(np.int16) for k in 'cpgv')
quiet = z['params'][:, list(z['keys']).index('quiet')]
sel = quiet >= 2
c, p, g = c[sel], p[sel], g[sel]
rows = []
for dec in (3, 7, 11, 15, 19):
    b = slice(dec-3, dec+1)
    inn = (g[..., b] == 0).all(-1)
    P = p[..., dec-3:dec-1].sum(-1); cm = c[..., b].mean(-1)
    ing = g[..., dec] == 0
    Pg = (p[..., dec-3:dec-1].sum(-1) * ing).sum(-1, keepdims=True)
    ng = ing.sum(-1, keepdims=True)
    Poth = np.broadcast_to((Pg - P*ing) / np.maximum(ng - 1, 1), P.shape)
    left = g[..., dec+1] != 0
    rows.append(pd.DataFrame({'dec': dec, 'P': P[inn], 'c': cm[inn], 'left': left[inn], 'Poth': Poth[inn]}))
    # joiners
    out = g[..., dec] == 1
    join = (g[..., dec+1] == 0) & out
    Pmean = np.broadcast_to(Pg / np.maximum(ng, 1), out.shape)
    rows[-1] = rows[-1]
    J = pd.DataFrame({'dec': dec, 'Pm': Pmean[out], 'join': join[out], 'c': c[..., dec][out]})
    rows.append(J.assign(kind='J'))
L = pd.concat([r for r in rows if 'kind' not in r]); J = pd.concat([r for r in rows if 'kind' in r])
L['Pb'] = pd.cut(L.P, [-1, 0, 10, 20, 30, 45, 60]); L['cb'] = pd.cut(L.c, [-1, 5, 10, 15, 20])
print(L.groupby(['Pb', 'cb'], observed=True).left.mean().unstack().round(3))
L['Pob'] = pd.cut(L.Poth, [-1, 0, 5, 10, 20, 60])
print('own P=0: by others mean block punishment'); print(L[L.P == 0].groupby(['Pob', 'cb'], observed=True).left.mean().unstack().round(3))
J['Pmb'] = pd.cut(J.Pm, [-1, 0, 5, 10, 20, 60]); J['cb'] = pd.cut(J.c, [-1, 5, 10, 15, 20])
print('join rate from the other group by rule group mean block punishment, joiner c'); print(J.groupby(['Pmb', 'cb'], observed=True)['join'].mean().unstack().round(3))
