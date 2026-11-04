"""v3: member flows at the switch decisions (cached v1+v2 sweeps). For each decision,
the rule's group's block mean contribution per member, its block punishment per member
in the block's first two rounds and in the decision round, and the sizes; tabulates the
expected net member gain (joiners - leavers) and both rates."""
import sys, numpy as np, pandas as pd
rows = []
for f in sys.argv[1:]:
    z = np.load(f); c, p, g = (z[k].astype(np.int16) for k in 'cpg')
    for dec in (3, 7, 11, 15, 19):
        b = slice(dec-3, dec+1)
        ours = g[..., dec] == 0; theirs = g[..., dec] == 1
        n0 = ours.sum(-1); n1 = theirs.sum(-1)
        gm = (g[..., b] == 0)
        cm0 = (c[..., b]*gm).sum((-1, -2)) / np.maximum(gm.sum((-1, -2)), 1)
        pm0 = (p[..., b]*gm).sum((-1, -2)) / np.maximum(gm.sum((-1, -2)), 1) * 4   # per member per block
        hm = (g[..., b] == 1)
        cm1 = (c[..., b]*hm).sum((-1, -2)) / np.maximum(hm.sum((-1, -2)), 1)
        pm1 = (p[..., b]*hm).sum((-1, -2)) / np.maximum(hm.sum((-1, -2)), 1) * 4
        lv = (ours & (g[..., dec+1] != 0)).sum(-1); jn = (theirs & (g[..., dec+1] == 0)).sum(-1)
        rows.append(pd.DataFrame({'dec': dec, 'n0': n0.ravel(), 'cm0': cm0.ravel(), 'pm0': pm0.ravel(),
            'cm1': cm1.ravel(), 'pm1': pm1.ravel(), 'lv': lv.ravel(), 'jn': jn.ravel(), 'n1': n1.ravel()}))
d = pd.concat(rows); d = d[(d.n0 > 0) & (d.n1 > 0)]
d['net'] = d.jn - d.lv
d['lvr'] = d.lv / d.n0; d['jnr'] = d.jn / d.n1
d['dc'] = pd.cut(d.cm0 - d.cm1, [-20, -3, -1, 1, 3, 20])
d['pb'] = pd.cut(d.pm0, [-1, 0, 5, 10, 20, 40, 200])
pd.set_option('display.width', 200)
for dec in (3, 11):
    x = d[d.dec == dec]
    print('decision', dec, 'net members by our-minus-their contribution (rows) and our punishment per member in block (cols)')
    print(x.pivot_table(index='dc', columns='pb', values='net', aggfunc='mean', observed=True).round(2))
    print(x.pivot_table(index='dc', columns='pb', values='jnr', aggfunc='mean', observed=True).round(2))
    print(x.pivot_table(index='dc', columns='pb', values='lvr', aggfunc='mean', observed=True).round(2))
    print(x.pivot_table(index='dc', columns='pb', values='net', aggfunc='size', observed=True))
