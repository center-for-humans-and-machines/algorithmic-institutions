"""On cached sweep arrays: (1) next-round contribution change of members of the rule's
group by own contribution and punishment received; (2) the gain k rounds later;
(3) leave rate at a decision by punishment received in the block and in its last round."""
import sys, numpy as np, pandas as pd
z = np.load(sys.argv[1]); c, p, g, v = (z[k].astype(np.int16) for k in 'cpgv')
T = 24
rows = []
for t in range(23):
    inn = (g[..., t] == 0) & (v[..., t] == 1)
    for k in (1, 2, 3):
        if t + k > 23: continue
    stay1 = g[..., t+1] == 0
    m = inn & stay1
    rows.append(pd.DataFrame({'t': t, 'c': c[..., t][m], 'p': p[..., t][m], 'c1': c[..., t+1][m],
                              'c3': c[..., min(t+3, 23)][m], 'same3': (g[..., min(t+3,23)][m] == 0)}))
d = pd.concat(rows)
d['cb'] = pd.cut(d.c, [-1, 0, 3, 6, 9, 12, 15, 20]); d['pb'] = pd.cut(d.p, [-1, 0, 5, 10, 15, 20, 25, 30])
d['dc1'] = d.c1 - d.c
tab = d.groupby(['cb', 'pb']).agg(dc1=('dc1', 'mean'), n=('dc1', 'size'))
print('c_{t+1}-c_t by own c (rows) and punishment (cols), all rounds'); print(tab.dc1.where(tab.n > 300).unstack().round(2))
e = d[d.same3 & (d.t <= 20)].copy(); e['dc3'] = e.c3 - e.c
tab = e.groupby(['cb', 'pb']).agg(dc3=('dc3', 'mean'), n=('dc3', 'size'))
print('c_{t+3}-c_t (staying)'); print(tab.dc3.where(tab.n > 300).unstack().round(2))
# leaving
rows = []
for dec in (3, 7, 11, 15, 19):
    b = slice(dec-3, dec+1)
    inn = (g[..., b] == 0).all(-1)
    P = p[..., b].sum(-1); Pl = p[..., dec]; cm = c[..., b].mean(-1)
    left = g[..., dec+1] != 0
    rows.append(pd.DataFrame({'dec': dec, 'P': P[inn], 'Pl': Pl[inn], 'c': cm[inn], 'left': left[inn]}))
L = pd.concat(rows)
L['Pb'] = pd.cut(L.P, [-1, 0, 10, 20, 40, 60, 120]); L['cb'] = pd.cut(L.c, [-1, 5, 10, 15, 20])
print('leave rate, block punishment x mean c'); print(L.groupby(['Pb', 'cb']).left.mean().unstack().round(3))
print('leave rate by decision, punished in block vs not'); print(L.groupby(['dec', L.P > 0]).left.mean().unstack().round(3))
L['Plb'] = pd.cut(L.Pl, [-1, 0, 10, 20, 30])
print('leave rate by punishment in decision round'); print(L.groupby(['Plb', 'cb']).left.mean().unstack().round(3))
