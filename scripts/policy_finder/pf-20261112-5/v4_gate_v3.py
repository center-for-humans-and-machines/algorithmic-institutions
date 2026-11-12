"""v3 sweep gates at block starts, by region (hi, D): round 4 gate (first=0: nblocks>=2 vs 1; first=1 vs 2),
round 8 gate (first=0: nblocks 3 vs 2; first=1: nblocks>=2 vs 1). Rest-of-game mean lead per round, member diff."""
import numpy as np, pandas as pd
z = np.load('/tmp/pf/v3.npz'); c, p, m, v = [z[k].astype(np.float32) for k in ['c', 'p', 'mine', 'v']]
P = pd.DataFrame(z['params'], columns=['D', 'hi', 'first', 'nblocks', 'w'], index=z['param_idx'])
pt = z['point'].astype(int); par = P.loc[pt].reset_index(drop=True)
pool = 1.6 * c * v - p
L = (pool * m).sum(1) - (pool * (1 - m)).sum(1)
f, nb, hi, D, w = [par[k].values for k in ['first', 'nblocks', 'hi', 'D', 'w']]
gates = {'r0 (first0 vs first1)': (0, f == 0, f == 1),
         'r4 first0 nb>=2 vs nb1': (4, (f == 0) & (nb >= 2), (f == 0) & (nb == 1)),
         'r4 first1 vs first2': (4, f == 1, f == 2),
         'r8 first0 nb3 vs nb2': (8, (f == 0) & (nb == 3), (f == 0) & (nb == 2)),
         'r8 first1 nb>=2 vs nb1': (8, (f == 1) & (nb >= 2), (f == 1) & (nb == 1))}
regions = {'all': np.ones_like(f, bool), 'hi<=7': hi <= 7, 'hi<=7,D>8': (hi <= 7) & (D > 8), 'hi<=7,D<=8': (hi <= 7) & (D <= 8), 'hi>7': hi > 7}
rows = []
for g, (r0, t_, c_) in gates.items():
    for rn, rg in regions.items():
        for ww in [1]:
            T = t_ & rg & (w == ww); C = c_ & rg & (w == ww)
            lt, lc = L[T][:, r0:].mean(1), L[C][:, r0:].mean(1)
            rows.append(dict(gate=g, region=rn, w=ww, nt=T.sum(), nc=C.sum(), diff=lt.mean() - lc.mean(),
                             se=np.sqrt(lt.var() / len(lt) + lc.var() / len(lc)), dose=(p * m)[T][:, :, r0].sum(1).mean(),
                             dmem=m[T][:, :, 23].sum(1).mean() - m[C][:, :, 23].sum(1).mean(),
                             dc_mine=((c*m)[T][:, :, r0+1:].sum((1,2))/np.maximum(m[T][:, :, r0+1:].sum((1,2)),1)).mean() - ((c*m)[C][:, :, r0+1:].sum((1,2))/np.maximum(m[C][:, :, r0+1:].sum((1,2)),1)).mean(),
                             dc_oth=((c*(1-m))[T][:, :, r0+1:].sum((1,2))/np.maximum((1-m)[T][:, :, r0+1:].sum((1,2)),1)).mean() - ((c*(1-m))[C][:, :, r0+1:].sum((1,2))/np.maximum((1-m)[C][:, :, r0+1:].sum((1,2)),1)).mean()))
print(pd.DataFrame(rows).round(2).to_string())
