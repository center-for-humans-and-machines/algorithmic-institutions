"""v2 sweep block gate, w=1 points only: at block b (round 4b), points with blocks>b punish round 4b (treated),
blocks==b do not (control), identical rules before. Rest-of-game mean lead (rounds 4b..23, per round), my members at
end, my/other contribution, split by D bin. Prints treated-minus-control too."""
import numpy as np, pandas as pd
z = np.load('/tmp/pf/v2.npz'); c, p, m, v = [z[k].astype(np.float32) for k in ['c', 'p', 'mine', 'v']]
P = pd.DataFrame(z['params'], columns=['D', 'lo', 'hi', 'w', 'blocks'], index=z['param_idx'])
pt = z['point'].astype(int); par = P.loc[pt].reset_index(drop=True)
pool = 1.6 * c * v - p
L = (pool * m).sum(1) - (pool * (1 - m)).sum(1)
Db = pd.cut(par.D, [0, 4, 8, 12.1]).astype(str).values
for w in [1, 2]:
  for b in range(1, 6):
    r0 = 4 * b
    rows = []
    for db in sorted(np.unique(Db)):
        base = (par.w == w).values & (Db == db)
        t_ = base & (par.blocks > b).values; c_ = base & (par.blocks == b).values
        lt, lc = L[t_][:, r0:].mean(1), L[c_][:, r0:].mean(1)
        rows.append(dict(D=db, nt=t_.sum(), nc=c_.sum(), diff=lt.mean() - lc.mean(),
                         se=np.sqrt(lt.var() / len(lt) + lc.var() / len(lc)),
                         dose=(p * m)[t_][:, :, r0].sum(1).mean(),
                         dmem=m[t_][:, :, 23].sum(1).mean() - m[c_][:, :, 23].sum(1).mean()))
    print('w', w, 'block', b); print(pd.DataFrame(rows).round(2).to_string())
