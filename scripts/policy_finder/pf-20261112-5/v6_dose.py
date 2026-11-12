"""v6: individual dose-response at round R, from aligned pairs identical through round R's contributions (and earlier
punishments). Per member of my group with own dose a in one game and 0 in the other, others' points equal in both:
d mean contribution over rounds R+1..R+3, d mean over R+1..min(R+12,23) while in either group (all rounds),
d stay at R+4, by dose bin and by c at R. Usage: v6_dose.py R [quiet]"""
import numpy as np, itertools, sys, pandas as pd
R = int(sys.argv[1]) if len(sys.argv) > 1 else 0
QUIET = len(sys.argv) > 2 and sys.argv[2] == 'quiet'  # keep only pairs where my group gets nothing in R+1..R+3 in both
V = ['v1', 'v2', 'v3', 'v4', 'v5']
A = {v: np.load(f'/tmp/pf/al_{v}.npz') for v in V}
recs = []
for a, b in itertools.combinations(V, 2):
    for za, zb in [(A[a], A[b]), (A[b], A[a])]:
        same = (za['c'][:, :, :R + 1] == zb['c'][:, :, :R + 1]).all((1, 2)) & (za['p'][:, :, :R] == zb['p'][:, :, :R]).all((1, 2))
        m = za['mine'][:, :, R].astype(bool)
        pa, pb = (za['p'][:, :, R] * m).astype(float), (zb['p'][:, :, R] * m).astype(float)
        oa = pa.sum(1, keepdims=True) - pa; ob = pb.sum(1, keepdims=True) - pb
        quiet = ((za['p'][:, :, R + 1:R + 4] * za['mine'][:, :, R + 1:R + 4]).sum((1, 2)) == 0) & ((zb['p'][:, :, R + 1:R + 4] * zb['mine'][:, :, R + 1:R + 4]).sum((1, 2)) == 0)
        if QUIET: same = same & quiet
        sel = same[:, None] & m & (pa > 0) & (pb == 0) & (oa == ob)
        g, i = np.nonzero(sel)
        if len(g) == 0: continue
        e1 = min(R + 4, 24); e2 = 24
        ya = za['c'][g, i, R + 1:e1].astype(float).mean(1); yb = zb['c'][g, i, R + 1:e1].astype(float).mean(1)
        la = za['c'][g, i, R + 1:e2].astype(float).mean(1); lb = zb['c'][g, i, R + 1:e2].astype(float).mean(1)
        ri = min(R + 4, 23)
        sa = za['mine'][g, i, ri].astype(float); sb = zb['mine'][g, i, ri].astype(float)
        recs.append(pd.DataFrame(dict(dose=pa[g, i], c=za['c'][g, i, R], dy=ya - yb, dlong=la - lb, dstay=sa - sb,
                                      oth=oa[g, 0] if False else oa[g, i])))
d = pd.concat(recs, ignore_index=True)
print('R', R, 'n', len(d))
pd.set_option('display.width', 200)
d['db'] = pd.cut(d.dose, [0, 5, 10, 15, 20, 25, 29, 30])
d['cb'] = pd.cut(d.c, [-1, 2, 5, 9, 20])
t = d.groupby(['db'], observed=True)[['dy', 'dlong', 'dstay']].agg(['mean', 'count']).round(3)
print(t.to_string())
t = d.groupby(['cb', 'db'], observed=True)[['dy', 'dlong', 'dstay']].mean().round(3)
print(t.unstack(0).to_string())
