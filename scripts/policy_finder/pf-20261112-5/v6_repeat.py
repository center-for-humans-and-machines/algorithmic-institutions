"""v6: at round R (4 or 8), does a punished member's answer depend on (a) having been punished by me before (repeat),
(b) being a newcomer, (c) their contribution relative to the others in my group? Aligned pairs identical through
round R's contributions; member punished in one game (dose >= 15) and not in the other, others' points equal; quiet
pairs (nothing dealt by me in R+1..R+3). Outcomes: d mean contribution R+1..R+3, d mean R+1..23, d stay at R+4,
and d lead R..23. Usage: v6_repeat.py R"""
import numpy as np, itertools, sys, pandas as pd
R = int(sys.argv[1]) if len(sys.argv) > 1 else 4
V = ['v1', 'v2', 'v3', 'v4', 'v5']
A = {v: np.load(f'/tmp/pf/al_{v}.npz') for v in V}
recs = []
for a, b in itertools.permutations(V, 2):
    za, zb = A[a], A[b]
    same = (za['c'][:, :, :R + 1] == zb['c'][:, :, :R + 1]).all((1, 2)) & (za['p'][:, :, :R] == zb['p'][:, :, :R]).all((1, 2))
    quiet = ((za['p'][:, :, R + 1:R + 4] * za['mine'][:, :, R + 1:R + 4]).sum((1, 2)) == 0) & ((zb['p'][:, :, R + 1:R + 4] * zb['mine'][:, :, R + 1:R + 4]).sum((1, 2)) == 0)
    same &= quiet
    m = za['mine'][:, :, R].astype(bool)
    pa, pb = (za['p'][:, :, R] * m).astype(float), (zb['p'][:, :, R] * m).astype(float)
    oa = pa.sum(1, keepdims=True) - pa; ob = pb.sum(1, keepdims=True) - pb
    sel = same[:, None] & m & (pa >= 15) & (pb == 0) & (oa == ob)
    g, i = np.nonzero(sel)
    if len(g) == 0: continue
    prevp = (za['p'][g, i, :R] * za['mine'][g, i, :R]).sum(1)
    newc = ~za['mine'][g, i, R - 1].astype(bool)
    cR = za['c'][g, i, R].astype(float)
    mm = za['mine'][g, :, R].astype(float); csum = (za['c'][g, :, R] * mm).sum(1); n = mm.sum(1)
    others_mean = (csum - cR) / np.maximum(n - 1, 1)
    c_prev = za['c'][g, i, R - 1].astype(float)
    ya = za['c'][g, i, R + 1:R + 4].astype(float).mean(1); yb = zb['c'][g, i, R + 1:R + 4].astype(float).mean(1)
    la = za['c'][g, i, R + 1:].astype(float).mean(1); lb = zb['c'][g, i, R + 1:].astype(float).mean(1)
    sa = za['mine'][g, i, R + 4].astype(float); sb = zb['mine'][g, i, R + 4].astype(float)
    recs.append(pd.DataFrame(dict(dose=pa[g, i], c=cR, rep=prevp > 0, new=newc, gap=others_mean - cR, drop=c_prev - cR,
                                  n=n, dy=ya - yb, dlong=la - lb, dstay=sa - sb)))
d = pd.concat(recs, ignore_index=True)
print('R', R, 'n', len(d), 'mean dose %.1f' % d.dose.mean())
pd.set_option('display.width', 200)
agg = lambda x: x[['dy', 'dlong', 'dstay']].agg(['mean']).round(3).assign(n=len(x))
d['cb'] = pd.cut(d.c, [-1, 2, 5, 9, 20])
for k in ['rep', 'new']:
    print(d.groupby(['cb', k], observed=True)[['dy', 'dlong', 'dstay']].mean().round(3).join(d.groupby(['cb', k], observed=True).size().rename('n')).to_string())
d['gapb'] = pd.cut(d.gap, [-20, 0, 3, 6, 10, 20])
print(d[d.c <= 5].groupby('gapb', observed=True)[['dy', 'dlong', 'dstay']].mean().round(3).join(d[d.c <= 5].groupby('gapb', observed=True).size().rename('n')).to_string())
d['nb'] = d.n
print(d[d.c <= 5].groupby('nb', observed=True)[['dy', 'dlong', 'dstay']].mean().round(3).join(d[d.c <= 5].groupby('nb', observed=True).size().rename('n')).to_string())
