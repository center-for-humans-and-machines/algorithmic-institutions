"""v4: does the cost of punishment in members depend on group size? Per-person switch
probability at decisions 7, 11, 15 (OLS, linear prob.) on the block-mean common good of the
person's own group and the other group, for members of the rule's group (leavers) and of ah's
group (joiners). Then, at the group level: net member gain on the rule group's total
punishment in the block, by rule-group size at the block."""
import sys, numpy as np, pandas as pd
z = np.load(sys.argv[1]); c, p, g, v = (z[k].astype(np.float32) for k in 'cpgv')
c = c.reshape(-1, 8, 24); p = p.reshape(-1, 8, 24); g = g.reshape(-1, 8, 24); v = v.reshape(-1, 8, 24)
def cg(grp):
    m = (g == grp)*v; n = m.sum(1)
    return np.where(n > 0, ((1.6*c - p)*m).sum(1)/np.maximum(n, 1), np.nan)
cg0, cg1 = cg(0), cg(1); n0 = (g == 0).sum(1)
for d in (7, 11, 15):
    rr = slice(d-3, d+1)
    a0 = np.nanmean(cg0[:, rr], 1); a1 = np.nanmean(cg1[:, rr], 1)
    for grp, own, oth, lab in ((0, a0, a1, 'leave rule grp'), (1, a1, a0, 'leave ah grp')):
        m = g[:, :, d] == grp
        sw = (g[:, :, d+1] != g[:, :, d])
        e, i = np.nonzero(m)
        y = sw[e, i].astype(float); X = np.column_stack([np.ones(len(y)), own[e], oth[e]])
        ok = ~np.isnan(X).any(1)
        b = np.linalg.lstsq(X[ok], y[ok], rcond=None)[0]
        print(f'dec {d} {lab}: base {b[0]:.3f} own cg {b[1]:.4f} other cg {b[2]:.4f}')
    # group-level: net gain on rule group's total punishment in block, by size
    y = n0[:, d+1] - n0[:, d]
    P = (p[:, :, rr]*(g[:, :, rr] == 0)).sum((1, 2))
    C = (c[:, :, rr]*(g[:, :, rr] == 0)).sum((1, 2))/np.maximum((g[:, :, rr] == 0).sum((1, 2)), 1)
    for nlo, nhi in ((1, 3), (4, 4), (5, 8)):
        m = (n0[:, d] >= nlo) & (n0[:, d] <= nhi)
        X = np.column_stack([np.ones(m.sum()), P[m], C[m], np.nan_to_num(a1[m])])
        b = np.linalg.lstsq(X, y[m], rcond=None)[0]
        print(f'   size {nlo}-{nhi} (n={m.sum()}): net gain per 10 pts total punishment {10*b[1]:.3f}, per pt mean contribution {b[2]:.3f}')
