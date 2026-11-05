"""v7: spillover in the clone sweeps (v4-v6), round 0 -> rounds 1 and 4. Round-0 c is
exogenous; p0 follows from c0 and a rule that varies across 768 points. Regress a
player's later c on own c0 (dummies), own p0 (dummies 0,1-2,3-5,6-10,11-20,21-30),
others' mean c0 and others' p0: total, and its concentration (max)."""
import sys, numpy as np
sys.path.insert(0, '/opt/algorithmic-institutions/scripts/policy_finder/pf-20261105-3')
import v3_load
C, P, V = [], [], []
for v in ['pf-20261105-3-v4', 'pf-20261105-3-v5', 'pf-20261105-3-v6']:
    a = v3_load.load(v)
    C.append(a['c'].reshape(-1, 4, 24)); P.append(a['p'].reshape(-1, 4, 24)); V.append(a['v'].reshape(-1, 4, 24))
c = np.concatenate(C); p = np.concatenate(P); vv = np.concatenate(V)
ok = vv[:, :, :6].all((1, 2))
c, p = c[ok], p[ok]
G = c.shape[0]
print('games', G)
rows = []
for i in range(4):
    o = [j for j in range(4) if j != i]
    rows.append(np.stack([c[:, i, 0], p[:, i, 0], c[:, o, 0].mean(1), p[:, o, 0].sum(1), p[:, o, 0].max(1),
                          c[:, i, 1], c[:, i, 4], c[:, o, 4].mean(1), c[:, :, 1:6].mean((1, 2)), p[:, :, 1:6].mean((1, 2))], 1))
X = np.concatenate(rows)
c0, p0, oc, ops, opm, c1, c4, oc4, cg, pg = X.T
def bins(x, edges):
    return np.digitize(x, edges)
pb = bins(p0, [0.5, 2.5, 5.5, 10.5, 20.5])
cb = c0.astype(int)
ocb = bins(oc, [5, 8, 11, 14, 17])
# design: dummies for (cb), (pb), (ocb) + ops + opm
def dummies(z, n):
    D = np.zeros((len(z), n)); D[np.arange(len(z)), z] = 1; return D[:, 1:]
for nm, ycol in [('c1', c1), ('c4', c4), ('oc4 (others r4)', oc4)]:
    Xd = np.column_stack([np.ones(len(c0)), dummies(cb, 21), dummies(pb, 6), dummies(ocb, 6), ops, opm])
    beta, *_ = np.linalg.lstsq(Xd, ycol, rcond=None)
    print(nm, 'own p bins 1-2/3-5/6-10/11-20/21-30:', beta[21:26].round(2), 'others p sum', beta[-2].round(3), 'others p max', beta[-1].round(3))
# at fixed others' total punishment, concentrated vs spread: among players with own p0 = 0 and c0 >= 15
m = (p0 == 0) & (c0 >= 15)
for lo_, hi_ in [(8, 16), (16, 30), (30, 50)]:
    mm = m & (ops >= lo_) & (ops < hi_)
    conc = opm / np.maximum(ops, 1)
    for a_, b_ in [(0, 0.5), (0.5, 0.8), (0.8, 1.01)]:
        k = mm & (conc >= a_) & (conc < b_)
        if k.sum() > 200:
            print(f'others p {lo_}-{hi_} max share {a_}-{b_}: n {k.sum():6d} c0 {c0[k].mean():.2f} oc0 {oc[k].mean():.2f} c1-c0 {(c1 - c0)[k].mean():+.2f} c4-c0 {(c4 - c0)[k].mean():+.2f}')
