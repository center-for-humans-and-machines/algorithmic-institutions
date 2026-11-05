"""v6: the index-FE fit of f6 puts its peak at the low-T / high-A corner. Is that a real open direction?
Refit on a wider region (T 10-16.5, A 0-6, incl. v2), and look at the raw FE-adjusted residual means near the
corner versus near v4's peak (no quadratic assumed): local comparisons of points' scores minus their index effect."""
import numpy as np, json
B = 'plots/simulation/policy_finder/'
rows = []
for vi, ver in enumerate(['pf-20261105-11-v2', 'pf-20261105-11-v3', 'pf-20261105-11-v4', 'pf-20261105-11-v5']):
    for pt in json.load(open(f'{B}{ver}_sweep/sweep.json'))['points']:
        q = pt['params']; k = int(pt['name'].split('_s')[-1])
        rows.append([vi, k, q['T'], q['A'], q['B'], q['S'], q.get('N', 99), pt['pool']])
R = np.array(rows); vi, k, T, A, Bp, Sv, N, y = R.T
reg = (T >= 10) & (T <= 16.5) & (Bp >= 1.0) & (Bp <= 3.2) & (Sv >= 5) & (Sv <= 12) & (N >= 5)
def X(T, A, lB, Sv):
    return np.column_stack([T, A, lB, Sv, T*T, A*A, lB*lB, Sv*Sv, T*A, T*lB, A*lB, T*Sv, lB*Sv, A*Sv])
m = reg; kk = k[m].astype(int); idx = np.unique(kk); D = (kk[:, None] == idx[None, :]).astype(float)
g = np.minimum(N[m], 12)
XF = np.column_stack([X(T[m], A[m], np.log(Bp[m]), Sv[m]), g, g*g, D])
bF, *_ = np.linalg.lstsq(XF, y[m], rcond=None)
nq = 14
u = dict(zip(idx, bF[nq+2:]))
yadj = y[m] - np.array([u[i] for i in kk])  # score with its seed effect removed
print('n', m.sum(), 'resid sd', np.std(y[m] - XF @ bF))
# fine shape descriptors: fine at c=12 and at c=6 and at c=0
def fine(c, T, A, B): return np.where(c < T, np.minimum(30, A + B*(T - c)), 0)
Tm, Am, Bm, Sm = T[m], A[m], Bp[m], Sv[m]
f12, f9, f5, f0 = fine(12, Tm, Am, Bm), fine(9, Tm, Am, Bm), fine(5, Tm, Am, Bm), fine(0, Tm, Am, Bm)
off = np.mean(yadj) - 0
print('FE-adjusted mean score by (T bin, A bin), S 6-10, B 1.5-2.6 [n]:')
sel = (Sm >= 6) & (Sm <= 10) & (Bm >= 1.5) & (Bm <= 2.6)
for tlo in np.arange(10, 16.5, 1):
    out = []
    for alo in [0, 1.5, 3, 4.5]:
        mm = sel & (Tm >= tlo) & (Tm < tlo + 1) & (Am >= alo) & (Am < alo + 1.5)
        out.append(f'{yadj[mm].mean():6.1f}[{mm.sum():2d}]' if mm.sum() >= 3 else '   -     ')
    print(f' T {tlo:4.1f}-{tlo+1:4.1f}:', ' '.join(out), '  (A 0-1.5, 1.5-3, 3-4.5, 4.5-6)')
print('by fine at c=12 (rule fines near-target players?):')
for a, b in [(0, 0.01), (0.01, 3), (3, 5), (5, 7), (7, 12)]:
    mm = sel & (f12 >= a) & (f12 < b)
    print(f'  f12 [{a},{b}) n{mm.sum():3d} {yadj[mm].mean():6.1f}  f5 {f5[mm].mean():5.1f} f0 {f0[mm].mean():5.1f}')
print('by fine at c=9:')
for a, b in [(0, 4), (4, 7), (7, 9), (9, 11), (11, 14), (14, 30)]:
    mm = sel & (f9 >= a) & (f9 < b)
    print(f'  f9 [{a},{b}) n{mm.sum():3d} {yadj[mm].mean():6.1f}  f12 {f12[mm].mean():5.1f} f0 {f0[mm].mean():5.1f}')
print('by fine at c=0:')
for a, b in [(0, 18), (18, 22), (22, 25), (25, 28), (28, 30.1)]:
    mm = sel & (f0 >= a) & (f0 < b)
    print(f'  f0 [{a},{b}) n{mm.sum():3d} {yadj[mm].mean():6.1f}  f9 {f9[mm].mean():5.1f} f12 {f12[mm].mean():5.1f}')
np.savez('/tmp/pf6/fe_points.npz', T=Tm, A=Am, B=Bm, S=Sm, yadj=yadj, ver=vi[m])
