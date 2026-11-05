"""v6: sweep point s<k> uses the same seeds in every version (round-0 lineups identical), so the score of index k
shares a seed effect u_k across versions. Fit score = quadratic(T, A, log B, S) [+ give-up term for v3] + u_k
(index fixed effects) on the v2-v5 sweeps, compare with the fit without u_k, and find the peak."""
import numpy as np, json
B = 'plots/simulation/policy_finder/'
rows = []
for vi, ver in enumerate(['pf-20261105-11-v2', 'pf-20261105-11-v3', 'pf-20261105-11-v4', 'pf-20261105-11-v5']):
    for pt in json.load(open(f'{B}{ver}_sweep/sweep.json'))['points']:
        q = pt['params']; k = int(pt['name'].split('_s')[-1])
        rows.append([vi, k, q['T'], q['A'], q['B'], q['S'], q.get('N', 99), pt['pool'], pt['pool_se']])
R = np.array(rows)
vi, k, T, A, Bp, Sv, N, y, se = R.T
reg = (T >= 11) & (T <= 16.5) & (Bp >= 1.0) & (Bp <= 3.2) & (A <= 6) & (Sv >= 5) & (Sv <= 12) & (N >= 5)
print('points in region', reg.sum(), 'per version', [int((reg & (vi == j)).sum()) for j in range(4)])
def X(T, A, lB, Sv, N):
    g = np.minimum(N, 12)  # give-up count, 12 = effectively off for these S
    return np.column_stack([np.ones_like(T), T, A, lB, Sv, T*T, A*A, lB*lB, Sv*Sv, T*A, T*lB, A*lB, T*Sv, lB*Sv, A*Sv, g, g*g])
m = reg
Xq = X(T[m], A[m], np.log(Bp[m]), Sv[m], N[m]); yy = y[m]; kk = k[m].astype(int)
b0, *_ = np.linalg.lstsq(Xq, yy, rcond=None); r0 = yy - Xq @ b0
idx = np.unique(kk); D = (kk[:, None] == idx[None, :]).astype(float)
XF = np.column_stack([Xq[:, 1:], D])
bF, *_ = np.linalg.lstsq(XF, yy, rcond=None); rF = yy - XF @ bF
dof0 = len(yy) - Xq.shape[1]; dofF = len(yy) - np.linalg.matrix_rank(XF)
print(f'resid sd no FE {np.sqrt((r0**2).sum()/dof0):.2f}, with index FE {np.sqrt((rF**2).sum()/dofF):.2f} (dof {dofF}); mean pool_se {se[m].mean():.2f}')
uk = bF[Xq.shape[1]-1:]
print(f'sd of index effects {uk.std():.2f}')
bq = np.concatenate([[0], bF[:Xq.shape[1]-1]])
TT, AA, BB, SS = np.meshgrid(np.linspace(11.5, 16, 19), np.linspace(0, 5, 11), np.log(np.linspace(1.1, 3.0, 20)), np.arange(5, 13), indexing='ij')
G = X(TT.ravel(), AA.ravel(), BB.ravel(), SS.ravel(), np.full(TT.size, 12.0))
for name, b in [('noFE', b0), ('FE', bq)]:
    pr = G @ b; j = pr.argmax()
    off = (G @ b0).mean() - pr.mean() if name == 'FE' else 0
    print(name, f'peak T {TT.ravel()[j]:.2f} A {AA.ravel()[j]:.1f} B {np.exp(BB.ravel()[j]):.2f} S {SS.ravel()[j]}  rel value {pr[j]+off:.1f}')
    for nm, ar in [('T', TT), ('A', AA), ('B', BB), ('S', SS)]:
        prof = {}
        for val in np.unique(ar):
            mm = ar.ravel() == val
            prof[round(float(np.exp(val) if nm == 'B' else val), 2)] = round(float(pr[mm].max() + off), 1)
        print('  ', nm, prof)
np.save('/tmp/pf6/crn_beta.npy', bq)
# check against the paired GP finals (seed 1000): v4_gp - v5_gp = +0.98 +- 0.44, v4_gp - v3_gp = +4.2 (N=3)
for nm, q in [('v2_gp', (14.30, 1.70, 1.56, 8, 99)), ('v4_gp', (13.29, 2.43, 1.76, 8, 99)), ('v5_gp', (12.62, 2.83, 1.98, 7, 99)),
              ('v3_gp', (13.43, 2.21, 1.76, 7, 3)), ('v5_s089', (12.37, 1.40, 2.22, 7, 99)), ('ridge', (11.75, 5.0, 2.1, 8, 99))]:
    xx = X(*[np.array([float(v)]) for v in (q[0], q[1], np.log(q[2]), q[3], q[4])])
    print(nm, f'FE model {float(xx @ bq) + (G @ b0).mean() - (G @ bq).mean():.2f}  noFE {float(xx @ b0):.2f}')
