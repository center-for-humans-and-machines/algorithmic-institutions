"""v6: S profile with S as categories (dummies), quadratic in T, A, log B, index fixed effects; v2-v5 sweeps."""
import numpy as np, json
B = 'plots/simulation/policy_finder/'
rows = []
for vi, ver in enumerate(['pf-20261105-11-v2', 'pf-20261105-11-v3', 'pf-20261105-11-v4', 'pf-20261105-11-v5']):
    for pt in json.load(open(f'{B}{ver}_sweep/sweep.json'))['points']:
        q = pt['params']; k = int(pt['name'].split('_s')[-1])
        rows.append([vi, k, q['T'], q['A'], q['B'], q['S'], q.get('N', 99), pt['pool']])
R = np.array(rows); vi, k, T, A, Bp, Sv, N, y = R.T
for nmin in [99, 5]:
    m = (T >= 11) & (T <= 16.5) & (Bp >= 1.0) & (Bp <= 3.2) & (A <= 6) & (Sv >= 4) & (Sv <= 14) & (N >= nmin)
    lB = np.log(Bp[m]); kk = k[m].astype(int); idx = np.unique(kk)
    Ss = np.arange(4, 15); SD = (Sv[m][:, None] == Ss[None, 1:]).astype(float)
    g = np.minimum(N[m], 14)
    Xq = np.column_stack([T[m], A[m], lB, T[m]**2, A[m]**2, lB**2, T[m]*A[m], T[m]*lB, A[m]*lB, g, g*g, SD, (kk[:, None] == idx[None, :])])
    b, *_ = np.linalg.lstsq(Xq, y[m], rcond=None)
    res = y[m] - Xq @ b
    sd = np.sqrt((res**2).sum() / (len(res) - np.linalg.matrix_rank(Xq)))
    cov = sd**2 * np.linalg.pinv(Xq.T @ Xq)
    off = 11
    prof = np.concatenate([[0], b[off:off + len(Ss) - 1]]); se = np.concatenate([[0], np.sqrt(np.diag(cov))[off:off + len(Ss) - 1]])
    cnt = [(Sv[m] == s).sum() for s in Ss]
    print(f'N>={nmin}: n {m.sum()}, resid sd {sd:.2f}. S effect relative to S=4:')
    print('  ' + ' '.join(f'S{s}:{p - prof[4]:+.1f}±{e:.1f}[{c}]' for s, p, e, c in zip(Ss, prof, se, cnt)), '(shown relative to S=8)')
