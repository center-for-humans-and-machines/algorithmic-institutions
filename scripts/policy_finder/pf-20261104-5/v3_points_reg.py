"""v3: regress v2 sweep point outcomes on where the punishment went (v3_points.csv)."""
import numpy as np, pandas as pd
D = pd.read_csv('scripts/policy_finder/pf-20261104-5/v3_points.csv')
def reg(y, xs, d=D):
    X = np.c_[np.ones(len(d)), d[xs].values]
    b, *_ = np.linalg.lstsq(X, d[y].values, rcond=None)
    res = d[y].values - X @ b
    s2 = res.var() * len(d) / (len(d) - X.shape[1])
    se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
    r2 = 1 - res.var() / d[y].var()
    print(f'{y} R2={r2:.2f}'); [print(f'  {n:8s} {bb:8.4f} ({s:.4f})') for n, bb, s in zip(['const'] + xs, b, se)]
xs = ['P_c0_2', 'P_c3_6', 'P_c7_10', 'P_c11_', 'P_late']
for y in ['score', 'cm_late', 'n_late']: reg(y, xs)
reg('score', ['P_dec', 'P_non', 'P_late'])
D['P_tot'] = D.P_dec + D.P_non
reg('score', ['P_c0_2', 'P_c3_6', 'P_c7_10', 'P_c11_', 'P_late', 'P_dec'])
print(D.sort_values('score', ascending=False).head(12)[['run', 'score', 'P_c0_2', 'P_c3_6', 'P_c7_10', 'P_c11_', 'P_dec', 'P_late', 'n_late', 'cm_late', 'k', 'thr', 'cap', 'decay', 't_stop']].round(1).to_string())
