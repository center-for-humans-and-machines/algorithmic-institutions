"""v4: regress v3-sweep point outcomes on where/when punishment went (v4_points.csv)."""
import numpy as np, pandas as pd
D = pd.read_csv('scripts/policy_finder/pf-20261104-5/v4_points.csv')
def ols(y, X):
    Xm = np.column_stack([np.ones(len(D))] + [D[x] for x in X])
    b, *_ = np.linalg.lstsq(Xm, D[y], rcond=None)
    r = D[y] - Xm @ b
    s2 = r @ r / (len(D) - Xm.shape[1])
    se = np.sqrt(np.diag(s2 * np.linalg.inv(Xm.T @ Xm)))
    print(f'{y}: R2={1 - r.var() / D[y].var():.2f}')
    for n, bb, ss in zip(['const'] + X, b, se):
        print(f'   {n:10s} {bb:8.3f} ({ss:.3f})')
X1 = ['P0_c0_2', 'P0_c3_6', 'P0_c7_10', 'P0_c11_', 'P3_c0_2', 'P3_c3_6', 'P3_c7_10', 'P3_c11_']
for y in ['score', 'oscore', 'cm5', 'n1', 'n5']:
    ols(y, X1)
X2 = ['P_r0', 'P_r1', 'P_r2', 'P_r3', 'P_r4', 'P_r5', 'P_r6_11', 'P_r12_']
for y in ['score', 'cm3', 'n1']:
    ols(y, X2)
print(D[['score', 'oscore', 'n1', 'n5', 'cm0', 'cm1', 'cm5', 'pool0']].corr().round(2))
print(D.sort_values('score', ascending=False).head(12)[['run', 'score', 'k', 'thr', 'cap', 'budget', 'quiet', 't_stop', 'n1', 'n5', 'cm1', 'cm5', 'P_r0', 'P_r12_']].round(2).to_string())
