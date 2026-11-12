"""v5: fit the natural experiments (v5_natexp.py): per divergence round r*, outcome difference (lead from r* on, per
round of the game) on the punishment-point difference at r*, split by target categories. OLS through the origin."""
import numpy as np, sys
R = np.load('/tmp/pf/natexp.npy')
r, dO, dT, dM = R[:, 0], R[:, 4], R[:, 5], R[:, 6]
dp, c, mine, new, n = R[:, 7:15], R[:, 15:23], R[:, 23:31], R[:, 31:39], R[:, 39]
lead = dO - dT
def fit(sel, feats, names, label):
    X = np.stack([f[sel] for f in feats], 1) / 10.0
    out = []
    for nm, y in [('lead', lead), ('own', dO), ('oth', dT), ('mem', dM)]:
        yy = y[sel]
        b, *_ = np.linalg.lstsq(X, yy, rcond=None); res = yy - X @ b
        # robust (HC0) se
        XtXi = np.linalg.pinv(X.T @ X); meat = (X * res[:, None] ** 2).T @ X; se = np.sqrt(np.diag(XtXi @ meat @ XtXi))
        out.append(f'  {nm}: ' + ' '.join(f'{a}:{bb:+.2f}({ss:.2f})' for a, bb, ss in zip(names, b, se)))
    print(label, 'n=%d' % sel.sum(), 'mean |dp| per game %.1f' % np.abs(dp[sel]).sum(1).mean()); print('\n'.join(out))
for rs in [0, 1, 2, 3, 4, 5, 8, 12, 16, 20]:
    sel = r == rs
    if sel.sum() < 200: continue
    fit(sel, [dp.sum(1)], ['all'], f'r*={rs}')
    bands = [(0, 2), (3, 5), (6, 9), (10, 20)]
    fit(sel, [(dp * ((c >= lo) & (c <= hi))).sum(1) for lo, hi in bands], [f'c{lo}-{hi}' for lo, hi in bands], f'r*={rs} by c')
    if rs > 0:
        fit(sel, [(dp * new).sum(1), (dp * (1 - new)).sum(1)], ['new', 'stay'], f'r*={rs} by newcomer')
        fit(sel, [(dp * new * (c <= 5)).sum(1), (dp * (1 - new) * (c <= 5)).sum(1), (dp * (c > 5)).sum(1)], ['new<=5', 'stay<=5', 'c>5'], f'r*={rs} by newcomer, low c')
    fit(sel, [dp.sum(1) * (n <= 3), dp.sum(1) * (n == 4), dp.sum(1) * (n == 5), dp.sum(1) * (n >= 6)], ['n<=3', 'n4', 'n5', 'n>=6'], f'r*={rs} by my size')
