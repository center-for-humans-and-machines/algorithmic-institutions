"""v5: natural experiments with the later punishment handled: (a) control for later points difference, (b) only pairs
whose later points are equal (difference 0). Outcomes per 10 points at r*, per round of the game."""
import numpy as np
R = np.load('/tmp/pf/natexp.npy')
r, dO, dT, dM = R[:, 0], R[:, 4], R[:, 5], R[:, 6]
dp, c, mine, new, n, ldp, lsum = R[:, 7:15], R[:, 15:23], R[:, 23:31], R[:, 31:39], R[:, 39], R[:, 40], R[:, 41]
lead = dO - dT
def fit(sel, feats, names, label, ys=('lead', 'own', 'oth', 'mem')):
    X = np.stack([f[sel] for f in feats], 1) / 10.0
    print(label, 'n=%d' % sel.sum(), 'mean |dp| %.1f' % np.abs(dp[sel]).sum(1).mean())
    for nm, y in [('lead', lead), ('own', dO), ('oth', dT), ('mem', dM)]:
        if nm not in ys: continue
        yy = y[sel]; b, *_ = np.linalg.lstsq(X, yy, rcond=None); res = yy - X @ b
        XtXi = np.linalg.pinv(X.T @ X); meat = (X * res[:, None] ** 2).T @ X; se = np.sqrt(np.diag(XtXi @ meat @ XtXi))
        print(f'  {nm}: ' + ' '.join(f'{a}:{bb:+.2f}({ss:.2f})' for a, bb, ss in zip(names, b, se)))
bands = [(0, 2), (3, 5), (6, 9), (10, 20)]
for rs in [0, 1, 4, 5, 8, 12, 16, 20]:
    base = r == rs
    for lab, sel in [('ctrl later', base), ('later equal', base & (np.abs(ldp) < 0.5)), ('none later', base & (lsum < 0.5))]:
        if sel.sum() < 100: continue
        extra = [ldp] if lab == 'ctrl later' else []
        en = ['later'] if extra else []
        fit(sel, [dp.sum(1)] + extra, ['all'] + en, f'r*={rs} [{lab}]')
        fit(sel, [(dp * ((c >= lo) & (c <= hi))).sum(1) for lo, hi in bands] + extra, [f'c{lo}-{hi}' for lo, hi in bands] + en, f'r*={rs} [{lab}] by c', ys=('lead', 'own', 'oth'))
        if rs > 0:
            fit(sel, [(dp * new * (c <= 5)).sum(1), (dp * (1 - new) * (c <= 5)).sum(1), (dp * (c > 5)).sum(1)] + extra, ['new<=5', 'stay<=5', 'c>5'] + en, f'r*={rs} [{lab}] by newcomer', ys=('lead', 'own', 'oth'))
