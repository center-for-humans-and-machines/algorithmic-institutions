"""v5: slot fixed-effects regression of the sweep points' lead (and own/other pool, members) on the punishment
points dealt per game in each phase of the game, over all four versions' sweeps (1024 points, 256 shared seed slots)."""
import numpy as np, sys
z = np.load('/tmp/pf/feats.npz')
V = sys.argv[1].split(',') if len(sys.argv) > 1 else ['v1', 'v2', 'v3', 'v4']
split = len(sys.argv) > 2 and sys.argv[2] == 'split'
phases = {'r0': [0], 'r1-3': [1, 2, 3], 'r4': [4], 'r5-7': [5, 6, 7], 'r8': [8], 'r9-11': [9, 10, 11], 'r12': [12],
          'r13-19': list(range(13, 20)), 'r20-23': [20, 21, 22, 23]}
X, Y, S, names = [], {k: [] for k in ['lead', 'own', 'oth', 'mem']}, [], []
for v in V:
    F = z[f'{v}_F']  # [pts, 3 bins, 24]
    cols = []
    for ph, rr in phases.items():
        if split:
            cols += [F[:, 0, rr].sum(1), F[:, 1:, rr].sum((1, 2))]
        else:
            cols.append(F[:, :, rr].sum((1, 2)))
    X.append(np.stack(cols, 1) / 10.0); S.append(z[f'{v}_slot'])
    for k in Y: Y[k].append(z[f'{v}_{k}'])
names = [f'{ph}{b}' for ph in phases for b in (['(c<=5)', '(c>5)'] if split else [''])]
X = np.concatenate(X); S = np.concatenate(S)
def demean(a):
    a = a.copy().astype(float)
    for s in np.unique(S):
        i = S == s; a[i] -= a[i].mean(0)
    return a
Xd = demean(X)
print('versions', V, 'n', len(S), '(coefficients per 10 points dealt per game)')
for k in Y:
    y = np.concatenate(Y[k]); yd = demean(y)
    b, *_ = np.linalg.lstsq(Xd, yd, rcond=None)
    r = yd - Xd @ b; dof = len(y) - len(np.unique(S)) - X.shape[1]
    s2 = (r ** 2).sum() / dof; cov = s2 * np.linalg.pinv(Xd.T @ Xd); se = np.sqrt(np.diag(cov))
    print(k, 'resid sd %.2f' % np.sqrt(s2), ' '.join(f'{n}:{bb:+.2f}({ss:.2f})' for n, bb, ss in zip(names, b, se)))
