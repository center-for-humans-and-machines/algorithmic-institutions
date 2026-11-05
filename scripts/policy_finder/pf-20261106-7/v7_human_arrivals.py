# v7: human data. At decisions 2-5 (after rounds 7, 11, 15, 19), exit rate of arrivals (joined at the previous
# decision) vs stayers, by the punishment received in the block; and a pooled linear probability fit.
import sys; sys.path.insert(0, '/opt/algorithmic-institutions/scripts/policy_finder/pf-20261106-7')
import numpy as np
from load import load
df = load()
def arr3(col):
    return df.pivot_table(index=['game', 'player_id'], columns='round_number', values=col).to_numpy().reshape(-1, 8, 24)
C, P, G = arr3('contribution'), arr3('punishment'), arr3('group_id')
rows = []
for dec in [7, 11, 15, 19]:
    b0 = dec - 3
    arr = (G[:, :, b0] != G[:, :, b0 - 1])
    left = (G[:, :, dec + 1] != G[:, :, dec])
    pb = P[:, :, b0:dec + 1].sum(2)
    p0 = P[:, :, b0]
    cb = C[:, :, b0:dec + 1].mean(2)
    for g in range(len(C)):
        for i in range(8):
            rows.append((dec, arr[g, i], left[g, i], pb[g, i], p0[g, i], cb[g, i]))
R = np.array(rows, float)
for a, nm in [(0, 'stayers'), (1, 'arrivals')]:
    for lo, hi in [(0, 0.5), (0.5, 6), (6, 1e9)]:
        s = (R[:, 1] == a) & (R[:, 3] >= lo) & (R[:, 3] < hi)
        print(f'{nm:8s} block punishment [{lo},{hi}) n {s.sum():4d} exit {R[s, 2].mean():.3f} c {R[s, 5].mean():.1f}')
X = np.c_[np.ones(len(R)), R[:, 1], np.sqrt(R[:, 3]), R[:, 1] * np.sqrt(R[:, 3]), R[:, 5]]
b = np.linalg.lstsq(X, R[:, 2], rcond=None)[0]
r = R[:, 2] - X @ b; se = np.sqrt(np.diag((r**2).sum() / (len(r) - 5) * np.linalg.inv(X.T @ X)))
print('exit ~ 1 + arr + sqrt(pblock) + arr*sqrt(pblock) + c:', np.round(b, 4), 'se', np.round(se, 4))
