# v7: group-level marginal value of round-0 punishment, by the group's round-0 mean m0.
# Outcome: game pool (mean over 24 rounds, punishment costs included) or pool rounds 1-23 sum.
# Treatment: total round-0 punishment P0 (design-driven given the round-0 contributions).
# Controls: sorted round-0 contributions of our 4 members (exogenous), their squares, opponent.
# Also P0^2 to see curvature.
import numpy as np
d = np.load('/tmp/pf-20261106-9-v6_sweep.npz')
c = d['c'].astype(float); p = d['p'].astype(float); g = d['g']; ours = d['ours']; opp = d['opp']
mine = (g == ours[:, None, None])
poolr = 1.6 * (c * mine).sum(1) - (p * mine).sum(1)
tot = poolr.sum(1)  # whole-game pool sum (includes the round-0 cost)
c0 = c[:, :, 0]; m0 = (c0 * mine[:, :, 0]).sum(1) / 4
c0s = np.sort(np.where(mine[:, :, 0], c0, np.inf), 1)[:, :4]
P0 = (p[:, :, 0] * mine[:, :, 0]).sum(1)
P1 = (p[:, :, 1] * mine[:, :, 1]).sum(1)
qm = np.quantile(m0, [1 / 3, 2 / 3]); b = np.digitize(m0, qm)
def ols(y, X):
    X = np.column_stack([np.ones(len(y))] + X)
    bb = np.linalg.lstsq(X, y, rcond=None)[0]
    r = y - X @ bb; s2 = r @ r / (len(y) - X.shape[1]); se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
    return bb, se
for t in range(3):
    s = b == t
    ctrl = [c0s[s][:, j] for j in range(4)] + [c0s[s][:, j] ** 2 for j in range(4)] + [(opp[s] == o).astype(float) for o in (2, 3, 4)]
    bb, se = ols(tot[s], [P0[s]] + ctrl)
    print(f'm0 tercile {t} (m0 {m0[s].mean():.2f}, P0 mean {P0[s].mean():.1f}, P1 {P1[s].mean():.1f}): d game-pool-sum / dP0 {bb[1]:+.2f} ({se[1]:.2f})')
    bb, se = ols(tot[s], [P0[s], P0[s] ** 2] + ctrl)
    print(f'     quadratic: P0 {bb[1]:+.3f} P0^2 {bb[2]:+.5f} -> peak P0 {-bb[1] / (2 * bb[2]) if bb[2] < 0 else float("nan"):.1f}')
