"""v4: in v3's sweep (part 1), how leaving at a decision depends on the punishment received in
each of the 4 rounds of the block before it (position 0..3), for members of our group at the
decision round; linear probability model with controls (own contribution in the block, own
and other group per-member pool share at the decision round, own group size).
Also: next-round contribution response by position in block (does a punishment at block
position 0 lift contributions as much as at position 2?)."""
import sys, numpy as np
sys.path.insert(0, 'scripts/policy_finder/pf-20261104-5')
from v3_load import load
A = load('plots/simulation/policy_finder/pf-20261104-5-v3_sweep/per_round_p1of2.parquet')
c, p, g, v, cg = A['c'], A['p'], A['g'], A['v'], A['cg']
def ols(y, X, names):
    X = np.column_stack([np.ones(len(y))] + X)
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ b
    se = np.sqrt(np.diag(r @ r / (len(y) - X.shape[1]) * np.linalg.pinv(X.T @ X)))
    for n, bb, ss in zip(['const'] + names, b, se):
        print(f'   {n:12s} {100*bb:8.3f}pp ({100*ss:.3f})')
for d in [3, 7, 11]:
    m = (g[:, d] == 0) & (g[:, d - 3] == 0)
    idx = np.nonzero(m)
    gi, pi = idx
    y = (g[gi, d + 1, pi] != 0).astype(float)
    # group shares at d
    own_n = (g[:, d] == 0).sum(1)[gi]
    ocg = np.array([cg[gi, d][k][g[gi, d][k] == 0][0] for k in range(len(gi))]) if False else None
    sh0 = np.where((g[:, d] == 0), cg[:, d], np.nan); sh1 = np.where((g[:, d] == 1), cg[:, d], np.nan)
    s0 = np.nanmean(sh0, 1)[gi]; s1 = np.nan_to_num(np.nanmean(sh1, 1), nan=0)[gi]
    X = [p[gi, d - 3 + j, pi] for j in range(4)] + [c[gi, d - 3:d + 1, pi].mean(1), s0, s1, own_n]
    print(f'== decision after round {d}: n={len(y)}, leave rate {y.mean():.3f}')
    ols(y, X, ['p_pos0', 'p_pos1', 'p_pos2', 'p_pos3', 'c_block', 'share_own', 'share_oth', 'n_own'])
# contribution response: c(t+1)-c(t) on p(t) by position (t in 0..2, 4..6, 8..10), stayers in our group, c(t)<10
print('== contribution response c(t+1)-c(t) per punishment point, stayers, c(t) < 10')
for t in [0, 1, 2, 4, 5, 6, 8, 9, 10]:
    m = (g[:, t] == 0) & (g[:, t + 1] == 0) & v[:, t] & v[:, t + 1] & (c[:, t] < 10)
    y = (c[:, t + 1] - c[:, t])[m]
    X = np.column_stack([np.ones(m.sum()), p[:, t][m], c[:, t][m]])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    print(f'   t={t:2d} slope {b[1]:.3f}  n={m.sum()}')
