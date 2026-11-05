# v7: value of a round-1 punishment point, first-time vs repeat, on the member's own total
# contribution to our pool over rounds 2-23 (1.6 x c while in our group), per opponent.
# Pre-treatment controls only (c0, c1, group means, p0, rep). Also the groupmates' spillover:
# sum over the other originals of the same outcome, regressed on the group's round-1 punishment
# split into first-time and repeat points.
import numpy as np
d = np.load('/tmp/pf-20261106-9-v6_sweep.npz')
c = d['c'].astype(float); p = d['p'].astype(float); g = d['g']; ours = d['ours']; opp = d['opp']
mine = (g == ours[:, None, None])
orig = mine[:, :, 0] & mine[:, :, 1]
def gm(t):
    m = mine[:, :, t]; return ((c[:, :, t] * m).sum(1) / np.maximum(m.sum(1), 1))[:, None] * np.ones((1, 8))
m0, m1 = gm(0), gm(1)
val = 1.6 * (c[:, :, 2:] * mine[:, :, 2:]).sum(-1)
O = np.repeat(opp[:, None], 8, 1)
p0, p1 = p[:, :, 0] * orig, p[:, :, 1] * orig
rep = (p0 > 0).astype(float)
def ols(y, X, names, mask):
    X = np.column_stack([np.ones(mask.sum())] + [x[mask] for x in X]); y = y[mask]
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    r = y - X @ b; s2 = r @ r / (len(y) - X.shape[1]); se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
    print('  n', mask.sum(), ' '.join(f'{n} {bb:+.3f}({ss:.3f})' for n, bb, ss in zip(['1'] + names, b, se) if n[0] in 'pP' or n == 'rep'))
X = [c[:, :, 0], c[:, :, 1], m0, m1, p0, p1 * (1 - rep), p1 * rep, rep]
names = ['c0', 'c1', 'm0', 'm1', 'p0', 'p1_first', 'p1_rep', 'rep']
# spillover: groupmates' value vs the rest of the group's p1 split
P1f = (p1 * (1 - rep)).sum(1, keepdims=True) - p1 * (1 - rep)
P1r = (p1 * rep).sum(1, keepdims=True) - p1 * rep
P0o = p0.sum(1, keepdims=True) - p0
for o in [0, 1, 2, 3, 4]:
    om = orig & ((O == o) if o else True)
    print('opp', o or 'all', 'own value 2-23'); ols(val, X, names, om)
    print('   with others p0/p1 split'); ols(val, X + [P0o, P1f, P1r], names + ['P0oth', 'P1f_oth', 'P1r_oth'], om)
