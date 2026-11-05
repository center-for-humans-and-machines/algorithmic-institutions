# v7: in v6's sweep, original members: does a round-1 punishment work differently for members
# already punished in round 0 (repeat) than for first-timers? Outcomes: stay at round 4, mean
# contribution rounds 8-15 when in our group. OLS with controls c0, c1, group mean in rounds 0/1.
import numpy as np
d = np.load('/tmp/pf-20261106-9-v6_sweep.npz')
c = d['c'].astype(float); p = d['p'].astype(float); g = d['g']; ours = d['ours']; opp = d['opp']
mine = (g == ours[:, None, None])
orig = mine[:, :, 0]
def gm(t):
    m = mine[:, :, t]; return ((c[:, :, t] * m).sum(1) / np.maximum(m.sum(1), 1))[:, None] * np.ones((1, 8))
m0, m1 = gm(0), gm(1)
stay = mine[:, :, 4].astype(float)
late_m = mine[:, :, 8:16]
late = (c[:, :, 8:16] * late_m).sum(-1) / np.maximum(late_m.sum(-1), 1)
haslate = late_m.sum(-1) >= 4
O = np.repeat(opp[:, None], 8, 1)
sel = orig & mine[:, :, 1]
p0, p1 = p[:, :, 0], p[:, :, 1]
rep = (p0 > 0).astype(float)
def ols(y, X, names, mask):
    X = np.column_stack([np.ones(mask.sum())] + [x[mask] for x in X]); y = y[mask]
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    r = y - X @ b; s2 = r @ r / (len(y) - X.shape[1]); se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
    print('  n', mask.sum(), ' '.join(f'{n} {bb:+.4f}({ss:.4f})' for n, bb, ss in zip(['1'] + names, b, se)))
X = [c[:, :, 0], c[:, :, 1], m0, m1, p0, p1 * (1 - rep), p1 * rep, rep]
names = ['c0', 'c1', 'm0', 'm1', 'p0', 'p1_first', 'p1_rep', 'rep']
for o in [0, 1, 2, 3, 4]:
    om = sel & ((O == o) if o else True)
    print('opp', o or 'all', 'stay4'); ols(stay, X, names, om)
    print('opp', o or 'all', 'late c | stayed'); ols(late, X, names, om & haslate)
# binned: round-1 shortfall 1-4, by repeat status and p1 dose
s1 = m1 - c[:, :, 1]
for r in [0, 1]:
    for lo, hi in [(0, 1), (1, 8), (8, 16), (16, 31)]:
        mk = sel & (s1 > 1) & (s1 < 4) & (rep == r) & (p1 >= lo) & (p1 < hi)
        mk2 = mk & haslate
        print(f'rep {r} p1 {lo}-{hi}: n {mk.sum():6d} stay {stay[mk].mean():.3f} late {late[mk2].mean():.2f} c1 {c[:, :, 1][mk].mean():.2f}')
