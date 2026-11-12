"""v6: spillover of round-0 punishment. Pairs of aligned games (same slot, seat, episode; two versions), identical
through round 0's contributions. Per my-group member: d(own punishment at 0), d(others' punishment at 0, summed over
the other members of my group), outcome d(mean contribution rounds 1-3). Regressions on the pooled pairs, plus the
group-level outcome (my group's total contribution rounds 1-3, and members at round 4..7)."""
import numpy as np, itertools
V = ['v1', 'v2', 'v3', 'v4', 'v5']
A = {v: np.load(f'/tmp/pf/al_{v}.npz') for v in V}
rows_own, rows_oth, rows_y, rows_c0, rows_stay = [], [], [], [], []
G = {'dP': [], 'dy': [], 'dmem': [], 'dn': []}
rng = np.random.default_rng(0)
for a, b in itertools.combinations(V, 2):
    za, zb = A[a], A[b]
    m = za['mine'][:, :, 0].astype(bool)
    pa, pb = za['p'][:, :, 0].astype(float), zb['p'][:, :, 0].astype(float)
    ca, cb = za['c'][:, :, 1:4].astype(float).mean(2), zb['c'][:, :, 1:4].astype(float).mean(2)
    oa = (pa * m).sum(1, keepdims=True) - pa; ob = (pb * m).sum(1, keepdims=True) - pb
    sa = za['mine'][:, :, 4].astype(float); sb = zb['mine'][:, :, 4].astype(float)
    sel = m & (za['v'][:, :, 0] == 1)
    rows_own.append((pa - pb)[sel]); rows_oth.append((oa - ob)[sel]); rows_y.append((ca - cb)[sel])
    rows_c0.append(za['c'][:, :, 0][sel]); rows_stay.append((sa - sb)[sel])
    # group level
    dP = ((pa - pb) * m).sum(1)
    tot_a = (za['c'][:, :, 1:4] * m[:, :, None]).sum((1, 2)) / 3; tot_b = (zb['c'][:, :, 1:4] * m[:, :, None]).sum((1, 2)) / 3
    G['dP'].append(dP); G['dy'].append(tot_a - tot_b)
    G['dmem'].append(za['mine'][:, :, 4].sum(1).astype(float) - zb['mine'][:, :, 4].sum(1))
    G['dn'].append(((pa > 0) & m).sum(1).astype(float) - ((pb > 0) & m).sum(1))
own = np.concatenate(rows_own); oth = np.concatenate(rows_oth); y = np.concatenate(rows_y); c0 = np.concatenate(rows_c0)
stay = np.concatenate(rows_stay)
print('pairs x members', len(y), 'with own diff', (own != 0).sum(), 'with others diff', (oth != 0).sum())
def ols(X, y, names):
    X = np.column_stack([X, np.ones(len(y))]); b, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ b; s2 = r @ r / (len(y) - X.shape[1]); se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
    print('  ' + ' '.join(f'{n}:{bb:+.4f}({ss:.4f})' for n, bb, ss in zip(names + ['1'], b, se)))
# unpunished in both: response to others' punishment
u = (own == 0)
print('members unpunished in both pairs', u.sum())
for lo, hi in [(0, 5), (6, 10), (11, 20)]:
    s = u & (c0 >= lo) & (c0 <= hi)
    print(f'c0 {lo}-{hi} n={s.sum()}: d contribution r1-3 per point to others')
    ols(oth[s] / 10, y[s], ['oth/10'])
    print('   stay to round 4:'); ols(oth[s] / 10, stay[s], ['oth/10'])
print('all members, own and others')
for lo, hi in [(0, 5), (6, 10)]:
    s = (c0 >= lo) & (c0 <= hi)
    ols(np.column_stack([own[s] / 10, oth[s] / 10]), y[s], ['own/10', 'oth/10'])
    ols(np.column_stack([own[s] / 10, oth[s] / 10]), stay[s], ['own/10 stay', 'oth/10 stay'])
dP = np.concatenate(G['dP']); dy = np.concatenate(G['dy']); dm = np.concatenate(G['dmem']); dn = np.concatenate(G['dn'])
print('group: d total contribution per round r1-3 and d members at round 4, on d points and d number punished')
ols(np.column_stack([dP / 10, dn]), dy, ['P/10', 'n_pun'])
ols(np.column_stack([dP / 10, dn]), dm, ['P/10', 'n_pun'])
