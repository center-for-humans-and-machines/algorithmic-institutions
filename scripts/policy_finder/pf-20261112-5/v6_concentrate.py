"""v6: concentrated vs spread shocks. Pairs of aligned games identical in all contributions and punishments before round
R and in the contributions of round R, whose round-R punishments differ. Group-level regressions of
  dy  = d (my group's total contribution, mean over rounds R+1..R+3),
  dm  = d (my members at round R+4, after the next decision),
  dl  = d (lead per round over rounds R..23), controlling for d (my points dealt after R)/10,
on dP (total points at R)/10 and dn (number of members punished at R). Usage: v6_concentrate.py R"""
import numpy as np, itertools, sys
R = int(sys.argv[1]) if len(sys.argv) > 1 else 0
V = ['v1', 'v2', 'v3', 'v4', 'v5']
A = {v: np.load(f'/tmp/pf/al_{v}.npz') for v in V}
cols = {k: [] for k in ['dP', 'dn', 'dtop', 'dy', 'dm', 'dl', 'dlater', 'n', 'dyo']}
for a, b in itertools.combinations(V, 2):
    za, zb = A[a], A[b]
    same = (za['c'][:, :, :R + 1] == zb['c'][:, :, :R + 1]).all((1, 2)) & (za['p'][:, :, :R] == zb['p'][:, :, :R]).all((1, 2))
    m = za['mine'][:, :, R].astype(bool)
    pa, pb = za['p'][:, :, R] * m, zb['p'][:, :, R] * m
    diff = (pa != pb).any(1)
    s = same & diff
    if s.sum() == 0: continue
    pa, pb, m = pa[s].astype(float), pb[s].astype(float), m[s]
    def tot_c(z):
        mm = z['mine'][s][:, :, R + 1:R + 4].astype(float)
        return (z['c'][s][:, :, R + 1:R + 4] * mm).sum((1, 2)) / 3
    def oth_c(z):
        mm = 1 - z['mine'][s][:, :, R + 1:R + 4].astype(float)
        return (z['c'][s][:, :, R + 1:R + 4] * mm).sum((1, 2)) / 3
    def lead(z):
        c = z['c'][s][:, :, R:].astype(float); p = z['p'][s][:, :, R:].astype(float); mm = z['mine'][s][:, :, R:].astype(bool)
        pool = 1.6 * c * z['v'][s][:, :, R:] - p
        return ((pool * mm).sum(1) - (pool * ~mm).sum(1)).mean(1)
    def later(z):
        return (z['p'][s][:, :, R + 1:] * z['mine'][s][:, :, R + 1:]).sum((1, 2)).astype(float)
    cols['dP'].append((pa - pb).sum(1) / 10); cols['dn'].append((pa > 0).sum(1) - (pb > 0).sum(1.0 and 1))
    cols['dtop'].append((pa.max(1) - pb.max(1)) / 10)
    cols['dy'].append(tot_c(za) - tot_c(zb)); cols['dyo'].append(oth_c(za) - oth_c(zb))
    ri = min(R + 4, 23)
    cols['dm'].append(za['mine'][s][:, :, ri].sum(1).astype(float) - zb['mine'][s][:, :, ri].sum(1))
    cols['dl'].append(lead(za) - lead(zb)); cols['dlater'].append((later(za) - later(zb)) / 10)
    cols['n'].append(m.sum(1))
D = {k: np.concatenate(v).astype(float) for k, v in cols.items()}
print('R', R, 'pairs', len(D['dP']))
def ols(X, y, names):
    X = np.column_stack([X, np.ones(len(y))]); b, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ b; s2 = r @ r / (len(y) - X.shape[1]); se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
    print('  ' + ' '.join(f'{n}:{bb:+.3f}({ss:.3f})' for n, bb, ss in zip(names + ['1'], b, se)))
X = np.column_stack([D['dP'], D['dn']])
for k in ['dy', 'dyo', 'dm']:
    print(k); ols(X, D[k], ['P/10', 'n'])
print('dl (ctrl later points)'); ols(np.column_stack([X, D['dlater']]), D['dl'], ['P/10', 'n', 'later/10'])
print('with top dose: dy, dm, dl')
X2 = np.column_stack([D['dP'], D['dn'], D['dtop']])
ols(X2, D['dy'], ['P/10', 'n', 'top/10']); ols(X2, D['dm'], ['P/10', 'n', 'top/10'])
ols(np.column_stack([X2, D['dlater']]), D['dl'], ['P/10', 'n', 'top/10', 'later/10'])
