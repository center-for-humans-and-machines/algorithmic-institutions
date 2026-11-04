"""Pooled sweeps (rows from v5_abs_response.py): at-mean members (c / others' mean in [0.9, 1.15]),
by absolute c and punishment: change in c 3 rounds later and the share still in my group after the
next decision. Plus OLS of the change on p bins with c, o controls, by c/o class."""
import numpy as np
R = np.load('/tmp/pf/abs_rows.npy'); t, ci, oi, pi, c2, g2, stay = R.T
r = ci / np.maximum(oi, 0.5)
pb = [(0, 1), (1, 6), (6, 13), (13, 22), (22, 31)]
for lo, hi in [(0.9, 1.15)]:
    for clo, chi in [(4, 8), (8, 12), (12, 16), (16, 19)]:
        line = f'r {lo}-{hi} c {clo}-{chi}:'
        for plo, phi in pb:
            s = (r >= lo) & (r < hi) & (ci >= clo) & (ci < chi) & (pi >= plo) & (pi < phi) & (t >= 1)
            s2 = s & (g2 == 1)
            if s2.sum() < 300: line += '   --'; continue
            line += f' p{plo}-{phi}: d{(c2[s2]-ci[s2]).mean():+5.2f} st{stay[s].mean():.2f} n{s2.sum()//1000}k'
        print(line)
# OLS by ratio class
print('OLS: d(c,t+3) on p-bin dummies + c + o + round dummies, rounds>=1, still in group')
for lo, hi in [(0, 0.5), (0.5, 0.8), (0.8, 0.95), (0.95, 1.1), (1.1, 1.5)]:
    s = (r >= lo) & (r < hi) & (g2 == 1) & (t >= 1)
    X = [np.ones(s.sum()), ci[s], oi[s]] + [(t[s] == tt) * 1.0 for tt in [1, 2, 4, 5, 6, 8, 9]]
    names = []
    for plo, phi in pb[1:]:
        X.append(((pi[s] >= plo) & (pi[s] < phi)) * 1.0); names.append(f'p{plo}-{phi}')
    X = np.column_stack(X); y = c2[s] - ci[s]
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    # stay
    s3 = (r >= lo) & (r < hi) & (t >= 1)
    X3 = [np.ones(s3.sum()), ci[s3], oi[s3]] + [(t[s3] == tt) * 1.0 for tt in [1, 2, 4, 5, 6, 8, 9]]
    for plo, phi in pb[1:]:
        X3.append(((pi[s3] >= plo) & (pi[s3] < phi)) * 1.0)
    b3, *_ = np.linalg.lstsq(np.column_stack(X3), stay[s3], rcond=None)
    print(f'r {lo}-{hi} n{s.sum()//1000}k', ' '.join(f'{n}: {bb:+.2f}/{bs:+.3f}' for n, bb, bs in zip(names, b[-4:], b3[-4:])))
