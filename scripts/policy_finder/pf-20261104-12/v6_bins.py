"""Effect of the dials on the pool within bins of a pre-treatment state: per bin,
OLS of the episode pool (rounds 4-23 for state at round 3; whole game for round-0
state) on the standardised dials (log slope, log decay) across design points.
usage: v6_bins.py v4 v5"""
import numpy as np, sys
Ys, Ss, Ps, S0 = [], [], [], []
for tag in sys.argv[1:]:
    z = np.load(f'/tmp/{tag}.npz')
    c, p, g = z['c'].astype(np.float32), z['p'].astype(np.float32), z['g']
    par = z['params'].copy(); par[:, 2] = np.log(par[:, 2]); par[:, 3] = np.log(par[:, 3])
    mine = (g == 0)
    pr = ((1.6 * c - p) * mine).sum(2)
    own = lambda t: (c[..., t] * mine[..., t]).sum(-1) / np.maximum(mine[..., t].sum(-1), 1)
    Ys.append(np.stack([pr[..., 4:].mean(-1), pr.mean(-1)], -1)); Ss.append(own(3)); S0.append(own(0))
    Ps.append(np.broadcast_to(par[:, None, :], par.shape[:1] + c.shape[1:2] + par.shape[1:]))
Y = np.concatenate(Ys).reshape(-1, 2); s3 = np.concatenate(Ss).ravel(); s0 = np.concatenate(S0).ravel()
Pm = np.concatenate(Ps).reshape(-1, 6); Pm = (Pm - Pm.mean(0)) / Pm.std(0)
names = ['thr', 'peer', 'lslope', 'ldecay', 'follow', 'size']
for lab, s, y in (('own mean at round 3 -> pool 4-23', s3, Y[:, 0]), ('own mean at round 0 -> pool 0-23', s0, Y[:, 1])):
    print('---', lab)
    qs = np.quantile(s, [0, .2, .4, .6, .8, 1])
    for a, b in zip(qs[:-1], qs[1:]):
        m = (s >= a) & (s <= b)
        A = np.c_[np.ones(m.sum()), Pm[m], Pm[m] ** 2]
        bb, *_ = np.linalg.lstsq(A, y[m], rcond=None)
        print(f'  [{a:5.1f},{b:5.1f}] n={m.sum():6d} pool {y[m].mean():6.1f} ' + ' '.join(f'{nm} {bb[1+i]:+5.2f}' for i, nm in enumerate(names)))
# where the quadratic in each dial peaks, per bin (in raw units)
print('--- peak of the per-bin quadratic (others at the design centre)')
raw = np.concatenate(Ps).reshape(-1, 6); mu, sd = raw.mean(0), raw.std(0)
for lab, s, y in (('own3 -> pool 4-23', s3, Y[:, 0]), ('own0 -> pool 0-23', s0, Y[:, 1])):
    qs = np.quantile(s, [0, .2, .4, .6, .8, 1])
    for a, b in zip(qs[:-1], qs[1:]):
        m = (s >= a) & (s <= b)
        A = np.c_[np.ones(m.sum()), Pm[m], Pm[m] ** 2]
        bb, *_ = np.linalg.lstsq(A, y[m], rcond=None)
        out = []
        for i, nm in enumerate(names):
            b1, b2 = bb[1 + i], bb[7 + i]
            xs = np.clip(-b1 / (2 * b2), -1.8, 1.8) if b2 < 0 else np.sign(b1) * 1.8
            v = mu[i] + xs * sd[i]
            v = np.exp(v) if nm.startswith('l') else v
            out.append(f'{nm} {v:6.2f}')
        print(f'  {lab} [{a:5.1f},{b:5.1f}] ' + ' '.join(out))
