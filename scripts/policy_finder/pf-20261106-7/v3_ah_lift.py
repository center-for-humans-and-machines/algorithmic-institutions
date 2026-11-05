# v3: lift of later contributions over unpunished peers, by round of the punishment, from a sweep's arrays
# compares players at c<=9 punished p in round t with unpunished ones at the same c band and round; outcome c(t+4)-c(t)
# also: switching at the next decision for the same players
import numpy as np, sys
d = np.load(sys.argv[1] if len(sys.argv)>1 else '/tmp/v2sweep.npz')
rng = np.random.default_rng(3); idx = rng.choice(len(d['C']), 60000, replace=False)
C, P, G, V = [d[k][idx].astype(np.float32) for k in 'CPGV']
cum = np.concatenate([np.zeros_like(P[:, :, :1]), np.cumsum(P, -1)[:, :, :-1]], -1)
for t in [0, 1, 2, 4, 5, 6, 8, 9, 12, 13, 16]:
    nd = (t // 4) * 4 + 3  # decision round at end of this block
    out = []
    for lo, hi in [(0, 4), (5, 9)]:
        base = (V[:, :, t] > 0) & (C[:, :, t] >= lo) & (C[:, :, t] <= hi)
        stay = G[:, :, min(t+4, 23)] == G[:, :, t]
        y = C[:, :, min(t+4, 23)] - C[:, :, t]
        sw = (G[:, :, nd+1] != G[:, :, nd]) if nd < 20 else np.zeros_like(stay)
        m0 = base & (P[:, :, t] == 0)
        if m0.sum() < 200: continue
        res = [f'c{lo}-{hi} p0: dy {y[m0 & stay].mean():+.2f} sw {sw[m0].mean():.2f} (n{m0.sum()})']
        for a, b in [(1, 5), (6, 12), (13, 30)]:
            m = base & (P[:, :, t] >= a) & (P[:, :, t] <= b)
            if m.sum() > 200:
                res.append(f'p{a}-{b}: lift {y[m & stay].mean() - y[m0 & stay].mean():+.2f} sw {sw[m].mean():.2f}')
        out.append(' '.join(res))
    print('round', t, ' || '.join(out))
