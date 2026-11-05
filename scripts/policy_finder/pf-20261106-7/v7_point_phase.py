# v7: per opponent, across sweep points, pool vs my punishment per member-round in phases (r0, r1-3, r4-11, r12-23)
import numpy as np, sys
d = np.load(sys.argv[1] if len(sys.argv) > 1 else '/tmp/v6sweep.npz')
C, P, G = [d[k].astype(np.float32) for k in 'CPG']
seat = d['seat']; mine = (G == seat[:, None, None])
nm = mine.sum(1)
pm = (P * mine).sum(1) / np.maximum(nm, 1)
pool = (1.6 * (C * mine).sum(1) - (P * mine).sum(1)).mean(1)
mem = nm.mean(1); cpm = ((C * mine).sum(1) / np.maximum(nm, 1))[:, 12:].mean(1)
phases = [(0, 1), (1, 4), (4, 12), (12, 24)]
F = np.stack([pm[:, a:b].mean(1) for a, b in phases], 1)
pts = np.unique(d['point'])
for o in np.unique(d['opp']):
    rows = []
    for q in pts:
        s = (d['point'] == q) & (d['opp'] == o)
        rows.append(np.r_[F[s].mean(0), pool[s].mean(), mem[s].mean(), cpm[s].mean()])
    R = np.array(rows)
    X = np.c_[np.ones(len(R)), R[:, :4], R[:, :4] ** 2]
    for j, nm_ in [(4, 'pool'), (5, 'members'), (6, 'c/mem 12+')]:
        b, *_ = np.linalg.lstsq(X, R[:, j], rcond=None)
        # optimum per phase (quadratic), others at mean
        opt = [-b[1 + i] / (2 * b[5 + i]) if b[5 + i] < 0 else np.nan for i in range(4)]
        print('opp', o, '%-10s' % nm_, 'lin', np.round(b[1:5], 3), 'quad', np.round(b[5:9], 4), 'opt', np.round(opt, 1) if j == 4 else '')
    print('   phase means', np.round(R[:, :4].mean(0), 2), 'ranges', np.round(R[:, :4].min(0), 1), np.round(R[:, :4].max(0), 1))
