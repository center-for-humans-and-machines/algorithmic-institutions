"""v5: per sweep point (all four versions), per-game punishment points dealt by my manager per round, split by the
target's contribution that round (<=5, 6-9, >=10), plus the per-point lead, own/other pool, members. Saves /tmp/pf/feats.npz.
Uses the compact npz files from compact.py (/tmp/pf/v1..v4.npz)."""
import numpy as np
out = {}
for i, v in enumerate(['v1', 'v2', 'v3', 'v4']):
    z = np.load(f'/tmp/pf/{v}.npz')
    c = z['c'].astype(np.float32); p = z['p'].astype(np.float32); m = z['mine'].astype(np.float32); val = z['v'].astype(np.float32)
    pt = z['point']; seat = z['seat']
    pool = 1.6 * c * val - p
    lead = ((pool * m).sum(1) - (pool * (1 - m)).sum(1)).mean(1)  # per game, mean over rounds
    own = (pool * m).sum(1).mean(1); oth = (pool * (1 - m)).sum(1).mean(1)
    mem = m.sum(1).mean(1)
    bins = [(c <= 5), (c > 5) & (c < 10), (c >= 10)]
    PB = np.stack([(p * m * b).sum(1) for b in bins], 1)  # [games, 3, 24]
    pts = np.unique(pt)
    F = np.zeros((len(pts), 3, 24)); L = np.zeros(len(pts)); O = np.zeros(len(pts)); T = np.zeros(len(pts)); M = np.zeros(len(pts)); SE = np.zeros(len(pts))
    for j, k in enumerate(pts):
        s = pt == k
        F[j] = PB[s].mean(0); L[j] = lead[s].mean(); SE[j] = lead[s].std() / np.sqrt(s.sum()); O[j] = own[s].mean(); T[j] = oth[s].mean(); M[j] = mem[s].mean()
    out[v] = dict(slot=pts, F=F, lead=L, se=SE, own=O, oth=T, mem=M, params=z['params'], param_idx=z['param_idx'])
    print(v, 'lead mean %.2f' % L.mean(), 'pts/game %.1f' % F.sum((1, 2)).mean())
    del c, p, m, val, pool, z
np.savez('/tmp/pf/feats.npz', **{f'{v}_{k}': a for v, d in out.items() for k, a in d.items()})
