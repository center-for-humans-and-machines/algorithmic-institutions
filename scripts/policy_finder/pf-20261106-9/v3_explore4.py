# v3: later punishments (rounds 4,8,12,16 = first round of a block, horizon 3 within the block):
# extra change in contribution vs unpunished members of the same cell (own c, group-mean bin),
# split by newcomer (joined this round) vs incumbent, and by prior punishment by us.
import numpy as np
d = np.load('/tmp/v2sweep.npz')
c, p, g, ours = d['c'].astype(np.float32), d['p'].astype(np.float32), d['g'], d['ours']
mine = (g == ours[:, None, None])
pu = p * mine
cnt = np.cumsum(pu > 0, -1) - (pu > 0)
bins = [0, 1, 6, 13, 21, 31]
for t0 in [4, 8, 12, 16]:
  for h in [3]:
    t = t0
    stay = mine[:, :, t] & mine[:, :, t + h]
    new = stay & ~mine[:, :, t - 1]
    gm = (c[:, :, t] * mine[:, :, t]).sum(1) / np.maximum(mine[:, :, t].sum(1), 1)
    gmb = np.round(np.broadcast_to(gm[:, None], c.shape[:2]) / 2)
    dy = (c[:, :, t + h] - c[:, :, t])
    own = c[:, :, t]; pp = pu[:, :, t]
    print(f'=== round {t}')
    for name, m in [('newcomer', new), ('incumbent never punished', stay & ~new & (cnt[:, :, t] == 0)),
                    ('incumbent punished before', stay & ~new & (cnt[:, :, t] > 0))]:
        cell = (own[m] * 100 + gmb[m]).astype(int)
        uc, inv = np.unique(cell, return_inverse=True)
        z = pp[m] == 0
        bs = np.bincount(inv, weights=dy[m] * z, minlength=len(uc)); bn = np.bincount(inv, weights=z.astype(float), minlength=len(uc))
        ok = bn[inv] >= 20
        resid = dy[m] - bs[inv] / np.maximum(bn[inv], 1)
        out = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            for gname, gm_ in [('c<=6', own[m] <= 6), ('c>6', own[m] > 6)]:
                s = ok & (pp[m] >= lo) & (pp[m] < hi) & gm_
                if s.sum() > 100 and lo > 0:
                    out.append(f'{lo}-{hi-1} {gname}: {resid[s].mean():+.2f} (n={s.sum()})')
        print(f'  {name:26s} n={m.sum():7d}', '; '.join(out))
