# v3: shape of the first-punishment response. Our members first punished by us at round t (none
# before), vs same-cell unpunished members: change c[t+3]-c[t] by amount bin, within cells of
# (own c[t], group-mean bin) to remove selection. Rounds 0,1 (opening) and 4-13 (later arrivals).
import numpy as np
d = np.load('/tmp/v2sweep.npz')
c, p, g, ours = d['c'].astype(np.float32), d['p'].astype(np.float32), d['g'], d['ours']
mine = (g == ours[:, None, None])
pu = p * mine
cnt = np.cumsum(pu > 0, -1) - (pu > 0)
bins = [0, 1, 4, 8, 13, 19, 25, 31]
for rounds, h in [([0], 3), ([0], 8), ([1], 2), ([4, 5, 8, 9, 12, 13], 2)]:
    rows = {}
    for t in rounds:
        stay = mine[:, :, t] & mine[:, :, t + h] & mine[:, :, min(t + h, (t // 4) * 4 + 3)]
        stay &= cnt[:, :, t] == 0
        gm = (c[:, :, t] * mine[:, :, t]).sum(1) / np.maximum(mine[:, :, t].sum(1), 1)
        gmb = np.round(np.broadcast_to(gm[:, None], c.shape[:2]) / 2)
        dy = (c[:, :, t + h] - c[:, :, t])[stay]
        own = c[:, :, t][stay]; gb = gmb[stay]; pp = pu[:, :, t][stay]
        cell = (own * 100 + gb).astype(int)
        # cell baseline: mean dy among unpunished in the cell
        uc, inv = np.unique(cell, return_inverse=True)
        base_s = np.bincount(inv, weights=dy * (pp == 0), minlength=len(uc))
        base_n = np.bincount(inv, weights=(pp == 0).astype(float), minlength=len(uc))
        ok = base_n[inv] >= 30
        resid = dy - base_s[inv] / np.maximum(base_n[inv], 1)
        for lo, hi in zip(bins[:-1], bins[1:]):
            for grp, m in [('low c<=6', own <= 6), ('mid 7-12', (own > 6) & (own <= 12)), ('high>12', own > 12)]:
                s = ok & (pp >= lo) & (pp < hi) & m
                rows.setdefault((lo, hi, grp), []).append((resid[s].sum(), s.sum()))
    print(f'=== rounds {rounds}, horizon {h}: extra change vs unpunished in same cell')
    for (lo, hi, grp), v in rows.items():
        s = sum(a for a, b in v); n = sum(b for a, b in v)
        if n > 200:
            print(f'  p {lo:2d}-{hi - 1:2d} {grp:9s} n={n:7d} extra={s / n:+.2f}  per point={s / n / max((lo + hi - 1) / 2, 0.5):+.3f}')
