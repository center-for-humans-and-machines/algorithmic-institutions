# v4: shape of the round-0 response to our punishment, by shortfall below the group mean (v3 sweep).
# Outcome: c[1] and c[3] (members still with us), as a difference from the unpunished members of
# the same cell (own c[0] in bins of 2, shortfall bin), averaged over cells; by punishment bin.
import numpy as np
d = np.load('/tmp/v3sweep.npz')
c, p, g, ours = d['c'].astype(np.float32), d['p'].astype(np.float32), d['g'], d['ours']
mine = (g == ours[:, None, None])
n0 = mine[:, :, 0].sum(1)
gm = (c[:, :, 0] * mine[:, :, 0]).sum(1) / n0
short = gm[:, None] - c[:, :, 0]
sel = mine[:, :, 0] & mine[:, :, 3]
pb = [0, 1, 4, 8, 13, 19, 25, 31]
for sl, sh in [(0, 1.5), (1.5, 3), (3, 5), (5, 8), (8, 20)]:
    for h in [1, 3]:
        out = []
        for lo, hi in zip(pb[:-1], pb[1:]):
            diffs, ws = [], []
            for cb in range(0, 20, 2):
                cell = sel & (short >= sl) & (short < sh) & (c[:, :, 0] >= cb) & (c[:, :, 0] < cb + 2)
                base = cell & (p[:, :, 0] == 0)
                tr = cell & (p[:, :, 0] >= lo) & (p[:, :, 0] < hi)
                if base.sum() > 30 and tr.sum() > 30 and lo > 0:
                    diffs.append(c[:, :, h][tr].mean() - c[:, :, h][base].mean()); ws.append(tr.sum())
            if ws:
                out.append(f'[{lo},{hi}) {np.average(diffs, weights=ws):+.2f} ({sum(ws)//1000}k)')
        print(f'short [{sl},{sh}) h={h}: ' + ' '.join(out))
