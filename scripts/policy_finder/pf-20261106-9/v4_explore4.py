# v4: across v3's design points (per opponent), how the score relates to how much punishment the
# point dealt where: originals (in our group since round 0) in rounds 0-1, originals in rounds 4-7,
# newcomers (joined at the latest switch) in their block, everyone in rounds 8-23 otherwise.
# Point-level OLS of mean pool on these per-game punishment totals.
import numpy as np, json
d = np.load('/tmp/v3sweep.npz')
c, p, g, ours, pt, opp = d['c'].astype(np.float32), d['p'].astype(np.float32), d['g'], d['ours'], d['pt'], d['opp']
mine = (g == ours[:, None, None])
pu = p * mine
pool = (1.6 * (c * mine).sum(1) - pu.sum(1)).mean(1)          # per game
orig = np.cumprod(mine, -1).astype(bool)                       # with us since round 0
blk = np.arange(24) // 4
start = np.zeros_like(mine)
for t in range(24):
    if t % 4 == 0 and t > 0:
        start[:, :, t] = mine[:, :, t] & ~mine[:, :, t - 1]
    elif t > 0:
        start[:, :, t] = start[:, :, t - 1] & mine[:, :, t]
r = np.arange(24)
cats = {
    'orig r0-1': (pu * orig * (r < 2)).sum((1, 2)),
    'orig r4-7': (pu * orig * ((r >= 4) & (r < 8))).sum((1, 2)),
    'orig r8+': (pu * orig * (r >= 8)).sum((1, 2)),
    'new r4-7': (pu * start * ((r >= 4) & (r < 8))).sum((1, 2)),
    'new r8-11': (pu * start * ((r >= 8) & (r < 12))).sum((1, 2)),
    'new r12+': (pu * start * (r >= 12)).sum((1, 2)),
    'other': (pu * ~orig * ~start).sum((1, 2)),
}
names = list(cats)
for o in [1, 2, 3, 4, 0]:
    rows, ys = [], []
    for s in np.unique(pt):
        m = (pt == s) & ((opp == o) if o else True)
        rows.append([cats[k][m].mean() for k in names]); ys.append(pool[m].mean())
    X = np.array(rows); y = np.array(ys)
    b = np.linalg.lstsq(np.column_stack([np.ones(len(y)), X]), y, rcond=None)[0]
    print(f'opp {o or "all"}: ' + ' '.join(f'{k}={v:+.3f}(mean {X[:, i].mean():.0f})' for i, (k, v) in enumerate(zip(names, b[1:]))))
