# v3: in v2's final (gp point), our members' contribution by whether/when they were punished by us,
# and newcomers vs incumbents, per opponent.
import numpy as np
d = np.load('/tmp/v2final.npz')
c, p, v, g, ours, pt, opp = [d[k] for k in ['c', 'p', 'v', 'g', 'ours', 'pt', 'opp']]
sel = pt == -1
c, p, v, g, ours, opp = c[sel].astype(float), p[sel].astype(float), v[sel], g[sel], ours[sel], opp[sel]
mine = g == ours[:, None, None]          # [n,8,24]
pun_by_us = p * mine
# exposure: cumulative punishment received from us before round t
cum = np.cumsum(pun_by_us, -1) - pun_by_us
first_round_with_us = np.argmax(mine, -1)  # first round in our group
for o in range(1, 5):
    m = opp == o
    print('==== opponent', o)
    # newcomers at round 4,8,12: members in our group at round r who were not at r-1
    for r in [4, 8, 12, 16]:
        new = mine[m][:, :, r] & ~mine[m][:, :, r - 1]
        inc = mine[m][:, :, r] & mine[m][:, :, r - 1]
        cc = c[m]
        row = []
        for h in [0, 1, 2, 3]:
            row.append((cc[:, :, r + h][new].mean().round(2), cc[:, :, r + h][inc].mean().round(2)))
        print(f' round {r}: n_new/game {new.sum(1).mean():.2f}; c new vs incumbent at r..r+3', row,
              ' new prev c (other grp)', cc[:, :, r - 1][new].mean().round(2))
