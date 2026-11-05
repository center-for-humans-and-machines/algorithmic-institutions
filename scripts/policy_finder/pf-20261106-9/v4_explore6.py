# v4: in v3's sweep, a member's chance of leaving us at a decision (after rounds 3, 7, 11) by how
# much we punished it in that block, per opponent; and the members' contribution at the block's
# last round of those who leave vs stay.
import numpy as np
d = np.load('/tmp/v3sweep.npz')
c, p, g, ours, opp = d['c'].astype(np.float32), d['p'].astype(np.float32), d['g'], d['ours'], d['opp']
mine = (g == ours[:, None, None])
pu = p * mine
bins = [0, 1, 6, 13, 21, 31, 100]
for o in [1, 2, 3, 4]:
    m = opp == o
    print('== opponent', o)
    for dr in [3, 7, 11]:
        b0 = dr - 3
        inblk = mine[m][:, :, dr]
        pb = pu[m][:, :, b0:dr + 1].sum(-1)
        leave = inblk & ~mine[m][:, :, dr + 1]
        row = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            s = inblk & (pb >= lo) & (pb < hi)
            row.append(f'[{lo},{hi}) {leave[s].mean():.2f} (n={s.mean():.2f})')
        cl = c[m][:, :, dr]
        print(f' dec {dr}: ', ' '.join(row), f'| c leavers {cl[leave].mean():.1f} stayers {cl[inblk & ~leave].mean():.1f}')
