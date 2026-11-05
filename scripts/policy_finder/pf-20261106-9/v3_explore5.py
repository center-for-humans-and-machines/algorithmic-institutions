# v3: do newcomers punished in their first block with us leave more at the next decision?
# v2 sweep; newcomers at rounds 4, 8, 12 who are still with us at the block's last round.
import numpy as np
d = np.load('/tmp/v2sweep.npz')
p, g, ours = d['p'].astype(np.float32), d['g'], d['ours']
mine = (g == ours[:, None, None])
pu = p * mine
for r in [4, 8, 12]:
    new = mine[:, :, r] & ~mine[:, :, r - 1] & mine[:, :, r + 3]
    blk = pu[:, :, r:r + 4].sum(-1)
    leave = ~mine[:, :, r + 4]
    out = []
    for lo, hi in [(0, 1), (1, 6), (6, 13), (13, 21), (21, 31), (31, 200)]:
        s = new & (blk >= lo) & (blk < hi)
        out.append(f'{lo}-{hi - 1}: {leave[s].mean():.2f} (n={s.sum()})')
    print(f'newcomers at {r}: leave next decision by block punishment', '; '.join(out))
