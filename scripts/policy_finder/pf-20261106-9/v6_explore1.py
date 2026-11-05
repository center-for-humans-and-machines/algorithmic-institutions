# v5: v4 final (gp point) per opponent and round: our members, our mean contribution, our punishment
# per member, our pool; the other group's members, contribution, punishment per member.
import numpy as np
d = np.load('/tmp/pf-20261106-9-v5_final.npz')
import sys
PT = int(sys.argv[1]) if len(sys.argv) > 1 else -1  # -1: the gp point
sel = d['pt'] == PT
c, p, v, g, ours, opp = [d[k][sel] for k in ['c', 'p', 'v', 'g', 'ours', 'opp']]
c = c.astype(float); p = p.astype(float)
mine = g == ours[:, None, None]
oth = ~mine
for o in range(1, 5):
    m = opp == o
    M, O, cc, pp = mine[m], oth[m], c[m], p[m]
    nM, nO = M.sum(1), O.sum(1)
    pool = (1.6 * (cc * M).sum(1) - (pp * M).sum(1))
    print(f'== opponent {o}: pool {pool.mean():.1f}')
    print('rnd  nUs  cUs  pUs  pool | nOth cOth pOth  poolOth')
    for t in range(24):
        f = lambda x, mask: (x[:, :, t] * mask[:, :, t]).sum() / max(mask[:, :, t].sum(), 1)
        poolO = (1.6 * (cc * O).sum(1) - (pp * O).sum(1))[:, t].mean()
        print(f'{t:3d} {nM[:, t].mean():4.2f} {f(cc, M):5.2f} {f(pp, M):5.2f} {pool[:, t].mean():5.1f} | {nO[:, t].mean():4.2f} {f(cc, O):5.2f} {f(pp, O):5.2f} {poolO:5.1f}')
