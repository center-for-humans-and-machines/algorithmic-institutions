# v7: per opponent and round, from a final npz: my members, contribution per member, my punishment per member,
# the opponent's punishment per member, my pool; for the point named by index (-1 = gp)
import numpy as np, sys
d = np.load(sys.argv[1] if len(sys.argv) > 1 else '/tmp/v6final.npz')
pt = int(sys.argv[2]) if len(sys.argv) > 2 else -1
sel = d['point'] == pt
C, P, G, V = [d[k][sel].astype(float) for k in 'CPGV']
opp, seat = d['opp'][sel], d['seat'][sel]
mine = (G == seat[:, None, None])
for o in np.unique(opp):
    s = opp == o
    m = mine[s]; c = C[s]; p = P[s]
    nm = m.sum(1); no = (~m).sum(1)
    cm = (c * m).sum(1) / np.maximum(nm, 1); pm = (p * m).sum(1) / np.maximum(nm, 1)
    po = (p * ~m).sum(1) / np.maximum(no, 1)
    pool = 1.6 * (c * m).sum(1) - (p * m).sum(1)
    pool_o = 1.6 * (c * ~m).sum(1) - (p * ~m).sum(1)
    print('opp', o, 'pool %.1f oppgroup pool %.1f' % (pool.mean(), pool_o.mean()))
    for name, a in [('members', nm), ('c/member', cm), ('my p/mem', pm), ('opp p/mem', po), ('pool', pool), ('opp pool', pool_o)]:
        print('  %-9s' % name, ' '.join('%5.1f' % v for v in a.mean(0)))
