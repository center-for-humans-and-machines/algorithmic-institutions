# v6: per-round my-group members, contribution per valid member, punishment per member, pool
# for each final point of v5 (and v4's gp), per opponent
import numpy as np, sys
f = sys.argv[1]
d = np.load(f)
C, P, G, V = d['C'].astype(float), d['P'].astype(float), d['G'], d['V'].astype(float)
seat = d['seat']; opp = d['opp']; pt = d['point']
mine = (G == seat[:, None, None]).astype(float)
for p in np.unique(pt):
    for o in [1, 2, 3, 4]:
        s = (pt == p) & (opp == o)
        m = mine[s]; th_ = 1 - m
        mem = m.sum(1).mean(0)
        cpm = (C[s] * m * V[s]).sum(1).sum(0) / np.maximum((m * V[s]).sum(1).sum(0), 1)
        ppm = (P[s] * m).sum(1).sum(0) / np.maximum(m.sum(1).sum(0), 1)
        pool = (1.6 * (C[s] * m).sum(1) - (P[s] * m).sum(1)).mean(0)
        opool = (1.6 * (C[s] * th_).sum(1) - (P[s] * th_).sum(1)).mean(0)
        ocpm = (C[s] * th_ * V[s]).sum(1).sum(0) / np.maximum((th_ * V[s]).sum(1).sum(0), 1)
        oppm = (P[s] * th_).sum(1).sum(0) / np.maximum(th_.sum(1).sum(0), 1)
        print(f'pt{p} opp{o} pool {pool.mean():.1f} vs {opool.mean():.1f}')
        for name, a in [('mem', mem), ('cpm', cpm), ('ppm', ppm), ('ocpm', ocpm), ('oppm', oppm), ('pool', pool)]:
            print('  ', name.ljust(5), ' '.join(f'{x:5.1f}' for x in a))
