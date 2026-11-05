# v6: dose-response of round-0 punishment in v5's sweep. For our 4 original members: shortfall
# below the round-0 group mean, punishment p0, then contribution in round 2 (all stay through 3),
# whether they leave at the round-3 decision, and mean contribution in rounds 8-15 if they stayed.
import numpy as np
d = np.load('/tmp/pf-20261106-9-v5_sweep.npz')
c, p, g, ours = d['c'].astype(float), d['p'].astype(float), d['g'], d['ours']
mine = g == ours[:, None, None]
M0 = mine[:, :, 0]
mean0 = (c[:, :, 0] * M0).sum(1, keepdims=True) / 4
short = mean0 - c[:, :, 0]
p0, p1 = p[:, :, 0], p[:, :, 1]
c2 = c[:, :, 2]
stay = mine[:, :, 4]
late = c[:, :, 8:16].mean(-1)
stayall = mine[:, :, 4:16].all(-1)
sel = M0
print('shortfall bin x p0 bin: n, c0, c2, stay4, late(if stayed whole)')
for lo, hi in [(-20, -2), (-2, 0.01), (0.01, 2), (2, 4), (4, 7), (7, 20)]:
    for plo, phi in [(0, 0.5), (0.5, 6), (6, 15), (15, 25), (25, 31)]:
        m = sel & (short >= lo) & (short < hi) & (p0 >= plo) & (p0 < phi)
        if m.sum() < 200: continue
        ml = m & stayall
        print(f'short [{lo},{hi}) p0 [{plo},{phi}) n {m.sum():6d} c0 {c[:, :, 0][m].mean():5.2f} p1 {p1[m].mean():5.1f} c2 {c2[m].mean():5.2f} stay {stay[m].mean():.3f} late {late[ml].mean():5.2f}')
