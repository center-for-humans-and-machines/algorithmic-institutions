# v6: newcomers in v5's sweep. Players who join our group at round 4 (in the other group in round 3):
# their contribution at arrival relative to the incumbents, punishment received in rounds 4-5, and
# their contribution in rounds 6-7 and 8-11 if they stay; per opponent, and the same for incumbents.
import numpy as np
d = np.load('/tmp/pf-20261106-9-v5_sweep.npz')
c, p, g, ours, opp = d['c'].astype(float), d['p'].astype(float), d['g'], d['ours'], d['opp']
mine = g == ours[:, None, None]
j = mine[:, :, 4] & ~mine[:, :, 3]
inc = mine[:, :, 4] & mine[:, :, 3]
ninc = inc.sum(1, keepdims=True)
incmean4 = (c[:, :, 4] * inc).sum(1, keepdims=True) / np.maximum(ninc, 1)
incmean3 = (c[:, :, 3] * inc).sum(1, keepdims=True) / np.maximum(ninc, 1)
st = mine[:, :, 4:12].all(-1)
P45 = p[:, :, 4] + p[:, :, 5]
for o in range(1, 5):
    m = (opp == o)[:, None]
    J, I = j & m, inc & m
    print(f'opp {o}: joiners/game {J.sum()/m.sum():.2f}; joiner c3 (there) {c[:, :, 3][J].mean():.2f} c4 {c[:, :, 4][J].mean():.2f} vs incumbents c3 {c[:, :, 3][I].mean():.2f} c4 {c[:, :, 4][I].mean():.2f}; '
          f'joiner p4+p5 {P45[J].mean():.1f} incumbents {P45[I].mean():.1f}; joiner c8-11 {c[:, :, 8:12].mean(-1)[J & st].mean():.2f} inc {c[:, :, 8:12].mean(-1)[I & st].mean():.2f}')
    # dose response for joiners below incumbent mean
    sh = incmean4 - c[:, :, 4]
    for lo, hi in [(-20, 0), (0, 3), (3, 20)]:
        for plo, phi in [(0, 0.5), (0.5, 15), (15, 61)]:
            mm = J & (sh >= lo) & (sh < hi) & (P45 >= plo) & (P45 < phi)
            if mm.sum() < 100: continue
            print(f'   short [{lo},{hi}) p45 [{plo},{phi}) n {mm.sum():6d} c4 {c[:, :, 4][mm].mean():5.2f} stay8 {mine[:, :, 8][mm].mean():.3f} c8-11 {c[:, :, 8:12].mean(-1)[mm & st].mean():5.2f}')
