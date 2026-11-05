"""v6: nonparametric check of f3: score by T bin within lineup bins of the lowest valid c0 (plateau B 1.3-2.6, S 6-10)."""
import numpy as np
z = np.load('/tmp/pf6/groups.npz'); R = z['R']; cols = list(z['cols'])
g = {k: R[:, i] for i, k in enumerate(cols)}
reg = (g['B'] >= 1.3) & (g['B'] <= 2.6) & (g['A'] <= 5) & (g['S'] >= 6) & (g['S'] <= 10) & (g['n0'] == 4)
for lo, hi in [(0, 0), (1, 4), (5, 8), (9, 11), (12, 20)]:
    m0 = reg & (g['s0'] >= lo) & (g['s0'] <= hi)
    out = []
    for a in np.arange(11, 17.5, 1.0):
        m = m0 & (g['T'] >= a) & (g['T'] < a + 1)
        if m.sum() > 300:
            out.append(f'T{a:.0f}: {g["score"][m].mean():6.1f}±{g["score"][m].std()/np.sqrt(m.sum()):.1f}')
    print(f's0 {lo}-{hi} ({m0.mean()/reg.mean():.0%})', ' '.join(out))
