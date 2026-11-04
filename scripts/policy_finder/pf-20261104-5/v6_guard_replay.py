"""v6: rough block-0 replay of the share guard on v5's sweep games (part 1). For a margin m,
cut = min(pblk, relu(16*(m - gap))) is the block-0 punishment the guard removes (n=4, ignoring
the lift lost and the rounds' order); the new gap is gap + cut/16 + lift_back, and the
expected dn is read off the empirical dn(gap) curve. Printed for v5's near-peak points
and for all points."""
import numpy as np, pandas as pd
s = pd.read_parquet('scripts/policy_finder/pf-20261104-5/v6_race.parquet')
grid = np.linspace(-30, 30, 121)
mid = (grid[1:] + grid[:-1]) / 2
idx = np.clip(np.digitize(s.gap, grid) - 1, 0, len(mid) - 1)
curve = pd.Series(s.dn.values).groupby(idx).mean().reindex(range(len(mid))).interpolate().bfill().ffill().values
f = lambda g: np.interp(g, mid, curve)
s['k'] = 0.0
import json
sw = {p['name']: p['params'] for p in json.load(open('plots/simulation/policy_finder/pf-20261104-5-v5_sweep/sweep.json'))['points']}
P = pd.DataFrame([sw[r] for r in s.run]); s = pd.concat([s, P.add_prefix('x_')], axis=1)
near = (s.x_k.between(4, 9)) & (s.x_thr.between(9.3, 11.3)) & (s.x_w.between(0.25, 0.75))
for lab, q in [('near peak', s[near]), ('all', s)]:
    print(lab, len(q), 'mean pblk', q.pblk.mean().round(1), 'mean gap', q.gap.mean().round(2), 'dn', q.dn.mean().round(2), 'fitted', f(q.gap).mean().round(2))
    for m in [-12, -8, -4, -2, 0, 2, 4]:
        cut = np.minimum(q.pblk, np.maximum(16 * (m - q.gap), 0))
        # the cut also forgoes part of the within-block lift: ~0.045 share per point (v5 round-0 numbers), so net 1/16-0.045
        g_new = q.gap + cut * (1 / 16 - 0.045)
        g_new_nolift = q.gap + cut / 16
        print(f'  m={m:+d}: games cut {np.mean(cut > 0):.2f}, points cut {cut.mean():.1f} of {q.pblk.mean():.1f}, '
              f'dn gain {(f(g_new) - f(q.gap)).mean():+.3f} (no lift loss {(f(g_new_nolift) - f(q.gap)).mean():+.3f})')
