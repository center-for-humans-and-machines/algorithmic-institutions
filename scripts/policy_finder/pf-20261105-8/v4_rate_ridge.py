"""v4: is curvature/standard (rate) the quantity v2's and v3's sweeps score on? Mean score by
rate bin, and a quadratic fit on (rate, standard) vs (curvature, standard)."""
import json, numpy as np
for v in ['pf-20261105-8-v2', 'pf-20261105-8-v3']:
    s = json.load(open(f'plots/simulation/policy_finder/{v}_sweep/sweep.json'))['points']
    S = np.array([p['params']['standard'] for p in s]); K = np.array([p['params']['curvature'] for p in s])
    y = np.array([p['pool'] for p in s]); r = K / S
    for lo, hi in [(0, .07), (.07, .09), (.09, .11), (.11, .13), (.13, .15), (.15, .18), (.18, 1)]:
        m = (r >= lo) & (r < hi)
        if m.sum(): print(v, f'rate {lo:.2f}-{hi:.2f} n {m.sum():3d} mean {y[m].mean():.1f} top {y[m].max():.1f}')
    def r2(a, b):
        D = np.column_stack([np.ones_like(a), a, b, a * a, b * b, a * b]); be, *_ = np.linalg.lstsq(D, y, rcond=None)
        return 1 - (y - D @ be).var() / y.var()
    print(v, 'R2 quad (rate,S)', round(r2(r, S), 3), ' (curv,S)', round(r2(K, S), 3), ' rate alone', round(r2(r, 0 * S), 3))
