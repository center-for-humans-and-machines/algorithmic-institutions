"""v5: the near-full nudge (a minimum dose for 16-19 contributors) on the thr-free v4 form,
4 seeds x 3000 games with common random numbers."""
import sys, warnings, numpy as np, torch as th
warnings.filterwarnings('ignore')
sys.path.insert(0, '/opt/algorithmic-institutions/scripts/policy_finder/pf-20261105-3')
from v5_levers import simulate
th.set_num_threads(2)
def rule(pmax=30, gamma=2.0, horizon=7, lthr=4, lstop=15, floor=0, flo=0, fstop=7):
    def f(c, v, p, t):
        cn = c[..., -1]
        d = (pmax * (1 - cn / 20).clamp(min=0) ** gamma).clamp(max=30) * (cn < 20).float()
        if t < fstop:
            d = th.maximum(d, floor * ((cn >= flo) & (cn < 20)).float())
        if t < horizon: return d
        if t < lstop: return d * (cn < lthr).float()
        return th.zeros_like(cn)
    return f
def sc(**kw):
    return np.array([simulate(rule(**kw), n=3000, seed=s)[0] for s in (701, 702, 703, 704)])
b = sc(); print('base', b.mean().round(2), flush=True)
for kw in [dict(floor=1, fstop=7), dict(floor=2, fstop=7), dict(floor=3, fstop=7), dict(floor=2, fstop=4),
           dict(floor=3, fstop=4), dict(floor=4, fstop=4), dict(floor=2, flo=16, fstop=7), dict(floor=3, flo=17, fstop=5),
           dict(floor=2, fstop=10)]:
    d = sc(**kw) - b
    print(kw, f'{d.mean():+.2f}', np.round(d, 1), flush=True)
