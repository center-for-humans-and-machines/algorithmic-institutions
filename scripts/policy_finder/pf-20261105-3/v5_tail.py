"""v5: middle/tail parametrisation dose = pmax (1 - c/cref)^gamma, gamma = log(d10/pmax)/log(1 - 10/cref),
no dose at c = 20; shock < horizon, free riders c < lthr until lstop. 4 seeds x 3000 games."""
import sys, math, warnings, numpy as np, torch as th
warnings.filterwarnings('ignore')
sys.path.insert(0, '/opt/algorithmic-institutions/scripts/policy_finder/pf-20261105-3')
from v5_levers import simulate
th.set_num_threads(2)
def rule(pmax=30, d10=6.5, cref=20, horizon=7, lthr=4, lstop=15):
    g = math.log(d10 / pmax) / math.log(1 - 10 / cref)
    def f(c, v, p, t):
        cn = c[..., -1]
        d = (pmax * (1 - cn / cref).clamp(min=0) ** g).clamp(max=30) * (cn < 20).float()
        if t < horizon: return d
        if t < lstop: return d * (cn < lthr).float()
        return th.zeros_like(cn)
    return f
def doses(**kw):
    q = dict(pmax=30, d10=6.5, cref=20); q.update({k: v for k, v in kw.items() if k in q})
    g = math.log(q['d10'] / q['pmax']) / math.log(1 - 10 / q['cref'])
    return [int(min(30, q['pmax'] * max(0, 1 - c / q['cref']) ** g)) for c in range(0, 20, 2)] + [round(g, 2)]
def sc(**kw):
    return np.array([simulate(rule(**kw), n=3000, seed=s)[0] for s in (801, 802, 803, 804)])
if __name__ == '__main__':
    b = sc(); print('base', b.mean().round(2), doses(), flush=True)
    for kw in [dict(cref=22), dict(cref=24), dict(cref=28), dict(cref=36), dict(cref=60),
               dict(d10=4.5), dict(d10=8.5), dict(d10=11), dict(d10=8.5, cref=28), dict(d10=4.5, cref=28)]:
        d = sc(**kw) - b
        print(kw, f'{d.mean():+.2f}', np.round(d, 1), doses(**kw), flush=True)
