"""v4: compare candidate rule shapes on the surrogate, 3 seeds x 4000 games."""
import sys, warnings, numpy as np, torch as th
warnings.filterwarnings('ignore')
sys.path.insert(0, '/opt/algorithmic-institutions/scripts/policy_finder/pf-20261105-3')
from v4_sim import simulate
from v4_levers import shock, f_base, f_late
th.set_num_threads(2)

def late2(thr, pmax, gamma, horizon, lthr, lstop, ldose=30):
    def f(c, v, p, t):
        cn = c[..., -1]
        if t < horizon: return shock(cn, thr, pmax, gamma)
        if t < lstop: return ldose * (cn < lthr).float()
        return th.zeros_like(cn)
    return f

def nudge(thr, pmax, gamma, horizon, k, lo, t1):
    def f(c, v, p, t):
        cn = c[..., -1]
        d = shock(cn, thr, pmax, gamma) * float(t < horizon)
        return th.maximum(d, k * ((cn >= lo) & (cn < 20)).float() * float(t < t1))
    return f

C = {
 'base 21/30/2.3/10': lambda c,v,p,t: f_base(c,v,p,t,21,30,2.3,10),
 'base 22.4/32/2.5/10': lambda c,v,p,t: f_base(c,v,p,t,22.4,32,2.5,10),
 'v2 final 20/30/2.09/9': lambda c,v,p,t: f_base(c,v,p,t,20,30,2.09,9),
 'late 14/29.4/1/6 lthr1.14 lstop16': late2(14, 29.4, 1.0, 6, 1.14, 16),
 'base21+late lthr3 lstop16 h9': late2(21, 30, 2.3, 9, 3, 16),
 'base21+late lthr2 lstop14 h10': late2(21, 30, 2.3, 10, 2, 14),
 'base21+late lthr5 lstop15 h8 dose20': late2(21, 30, 2.3, 8, 5, 15, 20),
 'base21+nudge4 c18-19 t<5': nudge(21, 30, 2.3, 10, 4, 18, 5),
 'base21+nudge2 c16-19 t<5': nudge(21, 30, 2.3, 10, 2, 16, 5),
}
for k, f in C.items():
    s = [simulate(f, n=4000, seed=s)[0] for s in (301, 302, 303)]
    print(f'{k:40s} {np.mean(s):.2f}  ({" ".join(f"{x:.1f}" for x in s)})', flush=True)
