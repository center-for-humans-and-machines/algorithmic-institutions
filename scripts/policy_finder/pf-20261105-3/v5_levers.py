"""v5: screen new levers against v4's best on the refit surrogate (3 seeds x n games,
common random numbers). python v5_levers.py <group> [n]"""
import sys, warnings, numpy as np, torch as th
warnings.filterwarnings('ignore')
sys.path.insert(0, '/opt/algorithmic-institutions/scripts/policy_finder/pf-20261105-3')
import v5_sim  # loads the v5 surrogate
from v4_sim import simulate
th.set_num_threads(2)


def shock(cn, thr, pmax, gamma):
    s = (thr - cn).clamp(min=0)
    return (pmax * (s / thr).clamp(0, 1) ** gamma).clamp(max=30) * (cn < 20).float()


def v4(thr, pmax, gamma, horizon, lthr, lstop):
    def f(c, v, p, t):
        cn = c[..., -1]; d = shock(cn, thr, pmax, gamma)
        if t < horizon: return d
        if t < lstop: return d * (cn < lthr).float()
        return th.zeros_like(cn)
    return f


def v4x(thr=24, pmax=30, gamma=2.9, horizon=7, lthr=4, lstop=15, nudge=0, nlo=16, nstop=5,
        regrp=None, restop=20, mid=None, midstop=15, tail_lthr=0, tail_stop=23, cfree=20, rel_drop=None):
    """v4 plus optional levers:
    nudge: nudge points for nlo <= c < 20 in rounds < nstop
    regrp: after the shock, if the others' mean c < regrp, the full shock dose again (until restop)
    mid: in the late phase, players with c < mid who dropped by >= 5 since last round get the shock dose
    tail_lthr/tail_stop: after lstop, c < tail_lthr still get the dose until tail_stop
    cfree: no punishment at c >= cfree
    rel_drop: in the late phase, c < lthr only counts if the player was >= rel_drop last round (a relapse)"""
    def f(c, v, p, t):
        cn = c[..., -1]; d = shock(cn, thr, pmax, gamma) * (cn < cfree).float()
        on = th.zeros_like(cn)
        if t < horizon:
            on = th.ones_like(cn)
        elif t < lstop:
            low = (cn < lthr).float()
            if rel_drop is not None:
                low = low * (c[..., -2] >= rel_drop).float()
            on = low
        elif t < tail_stop:
            on = (cn < tail_lthr).float()
        if t >= horizon and regrp is not None and t < restop:
            vv = v[..., -1]
            tot = (cn * vv).sum(1, keepdim=True); n = vv.sum(1, keepdim=True)
            om = (tot - cn * vv) / (n - vv).clamp(min=1)
            on = th.maximum(on, (om < regrp).float() * (cn < om).float())
        if t >= horizon and mid is not None and t < midstop:
            on = th.maximum(on, (cn < mid).float() * ((c[..., -2] - cn) >= 5).float())
        out = d * on
        if nudge and t < nstop:
            out = th.maximum(out, nudge * ((cn >= nlo) & (cn < 20)).float())
        return out
    return f


def reparam(pmax, gamma, cfree, horizon, lthr, lstop):
    def f(c, v, p, t):
        cn = c[..., -1]
        d = (pmax * (1 - cn / 20).clamp(min=0) ** gamma).clamp(max=30) * (cn < cfree).float()
        if t < horizon: return d
        if t < lstop: return d * (cn < lthr).float()
        return th.zeros_like(cn)
    return f


GP = dict(thr=24, pmax=30.08, gamma=2.93, horizon=7, lthr=4, lstop=15)
S187 = dict(thr=23, pmax=29.4, gamma=2.6, horizon=8, lthr=3, lstop=13)
GROUPS = {
 'a': {
  'v4 gp': v4(**GP), 'v4 s187': v4(**S187),
  'reparam 30/2.2/20 h7 l4 s15': reparam(30, 2.2, 20, 7, 4, 15),
  'reparam 30/2.0/18 h7 l4 s15': reparam(30, 2.0, 18, 7, 4, 15),
  'gp+nudge2 c16-19 t<5': v4x(**GP, nudge=2),
  'gp+nudge3 c17-19 t<7': v4x(**GP, nudge=3, nlo=17, nstop=7),
  'gp+regroup om<12 until 20': v4x(**GP, regrp=12),
  'gp+regroup om<10 until 18': v4x(**GP, regrp=10, restop=18),
  'gp+mid drop c<10 until 15': v4x(**GP, mid=10),
  'gp+tail c<1 until 20': v4x(**GP, tail_lthr=1, tail_stop=20),
  'gp+relapse only (prev>=4)': v4x(**GP, rel_drop=4),
 },
}



def dyn(pmax=30, gamma=2.2, hmin=4, hmax=10, gm=16, lthr=4, lstop=15, stat='mean', tau=0.0, rel=None):
    """dose pmax (1 - c/20)^gamma. Shock for everyone in rounds < hmin; in rounds hmin..hmax-1 the
    shock continues only in groups whose stat (mean / min of valid c) is < gm; otherwise (and after
    hmax) only free riders c < lthr until lstop. tau: taper of the shock, (1 - t/hmax)^tau.
    rel: during the shock only players below the others' mean - rel are dosed."""
    def f(c, v, p, t):
        cn = c[..., -1]; vv = v[..., -1]
        d = (pmax * (1 - cn / 20).clamp(min=0) ** gamma).clamp(max=30) * (cn < 20).float()
        tot = (cn * vv).sum(1, keepdim=True); n = vv.sum(1, keepdim=True).clamp(min=1)
        if stat == 'mean':
            g = tot / n
        else:
            g = th.where(vv > 0, cn, th.full_like(cn, 99.)).min(1, keepdim=True).values
        shock_on = th.ones_like(cn) * float(t < hmin) + (g < gm).float().expand_as(cn) * float(hmin <= t < hmax)
        shock_on = shock_on.clamp(max=1) * (1 - t / hmax) ** tau if t < hmax else th.zeros_like(cn)
        if rel is not None:
            om = (tot - cn * vv) / (n - vv).clamp(min=1)
            shock_on = shock_on * (cn < om - rel).float()
        late = (cn < lthr).float() * float(t < lstop)
        return d * th.maximum(shock_on, late)
    return f


GROUPS['b'] = {
 'reparam base 30/2.2 h7': dyn(hmin=7, hmax=7),
 'taper h10 tau0.5': dyn(hmin=10, hmax=10, tau=0.5),
 'taper h12 tau1': dyn(hmin=12, hmax=12, tau=1.0),
 'dyn mean<16 h4-10': dyn(hmin=4, hmax=10, gm=16),
 'dyn mean<15 h3-12': dyn(hmin=3, hmax=12, gm=15),
 'dyn mean<17 h5-10': dyn(hmin=5, hmax=10, gm=17),
 'dyn min<12 h4-10': dyn(hmin=4, hmax=10, gm=12, stat='min'),
 'dyn min<10 h3-12': dyn(hmin=3, hmax=12, gm=10, stat='min'),
 'rel shock om-0 h7': dyn(hmin=7, hmax=7, rel=0.0),
 'rel shock om-3 h7': dyn(hmin=7, hmax=7, rel=3.0),
}


if __name__ == '__main__':
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
    for k, f in GROUPS[sys.argv[1]].items():
        s = [simulate(f, n=n, seed=sd)[0] for sd in (501, 502, 503)]
        print(f'{k:36s} {np.mean(s):.2f}  ({" ".join(f"{x:.1f}" for x in s)})', flush=True)
