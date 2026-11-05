"""v4: screen rule families on the surrogate. Each family is a python function of
(c, v, p, t, **params); a random/local search maximises its surrogate score with
common random numbers. python v4_levers.py <family> [n_iter]"""
import sys, json, warnings, numpy as np, torch as th
warnings.filterwarnings('ignore')
sys.path.insert(0, '/opt/algorithmic-institutions/scripts/policy_finder/pf-20261105-3')
from v4_sim import simulate

th.set_num_threads(2)


def shock(cn, thr, pmax, gamma):
    s = (thr - cn).clamp(min=0)
    return (pmax * (s / thr).clamp(0, 1) ** gamma).clamp(max=30) * (cn < 20).float()


def f_base(c, v, p, t, thr, pmax, gamma, horizon):
    return shock(c[..., -1], thr, pmax, gamma) * float(t < horizon)


def f_forgive(c, v, p, t, thr, pmax, gamma, horizon, rise, keep):
    """players who rose by >= rise since last round get keep x the dose"""
    cn = c[..., -1]
    d = shock(cn, thr, pmax, gamma) * float(t < horizon)
    if t > 0:
        up = ((cn - c[..., -2]) >= rise).float() * v[..., -2]
        d = d * (1 - up * (1 - keep))
    return d


def f_budget(c, v, p, t, thr, pmax, gamma, horizon, budget):
    """no more shock once a player has absorbed `budget` points"""
    d = shock(c[..., -1], thr, pmax, gamma) * float(t < horizon)
    spent = p.sum(-1) if p.shape[-1] else th.zeros_like(d)
    return d * (spent < budget).float()


def f_rel(c, v, p, t, thr, pmax, gamma, horizon, rel):
    """standard = (1-rel)*thr + rel*(max of the others' c)"""
    cn = c[..., -1]
    m = th.where(v[..., -1] > 0, cn, th.zeros_like(cn))
    gmax = m.max(1, keepdim=True).values.expand_as(cn)
    st = (1 - rel) * thr + rel * gmax
    s = (st - cn).clamp(min=0)
    return (pmax * (s / thr).clamp(0, 1) ** gamma).clamp(max=30) * (cn < 20).float() * float(t < horizon)


def f_drop(c, v, p, t, thr, pmax, gamma, horizon, dk, dstop):
    """after the shock: dk points per point a player dropped since last round, until dstop"""
    cn = c[..., -1]
    d = shock(cn, thr, pmax, gamma) * float(t < horizon)
    if t > 0 and t < dstop:
        drop = (c[..., -2] - cn).clamp(min=0) * v[..., -2]
        d = th.maximum(d, dk * drop * (cn < 20).float())
    return d


def f_late(c, v, p, t, thr, pmax, gamma, horizon, lthr, lstop):
    """after the shock, keep full shock doses only for c < lthr until lstop"""
    cn = c[..., -1]
    d = shock(cn, thr, pmax, gamma)
    if t < horizon:
        return d
    if t < lstop:
        return d * (cn < lthr).float()
    return th.zeros_like(cn)


def f_start(c, v, p, t, thr, pmax, gamma, horizon, s0):
    """round 0 dose scaled by s0"""
    d = shock(c[..., -1], thr, pmax, gamma) * float(t < horizon)
    return d * (s0 if t == 0 else 1.0)


def f_typed(c, v, p, t, thr, pmax, gamma, horizon, wmean):
    """shortfall judged on a blend of current c and own running mean"""
    cn = c[..., -1]
    vs = v.sum(-1).clamp(min=1)
    mean = (c * v).sum(-1) / vs
    ce = (1 - wmean) * cn + wmean * mean
    return shock(ce, thr, pmax, gamma) * (cn < 20).float() * float(t < horizon)


FAM = {
    'base': (f_base, dict(thr=(14, 24), pmax=(20, 45), gamma=(0.8, 3.5), horizon=(5, 16))),
    'forgive': (f_forgive, dict(thr=(14, 24), pmax=(20, 45), gamma=(0.8, 3.5), horizon=(5, 16), rise=(1, 12), keep=(0, 1))),
    'budget': (f_budget, dict(thr=(14, 24), pmax=(20, 45), gamma=(0.8, 3.5), horizon=(5, 22), budget=(20, 200))),
    'rel': (f_rel, dict(thr=(14, 24), pmax=(20, 45), gamma=(0.8, 3.5), horizon=(5, 16), rel=(-0.5, 1))),
    'drop': (f_drop, dict(thr=(14, 24), pmax=(20, 45), gamma=(0.8, 3.5), horizon=(5, 16), dk=(0, 4), dstop=(8, 23))),
    'late': (f_late, dict(thr=(14, 24), pmax=(20, 45), gamma=(0.8, 3.5), horizon=(3, 14), lthr=(0, 16), lstop=(8, 23))),
    'start': (f_start, dict(thr=(14, 24), pmax=(20, 45), gamma=(0.8, 3.5), horizon=(5, 16), s0=(0, 1.5))),
    'typed': (f_typed, dict(thr=(14, 24), pmax=(20, 45), gamma=(0.8, 3.5), horizon=(5, 16), wmean=(-0.5, 1))),
}
INT = {'horizon', 'dstop', 'lstop'}


def score(fam, prm, n=1500, seed=11):
    fn, _ = FAM[fam]
    return simulate(lambda c, v, p, t: fn(c, v, p, t, **prm), n=n, seed=seed)[0]


def search(fam, n_iter=120, start=None):
    fn, rng = FAM[fam]
    r = np.random.default_rng(0)
    def draw():
        return {k: (int(r.integers(lo, hi + 1)) if k in INT else float(r.uniform(lo, hi))) for k, (lo, hi) in rng.items()}
    best = start or dict(thr=20, pmax=30, gamma=2.1, horizon=9)
    for k, (lo, hi) in rng.items():
        best.setdefault(k, int((lo + hi) / 2) if k in INT else (lo + hi) / 2)
    bs = score(fam, best)
    hist = []
    for i in range(n_iter):
        if i < n_iter // 3:
            cand = draw()
        else:  # local perturbation
            cand = dict(best)
            for k in r.choice(list(rng), size=2, replace=False):
                lo, hi = rng[k]
                if k in INT:
                    cand[k] = int(np.clip(best[k] + r.integers(-2, 3), lo, hi))
                else:
                    cand[k] = float(np.clip(best[k] + r.normal(0, 0.15 * (hi - lo)), lo, hi))
        s = score(fam, cand)
        hist.append((s, cand))
        if s > bs:
            bs, best = s, cand
            print(f'{i:4d} {bs:.2f} {json.dumps({k: round(x, 2) for k, x in best.items()})}', flush=True)
    # re-score the top 5 on fresh seeds
    hist.sort(key=lambda x: -x[0])
    print('fresh:')
    for s, cand in hist[:5]:
        print(f'  {s:.2f} -> {score(fam, cand, n=4000, seed=99):.2f} {json.dumps({k: round(x, 2) for k, x in cand.items()})}', flush=True)
    return best, bs


def f_late2(c, v, p, t, thr, pmax, gamma, horizon, lthr, ldose, lstop):
    """shock until horizon, then a flat ldose for c < lthr until lstop"""
    cn = c[..., -1]
    if t < horizon:
        return shock(cn, thr, pmax, gamma)
    if t < lstop:
        return ldose * (cn < lthr).float()
    return th.zeros_like(cn)


FAM['late2'] = (f_late2, dict(thr=(16, 24), pmax=(22, 40), gamma=(1.2, 3.5), horizon=(4, 13),
                              lthr=(1, 10), ldose=(5, 30), lstop=(10, 22)))
INT.add('lstop')


if __name__ == '__main__':
    st = json.loads(sys.argv[3]) if len(sys.argv) > 3 else None
    search(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 120, st)
