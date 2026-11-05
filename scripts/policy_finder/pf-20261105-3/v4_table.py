"""v4: optimise a free dose table dose[c band, round band] on the surrogate with a
simple evolution strategy (common random numbers), to see the optimum's shape.
python v4_table.py [generations]"""
import sys, json, warnings, numpy as np, torch as th
warnings.filterwarnings('ignore')
sys.path.insert(0, '/opt/algorithmic-institutions/scripts/policy_finder/pf-20261105-3')
from v4_sim import simulate
from v4_levers import shock

th.set_num_threads(2)
CB = [0, 3, 6, 9, 12, 15, 18, 20, 21]          # c bands [0,3) [3,6) ... [18,20) [20]
TB = [0, 1, 2, 3, 5, 8, 11, 14, 17, 24]        # round bands
NC, NT = len(CB) - 1, len(TB) - 1
cband = th.tensor([np.searchsorted(CB, x, side='right') - 1 for x in range(21)])
tband = [int(np.searchsorted(TB, t, side='right') - 1) for t in range(24)]


def make(table):
    tab = th.tensor(table, dtype=th.float).reshape(NC, NT)
    def fn(c, v, p, t):
        return tab[cband[c[..., -1].long()], tband[t]]
    return fn


def init_from_base(thr=21, pmax=30, gamma=2.3, horizon=10):
    tab = np.zeros((NC, NT))
    for i in range(NC):
        cm = (CB[i] + CB[i + 1] - 1) / 2
        d = float(shock(th.tensor(float(cm)), thr, pmax, gamma))
        for j in range(NT):
            tab[i, j] = d if TB[j] < horizon else 0
    return tab.ravel()


def run(gens=40, lam=16, n=1200):
    r = np.random.default_rng(1)
    mean = init_from_base()
    sigma = 3.0
    seed = 5
    f0 = simulate(make(np.clip(mean, 0, 30)), n=n, seed=seed)[0]
    print('init', f0, flush=True)
    best = (f0, mean.copy())
    for g in range(gens):
        eps = r.normal(size=(lam // 2, mean.size)); eps = np.concatenate([eps, -eps])
        fs = np.array([simulate(make(np.clip(mean + sigma * e, 0, 30)), n=n, seed=seed + g)[0] for e in eps])
        rank = np.argsort(np.argsort(-fs))
        w = np.maximum(0, np.log(lam / 2 + 1) - np.log(rank + 1)); w /= w.sum()
        mean = np.clip(mean + sigma * (w[:, None] * eps).sum(0) * 1.0, 0, 30)
        sigma *= 0.97
        fm = simulate(make(mean), n=n, seed=1000)[0]
        print(f'gen {g} pop max {fs.max():.1f} mean-table {fm:.2f} sigma {sigma:.2f}', flush=True)
        if fm > best[0]:
            best = (fm, mean.copy())
    np.save('/tmp/v4_table.npy', best[1])
    fresh = simulate(make(best[1]), n=4000, seed=99)[0]
    base = simulate(make(np.clip(init_from_base(), 0, 30)), n=4000, seed=99)[0]
    print('fresh best table', fresh, 'base table', base)
    print(np.round(best[1].reshape(NC, NT), 1))


if __name__ == '__main__':
    run(int(sys.argv[1]) if len(sys.argv) > 1 else 40)
