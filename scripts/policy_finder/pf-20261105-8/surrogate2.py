"""Nonparametric surrogate: next contribution drawn from the reference-sim rows (both
managers) that match on (own c bin, others' mean bin, own punishment bin, round bin),
falling back to coarser keys. Only for a rough sense of the ranges."""
import pandas as pd, numpy as np, warnings, sys
warnings.filterwarnings('ignore')
r = pd.read_parquet('plots/simulation/29_EXP1_reference/per_round.parquet')
r['player'] = r.participant_code.str.split('_').str[0].astype(int)
def arr(run, col):
    x = r[r.run.str.contains(run)].pivot_table(index=['episode', 'player'], columns='round_number', values=col)
    return x.values.reshape(-1, 4, 24)
cb = lambda c: np.digitize(c, [1, 4, 8, 11, 14, 17, 20])
ob = lambda o: np.digitize(o, [3, 6, 9, 12, 15, 18])
pb = lambda p: np.digitize(p, [1, 3, 6, 10, 15])
tb = lambda t: np.minimum(t // 6, 3)
table = {}; table2 = {}
for run in ['zero', 'humanlike']:
    C = arr(run, 'contribution'); P = arr(run, 'punishment')
    for t in range(23):
        c = C[:, :, t]; oth = (c.sum(1, keepdims=True) - c) / 3; p = P[:, :, t]
        k = np.stack([cb(c), ob(oth), pb(p), np.full_like(c, tb(t))], -1).reshape(-1, 4).astype(int)
        nx = C[:, :, t+1].reshape(-1)
        for kk, v in zip(map(tuple, k), nx):
            table.setdefault(kk, []).append(v); table2.setdefault(kk[:3], []).append(v)
table = {k: np.array(v) for k, v in table.items()}; table2 = {k: np.array(v) for k, v in table2.items()}
c0 = arr('zero', 'contribution')[:, :, 0]

def simulate(rule, E=1000, seed=0):
    rng = np.random.default_rng(seed)
    C = np.zeros((E, 4, 24)); P = np.zeros((E, 4, 24))
    C[:, :, 0] = c0[rng.integers(0, len(c0), E)]
    score = np.zeros(E)
    for t in range(24):
        c = C[:, :, t]
        p = np.clip(np.floor(rule(C[:, :, :t+1], P[:, :, :t], t)), 0, 30)
        P[:, :, t] = p
        pay = 20 - c - p + (1.6 * c.sum(1, keepdims=True) - p.sum(1, keepdims=True)) / 4
        score += pay.mean(1)
        if t < 23:
            oth = (c.sum(1, keepdims=True) - c) / 3
            k = np.stack([cb(c), ob(oth), pb(p), np.full_like(c, tb(t))], -1).astype(int)
            for e in range(E):
                for i in range(4):
                    kk = tuple(k[e, i]); s = table.get(kk)
                    if s is None or len(s) < 20:
                        s = table2.get(kk[:3])
                    if s is None:
                        s = table2.get((kk[0], kk[1], 0))
                    C[e, i, t+1] = s[rng.integers(len(s))]
    return score.mean(), C.mean((0, 1)), P.mean()
