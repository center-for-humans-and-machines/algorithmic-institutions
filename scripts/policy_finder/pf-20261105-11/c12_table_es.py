"""v3: a flexible table policy, punishment = table[round bin, contribution bin], optimised by a
simple evolution strategy on the mean of both surrogates, started from v2's best. Not a rule
candidate itself: a probe for structure the (T, A, B, S) family cannot express."""
import sys, numpy as np, json; sys.path.insert(0, 'scripts/policy_finder/pf-20261105-11')
from screen import *

TB = [0, 1, 2, 3, 5, 8, 12, 18, 24]          # round bins [TB[i], TB[i+1])
CB = [0, 4, 8, 12, 15, 18, 20, 21]            # contribution bins
nT, nC = len(TB) - 1, len(CB) - 1
tbin = np.zeros(24, int)
for i in range(nT):
    tbin[TB[i]:TB[i + 1]] = i
cbin = np.zeros(21, int)
for j in range(nC):
    cbin[CB[j]:CB[j + 1]] = j
cbin_t = th.tensor(cbin)


def table_rule(tab):
    tab = th.tensor(tab, dtype=th.float32)
    def f(c, v, p, t):
        row = tab[tbin[t]]
        return row[cbin_t[c[..., -1].long()]]
    return f


def v2_table():
    tab = np.zeros((nT, nC))
    for i in range(nT):
        for j in range(nC):
            cm = (CB[j] + CB[j + 1] - 1) / 2
            s = max(0, 14.3 - cm)
            tab[i, j] = (1.7 + 1.56 * s) if (s > 0 and TB[i] <= 8) else 0
    return tab


def sc(tab, seed, B=2000):
    r = ev(table_rule(np.clip(tab, 0, 30)), B=B, seed=seed)
    return (r['mlp']['score'] + r['gru']['score']) / 2


rng = np.random.default_rng(1)
mean = v2_table(); sigma = 1.5
print('v2 table', sc(mean, 5, 4000), flush=True)
for gen in range(int(sys.argv[1])):
    pop = 10
    eps = rng.normal(size=(pop, nT, nC))
    seed = 1000 + gen
    fp = np.array([sc(mean + sigma * e, seed) for e in eps])
    fm = np.array([sc(mean - sigma * e, seed) for e in eps])
    adv = fp - fm
    g = (adv[:, None, None] * eps).mean(0) / (adv.std() + 1e-6)
    mean = np.clip(mean + 0.6 * sigma * g, 0, 30)
    if gen % 5 == 4:
        print(gen, 'score', round(sc(mean, 5, 4000), 1), flush=True)
        print(np.round(mean, 1), flush=True)
        json.dump(mean.tolist(), open('/tmp/pf3/table.json', 'w'))
