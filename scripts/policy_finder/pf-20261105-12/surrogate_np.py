"""Nonparametric Markov surrogate of the AH from both reference runs: c_next sampled
from rows with the same own contribution, others'-mean bin and punishment bin.
Monte Carlo rollouts of candidate rules. No memory, confounded; a guide for ranges."""
import numpy as np, pandas as pd, itertools, sys
rng = np.random.default_rng(0)
r = pd.read_parquet('plots/simulation/29_EXP1_reference/per_round.parquet')
r['player'] = r.participant_code.str.split('_').str[0].astype(int)
r['c'] = np.where(r.contribution_valid, r.contribution, 0)
r = r.sort_values(['run', 'episode', 'player', 'round_number'])
gg = r.groupby(['run', 'episode', 'player'])
r['c_n1'] = gg.c.shift(-1); r['vn'] = gg.contribution_valid.shift(-1)
key = ['run', 'episode', 'round_number']
r['oth'] = (r.groupby(key).c.transform('sum') - r.c) / 3
s = r[r.contribution_valid & (r.vn == True)].dropna(subset=['c_n1'])
OB = np.array([4.5, 8.5, 12.5, 16.5]); PB = np.array([0.5, 1.5, 2.5, 4.5, 7.5, 12.5])
ob = np.searchsorted(OB, s.oth.values); pb = np.searchsorted(PB, s.punishment.values)
cell = s.c.values.astype(int) * 100 + ob * 10 + pb
cn = s.c_n1.values.astype(int)
order = np.argsort(cell); cell_s = cell[order]; cn_s = cn[order]
uc, st, cnt = np.unique(cell_s, return_index=True, return_counts=True)
lut = {u: (a, n) for u, a, n in zip(uc, st, cnt)}
start = r[(r.round_number == 0) & r.contribution_valid].c.values

def nxt(c, oth, p):
    out = np.empty_like(c)
    for i in range(len(c)):
        o = np.searchsorted(OB, oth[i]); q = np.searchsorted(PB, p[i])
        for qq in [q] + [q - 1, q + 1, q - 2]:
            k = int(c[i]) * 100 + o * 10 + min(max(qq, 0), 6)
            if k in lut and lut[k][1] >= 5:
                a, n = lut[k]; out[i] = cn_s[a + rng.integers(n)]; break
        else:
            out[i] = c[i]
    return out

def roll(rule, n=300):
    tot = 0
    for e in range(n):
        c = rng.choice(start, 4).astype(float)
        for t in range(24):
            p = np.floor(np.clip(rule(c, t), 0, 30))
            tot += 20 + 0.6 * c.mean() - 2 * p.mean()
            oth = (c.sum() - c) / 3
            c = nxt(c, oth, p).astype(float)
    return tot / n

print('zero', round(roll(lambda c, t: 0 * c), 1))
res = []
for T, base, k, L in itertools.product([6, 9, 12, 15, 20], [0, 1, 2], [0, 0.25, 0.6], [1, 4]):
    if base == 0 and k == 0: continue
    f = lambda c, t: np.where((c < T) & (t < 24 - L), base + k * (T - c), 0)
    res.append((round(roll(f, 150), 1), T, base, k, L)); print(res[-1], flush=True)
res.sort(key=lambda x: -x[0])
print('best', res[:10])
