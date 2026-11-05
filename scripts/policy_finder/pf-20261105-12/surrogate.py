"""Crude surrogate of the AH: fit E[c_next | c, others, p] on both reference runs with
nonlinear punishment features, then roll out mean dynamics (4 players, heterogeneous start)
under candidate rules to get a feel for parameter ranges. Confounded; a guide only."""
import numpy as np, pandas as pd, itertools
r = pd.read_parquet('plots/simulation/29_EXP1_reference/per_round.parquet')
r['player'] = r.participant_code.str.split('_').str[0].astype(int)
r['c'] = np.where(r.contribution_valid, r.contribution, 0)
r = r.sort_values(['run', 'episode', 'player', 'round_number'])
gg = r.groupby(['run', 'episode', 'player'])
r['c_n1'] = gg.c.shift(-1); r['vn'] = gg.contribution_valid.shift(-1)
key = ['run', 'episode', 'round_number']
r['nv'] = r.groupby(key).contribution_valid.transform('sum')
r['oth'] = (r.groupby(key).c.transform('sum') - r.c) / (r.nv - r.contribution_valid).clip(lower=1)
r['pc'] = gg.punishment.cumsum() - r.punishment  # punishment received before
s = r[r.contribution_valid & (r.vn == True)].dropna(subset=['c_n1'])

def feats(c, oth, p):
    p = np.asarray(p, float); c = np.asarray(c, float)
    hi = c / 20.0
    return np.column_stack([np.ones_like(c), c, oth, (p > 0), np.sqrt(p), (p > 0) * hi, np.sqrt(p) * hi, c ** 2 / 20])

X = feats(s.c.values, s.oth.values, s.punishment.values)
b = np.linalg.lstsq(X, s.c_n1.values, rcond=None)[0]
print('coef', b.round(3))
start = np.array([[s.c[s.round_number == 0].quantile(q) for q in (0.15, 0.4, 0.65, 0.9)]])
start = r[r.round_number == 0].c.quantile([0.125, 0.375, 0.625, 0.875]).values

def roll(rule):
    c = start.copy(); tot = 0
    for t in range(24):
        p = np.floor(np.clip(rule(c, t), 0, 30))
        tot += 20 + 0.6 * c.mean() - 2 * p.mean()
        oth = (c.sum() - c) / 3
        c = np.clip(feats(c, oth, p) @ b, 0, 20)
    return tot, c.round(1)

print('zero', roll(lambda c, t: 0 * c))
res = []
for T, base, k, L in itertools.product([6, 8, 10, 12, 14, 16, 20], [0, 1, 2, 3], [0, 0.1, 0.25, 0.5, 1], [0, 2, 4, 8]):
    f = lambda c, t: np.where((c < T) & (t < 24 - L), base + k * (T - c), 0)
    sc, cend = roll(f)
    res.append((sc, T, base, k, L, cend))
res.sort(key=lambda x: -x[0])
for x in res[:15]: print(x)
