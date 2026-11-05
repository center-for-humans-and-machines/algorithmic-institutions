"""Crude linear surrogate of the clones, fit on both reference runs, used only
to place the sweep ranges of the draft rule (confounded; not the real sim).
Per contribution bin: next_c ~ 1 + c + prev_c + oth_c + own p pieces + oth_p,
bootstrap residuals, clip to 0..20, round. Timeouts ignored."""
import numpy as np, pandas as pd, itertools
rng = np.random.default_rng(0)
d = pd.read_parquet('plots/simulation/29_EXP1_reference/per_round.parquet')
d['valid'] = d.contribution_valid.astype(int)
d['c'] = np.where(d.valid == 1, d.contribution, 0)
d = d.sort_values(['run', 'episode', 'participant_code', 'round_number'])
g = d.groupby(['run', 'episode', 'participant_code'])
d['next_c'] = g['c'].shift(-1); d['next_valid'] = g['valid'].shift(-1)
d['prev_c'] = g['c'].shift(1).fillna(d.c)
gr = d.groupby(['run', 'episode', 'round_number'])
d['oth_c'] = (gr['c'].transform('sum') - d.c) / 3
d['oth_p'] = (gr['punishment'].transform('sum') - d.punishment) / 3
x = d[(d.valid == 1) & (d.next_valid == 1)]
edges = [-1, 4, 9, 14, 19, 20]
def feats(c, pc, oc, p, op):
    return np.stack([np.ones_like(c), c, pc, oc, np.minimum(p, 3), np.clip(p - 3, 0, 7),
                     np.clip(p - 10, 0, 20), op], -1)
B, R = [], []
for lo, hi in zip(edges[:-1], edges[1:]):
    xb = x[(x.c > lo) & (x.c <= hi)]
    X = feats(xb.c.values, xb.prev_c.values, xb.oth_c.values, xb.punishment.values, xb.oth_p.values)
    b = np.linalg.lstsq(X, xb.next_c.values, rcond=None)[0]
    B.append(b); R.append(xb.next_c.values - X @ b)
c0 = d[(d.round_number == 0) & (d.valid == 1)].c.values

def play(rule, n=2000):
    c = rng.choice(c0, (n, 4)).astype(float); pc = c.copy(); total = np.zeros(n)
    for t in range(24):
        p = np.floor(np.clip(rule(c, t), 0, 30))
        pay = 20 - c - p + (1.6 * c.sum(1, keepdims=True) - p.sum(1, keepdims=True)) / 4
        total += pay.mean(1)
        oc = (c.sum(1, keepdims=True) - c) / 3; op = (p.sum(1, keepdims=True) - p) / 3
        X = feats(c, pc, oc, p, op); nc = np.zeros_like(c)
        bins = np.digitize(c, edges[1:-1], right=True)
        for i in range(5):
            m = bins == i
            nc[m] = X[m] @ B[i] + rng.choice(R[i], m.sum())
        pc, c = c, np.clip(np.round(nc), 0, 20)
    return total.mean()

def draft(cs, k, pmax, H):
    return lambda c, t: np.minimum(pmax, k * np.clip(cs - c, 0, None)) * min(1, max(0, (23 - t) / H))
print('zero', round(play(lambda c, t: 0 * c), 1))
res = []
for cs, k, pmax, H in itertools.product([8, 12, 16, 20], [0.25, 0.5, 1, 2], [3, 6, 10, 30], [1, 6, 12, 24]):
    res.append((play(draft(cs, k, pmax, H), 1000), cs, k, pmax, H))
res.sort(reverse=True)
for r in res[:15]: print(np.round(r, 2))
print('worst', np.round(res[-3:], 2))

# calibration: a humanlike-ish manager (mean punishment by c bin from the reference)
def humanish(c, t):
    m = np.array([6.0, 4.55, 2.91, 1.05, 0.29])[np.digitize(c, edges[1:-1], right=True)]
    return rng.poisson(m)
print('humanish', round(play(humanish), 1))
