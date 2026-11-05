"""Crude empirical toy model of the clones, fitted on both reference runs, to
compare candidate rules by order of magnitude. Confounded (punishment in the
humanlike run is not random), so only for rough guidance."""
import pandas as pd, numpy as np
import warnings; warnings.filterwarnings('ignore')
rng = np.random.default_rng(0)
sim = pd.read_parquet('plots/simulation/29_EXP1_reference/per_round.parquet')
rows = []
for run, d in sim.groupby('run'):
    d = d.copy(); d['pl'] = d.participant_code.str.split('_').str[0].astype(int)
    d.loc[~d.contribution_valid, 'contribution'] = np.nan
    C = d.pivot_table(index='episode', columns=['pl', 'round_number'], values='contribution', dropna=False)
    P = d.pivot_table(index='episode', columns=['pl', 'round_number'], values='punishment')
    c = C.values.reshape(-1, 4, 24); p = P.values.reshape(-1, 4, 24)
    for t in range(23):
        for i in range(4):
            o = [j for j in range(4) if j != i]
            rows.append(pd.DataFrame(dict(c=c[:, i, t], c1=c[:, i, t + 1], p=p[:, i, t],
                                          co=np.nanmean(c[:, o, t], 1))))
x = pd.concat(rows).dropna()

def feats(c, co, p):
    lp = np.log1p(p); pos = (p > 0).astype(float)
    lo, mid, hi, full = c <= 5, (c > 5) & (c <= 10), (c > 10) & (c <= 15), c > 15
    return np.stack([np.ones_like(c), c, co, c * co / 20, lp * lo, lp * mid, pos * mid,
                     pos * hi, lp * hi, lp * full, pos * full], -1)

X = feats(x.c.values, x.co.values, x.p.values)
b = np.linalg.lstsq(X, x.c1.values, rcond=None)[0]
res = x.c1.values - X @ b
print('coef', b.round(3), 'resid sd', res.std().round(2))

c0 = sim[(sim.round_number == 0) & sim.contribution_valid].contribution.values

def play(rule, n=4000):
    c = rng.choice(c0, (n, 4)).astype(float); hist = []; total = 0
    for t in range(24):
        pun = np.clip(np.floor(rule(c, t)), 0, 30)
        total += (80 + 0.6 * c.sum(1) - 2 * pun.sum(1)).mean() / 4
        co = (c.sum(1, keepdims=True) - c) / 3
        nxt = feats(c, co, pun) @ b + rng.choice(res, c.shape)
        c = np.clip(np.round(nxt), 0, 20)
    return total

print('zero', round(play(lambda c, t: 0 * c), 1))
for T in ():
    for a in (0, 2, 4):
        for k in (0, 0.5, 1.0):
            for tend in (16, 20, 24):
                r = lambda c, t: (c < T) * (a + k * (T - c)) * (t < tend)
                print(T, a, k, tend, round(play(r), 1))

def traj(rule, n=4000):
    c = rng.choice(c0, (n, 4)).astype(float); out = []
    for t in range(24):
        pun = np.clip(np.floor(rule(c, t)), 0, 30); out.append((c.mean(), pun.mean()))
        co = (c.sum(1, keepdims=True) - c) / 3
        c = np.clip(np.round(feats(c, co, pun) @ b + rng.choice(res, c.shape)), 0, 20)
    return np.array(out)
hl = lambda c, t: np.select([c <= 5, c <= 10, c <= 15, c < 20], [8.2, 3.0, 2.2, 1.5], 0.3)
print('zero traj', traj(lambda c, t: 0 * c)[::4].round(2).tolist())
print('humanlike-ish traj', traj(hl)[::4].round(2).tolist(), 'score', round(play(hl), 1))

def play_crn(rule, n=20000):
    global rng
    rng = np.random.default_rng(1); return play(rule, n)
print('zero crn', round(play_crn(lambda c, t: 0 * c), 2))
res_ = []
for T in (4, 6, 8, 10, 12, 14):
    for a in (0, 1, 2, 3):
        for k in (0, 0.25, 0.5, 1.0):
            for tend in (8, 12, 16, 20):
                r = lambda c, t: (c < T) * (a + k * (T - c)) * (t < tend)
                res_.append((T, a, k, tend, round(play_crn(r, 8000), 2)))
res_.sort(key=lambda r: -r[-1]); print(res_[:20])
