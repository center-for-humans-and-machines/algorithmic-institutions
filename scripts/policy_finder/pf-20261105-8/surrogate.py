"""Crude surrogate of the clones fitted on the reference sims (both managers):
c_i(t+1) = linear(c_i(t), others' mean(t), p_i(t), p_i(t)*[c_i>=15], others' p(t),
                  decayed punishment history h_i(t)) + resampled residual, clipped 0..20.
Used only to size the sweep ranges, not to pick values."""
import pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
r = pd.read_parquet('plots/simulation/29_EXP1_reference/per_round.parquet')
r['player'] = r.participant_code.str.split('_').str[0].astype(int)
def arr(run, col):
    x = r[r.run.str.contains(run)].pivot_table(index=['episode', 'player'], columns='round_number', values=col)
    return x.values.reshape(-1, 4, 24)
DECAY = 0.8
def feats(C, P):
    E = C.shape[0]
    H = np.zeros_like(P)
    for t in range(24):
        H[:, :, t] = (H[:, :, t-1] * DECAY if t else 0) + P[:, :, t]
    rows = []
    for t in range(23):
        c = C[:, :, t]; oth = (c.sum(1, keepdims=True) - c) / 3
        p = P[:, :, t]; op = p.sum(1, keepdims=True) - p
        X = np.stack([np.ones_like(c), c, oth, p, p * (c >= 15), op, H[:, :, t] - p, np.full_like(c, t)], -1)
        rows.append((X.reshape(-1, X.shape[-1]), C[:, :, t+1].reshape(-1)))
    X = np.concatenate([a for a, _ in rows]); y = np.concatenate([b for _, b in rows])
    return X, y
Xs, ys = [], []
for run in ['zero', 'humanlike']:
    C = arr(run, 'contribution'); P = arr(run, 'punishment')
    X, y = feats(C, P); Xs.append(X); ys.append(y)
X = np.concatenate(Xs); y = np.concatenate(ys)
b = np.linalg.lstsq(X, y, rcond=None)[0]
res = y - X @ b
names = ['1', 'c', 'oth', 'p', 'p*high', 'others_p', 'hist', 't']
print(dict(zip(names, np.round(b, 3))), 'resid sd', res.std().round(2))
c0 = arr('zero', 'contribution')[:, :, 0]
np.save('/tmp/pf8_b.npy', b); np.save('/tmp/pf8_res.npy', res); np.save('/tmp/pf8_c0.npy', c0)

def simulate(rule, E=2000, seed=0):
    rng = np.random.default_rng(seed)
    C = np.zeros((E, 4, 24)); P = np.zeros((E, 4, 24)); H = np.zeros((E, 4))
    C[:, :, 0] = c0[rng.integers(0, len(c0), E)]
    score = 0
    for t in range(24):
        c = C[:, :, t]
        p = np.clip(np.floor(rule(C[:, :, :t+1], t)), 0, 30)
        P[:, :, t] = p
        pay = 20 - c - p + (1.6 * c.sum(1, keepdims=True) - p.sum(1, keepdims=True)) / 4
        score += pay.mean(1)
        if t < 23:
            oth = (c.sum(1, keepdims=True) - c) / 3; op = p.sum(1, keepdims=True) - p
            x = np.stack([np.ones_like(c), c, oth, p, p * (c >= 15), op, H, np.full_like(c, t)], -1)
            H = H * DECAY + p
            nxt = x @ b + rng.choice(res, size=c.shape)
            C[:, :, t+1] = np.clip(np.round(nxt), 0, 20)
    return score.mean(), C.mean((0, 1)), P.mean()
if __name__ == '__main__':
    s, cm, pm = simulate(lambda C, t: np.zeros(C.shape[:2]))
    print('zero', round(s, 1), np.round(cm[::4], 1))
