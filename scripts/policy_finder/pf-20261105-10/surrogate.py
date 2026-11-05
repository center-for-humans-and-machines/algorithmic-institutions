"""Surrogate of the contribution clones, fitted on the reference sims, to compare rule shapes.

Features per player and round t (predicting c at t+1): own c_t, p_t, c_{t-1}, p_{t-1},
running mean of own c and p (a stand-in for the RNN state), others' mean c_t and p_t.
Timed-out inputs keep the clone's placeholder; timed-out targets are dropped; the surrogate sim has no timeouts.
"""
import pandas as pd, numpy as np, pickle, sys
from sklearn.ensemble import HistGradientBoostingClassifier

def features(C, P, t):
    """C: [G,4,t+1] contributions, P: [G,4,t+1] punishments incl. round t. -> [G*4, F]"""
    G, A, _ = C.shape
    c, p = C[..., t], P[..., t]
    cl = C[..., t-1] if t > 0 else C[..., 0]
    pl = P[..., t-1] if t > 0 else np.zeros_like(p)
    first = np.full_like(c, 1.0 if t == 0 else 0.0)
    mc, mp = C[..., :t+1].mean(-1), P[..., :t+1].mean(-1)
    oc = (c.sum(1, keepdims=True) - c) / (A - 1)
    op = (p.sum(1, keepdims=True) - p) / (A - 1)
    return np.stack([c, p, cl, pl, mc, mp, oc, op, first], -1).reshape(G * A, -1)

def build(r):
    r = r.copy()
    r['pid'] = r.participant_code.str.split('_').str[0].astype(int)
    X, Y = [], []
    for run, e in r.groupby('run'):
        e = e.sort_values(['episode', 'pid', 'round_number'])
        G = e.episode.nunique()
        C = e.contribution.values.reshape(G, 4, 24).astype(float)
        P = e.punishment.values.reshape(G, 4, 24).astype(float)
        V = e.contribution_valid.values.reshape(G, 4, 24)
        # the contribution column holds the clone's placeholder input on timeouts,
        # which is what the clone itself saw; drop only targets that timed out
        for t in range(23):
            ok = V[..., t+1].reshape(-1)
            X.append(features(C, P, t)[ok]); Y.append(C[..., t+1].reshape(-1)[ok])
    return np.concatenate(X), np.concatenate(Y).astype(int)

if __name__ == '__main__':
    r = pd.read_parquet('plots/simulation/29_EXP1_reference/per_round.parquet')
    X, Y = build(r)
    print(X.shape)
    m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, max_leaf_nodes=31, random_state=0)
    m.fit(X, Y)
    pickle.dump(m, open('/tmp/pf10_surrogate.pkl', 'wb'))
    print('fit done')
