"""Simulate rule shapes on the surrogate (surrogate.py). Score = sum over rounds of mean payoff."""
import numpy as np, pandas as pd, pickle, sys
sys.path.insert(0, 'scripts/policy_finder/pf-20261105-10')
from surrogate import features
m = pickle.load(open('/tmp/pf10_surrogate.pkl', 'rb'))
r = pd.read_parquet('plots/simulation/29_EXP1_reference/per_round.parquet')
r0 = r[r.round_number == 0].copy()
r0['pid'] = r0.participant_code.str.split('_').str[0].astype(int)
C0 = r0.sort_values(['run', 'episode', 'pid']).contribution.values.reshape(-1, 4).astype(float)
classes = m.classes_.astype(float)

def simulate(policy, G=2000, seed=0, T=24, ret_traj=False):
    rng = np.random.default_rng(seed)
    C = np.zeros((G, 4, T)); P = np.zeros((G, 4, T))
    C[..., 0] = C0[rng.integers(0, len(C0), G)]
    score = np.zeros(G)
    for t in range(T):
        p = np.floor(np.clip(policy(C[..., :t+1], P[..., :t], t), 0, 30))
        P[..., t] = p
        c = C[..., t]
        score += (20 - c - p + (1.6 * c.sum(1, keepdims=True) - p.sum(1, keepdims=True)) / 4).mean(1)
        if t < T - 1:
            pr = m.predict_proba(features(C, P, t))
            u = rng.random((pr.shape[0], 1))
            idx = (pr.cumsum(1) < u).sum(1).clip(max=len(classes) - 1)
            C[..., t+1] = classes[idx].reshape(G, 4)
    if ret_traj:
        return score, C, P
    return score.mean(), score.std() / np.sqrt(G), C.mean((0, 1)), P.mean((0, 1))

zero = lambda c, p, t: np.zeros_like(c[..., -1])
if __name__ == '__main__':
    s, se, cm, pm = simulate(zero)
    print('zero', round(s, 1), round(se, 1), cm.round(1))
    # humanlike proxy: draw p from humanlike's punishments of the same contribution
    h = r[r.run.str.contains('humanlike') & r.contribution_valid]
    byc = {k: v.punishment.values for k, v in h.groupby('contribution')}
    rng = np.random.default_rng(1)
    def hl(c, p, t):
        x = c[..., -1]; out = np.zeros_like(x)
        for k, vals in byc.items():
            mask = x == k
            out[mask] = rng.choice(vals, mask.sum())
        return out
    s, se, cm, pm = simulate(hl)
    print('humanlike proxy', round(s, 1), round(se, 1), cm.round(1), pm.round(2))
