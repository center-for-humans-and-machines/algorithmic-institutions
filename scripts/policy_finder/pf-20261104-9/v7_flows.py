"""v7: flows at each decision in v6 final gp vs zero_vs_ah: leavers from my group, joiners from ah's,
per episode, and leave rates of my members by their c relative to others (laggard / not)."""
import pyarrow.parquet as pq, numpy as np
cols = ['episode', 'participant_code', 'round_number', 'punishment', 'contribution', 'group_id', 'run']
def arr(path, key):
    t = pq.read_table(path, columns=cols).to_pandas(); t = t[t.run.str.contains(key)]
    ne = t.episode.max() + 1; pl = t.participant_code.str.split('_').str[0].astype(int)
    C = np.zeros((ne, 8, 24)); G = np.zeros_like(C); P = np.zeros_like(C)
    C[t.episode, pl, t.round_number] = t.contribution; G[t.episode, pl, t.round_number] = t.group_id; P[t.episode, pl, t.round_number] = t.punishment
    return C, G, P
for name, path, key in [('gp', '/opt/algorithmic-institutions/plots/simulation/policy_finder/pf-20261104-9-v6_sweep/final/per_round.parquet', '_gp_'),
                        ('s202', '/opt/algorithmic-institutions/plots/simulation/policy_finder/pf-20261104-9-v6_sweep/final/per_round.parquet', 's202'),
                        ('zero', '/opt/algorithmic-institutions/plots/simulation/25_LEVIN_run1_ah_zero_pairings_batched/per_round.parquet', 'zero_vs_ah')]:
    C, G, P = arr(path, key)
    print(name)
    for r in [3, 7, 11, 15, 19]:
        mi = G[..., r] == 0; ot = G[..., r] == 1
        out = (mi & (G[..., r + 1] == 1)).sum(-1).mean(); inn = (ot & (G[..., r + 1] == 0)).sum(-1).mean()
        # my members' mean c over the phase relative to others
        cph = C[..., r - 3:r + 1].mean(-1); n = mi.sum(-1, keepdims=True)
        oc = ((cph * mi).sum(-1, keepdims=True) - cph) / np.maximum(n - 1, 1)
        lag = mi & (cph < 0.8 * oc); hi = mi & (cph >= 0.8 * oc)
        punph = P[..., r - 3:r + 1].sum(-1) > 0
        lv = lambda s: (s & (G[..., r + 1] == 1)).sum() / max(s.sum(), 1)
        # joiners' c (phase before) and after
        j = ot & (G[..., r + 1] == 0)
        cj = C[..., r - 3:r + 1].mean(-1)[j].mean() if j.sum() else np.nan
        lvm = (mi & (G[..., r + 1] == 1)); cl = C[..., r - 3:r + 1].mean(-1)[lvm].mean()
        print(f' after {r}: out {out:.2f} (c {cl:.1f}) in {inn:.2f} (c {cj:.1f}) | leave laggard {lv(lag):.3f} n {lag.sum(-1).mean():.2f}, others {lv(hi):.3f}; punished {lv(mi & punph):.3f} unpunished {lv(mi & ~punph):.3f}; ah leave rate {inn / max(ot.sum(-1).mean(), 1e-9):.3f}')
