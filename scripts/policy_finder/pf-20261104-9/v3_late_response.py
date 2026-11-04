"""Late-game (rounds 10-20) response to punishment, from v3's final run (all 6 points).
For each player-round: own c, others' mean c in the group, punishment; then c 1 and 3 rounds later.
Bins by relative contribution and punishment; separately for my group (0) and ah's (1)."""
import pandas as pd, numpy as np
fn = '/opt/algorithmic-institutions/plots/simulation/policy_finder/pf-20261104-9-v3_sweep/final/per_round.parquet'
d = pd.read_parquet(fn, columns=['run', 'episode', 'participant_code', 'round_number', 'group_id', 'contribution', 'punishment', 'contribution_valid'])
d['pl'] = d.participant_code.str.split('_').str[0].astype(int)
d['run'] = d.run.astype('category').cat.codes
d = d.sort_values(['run', 'episode', 'pl', 'round_number'])
n = len(d) // 24
C = d.contribution.values.reshape(n, 24).astype(float); P = d.punishment.values.reshape(n, 24).astype(float)
G = d.group_id.values.reshape(n, 24); V = d.contribution_valid.values.reshape(n, 24)
# others' mean per round
C8 = C.reshape(-1, 8, 24); G8 = G.reshape(-1, 8, 24); V8 = V.reshape(-1, 8, 24)
oth = np.zeros_like(C8)
for g in (0, 1):
    m = (G8 == g) * V8
    s = (C8 * m).sum(1, keepdims=True); k = m.sum(1, keepdims=True)
    o = (s - C8 * m) / np.maximum(k - m, 1)
    oth = np.where(G8 == g, o, oth)
oth = oth.reshape(n, 24)
rows = []
for t in range(10, 21):
    for lag in (1, 3):
        if t + lag > 23: continue
        stay = (G[:, t] == G[:, t + lag]) & (V[:, t] == 1) & (V[:, t + lag] == 1)
        rows.append(pd.DataFrame(dict(t=t, lag=lag, g=G[stay, t], c=C[stay, t], o=oth[stay, t], p=P[stay, t], dc=C[stay, t + lag] - C[stay, t])))
r = pd.concat(rows)
r['rel'] = pd.cut(r.c / r.o.clip(lower=1), [0, 0.5, 0.8, 1.0, 1.2, 10], right=False)
r['pb'] = pd.cut(r.p, [-1, 0, 2, 5, 10, 31])
pd.set_option('display.width', 200)
for g in (0, 1):
    print('group', g)
    print(r[r.g == g].groupby(['lag', 'rel', 'pb'], observed=True).dc.agg(['mean', 'size']).round(2).unstack('pb').to_string())
